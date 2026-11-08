from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

from train_vae import QuarryCVAE


# ============================= CONFIG =============================


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent

    raise FileNotFoundError(
        "Не найден pyproject.toml. Укажи PROJECT_ROOT вручную."
    )


PROJECT_ROOT = find_project_root()

SOURCE_DEM = PROJECT_ROOT / "data/derived/dem_sever_quarry.tif"
SOURCE_DEM = PROJECT_ROOT / "outputs/real_dem_contour_fill_20261107_200511_616752/dem_filled.tif"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "cvae_20261107_201632_371010"
    / "best_model.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / f"full_dem_reconstruction_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
)

SOURCE_BAND = 1

# Полное окно модели.
PATCH_SIZE = 64

# Центральная область, которую скрываем и предсказываем.
RECONSTRUCTION_SIZE = 16

# Шаг центральной области.
# 8 — перекрытие 50% для блока 16 × 16.
STRIDE = 8

INFERENCE_BATCH_SIZE = 32

DEVICE = "auto"
SEED = 42
STOCHASTIC = False

# Доля известных пикселей во всём окне 64 × 64
# после скрытия центрального блока.
MIN_KNOWN_FRACTION = 0.25

# Веса объединения предсказаний центральных блоков.
BLEND_SIGMA_FRACTION = 0.30
MIN_BLEND_WEIGHT = 0.05

# 0 — одинаковый общий вес окон независимо от доли контекста.
CONTEXT_WEIGHT_POWER = 1.0

EXTRA_NODATA_VALUE = None

PRINT_EVERY_WINDOWS = 1000

SAVE_MULTIBAND_GEOTIFF = True
SAVE_PREDICTION_COUNT = True
SAVE_WINDOW_SPREAD = True
SAVE_DIAGNOSTIC_MASKS = True

GEOTIFF_BLOCK_SIZE = 256
OVERVIEW_MAX_SIDE = 1600

DETAIL_SIZE = 256

# None — автоматически выбрать область с известными высотами.
# Либо явно задать верхний левый угол фрагмента: (row, col).
DETAIL_TOP_LEFT = None

# ==================================================================


def write_json(path, data):
    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def validate_settings():
    if not SOURCE_DEM.is_file():
        raise FileNotFoundError(SOURCE_DEM)

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(CHECKPOINT_PATH)

    if PATCH_SIZE != 64:
        raise ValueError("Ожидается PATCH_SIZE=64.")

    if (
        not isinstance(RECONSTRUCTION_SIZE, int)
        or not 1 <= RECONSTRUCTION_SIZE < PATCH_SIZE
    ):
        raise ValueError(
            "RECONSTRUCTION_SIZE должен быть целым от 1 до 63."
        )

    if (PATCH_SIZE - RECONSTRUCTION_SIZE) % 2:
        raise ValueError(
            "PATCH_SIZE - RECONSTRUCTION_SIZE должно быть чётным."
        )

    if (
        not isinstance(STRIDE, int)
        or not 1 <= STRIDE <= RECONSTRUCTION_SIZE
    ):
        raise ValueError(
            "STRIDE должен быть целым от 1 до RECONSTRUCTION_SIZE."
        )

    if INFERENCE_BATCH_SIZE < 1:
        raise ValueError("INFERENCE_BATCH_SIZE должен быть >= 1.")

    if not 0 < MIN_KNOWN_FRACTION < 1:
        raise ValueError("Некорректный MIN_KNOWN_FRACTION.")

    if BLEND_SIGMA_FRACTION <= 0:
        raise ValueError("BLEND_SIGMA_FRACTION должен быть > 0.")

    if not 0 < MIN_BLEND_WEIGHT <= 1:
        raise ValueError("Некорректный MIN_BLEND_WEIGHT.")

    if CONTEXT_WEIGHT_POWER < 0:
        raise ValueError("CONTEXT_WEIGHT_POWER должен быть >= 0.")

    if not isinstance(SOURCE_BAND, int) or SOURCE_BAND < 1:
        raise ValueError("SOURCE_BAND должен быть целым >= 1.")

    if (
        not isinstance(GEOTIFF_BLOCK_SIZE, int)
        or GEOTIFF_BLOCK_SIZE < 16
        or GEOTIFF_BLOCK_SIZE % 16
    ):
        raise ValueError("GEOTIFF_BLOCK_SIZE должен быть кратен 16.")

    if min(PRINT_EVERY_WINDOWS, OVERVIEW_MAX_SIDE, DETAIL_SIZE) < 1:
        raise ValueError("Некорректные параметры вывода.")


def choose_device():
    if DEVICE == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if DEVICE not in {"cpu", "mps", "cuda"}:
        raise ValueError("DEVICE: auto/cpu/mps/cuda.")

    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна.")

    if DEVICE == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS недоступен.")

    return torch.device(DEVICE)


def load_model(device):
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    preprocessing = checkpoint["preprocessing"]

    if preprocessing.get("normalization") != (
        "local min-max from known pixels"
    ):
        raise ValueError(
            "Этому скрипту нужен checkpoint с нормализацией "
            "'local min-max from known pixels'."
        )

    if preprocessing.get("known_mask_convention") != "1=known, 0=missing":
        raise ValueError(
            "Checkpoint использует другое соглашение о маске "
            "или не содержит его описания."
        )

    if int(preprocessing["patch_size"]) != PATCH_SIZE:
        raise ValueError("Размер окна не совпадает с checkpoint.")

    minimum_scale = float(preprocessing["min_height_scale"])

    if not np.isfinite(minimum_scale) or minimum_scale <= 0:
        raise ValueError("Некорректный min_height_scale.")

    model = QuarryCVAE(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def load_dem():
    with rasterio.open(SOURCE_DEM) as src:
        if src.count < SOURCE_BAND:
            raise ValueError(f"В DEM нет канала {SOURCE_BAND}.")

        band = src.read(
            SOURCE_BAND,
            masked=True,
            out_dtype="float64",
        )

        raw = np.asarray(band.data)

        valid = (
            ~np.ma.getmaskarray(band)
            & np.isfinite(raw)
        )

        if EXTRA_NODATA_VALUE is not None:
            valid &= raw != EXTRA_NODATA_VALUE

        scale = float(src.scales[SOURCE_BAND - 1])
        offset = float(src.offsets[SOURCE_BAND - 1])

        if not np.isfinite([scale, offset]).all():
            raise ValueError("Некорректные scale/offset.")

        with np.errstate(over="ignore", invalid="ignore"):
            original = raw * scale + offset

        valid &= np.isfinite(original)
        original[~valid] = np.nan

        metadata = {
            "height": src.height,
            "width": src.width,
            "transform": src.transform,
            "crs": src.crs,
            "resolution": list(src.res),
            "source_scale": scale,
            "source_offset": offset,
        }

    if not valid.any():
        raise ValueError("В DEM нет известных высот.")

    return original, valid, metadata


def block_starts(length):
    if length <= RECONSTRUCTION_SIZE:
        return [0]

    starts = list(
        range(
            0,
            length - RECONSTRUCTION_SIZE + 1,
            STRIDE,
        )
    )

    last = length - RECONSTRUCTION_SIZE

    if starts[-1] != last:
        starts.append(last)

    return starts


def blending_weights():
    coordinates = (
        np.arange(RECONSTRUCTION_SIZE, dtype=np.float64)
        - (RECONSTRUCTION_SIZE - 1) / 2
    )

    sigma = RECONSTRUCTION_SIZE * BLEND_SIGMA_FRACTION
    axis = np.exp(-0.5 * (coordinates / sigma) ** 2)

    weights = np.outer(axis, axis)
    weights /= weights.max()

    return np.maximum(weights, MIN_BLEND_WEIGHT)


@torch.inference_mode()
def reconstruct_full_surface(
    model,
    original,
    valid,
    preprocessing,
    device,
):
    model.eval()
    started = time.perf_counter()

    height, width = original.shape

    halo = (PATCH_SIZE - RECONSTRUCTION_SIZE) // 2

    # Padding содержит NoData, а не искусственно отражённые высоты.
    # Это позволяет обрабатывать края исходного растра
    # без выдумывания внешнего опорного контекста.
    bottom_extra = max(0, RECONSTRUCTION_SIZE - height)
    right_extra = max(0, RECONSTRUCTION_SIZE - width)

    padding = (
        (halo, halo + bottom_extra),
        (halo, halo + right_extra),
    )

    padded_dem = np.pad(
        original,
        padding,
        mode="constant",
        constant_values=np.nan,
    )

    padded_valid = np.pad(
        valid,
        padding,
        mode="constant",
        constant_values=False,
    )

    # Итог формируется только по предсказаниям модели.
    mean = np.zeros(original.shape, dtype=np.float64)
    m2 = np.zeros(original.shape, dtype=np.float64)
    weight_sum = np.zeros(original.shape, dtype=np.float64)
    counts = np.zeros(original.shape, dtype=np.uint32)

    blend = blending_weights()
    minimum_scale = float(preprocessing["min_height_scale"])

    center_region = np.s_[
        halo:halo + RECONSTRUCTION_SIZE,
        halo:halo + RECONSTRUCTION_SIZE,
    ]

    rows = block_starts(height)
    cols = block_starts(width)

    batch_inputs = []
    batch_info = []

    processed = 0
    skipped_context = 0
    skipped_empty = 0

    def flush():
        nonlocal processed

        if not batch_inputs:
            return

        tensor = torch.from_numpy(
            np.stack(batch_inputs)
        ).to(device)

        predictions = model(
            tensor,
            stochastic=STOCHASTIC,
        ).cpu().numpy()[:, 0]

        if not np.isfinite(predictions).all():
            raise RuntimeError("Модель вернула NaN/Inf.")

        for prediction, info in zip(predictions, batch_info):
            row, col, core_height, core_width, offset, scale, context_weight = info

            output_region = np.s_[
                row:row + core_height,
                col:col + core_width,
            ]

            # Предсказываем только ячейки исходного покрытия.
            local_target = valid[output_region]

            predicted_core = prediction[
                halo:halo + core_height,
                halo:halo + core_width,
            ]

            values = (
                predicted_core[local_target].astype(np.float64)
                * scale
                + offset
            )

            weights = (
                blend[:core_height, :core_width][local_target]
                * context_weight
            )

            local_mean = mean[output_region]
            local_m2 = m2[output_region]
            local_weight_sum = weight_sum[output_region]
            local_counts = counts[output_region]

            old_weights = local_weight_sum[local_target]
            new_weights = old_weights + weights

            old_mean = local_mean[local_target]
            delta = values - old_mean

            updated_mean = (
                old_mean + delta * weights / new_weights
            )

            local_m2[local_target] += (
                weights * delta * (values - updated_mean)
            )
            local_mean[local_target] = updated_mean
            local_weight_sum[local_target] = new_weights
            local_counts[local_target] += 1

        previous = processed
        processed += len(batch_inputs)

        if (
            processed // PRINT_EVERY_WINDOWS
            > previous // PRINT_EVERY_WINDOWS
        ):
            elapsed = time.perf_counter() - started
            print(
                f"Обработано окон: {processed:,} | "
                f"{elapsed:.1f} с"
            )

        batch_inputs.clear()
        batch_info.clear()

    for row in rows:
        for col in cols:
            core_height = min(
                RECONSTRUCTION_SIZE,
                height - row,
            )
            core_width = min(
                RECONSTRUCTION_SIZE,
                width - col,
            )

            output_region = np.s_[
                row:row + core_height,
                col:col + core_width,
            ]

            if not valid[output_region].any():
                skipped_empty += 1
                continue

            # За счёт padding центральный блок окна соответствует
            # output_region в исходной DEM.
            patch = padded_dem[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE,
            ]

            known = padded_valid[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE,
            ].copy()

            # Скрываем весь предсказываемый центральный блок.
            known[center_region] = False

            known_fraction = float(known.mean())

            if (
                known_fraction < MIN_KNOWN_FRACTION
                or not known.any()
            ):
                skipped_context += 1
                continue

            visible = patch[known]

            # Параметры нормализации не используют скрытый блок.
            offset = float(visible.min())
            scale = max(
                float(visible.max()) - offset,
                minimum_scale,
            )

            normalized = np.zeros(
                (PATCH_SIZE, PATCH_SIZE),
                dtype=np.float32,
            )

            normalized[known] = (
                (visible - offset) / scale
            ).astype(np.float32)

            if not np.isfinite(normalized).all():
                raise RuntimeError("Ошибка нормализации окна.")

            condition = np.stack([
                normalized,
                known.astype(np.float32),
            ])

            batch_inputs.append(condition)

            batch_info.append((
                row,
                col,
                core_height,
                core_width,
                offset,
                scale,
                known_fraction ** CONTEXT_WEIGHT_POWER,
            ))

            if len(batch_inputs) >= INFERENCE_BATCH_SIZE:
                flush()

    flush()

    predicted = valid & (weight_sum > 0)

    # Не подставляем исходные высоты в необработанные места.
    rebuilt = np.full(
        original.shape,
        np.nan,
        dtype=np.float64,
    )
    rebuilt[predicted] = mean[predicted]

    spread = np.full(
        original.shape,
        np.nan,
        dtype=np.float64,
    )
    spread[predicted] = np.sqrt(
        np.maximum(
            m2[predicted] / weight_sum[predicted],
            0,
        )
    )

    if not np.isfinite(rebuilt[predicted]).all():
        raise RuntimeError("NaN/Inf в восстановленных значениях.")

    if np.isfinite(rebuilt[~valid]).any():
        raise RuntimeError("Заполнены исходные NoData вне покрытия.")

    statistics = {
        "processed_windows": processed,
        "skipped_empty_windows": skipped_empty,
        "skipped_insufficient_context": skipped_context,
        "inference_seconds": float(time.perf_counter() - started),
        "source_values_copied_into_output": False,
        "predictions_reused_as_context": False,
    }

    return rebuilt, predicted, spread, counts, statistics


def geotiff_profile(metadata, count=1, integer=False):
    return {
        "driver": "GTiff",
        "height": metadata["height"],
        "width": metadata["width"],
        "count": count,
        "dtype": "uint32" if integer else "float64",
        "crs": metadata["crs"],
        "transform": metadata["transform"],
        "nodata": None if integer else np.nan,
        "compress": "deflate",
        "predictor": 2 if integer else 3,
        "tiled": True,
        "blockxsize": GEOTIFF_BLOCK_SIZE,
        "blockysize": GEOTIFF_BLOCK_SIZE,
        "BIGTIFF": "IF_SAFER",
    }


def write_geotiff(path, array, metadata, integer=False, name=None):
    with rasterio.open(
        path,
        "w",
        **geotiff_profile(metadata, integer=integer),
    ) as dst:
        for _, window in dst.block_windows(1):
            rows, cols = window.toslices()

            dst.write(
                np.asarray(
                    array[rows, cols],
                    dtype=np.uint32 if integer else np.float64,
                ),
                1,
                window=window,
            )

        if name:
            dst.set_band_description(1, name)


def save_results(
    original,
    rebuilt,
    predicted,
    spread,
    counts,
    metadata,
):
    difference = np.full(
        original.shape,
        np.nan,
        dtype=np.float64,
    )

    difference[predicted] = (
        rebuilt[predicted] - original[predicted]
    )

    layers = [
        ("dem_reconstructed.tif", "model_reconstructed_surface", rebuilt),
        ("dem_original.tif", "original_surface", original),
        (
            "reconstruction_difference.tif",
            "reconstructed_minus_original",
            difference,
        ),
    ]

    if SAVE_WINDOW_SPREAD:
        layers.append((
            "window_prediction_spread.tif",
            "window_prediction_spread_not_true_error",
            spread,
        ))

    for filename, name, array in layers:
        write_geotiff(
            OUTPUT_DIR / filename,
            array,
            metadata,
            name=name,
        )

    if SAVE_MULTIBAND_GEOTIFF:
        with rasterio.open(
            OUTPUT_DIR / "dem_comparison.tif",
            "w",
            **geotiff_profile(metadata, count=len(layers)),
        ) as dst:
            for band, (_, name, _) in enumerate(layers, start=1):
                dst.set_band_description(band, name)

            for _, window in dst.block_windows(1):
                rows, cols = window.toslices()

                for band, (_, _, array) in enumerate(layers, start=1):
                    dst.write(
                        np.asarray(array[rows, cols], dtype=np.float64),
                        band,
                        window=window,
                    )

    if SAVE_PREDICTION_COUNT:
        write_geotiff(
            OUTPUT_DIR / "prediction_count.tif",
            counts,
            metadata,
            integer=True,
            name="number_of_contributing_predictions",
        )

    if SAVE_DIAGNOSTIC_MASKS:
        original_valid = np.isfinite(original)

        write_geotiff(
            OUTPUT_DIR / "reconstructed_pixels.tif",
            predicted,
            metadata,
            integer=True,
            name="successfully_reconstructed_pixels",
        )

        write_geotiff(
            OUTPUT_DIR / "unreconstructed_pixels.tif",
            original_valid & ~predicted,
            metadata,
            integer=True,
            name="source_pixels_without_model_prediction",
        )

    write_json(
        OUTPUT_DIR / "dem_layers.json",
        {
            "multiband_file": (
                "dem_comparison.tif"
                if SAVE_MULTIBAND_GEOTIFF else None
            ),
            "layers": [
                {"band": band, "file": filename, "name": name}
                for band, (filename, name, _) in enumerate(layers, start=1)
            ],
            "difference_definition": "reconstructed minus original",
            "difference_is_true_accuracy": False,
        },
    )

    return difference


def save_figures(original, rebuilt, difference, predicted):
    step = max(
        1,
        int(np.ceil(max(original.shape) / OVERVIEW_MAX_SIDE)),
    )

    finite = original[::step, ::step]
    finite = finite[np.isfinite(finite)]

    if not finite.size:
        finite = original[np.isfinite(original)]

    low = float(finite.min())
    high = max(float(finite.max()), low + 1e-6)

    finite_difference = difference[np.isfinite(difference)]

    limit = (
        max(
            float(np.percentile(np.abs(finite_difference), 98)),
            0.01,
        )
        if finite_difference.size else 0.01
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for index, (axis, array, title) in enumerate(zip(
        axes,
        [original, rebuilt, difference],
        ["Original DEM", "Fully reconstructed DEM", "Difference"],
    )):
        is_difference = index == 2

        image = axis.imshow(
            array[::step, ::step],
            cmap="RdBu_r" if is_difference else "terrain",
            vmin=-limit if is_difference else low,
            vmax=limit if is_difference else high,
        )
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(image, ax=axis, shrink=0.7)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "overview.png", dpi=170)
    plt.close(fig)

    if DETAIL_TOP_LEFT is None:
        if predicted.any():
            center_index = np.flatnonzero(predicted.ravel())[
                int(predicted.sum()) // 2
            ]
        else:
            center_index = np.flatnonzero(np.isfinite(original).ravel())[0]

        center_row, center_col = np.unravel_index(
            center_index,
            original.shape,
        )

        row = max(0, int(center_row) - DETAIL_SIZE // 2)
        col = max(0, int(center_col) - DETAIL_SIZE // 2)

    else:
        row, col = map(int, DETAIL_TOP_LEFT)

        if not (
            0 <= row < original.shape[0]
            and 0 <= col < original.shape[1]
        ):
            raise ValueError("DETAIL_TOP_LEFT находится вне растра.")

    region = np.s_[
        row:min(row + DETAIL_SIZE, original.shape[0]),
        col:min(col + DETAIL_SIZE, original.shape[1]),
    ]

    source = original[region]
    result = rebuilt[region]
    local_difference = difference[region]

    values = source[np.isfinite(source)]

    local_low = float(values.min()) if values.size else low
    local_high = (
        max(float(values.max()), local_low + 1e-6)
        if values.size else high
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for index, (axis, array, title) in enumerate(zip(
        axes,
        [source, result, local_difference],
        ["Original detail", "Reconstructed detail", "Difference"],
    )):
        is_difference = index == 2

        image = axis.imshow(
            array,
            cmap="RdBu_r" if is_difference else "terrain",
            vmin=-limit if is_difference else local_low,
            vmax=limit if is_difference else local_high,
        )
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(image, ax=axis, shrink=0.75)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "detail.png", dpi=180)
    plt.close(fig)


def main():
    validate_settings()

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    device = choose_device()
    model, checkpoint = load_model(device)
    original, valid, metadata = load_dem()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in globals().items()
        if key.isupper()
        and isinstance(
            value,
            (str, int, float, bool, tuple, dict, Path, type(None)),
        )
    }
    config["selected_device"] = str(device)
    config["torch_version"] = str(torch.__version__)

    write_json(OUTPUT_DIR / "config.json", config)

    print(f"Device: {device}")
    print(f"DEM: {original.shape}")
    print(f"Resolution: {metadata['resolution']}")
    print(f"Context window: {PATCH_SIZE} × {PATCH_SIZE}")
    print(
        f"Reconstructed block: "
        f"{RECONSTRUCTION_SIZE} × {RECONSTRUCTION_SIZE}"
    )
    print(f"Stride: {STRIDE}")
    print(f"Source valid pixels: {int(valid.sum()):,}")

    if metadata["crs"] is None:
        print("CRS отсутствует; координаты не преобразуются.")

    rebuilt, predicted, spread, counts, statistics = (
        reconstruct_full_surface(
            model,
            original,
            valid,
            checkpoint["preprocessing"],
            device,
        )
    )

    difference = save_results(
        original,
        rebuilt,
        predicted,
        spread,
        counts,
        metadata,
    )

    values = difference[predicted]

    comparison = {
        "rmse_difference": (
            float(np.sqrt(np.mean(values ** 2)))
            if values.size else None
        ),
        "mae_difference": (
            float(np.mean(np.abs(values)))
            if values.size else None
        ),
        "mean_difference": (
            float(np.mean(values))
            if values.size else None
        ),
        "max_absolute_difference": (
            float(np.max(np.abs(values)))
            if values.size else None
        ),
    }

    valid_count = int(valid.sum())
    predicted_count = int(predicted.sum())

    report = {
        "source": str(SOURCE_DEM),
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "device": str(device),
        "shape": list(original.shape),
        "resolution": metadata["resolution"],
        "crs": (
            metadata["crs"].to_string()
            if metadata["crs"] is not None else None
        ),
        "context_window": PATCH_SIZE,
        "reconstruction_block": RECONSTRUCTION_SIZE,
        "stride": STRIDE,
        "source_valid_pixels": valid_count,
        "reconstructed_pixels": predicted_count,
        "unreconstructed_pixels": valid_count - predicted_count,
        "coverage": predicted_count / valid_count,
        "original_nodata_preserved": True,
        "target_values_hidden_during_prediction": True,
        "comparison_note": (
            "Differences relative to the source DEM, "
            "not an independent estimate of true terrain accuracy"
        ),
        **statistics,
        **comparison,
    }

    write_json(OUTPUT_DIR / "report.json", report)

    save_figures(
        original,
        rebuilt,
        difference,
        predicted,
    )

    print("\nГотово.")
    print(f"Пересоздано пикселей: {predicted_count:,}")
    print(f"Покрытие: {100 * report['coverage']:.2f}%")

    if comparison["rmse_difference"] is not None:
        print(
            f"RMSE отличия от исходной DEM: "
            f"{comparison['rmse_difference']:.3f}"
        )
        print(
            f"Среднее отличие: "
            f"{comparison['mean_difference']:.3f}"
        )

    if predicted_count < valid_count:
        print(
            "Часть исходного покрытия не реконструирована "
            "из-за недостатка известного контекста. "
            "В результате эти ячейки оставлены NoData."
        )

    print(f"Результаты: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()