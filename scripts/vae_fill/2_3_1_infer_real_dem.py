from __future__ import annotations

import csv
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
from scipy import ndimage

from train_vae import QuarryCVAE


# ============================= CONFIG =============================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DEM = PROJECT_ROOT / "data/derived/dem_sever_quarry.tif"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "cvae_inpaint_20261107_155343_972452"
    / "best_model.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / f"real_dem_fill_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
)

SOURCE_BAND = 1

PATCH_SIZE = 64
STRIDE = 8
INFERENCE_BATCH_SIZE = 16

DEVICE = "auto"
SEED = 42
STOCHASTIC = False

MIN_KNOWN_FRACTION = 0.40

# Гауссово объединение перекрывающихся окон.
BLEND_SIGMA_FRACTION = 0.25
MIN_BLEND_WEIGHT = 1e-3

# Эвристический вес окна по количеству известных высот.
# 0 — не учитывать этот фактор.
CONTEXT_WEIGHT_POWER = 2.0

PRINT_EVERY_WINDOWS = 500

# -------------------- Поиск пропусков --------------------

# 8 — учитываем диагональные связи с внешней областью.
# 4 — только горизонтальные/вертикальные связи.
HOLE_CONNECTIVITY = 8

# Необязательное ограничение площади внутреннего отверстия.
# None — пытаться заполнить все внутренние отверстия.
MAX_HOLE_PIXELS = None

# Дополнительный NoData в исходных, ещё не масштабированных значениях.
# Не задавай 0, если нулевая высота может быть валидной.
EXTRA_NODATA_VALUE = None

# -------------------- Сохранение --------------------

SAVE_MULTIBAND_GEOTIFF = True
SAVE_DIAGNOSTIC_MASKS = True
SAVE_PREDICTION_COUNT = True
SAVE_WINDOW_SPREAD = True

GEOTIFF_BLOCK_SIZE = 256

OVERVIEW_MAX_SIDE = 1600
DETAIL_EXAMPLES = 8
DETAIL_MARGIN = 16

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


def write_csv(path, records):
    if not records:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0]),
        )
        writer.writeheader()
        writer.writerows(records)


def validate_settings():
    if not SOURCE_DEM.is_file():
        raise FileNotFoundError(SOURCE_DEM)

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(CHECKPOINT_PATH)

    if PATCH_SIZE != 64:
        raise ValueError("Текущая модель настроена на PATCH_SIZE=64.")

    if not isinstance(STRIDE, int) or not 1 <= STRIDE <= PATCH_SIZE:
        raise ValueError("STRIDE должен быть целым от 1 до PATCH_SIZE.")

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

    if HOLE_CONNECTIVITY not in {4, 8}:
        raise ValueError("HOLE_CONNECTIVITY должен быть 4 или 8.")

    if (
        MAX_HOLE_PIXELS is not None
        and (
            not isinstance(MAX_HOLE_PIXELS, int)
            or MAX_HOLE_PIXELS < 1
        )
    ):
        raise ValueError("MAX_HOLE_PIXELS: положительное целое или None.")

    if (
        not isinstance(SOURCE_BAND, int)
        or SOURCE_BAND < 1
    ):
        raise ValueError("SOURCE_BAND должен быть целым >= 1.")

    if (
        not isinstance(GEOTIFF_BLOCK_SIZE, int)
        or GEOTIFF_BLOCK_SIZE < 16
        or GEOTIFF_BLOCK_SIZE % 16
    ):
        raise ValueError("GEOTIFF_BLOCK_SIZE должен быть кратен 16.")

    if min(
        PRINT_EVERY_WINDOWS,
        OVERVIEW_MAX_SIDE,
        DETAIL_EXAMPLES,
    ) < 1:
        raise ValueError("Некорректные параметры вывода.")

    if DETAIL_MARGIN < 0:
        raise ValueError("DETAIL_MARGIN должен быть >= 0.")


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
        raise ValueError("Несовместимая нормализация checkpoint.")

    if preprocessing.get("known_mask_convention") != "1=known, 0=missing":
        raise ValueError("Несовместимое соглашение о маске.")

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
            raise ValueError(
                f"В DEM отсутствует канал {SOURCE_BAND}."
            )

        if min(src.height, src.width) < PATCH_SIZE:
            raise ValueError(
                "Размер DEM меньше 64 пикселей по одной из осей."
            )

        band = src.read(
            SOURCE_BAND,
            masked=True,
            out_dtype="float64",
        )

        raw = np.asarray(band.data)

        known = (
            ~np.ma.getmaskarray(band)
            & np.isfinite(raw)
        )

        if EXTRA_NODATA_VALUE is not None:
            known &= raw != EXTRA_NODATA_VALUE

        scale = float(src.scales[SOURCE_BAND - 1])
        offset = float(src.offsets[SOURCE_BAND - 1])

        if not np.isfinite([scale, offset]).all():
            raise ValueError("Некорректные scale/offset.")

        with np.errstate(over="ignore", invalid="ignore"):
            observed = raw * scale + offset

        known &= np.isfinite(observed)
        observed[~known] = np.nan

        metadata = {
            "height": src.height,
            "width": src.width,
            "transform": src.transform,
            "crs": src.crs,
            "resolution": list(src.res),
            "source_scale": scale,
            "source_offset": offset,
            "source_nodata": (
                float(src.nodatavals[SOURCE_BAND - 1])
                if src.nodatavals[SOURCE_BAND - 1] is not None
                and np.isfinite(src.nodatavals[SOURCE_BAND - 1])
                else None
            ),
        }

    if not known.any():
        raise ValueError("В исходной DEM нет известных высот.")

    return observed, known, metadata


def detect_internal_holes(known):
    """
    Компоненты NoData, связанные с границей растра,
    считаются внешней областью и исключаются.
    """
    structure = ndimage.generate_binary_structure(
        2,
        2 if HOLE_CONNECTIVITY == 8 else 1,
    )

    missing = ~known
    labels, component_count = ndimage.label(
        missing,
        structure=structure,
    )

    sizes = np.bincount(
        labels.ravel(),
        minlength=component_count + 1,
    )

    border_labels = np.unique(
        np.concatenate([
            labels[0, :],
            labels[-1, :],
            labels[:, 0],
            labels[:, -1],
        ])
    )

    external_lookup = np.zeros(
        component_count + 1,
        dtype=bool,
    )
    external_lookup[border_labels] = True
    external_lookup[0] = False

    external = external_lookup[labels]

    internal_lookup = np.ones(
        component_count + 1,
        dtype=bool,
    )
    internal_lookup[0] = False
    internal_lookup[external_lookup] = False

    target_lookup = internal_lookup.copy()

    if MAX_HOLE_PIXELS is not None:
        target_lookup &= sizes <= MAX_HOLE_PIXELS
        target_lookup[0] = False

    internal = internal_lookup[labels]
    target = target_lookup[labels]
    excluded = internal & ~target

    # find_objects возвращает bounding box каждого компонента.
    component_slices = ndimage.find_objects(
        labels,
        max_label=component_count,
    )

    records = []

    for component_id in np.flatnonzero(internal_lookup):
        region = component_slices[component_id - 1]

        if region is None:
            continue

        row_slice, col_slice = region

        records.append({
            "hole_id": int(component_id),
            "row": row_slice.start,
            "col": col_slice.start,
            "bbox_height": row_slice.stop - row_slice.start,
            "bbox_width": col_slice.stop - col_slice.start,
            "pixels": int(sizes[component_id]),
            "selected_for_inference": int(
                target_lookup[component_id]
            ),
        })

    statistics = {
        "nodata_components": int(component_count),
        "internal_holes": len(records),
        "selected_holes": int(target_lookup.sum()),
        "internal_hole_pixels": int(internal.sum()),
        "target_pixels": int(target.sum()),
        "external_nodata_pixels": int(external.sum()),
        "excluded_internal_pixels": int(excluded.sum()),
    }

    print("\nПоиск реальных пропусков")
    print(f"Внутренних отверстий: {statistics['internal_holes']:,}")
    print(f"Выбрано отверстий: {statistics['selected_holes']:,}")
    print(f"Пикселей для заполнения: {statistics['target_pixels']:,}")
    print(
        "Внешних NoData, не заполняются: "
        f"{statistics['external_nodata_pixels']:,}"
    )

    return labels, internal, target, external, excluded, records, statistics


def window_starts(length):
    starts = list(range(0, length - PATCH_SIZE + 1, STRIDE))

    last = length - PATCH_SIZE
    if starts[-1] != last:
        starts.append(last)

    return starts


def blend_weights():
    coordinates = (
        np.arange(PATCH_SIZE, dtype=np.float64)
        - (PATCH_SIZE - 1) / 2
    )

    sigma = PATCH_SIZE * BLEND_SIGMA_FRACTION
    axis = np.exp(-0.5 * (coordinates / sigma) ** 2)

    weights = np.outer(axis, axis)
    weights /= weights.max()

    return np.maximum(weights, MIN_BLEND_WEIGHT)


@torch.inference_mode()
def reconstruct(model, observed, known, target, preprocessing, device):
    """
    Восстанавливаются только target-пиксели.

    Статистика накапливается в компактных массивах,
    содержащих только внутренние пропуски.
    """
    model.eval()
    started = time.perf_counter()

    target_indices = np.flatnonzero(target.ravel())
    target_count = len(target_indices)

    if target_count >= np.iinfo(np.int32).max:
        raise ValueError("Слишком много целевых пикселей для int32.")

    index_map = np.full(observed.shape, -1, dtype=np.int32)
    index_map.ravel()[target_indices] = np.arange(
        target_count,
        dtype=np.int32,
    )

    # Взвешенное онлайн-накопление среднего и дисперсии.
    mean = np.zeros(target_count, dtype=np.float64)
    m2 = np.zeros(target_count, dtype=np.float64)
    weight_sum = np.zeros(target_count, dtype=np.float64)
    prediction_count = np.zeros(target_count, dtype=np.uint32)

    blend = blend_weights()
    minimum_scale = float(preprocessing["min_height_scale"])

    row_presence = target.any(axis=1)
    col_presence = target.any(axis=0)

    rows = [
        row for row in window_starts(observed.shape[0])
        if row_presence[row:row + PATCH_SIZE].any()
    ]
    cols = [
        col for col in window_starts(observed.shape[1])
        if col_presence[col:col + PATCH_SIZE].any()
    ]

    batch_inputs = []
    batch_info = []

    processed = 0
    skipped = 0

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
            row, col, offset, scale, context_weight = info

            region = np.s_[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE,
            ]

            local_target = target[region]
            ids = index_map[region][local_target]

            weights = (
                blend[local_target] * context_weight
            )

            values = (
                prediction[local_target].astype(np.float64)
                * scale
                + offset
            )

            old_weights = weight_sum[ids]
            new_weights = old_weights + weights

            delta = values - mean[ids]
            updated_mean = (
                mean[ids] + delta * weights / new_weights
            )

            m2[ids] += (
                weights * delta * (values - updated_mean)
            )
            mean[ids] = updated_mean
            weight_sum[ids] = new_weights
            prediction_count[ids] += 1

        previous = processed
        processed += len(batch_inputs)

        if (
            processed // PRINT_EVERY_WINDOWS
            > previous // PRINT_EVERY_WINDOWS
        ):
            print(f"Обработано окон: {processed:,}")

        batch_inputs.clear()
        batch_info.clear()

    for row in rows:
        for col in cols:
            region = np.s_[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE,
            ]

            if not target[region].any():
                continue

            local_known = known[region]
            fraction = float(local_known.mean())

            if (
                fraction < MIN_KNOWN_FRACTION
                or not local_known.any()
            ):
                skipped += 1
                continue

            visible = observed[region][local_known]

            offset = float(visible.min())
            scale = max(
                float(visible.max()) - offset,
                minimum_scale,
            )

            normalized = np.zeros(
                (PATCH_SIZE, PATCH_SIZE),
                dtype=np.float32,
            )
            normalized[local_known] = (
                (visible - offset) / scale
            ).astype(np.float32)

            if not np.isfinite(normalized).all():
                raise RuntimeError("Ошибка нормализации окна.")

            batch_inputs.append(np.stack([
                normalized,
                local_known.astype(np.float32),
            ]))

            batch_info.append((
                row,
                col,
                offset,
                scale,
                fraction ** CONTEXT_WEIGHT_POWER,
            ))

            if len(batch_inputs) >= INFERENCE_BATCH_SIZE:
                flush()

    flush()

    usable = weight_sum > 0
    filled_indices = target_indices[usable]

    completed = observed.copy()
    completed.ravel()[filled_indices] = mean[usable]

    if not np.isfinite(completed.ravel()[filled_indices]).all():
        raise RuntimeError("NaN/Inf после объединения предсказаний.")

    filled = np.zeros(observed.shape, dtype=bool)
    filled.ravel()[filled_indices] = True

    spread = np.full(observed.shape, np.nan, dtype=np.float64)
    spread.ravel()[filled_indices] = np.sqrt(
        np.maximum(m2[usable] / weight_sum[usable], 0)
    )

    counts = np.zeros(observed.shape, dtype=np.uint32)
    counts.ravel()[target_indices] = prediction_count

    statistics = {
        "processed_windows": processed,
        "skipped_windows": skipped,
        "inference_seconds": float(time.perf_counter() - started),
        "blend": "gaussian",
        "predictions_reused_as_context": False,
    }

    return completed, filled, spread, counts, statistics


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
    profile = geotiff_profile(metadata, integer=integer)

    with rasterio.open(path, "w", **profile) as dst:
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


def save_surfaces(observed, completed, filled, spread, metadata):
    restored_only = np.full(observed.shape, np.nan, dtype=np.float64)
    restored_only[filled] = completed[filled]

    layers = [
        (
            "dem_filled.tif",
            "completed_surface",
            completed,
        ),
        (
            "dem_original_with_gaps.tif",
            "original_surface_with_real_gaps",
            observed,
        ),
        (
            "dem_restored_only.tif",
            "restored_elevations_only",
            restored_only,
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
            for band_index, (_, name, _) in enumerate(layers, start=1):
                dst.set_band_description(band_index, name)

            for _, window in dst.block_windows(1):
                rows, cols = window.toslices()

                for band_index, (_, _, array) in enumerate(
                    layers,
                    start=1,
                ):
                    dst.write(
                        np.asarray(array[rows, cols], dtype=np.float64),
                        band_index,
                        window=window,
                    )

    write_json(
        OUTPUT_DIR / "dem_layers.json",
        {
            "multiband_file": (
                "dem_comparison.tif"
                if SAVE_MULTIBAND_GEOTIFF else None
            ),
            "layers": [
                {
                    "band": index,
                    "file": filename,
                    "name": name,
                }
                for index, (filename, name, _) in enumerate(
                    layers,
                    start=1,
                )
            ],
            "error_layer": None,
            "reason": "True elevations inside real gaps are unknown",
        },
    )


def make_hole_report(labels, filled, counts, records):
    result = []

    for record in records:
        row = record["row"]
        col = record["col"]

        region = np.s_[
            row:row + record["bbox_height"],
            col:col + record["bbox_width"],
        ]

        local_hole = labels[region] == record["hole_id"]
        local_filled = local_hole & filled[region]

        filled_count = int(local_filled.sum())
        local_counts = counts[region][local_filled]

        result.append({
            **record,
            "filled_pixels": filled_count,
            "unfilled_pixels": record["pixels"] - filled_count,
            "coverage": filled_count / record["pixels"],
            "mean_contributing_windows": (
                float(local_counts.mean())
                if local_counts.size else None
            ),
        })

    return result


def save_figures(observed, completed, filled, spread, records):
    step = max(
        1,
        int(np.ceil(max(observed.shape) / OVERVIEW_MAX_SIDE)),
    )

    sampled = observed[::step, ::step]
    finite = sampled[np.isfinite(sampled)]

    if not finite.size:
        finite = observed[np.isfinite(observed)]

    low = float(finite.min())
    high = max(float(finite.max()), low + 1e-6)

    restored_view = np.where(filled, completed, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for axis, array, title in zip(
        axes,
        [observed, completed, restored_view],
        [
            "Original DEM with real gaps",
            "Completed DEM",
            "Restored elevations only",
        ],
    ):
        image = axis.imshow(
            array[::step, ::step],
            cmap="terrain",
            vmin=low,
            vmax=high,
        )
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(image, ax=axis, shrink=0.7)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "overview.png", dpi=170)
    plt.close(fig)

    selected = [
        record for record in records
        if record["selected_for_inference"]
    ]

    if not selected:
        return

    selected = sorted(
        selected,
        key=lambda item: item["pixels"],
    )

    count = min(DETAIL_EXAMPLES, len(selected))
    indices = np.linspace(0, len(selected) - 1, count, dtype=int)

    fig, axes = plt.subplots(
        count, 3,
        figsize=(15, 3.5 * count),
        squeeze=False,
    )

    for line, index in enumerate(indices):
        record = selected[index]

        r0 = max(0, record["row"] - DETAIL_MARGIN)
        c0 = max(0, record["col"] - DETAIL_MARGIN)

        r1 = min(
            observed.shape[0],
            record["row"] + record["bbox_height"] + DETAIL_MARGIN,
        )
        c1 = min(
            observed.shape[1],
            record["col"] + record["bbox_width"] + DETAIL_MARGIN,
        )

        region = np.s_[r0:r1, c0:c1]

        before = observed[region]
        after = completed[region]
        local_spread = spread[region]

        finite_after = after[np.isfinite(after)]

        if finite_after.size:
            lo = float(finite_after.min())
            hi = max(float(finite_after.max()), lo + 1e-6)
        else:
            lo, hi = low, high

        for column, (image_data, title) in enumerate(zip(
            [before, after, local_spread],
            ["Observed", "Completed", "Window spread, height units"],
        )):
            axis = axes[line, column]

            if column == 2:
                image = axis.imshow(image_data, cmap="magma")
            else:
                image = axis.imshow(
                    image_data,
                    cmap="terrain",
                    vmin=lo,
                    vmax=hi,
                )

            axis.set_title(
                f"Hole {record['hole_id']} | {title}",
                fontsize=9,
            )
            axis.axis("off")
            fig.colorbar(image, ax=axis, shrink=0.7)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hole_examples.png", dpi=150)
    plt.close(fig)


def main():
    validate_settings()

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    observed, known, metadata = load_dem()

    (
        labels,
        internal,
        target,
        external,
        excluded,
        records,
        hole_statistics,
    ) = detect_internal_holes(known)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    device = choose_device()

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

    write_json(OUTPUT_DIR / "config.json", config)

    print(f"\nDevice: {device}")
    print(f"DEM: {observed.shape}")
    print(f"Resolution: {metadata['resolution']}")

    if metadata["crs"] is None:
        print("CRS отсутствует; новая CRS не назначается.")

    # Сохраняем исходную поверхность до инференса.
    write_geotiff(
        OUTPUT_DIR / "dem_original_with_gaps.tif",
        observed,
        metadata,
        name="original_surface_with_real_gaps",
    )

    if target.any():
        model, checkpoint = load_model(device)

        completed, filled, spread, counts, inference_statistics = (
            reconstruct(
                model,
                observed,
                known,
                target,
                checkpoint["preprocessing"],
                device,
            )
        )
        checkpoint_epoch = int(checkpoint["epoch"])

    else:
        print("Внутренних пропусков для заполнения нет.")

        completed = observed.copy()
        filled = np.zeros(observed.shape, dtype=bool)
        spread = np.full(observed.shape, np.nan, dtype=np.float64)
        counts = np.zeros(observed.shape, dtype=np.uint32)

        inference_statistics = {
            "processed_windows": 0,
            "skipped_windows": 0,
            "inference_seconds": 0.0,
        }
        checkpoint_epoch = None

    # Строгие проверки: никакой экстраполяции и изменения опорных данных.
    if not np.array_equal(completed[known], observed[known]):
        raise RuntimeError("Известные высоты изменились.")

    if np.isfinite(completed[external]).any():
        raise RuntimeError("Внешняя область неожиданно заполнена.")

    if np.isfinite(completed[excluded]).any():
        raise RuntimeError("Исключённые отверстия неожиданно заполнены.")

    if (filled & ~target).any():
        raise RuntimeError("Заполнены пиксели вне целевой маски.")

    save_surfaces(
        observed,
        completed,
        filled,
        spread,
        metadata,
    )

    if SAVE_DIAGNOSTIC_MASKS:
        masks = [
            ("internal_holes.tif", internal),
            ("target_holes.tif", target),
            ("external_nodata.tif", external),
            ("excluded_internal_holes.tif", excluded),
            ("filled_pixels.tif", filled),
            ("unfilled_internal_holes.tif", internal & ~filled),
        ]

        for filename, mask in masks:
            write_geotiff(
                OUTPUT_DIR / filename,
                mask,
                metadata,
                integer=True,
                name=Path(filename).stem,
            )

        write_geotiff(
            OUTPUT_DIR / "nodata_component_ids.tif",
            labels,
            metadata,
            integer=True,
            name="nodata_component_id",
        )

    if SAVE_PREDICTION_COUNT:
        write_geotiff(
            OUTPUT_DIR / "prediction_count.tif",
            counts,
            metadata,
            integer=True,
            name="number_of_contributing_windows",
        )

    hole_report = make_hole_report(
        labels,
        filled,
        counts,
        records,
    )
    write_csv(OUTPUT_DIR / "holes.csv", hole_report)

    target_pixels = int(target.sum())
    filled_pixels = int(filled.sum())

    report = {
        "source": str(SOURCE_DEM),
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": checkpoint_epoch,
        "device": str(device),
        "shape": list(observed.shape),
        "resolution": metadata["resolution"],
        "crs": (
            metadata["crs"].to_string()
            if metadata["crs"] is not None else None
        ),
        "hole_connectivity": HOLE_CONNECTIVITY,
        "patch_size": PATCH_SIZE,
        "stride": STRIDE,
        "filled_pixels": filled_pixels,
        "unfilled_target_pixels": target_pixels - filled_pixels,
        "target_coverage": (
            filled_pixels / target_pixels
            if target_pixels else None
        ),
        "known_pixels_unchanged": True,
        "external_nodata_preserved": True,
        "artificial_defects_created": False,
        "rmse": None,
        "accuracy_note": (
            "True elevations inside real gaps are unknown. "
            "Window spread is disagreement between predictions, "
            "not a calibrated error estimate."
        ),
        **hole_statistics,
        **inference_statistics,
    }

    write_json(OUTPUT_DIR / "report.json", report)

    save_figures(
        observed,
        completed,
        filled,
        spread,
        records,
    )

    print("\nГотово.")
    print(f"Восстановлено пикселей: {filled_pixels:,}")

    if target_pixels:
        print(
            "Покрытие выбранных отверстий: "
            f"{100 * filled_pixels / target_pixels:.2f}%"
        )

    if target_pixels > filled_pixels:
        print(
            "Некоторые внутренние пропуски остались незаполненными: "
            "не хватило подходящих окон с известным контекстом."
        )

    print("Внешняя область не изменена.")
    print(f"Результаты: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()