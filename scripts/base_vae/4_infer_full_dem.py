from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from rasterio.windows import Window

from train_vae import QuarryCVAE


# ============================= CONFIG =============================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DEM = PROJECT_ROOT / "data/raw/1m/grib_1m.tif"

# Укажи каталог своего обучения.
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "cvae_20261107_122414_747615"
    / "best_model.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / f"full_dem_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
)

PATCH_SIZE = 64
STRIDE = 32
INFERENCE_BATCH_SIZE = 32

# Искусственные пропуски.
NUMBER_OF_HOLES = 100
MIN_HOLE_SIZE = 8
MAX_HOLE_SIZE = 28

# Требуем наличие исходных данных вокруг каждого пропуска.
CONTEXT_MARGIN = 16

# Пропуски не должны соприкасаться.
HOLE_SEPARATION = 8

MAX_GENERATION_ATTEMPTS = 100_000

# Слишком повреждённые окна не используем.
MIN_KNOWN_FRACTION = 0.50

# Минимальный вес края окна при объединении.
MIN_BLEND_WEIGHT = 0.05

SEED = 42

# "auto", "cpu", "mps", "cuda".
DEVICE = "auto"

# Детерминированный режим, как при validation.
STOCHASTIC = False

# Дополнительный NoData, если он не прописан в TIFF.
EXTRA_NODATA_VALUE = None

# Если сохраняемая картинка слишком велика — прореживаем отображение.
OVERVIEW_MAX_SIDE = 1600

# Вывод прогресса.
PRINT_EVERY_WINDOWS = 500

# ==================================================================


def choose_device() -> torch.device:
    if DEVICE == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if DEVICE not in {"cpu", "mps", "cuda"}:
        raise ValueError("DEVICE должен быть auto, cpu, mps или cuda.")

    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна.")

    if DEVICE == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS недоступен.")

    return torch.device(DEVICE)


def validate_settings() -> None:
    if PATCH_SIZE != 64:
        raise ValueError("Ожидаются фрагменты 64 × 64.")

    if not 1 <= STRIDE <= PATCH_SIZE:
        raise ValueError("STRIDE должен быть от 1 до PATCH_SIZE.")

    if INFERENCE_BATCH_SIZE < 1:
        raise ValueError("INFERENCE_BATCH_SIZE должен быть >= 1.")

    if NUMBER_OF_HOLES < 1:
        raise ValueError("NUMBER_OF_HOLES должен быть >= 1.")

    if not 1 <= MIN_HOLE_SIZE <= MAX_HOLE_SIZE < PATCH_SIZE:
        raise ValueError("Некорректный размер пропусков.")

    if CONTEXT_MARGIN < 0 or HOLE_SEPARATION < 0:
        raise ValueError("Отступы не могут быть отрицательными.")

    if not 0 < MIN_KNOWN_FRACTION < 1:
        raise ValueError("Некорректный MIN_KNOWN_FRACTION.")

    if not 0 < MIN_BLEND_WEIGHT <= 1:
        raise ValueError("Некорректный MIN_BLEND_WEIGHT.")


def load_model(
    device: torch.device,
) -> tuple[QuarryCVAE, dict]:
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
            "Нужна модель с локальной min-max нормализацией."
        )

    if preprocessing.get("known_mask_convention") != (
        "1=known, 0=missing"
    ):
        raise ValueError("Неожиданное соглашение о маске.")

    if int(preprocessing["patch_size"]) != PATCH_SIZE:
        raise ValueError("Размер окна не совпадает с checkpoint.")

    model = QuarryCVAE(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def load_dem() -> tuple[np.ndarray, np.ndarray, dict]:
    with rasterio.open(SOURCE_DEM) as src:
        if src.count != 1:
            raise ValueError(
                "Ожидается одноканальная DEM. "
                f"В файле каналов: {src.count}."
            )

        if min(src.height, src.width) < PATCH_SIZE:
            raise ValueError("DEM меньше окна модели.")

        band = src.read(1, masked=True, out_dtype="float64")
        raw = np.asarray(band.data)

        valid = (
            ~np.ma.getmaskarray(band)
            & np.isfinite(raw)
        )

        if EXTRA_NODATA_VALUE is not None:
            valid &= raw != EXTRA_NODATA_VALUE

        # Приводим значения к физическим высотам.
        dem = raw * src.scales[0] + src.offsets[0]
        valid &= np.isfinite(dem)

        dem[~valid] = np.nan

        metadata = {
            "height": src.height,
            "width": src.width,
            "transform": src.transform,
            "crs": src.crs,
            "source_scale": float(src.scales[0]),
            "source_offset": float(src.offsets[0]),
        }

    if not valid.any():
        raise ValueError("В DEM нет пригодных высот.")

    return dem, valid, metadata


def generate_holes(
    valid: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    rng = np.random.default_rng(SEED)

    height, width = valid.shape
    holes = np.zeros(valid.shape, dtype=bool)
    records = []

    for _ in range(MAX_GENERATION_ATTEMPTS):
        if len(records) >= NUMBER_OF_HOLES:
            break

        hole_height = int(
            rng.integers(MIN_HOLE_SIZE, MAX_HOLE_SIZE + 1)
        )
        hole_width = int(
            rng.integers(MIN_HOLE_SIZE, MAX_HOLE_SIZE + 1)
        )

        row_min = CONTEXT_MARGIN
        col_min = CONTEXT_MARGIN

        row_max = height - CONTEXT_MARGIN - hole_height
        col_max = width - CONTEXT_MARGIN - hole_width

        if row_max < row_min or col_max < col_min:
            continue

        row = int(rng.integers(row_min, row_max + 1))
        col = int(rng.integers(col_min, col_max + 1))

        r1 = row + hole_height
        c1 = col + hole_width

        # В пропуске и окружающем контексте должны быть исходные данные.
        context_valid = valid[
            row - CONTEXT_MARGIN:r1 + CONTEXT_MARGIN,
            col - CONTEXT_MARGIN:c1 + CONTEXT_MARGIN,
        ]

        if not context_valid.all():
            continue

        sr0 = max(0, row - HOLE_SEPARATION)
        sc0 = max(0, col - HOLE_SEPARATION)
        sr1 = min(height, r1 + HOLE_SEPARATION)
        sc1 = min(width, c1 + HOLE_SEPARATION)

        if holes[sr0:sr1, sc0:sc1].any():
            continue

        if rng.random() < 0.5:
            local_hole = np.ones(
                (hole_height, hole_width),
                dtype=bool,
            )
            shape = "rectangle"
        else:
            yy, xx = np.mgrid[:hole_height, :hole_width]

            local_hole = (
                (
                    (yy - (hole_height - 1) / 2)
                    / (hole_height / 2)
                ) ** 2
                + (
                    (xx - (hole_width - 1) / 2)
                    / (hole_width / 2)
                ) ** 2
            ) <= 1

            shape = "ellipse"

        holes[row:r1, col:c1] |= local_hole

        records.append({
            "id": len(records) + 1,
            "row": row,
            "col": col,
            "height": hole_height,
            "width": hole_width,
            "shape": shape,
            "pixels": int(local_hole.sum()),
        })

    if len(records) != NUMBER_OF_HOLES:
        raise RuntimeError(
            f"Создано только {len(records)} из {NUMBER_OF_HOLES} "
            "пропусков. Уменьши их количество, размеры или отступы."
        )

    return holes, records


def window_starts(length: int) -> list[int]:
    starts = list(range(0, length - PATCH_SIZE + 1, STRIDE))

    last = length - PATCH_SIZE

    if starts[-1] != last:
        starts.append(last)

    return starts


def blending_weights() -> np.ndarray:
    axis_weights = np.hanning(PATCH_SIZE)
    weights = np.outer(axis_weights, axis_weights)
    weights = np.maximum(weights, MIN_BLEND_WEIGHT)

    return weights.astype(np.float64)


@torch.inference_mode()
def reconstruct(
    model: QuarryCVAE,
    observed: np.ndarray,
    known: np.ndarray,
    holes: np.ndarray,
    preprocessing: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    height, width = observed.shape

    prediction_sum = np.zeros(
        observed.shape,
        dtype=np.float64,
    )
    weight_sum = np.zeros(
        observed.shape,
        dtype=np.float32,
    )

    weights = blending_weights()

    minimum_scale = float(preprocessing["min_height_scale"])

    batch_inputs = []
    batch_metadata = []

    processed_windows = 0
    skipped_windows = 0

    def flush_batch() -> None:
        nonlocal processed_windows

        if not batch_inputs:
            return

        inputs = np.stack(batch_inputs, axis=0)
        tensor = torch.from_numpy(inputs).to(device)

        predictions = model(
            tensor,
            stochastic=STOCHASTIC,
        ).cpu().numpy()[:, 0]

        if not np.isfinite(predictions).all():
            raise RuntimeError("Модель вернула NaN/Inf.")

        for prediction, item in zip(predictions, batch_metadata):
            row, col, offset, scale = item

            region = np.s_[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE,
            ]

            local_holes = holes[region]

            prediction_m = (
                prediction.astype(np.float64) * scale + offset
            )

            # Накапливаем только предсказания искусственных пропусков.
            local_weights = weights * local_holes

            prediction_sum[region] += prediction_m * local_weights
            weight_sum[region] += local_weights.astype(np.float32)

        previous = processed_windows
        processed_windows += len(batch_inputs)

        if (
            processed_windows // PRINT_EVERY_WINDOWS
            > previous // PRINT_EVERY_WINDOWS
        ):
            print(f"Обработано окон: {processed_windows}")

        batch_inputs.clear()
        batch_metadata.clear()

    for row in window_starts(height):
        for col in window_starts(width):
            region = np.s_[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE,
            ]

            if not holes[region].any():
                continue

            local_known = known[region]

            if local_known.mean() < MIN_KNOWN_FRACTION:
                skipped_windows += 1
                continue

            local_dem = observed[region]
            visible = local_dem[local_known]

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

            condition = np.stack(
                [
                    normalized,
                    local_known.astype(np.float32),
                ],
                axis=0,
            )

            batch_inputs.append(condition)
            batch_metadata.append((row, col, offset, scale))

            if len(batch_inputs) >= INFERENCE_BATCH_SIZE:
                flush_batch()

    flush_batch()

    filled = holes & (weight_sum > 0)

    completed = observed.copy()
    completed[filled] = (
        prediction_sum[filled] / weight_sum[filled]
    )

    statistics = {
        "processed_windows": processed_windows,
        "skipped_windows": skipped_windows,
        "filled_pixels": int(filled.sum()),
        "unfilled_pixels": int((holes & ~filled).sum()),
    }

    return completed, filled, statistics


def write_geotiff(
    path: Path,
    array: np.ndarray,
    metadata: dict,
    is_mask: bool = False,
) -> None:
    profile = {
        "driver": "GTiff",
        "height": metadata["height"],
        "width": metadata["width"],
        "count": 1,
        "dtype": "uint8" if is_mask else "float64",
        "transform": metadata["transform"],
        "crs": metadata["crs"],
        "nodata": None if is_mask else np.nan,
        "compress": "deflate",
        "predictor": 2 if is_mask else 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
    }

    with rasterio.open(path, "w", **profile) as dst:
        # Запись блоками не создаёт ещё одну полную копию массива.
        for _, window in dst.block_windows(1):
            rows, cols = window.toslices()
            block = array[rows, cols]

            if is_mask:
                block = block.astype(np.uint8)

            dst.write(block, 1, window=window)

        dst.update_tags(
            project="QuarryVAE",
            processing=(
                "Artificial hole mask: 1=hole, 0=other"
                if is_mask
                else "Physical elevations; source scale/offset applied"
            ),
        )


def save_overview(
    original: np.ndarray,
    observed: np.ndarray,
    completed: np.ndarray,
    error: np.ndarray,
    holes: list[dict],
) -> None:
    step = max(
        1,
        int(np.ceil(max(original.shape) / OVERVIEW_MAX_SIDE)),
    )

    original_small = original[::step, ::step]
    observed_small = observed[::step, ::step]
    completed_small = completed[::step, ::step]

    vmin = float(np.nanmin(original_small))
    vmax = float(np.nanmax(original_small))

    if vmax <= vmin:
        vmax = vmin + 1.0

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))

    for axis, image, title in zip(
        axes,
        [original_small, observed_small, completed_small],
        ["Original DEM", "DEM with artificial holes", "Filled DEM"],
    ):
        shown = axis.imshow(
            image,
            cmap="terrain",
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(shown, ax=axis, shrink=0.7)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "overview.png", dpi=180)
    plt.close(fig)

    # Отдельный крупный план первого пропуска.
    first = holes[0]

    r0 = max(0, first["row"] - 20)
    c0 = max(0, first["col"] - 20)

    r1 = min(
        original.shape[0],
        first["row"] + first["height"] + 20,
    )
    c1 = min(
        original.shape[1],
        first["col"] + first["width"] + 20,
    )

    region = np.s_[r0:r1, c0:c1]

    local_original = original[region]
    local_completed = completed[region]
    local_error = error[region]

    low = float(
        min(
            np.nanmin(local_original),
            np.nanmin(local_completed),
        )
    )
    high = float(
        max(
            np.nanmax(local_original),
            np.nanmax(local_completed),
        )
    )

    if high <= low:
        high = low + 1.0

    finite_error = local_error[np.isfinite(local_error)]
    limit = (
        max(float(np.abs(finite_error).max()), 0.01)
        if finite_error.size
        else 0.01
    )

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))

    for index, (axis, image, title) in enumerate(zip(
        axes,
        [
            local_original,
            observed[region],
            local_completed,
            local_error,
        ],
        ["Original", "With hole", "Filled", "Error, m"],
    )):
        shown = axis.imshow(
            image,
            cmap="RdBu_r" if index == 3 else "terrain",
            vmin=-limit if index == 3 else low,
            vmax=limit if index == 3 else high,
        )

        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(shown, ax=axis, shrink=0.75)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hole_detail.png", dpi=180)
    plt.close(fig)


def main() -> None:
    validate_settings()

    if not SOURCE_DEM.is_file():
        raise FileNotFoundError(SOURCE_DEM)

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(CHECKPOINT_PATH)

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = choose_device()
    model, checkpoint = load_model(device)

    print(f"Device: {device}")
    print(f"Source: {SOURCE_DEM}")

    original, valid, metadata = load_dem()

    print(f"DEM shape: {original.shape}")
    print(f"Valid pixels: {int(valid.sum()):,}")

    if metadata["crs"] is None:
        print("CRS отсутствует; сохраняем результаты без назначения CRS.")

    holes, hole_records = generate_holes(valid)

    known = valid & ~holes

    observed = original.copy()
    observed[holes] = np.nan

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    print(f"Created holes: {len(hole_records)}")
    print(f"Hidden pixels: {int(holes.sum()):,}")

    # Сначала сохраняем повреждённую копию и маску.
    write_geotiff(
        OUTPUT_DIR / "dem_with_holes.tif",
        observed,
        metadata,
    )
    write_geotiff(
        OUTPUT_DIR / "artificial_holes.tif",
        holes,
        metadata,
        is_mask=True,
    )

    completed, filled, statistics = reconstruct(
        model=model,
        observed=observed,
        known=known,
        holes=holes,
        preprocessing=checkpoint["preprocessing"],
        device=device,
    )

    if not np.array_equal(
        completed[known],
        original[known],
    ):
        raise RuntimeError("Известные высоты изменились.")

    if np.isfinite(completed[~valid]).any():
        raise RuntimeError("Исходные NoData неожиданно заполнены.")

    error = np.full(
        original.shape,
        np.nan,
        dtype=np.float64,
    )
    error[filled] = completed[filled] - original[filled]

    error_values = error[filled]

    metrics = {}

    if error_values.size:
        metrics = {
            "rmse_m": float(
                np.sqrt(np.mean(error_values ** 2))
            ),
            "mae_m": float(
                np.mean(np.abs(error_values))
            ),
            "bias_m": float(
                np.mean(error_values)
            ),
            "max_absolute_error_m": float(
                np.max(np.abs(error_values))
            ),
        }

    write_geotiff(
        OUTPUT_DIR / "dem_filled.tif",
        completed,
        metadata,
    )
    write_geotiff(
        OUTPUT_DIR / "reconstruction_error.tif",
        error,
        metadata,
    )

    # Позволяет отличить восстановленные пиксели от оставшихся пропусков.
    write_geotiff(
        OUTPUT_DIR / "filled_pixels.tif",
        filled,
        metadata,
        is_mask=True,
    )

    report = {
        "source": str(SOURCE_DEM),
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "device": str(device),
        "seed": SEED,
        "stochastic": STOCHASTIC,
        "shape": list(original.shape),
        "crs": (
            metadata["crs"].to_string()
            if metadata["crs"] is not None
            else None
        ),
        "source_scale": metadata["source_scale"],
        "source_offset": metadata["source_offset"],
        "patch_size": PATCH_SIZE,
        "stride": STRIDE,
        "number_of_holes": len(hole_records),
        "hidden_pixels": int(holes.sum()),
        "coverage_fraction": float(filled.sum() / holes.sum()),
        "metrics_region": "successfully filled artificial holes",
        "known_pixels_unchanged": True,
        "original_nodata_preserved": True,
        **statistics,
        **metrics,
        "holes": hole_records,
    }

    (OUTPUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    save_overview(
        original,
        observed,
        completed,
        error,
        hole_records,
    )

    print("\nГотово.")
    print(f"Filled pixels: {statistics['filled_pixels']:,}")
    print(f"Unfilled pixels: {statistics['unfilled_pixels']:,}")

    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")

    print(f"Results: {OUTPUT_DIR}")

    if statistics["unfilled_pixels"]:
        print(
            "Внимание: часть пропусков осталась незаполненной. "
            "Метрики рассчитаны только по восстановленной части."
        )


if __name__ == "__main__":
    main()