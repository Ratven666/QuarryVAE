from __future__ import annotations

import csv
import json
import time
from collections import Counter
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DEM = PROJECT_ROOT / "data/raw/1m/grib_1m.tif"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "cvae_inpaint_20261107_155343_972452"
    / "best_model.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / f"full_dem_defects_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
)

PATCH_SIZE = 64
STRIDE = 16
INFERENCE_BATCH_SIZE = 16

DEVICE = "auto"
SEED = 42
STOCHASTIC = False

MIN_KNOWN_FRACTION = 0.40
MIN_BLEND_WEIGHT = 0.05

PRINT_EVERY_WINDOWS = 500

# -------------------- Дефекты --------------------

DEFECTS_PER_GROUP = 20

DEFECT_TYPES = (
    "square",
    "circle",
    "ellipse",
    "stripe",
    "capsule",
)

SIZE_GROUPS = ("small", "medium", "large")

# Все размеры — в пикселях.
SQUARE_SIZES = {
    "small": (6, 12),
    "medium": (14, 26),
    "large": (30, 52),
}

CIRCLE_DIAMETERS = {
    "small": (6, 12),
    "medium": (14, 26),
    "large": (30, 54),
}

ELLIPSE_AXES = {
    "small": ((12, 24), (5, 10)),
    "medium": ((28, 56), (8, 18)),
    "large": ((60, 110), (14, 32)),
}

STRIPE_DIMENSIONS = {
    "small": ((30, 60), (3, 5)),
    "medium": ((65, 120), (4, 8)),
    "large": ((130, 240), (6, 12)),
}

EDGE_MARGIN = 4
VALID_CONTEXT_MARGIN = 8
DEFECT_SEPARATION = 4

PLACEMENT_ATTEMPTS_PER_DEFECT = 500

EXTRA_NODATA_VALUE = None

# -------------------- Сохранение --------------------

SAVE_MULTIBAND_GEOTIFF = True
SAVE_DIAGNOSTIC_MASKS = True

GEOTIFF_BLOCK_SIZE = 256

OVERVIEW_MAX_SIDE = 1600
DETAIL_EXAMPLES = 12

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

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
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
        raise ValueError("Ожидается PATCH_SIZE=64.")

    if not isinstance(STRIDE, int) or not 1 <= STRIDE <= PATCH_SIZE:
        raise ValueError("STRIDE должен быть целым от 1 до PATCH_SIZE.")

    if INFERENCE_BATCH_SIZE < 1:
        raise ValueError("INFERENCE_BATCH_SIZE должен быть >= 1.")

    if not 0 < MIN_KNOWN_FRACTION < 1:
        raise ValueError("Некорректный MIN_KNOWN_FRACTION.")

    if not 0 < MIN_BLEND_WEIGHT <= 1:
        raise ValueError("Некорректный MIN_BLEND_WEIGHT.")

    if PRINT_EVERY_WINDOWS < 1:
        raise ValueError("PRINT_EVERY_WINDOWS должен быть >= 1.")

    if DEFECTS_PER_GROUP < 1:
        raise ValueError("DEFECTS_PER_GROUP должен быть >= 1.")

    if PLACEMENT_ATTEMPTS_PER_DEFECT < 1:
        raise ValueError("PLACEMENT_ATTEMPTS_PER_DEFECT должен быть >= 1.")

    if min(
        EDGE_MARGIN,
        VALID_CONTEXT_MARGIN,
        DEFECT_SEPARATION,
    ) < 0:
        raise ValueError("Отступы не могут быть отрицательными.")

    if DETAIL_EXAMPLES < 1 or OVERVIEW_MAX_SIDE < 1:
        raise ValueError("Некорректные параметры визуализации.")

    if (
        not isinstance(GEOTIFF_BLOCK_SIZE, int)
        or GEOTIFF_BLOCK_SIZE < 16
        or GEOTIFF_BLOCK_SIZE % 16 != 0
    ):
        raise ValueError("GEOTIFF_BLOCK_SIZE должен быть кратен 16.")

    allowed_types = {
        "square",
        "circle",
        "ellipse",
        "stripe",
        "capsule",
    }

    if not DEFECT_TYPES or not set(DEFECT_TYPES).issubset(allowed_types):
        raise ValueError("Некорректные DEFECT_TYPES.")

    if not SIZE_GROUPS:
        raise ValueError("SIZE_GROUPS не должен быть пустым.")

    for group in SIZE_GROUPS:
        bounds_list = [
            SQUARE_SIZES[group],
            CIRCLE_DIAMETERS[group],
            *ELLIPSE_AXES[group],
            *STRIPE_DIMENSIONS[group],
        ]

        for lower, upper in bounds_list:
            if not (
                np.isfinite([lower, upper]).all()
                and 0 < lower <= upper
            ):
                raise ValueError(
                    f"Некорректные размеры для группы {group}."
                )


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
            "Нужен checkpoint с локальной min-max нормализацией."
        )

    if preprocessing.get("known_mask_convention") != "1=known, 0=missing":
        raise ValueError("Несовместимое соглашение о маске.")

    if int(preprocessing["patch_size"]) != PATCH_SIZE:
        raise ValueError("Размер окна не совпадает с checkpoint.")

    minimum_scale = float(
        preprocessing["min_height_scale"]
    )

    if not np.isfinite(minimum_scale) or minimum_scale <= 0:
        raise ValueError(
            "Некорректный min_height_scale в checkpoint."
        )

    model = QuarryCVAE(
        **checkpoint["model_config"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, checkpoint


def load_dem():
    with rasterio.open(SOURCE_DEM) as src:
        if src.count != 1:
            raise ValueError("Ожидается одноканальная DEM.")

        if min(src.height, src.width) < PATCH_SIZE:
            raise ValueError("DEM меньше окна модели.")

        band = src.read(
            1,
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

        scale = float(src.scales[0])
        offset = float(src.offsets[0])

        if not np.isfinite([scale, offset]).all():
            raise ValueError("Некорректные scale/offset исходной DEM.")

        with np.errstate(
            over="ignore",
            invalid="ignore",
        ):
            dem = raw * scale + offset

        valid &= np.isfinite(dem)
        dem[~valid] = np.nan

        metadata = {
            "height": src.height,
            "width": src.width,
            "transform": src.transform,
            "crs": src.crs,
            "source_scale": scale,
            "source_offset": offset,
            "resolution": list(src.res),
        }

    if not valid.any():
        raise ValueError("Нет валидных высот.")

    return dem, valid, metadata


def sample_range(rng, bounds):
    return float(
        rng.uniform(bounds[0], bounds[1])
    )


def make_template(rng, kind, group):
    """Создаёт локальную растровую маску дефекта."""
    angle = float(rng.uniform(0, np.pi))

    if kind == "square":
        length = width = sample_range(
            rng,
            SQUARE_SIZES[group],
        )

    elif kind == "circle":
        length = width = sample_range(
            rng,
            CIRCLE_DIAMETERS[group],
        )
        angle = 0.0

    elif kind == "ellipse":
        major, minor = ELLIPSE_AXES[group]
        length = sample_range(rng, major)
        width = sample_range(rng, minor)

    elif kind in {"stripe", "capsule"}:
        long_bounds, width_bounds = STRIPE_DIMENSIONS[group]
        length = sample_range(rng, long_bounds)
        width = sample_range(rng, width_bounds)

    else:
        raise ValueError(kind)

    cosine = np.cos(angle)
    sine = np.sin(angle)

    half_x = (
        abs(cosine) * length / 2
        + abs(sine) * width / 2
    )

    half_y = (
        abs(sine) * length / 2
        + abs(cosine) * width / 2
    )

    radius_x = int(np.ceil(half_x)) + 1
    radius_y = int(np.ceil(half_y)) + 1

    yy, xx = np.mgrid[
        -radius_y:radius_y + 1,
        -radius_x:radius_x + 1,
    ]

    u = xx * cosine + yy * sine
    v = -xx * sine + yy * cosine

    if kind in {"square", "stripe"}:
        mask = (
            (np.abs(u) <= length / 2)
            & (np.abs(v) <= width / 2)
        )

    elif kind in {"circle", "ellipse"}:
        mask = (
            (u / (length / 2)) ** 2
            + (v / (width / 2)) ** 2
        ) <= 1

    else:
        segment_half = max(
            (length - width) / 2,
            0,
        )

        distance_u = np.maximum(
            np.abs(u) - segment_half,
            0,
        )

        mask = (
            distance_u ** 2 + v ** 2
        ) <= (width / 2) ** 2

    rows, cols = np.nonzero(mask)

    if not rows.size:
        raise RuntimeError("Сформирована пустая маска дефекта.")

    mask = mask[
        rows.min():rows.max() + 1,
        cols.min():cols.max() + 1,
    ]

    return mask, {
        "type": kind,
        "size_group": group,
        "length_px": length,
        "width_px": width,
        "angle_deg": float(np.degrees(angle)),
    }


def create_defects(valid):
    rng = np.random.default_rng(SEED)

    labels = np.zeros(
        valid.shape,
        dtype=np.uint32,
    )

    height, width = valid.shape

    plan = [
        (kind, group)
        for kind in DEFECT_TYPES
        for group in SIZE_GROUPS
        for _ in range(DEFECTS_PER_GROUP)
    ]

    rng.shuffle(plan)

    records = []
    failed = Counter()

    for kind, group in plan:
        placed = False

        for _ in range(PLACEMENT_ATTEMPTS_PER_DEFECT):
            mask, info = make_template(
                rng,
                kind,
                group,
            )

            mh, mw = mask.shape

            margin = max(
                EDGE_MARGIN,
                VALID_CONTEXT_MARGIN,
            )

            max_row = height - margin - mh
            max_col = width - margin - mw

            if max_row < margin or max_col < margin:
                continue

            row = int(
                rng.integers(margin, max_row + 1)
            )

            col = int(
                rng.integers(margin, max_col + 1)
            )

            r1 = row + mh
            c1 = col + mw

            context = valid[
                row - VALID_CONTEXT_MARGIN:r1 + VALID_CONTEXT_MARGIN,
                col - VALID_CONTEXT_MARGIN:c1 + VALID_CONTEXT_MARGIN,
            ]

            if not context.all():
                continue

            sr0 = max(0, row - DEFECT_SEPARATION)
            sc0 = max(0, col - DEFECT_SEPARATION)

            sr1 = min(height, r1 + DEFECT_SEPARATION)
            sc1 = min(width, c1 + DEFECT_SEPARATION)

            if labels[sr0:sr1, sc0:sc1].any():
                continue

            defect_id = len(records) + 1

            region = labels[row:r1, col:c1]
            region[mask] = defect_id

            records.append({
                "id": defect_id,
                **info,
                "row": row,
                "col": col,
                "bbox_height": mh,
                "bbox_width": mw,
                "pixels": int(mask.sum()),
            })

            placed = True
            break

        if not placed:
            failed[f"{kind}/{group}"] += 1

    if not records:
        raise RuntimeError(
            "Не удалось разместить дефекты. "
            "Уменьши размеры или требования к контексту."
        )

    print(
        f"Создано дефектов: "
        f"{len(records)} / {len(plan)}"
    )

    counts = Counter(
        f"{record['type']}/{record['size_group']}"
        for record in records
    )

    for key, count in sorted(counts.items()):
        print(f"  {key}: {count}")

    if failed:
        print(
            "Не удалось разместить:",
            dict(failed),
        )

    return labels, records, dict(failed)


def window_starts(length):
    if length < PATCH_SIZE:
        raise ValueError(
            "Размер оси меньше окна модели."
        )

    starts = list(
        range(
            0,
            length - PATCH_SIZE + 1,
            STRIDE,
        )
    )

    last = length - PATCH_SIZE

    if starts[-1] != last:
        starts.append(last)

    return starts


@torch.inference_mode()
def reconstruct(
    model,
    observed,
    valid,
    labels,
    preprocessing,
    device,
):
    model.eval()

    started = time.perf_counter()

    height, width = observed.shape

    holes = labels > 0

    # Только реальные известные высоты.
    # Предсказания не становятся новым контекстом.
    known = valid & ~holes

    if not np.isfinite(observed[known]).all():
        raise ValueError(
            "В известных пикселях есть NaN/Inf."
        )

    prediction_sum = np.zeros(
        observed.shape,
        dtype=np.float64,
    )

    weight_sum = np.zeros(
        observed.shape,
        dtype=np.float64,
    )

    axis = np.hanning(PATCH_SIZE)

    blend = np.maximum(
        np.outer(axis, axis),
        MIN_BLEND_WEIGHT,
    )

    batch_inputs = []
    batch_info = []

    processed = 0
    skipped = 0

    minimum_scale = float(
        preprocessing["min_height_scale"]
    )

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
            raise RuntimeError(
                "Модель вернула NaN/Inf."
            )

        for prediction, info in zip(
            predictions,
            batch_info,
        ):
            row, col, offset, scale = info

            region = np.s_[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE,
            ]

            local_weights = (
                blend * holes[region]
            )

            prediction_m = (
                prediction.astype(np.float64)
                * scale
                + offset
            )

            prediction_sum[region] += (
                prediction_m * local_weights
            )

            weight_sum[region] += local_weights

        previous = processed
        processed += len(batch_inputs)

        if (
            processed // PRINT_EVERY_WINDOWS
            > previous // PRINT_EVERY_WINDOWS
        ):
            print(
                f"Обработано окон: {processed:,}"
            )

        batch_inputs.clear()
        batch_info.clear()

    for row in window_starts(height):
        for col in window_starts(width):
            region = np.s_[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE,
            ]

            if not holes[region].any():
                continue

            local_known = known[region]

            if (
                local_known.mean() < MIN_KNOWN_FRACTION
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
                raise RuntimeError(
                    "Ошибка нормализации окна."
                )

            batch_inputs.append(
                np.stack(
                    [
                        normalized,
                        local_known.astype(np.float32),
                    ],
                    axis=0,
                )
            )

            batch_info.append(
                (row, col, offset, scale)
            )

            if len(batch_inputs) >= INFERENCE_BATCH_SIZE:
                flush()

    flush()

    filled = holes & (weight_sum > 0)

    completed = observed.copy()

    completed[filled] = (
        prediction_sum[filled]
        / weight_sum[filled]
    )

    if not np.isfinite(completed[filled]).all():
        raise RuntimeError(
            "NaN/Inf после объединения предсказаний."
        )

    return completed, filled, {
        "processed_windows": processed,
        "skipped_windows": skipped,
        "inference_seconds": float(
            time.perf_counter() - started
        ),
    }


def geotiff_profile(
    metadata,
    count=1,
    integer=False,
):
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


def write_geotiff(
    path,
    array,
    metadata,
    integer=False,
    name=None,
    description=None,
):
    profile = geotiff_profile(
        metadata,
        count=1,
        integer=integer,
    )

    with rasterio.open(
        path,
        "w",
        **profile,
    ) as dst:
        for _, window in dst.block_windows(1):
            rows, cols = window.toslices()

            block = np.asarray(
                array[rows, cols],
                dtype=(
                    np.uint32
                    if integer
                    else np.float64
                ),
            )

            dst.write(
                block,
                1,
                window=window,
            )

        if name is not None:
            dst.set_band_description(1, name)

        if description is not None:
            dst.update_tags(
                1,
                description=description,
            )


def save_dem_layers(
    original,
    observed,
    completed,
    filled,
    metadata,
):
    expected_shape = (
        metadata["height"],
        metadata["width"],
    )

    for name, array in (
        ("original", original),
        ("observed", observed),
        ("completed", completed),
        ("filled", filled),
    ):
        if array.shape != expected_shape:
            raise ValueError(
                f"Неверный размер {name}: {array.shape}."
            )

    if not np.isfinite(original[filled]).all():
        raise ValueError(
            "Для восстановленных пикселей отсутствует "
            "базовая высота."
        )

    if not np.isfinite(completed[filled]).all():
        raise ValueError(
            "В восстановленных пикселях есть NaN/Inf."
        )

    restored_only = np.full(
        expected_shape,
        np.nan,
        dtype=np.float64,
    )

    restored_only[filled] = completed[filled]

    error = np.full(
        expected_shape,
        np.nan,
        dtype=np.float64,
    )

    error[filled] = (
        completed[filled] - original[filled]
    )

    layers = [
        {
            "filename": "dem_filled.tif",
            "name": "completed_surface",
            "description": (
                "Observed elevations plus reconstructed elevations; "
                "unresolved gaps remain NoData"
            ),
            "array": completed,
        },
        {
            "filename": "dem_with_defects.tif",
            "name": "surface_with_gaps",
            "description": (
                "Surface after artificial gaps were introduced"
            ),
            "array": observed,
        },
        {
            "filename": "dem_restored_only.tif",
            "name": "restored_elevations_only",
            "description": (
                "Reconstructed elevations at successfully filled "
                "pixels only; all other pixels are NoData"
            ),
            "array": restored_only,
        },
        {
            "filename": "reconstruction_error.tif",
            "name": "reconstruction_error",
            "description": (
                "Reconstructed minus original elevation "
                "at successfully filled pixels only"
            ),
            "array": error,
        },
        {
            "filename": "dem_original.tif",
            "name": "original_surface",
            "description": (
                "Original surface before artificial gaps"
            ),
            "array": original,
        },
    ]

    print("\nСохранение пяти слоёв DEM")

    for layer in layers:
        write_geotiff(
            OUTPUT_DIR / layer["filename"],
            layer["array"],
            metadata,
            name=layer["name"],
            description=layer["description"],
        )

        print(f"  {layer['filename']}")

    if SAVE_MULTIBAND_GEOTIFF:
        comparison_path = (
            OUTPUT_DIR / "dem_comparison.tif"
        )

        profile = geotiff_profile(
            metadata,
            count=len(layers),
        )

        with rasterio.open(
            comparison_path,
            "w",
            **profile,
        ) as dst:
            for band_index, layer in enumerate(
                layers,
                start=1,
            ):
                dst.set_band_description(
                    band_index,
                    layer["name"],
                )

                dst.update_tags(
                    band_index,
                    description=layer["description"],
                )

            # Запись блоками без np.stack пяти полных DEM.
            for _, window in dst.block_windows(1):
                rows, cols = window.toslices()

                for band_index, layer in enumerate(
                    layers,
                    start=1,
                ):
                    block = np.asarray(
                        layer["array"][rows, cols],
                        dtype=np.float64,
                    )

                    dst.write(
                        block,
                        band_index,
                        window=window,
                    )

        print(f"  {comparison_path.name}")

    write_json(
        OUTPUT_DIR / "dem_layers.json",
        {
            "multiband_file": (
                "dem_comparison.tif"
                if SAVE_MULTIBAND_GEOTIFF
                else None
            ),
            "nodata": "NaN",
            "error_definition": (
                "reconstructed elevation minus original elevation"
            ),
            "layers": [
                {
                    "band": band_index,
                    "name": layer["name"],
                    "file": layer["filename"],
                    "description": layer["description"],
                }
                for band_index, layer in enumerate(
                    layers,
                    start=1,
                )
            ],
        },
    )

    return error


def error_metrics(values):
    if not values.size:
        return {
            "rmse_m": None,
            "mae_m": None,
            "bias_m": None,
            "max_absolute_error_m": None,
        }

    return {
        "rmse_m": float(
            np.sqrt(np.mean(values ** 2))
        ),
        "mae_m": float(
            np.mean(np.abs(values))
        ),
        "bias_m": float(
            np.mean(values)
        ),
        "max_absolute_error_m": float(
            np.max(np.abs(values))
        ),
    }


def build_reports(
    original,
    completed,
    labels,
    filled,
    records,
):
    per_defect = []

    for record in records:
        row = record["row"]
        col = record["col"]

        region = np.s_[
            row:row + record["bbox_height"],
            col:col + record["bbox_width"],
        ]

        local_defect = (
            labels[region] == record["id"]
        )

        local_filled = (
            local_defect & filled[region]
        )

        values = (
            completed[region][local_filled]
            - original[region][local_filled]
        )

        per_defect.append({
            **record,
            "filled_pixels": int(
                local_filled.sum()
            ),
            "unfilled_pixels": int(
                local_defect.sum()
                - local_filled.sum()
            ),
            "coverage": float(
                local_filled.sum() / record["pixels"]
            ),
            **error_metrics(values),
        })

    grouped = []

    for kind in DEFECT_TYPES:
        for group in SIZE_GROUPS:
            selected = [
                record
                for record in records
                if (
                    record["type"] == kind
                    and record["size_group"] == group
                )
            ]

            if not selected:
                continue

            errors = []
            total_pixels = 0
            filled_pixels = 0

            for record in selected:
                row = record["row"]
                col = record["col"]

                region = np.s_[
                    row:row + record["bbox_height"],
                    col:col + record["bbox_width"],
                ]

                local = (
                    (labels[region] == record["id"])
                    & filled[region]
                )

                errors.append(
                    completed[region][local]
                    - original[region][local]
                )

                total_pixels += record["pixels"]
                filled_pixels += int(local.sum())

            values = np.concatenate(errors)

            grouped.append({
                "type": kind,
                "size_group": group,
                "defects": len(selected),
                "pixels": total_pixels,
                "filled_pixels": filled_pixels,
                "unfilled_pixels": (
                    total_pixels - filled_pixels
                ),
                "coverage": (
                    filled_pixels / total_pixels
                ),
                **error_metrics(values),
            })

    return per_defect, grouped


def save_figures(
    original,
    observed,
    completed,
    error,
    records,
):
    step = max(
        1,
        int(
            np.ceil(
                max(original.shape) / OVERVIEW_MAX_SIDE
            )
        ),
    )

    arrays = [
        original[::step, ::step],
        observed[::step, ::step],
        completed[::step, ::step],
    ]

    finite_original = arrays[0][
        np.isfinite(arrays[0])
    ]

    if not finite_original.size:
        finite_original = original[
            np.isfinite(original)
        ]

    low = float(finite_original.min())
    high = max(
        float(finite_original.max()),
        low + 1e-6,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6),
    )

    for axis, array, title in zip(
        axes,
        arrays,
        [
            "Original DEM",
            "With defects",
            "Completed DEM",
        ],
    ):
        image = axis.imshow(
            array,
            cmap="terrain",
            vmin=low,
            vmax=high,
        )

        axis.set_title(title)
        axis.axis("off")

        fig.colorbar(
            image,
            ax=axis,
            shrink=0.7,
        )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "overview.png",
        dpi=180,
    )
    plt.close(fig)

    candidates = []

    for kind in DEFECT_TYPES:
        for group in SIZE_GROUPS:
            matches = [
                record
                for record in records
                if (
                    record["type"] == kind
                    and record["size_group"] == group
                )
            ]

            if matches:
                candidates.append(matches[0])

    if not candidates:
        return

    count = min(
        DETAIL_EXAMPLES,
        len(candidates),
    )

    chosen_indices = np.linspace(
        0,
        len(candidates) - 1,
        count,
        dtype=int,
    )

    examples = [
        candidates[index]
        for index in chosen_indices
    ]

    fig, axes = plt.subplots(
        count,
        4,
        figsize=(17, 3.2 * count),
        squeeze=False,
    )

    for row_index, record in enumerate(examples):
        r0 = max(
            0,
            record["row"] - 12,
        )
        c0 = max(
            0,
            record["col"] - 12,
        )

        r1 = min(
            original.shape[0],
            record["row"] + record["bbox_height"] + 12,
        )
        c1 = min(
            original.shape[1],
            record["col"] + record["bbox_width"] + 12,
        )

        region = np.s_[r0:r1, c0:c1]

        source = original[region]
        result = completed[region]
        local_error = error[region]

        finite_source = source[
            np.isfinite(source)
        ]
        finite_result = result[
            np.isfinite(result)
        ]

        lo = float(finite_source.min())
        hi = float(finite_source.max())

        if finite_result.size:
            lo = min(
                lo,
                float(finite_result.min()),
            )
            hi = max(
                hi,
                float(finite_result.max()),
            )

        hi = max(
            hi,
            lo + 1e-6,
        )

        finite_error = local_error[
            np.isfinite(local_error)
        ]

        error_limit = (
            max(
                float(np.abs(finite_error).max()),
                0.01,
            )
            if finite_error.size
            else 0.01
        )

        images = [
            source,
            observed[region],
            result,
            local_error,
        ]

        titles = [
            "Original",
            "Defect",
            "Completed",
            "Error, m",
        ]

        for col_index, (array, title) in enumerate(
            zip(images, titles)
        ):
            axis = axes[row_index, col_index]
            is_error = col_index == 3

            image = axis.imshow(
                array,
                cmap=(
                    "RdBu_r"
                    if is_error
                    else "terrain"
                ),
                vmin=(
                    -error_limit
                    if is_error
                    else lo
                ),
                vmax=(
                    error_limit
                    if is_error
                    else hi
                ),
            )

            axis.set_title(
                f"{record['type']}/{record['size_group']} | {title}",
                fontsize=9,
            )
            axis.axis("off")

            fig.colorbar(
                image,
                ax=axis,
                shrink=0.7,
            )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "defect_examples.png",
        dpi=150,
    )
    plt.close(fig)


def save_run_config(device):
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in globals().items()
        if (
            key.isupper()
            and isinstance(
                value,
                (
                    str,
                    int,
                    float,
                    bool,
                    tuple,
                    dict,
                    Path,
                    type(None),
                ),
            )
        )
    }

    config["selected_device"] = str(device)
    config["torch_version"] = str(torch.__version__)
    config["numpy_version"] = str(np.__version__)

    write_json(
        OUTPUT_DIR / "config.json",
        config,
    )


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

    print(f"Device: {device}")
    print(f"DEM: {original.shape}")
    print(f"Resolution: {metadata['resolution']}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")

    if metadata["crs"] is None:
        print(
            "CRS отсутствует: результаты сохраняются "
            "без назначения системы координат."
        )

    labels, records, failed = create_defects(valid)

    holes = labels > 0

    observed = original.copy()
    observed[holes] = np.nan

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    save_run_config(device)

    write_csv(
        OUTPUT_DIR / "defect_geometry.csv",
        records,
    )

    # Сохраняем повреждённую DEM до начала инференса.
    write_geotiff(
        OUTPUT_DIR / "dem_with_defects.tif",
        observed,
        metadata,
        name="surface_with_gaps",
    )

    if SAVE_DIAGNOSTIC_MASKS:
        write_geotiff(
            OUTPUT_DIR / "defect_ids.tif",
            labels,
            metadata,
            integer=True,
            name="artificial_defect_id",
            description="0=background; positive value=defect ID",
        )

    completed, filled, statistics = reconstruct(
        model=model,
        observed=observed,
        valid=valid,
        labels=labels,
        preprocessing=checkpoint["preprocessing"],
        device=device,
    )

    unchanged = valid & ~holes

    if not np.array_equal(
        completed[unchanged],
        original[unchanged],
    ):
        raise RuntimeError(
            "Известные высоты изменились."
        )

    if np.isfinite(completed[~valid]).any():
        raise RuntimeError(
            "Исходные NoData неожиданно заполнены."
        )

    # Пять основных слоёв и общий пятиканальный GeoTIFF.
    error = save_dem_layers(
        original=original,
        observed=observed,
        completed=completed,
        filled=filled,
        metadata=metadata,
    )

    if SAVE_DIAGNOSTIC_MASKS:
        write_geotiff(
            OUTPUT_DIR / "filled_pixels.tif",
            filled,
            metadata,
            integer=True,
            name="filled_pixels",
            description="1=successfully reconstructed artificial gap",
        )

        write_geotiff(
            OUTPUT_DIR / "unfilled_pixels.tif",
            holes & ~filled,
            metadata,
            integer=True,
            name="unfilled_pixels",
            description="1=artificial gap left unfilled",
        )

    per_defect, grouped = build_reports(
        original=original,
        completed=completed,
        labels=labels,
        filled=filled,
        records=records,
    )

    write_csv(
        OUTPUT_DIR / "defects.csv",
        per_defect,
    )

    write_csv(
        OUTPUT_DIR / "metrics_by_type_and_size.csv",
        grouped,
    )

    report = {
        "source": str(SOURCE_DEM),
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "device": str(device),
        "seed": SEED,
        "stochastic": STOCHASTIC,
        "shape": list(original.shape),
        "resolution": metadata["resolution"],
        "patch_size": PATCH_SIZE,
        "stride": STRIDE,
        "min_known_fraction": MIN_KNOWN_FRACTION,
        "created_defects": len(records),
        "unplaced_defects": failed,
        "hidden_pixels": int(holes.sum()),
        "filled_pixels": int(filled.sum()),
        "unfilled_pixels": int(
            (holes & ~filled).sum()
        ),
        "coverage": float(
            filled.sum() / holes.sum()
        ),
        "known_pixels_unchanged": True,
        "original_nodata_preserved": True,
        "predictions_reused_as_context": False,
        "metrics_region": (
            "successfully reconstructed artificial holes"
        ),
        "error_definition": (
            "reconstructed elevation minus original elevation"
        ),
        "source_scale": metadata["source_scale"],
        "source_offset": metadata["source_offset"],
        "crs": (
            metadata["crs"].to_string()
            if metadata["crs"] is not None
            else None
        ),
        "main_layers": [
            "dem_filled.tif",
            "dem_with_defects.tif",
            "dem_restored_only.tif",
            "reconstruction_error.tif",
            "dem_original.tif",
        ],
        "multiband_file": (
            "dem_comparison.tif"
            if SAVE_MULTIBAND_GEOTIFF
            else None
        ),
        **statistics,
        **error_metrics(error[filled]),
    }

    write_json(
        OUTPUT_DIR / "report.json",
        report,
    )

    save_figures(
        original=original,
        observed=observed,
        completed=completed,
        error=error,
        records=records,
    )

    print("\nГотово.")
    print(
        f"Скрыто пикселей: "
        f"{report['hidden_pixels']:,}"
    )
    print(
        f"Восстановлено: "
        f"{report['filled_pixels']:,}"
    )
    print(
        f"Покрытие: "
        f"{100 * report['coverage']:.2f}%"
    )

    if report["rmse_m"] is not None:
        print(f"RMSE: {report['rmse_m']:.3f} м")
        print(f"MAE: {report['mae_m']:.3f} м")
        print(f"Bias: {report['bias_m']:.3f} м")

    if report["unfilled_pixels"]:
        print(
            "Внимание: часть пропусков осталась незаполненной. "
            "Метрики рассчитаны только по восстановленным пикселям."
        )

    print(f"Результаты: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()