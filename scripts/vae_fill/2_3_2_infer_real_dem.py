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
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

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
    / f"real_dem_contour_fill_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
)

SOURCE_BAND = 1

PATCH_SIZE = 64
STRIDE = 8
INFERENCE_BATCH_SIZE = 16

DEVICE = "auto"
SEED = 42
STOCHASTIC = False

# Меняется только область разрешённого заполнения.
# Требование к известному контексту оставляем.
MIN_KNOWN_FRACTION = 0.40

BLEND_SIGMA_FRACTION = 0.25
MIN_BLEND_WEIGHT = 1e-3
CONTEXT_WEIGHT_POWER = 2.0

PRINT_EVERY_WINDOWS = 500

# -------------------- Область заполнения --------------------

# Контур — выпуклая оболочка всех известных ячеек.
# Внутри него разрешены все NoData, независимо
# от связи с внешней областью.
DOMAIN_MODE = "convex_hull"

# Связность нужна только для группировки пустот в отчёте,
# а не для исключения связанных с краем областей.
HOLE_CONNECTIVITY = 8

# None — без ограничения площади пустоты.
MAX_HOLE_PIXELS = None

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

    if PATCH_SIZE != 64:
        raise ValueError("Ожидается PATCH_SIZE=64.")

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

    if DOMAIN_MODE != "convex_hull":
        raise ValueError("В этой версии DOMAIN_MODE='convex_hull'.")

    if HOLE_CONNECTIVITY not in {4, 8}:
        raise ValueError("HOLE_CONNECTIVITY: 4 или 8.")

    if MAX_HOLE_PIXELS is not None:
        if (
            not isinstance(MAX_HOLE_PIXELS, int)
            or MAX_HOLE_PIXELS < 1
        ):
            raise ValueError("MAX_HOLE_PIXELS: целое >= 1 или None.")

    if not isinstance(SOURCE_BAND, int) or SOURCE_BAND < 1:
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
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(CHECKPOINT_PATH)

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
        raise ValueError("Несовместимый размер окна.")

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
            raise ValueError(f"Нет канала {SOURCE_BAND}.")

        if min(src.height, src.width) < PATCH_SIZE:
            raise ValueError("DEM меньше окна 64 × 64.")

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
        }

    if not known.any():
        raise ValueError("В DEM нет известных высот.")

    return observed, known, metadata


def build_outer_domain(known):
    """
    Выпуклая оболочка площадей известных пикселей.

    Контур строится в координатах сетки:
    X — столбец, Y — строка.
    """
    print("\nПостроение внешнего контура")

    # Внутренние пиксели не влияют на выпуклую оболочку.
    edge = known & ~ndimage.binary_erosion(
        known,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )

    rows, cols = np.nonzero(edge)

    centers = np.column_stack([
        cols.astype(np.float64) + 0.5,
        rows.astype(np.float64) + 0.5,
    ])

    # Сначала сокращаем набор до вершин оболочки центров.
    # При вырожденном наборе оставляем крайние точки.
    if len(centers) >= 3:
        try:
            center_hull = ConvexHull(centers)
            centers = centers[center_hull.vertices]
        except QhullError:
            extreme_indices = np.unique([
                np.argmin(centers[:, 0]),
                np.argmax(centers[:, 0]),
                np.argmin(centers[:, 1]),
                np.argmax(centers[:, 1]),
            ])
            centers = centers[extreme_indices]

    # Учитываем площадь пикселя, а не только его центр.
    offsets = np.array([
        [-0.5, -0.5],
        [0.5, -0.5],
        [0.5, 0.5],
        [-0.5, 0.5],
    ])

    corners = (
        centers[:, None, :] + offsets[None, :, :]
    ).reshape(-1, 2)

    corners = np.unique(corners, axis=0)
    hull = ConvexHull(corners)

    contour_pixels = corners[hull.vertices]
    ring = np.vstack([
        contour_pixels,
        contour_pixels[0],
    ])

    polygon = {
        "type": "Polygon",
        "coordinates": [ring.tolist()],
    }

    domain = rasterize(
        [(polygon, 1)],
        out_shape=known.shape,
        transform=Affine.identity(),
        fill=0,
        all_touched=False,
        dtype="uint8",
    ).astype(bool)

    # Не допускаем потери исходных известных ячеек
    # из-за численной погрешности растеризации контура.
    domain |= known

    print(f"Вершин контура: {len(contour_pixels)}")
    print(f"Ячеек внутри контура: {int(domain.sum()):,}")

    return domain, contour_pixels


def detect_target_areas(known, domain):
    """
    Связь NoData с границей больше не исключает область.
    Условие: NoData находится внутри domain.
    """
    missing_inside = domain & ~known

    structure = ndimage.generate_binary_structure(
        2,
        2 if HOLE_CONNECTIVITY == 8 else 1,
    )

    labels, component_count = ndimage.label(
        missing_inside,
        structure=structure,
    )

    sizes = np.bincount(
        labels.ravel(),
        minlength=component_count + 1,
    )

    selected_lookup = np.ones(
        component_count + 1,
        dtype=bool,
    )
    selected_lookup[0] = False

    if MAX_HOLE_PIXELS is not None:
        selected_lookup &= sizes <= MAX_HOLE_PIXELS
        selected_lookup[0] = False

    target = selected_lookup[labels]
    excluded = missing_inside & ~target
    outside = ~domain

    records = []

    if component_count:
        slices = ndimage.find_objects(
            labels,
            max_label=component_count,
        )

        for component_id, region in enumerate(slices, start=1):
            if region is None:
                continue

            row_slice, col_slice = region

            records.append({
                "hole_id": component_id,
                "row": row_slice.start,
                "col": col_slice.start,
                "bbox_height": row_slice.stop - row_slice.start,
                "bbox_width": col_slice.stop - col_slice.start,
                "pixels": int(sizes[component_id]),
                "selected_for_inference": int(
                    selected_lookup[component_id]
                ),
            })

    statistics = {
        "missing_components_inside_contour": component_count,
        "selected_components": int(selected_lookup.sum()),
        "missing_pixels_inside_contour": int(missing_inside.sum()),
        "target_pixels": int(target.sum()),
        "excluded_pixels": int(excluded.sum()),
        "outside_contour_pixels": int(outside.sum()),
    }

    print("\nОбласти заполнения")
    print(f"Компонентов пустот: {component_count:,}")
    print(f"Целевых пикселей: {statistics['target_pixels']:,}")
    print(f"Вне контура: {statistics['outside_contour_pixels']:,}")

    return (
        labels,
        missing_inside,
        target,
        outside,
        excluded,
        records,
        statistics,
    )


def window_starts(length):
    starts = list(
        range(0, length - PATCH_SIZE + 1, STRIDE)
    )

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
    model.eval()
    started = time.perf_counter()

    target_indices = np.flatnonzero(target.ravel())
    target_count = len(target_indices)

    if target_count >= np.iinfo(np.int32).max:
        raise ValueError("Слишком много целевых пикселей.")

    index_map = np.full(observed.shape, -1, dtype=np.int32)
    index_map.ravel()[target_indices] = np.arange(
        target_count,
        dtype=np.int32,
    )

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

    spread = np.full(
        observed.shape,
        np.nan,
        dtype=np.float64,
    )
    spread.ravel()[filled_indices] = np.sqrt(
        np.maximum(m2[usable] / weight_sum[usable], 0)
    )

    counts = np.zeros(observed.shape, dtype=np.uint32)
    counts.ravel()[target_indices] = prediction_count

    statistics = {
        "processed_windows": processed,
        "skipped_windows": skipped,
        "inference_seconds": float(time.perf_counter() - started),
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


def save_surfaces(observed, completed, filled, spread, metadata):
    restored_only = np.full(
        observed.shape,
        np.nan,
        dtype=np.float64,
    )
    restored_only[filled] = completed[filled]

    layers = [
        ("dem_filled.tif", "completed_surface", completed),
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
                {"band": index, "file": filename, "name": name}
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
    results = []

    for record in records:
        row, col = record["row"], record["col"]

        region = np.s_[
            row:row + record["bbox_height"],
            col:col + record["bbox_width"],
        ]

        local = (
            (labels[region] == record["hole_id"])
            & filled[region]
        )

        filled_count = int(local.sum())
        local_counts = counts[region][local]

        results.append({
            **record,
            "filled_pixels": filled_count,
            "unfilled_pixels": record["pixels"] - filled_count,
            "coverage": filled_count / record["pixels"],
            "mean_contributing_windows": (
                float(local_counts.mean())
                if local_counts.size else None
            ),
        })

    return results


def save_figures(observed, completed, filled, spread, domain, records):
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

    restored = np.where(filled, completed, np.nan)

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    images = [observed, completed, restored, domain]
    titles = [
        "Original DEM",
        "Completed DEM",
        "Restored elevations only",
        "Allowed outer domain",
    ]

    for index, (axis, array, title) in enumerate(
        zip(axes, images, titles)
    ):
        if index == 3:
            image = axis.imshow(
                array[::step, ::step],
                cmap="gray",
                vmin=0,
                vmax=1,
            )
        else:
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
    fig.savefig(OUTPUT_DIR / "overview.png", dpi=160)
    plt.close(fig)

    selected = [
        record for record in records
        if record["selected_for_inference"]
    ]

    if not selected:
        return

    selected.sort(key=lambda item: item["pixels"])

    count = min(DETAIL_EXAMPLES, len(selected))
    indices = np.linspace(0, len(selected) - 1, count, dtype=int)

    fig, axes = plt.subplots(
        count,
        3,
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
        after = completed[region]
        finite_after = after[np.isfinite(after)]

        lo = float(finite_after.min()) if finite_after.size else low
        hi = (
            max(float(finite_after.max()), lo + 1e-6)
            if finite_after.size else high
        )

        for column, (array, title) in enumerate(zip(
            [observed[region], after, spread[region]],
            ["Observed", "Completed", "Window spread"],
        )):
            axis = axes[line, column]

            if column == 2:
                image = axis.imshow(array, cmap="magma")
            else:
                image = axis.imshow(
                    array,
                    cmap="terrain",
                    vmin=lo,
                    vmax=hi,
                )

            axis.set_title(
                f"Area {record['hole_id']} | {title}",
                fontsize=9,
            )
            axis.axis("off")
            fig.colorbar(image, ax=axis, shrink=0.7)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hole_examples.png", dpi=140)
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

    domain, contour_pixels = build_outer_domain(known)

    (
        labels,
        missing_inside,
        target,
        outside,
        excluded,
        records,
        domain_statistics,
    ) = detect_target_areas(known, domain)

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

    # Контур в исходных координатах DEM, без локального смещения.
    transform = metadata["transform"]
    contour_world = [
        list(transform * (float(col), float(row)))
        for col, row in contour_pixels
    ]
    contour_world.append(contour_world[0])

    write_json(
        OUTPUT_DIR / "outer_contour.json",
        {
            "coordinate_space": "source DEM coordinates",
            "crs_wkt": (
                metadata["crs"].to_wkt()
                if metadata["crs"] is not None else None
            ),
            "ring": contour_world,
            "method": "convex hull of valid pixel footprints",
        },
    )

    write_geotiff(
        OUTPUT_DIR / "dem_original_with_gaps.tif",
        observed,
        metadata,
        name="original_surface_with_real_gaps",
    )

    print(f"\nDevice: {device}")
    print(f"DEM: {observed.shape}")
    print(f"Resolution: {metadata['resolution']}")

    if metadata["crs"] is None:
        print("CRS отсутствует; координаты сохраняются без преобразования.")

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
        print("Пустот внутри контура нет.")

        completed = observed.copy()
        filled = np.zeros(observed.shape, dtype=bool)
        spread = np.full(observed.shape, np.nan, dtype=np.float64)
        counts = np.zeros(observed.shape, dtype=np.uint32)

        inference_statistics = {
            "processed_windows": 0,
            "skipped_windows": 0,
            "inference_seconds": 0.0,
            "predictions_reused_as_context": False,
        }
        checkpoint_epoch = None

    if not np.array_equal(completed[known], observed[known]):
        raise RuntimeError("Известные высоты изменились.")

    if np.isfinite(completed[outside]).any():
        raise RuntimeError("Появились высоты вне внешнего контура.")

    if np.isfinite(completed[excluded]).any():
        raise RuntimeError("Заполнены исключённые области.")

    if (filled & ~target).any():
        raise RuntimeError("Заполнение вышло за пределы целевой маски.")

    save_surfaces(
        observed,
        completed,
        filled,
        spread,
        metadata,
    )

    if SAVE_DIAGNOSTIC_MASKS:
        masks = [
            ("outer_domain.tif", domain),
            ("missing_inside_contour.tif", missing_inside),
            ("target_holes.tif", target),
            ("outside_contour.tif", outside),
            ("excluded_holes.tif", excluded),
            ("filled_pixels.tif", filled),
            ("unfilled_inside_contour.tif", missing_inside & ~filled),
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
            OUTPUT_DIR / "hole_component_ids.tif",
            labels,
            metadata,
            integer=True,
            name="missing_component_id",
        )

    if SAVE_PREDICTION_COUNT:
        write_geotiff(
            OUTPUT_DIR / "prediction_count.tif",
            counts,
            metadata,
            integer=True,
            name="number_of_contributing_windows",
        )

    write_csv(
        OUTPUT_DIR / "holes.csv",
        make_hole_report(labels, filled, counts, records),
    )

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
        "domain_mode": DOMAIN_MODE,
        "patch_size": PATCH_SIZE,
        "stride": STRIDE,
        "min_known_fraction": MIN_KNOWN_FRACTION,
        "filled_pixels": filled_pixels,
        "unfilled_target_pixels": target_pixels - filled_pixels,
        "target_coverage": (
            filled_pixels / target_pixels
            if target_pixels else None
        ),
        "known_pixels_unchanged": True,
        "outside_contour_preserved": True,
        "artificial_defects_created": False,
        "rmse": None,
        "accuracy_note": (
            "True heights inside real gaps are unknown. "
            "Window spread is not a calibrated accuracy estimate."
        ),
        **domain_statistics,
        **inference_statistics,
    }

    write_json(OUTPUT_DIR / "report.json", report)

    save_figures(
        observed,
        completed,
        filled,
        spread,
        domain,
        records,
    )

    print("\nГотово.")
    print(f"Восстановлено пикселей: {filled_pixels:,}")

    if target_pixels:
        print(
            f"Покрытие: "
            f"{100 * filled_pixels / target_pixels:.2f}%"
        )

    if filled_pixels < target_pixels:
        print(
            "Часть пустот не заполнена: "
            "окна с достаточным известным контекстом отсутствуют."
        )

    print("Вне построенного контура заполнение не выполнялось.")
    print(f"Результаты: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()