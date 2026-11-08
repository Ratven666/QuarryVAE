from __future__ import annotations

import csv
import json
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
    / "cvae_inpaint"
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

# Минимальная доля известных пикселей в обрабатываемом окне.
MIN_KNOWN_FRACTION = 0.40

MIN_BLEND_WEIGHT = 0.05

# Количество экземпляров каждого типа и каждого диапазона размеров.
DEFECTS_PER_GROUP = 20

DEFECT_TYPES = (
    "square",
    "circle",
    "ellipse",
    "stripe",
    "capsule",
)

SIZE_GROUPS = ("small", "medium", "large")

# Все размеры ниже — в пикселях, не в единицах CRS.
#
# square: длина стороны.
SQUARE_SIZES = {
    "small": (6, 12),
    "medium": (14, 26),
    "large": (30, 52),
}

# circle: диаметр.
CIRCLE_DIAMETERS = {
    "small": (6, 12),
    "medium": (14, 26),
    "large": (30, 54),
}

# ellipse: длины главной и малой осей.
ELLIPSE_AXES = {
    "small": ((12, 24), (5, 10)),
    "medium": ((28, 56), (8, 18)),
    "large": ((60, 110), (14, 32)),
}

# stripe/capsule: полная длина и ширина.
STRIPE_DIMENSIONS = {
    "small": ((30, 60), (3, 5)),
    "medium": ((65, 120), (4, 8)),
    "large": ((130, 240), (6, 12)),
}

# Пустой отступ от границы изображения.
EDGE_MARGIN = 4

# Требование исходных данных вокруг формы.
VALID_CONTEXT_MARGIN = 8

# Минимальное разделение между дефектами.
DEFECT_SEPARATION = 4

PLACEMENT_ATTEMPTS_PER_DEFECT = 500

# Дополнительный NoData, если он не указан в метаданных.
EXTRA_NODATA_VALUE = None

OVERVIEW_MAX_SIDE = 1600
DETAIL_EXAMPLES = 12

# ==================================================================



def write_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_csv(path, records):
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


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
        raise ValueError("Нужен checkpoint с локальной min-max нормализацией.")

    if preprocessing.get("known_mask_convention") != "1=known, 0=missing":
        raise ValueError("Несовместимое соглашение о маске.")

    if int(preprocessing["patch_size"]) != PATCH_SIZE:
        raise ValueError("Размер окна не совпадает с checkpoint.")

    model = QuarryCVAE(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def load_dem():
    with rasterio.open(SOURCE_DEM) as src:
        if src.count != 1:
            raise ValueError("Ожидается одноканальная DEM.")
        if min(src.height, src.width) < PATCH_SIZE:
            raise ValueError("DEM меньше окна модели.")

        band = src.read(1, masked=True, out_dtype="float64")
        raw = np.asarray(band.data)

        valid = ~np.ma.getmaskarray(band) & np.isfinite(raw)

        if EXTRA_NODATA_VALUE is not None:
            valid &= raw != EXTRA_NODATA_VALUE

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
        raise ValueError("Нет валидных высот.")

    return dem, valid, metadata


def sample_range(rng, bounds):
    return float(rng.uniform(bounds[0], bounds[1]))


def make_template(rng, kind, group):
    """Создаёт локальную растровую маску повёрнутой формы."""
    angle = float(rng.uniform(0, np.pi))

    if kind == "square":
        length = width = sample_range(rng, SQUARE_SIZES[group])

    elif kind == "circle":
        length = width = sample_range(rng, CIRCLE_DIAMETERS[group])
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

    cosine, sine = np.cos(angle), np.sin(angle)

    # Консервативная ограничивающая рамка повёрнутой формы.
    half_x = abs(cosine) * length / 2 + abs(sine) * width / 2
    half_y = abs(sine) * length / 2 + abs(cosine) * width / 2

    radius_x = int(np.ceil(half_x)) + 1
    radius_y = int(np.ceil(half_y)) + 1

    yy, xx = np.mgrid[
        -radius_y:radius_y + 1,
        -radius_x:radius_x + 1,
    ]

    u = xx * cosine + yy * sine
    v = -xx * sine + yy * cosine

    if kind in {"square", "stripe"}:
        mask = (np.abs(u) <= length / 2) & (np.abs(v) <= width / 2)

    elif kind in {"circle", "ellipse"}:
        mask = (u / (length / 2)) ** 2 + (v / (width / 2)) ** 2 <= 1

    else:
        # Полная длина капсулы включает округлые окончания.
        segment_half = max((length - width) / 2, 0)
        distance_u = np.maximum(np.abs(u) - segment_half, 0)
        mask = distance_u ** 2 + v ** 2 <= (width / 2) ** 2

    rows, cols = np.nonzero(mask)
    mask = mask[
        rows.min():rows.max() + 1,
        cols.min():cols.max() + 1,
    ]

    info = {
        "type": kind,
        "size_group": group,
        "length_px": length,
        "width_px": width,
        "angle_deg": float(np.degrees(angle)),
    }

    return mask, info


def create_defects(valid):
    rng = np.random.default_rng(SEED)

    labels = np.zeros(valid.shape, dtype=np.uint32)
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
            mask, info = make_template(rng, kind, group)
            mh, mw = mask.shape

            margin = max(EDGE_MARGIN, VALID_CONTEXT_MARGIN)

            max_row = height - margin - mh
            max_col = width - margin - mw

            if max_row < margin or max_col < margin:
                continue

            row = int(rng.integers(margin, max_row + 1))
            col = int(rng.integers(margin, max_col + 1))

            r1, c1 = row + mh, col + mw

            # Консервативно требуем валидный прямоугольный контекст.
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

            # Не допускаем перекрытий и близких соседних дефектов.
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

    print(f"Создано дефектов: {len(records)} / {len(plan)}")
    for key, count in sorted(Counter(
        f"{r['type']}/{r['size_group']}" for r in records
    ).items()):
        print(f"  {key}: {count}")

    if failed:
        print("Не удалось разместить:", dict(failed))

    return labels, records, dict(failed)


def window_starts(length):
    starts = list(range(0, length - PATCH_SIZE + 1, STRIDE))
    last = length - PATCH_SIZE

    if starts[-1] != last:
        starts.append(last)

    return starts


@torch.inference_mode()
def reconstruct(model, observed, valid, labels, preprocessing, device):
    height, width = observed.shape

    # Все окна используют один и тот же исходный контекст.
    # Предсказания не превращаем в новые известные высоты.
    known = valid & (labels == 0)

    prediction_sum = np.zeros(observed.shape, dtype=np.float64)
    weight_sum = np.zeros(observed.shape, dtype=np.float32)

    axis = np.hanning(PATCH_SIZE)
    blend = np.maximum(
        np.outer(axis, axis),
        MIN_BLEND_WEIGHT,
    )

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

        for prediction, (row, col, offset, scale) in zip(
            predictions, batch_info
        ):
            region = np.s_[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE,
            ]

            missing = labels[region] > 0
            local_weights = blend * missing

            prediction_m = (
                prediction.astype(np.float64) * scale + offset
            )

            prediction_sum[region] += prediction_m * local_weights
            weight_sum[region] += local_weights.astype(np.float32)

        previous = processed
        processed += len(batch_inputs)

        if processed // 500 > previous // 500:
            print(f"Обработано окон: {processed}")

        batch_inputs.clear()
        batch_info.clear()

    minimum_scale = float(preprocessing["min_height_scale"])

    for row in window_starts(height):
        for col in window_starts(width):
            region = np.s_[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE,
            ]

            if not (labels[region] > 0).any():
                continue

            local_known = known[region]
            if local_known.mean() < MIN_KNOWN_FRACTION:
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
            batch_info.append((row, col, offset, scale))

            if len(batch_inputs) >= INFERENCE_BATCH_SIZE:
                flush()

    flush()

    filled = (labels > 0) & (weight_sum > 0)

    completed = observed.copy()
    completed[filled] = (
        prediction_sum[filled] / weight_sum[filled]
    )

    return completed, filled, {
        "processed_windows": processed,
        "skipped_windows": skipped,
    }


def write_geotiff(path, array, metadata, labels=False):
    profile = {
        "driver": "GTiff",
        "height": metadata["height"],
        "width": metadata["width"],
        "count": 1,
        "dtype": "uint32" if labels else "float64",
        "crs": metadata["crs"],
        "transform": metadata["transform"],
        "nodata": None if labels else np.nan,
        "compress": "deflate",
        "predictor": 2 if labels else 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
    }

    with rasterio.open(path, "w", **profile) as dst:
        for _, window in dst.block_windows(1):
            rows, cols = window.toslices()
            block = array[rows, cols]
            if labels:
                block = block.astype(np.uint32)
            dst.write(block, 1, window=window)


def error_metrics(values):
    if not values.size:
        return {
            "rmse_m": None,
            "mae_m": None,
            "bias_m": None,
            "max_absolute_error_m": None,
        }

    return {
        "rmse_m": float(np.sqrt(np.mean(values ** 2))),
        "mae_m": float(np.mean(np.abs(values))),
        "bias_m": float(np.mean(values)),
        "max_absolute_error_m": float(np.max(np.abs(values))),
    }


def build_reports(original, completed, labels, filled, records):
    per_defect = []

    for record in records:
        row, col = record["row"], record["col"]
        region = np.s_[
            row:row + record["bbox_height"],
            col:col + record["bbox_width"],
        ]

        local_defect = labels[region] == record["id"]
        local_filled = local_defect & filled[region]

        values = (
            completed[region][local_filled]
            - original[region][local_filled]
        )

        per_defect.append({
            **record,
            "filled_pixels": int(local_filled.sum()),
            "coverage": float(local_filled.sum() / record["pixels"]),
            **error_metrics(values),
        })

    grouped = []

    for kind in DEFECT_TYPES:
        for group in SIZE_GROUPS:
            selected = [
                r for r in records
                if r["type"] == kind and r["size_group"] == group
            ]
            if not selected:
                continue

            # Собираем ошибки только из локальных рамок дефектов.
            errors = []
            total_pixels = 0
            filled_pixels = 0

            for record in selected:
                row, col = record["row"], record["col"]
                region = np.s_[
                    row:row + record["bbox_height"],
                    col:col + record["bbox_width"],
                ]
                local = (
                    (labels[region] == record["id"])
                    & filled[region]
                )
                errors.append(
                    completed[region][local] - original[region][local]
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
                "coverage": filled_pixels / total_pixels,
                **error_metrics(values),
            })

    return per_defect, grouped


def save_figures(original, observed, completed, error, labels, records):
    step = max(
        1,
        int(np.ceil(max(original.shape) / OVERVIEW_MAX_SIDE)),
    )

    arrays = [
        original[::step, ::step],
        observed[::step, ::step],
        completed[::step, ::step],
    ]

    low = float(np.nanmin(arrays[0]))
    high = max(float(np.nanmax(arrays[0])), low + 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for axis, array, title in zip(
        axes, arrays, ["Original DEM", "With defects", "Completed DEM"]
    ):
        image = axis.imshow(
            array, cmap="terrain", vmin=low, vmax=high
        )
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(image, ax=axis, shrink=0.7)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "overview.png", dpi=180)
    plt.close(fig)

    # Берём примеры из разных типов и диапазонов размеров.
    candidates = []
    for kind in DEFECT_TYPES:
        for group in SIZE_GROUPS:
            matches = [
                r for r in records
                if r["type"] == kind and r["size_group"] == group
            ]
            if matches:
                candidates.append(matches[0])

    count = min(DETAIL_EXAMPLES, len(candidates))
    chosen_indices = np.linspace(
        0, len(candidates) - 1, count, dtype=int
    )
    examples = [candidates[i] for i in chosen_indices]

    fig, axes = plt.subplots(
        count, 4, figsize=(17, 3.2 * count), squeeze=False
    )

    for row_index, record in enumerate(examples):
        r0 = max(0, record["row"] - 12)
        c0 = max(0, record["col"] - 12)
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

        finite_result = result[np.isfinite(result)]
        lo = float(np.nanmin(source))
        hi = float(np.nanmax(source))

        if finite_result.size:
            lo = min(lo, float(finite_result.min()))
            hi = max(hi, float(finite_result.max()))

        hi = max(hi, lo + 1e-6)

        finite_error = local_error[np.isfinite(local_error)]
        limit = (
            max(float(np.abs(finite_error).max()), 0.01)
            if finite_error.size else 0.01
        )

        for col_index, (array, title) in enumerate(zip(
            [source, observed[region], result, local_error],
            ["Original", "Defect", "Completed", "Error, m"],
        )):
            axis = axes[row_index, col_index]
            is_error = col_index == 3

            image = axis.imshow(
                array,
                cmap="RdBu_r" if is_error else "terrain",
                vmin=-limit if is_error else lo,
                vmax=limit if is_error else hi,
            )
            axis.set_title(
                f"{record['type']}/{record['size_group']} | {title}",
                fontsize=9,
            )
            axis.axis("off")
            fig.colorbar(image, ax=axis, shrink=0.7)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "defect_examples.png", dpi=150)
    plt.close(fig)


def main():
    if PATCH_SIZE != 64 or not 1 <= STRIDE <= PATCH_SIZE:
        raise ValueError("Проверь PATCH_SIZE/STRIDE.")
    if not 0 < MIN_KNOWN_FRACTION < 1:
        raise ValueError("Некорректный MIN_KNOWN_FRACTION.")
    if not 0 < MIN_BLEND_WEIGHT <= 1:
        raise ValueError("Некорректный MIN_BLEND_WEIGHT.")
    if INFERENCE_BATCH_SIZE < 1 or DEFECTS_PER_GROUP < 1:
        raise ValueError("Некорректный размер batch/количество дефектов.")
    if DETAIL_EXAMPLES < 1:
        raise ValueError("DETAIL_EXAMPLES должен быть >= 1.")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = choose_device()
    model, checkpoint = load_model(device)

    original, valid, metadata = load_dem()

    print(f"Device: {device}")
    print(f"DEM: {original.shape}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")

    if metadata["crs"] is None:
        print("CRS отсутствует: сохраняем без назначения системы координат.")

    labels, records, failed = create_defects(valid)
    holes = labels > 0

    observed = original.copy()
    observed[holes] = np.nan

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    write_geotiff(
        OUTPUT_DIR / "dem_with_defects.tif",
        observed, metadata,
    )
    write_geotiff(
        OUTPUT_DIR / "defect_ids.tif",
        labels, metadata, labels=True,
    )

    completed, filled, statistics = reconstruct(
        model,
        observed,
        valid,
        labels,
        checkpoint["preprocessing"],
        device,
    )

    unchanged = valid & ~holes
    if not np.array_equal(completed[unchanged], original[unchanged]):
        raise RuntimeError("Известные высоты изменились.")
    if np.isfinite(completed[~valid]).any():
        raise RuntimeError("Исходные NoData неожиданно заполнены.")

    error = np.full(original.shape, np.nan, dtype=np.float64)
    error[filled] = completed[filled] - original[filled]

    write_geotiff(
        OUTPUT_DIR / "dem_filled.tif",
        completed, metadata,
    )
    write_geotiff(
        OUTPUT_DIR / "reconstruction_error.tif",
        error, metadata,
    )
    write_geotiff(
        OUTPUT_DIR / "filled_pixels.tif",
        filled, metadata, labels=True,
    )
    write_geotiff(
        OUTPUT_DIR / "unfilled_pixels.tif",
        holes & ~filled, metadata, labels=True,
    )

    per_defect, grouped = build_reports(
        original, completed, labels, filled, records
    )

    write_csv(OUTPUT_DIR / "defects.csv", per_defect)
    write_csv(OUTPUT_DIR / "metrics_by_type_and_size.csv", grouped)

    report = {
        "source": str(SOURCE_DEM),
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "device": str(device),
        "seed": SEED,
        "stochastic": STOCHASTIC,
        "shape": list(original.shape),
        "patch_size": PATCH_SIZE,
        "stride": STRIDE,
        "min_known_fraction": MIN_KNOWN_FRACTION,
        "created_defects": len(records),
        "unplaced_defects": failed,
        "hidden_pixels": int(holes.sum()),
        "filled_pixels": int(filled.sum()),
        "unfilled_pixels": int((holes & ~filled).sum()),
        "coverage": float(filled.sum() / holes.sum()),
        "known_pixels_unchanged": True,
        "original_nodata_preserved": True,
        "metrics_region": "successfully reconstructed artificial holes",
        "source_scale": metadata["source_scale"],
        "source_offset": metadata["source_offset"],
        "crs": (
            metadata["crs"].to_string()
            if metadata["crs"] is not None else None
        ),
        **statistics,
        **error_metrics(error[filled]),
    }

    write_json(OUTPUT_DIR / "report.json", report)

    save_figures(
        original, observed, completed, error, labels, records
    )

    print("\nГотово.")
    print(f"Скрыто пикселей: {report['hidden_pixels']:,}")
    print(f"Восстановлено: {report['filled_pixels']:,}")
    print(f"Покрытие: {100 * report['coverage']:.2f}%")

    if report["rmse_m"] is not None:
        print(f"RMSE: {report['rmse_m']:.3f} м")
        print(f"MAE: {report['mae_m']:.3f} м")

    if report["unfilled_pixels"]:
        print(
            "Внимание: часть пропусков не заполнена. "
            "Метрики относятся только к восстановленным пикселям."
        )

    print(f"Результаты: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
