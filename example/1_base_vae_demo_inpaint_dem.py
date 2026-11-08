from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import torch

from scripts.base_vae.train_vae import QuarryCVAE


# ============================= CONFIG =============================

ROOT = next(
    p for p in Path(__file__).resolve().parents
    if (p / "pyproject.toml").is_file()
)

SOURCE_DEM = ROOT / "data/raw/grib_1m.tif"
CHECKPOINT = ROOT / "models/cvae_base/best_model.pt"
OUTPUT = ROOT / "outputs" / f"dem_fill_{datetime.now():%Y%m%d_%H%M%S_%f}"

PATCH_SIZE = 64
STRIDE = 16
BATCH_SIZE = 32

NUMBER_OF_HOLES = 100

HOLE_TYPES = (
    "circle",
    "ellipse",
    "square",
    "rectangle",
    "stripe",
    "crack",
    "blob",
)

# Диапазоны характерного размера в пикселях.
HOLE_SIZES = (
    (5, 12),
    (13, 28),
    (29, 56),
)

CONTEXT_MARGIN = 16
HOLE_SEPARATION = 8
ATTEMPTS_PER_HOLE = 1000

MIN_KNOWN_FRACTION = 0.50
MIN_BLEND_WEIGHT = 0.05

DEVICE = "auto"
SEED = 42
STOCHASTIC = False
EXTRA_NODATA = None

# ==================================================================


def load_dem():
    with rasterio.open(SOURCE_DEM) as src:
        band = src.read(1, masked=True, out_dtype="float64")
        raw = np.asarray(band.data)
        valid = ~np.ma.getmaskarray(band) & np.isfinite(raw)

        if EXTRA_NODATA is not None:
            valid &= raw != EXTRA_NODATA

        dem = raw * src.scales[0] + src.offsets[0]
        valid &= np.isfinite(dem)
        dem[~valid] = np.nan

        profile = dict(
            driver="GTiff",
            height=src.height,
            width=src.width,
            count=1,
            dtype="float64",
            crs=src.crs,
            transform=src.transform,
            nodata=np.nan,
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="IF_SAFER",
        )

    return dem, valid, profile


def make_hole(rng, kind, size_range):
    size = float(rng.uniform(*size_range))
    angle = rng.uniform(0, 2 * np.pi)

    if kind in {"circle", "square"}:
        length = width = size
    elif kind in {"ellipse", "rectangle"}:
        length = size * rng.uniform(1.2, 2.5)
        width = size * rng.uniform(0.4, 0.9)
    elif kind in {"stripe", "crack"}:
        length = size * rng.uniform(2.0, 4.0)
        width = max(2.0, size * rng.uniform(0.10, 0.25))
    else:
        length = size * rng.uniform(0.8, 1.5)
        width = size * rng.uniform(0.7, 1.3)

    # Запас для поворота, извилистости и неровных границ.
    radius = int(np.ceil(np.hypot(length, width))) + 3
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]

    u = xx * np.cos(angle) + yy * np.sin(angle)
    v = -xx * np.sin(angle) + yy * np.cos(angle)

    if kind in {"circle", "ellipse"}:
        mask = (u / (length / 2)) ** 2 + (v / (width / 2)) ** 2 <= 1

    elif kind in {"square", "rectangle"}:
        mask = (np.abs(u) <= length / 2) & (np.abs(v) <= width / 2)

    elif kind == "stripe":
        # Полоса со скруглёнными концами.
        along = np.maximum(np.abs(u) - max((length - width) / 2, 0), 0)
        mask = along ** 2 + v ** 2 <= (width / 2) ** 2

    elif kind == "crack":
        phase = rng.uniform(0, 2 * np.pi)
        wave = (
            width * rng.uniform(0.8, 2.0)
            * np.sin(u / length * rng.uniform(2, 4) * np.pi + phase)
        )
        thickness = width / 2 * (
            1 + 0.25 * np.sin(5 * u / length * np.pi)
        )
        mask = (
            (np.abs(u) <= length / 2)
            & (np.abs(v - wave) <= thickness)
        )

    else:
        # Нерегулярное пятно с плавно меняющейся радиальной границей.
        theta = np.arctan2(v / (width / 2), u / (length / 2))
        distance = np.hypot(u / (length / 2), v / (width / 2))

        boundary = (
            1
            + rng.uniform(0.10, 0.22) * np.sin(
                3 * theta + rng.uniform(0, 2 * np.pi)
            )
            + rng.uniform(0.05, 0.15) * np.sin(
                5 * theta + rng.uniform(0, 2 * np.pi)
            )
        )
        mask = distance <= boundary

    rows, cols = np.nonzero(mask)
    return mask[
        rows.min():rows.max() + 1,
        cols.min():cols.max() + 1,
    ]


def generate_holes(valid):
    rng = np.random.default_rng(SEED)
    height, width = valid.shape
    holes = np.zeros(valid.shape, dtype=bool)

    # Типы и диапазоны размеров представлены примерно поровну.
    combinations = [
        (kind, size_index)
        for kind in HOLE_TYPES
        for size_index in range(len(HOLE_SIZES))
    ]
    plan = []

    while len(plan) < NUMBER_OF_HOLES:
        order = rng.permutation(len(combinations))
        plan.extend(combinations[i] for i in order)

    plan = plan[:NUMBER_OF_HOLES]
    rng.shuffle(plan)

    created = {}

    for kind, size_index in plan:
        placed = False

        for _ in range(ATTEMPTS_PER_HOLE):
            mask = make_hole(rng, kind, HOLE_SIZES[size_index])
            h, w = mask.shape

            max_row = height - CONTEXT_MARGIN - h
            max_col = width - CONTEXT_MARGIN - w

            if min(max_row, max_col) < CONTEXT_MARGIN:
                continue

            row = int(rng.integers(CONTEXT_MARGIN, max_row + 1))
            col = int(rng.integers(CONTEXT_MARGIN, max_col + 1))
            r1, c1 = row + h, col + w

            if not valid[
                row - CONTEXT_MARGIN:r1 + CONTEXT_MARGIN,
                col - CONTEXT_MARGIN:c1 + CONTEXT_MARGIN,
            ].all():
                continue

            if holes[
                max(0, row - HOLE_SEPARATION):min(height, r1 + HOLE_SEPARATION),
                max(0, col - HOLE_SEPARATION):min(width, c1 + HOLE_SEPARATION),
            ].any():
                continue

            holes[row:r1, col:c1] |= mask
            key = (kind, size_index)
            created[key] = created.get(key, 0) + 1
            placed = True
            break

        if not placed:
            print(f"Не размещён: {kind}, диапазон {HOLE_SIZES[size_index]}")

    if not holes.any():
        raise RuntimeError("Не удалось разместить пропуски.")

    print(f"Создано пропусков: {sum(created.values())}/{NUMBER_OF_HOLES}")
    for (kind, size_index), count in sorted(created.items()):
        print(f"  {kind:10s} {str(HOLE_SIZES[size_index]):10s}: {count}")

    return holes


def starts(length):
    last = length - PATCH_SIZE
    return sorted(set(range(0, last + 1, STRIDE)) | {last})


@torch.inference_mode()
def reconstruct(model, observed, known, holes, minimum_scale, device):
    total = np.zeros(observed.shape)
    weights = np.zeros(observed.shape)
    axis = np.hanning(PATCH_SIZE)
    blend = np.maximum(np.outer(axis, axis), MIN_BLEND_WEIGHT)
    inputs, positions = [], []

    def flush():
        if not inputs:
            return

        tensor = torch.from_numpy(np.stack(inputs)).to(device)
        predictions = model(tensor, stochastic=STOCHASTIC)[:, 0].cpu().numpy()

        for prediction, (row, col, offset, scale) in zip(predictions, positions):
            region = np.s_[row:row + PATCH_SIZE, col:col + PATCH_SIZE]
            mask = holes[region]
            local_weights = blend[mask]
            values = prediction[mask].astype(np.float64) * scale + offset

            total_view, weight_view = total[region], weights[region]
            total_view[mask] += values * local_weights
            weight_view[mask] += local_weights

        inputs.clear()
        positions.clear()

    for row in starts(observed.shape[0]):
        for col in starts(observed.shape[1]):
            region = np.s_[row:row + PATCH_SIZE, col:col + PATCH_SIZE]
            mask = known[region]

            if (
                not holes[region].any()
                or not mask.any()
                or mask.mean() < MIN_KNOWN_FRACTION
            ):
                continue

            visible = observed[region][mask]
            offset = float(visible.min())
            scale = max(float(visible.max()) - offset, minimum_scale)

            normalized = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
            normalized[mask] = (visible - offset) / scale

            inputs.append(np.stack([normalized, mask.astype(np.float32)]))
            positions.append((row, col, offset, scale))

            if len(inputs) >= BATCH_SIZE:
                flush()

    flush()

    filled = holes & (weights > 0)
    completed = observed.copy()
    completed[filled] = total[filled] / weights[filled]
    return completed, filled


def save_tif(name, array, profile):
    with rasterio.open(OUTPUT / name, "w", **profile) as dst:
        for _, window in dst.block_windows(1):
            rows, cols = window.toslices()
            dst.write(array[rows, cols], 1, window=window)


def main():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(
        DEVICE if DEVICE != "auto" else
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = QuarryCVAE(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    original, valid, profile = load_dem()
    if min(original.shape) < PATCH_SIZE:
        raise ValueError("DEM меньше окна модели.")

    holes = generate_holes(valid)
    observed = original.copy()
    observed[holes] = np.nan

    completed, filled = reconstruct(
        model, observed, valid & ~holes, holes,
        float(checkpoint["preprocessing"]["min_height_scale"]), device,
    )

    difference = completed - original
    fill_error = np.where(filled, difference, np.nan)
    filled_only = np.where(filled, completed, np.nan)

    OUTPUT.mkdir(parents=True, exist_ok=False)

    for name, array in (
        ("dem_reconstructed.tif", completed),
        ("dem_original.tif", original),
        ("reconstruction_difference.tif", difference),
        ("fill_error.tif", fill_error),
        ("dem_filled_only.tif", filled_only),
    ):
        save_tif(name, array, profile)

    print(f"Восстановлено: {filled.sum():,}/{holes.sum():,} пикселей")
    if filled.any():
        errors = fill_error[filled]
        print(f"RMSE: {np.sqrt(np.mean(errors ** 2)):.3f}")
        print(f"MAE: {np.mean(np.abs(errors)):.3f}")
    print(f"Сохранено: {OUTPUT}")


if __name__ == "__main__":
    main()