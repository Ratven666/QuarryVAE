from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import torch

from scripts.vae_fill.train_vae import QuarryCVAE


# ============================= CONFIG =============================

ROOT = next(
    p for p in Path(__file__).resolve().parents
    if (p / "pyproject.toml").is_file()
)

SOURCE_DEM = ROOT / "data/raw/grib_1m.tif"
CHECKPOINT = ROOT / "models/cvae_inpaint/best_model.pt"
OUTPUT = ROOT / "outputs" / f"dem_fill_{datetime.now():%Y%m%d_%H%M%S_%f}"

PATCH_SIZE = 64
STRIDE = 16
BATCH_SIZE = 16

DEVICE = "auto"
SEED = 42
STOCHASTIC = False

MIN_KNOWN_FRACTION = 0.40
MIN_BLEND_WEIGHT = 0.05
EXTRA_NODATA = None

DEFECTS_PER_GROUP = 20
DEFECT_TYPES = ("square", "circle", "ellipse", "stripe", "capsule")
SIZE_GROUPS = ("small", "medium", "large")

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
CONTEXT_MARGIN = 8
DEFECT_SEPARATION = 4
ATTEMPTS_PER_DEFECT = 500

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


def make_template(rng, kind, group):
    angle = rng.uniform(0, np.pi)

    if kind == "square":
        length = width = rng.uniform(*SQUARE_SIZES[group])
    elif kind == "circle":
        length = width = rng.uniform(*CIRCLE_DIAMETERS[group])
        angle = 0.0
    elif kind == "ellipse":
        major, minor = ELLIPSE_AXES[group]
        length, width = rng.uniform(*major), rng.uniform(*minor)
    else:
        long_bounds, width_bounds = STRIPE_DIMENSIONS[group]
        length = rng.uniform(*long_bounds)
        width = rng.uniform(*width_bounds)

    cosine, sine = np.cos(angle), np.sin(angle)
    rx = int(np.ceil(abs(cosine) * length / 2 + abs(sine) * width / 2)) + 1
    ry = int(np.ceil(abs(sine) * length / 2 + abs(cosine) * width / 2)) + 1

    yy, xx = np.mgrid[-ry:ry + 1, -rx:rx + 1]
    u, v = xx * cosine + yy * sine, -xx * sine + yy * cosine

    if kind in {"square", "stripe"}:
        mask = (np.abs(u) <= length / 2) & (np.abs(v) <= width / 2)
    elif kind in {"circle", "ellipse"}:
        mask = (u / (length / 2)) ** 2 + (v / (width / 2)) ** 2 <= 1
    else:
        distance = np.maximum(np.abs(u) - max((length - width) / 2, 0), 0)
        mask = distance ** 2 + v ** 2 <= (width / 2) ** 2

    rows, cols = np.nonzero(mask)
    return mask[
        rows.min():rows.max() + 1,
        cols.min():cols.max() + 1,
    ]


def generate_holes(valid):
    rng = np.random.default_rng(SEED)
    height, width = valid.shape
    holes = np.zeros(valid.shape, dtype=bool)

    plan = [
        (kind, group)
        for kind in DEFECT_TYPES
        for group in SIZE_GROUPS
        for _ in range(DEFECTS_PER_GROUP)
    ]
    rng.shuffle(plan)
    created = Counter()

    for kind, group in plan:
        for _ in range(ATTEMPTS_PER_DEFECT):
            mask = make_template(rng, kind, group)
            h, w = mask.shape
            margin = max(EDGE_MARGIN, CONTEXT_MARGIN)
            max_row, max_col = height - margin - h, width - margin - w

            if min(max_row, max_col) < margin:
                continue

            row = int(rng.integers(margin, max_row + 1))
            col = int(rng.integers(margin, max_col + 1))
            r1, c1 = row + h, col + w

            if not valid[
                row - CONTEXT_MARGIN:r1 + CONTEXT_MARGIN,
                col - CONTEXT_MARGIN:c1 + CONTEXT_MARGIN,
            ].all():
                continue

            if holes[
                max(0, row - DEFECT_SEPARATION):min(height, r1 + DEFECT_SEPARATION),
                max(0, col - DEFECT_SEPARATION):min(width, c1 + DEFECT_SEPARATION),
            ].any():
                continue

            holes[row:r1, col:c1] |= mask
            created[kind, group] += 1
            break

    if not holes.any():
        raise RuntimeError("Не удалось разместить дефекты.")

    print(f"Создано дефектов: {sum(created.values())}/{len(plan)}")
    for kind in DEFECT_TYPES:
        for group in SIZE_GROUPS:
            print(f"  {kind:8s} {group:6s}: {created[kind, group]}")

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
            values = prediction[mask].astype(np.float64) * scale + offset
            local_weights = blend[mask]

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
