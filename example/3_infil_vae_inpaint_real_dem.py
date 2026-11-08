from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

from tqdm.auto import tqdm

from scripts.vae_fill.train_vae import QuarryCVAE


# ============================= CONFIG =============================

ROOT = next(
    p for p in Path(__file__).resolve().parents
    if (p / "pyproject.toml").is_file()
)

SOURCE_DEM = ROOT / "data/derived/dem_real_photo_quarry_01.tif"
CHECKPOINT = ROOT / "models/cvae_inpaint/best_model.pt"
OUTPUT = ROOT / "outputs" / f"real_dem_fill_{datetime.now():%Y%m%d_%H%M%S_%f}"

SOURCE_BAND = 1

PATCH_SIZE = 64
STRIDE = 8
BATCH_SIZE = 16

DEVICE = "auto"
SEED = 42
STOCHASTIC = False

MIN_KNOWN_FRACTION = 0.40
BLEND_SIGMA_FRACTION = 0.25
MIN_BLEND_WEIGHT = 1e-3
CONTEXT_WEIGHT_POWER = 2.0

MAX_HOLE_PIXELS = None
EXTRA_NODATA = None

# ==================================================================


def load_dem():
    with rasterio.open(SOURCE_DEM) as src:
        band = src.read(SOURCE_BAND, masked=True, out_dtype="float64")
        raw = np.asarray(band.data)
        known = ~np.ma.getmaskarray(band) & np.isfinite(raw)

        if EXTRA_NODATA is not None:
            known &= raw != EXTRA_NODATA

        dem = (
            raw * src.scales[SOURCE_BAND - 1]
            + src.offsets[SOURCE_BAND - 1]
        )
        known &= np.isfinite(dem)
        dem[~known] = np.nan

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

    return dem, known, profile


def outer_domain(known):
    edge = known & ~ndimage.binary_erosion(
        known,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    rows, cols = np.nonzero(edge)
    centers = np.column_stack([cols + 0.5, rows + 0.5])

    if len(centers) >= 3:
        try:
            centers = centers[ConvexHull(centers).vertices]
        except QhullError:
            centers = centers[np.unique([
                centers[:, 0].argmin(),
                centers[:, 0].argmax(),
                centers[:, 1].argmin(),
                centers[:, 1].argmax(),
            ])]

    offsets = np.array([
        [-0.5, -0.5],
        [0.5, -0.5],
        [0.5, 0.5],
        [-0.5, 0.5],
    ])
    corners = np.unique(
        (centers[:, None] + offsets).reshape(-1, 2),
        axis=0,
    )
    contour = corners[ConvexHull(corners).vertices]
    ring = np.vstack([contour, contour[0]]).tolist()

    domain = rasterize(
        [({"type": "Polygon", "coordinates": [ring]}, 1)],
        out_shape=known.shape,
        transform=Affine.identity(),
        fill=0,
        dtype="uint8",
    ).astype(bool)

    return domain | known


def target_mask(known):
    target = outer_domain(known) & ~known

    if MAX_HOLE_PIXELS is not None:
        labels, count = ndimage.label(
            target,
            structure=np.ones((3, 3), dtype=bool),
        )
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        selected = sizes <= MAX_HOLE_PIXELS
        selected[0] = False
        target = selected[labels]

    return target


def starts(length):
    last = length - PATCH_SIZE
    return sorted(set(range(0, last + 1, STRIDE)) | {last})


def blend_weights():
    x = np.arange(PATCH_SIZE) - (PATCH_SIZE - 1) / 2
    axis = np.exp(-0.5 * (x / (PATCH_SIZE * BLEND_SIGMA_FRACTION)) ** 2)
    weights = np.outer(axis, axis)
    return np.maximum(weights / weights.max(), MIN_BLEND_WEIGHT)


@torch.inference_mode()
def reconstruct(model, observed, known, target, minimum_scale, device):
    total = np.zeros(observed.shape)
    weights = np.zeros(observed.shape)
    blend = blend_weights()
    inputs, positions = [], []
    processed = skipped = 0

    def flush():
        nonlocal processed
        if not inputs:
            return

        tensor = torch.from_numpy(np.stack(inputs)).to(device)
        predictions = model(
            tensor, stochastic=STOCHASTIC
        )[:, 0].cpu().numpy()

        for prediction, (row, col, offset, scale, context) in zip(
            predictions, positions
        ):
            region = np.s_[row:row + PATCH_SIZE, col:col + PATCH_SIZE]
            mask = target[region]
            values = prediction[mask].astype(np.float64) * scale + offset
            local_weights = blend[mask] * context

            total_view, weight_view = total[region], weights[region]
            total_view[mask] += values * local_weights
            weight_view[mask] += local_weights

        processed += len(inputs)
        inputs.clear()
        positions.clear()

    rows = starts(observed.shape[0])
    cols = starts(observed.shape[1])

    with tqdm(
        total=len(rows) * len(cols),
        desc="Обработка окон",
        unit="окно",
        dynamic_ncols=True,
    ) as progress:
        for row in rows:
            for col in cols:
                region = np.s_[row:row + PATCH_SIZE, col:col + PATCH_SIZE]

                if target[region].any():
                    mask = known[region]
                    fraction = float(mask.mean())

                    if mask.any() and fraction >= MIN_KNOWN_FRACTION:
                        visible = observed[region][mask]
                        offset = float(visible.min())
                        scale = max(
                            float(visible.max()) - offset,
                            minimum_scale,
                        )

                        normalized = np.zeros(
                            (PATCH_SIZE, PATCH_SIZE), dtype=np.float32
                        )
                        normalized[mask] = (visible - offset) / scale

                        inputs.append(np.stack([
                            normalized,
                            mask.astype(np.float32),
                        ]))
                        positions.append((
                            row, col, offset, scale,
                            fraction ** CONTEXT_WEIGHT_POWER,
                        ))

                        if len(inputs) >= BATCH_SIZE:
                            flush()
                    else:
                        skipped += 1

                progress.update(1)

                if progress.n % 500 == 0:
                    progress.set_postfix(
                        рассчитано=processed,
                        без_контекста=skipped,
                        refresh=False,
                    )

        flush()
        progress.set_postfix(
            рассчитано=processed,
            без_контекста=skipped,
        )

    filled = target & (weights > 0)
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

    original, known, profile = load_dem()
    if not known.any() or min(original.shape) < PATCH_SIZE:
        raise ValueError("DEM не содержит данных или меньше окна модели.")

    target = target_mask(known)

    if target.any():
        device = torch.device(
            DEVICE if DEVICE != "auto" else
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        model = QuarryCVAE(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()

        completed, filled = reconstruct(
            model, original, known, target,
            float(checkpoint["preprocessing"]["min_height_scale"]), device,
        )
    else:
        completed = original.copy()
        filled = np.zeros(original.shape, dtype=bool)

    restored_only = np.where(filled, completed, np.nan)

    OUTPUT.mkdir(parents=True, exist_ok=False)

    for name, array in (
        ("dem_filled.tif", completed),
        ("dem_original.tif", original),
        ("dem_restored_only.tif", restored_only),
    ):
        save_tif(name, array, profile)

    print(f"Восстановлено: {filled.sum():,}/{target.sum():,} пикселей")
    print(f"Сохранено: {OUTPUT}")


if __name__ == "__main__":
    main()
