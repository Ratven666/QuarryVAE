from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import torch

from scripts.base_vae.train_vae import QuarryCVAE


# ============================= CONFIG =============================

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)

SOURCE_DEM = (
    PROJECT_ROOT
    / "outputs/real_dem_contour_fill_20261107_200511_616752/dem_filled.tif"
)
CHECKPOINT_PATH = PROJECT_ROOT / "models/cvae_base/best_model.pt"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / f"full_dem_reconstruction_{datetime.now():%Y%m%d_%H%M%S_%f}"
)

SOURCE_BAND = 1

PATCH_SIZE = 64
RECONSTRUCTION_SIZE = 16
STRIDE = 8
BATCH_SIZE = 32

DEVICE = "auto"
SEED = 42
STOCHASTIC = False

MIN_KNOWN_FRACTION = 0.25
BLEND_SIGMA_FRACTION = 0.30
MIN_BLEND_WEIGHT = 0.05
CONTEXT_WEIGHT_POWER = 1.0

EXTRA_NODATA_VALUE = None
PRINT_EVERY = 1000

# ==================================================================


def load_dem():
    with rasterio.open(SOURCE_DEM) as src:
        band = src.read(SOURCE_BAND, masked=True, out_dtype="float64")
        raw = np.asarray(band.data)

        valid = ~np.ma.getmaskarray(band) & np.isfinite(raw)
        if EXTRA_NODATA_VALUE is not None:
            valid &= raw != EXTRA_NODATA_VALUE

        dem = (
            raw * src.scales[SOURCE_BAND - 1]
            + src.offsets[SOURCE_BAND - 1]
        )
        valid &= np.isfinite(dem)
        dem[~valid] = np.nan

        profile = {
            "driver": "GTiff",
            "height": src.height,
            "width": src.width,
            "count": 1,
            "dtype": "float64",
            "crs": src.crs,
            "transform": src.transform,
            "nodata": np.nan,
            "compress": "deflate",
            "predictor": 3,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "BIGTIFF": "IF_SAFER",
        }

    return dem, valid, profile


def starts(length):
    last = max(0, length - RECONSTRUCTION_SIZE)
    return sorted(set(range(0, last + 1, STRIDE)) | {last})


def blend_weights():
    x = np.arange(RECONSTRUCTION_SIZE) - (RECONSTRUCTION_SIZE - 1) / 2
    axis = np.exp(
        -0.5 * (x / (RECONSTRUCTION_SIZE * BLEND_SIGMA_FRACTION)) ** 2
    )
    weights = np.outer(axis, axis)
    return np.maximum(weights / weights.max(), MIN_BLEND_WEIGHT)


@torch.inference_mode()
def reconstruct(model, dem, valid, minimum_scale, device):
    height, width = dem.shape
    size = RECONSTRUCTION_SIZE
    halo = (PATCH_SIZE - size) // 2

    padding = (
        (halo, halo + max(0, size - height)),
        (halo, halo + max(0, size - width)),
    )
    padded_dem = np.pad(dem, padding, constant_values=np.nan)
    padded_valid = np.pad(valid, padding, constant_values=False)

    total = np.zeros(dem.shape, dtype=np.float64)
    weights = np.zeros(dem.shape, dtype=np.float64)
    blend = blend_weights()

    inputs, positions = [], []
    processed = 0

    def flush():
        nonlocal processed
        if not inputs:
            return

        tensor = torch.from_numpy(np.stack(inputs)).to(device)
        predictions = model(
            tensor, stochastic=STOCHASTIC
        )[:, 0].cpu().numpy()

        for prediction, (row, col, h, w, offset, scale, context) in zip(
            predictions, positions
        ):
            region = np.s_[row:row + h, col:col + w]
            mask = valid[region]

            values = prediction[halo:halo + h, halo:halo + w]
            values = values[mask].astype(np.float64) * scale + offset
            local_weights = blend[:h, :w][mask] * context

            total_view = total[region]
            weight_view = weights[region]
            total_view[mask] += values * local_weights
            weight_view[mask] += local_weights

        previous = processed
        processed += len(inputs)
        if processed // PRINT_EVERY > previous // PRINT_EVERY:
            print(f"Обработано окон: {processed:,}")

        inputs.clear()
        positions.clear()

    for row in starts(height):
        for col in starts(width):
            h, w = min(size, height - row), min(size, width - col)

            if not valid[row:row + h, col:col + w].any():
                continue

            patch = padded_dem[row:row + PATCH_SIZE, col:col + PATCH_SIZE]
            known = padded_valid[
                row:row + PATCH_SIZE, col:col + PATCH_SIZE
            ].copy()

            # Скрываем предсказываемый блок.
            known[halo:halo + size, halo:halo + size] = False
            fraction = float(known.mean())

            if fraction < MIN_KNOWN_FRACTION or not known.any():
                continue

            visible = patch[known]
            offset = float(visible.min())
            scale = max(float(visible.max()) - offset, minimum_scale)

            normalized = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
            normalized[known] = (visible - offset) / scale

            inputs.append(np.stack([normalized, known.astype(np.float32)]))
            positions.append((
                row, col, h, w, offset, scale,
                fraction ** CONTEXT_WEIGHT_POWER,
            ))

            if len(inputs) >= BATCH_SIZE:
                flush()

    flush()

    predicted = valid & (weights > 0)
    result = np.full(dem.shape, np.nan)
    result[predicted] = total[predicted] / weights[predicted]

    print(f"Реконструировано: {predicted.sum():,} / {valid.sum():,}")
    return result


def save_tif(path, array, profile):
    with rasterio.open(path, "w", **profile) as dst:
        for _, window in dst.block_windows(1):
            rows, cols = window.toslices()
            dst.write(array[rows, cols], 1, window=window)


def main():
    torch.manual_seed(SEED)

    device = DEVICE
    if device == "auto":
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    device = torch.device(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH, map_location="cpu", weights_only=True
    )
    model = QuarryCVAE(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    dem, valid, profile = load_dem()
    print(f"Устройство: {device}; DEM: {dem.shape}")

    rebuilt = reconstruct(
        model,
        dem,
        valid,
        float(checkpoint["preprocessing"]["min_height_scale"]),
        device,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    for filename, array in (
        ("dem_reconstructed.tif", rebuilt),
        ("dem_original.tif", dem),
        ("reconstruction_difference.tif", rebuilt - dem),
    ):
        save_tif(OUTPUT_DIR / filename, array, profile)

    print(f"Сохранено: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
