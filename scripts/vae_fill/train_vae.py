from __future__ import annotations

import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


# ============================= CONFIG =============================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/processed/patches_64_multi_d4"

RUN_NAME = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / f"cvae_inpaint_{RUN_NAME}"

PATCH_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 250

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5

BASE_CHANNELS = 32
LATENT_DIM = 32

MIN_HEIGHT_SCALE = 1.0

# Внутренняя полоса вдоль границы пропуска, пиксели.
BOUNDARY_WIDTH = 3

# Функции потерь в локально нормализованных высотах.
HEIGHT_L1_WEIGHT = 1.0
HEIGHT_MSE_WEIGHT = 0.5
BOUNDARY_HEIGHT_WEIGHT = 0.5
SEAM_WEIGHT = 1.0
GRADIENT_WEIGHT = 0.5
LAPLACIAN_WEIGHT = 0.1

GRADIENT_STEPS = (1, 2, 4)

# Дополнительное обучение режима, используемого при инференсе.
PRIOR_RECONSTRUCTION_WEIGHT = 1.0

KL_MAX_WEIGHT = 1e-3
KL_WARMUP_EPOCHS = 40

GRADIENT_CLIP = 5.0
EARLY_STOPPING_PATIENCE = 35

# Критерий выбора лучшей модели:
# RMSE высот + вес * MAE перепадов через границу
#             + вес * MAE соседних перепадов внутри/около пропуска.
#
# Все три величины здесь в метрах:
# градиентные метрики — ошибки ПЕРЕПАДОВ, не углы уклона.
SELECTION_SEAM_WEIGHT = 0.5
SELECTION_GRADIENT_WEIGHT = 0.25

MIN_HOLE_SIZE = 8
MAX_HOLE_SIZE = 28
MIN_HOLE_FRACTION = 0.03
MAX_HOLE_FRACTION = 0.35
MAX_MASK_SHAPES = 3
MASK_GENERATION_ATTEMPTS = 100

# Добавляем узкие линейные пропуски к прямоугольникам и эллипсам.
ENABLE_LINE_MASKS = True

SEED = 42
NUM_WORKERS = 0
DEVICE = "auto"

MAX_TRAIN_SAMPLES = None
EXAMPLE_COUNT = 4

# ==================================================================


def validate_settings():
    if PATCH_SIZE != 64:
        raise ValueError("Ожидается PATCH_SIZE=64.")
    if BASE_CHANNELS <= 0 or BASE_CHANNELS % 8:
        raise ValueError("BASE_CHANNELS должен быть кратен 8.")
    if LATENT_DIM < 1 or BATCH_SIZE < 1 or EPOCHS < 1:
        raise ValueError("Некорректные параметры обучения.")
    if MIN_HEIGHT_SCALE <= 0:
        raise ValueError("MIN_HEIGHT_SCALE должен быть > 0.")
    if not 0 < MIN_HOLE_SIZE <= MAX_HOLE_SIZE < PATCH_SIZE:
        raise ValueError("Некорректные размеры пропусков.")
    if not 0 < MIN_HOLE_FRACTION <= MAX_HOLE_FRACTION < 1:
        raise ValueError("Некорректные доли пропусков.")
    if BOUNDARY_WIDTH < 1:
        raise ValueError("BOUNDARY_WIDTH должен быть >= 1.")
    if not GRADIENT_STEPS or any(
        not 1 <= step < PATCH_SIZE for step in GRADIENT_STEPS
    ):
        raise ValueError("Некорректные GRADIENT_STEPS.")
    if MAX_MASK_SHAPES < 1 or MASK_GENERATION_ATTEMPTS < 1:
        raise ValueError("Некорректные параметры масок.")
    if MAX_TRAIN_SAMPLES is not None and MAX_TRAIN_SAMPLES < 1:
        raise ValueError("MAX_TRAIN_SAMPLES должен быть None или >= 1.")
    if NUM_WORKERS < 0 or EXAMPLE_COUNT < 1:
        raise ValueError("Некорректные параметры загрузки/визуализации.")
    if EARLY_STOPPING_PATIENCE < 1:
        raise ValueError("Некорректный EARLY_STOPPING_PATIENCE.")


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def choose_device():
    if DEVICE == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if DEVICE not in {"cpu", "cuda", "mps"}:
        raise ValueError("DEVICE: auto/cpu/cuda/mps.")
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна.")
    if DEVICE == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS недоступен.")
    return torch.device(DEVICE)


def write_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def make_hole_mask(rng):
    yy, xx = np.mgrid[:PATCH_SIZE, :PATCH_SIZE]

    for _ in range(MASK_GENERATION_ATTEMPTS):
        hole = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=bool)

        for _ in range(int(rng.integers(1, MAX_MASK_SHAPES + 1))):
            shape_type = int(rng.integers(0, 3 if ENABLE_LINE_MASKS else 2))

            height = int(rng.integers(MIN_HOLE_SIZE, MAX_HOLE_SIZE + 1))
            width = int(rng.integers(MIN_HOLE_SIZE, MAX_HOLE_SIZE + 1))
            row = int(rng.integers(0, PATCH_SIZE - height + 1))
            col = int(rng.integers(0, PATCH_SIZE - width + 1))

            if shape_type == 0:
                hole[row:row + height, col:col + width] = True

            elif shape_type == 1:
                cy = row + (height - 1) / 2
                cx = col + (width - 1) / 2
                hole |= (
                    ((yy - cy) / (height / 2)) ** 2
                    + ((xx - cx) / (width / 2)) ** 2
                ) <= 1

            else:
                # Капсула вокруг произвольно ориентированного отрезка.
                x0, y0 = rng.uniform(0, PATCH_SIZE - 1, size=2)
                angle = rng.uniform(0, 2 * np.pi)
                length = rng.uniform(MIN_HOLE_SIZE, MAX_HOLE_SIZE)
                radius = rng.uniform(2, 4)

                x1 = np.clip(x0 + length * np.cos(angle), 0, PATCH_SIZE - 1)
                y1 = np.clip(y0 + length * np.sin(angle), 0, PATCH_SIZE - 1)

                dx, dy = x1 - x0, y1 - y0
                denominator = max(dx * dx + dy * dy, 1e-8)
                t = np.clip(
                    ((xx - x0) * dx + (yy - y0) * dy) / denominator,
                    0,
                    1,
                )
                hole |= (
                    (xx - x0 - t * dx) ** 2
                    + (yy - y0 - t * dy) ** 2
                ) <= radius ** 2

        if MIN_HOLE_FRACTION <= hole.mean() <= MAX_HOLE_FRACTION:
            return hole

    raise RuntimeError("Не удалось сформировать маску пропуска.")


class DEMDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.epoch = 0
        self.files = sorted((DATA_DIR / split).glob("*.npy"))

        if split == "train" and MAX_TRAIN_SAMPLES is not None:
            rng = np.random.default_rng(SEED)
            indices = rng.permutation(len(self.files))[:MAX_TRAIN_SAMPLES]
            self.files = [self.files[int(i)] for i in indices]

        if not self.files:
            raise FileNotFoundError(f"Нет .npy в {DATA_DIR / split}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        dem = np.asarray(
            np.load(self.files[index], allow_pickle=False),
            dtype=np.float64,
        )

        if dem.shape != (PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"Неверный размер: {self.files[index]}")
        if not np.isfinite(dem).all():
            raise ValueError(f"NaN/Inf: {self.files[index]}")

        split_code = {"train": 0, "val": 1, "test": 2}[self.split]
        epoch = self.epoch if self.split == "train" else 0
        rng = np.random.default_rng(
            np.random.SeedSequence([SEED, split_code, epoch, index])
        )

        hole = make_hole_mask(rng)
        known = ~hole

        offset = float(dem[known].min())
        scale = max(
            float(dem[known].max()) - offset,
            MIN_HEIGHT_SCALE,
        )

        target = ((dem - offset) / scale).astype(np.float32)
        corrupted = np.where(known, target, 0).astype(np.float32)

        if not np.isfinite(target).all():
            raise ValueError(f"Ошибка нормализации: {self.files[index]}")

        return {
            "condition": torch.from_numpy(
                np.stack([corrupted, known.astype(np.float32)])
            ),
            "target": torch.from_numpy(target[None]),
            "hole": torch.from_numpy(hole.astype(np.float32)[None]),
            "offset": torch.tensor(offset, dtype=torch.float32),
            "scale": torch.tensor(scale, dtype=torch.float32),
        }


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.GroupNorm(8, out_channels),
        nn.SiLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.GroupNorm(8, out_channels),
        nn.SiLU(),
    )


class Encoder(nn.Module):
    def __init__(self, in_channels, base):
        super().__init__()
        self.block1 = conv_block(in_channels, base)
        self.block2 = conv_block(base, base * 2)
        self.block3 = conv_block(base * 2, base * 4)
        self.block4 = conv_block(base * 4, base * 8)

    def forward(self, x):
        f1 = self.block1(x)
        f2 = self.block2(F.avg_pool2d(f1, 2))
        f3 = self.block3(F.avg_pool2d(f2, 2))
        f4 = self.block4(F.avg_pool2d(f3, 2))
        return [f1, f2, f3, f4]


class GaussianHead(nn.Module):
    def __init__(self, channels, latent_dim):
        super().__init__()
        self.head = nn.Linear(channels, latent_dim * 2)

    def forward(self, features):
        mu, logvar = self.head(features.mean(dim=(2, 3))).chunk(2, dim=1)
        return mu, logvar.clamp(-8, 8)


class QuarryCVAE(nn.Module):
    def __init__(
        self,
        base_channels=BASE_CHANNELS,
        latent_dim=LATENT_DIM,
    ):
        super().__init__()
        b = base_channels

        self.context_encoder = Encoder(2, b)
        self.posterior_encoder = Encoder(3, b)

        self.prior_head = GaussianHead(b * 8, latent_dim)
        self.posterior_head = GaussianHead(b * 8, latent_dim)

        self.bottleneck = conv_block(b * 8 + latent_dim, b * 8)
        self.up3 = conv_block(b * 8 + b * 4, b * 4)
        self.up2 = conv_block(b * 4 + b * 2, b * 2)
        self.up1 = conv_block(b * 2 + b, b)
        self.output = nn.Conv2d(b, 1, 1)

    @staticmethod
    def sample(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, features, z):
        f1, f2, f3, f4 = features

        spatial_z = z[:, :, None, None].expand(
            -1, -1, f4.shape[2], f4.shape[3]
        )
        x = self.bottleneck(torch.cat([f4, spatial_z], dim=1))

        for block, skip in (
            (self.up3, f3),
            (self.up2, f2),
            (self.up1, f1),
        ):
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            x = block(torch.cat([x, skip], dim=1))

        return self.output(x)

    def forward(
        self,
        condition,
        target=None,
        stochastic=False,
        return_prior=False,
    ):
        features = self.context_encoder(condition)
        p_mu, p_logvar = self.prior_head(features[-1])

        if target is None:
            z = self.sample(p_mu, p_logvar) if stochastic else p_mu
            return self.decode(features, z)

        posterior_features = self.posterior_encoder(
            torch.cat([condition, target], dim=1)
        )
        q_mu, q_logvar = self.posterior_head(posterior_features[-1])

        posterior_prediction = self.decode(
            features, self.sample(q_mu, q_logvar)
        )
        parameters = (q_mu, q_logvar, p_mu, p_logvar)

        if return_prior:
            prior_prediction = self.decode(features, p_mu)
            return posterior_prediction, parameters, prior_prediction

        return posterior_prediction, parameters


def masked_mean(values, mask):
    """Равный вес образцов, независимо от площади пропуска."""
    dimensions = tuple(range(1, values.ndim))
    numerator = (values * mask).sum(dim=dimensions)
    denominator = mask.sum(dim=dimensions).clamp_min(1)
    return (numerator / denominator).mean()


def conditional_kl(parameters):
    q_mu, q_logvar, p_mu, p_logvar = parameters
    return (
        0.5 * (
            p_logvar - q_logvar
            + torch.exp(q_logvar - p_logvar)
            + (q_mu - p_mu).square() * torch.exp(-p_logvar)
            - 1
        )
    ).mean()


def boundary_band(hole):
    # Erosion пропуска; пиксели у края растра не считаются
    # границей с известной областью за пределами изображения.
    kernel = 2 * BOUNDARY_WIDTH + 1
    nearby_known = F.max_pool2d(
        1 - hole,
        kernel_size=kernel,
        stride=1,
        padding=BOUNDARY_WIDTH,
    )
    return hole * nearby_known


def differences(x, step=1):
    dx = (x[..., step:] - x[..., :-step]) / step
    dy = (x[..., step:, :] - x[..., :-step, :]) / step
    return dx, dy


def affected_edge_masks(hole, step=1):
    # Ребро затронуто, если хотя бы один его конец внутри пропуска.
    return (
        torch.maximum(hole[..., step:], hole[..., :-step]),
        torch.maximum(hole[..., step:, :], hole[..., :-step, :]),
    )


def seam_masks(hole):
    # Только соседние пары через границу known / missing.
    return (
        (hole[..., 1:] - hole[..., :-1]).abs(),
        (hole[..., 1:, :] - hole[..., :-1, :]).abs(),
    )


def laplacian(x):
    # Без padding: исключаем искусственные условия на краях фрагмента.
    return (
        x[..., 1:-1, 2:]
        + x[..., 1:-1, :-2]
        + x[..., 2:, 1:-1]
        + x[..., :-2, 1:-1]
        - 4 * x[..., 1:-1, 1:-1]
    )


def inpainting_loss(prediction, target, hole):
    completed = prediction * hole + target * (1 - hole)
    error = completed - target

    height_l1 = masked_mean(error.abs(), hole)
    height_mse = masked_mean(error.square(), hole)
    boundary = masked_mean(error.abs(), boundary_band(hole))

    pred_dx, pred_dy = differences(completed)
    true_dx, true_dy = differences(target)

    seam_x, seam_y = seam_masks(hole)

    seam = 0.5 * (
        masked_mean((pred_dx - true_dx).abs(), seam_x)
        + masked_mean((pred_dy - true_dy).abs(), seam_y)
    )

    gradient = prediction.new_zeros(())

    for step in GRADIENT_STEPS:
        pdx, pdy = differences(completed, step)
        tdx, tdy = differences(target, step)
        mx, my = affected_edge_masks(hole, step)

        gradient = gradient + 0.5 * (
            masked_mean((pdx - tdx).abs(), mx)
            + masked_mean((pdy - tdy).abs(), my)
        )

    gradient = gradient / len(GRADIENT_STEPS)

    # Сравниваем кривизну с истинной, не минимизируем её до нуля.
    lap_mask = F.max_pool2d(hole, 3, stride=1)[..., :, :]
    curvature = masked_mean(
        (laplacian(completed) - laplacian(target)).abs(),
        lap_mask,
    )

    total = (
        HEIGHT_L1_WEIGHT * height_l1
        + HEIGHT_MSE_WEIGHT * height_mse
        + BOUNDARY_HEIGHT_WEIGHT * boundary
        + SEAM_WEIGHT * seam
        + GRADIENT_WEIGHT * gradient
        + LAPLACIAN_WEIGHT * curvature
    )

    components = {
        "height_l1": height_l1,
        "height_mse": height_mse,
        "boundary": boundary,
        "seam": seam,
        "gradient": gradient,
        "curvature": curvature,
    }
    return total, components


def train_epoch(model, loader, optimizer, device, beta):
    model.train()
    totals = {}
    samples = 0

    for batch in loader:
        condition = batch["condition"].to(device)
        target = batch["target"].to(device)
        hole = batch["hole"].to(device)

        optimizer.zero_grad(set_to_none=True)

        posterior, parameters, prior = model(
            condition,
            target=target,
            return_prior=True,
        )

        posterior_loss, _ = inpainting_loss(posterior, target, hole)
        prior_loss, components = inpainting_loss(prior, target, hole)
        kl = conditional_kl(parameters)

        loss = (
            posterior_loss
            + PRIOR_RECONSTRUCTION_WEIGHT * prior_loss
            + beta * kl
        )

        if not torch.isfinite(loss).item():
            raise RuntimeError("NaN/Inf в функции потерь.")

        loss.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP,
            error_if_nonfinite=True,
        )
        optimizer.step()

        size = target.shape[0]
        samples += size

        values = {
            "train_loss": loss,
            "train_posterior_loss": posterior_loss,
            "train_prior_loss": prior_loss,
            "train_kl": kl,
            **{f"train_{key}": value for key, value in components.items()},
        }

        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value.item() * size

    return {key: value / samples for key, value in totals.items()}


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()

    squared = absolute = pixels = 0.0
    boundary_absolute = boundary_pixels = 0.0
    seam_absolute = seam_edges = 0.0
    gradient_absolute = gradient_edges = 0.0

    for batch in loader:
        condition = batch["condition"].to(device)
        target = batch["target"].to(device)
        hole = batch["hole"].to(device)
        scale = batch["scale"].to(device).view(-1, 1, 1, 1)

        prediction = model(condition, stochastic=False)
        completed = prediction * hole + target * (1 - hole)
        error_m = (completed - target) * scale

        if not torch.isfinite(error_m).all().item():
            raise RuntimeError("NaN/Inf при validation/test.")

        squared += (error_m.square() * hole).sum().item()
        absolute += (error_m.abs() * hole).sum().item()
        pixels += hole.sum().item()

        band = boundary_band(hole)
        boundary_absolute += (error_m.abs() * band).sum().item()
        boundary_pixels += band.sum().item()

        # Производные разности равны разности производных.
        dx, dy = differences(error_m)

        sx, sy = seam_masks(hole)
        gx, gy = affected_edge_masks(hole)

        seam_absolute += (
            (dx.abs() * sx).sum() + (dy.abs() * sy).sum()
        ).item()
        seam_edges += (sx.sum() + sy.sum()).item()

        gradient_absolute += (
            (dx.abs() * gx).sum() + (dy.abs() * gy).sum()
        ).item()
        gradient_edges += (gx.sum() + gy.sum()).item()

    if min(pixels, boundary_pixels, seam_edges, gradient_edges) <= 0:
        raise RuntimeError("Недостаточно пикселей/рёбер для оценки.")

    result = {
        "rmse_m": float(np.sqrt(squared / pixels)),
        "mae_m": absolute / pixels,
        "boundary_mae_m": boundary_absolute / boundary_pixels,
        "seam_increment_mae_m": seam_absolute / seam_edges,
        "gradient_increment_mae_m": gradient_absolute / gradient_edges,
    }

    result["score"] = (
        result["rmse_m"]
        + SELECTION_SEAM_WEIGHT * result["seam_increment_mae_m"]
        + SELECTION_GRADIENT_WEIGHT * result["gradient_increment_mae_m"]
    )
    return result


def save_history(history):
    with (OUTPUT_DIR / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    groups = [
        (
            axes[0, 0],
            ["train_posterior_loss", "train_prior_loss"],
            "Normalized reconstruction objective",
        ),
        (
            axes[0, 1],
            ["val_rmse_m", "val_mae_m", "val_boundary_mae_m"],
            "Height errors, m",
        ),
        (
            axes[1, 0],
            ["val_seam_increment_mae_m", "val_gradient_increment_mae_m"],
            "Height-increment errors, m",
        ),
        (
            axes[1, 1],
            ["val_score"],
            "Model selection score",
        ),
    ]

    for axis, keys, title in groups:
        for key in keys:
            axis.plot(epochs, [row[key] for row in history], label=key)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "learning_curves.png", dpi=180)
    plt.close(fig)


@torch.inference_mode()
def save_examples(model, dataset, device, filename):
    model.eval()
    count = min(EXAMPLE_COUNT, len(dataset))
    fig, axes = plt.subplots(
        count, 5, figsize=(19, 3.5 * count), squeeze=False
    )

    for row in range(count):
        item = dataset[row]

        prediction = model(
            item["condition"][None].to(device)
        )[0, 0].cpu().numpy().astype(np.float64)

        target = item["target"][0].numpy().astype(np.float64)
        hole = item["hole"][0].numpy().astype(bool)
        scale, offset = float(item["scale"]), float(item["offset"])

        original = target * scale + offset
        completed = np.where(hole, prediction, target) * scale + offset
        observed = np.where(hole, np.nan, original)
        error = np.where(hole, (prediction - target) * scale, np.nan)

        # Модуль градиента только для иллюстрации, не угол уклона.
        gy, gx = np.gradient(completed)
        gradient_magnitude = np.hypot(gx, gy)

        low = float(min(original.min(), completed.min()))
        high = float(max(original.max(), completed.max()))
        high = max(high, low + 1e-6)

        error_limit = max(float(np.nanmax(np.abs(error))), 0.01)

        images = [original, observed, completed, error, gradient_magnitude]
        titles = [
            "Original",
            "Observed",
            "Completed",
            "Hole error, m",
            "Completed gradient, m/pixel",
        ]

        for col, (image, title) in enumerate(zip(images, titles)):
            axis = axes[row, col]
            if col == 3:
                shown = axis.imshow(
                    image, cmap="RdBu_r",
                    vmin=-error_limit, vmax=error_limit,
                )
            elif col == 4:
                shown = axis.imshow(image, cmap="magma")
            else:
                shown = axis.imshow(
                    image, cmap="terrain", vmin=low, vmax=high
                )
            axis.set_title(title, fontsize=10)
            axis.axis("off")
            fig.colorbar(shown, ax=axis, shrink=0.75)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=180)
    plt.close(fig)


def preprocessing_config():
    return {
        "patch_size": PATCH_SIZE,
        "normalization": "local min-max from known pixels",
        "min_height_scale": MIN_HEIGHT_SCALE,
        "known_mask_convention": "1=known, 0=missing",
        "condition_channels": [
            "normalized_dem_with_zero_in_holes",
            "known_mask",
        ],
        "output_activation": "linear",
        "clip_normalized_values": False,
        "assumed_height_unit": "metre",
    }


def save_checkpoint(model, optimizer, scheduler, epoch, validation):
    checkpoint = {
        "epoch": epoch,
        "val_rmse_m": validation["rmse_m"],
        "val_score": validation["score"],
        "validation_metrics": validation,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": {
            "base_channels": BASE_CHANNELS,
            "latent_dim": LATENT_DIM,
        },
        "preprocessing": preprocessing_config(),
    }

    temporary = OUTPUT_DIR / "best_model.tmp"
    torch.save(checkpoint, temporary)
    temporary.replace(OUTPUT_DIR / "best_model.pt")


def main():
    validate_settings()
    seed_everything()
    device = choose_device()

    datasets = {
        split: DEMDataset(split)
        for split in ("train", "val", "test")
    }

    generator = torch.Generator().manual_seed(SEED)

    loaders = {
        split: DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=(split == "train"),
            num_workers=NUM_WORKERS,
            persistent_workers=False,
            generator=generator if split == "train" else None,
        )
        for split, dataset in datasets.items()
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in globals().items()
        if key.isupper()
        and isinstance(
            value, (str, int, float, bool, tuple, Path, type(None))
        )
    }
    config.update({
        "selected_device": str(device),
        "torch_version": str(torch.__version__),
        "preprocessing": preprocessing_config(),
        "sample_counts": {
            key: len(value) for key, value in datasets.items()
        },
    })
    write_json(OUTPUT_DIR / "config.json", config)

    model = QuarryCVAE().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )

    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Samples: {config['sample_counts']}")
    print(f"Output: {OUTPUT_DIR}\n")

    history = []
    best_score = float("inf")
    stale_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        started = time.perf_counter()
        datasets["train"].epoch = epoch

        beta = KL_MAX_WEIGHT * min(
            1.0, epoch / max(1, KL_WARMUP_EPOCHS)
        )

        training = train_epoch(
            model, loaders["train"], optimizer, device, beta
        )
        validation = evaluate(model, loaders["val"], device)

        score = validation["score"]
        if not np.isfinite(score):
            raise RuntimeError("Validation score содержит NaN/Inf.")

        scheduler.step(score)

        improved = score < best_score
        if improved:
            best_score = score
            stale_epochs = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, validation
            )
        else:
            stale_epochs += 1

        history.append({
            "epoch": epoch,
            **training,
            **{f"val_{key}": value for key, value in validation.items()},
            "beta": beta,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        save_history(history)

        if improved:
            save_examples(
                model, datasets["val"], device, "best_val_examples.png"
            )

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"prior loss={training['train_prior_loss']:.5f} | "
            f"RMSE={validation['rmse_m']:.3f} m | "
            f"boundary MAE={validation['boundary_mae_m']:.3f} m | "
            f"seam={validation['seam_increment_mae_m']:.3f} m | "
            f"score={score:.3f} | "
            f"{time.perf_counter() - started:.1f} s"
            + (" | BEST" if improved else "")
        )

        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            print("Early stopping.")
            break

    checkpoint = torch.load(
        OUTPUT_DIR / "best_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    test_metrics = evaluate(model, loaders["test"], device)

    write_json(
        OUTPUT_DIR / "metrics.json",
        {
            "best_epoch": checkpoint["epoch"],
            "validation": checkpoint["validation_metrics"],
            "test": test_metrics,
            "evaluation": "deterministic conditional prior mean",
            "preprocessing": preprocessing_config(),
        },
    )

    save_examples(
        model, datasets["test"], device, "test_examples.png"
    )

    print(f"\nBest epoch: {checkpoint['epoch']}")
    print("Test metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()