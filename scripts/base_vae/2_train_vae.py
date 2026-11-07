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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data/processed/patches_64_d4"

RUN_NAME = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / f"cvae_{RUN_NAME}"

PATCH_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 150

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5

BASE_CHANNELS = 32
LATENT_DIM = 32

# Локальная нормализация по известным пикселям:
# normalized = (height - visible_min) / scale
# scale = max(visible_max - visible_min, MIN_HEIGHT_SCALE)
#
# Исходные высоты предполагаются в метрах.
MIN_HEIGHT_SCALE = 1.0

# Веса функции потерь.
KNOWN_LOSS_WEIGHT = 0.1
GRADIENT_LOSS_WEIGHT = 0.2

# KL усредняется по объектам и латентным координатам.
KL_MAX_WEIGHT = 1e-3
KL_WARMUP_EPOCHS = 30

EARLY_STOPPING_PATIENCE = 25
GRADIENT_CLIP = 5.0

# Синтетические пропуски.
MIN_HOLE_SIZE = 8
MAX_HOLE_SIZE = 28

MIN_HOLE_FRACTION = 0.03
MAX_HOLE_FRACTION = 0.35

MAX_MASK_SHAPES = 3
MASK_GENERATION_ATTEMPTS = 100

SEED = 42

# Для macOS/PyCharm первый запуск лучше выполнить с 0.
NUM_WORKERS = 0

# "auto", "cpu", "mps", "cuda".
DEVICE = "auto"

# None — использовать все обучающие образцы.
MAX_TRAIN_SAMPLES = None

# Сколько примеров сохранять на иллюстрациях.
EXAMPLE_COUNT = 4

# ==================================================================


def validate_settings() -> None:
    if PATCH_SIZE != 64:
        raise ValueError(
            "Текущий эксперимент настроен на фрагменты 64 × 64."
        )

    if BATCH_SIZE <= 0 or EPOCHS <= 0:
        raise ValueError("BATCH_SIZE и EPOCHS должны быть положительными.")

    if BASE_CHANNELS <= 0 or BASE_CHANNELS % 8 != 0:
        raise ValueError("BASE_CHANNELS должен быть положительным и кратным 8.")

    if LATENT_DIM <= 0:
        raise ValueError("LATENT_DIM должен быть положительным.")

    if MIN_HEIGHT_SCALE <= 0:
        raise ValueError("MIN_HEIGHT_SCALE должен быть положительным.")

    if not 0 < MIN_HOLE_SIZE <= MAX_HOLE_SIZE <= PATCH_SIZE:
        raise ValueError("Некорректные размеры пропусков.")

    if not 0 < MIN_HOLE_FRACTION <= MAX_HOLE_FRACTION < 1:
        raise ValueError("Некорректные доли пропусков.")

    if MAX_MASK_SHAPES < 1 or MASK_GENERATION_ATTEMPTS < 1:
        raise ValueError("Некорректные параметры генерации масок.")

    if MAX_TRAIN_SAMPLES is not None and MAX_TRAIN_SAMPLES < 1:
        raise ValueError("MAX_TRAIN_SAMPLES должен быть None или >= 1.")

    if EARLY_STOPPING_PATIENCE < 1 or EXAMPLE_COUNT < 1:
        raise ValueError("Некорректные параметры сохранения/остановки.")

    if NUM_WORKERS < 0:
        raise ValueError("NUM_WORKERS не может быть отрицательным.")

    if DEVICE not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("DEVICE должен быть auto, cpu, mps или cuda.")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def choose_device() -> torch.device:
    if DEVICE == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA недоступна.")
        return torch.device("cuda")

    if DEVICE == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS недоступен.")
        return torch.device("mps")

    if DEVICE == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def make_hole_mask(rng: np.random.Generator) -> np.ndarray:
    """True — пиксель искусственного пропуска."""
    yy, xx = np.mgrid[:PATCH_SIZE, :PATCH_SIZE]

    for _ in range(MASK_GENERATION_ATTEMPTS):
        hole = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=bool)

        shape_count = int(rng.integers(1, MAX_MASK_SHAPES + 1))

        for _ in range(shape_count):
            height = int(
                rng.integers(MIN_HOLE_SIZE, MAX_HOLE_SIZE + 1)
            )
            width = int(
                rng.integers(MIN_HOLE_SIZE, MAX_HOLE_SIZE + 1)
            )

            row = int(rng.integers(0, PATCH_SIZE - height + 1))
            col = int(rng.integers(0, PATCH_SIZE - width + 1))

            if rng.random() < 0.5:
                hole[row:row + height, col:col + width] = True
            else:
                center_y = row + (height - 1) / 2
                center_x = col + (width - 1) / 2

                ellipse = (
                    ((yy - center_y) / (height / 2)) ** 2
                    + ((xx - center_x) / (width / 2)) ** 2
                ) <= 1

                hole |= ellipse

        if MIN_HOLE_FRACTION <= hole.mean() <= MAX_HOLE_FRACTION:
            return hole

    raise RuntimeError(
        "Не удалось создать маску. Проверь размеры и доли пропусков."
    )


class DEMDataset(Dataset):
    def __init__(self, split: str) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Неизвестная выборка: {split}")

        self.split = split
        self.epoch = 0

        self.files = sorted((DATA_DIR / split).glob("*.npy"))

        if split == "train" and MAX_TRAIN_SAMPLES is not None:
            rng = np.random.default_rng(SEED)
            indices = rng.permutation(len(self.files))
            indices = indices[:MAX_TRAIN_SAMPLES]
            self.files = [self.files[int(i)] for i in indices]

        if not self.files:
            raise FileNotFoundError(
                f"Нет .npy-файлов в каталоге:\n{DATA_DIR / split}"
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict:
        path = self.files[index]

        dem = np.asarray(
            np.load(path, allow_pickle=False),
            dtype=np.float32,
        )

        if dem.shape != (PATCH_SIZE, PATCH_SIZE):
            raise ValueError(
                f"Некорректный размер {dem.shape}:\n{path}"
            )

        if not np.isfinite(dem).all():
            raise ValueError(f"Невалидные высоты:\n{path}")

        split_code = {"train": 0, "val": 1, "test": 2}[self.split]

        # Train: новые маски каждую эпоху.
        # Val/test: фиксированные маски.
        mask_epoch = self.epoch if self.split == "train" else 0

        rng = np.random.default_rng(
            np.random.SeedSequence(
                [SEED, split_code, mask_epoch, index]
            )
        )

        hole = make_hole_mask(rng)
        known = ~hole

        # Параметры нормализации вычисляются только по контексту.
        visible = dem[known]
        offset = float(visible.min())
        visible_range = float(visible.max()) - offset

        scale = max(visible_range, MIN_HEIGHT_SCALE)

        # Промежуточные вычисления в float64.
        target = (
            (dem.astype(np.float64) - offset) / scale
        ).astype(np.float32)

        if not np.isfinite(target).all():
            raise ValueError(f"Ошибка нормализации:\n{path}")

        known_float = known.astype(np.float32)
        corrupted = np.where(known, target, 0.0).astype(np.float32)

        condition = np.stack(
            [corrupted, known_float],
            axis=0,
        )

        return {
            "condition": torch.from_numpy(condition),
            "target": torch.from_numpy(target[None]),
            "hole": torch.from_numpy(hole.astype(np.float32)[None]),
            "offset": torch.tensor(offset, dtype=torch.float32),
            "scale": torch.tensor(scale, dtype=torch.float32),
        }


def conv_block(
    in_channels: int,
    out_channels: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.GroupNorm(8, out_channels),
        nn.SiLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.GroupNorm(8, out_channels),
        nn.SiLU(),
    )


class Encoder(nn.Module):
    def __init__(self, in_channels: int, base: int) -> None:
        super().__init__()

        self.block1 = conv_block(in_channels, base)
        self.block2 = conv_block(base, base * 2)
        self.block3 = conv_block(base * 2, base * 4)
        self.block4 = conv_block(base * 4, base * 8)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f1 = self.block1(x)
        f2 = self.block2(F.avg_pool2d(f1, 2))
        f3 = self.block3(F.avg_pool2d(f2, 2))
        f4 = self.block4(F.avg_pool2d(f3, 2))

        return [f1, f2, f3, f4]


class GaussianHead(nn.Module):
    def __init__(self, channels: int, latent_dim: int) -> None:
        super().__init__()
        self.head = nn.Linear(channels, latent_dim * 2)

    def forward(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = features.mean(dim=(2, 3))
        mu, logvar = self.head(pooled).chunk(2, dim=1)

        return mu, logvar.clamp(-8.0, 8.0)


class QuarryCVAE(nn.Module):
    def __init__(
        self,
        base_channels: int = BASE_CHANNELS,
        latent_dim: int = LATENT_DIM,
    ) -> None:
        super().__init__()

        base = base_channels

        # Контекст: повреждённая DEM + маска известных пикселей.
        self.context_encoder = Encoder(2, base)

        # Posterior дополнительно видит полную целевую DEM.
        self.posterior_encoder = Encoder(3, base)

        self.prior_head = GaussianHead(base * 8, latent_dim)
        self.posterior_head = GaussianHead(base * 8, latent_dim)

        self.bottleneck = conv_block(
            base * 8 + latent_dim,
            base * 8,
        )
        self.up3 = conv_block(base * 8 + base * 4, base * 4)
        self.up2 = conv_block(base * 4 + base * 2, base * 2)
        self.up1 = conv_block(base * 2 + base, base)

        # Без Sigmoid: скрытые высоты могут выходить за [0, 1].
        self.output = nn.Conv2d(base, 1, kernel_size=1)

    @staticmethod
    def sample(
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        noise = torch.randn_like(mu)
        return mu + noise * torch.exp(0.5 * logvar)

    def decode(
        self,
        features: list[torch.Tensor],
        z: torch.Tensor,
    ) -> torch.Tensor:
        f1, f2, f3, f4 = features

        spatial_z = z[:, :, None, None].expand(
            -1,
            -1,
            f4.shape[2],
            f4.shape[3],
        )

        x = self.bottleneck(
            torch.cat([f4, spatial_z], dim=1)
        )

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
        condition: torch.Tensor,
        target: torch.Tensor | None = None,
        stochastic: bool = False,
    ):
        context_features = self.context_encoder(condition)

        prior_mu, prior_logvar = self.prior_head(
            context_features[-1]
        )

        if target is not None:
            posterior_input = torch.cat(
                [condition, target],
                dim=1,
            )
            posterior_features = self.posterior_encoder(
                posterior_input
            )

            posterior_mu, posterior_logvar = self.posterior_head(
                posterior_features[-1]
            )

            z = self.sample(posterior_mu, posterior_logvar)
            prediction = self.decode(context_features, z)

            parameters = (
                posterior_mu,
                posterior_logvar,
                prior_mu,
                prior_logvar,
            )

            return prediction, parameters

        if stochastic:
            z = self.sample(prior_mu, prior_logvar)
        else:
            z = prior_mu

        return self.decode(context_features, z)


def masked_mean_per_sample(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Каждый образец имеет одинаковый вес независимо от площади маски."""
    dimensions = tuple(range(1, values.ndim))

    numerator = (values * mask).sum(dim=dimensions)
    denominator = mask.sum(dim=dimensions).clamp_min(1.0)

    return (numerator / denominator).mean()


def conditional_kl(parameters: tuple) -> torch.Tensor:
    q_mu, q_logvar, p_mu, p_logvar = parameters

    kl = 0.5 * (
        p_logvar
        - q_logvar
        + torch.exp(q_logvar - p_logvar)
        + (q_mu - p_mu).square() * torch.exp(-p_logvar)
        - 1.0
    )

    return kl.mean()


def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    hole: torch.Tensor,
) -> torch.Tensor:
    squared_error = (prediction - target).square()

    hole_loss = masked_mean_per_sample(
        squared_error,
        hole,
    )
    known_loss = masked_mean_per_sample(
        squared_error,
        1.0 - hole,
    )

    # Для градиентного слагаемого известные высоты сохраняем.
    completed = prediction * hole + target * (1.0 - hole)

    dx_error = (
        (completed[..., 1:] - completed[..., :-1])
        - (target[..., 1:] - target[..., :-1])
    ).abs()

    dy_error = (
        (completed[..., 1:, :] - completed[..., :-1, :])
        - (target[..., 1:, :] - target[..., :-1, :])
    ).abs()

    dx_mask = torch.maximum(
        hole[..., 1:],
        hole[..., :-1],
    )
    dy_mask = torch.maximum(
        hole[..., 1:, :],
        hole[..., :-1, :],
    )

    gradient_loss = 0.5 * (
        masked_mean_per_sample(dx_error, dx_mask)
        + masked_mean_per_sample(dy_error, dy_mask)
    )

    return (
        hole_loss
        + KNOWN_LOSS_WEIGHT * known_loss
        + GRADIENT_LOSS_WEIGHT * gradient_loss
    )


def train_epoch(
    model: QuarryCVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float,
) -> dict:
    model.train()

    total_loss = 0.0
    total_kl = 0.0
    total_samples = 0

    squared_error_m = 0.0
    hole_pixels = 0.0

    for batch in loader:
        condition = batch["condition"].to(device)
        target = batch["target"].to(device)
        hole = batch["hole"].to(device)
        scale = batch["scale"].to(device).view(-1, 1, 1, 1)

        optimizer.zero_grad(set_to_none=True)

        prediction, parameters = model(
            condition,
            target=target,
        )

        reconstruction = reconstruction_loss(
            prediction,
            target,
            hole,
        )
        kl = conditional_kl(parameters)
        loss = reconstruction + beta * kl

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                "NaN/Inf в функции потерь. Проверь данные и нормализацию."
            )

        loss.backward()

        nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP,
            error_if_nonfinite=True,
        )

        optimizer.step()

        batch_size = target.shape[0]
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_kl += kl.item() * batch_size

        error_m = (prediction.detach() - target) * scale

        squared_error_m += (
            error_m.square() * hole
        ).sum().item()

        hole_pixels += hole.sum().item()

    return {
        "train_loss": total_loss / total_samples,
        "train_kl": total_kl / total_samples,
        "train_posterior_rmse_m": float(
            np.sqrt(squared_error_m / hole_pixels)
        ),
    }


@torch.no_grad()
def evaluate(
    model: QuarryCVAE,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()

    squared_error_m = 0.0
    absolute_error_m = 0.0
    hole_pixels = 0.0

    for batch in loader:
        condition = batch["condition"].to(device)
        target = batch["target"].to(device)
        hole = batch["hole"].to(device)
        scale = batch["scale"].to(device).view(-1, 1, 1, 1)

        # Во время оценки целевая DEM в модель не передаётся.
        prediction = model(condition, stochastic=False)

        error_m = (prediction - target) * scale

        if not torch.isfinite(error_m).all().item():
            raise RuntimeError("NaN/Inf при оценке модели.")

        squared_error_m += (
            error_m.square() * hole
        ).sum().item()

        absolute_error_m += (
            error_m.abs() * hole
        ).sum().item()

        hole_pixels += hole.sum().item()

    if hole_pixels == 0:
        raise RuntimeError("В оценочной выборке нет пропусков.")

    return {
        "rmse_m": float(np.sqrt(squared_error_m / hole_pixels)),
        "mae_m": float(absolute_error_m / hole_pixels),
    }


def save_history(history: list[dict]) -> None:
    with (OUTPUT_DIR / "history.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(history[0]),
        )
        writer.writeheader()
        writer.writerows(history)

    epochs = [record["epoch"] for record in history]

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    axes[0].plot(
        epochs,
        [record["train_loss"] for record in history],
        label="Train objective",
    )
    axes[0].set_ylabel("Normalized loss")

    axes[1].plot(
        epochs,
        [record["train_posterior_rmse_m"] for record in history],
        label="Train: posterior sample",
    )
    axes[1].plot(
        epochs,
        [record["val_rmse_m"] for record in history],
        label="Validation: prior mean",
    )
    axes[1].set_ylabel("Hole RMSE, m")

    axes[2].plot(
        epochs,
        [record["train_kl"] for record in history],
        label="Mean KL(q || p)",
    )
    axes[2].set_ylabel("KL per latent dimension")

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "learning_curves.png",
        dpi=180,
    )
    plt.close(fig)


@torch.no_grad()
def save_examples(
    model: QuarryCVAE,
    dataset: DEMDataset,
    device: torch.device,
    filename: str,
) -> None:
    model.eval()

    count = min(EXAMPLE_COUNT, len(dataset))

    fig, axes = plt.subplots(
        count,
        4,
        figsize=(14, 3.4 * count),
        squeeze=False,
    )

    for row in range(count):
        item = dataset[row]

        condition = item["condition"][None].to(device)
        prediction = model(
            condition,
            stochastic=False,
        )[0, 0].cpu().numpy()

        target = item["target"][0].numpy()
        hole = item["hole"][0].numpy().astype(bool)

        offset = float(item["offset"])
        scale = float(item["scale"])

        original = target.astype(np.float64) * scale + offset

        completed_normalized = np.where(
            hole,
            prediction,
            target,
        )
        completed = (
            completed_normalized.astype(np.float64) * scale + offset
        )

        corrupted = np.where(hole, np.nan, original)
        error = np.where(
            hole,
            (prediction.astype(np.float64) - target) * scale,
            np.nan,
        )

        vmin = float(min(original.min(), completed.min()))
        vmax = float(max(original.max(), completed.max()))

        if vmax <= vmin:
            vmax = vmin + 1.0

        error_limit = max(
            float(np.nanmax(np.abs(error))),
            0.01,
        )

        sample_rmse = float(
            np.sqrt(np.mean(error[hole] ** 2))
        )

        images = [
            original,
            corrupted,
            completed,
            error,
        ]
        titles = [
            "Original DEM",
            "Observed DEM",
            "Completed DEM",
            f"Hole error, m | RMSE={sample_rmse:.2f}",
        ]

        for col, (image, title) in enumerate(zip(images, titles)):
            axis = axes[row, col]

            if col == 3:
                shown = axis.imshow(
                    image,
                    cmap="RdBu_r",
                    vmin=-error_limit,
                    vmax=error_limit,
                )
            else:
                shown = axis.imshow(
                    image,
                    cmap="terrain",
                    vmin=vmin,
                    vmax=vmax,
                )

            axis.set_title(title, fontsize=10)
            axis.axis("off")
            fig.colorbar(shown, ax=axis, shrink=0.75)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=180,
    )
    plt.close(fig)


def preprocessing_config() -> dict:
    return {
        "patch_size": PATCH_SIZE,
        "normalization": "local min-max from known pixels",
        "min_height_scale": MIN_HEIGHT_SCALE,
        "offset": "minimum of known pixels",
        "scale": "max(known maximum - known minimum, min_height_scale)",
        "condition_channels": [
            "normalized_dem_with_zero_in_holes",
            "known_mask",
        ],
        "known_mask_convention": "1=known, 0=missing",
        "output_activation": "linear",
        "clip_normalized_values": False,
        "assumed_height_unit": "metre",
    }


def save_checkpoint(
    model: QuarryCVAE,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    val_rmse: float,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "val_rmse_m": val_rmse,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": {
            "base_channels": BASE_CHANNELS,
            "latent_dim": LATENT_DIM,
        },
        "preprocessing": preprocessing_config(),
    }

    temporary_path = OUTPUT_DIR / "best_model.tmp"
    final_path = OUTPUT_DIR / "best_model.pt"

    torch.save(checkpoint, temporary_path)
    temporary_path.replace(final_path)


def main() -> None:
    validate_settings()
    seed_everything(SEED)

    device = choose_device()

    train_dataset = DEMDataset("train")
    val_dataset = DEMDataset("val")
    test_dataset = DEMDataset("test")

    generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=generator,
        persistent_workers=False,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=False,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in globals().items()
        if key.isupper()
        and isinstance(
            value,
            (str, int, float, bool, tuple, Path, type(None)),
        )
    }

    config.update({
        "selected_device": str(device),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "preprocessing": preprocessing_config(),
        "sample_counts": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
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
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Device: {device}")
    print(f"Parameters: {parameter_count:,}")
    print(
        f"Samples: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, "
        f"test={len(test_dataset)}"
    )
    print("Normalization: local min-max from known pixels")
    print(f"Minimum height scale: {MIN_HEIGHT_SCALE} m")
    print(f"Output: {OUTPUT_DIR}\n")

    history = []

    best_rmse = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        started = time.perf_counter()

        train_dataset.epoch = epoch

        beta = KL_MAX_WEIGHT * min(
            1.0,
            epoch / max(KL_WARMUP_EPOCHS, 1),
        )

        training = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            beta,
        )

        validation = evaluate(
            model,
            val_loader,
            device,
        )

        val_rmse = validation["rmse_m"]

        if not np.isfinite(val_rmse):
            raise RuntimeError("Validation RMSE содержит NaN/Inf.")

        scheduler.step(val_rmse)

        improved = val_rmse < best_rmse

        if improved:
            best_rmse = val_rmse
            best_epoch = epoch
            stale_epochs = 0

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_rmse,
            )
        else:
            stale_epochs += 1

        record = {
            "epoch": epoch,
            **training,
            "val_rmse_m": val_rmse,
            "val_mae_m": validation["mae_m"],
            "beta": beta,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
        }

        history.append(record)
        save_history(history)

        if improved:
            save_examples(
                model,
                val_dataset,
                device,
                "best_val_examples.png",
            )

        duration = time.perf_counter() - started

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"loss={training['train_loss']:.6f} | "
            f"KL={training['train_kl']:.5f} | "
            f"val RMSE={val_rmse:.3f} m | "
            f"val MAE={validation['mae_m']:.3f} m | "
            f"beta={beta:.6f} | "
            f"{duration:.1f} s"
            + (" | BEST" if improved else "")
        )

        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            print("\nEarly stopping.")
            break

    checkpoint = torch.load(
        OUTPUT_DIR / "best_model.pt",
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    test_metrics = evaluate(
        model,
        test_loader,
        device,
    )

    report = {
        "best_epoch": best_epoch,
        "best_val_rmse_m": best_rmse,
        "test_rmse_m": test_metrics["rmse_m"],
        "test_mae_m": test_metrics["mae_m"],
        "evaluation": "decode at conditional prior mean",
        "metric_region": "synthetic holes only",
        "metric_aggregation": "pooled over all missing pixels",
        "preprocessing": preprocessing_config(),
    }

    write_json(
        OUTPUT_DIR / "metrics.json",
        report,
    )

    save_examples(
        model,
        test_dataset,
        device,
        "test_examples.png",
    )

    print("\nОбучение завершено.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation RMSE: {best_rmse:.3f} m")
    print(f"Test RMSE: {test_metrics['rmse_m']:.3f} m")
    print(f"Test MAE: {test_metrics['mae_m']:.3f} m")
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()