from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from train_vae import QuarryCVAE


# ============================= CONFIG =============================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Укажи путь к модели своего запуска.
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "cvae_20261107_122414_747615"
    / "best_model.pt"
)

# "demo" — искусственный пропуск в полном тестовом фрагменте.
# "real" — реальные пропуски во входном массиве.
MODE = "demo"

# Если None и MODE="demo", берём первый .npy из test.
# Для MODE="real" нужно указать путь явно.
INPUT_PATH = None

# Пример:
# INPUT_PATH = PROJECT_ROOT / "data/inference/dem_with_hole.npy"

TEST_DIR = PROJECT_ROOT / "data/processed/patches_64_d4/test"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
)

# Дополнительное значение NoData для режима real.
# NaN и Inf распознаются автоматически.
NODATA_VALUE = None

# При необходимости можно передать отдельную маску .npy.
# 1 / True — известный пиксель; 0 / False — пропуск.
KNOWN_MASK_PATH = None

# Прямоугольный пропуск для demo: [начало:конец].
DEMO_ROW_START = 20
DEMO_ROW_END = 44
DEMO_COL_START = 20
DEMO_COL_END = 44

# False — детерминированное декодирование среднего кода prior,
# как при выборе лучшей модели во время обучения.
# True — случайная выборка из условного prior.
STOCHASTIC = False

SEED = 42

# "auto", "cpu", "mps", "cuda".
DEVICE = "auto"

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


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[QuarryCVAE, dict]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint не найден:\n{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    preprocessing = checkpoint["preprocessing"]

    if preprocessing.get("normalization") != (
        "local min-max from known pixels"
    ):
        raise ValueError(
            "Checkpoint использует другую нормализацию. "
            "Нужна модель из последней версии train_vae.py."
        )

    if preprocessing.get("known_mask_convention") != (
        "1=known, 0=missing"
    ):
        raise ValueError("Неожиданное соглашение о маске.")

    model = QuarryCVAE(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def resolve_input_path() -> Path:
    if INPUT_PATH is not None:
        return Path(INPUT_PATH)

    if MODE == "real":
        raise ValueError("Для MODE='real' задай INPUT_PATH.")

    candidates = sorted(TEST_DIR.glob("*.npy"))

    if not candidates:
        raise FileNotFoundError(
            f"В тестовом каталоге нет .npy:\n{TEST_DIR}"
        )

    return candidates[0]


def load_input(
    input_path: Path,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    dem = np.asarray(
        np.load(input_path, allow_pickle=False),
        dtype=np.float64,
    )

    if dem.shape != (patch_size, patch_size):
        raise ValueError(
            f"Ожидается {(patch_size, patch_size)}, "
            f"получено {dem.shape}."
        )

    if MODE == "demo":
        if not np.isfinite(dem).all():
            raise ValueError(
                "Для demo нужен полный фрагмент без NaN/Inf."
            )

        if (
            NODATA_VALUE is not None
            and np.any(dem == NODATA_VALUE)
        ):
            raise ValueError("В demo-фрагменте присутствует NoData.")

        if not (
            0 <= DEMO_ROW_START < DEMO_ROW_END <= patch_size
            and 0 <= DEMO_COL_START < DEMO_COL_END <= patch_size
        ):
            raise ValueError("Некорректные границы demo-пропуска.")

        reference = dem.copy()
        known = np.ones(dem.shape, dtype=bool)

        known[
            DEMO_ROW_START:DEMO_ROW_END,
            DEMO_COL_START:DEMO_COL_END,
        ] = False

    elif MODE == "real":
        reference = None
        known = np.isfinite(dem)

        if NODATA_VALUE is not None:
            known &= dem != NODATA_VALUE

        if KNOWN_MASK_PATH is not None:
            external_mask = np.load(
                KNOWN_MASK_PATH,
                allow_pickle=False,
            )

            if external_mask.shape != dem.shape:
                raise ValueError(
                    "Размер отдельной маски не совпадает с DEM."
                )

            if not np.isin(external_mask, [0, 1]).all():
                raise ValueError(
                    "Отдельная маска должна содержать только 0 и 1."
                )

            known &= external_mask.astype(bool)

    else:
        raise ValueError("MODE должен быть 'demo' или 'real'.")

    if not known.any():
        raise ValueError(
            "Нет известных высот: восстановление по контексту невозможно."
        )

    # Скрытые значения не передаём функции восстановления.
    observed = np.where(known, dem, np.nan)

    return observed, known, reference


@torch.inference_mode()
def fill_patch(
    model: QuarryCVAE,
    dem: np.ndarray,
    known: np.ndarray,
    preprocessing: dict,
    device: torch.device,
    stochastic: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    dem: высоты в метрах, shape=(64, 64).
    known: True — известная высота, False — пропуск.
    """
    dem = np.asarray(dem, dtype=np.float64)
    known = np.asarray(known, dtype=bool)

    patch_size = int(preprocessing["patch_size"])

    if dem.shape != (patch_size, patch_size):
        raise ValueError("Неподдерживаемый размер фрагмента.")

    if known.shape != dem.shape:
        raise ValueError("Размер маски не совпадает с DEM.")

    if not known.any():
        raise ValueError("Во фрагменте нет известных высот.")

    if not np.isfinite(dem[known]).all():
        raise ValueError("Среди известных высот есть NaN/Inf.")

    offset = float(dem[known].min())
    visible_range = float(dem[known].max()) - offset

    scale = max(
        visible_range,
        float(preprocessing["min_height_scale"]),
    )

    metadata = {
        "offset_m": offset,
        "scale_m": scale,
        "visible_range_m": visible_range,
        "missing_pixels": int((~known).sum()),
        "missing_fraction": float((~known).mean()),
    }

    if known.all():
        return dem.copy(), metadata

    normalized = np.zeros(dem.shape, dtype=np.float32)
    normalized[known] = (
        (dem[known] - offset) / scale
    ).astype(np.float32)

    if not np.isfinite(normalized).all():
        raise ValueError("NaN/Inf после нормализации входа.")

    condition = np.stack(
        [normalized, known.astype(np.float32)],
        axis=0,
    )

    tensor = torch.from_numpy(condition)[None].to(device)

    prediction = model(
        tensor,
        stochastic=stochastic,
    )[0, 0].cpu().numpy()

    if not np.isfinite(prediction).all():
        raise RuntimeError("Модель вернула NaN/Inf.")

    prediction_m = (
        prediction.astype(np.float64) * scale + offset
    )

    # Известные высоты не заменяем предсказанием.
    completed = dem.copy()
    completed[~known] = prediction_m[~known]

    return completed, metadata


def save_figure(
    observed: np.ndarray,
    completed: np.ndarray,
    known: np.ndarray,
    reference: np.ndarray | None,
) -> None:
    if reference is None:
        images = [observed, completed]
        titles = ["Observed DEM", "Completed DEM"]
    else:
        error = np.where(
            ~known,
            completed - reference,
            np.nan,
        )

        images = [reference, observed, completed, error]
        titles = [
            "Reference DEM",
            "Observed DEM",
            "Completed DEM",
            "Error in hole, m",
        ]

    height_arrays = [observed, completed]

    if reference is not None:
        height_arrays.append(reference)

    vmin = min(float(np.nanmin(a)) for a in height_arrays)
    vmax = max(float(np.nanmax(a)) for a in height_arrays)

    if vmax <= vmin:
        vmax = vmin + 1.0

    fig, axes = plt.subplots(
        1,
        len(images),
        figsize=(4.3 * len(images), 4.2),
        squeeze=False,
    )

    for index, (image, title) in enumerate(zip(images, titles)):
        axis = axes[0, index]
        is_error = reference is not None and index == 3

        if is_error:
            finite_error = image[np.isfinite(image)]
            limit = (
                max(float(np.abs(finite_error).max()), 0.01)
                if finite_error.size
                else 0.01
            )

            shown = axis.imshow(
                image,
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
            )
        else:
            shown = axis.imshow(
                image,
                cmap="terrain",
                vmin=vmin,
                vmax=vmax,
            )

        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(shown, ax=axis, shrink=0.75)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "inference.png", dpi=180)
    plt.close(fig)


def main() -> None:
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = choose_device()
    model, checkpoint = load_model(CHECKPOINT_PATH, device)

    preprocessing = checkpoint["preprocessing"]
    patch_size = int(preprocessing["patch_size"])

    input_path = resolve_input_path()

    observed, known, reference = load_input(
        input_path,
        patch_size,
    )

    completed, metadata = fill_patch(
        model=model,
        dem=observed,
        known=known,
        preprocessing=preprocessing,
        device=device,
        stochastic=STOCHASTIC,
    )

    # Проверяем сохранность исходных известных высот.
    if not np.array_equal(completed[known], observed[known]):
        raise RuntimeError("Известные высоты неожиданно изменились.")

    report = {
        "checkpoint": str(CHECKPOINT_PATH),
        "input": str(input_path),
        "device": str(device),
        "mode": MODE,
        "stochastic": STOCHASTIC,
        "seed": SEED,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        **metadata,
    }

    if reference is not None and (~known).any():
        error = completed[~known] - reference[~known]

        report["hole_rmse_m"] = float(
            np.sqrt(np.mean(error ** 2))
        )
        report["hole_mae_m"] = float(
            np.mean(np.abs(error))
        )
        report["hole_bias_m"] = float(
            np.mean(error)
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    np.save(
        OUTPUT_DIR / "observed.npy",
        observed,
        allow_pickle=False,
    )
    np.save(
        OUTPUT_DIR / "known_mask.npy",
        known,
        allow_pickle=False,
    )
    np.save(
        OUTPUT_DIR / "completed.npy",
        completed,
        allow_pickle=False,
    )

    if reference is not None:
        np.save(
            OUTPUT_DIR / "reference.npy",
            reference,
            allow_pickle=False,
        )

    (OUTPUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    save_figure(
        observed,
        completed,
        known,
        reference,
    )

    print(f"Device: {device}")
    print(f"Input: {input_path}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Missing pixels: {metadata['missing_pixels']}")
    print(f"Normalization offset: {metadata['offset_m']:.3f} m")
    print(f"Normalization scale: {metadata['scale_m']:.3f} m")

    if "hole_rmse_m" in report:
        print(f"Hole RMSE: {report['hole_rmse_m']:.3f} m")
        print(f"Hole MAE: {report['hole_mae_m']:.3f} m")
        print(f"Hole bias: {report['hole_bias_m']:.3f} m")

    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()