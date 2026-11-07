from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


# ============================= CONFIG =============================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DEM = PROJECT_ROOT / "data/raw/1m/grib_1m.tif"

PATCHES_DIR = PROJECT_ROOT / "data/interim/patches_64"
AUGMENTED_DIR = PROJECT_ROOT / "data/processed/patches_64_d4"

# Какие этапы выполнять.
RUN_TILING = True
RUN_AUGMENTATION = True

# Размер и шаг окна в пикселях.
PATCH_SIZE = 64
PATCH_STRIDE = 64

# Пространственное разделение.
# "auto" — вдоль большей размерности растра, либо явно "x" / "y".
SPLIT_AXIS = "auto"

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
# Остальная часть отводится под test.

# Ширина исключаемой полосы между train / val и val / test.
SPLIT_GAP = 256

# Аугментация применяется только к train.
ROTATIONS = (0, 90, 180, 270)
MIRROR_OPTIONS = (False, True)

# ==================================================================


def prepare_output_directory(path: Path) -> None:
    """Не допускает смешивания результатов разных запусков."""
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Каталог результатов не пуст:\n{path}\n"
            "Укажи новый путь или предварительно удали старые результаты."
        )

    path.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        (path / split).mkdir(exist_ok=True)


def write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_manifest(path: Path, records: list[dict]) -> None:
    if not records:
        raise RuntimeError("Не получено ни одного образца.")

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def validate_tiling_settings() -> None:
    if PATCH_SIZE <= 0:
        raise ValueError("PATCH_SIZE должен быть положительным.")

    if not 0 < PATCH_STRIDE <= PATCH_SIZE:
        raise ValueError(
            "PATCH_STRIDE должен находиться в диапазоне от 1 до PATCH_SIZE."
        )

    if SPLIT_GAP < 0:
        raise ValueError("SPLIT_GAP не может быть отрицательным.")

    if SPLIT_AXIS not in {"auto", "x", "y"}:
        raise ValueError('SPLIT_AXIS должен быть "auto", "x" или "y".')

    if not (
        0 < TRAIN_FRACTION < 1
        and 0 < VAL_FRACTION < 1
        and TRAIN_FRACTION + VAL_FRACTION < 1
    ):
        raise ValueError("Некорректные доли пространственного разделения.")


def make_spatial_regions(
    length: int,
) -> list[tuple[str, int, int]]:
    usable_length = length - 2 * SPLIT_GAP

    if usable_length <= 0:
        raise ValueError(
            "Размер растра недостаточен для выбранного SPLIT_GAP."
        )

    train_length = int(usable_length * TRAIN_FRACTION)
    val_length = int(usable_length * VAL_FRACTION)

    val_start = train_length + SPLIT_GAP
    test_start = val_start + val_length + SPLIT_GAP

    regions = [
        ("train", 0, train_length),
        ("val", val_start, val_start + val_length),
        ("test", test_start, length),
    ]

    if any(end - start < PATCH_SIZE for _, start, end in regions):
        raise ValueError(
            "Одна из пространственных полос меньше PATCH_SIZE. "
            "Уменьши SPLIT_GAP, измени доли или ось разделения."
        )

    return regions


def tile_dem() -> None:
    validate_tiling_settings()

    if not SOURCE_DEM.is_file():
        raise FileNotFoundError(f"Исходная DEM не найдена:\n{SOURCE_DEM}")

    records = []
    rejected = Counter()
    candidates = Counter()

    with rasterio.open(SOURCE_DEM) as src:
        if src.crs is None:
            print(
                "Предупреждение: у исходной DEM отсутствует CRS. "
                "Продолжаем нарезку по пикселям без назначения "
                "системы координат."
            )

        if min(src.width, src.height) < PATCH_SIZE:
            raise ValueError("Исходная DEM меньше размера фрагмента.")

        axis = SPLIT_AXIS
        if axis == "auto":
            axis = "x" if src.width >= src.height else "y"

        length = src.width if axis == "x" else src.height
        regions = make_spatial_regions(length)

        prepare_output_directory(PATCHES_DIR)

        write_json(
            PATCHES_DIR / "config.json",
            {
                "source": str(SOURCE_DEM),
                "patch_size": PATCH_SIZE,
                "patch_stride": PATCH_STRIDE,
                "split_gap_pixels": SPLIT_GAP,
                "split_axis": axis,
                "train_fraction": TRAIN_FRACTION,
                "val_fraction": VAL_FRACTION,
                "regions_pixels": regions,
                "source_width": src.width,
                "source_height": src.height,
                "crs": src.crs.to_string() if src.crs is not None else None,
                "resolution": list(src.res),
                "source_transform": list(src.transform)[:6],
                "source_band": 1,
                "source_scale": src.scales[0],
                "source_offset": src.offsets[0],
                "height_policy": "scale and offset applied; no normalization",
                "nodata_policy": "reject patch with any invalid pixel",
            },
        )

        print("\nЭтап 1. Нарезка DEM")
        print(f"Источник: {SOURCE_DEM}")
        print(f"Размер растра: {src.width} × {src.height}")
        print(f"Разрешение: {src.res}")
        print(f"CRS: {src.crs}")
        print(f"Ось разделения: {axis}")
        print(f"Размер окна: {PATCH_SIZE}; шаг: {PATCH_STRIDE}")

        for split, start, end in regions:
            if axis == "x":
                rows = range(
                    0,
                    src.height - PATCH_SIZE + 1,
                    PATCH_STRIDE,
                )
                cols = range(
                    start,
                    end - PATCH_SIZE + 1,
                    PATCH_STRIDE,
                )
            else:
                rows = range(
                    start,
                    end - PATCH_SIZE + 1,
                    PATCH_STRIDE,
                )
                cols = range(
                    0,
                    src.width - PATCH_SIZE + 1,
                    PATCH_STRIDE,
                )

            for row in rows:
                for col in cols:
                    candidates[split] += 1

                    window = Window(
                        col_off=col,
                        row_off=row,
                        width=PATCH_SIZE,
                        height=PATCH_SIZE,
                    )

                    band = src.read(
                        1,
                        window=window,
                        masked=True,
                        out_dtype="float64",
                    )

                    raw = np.asarray(band.data)
                    valid = (
                        ~np.ma.getmaskarray(band)
                        & np.isfinite(raw)
                    )

                    if not valid.all():
                        rejected[split] += 1
                        continue

                    with np.errstate(over="ignore", invalid="ignore"):
                        dem = (
                            raw * src.scales[0] + src.offsets[0]
                        ).astype(np.float32)

                    if not np.isfinite(dem).all():
                        rejected[split] += 1
                        continue

                    patch_id = f"r{row:07d}_c{col:07d}"
                    relative_path = Path(split) / f"{patch_id}.tif"

                    with rasterio.open(
                        PATCHES_DIR / relative_path,
                        "w",
                        driver="GTiff",
                        height=PATCH_SIZE,
                        width=PATCH_SIZE,
                        count=1,
                        dtype="float32",
                        crs=src.crs,
                        transform=src.window_transform(window),
                        compress="deflate",
                        predictor=3,
                    ) as dst:
                        dst.write(dem, 1)

                    left, bottom, right, top = rasterio.windows.bounds(
                        window,
                        src.transform,
                    )

                    records.append(
                        {
                            "patch_id": patch_id,
                            "split": split,
                            "path": relative_path.as_posix(),
                            "row": row,
                            "col": col,
                            "size": PATCH_SIZE,
                            "left": left,
                            "bottom": bottom,
                            "right": right,
                            "top": top,
                            "z_min": float(dem.min()),
                            "z_max": float(dem.max()),
                            "z_mean": float(dem.mean(dtype=np.float64)),
                            "z_std": float(dem.std(dtype=np.float64)),
                        }
                    )

    counts = Counter(record["split"] for record in records)

    for split in ("train", "val", "test"):
        print(
            f"{split:5s}: "
            f"окон={candidates[split]}, "
            f"сохранено={counts[split]}, "
            f"отброшено={rejected[split]}"
        )

    if not records:
        raise RuntimeError(
            "Нет пригодных фрагментов. Проверь NoData и настройки нарезки."
        )

    write_manifest(PATCHES_DIR / "manifest.csv", records)

    missing = {"train", "val", "test"} - set(counts)
    if missing:
        raise RuntimeError(
            f"Пустые выборки: {sorted(missing)}. "
            "Проверь расположение полос и распределение NoData. "
            "Аугментация не запущена."
        )


def augment_patches() -> None:
    manifest_path = PATCHES_DIR / "manifest.csv"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Манифест исходных фрагментов не найден:\n{manifest_path}"
        )

    if (
        not ROTATIONS
        or len(set(ROTATIONS)) != len(ROTATIONS)
        or any(angle not in (0, 90, 180, 270) for angle in ROTATIONS)
    ):
        raise ValueError(
            "ROTATIONS должен содержать уникальные углы из "
            "(0, 90, 180, 270)."
        )

    if (
        not MIRROR_OPTIONS
        or len(set(MIRROR_OPTIONS)) != len(MIRROR_OPTIONS)
        or any(type(value) is not bool for value in MIRROR_OPTIONS)
    ):
        raise ValueError(
            "MIRROR_OPTIONS должен содержать уникальные "
            "булевы значения."
        )

    with manifest_path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        source_records = list(csv.DictReader(file))

    if not source_records:
        raise ValueError("Исходный манифест пуст.")

    splits = {record["split"] for record in source_records}
    if splits != {"train", "val", "test"}:
        raise ValueError(
            "В манифесте должны присутствовать только "
            "непустые train, val и test."
        )

    prepare_output_directory(AUGMENTED_DIR)

    records = []

    print("\nЭтап 2. Аугментация фрагментов")

    for record in source_records:
        split = record["split"]
        patch_id = record["patch_id"]

        with rasterio.open(PATCHES_DIR / record["path"]) as src:
            dem = src.read(1, out_dtype="float32")

        if dem.shape != (PATCH_SIZE, PATCH_SIZE):
            raise ValueError(
                f"Неожиданный размер {dem.shape}: {record['path']}. "
                f"Ожидается {(PATCH_SIZE, PATCH_SIZE)}."
            )

        if not np.isfinite(dem).all():
            raise ValueError(
                f"Невалидные значения в фрагменте: {record['path']}"
            )

        if split == "train":
            operations = [
                (angle, mirror)
                for angle in ROTATIONS
                for mirror in MIRROR_OPTIONS
            ]
        else:
            operations = [(0, False)]

        for angle, mirror in operations:
            sample = np.rot90(dem, k=angle // 90)

            if mirror:
                sample = np.fliplr(sample)

            sample = np.ascontiguousarray(sample, dtype=np.float32)

            operation = f"rot{angle:03d}_flip{int(mirror)}"
            sample_id = f"{patch_id}_{operation}"
            relative_path = Path(split) / f"{sample_id}.npy"

            np.save(
                AUGMENTED_DIR / relative_path,
                sample,
                allow_pickle=False,
            )

            records.append(
                {
                    **record,
                    "sample_id": sample_id,
                    "path": relative_path.as_posix(),
                    "source_path": record["path"],
                    "rotation_ccw_deg": angle,
                    "flip_lr_after_rotation": int(mirror),
                }
            )

    write_manifest(AUGMENTED_DIR / "manifest.csv", records)

    write_json(
        AUGMENTED_DIR / "config.json",
        {
            "source": str(PATCHES_DIR),
            "patch_size": PATCH_SIZE,
            "train_rotations_ccw_deg": list(ROTATIONS),
            "train_mirror_options": list(MIRROR_OPTIONS),
            "val_test_augmentation": False,
            "array_shape": [PATCH_SIZE, PATCH_SIZE],
            "dtype": "float32",
            "normalization": None,
            "georeferencing": (
                "Manifest bounds describe the original patch, "
                "not the orientation of the augmented array."
            ),
        },
    )

    counts = Counter(record["split"] for record in records)

    for split in ("train", "val", "test"):
        print(f"{split:5s}: сохранено={counts[split]}")


def main() -> None:
    if not RUN_TILING and not RUN_AUGMENTATION:
        print("Оба этапа отключены. Измени настройки в начале файла.")
        return

    if RUN_TILING:
        tile_dem()

    if RUN_AUGMENTATION:
        augment_patches()

    print("\nПодготовка данных завершена.")


if __name__ == "__main__":
    main()