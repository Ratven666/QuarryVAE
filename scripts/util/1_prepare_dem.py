# from __future__ import annotations
#
# import csv
# import json
# from collections import Counter
# from pathlib import Path
#
# import numpy as np
# import rasterio
# from rasterio.windows import Window
#
#
# # ============================= CONFIG =============================
#
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
#
# SOURCE_DEM = PROJECT_ROOT / "data/raw/1m/grib_1m.tif"
#
# PATCHES_DIR = PROJECT_ROOT / "data/interim/patches_64"
# AUGMENTED_DIR = PROJECT_ROOT / "data/processed/patches_64_d4"
#
# # Какие этапы выполнять.
# RUN_TILING = True
# RUN_AUGMENTATION = True
#
# # Размер и шаг окна в пикселях.
# PATCH_SIZE = 64
# PATCH_STRIDE = 64
#
# # Пространственное разделение.
# # "auto" — вдоль большей размерности растра, либо явно "x" / "y".
# SPLIT_AXIS = "auto"
#
# TRAIN_FRACTION = 0.70
# VAL_FRACTION = 0.15
# # Остальная часть отводится под test.
#
# # Ширина исключаемой полосы между train / val и val / test.
# SPLIT_GAP = 256
#
# # Аугментация применяется только к train.
# ROTATIONS = (0, 90, 180, 270)
# MIRROR_OPTIONS = (False, True)
#
# # ==================================================================
#
#
# def prepare_output_directory(path: Path) -> None:
#     """Не допускает смешивания результатов разных запусков."""
#     if path.exists() and any(path.iterdir()):
#         raise FileExistsError(
#             f"Каталог результатов не пуст:\n{path}\n"
#             "Укажи новый путь или предварительно удали старые результаты."
#         )
#
#     path.mkdir(parents=True, exist_ok=True)
#
#     for split in ("train", "val", "test"):
#         (path / split).mkdir(exist_ok=True)
#
#
# def write_json(path: Path, data: dict) -> None:
#     path.write_text(
#         json.dumps(data, indent=2, ensure_ascii=False),
#         encoding="utf-8",
#     )
#
#
# def write_manifest(path: Path, records: list[dict]) -> None:
#     if not records:
#         raise RuntimeError("Не получено ни одного образца.")
#
#     with path.open("w", newline="", encoding="utf-8") as file:
#         writer = csv.DictWriter(file, fieldnames=list(records[0]))
#         writer.writeheader()
#         writer.writerows(records)
#
#
# def validate_tiling_settings() -> None:
#     if PATCH_SIZE <= 0:
#         raise ValueError("PATCH_SIZE должен быть положительным.")
#
#     if not 0 < PATCH_STRIDE <= PATCH_SIZE:
#         raise ValueError(
#             "PATCH_STRIDE должен находиться в диапазоне от 1 до PATCH_SIZE."
#         )
#
#     if SPLIT_GAP < 0:
#         raise ValueError("SPLIT_GAP не может быть отрицательным.")
#
#     if SPLIT_AXIS not in {"auto", "x", "y"}:
#         raise ValueError('SPLIT_AXIS должен быть "auto", "x" или "y".')
#
#     if not (
#         0 < TRAIN_FRACTION < 1
#         and 0 < VAL_FRACTION < 1
#         and TRAIN_FRACTION + VAL_FRACTION < 1
#     ):
#         raise ValueError("Некорректные доли пространственного разделения.")
#
#
# def make_spatial_regions(
#     length: int,
# ) -> list[tuple[str, int, int]]:
#     usable_length = length - 2 * SPLIT_GAP
#
#     if usable_length <= 0:
#         raise ValueError(
#             "Размер растра недостаточен для выбранного SPLIT_GAP."
#         )
#
#     train_length = int(usable_length * TRAIN_FRACTION)
#     val_length = int(usable_length * VAL_FRACTION)
#
#     val_start = train_length + SPLIT_GAP
#     test_start = val_start + val_length + SPLIT_GAP
#
#     regions = [
#         ("train", 0, train_length),
#         ("val", val_start, val_start + val_length),
#         ("test", test_start, length),
#     ]
#
#     if any(end - start < PATCH_SIZE for _, start, end in regions):
#         raise ValueError(
#             "Одна из пространственных полос меньше PATCH_SIZE. "
#             "Уменьши SPLIT_GAP, измени доли или ось разделения."
#         )
#
#     return regions
#
#
# def tile_dem() -> None:
#     validate_tiling_settings()
#
#     if not SOURCE_DEM.is_file():
#         raise FileNotFoundError(f"Исходная DEM не найдена:\n{SOURCE_DEM}")
#
#     records = []
#     rejected = Counter()
#     candidates = Counter()
#
#     with rasterio.open(SOURCE_DEM) as src:
#         if src.crs is None:
#             print(
#                 "Предупреждение: у исходной DEM отсутствует CRS. "
#                 "Продолжаем нарезку по пикселям без назначения "
#                 "системы координат."
#             )
#
#         if min(src.width, src.height) < PATCH_SIZE:
#             raise ValueError("Исходная DEM меньше размера фрагмента.")
#
#         axis = SPLIT_AXIS
#         if axis == "auto":
#             axis = "x" if src.width >= src.height else "y"
#
#         length = src.width if axis == "x" else src.height
#         regions = make_spatial_regions(length)
#
#         prepare_output_directory(PATCHES_DIR)
#
#         write_json(
#             PATCHES_DIR / "config.json",
#             {
#                 "source": str(SOURCE_DEM),
#                 "patch_size": PATCH_SIZE,
#                 "patch_stride": PATCH_STRIDE,
#                 "split_gap_pixels": SPLIT_GAP,
#                 "split_axis": axis,
#                 "train_fraction": TRAIN_FRACTION,
#                 "val_fraction": VAL_FRACTION,
#                 "regions_pixels": regions,
#                 "source_width": src.width,
#                 "source_height": src.height,
#                 "crs": src.crs.to_string() if src.crs is not None else None,
#                 "resolution": list(src.res),
#                 "source_transform": list(src.transform)[:6],
#                 "source_band": 1,
#                 "source_scale": src.scales[0],
#                 "source_offset": src.offsets[0],
#                 "height_policy": "scale and offset applied; no normalization",
#                 "nodata_policy": "reject patch with any invalid pixel",
#             },
#         )
#
#         print("\nЭтап 1. Нарезка DEM")
#         print(f"Источник: {SOURCE_DEM}")
#         print(f"Размер растра: {src.width} × {src.height}")
#         print(f"Разрешение: {src.res}")
#         print(f"CRS: {src.crs}")
#         print(f"Ось разделения: {axis}")
#         print(f"Размер окна: {PATCH_SIZE}; шаг: {PATCH_STRIDE}")
#
#         for split, start, end in regions:
#             if axis == "x":
#                 rows = range(
#                     0,
#                     src.height - PATCH_SIZE + 1,
#                     PATCH_STRIDE,
#                 )
#                 cols = range(
#                     start,
#                     end - PATCH_SIZE + 1,
#                     PATCH_STRIDE,
#                 )
#             else:
#                 rows = range(
#                     start,
#                     end - PATCH_SIZE + 1,
#                     PATCH_STRIDE,
#                 )
#                 cols = range(
#                     0,
#                     src.width - PATCH_SIZE + 1,
#                     PATCH_STRIDE,
#                 )
#
#             for row in rows:
#                 for col in cols:
#                     candidates[split] += 1
#
#                     window = Window(
#                         col_off=col,
#                         row_off=row,
#                         width=PATCH_SIZE,
#                         height=PATCH_SIZE,
#                     )
#
#                     band = src.read(
#                         1,
#                         window=window,
#                         masked=True,
#                         out_dtype="float64",
#                     )
#
#                     raw = np.asarray(band.data)
#                     valid = (
#                         ~np.ma.getmaskarray(band)
#                         & np.isfinite(raw)
#                     )
#
#                     if not valid.all():
#                         rejected[split] += 1
#                         continue
#
#                     with np.errstate(over="ignore", invalid="ignore"):
#                         dem = (
#                             raw * src.scales[0] + src.offsets[0]
#                         ).astype(np.float32)
#
#                     if not np.isfinite(dem).all():
#                         rejected[split] += 1
#                         continue
#
#                     patch_id = f"r{row:07d}_c{col:07d}"
#                     relative_path = Path(split) / f"{patch_id}.tif"
#
#                     with rasterio.open(
#                         PATCHES_DIR / relative_path,
#                         "w",
#                         driver="GTiff",
#                         height=PATCH_SIZE,
#                         width=PATCH_SIZE,
#                         count=1,
#                         dtype="float32",
#                         crs=src.crs,
#                         transform=src.window_transform(window),
#                         compress="deflate",
#                         predictor=3,
#                     ) as dst:
#                         dst.write(dem, 1)
#
#                     left, bottom, right, top = rasterio.windows.bounds(
#                         window,
#                         src.transform,
#                     )
#
#                     records.append(
#                         {
#                             "patch_id": patch_id,
#                             "split": split,
#                             "path": relative_path.as_posix(),
#                             "row": row,
#                             "col": col,
#                             "size": PATCH_SIZE,
#                             "left": left,
#                             "bottom": bottom,
#                             "right": right,
#                             "top": top,
#                             "z_min": float(dem.min()),
#                             "z_max": float(dem.max()),
#                             "z_mean": float(dem.mean(dtype=np.float64)),
#                             "z_std": float(dem.std(dtype=np.float64)),
#                         }
#                     )
#
#     counts = Counter(record["split"] for record in records)
#
#     for split in ("train", "val", "test"):
#         print(
#             f"{split:5s}: "
#             f"окон={candidates[split]}, "
#             f"сохранено={counts[split]}, "
#             f"отброшено={rejected[split]}"
#         )
#
#     if not records:
#         raise RuntimeError(
#             "Нет пригодных фрагментов. Проверь NoData и настройки нарезки."
#         )
#
#     write_manifest(PATCHES_DIR / "manifest.csv", records)
#
#     missing = {"train", "val", "test"} - set(counts)
#     if missing:
#         raise RuntimeError(
#             f"Пустые выборки: {sorted(missing)}. "
#             "Проверь расположение полос и распределение NoData. "
#             "Аугментация не запущена."
#         )
#
#
# def augment_patches() -> None:
#     manifest_path = PATCHES_DIR / "manifest.csv"
#
#     if not manifest_path.is_file():
#         raise FileNotFoundError(
#             f"Манифест исходных фрагментов не найден:\n{manifest_path}"
#         )
#
#     if (
#         not ROTATIONS
#         or len(set(ROTATIONS)) != len(ROTATIONS)
#         or any(angle not in (0, 90, 180, 270) for angle in ROTATIONS)
#     ):
#         raise ValueError(
#             "ROTATIONS должен содержать уникальные углы из "
#             "(0, 90, 180, 270)."
#         )
#
#     if (
#         not MIRROR_OPTIONS
#         or len(set(MIRROR_OPTIONS)) != len(MIRROR_OPTIONS)
#         or any(type(value) is not bool for value in MIRROR_OPTIONS)
#     ):
#         raise ValueError(
#             "MIRROR_OPTIONS должен содержать уникальные "
#             "булевы значения."
#         )
#
#     with manifest_path.open(
#         newline="",
#         encoding="utf-8",
#     ) as file:
#         source_records = list(csv.DictReader(file))
#
#     if not source_records:
#         raise ValueError("Исходный манифест пуст.")
#
#     splits = {record["split"] for record in source_records}
#     if splits != {"train", "val", "test"}:
#         raise ValueError(
#             "В манифесте должны присутствовать только "
#             "непустые train, val и test."
#         )
#
#     prepare_output_directory(AUGMENTED_DIR)
#
#     records = []
#
#     print("\nЭтап 2. Аугментация фрагментов")
#
#     for record in source_records:
#         split = record["split"]
#         patch_id = record["patch_id"]
#
#         with rasterio.open(PATCHES_DIR / record["path"]) as src:
#             dem = src.read(1, out_dtype="float32")
#
#         if dem.shape != (PATCH_SIZE, PATCH_SIZE):
#             raise ValueError(
#                 f"Неожиданный размер {dem.shape}: {record['path']}. "
#                 f"Ожидается {(PATCH_SIZE, PATCH_SIZE)}."
#             )
#
#         if not np.isfinite(dem).all():
#             raise ValueError(
#                 f"Невалидные значения в фрагменте: {record['path']}"
#             )
#
#         if split == "train":
#             operations = [
#                 (angle, mirror)
#                 for angle in ROTATIONS
#                 for mirror in MIRROR_OPTIONS
#             ]
#         else:
#             operations = [(0, False)]
#
#         for angle, mirror in operations:
#             sample = np.rot90(dem, k=angle // 90)
#
#             if mirror:
#                 sample = np.fliplr(sample)
#
#             sample = np.ascontiguousarray(sample, dtype=np.float32)
#
#             operation = f"rot{angle:03d}_flip{int(mirror)}"
#             sample_id = f"{patch_id}_{operation}"
#             relative_path = Path(split) / f"{sample_id}.npy"
#
#             np.save(
#                 AUGMENTED_DIR / relative_path,
#                 sample,
#                 allow_pickle=False,
#             )
#
#             records.append(
#                 {
#                     **record,
#                     "sample_id": sample_id,
#                     "path": relative_path.as_posix(),
#                     "source_path": record["path"],
#                     "rotation_ccw_deg": angle,
#                     "flip_lr_after_rotation": int(mirror),
#                 }
#             )
#
#     write_manifest(AUGMENTED_DIR / "manifest.csv", records)
#
#     write_json(
#         AUGMENTED_DIR / "config.json",
#         {
#             "source": str(PATCHES_DIR),
#             "patch_size": PATCH_SIZE,
#             "train_rotations_ccw_deg": list(ROTATIONS),
#             "train_mirror_options": list(MIRROR_OPTIONS),
#             "val_test_augmentation": False,
#             "array_shape": [PATCH_SIZE, PATCH_SIZE],
#             "dtype": "float32",
#             "normalization": None,
#             "georeferencing": (
#                 "Manifest bounds describe the original patch, "
#                 "not the orientation of the augmented array."
#             ),
#         },
#     )
#
#     counts = Counter(record["split"] for record in records)
#
#     for split in ("train", "val", "test"):
#         print(f"{split:5s}: сохранено={counts[split]}")
#
#
# def main() -> None:
#     if not RUN_TILING and not RUN_AUGMENTATION:
#         print("Оба этапа отключены. Измени настройки в начале файла.")
#         return
#
#     if RUN_TILING:
#         tile_dem()
#
#     if RUN_AUGMENTATION:
#         augment_patches()
#
#     print("\nПодготовка данных завершена.")
#
#
# if __name__ == "__main__":
#     main()

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


# ============================= CONFIG =============================


def find_project_root() -> Path:
    """Ищет корень проекта по pyproject.toml."""
    script_path = Path(__file__).resolve()

    for parent in script_path.parents:
        if (parent / "pyproject.toml").is_file():
            return parent

    raise FileNotFoundError(
        "Не найден pyproject.toml в родительских каталогах скрипта. "
        "Укажи PROJECT_ROOT вручную."
    )


PROJECT_ROOT = find_project_root()

# Добавь сюда все исходные DEM.
SOURCE_DEMS = [
    PROJECT_ROOT / "data/raw/1m/grib_1m.tif",
    PROJECT_ROOT / "data/derived/dem_from_densified_breaklines_1m.tif",

    # Примеры дополнительных источников:
    # PROJECT_ROOT / "data/raw/1m/quarry_02.tif",
    # PROJECT_ROOT / "data/derived/dem_from_densified_breaklines_1m.tif",
]

PATCHES_DIR = PROJECT_ROOT / "data/interim/patches_64_multi"
AUGMENTED_DIR = PROJECT_ROOT / "data/processed/patches_64_multi_d4"

RUN_TILING = True
RUN_AUGMENTATION = True

PATCH_SIZE = 64
PATCH_STRIDE = 64

# "auto", "x", "y".
SPLIT_AXIS = "auto"

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15

# Буферы между пространственными полосами, в пикселях.
SPLIT_GAP = 256

ROTATIONS = (0, 90, 180, 270)
MIRROR_OPTIONS = (False, True)

SOURCE_BAND = 1

# Дополнительный NoData для всех исходных файлов.
# NoData и маски, прописанные в TIFF, учитываются автоматически.
EXTRA_NODATA_VALUE = None

# None — не проверять разрешение.
# Если нужен единый физический масштаб, оставь (1.0, 1.0).
EXPECTED_RESOLUTION = (1.0, 1.0)
RESOLUTION_TOLERANCE = 1e-6

PRINT_EVERY_WINDOWS = 1000

# ==================================================================


SPLITS = ("train", "val", "test")


def write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def write_manifest(path: Path, records: list[dict]) -> None:
    if not records:
        raise RuntimeError("Не получено ни одного образца.")

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0]),
        )
        writer.writeheader()
        writer.writerows(records)


def ensure_empty_output(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(path)

        if any(path.iterdir()):
            raise FileExistsError(
                f"Каталог результатов не пуст:\n{path}\n"
                "Укажи новый каталог или удали старые результаты."
            )


def prepare_output_directory(path: Path) -> None:
    ensure_empty_output(path)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split in SPLITS:
        (path / split).mkdir(exist_ok=True)


def validate_tiling_settings() -> None:
    if not SOURCE_DEMS:
        raise ValueError("SOURCE_DEMS не должен быть пустым.")

    if not isinstance(PATCH_SIZE, int) or PATCH_SIZE <= 0:
        raise ValueError("PATCH_SIZE должен быть целым > 0.")

    if (
        not isinstance(PATCH_STRIDE, int)
        or not 0 < PATCH_STRIDE <= PATCH_SIZE
    ):
        raise ValueError(
            "PATCH_STRIDE должен быть целым от 1 до PATCH_SIZE."
        )

    if not isinstance(SPLIT_GAP, int) or SPLIT_GAP < 0:
        raise ValueError("SPLIT_GAP должен быть целым >= 0.")

    if SPLIT_AXIS not in {"auto", "x", "y"}:
        raise ValueError("SPLIT_AXIS: auto/x/y.")

    if not (
        0 < TRAIN_FRACTION < 1
        and 0 < VAL_FRACTION < 1
        and TRAIN_FRACTION + VAL_FRACTION < 1
    ):
        raise ValueError("Некорректные доли train/val.")

    if not isinstance(SOURCE_BAND, int) or SOURCE_BAND < 1:
        raise ValueError("SOURCE_BAND должен быть целым >= 1.")

    if PRINT_EVERY_WINDOWS < 1:
        raise ValueError("PRINT_EVERY_WINDOWS должен быть >= 1.")

    if EXPECTED_RESOLUTION is not None:
        expected = np.asarray(
            EXPECTED_RESOLUTION,
            dtype=np.float64,
        )

        if (
            expected.shape != (2,)
            or not np.isfinite(expected).all()
            or (expected <= 0).any()
        ):
            raise ValueError(
                "EXPECTED_RESOLUTION: два положительных числа или None."
            )

    if RESOLUTION_TOLERANCE < 0:
        raise ValueError("Некорректный RESOLUTION_TOLERANCE.")


def validate_augmentation_settings() -> None:
    if (
        not ROTATIONS
        or len(set(ROTATIONS)) != len(ROTATIONS)
        or any(
            angle not in (0, 90, 180, 270)
            for angle in ROTATIONS
        )
    ):
        raise ValueError(
            "ROTATIONS: уникальные углы из (0, 90, 180, 270)."
        )

    if (
        not MIRROR_OPTIONS
        or len(set(MIRROR_OPTIONS)) != len(MIRROR_OPTIONS)
        or any(
            type(value) is not bool
            for value in MIRROR_OPTIONS
        )
    ):
        raise ValueError(
            "MIRROR_OPTIONS: уникальные значения False/True."
        )


def resolve_sources() -> list[Path]:
    sources = []
    seen = set()

    for source in SOURCE_DEMS:
        path = Path(source).expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"DEM не найдена:\n{path}"
            )

        if path in seen:
            raise ValueError(
                f"Одна DEM указана повторно:\n{path}"
            )

        seen.add(path)
        sources.append(path)

    return sources


def make_source_id(path: Path, index: int) -> str:
    safe_stem = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        path.stem,
    ).strip("_")

    safe_stem = safe_stem[:48] or "dem"

    path_hash = hashlib.blake2b(
        str(path).encode("utf-8"),
        digest_size=4,
    ).hexdigest()

    return f"dem{index:03d}_{safe_stem}_{path_hash}"


def make_spatial_regions(
    length: int,
) -> list[tuple[str, int, int]]:
    usable_length = length - 2 * SPLIT_GAP

    if usable_length <= 0:
        raise ValueError(
            "Размер DEM недостаточен для SPLIT_GAP."
        )

    train_length = int(
        usable_length * TRAIN_FRACTION
    )
    val_length = int(
        usable_length * VAL_FRACTION
    )

    val_start = train_length + SPLIT_GAP
    test_start = val_start + val_length + SPLIT_GAP

    regions = [
        ("train", 0, train_length),
        ("val", val_start, val_start + val_length),
        ("test", test_start, length),
    ]

    if any(
        end - start < PATCH_SIZE
        for _, start, end in regions
    ):
        raise ValueError(
            "Одна из пространственных полос меньше PATCH_SIZE. "
            "Уменьши SPLIT_GAP, измени доли или ось разделения."
        )

    return regions


def inspect_sources(sources: list[Path]) -> list[dict]:
    """Проверяет все источники до создания выходных файлов."""
    descriptors = []

    for index, source in enumerate(sources, start=1):
        with rasterio.open(source) as src:
            if src.count < SOURCE_BAND:
                raise ValueError(
                    f"В {source.name} отсутствует канал {SOURCE_BAND}."
                )

            if min(src.width, src.height) < PATCH_SIZE:
                raise ValueError(
                    f"DEM {source.name} меньше PATCH_SIZE."
                )

            if EXPECTED_RESOLUTION is not None:
                if not np.allclose(
                    src.res,
                    EXPECTED_RESOLUTION,
                    atol=RESOLUTION_TOLERANCE,
                    rtol=0,
                ):
                    raise ValueError(
                        f"Неверное разрешение {source.name}: "
                        f"{src.res}, ожидается {EXPECTED_RESOLUTION}. "
                        "Пересэмплируй DEM отдельно либо "
                        "задай EXPECTED_RESOLUTION=None."
                    )

            scale = float(
                src.scales[SOURCE_BAND - 1]
            )
            offset = float(
                src.offsets[SOURCE_BAND - 1]
            )

            if not np.isfinite([scale, offset]).all():
                raise ValueError(
                    f"Некорректные scale/offset: {source.name}."
                )

            axis = SPLIT_AXIS

            if axis == "auto":
                axis = (
                    "x"
                    if src.width >= src.height
                    else "y"
                )

            length = (
                src.width
                if axis == "x"
                else src.height
            )

            try:
                regions = make_spatial_regions(length)
            except ValueError as error:
                raise ValueError(
                    f"{source.name}: {error}"
                ) from error

            descriptors.append({
                "source_id": make_source_id(source, index),
                "source_path": str(source),
                "width": src.width,
                "height": src.height,
                "crs": (
                    src.crs.to_string()
                    if src.crs is not None
                    else None
                ),
                "resolution": list(src.res),
                "transform": list(src.transform)[:6],
                "band": SOURCE_BAND,
                "scale": scale,
                "offset": offset,
                "split_axis": axis,
                "regions_pixels": regions,
            })

    return descriptors


def window_bounds(
    transform,
    row: int,
    col: int,
) -> tuple[float, float, float, float]:
    # Ограничивающий прямоугольник по четырём углам окна,
    # в том числе для повёрнутой сетки.
    corners = [
        transform * (col, row),
        transform * (col + PATCH_SIZE, row),
        transform * (col, row + PATCH_SIZE),
        transform * (col + PATCH_SIZE, row + PATCH_SIZE),
    ]

    x, y = np.asarray(corners).T

    return (
        float(x.min()),
        float(y.min()),
        float(x.max()),
        float(y.max()),
    )


def tile_dem() -> None:
    validate_tiling_settings()

    sources = resolve_sources()
    descriptors = inspect_sources(sources)

    # Все файлы проверены; теперь создаём результаты.
    prepare_output_directory(PATCHES_DIR)

    records = []
    source_summaries = []

    print("\nЭтап 1. Нарезка нескольких DEM")
    print(f"Источников: {len(descriptors)}")
    print(f"Размер окна: {PATCH_SIZE}")
    print(f"Шаг: {PATCH_STRIDE}")

    for descriptor in descriptors:
        source = Path(descriptor["source_path"])
        source_id = descriptor["source_id"]
        axis = descriptor["split_axis"]

        candidates = Counter()
        rejected = Counter()
        saved = Counter()

        print(f"\nИсточник: {source.name}")
        print(f"ID: {source_id}")
        print(
            f"Размер: {descriptor['width']} × "
            f"{descriptor['height']}"
        )
        print(f"Разрешение: {descriptor['resolution']}")
        print(f"Ось разделения: {axis}")

        if descriptor["crs"] is None:
            print(
                "CRS отсутствует. "
                "Нарезка выполняется без назначения CRS."
            )

        with rasterio.open(source) as src:
            for split, start, end in descriptor["regions_pixels"]:
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
                            SOURCE_BAND,
                            window=window,
                            masked=True,
                            out_dtype="float64",
                        )

                        raw = np.asarray(band.data)

                        valid = (
                            ~np.ma.getmaskarray(band)
                            & np.isfinite(raw)
                        )

                        if EXTRA_NODATA_VALUE is not None:
                            valid &= raw != EXTRA_NODATA_VALUE

                        if not valid.all():
                            rejected[split] += 1
                        else:
                            with np.errstate(
                                over="ignore",
                                invalid="ignore",
                            ):
                                dem = (
                                    raw * descriptor["scale"]
                                    + descriptor["offset"]
                                ).astype(np.float32)

                            if not np.isfinite(dem).all():
                                rejected[split] += 1
                            else:
                                patch_id = (
                                    f"{source_id}_"
                                    f"r{row:07d}_c{col:07d}"
                                )

                                relative_path = (
                                    Path(split) / f"{patch_id}.tif"
                                )

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

                                left, bottom, right, top = window_bounds(
                                    src.transform,
                                    row,
                                    col,
                                )

                                records.append({
                                    "patch_id": patch_id,
                                    "source_id": source_id,
                                    "source_path": str(source),
                                    "split": split,
                                    "path": relative_path.as_posix(),
                                    "row": row,
                                    "col": col,
                                    "size": PATCH_SIZE,
                                    "resolution_x": float(src.res[0]),
                                    "resolution_y": float(src.res[1]),
                                    "left": left,
                                    "bottom": bottom,
                                    "right": right,
                                    "top": top,
                                    "z_min": float(dem.min()),
                                    "z_max": float(dem.max()),
                                    "z_mean": float(
                                        dem.mean(dtype=np.float64)
                                    ),
                                    "z_std": float(
                                        dem.std(dtype=np.float64)
                                    ),
                                })

                                saved[split] += 1

                        if candidates[split] % PRINT_EVERY_WINDOWS == 0:
                            print(
                                f"  {split}: "
                                f"проверено={candidates[split]:,}, "
                                f"сохранено={saved[split]:,}"
                            )

        for split in SPLITS:
            print(
                f"  {split:5s}: "
                f"окон={candidates[split]}, "
                f"сохранено={saved[split]}, "
                f"отброшено={rejected[split]}"
            )

        missing_local = [
            split
            for split in SPLITS
            if saved[split] == 0
        ]

        if missing_local:
            print(
                "  Предупреждение: у этого источника пустые выборки: "
                f"{missing_local}"
            )

        source_summaries.append({
            **descriptor,
            "candidates": {
                split: candidates[split]
                for split in SPLITS
            },
            "saved": {
                split: saved[split]
                for split in SPLITS
            },
            "rejected": {
                split: rejected[split]
                for split in SPLITS
            },
        })

    total_counts = Counter(
        record["split"]
        for record in records
    )

    write_json(
        PATCHES_DIR / "config.json",
        {
            "sources": source_summaries,
            "patch_size": PATCH_SIZE,
            "patch_stride": PATCH_STRIDE,
            "split_gap_pixels": SPLIT_GAP,
            "train_fraction": TRAIN_FRACTION,
            "val_fraction": VAL_FRACTION,
            "split_policy": "spatial strips separately within each DEM",
            "expected_resolution": EXPECTED_RESOLUTION,
            "extra_nodata_value": EXTRA_NODATA_VALUE,
            "height_policy": (
                "band scale/offset applied; no normalization"
            ),
            "nodata_policy": "reject any patch with invalid pixels",
            "total_counts": {
                split: total_counts[split]
                for split in SPLITS
            },
        },
    )

    if not records:
        raise RuntimeError(
            "Не получено ни одного пригодного фрагмента."
        )

    write_manifest(
        PATCHES_DIR / "manifest.csv",
        records,
    )

    print("\nОбщий результат нарезки:")

    for split in SPLITS:
        print(
            f"  {split:5s}: {total_counts[split]:,}"
        )

    missing = [
        split
        for split in SPLITS
        if total_counts[split] == 0
    ]

    if missing:
        raise RuntimeError(
            f"В объединённом наборе пустые выборки: {missing}. "
            "Аугментация не запущена."
        )


def augment_patches() -> None:
    validate_augmentation_settings()

    manifest_path = PATCHES_DIR / "manifest.csv"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Манифест не найден:\n{manifest_path}"
        )

    with manifest_path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        source_records = list(csv.DictReader(file))

    if not source_records:
        raise ValueError("Исходный манифест пуст.")

    required_fields = {
        "patch_id",
        "source_id",
        "source_path",
        "split",
        "path",
        "size",
    }

    if not required_fields.issubset(source_records[0]):
        raise ValueError(
            "Манифест имеет старый или несовместимый формат. "
            "Сначала выполни нарезку этой версией скрипта."
        )

    available_splits = {
        record["split"]
        for record in source_records
    }

    if available_splits != set(SPLITS):
        raise ValueError(
            "Нужны непустые train, val, test."
        )

    patch_ids = [
        record["patch_id"]
        for record in source_records
    ]

    if len(patch_ids) != len(set(patch_ids)):
        raise ValueError(
            "В манифесте повторяются patch_id."
        )

    # Проверяем пути и размеры по манифесту до создания результатов.
    for record in source_records:
        if int(record["size"]) != PATCH_SIZE:
            raise ValueError(
                f"Размер фрагментов в манифесте не равен {PATCH_SIZE}."
            )

        if not (PATCHES_DIR / record["path"]).is_file():
            raise FileNotFoundError(
                PATCHES_DIR / record["path"]
            )

    prepare_output_directory(AUGMENTED_DIR)

    train_operations = [
        (angle, mirror)
        for angle in ROTATIONS
        for mirror in MIRROR_OPTIONS
    ]

    records = []

    print("\nЭтап 2. Аугментация объединённого набора")

    for index, record in enumerate(
        source_records,
        start=1,
    ):
        split = record["split"]
        patch_id = record["patch_id"]

        with rasterio.open(
            PATCHES_DIR / record["path"]
        ) as src:
            band = src.read(
                1,
                masked=True,
                out_dtype="float32",
            )

            dem = np.asarray(band.data)

            if dem.shape != (PATCH_SIZE, PATCH_SIZE):
                raise ValueError(
                    f"Неожиданный размер {dem.shape}: "
                    f"{record['path']}"
                )

            if (
                np.ma.getmaskarray(band).any()
                or not np.isfinite(dem).all()
            ):
                raise ValueError(
                    f"Невалидные значения: {record['path']}"
                )

        operations = (
            train_operations
            if split == "train"
            else [(0, False)]
        )

        for angle, mirror in operations:
            sample = np.rot90(
                dem,
                k=angle // 90,
            )

            if mirror:
                sample = np.fliplr(sample)

            sample = np.ascontiguousarray(
                sample,
                dtype=np.float32,
            )

            operation = (
                f"rot{angle:03d}_flip{int(mirror)}"
            )
            sample_id = f"{patch_id}_{operation}"

            relative_path = (
                Path(split) / f"{sample_id}.npy"
            )

            np.save(
                AUGMENTED_DIR / relative_path,
                sample,
                allow_pickle=False,
            )

            records.append({
                **record,
                "sample_id": sample_id,
                "path": relative_path.as_posix(),
                "source_patch_path": record["path"],
                "rotation_ccw_deg": angle,
                "flip_lr_after_rotation": int(mirror),
            })

        if index % PRINT_EVERY_WINDOWS == 0:
            print(
                f"Обработано исходных фрагментов: "
                f"{index:,}/{len(source_records):,}"
            )

    write_manifest(
        AUGMENTED_DIR / "manifest.csv",
        records,
    )

    counts = Counter(
        record["split"]
        for record in records
    )

    counts_by_source = {}

    for record in records:
        source_id = record["source_id"]

        if source_id not in counts_by_source:
            counts_by_source[source_id] = {
                split: 0
                for split in SPLITS
            }

        counts_by_source[source_id][record["split"]] += 1

    write_json(
        AUGMENTED_DIR / "config.json",
        {
            "source_patches_dir": str(PATCHES_DIR),
            "source_manifest": str(manifest_path),
            "patch_size": PATCH_SIZE,
            "train_rotations_ccw_deg": list(ROTATIONS),
            "train_mirror_options": list(MIRROR_OPTIONS),
            "val_test_augmentation": False,
            "array_shape": [PATCH_SIZE, PATCH_SIZE],
            "dtype": "float32",
            "normalization": None,
            "total_counts": {
                split: counts[split]
                for split in SPLITS
            },
            "counts_by_source": counts_by_source,
            "georeferencing": (
                "Manifest bounds describe the original patch, "
                "not the orientation of the augmented array."
            ),
        },
    )

    print("\nОбщий результат аугментации:")

    for split in SPLITS:
        print(f"  {split:5s}: {counts[split]:,}")


def main() -> None:
    if not RUN_TILING and not RUN_AUGMENTATION:
        print("Оба этапа отключены.")
        return

    # Если нужны оба этапа, проверяем всё доступное заранее.
    if RUN_AUGMENTATION:
        validate_augmentation_settings()
        ensure_empty_output(AUGMENTED_DIR)

    if RUN_TILING:
        tile_dem()

    if RUN_AUGMENTATION:
        augment_patches()

    print("\nПодготовка объединённого набора завершена.")
    print(f"Фрагменты GeoTIFF: {PATCHES_DIR}")
    print(f"Массивы для обучения: {AUGMENTED_DIR}")


if __name__ == "__main__":
    main()