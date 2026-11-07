from __future__ import annotations

import json
import math
import tempfile
import time
from pathlib import Path

import laspy
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin


# ============================= CONFIG =============================


def find_project_root() -> Path:
    script_path = Path(__file__).resolve()

    for parent in script_path.parents:
        if (parent / "pyproject.toml").is_file():
            return parent

    raise FileNotFoundError(
        "Не найден pyproject.toml. Укажи PROJECT_ROOT вручную."
    )


PROJECT_ROOT = find_project_root()

INPUT_LAZ = PROJECT_ROOT / "data/raw/sever_quarry.las"

OUTPUT_DEM = PROJECT_ROOT / "data/derived/dem_sever_quarry.tif"

# Размер квадратной ячейки в единицах исходных XY.
# При метрических координатах 1.0 означает 1 метр.
RESOLUTION = 1.0

# None — читать CRS из LAS/LAZ.
# Если CRS в файле отсутствует, результат сохранится без CRS.
# Явно заданная CRS только назначается, без перепроецирования.
OUTPUT_CRS = None

# -------------------- Чтение --------------------

POINTS_PER_CHUNK = 1_000_000

# Исключать точки, отмеченные флагом withheld.
IGNORE_WITHHELD = True

# None — любые классы.
# Например, (2,) — использовать только точки класса 2.
INCLUDED_CLASSES = None

# По умолчанию не исключаем классы дополнительно.
EXCLUDED_CLASSES = ()

# Необязательный диапазон высот в исходных единицах Z.
MIN_ALLOWED_Z = None
MAX_ALLOWED_Z = None

# -------------------- Фильтрация ячеек --------------------

# При меньшем числе точек не делаем статистических выводов:
# оставляем все точки, прошедшие предварительные фильтры.
MIN_POINTS_FOR_OUTLIER_FILTER = 5

MAD_THRESHOLD = 3.5

# Минимальный абсолютный порог относительно медианы.
# Нужен в том числе при MAD=0.
# Предполагается, что Z выражен в метрах.
MIN_OUTLIER_DISTANCE = 0.15

# -------------------- Временные данные --------------------

# Все точки одной ячейки всегда попадают в один bucket.
TEMP_BUCKETS = 256

# Максимальный размер одного bucket перед загрузкой в RAM.
# При превышении скрипт остановится и предложит увеличить TEMP_BUCKETS.
MAX_BUCKET_FILE_MB = 256

# None — системный временный каталог.
# Желательно использовать быстрый локальный диск.
TEMP_DIRECTORY = None

# -------------------- Результаты --------------------

SAVE_COUNT_RASTERS = True

# Отдельная маска: 1 — ячейка содержит меньше
# MIN_POINTS_FOR_OUTLIER_FILTER точек, статистическая фильтрация
# для неё не выполнялась.
SAVE_SPARSE_CELL_MASK = True

MAX_RASTER_PIXELS = 200_000_000

BLOCK_SIZE = 256
OVERWRITE = False

# ==================================================================


POINT_RECORD_DTYPE = np.dtype([
    ("cell", "<i8"),
    ("z", "<f8"),
])


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


def validate_settings(resolution: float) -> None:
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("RESOLUTION должен быть положительным.")

    integer_settings = {
        "POINTS_PER_CHUNK": POINTS_PER_CHUNK,
        "TEMP_BUCKETS": TEMP_BUCKETS,
        "MIN_POINTS_FOR_OUTLIER_FILTER": MIN_POINTS_FOR_OUTLIER_FILTER,
        "MAX_RASTER_PIXELS": MAX_RASTER_PIXELS,
    }

    for name, value in integer_settings.items():
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} должен быть целым >= 1.")

    if MIN_POINTS_FOR_OUTLIER_FILTER < 3:
        raise ValueError(
            "MIN_POINTS_FOR_OUTLIER_FILTER должен быть >= 3."
        )

    if MAD_THRESHOLD <= 0 or MIN_OUTLIER_DISTANCE <= 0:
        raise ValueError("Некорректные параметры фильтра выбросов.")

    if MAX_BUCKET_FILE_MB <= 0:
        raise ValueError("MAX_BUCKET_FILE_MB должен быть > 0.")

    if (
        not isinstance(BLOCK_SIZE, int)
        or BLOCK_SIZE < 16
        or BLOCK_SIZE % 16 != 0
    ):
        raise ValueError("BLOCK_SIZE должен быть кратен 16.")

    if (
        MIN_ALLOWED_Z is not None
        and MAX_ALLOWED_Z is not None
        and MIN_ALLOWED_Z > MAX_ALLOWED_Z
    ):
        raise ValueError("MIN_ALLOWED_Z больше MAX_ALLOWED_Z.")


def inspect_cloud(input_path: Path, resolution: float) -> dict:
    with laspy.open(input_path) as reader:
        header = reader.header

        if header.point_count == 0:
            raise ValueError("Облако не содержит точек.")

        minimum = np.asarray(header.mins, dtype=np.float64)
        maximum = np.asarray(header.maxs, dtype=np.float64)

        if not (
            np.isfinite(minimum).all()
            and np.isfinite(maximum).all()
            and (maximum >= minimum).all()
        ):
            raise ValueError("Некорректные границы облака в заголовке.")

        if OUTPUT_CRS is not None:
            crs = CRS.from_user_input(OUTPUT_CRS)
        else:
            parsed_crs = header.parse_crs()
            crs = (
                CRS.from_wkt(parsed_crs.to_wkt())
                if parsed_crs is not None
                else None
            )

        if crs is not None and crs.is_geographic:
            raise ValueError(
                "XY облака заданы в географической CRS. "
                "Для метрической DEM сначала перепроецируй облако. "
                "Этот скрипт координаты не преобразует."
            )

        # Ячейки определяются через floor(X / resolution),
        # floor(Y / resolution).
        #
        # Благодаря +1 точка на крайней координатной линии
        # также входит в растр.
        ix_min = math.floor(float(minimum[0]) / resolution)
        ix_max = math.floor(float(maximum[0]) / resolution)

        iy_min = math.floor(float(minimum[1]) / resolution)
        iy_max = math.floor(float(maximum[1]) / resolution)

        width = ix_max - ix_min + 1
        height = iy_max - iy_min + 1

        if width * height > MAX_RASTER_PIXELS:
            raise ValueError(
                f"Слишком большой растр: {width} × {height}.\n"
                "Проверь границы облака или увеличь RESOLUTION."
            )

        if width * height > np.iinfo(np.int64).max:
            raise ValueError("Размер сетки превышает диапазон int64.")

        left = ix_min * resolution
        top = (iy_max + 1) * resolution

        return {
            "point_count": int(header.point_count),
            "source_min": minimum.tolist(),
            "source_max": maximum.tolist(),
            "ix_min": ix_min,
            "iy_max": iy_max,
            "width": width,
            "height": height,
            "transform": from_origin(
                left,
                top,
                resolution,
                resolution,
            ),
            "crs": crs,
            "resolution": resolution,
        }


def distribute_points(
    input_path: Path,
    temporary_path: Path,
    grid: dict,
) -> tuple[list[Path], dict]:
    bucket_paths = [
        temporary_path / f"bucket_{index:05d}.bin"
        for index in range(TEMP_BUCKETS)
    ]

    statistics = {
        "read_points": 0,
        "nonfinite_points": 0,
        "attribute_filtered_points": 0,
        "accepted_for_cell_processing": 0,
    }

    included = (
        np.asarray(INCLUDED_CLASSES)
        if INCLUDED_CLASSES is not None
        else None
    )
    excluded = np.asarray(EXCLUDED_CLASSES)

    print("\nЭтап 1. Распределение точек по временным файлам")

    with laspy.open(input_path) as reader:
        for chunk_index, points in enumerate(
            reader.chunk_iterator(POINTS_PER_CHUNK),
            start=1,
        ):
            x = np.asarray(points.x, dtype=np.float64)
            y = np.asarray(points.y, dtype=np.float64)
            z = np.asarray(points.z, dtype=np.float64)

            statistics["read_points"] += len(z)

            finite = (
                np.isfinite(x)
                & np.isfinite(y)
                & np.isfinite(z)
            )

            statistics["nonfinite_points"] += int(
                (~finite).sum()
            )

            selected = finite.copy()

            if IGNORE_WITHHELD:
                selected &= ~np.asarray(
                    points.withheld,
                    dtype=bool,
                )

            if included is not None or excluded.size:
                classification = np.asarray(
                    points.classification
                )

                if included is not None:
                    selected &= np.isin(
                        classification,
                        included,
                    )

                if excluded.size:
                    selected &= ~np.isin(
                        classification,
                        excluded,
                    )

            if MIN_ALLOWED_Z is not None:
                selected &= z >= MIN_ALLOWED_Z

            if MAX_ALLOWED_Z is not None:
                selected &= z <= MAX_ALLOWED_Z

            statistics["attribute_filtered_points"] += int(
                (finite & ~selected).sum()
            )

            x = x[selected]
            y = y[selected]
            z = z[selected]

            if z.size:
                cols = (
                    np.floor(x / grid["resolution"])
                    - grid["ix_min"]
                ).astype(np.int64)

                rows = (
                    grid["iy_max"]
                    - np.floor(y / grid["resolution"])
                ).astype(np.int64)

                inside = (
                    (cols >= 0)
                    & (cols < grid["width"])
                    & (rows >= 0)
                    & (rows < grid["height"])
                )

                if not inside.all():
                    raise ValueError(
                        "Некоторые точки выходят за границы "
                        "сетки, рассчитанной по заголовку LAZ. "
                        "Проверь актуальность границ файла."
                    )

                cell_ids = rows * grid["width"] + cols

                # Остаток от номера ячейки определяет bucket.
                # Одна ячейка не может попасть в разные buckets.
                bucket_ids = cell_ids % TEMP_BUCKETS

                order = np.argsort(
                    bucket_ids,
                    kind="stable",
                )

                sorted_buckets = bucket_ids[order]

                records = np.empty(
                    len(z),
                    dtype=POINT_RECORD_DTYPE,
                )
                records["cell"] = cell_ids[order]
                records["z"] = z[order]

                boundaries = np.concatenate([
                    np.array([0]),
                    np.flatnonzero(
                        np.diff(sorted_buckets)
                    ) + 1,
                    np.array([len(records)]),
                ])

                for start, end in zip(
                    boundaries[:-1],
                    boundaries[1:],
                ):
                    bucket_id = int(
                        sorted_buckets[start]
                    )

                    with bucket_paths[bucket_id].open("ab") as file:
                        records[start:end].tofile(file)

                statistics["accepted_for_cell_processing"] += len(z)

            print(
                f"  Порция {chunk_index}: "
                f"прочитано={statistics['read_points']:,}, "
                f"принято={statistics['accepted_for_cell_processing']:,}"
            )

    if statistics["accepted_for_cell_processing"] == 0:
        raise ValueError("После предварительных фильтров точек не осталось.")

    return bucket_paths, statistics


def filter_cell_heights(heights: np.ndarray) -> np.ndarray:
    """
    Однократная фильтрация по медиане/MAD внутри одной ячейки.
    Возвращает маску оставленных точек.
    """
    if len(heights) < MIN_POINTS_FOR_OUTLIER_FILTER:
        return np.ones(len(heights), dtype=bool)

    median = float(np.median(heights))

    deviations = np.abs(
        heights - median
    )

    mad = float(
        np.median(deviations)
    )

    threshold = max(
        MAD_THRESHOLD * 1.4826 * mad,
        MIN_OUTLIER_DISTANCE,
    )

    return deviations <= threshold


def create_work_arrays(temporary_path: Path, grid: dict) -> dict:
    shape = (
        grid["height"],
        grid["width"],
    )

    arrays = {}

    specifications = {
        "dem": ("float64", np.nan),
        "count_before": ("uint32", 0),
        "count_after": ("uint32", 0),
        "count_rejected": ("uint32", 0),
        "sparse": ("uint8", 0),
    }

    for name, (dtype, fill_value) in specifications.items():
        array = np.memmap(
            temporary_path / f"grid_{name}.dat",
            mode="w+",
            dtype=dtype,
            shape=shape,
        )

        array[:] = fill_value
        arrays[name] = array

    return arrays


def process_buckets(
    bucket_paths: list[Path],
    arrays: dict,
) -> dict:
    print("\nЭтап 2. Фильтрация и усреднение внутри ячеек")

    existing_paths = [
        path
        for path in bucket_paths
        if path.exists()
    ]

    maximum_bytes = int(
        MAX_BUCKET_FILE_MB * 1024 ** 2
    )

    # Проверяем до обработки, чтобы не остановиться в середине.
    for path in existing_paths:
        if path.stat().st_size > maximum_bytes:
            raise MemoryError(
                f"Временный файл {path.name} занимает "
                f"{path.stat().st_size / 1024**2:.1f} MiB.\n"
                f"Лимит: {MAX_BUCKET_FILE_MB} MiB.\n"
                "Увеличь TEMP_BUCKETS или MAX_BUCKET_FILE_MB. "
                "Если в одной ячейке чрезвычайно много точек, "
                "увеличение TEMP_BUCKETS не разделит её."
            )

    dem_flat = arrays["dem"].reshape(-1)
    before_flat = arrays["count_before"].reshape(-1)
    after_flat = arrays["count_after"].reshape(-1)
    rejected_flat = arrays["count_rejected"].reshape(-1)
    sparse_flat = arrays["sparse"].reshape(-1)

    statistics = {
        "occupied_cells": 0,
        "sparse_cells_without_statistical_filter": 0,
        "cells_with_rejected_points": 0,
        "retained_points": 0,
        "cell_outlier_points": 0,
    }

    maximum_count = np.iinfo(np.uint32).max

    for bucket_index, path in enumerate(existing_paths, start=1):
        records = np.fromfile(
            path,
            dtype=POINT_RECORD_DTYPE,
        )

        records.sort(
            order="cell",
            kind="quicksort",
        )

        boundaries = np.concatenate([
            np.array([0]),
            np.flatnonzero(
                np.diff(records["cell"])
            ) + 1,
            np.array([len(records)]),
        ])

        for start, end in zip(
            boundaries[:-1],
            boundaries[1:],
        ):
            cell_id = int(
                records["cell"][start]
            )
            heights = records["z"][start:end]

            count_before = len(heights)

            if count_before > maximum_count:
                raise OverflowError(
                    "Количество точек в одной ячейке "
                    "превышает диапазон uint32."
                )

            keep = filter_cell_heights(heights)
            retained = heights[keep]

            count_after = len(retained)
            rejected = count_before - count_after

            before_flat[cell_id] = count_before
            after_flat[cell_id] = count_after
            rejected_flat[cell_id] = rejected

            if count_before < MIN_POINTS_FOR_OUTLIER_FILTER:
                sparse_flat[cell_id] = 1
                statistics[
                    "sparse_cells_without_statistical_filter"
                ] += 1

            if count_after:
                # Среднее только оставшихся исходных высот.
                dem_flat[cell_id] = float(
                    retained.mean(dtype=np.float64)
                )

            statistics["occupied_cells"] += 1
            statistics["retained_points"] += count_after
            statistics["cell_outlier_points"] += rejected

            if rejected:
                statistics["cells_with_rejected_points"] += 1

        del records

        print(
            f"  Bucket {bucket_index}/{len(existing_paths)}: "
            f"ячеек={statistics['occupied_cells']:,}, "
            f"выбросов={statistics['cell_outlier_points']:,}"
        )

    return statistics


def write_geotiff(
    output_path: Path,
    array: np.ndarray,
    grid: dict,
    dtype: str,
    nodata,
    description: str,
) -> None:
    profile = {
        "driver": "GTiff",
        "height": grid["height"],
        "width": grid["width"],
        "count": 1,
        "dtype": dtype,
        "crs": grid["crs"],
        "transform": grid["transform"],
        "nodata": nodata,
        "compress": "deflate",
        "predictor": 3 if dtype.startswith("float") else 2,
        "tiled": True,
        "blockxsize": BLOCK_SIZE,
        "blockysize": BLOCK_SIZE,
        "BIGTIFF": "IF_SAFER",
    }

    temporary_output = output_path.with_name(
        output_path.stem + ".building.tif"
    )

    valid_pixels = 0
    minimum = float("inf")
    maximum = float("-inf")

    try:
        with rasterio.open(
            temporary_output,
            "w",
            **profile,
        ) as dst:
            for _, window in dst.block_windows(1):
                rows, cols = window.toslices()

                block = np.asarray(
                    array[rows, cols],
                    dtype=dtype,
                )

                dst.write(
                    block,
                    1,
                    window=window,
                )

                if dtype.startswith("float"):
                    finite = np.isfinite(block)
                    valid_pixels += int(finite.sum())

                    if finite.any():
                        minimum = min(
                            minimum,
                            float(block[finite].min()),
                        )
                        maximum = max(
                            maximum,
                            float(block[finite].max()),
                        )

            dst.set_band_description(
                1,
                description,
            )

            dst.update_tags(
                source=str(INPUT_LAZ),
                aggregation="mean after per-cell median/MAD filtering",
                interpolation="none",
            )

        temporary_output.replace(output_path)

    finally:
        if temporary_output.exists():
            temporary_output.unlink()

    return {
        "valid_pixels": valid_pixels,
        "z_min": minimum if valid_pixels else None,
        "z_max": maximum if valid_pixels else None,
    }


def close_work_arrays(arrays: dict) -> None:
    for array in arrays.values():
        array.flush()
        array._mmap.close()

    arrays.clear()


def build_dem(
    input_path: Path | str,
    output_path: Path | str,
    resolution: float,
) -> Path:
    validate_settings(resolution)

    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    output_paths = [output_path]

    if SAVE_COUNT_RASTERS:
        output_paths.extend([
            output_path.with_name(
                output_path.stem + "_count_before.tif"
            ),
            output_path.with_name(
                output_path.stem + "_count_after.tif"
            ),
            output_path.with_name(
                output_path.stem + "_count_rejected.tif"
            ),
        ])

    if SAVE_SPARSE_CELL_MASK:
        output_paths.append(
            output_path.with_name(
                output_path.stem + "_sparse_cells.tif"
            )
        )

    if not OVERWRITE:
        for path in output_paths:
            if path.exists():
                raise FileExistsError(
                    f"Результат уже существует:\n{path}"
                )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    grid = inspect_cloud(
        input_path,
        resolution,
    )

    print(f"Источник: {input_path}")
    print(f"Точек в заголовке: {grid['point_count']:,}")
    print(
        f"DEM: {grid['width']} × {grid['height']}, "
        f"шаг={resolution}"
    )

    if grid["crs"] is None:
        print(
            "Предупреждение: CRS отсутствует. "
            "Система координат не назначается автоматически."
        )

    started = time.perf_counter()

    temporary_parent = (
        str(TEMP_DIRECTORY)
        if TEMP_DIRECTORY is not None
        else None
    )

    with tempfile.TemporaryDirectory(
        prefix="quarryvae_laz_dem_",
        dir=temporary_parent,
    ) as temporary:
        temporary_path = Path(temporary)

        bucket_paths, input_statistics = distribute_points(
            input_path,
            temporary_path,
            grid,
        )

        arrays = create_work_arrays(
            temporary_path,
            grid,
        )

        try:
            cell_statistics = process_buckets(
                bucket_paths,
                arrays,
            )

            dem_statistics = write_geotiff(
                output_path,
                arrays["dem"],
                grid,
                dtype="float64",
                nodata=np.nan,
                description="Mean elevation after per-cell outlier filtering",
            )

            if SAVE_COUNT_RASTERS:
                for name, description in [
                    ("count_before", "Points before per-cell filtering"),
                    ("count_after", "Points retained after per-cell filtering"),
                    ("count_rejected", "Points rejected by per-cell filtering"),
                ]:
                    write_geotiff(
                        output_path.with_name(
                            output_path.stem + f"_{name}.tif"
                        ),
                        arrays[name],
                        grid,
                        dtype="uint32",
                        nodata=None,
                        description=description,
                    )

            if SAVE_SPARSE_CELL_MASK:
                write_geotiff(
                    output_path.with_name(
                        output_path.stem + "_sparse_cells.tif"
                    ),
                    arrays["sparse"],
                    grid,
                    dtype="uint8",
                    nodata=None,
                    description=(
                        "1=too few points for statistical outlier filtering"
                    ),
                )

        finally:
            close_work_arrays(arrays)

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "resolution": resolution,
        "width": grid["width"],
        "height": grid["height"],
        "crs": (
            grid["crs"].to_string()
            if grid["crs"] is not None
            else None
        ),
        "source_min": grid["source_min"],
        "source_max": grid["source_max"],
        "aggregation": "mean of retained points per cell",
        "interpolation": None,
        "empty_cell_value": "NoData",
        "filter": {
            "method": "median and MAD within each cell",
            "mad_threshold": MAD_THRESHOLD,
            "minimum_absolute_distance": MIN_OUTLIER_DISTANCE,
            "minimum_points": MIN_POINTS_FOR_OUTLIER_FILTER,
            "small_sample_policy": "retain all eligible points",
        },
        **input_statistics,
        **cell_statistics,
        **dem_statistics,
        "elapsed_seconds": float(
            time.perf_counter() - started
        ),
    }

    write_json(
        output_path.with_suffix(".json"),
        report,
    )

    print("\nГотово.")
    print(
        f"Ячеек с данными: "
        f"{report['valid_pixels']:,}"
    )
    print(
        f"Отфильтровано выбросов внутри ячеек: "
        f"{report['cell_outlier_points']:,}"
    )
    print(
        f"Редких ячеек без статистического фильтра: "
        f"{report['sparse_cells_without_statistical_filter']:,}"
    )
    print(
        f"Время: {report['elapsed_seconds']:.1f} с"
    )
    print(f"GeoTIFF: {output_path}")

    return output_path


if __name__ == "__main__":
    build_dem(
        input_path=INPUT_LAZ,
        output_path=OUTPUT_DEM,
        resolution=RESOLUTION,
    )