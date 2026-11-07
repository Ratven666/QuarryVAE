from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio


# ============================= CONFIG =============================

# Для расположения:
# QuarryVAE/scripts/vae_fill/3_export_dem_geometry.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Можно экспортировать исходную, повреждённую или заполненную DEM.
INPUT_DEM = (
    PROJECT_ROOT
    / "outputs"
    / "full_dem_defects_20261107_184938_460641"
    / "dem_restored_only.tif"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / f"dem_geometry_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
)

EXPORT_POINT_CLOUD = True
EXPORT_MESH = True

# 1 — каждый пиксель.
# 2 — каждый второй пиксель по обеим осям.
# 4 — каждый четвёртый и т. д.
#
# Это прореживание, а не усреднение высот.
GRID_STEP = 1

# Дополнительный NoData, если он не прописан в TIFF.
EXTRA_NODATA_VALUE = None

# Локальные координаты удобнее для просмотра геометрии.
# Обратное преобразование:
# global_xyz = exported_xyz + coordinate_origin.
USE_LOCAL_COORDINATES = True

# None — выбрать начало автоматически по первой валидной точке.
# Либо явно задать (X0, Y0, Z0), одинаковое для нескольких экспортов.
COORDINATE_ORIGIN = None

# Пример:
# COORDINATE_ORIGIN = (350000.0, 6650000.0, 0.0)

# Размер порции записи вершин.
VERTEX_WRITE_CHUNK = 500_000

# ==================================================================


def write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def read_dem():
    with rasterio.open(INPUT_DEM) as src:
        if src.count != 1:
            raise ValueError("Ожидается одноканальная DEM.")

        band = src.read(
            1,
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

        dem = raw * src.scales[0] + src.offsets[0]
        valid &= np.isfinite(dem)

        metadata = {
            "height": src.height,
            "width": src.width,
            "transform": src.transform,
            "crs": src.crs,
            "source_scale": float(src.scales[0]),
            "source_offset": float(src.offsets[0]),
        }

    if not valid.any():
        raise ValueError("В DEM нет валидных высот.")

    return dem, valid, metadata


def prepare_vertices(dem, valid, transform):
    # Исходные индексы выбранных строк и столбцов.
    rows = np.arange(0, dem.shape[0], GRID_STEP)
    cols = np.arange(0, dem.shape[1], GRID_STEP)

    sampled_dem = dem[np.ix_(rows, cols)]
    sampled_valid = valid[np.ix_(rows, cols)]

    local_rows, local_cols = np.nonzero(sampled_valid)

    if not local_rows.size:
        raise ValueError("После прореживания не осталось точек.")

    source_rows = rows[local_rows].astype(np.float64) + 0.5
    source_cols = cols[local_cols].astype(np.float64) + 0.5

    x = (
        transform.a * source_cols
        + transform.b * source_rows
        + transform.c
    )
    y = (
        transform.d * source_cols
        + transform.e * source_rows
        + transform.f
    )
    z = sampled_dem[local_rows, local_cols]

    if not USE_LOCAL_COORDINATES:
        origin = np.zeros(3, dtype=np.float64)

    elif COORDINATE_ORIGIN is not None:
        origin = np.asarray(
            COORDINATE_ORIGIN,
            dtype=np.float64,
        )

        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("COORDINATE_ORIGIN должен содержать три числа.")

    else:
        # X/Y переносим к первой точке, высоты оставляем абсолютными.
        origin = np.array(
            [x[0], y[0], 0.0],
            dtype=np.float64,
        )

    vertices = np.column_stack([
        x - origin[0],
        y - origin[1],
        z - origin[2],
    ]).astype("<f8", copy=False)

    # Индекс вершины для каждой ячейки прореженной сетки.
    vertex_ids = np.full(
        sampled_valid.shape,
        -1,
        dtype=np.int64,
    )
    vertex_ids[local_rows, local_cols] = np.arange(
        len(vertices),
        dtype=np.int64,
    )

    return vertices, vertex_ids, rows, cols, origin


def make_invalid_integral(valid):
    """
    Для GRID_STEP > 1 проверяем не только выбранные вершины,
    но и исходные пиксели между ними.
    """
    if GRID_STEP == 1:
        return None

    integral = np.zeros(
        (valid.shape[0] + 1, valid.shape[1] + 1),
        dtype=np.int64,
    )

    np.cumsum(
        ~valid,
        axis=0,
        dtype=np.int64,
        out=integral[1:, 1:],
    )
    np.cumsum(
        integral[1:, 1:],
        axis=1,
        out=integral[1:, 1:],
    )

    return integral


def iter_faces(vertex_ids, rows, cols, transform, invalid_integral):
    """
    Каждая валидная четверка вершин даёт два треугольника.
    Через NoData грани не строятся.
    """
    determinant = transform.a * transform.e - transform.b * transform.d

    if determinant == 0:
        raise ValueError("Вырожденное аффинное преобразование.")

    if len(rows) < 2 or len(cols) < 2:
        return

    for row_index in range(len(rows) - 1):
        a = vertex_ids[row_index, :-1]
        b = vertex_ids[row_index, 1:]
        c = vertex_ids[row_index + 1, :-1]
        d = vertex_ids[row_index + 1, 1:]

        good = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)

        if invalid_integral is not None:
            # Проверяем весь прямоугольник исходной сетки,
            # включающий четыре выбранные вершины.
            top = int(rows[row_index])
            bottom = int(rows[row_index + 1]) + 1

            left = cols[:-1]
            right = cols[1:] + 1

            invalid_count = (
                invalid_integral[bottom, right]
                - invalid_integral[top, right]
                - invalid_integral[bottom, left]
                + invalid_integral[top, left]
            )
            good &= invalid_count == 0

        if not good.any():
            continue

        first = np.column_stack([a[good], b[good], c[good]])
        second = np.column_stack([b[good], d[good], c[good]])

        faces = np.concatenate([first, second], axis=0)

        # Ориентируем треугольники нормалью в сторону +Z.
        if determinant < 0:
            faces = faces[:, [0, 2, 1]]

        yield faces.astype("<i4")


def ply_header(vertex_count, face_count=None):
    lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment Exported from DEM by QuarryVAE",
        "comment Coordinates and CRS are described in metadata.json",
        f"element vertex {vertex_count}",
        "property double x",
        "property double y",
        "property double z",
    ]

    if face_count is not None:
        lines.extend([
            f"element face {face_count}",
            "property list uchar int vertex_indices",
        ])

    lines.append("end_header")
    return ("\n".join(lines) + "\n").encode("ascii")


def write_vertices(file, vertices):
    for start in range(0, len(vertices), VERTEX_WRITE_CHUNK):
        vertices[start:start + VERTEX_WRITE_CHUNK].tofile(file)


def export_cloud(path, vertices):
    with path.open("wb") as file:
        file.write(ply_header(len(vertices)))
        write_vertices(file, vertices)


def export_mesh(
    path,
    vertices,
    vertex_ids,
    rows,
    cols,
    transform,
    invalid_integral,
):
    def face_iterator():
        return iter_faces(
            vertex_ids,
            rows,
            cols,
            transform,
            invalid_integral,
        )

    # Количество граней нужно записать в заголовок заранее.
    face_count = sum(len(faces) for faces in face_iterator())

    # Без выравнивания: uchar + 3 * int32 = 13 байт.
    face_dtype = np.dtype([
        ("count", "u1"),
        ("indices", "<i4", (3,)),
    ], align=False)

    with path.open("wb") as file:
        file.write(ply_header(len(vertices), face_count))
        write_vertices(file, vertices)

        for faces in face_iterator():
            records = np.empty(len(faces), dtype=face_dtype)
            records["count"] = 3
            records["indices"] = faces
            records.tofile(file)

    return face_count


def main():
    if not INPUT_DEM.is_file():
        raise FileNotFoundError(INPUT_DEM)

    if GRID_STEP < 1 or not isinstance(GRID_STEP, int):
        raise ValueError("GRID_STEP должен быть целым числом >= 1.")

    if VERTEX_WRITE_CHUNK < 1:
        raise ValueError("VERTEX_WRITE_CHUNK должен быть >= 1.")

    if not EXPORT_POINT_CLOUD and not EXPORT_MESH:
        raise ValueError("Оба варианта экспорта отключены.")

    dem, valid, metadata = read_dem()

    crs = metadata["crs"]
    if crs is not None and crs.is_geographic:
        raise ValueError(
            "DEM имеет географическую CRS: X/Y в угловых единицах. "
            "Перед построением метрической 3D-поверхности "
            "перепроецируй DEM в подходящую проекционную CRS."
        )

    if crs is None:
        print(
            "Предупреждение: CRS отсутствует. "
            "Координаты берутся из transform без изменения единиц."
        )

    vertices, vertex_ids, rows, cols, origin = prepare_vertices(
        dem,
        valid,
        metadata["transform"],
    )

    # Для граней используем знаковые 32-битные индексы PLY.
    if EXPORT_MESH and len(vertices) > np.iinfo(np.int32).max:
        raise ValueError(
            "Слишком много вершин для int32 индексов. "
            "Увеличь GRID_STEP или раздели DEM."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    print(f"DEM: {dem.shape}")
    print(f"GRID_STEP: {GRID_STEP}")
    print(f"Vertices: {len(vertices):,}")
    print(f"Coordinate origin: {origin.tolist()}")

    if EXPORT_POINT_CLOUD:
        path = OUTPUT_DIR / "point_cloud.ply"
        export_cloud(path, vertices)
        print(f"Point cloud: {path}")

    face_count = None

    if EXPORT_MESH:
        invalid_integral = make_invalid_integral(valid)

        path = OUTPUT_DIR / "surface_mesh.ply"
        face_count = export_mesh(
            path,
            vertices,
            vertex_ids,
            rows,
            cols,
            metadata["transform"],
            invalid_integral,
        )

        print(f"Triangles: {face_count:,}")
        print(f"Mesh: {path}")

    write_json(
        OUTPUT_DIR / "metadata.json",
        {
            "source": str(INPUT_DEM),
            "source_shape": list(dem.shape),
            "grid_step": GRID_STEP,
            "vertices": len(vertices),
            "triangles": face_count,
            "crs_wkt": crs.to_wkt() if crs is not None else None,
            "source_transform": list(metadata["transform"])[:6],
            "source_scale": metadata["source_scale"],
            "source_offset": metadata["source_offset"],
            "coordinate_origin": origin.tolist(),
            "coordinate_restoration": (
                "source_xyz = exported_xyz + coordinate_origin"
            ),
            "vertex_position": "pixel center",
            "coordinate_storage": "float64",
            "mesh_policy": (
                "two triangles per fully valid grid cell; "
                "no bridging across source NoData"
            ),
        },
    )

    print(f"\nЭкспорт завершён: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()