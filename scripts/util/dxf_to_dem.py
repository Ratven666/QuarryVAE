from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import ezdxf
import matplotlib.tri as mtri
import numpy as np
import rasterio
import triangle
from rasterio.crs import CRS
from rasterio.transform import from_origin
from shapely import STRtree
from shapely.geometry import LineString, box
from shapely.ops import unary_union


# ============================= CONFIG =============================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DXF = PROJECT_ROOT / "data/raw/3-D.dxf"

OUTPUT_DEM = (
    PROJECT_ROOT
    / "data/derived/dem_from_densified_breaklines_1m.tif"
)

# Шаг выходного растра в единицах XY.
RESOLUTION = 1

# Максимальная длина сегмента бровки в плане перед TIN.
DENSIFY_STEP = 1.0

LAYERS = ("CREST", "TOE")

# CRS только назначается, без перепроецирования.
OUTPUT_CRS = None

INCLUDE_LINE_ENTITIES = False

XY_TOLERANCE = 1e-6
Z_CONFLICT_TOLERANCE = 0.10
NODE_ROUND_DECIMALS = 8

# error / max / min / mean.
HEIGHT_CONFLICT_POLICY = "max"

MAX_PRINTED_CONFLICTS = 10

# Необязательное исключение треугольников с длинными рёбрами.
# Это удаляет участки поверхности, а не исправляет их.
MAX_TRIANGLE_EDGE = None

MAX_DENSIFIED_VERTICES = 2_000_000
MAX_RASTER_PIXELS = 200_000_000

BLOCK_SIZE = 256

SAVE_TIN = True

# Сохранить фактические сгущённые сегменты, переданные в Triangle.
SAVE_DENSIFIED_DXF = True

OVERWRITE = False

# ==================================================================


def write_json(path, data):
    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def write_csv(path, records, fields):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def validate_settings(resolution, densify_step):
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("RESOLUTION должен быть > 0.")

    if not np.isfinite(densify_step) or densify_step <= 0:
        raise ValueError("DENSIFY_STEP должен быть > 0.")

    if XY_TOLERANCE <= 0 or Z_CONFLICT_TOLERANCE < 0:
        raise ValueError("Некорректные допуски.")

    if HEIGHT_CONFLICT_POLICY not in {"error", "max", "min", "mean"}:
        raise ValueError("HEIGHT_CONFLICT_POLICY: error/max/min/mean.")

    if (
        not isinstance(NODE_ROUND_DECIMALS, int)
        or NODE_ROUND_DECIMALS < 0
    ):
        raise ValueError("Некорректный NODE_ROUND_DECIMALS.")

    if (
        not isinstance(BLOCK_SIZE, int)
        or BLOCK_SIZE < 16
        or BLOCK_SIZE % 16
    ):
        raise ValueError("BLOCK_SIZE должен быть кратен 16.")

    if MAX_TRIANGLE_EDGE is not None and MAX_TRIANGLE_EDGE <= 0:
        raise ValueError("Некорректный MAX_TRIANGLE_EDGE.")

    if MAX_DENSIFIED_VERTICES < 3 or MAX_RASTER_PIXELS < 1:
        raise ValueError("Некорректные ограничения размера.")


def read_segments(path, layers):
    document = ezdxf.readfile(path)
    selected_layers = set(layers) if layers is not None else None

    starts = []
    ends = []

    entity_counts = Counter()
    skipped = Counter()
    vertical_conflicts = []

    def add_segment(first, second, layer, handle):
        first = np.asarray(first, dtype=np.float64)
        second = np.asarray(second, dtype=np.float64)

        if not (
            np.isfinite(first).all()
            and np.isfinite(second).all()
        ):
            raise ValueError(f"NaN/Inf в объекте {handle}.")

        length_xy = float(
            np.linalg.norm(second[:2] - first[:2])
        )

        if length_xy <= XY_TOLERANCE:
            dz = float(second[2] - first[2])

            if abs(dz) > Z_CONFLICT_TOLERANCE:
                vertical_conflicts.append({
                    "layer": layer,
                    "handle": handle,
                    "x1": float(first[0]),
                    "y1": float(first[1]),
                    "z1": float(first[2]),
                    "x2": float(second[0]),
                    "y2": float(second[1]),
                    "z2": float(second[2]),
                    "dz": dz,
                })

                skipped["vertical_conflicts"] += 1

                if len(vertical_conflicts) <= MAX_PRINTED_CONFLICTS:
                    print(
                        f"Вертикальный конфликт: {layer}, "
                        f"handle={handle}, ΔZ={dz:.3f}"
                    )

                if HEIGHT_CONFLICT_POLICY == "error":
                    raise ValueError(
                        "Вертикальный сегмент не представим "
                        "однозначной DEM."
                    )
            else:
                skipped["zero_length_segments"] += 1

            return

        starts.append(first)
        ends.append(second)

    for entity in document.modelspace():
        layer = entity.dxf.layer

        if selected_layers is not None and layer not in selected_layers:
            continue

        kind = entity.dxftype()
        handle = entity.dxf.handle

        if kind == "POLYLINE":
            if not entity.is_3d_polyline:
                skipped["non_3d_POLYLINE"] += 1
                continue

            points = np.asarray(
                [tuple(point) for point in entity.points_in_wcs()],
                dtype=np.float64,
            )

            if len(points) < 2:
                skipped["short_POLYLINE"] += 1
                continue

            for first, second in zip(points[:-1], points[1:]):
                add_segment(first, second, layer, handle)

            if entity.is_closed:
                add_segment(points[-1], points[0], layer, handle)

            entity_counts[layer] += 1

        elif kind == "LINE" and INCLUDE_LINE_ENTITIES:
            add_segment(
                tuple(entity.dxf.start),
                tuple(entity.dxf.end),
                layer,
                handle,
            )
            entity_counts[layer] += 1

        elif kind in {"LWPOLYLINE", "SPLINE", "ARC", "INSERT"}:
            skipped[kind] += 1

    if not starts:
        raise ValueError("Подходящие ненулевые 3D-сегменты не найдены.")

    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)

    lengths = np.linalg.norm(
        ends[:, :2] - starts[:, :2],
        axis=1,
    )

    print("\nОбъектов по слоям:", dict(entity_counts))
    print(f"Исходных сегментов: {len(starts):,}")
    print(f"Максимальная исходная длина: {lengths.max():.3f}")

    if skipped:
        print("Пропущено:", dict(skipped))

    return (
        starts,
        ends,
        dict(entity_counts),
        dict(skipped),
        vertical_conflicts,
        float(lengths.max()),
    )


def iter_lines(geometry):
    if geometry.geom_type == "LineString":
        yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from iter_lines(part)


def resolve_height(estimates):
    spread = float(estimates.max() - estimates.min())

    if spread <= Z_CONFLICT_TOLERANCE:
        return float(estimates.mean())

    if HEIGHT_CONFLICT_POLICY == "max":
        return float(estimates.max())

    if HEIGHT_CONFLICT_POLICY == "min":
        return float(estimates.min())

    if HEIGHT_CONFLICT_POLICY == "mean":
        return float(estimates.mean())

    raise ValueError(f"Конфликт высот: {estimates.tolist()}")


def build_planar_graph(starts, ends):
    """
    Сначала разрешаем пересечения исходных линий,
    затем сгущаем полученный согласованный граф.
    """
    origin = np.minimum(
        starts[:, :2].min(axis=0),
        ends[:, :2].min(axis=0),
    )

    local_starts = starts[:, :2] - origin
    local_ends = ends[:, :2] - origin

    source_lines = [
        LineString([first, second])
        for first, second in zip(local_starts, local_ends)
    ]
    tree = STRtree(source_lines)

    print("\nРазбиение пересечений...")
    noded = unary_union(source_lines)

    vertices = []
    segments = []
    vertex_lookup = {}
    segment_lookup = set()

    def vertex_id(point):
        key = tuple(np.round(point, NODE_ROUND_DECIMALS))

        if key not in vertex_lookup:
            vertex_lookup[key] = len(vertices)
            vertices.append(np.asarray(point, dtype=np.float64))

        return vertex_lookup[key]

    for line in iter_lines(noded):
        coordinates = np.asarray(line.coords, dtype=np.float64)

        for first, second in zip(coordinates[:-1], coordinates[1:]):
            a = vertex_id(first)
            b = vertex_id(second)

            if a == b:
                continue

            key = tuple(sorted((a, b)))

            if key not in segment_lookup:
                segment_lookup.add(key)
                segments.append(key)

    vertices = np.asarray(vertices, dtype=np.float64)
    segments = np.asarray(segments, dtype=np.int32)

    if len(vertices) < 3 or not len(segments):
        raise ValueError("Недостаточно узлов для построения поверхности.")

    heights = np.empty(len(vertices), dtype=np.float64)
    conflicts = []
    maximum_conflict = 0.0

    for index, point in enumerate(vertices):
        candidates = tree.query(
            box(
                point[0] - XY_TOLERANCE,
                point[1] - XY_TOLERANCE,
                point[0] + XY_TOLERANCE,
                point[1] + XY_TOLERANCE,
            )
        )

        if not len(candidates):
            raise RuntimeError(
                f"Не найден сегмент для узла {(point + origin).tolist()}."
            )

        first = local_starts[candidates]
        direction = local_ends[candidates] - first

        length_squared = np.sum(direction ** 2, axis=1)
        t = np.sum((point - first) * direction, axis=1) / length_squared
        t = np.clip(t, 0, 1)

        projections = first + t[:, None] * direction
        close = (
            np.linalg.norm(projections - point, axis=1)
            <= XY_TOLERANCE
        )

        candidates = candidates[close]
        t = t[close]

        if not len(candidates):
            raise RuntimeError("Не удалось определить Z узла.")

        estimates = (
            starts[candidates, 2]
            + t * (ends[candidates, 2] - starts[candidates, 2])
        )

        spread = float(estimates.max() - estimates.min())
        maximum_conflict = max(maximum_conflict, spread)

        if spread > Z_CONFLICT_TOLERANCE:
            global_point = point + origin

            if HEIGHT_CONFLICT_POLICY == "error":
                raise ValueError(
                    f"Конфликт Z в XY={global_point.tolist()}: "
                    f"{estimates.tolist()}"
                )

            chosen = resolve_height(estimates)

            conflicts.append({
                "node_id": index,
                "x": float(global_point[0]),
                "y": float(global_point[1]),
                "z_min": float(estimates.min()),
                "z_max": float(estimates.max()),
                "z_selected": chosen,
                "spread": spread,
                "policy": HEIGHT_CONFLICT_POLICY,
            })

            if len(conflicts) <= MAX_PRINTED_CONFLICTS:
                print(
                    f"Конфликт узла: XY={global_point.tolist()}, "
                    f"ΔZ={spread:.3f}, выбрано Z={chosen:.3f}"
                )

            heights[index] = chosen
        else:
            heights[index] = float(estimates.mean())

    print(f"Узлов до сгущения: {len(vertices):,}")
    print(f"Сегментов до сгущения: {len(segments):,}")
    print(f"Конфликтных узлов: {len(conflicts):,}")

    return (
        vertices,
        heights,
        segments,
        origin,
        conflicts,
        maximum_conflict,
    )


def densify_graph(vertices, heights, segments, step):
    """
    Делит каждое ребро на ceil(length_xy / step) частей.
    Z интерполируется линейно между согласованными концами.
    """
    first_xy = vertices[segments[:, 0]]
    second_xy = vertices[segments[:, 1]]

    lengths = np.linalg.norm(
        second_xy - first_xy,
        axis=1,
    )

    parts = np.maximum(
        1,
        np.ceil(lengths / step).astype(np.int64),
    )

    added_count = int(np.sum(parts - 1))
    total_vertices = len(vertices) + added_count
    total_segments = int(parts.sum())

    if total_vertices > MAX_DENSIFIED_VERTICES:
        raise ValueError(
            f"После сгущения получится {total_vertices:,} вершин. "
            "Увеличь DENSIFY_STEP или MAX_DENSIFIED_VERTICES."
        )

    if total_vertices > np.iinfo(np.int32).max:
        raise ValueError("Превышен диапазон индексов int32.")

    dense_xy = np.empty((total_vertices, 2), dtype=np.float64)
    dense_z = np.empty(total_vertices, dtype=np.float64)
    dense_segments = np.empty((total_segments, 2), dtype=np.int32)

    dense_xy[:len(vertices)] = vertices
    dense_z[:len(vertices)] = heights

    next_vertex = len(vertices)
    next_segment = 0

    for (a, b), number in zip(segments, parts):
        number = int(number)

        if number == 1:
            dense_segments[next_segment] = (a, b)
            next_segment += 1
            continue

        fractions = np.arange(1, number, dtype=np.float64) / number
        interior_count = number - 1

        ids = np.arange(
            next_vertex,
            next_vertex + interior_count,
            dtype=np.int32,
        )

        dense_xy[ids] = (
            vertices[a]
            + fractions[:, None] * (vertices[b] - vertices[a])
        )
        dense_z[ids] = (
            heights[a]
            + fractions * (heights[b] - heights[a])
        )

        chain = np.concatenate([
            np.array([a], dtype=np.int32),
            ids,
            np.array([b], dtype=np.int32),
        ])

        dense_segments[
            next_segment:next_segment + number
        ] = np.column_stack([chain[:-1], chain[1:]])

        next_vertex += interior_count
        next_segment += number

    resulting_lengths = np.linalg.norm(
        dense_xy[dense_segments[:, 1]]
        - dense_xy[dense_segments[:, 0]],
        axis=1,
    )

    maximum_length = float(resulting_lengths.max())

    if maximum_length > step * (1 + 1e-9) + XY_TOLERANCE:
        raise RuntimeError("Не выполнено ограничение длины сегментов.")

    print("\nСгущение завершено.")
    print(f"Добавлено вершин: {added_count:,}")
    print(f"Всего вершин: {total_vertices:,}")
    print(f"Всего сегментов: {total_segments:,}")
    print(f"Максимальная длина сегмента: {maximum_length:.6f}")

    return dense_xy, dense_z, dense_segments, maximum_length


def save_densified_dxf(path, xy, z, segments, origin):
    document = ezdxf.new("R2010")
    document.layers.new("DENSIFIED_BREAKLINES")

    modelspace = document.modelspace()

    # Объекты LINE сохраняют непосредственно рёбра PSLG.
    for a, b in segments:
        first = (
            float(xy[a, 0] + origin[0]),
            float(xy[a, 1] + origin[1]),
            float(z[a]),
        )
        second = (
            float(xy[b, 0] + origin[0]),
            float(xy[b, 1] + origin[1]),
            float(z[b]),
        )

        modelspace.add_line(
            first,
            second,
            dxfattribs={"layer": "DENSIFIED_BREAKLINES"},
        )

    document.saveas(path)


def create_tin(xy, z, segments):
    if np.linalg.matrix_rank(xy - xy.mean(axis=0)) < 2:
        raise ValueError("Все точки лежат на одной прямой.")

    print("\nПостроение constrained TIN...")

    result = triangle.triangulate(
        {
            "vertices": xy,
            "segments": segments,
            "vertex_attributes": z[:, None],
        },
        "pcQ",
    )

    if "triangles" not in result or "vertex_attributes" not in result:
        raise RuntimeError("Не удалось построить TIN с высотами.")

    tin_xy = np.asarray(result["vertices"], dtype=np.float64)
    faces = np.asarray(result["triangles"], dtype=np.int32)
    tin_z = np.asarray(
        result["vertex_attributes"],
        dtype=np.float64,
    ).reshape(len(tin_xy), -1)[:, 0]

    if not np.isfinite(tin_z).all():
        raise RuntimeError("NaN/Inf в высотах TIN.")

    triangulation = mtri.Triangulation(
        tin_xy[:, 0],
        tin_xy[:, 1],
        faces,
    )

    mask = np.zeros(len(faces), dtype=bool)

    if MAX_TRIANGLE_EDGE is not None:
        triangle_xy = tin_xy[faces]

        lengths = np.stack([
            np.linalg.norm(triangle_xy[:, 0] - triangle_xy[:, 1], axis=1),
            np.linalg.norm(triangle_xy[:, 1] - triangle_xy[:, 2], axis=1),
            np.linalg.norm(triangle_xy[:, 2] - triangle_xy[:, 0], axis=1),
        ], axis=1)

        mask = lengths.max(axis=1) > MAX_TRIANGLE_EDGE

        if mask.all():
            raise ValueError("Фильтр длинных рёбер исключил всю TIN.")

        triangulation.set_mask(mask)

    interpolator = mtri.LinearTriInterpolator(triangulation, tin_z)
    triangulation.get_trifinder()

    print(f"Вершин TIN: {len(tin_xy):,}")
    print(f"Треугольников: {len(faces):,}")
    print(f"Исключено треугольников: {int(mask.sum()):,}")

    return tin_xy, tin_z, faces, mask, interpolator


def rasterize_tin(output_path, resolution, crs, tin_xy, origin, interpolator):
    global_xy = tin_xy + origin
    xmin, ymin = global_xy.min(axis=0)
    xmax, ymax = global_xy.max(axis=0)

    left = math.floor(xmin / resolution) * resolution
    right = math.ceil(xmax / resolution) * resolution
    bottom = math.floor(ymin / resolution) * resolution
    top = math.ceil(ymax / resolution) * resolution

    width = int(round((right - left) / resolution))
    height = int(round((top - bottom) / resolution))

    if min(width, height) < 1:
        raise ValueError("Некорректные размеры растра.")

    if width * height > MAX_RASTER_PIXELS:
        raise ValueError(
            f"Слишком большой растр: {width} × {height}. "
            "Увеличь шаг сетки."
        )

    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": from_origin(left, top, resolution, resolution),
        "nodata": np.nan,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": BLOCK_SIZE,
        "blockysize": BLOCK_SIZE,
        "BIGTIFF": "IF_SAFER",
    }

    temporary = output_path.with_name(
        output_path.stem + ".building.tif"
    )

    valid_pixels = 0
    z_min = float("inf")
    z_max = float("-inf")

    print(f"\nDEM: {width} × {height}, шаг={resolution}")

    try:
        with rasterio.open(temporary, "w", **profile) as dst:
            for number, (_, window) in enumerate(
                dst.block_windows(1),
                start=1,
            ):
                columns = (
                    np.arange(int(window.width))
                    + int(window.col_off)
                    + 0.5
                )
                rows = (
                    np.arange(int(window.height))
                    + int(window.row_off)
                    + 0.5
                )

                x = left + columns * resolution - origin[0]
                y = top - rows * resolution - origin[1]

                xx, yy = np.meshgrid(x, y)

                block = np.asarray(
                    interpolator(xx, yy).filled(np.nan),
                    dtype=np.float32,
                )

                finite = np.isfinite(block)
                valid_pixels += int(finite.sum())

                if finite.any():
                    z_min = min(z_min, float(block[finite].min()))
                    z_max = max(z_max, float(block[finite].max()))

                dst.write(block, 1, window=window)

                if number % 100 == 0:
                    print(f"Записано блоков: {number}")

            dst.set_band_description(
                1,
                "Elevation from densified constrained breakline TIN",
            )
            dst.update_tags(
                height_conflict_policy=HEIGHT_CONFLICT_POLICY,
            )

        if not valid_pixels:
            raise RuntimeError("DEM не содержит валидных пикселей.")

        temporary.replace(output_path)

    finally:
        if temporary.exists():
            temporary.unlink()

    return {
        "width": width,
        "height": height,
        "valid_pixels": valid_pixels,
        "z_min": z_min,
        "z_max": z_max,
        "bounds": {
            "left": left,
            "bottom": bottom,
            "right": right,
            "top": top,
        },
    }


def build_dem(
    input_path,
    output_path,
    resolution,
    densify_step,
    crs=None,
    layers=None,
):
    validate_settings(resolution, densify_step)

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    if output_path.exists() and not OVERWRITE:
        raise FileExistsError(
            f"Файл существует: {output_path}\n"
            "Измени путь или задай OVERWRITE=True."
        )

    output_crs = CRS.from_user_input(crs) if crs is not None else None

    if output_crs is not None and output_crs.is_geographic:
        raise ValueError(
            "Нужны плоские XY. Скрипт не выполняет перепроецирование."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    (
        starts,
        ends,
        entity_counts,
        skipped,
        vertical_conflicts,
        original_max_length,
    ) = read_segments(input_path, layers)

    write_csv(
        output_path.with_suffix(".vertical_conflicts.csv"),
        vertical_conflicts,
        [
            "layer", "handle",
            "x1", "y1", "z1",
            "x2", "y2", "z2", "dz",
        ],
    )

    (
        xy,
        z,
        segments,
        origin,
        node_conflicts,
        maximum_conflict,
    ) = build_planar_graph(starts, ends)

    write_csv(
        output_path.with_suffix(".node_conflicts.csv"),
        node_conflicts,
        [
            "node_id", "x", "y",
            "z_min", "z_max", "z_selected",
            "spread", "policy",
        ],
    )

    original_graph_vertices = len(xy)
    original_graph_segments = len(segments)

    xy, z, segments, dense_max_length = densify_graph(
        xy,
        z,
        segments,
        densify_step,
    )

    if SAVE_DENSIFIED_DXF:
        print("\nСохранение сгущённых рёбер в DXF...")

        save_densified_dxf(
            output_path.with_suffix(".densified.dxf"),
            xy,
            z,
            segments,
            origin,
        )

    tin_xy, tin_z, faces, mask, interpolator = create_tin(
        xy,
        z,
        segments,
    )

    raster_report = rasterize_tin(
        output_path,
        resolution,
        output_crs,
        tin_xy,
        origin,
        interpolator,
    )

    if SAVE_TIN:
        np.savez_compressed(
            output_path.with_suffix(".tin.npz"),
            xy=tin_xy + origin,
            z=tin_z,
            triangles=faces,
            triangle_mask=mask,
        )

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "resolution": resolution,
        "densify_step": densify_step,
        "crs": output_crs.to_string() if output_crs is not None else None,
        "selected_layers": list(layers) if layers is not None else None,
        "entities_by_layer": entity_counts,
        "skipped_entities": skipped,
        "original_max_segment_length": original_max_length,
        "graph_vertices_before_densification": original_graph_vertices,
        "graph_segments_before_densification": original_graph_segments,
        "graph_vertices_after_densification": len(xy),
        "graph_segments_after_densification": len(segments),
        "densified_max_segment_length": dense_max_length,
        "tin_vertices": len(tin_xy),
        "tin_triangles": len(faces),
        "masked_triangles": int(mask.sum()),
        "height_conflict_policy": HEIGHT_CONFLICT_POLICY,
        "vertical_conflicts": len(vertical_conflicts),
        "node_conflicts": len(node_conflicts),
        "maximum_node_z_conflict": maximum_conflict,
        "densification_interpolation": "linear XYZ along planar segments",
        "domain": "convex hull of accepted breaklines",
        **raster_report,
    }

    write_json(output_path.with_suffix(".json"), report)

    print("\nГотово.")
    print(f"GeoTIFF: {output_path}")
    print(
        f"Высоты: {raster_report['z_min']:.3f} … "
        f"{raster_report['z_max']:.3f}"
    )

    if output_crs is None:
        print("CRS не назначена: координаты сохранены как в DXF.")

    if vertical_conflicts or node_conflicts:
        print(
            "Есть конфликты высот. "
            "Проверь CSV; применено правило "
            f"{HEIGHT_CONFLICT_POLICY!r}."
        )

    return output_path


if __name__ == "__main__":
    build_dem(
        input_path=INPUT_DXF,
        output_path=OUTPUT_DEM,
        resolution=RESOLUTION,
        densify_step=DENSIFY_STEP,
        crs=OUTPUT_CRS,
        layers=LAYERS,
    )