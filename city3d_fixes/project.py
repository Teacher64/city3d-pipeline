"""Immutable project revisions and lossless homogeneous LAS worksets."""
import copy
import hashlib
import json
import os
import uuid
from pathlib import Path
from datetime import datetime, timezone
import laspy
import numpy as np
from pyproj import CRS
from shapely import concave_hull, intersects_xy
from shapely.geometry import MultiPoint, mapping, shape
from shapely.ops import unary_union, transform
from pyproj.transformer import TransformerGroup


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def cancelled(cancel):
    if cancel():
        raise RuntimeError('Операция отменена; активная редакция не изменена')


def validate_polygon(geometry):
    if geometry.is_empty or geometry.geom_type not in ('Polygon', 'MultiPolygon') or not geometry.is_valid:
        raise ValueError('Граница должна быть непустым валидным Polygon/MultiPolygon без самопересечений')
    return geometry


def to_wgs84(geometry, crs_wkt):
    group = TransformerGroup(CRS.from_user_input(crs_wkt).to_2d(), 'EPSG:4326',
                             always_xy=True, allow_ballpark=False)
    if not group.best_available or not group.transformers:
        raise ValueError('Нет доступного точного преобразования в WGS84')
    # Densification in native units controls curved-edge transformation.
    from shapely import segmentize
    b = geometry.bounds
    geometry = segmentize(geometry, max(max(b[2]-b[0], b[3]-b[1])/200, 0.01))
    result = transform(group.transformers[0].transform, geometry)
    bounds = result.bounds
    if not all(np.isfinite(bounds)) or not (-180 <= bounds[0] <= bounds[2] <= 180 and -90 <= bounds[1] <= bounds[3] <= 90):
        raise ValueError('Некорректное преобразование границы')
    return validate_polygon(result)


def point_boundary(path, ratio=0.05, cancel=lambda: False):
    """Concave hull uses ALL XY points; no random preview sampling."""
    blocks = []
    with laspy.open(path) as reader:
        for points in reader.chunk_iterator(500_000):
            cancelled(cancel)
            blocks.append(np.column_stack((points.x, points.y)))
    if not blocks:
        raise ValueError('Пустое облако')
    geometry = concave_hull(MultiPoint(np.concatenate(blocks)), ratio=ratio, allow_holes=False)
    return validate_polygon(geometry)


def build_revision(records, name, directory, polygon=None, ratio=0.05, project_id=None,
                   cancel=lambda: False, progress=lambda text: None):
    if not name.strip() or not records:
        raise ValueError('Введите название района и отметьте LAS/LAZ')
    if any(Path(r['path']).suffix.lower() not in ('.las', '.laz') for r in records):
        raise ValueError('Рабочая подборка этого патча поддерживает LAS/LAZ')
    # Content duplicates must not silently double point density.
    if len({r['sha256'] for r in records}) != len(records):
        raise ValueError('В выборе есть одинаковые по содержимому файлы; снимите дубликаты')
    crss = [CRS.from_wkt(r['crs_wkt']) if r.get('crs_wkt') else None for r in records]
    if any(c is None for c in crss):
        raise ValueError('Для всех выбранных облаков нужно назначить CRS')
    crs = crss[0]
    if not crs.is_projected or any(abs(a.unit_conversion_factor - 1) > 1e-12 for a in crs.axis_info[:2]):
        raise ValueError('Для геометрической обработки нужна проекция с XY в метрах')
    if any(not crs.equals(c, ignore_axis_order=True) for c in crss[1:]):
        raise ValueError('CRS облаков различаются. Сначала выполните проверенное преобразование; простое назначение CRS не преобразует координаты')
    headers = []
    for r in records:
        cancelled(cancel)
        if sha(r['path']) != r['sha256']:
            raise ValueError('Файл изменился после регистрации: ' + r['path'])
        with laspy.open(r['path']) as reader:
            headers.append(copy.deepcopy(reader.header))
    signature = lambda h: (h.point_format.id, h.point_format.dtype().descr,
                           [(d.name, d.scales.tolist() if d.scales is not None else None,
                             d.offsets.tolist() if d.offsets is not None else None)
                            for d in h.point_format.extra_dimensions])
    if any(signature(h) != signature(headers[0]) for h in headers[1:]):
        raise ValueError('Различаются LAS point format/Extra Bytes. Без потери атрибутов объединять их пока нельзя; выберите совместимую группу')
    if headers[0].point_format.id in (4,5,9,10):
        raise ValueError('Waveform LAS требует отдельного сохранения waveform packets; этот патч такие файлы не объединяет')
    boundary = validate_polygon(polygon) if polygon is not None else unary_union([
        point_boundary(r['path'], ratio, cancel) for r in records])
    validate_polygon(boundary)
    root = Path(directory) / str(uuid.uuid4())
    root.mkdir(parents=True, exist_ok=False)
    out = root / 'workset.las'
    header = copy.deepcopy(headers[0])
    header.scales = np.min([h.scales for h in headers], axis=0)
    header.offsets = np.min([h.mins for h in headers], axis=0)
    high = np.max([h.maxs for h in headers], axis=0)
    if np.any((high-header.offsets)/header.scales > 2**31-1):
        raise ValueError('Общий диапазон не помещается в LAS int32 при исходной точности')
    names = set(header.point_format.dimension_names)
    if names & {'c3_source_index', 'c3_original_index'}:
        raise ValueError('Выбран производный workset вместо оригинала; выберите исходные LAS')
    header.add_extra_dims([laspy.ExtraBytesParams(name='c3_source_index', type=np.uint32),
                           laspy.ExtraBytesParams(name='c3_original_index', type=np.uint64)])
    header.add_crs(crs)
    count = 0
    sources = []
    try:
        with laspy.open(out, mode='w', header=header) as writer:
            for index, (r, h) in enumerate(zip(records, headers)):
                selected = 0
                offset = 0
                progress('Обработка ' + r['path'])
                clipped_path = root / f'source_{index:04d}.las'
                local_header = copy.deepcopy(h)
                local_header.add_crs(crs)
                local_header.add_extra_dims([laspy.ExtraBytesParams(name='c3_source_index', type=np.uint32),
                                             laspy.ExtraBytesParams(name='c3_original_index', type=np.uint64)])
                with laspy.open(clipped_path, mode='w', header=local_header) as local_writer, laspy.open(r['path']) as reader:
                    for points in reader.chunk_iterator(500_000):
                        cancelled(cancel)
                        mask = intersects_xy(boundary, np.asarray(points.x), np.asarray(points.y))
                        ids = np.flatnonzero(mask) + offset
                        offset += len(points)
                        keep = points[mask]
                        if not len(keep):
                            continue
                        batch = laspy.ScaleAwarePointRecord.zeros(len(keep), header=header)
                        for dimension in h.point_format.dimension_names:
                            if dimension not in ('X','Y','Z'):
                                batch[dimension] = keep[dimension]
                        batch.x, batch.y, batch.z = keep.x, keep.y, keep.z
                        batch.c3_source_index = np.full(len(keep), index, dtype=np.uint32)
                        batch.c3_original_index = ids.astype(np.uint64)
                        writer.write_points(batch)
                        local_batch=laspy.ScaleAwarePointRecord(batch.array.copy(),header.point_format,
                                                               header.scales.copy(),header.offsets.copy())
                        local_batch.change_scaling(scales=local_header.scales,offsets=local_header.offsets)
                        local_writer.write_points(local_batch)
                        selected += len(keep)
                sources.append(dict(r, source_index=index, selected_points=selected,
                                    clipped_path=str(clipped_path), source_header_vlrs='preserved in clipped_path'))
                count += selected
        if not count:
            raise ValueError('В границе нет точек выбранных облаков')
        cancelled(cancel)
        project = dict(schema='city3d:project:1', project_id=project_id or str(uuid.uuid4()),
                       revision_id=root.name, name=name.strip(), created_at=datetime.now(timezone.utc).isoformat(),
                       sources=sources, crs_wkt=crs.to_wkt(), boundary=mapping(boundary),
                       boundary_method='operator_polygon' if polygon is not None else f'concave_hull_all_points_ratio_{ratio}',
                       path=str(out), point_count=count, sha256=sha(out),
                       overlap_policy='keep_all_with_source_identity', ontology_version='city3d:1',
                       vertical_policy='unchanged; operator must confirm common height system')
        atomic_json(root/'project.json', project)
        try:
            geom = to_wgs84(boundary, crs.to_wkt())
            atomic_json(root/'boundary.geojson', {'type':'FeatureCollection','features':[
                {'type':'Feature','geometry':mapping(geom),'properties':{
                    k:project[k] for k in ('name','project_id','revision_id','boundary_method')}}]})
        except ValueError as exc:
            project['boundary_wgs84_error'] = str(exc)
            atomic_json(root/'project.json', project)
        return project
    except BaseException:
        # Keep failed files for diagnostics, but never mark/activate them as a project.
        atomic_json(root/'failed.json', {'status':'failed','name':name})
        raise
