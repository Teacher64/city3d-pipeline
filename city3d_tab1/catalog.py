"""Streaming LAS/LAZ inventory. Never modifies source files or invents a CRS."""
import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import laspy
import numpy as np
from pyproj import CRS
from pyproj.database import query_crs_info
from pyproj.transformer import TransformerGroup


# CITY3D_SQLITE_CLOSE_FIX_20261019
class _City3DClosingSQLite(sqlite3.Connection):
    def __exit__(self, *exc):
        try:
            return super().__exit__(*exc)
        finally:
            self.close()


EXTENSIONS = {'.las', '.laz', '.ply', '.pcd', '.xyz', '.e57', '.bin', '.txt'}


class Cancelled(Exception):
    pass


def check_cancel(cancel):
    if cancel():
        raise Cancelled('Операция отменена')


def discover(inputs, recursive=True):
    """No arbitrary file limit; do not follow directory symlinks."""
    seen = set()
    for raw in inputs:
        p = Path(raw).expanduser()
        if p.is_dir():
            for root, dirs, files in os.walk(p, followlinks=False):
                dirs.sort()
                for name in sorted(files):
                    f = Path(root, name)
                    if f.suffix.lower() in EXTENSIONS:
                        resolved = str(f.resolve())
                        if resolved not in seen:
                            seen.add(resolved)
                            yield resolved
                if not recursive:
                    dirs.clear()
        else:
            resolved = str(p.resolve())
            if resolved not in seen:
                seen.add(resolved)
                yield resolved


def crs_catalog():
    """All installed EPSG/ESRI definitions; Russia is a searchable subset."""
    return sorted([(f'{i.auth_name}:{i.code}', i.name,
                    i.area_of_use.name if i.area_of_use else '')
                   for authority in ('EPSG', 'ESRI')
                   for i in query_crs_info(auth_name=authority, allow_deprecated=False)],
                  key=lambda r: (r[1], r[0]))


def resolve_crs(header, path, override=None):
    candidates, errors = [], []
    for preferred, name in ((True, 'LAS WKT/GeoTIFF'), (False, 'LAS GeoTIFF/WKT')):
        try:
            c = header.parse_crs(prefer_wkt=preferred)
            if c is not None:
                candidates.append((name, c))
        except Exception as exc:
            errors.append(f'{name}: {exc}')
    for suffix in ('.prj', '.wkt'):
        sidecar = Path(path).with_suffix(suffix)
        if sidecar.is_file():
            try:
                candidates.append((sidecar.name, CRS.from_user_input(sidecar.read_text(encoding='utf-8-sig'))))
            except Exception as exc:
                errors.append(f'{sidecar.name}: {exc}')
    evidence = [{'source': s, 'wkt': c.to_wkt()} for s, c in candidates]
    if override:
        c = CRS.from_user_input(override)
        return c, 'operator', evidence, errors
    if errors:
        return None, 'invalid_metadata', evidence, errors
    if not candidates:
        return None, 'unknown', evidence, errors
    first = candidates[0][1]
    if any(not first.equals(c, ignore_axis_order=True) for _, c in candidates[1:]):
        return None, 'conflict', evidence, ['Противоречивые CRS: требуется выбор оператора']
    return first, candidates[0][0], evidence, errors


def geographic_bbox(crs, bounds):
    """Approximate catalogue coverage, NOT an actual survey boundary."""
    if crs is None or bounds is None:
        return None, 'Нет подтверждённого CRS/точек'
    try:
        horizontal = crs.to_2d()
        group = TransformerGroup(horizontal, CRS.from_epsg(4326), always_xy=True,
                                 allow_ballpark=False)
        if not group.best_available or not group.transformers:
            return None, 'Нет точного доступного преобразования; нужны параметры/сетки PROJ'
        t = group.transformers[0]
        west, south, east, north = t.transform_bounds(
            bounds[0], bounds[1], bounds[3], bounds[4], densify_pts=41, errcheck=True)
        if not all(np.isfinite([west, south, east, north])):
            raise ValueError('Нечисловые координаты')
        if not (-180 <= west <= 180 and -180 <= east <= 180 and -90 <= south <= north <= 90):
            raise ValueError('Выход за границы WGS84')
        if west >= east or south >= north:
            raise ValueError('Вырожденный контур или пересечение антимеридиана: ручная проверка')
        wkt = f'MULTIPOLYGON((({west} {south},{east} {south},{east} {north},{west} {north},{west} {south})))'
        return wkt, f'{t.description}; accuracy={t.accuracy} m; horizontal only'
    except Exception as exc:
        return None, str(exc)


def inspect_file(path, override=None, scan_type='unknown', cancel=lambda: False,
                 progress=lambda n: None, chunk_size=500_000):
    p = Path(path).resolve()
    before = p.stat()
    record = dict(path=str(p), size=before.st_size, mtime_ns=before.st_mtime_ns,
                  format=p.suffix.lower(), scan_type=scan_type, state='indexed',
                  reviewed=False, observed_at=datetime.now(timezone.utc).isoformat(),
                  point_count=None, classes={}, bounds=None, crs_wkt=None,
                  crs_authority=None, crs_source='unknown', crs_evidence=[], warnings=[],
                  footprint_method='bbox_from_all_points_approximate', footprint_wgs84=None)
    if p.suffix.lower() in ('.las', '.laz'):
        with laspy.open(p) as reader:
            h = reader.header
            crs, source, evidence, errors = resolve_crs(h, p, override)
            record.update(las_version=str(h.version), point_format=h.point_format.id,
                          scales=h.scales.tolist(), offsets=h.offsets.tolist(),
                          dimensions=list(h.point_format.dimension_names),
                          header_point_count=int(h.point_count), crs_source=source,
                          crs_evidence=evidence, warnings=errors)
            if crs is not None:
                record['crs_wkt'] = crs.to_wkt()
                record['crs_authority'] = crs.to_authority()
            counts = np.zeros(256, dtype=np.int64)
            lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
            count = 0
            flags = dict(synthetic=0, key_point=0, withheld=0, overlap=0)
            for points in reader.chunk_iterator(chunk_size):
                check_cancel(cancel)
                xyz = np.column_stack((points.x, points.y, points.z))
                if not np.isfinite(xyz).all():
                    raise ValueError('Нечисловые координаты в LAS')
                if len(points):
                    lo = np.minimum(lo, xyz.min(axis=0))
                    hi = np.maximum(hi, xyz.max(axis=0))
                counts += np.bincount(np.asarray(points.classification), minlength=256)
                for key in flags:
                    if key in record['dimensions']:
                        flags[key] += int(np.count_nonzero(np.asarray(points[key])))
                count += len(points)
                progress(int(80 * count / max(1, h.point_count)))
            if count != h.point_count:
                raise ValueError('Количество прочитанных точек не совпало с заголовком')
            bounds = np.concatenate((lo, hi)).tolist() if count else None
            record.update(point_count=count, bounds=bounds, flags=flags,
                          classes={str(i): int(n) for i, n in enumerate(counts) if n},
                          assigned_class_count=int(counts[2:].sum()))
            record['footprint_wgs84'], record['transformation'] = geographic_bbox(crs, bounds)
    else:
        record['warnings'].append('Инвентаризация без разбора точек: полный просмотр — штатным загрузчиком')
        if override:
            crs = CRS.from_user_input(override)
            record.update(crs_wkt=crs.to_wkt(), crs_authority=crs.to_authority(), crs_source='operator')
    digest = hashlib.sha256()
    with p.open('rb') as stream:
        while True:
            check_cancel(cancel)
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    after = p.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Файл изменился во время чтения; повторите индексацию')
    record['sha256'] = digest.hexdigest()
    record['file_id'] = str(uuid.uuid5(uuid.NAMESPACE_URL, 'city3d:sha256:' + record['sha256']))
    record['inspection_id'] = str(uuid.uuid4())
    progress(100)
    return record


class Journal:
    """Local durable journal, including when PostgreSQL is unavailable."""
    def __init__(self, path=None):
        self.path = Path(path or Path.home() / '.local/share/city3d/tab1_inventory.sqlite3')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS inventory (path TEXT PRIMARY KEY, record TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS inspections (id TEXT PRIMARY KEY, record TEXT NOT NULL)')

    def connect(self):
        return sqlite3.connect(self.path, timeout=30, factory=_City3DClosingSQLite)

    def save(self, record):
        payload = json.dumps(record, ensure_ascii=False)
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO inventory VALUES (?,?)', (record['path'], payload))
            db.execute('INSERT OR REPLACE INTO inspections VALUES (?,?)', (record['inspection_id'], payload))

    def records(self):
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute('SELECT record FROM inventory ORDER BY path')]
