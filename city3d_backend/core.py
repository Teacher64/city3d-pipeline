import copy
import hashlib
import json
import os
import resource
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import cKDTree
from .settings import atomic_json
from . import __version__

VALID_CLASSES = set(range(0, 8)) | set(range(9, 12)) | set(range(13, 23)) | set(range(64, 256))
EXTRA_FIELDS = {'c3_orig_class': 'u1', 'c3_semantic': 'u2', 'c3_geometry': 'u1',
                'c3_object': 'u8', 'c3_score': 'f4', 'c3_hag': 'f4',
                'c3_hag_valid': 'u1', 'c3_source': 'u1', 'c3_review': 'u1'}
FEATURE_NAMES = ['hag', 'hag_valid', 'planarity', 'linearity', 'scattering', 'curvature',
                 'roughness', 'verticality', 'density', 'z_spread', 'intensity',
                 'return_number', 'number_of_returns', 'red', 'green', 'blue', 'nir']

class Cancelled(RuntimeError):
    pass

@dataclass
class Config:
    tile_size: float = 40.0
    halo: float = 20.0
    radius: float = 1.0
    neighbors: int = 32
    max_tile_points: int = 500000
    chunk_size: int = 100000
    ground_backend: str = 'smrf'
    max_ground_distance: float = 5.0
    units: str = ''
    preserve_classes: bool = True
    score_threshold: float = 0.6
    model: str = ''
    allowed_classes: tuple = tuple(sorted(VALID_CLASSES))
    scanning_type: str = 'unknown'
    resume: bool = True
    output_format: str = 'las'

    def validate(self):
        if self.tile_size <= 0 or self.radius <= 0 or self.halo < max(self.radius, self.max_ground_distance):
            raise ValueError('tile_size/radius > 0; halo >= radius и max_ground_distance')
        if self.ground_backend not in ('smrf', 'csf', 'trusted', 'legacy'):
            raise ValueError('Неизвестный ground backend')
        if self.ground_backend == 'smrf' and self.halo < 18:
            raise ValueError('SMRF window=18 требует halo >=18 м')
        if self.neighbors < 3 or self.max_tile_points < self.neighbors or self.chunk_size < 1:
            raise ValueError('Недопустимый бюджет тайла/число соседей')
        if not 0 <= self.score_threshold <= 1 or self.max_ground_distance <= 0:
            raise ValueError('Недопустимый score/distance')
        if not set(self.allowed_classes) <= VALID_CLASSES:
            raise ValueError('Зарезервированные LAS-классы нельзя назначать')
        if self.output_format not in ('las', 'laz'):
            raise ValueError('output_format: las или laz')
        if self.units not in ('', 'metres'):
            raise ValueError('Поддерживаются метрические координаты')
        return self

def check(cancel):
    if cancel and cancel():
        raise Cancelled('Расчёт отменён')

def sha256(path, cancel=None):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            check(cancel)
            h.update(b)
    return h.hexdigest()

def inspect_header(header, config):
    if header.point_count == 0:
        raise ValueError('Пустое облако')
    if header.point_format.id in (4, 5, 9, 10) or header.start_of_waveform_data_packet_record:
        raise ValueError('Waveform LAS не поддерживается: экспорт остановлен до потери данных')
    if np.any(~np.isfinite(header.scales)) or np.any(header.scales <= 0):
        raise ValueError('Некорректные LAS scales')
    crs = header.parse_crs()
    if crs is None:
        if config.units != 'metres':
            raise ValueError('Неизвестны единицы CRS. Подтвердите metres для локальной системы')
    elif not crs.is_projected or any(abs(a.unit_conversion_factor - 1) > 1e-10 for a in crs.axis_info[:2]):
        raise ValueError('Нужен проекционный CRS в метрах; перепроецируйте вход отдельно')
    # LAS header gives no reliable units for Z; caller must explicitly confirm them.
    if config.units != 'metres':
        raise ValueError('Подтвердите единицы XY/Z: metres; преобразование высот здесь не выполняется')
    stale = [(v.user_id, v.record_id) for v in list(header.vlrs) + list(header.evlrs or [])
             if v.user_id.lower().startswith(('lasindex', 'lax', 'copc'))]
    if stale:
        raise ValueError(f'Индексированные VLR/EVLR требуют специального экспорта: {stale}')
    return crs

def idw(xy, ground_xy, ground_z, max_distance):
    if len(ground_z) == 0:
        return np.full(len(xy), np.nan), np.zeros(len(xy), bool)
    dist, idx = cKDTree(ground_xy).query(xy, k=min(8, len(ground_z)))
    dist = np.asarray(dist).reshape(len(xy), -1)
    idx = np.asarray(idx).reshape(len(xy), -1)
    w = np.where(dist <= max_distance, 1 / np.maximum(dist, 1e-9)**2, 0)
    total = w.sum(axis=1)
    valid = total > 0
    z = np.full(len(xy), np.nan)
    z[valid] = (ground_z[idx[valid]] * w[valid]).sum(axis=1) / total[valid]
    return z, valid

def run_process(args, cancel=None):
    # File-backed output avoids deadlock on full stdout/stderr pipes.
    with tempfile.TemporaryFile() as log:
        p = subprocess.Popen(args, stdout=log, stderr=log)
        try:
            while p.poll() is None:
                check(cancel)
                time.sleep(.1)
        except BaseException:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            raise
        log.seek(0)
        message = log.read().decode(errors='replace')
        if p.returncode:
            raise RuntimeError('PDAL: ' + message[-4000:])
        return message

def ground_mask(points, classes, config, directory, cancel=None):
    if config.ground_backend == 'trusted':
        return classes == 2
    if config.ground_backend == 'legacy':
        # Explicit baseline; not a validated ground extractor.
        return points[:, 2] <= np.percentile(points[:, 2], 3) + .15
    if shutil.which('pdal') is None:
        raise RuntimeError('PDAL отсутствует; выберите trusted для проверенных классов земли')
    header = laspy.LasHeader(point_format=3, version='1.2')
    header.scales = np.array([.001] * 3)
    header.offsets = points.min(axis=0)
    header.add_extra_dim(laspy.ExtraBytesParams(name='c3_tile_id', type='u8'))
    las = laspy.LasData(header)
    las.x, las.y, las.z = points.T
    las.return_number = np.ones(len(points), np.uint8)
    las.number_of_returns = np.ones(len(points), np.uint8)
    las.c3_tile_id = np.arange(len(points), dtype=np.uint64)
    inp, out = directory / 'ground_input.las', directory / 'ground_output.las'
    las.write(inp)
    stage = {'type': 'filters.' + config.ground_backend}
    if config.ground_backend == 'smrf':
        stage.update(window=18., slope=.15, threshold=.5, scalar=1.25)
    pipeline = [{'type':'readers.las', 'filename':str(inp), 'use_eb_vlr':True}, stage, {'type': 'writers.las', 'filename': str(out),
                                  'minor_version': 4, 'dataformat_id': 6, 'extra_dims': 'all'}]
    spec = directory / 'ground_pipeline.json'
    atomic_json(spec, {'pipeline': pipeline})
    run_process(['pdal', 'pipeline', str(spec)], cancel)
    result = laspy.read(out)
    ids = np.asarray(result.c3_tile_id).astype(np.int64)
    if len(ids) != len(points) or not np.array_equal(np.sort(ids), np.arange(len(points))):
        raise RuntimeError('PDAL нарушил обратную индексацию точек')
    mask = np.zeros(len(points), bool)
    mask[ids] = np.asarray(result.classification) == 2
    for p in (inp, out, spec):
        p.unlink(missing_ok=True)
    return mask

def features(points, core_indices, hag, valid, attributes, config, cancel=None):
    tree = cKDTree(points)
    result = np.full((len(core_indices), len(FEATURE_NAMES)), np.nan, np.float32)
    result[:, 0] = hag[core_indices]
    result[:, 1] = valid[core_indices]
    for start in range(0, len(core_indices), 4096):
        check(cancel)
        ids = core_indices[start:start + 4096]
        dist, neighbors = tree.query(points[ids], k=min(config.neighbors, len(points)),
                                     distance_upper_bound=config.radius, workers=1)
        for row, (d, nn) in enumerate(zip(dist, neighbors), start):
            nn = nn[np.isfinite(d)]
            if len(nn) < 3:
                continue
            local = points[nn] - points[nn].mean(axis=0)
            eig, vec = np.linalg.eigh(local.T @ local / len(nn))
            eig = np.maximum(eig, 0)
            e0, e1, e2 = eig
            denom = max(e2, 1e-12)
            result[row, 2:10] = [(e1-e0)/denom, (e2-e1)/denom, e0/denom,
                                 e0/max(eig.sum(), 1e-12), np.sqrt(e0),
                                 1-abs(vec[2, 0]), len(nn)/(4*np.pi*config.radius**3/3),
                                 np.ptp(points[nn, 2])]
    for col, name in enumerate(FEATURE_NAMES[10:], 10):
        if name in attributes:
            result[:, col] = attributes[name][core_indices]
    return result

def model_config(config):
    return {k: v for k, v in asdict(config).items() if k in
            ('radius', 'neighbors', 'ground_backend', 'max_ground_distance', 'scanning_type', 'halo', 'tile_size')}

def load_model(path, config):
    # Only load trusted locally-created joblib files (pickle is executable).
    import joblib
    bundle = joblib.load(path)
    if bundle.get('feature_names') != FEATURE_NAMES or bundle.get('version') not in ('7.4.0', __version__):
        raise ValueError('Несовместимая схема признаков/версия модели')
    if bundle.get('feature_config') != model_config(config):
        raise ValueError('Параметры признаков не совпадают с обучением')
    return bundle

def classify(feature_matrix, original, ground, config, bundle=None):
    n = len(original)
    labels = np.ones(n, np.uint8)
    score = np.zeros(n, np.float32)
    source = np.zeros(n, np.uint8)
    if bundle is not None:
        probs = bundle['model'].predict_proba(feature_matrix)
        labels = bundle['model'].classes_[probs.argmax(axis=1)].astype(np.uint8)
        score = probs.max(axis=1).astype(np.float32)
        source[:] = 3
        labels[score < config.score_threshold] = 1
    elif config.ground_backend != 'legacy':
        labels[ground] = 2
        # SMRF/CSF rule score is not a calibrated probability.
        score[ground] = 1
        source[ground] = 2
    labels[~np.isin(labels, config.allowed_classes)] = 1
    if config.preserve_classes:
        locked = ~np.isin(original, (0, 1))
        labels[locked] = original[locked]
        score[locked] = np.nan
        source[locked] = 1
    review = ((labels <= 1) | ((source == 3) & (score < config.score_threshold))).astype(np.uint8)
    return labels, score, source, review

def schema(config):
    return {'version': 1, 'asprs_classes': sorted(VALID_CLASSES), 'reserved': [8, 12, *range(23, 64)],
            'geometry': {'0': 'unknown', '1': 'planar', '2': 'linear', '3': 'scattered'},
            'source': {'0': 'unknown', '1': 'preserved', '2': 'ground_rule', '3': 'ExtraTrees'},
            'fields': EXTRA_FIELDS, 'object_id_nodata': 0, 'float_nodata': 'NaN',
            'score': 'uncalibrated; not accuracy or calibrated probability',
            'classes': {str(c): {'status': 'preservation; recognition requires labelled model' if c != 2 else
                                 'preservation; ground backend recognition',
                                 'enabled': c in config.allowed_classes} for c in sorted(VALID_CLASSES)},
            'restricted': '20/22 and custom semantics require labelled model; geometry alone is insufficient'}

def prepare_tiles(path, directory, config, cancel=None):
    tile_dir = directory / 'tiles'
    tile_dir.mkdir(exist_ok=True)
    for old in tile_dir.glob('*.bin'):
        old.unlink()
    index = {}
    with laspy.open(path) as reader:
        header = reader.header
        origin = np.asarray(header.mins, dtype=np.float64)
        offset = 0
        for chunk in reader.chunk_iterator(config.chunk_size):
            check(cancel)
            xyz = np.column_stack((chunk.x, chunk.y, chunk.z)) - origin
            if not np.isfinite(xyz).all():
                raise ValueError('Неконечные координаты')
            tiles = np.floor(xyz[:, :2] / config.tile_size).astype(np.int64)
            # Store point IDs once, tile data as append-only structured records.
            attrs = [n for n in FEATURE_NAMES[10:] if n in chunk.point_format.dimension_names]
            dtype = [('id', '<u8'), ('xyz', '<f8', (3,)), ('class', 'u1')] + [(n, '<f8') for n in attrs]
            records = np.empty(len(chunk), dtype=dtype)
            records['id'] = np.arange(offset, offset + len(chunk))
            records['xyz'] = xyz
            records['class'] = np.asarray(chunk.classification)
            for n in attrs:
                records[n] = np.asarray(chunk[n])
            unique, inv = np.unique(tiles, axis=0, return_inverse=True)
            order = np.argsort(inv, kind='stable')
            boundaries = np.flatnonzero(np.diff(inv[order])) + 1
            for ti, ids in zip(unique, np.split(order, boundaries)):
                key = tuple(map(int, ti))
                file = tile_dir / f'{key[0]}_{key[1]}.bin'
                with file.open('ab') as f:
                    records[ids].tofile(f)
                index[key] = index.get(key, 0) + len(ids)
            offset += len(chunk)
    return index, np.dtype(dtype), origin

def process(path, output_dir, config=None, progress=None, cancel=None, collect_features=False):
    config = (config or Config()).validate()
    path, directory = Path(path).resolve(), Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / ('m1_classified.' + config.output_format)
    if path == final:
        raise ValueError('Вход и выход совпадают')
    started = time.perf_counter()
    timings = {}
    manifest_path = directory / 'run_manifest.json'
    fingerprint = sha256(path, cancel)
    with laspy.open(path) as reader:
        header = copy.deepcopy(reader.header)
        crs = inspect_header(header, config)
        n = header.point_count
        if collect_features and n > 2000000:
            raise MemoryError('Обучение: ограничение 2 млн точек на территорию; подготовьте репрезентативную выборку')
    # Ensure algorithm/version/model changes invalidate tile cache.
    model_hash = sha256(config.model, cancel) if config.model else ''
    run_key = hashlib.sha256(json.dumps([fingerprint, asdict(config), __version__, model_hash], sort_keys=True).encode()).hexdigest()
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if previous and previous.get('run_key') != run_key:
        raise ValueError('Каталог содержит другой запуск: выберите новую выходную папку')
    manifest = dict(version=__version__, input=str(path), input_sha256=fingerprint, run_key=run_key,
                    n_points=n, crs=crs.to_string() if crs else None, status='running',
                    model_sha256=model_hash, completed_tiles=previous.get('completed_tiles', []),
                    artifacts={}, config=asdict(config))
    atomic_json(manifest_path, manifest)
    atomic_json(directory / 'resolved_config.json', asdict(config))
    atomic_json(directory / 'class_schema.json', schema(config))
    temp_output = directory / ('.m1_classified.tmp.' + config.output_format)
    bundle = None
    try:
        if config.model:
            bundle = load_model(config.model, config)
        check(cancel)
        t = time.perf_counter()
        index, dtype, origin = prepare_tiles(path, directory, config, cancel)
        timings['read_index_s'] = time.perf_counter() - t
        manifest['local_origin'] = origin.tolist()
        results = directory / 'results'
        results.mkdir(exist_ok=True)
        done = set(manifest['completed_tiles']) if config.resume else set()
        span = int(np.ceil(config.halo / config.tile_size))
        counts = {}
        feature_parts, label_parts, spatial_groups = [], [], []
        t = time.perf_counter()
        for ti, key in enumerate(sorted(index)):
            check(cancel)
            name = f'{key[0]}_{key[1]}'
            cached = results / (name + '.npz')
            if name in done and cached.exists():
                result = dict(np.load(cached, allow_pickle=False))
            else:
                core = np.fromfile(directory / 'tiles' / (name + '.bin'), dtype=dtype)
                lower = np.array(key) * config.tile_size - config.halo
                upper = (np.array(key) + 1) * config.tile_size + config.halo
                context = [core]
                total = len(core)
                if total > config.max_tile_points:
                    raise MemoryError('Тайл превышает бюджет: уменьшите tile_size')
                for dx in range(-span, span+1):
                    for dy in range(-span, span+1):
                        neighbor = (key[0]+dx, key[1]+dy)
                        if neighbor == key or neighbor not in index:
                            continue
                        file = directory / 'tiles' / f'{neighbor[0]}_{neighbor[1]}.bin'
                        arr = np.memmap(file, dtype=dtype, mode='r')
                        for start in range(0, len(arr), config.chunk_size):
                            check(cancel)
                            part = arr[start:start+config.chunk_size]
                            keep = np.all((part['xyz'][:, :2] >= lower) & (part['xyz'][:, :2] <= upper), axis=1)
                            selected = part[keep].copy()
                            total += len(selected)
                            if total > config.max_tile_points:
                                raise MemoryError('Halo превышает бюджет: уменьшите тайл/радиус или увеличьте бюджет')
                            context.append(selected)
                all_records = np.concatenate(context)
                xyz = all_records['xyz']
                stage_t = time.perf_counter()
                gm = ground_mask(xyz, all_records['class'], config, directory, cancel)
                timings['ground_s'] = timings.get('ground_s', 0) + time.perf_counter()-stage_t
                ground_z, valid = idw(xyz[:, :2], xyz[gm, :2], xyz[gm, 2], config.max_ground_distance)
                hag = xyz[:, 2] - ground_z
                # Ground labels are based on backend, never on unbounded negative HAG.
                ids = np.arange(len(core))
                stage_t = time.perf_counter()
                attrs = {name: all_records[name] for name in FEATURE_NAMES[10:] if name in dtype.names}
                feat = features(xyz, ids, hag, valid, attrs, config, cancel)
                timings['features_s'] = timings.get('features_s', 0) + time.perf_counter()-stage_t
                labels, score, source, review = classify(feat, core['class'], gm[ids], config, bundle)
                geometry = np.zeros(len(core), np.uint8)
                geometry[feat[:, 2] > .6] = 1
                geometry[feat[:, 3] > .7] = 2
                geometry[feat[:, 4] > .4] = 3
                result = dict(id=core['id'], original=core['class'], labels=labels, score=score,
                              source=source, review=review, hag=hag[ids].astype(np.float32),
                              valid=valid[ids].astype(np.uint8), geometry=geometry, features=feat)
                tmp = cached.with_suffix('.tmp.npz')
                np.savez(tmp, **result)
                os.replace(tmp, cached)
                done.add(name)
                manifest['completed_tiles'] = sorted(done)
                atomic_json(manifest_path, manifest)
            for cls, number in zip(*np.unique(result['labels'], return_counts=True)):
                counts[str(int(cls))] = counts.get(str(int(cls)), 0) + int(number)
            if collect_features:
                feature_parts.append(result['features'])
                label_parts.append(result['original'])
                spatial_groups.append(np.full(len(result['id']), ti, dtype=np.int64))
            if progress:
                progress(int(10 + 70*(ti+1)/len(index)), f'тайл {ti+1}/{len(index)}')
        timings['tiles_s'] = time.perf_counter()-t
        # Disk-backed point-ID scatter restores exact input order.
        pred_dtype = np.dtype([('labels','u1'), ('original','u1'), ('geometry','u1'), ('score','f4'),
                               ('source','u1'), ('review','u1'), ('hag','f4'), ('valid','u1')])
        predictions = np.memmap(directory / 'predictions.bin', dtype=pred_dtype, mode='w+', shape=(n,))
        seen = np.memmap(directory / 'seen.bin', dtype='u1', mode='w+', shape=(n,))
        seen[:] = 0
        for key in sorted(index):
            check(cancel)
            with np.load(results / f'{key[0]}_{key[1]}.npz', allow_pickle=False) as result:
                ids = result['id']
                if np.any(ids >= n) or np.any(seen[ids]) or len(np.unique(ids)) != len(ids):
                    raise RuntimeError('Нарушен point ID')
                seen[ids] = 1
                for name in pred_dtype.names:
                    predictions[name][ids] = result[name]
        if not np.all(seen):
            raise RuntimeError('Не все входные точки обработаны')
        predictions.flush()
        t = time.perf_counter()
        # Preserve point format, all source fields, CRS, scales, VLR and EVLR.
        # Legacy formats cannot represent new classes >31; fail rather than truncate.
        if header.point_format.id < 6 and any(int(c) > 31 for c in counts):
            raise ValueError('Пользовательские классы требуют исходного LAS point format 6–10; преобразуйте явно')
        if any(name in header.point_format.dimension_names for name in EXTRA_FIELDS):
            raise ValueError('Вход уже содержит c3_* поля; используйте оригинал для повторной классификации')
        for name, kind in EXTRA_FIELDS.items():
            header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=kind))
        review_count = 0
        with laspy.open(path) as reader, laspy.open(temp_output, mode='w', header=header) as writer:
            offset = 0
            for chunk in reader.chunk_iterator(config.chunk_size):
                check(cancel)
                p = predictions[offset:offset+len(chunk)]
                record = laspy.ScaleAwarePointRecord.zeros(len(chunk), header=header)
                for name in chunk.point_format.dimension_names:
                    record[name] = chunk[name]
                record.classification = p['labels']
                for field, values in {'c3_orig_class': p['original'], 'c3_semantic': p['labels'],
                                     'c3_geometry': p['geometry'], 'c3_object': np.zeros(len(p), np.uint64),
                                     'c3_score': p['score'], 'c3_hag': p['hag'], 'c3_hag_valid': p['valid'],
                                     'c3_source': p['source'], 'c3_review': p['review']}.items():
                    record[field] = values
                writer.write_points(record)
                review_count += int(p['review'].sum())
                offset += len(chunk)
            if reader.header.evlrs:
                writer.write_evlrs(reader.header.evlrs)
        timings['write_s'] = time.perf_counter()-t
        check(cancel)
        verify_roundtrip(path, temp_output, config.chunk_size, cancel)
        if sha256(path,cancel)!=fingerprint:
            raise ValueError('Исходный файл изменился во время обработки')
        os.replace(temp_output, final)
        timings['total_s'] = time.perf_counter()-started
        metrics = dict(accuracy=None, quality_score=None, counts=counts,
                       review_fraction=review_count/n, evaluated=False,
                       note='Нет независимой разметки: покрытие не является точностью')
        atomic_json(directory / 'metrics.json', metrics)
        atomic_json(directory / 'performance.json', dict(stages=timings, points_per_second=n/timings['total_s'],
                    peak_self_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    peak_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                    note='RSS self/children отдельно; не peak RSS всей группы', input_bytes=path.stat().st_size,
                    output_bytes=final.stat().st_size, resumed=bool(previous)))
        for name, data in [('m1_buildings.json', {'n_buildings': 0, 'buildings': [], 'status': 'not reconstructed'}),
                           ('m1_planes.json', {'roof_planes': [], 'wall_planes': [], 'status': 'geometry features only'}),
                           ('m1_vectors.json', metrics)]:
            atomic_json(directory / name, data)
        (directory / 'm1_report.txt').write_text(f'City3D {__version__}\nТочек: {n}\nКлассы: {counts}\n'
            f'Время: {timings["total_s"]:.3f} с\nНезависимая accuracy не измерена.\n'
            'Roof/wall и object instances не реконструированы. RGB сохранён.\n', encoding='utf-8')
        manifest.update(status='success', artifacts={'classified': str(final)}, timings=timings)
        atomic_json(manifest_path, manifest)
        if progress:
            progress(100, 'LAS проверен и сохранён')
        if collect_features:
            return np.concatenate(feature_parts), np.concatenate(label_parts), np.concatenate(spatial_groups)
        return final
    except BaseException as e:
        temp_output.unlink(missing_ok=True)
        manifest.update(status='cancelled' if isinstance(e, Cancelled) else 'failed', error=str(e))
        atomic_json(manifest_path, manifest)
        raise

def verify_roundtrip(source, output, chunk_size=100000, cancel=None):
    with laspy.open(source) as a, laspy.open(output) as b:
        if a.header.point_count != b.header.point_count:
            raise RuntimeError('N_out != N_in')
        if not np.array_equal(a.header.scales, b.header.scales) or not np.array_equal(a.header.offsets, b.header.offsets):
            raise RuntimeError('Scales/offsets изменены')
        for src in a.chunk_iterator(chunk_size):
            check(cancel)
            dst = b.read_points(len(src))
            for name in src.point_format.dimension_names:
                if name == 'classification':
                    continue
                if not np.array_equal(np.asarray(src[name]), np.asarray(dst[name]), equal_nan=True):
                    raise RuntimeError('Изменено исходное измерение: ' + name)
        if str(a.header.parse_crs()) != str(b.header.parse_crs()):
            raise RuntimeError('CRS изменён')
        def records(header):
            return [(v.user_id, v.record_id, v.record_data_bytes()) for v in
                    list(header.vlrs) + list(header.evlrs or [])
                    if not (v.user_id == 'LASF_Spec' and v.record_id == 4) and v.user_id != 'laszip encoded']
        if records(a.header) != records(b.header):
            raise RuntimeError('VLR/EVLR изменены')
