import argparse
import json
from pathlib import Path
import numpy as np
from .core import Config, FEATURE_NAMES, VALID_CLASSES, model_config, process, load_model
from .settings import atomic_json
from . import __version__

def evaluate(y, pred):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    classes = np.unique(np.concatenate([y, pred]))
    cm = confusion_matrix(y, pred, labels=classes)
    p, r, f, support = precision_recall_fscore_support(y, pred, labels=classes, zero_division=0)
    denom = cm.sum(axis=0) + cm.sum(axis=1) - cm.diagonal()
    iou = np.divide(cm.diagonal(), denom, out=np.zeros(len(classes), float), where=denom != 0)
    return dict(classes=classes.tolist(), confusion_matrix=cm.tolist(), precision=p.tolist(),
                recall=r.tolist(), f1=f.tolist(), support=support.tolist(), iou=iou.tolist(),
                macro_f1=float(f.mean()), miou=float(iou.mean()), absent_policy='union of reference and prediction')

def train(x, y, groups, config, target):
    import joblib
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupShuffleSplit
    mask = np.isin(y, sorted(VALID_CLASSES - {0, 1}))
    x, y, groups = x[mask], y[mask], groups[mask]
    if len(np.unique(groups)) < 3 or len(np.unique(y)) < 2:
        raise ValueError('Нужны минимум 3 независимые территории и 2 размеченных класса')
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=.3, random_state=42).split(x, y, groups))
    if set(np.unique(y[b])) - set(np.unique(y[a])):
        raise ValueError('В holdout есть класс без обучающих точек; расширьте разметку')
    model = make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
                          ExtraTreesClassifier(n_estimators=128, random_state=42, n_jobs=1, class_weight='balanced'))
    model.fit(x[a], y[a])
    metrics = evaluate(y[b], model.predict(x[b]))
    target = Path(target)
    if target.exists():
        raise ValueError('Модель уже существует: укажите новое имя, лучший checkpoint не перезаписывается')
    target.parent.mkdir(parents=True, exist_ok=True)
    bundle = dict(model=model, feature_names=FEATURE_NAMES, feature_config=model_config(config), version=__version__,
                  metrics=metrics, seed=42, train_groups=np.unique(groups[a]).tolist(),
                  holdout_groups=np.unique(groups[b]).tolist())
    tmp = target.with_suffix('.tmp')
    joblib.dump(bundle, tmp)
    tmp.replace(target)
    atomic_json(target.with_suffix('.metrics.json'), {k:v for k,v in bundle.items() if k != 'model'})
    return metrics

def main(argv=None):
    import sys
    actual = argv if argv is not None else sys.argv[1:]
    if actual and actual[0] in ('pipeline','inventory','models','db'):
        from .admin_cli import main as admin_main
        return admin_main(actual)
    parser = argparse.ArgumentParser(description='City3D CPU LAS classifier; XY/Z metres must be confirmed')
    parser.add_argument('mode', choices=['infer','train','evaluate'])
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--units', choices=['metres'], required=True)
    parser.add_argument('--ground', choices=['smrf','csf','trusted','legacy'], default='smrf')
    parser.add_argument('--model', default='')
    parser.add_argument('--format', choices=['las','laz'], default='las')
    parser.add_argument('--replace-classes', action='store_true')
    args = parser.parse_args(argv)
    values = json.loads(args.config.read_text()) if args.config else {}
    values.update(units=args.units, ground_backend=args.ground, model=args.model,
                  preserve_classes=not args.replace_classes, output_format=args.format)
    config = Config(**values).validate()
    if args.mode == 'train':
        # Training manifest lists independent non-overlapping labelled territories.
        manifest = json.loads(args.input.read_text())
        entries = manifest['territories']
        if len(entries) < 3 or len({str(e['group']) for e in entries}) < 3:
            raise ValueError('Минимум 3 независимые территории в manifest')
        boxes, xs, ys, gs, hashes = [], [], [], [], []
        import laspy
        from .core import sha256
        crs_key = None
        for i, entry in enumerate(entries):
            path = Path(entry['path'])
            if not path.is_absolute():
                path = args.input.parent / path
            with laspy.open(path) as reader:
                h = reader.header
                current_crs = str(h.parse_crs())
                if crs_key is not None and crs_key != current_crs:
                    raise ValueError('CRS территорий различаются')
                crs_key = current_crs
                box = (h.mins[:2].copy(), h.maxs[:2].copy())
                if any(np.all(box[0]-config.halo <= hi) and np.all(lo-config.halo <= box[1]) for lo,hi in boxes):
                    raise ValueError('Территории пересекаются или ближе halo: риск spatial leakage')
                boxes.append(box)
            features, labels, _ = process(path, args.output.parent / (args.output.stem + f'_territory_{i}'),
                                           config, collect_features=True)
            xs.append(features); ys.append(labels)
            gs.append(np.full(len(labels), str(entry['group'])))
            hashes.append({'path':str(path), 'sha256':sha256(path), 'group':str(entry['group'])})
        metrics = train(np.concatenate(xs), np.concatenate(ys), np.concatenate(gs), config, args.output)
        atomic_json(args.output.with_suffix('.sources.json'), {'territories':hashes})
        print(json.dumps(metrics))
    elif args.mode == 'evaluate':
        if not args.model:
            raise ValueError('--model обязателен')
        if config.preserve_classes:
            raise ValueError('Evaluation требует --replace-classes')
        # Run inference then evaluate against source labels, excluding 0/1.
        import laspy
        out = process(args.input, args.output, config)
        reference = laspy.read(args.input).classification
        prediction = laspy.read(out).classification
        mask = ~np.isin(reference, [0,1])
        if config.preserve_classes:
            raise ValueError('Evaluation требует --replace-classes, иначе исходные метки блокируются')
        if not mask.any():
            raise ValueError('Нет эталонных меток')
        metrics = evaluate(np.asarray(reference)[mask], np.asarray(prediction)[mask])
        atomic_json(args.output / 'evaluation.json', metrics)
        print(json.dumps(metrics))
    else:
        from .runner import PipelineRunner,Task
        from .database import LocalJournal
        from dataclasses import asdict
        result=PipelineRunner(journal=LocalJournal(args.output/'outbox.sqlite')).run(
            [Task('M1','m1.cpu',inputs={'asset':str(args.input)},params={'config':asdict(config)})],args.output,
            progress=lambda pct,note:print(f'{pct}% {note}',flush=True))
        if result['status']!='success':
            raise RuntimeError(json.dumps(result['steps'],ensure_ascii=False))
        print(result['steps']['M1']['artifacts']['classified']['path'])

if __name__ == '__main__':
    main()
