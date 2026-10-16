#!/usr/bin/env python3
"""Conservative, hash-pinned installer. Default: dry run; --apply: backup and patch."""
import argparse
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from datetime import datetime, timezone

BASE_HASH = '4543cea20d343b73f5321d2117face7ae5f6394a91632c666ca8b5aa68230fa6'
MARKER = '# CITY3D_TAB1_PATCH_1_0_0'
HOOK = '\n' + MARKER + '\nfrom city3d_tab1.gui import upgrade_load_tab as _upgrade_city3d_tab1\nLoadDataTab = _upgrade_city3d_tab1(LoadDataTab, globals())\n\n'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def patched_text(source):
    tree = ast.parse(source)
    names = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    if 'LoadDataTab' not in names or 'MainWindow' not in names:
        raise ValueError('Не найдены ожидаемые классы')
    lines = source.splitlines(keepends=True)
    lines.insert(names['MainWindow'].lineno - 1, HOOK)
    result = ''.join(lines)
    compile(result, '<patched City3D>', 'exec')
    return result


def atomic_write(path, content, mode=None):
    fd, name = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        if mode is not None:
            os.chmod(name, mode)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def check_environment():
    print('Python:', sys.executable, sys.version.split()[0])
    missing = []
    for name in ('numpy', 'laspy', 'pyproj', 'PyQt5.QtWidgets'):
        try:
            module = importlib.import_module(name)
            print(name, getattr(module, '__version__', 'OK'))
        except Exception as exc:
            missing.append(name)
            print(name, 'НЕ ДОСТУПЕН:', exc)
    try:
        import laspy
        print('LAZ backends:', laspy.LazBackend.detect_available())
    except ImportError:
        pass
    try:
        import psycopg
        print('PostgreSQL: psycopg', psycopg.__version__)
    except ImportError:
        try:
            import psycopg2
            print('PostgreSQL: psycopg2', psycopg2.__version__)
        except ImportError:
            print('PostgreSQL: драйвер отсутствует; локальный реестр работает без БД')
    if missing:
        raise RuntimeError('Установите недостающие зависимости в активное city3d_env; автоматически окружение не меняется')


def install(target, backups, apply=False, debs=()):
    target = Path(target).expanduser().resolve(strict=True)
    source = target.read_text(encoding='utf-8')
    package = Path(__file__).resolve().parent / 'city3d_tab1'
    destination = target.parent / 'city3d_tab1'
    if MARKER in source:
        expected = {str(p.relative_to(package)): digest(p) for p in package.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        if all((destination / p).is_file() and digest(destination / p) == sha for p, sha in expected.items()):
            print('Патч уже установлен; повторное применение не требуется.')
            return
        raise RuntimeError('Маркер уже есть, но модули отличаются. Сначала откатите предыдущую установку.')
    if digest(target) != BASE_HASH:
        raise RuntimeError('SHA256 не совпадает с предоставленным новым city3d_pipeline_v5(2).py. '
                           'Чужую или изменённую версию автоматически не патчим. Пришлите фактически запускаемый файл.')
    if destination.exists():
        raise RuntimeError('Каталог city3d_tab1 уже существует без маркера. Не перезаписываем неизвестные файлы.')
    output = patched_text(source).encode('utf-8')
    for file in package.glob('*.py'):
        compile(file.read_text(encoding='utf-8'), str(file), 'exec')
    print('Цель:', target)
    print('Изменение: один импорт-перехват LoadDataTab; M1 и другие вкладки без изменений.')
    if not apply:
        print('Проверка завершена. Для записи добавьте --apply. Закройте City3D перед установкой.')
        return
    check_environment()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    backup = Path(backups).expanduser().resolve() / ('tab1_' + stamp)
    backup.mkdir(parents=True, exist_ok=False)
    original = backup / target.name
    shutil.copy2(target, original)
    candidates = list(target.parent.glob('*.deb')) + list((target.parent / 'dist').glob('*.deb')) + [Path(d).resolve(strict=True) for d in debs]
    for i, file in enumerate(dict.fromkeys(candidates)):
        if file.is_file():
            out = backup / 'deb' / str(i)
            out.mkdir(parents=True)
            shutil.copy2(file, out / file.name)
    staged = Path(tempfile.mkdtemp(prefix='.city3d_tab1_stage_', dir=target.parent))
    try:
        shutil.copytree(package, staged / 'city3d_tab1', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        manifest = dict(target=str(target), original=str(original), before=digest(original),
                        after=hashlib.sha256(output).hexdigest(), package=str(destination),
                        package_hashes={str(p.relative_to(staged / 'city3d_tab1')): digest(p)
                            for p in (staged / 'city3d_tab1').rglob('*') if p.is_file()})
        (backup / 'restore.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        os.replace(staged / 'city3d_tab1', destination)
        try:
            atomic_write(target, output, target.stat().st_mode)
        except BaseException:
            shutil.rmtree(destination)
            raise
    finally:
        shutil.rmtree(staged)
    print('Установлено. Резервная копия и манифест отката:', backup / 'restore.json')


def restore(manifest_path):
    manifest_path = Path(manifest_path).resolve(strict=True)
    data = json.loads(manifest_path.read_text(encoding='utf-8'))
    target, original, package = map(Path, (data['target'], data['original'], data['package']))
    if digest(target) != data['after'] or digest(original) != data['before']:
        raise RuntimeError('Файл после установки или резервная копия изменились; автоматический откат остановлен.')
    if package != target.parent / 'city3d_tab1':
        raise RuntimeError('Некорректный путь модуля в манифесте')
    current = {str(p.relative_to(package)): digest(p) for p in package.rglob('*')
               if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    if current != data['package_hashes']:
        raise RuntimeError('Модули патча изменены; сначала сохраните изменения отдельно.')
    retired = manifest_path.parent / 'removed_patch'
    if retired.exists():
        raise RuntimeError('Откат уже выполнялся')
    atomic_write(target, original.read_bytes(), original.stat().st_mode)
    shutil.move(str(package), str(retired))
    print('Исходный код восстановлен. Реестр, БД и данные не удалены.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', help='Фактически запускаемый новый city3d_pipeline_v5.py')
    parser.add_argument('--backup-dir', default='/mnt/data_sda1/city3d-backups')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--check-env', action='store_true')
    parser.add_argument('--restore', metavar='RESTORE_JSON')
    parser.add_argument('--deb', action='append', default=[], help='Дополнительный существующий DEB для резервной копии')
    args = parser.parse_args()
    if args.restore:
        restore(args.restore)
    elif args.target:
        install(args.target, args.backup_dir, args.apply, args.deb)
    elif args.check_env:
        check_environment()
    else:
        parser.error('Укажите --target, --check-env или --restore')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('ОСТАНОВЛЕНО:', exc, file=sys.stderr)
        sys.exit(1)
