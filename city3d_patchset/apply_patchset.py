#!/usr/bin/env python3
"""Install cumulative patch 01/02/03. Default is validation only."""
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
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parent
START='# BEGIN CITY3D_FIXSET_20261016'
END='# END CITY3D_FIXSET_20261016'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transform(text,level):
    if START in text:
        start=text.index(START);end=text.index(END,start)+len(END)
        text=text[:start]+text[end:]
    tree=ast.parse(text)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main']
    if len(nodes)!=1:raise ValueError('Не найден однозначный main()')
    # Explicit type fixes PostgreSQL's untyped parameter $9 without changing tuple arity.
    old='CASE WHEN %s IS NULL THEN NULL'
    new='CASE WHEN %s::text IS NULL THEN NULL'
    if old in text:text=text.replace(old,new)
    elif new not in text:raise ValueError('Не найден ожидаемый SQL start_run')
    tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    lines=text.splitlines(keepends=True)
    hook=f'{START}\nfrom city3d_fixes.integration import install as _install_city3d_fixes\n_install_city3d_fixes(globals(), level={level})\n{END}\n\n'
    lines.insert(node.lineno-1,hook)
    result=''.join(lines)
    compile(result,'patched_city3d','exec')
    return result


def atomic(path,data):
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.')
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.chmod(tmp,path.stat().st_mode if path.exists() else 0o644)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def verify(path,level):
    compile(path.read_text(encoding='utf-8'),str(path),'exec')
    report={'status':'verified_static','target':str(path),'sha256':sha(path),'patch':level,
            'python':sys.executable,'modules':{},'runtime':'GUI/PostGIS/model inference not executed by installer'}
    for folder in ('city3d_tab1','city3d_fixes'):
        directory=path.parent/folder
        if not directory.is_dir():raise ValueError('Отсутствует '+str(directory))
        for source in directory.glob('*.py'):
            compile(source.read_text(),str(source),'exec')
            report['modules'][str(source.relative_to(path.parent))]=sha(source)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',default='/mnt/data_sda1/PR_3D_SPB/city3d-pipeline/city3d_pipeline_v5.py')
    p.add_argument('--patch',choices=['01','02','03'],default='01')
    p.add_argument('--apply',action='store_true')
    p.add_argument('--backup-dir',default='/mnt/data_sda1/city3d-backups')
    p.add_argument('--restore')
    args=p.parse_args()
    if args.restore:
        backup=Path(args.restore).resolve();manifest=json.loads((backup/'manifest.json').read_text())
        target=Path(manifest['target'])
        for relative,expected in manifest['after'].items():
            current=target.parent/relative
            if not current.is_file() or sha(current)!=expected:raise ValueError('Изменён после установки: '+relative)
        for relative in manifest['after']:
            old=backup/'files'/relative;dest=target.parent/relative
            if old.exists():atomic(dest,old.read_bytes())
            else:dest.unlink()
        print('Откат выполнен. Реестры, районы, исходные данные и БД сохранены.');return
    target=Path(args.target).resolve(strict=True)
    text=target.read_text(encoding='utf-8');level=int(args.patch)
    if START not in text:
        expected=json.loads((ROOT/'supported_source.json').read_text())['sha256']
        if sha(target)!=expected:raise ValueError('Исходник отличается от переданного city3d_pipeline_v5(3).py; автоматическая правка остановлена')
    else:
        # The baseline must remain exactly the known source apart from patch SQL/hook.
        baseline=text[:text.index(START)]+text[text.index(END)+len(END):]
        baseline=baseline.replace('CASE WHEN %s::text IS NULL THEN NULL','CASE WHEN %s IS NULL THEN NULL')
        reference=ROOT/'fixtures'/'city3d_pipeline_v5.py'
        expected=json.loads((ROOT/'supported_source.json').read_text())['sha256']
        if sha(reference)!=expected:raise ValueError('Повреждён эталонный исходник fixtures')
        if ast.dump(ast.parse(baseline))!=ast.dump(ast.parse(reference.read_text(encoding='utf-8'))):
            raise ValueError('Исходные функции изменены после патча; нужна адаптация')
    payload={target.name:transform(text,level).encode()}
    for folder in ('city3d_fixes','city3d_tab1'):
        for source in (ROOT/'payload'/folder).rglob('*'):
            if not source.is_file() or '__pycache__' in source.parts:continue
            relative=str(source.relative_to(ROOT/'payload'))
            dest=target.parent/relative
            # Existing Tab1 implementation is not overwritten. Fix applied at runtime.
            if folder=='city3d_tab1' and dest.exists():continue
            payload[relative]=source.read_bytes()
    for relative,data in payload.items():
        if relative.endswith('.py'):compile(data,relative,'exec')
    print('Цель:',target,'; совокупный патч:',args.patch,'; файлов:',len(payload))
    if not args.apply:
        print('Проверка завершена; для установки закройте City3D и добавьте --apply.');return
    if level>=2:
        for name in ('laspy','pyproj','numpy','shapely','PyQt5.QtWidgets'):
            importlib.import_module(name)
        import shapely
        if not hasattr(shapely,'concave_hull') or shapely.geos_version<(3,11,0):
            raise RuntimeError('Для патча 02/03 нужны Shapely >= 2 и GEOS >= 3.11. Окружение автоматически не изменяется.')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    backup=Path(args.backup_dir).expanduser()/('city3d_fixset_'+stamp)
    backup.mkdir(parents=True,exist_ok=False)
    before={};after={}
    for relative,data in payload.items():
        dest=target.parent/relative
        if dest.exists():
            old=backup/'files'/relative;old.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dest,old)
            before[relative]=sha(dest)
        else:before[relative]=None
        after[relative]=hashlib.sha256(data).hexdigest()
    manifest=dict(target=str(target),patch=level,before=before,after=after)
    (backup/'manifest.json').write_text(json.dumps(manifest,indent=2))
    written=[]
    try:
        # Package files first; main entry point last.
        for relative in sorted(payload,key=lambda r:r==target.name):
            dest=target.parent/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            atomic(dest,payload[relative]);written.append(relative)
        report=verify(target,level)
        (backup/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print('Установлено. Резервная копия:',backup)
        print('Отчёт:',backup/'verification.json')
    except BaseException:
        for relative in reversed(written):
            dest=target.parent/relative;old=backup/'files'/relative
            if old.exists():atomic(dest,old.read_bytes())
            else:dest.unlink(missing_ok=True)
        raise


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('ОШИБКА:',exc,file=sys.stderr);sys.exit(1)
