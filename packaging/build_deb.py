#!/usr/bin/env python3
"""Build source-only application with one coherent distro-owned Python/Qt runtime."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent.parent
VERSION = '7.5.0'
arch = subprocess.check_output(['dpkg','--print-architecture'],text=True).strip()
os_release = dict(line.strip().split('=',1) for line in Path('/etc/os-release').read_text().splitlines() if '=' in line)
if os_release.get('ID') != 'ubuntu' or os_release.get('VERSION_ID','').strip('"') != '26.04':
    raise SystemExit('Сборка требует Ubuntu 26.04; проверьте runtime на целевой системе')
os.environ['SOURCE_DATE_EPOCH'] = '1791972000'
os.environ['TZ'] = 'UTC'
stage = ROOT / 'build/pkg'
if stage.exists():
    shutil.rmtree(stage)
code = stage / 'usr/share/city3d'
code.mkdir(parents=True)
for name in ('city3d_pipeline_v5.py','city3d_backend','resources','migrations','crslist.json','classifier.txt','models.json'):
    src = ROOT / name
    if src.is_dir():
        shutil.copytree(src,code/name,ignore=shutil.ignore_patterns('__pycache__'))
    else:
        shutil.copy2(src,code/name)
for name, target in [('city3d','city3d_pipeline_v5.py'),('city3d-cli','-m city3d_backend.cli')]:
    file = stage / 'usr/bin' / name
    file.parent.mkdir(parents=True,exist_ok=True)
    file.write_text('#!/bin/sh\nexport OPENBLAS_NUM_THREADS="${CITY3D_BLAS_THREADS:-1}"\n'
                    'export OMP_NUM_THREADS="${CITY3D_BLAS_THREADS:-1}"\n'
                    'export CITY3D_WORKERS="${CITY3D_WORKERS:-4}"\n'
                    'cd /usr/share/city3d || exit 1\nexec /usr/bin/python3 '+target+' "$@"\n')
    file.chmod(0o755)
desktop = stage/'usr/share/applications/city3d.desktop'
desktop.parent.mkdir(parents=True)
desktop.write_text('[Desktop Entry]\nType=Application\nName=City3D\nComment=LAS classification and 3D point cloud tools\nExec=city3d\nIcon=city3d\nTerminal=false\nCategories=Graphics;Science;Geography;\n')
icon = stage/'usr/share/icons/hicolor/scalable/apps/city3d.svg'
icon.parent.mkdir(parents=True)
icon.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="12" fill="#20252d"/><path d="M10 48V27h12v21m5 0V14h14v34m5 0V23h9v25" stroke="#4ec9b0" fill="none" stroke-width="4"/></svg>')
doc = stage/'usr/share/doc/city3d'
doc.mkdir(parents=True)
for name in ('README_RU.md','RELEASE_NOTES.md','LICENSES.md','dependencies-ubuntu26.txt'):
    shutil.copy2(ROOT/name,doc/name)
control = stage/'DEBIAN/control'
control.parent.mkdir()
control.write_text(f'''Package: city3d
Version: {VERSION}
Section: science
Priority: optional
Architecture: {arch}
Maintainer: City3D local build <city3d@localhost>
Depends: python3 (>= 3.14), python3 (<< 3.15), python3-numpy, python3-scipy, python3-laspy (>= 2.7), python3-laszip, python3-sklearn, python3-joblib, python3-pyproj, python3-open3d, python3-pyqt5, python3-pyqtgraph, python3-opengl, python3-rasterio, python3-shapely, python3-affine, python3-psycopg
Recommends: pdal
Suggests: python3-torch, python3-pyqt5.qtwebengine
Description: City3D point cloud desktop and CPU classification backend
 Preserves LAS attributes and offers tile-based ground and supervised CPU inference.
 SMRF/CSF require separately available PDAL. No weights are downloaded.
''')
md5 = []
for file in sorted(stage.rglob('*')):
    if file.is_file() and 'DEBIAN' not in file.parts:
        md5.append(hashlib.md5(file.read_bytes()).hexdigest()+'  '+str(file.relative_to(stage)))
(stage/'DEBIAN/md5sums').write_text('\n'.join(md5)+'\n')
for file in stage.rglob('*'):
    if file.is_file() and not str(file).startswith(str(stage/'usr/bin')):
        file.chmod(0o644)
    if file.is_dir():
        file.chmod(0o755)
    os.utime(file,(int(os.environ['SOURCE_DATE_EPOCH']),)*2)
output = ROOT/f'city3d_{VERSION}_ubuntu26.04_{arch}.deb'
subprocess.run(['dpkg-deb','--root-owner-group','-Zxz','-z6','--build',str(stage),str(output)],check=True)
archive = ROOT/f'city3d-{VERSION}-source.tar.gz'
# Deterministic source archive with full source, backup, build recipes and tests.
import gzip
with archive.open('wb') as raw, gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=int(os.environ['SOURCE_DATE_EPOCH'])) as gz, tarfile.open(fileobj=gz, mode='w') as tar:
    paths = ['city3d_pipeline_v5.py','city3d_pipeline_v5.py.bak-original','city3d_backend','tests','packaging','resources','migrations',
             'crslist.json','models.json','classifier.txt','README_RU.md','RELEASE_NOTES.md','LICENSES.md',
             'dependencies-ubuntu26.txt','deploy_to_original.sh','m1_example.json']
    for name in paths:
        path = ROOT/name
        files = sorted(path.rglob('*')) if path.is_dir() else [path]
        for file in files:
            if not file.is_file() or '__pycache__' in file.parts:
                continue
            info=tar.gettarinfo(str(file),arcname='city3d-'+VERSION+'/'+str(file.relative_to(ROOT)))
            info.uid=info.gid=0;info.uname=info.gname='root';info.mtime=int(os.environ['SOURCE_DATE_EPOCH'])
            with file.open('rb') as f:
                tar.addfile(info,f)
(ROOT/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in (output,archive)))
print(output)
