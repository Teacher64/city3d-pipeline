"""Panorama capability discovery; no invented native ABI or silent GDAL substitution."""
import os
from pathlib import Path

def discover(root=None):
    root=Path(root or os.environ.get('CITY3D_PANORAMA_SDK','/opt/panorama'))
    candidates=list(root.rglob('*mapapi*.so*')) if root.is_dir() else []
    docs=list(root.rglob('*.h'))[:20] if root.is_dir() else []
    return {'root':str(root),'libraries':[str(p) for p in candidates],'headers':[str(p) for p in docs],
            'status':'blocked','geometry_import':False,'exact_rsc_rendering':False,
            'reason':'SDK/version/ABI/license and supported MAPAPI signatures need verification',
            'documentation':['https://help.gisinfo.ru/api/python/index.html','https://help.gisinfo.ru/api/cpp/index.html']}

def import_sxf(path,rsc,root=None):
    if Path(path).suffix.lower()!='.sxf' or Path(rsc).suffix.lower()!='.rsc':raise ValueError('Нужны настоящие SXF и RSC')
    report=discover(root)
    raise RuntimeError('Panorama blocked: '+report['reason'])
