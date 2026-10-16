#!/usr/bin/env python3
"""Inventory only: does not import checkpoints, download, or execute external projects."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import zipfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--models',default='/home/michel/projects/city3d_models')
    parser.add_argument('--projects',default='/home/michel/projects')
    parser.add_argument('--sdk',default='/opt/panorama')
    parser.add_argument('--output',default='models_sdk_report.json')
    args=parser.parse_args()
    report={'python':sys.version,'executable':sys.executable,'models':[],'projects':{},'sdk':{},'dependencies':{}}
    for name in ('torch','torch_geometric','torch_scatter','torch_sparse','torch_points3d','segmentation_models_pytorch','ultralytics','osgeo','rasterio'):
        try:report['dependencies'][name]=importlib.util.find_spec(name) is not None
        except Exception:report['dependencies'][name]=False
    for path in Path(args.models).rglob('*'):
        if not path.is_file() or path.suffix.lower() not in ('.pth','.pt','.ckpt','.onnx','.safetensors','.json','.yaml','.yml'):continue
        with path.open('rb') as f:head=f.read(200)
        row={'path':str(path),'bytes':path.stat().st_size,'lfs_pointer':head.startswith(b'version https://git-lfs.github.com/spec/v1'),'symlink':path.is_symlink()}
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:row['archive_members']=z.namelist()[:50]
        report['models'].append(row)
    for name in ('Cloud2BIM','RoomFormer','SpatialLM','CADReasoner','3d_cad_pointcloud'):
        root=Path(args.projects)/name
        report['projects'][name]={'exists':root.is_dir(),'configs':[str(p) for p in root.rglob('*.yaml')][:50],
                                'requirements':[str(p) for p in root.glob('*requirements*')],
                                'python_entry_candidates':[str(p) for p in root.glob('*.py')]}
    root=Path(args.sdk)
    report['sdk']={'root':str(root),'exists':root.is_dir(),'headers':[str(p) for p in root.rglob('*.h')][:100],
                   'libraries':[str(p) for p in root.rglob('*.so*')][:100],
                   'classifiers':[str(p) for p in root.rglob('*.rsc')][:30]}
    Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('Сохранён отчёт:',Path(args.output).resolve())


if __name__=='__main__':main()
