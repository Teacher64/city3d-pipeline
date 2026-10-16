"""Spatially grouped LAS train/validation/test split; no overlap leakage."""
import json
import random
import shutil
from pathlib import Path
import laspy
from shapely.geometry import box
from .project import atomic_json


def grouped_split(paths,train_ratio=.7,val_ratio=.15,seed=42):
    if not 0<train_ratio<1 or not 0<val_ratio<1 or train_ratio+val_ratio>=1:
        raise ValueError('Некорректные доли train/validation/test')
    paths=list(map(Path,paths));bounds=[];crs=None
    for path in paths:
        with laspy.open(path) as reader:
            h=reader.header;c=h.parse_crs()
            if c is None:raise ValueError('Нужен CRS: '+str(path))
            if crs is not None and not crs.equals(c,ignore_axis_order=True):raise ValueError('Разные CRS в датасете; сначала нормализуйте')
            crs=c;bounds.append(box(h.mins[0],h.mins[1],h.maxs[0],h.maxs[1]))
    parent=list(range(len(paths)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,a in enumerate(bounds):
        for j in range(i):
            if a.intersects(bounds[j]):parent[root(i)]=root(j)
    groups={}
    for i,path in enumerate(paths):groups.setdefault(root(i),[]).append(path)
    groups=list(groups.values());random.Random(seed).shuffle(groups)
    if len(groups)<3:
        raise ValueError('Меньше трёх непересекающихся групп. Нельзя честно разделить перекрывающиеся съёмки на train/val/test; добавьте независимые районы')
    n=len(groups);nt=max(1,min(n-2,round(n*train_ratio)));nv=max(1,min(n-nt-1,round(n*val_ratio)))
    return {key:[p for group in items for p in group] for key,items in
            [('train',groups[:nt]),('val',groups[nt:nt+nv]),('test',groups[nt+nv:])]}


def install_dataset(ns):
    base=ns['DatasetPreparer']
    class Dataset(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            path=self.root/'metadata.json'
            if path.exists():self.metadata.update(json.loads(path.read_text()))
        def split_dataset(self,train_ratio=.7,val_ratio=.15,seed=42):
            files=sorted(p for p in (self.root/'raw').iterdir() if p.suffix.lower() in ('.las','.laz'))
            splits=grouped_split(files,train_ratio,val_ratio,seed)
            # Never mix a fresh split with files from an earlier split.
            if any(any((self.root/'processed'/s).iterdir()) for s in splits if (self.root/'processed'/s).exists()):
                raise ValueError('processed содержит предыдущую выборку. Создайте новую папку датасета, чтобы избежать смешения')
            for key,paths in splits.items():
                folder=self.root/'processed'/key;folder.mkdir(parents=True,exist_ok=True)
                for path in paths:shutil.copy2(path,folder/path.name)
                self.metadata['splits'][key]=len(paths)
            self.metadata['split_policy']='connected_xy_bbox_groups_same_crs'
            self.metadata['seed']=seed
            self._save_metadata()
            return splits
    ns['DatasetPreparer']=Dataset
