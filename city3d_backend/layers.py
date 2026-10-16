"""Offline vector/raster/cloud layers and explicit horizontal CRS alignment."""
from dataclasses import asdict,dataclass
import json
from pathlib import Path
from uuid import uuid4
import numpy as np
from pyproj import CRS,Transformer
from shapely.geometry import shape,box
from shapely.ops import transform,unary_union
from .settings import atomic_json

@dataclass
class Layer:
    uuid: str
    name: str
    source: str
    kind: str
    native_crs: str
    display_crs: str='EPSG:3857'
    vertical_crs: str | None=None
    visible: bool=True
    opacity: float=1.
    order: int=0
    state: str='ready'
    style: dict | None=None

class LayerManager:
    def __init__(self):self.layers=[]
    def add(self,source,kind,crs,**kwargs):
        CRS.from_user_input(crs)
        layer=Layer(str(uuid4()),Path(source).name,str(Path(source).resolve()),kind,crs,order=len(self.layers),**kwargs)
        self.layers.append(layer);return layer
    def update(self,uid,**values):
        layer=next(l for l in self.layers if l.uuid==uid)
        if set(values)-{'visible','opacity','order','display_crs','style','name'}:raise ValueError('Unknown layer property')
        if 'opacity' in values and not 0<=values['opacity']<=1:raise ValueError('Opacity range 0..1')
        for key,value in values.items():setattr(layer,key,value)
    def save(self,path):atomic_json(path,{'version':1,'layers':[asdict(l) for l in self.layers]})
    @classmethod
    def load(cls,path):
        data=json.loads(Path(path).read_text());manager=cls()
        if data['version']!=1:raise ValueError('Layer state version mismatch')
        manager.layers=[Layer(**r) for r in data['layers']]
        if len({l.uuid for l in manager.layers})!=len(manager.layers):raise ValueError('Duplicate layer ID')
        return manager

def load_vector(path,crs=None):
    path=Path(path)
    if path.suffix.lower() not in ('.geojson','.json'):
        raise ValueError('Этот reader поддерживает GeoJSON; GPKG/SHP требуют отдельного проверенного provider')
    data=json.loads(path.read_text())
    if data.get('type')!='FeatureCollection':raise ValueError('GeoJSON FeatureCollection required')
    crs=crs or data.get('crs',{}).get('properties',{}).get('name') or 'EPSG:4326'
    source=CRS.from_user_input(crs);projector=Transformer.from_crs(source,4326,always_xy=True)
    rows=[];ids=set()
    for i,f in enumerate(data['features']):
        uid=str(f.get('id',i))
        if uid in ids:raise ValueError('Duplicate source feature ID')
        ids.add(uid)
        geom=shape(f['geometry'])
        if geom.is_empty or not geom.is_valid:raise ValueError('Invalid/empty vector geometry: '+uid)
        normalized=transform(projector.transform,geom)
        if not np.isfinite(np.asarray(normalized.bounds)).all():raise ValueError('CRS transform failed')
        rows.append(dict(id=uid,native=geom,normalized=normalized,properties=f.get('properties',{})))
    if not rows:raise ValueError('Empty vector layer')
    return dict(crs=source.to_string(),srid=source.to_epsg() or 0,features=rows,
                footprint=unary_union([r['normalized'] for r in rows]).envelope,footprint_approximation='bbox')

def sample_raster(path,xy,points_crs):
    import rasterio
    xy=np.asarray(xy,dtype=np.float64)
    with rasterio.open(path) as raster:
        if raster.crs is None:raise ValueError('Raster CRS missing')
        tr=Transformer.from_crs(points_crs,raster.crs,always_xy=True)
        x,y=tr.transform(xy[:,0],xy[:,1])
        from rasterio.transform import rowcol
        rows,cols=rowcol(raster.transform,x,y)
        valid=(np.asarray(rows)>=0)&(np.asarray(rows)<raster.height)&(np.asarray(cols)>=0)&(np.asarray(cols)<raster.width)
        result=np.full((len(xy),raster.count),np.nan,np.float64)
        for i,values in zip(np.flatnonzero(valid),raster.sample(np.column_stack([x,y])[valid],masked=True)):
            result[i]=np.ma.asarray(values).astype(float).filled(np.nan)
        valid&=np.isfinite(result).all(axis=1)
        return result,valid

def preview_cloud(path,target_crs,max_points=100000,cancel=None):
    import laspy
    with laspy.open(path) as r:
        crs=r.header.parse_crs()
        if crs is None:raise ValueError('Cloud CRS missing')
        n=r.header.point_count;step=max(1,int(np.ceil(n/max_points)));arrays=[];ids=[];offset=0;extras={}
        fields=[name for name in ("classification","intensity","red","green","blue","c3_hag","c3_score","c3_source","c3_review") if name in r.header.point_format.dimension_names]
        for chunk in r.chunk_iterator(100000):
            from .core import check
            check(cancel)
            local=np.flatnonzero((np.arange(len(chunk))+offset)%step==0)
            arrays.append(np.column_stack((chunk.x,chunk.y,chunk.z))[local]);ids.append(local+offset)
            for name in fields:extras.setdefault(name,[]).append(np.asarray(chunk[name])[local])
            offset+=len(chunk)
    xyz=np.concatenate(arrays);tr=Transformer.from_crs(crs,target_crs,always_xy=True)
    x,y=tr.transform(xyz[:,0],xyz[:,1]);display=np.column_stack([x,y,xyz[:,2]])
    return {'native':xyz,'display':display,'point_ids':np.concatenate(ids),'source_crs':crs.to_string(),
            'sampling':'deterministic original-record stride','input_count':n,'attributes':{key:np.concatenate(value) for key,value in extras.items()}}

def raster_preview(path,target_crs,max_size=1024):
    import rasterio
    from rasterio.warp import calculate_default_transform,reproject,Resampling
    with rasterio.open(path) as src:
        if src.crs is None:raise ValueError('Raster CRS missing')
        affine,width,height=calculate_default_transform(src.crs,target_crs,src.width,src.height,*src.bounds)
        ratio=max(width/max_size,height/max_size,1);width=max(1,int(width/ratio));height=max(1,int(height/ratio))
        from affine import Affine
        affine=affine*Affine.scale(ratio)
        out=np.full((min(src.count,3),height,width),np.nan,np.float32)
        for band in range(len(out)):
            reproject(rasterio.band(src,band+1),out[band],src_transform=src.transform,src_crs=src.crs,
                      dst_transform=affine,dst_crs=target_crs,src_nodata=src.nodata,dst_nodata=np.nan,resampling=Resampling.bilinear)
        return out,affine
