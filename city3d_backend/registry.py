import importlib.util
import json
from pathlib import Path
import shutil
from .core import sha256

DEFINITIONS={
 'city3d_unet_cpu':('m3.raster','semantic2d',['torch','rasterio'],'M3/city3d_tiny_unet.npz','locally trained raster baseline'),
 'ransac':('m1.cpu','geometry',['laspy','scipy','sklearn'],None,'ALS/MLS/TLS'),
 'randla_kitti':(None,'semantic',['torch'], 'models/M1/randlanet_semantickitti.pth','MLS'),
 'kpconv_s3dis':(None,'semantic',['torch'],'models/M1/kpconv_s3dis.pth','indoor'),
 'ptv3_scannet':(None,'semantic',['torch'],'models/M1/ptv3_scannet.pth','indoor'),
 'spt_nano_dales':(None,'semantic',['torch','torch_geometric','torch_scatter'],'spt-2_nano_dales.ckpt','ALS DALES'),
 'yolov8n':('yolo.detect','detect',['ultralytics'],'yolov8n.pt','COCO photographs'),
 'yoloworld_s':(None,'detect_open',['ultralytics'],'yolov8s-worldv2.pt','COCO/LVIS photographs'),
 'unet_buildings':(None,'semantic2d',['torch','segmentation_models_pytorch'],'models/M3/unet_resnet34_buildings.pth','orthophoto'),
 'geoai_buildings':(None,'semantic2d',['geoai','rasterio','torch'],'models/M3/geoai_buildings_v1.pth','unverified model definition'),
 'sam_vit_h':(None,'interactive',['segment_anything','torch'],'models/M3/sam_vit_h_4b8939.pth','annotation'),
 'terrain_rf':('m4.features','feature',['rasterio','scipy'],None,'DEM features; not trained RF'),
 'asprs_stat':('m5.qa','reference',['laspy','numpy'],None,'LAS label QA; not independent validation'),
 'gis_vector':('m6.vector','reference',['shapely','pyproj'],None,'vector import; SXF requires Panorama'),
 'lgbm_fuse':(None,'fuse',['lightgbm'],'models/ensemble/lgbm_v1.txt','requires OOF training'),
 'probability_fuse':('semantic.fuse','fuse',['numpy','pyproj'],None,'compatible canonical evidence'),
}

def inventory(models_root,external_root=None):
    models_root=Path(models_root).expanduser().resolve();out=[]
    for mid,(adapter,role,deps,weight,domain) in DEFINITIONS.items():
        missing=[]
        for name in deps:
            try:found=importlib.util.find_spec(name) is not None
            except (ImportError,ValueError):found=False
            if not found:missing.append(name)
        path=models_root/(weight.removeprefix("models/")) if weight else None
        if not adapter:status='adapter_missing'
        elif missing:status='dependencies_missing'
        elif path is not None and not path.is_file():status='weights_missing'
        elif mid=='ransac' and not shutil.which('pdal'):status='ready_trusted_ground_only'
        else:status='ready_for_smoke'
        out.append(dict(id=mid,adapter=adapter,role=role,domain=domain,status=status,missing_dependencies=missing,
                        checkpoint=str(path) if path else None,checkpoint_exists=bool(path and path.is_file()),
                        checkpoint_sha256=sha256(path) if path and path.is_file() else None,
                        validated=False,approved=False,infer_enabled=status in ('ready_for_smoke','ready_trusted_ground_only'),
                        train_enabled=mid in ('ransac','city3d_unet_cpu'),reason='No adapter/config/class-map smoke proof' if not adapter else status))
    if external_root:
        for name in ('Cloud2BIM','RoomFormer','SpatialLM','CADReasoner','3d_cad_pointcloud'):
            root=Path(external_root)/name
            out.append(dict(id=name,status='adapter_missing',path=str(root),code_present=(root/'.git').is_dir(),
                            validated=False,approved=False,infer_enabled=False,reason='External project requires verified CLI contract'))
    return out

def discover_datasets(root,limit=500):
    import laspy
    rows=[]
    for p in Path(root).rglob('*'):
        if not p.is_file() or p.suffix.lower() not in ('.las','.laz','.tif','.tiff','.geojson','.npz'):continue
        entry=dict(path=str(p),bytes=p.stat().st_size,ground_truth_status='unreviewed',license='unknown')
        try:
            if p.suffix.lower() in ('.las','.laz'):
                with laspy.open(p) as r:
                    h=r.header;crs=h.parse_crs();entry.update(points=h.point_count,crs=crs.to_string() if crs else None,
                         bbox=list(map(float,h.mins))+list(map(float,h.maxs)),channels=list(h.point_format.dimension_names))
            elif p.suffix.lower() in ('.tif','.tiff'):
                import rasterio
                with rasterio.open(p) as r:
                    entry.update(crs=str(r.crs),shape=[r.height,r.width],bands=r.count,nodata=r.nodata)
            rows.append(entry)
        except Exception as e:entry['error']=str(e);rows.append(entry)
        if len(rows)>=limit:break
    return rows
