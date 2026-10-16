from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter
from pyproj import CRS
from .core import check

def terrain_features(z,affine):
    z=np.asarray(z,dtype=np.float64)
    if min(z.shape)<3:raise ValueError('DEM block requires >=3 pixels per axis')
    # Affine column/row basis; inverse transpose converts pixel derivatives to metric gradients.
    basis=np.array([[affine.a,affine.b],[affine.d,affine.e]],float)
    if abs(np.linalg.det(basis))<1e-12:raise ValueError('Degenerate geotransform')
    row,col=np.gradient(z)
    inverse=np.linalg.inv(basis.T)
    gx=inverse[0,0]*col+inverse[0,1]*row;gy=inverse[1,0]*col+inverse[1,1]*row
    slope=np.degrees(np.arctan(np.hypot(gx,gy)))
    aspect=np.mod(np.degrees(np.arctan2(-gx,-gy)),360);aspect[np.hypot(gx,gy)<1e-12]=np.nan
    valid=np.isfinite(z);mass=uniform_filter(valid.astype(float),size=5,mode='nearest')
    avg=uniform_filter(np.nan_to_num(z),size=5,mode='nearest')/np.maximum(mass,1e-12)
    tpi=z-avg
    gx_r,gx_c=np.gradient(gx);gy_r,gy_c=np.gradient(gy)
    curvature=inverse[0,0]*gx_c+inverse[0,1]*gx_r+inverse[1,0]*gy_c+inverse[1,1]*gy_r
    result=np.stack([slope,aspect,tpi,curvature]).astype(np.float32)
    result[:,~valid]=np.nan
    return result

def process_dem(path,output,units='metres',cancel=None,progress=None):
    import rasterio
    from rasterio.windows import Window
    if units!='metres':raise ValueError('Подтвердите метры XY/Z DEM')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);tmp=output.with_suffix('.tmp.tif')
    try:
        with rasterio.open(path) as src:
            crs=CRS.from_user_input(src.crs)
            if not crs.is_projected or any(abs(a.unit_conversion_factor-1)>1e-10 for a in crs.axis_info[:2]):
                raise ValueError('DEM требует проекционный CRS в метрах')
            profile=src.profile.copy();profile.update(count=4,dtype='float32',nodata=np.nan,compress='deflate')
            profile.pop('photometric',None)
            with rasterio.open(tmp,'w',**profile) as dst:
                for index,name in enumerate(('slope_degrees','aspect_degrees','TPI_metres','curvature_inverse_metres'),1):dst.set_band_description(index,name)
                rows=list(range(0,src.height,512));cols=list(range(0,src.width,512));total=len(rows)*len(cols);done=0
                for row in rows:
                    for col in cols:
                        check(cancel);h=min(512,src.height-row);w=min(512,src.width-col)
                        top=max(0,row-3);left=max(0,col-3);bottom=min(src.height,row+h+3);right=min(src.width,col+w+3)
                        window=Window(left,top,right-left,bottom-top)
                        z=src.read(1,window=window,masked=True).astype(float).filled(np.nan)
                        f=terrain_features(z,src.window_transform(window))
                        dst.write(f[:,row-top:row-top+h,col-left:col-left+w],window=Window(col,row,w,h))
                        done+=1
                        if progress:progress(int(100*done/total),'DEM window')
        tmp.replace(output);return output
    except BaseException:
        tmp.unlink(missing_ok=True);raise
