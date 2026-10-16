import gzip
import json
import math
import urllib.request
import urllib.parse
from pathlib import Path
from datetime import datetime, timezone
from shapely.geometry import Polygon,LineString,shape,mapping,box
from shapely.ops import unary_union
from .project import atomic_json,to_wgs84,cancelled


def fetch_osm(project,directory,cancel=lambda:False):
    boundary=to_wgs84(shape(project['boundary']),project['crs_wkt'])
    west,south,east,north=boundary.bounds
    if east-west>0.25 or north-south>0.25:
        raise ValueError('Для этого загрузчика выберите район до 0.25° по каждой оси; крупный район требует пакетных запросов Overpass')
    query=f'''[out:json][timeout:90];(way[building]({south},{west},{north},{east});
        way[highway]({south},{west},{north},{east});way[waterway]({south},{west},{north},{east}););out geom;'''
    req=urllib.request.Request('https://overpass-api.de/api/interpreter',
        data=urllib.parse.urlencode({'data':query}).encode(),headers={'User-Agent':'City3D-project-references/1.0'})
    cancelled(cancel)
    with urllib.request.urlopen(req,timeout=120) as response:payload=json.load(response)
    if payload.get('remark'):raise ValueError('Overpass не подтвердил полный результат: '+str(payload['remark']))
    features=[]
    for item in payload.get('elements',[]):
        cancelled(cancel)
        points=[(v['lon'],v['lat']) for v in item.get('geometry',[])]
        if len(points)<2:continue
        tags=item.get('tags',{})
        geom=Polygon(points) if 'building' in tags and len(points)>=4 and points[0]==points[-1] else LineString(points)
        if not geom.is_valid:continue
        clipped=geom.intersection(boundary)
        if clipped.is_empty:continue
        features.append({'type':'Feature','geometry':mapping(clipped),'properties':dict(tags,
                         osm_id=item['id'],osm_type=item['type'],area_name=project['name'],
                         project_id=project['project_id'],revision_id=project['revision_id'],
                         ground_truth_status='unreviewed',source='OpenStreetMap contributors',license='ODbL')})
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    output=directory/'osm_ways.geojson'
    atomic_json(output,{'type':'FeatureCollection','features':features})
    atomic_json(directory/'osm_provenance.json',{'source':'https://www.openstreetmap.org/copyright',
        'retrieved_at':datetime.now(timezone.utc).isoformat(),'query':query,'features':len(features),
        'coverage':'building/highway/waterway ways only; multipolygon relations not imported',
        'project_id':project['project_id'],'area_name':project['name'],'reviewed':False})
    return output


def fetch_dem(project,directory,cancel=lambda:False):
    """Mapzen Skadi multi-source DEM, explicitly NOT pure NASA SRTM."""
    import rasterio
    from rasterio.merge import merge
    from rasterio.mask import mask
    from rasterio.io import MemoryFile
    boundary=to_wgs84(shape(project['boundary']),project['crs_wkt'])
    west,south,east,north=boundary.bounds
    tiles=[(lat,lon) for lat in range(math.floor(south),math.ceil(north))
           for lon in range(math.floor(west),math.ceil(east)) if boundary.intersects(box(lon,lat,lon+1,lat+1))]
    if len(tiles)>16:raise ValueError('Больше 16 DEM-тайлов: нужна отдельная пакетная загрузка')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    paths=[];urls=[]
    for lat,lon in tiles:
        cancelled(cancel)
        name=f"{'N' if lat>=0 else 'S'}{abs(lat):02d}{'E' if lon>=0 else 'W'}{abs(lon):03d}"
        url=f'https://s3.amazonaws.com/elevation-tiles-prod/skadi/{name[:3]}/{name}.hgt.gz'
        path=directory/(name+'.hgt')
        req=urllib.request.Request(url,headers={'User-Agent':'City3D-project-references/1.0'})
        with urllib.request.urlopen(req,timeout=90) as response:
            raw=gzip.decompress(response.read())
        if len(raw) not in (1201*1201*2,3601*3601*2):raise ValueError('Неожиданный размер HGT: '+name)
        path.write_bytes(raw);paths.append(path);urls.append(url)
    datasets=[rasterio.open(p) for p in paths]
    try:
        mosaic,affine=merge(datasets,nodata=-32768)
        profile=datasets[0].profile.copy();profile.update(driver='GTiff',height=mosaic.shape[1],width=mosaic.shape[2],transform=affine,nodata=-32768)
        with MemoryFile() as memory:
            with memory.open(**profile) as temporary:
                temporary.write(mosaic)
                cropped,transform=mask(temporary,[mapping(boundary)],crop=True,nodata=-32768)
            profile.update(height=cropped.shape[1],width=cropped.shape[2],transform=transform,compress='deflate')
            output=directory/'dem_skadi.tif'
            with rasterio.open(output,'w',**profile) as writer:writer.write(cropped)
        atomic_json(directory/'dem_provenance.json',dict(source='Mapzen Terrain Tiles / Skadi, multiple source DEM',
            urls=urls,area_name=project['name'],project_id=project['project_id'],reviewed=False,
            source_documentation='https://github.com/tilezen/joerd/blob/master/docs/attribution.md'))
        return output
    finally:
        for dataset in datasets:dataset.close()


def sheets_for_geometry(cursor,geometry):
    cursor.execute("SELECT to_regclass('topomap.topomaps')")
    if not cursor.fetchone()[0]:return []
    cursor.execute('''SELECT DISTINCT name FROM topomap.topomaps
        WHERE scale=50000 AND ST_Intersects("SP_GEOMETRY",ST_SetSRID(ST_GeomFromGeoJSON(%s),4326)) ORDER BY name''',
        (json.dumps(mapping(geometry)),))
    return [row[0] for row in cursor.fetchall()]


def reference_geometry(path):
    path=Path(path)
    if path.suffix.lower() in ('.las','.laz'):
        import laspy
        with laspy.open(path) as reader:
            h=reader.header;c=h.parse_crs()
            if c is None:raise ValueError('CRS эталона не определён')
            return to_wgs84(box(h.mins[0],h.mins[1],h.maxs[0],h.maxs[1]),c.to_wkt())
    if path.suffix.lower()=='.geojson':
        data=json.loads(path.read_text())
        if data.get('crs'):raise ValueError('GeoJSON с нестандартным CRS требует явного преобразования')
        return unary_union([shape(f['geometry']) for f in data.get('features',[]) if f.get('geometry')])
    if path.suffix.lower() in ('.tif','.tiff'):
        import rasterio
        with rasterio.open(path) as reader:
            if reader.crs is None:raise ValueError('CRS растра неизвестен')
            return to_wgs84(box(*reader.bounds),reader.crs.to_wkt())
    raise ValueError('Определение охвата формата ещё не подключено')


def install_references(ns):
    from PyQt5.QtCore import QThread
    from PyQt5.QtWidgets import QTableWidgetItem,QMessageBox
    from city3d_tab1.database import connect
    base=ns['ReferenceDBTab']
    class References(base):
        def __init__(self,mw):
            super().__init__(mw)
            self.btn_download_srtm.setText('Скачать DEM Skadi (несколько источников)')
            self.btn_download_srtm.setToolTip('Не чистый NASA SRTM; происхождение сохраняется рядом с файлом.')
        def _download_external(self,kind):
            p=getattr(self.main_window,'active_project',None)
            if not p:QMessageBox.warning(self,'Район','Сначала примените рабочую подборку и границу.');return
            if getattr(self,'_worker',None) is not None and self._worker.isRunning():return
            self.download_project=p
            directory=Path(ns['WORKSPACE'])/'references'/p['revision_id']
            def task(log,progress):
                cancel=lambda:QThread.currentThread().isInterruptionRequested()
                outputs=[]
                if kind in ('osm','all'):outputs.append(fetch_osm(p,directory,cancel))
                if kind in ('srtm','all'):outputs.append(fetch_dem(p,directory,cancel))
                # SQLite ReferenceDatabase is opened by original GUI; do not share its connection with this thread.
                self.downloaded_paths=[str(x) for x in outputs]
                for x in outputs:log('Скачано и обрезано по границе: '+str(x),'SUCCESS')
                return True
            self.downloaded_paths=[]
            w=ns['WorkerThread'](task);w.log_signal.connect(lambda text,level:self._dl(level+': '+text))
            w.task_done.connect(self.download_done);self._worker=w;w.start()
        def download_done(self,ok):
            if ok:
                # Keep GIS/DEM out of references_clouds and its point-cloud loader.
                import sqlite3
                connection=sqlite3.connect(self.db.db_path)
                connection.execute('''CREATE TABLE IF NOT EXISTS city3d_external_references
                    (path TEXT PRIMARY KEY,name TEXT,kind TEXT,metadata TEXT,created_at TEXT)''')
                try:
                    for path in self.downloaded_paths:
                        p=self.download_project
                        connection.execute('INSERT OR REPLACE INTO city3d_external_references VALUES (?,?,?,?,?)',
                            (path,Path(path).stem,'dem' if path.endswith('.tif') else 'gis',json.dumps(p),datetime.now(timezone.utc).isoformat()))
                    connection.commit()
                finally:connection.close()
                self.refresh()
            self._dl('Готово' if ok else 'Ошибка загрузки; см. журнал')
        def refresh(self):
            super().refresh()
            import sqlite3
            connection=sqlite3.connect(self.db.db_path)
            try:
                exists=connection.execute("SELECT name FROM sqlite_master WHERE name='city3d_external_references'").fetchone()
                if exists:
                    for path,name,kind,date in connection.execute('SELECT path,name,kind,created_at FROM city3d_external_references'):
                        row=self.table.rowCount();self.table.insertRow(row)
                        for col,value in enumerate((kind,name,path,date)):self.table.setItem(row,col,QTableWidgetItem(value))
            finally:connection.close()
            self.table.setColumnCount(5)
            self.table.setHorizontalHeaderLabels(['Категория','Название','Путь','Дата','Листы 1:50000'])
            for row in range(self.table.rowCount()):self.table.setItem(row,4,QTableWidgetItem('не определено'))
            # Explicit supplementary action avoids a long blocking DB/geospatial pass on every GUI refresh.
            if not hasattr(self,'sheet_button'):
                from PyQt5.QtWidgets import QPushButton
                self.sheet_button=QPushButton('Определить листы 1:50000 по слою topomap.topomaps')
                self.sheet_button.clicked.connect(self.update_sheets);self.layout().addWidget(self.sheet_button)
            if self.table.rowCount():
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(0,self.update_sheets)
        def update_sheets(self):
            if getattr(self,'_sheet_worker',None) is not None and self._sheet_worker.isRunning():return
            paths=[self.table.item(row,2).text() for row in range(self.table.rowCount())]
            def task(log,progress):
                connection=connect(ns['DB_PARAMS'])
                self.sheet_results={}
                try:
                    with connection.cursor() as cursor:
                        cursor.execute("SET statement_timeout='10s'")
                        for path in paths:
                            try:self.sheet_results[path]=', '.join(sheets_for_geometry(cursor,reference_geometry(path))) or 'нет листов 1:50000 в слое'
                            except Exception as exc:
                                connection.rollback();self.sheet_results[path]=str(exc)
                finally:connection.close()
                return True
            def done(ok):
                if ok:
                    for row in range(self.table.rowCount()):
                        path=self.table.item(row,2).text()
                        self.table.setItem(row,4,QTableWidgetItem(self.sheet_results.get(path,'не определено')))
            w=ns['WorkerThread'](task);w.log_signal.connect(self.log.log);w.task_done.connect(done);self._sheet_worker=w;w.start()
    ns['ReferenceDBTab']=References
