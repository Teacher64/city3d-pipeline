"""LAS preview, compact controls and reversible project membership removal."""
import json
import math
import traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QPointF
from PyQt5.QtGui import QPainter, QColor, QImage
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLabel, QTabWidget, QMessageBox, QSizePolicy)


def read_preview(records, cancel=lambda: False, limit=200000):
    import laspy
    from pyproj import CRS
    if not records: raise ValueError('Выберите строки LAS/LAZ для просмотра.')
    if len(records)>1:
        crss=[CRS.from_wkt(r['crs_wkt']) if r.get('crs_wkt') else None for r in records]
        if any(c is None for c in crss) or any(not crss[0].equals(c, ignore_axis_order=True) for c in crss[1:]):
            raise ValueError('Для совместного просмотра нужны одинаковые CRS. Выделите одну строку: отдельный LAS можно смотреть без CRS.')
    chunks=[]; total=0
    budget=max(1,limit//len(records))
    for record in records:
        path=Path(record['path'])
        if path.suffix.lower() not in ('.las','.laz'): raise ValueError('Предпросмотр этого исправления: LAS/LAZ.')
        with laspy.open(path) as reader:
            n=reader.header.point_count; total+=n
            stride=max(1,math.ceil(n/budget)); offset=0
            for batch in reader.chunk_iterator(250000):
                if cancel(): raise RuntimeError('Предпросмотр отменён')
                start=(-offset)%stride
                ids=np.arange(start,len(batch),stride)
                if len(ids): chunks.append(np.column_stack((np.asarray(batch.x)[ids],np.asarray(batch.y)[ids],np.asarray(batch.z)[ids])))
                offset+=len(batch)
    if not chunks: raise ValueError('В выбранных LAS нет точек.')
    points=np.concatenate(chunks)
    points=points[np.isfinite(points).all(axis=1)]
    if not len(points): raise ValueError('Нет точек с конечными XYZ.')
    if len(points)>limit: points=points[np.linspace(0,len(points)-1,limit,dtype=int)]
    return dict(points=points,total=total,files=len(records))


class PlanView(QWidget):
    """CPU plan view: absolute XY, wheel zoom, left-button pan. No OpenGL."""
    def __init__(self,parent=None):
        super().__init__(parent); self.points=np.empty((0,3)); self.center=np.zeros(2)
        self.scale=1.;self.drag=None;self.image=None;self.setMouseTracking(True)
        self.setMinimumSize(100,100)
    def set_points(self,points):
        self.points=np.asarray(points); self.reset()
    def reset(self):
        if len(self.points):
            lo=self.points[:,:2].min(axis=0); hi=self.points[:,:2].max(axis=0)
            self.center=(lo+hi)/2
            self.scale=min(max(1,self.width()-30)/max(hi[0]-lo[0],.01), max(1,self.height()-30)/max(hi[1]-lo[1],.01))
        self.image=None; self.update()
    def world(self,pos):
        return self.center+np.array([pos.x()-self.width()/2, self.height()/2-pos.y()])/self.scale
    def paintEvent(self,event):
        if self.image is None:
            w,h=max(1,self.width()),max(1,self.height())
            a=np.empty((h,w,4),dtype=np.uint8);a[:]=[32,34,38,255]
            if len(self.points):
                xy=(self.points[:,:2]-self.center)*self.scale
                x=np.rint(xy[:,0]+w/2).astype(int);y=np.rint(h/2-xy[:,1]).astype(int)
                ok=(x>=0)&(x<w)&(y>=0)&(y<h)
                z=self.points[:,2];t=(z-z.min())/max(float(np.ptp(z)),1e-9)
                colors=np.column_stack((50+205*t,210-80*t,245-170*t,np.full(len(t),255))).astype(np.uint8)
                a[y[ok],x[ok]]=colors[ok]
            self.image=QImage(a.data,w,h,4*w,QImage.Format_RGBA8888).copy()
        p=QPainter(self);p.drawImage(0,0,self.image);p.setPen(QColor('white'))
        p.drawText(10,20,'План XY • колесо — масштаб • левая кнопка — перемещение')
        if not len(self.points):p.drawText(10,45,'Добавьте LAS/LAZ или выделите строку реестра.')
        p.end()
    def resizeEvent(self,event): self.image=None;self.update()
    def wheelEvent(self,event):
        before=self.world(event.position());self.scale*=1.25**(event.angleDelta().y()/120)
        self.scale=min(max(self.scale,1e-9),1e9)
        self.center+=before-self.world(event.position());self.image=None;self.update();event.accept()
    def mousePressEvent(self,event):
        if event.button()==Qt.LeftButton:self.drag=event.pos()
    def mouseReleaseEvent(self,event):self.drag=None
    def mouseMoveEvent(self,event):
        if self.drag is not None:
            d=event.pos()-self.drag;self.center+=np.array([-d.x(),d.y()])/self.scale
            self.drag=event.pos();self.image=None;self.update()
        x,y=self.world(event.pos());self.setToolTip(f'X: {x:.3f}   Y: {y:.3f} (координаты текущего вида)')


def install(ns):
    from city3d_fixes.project import atomic_json
    workspace=Path(ns['WORKSPACE'])
    removed_file=workspace/'city3d_removed_inputs.json'
    base=ns['LoadDataTab']
    class Load(base):
        def __init__(self,*args,**kwargs):
            self._initializing=True;self._preview_pending=False;self._preview_points=None
            self._removed=set(json.loads(removed_file.read_text()) if removed_file.exists() else [])
            super().__init__(*args,**kwargs)
            self._initializing=False
            self._preview_timer=QTimer(self);self._preview_timer.setSingleShot(True)
            self._preview_timer.timeout.connect(self._automatic_preview)
            content=self._load_scroll.widget();layout=content.layout()
            panel=QWidget();grid=QGridLayout(panel);grid.setContentsMargins(0,0,0,0)
            captions={'Добавить файлы':'Добавить LAS…','Добавить папку':'Папка…','Записать реестр в БД':'Записать в БД',
                'Экспорт реестра JSON':'Реестр JSON','Сохранить выбор / название':'Сохранить выбор',
                'Отметить все':'Выбрать все','Снять все':'Снять выбор','Восстановить последний рабочий район':'Восстановить район'}
            buttons=[]
            for b in content.findChildren(QPushButton):
                if b.text() in captions:
                    b.setText(captions[b.text()]);buttons.append(b)
                elif b.text()=='Обзор...': b.hide()
            self.preview_button=QPushButton('Показать LAS');self.preview_button.clicked.connect(self.preview)
            self.remove_button=QPushButton('Убрать из проекта');self.remove_button.clicked.connect(self.remove_inputs)
            buttons.extend([self.preview_button,self.remove_button])
            for i,b in enumerate(buttons):grid.addWidget(b,i//2,i%2)
            layout.insertWidget(0,panel)
            self.setStyleSheet('QPushButton { min-height: 0px; padding: 3px 7px; font-weight: normal; }')
            for b in self.findChildren(QPushButton):
                b.setMinimumHeight(0);b.setMaximumHeight(30);b.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Fixed)
            self.btn_load.setText('Применить район')
            self.layout().setAlignment(self.btn_load,Qt.AlignLeft)
            self.btn_load.setMinimumWidth(180)
            self.views=QTabWidget();self.plan=PlanView();self.views.addTab(self.plan,'План XY')
            # replaceWidget detaches the old viewer without destroying its connections.
            self._load_splitter.replaceWidget(1,self.views)
            self.views.addTab(self.viewer,'3D');self.views.currentChanged.connect(self.update_3d)
            self.table.itemSelectionChanged.connect(self.schedule_preview)
            self.status.setMaximumHeight(64)
            self.status.setText('Добавьте LAS: предпросмотр появится автоматически. «Применить район» передаёт отмеченные файлы на остальные вкладки.')
            self._load_splitter.setSizes([440,850])
        def accept_record(self,record):
            if self._initializing and record['path'] in self._removed:return
            fresh=record['path'] not in self.records
            super().accept_record(record)
            if not self._initializing:
                self._removed.discard(record['path']);atomic_json(removed_file,sorted(self._removed))
                if fresh:
                    row=next(i for i in range(self.table.rowCount()) if self.table.item(i,0).text()==record['path'])
                    self.table.item(row,0).setCheckState(Qt.Checked)
                    self.table.selectRow(row)
                    if not self.area_name.text().strip():self.area_name.setText(Path(record['path']).stem)
                self.schedule_preview()
        def schedule_preview(self,*args):
            if self._initializing:return
            self._preview_pending=True;self._preview_timer.start(250)
        def _automatic_preview(self):
            if self.busy():self._preview_timer.start(250);return
            if self._preview_pending:self._preview_pending=False;self.preview()
        def preview(self):
            if self.busy():self.status.setText('Выполняется операция; предпросмотр доступен после её завершения.');return
            paths=self.selected_paths()
            records=[self.records[p] for p in paths] if paths else self.chosen()
            if not records:self.status.setText('Выделите строку LAS или отметьте файлы флажками.');return
            self.status.setText('Чтение LAS для предпросмотра…')
            self.start_job(lambda job:read_preview(records,job.isInterruptionRequested),self.show_preview)
        def show_preview(self,data):
            self._preview_points=data['points'];self.plan.set_points(data['points'])
            self.views.setCurrentIndex(0)
            self.status.setText(f"Просмотр: LAS {data['files']}, всего точек {data['total']:,}, на экране {len(data['points']):,}. Для работы на других вкладках: «Применить район».")
        def update_3d(self,index):
            if index!=1 or self._preview_points is None:return
            try:self.viewer.set_pointcloud(SimpleNamespace(points=self._preview_points))
            except Exception:self.show_error('3D: '+traceback.format_exc()+'\nДоступен просмотр «План XY».')
        def start_job(self,function,result=None):
            def guarded(value):
                try:
                    if result:result(value)
                except Exception:self.show_error(traceback.format_exc())
            def task(job):
                try:return function(job)
                except Exception as exc:raise RuntimeError(traceback.format_exc()) from exc
            super().start_job(task,guarded)
        def show_error(self,text):
            summary=next((line for line in reversed(text.splitlines()) if line.strip()),'Ошибка загрузки')
            super().show_error(summary+' — подробности в city3d_load_errors.log')
            self.status.setToolTip(text)
            log=workspace/'city3d_load_errors.log';log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('a',encoding='utf-8') as f:f.write(text+'\n\n')
        def activate(self,result):
            # Preserve full workset for processing. A sample is only used for display.
            original_display=self.viewer.set_pointcloud
            self.viewer.set_pointcloud=lambda cloud:None
            try:super().activate(result)
            finally:self.viewer.set_pointcloud=original_display
            points=result[1]['points'];step=max(1,math.ceil(len(points)/200000))
            self._preview_points=points[::step];self.plan.set_points(self._preview_points)
            self.views.setCurrentIndex(0)
        def remove_inputs(self):
            if self.busy() or any(w.isRunning() for w in ns.get('_ACTIVE_WORKERS',[])):
                self.status.setText('Дождитесь завершения операций перед удалением из проекта.');return
            paths=set(self.selected_paths())
            if not paths:self.status.setText('Выделите строки для удаления (Ctrl / Shift). Флажки задают состав обработки.');return
            if QMessageBox.question(self,'Убрать из проекта',f'Убрать {len(paths)} файлов из списка? Исходные LAS и архив результатов останутся на диске. Если затронут активный район, его потребуется применить заново.',QMessageBox.Yes|QMessageBox.No)!=QMessageBox.Yes:return
            mw=self.main_window;p=getattr(mw,'active_project',None)
            affected=bool(p and paths.intersection(r['path'] for r in p['sources']))
            if affected:
                state=workspace/'city3d_project_state.json'
                if state.exists():
                    # Keep the previous state as an archive, but never restore it implicitly.
                    archive=workspace/'projects'/('removed_state_'+p['revision_id']+'.json')
                    archive.parent.mkdir(parents=True,exist_ok=True);atomic_json(archive,json.loads(state.read_text()))
                    state.unlink()
                mw.active_project=None
                for name in ('pcd','pcd_path','pcd_data','clean_pcd','clean_pcd_path','cropped_pcd','cropped_pcd_path','cropped_data','input_crs_wkt','input_asset_id','input_inspection_id'):
                    setattr(mw,name,None)
                for i in range(mw.tabs.count()):
                    t=mw.tabs.widget(i)
                    if hasattr(t,'m5_las_list'):t.m5_las_list.clear()
                mw.scanning_type='unknown'
                self.crs_detector.crs=None
                mw.refresh_project_banner()
            self.table.blockSignals(True)
            for i in reversed(range(self.table.rowCount())):
                path=self.table.item(i,0).text()
                if path in paths:self.table.removeRow(i);self.records.pop(path,None)
            self.table.blockSignals(False)
            self._removed.update(paths);atomic_json(removed_file,sorted(self._removed));self.save_draft()
            self._preview_timer.stop();self._preview_pending=False;self._preview_points=None
            self.plan.set_points(np.empty((0,3)));self.views.setCurrentIndex(0)
            if self.viewer.scatter is not None:
                self.viewer.view.removeItem(self.viewer.scatter);self.viewer.scatter=None
            self.viewer.pcd=None;self.viewer.center=None;self.viewer.info_label.setText('Нет данных')
            self.file_path.clear();self.crs_display.clear();self.crs_source.clear();self.info_label.setText('Файлы убраны из списка')
            self.status.setText('Файлы убраны; исходники сохранены.'+(' Примените оставшиеся LAS заново.' if affected else ''))
    ns['LoadDataTab']=Load
    base_window=ns['MainWindow']
    class Window(base_window):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.refresh_project_banner()
        def refresh_project_banner(self,*args):
            super().refresh_project_banner(*args)
            for label in getattr(self,'project_labels',[]):label.setVisible(bool(self.active_project))
    ns['MainWindow']=Window

# Additional layer over the bundled LAS/UI repair.
import hashlib
import threading
import secrets
import mimetypes
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime, timezone
import uuid

_previous_install = install


def sparse_points(path, cancel=lambda: False, limit=16000):
    """Read <=32 small windows, never scan every record merely for display."""
    import laspy
    with laspy.open(path,read_evlrs=False) as reader:
        n=reader.header.point_count
        if not n:raise ValueError('Пустой LAS')
        windows=min(8 if reader.header.are_points_compressed else 32,max(1,n//512))
        size=min(n,max(1,limit//windows))
        starts=np.linspace(0,max(0,n-size),windows,dtype=np.int64)
        arrays=[]
        for start in starts:
            if cancel():raise RuntimeError('Просмотр отменён')
            reader.seek(int(start));p=reader.read_points(size)
            arrays.append(np.column_stack((p.x,p.y,p.z)))
        return np.concatenate(arrays),int(n)


def quick_record(path,override=None,scan_type='unknown'):
    import laspy
    from city3d_tab1.catalog import resolve_crs
    p=Path(path).resolve();s=p.stat()
    with laspy.open(p,read_evlrs=True) as reader:
        h=reader.header;c,source,evidence,errors=resolve_crs(h,p,override)
        bounds=np.r_[h.mins,h.maxs].tolist()
        return dict(path=str(p),size=s.st_size,mtime_ns=s.st_mtime_ns,format=p.suffix.lower(),
            scan_type=scan_type,state='header_only',reviewed=False,observed_at=datetime.now(timezone.utc).isoformat(),
            point_count=int(h.point_count),classes={},bounds=bounds,crs_wkt=c.to_wkt() if c else None,
            crs_authority=c.to_authority() if c else None,crs_source=source,crs_evidence=evidence,warnings=errors,
            footprint_method='header_bbox_unverified',footprint_wgs84=None,sha256=None,
            file_id=str(uuid.uuid5(uuid.NAMESPACE_URL,'city3d:unverified-path:'+str(p))),inspection_id=str(uuid.uuid4()))


def geographic_scene(records,cancel=lambda:False):
    from pyproj import CRS
    from pyproj.transformer import TransformerGroup
    files=[]
    for r in records:
        if cancel():raise RuntimeError('Просмотр отменён')
        item=dict(name=Path(r['path']).name)
        try:
            if not r.get('crs_wkt'):raise ValueError('CRS не задана или конфликтует: назначьте её выделенной строке. Зону по координатам не угадываем.')
            c=CRS.from_wkt(r['crs_wkt']).to_2d()
            group=TransformerGroup(c,'EPSG:4326',always_xy=True,allow_ballpark=False)
            if not group.transformers or not group.best_available:raise ValueError('Недоступны параметры/сетки точного преобразования CRS в WGS84.')
            stat=Path(r['path']).stat()
            if (stat.st_size,stat.st_mtime_ns)!=(r['size'],r['mtime_ns']):raise ValueError('Файл изменился: добавьте его повторно.')
            tr=group.transformers[0];pts,n=sparse_points(r['path'],cancel,min(16000,max(1,100000//max(1,len(records)))))
            lon,lat=tr.transform(pts[:,0],pts[:,1],errcheck=True)
            xy=np.column_stack((lon,lat));ok=np.isfinite(xy).all(axis=1)&(np.abs(xy[:,0])<=180)&(np.abs(xy[:,1])<=90)
            if not ok.all():raise ValueError('Преобразование дало недопустимые координаты; проверьте назначенную CRS.')
            # Header envelope for framing only; never written as a work boundary.
            b=r['bounds'];t=np.linspace(0,1,17)
            x=np.r_[b[0]+t*(b[3]-b[0]),np.full(17,b[3]),b[3]-t*(b[3]-b[0]),np.full(17,b[0])]
            y=np.r_[np.full(17,b[1]),b[1]+t*(b[4]-b[1]),np.full(17,b[4]),b[4]-t*(b[4]-b[1])]
            ex,ey=tr.transform(x,y,errcheck=True);extent=np.column_stack((ex,ey))
            if not np.isfinite(extent).all() or (np.abs(extent[:,0])>180).any() or (np.abs(extent[:,1])>90).any():raise ValueError('Неверный охват заголовка/CRS')
            center=[float(np.degrees(np.arctan2(np.sin(np.radians(lon)).mean(),np.cos(np.radians(lon)).mean()))),float(np.mean(lat))]
            item.update(points=xy.tolist(),total=n,center=center,extent=extent.tolist(),transform=f'{c.to_string()} → WGS84; точность операции {tr.accuracy} м')
        except Exception as exc:item['error']=str(exc)
        files.append(item)
    return dict(files=files)


class GlobeServer:
    def __init__(self,assets,cache=None,mbtiles=None,services=None):
        self.services=ServiceStore(services or Path(assets).parent/'city3d_services.json',Path(cache or Path(assets).parent/'city3d_tile_cache')/'services')
        self.tiles=TileStore(cache or Path(assets).parent/'city3d_tile_cache',mbtiles)
        self.scene=dict(revision=0,files=[]);self.lock=threading.Lock();token=secrets.token_urlsafe(24)
        owner=self;root=Path(assets).resolve()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                from urllib.parse import urlsplit,unquote
                path=unquote(urlsplit(self.path).path)
                prefix='/'+token+'/'
                if not path.startswith(prefix):self.send_error(404);return
                rel=path[len(prefix):] or 'globe.html'
                if rel.startswith('credits/'):
                    data=json.dumps(dict(text=owner.services.vendor.attribution(rel.split('/')[-1])),ensure_ascii=False).encode();mime='application/json'
                elif rel=='providers':
                    data=json.dumps(owner.services.public(),ensure_ascii=False).encode();mime='application/json'
                elif rel.startswith('services/'):
                    try:
                        _,id,mode,z,x,y=rel.split('/')
                        data,mime=owner.services.get(id,mode,int(z),int(x),int(y.removesuffix('.png')),self.headers.get('Referer',''))
                    except Exception:
                        self.send_error(404,'Service tile unavailable');return
                elif rel.startswith('tiles/'):
                    try:
                        _,mode,z,x,y=rel.split('/')
                        data,mime=owner.tiles.get(mode,int(z),int(x),int(y.removesuffix('.png')),self.headers.get('Referer',''))
                    except Exception:
                        import base64
                        data=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNgAAIAAAUAAarVyFEAAAAASUVORK5CYII=');mime='image/png'
                elif rel=='scene':
                    with owner.lock:data=json.dumps(owner.scene,ensure_ascii=False,allow_nan=False).encode()
                    mime='application/json'
                else:
                    p=(root/rel).resolve()
                    if not p.is_relative_to(root) or not p.is_file():self.send_error(404);return
                    data=p.read_bytes();mime=mimetypes.guess_type(p.name)[0] or 'application/octet-stream'
                self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)))
                self.send_header('Cache-Control','no-store' if rel in ('scene','providers') or rel.startswith('credits/') or rel.startswith(('tiles/','services/')) else 'public, max-age=86400')
                self.send_header('X-Content-Type-Options','nosniff');self.end_headers()
                try:self.wfile.write(data)
                except (BrokenPipeError,ConnectionResetError):pass
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.server.daemon_threads=True
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.url=f'http://127.0.0.1:{self.server.server_port}/{token}/'
    def update(self,data):
        with self.lock:self.scene=dict(data,revision=self.scene['revision']+1)
    def close(self):self.server.shutdown();self.server.server_close()


def install(ns):
    _previous_install(ns)
    from city3d_fixes.project import atomic_json
    from city3d_tab1.catalog import discover,inspect_file
    base=ns['LoadDataTab']
    class GeoLoad(base):
        def __init__(self,*args,**kw):
            self._globe=None;self._scene=None;self._continuation=None
            super().__init__(*args,**kw)
            self._services_config=Path(ns['WORKSPACE'])/'city3d_services.json'
            self._mbtiles_config=Path(ns['WORKSPACE'])/'city3d_globe_settings.json'
            config=json.loads(self._mbtiles_config.read_text()) if self._mbtiles_config.exists() else {}
            self._mbtiles=config.get('mbtiles')
            self.globe_button=QPushButton('Глобус Земли ↗');self.globe_button.clicked.connect(self.open_globe)
            self.audit_button=QPushButton('Проверить LAS полностью');self.audit_button.clicked.connect(lambda:self.ensure_verified(list(self.records.values()),lambda:self.status.setText('Полная проверка завершена.')))
            grid=self.preview_button.parentWidget().layout();row=grid.rowCount();grid.addWidget(self.globe_button,row,0);grid.addWidget(self.audit_button,row,1)
            self.mbtiles_button=QPushButton('База OSM (MBTiles)…');self.mbtiles_button.clicked.connect(self.choose_mbtiles)
            grid.addWidget(self.mbtiles_button,row+1,0)
            self.services_button=QPushButton('Геосервисы…');self.services_button.clicked.connect(self.edit_services);self.services_button.setMaximumHeight(30)
            grid.addWidget(self.services_button,row+1,1)
            for b in (self.globe_button,self.audit_button,self.mbtiles_button):b.setMaximumHeight(30)
            self.btn_load.setText('Подготовить рабочий район')
            self.status.setText('Быстрое добавление по заголовкам. «Глобус Земли» показывает LAS разных CRS; полная проверка запускается перед обработкой/БД.')
        def index_paths(self,paths,override=None,assignment=None):
            if self.busy():self.status.setText('Дождитесь текущей операции.');return
            recursive=self.recursive.isChecked();scan=self.current_scanning_type()
            def work(job):
                count=0
                for path in discover(paths,recursive):
                    if job.isInterruptionRequested():raise RuntimeError('Добавление отменено')
                    if Path(path).suffix.lower() not in ('.las','.laz'):
                        job.error.emit('Быстрое добавление поддерживает LAS/LAZ: '+str(path));continue
                    try:
                        r=quick_record(path,override,scan)
                        if assignment:r['crs_assignment']=assignment
                        self.journal.save(r);job.item.emit(r);count+=1
                    except Exception as exc:job.error.emit(str(exc))
                return count
            self.start_job(work,lambda count:self.status.setText(f'Добавлено по заголовкам: {count}. Классы и SHA256 ещё не проверены.'))
        def accept_record(self,r):
            old_scan=None
            if hasattr(self,'table') and self.table.columnCount()>5:
                for i in range(self.table.rowCount()):
                    if self.table.item(i,0).text()==r['path'] and self.table.cellWidget(i,5):old_scan=self.table.cellWidget(i,5).currentText()
            super().accept_record(r)
            if r['path'] not in self.records:return
            row=next(i for i in range(self.table.rowCount()) if self.table.item(i,0).text()==r['path'])
            if old_scan is not None and self.table.columnCount()>5:self.table.cellWidget(row,5).setCurrentText(old_scan)
            if r.get('state')=='header_only':
                self.table.item(row,3).setText('Не подсчитаны — заголовок')
                self.table.item(row,4).setText('Быстрый обзор; полная проверка не выполнена')
        def preview(self):
            if self.busy():return
            records=self.chosen() or [self.records[p] for p in self.selected_paths()]
            if not records:self.status.setText('Добавьте LAS/LAZ или выделите строки.');return
            snapshot=[dict(r) for r in records]
            self.status.setText('Выборочное чтение LAS и преобразование для глобуса…')
            def done(data):
                self._scene=data
                if self._globe:self._globe.update(data)
                points=[p for f in data['files'] if not f.get('error') for p in f['points']]
                if points:
                    self.plan.set_points(np.column_stack((np.asarray(points),np.zeros(len(points)))))
                    self.views.setTabText(0,'Обзор WGS84 (долгота/широта)');self.views.setCurrentIndex(0)
                else:self.plan.set_points(np.empty((0,3)))
                self.views.setTabEnabled(1,False)
                self._preview_points=None # no fake height or mixed CRS sent to native 3D viewer
                n=sum('error' not in f for f in data['files'])
                messages=[f['name']+': '+f['error'] for f in data['files'] if f.get('error')]
                self.status.setText(f'Готово для глобуса: {n}/{len(data["files"])}. Нажмите «Глобус Земли ↗».'+(' Есть LAS без пригодной CRS — подробности в подсказке.' if messages else ''))
                self.status.setToolTip('\n'.join(messages))
                if self._open_when_ready:self._open_when_ready=False;self.open_globe()
            self._open_when_ready=getattr(self,'_open_when_ready',False)
            self.start_job(lambda job:geographic_scene(snapshot,job.isInterruptionRequested),done)
        def edit_services(self):
            from PyQt5.QtWidgets import QDialog,QPlainTextEdit,QDialogButtonBox
            dialog=QDialog(self);dialog.setWindowTitle('Реестр XYZ / WMS / WMTS');dialog.resize(850,650)
            layout=QVBoxLayout(dialog);hint=QLabel('Редактируйте параметры из GetCapabilities поставщика. WMTS: WebMercator XYZ. ЕЭКО включается после задания доступа и layers. Кэширование задаётся cache_allowed.');hint.setWordWrap(True);layout.addWidget(hint)
            edit=QPlainTextEdit();edit.setPlainText(self._services_config.read_text() if self._services_config.exists() else json.dumps(DEFAULT_SERVICES,ensure_ascii=False,indent=2));layout.addWidget(edit)
            buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel);layout.addWidget(buttons);buttons.rejected.connect(dialog.reject)
            def save():
                try:
                    data=validate_services(json.loads(edit.toPlainText()));atomic_json(self._services_config,data);dialog.accept();self.status.setText('Реестр сохранён. Обновите страницу глобуса для нового списка подложек.')
                except Exception as exc:QMessageBox.warning(dialog,'Настройки сервиса',str(exc))
            buttons.accepted.connect(save);dialog.exec_()
        def choose_mbtiles(self):
            from PyQt5.QtWidgets import QFileDialog
            path,_=QFileDialog.getOpenFileName(self,'Растровая база OSM','','MBTiles (*.mbtiles)')
            if not path:return
            try:
                store=self._globe.tiles if self._globe else TileStore(Path(ns['WORKSPACE'])/'cache/osm')
                metadata=store.set_mbtiles(path);self._mbtiles=str(Path(path).resolve())
                atomic_json(self._mbtiles_config,dict(mbtiles=self._mbtiles))
                self.status.setText('MBTiles подключена: '+metadata.get('name',Path(path).name)+'. В глобусе выберите «OSM: локальная MBTiles».')
            except Exception as exc:self.show_error(str(exc))
        def activate(self,result):
            super().activate(result)
            self.views.setTabText(0,'План XY рабочего района');self.views.setTabEnabled(1,True)
        def open_globe(self):
            if self._scene is None:
                self._open_when_ready=True
                if not self.busy():self.preview()
                else:self.schedule_preview()
                return
            if self._globe is None:
                self._globe=GlobeServer(Path(__file__).with_name('city3d_globe_assets'),Path(ns['WORKSPACE'])/'cache/osm',self._mbtiles,self._services_config)
                import atexit
                atexit.register(self._globe.close)
            self._globe.update(self._scene)
            if not webbrowser.open(self._globe.url):self.status.setText('Откройте адрес в браузере: '+self._globe.url)
        def ensure_verified(self,records,callback):
            if self.busy():self.status.setText('Дождитесь текущей операции.');return
            pending=[dict(r) for r in records if not r.get('sha256') or r.get('state')=='header_only' or (Path(r['path']).stat().st_size,Path(r['path']).stat().st_mtime_ns)!=(r['size'],r['mtime_ns'])]
            if not pending:callback();return
            self._preview_pending=False;self._preview_timer.stop();self._continuation=callback
            self.status.setText('Полная проверка LAS перед обработкой/БД: классы, SHA256, координаты. Это может занять время.')
            def work(job):
                for old in pending:
                    r=inspect_file(old['path'],old.get('crs_wkt') if old.get('crs_source')=='operator' else None,old.get('scan_type','unknown'),job.isInterruptionRequested,job.progress.emit)
                    if old.get('crs_assignment'):r['crs_assignment']=old['crs_assignment']
                    self.journal.save(r);job.item.emit(r)
                return True
            def done(value):
                self._preview_pending=False;self._preview_timer.stop()
                def resume():
                    if self.busy():QTimer.singleShot(30,resume);return
                    fn=self._continuation;self._continuation=None
                    if fn:fn()
                QTimer.singleShot(30,resume)
            self.start_job(work,done)
        def build(self,polygon=None,ratio=.05):
            records=self.chosen()
            if not records:self.status.setText('Отметьте LAS флажками.');return
            def apply():super(GeoLoad,self).build(polygon,ratio)
            self.ensure_verified(records,apply)
        def sync_db(self):self.ensure_verified(list(self.records.values()),lambda:super(GeoLoad,self).sync_db())
        def remove_inputs(self):
            before=set(self.records)
            super().remove_inputs()
            if set(self.records)==before:return
            self._scene=None
            if self._globe:self._globe.update(dict(files=[]))
    ns['LoadDataTab']=GeoLoad

import sqlite3
import time
import re
import urllib.request
import urllib.error
from email.utils import parsedate_to_datetime


class TileStore:
    def __init__(self,root,mbtiles=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.mbtiles=Path(mbtiles) if mbtiles else None
        self.lock=threading.Lock();self.network=threading.Semaphore(4)
        self.db=self.root/'osm_cache.sqlite3'
        with sqlite3.connect(self.db) as c:
            c.execute('PRAGMA journal_mode=WAL')
            c.execute('CREATE TABLE IF NOT EXISTS tiles(z INTEGER,x INTEGER,y INTEGER,data BLOB,mime TEXT,expires REAL,etag TEXT,modified TEXT,PRIMARY KEY(z,x,y))')
    def set_mbtiles(self,path):
        p=Path(path).resolve()
        with sqlite3.connect(p.as_uri()+'?mode=ro',uri=True) as c:
            metadata=dict(c.execute('SELECT name,value FROM metadata'))
            if metadata.get('format','').lower() not in ('png','jpg','jpeg','webp'):raise ValueError('Нужна растровая MBTiles (PNG/JPEG/WebP), не PBF/MVT.')
            c.execute('SELECT zoom_level,tile_column,tile_row,tile_data FROM tiles LIMIT 1').fetchone()
        self.mbtiles=p;return metadata
    def get(self,mode,z,x,y,referer=''):
        if not(0<=z<=19 and 0<=x<2**z and 0<=y<2**z):raise ValueError('Недопустимый адрес тайла')
        if mode=='mbtiles':
            if not self.mbtiles:raise FileNotFoundError('База MBTiles не выбрана')
            with sqlite3.connect(self.mbtiles.resolve().as_uri()+'?mode=ro',uri=True) as c:
                row=c.execute('SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?',(z,x,2**z-1-y)).fetchone()
            if not row:raise FileNotFoundError('Тайл отсутствует в MBTiles')
            data=bytes(row[0]);return data,self.mime(data)
        with sqlite3.connect(self.db) as c:row=c.execute('SELECT data,mime,expires,etag,modified FROM tiles WHERE z=? AND x=? AND y=?',(z,x,y)).fetchone()
        if row and (mode=='cache' or row[2]>time.time()):return bytes(row[0]),row[1]
        if mode=='cache':raise FileNotFoundError('Тайл ещё не сохранён')
        if mode!='online':raise ValueError('Неизвестный режим')
        headers={'User-Agent':'City3D-LocalGlobe/1.0'}
        if referer:headers['Referer']=referer
        if row:
            if row[3]:headers['If-None-Match']=row[3]
            if row[4]:headers['If-Modified-Since']=row[4]
        try:
            with self.network:
                try:
                    response=urllib.request.urlopen(urllib.request.Request(f'https://tile.openstreetmap.org/{z}/{x}/{y}.png',headers=headers),timeout=15)
                except urllib.error.HTTPError as exc:
                    if exc.code==304 and row:response=exc
                    else:raise
                with response:
                    h=response.headers;data=bytes(row[0]) if response.code==304 else response.read(5*1024*1024)
                mime=self.mime(data);expires=time.time()+604800
                age=re.search(r'max-age=(\d+)',h.get('Cache-Control',''))
                if age:expires=time.time()+int(age.group(1))
                elif h.get('Expires'):
                    try:expires=parsedate_to_datetime(h['Expires']).timestamp()
                    except Exception:pass
                if 'no-store' not in h.get('Cache-Control',''):
                    with sqlite3.connect(self.db) as c:c.execute('INSERT OR REPLACE INTO tiles VALUES(?,?,?,?,?,?,?,?)',(z,x,y,data,mime,expires,h.get('ETag',row[3] if row else None),h.get('Last-Modified',row[4] if row else None)))
                return data,mime
        except Exception:
            if row:return bytes(row[0]),row[1]
            raise
    @staticmethod
    def mime(data):
        if data.startswith(b'\x89PNG'):return 'image/png'
        if data.startswith(b'\xff\xd8'):return 'image/jpeg'
        if data.startswith(b'RIFF') and data[8:12]==b'WEBP':return 'image/webp'
        raise ValueError('Ответ не является растровым тайлом')

DEFAULT_SERVICES=[
 {'id':'nasa_gibs','name':'NASA GIBS — Blue Marble + рельеф','type':'WMTS','enabled':True,'url':'https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/wmts.cgi','layer':'BlueMarble_ShadedRelief_Bathymetry','matrix_set':'GoogleMapsCompatible_Level8','maxzoom':8,'format':'image/jpeg','attribution':'NASA GIBS / Earth Observatory','cache_allowed':True},
 {'id':'eeko','name':'ЕЭКО / Росреестр — настройте доступ и слой','type':'WMS','enabled':False,'url':'https://ngw.eeko.cgkipd.ru/api/resource/849582/wms','layers':'','version':'1.1.1','crs':'EPSG:3857','maxzoom':19,'attribution':'Росреестр / Центр геодезии, картографии и ИПД','cache_allowed':False,'note':'Адрес-пример из официальной инструкции ФППД. Актуальность и доступ не подтверждены. Укажите предоставленный оператору URL и имя слоя.'}
]


def validate_services(rows):
    from urllib.parse import urlsplit
    if not isinstance(rows,list):raise ValueError('Нужен JSON-массив сервисов')
    seen=set()
    for r in rows:
        key=r.get('id','')
        if not re.fullmatch(r'[a-zA-Z0-9_-]+',key) or key in seen:raise ValueError('Некорректный/повторный id')
        seen.add(key)
        if not r.get('enabled',True):continue
        if r.get('type') not in ('XYZ','WMS','WMTS','GOOGLE','BING','YANDEX'):raise ValueError('Тип сервиса: XYZ, WMS или WMTS')
        if urlsplit(r.get('url','')).scheme not in ('https','http'):raise ValueError('URL должен быть http(s)')
        if not r.get('name') or not r.get('attribution'):raise ValueError('Нужны name и attribution')
        if not 0<=int(r.get('maxzoom',19))<=22:raise ValueError('maxzoom должен быть 0–22')
        if r['type']=='WMS':
            if not r.get('layers'):raise ValueError('WMS: заполните layers из GetCapabilities')
            if r.get('crs','EPSG:3857') not in ('EPSG:3857','EPSG:4326','CRS:84'):raise ValueError('WMS: поддержаны EPSG:3857/4326/CRS:84')
        if r['type']=='WMTS' and (not r.get('layer') or not r.get('matrix_set')):raise ValueError('WMTS: нужны layer и matrix_set; сетка WebMercator XYZ')
    return rows


def service_url(config,z,x,y):
    from urllib.parse import urlsplit,urlunsplit,parse_qsl,urlencode
    kind=config['type'];url=config['url']
    if kind=='XYZ':
        return url.replace('{z}',str(z)).replace('{x}',str(x)).replace('{-y}',str(2**z-1-y)).replace('{y}',str(y))
    if kind=='WMTS':
        labels=config.get('matrix_labels')
        matrix=labels[z] if labels else config.get('matrix_prefix','')+str(z)
        args=dict(SERVICE='WMTS',VERSION='1.0.0',REQUEST='GetTile',LAYER=config['layer'],STYLE=config.get('style','default'),FORMAT=config.get('format','image/png'),TILEMATRIXSET=config['matrix_set'],TILEMATRIX=matrix,TILEROW=y,TILECOL=x)
    else:
        half=20037508.342789244;size=half*2/(2**z)
        bounds=[-half+x*size,half-(y+1)*size,-half+(x+1)*size,half-y*size]
        crs=config.get('crs','EPSG:3857');version=config.get('version','1.1.1')
        if crs!='EPSG:3857':
            west=x/(2**z)*360-180;east=(x+1)/(2**z)*360-180
            north=math.degrees(math.atan(math.sinh(math.pi*(1-2*y/(2**z)))))
            south=math.degrees(math.atan(math.sinh(math.pi*(1-2*(y+1)/(2**z)))))
            bounds=[south,west,north,east] if version=='1.3.0' and crs=='EPSG:4326' else [west,south,east,north]
        args=dict(SERVICE='WMS',VERSION=version,REQUEST='GetMap',LAYERS=config['layers'],STYLES=config.get('styles',''),FORMAT=config.get('format','image/png'),WIDTH=256,HEIGHT=256,TRANSPARENT='TRUE',BBOX=','.join(map(str,bounds)))
        args['CRS' if version=='1.3.0' else 'SRS']=crs
    args.update(config.get('parameters',{}))
    parts=urlsplit(url);query=dict(parse_qsl(parts.query));query.update(args)
    return urlunsplit((parts.scheme,parts.netloc,parts.path,urlencode(query),parts.fragment))


class ServiceStore:
    def __init__(self,path,cache):
        self.path=Path(path);self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True)
        self.sem=threading.Semaphore(4);self.vendor=VendorClient()
    def configs(self):return validate_services(json.loads(self.path.read_text()) if self.path.exists() else DEFAULT_SERVICES)
    def public(self):return [dict(id=r['id'],name=r['name'],attribution=r['attribution'],maxzoom=r.get('maxzoom',19),cache_allowed=r.get('cache_allowed',False) and r['type'] not in ('GOOGLE','BING','YANDEX')) for r in self.configs() if r.get('enabled',True)]
    def get(self,id,mode,z,x,y,referer):
        config=next((r for r in self.configs() if r['id']==id and r.get('enabled',True)),None)
        if config is None:raise ValueError('Сервис не настроен')
        if not(0<=z<=int(config.get('maxzoom',19)) and 0<=x<2**z and 0<=y<2**z):raise ValueError('Адрес тайла вне диапазона')
        licensed=config['type'] in ('GOOGLE','BING','YANDEX')
        if licensed and mode!='online':raise ValueError('Для данного API локальное архивирование отключено')
        url=self.vendor.tile_url(config,z,x,y) if licensed else service_url(config,z,x,y);key=hashlib.sha256((json.dumps(config,sort_keys=True)+url).encode()).hexdigest()
        tile=self.cache/(key+'.tile');allowed=config.get('cache_allowed',False) and not licensed
        meta=tile.with_suffix('.json')
        info=json.loads(meta.read_text()) if meta.exists() else {}
        if allowed and tile.exists() and (mode=='cache' or info.get('expires',0)>time.time()):
            data=tile.read_bytes();return data,TileStore.mime(data)
        if mode=='cache':raise FileNotFoundError('Тайл сервиса отсутствует в кэше или кэширование не разрешено настройкой')
        headers={'User-Agent':'City3D-LocalGlobe/1.0'}
        if referer:headers['Referer']=referer
        # Token supplied through an environment variable; not sent to the browser or stored in the registry.
        if config.get('bearer_env'):
            import os
            token=os.environ.get(config['bearer_env'])
            if not token:raise ValueError('Не задана переменная с токеном сервиса')
            headers['Authorization']='Bearer '+token
        with self.sem,urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=15) as response:
            data=response.read(8*1024*1024);mime=TileStore.mime(data)
            if allowed and 'no-store' not in response.headers.get('Cache-Control',''):
                temp=tile.with_suffix('.'+uuid.uuid4().hex+'.tmp');temp.write_bytes(data);temp.replace(tile)
                expires=time.time()+604800
                age=re.search(r'max-age=(\d+)',response.headers.get('Cache-Control',''))
                if age:expires=time.time()+int(age.group(1))
                elif response.headers.get('Expires'):
                    try:expires=parsedate_to_datetime(response.headers['Expires']).timestamp()
                    except Exception:pass
                from city3d_fixes.project import atomic_json
                atomic_json(meta,dict(expires=expires))
        return data,mime

class VendorClient:
    def __init__(self):
        self.lock=threading.Lock();self.sessions={};self.credits={}
    def json_request(self,url,body=None):
        headers={'User-Agent':'City3D-LocalGlobe/1.0'}
        data=None
        if body is not None:headers['Content-Type']='application/json';data=json.dumps(body).encode()
        with urllib.request.urlopen(urllib.request.Request(url,data=data,headers=headers),timeout=15) as r:return json.loads(r.read(4*1024*1024))
    def tile_url(self,c,z,x,y):
        import os
        from urllib.parse import urlencode
        key=os.environ.get(c.get('key_env',''),'')
        if not key:raise ValueError('Не задан ключ: '+c.get('key_env','key_env'))
        name=c['type'];identity=hashlib.sha256((json.dumps(c,sort_keys=True)+key).encode()).hexdigest()
        if name=='YANDEX':
            return 'https://tiles.api-maps.yandex.ru/v1/tiles/?'+urlencode(dict(apikey=key,lang='ru_RU',l='map',projection='web_mercator',maptype=c.get('maptype','map'),z=z,x=x,y=y))
        with self.lock:
            session=self.sessions.get(identity)
            if not session or session['expires']<time.time()+60:
                if name=='GOOGLE':
                    raw=self.json_request('https://tile.googleapis.com/v1/createSession?'+urlencode({'key':key}),dict(mapType=c.get('map_type','satellite'),language='ru-RU',region='RU'))
                    session=dict(raw=raw,expires=float(raw['expiry']))
                else:
                    raw=self.json_request('https://dev.virtualearth.net/REST/v1/Imagery/Metadata/'+c.get('imagery','Aerial')+'?'+urlencode(dict(key=key,include='ImageryProviders',uriScheme='https')))
                    if raw.get('statusCode')!=200:raise ValueError('Bing: доступ к Imagery Metadata не подтверждён')
                    session=dict(raw=raw,expires=time.time()+3600)
                self.sessions[identity]=session
        raw=session['raw']
        if name=='GOOGLE':
            north=math.degrees(math.atan(math.sinh(math.pi*(1-2*y/(2**z)))))
            south=math.degrees(math.atan(math.sinh(math.pi*(1-2*(y+1)/(2**z)))))
            viewport=self.json_request('https://tile.googleapis.com/tile/v1/viewport?'+urlencode(dict(key=key,session=raw['session'],zoom=z,west=x/(2**z)*360-180,east=(x+1)/(2**z)*360-180,north=north,south=south)))
            with self.lock:self.credits.setdefault(c['id'],set()).add(viewport.get('copyright','Google Maps'))
            return f'https://tile.googleapis.com/v1/2dtiles/{z}/{x}/{y}?'+urlencode(dict(key=key,session=raw['session']))
        resource=raw['resourceSets'][0]['resources'][0]
        if not resource.get('zoomMin',0)<=z<=resource.get('zoomMax',22):raise ValueError('Масштаб вне доступного диапазона Bing')
        credits=[raw.get('copyright','Microsoft Bing')]+[p.get('attribution','') for p in resource.get('imageryProviders',[]) or []]
        with self.lock:self.credits.setdefault(c['id'],set()).update(credits)
        quad=''.join(str((1 if x&(1<<i) else 0)+(2 if y&(1<<i) else 0)) for i in reversed(range(z)))
        return resource['imageUrl'].replace('{quadkey}',quad).replace('{subdomain}',(resource.get('imageUrlSubdomains') or [''])[0]).replace('{culture}','ru-RU').replace('{BingMapsKey}',key).replace('http://','https://',1)
    def attribution(self,id):
        with self.lock:return ' | '.join(sorted(self.credits.get(id,set())))

DEFAULT_SERVICES.extend([
 {'id':'google','name':'Google Maps — спутник','type':'GOOGLE','enabled':False,'url':'https://tile.googleapis.com','key_env':'CITY3D_GOOGLE_KEY','map_type':'satellite','maxzoom':22,'attribution':'Google Maps','cache_allowed':False,'note':'Официальный Map Tiles API. Требуется ключ с подключённым API; запросы могут тарифицироваться. Сессия создаётся автоматически.'},
 {'id':'bing','name':'Bing Maps — Aerial','type':'BING','enabled':False,'url':'https://dev.virtualearth.net','key_env':'CITY3D_BING_KEY','imagery':'Aerial','maxzoom':19,'attribution':'Microsoft Bing Maps','cache_allowed':False,'note':'Нужен действующий доступ Bing Maps Enterprise. Basic прекращён; проверьте доступ своей учётной записи.'},
 {'id':'yandex','name':'Яндекс Карты','type':'YANDEX','enabled':False,'url':'https://tiles.api-maps.yandex.ru','key_env':'CITY3D_YANDEX_KEY','maptype':'map','maxzoom':20,'attribution':'Яндекс Карты','cache_allowed':False,'note':'Tiles API, projection=web_mercator. Требуется ключ Tiles API и соблюдение условий атрибуции.'},
 {'id':'esri_imagery','name':'Esri World Imagery','type':'XYZ','enabled':True,'url':'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}','maxzoom':19,'attribution':'Esri, Maxar, Earthstar Geographics, and the GIS User Community','cache_allowed':False},
 {'id':'roskadastr','name':'Роскадастр — предоставленный геосервис','type':'WMS','enabled':False,'url':'https://kadastr.ru','layers':'','crs':'EPSG:3857','maxzoom':19,'attribution':'ППК Роскадастр','cache_allowed':False,'note':'Укажите URL геосервиса и layers из предоставленного GetCapabilities. Адрес сайта не является WMS.'},
 {'id':'nspd','name':'НСПД России — предоставленный геосервис','type':'WMS','enabled':False,'url':'https://nspd.gov.ru','layers':'','crs':'EPSG:3857','maxzoom':19,'attribution':'НСПД России / Росреестр','cache_allowed':False,'note':'Укажите доступный оператору WMS/WMTS/XYZ endpoint и параметры. Автоматический вход и обход ограничений не выполняются.'}
])
