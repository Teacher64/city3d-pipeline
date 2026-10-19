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
        def closeEvent(self,event):
            super().closeEvent(event)
            if event.isAccepted():
                for i in range(self.tabs.count()):
                    globe=getattr(self.tabs.widget(i),'_globe',None)
                    if globe is not None:globe.close()
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
            tr=group.transformers[0];pts,n=sparse_points(r['path'],cancel,min(48000,max(1,300000//max(1,len(records)))))
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
            item.update(points=xy.tolist(),total=n,center=center,extent=extent.tolist(),transform=f'{_city3d_crs_label(c)} → WGS84; операция {tr.accuracy} м',source_crs_wkt=r['crs_wkt'])
        except Exception as exc:item['error']=str(exc)
        files.append(item)
    return dict(files=files)


# CITY3D_GLOBE_FD_FIX_20261019
import errno
import sqlite3
import socket
import sys
import contextlib


class _City3DClosingSQLite(sqlite3.Connection):
    """Retain transaction semantics and release the DB handle on every exit."""
    def __exit__(self, *exc):
        try:
            return super().__exit__(*exc)
        finally:
            self.close()


@contextlib.contextmanager
def _city3d_response(request, **kwargs):
    response = None
    try:
        try:
            response = urllib.request.urlopen(request, **kwargs)
        except urllib.error.HTTPError as exc:
            exc.close()
            raise
        yield response
    finally:
        if response is not None:
            response.close()


def _city3d_fd_metrics():
    try:
        import resource
        soft,hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (ImportError,OSError):
        soft=hard=None
    try:
        count=len(list(Path('/proc/self/fd').iterdir()))
    except OSError:
        count=None
    return dict(open_fds=count,soft_limit=soft,hard_limit=hard)


def _city3d_empty_tile():
    import base64
    return base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABpfZFQAAAAABJRU5ErkJggg=='),'image/png'


class _City3DBoundedServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    request_queue_size = 16
    max_workers = 12

    def __init__(self,*args,**kwargs):
        self._slots=threading.BoundedSemaphore(self.max_workers)
        self._state_lock=threading.Lock()
        self._clients=set()
        self._stopping=threading.Event()
        self._active=0;self._peak=0;self._rejected=0;self._errors=0
        self._last_error=None;self._last_warning=0.
        super().__init__(*args,**kwargs)

    def process_request(self,request,address):
        if self._stopping.is_set() or not self._slots.acquire(blocking=False):
            with self._state_lock:self._rejected+=1
            try:
                request.settimeout(.1)
                request.sendall(b'HTTP/1.0 503 Service Unavailable\r\nConnection: close\r\nRetry-After: 1\r\nContent-Length: 0\r\n\r\n')
            except OSError:pass
            finally:self.shutdown_request(request)
            return
        with self._state_lock:
            self._clients.add(request);self._active+=1;self._peak=max(self._peak,self._active)
        try:
            request.settimeout(8.)
            super().process_request(request,address)
        except BaseException:
            self.shutdown_request(request);self._release(request)
            raise

    def _release(self,request):
        with self._state_lock:
            self._clients.discard(request);self._active-=1
        self._slots.release()

    def process_request_thread(self,request,address):
        try:
            super().process_request_thread(request,address)
        finally:
            self._release(request)

    def record_error(self,exc):
        kind=type(exc).__name__
        number=getattr(exc,'errno',None)
        with self._state_lock:
            self._errors+=1;self._last_error=dict(type=kind,errno=number)
            now=time.monotonic();warn=now-self._last_warning>10
            if warn:self._last_warning=now
        if warn:
            # The existing stderr descriptor is used; no log file is opened in an error path.
            print(f'City3D globe request: {kind}, errno={number}; {_city3d_fd_metrics()}',file=sys.stderr)

    def handle_error(self,request,client_address):
        exc=sys.exc_info()[1]
        if not isinstance(exc,(BrokenPipeError,ConnectionResetError,TimeoutError)):
            self.record_error(exc)

    def metrics(self):
        with self._state_lock:
            data=dict(active_requests=self._active,peak_requests=self._peak,max_requests=self.max_workers,
                      rejected_requests=self._rejected,request_errors=self._errors,last_error=self._last_error)
        data.update(_city3d_fd_metrics());return data

    def stop_clients(self):
        self._stopping.set()
        with self._state_lock:clients=list(self._clients)
        for client in clients:
            try:client.shutdown(socket.SHUT_RDWR)
            except OSError:pass
            client.close()


class GlobeServer:
    def __init__(self,assets,cache=None,mbtiles=None,services=None):
        self.services=ServiceStore(services or Path(assets).parent/'city3d_services.json',Path(cache or Path(assets).parent/'city3d_tile_cache')/'services')
        self.tiles=TileStore(cache or Path(assets).parent/'city3d_tile_cache',mbtiles)
        self.scene=dict(revision=0,files=[]);self.lock=threading.Lock();token=secrets.token_urlsafe(24)
        self._close_lock=threading.Lock();self._closed=False
        owner=self;root=Path(assets).resolve()
        class Handler(BaseHTTPRequestHandler):
            protocol_version='HTTP/1.0'
            def log_message(self,*args):pass
            def do_GET(self):
                self.close_connection=True
                try:self._get()
                except (BrokenPipeError,ConnectionResetError,TimeoutError):pass
                except Exception as exc:
                    self.server.record_error(exc)
                    try:self.send_error(503 if getattr(exc,'errno',None) in (errno.EMFILE,errno.ENFILE) else 500,'Local globe request failed')
                    except OSError:pass
            def _get(self):
                from urllib.parse import urlsplit,unquote,parse_qs
                parsed=urlsplit(self.path);query=parse_qs(parsed.query)
                path=unquote(parsed.path);prefix='/'+token+'/'
                if not path.startswith(prefix):self.send_error(404);return
                rel=path[len(prefix):] or 'globe.html'
                if rel=='health':
                    data=json.dumps(owner.server.metrics(),ensure_ascii=False).encode();mime='application/json'
                elif rel.startswith('credits/'):
                    data=json.dumps(dict(text=owner.services.vendor.attribution(rel.split('/')[-1])),ensure_ascii=False).encode();mime='application/json'
                elif rel=='cache-status':
                    data=json.dumps(cache_status(owner.tiles.root,owner.services,owner.tiles.mbtiles),ensure_ascii=False).encode();mime='application/json'
                elif rel=='providers':
                    data=json.dumps(owner.services.public(),ensure_ascii=False).encode();mime='application/json'
                elif rel.startswith('services/'):
                    try:
                        _,id,mode,z,x,y=rel.split('/')
                        data,mime=owner.services.get(id,mode,int(z),int(x),int(y.removesuffix('.png')),self.headers.get('Referer',''))
                    except FileNotFoundError:
                        data,mime=_city3d_empty_tile()
                    except Exception as exc:
                        self.server.record_error(exc);self.send_error(503,'Service tile unavailable');return
                elif rel.startswith('tiles/'):
                    try:
                        _,mode,z,x,y=rel.split('/')
                        data,mime=owner.tiles.get(mode,int(z),int(x),int(y.removesuffix('.png')),self.headers.get('Referer',''))
                    except FileNotFoundError:
                        data,mime=_city3d_empty_tile()
                    except Exception as exc:
                        self.server.record_error(exc);self.send_error(503,'Local tile unavailable');return
                elif rel=='scene':
                    with owner.lock:
                        scene=dict(revision=owner.scene['revision'],unchanged=True) if query.get('since')==[str(owner.scene['revision'])] else owner.scene
                        data=json.dumps(scene,ensure_ascii=False,allow_nan=False).encode()
                    mime='application/json'
                else:
                    p=(root/rel).resolve()
                    if not p.is_relative_to(root) or not p.is_file():self.send_error(404);return
                    data=p.read_bytes();mime=mimetypes.guess_type(p.name)[0] or 'application/octet-stream'
                self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)))
                self.send_header('Cache-Control','no-store' if rel in ('scene','providers','health','cache-status') or rel.startswith(('credits/','tiles/','services/')) else 'public, max-age=86400')
                self.send_header('Connection','close');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)
        self.server=_City3DBoundedServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,kwargs={'poll_interval':.1},daemon=True,name='City3D-globe-server');self.thread.start()
        self.url=f'http://127.0.0.1:{self.server.server_port}/{token}/'
    def update(self,data):
        with self.lock:self.scene=dict(data,revision=self.scene['revision']+1)
    def close(self):
        with self._close_lock:
            if self._closed:return
            self._closed=True
        self.server._stopping.set();self.server.shutdown();self.server.stop_clients();self.server.server_close()
        if self.thread is not threading.current_thread():self.thread.join(timeout=1.)


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
        with sqlite3.connect(self.db,factory=_City3DClosingSQLite) as c:
            c.execute('PRAGMA journal_mode=WAL')
            c.execute('CREATE TABLE IF NOT EXISTS tiles(z INTEGER,x INTEGER,y INTEGER,data BLOB,mime TEXT,expires REAL,etag TEXT,modified TEXT,PRIMARY KEY(z,x,y))')
    def set_mbtiles(self,path):
        p=Path(path).resolve()
        with sqlite3.connect(p.as_uri()+'?mode=ro',uri=True,factory=_City3DClosingSQLite) as c:
            metadata=dict(c.execute('SELECT name,value FROM metadata'))
            if metadata.get('format','').lower() not in ('png','jpg','jpeg','webp'):raise ValueError('Нужна растровая MBTiles (PNG/JPEG/WebP), не PBF/MVT.')
            c.execute('SELECT zoom_level,tile_column,tile_row,tile_data FROM tiles LIMIT 1').fetchone()
        self.mbtiles=p;return metadata
    def get(self,mode,z,x,y,referer=''):
        if not(0<=z<=19 and 0<=x<2**z and 0<=y<2**z):raise ValueError('Недопустимый адрес тайла')
        if mode=='mbtiles':
            if not self.mbtiles:raise FileNotFoundError('База MBTiles не выбрана')
            with sqlite3.connect(self.mbtiles.resolve().as_uri()+'?mode=ro',uri=True,factory=_City3DClosingSQLite) as c:
                row=c.execute('SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?',(z,x,2**z-1-y)).fetchone()
            if not row:raise FileNotFoundError('Тайл отсутствует в MBTiles')
            data=bytes(row[0]);return data,self.mime(data)
        with sqlite3.connect(self.db,factory=_City3DClosingSQLite) as c:row=c.execute('SELECT data,mime,expires,etag,modified FROM tiles WHERE z=? AND x=? AND y=?',(z,x,y)).fetchone()
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
                    else:exc.close();raise
                with response:
                    h=response.headers;data=bytes(row[0]) if response.code==304 else response.read(5*1024*1024)
                mime=self.mime(data);expires=time.time()+604800
                age=re.search(r'max-age=(\d+)',h.get('Cache-Control',''))
                if age:expires=time.time()+int(age.group(1))
                elif h.get('Expires'):
                    try:expires=parsedate_to_datetime(h['Expires']).timestamp()
                    except Exception:pass
                if 'no-store' not in h.get('Cache-Control',''):
                    with sqlite3.connect(self.db,factory=_City3DClosingSQLite) as c:c.execute('INSERT OR REPLACE INTO tiles VALUES(?,?,?,?,?,?,?,?)',(z,x,y,data,mime,expires,h.get('ETag',row[3] if row else None),h.get('Last-Modified',row[4] if row else None)))
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
        tile=self.cache/config['id']/(key+'.tile');allowed=config.get('cache_allowed',False) and not licensed
        legacy=self.cache/(key+'.tile');cached=tile if tile.exists() else legacy
        meta=cached.with_suffix('.json')
        info=json.loads(meta.read_text()) if meta.exists() else {}
        if allowed and cached.exists() and (mode=='cache' or info.get('expires',0)>time.time()):
            data=cached.read_bytes();return data,TileStore.mime(data)
        if mode=='cache':raise FileNotFoundError('Тайл сервиса отсутствует в кэше или кэширование не разрешено настройкой')
        headers={'User-Agent':'City3D-LocalGlobe/1.0'}
        if referer:headers['Referer']=referer
        # Token supplied through an environment variable; not sent to the browser or stored in the registry.
        if config.get('bearer_env'):
            import os
            token=os.environ.get(config['bearer_env'])
            if not token:raise ValueError('Не задана переменная с токеном сервиса')
            headers['Authorization']='Bearer '+token
        with self.sem,_city3d_response(urllib.request.Request(url,headers=headers),timeout=15) as response:
            data=response.read(8*1024*1024);mime=TileStore.mime(data)
            if allowed and 'no-store' not in response.headers.get('Cache-Control',''):
                tile.parent.mkdir(parents=True,exist_ok=True)
                temp=tile.with_suffix('.'+uuid.uuid4().hex+'.tmp');temp.write_bytes(data);temp.replace(tile)
                expires=time.time()+604800
                age=re.search(r'max-age=(\d+)',response.headers.get('Cache-Control',''))
                if age:expires=time.time()+int(age.group(1))
                elif response.headers.get('Expires'):
                    try:expires=parsedate_to_datetime(response.headers['Expires']).timestamp()
                    except Exception:pass
                from city3d_fixes.project import atomic_json
                atomic_json(tile.with_suffix('.json'),dict(expires=expires,provider=config['id'],name=config['name'],bytes=len(data),stored_at=time.time()))
        return data,mime

class VendorClient:
    def __init__(self):
        self.lock=threading.Lock();self.sessions={};self.credits={}
    def json_request(self,url,body=None):
        headers={'User-Agent':'City3D-LocalGlobe/1.0'}
        data=None
        if body is not None:headers['Content-Type']='application/json';data=json.dumps(body).encode()
        with _city3d_response(urllib.request.Request(url,data=data,headers=headers),timeout=15) as r:return json.loads(r.read(4*1024*1024))
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

# Spatial overview and point-derived occupied-cell coverage. No training data change.
import gzip
import os
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDoubleSpinBox,QSpinBox,QCheckBox


def _city3d_crs_label(c):
    source=c.source_crs if c.is_bound and c.source_crs else c
    authority=source.to_authority()
    label=':'.join(authority) if authority else source.name
    return (label or 'Пользовательская CRS')[:100]


def sparse_points(path,cancel=lambda:False,limit=16000):
    """LAS: stratified indices across the WHOLE file, rather than scan-line windows.

    LAZ gets a bounded quick overview; the complete spatial pass is sequential.
    """
    import laspy
    p=Path(path)
    with laspy.open(p,read_evlrs=False) as reader:
        n=int(reader.header.point_count);compressed=reader.header.are_points_compressed
    if n<=0:raise ValueError('Пустой LAS')
    if cancel():raise RuntimeError('Просмотр отменён')
    if not compressed:
        k=min(n,int(limit));rng=np.random.default_rng(378193)
        edges=np.linspace(0,n,k+1,dtype=np.int64)
        ids=edges[:-1]+(rng.random(k)*(edges[1:]-edges[:-1])).astype(np.int64)
        with laspy.mmap(str(p)) as mmap:
            points=mmap.points[ids]
            result=np.column_stack((points.x,points.y,points.z))
            del points
        if cancel():raise RuntimeError('Просмотр отменён')
        return result,n
    with laspy.open(p,read_evlrs=False) as reader:
        windows=min(64,max(1,n//512));size=min(n,max(1,int(limit)//windows))
        starts=np.linspace(0,max(0,n-size),windows,dtype=np.int64);blocks=[]
        for start in starts:
            if cancel():raise RuntimeError('Просмотр отменён')
            reader.seek(int(start));points=reader.read_points(size)
            blocks.append(np.column_stack((points.x,points.y,points.z)))
        return np.concatenate(blocks),n


def _coverage_pack(x,y):
    if (np.abs(x)>2**31-2).any() or (np.abs(y)>2**31-2).any():
        raise ValueError('Слишком большой диапазон индексов покрытия: проверьте CRS / увеличьте шаг')
    return ((y.astype(np.uint64)&np.uint64(0xffffffff))<<np.uint64(32))|(x.astype(np.uint64)&np.uint64(0xffffffff))


def _coverage_polygon(keys,origin,step,cancel):
    from shapely.geometry import box
    from shapely import union_all,simplify
    packed=np.fromiter(keys,dtype=np.uint64,count=len(keys))
    rows=(packed>>np.uint64(32)).astype(np.uint32).view(np.int32)
    cols=(packed&np.uint64(0xffffffff)).astype(np.uint32).view(np.int32)
    order=np.lexsort((cols,rows));rows=rows[order];cols=cols[order]
    starts=np.r_[0,np.flatnonzero((np.diff(rows)!=0)|(np.diff(cols)!=1))+1]
    ends=np.r_[starts[1:]-1,len(rows)-1]
    if len(starts)>200000:raise ValueError('Слишком много отдельных ячеек/разрывов; увеличьте шаг покрытия')
    merged=[]
    for first in range(0,len(starts),4096):
        if cancel():raise RuntimeError('Покрытие отменено')
        rectangles=[box(origin[0]+int(cols[s])*step,origin[1]+int(rows[s])*step,
                        origin[0]+(int(cols[e])+1)*step,origin[1]+(int(rows[s])+1)*step)
                    for s,e in zip(starts[first:first+4096],ends[first:first+4096])]
        merged.append(union_all(rectangles))
    geometry=simplify(union_all(merged),0.,preserve_topology=True)
    if geometry.is_empty or not geometry.is_valid or geometry.geom_type not in ('Polygon','MultiPolygon'):
        raise ValueError('Не удалось построить валидное покрытие')
    parts=list(geometry.geoms) if geometry.geom_type=='MultiPolygon' else [geometry]
    vertices=sum(len(p.exterior.coords)+sum(len(r.coords) for r in p.interiors) for p in parts)
    if vertices>250000 or len(parts)>5000:raise ValueError('Покрытие слишком сложное для глобуса; увеличьте шаг')
    return geometry,vertices


def _spatial_sample_prune(sample,limit):
    if len(sample)<=limit:return sample
    keys=np.fromiter(sample,dtype=np.uint64,count=len(sample))
    with np.errstate(over='ignore'):
        rank=keys+np.uint64(0x9e3779b97f4a7c15)
        rank=(rank^(rank>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
        rank=(rank^(rank>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
        rank=rank^(rank>>np.uint64(31))
    chosen=np.argpartition(rank,limit-1)[:limit]
    return {int(keys[i]):sample[int(keys[i])] for i in chosen}


def full_spatial_preview(record,step,limit,cache,cancel=lambda:False,progress=lambda text:None,max_cells=2000000):
    import laspy
    from pyproj import CRS
    from pyproj.transformer import TransformerGroup
    from shapely.geometry import mapping
    from shapely.ops import transform
    path=Path(record['path']);before=path.stat()
    if not record.get('crs_wkt'):raise ValueError('Для покрытия назначьте исходную CRS')
    if (before.st_size,before.st_mtime_ns)!=(record['size'],record['mtime_ns']):raise ValueError('LAS изменён; добавьте его заново')
    definition=record['crs_wkt']
    key=hashlib.sha256(json.dumps(['coverage:1',str(path.resolve()),before.st_size,before.st_mtime_ns,definition,step,limit,max_cells],sort_keys=True).encode()).hexdigest()
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True);destination=cache/(key+'.json.gz')
    if destination.is_file():
        try:
            with gzip.open(destination,'rt',encoding='utf-8') as stream:item=json.load(stream)
            if item.get('coverage_key')==key:return dict(item,coverage_cached=True)
        except (OSError,ValueError):pass # incomplete preview cache may be regenerated from the original
    c=CRS.from_user_input(definition).to_2d()
    group=TransformerGroup(c,'EPSG:4326',always_xy=True,allow_ballpark=False)
    if not group.transformers or not group.best_available:raise ValueError('Нет точной локальной операции CRS → WGS84')
    geo=group.transformers[0]
    with laspy.open(path) as reader:
        h=reader.header;n=int(h.point_count)
        if n==0:raise ValueError('Пустое облако')
        if c.is_projected:
            fx,fy=[a.unit_conversion_factor for a in c.axis_info[:2]]
            def metric(x,y):return np.asarray(x)*fx,np.asarray(y)*fy
            def geographic(x,y):return geo.transform(np.asarray(x)/fx,np.asarray(y)/fy,errcheck=True)
            lo=np.asarray(h.mins[:2])*[fx,fy];hi=np.asarray(h.maxs[:2])*[fx,fy]
        else:
            lng,lat=geo.transform(float((h.mins[0]+h.maxs[0])/2),float((h.mins[1]+h.maxs[1])/2),errcheck=True)
            epsg=(32661 if lat>=84 else 32761) if lat>=84 or lat<-80 else ((32600 if lat>=0 else 32700)+min(60,max(1,int((lng+180)//6)+1)))
            to_metric=TransformerGroup(c,CRS(epsg),always_xy=True,allow_ballpark=False)
            to_geo=TransformerGroup(CRS(epsg),CRS(4326),always_xy=True,allow_ballpark=False)
            if not to_metric.best_available or not to_metric.transformers or not to_geo.best_available or not to_geo.transformers:raise ValueError('Нет локальной метрической операции для покрытия')
            def metric(x,y):return to_metric.transformers[0].transform(x,y,errcheck=True)
            def geographic(x,y):return to_geo.transformers[0].transform(x,y,errcheck=True)
            a,b=metric([h.mins[0],h.maxs[0],h.maxs[0],h.mins[0]],[h.mins[1],h.mins[1],h.maxs[1],h.maxs[1]])
            lo=np.array([np.min(a),np.min(b)]);hi=np.array([np.max(a),np.max(b)])
        if not np.isfinite(np.r_[lo,hi]).all():raise ValueError('Некорректный охват LAS')
        origin=np.floor(lo/step)*step
        span=np.maximum(hi-lo,step)
        sample_step=max(step,math.sqrt(float(span[0]*span[1])/max(1,limit)))
        cells=set();samples={};count=0
        for batch in reader.chunk_iterator(250000):
            if cancel():raise RuntimeError('Построение покрытия отменено')
            x,y=metric(np.asarray(batch.x),np.asarray(batch.y));x,y=np.asarray(x),np.asarray(y)
            if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Некорректные координаты в LAS')
            ix=np.floor((x-origin[0])/step).astype(np.int64);iy=np.floor((y-origin[1])/step).astype(np.int64)
            cells.update(np.unique(_coverage_pack(ix,iy)).tolist())
            if len(cells)>max_cells:raise ValueError(f'Более {max_cells:,} ячеек покрытия при шаге {step:g} м; увеличьте шаг')
            sx=np.floor((x-origin[0])/sample_step).astype(np.int64);sy=np.floor((y-origin[1])/sample_step).astype(np.int64)
            unique,indices=np.unique(_coverage_pack(sx,sy),return_index=True)
            z=np.asarray(batch.z)
            for k,i in zip(unique.tolist(),indices.tolist()):
                if k not in samples:samples[k]=(float(x[i]),float(y[i]),float(z[i]))
            if len(samples)>2*limit:samples=_spatial_sample_prune(samples,limit)
            count+=len(batch);progress(f'{path.name}: покрытие по всем точкам {count:,}/{n:,}; шаг {step:g} м')
        if count!=n:raise ValueError('Число точек не соответствует заголовку LAS')
    if cancel():raise RuntimeError('Покрытие отменено')
    after=path.stat()
    if (after.st_size,after.st_mtime_ns)!=(before.st_size,before.st_mtime_ns):raise ValueError('Источник изменился при чтении')
    progress(path.name+': построение контура занятых ячеек…')
    polygon,vertices=_coverage_polygon(cells,origin,step,cancel)
    footprint=transform(geographic,polygon)
    b=footprint.bounds
    if not footprint.is_valid or not np.isfinite(b).all() or not(-180<=b[0]<=b[2]<=180 and -90<=b[1]<=b[3]<=90) or b[2]-b[0]>180:
        raise ValueError('Покрытие не представимо одним районом WGS84; проверьте CRS / пересечение 180°')
    samples=_spatial_sample_prune(samples,limit);pts=np.asarray(list(samples.values()))
    lon,lat=geographic(pts[:,0],pts[:,1]);points=np.column_stack((lon,lat))
    if not np.isfinite(points).all():raise ValueError('Некорректные точки WGS84')
    item=dict(name=path.name,total=n,points=points.tolist(),center=[float(footprint.centroid.x),float(footprint.centroid.y)],
        extent=[[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]],transform=_city3d_crs_label(c)+' → WGS84',source_crs_wkt=definition,
        footprint=mapping(footprint),coverage_complete=True,coverage_cell_m=step,coverage_key=key,
        coverage_points=count,coverage_cells=len(cells),coverage_vertices=vertices,coverage_cached=False,
        sampling_method='spatial_grid_all_points',sample_grid_m=sample_step)
    temporary=destination.with_suffix('.'+uuid.uuid4().hex+'.tmp')
    try:
        with gzip.open(temporary,'wt',encoding='utf-8') as stream:json.dump(item,stream,ensure_ascii=False,allow_nan=False)
        temporary.replace(destination)
    finally:temporary.unlink(missing_ok=True)
    return item


def full_coverage_scene(records,step,limit,cache,cancel,progress):
    items=[]
    for r in records:
        if cancel():raise RuntimeError('Построение покрытия отменено')
        try:item=full_spatial_preview(r,step,max(1,limit//max(1,len(records))),cache,cancel,progress)
        except Exception as exc:
            if cancel():raise
            item=geographic_scene([r],cancel)['files'][0]
            item['coverage_error']=str(exc)
        items.append(item)
    return dict(files=items)


_coverage_base_install=install

def install(ns):
    _coverage_base_install(ns)
    base=ns['LoadDataTab'];workspace=Path(ns['WORKSPACE'])
    class CoverageLoad(base):
        coverage_progress=pyqtSignal(str)
        def __init__(self,*args,**kwargs):
            self._coverage_signature=None
            super().__init__(*args,**kwargs)
            self.coverage_progress.connect(self.status.setText,Qt.QueuedConnection)
            grid=self.preview_button.parentWidget().layout();row=grid.rowCount()
            self.coverage_button=QPushButton('Покрытие LAS по всем точкам');self.coverage_button.setMaximumHeight(30);self.coverage_button.clicked.connect(lambda:self.full_coverage(True))
            grid.addWidget(self.coverage_button,row,0,1,2)
            panel=QWidget();form=QGridLayout(panel);form.setContentsMargins(0,0,0,0)
            self.coverage_step=QDoubleSpinBox();self.coverage_step.setRange(.1,1000.);self.coverage_step.setDecimals(1);self.coverage_step.setValue(5.);self.coverage_step.setSuffix(' м')
            self.coverage_step.setToolTip('Шаг сетки покрытия по всем точкам. Меньше — подробнее; большие площади требуют большего шага. Исходные точки не сдвигаются.')
            self.point_budget=QSpinBox();self.point_budget.setRange(10000,500000);self.point_budget.setSingleStep(10000);self.point_budget.setValue(100000)
            form.addWidget(QLabel('Покрытие:'),0,0);form.addWidget(self.coverage_step,0,1);form.addWidget(QLabel('Точек на глобусе:'),1,0);form.addWidget(self.point_budget,1,1)
            grid.addWidget(panel,row+1,0,1,2)
            self.auto_coverage=QCheckBox('Считать покрытие при открытии глобуса');self.auto_coverage.setChecked(True);grid.addWidget(self.auto_coverage,row+2,0,1,2)
            self._coverage_timer=QTimer(self);self._coverage_timer.setSingleShot(True);self._coverage_timer.timeout.connect(lambda:self.full_coverage(False))
        def _coverage_inputs(self):return self.chosen() or [self.records[p] for p in self.selected_paths()]
        def _coverage_key(self,records):
            return json.dumps([[(r['path'],r.get('size'),r.get('mtime_ns'),r.get('crs_wkt')) for r in records],self.coverage_step.value(),self.point_budget.value()],sort_keys=True)
        def preview(self):
            if self.busy():return
            records=self._coverage_inputs()
            if not records:self.status.setText('Добавьте LAS/LAZ или выделите строки.');return
            snapshot=[dict(r) for r in records]
            self._coverage_signature=None
            def done(data):
                self._scene=data
                if self._globe:self._globe.update(data)
                self._display_coverage_scene(data)
                self.status.setText('Быстрый пространственный обзор. Полное покрытие: кнопка «Покрытие LAS по всем точкам».')
                if getattr(self,'_open_when_ready',False):self._open_when_ready=False;self.open_globe()
                elif self._globe and self.auto_coverage.isChecked():self._coverage_timer.start(80)
            self.start_job(lambda job:geographic_scene(snapshot,job.isInterruptionRequested),done)
        def _display_coverage_scene(self,data):
            points=[p for f in data['files'] if not f.get('error') for p in f['points']]
            self.plan.set_points(np.column_stack((np.asarray(points),np.zeros(len(points)))) if points else np.empty((0,3)))
            self.views.setTabText(0,'Обзор WGS84 (долгота/широта)');self.views.setCurrentIndex(0);self.views.setTabEnabled(1,False);self._preview_points=None
            self.status.setToolTip('\n'.join(f['name']+': '+f.get('error',f.get('coverage_error','')) for f in data['files'] if f.get('error') or f.get('coverage_error')))
        def open_globe(self):
            super().open_globe()
            if self._globe and self._scene is not None and self.auto_coverage.isChecked():self._coverage_timer.start(80)
        def full_coverage(self,force=False):
            if self.busy():self._coverage_timer.start(150);return
            records=self._coverage_inputs()
            if not records:return
            signature=self._coverage_key(records)
            if not force and signature==self._coverage_signature:return
            snapshot=[dict(r) for r in records];step=self.coverage_step.value();limit=self.point_budget.value()
            self._preview_timer.stop();self._preview_pending=False
            def task(job):
                return full_coverage_scene(snapshot,step,limit,workspace/'cache/las_coverage',job.isInterruptionRequested,
                    lambda text:(self.coverage_progress.emit(text),job.progress.emit(50)))
            def done(data):
                self._preview_timer.stop();self._preview_pending=False
                if signature!=self._coverage_key(self._coverage_inputs()):
                    self._preview_pending=True;self._preview_timer.start(80);return
                self._coverage_signature=signature;self._scene=data
                if self._globe:self._globe.update(data)
                self._display_coverage_scene(data)
                count=sum(bool(f.get('coverage_complete')) for f in data['files'])
                self.status.setText(f'Покрытие по всем точкам: {count}/{len(data["files"])} LAS; шаг {step:g} м. Подробности ошибок — в подсказке.')
            self.status.setText('Построение покрытия по всем точкам; первый проход по крупному LAS/LAZ может занять время…')
            self.start_job(task,done)
        def cancel_job(self):
            if hasattr(self,'_coverage_timer'):self._coverage_timer.stop()
            super().cancel_job()
        def remove_inputs(self):
            super().remove_inputs();self._coverage_signature=None
            if hasattr(self,'_coverage_timer'):self._coverage_timer.stop()
    ns['LoadDataTab']=CoverageLoad

# Human-readable cache status. No network requests, credentials or cache deletion.
def cache_status(root,services=None,mbtiles=None):
    root=Path(root);rows=[]
    database=root/'osm_cache.sqlite3';count=0;size=0;latest=None
    if database.is_file():
        with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True,factory=_City3DClosingSQLite) as conn:
            count=int(conn.execute('SELECT COUNT(*) FROM tiles').fetchone()[0])
        for path in (database,Path(str(database)+'-wal'),Path(str(database)+'-shm')):
            if path.is_file():size+=path.stat().st_size
        latest=database.stat().st_mtime if count else None
    rows.append(dict(id='osm',name='OSM',storage='SQLite',path=str(database.resolve()),tiles=count,bytes=size,last_write=latest,
                     policy='Сохранение включено: режим интернет + кэш'))
    configs=services.configs() if services is not None else DEFAULT_SERVICES
    for config in configs:
        if not config.get('enabled',True):continue
        folder=root/'services'/config['id'];tile_files=list(folder.glob('*.tile')) if folder.exists() else []
        allowed=config.get('cache_allowed',False) and config['type'] not in ('GOOGLE','BING','YANDEX')
        rows.append(dict(id=config['id'],name=config['name'],storage='Файлы тайлов',path=str(folder.resolve()),
            tiles=len(tile_files),bytes=sum(p.stat().st_size for p in tile_files),
            last_write=max((p.stat().st_mtime for p in tile_files),default=None),
            policy='Сохранение включено' if allowed else 'Сохранение выключено в адаптере/реестре'))
    legacy=list((root/'services').glob('*.tile'))
    if legacy:rows.append(dict(id='legacy_services',name='Ранее сохранённые геосервисы',storage='Общая папка предыдущей версии',
        path=str((root/'services').resolve()),tiles=len(legacy),bytes=sum(p.stat().st_size for p in legacy),
        last_write=max(p.stat().st_mtime for p in legacy),policy='Сохранены; читаются по старым ключам'))
    if mbtiles and Path(mbtiles).is_file():
        p=Path(mbtiles);rows.append(dict(id='mbtiles',name='Подключённая MBTiles',storage='Готовая база',path=str(p.resolve()),
                                      tiles=None,bytes=p.stat().st_size,last_write=p.stat().st_mtime,policy='Только чтение; исходная база не изменяется'))
    return dict(root=str(root.resolve()),providers=rows)


_cache_base_install=install

def install(ns):
    _cache_base_install(ns)
    base=ns['LoadDataTab'];workspace=Path(ns['WORKSPACE'])
    class CacheLoad(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            grid=self.preview_button.parentWidget().layout();row=grid.rowCount()
            button=QPushButton('Кэш подложек…');button.setMaximumHeight(30);button.clicked.connect(self.show_cache_status)
            grid.addWidget(button,row,0,1,2)
        def show_cache_status(self):
            from PyQt5.QtWidgets import QDialog,QTableWidget,QTableWidgetItem,QDialogButtonBox,QHeaderView
            from PyQt5.QtGui import QDesktopServices
            from PyQt5.QtCore import QUrl
            from datetime import datetime
            root=workspace/'cache/osm'
            try:
                store=self._globe.services if self._globe else ServiceStore(self._services_config,root/'services')
                info=cache_status(root,store,self._mbtiles)
                dialog=QDialog(self);dialog.setWindowTitle('Постоянный кэш картографических подложек');dialog.resize(1050,440)
                layout=QVBoxLayout(dialog);label=QLabel('OSM хранится в SQLite; геосервисы — в папках поставщиков. Данные на диск записываются после успешного ответа сервиса.');label.setWordWrap(True);layout.addWidget(label)
                table=QTableWidget(len(info['providers']),6);table.setHorizontalHeaderLabels(['Подложка','Тайлов','Размер, МиБ','Последняя запись','Сохранение','Путь'])
                table.setEditTriggers(QTableWidget.NoEditTriggers)
                for i,r in enumerate(info['providers']):
                    values=[r['name'],str(r['tiles']) if r['tiles'] is not None else '—',f"{r['bytes']/1048576:.2f}",
                            datetime.fromtimestamp(r['last_write']).strftime('%d.%m.%Y %H:%M:%S') if r['last_write'] else 'Нет записей',r['policy'],r['path']]
                    for col,text in enumerate(values):item=QTableWidgetItem(text);item.setToolTip(text);table.setItem(i,col,item)
                table.horizontalHeader().setSectionResizeMode(5,QHeaderView.Stretch);layout.addWidget(table)
                open_folder=QPushButton('Открыть папку кэша');open_folder.clicked.connect(lambda:QDesktopServices.openUrl(QUrl.fromLocalFile(str(root.resolve()))));layout.addWidget(open_folder)
                buttons=QDialogButtonBox(QDialogButtonBox.Close);buttons.rejected.connect(dialog.reject);layout.addWidget(buttons);dialog.exec_()
            except Exception as exc:self.show_error('Не удалось прочитать состояние кэша: '+str(exc))
    ns['LoadDataTab']=CacheLoad
