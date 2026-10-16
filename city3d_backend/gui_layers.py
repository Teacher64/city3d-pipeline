"""Qt5 offline layer view; rendering stays in the GUI thread."""
from pathlib import Path
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt,QThread,pyqtSignal,QRectF
from PyQt5.QtGui import QTransform
from PyQt5.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QPushButton,QFileDialog,QListWidget,QListWidgetItem,QSlider,QLabel,QAbstractItemView,QComboBox
from pyproj import Transformer
from .layers import LayerManager,load_vector,preview_cloud,raster_preview

class Loader(QThread):
    result=pyqtSignal(object)
    failure=pyqtSignal(str)
    def __init__(self,path,kind):super().__init__();self.path=path;self.kind=kind
    def run(self):
        try:
            if self.kind=='cloud':data=preview_cloud(self.path,'EPSG:3857',cancel=self.isInterruptionRequested)
            elif self.kind=='raster':
                import rasterio
                array,affine=raster_preview(self.path,'EPSG:3857')
                with rasterio.open(self.path) as r:crs=str(r.crs)
                data={'array':array,'affine':affine,'source_crs':crs}
            else:
                vector=load_vector(self.path);tr=Transformer.from_crs(vector['crs'],3857,always_xy=True)
                from shapely.ops import transform
                data={'source_crs':vector['crs'],'features':[(f['id'],transform(tr.transform,f['native']),f['properties']) for f in vector['features']]}
            if not self.isInterruptionRequested():self.result.emit(data)
        except Exception as e:self.failure.emit(str(e))

class LayerTab(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent);self.manager=LayerManager();self.items={};self.origin=None;self.worker=None;self.pending=[];self.cloud_data={}
        layout=QVBoxLayout(self);buttons=QHBoxLayout()
        self.add_button=QPushButton('Добавить слой');self.add_button.clicked.connect(self.browse);buttons.addWidget(self.add_button)
        for text,slot in [('Вписать',self.fit),('Сохранить сцену',self.save_scene),('Открыть сцену',self.open_scene)]:
            b=QPushButton(text);b.clicked.connect(slot);buttons.addWidget(b)
        self.color_mode=QComboBox();self.color_mode.addItems(['Высота','Классы','RGB','Intensity','HAG','Score']);self.color_mode.currentTextChanged.connect(self.recolor);buttons.addWidget(self.color_mode)
        layout.addLayout(buttons);split=QHBoxLayout()
        self.list=QListWidget();self.list.setDragDropMode(QAbstractItemView.InternalMove)
        self.list.itemChanged.connect(self.visibility_changed);self.list.currentItemChanged.connect(self.selected)
        self.list.model().rowsMoved.connect(self.reorder);self.list.setMaximumWidth(280);split.addWidget(self.list)
        self.plot=pg.PlotWidget();self.plot.setAspectLocked(True);self.plot.showGrid(x=True,y=True,alpha=.2)
        self.plot.setLabel('bottom','Восток от local origin','m');self.plot.setLabel('left','Север от local origin','m')
        split.addWidget(self.plot,1);layout.addLayout(split)
        self.opacity=QSlider(Qt.Horizontal);self.opacity.setRange(0,100);self.opacity.setValue(100);self.opacity.valueChanged.connect(self.opacity_changed)
        layout.addWidget(QLabel('Прозрачность выбранного слоя'));layout.addWidget(self.opacity)
        self.info=QLabel('Offline: LAS/LAZ, GeoJSON, GeoTIFF. Двойной клик на точке показывает исходные координаты. Перетаскивайте слои для порядка.')
        self.info.setWordWrap(True);layout.addWidget(self.info)
        self.plot.scene().sigMouseClicked.connect(self.map_click)
    def browse(self):
        path,_=QFileDialog.getOpenFileName(self,'Слой','','GIS (*.las *.laz *.geojson *.tif *.tiff)')
        if path:self.load_path(path)
    def load_path(self,path,existing=None):
        if self.worker is not None and self.worker.isRunning():
            self.pending.append((path,existing));return
        suffix=Path(path).suffix.lower();kind='cloud' if suffix in ('.las','.laz') else ('raster' if suffix in ('.tif','.tiff') else 'vector')
        self.add_button.setEnabled(False);self.info.setText('Чтение и преобразование слоя: '+Path(path).name)
        self.worker=Loader(path,kind)
        self.worker.result.connect(lambda data:self.render(path,kind,data,existing))
        self.worker.failure.connect(self.info.setText)
        self.worker.finished.connect(self.finished);self.worker.start()
    def finished(self):
        self.add_button.setEnabled(True)
        if self.pending:
            path,existing=self.pending.pop(0);self.load_path(path,existing)
    def render(self,path,kind,data,existing=None):
        layer=existing or self.manager.add(path,kind,data['source_crs'])
        rendered=[]
        if kind=='cloud':
            xy=data['display'][:,:2]
            if self.origin is None:self.origin=xy.min(axis=0)
            scatter=pg.ScatterPlotItem(x=xy[:,0]-self.origin[0],y=xy[:,1]-self.origin[1],size=3,pen=None,
                                      brush=pg.mkBrush(78,201,176),data=np.arange(len(xy)))
            scatter.sigClicked.connect(lambda item,spots,event:self.pick_cloud(data,spots));rendered.append(scatter)
            self.cloud_data[layer.uuid]=(data,scatter)
            description=f"{data['input_count']:,} исходных точек; показано {len(xy):,} по устойчивой выборке"
        elif kind=='vector':
            def draw(geom):
                if hasattr(geom,'geoms'):
                    for part in geom.geoms:draw(part)
                    return
                if geom.geom_type=='Polygon':
                    rings=[geom.exterior,*geom.interiors]
                elif geom.geom_type in ('LineString','LinearRing'):rings=[geom]
                elif geom.geom_type=='Point':rings=[geom]
                else:return
                for ring in rings:
                    xy=np.asarray(ring.coords)[:,:2]
                    if self.origin is None:self.origin=xy.min(axis=0)
                    rendered.append(pg.PlotDataItem(xy[:,0]-self.origin[0],xy[:,1]-self.origin[1],pen=pg.mkPen('#e6b34a',width=1),symbol='o' if geom.geom_type=='Point' else None))
            for uid,geom,attrs in data['features']:draw(geom)
            description=f"{len(data['features']):,} объектов; GeoJSON, горизонтальное перепроецирование"
        else:
            arr=data['array'];affine=data['affine']
            x0,y0=affine*(0,0);x1,y1=affine*(arr.shape[2],arr.shape[1])
            if self.origin is None:self.origin=np.array([min(x0,x1),min(y0,y1)])
            if len(arr)==1:
                image=arr[0][::-1,:].T
            else:image=np.moveaxis(arr[:,::-1,:],0,-1).transpose(1,0,2)
            item=pg.ImageItem(image);item.setRect(QRectF(x0-self.origin[0],y1-self.origin[1],x1-x0,y0-y1));rendered.append(item)
            description='Растр перепроецирован для показа; исходник сохранён, Z курсора неизвестен'
        for item in rendered:self.plot.addItem(item);item.setOpacity(layer.opacity);item.setVisible(layer.visible)
        self.items[layer.uuid]=rendered
        li=QListWidgetItem(layer.name);li.setData(Qt.UserRole,layer.uuid);li.setFlags(li.flags()|Qt.ItemIsUserCheckable);li.setCheckState(Qt.Checked if layer.visible else Qt.Unchecked)
        self.list.addItem(li);self.list.setCurrentItem(li);self.reorder();self.fit();self.info.setText(description);self.recolor()
    def fit(self):self.plot.enableAutoRange()
    def selected(self,*args):
        item=self.list.currentItem()
        if item:
            layer=next(l for l in self.manager.layers if l.uuid==item.data(Qt.UserRole));self.opacity.setValue(int(layer.opacity*100));self.recolor()
    def recolor(self,*args):
        item=self.list.currentItem()
        if not item or item.data(Qt.UserRole) not in self.cloud_data:return
        data,scatter=self.cloud_data[item.data(Qt.UserRole)];attrs=data['attributes'];mode=self.color_mode.currentText()
        if mode=='RGB' and all(k in attrs for k in ('red','green','blue')):
            colors=np.column_stack([attrs[k] for k in ('red','green','blue')]).astype(float)/65535
        else:
            key={'Классы':'classification','Intensity':'intensity','HAG':'c3_hag','Score':'c3_score'}.get(mode)
            values=data['native'][:,2] if mode=='Высота' else attrs.get(key)
            if values is None:
                self.info.setText('Выбранный канал отсутствует: '+mode);return
            values=np.asarray(values,float);finite=np.isfinite(values)
            norm=np.zeros(len(values))
            if finite.any():
                lo,hi=np.nanmin(values),np.nanmax(values);norm[finite]=(values[finite]-lo)/max(hi-lo,1e-12)
            colors=pg.colormap.get('viridis').map(norm,mode='float')[:,:3];colors[~finite]=.5
        scatter.setBrush([pg.mkBrush(*(np.clip(color,0,1)*255).astype(int),200) for color in colors])
    def opacity_changed(self,value):
        item=self.list.currentItem()
        if item:
            uid=item.data(Qt.UserRole);self.manager.update(uid,opacity=value/100)
            for p in self.items.get(uid,[]):p.setOpacity(value/100)
    def visibility_changed(self,item):
        uid=item.data(Qt.UserRole);visible=item.checkState()==Qt.Checked;self.manager.update(uid,visible=visible)
        for p in self.items.get(uid,[]):p.setVisible(visible)
    def reorder(self,*args):
        for row in range(self.list.count()):
            uid=self.list.item(row).data(Qt.UserRole);self.manager.update(uid,order=row)
            for item in self.items.get(uid,[]):item.setZValue(self.list.count()-row)
    def pick_cloud(self,data,spots):
        if not spots:return
        i=int(spots[0].data());xyz=data['native'][i];lon,lat=Transformer.from_crs(data['source_crs'],4326,always_xy=True).transform(xyz[0],xyz[1])
        attrs=data.get("attributes",{});label=int(attrs["classification"][i]) if "classification" in attrs else None
        provenance=int(attrs["c3_source"][i]) if "c3_source" in attrs else None
        self.info.setText(f"LAS class={label}; source={provenance}; original record ID={int(data['point_ids'][i])}; XYZ={xyz.tolist()} ({data['source_crs']}); WGS84 Б={lat:.8f}, Л={lon:.8f}; Z — исходная высота, HAG отдельно")
    def map_click(self,event):
        if self.origin is None:return
        point=self.plot.plotItem.vb.mapSceneToView(event.scenePos());x,y=point.x()+self.origin[0],point.y()+self.origin[1]
        lon,lat=Transformer.from_crs(3857,4326,always_xy=True).transform(x,y)
        if not event.double():self.info.setText(f"Курсор WGS84 Б={lat:.8f}, Л={lon:.8f}; display XY={x:.3f}, {y:.3f}; Z=NoData")
    def save_scene(self):
        path,_=QFileDialog.getSaveFileName(self,'Сохранить сцену','','JSON (*.json)')
        if path:self.manager.save(path)
    def open_scene(self):
        path,_=QFileDialog.getOpenFileName(self,'Открыть сцену','','JSON (*.json)')
        if not path:return
        self.manager=LayerManager.load(path);self.plot.clear();self.list.clear();self.items={};self.origin=None
        for layer in sorted(self.manager.layers,key=lambda l:l.order):self.load_path(layer.source,layer)

class OfflineMapViewer(LayerTab):
    def __init__(self,parent=None):super().__init__(parent);self.last_truncated=False;self.last_feature_count=0
    def show_geojson(self,path):
        self.load_path(path);return True
