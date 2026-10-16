"""2D polygon editor: vertex drag and three-click arc replacement."""
import copy
import numpy as np
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox, QMenu
from PyQt5.QtGui import QCursor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from shapely.geometry import Polygon, MultiPolygon
from .project import validate_polygon


class BoundaryEditor(QDialog):
    def __init__(self, geometry, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Граница района — координаты исходной проекции')
        self.resize(1100, 780)
        polys = list(geometry.geoms) if geometry.geom_type == 'MultiPolygon' else [geometry]
        self.rings = [[np.array(p.exterior.coords)[:-1].tolist()] +
                      [np.array(r.coords)[:-1].tolist() for r in p.interiors] for p in polys]
        self.original = copy.deepcopy(self.rings)
        self.history = []
        self.mode = 'vertex'
        self.drag = None
        self.anchors = []
        self.arc = None
        self.new_points = []
        self.geometry = geometry
        layout = QVBoxLayout(self)
        self.hint = QLabel('Перетаскивайте вершины левой кнопкой. Панорама/масштаб — панель инструментов.')
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        figure = Figure()
        self.canvas = FigureCanvasQTAgg(figure)
        self.axes = figure.add_subplot(111)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        row = QHBoxLayout()
        for text, fn in [('Вершины', lambda: self.set_mode('vertex')),
                         ('Заменить участок (3 клика)', lambda: self.set_mode('arc')),
                         ('Отменить правку', self.undo),('Завершить участок', self.finish_arc),
                         ('Применить границу', self.accept_geometry),('Отмена', self.reject)]:
            b = QPushButton(text); b.clicked.connect(fn); row.addWidget(b)
        layout.addLayout(row)
        self.canvas.mpl_connect('button_press_event', self.press)
        self.canvas.mpl_connect('motion_notify_event', self.move)
        self.canvas.mpl_connect('button_release_event', self.release)
        self.draw(reset=True)

    def set_mode(self, mode):
        self.mode = mode; self.anchors=[]; self.arc=None; self.new_points=[]
        self.hint.setText('1: первая вершина; 2: вторая вершина того же кольца; 3: ребро заменяемого участка. '
                          'Затем новые положения левой кнопкой; завершение — меню правой кнопки или двойной правый клик.'
                          if mode == 'arc' else 'Перетаскивайте вершины левой кнопкой.')
        self.draw()

    def draw(self, reset=False, cursor=None):
        limits = (self.axes.get_xlim(), self.axes.get_ylim())
        self.axes.clear()
        for poly in self.rings:
            for ring in poly:
                a = np.array(ring + ring[:1])
                self.axes.plot(a[:,0],a[:,1],'-o',markersize=3)
        if self.arc:
            p,r,start,end = self.arc
            a = [self.rings[p][r][start]] + self.new_points
            if cursor is not None:a.append(cursor)
            a.append(self.rings[p][r][end])
            a=np.array(a);self.axes.plot(a[:,0],a[:,1],'r--')
        self.axes.set_aspect('equal', adjustable='datalim')
        self.axes.set_xlabel('X');self.axes.set_ylabel('Y');self.axes.grid(True)
        if not reset:self.axes.set_xlim(limits[0]);self.axes.set_ylim(limits[1])
        self.canvas.draw_idle()

    def nearest(self, event):
        best = None
        for p,poly in enumerate(self.rings):
            for r,ring in enumerate(poly):
                xy=self.axes.transData.transform(np.array(ring))
                distances=np.sum((xy-[event.x,event.y])**2,axis=1)
                i=int(distances.argmin());d=distances[i]
                if best is None or d<best[0]:best=(d,p,r,i)
        return best[1:] if best and best[0]<=20**2 else None

    def press(self,event):
        if event.button == 3:
            if event.dblclick:self.finish_arc();return
            menu=QMenu(self)
            menu.addAction('Завершить редактирование участка',self.finish_arc)
            menu.addAction('Отменить правку',self.undo)
            menu.addAction('Применить границу',self.accept_geometry)
            menu.exec_(QCursor.pos());return
        if event.inaxes != self.axes or event.button != 1 or self.toolbar.mode:return
        vertex=self.nearest(event)
        if self.mode=='vertex':
            if vertex:self.history.append(copy.deepcopy(self.rings));self.drag=vertex
        elif self.arc:
            self.new_points.append([event.xdata,event.ydata]);self.draw()
        elif len(self.anchors)<2:
            if vertex:
                if self.anchors and (vertex[:2]!=self.anchors[0][:2] or vertex==self.anchors[0]):return
                self.anchors.append(vertex)
                self.hint.setText('Выбрано вершин: '+str(len(self.anchors))+'; следующий клик — '+('вторая вершина' if len(self.anchors)==1 else 'на ребре нужного участка'))
        else:
            p,r,start=self.anchors[0];end=self.anchors[1][2]
            ring=np.array(self.rings[p][r]);xy=self.axes.transData.transform(ring)
            a=xy;b=np.roll(xy,-1,axis=0);v=b-a
            t=np.clip(np.sum((np.array([event.x,event.y])-a)*v,axis=1)/np.maximum(np.sum(v*v,axis=1),1e-20),0,1)
            dist=np.sum((a+t[:,None]*v-[event.x,event.y])**2,axis=1)
            edge=int(dist.argmin())
            if dist[edge]>20**2:return
            forward=[];i=start
            while i!=end:forward.append(i);i=(i+1)%len(ring)
            if edge not in forward:start,end=end,start
            self.arc=(p,r,start,end)
            self.hint.setText('Укажите новые вершины участка. Правая кнопка — завершение.')
            self.draw()

    def move(self,event):
        if event.inaxes!=self.axes:return
        if self.drag:
            p,r,i=self.drag;self.rings[p][r][i]=[event.xdata,event.ydata];self.draw()
        elif self.arc:self.draw(cursor=[event.xdata,event.ydata])

    def release(self,event):
        self.drag=None

    def finish_arc(self):
        if not self.arc:return
        p,r,start,end=self.arc;ring=self.rings[p][r]
        self.history.append(copy.deepcopy(self.rings))
        replacement=[ring[start]]+self.new_points+[ring[end]]
        i=(end+1)%len(ring)
        while i!=start:replacement.append(ring[i]);i=(i+1)%len(ring)
        self.rings[p][r]=replacement
        self.arc=None;self.new_points=[];self.anchors=[];self.draw()

    def undo(self):
        if self.history:self.rings=self.history.pop()
        self.arc=None;self.anchors=[];self.new_points=[];self.drag=None;self.draw()

    def accept_geometry(self):
        self.finish_arc()
        try:
            polys=[Polygon(p[0],p[1:]) for p in self.rings]
            self.geometry=validate_polygon(polys[0] if len(polys)==1 else MultiPolygon(polys))
            self.accept()
        except Exception as exc:QMessageBox.warning(self,'Граница',str(exc))
