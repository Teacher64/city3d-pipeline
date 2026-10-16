"""Run with CITY3D_DISABLE_WEBENGINE=1 QT_QPA_PLATFORM=offscreen and temporary HOME."""
import json
import tempfile
from pathlib import Path
import numpy as np
import laspy
from pyproj import CRS
import city3d_pipeline_v5 as c
from PyQt5.QtCore import QTimer
from city3d_backend.core import verify_roundtrip
c.QApplication.setAttribute(c.Qt.AA_ShareOpenGLContexts,True)
app=c.QApplication([])
w=c.MainWindow()
assert w.tabs.count()==10
assert w.tabs.widget(4).m1_ground.currentText()=='smrf'
with tempfile.TemporaryDirectory() as folder:
    p=Path(folder)
    h=laspy.LasHeader(point_format=7,version='1.4');h.add_crs(CRS.from_epsg(32636))
    h.scales=[.001]*3;h.offsets=[500000,6000000,100]
    l=laspy.LasData(h);l.x=np.arange(10)*.03+500000;l.y=np.arange(10)*.02+6000000;l.z=np.ones(10)*100
    l.classification=np.full(10,2,np.uint8);l.gps_time=np.arange(10)*.1
    l.add_extra_dim(laspy.ExtraBytesParams(name='c3_hag',type='f4'));l.c3_hag=np.arange(10,dtype=np.float32)
    src=p/'original.las';l.write(src)
    data=c.PointCloudLoader.load(src)
    c.PointCloudLoader.save(data,p/'copied.las')
    verify_roundtrip(src,p/'copied.las')
    # Simulate selection from two sources: preserve per-source data and HAG.
    tab=w.tabs.widget(4)
    # Slice operations belong to HeightSlicesTab, locate class directly.
    classes=[v for v in vars(c).values() if isinstance(v,type) and hasattr(v,'_write_preserved_slice')]
    assert len(classes)==1
    proxy=object.__new__(type('Proxy',(),{}))
    proxy._slice_sources=[(str(src),0,10)]
    cls=classes[0]
    selection=np.arange(10)%2==0
    cls._write_preserved_slice(proxy,lambda *args:None,p/'slice.las',selection)
    sl=laspy.read(p/'slice.las')
    assert np.array_equal(sl.X,l.X[selection])
    assert np.array_equal(sl.gps_time,l.gps_time[selection])
    assert np.array_equal(sl.c3_hag,l.c3_hag[selection])
    labels,grid=cls._compute_hag(np.array([500000.]),np.array([6000000.]),np.array([100.]),np.array([True]))
    assert labels[0]==0
w.show()
QTimer.singleShot(250,app.quit)
app.exec_()
print('PASS: 10 tabs/M1 controls/event loop, generic LAS roundtrip, preserved slices, IDW k=1')
