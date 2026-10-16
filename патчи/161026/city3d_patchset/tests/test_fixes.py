import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types
import laspy
import numpy as np
import pytest
from pyproj import CRS
from shapely.geometry import box,Polygon,shape
from city3d_tab1.catalog import inspect_file
from city3d_fixes.db import connection_options
from city3d_fixes.project import build_revision,point_boundary,validate_polygon
from city3d_fixes.dataset import grouped_split
from city3d_fixes.services import export_cloud,scan_archive

ROOT=Path(__file__).parents[1]


def cloud(path,shift=0,classes=None,crs=32636,offset=0):
    h=laspy.LasHeader(point_format=6,version='1.4')
    h.offsets=np.array([offset,0,0],float)
    h.add_crs(CRS.from_epsg(crs))
    h.add_extra_dim(laspy.ExtraBytesParams(name='custom_value',type=np.float32))
    d=laspy.LasData(h)
    d.x=np.array([0,0,2,2,1])+500000+shift
    d.y=np.array([0,2,2,0,1])+6600000
    d.z=np.array([10,11,12,13,14])
    d.classification=np.array(classes or [2,6,5,1,0],dtype=np.uint8)
    d.intensity=np.arange(5)+100
    d.custom_value=np.arange(5)*.25
    d.write(path)
    return inspect_file(path,scan_type='TLS' if shift else 'ALS_uav')


def test_db_keys():
    from psycopg.conninfo import make_conninfo
    opts=connection_options(dict(dbname='metadata',host='localhost',schema_refs='refs',schema_history='history',sslmode='prefer'))
    assert 'schema_refs' not in opts and 'schema_history' not in opts
    assert 'dbname=metadata' in make_conninfo(**opts)
    assert connection_options({'database':'city3d'})=={'dbname':'city3d'}


def test_merged_all_sources_and_identity(tmp_path):
    a=cloud(tmp_path/'a.las');b=cloud(tmp_path/'b.las',shift=1,offset=500000)
    p=build_revision([a,b],'Район СПб',tmp_path/'projects')
    data=laspy.read(p['path'])
    assert len(data.points)==10
    assert set(data.c3_source_index)=={0,1}
    assert list(data.c3_original_index)==list(range(5))*2
    assert list(data.classification)==[2,6,5,1,0]*2
    assert list(data.custom_value)==[0,.25,.5,.75,1]*2
    assert p['sources'][1]['scan_type']=='TLS'
    assert (Path(p['path']).parent/'project.json').exists()
    clipped=laspy.read(p['sources'][1]['clipped_path'])
    assert list(clipped.c3_original_index)==list(range(5))
    assert np.allclose(clipped.x,[500001,500001,500003,500003,500002])


def test_polygon_clip_includes_edges(tmp_path):
    r=cloud(tmp_path/'a.las')
    polygon=box(500000,6600000,500001,6600002)
    p=build_revision([r],'clip',tmp_path/'p',polygon=polygon)
    d=laspy.read(p['path'])
    assert len(d.points)==3
    assert list(d.c3_original_index)==[0,1,4]
    assert shape(p['boundary']).equals(polygon)


def test_duplicate_and_crs_guards(tmp_path):
    a=cloud(tmp_path/'a.las');b=cloud(tmp_path/'b.las',shift=5,crs=32637)
    with pytest.raises(ValueError,match='одинаковые'):build_revision([a,a],'p',tmp_path/'p')
    with pytest.raises(ValueError,match='CRS'):build_revision([a,b],'p',tmp_path/'p')
    with pytest.raises(RuntimeError,match='отменена'):build_revision([a],'p',tmp_path/'p',cancel=lambda:True)


def test_unknown_changed_and_invalid_polygon(tmp_path):
    a=cloud(tmp_path/'a.las')
    unknown=dict(a,crs_wkt=None)
    with pytest.raises(ValueError,match='CRS'):build_revision([unknown],'p',tmp_path/'p')
    with pytest.raises(ValueError):validate_polygon(Polygon([(0,0),(2,2),(0,2),(2,0)]))
    with open(a['path'],'ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='изменился'):build_revision([a],'p',tmp_path/'p')


@pytest.mark.parametrize('kind,suffix',[('LAS','.las'),('LAZ','.laz'),('CSV','.csv'),('BOUNDARY','.geojson')])
def test_export(tmp_path,kind,suffix):
    a=cloud(tmp_path/'a.las');p=build_revision([a],'name',tmp_path/'p')
    out=tmp_path/('export'+suffix)
    export_cloud(p['path'],out,kind,p)
    assert out.stat().st_size>0
    assert out.with_suffix(suffix+'.provenance.json').exists()
    if kind in ('LAS','LAZ'):
        d=laspy.read(out);assert len(d.points)==5 and list(d.classification)==[2,6,5,1,0]
    if kind=='BOUNDARY':assert json.loads(out.read_text())['features'][0]['properties']['name']=='name'
    with pytest.raises(ValueError,match='существует'):export_cloud(p['path'],out,kind,p)


def test_group_split_overlap(tmp_path):
    files=[]
    for i,shift in enumerate([0,1,100,200]):
        p=tmp_path/f'{i}.las';cloud(p,shift);files.append(p)
    split=grouped_split(files)
    assert all(split.values())
    assert next(k for k,v in split.items() if files[0] in v)==next(k for k,v in split.items() if files[1] in v)
    with pytest.raises(ValueError,match='Меньше трёх'):grouped_split(files[:2])


def test_history_does_not_invent_success(tmp_path):
    root=tmp_path/'20261016-100000_M1';root.mkdir()
    (root/'results').mkdir();(root/'results/a.npz').touch()
    rows=scan_archive(tmp_path)
    assert len(rows)==1 and rows[0]['status']=='archive_unverified'
    assert rows[0]['area_name']=='не установлено'
    (root/'project_context.json').write_text(json.dumps({'name':'Район','project_id':'id'}))
    assert scan_archive(tmp_path)[0]['area_name']=='Район'


def test_editor_arc_and_undo(monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM','offscreen')
    from PyQt5.QtWidgets import QApplication
    from city3d_fixes.editor import BoundaryEditor
    app=QApplication.instance() or QApplication([])
    e=BoundaryEditor(box(0,0,10,10))
    original=copy.deepcopy(e.rings)
    e.arc=(0,0,0,2);e.new_points=[[15,5]]
    e.finish_arc()
    assert [15,5] in e.rings[0][0]
    e.undo();assert e.rings==original
    e.accept_geometry();assert e.geometry.area==100
    e.close()


def test_installer_cumulative_and_restore(tmp_path):
    source=ROOT/'fixtures/city3d_pipeline_v5.py'
    if not source.exists():pytest.skip('Проверка установки требует переданного исходника')
    target=tmp_path/'city3d_pipeline_v5.py';shutil.copy2(source,target)
    command=[sys.executable,str(ROOT/'apply_patchset.py'),'--target',str(target),'--backup-dir',str(tmp_path/'backup')]
    for level in ('01','02','03'):
        r=subprocess.run(command+['--patch',level,'--apply'],capture_output=True,text=True)
        assert r.returncode==0,r.stdout+r.stderr
        assert 'CASE WHEN %s::text IS NULL' in target.read_text()
        assert (tmp_path/'city3d_tab1/gui.py').exists()
        assert (tmp_path/'city3d_fixes/integration.py').exists()
    backups=sorted((tmp_path/'backup').iterdir())
    for b in reversed(backups):
        r=subprocess.run([sys.executable,str(ROOT/'apply_patchset.py'),'--restore',str(b)],capture_output=True,text=True)
        assert r.returncode==0,r.stdout+r.stderr
    assert target.read_bytes()==source.read_bytes()


def test_mock_osm_clips_and_tags(tmp_path,monkeypatch):
    import io
    from city3d_fixes import references
    monkeypatch.setattr(references,'to_wgs84',lambda *args:box(30,59,30.01,59.01))
    payload={'elements':[{'type':'way','id':1,'tags':{'highway':'residential'},
                         'geometry':[{'lon':29.99,'lat':59.005},{'lon':30.02,'lat':59.005}]}]}
    monkeypatch.setattr(references.urllib.request,'urlopen',lambda *a,**k:io.BytesIO(json.dumps(payload).encode()))
    p=dict(boundary=mapping_box(),crs_wkt='EPSG:32636',name='name',project_id='id',revision_id='revision')
    out=references.fetch_osm(p,tmp_path)
    feature=json.loads(out.read_text())['features'][0]
    assert feature['properties']['area_name']=='name'
    assert shape(feature['geometry']).bounds==(30,59.005,30.01,59.005)


def mapping_box():
    from shapely.geometry import mapping
    return mapping(box(0,0,1,1))


def test_mock_dem_crop(tmp_path,monkeypatch):
    import io,gzip
    import rasterio
    from city3d_fixes import references
    monkeypatch.setattr(references,'to_wgs84',lambda *args:box(30.01,59.01,30.02,59.02))
    raw=np.full((1201,1201),123,dtype='>i2').tobytes()
    monkeypatch.setattr(references.urllib.request,'urlopen',lambda *a,**k:io.BytesIO(gzip.compress(raw)))
    project=dict(boundary=mapping_box(),crs_wkt='EPSG:32636',name='area',project_id='uuid')
    output=references.fetch_dem(project,tmp_path)
    with rasterio.open(output) as reader:
        assert reader.width<1201 and reader.height<1201
        assert 123 in reader.read(1)
    provenance=json.loads((tmp_path/'dem_provenance.json').read_text())
    assert 'multiple source' in provenance['source']


def test_real_load_tab_checkboxes_and_active_project(tmp_path,monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM','offscreen')
    from PyQt5 import QtWidgets,QtCore,QtGui
    from city3d_tab1 import gui
    from city3d_tab1.catalog import Journal
    from city3d_fixes.integration import install_project
    monkeypatch.setattr(gui,'Journal',lambda:Journal(tmp_path/'inventory.sqlite'))
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    source=ROOT/'fixtures/city3d_pipeline_v5.py'
    if not source.exists():pytest.skip('Нужен исходник приложения')
    tree=ast.parse(source.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LoadDataTab')
    ns={k:getattr(QtWidgets,k) for k in dir(QtWidgets) if not k.startswith('_')}
    ns.update(Qt=QtCore.Qt,Path=Path,WORKSPACE=tmp_path,VERTICAL_CRS_LIST=[],
              CRSDetector=lambda **kw:types.SimpleNamespace(),_ACTIVE_WORKERS=[])
    class Viewer(QtWidgets.QWidget):
        def set_pointcloud(self,pcd):self.pcd=pcd
    ns['PointCloudViewer']=Viewer
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual LoadDataTab>','exec'),ns)
    ns['LoadDataTab']=gui.upgrade_load_tab(ns['LoadDataTab'],ns)
    ns.update(DatasetPreparer=object,DatasetPrepareTab=QtWidgets.QWidget,LogWidget=QtWidgets.QTextEdit)
    class Main(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__();self.history_db=None;self.epsg_catalog=None
            self.tabs=QtWidgets.QTabWidget();self.setCentralWidget(self.tabs)
            self.tabs.addTab(ns['LoadDataTab'](self),'load')
    ns['MainWindow']=Main
    ns['o3d']=types.SimpleNamespace(geometry=types.SimpleNamespace(PointCloud=types.SimpleNamespace),
                                  utility=types.SimpleNamespace(Vector3dVector=lambda p:p))
    install_project(ns)
    mw=ns['MainWindow']();tab=mw.tabs.widget(0)
    a=cloud(tmp_path/'a.las');b=cloud(tmp_path/'b.las',shift=1)
    tab.accept_record(a);tab.accept_record(b)
    tab.check_all(True);tab.area_name.setText('Тестовый район')
    assert len(tab.chosen())==2
    p=build_revision(tab.chosen(),'Тестовый район',tmp_path/'projects')
    las=laspy.read(p['path'])
    tab.activate((p,{'points':np.column_stack((las.x,las.y,las.z)),'classification':np.array(las.classification)}))
    assert mw.pcd_path==p['path'] and len(mw.pcd.points)==10
    assert 'Тестовый район' in mw.project_labels[0].text()
    assert len(tab.chosen())==2
    mw.close()
