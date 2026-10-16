import ast
import json
from pathlib import Path
import shutil
import sys
import types

import laspy
import numpy as np
import pytest
from pyproj import CRS

from city3d_tab1.catalog import inspect_file, discover, Journal, Cancelled
from city3d_tab1.custom_crs import CustomCatalog
import install_tab1


def make_las(path, crs=32637, classes=(0, 1, 2, 6, 5, 255)):
    header = laspy.LasHeader(point_format=6, version='1.4')
    if crs:
        header.add_crs(CRS.from_epsg(crs))
    data = laspy.LasData(header)
    n = len(classes)
    data.x = 500000 + np.arange(n) * 10
    data.y = 6000000 + np.arange(n) * 12
    data.z = 100 + np.arange(n)
    data.classification = classes
    data.write(path)


@pytest.mark.parametrize('suffix', ['.las', '.laz'])
def test_stream_histogram_and_identity(tmp_path, suffix):
    path = tmp_path / ('input' + suffix)
    make_las(path)
    before = path.read_bytes()
    record = inspect_file(path, chunk_size=2)
    assert record['point_count'] == 6
    assert record['classes'] == {'0': 1, '1': 1, '2': 1, '5': 1, '6': 1, '255': 1}
    assert record['assigned_class_count'] == 4
    assert record['crs_authority'] == ('EPSG', '32637')
    assert record['footprint_wgs84'].startswith('MULTIPOLYGON')
    assert record['state'] == 'indexed' and not record['reviewed']
    assert path.read_bytes() == before
    same = inspect_file(path, chunk_size=3)
    assert same['file_id'] == record['file_id']
    assert same['inspection_id'] != record['inspection_id']


def test_unknown_conflict_override(tmp_path):
    path = tmp_path / 'unknown.las'
    make_las(path, None)
    r = inspect_file(path)
    assert r['crs_source'] == 'unknown' and r['footprint_wgs84'] is None
    make_las(path)
    path.with_suffix('.prj').write_text(CRS.from_epsg(32638).to_wkt())
    r = inspect_file(path)
    assert r['crs_source'] == 'conflict' and r['crs_wkt'] is None
    r = inspect_file(path, override='EPSG:32637')
    assert r['crs_source'] == 'operator' and r['footprint_wgs84']


def test_cancel_and_journal(tmp_path):
    path = tmp_path / 'input.las'
    make_las(path)
    with pytest.raises(Cancelled):
        inspect_file(path, cancel=lambda: True)
    r = inspect_file(path)
    journal = Journal(tmp_path / 'inventory.sqlite3')
    journal.save(r)
    journal.save(r)
    assert len(journal.records()) == 1
    assert journal.records()[0]['sha256'] == r['sha256']


def test_discovery(tmp_path):
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'a.LAS').touch()
    (tmp_path / 'sub/b.laz').touch()
    (tmp_path / 'ignore.pdf').touch()
    assert len(list(discover([tmp_path, tmp_path / 'a.LAS']))) == 2
    assert len(list(discover([tmp_path], recursive=False))) == 1


def test_custom_crs_crud_versions(tmp_path):
    catalog = CustomCatalog(tmp_path / 'custom.json')
    one = catalog.save('МСК тест, зона 1', 'EPSG:32637', 'Тестовый документ')
    snapshot = one['wkt']
    two = catalog.save('МСК тест, исправленная', 'EPSG:32638', 'Документ 2', one['id'])
    assert two['id'] == one['id'] and two['revision'] == 2
    assert snapshot != two['wkt']
    assert len(list((tmp_path / 'custom_crs_history').glob('*.json'))) == 1
    catalog.delete(one['id'])
    assert catalog.entries() == []
    assert len(list((tmp_path / 'custom_crs_history').glob('*.json'))) == 2
    with pytest.raises(ValueError):
        catalog.save('', 'EPSG:32637', '')
    with pytest.raises(Exception):
        catalog.save('bad', 'not a CRS', 'test')


def test_sql_parses():
    from pglast import parse_sql
    sql = Path(__file__).parents[1] / 'city3d_tab1/schema.sql'
    assert parse_sql(sql.read_text())


def test_installer_and_rollback(tmp_path, monkeypatch):
    source = Path(__file__).parents[2] / 'upload/city3d_pipeline_v5(2).py'
    if not source.exists():
        pytest.skip('Integration source not bundled; supply original upload to repeat')
    target = tmp_path / 'city3d_pipeline_v5.py'
    shutil.copy2(source, target)
    before = target.read_bytes()
    install_tab1.install(target, tmp_path / 'backup')
    assert target.read_bytes() == before
    monkeypatch.setattr(install_tab1, 'check_environment', lambda: None)
    install_tab1.install(target, tmp_path / 'backup', apply=True)
    first = target.read_bytes()
    install_tab1.install(target, tmp_path / 'backup', apply=True)
    assert target.read_bytes() == first
    tree_before, tree_after = ast.parse(before), ast.parse(first)
    old_nodes = [ast.dump(n) for n in tree_before.body]
    new_nodes = [ast.dump(n) for n in tree_after.body]
    # Every original top-level definition must remain byte-independent AST-identical.
    assert all(n in new_nodes for n in old_nodes)
    restore = next((tmp_path / 'backup').glob('*/restore.json'))
    install_tab1.restore(restore)
    assert target.read_bytes() == before
    assert not (tmp_path / 'city3d_tab1').exists()


def test_refuse_unknown(tmp_path):
    target = tmp_path / 'app.py'
    target.write_text('print("unknown")')
    with pytest.raises(RuntimeError):
        install_tab1.install(target, tmp_path / 'backups', apply=True)
    assert target.read_text() == 'print("unknown")'


def test_gui_catalog_wiring(tmp_path, monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QLabel
    from city3d_tab1 import gui
    monkeypatch.setattr(gui, 'Journal', lambda: Journal(tmp_path / 'gui.sqlite3'))
    app = QApplication.instance() or QApplication([])
    class Base(QWidget):
        def __init__(self, main):
            super().__init__()
            self.main_window = main
            self.crs_detector = types.SimpleNamespace()
            QVBoxLayout(self)
            self.btn_load = QPushButton()
            self.file_path = QLineEdit()
            self.crs_display = QLineEdit()
            self.crs_source = QLineEdit()
            self.info_label = QLabel()
        def current_scanning_type(self):
            return 'unknown'
    patched = gui.upgrade_load_tab(Base, {})
    widget = patched(types.SimpleNamespace())
    path = tmp_path / 'a.las'
    make_las(path)
    record = inspect_file(path)
    widget.accept_record(record)
    widget.table.selectRow(0)
    app.processEvents()
    assert widget.file_path.text() == str(path)
    assert widget.table.rowCount() == 1
    widget.close()
