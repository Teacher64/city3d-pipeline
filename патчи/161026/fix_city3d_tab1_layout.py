#!/usr/bin/env python3
# City3D GUI repair; standard-library installer, Qt self-test in existing venv.
RUNTIME = '"""City3D load-tab layout repair. Reuses existing widgets and signal connections."""\nfrom PyQt5.QtCore import Qt\nfrom PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QBoxLayout,\n    QLayout, QSplitter, QScrollArea, QSizePolicy, QLabel, QPushButton, QApplication)\n\n\ndef _vertical_rows(layout):\n    # A single narrow column avoids minimum-width overflow from long toolbars.\n    if isinstance(layout, QHBoxLayout):\n        layout.setDirection(QBoxLayout.TopToBottom)\n    for i in range(layout.count()):\n        item = layout.itemAt(i)\n        if item.layout():\n            _vertical_rows(item.layout())\n\n\ndef repair(tab):\n    if getattr(tab, \'_layout_repaired_20261016\', False):\n        return\n    root = tab.layout()\n    original = next((root.itemAt(i).widget() for i in range(root.count())\n                     if isinstance(root.itemAt(i).widget(), QSplitter)\n                     and root.itemAt(i).widget().indexOf(tab.viewer) >= 0), None)\n    if original is None or original.count() != 2:\n        raise RuntimeError(\'Вкладка загрузки имеет неизвестную компоновку; нужен анализ её версии.\')\n    top = original.widget(0)\n    items = []\n    while root.count():\n        items.append(root.takeAt(0))\n    content = QWidget()\n    content.setObjectName(\'city3d_load_controls\')\n    controls = QVBoxLayout(content)\n    controls.setSizeConstraint(QLayout.SetMinAndMaxSize)\n    for item in items:\n        if item.widget() is original:\n            continue\n        if item.layout():\n            controls.addLayout(item.layout())\n        elif item.widget():\n            controls.addWidget(item.widget())\n        else:\n            controls.addItem(item)\n    controls.addWidget(top)\n    _vertical_rows(controls)\n    # Root rows from earlier patches now scroll; the existing viewer stays live.\n    scroll = QScrollArea()\n    scroll.setObjectName(\'city3d_load_scroll\')\n    scroll.setWidgetResizable(True)\n    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)\n    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)\n    scroll.setWidget(content)\n    scroll.setMinimumSize(200, 100)\n    tab.table.setMinimumHeight(220)\n    tab.table.setMaximumHeight(340)\n    tab.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)\n    tab.table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)\n    tab.table.setColumnWidth(0, 280)\n    tab.table.setColumnWidth(5, 140)\n    tab.table.setToolTip(\'Флажок «В работу» включает файл в район. Выделение строки показывает её CRS. \'\n                         \'После изменения флажков нажмите «Применить отмеченные LAS».\')\n    tab.file_path.setToolTip(\'Путь выделенной строки или применённого рабочего LAS; состав района задают флажки.\')\n    viewer = tab.viewer\n    viewer.setMinimumSize(0, 0)\n    viewer.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)\n    split = QSplitter(Qt.Horizontal)\n    split.setObjectName(\'city3d_load_splitter\')\n    split.addWidget(scroll)\n    split.addWidget(viewer)\n    split.setChildrenCollapsible(False)\n    split.setStretchFactor(0, 0)\n    split.setStretchFactor(1, 1)\n    split.setSizes([520, 600])\n    root.addWidget(split, 1)\n    # Move, don\'t recreate: load and cancel retain their original connected slots.\n    root.addWidget(tab.btn_load)\n    tab.btn_load.setText(\'Применить отмеченные LAS → все вкладки\')\n    tab.btn_load.setMinimumWidth(0)\n    tab.btn_load.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)\n    footer = QHBoxLayout()\n    tab.status.setMinimumWidth(0)\n    tab.status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)\n    tab.status.setWordWrap(True)\n    footer.addWidget(tab.status, 1)\n    footer.addWidget(tab.progress)\n    cancel = next((b for b in content.findChildren(QPushButton) if b.text() == \'Отмена\'), None)\n    if cancel:\n        footer.addWidget(cancel)\n    root.addLayout(footer)\n    original.setParent(None)\n    original.deleteLater()\n    tab._layout_repaired_20261016 = True\n    tab._load_scroll = scroll\n    tab._load_splitter = split\n\n\ndef install(ns):\n    base = ns[\'LoadDataTab\']\n    class LoadTab(base):\n        def __init__(self, *args, **kwargs):\n            super().__init__(*args, **kwargs)\n            repair(self)\n    ns[\'LoadDataTab\'] = LoadTab\n    base_window = ns[\'MainWindow\']\n    class Window(base_window):\n        def __init__(self, *args, **kwargs):\n            super().__init__(*args, **kwargs)\n            self.setMinimumSize(900, 600)\n            self.tabs.setUsesScrollButtons(True)\n            self.tabs.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)\n            screen = self.screen() or QApplication.primaryScreen()\n            if screen:\n                available = screen.availableGeometry()\n                self.setMinimumSize(min(900, available.width()), min(600, available.height()))\n                self.resize(min(1500, available.width()), min(950, available.height()))\n    ns[\'MainWindow\'] = Window\n'
TEST_CODE = "import ast\nimport json\nimport os\nfrom pathlib import Path\nimport sys\nimport tempfile\nimport types\n\n\ndef check(source, project, runtime):\n    os.environ['QT_QPA_PLATFORM'] = 'offscreen'\n    sys.path.insert(0, str(project))\n    from PyQt5 import QtWidgets as W\n    from PyQt5.QtCore import Qt, QPoint\n    from PyQt5.QtTest import QTest\n    from pyproj import CRS\n    import city3d_tab1.gui as inventory\n    from city3d_fixes.integration import install_project\n    app = W.QApplication.instance() or W.QApplication([])\n    tree = ast.parse(source)\n    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LoadDataTab')\n    base_main = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MainWindow')\n    init_ui = next(n for n in base_main.body if isinstance(n, ast.FunctionDef) and n.name == 'init_ui')\n    style = next(n.value.args[0].value for n in init_ui.body if isinstance(n, ast.Expr)\n                 and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)\n                 and n.value.func.attr == 'setStyleSheet')\n    class Detector:\n        def __init__(self, **kw): pass\n    class Viewer(W.QWidget):\n        def set_pointcloud(self, value): pass\n    records = [dict(path=f'/test/cloud_{i}.las', point_count=100, crs_wkt=CRS.from_epsg(32636).to_wkt(),\n                    crs_authority=['EPSG','32636'], crs_source='header', classes={'2':100},\n                    footprint_wgs84=None, scan_type='TLS') for i in range(40)]\n    class Journal:\n        def records(self): return records\n    original_journal = inventory.Journal\n    inventory.Journal = Journal\n    try:\n        with tempfile.TemporaryDirectory() as work:\n            ns = dict(vars(W), Qt=Qt, CRSDetector=Detector, PointCloudViewer=Viewer,\n                      VERTICAL_CRS_LIST=[('unknown','Не задана')], WORKSPACE=Path(work),\n                      DatasetPreparer=object, DatasetPrepareTab=W.QWidget, MainWindow=W.QMainWindow,\n                      DB_PARAMS={})\n            exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual LoadDataTab>', 'exec'), ns)\n            ns['LoadDataTab'] = inventory.upgrade_load_tab(ns['LoadDataTab'], ns)\n            install_project(ns)\n            repair_ns = {}\n            exec(compile(runtime, '<repair>', 'exec'), repair_ns)\n            repair_ns['install'](ns)\n            mw = types.SimpleNamespace(epsg_catalog=None, active_project=None)\n            tab = ns['LoadDataTab'](mw)\n            tab.setStyleSheet(style)\n            tab.show()\n            # Real connected actions; expensive jobs intercepted at their boundary.\n            calls = []\n            tab.build = lambda *a, **kw: calls.append('build')\n            tab.start_job = lambda *a, **kw: calls.append('job')\n            tab.index_paths = lambda paths: calls.append(paths)\n            buttons = {b.text(): b for b in tab.findChildren(W.QPushButton)}\n            buttons['Отметить все'].click()\n            assert len(tab.chosen()) == 40\n            buttons['Снять все'].click()\n            assert len(tab.chosen()) == 0\n            tab.table.item(0,0).setCheckState(Qt.Checked)\n            tab.table.item(1,0).setCheckState(Qt.Checked)\n            tab.table.cellWidget(1,5).setCurrentText('MLS')\n            assert [r['scan_type'] for r in tab.chosen()] == ['TLS', 'MLS']\n            tab.area_name.setText('Проверка GUI')\n            buttons['Сохранить выбор / название'].click()\n            saved = json.loads((Path(work)/'city3d_selection.json').read_text())\n            assert saved['name'] == 'Проверка GUI' and len(saved['selected']) == 2\n            buttons['Записать реестр в БД'].click()\n            assert calls[-1] == 'job'\n            tab.btn_load.click()\n            assert calls[-1] == 'build'\n            tab.table.selectRow(1)\n            assert tab.file_path.text() == records[1]['path']\n            picker = W.QFileDialog.getOpenFileNames\n            W.QFileDialog.getOpenFileNames = lambda *a, **kw: (['/test/a.las','/test/b.las'], '')\n            try: buttons['Добавить файлы'].click()\n            finally: W.QFileDialog.getOpenFileNames = picker\n            assert calls[-1] == ['/test/a.las','/test/b.las']\n            geometry = []\n            for width,height in [(800,480), (1024,600), (1366,650), (1920,900)]:\n                tab.resize(width,height)\n                for _ in range(4): app.processEvents()\n                assert tab.btn_load.isVisible() and tab.rect().contains(tab.btn_load.geometry())\n                scroll = tab._load_scroll\n                assert scroll.verticalScrollBar().maximum() > 0\n                for button in tab.findChildren(W.QPushButton):\n                    if scroll.widget().isAncestorOf(button):\n                        scroll.ensureWidgetVisible(button, 8, 8)\n                        app.processEvents()\n                        p = button.mapTo(scroll.viewport(), QPoint(0,0))\n                        assert p.y() >= 0 and p.y()+button.height() <= scroll.viewport().height(), button.text()\n                # No two visible control buttons occupy the same content rectangle.\n                group = [b for b in tab.findChildren(W.QPushButton) if scroll.widget().isAncestorOf(b)]\n                for i,b in enumerate(group):\n                    rect = b.rect(); rect.moveTopLeft(b.mapTo(scroll.widget(), QPoint(0,0)))\n                    for other in group[:i]:\n                        r = other.rect(); r.moveTopLeft(other.mapTo(scroll.widget(), QPoint(0,0)))\n                        assert not rect.intersects(r), (b.text(),other.text())\n                geometry.append([width,height])\n            before = len(tab.findChildren(W.QScrollArea))\n            repair_ns['repair'](tab)\n            assert len(tab.findChildren(W.QScrollArea)) == before\n            tab.close()\n            return dict(status='passed', sizes=geometry, records=40,\n                        checks=['all_buttons_scroll_reachable','buttons_no_overlap','apply_visible',\n                                'select_all_clear','two_files_and_scan_types','save_name_and_selection',\n                                'database_button_dispatch','apply_button_dispatch','file_dialog_dispatch',\n                                'selected_row_crs','idempotent_layout'],\n                        limitations='Offscreen Qt; viewer stub; no real DB, GPU, LAS processing or full application launch.')\n    finally:\n        inventory.Journal = original_journal\n"
SOURCE_SHA = '5e1fa99fbf7c00845d628598e9ace6598b0b86d297a5564b7f8088781be86a9f'
DEPENDENCIES = {'city3d_tab1/gui.py': 'ccfcdbeaa587caa8e019d24496c62f5f2e95c39582b5e168455820e7cb4e2914', 'city3d_fixes/integration.py': '93b98ff3e08396c82c27890c05440dc00f650a75cf14815585c036cabd256dcf'}

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

MARKER = '# CITY3D_LOAD_LAYOUT_REPAIR_20261016'
HOOK = '\n' + MARKER + '\nfrom city3d_load_layout_fix import install as _install_load_layout_fix\n_install_load_layout_fix(globals())\n'
MODULE = 'city3d_load_layout_fix.py'


def digest(data): return hashlib.sha256(data).hexdigest()


def atomic(path, data):
    fd, temp = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        if path.exists(): os.chmod(temp, path.stat().st_mode & 0o777)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def report(path, data):
    atomic(path, json.dumps(data, ensure_ascii=False, indent=2).encode())


def main():
    parser = argparse.ArgumentParser(description='Исправление GUI вкладки 1 после патчей 01–03. По умолчанию только проверка.')
    parser.add_argument('--target', type=Path, default=Path('/mnt/data_sda1/PR_3D_SPB/city3d-pipeline/city3d_pipeline_v5.py'))
    parser.add_argument('--backups', type=Path, default=Path('/mnt/data_sda1/city3d-backups'))
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--restore', type=Path)
    parser.add_argument('--self-test', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    target = args.target.expanduser().resolve()
    if args.self_test:
        scope = {}
        exec(compile(TEST_CODE, '<gui-check>', 'exec'), scope)
        print(json.dumps(scope['check'](target.read_text(encoding='utf-8'), target.parent, RUNTIME), ensure_ascii=False))
        return
    if args.restore:
        folder = args.restore.resolve()
        manifest = json.loads((folder/'manifest.json').read_text())
        if str(target) != manifest['target']: raise ValueError('Резервная копия относится к другому --target')
        for name, item in manifest['files'].items():
            p = target.parent/name
            if not p.exists() or digest(p.read_bytes()) != item['after']:
                raise ValueError('После установки файл изменён; автоматический откат остановлен: '+name)
            if item['before'] and digest((folder/name).read_bytes()) != item['before']:
                raise ValueError('Повреждена резервная копия: '+name)
        for name,item in manifest['files'].items():
            p = target.parent/name
            if item['before']: atomic(p, (folder/name).read_bytes())
            else: p.unlink()
        print('Восстановлено:', target)
        return
    data = target.read_bytes()
    text = data.decode('utf-8')
    if MARKER in text:
        if HOOK not in text or digest(text.replace(HOOK,'',1).encode()) != SOURCE_SHA:
            raise ValueError('Основной файл изменён после GUI-патча; нужен анализ текущей версии.')
        module = target.parent/MODULE
        if not module.exists() or module.read_bytes() != RUNTIME.encode():
            raise ValueError('Файл GUI-модуля отсутствует или изменён.')
        print('Этот GUI-патч уже установлен; повторная установка не требуется.')
        return
    if digest(data) != SOURCE_SHA:
        raise ValueError('Основной файл отличается от присланного city3d_pipeline_v5(4).py. Ничего не изменено; передайте текущий файл для адаптации.')
    for rel, expected in DEPENDENCIES.items():
        p = target.parent/rel
        if not p.exists() or digest(p.read_bytes()) != expected:
            raise ValueError('Модуль отличается от версии патчей 01–03: '+str(p)+'. Передайте его для адаптации.')
    if (target.parent/MODULE).exists(): raise ValueError('Файл '+MODULE+' уже существует без отметки установки; ничего не перезаписываем.')
    anchor = '# END CITY3D_FIXSET_20261016\n'
    if text.count(anchor) != 1: raise ValueError('Не найден однозначный конец подключения патчей 01–03.')
    modified = text.replace(anchor, anchor+HOOK, 1).encode()
    compile(modified, str(target), 'exec'); compile(RUNTIME, MODULE, 'exec')
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    checked = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--self-test', '--target', str(target)],
                             capture_output=True, text=True, env=env, timeout=90)
    results = dict(time=datetime.now(timezone.utc).isoformat(), python=sys.version,
                   target=str(target), source_sha256=SOURCE_SHA, status='verification_failed',
                   qt_stdout=checked.stdout, qt_stderr=checked.stderr)
    report_path = target.parent/'city3d_gui_repair_verification.json'
    if checked.returncode:
        report(report_path, results)
        raise RuntimeError('GUI-тест не прошёл; установка отменена. Отчёт: '+str(report_path)+'\n'+checked.stderr)
    results['gui_test'] = json.loads(checked.stdout.strip().splitlines()[-1])
    results['status'] = 'dry_run_passed'
    report(report_path, results)
    if not args.apply:
        print('Проверки прошли. Код не изменён. Отчёт:', report_path)
        print('Закройте City3D и повторите команду с --apply.')
        return
    args.backups.mkdir(parents=True, exist_ok=True)
    folder = args.backups/('city3d_gui_layout_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    folder.mkdir()
    outputs = {MODULE:RUNTIME.encode(), target.name:modified}
    manifest = dict(target=str(target), files={})
    for name,new in outputs.items():
        p = target.parent/name
        before = p.read_bytes() if p.exists() else None
        if before is not None: shutil.copy2(p, folder/name)
        manifest['files'][name] = dict(before=digest(before) if before is not None else None, after=digest(new))
    report(folder/'manifest.json', manifest)
    changed = []
    try:
        for name,new in outputs.items():
            atomic(target.parent/name,new); changed.append(name)
        for name,new in outputs.items():
            if (target.parent/name).read_bytes() != new: raise RuntimeError('Ошибка проверки записи '+name)
        results['status'] = 'installed'; results['backup'] = str(folder)
        report(folder/'verification.json', results)
        report(report_path, results)
    except BaseException:
        for name in reversed(changed):
            if manifest['files'][name]['before']: atomic(target.parent/name,(folder/name).read_bytes())
            else: (target.parent/name).unlink(missing_ok=True)
        raise
    print('Установлено. Резервная копия:', folder)
    print('Отчёт:', report_path)
    print('Проверены компоновка и вызов обработчиков. Работа БД/GPU/полного конвейера требует проверки в запущенном City3D.')


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        print('ОШИБКА:', exc, file=sys.stderr)
        sys.exit(1)
