"""Subclass hook: original M1, other tabs, models and settings stay intact."""
import json
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QAbstractItemView, QCheckBox, QDialog, QDialogButtonBox,
    QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMessageBox, QProgressBar, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QPlainTextEdit)
from pyproj import CRS

from .catalog import Journal, Cancelled, crs_catalog, discover, inspect_file
from .database import sync
from .custom_crs import CustomCatalog


class Job(QThread):
    item = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    result = pyqtSignal(object)

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function

    def run(self):
        try:
            self.result.emit(self.function(self))
        except Cancelled:
            self.error.emit('Отменено; уже записанные результаты сохранены')
        except Exception as exc:
            self.error.emit(str(exc))


class CRSPicker(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('CRS: EPSG / ESRI / пользовательское определение')
        self.resize(850, 650)
        self.definition = None
        self.assignment = None
        self.custom_catalog = CustomCatalog()
        self.edit_id = None
        layout = QVBoxLayout(self)
        self.search = QLineEdit()
        self.search.setPlaceholderText('Название, код, область: MSK / МСК / Pulkovo / Russia / 28407')
        layout.addWidget(self.search)
        self.russia = QCheckBox('Российские системы и области применения')
        self.russia.setChecked(True)
        layout.addWidget(self.russia)
        self.entries = crs_catalog()
        self.list = QListWidget()
        layout.addWidget(self.list)
        self.list.currentItemChanged.connect(self.select)
        self.definition_edit = QPlainTextEdit()
        self.definition_edit.setPlaceholderText('EPSG:… / ESRI:… либо полное WKT / PROJJSON официальной МСК')
        self.definition_edit.setMaximumHeight(130)
        layout.addWidget(self.definition_edit)
        self.local_name = QLineEdit()
        self.local_name.setPlaceholderText('Название пользовательской МСК: регион, зона, вариант')
        layout.addWidget(self.local_name)
        self.local_source = QLineEdit()
        self.local_source.setPlaceholderText('Источник параметров: документ / организация / дата / ссылка')
        layout.addWidget(self.local_source)
        custom_buttons = QHBoxLayout()
        for title, action in (('Новая МСК', self.new_custom), ('Сохранить / обновить МСК', self.save_custom),
                              ('Удалить выбранную МСК', self.delete_custom)):
            button = QPushButton(title)
            button.clicked.connect(action)
            custom_buttons.addWidget(button)
        layout.addLayout(custom_buttons)
        import_button = QPushButton('Импорт определения .prj / .wkt / .json')
        import_button.clicked.connect(self.import_definition)
        layout.addWidget(import_button)
        layout.addWidget(QLabel('Назначение CRS не меняет координаты и высоты исходного файла.'))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.search.textChanged.connect(self.refresh)
        self.russia.toggled.connect(self.refresh)
        self.refresh()

    def refresh(self):
        self.list.clear()
        query = self.search.text().casefold().split()
        for entry in self.custom_catalog.entries():
            text = f"{entry['name']} — {entry['id']} — версия {entry['revision']} — {entry['source']}"
            if all(word in text.casefold() for word in query):
                item = QListWidgetItem(text)
                item.setData(Qt.UserRole, entry)
                self.list.addItem(item)
        for code, name, area in self.entries:
            text = f'{code} — {name} — {area}'
            folded = text.casefold()
            if self.russia.isChecked() and not any(s in folded for s in ('russia', 'russian', 'msk', 'мск', 'pulkovo', 'gsk-2011')):
                continue
            if all(word in folded for word in query):
                item = QListWidgetItem(text)
                item.setData(Qt.UserRole, code)
                self.list.addItem(item)

    def select(self, item, previous=None):
        if item:
            data = item.data(Qt.UserRole)
            if isinstance(data, dict):
                self.edit_id = data['id']
                self.definition_edit.setPlainText(data['wkt'])
                self.local_name.setText(data['name'])
                self.local_source.setText(data['source'])
            else:
                self.edit_id = None
                self.definition_edit.setPlainText(data)
                self.local_name.clear()
                self.local_source.setText(data)

    def new_custom(self):
        self.edit_id = None
        self.list.clearSelection()
        self.local_name.clear()
        self.local_source.clear()
        self.definition_edit.clear()

    def save_custom(self):
        try:
            entry = self.custom_catalog.save(self.local_name.text(), self.definition_edit.toPlainText(),
                                             self.local_source.text(), self.edit_id)
            self.edit_id = entry['id']
            self.search.clear()
            self.refresh()
            QMessageBox.information(self, 'МСК сохранена',
                f"{entry['name']}, версия {entry['revision']}. Предыдущий справочник сохранён в истории.")
        except Exception as exc:
            QMessageBox.warning(self, 'Справочник МСК', str(exc))

    def delete_custom(self):
        if not self.edit_id:
            QMessageBox.information(self, 'МСК', 'Выберите пользовательскую МСК. EPSG/ESRI не удаляются.')
            return
        if QMessageBox.question(self, 'Удалить МСК',
            'Удалить запись из справочника? Определения в ранее зарегистрированных облаках сохранятся.',
            QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            try:
                self.custom_catalog.delete(self.edit_id)
                self.new_custom()
                self.refresh()
            except Exception as exc:
                QMessageBox.warning(self, 'Справочник МСК', str(exc))

    def import_definition(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Официальное определение CRS', '', 'CRS (*.prj *.wkt *.json);;Все (*)')
        if path:
            try:
                self.definition_edit.setPlainText(Path(path).read_text(encoding='utf-8-sig'))
            except Exception as exc:
                QMessageBox.warning(self, 'CRS', str(exc))

    def validate(self):
        try:
            self.definition = CRS.from_user_input(self.definition_edit.toPlainText()).to_wkt()
            entry = next((x for x in self.custom_catalog.entries() if x['id'] == self.edit_id), None)
            if entry and entry['wkt'] == self.definition:
                self.assignment = dict(entry)
            else:
                self.assignment = dict(name=self.local_name.text(), source=self.local_source.text(),
                                       wkt=self.definition, verification='operator_supplied')
            self.accept()
        except Exception as exc:
            QMessageBox.warning(self, 'Некорректный CRS', str(exc))


def upgrade_load_tab(base, namespace):
    class InventoryLoadTab(base):
        def __init__(self, main_window):
            self.journal = Journal()
            self.records = {}
            self.job = None
            self.active_record = None
            super().__init__(main_window)
            self.btn_load.setText('Загрузить выбранный файл полностью → вкладки 2–6')
            panel = QVBoxLayout()
            row = QHBoxLayout()
            self.catalog_buttons = []
            for title, action, tip in (
                ('Добавить файлы', self.browse, 'Любое количество файлов; содержимое читается по очереди.'),
                ('Добавить папку', self.browse_folder, 'LAS/LAZ и другие поддержанные форматы; вложенные папки по флажку.'),
                ('Записать реестр в БД', self.sync_db, 'Создаёт дополнительные таблицы als.city3d_input_*. Исходный SQL-дамп не выполняется.'),
                ('Экспорт реестра JSON', self.export_inventory, 'Метаданные, UUID, SHA256, CRS и полный состав LAS-классов.')):
                button = QPushButton(title)
                button.setToolTip(tip)
                button.clicked.connect(action)
                row.addWidget(button)
                self.catalog_buttons.append(button)
            self.recursive = QCheckBox('Вложенные папки')
            self.recursive.setChecked(True)
            row.addWidget(self.recursive)
            panel.addLayout(row)
            self.table = QTableWidget(0, 5)
            self.table.setHorizontalHeaderLabels(['Файл', 'Точек', 'CRS / источник', 'Классы ID:число', 'Статус'])
            self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
            self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.table.setMinimumHeight(130)
            self.table.setMaximumHeight(220)
            self.table.setColumnWidth(0, 300)
            self.table.setColumnWidth(2, 240)
            self.table.setColumnWidth(3, 220)
            self.table.itemSelectionChanged.connect(self.selection_changed)
            panel.addWidget(self.table)
            row2 = QHBoxLayout()
            self.status = QLabel('Реестр входных данных; индексация не означает обработку или проверку эталона')
            self.status.setWordWrap(True)
            row2.addWidget(self.status, 1)
            self.progress = QProgressBar()
            self.progress.setMaximumWidth(130)
            row2.addWidget(self.progress)
            cancel = QPushButton('Отмена')
            cancel.clicked.connect(self.cancel_job)
            row2.addWidget(cancel)
            panel.addLayout(row2)
            self.layout().insertLayout(0, panel)
            self.file_path.setToolTip('Один активный файл для действующих вкладок. Реестр выше содержит всю подборку.')
            self.crs_display.setToolTip('CRS выбирается отдельно для каждого файла; неизвестные системы не угадываются.')
            for record in self.journal.records():
                self.accept_record(record)

        def busy(self):
            return self.job is not None and self.job.isRunning()

        def start_job(self, function, result=None):
            if self.busy():
                QMessageBox.information(self, 'Выполняется операция', 'Дождитесь завершения или нажмите «Отмена».')
                return
            job = Job(function, self)
            self.job = job
            job.item.connect(self.accept_record)
            job.error.connect(self.show_error)
            job.progress.connect(self.progress.setValue)
            if result:
                job.result.connect(result)
            workers = namespace.setdefault('_ACTIVE_WORKERS', [])
            workers.append(job)
            def finished():
                if job in workers:
                    workers.remove(job)
                for b in self.catalog_buttons + [self.btn_load]:
                    b.setEnabled(True)
                self.progress.setValue(100)
                job.deleteLater()
                if self.job is job:
                    self.job = None
            job.finished.connect(finished)
            for b in self.catalog_buttons + [self.btn_load]:
                b.setEnabled(False)
            self.progress.setValue(0)
            job.start()

        def cancel_job(self):
            if self.busy():
                self.job.requestInterruption()

        def show_error(self, text):
            self.status.setText(text)
            log = getattr(self.main_window, 'log_widget', None)
            if log:
                log.log(text, 'ERROR')

        def browse(self):
            paths, _ = QFileDialog.getOpenFileNames(self, 'Добавить облака', '',
                'Облака (*.las *.laz *.LAS *.LAZ *.ply *.pcd *.xyz *.e57 *.bin *.txt);;Все (*)')
            if paths:
                self.index_paths(paths)

        def browse_folder(self):
            path = QFileDialog.getExistingDirectory(self, 'Добавить папку облаков')
            if path:
                self.index_paths([path])

        def index_paths(self, paths, override=None, assignment=None):
            recursive = self.recursive.isChecked()
            scan_type = self.current_scanning_type()
            def task(job):
                ok, failed = 0, 0
                for path in discover(paths, recursive):
                    if job.isInterruptionRequested():
                        raise Cancelled()
                    try:
                        record = inspect_file(path, override, scan_type, job.isInterruptionRequested,
                                              job.progress.emit)
                        if assignment:
                            record['crs_assignment'] = dict(assignment)
                        self.journal.save(record)
                        job.item.emit(record)
                        ok += 1
                    except Cancelled:
                        raise
                    except Exception as exc:
                        failed += 1
                        job.error.emit(f'{path}: {exc}')
                return f'Зарегистрировано: {ok}; ошибок: {failed}. Для PostGIS нажмите «Записать реестр в БД».'
            self.status.setText('Чтение файлов и подсчёт всех LAS-классов…')
            self.start_job(task, self.status.setText)

        def accept_record(self, record):
            path = record['path']
            self.records[path] = record
            row = next((i for i in range(self.table.rowCount()) if self.table.item(i, 0).text() == path), None)
            if row is None:
                row = self.table.rowCount()
                self.table.insertRow(row)
            authority = record.get('crs_authority')
            crs_label = ':'.join(authority) if authority else ('WKT' if record.get('crs_wkt') else 'НЕ ОПРЕДЕЛЁН')
            values = [path, str(record['point_count']) if record['point_count'] is not None else '—',
                      crs_label + ' / ' + record['crs_source'],
                      ', '.join(f'{k}:{v}' for k, v in record['classes'].items()),
                      'индексирован; ' + ('контур bbox' if record['footprint_wgs84'] else 'нет WGS84-контура')]
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setToolTip(json.dumps(record, ensure_ascii=False, indent=2))
                self.table.setItem(row, column, item)

        def selected_paths(self):
            return [self.table.item(i.row(), 0).text() for i in self.table.selectionModel().selectedRows()]

        def selection_changed(self):
            paths = self.selected_paths()
            if paths:
                self.file_path.setText(paths[0])
                r = self.records[paths[0]]
                self.crs_display.setText(CRS.from_wkt(r['crs_wkt']).name if r['crs_wkt'] else 'Не определён / конфликт')
                self.crs_source.setText(r['crs_source'])

        def open_crs_dialog(self):
            if self.busy():
                return
            paths = self.selected_paths() or ([self.file_path.text()] if self.file_path.text() else [])
            if not paths:
                QMessageBox.information(self, 'CRS', 'Сначала добавьте и выделите файлы.')
                return
            try:
                picker = CRSPicker(self)
            except Exception as exc:
                QMessageBox.warning(self, 'Справочник МСК', str(exc))
                return
            if picker.exec_() == QDialog.Accepted:
                if QMessageBox.question(self, 'Назначить CRS',
                    f'Назначить выбранную систему для {len(paths)} файлов? Координаты не преобразуются.',
                    QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
                    self.index_paths(paths, picker.definition, picker.assignment)

        def _ask_crs_dialog(self):
            self.open_crs_dialog()

        def sync_db(self):
            params = dict(namespace.get('DB_PARAMS', {}))
            if not params:
                QMessageBox.warning(self, 'БД', 'Настройте соединение в разделе «Настройки».')
                return
            records = list(self.records.values())
            target = f"{params.get('host', 'local')}/{params.get('dbname', params.get('database', ''))}"
            if QMessageBox.question(self, 'Запись в PostGIS',
                    f'База: {target}\nСоздать дополнительные таблицы и записать {len(records)} записей?',
                    QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                return
            self.start_job(lambda job: sync(records, params, job.isInterruptionRequested),
                           lambda n: self.status.setText(f'PostGIS: сохранено {n} записей; повторная запись идемпотентна.'))

        def export_inventory(self):
            path, _ = QFileDialog.getSaveFileName(self, 'Реестр', 'city3d_inputs.json', 'JSON (*.json)')
            if path:
                try:
                    Path(path).write_text(json.dumps(list(self.records.values()), ensure_ascii=False, indent=2), encoding='utf-8')
                except Exception as exc:
                    QMessageBox.warning(self, 'Экспорт реестра', str(exc))

        def load_pcd(self):
            path = self.file_path.text().strip()
            if not Path(path).is_file():
                QMessageBox.warning(self, 'Загрузка', 'Выберите существующий файл.')
                return
            record = self.records.get(str(Path(path).resolve()))
            if record is None:
                self.index_paths([path])
                self.status.setText('Сначала индексируем файл; затем нажмите полную загрузку ещё раз.')
                return
            if record['size'] != Path(path).stat().st_size or record['mtime_ns'] != Path(path).stat().st_mtime_ns:
                QMessageBox.warning(self, 'Файл изменён', 'Повторно добавьте файл, чтобы обновить метаданные.')
                return
            if record.get('crs_wkt') and CRS.from_wkt(record['crs_wkt']).to_epsg() is None:
                QMessageBox.information(self, 'Пользовательская МСК',
                    'CRS сохранён в реестре как WKT. Старые вкладки пока передают CRS через EPSG; '
                    'автоматическое перепроецирование и экспорт этой МСК требуют отдельного патча. '
                    'Облако будет загружено в исходных координатах.')
            def task(job):
                data = namespace['PointCloudLoader'].load(path)
                if job.isInterruptionRequested():
                    raise Cancelled()
                return data
            self.start_job(task, lambda data: self.loaded(path, record, data))

        def loaded(self, path, record, data):
            o3d = namespace['o3d']
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data['points'])
            if data.get('colors') is not None:
                pcd.colors = o3d.utility.Vector3dVector(data['colors'])
            mw = self.main_window
            mw.pcd, mw.pcd_path, mw.pcd_data = pcd, path, data
            for name in ('clean_pcd', 'clean_pcd_path', 'cropped_pcd', 'cropped_pcd_path', 'cropped_data'):
                if hasattr(mw, name):
                    setattr(mw, name, None)
            mw.scanning_type = self.current_scanning_type()
            mw.input_asset_id = record['file_id']
            mw.input_inspection_id = record['inspection_id']
            mw.input_crs_wkt = record['crs_wkt']
            crs = CRS.from_wkt(record['crs_wkt']) if record['crs_wkt'] else None
            self.crs_detector.crs = crs.to_epsg() if crs else None
            self.crs_detector.crs_source = record['crs_source']
            self.crs_detector.crs_description = crs.name if crs else 'Не определён'
            mw.crs_detector = self.crs_detector
            self.active_record = record
            self.crs_display.setText(crs.name if crs else 'Не определён: задайте CRS перед перепроецированием')
            self.crs_source.setText(record['crs_source'])
            self.info_label.setText(f"Полностью загружено точек: {len(data['points']):,}\n"
                                    f"UUID: {record['file_id']}\nКлассы: {record['classes']}")
            self.viewer.set_pointcloud(pcd)
            self.status.setText('Активный файл загружен полностью. Остальная подборка остаётся в реестре.')

    InventoryLoadTab.__name__ = 'LoadDataTab'
    return InventoryLoadTab
