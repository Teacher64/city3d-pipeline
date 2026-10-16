import copy
import html
import json
import shutil
from pathlib import Path
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QThread, QTimer
from PyQt5.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QLabel,QLineEdit,QPushButton,
    QComboBox,QTableWidgetItem,QMessageBox,QDoubleSpinBox,QFileDialog)


def install(ns, level=1):
    from .db import connection_options
    import city3d_tab1.database as db
    original_connect=db.connect
    db.connect=lambda params: original_connect(connection_options(params))
    base_log=ns['LogWidget']
    class SafeLog(base_log):
        requested=pyqtSignal(str,str)
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.requested.connect(self.deliver,Qt.QueuedConnection)
        def log(self,message,level='INFO'):
            self.requested.emit(str(message),str(level))
        @pyqtSlot(str,str)
        def deliver(self,message,level):
            super().log(html.escape(message),level)
    ns['LogWidget']=SafeLog
    if level<2:return
    install_project(ns)
    if level>=3:
        from .services import install_services
        install_services(ns)


def install_project(ns):
    from .dataset import install_dataset
    install_dataset(ns)
    from .project import atomic_json,build_revision
    from .db import register_project
    from shapely.geometry import shape
    state_file=Path(ns['WORKSPACE'])/'city3d_project_state.json'
    draft_file=Path(ns['WORKSPACE'])/'city3d_selection.json'
    base_load=ns['LoadDataTab']
    class ProjectLoad(base_load):
        def __init__(self,mw):
            self.draft=json.loads(draft_file.read_text()) if draft_file.exists() else {}
            super().__init__(mw)
            self.table.setColumnCount(6)
            self.table.setHorizontalHeaderLabels(['В работу / файл','Точек','CRS','Классы','Статус','Вид сканирования'])
            row=QHBoxLayout()
            row.addWidget(QLabel('Район работ:'))
            self.area_name=QLineEdit(self.draft.get('name',''))
            row.addWidget(self.area_name,1)
            save=QPushButton('Сохранить выбор / название')
            save.clicked.connect(self.save_draft);row.addWidget(save)
            select=QPushButton('Отметить все');select.clicked.connect(lambda:self.check_all(True));row.addWidget(select)
            clear=QPushButton('Снять все');clear.clicked.connect(lambda:self.check_all(False));row.addWidget(clear)
            self.layout().insertLayout(0,row)
            self.btn_load.setText('Применить отмеченные LAS к району → все вкладки')
            self.file_path.setReadOnly(True)
            self.file_path.setPlaceholderText('Рабочие файлы выбираются флажками в таблице')
            for record in list(self.records.values()):self.accept_record(record)
            self.table.itemChanged.connect(self.draft_changed)
            self.area_name.textChanged.connect(self.draft_changed)
            self.restore_button=QPushButton('Восстановить последний рабочий район')
            self.restore_button.clicked.connect(self.restore_active)
            self.layout().insertWidget(1,self.restore_button)

        def accept_record(self,record):
            self.table.blockSignals(True)
            old_row=next((i for i in range(self.table.rowCount()) if self.table.item(i,0).text()==record['path']),None)
            checked=self.table.item(old_row,0).checkState() if old_row is not None else None
            super().accept_record(record)
            row=next(i for i in range(self.table.rowCount()) if self.table.item(i,0).text()==record['path'])
            item=self.table.item(row,0)
            item.setFlags(item.flags()|Qt.ItemIsUserCheckable)
            item.setCheckState(checked if checked is not None else (Qt.Checked if record['path'] in self.draft.get('selected',[]) else Qt.Unchecked))
            if self.table.columnCount()>=6:
                combo=QComboBox()
                combo.addItems(['unknown','ALS_manned','ALS_uav','MLS','TLS','handheld','stationary'])
                combo.setCurrentText(self.draft.get('scan_types',{}).get(record['path'],record.get('scan_type','unknown')))
                combo.currentTextChanged.connect(self.draft_changed)
                self.table.setCellWidget(row,5,combo)
            self.table.blockSignals(False)

        def chosen(self):
            result=[]
            for row in range(self.table.rowCount()):
                item=self.table.item(row,0)
                if item.checkState()==Qt.Checked:
                    record=dict(self.records[item.text()])
                    record['scan_type']=self.table.cellWidget(row,5).currentText()
                    result.append(record)
            return result

        def draft_changed(self,*args):
            if hasattr(self,'area_name'):
                self.status.setText('Выбор/название изменены. Нажмите «Применить отмеченные LAS», чтобы обновить рабочий район.')

        def check_all(self,value):
            for row in range(self.table.rowCount()):self.table.item(row,0).setCheckState(Qt.Checked if value else Qt.Unchecked)

        def save_draft(self):
            records=self.chosen()
            atomic_json(draft_file,dict(name=self.area_name.text().strip(),selected=[r['path'] for r in records],
                                       scan_types={r['path']:r['scan_type'] for r in records}))
            self.status.setText('Выбор сохранён. Рабочая редакция меняется только кнопкой «Применить».')

        def load_pcd(self):
            self.build()

        def build(self,polygon=None,ratio=0.05):
            if any(w.isRunning() for w in ns.get('_ACTIVE_WORKERS',[])):
                QMessageBox.warning(self,'Занято','Дождитесь завершения текущих операций.');return
            records=self.chosen();name=self.area_name.text().strip()
            if not records or not name:
                QMessageBox.warning(self,'Район','Введите название района и отметьте LAS/LAZ.');return
            if QMessageBox.question(self,'Система высот',
                'Подтвердите: все выбранные LAS имеют согласованные единицы Z и одну систему высот. '
                'Патч не выполняет вертикальное преобразование. Граница по всем точкам и полная загрузка требуют RAM.',
                QMessageBox.Yes|QMessageBox.No)!=QMessageBox.Yes:return
            self.save_draft()
            previous=getattr(self.main_window,'active_project',None)
            params=dict(ns['DB_PARAMS'])
            def task(job):
                p=build_revision(records,name,Path(ns['WORKSPACE'])/'projects',polygon,ratio,
                    previous['project_id'] if previous and previous['name']==name else None,
                    job.isInterruptionRequested,lambda text:job.progress.emit(25))
                from .db import enrich_project
                p=enrich_project(p,params)
                data=ns['PointCloudLoader'].load(p['path'])
                if job.isInterruptionRequested():raise RuntimeError('Отменено до активации')
                return p,data
            self.start_job(task,self.activate)

        def activate(self,result):
            from city3d_tab1.catalog import inspect_file
            p,data=result
            # Metadata is already computed; avoid rereading the full workset in the GUI thread.
            record=dict(path=p['path'],file_id=p['project_id'],inspection_id=p['revision_id'],
                        crs_wkt=p['crs_wkt'],crs_source='project_revision',classes={})
            super().loaded(p['path'],record,data)
            self.main_window.active_project=p
            types={r['scan_type'] for r in p['sources']}
            self.main_window.scanning_type=next(iter(types)) if len(types)==1 else 'unknown'
            atomic_json(state_file,p)
            self.main_window.refresh_project_banner()
            for i in range(self.main_window.tabs.count()):
                tab=self.main_window.tabs.widget(i)
                if hasattr(tab,'m5_las_list'):
                    tab.m5_las_list.clear()
                    tab.m5_las_list.addItems([r['clipped_path'] for r in p['sources'] if r['selected_points']])
            self.file_path.setText(p['path'])
            self.status.setText(f"Район «{p['name']}»: файлов {len(p['sources'])}, точек {p['point_count']:,}. Все отмеченные LAS включены.")

        def restore_active(self):
            if not state_file.exists():return
            p=json.loads(state_file.read_text())
            from .project import sha
            def task(job):
                if sha(p['path'])!=p['sha256']:raise ValueError('Рабочий LAS изменён, восстановление остановлено')
                return p,ns['PointCloudLoader'].load(p['path'])
            self.start_job(task,self.activate)

        def sync_db(self):
            p=getattr(self.main_window,'active_project',None)
            params=dict(ns['DB_PARAMS']);records=list(self.records.values())
            def task(job):
                from city3d_tab1.database import sync
                n=sync(records,params,job.isInterruptionRequested)
                if p:register_project(p,params)
                return n
            self.start_job(task,lambda n:self.status.setText(f'В БД сохранён реестр: {n}; район: '+(p['name'] if p else 'ещё не применён')))
    ns['LoadDataTab']=ProjectLoad

    class PolygonTab(QWidget):
        def __init__(self,mw):
            super().__init__();self.main_window=mw
            layout=QVBoxLayout(self)
            self.info=QLabel('Граница строится по точкам каждого выбранного LAS; объединение сохраняет отдельные части района.')
            self.info.setWordWrap(True);layout.addWidget(self.info)
            self.ratio=QDoubleSpinBox();self.ratio.setRange(0,1);self.ratio.setValue(.05);self.ratio.setSingleStep(.05)
            self.ratio.setToolTip('Concave hull: 0 — наиболее подробная оболочка, 1 — выпуклая. Это оценка покрытия по точкам, не юридическая граница.')
            layout.addWidget(self.ratio)
            for text,fn in [('Построить по всем отмеченным LAS',self.auto_bbox),('Редактировать границу мышью',self.edit)]:
                b=QPushButton(text);b.clicked.connect(fn);layout.addWidget(b)
            self.log=ns['LogWidget']();layout.addWidget(self.log)
        def auto_bbox(self):self.main_window.tabs.widget(0).build(ratio=self.ratio.value())
        def get_bbox(self):
            p=getattr(self.main_window,'active_project',None)
            if not p:return None
            b=shape(p['boundary']).bounds
            import laspy
            with laspy.open(p['path']) as reader:
                return b[0],b[1],float(reader.header.mins[2]),b[2],b[3],float(reader.header.maxs[2])
        def edit(self):
            p=getattr(self.main_window,'active_project',None)
            if not p:QMessageBox.warning(self,'Граница','Сначала примените подборку на вкладке загрузки.');return
            from .editor import BoundaryEditor
            editor=BoundaryEditor(shape(p['boundary']),self)
            if editor.exec_():self.main_window.tabs.widget(0).build(polygon=editor.geometry)
    ns['BoundingBoxTab']=PolygonTab

    base_main=ns['MainWindow']
    class ProjectWindow(base_main):
        def __init__(self):
            self.active_project=None
            super().__init__()
            self.project_labels=[]
            for i in range(self.tabs.count()):
                label=QLabel('Район не выбран — вкладка «Загрузка»')
                label.setWordWrap(True)
                self.tabs.widget(i).layout().insertWidget(0,label)
                self.project_labels.append(label)
            self.tabs.currentChanged.connect(self.refresh_project_banner)
        def refresh_project_banner(self,*args):
            p=self.active_project
            text=f"Район: {p['name']} | LAS: {len(p['sources'])} | редакция: {p['revision_id'][:8]}" if p else 'Район не выбран — вкладка «Загрузка»'
            for label in self.project_labels:label.setText(text)
    ns['MainWindow']=ProjectWindow

    base_dataset=ns['DatasetPrepareTab']
    class ProjectDataset(base_dataset):
        def __init__(self,mw):
            super().__init__(mw)
            b=QPushButton('Добавить все LAS текущего района (внутри границы)')
            b.clicked.connect(self.add_project);self.layout().insertWidget(0,b)
        def add_project(self):
            p=getattr(self.main_window,'active_project',None)
            if not p:QMessageBox.warning(self,'Датасет','Примените район на вкладке загрузки.');return
            if self.main_window.dataset_preparer is None:self.create_structure()
            preparer=self.main_window.dataset_preparer
            def task(emit_log,emit_progress):
                for i,r in enumerate(p['sources']):
                    if not r['selected_points']:continue
                    dst=preparer.root/'raw'/f"{p['revision_id']}_{i:04d}.las"
                    if not dst.exists():shutil.copy2(r['clipped_path'],dst)
                    if str(dst) not in preparer.metadata['source_files']:preparer.add_scene(dst,copy_raw=False)
                    emit_progress(int(100*(i+1)/len(p['sources'])))
                preparer.metadata.setdefault('project_revisions',{})[p['revision_id']]=p
                preparer.metadata['ontology_version']='city3d:1'
                preparer._save_metadata()
                emit_log('Добавлены исходные LAS внутри границы; метки остаются unreviewed, это не автоматическое создание эталонов.','SUCCESS')
                return True
            w=ns['WorkerThread'](task);w.log_signal.connect(self.log.log);w.task_done.connect(self._on_files_added);self._worker=w;w.start()
    ns['DatasetPrepareTab']=ProjectDataset
