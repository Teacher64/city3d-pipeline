import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
import laspy
import numpy as np
from shapely.geometry import shape, mapping
from .project import atomic_json,sha,to_wgs84


def export_cloud(source, destination, kind, project=None, cancel=lambda: False):
    """Export an explicit source. Never treats raw export as a validated result."""
    source,destination=Path(source),Path(destination)
    if source.resolve()==destination.resolve():raise ValueError('Нельзя перезаписывать исходный файл')
    if destination.exists():raise ValueError('Файл уже существует; выберите новое имя')
    temp=destination.with_name(destination.stem+'.'+uuid.uuid4().hex+'.part'+destination.suffix)
    destination.parent.mkdir(parents=True,exist_ok=True)
    count=0
    try:
        if kind in ('LAS','LAZ'):
            with laspy.open(source) as reader,laspy.open(temp,mode='w',header=reader.header,
                                                       do_compress=kind=='LAZ') as writer:
                for points in reader.chunk_iterator(500_000):
                    if cancel():raise RuntimeError('Отменено')
                    writer.write_points(points);count+=len(points)
        elif kind=='CSV':
            with laspy.open(source) as reader,temp.open('w',encoding='utf-8') as out:
                out.write('x,y,z,classification,source_index,original_index\n')
                for points in reader.chunk_iterator(100_000):
                    if cancel():raise RuntimeError('Отменено')
                    dims=set(points.point_format.dimension_names)
                    for i in range(len(points)):
                        si=int(points.c3_source_index[i]) if 'c3_source_index' in dims else ''
                        oi=int(points.c3_original_index[i]) if 'c3_original_index' in dims else count+i
                        out.write(f'{points.x[i]},{points.y[i]},{points.z[i]},{int(points.classification[i])},{si},{oi}\n')
                    count+=len(points)
        elif kind=='BOUNDARY':
            if not project:raise ValueError('Нет активного района')
            geometry=to_wgs84(shape(project['boundary']),project['crs_wkt'])
            atomic_json(temp,{'type':'FeatureCollection','features':[{'type':'Feature',
                'geometry':mapping(geometry),'properties':{k:project[k] for k in
                ('name','project_id','revision_id','boundary_method','sheets_50k','sheets_status') if k in project}}]})
        else:raise ValueError('Неподдерживаемый формат')
        if not temp.is_file() or not temp.stat().st_size:raise ValueError('Пустой результат')
        if cancel():raise RuntimeError('Отменено')
        temp.replace(destination)
        report=dict(status='exported',validation='not_a_training_quality_claim',format=kind,
                    source=str(source),source_sha256=sha(source),output=str(destination),sha256=sha(destination),
                    point_count=count,project_id=project.get('project_id') if project else None,
                    area_name=project.get('name') if project else None)
        atomic_json(destination.with_suffix(destination.suffix+'.provenance.json'),report)
        return str(destination)
    finally:temp.unlink(missing_ok=True)


def scan_archive(root):
    root=Path(root)
    rows=[]
    if not root.exists():return rows
    for directory,dirs,files in os.walk(root):
        dirs[:]=[d for d in dirs if d not in ('results','tiles','augmented','cache','checkpoints')]
        p=Path(directory)
        if p==root:continue
        if not ('result_snapshot.json' in files or 'project_context.json' in files or
                (p.parent==root and ('_M' in p.name or p.name.startswith('pipeline_')))):continue
        row=dict(id='archive:'+str(uuid.uuid5(uuid.NAMESPACE_URL,str(p.resolve()))),
                 started_at=datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat(),
                 method=p.name.rsplit('_',1)[-1],scanning_type='unknown',crs_epsg=None,
                 n_points=None,duration_sec=None,status='archive_unverified',output_dir=str(p),
                 area_name='не установлено',sheets_50k=[],metrics={},hyperparams={},source_path=None,dataset_root=None)
        errors=[]
        for filename in ('project_context.json','result_snapshot.json','metrics.json'):
            file=p/filename
            if not file.exists():continue
            try:
                payload=json.loads(file.read_text())
                if filename=='project_context.json':
                    row.update(area_name=payload.get('name','не установлено'),
                               project_id=payload.get('project_id'),sheets_50k=payload.get('sheets_50k',[]),
                               source_path=payload.get('path'))
                elif filename=='result_snapshot.json':
                    row['snapshot_status']=payload.get('status')
                    row['status']='snapshot_'+str(payload.get('status','unknown'))
                else:row['metrics']=payload
            except Exception as exc:errors.append(filename+': '+str(exc))
        row['errors']=errors
        rows.append(row)
    return sorted(rows,key=lambda r:r['started_at'],reverse=True)


def install_services(ns):
    from PyQt5.QtCore import QTimer,QThread
    from PyQt5.QtWidgets import QPushButton,QFileDialog,QMessageBox,QComboBox,QHBoxLayout,QLabel,QTableWidgetItem
    base_export=ns['ExportTab']
    class Export(base_export):
        def __init__(self,mw):
            super().__init__(mw)
            row=QHBoxLayout()
            row.addWidget(QLabel('Прямой экспорт облака/границы:'))
            self.direct_format=QComboBox();self.direct_format.addItems(['LAS','LAZ','CSV','BOUNDARY'])
            row.addWidget(self.direct_format)
            b=QPushButton('Экспорт текущего района');b.clicked.connect(self.export_project);row.addWidget(b)
            b=QPushButton('Экспорт выбранного LAS-результата');b.clicked.connect(self.export_result);row.addWidget(b)
            self.layout().insertLayout(0,row)
            self.use_cropped.setToolTip('Старые объектные экспортеры требуют отдельной доработки; прямой экспорт выше использует рабочую границу.')
        def _input_path(self):
            p=getattr(self.main_window,'active_project',None)
            return p['path'] if p else getattr(self.main_window,'pcd_path',None)
        def export_project(self):
            self.direct(self._input_path(),getattr(self.main_window,'active_project',None))
        def export_result(self):
            path,_=QFileDialog.getOpenFileName(self,'Выберите конкретный результат LAS/LAZ',str(ns['TRAINING_OUTPUT_DIR']),'LAS (*.las *.laz)')
            if path:self.direct(path,None)
        def direct(self,path,project):
            if not path:QMessageBox.warning(self,'Экспорт','Нет активного LAS');return
            kind=self.direct_format.currentText()
            suffix={'LAS':'.las','LAZ':'.laz','CSV':'.csv','BOUNDARY':'.geojson'}[kind]
            output,_=QFileDialog.getSaveFileName(self,'Новый файл экспорта',str(ns['WORKSPACE']/('export'+suffix)))
            if not output:return
            if not output.lower().endswith(suffix):output+=suffix
            def task(log,progress):
                result=export_cloud(path,output,kind,project,lambda:QThread.currentThread().isInterruptionRequested())
                log('Экспорт создан: '+result,'SUCCESS');self._last_output=result;return True
            w=ns['WorkerThread'](task);w.log_signal.connect(self.log.log)
            # Avoid the old Open3D LAS preview path.
            w.task_done.connect(lambda ok:self.log.log('Проверка экспорта: '+('успешно' if ok else 'ошибка'),'INFO'))
            self._worker=w;w.start()
    ns['ExportTab']=Export

    base_history=ns['HistoryTab']
    class History(base_history):
        def showEvent(self,event):
            super().showEvent(event)
            self.reload()
        def reload(self):
            super().reload()
            archived=scan_archive(ns['TRAINING_OUTPUT_DIR'])
            self.archive_rows={}
            present={self.table.item(i,8).text() for i in range(self.table.rowCount()) if self.table.item(i,8)}
            self.table.setColumnCount(11)
            self.table.setHorizontalHeaderLabels(['#','Начало','Метод','Тип','CRS','Точек','Длит.','Статус','Папка','Район','Листы 1:50000'])
            m,t,s=self._current_filters()
            for record in archived:
                if record['output_dir'] in present:continue
                if m and record['method']!=m:continue
                if t and record['scanning_type']!=t:continue
                if s and record['status']!=s:continue
                row=self.table.rowCount();self.table.insertRow(row)
                values=[record[k] for k in ('id','started_at','method','scanning_type','crs_epsg','n_points','duration_sec','status','output_dir','area_name')]
                values.append(', '.join(record['sheets_50k']) or 'не установлено')
                for col,value in enumerate(values):self.table.setItem(row,col,QTableWidgetItem(str(value)))
                self.archive_rows[record['id']]=record
            atomic_json(Path(ns['WORKSPACE'])/'archive_history.json',archived)
            self.info.setText(f'История: {self.table.rowCount()} записей БД/диска. archive_unverified — найден каталог, успех не доказан.')
            hdb=getattr(self.main_window,'history_db',None)
            if hdb and hdb.is_available and not (getattr(self,'_archive_worker',None) and self._archive_worker.isRunning()):
                params=dict(ns['DB_PARAMS'])
                def task(log,progress):
                    from .db import sync_archive_records
                    sync_archive_records(archived,params)
                    log('Архив индексирован в БД: '+str(len(archived)),'SUCCESS')
                    return True
                w=ns['WorkerThread'](task)
                w.log_signal.connect(lambda text,level:self.info.setText(text))
                self._archive_worker=w;w.start()
        def _selected_run(self):
            rows=self.table.selectionModel().selectedRows()
            if rows:
                key=self.table.item(rows[0].row(),0).text()
                if key in self.archive_rows:return self.archive_rows[key]
            return super()._selected_run()
    ns['HistoryTab']=History

    # Persist project identity when each new legacy training run is created.
    base_training=ns['TrainingTab']
    class Training(base_training):
        def _make_run_dir(self,*args,**kwargs):
            output=super()._make_run_dir(*args,**kwargs)
            project=getattr(self.main_window,'active_project',None)
            if project:atomic_json(Path(output)/'project_context.json',project)
            return output
    ns['TrainingTab']=Training

    base_pipeline=ns['PipelineComposer']
    class Pipeline(base_pipeline):
        def _run_pipeline(self):
            import copy
            self._project_context=copy.deepcopy(getattr(self.training_tab.main_window,'active_project',None))
            return super()._run_pipeline()
        def _pipeline_task_real(self,emit_log,emit_progress,steps,out_dir,snapshot=None):
            context=getattr(self,'_project_context',None)
            if context:atomic_json(Path(out_dir)/'project_context.json',context)
            return super()._pipeline_task_real(emit_log,emit_progress,steps,out_dir,snapshot)
    ns['PipelineComposer']=Pipeline

    from .references import install_references
    install_references(ns)
