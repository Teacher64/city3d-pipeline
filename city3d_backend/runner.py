"""Single DAG execution engine for GUI/CLI with fail-closed exports and typed contracts."""
from dataclasses import asdict,dataclass,field
import hashlib
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4
from .core import Config,Cancelled,check,process,sha256
from .settings import atomic_json

@dataclass(frozen=True)
class Task:
    id: str
    executor: str
    mode: str='infer'
    dependencies: tuple=()
    inputs: dict=field(default_factory=dict)
    params: dict=field(default_factory=dict)
    optional: bool=False

@dataclass(frozen=True)
class Artifact:
    kind: str
    path: str
    sha256: str
    uuid: str
    @classmethod
    def file(cls,kind,path):return cls(kind,str(Path(path).resolve()),sha256(path),str(uuid4()))
    def verify(self):
        if sha256(self.path)!=self.sha256:raise ValueError('Artifact changed after snapshot: '+self.path)
        if self.kind=='evidence_index':
            for tile in json.loads(Path(self.path).read_text())['tiles']:
                path=Path(tile['path'])
                if sha256(path)!=tile['sha256'] or sha256(path.with_suffix('.json'))!=tile['metadata_sha256']:
                    raise ValueError('Evidence tile changed after snapshot')

@dataclass(frozen=True)
class ResultSnapshot:
    uuid: str
    run_uuid: str
    status: str
    artifacts: tuple
    schema: str='city3d:1'
    qa_status: str='unreviewed'
    spatial: dict=field(default_factory=dict)
    def verify(self):
        if self.status!='success':raise ValueError('Export requires successful upstream snapshot')
        for artifact in self.artifacts:artifact.verify()
    def save(self,path):atomic_json(path,asdict(self))

@dataclass(frozen=True)
class Executor:
    modes: tuple
    inputs: dict
    function: object
    outputs: dict=field(default_factory=dict)

class PipelineRunner:
    def __init__(self,executors=None,journal=None,repository=None):
        self.executors=executors or builtin_executors();self.journal=journal;self.repository=repository
    def preflight(self,tasks):
        ids=[t.id for t in tasks]
        if len(set(ids))!=len(ids):raise ValueError('Duplicate task ID')
        tasks_by_id={t.id:t for t in tasks};order=[];visited=set();active=set();issues={}
        def visit(uid):
            if uid in active:raise ValueError('Pipeline cycle')
            if uid in visited:return
            if uid not in tasks_by_id:raise ValueError('Missing dependency '+uid)
            active.add(uid)
            for dependency in tasks_by_id[uid].dependencies:visit(dependency)
            active.remove(uid);visited.add(uid);order.append(tasks_by_id[uid])
        for uid in ids:visit(uid)
        for t in order:
            reasons=[];executor=self.executors.get(t.executor)
            if executor is None:reasons.append('adapter_missing: '+t.executor)
            elif t.mode not in executor.modes:reasons.append('unsupported mode: '+t.mode)
            else:
                for name,kind in executor.inputs.items():
                    value=t.inputs.get(name)
                    if value is None:reasons.append('missing input '+name)
                    elif isinstance(value,dict):
                        dependency=value.get('task')
                        if dependency not in t.dependencies:reasons.append('input dependency not declared')
                        elif dependency in tasks_by_id:
                            upstream=self.executors.get(tasks_by_id[dependency].executor)
                            if upstream and upstream.outputs:
                                actual=upstream.outputs.get(value.get('output'))
                                if actual!=kind:reasons.append('input type mismatch: '+str(actual)+' -> '+kind)
                    elif kind in ('file','LAS','DEM','vector','evidence') and not Path(value).is_file():reasons.append('file missing: '+str(value))
            if t.executor in ('yolo.detect','yolo.world','m3.raster'):
                if not t.params.get('checkpoint') or not Path(t.params['checkpoint']).is_file():reasons.append('local checkpoint missing')
                if t.executor=='yolo.world':reasons.append('offline text encoder/config not verified')
            if t.executor=='m1.cpu':
                try:Config(**t.params.get('config',{})).validate()
                except Exception as e:reasons.append(str(e))
            if reasons:issues[t.id]=reasons
        return order,issues
    def run(self,tasks,directory,cancel=None,progress=None,spatial=None):
        from . import __version__
        # Freeze all mutable user parameters at the boundary, shared by every UI and CLI path.
        tasks=[Task(**dict(json.loads(json.dumps(asdict(t))),dependencies=tuple(t.dependencies))) for t in tasks]
        directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
        order,issues=self.preflight(tasks)
        if spatial and spatial.get('height_offset') not in (None,0,0.0):
            for task in tasks:
                if task.executor in ('m1.cpu','m4.features'):
                    issues.setdefault(task.id,[]).append('Nonzero vertical offset is not a verified CRS/geoid transform; apply a reviewed transform before this pipeline')
        definition=[asdict(t) for t in tasks]
        hashes={}
        for task in tasks:
            for name,value in task.inputs.items():
                if isinstance(value,str) and Path(value).is_file():hashes[task.id+':'+name]=sha256(value,cancel)
            if task.params.get('checkpoint') and Path(task.params['checkpoint']).is_file():hashes[task.id+':checkpoint']=sha256(task.params['checkpoint'],cancel)
            if task.executor=='m3.train' and task.inputs.get('dataset') and Path(task.inputs['dataset']).is_file():
                from .raster_model import validate_manifest
                dataset=validate_manifest(task.inputs['dataset'])
                for index,entry in enumerate(dataset['tiles']):
                    hashes[task.id+':image:'+str(index)]=sha256(entry['image'],cancel)
                    hashes[task.id+':mask:'+str(index)]=sha256(entry['mask'],cancel)
            if task.params.get('parent') and Path(task.params['parent']).is_file():hashes[task.id+':parent']=sha256(task.params['parent'],cancel)
            if task.params.get('checkpoint') and Path(task.params['checkpoint']).with_suffix('.json').is_file():hashes[task.id+':checkpoint_metadata']=sha256(Path(task.params['checkpoint']).with_suffix('.json'),cancel)
            model=task.params.get('config',{}).get('model')
            if model and Path(model).is_file():hashes[task.id+':model']=sha256(model,cancel)
        cache_key=hashlib.sha256(json.dumps([definition,hashes,spatial or {},__version__],sort_keys=True).encode()).hexdigest()
        manifest_path=directory/'pipeline_manifest.json'
        previous=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if previous and previous.get('cache_key')!=cache_key:raise ValueError('Pipeline input/schema/model changed; choose a new run directory')
        manifest=dict(run_uuid=previous.get('run_uuid',str(uuid4())),status='ready',definition=definition,
                      cache_key=cache_key,directory=str(directory),input_hashes=hashes,code_version=__version__,steps={},spatial=spatial or {},db_registration='pending')
        outputs={};failed=set();all_artifacts=[]
        required_bad={t.id for t in order if t.id in issues and not t.optional}
        if required_bad:
            for t in order:manifest['steps'][t.id]={'status':'blocked','reason':issues.get(t.id,['required preflight failed'])}
            manifest['status']='blocked';self._persist(manifest,manifest_path);return manifest
        manifest['status']='running';self._persist(manifest,manifest_path)
        for index,t in enumerate(order):
            if t.id in issues:
                manifest['steps'][t.id]={'status':'skipped','reason':issues[t.id]};failed.add(t.id);continue
            if any(dep in failed for dep in t.dependencies):
                manifest['steps'][t.id]={'status':'blocked','reason':['upstream failure']};failed.add(t.id);continue
            try:
                check(cancel)
                cached=previous.get('steps',{}).get(t.id,{})
                if cached.get('status')=='success':
                    artifacts={k:Artifact(**v) for k,v in cached['artifacts'].items()}
                    for artifact in artifacts.values():artifact.verify()
                    outputs[t.id]=artifacts;all_artifacts.extend(artifacts.values());manifest['steps'][t.id]=cached
                    self._persist(manifest,manifest_path)
                    if progress:progress(int(100*(index+1)/len(tasks)),t.id+' verified cached result')
                    continue
                resolved={}
                for name,value in t.inputs.items():
                    if isinstance(value,dict):
                        artifact=outputs[value['task']][value['output']];artifact.verify();resolved[name]=artifact.path
                    else:resolved[name]=value
                # Single-task output compatibility: historical m1_classified.* stays at output root.
                out=directory if len(tasks)==1 else directory/t.id
                out.mkdir(parents=True,exist_ok=True)
                ctx={'cancel':cancel,'progress':progress,'directory':out,'run_uuid':manifest['run_uuid'],
                     'spatial':spatial or {},'snapshot':ResultSnapshot(str(uuid4()),manifest['run_uuid'],'success',tuple(all_artifacts),spatial=spatial or {})}
                artifacts=self.executors[t.executor].function(t.mode,resolved,t.params,ctx)
                for artifact in artifacts.values():artifact.verify()
                outputs[t.id]=artifacts;all_artifacts.extend(artifacts.values())
                manifest['steps'][t.id]={'status':'success','artifacts':{k:asdict(v) for k,v in artifacts.items()}}
                if progress:progress(int(100*(index+1)/len(tasks)),f'{t.id} completed')
            except Cancelled as e:
                failed.add(t.id);manifest['steps'][t.id]={'status':'cancelled','error':str(e)}
                for remainder in order[index+1:]:manifest['steps'][remainder.id]={'status':'blocked','reason':['cancelled']}
                manifest['status']='cancelled';self._persist(manifest,manifest_path);return manifest
            except Exception as e:
                failed.add(t.id);manifest['steps'][t.id]={'status':'failed','error':str(e)}
            self._persist(manifest,manifest_path)
        required_failure=any(t.id in failed and not t.optional for t in tasks)
        manifest['status']='failed' if required_failure else ('partial' if failed else 'success')
        snapshot=ResultSnapshot(str(uuid4()),manifest['run_uuid'],manifest['status'],tuple(all_artifacts),spatial=spatial or {})
        snapshot.save(directory/'result_snapshot.json');manifest['snapshot_uuid']=snapshot.uuid
        self._persist(manifest,manifest_path);return manifest
    def _persist(self,manifest,path):
        atomic_json(path,manifest)
        if self.journal:
            self.journal.enqueue(manifest['run_uuid'],'run',manifest)
            if self.repository:
                synced=self.journal.sync(self.repository)
                manifest['db_registration']='synced' if synced else 'pending'
                atomic_json(path,manifest)

def m1(mode,inputs,params,ctx):
    config=Config(**params.get('config',{}));out=process(inputs['asset'],ctx['directory'],config,ctx['progress'],ctx['cancel'])
    artifacts={'classified':Artifact.file('LAS',out),'metrics':Artifact.file('metrics',ctx['directory']/'metrics.json')}
    # Immutable record IDs and categorical evidence are emitted per tile; no dense whole-scene probabilities.
    import laspy,numpy as np
    from uuid import uuid5,NAMESPACE_URL
    from .semantic import EvidenceBatch
    with laspy.open(inputs['asset']) as reader:crs=reader.header.parse_crs()
    if crs is not None:
        manifest=json.loads((ctx['directory']/'run_manifest.json').read_text())
        asset_uuid=str(uuid5(NAMESPACE_URL,'city3d:content:'+manifest['input_sha256']))
        files=[];evidence_dir=ctx['directory']/'semantic_evidence';evidence_dir.mkdir(exist_ok=True)
        for tile in sorted((ctx['directory']/'results').glob('*.npz')):
            check(ctx['cancel'])
            with np.load(tile,allow_pickle=False) as records:
                labels=records['labels'];known=~np.isin(labels,[0,1,8,12]);classes=tuple(int(x) for x in np.unique(labels[known]))
                values=np.zeros((len(labels),len(classes)),np.float64)
                for col,code in enumerate(classes):values[:,col]=labels==code
                batch=EvidenceBatch(asset_uuid,manifest['input_sha256'],records['id'],tuple('las:'+str(c) for c in classes),
                      values,np.ones(len(classes),bool),np.ones(len(labels),bool),(~known).astype(float),crs.to_string(),
                      run_uuid=ctx['run_uuid'],origin=tuple(manifest['local_origin']),
                      provenance=json.dumps({'source':'M1','categorical_only':True,'uncalibrated_score':'c3_score in LAS'}),
                      value_kind='hard_labels_only_uncalibrated',original_labels=records['original'],original_label_schema='ASPRS source LAS classification')
                filename=evidence_dir/tile.name;batch.save(filename)
                files.append({'path':str(filename),'sha256':sha256(filename),'metadata_sha256':sha256(filename.with_suffix('.json')),
                              'support_ids_sha256':hashlib.sha256(np.sort(records['id']).tobytes()).hexdigest()})
        index=ctx['directory']/'semantic_evidence_index.json';atomic_json(index,{'schema':'city3d:1','asset_version':asset_uuid,
                    'input_sha256':manifest['input_sha256'],'evidence_kind':'hard_labels_only_uncalibrated','tiles':files})
        artifacts['evidence']=Artifact.file('evidence_index',index)
    return artifacts

def dem(mode,inputs,params,ctx):
    from .terrain import process_dem
    out=process_dem(inputs['asset'],ctx['directory']/'m4_features.tif',params.get('units',''),ctx['cancel'],ctx['progress'])
    return {'features':Artifact.file('raster_features',out)}

def qa(mode,inputs,params,ctx):
    import laspy,numpy as np
    counts={};header=None
    with laspy.open(inputs['asset']) as r:
        header=r.header
        for chunk in r.chunk_iterator(100000):
            check(ctx['cancel'])
            for code,count in zip(*np.unique(chunk.classification,return_counts=True)):
                counts[str(int(code))]=counts.get(str(int(code)),0)+int(count)
    out=ctx['directory']/'m5_qa.json';atomic_json(out,{'counts':counts,'point_count':header.point_count,'accuracy':None,'qa_status':'unreviewed','class_schema':'ASPRS source format '+str(header.point_format.id)})
    return {'qa':Artifact.file('label_qa',out)}

def vector(mode,inputs,params,ctx):
    from .layers import load_vector
    from shapely.geometry import mapping
    data=load_vector(inputs['asset'],params.get('crs'));out=ctx['directory']/'m6_normalized.geojson'
    atomic_json(out,{'type':'FeatureCollection','features':[{'type':'Feature','id':f['id'],'geometry':mapping(f['normalized']),'properties':f['properties']} for f in data['features']]})
    meta=ctx['directory']/'m6_provenance.json';atomic_json(meta,{'input_sha256':sha256(inputs['asset']),'native_crs':data['crs'],'source_count':len(data['features']),'qa_status':'unreviewed','loss_report':'GeoJSON geographic coordinates; native geometry retained in source and DB import'})
    return {'vector':Artifact.file('vector',out),'provenance':Artifact.file('metadata',meta)}

def fuse(mode,inputs,params,ctx):
    import numpy as np
    from .semantic import EvidenceBatch,SemanticBus
    if not inputs:raise ValueError('Fusion requires explicit canonical evidence inputs')
    paths=list(inputs.values())
    if all(str(p).endswith('.json') for p in paths):
        indices={name:json.loads(Path(path).read_text()) for name,path in inputs.items()}
        tile_maps={}
        for name,index in indices.items():
            mapping={}
            for tile in index['tiles']:
                key=tile.get('support_ids_sha256')
                if key is None:
                    with np.load(tile['path'],allow_pickle=False) as data:key=hashlib.sha256(np.sort(data['support_ids']).tobytes()).hexdigest()
                if key in mapping:raise ValueError('Duplicate tile support')
                mapping[key]=tile
            tile_maps[name]=mapping
        reference=next(iter(tile_maps.values()))
        if any(set(m)!=set(reference) for m in tile_maps.values()):
            raise ValueError('Tile support partitions differ; repartition by stable original IDs before fusion')
        directory=ctx['directory']/'fused_tiles';directory.mkdir(exist_ok=True);files=[];conflicts=0
        for number,key in enumerate(sorted(reference)):
            check(ctx['cancel']);bus=SemanticBus()
            for name,mapping in tile_maps.items():bus.register(name,EvidenceBatch.load(mapping[key]['path']))
            batch,conflict=bus.fuse(params.get('weights'));out=directory/f'{number}.npz';batch.save(out);conflicts+=int(conflict.sum())
            files.append({'path':str(out),'sha256':sha256(out),'metadata_sha256':sha256(out.with_suffix('.json')),'support_ids_sha256':key})
        index=ctx['directory']/'fused_index.json';atomic_json(index,{'schema':batch.schema_version,'asset_version':batch.asset_version,'input_sha256':batch.asset_sha256,'tiles':files,'quality':None})
        report=ctx['directory']/'fusion_review.json';atomic_json(report,{'contradictory_points':conflicts,'quality':None,'rule':'weighted_mixture_uncalibrated; matching immutable support IDs'})
        return {'evidence':Artifact.file('evidence_index',index),'review':Artifact.file('metadata',report)}
    if any(str(p).endswith('.json') for p in paths):raise ValueError('Do not mix tile indices and individual evidence batches')
    bus=SemanticBus()
    for name,path in inputs.items():bus.register(name,EvidenceBatch.load(path))
    batch,conflict=bus.fuse(params.get('weights'));out=ctx['directory']/'fused.npz';batch.save(out)
    report=ctx['directory']/'fusion_review.json';atomic_json(report,{'contradictory_points':int(conflict.sum()),'quality':None,'rule':'weighted_mixture_uncalibrated'})
    return {'evidence':Artifact.file('evidence',out),'review':Artifact.file('metadata',report)}


def export_las(mode,inputs,params,ctx):
    # Only frozen successful upstream artifacts may be copied/exported.
    ctx['snapshot'].verify()
    if not any(a.path==str(Path(inputs['asset']).resolve()) and a.kind=='LAS' for a in ctx['snapshot'].artifacts):
        raise ValueError('Export input is not a classified LAS member of ResultSnapshot')
    import laspy
    source=Path(inputs['asset']);out=ctx['directory']/source.name
    if out==source:raise ValueError('Export must use a separate output directory')
    tmp=out.with_suffix('.tmp'+out.suffix);shutil.copyfile(source,tmp)
    with laspy.open(tmp) as r:
        count=r.header.point_count;crs=r.header.parse_crs()
        if not count:raise ValueError('Export is empty')
    if sha256(tmp)!=sha256(source):raise ValueError('LAS export checksum mismatch')
    tmp.replace(out)
    report=ctx['directory']/'export_profile.json';atomic_json(report,{'snapshot_uuid':ctx['snapshot'].uuid,'kind':'classified point cloud','point_count':count,'crs':str(crs),'sha256':sha256(out),'LOD':None})
    return {'export':Artifact.file('LAS',out),'profile':Artifact.file('export_profile',report)}

def yolo(mode,inputs,params,ctx):
    if params.get('model_id') not in ('yolov8n','yoloworld_s'):raise ValueError('Exact YOLO model ID required')
    if params['model_id']=='yoloworld_s':raise RuntimeError('YOLO-World local text encoder/config not verified')
    from ultralytics import YOLO
    checkpoint=Path(params['checkpoint']).resolve()
    if not checkpoint.is_file():raise ValueError('Local checkpoint missing; automatic downloads disabled')
    model=YOLO(str(checkpoint));results=model.predict(source=inputs['asset'],device='cpu',verbose=False)
    rows=[]
    for r in results:
        if r.boxes is not None:
            for xyxy,score,label in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy(),r.boxes.cls.cpu().numpy()):
                rows.append({'xyxy':xyxy.tolist(),'score':float(score),'raw_class':int(label)})
    out=ctx['directory']/'detections.json';atomic_json(out,{'boxes':rows,'coordinates':'pixel; no automatic 3D column transfer','checkpoint_sha256':sha256(checkpoint),'model_id':params['model_id']})
    return {'detections':Artifact.file('detections',out)}

def builtin_executors():
    return {'m3.train':Executor(('train',),{'dataset':'file'},raster_train,{'model':'raster_model','training':'training_report'}),
            'm3.raster':Executor(('infer',),{'asset':'file'},raster_infer,{'raster':'raster_semantic','mapping':'metadata'}),
            'm1.cpu':Executor(('infer',),{'asset':'LAS'},m1,{'classified':'LAS','metrics':'metrics','evidence':'evidence_index'}),
            'm4.features':Executor(('infer',),{'asset':'DEM'},dem,{'features':'raster_features'}),
            'm5.qa':Executor(('evaluate',),{'asset':'LAS'},qa,{'qa':'label_qa'}),
            'm6.vector':Executor(('infer',),{'asset':'vector'},vector,{'vector':'vector','provenance':'metadata'}),
            'semantic.fuse':Executor(('fuse',),{},fuse),
            'export.las':Executor(('export',),{'asset':'LAS'},export_las),
            'yolo.detect':Executor(('infer',),{'asset':'file'},yolo)}

def raster_train(mode,inputs,params,ctx):
    from .raster_model import train
    target=ctx['directory']/'m3_tiny_unet.npz'
    out=train(inputs['dataset'],target,epochs=params.get('epochs',10),learning_rate=params.get('learning_rate',.001),
              seed=params.get('seed',42),cancel=ctx['cancel'],progress=ctx['progress'],parent=params.get('parent'))
    return {'model':Artifact.file('raster_model',out),'training':Artifact.file('training_report',out.with_suffix('.json'))}

def raster_infer(mode,inputs,params,ctx):
    from .raster_model import infer
    out=infer(inputs['asset'],params['checkpoint'],ctx['directory']/'m3_labels.tif',ctx['cancel'],ctx['progress'])
    return {'raster':Artifact.file('raster_semantic',out),'mapping':Artifact.file('metadata',out.with_suffix('.json'))}

def verify_snapshot_member(path):
    """Legacy exporter bridge: verify a selected file belongs to a frozen result."""
    source=Path(path).resolve()
    for directory in (source.parent,source.parent.parent):
        file=directory/'result_snapshot.json'
        if not file.is_file():continue
        data=json.loads(file.read_text())
        snapshot=ResultSnapshot(data['uuid'],data['run_uuid'],data['status'],tuple(Artifact(**a) for a in data['artifacts']),
                                schema=data.get('schema','city3d:1'),qa_status=data.get('qa_status','unreviewed'),spatial=data.get('spatial',{}))
        snapshot.verify()
        if any(Path(a.path).resolve()==source for a in snapshot.artifacts):return snapshot
    raise ValueError('Выберите файл из успешного ResultSnapshot; исходный/устаревший файл не является подтверждённым результатом')
