"""Versioned semantic evidence with explicit identity and unknown probability mass."""
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from uuid import UUID, uuid4, uuid5, NAMESPACE_URL
import numpy as np
from pyproj import CRS
from .settings import atomic_json

ONTOLOGY_VERSION = 'city3d:1'

@dataclass(frozen=True)
class SemanticClass:
    key: str
    name_ru: str
    name_en: str
    parent: str | None = None
    part_of: str | None = None
    domain: str = 'outdoor'
    geometry: str = 'unknown'
    las_code: int | None = None
    @property
    def uuid(self):
        return str(uuid5(NAMESPACE_URL, 'https://city3d.local/classes/' + self.key))

class Ontology:
    def __init__(self, classes, version=ONTOLOGY_VERSION):
        self.version = version
        self.classes = {c.key:c for c in classes}
        if len(self.classes) != len(classes):
            raise ValueError('Коллизия semantic class key')
        for c in classes:
            for related in (c.parent,c.part_of):
                if related is not None and related not in self.classes:
                    raise ValueError('Неизвестная связь: '+related)
    def diff(self, incoming):
        if incoming.version == self.version and incoming.classes != self.classes:
            raise ValueError('Изменения онтологии требуют новой версии')
        return {'added':sorted(incoming.classes.keys()-self.classes.keys()),
                'removed':sorted(self.classes.keys()-incoming.classes.keys()),
                'changed':[k for k in self.classes.keys() & incoming.classes.keys() if self.classes[k]!=incoming.classes[k]]}
    def to_dict(self):
        from dataclasses import asdict
        return {'version':self.version,'classes':[dict(asdict(c),uuid=c.uuid) for c in self.classes.values()]}

def default_ontology():
    basic=[SemanticClass('state:'+x,ru,x,domain='any') for x,ru in
           [('unknown','Неизвестно'),('unobserved','Не наблюдалось'),('ignored','Игнорируется'),
            ('noise','Шум'),('review','Требует проверки')]]
    names={0:'created',1:'unclassified',2:'ground',3:'vegetation_low',4:'vegetation_medium',5:'vegetation_high',
           6:'building',7:'low_noise',9:'water',10:'rail',11:'road_surface',13:'wire_guard',14:'wire_conductor',
           15:'transmission_tower',16:'wire_connector',17:'bridge_deck',18:'high_noise',19:'overhead_structure',
           20:'ignored_ground',21:'snow',22:'temporal_exclusion'}
    basic += [SemanticClass('las:'+str(code),name,name,las_code=code) for code,name in names.items()]
    basic += [SemanticClass('city:'+x,name,x,part_of=part) for x,name,part in
              [('vegetation','Растительность без высотного подкласса',None),('roof','Крыша','las:6'),
               ('wall','Стена','las:6'),('fence','Ограждение',None),('pole','Столб',None),
               ('pipe','Труба',None),('traffic_sign','Дорожный знак',None),('vehicle','Транспорт',None),
               ('person','Человек',None),('sidewalk','Тротуар',None),('parking','Парковка',None),
               ('terrain','Рельеф',None),('trunk','Ствол дерева',None)]]
    return Ontology(basic)

@dataclass(frozen=True)
class EvidenceBatch:
    asset_version: str
    asset_sha256: str
    support_ids: np.ndarray
    class_keys: tuple
    probabilities: np.ndarray
    available: np.ndarray
    valid: np.ndarray
    unknown_mass: np.ndarray
    crs: str
    schema_version: str = ONTOLOGY_VERSION
    mapping_version: str = 'identity:1'
    artifact_uuid: str = ''
    run_uuid: str = ''
    model_uuid: str = ''
    vertical_crs: str | None = None
    origin: tuple = (0.,0.,0.)
    provenance: str = ''
    value_kind: str = 'probabilities_uncalibrated'
    original_labels: np.ndarray | None = None
    original_label_schema: str | None = None
    def __post_init__(self):
        UUID(self.asset_version)
        if len(self.asset_sha256)!=64 or any(c not in '0123456789abcdef' for c in self.asset_sha256):
            raise ValueError('Нужен input SHA256')
        ids=np.asarray(self.support_ids)
        p=np.asarray(self.probabilities,dtype=np.float64)
        a=np.asarray(self.available,dtype=bool)
        valid=np.asarray(self.valid,dtype=bool)
        unknown=np.asarray(self.unknown_mass,dtype=np.float64)
        n,c=p.shape if p.ndim==2 else (0,0)
        if ids.shape!=(n,) or not np.issubdtype(ids.dtype,np.integer) or np.any(ids<0) or len(np.unique(ids))!=n:
            raise ValueError('Некорректные support IDs')
        if c!=len(self.class_keys) or len(set(self.class_keys))!=c or valid.shape!=(n,) or unknown.shape!=(n,):
            raise ValueError('Размерность evidence')
        if a.shape==(c,):
            a=np.broadcast_to(a,(n,c)).copy()
        if a.shape!=p.shape or np.any(p[~a]!=0) or not np.isfinite(p).all() or np.any(p<0):
            raise ValueError('Недоступный класс не должен содержать probability')
        if not np.isfinite(unknown).all() or np.any(unknown<0) or np.any(unknown>1):
            raise ValueError('Некорректная unknown mass')
        if np.any(abs(p.sum(axis=1)[valid]+unknown[valid]-1)>1e-5):
            raise ValueError('Вероятностная масса должна сохраняться')
        if self.original_labels is not None:
            raw=np.asarray(self.original_labels)
            if raw.shape!=(n,) or raw.dtype.hasobject or not self.original_label_schema:
                raise ValueError('Original labels require matching shape, safe numeric dtype and explicit raw schema')
            raw=np.array(raw,copy=True);raw.setflags(write=False);object.__setattr__(self,'original_labels',raw)
        CRS.from_user_input(self.crs)
        for key,array in [('support_ids',ids),('probabilities',p),('available',a),('valid',valid),('unknown_mass',unknown)]:
            copied=np.array(array,copy=True);copied.setflags(write=False);object.__setattr__(self,key,copied)
        if not self.artifact_uuid:object.__setattr__(self,'artifact_uuid',str(uuid4()))
        if not self.run_uuid:object.__setattr__(self,'run_uuid',str(uuid4()))
    def reordered(self, ids):
        order=np.argsort(self.support_ids)
        requested=np.asarray(ids)
        pos=np.searchsorted(self.support_ids[order],requested)
        if len(requested)!=len(order) or np.any(pos>=len(order)) or not np.array_equal(self.support_ids[order][pos],requested):
            raise ValueError('Наборы support IDs различаются')
        idx=order[pos]
        return replace(self,support_ids=requested,probabilities=self.probabilities[idx],available=self.available[idx],
                       valid=self.valid[idx],unknown_mass=self.unknown_mass[idx],
                       original_labels=self.original_labels[idx] if self.original_labels is not None else None)
    def save(self,path):
        from dataclasses import fields
        path=Path(path)
        metadata={f.name:getattr(self,f.name) for f in fields(self) if not isinstance(getattr(self,f.name),np.ndarray)}
        tmp=path.with_suffix('.tmp.npz');path.parent.mkdir(parents=True,exist_ok=True)
        extra={'original_labels':self.original_labels} if self.original_labels is not None else {}
        np.savez_compressed(tmp,**extra,support_ids=self.support_ids,probabilities=self.probabilities,
                            available=self.available,valid=self.valid,unknown_mass=self.unknown_mass)
        tmp.replace(path)
        metadata['data_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        atomic_json(path.with_suffix('.json'),metadata)
    @classmethod
    def load(cls,path):
        path=Path(path);meta=json.loads(path.with_suffix('.json').read_text())
        if meta.pop('data_sha256')!=hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('Evidence checksum mismatch')
        meta['class_keys']=tuple(meta['class_keys']);meta['origin']=tuple(meta['origin'])
        with np.load(path,allow_pickle=False) as arrays:
            return cls(**meta,**dict(arrays))

@dataclass(frozen=True)
class Mapping:
    source_schema: str
    target_schema: str
    version: str
    # source -> (target or None, relation). Ambiguous/narrower remains unknown.
    rules: dict
    def apply(self,batch):
        if batch.schema_version!=self.source_schema:
            raise ValueError('Mapping schema version mismatch')
        keys=tuple(sorted({target for target,relation in self.rules.values() if target is not None and relation in ('exact','broader')}))
        p=np.zeros((len(batch.support_ids),len(keys)))
        avail=np.zeros_like(p,dtype=bool);unknown=batch.unknown_mass.copy()
        for col,source in enumerate(batch.class_keys):
            target,relation=self.rules.get(source,(None,'unmapped'))
            if target is None or relation not in ('exact','broader'):
                unknown += batch.probabilities[:,col]
            else:
                idx=keys.index(target);p[:,idx]+=batch.probabilities[:,col];avail[:,idx]|=batch.available[:,col]
        return replace(batch,class_keys=keys,probabilities=p,available=avail,unknown_mass=unknown,
                       schema_version=self.target_schema,mapping_version=self.version,artifact_uuid=str(uuid4()))

class SemanticBus:
    def __init__(self):self.batches={}
    def register(self,name,batch):
        if name in self.batches:raise ValueError('Evidence source already registered')
        if self.batches:
            first=next(iter(self.batches.values()))
            if (first.asset_version,first.asset_sha256,first.schema_version,first.vertical_crs)!=(batch.asset_version,batch.asset_sha256,batch.schema_version,batch.vertical_crs):
                raise ValueError('Asset/schema/vertical CRS mismatch')
            if CRS.from_user_input(first.crs)!=CRS.from_user_input(batch.crs):raise ValueError('CRS mismatch')
            batch=batch.reordered(first.support_ids)
        self.batches[name]=batch
    def fuse(self,weights=None):
        if not self.batches:raise ValueError('Нет evidence')
        weights=weights or {}
        batches=list(self.batches.values());first=batches[0]
        keys=tuple(sorted(set().union(*(b.class_keys for b in batches))))
        n=len(first.support_ids);total=np.zeros(n);p=np.zeros((n,len(keys)));a=np.zeros_like(p,bool);u=np.zeros(n)
        winners=[]
        for name,b in self.batches.items():
            weight=float(weights.get(name,1))
            if not np.isfinite(weight) or weight<0:raise ValueError('Fusion weight must be finite/nonnegative')
            w=b.valid.astype(float)*weight;total+=w;u+=b.unknown_mass*w
            for col,key in enumerate(b.class_keys):
                k=keys.index(key);p[:,k]+=b.probabilities[:,col]*w;a[:,k]|=b.available[:,col]&b.valid&(weight>0)
            winners.append(np.array([b.class_keys[k] for k in b.probabilities.argmax(axis=1)]) if b.class_keys else np.full(n,'unknown'))
        valid=total>0
        p[valid]/=total[valid,None];u[valid]/=total[valid];u[~valid]=1
        conflict=np.zeros(n,bool)
        for i in range(1,len(winners)):
            conflict|=(winners[i]!=winners[0])&batches[i].valid&batches[0].valid
        result=replace(first,class_keys=keys,probabilities=p,available=a,valid=valid,unknown_mass=u,
                       artifact_uuid=str(uuid4()),provenance=json.dumps({'inputs':[b.artifact_uuid for b in batches],'weights':weights}),
                       value_kind='weighted_mixture_uncalibrated')
        return result,conflict
