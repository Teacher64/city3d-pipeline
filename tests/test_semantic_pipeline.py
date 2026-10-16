import json
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4
import numpy as np
from affine import Affine
from city3d_backend.semantic import EvidenceBatch,Mapping,SemanticBus,default_ontology
from city3d_backend.runner import Task,PipelineRunner,Executor,Artifact,ResultSnapshot
from city3d_backend.layers import LayerManager,load_vector,sample_raster
from city3d_backend.terrain import terrain_features,process_dem
from city3d_backend.database import identifier,LocalJournal,load_connection_file

class SemanticTests(unittest.TestCase):
    def setUp(self):self.asset=str(uuid4())
    def batch(self,ids=(0,1),values=((.6,.4),(.3,.7)),keys=('las:2','las:6'),**kwargs):
        return EvidenceBatch(self.asset,'a'*64,np.array(ids),keys,np.array(values),np.ones(len(keys),bool),np.ones(len(ids),bool),np.zeros(len(ids)), 'EPSG:32636',**kwargs)
    def test_reordered_evidence(self):
        bus=SemanticBus();bus.register('a',self.batch());bus.register('b',self.batch((1,0),((.3,.7),(.6,.4))))
        result,conflict=bus.fuse();np.testing.assert_allclose(result.probabilities,[[.6,.4],[.3,.7]]);self.assertFalse(conflict.any())
    def test_same_length_different_ids_rejected(self):
        bus=SemanticBus();bus.register('a',self.batch())
        with self.assertRaises(ValueError):bus.register('b',self.batch((10,11)))
    def test_asset_version_rejected(self):
        bus=SemanticBus();bus.register('a',self.batch());self.asset=str(uuid4())
        with self.assertRaises(ValueError):bus.register('b',self.batch())
    def test_coarse_to_fine_keeps_unknown(self):
        batch=self.batch(keys=('coarse_vegetation','building'))
        mapping=Mapping('city3d:1','city3d:2','m:2',{'coarse_vegetation':('las:3','narrower'),'building':('las:6','exact')})
        result=mapping.apply(batch);np.testing.assert_allclose(result.unknown_mass,[.6,.3]);np.testing.assert_allclose(result.probabilities[:,0],[.4,.7])
    def test_aggregation_preserves_mass(self):
        batch=self.batch(keys=('low','high'));result=Mapping('city3d:1','target:1','m:1',{'low':('vegetation','broader'),'high':('vegetation','broader')}).apply(batch)
        np.testing.assert_allclose(result.probabilities,[[1],[1]])
    def test_mapping_mismatch(self):
        with self.assertRaises(ValueError):Mapping('wrong','target','m',{}).apply(self.batch())
    def test_missing_class_is_unavailable(self):
        a=EvidenceBatch(self.asset,'a'*64,np.array([0]),('las:2',),np.array([[.4]]),np.array([True]),np.array([True]),np.array([.6]),'EPSG:32636')
        bus=SemanticBus();bus.register('a',a);result,_=bus.fuse();self.assertEqual(result.class_keys,('las:2',));self.assertAlmostEqual(result.unknown_mass[0],.6)
    def test_contradiction_review(self):
        bus=SemanticBus();bus.register('a',self.batch());bus.register('b',self.batch(values=((.1,.9),(.8,.2))))
        _,conflict=bus.fuse();self.assertTrue(conflict.all())
    def test_immutable_and_roundtrip(self):
        b=self.batch()
        with self.assertRaises(ValueError):b.probabilities[0,0]=0
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'evidence.npz';b.save(p);loaded=EvidenceBatch.load(p);self.assertEqual(loaded.asset_version,b.asset_version)
            p.write_bytes(p.read_bytes()+b'corrupt')
            with self.assertRaises(ValueError):EvidenceBatch.load(p)
    def test_original_labels_survive_mapping_and_save(self):
        from dataclasses import replace
        batch=replace(self.batch(),original_labels=np.array([50,70]),original_label_schema='SemanticKITTI raw')
        mapped=Mapping('city3d:1','city3d:2','m2',{'las:2':('ground','exact'),'las:6':('building','exact')}).apply(batch)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'mapped.npz';mapped.save(path);loaded=EvidenceBatch.load(path)
            self.assertEqual(loaded.original_labels.tolist(),[50,70]);self.assertEqual(loaded.original_label_schema,'SemanticKITTI raw')
    def test_vegetation_subclasses_and_parts(self):
        ontology=default_ontology();self.assertEqual(ontology.classes['city:roof'].part_of,'las:6')
        self.assertEqual(len({ontology.classes[x].uuid for x in ('las:3','las:4','las:5')}),3)

class PipelineTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_cycle(self):
        tasks=[Task('a','x',dependencies=('b',)),Task('b','x',dependencies=('a',))]
        with self.assertRaises(ValueError):PipelineRunner().preflight(tasks)
    def test_missing_adapter_preflight(self):
        result=PipelineRunner().run([Task('neural','unverified.neural')],self.root/'out');self.assertEqual(result['status'],'blocked')
    def test_failure_blocks_dependent_export(self):
        def fail(*args):raise RuntimeError('failed upstream')
        executed=[]
        def consume(*args):executed.append(True);return {}
        executors={'bad':Executor(('infer',),{},fail),'export':Executor(('export',),{},consume)}
        result=PipelineRunner(executors).run([Task('a','bad'),Task('b','export','export',dependencies=('a',))],self.root/'out')
        self.assertEqual(result['steps']['b']['status'],'blocked');self.assertFalse(executed)
    def test_type_mismatch_preflight(self):
        p=self.root/'a.geojson';p.write_text('{}')
        tasks=[Task('vector','m6.vector',inputs={'asset':str(p)}),Task('export','export.las','export',('vector',),{'asset':{'task':'vector','output':'vector'}})]
        _,issues=PipelineRunner().preflight(tasks);self.assertIn('export',issues)
    def test_snapshot_hash_gate(self):
        p=self.root/'x';p.write_text('one');a=Artifact.file('LAS',p);snap=ResultSnapshot(str(uuid4()),str(uuid4()),'success',(a,));snap.verify();p.write_text('two')
        with self.assertRaises(ValueError):snap.verify()
    def test_optional_unavailable_is_partial(self):
        def valid(*args):return {}
        runner=PipelineRunner({'okay':Executor(('infer',),{},valid)})
        result=runner.run([Task('a','missing',optional=True),Task('b','okay')],self.root/'out');self.assertEqual(result['status'],'partial')
    def test_offline_journal_is_idempotent(self):
        journal=LocalJournal(self.root/'outbox.sqlite');uid=str(uuid4());journal.enqueue(uid,'run',{'run_uuid':uid});journal.enqueue(uid,'run',{'run_uuid':uid})
        class Repo:
            def record_run(self,r):pass
        self.assertEqual(journal.sync(Repo()),1);self.assertEqual(journal.sync(Repo()),0)

class GeoTests(unittest.TestCase):
    def test_rotated_affine_slope(self):
        affine=Affine.rotation(25)*Affine.scale(2,-3);row,col=np.mgrid[:10,:12]
        x=affine.a*col+affine.b*row;y=affine.d*col+affine.e*row
        result=terrain_features(.1*x+.2*y,affine)
        np.testing.assert_allclose(result[0],np.degrees(np.arctan(np.hypot(.1,.2))),atol=1e-5)
        np.testing.assert_allclose(result[3],0,atol=1e-5)
    def test_nodata_and_raster_sampler(self):
        import rasterio
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'dem.tif';affine=Affine.translation(500000,6000000)*Affine.scale(1,-1)
            data=np.arange(100,dtype=np.float32).reshape(10,10);data[3,3]=-9999
            with rasterio.open(p,'w',driver='GTiff',width=10,height=10,count=1,dtype='float32',crs='EPSG:32636',transform=affine,nodata=-9999) as dst:dst.write(data,1)
            values,valid=sample_raster(p,[[500003.5,5999996.5],[500100.,6000100.]],'EPSG:32636');self.assertFalse(valid.any());self.assertTrue(np.isnan(values).all())
            out=process_dem(p,Path(d)/'features.tif')
            with rasterio.open(out) as src:self.assertEqual(src.count,4);self.assertTrue(np.isnan(src.read(1)[3,3]))
    def test_layer_state(self):
        with tempfile.TemporaryDirectory() as d:
            manager=LayerManager();layer=manager.add('test.geojson','vector','EPSG:4326');manager.update(layer.uuid,opacity=.4,visible=False)
            file=Path(d)/'layers.json';manager.save(file);loaded=LayerManager.load(file);self.assertEqual(loaded.layers[0].opacity,.4);self.assertFalse(loaded.layers[0].visible)
    def test_vector_native_z_and_reprojection(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'v.geojson';p.write_text(json.dumps({'type':'FeatureCollection','features':[{'type':'Feature','id':'42','geometry':{'type':'Point','coordinates':[500000,6000000,25]},'properties':{'class':'building'}}]}))
            v=load_vector(p,'EPSG:32636');self.assertEqual(v['features'][0]['native'].z,25);self.assertEqual(v['features'][0]['normalized'].z,25)
            self.assertTrue(v['features'][0]['normalized'].x<180)
    def test_sql_identifier_rejects_injection(self):
        with self.assertRaises(ValueError):identifier('history; DROP TABLE things')
    def test_pgadmin_connection(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.json';p.write_text(json.dumps({'Servers':{'1':{'Host':'localhost','Port':5432,'Username':'app','MaintenanceDB':'postgres','UseSSHTunnel':False}}}))
            cfg=load_connection_file(p);self.assertEqual(cfg['dbname'],'postgres');self.assertNotIn('password',cfg)

if __name__=='__main__':unittest.main()
