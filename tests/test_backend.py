import json
import tempfile
import unittest
from pathlib import Path
import laspy
import numpy as np
from pyproj import CRS
from city3d_backend.core import Config, Cancelled, idw, process, verify_roundtrip, features, classify
from city3d_backend.cli import train, evaluate
from city3d_backend.settings import save_settings, read_settings, resolve_paths

class BackendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()
    def cloud(self, labels=True, extension='las', count=225, fmt=7):
        h = laspy.LasHeader(point_format=fmt, version='1.4')
        h.scales = [.001]*3
        h.offsets = [500000,6000000,100]
        h.add_crs(CRS.from_epsg(32636))
        h.add_extra_dim(laspy.ExtraBytesParams(name='custom', type='i4'))
        from laspy.vlrs.vlrlist import VLRList
        h.evlrs = VLRList([laspy.VLR(user_id='test', record_id=99, record_data=b'preserve EVLR')])
        cloud = laspy.LasData(h)
        xy = np.arange(count)
        cloud.x = 500000 + (xy % 15)*.3
        cloud.y = 6000000 + (xy // 15)*.3
        cloud.z = 100 + ((xy % 15)*.01)
        cloud.classification = np.full(count,2 if labels else 1,np.uint8)
        cloud.intensity = xy.astype(np.uint16)
        cloud.gps_time = xy*.01
        cloud.return_number = np.ones(count, np.uint8)
        cloud.number_of_returns = np.ones(count, np.uint8)
        cloud.synthetic = xy%2
        cloud.withheld = xy%3 == 0
        cloud.overlap = xy%5 == 0
        cloud.custom = xy.astype(np.int32)
        if fmt == 7:
            cloud.red = xy.astype(np.uint16)*10
            cloud.green = xy.astype(np.uint16)*20
            cloud.blue = xy.astype(np.uint16)*30
        path = self.root / ('input.'+extension)
        cloud.write(path)
        return path
    def config(self, **kwargs):
        return Config(units='metres', ground_backend='trusted', **kwargs)
    def test_idw_single_and_distance(self):
        z, valid = idw(np.array([[0.,0.],[10.,0.]]),np.array([[0.,0.]]),np.array([5.]),2.)
        self.assertEqual(z[0],5)
        self.assertFalse(valid[1]); self.assertTrue(np.isnan(z[1]))
    def test_no_ground(self):
        p=self.cloud(labels=False)
        out=process(p,self.root/'out',self.config())
        result=laspy.read(out)
        self.assertTrue(np.isnan(result.c3_hag).all())
        self.assertTrue((result.classification==1).all())
    def test_roundtrip_and_resume(self):
        p=self.cloud()
        out=process(p,self.root/'out',self.config())
        verify_roundtrip(p,out)
        self.assertEqual(len(laspy.read(out).points),225)
        self.assertEqual(json.loads((out.parent/'run_manifest.json').read_text())['status'],'success')
        process(p,self.root/'out',self.config())
        self.assertTrue(json.loads((out.parent/'performance.json').read_text())['resumed'])
    def test_laz_roundtrip(self):
        p=self.cloud(extension='laz')
        out=process(p,self.root/'out',self.config())
        verify_roundtrip(p,out)
    def test_smrf_actual(self):
        p=self.cloud(labels=False)
        out=process(p,self.root/'out',Config(units='metres'))
        self.assertTrue((laspy.read(out).classification==2).any())
    def test_csf_actual(self):
        p=self.cloud(labels=False)
        out=process(p,self.root/'out',Config(units='metres',ground_backend='csf'))
        verify_roundtrip(p,out)
    def test_empty(self):
        p=self.cloud(count=0)
        with self.assertRaisesRegex(ValueError,'Пустое'):
            process(p,self.root/'out',self.config())
    def test_cancel_no_final(self):
        p=self.cloud()
        calls=[0]
        def stop():
            calls[0]+=1
            return calls[0]>=5
        with self.assertRaises(Cancelled):
            process(p,self.root/'out',self.config(),cancel=stop)
        self.assertFalse((self.root/'out/m1_classified.las').exists())
    def test_budget(self):
        p=self.cloud()
        with self.assertRaises(MemoryError):
            process(p,self.root/'out',self.config(max_tile_points=40))
        self.assertEqual(json.loads((self.root/'out/run_manifest.json').read_text())['status'],'failed')
    def test_units(self):
        p=self.cloud()
        with self.assertRaisesRegex(ValueError,'единицы'):
            process(p,self.root/'out',Config(ground_backend='trusted'))
    def test_high_unknown_not_building_and_locked_classes(self):
        feat=np.zeros((4,17),np.float32); feat[:,0]=[10,15,-20,8]
        c=self.config()
        labels,_,_,_=classify(feat,np.array([1,1,1,17]),np.zeros(4,bool),c)
        self.assertEqual(labels.tolist(),[1,1,1,17])
    def test_tile_boundary_original_order(self):
        p=self.cloud()
        out=process(p,self.root/'out',self.config(tile_size=2,halo=5))
        verify_roundtrip(p,out)
        self.assertGreater(len(json.loads((out.parent/'run_manifest.json').read_text())['completed_tiles']),1)
    def test_float64_precision(self):
        p=self.cloud()
        out=process(p,self.root/'out',self.config())
        self.assertEqual(laspy.read(p).X.tolist(),laspy.read(out).X.tolist())
        self.assertEqual(laspy.read(p).Y.tolist(),laspy.read(out).Y.tolist())
    def test_train_holdout(self):
        rng=np.random.default_rng(42)
        x=rng.normal(size=(300,17)).astype(np.float32)
        y=np.where(x[:,0]>0,6,5)
        groups=np.repeat(np.arange(6),50)
        m=train(x,y,groups,self.config(),self.root/'model.joblib')
        self.assertGreater(m['macro_f1'],.85)
        self.assertTrue((self.root/'model.metrics.json').exists())
    def test_settings_backup_preserves_unknowns(self):
        file=self.root/'settings.json'
        old={'paths':{'workspace':'/tmp/custom','slices_dir':'/tmp/slices'},'unknown':42,'database':{'password':'do-not-print'}}
        save_settings(old,file);save_settings(old,file)
        self.assertTrue(file.with_suffix('.json.bak').exists())
        self.assertEqual(read_settings(file)['unknown'],42)
        self.assertEqual(resolve_paths(old,self.root)['slices_dir'],Path('/tmp/slices'))

if __name__=='__main__':
    unittest.main()

class ReliabilityTests(unittest.TestCase):
    setUp = BackendTests.setUp
    tearDown = BackendTests.tearDown
    cloud = BackendTests.cloud
    config = BackendTests.config
    def test_cancel_resume_incomplete_tile(self):
        path=self.cloud()
        calls=[0]
        def stop():
            calls[0]+=1
            return calls[0]==12
        try:
            process(path,self.root/'cancel_resume',self.config(tile_size=2,halo=5),cancel=stop)
        except Cancelled:
            pass
        out=process(path,self.root/'cancel_resume',self.config(tile_size=2,halo=5))
        verify_roundtrip(path,out)
    def test_laz_output(self):
        p=self.cloud()
        out=process(p,self.root/'out_laz',self.config(output_format='laz'))
        self.assertEqual(out.suffix,'.laz')
        verify_roundtrip(p,out)
    def test_waveform_rejected(self):
        header=laspy.LasHeader(point_format=9,version='1.4')
        las=laspy.LasData(header)
        las.x=np.array([1.]);las.y=np.array([1.]);las.z=np.array([1.])
        p=self.root/'wave.las';las.write(p)
        with self.assertRaisesRegex(ValueError,'Waveform'):
            process(p,self.root/'waveout',self.config())
    def test_write_failure_not_success(self):
        from unittest.mock import patch
        p=self.cloud()
        # Writer opens only after features; input reader still works.
        original=laspy.open
        def fail_write(*args,**kwargs):
            if kwargs.get('mode')=='w':
                raise OSError('synthetic disk full')
            return original(*args,**kwargs)
        with patch('city3d_backend.core.laspy.open',side_effect=fail_write):
            with self.assertRaisesRegex(OSError,'disk full'):
                process(p,self.root/'failed_write',self.config())
        self.assertFalse((self.root/'failed_write/m1_classified.las').exists())
        self.assertEqual(json.loads((self.root/'failed_write/run_manifest.json').read_text())['status'],'failed')

class ModelIntegrationTests(unittest.TestCase):
    setUp = BackendTests.setUp
    tearDown = BackendTests.tearDown
    cloud = BackendTests.cloud
    config = BackendTests.config
    def test_trained_model_inference(self):
        p=self.cloud()
        rng=np.random.default_rng(42)
        x=rng.normal(size=(300,17)).astype(np.float32)
        labels=np.where(x[:,0]>0,6,5)
        target=self.root/'outdoor.joblib'
        config=self.config(preserve_classes=False)
        train(x,labels,np.repeat(np.arange(6),50),config,target)
        config.model=str(target)
        out=process(p,self.root/'inference',config)
        self.assertTrue(set(np.unique(laspy.read(out).classification)).issubset({1,5,6}))
