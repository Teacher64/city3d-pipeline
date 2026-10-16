from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
import laspy
from pyproj import CRS
from city3d_backend.runner import PipelineRunner,Task
from city3d_backend.core import verify_roundtrip

class RunnerE2ETests(unittest.TestCase):
    def test_shared_cpu_evidence_fusion_export_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'original.las'
            h=laspy.LasHeader(point_format=6,version='1.4');h.scales=[.001]*3;h.offsets=[500000,6000000,100];h.add_crs(CRS.from_epsg(32636))
            las=laspy.LasData(h);x,y=np.meshgrid(np.arange(5)*.2,np.arange(5)*.2)
            las.x=500000+x.ravel();las.y=6000000+y.ravel();las.z=100+x.ravel()*.01;las.classification=np.full(25,2,np.uint8);las.write(source)
            config={'units':'metres','ground_backend':'trusted'}
            tasks=[Task('cpu_a','m1.cpu',inputs={'asset':str(source)},params={'config':config}),Task('cpu_b','m1.cpu',inputs={'asset':str(source)},params={'config':config}),
                   Task('fuse','semantic.fuse','fuse',('cpu_a','cpu_b'),{'a':{'task':'cpu_a','output':'evidence'},'b':{'task':'cpu_b','output':'evidence'}}),
                   Task('export','export.las','export',('cpu_a',),{'asset':{'task':'cpu_a','output':'classified'}})]
            runner=PipelineRunner();out=root/'out';result=runner.run(tasks,out)
            self.assertEqual(result['status'],'success',str(result['steps']))
            verify_roundtrip(source,result['steps']['export']['artifacts']['export']['path'])
            old_ids={key:value['artifacts'] for key,value in result['steps'].items()}
            again=runner.run(tasks,out);self.assertEqual(result['run_uuid'],again['run_uuid']);self.assertEqual(old_ids,{k:v['artifacts'] for k,v in again['steps'].items()})
            report=out/'fuse/fusion_review.json';self.assertEqual(json.loads(report.read_text())['contradictory_points'],0)
