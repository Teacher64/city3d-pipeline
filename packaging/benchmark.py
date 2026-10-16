import json
from pathlib import Path
import statistics
import sys
import numpy as np
import laspy
from pyproj import CRS
from city3d_backend.core import Config,process
root=Path(sys.argv[1] if len(sys.argv)>1 else 'build/benchmark')
root.mkdir(parents=True,exist_ok=True)
p=root/'slope.las'
h=laspy.LasHeader(point_format=6,version='1.4');h.scales=[.001]*3;h.offsets=[500000,6000000,100];h.add_crs(CRS.from_epsg(32636))
x,y=np.meshgrid(np.arange(45)*.3,np.arange(45)*.3)
las=laspy.LasData(h);las.x=500000+x.ravel();las.y=6000000+y.ravel();las.z=100+.03*x.ravel()+.01*y.ravel()
las.classification=np.ones(x.size,np.uint8);las.write(p)
runs=[]
for i in range(3):
 out=process(p,root/f'run{i}',Config(units='metres'))
 perf=json.loads((out.parent/'performance.json').read_text())
 runs.append(perf)
print(json.dumps({'synthetic_points':x.size,'median_seconds':statistics.median(r['stages']['total_s'] for r in runs),
                  'min_seconds':min(r['stages']['total_s'] for r in runs),'max_seconds':max(r['stages']['total_s'] for r in runs),
                  'cache':'OS cache uncontrolled; no cold-cache claim','runs':runs},indent=2))
