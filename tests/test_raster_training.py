import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import rasterio
from affine import Affine
from city3d_backend.raster_model import train,load,infer,validate_manifest

class RasterModelTests(unittest.TestCase):
    def test_actual_training_reload_inverse_grid_and_finetune(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);tiles=[]
            for i,split in enumerate(('train','val')):
                image=root/f'image{i}.tif';mask=root/f'mask{i}.tif';affine=Affine.translation(500000+i*100,6000000)*Affine.scale(1,-1)
                x=np.zeros((3,16,16),np.float32);x[:,:,8:]=255;y=np.zeros((16,16),np.uint8);y[:,8:]=1
                profile=dict(driver='GTiff',width=16,height=16,crs='EPSG:32636',transform=affine)
                with rasterio.open(image,'w',count=3,dtype='float32',**profile) as r:r.write(x)
                with rasterio.open(mask,'w',count=1,dtype='uint8',**profile) as r:r.write(y,1)
                tiles.append({'image':str(image),'mask':str(mask),'group':str(i),'split':split})
            manifest=root/'training.json';manifest.write_text(json.dumps({'class_keys':['state:unknown','las:6'],'bands':[1,2,3],'labels_provenance':'gold','tiles':tiles}))
            model=train(manifest,root/'model.npz',epochs=3)
            net,meta=load(model);self.assertEqual(len(meta['history']),3);self.assertFalse(meta['pretrained'])
            self.assertTrue(all(np.isfinite(e['loss']) for e in meta['history']))
            out=infer(tiles[1]['image'],model,root/'labels.tif')
            with rasterio.open(out) as r:
                self.assertEqual(r.count,2);self.assertEqual(r.shape,(16,16));self.assertEqual(r.transform,Affine.translation(500100,6000000)*Affine.scale(1,-1))
                self.assertTrue(np.isfinite(r.read()).all())
            child=train(manifest,root/'child.npz',epochs=1,parent=model)
            _,child_meta=load(child);self.assertEqual(len(child_meta['parent_sha256']),64)
            tiles[1]['group']=tiles[0]['group'];manifest.write_text(json.dumps({'class_keys':['state:unknown','las:6'],'bands':[1,2,3],'labels_provenance':'gold','tiles':tiles}))
            with self.assertRaises(ValueError):validate_manifest(manifest)
