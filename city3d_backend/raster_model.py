"""Trainable CPU U-Net baseline; no fake pretrained checkpoint or remote code."""
import json
from pathlib import Path
from uuid import uuid4
import numpy as np
from .core import check,sha256
from .settings import atomic_json

ARCHITECTURE='city3d-tiny-unet-v1'

def make_model(bands,classes):
    import torch
    from torch import nn
    class TinyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=nn.Sequential(nn.Conv2d(bands,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,16,3,padding=1),nn.ReLU())
            self.bottleneck=nn.Sequential(nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,32,3,padding=1),nn.ReLU())
            self.decoder=nn.Sequential(nn.Conv2d(48,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,classes,1))
        def forward(self,x):
            encoded=self.encoder(x)
            bottom=self.bottleneck(torch.nn.functional.max_pool2d(encoded,2,ceil_mode=True))
            decoded=torch.nn.functional.interpolate(bottom,size=encoded.shape[-2:],mode='bilinear',align_corners=False)
            return self.decoder(torch.cat([encoded,decoded],dim=1))
    return TinyUNet()

def read_pair(image,mask,bands):
    import rasterio
    with rasterio.open(image) as r,rasterio.open(mask) as gt:
        if r.crs is None or r.crs!=gt.crs or r.transform!=gt.transform or r.width!=gt.width or r.height!=gt.height:
            raise ValueError('Image/mask CRS, affine and pixel grid must match exactly')
        if r.width*r.height>4_000_000:raise MemoryError('Training tile >4M pixels: prepare smaller spatial tiles')
        x=r.read(bands,masked=True).astype(np.float32).filled(np.nan)
        labels=gt.read(1,masked=True).astype(np.float64).filled(np.nan)
        return x,labels,{'crs':str(r.crs),'transform':tuple(r.transform),'shape':[r.height,r.width]}

def validate_manifest(path):
    import rasterio
    path=Path(path);manifest=json.loads(path.read_text())
    keys=manifest['class_keys'];bands=manifest['bands'];entries=manifest['tiles']
    if len(keys)<2 or len(set(keys))!=len(keys) or not bands:raise ValueError('Need >=2 unique canonical classes and explicit bands')
    if manifest.get('labels_provenance')!='gold':raise ValueError('Training requires explicit reviewed gold labels; pseudo labels are separate')
    groups={};bounds=[];resolved=[];crs=None
    for e in entries:
        split=e['split'];group=str(e['group'])
        if split not in ('train','val','test'):raise ValueError('Unknown split')
        if group in groups and groups[group]!=split:raise ValueError('Spatial group crosses train/val/test')
        groups[group]=split
        image=Path(e['image']);mask=Path(e['mask'])
        if not image.is_absolute():image=path.parent/image
        if not mask.is_absolute():mask=path.parent/mask
        with rasterio.open(image) as r:
            if crs is not None and crs!=r.crs:raise ValueError('Raster training territories require the same declared CRS')
            crs=r.crs
            box=np.array(r.bounds)
            for previous,previous_split in bounds:
                overlaps=box[0]<previous[2] and box[2]>previous[0] and box[1]<previous[3] and box[3]>previous[1]
                if overlaps and split!=previous_split:raise ValueError('Overlapping imagery leaks across spatial splits')
            bounds.append((box,split))
        resolved.append(dict(e,image=str(image),mask=str(mask)))
    if not any(e['split']=='train' for e in resolved) or not any(e['split']=='val' for e in resolved):raise ValueError('Need independent train and val territories')
    manifest['tiles']=resolved;return manifest

def train(manifest_path,target,epochs=10,learning_rate=.001,seed=42,cancel=None,progress=None,parent=None):
    import torch
    from .cli import evaluate
    target=Path(target)
    if target.exists():raise ValueError('Checkpoint exists: use a new versioned filename')
    if epochs<1 or epochs>10000:raise ValueError('Epoch bounds')
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
    manifest=validate_manifest(manifest_path);keys=manifest['class_keys'];bands=manifest['bands']
    mean=np.asarray(manifest.get('mean',[0]*len(bands)),np.float32)[:,None,None]
    std=np.asarray(manifest.get('std',[255]*len(bands)),np.float32)[:,None,None]
    if mean.shape[0]!=len(bands) or std.shape[0]!=len(bands) or np.any(std<=0):raise ValueError('Band normalization mismatch')
    model=make_model(len(bands),len(keys));parent_hash=None
    if parent:
        previous,meta=load(parent)
        if meta['class_keys']!=keys or meta['bands']!=bands or meta['mean']!=mean.ravel().tolist() or meta['std']!=std.ravel().tolist():raise ValueError('Fine-tuning requires compatible architecture/class map/normalization')
        model.load_state_dict(previous.state_dict());parent_hash=sha256(parent)
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    loss_fn=torch.nn.CrossEntropyLoss(ignore_index=-100)
    best=-1;best_state=None;history=[];hashes=[]
    # Stream territory/patch reads per epoch; no entire dataset tensors retained.
    for entry in manifest['tiles']:
        hashes.append({'image_sha256':sha256(entry['image']),'mask_sha256':sha256(entry['mask']),'group':entry['group'],'split':entry['split']})
    for epoch in range(epochs):
        check(cancel);loss_total=0;steps=0;model.train()
        for entry in manifest['tiles']:
            if entry['split']!='train':continue
            x,y,_=read_pair(entry['image'],entry['mask'],bands)
            valid=np.isfinite(x).all(axis=0)&np.isfinite(y)&(y>=0)&(y<len(keys))&(y==np.floor(y))
            x=np.nan_to_num((x-mean)/std);labels=np.where(valid,y,-100).astype(np.int64)
            for row in range(0,x.shape[1],128):
                for col in range(0,x.shape[2],128):
                    check(cancel)
                    patch=x[:,row:row+128,col:col+128];gt=labels[row:row+128,col:col+128]
                    if min(gt.shape)<2 or np.all(gt==-100):continue
                    optimizer.zero_grad();logits=model(torch.from_numpy(patch[None]))
                    loss=loss_fn(logits,torch.from_numpy(gt[None]));loss.backward();optimizer.step();loss_total+=float(loss.detach());steps+=1
        if not steps:raise ValueError('No valid supervised training pixels')
        model.eval();truth=[];prediction=[]
        with torch.no_grad():
            for entry in manifest['tiles']:
                if entry['split']!='val':continue
                check(cancel);x,y,_=read_pair(entry['image'],entry['mask'],bands)
                valid=np.isfinite(x).all(axis=0)&np.isfinite(y)&(y>=0)&(y<len(keys))&(y==np.floor(y))
                x=np.nan_to_num((x-mean)/std)
                for row in range(0,x.shape[1],128):
                    for col in range(0,x.shape[2],128):
                        region=valid[row:row+128,col:col+128]
                        if not region.any():continue
                        pred=model(torch.from_numpy(x[:,row:row+128,col:col+128][None])).argmax(1).numpy()[0]
                        truth.append(y[row:row+128,col:col+128][region].astype(int));prediction.append(pred[region])
        if not truth:raise ValueError('No valid independent validation pixels')
        metrics=evaluate(np.concatenate(truth),np.concatenate(prediction));score=metrics['macro_f1']
        history.append({'epoch':epoch+1,'loss':loss_total/steps,'validation':metrics})
        if score>best:best=score;best_state={k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()}
        if progress:progress(int(100*(epoch+1)/epochs),f'M3 epoch {epoch+1}/{epochs}, loss={loss_total/steps:.5f}')
    for entry,expected in zip(manifest['tiles'],hashes):
        if sha256(entry['image'])!=expected['image_sha256'] or sha256(entry['mask'])!=expected['mask_sha256']:
            raise ValueError('Training sources changed during training')
    check(cancel);target.parent.mkdir(parents=True,exist_ok=True)
    if target.suffix!='.npz':raise ValueError('Safe baseline checkpoint format is .npz')
    temp=target.with_suffix('.tmp.npz');np.savez_compressed(temp,**best_state);temp.replace(target)
    meta=dict(architecture=ARCHITECTURE,bands=bands,class_keys=keys,mean=mean.ravel().tolist(),std=std.ravel().tolist(),
              seed=seed,epochs=epochs,learning_rate=learning_rate,history=history,validation_macro_f1=best,
              input_hashes=hashes,training_run_uuid=str(uuid4()),parent_sha256=parent_hash,status='candidate',
              checkpoint_sha256=sha256(target),labels_provenance='gold',pretrained=False)
    atomic_json(target.with_suffix('.json'),meta);return target

def load(path):
    import torch
    path=Path(path);meta=json.loads(path.with_suffix('.json').read_text())
    if meta['architecture']!=ARCHITECTURE or sha256(path)!=meta['checkpoint_sha256']:raise ValueError('Model architecture/checksum mismatch')
    model=make_model(len(meta['bands']),len(meta['class_keys']))
    with np.load(path,allow_pickle=False) as state:
        model.load_state_dict({k:torch.from_numpy(state[k].copy()) for k in state.files},strict=True)
    return model.eval(),meta

def infer(path,checkpoint,output,cancel=None,progress=None):
    import rasterio,torch
    from rasterio.windows import Window
    source_hash=sha256(path)
    model,meta=load(checkpoint);torch.set_num_threads(1)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);tmp=output.with_suffix('.tmp.tif')
    mean=np.asarray(meta['mean'],np.float32)[:,None,None];std=np.asarray(meta['std'],np.float32)[:,None,None]
    try:
        with rasterio.open(path) as src:
            if src.crs is None:raise ValueError('Raster CRS required')
            if max(meta['bands'])>src.count:raise ValueError('Checkpoint input bands missing')
            profile=src.profile.copy();profile.update(count=2,dtype='float32',nodata=np.nan,compress='deflate');profile.pop('photometric',None)
            rows=list(range(0,src.height,256));cols=list(range(0,src.width,256));total=len(rows)*len(cols);done=0
            with rasterio.open(tmp,'w',**profile) as dst,torch.no_grad():
                dst.set_band_description(1,'training_label_id');dst.set_band_description(2,'uncalibrated_max_softmax')
                for row in rows:
                    for col in cols:
                        check(cancel);height=min(256,src.height-row);width=min(256,src.width-col)
                        top=max(0,row-16);left=max(0,col-16);bottom=min(src.height,row+height+16);right=min(src.width,col+width+16)
                        window=Window(left,top,right-left,bottom-top)
                        x=src.read(meta['bands'],window=window,masked=True).astype(np.float32).filled(np.nan)
                        valid=np.isfinite(x).all(axis=0);x=np.nan_to_num((x-mean)/std)
                        p=torch.softmax(model(torch.from_numpy(x[None])),dim=1).numpy()[0]
                        values=np.stack([p.argmax(axis=0).astype(np.float32),p.max(axis=0)])
                        values[:,~valid]=np.nan;values=values[:,row-top:row-top+height,col-left:col-left+width]
                        dst.write(values,window=Window(col,row,width,height));done+=1
                        if progress:progress(int(100*done/total),'M3 raster window')
        if sha256(path)!=source_hash:raise ValueError('Raster source changed during inference')
        tmp.replace(output)
        atomic_json(output.with_suffix('.json'),{'class_keys':meta['class_keys'],'checkpoint_sha256':sha256(checkpoint),
                    'input_sha256':sha256(path),'output_sha256':sha256(output),'crs_policy':'input pixel grid preserved',
                    'evidence_kind':'raster output, not automatically transferred to points','accuracy':None})
        return output
    except BaseException:tmp.unlink(missing_ok=True);raise
