"""Explicit, verified downloads; never executed automatically during inference/install."""
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
from .settings import atomic_json
from .core import check

SPT_NANO_DALES = dict(id='spt_nano_dales',
 url='https://zenodo.org/records/8042712/files/spt-2_nano_dales.ckpt?download=1',
 filename='spt-2_nano_dales.ckpt',md5='d7442d12b6986deaeb94465f7eb52d04',
 source='https://zenodo.org/records/8042712',upstream_version='v1',
 domain='ALS DALES',license_status='not independently established; do not redistribute',
 format='pytorch lightning checkpoint; never automatically unpickled',
 dependencies=['official SPT revision/config','torch_scatter','torch_sparse','compiled partition libraries'])

def download(spec,directory,cancel=None,retries=2,timeout=15):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    filename=spec['filename']
    if filename in ('.','..') or Path(filename).name!=filename:raise ValueError('Filename must not contain paths')
    if not spec['url'].startswith('https://'):raise ValueError('HTTPS URL required')
    target=directory/filename;part=target.with_suffix(target.suffix+'.part')
    state=directory/(filename+'.download.json')
    record=dict(spec,status='planned',sha256_reference=spec.get('sha256'),local_sha256=None)
    if state.exists():
        previous=json.loads(state.read_text())
        if previous.get('url')!=spec['url']:
            raise ValueError('Download identity changed; use separate cache directory')
    def verify(file):
        hashes={name:hashlib.new(name) for name in ('sha256','md5')}
        size=0
        with file.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):
                check(cancel);size+=len(block)
                for h in hashes.values():h.update(block)
        if not size:raise ValueError('Empty download')
        for name in hashes:
            if spec.get(name) and hashes[name].hexdigest()!=spec[name]:raise ValueError('Checksum mismatch: '+name)
        if spec.get('size') and size!=spec['size']:raise ValueError('Download size mismatch')
        return size,hashes['sha256'].hexdigest()
    if target.exists():
        size,sha=verify(target);record.update(status='downloaded',bytes=size,local_sha256=sha)
        atomic_json(state,record);return target
    last=None
    for attempt in range(retries+1):
        check(cancel)
        try:
            offset=part.stat().st_size if part.exists() else 0
            req=urllib.request.Request(spec['url'],headers={'User-Agent':'City3D/7.5.0 local model manager',
                                      **({'Range':f'bytes={offset}-'} if offset else {})})
            record.update(status='downloading',attempt=attempt+1);atomic_json(state,record)
            with urllib.request.urlopen(req,timeout=timeout) as response:
                resumed=response.status==206 and offset>0
                if resumed and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                    raise ValueError('Invalid Content-Range')
                with part.open('ab' if resumed else 'wb') as f:
                    while True:
                        check(cancel);block=response.read(1024*1024)
                        if not block:break
                        f.write(block)
                    f.flush();os.fsync(f.fileno())
            size,sha=verify(part);os.replace(part,target)
            record.update(status='downloaded',bytes=size,local_sha256=sha,
                          verification='upstream checksum' if spec.get('sha256') or spec.get('md5') else 'local hash only')
            atomic_json(state,record);return target
        except Exception as e:
            from .core import Cancelled
            last=e;record.update(status='cancelled' if isinstance(e,Cancelled) else 'failed',error=str(e));atomic_json(state,record)
            if isinstance(e,Cancelled):raise
            if attempt<retries:time.sleep(min(2**attempt,4))
    raise RuntimeError('Download failed: '+str(last))

def import_bundle(bundle,directory):
    """Validate a directory bundle; only copy files with manifest SHA256."""
    import shutil
    bundle=Path(bundle);directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((bundle/'bundle.json').read_text())
    outputs=[]
    for entry in manifest['files']:
        name=entry['name']
        if Path(name).name!=name:raise ValueError('Unsafe bundle name')
        src=bundle/name
        if src.is_symlink() or hashlib.sha256(src.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('Bundle checksum/symlink error')
        dst=directory/name
        if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('Refusing to overwrite different cached weights')
        tmp=dst.with_suffix(dst.suffix+'.part');shutil.copyfile(src,tmp);tmp.replace(dst);outputs.append(str(dst))
    return outputs
