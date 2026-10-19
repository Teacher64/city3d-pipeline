#!/usr/bin/env python3
"""Read-only Linux FD snapshot of a running City3D process."""
import argparse,json,os,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

def snapshot(pid,details=False):
    root=Path('/proc')/str(pid);categories=Counter();paths=[]
    for entry in (root/'fd').iterdir():
        try:target=os.readlink(entry)
        except OSError:continue
        low=target.lower()
        if target.startswith('socket:'):kind='sockets'
        elif any(s in low for s in ('.sqlite','.mbtiles','-wal','-shm')):kind='sqlite_and_map_databases'
        elif low.endswith(('.las','.laz')):kind='LAS_LAZ'
        elif 'memfd:' in low or 'wayland' in low or '/dev/dri/' in low:kind='graphics_shared_memory'
        elif target.startswith('anon_inode:'):kind='event_descriptors'
        elif target.startswith('pipe:'):kind='pipes'
        else:kind='other_files'
        categories[kind]+=1
        if details:paths.append(dict(fd=entry.name,kind=kind,target=target))
    limit=next((line.strip() for line in (root/'limits').read_text().splitlines() if line.startswith('Max open files')),'unknown')
    result=dict(pid=pid,time=datetime.now(timezone.utc).isoformat(),total=sum(categories.values()),categories=dict(categories),limit=limit)
    if details:result['descriptors']=paths
    return result

def main():
    parser=argparse.ArgumentParser(description='Диагностика открытых дескрипторов City3D; ничего не меняет')
    parser.add_argument('--pid',type=int);parser.add_argument('--details',action='store_true');parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    pids=[args.pid] if args.pid else []
    if not pids:
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:
                arguments=(proc/'cmdline').read_bytes().split(b'\0')
                if any(Path(a.decode(errors='replace')).name=='city3d_pipeline_v5.py' for a in arguments if a):pids.append(int(proc.name))
            except (PermissionError,FileNotFoundError,ProcessLookupError):pass
    result=[]
    for pid in pids:
        try:result.append(snapshot(pid,args.details))
        except (PermissionError,FileNotFoundError,ProcessLookupError) as exc:result.append(dict(pid=pid,error=type(exc).__name__))
    data=dict(processes=result,message='City3D не найден: запустите программу или укажите --pid' if not result else 'Read-only snapshot')
    text=json.dumps(data,ensure_ascii=False,indent=2)
    if args.out:args.out.write_text(text,encoding='utf-8')
    print(text)

if __name__=='__main__':main()
