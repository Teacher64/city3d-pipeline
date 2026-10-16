#!/usr/bin/env python3
"""Executable verification report; live DB check is read-only and opt-in."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--target',default='/mnt/data_sda1/PR_3D_SPB/city3d-pipeline/city3d_pipeline_v5.py')
    parser.add_argument('--live-db',action='store_true')
    parser.add_argument('--report',default='verification_report.json')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    target=Path(args.target)
    report=dict(at=datetime.now(timezone.utc).isoformat(),python=sys.executable,python_version=sys.version,
                target=str(target),dependencies={},checks={})
    for name in ('laspy','pyproj','shapely','PyQt5.QtWidgets','matplotlib','pytest','rasterio'):
        try:
            m=importlib.import_module(name);report['dependencies'][name]={'status':'available','version':getattr(m,'__version__','unknown')}
        except Exception as exc:report['dependencies'][name]={'status':'missing','error':str(exc)}
    if target.is_file():
        try:
            compile(target.read_text(),str(target),'exec');report['checks']['main_syntax']='passed'
        except Exception as exc:report['checks']['main_syntax']=str(exc)
    else:report['checks']['main_syntax']='target_not_found'
    env=dict(os.environ)
    env['QT_QPA_PLATFORM']='offscreen'
    env['PYTHONPATH']=os.pathsep.join([str(target.resolve().parent),str(root/'payload'),env.get('PYTHONPATH','')])
    result=subprocess.run([sys.executable,'-m','pytest',str(root/'tests'),'-q'],env=env,capture_output=True,text=True)
    report['tests']={'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
    if args.live_db:
        try:
            sys.path.insert(0,str(target.resolve().parent));sys.path.append(str(root/'payload'))
            from city3d_fixes.db import connection_options
            from city3d_tab1.database import connect
            settings_path=Path.home()/'.config/city3d/settings.json'
            settings=json.loads(settings_path.read_text()) if settings_path.exists() else {}
            params=settings.get('database',{})
            if not params:raise ValueError('В settings.json нет явных настроек database; подключение не угадывается')
            conn=connect(connection_options(params))
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT current_database(),PostGIS_Version(),CASE WHEN %s::text IS NULL THEN NULL ELSE ST_GeomFromText(%s,4326) END",(None,None))
                    row=cur.fetchone();report['checks']['live_db']={'status':'passed','database':row[0],'postgis':row[1]}
            finally:conn.close()
        except Exception as exc:report['checks']['live_db']={'status':'failed','error':str(exc)}
    report['status']='passed' if result.returncode==0 and report['checks'].get('main_syntax')=='passed' and (not args.live_db or report['checks']['live_db']['status']=='passed') else 'failed_or_incomplete'
    output=Path(args.report).resolve();output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('Проверка:',report['status']);print('Отчёт:',output);print(result.stdout)
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':sys.exit(main())
