import argparse
import json
from pathlib import Path
from .settings import atomic_json

def main(argv=None):
    parser=argparse.ArgumentParser(description='City3D pipeline, inventory, models, existing PostGIS administration')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('pipeline');p.add_argument('definition',type=Path);p.add_argument('output',type=Path)
    p=sub.add_parser('inventory');p.add_argument('models_root',type=Path);p.add_argument('--datasets-root',type=Path);p.add_argument('--output',type=Path,default=Path('inventory.json'))
    p=sub.add_parser('models');p.add_argument('action',choices=['download-spt','import-bundle']);p.add_argument('directory',type=Path);p.add_argument('--bundle',type=Path)
    p=sub.add_parser('db');p.add_argument('action',choices=['inspect','backup','restore-test','migrate','import-vector','sync']);p.add_argument('--connection-file',type=Path);p.add_argument('--database');p.add_argument('--settings',type=Path);p.add_argument('--backup-manifest',type=Path);p.add_argument('--output',type=Path);p.add_argument('--input',type=Path);p.add_argument('--crs');p.add_argument('--journal',type=Path)
    args=parser.parse_args(argv)
    if args.command=='pipeline':
        from .runner import PipelineRunner,Task
        from .database import LocalJournal
        data=json.loads(args.definition.read_text());tasks=[Task(**t) for t in data['tasks']]
        journal=LocalJournal(args.output/'outbox.sqlite')
        result=PipelineRunner(journal=journal).run(tasks,args.output,spatial=data.get('spatial',{}))
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if result['status'] not in ('success','partial'):raise SystemExit(1)
    elif args.command=='inventory':
        from .registry import inventory,discover_datasets
        result={'models':inventory(args.models_root),'datasets':discover_datasets(args.datasets_root) if args.datasets_root else []}
        atomic_json(args.output,result);print(args.output)
    elif args.command=='models':
        from .downloads import download,SPT_NANO_DALES,import_bundle
        if args.action=='download-spt':print(download(SPT_NANO_DALES,args.directory))
        elif args.bundle:print(import_bundle(args.bundle,args.directory))
        else:raise ValueError('--bundle required')
    else:
        from .database import load_connection_file,inspect,backup,restore_test,migrate,Repository,LocalJournal
        from .settings import read_settings
        settings=read_settings(args.settings) if args.settings else read_settings()
        if args.connection_file:settings=dict(settings,database=load_connection_file(args.connection_file))
        if args.database:settings.setdefault('database',{})['dbname']=args.database
        if args.action=='inspect':print(json.dumps(inspect(settings),ensure_ascii=False,indent=2))
        elif args.action=='backup':
            if not args.output:raise ValueError('--output protected backup directory required')
            print(backup(args.output,settings))
        elif args.action=='restore-test':
            if not args.backup_manifest:raise ValueError('--backup-manifest required')
            print(restore_test(args.backup_manifest,settings))
        elif args.action=='migrate':print(migrate(settings,args.backup_manifest))
        elif args.action=='import-vector':
            if not args.input or not args.crs:raise ValueError('--input and --crs required')
            print(Repository(settings).import_vector(args.input,args.crs))
        elif args.action=='sync':
            if not args.journal:raise ValueError('--journal required')
            print(LocalJournal(args.journal).sync(Repository(settings)))

if __name__=='__main__':main()
