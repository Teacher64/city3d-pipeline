"""Explicit migrations and per-job PostGIS repository; no DDL on GUI connect."""
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import tempfile
from uuid import uuid4,uuid5,NAMESPACE_URL
from .settings import read_settings,atomic_json
from .core import sha256


# CITY3D_SQLITE_CLOSE_FIX_20261019
class _City3DClosingSQLite(sqlite3.Connection):
    def __exit__(self, *exc):
        try:
            return super().__exit__(*exc)
        finally:
            self.close()


STATUSES=('planned','ready','running','success','failed','cancelled','skipped','blocked','partial')

def identifier(name):
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',name):raise ValueError('Недопустимое имя SQL schema')
    return name

def load_connection_file(path,server_id=None):
    data=json.loads(Path(path).read_text())
    if 'Servers' not in data:
        return data.get('database',data)
    servers=data['Servers']
    key=str(server_id) if server_id is not None else next(iter(servers))
    source=servers[key]
    if source.get('UseSSHTunnel'):
        raise ValueError('SSH tunnel must be established separately; pgAdmin tunnel is not a libpq endpoint')
    result={target:source[key] for key,target in [('Host','host'),('Port','port'),('Username','user'),('MaintenanceDB','dbname'),('Password','password'),('SSLMode','sslmode')] if key in source}
    for key,value in source.get('ConnectionParameters',{}).items():
        if key in ('sslmode','sslcert','sslkey','sslrootcert','connect_timeout'):result[key]=value
    return result

def connection_params(settings=None):
    cfg=(settings if settings is not None else read_settings()).get('database',{})
    external=(settings if settings is not None else read_settings()).get('database_config_file') or os.environ.get('CITY3D_DB_CONFIG')
    if external and not cfg:
        cfg=load_connection_file(external)
    identifier(cfg.get('schema_history','history'));identifier(cfg.get('schema_refs','refs'))
    if cfg.get('service'):return {'service':cfg['service'],'connect_timeout':5}
    return {k:v for k,v in cfg.items() if k in ('dbname','host','port','user','password','sslmode','sslcert','sslkey','sslrootcert')} | {'connect_timeout':5}

def connect(settings=None):
    import psycopg
    return psycopg.connect(**connection_params(settings))

def inspect(settings=None):
    with connect(settings) as conn,conn.cursor() as cur:
        cur.execute("SELECT current_database(),current_user,version()")
        db,user,version=cur.fetchone()
        cur.execute("SELECT extname,extversion FROM pg_extension ORDER BY extname")
        ext=cur.fetchall()
        cur.execute("SELECT nspname FROM pg_namespace WHERE nspname !~ '^pg_' AND nspname <> 'information_schema'")
        schemas=[r[0] for r in cur.fetchall()]
        return dict(database=db,user=user,postgresql=version,extensions=ext,schemas=schemas)

def backup(directory,settings=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);directory.chmod(0o700)
    params=connection_params(settings);uid=str(uuid4());output=directory/(uid+'.dump')
    # libpq credentials through protected environment/file, never command line.
    env=os.environ.copy()
    for k,envkey in [('dbname','PGDATABASE'),('host','PGHOST'),('port','PGPORT'),('user','PGUSER'),('password','PGPASSWORD'),('service','PGSERVICE')]:
        if k in params:env[envkey]=str(params[k])
    result=subprocess.run(['pg_dump','--format=custom','--file',str(output)],env=env,capture_output=True,text=True)
    if result.returncode:
        output.unlink(missing_ok=True);raise RuntimeError('pg_dump failed; check database connectivity/permissions')
    output.chmod(0o600)
    subprocess.run(['pg_restore','--list',str(output)],check=True,stdout=subprocess.DEVNULL)
    atomic_json(directory/(uid+'.json'),dict(id=uid,path=str(output),sha256=sha256(output),catalog=inspect(settings),restore_tested=False))
    return output

def sqlite_backup(source,target):
    with sqlite3.connect('file:'+str(Path(source).resolve())+'?mode=ro',uri=True,factory=_City3DClosingSQLite) as src,sqlite3.connect(target,factory=_City3DClosingSQLite) as dst:
        src.backup(dst)

def migrate(settings=None,backup_manifest=None):
    # Live migration requires an actual backup manifest, not a filename placeholder.
    if backup_manifest is None:raise ValueError('Сначала создайте проверенный pg_dump backup manifest')
    manifest=json.loads(Path(backup_manifest).read_text())
    if not manifest.get('restore_tested'):
        raise ValueError('Сначала проверьте восстановление backup в отдельной БД; restore_tested=false')
    if sha256(manifest['path'])!=manifest['sha256']:raise ValueError('Backup checksum mismatch')
    script=Path(__file__).resolve().parent.parent/'migrations/002_catalog.sql'
    sql=script.read_text();checksum=hashlib.sha256(sql.encode()).hexdigest()
    import psycopg
    from psycopg import sql as psql
    with connect(settings) as conn,conn.cursor() as cur:
        cur.execute('SELECT pg_advisory_xact_lock(%s)',(75002,))
        cur.execute("SELECT 1 FROM pg_extension WHERE extname='postgis'")
        if cur.fetchone() is None:raise RuntimeError('PostGIS отсутствует; DBA должен установить extension')
        cur.execute("SELECT to_regclass('city3d_meta2.schema_migrations')")
        if cur.fetchone()[0]:
            cur.execute('SELECT checksum FROM city3d_meta2.schema_migrations WHERE version=2')
            row=cur.fetchone()
            if row:
                if row[0]!=checksum:raise ValueError('Applied migration checksum mismatch')
                return 'already applied'
        cur.execute(sql)
        cfg=(settings if settings is not None else read_settings()).get('database',{})
        hist=identifier(cfg.get('schema_history','history'))
        cur.execute('SELECT to_regclass(%s)',(hist+'.training_runs',))
        if cur.fetchone()[0]:
            cur.execute("SELECT c.conname FROM pg_constraint c JOIN pg_class t ON t.oid=c.conrelid JOIN pg_namespace n ON n.oid=t.relnamespace WHERE n.nspname=%s AND t.relname='training_runs' AND c.contype='c' AND pg_get_constraintdef(c.oid) LIKE '%%status%%'",(hist,))
            for name, in cur.fetchall():
                cur.execute(psql.SQL('ALTER TABLE {}.training_runs DROP CONSTRAINT {}').format(psql.Identifier(hist),psql.Identifier(name)))
            cur.execute(psql.SQL('ALTER TABLE {}.training_runs ADD CONSTRAINT city3d_run_status_v2 CHECK(status = ANY({}))').format(psql.Identifier(hist),psql.Literal(list(STATUSES))))
        cur.execute('INSERT INTO city3d_meta2.schema_migrations(version,checksum) VALUES(2,%s)',(checksum,))
    return 'applied'

class Repository:
    def __init__(self,settings=None):self.settings=settings
    def register_asset(self,path,crs,footprint_wkt=None,metadata=None):
        from psycopg.types.json import Jsonb
        path=Path(path).resolve();digest=sha256(path)
        asset=str(uuid5(NAMESPACE_URL,'city3d:asset:'+str(path)));version=str(uuid5(NAMESPACE_URL,'city3d:content:'+digest))
        with connect(self.settings) as conn,conn.cursor() as cur:
            cur.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))',(digest,))
            cur.execute('SELECT id FROM city3d_meta2.asset_versions WHERE sha256=%s',(digest,))
            row=cur.fetchone()
            if row:version=str(row[0])
            else:
                cur.execute('INSERT INTO city3d_meta2.assets(id,logical_uri,media_type) VALUES(%s,%s,%s) ON CONFLICT(logical_uri) DO NOTHING',(asset,str(path),path.suffix))
                cur.execute('INSERT INTO city3d_meta2.asset_versions(id,asset_id,sha256,size_bytes,native_crs,footprint,provenance) VALUES(%s,%s,%s,%s,%s,ST_Force2D(ST_GeomFromText(%s,4326)),%s)',(version,asset,digest,path.stat().st_size,crs,footprint_wkt,Jsonb(metadata or {})))
            cur.execute('INSERT INTO city3d_meta2.asset_locations(version_id,uri) VALUES(%s,%s) ON CONFLICT DO NOTHING',(version,str(path)))
        return version
    def import_vector(self,path,crs):
        from psycopg.types.json import Jsonb
        from .layers import load_vector
        before=sha256(path)
        layer=load_vector(path,crs)
        if sha256(path)!=before:raise ValueError('Input changed during geometry import')
        version=self.register_asset(path,layer['crs'],layer['footprint'].wkt,{'source':'vector','qa':'unreviewed'})
        layer_id=str(uuid5(NAMESPACE_URL,'city3d:layer:'+str(Path(path).resolve())))
        with connect(self.settings) as conn,conn.cursor() as cur:
            cur.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))',(layer_id,))
            cur.execute('INSERT INTO city3d_meta2.layers(id,asset_version,name,native_crs) VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING',(layer_id,version,Path(path).name,layer['crs']))
            cur.execute('INSERT INTO city3d_meta2.layer_asset_versions(layer_id,asset_version) VALUES(%s,%s) ON CONFLICT DO NOTHING',(layer_id,version))
            objects=[]
            for feature in layer['features']:
                key=str(feature['id']);oid=str(uuid5(NAMESPACE_URL,layer_id+':'+key));vid=str(uuid5(NAMESPACE_URL,oid+':'+version))
                objects.append((oid,vid,key,feature))
            cur.executemany('INSERT INTO city3d_meta2.objects(id,layer_id,source_feature_id) VALUES(%s,%s,%s) ON CONFLICT DO NOTHING',[(o,layer_id,k) for o,v,k,f in objects])
            cur.execute('SELECT object_id, max(revision) FROM city3d_meta2.object_versions WHERE object_id = ANY(%s::uuid[]) GROUP BY object_id',([o for o,v,k,f in objects],))
            revisions={str(oid):revision for oid,revision in cur.fetchall()}
            cur.executemany('INSERT INTO city3d_meta2.object_versions(id,object_id,revision,native_geometry,normalized_geometry,attributes,provenance) VALUES(%s,%s,%s,ST_GeomFromText(%s,%s),ST_Force2D(ST_GeomFromText(%s,4326)),%s,%s) ON CONFLICT DO NOTHING',[(v,o,revisions.get(o,0)+1,f['native'].wkt,layer['srid'],f['normalized'].wkt,Jsonb(f['properties']),Jsonb({'asset_version':version,'source_feature_id':k})) for o,v,k,f in objects])
        return dict(asset_version=version,layer_uuid=layer_id,objects=len(objects))
    def find_objects(self,bounds,limit=1000,offset=0):
        if not 1<=limit<=10000 or offset<0:raise ValueError('Pagination bounds')
        with connect(self.settings) as conn,conn.cursor() as cur:
            cur.execute('SELECT object_id,id,class_key,ST_AsGeoJSON(normalized_geometry),attributes FROM city3d_meta2.object_versions WHERE ST_Intersects(normalized_geometry,ST_MakeEnvelope(%s,%s,%s,%s,4326)) ORDER BY object_id,revision LIMIT %s OFFSET %s',(*bounds,limit,offset))
            return cur.fetchall()
    def link_sheets(self,asset_version):
        with connect(self.settings) as conn,conn.cursor() as cur:
            cur.execute('INSERT INTO city3d_meta2.asset_sheets(asset_version,sheet_id,area_m2,boundary_touch) SELECT a.id,s.id,ST_Area(ST_Intersection(a.footprint,s.normalized_frame)::geography),ST_Touches(a.footprint,s.normalized_frame) FROM city3d_meta2.asset_versions a JOIN city3d_meta2.map_sheets s ON ST_Intersects(a.footprint,s.normalized_frame) WHERE a.id=%s ON CONFLICT(asset_version,sheet_id) DO UPDATE SET area_m2=EXCLUDED.area_m2,boundary_touch=EXCLUDED.boundary_touch',(asset_version,))
            return cur.rowcount
    def record_run(self,manifest):
        from psycopg.types.json import Jsonb
        with connect(self.settings) as conn,conn.cursor() as cur:
            cur.execute('INSERT INTO city3d_meta2.runs(id,status,definition,manifest) VALUES(%s,%s,%s,%s) ON CONFLICT(id) DO UPDATE SET status=EXCLUDED.status,manifest=EXCLUDED.manifest',(manifest['run_uuid'],manifest['status'],Jsonb(manifest.get('definition',{})),Jsonb(manifest)))
            for task_id,step in manifest.get('steps',{}).items():
                cur.execute('INSERT INTO city3d_meta2.steps(run_id,task_id,status,manifest) VALUES(%s,%s,%s,%s) ON CONFLICT(run_id,task_id) DO UPDATE SET status=EXCLUDED.status,manifest=EXCLUDED.manifest',(manifest['run_uuid'],task_id,step['status'],Jsonb(step)))
                for artifact in step.get('artifacts',{}).values():
                    cur.execute('INSERT INTO city3d_meta2.artifacts(id,run_id,media_type,sha256,uri,metadata) VALUES(%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING',(artifact['uuid'],manifest['run_uuid'],artifact['kind'],artifact['sha256'],artifact['path'],Jsonb(artifact)))
            snapshot_file=Path(manifest.get('directory',''))/'result_snapshot.json'
            if manifest.get('snapshot_uuid') and snapshot_file.is_file():
                snapshot=json.loads(snapshot_file.read_text())
                if snapshot['uuid']!=manifest['snapshot_uuid']:raise ValueError('Snapshot UUID mismatch during DB sync')
                cur.execute('INSERT INTO city3d_meta2.result_snapshots(id,run_id,definition,sha256,qa_status) VALUES(%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING',(snapshot['uuid'],manifest['run_uuid'],Jsonb(snapshot),sha256(snapshot_file),snapshot['qa_status']))


class LocalJournal:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(self.path,factory=_City3DClosingSQLite) as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS outbox(id TEXT PRIMARY KEY, topic TEXT NOT NULL,payload TEXT NOT NULL,status TEXT NOT NULL DEFAULT "pending",error TEXT)')
    def enqueue(self,uid,topic,payload):
        with sqlite3.connect(self.path,factory=_City3DClosingSQLite) as conn:
            conn.execute('INSERT INTO outbox(id,topic,payload) VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload,status="pending"',(uid,topic,json.dumps(payload,ensure_ascii=False)))
    def sync(self,repository):
        successes=0
        with sqlite3.connect(self.path,factory=_City3DClosingSQLite) as conn:
            for uid,topic,payload in conn.execute('SELECT id,topic,payload FROM outbox WHERE status="pending"').fetchall():
                try:
                    if topic!='run':raise ValueError('Unsupported journal topic')
                    repository.record_run(json.loads(payload));conn.execute('UPDATE outbox SET status="synced",error=NULL WHERE id=?',(uid,));successes+=1
                except Exception as e:
                    conn.execute('UPDATE outbox SET error=? WHERE id=?',(type(e).__name__,uid))
        return successes

def restore_test(manifest_path,settings=None,target_database=None):
    """Restore to a NEW isolated DB. Never overwrite an existing database."""
    from psycopg import sql
    manifest_path=Path(manifest_path);manifest=json.loads(manifest_path.read_text())
    if sha256(manifest['path'])!=manifest['sha256']:raise ValueError('Backup checksum mismatch')
    target=target_database or ('city3d_restore_check_'+uuid4().hex[:12]);identifier(target)
    params=connection_params(settings)
    import psycopg
    with psycopg.connect(**params,autocommit=True) as admin,admin.cursor() as cur:
        cur.execute('SELECT 1 FROM pg_database WHERE datname=%s',(target,))
        if cur.fetchone():raise ValueError('Restore target already exists; refusing to overwrite')
        cur.execute(sql.SQL('CREATE DATABASE {} TEMPLATE template0').format(sql.Identifier(target)))
    env=os.environ.copy()
    for key,variable in [('host','PGHOST'),('port','PGPORT'),('user','PGUSER'),('password','PGPASSWORD'),('service','PGSERVICE')]:
        if key in params:env[variable]=str(params[key])
    env['PGDATABASE']=target
    result=subprocess.run(['pg_restore','--exit-on-error','--dbname',target,manifest['path']],env=env,capture_output=True,text=True)
    if result.returncode:raise RuntimeError('Restore failed; isolated test DB retained for inspection, live DB unchanged')
    validation=dict(settings or {});validation['database']=dict(params,dbname=target)
    restored=inspect(validation)
    manifest.update(restore_tested=True,restore_database=target,restore_catalog=restored)
    atomic_json(manifest_path,manifest)
    return target

def reconcile_sqlite_references(sqlite_path,repository):
    """Idempotent legacy reference registration; source stays intact, no gold-label assumption."""
    from .layers import load_vector
    import laspy
    from pyproj import Transformer
    from shapely.geometry import box
    report=[]
    with sqlite3.connect('file:'+str(Path(sqlite_path).resolve())+'?mode=ro',uri=True,factory=_City3DClosingSQLite) as legacy:
        rows=legacy.execute('SELECT id,name,path FROM references_clouds').fetchall()
    for legacy_id,name,location in rows:
        path=Path(location)
        entry={'legacy_id':legacy_id,'name':name,'path':str(path)}
        try:
            if not path.is_file():raise FileNotFoundError('Asset path missing')
            if path.suffix.lower() in ('.las','.laz'):
                with laspy.open(path) as reader:
                    crs=reader.header.parse_crs()
                    if crs is None:raise ValueError('CRS unknown')
                    tr=Transformer.from_crs(crs,4326,always_xy=True)
                    x0,y0=tr.transform(*reader.header.mins[:2]);x1,y1=tr.transform(*reader.header.maxs[:2])
                    version=repository.register_asset(path,crs.to_string(),box(min(x0,x1),min(y0,y1),max(x0,x1),max(y0,y1)).wkt,{'legacy_id':legacy_id,'footprint_approximation':'bbox','qa':'unreviewed'})
            elif path.suffix.lower()=='.geojson':version=repository.import_vector(path,'EPSG:4326')['asset_version']
            else:raise ValueError('Legacy format requires verified reader')
            with connect(repository.settings) as conn,conn.cursor() as cur:
                cur.execute('INSERT INTO city3d_meta2.legacy_aliases(entity_type,legacy_schema,legacy_id,canonical_uuid) VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING',('asset_version','sqlite:'+str(Path(sqlite_path).resolve()),str(legacy_id),version))
            entry.update(status='registered',asset_version=version)
        except Exception as e:entry.update(status='blocked',error_type=type(e).__name__)
        report.append(entry)
    return report
