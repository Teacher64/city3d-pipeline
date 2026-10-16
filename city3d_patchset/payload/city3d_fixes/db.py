import json
from pathlib import Path


def enrich_project(project,params):
    from city3d_tab1.database import connect
    from .references import sheets_for_geometry
    from .project import atomic_json,to_wgs84
    from shapely.geometry import shape
    project=dict(project)
    try:
        connection=connect(connection_options(params))
        try:
            with connection.cursor() as cursor:
                cursor.execute("SET statement_timeout='10s'")
                project['sheets_50k']=sheets_for_geometry(cursor,to_wgs84(shape(project['boundary']),project['crs_wkt']))
                project['sheets_status']='matched' if project['sheets_50k'] else 'no_matching_50k_map_polygons'
        finally:connection.close()
    except Exception as exc:
        project['sheets_50k']=[]
        project['sheets_status']='unavailable: '+type(exc).__name__
    folder=Path(project['path']).parent
    atomic_json(folder/'project.json',project)
    boundary=folder/'boundary.geojson'
    if boundary.exists():
        data=json.loads(boundary.read_text())
        for f in data['features']:
            f['properties']['sheets_50k']=project['sheets_50k']
            f['properties']['sheets_status']=project['sheets_status']
        atomic_json(boundary,data)
    return project


def connection_options(params):
    # These are application keys, not libpq connection keywords.
    options = {k:v for k,v in params.items() if k not in
               ('schema_refs','schema_history','schema','enabled','config_file')}
    if 'database' in options:
        options['dbname'] = options.pop('database')
    return options


def register_project(project, params):
    from city3d_tab1.database import connect
    connection = connect(connection_options(params))
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute("SET LOCAL lock_timeout='5s'")
                cursor.execute("SET LOCAL statement_timeout='30s'")
                cursor.execute('CREATE SCHEMA IF NOT EXISTS als')
                cursor.execute('''CREATE TABLE IF NOT EXISTS als.city3d_project_revisions (
                    revision_id uuid PRIMARY KEY, project_id uuid NOT NULL,
                    name text NOT NULL, created_at timestamptz NOT NULL,
                    boundary_native public.geometry NOT NULL, crs_wkt text NOT NULL,
                    metadata jsonb NOT NULL)''')
                cursor.execute('''INSERT INTO als.city3d_project_revisions VALUES
                    (%s,%s,%s,%s,ST_SetSRID(ST_GeomFromGeoJSON(%s),0),%s,%s::jsonb)
                    ON CONFLICT(revision_id) DO NOTHING''',
                    (project['revision_id'],project['project_id'],project['name'],project['created_at'],
                     json.dumps(project['boundary']),project['crs_wkt'],json.dumps(project)))
    finally:
        connection.close()


def sync_archive_records(records,params):
    import re
    from city3d_tab1.database import connect
    schema=params.get('schema_history','history')
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',schema):raise ValueError('Недопустимая схема истории')
    connection=connect(connection_options(params))
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute("SET LOCAL lock_timeout='5s'")
                cursor.execute("SET LOCAL statement_timeout='30s'")
                cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
                cursor.execute(f'''CREATE TABLE IF NOT EXISTS "{schema}".city3d_archive_catalog
                    (path text PRIMARY KEY,area_name text,metadata jsonb NOT NULL,indexed_at timestamptz NOT NULL DEFAULT now())''')
                for r in records:
                    cursor.execute(f'''INSERT INTO "{schema}".city3d_archive_catalog(path,area_name,metadata)
                        VALUES (%s,%s,%s::jsonb) ON CONFLICT(path) DO UPDATE SET
                        area_name=excluded.area_name,metadata=excluded.metadata,indexed_at=now()''',
                        (r['output_dir'],r['area_name'],json.dumps(r)))
    finally:connection.close()
