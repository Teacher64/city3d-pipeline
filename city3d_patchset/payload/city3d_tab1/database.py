import json
from pathlib import Path


def connect(params):
    options = dict(params)
    if 'database' in options:
        options['dbname'] = options.pop('database')
    options.setdefault('connect_timeout', 8)
    try:
        import psycopg
        return psycopg.connect(**options)
    except ImportError:
        import psycopg2
        return psycopg2.connect(**options)


def sync(records, params, cancel=lambda: False):
    from .catalog import check_cancel
    connection = connect(params)
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute("SET LOCAL lock_timeout = '5s'")
                cursor.execute("SET LOCAL statement_timeout = '30s'")
                cursor.execute("SELECT to_regtype('public.geometry')")
                if cursor.fetchone()[0] is None:
                    raise RuntimeError('В выбранной БД нет PostGIS public.geometry')
                cursor.execute('SELECT pg_advisory_xact_lock(731017501)')
                cursor.execute(Path(__file__).with_name('schema.sql').read_text(encoding='utf-8'))
                for r in records:
                    check_cancel(cancel)
                    cursor.execute('''INSERT INTO als.city3d_input_files(file_id,sha256,size_bytes)
                        VALUES (%s,%s,%s) ON CONFLICT(file_id) DO NOTHING''',
                                   (r['file_id'], r['sha256'], r['size']))
                    cursor.execute('''INSERT INTO als.city3d_input_inspections
                        (inspection_id,file_id,source_path,observed_at,point_count,classes,
                         crs_wkt,crs_source,native_bounds,footprint_wgs84,footprint_method,state,metadata)
                        VALUES (%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb,
                         ST_GeomFromText(%s,4326),%s,%s,%s::jsonb)
                        ON CONFLICT(inspection_id) DO NOTHING''',
                                   (r['inspection_id'], r['file_id'], r['path'], r['observed_at'],
                                    r['point_count'], json.dumps(r['classes']), r['crs_wkt'],
                                    r['crs_source'], json.dumps(r['bounds']), r['footprint_wgs84'],
                                    r['footprint_method'], r['state'], json.dumps(r)))
        return len(records)
    finally:
        connection.close()
