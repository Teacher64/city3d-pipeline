CREATE SCHEMA IF NOT EXISTS city3d_meta2;
CREATE TABLE IF NOT EXISTS city3d_meta2.schema_migrations (
 version integer PRIMARY KEY, checksum text NOT NULL, applied_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS city3d_meta2.assets (
 id uuid PRIMARY KEY, logical_uri text NOT NULL UNIQUE, media_type text NOT NULL, created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS city3d_meta2.asset_versions (
 id uuid PRIMARY KEY, asset_id uuid NOT NULL REFERENCES city3d_meta2.assets(id), sha256 char(64) NOT NULL UNIQUE,
 size_bytes bigint NOT NULL CHECK(size_bytes>=0), native_crs text NOT NULL, vertical_crs text,
 footprint geometry(Geometry,4326), acquisition jsonb NOT NULL DEFAULT '{}', license text,
 provenance jsonb NOT NULL DEFAULT '{}', parent_version uuid REFERENCES city3d_meta2.asset_versions(id),
 created_at timestamptz NOT NULL DEFAULT now());
CREATE INDEX IF NOT EXISTS asset_versions_footprint ON city3d_meta2.asset_versions USING gist(footprint);
CREATE INDEX IF NOT EXISTS asset_versions_parent ON city3d_meta2.asset_versions(parent_version);
CREATE TABLE IF NOT EXISTS city3d_meta2.asset_locations (
 version_id uuid NOT NULL REFERENCES city3d_meta2.asset_versions(id), uri text NOT NULL, verified_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(version_id,uri));
CREATE TABLE IF NOT EXISTS city3d_meta2.semantic_schemas (
 version text PRIMARY KEY, definition jsonb NOT NULL, sha256 char(64) NOT NULL UNIQUE);
CREATE TABLE IF NOT EXISTS city3d_meta2.semantic_mappings (
 version text PRIMARY KEY, source_schema text NOT NULL, target_schema text NOT NULL REFERENCES city3d_meta2.semantic_schemas(version),
 definition jsonb NOT NULL, sha256 char(64) NOT NULL UNIQUE);
CREATE TABLE IF NOT EXISTS city3d_meta2.layers (
 id uuid PRIMARY KEY, asset_version uuid NOT NULL REFERENCES city3d_meta2.asset_versions(id), name text NOT NULL,
 native_crs text NOT NULL, vertical_crs text, style jsonb NOT NULL DEFAULT '{}', UNIQUE(asset_version,name));
CREATE TABLE IF NOT EXISTS city3d_meta2.objects (
 id uuid PRIMARY KEY, layer_id uuid NOT NULL REFERENCES city3d_meta2.layers(id), source_feature_id text NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(), UNIQUE(layer_id,source_feature_id));
CREATE TABLE IF NOT EXISTS city3d_meta2.object_versions (
 id uuid PRIMARY KEY, object_id uuid NOT NULL REFERENCES city3d_meta2.objects(id), revision integer NOT NULL CHECK(revision>0),
 native_geometry geometry NOT NULL, normalized_geometry geometry(Geometry,4326) NOT NULL,
 attributes jsonb NOT NULL DEFAULT '{}', class_key text, schema_version text REFERENCES city3d_meta2.semantic_schemas(version),
 provenance jsonb NOT NULL DEFAULT '{}', valid_time timestamptz, qa_status text NOT NULL DEFAULT 'unreviewed'
 CHECK(qa_status IN ('unreviewed','gold','pseudo','rejected')), UNIQUE(object_id,revision));
CREATE INDEX IF NOT EXISTS objects_normalized_geometry ON city3d_meta2.object_versions USING gist(normalized_geometry);
CREATE INDEX IF NOT EXISTS objects_native_geometry ON city3d_meta2.object_versions USING gist(native_geometry);
CREATE INDEX IF NOT EXISTS objects_class ON city3d_meta2.object_versions(class_key);
CREATE TABLE IF NOT EXISTS city3d_meta2.map_sheets (
 id uuid PRIMARY KEY, nomenclature text NOT NULL, scale integer NOT NULL CHECK(scale>0), scheme_version text NOT NULL,
 native_crs text NOT NULL, normalized_frame geometry(Geometry,4326) NOT NULL, source_asset uuid REFERENCES city3d_meta2.asset_versions(id),
 parent_sheet uuid REFERENCES city3d_meta2.map_sheets(id), UNIQUE(nomenclature,scale,scheme_version));
CREATE INDEX IF NOT EXISTS map_sheets_frame ON city3d_meta2.map_sheets USING gist(normalized_frame);
CREATE TABLE IF NOT EXISTS city3d_meta2.asset_sheets (
 asset_version uuid NOT NULL REFERENCES city3d_meta2.asset_versions(id), sheet_id uuid NOT NULL REFERENCES city3d_meta2.map_sheets(id),
 area_m2 double precision NOT NULL CHECK(area_m2>=0), boundary_touch boolean NOT NULL,
 PRIMARY KEY(asset_version,sheet_id));
CREATE TABLE IF NOT EXISTS city3d_meta2.dataset_versions (
 id uuid PRIMARY KEY, manifest jsonb NOT NULL, sha256 char(64) NOT NULL UNIQUE, ground_truth_status text NOT NULL,
 schema_version text REFERENCES city3d_meta2.semantic_schemas(version), created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS city3d_meta2.split_members (
 dataset_version uuid NOT NULL REFERENCES city3d_meta2.dataset_versions(id), asset_version uuid NOT NULL REFERENCES city3d_meta2.asset_versions(id),
 spatial_group text NOT NULL, split text NOT NULL CHECK(split IN ('train','val','test')), PRIMARY KEY(dataset_version,asset_version));
CREATE TABLE IF NOT EXISTS city3d_meta2.model_versions (
 id uuid PRIMARY KEY, model_id text NOT NULL, checkpoint_sha256 char(64), config jsonb NOT NULL, status text NOT NULL,
 dataset_version uuid REFERENCES city3d_meta2.dataset_versions(id), parent_model uuid REFERENCES city3d_meta2.model_versions(id),
 license text, created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS city3d_meta2.evaluations (
 id uuid PRIMARY KEY, model_version uuid NOT NULL REFERENCES city3d_meta2.model_versions(id), dataset_version uuid NOT NULL REFERENCES city3d_meta2.dataset_versions(id),
 metrics jsonb NOT NULL, independent boolean NOT NULL, created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS city3d_meta2.runs (
 id uuid PRIMARY KEY, status text NOT NULL CHECK(status IN ('planned','ready','running','success','failed','cancelled','skipped','blocked','partial')),
 definition jsonb NOT NULL, manifest jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now());
CREATE INDEX IF NOT EXISTS runs_status_time ON city3d_meta2.runs(status,created_at);
CREATE TABLE IF NOT EXISTS city3d_meta2.steps (
 run_id uuid NOT NULL REFERENCES city3d_meta2.runs(id), task_id text NOT NULL, status text NOT NULL,
 manifest jsonb NOT NULL, PRIMARY KEY(run_id,task_id));
CREATE TABLE IF NOT EXISTS city3d_meta2.artifacts (
 id uuid PRIMARY KEY, run_id uuid NOT NULL REFERENCES city3d_meta2.runs(id), asset_version uuid REFERENCES city3d_meta2.asset_versions(id),
 media_type text NOT NULL, sha256 char(64) NOT NULL, uri text NOT NULL, metadata jsonb NOT NULL,
 UNIQUE(run_id,sha256,uri));
CREATE TABLE IF NOT EXISTS city3d_meta2.artifact_dependencies (
 artifact_id uuid NOT NULL REFERENCES city3d_meta2.artifacts(id), input_artifact uuid NOT NULL REFERENCES city3d_meta2.artifacts(id),
 PRIMARY KEY(artifact_id,input_artifact));
CREATE TABLE IF NOT EXISTS city3d_meta2.result_snapshots (
 id uuid PRIMARY KEY, run_id uuid NOT NULL REFERENCES city3d_meta2.runs(id), definition jsonb NOT NULL,
 sha256 char(64) NOT NULL UNIQUE, qa_status text NOT NULL, created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS city3d_meta2.snapshot_members (
 snapshot_id uuid NOT NULL REFERENCES city3d_meta2.result_snapshots(id), object_version uuid NOT NULL REFERENCES city3d_meta2.object_versions(id),
 PRIMARY KEY(snapshot_id,object_version));
CREATE TABLE IF NOT EXISTS city3d_meta2.export_jobs (
 id uuid PRIMARY KEY, snapshot_id uuid NOT NULL REFERENCES city3d_meta2.result_snapshots(id), profile jsonb NOT NULL,
 status text NOT NULL, output_sha256 char(64), validation jsonb NOT NULL DEFAULT '{}');
CREATE TABLE IF NOT EXISTS city3d_meta2.legacy_aliases (
 entity_type text NOT NULL, legacy_schema text NOT NULL, legacy_id text NOT NULL, canonical_uuid uuid NOT NULL,
 PRIMARY KEY(entity_type,legacy_schema,legacy_id));
CREATE TABLE IF NOT EXISTS city3d_meta2.review_events (
 id uuid PRIMARY KEY, object_version uuid REFERENCES city3d_meta2.object_versions(id), reviewer text NOT NULL,
 decision text NOT NULL, provenance jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS city3d_meta2.outbox_jobs (
 id uuid PRIMARY KEY, topic text NOT NULL, payload jsonb NOT NULL, status text NOT NULL DEFAULT 'pending',
 created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS city3d_meta2.backup_manifests (
 id uuid PRIMARY KEY, definition jsonb NOT NULL, sha256 char(64) NOT NULL, restore_tested boolean NOT NULL DEFAULT false,
 created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS city3d_meta2.layer_asset_versions (
 layer_id uuid NOT NULL REFERENCES city3d_meta2.layers(id), asset_version uuid NOT NULL REFERENCES city3d_meta2.asset_versions(id),
 PRIMARY KEY(layer_id,asset_version));
