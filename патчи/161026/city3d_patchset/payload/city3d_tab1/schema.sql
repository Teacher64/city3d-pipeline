-- Additive migration. Supplied metadata dump is NOT executed.
-- Requires an existing PostgreSQL database with PostGIS installed.
CREATE SCHEMA IF NOT EXISTS als;
CREATE TABLE IF NOT EXISTS als.city3d_input_files (
    file_id uuid PRIMARY KEY,
    sha256 text NOT NULL UNIQUE CHECK (length(sha256)=64),
    size_bytes bigint NOT NULL CHECK (size_bytes >= 0),
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS als.city3d_input_inspections (
    inspection_id uuid PRIMARY KEY,
    file_id uuid NOT NULL REFERENCES als.city3d_input_files(file_id),
    source_path text NOT NULL,
    observed_at timestamptz NOT NULL,
    point_count bigint,
    classes jsonb NOT NULL,
    crs_wkt text,
    crs_source text NOT NULL,
    native_bounds jsonb,
    footprint_wgs84 public.geometry(MultiPolygon,4326),
    footprint_method text NOT NULL,
    state text NOT NULL CHECK(state IN ('indexed','processed','failed')),
    metadata jsonb NOT NULL,
    -- Optional association to the pre-existing business record, not an ASPRS class ID.
    legacy_als_id bigint
);
CREATE INDEX IF NOT EXISTS city3d_input_inspections_file_idx
 ON als.city3d_input_inspections(file_id);
CREATE INDEX IF NOT EXISTS city3d_input_inspections_geom_idx
 ON als.city3d_input_inspections USING gist(footprint_wgs84);
CREATE INDEX IF NOT EXISTS city3d_input_inspections_time_idx
 ON als.city3d_input_inspections(observed_at);
COMMENT ON COLUMN als.city3d_input_inspections.footprint_wgs84 IS
 'Approximate bounding envelope, not surveyed coverage; NULL for unresolved CRS/transform.';
COMMENT ON COLUMN als.city3d_input_inspections.legacy_als_id IS
 'Optional als.als.id; no fabricated owner/contract/laser values. Link validation belongs to operator workflow.';
