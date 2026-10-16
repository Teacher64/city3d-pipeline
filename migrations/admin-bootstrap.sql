-- Existing City3D database: DBA installs only missing required extension.
-- Run inside the EXISTING target database. No CREATE DATABASE or role/password changes.
CREATE EXTENSION IF NOT EXISTS postgis;
-- Normal application role needs USAGE on city3d_meta2 and data privileges.
-- Replace :application_role through a trusted SQL client identifier mechanism.
-- Migration role is separate; application does not run DDL at connection time.
