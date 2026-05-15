"""SQLite DDL / reset script for the BossDB pair-focused catalog."""

from __future__ import annotations

BOSSDB_PAIRS_CATALOG_RESET_SCRIPT = """
CREATE TABLE IF NOT EXISTS datasets (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  dataset_name TEXT NOT NULL UNIQUE,
  stage TEXT,
  segmentation_challenge TEXT,
  description TEXT,
  sample_organism TEXT,
  bio_target TEXT,
  bio_target_type TEXT,
  bio_target_source TEXT,
  bio_target_confidence REAL,
  sample_type TEXT,
  sample_subtype TEXT,
  sample_name TEXT,
  grid_spacing TEXT,
  grid_dimensions TEXT,
  grid_axes TEXT,
  grid_spacing_unit TEXT,
  grid_dimensions_unit TEXT,
  mitochondria_in_layer_names INTEGER NOT NULL DEFAULT 0,
  download_volume_url TEXT,
  download_volume_image_stack TEXT,
  download_mito_mask_url TEXT,
  download_mito_mask_quality TEXT,
  download_mito_mask_segmentation_kind TEXT,
  download_mito_prediction_url TEXT,
  s3_probe_img_path TEXT,
  s3_probe_seg_path TEXT,
  s3_probe_voxel_size TEXT,
  s3_probe_label_names TEXT,
  s3_probe_error TEXT
);
CREATE TABLE IF NOT EXISTS dataset_sources (
  dataset_id INTEGER NOT NULL,
  source_url TEXT NOT NULL,
  PRIMARY KEY (dataset_id, source_url),
  FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS dataset_layers (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  dataset_id INTEGER NOT NULL,
  layer_role TEXT NOT NULL,
  layer_kind TEXT NOT NULL,
  layer_name TEXT NOT NULL,
  layer_format TEXT,
  layer_stage TEXT,
  layer_url TEXT,
  layer_content_type TEXT,
  semantic INTEGER NOT NULL DEFAULT 0,
  raw_token TEXT,
  FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS dataset_resolved (
  dataset_id INTEGER PRIMARY KEY,
  dataset_name TEXT NOT NULL UNIQUE,
  resolved_em_path TEXT,
  resolved_mito_seg_path TEXT,
  resolved_voxel_nm TEXT,
  path_source TEXT,
  ready_labeled INTEGER NOT NULL DEFAULT 0,
  ready_em_only INTEGER NOT NULL DEFAULT 0,
  FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);
DELETE FROM dataset_layers;
DELETE FROM dataset_sources;
DELETE FROM dataset_resolved;
DELETE FROM datasets;
"""
