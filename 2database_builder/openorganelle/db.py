"""SQLite schema, migrations, and data loading for mitoFoundation."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .bioclass import classify_bio_target
from .parser import DatasetRecord


DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS datasets (
  id                          INTEGER PRIMARY KEY AUTOINCREMENT,
  dataset_name                TEXT    NOT NULL UNIQUE,
  stage                       TEXT,
  segmentation_challenge      TEXT,
  description                 TEXT,
  sample_organism             TEXT,
  bio_target                  TEXT,
  bio_target_type             TEXT,
  bio_target_source           TEXT,
  bio_target_confidence       REAL,
  sample_type                 TEXT,
  sample_subtype              TEXT,
  sample_name                 TEXT,
  grid_spacing                TEXT,
  grid_dimensions             TEXT,
  grid_axes                   TEXT,
  grid_spacing_unit           TEXT,
  grid_dimensions_unit        TEXT,
  mitochondria_in_layer_names INTEGER NOT NULL DEFAULT 0,
  -- scraped download hints (from PostgREST / LLM)
  download_volume_url         TEXT,
  download_volume_image_stack TEXT,
  download_mito_mask_url      TEXT,
  download_mito_mask_quality  TEXT,
  download_mito_mask_segmentation_kind TEXT,
  download_mito_prediction_url TEXT,
  -- S3-probed paths (authoritative; written by s3_probe.py after scraping)
  s3_probe_img_path           TEXT,
  s3_probe_seg_path           TEXT,
  s3_probe_voxel_size         TEXT,   -- JSON array e.g. "[8.0, 8.0, 8.0]"
  s3_probe_label_names        TEXT,   -- JSON array of all label dirs found
  s3_probe_error              TEXT    -- non-null when probe failed
);

CREATE TABLE IF NOT EXISTS dataset_sources (
  dataset_id  INTEGER NOT NULL,
  source_url  TEXT    NOT NULL,
  PRIMARY KEY (dataset_id, source_url),
  FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS dataset_layers (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  dataset_id        INTEGER NOT NULL,
  layer_role        TEXT    NOT NULL,
  layer_kind        TEXT    NOT NULL,
  layer_name        TEXT    NOT NULL,
  layer_format      TEXT,
  layer_stage       TEXT,
  layer_url         TEXT,
  layer_content_type TEXT,
  semantic          INTEGER NOT NULL DEFAULT 0,
  raw_token         TEXT,
  FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_layers_dataset ON dataset_layers(dataset_id);
CREATE INDEX IF NOT EXISTS idx_layers_name    ON dataset_layers(layer_name);
CREATE INDEX IF NOT EXISTS idx_layers_role    ON dataset_layers(layer_role);

-- Denormalized download resolution (filled after S3 probe via materialize_resolved_paths)
CREATE TABLE IF NOT EXISTS dataset_resolved (
  dataset_id           INTEGER PRIMARY KEY,
  dataset_name         TEXT    NOT NULL UNIQUE,
  resolved_em_path     TEXT,
  resolved_mito_seg_path TEXT,
  resolved_voxel_nm    TEXT,
  path_source          TEXT,
  ready_labeled        INTEGER NOT NULL DEFAULT 0,
  ready_em_only        INTEGER NOT NULL DEFAULT 0,
  FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_resolved_ready_labeled ON dataset_resolved(ready_labeled);
CREATE INDEX IF NOT EXISTS idx_resolved_source ON dataset_resolved(path_source);
"""

_RESOLVED_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dataset_resolved (
  dataset_id           INTEGER PRIMARY KEY,
  dataset_name         TEXT    NOT NULL UNIQUE,
  resolved_em_path     TEXT,
  resolved_mito_seg_path TEXT,
  resolved_voxel_nm    TEXT,
  path_source          TEXT,
  ready_labeled        INTEGER NOT NULL DEFAULT 0,
  ready_em_only        INTEGER NOT NULL DEFAULT 0,
  FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_resolved_ready_labeled ON dataset_resolved(ready_labeled);
CREATE INDEX IF NOT EXISTS idx_resolved_source ON dataset_resolved(path_source);
"""

_PROBE_COLS = [
    ("s3_probe_img_path",    "TEXT"),
    ("s3_probe_seg_path",    "TEXT"),
    ("s3_probe_voxel_size",  "TEXT"),
    ("s3_probe_label_names", "TEXT"),
    ("s3_probe_error",       "TEXT"),
]
_DOWNLOAD_COLS = [
    ("download_volume_url",          "TEXT"),
    ("download_volume_image_stack",  "TEXT"),
    ("download_mito_mask_url",       "TEXT"),
    ("download_mito_mask_quality",   "TEXT"),
    ("download_mito_mask_segmentation_kind", "TEXT"),
    ("download_mito_prediction_url", "TEXT"),
]
_BIO_COLS = [
    ("bio_target", "TEXT"),
    ("bio_target_type", "TEXT"),
    ("bio_target_source", "TEXT"),
    ("bio_target_confidence", "REAL"),
]
_LAYER_COLS = [
    ("layer_url",          "TEXT"),
    ("layer_content_type", "TEXT"),
]


def connect(db_path: Path) -> sqlite3.Connection:
    # Use a busy timeout + WAL so short concurrent readers/writers do not fail
    # with immediate "database is locked" during Stage-2 materialization/probing.
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA busy_timeout = 30000;")
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
    except Exception:
        # Keep compatibility with filesystems/SQLite builds that reject WAL.
        pass
    conn.row_factory = sqlite3.Row
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    """Add any columns introduced after initial schema creation."""
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(datasets)")
    dcols = {row[1] for row in cur.fetchall()}
    for col, typ in _DOWNLOAD_COLS + _PROBE_COLS + _BIO_COLS:
        if col not in dcols:
            cur.execute(f"ALTER TABLE datasets ADD COLUMN {col} {typ}")

    cur.execute("PRAGMA table_info(dataset_layers)")
    lcols = {row[1] for row in cur.fetchall()}
    for col, typ in _LAYER_COLS:
        if col not in lcols:
            cur.execute(f"ALTER TABLE dataset_layers ADD COLUMN {col} {typ}")

    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='dataset_resolved'"
    )
    if cur.fetchone() is None:
        cur.executescript(_RESOLVED_TABLE_SQL)
    conn.commit()


def build_sqlite_db(records: list[DatasetRecord], sqlite_path: Path) -> dict[str, int]:
    """Create/replace the SQLite DB from parsed DatasetRecord list."""
    if not records:
        raise RuntimeError("No dataset rows parsed from markdown.")

    conn = connect(sqlite_path)
    cur = conn.cursor()
    cur.executescript(DDL)
    migrate(conn)

    cur.execute("DELETE FROM dataset_layers")
    cur.execute("DELETE FROM dataset_sources")
    cur.execute("DELETE FROM dataset_resolved")
    cur.execute("DELETE FROM datasets")

    n_layers = n_sources = n_mito = 0
    for d in records:
        if (d.download_mito_mask_quality or "").strip().lower() == "good":
            n_mito += 1
        dataset_id = _insert_dataset(cur, d)
        if d.source_url:
            cur.execute(
                "INSERT OR IGNORE INTO dataset_sources(dataset_id, source_url) VALUES (?, ?)",
                (dataset_id, d.source_url),
            )
            n_sources += 1
        for layer in d.layers:
            cur.execute(
                """
                INSERT INTO dataset_layers(
                  dataset_id, layer_role, layer_kind, layer_name,
                  layer_format, layer_stage, layer_url, layer_content_type,
                  semantic, raw_token
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    layer.layer_role,
                    layer.layer_kind,
                    layer.layer_name,
                    layer.layer_format,
                    layer.layer_stage,
                    layer.layer_url,
                    layer.layer_content_type,
                    1 if layer.semantic else 0,
                    layer.raw_token,
                ),
            )
            n_layers += 1

    conn.commit()
    conn.close()
    return {"datasets": len(records), "layers": n_layers,
            "source_links": n_sources, "mito_datasets": n_mito}


def _insert_dataset(cur: sqlite3.Cursor, d: DatasetRecord) -> int:
    # User-facing catalog semantics: treat sample subtype as the primary type when present.
    effective_sample_type = (d.sample_subtype or "").strip() or d.sample_type
    bio = classify_bio_target(
        dataset_name=d.dataset_name or "",
        sample_type=effective_sample_type or "",
        sample_subtype=d.sample_subtype or "",
        sample_name=d.sample_name or "",
        description=d.description or "",
        layer_names=[ly.layer_name for ly in d.layers],
    )
    cur.execute(
        """
        INSERT INTO datasets(
          dataset_name, stage, segmentation_challenge, description,
          sample_organism, bio_target, bio_target_type, bio_target_source, bio_target_confidence,
          sample_type, sample_subtype, sample_name,
          grid_spacing, grid_dimensions, grid_axes, grid_spacing_unit, grid_dimensions_unit,
          mitochondria_in_layer_names,
          download_volume_url, download_volume_image_stack,
          download_mito_mask_url, download_mito_mask_quality,
          download_mito_mask_segmentation_kind, download_mito_prediction_url
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(dataset_name) DO UPDATE SET
          stage                       = excluded.stage,
          segmentation_challenge      = excluded.segmentation_challenge,
          description                 = excluded.description,
          sample_organism             = excluded.sample_organism,
          bio_target                  = excluded.bio_target,
          bio_target_type             = excluded.bio_target_type,
          bio_target_source           = excluded.bio_target_source,
          bio_target_confidence       = excluded.bio_target_confidence,
          sample_type                 = excluded.sample_type,
          sample_subtype              = excluded.sample_subtype,
          sample_name                 = excluded.sample_name,
          grid_spacing                = excluded.grid_spacing,
          grid_dimensions             = excluded.grid_dimensions,
          grid_axes                   = excluded.grid_axes,
          grid_spacing_unit           = excluded.grid_spacing_unit,
          grid_dimensions_unit        = excluded.grid_dimensions_unit,
          mitochondria_in_layer_names = excluded.mitochondria_in_layer_names,
          download_volume_url         = excluded.download_volume_url,
          download_volume_image_stack = excluded.download_volume_image_stack,
          download_mito_mask_url      = excluded.download_mito_mask_url,
          download_mito_mask_quality  = excluded.download_mito_mask_quality,
          download_mito_mask_segmentation_kind = excluded.download_mito_mask_segmentation_kind,
          download_mito_prediction_url= excluded.download_mito_prediction_url
        """,
        (
            d.dataset_name, d.stage, d.segmentation_challenge, d.description,
            d.sample_organism, bio.target, bio.target_type, bio.source, bio.confidence,
            effective_sample_type, d.sample_subtype, d.sample_name,
            d.grid_spacing, d.grid_dimensions, d.grid_axes,
            d.grid_spacing_unit, d.grid_dimensions_unit,
            1 if d.mitochondria_in_layer_names else 0,
            d.download_volume_url, d.download_volume_image_stack,
            d.download_mito_mask_url, d.download_mito_mask_quality,
            d.download_mito_mask_segmentation_kind, d.download_mito_prediction_url,
        ),
    )
    cur.execute("SELECT id FROM datasets WHERE dataset_name = ?", (d.dataset_name,))
    r = cur.fetchone()
    if r is None:
        raise RuntimeError(f"Failed to resolve dataset id for {d.dataset_name}")
    return int(r[0])


def write_probe_results(db_path: Path, probe_results: dict[str, dict]) -> int:
    """
    Write S3 probe results back into the datasets table.
    Returns number of rows updated.
    """
    import json

    conn = connect(db_path)
    migrate(conn)
    cur = conn.cursor()
    updated = 0
    for dataset_id, probe in probe_results.items():
        err = probe.get("error")
        img = probe.get("em_image_path") or ""
        # Downstream policy: prediction-only mito masks are not training-ready by default.
        seg = probe.get("mito_seg_path") or ""
        voxel = probe.get("voxel_size_nm")
        labels = probe.get("label_names", [])
        cur.execute(
            """
            UPDATE datasets SET
              s3_probe_img_path    = ?,
              s3_probe_seg_path    = ?,
              s3_probe_voxel_size  = ?,
              s3_probe_label_names = ?,
              s3_probe_error       = ?
            WHERE dataset_name = ?
            """,
            (
                img or None,
                seg or None,
                json.dumps(voxel) if voxel else None,
                json.dumps(labels) if labels else None,
                err or None,
                dataset_id,
            ),
        )
        updated += cur.rowcount
    conn.commit()
    conn.close()
    return updated


def effective_paths(row: Any) -> tuple[str, str]:
    """
    Return (img_path, seg_path) for a dataset row, preferring S3-probed paths
    over scraped ones.
    """
    img = (row["s3_probe_img_path"] or "").strip() or \
          (row["download_volume_url"] or "").strip()
    seg = (row["s3_probe_seg_path"] or "").strip() or \
          (row["download_mito_mask_url"] or "").strip()
    return img, seg
