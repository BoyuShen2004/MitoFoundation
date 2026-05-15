"""SQLite schema, migrations, and connection helpers for the central registry."""

from __future__ import annotations

import sqlite3
from pathlib import Path

# schema.py lives at <repo>/agent/orchestration/registry/schema.py
# so repo root is parents[3], not parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY_PATH = _REPO_ROOT / "data" / "registry.sqlite"

DDL = """
PRAGMA journal_mode = DELETE;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS providers (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  name       TEXT    NOT NULL UNIQUE,
  base_url   TEXT,
  created_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS datasets (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  provider_id     INTEGER NOT NULL REFERENCES providers(id) ON DELETE CASCADE,
  stable_id       TEXT    NOT NULL,
  display_name    TEXT,
  first_seen_at   TEXT    NOT NULL,
  last_seen_at    TEXT    NOT NULL,
  last_changed_at TEXT,
  metadata_json   TEXT,
  UNIQUE (provider_id, stable_id)
);
CREATE INDEX IF NOT EXISTS idx_datasets_provider ON datasets(provider_id);
CREATE INDEX IF NOT EXISTS idx_datasets_stable   ON datasets(stable_id);

CREATE TABLE IF NOT EXISTS assets (
  id                  INTEGER PRIMARY KEY AUTOINCREMENT,
  dataset_id          INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  asset_type          TEXT    NOT NULL,
  remote_url          TEXT    NOT NULL,
  content_fingerprint TEXT,
  byte_size           INTEGER,
  last_checked_at     TEXT,
  UNIQUE (dataset_id, asset_type, remote_url)
);
CREATE INDEX IF NOT EXISTS idx_assets_dataset ON assets(dataset_id);
CREATE INDEX IF NOT EXISTS idx_assets_type    ON assets(asset_type);

CREATE TABLE IF NOT EXISTS downloads (
  id                  INTEGER PRIMARY KEY AUTOINCREMENT,
  asset_id            INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
  local_path          TEXT    NOT NULL,
  status              TEXT    NOT NULL DEFAULT 'pending',
  started_at          TEXT,
  finished_at         TEXT,
  error               TEXT,
  verified_fingerprint TEXT
);
CREATE INDEX IF NOT EXISTS idx_downloads_asset  ON downloads(asset_id);
CREATE INDEX IF NOT EXISTS idx_downloads_status ON downloads(status);

CREATE TABLE IF NOT EXISTS preprocess_configs (
  config_fingerprint TEXT PRIMARY KEY,
  config_json        TEXT NOT NULL,
  created_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS preprocess_runs (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  dataset_id        INTEGER REFERENCES datasets(id) ON DELETE SET NULL,
  input_fingerprint TEXT    NOT NULL,
  config_fingerprint TEXT   NOT NULL REFERENCES preprocess_configs(config_fingerprint),
  status            TEXT    NOT NULL DEFAULT 'pending',
  output_paths_json TEXT,
  started_at        TEXT,
  finished_at       TEXT,
  error             TEXT
);
CREATE INDEX IF NOT EXISTS idx_preprocess_input  ON preprocess_runs(input_fingerprint);
CREATE INDEX IF NOT EXISTS idx_preprocess_config ON preprocess_runs(config_fingerprint);
CREATE INDEX IF NOT EXISTS idx_preprocess_status ON preprocess_runs(status);

-- Download batch provenance: one record per download run (e.g. openorganelle_mito_20260420_012621).
CREATE TABLE IF NOT EXISTS download_batches (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  batch_id        TEXT    NOT NULL UNIQUE,
  provider        TEXT    NOT NULL,
  profile_hash    TEXT,
  profile_json    TEXT,
  run_folder      TEXT,
  download_asset_completions INTEGER NOT NULL DEFAULT 0,
  status          TEXT    NOT NULL DEFAULT 'in_progress',
  created_at      TEXT    NOT NULL,
  finished_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_batches_provider ON download_batches(provider);
CREATE INDEX IF NOT EXISTS idx_batches_status   ON download_batches(status);

-- Item-level tracking per dataset×asset_type within a batch.
-- local_path points to the final training output file/dir.
-- status: pending | present | missing_or_deleted_local | failed
CREATE TABLE IF NOT EXISTS batch_items (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  batch_db_id     INTEGER NOT NULL REFERENCES download_batches(id) ON DELETE CASCADE,
  dataset_id      INTEGER REFERENCES datasets(id) ON DELETE SET NULL,
  stable_id       TEXT    NOT NULL,
  asset_type      TEXT    NOT NULL,
  local_path      TEXT,
  status          TEXT    NOT NULL DEFAULT 'pending',
  completed_at    TEXT,
  UNIQUE(batch_db_id, stable_id, asset_type)
);
CREATE INDEX IF NOT EXISTS idx_batch_items_batch  ON batch_items(batch_db_id);
CREATE INDEX IF NOT EXISTS idx_batch_items_status ON batch_items(status);
CREATE INDEX IF NOT EXISTS idx_batch_items_stable ON batch_items(stable_id);

-- Persistent delete history (independent of current batch_item status).
CREATE TABLE IF NOT EXISTS deletion_events (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  provider        TEXT    NOT NULL DEFAULT 'OpenOrganelle',
  stable_id       TEXT    NOT NULL,
  asset_type      TEXT,
  local_path      TEXT    NOT NULL,
  deleted_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_deletion_events_provider ON deletion_events(provider);
CREATE INDEX IF NOT EXISTS idx_deletion_events_stable   ON deletion_events(stable_id);
CREATE INDEX IF NOT EXISTS idx_deletion_events_deleted  ON deletion_events(deleted_at);
"""

_MIGRATION_STMTS: list[str] = [
    # v2: per-download profile hash so n_crops changes trigger new downloads.
    """
    ALTER TABLE downloads ADD COLUMN download_profile_hash TEXT;
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_downloads_profile
    ON downloads(asset_id, download_profile_hash);
    """,
    # v3: soft-hide datasets from training without deleting files.
    """
    ALTER TABLE datasets ADD COLUMN hidden_from_training INTEGER NOT NULL DEFAULT 0;
    """,
    # v4: per-batch asset download count (em + seg) summed for Stage-0 Log.
    """
    ALTER TABLE download_batches ADD COLUMN download_asset_completions INTEGER NOT NULL DEFAULT 0;
    """,
]


def connect(path: Path | None = None) -> sqlite3.Connection:
    p = path or DEFAULT_REGISTRY_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = DELETE;")
    conn.row_factory = sqlite3.Row
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)
    for stmt in _MIGRATION_STMTS:
        try:
            conn.executescript(stmt)
        except sqlite3.OperationalError:
            pass
    conn.commit()


def open_registry(path: Path | None = None) -> sqlite3.Connection:
    conn = connect(path)
    migrate(conn)
    return conn
