"""Create and populate a small SQLite DB mirroring probe JSON keys."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS scrape_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  site TEXT NOT NULL,
  url TEXT NOT NULL,
  last_run_at_iso TEXT,
  probe_json_path TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS datasets (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  scrape_id INTEGER NOT NULL,
  dataset_key TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  FOREIGN KEY (scrape_id) REFERENCES scrape_runs(id)
);
CREATE INDEX IF NOT EXISTS idx_datasets_scrape ON datasets(scrape_id);
"""


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def ingest_probe(
    conn: sqlite3.Connection,
    probe_path: Path,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    data = json.loads(probe_path.read_text(encoding="utf-8"))
    site = str(data.get("site") or probe_path.stem)
    url = str(data.get("url") or "")
    last_run = str(data.get("last_run_at_iso") or "")
    if project_root is not None:
        try:
            rel = probe_path.resolve().relative_to(project_root.resolve()).as_posix()
        except ValueError:
            rel = probe_path.resolve().as_posix()
    else:
        rel = probe_path.as_posix()
    cur = conn.cursor()
    cur.execute("DELETE FROM datasets WHERE scrape_id IN (SELECT id FROM scrape_runs WHERE probe_json_path=?)", (rel,))
    cur.execute("DELETE FROM scrape_runs WHERE probe_json_path=?", (rel,))
    cur.execute(
        "INSERT INTO scrape_runs (site, url, last_run_at_iso, probe_json_path) VALUES (?,?,?,?)",
        (site, url, last_run, rel),
    )
    scrape_id = int(cur.lastrowid)
    datasets = data.get("datasets") or {}
    if isinstance(datasets, dict):
        for key, payload in datasets.items():
            cur.execute(
                "INSERT INTO datasets (scrape_id, dataset_key, payload_json) VALUES (?,?,?)",
                (scrape_id, str(key), json.dumps(payload, ensure_ascii=False)),
            )
    conn.commit()
    return {"scrape_id": scrape_id, "datasets_written": len(datasets) if isinstance(datasets, dict) else 0}
