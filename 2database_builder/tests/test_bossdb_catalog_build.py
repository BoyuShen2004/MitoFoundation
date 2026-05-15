from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path


_SCHEMA_ROOT = Path(__file__).resolve().parents[1]
if str(_SCHEMA_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCHEMA_ROOT))

from master.catalog_build import run_catalog_build


def test_run_catalog_build_generates_bossdb_outputs(tmp_path: Path) -> None:
    md_dir = tmp_path / "md"
    db_dir = tmp_path / "db"
    schema_dir = tmp_path / "schema"
    md_dir.mkdir(parents=True)

    # Minimal BossDB-shaped markdown: slug list + one dataset row.
    (md_dir / "BossDB.md").write_text(
        "\n".join(
            [
                "# BossDB",
                "### All datasets on the site (catalog slug list)",
                "1. **`bossdb_proj_001`**",
                "",
                "- dataset_name=bossdb_proj_001 | stage=raw",
            ]
        ),
        encoding="utf-8",
    )

    inv_path = tmp_path / "inventory_bossdb.jsonl"
    inv_records = [
        {
            "source_site": "BossDB",
            "project_id": "colA/exp1/em",
            "project_name": "colA/exp1/em",
            "collection": "colA",
            "experiment": "exp1",
            "channel": "em",
            "uri": "bossdb://colA/exp1/em",
            "channel_type": "image",
            "data_type": "uint8",
            "labeling": "",
            "voxel_size_nm": [16.0, 16.0, 16.0],
            "organism": "mouse",
            "modality": "em",
            "tags": [],
            "description": "",
            "public": True,
            "raw_json": {},
        },
        {
            "source_site": "BossDB",
            "project_id": "colA/exp1/mito_seg",
            "project_name": "colA/exp1/mito_seg",
            "collection": "colA",
            "experiment": "exp1",
            "channel": "mito_seg",
            "uri": "bossdb://colA/exp1/mito_seg",
            "channel_type": "annotation",
            "data_type": "uint64",
            "labeling": "mitochondria",
            "voxel_size_nm": [16.0, 16.0, 16.0],
            "organism": "mouse",
            "modality": "em",
            "tags": [],
            "description": "",
            "public": True,
            "raw_json": {},
        },
    ]
    inv_path.write_text("\n".join(json.dumps(r) for r in inv_records) + "\n", encoding="utf-8")

    old_inv = os.environ.get("BOSSDB_INVENTORY_PATH")
    os.environ["BOSSDB_INVENTORY_PATH"] = str(inv_path)
    try:
        res = run_catalog_build(
            md_dir=md_dir,
            db_dir=db_dir,
            schema_md_dir=schema_dir,
            sites=["BossDB"],
            skip_s3_probe=True,
            sync_registry=False,
        )
    finally:
        if old_inv is None:
            os.environ.pop("BOSSDB_INVENTORY_PATH", None)
        else:
            os.environ["BOSSDB_INVENTORY_PATH"] = old_inv
    assert len(res) == 1
    assert (db_dir / "BossDB.db").is_file()
    assert (schema_dir / "BossDB.schema.md").is_file()
    assert not (db_dir / "BossDB_pairs.db").exists()
    assert not (schema_dir / "BossDB_pairs.schema.md").exists()
    c = sqlite3.connect(str(db_dir / "BossDB.db"))
    try:
        n = int(c.execute("SELECT COUNT(*) FROM datasets WHERE lower(coalesce(download_mito_mask_quality,''))='good'").fetchone()[0])
        assert n == 0
    finally:
        c.close()

