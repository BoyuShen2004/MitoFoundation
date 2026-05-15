"""Build ``BossDB_pairs.db`` + schema markdown from paired inventory rows."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from openorganelle.bioclass import classify_bio_target

from .db import BOSSDB_PAIRS_CATALOG_RESET_SCRIPT
from .parser import load_bossdb_pairs
from .report import render_bossdb_pairs_schema_markdown


def build_bossdb_pairs_catalog(
    *,
    out_db: Path,
    out_schema_md: Path,
    inventory_path: Path | None = None,
) -> dict[str, int]:
    """Write ``BossDB_pairs.db`` + schema markdown from inventory pairs."""
    pairs = load_bossdb_pairs(inventory_path=inventory_path)

    out_db.parent.mkdir(parents=True, exist_ok=True)
    out_schema_md.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(out_db))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(BOSSDB_PAIRS_CATALOG_RESET_SCRIPT)
        for p in pairs:
            did_name = str(p.get("project_id") or "").strip()
            if not did_name:
                continue
            img_uri = str(p.get("img_uri") or "").strip()
            seg_uri = str(p.get("seg_uri") or "").strip()
            img_ch = str(p.get("img_channel") or "").strip() or "image"
            seg_ch = str(p.get("seg_channel") or "").strip() or "mito_seg"
            voxel = p.get("voxel_size_nm") or []
            bio = classify_bio_target(
                dataset_name=did_name,
                sample_type="bossdb",
                sample_subtype="",
                sample_name=did_name,
                description=" ".join(
                    x
                    for x in [
                        str(p.get("description") or "").strip(),
                        str(p.get("img_channel") or "").strip(),
                        str(p.get("seg_channel") or "").strip(),
                    ]
                    if x
                ),
                layer_names=[img_ch, seg_ch],
            )
            sample_type = bio.target_type if bio.target_type and bio.target_type != "unknown" else "unknown"
            sample_subtype = bio.target if bio.target and bio.target != "unknown" else ""
            conn.execute(
                """
                INSERT INTO datasets(
                  dataset_name, stage, description, sample_organism,
                  bio_target, bio_target_type, bio_target_source, bio_target_confidence,
                  sample_type, sample_subtype, sample_name,
                  mitochondria_in_layer_names, download_volume_url, download_mito_mask_url,
                  download_mito_mask_quality, download_mito_mask_segmentation_kind,
                  s3_probe_img_path, s3_probe_seg_path, s3_probe_voxel_size
                ) VALUES (?, 'prod', ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, 'good', 'annotation', ?, ?, ?)
                """,
                (
                    did_name,
                    "BossDB pair-ready (image + mito annotation)",
                    str(p.get("organism") or ""),
                    bio.target,
                    bio.target_type,
                    bio.source,
                    bio.confidence,
                    sample_type,
                    sample_subtype,
                    did_name,
                    img_uri,
                    seg_uri,
                    img_uri,
                    seg_uri,
                    json.dumps(voxel, ensure_ascii=False) if voxel else None,
                ),
            )
            row = conn.execute("SELECT id FROM datasets WHERE dataset_name = ?", (did_name,)).fetchone()
            if row is None:
                continue
            did = int(row["id"])
            conn.execute(
                "INSERT OR IGNORE INTO dataset_sources(dataset_id, source_url) VALUES (?, ?)",
                (did, f"https://api.bossdb.io/v1/{did_name}"),
            )
            conn.execute(
                """
                INSERT INTO dataset_layers(
                  dataset_id, layer_role, layer_kind, layer_name, layer_url, layer_content_type, raw_token
                ) VALUES (?, 'raw_or_other_layers', 'image', ?, ?, 'em', ?)
                """,
                (did, img_ch, img_uri, f"image:{img_ch}[url={img_uri}][content_type=em]"),
            )
            conn.execute(
                """
                INSERT INTO dataset_layers(
                  dataset_id, layer_role, layer_kind, layer_name, layer_url, layer_content_type, raw_token
                ) VALUES (?, 'ground_truths', 'image', ?, ?, 'segmentation', ?)
                """,
                (did, seg_ch, seg_uri, f"image:{seg_ch}[url={seg_uri}][content_type=segmentation]"),
            )
            conn.execute(
                """
                INSERT INTO dataset_resolved(
                  dataset_id, dataset_name, resolved_em_path, resolved_mito_seg_path,
                  resolved_voxel_nm, path_source, ready_labeled, ready_em_only
                ) VALUES (?, ?, ?, ?, ?, 'bossdb_inventory_pair', 1, 0)
                """,
                (
                    did,
                    did_name,
                    img_uri,
                    seg_uri,
                    json.dumps(voxel, ensure_ascii=False) if voxel else None,
                ),
            )
        conn.commit()
    finally:
        conn.close()

    out_schema_md.write_text(
        render_bossdb_pairs_schema_markdown(out_db=out_db, n_pairs=len(pairs)),
        encoding="utf-8",
    )
    return {"datasets": len(pairs), "layers": len(pairs) * 2, "mito_datasets": len(pairs)}
