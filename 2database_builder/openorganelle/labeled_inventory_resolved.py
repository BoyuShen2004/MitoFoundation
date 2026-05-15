"""Labeled OpenOrganelle inventory from ``dataset_resolved`` (Stage-2 SQLite).

Used by the Stage-3 OpenOrganelle agent and the registry provider so filtering
and path fallbacks stay in one place. This is **not** the same as
:func:`download_inventory.inventory_from_db`, which builds paths from
``datasets`` + ``dataset_layers`` rows.

Requires ``mitoFoundation2/2database_builder`` on ``sys.path`` for
``from openorganelle.labeled_inventory_resolved import ...``.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def labeled_mito_inventory_from_resolved_db(
    db_path: Path,
    *,
    mito_only: bool = True,
    require_paths: bool = True,
) -> dict[str, dict[str, Any]]:
    """Read resolved download paths from a database_builder catalog (joined view)."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT d.dataset_name, r.resolved_em_path, r.resolved_mito_seg_path,
               r.resolved_voxel_nm, r.path_source, r.ready_labeled, r.ready_em_only,
               d.grid_dimensions, d.grid_axes,
               d.download_mito_mask_quality, d.download_mito_mask_url,
               d.download_volume_url, d.s3_probe_img_path
        FROM dataset_resolved r
        JOIN datasets d ON d.id = r.dataset_id
        ORDER BY d.dataset_name
        """
    ).fetchall()
    conn.close()
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        name = (r["dataset_name"] or "").strip()
        if not name:
            continue
        img = (r["resolved_em_path"] or "").strip()
        seg = (r["resolved_mito_seg_path"] or "").strip()
        img_fallback_probe = (r["s3_probe_img_path"] or "").strip()
        img_fallback_scraped = (r["download_volume_url"] or "").strip()
        seg_fallback = (r["download_mito_mask_url"] or "").strip()
        if not img:
            img = img_fallback_probe or img_fallback_scraped
        if mito_only and not seg:
            seg = seg_fallback
        if require_paths and not img:
            continue
        if mito_only and require_paths and not seg:
            continue
        if mito_only:
            qual = (r["download_mito_mask_quality"] or "").strip().lower()
            mask_url_l = (r["download_mito_mask_url"] or "").strip().lower()
            seg_l = seg.lower()
            if ("pred" in seg_l or "inference" in seg_l) and seg_fallback:
                seg = seg_fallback
                seg_l = seg.lower()
            if qual != "good":
                continue
            if "pred" in seg_l or "inference" in seg_l:
                continue
            if "pred" in mask_url_l or "inference" in mask_url_l:
                continue
        voxel = [8.0, 8.0, 8.0]
        if r["resolved_voxel_nm"]:
            try:
                voxel = json.loads(r["resolved_voxel_nm"])
            except Exception:
                pass
        entry: dict[str, Any] = {
            "dataset_name": name,
            "img_path": img,
            "seg_path": seg,
            "voxel_size_nm": voxel,
            "_source": r["path_source"] or "resolved",
        }
        gd = (r["grid_dimensions"] or "").strip() if r["grid_dimensions"] else ""
        ga = (r["grid_axes"] or "").strip() if r["grid_axes"] else ""
        if gd:
            entry["grid_dimensions"] = r["grid_dimensions"]
        if ga:
            entry["grid_axes"] = r["grid_axes"]
        out[name] = entry
    return out
