"""
Derive ready-to-use img/seg paths from a database_builder SQLite database.
S3-probed paths (written by s3_probe.py) take priority over scraped hints.
"""

from __future__ import annotations

import ast
import json
import os
import sqlite3
import sys
from pathlib import Path
import time
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCHEMA_ROOT = _REPO_ROOT / "2database_builder"
if str(_SCHEMA_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCHEMA_ROOT))
from openorganelle.db import connect, effective_paths, migrate

JANELIA_BUCKET_PREFIX = "s3://janelia-cosem-datasets"
DEFAULT_EM_SCALE  = "s1"   # 2× downsampled — safe default
DEFAULT_SEG_SCALE = "s0"   # full resolution segmentation


# ── helpers ────────────────────────────────────────────────────────────────────

def _parse_num_vector(text: str | None) -> list[float] | None:
    if not text or not str(text).strip():
        return None
    try:
        v = ast.literal_eval(text.strip())
    except (SyntaxError, ValueError):
        try:
            v = json.loads(text.strip())
        except Exception:
            return None
    if isinstance(v, (list, tuple)) and v and all(isinstance(x, (int, float)) for x in v):
        return [float(x) for x in v]
    return None


def _best_voxel(row: sqlite3.Row) -> list[float] | None:
    """Prefer S3-probed voxel size; fall back to scraped grid_spacing."""
    v = _parse_num_vector(row["s3_probe_voxel_size"] if "s3_probe_voxel_size" in row.keys() else None)
    if v:
        return v
    return _parse_num_vector(row["grid_spacing"])


# ── path scoring for layer fallback ───────────────────────────────────────────

def _score_seg_layer(row: sqlite3.Row) -> int:
    role = (row["layer_role"] or "").lower()
    kind = (row["layer_kind"] or "").lower()
    name = (row["layer_name"] or "").lower()
    fmt  = (row["layer_format"] or "").lower()
    if kind != "image":
        return -1000
    score = 0
    if role == "ground_truths":   score += 100
    elif role == "predictions":   score += 10
    if fmt == "n5":               score += 50
    elif fmt == "zarr":           score += 30
    if row["layer_url"]:          score += 25
    if "mito_seg" in name or "mito-mem_seg" in name: score += 40
    elif "empanada-mito_seg" in name:                score += 35
    elif "mito" in name and "seg" in name:           score += 20
    if "pred" in name or "inference" in name: score -= 40
    return score


def _allow_prediction_masks() -> bool:
    return os.getenv("ALLOW_PREDICTION_MASKS", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _is_prediction_like_url(u: str | None) -> bool:
    s = (u or "").strip().lower()
    return ("inference" in s) or ("pred" in s) or ("prediction" in s)


def _is_prediction_like_layer(row: sqlite3.Row) -> bool:
    role = (row["layer_role"] or "").lower()
    name = (row["layer_name"] or "").lower()
    url = (row["layer_url"] or "").lower()
    return (
        role == "predictions"
        or "pred" in name
        or "prediction" in name
        or "inference" in name
        or "pred" in url
        or "inference" in url
    )


def _pick_best_seg_layer(
    layers: list[sqlite3.Row], *, allow_prediction: bool
) -> sqlite3.Row | None:
    mito = [r for r in layers
            if (r["layer_kind"] or "").lower() == "image"
            and "mito" in (r["layer_name"] or "").lower()]
    if not allow_prediction:
        mito = [r for r in mito if not _is_prediction_like_layer(r)]
    return max(mito, key=_score_seg_layer) if mito else None


def _catalog_fields(dataset_row: sqlite3.Row) -> dict[str, Any]:
    """Pass through OpenOrganelle catalog grid (same semantics as scraped OpenOrganelle.md)."""
    out: dict[str, Any] = {}
    keys = dataset_row.keys()
    gd = (dataset_row["grid_dimensions"] or "").strip() if "grid_dimensions" in keys else ""
    ga = (dataset_row["grid_axes"] or "").strip() if "grid_axes" in keys else ""
    if gd:
        out["grid_dimensions"] = dataset_row["grid_dimensions"]
    if ga:
        out["grid_axes"] = dataset_row["grid_axes"]
    return out


# ── per-dataset path resolution ────────────────────────────────────────────────

def resolve_paths(dataset_row: sqlite3.Row, layers: list[sqlite3.Row]) -> dict[str, Any]:
    allow_pred = _allow_prediction_masks()
    name   = dataset_row["dataset_name"]
    voxel  = _best_voxel(dataset_row)
    unit   = dataset_row["grid_spacing_unit"]

    # 1. S3-probed paths (most authoritative)
    img_probe = (dataset_row["s3_probe_img_path"] or "").strip() if "s3_probe_img_path" in dataset_row.keys() else ""
    seg_probe = (dataset_row["s3_probe_seg_path"] or "").strip() if "s3_probe_seg_path" in dataset_row.keys() else ""
    if img_probe and seg_probe:
        return dict(
            dataset_name=name,
            img_path=img_probe,
            seg_path=seg_probe,
            voxel_size_nm=voxel,
            grid_spacing_unit=unit,
            _source="s3_probe",
            **_catalog_fields(dataset_row),
        )

    # 2. Scraped PostgREST download_* columns
    vol_u  = (dataset_row["download_volume_url"] or "").strip()
    mito_u = (dataset_row["download_mito_mask_url"] or "").strip()
    pred_u = (dataset_row["download_mito_prediction_url"] or "").strip()
    mask_q = (dataset_row["download_mito_mask_quality"] or "").strip().lower() if "download_mito_mask_quality" in dataset_row.keys() else ""
    if mito_u and (not allow_pred):
        # Guard against prediction-only masks accidentally placed in mask field.
        if _is_prediction_like_url(mito_u) and mask_q != "good":
            mito_u = ""
    if vol_u and mito_u:
        return dict(
            dataset_name=name,
            img_path=vol_u,
            seg_path=mito_u,
            voxel_size_nm=voxel,
            grid_spacing_unit=unit,
            _source="scraped_postgrest",
            **_catalog_fields(dataset_row),
        )
    if allow_pred and vol_u and pred_u:
        return dict(
            dataset_name=name,
            img_path=vol_u,
            seg_path=pred_u,
            voxel_size_nm=voxel,
            grid_spacing_unit=unit,
            _source="scraped_postgrest_pred_only",
            **_catalog_fields(dataset_row),
        )

    # 3. Layer URL from DB
    seg_layer = _pick_best_seg_layer(layers, allow_prediction=allow_pred)
    if seg_layer and seg_layer["layer_url"]:
        img = vol_u or (
            f"{JANELIA_BUCKET_PREFIX}/{name}/{name}.n5/em/fibsem-uint8/{DEFAULT_EM_SCALE}"
        )
        return dict(
            dataset_name=name,
            img_path=img,
            seg_path=seg_layer["layer_url"],
            voxel_size_nm=voxel,
            grid_spacing_unit=unit,
            _source="layer_url",
            **_catalog_fields(dataset_row),
        )

    # 4. Template fallback (least reliable — flag it)
    if seg_layer:
        seg_name = seg_layer["layer_name"]
        root = f"{JANELIA_BUCKET_PREFIX}/{name}/{name}.n5"
        return dict(
            dataset_name=name,
            img_path=vol_u or f"{root}/em/fibsem-uint8/{DEFAULT_EM_SCALE}",
            seg_path=f"{root}/labels/{seg_name}/{DEFAULT_SEG_SCALE}",
            voxel_size_nm=voxel,
            grid_spacing_unit=unit,
            _source="template_fallback",
            _note="Unverified template — run database_builder with S3 probe to confirm.",
            **_catalog_fields(dataset_row),
        )

    return dict(
        dataset_name=name,
        img_path="",
        seg_path="",
        voxel_size_nm=voxel,
        grid_spacing_unit=unit,
        _source="incomplete",
        **_catalog_fields(dataset_row),
    )


def materialize_resolved_paths(db_path: Path) -> dict[str, Any]:
    """
    Fill ``dataset_resolved`` with one row per dataset using :func:`resolve_paths`.

    This is the **single downstream-friendly view** of “what to download”: it merges
    S3-probed columns, scraped PostgREST URLs, layer tokens, and template fallbacks
    in one place, with boolean readiness flags.

    Call after ``write_probe_results`` so ``s3_probe_*`` columns are visible.
    """
    import json

    conn = connect(db_path)
    migrate(conn)
    cur = conn.cursor()
    cur.execute("DELETE FROM dataset_resolved")
    rows = cur.execute("SELECT * FROM datasets ORDER BY dataset_name").fetchall()
    ready_labeled = 0
    ready_em = 0
    incomplete = 0
    by_source: dict[str, int] = {}

    for ds in rows:
        layers = conn.execute(
            "SELECT * FROM dataset_layers WHERE dataset_id = ? ORDER BY layer_role, layer_name",
            (ds["id"],),
        ).fetchall()
        entry = resolve_paths(ds, layers)
        em = (entry.get("img_path") or "").strip()
        seg = (entry.get("seg_path") or "").strip()
        src = entry.get("_source") or "incomplete"
        voxel = entry.get("voxel_size_nm")
        # ready_labeled=1 only for genuinely non-prediction mito masks.
        # download_mito_mask_quality='good' is the authoritative signal set by the
        # scraper; without it the seg path may be a prediction/inference layer.
        quality = (ds["download_mito_mask_quality"] or "").strip().lower() if "download_mito_mask_quality" in ds.keys() else ""
        rl = 1 if (em and seg and quality == "good") else 0
        re_only = 1 if em else 0
        if rl:
            ready_labeled += 1
        if re_only:
            ready_em += 1
        if not em or not seg:
            incomplete += 1
        by_source[src] = by_source.get(src, 0) + 1

        cur.execute(
            """
            INSERT INTO dataset_resolved(
              dataset_id, dataset_name, resolved_em_path, resolved_mito_seg_path,
              resolved_voxel_nm, path_source, ready_labeled, ready_em_only
            ) VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                ds["id"],
                ds["dataset_name"],
                em or None,
                seg or None,
                json.dumps(voxel) if voxel else None,
                src,
                rl,
                re_only,
            ),
        )

    # Rarely, another short-lived connection can still hold a write lock;
    # retry briefly instead of failing Stage-2 end-to-end.
    last_exc: Exception | None = None
    for _ in range(3):
        try:
            conn.commit()
            last_exc = None
            break
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower():
                raise
            last_exc = exc
            time.sleep(0.5)
    if last_exc is not None:
        raise last_exc
    conn.close()
    return {
        "resolved_rows": len(rows),
        "ready_labeled": ready_labeled,
        "ready_em": ready_em,
        "resolved_incomplete": incomplete,
        "by_source": by_source,
    }


# ── public API ─────────────────────────────────────────────────────────────────

def inventory_from_db(
    db_path: Path,
    *,
    mito_only: bool = True,
    require_paths: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Build a {dataset_name: entry} dict from a database_builder SQLite DB.
    S3-probed paths (from the s3_probe columns) are preferred over scraped hints.
    """
    conn = connect(db_path)
    migrate(conn)
    # Filter to datasets with an explicitly good (non-prediction) mito mask.
    # "mitochondria_in_layer_names=1" captures all 17 mito-related datasets including
    # the 5 that only have prediction/inference layers; "good" quality restricts to the
    # 12 with curated GT mito segmentation labels suitable for supervised training.
    where = "WHERE download_mito_mask_quality = 'good'" if mito_only else ""
    rows = conn.execute(
        f"SELECT * FROM datasets {where} ORDER BY dataset_name"
    ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for ds in rows:
        layers = conn.execute(
            "SELECT * FROM dataset_layers WHERE dataset_id = ? ORDER BY layer_role, layer_name",
            (ds["id"],),
        ).fetchall()
        entry = resolve_paths(ds, layers)
        if require_paths and (not entry["img_path"] or not entry["seg_path"]):
            continue
        out[ds["dataset_name"]] = entry
    conn.close()
    return out


def emit_python_module(data: dict[str, dict[str, Any]], var_name: str = "DATASETS") -> str:
    """Return a Python DATASETS dict string (compatible with download_cellmap.py)."""
    lines = [
        '"""Auto-generated DATASETS dict from database_builder DB. Do not edit manually."""',
        "from __future__ import annotations",
        "",
        f"{var_name} = {{",
    ]
    for key in sorted(data):
        e = data[key]
        py_key = key.replace("-", "_")
        vs = e.get("voxel_size_nm")
        vs_repr = repr(vs) if vs else "[16.0, 16.0, 16.0]  # FIXME: voxel size unknown"
        gd = e.get("grid_dimensions")
        ga = e.get("grid_axes")
        extra = ""
        if gd is not None:
            extra += f'\n        "grid_dimensions": {gd!r},'
        if ga is not None:
            extra += f'\n        "grid_axes": {ga!r},'
        lines += [
            f'    "{py_key}": {{',
            f'        "img_path": {e["img_path"]!r},',
            f'        "seg_path": {e["seg_path"]!r},',
            f'        "voxel_size": {vs_repr},{extra}',
            f'        # source: {e.get("_source", "?")}',
            "    },",
        ]
    lines.append("}")
    return "\n".join(lines)
