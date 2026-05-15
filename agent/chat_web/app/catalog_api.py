"""Database Catalog Viewer API — read-only endpoints for browsing stage-2 catalog DBs.

All routes are prefixed /api/studio/catalog.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/studio/catalog", tags=["catalog"])

_get_project_root: Any = None


def configure_catalog(*, get_project_root: Any) -> None:
    global _get_project_root
    _get_project_root = get_project_root


def _root() -> Path:
    if _get_project_root is None:
        raise RuntimeError("catalog_api not configured")
    return _get_project_root()


def _db_dir(root: Path) -> Path:
    return root / "2database_builder" / "outputs" / "databases"


def _db_path(root: Path, stem: str) -> Path:
    safe = "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in stem.strip())
    return _db_dir(root) / f"{safe}.db"


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


# Organelle patterns used for dynamic filter generation.
# Each (key, regex) pair — regex is tested against the full set of layer names found in the DB.
_ORGANELLE_PATTERNS: list[tuple[str, str]] = [
    ("mito",       r"mito"),
    ("nucleus",    r"nucle"),
    ("er",         r"\ber[\-_]|endoplasmic"),
    ("vesicle",    r"vesicle"),
    ("golgi",      r"golgi"),
    ("centrosome", r"cent[\-_]?seg|centrosome"),
    ("chromatin",  r"chrom"),
    ("ribosome",   r"ribosome"),
    ("lysosome",   r"lysosome"),
    ("pm",         r"\bpm[\-_]seg|\bpm_"),
    ("er_mito_contact", r"er.mito.contact"),
    ("lm",         r"\blm[\-_]\d"),
]

_ORGANELLE_LABELS: dict[str, str] = {
    "mito":            "Mitochondria",
    "nucleus":         "Nucleus",
    "er":              "Endoplasmic reticulum",
    "vesicle":         "Vesicle",
    "golgi":           "Golgi",
    "centrosome":      "Centrosome",
    "chromatin":       "Chromatin",
    "ribosome":        "Ribosome",
    "lysosome":        "Lysosome",
    "pm":              "Plasma membrane",
    "er_mito_contact": "ER–mito contacts",
    "lm":              "Light microscopy",
}


# SQL expression that normalizes layer_role for backward-compat with DB rows written
# before path-based classification was enforced.  Layers whose upstream role is
# "ground_truths" but whose URL/name/content signals indicate inference or prediction
# are reclassified to "predictions" at query time so the UI never shows them under
# "Ground truths".
_EFF_ROLE_EXPR = """CASE
    WHEN layer_role = 'ground_truths' AND (
        INSTR(LOWER(COALESCE(layer_url,'')), 'inference') > 0
        OR INSTR(COALESCE(layer_url,''), '_pred') > 0
        OR INSTR(COALESCE(layer_url,''), '_prediction') > 0
        OR INSTR(COALESCE(layer_name,''), '_pred') > 0
        OR INSTR(COALESCE(layer_name,''), '_prediction') > 0
        OR INSTR(LOWER(COALESCE(layer_content_type,'')), 'prediction') > 0
        OR INSTR(LOWER(COALESCE(layer_stage,'')), 'prediction') > 0
    ) THEN 'predictions'
    ELSE layer_role
END"""


def _normalize_layer_dict(layer: dict) -> dict:
    """Apply _EFF_ROLE_EXPR logic in Python for detail-endpoint rows."""
    role = layer.get("layer_role") or ""
    if role != "ground_truths":
        return layer
    url = (layer.get("layer_url") or "").lower()
    name = layer.get("layer_name") or ""
    ct = (layer.get("layer_content_type") or "").lower()
    st = (layer.get("layer_stage") or "").lower()
    if (
        "inference" in url
        or "_pred" in url
        or "_prediction" in url
        or "_pred" in name
        or "_prediction" in name
        or "prediction" in ct
        or "prediction" in st
    ):
        layer = dict(layer)
        layer["layer_role"] = "predictions"
    return layer


def _tables_exist(conn: sqlite3.Connection, *names: str) -> bool:
    existing = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    return all(n in existing for n in names)


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except Exception:
        return set()
    return {str(r[1]) for r in rows}


# ── GET /databases ─────────────────────────────────────────────────────────────

@router.get("/databases")
def catalog_list_databases() -> dict[str, Any]:
    root = _root()
    db_dir = _db_dir(root)
    out: list[dict[str, Any]] = []
    if db_dir.is_dir():
        for p in sorted(db_dir.glob("*.db"), key=lambda x: x.name.lower()):
            out.append({
                "stem": p.stem,
                "path": str(p.relative_to(root)),
                "exists": p.is_file(),
                "size_bytes": p.stat().st_size if p.is_file() else None,
            })
    return {"databases": out}


# ── GET /{stem}/filters ────────────────────────────────────────────────────────

@router.get("/{stem}/filters")
def catalog_filters(stem: str) -> dict[str, Any]:
    root = _root()
    p = _db_path(root, stem)
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"Database '{stem}' not found.")
    try:
        conn = _connect(p)
    except sqlite3.OperationalError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not open catalog database '{p}': {exc}",
        ) from exc
    try:
        try:
            return _build_filter_spec(conn)
        except sqlite3.OperationalError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Catalog database is temporarily unavailable: {exc}",
            ) from exc
    finally:
        conn.close()


def _parse_json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        v = json.loads(raw)
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
    except Exception:
        pass
    return [raw.strip()] if raw.strip() else []


def _build_filter_spec(conn: sqlite3.Connection) -> dict[str, Any]:
    if not _tables_exist(conn, "datasets"):
        return {
            "organisms": [], "sample_types": [], "organelles_present": [],
            "organelle_labels": {}, "content_types": [], "mito_mask_qualities": [],
            "stages": [], "sample_subtypes": [], "layer_roles": [],
            "bio_targets": [], "bio_target_types": [],
            "boolean_facets": {},
        }

    # Distinct organisms (stored as JSON arrays or plain strings)
    organism_set: set[str] = set()
    for row in conn.execute(
        "SELECT DISTINCT sample_organism FROM datasets WHERE sample_organism IS NOT NULL"
    ).fetchall():
        for v in _parse_json_list(row[0]):
            organism_set.add(v)
    organisms = sorted(organism_set, key=str.lower)

    # Distinct sample types
    sample_types = [
        r[0] for r in conn.execute(
            "SELECT DISTINCT sample_type FROM datasets WHERE sample_type IS NOT NULL ORDER BY sample_type"
        ).fetchall()
    ]
    # Stage-2 catalog UX policy: do not expose a separate subtype facet because
    # ``sample_type`` is normalized to subtype when available.
    sample_subtypes: list[str] = []
    stages = [
        r[0] for r in conn.execute(
            "SELECT DISTINCT stage FROM datasets WHERE stage IS NOT NULL ORDER BY stage"
        ).fetchall()
    ]
    dcols = _table_columns(conn, "datasets")
    # Stage-2 catalog UX policy: do not expose biological target facet; sample type is primary.
    bio_targets: list[str] = []
    # Stage-2 catalog UX policy: do not expose biological target *type* facet; it is redundant.
    bio_target_types: list[str] = []

    # Which organelles appear in layer names (regex scan)
    all_layer_names_blob = " ".join(
        r[0] or "" for r in conn.execute("SELECT layer_name FROM dataset_layers").fetchall()
        if _tables_exist(conn, "dataset_layers")
    ) if _tables_exist(conn, "dataset_layers") else ""

    organelles_present: list[str] = []
    for key, pattern in _ORGANELLE_PATTERNS:
        if re.search(pattern, all_layer_names_blob, re.IGNORECASE):
            organelles_present.append(key)

    # Distinct layer content types
    content_types: list[str] = []
    if _tables_exist(conn, "dataset_layers"):
        content_types = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT layer_content_type FROM dataset_layers "
                "WHERE layer_content_type IS NOT NULL ORDER BY layer_content_type"
            ).fetchall()
        ]
        # Stage-2 catalog UX policy: do not expose layer-role facet; it is redundant.
        layer_roles: list[str] = []
    else:
        layer_roles = []

    # Distinct mito mask quality values
    # Stage-2 catalog UX policy: hide raw quality facet (good/unset) to avoid
    # redundancy with mito boolean toggles.
    mito_mask_qualities: list[str | None] = []

    # Boolean facet counts
    boolean_facets: dict[str, dict[str, Any]] = {}
    n_s3 = conn.execute(
        "SELECT COUNT(*) FROM datasets WHERE s3_probe_img_path IS NOT NULL"
    ).fetchone()[0]
    boolean_facets["has_s3_probe"] = {"label": "Has S3 probe", "count": n_s3}

    n_mito_gt = conn.execute(
        "SELECT COUNT(*) FROM datasets WHERE mitochondria_in_layer_names = 1"
    ).fetchone()[0]
    boolean_facets["has_mito_gt"] = {"label": "Has mito label in layers", "count": n_mito_gt}

    n_good_mito = conn.execute(
        "SELECT COUNT(*) FROM datasets WHERE download_mito_mask_quality = 'good'"
    ).fetchone()[0]
    boolean_facets["has_good_mito_mask"] = {"label": "Good mito mask (non-prediction)", "count": n_good_mito}

    if _tables_exist(conn, "dataset_resolved"):
        n_labeled = conn.execute(
            "SELECT COUNT(*) FROM dataset_resolved WHERE ready_labeled = 1"
        ).fetchone()[0]
        boolean_facets["ready_labeled"] = {"label": "Download-ready (labeled)", "count": n_labeled}
        n_em = conn.execute(
            "SELECT COUNT(*) FROM dataset_resolved WHERE ready_em_only = 1"
        ).fetchone()[0]
        boolean_facets["ready_em"] = {"label": "Download-ready (EM)", "count": n_em}

    if _tables_exist(conn, "dataset_layers"):
        n_em_layers = conn.execute(
            "SELECT COUNT(DISTINCT dataset_id) FROM dataset_layers WHERE layer_content_type = 'em'"
        ).fetchone()[0]
        boolean_facets["has_em_layers"] = {"label": "Has EM image layers", "count": n_em_layers}
        # Use effective (path-normalized) role for accurate prediction count.
        n_pred = conn.execute(
            f"SELECT COUNT(DISTINCT dataset_id) FROM dataset_layers WHERE ({_EFF_ROLE_EXPR}) = 'predictions'"
        ).fetchone()[0]
        boolean_facets["has_predictions"] = {"label": "Has prediction layers", "count": n_pred}

    return {
        "organisms": organisms,
        "sample_types": sample_types,
        "organelles_present": organelles_present,
        "organelle_labels": _ORGANELLE_LABELS,
        "content_types": content_types,
        "mito_mask_qualities": mito_mask_qualities,
        "stages": stages,
        "sample_subtypes": sample_subtypes,
        "layer_roles": layer_roles,
        "bio_targets": bio_targets,
        "bio_target_types": bio_target_types,
        "boolean_facets": boolean_facets,
    }


# ── GET /{stem}/datasets ───────────────────────────────────────────────────────

@router.get("/{stem}/datasets")
def catalog_datasets(
    stem: str,
    stage: list[str] = Query(default=[]),
    organism: list[str] = Query(default=[]),
    sample_type: list[str] = Query(default=[]),
    sample_subtype: list[str] = Query(default=[]),
    bio_target: list[str] = Query(default=[]),
    bio_target_type: list[str] = Query(default=[]),
    mito_mask_quality: list[str] = Query(default=[]),
    content_type: list[str] = Query(default=[]),
    layer_role: list[str] = Query(default=[]),
    organelle: list[str] = Query(default=[]),
    has_s3_probe: bool | None = Query(default=None),
    ready_labeled: bool | None = Query(default=None),
    ready_em: bool | None = Query(default=None),
    has_mito_gt: bool | None = Query(default=None),
    has_good_mito_mask: bool | None = Query(default=None),
    has_em_layers: bool | None = Query(default=None),
    has_predictions: bool | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    root = _root()
    p = _db_path(root, stem)
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"Database '{stem}' not found.")
    conn = _connect(p)
    try:
        has_layers = _tables_exist(conn, "dataset_layers")
        has_resolved = _tables_exist(conn, "dataset_resolved")
        return _query_datasets(
            conn,
            has_layers=has_layers,
            has_resolved=has_resolved,
            stage=stage,
            organism=organism,
            sample_type=sample_type,
            sample_subtype=sample_subtype,
            bio_target=bio_target,
            bio_target_type=bio_target_type,
            mito_mask_quality=mito_mask_quality,
            content_type=content_type,
            layer_role=layer_role,
            organelle=organelle,
            has_s3_probe=has_s3_probe,
            ready_labeled=ready_labeled,
            ready_em=ready_em,
            has_mito_gt=has_mito_gt,
            has_good_mito_mask=has_good_mito_mask,
            has_em_layers=has_em_layers,
            has_predictions=has_predictions,
            limit=limit,
            offset=offset,
        )
    finally:
        conn.close()


def _query_datasets(
    conn: sqlite3.Connection,
    *,
    has_layers: bool,
    has_resolved: bool,
    stage: list[str],
    organism: list[str],
    sample_type: list[str],
    sample_subtype: list[str],
    bio_target: list[str],
    bio_target_type: list[str],
    mito_mask_quality: list[str],
    content_type: list[str],
    layer_role: list[str],
    organelle: list[str],
    has_s3_probe: bool | None,
    ready_labeled: bool | None,
    ready_em: bool | None,
    has_mito_gt: bool | None,
    has_good_mito_mask: bool | None,
    has_em_layers: bool | None,
    has_predictions: bool | None,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    dcols = _table_columns(conn, "datasets")
    has_bio_target = "bio_target" in dcols
    has_bio_target_type = "bio_target_type" in dcols
    has_bio_target_source = "bio_target_source" in dcols
    has_bio_target_confidence = "bio_target_confidence" in dcols
    # Build CTE for per-dataset layer aggregation.
    # Use _EFF_ROLE_EXPR (path-normalized role) so that inference/prediction layers
    # stored under 'ground_truths' in old DB rows are counted correctly.
    if has_layers:
        layer_cte = f"""
layer_agg AS (
    SELECT
        dataset_id,
        SUM(CASE WHEN ({_EFF_ROLE_EXPR}) = 'ground_truths' THEN 1 ELSE 0 END) AS n_gt,
        SUM(CASE WHEN ({_EFF_ROLE_EXPR}) = 'predictions' THEN 1 ELSE 0 END) AS n_pred,
        SUM(CASE WHEN ({_EFF_ROLE_EXPR}) = 'raw_or_other_layers' THEN 1 ELSE 0 END) AS n_raw,
        SUM(CASE WHEN layer_content_type = 'em' THEN 1 ELSE 0 END) AS n_em,
        GROUP_CONCAT(layer_name, '|') AS all_layer_names
    FROM dataset_layers
    GROUP BY dataset_id
)"""
        layer_join = "LEFT JOIN layer_agg la ON la.dataset_id = d.id"
        layer_cols = (
            "COALESCE(la.n_gt,0) AS n_gt, COALESCE(la.n_pred,0) AS n_pred, "
            "COALESCE(la.n_raw,0) AS n_raw, COALESCE(la.n_em,0) AS n_em, "
            "la.all_layer_names"
        )
        cte_clause = f"WITH {layer_cte}"
    else:
        layer_join = ""
        layer_cols = "0 AS n_gt, 0 AS n_pred, 0 AS n_raw, 0 AS n_em, NULL AS all_layer_names"
        cte_clause = ""

    if has_resolved:
        resolved_join = "LEFT JOIN dataset_resolved r ON r.dataset_id = d.id"
        resolved_cols = "r.ready_labeled, r.ready_em_only, r.path_source, r.resolved_voxel_nm"
    else:
        resolved_join = ""
        resolved_cols = "NULL AS ready_labeled, NULL AS ready_em_only, NULL AS path_source, NULL AS resolved_voxel_nm"

    conditions: list[str] = []
    params: list[Any] = []

    # Organism filter (LIKE on JSON array or plain string)
    if stage:
        phs = ",".join("?" for _ in stage)
        conditions.append(f"d.stage IN ({phs})")
        params.extend(stage)

    # Organism filter (LIKE on JSON array or plain string)
    if organism:
        org_conds = " OR ".join("d.sample_organism LIKE ?" for _ in organism)
        conditions.append(f"({org_conds})")
        params.extend(f"%{v}%" for v in organism)

    # Sample type
    if sample_type:
        phs = ",".join("?" for _ in sample_type)
        conditions.append(f"d.sample_type IN ({phs})")
        params.extend(sample_type)
    if sample_subtype:
        phs = ",".join("?" for _ in sample_subtype)
        conditions.append(f"d.sample_subtype IN ({phs})")
        params.extend(sample_subtype)
    if bio_target and has_bio_target:
        phs = ",".join("?" for _ in bio_target)
        conditions.append(f"d.bio_target IN ({phs})")
        params.extend(bio_target)
    if bio_target_type and has_bio_target_type:
        phs = ",".join("?" for _ in bio_target_type)
        conditions.append(f"d.bio_target_type IN ({phs})")
        params.extend(bio_target_type)

    # Mito mask quality (handle "null" string → IS NULL)
    if mito_mask_quality:
        non_null = [v for v in mito_mask_quality if v != "null"]
        has_null_q = "null" in mito_mask_quality
        sub: list[str] = []
        if non_null:
            phs = ",".join("?" for _ in non_null)
            sub.append(f"d.download_mito_mask_quality IN ({phs})")
            params.extend(non_null)
        if has_null_q:
            sub.append("d.download_mito_mask_quality IS NULL")
        conditions.append(f"({' OR '.join(sub)})")

    # Content type (via subquery into dataset_layers)
    if content_type and has_layers:
        phs = ",".join("?" for _ in content_type)
        conditions.append(
            f"d.id IN (SELECT dataset_id FROM dataset_layers WHERE layer_content_type IN ({phs}))"
        )
        params.extend(content_type)
    if layer_role and has_layers:
        phs = ",".join("?" for _ in layer_role)
        conditions.append(
            f"d.id IN (SELECT dataset_id FROM dataset_layers WHERE ({_EFF_ROLE_EXPR}) IN ({phs}))"
        )
        params.extend(layer_role)

    # Organelle presence (LIKE on aggregated layer names)
    if organelle and has_layers:
        org_conds = " AND ".join("la.all_layer_names LIKE ?" for _ in organelle)
        conditions.append(f"({org_conds})")
        params.extend(f"%{v}%" for v in organelle)

    # Boolean facets
    if has_s3_probe is True:
        conditions.append("d.s3_probe_img_path IS NOT NULL")
    elif has_s3_probe is False:
        conditions.append("d.s3_probe_img_path IS NULL")

    if has_mito_gt is True:
        conditions.append("d.mitochondria_in_layer_names = 1")
    elif has_mito_gt is False:
        conditions.append("d.mitochondria_in_layer_names = 0")

    if has_good_mito_mask is True:
        conditions.append("d.download_mito_mask_quality = 'good'")
    elif has_good_mito_mask is False:
        conditions.append("(d.download_mito_mask_quality IS NULL OR d.download_mito_mask_quality != 'good')")

    if has_resolved:
        if ready_labeled is True:
            conditions.append("r.ready_labeled = 1")
        elif ready_labeled is False:
            conditions.append("(r.ready_labeled IS NULL OR r.ready_labeled = 0)")
        if ready_em is True:
            conditions.append("r.ready_em_only = 1")
        elif ready_em is False:
            conditions.append("(r.ready_em_only IS NULL OR r.ready_em_only = 0)")

    if has_layers:
        if has_em_layers is True:
            conditions.append("COALESCE(la.n_em,0) > 0")
        elif has_em_layers is False:
            conditions.append("COALESCE(la.n_em,0) = 0")
        if has_predictions is True:
            conditions.append("COALESCE(la.n_pred,0) > 0")
        elif has_predictions is False:
            conditions.append("COALESCE(la.n_pred,0) = 0")

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    base_sql = f"""
{cte_clause}
SELECT
    d.id, d.dataset_name, d.sample_organism, d.sample_type, d.sample_subtype,
    {("d.bio_target" if has_bio_target else "NULL")} AS bio_target,
    {("d.bio_target_type" if has_bio_target_type else "NULL")} AS bio_target_type,
    {("d.bio_target_source" if has_bio_target_source else "NULL")} AS bio_target_source,
    {("d.bio_target_confidence" if has_bio_target_confidence else "NULL")} AS bio_target_confidence,
    d.mitochondria_in_layer_names, d.download_mito_mask_quality,
    d.s3_probe_img_path, d.s3_probe_error, d.stage, d.description,
    {resolved_cols},
    {layer_cols}
FROM datasets d
{resolved_join}
{layer_join}
{where_clause}
""".strip()

    # Total count
    count_sql = f"""
{cte_clause}
SELECT COUNT(*) FROM datasets d
{resolved_join}
{layer_join}
{where_clause}
""".strip()

    total = conn.execute(count_sql, params).fetchone()[0]

    rows = conn.execute(
        f"{base_sql} ORDER BY d.dataset_name LIMIT ? OFFSET ?",
        params + [limit, offset],
    ).fetchall()

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "datasets": [dict(r) for r in rows],
    }


# ── GET /{stem}/dataset/{name} ─────────────────────────────────────────────────

@router.get("/{stem}/dataset/{name:path}")
def catalog_dataset_detail(stem: str, name: str) -> dict[str, Any]:
    root = _root()
    p = _db_path(root, stem)
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"Database '{stem}' not found.")
    conn = _connect(p)
    try:
        row = conn.execute(
            "SELECT * FROM datasets WHERE dataset_name = ?", (name,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found.")
        dataset = dict(row)

        layers: list[dict] = []
        if _tables_exist(conn, "dataset_layers"):
            layers = [
                _normalize_layer_dict(dict(r)) for r in conn.execute(
                    "SELECT * FROM dataset_layers WHERE dataset_id = ? "
                    "ORDER BY layer_role, layer_kind, layer_name",
                    (dataset["id"],),
                ).fetchall()
            ]

        resolved = None
        if _tables_exist(conn, "dataset_resolved"):
            r = conn.execute(
                "SELECT * FROM dataset_resolved WHERE dataset_id = ?",
                (dataset["id"],),
            ).fetchone()
            resolved = dict(r) if r else None

        sources: list[str] = []
        if _tables_exist(conn, "dataset_sources"):
            sources = [
                r["source_url"] for r in conn.execute(
                    "SELECT source_url FROM dataset_sources WHERE dataset_id = ?",
                    (dataset["id"],),
                ).fetchall()
            ]

        return {
            "dataset": dataset,
            "layers": layers,
            "resolved": resolved,
            "sources": sources,
        }
    finally:
        conn.close()
