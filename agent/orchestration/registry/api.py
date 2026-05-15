"""CRUD helpers for the central registry.

All functions accept an open sqlite3.Connection (with row_factory=sqlite3.Row)
and commit within the same transaction as the caller. Call conn.commit() after
a batch of writes, or wrap in a context manager.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any
from agent.orchestration.time_utils import now_us_eastern_iso


def _now() -> str:
    return now_us_eastern_iso(timespec="seconds")


# ── providers ─────────────────────────────────────────────────────────────────

def upsert_provider(conn: sqlite3.Connection, *, name: str, base_url: str = "") -> int:
    now = _now()
    conn.execute(
        """
        INSERT INTO providers (name, base_url, created_at)
        VALUES (?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET base_url = excluded.base_url
        """,
        (name, base_url, now),
    )
    row = conn.execute("SELECT id FROM providers WHERE name = ?", (name,)).fetchone()
    return int(row["id"])


def get_provider_id(conn: sqlite3.Connection, name: str) -> int | None:
    row = conn.execute("SELECT id FROM providers WHERE name = ?", (name,)).fetchone()
    return int(row["id"]) if row else None


# ── datasets ──────────────────────────────────────────────────────────────────

def upsert_dataset(
    conn: sqlite3.Connection,
    *,
    provider_id: int,
    stable_id: str,
    display_name: str = "",
    metadata: dict[str, Any] | None = None,
    changed: bool = False,
) -> int:
    now = _now()
    meta_json = json.dumps(metadata or {}, ensure_ascii=False)
    existing = conn.execute(
        "SELECT id, last_changed_at FROM datasets WHERE provider_id = ? AND stable_id = ?",
        (provider_id, stable_id),
    ).fetchone()
    if existing is None:
        conn.execute(
            """
            INSERT INTO datasets
              (provider_id, stable_id, display_name, first_seen_at, last_seen_at,
               last_changed_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (provider_id, stable_id, display_name or stable_id, now, now,
             now if changed else None, meta_json),
        )
    else:
        conn.execute(
            """
            UPDATE datasets
            SET display_name = ?, last_seen_at = ?,
                last_changed_at = CASE WHEN ? THEN ? ELSE last_changed_at END,
                metadata_json = ?
            WHERE provider_id = ? AND stable_id = ?
            """,
            (display_name or stable_id, now, 1 if changed else 0, now,
             meta_json, provider_id, stable_id),
        )
    row = conn.execute(
        "SELECT id FROM datasets WHERE provider_id = ? AND stable_id = ?",
        (provider_id, stable_id),
    ).fetchone()
    return int(row["id"])


def get_dataset_id(
    conn: sqlite3.Connection, provider_id: int, stable_id: str
) -> int | None:
    row = conn.execute(
        "SELECT id FROM datasets WHERE provider_id = ? AND stable_id = ?",
        (provider_id, stable_id),
    ).fetchone()
    return int(row["id"]) if row else None


def list_datasets(conn: sqlite3.Connection, provider_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM datasets WHERE provider_id = ? ORDER BY stable_id",
        (provider_id,),
    ).fetchall()


def get_dataset_spacing_nm(
    conn: sqlite3.Connection, provider_id: int, stable_id: str
) -> list[float] | None:
    """Return ZYX voxel spacing from registry metadata, or None."""
    row = conn.execute(
        "SELECT metadata_json FROM datasets WHERE provider_id = ? AND stable_id = ?",
        (provider_id, stable_id),
    ).fetchone()
    if not row:
        return None
    try:
        meta = json.loads(row["metadata_json"] or "{}")
        voxel = meta.get("voxel_size_nm")
        if isinstance(voxel, list) and len(voxel) >= 3:
            return [float(v) for v in voxel[:3]]
    except Exception:
        pass
    return None


# ── assets ────────────────────────────────────────────────────────────────────

def upsert_asset(
    conn: sqlite3.Connection,
    *,
    dataset_id: int,
    asset_type: str,
    remote_url: str,
    content_fingerprint: str | None = None,
    byte_size: int | None = None,
) -> int:
    now = _now()
    conn.execute(
        """
        INSERT INTO assets (dataset_id, asset_type, remote_url, content_fingerprint,
                            byte_size, last_checked_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(dataset_id, asset_type, remote_url) DO UPDATE SET
          content_fingerprint = COALESCE(excluded.content_fingerprint, content_fingerprint),
          byte_size           = COALESCE(excluded.byte_size, byte_size),
          last_checked_at     = excluded.last_checked_at
        """,
        (dataset_id, asset_type, remote_url, content_fingerprint, byte_size, now),
    )
    row = conn.execute(
        "SELECT id FROM assets WHERE dataset_id = ? AND asset_type = ? AND remote_url = ?",
        (dataset_id, asset_type, remote_url),
    ).fetchone()
    return int(row["id"])


def get_asset(
    conn: sqlite3.Connection, dataset_id: int, asset_type: str
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT a.* FROM assets a
        WHERE a.dataset_id = ? AND a.asset_type = ?
        ORDER BY a.last_checked_at DESC LIMIT 1
        """,
        (dataset_id, asset_type),
    ).fetchone()


# ── download profile hash ─────────────────────────────────────────────────────

def make_download_profile_hash(
    *,
    n_crops: int,
    chunk_zyx: tuple[int, int, int] | list[int],
    voxel_nm_zyx: tuple[float, float, float] | list[float],
    mode: str = "labeled",
    foundation: bool = True,
) -> str:
    """Return a 16-hex SHA-256 fingerprint of the given download parameters.

    Two calls with identical parameters always return the same hash; any change
    (e.g. n_crops 1 → 4) produces a different hash so the registry treats them
    as separate download events.
    """
    canonical = json.dumps(
        {
            "n_crops": int(n_crops),
            "chunk_zyx": [int(v) for v in chunk_zyx],
            "voxel_nm_zyx": [float(v) for v in voxel_nm_zyx],
            "mode": str(mode),
            "foundation": bool(foundation),
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# Default profile: 1 crop, 128³ voxels, 16 nm isotropic, labeled, foundation on.
# NOTE: legacy download rows written before profile tracking (or with the old 512³
# default) will not match this hash and will show as needing re-download for the
# new profile — this is intentional since the crop size changed.
DEFAULT_DOWNLOAD_PROFILE_HASH: str = make_download_profile_hash(
    n_crops=1,
    chunk_zyx=(128, 128, 128),
    voxel_nm_zyx=(16.0, 16.0, 16.0),
    mode="labeled",
    foundation=True,
)

# Legacy 512³ profile hash — used to recognise old download records so that
# existing databases are not silently corrupted by the dimension change.
LEGACY_512_DOWNLOAD_PROFILE_HASH: str = make_download_profile_hash(
    n_crops=1,
    chunk_zyx=(512, 512, 512),
    voxel_nm_zyx=(16.0, 16.0, 16.0),
    mode="labeled",
    foundation=True,
)


# ── downloads ─────────────────────────────────────────────────────────────────

def record_download_start(
    conn: sqlite3.Connection,
    asset_id: int,
    local_path: str,
    *,
    download_profile_hash: str | None = None,
) -> int:
    now = _now()
    conn.execute(
        """
        INSERT INTO downloads (asset_id, local_path, status, started_at, download_profile_hash)
        VALUES (?, ?, 'downloading', ?, ?)
        """,
        (asset_id, local_path, now, download_profile_hash),
    )
    row = conn.execute(
        "SELECT id FROM downloads WHERE asset_id = ? AND local_path = ? ORDER BY id DESC LIMIT 1",
        (asset_id, local_path),
    ).fetchone()
    return int(row["id"])


def record_download_complete(
    conn: sqlite3.Connection, download_id: int, *, verified_fingerprint: str | None = None
) -> None:
    conn.execute(
        """
        UPDATE downloads
        SET status = 'complete', finished_at = ?, verified_fingerprint = ?
        WHERE id = ?
        """,
        (_now(), verified_fingerprint, download_id),
    )


def record_download_failed(
    conn: sqlite3.Connection, download_id: int, error: str
) -> None:
    conn.execute(
        """
        UPDATE downloads SET status = 'failed', finished_at = ?, error = ?
        WHERE id = ?
        """,
        (_now(), error[:2000], download_id),
    )


def find_complete_download(
    conn: sqlite3.Connection, asset_id: int
) -> sqlite3.Row | None:
    """Return the most recent complete download for this asset, if any."""
    return conn.execute(
        """
        SELECT * FROM downloads
        WHERE asset_id = ? AND status = 'complete'
        ORDER BY finished_at DESC LIMIT 1
        """,
        (asset_id,),
    ).fetchone()


def is_asset_downloaded(conn: sqlite3.Connection, asset_id: int) -> bool:
    row = find_complete_download(conn, asset_id)
    if row is None:
        return False
    lp = Path(row["local_path"])
    return lp.is_file() or lp.is_dir()


def find_complete_download_for_profile(
    conn: sqlite3.Connection,
    asset_id: int,
    profile_hash: str,
) -> sqlite3.Row | None:
    """Return the most recent complete download matching ``profile_hash``.

    Legacy rows (``download_profile_hash IS NULL``) were written with the old
    512³ default and are treated as matching ``LEGACY_512_DOWNLOAD_PROFILE_HASH``
    so those downloads are not invalidated when checking the 512³ profile.
    The new 128³ default only matches rows with the explicit 128³ hash.
    """
    if profile_hash == LEGACY_512_DOWNLOAD_PROFILE_HASH:
        # Match explicit legacy hash OR NULL rows (pre-profile-tracking completes).
        return conn.execute(
            """
            SELECT * FROM downloads
            WHERE asset_id = ? AND status = 'complete'
              AND (download_profile_hash = ? OR download_profile_hash IS NULL)
            ORDER BY finished_at DESC LIMIT 1
            """,
            (asset_id, profile_hash),
        ).fetchone()
    else:
        # All other profiles (including the new 128³ default): exact match only.
        return conn.execute(
            """
            SELECT * FROM downloads
            WHERE asset_id = ? AND status = 'complete'
              AND download_profile_hash = ?
            ORDER BY finished_at DESC LIMIT 1
            """,
            (asset_id, profile_hash),
        ).fetchone()


def is_asset_downloaded_for_profile(
    conn: sqlite3.Connection,
    asset_id: int,
    profile_hash: str,
) -> bool:
    """True if a complete, locally-present download exists for the given profile.

    Use this instead of :func:`is_asset_downloaded` when n_crops or other
    download parameters are part of the skip decision.
    """
    row = find_complete_download_for_profile(conn, asset_id, profile_hash)
    if row is None:
        return False
    lp = Path(row["local_path"])
    return lp.is_file() or lp.is_dir()


def find_complete_download_for_dataset_type_profile(
    conn: sqlite3.Connection,
    dataset_id: int,
    asset_type: str,
    profile_hash: str,
) -> sqlite3.Row | None:
    """Return the most recent complete download for any asset row matching
    ``(dataset_id, asset_type)`` and ``profile_hash``.

    Unlike :func:`find_complete_download_for_profile`, this queries across
    **all** asset rows for the dataset+type pair.  This is URL-churn-safe: if
    re-scraping inserts a new asset row with a slightly different ``remote_url``,
    a prior complete download recorded against the old row still satisfies the
    skip check.
    """
    if profile_hash == LEGACY_512_DOWNLOAD_PROFILE_HASH:
        # Legacy 512³ profile: also match NULL rows (pre-profile-tracking).
        return conn.execute(
            """
            SELECT d.* FROM downloads d
            JOIN assets a ON a.id = d.asset_id
            WHERE a.dataset_id = ? AND a.asset_type = ?
              AND d.status = 'complete'
              AND (d.download_profile_hash = ? OR d.download_profile_hash IS NULL)
            ORDER BY d.finished_at DESC LIMIT 1
            """,
            (dataset_id, asset_type, profile_hash),
        ).fetchone()
    else:
        return conn.execute(
            """
            SELECT d.* FROM downloads d
            JOIN assets a ON a.id = d.asset_id
            WHERE a.dataset_id = ? AND a.asset_type = ?
              AND d.status = 'complete'
              AND d.download_profile_hash = ?
            ORDER BY d.finished_at DESC LIMIT 1
            """,
            (dataset_id, asset_type, profile_hash),
        ).fetchone()


def _labeled_pairs_present_under_images_dir(
    conn: sqlite3.Connection,
    dataset_id: int,
    images_dir: Path,
) -> bool:
    """When ``local_path`` is an images directory, require paired files.

    Downloader rows store the **images directory** on ``em_volume`` completes. A bare
    ``Path.is_dir()`` stays true after users delete paired files,
    which incorrectly skipped re-downloads. We only tighten checks for the standard
    training layouts so other callers (tests, custom dirs) keep directory-exists
    semantics.
    """
    if not images_dir.is_dir():
        return True

    name_l = images_dir.name.lower()
    if name_l == "imagestr":
        labels_dir = images_dir.parent / "labelsTr"
        im_suffix = "_0000.nii.gz"
        seg_suffix = ".nii.gz"
        im_pattern = "{tag}*_0000.nii.gz"
    elif name_l == "imagests":
        labels_dir = images_dir.parent / "labelsTs"
        im_suffix = "_0000.nii.gz"
        seg_suffix = ".nii.gz"
        im_pattern = "{tag}*_0000.nii.gz"
    else:
        return True

    if not labels_dir.is_dir():
        return True

    row_ds = conn.execute(
        "SELECT stable_id FROM datasets WHERE id = ?",
        (dataset_id,),
    ).fetchone()
    if row_ds is None:
        return True
    sid = (row_ds["stable_id"] or "").strip()
    if not sid:
        return True
    tag = sid.replace("-", "_").replace("/", "_")
    im_files = sorted(images_dir.glob(im_pattern.format(tag=tag)))
    if not im_files:
        return False
    for im_p in im_files:
        if im_suffix and im_p.name.endswith(im_suffix):
            seg_name = im_p.name[: -len(im_suffix)] + seg_suffix
        else:
            continue
        seg_p = labels_dir / seg_name
        if not seg_p.is_file():
            return False
    return True


def is_dataset_type_downloaded_for_profile(
    conn: sqlite3.Connection,
    dataset_id: int,
    asset_type: str,
    profile_hash: str,
) -> bool:
    """True if **any** asset row for ``(dataset_id, asset_type)`` has a
    complete, locally-present download for ``profile_hash``.

    Use this in preference to :func:`is_asset_downloaded_for_profile` when
    URL churn (re-scrape changing ``remote_url`` slightly) must not force a
    re-download of data that was already fetched.
    """
    row = find_complete_download_for_dataset_type_profile(
        conn, dataset_id, asset_type, profile_hash
    )
    if row is None:
        return False

    # Robustness: if any tracked batch item for this dataset+asset_type is marked
    # missing/deleted locally, force re-download even if a historical complete
    # download row exists for the same profile.
    missing_row = conn.execute(
        """
        SELECT 1
        FROM batch_items bi
        WHERE bi.dataset_id = ?
          AND bi.asset_type = ?
          AND bi.status = 'missing_or_deleted_local'
        LIMIT 1
        """,
        (dataset_id, asset_type),
    ).fetchone()
    if missing_row is not None:
        return False

    lp = Path(row["local_path"])
    if lp.is_file():
        return True
    if not lp.is_dir():
        return False
    if asset_type == "em_volume" and not _labeled_pairs_present_under_images_dir(
        conn, dataset_id, lp
    ):
        return False
    return True


# ── preprocess configs ────────────────────────────────────────────────────────

def fingerprint_config(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def ensure_preprocess_config(conn: sqlite3.Connection, config: dict[str, Any]) -> str:
    fp = fingerprint_config(config)
    now = _now()
    conn.execute(
        """
        INSERT OR IGNORE INTO preprocess_configs (config_fingerprint, config_json, created_at)
        VALUES (?, ?, ?)
        """,
        (fp, json.dumps(config, ensure_ascii=False), now),
    )
    return fp


# ── preprocess runs ───────────────────────────────────────────────────────────

def fingerprint_file(path: Path) -> str:
    """SHA-256 of file contents (first 10 MB) as a 16-hex string."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
                if f.tell() >= 10 * 1024 * 1024:
                    h.update(b"\x00truncated")
                    break
    except OSError:
        return ""
    return h.hexdigest()[:16]


def find_complete_preprocess_run(
    conn: sqlite3.Connection,
    input_fingerprint: str,
    config_fingerprint: str,
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT * FROM preprocess_runs
        WHERE input_fingerprint = ? AND config_fingerprint = ? AND status = 'complete'
        ORDER BY finished_at DESC LIMIT 1
        """,
        (input_fingerprint, config_fingerprint),
    ).fetchone()


def record_preprocess_start(
    conn: sqlite3.Connection,
    *,
    dataset_id: int | None,
    input_fingerprint: str,
    config_fingerprint: str,
) -> int:
    now = _now()
    conn.execute(
        """
        INSERT INTO preprocess_runs
          (dataset_id, input_fingerprint, config_fingerprint, status, started_at)
        VALUES (?, ?, ?, 'running', ?)
        """,
        (dataset_id, input_fingerprint, config_fingerprint, now),
    )
    row = conn.execute(
        """
        SELECT id FROM preprocess_runs
        WHERE input_fingerprint = ? AND config_fingerprint = ?
        ORDER BY id DESC LIMIT 1
        """,
        (input_fingerprint, config_fingerprint),
    ).fetchone()
    return int(row["id"])


def record_preprocess_complete(
    conn: sqlite3.Connection, run_id: int, output_paths: list[str]
) -> None:
    conn.execute(
        """
        UPDATE preprocess_runs
        SET status = 'complete', finished_at = ?, output_paths_json = ?
        WHERE id = ?
        """,
        (_now(), json.dumps(output_paths, ensure_ascii=False), run_id),
    )


def record_preprocess_failed(
    conn: sqlite3.Connection, run_id: int, error: str
) -> None:
    conn.execute(
        """
        UPDATE preprocess_runs
        SET status = 'failed', finished_at = ?, error = ?
        WHERE id = ?
        """,
        (_now(), error[:2000], run_id),
    )


# ── download batches ──────────────────────────────────────────────────────────

def create_download_batch(
    conn: sqlite3.Connection,
    *,
    batch_id: str,
    provider: str,
    profile_hash: str | None = None,
    profile_json: dict[str, Any] | None = None,
    run_folder: str | None = None,
) -> int:
    """Insert a new download batch record; return its integer PK."""
    now = _now()
    conn.execute(
        """
        INSERT OR IGNORE INTO download_batches
          (batch_id, provider, profile_hash, profile_json, run_folder, status, created_at)
        VALUES (?, ?, ?, ?, ?, 'in_progress', ?)
        """,
        (
            batch_id,
            provider,
            profile_hash,
            json.dumps(profile_json or {}, ensure_ascii=False),
            run_folder,
            now,
        ),
    )
    row = conn.execute(
        "SELECT id FROM download_batches WHERE batch_id = ?", (batch_id,)
    ).fetchone()
    return int(row["id"])


def get_download_batch(conn: sqlite3.Connection, batch_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM download_batches WHERE batch_id = ?", (batch_id,)
    ).fetchone()


def update_batch_status(conn: sqlite3.Connection, batch_db_id: int, status: str) -> None:
    finished = _now() if status in ("complete", "partial", "failed") else None
    conn.execute(
        """
        UPDATE download_batches SET status = ?, finished_at = COALESCE(?, finished_at)
        WHERE id = ?
        """,
        (status, finished, batch_db_id),
    )


def list_download_batches(
    conn: sqlite3.Connection, provider: str | None = None
) -> list[sqlite3.Row]:
    if provider:
        return conn.execute(
            "SELECT * FROM download_batches WHERE provider = ? ORDER BY created_at DESC",
            (provider,),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM download_batches ORDER BY created_at DESC"
    ).fetchall()


def purge_deletion_events_when_no_download_batches(conn: sqlite3.Connection) -> int:
    """If there are no download batch rows, clear persistent delete history.

    Intended for explicit repair flows.  The Stage-0 inventory catalogue no
    longer calls this (so delete history survives on-disk-only layouts with no
    ``download_batches`` rows).  Full reset clears ``deletion_events`` via
    ``DELETE`` in the reset endpoint instead.

    Returns rows deleted, or 0.
    """
    n = int(conn.execute("SELECT COUNT(*) FROM download_batches").fetchone()[0])
    if n:
        return 0
    try:
        cur = conn.execute("DELETE FROM deletion_events")
        return int(getattr(cur, "rowcount", 0) or 0)
    except sqlite3.OperationalError:
        return 0


# ── batch items ───────────────────────────────────────────────────────────────

def upsert_batch_item(
    conn: sqlite3.Connection,
    *,
    batch_db_id: int,
    dataset_id: int | None,
    stable_id: str,
    asset_type: str,
    local_path: str | None = None,
    status: str = "pending",
) -> int:
    now = _now()
    completed = now if status in ("present",) else None
    conn.execute(
        """
        INSERT INTO batch_items
          (batch_db_id, dataset_id, stable_id, asset_type, local_path, status, completed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(batch_db_id, stable_id, asset_type) DO UPDATE SET
          local_path   = COALESCE(excluded.local_path, local_path),
          status       = excluded.status,
          completed_at = COALESCE(excluded.completed_at, completed_at),
          dataset_id   = COALESCE(excluded.dataset_id, dataset_id)
        """,
        (batch_db_id, dataset_id, stable_id, asset_type, local_path, status, completed),
    )
    row = conn.execute(
        """
        SELECT id FROM batch_items
        WHERE batch_db_id = ? AND stable_id = ? AND asset_type = ?
        """,
        (batch_db_id, stable_id, asset_type),
    ).fetchone()
    return int(row["id"])


def update_batch_item_status(
    conn: sqlite3.Connection, item_id: int, status: str
) -> None:
    now = _now()
    completed = now if status in ("present",) else None
    conn.execute(
        """
        UPDATE batch_items
        SET status = ?,
            completed_at = CASE WHEN ? IS NOT NULL THEN ? ELSE completed_at END
        WHERE id = ?
        """,
        (status, completed, completed, item_id),
    )


def list_batch_items(
    conn: sqlite3.Connection, batch_db_id: int
) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM batch_items WHERE batch_db_id = ? ORDER BY stable_id, asset_type",
        (batch_db_id,),
    ).fetchall()


def reconcile_batch_items(conn: sqlite3.Connection, batch_db_id: int) -> int:
    """Check local file existence for all batch items; update stale statuses.

    Returns the number of items whose status changed (present→missing or
    missing→present based on current disk state).
    """
    items = list_batch_items(conn, batch_db_id)
    changed = 0
    for item in items:
        lp_str = item["local_path"]
        if not lp_str:
            continue
        lp = Path(lp_str)
        exists = lp.is_file() or lp.is_dir()
        current = item["status"]
        if exists and current == "missing_or_deleted_local":
            update_batch_item_status(conn, int(item["id"]), "present")
            changed += 1
        elif not exists and current == "present":
            update_batch_item_status(conn, int(item["id"]), "missing_or_deleted_local")
            changed += 1
    return changed


def get_batch_items_by_stable_id(
    conn: sqlite3.Connection, batch_db_id: int, stable_id: str
) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM batch_items WHERE batch_db_id = ? AND stable_id = ?",
        (batch_db_id, stable_id),
    ).fetchall()


def find_missing_batch_items(
    conn: sqlite3.Connection, batch_db_id: int
) -> list[sqlite3.Row]:
    """Return batch items that are missing or deleted on disk (file does not exist)."""
    items = list_batch_items(conn, batch_db_id)
    missing: list[sqlite3.Row] = []
    for item in items:
        lp_str = item["local_path"]
        if not lp_str:
            missing.append(item)
            continue
        lp = Path(lp_str)
        if not (lp.is_file() or lp.is_dir()):
            missing.append(item)
    return missing


# ── dataset training visibility ───────────────────────────────────────────────

def hide_dataset(conn: sqlite3.Connection, dataset_id: int) -> None:
    """Exclude dataset from training datalist/YAML generation (soft hide)."""
    conn.execute(
        "UPDATE datasets SET hidden_from_training = 1 WHERE id = ?", (dataset_id,)
    )


def unhide_dataset(conn: sqlite3.Connection, dataset_id: int) -> None:
    """Re-include dataset in training datalist/YAML generation."""
    conn.execute(
        "UPDATE datasets SET hidden_from_training = 0 WHERE id = ?", (dataset_id,)
    )


def is_dataset_hidden(conn: sqlite3.Connection, dataset_id: int) -> bool:
    row = conn.execute(
        "SELECT hidden_from_training FROM datasets WHERE id = ?", (dataset_id,)
    ).fetchone()
    return bool(row and row["hidden_from_training"])


def list_hidden_datasets(conn: sqlite3.Connection, provider_id: int) -> list[int]:
    """Return dataset IDs that are hidden from training for a given provider."""
    rows = conn.execute(
        "SELECT id FROM datasets WHERE provider_id = ? AND hidden_from_training = 1",
        (provider_id,),
    ).fetchall()
    return [int(r["id"]) for r in rows]


def get_dataset_by_stable_id(
    conn: sqlite3.Connection, provider_id: int, stable_id: str
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM datasets WHERE provider_id = ? AND stable_id = ?",
        (provider_id, stable_id),
    ).fetchone()


def delete_batch_item_by_path(
    conn: sqlite3.Connection, local_path: str
) -> int:
    """Mark batch items whose local_path matches as missing_or_deleted_local.

    Returns count of rows updated.
    """
    cur = conn.execute(
        """
        UPDATE batch_items
        SET status = 'missing_or_deleted_local'
        WHERE local_path = ? AND status = 'present'
        """,
        (local_path,),
    )
    return cur.rowcount
