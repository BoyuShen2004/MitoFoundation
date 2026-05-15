"""Incremental planners: compute what work needs to be done.

Each planner returns a plan list — callers decide how to execute it.
Planners only read from the registry + provider; they never download or modify files.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .api import (
    get_asset,
    get_provider_id,
    is_asset_downloaded,
    is_asset_downloaded_for_profile,
    is_dataset_type_downloaded_for_profile,
    find_complete_preprocess_run,
    fingerprint_config,
    fingerprint_file,
    list_batch_items,
    find_missing_batch_items,
    reconcile_batch_items,
)
from .providers.base import BaseProvider, AssetSpec


# ── plan_discovery ─────────────────────────────────────────────────────────────

@dataclass
class DiscoveryPlan:
    new_stable_ids: list[str]
    total_in_registry: int


def plan_discovery(conn: sqlite3.Connection, provider: BaseProvider) -> DiscoveryPlan:
    """Ingest discovery data and report which dataset IDs are newly registered."""
    new_ids = provider.ingest_discovery(conn)
    provider_id = get_provider_id(conn, provider.name) or 0
    total = conn.execute(
        "SELECT COUNT(*) FROM datasets WHERE provider_id = ?", (provider_id,)
    ).fetchone()[0]
    return DiscoveryPlan(new_stable_ids=new_ids, total_in_registry=total)


# ── plan_downloads ─────────────────────────────────────────────────────────────

@dataclass
class AssetDownloadPlan:
    stable_dataset_id: str
    asset_type: str            # "em_volume" | "mito_seg"
    remote_url: str
    asset_id: int
    reason: str                # "missing" | "no_local_file"
    download_profile_hash: str | None = None  # None → legacy / no profile


def plan_downloads(
    conn: sqlite3.Connection,
    provider: BaseProvider,
    *,
    download_profile_hash: str | None = None,
    stable_id_allowlist: frozenset[str] | None = None,
) -> list[AssetDownloadPlan]:
    """Return assets that have no complete, locally-present download.

    When ``download_profile_hash`` is supplied (e.g. the result of
    :func:`registry.api.make_download_profile_hash`), the skip check is
    profile-aware: an asset completed under a *different* profile (e.g. a
    different n_crops value) is **not** considered done and will appear in the
    returned list.  Pass ``None`` for the legacy profile-agnostic behavior.

    When ``stable_id_allowlist`` is set (e.g. OpenOrganelle labeled training
    inventory), only assets whose ``stable_dataset_id`` is in the set are
    considered — matching the datasets the labeled downloader script will run.
    """
    specs = provider.resolve_assets(conn)
    if stable_id_allowlist is not None:
        specs = [s for s in specs if s.stable_dataset_id in stable_id_allowlist]
    plans: list[AssetDownloadPlan] = []

    provider_id = get_provider_id(conn, provider.name)
    if provider_id is None:
        return plans

    for spec in specs:
        from .api import get_dataset_id
        dataset_id = get_dataset_id(conn, provider_id, spec.stable_dataset_id)
        if dataset_id is None:
            continue

        if download_profile_hash is not None:
            # URL-churn-safe check: any asset row for this (dataset, type) satisfies skip.
            if is_dataset_type_downloaded_for_profile(
                conn, dataset_id, spec.asset_type, download_profile_hash
            ):
                continue

        asset = get_asset(conn, dataset_id, spec.asset_type)
        if asset is None:
            continue
        asset_id = int(asset["id"])

        if download_profile_hash is None:
            if is_asset_downloaded(conn, asset_id):
                continue

        plans.append(AssetDownloadPlan(
            stable_dataset_id=spec.stable_dataset_id,
            asset_type=spec.asset_type,
            remote_url=spec.remote_url,
            asset_id=asset_id,
            reason="missing",
            download_profile_hash=download_profile_hash,
        ))

    return plans


def plan_downloads_summary(plans: list[AssetDownloadPlan]) -> str:
    em = sum(1 for p in plans if p.asset_type == "em_volume")
    seg = sum(1 for p in plans if p.asset_type == "mito_seg")
    return f"{len(plans)} assets to download (em_volume={em}, mito_seg={seg})"


# ── plan_preprocess ────────────────────────────────────────────────────────────

@dataclass
class PreprocessPlan:
    input_path: Path
    input_fingerprint: str
    config_fingerprint: str
    dataset_id: int | None
    reason: str           # "no_run" | "no_output_file"


def plan_preprocess(
    conn: sqlite3.Connection,
    raw_files: list[Path],
    config: dict[str, Any],
    *,
    provider: BaseProvider | None = None,
) -> list[PreprocessPlan]:
    """Return raw files that need preprocessing given the config fingerprint."""
    config_fp = fingerprint_config(config)
    plans: list[PreprocessPlan] = []

    provider_id = get_provider_id(conn, provider.name) if provider else None

    for raw_path in raw_files:
        if not raw_path.is_file():
            continue
        input_fp = fingerprint_file(raw_path)
        if not input_fp:
            continue

        run = find_complete_preprocess_run(conn, input_fp, config_fp)
        if run is not None:
            output_paths = _parse_output_paths(run["output_paths_json"])
            if output_paths and all(Path(p).is_file() for p in output_paths):
                continue
            reason = "no_output_file"
        else:
            reason = "no_run"

        dataset_id: int | None = None
        if provider_id is not None:
            from .api import get_dataset_id
            stem = _stem_to_dataset_id_guess(raw_path)
            if stem:
                dataset_id = get_dataset_id(conn, provider_id, stem)

        plans.append(PreprocessPlan(
            input_path=raw_path,
            input_fingerprint=input_fp,
            config_fingerprint=config_fp,
            dataset_id=dataset_id,
            reason=reason,
        ))

    return plans


# ── plan_batch_redownload ──────────────────────────────────────────────────────

@dataclass
class BatchItemRedoPlan:
    batch_db_id: int
    item_id: int
    stable_id: str
    asset_type: str
    local_path: str | None
    reason: str  # "missing_local_file" | "no_local_path"


def plan_batch_redownload(
    conn: sqlite3.Connection,
    batch_db_id: int,
) -> list[BatchItemRedoPlan]:
    """Return batch items whose local file/dir is absent from disk.

    Reconciles item statuses in-place so the registry stays current,
    then returns items that need to be re-fetched.
    """
    reconcile_batch_items(conn, batch_db_id)
    conn.commit()

    missing = find_missing_batch_items(conn, batch_db_id)
    plans: list[BatchItemRedoPlan] = []
    for item in missing:
        lp = item["local_path"]
        reason = "no_local_path" if not lp else "missing_local_file"
        plans.append(BatchItemRedoPlan(
            batch_db_id=batch_db_id,
            item_id=int(item["id"]),
            stable_id=item["stable_id"],
            asset_type=item["asset_type"],
            local_path=lp,
            reason=reason,
        ))
    return plans


def plan_batch_redownload_summary(plans: list[BatchItemRedoPlan]) -> str:
    return f"{len(plans)} batch item(s) need re-download"


def _parse_output_paths(json_str: str | None) -> list[str]:
    if not json_str:
        return []
    try:
        import json
        v = json.loads(json_str)
        return v if isinstance(v, list) else []
    except Exception:
        return []


def _stem_to_dataset_id_guess(path: Path) -> str:
    """Heuristic: map a raw H5 filename back to a dataset slug."""
    import re
    name = path.stem
    name = re.sub(r"_(im|seg)$", "", name, flags=re.I)
    name = re.sub(r"\.(im|seg)$", "", name, flags=re.I)
    name = re.sub(r"_vol\d+$", "", name, flags=re.I)
    return name.strip()
