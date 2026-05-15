"""BossDB provider adapter for the central registry.

Discovery:  calls bossdb.metadata_client.discover() (live API).
Assets:     normalises BossDBDataset channel pairs → AssetSpec list.
ScriptGen:  delegates to bossdb.scriptgen.write_bossdb_outputs().
"""

from __future__ import annotations

import logging
import re
import sqlite3
from typing import Any

from ..api import upsert_provider, upsert_dataset, upsert_asset, get_dataset_id
from .base import AssetSpec
from .common import ensure_import_path, repo_root
from .downloader_codegen import generate_bossdb_script

# Repo root: .../mitoFoundation2 (not .../orchestration).
_REPO_ROOT = repo_root()
_LOG = logging.getLogger(__name__)

# Add stage-3 package to path so we can import bossdb.*
_STAGE3 = _REPO_ROOT / "3data_downloader"
ensure_import_path(_STAGE3)
_INV_ROOT = _REPO_ROOT / "0inventory"
ensure_import_path(_INV_ROOT)

class BossDBProvider:
    """Provider adapter for BossDB datasets."""

    name     = "BossDB"
    base_url = "https://api.bossdb.io"

    def __init__(
        self,
        probe_path: Path | None = None,
        catalog_db: Path | None = None,
    ) -> None:
        # Accepted for factory parity with OpenOrganelleProvider; BossDB uses live API.
        _ = (probe_path, catalog_db)
        self._datasets_cache: list[Any] | None = None

    def _discover(self, *, refresh: bool = False) -> list[Any]:
        from bossdb.metadata_client import discover as _discover

        if refresh or self._datasets_cache is None:
            self._datasets_cache = _discover()
        return self._datasets_cache

    # ── BaseProvider protocol ─────────────────────────────────────────────────

    def ingest_discovery(self, conn: sqlite3.Connection) -> list[str]:
        """Load BossDB channels into the registry; return newly-added stable IDs."""
        _LOG.info("BossDBProvider: running live discovery …")
        datasets = self._discover(refresh=True)

        provider_id = upsert_provider(conn, name=self.name, base_url=self.base_url)
        conn.commit()

        new_ids: list[str] = []
        for ds in datasets:
            if not ds.project_id:
                continue
            existing = get_dataset_id(conn, provider_id, ds.project_id)
            upsert_dataset(
                conn,
                provider_id=provider_id,
                stable_id=ds.project_id,
                display_name=ds.project_name or ds.project_id,
                metadata={
                    "channel_type": ds.channel_type,
                    "data_type":    ds.data_type,
                    "organism":     ds.organism,
                    "modality":     ds.modality,
                    "tags":         ds.tags,
                    "voxel_size_nm": ds.voxel_size_nm,
                    "uri":          ds.uri,
                },
                changed=(existing is None),
            )
            if existing is None:
                new_ids.append(ds.project_id)

        conn.commit()
        _LOG.info("BossDBProvider.ingest_discovery: %d new dataset(s)", len(new_ids))
        return new_ids

    def resolve_assets(self, conn: sqlite3.Connection) -> list[AssetSpec]:
        """Return paired (em_volume, mito_seg) asset specs from live discovery."""
        from bossdb_inventory import pair_labeled_datasets

        datasets = self._discover(refresh=False)
        pairs    = pair_labeled_datasets(datasets)
        provider_id = upsert_provider(conn, name=self.name, base_url=self.base_url)

        specs: list[AssetSpec] = []
        for p in pairs:
            stable_id = p["project_id"]
            dataset_id = get_dataset_id(conn, provider_id, stable_id)
            if dataset_id is None:
                dataset_id = upsert_dataset(
                    conn,
                    provider_id=provider_id,
                    stable_id=stable_id,
                    display_name=stable_id,
                    metadata={
                        "collection": p.get("collection"),
                        "experiment": p.get("experiment"),
                        "voxel_size_nm": p.get("voxel_size_nm") or [],
                    },
                )
            upsert_asset(
                conn,
                dataset_id=dataset_id,
                asset_type="em_volume",
                remote_url=p["img_uri"],
            )
            upsert_asset(
                conn,
                dataset_id=dataset_id,
                asset_type="mito_seg",
                remote_url=p["seg_uri"],
            )
            specs.append(AssetSpec(
                stable_dataset_id = stable_id,
                asset_type        = "em_volume",
                remote_url        = p["img_uri"],
                voxel_size_nm     = p.get("voxel_size_nm") or None,
                metadata          = {"channel": p["img_channel"]},
            ))
            specs.append(AssetSpec(
                stable_dataset_id = stable_id,
                asset_type        = "mito_seg",
                remote_url        = p["seg_uri"],
                voxel_size_nm     = p.get("voxel_size_nm") or None,
                metadata          = {"channel": p["seg_channel"]},
            ))
        conn.commit()
        return specs

    def labeled_inventory_stable_ids(self) -> frozenset[str]:
        """Experiment-level IDs used by BossDB script generation."""
        from bossdb_inventory import pair_labeled_datasets

        datasets = self._discover(refresh=False)
        pairs = pair_labeled_datasets(datasets)
        return frozenset(str(p.get("project_id", "")).strip() for p in pairs if p.get("project_id"))

    def get_spacing_nm(self, stable_id: str) -> list[float] | None:
        key = (stable_id or "").strip()
        if not key:
            return None
        m = re.match(r"^(.*?)(?:_vol(\d+)|_(\d+))$", key, flags=re.IGNORECASE)
        base = (m.group(1).strip() if m else key)
        datasets = self._discover(refresh=False)
        for ds in datasets:
            pid = str(getattr(ds, "project_id", "") or "").strip()
            if not pid:
                continue
            if key == pid or key.startswith(f"{pid}_") or base == pid:
                vox = getattr(ds, "voxel_size_nm", None)
                if vox:
                    return vox
        return None

    def generate_downloader_script(
        self,
        *,
        mode: str = "labeled",
        chunk_zyx: str = "128,128,128",
        n_crops: int = 1,
        voxel_size_nm_zyx: str = "16.0,16.0,16.0",
        **kwargs: Any,
    ) -> Path | None:
        if mode != "labeled":
            return None

        datasets = self._discover(refresh=True)
        if not datasets:
            _LOG.warning("BossDBProvider.generate_downloader_script: live discovery returned no datasets.")
            return None

        return generate_bossdb_script(
            datasets=datasets,
            mode=mode,
            chunk_zyx=chunk_zyx,
            n_crops=n_crops,
            voxel_size_nm_zyx=voxel_size_nm_zyx,
            dataset_splits=kwargs.get("dataset_splits"),
        )

    def download_profile_foundation(self, *, mode: str, foundation_query: bool) -> bool:  # noqa: ARG002
        """BossDB labeled scripts do not use OpenOrganelle-style foundation resampling."""
        return False

    def studio_env_vars(
        self,
        *,
        chunk_zyx: str,
        n_crops: int,
        voxel_size_nm_zyx: str,
        physical_zyx: str,
        no_foundation: bool = False,
    ) -> dict[str, str]:
        return {}
