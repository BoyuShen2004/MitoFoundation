"""OpenOrganelle provider adapter.

Discovery:  reads 1web_scraper_01/outputs/OpenOrganelle.probe.json
Assets:     reads 2database_builder/outputs/databases/OpenOrganelle.db (dataset_resolved)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

from ..api import (
    upsert_provider,
    upsert_dataset,
    upsert_asset,
    get_dataset_id,
)
from .base import AssetSpec
from .common import parse_voxel_json, repo_root, stable_id_variants
from .downloader_codegen import generate_openorganelle_script

# Repo root: .../mitoFoundation2 (not .../orchestration).
_REPO_ROOT = repo_root()
_LOG = logging.getLogger(__name__)


def _default_probe_path() -> Path:
    env = os.environ.get("OPENORGANELLE_PROBE_JSON", "").strip()
    if env:
        return Path(env)
    return _REPO_ROOT / "1web_scraper_01" / "outputs" / "OpenOrganelle.probe.json"


def _default_catalog_db() -> Path:
    env = os.environ.get("OPENORGANELLE_CATALOG_DB", "").strip()
    if env:
        return Path(env)
    candidates = [_REPO_ROOT / "2database_builder" / "outputs" / "databases" / "OpenOrganelle.db"]
    return next((p for p in candidates if p.is_file()), candidates[0])


class OpenOrganelleProvider:
    """Provider adapter for the OpenOrganelle (Janelia COSEM) dataset catalog."""

    name = "OpenOrganelle"
    base_url = "https://openorganelle.janelia.org"

    def __init__(
        self,
        probe_path: Path | None = None,
        catalog_db: Path | None = None,
    ) -> None:
        self._probe_path = probe_path or _default_probe_path()
        self._catalog_db = catalog_db or _default_catalog_db()
        self._voxel_cache: dict[str, list[float]] | None = None

    # ── discovery ─────────────────────────────────────────────────────────────

    def ingest_discovery(self, conn: sqlite3.Connection) -> list[str]:
        """Read probe JSON → upsert provider + datasets. Return newly-seen stable IDs."""
        if not self._probe_path.is_file():
            print(f"[WARN] OpenOrganelleProvider: probe JSON not found: {self._probe_path}")
            return []

        try:
            doc = json.loads(self._probe_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WARN] OpenOrganelleProvider: cannot read probe JSON: {exc}")
            return []

        provider_id = upsert_provider(conn, name=self.name, base_url=self.base_url)
        datasets_doc = doc.get("datasets") or {}
        if not isinstance(datasets_doc, dict):
            return []

        new_ids: list[str] = []
        for stable_id, payload in datasets_doc.items():
            stable_id = str(stable_id).strip()
            if not stable_id:
                continue
            if not isinstance(payload, dict):
                payload = {}
            meta = _extract_metadata(stable_id, payload)
            existing = get_dataset_id(conn, provider_id, stable_id)
            dataset_id = upsert_dataset(
                conn,
                provider_id=provider_id,
                stable_id=stable_id,
                display_name=stable_id,
                metadata=meta,
                changed=(existing is None),
            )
            if existing is None:
                new_ids.append(stable_id)

        conn.commit()
        print(
            f"[INFO] OpenOrganelleProvider.ingest_discovery: "
            f"{len(datasets_doc)} datasets in probe JSON, {len(new_ids)} newly registered"
        )
        return new_ids

    # ── asset resolution ──────────────────────────────────────────────────────

    def resolve_assets(self, conn: sqlite3.Connection) -> list[AssetSpec]:
        """Read dataset_resolved from catalog DB → return AssetSpec list."""
        if not self._catalog_db.is_file():
            print(f"[WARN] OpenOrganelleProvider: catalog DB not found: {self._catalog_db}")
            return []

        try:
            db_conn = sqlite3.connect(str(self._catalog_db))
            db_conn.row_factory = sqlite3.Row
            rows = db_conn.execute(
                """
                SELECT r.dataset_name, r.resolved_em_path, r.resolved_mito_seg_path,
                       r.resolved_voxel_nm, r.path_source, r.ready_labeled, r.ready_em_only
                FROM dataset_resolved r
                ORDER BY r.dataset_name
                """
            ).fetchall()
            db_conn.close()
        except Exception as exc:
            print(f"[WARN] OpenOrganelleProvider: cannot read catalog DB: {exc}")
            return []

        provider_id = upsert_provider(conn, name=self.name, base_url=self.base_url)
        specs: list[AssetSpec] = []

        for row in rows:
            name = (row["dataset_name"] or "").strip()
            em = (row["resolved_em_path"] or "").strip()
            seg = (row["resolved_mito_seg_path"] or "").strip()
            if not name or not em:
                continue

            voxel = parse_voxel_json(row["resolved_voxel_nm"])
            dataset_id = get_dataset_id(conn, provider_id, name)
            if dataset_id is None:
                dataset_id = upsert_dataset(
                    conn, provider_id=provider_id, stable_id=name,
                    display_name=name, metadata={"voxel_size_nm": voxel},
                )

            upsert_asset(conn, dataset_id=dataset_id, asset_type="em_volume", remote_url=em)
            specs.append(AssetSpec(
                stable_dataset_id=name,
                asset_type="em_volume",
                remote_url=em,
                voxel_size_nm=voxel,
            ))

            if seg:
                upsert_asset(conn, dataset_id=dataset_id, asset_type="mito_seg", remote_url=seg)
                specs.append(AssetSpec(
                    stable_dataset_id=name,
                    asset_type="mito_seg",
                    remote_url=seg,
                    voxel_size_nm=voxel,
                ))

        conn.commit()
        _LOG.debug(
            "OpenOrganelleProvider.resolve_assets: %d dataset_resolved rows -> %d asset specs",
            len(rows),
            len(specs),
        )
        return specs

    def labeled_inventory_stable_ids(self) -> frozenset[str]:
        """Dataset names in the same labeled set as the generated downloader script.

        Uses :func:`labeled_inventory_resolved.labeled_mito_inventory_from_resolved_db`
        (good mito masks, resolved paths, no prediction/inference URLs). Empty if the
        catalog DB is missing.
        """
        db = self._catalog_db.expanduser().resolve()
        if not db.is_file():
            return frozenset()
        from .common import ensure_import_path  # noqa: PLC0415

        ensure_import_path(_REPO_ROOT / "2database_builder" / "openorganelle")
        from labeled_inventory_resolved import labeled_mito_inventory_from_resolved_db  # noqa: PLC0415

        inv = labeled_mito_inventory_from_resolved_db(db, mito_only=True, require_paths=True)
        return frozenset(inv.keys())

    # ── downloader script generation ─────────────────────────────────────────

    def generate_downloader_script(
        self,
        *,
        mode: str,
        chunk_zyx: str,
        n_crops: int,
        voxel_size_nm_zyx: str,
        **kwargs: Any,
    ) -> Path | None:
        """Delegate to ``openorganelle.agent.generate_openorganelle_downloader_script``.

        Always reloads ``openorganelle.agent`` from disk so that edits to
        LABELED_TEMPLATE / UNLABELED_TEMPLATE take effect without a server restart.
        """
        return generate_openorganelle_script(
            mode=mode,
            chunk_zyx=chunk_zyx,
            n_crops=n_crops,
            voxel_size_nm_zyx=voxel_size_nm_zyx,
            dataset_splits=kwargs.get("dataset_splits"),
        )

    def download_profile_foundation(self, *, mode: str, foundation_query: bool) -> bool:
        """Labeled OpenOrganelle downloads use foundation-aware registry profile hashes."""
        return bool(foundation_query) if (mode or "").strip().lower() == "labeled" else False

    def studio_env_vars(
        self,
        *,
        chunk_zyx: str,
        n_crops: int,
        voxel_size_nm_zyx: str,
        physical_zyx: str,
        no_foundation: bool = False,
    ) -> dict[str, str]:
        """Return ``MITO2_STUDIO_OG_*`` env vars consumed by ``openorganelle/agent.py``."""
        return {
            "MITO2_STUDIO_OG_CHUNK": chunk_zyx,
            "MITO2_STUDIO_OG_N_CROPS": str(max(1, n_crops)),
            "MITO2_STUDIO_OG_VOXEL_NM": voxel_size_nm_zyx,
            "MITO2_STUDIO_OG_PHYSICAL_NM": physical_zyx,
            "MITO2_STUDIO_OG_NO_FOUNDATION": "1" if no_foundation else "0",
        }

    # ── spacing lookup ────────────────────────────────────────────────────────

    def get_spacing_nm(self, stable_id: str) -> list[float] | None:
        if self._voxel_cache is None:
            self._voxel_cache = self._build_voxel_cache()
        # Try exact match, then strip trailing numeric suffix (e.g. "_vol1")
        key = stable_id.strip()
        for probe in stable_id_variants(key):
            got = self._voxel_cache.get(probe)
            if got:
                return got
        return None

    def _build_voxel_cache(self) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        if not self._catalog_db.is_file():
            return out
        try:
            db_conn = sqlite3.connect(str(self._catalog_db))
            db_conn.row_factory = sqlite3.Row
            rows = db_conn.execute(
                """
                SELECT d.dataset_name, r.resolved_voxel_nm
                FROM dataset_resolved r JOIN datasets d ON d.id = r.dataset_id
                """
            ).fetchall()
            db_conn.close()
        except Exception:
            return out
        for row in rows:
            name = (row["dataset_name"] or "").strip()
            voxel = parse_voxel_json(row["resolved_voxel_nm"])
            if name and voxel:
                out[name] = voxel
        return out


# ── helpers ────────────────────────────────────────────────────────────────────

def _extract_metadata(stable_id: str, payload: dict) -> dict:
    meta: dict = {"probe_source": "OpenOrganelle.probe.json"}
    for k in ("first_seen_at_iso", "first_seen_scrape_batch", "first_seen_markdown_file",
              "em_image_path", "mito_seg_path", "voxel_size_nm"):
        if k in payload:
            meta[k] = payload[k]
    voxel = payload.get("voxel_size_nm")
    if isinstance(voxel, list) and len(voxel) >= 3:
        meta["voxel_size_nm"] = voxel
    return meta


