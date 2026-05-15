"""Tests verifying 128^3 dimension defaults are consistent across the codebase.

Checks:
- registry/api.py DEFAULT_DOWNLOAD_PROFILE_HASH encodes 128^3
- LEGACY_512_DOWNLOAD_PROFILE_HASH encodes 512^3 (backward compat)
- The two hashes are distinct
- Legacy NULL download rows still satisfy the 512^3 profile check
- Legacy rows do NOT satisfy the new 128^3 default profile check
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from agent.orchestration.registry.schema import migrate
from agent.orchestration.registry.api import (
    DEFAULT_DOWNLOAD_PROFILE_HASH,
    LEGACY_512_DOWNLOAD_PROFILE_HASH,
    make_download_profile_hash,
    upsert_provider,
    upsert_dataset,
    upsert_asset,
    record_download_start,
    record_download_complete,
    find_complete_download_for_profile,
    is_asset_downloaded_for_profile,
    find_complete_download_for_dataset_type_profile,
    is_dataset_type_downloaded_for_profile,
)


def _mem() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    migrate(conn)
    return conn


class TestDefaultHashes:
    def test_default_hash_encodes_128(self):
        expected = make_download_profile_hash(
            n_crops=1,
            chunk_zyx=(128, 128, 128),
            voxel_nm_zyx=(16.0, 16.0, 16.0),
            mode="labeled",
            foundation=True,
        )
        assert DEFAULT_DOWNLOAD_PROFILE_HASH == expected, (
            "DEFAULT_DOWNLOAD_PROFILE_HASH must encode 128^3 voxels"
        )

    def test_legacy_hash_encodes_512(self):
        expected = make_download_profile_hash(
            n_crops=1,
            chunk_zyx=(512, 512, 512),
            voxel_nm_zyx=(16.0, 16.0, 16.0),
            mode="labeled",
            foundation=True,
        )
        assert LEGACY_512_DOWNLOAD_PROFILE_HASH == expected, (
            "LEGACY_512_DOWNLOAD_PROFILE_HASH must encode 512^3 voxels"
        )

    def test_128_and_512_hashes_differ(self):
        assert DEFAULT_DOWNLOAD_PROFILE_HASH != LEGACY_512_DOWNLOAD_PROFILE_HASH, (
            "128^3 and 512^3 profiles must produce different hashes"
        )


class TestLegacyNullRowCompat:
    """Legacy rows (download_profile_hash IS NULL) must match the 512^3 legacy profile
    but NOT the new 128^3 default profile.
    """

    def _create_legacy_null_download(self, conn, tmp_path: Path) -> tuple[int, int]:
        """Seed a download row with NULL profile hash (simulates pre-migration record)."""
        pid = upsert_provider(conn, name="OpenOrganelle", base_url="")
        did = upsert_dataset(conn, provider_id=pid, stable_id="jrc_test")
        aid = upsert_asset(conn, dataset_id=did, asset_type="em_volume",
                           remote_url="s3://bucket/test")
        local = tmp_path / "test_dir"
        local.mkdir()
        # Insert NULL-hash row directly (simulates old migration state).
        conn.execute(
            "INSERT INTO downloads (asset_id, local_path, status, finished_at, download_profile_hash) "
            "VALUES (?, ?, 'complete', '2024-01-01T00:00:00+00:00', NULL)",
            (aid, str(local)),
        )
        conn.commit()
        return did, aid

    def test_null_row_satisfies_legacy_512_profile(self, tmp_path: Path):
        conn = _mem()
        did, aid = self._create_legacy_null_download(conn, tmp_path)
        assert is_asset_downloaded_for_profile(conn, aid, LEGACY_512_DOWNLOAD_PROFILE_HASH), (
            "NULL-hash row must satisfy the legacy 512^3 profile"
        )

    def test_null_row_does_not_satisfy_128_default(self, tmp_path: Path):
        conn = _mem()
        did, aid = self._create_legacy_null_download(conn, tmp_path)
        assert not is_asset_downloaded_for_profile(conn, aid, DEFAULT_DOWNLOAD_PROFILE_HASH), (
            "NULL-hash (legacy 512^3) row must NOT satisfy the new 128^3 default profile"
        )

    def test_explicit_128_row_satisfies_128_profile(self, tmp_path: Path):
        conn = _mem()
        pid = upsert_provider(conn, name="OpenOrganelle", base_url="")
        did = upsert_dataset(conn, provider_id=pid, stable_id="jrc_test2")
        aid = upsert_asset(conn, dataset_id=did, asset_type="em_volume",
                           remote_url="s3://bucket/test2")
        local = tmp_path / "test2_dir"
        local.mkdir()
        dl_id = record_download_start(conn, aid, str(local),
                                      download_profile_hash=DEFAULT_DOWNLOAD_PROFILE_HASH)
        record_download_complete(conn, dl_id)
        conn.commit()
        assert is_asset_downloaded_for_profile(conn, aid, DEFAULT_DOWNLOAD_PROFILE_HASH)
        assert not is_asset_downloaded_for_profile(conn, aid, LEGACY_512_DOWNLOAD_PROFILE_HASH)


class TestDimensionDefaultConsistency:
    def test_plan_downloads_uses_128_default(self, tmp_path: Path):
        """Verify that plan_downloads with the 128^3 profile skips a 128^3 complete download
        but not a 512^3 complete download.
        """
        from agent.orchestration.registry.planners import plan_downloads
        from agent.orchestration.registry.providers.base import AssetSpec

        conn = _mem()
        pid = upsert_provider(conn, name="OpenOrganelle", base_url="")
        did = upsert_dataset(conn, provider_id=pid, stable_id="jrc_hela-2")
        url = "s3://bucket/hela"
        aid = upsert_asset(conn, dataset_id=did, asset_type="em_volume", remote_url=url)
        conn.commit()

        local = tmp_path / "hela_dir"
        local.mkdir()
        dl_id = record_download_start(conn, aid, str(local),
                                      download_profile_hash=DEFAULT_DOWNLOAD_PROFILE_HASH)
        record_download_complete(conn, dl_id)
        conn.commit()

        class _FP:
            name = "OpenOrganelle"
            base_url = ""
            def ingest_discovery(self, c): return []
            def resolve_assets(self, c):
                return [AssetSpec(stable_dataset_id="jrc_hela-2", asset_type="em_volume", remote_url=url)]
            def get_spacing_nm(self, s): return None

        plans_128 = plan_downloads(conn, _FP(), download_profile_hash=DEFAULT_DOWNLOAD_PROFILE_HASH)
        plans_512 = plan_downloads(conn, _FP(), download_profile_hash=LEGACY_512_DOWNLOAD_PROFILE_HASH)

        assert plans_128 == [], "128^3 profile: already downloaded → plan must be empty"
        assert len(plans_512) == 1, "512^3 profile: no 512^3 download exists → plan must have 1 item"
