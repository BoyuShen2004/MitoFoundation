"""Tests for n_crops-aware download profile identity.

Invariants:
- complete download for n_crops=1 → skip when asked again with n_crops=1
- complete for n_crops=1 → do NOT skip when asked with n_crops=2
- nothing pending → plan_downloads returns empty list
- legacy rows (NULL profile hash) count as the LEGACY 512^3 profile, not the new 128^3 default
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
import sqlite3

from agent.orchestration.registry.schema import open_registry
from agent.orchestration.registry.api import (
    DEFAULT_DOWNLOAD_PROFILE_HASH,
    LEGACY_512_DOWNLOAD_PROFILE_HASH,
    make_download_profile_hash,
    upsert_provider,
    upsert_dataset,
    upsert_asset,
    record_download_start,
    record_download_complete,
    is_asset_downloaded,
    is_asset_downloaded_for_profile,
    find_complete_download_for_profile,
)
from agent.orchestration.registry.planners import plan_downloads


# ── helpers ────────────────────────────────────────────────────────────────────

def _in_memory_registry() -> sqlite3.Connection:
    """Return an in-memory registry DB (already migrated)."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    from agent.orchestration.registry.schema import migrate
    migrate(conn)
    return conn


def _seed_asset(conn: sqlite3.Connection) -> tuple[int, int]:
    """Insert a provider + dataset + em_volume asset; return (dataset_id, asset_id)."""
    provider_id = upsert_provider(conn, name="TestProvider", base_url="https://example.com")
    dataset_id = upsert_dataset(conn, provider_id=provider_id, stable_id="test-ds-001")
    asset_id = upsert_asset(conn, dataset_id=dataset_id, asset_type="em_volume",
                            remote_url="s3://bucket/test-ds-001/em")
    conn.commit()
    return dataset_id, asset_id


def _fake_local_path(tmp_path: Path, name: str = "output") -> Path:
    p = tmp_path / name
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── make_download_profile_hash ─────────────────────────────────────────────────

class TestMakeDownloadProfileHash:
    def test_deterministic(self):
        h1 = make_download_profile_hash(
            n_crops=1, chunk_zyx=(512, 512, 512), voxel_nm_zyx=(16.0, 16.0, 16.0)
        )
        h2 = make_download_profile_hash(
            n_crops=1, chunk_zyx=(512, 512, 512), voxel_nm_zyx=(16.0, 16.0, 16.0)
        )
        assert h1 == h2

    def test_different_n_crops_different_hash(self):
        h1 = make_download_profile_hash(
            n_crops=1, chunk_zyx=(512, 512, 512), voxel_nm_zyx=(16.0, 16.0, 16.0)
        )
        h4 = make_download_profile_hash(
            n_crops=4, chunk_zyx=(512, 512, 512), voxel_nm_zyx=(16.0, 16.0, 16.0)
        )
        assert h1 != h4

    def test_different_chunk_different_hash(self):
        h1 = make_download_profile_hash(
            n_crops=1, chunk_zyx=(512, 512, 512), voxel_nm_zyx=(16.0, 16.0, 16.0)
        )
        h2 = make_download_profile_hash(
            n_crops=1, chunk_zyx=(256, 256, 256), voxel_nm_zyx=(16.0, 16.0, 16.0)
        )
        assert h1 != h2

    def test_different_voxel_different_hash(self):
        h1 = make_download_profile_hash(
            n_crops=1, chunk_zyx=(512, 512, 512), voxel_nm_zyx=(16.0, 16.0, 16.0)
        )
        h2 = make_download_profile_hash(
            n_crops=1, chunk_zyx=(512, 512, 512), voxel_nm_zyx=(8.0, 8.0, 8.0)
        )
        assert h1 != h2

    def test_default_profile_hash_is_128(self):
        # Default hash now encodes 128^3; legacy 512^3 hash is LEGACY_512_DOWNLOAD_PROFILE_HASH.
        expected_128 = make_download_profile_hash(
            n_crops=1,
            chunk_zyx=(128, 128, 128),
            voxel_nm_zyx=(16.0, 16.0, 16.0),
            mode="labeled",
            foundation=True,
        )
        assert DEFAULT_DOWNLOAD_PROFILE_HASH == expected_128

    def test_legacy_512_profile_hash_stable(self):
        # The legacy 512^3 hash is used for backward-compat NULL-row matching.
        expected_512 = make_download_profile_hash(
            n_crops=1,
            chunk_zyx=(512, 512, 512),
            voxel_nm_zyx=(16.0, 16.0, 16.0),
            mode="labeled",
            foundation=True,
        )
        assert LEGACY_512_DOWNLOAD_PROFILE_HASH == expected_512

    def test_16_hex_chars(self):
        h = make_download_profile_hash(
            n_crops=1, chunk_zyx=(512, 512, 512), voxel_nm_zyx=(16.0, 16.0, 16.0)
        )
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)


# ── is_asset_downloaded_for_profile ───────────────────────────────────────────

class TestIsAssetDownloadedForProfile:
    def test_complete_n_crops1_skips_on_n_crops1(self, tmp_path: Path):
        """Primary regression: same profile → skip."""
        conn = _in_memory_registry()
        _, asset_id = _seed_asset(conn)
        local = _fake_local_path(tmp_path, "ds001")
        profile_1 = make_download_profile_hash(
            n_crops=1, chunk_zyx=(512, 512, 512), voxel_nm_zyx=(16.0, 16.0, 16.0)
        )
        dl_id = record_download_start(conn, asset_id, str(local),
                                      download_profile_hash=profile_1)
        record_download_complete(conn, dl_id)
        conn.commit()

        assert is_asset_downloaded_for_profile(conn, asset_id, profile_1) is True

    def test_complete_n_crops1_does_not_skip_n_crops4(self, tmp_path: Path):
        """Core requirement: n_crops=1 complete must not block n_crops=4."""
        conn = _in_memory_registry()
        _, asset_id = _seed_asset(conn)
        local = _fake_local_path(tmp_path, "ds001")
        profile_1 = make_download_profile_hash(
            n_crops=1, chunk_zyx=(512, 512, 512), voxel_nm_zyx=(16.0, 16.0, 16.0)
        )
        profile_4 = make_download_profile_hash(
            n_crops=4, chunk_zyx=(512, 512, 512), voxel_nm_zyx=(16.0, 16.0, 16.0)
        )
        dl_id = record_download_start(conn, asset_id, str(local),
                                      download_profile_hash=profile_1)
        record_download_complete(conn, dl_id)
        conn.commit()

        assert is_asset_downloaded_for_profile(conn, asset_id, profile_4) is False

    def test_no_download_record_returns_false(self):
        conn = _in_memory_registry()
        _, asset_id = _seed_asset(conn)
        profile = make_download_profile_hash(
            n_crops=1, chunk_zyx=(512, 512, 512), voxel_nm_zyx=(16.0, 16.0, 16.0)
        )
        assert is_asset_downloaded_for_profile(conn, asset_id, profile) is False

    def test_local_path_must_exist(self, tmp_path: Path):
        """Complete record but deleted local path → not considered done."""
        conn = _in_memory_registry()
        _, asset_id = _seed_asset(conn)
        local = tmp_path / "deleted_folder"
        profile = DEFAULT_DOWNLOAD_PROFILE_HASH
        dl_id = record_download_start(conn, asset_id, str(local),
                                      download_profile_hash=profile)
        record_download_complete(conn, dl_id)
        conn.commit()

        # Path does not exist → should return False
        assert is_asset_downloaded_for_profile(conn, asset_id, profile) is False

    def test_legacy_null_profile_counts_as_legacy_512(self, tmp_path: Path):
        """Rows written before migration (NULL hash) count for the legacy 512^3 profile,
        NOT the new 128^3 default profile.
        """
        conn = _in_memory_registry()
        _, asset_id = _seed_asset(conn)
        local = _fake_local_path(tmp_path, "legacy")
        # Simulate a pre-migration row by passing no profile hash.
        dl_id = record_download_start(conn, asset_id, str(local),
                                      download_profile_hash=None)
        record_download_complete(conn, dl_id)
        conn.commit()

        # Legacy 512^3 profile → NULL row matches.
        assert is_asset_downloaded_for_profile(
            conn, asset_id, LEGACY_512_DOWNLOAD_PROFILE_HASH
        ) is True

        # New 128^3 default profile → NULL row must NOT match.
        assert is_asset_downloaded_for_profile(
            conn, asset_id, DEFAULT_DOWNLOAD_PROFILE_HASH
        ) is False

    def test_legacy_null_profile_does_not_match_non_default(self, tmp_path: Path):
        """Legacy NULL rows must not satisfy a non-default profile request."""
        conn = _in_memory_registry()
        _, asset_id = _seed_asset(conn)
        local = _fake_local_path(tmp_path, "legacy2")
        dl_id = record_download_start(conn, asset_id, str(local),
                                      download_profile_hash=None)
        record_download_complete(conn, dl_id)
        conn.commit()

        profile_4 = make_download_profile_hash(
            n_crops=4, chunk_zyx=(512, 512, 512), voxel_nm_zyx=(16.0, 16.0, 16.0)
        )
        assert is_asset_downloaded_for_profile(conn, asset_id, profile_4) is False


# ── plan_downloads ─────────────────────────────────────────────────────────────

class TestPlanDownloadsWithProfile:
    def _make_provider(self, conn: sqlite3.Connection, tmp_path: Path):
        """Build a tiny in-memory provider with one dataset already downloaded."""
        import sqlite3 as _sq3
        from agent.orchestration.registry.providers.base import AssetSpec

        # Seed registry with one completed download (n_crops=1).
        provider_id = upsert_provider(conn, name="TestProvider2",
                                      base_url="https://example2.com")
        dataset_id = upsert_dataset(conn, provider_id=provider_id,
                                    stable_id="ds-abc")
        asset_id = upsert_asset(conn, dataset_id=dataset_id,
                                asset_type="em_volume",
                                remote_url="s3://bucket/ds-abc/em")
        conn.commit()

        local = tmp_path / "ds_abc"
        local.mkdir()
        profile_1 = make_download_profile_hash(
            n_crops=1, chunk_zyx=(512, 512, 512), voxel_nm_zyx=(16.0, 16.0, 16.0)
        )
        dl_id = record_download_start(conn, asset_id, str(local),
                                      download_profile_hash=profile_1)
        record_download_complete(conn, dl_id)
        conn.commit()

        class _FakeProvider:
            name = "TestProvider2"
            base_url = "https://example2.com"

            def ingest_discovery(self, c): return []
            def resolve_assets(self, c):
                return [AssetSpec(
                    stable_dataset_id="ds-abc",
                    asset_type="em_volume",
                    remote_url="s3://bucket/ds-abc/em",
                )]
            def get_spacing_nm(self, sid): return None

        return _FakeProvider()

    def test_nothing_pending_for_completed_profile(self, tmp_path: Path):
        conn = _in_memory_registry()
        provider = self._make_provider(conn, tmp_path)
        profile_1 = make_download_profile_hash(
            n_crops=1, chunk_zyx=(512, 512, 512), voxel_nm_zyx=(16.0, 16.0, 16.0)
        )
        plans = plan_downloads(conn, provider, download_profile_hash=profile_1)
        assert plans == [], f"Expected no pending, got {plans}"

    def test_pending_for_different_n_crops(self, tmp_path: Path):
        conn = _in_memory_registry()
        provider = self._make_provider(conn, tmp_path)
        profile_4 = make_download_profile_hash(
            n_crops=4, chunk_zyx=(512, 512, 512), voxel_nm_zyx=(16.0, 16.0, 16.0)
        )
        plans = plan_downloads(conn, provider, download_profile_hash=profile_4)
        assert len(plans) == 1, f"Expected 1 pending for n_crops=4, got {plans}"
        assert plans[0].stable_dataset_id == "ds-abc"
        assert plans[0].download_profile_hash == profile_4

    def test_no_profile_uses_legacy_behavior(self, tmp_path: Path):
        """None profile_hash → backward-compat: any complete download blocks redownload."""
        conn = _in_memory_registry()
        provider = self._make_provider(conn, tmp_path)
        plans = plan_downloads(conn, provider, download_profile_hash=None)
        assert plans == [], "Legacy mode: completed asset should be skipped"


class TestPlanDownloadsAllowlist:
    """``stable_id_allowlist`` matches labeled-script inventory filtering (OpenOrganelle Studio)."""

    def test_allowlist_keeps_only_listed_datasets(self, tmp_path: Path):
        from agent.orchestration.registry.providers.base import AssetSpec

        conn = _in_memory_registry()
        pid = upsert_provider(conn, name="TestOO", base_url="https://example.com")
        for sid in ("keep-me", "drop-me"):
            did = upsert_dataset(
                conn,
                provider_id=pid,
                stable_id=sid,
                display_name=sid,
                metadata={},
            )
            upsert_asset(
                conn,
                dataset_id=did,
                asset_type="em_volume",
                remote_url=f"s3://bucket/{sid}/em",
            )
        conn.commit()

        class _TwoAssetProvider:
            name = "TestOO"
            base_url = "https://example.com"

            def ingest_discovery(self, c):
                return []

            def resolve_assets(self, c):
                return [
                    AssetSpec(
                        stable_dataset_id="keep-me",
                        asset_type="em_volume",
                        remote_url="s3://bucket/keep-me/em",
                    ),
                    AssetSpec(
                        stable_dataset_id="drop-me",
                        asset_type="em_volume",
                        remote_url="s3://bucket/drop-me/em",
                    ),
                ]

            def get_spacing_nm(self, sid):
                return None

        provider = _TwoAssetProvider()
        profile_1 = make_download_profile_hash(
            n_crops=1, chunk_zyx=(512, 512, 512), voxel_nm_zyx=(16.0, 16.0, 16.0)
        )
        plans_all = plan_downloads(conn, provider, download_profile_hash=profile_1)
        assert len(plans_all) == 2

        plans_f = plan_downloads(
            conn,
            provider,
            download_profile_hash=profile_1,
            stable_id_allowlist=frozenset({"keep-me"}),
        )
        assert len(plans_f) == 1
        assert plans_f[0].stable_dataset_id == "keep-me"
