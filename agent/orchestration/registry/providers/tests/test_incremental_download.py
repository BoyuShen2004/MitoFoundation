"""Incremental download behavior tests.

Case A: first run — plan_downloads returns all assets (nothing downloaded yet).
Case B: second run with same profile — plan_downloads returns empty; no new folder created.
Case C: new dataset/asset appears after first run — only the new asset is planned.
Case D (URL churn): re-scrape changes remote_url; prior complete download still satisfies skip.
Case E (recording round-trip): record_download_start/complete → plan_downloads returns empty.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest

from agent.orchestration.registry.schema import open_registry, migrate
from agent.orchestration.registry.api import (
    make_download_profile_hash,
    LEGACY_512_DOWNLOAD_PROFILE_HASH,
    upsert_provider,
    upsert_dataset,
    upsert_asset,
    record_download_start,
    record_download_complete,
    is_asset_downloaded_for_profile,
    is_dataset_type_downloaded_for_profile,
    find_complete_download_for_dataset_type_profile,
)
from agent.orchestration.registry.planners import plan_downloads
from agent.orchestration.registry.providers.base import AssetSpec


# ── Helpers ────────────────────────────────────────────────────────────────────

def _in_memory_registry() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    migrate(conn)
    return conn


# Use the legacy 512^3 profile hash — these tests verify skip-logic behaviour
# for the old profile and must not be invalidated by the 128^3 default change.
_PROFILE = LEGACY_512_DOWNLOAD_PROFILE_HASH

_DATASETS = [
    ("jrc_hela-2", "s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5/em/fibsem-uint8"),
    ("jrc_jurkat-1", "s3://janelia-cosem-datasets/jrc_jurkat-1/jrc_jurkat-1.n5/em/fibsem-uint8"),
]


def _seed_provider_and_assets(
    conn: sqlite3.Connection,
    dataset_names: list[str] = None,
) -> tuple[int, dict[str, int]]:
    """Insert provider + datasets + em_volume assets. Return (provider_id, {name: asset_id})."""
    if dataset_names is None:
        dataset_names = [name for name, _ in _DATASETS]
    urls = dict(_DATASETS)
    pid = upsert_provider(conn, name="OpenOrganelle", base_url="https://openorganelle.janelia.org")
    asset_ids: dict[str, int] = {}
    for name in dataset_names:
        url = urls.get(name, f"s3://bucket/{name}/em")
        did = upsert_dataset(conn, provider_id=pid, stable_id=name)
        aid = upsert_asset(conn, dataset_id=did, asset_type="em_volume", remote_url=url)
        asset_ids[name] = aid
    conn.commit()
    return pid, asset_ids


def _fake_provider(conn: sqlite3.Connection, dataset_names: list[str] = None):
    if dataset_names is None:
        dataset_names = [name for name, _ in _DATASETS]
    urls = dict(_DATASETS)

    class _FakeProvider:
        name = "OpenOrganelle"
        base_url = "https://openorganelle.janelia.org"

        def ingest_discovery(self, c):
            return []

        def resolve_assets(self, c):
            return [
                AssetSpec(
                    stable_dataset_id=n,
                    asset_type="em_volume",
                    remote_url=urls.get(n, f"s3://bucket/{n}/em"),
                )
                for n in dataset_names
            ]

        def get_spacing_nm(self, sid):
            return None

    return _FakeProvider()


# ── Case A: first run ──────────────────────────────────────────────────────────

class TestCaseA_FirstRun:
    """No previous downloads — plan must include all assets."""

    def test_all_assets_planned_on_first_run(self):
        conn = _in_memory_registry()
        _seed_provider_and_assets(conn)
        provider = _fake_provider(conn)

        plans = plan_downloads(conn, provider, download_profile_hash=_PROFILE)

        assert len(plans) == len(_DATASETS), (
            f"Expected {len(_DATASETS)} assets planned, got {len(plans)}"
        )
        planned_ids = {p.stable_dataset_id for p in plans}
        expected_ids = {name for name, _ in _DATASETS}
        assert planned_ids == expected_ids

    def test_plan_has_correct_asset_types(self):
        conn = _in_memory_registry()
        _seed_provider_and_assets(conn)
        provider = _fake_provider(conn)

        plans = plan_downloads(conn, provider, download_profile_hash=_PROFILE)

        assert all(p.asset_type == "em_volume" for p in plans)
        assert all(p.reason == "missing" for p in plans)

    def test_plan_carries_profile_hash(self):
        conn = _in_memory_registry()
        _seed_provider_and_assets(conn)
        provider = _fake_provider(conn)

        plans = plan_downloads(conn, provider, download_profile_hash=_PROFILE)

        assert all(p.download_profile_hash == _PROFILE for p in plans)


# ── Case B: second run (unchanged source) ─────────────────────────────────────

class TestCaseB_SecondRunNoNewData:
    """After successful downloads, plan must be empty and no folder created."""

    def test_zero_pending_after_all_complete(self, tmp_path: Path):
        conn = _in_memory_registry()
        _, asset_ids = _seed_provider_and_assets(conn)
        provider = _fake_provider(conn)

        # Simulate successful first-run downloads (one local dir per dataset).
        for name, aid in asset_ids.items():
            local_dir = tmp_path / name
            local_dir.mkdir()
            dl_id = record_download_start(conn, aid, str(local_dir),
                                          download_profile_hash=_PROFILE)
            record_download_complete(conn, dl_id)
        conn.commit()

        plans = plan_downloads(conn, provider, download_profile_hash=_PROFILE)

        assert plans == [], (
            f"Expected 0 pending on second run, got {len(plans)}: "
            + ", ".join(p.stable_dataset_id for p in plans)
        )

    def test_no_output_folder_created_on_noop(self, tmp_path: Path):
        """Verify that no new timestamped directory is created when plan is empty.

        This checks the guard logic: if plan_downloads returns [] the caller
        must not create output directories. We confirm via the plan being empty
        (the mkdir guard in the template / API is upstream of actual mkdir).
        """
        conn = _in_memory_registry()
        _, asset_ids = _seed_provider_and_assets(conn)
        provider = _fake_provider(conn)

        # Mark all assets downloaded.
        for name, aid in asset_ids.items():
            local_dir = tmp_path / name
            local_dir.mkdir()
            dl_id = record_download_start(conn, aid, str(local_dir),
                                          download_profile_hash=_PROFILE)
            record_download_complete(conn, dl_id)
        conn.commit()

        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        dirs_before = set(raw_dir.iterdir())

        plans = plan_downloads(conn, provider, download_profile_hash=_PROFILE)
        # Guard: only create output folder when plan is non-empty.
        if plans:
            (raw_dir / "openorganelle_mito_fake_stamp").mkdir()

        dirs_after = set(raw_dir.iterdir())
        assert dirs_after == dirs_before, (
            "A new folder was created even though plan was empty"
        )

    def test_different_profile_is_still_pending(self, tmp_path: Path):
        """A completed n_crops=1 run must NOT satisfy an n_crops=4 plan."""
        conn = _in_memory_registry()
        _, asset_ids = _seed_provider_and_assets(conn)
        provider = _fake_provider(conn)

        # Record n_crops=1 complete.
        for name, aid in asset_ids.items():
            local_dir = tmp_path / name
            local_dir.mkdir()
            dl_id = record_download_start(conn, aid, str(local_dir),
                                          download_profile_hash=_PROFILE)
            record_download_complete(conn, dl_id)
        conn.commit()

        profile_4 = make_download_profile_hash(
            n_crops=4, chunk_zyx=(512, 512, 512), voxel_nm_zyx=(16.0, 16.0, 16.0),
            mode="labeled", foundation=True,
        )
        plans = plan_downloads(conn, provider, download_profile_hash=profile_4)

        assert len(plans) == len(_DATASETS), (
            f"n_crops=4 profile should show all assets as pending, got {len(plans)}"
        )


# ── Case C: new dataset appears ───────────────────────────────────────────────

class TestCaseC_NewDatasetAppearsAfterFirstRun:
    """After first run, upstream adds a new dataset. Only the new asset downloads."""

    def test_only_new_dataset_planned_after_partial_complete(self, tmp_path: Path):
        conn = _in_memory_registry()
        # Seed only the first dataset initially.
        first_name = _DATASETS[0][0]
        second_name = _DATASETS[1][0]
        _seed_provider_and_assets(conn, dataset_names=[first_name])
        provider_initial = _fake_provider(conn, dataset_names=[first_name])

        # Simulate first run: download the first dataset.
        _, asset_ids_initial = _seed_provider_and_assets(conn, dataset_names=[first_name])
        local_dir = tmp_path / first_name
        local_dir.mkdir()
        dl_id = record_download_start(
            conn, asset_ids_initial[first_name], str(local_dir),
            download_profile_hash=_PROFILE,
        )
        record_download_complete(conn, dl_id)
        conn.commit()

        # Now upstream adds second_name — seed it into registry.
        _seed_provider_and_assets(conn, dataset_names=[second_name])
        provider_updated = _fake_provider(conn, dataset_names=[first_name, second_name])

        plans = plan_downloads(conn, provider_updated, download_profile_hash=_PROFILE)

        assert len(plans) == 1, (
            f"Expected only 1 new asset (the newly added dataset), got {len(plans)}: "
            + ", ".join(p.stable_dataset_id for p in plans)
        )
        assert plans[0].stable_dataset_id == second_name

    def test_first_dataset_skipped_second_downloaded(self, tmp_path: Path):
        """Verify the skipped dataset is exactly the one already complete."""
        conn = _in_memory_registry()
        first_name, first_url = _DATASETS[0]
        second_name, second_url = _DATASETS[1]

        # Seed both datasets.
        _seed_provider_and_assets(conn, dataset_names=[first_name, second_name])

        # Record only first as complete.
        pid = upsert_provider(conn, name="OpenOrganelle", base_url="https://openorganelle.janelia.org")
        from agent.orchestration.registry.api import get_dataset_id, get_asset
        did = get_dataset_id(conn, pid, first_name)
        asset = get_asset(conn, did, "em_volume")
        local_dir = tmp_path / first_name
        local_dir.mkdir()
        dl_id = record_download_start(conn, int(asset["id"]), str(local_dir),
                                      download_profile_hash=_PROFILE)
        record_download_complete(conn, dl_id)
        conn.commit()

        provider = _fake_provider(conn, dataset_names=[first_name, second_name])
        plans = plan_downloads(conn, provider, download_profile_hash=_PROFILE)

        planned_ids = {p.stable_dataset_id for p in plans}
        assert second_name in planned_ids, "New dataset must be in the plan"
        assert first_name not in planned_ids, "Already-downloaded dataset must be skipped"


# ── Case D: URL churn ─────────────────────────────────────────────────────────

class TestCaseD_UrlChurn:
    """Re-scrape produces a new remote_url for the same dataset+type.
    The prior complete download must still satisfy the skip check.
    """

    def test_url_churn_does_not_re_download(self, tmp_path: Path):
        """Old URL asset row has complete download; new URL asset row has no downloads.
        plan_downloads must return empty (skip the dataset).
        """
        conn = _in_memory_registry()
        name = _DATASETS[0][0]
        url_v1 = _DATASETS[0][1]
        url_v2 = url_v1 + "/s0"  # slightly different URL (common after re-scrape)

        pid = upsert_provider(conn, name="OpenOrganelle", base_url="https://openorganelle.janelia.org")
        did = upsert_dataset(conn, provider_id=pid, stable_id=name)
        # Insert URL-v1 asset and record completion.
        aid_v1 = upsert_asset(conn, dataset_id=did, asset_type="em_volume", remote_url=url_v1)
        conn.commit()
        local_dir = tmp_path / name
        local_dir.mkdir()
        dl_id = record_download_start(conn, aid_v1, str(local_dir),
                                      download_profile_hash=_PROFILE)
        record_download_complete(conn, dl_id)
        conn.commit()

        # Re-scrape: insert URL-v2 asset (new row, no downloads).
        aid_v2 = upsert_asset(conn, dataset_id=did, asset_type="em_volume", remote_url=url_v2)
        conn.commit()
        assert aid_v1 != aid_v2, "Different URLs must produce different asset rows"

        # plan_downloads using provider that returns URL-v2 → must still skip.
        class _ProviderV2:
            name = "OpenOrganelle"
            base_url = "https://openorganelle.janelia.org"
            def ingest_discovery(self, c): return []
            def resolve_assets(self, c):
                return [AssetSpec(stable_dataset_id=name, asset_type="em_volume", remote_url=url_v2)]
            def get_spacing_nm(self, sid): return None

        plans = plan_downloads(conn, _ProviderV2(), download_profile_hash=_PROFILE)
        assert plans == [], (
            f"URL churn must not trigger re-download; got {len(plans)} pending: "
            + ", ".join(p.stable_dataset_id for p in plans)
        )

    def test_find_complete_download_for_dataset_type_profile_finds_old_row(self, tmp_path: Path):
        """The new dataset-level function must locate the old-URL completion."""
        conn = _in_memory_registry()
        name = _DATASETS[0][0]
        url_v1 = _DATASETS[0][1]
        url_v2 = url_v1 + "/s0"

        pid = upsert_provider(conn, name="OpenOrganelle", base_url="https://openorganelle.janelia.org")
        did = upsert_dataset(conn, provider_id=pid, stable_id=name)
        aid_v1 = upsert_asset(conn, dataset_id=did, asset_type="em_volume", remote_url=url_v1)
        conn.commit()
        local_dir = tmp_path / name
        local_dir.mkdir()
        dl_id = record_download_start(conn, aid_v1, str(local_dir),
                                      download_profile_hash=_PROFILE)
        record_download_complete(conn, dl_id)
        conn.commit()

        # Insert URL-v2 (no downloads).
        upsert_asset(conn, dataset_id=did, asset_type="em_volume", remote_url=url_v2)
        conn.commit()

        row = find_complete_download_for_dataset_type_profile(conn, did, "em_volume", _PROFILE)
        assert row is not None, "Must find the old-URL download row"
        assert is_dataset_type_downloaded_for_profile(conn, did, "em_volume", _PROFILE)

    def test_is_dataset_type_downloaded_false_when_local_path_missing(self, tmp_path: Path):
        """Complete download record but deleted local path → not considered done."""
        conn = _in_memory_registry()
        name = _DATASETS[0][0]
        url_v1 = _DATASETS[0][1]

        pid = upsert_provider(conn, name="OpenOrganelle", base_url="https://openorganelle.janelia.org")
        did = upsert_dataset(conn, provider_id=pid, stable_id=name)
        aid = upsert_asset(conn, dataset_id=did, asset_type="em_volume", remote_url=url_v1)
        conn.commit()
        deleted_path = tmp_path / "deleted_dir"  # never created
        dl_id = record_download_start(conn, aid, str(deleted_path),
                                      download_profile_hash=_PROFILE)
        record_download_complete(conn, dl_id)
        conn.commit()

        assert not is_dataset_type_downloaded_for_profile(conn, did, "em_volume", _PROFILE)

    def test_is_dataset_type_downloaded_false_when_training_h5_pair_removed(
        self, tmp_path: Path
    ):
        """Labeled runs record ``imagesTr`` on complete rows — deleted NIfTI pairs must unpark re-download."""
        conn = _in_memory_registry()
        name = "jrc_hela-2"
        url_v1 = _DATASETS[0][1]

        pid = upsert_provider(conn, name="OpenOrganelle", base_url="https://openorganelle.janelia.org")
        did = upsert_dataset(conn, provider_id=pid, stable_id=name)
        aid = upsert_asset(conn, dataset_id=did, asset_type="em_volume", remote_url=url_v1)
        conn.commit()

        nn = tmp_path / "data" / "nnUNet_raw" / "Dataset001_mito2"
        images = nn / "imagesTr"
        labels = nn / "labelsTr"
        images.mkdir(parents=True)
        labels.mkdir()
        tag = name.replace("-", "_")
        im_f = images / f"{tag}_vol1_0000.nii.gz"
        seg_f = labels / f"{tag}_vol1.nii.gz"
        im_f.write_bytes(b"nii")
        seg_f.write_bytes(b"nii")

        dl_id = record_download_start(
            conn, aid, str(images), download_profile_hash=_PROFILE
        )
        record_download_complete(conn, dl_id)
        conn.commit()

        assert is_dataset_type_downloaded_for_profile(conn, did, "em_volume", _PROFILE)

        im_f.unlink()
        assert not is_dataset_type_downloaded_for_profile(conn, did, "em_volume", _PROFILE)


# ── Case E: recording round-trip ──────────────────────────────────────────────

class TestCaseE_RecordingRoundTrip:
    """Simulates what the generated script now does: record start → download →
    record complete. Verifies that plan_downloads then returns empty on re-run.
    """

    def test_plan_empty_after_record_complete(self, tmp_path: Path):
        conn = _in_memory_registry()
        name, url = _DATASETS[0]

        pid = upsert_provider(conn, name="OpenOrganelle", base_url="https://openorganelle.janelia.org")
        did = upsert_dataset(conn, provider_id=pid, stable_id=name)
        aid = upsert_asset(conn, dataset_id=did, asset_type="em_volume", remote_url=url)
        conn.commit()

        # Simulate generated script: record start, "download", record complete.
        local_dir = tmp_path / name
        local_dir.mkdir()
        dl_id = record_download_start(conn, aid, str(local_dir),
                                      download_profile_hash=_PROFILE)
        conn.commit()
        record_download_complete(conn, dl_id)
        conn.commit()

        class _Provider:
            name = "OpenOrganelle"
            base_url = "https://openorganelle.janelia.org"
            def ingest_discovery(self, c): return []
            def resolve_assets(self, c):
                return [AssetSpec(stable_dataset_id=name, asset_type="em_volume", remote_url=url)]
            def get_spacing_nm(self, sid): return None

        plans = plan_downloads(conn, _Provider(), download_profile_hash=_PROFILE)
        assert plans == [], f"After recording completion, plan must be empty; got {len(plans)}"

    def test_plan_pending_when_record_start_but_not_complete(self, tmp_path: Path):
        """A 'downloading' record (no completion) must not satisfy skip."""
        conn = _in_memory_registry()
        name, url = _DATASETS[0]

        pid = upsert_provider(conn, name="OpenOrganelle", base_url="https://openorganelle.janelia.org")
        did = upsert_dataset(conn, provider_id=pid, stable_id=name)
        aid = upsert_asset(conn, dataset_id=did, asset_type="em_volume", remote_url=url)
        conn.commit()

        local_dir = tmp_path / name
        local_dir.mkdir()
        # Only record_download_start, no complete.
        record_download_start(conn, aid, str(local_dir), download_profile_hash=_PROFILE)
        conn.commit()

        class _Provider:
            name = "OpenOrganelle"
            base_url = "https://openorganelle.janelia.org"
            def ingest_discovery(self, c): return []
            def resolve_assets(self, c):
                return [AssetSpec(stable_dataset_id=name, asset_type="em_volume", remote_url=url)]
            def get_spacing_nm(self, sid): return None

        plans = plan_downloads(conn, _Provider(), download_profile_hash=_PROFILE)
        assert len(plans) == 1, "Incomplete download must still show as pending"
