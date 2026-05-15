"""Tests for download batch provenance: create batch, upsert items, reconcile statuses,
selective re-download planning after one item is deleted.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from agent.orchestration.registry.schema import migrate
from agent.orchestration.registry.api import (
    upsert_provider,
    upsert_dataset,
    upsert_asset,
    record_download_start,
    record_download_complete,
    create_download_batch,
    update_batch_status,
    upsert_batch_item,
    update_batch_item_status,
    list_batch_items,
    reconcile_batch_items,
    find_missing_batch_items,
    delete_batch_item_by_path,
    get_download_batch,
    list_download_batches,
)
from agent.orchestration.registry.planners import plan_batch_redownload


def _mem() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    migrate(conn)
    return conn


_PROVIDER = "OpenOrganelle"
_DATASETS = ["jrc_hela-2", "jrc_jurkat-1", "jrc_macrophage-2", "jrc_cos7-1a", "jrc_sum159-1"]


def _seed(conn, tmp_path: Path):
    """Create provider, 5 datasets, a batch with 5 present items. Return (batch_db_id, item_ids, paths)."""
    pid = upsert_provider(conn, name=_PROVIDER, base_url="https://openorganelle.janelia.org")
    batch_db_id = create_download_batch(
        conn,
        batch_id="openorganelle_mito_20260420_012621",
        provider=_PROVIDER,
        profile_hash="abc123",
        profile_json={"n_crops": 1, "chunk_zyx": [128, 128, 128]},
        run_folder="openorganelle_mito_20260420_012621",
    )
    conn.commit()

    item_ids: dict[str, int] = {}
    paths: dict[str, Path] = {}
    for name in _DATASETS:
        did = upsert_dataset(conn, provider_id=pid, stable_id=name)
        local = tmp_path / f"{name}_im.h5"
        local.write_bytes(b"dummy")
        iid = upsert_batch_item(
            conn,
            batch_db_id=batch_db_id,
            dataset_id=did,
            stable_id=name,
            asset_type="preprocessed_em",
            local_path=str(local),
            status="present",
        )
        conn.commit()
        item_ids[name] = iid
        paths[name] = local
    update_batch_status(conn, batch_db_id, "complete")
    conn.commit()
    return batch_db_id, item_ids, paths


class TestBatchCreation:
    def test_batch_created_and_retrievable(self):
        conn = _mem()
        bid = create_download_batch(
            conn, batch_id="test_batch_001", provider=_PROVIDER,
            profile_hash="aaaa", run_folder="test_batch_001",
        )
        conn.commit()
        row = get_download_batch(conn, "test_batch_001")
        assert row is not None
        assert int(row["id"]) == bid
        assert row["provider"] == _PROVIDER
        assert row["status"] == "in_progress"

    def test_duplicate_batch_id_is_idempotent(self):
        conn = _mem()
        bid1 = create_download_batch(conn, batch_id="dup", provider=_PROVIDER)
        conn.commit()
        bid2 = create_download_batch(conn, batch_id="dup", provider=_PROVIDER)
        conn.commit()
        assert bid1 == bid2

    def test_list_batches_by_provider(self):
        conn = _mem()
        create_download_batch(conn, batch_id="og_1", provider="OpenOrganelle")
        create_download_batch(conn, batch_id="bb_1", provider="BossDB")
        conn.commit()
        og = list_download_batches(conn, "OpenOrganelle")
        bb = list_download_batches(conn, "BossDB")
        assert len(og) == 1 and og[0]["batch_id"] == "og_1"
        assert len(bb) == 1 and bb[0]["batch_id"] == "bb_1"

    def test_list_all_batches(self):
        conn = _mem()
        create_download_batch(conn, batch_id="og_1", provider="OpenOrganelle")
        create_download_batch(conn, batch_id="bb_1", provider="BossDB")
        conn.commit()
        all_batches = list_download_batches(conn)
        assert len(all_batches) == 2


class TestBatchItems:
    def test_five_items_present_after_seed(self, tmp_path: Path):
        conn = _mem()
        batch_db_id, item_ids, _ = _seed(conn, tmp_path)
        items = list_batch_items(conn, batch_db_id)
        assert len(items) == 5
        assert all(i["status"] == "present" for i in items)

    def test_item_status_update(self, tmp_path: Path):
        conn = _mem()
        batch_db_id, item_ids, paths = _seed(conn, tmp_path)
        first_name = _DATASETS[0]
        update_batch_item_status(conn, item_ids[first_name], "missing_or_deleted_local")
        conn.commit()
        items = list_batch_items(conn, batch_db_id)
        statuses = {i["stable_id"]: i["status"] for i in items}
        assert statuses[first_name] == "missing_or_deleted_local"
        assert all(statuses[n] == "present" for n in _DATASETS[1:])


class TestReconciliation:
    def test_reconcile_marks_deleted_file_as_missing(self, tmp_path: Path):
        conn = _mem()
        batch_db_id, item_ids, paths = _seed(conn, tmp_path)

        # Simulate user deleting one file.
        deleted_name = _DATASETS[2]
        paths[deleted_name].unlink()

        changed = reconcile_batch_items(conn, batch_db_id)
        conn.commit()

        assert changed == 1
        items = list_batch_items(conn, batch_db_id)
        statuses = {i["stable_id"]: i["status"] for i in items}
        assert statuses[deleted_name] == "missing_or_deleted_local"
        assert all(statuses[n] == "present" for n in _DATASETS if n != deleted_name)

    def test_reconcile_marks_restored_file_as_present(self, tmp_path: Path):
        conn = _mem()
        batch_db_id, item_ids, paths = _seed(conn, tmp_path)

        # Mark one item as missing.
        name = _DATASETS[0]
        update_batch_item_status(conn, item_ids[name], "missing_or_deleted_local")
        conn.commit()

        # File still exists → reconcile should restore it.
        changed = reconcile_batch_items(conn, batch_db_id)
        conn.commit()
        assert changed == 1
        items = list_batch_items(conn, batch_db_id)
        statuses = {i["stable_id"]: i["status"] for i in items}
        assert statuses[name] == "present"

    def test_find_missing_after_delete(self, tmp_path: Path):
        conn = _mem()
        batch_db_id, item_ids, paths = _seed(conn, tmp_path)
        target = _DATASETS[1]
        paths[target].unlink()

        missing = find_missing_batch_items(conn, batch_db_id)
        assert len(missing) == 1
        assert missing[0]["stable_id"] == target


class TestSelectiveRedownload:
    """Core scenario: delete 1 of 5 datasets → only that 1 should re-download."""

    def test_plan_redownload_returns_only_missing_item(self, tmp_path: Path):
        conn = _mem()
        batch_db_id, _, paths = _seed(conn, tmp_path)

        # Delete exactly one file.
        deleted = _DATASETS[3]
        paths[deleted].unlink()

        plans = plan_batch_redownload(conn, batch_db_id)

        assert len(plans) == 1, (
            f"Expected 1 item for re-download; got {len(plans)}: "
            + ", ".join(p.stable_id for p in plans)
        )
        assert plans[0].stable_id == deleted
        assert plans[0].reason == "missing_local_file"

    def test_plan_empty_when_all_files_present(self, tmp_path: Path):
        conn = _mem()
        batch_db_id, _, _ = _seed(conn, tmp_path)
        plans = plan_batch_redownload(conn, batch_db_id)
        assert plans == [], f"Expected 0 plans when all files present, got {len(plans)}"

    def test_plan_all_missing_when_all_deleted(self, tmp_path: Path):
        conn = _mem()
        batch_db_id, _, paths = _seed(conn, tmp_path)
        for p in paths.values():
            p.unlink()
        plans = plan_batch_redownload(conn, batch_db_id)
        assert len(plans) == len(_DATASETS)

    def test_redownload_status_resets_to_present_after_restore(self, tmp_path: Path):
        conn = _mem()
        batch_db_id, item_ids, paths = _seed(conn, tmp_path)

        deleted = _DATASETS[0]
        paths[deleted].unlink()

        # Simulate re-download: file is restored.
        paths[deleted].write_bytes(b"re-downloaded")

        plans = plan_batch_redownload(conn, batch_db_id)
        # After reconcile inside plan_batch_redownload, status should be back to present.
        assert plans == [], f"File restored → should have 0 plans, got {len(plans)}"


class TestDeleteBatchItemByPath:
    def test_marks_matching_items_missing(self, tmp_path: Path):
        conn = _mem()
        batch_db_id, _, paths = _seed(conn, tmp_path)

        target = _DATASETS[0]
        target_path = str(paths[target])
        changed = delete_batch_item_by_path(conn, target_path)
        conn.commit()

        assert changed == 1
        items = list_batch_items(conn, batch_db_id)
        statuses = {i["stable_id"]: i["status"] for i in items}
        assert statuses[target] == "missing_or_deleted_local"
        assert all(statuses[n] == "present" for n in _DATASETS[1:])

    def test_no_match_returns_zero(self, tmp_path: Path):
        conn = _mem()
        batch_db_id, _, _ = _seed(conn, tmp_path)
        changed = delete_batch_item_by_path(conn, "/nonexistent/path.h5")
        assert changed == 0
