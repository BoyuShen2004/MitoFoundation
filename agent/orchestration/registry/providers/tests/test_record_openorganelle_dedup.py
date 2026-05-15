"""Re-download must not add duplicate batch_items for the same dataset×asset."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_INV_ROOT = _REPO_ROOT / "0inventory"
if str(_INV_ROOT) not in sys.path:
    sys.path.insert(0, str(_INV_ROOT))

from agent.orchestration.registry.api import make_download_profile_hash
from agent.orchestration.registry.schema import open_registry


def test_second_download_batch_updates_rows(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "2database_builder" / "outputs").mkdir(parents=True)
    nn = repo / "data" / "nnUNet_raw" / "Dataset001_mito2"
    img_d = nn / "imagesTr"
    lbl_d = nn / "labelsTr"
    img_d.mkdir(parents=True)
    lbl_d.mkdir(parents=True)
    p_im = img_d / "jrc_hela-2_0000.nii.gz"
    p_seg = lbl_d / "jrc_hela-2.nii.gz"
    p_im.write_bytes(b"x")
    p_seg.write_bytes(b"y")

    sys.path.insert(0, str(repo))
    from download_history import record_openorganelle_download_batch  # noqa: PLC0415

    ph = make_download_profile_hash(
        n_crops=1,
        chunk_zyx=(128, 128, 128),
        voxel_nm_zyx=(16.0, 16.0, 16.0),
        mode="labeled",
        foundation=True,
    )
    profile = {
        "chunk_shape_zyx": [128, 128, 128],
        "n_crops": 1,
        "voxel_nm_zyx": [16.0, 16.0, 16.0],
        "foundation": True,
    }
    items = [
        {
            "stable_id": "jrc_hela-2",
            "primary_image_path": str(p_im.resolve()),
            "primary_label_path": str(p_seg.resolve()),
        }
    ]
    record_openorganelle_download_batch(
        repo,
        batch_id="openorganelle_mito_20260101_000001",
        profile_hash=ph,
        profile={**profile, "batch_id": "openorganelle_mito_20260101_000001"},
        items=items,
    )
    record_openorganelle_download_batch(
        repo,
        batch_id="openorganelle_mito_20260102_000002",
        profile_hash=ph,
        profile={**profile, "batch_id": "openorganelle_mito_20260102_000002"},
        items=items,
    )

    reg_path = repo / "data" / "registry.sqlite"
    conn = open_registry(reg_path)
    try:
        n_em = int(
            conn.execute(
                "SELECT COUNT(*) FROM batch_items WHERE stable_id = ? AND asset_type = ?",
                ("jrc_hela-2", "em_volume"),
            ).fetchone()[0]
        )
        n_seg = int(
            conn.execute(
                "SELECT COUNT(*) FROM batch_items WHERE stable_id = ? AND asset_type = ?",
                ("jrc_hela-2", "mito_seg"),
            ).fetchone()[0]
        )
        assert n_em == 1
        assert n_seg == 1
        n_batches = int(conn.execute("SELECT COUNT(*) FROM download_batches").fetchone()[0])
        assert n_batches == 2
        bid = conn.execute(
            """
            SELECT db.batch_id
            FROM batch_items bi
            JOIN download_batches db ON db.id = bi.batch_db_id
            WHERE bi.stable_id = ? AND bi.asset_type = 'em_volume'
            """,
            ("jrc_hela-2",),
        ).fetchone()["batch_id"]
        assert bid == "openorganelle_mito_20260102_000002"
    finally:
        conn.close()


def test_early_register_then_finalize_uses_pairs_times_two(tmp_path: Path) -> None:
    """Pre-run Log = 2×planned pairs; after record with one pair resolved → 2 EM+seg units."""
    repo = tmp_path / "repo"
    (repo / "2database_builder" / "outputs").mkdir(parents=True)
    nn = repo / "data" / "nnUNet_raw" / "Dataset001_mito2"
    img_d = nn / "imagesTr"
    lbl_d = nn / "labelsTr"
    img_d.mkdir(parents=True)
    lbl_d.mkdir(parents=True)
    p_im = img_d / "jrc_hela-2_0000.nii.gz"
    p_seg = lbl_d / "jrc_hela-2.nii.gz"
    p_im.write_bytes(b"x")
    p_seg.write_bytes(b"y")

    sys.path.insert(0, str(repo))
    from download_history import (  # noqa: PLC0415
        record_openorganelle_download_batch,
        register_openorganelle_labeled_run_planned,
    )

    ph = make_download_profile_hash(
        n_crops=1,
        chunk_zyx=(128, 128, 128),
        voxel_nm_zyx=(16.0, 16.0, 16.0),
        mode="labeled",
        foundation=True,
    )
    bid = "openorganelle_mito_planned_001"
    register_openorganelle_labeled_run_planned(
        repo,
        batch_id=bid,
        planned_pair_count=12,
        chunk_shape=(128, 128, 128),
        n_crops=1,
        voxel_nm=[16.0, 16.0, 16.0],
        foundation=True,
        datasets_planned=12,
        n_windows=1,
    )
    reg_path = repo / "data" / "registry.sqlite"
    conn = open_registry(reg_path)
    try:
        n0 = int(
            conn.execute(
                "SELECT download_asset_completions FROM download_batches WHERE batch_id = ?",
                (bid,),
            ).fetchone()[0]
        )
        assert n0 == 24
    finally:
        conn.close()

    profile = {
        "batch_id": bid,
        "chunk_shape_zyx": [128, 128, 128],
        "n_crops": 1,
        "voxel_nm_zyx": [16.0, 16.0, 16.0],
        "foundation": True,
    }
    items = [
        {
            "stable_id": "jrc_hela-2",
            "primary_image_path": str(p_im.resolve()),
            "primary_label_path": str(p_seg.resolve()),
        }
    ]
    record_openorganelle_download_batch(
        repo,
        batch_id=bid,
        profile_hash=ph,
        profile=profile,
        items=items,
    )
    conn = open_registry(reg_path)
    try:
        n1 = int(
            conn.execute(
                "SELECT download_asset_completions FROM download_batches WHERE batch_id = ?",
                (bid,),
            ).fetchone()[0]
        )
        assert n1 == 2
    finally:
        conn.close()


def test_finalize_with_empty_items_zeros_log_count(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "2database_builder" / "outputs").mkdir(parents=True)
    sys.path.insert(0, str(repo))
    from download_history import (  # noqa: PLC0415
        record_openorganelle_download_batch,
        register_openorganelle_labeled_run_planned,
    )

    ph = make_download_profile_hash(
        n_crops=1,
        chunk_zyx=(128, 128, 128),
        voxel_nm_zyx=(16.0, 16.0, 16.0),
        mode="labeled",
        foundation=True,
    )
    bid = "openorganelle_mito_empty_001"
    register_openorganelle_labeled_run_planned(
        repo,
        batch_id=bid,
        planned_pair_count=12,
        chunk_shape=(128, 128, 128),
        n_crops=1,
        voxel_nm=[16.0, 16.0, 16.0],
        foundation=True,
        datasets_planned=12,
        n_windows=1,
    )
    reg_path = repo / "data" / "registry.sqlite"
    profile = {
        "batch_id": bid,
        "chunk_shape_zyx": [128, 128, 128],
        "n_crops": 1,
        "voxel_nm_zyx": [16.0, 16.0, 16.0],
        "foundation": True,
    }
    record_openorganelle_download_batch(
        repo,
        batch_id=bid,
        profile_hash=ph,
        profile=profile,
        items=[],
        successful_pair_count=0,
    )
    conn = open_registry(reg_path)
    try:
        n = int(
            conn.execute(
                "SELECT download_asset_completions FROM download_batches WHERE batch_id = ?",
                (bid,),
            ).fetchone()[0]
        )
        assert n == 0
    finally:
        conn.close()


def test_empty_items_uses_successful_pair_count_for_log(tmp_path: Path) -> None:
    """When path assembly yields no items, still credit Log from successful pair count."""
    repo = tmp_path / "repo"
    (repo / "2database_builder" / "outputs").mkdir(parents=True)
    sys.path.insert(0, str(repo))
    from download_history import record_openorganelle_download_batch  # noqa: PLC0415

    ph = make_download_profile_hash(
        n_crops=1,
        chunk_zyx=(128, 128, 128),
        voxel_nm_zyx=(16.0, 16.0, 16.0),
        mode="labeled",
        foundation=True,
    )
    bid = "openorganelle_mito_fallback_001"
    profile = {
        "batch_id": bid,
        "chunk_shape_zyx": [128, 128, 128],
        "n_crops": 1,
        "voxel_nm_zyx": [16.0, 16.0, 16.0],
        "foundation": True,
    }
    record_openorganelle_download_batch(
        repo,
        batch_id=bid,
        profile_hash=ph,
        profile=profile,
        items=[],
        successful_pair_count=1,
    )
    reg_path = repo / "data" / "registry.sqlite"
    conn = open_registry(reg_path)
    try:
        n = int(
            conn.execute(
                "SELECT download_asset_completions FROM download_batches WHERE batch_id = ?",
                (bid,),
            ).fetchone()[0]
        )
        assert n == 2
    finally:
        conn.close()
