"""Shared downloader-script generation for Stage-3 providers (OpenOrganelle, BossDB, …)."""

from __future__ import annotations

import importlib
import sys
from typing import Any

from .common import ensure_import_path, parse_zyx_triplet, repo_root


def generate_labeled_stage3_script(
    *,
    site: str,
    mode: str,
    chunk_zyx: str,
    n_crops: int,
    voxel_size_nm_zyx: str,
    datasets: list[Any] | None = None,
    dataset_splits: Any = None,
) -> Path | None:
    """Generate a labeled Stage-3 downloader script for *site* (``openorganelle`` | ``bossdb``)."""
    key = (site or "").strip().lower()
    if key == "openorganelle":
        dl_root = repo_root() / "3data_downloader"
        ensure_import_path(dl_root)
        if "openorganelle.agent" in sys.modules:
            agent_mod = importlib.reload(sys.modules["openorganelle.agent"])
        else:
            import openorganelle.agent as agent_mod  # type: ignore[import]
        return agent_mod.generate_openorganelle_downloader_script(
            mode=mode,
            chunk_zyx=chunk_zyx,
            n_crops=n_crops,
            voxel_size_nm_zyx=voxel_size_nm_zyx,
            dataset_splits=dataset_splits,
        )
    if key == "bossdb":
        if mode != "labeled":
            return None
        if not datasets:
            return None
        dl_root = repo_root() / "3data_downloader"
        ensure_import_path(dl_root)
        inv_root = repo_root() / "0inventory"
        ensure_import_path(inv_root)
        from bossdb_inventory import pair_labeled_datasets  # noqa: PLC0415
        from bossdb.scriptgen import write_bossdb_outputs  # noqa: PLC0415
        from download_history import nnunet_dataset_root  # noqa: PLC0415

        pairs = pair_labeled_datasets(datasets)
        if not pairs:
            return None
        chunk_shape = parse_zyx_triplet(chunk_zyx, cast=int)
        voxel_nm = parse_zyx_triplet(voxel_size_nm_zyx, cast=float)
        py_path, _ = write_bossdb_outputs(
            dl_root / "outputs",
            pairs,
            chunk_shape=chunk_shape,
            n_crops=n_crops,
            voxel_size_nm=voxel_nm,
            training_root=str(nnunet_dataset_root(repo_root())),
            dataset_splits=dataset_splits if isinstance(dataset_splits, dict) else None,
        )
        return py_path
    raise ValueError(f"Unknown Stage-3 downloader site: {site!r}")


def generate_openorganelle_script(
    *,
    mode: str,
    chunk_zyx: str,
    n_crops: int,
    voxel_size_nm_zyx: str,
    dataset_splits: Any = None,
) -> Path | None:
    """Generate OpenOrganelle downloader script via stage-3 module."""
    return generate_labeled_stage3_script(
        site="openorganelle",
        mode=mode,
        chunk_zyx=chunk_zyx,
        n_crops=n_crops,
        voxel_size_nm_zyx=voxel_size_nm_zyx,
        dataset_splits=dataset_splits,
    )


def generate_bossdb_script(
    *,
    datasets: list[Any],
    mode: str,
    chunk_zyx: str,
    n_crops: int,
    voxel_size_nm_zyx: str,
    dataset_splits: Any = None,
) -> Path | None:
    """Generate BossDB downloader script via stage-3 scriptgen."""
    return generate_labeled_stage3_script(
        site="bossdb",
        mode=mode,
        chunk_zyx=chunk_zyx,
        n_crops=n_crops,
        voxel_size_nm_zyx=voxel_size_nm_zyx,
        datasets=datasets,
        dataset_splits=dataset_splits,
    )
