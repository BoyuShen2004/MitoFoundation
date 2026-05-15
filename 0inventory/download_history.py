"""OpenOrganelle direct-to-training: Stage 4 + registry download batch rows.

Inventory / Stage 0 catalogue reads ``registry.sqlite`` (``download_batches``,
``batch_items``) — no separate JSON manifest files.
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from time_utils import now_us_eastern_iso
from typing import Any


def nnunet_dataset_root(repo: Path) -> Path:
    from config.paths import nnunet_dataset_root as _nnunet_dataset_root

    return _nnunet_dataset_root(repo)


def _strip_ext(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    return path.stem


def _canonical_key(path: Path) -> str:
    """Match stage-4 ``_canonical_key`` (pairing / output basenames)."""
    name = _strip_ext(path)
    name = re.sub(r"_0000$", "", name, flags=re.I)
    name = re.sub(r"_(im|seg)$", "", name, flags=re.I)
    name = re.sub(r"\.(im|seg)$", "", name, flags=re.I)
    return name


def _resolve_training_output_pair(repo_root: Path, key: str) -> tuple[Path, Path]:
    """Return expected Stage-4 nnUNet paths for a crop *key* under Dataset001_mito2."""
    nn_root = nnunet_dataset_root(repo_root)
    nn_img = nn_root / "imagesTr" / f"{key}_0000.nii.gz"
    nn_seg = nn_root / "labelsTr" / f"{key}.nii.gz"
    return nn_img, nn_seg


def _resolve_inference_output_pair(repo: Path, key: str) -> tuple[Path, Path]:
    """Return expected Stage-3 inference output paths for a key."""
    nn_root = nnunet_dataset_root(repo)
    nn_img = nn_root / "imagesTs" / f"{key}_0000.nii.gz"
    nn_seg = nn_root / "labelsTs" / f"{key}.nii.gz"
    return nn_img, nn_seg


def _open_registry_conn(repo: Path) -> sqlite3.Connection:
    _ensure_repo_agent_import(repo)
    from agent.orchestration.registry.schema import open_registry  # noqa: PLC0415

    rp = (repo / "data" / "registry.sqlite").resolve()
    return open_registry(rp)


def _ensure_repo_agent_import(repo: Path) -> None:
    """Ensure ``agent.*`` resolves to repo package, not a local ``agent.py`` script."""
    rp = str(Path(repo).resolve())
    mod = sys.modules.get("agent")
    if mod is not None and not hasattr(mod, "__path__"):
        del sys.modules["agent"]
    try:
        while rp in sys.path:
            sys.path.remove(rp)
    except ValueError:
        pass
    sys.path.insert(0, rp)


def record_openorganelle_download_batch(
    repo: Path,
    *,
    batch_id: str,
    profile_hash: str,
    profile: dict[str, Any],
    items: list[dict[str, Any]],
    successful_pair_count: int | None = None,
) -> None:
    """Insert ``download_batches`` + ``batch_items`` for catalogue / Stage 0.

    Stage-0 **Log** counts **EM + mito_seg units** = ``2 ×`` **new image/label pair
    slots** for the batch. Pair slots come from ``len(items)`` when paths were
    resolved; if ``items`` is empty but the run still finished datasets, pass
    ``successful_pair_count`` (datasets × windows for that run) so Log is not
    stuck at zero when glob/path assembly misses files.
    """
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from agent.orchestration.registry.api import (  # noqa: PLC0415
        create_download_batch,
        get_dataset_id,
        upsert_batch_item,
        upsert_dataset,
        update_batch_status,
        upsert_provider,
    )

    conn = _open_registry_conn(repo)
    try:
        # If nothing completed: remove orphan rows, but if a planned batch already
        # exists (e.g. from ``register_openorganelle_labeled_run_planned``), zero
        # its Log and mark failed instead of deleting the card.
        if not items and (successful_pair_count is None or int(successful_pair_count) <= 0):
            row = conn.execute(
                "SELECT id, profile_json FROM download_batches WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            if row is None:
                conn.execute("DELETE FROM download_batches WHERE batch_id = ?", (batch_id,))
                conn.commit()
                return
            batch_db_id = int(row["id"])
            try:
                old_prof = json.loads((row["profile_json"] or "") or "{}")
            except Exception:
                old_prof = {}
            merged_profile = {**old_prof, **dict(profile)}
            merged_profile["download_asset_completions"] = 0
            now_iso = now_us_eastern_iso(timespec="seconds")
            conn.execute(
                """
                UPDATE download_batches
                SET download_asset_completions = 0,
                    profile_json = ?,
                    status = 'failed',
                    finished_at = ?
                WHERE id = ?
                """,
                (json.dumps(merged_profile, ensure_ascii=False), now_iso, batch_db_id),
            )
            conn.commit()
            return
        pid = upsert_provider(conn, name="OpenOrganelle", base_url="https://openorganelle.janelia.org/")
        conn.commit()
        batch_db_id = create_download_batch(
            conn,
            batch_id=batch_id,
            provider="OpenOrganelle",
            profile_hash=profile_hash,
            profile_json=dict(profile),
            run_folder="data/nnUNet_raw/Dataset001_mito2",
        )
        conn.commit()
        def _upsert_or_revive_item(
            *,
            stable_id: str,
            asset_type: str,
            local_path: str,
            dataset_id: int | None,
        ) -> None:
            now_iso = now_us_eastern_iso(timespec="seconds")
            # Re-download: revive a missing row first.
            row = conn.execute(
                """
                SELECT id
                FROM batch_items
                WHERE stable_id = ?
                  AND asset_type = ?
                  AND status = 'missing_or_deleted_local'
                ORDER BY id DESC
                LIMIT 1
                """,
                (stable_id, asset_type),
            ).fetchone()
            if row is not None:
                conn.execute(
                    """
                    UPDATE batch_items
                    SET batch_db_id = ?,
                        dataset_id = COALESCE(?, dataset_id),
                        local_path = ?,
                        status = 'present',
                        completed_at = ?
                    WHERE id = ?
                    """,
                    (batch_db_id, dataset_id, local_path, now_iso, int(row["id"])),
                )
                return
            # Same dataset+asset already tracked on another batch (e.g. re-download with a new
            # batch_id): update that row so inventory row count does not grow.
            row = conn.execute(
                """
                SELECT id FROM batch_items
                WHERE stable_id = ? AND asset_type = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (stable_id, asset_type),
            ).fetchone()
            if row is not None:
                keep_id = int(row["id"])
                conn.execute(
                    """
                    UPDATE batch_items
                    SET batch_db_id = ?,
                        dataset_id = COALESCE(?, dataset_id),
                        local_path = ?,
                        status = 'present',
                        completed_at = ?
                    WHERE id = ?
                    """,
                    (batch_db_id, dataset_id, local_path, now_iso, keep_id),
                )
                conn.execute(
                    """
                    DELETE FROM batch_items
                    WHERE stable_id = ? AND asset_type = ? AND id != ?
                    """,
                    (stable_id, asset_type, keep_id),
                )
                return
            upsert_batch_item(
                conn,
                batch_db_id=batch_db_id,
                dataset_id=dataset_id,
                stable_id=stable_id,
                asset_type=asset_type,
                local_path=local_path,
                status="present",
            )

        for it in items:
            sid = str(it.get("stable_id") or "").strip()
            if not sid:
                continue
            did = get_dataset_id(conn, pid, sid)
            if did is None:
                did = upsert_dataset(conn, provider_id=pid, stable_id=sid)
                conn.commit()
            img_p = (it.get("primary_image_path") or "").strip()
            seg_p = (it.get("primary_label_path") or "").strip()
            if img_p:
                _upsert_or_revive_item(
                    stable_id=sid,
                    asset_type="em_volume",
                    local_path=img_p,
                    dataset_id=did,
                )
            if seg_p:
                _upsert_or_revive_item(
                    stable_id=sid,
                    asset_type="mito_seg",
                    local_path=seg_p,
                    dataset_id=did,
                )
        if items:
            pairs_logged = len(items)
        elif successful_pair_count is not None and int(successful_pair_count) > 0:
            pairs_logged = int(successful_pair_count)
        else:
            pairs_logged = 0
        n_logged = 2 * max(0, pairs_logged)
        merged_profile = dict(profile)
        merged_profile["download_asset_completions"] = int(n_logged)
        merged_profile["planned_asset_downloads"] = int(n_logged)
        conn.execute(
            """
            UPDATE download_batches
            SET download_asset_completions = ?, profile_json = ?
            WHERE id = ?
            """,
            (
                int(n_logged),
                json.dumps(merged_profile, ensure_ascii=False),
                int(batch_db_id),
            ),
        )
        conn.commit()
        st = "complete" if items else "failed"
        update_batch_status(conn, batch_db_id, st)
        conn.commit()
    finally:
        conn.close()


def _make_profile_hash(
    repo: Path,
    *,
    n_crops: int,
    chunk_shape: tuple[int, int, int],
    voxel_nm: list[float],
    foundation: bool,
) -> str:
    _ensure_repo_agent_import(repo)
    from agent.orchestration.registry.api import make_download_profile_hash  # noqa: PLC0415

    return make_download_profile_hash(
        n_crops=n_crops,
        chunk_zyx=chunk_shape,
        voxel_nm_zyx=tuple(float(v) for v in voxel_nm[:3]),
        mode="labeled",
        foundation=foundation,
    )


def register_openorganelle_labeled_run_planned(
    repo: Path,
    *,
    batch_id: str,
    planned_pair_count: int,
    chunk_shape: tuple[int, int, int],
    n_crops: int,
    voxel_nm: list[float],
    foundation: bool,
    datasets_planned: int,
    n_windows: int,
) -> None:
    """Create/update ``download_batches`` so Stage-0 Log shows ``2 ×`` planned **pairs** immediately.

    ``planned_pair_count`` should match the downloader’s “Planned image/label pairs”
    (typically ``len(pending datasets) × windows_per_dataset``). Stored Log units are
    ``2 ×`` that (EM + seg per pair).
    """
    _ensure_repo_agent_import(repo)
    from agent.orchestration.registry.api import create_download_batch, upsert_provider  # noqa: PLC0415

    pp = max(0, int(planned_pair_count))
    n = 2 * pp
    ph = _make_profile_hash(
        repo, n_crops=n_crops, chunk_shape=chunk_shape, voxel_nm=voxel_nm, foundation=foundation
    )
    profile: dict[str, Any] = {
        "batch_id": batch_id,
        "chunk_shape_zyx": list(chunk_shape),
        "n_crops": int(n_crops),
        "voxel_nm_zyx": [float(x) for x in voxel_nm[:3]],
        "foundation": bool(foundation),
        "planned_asset_downloads": n,
        "download_asset_completions": n,
        "planned_pairs": pp,
        "datasets_planned": int(datasets_planned),
        "n_windows": int(n_windows),
    }
    conn = _open_registry_conn(repo)
    try:
        upsert_provider(conn, name="OpenOrganelle", base_url="https://openorganelle.janelia.org/")
        conn.commit()
        create_download_batch(
            conn,
            batch_id=batch_id,
            provider="OpenOrganelle",
            profile_hash=ph,
            profile_json=dict(profile),
            run_folder="data/nnUNet_raw/Dataset001_mito2",
        )
        conn.execute(
            """
            UPDATE download_batches
            SET profile_hash = ?,
                download_asset_completions = ?,
                profile_json = ?,
                status = 'in_progress',
                finished_at = NULL
            WHERE batch_id = ?
            """,
            (ph, n, json.dumps(profile, ensure_ascii=False), batch_id),
        )
        conn.commit()
    finally:
        conn.close()


def run_stage4_for_images(repo: Path, image_paths: list[Path], *, split_label_cc: bool = True) -> int:
    """Run stage-4 preprocessor on explicit training-side image paths. Returns exit code."""
    if not image_paths:
        return 0
    stage4 = repo / "3data_downloader" / "downloader_master/preprocess_agent.py"
    if not stage4.is_file():
        print(f"[WARN] Stage 4 agent missing: {stage4}", flush=True)
        return 1
    # Stage-4 pairs images+labels by key. When callers pass only image paths,
    # include matching label paths so split_label_cc is actually applied.
    selected: list[Path] = []
    for im in image_paths:
        p = Path(im).resolve()
        selected.append(p)
        name_l = p.name.lower()
        if name_l.endswith("_im.h5") or name_l.endswith("_0000.nii.gz"):
            # Common layout: .../images/<stem>_im.h5  <->  .../labels/<stem>_seg.h5
            try:
                lbl_dir = p.parent.parent / "labels" if p.parent.name.lower() == "images" else p.parent
            except Exception:
                lbl_dir = p.parent
            if name_l.endswith("_0000.nii.gz"):
                lbl_dir = p.parent.parent / "labelsTr" if p.parent.name.lower() == "imagestr" else p.parent
                seg_name = p.name[:-12] + ".nii.gz"
            else:
                seg_name = p.name[:-6] + "_seg.h5"
            seg = (lbl_dir / seg_name).resolve()
            if seg.is_file():
                selected.append(seg)
    selected = sorted({str(p): p for p in selected}.values(), key=lambda x: str(x).lower())
    payload = json.dumps([str(p.resolve()) for p in selected], ensure_ascii=True)
    argv = [
        sys.executable,
        str(stage4),
        "--task",
        "supervised",
        "--dataset-paths-json",
        payload,
    ]
    if split_label_cc:
        argv.append("--split-label-cc")
    proc = subprocess.run(argv, cwd=str(repo), capture_output=True, text=True)
    if proc.stdout:
        txt = proc.stdout
        # Keep integrated stage-4 logs visible but avoid polluting stage-3 downloader progress parsing.
        txt = re.sub(r"(?m)^\[PROGRESS\]\s+dataset\s+", "[PREPROCESS_PROGRESS] dataset ", txt)
        txt = re.sub(r"(?m)^\[DONE\]\s+dataset\s+", "[PREPROCESS_DONE] dataset ", txt)
        print(txt, end="" if txt.endswith("\n") else "\n", flush=True)
    if proc.stderr:
        err = proc.stderr
        err = re.sub(r"(?m)^\[PROGRESS\]\s+dataset\s+", "[PREPROCESS_PROGRESS] dataset ", err)
        err = re.sub(r"(?m)^\[DONE\]\s+dataset\s+", "[PREPROCESS_DONE] dataset ", err)
        print(err, end="" if err.endswith("\n") else "\n", flush=True)
    return int(proc.returncode)


def finalize_openorganelle_labeled_script_run(
    repo: Path,
    *,
    batch_id: str,
    successful_targets: list[str],
    img_dir: Path,
    seg_dir: Path,
    successful_targets_inference: list[str] | None = None,
    inference_img_dir: Path | None = None,
    inference_seg_dir: Path | None = None,
    chunk_shape: tuple[int, int, int],
    n_crops: int,
    voxel_nm: list[float],
    foundation: bool,
    run_preprocess: bool = True,
    n_windows: int | None = None,
    successful_pair_count_override: int | None = None,
) -> None:
    """After downloader: optional stage-4, then registry batch (post-process paths)."""
    n_win = max(1, int(n_windows)) if n_windows is not None else max(1, int(n_crops))
    successful_pairs = (
        max(0, int(successful_pair_count_override))
        if successful_pair_count_override is not None
        else len(successful_targets) * n_win
    )
    planned_em_seg = 2 * successful_pairs
    profile = {
        "batch_id": batch_id,
        "chunk_shape_zyx": list(chunk_shape),
        "n_crops": int(n_crops),
        "voxel_nm_zyx": [float(x) for x in voxel_nm[:3]],
        "foundation": bool(foundation),
        "planned_asset_downloads": int(planned_em_seg),
        "datasets_this_run": len(successful_targets),
        "n_windows": int(n_win),
    }
    ph = _make_profile_hash(
        repo, n_crops=n_crops, chunk_shape=chunk_shape, voxel_nm=voxel_nm, foundation=foundation
    )
    collected_by_dataset: list[tuple[str, list[Path]]] = []
    for name in successful_targets:
        tag = name.replace("-", "_")
        imgs = sorted(img_dir.glob(f"{tag}*_im.h5"))
        if not imgs:
            imgs = sorted(img_dir.glob(f"{tag}*_0000.nii.gz"))
        if imgs:
            collected_by_dataset.append((name, imgs))

    if run_preprocess:
        if not collected_by_dataset:
            raise RuntimeError("No downloaded image files found to preprocess.")
        # Enforce download+preprocess coupling dataset-by-dataset so a partial
        # Stage-4 failure cannot hide among otherwise successful targets.
        for name, imgs in collected_by_dataset:
            rc = run_stage4_for_images(repo, imgs, split_label_cc=True)
            if rc != 0:
                raise RuntimeError(f"Stage 4 preprocess exited {rc} for dataset '{name}'.")
            for raw_im in imgs:
                key0 = _canonical_key(raw_im)
                img_out, seg_out = _resolve_training_output_pair(repo, key0)
                if not img_out.is_file() or not seg_out.is_file():
                    raise RuntimeError(
                        f"Preprocess did not materialize expected outputs for '{name}' crop '{key0}': "
                        f"{img_out} and {seg_out}"
                    )

    items: list[dict[str, Any]] = []
    for name in successful_targets:
        tag = name.replace("-", "_")
        imgs = sorted(img_dir.glob(f"{tag}*_im.h5"))
        if not imgs:
            imgs = sorted(img_dir.glob(f"{tag}*_0000.nii.gz"))
        if not imgs:
            continue
        for raw_im in imgs:
            key0 = _canonical_key(raw_im)
            img_out, seg_out = _resolve_training_output_pair(repo, key0)
            suffix = ""
            if key0.startswith(tag):
                suffix = key0[len(tag):]
            stable = f"{name}{suffix}" if suffix else name
            items.append(
                {
                    "stable_id": stable,
                    "primary_image_path": str(img_out.resolve()) if img_out.is_file() else str(raw_im.resolve()),
                    "primary_label_path": str(seg_out.resolve()) if seg_out.is_file() else "",
                }
            )
    for name in (successful_targets_inference or []):
        tag = name.replace("-", "_")
        inf_dir = Path(inference_img_dir or (nnunet_dataset_root(repo) / "imagesTs"))
        imgs = sorted(inf_dir.glob(f"{tag}*_im.h5"))
        if not imgs:
            imgs = sorted(inf_dir.glob(f"{tag}*_0000.nii.gz"))
        if not imgs:
            continue
        for raw_im in imgs:
            key0 = _canonical_key(raw_im)
            img_out, seg_out = _resolve_inference_output_pair(repo, key0)
            suffix = ""
            if key0.startswith(tag):
                suffix = key0[len(tag):]
            stable = f"{name}{suffix}" if suffix else name
            items.append(
                {
                    "stable_id": stable,
                    "primary_image_path": str(img_out.resolve()) if img_out.is_file() else str(raw_im.resolve()),
                    "primary_label_path": str(seg_out.resolve()) if seg_out.is_file() else "",
                }
            )
    training_pairs_this_run = sum(
        1
        for it in items
        if "/imagesTr/" in str(it.get("primary_image_path") or "").replace("\\", "/")
    )
    inference_pairs_this_run = sum(
        1
        for it in items
        if "/imagesTs/" in str(it.get("primary_image_path") or "").replace("\\", "/")
    )
    profile["training_pairs_this_run"] = int(training_pairs_this_run)
    profile["inference_pairs_this_run"] = int(inference_pairs_this_run)
    profile["training_units_this_run"] = int(2 * training_pairs_this_run)
    profile["inference_units_this_run"] = int(2 * inference_pairs_this_run)
    try:
        record_openorganelle_download_batch(
            repo,
            batch_id=batch_id,
            profile_hash=ph,
            profile=profile,
            items=items,
            successful_pair_count=successful_pairs,
        )
    except Exception as exc:
        print(f"[WARN] registry batch record failed: {exc}", flush=True)
