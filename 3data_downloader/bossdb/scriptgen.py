"""BossDB download script and metadata-doc generator.

Produces two files under 3data_downloader/outputs/:
  download_bossdb_labeled.py   — executable downloader (intern + cloudvolume fallback)
  download_bossdb_labeled.md   — human-readable metadata doc

The generated .py matches OpenOrganelle labeled downloads:
  - Same nnUNet tree: ``imagesTr``/``labelsTr`` plus ``imagesTs``/``labelsTs`` (train/infer split)
  - NIfTI (.nii.gz) nnUNet naming via ``downloader_common.nnunet_labeled_export.write_h5`` (legacy hook name)
  - ``dataset.json`` synced like OpenOrganelle after successful runs
  - Registry pre-flight + completion recording
  - CLI flags: --dataset, --chunk-shape, --n-crops, --voxel-size-nm,
               --dry-run, --img-dir, --seg-dir, --window-policy
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Any

from config.paths import GENERATED_DOWNLOADER_PATHS_BLOCK


def _build_bossdb_embedded_splits(
    pairs: list[dict],
    *,
    n_crops: int,
    dataset_splits: dict[str, dict[str, int]] | None,
) -> dict[str, dict[str, int]]:
    """Per-``project_id`` train/infer window counts (OpenOrganelle-aligned defaults)."""
    _repo = Path(__file__).resolve().parents[2]
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))
    _stage3 = _repo / "3data_downloader"
    if str(_stage3) not in sys.path:
        sys.path.insert(0, str(_stage3))
    from downloader_common.labeled_train_infer_split import clamped_train_infer

    req_raw = dataset_splits or {}
    has_explicit_splits = isinstance(dataset_splits, dict) and len(dataset_splits) > 0
    req: dict[str, dict[str, int]] = {}
    for k, v in req_raw.items():
        if isinstance(v, dict):
            ks = str(k).strip()
            req[ks] = {
                "training": int(v.get("training", 1)),
                "inference": int(v.get("inference", 0)),
            }
    nc = max(1, int(n_crops))
    default_inf = max(0, min(15, nc - 1))
    out: dict[str, dict[str, int]] = {}
    for p in pairs:
        pid = str(p.get("project_id", "")).strip()
        if not pid:
            continue
        chosen: dict[str, int] | None = None
        for ck in (pid, pid.replace("/", "_"), pid.replace("/", "-")):
            if ck in req:
                chosen = req[ck]
                break
        if chosen is not None:
            tr, inf = clamped_train_infer(chosen)
        else:
            if has_explicit_splits:
                tr, inf = 0, 0
            else:
                tr, inf = clamped_train_infer(None, default_training=1, default_inference=default_inf)
        out[pid] = {"training": int(tr), "inference": int(inf)}
    return out


_BOSSDB_LABELED_TEMPLATE = '''\
#!/usr/bin/env python3
"""Auto-generated BossDB labeled downloader for mitoFoundation2.

Generated: __GEN_DATE__  (__GEN_ISO__)
Datasets:  __TOTAL_DATASETS__ paired (image + annotation)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ── nnUNet NIfTI export (``downloader_common`` — all Stage-3 downloaders) ───────────
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_STAGE3_SITE = _REPO / "3data_downloader"
if str(_STAGE3_SITE) not in sys.path:
    sys.path.insert(0, str(_STAGE3_SITE))
from downloader_common.nnunet_labeled_export import sync_dataset_json as _sync_dataset_json
from downloader_common.nnunet_labeled_export import write_h5
from downloader_common.labeled_train_infer_split import (
    alternating_volume_indices,
    clamped_train_infer,
    foundation_global_volume_pool_complete,
    global_crop_profile_completed,
    labeled_nnunet_vol_outputs_count_ge,
    mark_global_crop_profile_completed,
)

# ── intern / cloudvolume access ───────────────────────────────────────────────
try:
    from intern.remote.boss import BossRemote      # type: ignore
    from intern.resource.boss.resource import ChannelResource  # type: ignore
    _HAVE_INTERN = True
except ImportError:
    _HAVE_INTERN = False

try:
    from cloudvolume import CloudVolume  # type: ignore
    _HAVE_CV = True
except ImportError:
    _HAVE_CV = False

# ── registry helpers (optional) ──────────────────────────────────────────────
_SCRIPT_ROOT = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_ROOT.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from agent.orchestration.registry.schema import open_registry as _open_registry
    import agent.orchestration.registry.api as _registry_api
    _HAVE_REGISTRY = True
except Exception:
    _HAVE_REGISTRY = False

# ── constants ─────────────────────────────────────────────────────────────────
WEBSITE_NAME       = "BossDB"
DATA_TYPE          = "labeled (EM image + annotation, paired by experiment)"
BOSSDB_DATA_HOST   = "api.bossdb.io"

SCRIPT_GENERATED_DATE   = "__GEN_DATE__"
SCRIPT_GENERATED_AT_ISO = "__GEN_ISO__"
TOTAL_DATASETS          = __TOTAL_DATASETS__

__DOWNLOADER_PATHS_BLOCK__
DEFAULT_CHUNK_SHAPE     = __DEFAULT_CHUNK_SHAPE__
DEFAULT_N_CROPS         = __DEFAULT_N_CROPS__
DEFAULT_VOXEL_SIZE_NM   = __DEFAULT_VOXEL_SIZE_NM__
DEFAULT_WINDOW_POLICY   = "center"

print(f"[INFO] Labeled downloader loaded with {TOTAL_DATASETS} dataset(s).")
warnings.filterwarnings("ignore", category=FutureWarning)

# ── embedded dataset catalog ──────────────────────────────────────────────────
# Keys are "collection/experiment"; each value has img_channel + seg_channel.
DATASETS: dict[str, dict[str, Any]] = __DATASETS__

DATASET_SPLITS: dict[str, dict[str, Any]] = __DATASET_SPLITS__

DATASET_DOWNLOAD_PLAN = [
    {
        "project_id":  pid,
        "img_channel": cfg.get("img_channel", ""),
        "seg_channel": cfg.get("seg_channel", ""),
        "img_uri":     cfg.get("img_uri", ""),
        "seg_uri":     cfg.get("seg_uri", ""),
        "training": int((DATASET_SPLITS.get(pid) or {}).get("training", 1)),
        "inference": int((DATASET_SPLITS.get(pid) or {}).get("inference", 0)),
    }
    for pid, cfg in sorted(DATASETS.items())
]


# ── BossDB cutout access ──────────────────────────────────────────────────────
def _cutout_intern(
    collection: str, experiment: str, channel: str,
    resolution: int, x_range: list[int], y_range: list[int], z_range: list[int],
) -> np.ndarray:
    boss = BossRemote({"protocol": "https", "host": BOSSDB_DATA_HOST, "token": "public"})
    try:
        # intern expects (channel, collection, experiment)
        chan = boss.get_channel(channel, collection, experiment)
    except Exception:
        chan = ChannelResource(channel, experiment, collection)
    return boss.get_cutout(chan, resolution, x_range, y_range, z_range)


def _cutout_cloudvolume(
    collection: str, experiment: str, channel: str,
    resolution: int, x_range: list[int], y_range: list[int], z_range: list[int],
) -> np.ndarray:
    uri = f"boss://https://{BOSSDB_DATA_HOST}/{collection}/{experiment}/{channel}"
    vol = CloudVolume(uri, mip=resolution, fill_missing=True, progress=False)
    # CloudVolume returns [x, y, z, 1] — transpose to [z, y, x]
    data = vol[x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]]
    data = np.squeeze(np.asarray(data))
    if data.ndim == 3:
        data = data.transpose(2, 1, 0)
    return data


def get_cutout(
    collection: str, experiment: str, channel: str,
    resolution: int, x_range: list[int], y_range: list[int], z_range: list[int],
) -> np.ndarray:
    """Download a BossDB cutout; tries intern first, then cloudvolume."""
    if _HAVE_INTERN:
        return _cutout_intern(collection, experiment, channel, resolution, x_range, y_range, z_range)
    if _HAVE_CV:
        return _cutout_cloudvolume(collection, experiment, channel, resolution, x_range, y_range, z_range)
    raise ImportError(
        "Neither intern nor cloudvolume is installed.\\n"
        "  Install intern:      pip install intern\\n"
        "  Install cloudvolume: pip install cloud-volume"
    )


# ── volume geometry helpers ───────────────────────────────────────────────────
def _get_volume_shape_intern(collection: str, experiment: str, channel: str) -> tuple[int, int, int] | None:
    """Return (z, y, x) shape of the full volume at resolution 0, or None on failure."""
    try:
        boss = BossRemote({"protocol": "https", "host": BOSSDB_DATA_HOST, "token": "public"})
        from intern.resource.boss.resource import ExperimentResource
        exp_res  = ExperimentResource(experiment, collection)
        exp_obj  = boss.get_project(exp_res)
        cf_name  = exp_obj.coord_frame
        from intern.resource.boss.resource import CoordinateFrameResource
        cf_res   = CoordinateFrameResource(cf_name)
        cf_obj   = boss.get_project(cf_res)
        x_size   = cf_obj.x_stop - cf_obj.x_start
        y_size   = cf_obj.y_stop - cf_obj.y_start
        z_size   = cf_obj.z_stop - cf_obj.z_start
        return (int(z_size), int(y_size), int(x_size))
    except Exception:
        return None


def _get_volume_shape_cv(collection: str, experiment: str, channel: str) -> tuple[int, int, int] | None:
    try:
        uri = f"boss://https://{BOSSDB_DATA_HOST}/{collection}/{experiment}/{channel}"
        vol = CloudVolume(uri, mip=0, fill_missing=True, progress=False)
        # CloudVolume volume_size is in [x, y, z]
        x, y, z = vol.volume_size
        return (int(z), int(y), int(x))
    except Exception:
        return None


def get_volume_shape(collection: str, experiment: str, channel: str) -> tuple[int, int, int] | None:
    if _HAVE_INTERN:
        shape = _get_volume_shape_intern(collection, experiment, channel)
        if shape:
            return shape
    if _HAVE_CV:
        return _get_volume_shape_cv(collection, experiment, channel)
    return None


def _center_crop_offset(vol_shape: tuple[int, int, int], chunk: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(max(0, (s - c) // 2) for s, c in zip(vol_shape, chunk))  # type: ignore[return-value]


def _clamp_chunk(vol_shape: tuple[int, int, int], chunk: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(min(s, c) for s, c in zip(vol_shape, chunk))  # type: ignore[return-value]


def _crop_index_suffix(i: int, total: int) -> str:
    """Canonical crop suffix shared with other providers."""
    idx = max(1, int(i))
    if int(total) > 0:
        idx = min(idx, int(total))
    return f"_vol{idx}"


def _resolve_batch_coherent_suffixes(tag: str, img_dir: str, seg_dir: str, n_windows: int) -> list[str]:
    """If any base ``_volN`` collides, assign one shared ``_M`` suffix to all windows."""
    n = max(1, int(n_windows))
    base_suffixes = [f"_vol{i}" for i in range(1, n + 1)]

    def _exists(stem: str) -> bool:
        return os.path.exists(os.path.join(img_dir, f"{tag}{stem}_0000.nii.gz")) or os.path.exists(
            os.path.join(seg_dir, f"{tag}{stem}.nii.gz")
        )

    if not any(_exists(s) for s in base_suffixes):
        return base_suffixes
    m = 2
    while True:
        trial = [f"_vol{i}_{m}" for i in range(1, n + 1)]
        if all(not _exists(s) for s in trial):
            print(f"[INFO] {tag}: batch collision detected; applying shared suffix _{m} to all {n} crop(s)")
            return trial
        m += 1


def _safe_get_cutout(
    collection: str,
    experiment: str,
    channel: str,
    resolution: int,
    x_range: list[int],
    y_range: list[int],
    z_range: list[int],
    retries: int = 3,
) -> np.ndarray:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return get_cutout(collection, experiment, channel, resolution, x_range, y_range, z_range)
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(0.5 * attempt)
    assert last_exc is not None
    raise last_exc


def _probe_pair_ok(
    collection: str,
    experiment: str,
    img_channel: str,
    seg_channel: str,
    vol_shape: tuple[int, int, int],
    offset_zyx: tuple[int, int, int],
    chunk_zyx: tuple[int, int, int],
) -> bool:
    oz, oy, ox = offset_zyx
    dz = max(8, min(32, chunk_zyx[0], vol_shape[0]))
    dy = max(8, min(32, chunk_zyx[1], vol_shape[1]))
    dx = max(8, min(32, chunk_zyx[2], vol_shape[2]))
    ez = min(vol_shape[0], oz + dz)
    ey = min(vol_shape[1], oy + dy)
    ex = min(vol_shape[2], ox + dx)
    zr, yr, xr = [oz, ez], [oy, ey], [ox, ex]
    try:
        _safe_get_cutout(collection, experiment, img_channel, 0, xr, yr, zr, retries=1)
        _safe_get_cutout(collection, experiment, seg_channel, 0, xr, yr, zr, retries=1)
        return True
    except Exception:
        return False


def _candidate_offsets(
    vol_shape: tuple[int, int, int],
    chunk: tuple[int, int, int],
    n_random: int = 20,
) -> list[tuple[int, int, int]]:
    maxz = max(0, vol_shape[0] - chunk[0])
    maxy = max(0, vol_shape[1] - chunk[1])
    maxx = max(0, vol_shape[2] - chunk[2])
    gridz = sorted(set([0, maxz // 4, maxz // 2, (3 * maxz) // 4, maxz]))
    gridy = sorted(set([0, maxy // 4, maxy // 2, (3 * maxy) // 4, maxy]))
    gridx = sorted(set([0, maxx // 4, maxx // 2, (3 * maxx) // 4, maxx]))
    out = [(z, y, x) for z in gridz for y in gridy for x in gridx]
    for _ in range(max(0, n_random)):
        out.append((
            random.randint(0, maxz) if maxz > 0 else 0,
            random.randint(0, maxy) if maxy > 0 else 0,
            random.randint(0, maxx) if maxx > 0 else 0,
        ))
    return out


# ── registry helpers ──────────────────────────────────────────────────────────
def _make_profile(n_crops: int, chunk_shape: tuple, voxel_nm: list[float]) -> str | None:
    if not _HAVE_REGISTRY:
        return None
    try:
        return _registry_api.make_download_profile_hash(
            n_crops=n_crops,
            chunk_zyx=chunk_shape,
            voxel_nm_zyx=tuple(float(v) for v in voxel_nm[:3]),
            mode="labeled",
            foundation=False,
        )
    except Exception:
        return None


def _preflight(targets: list[str], chunk_shape: tuple, voxel_nm: list[float]) -> tuple[list[str], list[str]]:
    """Return *(pending, skipped)*. Registry-first — same precedence as OpenOrganelle labeled Stage 3.

    1. Per-side registry profiles plus **OpenOrganelle-style** `{tag}_vol*` stack counts under Tr/Ts.
    2. If still incomplete, split-agnostic global volume pool completion.
    """
    if not _HAVE_REGISTRY:
        return list(targets), []
    try:
        conn = _open_registry()
        pid = _registry_api.upsert_provider(conn, name="BossDB", base_url="https://api.bossdb.io")
        conn.commit()
        pending, skipped = [], []
        voxel = list(voxel_nm[:3])
        for proj_id in targets:
            tr, inf = clamped_train_infer(DATASET_SPLITS.get(proj_id))
            if tr + inf <= 0:
                skipped.append(proj_id)
                continue
            _tag = proj_id.replace("/", "_").replace("-", "_")
            _total_n = int(tr) + int(inf)
            _pool_done = foundation_global_volume_pool_complete(DEFAULT_LABELED_BASE, _tag, _total_n)
            if _pool_done and global_crop_profile_completed(DEFAULT_LABELED_BASE, _tag, _total_n):
                skipped.append(proj_id)
                continue
            did = _registry_api.get_dataset_id(conn, pid, proj_id)
            all_done = True
            for side_name, n_side in (("training", tr), ("inference", inf)):
                if n_side <= 0:
                    continue
                prof = _make_profile(int(n_side), chunk_shape, voxel)
                reg_ok = (
                    did is not None
                    and prof is not None
                    and _registry_api.is_dataset_type_downloaded_for_profile(conn, did, "em_volume", prof)
                    and _registry_api.is_dataset_type_downloaded_for_profile(conn, did, "mito_seg", prof)
                )
                fs_ok = labeled_nnunet_vol_outputs_count_ge(
                    proj_id,
                    nnunet_raw_root=DEFAULT_LABELED_BASE,
                    split=side_name,
                    min_pairs=int(n_side),
                )
                if not (reg_ok and fs_ok):
                    all_done = False
                    break
            if not all_done:
                if _pool_done:
                    mark_global_crop_profile_completed(DEFAULT_LABELED_BASE, _tag, _total_n)
                    all_done = True
            if all_done:
                skipped.append(proj_id)
            else:
                pending.append(proj_id)
        conn.close()
        return pending, skipped
    except Exception as exc:
        print(f"[REGISTRY] Pre-flight unavailable ({exc}); running all.")
        return list(targets), []


def _record_start(proj_id: str, img_uri: str, seg_uri: str,
                  img_dir: str, seg_dir: str, profile: str | None) -> tuple[list[int] | None, Any]:
    if not _HAVE_REGISTRY or profile is None:
        return None, None
    try:
        conn = _open_registry()
        pid  = _registry_api.upsert_provider(conn, name="BossDB", base_url="https://api.bossdb.io")
        conn.commit()
        did  = _registry_api.get_dataset_id(conn, pid, proj_id)
        if did is None:
            did = _registry_api.upsert_dataset(conn, provider_id=pid, stable_id=proj_id)
            conn.commit()
        em_id  = _registry_api.upsert_asset(conn, dataset_id=did, asset_type="em_volume",  remote_url=img_uri)
        seg_id = _registry_api.upsert_asset(conn, dataset_id=did, asset_type="mito_seg",   remote_url=seg_uri)
        conn.commit()
        dl_ids = [
            _registry_api.record_download_start(conn, em_id,  img_dir, download_profile_hash=profile),
            _registry_api.record_download_start(conn, seg_id, seg_dir, download_profile_hash=profile),
        ]
        conn.commit()
        return dl_ids, conn
    except Exception as exc:
        print(f"[REGISTRY] Warning: could not record start for {proj_id}: {exc}")
        return None, None


def _record_complete(conn: Any, dl_ids: list[int] | None, proj_id: str) -> None:
    if conn is None or dl_ids is None:
        return
    try:
        for dl_id in dl_ids:
            _registry_api.record_download_complete(conn, dl_id)
        conn.commit()
        conn.close()
        print(f"[REGISTRY] Recorded: {proj_id}")
    except Exception as exc:
        print(f"[REGISTRY] Warning: could not record completion for {proj_id}: {exc}")


def _update_metadata_md() -> None:
    md = Path(__file__).resolve().parent / "download_bossdb_labeled.md"
    if not md.is_file():
        return
    now = datetime.now()
    try:
        text = md.read_text(encoding="utf-8")
        import re
        text = re.sub(
            r"\\*\\*Last download:\\*\\*.*",
            f"**Last download:** {now.strftime(\'%Y-%m-%d\')}",
            text,
        )
        text = re.sub(
            r"\\*\\*Last Download At \\(ISO\\):\\*\\*.*",
            f"**Last Download At (ISO):** {now.replace(microsecond=0).isoformat()}",
            text,
        )
        md.write_text(text, encoding="utf-8")
    except Exception:
        pass


# ── dataset download ──────────────────────────────────────────────────────────
def download_bossdb_dataset(
    project_id: str,
    *,
    chunk_shape: tuple[int, int, int],
    n_crops: int,
    window_policy: str,
    img_dir: str,
    seg_dir: str,
    dry_run: bool = False,
) -> None:
    if project_id not in DATASETS:
        print(f"[ERROR] {project_id!r} not in DATASETS.")
        return
    cfg = DATASETS[project_id]
    collection  = cfg["collection"]
    experiment  = cfg["experiment"]
    img_channel = cfg["img_channel"]
    seg_channel = cfg["seg_channel"]

    print(f"[INFO] {project_id} | img={img_channel}  seg={seg_channel}")
    print(f"[INFO]   img_uri: {cfg['img_uri']}")
    print(f"[INFO]   seg_uri: {cfg['seg_uri']}")
    print(f"[INFO]   voxel_size_nm: {cfg.get('voxel_size_nm')}")

    if dry_run:
        print(f"[DRY-RUN] Would download {project_id} → img_dir={img_dir}  seg_dir={seg_dir}")
        return

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    vol_shape = get_volume_shape(collection, experiment, img_channel)
    if vol_shape is None:
        print(f"[WARN] Could not determine volume shape for {project_id}; using chunk as full extent.")
        vol_shape = chunk_shape

    ch_eff = _clamp_chunk(vol_shape, chunk_shape)
    if ch_eff != chunk_shape:
        print(f"[INFO] Chunk capped {chunk_shape} → {ch_eff} for volume {vol_shape}")

    if window_policy == "center" or n_crops == 1:
        requested_offsets = [_center_crop_offset(vol_shape, ch_eff)]
    else:
        # Non-overlapping grid
        axes = []
        for dim, ch in zip(vol_shape, ch_eff):
            n = max(1, dim // ch)
            axes.append([i * ch for i in range(n)])
        offsets_all = [(z, y, x) for z in axes[0] for y in axes[1] for x in axes[2]]
        center = _center_crop_offset(vol_shape, ch_eff)
        offsets_all.sort(
            key=lambda o: (o[0]-center[0])**2 + (o[1]-center[1])**2 + (o[2]-center[2])**2
        )
        requested_offsets = offsets_all[:n_crops]

    offsets: list[tuple[int, int, int]] = []
    for o in requested_offsets:
        if _probe_pair_ok(collection, experiment, img_channel, seg_channel, vol_shape, o, ch_eff):
            offsets.append(o)

    if len(offsets) < max(1, n_crops):
        for o in _candidate_offsets(vol_shape, ch_eff):
            if o in offsets:
                continue
            if _probe_pair_ok(collection, experiment, img_channel, seg_channel, vol_shape, o, ch_eff):
                offsets.append(o)
                if len(offsets) >= max(1, n_crops):
                    break

    if not offsets:
        raise RuntimeError(
            "Could not find any valid crop window where both image and label cutouts succeed."
        )
    offsets = offsets[:max(1, n_crops)]

    tag = project_id.replace("/", "_").replace("-", "_")
    planned_suffixes = _resolve_batch_coherent_suffixes(tag, img_dir, seg_dir, len(offsets))
    for idx, (oz, oy, ox) in enumerate(offsets, start=1):
        ez = min(vol_shape[0], oz + ch_eff[0])
        ey = min(vol_shape[1], oy + ch_eff[1])
        ex = min(vol_shape[2], ox + ch_eff[2])
        # intern/cloudvolume use x_range, y_range, z_range
        x_range = [ox, ex]
        y_range = [oy, ey]
        z_range = [oz, ez]
        suffix = planned_suffixes[idx - 1] if idx - 1 < len(planned_suffixes) else _crop_index_suffix(idx, len(offsets))
        label = f"{tag}{suffix}"
        print(f"[INFO]   crop {idx}/{len(offsets)}: z={z_range} y={y_range} x={x_range}")
        img = _safe_get_cutout(collection, experiment, img_channel, 0, x_range, y_range, z_range)
        seg = _safe_get_cutout(collection, experiment, seg_channel, 0, x_range, y_range, z_range)
        write_h5(os.path.join(img_dir, f"{label}_im.h5"),  np.asarray(img, dtype=img.dtype),  label + "_im")
        write_h5(os.path.join(seg_dir, f"{label}_seg.h5"), np.asarray(seg, dtype=seg.dtype), label + "_seg")
        print(f"[DONE]  crop {idx}: {label}_0000.nii.gz  {label}.nii.gz")


# ── CLI entry-point ───────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="Download BossDB labeled data (EM + annotation)")
    p.add_argument("--dataset", "-d", default=None, choices=list(DATASETS.keys()),
                   help="Download a single project_id; default: all")
    p.add_argument("--chunk-shape", "-c",
                   default=",".join(str(x) for x in DEFAULT_CHUNK_SHAPE),
                   help="Output voxel count z,y,x  (default: %(default)s)")
    p.add_argument("--n-crops",    type=int, default=DEFAULT_N_CROPS)
    p.add_argument("--voxel-size-nm",
                   default=",".join(str(x) for x in DEFAULT_VOXEL_SIZE_NM),
                   help="Target voxel spacing nm z,y,x  (informational only for BossDB)")
    p.add_argument("--window-policy", choices=["center", "grid"], default=DEFAULT_WINDOW_POLICY)
    p.add_argument("--img-dir",  default=None, help="Override image output directory")
    p.add_argument("--seg-dir",  default=None, help="Override label output directory")
    p.add_argument("--dry-run",  action="store_true", help="Print plan but do not download")
    p.add_argument("--list",     action="store_true", help="List datasets and exit")
    args = p.parse_args()

    if args.list:
        print(f"Available datasets ({TOTAL_DATASETS}) [mode=labeled] from embedded BossDB catalog:")
        for row in DATASET_DOWNLOAD_PLAN:
            print(f"  {row['project_id']}  img={row['img_channel']}  seg={row['seg_channel']}")
        return

    chunk_shape   = tuple(int(x) for x in args.chunk_shape.split(","))
    voxel_nm      = [float(x) for x in args.voxel_size_nm.split(",")]
    targets       = [args.dataset] if args.dataset else sorted(DATASETS.keys())

    pending, skipped = (targets, []) if args.dry_run else _preflight(targets, chunk_shape, voxel_nm)
    print(f"[PLAN] {len(pending)} of {len(targets)} dataset(s) need download; {len(skipped)} already complete.")
    if skipped:
        print(f"[SKIP] {', '.join(skipped)}")
    if not pending:
        print("[NOOP] All datasets already complete for this profile.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch = os.environ.get("MITO2_DOWNLOAD_BATCH_ID", "").strip() or f"bossdb_{stamp}"
    tr = DEFAULT_LABELED_BASE
    inf = DEFAULT_INFERENCE_BASE
    if args.img_dir and args.seg_dir:
        training_img_dir = args.img_dir
        training_seg_dir = args.seg_dir
        inference_img_dir = args.img_dir
        inference_seg_dir = args.seg_dir
    elif not args.img_dir and not args.seg_dir:
        training_img_dir = os.path.join(tr, "imagesTr")
        training_seg_dir = os.path.join(tr, "labelsTr")
        inference_img_dir = os.path.join(inf, "imagesTs")
        inference_seg_dir = os.path.join(inf, "labelsTs")
        os.makedirs(training_img_dir, exist_ok=True)
        os.makedirs(training_seg_dir, exist_ok=True)
        os.makedirs(inference_img_dir, exist_ok=True)
        os.makedirs(inference_seg_dir, exist_ok=True)
    else:
        raise SystemExit("Provide both --img-dir and --seg-dir, or neither.")

    planned_windows = 0
    for pid in pending:
        a, b = clamped_train_infer(DATASET_SPLITS.get(pid))
        planned_windows += int(a) + int(b)

    print("=" * 60)
    print("BossDB labeled download summary")
    print(f"- Website: {WEBSITE_NAME}")
    print(f"- Total datasets available in this file: {len(DATASETS)}")
    print(f"- Data type: {DATA_TYPE}")
    print(
        "- Native BossDB voxel sampling (not isotropic foundation resampling); registry ``foundation`` profile "
        "remains false — preflight/layout checks match OpenOrganelle labeled (per-side + global alternating)"
    )
    print(f"- Batch: {batch}")
    print(f"- Datasets: {len(pending)} pending (of {len(targets)} total)")
    print(f"- Windows per dataset: mixed per dataset (policy={args.window_policy})")
    print(f"- Planned image/label pairs: {planned_windows}")
    print("- Output: NIfTI (.nii.gz) nnUNet naming")
    print(f"- Requested output max voxels (z,y,x): {chunk_shape}")
    print(f"- Requested output spacing nm (z,y,x): {tuple(float(v) for v in voxel_nm)}")
    print("- Planned dataset downloads (dataset | image | label):")
    for row in DATASET_DOWNLOAD_PLAN:
        if row["project_id"] in pending:
            print(f"  - {row['project_id']} | img={row['img_uri']} | seg={row['seg_uri']}")
    print("=" * 60)

    success, failed, skipped_run = [], [], []
    for idx, proj_id in enumerate(pending, 1):
        sp = DATASET_SPLITS.get(proj_id) or {}
        train_n, infer_n = clamped_train_infer(sp)
        total_n = train_n + infer_n
        print(
            f"[PROGRESS] dataset {idx}/{len(pending)}: {proj_id} "
            f"(train={train_n} infer={infer_n} windows={total_n})"
        )
        cfg = DATASETS.get(proj_id, {})
        if total_n <= 0:
            print(f"[SKIP] dataset {idx}/{len(pending)}: {proj_id} (training=0, inference=0)")
            continue
        dl_ids_tr, reg_tr = (None, None)
        dl_ids_ts, reg_ts = (None, None)
        prof_tr = _make_profile(train_n, chunk_shape, voxel_nm) if train_n > 0 else None
        prof_ts = _make_profile(infer_n, chunk_shape, voxel_nm) if infer_n > 0 else None
        _status = "failed"
        if not args.dry_run:
            if train_n > 0:
                dl_ids_tr, reg_tr = _record_start(
                    proj_id,
                    str(cfg.get("img_uri", "")),
                    str(cfg.get("seg_uri", "")),
                    training_img_dir,
                    training_seg_dir,
                    prof_tr,
                )
            if infer_n > 0:
                dl_ids_ts, reg_ts = _record_start(
                    proj_id,
                    str(cfg.get("img_uri", "")),
                    str(cfg.get("seg_uri", "")),
                    inference_img_dir,
                    inference_seg_dir,
                    prof_ts,
                )
        try:
            download_bossdb_dataset(
                proj_id,
                chunk_shape=chunk_shape,
                n_crops=total_n,
                window_policy=args.window_policy,
                img_dir=training_img_dir,
                seg_dir=training_seg_dir,
                dry_run=args.dry_run,
            )
            if infer_n > 0 and not args.dry_run:
                _, ts_idx = alternating_volume_indices(train_n, infer_n)
                train_inst = Path(training_seg_dir).parent / "labelsTr-instance"
                infer_inst = Path(inference_seg_dir).parent / "labelsTs-instance"
                infer_inst.mkdir(parents=True, exist_ok=True)
                tag = proj_id.replace("/", "_").replace("-", "_")
                for vidx in ts_idx:
                    img_n = f"{tag}_vol{vidx}_0000.nii.gz"
                    lbl_n = f"{tag}_vol{vidx}.nii.gz"
                    ins_n = f"{tag}_vol{vidx}_instance.nii.gz"
                    s_img = Path(training_img_dir) / img_n
                    s_lbl = Path(training_seg_dir) / lbl_n
                    s_ins = train_inst / ins_n
                    d_img = Path(inference_img_dir) / img_n
                    d_lbl = Path(inference_seg_dir) / lbl_n
                    d_ins = infer_inst / ins_n
                    if s_img.is_file():
                        shutil.move(str(s_img), str(d_img))
                    if s_lbl.is_file():
                        shutil.move(str(s_lbl), str(d_lbl))
                    if s_ins.is_file():
                        shutil.move(str(s_ins), str(d_ins))
            if not args.dry_run:
                _record_complete(reg_tr, dl_ids_tr, proj_id)
                _record_complete(reg_ts, dl_ids_ts, proj_id)
                if total_n > 0:
                    tag_done = proj_id.replace("/", "_").replace("-", "_")
                    mark_global_crop_profile_completed(DEFAULT_LABELED_BASE, tag_done, total_n)
            success.append(proj_id)
            _status = "success"
        except Exception as exc:
            msg = str(exc)
            if "Could not find any valid crop window" in msg or "HTTP response: (500)" in msg:
                print(f"[WARN] {proj_id}: unavailable/unstable on BossDB backend, skipped. ({msg})")
                skipped_run.append(proj_id)
                _status = "skipped"
            else:
                print(f"[ERROR] {proj_id}: {exc}")
                failed.append(proj_id)
                _status = "failed"
        print(f"[DONE] dataset {idx}/{len(pending)}: {proj_id} [{_status}]")

    print(f"[SUMMARY] success={len(success)}  skipped={len(skipped_run)}  failed={len(failed)}")
    if skipped:
        print(f"  Already complete: {', '.join(skipped)}")
    if skipped_run:
        print(f"  Skipped: {', '.join(skipped_run)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
        raise SystemExit(1)
    if not args.dry_run:
        _update_metadata_md()
        ds_json = _sync_dataset_json(DEFAULT_LABELED_BASE)
        print(f"[DONE] dataset.json synced: {ds_json}")


if __name__ == "__main__":
    main()
'''


# ── metadata doc template ─────────────────────────────────────────────────────
_BOSSDB_MD_TEMPLATE = """\
# BossDB — Download script metadata (labeled)

**Source:** https://api.metadata.bossdb.org
**Script generated:** __GEN_DATE__
**Generated At (ISO):** __GEN_ISO__
**Website Name:** BossDB
**Downloader mode:** labeled — EM image + annotation (paired by experiment)
**Companion script:** `download_bossdb_labeled.py`

## Runtime parameters (CLI-overridable)

| Setting | Value |
| --- | --- |
| `chunk_shape` (voxels, z,y,x) | `__CHUNK_SHAPE__` |
| `n_crops` | __N_CROPS__ |
| `voxel_size_nm` (nm, z,y,x) | `__VOXEL_SIZE_NM__` |

## Paired datasets (__TOTAL_DATASETS__ experiments)

| project_id | img_channel | seg_channel | organism | modality |
| --- | --- | --- | --- | --- |
__DATASET_TABLE__

**Last download:** Not yet run
**Last Download At (ISO):** —
"""


# ── rendering ─────────────────────────────────────────────────────────────────

def render_download_script(
    pairs: list[dict],
    *,
    chunk_shape: tuple[int, int, int] = (128, 128, 128),
    n_crops: int = 1,
    voxel_size_nm: tuple[float, float, float] = (16.0, 16.0, 16.0),
    training_root: str = "",
    dataset_splits: dict[str, dict[str, int]] | None = None,
) -> str:
    now = datetime.now()
    datasets_repr = json.dumps(
        {p["project_id"]: p for p in pairs},
        indent=2,
        ensure_ascii=False,
    )
    splits = _build_bossdb_embedded_splits(pairs, n_crops=n_crops, dataset_splits=dataset_splits)
    splits_repr = json.dumps(splits, indent=2, sort_keys=True, ensure_ascii=False)
    script = _BOSSDB_LABELED_TEMPLATE
    script = script.replace("__GEN_DATE__", now.strftime("%Y-%m-%d"))
    script = script.replace("__GEN_ISO__",  now.replace(microsecond=0).isoformat())
    script = script.replace("__TOTAL_DATASETS__", str(len(pairs)))
    script = script.replace("__DATASETS__",       datasets_repr)
    script = script.replace("__DATASET_SPLITS__", splits_repr)
    script = script.replace("__DOWNLOADER_PATHS_BLOCK__", GENERATED_DOWNLOADER_PATHS_BLOCK)
    script = script.replace("__DEFAULT_CHUNK_SHAPE__",  repr(list(chunk_shape)))
    script = script.replace("__DEFAULT_N_CROPS__",      str(n_crops))
    script = script.replace("__DEFAULT_VOXEL_SIZE_NM__", repr(list(voxel_size_nm)))
    return script


def render_metadata_doc(
    pairs: list[dict],
    *,
    chunk_shape: tuple[int, int, int] = (128, 128, 128),
    n_crops: int = 1,
    voxel_size_nm: tuple[float, float, float] = (16.0, 16.0, 16.0),
) -> str:
    now = datetime.now()
    rows = "\n".join(
        f"| {p['project_id']} | {p['img_channel']} | {p['seg_channel']} "
        f"| {p.get('organism','')} | {p.get('modality','')} |"
        for p in pairs
    )
    doc = _BOSSDB_MD_TEMPLATE
    doc = doc.replace("__GEN_DATE__",      now.strftime("%Y-%m-%d"))
    doc = doc.replace("__GEN_ISO__",       now.replace(microsecond=0).isoformat())
    doc = doc.replace("__TOTAL_DATASETS__", str(len(pairs)))
    doc = doc.replace("__CHUNK_SHAPE__",   ", ".join(str(x) for x in chunk_shape))
    doc = doc.replace("__N_CROPS__",       str(n_crops))
    doc = doc.replace("__VOXEL_SIZE_NM__", ", ".join(str(x) for x in voxel_size_nm))
    doc = doc.replace("__DATASET_TABLE__", rows)
    return doc


def write_bossdb_outputs(
    outputs_dir: Path,
    pairs: list[dict],
    *,
    chunk_shape: tuple[int, int, int] = (128, 128, 128),
    n_crops: int = 1,
    voxel_size_nm: tuple[float, float, float] = (16.0, 16.0, 16.0),
    training_root: str = "",
    dataset_splits: dict[str, dict[str, int]] | None = None,
) -> tuple[Path, Path]:
    """Write download_bossdb_labeled.py + .md to *outputs_dir*.

    Returns (py_path, md_path).
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)
    py_path = outputs_dir / "download_bossdb_labeled.py"
    md_path = outputs_dir / "download_bossdb_labeled.md"

    script = render_download_script(
        pairs,
        chunk_shape=chunk_shape,
        n_crops=n_crops,
        voxel_size_nm=voxel_size_nm,
        training_root=training_root,
        dataset_splits=dataset_splits,
    )
    doc = render_metadata_doc(
        pairs,
        chunk_shape=chunk_shape,
        n_crops=n_crops,
        voxel_size_nm=voxel_size_nm,
    )

    py_path.write_text(script, encoding="utf-8")
    md_path.write_text(doc,    encoding="utf-8")
    try:
        py_path.chmod(py_path.stat().st_mode | 0o111)
    except OSError:
        pass
    return py_path, md_path
