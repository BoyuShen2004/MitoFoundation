#!/usr/bin/env python3
import ast
import json
import re
import numpy as np
import argparse
import os
import sys
import shutil
import warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore", category=FutureWarning, module=r"fibsem_tools\.io\.n5\.core")
from fibsem_tools.io import read

from pathlib import Path
import sys as _sys
# Generated scripts live under .../3data_downloader/outputs/ — package root is one level up.
_OG_ROOT = Path(__file__).resolve().parents[1]
if str(_OG_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_OG_ROOT))
from openorganelle.foundation_resample import download_foundation_labeled as _dl_foundation_labeled


TOTAL_DATASETS = 1
print(f"[INFO] Labeled downloader loaded with {TOTAL_DATASETS} dataset(s).")
WEBSITE_NAME = "OpenOrganelle"
DATA_TYPE = "labeled (raw EM + mitochondria good labels, non-prediction)"

SCRIPT_GENERATED_DATE = "2026-04-30"
SCRIPT_GENERATED_AT_ISO = "2026-04-30T23:19:38"

# Dataset configuration (explicit source addresses)
DATASETS = {
  "jrc_mus-liver": {
    "_source": "s3_probe",
    "dataset_name": "jrc_mus-liver",
    "download_split": {
      "inference": 1,
      "training": 1
    },
    "grid_axes": "['x', 'y', 'z']",
    "grid_dimensions": "[12750, 12725, 8938]",
    "img_path": "s3://janelia-cosem-datasets/jrc_mus-liver/jrc_mus-liver.n5/em/fibsem-uint8/s1",
    "seg_path": "s3://janelia-cosem-datasets/jrc_mus-liver/jrc_mus-liver.n5/labels/mito_seg/s0",
    "voxel_size_nm": [
      8.0,
      8.0,
      8.0
    ]
  }
}
DATASET_SPLITS = {
  "jrc_mus-liver": {
    "inference": 1,
    "training": 1
  }
}
DATASET_DOWNLOAD_PLAN = [
    {
        "dataset_name": name,
        "img_path": (cfg.get("img_path") or "").strip(),
        "seg_path": (cfg.get("seg_path") or "").strip(),
        "training": int((DATASET_SPLITS.get(name) or {}).get("training", 1)),
        "inference": int((DATASET_SPLITS.get(name) or {}).get("inference", 0)),
    }
    for name, cfg in sorted(DATASETS.items())
]

_REPO_FOR_PATHS = Path(__file__).resolve().parents[2]
if str(_REPO_FOR_PATHS) not in sys.path:
    sys.path.insert(0, str(_REPO_FOR_PATHS))
from config.paths import nnunet_dataset_root
_DEFAULT_NNUNET_DATASET = str(nnunet_dataset_root(_REPO_FOR_PATHS))
DEFAULT_LABELED_BASE = _DEFAULT_NNUNET_DATASET
DEFAULT_INFERENCE_BASE = _DEFAULT_NNUNET_DATASET
DEFAULT_DATASET = "jrc_mus-liver"
DEFAULT_OFFSETS = None
DEFAULT_CHUNK_SHAPE = (128, 128, 128)
DEFAULT_N_CROPS = 2
DEFAULT_VOXEL_SIZE_NM = [16.0, 16.0, 16.0]
FORCED_OUTPUT_SPACING_NM_ZYX = (16.0, 16.0, 16.0)
DEFAULT_FOUNDATION = True
DEFAULT_WINDOW_POLICY = "center"

_OG_STAGE3 = Path(__file__).resolve().parents[1]
if str(_OG_STAGE3) not in sys.path:
    sys.path.insert(0, str(_OG_STAGE3))
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

def _try_read(path: str, creds: dict):
    import zarr
    import s3fs
    base = path.rstrip("/")
    candidates = list(dict.fromkeys([path] + [base + s for s in ("/s0", "/s1", "/0", "/1", "")]))
    for cand in candidates:
        try:
            return read(path=cand, storage_options=creds)
        except Exception:
            pass
    if ".zarr" in path:
        s3 = s3fs.S3FileSystem(anon=creds.get("anon", True))
        zarr_root = path.split(".zarr")[0] + ".zarr"
        inner = path.split(".zarr", 1)[1].lstrip("/")
        zarr_root_s3map = zarr_root[len("s3://"):] if zarr_root.startswith("s3://") else zarr_root
        store = s3fs.S3Map(root=zarr_root_s3map, s3=s3, check=False)
        grp = zarr.open_group(store, mode="r")
        inner_base = inner.rstrip("/")
        for suffix in ["", "/s0", "/s1", "/0", "/1"]:
            key = (inner_base + suffix).lstrip("/")
            try:
                return grp[key]
            except Exception:
                pass
    raise RuntimeError("Could not open " + path)

def _as_array(obj):
    if hasattr(obj, "shape") and hasattr(obj, "dtype") and isinstance(getattr(obj, "shape"), tuple):
        return obj
    for key in ("s0", "s1", "0", "1"):
        try:
            cand = obj[key]
            if hasattr(cand, "shape") and hasattr(cand, "dtype") and isinstance(getattr(cand, "shape"), tuple):
                return cand
        except Exception:
            pass
    raise RuntimeError("Resolved object is not an array (missing shape/dtype).")

def _parse_num_vector_catalog(text):
    if not text or not str(text).strip():
        return None
    try:
        v = ast.literal_eval(str(text).strip())
    except (SyntaxError, ValueError):
        try:
            v = json.loads(str(text).strip())
        except Exception:
            return None
    if isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v):
        return [int(round(float(x))) for x in v]
    return None

def _catalog_dims_zyx(grid_dimensions, grid_axes):
    nums = _parse_num_vector_catalog(grid_dimensions)
    if not nums:
        return None
    axes_s = (grid_axes or "").strip()
    if not axes_s:
        return (nums[2], nums[1], nums[0])
    try:
        axes = ast.literal_eval(axes_s)
    except (SyntaxError, ValueError):
        axes = None
    if not isinstance(axes, (list, tuple)) or len(axes) != 3:
        return (nums[2], nums[1], nums[0])
    axis_to_len = {str(a).lower(): nums[i] for i, a in enumerate(axes)}
    if not all(k in axis_to_len for k in ("z", "y", "x")):
        return None
    return (axis_to_len["z"], axis_to_len["y"], axis_to_len["x"])

def _fibsem_pyramid_level(url):
    found = list(re.finditer(r"/s(\d+)(?:/|$)", url))
    if not found:
        return 0
    return int(found[-1].group(1))

def _expected_em_vol_shape_zyx(cfg, img_path):
    czyx = _catalog_dims_zyx(cfg.get("grid_dimensions"), cfg.get("grid_axes"))
    if czyx is None:
        return None
    cz, cy, cx = czyx
    lv = _fibsem_pyramid_level(img_path)
    d = 2 ** lv
    return (max(1, cz // d), max(1, cy // d), max(1, cx // d))

def _planning_vol_shape_em(cfg, img_path, img_shape):
    ce = _expected_em_vol_shape_zyx(cfg, img_path)
    if ce is None:
        return img_shape
    inter = tuple(min(int(ce[i]), int(img_shape[i])) for i in range(3))
    tol = 2
    if any(abs(int(ce[i]) - int(img_shape[i])) > tol for i in range(3)):
        print(f"[WARN] catalog EM shape {ce} vs opened {img_shape}; planning intersection {inter}")
    elif inter != img_shape:
        print(f"[INFO] planning volume capped (catalog∩opened): {inter} (opened {img_shape})")
    else:
        print(f"[INFO] catalog EM shape {ce} matches opened volume {img_shape}")
    return inter

def _effective_chunk(vol_shape, chunk_shape):
    return tuple(min(int(c), int(s)) for c, s in zip(chunk_shape, vol_shape))

def _center_offset(vol_shape, ch_eff):
    return tuple(max(0, (s - c) // 2) for s, c in zip(vol_shape, ch_eff))

def _window_from_offset(vol_shape, offset, ch_eff):
    starts, ends = [], []
    for dim, off, c in zip(vol_shape, offset, ch_eff):
        d, o, ch = int(dim), int(off), int(c)
        if d <= 0 or ch <= 0:
            return None
        s = max(0, min(o, d - ch))
        e = s + ch
        starts.append(s)
        ends.append(e)
    return tuple(starts), tuple(ends)

def _grid_offsets_non_overlapping(vol_shape, ch_eff, n_crops):
    if n_crops <= 1:
        return [_center_offset(vol_shape, ch_eff)], 1
    axes = []
    for dim, ch in zip(vol_shape, ch_eff):
        d, c = int(dim), int(ch)
        if d <= c:
            axes.append([0])
            continue
        n = max(1, d // c)
        axes.append([i * c for i in range(n)])
    offsets = [(z, y, x) for z in axes[0] for y in axes[1] for x in axes[2]]
    if not offsets:
        offsets = [_center_offset(vol_shape, ch_eff)]
    center = _center_offset(vol_shape, ch_eff)
    offsets.sort(key=lambda o: (o[0] - center[0]) ** 2 + (o[1] - center[1]) ** 2 + (o[2] - center[2]) ** 2)
    total_possible = len(offsets)
    return offsets[: min(int(n_crops), total_possible)], total_possible

def _axis_ratios(img_shape, seg_shape):
    return tuple(seg_shape[i] / float(img_shape[i]) for i in range(3))

def _img_window_to_seg_slices(z0, z1, y0, y1, x0, x1, img_shape, seg_shape):
    rz, ry, rx = _axis_ratios(img_shape, seg_shape)
    sz0 = max(0, int(np.floor(z0 * rz)))
    sz1 = min(seg_shape[0], max(sz0 + 1, int(np.ceil(z1 * rz))))
    sy0 = max(0, int(np.floor(y0 * ry)))
    sy1 = min(seg_shape[1], max(sy0 + 1, int(np.ceil(y1 * ry))))
    sx0 = max(0, int(np.floor(x0 * rx)))
    sx1 = min(seg_shape[2], max(sx0 + 1, int(np.ceil(x1 * rx))))
    return sz0, sz1, sy0, sy1, sx0, sx1

def _resize_seg_to_match(seg_crop, target_shape):
    seg_crop = np.asarray(seg_crop)
    if seg_crop.shape == target_shape:
        return seg_crop
    try:
        from scipy.ndimage import zoom as _zoom

        factors = tuple(target_shape[i] / seg_crop.shape[i] for i in range(3))
        return _zoom(seg_crop, factors, order=0)
    except Exception:
        # Fallback nearest-neighbor resize when scipy is unavailable.
        idx = tuple(np.clip(np.round(np.linspace(0, seg_crop.shape[i] - 1, target_shape[i])).astype(int), 0, seg_crop.shape[i] - 1) for i in range(3))
        return seg_crop[np.ix_(idx[0], idx[1], idx[2])]

def _mito_anchor_in_img_space(seg_group, img_shape, seg_shape, ch_eff_img, vol_clip=None):
    clip_extent = vol_clip if vol_clip is not None else img_shape
    Dz_s, Dy_s, Dx_s = seg_shape
    rz, ry, rx = _axis_ratios(img_shape, seg_shape)
    nz = min(64, max(8, Dz_s // 16))
    ny = min(64, max(8, Dy_s // 16))
    nx = min(64, max(8, Dx_s // 16))
    step_z = max(1, Dz_s // nz)
    step_y = max(1, Dy_s // ny)
    step_x = max(1, Dx_s // nx)
    coarse = np.asarray(seg_group[0:Dz_s:step_z, 0:Dy_s:step_y, 0:Dx_s:step_x])
    if coarse.size == 0 or not (coarse != 0).any():
        return _center_offset(clip_extent, ch_eff_img)
    zz, yy, xx = np.nonzero(coarse != 0)
    cz_s = int(np.clip(np.mean(zz) * step_z + step_z // 2, 0, Dz_s - 1))
    cy_s = int(np.clip(np.mean(yy) * step_y + step_y // 2, 0, Dy_s - 1))
    cx_s = int(np.clip(np.mean(xx) * step_x + step_x // 2, 0, Dx_s - 1))
    cz = int(np.clip(cz_s / rz, 0, img_shape[0] - 1))
    cy = int(np.clip(cy_s / ry, 0, img_shape[1] - 1))
    cx = int(np.clip(cx_s / rx, 0, img_shape[2] - 1))
    out = []
    for c, d, ch in zip((cz, cy, cx), clip_extent, ch_eff_img):
        o = int(c - ch // 2)
        o = max(0, min(o, int(d) - int(ch)))
        out.append(o)
    return tuple(out)

def download_openorganelle_data(dataset_name, offsets, chunk_shape=(512, 512, 512), n_crops=1, window_policy="center", img_dir=None, seg_dir=None, foundation=True, voxel_size_nm_override=None):
    if dataset_name not in DATASETS:
        print(f"Error: dataset {dataset_name!r} not found")
        print(f"Available datasets: {list(DATASETS.keys())}")
        return

    if not img_dir or not seg_dir:
        raise SystemExit("Internal error: img_dir/seg_dir must be resolved before dataset loop.")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    cfg = DATASETS[dataset_name]
    img_path = cfg["img_path"]
    seg_path = cfg["seg_path"]
    voxel_size_nm = list(FORCED_OUTPUT_SPACING_NM_ZYX)
    tag = dataset_name.replace("-", "_")

    print(f"Dataset: {dataset_name}")
    print(f"Image path: {img_path}")
    print(f"Label path: {seg_path}")
    print(f"Output images: {os.path.abspath(img_dir)}")
    print(f"Output labels: {os.path.abspath(seg_dir)}")
    print("-" * 60)

    if foundation:
        creds = {"anon": True}
        image_group = _as_array(_try_read(img_path, creds))
        seg_group = _as_array(_try_read(seg_path, creds))
        img_shape = tuple(int(image_group.shape[d]) for d in range(3))
        seg_shape = tuple(int(seg_group.shape[d]) for d in range(3))
        vol_shape = _planning_vol_shape_em(cfg, img_path, img_shape)
        try:
            _dl_foundation_labeled(
                dataset_name=dataset_name,
                cfg=cfg,
                image_group=image_group,
                seg_group=seg_group,
                img_path=img_path,
                img_shape=img_shape,
                seg_shape=seg_shape,
                vol_shape=vol_shape,
                offsets=offsets,
                window_policy=window_policy,
                img_dir=img_dir,
                seg_dir=seg_dir,
                n_crops=max(1, int(n_crops)),
                out_spacing_nm_zyx=tuple(float(v) for v in voxel_size_nm),
                out_max_voxels_zyx=tuple(int(v) for v in chunk_shape),
                write_h5_fn=write_h5,
            )
        except Exception as exc:
            print(
                f"[ERROR] foundation download failed "
                f"(spacing={tuple(float(v) for v in voxel_size_nm)}, "
                f"max_voxels={tuple(int(v) for v in chunk_shape)}): {exc}"
            )
            raise
        return

    creds = {"anon": True}
    image_group = _as_array(_try_read(img_path, creds))
    seg_group = _as_array(_try_read(seg_path, creds))

    img_shape = tuple(int(image_group.shape[d]) for d in range(3))
    seg_shape = tuple(int(seg_group.shape[d]) for d in range(3))
    vol_shape = _planning_vol_shape_em(cfg, img_path, img_shape)
    rz, ry, rx = _axis_ratios(img_shape, seg_shape)
    print(f"[INFO] {dataset_name}: seg/EM index ratio (zyx) ≈ ({rz:.4f}, {ry:.4f}, {rx:.4f})")

    ch_eff = _effective_chunk(vol_shape, chunk_shape)
    if ch_eff != tuple(int(x) for x in chunk_shape):
        print(f"[INFO] {dataset_name}: chunk capped {chunk_shape} -> {ch_eff} for volume {vol_shape}")

    if offsets is not None:
        win_offsets = list(offsets)
    elif int(n_crops) > 1:
        win_offsets, possible = _grid_offsets_non_overlapping(vol_shape, ch_eff, int(n_crops))
        if len(win_offsets) < int(n_crops):
            print(f"[INFO] {dataset_name}: requested {n_crops} crops, max non-overlap fit={possible} for chunk {ch_eff}")
        print(f"[INFO] {dataset_name}: non-overlap crops selected={len(win_offsets)} chunk {ch_eff}")
    elif window_policy == "mito":
        win_offsets = [_mito_anchor_in_img_space(seg_group, img_shape, seg_shape, ch_eff, vol_shape)]
        print(f"[INFO] {dataset_name}: mito-guided crop offset {win_offsets[0]} chunk {ch_eff}")
    else:
        win_offsets = [_center_offset(vol_shape, ch_eff)]
        print(f"[INFO] {dataset_name}: center crop offset {win_offsets[0]} chunk {ch_eff}")

    _planned_suffixes = _resolve_batch_coherent_suffixes(
        dataset_name=dataset_name,
        img_dir=img_dir,
        seg_dir=seg_dir,
        n_windows=len(win_offsets),
    )
    for i, offset in enumerate(win_offsets, start=1):
        suffix = _crop_index_suffix(i, len(win_offsets))
        resolved_suffix = _planned_suffixes[i - 1] if i - 1 < len(_planned_suffixes) else suffix
        win = _window_from_offset(vol_shape, offset, ch_eff)
        if win is None:
            print(f"[WARN] {dataset_name} vol{i}: invalid window at offset={offset}; skipped")
            continue
        (z0, y0, x0), (z1, y1, x1) = win
        if (z0, y0, x0) != tuple(int(x) for x in offset):
            print(f"[INFO] {dataset_name} vol{i}: shifted offset {tuple(offset)} -> {(z0, y0, x0)} to fit volume")
        image = image_group[z0:z1, y0:y1, x0:x1]
        sz0, sz1, sy0, sy1, sx0, sx1 = _img_window_to_seg_slices(
            z0, z1, y0, y1, x0, x1, img_shape, seg_shape
        )
        seg_crop = seg_group[sz0:sz1, sy0:sy1, sx0:sx1]
        seg = _resize_seg_to_match(seg_crop, image.shape)
        img_prefix = os.path.join(img_dir, f"{tag}{resolved_suffix}")
        seg_prefix = os.path.join(seg_dir, f"{tag}{resolved_suffix}")
        write_h5(f"{img_prefix}_im.h5", image, dataset=f"{tag}{resolved_suffix}_im")
        write_h5(f"{seg_prefix}_seg.h5", seg, dataset=f"{tag}{resolved_suffix}_seg")

def _record_last_download_in_metadata_md():
    from pathlib import Path
    md = Path(__file__).resolve().parent / "download_openorganelle_labeled.md"
    now = datetime.now()
    lines = [
        "# OpenOrganelle — Download script metadata (labeled)",
        "",
        "**Source:** https://openorganelle.janelia.org/datasets",
        "**Script generated:** " + SCRIPT_GENERATED_DATE,
        "**Generated At (ISO):** " + SCRIPT_GENERATED_AT_ISO,
        "**Website Name:** OpenOrganelle",
        "**Downloader mode:** labeled — EM image + mitochondria segmentation",
        "**Companion script:** `download_openorganelle_labeled.py`",
        "",
        "**Last download:** " + now.strftime("%Y-%m-%d"),
        "**Last Download At (ISO):** " + now.replace(microsecond=0).isoformat(),
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _reg_imports():
    '''Return (open_registry, api_module) after ensuring repo root is on sys.path.'''
    from pathlib import Path as _P
    import sys as _s
    _r = _P(__file__).resolve().parents[2]
    if str(_r) not in _s.path:
        _s.path.insert(0, str(_r))
    from agent.orchestration.registry.schema import open_registry as _or
    import agent.orchestration.registry.api as _api
    return _or, _api

def _filesystem_check_labeled(dataset_name, training_root_str):
    '''Scan nnUNet ``imagesTr`` for existing EM stacks for this dataset.
    Returns the images/ directory path string if any output exists, else None.
    '''
    _tag = dataset_name.replace("-", "_")
    _img = Path(training_root_str) / "imagesTr"
    if not _img.is_dir():
        return None
    if list(_img.glob(f"{_tag}*_0000.nii.gz")):
        return str(_img)
    return None

def _filesystem_has_expected_labeled_outputs(dataset_name, target, base_root, required_n=1):
    '''True iff expected image+label outputs for dataset exist on disk for target split/count.'''
    return labeled_nnunet_vol_outputs_count_ge(
        dataset_name,
        nnunet_raw_root=base_root,
        split=str(target),
        min_pairs=max(1, int(required_n)),
    )


def _resolve_batch_coherent_suffixes(
    *,
    dataset_name: str,
    img_dir: str,
    seg_dir: str,
    n_windows: int,
) -> list[str]:
    # If any base _volN collides, apply one shared _M suffix to all windows.
    n = max(1, int(n_windows))
    tag = dataset_name.replace("-", "_")
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
            print(f"[INFO] {dataset_name}: batch collision detected; applying shared suffix _{m} to all {n} crop(s)")
            return trial
        m += 1

def _preflight_registry_check(targets, chunk_shape, voxel_size_nm, foundation):
    '''Return (pending, skipped). Registry-first, filesystem-fallback.

    For each dataset:
      1. Check registry plus per-split nnUNet stacks on disk.
      2. If incomplete, accept split-agnostic global volume pool when satisfied.
      3. Registry-only URL-churn / adoption steps as implemented by registry APIs.

    Falls back to (list(targets), []) only if registry is completely unavailable.
    '''
    try:
        _open_registry, _api = _reg_imports()
        _conn = _open_registry()
        _pid = _api.upsert_provider(_conn, name="OpenOrganelle",
                                     base_url="https://openorganelle.janelia.org")
        _conn.commit()
        _voxel = list(voxel_size_nm) if voxel_size_nm else [16.0, 16.0, 16.0]
        pending, skipped = [], []
        for _ds in targets:
            _train_n, _infer_n = clamped_train_infer(DATASET_SPLITS.get(_ds))
            if (_train_n + _infer_n) <= 0:
                skipped.append(_ds)
                continue
            _tag = _ds.replace("-", "_")
            _total_n = int(_train_n) + int(_infer_n)
            _pool_done = foundation_global_volume_pool_complete(
                DEFAULT_LABELED_BASE, _tag, _total_n
            )
            if _pool_done and global_crop_profile_completed(DEFAULT_LABELED_BASE, _tag, _total_n):
                skipped.append(_ds)
                continue
            _did = _api.get_dataset_id(_conn, _pid, _ds)
            _all_done = True
            for (_target_ds, _n_side) in (("training", _train_n), ("inference", _infer_n)):
                if _n_side <= 0:
                    continue
                _profile = _api.make_download_profile_hash(
                    n_crops=int(_n_side),
                    chunk_zyx=chunk_shape,
                    voxel_nm_zyx=tuple(float(v) for v in _voxel[:3]),
                    mode="labeled",
                    foundation=foundation,
                )
                _reg_done = (
                    _did is not None
                    and _api.is_dataset_type_downloaded_for_profile(_conn, _did, "em_volume", _profile)
                    and _api.is_dataset_type_downloaded_for_profile(_conn, _did, "mito_seg", _profile)
                )
                _fs_done = _filesystem_has_expected_labeled_outputs(
                    _ds, _target_ds, DEFAULT_LABELED_BASE, required_n=int(_n_side)
                )
                if not (_reg_done and _fs_done):
                    _all_done = False
                    break
            if (not _all_done) and _pool_done:
                mark_global_crop_profile_completed(DEFAULT_LABELED_BASE, _tag, _total_n)
                _all_done = True
            if _all_done:
                skipped.append(_ds)
            else:
                pending.append(_ds)
        _conn.close()
        return pending, skipped
    except Exception as _exc:
        print(f"[REGISTRY] Pre-flight check unavailable ({_exc}); proceeding with all datasets.")
        return list(targets), []

def _registry_record_start(dataset_name, img_url, seg_url, img_dir, seg_dir,
                            n_crops, chunk_shape, voxel_nm, foundation):
    '''Record download start in registry for em_volume and mito_seg; return (dl_ids, conn) or (None, None).'''
    try:
        _open_registry, _api = _reg_imports()
        _conn = _open_registry()
        _pid = _api.upsert_provider(_conn, name="OpenOrganelle",
                                     base_url="https://openorganelle.janelia.org")
        _conn.commit()
        _did = _api.get_dataset_id(_conn, _pid, dataset_name)
        if _did is None:
            _did = _api.upsert_dataset(_conn, provider_id=_pid, stable_id=dataset_name)
            _conn.commit()
        _voxel = list(voxel_nm) if voxel_nm else [16.0, 16.0, 16.0]
        _profile = _api.make_download_profile_hash(
            n_crops=n_crops,
            chunk_zyx=chunk_shape,
            voxel_nm_zyx=tuple(float(v) for v in _voxel[:3]),
            mode="labeled",
            foundation=foundation,
        )
        _em_id = _api.upsert_asset(_conn, dataset_id=_did, asset_type="em_volume",
                                    remote_url=img_url)
        _conn.commit()
        _dl_ids = []
        _em_dl_id = _api.record_download_start(_conn, _em_id, str(img_dir),
                                                download_profile_hash=_profile)
        _dl_ids.append(_em_dl_id)
        _conn.commit()
        if seg_url and seg_dir:
            _seg_id = _api.upsert_asset(_conn, dataset_id=_did, asset_type="mito_seg",
                                         remote_url=seg_url)
            _conn.commit()
            _seg_dl_id = _api.record_download_start(_conn, _seg_id, str(seg_dir),
                                                     download_profile_hash=_profile)
            _dl_ids.append(_seg_dl_id)
            _conn.commit()
        return _dl_ids, _conn
    except Exception as _exc:
        print(f"[REGISTRY] Warning: could not record start for {dataset_name}: {_exc}")
        return None, None

def _registry_record_complete(conn, dl_ids, dataset_name):
    '''Record download completion for all tracked assets and close connection.'''
    if conn is None or dl_ids is None:
        return
    try:
        _, _api = _reg_imports()
        for _dl_id in (dl_ids if isinstance(dl_ids, list) else [dl_ids]):
            _api.record_download_complete(conn, _dl_id)
        conn.commit()
        conn.close()
        print(f"[REGISTRY] Recorded: {dataset_name}")
    except Exception as _exc:
        print(f"[REGISTRY] Warning: could not record completion for {dataset_name}: {_exc}")

def main():
    parser = argparse.ArgumentParser(description="Download OpenOrganelle labeled data")
    parser.add_argument("--dataset", "-d", type=str, default=None, choices=list(DATASETS.keys()))
    parser.add_argument(
        "--window-policy",
        "-w",
        type=str,
        choices=["center", "fixed", "mito"],
        default=DEFAULT_WINDOW_POLICY,
        help="center=fit chunk+volume center (default); mito=coarse seg anchor (labeled); fixed=require --offsets",
    )
    parser.add_argument("--offsets", "-o", type=str, nargs="+", default=None, help="z,y,x per window (with --window-policy fixed, or overrides auto placement)")
    parser.add_argument("--chunk-shape", "-c", type=str, default=",".join(map(str, DEFAULT_CHUNK_SHAPE)))
    parser.add_argument("--n-crops", type=int, default=DEFAULT_N_CROPS, help="Requested number of non-overlapping crops per dataset")
    parser.add_argument(
        "--voxel-size-nm",
        type=str,
        default=",".join(map(str, DEFAULT_VOXEL_SIZE_NM)),
        help="Voxel spacing (z,y,x nm) used for legacy --no-foundation crops",
    )
    parser.add_argument("--img-dir", type=str, default=None, help="Optional override; must be paired with --seg-dir")
    parser.add_argument("--seg-dir", type=str, default=None, help="Optional override; must be paired with --img-dir")
    parser.add_argument(
        "--no-foundation",
        action="store_true",
        help="Legacy: raw crop using --chunk-shape without resampling to isotropic 512³ @ 16 nm",
    )
    args = parser.parse_args()

    if args.window_policy == "fixed" and not args.offsets:
        raise SystemExit("--window-policy fixed requires at least one --offsets z,y,x")
    if args.offsets:
        offsets_arg = [tuple(map(int, s.split(","))) for s in args.offsets]
    elif args.window_policy == "mito":
        offsets_arg = None
    else:
        offsets_arg = None

    chunk_shape = tuple(map(int, args.chunk_shape.split(",")))
    voxel_size_nm_override = [16.0, 16.0, 16.0]
    targets = [args.dataset] if args.dataset else sorted(DATASETS.keys())
    _voxel_nm_pf = [16.0, 16.0, 16.0]
    _pending_targets, _skipped_targets = _preflight_registry_check(
        targets, chunk_shape, _voxel_nm_pf, not args.no_foundation
    )
    _pairs_for = {}
    for _ds in targets:
        _tr_n, _ts_n = clamped_train_infer(DATASET_SPLITS.get(_ds))
        _pairs_for[_ds] = max(0, int(_tr_n) + int(_ts_n))
    _n_total = len(targets)
    _n_skipped = len(_skipped_targets)
    _n_pending = len(_pending_targets)
    nwin_plan = 0
    for _ds in _pending_targets:
        nwin_plan += int(_pairs_for.get(_ds, 0))
    print(f"[PLAN] {_n_pending} of {_n_total} dataset(s) need download; {_n_skipped} already complete.")
    if _n_skipped > 0:
        print(f"[SKIP] Already complete: {', '.join(_skipped_targets)}")
    if _n_pending == 0:
        total_pairs = max(0, int(sum(int(_pairs_for.get(_ds, 0)) for _ds in targets)))
        print(
            f"[NOOP] No new assets to download — all {total_pairs} crop pair(s) "
            "are already complete for this profile."
        )
        print("[NOOP] No output folder will be created. Re-run with different n_crops/chunk/voxel for a new profile.")
        return
    targets = _pending_targets
    pair_count = sum(int(_pairs_for.get(_ds, 0)) for _ds in targets)
    print("=" * 60)
    print("Labeled download summary")
    print(f"- Website: {WEBSITE_NAME}")
    print(f"- Total datasets available in this file: {len(DATASETS)}")
    print(f"- Data type: {DATA_TYPE}")
    print(
        f"- Foundation mode: {not args.no_foundation} "
        f"(≤128 voxels/axis @ 16 nm isotropic, extent-limited; use --no-foundation for legacy crop)"
    )
    print(f"- Datasets: {len(targets)} pending (of {_n_total} total)")
    print(f"- Windows per dataset: mixed per dataset (policy={args.window_policy})")
    print(f"- Planned image/label pairs: {pair_count}")
    print("- Output: NIfTI (.nii.gz) nnUNet naming")
    print("- Planned dataset downloads (dataset | image | label):")
    _targets_set = set(targets)
    for row in DATASET_DOWNLOAD_PLAN:
        if row['dataset_name'] in _targets_set:
            print(f"  - {row['dataset_name']} | img={row['img_path']} | seg={row['seg_path']}")
    print("=" * 60)
    batch_id = None
    if (not args.img_dir) and (not args.seg_dir):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_id = os.environ.get("MITO2_DOWNLOAD_BATCH_ID", "").strip() or f"openorganelle_{stamp}"
        tr = DEFAULT_LABELED_BASE
        inf = DEFAULT_INFERENCE_BASE
        training_img_dir = os.path.join(tr, "imagesTr")
        training_seg_dir = os.path.join(tr, "labelsTr")
        inference_img_dir = os.path.join(inf, "imagesTs")
        inference_seg_dir = os.path.join(inf, "labelsTs")
        os.makedirs(training_img_dir, exist_ok=True)
        os.makedirs(training_seg_dir, exist_ok=True)
        os.makedirs(inference_img_dir, exist_ok=True)
        os.makedirs(inference_seg_dir, exist_ok=True)
        img_dir = training_img_dir
        seg_dir = training_seg_dir
        print(f"[BATCH_ID] {batch_id}")
        try:
            _rp_plan = Path(__file__).resolve().parents[2]
            import sys as _sys_plan
            if str(_rp_plan) not in _sys_plan.path:
                _sys_plan.path.insert(0, str(_rp_plan))
            _inv = _rp_plan / "0inventory"
            if str(_inv) not in _sys_plan.path:
                _sys_plan.path.insert(0, str(_inv))
            from download_history import register_openorganelle_labeled_run_planned
            _planned_pairs = int(pair_count)
            register_openorganelle_labeled_run_planned(
                _rp_plan,
                batch_id=batch_id,
                planned_pair_count=_planned_pairs,
                chunk_shape=chunk_shape,
                n_crops=int(args.n_crops),
                voxel_nm=list(_voxel_nm_pf),
                foundation=not args.no_foundation,
                datasets_planned=len(targets),
                n_windows=0,
            )
            print(f"[REGISTRY] Planned pairs / Log (2×pairs): {_planned_pairs} / {_planned_pairs * 2}", flush=True)
        except Exception as _plan_exc:
            print(f"[WARN] registry run plan record failed: {_plan_exc}", flush=True)
    elif args.img_dir and args.seg_dir:
        img_dir = args.img_dir
        seg_dir = args.seg_dir
        training_img_dir = img_dir
        training_seg_dir = seg_dir
        inference_img_dir = img_dir
        inference_seg_dir = seg_dir
    else:
        raise SystemExit("Provide both --img-dir and --seg-dir, or provide neither.")

    total_targets = len(targets)
    success_targets = []
    success_targets_training = []
    success_targets_inference = []
    failed_targets = []
    for idx, ds in enumerate(targets, start=1):
        _cfg = DATASETS.get(ds, {})
        _train_n, _infer_n = clamped_train_infer(DATASET_SPLITS.get(ds))
        _total_n = _train_n + _infer_n
        print(f"[PROGRESS] dataset {idx}/{total_targets}: {ds} (pairs={_total_n})")
        if _train_n + _infer_n <= 0:
            print(f"[SKIP] dataset {idx}/{total_targets}: {ds} (training=0, inference=0)")
            continue
        try:
            _dl_ids = None
            _reg_conn = None
            _dl_ids_ts = None
            _reg_conn_ts = None
            if _train_n > 0:
                _dl_ids, _reg_conn = _registry_record_start(
                    ds,
                    (_cfg.get("img_path") or "").strip(),
                    (_cfg.get("seg_path") or "").strip(),
                    training_img_dir,
                    training_seg_dir,
                    _train_n, chunk_shape, _voxel_nm_pf,
                    not args.no_foundation,
                )
            if _infer_n > 0:
                _dl_ids_ts, _reg_conn_ts = _registry_record_start(
                    ds,
                    (_cfg.get("img_path") or "").strip(),
                    (_cfg.get("seg_path") or "").strip(),
                    inference_img_dir,
                    inference_seg_dir,
                    _infer_n, chunk_shape, _voxel_nm_pf,
                    not args.no_foundation,
                )
            download_openorganelle_data(
                dataset_name=ds,
                offsets=offsets_arg,
                chunk_shape=chunk_shape,
                n_crops=_total_n,
                window_policy=args.window_policy,
                img_dir=training_img_dir,
                seg_dir=training_seg_dir,
                foundation=not args.no_foundation,
                voxel_size_nm_override=voxel_size_nm_override,
            )
            if _infer_n > 0:
                _, _ts_indices = alternating_volume_indices(_train_n, _infer_n)
                _tag = ds.replace("-", "_")
                _train_inst_dir = Path(training_seg_dir).parent / "labelsTr-instance"
                _infer_inst_dir = Path(inference_seg_dir).parent / "labelsTs-instance"
                _infer_inst_dir.mkdir(parents=True, exist_ok=True)
                for _vidx in _ts_indices:
                    _img_name = f"{_tag}_vol{_vidx}_0000.nii.gz"
                    _lbl_name = f"{_tag}_vol{_vidx}.nii.gz"
                    _inst_name = f"{_tag}_vol{_vidx}_instance.nii.gz"
                    _src_img = Path(training_img_dir) / _img_name
                    _src_lbl = Path(training_seg_dir) / _lbl_name
                    _src_inst = _train_inst_dir / _inst_name
                    _dst_img = Path(inference_img_dir) / _img_name
                    _dst_lbl = Path(inference_seg_dir) / _lbl_name
                    _dst_inst = _infer_inst_dir / _inst_name
                    if _src_img.exists():
                        shutil.move(str(_src_img), str(_dst_img))
                    if _src_lbl.exists():
                        shutil.move(str(_src_lbl), str(_dst_lbl))
                    if _src_inst.exists():
                        shutil.move(str(_src_inst), str(_dst_inst))
            _registry_record_complete(_reg_conn, _dl_ids, ds)
            _registry_record_complete(_reg_conn_ts, _dl_ids_ts, ds)
            if _total_n > 0:
                _tag = ds.replace("-", "_")
                mark_global_crop_profile_completed(DEFAULT_LABELED_BASE, _tag, _total_n)
            success_targets.append(ds)
            if _train_n > 0:
                success_targets_training.append(ds)
            if _infer_n > 0:
                success_targets_inference.append(ds)
            print(f"[DONE] dataset {idx}/{total_targets}: {ds} (pairs={_total_n})")
        except Exception as exc:
            print(f"[ERROR] {ds}: {exc}")
            failed_targets.append(ds)

    print(
        f"[SUMMARY] success={len(success_targets)}  failed={len(failed_targets)}"
    )
    if failed_targets:
        print(f"  Failed: {', '.join(failed_targets)}")

    if batch_id is not None:
        import sys as _sfin
        _rp = Path(__file__).resolve().parents[2]
        if str(_rp) not in _sfin.path:
            _sfin.path.insert(0, str(_rp))
        _inv = _rp / "0inventory"
        if str(_inv) not in _sfin.path:
            _sfin.path.insert(0, str(_inv))
        from download_history import finalize_openorganelle_labeled_script_run
        finalize_openorganelle_labeled_script_run(
            _rp,
            batch_id=batch_id,
            successful_targets=success_targets_training,
            img_dir=Path(training_img_dir),
            seg_dir=Path(training_seg_dir),
            successful_targets_inference=success_targets_inference,
            inference_img_dir=Path(inference_img_dir),
            inference_seg_dir=Path(inference_seg_dir),
            chunk_shape=chunk_shape,
            n_crops=int(args.n_crops),
            voxel_nm=list(_voxel_nm_pf),
            foundation=not args.no_foundation,
            run_preprocess=False,
            n_windows=0,
            successful_pair_count_override=sum(int(_pairs_for.get(ds, 0)) for ds in success_targets),
        )

    _record_last_download_in_metadata_md()
    ds_json = _sync_dataset_json(DEFAULT_LABELED_BASE)
    print(f"[DONE] dataset.json synced: {ds_json}")

if __name__ == "__main__":
    main()
