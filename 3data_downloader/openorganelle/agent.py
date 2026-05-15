#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import warnings

import numpy as np
from datetime import datetime
from pathlib import Path

try:
    import h5py  # type: ignore
except Exception:
    h5py = None
# Keep terminal/UI logs focused on run output (suppress known third-party deprecation noise).
warnings.filterwarnings("ignore", category=FutureWarning, module=r"fibsem_tools\.io\.n5\.core")
try:
    from fibsem_tools.io import read  # type: ignore
except Exception:
    read = None

try:
    from openorganelle.foundation_resample import download_foundation_labeled as _dl_foundation_labeled
    from openorganelle.foundation_resample import download_foundation_unlabeled as _dl_foundation_unlabeled
except Exception:
    _dl_foundation_labeled = None
    _dl_foundation_unlabeled = None

# Repo root is mitoFoundation2/ (four levels up from this file).
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_INV = ROOT / "0inventory"
if str(_INV) not in sys.path:
    sys.path.insert(0, str(_INV))

_DB_BUILDER_OO = ROOT / "2database_builder" / "openorganelle"
if str(_DB_BUILDER_OO) not in sys.path:
    # Import Stage-2 inventory helper by module file path name to avoid clashing
    # with this package's own ``openorganelle`` module namespace.
    sys.path.insert(0, str(_DB_BUILDER_OO))
from labeled_inventory_resolved import labeled_mito_inventory_from_resolved_db  # noqa: E402

inventory_from_db = labeled_mito_inventory_from_resolved_db

from config.paths import GENERATED_DOWNLOADER_PATHS_BLOCK  # noqa: E402
from download_history import nnunet_dataset_root  # noqa: E402


DEFAULT_DB = ROOT / "2database_builder" / "outputs" / "databases" / "OpenOrganelle.db"
# When None: `--execute` / generated scripts use --window-policy (default center).
DEFAULT_OFFSETS: list[tuple[int, int, int]] | None = None
# Foundation mode uses ≤128/axis @ 16 nm in foundation_resample.py
DEFAULT_CHUNK = (128, 128, 128)
DEFAULT_FOUNDATION = True
DEFAULT_WINDOW_POLICY = "center"
DEFAULT_MODE = "labeled"
TRAINING_ROOT = nnunet_dataset_root(ROOT)
# Default download targets (no timestamped data/raw batch folder).
LABELED_BASE = TRAINING_ROOT
UNLABELED_BASE = TRAINING_ROOT
# Generated download_*.py live next to this package (README paths use 3data_downloader/outputs).
GENERATED_PY_BASE = ROOT / "3data_downloader" / "outputs"
if not GENERATED_PY_BASE.parent.is_dir():
    GENERATED_PY_BASE = ROOT / "data_downloader" / "outputs"
JANELIA_BUCKET_PREFIX = "s3://janelia-cosem-datasets"

# Injected into generated download_*.py (path = 3data_downloader/)
FOUNDATION_IMPORTS_LABELED = """
from pathlib import Path
import sys as _sys
# Generated scripts live under .../3data_downloader/outputs/ — package root is one level up.
_OG_ROOT = Path(__file__).resolve().parents[1]
if str(_OG_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_OG_ROOT))
from openorganelle.foundation_resample import download_foundation_labeled as _dl_foundation_labeled
"""

FOUNDATION_IMPORTS_UNLABELED = """
from pathlib import Path
import sys as _sys
_OG_ROOT = Path(__file__).resolve().parents[1]
if str(_OG_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_OG_ROOT))
from openorganelle.foundation_resample import download_foundation_unlabeled as _dl_foundation_unlabeled
"""

LABELED_TEMPLATE = """#!/usr/bin/env python3
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
warnings.filterwarnings("ignore", category=FutureWarning, module=r"fibsem_tools\\.io\\.n5\\.core")
from fibsem_tools.io import read
__FOUNDATION_IMPORTS__

TOTAL_DATASETS = __TOTAL_DATASETS__
print(f"[INFO] Labeled downloader loaded with {TOTAL_DATASETS} dataset(s).")
WEBSITE_NAME = "OpenOrganelle"
DATA_TYPE = "labeled (raw EM + mitochondria good labels, non-prediction)"

SCRIPT_GENERATED_DATE = "__GEN_DATE__"
SCRIPT_GENERATED_AT_ISO = "__GEN_ISO__"

# Dataset configuration (explicit source addresses)
DATASETS = __DATASETS__
DATASET_SPLITS = __DATASET_SPLITS__
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

__DOWNLOADER_PATHS_BLOCK__
DEFAULT_DATASET = "__DEFAULT_DATASET__"
DEFAULT_OFFSETS = None
DEFAULT_CHUNK_SHAPE = __DEFAULT_CHUNK_SHAPE__
DEFAULT_N_CROPS = __DEFAULT_N_CROPS__
DEFAULT_VOXEL_SIZE_NM = __DEFAULT_VOXEL_SIZE_NM__
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
    found = list(re.finditer(r"/s(\\d+)(?:/|$)", url))
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
    md.write_text("\\n".join(lines) + "\\n", encoding="utf-8")

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
"""

UNLABELED_TEMPLATE = """#!/usr/bin/env python3
import ast
import json
import re
import numpy as np
import h5py
import argparse
import os
import warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore", category=FutureWarning, module=r"fibsem_tools\\.io\\.n5\\.core")
from fibsem_tools.io import read
__FOUNDATION_IMPORTS__

TOTAL_DATASETS = __TOTAL_DATASETS__
print(f"[INFO] Unlabeled downloader loaded with {TOTAL_DATASETS} dataset(s).")
WEBSITE_NAME = "OpenOrganelle"
DATA_TYPE = "unlabeled (raw EM only)"

SCRIPT_GENERATED_DATE = "__GEN_DATE__"
SCRIPT_GENERATED_AT_ISO = "__GEN_ISO__"

# Dataset configuration (explicit source addresses)
DATASETS = __DATASETS__
DATASET_DOWNLOAD_PLAN = [
    {
        "dataset_name": name,
        "img_path": (cfg.get("img_path") or "").strip(),
    }
    for name, cfg in sorted(DATASETS.items())
]
DEFAULT_UNLABELED_BASE = "__DEFAULT_UNLABELED_BASE__"
DEFAULT_RUN_PREFIX = "openorganelle_all"
DEFAULT_OFFSETS = None
DEFAULT_CHUNK_SHAPE = __DEFAULT_CHUNK_SHAPE__
DEFAULT_N_CROPS = __DEFAULT_N_CROPS__
DEFAULT_VOXEL_SIZE_NM = __DEFAULT_VOXEL_SIZE_NM__
DEFAULT_FOUNDATION = True
DEFAULT_WINDOW_POLICY = "center"

def write_h5(filename, data, dataset="main"):
    with h5py.File(filename, "w") as f:
        ds = f.create_dataset(dataset, data.shape, compression="lzf", dtype=data.dtype)
        ds[:] = data

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
    found = list(re.finditer(r"/s(\\d+)(?:/|$)", url))
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

def _crop_index_suffix(i: int, total: int) -> str:
    # Keep crop suffix style aligned across providers: ``_vol1``, ``_vol2``, ...
    idx = max(1, int(i))
    if int(total) > 0:
        idx = min(idx, int(total))
    return f"_vol{idx}"


def download_openorganelle_unlabeled_data(dataset_name, offsets, chunk_shape=(512, 512, 512), n_crops=1, window_policy="center", images_dir=None, foundation=True, voxel_size_nm_override=None, physical_size_nm_override=None):
    if dataset_name not in DATASETS:
        print(f"Error: dataset {dataset_name!r} not found")
        print(f"Available datasets: {list(DATASETS.keys())}")
        return
    if not images_dir:
        raise SystemExit("Internal error: images_dir must be resolved before dataset loop.")
    os.makedirs(images_dir, exist_ok=True)

    cfg = DATASETS[dataset_name]
    img_path = cfg["img_path"]
    voxel_size_nm = voxel_size_nm_override or cfg.get("voxel_size") or cfg.get("voxel_size_nm") or [8.0, 8.0, 8.0]
    tag = dataset_name.replace("-", "_")

    print(f"Dataset: {dataset_name}")
    print(f"Image path: {img_path}")
    print(f"Output images: {os.path.abspath(images_dir)}")
    print("-" * 60)

    creds = {"anon": True}
    image_group = _as_array(_try_read(img_path, creds))

    img_shape = tuple(int(image_group.shape[d]) for d in range(3))
    vol_shape = _planning_vol_shape_em(cfg, img_path, img_shape)

    if foundation:
        try:
            _dl_foundation_unlabeled(
                dataset_name=dataset_name,
                cfg=cfg,
                image_group=image_group,
                img_path=img_path,
                img_shape=img_shape,
                vol_shape=vol_shape,
                offsets=offsets,
                window_policy=window_policy,
                images_dir=images_dir,
                n_crops=int(n_crops),
                out_spacing_nm_zyx=tuple(float(v) for v in voxel_size_nm),
                out_max_voxels_zyx=tuple(int(v) for v in chunk_shape),
                physical_size_nm_zyx=tuple(float(v) for v in physical_size_nm_override) if physical_size_nm_override is not None else None,
                write_h5_fn=write_h5,
            )
        except Exception as exc:
            print(
                f"[ERROR] foundation download failed "
                f"(spacing={tuple(float(v) for v in voxel_size_nm)}, "
                f"max_voxels={tuple(int(v) for v in chunk_shape)}, "
                f"physical_nm={tuple(float(v) for v in physical_size_nm_override) if physical_size_nm_override is not None else 'auto'}): {exc}"
            )
            raise
        return

    ch_eff = _effective_chunk(vol_shape, chunk_shape)
    if ch_eff != tuple(int(x) for x in chunk_shape):
        print(f"[INFO] {dataset_name}: chunk capped {chunk_shape} -> {ch_eff} for volume {vol_shape}")

    if window_policy == "mito":
        print(f"[WARN] {dataset_name}: --window-policy mito applies to labeled only; using center")
    if offsets is not None:
        win_offsets = list(offsets)
    elif int(n_crops) > 1:
        win_offsets, possible = _grid_offsets_non_overlapping(vol_shape, ch_eff, int(n_crops))
        if len(win_offsets) < int(n_crops):
            print(f"[INFO] {dataset_name}: requested {n_crops} crops, max non-overlap fit={possible} for chunk {ch_eff}")
        print(f"[INFO] {dataset_name}: non-overlap crops selected={len(win_offsets)} chunk {ch_eff}")
    else:
        win_offsets = [_center_offset(vol_shape, ch_eff)]
        print(f"[INFO] {dataset_name}: center crop offset {win_offsets[0]} chunk {ch_eff}")

    for i, offset in enumerate(win_offsets, start=1):
        suffix = _crop_index_suffix(i, len(win_offsets))
        resolved_suffix = suffix
        win = _window_from_offset(vol_shape, offset, ch_eff)
        if win is None:
            print(f"[WARN] {dataset_name} vol{i}: invalid window at offset={offset}; skipped")
            continue
        (z0, y0, x0), (z1, y1, x1) = win
        if (z0, y0, x0) != tuple(int(x) for x in offset):
            print(f"[INFO] {dataset_name} vol{i}: shifted offset {tuple(offset)} -> {(z0, y0, x0)} to fit volume")
        image = image_group[z0:z1, y0:y1, x0:x1]
        suffix = _crop_index_suffix(i, len(win_offsets))
        prefix = os.path.join(images_dir, f"{tag}{suffix}")
        write_h5(f"{prefix}_im.h5", image, dataset=f"{tag}{suffix}_im")

def _record_last_download_in_metadata_md():
    from pathlib import Path
    md = Path(__file__).resolve().parent / "download_openorganelle_unlabeled.md"
    now = datetime.now()
    lines = [
        "# OpenOrganelle — Download script metadata (unlabeled)",
        "",
        "**Source:** https://openorganelle.janelia.org/datasets",
        "**Script generated:** " + SCRIPT_GENERATED_DATE,
        "**Generated At (ISO):** " + SCRIPT_GENERATED_AT_ISO,
        "**Website Name:** OpenOrganelle",
        "**Downloader mode:** unlabeled — EM image only",
        "**Companion script:** `download_openorganelle_unlabeled.py`",
        "",
        "**Last download:** " + now.strftime("%Y-%m-%d"),
        "**Last Download At (ISO):** " + now.replace(microsecond=0).isoformat(),
    ]
    md.write_text("\\n".join(lines) + "\\n", encoding="utf-8")

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

def _filesystem_check_unlabeled(dataset_name, training_root_str):
    '''Scan unlabeled cache folder for existing unlabeled EM stacks.'''
    _tag = dataset_name.replace("-", "_")
    _img = Path(training_root_str) / "unlabeled" / "images"
    if not _img.is_dir():
        return None
    if list(_img.glob(f"{_tag}*_im.h5")):
        return str(_img)
    return None

def _preflight_registry_check_unlabeled(targets, n_crops, chunk_shape, voxel_size_nm, foundation):
    '''Return (pending, skipped) for unlabeled EM-only assets.
    Registry-first, filesystem-fallback. URL-churn-safe.

    Falls back to (list(targets), []) if the registry is unavailable.
    '''
    try:
        _open_registry, _api = _reg_imports()
        _conn = _open_registry()
        _pid = _api.upsert_provider(_conn, name="OpenOrganelle",
                                     base_url="https://openorganelle.janelia.org")
        _conn.commit()
        _voxel = list(voxel_size_nm) if voxel_size_nm else [16.0, 16.0, 16.0]
        _profile = _api.make_download_profile_hash(
            n_crops=n_crops,
            chunk_zyx=chunk_shape,
            voxel_nm_zyx=tuple(float(v) for v in _voxel[:3]),
            mode="unlabeled",
            foundation=foundation,
        )
        pending, skipped = [], []
        for _ds in targets:
            _did = _api.get_dataset_id(_conn, _pid, _ds)
            # 1. Registry check
            if _did is not None and _api.is_dataset_type_downloaded_for_profile(
                _conn, _did, "em_volume", _profile
            ):
                skipped.append(_ds)
                continue
            # 2. Filesystem fallback
            _found_img_dir = _filesystem_check_unlabeled(_ds, DEFAULT_UNLABELED_BASE)
            if _found_img_dir:
                try:
                    _cfg = DATASETS.get(_ds, {})
                    _img_url = (_cfg.get("img_path") or "").strip()
                    if _did is None:
                        _did = _api.upsert_dataset(_conn, provider_id=_pid, stable_id=_ds)
                        _conn.commit()
                    _em_id = _api.upsert_asset(_conn, dataset_id=_did,
                                               asset_type="em_volume", remote_url=_img_url)
                    _conn.commit()
                    _dl_id = _api.record_download_start(
                        _conn, _em_id, _found_img_dir, download_profile_hash=_profile
                    )
                    _api.record_download_complete(_conn, _dl_id)
                    _conn.commit()
                    print(f"[REGISTRY] Adopted existing download: {_ds} → {_found_img_dir}")
                except Exception as _ae:
                    print(f"[REGISTRY] Warning: could not adopt {_ds}: {_ae}")
                skipped.append(_ds)
            else:
                pending.append(_ds)
        _conn.close()
        return pending, skipped
    except Exception as _exc:
        print(f"[REGISTRY] Pre-flight check unavailable ({_exc}); proceeding with all datasets.")
        return list(targets), []

def _registry_record_start_unlabeled(dataset_name, img_url, images_dir,
                                      n_crops, chunk_shape, voxel_nm, foundation):
    '''Record unlabeled download start in registry; return (dl_id, conn) or (None, None).'''
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
            mode="unlabeled",
            foundation=foundation,
        )
        _em_id = _api.upsert_asset(_conn, dataset_id=_did, asset_type="em_volume",
                                    remote_url=img_url)
        _conn.commit()
        _dl_id = _api.record_download_start(_conn, _em_id, str(images_dir),
                                             download_profile_hash=_profile)
        _conn.commit()
        return _dl_id, _conn
    except Exception as _exc:
        print(f"[REGISTRY] Warning: could not record start for {dataset_name}: {_exc}")
        return None, None

def _registry_record_complete(conn, dl_id, dataset_name):
    '''Record download completion and close connection.'''
    if conn is None or dl_id is None:
        return
    try:
        _, _api = _reg_imports()
        _api.record_download_complete(conn, dl_id)
        conn.commit()
        conn.close()
        print(f"[REGISTRY] Recorded: {dataset_name}")
    except Exception as _exc:
        print(f"[REGISTRY] Warning: could not record completion for {dataset_name}: {_exc}")

def main():
    parser = argparse.ArgumentParser(description="Download OpenOrganelle unlabeled data")
    parser.add_argument("--dataset", "-d", type=str, default=None, choices=list(DATASETS.keys()))
    parser.add_argument(
        "--window-policy",
        "-w",
        type=str,
        choices=["center", "fixed", "mito"],
        default=DEFAULT_WINDOW_POLICY,
        help="center=fit chunk+volume center (default); fixed=require --offsets; mito=not used (falls back to center)",
    )
    parser.add_argument("--offsets", "-o", type=str, nargs="+", default=None, help="z,y,x per window (required for fixed)")
    parser.add_argument("--chunk-shape", "-c", type=str, default=",".join(map(str, DEFAULT_CHUNK_SHAPE)))
    parser.add_argument("--n-crops", type=int, default=DEFAULT_N_CROPS, help="Requested number of non-overlapping crops per dataset")
    parser.add_argument(
        "--voxel-size-nm",
        type=str,
        default=",".join(map(str, DEFAULT_VOXEL_SIZE_NM)),
        help="Target output spacing (z,y,x nm) in foundation mode",
    )
    parser.add_argument("--physical-size-nm", type=str, default=None, help="Physical crop size in nm as z,y,x before resampling; default is voxel-size-nm * chunk-shape per axis")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional override for the run folder (will use <output-dir>/images)")
    parser.add_argument(
        "--no-foundation",
        action="store_true",
        help="Legacy: raw crop using --chunk-shape without spacing-aware foundation resampling",
    )
    args = parser.parse_args()

    if args.window_policy == "fixed" and not args.offsets:
        raise SystemExit("--window-policy fixed requires at least one --offsets z,y,x")
    if args.offsets:
        offsets_arg = [tuple(map(int, s.split(","))) for s in args.offsets]
    else:
        offsets_arg = None

    chunk_shape = tuple(map(int, args.chunk_shape.split(",")))
    voxel_size_nm_override = [float(x) for x in args.voxel_size_nm.split(",")] if args.voxel_size_nm else None
    physical_size_nm_override = [float(x) for x in args.physical_size_nm.split(",")] if args.physical_size_nm else None
    if len(chunk_shape) != 3:
        raise SystemExit("--chunk-shape must be z,y,x")
    if voxel_size_nm_override is not None and len(voxel_size_nm_override) != 3:
        raise SystemExit("--voxel-size-nm must be z,y,x")
    if physical_size_nm_override is not None and len(physical_size_nm_override) != 3:
        raise SystemExit("--physical-size-nm must be z,y,x")
    if any(int(v) <= 0 for v in chunk_shape):
        raise SystemExit("--chunk-shape values must be positive")
    if voxel_size_nm_override is not None and any(float(v) <= 0 for v in voxel_size_nm_override):
        raise SystemExit("--voxel-size-nm values must be positive")
    if physical_size_nm_override is not None and any(float(v) <= 0 for v in physical_size_nm_override):
        raise SystemExit("--physical-size-nm values must be positive")
    targets = [args.dataset] if args.dataset else sorted(DATASETS.keys())
    _voxel_nm_pf = list(voxel_size_nm_override) if voxel_size_nm_override else list(DEFAULT_VOXEL_SIZE_NM)
    _pending_targets, _skipped_targets = _preflight_registry_check_unlabeled(
        targets, int(args.n_crops), chunk_shape, _voxel_nm_pf, not args.no_foundation
    )
    _n_total = len(targets)
    _n_skipped = len(_skipped_targets)
    _n_pending = len(_pending_targets)
    nwin_plan = len(offsets_arg) if offsets_arg is not None else max(1, int(args.n_crops))
    print(f"[PLAN] {_n_pending} of {_n_total} dataset(s) need download; {_n_skipped} already complete.")
    if _n_skipped > 0:
        print(f"[SKIP] Already complete: {', '.join(_skipped_targets)}")
    if _n_pending == 0:
        total_pairs = max(0, int(_n_total) * int(nwin_plan))
        print(
            f"[NOOP] No new assets to download — all {total_pairs} crop pair(s) "
            "are already complete for this profile."
        )
        print("[NOOP] No output folder will be created. Re-run with different n_crops/chunk/voxel for a new profile.")
        return
    targets = _pending_targets
    nwin = len(offsets_arg) if offsets_arg is not None else max(1, int(args.n_crops))
    pair_count = len(targets) * nwin
    print("=" * 60)
    print("OpenOrganelle unlabeled download summary")
    print(f"- Website: {WEBSITE_NAME}")
    print(f"- Total datasets available in this file: {len(DATASETS)}")
    print(f"- Data type: {DATA_TYPE}")
    print(
        f"- Foundation mode: {not args.no_foundation} "
        f"(target spacing from --voxel-size-nm, max voxels from --chunk-shape, "
        f"physical crop from --physical-size-nm or derived spacing*shape; "
        f"use --no-foundation for legacy crop)"
    )
    print(f"- Datasets: {len(targets)} pending (of {_n_total} total)")
    print(f"- Windows per dataset: {nwin} (policy={args.window_policy})")
    print(f"- Planned image chunks: {pair_count}")
    print("- Output: NIfTI (.nii.gz) nnUNet naming")
    print(f"- Requested output max voxels (z,y,x): {chunk_shape}")
    print(f"- Requested output spacing nm (z,y,x): {tuple(float(v) for v in voxel_size_nm_override)}")
    if physical_size_nm_override is None:
        derived_phys = tuple(float(voxel_size_nm_override[i]) * float(chunk_shape[i]) for i in range(3))
        print(f"- Requested physical crop size nm (z,y,x): auto={derived_phys}")
    else:
        print(f"- Requested physical crop size nm (z,y,x): {tuple(float(v) for v in physical_size_nm_override)}")
    print("- Planned dataset downloads (dataset | image):")
    _targets_set = set(targets)
    for row in DATASET_DOWNLOAD_PLAN:
        if row['dataset_name'] in _targets_set:
            print(f"  - {row['dataset_name']} | img={row['img_path']}")
    print("=" * 60)

    if args.output_dir:
        run_dir = args.output_dir
    else:
        run_dir = os.path.join(DEFAULT_UNLABELED_BASE, "unlabeled")
    images_dir = os.path.join(run_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    print(f"[INFO] Unlabeled output directory: {images_dir}")

    total_targets = len(targets)
    for idx, ds in enumerate(targets, start=1):
        print(f"[PROGRESS] dataset {idx}/{total_targets}: {ds}")
        _cfg = DATASETS.get(ds, {})
        _dl_id, _reg_conn = _registry_record_start_unlabeled(
            ds,
            (_cfg.get("img_path") or "").strip(),
            images_dir,
            int(args.n_crops), chunk_shape, _voxel_nm_pf,
            not args.no_foundation,
        )
        download_openorganelle_unlabeled_data(
            dataset_name=ds,
            offsets=offsets_arg,
            chunk_shape=chunk_shape,
            n_crops=int(args.n_crops),
            window_policy=args.window_policy,
            images_dir=images_dir,
            foundation=not args.no_foundation,
            voxel_size_nm_override=voxel_size_nm_override,
            physical_size_nm_override=physical_size_nm_override,
        )
        _registry_record_complete(_reg_conn, _dl_id, ds)
        print(f"[DONE] dataset {idx}/{total_targets}: {ds}")

    _record_last_download_in_metadata_md()

if __name__ == "__main__":
    main()
"""


def write_h5(filename: Path | str, data, dataset: str = "main") -> None:
    """Legacy hook name for foundation_resample; delegates to ``downloader_common.nnunet_labeled_export``."""
    from downloader_common.nnunet_labeled_export import write_h5 as _write_nnunet

    _write_nnunet(str(filename), data, dataset)


def _foundation_write_h5_str(path: str, data, dataset: str = "main") -> None:
    write_h5(path, data, dataset)


def _sync_dataset_json(dataset_root: Path | str) -> Path:
    from downloader_common.nnunet_labeled_export import sync_dataset_json

    return sync_dataset_json(dataset_root)


def _try_read(path: str, creds: dict):
    if read is None:
        raise RuntimeError("fibsem_tools is required to read OpenOrganelle paths")
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
    raise RuntimeError(f"Could not open {path}")


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


def _parse_offset_strings(raw_offsets: list[str]) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for s in raw_offsets:
        try:
            z, y, x = map(int, s.split(","))
            out.append((z, y, x))
        except ValueError as exc:
            raise SystemExit(f"Bad offset {s!r} expected z,y,x") from exc
    return out


def _execute_offsets(
    raw_offsets: list[str] | None, window_policy: str
) -> list[tuple[int, int, int]] | None:
    if raw_offsets:
        return _parse_offset_strings(raw_offsets)
    if window_policy == "fixed":
        raise SystemExit("--window-policy fixed requires --offsets z,y,x [...]")
    return None


def _parse_chunk(raw: str) -> tuple[int, int, int]:
    try:
        z, y, x = map(int, raw.split(","))
    except ValueError as exc:
        raise SystemExit(f"Bad chunk shape {raw!r} expected z,y,x") from exc
    return (z, y, x)


def _parse_voxel_size_nm(raw: str) -> tuple[float, float, float]:
    try:
        z, y, x = (float(v) for v in raw.split(","))
    except ValueError as exc:
        raise SystemExit(f"Bad voxel size {raw!r} expected z,y,x") from exc
    return (z, y, x)


def _effective_chunk(
    vol_shape: tuple[int, int, int], chunk_shape: tuple[int, int, int]
) -> tuple[int, int, int]:
    return tuple(min(int(c), int(s)) for c, s in zip(chunk_shape, vol_shape))


def _center_offset(
    vol_shape: tuple[int, int, int], ch_eff: tuple[int, int, int]
) -> tuple[int, int, int]:
    return tuple(max(0, (s - c) // 2) for s, c in zip(vol_shape, ch_eff))


def _window_from_offset(
    vol_shape: tuple[int, int, int],
    offset: tuple[int, int, int],
    ch_eff: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    starts: list[int] = []
    ends: list[int] = []
    for dim, off, c in zip(vol_shape, offset, ch_eff):
        d, o, ch = int(dim), int(off), int(c)
        if d <= 0 or ch <= 0:
            return None
        s = max(0, min(o, d - ch))
        e = s + ch
        starts.append(s)
        ends.append(e)
    return (tuple(starts), tuple(ends))


def _grid_offsets_non_overlapping(
    vol_shape: tuple[int, int, int],
    ch_eff: tuple[int, int, int],
    n_crops: int,
) -> tuple[list[tuple[int, int, int]], int]:
    if n_crops <= 1:
        return ([_center_offset(vol_shape, ch_eff)], 1)
    axes: list[list[int]] = []
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
    offsets.sort(
        key=lambda o: (o[0] - center[0]) ** 2 + (o[1] - center[1]) ** 2 + (o[2] - center[2]) ** 2
    )
    total_possible = len(offsets)
    return (offsets[: min(int(n_crops), total_possible)], total_possible)


def _axis_ratios(
    img_shape: tuple[int, int, int], seg_shape: tuple[int, int, int]
) -> tuple[float, float, float]:
    return tuple(seg_shape[i] / float(img_shape[i]) for i in range(3))


def _img_window_to_seg_slices(
    z0: int,
    z1: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    img_shape: tuple[int, int, int],
    seg_shape: tuple[int, int, int],
) -> tuple[int, int, int, int, int, int]:
    rz, ry, rx = _axis_ratios(img_shape, seg_shape)
    sz0 = max(0, int(np.floor(z0 * rz)))
    sz1 = min(seg_shape[0], max(sz0 + 1, int(np.ceil(z1 * rz))))
    sy0 = max(0, int(np.floor(y0 * ry)))
    sy1 = min(seg_shape[1], max(sy0 + 1, int(np.ceil(y1 * ry))))
    sx0 = max(0, int(np.floor(x0 * rx)))
    sx1 = min(seg_shape[2], max(sx0 + 1, int(np.ceil(x1 * rx))))
    return sz0, sz1, sy0, sy1, sx0, sx1


def _resize_seg_to_match(seg_crop: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
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


def _mito_anchor_in_img_space(
    seg_group,
    img_shape: tuple[int, int, int],
    seg_shape: tuple[int, int, int],
    ch_eff_img: tuple[int, int, int],
    *,
    vol_clip: tuple[int, int, int] | None = None,
) -> tuple[int, int, int]:
    Dz_s, Dy_s, Dx_s = seg_shape
    rz, ry, rx = _axis_ratios(img_shape, seg_shape)
    nz = min(64, max(8, Dz_s // 16))
    ny = min(64, max(8, Dy_s // 16))
    nx = min(64, max(8, Dx_s // 16))
    step_z = max(1, Dz_s // nz)
    step_y = max(1, Dy_s // ny)
    step_x = max(1, Dx_s // nx)
    coarse = np.asarray(seg_group[0:Dz_s:step_z, 0:Dy_s:step_y, 0:Dx_s:step_x])
    clip_extent = vol_clip if vol_clip is not None else img_shape
    if coarse.size == 0 or not (coarse != 0).any():
        return _center_offset(clip_extent, ch_eff_img)
    zz, yy, xx = np.nonzero(coarse != 0)
    cz_s = int(np.clip(np.mean(zz) * step_z + step_z // 2, 0, Dz_s - 1))
    cy_s = int(np.clip(np.mean(yy) * step_y + step_y // 2, 0, Dy_s - 1))
    cx_s = int(np.clip(np.mean(xx) * step_x + step_x // 2, 0, Dx_s - 1))
    cz = int(np.clip(cz_s / rz, 0, img_shape[0] - 1))
    cy = int(np.clip(cy_s / ry, 0, img_shape[1] - 1))
    cx = int(np.clip(cx_s / rx, 0, img_shape[2] - 1))
    out: list[int] = []
    for c, d, ch in zip((cz, cy, cx), clip_extent, ch_eff_img):
        o = int(c - ch // 2)
        o = max(0, min(o, int(d) - int(ch)))
        out.append(o)
    return tuple(out)


def _select_inventory(db_path: Path, mode: str) -> dict[str, dict]:
    if mode == "labeled":
        return inventory_from_db(db_path, mito_only=True, require_paths=True)
    return _select_unlabeled_inventory_all(db_path)


def _parse_num_vector(text: str | None) -> list[float] | None:
    if not text or not str(text).strip():
        return None
    try:
        v = ast.literal_eval(text.strip())
    except (SyntaxError, ValueError):
        try:
            v = json.loads(text.strip())
        except Exception:
            return None
    if isinstance(v, (list, tuple)) and v and all(isinstance(x, (int, float)) for x in v):
        return [float(x) for x in v]
    return None


def _catalog_dims_zyx(
    grid_dimensions: str | None, grid_axes: str | None
) -> tuple[int, int, int] | None:
    """
    Map scraped/DB ``grid.grid_dimensions`` + ``grid.grid_axes`` (x,y,z voxel counts)
    to fibsem_tools array order (z, y, x).
    """
    nums_f = _parse_num_vector(grid_dimensions)
    if not nums_f or len(nums_f) != 3:
        return None
    nums = [int(round(x)) for x in nums_f]
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


def _fibsem_pyramid_level(url: str) -> int:
    """Parse trailing ``/sN`` in EM/seg paths (e.g. ``.../fibsem-uint8/s1`` → 1). Default 0."""
    found = list(re.finditer(r"/s(\d+)(?:/|$)", url))
    if not found:
        return 0
    return int(found[-1].group(1))


def _expected_em_vol_shape_zyx(cfg: dict, img_path: str) -> tuple[int, int, int] | None:
    cz_yx = _catalog_dims_zyx(cfg.get("grid_dimensions"), cfg.get("grid_axes"))
    if cz_yx is None:
        return None
    cz, cy, cx = cz_yx
    level = _fibsem_pyramid_level(img_path)
    d = 2**level
    return (max(1, cz // d), max(1, cy // d), max(1, cx // d))


def _planning_vol_shape_em(
    cfg: dict, img_path: str, img_shape: tuple[int, int, int]
) -> tuple[int, int, int]:
    """
    Crop planning extent: catalog EM size at this pyramid level, intersected with the
    opened array (same semantics as OpenOrganelle.md grid + ``/sN`` in resolved paths).
    """
    ce = _expected_em_vol_shape_zyx(cfg, img_path)
    if ce is None:
        return img_shape
    inter = tuple(min(int(ce[i]), int(img_shape[i])) for i in range(3))
    tol = 2
    if any(abs(int(ce[i]) - int(img_shape[i])) > tol for i in range(3)):
        print(
            f"[WARN] catalog EM shape {ce} vs opened {img_shape}; "
            f"planning intersection {inter}"
        )
    elif inter != img_shape:
        print(f"[INFO] planning volume capped (catalog∩opened): {inter} (opened {img_shape})")
    else:
        print(f"[INFO] catalog EM shape {ce} matches opened volume {img_shape}")
    return inter


def _select_unlabeled_inventory_all(db_path: Path) -> dict[str, dict]:
    """
    Build unlabeled inventory from ALL datasets, regardless of mitochondria flag.
    Prefer s3_probe_img_path, then download_volume_url, then a template fallback.
    """
    import sqlite3
    import re

    slug_re = re.compile(r"^[a-z][a-z0-9_\\-]+$")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT dataset_name, download_volume_url, s3_probe_img_path,
               grid_spacing, s3_probe_voxel_size, grid_dimensions, grid_axes
        FROM datasets
        ORDER BY dataset_name
        """
    ).fetchall()
    conn.close()

    out: dict[str, dict] = {}
    for r in rows:
        name = (r["dataset_name"] or "").strip()
        if not name:
            continue
        img_probe = (r["s3_probe_img_path"] or "").strip()
        img_scraped = (r["download_volume_url"] or "").strip()
        if img_probe:
            img = img_probe
            src = "s3_probe"
        elif img_scraped:
            img = img_scraped
            src = "scraped_postgrest"
        elif slug_re.match(name):
            img = f"{JANELIA_BUCKET_PREFIX}/{name}/{name}.n5/em/fibsem-uint8/s1"
            src = "template_fallback"
        else:
            continue

        voxel = _parse_num_vector(r["s3_probe_voxel_size"]) or _parse_num_vector(r["grid_spacing"]) or [8.0, 8.0, 8.0]
        row = {
            "dataset_name": name,
            "img_path": img,
            "voxel_size_nm": voxel,
            "_source": src,
        }
        gd = (r["grid_dimensions"] or "").strip() if r["grid_dimensions"] else ""
        ga = (r["grid_axes"] or "").strip() if r["grid_axes"] else ""
        if gd:
            row["grid_dimensions"] = r["grid_dimensions"]
        if ga:
            row["grid_axes"] = r["grid_axes"]
        out[name] = row
    return out


def _write_sidecar_md_on_generate(
    md_path: Path,
    mode: str,
    gen_date: str,
    gen_iso: str,
    *,
    chunk_shape: tuple[int, int, int],
    n_crops: int,
    voxel_size_nm: tuple[float, float, float],
) -> None:
    """Write companion .md next to generated download_*.py (OpenOrganelle.md-style header)."""
    last_d, last_iso = "Not yet run", "—"
    if md_path.is_file():
        text = md_path.read_text(encoding="utf-8")
        m1 = re.search(r"\*\*Last download:\*\*\s*(.+)", text)
        m2 = re.search(r"\*\*Last Download At \(ISO\):\*\*\s*(.+)", text)
        if m1 and m2:
            a, b = m1.group(1).strip(), m2.group(1).strip()
            if a and b and a not in ("Not yet run", "—"):
                last_d, last_iso = a, b
    stem = f"download_openorganelle_{mode}"
    mode_line = (
        "labeled — EM image + mitochondria segmentation"
        if mode == "labeled"
        else "unlabeled — EM image only"
    )
    title = "labeled" if mode == "labeled" else "unlabeled"
    cz, cy, cx = (int(x) for x in chunk_shape)
    vz, vy, vx = (float(x) for x in voxel_size_nm)
    body = f"""# OpenOrganelle — Download script metadata ({title})

**Source:** https://openorganelle.janelia.org/datasets
**Script generated:** {gen_date}
**Generated At (ISO):** {gen_iso}
**Website Name:** OpenOrganelle
**Downloader mode:** {mode_line}
**Companion script:** `{stem}.py`

## Runtime parameters (CLI-overridable)

Defaults come from `DEFAULT_CHUNK_SHAPE`, `DEFAULT_N_CROPS`, and `DEFAULT_VOXEL_SIZE_NM` in the companion script, but can be overridden at runtime via CLI flags (storage order **z, y, x**).

| Setting | Value |
| --- | --- |
| `chunk_shape` (voxels, z,y,x) | `{cz}, {cy}, {cx}` |
| `n_crops` | {max(1, int(n_crops))} |
| `voxel_size_nm` (nm, z,y,x) | `{vz}, {vy}, {vx}` |
| `physical_size_nm` (nm, z,y,x) | `auto = voxel_size_nm * chunk_shape` unless `--physical-size-nm` provided |

**Last download:** {last_d}
**Last Download At (ISO):** {last_iso}
"""
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(body + "\n", encoding="utf-8")


def _update_sidecar_last_download(md_path: Path, mode: str) -> None:
    """Refresh **Last download** lines (same header as scrape reports). Used after `agent.py --execute`."""
    now = datetime.now()
    last_d = now.strftime("%Y-%m-%d")
    last_iso = now.replace(microsecond=0).isoformat()
    if not md_path.is_file():
        return
    text = md_path.read_text(encoding="utf-8")
    text = re.sub(
        r"^\*\*Last download:\*\*\s*.+$",
        f"**Last download:** {last_d}",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^\*\*Last Download At \(ISO\):\*\*\s*.+$",
        f"**Last Download At (ISO):** {last_iso}",
        text,
        flags=re.MULTILINE,
    )
    md_path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def _write_generated_downloader(
    mode: str,
    inv: dict[str, dict],
    *,
    chunk_shape: tuple[int, int, int],
    n_crops: int,
    voxel_size_nm: tuple[float, float, float],
) -> Path:
    GENERATED_PY_BASE.mkdir(parents=True, exist_ok=True)
    if mode != "labeled":
        raise ValueError(f"Only labeled mode is supported, got {mode!r}")
    gen_now = datetime.now()
    gen_date = gen_now.date().isoformat()
    gen_iso = gen_now.replace(microsecond=0).isoformat()
    datasets_literal = json.dumps(inv, indent=2, sort_keys=True)
    dataset_splits_literal = json.dumps(
        {
            name: {
                "training": int((cfg.get("download_split") or {}).get("training", 1)),
                "inference": int((cfg.get("download_split") or {}).get("inference", 0)),
            }
            for name, cfg in sorted(inv.items())
        },
        indent=2,
        sort_keys=True,
    )
    default_dataset = sorted(inv.keys())[0] if inv else ""
    script_path = GENERATED_PY_BASE / "download_openorganelle_labeled.py"
    md_path = GENERATED_PY_BASE / "download_openorganelle_labeled.md"
    script = (
        LABELED_TEMPLATE.replace("__TOTAL_DATASETS__", str(len(inv)))
        .replace("__DATASETS__", datasets_literal)
        .replace("__DATASET_SPLITS__", dataset_splits_literal)
        .replace("__DOWNLOADER_PATHS_BLOCK__", GENERATED_DOWNLOADER_PATHS_BLOCK)
        .replace("__DEFAULT_DATASET__", default_dataset)
        .replace("__DEFAULT_CHUNK_SHAPE__", str(tuple(int(x) for x in chunk_shape)))
        .replace("__DEFAULT_N_CROPS__", str(max(1, int(n_crops))))
        .replace("__DEFAULT_VOXEL_SIZE_NM__", str([float(x) for x in voxel_size_nm]))
        .replace("__GEN_DATE__", gen_date)
        .replace("__GEN_ISO__", gen_iso)
        .replace("__FOUNDATION_IMPORTS__", FOUNDATION_IMPORTS_LABELED)
    )
    script_path.write_text(script, encoding="utf-8")
    _write_sidecar_md_on_generate(
        md_path,
        mode,
        gen_date,
        gen_iso,
        chunk_shape=chunk_shape,
        n_crops=n_crops,
        voxel_size_nm=voxel_size_nm,
    )
    # Keep outputs clean: unlabeled generation has been removed.
    for stale in (
        GENERATED_PY_BASE / "download_openorganelle_unlabeled.py",
        GENERATED_PY_BASE / "download_openorganelle_unlabeled.md",
    ):
        try:
            if stale.exists():
                stale.unlink()
        except Exception:
            pass
    return script_path


def _download_one(
    *,
    dataset_name: str,
    cfg: dict,
    offsets: list[tuple[int, int, int]] | None,
    chunk_shape: tuple[int, int, int],
    n_crops: int,
    voxel_size_nm_override: tuple[float, float, float] | None,
    physical_size_nm_override: tuple[float, float, float] | None,
    window_policy: str,
    mode: str,
    labeled_img_dir: Path | None,
    labeled_seg_dir: Path | None,
    unlabeled_run_dir: Path | None,
    foundation: bool = True,
) -> None:
    img_path = cfg["img_path"]
    seg_path = (cfg.get("seg_path") or "").strip()
    # Enforce isotropic 16 nm outputs for all Stage-3 downloads.
    voxel_size_nm = [16.0, 16.0, 16.0]
    tag = dataset_name.replace("-", "_")

    creds = {"anon": True}
    image_group = _as_array(_try_read(img_path, creds))
    seg_group = _as_array(_try_read(seg_path, creds)) if mode == "labeled" else None
    img_shape: tuple[int, int, int] | None = None
    seg_shape: tuple[int, int, int] | None = None
    if mode == "labeled":
        img_shape = tuple(int(image_group.shape[d]) for d in range(3))
        seg_shape = tuple(int(seg_group.shape[d]) for d in range(3))
        vol_shape = _planning_vol_shape_em(cfg, img_path, img_shape)
    else:
        img_shape = tuple(int(image_group.shape[d]) for d in range(3))
        vol_shape = _planning_vol_shape_em(cfg, img_path, img_shape)

    if foundation:
        if mode == "labeled":
            if _dl_foundation_labeled is None:
                raise RuntimeError("openorganelle.foundation_resample dependencies are not available")
            assert img_shape is not None and seg_shape is not None
            assert labeled_img_dir is not None and labeled_seg_dir is not None
            labeled_img_dir.mkdir(parents=True, exist_ok=True)
            labeled_seg_dir.mkdir(parents=True, exist_ok=True)
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
                img_dir=str(labeled_img_dir),
                seg_dir=str(labeled_seg_dir),
                n_crops=max(1, int(n_crops)),
                out_spacing_nm_zyx=tuple(float(v) for v in voxel_size_nm),
                out_max_voxels_zyx=tuple(int(v) for v in chunk_shape),
                physical_size_nm_zyx=(
                    tuple(float(v) for v in physical_size_nm_override)
                    if physical_size_nm_override is not None
                    else None
                ),
                write_h5_fn=_foundation_write_h5_str,
            )
            return
        assert unlabeled_run_dir is not None
        if _dl_foundation_unlabeled is None:
            raise RuntimeError("openorganelle.foundation_resample dependencies are not available")
        images_dir = unlabeled_run_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        _dl_foundation_unlabeled(
            dataset_name=dataset_name,
            cfg=cfg,
            image_group=image_group,
            img_path=img_path,
            img_shape=img_shape,
            vol_shape=vol_shape,
            offsets=offsets,
            window_policy=window_policy,
            images_dir=str(images_dir),
            n_crops=max(1, int(n_crops)),
            out_spacing_nm_zyx=tuple(float(v) for v in voxel_size_nm),
            out_max_voxels_zyx=tuple(int(v) for v in chunk_shape),
            physical_size_nm_zyx=(
                tuple(float(v) for v in physical_size_nm_override)
                if physical_size_nm_override is not None
                else None
            ),
            write_h5_fn=_foundation_write_h5_str,
        )
        return

    if mode == "labeled":
        assert seg_shape is not None
        rz, ry, rx = _axis_ratios(img_shape, seg_shape)
        print(f"[INFO] {dataset_name}: seg/EM index ratio (zyx) ≈ ({rz:.4f}, {ry:.4f}, {rx:.4f})")

    ch_eff = _effective_chunk(vol_shape, chunk_shape)
    if ch_eff != chunk_shape:
        print(f"[INFO] {dataset_name}: chunk capped {chunk_shape} -> {ch_eff} for volume {vol_shape}")

    if mode == "unlabeled" and window_policy == "mito":
        print(f"[WARN] {dataset_name}: --window-policy mito is labeled-only; using center")
        window_policy = "center"

    if offsets is not None:
        win_offsets = list(offsets)
    elif int(n_crops) > 1:
        win_offsets, possible = _grid_offsets_non_overlapping(vol_shape, ch_eff, int(n_crops))
        if len(win_offsets) < int(n_crops):
            print(f"[INFO] {dataset_name}: requested {n_crops} crops, max non-overlap fit={possible} for chunk {ch_eff}")
        print(f"[INFO] {dataset_name}: non-overlap crops selected={len(win_offsets)} chunk {ch_eff}")
    elif window_policy == "mito" and mode == "labeled":
        assert img_shape is not None and seg_shape is not None
        win_offsets = [
            _mito_anchor_in_img_space(
                seg_group, img_shape, seg_shape, ch_eff, vol_clip=vol_shape
            )
        ]
        print(f"[INFO] {dataset_name}: mito-guided crop offset {win_offsets[0]} chunk {ch_eff}")
    else:
        win_offsets = [_center_offset(vol_shape, ch_eff)]
        print(f"[INFO] {dataset_name}: center crop offset {win_offsets[0]} chunk {ch_eff}")

    if mode == "labeled":
        assert labeled_img_dir is not None and labeled_seg_dir is not None
        planned_suffixes = _resolve_batch_coherent_suffixes(
            dataset_name=dataset_name,
            img_dir=str(labeled_img_dir),
            seg_dir=str(labeled_seg_dir),
            n_windows=len(win_offsets),
        )
    else:
        planned_suffixes = _resolve_batch_coherent_suffixes(
            dataset_name=dataset_name,
            img_dir=str(unlabeled_run_dir),
            seg_dir=str(unlabeled_run_dir),
            n_windows=len(win_offsets),
        )

    for i, offset in enumerate(win_offsets, start=1):
        suffix = _crop_index_suffix(i, len(win_offsets))
        resolved_suffix = planned_suffixes[i - 1] if i - 1 < len(planned_suffixes) else suffix
        win = _window_from_offset(vol_shape, offset, ch_eff)
        if win is None:
            print(f"[WARN] {dataset_name} vol{i}: invalid window at offset={offset}; skipped")
            continue
        (z0, y0, x0), (z1, y1, x1) = win
        if (z0, y0, x0) != offset:
            print(f"[INFO] {dataset_name} vol{i}: shifted offset {offset} -> {(z0, y0, x0)} to fit volume")
        image = image_group[z0:z1, y0:y1, x0:x1]
        if mode == "labeled":
            assert img_shape is not None and seg_shape is not None
            sz0, sz1, sy0, sy1, sx0, sx1 = _img_window_to_seg_slices(
                z0, z1, y0, y1, x0, x1, img_shape, seg_shape
            )
            seg_crop = seg_group[sz0:sz1, sy0:sy1, sx0:sx1]
            seg = _resize_seg_to_match(seg_crop, image.shape)
            assert labeled_img_dir is not None and labeled_seg_dir is not None
            img_prefix = labeled_img_dir / f"{tag}{resolved_suffix}"
            seg_prefix = labeled_seg_dir / f"{tag}{resolved_suffix}"
        else:
            seg = None
            img_prefix = unlabeled_run_dir / f"{tag}{resolved_suffix}"
            seg_prefix = None

        write_h5(Path(f"{img_prefix}_im.h5"), image, f"{tag}{resolved_suffix}_im")
        if mode == "labeled":
            write_h5(Path(f"{seg_prefix}_seg.h5"), seg, f"{tag}{resolved_suffix}_seg")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenOrganelle downloader generator")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--mode", choices=("labeled",), default=DEFAULT_MODE)
    p.add_argument("--dataset", "-d", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument(
        "--window-policy",
        "-w",
        choices=("center", "fixed", "mito"),
        default=DEFAULT_WINDOW_POLICY,
        help="center(default)=per-dataset centered crop; mito=labeled seg coarse anchor; fixed=require --offsets",
    )
    p.add_argument("--offsets", "-o", nargs="+", default=None)
    p.add_argument("--chunk", "-c", default=",".join(map(str, DEFAULT_CHUNK)))
    p.add_argument("--n-crops", type=int, default=1, help="Requested non-overlapping crops per dataset (center when =1)")
    p.add_argument("--voxel-size-nm", default="16,16,16", help="Voxel spacing as z,y,x nm (foundation); isotropic 16 nm matches x,y,z 16,16,16")
    p.add_argument("--physical-size-nm", default="", help="Physical crop size in nm as z,y,x before resampling (optional)")
    p.add_argument("--run-name", default=None)
    p.add_argument("--list", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument(
        "--no-foundation",
        action="store_true",
        help="Legacy raw crop without resampling to 512³ @ 16 nm isotropic",
    )
    p.add_argument(
        "--no-registry",
        action="store_true",
        help="Disable registry skip/record (overrides default-on behavior; or set MITO2_DISABLE_REGISTRY=1).",
    )
    p.add_argument(
        "--use-registry",
        action="store_true",
        help="(Kept for backward compat — registry is on by default unless MITO2_DISABLE_REGISTRY=1).",
    )
    return p.parse_args()


def _registry_skip_and_record(
    *,
    dataset_name: str,
    img_url: str,
    seg_url: str,
    mode: str,
    output_dir: Path,
    n_crops: int = 1,
    chunk_shape: tuple[int, int, int] = (512, 512, 512),
    voxel_size_nm: list[float] | None = None,
    foundation: bool = True,
) -> tuple[bool, "callable"]:
    """Return (should_skip, record_fn).

    should_skip=True when a complete download already exists in the registry
    **for the same download profile** (n_crops + chunk_shape + voxel_size_nm +
    mode + foundation).  A different n_crops value produces a distinct profile
    hash so the registry will allow the new download to proceed even if an
    earlier run with a different n_crops completed successfully.

    record_fn() should be called after a successful download to persist the result.
    """
    try:
        import sys as _sys
        if str(ROOT) not in _sys.path:
            _sys.path.insert(0, str(ROOT))
        from agent.orchestration.registry.schema import open_registry
        from agent.orchestration.registry.api import (
            upsert_provider, upsert_asset,
            get_dataset_id, upsert_dataset,
            make_download_profile_hash,
            is_asset_downloaded_for_profile,
            record_download_start, record_download_complete,
        )
        conn = open_registry()
        provider_id = upsert_provider(conn, name="OpenOrganelle", base_url="https://openorganelle.janelia.org")
        conn.commit()

        dataset_id = get_dataset_id(conn, provider_id, dataset_name)
        if dataset_id is None:
            dataset_id = upsert_dataset(conn, provider_id=provider_id, stable_id=dataset_name)
            conn.commit()

        em_asset_id = upsert_asset(conn, dataset_id=dataset_id, asset_type="em_volume", remote_url=img_url)
        conn.commit()

        # Build a profile hash from the actual runtime parameters so that
        # different n_crops (or other parameter) combinations are tracked
        # independently in the registry.
        voxel = voxel_size_nm or [16.0, 16.0, 16.0]
        profile_hash = make_download_profile_hash(
            n_crops=n_crops,
            chunk_zyx=chunk_shape,
            voxel_nm_zyx=tuple(float(v) for v in voxel[:3]),
            mode=mode,
            foundation=foundation,
        )

        # Skip if a complete, locally-present download exists for this exact profile.
        if is_asset_downloaded_for_profile(conn, em_asset_id, profile_hash):
            conn.close()
            print(
                f"[REGISTRY] Skipping {dataset_name}: already downloaded "
                f"(profile n_crops={n_crops}, hash={profile_hash})"
            )
            return True, lambda: None

        dl_id = record_download_start(
            conn, em_asset_id, str(output_dir),
            download_profile_hash=profile_hash,
        )
        conn.commit()

        if seg_url and mode == "labeled":
            upsert_asset(conn, dataset_id=dataset_id, asset_type="mito_seg", remote_url=seg_url)
            conn.commit()

        def _record_done():
            try:
                record_download_complete(conn, dl_id)
                conn.commit()
                conn.close()
                print(f"[REGISTRY] Recorded download: {dataset_name} → {output_dir} (profile={profile_hash})")
            except Exception as exc:
                print(f"[REGISTRY] Warning: could not record download completion: {exc}")

        return False, _record_done
    except Exception as exc:
        print(f"[REGISTRY] Warning: registry check failed for {dataset_name}: {exc}")
        return False, lambda: None


def generate_openorganelle_downloader_script(
    *,
    mode: str,
    chunk_zyx: str,
    n_crops: int,
    voxel_size_nm_zyx: str,
    db_path: Path | None = None,
    dataset_splits: dict[str, dict[str, int]] | None = None,
) -> Path:
    """Write ``download_openorganelle_{mode}.py`` using explicit parameters (Pipeline Studio, in-process).

    ``chunk_zyx`` and ``voxel_size_nm_zyx`` use storage order **z,y,x** (same as ``--chunk`` / ``--voxel-size-nm``).
    """
    if mode != "labeled":
        raise ValueError(f"mode must be 'labeled', got {mode!r}")
    db = (db_path or DEFAULT_DB).expanduser().resolve()
    if not db.is_file():
        raise FileNotFoundError(f"DB not found: {db}")
    inv = _select_inventory(db, mode)
    if not inv:
        raise RuntimeError(f"No datasets available for mode={mode!r} in {db.name}")
    selected_inv: dict[str, dict] = {}
    requested = dataset_splits or {}
    has_explicit_splits = isinstance(dataset_splits, dict) and len(dataset_splits) > 0
    for name, cfg in inv.items():
        raw_split = requested.get(name, requested.get(name.replace("-", "_"), {})) or {}
        default_training = 0 if has_explicit_splits else 1
        default_inference = 0
        try:
            tr = max(0, int(raw_split.get("training", default_training)))
        except Exception:
            tr = default_training
        try:
            ts = max(0, int(raw_split.get("inference", default_inference)))
        except Exception:
            ts = default_inference
        tr = min(16, tr)
        ts = min(16, ts)
        if tr + ts > 16:
            ts = max(0, 16 - tr)
        if tr + ts <= 0:
            continue
        row = dict(cfg)
        row["download_split"] = {"training": tr, "inference": ts}
        selected_inv[name] = row
    if not selected_inv:
        raise RuntimeError("No datasets selected for download after applying dataset_splits.")
    chunk_shape = _parse_chunk(chunk_zyx)
    voxel_size_nm = _parse_voxel_size_nm(voxel_size_nm_zyx)
    return _write_generated_downloader(
        mode,
        selected_inv,
        chunk_shape=chunk_shape,
        n_crops=max(1, int(n_crops)),
        voxel_size_nm=voxel_size_nm,
    )


def _apply_studio_subprocess_env_overrides(args: argparse.Namespace) -> None:
    """When chat_web runs the generator via ``run_command(..., env=…)``, apply chunk / n_crops / voxel / foundation.

    This makes generation respect Pipeline Studio settings even if ``sys.argv`` forwarding through
    ``master/agent.py`` is wrong or the parent Uvicorn process was not restarted after an API update.
    """
    ch = os.environ.get("MITO2_STUDIO_OG_CHUNK", "").strip()
    if ch:
        args.chunk = ch
    nc = os.environ.get("MITO2_STUDIO_OG_N_CROPS", "").strip()
    if nc:
        args.n_crops = int(nc)
    vx = os.environ.get("MITO2_STUDIO_OG_VOXEL_NM", "").strip()
    if vx:
        args.voxel_size_nm = vx
    ps = os.environ.get("MITO2_STUDIO_OG_PHYSICAL_NM", "").strip()
    if ps:
        args.physical_size_nm = ps
    nf = os.environ.get("MITO2_STUDIO_OG_NO_FOUNDATION")
    if nf is not None and str(nf).strip() != "":
        args.no_foundation = str(nf).strip() == "1"


def main() -> None:
    args = parse_args()
    _apply_studio_subprocess_env_overrides(args)
    db_path = args.db.expanduser().resolve()
    if not db_path.is_file():
        raise SystemExit(f"[ERROR] DB not found: {db_path}")

    inv = _select_inventory(db_path, args.mode)
    if not inv:
        raise SystemExit(f"[ERROR] No datasets available for mode={args.mode!r} in {db_path.name}")

    chunk_shape = _parse_chunk(args.chunk)
    voxel_size_nm = (16.0, 16.0, 16.0)
    physical_size_nm = (
        _parse_voxel_size_nm(args.physical_size_nm)
        if str(getattr(args, "physical_size_nm", "")).strip()
        else None
    )
    script_path = _write_generated_downloader(
        args.mode,
        inv,
        chunk_shape=chunk_shape,
        n_crops=max(1, int(args.n_crops)),
        voxel_size_nm=voxel_size_nm,
    )
    print(f"[OK] Generated downloader script: {script_path}")
    print(
        f"Run with: python {script_path} --dataset <name> --window-policy center "
        f"(default: foundation uses script chunk/spacing/n-crops settings; add --no-foundation for legacy crop)"
    )

    if args.list:
        print(f"Available datasets ({len(inv)}) [mode={args.mode}] from {db_path.name}:")
        for name, cfg in sorted(inv.items()):
            print(f"  {name}")
            print(f"    img: {cfg.get('img_path', '')}")
            if args.mode == "labeled":
                print(f"    seg: {cfg.get('seg_path', '')}")
        return

    if not args.execute:
        print("Generate-only mode complete. No data downloaded.")
        print("Add --execute to run download directly from this command.")
        return

    offsets = _execute_offsets(args.offsets, args.window_policy)
    chunk_shape = _parse_chunk(args.chunk)
    targets = [args.dataset] if args.dataset else sorted(inv.keys())
    labeled_img_dir = None
    labeled_seg_dir = None
    unlabeled_run_dir = None
    batch_id: str | None = None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_id = (
        args.run_name.strip()
        if args.run_name and args.run_name.strip()
        else f"openorganelle_{stamp}"
    )
    labeled_img_dir = TRAINING_ROOT / "imagesTr"
    labeled_seg_dir = TRAINING_ROOT / "labelsTr"
    labeled_img_dir.mkdir(parents=True, exist_ok=True)
    labeled_seg_dir.mkdir(parents=True, exist_ok=True)
    print(f"[BATCH_ID] {batch_id}")

    _reg_disabled = (
        os.environ.get("MITO2_DISABLE_REGISTRY", "").strip().lower() in ("1", "true", "yes", "on")
        or getattr(args, "no_registry", False)
    )
    use_registry = not _reg_disabled
    failed: list[str] = []
    successful: list[str] = []
    for name in targets:
        key = name if name in inv else name.replace("_", "-")
        if key not in inv:
            failed.append(name)
            continue

        cfg = inv[key]
        output_dir = labeled_img_dir or unlabeled_run_dir or TRAINING_ROOT
        skip = False
        record_fn = lambda: None
        if use_registry:
            skip, record_fn = _registry_skip_and_record(
                dataset_name=key,
                img_url=cfg.get("img_path", ""),
                seg_url=cfg.get("seg_path", ""),
                mode=args.mode,
                output_dir=output_dir,
                n_crops=max(1, int(args.n_crops)),
                chunk_shape=chunk_shape,
                voxel_size_nm=list(voxel_size_nm),
                foundation=not args.no_foundation,
            )
        if skip:
            continue

        try:
            _download_one(
                dataset_name=key,
                cfg=cfg,
                offsets=offsets,
                chunk_shape=chunk_shape,
                n_crops=max(1, int(args.n_crops)),
                voxel_size_nm_override=voxel_size_nm,
                physical_size_nm_override=physical_size_nm,
                window_policy=args.window_policy,
                mode=args.mode,
                labeled_img_dir=labeled_img_dir,
                labeled_seg_dir=labeled_seg_dir,
                unlabeled_run_dir=unlabeled_run_dir,
                foundation=not args.no_foundation,
            )
            if use_registry:
                record_fn()
            successful.append(key)
        except Exception as exc:
            print(f"[ERROR] {key}: {exc}")
            failed.append(key)
    if failed:
        print(f"[WARN] Failed datasets: {failed}")

    if args.mode == "labeled" and batch_id is not None:
        _inv = ROOT / "0inventory"
        if str(_inv) not in sys.path:
            sys.path.insert(0, str(_inv))
        from download_history import finalize_openorganelle_labeled_script_run

        n_win_exec = len(offsets) if offsets else max(1, int(args.n_crops))
        finalize_openorganelle_labeled_script_run(
            ROOT,
            batch_id=batch_id,
            successful_targets=successful,
            img_dir=labeled_img_dir,  # type: ignore[arg-type]
            seg_dir=labeled_seg_dir,  # type: ignore[arg-type]
            chunk_shape=chunk_shape,
            n_crops=max(1, int(args.n_crops)),
            voxel_nm=list(voxel_size_nm),
            foundation=not args.no_foundation,
            run_preprocess=False,
            n_windows=int(n_win_exec),
        )
        ds_json = _sync_dataset_json(TRAINING_ROOT)
        print(f"[DONE] dataset.json synced: {ds_json}")

    _update_sidecar_last_download(
        GENERATED_PY_BASE / f"download_openorganelle_{args.mode}.md",
        args.mode,
    )


if __name__ == "__main__":
    main()

