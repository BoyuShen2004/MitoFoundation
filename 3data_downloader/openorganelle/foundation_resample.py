"""
Foundation-model training blocks: **16 nm** isotropic spacing, up to **512** voxels per axis.

Source spacing ``voxel_size_nm`` (with ``grid_axes``) is mapped to array order **z, y, x**.
Native crop size per axis = ceil(8192 nm / spacing_axis), capped by volume extent.

After cropping, volumes are resampled so each axis has
``min(512, round(physical_extent_nm / 16))`` voxels (no forced 512³ when an axis is shorter).
EM uses **order=1**, segmentation **order=0** (nearest).

Typical: 16 nm native → crop up to 512³ → output 512³; 8 nm → crop up to 1024³ → output 512³;
short axes keep fewer voxels at 16 nm.
"""

from __future__ import annotations

import ast
from typing import Any

import numpy as np
from scipy.ndimage import zoom

PHYSICAL_EDGE_NM = 512.0 * 16.0  # legacy default = 8192 nm = 8.192 um
OUT_SHAPE_ZYX = (512, 512, 512)
OUT_SPACING_NM_ISO = (16.0, 16.0, 16.0)


def spacing_nm_zyx(cfg: dict) -> tuple[float, float, float] | None:
    """Return voxel spacing in nm aligned with fibsem_tools array order (z, y, x)."""
    v = cfg.get("voxel_size_nm") or cfg.get("voxel_size")
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        return None
    try:
        f = [float(x) for x in v]
    except (TypeError, ValueError):
        return None
    if any(x <= 0 or np.isnan(x) for x in f):
        return None

    ga = cfg.get("grid_axes")
    if ga is not None and str(ga).strip():
        try:
            axes = ast.literal_eval(str(ga).strip())
        except (SyntaxError, ValueError, TypeError):
            axes = None
        if isinstance(axes, (list, tuple)) and len(axes) == 3:
            order = [str(a).lower() for a in axes]
            if order == ["x", "y", "z"]:
                return (f[2], f[1], f[0])
            if order == ["z", "y", "x"]:
                return (f[0], f[1], f[2])
    # Common probe layout: [sx, sy, sz] for x,y,z
    return (f[2], f[1], f[0])


def native_crop_shape_zyx(
    spacing_zyx: tuple[float, float, float],
    physical_size_nm_zyx: tuple[float, float, float],
) -> tuple[int, int, int]:
    """Voxel counts along z,y,x to cover requested physical size on each axis."""
    return tuple(int(np.ceil(float(physical_size_nm_zyx[i]) / float(spacing_zyx[i]))) for i in range(3))


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
    factors = tuple(target_shape[i] / seg_crop.shape[i] for i in range(3))
    return zoom(seg_crop, factors, order=0)


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


def _mito_anchor_offset(
    seg_group: Any,
    img_shape: tuple[int, int, int],
    seg_shape: tuple[int, int, int],
    ch_eff_img: tuple[int, int, int],
    vol_clip: tuple[int, int, int],
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
    if coarse.size == 0 or not (coarse != 0).any():
        return _center_offset(vol_clip, ch_eff_img)
    zz, yy, xx = np.nonzero(coarse != 0)
    cz_s = int(np.clip(np.mean(zz) * step_z + step_z // 2, 0, Dz_s - 1))
    cy_s = int(np.clip(np.mean(yy) * step_y + step_y // 2, 0, Dy_s - 1))
    cx_s = int(np.clip(np.mean(xx) * step_x + step_x // 2, 0, Dx_s - 1))
    cz = int(np.clip(cz_s / rz, 0, img_shape[0] - 1))
    cy = int(np.clip(cy_s / ry, 0, img_shape[1] - 1))
    cx = int(np.clip(cx_s / rx, 0, img_shape[2] - 1))
    out: list[int] = []
    for c, d, ch in zip((cz, cy, cx), vol_clip, ch_eff_img):
        o = int(c - ch // 2)
        o = max(0, min(o, int(d) - int(ch)))
        out.append(o)
    return tuple(out)


def target_shape_isotropic(
    crop_shape: tuple[int, int, int],
    spacing_zyx: tuple[float, float, float],
    *,
    out_spacing_nm_zyx: tuple[float, float, float] = OUT_SPACING_NM_ISO,
    max_voxels_zyx: tuple[int, int, int] = OUT_SHAPE_ZYX,
) -> tuple[int, int, int]:
    """
    Voxel counts at ``out_spacing_nm_zyx`` per axis, capped by ``max_voxels_zyx``.

    Physical extent on axis i is ``crop_shape[i] * spacing_zyx[i]`` (nm). At 16 nm isotropic
    sampling that is ``extent / out_spacing_nm`` voxels; we never exceed ``max_voxels`` per axis
    and never invent extent beyond the crop (no upsampling past ~512×spacing native edge).
    """
    out: list[int] = []
    for i in range(3):
        ext_nm = float(crop_shape[i]) * float(spacing_zyx[i])
        n = max(1, int(round(ext_nm / float(out_spacing_nm_zyx[i]))))
        out.append(min(int(max_voxels_zyx[i]), n))
    return (out[0], out[1], out[2])


def resample_crop_to_isotropic(
    vol: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    *,
    order: int,
    max_voxels_zyx: tuple[int, int, int] = OUT_SHAPE_ZYX,
    out_spacing_nm_zyx: tuple[float, float, float] = OUT_SPACING_NM_ISO,
) -> np.ndarray:
    """
    Map a native-spacing crop to target output spacing (nm) in z,y,x.

    Output shape per axis is
    ``min(max_voxels_axis, round(physical_nm / out_spacing_axis_nm))`` so short axes stay as
    short as the data allows.
    """
    v = np.asarray(vol)
    if v.ndim != 3:
        raise ValueError(f"expected 3D volume, got shape {v.shape}")
    tgt = target_shape_isotropic(
        v.shape,
        spacing_zyx,
        out_spacing_nm_zyx=out_spacing_nm_zyx,
        max_voxels_zyx=max_voxels_zyx,
    )
    if v.shape == tgt:
        return v
    factors = tuple(tgt[i] / float(v.shape[i]) for i in range(3))
    return zoom(v, factors, order=order)


def resample_crop_to_16nm(
    vol: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    *,
    order: int,
    max_voxels: int = OUT_SHAPE_ZYX[0],
    out_spacing_nm: float = OUT_SPACING_NM_ISO[0],
) -> np.ndarray:
    """Backward-compatible wrapper for legacy callers."""
    return resample_crop_to_isotropic(
        vol,
        spacing_zyx,
        order=order,
        max_voxels_zyx=(int(max_voxels), int(max_voxels), int(max_voxels)),
        out_spacing_nm_zyx=(float(out_spacing_nm), float(out_spacing_nm), float(out_spacing_nm)),
    )


def foundation_crop_shapes(
    spacing_zyx: tuple[float, float, float],
    vol_shape: tuple[int, int, int],
    physical_size_nm_zyx: tuple[float, float, float],
) -> tuple[int, int, int]:
    """Native crop (z,y,x) capped to volume for requested physical size."""
    need = native_crop_shape_zyx(spacing_zyx, physical_size_nm_zyx)
    return tuple(min(need[i], vol_shape[i]) for i in range(3))


def _grid_offsets_non_overlapping(
    vol_shape: tuple[int, int, int],
    ch_eff: tuple[int, int, int],
    n_crops: int,
) -> tuple[list[tuple[int, int, int]], int]:
    if n_crops <= 1:
        return [_center_offset(vol_shape, ch_eff)], 1
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
    offsets.sort(key=lambda o: (o[0] - center[0]) ** 2 + (o[1] - center[1]) ** 2 + (o[2] - center[2]) ** 2)
    total_possible = len(offsets)
    return offsets[: min(int(n_crops), total_possible)], total_possible


def _farthest_offsets_non_overlapping(
    vol_shape: tuple[int, int, int],
    ch_eff: tuple[int, int, int],
    n_crops: int,
) -> tuple[list[tuple[int, int, int]], int]:
    """
    Pick non-overlapping offsets using greedy farthest-point sampling on a coarse grid.

    Candidates are valid window starts; minimum separation is chunk size per axis
    to enforce non-overlap.
    """
    target = max(1, int(n_crops))
    axes: list[list[int]] = []
    for dim, ch in zip(vol_shape, ch_eff):
        d, c = int(dim), int(ch)
        if d <= c:
            axes.append([0])
            continue
        # Coarse but dense-enough starts (half-chunk stride + edge anchor)
        step = max(1, c // 2)
        vals = list(range(0, max(1, d - c + 1), step))
        if vals[-1] != d - c:
            vals.append(d - c)
        axes.append(sorted(set(vals)))

    candidates = [(z, y, x) for z in axes[0] for y in axes[1] for x in axes[2]]
    if not candidates:
        candidates = [_center_offset(vol_shape, ch_eff)]
    total_possible = len(candidates)
    center = _center_offset(vol_shape, ch_eff)

    def _non_overlap(a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
        # Non-overlap in all axes for equal-sized windows.
        return (
            abs(int(a[0]) - int(b[0])) >= int(ch_eff[0])
            or abs(int(a[1]) - int(b[1])) >= int(ch_eff[1])
            or abs(int(a[2]) - int(b[2])) >= int(ch_eff[2])
        )

    # Seed with the candidate nearest center.
    seed = min(
        candidates,
        key=lambda o: (o[0] - center[0]) ** 2 + (o[1] - center[1]) ** 2 + (o[2] - center[2]) ** 2,
    )
    picked: list[tuple[int, int, int]] = [seed]
    remaining = [c for c in candidates if c != seed]

    while remaining and len(picked) < target:
        valid = [c for c in remaining if all(_non_overlap(c, p) for p in picked)]
        if not valid:
            break
        # Farthest from already selected set (maximize minimum distance).
        nxt = max(
            valid,
            key=lambda c: min(
                (c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2 + (c[2] - p[2]) ** 2 for p in picked
            ),
        )
        picked.append(nxt)
        remaining.remove(nxt)

    return picked, total_possible


def _crop_index_suffix(i: int, total: int) -> str:
    # Match provider-wide naming convention: ``_vol1``, ``_vol2``, ...
    return f"_vol{int(i)}"


def _resolve_batch_coherent_suffixes(
    *,
    dataset_name: str,
    img_dir: str,
    seg_dir: str | None,
    n_windows: int,
) -> list[str]:
    """If any base ``_volN`` collides, assign one shared ``_M`` to every crop in the batch."""
    import os

    n = max(1, int(n_windows))
    tag = dataset_name.replace("-", "_")
    base_suffixes = [f"_vol{i}" for i in range(1, n + 1)]

    def _exists(stem: str) -> bool:
        hit_img = os.path.exists(os.path.join(img_dir, f"{tag}{stem}_0000.nii.gz"))
        hit_seg = bool(seg_dir) and os.path.exists(os.path.join(str(seg_dir), f"{tag}{stem}.nii.gz"))
        return bool(hit_img or hit_seg)

    if not any(_exists(s) for s in base_suffixes):
        return base_suffixes
    m = 2
    while True:
        trial = [f"_vol{i}_{m}" for i in range(1, n + 1)]
        if all(not _exists(s) for s in trial):
            print(f"[INFO] {dataset_name}: batch collision detected; applying shared suffix _{m} to all {n} crop(s)")
            return trial
        m += 1


def download_foundation_labeled(
    *,
    dataset_name: str,
    cfg: dict,
    image_group: Any,
    seg_group: Any,
    img_path: str,
    img_shape: tuple[int, int, int],
    seg_shape: tuple[int, int, int],
    vol_shape: tuple[int, int, int],
    offsets: list[tuple[int, int, int]] | None,
    window_policy: str,
    img_dir: str,
    seg_dir: str,
    n_crops: int = 1,
    out_spacing_nm_zyx: tuple[float, float, float] = OUT_SPACING_NM_ISO,
    out_max_voxels_zyx: tuple[int, int, int] = OUT_SHAPE_ZYX,
    physical_size_nm_zyx: tuple[float, float, float] | None = None,
    write_h5_fn,
) -> None:
    """Extract configurable isotropic foundation-style blocks for EM + seg."""
    import os

    spacing = spacing_nm_zyx(cfg)
    if spacing is None:
        raise RuntimeError(
            f"{dataset_name}: missing or invalid voxel_size_nm — cannot build 16 nm isotropic blocks"
        )

    if physical_size_nm_zyx is None:
        physical_size_nm_zyx = tuple(
            float(out_spacing_nm_zyx[i]) * float(out_max_voxels_zyx[i]) for i in range(3)
        )
    ch_need = foundation_crop_shapes(spacing, vol_shape, physical_size_nm_zyx)
    if min(ch_need) < 2:
        raise RuntimeError(f"{dataset_name}: volume too small for foundation crop {ch_need}")

    rz, ry, rx = _axis_ratios(img_shape, seg_shape)
    print(
        f"[INFO] {dataset_name}: foundation mode — spacing z,y,x nm ≈ {spacing}, "
        f"native crop z,y,x = {ch_need}, physical size nm {physical_size_nm_zyx}, "
        f"output spacing {out_spacing_nm_zyx}, max voxels {out_max_voxels_zyx}; "
        f"seg/EM ratio ≈ ({rz:.4f},{ry:.4f},{rx:.4f})"
    )

    if offsets is not None:
        win_offsets = list(offsets)
    elif int(n_crops) > 1:
        win_offsets, possible = _farthest_offsets_non_overlapping(vol_shape, ch_need, int(n_crops))
        if len(win_offsets) < int(n_crops):
            print(
                f"[INFO] {dataset_name}: requested {n_crops} crops, selected {len(win_offsets)} "
                f"(non-overlap limit for crop {ch_need}; candidate starts={possible})"
            )
        print(f"[INFO] {dataset_name}: farthest non-overlap crops selected={len(win_offsets)}")
    elif window_policy == "mito":
        win_offsets = [
            _mito_anchor_offset(seg_group, img_shape, seg_shape, ch_need, vol_shape)
        ]
        print(f"[INFO] {dataset_name}: mito-guided offset {win_offsets[0]}")
    else:
        win_offsets = [_center_offset(vol_shape, ch_need)]
        print(f"[INFO] {dataset_name}: center offset {win_offsets[0]}")

    tag = dataset_name.replace("-", "_")
    planned_suffixes = _resolve_batch_coherent_suffixes(
        dataset_name=dataset_name,
        img_dir=img_dir,
        seg_dir=seg_dir,
        n_windows=len(win_offsets),
    )
    for i, offset in enumerate(win_offsets, start=1):
        win = _window_from_offset(vol_shape, offset, ch_need)
        if win is None:
            print(f"[WARN] {dataset_name} vol{i}: invalid window; skipped")
            continue
        (z0, y0, x0), (z1, y1, x1) = win
        img_crop = np.asarray(image_group[z0:z1, y0:y1, x0:x1])
        sz0, sz1, sy0, sy1, sx0, sx1 = _img_window_to_seg_slices(
            z0, z1, y0, y1, x0, x1, img_shape, seg_shape
        )
        seg_crop = np.asarray(seg_group[sz0:sz1, sy0:sy1, sx0:sx1])
        seg_native = _resize_seg_to_match(seg_crop, img_crop.shape)

        img_out = resample_crop_to_isotropic(
            img_crop,
            spacing,
            order=1,
            out_spacing_nm_zyx=out_spacing_nm_zyx,
            max_voxels_zyx=out_max_voxels_zyx,
        )
        seg_out = resample_crop_to_isotropic(
            seg_native,
            spacing,
            order=0,
            out_spacing_nm_zyx=out_spacing_nm_zyx,
            max_voxels_zyx=out_max_voxels_zyx,
        )

        suffix = planned_suffixes[i - 1] if i - 1 < len(planned_suffixes) else _crop_index_suffix(i, len(win_offsets))
        img_prefix = os.path.join(img_dir, f"{tag}{suffix}")
        seg_prefix = os.path.join(seg_dir, f"{tag}{suffix}")
        write_h5_fn(f"{img_prefix}_im.h5", img_out, f"{tag}{suffix}_im")
        write_h5_fn(f"{seg_prefix}_seg.h5", seg_out, f"{tag}{suffix}_seg")
        # Match BossDB / MitoLE local stage-3 per-crop [DONE] lines (nnUNet naming).
        print(
            f"[DONE] {dataset_name} vol{i}/{len(win_offsets)}: "
            f"{tag}{suffix}_0000.nii.gz  {tag}{suffix}.nii.gz"
        )


def download_foundation_unlabeled(
    *,
    dataset_name: str,
    cfg: dict,
    image_group: Any,
    img_path: str,
    img_shape: tuple[int, int, int],
    vol_shape: tuple[int, int, int],
    offsets: list[tuple[int, int, int]] | None,
    window_policy: str,
    images_dir: str,
    n_crops: int = 1,
    out_spacing_nm_zyx: tuple[float, float, float] = OUT_SPACING_NM_ISO,
    out_max_voxels_zyx: tuple[int, int, int] = OUT_SHAPE_ZYX,
    physical_size_nm_zyx: tuple[float, float, float] | None = None,
    write_h5_fn,
) -> None:
    import os

    spacing = spacing_nm_zyx(cfg)
    if spacing is None:
        raise RuntimeError(
            f"{dataset_name}: missing or invalid voxel_size_nm — cannot build 16 nm isotropic blocks"
        )

    if physical_size_nm_zyx is None:
        physical_size_nm_zyx = tuple(
            float(out_spacing_nm_zyx[i]) * float(out_max_voxels_zyx[i]) for i in range(3)
        )
    ch_need = foundation_crop_shapes(spacing, vol_shape, physical_size_nm_zyx)
    if min(ch_need) < 2:
        raise RuntimeError(f"{dataset_name}: volume too small for foundation crop {ch_need}")

    print(
        f"[INFO] {dataset_name}: foundation mode — spacing z,y,x nm ≈ {spacing}, "
        f"native crop {ch_need}, physical size nm {physical_size_nm_zyx}, output spacing {out_spacing_nm_zyx}, "
        f"max voxels {out_max_voxels_zyx}"
    )

    if window_policy == "mito":
        print(f"[WARN] {dataset_name}: mito policy N/A for unlabeled; using center")
    if offsets is not None:
        win_offsets = list(offsets)
    elif int(n_crops) > 1:
        win_offsets, possible = _farthest_offsets_non_overlapping(vol_shape, ch_need, int(n_crops))
        if len(win_offsets) < int(n_crops):
            print(
                f"[INFO] {dataset_name}: requested {n_crops} crops, selected {len(win_offsets)} "
                f"(non-overlap limit for crop {ch_need}; candidate starts={possible})"
            )
        print(f"[INFO] {dataset_name}: farthest non-overlap crops selected={len(win_offsets)}")
    else:
        win_offsets = [_center_offset(vol_shape, ch_need)]

    tag = dataset_name.replace("-", "_")
    planned_suffixes = _resolve_batch_coherent_suffixes(
        dataset_name=dataset_name,
        img_dir=images_dir,
        seg_dir=None,
        n_windows=len(win_offsets),
    )
    for i, offset in enumerate(win_offsets, start=1):
        win = _window_from_offset(vol_shape, offset, ch_need)
        if win is None:
            print(f"[WARN] {dataset_name} vol{i}: invalid window; skipped")
            continue
        (z0, y0, x0), (z1, y1, x1) = win
        img_crop = np.asarray(image_group[z0:z1, y0:y1, x0:x1])
        img_out = resample_crop_to_isotropic(
            img_crop,
            spacing,
            order=1,
            out_spacing_nm_zyx=out_spacing_nm_zyx,
            max_voxels_zyx=out_max_voxels_zyx,
        )
        suffix = planned_suffixes[i - 1] if i - 1 < len(planned_suffixes) else _crop_index_suffix(i, len(win_offsets))
        prefix = os.path.join(images_dir, f"{tag}{suffix}")
        write_h5_fn(f"{prefix}_im.h5", img_out, f"{tag}{suffix}_im")
