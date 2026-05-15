#!/usr/bin/env python3
"""Postprocess border-contour predictions into instance labels via watershed."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed


def load_nii(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)
    return arr, img.affine


def save_nii(path: Path, arr: np.ndarray, affine: np.ndarray) -> None:
    out = nib.Nifti1Image(arr.astype(np.uint16), affine)
    nib.save(out, str(path))


def postprocess_one(pred: np.ndarray) -> np.ndarray:
    """Convert border-contour style prediction to instance IDs.

    Expected label convention:
    - 0: background
    - 1: foreground interior
    - 2: contour/border
    Any value >0 is treated as foreground candidate.
    """
    pred = np.asarray(pred)
    fg = pred > 0
    contour = pred == 2
    interior = fg & (~contour)

    markers, _ = ndi.label(interior)
    if markers.max() == 0:
        # Fallback for degenerate maps with no explicit interior.
        markers, _ = ndi.label(fg)

    dist = ndi.distance_transform_edt(fg)
    ws = watershed(-dist, markers=markers, mask=fg)
    return ws.astype(np.uint16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Directory with contour predictions (.nii/.nii.gz)")
    ap.add_argument("--output_dir", required=True, help="Directory for watershed instance outputs")
    args = ap.parse_args()

    in_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(list(in_dir.glob("*.nii.gz")) + list(in_dir.glob("*.nii")))
    print(f"[postprocess] input_dir={in_dir}")
    print(f"[postprocess] output_dir={out_dir}")
    print(f"[postprocess] found_files={len(files)}")
    if not files:
        print("[postprocess] no NIfTI files found")
        return 0

    ok = 0
    for p in files:
        try:
            arr, affine = load_nii(p)
            inst = postprocess_one(arr)
            out_path = out_dir / p.name
            save_nii(out_path, inst, affine)
            print(f"[postprocess] saved {out_path}")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[postprocess] failed {p}: {e}")
    print(f"[postprocess] completed={ok}/{len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

