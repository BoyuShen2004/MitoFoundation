from __future__ import annotations

import argparse
import os
from glob import glob
from pathlib import Path

import imageio.v2 as imageio
import nibabel as nib
import numpy as np
import tifffile as tiff
from tqdm import tqdm

try:
    import cc3d  # type: ignore
except Exception:
    cc3d = None

try:
    from scipy import ndimage as ndi  # type: ignore
except Exception:
    ndi = None

try:
    # Original MitoAnnotation implementation path (older pytorch-connectomics layout).
    from connectomics.data.utils.data_segmentation import seg_to_instance_bd  # type: ignore
except Exception:
    seg_to_instance_bd = None

try:
    from skimage.segmentation import find_boundaries  # type: ignore
except Exception:
    find_boundaries = None


def _smallest_uint_dtype(max_id: int) -> np.dtype:
    if max_id <= np.iinfo(np.uint8).max:
        return np.uint8
    if max_id <= np.iinfo(np.uint16).max:
        return np.uint16
    return np.uint32


def split_connected_components(labels: np.ndarray, *, connectivity: int = 6) -> np.ndarray:
    """Split each positive label into connected components with globally unique ids."""
    seg = np.asarray(labels)
    if seg.ndim != 3:
        raise ValueError(f"expected 3D labels, got shape {seg.shape}")
    if cc3d is not None:
        out = cc3d.connected_components(seg.astype(np.int64, copy=False), connectivity=int(connectivity))
        return out.astype(_smallest_uint_dtype(int(out.max())), copy=False)
    if ndi is None:
        raise RuntimeError("Need either cc3d or scipy.ndimage to split connected components.")
    out = np.zeros(seg.shape, dtype=np.uint32)
    next_id = 1
    struct = ndi.generate_binary_structure(3, 1 if int(connectivity) == 6 else 2)
    for val in np.unique(seg):
        if int(val) <= 0:
            continue
        comp, n = ndi.label(seg == val, structure=struct)
        for i in range(1, int(n) + 1):
            out[comp == i] = next_id
            next_id += 1
    return out.astype(_smallest_uint_dtype(int(out.max())), copy=False)


def generate_contour_map(mask: np.ndarray, width: int = 3) -> np.ndarray:
    """Mimic MitoAnnotation/generate_contour.py contour generation."""
    arr = np.asarray(mask)
    binary = (arr > 0).astype(np.uint8)
    squeeze_back = False
    if binary.ndim == 2:
        binary = binary[np.newaxis, ...]
        squeeze_back = True
    if seg_to_instance_bd is not None:
        contour = seg_to_instance_bd(binary, tsz_h=int(width))
    else:
        # Fallback when installed connectomics package layout does not expose
        # seg_to_instance_bd (observed in some pytorch-connectomics releases).
        if find_boundaries is not None:
            contour = find_boundaries(arr, mode="outer").astype(np.uint8)
        elif ndi is not None:
            dilated = ndi.binary_dilation(binary.astype(bool), structure=ndi.generate_binary_structure(binary.ndim, 1))
            contour = (dilated & (~binary.astype(bool))).astype(np.uint8)
        else:
            raise RuntimeError(
                "Contour generation needs connectomics.seg_to_instance_bd or skimage/scipy fallback dependencies."
            )
    if squeeze_back:
        contour = contour[0]
    return contour.astype(np.uint8)


def instance_to_bc_label(instance_labels: np.ndarray, *, contour_width: int = 3) -> np.ndarray:
    """Convert split instances to contour labels: background=0, interior=1, contour=2."""
    inst = np.asarray(instance_labels)
    contour = generate_contour_map(inst, width=int(contour_width))
    contour[contour > 0] = 2
    binary = (inst > 0).astype(np.uint8)
    saved_mask = binary + contour
    saved_mask[saved_mask > 2] = 1
    return saved_mask.astype(np.uint8, copy=False)


def preprocess_instance_to_bc(
    labels: np.ndarray, *, connectivity: int = 6, contour_width: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    instance = split_connected_components(labels, connectivity=connectivity)
    bc = instance_to_bc_label(instance, contour_width=contour_width)
    return instance, bc


def read_volume(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    p = Path(path)
    suffixes = p.suffixes
    if suffixes[-2:] == [".nii", ".gz"] or (suffixes and suffixes[-1].lower() == ".nii"):
        img = nib.load(str(p))
        return img.get_fdata().astype(np.uint16), img.affine
    if suffixes and suffixes[-1].lower() in (".tif", ".tiff"):
        return tiff.imread(str(p)).astype(np.uint16), None
    if suffixes and suffixes[-1].lower() == ".png":
        return imageio.imread(str(p)).astype(np.uint16), None
    raise ValueError(f"Unsupported file extension: {''.join(suffixes)}")


def write_volume(vol: np.ndarray, affine: np.ndarray | None, out_path: str) -> None:
    p = Path(out_path)
    suffixes = p.suffixes
    if suffixes[-2:] == [".nii", ".gz"] or (suffixes and suffixes[-1].lower() == ".nii"):
        header = None
        out_affine = affine
        if os.path.exists(out_path):
            try:
                existing = nib.load(out_path)
                header = existing.header
                out_affine = existing.affine
            except Exception:
                pass
        if out_affine is None:
            out_affine = np.eye(4, dtype=np.float32)
        nib.save(nib.Nifti1Image(vol.astype(np.uint8), out_affine, header), out_path)
        return
    if suffixes and suffixes[-1].lower() in (".tif", ".tiff"):
        tiff.imwrite(out_path, vol.astype(np.uint8), compression="zlib")
        return
    if suffixes and suffixes[-1].lower() == ".png":
        imageio.imwrite(out_path, vol.astype(np.uint8))
        return
    raise ValueError(f"Unsupported output extension: {''.join(suffixes)}")


def process_file(input_path: str, output_folder: str, width: int = 3) -> None:
    vol, affine = read_volume(input_path)
    contour = instance_to_bc_label(vol, contour_width=width)
    p = Path(input_path)
    suffixes = p.suffixes
    if suffixes[-2:] == [".nii", ".gz"]:
        base = p.name[:-7]
        ext = ".nii.gz"
    else:
        base = p.stem
        ext = suffixes[-1]
    out_path = os.path.join(output_folder, f"{base}{ext}")
    write_volume(contour, affine, out_path)


def process_folder(input_folder: str, output_folder: str, width: int = 3) -> None:
    os.makedirs(output_folder, exist_ok=True)
    files: list[str] = []
    for pat in ("/*.tif", "/*.tiff", "/*.nii", "/*.nii.gz", "/*.png"):
        files.extend(glob(input_folder + pat))
    for path in tqdm(files, desc="Processing masks"):
        process_file(path, output_folder, width=width)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate contour labels from masks.")
    parser.add_argument("-i", "--input_folder", required=True, help="Input folder")
    parser.add_argument("-o", "--output_folder", required=True, help="Output folder")
    parser.add_argument("-w", "--width", type=int, default=3, help="Contour width")
    args = parser.parse_args()
    process_folder(args.input_folder, args.output_folder, width=args.width)
