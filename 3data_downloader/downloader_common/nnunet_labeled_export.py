"""nnUNet NIfTI export and dataset.json sync shared by Stage-3 labeled downloaders.

OpenOrganelle and BossDB both write EM + label crops into
``data/nnUNet_raw/Dataset001_mito2`` using the same naming and this module so
preprocessing and ``dataset.json`` stay aligned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import nibabel as nib  # type: ignore
except ImportError:
    nib = None  # type: ignore


def write_h5(filename: str, data: Any, dataset: str = "main") -> None:
    """Backward-compatible name used by foundation hooks: writes nnUNet NIfTI, not HDF5.

    Expects *filename* ending with ``_im.h5`` or ``_seg.h5`` (legacy naming).
    """
    if nib is None:
        raise ImportError("nibabel is required to write .nii.gz outputs")
    p = Path(filename)
    name = p.name
    aff16 = np.eye(4, dtype=np.float32)
    aff16[0, 0] = 16.0
    aff16[1, 1] = 16.0
    aff16[2, 2] = 16.0
    if name.endswith("_im.h5"):
        out = p.with_name(name[:-6] + "_0000.nii.gz")
        arr = np.asarray(data, dtype=np.float32)
    elif name.endswith("_seg.h5"):
        out = p.with_name(name[:-7] + ".nii.gz")
        raw = np.asarray(data)
        try:
            from .generate_contour import preprocess_instance_to_bc  # noqa: PLC0415
        except Exception as exc:
            raise RuntimeError(
                "generate_contour dependency missing. Install scipy/cc3d as needed and ensure "
                "downloader_common.generate_contour is importable."
            ) from exc
        instance, bc = preprocess_instance_to_bc(raw, connectivity=6)
        if out.parent.name in ("labelsTr", "labelsTs"):
            inst_dir = out.parent.parent / f"{out.parent.name}-instance"
            inst_dir.mkdir(parents=True, exist_ok=True)
            stem = out.name[:-7] if out.name.endswith(".nii.gz") else out.stem
            inst_out = inst_dir / f"{stem}_instance.nii.gz"
            nib.save(
                nib.Nifti1Image(instance.astype(np.uint32, copy=False), affine=aff16),
                str(inst_out),
            )
        arr = bc.astype(np.uint8, copy=False)
    else:
        out = p.with_suffix(".nii.gz")
        arr = np.asarray(data, dtype=np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr, affine=aff16), str(out))


def sync_dataset_json(dataset_root: str | Path) -> Path:
    """Scan ``imagesTr`` / ``labelsTr`` / ``imagesTs`` and rewrite ``dataset.json``."""
    root = Path(dataset_root).resolve()
    images_tr = root / "imagesTr"
    images_ts = root / "imagesTs"
    labels_tr = root / "labelsTr"
    root.mkdir(parents=True, exist_ok=True)
    for p in (images_tr, images_ts, labels_tr):
        p.mkdir(parents=True, exist_ok=True)

    def _pairs_training() -> list[dict]:
        out = []
        for img in sorted(images_tr.glob("*_0000.nii.gz"), key=lambda x: x.name.lower()):
            base = img.name[:-12]
            lbl = labels_tr / f"{base}.nii.gz"
            if lbl.is_file():
                out.append({"image": f"./imagesTr/{img.name}", "label": f"./labelsTr/{lbl.name}"})
        return out

    test_entries = [
        f"./imagesTs/{p.name}"
        for p in sorted(images_ts.glob("*_0000.nii.gz"), key=lambda x: x.name.lower())
    ]
    training_entries = _pairs_training()
    dataset_json_path = root / "dataset.json"
    prev: dict = {}
    if dataset_json_path.is_file():
        try:
            prev = json.loads(dataset_json_path.read_text(encoding="utf-8"))
        except Exception:
            prev = {}

    data = dict(prev) if isinstance(prev, dict) else {}
    data["channel_names"] = {"0": "em"}
    data["labels"] = {"background": 0, "mitochondria": 1, "contour": 2}
    data["file_ending"] = ".nii.gz"
    data["name"] = str(data.get("name") or "Dataset001_mito2")
    data["description"] = str(data.get("description") or "MitoFoundation2 Stage-3 downloaded dataset")
    data["tensorImageSize"] = str(data.get("tensorImageSize") or "3D")
    data["numTraining"] = len(training_entries)
    data["numTest"] = len(test_entries)
    data["training"] = training_entries
    data["test"] = test_entries
    dataset_json_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return dataset_json_path
