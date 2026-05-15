"""Stage-4 preprocessor: raw H5/NRRD/NIfTI → nnUNet-ready volumes.

Writes under ``data/nnUNet_raw/Dataset001_mito2`` (``imagesTr/``, ``labelsTr/``).
After a successful supervised/finetune run, updates ``5model_training/configuration/mito_openorganelle_foundation.yaml``
so ``pytorch_connectomics`` training points at the **training-ready** stacks and refreshes median ZYX **resolution** (nm)
and **patch / input / window** sizes from median voxel shapes (unless ``--skip-training-yaml``).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.ndimage import generate_binary_structure  # type: ignore
    from scipy.ndimage import label as cc_label  # type: ignore
except Exception:  # pragma: no cover
    generate_binary_structure = None  # type: ignore[assignment]
    cc_label = None  # type: ignore[assignment]

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None

try:
    import nrrd  # type: ignore
except Exception:  # pragma: no cover
    nrrd = None

try:
    import nibabel as nib  # type: ignore
except Exception:  # pragma: no cover
    nib = None

import sys

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
_dl_common_root = _repo_root / "3data_downloader"
if str(_dl_common_root) not in sys.path:
    sys.path.insert(0, str(_dl_common_root))
from downloader_common.generate_contour import preprocess_instance_to_bc

try:
    from .training_yaml_sync import sync_mito_openorganelle_foundation_yaml
except ImportError:  # e.g. ``python master/agent.py`` without package context
    _pkg_root = Path(__file__).resolve().parent.parent
    if str(_pkg_root) not in sys.path:
        sys.path.insert(0, str(_pkg_root))
    try:
        from downloader_master.training_yaml_sync import sync_mito_openorganelle_foundation_yaml
    except ImportError:
        _local_dir = Path(__file__).resolve().parent
        if str(_local_dir) not in sys.path:
            sys.path.insert(0, str(_local_dir))
        from training_yaml_sync import sync_mito_openorganelle_foundation_yaml


def _default_project_root() -> Path:
    from config.paths import project_root

    return project_root()


def _strip_ext(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    return path.stem


def _canonical_key(path: Path) -> str:
    name = _strip_ext(path)
    name = re.sub(r"_(im|seg)$", "", name, flags=re.I)
    name = re.sub(r"\.(im|seg)$", "", name, flags=re.I)
    return name


def _float_list_attr(raw: Any) -> list[float]:
    """Coerce HDF5 / JSON-like attribute payloads to a flat float list (best-effort)."""
    vals: list[float] = []
    if raw is None:
        return vals
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if isinstance(raw, (list, tuple)):
        for x in raw:
            if isinstance(x, (list, tuple)):
                continue
            try:
                vals.append(float(x))
            except Exception:
                pass
        return [v for v in vals if v == v]
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8", errors="ignore")
        except Exception:
            return []
    if isinstance(raw, str):
        txt = raw.strip().strip("[]()")
        for part in txt.split(","):
            p = part.strip()
            if not p:
                continue
            try:
                vals.append(float(p))
            except Exception:
                pass
        return [v for v in vals if v == v]
    try:
        return [float(raw)] if raw == raw else []
    except Exception:
        return []


def _vector3_positive_spacing_attr(raw_any: Any) -> list[float]:
    vals = _float_list_attr(raw_any)
    if len(vals) >= 3 and all(v > 0 for v in vals[:3]):
        return [float(vals[0]), float(vals[1]), float(vals[2])]
    if isinstance(raw_any, (bytes, bytearray)):
        try:
            raw_any = raw_any.decode("utf-8", errors="ignore")
        except Exception:
            return []
    if isinstance(raw_any, str):
        txt = raw_any.strip()
        for fn in (json.loads, ast.literal_eval):
            try:
                parsed = fn(txt)
            except Exception:
                continue
            vals = _float_list_attr(parsed)
            if len(vals) >= 3 and all(v > 0 for v in vals[:3]):
                return [float(vals[0]), float(vals[1]), float(vals[2])]
    return []


def _spacing_from_h5_attr_dict(attrs: dict[str, Any]) -> list[float]:
    """Physical voxel size (Z, Y, X) in nm from merged HDF5 attributes (matches Studio inspect heuristics)."""
    spacing_keys = (
        "spacing",
        "voxel_size",
        "voxel_size_nm",
        "resolution",
        "element_size_um",
        "pixel_size",
        "voxel_size_nm_zyx",
        "out_spacing_nm",
        "out_spacing_nm_zyx",
    )
    for k in spacing_keys:
        if k not in attrs:
            continue
        v = _vector3_positive_spacing_attr(attrs[k])
        if v:
            if k == "element_size_um":
                return [float(v[0]) * 1000.0, float(v[1]) * 1000.0, float(v[2]) * 1000.0]
            return v
    for ak, av in attrs.items():
        lk = str(ak).lower()
        if any(
            s in lk
            for s in ("voxel_size", "voxel", "spacing_nm", "resolution", "element_size", "pixel_size")
        ):
            got = _vector3_positive_spacing_attr(av)
            if got:
                if "element_size_um" in lk or "voxel_size_um" in lk:
                    return [float(got[0]) * 1000.0, float(got[1]) * 1000.0, float(got[2]) * 1000.0]
                return got
    return []


def _h5_list_all_datasets(h5f: Any) -> list[tuple[str, Any]]:
    if h5py is None:
        return []

    out: list[tuple[str, Any]] = []

    def walk(grp: Any, prefix: str) -> None:
        for k in grp.keys():
            obj = grp[k]
            full = f"{prefix}/{k}" if prefix else k
            if isinstance(obj, h5py.Dataset):
                out.append((full, obj))
            elif isinstance(obj, h5py.Group):
                walk(obj, full)

    walk(h5f, "")
    return out


def _h5_ranked_3d_datasets(items: list[tuple[str, Any]]) -> list[tuple[int, str, Any]]:
    ranked: list[tuple[int, str, Any]] = []
    for full, ds in items:
        if not hasattr(ds, "ndim") or int(ds.ndim) < 3:
            continue
        try:
            n = int(np.prod(np.array(ds.shape[:3], dtype=np.int64)))
        except Exception:
            continue
        ranked.append((n, full, ds))
    ranked.sort(key=lambda t: -t[0])
    return ranked


def _h5_pick_volume_dataset(
    name_l: str,
    ranked: list[tuple[int, str, Any]],
    *,
    path_for_hints: Path | None = None,
) -> Any | None:
    """Prefer EM ``_im`` internal paths for image stacks; prefer ``_seg`` under ``labels/`` or ``*_seg`` names."""
    if not ranked:
        return None
    pl = str(path_for_hints).replace("\\", "/").lower() if path_for_hints is not None else ""
    under_labels = "/labels/" in pl
    treat_as_seg = ("_seg" in name_l) or under_labels
    if treat_as_seg:
        for _n, full, ds in ranked:
            if "_seg" in full.lower():
                return ds
        return ranked[0][2]
    for _n, full, ds in ranked:
        if "_im" in full.lower():
            return ds
    return ranked[0][2]


def _merge_h5_attrs_for_spacing(h5f: Any, items: list[tuple[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    merged.update(dict(h5f.attrs.items()))
    for _full, dso in items:
        attrs_obj = getattr(dso, "attrs", None)
        if attrs_obj is not None:
            merged.update(dict(attrs_obj.items()))
    return merged


def _resolve_spacing_nm_zyx_from_h5(path: Path, merged_attrs: dict[str, Any], *, warn_fallback: bool) -> list[float]:
    """Preserve native H5 spacing metadata for preprocess outputs.

    Stage-4 preprocessing should not reinterpret spacing from external catalogs or sidecars for H5 inputs.
    """
    sp = _spacing_from_h5_attr_dict(merged_attrs)
    if len(sp) == 3 and all(v > 0 for v in sp):
        return [float(sp[0]), float(sp[1]), float(sp[2])]
    # OpenOrganelle downloader H5 stacks often omit voxel attrs entirely.
    # Keep preprocess output consistent with stage-4 raw inspect fallback.
    p_l = str(path).replace("\\", "/").lower()
    if (
        "openorganelle_mito_" in p_l
        or "/data/nnunet_raw/dataset001_mito2/" in p_l
        or ("/data/raw/" in p_l and (p_l.endswith("_im.h5") or p_l.endswith("_seg.h5")))
    ):
        if warn_fallback:
            print(f"[INFO] spacing for {path.name}: using OpenOrganelle fallback [16, 16, 16] nm")
        return [16.0, 16.0, 16.0]
    if warn_fallback:
        print(
            f"[WARN] No Z,Y,X voxel spacing (nm) found in H5 attrs for {path}; "
            "using placeholder [1,1,1]."
        )
    return [1.0, 1.0, 1.0]


_OPENORGANELLE_VOXEL_DB_CACHE: dict[str, list[float]] | None = None


def _norm_dataset_token(raw: str) -> str:
    txt = re.sub(r"[^a-z0-9]+", "_", str(raw).strip().lower())
    return re.sub(r"_+", "_", txt).strip("_")


def _openorganelle_db_candidates() -> list[Path]:
    root = _default_project_root()
    return [root / "2database_builder" / "outputs" / "databases" / "OpenOrganelle.db"]


def _load_openorganelle_voxel_db() -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    db_path = next((p for p in _openorganelle_db_candidates() if p.is_file()), None)
    if db_path is None:
        return out
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT d.dataset_name AS dataset_name, r.resolved_voxel_nm AS resolved_voxel_nm
            FROM dataset_resolved r
            JOIN datasets d ON d.id = r.dataset_id
            """
        ).fetchall()
        conn.close()
    except Exception:
        return out
    for row in rows:
        name = str(row["dataset_name"] or "").strip()
        raw_voxel = row["resolved_voxel_nm"]
        if not name or not raw_voxel:
            continue
        try:
            vals = json.loads(str(raw_voxel))
        except Exception:
            continue
        if not isinstance(vals, list) or len(vals) < 3:
            continue
        try:
            zyx = [float(vals[0]), float(vals[1]), float(vals[2])]
        except Exception:
            continue
        if all(v > 0 for v in zyx):
            out[_norm_dataset_token(name)] = zyx
    return out


def _spacing_from_openorganelle_catalog(path: Path) -> list[float]:
    global _OPENORGANELLE_VOXEL_DB_CACHE
    if _OPENORGANELLE_VOXEL_DB_CACHE is None:
        _OPENORGANELLE_VOXEL_DB_CACHE = _load_openorganelle_voxel_db()
    if not _OPENORGANELLE_VOXEL_DB_CACHE:
        return []
    key = _norm_dataset_token(_canonical_key(path))
    probe = key
    while probe:
        got = _OPENORGANELLE_VOXEL_DB_CACHE.get(probe)
        if got and len(got) == 3:
            return [float(got[0]), float(got[1]), float(got[2])]
        if "_" not in probe:
            break
        tail = probe.rsplit("_", 1)[-1]
        if not tail.isdigit():
            break
        probe = probe.rsplit("_", 1)[0]
    return []


def _spacing_from_registry(path: Path, project_root: Path | None = None) -> list[float]:
    """Provider-agnostic spacing lookup via the central registry.

    Tries each registered provider's stored metadata (voxel_size_nm) for the
    dataset key derived from the file path. Falls back to OpenOrganelle catalog DB.
    Returns [] when no spacing found — caller should fall back further.
    """
    try:
        root = project_root or _default_project_root()
        import sys as _sys
        if str(root) not in _sys.path:
            _sys.path.insert(0, str(root))
        from agent.orchestration.registry.schema import open_registry as _open_reg
        from agent.orchestration.registry.api import get_dataset_spacing_nm as _get_sp, get_provider_id as _get_prov
        conn = _open_reg()
        key = _canonical_key(path)
        # Try all providers
        providers = conn.execute("SELECT id FROM providers").fetchall()
        for prow in providers:
            sp = _get_sp(conn, prow["id"], key)
            if sp:
                conn.close()
                return sp
        conn.close()
    except Exception:
        pass
    # Fallback to existing OpenOrganelle catalog DB
    return _spacing_from_openorganelle_catalog(path)


def _supported_raw_files(root: Path) -> list[Path]:
    # Intentionally omit ``data/raw``: OpenOrganelle + Studio use Dataset001_mito2 directly.
    # Legacy raw batches can still be passed via ``--dataset-path`` / ``--raw-run-subfolder``.
    from config.paths import nnunet_dataset_root

    nn = nnunet_dataset_root(root)
    candidates = [
        nn / "imagesTr",
        nn / "labelsTr",
        nn / "imagesTs",
        nn / "labelsTs",
        root / "data" / "data_unlabeled" / "data_raw",
        root / "data" / "data_labeled" / "data_raw",
    ]
    out: list[Path] = []
    exts = (".h5", ".nrrd", ".nii", ".nii.gz")
    for base in candidates:
        if not base.is_dir():
            continue
        for p in base.glob("**/*"):
            if p.is_file() and any(p.name.endswith(x) for x in exts):
                out.append(p.resolve())
    # Stable unique list
    uniq = sorted({str(p): p for p in out}.values(), key=lambda p: str(p).lower())
    return uniq


def _load_h5(path: Path) -> tuple[np.ndarray, list[float]]:
    if h5py is None:
        raise RuntimeError("h5py is required to read .h5 downloader outputs")
    with h5py.File(path, "r") as f:
        items = _h5_list_all_datasets(f)
        ranked = _h5_ranked_3d_datasets(items)
        name_l = path.name.lower()
        ds = _h5_pick_volume_dataset(name_l, ranked, path_for_hints=path)
        if ds is None:
            raise RuntimeError(f"No 3D dataset found in H5 file: {path}")
        arr = np.asarray(ds[:])
        merged = _merge_h5_attrs_for_spacing(f, items)
        spacing = _resolve_spacing_nm_zyx_from_h5(path, merged, warn_fallback=True)
    return arr, spacing


def _load_nrrd(path: Path) -> tuple[np.ndarray, list[float]]:
    if nrrd is None:
        raise RuntimeError("pynrrd is required to read .nrrd downloader outputs")
    arr, hdr = nrrd.read(str(path))
    spacing = _spacing_from_nrrd_header(dict(hdr))
    return np.asarray(arr), spacing


def _load_nii(path: Path) -> tuple[np.ndarray, list[float]]:
    if nib is None:
        raise RuntimeError("nibabel is required to read .nii/.nii.gz")
    img = nib.load(str(path))
    arr = np.asarray(img.get_fdata())
    zyx = list(reversed([float(v) for v in img.header.get_zooms()[:3]]))
    spacing = zyx if len(zyx) == 3 and all(v > 0 for v in zyx) else [1.0, 1.0, 1.0]
    return arr, spacing


def _load_volume(path: Path) -> tuple[np.ndarray, list[float]]:
    if path.name.endswith(".h5"):
        return _load_h5(path)
    if path.name.endswith(".nrrd"):
        return _load_nrrd(path)
    if path.name.endswith(".nii") or path.name.endswith(".nii.gz"):
        return _load_nii(path)
    raise RuntimeError(f"Unsupported volume format: {path}")


def _spacing_from_nrrd_header(hdr: dict[str, Any]) -> list[float]:
    spacing = [1.0, 1.0, 1.0]
    spacings = hdr.get("spacings")
    if isinstance(spacings, (list, tuple, np.ndarray)) and len(spacings) >= 3:
        vals = [float(spacings[i]) for i in range(3)]
        if all(v > 0 for v in vals):
            spacing = vals
    elif "space directions" in hdr:
        dirs = hdr.get("space directions")
        if isinstance(dirs, (list, tuple)) and len(dirs) >= 3:
            try:
                spacing = [max(abs(float(x)) for x in dirs[i]) for i in range(3)]
            except Exception:
                pass
    return spacing


def peek_volume_metadata(path: Path) -> dict[str, Any]:
    """Read shape, dtype, and spacing without loading the full volume array (best-effort)."""
    path = Path(path).resolve()
    row: dict[str, Any] = {
        "file": path.name,
        "path": str(path),
        "format": "",
        "shape": "",
        "dtype": "",
        "spacing_zyx": "",
        "error": "",
    }
    try:
        if path.name.endswith(".h5"):
            row["format"] = "h5"
            if h5py is None:
                raise RuntimeError("h5py not installed")
            with h5py.File(path, "r") as f:
                items = _h5_list_all_datasets(f)
                ranked = _h5_ranked_3d_datasets(items)
                ds = _h5_pick_volume_dataset(path.name.lower(), ranked, path_for_hints=path)
                if ds is None:
                    raise RuntimeError("no 3D datasets")
                merged = _merge_h5_attrs_for_spacing(f, items)
                shp = tuple(int(x) for x in ds.shape)
                row["shape"] = str(shp)
                row["dtype"] = str(ds.dtype)
                sp = _resolve_spacing_nm_zyx_from_h5(path, merged, warn_fallback=False)
                row["spacing_zyx"] = str([round(float(x), 6) for x in sp])
        elif path.name.endswith(".nrrd"):
            row["format"] = "nrrd"
            if nrrd is None:
                raise RuntimeError("pynrrd not installed")
            hdr = nrrd.read_header(str(path))
            sizes = hdr.get("sizes")
            if sizes is None:
                raise RuntimeError("NRRD header missing sizes")
            shp = tuple(int(x) for x in np.asarray(sizes).astype(np.int64).ravel())
            row["shape"] = str(shp)
            row["dtype"] = str(hdr.get("type", ""))
            sp = _spacing_from_nrrd_header(dict(hdr))
            row["spacing_zyx"] = str([round(float(x), 6) for x in sp])
        elif path.name.endswith(".nii") or path.name.endswith(".nii.gz"):
            row["format"] = "nii" if path.name.endswith(".nii") else "nii.gz"
            if nib is None:
                raise RuntimeError("nibabel not installed")
            img = nib.load(str(path))
            shp = tuple(int(x) for x in img.shape)
            row["shape"] = str(shp)
            row["dtype"] = str(np.dtype(img.get_data_dtype()))
            zooms = [float(v) for v in img.header.get_zooms()[: min(3, img.ndim)]]
            while len(zooms) < 3:
                zooms.append(1.0)
            zyx = list(reversed(zooms[:3]))
            if not all(v > 0 for v in zyx):
                zyx = [1.0, 1.0, 1.0]
            row["spacing_zyx"] = str([round(float(x), 6) for x in zyx])
        else:
            row["error"] = "unsupported extension"
    except Exception as exc:  # pragma: no cover
        row["error"] = str(exc)
    return row


def list_volume_files_under(folder: Path) -> list[Path]:
    """All .h5 / .nrrd / .nii / .nii.gz under ``folder`` (recursive), sorted."""
    root = Path(folder).resolve()
    if not root.is_dir():
        return []
    filtered: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if name.endswith(".nii.gz") or name.endswith(".h5") or name.endswith(".nrrd") or name.endswith(".nii"):
            filtered.append(p.resolve())
    return sorted({str(p): p for p in filtered}.values(), key=lambda p: str(p).lower())


def raw_data_root(project_root: Path | None = None) -> Path:
    root = Path(project_root).resolve() if project_root else _default_project_root()
    return (root / "data" / "raw").resolve()


def list_immediate_subfolders_of_raw(project_root: Path | None = None) -> list[Path]:
    raw = raw_data_root(project_root)
    if not raw.is_dir():
        return []
    return sorted((p for p in raw.iterdir() if p.is_dir()), key=lambda p: p.name.lower())


def _relabel_disconnected_components(seg: np.ndarray) -> np.ndarray:
    """Each connected component of every positive label value gets its own integer id (0 = background)."""
    if cc_label is None or generate_binary_structure is None:
        raise RuntimeError("scipy is required for connected-component label splitting")
    seg = np.asarray(seg)
    ndim = int(seg.ndim)
    if ndim < 2:
        return np.asarray(seg, dtype=np.int32)
    # Use face-connectivity in 3D (6-neighborhood) so diagonally touching
    # structures are treated as separate components.
    if ndim >= 3:
        struct = generate_binary_structure(3, 1)
    else:
        struct = generate_binary_structure(ndim, 1)
    out = np.zeros(seg.shape, dtype=np.int32)
    next_id = 1
    for val in sorted(int(x) for x in np.unique(seg).tolist()):
        if val == 0:
            continue
        mask = seg == val
        if not np.any(mask):
            continue
        labeled, nfeat = cc_label(mask.astype(np.uint8, copy=False), structure=struct)
        for comp_idx in range(1, int(nfeat) + 1):
            m = labeled == comp_idx
            out[m] = next_id
            next_id += 1
    return out


def _write_h5_volume(
    path: Path,
    arr: np.ndarray,
    spacing_nm_zyx: list[float],
    *,
    dataset_name: str,
    split_label_cc: bool = False,
    is_label: bool = False,
) -> None:
    if h5py is None:
        raise RuntimeError("h5py is required to write .h5 outputs")
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr)
    safe_name = re.sub(r"[^\w.\-]", "_", dataset_name).strip("._") or "main"
    voxel = np.asarray(
        [float(spacing_nm_zyx[0]), float(spacing_nm_zyx[1]), float(spacing_nm_zyx[2])],
        dtype=np.float64,
    )
    with h5py.File(path, "w") as f:
        if np.issubdtype(arr.dtype, np.integer) or arr.dtype == bool:
            dset_arr = np.asarray(arr, dtype=np.int32)
        else:
            dset_arr = np.asarray(arr, dtype=np.float32)
        ds = f.create_dataset(safe_name, data=dset_arr, compression="lzf")
        ds.attrs["voxel_size_nm"] = voxel
        ds.attrs["mito2_preprocessed"] = 1
        ds.attrs["mito2_split_label_cc"] = 1 if (is_label and split_label_cc) else 0
        f.attrs["voxel_size_nm"] = voxel
        f.attrs["mito2_preprocessed"] = 1
        f.attrs["mito2_split_label_cc"] = 1 if (is_label and split_label_cc) else 0


def _h5_has_split_label_cc_marker(path: Path) -> bool:
    """True iff H5 root or first dataset has mito2_split_label_cc == 1."""
    if h5py is None:
        return False
    p = Path(path).resolve()
    if not p.is_file():
        return False
    try:
        with h5py.File(p, "r") as f:
            root_marker = f.attrs.get("mito2_split_label_cc", 0)
            try:
                if int(root_marker) == 1:
                    return True
            except Exception:
                pass
            values = list(f.values())
            if values:
                d0 = values[0]
                ds_marker = d0.attrs.get("mito2_split_label_cc", 0)
                try:
                    return int(ds_marker) == 1
                except Exception:
                    return False
    except Exception:
        return False
    return False


def _write_nifti_zyx(path: Path, arr_zyx: np.ndarray, spacing_nm_zyx: list[float]) -> None:
    if nib is None:
        raise RuntimeError("nibabel is required to write .nii.gz outputs")
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr_zyx)
    # NIfTI convention commonly treats array as x,y,z; we store data with axis order (z,y,x)
    # and encode matching voxel spacing on diagonal for consistency with this pipeline.
    affine = np.eye(4, dtype=np.float32)
    affine[0, 0] = float(spacing_nm_zyx[0])
    affine[1, 1] = float(spacing_nm_zyx[1])
    affine[2, 2] = float(spacing_nm_zyx[2])
    img = nib.Nifti1Image(arr.astype(np.float32, copy=False), affine=affine)
    nib.save(img, str(path))


def _dedupe_finetune_entries(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stable dedupe by (image,label) so metadata never inflates from repeats."""
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for it in items:
        img = str(it.get("image") or "").strip()
        lbl = str(it.get("label") or "").strip()
        if not img or not lbl:
            continue
        key = (img, lbl)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def _split_train_val_test(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Create non-overlapping splits (never duplicate same sample across splits)."""
    items = _dedupe_finetune_entries(items)
    if not items:
        return {"training": [], "validation": [], "test": []}
    n = len(items)
    if n < 3:
        # Keep tiny runs compact: avoid duplicating rows into val/test.
        return {"training": list(items), "validation": [], "test": []}
    n_train = max(1, int(round(n * 0.8)))
    n_val = int(round(n * 0.1))
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)
    training = items[:n_train]
    validation = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]
    return {"training": training, "validation": validation, "test": test}


def _sanitize_finetune_payload(payload: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """Guarantee compact, non-overlapping JSON payload with existing paths only."""
    out: dict[str, list[dict[str, Any]]] = {"training": [], "validation": [], "test": []}
    seen: set[tuple[str, str]] = set()
    for split in ("training", "validation", "test"):
        rows = payload.get(split, [])
        if not isinstance(rows, list):
            continue
        clean_rows: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            img = str(row.get("image") or "").strip()
            seg = str(row.get("label") or "").strip()
            if not img or not seg:
                continue
            if not (Path(img).is_file() and Path(seg).is_file()):
                continue
            key = (img, seg)
            if key in seen:
                continue
            seen.add(key)
            clean_rows.append(row)
        out[split] = clean_rows
    return out


def _classify_files(files: list[Path]) -> tuple[list[Path], list[Path]]:
    imgs: list[Path] = []
    segs: list[Path] = []
    for p in files:
        n = _strip_ext(p).lower()
        if n.endswith("_seg") or "/labels/" in str(p).replace("\\", "/").lower():
            segs.append(p)
        elif n.endswith("_im") or "/images/" in str(p).replace("\\", "/").lower():
            imgs.append(p)
        else:
            # Treat unknown as potential image (for SSL tomograms)
            imgs.append(p)
    return imgs, segs


def _pair_images_labels(images: list[Path], labels: list[Path]) -> list[tuple[Path, Path]]:
    lbl_by_key = {_canonical_key(p): p for p in labels}
    pairs: list[tuple[Path, Path]] = []
    for img in images:
        key = _canonical_key(img)
        seg = lbl_by_key.get(key)
        if seg is not None:
            pairs.append((img, seg))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="mitoFoundation2 — stage 4 data preprocessor")
    parser.add_argument("--project-root", type=Path, default=_default_project_root(), help="Repo root")
    parser.add_argument("--task", choices=("finetune", "supervised"), default="supervised")
    parser.add_argument("--dataset-path", action="append", default=[], help="Specific raw file path(s) to preprocess")
    parser.add_argument("--dataset-paths-json", type=str, default="", help="JSON array of raw file paths")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: <project>/data/nnUNet_raw/Dataset001_mito2",
    )
    parser.add_argument(
        "--output-format",
        choices=("nifti", "h5"),
        default="h5",
        help="Volume format under Dataset001_mito2 (default h5).",
    )
    parser.add_argument(
        "--split-label-cc",
        action="store_true",
        help="Relabel segmentations so each disconnected component of every positive class gets a unique id.",
    )
    parser.add_argument(
        "--raw-run-subfolder",
        default="",
        help="If set, also include every <project>/data/raw/<name>/images/*_im.h5 or *_0000.nii.gz (after explicit paths).",
    )
    parser.add_argument(
        "--skip-training-yaml",
        action="store_true",
        help="Do not auto-update 5model_training/configuration/mito_openorganelle_foundation.yaml.",
    )
    parser.add_argument(
        "--training-yaml",
        type=Path,
        default=None,
        help="Override Hydra YAML path (default: MITO_TRAIN_CONFIG_YAML or mito_openorganelle_foundation.yaml).",
    )
    parser.add_argument(
        "--no-registry",
        action="store_true",
        help="Disable registry caching for this run (overrides default-on; or MITO2_DISABLE_REGISTRY=1).",
    )
    parser.add_argument(
        "--use-registry",
        action="store_true",
        help="(Kept for backward compat — registry caching is on by default unless MITO2_DISABLE_REGISTRY=1).",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    out_root = (
        args.output_root.resolve()
        if args.output_root
        else nnunet_dataset_root(root)
    )
    ft_img_out = out_root / "imagesTr"
    ft_lbl_out = out_root / "labelsTr"
    if args.task in ("finetune", "supervised"):
        ft_img_out.mkdir(parents=True, exist_ok=True)
        ft_lbl_out.mkdir(parents=True, exist_ok=True)

    selected: list[Path] = []
    if args.dataset_paths_json.strip():
        raw = json.loads(args.dataset_paths_json)
        if isinstance(raw, list):
            selected.extend(Path(str(x)).resolve() for x in raw)
    selected.extend(Path(x).resolve() for x in args.dataset_path)
    rf = (args.raw_run_subfolder or "").strip()
    if rf:
        run_dir = (root / "data" / "raw" / rf).resolve()
        if run_dir.is_dir():
            for im in sorted(run_dir.glob("images/*"), key=lambda p: str(p).lower()):
                if not im.is_file():
                    continue
                n = im.name.lower()
                if n.endswith("_im.h5") or n.endswith("_0000.nii.gz"):
                    selected.append(im.resolve())
        else:
            print(f"[WARN] --raw-run-subfolder not found on disk: {run_dir}")
    selected = sorted({str(p): p for p in selected}.values(), key=lambda p: str(p).lower())

    discovered = _supported_raw_files(root)
    if selected:
        allow = {str(p.resolve()) for p in discovered} | {str(p.resolve()) for p in selected}
        selected = [p for p in selected if str(p.resolve()) in allow]
        raw_files = sorted({str(p): p for p in selected}.values(), key=lambda p: str(p).lower())
    else:
        raw_files = discovered

    images, labels = _classify_files(raw_files)
    pairs = _pair_images_labels(images, labels)
    out_fmt = str(args.output_format).strip().lower()
    use_h5 = out_fmt == "h5"
    split_cc = bool(args.split_label_cc)
    if use_h5 and h5py is None:
        raise SystemExit("HDF5 output requires h5py.")
    if split_cc and (cc_label is None or generate_binary_structure is None):
        raise SystemExit("Connected-component label splitting requires scipy (pip install scipy).")

    _reg_disabled = (
        os.environ.get("MITO2_DISABLE_REGISTRY", "").strip().lower() in ("1", "true", "yes", "on")
        or getattr(args, "no_registry", False)
    )
    use_registry = not _reg_disabled
    # Policy: when split-label-cc is requested, always rewrite outputs instead of
    # trusting preprocess cache entries. This guarantees every label in this run
    # actually goes through disconnected-component relabeling and marker write.
    if split_cc and use_registry:
        print("[REGISTRY] split_label_cc enabled: bypassing preprocess cache for this run", flush=True)
        use_registry = False
    _registry_conn: Any = None
    _registry_config_fp: str = ""
    _registry_provider_id: int | None = None
    if use_registry:
        try:
            import sys as _sys
            _rr = root
            if str(_rr) not in _sys.path:
                _sys.path.insert(0, str(_rr))
            from agent.orchestration.registry.schema import open_registry as _open_registry
            from agent.orchestration.registry.api import (
                ensure_preprocess_config as _ensure_pp_cfg,
                find_complete_preprocess_run as _find_pp_run,
                record_preprocess_start as _pp_start,
                record_preprocess_complete as _pp_done,
                record_preprocess_failed as _pp_fail,
                fingerprint_file as _fp_file,
                get_provider_id as _get_prov_id,
            )
            _registry_conn = _open_registry()
            _pp_config = {
                "task": args.task,
                "output_format": args.output_format,
                "split_label_cc": bool(args.split_label_cc),
            }
            _registry_config_fp = _ensure_pp_cfg(_registry_conn, _pp_config)
            _registry_conn.commit()
            _registry_provider_id = _get_prov_id(_registry_conn, "OpenOrganelle")
        except Exception as _exc:
            print(f"[REGISTRY] Warning: could not open registry for preprocess caching: {_exc}")
            use_registry = False

    finetune_entries_flat: list[dict[str, Any]] = []
    finetune_skipped = 0
    if args.task in ("finetune", "supervised"):
        n_pairs = len(pairs)
        if n_pairs:
            print(f"- Datasets: {n_pairs}", flush=True)
        for i, (img_src, seg_src) in enumerate(pairs, start=1):
            key = _canonical_key(img_src)
            print(f"[PROGRESS] dataset {i}/{n_pairs}: {key}", flush=True)

            # Registry cache check
            _pp_run_id: int | None = None
            if use_registry and _registry_conn is not None:
                try:
                    _input_fp = _fp_file(img_src)
                    _cached = _find_pp_run(_registry_conn, _input_fp, _registry_config_fp)
                    if _cached is not None:
                        import json as _json
                        _out_paths = _json.loads(_cached["output_paths_json"] or "[]")
                        if _out_paths and all(Path(p).is_file() for p in _out_paths):
                            print(f"[REGISTRY] Skipping {key}: preprocess run cached", flush=True)
                            _entry_img = next((p for p in _out_paths if "_im" in p or "image" in p), _out_paths[0])
                            _entry_seg = next((p for p in _out_paths if "_seg" in p or "label" in p), "")
                            if split_cc and _entry_seg and not _h5_has_split_label_cc_marker(Path(_entry_seg)):
                                print(
                                    f"[REGISTRY] Cache stale for {key}: label missing mito2_split_label_cc=1 marker; "
                                    "reprocessing with split_label_cc enabled",
                                    flush=True,
                                )
                            elif split_cc and not _entry_seg:
                                print(
                                    f"[REGISTRY] Cache stale for {key}: could not locate cached label output; "
                                    "reprocessing with split_label_cc enabled",
                                    flush=True,
                                )
                            else:
                                if _entry_img and _entry_seg:
                                    img_shape_c: list[int] = [0, 0, 0]
                                    _cached_spacing = [8.0, 8.0, 8.0]
                                    try:
                                        import h5py as _h5
                                        with _h5.File(_entry_img, "r") as _hf:
                                            _all_items = list(_hf.values())
                                            if _all_items:
                                                _ds = _all_items[0]
                                                img_shape_c = [int(_ds.shape[d]) for d in range(min(3, _ds.ndim))]
                                            _sp_raw = _hf.attrs.get("voxel_size_nm") or _hf.attrs.get("spacing")
                                            if _sp_raw is not None:
                                                try:
                                                    _sp_vals = [float(v) for v in list(_sp_raw)[:3]]
                                                    if len(_sp_vals) == 3 and all(v > 0 for v in _sp_vals):
                                                        _cached_spacing = _sp_vals
                                                except Exception:
                                                    pass
                                    except Exception:
                                        pass
                                    finetune_entries_flat.append({
                                        "image": _entry_img,
                                        "label": _entry_seg,
                                        "resolution": _cached_spacing,
                                        "shape_zyx": img_shape_c,
                                    })
                                continue
                    _ds_id = None
                    if _registry_provider_id is not None:
                        from agent.orchestration.registry.api import get_dataset_id as _get_ds_id
                        _ds_id = _get_ds_id(_registry_conn, _registry_provider_id, key)
                    _pp_run_id = _pp_start(_registry_conn, dataset_id=_ds_id,
                                           input_fingerprint=_input_fp,
                                           config_fingerprint=_registry_config_fp)
                    _registry_conn.commit()
                except Exception as _exc:
                    print(f"[REGISTRY] Warning: cache check failed for {key}: {_exc}", flush=True)

            try:
                img_arr, img_spacing = _load_volume(img_src)
                seg_arr, _ = _load_volume(seg_src)
                if split_cc:
                    seg_arr = _relabel_disconnected_components(seg_arr)
            except Exception as exc:
                finetune_skipped += 1
                if finetune_skipped <= 20:
                    print(f"[WARN] skip finetune pair {img_src} / {seg_src}: {exc}", flush=True)
                if use_registry and _pp_run_id is not None and _registry_conn is not None:
                    try:
                        _pp_fail(_registry_conn, _pp_run_id, str(exc))
                        _registry_conn.commit()
                    except Exception:
                        pass
                continue
            if use_h5:
                # Keep Stage 3+preprocess outputs in the same canonical suffix format
                # used by downloader outputs to avoid duplicate <key>.h5 shadow files.
                img_dst = ft_img_out / f"{key}_im.h5"
                seg_dst = ft_lbl_out / f"{key}_seg.h5"
                _write_h5_volume(
                    img_dst,
                    img_arr.astype(np.float32, copy=False),
                    img_spacing,
                    dataset_name=f"{key}_im",
                    split_label_cc=False,
                    is_label=False,
                )
                _write_h5_volume(
                    seg_dst,
                    seg_arr,
                    img_spacing,
                    dataset_name=f"{key}_seg",
                    split_label_cc=split_cc,
                    is_label=True,
                )
            else:
                img_dst = ft_img_out / f"{key}.nii.gz"
                seg_dst = ft_lbl_out / f"{key}.nii.gz"
                inst_dst = out_root / "labelsTr-instance" / f"{key}_instance.nii.gz"
                inst_dst.parent.mkdir(parents=True, exist_ok=True)
                seg_instance, seg_bc = preprocess_instance_to_bc(seg_arr, connectivity=6, contour_width=3)
                _write_nifti_zyx(img_dst, img_arr.astype(np.float32, copy=False), img_spacing)
                _write_nifti_zyx(inst_dst, seg_instance.astype(np.float32, copy=False), img_spacing)
                _write_nifti_zyx(seg_dst, seg_bc.astype(np.float32, copy=False), img_spacing)
            img_shape = [int(img_arr.shape[i]) for i in range(min(3, int(img_arr.ndim)))]
            finetune_entries_flat.append(
                {
                    "image": str(img_dst),
                    "label": str(seg_dst),
                    "resolution": [float(img_spacing[0]), float(img_spacing[1]), float(img_spacing[2])],
                    "shape_zyx": img_shape,
                }
            )
            if use_registry and _pp_run_id is not None and _registry_conn is not None:
                try:
                    _pp_done(_registry_conn, _pp_run_id, [str(img_dst), str(seg_dst)])
                    _registry_conn.commit()
                except Exception as _exc:
                    print(f"[REGISTRY] Warning: could not record preprocess completion: {_exc}", flush=True)
            print(f"[DONE] dataset {i}/{n_pairs}: {key}", flush=True)
        finetune_entries_flat = _dedupe_finetune_entries(finetune_entries_flat)
        ft_payload = _sanitize_finetune_payload(_split_train_val_test(finetune_entries_flat))
        if pairs and not finetune_entries_flat:
            raise SystemExit(
                "No finetune pairs were converted. Install required readers/writers (h5py, pynrrd, nibabel) "
                "or select datasets in a supported format."
            )
    else:
        ft_payload = {"training": [], "validation": [], "test": []}

    yaml_sync: dict[str, Any] = {"skipped": True, "reason": "task without finetune or no entries"}
    if (
        args.task in ("finetune", "supervised")
        and finetune_entries_flat
        and not bool(args.skip_training_yaml)
    ):
        cfg_path = Path(args.training_yaml).resolve() if args.training_yaml else None
        yaml_sync = sync_mito_openorganelle_foundation_yaml(
            root,
            training=list(ft_payload["training"]),
            validation=list(ft_payload["validation"]),
            test=list(ft_payload["test"]),
            config_path=cfg_path,
        )
    elif bool(args.skip_training_yaml):
        yaml_sync = {"skipped": True, "reason": "--skip-training-yaml"}

    print(
        f"[SUMMARY] task={args.task} format={out_fmt} split_label_cc={split_cc} "
        f"selected={len(raw_files)} converted={len(finetune_entries_flat)} skipped={finetune_skipped}",
        flush=True,
    )
    if isinstance(yaml_sync, dict) and yaml_sync.get("skipped"):
        print(f"[SUMMARY] training_yaml_sync: skipped ({yaml_sync.get('reason', 'n/a')})", flush=True)
    elif isinstance(yaml_sync, dict):
        print(f"[SUMMARY] training_yaml_sync: {yaml_sync.get('message', 'updated')}", flush=True)


if __name__ == "__main__":
    main()
