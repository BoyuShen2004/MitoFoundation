from __future__ import annotations

import json
import importlib
import os
import re
import shutil
import threading
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import sys

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    h5py = None
try:
    import nibabel as nib  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    nib = None
try:
    import tifffile  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tifffile = None

from config.paths import mitole_root as _mitole_root
from config.paths import mitole_sources_xlsx as _mitole_sources_xlsx
from config.paths import nnunet_dataset_root

MITOLE_ROOT = _mitole_root()

# Log labels mirror OpenOrganelle / BossDB labeled stage-3 scripts (website line = data origin).
MITOLE_STAGE3_WEBSITE = "MitoLE"
MITOLE_STAGE3_DATA_TYPE = (
    "labeled (catalogue EM + mitochondria labels; isotropic foundation crop to nnUNet raw)"
)


def mitole_count_datasets_with_requested_crops(
    dataset_pairs: list[dict[str, str]],
    dataset_splits: dict[str, dict[str, int]],
) -> int:
    """Count catalogue rows with training+inference crop count > 0 (same notion as web script targets)."""
    n = 0
    for pair in dataset_pairs or []:
        ds = str(pair.get("dataset") or "").strip()
        if not ds:
            continue
        sp = dataset_splits.get(ds, {}) or {}
        tr_n = max(0, int(sp.get("training", 0) or 0))
        ts_n = max(0, int(sp.get("inference", 0) or 0))
        if tr_n + ts_n > 0:
            n += 1
    return int(n)

MITOLE_DEFAULT_REL_FOLDERS = [
    "Kedar_4yv2h",
    "Kedar_f536",
    "casser20",
    "cellmap24_cardiac",
    "cellmap24_jurkat",
    "cellmap24_kidney",
    "cellmap24_liver",
    "cellmap24_macrophage",
    "existing/microns",
    "guay21",
    "haberl18",
    "han24",
    "jiang25",
    "lucchi",
    "mitoNet_easy",
    "mitoNet_hard",
    "urocell",
    "wei20",
    "wilson19",
    "xiao18",
]

_ROW_CACHE_LOCK = threading.Lock()
_ROW_CACHE: dict[str, dict[str, Any]] = {}
_SPACING_LOOKUP_LOCK = threading.Lock()
_SPACING_LOOKUP_READY = False
_SPACING_BY_PATH: dict[str, list[float]] = {}
_MITOLE_SOURCES_XLSX = _mitole_sources_xlsx()


def _parse_spacing_triplet(raw: Any) -> list[float]:
    s = str(raw or "").strip().lower().replace(" ", "")
    if not s:
        return []
    parts = [p for p in s.split("x") if p]
    out: list[float] = []
    for p in parts[:3]:
        try:
            out.append(float(p))
        except Exception:
            return []
    return out if len(out) == 3 else []


def _ensure_spacing_lookup_loaded() -> None:
    global _SPACING_LOOKUP_READY
    if _SPACING_LOOKUP_READY:
        return
    with _SPACING_LOOKUP_LOCK:
        if _SPACING_LOOKUP_READY:
            return
        mapping: dict[str, list[float]] = {}
        def _norm_key(p: str) -> str:
            try:
                return os.path.normpath(str(p).strip())
            except Exception:
                return str(p).strip()
        def _col_to_idx(cell_ref: str) -> int:
            col = "".join(ch for ch in str(cell_ref or "") if ch.isalpha()).upper()
            out = 0
            for ch in col:
                out = out * 26 + (ord(ch) - ord("A") + 1)
            return max(0, out - 1)
        try:
            if _MITOLE_SOURCES_XLSX.is_file():
                with zipfile.ZipFile(str(_MITOLE_SOURCES_XLSX), "r") as zf:
                    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                    shared_strings: list[str] = []
                    if "xl/sharedStrings.xml" in zf.namelist():
                        ss_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
                        for si in ss_root.findall("a:si", ns):
                            t = "".join(node.text or "" for node in si.findall(".//a:t", ns))
                            shared_strings.append(t)
                    sheet_name = "xl/worksheets/sheet1.xml"
                    if sheet_name not in zf.namelist():
                        for n in zf.namelist():
                            if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"):
                                sheet_name = n
                                break
                    sh_root = ET.fromstring(zf.read(sheet_name))
                    headers: list[str] = []
                    idx: dict[str, int] = {}
                    row_i = 0
                    for row in sh_root.findall(".//a:sheetData/a:row", ns):
                        vals_by_idx: dict[int, str] = {}
                        for c in row.findall("a:c", ns):
                            tpe = c.attrib.get("t", "")
                            col_idx = _col_to_idx(c.attrib.get("r", ""))
                            v = c.find("a:v", ns)
                            raw_v = (v.text or "") if v is not None else ""
                            if tpe == "s":
                                try:
                                    sval = shared_strings[int(raw_v)]
                                except Exception:
                                    sval = ""
                                vals_by_idx[col_idx] = sval
                            elif tpe == "inlineStr":
                                is_t = c.find("a:is/a:t", ns)
                                vals_by_idx[col_idx] = (is_t.text or "") if is_t is not None else ""
                            else:
                                vals_by_idx[col_idx] = raw_v
                        max_idx = max(vals_by_idx.keys()) if vals_by_idx else -1
                        vals = [vals_by_idx.get(i, "") for i in range(max_idx + 1)]
                        if row_i == 0:
                            headers = [str(x or "").strip() for x in vals]
                            idx = {h: i for i, h in enumerate(headers)}
                            row_i += 1
                            continue
                        raw_path = str(vals[idx.get("raw_filepath", -1)] or "").strip() if idx.get("raw_filepath", -1) >= 0 and idx.get("raw_filepath", -1) < len(vals) else ""
                        lbl_path = str(vals[idx.get("label_filepath", -1)] or "").strip() if idx.get("label_filepath", -1) >= 0 and idx.get("label_filepath", -1) < len(vals) else ""
                        sp_raw = _parse_spacing_triplet(vals[idx.get("raw_voxel_spacing_um_zyx", -1)] if idx.get("raw_voxel_spacing_um_zyx", -1) >= 0 and idx.get("raw_voxel_spacing_um_zyx", -1) < len(vals) else "")
                        sp_lbl = _parse_spacing_triplet(vals[idx.get("label_voxel_spacing_um_zyx", -1)] if idx.get("label_voxel_spacing_um_zyx", -1) >= 0 and idx.get("label_voxel_spacing_um_zyx", -1) < len(vals) else "")
                        if raw_path and sp_raw:
                            mapping[_norm_key(raw_path)] = sp_raw
                        if lbl_path and sp_lbl:
                            mapping[_norm_key(lbl_path)] = sp_lbl
                        row_i += 1
        except Exception:
            mapping = {}
        _SPACING_BY_PATH.clear()
        _SPACING_BY_PATH.update(mapping)
        _SPACING_LOOKUP_READY = True


def config_path(project_root: Path) -> Path:
    return project_root / "data" / ".studio_mitole_folders.json"


def clean_rel_folder(value: str) -> str | None:
    s = str(value or "").strip().replace("\\", "/").strip("/")
    if not s:
        return None
    parts = [p for p in s.split("/") if p and p != "."]
    if not parts or any(p == ".." for p in parts):
        return None
    return "/".join(parts)


def abs_from_rel(rel_folder: str) -> Path:
    rel = clean_rel_folder(rel_folder)
    if not rel:
        raise ValueError(f"Invalid folder: {rel_folder!r}")
    base = MITOLE_ROOT.resolve()
    p = (base / rel).resolve()
    if not str(p).startswith(str(base) + os.sep) and p != base:
        raise ValueError(f"Folder escapes MitoLE root: {rel}")
    return p


def load_selected_rel_folders(project_root: Path) -> list[str]:
    cfg = config_path(project_root)
    out: list[str] = []
    if cfg.is_file():
        try:
            j = json.loads(cfg.read_text(encoding="utf-8"))
            vals = j.get("folders", []) if isinstance(j, dict) else []
            if isinstance(vals, list):
                for raw in vals:
                    rel = clean_rel_folder(str(raw))
                    if rel and rel not in out:
                        out.append(rel)
        except Exception:
            out = []
    if not out:
        out = list(MITOLE_DEFAULT_REL_FOLDERS)
    return out


def save_selected_rel_folders(project_root: Path, folders: list[str]) -> None:
    cfg = config_path(project_root)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    clean: list[str] = []
    for raw in folders:
        rel = clean_rel_folder(raw)
        if rel and rel not in clean:
            clean.append(rel)
    if not clean:
        clean = list(MITOLE_DEFAULT_REL_FOLDERS)
    cfg.write_text(json.dumps({"folders": clean}, indent=2), encoding="utf-8")


def scan_folder_rows(
    abs_folder: Path,
    rel_folder: str,
    inspect_dataset_file: Callable[[Path], dict[str, Any]],
    inspect_dataset_file_shallow: Callable[[Path], dict[str, Any]],
    inspect_coerce_row_json: Callable[[dict[str, Any]], None],
) -> list[dict[str, Any]]:
    _ensure_spacing_lookup_loaded()
    rows: list[dict[str, Any]] = []
    if not abs_folder.is_dir():
        return rows
    exts = (".h5", ".nii", ".nii.gz", ".tif", ".tiff")
    # Use os.walk instead of recursive glob+sort for significantly faster large-folder scans.
    for root, _, files in os.walk(abs_folder):
        for fname in files:
            name_l = fname.lower()
            if not name_l.endswith(exts):
                continue
            p = Path(root) / fname
            cache_key = ""
            try:
                st = p.stat()
                cache_key = f"{str(p)}::{int(st.st_mtime_ns)}::{int(st.st_size)}"
            except Exception:
                cache_key = ""
            if cache_key:
                with _ROW_CACHE_LOCK:
                    cached = _ROW_CACHE.get(cache_key)
                if isinstance(cached, dict):
                    row = dict(cached)
                    row["folder"] = rel_folder
                    rows.append(row)
                    continue
            try:
                row = inspect_dataset_file(p)
            except Exception:
                try:
                    row = inspect_dataset_file_shallow(p)
                except Exception:
                    row = {
                        "name": p.name,
                        "path": str(p),
                        "type": p.suffix.lstrip(".").lower(),
                        "dimensions": [],
                        "spacing": [],
                        "label_summary": "",
                    }
                    inspect_coerce_row_json(row)
            # Stage-1 browser no longer needs label-count metadata.
            row["label_summary"] = ""
            if (not isinstance(row.get("spacing"), list) or len(row.get("spacing") or []) == 0) and _SPACING_BY_PATH:
                sp = _SPACING_BY_PATH.get(os.path.normpath(str(p)))
                if sp:
                    row["spacing"] = list(sp)
            row["folder"] = rel_folder
            if cache_key:
                with _ROW_CACHE_LOCK:
                    _ROW_CACHE[cache_key] = {
                        "name": row.get("name", p.name),
                        "path": row.get("path", str(p)),
                        "type": row.get("type", ""),
                        "dimensions": row.get("dimensions", []),
                        "spacing": row.get("spacing", []),
                        "label_summary": "",
                    }
            rows.append(row)
    return rows


def list_all_subfolders() -> list[str]:
    out: list[str] = []
    if not MITOLE_ROOT.is_dir():
        return out
    base = MITOLE_ROOT.resolve()
    for p in sorted(base.glob("**/*"), key=lambda x: str(x).lower()):
        if not p.is_dir():
            continue
        rel = p.relative_to(base).as_posix().strip("/")
        if not rel:
            continue
        if any(seg.startswith(".") for seg in rel.split("/")):
            continue
        out.append(rel)
    return out


def _slug_token(value: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value or "").strip()).strip("._-")
    return s or "dataset"


def _mitole_nnunet_split_dirs(dataset_root: Path, split_name: str) -> tuple[Path, Path]:
    if split_name == "training":
        return dataset_root / "imagesTr", dataset_root / "labelsTr"
    return dataset_root / "imagesTs", dataset_root / "labelsTs"


def _mitole_foundation_stem_has_any_file(tag: str, img_dir: Path, lbl_dir: Path, stem: str) -> bool:
    """True if either EM stack or label exists for this stem (OpenOrganelle collision rule)."""
    return (img_dir / f"{tag}{stem}_0000.nii.gz").is_file() or (lbl_dir / f"{tag}{stem}.nii.gz").is_file()


def mitole_resolve_foundation_suffix_disk_only(
    tag: str, img_dir: Path, lbl_dir: Path, base_suffix: str
) -> str:
    """If ``_volN`` (or partial) already exists, use ``_volN_M`` (M>=2), matching labeled scrape."""
    resolved = base_suffix
    if _mitole_foundation_stem_has_any_file(tag, img_dir, lbl_dir, resolved):
        m = 2
        while True:
            trial = f"{base_suffix}_{m}"
            if not _mitole_foundation_stem_has_any_file(tag, img_dir, lbl_dir, trial):
                return trial
            m += 1
    return resolved


def mitole_planned_foundation_suffixes(
    dataset_root: Path, dataset: str, split_name: str, n_crops: int
) -> list[str]:
    """Plan batch suffixes for crops 1..n.

    If any base ``_volN`` collides on disk, assign one shared ``_M`` suffix to all crops in the batch.
    """
    if int(n_crops) <= 0:
        return []
    tag = _slug_token(dataset)
    img_dir, lbl_dir = _mitole_nnunet_split_dirs(dataset_root, split_name)
    n = int(n_crops)
    base = [f"_vol{i}" for i in range(1, n + 1)]
    if not any(_mitole_foundation_stem_has_any_file(tag, img_dir, lbl_dir, s) for s in base):
        return base
    m = 2
    while True:
        trial = [f"_vol{i}_{m}" for i in range(1, n + 1)]
        if all(not _mitole_foundation_stem_has_any_file(tag, img_dir, lbl_dir, s) for s in trial):
            return trial
        m += 1


def mitole_foundation_pair_complete(
    tag: str, img_dir: Path, lbl_dir: Path, resolved_suffix: str
) -> bool:
    img_f = img_dir / f"{tag}{resolved_suffix}_0000.nii.gz"
    lbl_f = lbl_dir / f"{tag}{resolved_suffix}.nii.gz"
    return img_f.is_file() and lbl_f.is_file()


def _mitole_alternating_volume_indices(n_training: int, n_inference: int) -> tuple[list[int], list[int]]:
    rp = Path(__file__).resolve().parents[1]
    if str(rp) not in sys.path:
        sys.path.insert(0, str(rp))
    mod = importlib.import_module("3data_downloader.downloader_common.labeled_train_infer_split")
    alternating_volume_indices = getattr(mod, "alternating_volume_indices")
    return alternating_volume_indices(n_training, n_inference)


def _mitole_total_profile_completed(dataset_root: Path, dataset: str, n_total: int) -> bool:
    rp = Path(__file__).resolve().parents[1]
    if str(rp) not in sys.path:
        sys.path.insert(0, str(rp))
    mod = importlib.import_module("3data_downloader.downloader_common.labeled_train_infer_split")
    fn = getattr(mod, "global_crop_profile_completed")
    return bool(fn(str(dataset_root), _slug_token(dataset), int(n_total)))


def _mitole_mark_total_profile_completed(dataset_root: Path, dataset: str, n_total: int) -> None:
    rp = Path(__file__).resolve().parents[1]
    if str(rp) not in sys.path:
        sys.path.insert(0, str(rp))
    mod = importlib.import_module("3data_downloader.downloader_common.labeled_train_infer_split")
    fn = getattr(mod, "mark_global_crop_profile_completed")
    fn(str(dataset_root), _slug_token(dataset), int(n_total))


def _mitole_instance_dir(dataset_root: Path, split_name: str) -> Path:
    return (
        dataset_root / "labelsTr-instance"
        if str(split_name).strip().lower() == "training"
        else dataset_root / "labelsTs-instance"
    )


def _mitole_foundation_stem_sort_key(stem: str) -> tuple[int, int, int]:
    """Sort key for ``_volN`` / ``_volN_M`` style stems (main index, collision suffix, tie-break)."""
    s = str(stem or "")
    m = re.match(r"^_vol(\d+)(?:_(\d+))?$", s, flags=re.IGNORECASE)
    if not m:
        return (10**9, 10**9, 0)
    return (int(m.group(1)), int(m.group(2) or 0), 0)


def _mitole_iter_volk_image_paths(tag: str, img_dir: Path, vol_k: int) -> list[Path]:
    """Candidate EM stacks for global volume index ``vol_k`` (includes collision ``_volK_M``)."""
    out: list[Path] = []
    exact = img_dir / f"{tag}_vol{vol_k}_0000.nii.gz"
    if exact.is_file():
        out.append(exact)
    for p in sorted(img_dir.glob(f"{tag}_vol{vol_k}_*_0000.nii.gz")):
        if p not in out and p.is_file():
            out.append(p)
    return out


def _mitole_foundation_stem_from_image_path(tag: str, img_path: Path) -> str | None:
    n = img_path.name
    if not n.endswith("_0000.nii.gz"):
        return None
    core = n[: -len("_0000.nii.gz")]
    if not core.startswith(tag):
        return None
    return core[len(tag) :]


def mitole_global_vol_complete(
    dataset_root: Path, dataset: str, global_k: int, total_n: int
) -> bool:
    """True iff global foundation crop ``global_k`` (1..total_n) is complete in either nnUNet split folder.

    This intentionally ignores train/inference assignment so split-only permutations (e.g. 1+1 -> 2+0)
    do not trigger redundant materialization of the same physical crops.
    """
    if global_k < 1 or total_n < 1 or global_k > total_n:
        return False
    tag = _slug_token(dataset)
    for split_name in ("training", "inference"):
        img_dir, lbl_dir = _mitole_nnunet_split_dirs(dataset_root, split_name)
        inst_dir = _mitole_instance_dir(dataset_root, split_name)
        for img_p in _mitole_iter_volk_image_paths(tag, img_dir, int(global_k)):
            stem = _mitole_foundation_stem_from_image_path(tag, img_p)
            if not stem or not mitole_foundation_pair_complete(tag, img_dir, lbl_dir, stem):
                continue
            inst_p = inst_dir / f"{tag}{stem}_instance.nii.gz"
            if inst_dir.is_dir() and not inst_p.is_file():
                continue
            return True
    return False


def _mitole_collect_union_foundation_rows(dataset_root: Path, tag: str) -> list[dict[str, Any]]:
    """All complete foundation-style nnUNet pairs for ``tag`` across Tr and Ts (with stems)."""
    rows: list[dict[str, Any]] = []
    for split_name in ("training", "inference"):
        img_dir, lbl_dir = _mitole_nnunet_split_dirs(dataset_root, split_name)
        inst_dir = _mitole_instance_dir(dataset_root, split_name)
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            continue
        for img_p in sorted(img_dir.glob(f"{tag}_vol*_0000.nii.gz")):
            stem = _mitole_foundation_stem_from_image_path(tag, img_p)
            if not stem or not re.match(r"^_vol\d+", stem, flags=re.IGNORECASE):
                continue
            if not mitole_foundation_pair_complete(tag, img_dir, lbl_dir, stem):
                continue
            inst_p = inst_dir / f"{tag}{stem}_instance.nii.gz"
            if inst_dir.is_dir() and not inst_p.is_file():
                continue
            lbl_p = lbl_dir / f"{tag}{stem}.nii.gz"
            rows.append(
                {
                    "split": split_name,
                    "stem": stem,
                    "img": img_p,
                    "lbl": lbl_p,
                    "inst": inst_p if inst_p.is_file() else None,
                }
            )
    return rows


def mitole_reconcile_foundation_layout(
    dataset_root: Path,
    dataset: str,
    tr_n: int,
    ts_n: int,
    log_fn: Callable[[str], None] | None = None,
) -> None:
    """Rename/move existing foundation stacks so ``_vol1``..``_vol{total}`` match the web alternating layout.

    Handles (a) stacks already named ``_vol{k}`` but in the wrong split folder, and (b) older MitoLE runs
    that numbered training and inference **per-split** (``_vol1`` in ``imagesTs`` for the first inference
    slot) by re-stamping the first ``total`` physical pairs to canonical global names, then shuffling.
    """
    total = int(tr_n) + int(ts_n)
    if total <= 0:
        return
    tag = _slug_token(dataset)
    if all(
        mitole_global_vol_complete(dataset_root, dataset, k, total)
        for k in range(1, total + 1)
    ):
        return

    rows = _mitole_collect_union_foundation_rows(dataset_root, tag)
    if len(rows) < total:
        return

    def _row_sort_key(r: dict[str, Any]) -> tuple[int, int, int, int]:
        split_ord = 0 if r["split"] == "training" else 1
        sk = _mitole_foundation_stem_sort_key(str(r.get("stem") or ""))
        return (split_ord, sk[0], sk[1], sk[2])

    rows.sort(key=_row_sort_key)
    chosen = rows[:total]

    # Phase 1 — rename to canonical ``_vol{1..total}`` (descending by target index avoids same-dir clashes).
    for gi in range(total, 0, -1):
        row = chosen[gi - 1]
        want_stem = f"_vol{gi}"
        cur_stem = str(row["stem"])
        if cur_stem == want_stem:
            continue
        split_name = str(row["split"])
        img_dir, lbl_dir = _mitole_nnunet_split_dirs(dataset_root, split_name)
        inst_dir = _mitole_instance_dir(dataset_root, split_name)
        src_img = row["img"]
        src_lbl = row["lbl"]
        src_inst = row.get("inst")
        dst_img = img_dir / f"{tag}{want_stem}_0000.nii.gz"
        dst_lbl = lbl_dir / f"{tag}{want_stem}.nii.gz"
        dst_inst = inst_dir / f"{tag}{want_stem}_instance.nii.gz"
        if dst_img.is_file() and src_img.resolve() != dst_img.resolve():
            continue
        if callable(log_fn):
            log_fn(
                f"[INFO] {dataset}: reconcile rename {cur_stem} -> {want_stem} "
                f"({split_name}) for global vol {gi}/{total}\n"
            )
        src_img.rename(dst_img)
        src_lbl.rename(dst_lbl)
        if src_inst is not None and Path(src_inst).is_file():
            Path(src_inst).rename(dst_inst)
        row["stem"] = want_stem
        row["img"] = dst_img
        row["lbl"] = dst_lbl
        row["inst"] = dst_inst if dst_inst.is_file() else None

    # Phase 2 — move each global volume into its target split (same basename as OpenOrganelle).
    tr_idx, ts_idx = _mitole_alternating_volume_indices(tr_n, ts_n)
    for gi in range(1, total + 1):
        want_split = "training" if gi in tr_idx else "inference"
        row = chosen[gi - 1]
        cur_split = str(row["split"])
        if cur_split == want_split:
            continue
        src_img_dir, src_lbl_dir = _mitole_nnunet_split_dirs(dataset_root, cur_split)
        dst_img_dir, dst_lbl_dir = _mitole_nnunet_split_dirs(dataset_root, want_split)
        src_inst_dir = _mitole_instance_dir(dataset_root, cur_split)
        dst_inst_dir = _mitole_instance_dir(dataset_root, want_split)
        stem = str(row["stem"])
        s_img = src_img_dir / f"{tag}{stem}_0000.nii.gz"
        s_lbl = src_lbl_dir / f"{tag}{stem}.nii.gz"
        s_inst = src_inst_dir / f"{tag}{stem}_instance.nii.gz"
        d_img = dst_img_dir / f"{tag}{stem}_0000.nii.gz"
        d_lbl = dst_lbl_dir / f"{tag}{stem}.nii.gz"
        d_inst = dst_inst_dir / f"{tag}{stem}_instance.nii.gz"
        if not s_img.is_file() or not s_lbl.is_file():
            continue
        if callable(log_fn):
            log_fn(
                f"[INFO] {dataset}: reconcile move {stem} {cur_split} -> {want_split} "
                f"(global vol {gi}/{total})\n"
            )
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)
        dst_inst_dir.mkdir(parents=True, exist_ok=True)
        if d_img.is_file() and s_img.resolve() != d_img.resolve():
            continue
        shutil.move(str(s_img), str(d_img))
        shutil.move(str(s_lbl), str(d_lbl))
        if s_inst.is_file():
            shutil.move(str(s_inst), str(d_inst))
        row["split"] = want_split
        row["img"] = d_img
        row["lbl"] = d_lbl
        row["inst"] = d_inst if d_inst.is_file() else None


def mitole_pending_and_requested_crop_counts(
    project_root: Path,
    dataset_pairs: list[dict[str, str]],
    dataset_splits: dict[str, dict[str, int]],
) -> tuple[int, int]:
    """Return ``(pending_new_crops, requested_crops_eligible)`` for progress bars / logs.

    Only counts pairs that pass the same path/security gates as the materializer.
    """
    dataset_root = nnunet_dataset_root(project_root)
    base = MITOLE_ROOT.resolve()
    pending = 0
    requested = 0
    for pair in dataset_pairs:
        dataset = str(pair.get("dataset") or "").strip()
        source_rel = str(pair.get("source") or "").strip()
        image_raw = str(pair.get("image_path") or "").strip()
        label_raw = str(pair.get("label_path") or "").strip()
        image_path = Path(image_raw).expanduser().resolve() if image_raw else None
        label_path = Path(label_raw).expanduser().resolve() if label_raw else None
        split = dataset_splits.get(dataset, {})
        tr_n = max(0, int(split.get("training", 0) or 0))
        ts_n = max(0, int(split.get("inference", 0) or 0))
        if tr_n + ts_n <= 0:
            continue
        if image_path is None or label_path is None:
            resolved = mitole_resolve_pair_from_source_rel(source_rel)
            if resolved is None:
                continue
            image_path, label_path = resolved
        if not image_path.is_file() or not label_path.is_file():
            continue
        if not str(image_path).startswith(str(base) + os.sep):
            continue
        if not str(label_path).startswith(str(base) + os.sep):
            continue
        total_n = int(tr_n) + int(ts_n)
        if total_n <= 0:
            continue
        requested += int(total_n)
        pool_complete = all(
            mitole_global_vol_complete(dataset_root, dataset, k, total_n)
            for k in range(1, total_n + 1)
        )
        profile_done = _mitole_total_profile_completed(dataset_root, dataset, total_n)
        if pool_complete and profile_done:
            continue
        if pool_complete and not profile_done:
            _mitole_mark_total_profile_completed(dataset_root, dataset, total_n)
            continue
        # If this profile total was never completed, re-materialize the entire set to preserve
        # OpenOrganelle collision semantics when total crop count changes.
        pending += int(total_n if not profile_done else max(0, total_n))
    return int(pending), int(requested)


def _mitole_pair_key_norm(name: str) -> str:
    s = str(name or "").lower()
    s = re.sub(r"\.nii\.gz$", "", s)
    s = re.sub(r"\.(h5|nii|tif|tiff)$", "", s)
    s = re.sub(r"([._-])(im|image|img)$", "", s)
    s = re.sub(r"([._-])(mito|seg|label|labels|mask|gt)$", "", s)
    s = re.sub(r"[._-](v\d+|vol\d+|slice\d+|patch\d+)$", "", s)
    return s


def _mitole_pair_is_label_path(p: Path) -> bool:
    n = p.name.lower()
    ps = str(p).lower()
    return bool(
        re.search(r"(label|labels|mask|_seg|segmentation|_gt|_mito)", n)
        or re.search(r"(\/labels?\/|\/masks?\/|\/mito\/)", ps)
    )


def mitole_resolve_pair_from_source_rel(source_rel: str) -> tuple[Path, Path] | None:
    """Resolve image/label paths under MitoLE (same rules as ``stage3_copy_selected_pairs``)."""
    try:
        folder = abs_from_rel(source_rel)
    except Exception:
        return None
    if not folder.is_dir():
        return None
    exts = (".h5", ".nii", ".nii.gz", ".tif", ".tiff")
    images: list[Path] = []
    labels: list[Path] = []
    for root, _, files in os.walk(folder):
        for fname in files:
            low = fname.lower()
            if not low.endswith(exts):
                continue
            p = Path(root) / fname
            if _mitole_pair_is_label_path(p):
                labels.append(p)
            else:
                images.append(p)
    label_by_key: dict[str, list[Path]] = {}
    for lb in labels:
        k = _mitole_pair_key_norm(lb.name)
        label_by_key.setdefault(k, []).append(lb)
    for im in sorted(images, key=lambda p: p.name.lower()):
        k = _mitole_pair_key_norm(im.name)
        bucket = label_by_key.get(k, [])
        if not bucket:
            continue
        lb = bucket.pop(0)
        label_by_key[k] = bucket
        return (im, lb)
    return None


def stage3_copy_selected_pairs(
    project_root: Path,
    dataset_pairs: list[dict[str, str]],
    dataset_splits: dict[str, dict[str, int]],
    log_fn: Callable[[str], None] | None = None,
    progress_fn: Callable[[int, int, int, str], None] | None = None,
    cancel_requested_fn: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Copy selected Stage-2 catalogue pairs into HPC Stage-3 output folders.

    This stage stays local-HPC only: source files are copied directly from MitoLE
    folders and never touch website-scrape provider logic. The selected
    train/inference split is materialized directly under Dataset001_mito2.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"mitole_local_{ts}"
    run_dir = (project_root / "hpc_data_pipeline" / "stage3" / "outputs" / run_name).resolve()
    images_dir = run_dir / "images"
    labels_dir = run_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = nnunet_dataset_root(project_root)
    images_tr = dataset_root / "imagesTr"
    labels_tr = dataset_root / "labelsTr"
    labels_tr_inst = dataset_root / "labelsTr-instance"
    images_ts = dataset_root / "imagesTs"
    labels_ts = dataset_root / "labelsTs"
    labels_ts_inst = dataset_root / "labelsTs-instance"
    for d in (images_tr, labels_tr, labels_tr_inst, images_ts, labels_ts, labels_ts_inst):
        d.mkdir(parents=True, exist_ok=True)

    base = MITOLE_ROOT.resolve()
    copied: list[dict[str, Any]] = []
    skipped: list[str] = []
    used_names: set[str] = set()
    train_count = 0
    infer_count = 0
    pending_total, requested_eligible = mitole_pending_and_requested_crop_counts(
        project_root, dataset_pairs, dataset_splits
    )
    completed_new = 0
    prog_denom = max(1, int(pending_total))
    n_catalog_ds = mitole_count_datasets_with_requested_crops(dataset_pairs, dataset_splits)
    already_ok = max(0, int(requested_eligible) - int(pending_total))
    if callable(progress_fn):
        progress_fn(0, prog_denom, 0, "")
    if callable(log_fn):
        log_fn(
            f"[PLAN] {int(pending_total)} crop pair(s) need new files now; "
            f"{already_ok} already complete on disk (nnUNet Dataset001_mito2).\n"
        )
        log_fn("=" * 60 + "\n")
        log_fn("Labeled download summary\n")
        log_fn(f"- Website: {MITOLE_STAGE3_WEBSITE}\n")
        log_fn("- Pipeline: MitoLE local (HPC catalogue; parity with web stage 3)\n")
        log_fn(f"- Data type: {MITOLE_STAGE3_DATA_TYPE}\n")
        log_fn("- Foundation mode: true (≤128 voxels/axis @ 16 nm isotropic, extent-limited)\n")
        log_fn(f"- Run: {run_name}\n")
        log_fn(f"- Total datasets in catalogue selection: {n_catalog_ds}\n")
        log_fn(f"- Planned image/label pairs: {int(requested_eligible)}\n")
        log_fn(f"- Pairs already on disk: {already_ok}\n")
        log_fn(f"- Pairs to materialize now: {int(pending_total)}\n")
        log_fn(
            "- Windows per dataset: global foundation crop count (training + inference); "
            "nnUNet folders follow web alternating split after staging in imagesTr\n"
        )
        log_fn("- Output: NIfTI (.nii.gz) nnUNet naming\n")
        log_fn(f"- nnUNet raw root: {dataset_root}\n")
        log_fn(f"- Stage-3 staging copy folder: {run_dir}\n")
        log_fn("=" * 60 + "\n")
        log_fn(
            f"[INFO] Progress denominator={int(pending_total)} new crop file(s); "
            f"eligible planned pairs={int(requested_eligible)}.\n"
        )

    def _full_ext(p: Path) -> str:
        n = p.name.lower()
        if n.endswith(".nii.gz"):
            return ".nii.gz"
        return p.suffix or ""

    def _load_volume(path: Path) -> np.ndarray:
        ext = _full_ext(path).lower()
        if ext == ".h5":
            if h5py is None:
                raise RuntimeError("h5py is required to read .h5 datasets")
            with h5py.File(str(path), "r") as f:
                vals = list(f.values())
                if not vals:
                    raise RuntimeError(f"No datasets in H5 file: {path}")
                return np.asarray(vals[0])
        if ext == ".nii.gz" or ext == ".nii":
            if nib is None:
                raise RuntimeError("nibabel is required to read NIfTI datasets")
            img = nib.load(str(path))
            return np.asarray(img.get_fdata())
        if ext in (".tif", ".tiff"):
            if tifffile is None:
                raise RuntimeError("tifffile is required to read TIFF datasets")
            return np.asarray(tifffile.imread(str(path)))
        raise RuntimeError(f"Unsupported source extension for crop generation: {path.name}")

    def _to_zyx(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr)
        if a.ndim < 3:
            raise RuntimeError("Volume must be 3D")
        if a.ndim > 3:
            a = a.squeeze()
            if a.ndim > 3:
                a = a.reshape(a.shape[-3], a.shape[-2], a.shape[-1])
        return np.asarray(a)

    def _spacing_zyx_from_path(path: Path) -> tuple[float, float, float] | None:
        ext = _full_ext(path).lower()
        try:
            if ext == ".h5" and h5py is not None:
                with h5py.File(str(path), "r") as f:
                    vals = list(f.values())
                    d = vals[0] if vals else None
                    for node in [d, f]:
                        if node is None:
                            continue
                        raw = node.attrs.get("voxel_size_nm") or node.attrs.get("spacing")
                        if raw is None:
                            continue
                        arr = np.asarray(raw).astype(float).reshape(-1).tolist()
                        if len(arr) >= 3:
                            return (float(arr[0]), float(arr[1]), float(arr[2]))
            if (ext == ".nii.gz" or ext == ".nii") and nib is not None:
                img = nib.load(str(path))
                zooms = [float(v) for v in img.header.get_zooms()[:3]]
                while len(zooms) < 3:
                    zooms.append(1.0)
                # Nib zooms are in x,y,z order; convert to z,y,x.
                return (zooms[2], zooms[1], zooms[0])
        except Exception:
            return None
        return None

    def _materialize_split_pair(
        *,
        split_name: str,
        dataset: str,
        unit_idx: int,
        total_in_split: int,
        selected_offset: tuple[int, int, int] | None = None,
        image_path: Path,
        label_path: Path,
        forced_nnunet_dirs: tuple[Path, Path, Path] | None = None,
        resolved_suffix_override: str | None = None,
    ) -> dict[str, str]:
        base_id = _slug_token(f"{dataset}_{split_name}_{unit_idx}")
        image_ext = _full_ext(image_path)
        label_ext = _full_ext(label_path)
        if forced_nnunet_dirs is not None:
            img_dir, lbl_dir, inst_dir = forced_nnunet_dirs
        elif split_name == "training":
            img_dir, lbl_dir, inst_dir = images_tr, labels_tr, labels_tr_inst
        else:
            img_dir, lbl_dir, inst_dir = images_ts, labels_ts, labels_ts_inst
        img_dst = img_dir / f"{base_id}_0000.nii.gz"
        lbl_dst = lbl_dir / f"{base_id}.nii.gz"
        try:
            if str(project_root.resolve()) not in sys.path:
                sys.path.insert(0, str(project_root.resolve()))
            fr = importlib.import_module("3data_downloader.openorganelle.foundation_resample")
            exp = importlib.import_module("3data_downloader.downloader_common.nnunet_labeled_export")
            foundation_crop_shapes = getattr(fr, "foundation_crop_shapes")
            _farthest_offsets_non_overlapping = getattr(fr, "_farthest_offsets_non_overlapping")
            _center_offset = getattr(fr, "_center_offset")
            _window_from_offset = getattr(fr, "_window_from_offset")
            _img_window_to_seg_slices = getattr(fr, "_img_window_to_seg_slices")
            _resize_seg_to_match = getattr(fr, "_resize_seg_to_match")
            resample_crop_to_isotropic = getattr(fr, "resample_crop_to_isotropic")
            _crop_index_suffix = getattr(fr, "_crop_index_suffix")
            write_h5 = getattr(exp, "write_h5")

            img_arr = _to_zyx(_load_volume(image_path))
            seg_arr = _to_zyx(_load_volume(label_path))
            spacing = _spacing_zyx_from_path(image_path) or (16.0, 16.0, 16.0)
            physical = (16.0 * 128.0, 16.0 * 128.0, 16.0 * 128.0)
            vol_shape = tuple(int(x) for x in img_arr.shape[:3])
            img_shape = vol_shape
            seg_shape = tuple(int(x) for x in seg_arr.shape[:3])
            ch_need = foundation_crop_shapes(spacing, vol_shape, physical)
            if selected_offset is not None:
                off = selected_offset
            elif int(total_in_split) <= 1:
                off = _center_offset(vol_shape, ch_need)
            else:
                offs, _ = _farthest_offsets_non_overlapping(
                    vol_shape, ch_need, int(total_in_split)
                )
                if not offs:
                    off = _center_offset(vol_shape, ch_need)
                else:
                    ix = min(max(0, int(unit_idx) - 1), len(offs) - 1)
                    off = offs[ix]
            win = _window_from_offset(vol_shape, off, ch_need)
            if win is None:
                raise RuntimeError("Invalid crop window")
            (z0, y0, x0), (z1, y1, x1) = win
            img_crop = np.asarray(img_arr[z0:z1, y0:y1, x0:x1])
            sz0, sz1, sy0, sy1, sx0, sx1 = _img_window_to_seg_slices(
                z0, z1, y0, y1, x0, x1, img_shape, seg_shape
            )
            seg_crop = np.asarray(seg_arr[sz0:sz1, sy0:sy1, sx0:sx1])
            seg_native = _resize_seg_to_match(seg_crop, img_crop.shape)
            img_out = resample_crop_to_isotropic(
                img_crop,
                spacing,
                order=1,
                out_spacing_nm_zyx=(16.0, 16.0, 16.0),
                max_voxels_zyx=(128, 128, 128),
            )
            seg_out = resample_crop_to_isotropic(
                seg_native,
                spacing,
                order=0,
                out_spacing_nm_zyx=(16.0, 16.0, 16.0),
                max_voxels_zyx=(128, 128, 128),
            )
            base_suffix = _crop_index_suffix(unit_idx, int(total_in_split))
            tag = _slug_token(dataset)
            resolved = str(resolved_suffix_override or "").strip() or mitole_resolve_foundation_suffix_disk_only(
                tag, img_dir, lbl_dir, base_suffix
            )
            write_h5(str(img_dir / f"{tag}{resolved}_im.h5"), img_out, f"{tag}{resolved}_im")
            write_h5(str(lbl_dir / f"{tag}{resolved}_seg.h5"), seg_out, f"{tag}{resolved}_seg")
            # Resolve real paths that write_h5 creates.
            img_dst = img_dir / f"{tag}{resolved}_0000.nii.gz"
            lbl_dst = lbl_dir / f"{tag}{resolved}.nii.gz"
            inst_dst = inst_dir / f"{tag}{resolved}_instance.nii.gz"
        except Exception:
            # Keep pipeline robust even for uncommon source formats.
            # Fallback preserves split counts by file materialization.
            safe_img_ext = image_ext or ".h5"
            safe_lbl_ext = label_ext or ".h5"
            img_dst = img_dir / f"{base_id}_im{safe_img_ext}"
            lbl_dst = lbl_dir / f"{base_id}_mito{safe_lbl_ext}"
            inst_dst = inst_dir / f"{base_id}_mito{safe_lbl_ext}"
            shutil.copy2(str(image_path), str(img_dst))
            shutil.copy2(str(label_path), str(lbl_dst))
            shutil.copy2(str(label_path), str(inst_dst))
        return {
            "nnunet_image_dst": str(img_dst),
            "nnunet_label_dst": str(lbl_dst),
            "nnunet_instance_dst": str(inst_dst),
            "split": split_name,
        }

    ds_idx = 0
    for pair in dataset_pairs:
        if callable(cancel_requested_fn) and cancel_requested_fn():
            raise RuntimeError("cancelled")
        dataset = str(pair.get("dataset") or "").strip()
        source_rel = str(pair.get("source") or "").strip()
        image_raw = str(pair.get("image_path") or "").strip()
        label_raw = str(pair.get("label_path") or "").strip()
        image_path = Path(image_raw).expanduser().resolve() if image_raw else None
        label_path = Path(label_raw).expanduser().resolve() if label_raw else None
        split = dataset_splits.get(dataset, {})
        n_requested = int(split.get("training", 0) or 0) + int(split.get("inference", 0) or 0)
        if n_requested <= 0:
            continue
        if image_path is None or label_path is None:
            resolved = mitole_resolve_pair_from_source_rel(source_rel)
            if resolved is None:
                skipped.append(f"{dataset}: no image/label pair found under source {source_rel}")
                if callable(log_fn):
                    log_fn(f"[WARN] {dataset}: no image/label pair found under source {source_rel}\n")
                continue
            image_path, label_path = resolved
        if not image_path.is_file() or not label_path.is_file():
            skipped.append(f"{dataset}: missing image/label file")
            if callable(log_fn):
                log_fn(f"[WARN] {dataset}: missing image/label file\n")
            continue
        if not str(image_path).startswith(str(base) + os.sep):
            skipped.append(f"{dataset}: image path outside MitoLE root")
            if callable(log_fn):
                log_fn(f"[WARN] {dataset}: image path outside MitoLE root\n")
            continue
        if not str(label_path).startswith(str(base) + os.sep):
            skipped.append(f"{dataset}: label path outside MitoLE root")
            if callable(log_fn):
                log_fn(f"[WARN] {dataset}: label path outside MitoLE root\n")
            continue

        ds_idx += 1
        if callable(log_fn):
            log_fn(
                f"[PROGRESS] dataset {ds_idx}/{n_catalog_ds}: {dataset} "
                f"(pairs={n_requested}; training={int(split.get('training', 0) or 0)} "
                f"inference={int(split.get('inference', 0) or 0)})\n"
            )
            _hdr_lines = [
                f"Dataset: {dataset}",
                f"Image path: {image_path}",
                f"Label path: {label_path}",
            ]
            tr_n0 = max(0, int(split.get("training", 0) or 0))
            ts_n0 = max(0, int(split.get("inference", 0) or 0))
            if tr_n0:
                _hdr_lines.append(f"Output images (training): {images_tr.resolve()}")
                _hdr_lines.append(f"Output labels (training): {labels_tr.resolve()}")
            if ts_n0:
                _hdr_lines.append(f"Output images (inference): {images_ts.resolve()}")
                _hdr_lines.append(f"Output labels (inference): {labels_ts.resolve()}")
            _hdr_lines.append("-" * 60)
            log_fn("\n".join(_hdr_lines) + "\n")

        tr_n = max(0, int(split.get("training", 0) or 0))
        ts_n = max(0, int(split.get("inference", 0) or 0))
        total_n = int(tr_n) + int(ts_n)
        pool_complete = all(
            mitole_global_vol_complete(dataset_root, dataset, k, total_n)
            for k in range(1, total_n + 1)
        )
        profile_done = _mitole_total_profile_completed(dataset_root, dataset, total_n)
        if pool_complete and not profile_done:
            _mitole_mark_total_profile_completed(dataset_root, dataset, total_n)
            profile_done = True
        if profile_done and pool_complete:
            if callable(log_fn):
                log_fn(
                    f"[SKIP] dataset {ds_idx}/{n_catalog_ds}: {dataset} "
                    f"(all {total_n} global foundation vol(s) already materialized for "
                    f"training={tr_n} inference={ts_n} under Dataset001_mito2)\n"
                )
            continue
        force_full_rematerialize = not profile_done
        if not force_full_rematerialize:
            mitole_reconcile_foundation_layout(dataset_root, dataset, tr_n, ts_n, log_fn)

        base_name = _slug_token(dataset)
        unique_name = base_name
        k = 2
        while unique_name in used_names:
            unique_name = f"{base_name}_{k}"
            k += 1
        used_names.add(unique_name)

        dst_im = images_dir / f"{unique_name}{image_path.suffix}"
        dst_lb = labels_dir / f"{unique_name}{label_path.suffix}"
        shutil.copy2(str(image_path), str(dst_im))
        shutil.copy2(str(label_path), str(dst_lb))
        nnunet_rows: list[dict[str, str]] = []
        all_offsets: list[tuple[int, int, int] | None] = []
        if total_n <= 1:
            all_offsets = [None] * total_n  # type: ignore[list-item]
        else:
            try:
                if str(project_root.resolve()) not in sys.path:
                    sys.path.insert(0, str(project_root.resolve()))
                fr = importlib.import_module("3data_downloader.openorganelle.foundation_resample")
                _farthest_offsets_non_overlapping = getattr(fr, "_farthest_offsets_non_overlapping")
                arr_img = _to_zyx(_load_volume(image_path))
                spacing = _spacing_zyx_from_path(image_path) or (16.0, 16.0, 16.0)
                foundation_crop_shapes = getattr(fr, "foundation_crop_shapes")
                ch_need = foundation_crop_shapes(
                    spacing,
                    tuple(int(x) for x in arr_img.shape[:3]),
                    (16.0 * 128.0, 16.0 * 128.0, 16.0 * 128.0),
                )
                offs, _ = _farthest_offsets_non_overlapping(
                    tuple(int(x) for x in arr_img.shape[:3]), ch_need, int(total_n)
                )
                if not offs:
                    all_offsets = [None] * total_n  # type: ignore[list-item]
                else:
                    all_offsets = list(offs)
                    while len(all_offsets) < total_n:
                        all_offsets.append(all_offsets[-1])
            except Exception:
                all_offsets = [None] * total_n  # type: ignore[list-item]
        planned_batch_suffixes = mitole_planned_foundation_suffixes(
            dataset_root, dataset, "training", total_n
        )

        staging_dirs = (images_tr, labels_tr, labels_tr_inst)
        tag_nm = _slug_token(dataset)
        _, ts_idx_list = _mitole_alternating_volume_indices(tr_n, ts_n)
        ts_idx_set = set(ts_idx_list)

        def _mitole_stem_main_vol_index(stem: str) -> int:
            mx = re.match(r"^_vol(\d+)", str(stem or ""), flags=re.IGNORECASE)
            return int(mx.group(1)) if mx else -1

        def _mitole_move_stems_for_inference_global_indices() -> None:
            labels_ts_inst.mkdir(parents=True, exist_ok=True)
            for vidx in sorted(ts_idx_set):
                moved_stems: set[str] = set()
                for pat in (f"{tag_nm}_vol{vidx}_0000.nii.gz", f"{tag_nm}_vol{vidx}_*_0000.nii.gz"):
                    for img_src in sorted(images_tr.glob(pat)):
                        if not img_src.is_file():
                            continue
                        stem = _mitole_foundation_stem_from_image_path(tag_nm, img_src)
                        if not stem or _mitole_stem_main_vol_index(stem) != int(vidx):
                            continue
                        if stem in moved_stems:
                            continue
                        moved_stems.add(stem)
                        lbl_src = labels_tr / f"{tag_nm}{stem}.nii.gz"
                        inst_src = labels_tr_inst / f"{tag_nm}{stem}_instance.nii.gz"
                        img_dst = images_ts / f"{tag_nm}{stem}_0000.nii.gz"
                        lbl_dst = labels_ts / f"{tag_nm}{stem}.nii.gz"
                        inst_dst = labels_ts_inst / f"{tag_nm}{stem}_instance.nii.gz"
                        if not lbl_src.is_file():
                            continue
                        if img_dst.is_file() and img_src.resolve() != img_dst.resolve():
                            continue
                        shutil.move(str(img_src), str(img_dst))
                        shutil.move(str(lbl_src), str(lbl_dst))
                        if inst_src.is_file():
                            shutil.move(str(inst_src), str(inst_dst))

        for k_vol in range(1, total_n + 1):
            if callable(cancel_requested_fn) and cancel_requested_fn():
                raise RuntimeError("cancelled")
            if (not force_full_rematerialize) and mitole_global_vol_complete(dataset_root, dataset, k_vol, total_n):
                if callable(log_fn):
                    log_fn(
                        f"[SKIP] {dataset}: global vol {k_vol}/{total_n} "
                        f"already complete (nnUNet pair on disk)\n"
                    )
                continue
            if callable(log_fn):
                log_fn(
                    f"[INFO] {dataset}: materializing global vol {k_vol}/{total_n} "
                    f"(staging: imagesTr; training={tr_n} inference={ts_n})\n"
                )
            if k_vol in ts_idx_set:
                infer_count += 1
            else:
                train_count += 1
            completed_new += 1
            if callable(progress_fn):
                progress_fn(completed_new, prog_denom, completed_new, dataset)
            _row = _materialize_split_pair(
                split_name="training",
                dataset=dataset,
                unit_idx=k_vol,
                total_in_split=total_n,
                selected_offset=all_offsets[k_vol - 1] if k_vol - 1 < len(all_offsets) else None,
                image_path=image_path,
                label_path=label_path,
                forced_nnunet_dirs=staging_dirs,
                resolved_suffix_override=(
                    planned_batch_suffixes[k_vol - 1]
                    if k_vol - 1 < len(planned_batch_suffixes)
                    else None
                ),
            )
            nnunet_rows.append(_row)
            if callable(log_fn):
                log_fn(
                    f"[DONE] {dataset}: global vol {k_vol}/{total_n}: "
                    f"{Path(_row['nnunet_image_dst']).name}  "
                    f"{Path(_row['nnunet_label_dst']).name}\n"
                )

        _mitole_move_stems_for_inference_global_indices()
        _mitole_mark_total_profile_completed(dataset_root, dataset, total_n)

        for row in nnunet_rows:
            img_p = Path(str(row.get("nnunet_image_dst") or ""))
            stem = _mitole_foundation_stem_from_image_path(tag_nm, img_p)
            if not stem:
                continue
            gvi = _mitole_stem_main_vol_index(stem)
            if gvi < 1:
                continue
            fin_split = "inference" if gvi in ts_idx_set else "training"
            row["split"] = fin_split
            if fin_split == "inference":
                row["nnunet_image_dst"] = str(images_ts / img_p.name)
                row["nnunet_label_dst"] = str(labels_ts / Path(str(row.get("nnunet_label_dst") or "")).name)
                inst_nm = Path(str(row.get("nnunet_instance_dst") or "")).name
                if inst_nm:
                    row["nnunet_instance_dst"] = str(labels_ts_inst / inst_nm)
            else:
                row["nnunet_image_dst"] = str(images_tr / img_p.name)
                row["nnunet_label_dst"] = str(labels_tr / Path(str(row.get("nnunet_label_dst") or "")).name)
                inst_nm = Path(str(row.get("nnunet_instance_dst") or "")).name
                if inst_nm:
                    row["nnunet_instance_dst"] = str(labels_tr_inst / inst_nm)
        if nnunet_rows:
            copied.append(
                {
                    "dataset": dataset,
                    "source": source_rel,
                    "image_src": str(image_path),
                    "label_src": str(label_path),
                    "image_dst": str(dst_im),
                    "label_dst": str(dst_lb),
                    "requested_crops": n_requested,
                    "training": int(split.get("training", 0) or 0),
                    "inference": int(split.get("inference", 0) or 0),
                    "nnunet_materialized": nnunet_rows,
                }
            )
            if callable(log_fn):
                log_fn(
                    f"[DONE] dataset {ds_idx}/{n_catalog_ds}: {dataset} "
                    f"(pairs={n_requested}; new nnUNet stacks={len(nnunet_rows)})\n"
                )

    manifest = {
        "ok": True,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "copied_pairs": len(copied),
        "nnunet_dataset_root": str(dataset_root),
        "nnunet_training_units": train_count,
        "nnunet_inference_units": infer_count,
        "skipped": skipped,
        "rows": copied,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ds_json_path: Path | None = None
    try:
        if str(project_root.resolve()) not in sys.path:
            sys.path.insert(0, str(project_root.resolve()))
        exp = importlib.import_module("3data_downloader.downloader_common.nnunet_labeled_export")
        sync_dataset_json = getattr(exp, "sync_dataset_json")
        ds_json_path = sync_dataset_json(dataset_root)
    except Exception as exc:
        if callable(log_fn):
            log_fn(f"[WARN] dataset.json sync failed: {exc}\n")
    if callable(log_fn):
        log_fn(
            f"[SUMMARY] catalogue_rows_materialized={len(copied)}  "
            f"catalogue_warnings={len(skipped)}  "
            f"training_vols_written={train_count}  inference_vols_written={infer_count}\n"
        )
        if ds_json_path is not None:
            log_fn(f"[DONE] dataset.json synced: {ds_json_path}\n")
    return manifest

