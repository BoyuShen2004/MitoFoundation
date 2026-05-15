"""Shared training vs inference window counts for labeled Stage-3 downloaders.

OpenOrganelle and BossDB generated scripts use the same clamping and alternating
volume assignment so multi-crop runs land in ``imagesTr``/``labelsTr`` vs
``imagesTs``/``labelsTs`` consistently.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def clamped_train_infer(
    split: dict | None,
    *,
    default_training: int = 1,
    default_inference: int = 0,
) -> tuple[int, int]:
    """Return ``(n_training_windows, n_inference_windows)`` with caps matching OpenOrganelle."""
    s = split or {}
    try:
        tr = max(0, min(16, int(s.get("training", default_training))))
    except Exception:
        tr = max(0, min(16, int(default_training)))
    try:
        ts = max(0, min(16, int(s.get("inference", default_inference))))
    except Exception:
        ts = max(0, min(16, int(default_inference)))
    if tr + ts > 16:
        ts = max(0, 16 - tr)
    return tr, ts


def labeled_nnunet_vol_outputs_count_ge(
    dataset_stable_key: str,
    *,
    nnunet_raw_root: str | Path,
    split: str,
    min_pairs: int,
) -> bool:
    """OpenOrganelle-style filesystem check per split folder.

    Counts complete `{tag}_vol*… nnUNet image stacks with paired label (+ instance when that dir exists).
    Robust for alternating layouts where training may hold `_vol1` and `_vol3` but not contiguous `_vol1.._vol{n}`.
    """
    need = int(min_pairs)
    if need <= 0:
        return True
    tag = str(dataset_stable_key or "").replace("/", "_").replace("-", "_")
    root = Path(str(nnunet_raw_root))
    is_inf = str(split).strip().lower() == "inference"
    img_dir = root / ("imagesTs" if is_inf else "imagesTr")
    lbl_dir = root / ("labelsTs" if is_inf else "labelsTr")
    inst_dir = root / ("labelsTs-instance" if is_inf else "labelsTr-instance")
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        return False
    imgs = sorted(img_dir.glob(f"{tag}_vol*_0000.nii.gz"))
    if not imgs:
        imgs = sorted(img_dir.glob(f"{tag}*_0000.nii.gz"))
    if not imgs:
        return False
    valid = 0
    for img in imgs:
        base = img.name[: -len("_0000.nii.gz")] if img.name.endswith("_0000.nii.gz") else img.name
        lbl = lbl_dir / f"{base}.nii.gz"
        inst = inst_dir / f"{base}_instance.nii.gz"
        if lbl.is_file() and (inst.is_file() or not inst_dir.is_dir()):
            valid += 1
    return valid >= max(1, need)


def alternating_volume_indices(n_training: int, n_inference: int) -> tuple[list[int], list[int]]:
    """1-based volume indices for training vs inference (alternating, training first)."""
    n_training = max(0, int(n_training))
    n_inference = max(0, int(n_inference))
    total = n_training + n_inference
    tr_idx: list[int] = []
    ts_idx: list[int] = []
    rem_tr = n_training
    rem_ts = n_inference
    want_tr = True
    for vol_idx in range(1, total + 1):
        if want_tr and rem_tr > 0:
            tr_idx.append(vol_idx)
            rem_tr -= 1
        elif (not want_tr) and rem_ts > 0:
            ts_idx.append(vol_idx)
            rem_ts -= 1
        elif rem_tr > 0:
            tr_idx.append(vol_idx)
            rem_tr -= 1
        elif rem_ts > 0:
            ts_idx.append(vol_idx)
            rem_ts -= 1
        want_tr = not want_tr
    return tr_idx, ts_idx


def foundation_global_volume_pool_complete(
    dataset_root: str | Path,
    tag: str,
    n_total: int,
) -> bool:
    """True when global volumes ``_vol1.._vol{n_total}`` exist as complete pairs in *either* split.

    This intentionally ignores train/inference assignment and only validates that the physical crop pool
    already exists on disk. It is used to avoid double-download when users change split counts while
    keeping the same total crop count (e.g. 1+1 -> 2+0).
    """
    total = max(0, int(n_total))
    if total <= 0:
        return True
    root = Path(str(dataset_root))
    t = str(tag)
    for k in range(1, total + 1):
        ok_any = False
        for split_inf in (False, True):
            img_dir = root / ("imagesTs" if split_inf else "imagesTr")
            lbl_dir = root / ("labelsTs" if split_inf else "labelsTr")
            inst_dir = root / ("labelsTs-instance" if split_inf else "labelsTr-instance")
            if not img_dir.is_dir() or not lbl_dir.is_dir():
                continue
            for pattern in (f"{t}_vol{k}_0000.nii.gz", f"{t}_vol{k}_*_0000.nii.gz"):
                for img_p in sorted(img_dir.glob(pattern)):
                    if not img_p.is_file() or not img_p.name.endswith("_0000.nii.gz"):
                        continue
                    base = img_p.name[: -len("_0000.nii.gz")]
                    if not base.startswith(t):
                        continue
                    stem = base[len(t) :]
                    mx = re.match(r"^_vol(\d+)", stem, flags=re.IGNORECASE)
                    if not mx or int(mx.group(1)) != int(k):
                        continue
                    lbl_p = lbl_dir / f"{base}.nii.gz"
                    inst_p = inst_dir / f"{base}_instance.nii.gz"
                    if lbl_p.is_file() and (inst_p.is_file() or not inst_dir.is_dir()):
                        ok_any = True
                        break
                if ok_any:
                    break
            if ok_any:
                break
        if not ok_any:
            return False
    return True


def _global_crop_profiles_dataset_json_path(dataset_root: str | Path) -> Path:
    return Path(str(dataset_root)) / "dataset.json"


def global_crop_profile_completed(dataset_root: str | Path, dataset_key: str, n_total: int) -> bool:
    """True iff dataset has a recorded completed run for this global crop total."""
    total = max(0, int(n_total))
    if total <= 0:
        return True
    p = _global_crop_profiles_dataset_json_path(dataset_root)
    if not p.is_file():
        return False
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    vals = (((raw or {}).get("mito2_global_crop_profiles") or {}).get(str(dataset_key)) or [])
    done = {int(v) for v in vals if str(v).isdigit()}
    return int(total) in done


def mark_global_crop_profile_completed(dataset_root: str | Path, dataset_key: str, n_total: int) -> None:
    """Record dataset completion for this global crop total inside ``dataset.json``."""
    total = max(0, int(n_total))
    if total <= 0:
        return
    p = _global_crop_profiles_dataset_json_path(dataset_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw: dict[str, object] = {}
    if p.is_file():
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                raw = dict(obj)
        except Exception:
            raw = {}
    prof = raw.get("mito2_global_crop_profiles")
    if not isinstance(prof, dict):
        prof = {}
    key = str(dataset_key)
    vals = prof.get(key) if isinstance(prof, dict) else None
    cur = set(int(x) for x in vals if str(x).isdigit()) if isinstance(vals, list) else set()
    cur.add(int(total))
    prof[key] = sorted(cur)
    raw["mito2_global_crop_profiles"] = prof
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)
