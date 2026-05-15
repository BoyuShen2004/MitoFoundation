#!/usr/bin/env python3
"""Evaluate watershed instance predictions against labelsTs-instance ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def load_nii(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    return np.asarray(img.dataobj)


def normalize_stem(name: str) -> str:
    s = name
    for suf in (".nii.gz", ".nii"):
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    if s.endswith("_0000"):
        s = s[: -len("_0000")]
    if s.endswith("_instance"):
        s = s[: -len("_instance")]
    return s


def compute_binary(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    p = pred > 0
    g = gt > 0
    tp = float(np.sum(p & g))
    fp = float(np.sum(p & (~g)))
    fn = float(np.sum((~p) & g))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {
        "binary_precision": precision,
        "binary_recall": recall,
        "binary_f1": f1,
        "binary_iou": iou,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="Postprocessed instance predictions directory")
    ap.add_argument("--gt_dir", required=True, help="Ground-truth instance label directory")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir).expanduser().resolve()
    gt_dir = Path(args.gt_dir).expanduser().resolve()

    pred_files = sorted(list(pred_dir.glob("*.nii.gz")) + list(pred_dir.glob("*.nii")))
    gt_files = sorted(list(gt_dir.glob("*.nii.gz")) + list(gt_dir.glob("*.nii")))
    gt_map = {normalize_stem(p.name): p for p in gt_files}

    print(f"[eval] pred_dir={pred_dir}")
    print(f"[eval] gt_dir={gt_dir}")
    print(f"[eval] pred_files={len(pred_files)} gt_files={len(gt_files)}")

    rows: list[dict[str, float | str]] = []
    for p in pred_files:
        key = normalize_stem(p.name)
        gt = gt_map.get(key)
        if gt is None:
            print(f"[eval] skip {p.name} (no matching gt)")
            continue
        try:
            pred_arr = np.squeeze(load_nii(p))
            gt_arr = np.squeeze(load_nii(gt))
            if pred_arr.shape != gt_arr.shape:
                print(f"[eval] skip {p.name} shape mismatch pred={pred_arr.shape} gt={gt_arr.shape}")
                continue
            m = compute_binary(pred_arr, gt_arr)
            row: dict[str, float | str] = {"case": key, **m}
            rows.append(row)
            print(
                f"[eval] {key} "
                f"f1={m['binary_f1']:.4f} "
                f"precision={m['binary_precision']:.4f} "
                f"recall={m['binary_recall']:.4f} "
                f"iou={m['binary_iou']:.4f}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"[eval] failed {p.name}: {e}")

    summary = {
        "n_cases": len(rows),
        "mean_binary_f1": float(np.mean([r["binary_f1"] for r in rows])) if rows else 0.0,
        "mean_binary_precision": float(np.mean([r["binary_precision"] for r in rows])) if rows else 0.0,
        "mean_binary_recall": float(np.mean([r["binary_recall"] for r in rows])) if rows else 0.0,
        "mean_binary_iou": float(np.mean([r["binary_iou"] for r in rows])) if rows else 0.0,
    }
    print(f"[eval] summary={summary}")
    # Machine-readable lines consumed by Studio API to return metrics directly to frontend.
    print(f"EVAL_SUMMARY_JSON={json.dumps(summary, separators=(',', ':'))}")
    print(f"EVAL_CASES_JSON={json.dumps(rows, separators=(',', ':'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

