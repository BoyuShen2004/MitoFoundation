#!/usr/bin/env python3
"""Smoke test: validate Stage 1→4 outputs are sane for PyTC supervised training.

Run from repo root:
    python scripts/smoke_test_pipeline.py

Exit 0 = all checks pass. Non-zero = at least one check failed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from config.paths import nnunet_dataset_root  # noqa: E402

PASS = "✓"
FAIL = "✗"
WARN = "!"
_failures: list[str] = []
_warnings: list[str] = []


def ok(msg: str) -> None:
    print(f"  {PASS} {msg}")


def fail(msg: str) -> None:
    _failures.append(msg)
    print(f"  {FAIL} {msg}")


def warn(msg: str) -> None:
    _warnings.append(msg)
    print(f"  {WARN} {msg}")


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── Stage 1: probe JSON ────────────────────────────────────────────────────────

section("Stage 1 — Scraper outputs")
probe_path = REPO / "1web_scraper_01" / "outputs" / "OpenOrganelle.probe.json"
if not probe_path.is_file():
    fail(f"probe JSON missing: {probe_path}")
else:
    try:
        probe = json.loads(probe_path.read_text())
        datasets = probe.get("datasets") or {}
        n = len(datasets)
        if n < 10:
            fail(f"probe JSON has only {n} datasets (expected ≥ 10)")
        else:
            ok(f"probe JSON: {n} datasets")
        batches = probe.get("scrape_batches") or []
        if not batches:
            warn("probe JSON: no scrape_batches recorded")
        else:
            ok(f"probe JSON: {len(batches)} scrape batch(es)")
    except Exception as exc:
        fail(f"probe JSON parse error: {exc}")

canonical_md = REPO / "1web_scraper_01" / "outputs" / "OpenOrganelle.md"
if canonical_md.is_file():
    ok(f"canonical markdown: {canonical_md.name}")
else:
    warn("canonical markdown (OpenOrganelle.md) missing — run stage 1 first")

# ── Stage 2: catalog DB ────────────────────────────────────────────────────────

section("Stage 2 — Catalog DB")
db_path = REPO / "2database_builder" / "outputs" / "databases" / "OpenOrganelle.db"
if not db_path.is_file():
    warn(f"catalog DB not yet built — run: python 2database_builder/master/agent.py")
else:
    try:
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        n_datasets = conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
        n_resolved = conn.execute("SELECT COUNT(*) FROM dataset_resolved WHERE ready_labeled=1").fetchone()[0]
        n_em_only  = conn.execute("SELECT COUNT(*) FROM dataset_resolved WHERE ready_em_only=1").fetchone()[0]
        conn.close()
        if n_datasets < 5:
            fail(f"catalog DB: only {n_datasets} datasets (expected ≥ 5)")
        else:
            ok(f"catalog DB: {n_datasets} datasets, {n_resolved} labeled-ready, {n_em_only} with EM")
        if n_resolved == 0:
            fail("catalog DB: zero labeled-ready datasets — run stage 2 with S3 probe")
    except Exception as exc:
        fail(f"catalog DB error: {exc}")

# ── Registry ───────────────────────────────────────────────────────────────────

section("Registry (data/registry.sqlite)")
reg_path = REPO / "data" / "registry.sqlite"
if not reg_path.is_file():
    warn("registry not yet created — run: python 2database_builder/master/agent.py")
else:
    try:
        from agent.orchestration.registry.schema import connect as reg_connect
        rconn = reg_connect(reg_path)
        n_providers = rconn.execute("SELECT COUNT(*) FROM providers").fetchone()[0]
        n_reg_datasets = rconn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
        n_assets = rconn.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
        n_complete_dl = rconn.execute("SELECT COUNT(*) FROM downloads WHERE status='complete'").fetchone()[0]
        n_pp = rconn.execute("SELECT COUNT(*) FROM preprocess_runs WHERE status='complete'").fetchone()[0]
        rconn.close()
        ok(f"registry: {n_providers} provider(s), {n_reg_datasets} datasets, {n_assets} assets")
        ok(f"registry: {n_complete_dl} complete download(s), {n_pp} complete preprocess run(s)")
        if n_reg_datasets == 0:
            warn("registry is empty — run: python 2database_builder/master/agent.py")
    except Exception as exc:
        fail(f"registry error: {exc}")

# ── Stage 3: downloads (legacy data/raw batches or direct-to-training) ───────

section("Stage 3 — Download outputs")
raw_base = REPO / "data" / "raw"
raw_h5_files: list[Path] = []
if not raw_base.is_dir():
    warn("data/raw/ missing (legacy batch layout optional)")
else:
    for run_dir in sorted(raw_base.iterdir()):
        if not run_dir.is_dir():
            continue
        img_dir = run_dir / "images"
        lbl_dir = run_dir / "labels"
        if img_dir.is_dir():
            imgs = sorted(img_dir.glob("*_im.h5"))
            lbls = sorted(lbl_dir.glob("*_seg.h5")) if lbl_dir.is_dir() else []
            raw_h5_files.extend(imgs)
            ok(f"{run_dir.name}: {len(imgs)} image(s), {len(lbls)} label(s)")
train_imgs = nnunet_dataset_root(REPO) / "imagesTr"
if not raw_h5_files and train_imgs.is_dir():
    direct = sorted(train_imgs.glob("*_0000.nii.gz")) + sorted(train_imgs.glob("*_im.h5"))
    raw_h5_files.extend([p for p in direct if p.suffix == ".h5"])
    if direct:
        ok(f"Dataset001_mito2/imagesTr: {len(direct)} stack file(s)")
if not raw_h5_files:
    warn("No legacy *_im.h5 under data/raw/*/images (NIfTI-only pipeline may skip H5 spot-check)")

# ── Stage 4: training data ────────────────────────────────────────────────────

section("Stage 4 — Training data")
nn_base = nnunet_dataset_root(REPO)
legacy_pre = REPO / "data" / "training"
ft_datalist = nn_base / "finetune_datalist.json"
if not ft_datalist.is_file():
    ft_datalist = legacy_pre / "finetune_datalist.json"

if nn_base.is_dir():
    n_nii = len(list((nn_base / "imagesTr").glob("*.nii.gz"))) if (nn_base / "imagesTr").is_dir() else 0
    ok(f"Dataset001_mito2 imagesTr: {n_nii} .nii.gz file(s) (0 is ok if downloads not run)")
else:
    warn("Dataset001_mito2 directory missing")

if not ft_datalist.is_file():
    warn("finetune_datalist.json missing — run stage 4 preprocess when applicable")
else:
    try:
        ft = json.loads(ft_datalist.read_text())
        n_train = len(ft.get("training") or [])
        n_val = len(ft.get("validation") or [])
        n_test = len(ft.get("test") or [])
        if n_train == 0 and n_val == 0:
            fail("finetune_datalist.json: 0 training entries — check stage 4 output")
        else:
            ok(f"finetune_datalist.json: train={n_train}, val={n_val}, test={n_test}")
        first = (ft.get("training") or ft.get("test") or [None])[0]
        if first:
            img_p = Path(first.get("image", ""))
            lbl_p = Path(first.get("label", ""))
            res = first.get("resolution")
            if not img_p.is_file():
                fail(f"datalist entry image missing on disk: {img_p}")
            else:
                ok(f"datalist sample image exists: {img_p.name}")
            if not lbl_p.is_file():
                fail(f"datalist entry label missing on disk: {lbl_p}")
            else:
                ok(f"datalist sample label exists: {lbl_p.name}")
            if not isinstance(res, list) or len(res) != 3 or not all(v > 0 for v in res):
                warn(f"datalist sample resolution suspicious: {res}")
            else:
                ok(f"datalist sample resolution: {res} nm (ZYX)")
            shape = first.get("shape_zyx")
            if shape and any(s < 16 for s in shape):
                warn(f"datalist sample shape very small: {shape} — may be a crop issue")
            elif shape:
                ok(f"datalist sample shape: {shape} (ZYX)")
    except Exception as exc:
        fail(f"finetune_datalist.json parse error: {exc}")

# H5 content spot-check
if raw_h5_files:
    section("H5 content spot-check (first raw file)")
    spot = raw_h5_files[0]
    try:
        import h5py
        with h5py.File(spot, "r") as hf:
            keys = list(hf.keys())
            if not keys:
                fail(f"{spot.name}: empty H5 file")
            else:
                ds = next(iter(hf.values()))
                shape = ds.shape if hasattr(ds, "shape") else None
                ok(f"{spot.name}: shape={shape}, internal keys={keys[:3]}")
                if shape and any(s == 0 for s in shape):
                    fail(f"{spot.name}: zero-size dimension in shape {shape}")
    except ImportError:
        warn("h5py not installed — skipping H5 content check")
    except Exception as exc:
        fail(f"H5 spot-check failed: {exc}")

# nnUNet raw NIfTI spot-check
section("nnUNet raw dataset check")
if not nn_base.is_dir():
    fail(f"nnUNet dataset missing: {nn_base}")
else:
    tr_i = nn_base / "imagesTr"
    tr_l = nn_base / "labelsTr"
    ts_i = nn_base / "imagesTs"
    ts_l = nn_base / "labelsTs"
    n_tr_i = len(list(tr_i.glob("*_0000.nii.gz"))) if tr_i.is_dir() else 0
    n_tr_l = len(list(tr_l.glob("*.nii.gz"))) if tr_l.is_dir() else 0
    n_ts_i = len(list(ts_i.glob("*_0000.nii.gz"))) if ts_i.is_dir() else 0
    n_ts_l = len(list(ts_l.glob("*.nii.gz"))) if ts_l.is_dir() else 0
    if n_tr_i == 0 or n_tr_l == 0:
        fail("nnUNet training set is empty under Dataset001_mito2")
    else:
        ok(f"nnUNet train: imagesTr={n_tr_i}, labelsTr={n_tr_l}")
    if n_ts_i == 0 or n_ts_l == 0:
        warn("nnUNet inference set empty under Dataset001_mito2 imagesTs/labelsTs")
    else:
        ok(f"nnUNet test: imagesTs={n_ts_i}, labelsTs={n_ts_l}")

# ── Summary ────────────────────────────────────────────────────────────────────

section("Summary")
if _warnings:
    print(f"\n{WARN} Warnings ({len(_warnings)}):")
    for w in _warnings:
        print(f"   {WARN} {w}")
if _failures:
    print(f"\n{FAIL} Failures ({len(_failures)}):")
    for f_ in _failures:
        print(f"   {FAIL} {f_}")
    print("\nSmoke test FAILED — address failures above.\n")
    sys.exit(1)
else:
    print(f"\n{PASS} All checks passed ({len(_warnings)} warning(s)).\n")
    sys.exit(0)
