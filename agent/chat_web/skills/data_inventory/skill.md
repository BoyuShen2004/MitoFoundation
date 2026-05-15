---
id: data_inventory
title: Local data inventory
label: Data inventory
---

Purpose
- Summarize what has been downloaded under `data/` using concrete path structure and registry evidence.

Actual data/ layout (from code, not assumptions)
  data/training/images/   -- EM crops: <dataset_tag>_vol<N>_im.h5 (canonical stage-3/4 output)
  data/training/labels/   -- mito seg crops: <dataset_tag>_vol<N>_seg.h5
  data/raw/<run>/         -- legacy raw batch subdirs (older downloads)
    images/ labels/
  data/raw_volumes/       -- stub-mode only (download_manifest.json, not volumes)
  data/inference/         -- inference outputs (stage-5, optional)
  data/ml_runs/           -- evaluation artifacts (benchmark_stub.json)

Primary evidence sources
  - Registry:    data/registry.sqlite (batch_items: local_path, status, stable_id)
  - File scan:   data/training/images/*.h5, data/training/labels/*.h5 (paired _im.h5 / _seg.h5)
  - Studio API:  GET /api/studio/data/inspect (scans raw, training, inference)

Report format
  - Count paired _im.h5 / _seg.h5 under data/training/ as "complete labeled pairs".
  - Count unpaired _im.h5 (no matching _seg.h5) as "EM only / incomplete".
  - Report registry batch_items where status != present as "pending or missing".
  - For data/raw/ subdirs: interpret folder names as run timestamps if they match openorganelle_mito_<timestamp>.
  - File type summary: H5 (lzf-compressed, up to 512^3 voxels), NRRD, NII, JSON.
  - Size estimates from file scan only; do not infer from registry alone.

Rules
  - Prefer registry evidence for status; prefer filesystem scan for counts and sizes.
  - If data/training/ is empty but registry shows present items: flag the discrepancy explicitly.
  - If uncertain, explicitly label as "inferred from path naming" or "not confirmed by registry".

Catalog-style question support
  - For questions like sample type, organism, modality, or source composition, prioritize structured metadata from:
    - `data/registry.sqlite` (`datasets.metadata_json`, provider joins),
    - latest `*.probe.json` dataset records,
    - built SQLite catalogs under `2database_builder/outputs/databases/`.
  - Return both:
    - count summary (how many match),
    - a small sample of matching dataset IDs/names.
  - If the field exists in one source family but not another, state the mismatch instead of guessing.

Downloaded & Predicted filters/count questions
  - For "how many under each filter" (training/inference/postprocessed/instance/border-contour/use-in-model), answer by intersecting:
    - location-based evidence (`data/training/`, `data/inference/`, postprocess output dirs),
    - registry/status evidence (`batch_items.status`, dataset flags),
    - training inclusion evidence (YAML list + `hidden_from_training`).
  - If a requested filter label is not explicitly represented in current artifacts, answer:
    "Not found in current artifacts: <filter label mapping>" and provide closest measurable proxy.
