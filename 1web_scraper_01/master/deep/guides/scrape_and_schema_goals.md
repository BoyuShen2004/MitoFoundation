## OpenOrganelle — extraction goals for supervised mito training

**Primary purpose:** surface per-dataset EM + non-prediction mito segmentation paths for downstream supervised mitochondria model training (PyTorch Connectomics). Every report must clearly distinguish training-quality GT labels from prediction/inference outputs.

---

### Training-quality mito label rules (apply to every dataset)

A label is **training-quality** (`mito_label_quality=good`) when **all** of the following hold:

1. Name or path includes `mito` (case-insensitive).
2. Not a prediction/inference output — exclude names/paths matching any of:
   - `_pred`, `mito_pred`, `prediction`, `inference`, `unproofread`, `empanada`
   - stored under `labels/inference/` in nested reconstruction paths
   - layer role token `predictions` in the layer inventory
3. Dense volumetric raster (not a mesh): layer type `image`, format `n5` or `zarr`.

**Record as `mito_seg_path`:** the best non-prediction mito path at full resolution (`s0`).  
**Record as `mito_pred_path`:** the best prediction-only path (separate field; null if absent).  
**Set `mito_label_quality`:** `”good”` when `mito_seg_path` is non-prediction; `”prediction_only”` when only prediction exists; `null` when no mito label found.  
**Set `mito_segmentation_kind`:** `”instance”` (unique ID per object) or `”semantic”` (binary/class mask).

**OpenOrganelle-specific nuances:**
- `mito_seg` (or `mito-mem_seg`) → training-quality.
- `mito_pred` → prediction only; do NOT use as `mito_seg_path`.
- `empanada-mito_seg` → automated model output; treat as prediction even though name includes `_seg`.
- `recon-1/labels/inference/mito_seg/s0` → prediction (under `inference/` subtree); treat as prediction.
- `recon-1/labels/groundtruth/mito_seg/s0` → training-quality.
- A dataset with only `recon-1/labels/inference/` has no training-quality mito label; set `mito_seg_path=null`.

**EM + label alignment rule:** when picking `download_volume_url` / `em_image_path`, prefer the path from the **same storage root** (`.n5` or `.zarr/recon-N`) as the chosen `mito_seg_path`. This avoids shape mismatches from mixed compression/scale conventions.

---

### Required download fields per dataset

1. `download_volume_url` — EM image path at `s1` scale (preferred for download).
2. `download_mito_mask_url` — Non-prediction mito mask at `s0` (null if absent).
3. `download_mito_prediction_url` — Prediction-only mito path (separate from mask).
4. `download_volume_leaf_url`, `download_mito_mask_leaf_url` — leaf S3 key when resolvable.
5. `*_scale_key` — `s0`, `s1`, etc. for each path when determinable.

### Layer token metadata

Capture in the appendix when visible:
- `image:<name>[format=...][stage=...][content_type=...][url=...]`
- `mesh:<name>[format=...][url=...]`

Prefer factual API/object-store evidence; do not invent paths.

---

### Grid normalization

**Heuristics only (implemented):** `master/deep/grid_llm_normalize.py` → `apply_grid_normalization_to_access()` rewrites `grid.grid_dimensions` / `grid.grid_dimensions_unit` using `0inventory/dataset_inventory.py` voxel heuristics (no LLM batch in code). Older docs mentioning `OPENORGANELLE_LLM_GRID_NORMALIZE` are obsolete.

### Incremental → registry → database_builder

Each scrape writes one new `outputs/OpenOrganelle_NN.md` per run. When new slugs appear vs `OpenOrganelle.probe.json`, the report includes `## Database builder — new datasets to add` before the Appendix.

After scrape, the pipeline automatically:
1. Runs `2database_builder/master/agent.py` → builds `OpenOrganelle.db` + fills `dataset_resolved`.
2. Syncs `data/registry.sqlite` with new datasets + assets (registry ON by default; disable with `MITO2_DISABLE_REGISTRY=1`).

S3 probe verbosity is off by default (`S3_PROBE_VERBOSE=1` to enable per-dataset lines).

