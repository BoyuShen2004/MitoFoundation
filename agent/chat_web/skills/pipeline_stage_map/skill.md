---
id: pipeline_stage_map
title: Pipeline stages and output paths (code-derived)
label: Pipeline map
---

Purpose
- Ground chat answers about data/ codepaths, generated output artifacts, and stage provenance using
  the concrete map below. All entries were derived by reading source files, not from assumptions.

Data directory layout (authoritative from code)
  data/
    training/           # LABELED_BASE = TRAINING_ROOT = data/training/ (openorganelle/agent.py)
      images/           # EM crops: lzf-compressed HDF5, 16nm iso after foundation_resample
        <dataset_tag>_vol<N>_im.h5
      labels/           # mito seg crops: lzf-compressed HDF5, binary (0/1)
        <dataset_tag>_vol<N>_seg.h5
    raw/                # legacy raw batch subdirs (studio inspect also scans here)
      <run_folder>/
        images/         # EM stacks from older batch runs (_im.h5 pattern)
        labels/
    raw_volumes/        # stub-mode only (download_manifest.json, no real volumes)
    inference/          # stage-5 inference outputs (optional)
    ml_runs/
      benchmark_stub.json

Stage -> output map

  Stage 1 (scrape)
    Module:   1web_scraper_01/master/
    Outputs:  1web_scraper_01/outputs/<site>.probe.json
              1web_scraper_01/websites/<slug>/site.md

  Stage 2 (database builder / DB)
    Module:   2database_builder/master/
    Outputs:  2database_builder/outputs/databases/OpenOrganelle.db  (dataset_resolved table)
              data/registry.sqlite                                  (central registry)

  Stage 2 -> registry ingestion
    Module:   orchestration/registry/providers/openorganelle.py:OpenOrganelleProvider
    Reads:    1web_scraper_01/outputs/OpenOrganelle.probe.json
              2database_builder/outputs/databases/OpenOrganelle.db
    Writes:   data/registry.sqlite (providers, datasets, assets tables)

  Stage 3 (download script generation)
    Module:   3data_downloader/downloader_master/scriptgen.py:write_generated_script()
    Outputs:  3data_downloader/outputs/download_<site>_<scope>.py
    On run:   generated script writes data/raw_volumes/download_manifest.json (stub mode)

  Stage 3 (actual OpenOrganelle download, labeled mode)
    Module:   3data_downloader/openorganelle/agent.py:download_openorganelle_data()
              3data_downloader/openorganelle/foundation_resample.py:download_foundation_labeled()
    Reads:    2database_builder/outputs/databases/OpenOrganelle.db (dataset_resolved: img_path, seg_path, voxel_size_nm)
    Writes:   data/training/images/<tag>_vol<N>_im.h5  (EM crop, lzf HDF5)
              data/training/labels/<tag>_vol<N>_seg.h5 (mito seg, lzf HDF5)
              3data_downloader/openorganelle/download_openorganelle_labeled.md (metadata sidecar)
    Crop:     physical 8192 nm edge; output = min(512, round(8192/spacing_per_axis)) voxels @ 16nm iso
    Registry: 0inventory/download_history.py:record_openorganelle_download_batch()
              -> writes download_batches + batch_items to registry.sqlite

  Stage 4 (preprocess)
    Module:   3data_downloader/downloader_master/preprocess_agent.py
    Inputs:   data/training/images/*_im.h5 (from stage 3)
    Outputs:  data/training/images/<key>_im.h5  (preprocessed, voxel attrs embedded)
              data/training/labels/<key>_seg.h5 (preprocessed)
    Also:     5model_training/configuration/mito_openorganelle_foundation.yaml
              (updated by training_yaml_sync.py with new basenames + median resolution)
    Key rule: <dataset>_vol<N>_im.h5 -> canonical key = <dataset>_vol<N>
              expected pair: data/training/images/<key>_im.h5 + data/training/labels/<key>_seg.h5

  Stage 5 (training)
    Config:   5model_training/configuration/mito_openorganelle_foundation.yaml
    Reads:    data/training/images/<basename>.h5
              data/training/labels/<basename>.h5
              (basenames listed under train.data.train.image / .label in YAML)
    Outputs:  5model_training/outputs/mito_openorganelle_foundation/checkpoints/

Registry provenance tables (data/registry.sqlite)
  download_batches  batch_id, provider, profile_hash, profile_json, run_folder,
                    status (in_progress|complete|failed), download_asset_completions
  batch_items       batch_db_id -> download_batches, stable_id, asset_type (em_volume|mito_seg),
                    local_path, status (pending|present|missing_or_deleted_local|failed)
  datasets          stable_id, hidden_from_training (0|1)
  downloads         asset_id, local_path, status, download_profile_hash
  preprocess_runs   dataset_id, input_fingerprint, status, output_paths_json
  Profile hash:     deterministic of (n_crops, chunk_zyx, voxel_nm, mode="labeled", foundation=True)
  Default path:     MITO2_REGISTRY_PATH env or data/registry.sqlite

Studio data API (live inspection evidence)
  GET /api/studio/data/inspect     -- scans data/raw/, data/training/, data/inference/ for H5/NRRD/NII/PT/JSON
  GET /api/studio/datasets/status  -- provenance from registry incl. hidden_from_training
  GET /api/studio/download-batches -- download_batches rows with item counts

How to answer common data questions
  "What is in data/?"
    -> Check data/training/images/ and data/training/labels/ first (canonical).
    -> Then data/raw/ (legacy runs).
    -> Evidence: registry batch_items.local_path or file scan.
  "Was dataset X downloaded?"
    -> Query batch_items WHERE stable_id=X AND asset_type IN (em_volume, mito_seg) AND status=present.
  "What produced file F?"
    -> Find batch_item WHERE local_path=F; join download_batches for batch_id and profile.
  "What does the generated script download?"
    -> Read 3data_downloader/outputs/download_*.py DATASETS variable.
  "What are the training inputs?"
    -> Read 5model_training/configuration/mito_openorganelle_foundation.yaml
       train.data.train.image and train.data.train.label.
  "Is preprocess done?"
    -> Check preprocess_runs.status, OR verify paired _im.h5 + _seg.h5 both present in data/training/.

Guardrails
- Never claim files exist in data/ without filesystem evidence (data/inspect result or direct path check).
- For registry provenance claims, name the table and column used.
- If registry.sqlite is absent: "Not found in current artifacts: data/registry.sqlite".
- Training YAML basenames are authoritative for "what is a training input" -- do not infer from filesystem alone.
- Canonical key pattern (<dataset>_vol<N>) must be confirmed from file names, not assumed.

Functional Q&A coverage (what mitoFoundation does)
- App purpose scope: website scrape/catalog building, local+remote data inventory, download/preprocess, model training, and execution reporting.
- For "what does module/page/stage do", answer by mapping user wording to stage modules above, then list:
  - inputs (files/tables),
  - actions (script/endpoint),
  - outputs (paths/tables),
  - next-stage dependency.
- For UI-control questions (buttons/actions), ground on route handlers and studio endpoints:
  `agent/chat_web/app/routes.py`, `agent/chat_web/app/studio_api.py`, `agent/chat_web/app/pipeline_chat.py`.

Data / Model / Downloaded+Predicted question map
- Data (inventory + scrape/local HPC pipelines)
  - inventory/status: `data/registry.sqlite` (`providers`, `datasets`, `batch_items`, `download_batches`)
  - scraped catalog facts: `1web_scraper_01/outputs/*.probe.json`, `2database_builder/outputs/databases/*.db`
  - local data presence: `data/training/`, `data/raw/`, `data/inference/`
- Model (training/inference/postprocess/eval)
  - training config + selected inputs: `5model_training/configuration/mito_openorganelle_foundation.yaml`
  - training outputs/checkpoints: `5model_training/outputs/.../checkpoints/`
  - inference/prediction artifacts: `data/inference/` and pipeline run sections
  - evaluation artifacts: `data/ml_runs/` (for example benchmark files)
- Downloaded & Predicted Data
  - use studio inspect/status endpoints for filtered counts:
    `GET /api/studio/data/inspect`, `GET /api/studio/datasets/status`, `GET /api/studio/download-batches`
  - "used in model" answers must reference either:
    1) training YAML membership, and/or
    2) `datasets.hidden_from_training` flag.

Plan and run-report interpretation
- Plan field meanings (for user clarification)
  - `sites`: sources/providers targeted by this run
  - `stages`: ordered work units (`scrape`, `database`, `download`, `training`)
  - `n_crops` / training/inference split: per-dataset crop allocation for download stage
  - sample/dataset include/exclude: filtering rules applied before download
- "What does this plan end up with?" should be answered as expected artifacts per stage and expected final report fields.
- Run-result diagnosis should include:
  - stage-level status (success/failed/skipped/cancelled),
  - first failing stage and immediate reason,
  - downstream consequence (what was skipped or incomplete),
  - recovery suggestion tied to the failing stage.
