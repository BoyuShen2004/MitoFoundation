---
id: dataset_download_truth
title: Dataset download status (registry + filesystem)
label: Download status
---

Purpose
- Answer what datasets are downloaded, where they are, and their processing state using local filesystem evidence.

Primary evidence sources
  - Registry (provenance, status):   data/registry.sqlite
      batch_items.status: pending | present | missing_or_deleted_local | failed
      batch_items.local_path: absolute path to the local file
      download_batches.batch_id and profile_json: download parameters used
  - File system (content, format):   data/training/images/*_im.h5
                                      data/training/labels/*_seg.h5
  - Generated scripts:               3data_downloader/outputs/download_*.py
      DATASETS dict contains: dataset_name, img_path (remote), seg_path (remote)
  - Stage-3 download metadata:       3data_downloader/openorganelle/download_openorganelle_labeled.md
  - Studio API inspect evidence:     GET /api/studio/data/inspect (raw, training, inference)
                                      GET /api/studio/datasets/status (registry provenance)

Canonical output paths (from openorganelle/agent.py constants)
  Training-ready (stage 3 + 4):
    data/training/images/<dataset_tag>_vol<N>_im.h5   (EM crop, lzf-compressed)
    data/training/labels/<dataset_tag>_vol<N>_seg.h5  (mito seg, lzf-compressed, binary 0/1)
  Legacy raw runs (data/raw/<run_folder>/images/*.h5):
    older batch downloads before TRAINING_ROOT was the canonical target
  Stub mode: data/raw_volumes/download_manifest.json (not real volumes)

Behavior
- Summarize dataset status by registry batch_items.status first, filesystem evidence second.
- For registry answers, name the table, column, and value used.
- Provide file/dir counts and dominant file extensions when available.
- For volume format questions: H5 files use lzf compression; spacing 16nm iso (Z,Y,X); shape up to 512^3.
- Mark uncertain claims as "inferred from naming" unless confirmed by registry or file content.

Guardrails
- Never claim a dataset is downloaded without registry batch_item status=present OR confirmed file existence.
- If data scan is truncated or partial, state that limitation clearly.
- Prefer registry.sqlite over filesystem glob for "was X downloaded?" (URL churn safe).
- If registry.sqlite is absent: "Not found in current artifacts: data/registry.sqlite".
- The hidden_from_training flag (datasets.hidden_from_training=1) means file exists but excluded from training.
- See also: `pipeline_stage_map` skill for the complete stage→output path map.

Downloaded vs used-in-model interpretation
- "Downloaded" is provenance/storage status (registry + file existence), not automatic training inclusion.
- "Use in model?" must be answered from:
  1) `datasets.hidden_from_training` (eligibility gate), and
  2) training YAML membership (actual configured training input list).
- If downloaded-but-not-used exists, report the count and why (hidden flag, missing pair, or not listed in YAML).

Filter count expectations
- For grouped count questions (training/inference/postprocessed/predicted/instance/border-contour), return:
  - exact count when an explicit artifact/table/field exists,
  - otherwise a clearly labeled proxy count with source.
- Do not collapse distinct scopes:
  - source-wide catalog total,
  - currently downloaded present subset,
  - model-configured subset.
