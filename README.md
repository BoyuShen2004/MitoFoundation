# MitoFoundation

End-to-end pipeline for mitochondria EM data: scrape catalogs (OpenOrganelle, BossDB, MitoLE), download and preprocess volumes, train and run nnU-Net, then postprocess and evaluate.

## Repository layout

- `0inventory/`: Stage-0 inventory and download-history helpers.
- `1web_scraper_01/`: Stage-1 website scraping and probe generation.
- `2database_builder/`: Stage-2 schema/catalog DB builders.
- `3data_downloader/`: Stage-3/4 download and preprocessing (`downloader_common/` for nnUNet export helpers).
- `5model_training/`: nnU-Net training/inference Slurm scripts and postprocessing (nnU-Net itself is **not** vendored in git).
- `agent/orchestration/`: Shared orchestration, registry, and pipeline helpers.
- `agent/chat_web/`: FastAPI backend and Pipeline Studio APIs.
- `frontend/`: Web UI (Vite + React).
- `config/paths.py`: Central path resolution (`MITO2_PROJECT_ROOT`, `nnUNet_*` env vars).
- `data/`: Runtime datasets (mostly gitignored; see below).
- `scripts/`: Smoke tests and guardrails.

## Prerequisites

- Python 3.10+ and `pip install -r requirements.txt`
- Optional: [Playwright](1web_scraper_01/master/playwright_browsers.py) for deep scrapes
- **nnU-Net v2** (installed locally under `5model_training/nnUNet`; see next section)
- GPU node for training/inference (Slurm scripts under `5model_training/slurm/scripts/`)

## nnU-Net setup (required for training / inference)

The upstream [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) repository is **not** included in this git repo. Clone it into `5model_training/`:

```bash
export MITO2_PROJECT_ROOT="${MITO2_PROJECT_ROOT:-$(pwd)}"
cd "${MITO2_PROJECT_ROOT}/5model_training"

git clone https://github.com/MIC-DKFZ/nnUNet.git nnUNet
cd nnUNet
```

### Conda environment

Create and activate a dedicated environment (name is arbitrary; Slurm scripts default to `nnunetv2`):

```bash
conda create -n nnunetv2 python=3.10 -y
conda activate nnunetv2
pip install -e .
```

Install this repo’s Python deps in the same env if you run the full pipeline from one environment:

```bash
cd "${MITO2_PROJECT_ROOT}"
pip install -r requirements.txt
```

### nnU-Net path environment variables

Point nnU-Net at the project `data/` trees (defaults match `config/paths.py`):

```bash
export MITO2_PROJECT_ROOT="${MITO2_PROJECT_ROOT:-/projects/weilab/shenb/mitoFoundation2}"
export nnUNet_raw="${MITO2_PROJECT_ROOT}/data/nnUNet_raw"
export nnUNet_preprocessed="${MITO2_PROJECT_ROOT}/data/nnUNet_preprocessed"
export nnUNet_results="${MITO2_PROJECT_ROOT}/data/nnUNet_results"
```

Add these to your shell profile or job script. Dataset **001** in nnU-Net corresponds to `data/nnUNet_raw/Dataset001_mito2/`.

### Verify

```bash
conda activate nnunetv2
which nnUNetv2_train
ls "${MITO2_PROJECT_ROOT}/data/nnUNet_raw/Dataset001_mito2/dataset.json"
```

## Quick start (pipeline + UI)

```bash
export MITO2_PROJECT_ROOT="${MITO2_PROJECT_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "${MITO2_PROJECT_ROOT}"

pip install -r requirements.txt

# Optional: LLM settings for chat / scrape agents
cp agent/orchestration/config/llm.json.example agent/orchestration/config/llm.json
# Edit llm.json with your API keys (never commit this file).

python scripts/smoke_test_pipeline.py

# Web UI
./mito2
# Open http://127.0.0.1:8765 (see agent/chat_web/main.py for MITO2_HOST / MITO2_PORT)
```

## Slurm jobs (training / inference)

Example scripts (edit `#SBATCH` headers, `CONDA_SH`, and mail user for your site):

```bash
mkdir -p 5model_training/slurm/logs/train 5model_training/slurm/logs/infer
sbatch 5model_training/slurm/scripts/train_nnunet_mito_foundation.sl
sbatch 5model_training/slurm/scripts/infer_nnunet_mito_foundation.sl
```

Scripts honor `MITO2_PROJECT_ROOT`, `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`, and `NNUNET_CONDA_ENV`.

## What is not in git

See `.gitignore`. In particular:

- `5model_training/nnUNet/` — clone separately (above)
- `data/nnUNet_*`, `data/raw`, `data/outputs/`, `*.sqlite`
- `hpc_data_pipeline/**/outputs/`
- `agent/orchestration/config/llm.json` — copy from `llm.json.example`
- `frontend/node_modules/`, `frontend/dist/`

## Conventions

- Prefer imports from `agent.orchestration.registry.*` and `agent.orchestration.pipeline.*`.
- Keep stage-specific logic inside stage folders (`1*`, `2*`, `3*`, `5*`).
- Keep cross-stage shared logic under `agent/orchestration/`.
- Resolve paths via `config.paths` or `MITO2_PROJECT_ROOT` / `nnUNet_*` environment variables.

```bash
python scripts/check_architecture_guardrails.py
```
