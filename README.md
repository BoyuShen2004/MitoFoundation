# mitoFoundation2

Mitochondria data pipeline project spanning discovery, schema/catalog building, downloading + preprocessing, and model training.

## Repository Layout

- `0inventory/`: Stage-0 inventory and download-history helpers.
- `1web_scraper_01/`: Stage-1 website scraping and probe generation.
- `2database_builder/`: Stage-2 schema/catalog DB builders (includes `openorganelle/labeled_inventory_resolved.py` for resolved-path inventory used by Stage-3 + the registry).
- `3data_downloader/`: Stage-3/4 download and preprocessing; `3data_downloader/downloader_common/` holds nnUNet export / split / contour helpers (`downloader_common.*` after putting `3data_downloader` on `sys.path`).
- `5model_training/`: Model training stack (PyTorch Connectomics).
- `agent/orchestration/`: Shared orchestration modules, including:
  - `agent/orchestration/registry/`: Central registry package.
  - `agent/orchestration/pipeline/`: Shared pipeline helpers.
  - `agent/orchestration/session_pipeline.py`: App session/pipeline state types.
  - `agent/orchestration/skills/`: Orchestration skill markdown only (`<slug>/skill.md`). Code: `skill_store.py` / `skill_api.py`.
- `agent/chat_web/`: Backend APIs for web workflows; chat skill markdown under `agent/chat_web/skills/<slug>/skill.md` (see `agent/chat_web/SKILLS.md`).
- `frontend/`: UI code.
- `scripts/`: Operational scripts (for example smoke tests).
- `data/`: Runtime datasets and generated training assets.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run smoke test:

```bash
python scripts/smoke_test_pipeline.py
```

## Conventions

- Prefer imports from `agent.orchestration.registry.*` and `agent.orchestration.pipeline.*`.
- Keep stage-specific logic inside stage folders (`1*`, `2*`, `3*`, `5*`).
- Keep cross-stage shared logic under `agent/orchestration/`.
- Keep inventory-specific logic under `0inventory/`.

See `docs/PROJECT_STRUCTURE.md` for organization rules.
For canonical ownership/import rules and migration policy, see:

- `docs/architecture/repo-layout.md`
- `docs/architecture/migration-map.md`
- `docs/architecture/duplication-audit.md`

Run architecture guardrails:

```bash
python scripts/check_architecture_guardrails.py
```
