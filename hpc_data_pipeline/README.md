# HPC Data Pipeline

This folder stores implementation files for the new **Existing Data From HPC Cluster** overhaul.

Current modules:

- `mitole_pipeline.py` — MitoLE root config, selected-subfolder persistence, path validation, and folder scan helpers.

Used by:

- `agent/chat_web/app/studio_api.py` endpoints:
  - `/api/studio/mitole/config`
  - `/api/studio/mitole/inspect`
  - `/api/studio/mitole/catalogue`
