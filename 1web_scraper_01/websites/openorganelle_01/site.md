---
display_name: "OpenOrganelle"
url: "https://openorganelle.janelia.org/"
slug: "openorganelle_01"
updated_at: "2026-04-30T22:58:10-04:00"
---

OpenOrganelle (COSEM) dataset portal at Janelia.

**Deep scrape:** Pipeline Studio runs ``python -m master.agent --sites OpenOrganelle`` with cwd ``1web_scraper_01/``. **Defaults (fast, rule-based):** one headless Chromium pass on the SPA shell to capture Supabase headers + paginated PostgREST ``dataset`` rows (appendix ``dataset_name=`` lines for ``2database_builder``); report markdown is **heuristic-only** (no LLM report body); **no** per-dataset Playwright UI tours unless ``PLAYWRIGHT_UI_SEGMENTATION_LAYERS=1``; minimal same-site page follows (override with ``OPENORGANELLE_FAST_SCRAPE=0`` for the older heavy crawl). Code: ``master/deep/catalog_pipeline.py`` + ``master/deep/``; guides: ``master/deep/guides/`` and ``guides/``.

**Outputs in this repo:** ``1web_scraper_01/outputs/OpenOrganelle.probe.json`` and ``OpenOrganelle.md`` (canonical mirror after each successful run).

## Data to look for

PostgREST/API evidence, S3 n5/zarr roots, EM paths and scales, `image:` / `mesh:` layers, mito_seg / mito_pred (GT vs prediction), grid metadata, and resolvable download URLs. See ``master/deep/guides/`` (and this folder’s ``guides/``) for pipeline-oriented notes.
