---
display_name: "BossDB"
url: "https://api.metadata.bossdb.org"
slug: "bossdb_01"
updated_at: "2026-04-30T23:01:45-04:00"
---

BossDB (Boss volumetric database) metadata catalog via REST API.

**Deep scrape:** Pipeline Studio runs `python -m master.agent --sites BossDB` with cwd `1web_scraper_01/`. Uses the BossDB metadata REST API directly (no Playwright). Code: `master/deep/bossdb_pipeline.py`; shared scrape/schema goals live in `1web_scraper_01/master/deep/guides/scrape_and_schema_goals.md` (optional per-site overlay: `websites/<slug>/guides/scrape_and_schema_goals.md`).

**Outputs in this repo:** `1web_scraper_01/outputs/BossDB.probe.json` and `BossDB.md` (canonical mirror after each successful run).

## Data to look for

Collection/experiment/channel triples, channel type (image vs annotation), data type (uint8/uint64), voxel sizes, organism and modality from tags, labeling inference (mitochondria, nucleus, er, etc.), and resolvable `bossdb://collection/experiment/channel` URIs. Pipeline-oriented notes are in `master/deep/guides/` and this folder’s optional `guides/`.
