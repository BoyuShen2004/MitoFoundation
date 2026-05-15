# Per-website scrape workspaces

Each subfolder under `1web_scraper_01/websites/` holds **one target site** for the scraper.

**Generic** instructions for agents live in **`1web_scraper_01/guides/`** (workflow, report template, output paths). Default catalog-pipeline prompts live under **`1web_scraper_01/master/deep/guides/`**; override per site in **`websites/<slug>/guides/`**.

| File | Purpose |
|------|---------|
| `site.md` | YAML front matter (`display_name`, `url`, `slug`, `updated_at`) plus profile and **Data to look for**. |
| `guides/*.md` | (Optional) Per-site markdown overlays consumed by `master/deep/guides_loader.py` (e.g. `scrape_and_schema_goals.md`). |

Structured probes are written under **`1web_scraper_01/outputs/`** (e.g. `<slug>.probe.json`). **Do not** add `datasets.json` under a workspace — tooling deletes it if present. **OpenOrganelle (and similar catalogs)** use the **`master.deep`** pipeline; canonical outputs are **`outputs/OpenOrganelle.probe.json`** and **`outputs/OpenOrganelle.md`** (no per-workspace slug copies).

**Folder names** are versioned: `{base}_01`, `{base}_02`, … where `base` is lowercase with **no spaces** (from the display name or the first hostname label). Legacy unversioned folders are renamed to `{base}_01` on first access. Optional `--slug` / Studio fields can target an existing slug or start a new batch.

If a URL has no scheme, tools default to **`http://`** (you can still type `https://` explicitly).

Create or update folders from **Pipeline Studio → Web scraper** or:

```bash
cd 1web_scraper_01
python agent.py 'example.org' --workspace --name 'Example Org' \
  --description 'Demo portal.' \
  --data-focus 'Any public datasets, APIs, or download pages.'
```
