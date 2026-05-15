"""Render schema markdown documentation for a generated database."""

from __future__ import annotations

from pathlib import Path


def _resolved_section(stats: dict) -> str:
    if stats.get("resolved_rows") is None:
        return ""
    by_src = stats.get("by_source") or {}
    lines = [f"| `{k}` | {v} |" for k, v in sorted(by_src.items())]
    tbl = "\n".join(lines) if lines else "| — | — |"
    return f"""

## Downstream table: `dataset_resolved`

Populated after the S3 probe by ``materialize_resolved_paths()`` (same rules as
``download_inventory.resolve_paths``): merges **probe**, **scraped PostgREST**,
**layer URLs**, and **template** fallbacks into one row per dataset.

| Metric | Value |
|--------|------:|
| Datasets (rows) | **{stats.get("resolved_rows", 0)}** |
| Ready for **labeled** download (EM + mito seg path) | **{stats.get("ready_labeled", 0)}** |
| Ready for **EM-only** / unlabeled (has EM path) | **{stats.get("ready_em", 0)}** |
| Rows missing EM or seg (not fully paired) | **{stats.get("resolved_incomplete", 0)}** |

### `dataset_resolved` columns

| Column | Meaning |
|--------|---------|
| `resolved_em_path` | Canonical EM / raw volume S3 URL |
| `resolved_mito_seg_path` | Mito (or best seg) path for labeled workflows |
| `resolved_voxel_nm` | JSON array of spacing (nm) |
| `path_source` | `s3_probe` / `scraped_postgrest` / `layer_url` / `template_fallback` / `incomplete` |
| `ready_labeled` | 1 if EM + non-prediction mito seg path present (`download_mito_mask_quality='good'`) |
| `ready_em_only` | 1 if an EM path exists |

### `path_source` counts

| path_source | Count |
|-------------|------:|
{tbl}

```sql
-- Labeled download candidates (good non-prediction mito masks only)
SELECT dataset_name, path_source, resolved_em_path, resolved_mito_seg_path
FROM dataset_resolved
WHERE ready_labeled = 1
ORDER BY dataset_name;

-- EM-only (unlabeled) candidates
SELECT dataset_name, path_source, resolved_em_path
FROM dataset_resolved
WHERE ready_em_only = 1 AND ready_labeled = 0
ORDER BY dataset_name;
```
"""


def render_schema_markdown(
    source_md: Path,
    sqlite_db: Path,
    stats: dict,
    *,
    merged_sources: list[Path] | None = None,
) -> str:
    extra = _resolved_section(stats)
    if merged_sources and len(merged_sources) > 1:
        src_lines = "\n".join(f"- `{p}`" for p in sorted(merged_sources))
        src_block = f"**Merged source markdowns ({len(merged_sources)} files):**\n\n{src_lines}\n\n"
    else:
        src_block = f"**Source markdown:** `{source_md}`\n\n"
    return f"""# Schema Summary: {source_md.stem}

{src_block}**Database file:** `{sqlite_db}`

## What This Database Contains

- One row per dataset in `datasets`
- One row per parsed layer token in `dataset_layers`
- Source/provenance URL(s) in `dataset_sources`
- After build + S3 probe: one row per dataset in `dataset_resolved` (canonical download paths)

## Row Counts

- `datasets`: **{stats.get("datasets", 0)}**
- `dataset_layers`: **{stats.get("layers", 0)}**
- `dataset_sources`: **{stats.get("source_links", 0)}**
- `datasets` with good (non-prediction) mito segmentation labels: **{stats.get("mito_datasets", 0)}**

## Table Definitions

### `datasets`

| Column | Type | Meaning |
|---|---|---|
| `id` | INTEGER PK | Surrogate id |
| `dataset_name` | TEXT UNIQUE | Dataset slug/name from markdown |
| `stage` | TEXT | Stage metadata (prod/dev etc.) |
| `segmentation_challenge` | TEXT | Raw string value from markdown |
| `description` | TEXT | Dataset description |
| `sample_organism` | TEXT | Parsed sample organism |
| `bio_target` | TEXT | Normalized biological target (e.g. kidney, liver, desmosome) |
| `bio_target_type` | TEXT | Target class (`organ`, `tissue`, `cell_type`, `subcellular_structure`, `unknown`) |
| `bio_target_source` | TEXT | Field that provided the strongest match (`dataset_name`, `sample_subtype`, etc.) |
| `bio_target_confidence` | REAL | Rule confidence score (0-1) |
| `sample_type` | TEXT | Parsed sample type |
| `sample_subtype` | TEXT | Parsed sample subtype |
| `sample_name` | TEXT | Parsed sample name |
| `grid_spacing` | TEXT | Raw grid spacing list |
| `grid_dimensions` | TEXT | Raw grid dimensions list |
| `grid_axes` | TEXT | Raw axis order |
| `grid_spacing_unit` | TEXT | Spacing unit |
| `grid_dimensions_unit` | TEXT | Dimensions unit |
| `mitochondria_in_layer_names` | INTEGER | 1/0 flag from markdown |
| `download_volume_url` | TEXT | Scraped default EM/raw volume URL (e.g. PostgREST ``image.url``) |
| `download_volume_image_stack` | TEXT | Optional stack id / subpath hint |
| `download_mito_mask_url` | TEXT | Scraped mitochondria mask URL (ground-truth layer) |
| `download_mito_prediction_url` | TEXT | Optional prediction mask URL |
| `s3_probe_*` | TEXT | Filled by live S3 listing (see `dataset_resolved` for merged view) |

### `dataset_layers`

| Column | Type | Meaning |
|---|---|---|
| `id` | INTEGER PK | Layer row id |
| `dataset_id` | INTEGER FK | References `datasets.id` |
| `layer_role` | TEXT | `ground_truths` / `predictions` / `raw_or_other_layers` |
| `layer_kind` | TEXT | `image` or `mesh` |
| `layer_name` | TEXT | Layer token name (e.g. `mito_seg`) |
| `layer_format` | TEXT | Parsed `format=` value |
| `layer_stage` | TEXT | Parsed `stage=` value |
| `layer_url` | TEXT | Parsed `[url=…]` when present (direct array source) |
| `layer_content_type` | TEXT | Parsed `[content_type=…]` (e.g. segmentation vs raw) |
| `semantic` | INTEGER | 1 when `[semantic]` flag appears |
| `raw_token` | TEXT | Original parsed token |

### `dataset_sources`

| Column | Type | Meaning |
|---|---|---|
| `dataset_id` | INTEGER FK | References `datasets.id` |
| `source_url` | TEXT | Provenance URL parsed from line suffix |

## Query Examples

```sql
-- Datasets with good (non-prediction) mito segmentation labels — training-ready set
SELECT dataset_name
FROM datasets
WHERE download_mito_mask_quality = 'good'
ORDER BY dataset_name;
```

```sql
-- Per-dataset segmentation layer mapping
SELECT d.dataset_name, l.layer_role, l.layer_kind, l.layer_name
FROM dataset_layers l
JOIN datasets d ON d.id = l.dataset_id
ORDER BY d.dataset_name, l.layer_role, l.layer_kind, l.layer_name;
```

```sql
-- Raw scraped columns only (before merged resolution)
SELECT dataset_name, download_volume_url, download_mito_mask_url
FROM datasets
WHERE download_volume_url IS NOT NULL AND download_mito_mask_url IS NOT NULL
ORDER BY dataset_name;
```
{extra}"""
