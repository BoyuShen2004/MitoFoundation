{{scrape_goals}}

Generate a markdown report with this exact structure and headings:

# <Website Title> — Mitochondria Key Information

**Source:** {{url}}
**Scraped:** {{today}}
**Scraped At (ISO):** {{scraped_at_iso}}
**Website Name:** {{website_name}}
**Mitochondria Relevance:** <High/Medium/Low with one-line justification>

## Data Metadata (Required)
- **Last Upload / Last Modification:** <date or "Not explicitly stated">
- **Data Volume:** <counts, scale, dataset size, or "Not explicitly stated">
- **Organelles in Data:** <list; must mention mitochondria when present>
- **Labeling Status:** <Labeled / Unlabeled / Mixed-Intermediate / Not explicitly stated>
- **Label Evidence:** <how labeling status was inferred>

## Segmentation & annotations (for downstream schema)
Use appendix rows as source of truth for per-dataset layers and URLs.
For “good mitochondria labels”, use only **`mito_label_quality=good`** rows (non-prediction per appendix rules). Exclude prediction/inference/unproofread layers.
Tag each usable mitochondria label as **`instance`** or **`semantic`** (from `mito_segmentation_kind` / naming). Report counts: total good mito datasets, and instance vs semantic (overlap allowed).

## Mitochondria-Specific Summary
<1–3 sentences>

## How to obtain mitochondria-related data (PRIMARY — engineer handoff)
For OpenOrganelle specifically, include:
- exact `download_*` URL fields from appendix
- `*_leaf_url` and `*_scale_key` if present
- note whether path is GT mask vs prediction
- prefer image paths that align with the chosen mask path (same store root / scale key) to avoid compressed-vs-uncompressed shape mismatch.

## Key Findings
## Databases & Datasets
At most **15** example slugs **or** one sentence pointing to the Appendix. **Never** enumerate the full catalog (no long `* \`slug\`` lists). Omit `segmentation_challenge`.

## Quality, Gaps, and Future Work
## Additional Notes

Rules: do not invent buckets/APIs; return only markdown.

Scraped content:
{{scraped_text}}

