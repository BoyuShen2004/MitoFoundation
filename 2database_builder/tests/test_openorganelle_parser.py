from __future__ import annotations

import sys
from pathlib import Path

# Match other tests: repo root (for ``0inventory``) + ``2database_builder/`` on sys.path.
_SCHEMA_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (_REPO_ROOT, _SCHEMA_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from openorganelle.parser import parse_markdown_text


def test_parse_openorganelle_appendix_dash_and_machine_row_shapes() -> None:
    """Rule-based scrapes still emit appendix rows; machine block omits the list bullet."""
    md = """
# OpenOrganelle — Mitochondria Key Information

**Source:** https://openorganelle.janelia.org/datasets

## Appendix: Dataset catalog (per-dataset → schema)

### All datasets on the site (catalog slug list)

1. **`slug-a`**
2. **`slug-b`**

### Primary datasets (spatial + segmentation / annotation layers)

- dataset_name=slug-a | stage=prod | grid.grid_spacing=[4.0, 4.0, 3.24] | grid.grid_dimensions=[10, 20, 30] | ground_truths=image:em[format=zarr][url=s3://bucket/em]

### Canonical dataset rows (machine, v1)

```text
BEGIN_DATASET_ROWS_V1
dataset_name=slug-b | stage=dev | predictions=image:mito_pred[format=n5]
END_DATASET_ROWS_V1
```
"""
    rows = parse_markdown_text(md)
    by = {r.dataset_name: r for r in rows}
    assert set(by) == {"slug-a", "slug-b"}
    assert by["slug-a"].stage == "prod"
    assert by["slug-b"].stage == "dev"
    assert any(l.layer_name == "em" for l in by["slug-a"].layers)
    assert any(l.layer_name == "mito_pred" for l in by["slug-b"].layers)
