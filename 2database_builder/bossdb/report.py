"""Markdown schema notes for BossDB pair catalogs."""

from __future__ import annotations

from pathlib import Path


def render_bossdb_pairs_schema_markdown(*, out_db: Path, n_pairs: int) -> str:
    return "\n".join(
        [
            "# BossDB Pair Catalog Schema",
            "",
            "This catalog is generated from `inventory_bossdb.jsonl` using paired",
            "image + mitochondria-seg channels in the same experiment.",
            "",
            f"- Output DB: `{out_db.name}`",
            f"- Pair-ready datasets: **{n_pairs}**",
            "- `dataset_resolved.ready_labeled = 1` for all rows",
            "- `resolved_em_path` / `resolved_mito_seg_path` are BossDB URIs",
            "  (`bossdb://collection/experiment/channel`).",
        ]
    )
