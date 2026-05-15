"""BossDB-specific catalog builders.

Creates a pair-focused catalog DB for Studio browsing:
``BossDB_pairs.db`` contains only experiments that have both EM image and
mitochondria-seg channels (one row per paired experiment).

Layout mirrors ``openorganelle/`` (parser, builder, report, db).
"""

from __future__ import annotations

from .builder import build_bossdb_pairs_catalog
from .parser import default_inventory_path, default_probe_path, load_bossdb_pairs, repo_root

__all__ = [
    "build_bossdb_pairs_catalog",
    "default_inventory_path",
    "default_probe_path",
    "load_bossdb_pairs",
    "repo_root",
]
