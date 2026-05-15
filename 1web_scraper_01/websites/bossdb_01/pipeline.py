"""Site-specific scrape pipeline for BossDB (metadata API).

Implementation lives under ``master.deep``; this module is the supported
import path for symmetry with ``websites.openorganelle_01.pipeline``.
"""

from __future__ import annotations

from master.deep.bossdb_pipeline import BossDBScraper

__all__ = ["BossDBScraper"]
