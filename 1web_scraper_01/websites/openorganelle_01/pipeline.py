"""Site-specific scrape pipeline for OpenOrganelle-style catalog sites.

``CatalogSitePipeline`` is shared with other HTTP catalog sites; this module
is the supported import path for symmetry with ``websites.bossdb_01.pipeline``.
"""

from __future__ import annotations

from master.deep.catalog_pipeline import CatalogSitePipeline

__all__ = ["CatalogSitePipeline"]
