from __future__ import annotations

from typing import Any

from master.deep.llm_strategy import plan_scrape_route
from master.generic_scraper import GenericWebsiteScraper


def _openorganelle_workspace(website_name: str, url: str) -> bool:
    w = (website_name or "").lower()
    u = (url or "").lower()
    return "openorganelle" in w or "openorganelle.janelia.org" in u


def get_scraper(website_name: str, url: str, *, llm: Any):
    """Choose ``catalog_pipeline`` (full capabilities) vs ``http_snapshot`` (fetch + markdown only).

    BossDB is routed explicitly — it uses its own metadata-API pipeline rather than Playwright/S3.
    """
    if website_name.lower().startswith("bossdb"):
        from websites.bossdb_01.pipeline import BossDBScraper

        print("[INFO] Scrape route (explicit): websites.bossdb_01.pipeline", flush=True)
        return BossDBScraper()

    mode = plan_scrape_route(llm, website_name=website_name, url=url)
    # `plan_scrape_route` may use one small model call when SCRAPE_LLM_ROUTE is enabled,
    # but the route label should not imply LLM-written report bodies (reports are heuristic).
    import os
    routed = os.getenv("SCRAPE_LLM_ROUTE", "1").strip().lower() not in ("0", "false", "no", "off")
    tag = "route" if routed else "route (default)"
    print(f"[INFO] Scrape {tag}: {mode}", flush=True)
    if mode == "http_snapshot":
        return GenericWebsiteScraper()
    if _openorganelle_workspace(website_name, url):
        from websites.openorganelle_01.pipeline import CatalogSitePipeline

        print("[INFO] Scrape route: websites.openorganelle_01.pipeline", flush=True)
        return CatalogSitePipeline()
    from master.deep.catalog_pipeline import CatalogSitePipeline

    return CatalogSitePipeline()
