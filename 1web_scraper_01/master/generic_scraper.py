from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

from master.deep.brave_augment import augment_scrape_with_brave
from master.deep.fetch import scrape_site_deep
from master.deep.llm_report import generate_markdown_for_site
from master.deep.tools import _save_markdown, brave_search_impl


class GenericWebsiteScraper:
    """Fallback scraper: deep scrape + heuristic markdown (no S3 probing)."""

    def scrape_and_report(
        self,
        *,
        llm,
        website_name: str,
        url: str,
        scraped_at_iso: str,
        spa_hosts: frozenset[str],
        max_pages: int,
        output_dir: Path,
    ) -> Dict[str, dict]:
        scraped, access_signals = scrape_site_deep(
            url, website_name, max_pages=max_pages, spa_hosts=spa_hosts
        )
        scraped = augment_scrape_with_brave(
            scraped, website_name, url, access_signals, brave_search_impl
        )
        markdown = generate_markdown_for_site(
            llm, website_name, url, scraped, scraped_at_iso, access_signals
        )
        _save_markdown(filename=website_name, content=markdown)
        print(f"[INFO] Saved {website_name}.md at {datetime.now().isoformat(timespec='seconds')}")
        return {}

