"""Append Brave Search snippets to scrape text (optional)."""

from __future__ import annotations

import os
from urllib.parse import urlparse


def augment_scrape_with_brave(
    scraped: str,
    website_name: str,
    url: str,
    access_signals: dict,
    brave_search_impl,
) -> str:
    """Run 1–3 Brave queries when API key is set and scrape is thin or missing XHR."""
    if not os.getenv("BRAVE_SEARCH_API_KEY"):
        return scraped
    if access_signals.get("playwright_scrape_blocked"):
        return scraped

    netloc = urlparse(url).netloc
    parts = [
        brave_search_impl(
            f"{website_name} download data bulk S3 AWS CLI API documentation "
            f"mitochondria organelle site:{netloc}"
        ),
    ]
    if "You need to enable JavaScript" in scraped or len(scraped.strip()) < 1800:
        parts.append(
            brave_search_impl(
                f"{website_name} dataset catalog segmentation volume EM site:{netloc}"
            )
        )
    if not access_signals.get("xhr_json_urls"):
        parts.append(
            brave_search_impl(
                f"{website_name} API documentation dataset list JSON "
                f"download zarr N5 BossDB S3 manifest github site:{netloc}"
            )
        )
    return (
        scraped
        + "\n\n=== Brave Search (download methods & discovery) ===\n"
        + "\n\n---\n\n".join(parts)
        + "\n"
    )
