"""Parse ``guides/websites.md`` and decide when to use Playwright for a URL."""

from __future__ import annotations

import os
from typing import List
from urllib.parse import urlparse

from .config import WEBSITES_FILE


def parse_websites(raw_text: str) -> List[tuple[str, str]]:
    """Extract (website_name, url) entries from ``guides/websites.md``."""
    websites: List[tuple[str, str]] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if ":" in line and "http" in line:
            name, maybe_url = line.split(":", 1)
            url = maybe_url.strip()
            if url.startswith(("http://", "https://")):
                websites.append((name.strip(), url))
                continue

        if line.startswith(("http://", "https://")):
            domain = line.split("//", 1)[-1].split("/", 1)[0]
            inferred_name = domain.split(".")[0]
            websites.append((inferred_name, line))

    return websites


def spa_hosts_from_websites_md() -> frozenset[str]:
    """Hostnames for every URL in websites definitions.

    Preferred:
      - ``<site>/guides/websites.md`` per website.
    Legacy:
      - ``web_scraper_01/guides/websites.md``.
    """
    raw = ""
    if WEBSITES_FILE.exists():
        raw = WEBSITES_FILE.read_text(encoding="utf-8")
    else:
        from .config import BASE_DIR
        chunks: list[str] = []
        for d in sorted(BASE_DIR.iterdir()):
            if not d.is_dir():
                continue
            p = d / "guides" / "websites.md"
            if p.is_file():
                chunks.append(p.read_text(encoding="utf-8").strip())
        raw = "\n".join([c for c in chunks if c])

    entries = parse_websites(raw)
    hosts: list[str] = []
    for _name, url in entries:
        h = urlparse(url).netloc.lower().split(":")[0]
        if h:
            hosts.append(h)
    return frozenset(hosts)


def should_use_playwright(
    url: str, spa_hosts: frozenset[str] | None = None
) -> bool:
    """Render with Chromium when static HTML is insufficient (SPAs)."""
    flag = os.getenv("USE_PLAYWRIGHT", "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return False
    host = urlparse(url).netloc.lower().split(":")[0]
    configured = spa_hosts if spa_hosts is not None else spa_hosts_from_websites_md()
    if host in configured:
        return True
    for part in os.getenv("PLAYWRIGHT_EXTRA_HOSTS", "").split(","):
        p = part.strip().lower()
        if p and (host == p or host.endswith("." + p)):
            return True
    return False
