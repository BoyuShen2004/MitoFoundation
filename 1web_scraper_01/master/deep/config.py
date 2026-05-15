"""Shared paths, HTTP defaults, and Playwright install helper."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # ``1web_scraper_01``
DEEP_GUIDES_DIR = Path(__file__).resolve().parent / "guides"
# Generic agent prompts: ``1web_scraper_01/guides/`` (see ``guides/README.md``).
GUIDES_DIR = BASE_DIR / "guides"
# SPA host list: prefer bundled pipeline ``master/deep/guides/websites.md``, else top-level guides.
_wf_deep = DEEP_GUIDES_DIR / "websites.md"
_wf_top = GUIDES_DIR / "websites.md"
WEBSITES_FILE = _wf_deep if _wf_deep.is_file() else _wf_top
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TODAY = date.today().isoformat()
SCRAPED_AT = datetime.now().isoformat(timespec="seconds")

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

DEEP_SCRAPE_KEYWORDS = [
    "mito",
    "mitochond",
    "dataset",
    "data",
    "annotation",
    "label",
    "segmentation",
    "release",
    "about",
    "docs",
    "help",
    "faq",
    "download",
    "api",
    "publication",
    "export",
    "rest",
    "graphql",
    "oauth",
    "token",
    "sdk",
    "catalog",
    "browse",
    "explore",
    "search",
    "volume",
    "zarr",
    "n5",
    "boss",
    "collection",
    "experiment",
    "tomogram",
    "fib-sem",
    "em",
]

RAW_HTML_SAMPLE = 200_000


def _env_truthy(name: str, default: str = "1") -> bool:
    val = os.getenv(name, default).strip().lower()
    return val in ("1", "true", "yes", "on")


def ensure_playwright_chromium_installed() -> tuple[bool, str]:
    """Auto-run `python -m playwright install chromium` if binaries are missing."""
    cmd = [sys.executable, "-m", "playwright", "install", "chromium"]
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            return True, ""
        tail = (res.stdout or "").strip()
        if len(tail) > 2000:
            tail = tail[-2000:]
        return False, tail or f"command failed with code {res.returncode}"
    except Exception as e:
        return False, str(e)
