"""LangChain tools + Brave helper used by ``agent.py``."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from langchain_core.tools import StructuredTool, tool
from pydantic import BaseModel, Field

from .config import BASE_DIR, OUTPUT_DIR, WEBSITES_FILE
from .fetch import fetch_page
from .s3_probe import probe_dataset_s3_paths, probe_result_to_download_config


_GENERATED_AT_ET_RE = re.compile(
    r"^\*\*Generated At \(US/Eastern\):\*\*.*\n?",
    flags=re.MULTILINE,
)


def _generated_at_us_eastern() -> str:
    now_et = datetime.now(ZoneInfo("America/New_York"))
    return now_et.strftime("%Y-%m-%d %H:%M:%S %Z%z")


def _inject_generated_at_header(content: str) -> str:
    """Add a deterministic generation timestamp header for markdown outputs."""
    cleaned = _GENERATED_AT_ET_RE.sub("", content).lstrip("\n")
    stamp = f"**Generated At (US/Eastern):** {_generated_at_us_eastern()}"
    return f"{stamp}\n\n{cleaned}"


def brave_search_impl(query: str) -> str:
    """Brave Search API (shared with CLI workflow)."""
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        return "BRAVE_SEARCH_API_KEY is not set in .env."
    try:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": api_key,
            },
            params={"q": query, "count": 5, "country": "US", "search_lang": "en"},
            timeout=25,
        )
        resp.raise_for_status()
        results = resp.json().get("web", {}).get("results", [])
        if not results:
            return f"No Brave results for: {query}"
        lines = [f"Top Brave results for: {query}"]
        for i, item in enumerate(results[:5], start=1):
            title = item.get("title", "Untitled")
            url = item.get("url", "")
            desc = (item.get("description") or "").replace("\n", " ")
            lines.append(f"{i}. {title}\n   URL: {url}\n   Snippet: {desc}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Brave search error: {exc}"


@tool
def read_websites_list() -> str:
    """Read website URL definitions.

    Preferred layout:
      - ``<site>/guides/websites.md`` for each website.

    Legacy layout (still supported):
      - ``web_scraper_01/guides/websites.md``.
    """
    # Legacy single file support.
    if WEBSITES_FILE.exists():
        return WEBSITES_FILE.read_text(encoding="utf-8")

    # New per-site guides layout.
    chunks: list[str] = []
    for d in sorted(BASE_DIR.iterdir()):
        if not d.is_dir():
            continue
        p = d / "guides" / "websites.md"
        if p.is_file():
            chunks.append(p.read_text(encoding="utf-8").strip())

    if not chunks:
        return (
            f"Error: no websites.md found. "
            f"Expected legacy {WEBSITES_FILE} or per-site '<site>/guides/websites.md'."
        )

    # Keep the original format: one `Name: https://...` per line.
    return "\n".join([c for c in chunks if c])


@tool
def scrape_website(url: str) -> str:
    """Fetch one URL; return cleaned text (max ~10k chars)."""
    page = fetch_page(url)
    if not page.get("ok"):
        return f"Error scraping {url}: {page.get('error', 'Unknown error')}"
    text = page.get("text", "")
    if len(text) > 10_000:
        text = text[:10_000] + "\n\n...[truncated]"
    return (
        f"URL: {url}\n"
        f"Page title: {page.get('title')}\n"
        f"HTTP Last-Modified header: {page.get('last_modified')}\n\n"
        f"{text}"
    )


class SaveMarkdownInput(BaseModel):
    filename: str = Field(description="Filename stem, e.g. mitomap_genome_db")
    content: str = Field(description="Full Markdown body")


def _save_markdown(filename: str, content: str) -> str:
    safe = re.sub(r"[^\w\-]", "_", filename.strip())
    path = OUTPUT_DIR / f"{safe}.md"
    path.write_text(_inject_generated_at_header(content), encoding="utf-8")
    return f"Saved → {path}"


def _save_markdown_to_path(path: Path, content: str) -> Path:
    """Write markdown to an explicit path (e.g. ``OpenOrganelle_02.md``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_inject_generated_at_header(content), encoding="utf-8")
    return path


save_markdown = StructuredTool.from_function(
    func=_save_markdown,
    name="save_markdown",
    description="Save Markdown under outputs/ (filename without extension).",
    args_schema=SaveMarkdownInput,
)


@tool
def list_saved_files() -> str:
    """List .md files in outputs/."""
    files = sorted(OUTPUT_DIR.glob("*.md"))
    if not files:
        return "No files saved yet in outputs/."
    return "Saved files:\n" + "\n".join(f"  - {f.name}" for f in files)


@tool
def brave_search(query: str) -> str:
    """Brave Search (needs BRAVE_SEARCH_API_KEY)."""
    return brave_search_impl(query)


class S3ProbeInput(BaseModel):
    dataset_id: str = Field(description="Dataset slug/ID to probe (e.g. 'aic_desmosome-3', 'jrc_mus-kidney')")
    source: str = Field(
        default="openorganelle",
        description="Dataset source/repository. Supported: 'openorganelle' (Janelia COSEM). Default: openorganelle.",
    )


def _probe_s3_paths(dataset_id: str, source: str = "openorganelle") -> str:
    """
    Probe S3 bucket (no credentials needed) to discover exact download paths
    for a biomedical EM dataset. Returns JSON with em_image_path, mito_seg_path,
    mito_pred_path, voxel_size_nm, and all available label names.
    """
    result = probe_dataset_s3_paths(dataset_id.strip(), source=source)
    cfg = probe_result_to_download_config(result)
    out = {
        "probe": result,
        "download_config": cfg,
    }
    return json.dumps(out, indent=2)


probe_s3_paths = StructuredTool.from_function(
    func=_probe_s3_paths,
    name="probe_s3_paths",
    description=(
        "Probe S3 bucket (anonymous, no credentials) to discover exact download paths "
        "for a biomedical EM dataset. Works for OpenOrganelle/Janelia COSEM datasets. "
        "Returns em_image_path, mito_seg_path, voxel_size_nm, and all label names. "
        "Use this after identifying a dataset_id from scraped metadata."
    ),
    args_schema=S3ProbeInput,
)


def get_agent_tools():
    """Tools for ``create_agent`` (adds Brave + S3 probe tools)."""
    tools = [
        read_websites_list,
        scrape_website,
        save_markdown,
        list_saved_files,
        probe_s3_paths,
    ]
    if os.getenv("BRAVE_SEARCH_API_KEY"):
        tools.append(brave_search)
        print("[INFO] Brave search tool enabled.")
    print("[INFO] S3 path probe tool enabled (anonymous).")
    return tools
