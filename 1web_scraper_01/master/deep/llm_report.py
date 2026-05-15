"""Heuristic markdown reports + data-acquisition appendix for catalog / generic scrapes.

Each :func:`generate_markdown_for_site` / :func:`generate_incremental_markdown_for_site` call
returns **one** markdown string; the catalog pipeline saves it **once** per scrape invocation.
There is **no** LLM-written report body — the appendix (PostgREST / XHR signals) carries
``dataset_name=`` rows for ``2database_builder``.
"""

from __future__ import annotations

import os
import re
from typing import Any

from .config import TODAY
from .signals import (
    format_data_acquisition_playbook_md,
    heuristic_extract,
    merge_site_access_signals,
)

_DBS_SECTION_HEADING_RE = re.compile(
    r"(?m)^##\s+Databases\s*&\s*Datasets[^\n]*\s*$",
    re.IGNORECASE,
)


def _sanitize_bloated_databases_section(
    markdown: str,
    *,
    max_non_empty_lines: int = 28,
) -> str:
    """Cap an oversized ``## Databases & Datasets`` section (slug dumps redundant with appendix)."""
    m = _DBS_SECTION_HEADING_RE.search(markdown)
    if not m:
        return markdown
    head_end = m.end()
    tail = markdown[head_end:]
    next_sec = re.search(r"(?m)^##\s+\S", tail)
    if next_sec:
        section_body = tail[: next_sec.start()]
        after = tail[next_sec.start() :]
    else:
        section_body = tail
        after = ""
    nonempty = [ln for ln in section_body.splitlines() if ln.strip()]
    if len(nonempty) <= max_non_empty_lines:
        return markdown
    stub = (
        f"{m.group(0).rstrip()}\n\n"
        "_Omitted a long enumerated slug list here — see **Appendix → All datasets on the site "
        "(catalog slug list)** for the full index. "
        f"(Auto-trimmed **{len(nonempty)}** redundant lines.)_\n\n"
    )
    return markdown[: m.start()] + stub + after


def _report_char_budget() -> int:
    return int(
        os.getenv("SCRAPE_REPORT_CHAR_BUDGET")
        or os.getenv("LLM_SCRAPE_CHAR_BUDGET", "40000")
    )


def _report_char_budget_incremental() -> int:
    raw = os.getenv("SCRAPE_REPORT_CHAR_BUDGET_INCREMENTAL")
    if raw is not None and str(raw).strip():
        try:
            return max(4000, int(raw))
        except ValueError:
            pass
    base = _report_char_budget()
    return min(base, 80_000)


_DATABASE_BUILDER_HEADING = "## Database builder — new datasets to add"


def _format_database_builder_new_datasets_md(new_dataset_ids: list[str], *, per_line: int = 5) -> str:
    """Compact slug list for `2database_builder` handoff (incremental runs only)."""
    ids = sorted({x.strip() for x in new_dataset_ids if x and str(x).strip()}, key=str.lower)
    if not ids:
        return ""
    lines: list[str] = []
    for i in range(0, len(ids), max(1, per_line)):
        chunk = ids[i : i + per_line]
        lines.append(", ".join(f"`{x}`" for x in chunk))
    body = "\n".join(lines)
    return (
        f"{_DATABASE_BUILDER_HEADING}\n\n"
        "Run **`mitoFoundation2/2database_builder`** on the scraped markdown folder so these "
        "catalog slugs are merged into SQLite + catalog docs. "
        "(Same slugs appear under `OpenOrganelle.probe.json` → `scrape_batches` → "
        "`dataset_ids_added` for the matching batch.)\n\n"
        f"{body}\n"
    )


def _finalize_full_markdown_with_appendix(
    website_name: str,
    url: str,
    scraped_at_iso: str,
    access_signals: dict,
    markdown: str,
    *,
    fallback_heuristic: dict | None,
    database_builder_new_ids: list[str] | None = None,
) -> str:
    for _heading in ("## Disease Associations", "## Genetics & Genomics"):
        markdown = re.sub(
            re.escape(_heading) + r"\n.*?(?=\n## |\Z)",
            "",
            markdown,
            flags=re.DOTALL,
        )

    markdown = re.sub(r"^.*Scraped At \(ISO\):.*\n?", "", markdown, flags=re.MULTILINE)
    markdown = re.sub(
        r"(\*\*Scraped:\*\*\s*.*\n)",
        rf"\1**Scraped At (ISO):** {scraped_at_iso}\n",
        markdown,
        count=1,
    )

    markdown = _sanitize_bloated_databases_section(markdown)

    if database_builder_new_ids and _DATABASE_BUILDER_HEADING not in markdown:
        markdown = (
            markdown.rstrip()
            + "\n\n"
            + _format_database_builder_new_datasets_md(database_builder_new_ids)
        )

    appendix = format_data_acquisition_playbook_md(
        website_name, url, access_signals, heuristic=fallback_heuristic
    )
    if appendix.strip() not in markdown:
        markdown = markdown.rstrip() + "\n\n" + appendix

    return markdown


def _finalize_incremental_markdown(
    website_name: str,
    url: str,
    scraped_at_iso: str,
    access_signals: dict,
    markdown: str,
    *,
    new_dataset_ids: list[str] | None = None,
) -> str:
    markdown = re.sub(r"^.*Scraped At \(ISO\):.*\n?", "", markdown, flags=re.MULTILINE)
    markdown = re.sub(
        r"(\*\*Scraped:\*\*\s*.*\n)",
        rf"\1**Scraped At (ISO):** {scraped_at_iso}\n",
        markdown,
        count=1,
    )

    out = _sanitize_bloated_databases_section(markdown.rstrip())
    if new_dataset_ids and _DATABASE_BUILDER_HEADING not in out:
        out = out + "\n\n" + _format_database_builder_new_datasets_md(new_dataset_ids)

    appendix = format_data_acquisition_playbook_md(
        website_name, url, access_signals, heuristic=None
    )
    if appendix.strip() not in out:
        out = out + "\n\n" + appendix

    return out


def _markdown_full_heuristic_only(
    website_name: str,
    url: str,
    scraped_text: str,
    scraped_at_iso: str,
    access_signals: dict,
    *,
    database_builder_new_ids: list[str] | None = None,
) -> str:
    h = heuristic_extract(scraped_text)
    key_findings = (
        "\n".join(f"- {x}" for x in h["key_lines"]) or "- Not explicitly stated"
    )
    markdown = f"""# {website_name} — Mitochondria Key Information

**Source:** {url}
**Scraped:** {TODAY}
**Scraped At (ISO):** {scraped_at_iso}
**Website Name:** {website_name}
**Mitochondria Relevance:** Not explicitly stated (heuristic extraction from scrape text)

## Data Metadata (Required)
- **Last Upload / Last Modification:** {h["last_mod"]}
- **Data Volume:** {h["data_volume"]}
- **Organelles in Data:** {h["organelles"]}
- **Labeling Status:** {h["label_status"]}
- **Label Evidence:** Keyword / pattern extraction from scrape text.

## Mitochondria-Specific Summary
Heuristic extraction plus data-acquisition appendix below.

## How to obtain mitochondria-related data (PRIMARY — engineer handoff)
> Use Appendix reference workflow and URL lists.

### 1–6. (template)
Follow Appendix for S3/API/XHR hints.

## Key Findings
{key_findings}

## Databases & Datasets
- See Appendix.

## Quality, Gaps, and Future Work
- **Data Quality Risks:** Heuristic-only summary; verify against primary sources.
- **Useful Next Steps:** Use PostgREST-backed inventory rows in the Appendix for authoritative paths.

## Additional Notes
Deterministic extraction from scrape text (no LLM report body).
"""
    return _finalize_full_markdown_with_appendix(
        website_name,
        url,
        scraped_at_iso,
        access_signals,
        markdown,
        fallback_heuristic=h,
        database_builder_new_ids=database_builder_new_ids,
    )


def generate_markdown_for_site(
    _llm: Any,
    website_name: str,
    url: str,
    scraped_text: str,
    scraped_at_iso: str,
    access_signals: dict,
    *,
    database_builder_new_ids: list[str] | None = None,
) -> str:
    """Build one markdown report (heuristic body + appendix). ``_llm`` is ignored (routing-only API)."""
    if access_signals.get("playwright_scrape_blocked"):
        appendix = format_data_acquisition_playbook_md(
            website_name, url, merge_site_access_signals([]), heuristic=None
        )
        detail = (access_signals.get("scrape_failure_message") or "").strip()
        return (
            f"# {website_name} — Scrape blocked (Playwright setup required)\n\n"
            f"**Source:** {url}\n**Scraped:** {TODAY}\n**Scraped At (ISO):** {scraped_at_iso}\n"
            f"**Website Name:** {website_name}\n"
            f"**Mitochondria Relevance:** Not evaluated — browser scrape did not run.\n\n"
            f"## Status\n\n**No catalog or API data was collected.** "
            f"Install Chromium for Playwright in this environment.\n\n"
            f"```bash\npip install playwright\npython -m playwright install chromium\n```\n\n"
            f"See README for `PLAYWRIGHT_NO_SANDBOX` if needed.\n\n## Technical detail\n\n{detail}\n\n"
            + appendix
        )

    max_scrape = _report_char_budget()
    if len(scraped_text) > max_scrape:
        scraped_text = scraped_text[:max_scrape] + "\n...[overall scrape truncated]"

    return _markdown_full_heuristic_only(
        website_name,
        url,
        scraped_text,
        scraped_at_iso,
        access_signals,
        database_builder_new_ids=database_builder_new_ids,
    )


def filter_scrape_text_for_dataset_ids(scraped_text: str, dataset_ids: list[str]) -> str:
    """Keep lines that mention any of the dataset slugs (for incremental report context)."""
    if not dataset_ids:
        return ""
    lowered = {d.strip().lower() for d in dataset_ids if d and str(d).strip()}
    lines_out: list[str] = []
    for ln in scraped_text.splitlines():
        ll = ln.lower()
        if any(s in ll for s in lowered):
            lines_out.append(ln)
    body = "\n".join(lines_out).strip()
    if not body:
        return (
            "(No scrape lines matched new dataset IDs by substring; "
            "catalog API listing is still authoritative.)\n"
        )
    return body


def _markdown_incremental_heuristic_only(
    website_name: str,
    url: str,
    scraped_text: str,
    scraped_at_iso: str,
    access_signals: dict,
    *,
    new_dataset_ids: list[str],
    scrape_batch_index: int,
    markdown_filename: str,
    prior_known_count: int,
) -> str:
    h = heuristic_extract(scraped_text)
    slug_lines = filter_scrape_text_for_dataset_ids(scraped_text, new_dataset_ids)
    max_ex = 14_000
    if len(slug_lines) > max_ex:
        slug_lines = slug_lines[:max_ex] + "\n...[excerpt truncated]"

    status = (
        "Heuristic incremental report. New slugs are listed explicitly; "
        "Appendix has full catalog signals."
    )

    markdown = f"""# {website_name} — Incremental scrape (batch {scrape_batch_index})

**Source:** {url}
**Scraped:** {TODAY}
**Scraped At (ISO):** {scraped_at_iso}
**Website Name:** {website_name}
**Incremental markdown file:** `{markdown_filename}`
**Previously known datasets (registry):** {prior_known_count}
**New dataset slugs in this batch:** {len(new_dataset_ids)}

## Status
{status}

## New datasets in this run (vs prior registry)
These **{len(new_dataset_ids)}** slug(s) were not in the combined probe + markdown registry before this run. The canonical slug list is in **Database builder — new datasets to add** below (before the Appendix).

## Scrape lines mentioning new dataset slugs
{slug_lines if slug_lines.strip() else "(No page text lines matched these slugs by substring — catalog API + Appendix are still authoritative.)"}

## Key findings (keyword heuristics on full scrape text)
{chr(10).join(f"- {x}" for x in h["key_lines"]) or "- Not explicitly stated"}
"""
    return _finalize_incremental_markdown(
        website_name,
        url,
        scraped_at_iso,
        access_signals,
        markdown,
        new_dataset_ids=new_dataset_ids,
    )


def generate_incremental_markdown_for_site(
    _llm: Any,
    website_name: str,
    url: str,
    scraped_text: str,
    scraped_at_iso: str,
    access_signals: dict,
    *,
    new_dataset_ids: list[str],
    scrape_batch_index: int,
    markdown_filename: str,
    prior_known_count: int,
) -> str:
    """Incremental markdown (heuristic body + appendix). ``_llm`` is ignored."""
    if access_signals.get("playwright_scrape_blocked"):
        return generate_markdown_for_site(
            _llm, website_name, url, scraped_text, scraped_at_iso, access_signals
        )

    max_scrape = _report_char_budget_incremental()
    st = (
        scraped_text
        if len(scraped_text) <= max_scrape
        else scraped_text[:max_scrape] + "\n...[truncated]"
    )

    return _markdown_incremental_heuristic_only(
        website_name,
        url,
        st,
        scraped_at_iso,
        access_signals,
        new_dataset_ids=new_dataset_ids,
        scrape_batch_index=scrape_batch_index,
        markdown_filename=markdown_filename,
        prior_known_count=prior_known_count,
    )
