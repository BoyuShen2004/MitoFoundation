"""LLM chooses between the heavy catalog pipeline and a lightweight HTTP+markdown path."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Literal

ScrapeMode = Literal["catalog_pipeline", "http_snapshot"]


def plan_scrape_route(
    llm: Any,
    *,
    website_name: str,
    url: str,
) -> ScrapeMode:
    """Ask the model which scrape strategy fits the target (**one** ``llm.invoke`` when enabled).

    Does **not** write any ``outputs/*.md`` — only returns ``catalog_pipeline`` vs ``http_snapshot``.
    Injects an optional excerpt from ``scrape_and_schema_goals.md`` (per-site ``websites/<slug>/guides/``,
    then ``master/deep/guides/``, then generic ``guides/``) so skills shape the decision.

    Set ``SCRAPE_LLM_ROUTE=0`` to always use ``catalog_pipeline`` (no routing LLM call).
    """
    if os.getenv("SCRAPE_LLM_ROUTE", "1").strip().lower() in ("0", "false", "no", "off"):
        return "catalog_pipeline"

    from .guides_loader import load_guide_for_website_optional

    goals_excerpt = load_guide_for_website_optional(
        "scrape_and_schema_goals.md", website_name, default=""
    ).strip()
    if len(goals_excerpt) > 6000:
        goals_excerpt = goals_excerpt[:6000] + "\n…(truncated)…"

    sys = (
        "You decide how to scrape a scientific data catalog or portal. "
        "Reply with a single JSON object only, no markdown fences.\n"
        "- catalog_pipeline: JavaScript/SPA sites, PostgREST/GraphQL/API-heavy catalogs, "
        "incremental probe JSON, optional S3 path probing (use for rich portals).\n"
        "- http_snapshot: Mostly static HTML landing pages, no need for headless browser "
        "or structured API harvest (lighter: fetch + LLM markdown only).\n"
        'Schema: {"mode":"catalog_pipeline"} or {"mode":"http_snapshot"}'
    )
    human = (
        f"Website label: {website_name}\nURL: {url}\n\n"
        f"## Goals excerpt (from guides; may be empty)\n\n{goals_excerpt or '(none)'}\n"
    )
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        out = llm.invoke([SystemMessage(content=sys), HumanMessage(content=human)])
        text = (getattr(out, "content", None) or str(out)).strip()
        m = re.search(r"\{[^{}]*\"mode\"[^{}]*\}", text)
        if m:
            text = m.group(0)
        data = json.loads(text)
        mode = str(data.get("mode", "")).strip().lower()
        if mode in ("http_snapshot", "snapshot", "light", "lightweight"):
            return "http_snapshot"
        return "catalog_pipeline"
    except Exception:
        return "catalog_pipeline"
