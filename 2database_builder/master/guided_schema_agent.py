"""
LLM-guided stage-2 orchestration: read per-site ``websites/*/`` instructions, then run catalog / probe-inventory steps.

Uses the same ``agent.chat_web.app.llm.complete`` stack as stage-1 (Codex / OpenAI from repo settings).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_INV_ROOT = _REPO_ROOT / "0inventory"
if str(_INV_ROOT) not in sys.path:
    sys.path.insert(0, str(_INV_ROOT))

load_dotenv = None
try:
    from dotenv import load_dotenv as _ld
except ImportError:
    _ld = None
if _ld:
    _ld(_REPO_ROOT / ".env", override=False)
os.environ.setdefault("MITO2_PROJECT_ROOT", str(_REPO_ROOT))

from .catalog_build import DEFAULT_MD_DIR, run_catalog_build
from probe_inventory_db import ingest_probe, open_db


def _websites_dir() -> Path:
    return _REPO_ROOT / "1web_scraper_01" / "websites"


def _load_generic_guides() -> str:
    parts: list[str] = []
    for name in ("scrape_and_schema_goals.md", "outputs_convention.md"):
        p = _REPO_ROOT / "1web_scraper_01" / "guides" / name
        if p.is_file():
            parts.append(f"### {p.name}\n{p.read_text(encoding='utf-8')[:8000]}")
    return "\n\n".join(parts) if parts else "(no generic guides found)"


def _site_slugs(selected: set[str] | None) -> list[str]:
    base = _websites_dir()
    if not base.is_dir():
        return []
    out: list[str] = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        slug = d.name
        if selected is not None and slug.lower() not in selected:
            continue
        if (d / "site.md").is_file() or (d / "guides").is_dir():
            out.append(slug)
    return out


def _site_bundle_text(slug: str, max_chars: int = 12000) -> str:
    d = _websites_dir() / slug
    chunks: list[str] = [f"## Website folder: {slug}"]
    for fname in ("site.md",):
        p = d / fname
        if p.is_file():
            chunks.append(f"### {fname}\n{p.read_text(encoding='utf-8')[:max_chars]}")
    gdir = d / "guides"
    if gdir.is_dir():
        for gp in sorted(gdir.glob("*.md")):
            chunks.append(f"### guides/{gp.name}\n{gp.read_text(encoding='utf-8')[:6000]}")
    return "\n\n".join(chunks)


def _list_markdown_stems(md_dir: Path) -> list[str]:
    if not md_dir.is_dir():
        return []
    return sorted(p.stem for p in md_dir.glob("*.md") if p.is_file())


def _strip_json_fence(raw: str) -> str:
    t = raw.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", t, re.I)
    if m:
        return m.group(1).strip()
    return t


def _complete(messages: list[dict[str, str]], *, max_tokens: int = 2048) -> str:
    from agent.chat_web.app.llm import LLMUnavailableError, complete

    try:
        return complete(messages, max_tokens=max_tokens)
    except LLMUnavailableError as e:
        raise SystemExit(f"[ERROR] LLM unavailable: {e}") from e


def _parse_plan(text: str) -> dict:
    raw = _strip_json_fence(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise SystemExit(f"[ERROR] Model did not return valid JSON: {e}\n---\n{raw[:2000]}") from e


def run_guided(*, selected_site_slugs: list[str] | None, user_goal: str | None) -> None:
    sel = {s.lower().strip() for s in selected_site_slugs} if selected_site_slugs else None
    slugs = _site_slugs(sel)
    md_dir = DEFAULT_MD_DIR.expanduser().resolve()
    stems = _list_markdown_stems(md_dir)

    site_docs = "\n\n".join(_site_bundle_text(s) for s in slugs) if slugs else "(no website workspaces yet)"
    generic = _load_generic_guides()

    goal = (user_goal or "").strip() or (
        "Build or refresh catalog SQLite databases and schema markdown from scraped reports under outputs/, "
        "honouring each site's site.md (and optional guides/). Run Pipeline Studio probe inventory ingest if appropriate."
    )

    system = (
        "You are a pipeline orchestrator for mitoFoundation2 stage 2 (database builder / catalog). "
        "You MUST respond with a single JSON object only (no markdown outside JSON). "
        "Response JSON shape:\n"
        "{\n"
        '  "build_catalog": true|false,\n'
        '  "catalog_site_tokens": ["OpenOrganelle"] | null,\n'
        '  "skip_s3_probe": true|false,\n'
        '  "ingest_latest_probe_inventory": true|false,\n'
        '  "probe_inventory_relative_path": "1web_scraper_01/outputs/foo.probe.json" | null,\n'
        '  "notes": "short rationale"\n'
        "}\n"
        "Rules:\n"
        "- If scraped markdown stems include a site name (e.g. OpenOrganelle.md), set build_catalog true unless "
        "site docs say to defer.\n"
        "- catalog_site_tokens: list of stems to pass to the builder (null = auto one job per website: "
        "prefer canonical `<Site>.md`; if missing, merge `<Site>_NN.md`).\n"
        "- For Janelia/OpenOrganelle-style COSEM catalogs, keep skip_s3_probe false unless the site doc says "
        "probing is unnecessary.\n"
        "- ingest_latest_probe_inventory true keeps data/inventory.sqlite aligned for the web UI unless docs say "
        "to skip.\n"
    )

    user = (
        f"## User goal\n{goal}\n\n"
        f"## Markdown stems in {md_dir} ({len(stems)})\n{', '.join(stems) or '(none)'}\n\n"
        f"## Site workspace docs\n{site_docs}\n\n"
        f"## Generic guides excerpt\n{generic}\n"
    )

    plan = _parse_plan(_complete([{"role": "system", "content": system}, {"role": "user", "content": user}]))

    print(json.dumps(plan, indent=2))

    if plan.get("build_catalog"):
        tokens = plan.get("catalog_site_tokens")
        sites: list[str] | None = None
        if isinstance(tokens, list):
            sites = [str(x).strip() for x in tokens if str(x).strip()]
            if not sites:
                sites = None
        skip_probe = bool(plan.get("skip_s3_probe"))
        run_catalog_build(md_dir=md_dir, sites=sites, skip_s3_probe=skip_probe)

    if plan.get("ingest_latest_probe_inventory"):
        rel = plan.get("probe_inventory_relative_path")
        out_db = _REPO_ROOT / "data" / "inventory.sqlite"
        if isinstance(rel, str) and rel.strip():
            probe = (_REPO_ROOT / rel.strip()).resolve()
        else:
            out_dir = _REPO_ROOT / "1web_scraper_01" / "outputs"
            probes = sorted(out_dir.glob("*.probe.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not probes:
                print("[WARN] ingest_latest_probe_inventory requested but no *.probe.json found.")
                return
            probe = probes[0]
        conn = open_db(out_db)
        summary = ingest_probe(conn, probe, project_root=_REPO_ROOT)
        conn.close()
        print({"db": str(out_db), "probe": str(probe), **summary})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2 — LLM-guided database builder / catalog orchestration")
    p.add_argument(
        "--sites",
        nargs="*",
        default=None,
        help="Optional website folder slugs under 1web_scraper_01/websites/ to include in context",
    )
    p.add_argument("--goal", default="", help="Optional natural-language goal for the planner")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_guided(selected_site_slugs=args.sites, user_goal=args.goal or None)


if __name__ == "__main__":
    main()
