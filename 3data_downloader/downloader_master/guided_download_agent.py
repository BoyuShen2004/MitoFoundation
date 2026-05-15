"""
LLM-guided stage-3 orchestration: read site docs, then run OpenOrganelle generator, BossDB stub, or volume stub.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
if load_dotenv:
    load_dotenv(_REPO / ".env", override=False)
os.environ.setdefault("MITO2_PROJECT_ROOT", str(_REPO))


def _websites_dir() -> Path:
    return _REPO / "1web_scraper_01" / "websites"


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


def _site_bundle(slug: str) -> str:
    d = _websites_dir() / slug
    parts = [f"## {slug}"]
    for fn in ("site.md",):
        p = d / fn
        if p.is_file():
            parts.append(f"### {fn}\n{p.read_text(encoding='utf-8')[:10000]}")
    return "\n\n".join(parts)


def _schema_summary() -> str:
    db = _REPO / "2database_builder" / "outputs" / "databases"
    if not db.is_dir():
        return "(no catalog databases yet)"
    dbs = sorted(db.glob("*.db"))
    return "Catalog DBs:\n" + "\n".join(f"  - {p.name}" for p in dbs) if dbs else "(no *.db)"


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
    return json.loads(raw)


def _run_openorganelle(argv_extra: list[str]) -> None:
    dl = Path(__file__).resolve().parent.parent
    if str(dl) not in sys.path:
        sys.path.insert(0, str(dl))
    from openorganelle.agent import main as og_main

    sys.argv = [str(dl / "openorganelle" / "agent.py")] + argv_extra
    og_main()


def run_guided(*, selected: list[str] | None, user_goal: str | None) -> None:
    sel = {s.lower().strip() for s in selected} if selected else None
    slugs = _site_slugs(sel)
    docs = "\n\n".join(_site_bundle(s) for s in slugs) if slugs else "(no website workspaces)"
    goal = (user_goal or "").strip() or (
        "Choose how to run stage 3: OpenOrganelle foundation download script generation, BossDB (if documented), "
        "or the simple volume stub. Respect site profile (site.md) and repo guide constraints."
    )

    system = (
        "You orchestrate mitoFoundation2 stage 3 (downloads). "
        "Respond with ONE JSON object only:\n"
        "{\n"
        '  "mode": "openorganelle" | "bossdb" | "volume_stub",\n'
        '  "argv": ["--list"] ,\n'
        '  "notes": "..."\n'
        "}\n"
        "`argv` are extra arguments for that mode (excluding program name). "
        "For openorganelle, use flags like --db, --mode, --execute, --dataset, --list. "
        "Default DB path is fine to omit. For volume_stub, use --site, --n-crops, etc.\n"
    )
    user = f"## Goal\n{goal}\n\n## Catalog status\n{_schema_summary()}\n\n## Site docs\n{docs}\n"

    plan = _parse_plan(_complete([{"role": "system", "content": system}, {"role": "user", "content": user}]))
    print(json.dumps(plan, indent=2))

    mode = str(plan.get("mode") or "openorganelle").lower().strip()
    argv = plan.get("argv")
    if not isinstance(argv, list):
        argv = []
    argv = [str(x) for x in argv]

    if mode == "volume_stub":
        py = sys.executable
        s3 = _REPO / "3data_downloader"
        cmd = [py, str(s3 / "downloader_master" / "agent.py"), "--volume-stub"] + argv
        subprocess.check_call(cmd, cwd=str(s3))
        return
    if mode == "bossdb":
        py = sys.executable
        s3 = _REPO / "3data_downloader"
        cmd = [py, str(s3 / "downloader_master" / "agent.py"), "--site", "bossdb", "--download"] + argv
        subprocess.check_call(cmd, cwd=str(s3))
        return

    _run_openorganelle(argv)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 3 — LLM-guided downloader")
    p.add_argument("--sites", nargs="*", default=None, help="Website slugs for context")
    p.add_argument("--goal", default="")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_guided(selected=args.sites, user_goal=args.goal or None)


if __name__ == "__main__":
    main()
