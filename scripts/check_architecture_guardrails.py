#!/usr/bin/env python3
"""Architecture guardrail checks for mitoFoundation2.

Checks:
1) Forbid legacy imports from moved root packages (registry/pipeline).
2) Ensure approved compatibility shims remain thin forwarders.
3) Warn when generated output appears to be edited as source of truth.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PY_FILES = [p for p in ROOT.rglob("*.py") if ".venv" not in p.parts and "node_modules" not in p.parts]

FORBIDDEN_PATTERNS = [
    re.compile(r"^\s*from\s+registry(\.|\\s)", re.MULTILINE),
    re.compile(r"^\s*import\s+registry(\.|\\s|$)", re.MULTILINE),
    re.compile(r"^\s*from\s+pipeline(\.|\\s)", re.MULTILINE),
    re.compile(r"^\s*import\s+pipeline(\.|\\s|$)", re.MULTILINE),
]

SHIMS = {
    ROOT / "1web_scraper_01" / "agent.py": "from master.agent import main",
    ROOT / "2database_builder" / "agent.py": "from master.agent import main",
    ROOT / "3data_downloader" / "raw_browser.py": "from downloader_master.raw_browser import main",
}


def check_forbidden_imports() -> list[str]:
    errors: list[str] = []
    for path in PY_FILES:
        text = path.read_text(encoding="utf-8")
        for pat in FORBIDDEN_PATTERNS:
            if pat.search(text):
                errors.append(f"forbidden import pattern in {path.relative_to(ROOT)}")
                break
    return errors


def check_shims() -> list[str]:
    errors: list[str] = []
    for path, required_line in SHIMS.items():
        if not path.exists():
            errors.append(f"missing required compatibility shim: {path.relative_to(ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        if required_line not in text:
            errors.append(f"shim no longer forwards to canonical target: {path.relative_to(ROOT)}")
        if len(text.splitlines()) > 30:
            errors.append(f"shim too large; likely duplicated logic: {path.relative_to(ROOT)}")
    return errors


def check_generated_outputs() -> list[str]:
    warnings: list[str] = []
    dist_dir = ROOT / "frontend" / "dist"
    source_dir = ROOT / "frontend" / "public" / "training-yaml-ui"
    if dist_dir.is_dir() and source_dir.is_dir():
        for rel in ("training-yaml-ui/index.html", "training-yaml-ui/app.js"):
            if (dist_dir / rel).is_file() and (source_dir / rel).is_file():
                # Informational warning only; these files can diverge after build.
                warnings.append(
                    f"generated artifact present with matching source: frontend/dist/{rel} "
                    f"(source is frontend/public/{rel})"
                )
    return warnings


def main() -> int:
    errors = []
    errors.extend(check_forbidden_imports())
    errors.extend(check_shims())
    warnings = check_generated_outputs()

    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("Architecture guardrails passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

