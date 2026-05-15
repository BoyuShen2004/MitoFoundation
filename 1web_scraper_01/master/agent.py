"""CLI dispatcher for stage 1 (generic HTTP probe + optional website workspace + deep scrape)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .generic_scrape import run_generic_scrape
from .website_workspace import run_website_workspace_scrape


def _default_project_root() -> Path:
    from config.paths import project_root

    return project_root()


def main() -> None:
    if "--sites" in sys.argv:
        from .deep_scrape_agent import main as deep_main

        deep_main()
        return

    parser = argparse.ArgumentParser(description="mitoFoundation2 — stage 1 generic web scrape")
    parser.add_argument(
        "url",
        nargs="?",
        default="",
        help="Landing page URL (required unless using --help)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=_default_project_root(),
        help="Repo root (default: mitoFoundation2 root or MITO2_PROJECT_ROOT)",
    )
    parser.add_argument(
        "--workspace",
        action="store_true",
        help="Write 1web_scraper_01/websites/<slug>/site.md and outputs probe JSON (generic scrape).",
    )
    parser.add_argument(
        "--name",
        "--display-name",
        dest="display_name",
        default="",
        help="Display name for website folder (slug); default from URL host if omitted.",
    )
    parser.add_argument(
        "--description",
        default="",
        help="Short markdown description stored in site.md.",
    )
    parser.add_argument(
        "--data-focus",
        default="",
        help="What datasets or data types to look for (generic; organelle filter is downstream).",
    )
    parser.add_argument(
        "--slug",
        default="",
        help="Optional websites/ folder slug (lowercase alphanumeric; overrides name/URL derivation).",
    )
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    url = (args.url or "").strip()
    if args.workspace:
        if not url:
            parser.error("URL required for --workspace")
        res = run_website_workspace_scrape(
            root,
            display_name=args.display_name,
            url=url,
            description=args.description,
            data_focus=args.data_focus,
            slug_override=args.slug.strip() or None,
            write_probe_legacy=True,
        )
        print(res)
        raise SystemExit(0 if res.get("ok") else 1)
    if not url:
        parser.error("URL required (or use --workspace with URL)")
    res = run_generic_scrape(root, url)
    print(res)


if __name__ == "__main__":
    main()
