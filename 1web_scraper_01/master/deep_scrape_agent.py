"""Multi-site catalog scrape — invoked via ``python -m master.agent --sites …``.

Outputs live under ``1web_scraper_01/outputs/`` (probe JSON, markdown, S3 probe sidecars).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# ``1web_scraper_01/`` on sys.path so ``master`` resolves when run as ``python -m master.agent``.
WS_ROOT = Path(__file__).resolve().parent.parent
if str(WS_ROOT) not in sys.path:
    sys.path.insert(0, str(WS_ROOT))

_REPO_ROOT = WS_ROOT.parent.resolve()
# Parent repo (``mitoFoundation2/``) so ``import agent.chat_web`` works for Codex scrape LLM routing.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
load_dotenv(_REPO_ROOT / ".env", override=False)
os.environ.setdefault("MITO2_PROJECT_ROOT", str(_REPO_ROOT))

from master.deep.config import TODAY
from master.deep.from_website import (
    write_json_report,
    write_python_downloader,
)
from master.deep.scrape_llm import build_llm
from master.deep.tools import list_saved_files, read_websites_list
from master.deep.websites import (
    parse_websites,
    spa_hosts_from_websites_md,
)

from master.scraper_registry import get_scraper

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
# Self-contained under ``outputs/`` (no separate data_downloader tree).
PROBE_JSON_OUT = OUTPUT_DIR / "s3_probed_paths.json"
PROBE_PY_OUT = OUTPUT_DIR / "probed_downloader.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scrape per-website Markdown reports and (for supported sites) probe S3 paths.",
    )
    p.add_argument(
        "--sites",
        nargs="*",
        default=None,
        help="Restrict scraping to one or more website names from per-site guides (e.g. --sites OpenOrganelle).",
    )
    return p.parse_args()


def run(*, selected_sites: list[str] | None = None) -> None:
    llm = build_llm()
    scraped_at_iso = datetime.now().isoformat(timespec="seconds")
    print(f"[INFO] Run start at: {scraped_at_iso}")

    raw = read_websites_list.invoke({})
    websites = parse_websites(raw)
    if not websites:
        raise RuntimeError("No valid website entries found in per-site <site>/guides/websites.md")

    if selected_sites:
        selected = {s.lower().strip() for s in selected_sites if s and s.strip()}
        websites = [(n, u) for (n, u) in websites if n.lower().strip() in selected]
        if not websites:
            raise SystemExit(
                f"[ERROR] None of the requested sites were found in per-site guides: {selected_sites}"
            )

    spa_hosts = spa_hosts_from_websites_md()
    deep_pages = int(os.getenv("DEEP_SCRAPE_MAX_PAGES", "20"))

    all_probe_results: dict[str, dict] = {}

    for website_name, url in websites:
        site_start = datetime.now().isoformat(timespec="seconds")
        print(f"[INFO] Scraping {website_name} at {site_start}")

        scraper = get_scraper(website_name, url, llm=llm)
        probe_results = scraper.scrape_and_report(
            llm=llm,
            website_name=website_name,
            url=url,
            scraped_at_iso=scraped_at_iso,
            spa_hosts=spa_hosts,
            max_pages=deep_pages,
            output_dir=OUTPUT_DIR,
        )
        all_probe_results.update(probe_results)

    if all_probe_results:
        success = sum(
            1
            for r in all_probe_results.values()
            if not r.get("error") and (r.get("em_image_path") or r.get("mito_seg_path"))
        )
        print(
            f"[INFO] S3 probe complete: {success}/{len(all_probe_results)} datasets with usable paths."
        )
        # Legacy aggregates duplicate what OpenOrganelle already stores in ``OpenOrganelle.probe.json``.
        _legacy = (os.getenv("MITO2_WRITE_LEGACY_S3_PROBE_AGGREGATES") or "").strip().lower()
        if _legacy in ("1", "true", "yes"):
            write_json_report(all_probe_results, PROBE_JSON_OUT)
            write_python_downloader(all_probe_results, PROBE_PY_OUT)
            print(
                f"[INFO] Wrote legacy sidecars (MITO2_WRITE_LEGACY_S3_PROBE_AGGREGATES): "
                f"{PROBE_JSON_OUT.name}, {PROBE_PY_OUT.name}"
            )
    else:
        print("[INFO] No S3 probe results (no supported sites scraped, or no IDs found).")

    print(list_saved_files.invoke({}))


def main() -> None:
    print("=" * 64)
    print("  Mitochondria Data Web Scraping Agent")
    print(f"  Started At: {datetime.now().isoformat(timespec='seconds')}")
    print(f"  Date: {TODAY}")
    print("=" * 64)

    args = parse_args()
    run(selected_sites=args.sites)


if __name__ == "__main__":
    main()
