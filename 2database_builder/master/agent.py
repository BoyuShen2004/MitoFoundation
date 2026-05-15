"""Stage 2 CLI: catalog SQLite (scraped .md), optional LLM-guided orchestration, probe → inventory.sqlite."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .catalog_build import DEFAULT_MD_DIR, run_catalog_build
from .guided_schema_agent import main as guided_main

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INV_ROOT = _REPO_ROOT / "0inventory"
if str(_INV_ROOT) not in sys.path:
    sys.path.insert(0, str(_INV_ROOT))
from probe_inventory_db import ingest_probe, open_db


def _registry_enabled() -> bool:
    """Registry sync is ON by default. Set MITO2_DISABLE_REGISTRY=1 to opt out."""
    return os.environ.get("MITO2_DISABLE_REGISTRY", "").strip().lower() not in (
        "1", "true", "yes", "on"
    )


def _default_project_root() -> Path:
    from config.paths import project_root

    return project_root()


def _ingest_latest_probe(project_root: Path, probe_arg: str) -> dict:
    out_db = project_root / "data" / "inventory.sqlite"
    if probe_arg.strip():
        probe = (project_root / probe_arg.strip()).resolve()
    else:
        out_dir = project_root / "1web_scraper_01" / "outputs"
        probes = sorted(out_dir.glob("*.probe.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not probes:
            return {"skipped": True, "reason": "no probe JSON"}
        probe = probes[0]
    conn = open_db(out_db)
    summary = ingest_probe(conn, probe, project_root=project_root)
    conn.close()
    return {"db": str(out_db), "probe": str(probe), **summary}


def _site_stem_from_probe_arg(probe_arg: str) -> str:
    p = (probe_arg or "").strip()
    if not p:
        return ""
    name = Path(p).name
    if name.endswith(".probe.json"):
        return name[: -len(".probe.json")].strip()
    if name.endswith(".json"):
        return name[: -len(".json")].strip()
    return Path(name).stem.strip()


def main() -> None:
    if "--guided" in sys.argv:
        argv = [a for a in sys.argv if a != "--guided"]
        guided_main(argv[1:])
        return

    parser = argparse.ArgumentParser(description="mitoFoundation2 — stage 2 database builder / catalog")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=_default_project_root(),
        help="Repo root (default: mitoFoundation2 or MITO2_PROJECT_ROOT)",
    )
    parser.add_argument(
        "--probe",
        default="",
        help="Relative path to a .probe.json (default: newest under 1web_scraper_01/outputs)",
    )
    parser.add_argument(
        "--probe-inventory-only",
        action="store_true",
        help="Only refresh data/inventory.sqlite from probe JSON (skip catalog DB build).",
    )
    parser.add_argument(
        "--catalog-only",
        action="store_true",
        help="Only build catalog databases from scraped markdown (skip probe inventory ingest).",
    )
    parser.add_argument(
        "--catalog-sites",
        nargs="*",
        default=None,
        help="Optional site tokens for catalog build (same semantics as catalog_build --sites).",
    )
    parser.add_argument(
        "--skip-s3-probe",
        action="store_true",
        help="Skip Janelia S3 probe during catalog build.",
    )
    parser.add_argument(
        "--no-registry",
        action="store_true",
        help="Disable central registry sync after catalog build (overrides MITO2_DISABLE_REGISTRY).",
    )
    args, _unknown = parser.parse_known_args()
    root = Path(args.project_root).resolve()

    sync_reg = _registry_enabled() and not getattr(args, "no_registry", False)

    if args.probe_inventory_only:
        summary = _ingest_latest_probe(root, args.probe)
        print(summary)
        raise SystemExit(0 if not summary.get("skipped") else 1)

    md_dir = (root / "1web_scraper_01" / "outputs").resolve()
    md_files = list(md_dir.glob("*.md")) if md_dir.is_dir() else []
    catalog_ran = False
    catalog_sites = args.catalog_sites if args.catalog_sites else None
    if not catalog_sites:
        inferred = _site_stem_from_probe_arg(args.probe)
        if inferred:
            # Keep catalog build provider-scoped when caller selects a specific probe.
            catalog_sites = [inferred]
    if md_files:
        run_catalog_build(
            md_dir=md_dir,
            sites=catalog_sites,
            skip_s3_probe=args.skip_s3_probe,
            sync_registry=sync_reg,
        )
        catalog_ran = True
    else:
        print(f"[INFO] No scraped .md under {md_dir}; skipping catalog SQLite build.")

    if not args.catalog_only:
        summary = _ingest_latest_probe(root, args.probe)
        print(summary)
        if summary.get("skipped") and not catalog_ran:
            raise SystemExit(1)
        if summary.get("skipped") and catalog_ran:
            print("[WARN] No probe JSON to ingest; data/inventory.sqlite unchanged.")


if __name__ == "__main__":
    main()
