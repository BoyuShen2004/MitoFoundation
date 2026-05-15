"""
from_website.py — Extract dataset IDs from scraped markdown / SQLite DB,
probe S3 for exact download paths, and emit a ready-to-use DATASETS dict
(compatible with download_cellmap.py).

Usage:
  python -m openorganelle.mito_scrape.from_website --source openorganelle [--limit N] [--dataset ID ...]
  python -m openorganelle.mito_scrape.from_website --db /path/to/OpenOrganelle.db [--limit N]
  python -m openorganelle.mito_scrape.from_website --ids aic_desmosome-3 jrc_mus-kidney
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

from .s3_probe import probe_janelia_dataset, probe_result_to_download_config


def _default_repo_root() -> Path:
    """``master/deep/from_website.py`` → repo root (mitoFoundation2)."""
    import os

    env = (os.environ.get("MITO2_PROJECT_ROOT") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def _default_scraper_outputs() -> Path:
    return _default_repo_root() / "1web_scraper_01" / "outputs"


# Default paths (self-contained under ``1web_scraper_01/outputs/`` unless overridden on CLI)
_repo = _default_repo_root()
_out = _default_scraper_outputs()
DEFAULT_DB = _repo / "2database_builder" / "outputs" / "databases" / "OpenOrganelle.db"
DEFAULT_OUT_JSON = _out / "s3_probed_paths.json"
DEFAULT_OUT_PY = _out / "probed_downloader.py"


# ── DB helpers ─────────────────────────────────────────────────────────────────

def load_dataset_ids_from_db(
    db_path: Path,
    *,
    mito_only: bool = True,
    limit: Optional[int] = None,
) -> list[str]:
    """Read dataset_name values from an OpenOrganelle SQLite DB."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    where = "WHERE mitochondria_in_layer_names = 1" if mito_only else ""
    order = "ORDER BY dataset_name"
    lim = f"LIMIT {limit}" if limit else ""
    rows = conn.execute(
        f"SELECT dataset_name FROM datasets {where} {order} {lim}"
    ).fetchall()
    conn.close()
    return [r["dataset_name"] for r in rows]


# ── batch prober ───────────────────────────────────────────────────────────────

def _short_url(u: str | None, max_len: int = 96) -> str:
    s = (u or "").strip()
    if not s:
        return ""
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _print_probe_status(result: dict, cfg: dict | None) -> None:
    """Human-readable lines for database_builder / CLI (honest about partial S3 probes)."""
    if cfg is not None:
        return
    if result.get("error"):
        return
    em = result.get("em_image_path")
    seg = result.get("mito_seg_path") or result.get("mito_pred_path")
    labels = result.get("label_names") or []
    vz = result.get("voxel_size_nm")

    if em and seg:
        # Should not happen if cfg is None; print anyway for debugging
        print("  NOTE: probe has em+seg but config normalizer returned None (unexpected).")
        return

    if em:
        print(f"  img:  {_short_url(em)}")
        print(
            "  NOTE: EM path found; no mito_seg/mito_pred in this S3 walk — "
            "labeled paths may still come from PostgREST/layers in the DB."
        )
        if vz:
            print(f"  size: {vz} nm")
        return

    if seg:
        print(f"  seg:  {_short_url(seg)}")
        print(
            "  NOTE: mito path found but no EM array in this probe — "
            "check zarr/n5 layout or use scraped download_volume_url."
        )
        return

    if labels:
        n = len(labels)
        head = ", ".join(labels[:5])
        tail = f" …+{n - 5} more" if n > 5 else ""
        print(f"  NOTE: S3 listed {n} label dir(s) ({head}{tail}) but EM/mito heuristics did not match.")
        print(
            "  HINT: database_builder merge (resolve_paths) or unlabeled template may still supply img paths."
        )
        return

    print(
        "  WARNING: no EM or mito paths from S3 listing — "
        "dataset prefix may be empty or layout differs from probe heuristics."
    )


def batch_probe(
    dataset_ids: list[str],
    *,
    delay_sec: float = 1.0,
    verbose: bool = True,
) -> dict[str, dict]:
    """
    Probe S3 for each dataset ID. Returns {dataset_id: probe_result}.
    Adds a small delay between requests to be polite to S3.
    """
    results: dict[str, dict] = {}
    total = len(dataset_ids)
    for i, did in enumerate(dataset_ids, start=1):
        if verbose:
            print(f"[{i}/{total}] Probing {did} ...", flush=True)
        result = probe_janelia_dataset(did)
        results[did] = result
        cfg = probe_result_to_download_config(result)
        if verbose:
            if result.get("error"):
                print(f"  ERROR: {result['error']}")
            elif cfg:
                print(f"  img:  {cfg['img_path']}")
                print(f"  seg:  {cfg['seg_path']}")
                print(f"  size: {cfg['voxel_size']} nm")
            else:
                _print_probe_status(result, cfg)
        if i < total:
            time.sleep(delay_sec)
    return results


# ── output writers ─────────────────────────────────────────────────────────────

def write_json_report(results: dict[str, dict], out_path: Path) -> None:
    """Write full probe results as JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "total": len(results),
        "datasets": results,
    }
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[OK] JSON report → {out_path}")


def write_python_downloader(results: dict[str, dict], out_path: Path) -> None:
    """
    Write a Python module containing a DATASETS dict ready for use with
    download_cellmap.py (or the generated_downloader_N.py pattern).
    """
    lines = [
        '"""',
        "Auto-generated by openorganelle.mito_scrape.from_website",
        "S3 paths discovered by direct bucket listing (no credentials required).",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "DATASETS = {",
    ]

    for did, probe in sorted(results.items()):
        cfg = probe_result_to_download_config(probe)
        if not cfg:
            lines.append(f"    # {did}: SKIPPED (no usable paths found)")
            continue
        key = did.replace("-", "_")
        voxel = cfg["voxel_size"]
        label_names = cfg.get("_label_names", [])
        lines.extend([
            f'    "{key}": {{',
            f'        "img_path": {cfg["img_path"]!r},',
            f'        "seg_path": {cfg["seg_path"]!r},',
            f'        "voxel_size": {voxel!r},',
            f'        # available labels: {label_names}',
            f'    }},',
        ])

    lines.extend([
        "}",
        "",
        "# ── quick download helper (mirrors download_cellmap.py) ────────────────────",
        "# from download_cellmap import download_cellmap_data",
        "# download_cellmap_data('aic_desmosome_3', offsets=[(0,0,0)],",
        "#     chunk_shape=(512,256,512), output_format='h5', output_dir='./data')",
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Python downloader → {out_path}")


def write_schema_markdown(results: dict[str, dict], out_path: Path) -> None:
    """
    Write a human-readable schema markdown summarising all probed datasets and
    their exact S3 download paths.
    """
    import datetime
    lines = [
        "# Probed Dataset Download Schema",
        "",
        f"_Generated by mitoFoundation2 `from_website.py` — {datetime.date.today()}_",
        "",
        "## Overview",
        "",
        f"- **Total datasets probed:** {len(results)}",
        f"- **Datasets with usable paths:** {sum(1 for r in results.values() if probe_result_to_download_config(r))}",
        "",
        "## Download Method",
        "",
        "All paths point to anonymous-accessible AWS S3 (`s3://janelia-cosem-datasets/...`).",
        "Use `fibsem_tools.io.read(path=..., storage_options={'anon': True})` to read arrays.",
        "Or use `download_cellmap.py` with the paths below.",
        "",
        "## Datasets",
        "",
        "| Dataset | EM Image Path | Mito Seg Path | Voxel Size (nm) | All Labels |",
        "|---------|--------------|---------------|-----------------|------------|",
    ]

    for did, probe in sorted(results.items()):
        cfg = probe_result_to_download_config(probe)
        if not cfg:
            lines.append(f"| `{did}` | *(no path found)* | *(no path found)* | — | — |")
            continue
        img = cfg["img_path"].replace("s3://janelia-cosem-datasets/", "s3://…/")
        seg = cfg["seg_path"].replace("s3://janelia-cosem-datasets/", "s3://…/")
        voxel = cfg["voxel_size"]
        labels = ", ".join(f"`{l}`" for l in cfg.get("_label_names", [])[:5])
        if len(cfg.get("_label_names", [])) > 5:
            labels += f" _(+{len(cfg['_label_names']) - 5} more)_"
        lines.append(f"| `{did}` | `{img}` | `{seg}` | {voxel} | {labels} |")

    lines += [
        "",
        "## Notes",
        "",
        "- EM image paths use scale `s1` (2× downsampled) by default. Change to `s0` for full resolution.",
        "- Segmentation paths use scale `s0` (full resolution).",
        "- Voxel sizes are read from `attributes.json` in the S3 store.",
        "- Re-run `from_website.py` to refresh paths when new datasets are added.",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Schema markdown → {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Probe S3 for dataset download paths and emit schema + downloader.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Probe all mito datasets from OpenOrganelle DB
  python -m openorganelle.mito_scrape.from_website --db /path/to/OpenOrganelle.db

  # Probe specific datasets by ID
  python -m openorganelle.mito_scrape.from_website --ids aic_desmosome-3 jrc_mus-kidney jrc_mus-liver

  # Limit to first 10 mito datasets from DB
  python -m openorganelle.mito_scrape.from_website --db /path/to/OpenOrganelle.db --limit 10

  # Include non-mito datasets too
  python -m openorganelle.mito_scrape.from_website --db /path/to/OpenOrganelle.db --all-datasets
        """,
    )
    p.add_argument("--db", type=Path, default=None,
                   help="SQLite DB path (default: OpenOrganelle.db in 2database_builder/outputs/databases)")
    p.add_argument("--ids", nargs="+", default=None,
                   help="Explicit dataset IDs to probe (bypasses DB)")
    p.add_argument("--limit", type=int, default=None,
                   help="Max number of datasets to probe")
    p.add_argument("--all-datasets", action="store_true",
                   help="Include non-mito datasets (default: mito-only)")
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON,
                   help=f"Output JSON path (default: {DEFAULT_OUT_JSON})")
    p.add_argument("--out-py", type=Path, default=DEFAULT_OUT_PY,
                   help=f"Output Python downloader path (default: {DEFAULT_OUT_PY})")
    p.add_argument(
        "--out-md",
        type=Path,
        default=_default_repo_root() / "1web_scraper_01" / "outputs" / "ProbeResults.schema.md",
        help="Output schema markdown path",
    )
    p.add_argument("--delay", type=float, default=1.0,
                   help="Delay in seconds between S3 requests (default: 1.0)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.ids:
        dataset_ids = args.ids
        if args.limit:
            dataset_ids = dataset_ids[:args.limit]
    else:
        db_path = args.db or DEFAULT_DB
        if not db_path.exists():
            print(f"[ERROR] DB not found: {db_path}", file=sys.stderr)
            print("Hint: run 2database_builder/agent.py first, or pass --ids explicitly.", file=sys.stderr)
            sys.exit(1)
        dataset_ids = load_dataset_ids_from_db(
            db_path,
            mito_only=not args.all_datasets,
            limit=args.limit,
        )

    if not dataset_ids:
        print("[WARN] No dataset IDs to probe.")
        sys.exit(0)

    print(f"[INFO] Probing {len(dataset_ids)} dataset(s) ...")
    results = batch_probe(dataset_ids, delay_sec=args.delay, verbose=True)

    write_json_report(results, args.out_json)
    write_python_downloader(results, args.out_py)
    write_schema_markdown(results, args.out_md)

    success = sum(1 for r in results.values() if probe_result_to_download_config(r))
    print(f"\n[DONE] {success}/{len(results)} datasets have usable download paths.")


if __name__ == "__main__":
    main()
