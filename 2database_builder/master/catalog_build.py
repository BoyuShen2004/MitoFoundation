"""
Markdown catalog → SQLite + schema markdown + Janelia S3 probe (mitoFoundation parity).

Self-contained under ``mitoFoundation2``; probes reuse ``1web_scraper_01`` batch_probe.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path

# ``2database_builder/`` on sys.path when run as ``python agent.py`` from that folder.
_SCHEMA_ROOT = Path(__file__).resolve().parent.parent
if str(_SCHEMA_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCHEMA_ROOT))
_REPO_ROOT = Path(__file__).resolve().parents[2]
_INV_ROOT = _REPO_ROOT / "0inventory"
if str(_INV_ROOT) not in sys.path:
    sys.path.insert(0, str(_INV_ROOT))

from openorganelle.builder import build_merged_site, build_one_markdown
from openorganelle.db import connect, migrate, write_probe_results
from download_inventory import materialize_resolved_paths
from openorganelle.report import render_schema_markdown

_JANELIA_SLUG_RE = re.compile(r"^[a-z][a-z0-9_\-]+$")
_NUMBERED_MD_RE = re.compile(r"^(?P<base>.+)_(?P<n>\d+)$")


def _repo_root() -> Path:
    from config.paths import project_root

    return project_root()


REPO_ROOT = _repo_root()
DATABASE_BUILDER_ROOT = _SCHEMA_ROOT
DEFAULT_SCRAPER_ROOT = REPO_ROOT / "1web_scraper_01"
DEFAULT_MD_DIR = DEFAULT_SCRAPER_ROOT / "outputs"
DEFAULT_DB_DIR = DATABASE_BUILDER_ROOT / "outputs" / "databases"
DEFAULT_SCHEMA_MD = DATABASE_BUILDER_ROOT / "outputs" / "schema_md"
DEFAULT_PROBE_LIMIT = 200
DEFAULT_PROBE_DELAY = 0.8


def _ensure_batch_probe():
    """Load stage-1 S3 probe functions without ``master`` package-name collisions.

    ``2database_builder`` itself uses a top-level ``master`` package, so importing
    ``master.deep.from_website`` can resolve to the wrong package depending on ``sys.path``.
    We load ``1web_scraper_01/master/deep/s3_probe.py`` by file path and expose a compatible
    ``batch_probe(dataset_ids, delay_sec, verbose)`` callable.
    """
    ws01 = REPO_ROOT / "1web_scraper_01"
    if not ws01.is_dir():
        raise RuntimeError(f"Missing 1web_scraper_01 at {ws01}")
    s3_probe_py = ws01 / "master" / "deep" / "s3_probe.py"
    if not s3_probe_py.is_file():
        raise RuntimeError(f"Missing stage-1 probe module at {s3_probe_py}")

    spec = importlib.util.spec_from_file_location("mito2_stage1_s3_probe", s3_probe_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load stage-1 probe module: {s3_probe_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    probe_janelia_dataset = getattr(mod, "probe_janelia_dataset", None)
    probe_result_to_download_config = getattr(mod, "probe_result_to_download_config", None)
    if not callable(probe_janelia_dataset) or not callable(probe_result_to_download_config):
        raise RuntimeError("stage-1 s3_probe.py missing required probe functions")

    def batch_probe(dataset_ids: list[str], *, delay_sec: float = 1.0, verbose: bool = True) -> dict[str, dict]:
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
                    print(f"  img:  {cfg.get('img_path')}")
                    print(f"  seg:  {cfg.get('seg_path')}")
                    print(f"  size: {cfg.get('voxel_size')} nm")
            if i < total:
                time.sleep(delay_sec)
        return results

    return batch_probe


def _is_janelia_db(db_path: Path) -> bool:
    try:
        c = connect(db_path)
        rows = c.execute("SELECT dataset_name FROM datasets LIMIT 10").fetchall()
        c.close()
        if not rows:
            return False
        sluglike = sum(1 for r in rows if _JANELIA_SLUG_RE.match(r["dataset_name"] or ""))
        return sluglike >= max(1, len(rows) // 2)
    except Exception:
        return False


def _probe_db(db_path: Path, limit: int, delay: float, verbose: bool = True) -> int:
    if not _is_janelia_db(db_path):
        print(f"[INFO] S3 probe: skipping {db_path.name} (not a Janelia/COSEM DB)")
        return 0
    try:
        batch_probe = _ensure_batch_probe()
    except Exception as e:
        print(f"[WARN] Cannot load batch_probe (S3 probe skipped): {e}")
        return 0

    conn = connect(db_path)
    migrate(conn)
    rows = conn.execute("SELECT dataset_name FROM datasets ORDER BY dataset_name").fetchall()
    conn.close()

    ids = [r["dataset_name"] for r in rows[:limit]]
    if not ids:
        print("[INFO] No datasets in DB to probe.")
        return 0

    print(f"[INFO] S3 probe: {len(ids)} dataset(s) from {db_path.name} ...")
    results = batch_probe(ids, delay_sec=delay, verbose=verbose)
    n = write_probe_results(db_path, results)
    print(f"[INFO] S3 probe: updated {n} rows in {db_path.name}")
    return n


def _output_stem_for_site(token: str, matched: list[Path]) -> str:
    tok = token.strip()
    tok_l = tok.lower()
    if len(matched) == 1:
        s = matched[0].stem
        if s.lower() == tok_l:
            return s
        return tok
    return tok


def _merge_jobs_by_output_stem(jobs: list[tuple[str, list[Path]]]) -> list[tuple[str, list[Path]]]:
    buckets: OrderedDict[str, list[Path]] = OrderedDict()
    for stem, paths in jobs:
        if stem not in buckets:
            buckets[stem] = []
        seen = {p.resolve() for p in buckets[stem]}
        for p in paths:
            rp = p.resolve()
            if rp not in seen:
                buckets[stem].append(p)
                seen.add(rp)
    return [(s, sorted(ps, key=lambda p: p.name)) for s, ps in buckets.items()]


def _stem_base_and_index(stem: str) -> tuple[str, int | None]:
    m = _NUMBERED_MD_RE.match(stem)
    if not m:
        return stem, None
    try:
        return m.group("base"), int(m.group("n"))
    except ValueError:
        return stem, None


def _canonical_preferred_jobs(all_md: list[Path]) -> list[tuple[str, list[Path]]]:
    """One build job per website stem.

    Preference order:
    1) canonical ``<Site>.md`` if present (single source of truth)
    2) otherwise merge numbered ``<Site>_NN.md`` files into one job
    """
    canonical: dict[str, Path] = {}
    numbered: dict[str, list[Path]] = {}
    display: dict[str, str] = {}
    for p in sorted(all_md, key=lambda x: x.name):
        base, idx = _stem_base_and_index(p.stem)
        key = base.lower()
        display.setdefault(key, base)
        if idx is None:
            canonical[key] = p
        else:
            numbered.setdefault(key, []).append(p)

    jobs: list[tuple[str, list[Path]]] = []
    for key in sorted(set(display.keys())):
        out_stem = canonical[key].stem if key in canonical else display[key]
        if key in canonical:
            jobs.append((out_stem, [canonical[key]]))
        else:
            jobs.append((out_stem, sorted(numbered.get(key, []), key=lambda p: p.name)))
    return jobs


def _canonical_preferred_subset(token: str, matched: list[Path]) -> tuple[str, list[Path]]:
    """For a selected token, prefer canonical ``token.md`` over ``token_*.md`` when available."""
    by_stem = {p.stem.lower(): p for p in matched}
    tok_l = token.strip().lower()
    if tok_l in by_stem:
        p = by_stem[tok_l]
        return p.stem, [p]
    return _output_stem_for_site(token, matched), matched


def resolve_build_jobs(md_dir: Path, selected_sites: list[str] | None) -> list[tuple[str, list[Path]]]:
    md_dir = md_dir.expanduser().resolve()
    if not md_dir.is_dir():
        raise RuntimeError(f"Input markdown folder not found: {md_dir}")

    all_md = sorted([p for p in md_dir.glob("*.md") if p.is_file()])
    if not all_md:
        raise SystemExit(f"[ERROR] No .md files in {md_dir}")

    if not selected_sites:
        return _canonical_preferred_jobs(all_md)

    jobs: list[tuple[str, list[Path]]] = []
    seen_tokens: set[str] = set()
    for raw in selected_sites:
        token = raw.strip()
        if not token:
            continue
        key = token.lower()
        if key in seen_tokens:
            continue
        seen_tokens.add(key)

        tok_l = token.lower()
        matched = [p for p in all_md if p.stem.lower() == tok_l or p.stem.lower().startswith(tok_l + "_")]
        if not matched:
            raise SystemExit(
                f"[ERROR] No markdown matched site {token!r} in {md_dir}. "
                f"Available stems: {', '.join(p.stem for p in all_md)}"
            )
        matched = sorted(matched)
        out_stem, chosen = _canonical_preferred_subset(token, matched)
        jobs.append((out_stem, chosen))

    return _merge_jobs_by_output_stem(jobs)


def _clean_output_stems(db_dir: Path, schema_md_dir: Path, output_stems: list[str]) -> tuple[int, int]:
    removed_db = removed_md = 0
    for stem in sorted(set(output_stems)):
        dbp = db_dir / f"{stem}.db"
        mdp = schema_md_dir / f"{stem}.schema.md"
        if dbp.is_file():
            dbp.unlink()
            removed_db += 1
        if mdp.is_file():
            mdp.unlink()
            removed_md += 1
    return removed_db, removed_md


def _sync_registry_after_build(out_stem: str, catalog_db_path: Path) -> None:
    """Populate the central registry from the freshly-built catalog DB."""
    try:
        import sys as _sys
        _rr = REPO_ROOT
        if str(_rr) not in _sys.path:
            _sys.path.insert(0, str(_rr))
        from agent.orchestration.registry.schema import open_registry
        from agent.orchestration.registry.providers import get_provider, known_providers

        probe_path = DEFAULT_SCRAPER_ROOT / "outputs" / f"{out_stem}.probe.json"
        try:
            # Use the factory so that any registered provider can be synced by stem name.
            provider = get_provider(
                out_stem,
                probe_path=probe_path if probe_path.is_file() else None,
                catalog_db=catalog_db_path,
            )
        except KeyError:
            # Unknown provider stem — fall back to OpenOrganelle for backward compat
            # (the catalog DB schema is OO-derived and understood by OpenOrganelleProvider).
            from agent.orchestration.registry.providers.openorganelle import OpenOrganelleProvider
            provider = OpenOrganelleProvider(
                probe_path=probe_path if probe_path.is_file() else None,
                catalog_db=catalog_db_path,
            )
        conn = open_registry()
        provider.ingest_discovery(conn)
        provider.resolve_assets(conn)
        conn.commit()
        conn.close()
        print(f"[INFO] Registry synced from {catalog_db_path.name}")
    except Exception as exc:
        print(f"[WARN] Registry sync failed (non-fatal): {exc}")


def parse_catalog_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build catalog SQLite + schema markdown from scraped .md files.")
    p.add_argument("--md-dir", type=Path, default=DEFAULT_MD_DIR)
    p.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    p.add_argument("--schema-md-dir", type=Path, default=DEFAULT_SCHEMA_MD)
    p.add_argument(
        "--sites",
        nargs="*",
        default=None,
        help=(
            "Stem tokens matching scraped markdown. Canonical `<Site>.md` is preferred when present; "
            "otherwise matching `<Site>_NN.md` files are merged."
        ),
    )
    p.add_argument(
        "--skip-s3-probe",
        action="store_true",
        help="Skip Janelia S3 batch probe (not recommended for download-ready paths).",
    )
    p.add_argument(
        "--sync-registry",
        action="store_true",
        help="After building, sync datasets and assets into data/registry.sqlite.",
    )
    return p.parse_args(argv)


def run_catalog_build(
    *,
    md_dir: Path | None = None,
    db_dir: Path | None = None,
    schema_md_dir: Path | None = None,
    sites: list[str] | None = None,
    skip_s3_probe: bool = False,
    sync_registry: bool = False,
) -> list[tuple[str, list[Path], dict[str, int]]]:
    db_d = (db_dir or DEFAULT_DB_DIR).expanduser().resolve()
    schema_d = (schema_md_dir or DEFAULT_SCHEMA_MD).expanduser().resolve()
    db_d.mkdir(parents=True, exist_ok=True)
    schema_d.mkdir(parents=True, exist_ok=True)

    md_d = (md_dir or DEFAULT_MD_DIR).expanduser().resolve()
    jobs = resolve_build_jobs(md_d, sites)
    output_stems = [j[0] for j in jobs]

    removed = _clean_output_stems(db_d, schema_d, output_stems)

    print(f"[INFO] Input:  {md_d}")
    print(f"[INFO] DB out: {db_d}")
    print(f"[INFO] Schema: {schema_d}")
    print(f"[INFO] Cleared: {removed[0]} *.db, {removed[1]} *.schema.md")

    results: list[tuple[str, list[Path], dict[str, int]]] = []
    for out_stem, md_paths in jobs:
        db_path = db_d / f"{out_stem}.db"
        schema_md_path = schema_d / f"{out_stem}.schema.md"
        if len(md_paths) == 1:
            stats = build_one_markdown(md_paths[0], db_path, schema_md_path)
        else:
            stats = build_merged_site(out_stem, md_paths, db_path, schema_md_path)
        results.append((out_stem, md_paths, stats))
        src_desc = ", ".join(p.name for p in md_paths) if len(md_paths) > 1 else md_paths[0].name
        print(
            f"[OK] {src_desc} → {out_stem}.db "
            f"(datasets={stats['datasets']}, layers={stats['layers']}, mito={stats['mito_datasets']})"
        )

    for out_stem, md_paths, stats in results:
        db_path = db_d / f"{out_stem}.db"
        schema_md_path = schema_d / f"{out_stem}.schema.md"
        if not db_path.is_file():
            continue
        if not skip_s3_probe:
            _probe_db(db_path, DEFAULT_PROBE_LIMIT, DEFAULT_PROBE_DELAY)
        rstats = materialize_resolved_paths(db_path)
        merged = {**stats, **rstats}
        source_for_render = md_d / f"{out_stem}.md"
        render_kw: dict = {}
        if len(md_paths) > 1:
            render_kw["merged_sources"] = sorted(md_paths)
        schema_md_path.write_text(
            render_schema_markdown(source_for_render, db_path, merged, **render_kw),
            encoding="utf-8",
        )
        by_src = rstats.get("by_source") or {}
        src_bits = ", ".join(f"{k}={v}" for k, v in sorted(by_src.items(), key=lambda kv: (-kv[1], kv[0]))[:6])
        em_only = max(0, rstats["ready_em"] - rstats["ready_labeled"])
        print(
            f"[INFO] dataset_resolved: {rstats['ready_labeled']}/{rstats['resolved_rows']} labeled-ready, "
            f"{rstats['ready_em']} with resolved EM ({em_only} EM-only), "
            f"path_source[{src_bits}]"
        )
        if sync_registry:
            _sync_registry_after_build(out_stem, db_path)

    return results


def main_catalog(argv: list[str] | None = None) -> None:
    args = parse_catalog_args(argv)
    run_catalog_build(
        md_dir=args.md_dir,
        db_dir=args.db_dir,
        schema_md_dir=args.schema_md_dir,
        sites=args.sites if args.sites else None,
        skip_s3_probe=args.skip_s3_probe,
        sync_registry=getattr(args, "sync_registry", False),
    )


if __name__ == "__main__":
    main_catalog()
