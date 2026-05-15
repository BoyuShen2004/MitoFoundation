from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict

from .brave_augment import augment_scrape_with_brave
from .fetch import scrape_site_deep
from .from_website import batch_probe
from .grid_llm_normalize import apply_grid_normalization_to_access
from .llm_report import (
    generate_markdown_for_site,
)
from .probe_registry import (
    annotate_new_entries,
    dataset_ids_from_probe,
    ensure_probe_shell,
    known_ids_from_flat_legacy_markdown,
    known_ids_from_numbered_markdowns,
    load_probe_json,
    migrate_legacy_probe_without_batches,
    next_scrape_batch_index,
    probe_json_path,
    save_probe_json,
    site_filename_safe,
)
from .supabase_bulk import (
    fetch_dataset_rows_paginated,
    resolve_dataset_rest_url,
    sorted_unique_dataset_slugs,
)
from .tools import _save_markdown_to_path, brave_search_impl


def _is_openorganelle_site(website_name: str, url: str) -> bool:
    w = (website_name or "").lower()
    u = (url or "").lower()
    return "openorganelle" in w or "openorganelle.janelia.org" in u


def _openorganelle_fast_scrape_enabled() -> bool:
    return (os.getenv("OPENORGANELLE_FAST_SCRAPE", "1").strip().lower() not in ("0", "false", "no", "off"))


_OO_ENV_KEYS = (
    "PLAYWRIGHT_UI_SEGMENTATION_LAYERS",
    "PLAYWRIGHT_MAX_FOLLOW",
)


def _snapshot_openorganelle_env() -> dict[str, str | None]:
    return {k: os.environ.get(k) for k in _OO_ENV_KEYS}


def _restore_openorganelle_env(snap: dict[str, str | None]) -> None:
    for k, v in snap.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _apply_openorganelle_catalog_defaults(website_name: str, url: str) -> None:
    """OpenOrganelle: rule-based reports by default; optional fast browser profile.

    Uses ``os.environ.setdefault`` so values in ``.env`` / the shell still win.
    """
    if not _is_openorganelle_site(website_name, url):
        return
    if not _openorganelle_fast_scrape_enabled():
        return
    os.environ.setdefault("PLAYWRIGHT_UI_SEGMENTATION_LAYERS", "0")
    os.environ.setdefault("PLAYWRIGHT_MAX_FOLLOW", "0")


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def _openorganelle_scrape_skip_s3_probe(website_name: str, site_url: str) -> bool:
    """Skip stage-1 ``batch_probe`` during scrape for OpenOrganelle only.

    ``2database_builder/master/catalog_build.py`` re-probes Janelia S3 into SQLite by default
    (unless ``--skip-s3-probe``), so scrape-time probing is often duplicate wall time.

    Default behavior for OpenOrganelle is now to **skip** scrape-time S3 probing because
    ``2database_builder/master/catalog_build.py`` already probes S3 into SQLite by default.

    - Set ``OPENORGANELLE_SCRAPE_SKIP_S3_PROBE=0`` to force scrape-time S3 probing.
    - Set ``OPENORGANELLE_SCRAPE_SKIP_S3_PROBE=1`` to explicitly skip (same as default).
    """
    if not _is_openorganelle_site(website_name, site_url):
        return False
    raw = os.getenv("OPENORGANELLE_SCRAPE_SKIP_S3_PROBE", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    return True


def _openorganelle_skip_brave(website_name: str, site_url: str) -> bool:
    """Skip Brave augmentation for OpenOrganelle when ``OPENORGANELLE_SKIP_BRAVE=1``."""
    return _is_openorganelle_site(website_name, site_url) and _env_truthy(
        "OPENORGANELLE_SKIP_BRAVE"
    )


def _catalog_batch_probe(
    dataset_ids: list[str],
    *,
    website_name: str,
    site_url: str,
    delay_sec: float,
    verbose: bool,
) -> dict[str, dict]:
    if not dataset_ids:
        return {}
    if _openorganelle_scrape_skip_s3_probe(website_name, site_url):
        print(
            "[INFO] OpenOrganelle: skipping scrape-time S3 batch_probe "
            f"({len(dataset_ids)} dataset(s)). Run Stage 2 database build (default S3 probe) "
            "to fill `s3_probe_*` / `dataset_resolved` in SQLite.",
            flush=True,
        )
        return {
            did: {
                "error": "scrape_s3_probe_skipped",
                "note": "Run catalog_build S3 probe for paths; appendix PostgREST rows remain in .md",
            }
            for did in dataset_ids
        }
    return batch_probe(dataset_ids, delay_sec=delay_sec, verbose=verbose)


_JANELIA_DATASET_RE = re.compile(r"dataset_name=([a-z][a-z0-9_\-]+)", re.IGNORECASE)

_NUMBERED_SITE_MD_RE = re.compile(r"^(.+)_(\d+)\.md$", re.IGNORECASE)
_CANON_SUMMARY_START = "<!-- mito2_canonical_summary:start -->"
_CANON_SUMMARY_END = "<!-- mito2_canonical_summary:end -->"


def _strip_mito2_canonical_summary_block(text: str) -> str:
    """Remove a prior canonical wrapper so numbered reports do not nest summaries."""
    out = text
    while True:
        start = out.find(_CANON_SUMMARY_START)
        if start == -1:
            break
        end = out.find(_CANON_SUMMARY_END, start)
        if end == -1:
            out = out[:start].rstrip()
            break
        end += len(_CANON_SUMMARY_END)
        out = (out[:start] + out[end:]).strip()
    return out


def _ordered_numbered_markdown_paths(output_dir: Path, site_safe: str) -> list[Path]:
    """Return ``OpenOrganelle_01.md``, ``_02.md``, … sorted by numeric batch index."""
    found: list[tuple[int, Path]] = []
    for p in output_dir.iterdir():
        if not p.is_file():
            continue
        m = _NUMBERED_SITE_MD_RE.match(p.name)
        if not m:
            continue
        stem, num_s = m.group(1), m.group(2)
        if stem.lower() != site_safe.lower():
            continue
        found.append((int(num_s), p))
    found.sort(key=lambda t: t[0])
    return [p for _, p in found]


def _synthesize_canonical_body_from_numbered(
    site_safe: str, paths: list[Path]
) -> str:
    """Concatenate all run reports in batch order for downstream schema ingestion."""
    parts: list[str] = []
    for p in paths:
        try:
            raw = p.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"[WARN] Canonical synthesis: could not read {p}: {exc}")
            continue
        chunk = _strip_mito2_canonical_summary_block(raw).strip()
        if not chunk:
            continue
        m = _NUMBERED_SITE_MD_RE.match(p.name)
        idx = int(m.group(2)) if m else None
        label = f"batch {idx}" if idx is not None else p.name
        parts.append(f"## Run report {label} — `{p.name}`\n\n{chunk}")
    return "\n\n---\n\n".join(parts)


def _site_url_from_probe_or_default(
    probe_doc: dict | None, *, default_url: str | None
) -> str:
    if isinstance(probe_doc, dict):
        u = (probe_doc.get("url") or "").strip()
        if u:
            return u
    return (default_url or "").strip()


def extract_janelia_dataset_ids(markdown_text: str) -> list[str]:
    """Pull all dataset_name= values from a scraped markdown report."""
    seen = set()
    ids: list[str] = []
    for m in _JANELIA_DATASET_RE.finditer(markdown_text):
        did = m.group(1).strip()
        if did and did not in seen:
            seen.add(did)
            ids.append(did)
    return ids


def _live_dataset_ids(access: dict) -> tuple[list[str], str]:
    """Authoritative catalog from PostgREST when possible; else Playwright name list."""
    xhr = access.get("xhr_json_urls") or []
    apis = access.get("supabase_apikeys") or []
    auths = access.get("supabase_authorizations") or []
    rest_u, src = resolve_dataset_rest_url(xhr, [], apis)
    if rest_u:
        rows, meta = fetch_dataset_rows_paginated(rest_u, apis, auths)
        if rows and meta.get("ok"):
            names = sorted_unique_dataset_slugs(rows)
            if names:
                return names, f"postgrest:{src}"
        err = meta.get("error")
        print(
            f"[WARN] PostgREST catalog fetch failed ({err}); "
            "falling back to dataset_names_sorted from the browser scrape."
        )
    names = list(access.get("dataset_names_sorted") or [])
    return names, "dataset_names_sorted_fallback"


def _env_s3_probe_verbose() -> bool:
    """Per-dataset S3 probe lines on stderr. Default off (set ``S3_PROBE_VERBOSE=1`` to enable)."""
    return os.getenv("S3_PROBE_VERBOSE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _env_registry_use_numbered_md() -> bool:
    """When True, ``dataset_name=`` slugs in ``OpenOrganelle_NN.md`` count as known.

    Since reports now write to canonical ``OpenOrganelle.md`` only, this defaults to ``False``.
    Set ``OPENORGANELLE_REGISTRY_USE_NUMBERED_MD=1`` only for one-time migration windows where
    legacy ``OpenOrganelle_NN.md`` files still exist and should contribute to known IDs.
    """
    ev = os.getenv("OPENORGANELLE_REGISTRY_USE_NUMBERED_MD", "").strip().lower()
    if ev in ("0", "false", "no", "off"):
        return False
    if ev in ("1", "true", "yes", "on"):
        return True
    return False


def _next_report_markdown(
    output_dir: Path, site_safe: str
) -> tuple[str, Path]:
    """Return canonical ``(filename, path)`` for this run's primary report markdown."""
    name = f"{site_safe}.md"
    return name, output_dir / name


def _refresh_canonical_site_markdown(
    output_dir: Path,
    site_safe: str,
    *,
    website_name: str,
    site_url: str,
    latest_md_path: Path,
    probe_doc: dict | None = None,
) -> Path | None:
    """Refresh ``{site}.md`` as a synthesized view of numbered run files.

    Numbered reports (``{site}_NN.md``) remain the run-by-run history. The canonical file body is
    **always rebuilt** from those numbered files only (in numeric batch order); the previous
    ``{site}.md`` body is never used as input. Downstream database builds should read this file for
    one merged view of every scrape.
    """
    canonical = output_dir / f"{site_safe}.md"
    if latest_md_path.resolve() == canonical.resolve():
        return canonical

    numbered_paths = _ordered_numbered_markdown_paths(output_dir, site_safe)
    if not numbered_paths:
        if latest_md_path.is_file() and latest_md_path.resolve() != canonical.resolve():
            numbered_paths = [latest_md_path]
        else:
            return None

    numbered_reports: list[tuple[str, str]] = []
    for p in numbered_paths:
        try:
            numbered_reports.append((p.name, p.read_text(encoding="utf-8")))
        except OSError as exc:
            print(f"[WARN] Canonical synthesis: could not read {p}: {exc}")
    if not numbered_reports:
        return None

    detailed_merged_body = _synthesize_canonical_body_from_numbered(site_safe, numbered_paths)
    if not detailed_merged_body:
        return None
    body = detailed_merged_body

    batches = []
    if isinstance(probe_doc, dict):
        raw_batches = probe_doc.get("scrape_batches")
        if isinstance(raw_batches, list):
            batches = [b for b in raw_batches if isinstance(b, dict)]
    total_added = 0
    for b in batches:
        ds = b.get("dataset_ids_added")
        if isinstance(ds, list):
            total_added += len(ds)

    hist = []
    for b in batches[-5:]:
        bi = b.get("scrape_batch_index")
        mf = b.get("markdown_file") or "-"
        ds = b.get("dataset_ids_added")
        n = len(ds) if isinstance(ds, list) else 0
        hist.append(f"- batch {bi}: `{mf}` (+{n} dataset slugs)")
    history = "\n".join(hist) if hist else "- (no scrape_batches metadata)"

    merged_list = ", ".join(f"`{p.name}`" for p in numbered_paths)
    latest_note = (
        f"`{latest_md_path.name}`"
        if latest_md_path.is_file()
        else "(unknown)"
    )
    summary = (
        "<!-- mito2_canonical_summary:start -->\n"
        f"## Canonical report synthesis ({site_safe})\n\n"
        f"- Source run reports (`{site_safe}_NN.md`): **{len(numbered_paths)}**\n"
        "- Merge strategy: **Deterministic concatenation** of numbered run reports\n"
        f"- Bodies merged **in batch order** (previous `{site_safe}.md` is not an input): "
        f"{merged_list}\n"
        f"- Probe registry batches (`{site_safe}.probe.json`): **{len(batches)}**\n"
        f"- Total `dataset_ids_added` across batches: **{total_added}**\n"
        f"- Latest invocation wrote: {latest_note}\n\n"
        "### Recent batch history\n"
        f"{history}\n\n"
        "<!-- mito2_canonical_summary:end -->\n\n"
    )
    try:
        _save_markdown_to_path(canonical, summary + body)
    except OSError:
        return None
    return canonical


class CatalogSitePipeline:
    """Heavy catalog pipeline: Playwright/SPA capture, PostgREST/API signals, registry probe JSON, S3 probes.

    Each run writes a single canonical report file (``OpenOrganelle.md``).
    Report bodies are **heuristic-only** (``master/deep/llm_report.py``); grid normalization and registry diff stay non-LLM.

    Env flags such as ``OPENORGANELLE_*`` retain historical names but apply to any site using this pipeline.

    **OpenOrganelle — faster scrape (same downstream DB quality if you catalog-build with S3 probe):**

    - ``OPENORGANELLE_SCRAPE_SKIP_S3_PROBE=1`` — omit scrape-time ``batch_probe`` (large win). Probe JSON
      gets placeholder rows; run ``python -m master.catalog_build`` under ``2database_builder`` **without**
      ``--skip-s3-probe`` so SQLite gets real ``s3_probe_*`` / ``dataset_resolved`` (default build behavior).
    - ``OPENORGANELLE_SKIP_BRAVE=1`` — skip Brave Search augmentation when a Brave API key is set.
    """

    def scrape_and_report(
        self,
        *,
        llm,
        website_name: str,
        url: str,
        scraped_at_iso: str,
        spa_hosts: frozenset[str],
        max_pages: int,
        output_dir: Path,
    ) -> Dict[str, dict]:
        oo_env_snap = _snapshot_openorganelle_env()
        try:
            _apply_openorganelle_catalog_defaults(website_name, url)
            max_pages_effective = max_pages
            if _is_openorganelle_site(website_name, url) and _openorganelle_fast_scrape_enabled():
                cap = max(1, int(os.getenv("OPENORGANELLE_FAST_MAX_PAGES", "2")))
                max_pages_effective = min(max_pages, cap)

            prev_grid_heuristic = os.environ.get("INVENTORY_GRID_NORMALIZE_VOXELS")
            try:
                os.environ["INVENTORY_GRID_NORMALIZE_VOXELS"] = "0"
                scraped, access_signals = scrape_site_deep(
                    url,
                    website_name,
                    max_pages=max_pages_effective,
                    spa_hosts=spa_hosts,
                )
            finally:
                if prev_grid_heuristic is None:
                    os.environ.pop("INVENTORY_GRID_NORMALIZE_VOXELS", None)
                else:
                    os.environ["INVENTORY_GRID_NORMALIZE_VOXELS"] = prev_grid_heuristic

            if _openorganelle_skip_brave(website_name, url):
                print(
                    "[INFO] OPENORGANELLE_SKIP_BRAVE=1: skipping Brave Search augmentation.",
                    flush=True,
                )
            else:
                scraped = augment_scrape_with_brave(
                    scraped, website_name, url, access_signals, brave_search_impl
                )
            apply_grid_normalization_to_access(access_signals)

            return self._registry_mode_scrape_and_probe(
                llm=llm,
                website_name=website_name,
                url=url,
                scraped=scraped,
                scraped_at_iso=scraped_at_iso,
                access_signals=access_signals,
                output_dir=output_dir,
            )
        finally:
            _restore_openorganelle_env(oo_env_snap)

    def _registry_mode_scrape_and_probe(
        self,
        *,
        llm,
        website_name: str,
        url: str,
        scraped: str,
        scraped_at_iso: str,
        access_signals: dict,
        output_dir: Path,
    ) -> Dict[str, dict]:
        site_safe = site_filename_safe(website_name)
        probe_path = probe_json_path(output_dir, website_name)
        run_started = datetime.now().isoformat(timespec="seconds")

        live_ids, live_src = _live_dataset_ids(access_signals)
        live_ids = sorted(set(live_ids), key=str.lower)
        print(
            f"[INFO] Live catalog: {len(live_ids)} dataset slugs (source={live_src})"
        )

        probe_doc = ensure_probe_shell(
            load_probe_json(probe_path),
            site_name=website_name,
            site_url=url,
        )
        probe_doc, migrated_legacy = migrate_legacy_probe_without_batches(probe_doc)
        if migrated_legacy:
            save_probe_json(probe_path, probe_doc)
            print(
                "[INFO] Migrated legacy OpenOrganelle.probe.json: added scrape_batches[1] "
                "for pre-existing datasets."
            )
        known_from_probe = dataset_ids_from_probe(probe_doc)
        known_from_md = known_ids_from_numbered_markdowns(output_dir, site_safe)
        known_from_flat = known_ids_from_flat_legacy_markdown(output_dir, site_safe)
        known = known_from_probe | known_from_flat
        if _env_registry_use_numbered_md():
            known |= known_from_md

        new_ids = sorted(
            {x for x in live_ids if x not in known},
            key=str.lower,
        )
        print(
            f"[INFO] Registry: probe={len(known_from_probe)}, "
            f"numbered_md={len(known_from_md)} (merged={_env_registry_use_numbered_md()}), "
            f"flat={len(known_from_flat)} → known={len(known)}, "
            f"new vs catalog={len(new_ids)}"
        )

        if not live_ids:
            print(
                "[ERROR] No dataset slugs from the live catalog "
                "(PostgREST fetch may have failed). Keeping previous outputs unchanged."
            )
            probe_doc["last_catalog_source"] = live_src
            probe_doc["last_failed_empty_catalog_at_iso"] = datetime.now().isoformat(
                timespec="seconds"
            )
            save_probe_json(probe_path, probe_doc)
            return {}

        batch_index = next_scrape_batch_index(probe_doc)
        md_name, md_path = _next_report_markdown(output_dir, site_safe)

        markdown = generate_markdown_for_site(
            llm,
            website_name,
            url,
            scraped,
            scraped_at_iso,
            access_signals,
            database_builder_new_ids=new_ids,
        )
        _save_markdown_to_path(md_path, markdown)
        print(f"[INFO] Report markdown overwritten → {md_path}")

        delay = float(os.getenv("S3_PROBE_DELAY_SEC", "0.8"))
        max_probe = int(os.getenv("S3_PROBE_MAX_DATASETS", "500"))
        ids_to_probe = live_ids[:max_probe]

        results = _catalog_batch_probe(
            ids_to_probe,
            website_name=website_name,
            site_url=url,
            delay_sec=delay,
            verbose=_env_s3_probe_verbose(),
        )
        annotated = annotate_new_entries(
            results,
            batch_index=batch_index,
            started_at_iso=run_started,
            markdown_file=md_name,
        )

        probe_doc["url"] = url
        probe_doc["last_catalog_source"] = live_src
        probe_doc["last_run_at_iso"] = datetime.now().isoformat(timespec="seconds")
        latest_batch = {
            "scrape_batch_index": batch_index,
            "markdown_file": md_name,
            "started_at_iso": run_started,
            "ended_at_iso": datetime.now().isoformat(timespec="seconds"),
            "live_catalog_source": live_src,
            "dataset_ids_added": sorted(new_ids, key=str.lower),
            "note": (
                "registry mode: outputs overwritten from current live catalog; S3 probe skipped at scrape — run catalog_build"
                if _openorganelle_scrape_skip_s3_probe(website_name, url)
                else "registry mode: outputs overwritten from current live catalog"
            ),
        }
        # Overwrite with a single latest-run snapshot JSON each scrape.
        probe_doc["datasets"] = annotated
        probe_doc["scrape_batches"] = [latest_batch]
        save_probe_json(probe_path, probe_doc)
        probe_tail = (
            "S3 probe skipped at scrape — run catalog_build for paths"
            if _openorganelle_scrape_skip_s3_probe(website_name, url)
            else "S3 probe details in .md + JSON"
        )
        print(
            f"[INFO] Probe registry overwritten → {probe_path} "
            f"({len(annotated)} dataset(s); {probe_tail})"
        )
        return results

    def _run_s3_probe_pass(
        self,
        *,
        site_name: str,
        site_url: str,
        markdown_content: str,
        output_dir: Path,
    ) -> Dict[str, dict]:
        dataset_ids = extract_janelia_dataset_ids(markdown_content)
        if not dataset_ids:
            print(f"[INFO] S3 probe: no dataset IDs found in {site_name} markdown")
            return {}

        delay = float(os.getenv("S3_PROBE_DELAY_SEC", "0.8"))
        max_probe = int(os.getenv("S3_PROBE_MAX_DATASETS", "200"))
        ids_to_probe = dataset_ids[:max_probe]

        print(
            f"[INFO] S3 probe: {len(ids_to_probe)} dataset ID(s) from {site_name} "
            f"(quiet; set S3_PROBE_VERBOSE=1 for per-dataset lines)"
        )
        results = _catalog_batch_probe(
            ids_to_probe,
            website_name=site_name,
            site_url=site_url,
            delay_sec=delay,
            verbose=_env_s3_probe_verbose(),
        )

        safe_name = re.sub(r"[^\w\-]", "_", site_name)
        probe_path = output_dir / f"{safe_name}.probe.json"
        payload = {"site": site_name, "url": site_url, "datasets": results}
        probe_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[INFO] S3 probe: saved → {probe_path}")
        return results
