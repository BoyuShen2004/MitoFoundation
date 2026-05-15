"""Single-file probe registry for OpenOrganelle: merge S3 probes and scrape-batch metadata.

JSON has no comments; use ``scrape_batches`` plus per-dataset ``first_seen_*`` fields for audit trails.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

PROBE_SUFFIX = ".probe.json"


def site_filename_safe(site_name: str) -> str:
    return re.sub(r"[^\w\-]", "_", site_name.strip())


def probe_json_path(output_dir: Path, site_name: str) -> Path:
    return output_dir / f"{site_filename_safe(site_name)}{PROBE_SUFFIX}"


def _numbered_md_pattern(site_safe: str) -> re.Pattern[str]:
    # OpenOrganelle_01.md
    return re.compile(rf"^{re.escape(site_safe)}_(\d+)\.md$", re.IGNORECASE)


def max_existing_markdown_number(output_dir: Path, site_safe: str) -> int:
    """Return highest N for ``{site_safe}_N.md``, or 0 if none."""
    pat = _numbered_md_pattern(site_safe)
    best = 0
    for p in output_dir.glob(f"{site_safe}_*.md"):
        m = pat.match(p.name)
        if m:
            try:
                best = max(best, int(m.group(1)))
            except ValueError:
                continue
    return best


def next_markdown_number(output_dir: Path, site_safe: str) -> int:
    """Next index after the highest existing ``{site_safe}_NN.md`` (does not fill gaps)."""
    return max_existing_markdown_number(output_dir, site_safe) + 1


def next_free_markdown_index(output_dir: Path, site_safe: str) -> int:
    """Smallest positive *N* such that ``{site_safe}_{N:02d}.md`` is missing (fills gaps).

    Use this for new report filenames so deleting ``OpenOrganelle_02.md`` etc. frees that
    slot. Scrape batch indices in ``OpenOrganelle.probe.json`` stay monotonic separately.
    """
    pat = _numbered_md_pattern(site_safe)
    used: set[int] = set()
    for p in output_dir.glob(f"{site_safe}_*.md"):
        m = pat.match(p.name)
        if m:
            try:
                used.add(int(m.group(1)))
            except ValueError:
                continue
    n = 1
    while n in used:
        n += 1
    return n


def load_probe_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_probe_json(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=True), encoding="utf-8")


def dataset_ids_from_probe(doc: dict[str, Any]) -> set[str]:
    ds = doc.get("datasets")
    if not isinstance(ds, dict):
        return set()
    return {str(k) for k in ds.keys() if k}


def extract_dataset_ids_from_markdown(text: str) -> set[str]:
    """Same dataset_name= slug pattern as scraper appendix."""
    pat = re.compile(r"dataset_name=([a-z][a-z0-9_\-]+)", re.IGNORECASE)
    seen: set[str] = set()
    for m in pat.finditer(text):
        did = m.group(1).strip()
        if did:
            seen.add(did)
    return seen


def known_ids_from_numbered_markdowns(output_dir: Path, site_safe: str) -> set[str]:
    """Union of dataset_name= slugs from all ``{site_safe}_NN.md`` files."""
    out: set[str] = set()
    pat = _numbered_md_pattern(site_safe)
    for p in sorted(output_dir.glob(f"{site_safe}_*.md")):
        if not pat.match(p.name):
            continue
        try:
            out |= extract_dataset_ids_from_markdown(p.read_text(encoding="utf-8"))
        except OSError:
            continue
    return out


def known_ids_from_flat_legacy_markdown(output_dir: Path, site_safe: str) -> set[str]:
    """Slugs from legacy flat ``{site_safe}.md`` (pre-numbered reports), if present."""
    p = output_dir / f"{site_safe}.md"
    if not p.is_file():
        return set()
    try:
        return extract_dataset_ids_from_markdown(p.read_text(encoding="utf-8"))
    except OSError:
        return set()


def ensure_probe_shell(
    doc: dict[str, Any],
    *,
    site_name: str,
    site_url: str,
) -> dict[str, Any]:
    out = dict(doc)
    out.setdefault("site", site_name)
    out.setdefault("url", site_url)
    if not isinstance(out.get("datasets"), dict):
        out["datasets"] = {}
    if not isinstance(out.get("scrape_batches"), list):
        out["scrape_batches"] = []
    out.setdefault(
        "_registry_note",
        "JSON cannot have comments. Use scrape_batches (per run) and per-dataset "
        "first_seen_* fields to see which scrape added each dataset.",
    )
    return out


def migrate_legacy_probe_without_batches(doc: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """If datasets exist but scrape_batches was never written, add one legacy batch (index 1).

    Returns (doc, True) when the document was modified.
    """
    out = dict(doc)
    batches = out.get("scrape_batches")
    ds = out.get("datasets")
    if batches:
        return out, False
    if not isinstance(ds, dict) or not ds:
        out["scrape_batches"] = []
        return out, False
    out["scrape_batches"] = [
        {
            "scrape_batch_index": 1,
            "markdown_file": None,
            "started_at_iso": None,
            "ended_at_iso": None,
            "dataset_ids_added": sorted(ds.keys(), key=str.lower),
            "note": "legacy import: datasets already in probe JSON before scrape_batches tracking",
        }
    ]
    return out, True


def next_scrape_batch_index(doc: dict[str, Any]) -> int:
    """Monotonic index for the next run (1-based)."""
    batches = doc.get("scrape_batches")
    if not isinstance(batches, list) or not batches:
        return 1
    mx = 0
    for b in batches:
        if not isinstance(b, dict):
            continue
        try:
            mx = max(mx, int(b.get("scrape_batch_index") or 0))
        except (TypeError, ValueError):
            continue
    return mx + 1


def annotate_new_entries(
    probe_results: dict[str, dict],
    *,
    batch_index: int,
    started_at_iso: str,
    markdown_file: str,
) -> dict[str, dict]:
    """Copy probe results and add registry metadata on each new dataset entry."""
    out: dict[str, dict] = {}
    for did, row in probe_results.items():
        if not isinstance(row, dict):
            row = {}
        else:
            row = dict(row)
        row["first_seen_scrape_batch"] = batch_index
        row["first_seen_at_iso"] = started_at_iso
        row["first_seen_markdown_file"] = markdown_file
        out[did] = row
    return out
