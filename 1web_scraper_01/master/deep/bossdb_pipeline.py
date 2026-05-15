"""BossDB catalog scraper pipeline.

Uses the BossDB metadata REST API (no Playwright, no S3 probing) to enumerate
channels and write a probe JSON plus canonical ``BossDB.md`` report.

The metadata ``/projects/query`` endpoint only accepts **schema field filters**
(e.g. ``{"Public": true}``).  It rejects pagination keys such as ``page`` /
``page_size``, which previously caused HTTP 400.  This pipeline therefore calls
``/api/latest/channels/query`` with ``{}`` (plus optional filters) and unwraps
JSON:API ``{type, id, attributes}`` records into flat channel documents.

Entry point: ``BossDBScraper.scrape_and_report()`` — called by
``master/scraper_registry.py`` when ``website_name.lower().startswith("bossdb")``.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from .probe_registry import (
    annotate_new_entries,
    dataset_ids_from_probe,
    ensure_probe_shell,
    load_probe_json,
    next_scrape_batch_index,
    probe_json_path,
    save_probe_json,
    site_filename_safe,
)
from .tools import _save_markdown_to_path

_META_BASE   = "https://api.metadata.bossdb.org"
# Channel-level query returns one JSON:API resource per BossDB channel (3180+ rows
# as of 2026-04); the API returns the full ``data`` array in one response for ``{}``.
_QUERY_URL   = f"{_META_BASE}/api/latest/channels/query"
_SCHEMA_URL  = f"{_META_BASE}/api/latest/channels/schema"
_DEFAULT_TIMEOUT = 30
_DEFAULT_RETRIES = 3
_BACKOFF     = 2.0


# ── HTTP helpers (stdlib only — no extra deps) ────────────────────────────────

def _http_get(url: str, *, timeout: int, retries: int) -> Any:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, OSError) as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(_BACKOFF * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last}")


def _http_post(url: str, body: dict, *, timeout: int, retries: int) -> Any:
    payload = json.dumps(body).encode()
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode(errors="replace")[:4000]
            except Exception:
                detail = ""
            # Client errors are not helped by retries.
            if 400 <= exc.code < 500 and exc.code != 429:
                raise RuntimeError(
                    f"POST {url} HTTP {exc.code}: {detail or exc.reason}"
                ) from exc
            last = RuntimeError(f"POST {url} HTTP {exc.code}: {detail or exc.reason}")
            if attempt < retries - 1:
                time.sleep(_BACKOFF * (attempt + 1))
        except (urllib.error.URLError, OSError) as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(_BACKOFF * (attempt + 1))
    raise RuntimeError(f"POST {url} failed after {retries} attempts: {last}")


# ── normalisation ─────────────────────────────────────────────────────────────

def _normalise_channel_type(raw: str) -> str:
    t = (raw or "").lower().strip()
    if any(k in t for k in ("annotation", "seg", "label")):
        return "annotation"
    if any(k in t for k in ("image", "raw", "em")):
        return "image"
    return t or "unknown"


def _prediction_like_channel_name(channel: str) -> bool:
    """Heuristic aligned with OpenOrganelle: exclude prediction / unproofread style names."""
    n = (channel or "").lower()
    if not n.strip():
        return False
    banned = (
        "pred", "prediction", "inference", "unproofread", "empanada",
        "_pred", "mito_pred", "pred_mito",
    )
    return any(b in n for b in banned)


def _em_volume_like(entry: dict) -> bool:
    """True if this channel is plausibly a 3D EM / X-ray style volume (pairs with mito GT)."""
    if entry.get("channel_type") != "image":
        return False
    dt = str(entry.get("data_type") or "").strip().lower()
    if dt in ("imagery", "other", ""):
        return True
    mod = str(entry.get("modality") or "").lower()
    if any(
        k in mod
        for k in (
            "electron",
            "fib",
            "sem",
            "tem",
            "xnh",
            "x-ray",
            "microscopy",
        )
    ):
        return True
    ch = str(entry.get("channel") or "").lower()
    if ch in ("em", "sbem", "fibsem", "tem", "xnh") or "fibsem" in ch or ch.endswith("_em"):
        return True
    return False


def _mito_label_quality_for_entry(
    *,
    channel_type: str,
    labeling: str,
    channel: str,
) -> str:
    if channel_type == "image":
        return "n/a"
    if channel_type != "annotation":
        return "n/a"
    if labeling == "mitochondria":
        return "prediction_like" if _prediction_like_channel_name(channel) else "good"
    return "non_mito"


def _experiment_key(collection: str, experiment: str) -> str:
    return f"{collection}/{experiment}"


def _apply_mito_training_highlights(entries: dict[str, dict]) -> None:
    """Set OpenOrganelle-style triage flags using experiment-level EM + good GT mito pairing."""
    by_exp: dict[str, list[dict]] = {}
    for row in entries.values():
        if not isinstance(row, dict):
            continue
        ck = _experiment_key(str(row.get("collection") or ""), str(row.get("experiment") or ""))
        by_exp.setdefault(ck, []).append(row)

    for rows in by_exp.values():
        has_em = any(_em_volume_like(r) for r in rows)
        good_mito_rows = [
            r
            for r in rows
            if r.get("channel_type") == "annotation"
            and r.get("labeling") == "mitochondria"
            and r.get("mito_label_quality") == "good"
        ]
        highlight = has_em and bool(good_mito_rows)
        for r in rows:
            r["mito_training_experiment_highlight"] = "yes" if highlight else "no"
            if highlight and r in good_mito_rows:
                r["good_mitochondria_gt_mask"] = "yes"
            else:
                r["good_mitochondria_gt_mask"] = "no"


_MITO_TRIAGE_KEYS: frozenset[str] = frozenset(
    {
        "mito_label_quality",
        "mito_training_experiment_highlight",
        "good_mitochondria_gt_mask",
    }
)


def _merge_mito_triage_into_probe(datasets: dict[str, Any], live_rows: dict[str, dict]) -> None:
    """Refresh triage fields on existing probe rows when re-scraping the full catalog."""
    for pid, src in live_rows.items():
        if pid not in datasets or not isinstance(datasets[pid], dict):
            continue
        dst = datasets[pid]
        for k in _MITO_TRIAGE_KEYS:
            if k in src:
                dst[k] = src[k]


def _infer_labeling(channel: str, channel_type: str) -> str:
    if channel_type != "annotation":
        return ""
    ch = channel.lower()
    if "mito" in ch or "mitochondria" in ch:
        return "mitochondria"
    if "nucleus" in ch or "nuclei" in ch:
        return "nucleus"
    if ch == "er" or "endoplasmic" in ch:
        return "er"
    return "annotation"


def _extract_voxel_size(raw: dict) -> list[float]:
    """Return [z, y, x] voxel sizes in nm, or [] if not determinable."""
    ir = raw.get("ImageResolution")
    if isinstance(ir, dict):
        vs = ir.get("VoxelSize") or ir.get("voxelSize")
        unit = str(ir.get("VoxelUnit") or ir.get("voxelUnit") or "nm").strip().lower()
        if isinstance(vs, dict):
            try:
                x = float(vs.get("X") or vs.get("x") or 0)
                y = float(vs.get("Y") or vs.get("y") or 0)
                z = float(vs.get("Z") or vs.get("z") or 0)
            except (TypeError, ValueError):
                x = y = z = 0.0
            if x > 0 and y > 0 and z > 0:
                mult = 1000.0 if unit in ("um", "µm", "micrometer", "micrometers", "micron", "microns") else 1.0
                return [float(z) * mult, float(y) * mult, float(x) * mult]
    for key in ("voxel_size", "voxelSize", "resolution", "voxel_sizes"):
        val = raw.get(key)
        if isinstance(val, dict):
            x = float(val.get("x") or val.get("x_voxel_size") or 0)
            y = float(val.get("y") or val.get("y_voxel_size") or 0)
            z = float(val.get("z") or val.get("z_voxel_size") or 0)
            if x > 0 and y > 0 and z > 0:
                return [z, y, x]
        elif isinstance(val, (list, tuple)) and len(val) >= 3:
            try:
                xyz = [float(val[0]), float(val[1]), float(val[2])]
                if all(v > 0 for v in xyz):
                    return [xyz[2], xyz[1], xyz[0]]  # API returns x,y,z → store z,y,x
            except (TypeError, ValueError):
                pass
    cf = raw.get("coordinate_frame") or raw.get("coordinateFrame") or {}
    if isinstance(cf, dict):
        x = float(cf.get("x_voxel_size") or cf.get("xVoxelSize") or 0)
        y = float(cf.get("y_voxel_size") or cf.get("yVoxelSize") or 0)
        z = float(cf.get("z_voxel_size") or cf.get("zVoxelSize") or 0)
        if x > 0 and y > 0 and z > 0:
            return [z, y, x]
    return []


def _infer_organism(tags: list[str]) -> str:
    species = ["mouse", "rat", "human", "zebrafish", "drosophila", "c. elegans", "fly", "macaque", "pig"]
    for t in tags:
        tl = t.lower()
        for sp in species:
            if sp in tl:
                return sp
    return ""


def _infer_modality(tags: list[str]) -> str:
    # Longer/more-specific keywords checked first to avoid "em" matching inside "fibsem".
    mods = sorted(
        ["em", "fibsem", "fib-sem", "sstem", "tem", "lm", "fluorescence", "xray", "mri"],
        key=len, reverse=True,
    )
    for t in tags:
        tl = t.lower()
        for mod in mods:
            if mod in tl:
                return mod
    return ""


def _unwrap_jsonapi_resource(raw: dict) -> dict:
    """Flatten a JSON:API style ``{type, id, attributes}`` channel resource."""
    attrs = raw.get("attributes")
    if isinstance(attrs, dict):
        flat = dict(attrs)
        if raw.get("id") and "_metadata_document_id" not in flat:
            flat["_metadata_document_id"] = str(raw["id"])
        return flat
    return raw


def _parse_bossdb_triple(flat: dict) -> tuple[str, str, str] | None:
    """Resolve (collection, experiment, channel) from BossDB metadata fields."""
    bid = (flat.get("ID") or "").strip()
    if bid.startswith("bossdb://"):
        rest = bid[len("bossdb://") :].strip("/")
        parts = [p for p in rest.split("/") if p]
        if len(parts) >= 3:
            return parts[0], parts[1], parts[2]

    boss_url = (flat.get("BossDBURI") or "").strip()
    m = re.search(r"/collection/([^/]+)/experiment/([^/]+)/channel/([^/?#]+)", boss_url)
    if m:
        return m.group(1), m.group(2), m.group(3)

    exp_uri = (flat.get("Experiment") or "").strip()
    if exp_uri.startswith("bossdb://"):
        rest = exp_uri[len("bossdb://") :].strip("/")
        eparts = [p for p in rest.split("/") if p]
        ch_name = (flat.get("Name") or "").strip()
        if len(eparts) >= 2 and ch_name:
            return eparts[0], eparts[1], ch_name

    collection = (flat.get("collection") or flat.get("Collection") or flat.get("collection_name") or "").strip()
    experiment = (flat.get("experiment") or flat.get("Experiment") or flat.get("experiment_name") or "").strip()
    channel    = (flat.get("channel")    or flat.get("Name")        or flat.get("channel_name")    or "").strip()
    if collection and experiment and channel:
        return collection, experiment, channel
    return None


def _normalise_record(raw: dict) -> dict | None:
    """Normalise one API record into a probe-entry dict.  Returns None if incomplete."""
    if not isinstance(raw, dict):
        return None
    flat = _unwrap_jsonapi_resource(raw)
    triple = _parse_bossdb_triple(flat)
    if not triple:
        return None
    collection, experiment, channel = triple

    project_id = f"{collection}/{experiment}/{channel}"
    uri        = f"bossdb://{project_id}"

    tags_raw = flat.get("tags") or flat.get("labels") or flat.get("Keywords") or []
    if isinstance(tags_raw, str):
        tags_raw = [t.strip() for t in tags_raw.split(",") if t.strip()]
    elif not isinstance(tags_raw, list):
        tags_raw = []
    tags = [str(t) for t in tags_raw if t]

    species_list = flat.get("Species") or []
    organism_flat = ""
    if isinstance(species_list, list) and species_list:
        bits: list[str] = []
        for s in species_list:
            if isinstance(s, dict):
                name = (s.get("Name") or s.get("name") or "").strip()
                if name:
                    bits.append(name)
            elif s:
                bits.append(str(s))
        organism_flat = ", ".join(bits)

    organism = (
        (flat.get("organism") or flat.get("species") or organism_flat or "").strip()
        or _infer_organism(tags + [flat.get("Description") or "", flat.get("Name") or ""])
    )

    modality_bits: list[str] = []
    for key in ("ImagingModalityGeneral", "ImagingModalitySpecific"):
        v = flat.get(key)
        if isinstance(v, str):
            vv = v.strip()
            if vv and vv not in ("None", "Other"):
                modality_bits.append(vv)
    modality = ", ".join(modality_bits) or _infer_modality(tags)
    modality_general = str(flat.get("ImagingModalityGeneral") or "").strip()
    modality_specific = str(flat.get("ImagingModalitySpecific") or "").strip()

    raw_type = flat.get("ChannelType") or flat.get("type") or flat.get("channel_type") or flat.get("channelType") or ""
    ch_type = _normalise_channel_type(str(raw_type))
    data_type = str(
        flat.get("DataType")
        or flat.get("datatype")
        or flat.get("data_type")
        or flat.get("dtype")
        or ""
    ).strip()
    labeling = _infer_labeling(channel, ch_type)
    voxel = _extract_voxel_size(flat)
    public = bool(flat.get("Public", True))
    description = (flat.get("Description") or flat.get("description") or "").strip()
    mito_label_quality = _mito_label_quality_for_entry(
        channel_type=ch_type, labeling=labeling, channel=channel,
    )
    source_url = str(flat.get("BossDBURI") or "").strip()
    species_raw = flat.get("Species")
    if not isinstance(species_raw, list):
        species_raw = []
    contributor_names: list[str] = []
    contribs = flat.get("Contributors")
    if isinstance(contribs, list):
        for c in contribs:
            if not isinstance(c, dict):
                continue
            n = str(c.get("Name") or "").strip()
            if n:
                contributor_names.append(n)
    image_resolution = flat.get("ImageResolution")
    if not isinstance(image_resolution, dict):
        image_resolution = {}

    return {
        "project_id":    project_id,
        "collection":    collection,
        "experiment":    experiment,
        "channel":       channel,
        "uri":           uri,
        "channel_type":  ch_type,
        "data_type":     data_type,
        "labeling":      labeling,
        "voxel_size_nm": voxel,
        "organism":      organism,
        "modality":      modality,
        "tags":          tags,
        "description":   description,
        "public":        public,
        "error":         None,
        "source_url":    source_url,
        "metadata_doc_id": str(flat.get("_metadata_document_id") or flat.get("_id") or ""),
        "species_raw":   species_raw,
        "modality_general": modality_general,
        "modality_specific": modality_specific,
        "contributors": contributor_names,
        "image_resolution": image_resolution,
        "source_api": _QUERY_URL,
        # OpenOrganelle-style triage (final two set in ``_apply_mito_training_highlights``)
        "mito_label_quality": mito_label_quality,
        "mito_training_experiment_highlight": "no",
        "good_mitochondria_gt_mask": "no",
    }


# ── channels/query (single payload; no unsupported pagination keys) ───────────

def _channel_query_body() -> dict[str, Any]:
    """Build POST body for ``/channels/query`` — only **valid schema keys**."""
    extra: dict[str, Any] = {}
    raw_json = os.environ.get("BOSSDB_CHANNEL_QUERY_JSON", "").strip()
    if raw_json:
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"[ERROR] BOSSDB_CHANNEL_QUERY_JSON must be valid JSON object: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise SystemExit("[ERROR] BOSSDB_CHANNEL_QUERY_JSON must decode to a JSON object.")
        extra.update(parsed)

    legacy_tags = os.environ.get("BOSSDB_QUERY_TAGS", "").strip()
    if legacy_tags:
        print(
            "[WARN] BOSSDB_QUERY_TAGS is ignored — the metadata API does not accept a "
            "'tags' filter on /channels/query. Use BOSSDB_CHANNEL_QUERY_JSON with "
            "schema keys instead, e.g. {\"Public\": true, \"ChannelType\": \"Image\"}.",
            flush=True,
        )

    return dict(extra)


def _query_all_channel_records(
    *,
    body: dict[str, Any],
    timeout: int,
    retries: int,
) -> list[dict]:
    """POST ``/api/latest/channels/query``; return raw JSON:API ``data`` list."""
    resp = _http_post(_QUERY_URL, body, timeout=timeout, retries=retries)

    if isinstance(resp, list):
        return [x for x in resp if isinstance(x, dict)]

    items: list[dict] = []
    if isinstance(resp, dict):
        for key in ("data", "results", "items", "channels", "entries"):
            if isinstance(resp.get(key), list):
                items = resp[key]  # type: ignore[assignment]
                break
        if not items and ("attributes" in resp or "BossDBURI" in resp):
            items = [resp]
    return [x for x in items if isinstance(x, dict)]


# ── markdown generation ───────────────────────────────────────────────────────

def _mito_highlight_stats(full_catalog: dict[str, dict]) -> tuple[int, int, list[dict]]:
    """(n_highlight_experiments, n_good_gt_mito_channels, sorted_good_rows)."""
    exp_ok: set[str] = set()
    good_rows: list[dict] = []
    for row in full_catalog.values():
        if not isinstance(row, dict):
            continue
        if row.get("good_mitochondria_gt_mask") == "yes":
            good_rows.append(row)
            exp_ok.add(_experiment_key(str(row.get("collection") or ""), str(row.get("experiment") or "")))
    good_rows.sort(key=lambda r: str(r.get("project_id") or ""))
    return len(exp_ok), len(good_rows), good_rows


def _render_batch_markdown(
    *,
    site_name: str,
    batch_index: int,
    started_at_iso: str,
    new_entries: dict[str, dict],
    total_after: int,
    api_url: str,
    full_catalog: dict[str, dict],
) -> str:
    now_date = started_at_iso[:10]
    all_ids = sorted(full_catalog.keys())
    lines: list[str] = [
        f"# {site_name} — Mitochondria Key Information",
        "",
        f"**Source:** {api_url}",
        f"**Scraped:** {now_date}",
        f"**Scraped At (ISO):** {started_at_iso}",
        f"**Website Name:** {site_name}",
        "**Mitochondria Relevance:** Not explicitly stated (metadata API extraction + deterministic rules)",
        "",
        "## Data Metadata (Required)",
        f"- **Last Upload / Last Modification:** {now_date} (metadata API run date)",
        f"- **Data Volume:** {len(all_ids)} catalog channel entries",
        "- **Organelles in Data:** mitochondria plus other organelles (inferred from channel names / labels)",
        "- **Labeling Status:** Mixed (image + annotation channels)",
        "- **Label Evidence:** Deterministic parsing from BossDB metadata API fields.",
        "",
        "## Mitochondria-Specific Summary",
        "Deterministic extraction plus appendix rows below.",
        "",
        "## How to obtain mitochondria-related data (PRIMARY — engineer handoff)",
        "> Use Appendix reference workflow and canonical `dataset_name=` rows.",
        "",
        "### 1–6. (template)",
        "Follow Appendix for slug list, dataset rows, and BossDB URI fields.",
        "",
        "## Key Findings",
        f"- Source API: `{_QUERY_URL}`",
        f"- Catalog entries parsed: {len(all_ids)}",
        f"- New entries vs prior probe: {len(new_entries)}",
        f"- Probe batch index: {batch_index}",
        "",
        "## Databases & Datasets",
        "- See Appendix.",
        "",
    ]

    n_exp, n_gt, good_rows = _mito_highlight_stats(full_catalog)
    lines += [
        "## Quality, Gaps, and Future Work",
        "- **Data Quality Risks:** API metadata may omit some biology-specific context; verify key channels before download/training.",
        f"- **Useful Next Steps:** prioritize the {n_gt} good mitochondria GT channel(s) in highlighted experiments.",
        "",
        "## Additional Notes",
        f"BossDB batch index: {batch_index}; new entries this run: {len(new_entries)}; total in probe after write: {total_after}.",
        "",
        "## Appendix: Dataset catalog (per-dataset → schema)",
        "",
        "Each primary row is one BossDB channel mapped to the same `dataset_name=` schema-row contract used by OpenOrganelle.",
        "",
        f"- **Source:** `{api_url}`",
        f"- **Programmatic source:** `{_QUERY_URL}`",
        "",
        "### Mitochondria supervised-training highlights (3D EM + good GT mito)",
        "",
        "Aligned with OpenOrganelle appendix semantics: use rows with **`mito_label_quality=good`**, **`good_mitochondria_gt_mask=yes`**, and **`mito_training_experiment_highlight=yes`** (same **collection/experiment** has at least one **EM / imagery** channel plus a **non-prediction** mitochondria annotation). Exclude channels whose names suggest predictions (`pred`, `prediction`, `inference`, `unproofread`, `empanada`).",
        "",
        f"- **Experiments with EM + good GT mito (full catalog):** {n_exp}",
        f"- **Good GT mito mask channels (full catalog):** {n_gt}",
        "",
    ]
    if good_rows:
        lines.append("### Good GT mito channels (sample — full list in `BossDB.probe.json`)")
        lines.append("")
        for row in good_rows[:80]:
            pid = row.get("project_id", "")
            uri = row.get("uri", "")
            lines.append(f"- **`{pid}`** — `{uri}` — `mito_label_quality=good` `good_mitochondria_gt_mask=yes`")
        if len(good_rows) > 80:
            lines.append(f"- _…and {len(good_rows) - 80} more (see probe JSON)._")
        lines.append("")

    if new_entries:
        lines += [
            "### New channels this run (vs prior probe)",
            "",
            "| project_id | channel_type | data_type | mito_label_quality | highlight | good_mito_gt | organism | modality | voxel_size_nm | uri |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for pid, entry in sorted(new_entries.items()):
            vox = entry.get("voxel_size_nm") or []
            vox_s = ", ".join(f"{v:.1f}" for v in vox) if vox else "—"
            lines.append(
                f"| {pid} "
                f"| {entry.get('channel_type', '')} "
                f"| {entry.get('data_type', '')} "
                f"| {entry.get('mito_label_quality', '')} "
                f"| {entry.get('mito_training_experiment_highlight', '')} "
                f"| {entry.get('good_mitochondria_gt_mask', '')} "
                f"| {entry.get('organism', '')} "
                f"| {entry.get('modality', '')} "
                f"| {vox_s} "
                f"| {entry.get('uri', '')} |"
            )
        lines += [""]

    else:
        lines += ["### New channels this run (vs prior probe)", "", "_(none — all channels already in registry)_", ""]

    lines += [
        "### All datasets on the site (catalog slug list)",
        "",
        f"Ordered list of **{len(all_ids)}** BossDB channel slug(s).",
        "",
    ]
    for i, pid in enumerate(all_ids, start=1):
        lines.append(f"{i}. **`{pid}`**")
    lines += [
        "",
        "### Primary datasets (spatial + segmentation / annotation layers)",
        "",
        "Each row below is normalized for database builder ingestion, using fields analogous to OpenOrganelle appendix rows.",
        "",
    ]
    for pid, entry in sorted(full_catalog.items()):
        vox = entry.get("voxel_size_nm") or []
        spacing = f"[{', '.join(str(float(v)) for v in vox)}]" if vox else ""
        org = entry.get("organism") or ""
        desc = str(entry.get("description") or "").replace("|", "/")
        has_mito = "yes" if str(entry.get("labeling") or "") == "mitochondria" else "no"
        dl_vol = entry.get("uri") if str(entry.get("channel_type") or "") == "image" else ""
        dl_mask = entry.get("uri") if str(entry.get("mito_label_quality") or "") == "good" else ""
        dl_pred = entry.get("uri") if str(entry.get("mito_label_quality") or "") == "prediction_like" else ""
        line = (
            f"- dataset_name={pid}"
            f" | stage=prod"
            f" | description={desc}"
            f" | sample.organism={org}"
            f" | sample.type=bossdb"
            f" | grid.grid_spacing={spacing}"
            f" | grid.grid_axes=['z','y','x']"
            f" | grid.grid_spacing_unit=nm"
            f" | mitochondria_in_layer_names={has_mito}"
            f" | download_volume_url={dl_vol}"
            f" | download_mito_mask_url={dl_mask}"
            f" | download_mito_mask_quality={entry.get('mito_label_quality', '') if dl_mask else ''}"
            f" | download_mito_mask_segmentation_kind={'annotation' if dl_mask else ''}"
            f" | download_mito_prediction_url={dl_pred}"
        )
        lines.append(line)
    lines += [""]

    return "\n".join(lines)


# ── main scraper class ────────────────────────────────────────────────────────

class BossDBScraper:
    """Scrape BossDB via its metadata REST API.

    Implements the same ``scrape_and_report()`` interface as
    ``CatalogSitePipeline`` and ``GenericWebsiteScraper`` so it can be returned
    from ``master/scraper_registry.py:get_scraper()`` transparently.
    """

    def scrape_and_report(
        self,
        *,
        llm: Any,
        website_name: str,
        url: str,
        scraped_at_iso: str,
        spa_hosts: frozenset,
        max_pages: int,
        output_dir: Path,
    ) -> dict[str, dict]:
        timeout = int(os.environ.get("BOSSDB_API_TIMEOUT", str(_DEFAULT_TIMEOUT)))
        retries = int(os.environ.get("BOSSDB_API_RETRIES", str(_DEFAULT_RETRIES)))

        qbody = _channel_query_body()
        print(f"[BOSSDB] Querying metadata API: {_QUERY_URL}", flush=True)
        if qbody:
            print(f"[BOSSDB] Channel query filter: {qbody}", flush=True)

        try:
            raw_records = _query_all_channel_records(body=qbody, timeout=timeout, retries=retries)
        except Exception as exc:
            print(f"[ERROR] BossDB metadata API failed: {exc}", flush=True)
            raise

        print(f"[BOSSDB] Retrieved {len(raw_records)} raw channel records.", flush=True)

        # Normalise
        all_entries: dict[str, dict] = {}
        for raw in raw_records:
            if not isinstance(raw, dict):
                continue
            entry = _normalise_record(raw)
            if entry and entry["project_id"] not in all_entries:
                all_entries[entry["project_id"]] = entry

        print(f"[BOSSDB] Normalised {len(all_entries)} unique channels.", flush=True)

        _apply_mito_training_highlights(all_entries)
        n_exp, n_gt, _ = _mito_highlight_stats(all_entries)
        print(
            f"[BOSSDB] Mito training highlights: {n_exp} experiment(s) with EM+good GT mito, "
            f"{n_gt} good GT mito channel(s).",
            flush=True,
        )

        # Load existing probe JSON (incremental)
        site_safe   = site_filename_safe(website_name)   # "BossDB"
        probe_path  = probe_json_path(output_dir, website_name)
        probe_doc   = load_probe_json(probe_path)
        probe_doc   = ensure_probe_shell(probe_doc, site_name=website_name, site_url=url)

        existing_ids = dataset_ids_from_probe(probe_doc)
        new_ids      = [pid for pid in all_entries if pid not in existing_ids]

        print(
            f"[BOSSDB] {len(existing_ids)} already in probe; "
            f"{len(new_ids)} new channels to add.",
            flush=True,
        )

        # Next batch index and canonical markdown filename
        batch_index = next_scrape_batch_index(probe_doc)
        md_filename = f"{site_safe}.md"
        md_path     = output_dir / md_filename

        # Annotate new entries
        new_probe_results = {pid: all_entries[pid] for pid in new_ids}
        annotated = annotate_new_entries(
            new_probe_results,
            batch_index=batch_index,
            started_at_iso=scraped_at_iso,
            markdown_file=md_filename,
        )

        # Merge into probe doc
        probe_doc["datasets"].update(annotated)
        _merge_mito_triage_into_probe(probe_doc["datasets"], all_entries)
        probe_doc["scrape_batches"].append({
            "scrape_batch_index":  batch_index,
            "markdown_file":       md_filename,
            "started_at_iso":      scraped_at_iso,
            "ended_at_iso":        datetime.now().isoformat(timespec="seconds"),
            "live_catalog_source": "bossdb:metadata_api",
            "dataset_ids_added":   sorted(new_ids),
            "note":                f"BossDB metadata API scrape — {len(new_ids)} new channels",
        })
        probe_doc["last_run_at_iso"]      = scraped_at_iso
        probe_doc["last_catalog_source"]  = "bossdb:metadata_api"

        save_probe_json(probe_path, probe_doc)
        print(f"[BOSSDB] Probe JSON saved: {probe_path}", flush=True)

        # Write canonical markdown
        md_body = _render_batch_markdown(
            site_name=website_name,
            batch_index=batch_index,
            started_at_iso=scraped_at_iso,
            new_entries=annotated,
            total_after=len(probe_doc["datasets"]),
            api_url=url,
            full_catalog=all_entries,
        )
        _save_markdown_to_path(md_path, md_body)
        print(f"[BOSSDB] Canonical markdown: {md_path}", flush=True)

        return annotated
