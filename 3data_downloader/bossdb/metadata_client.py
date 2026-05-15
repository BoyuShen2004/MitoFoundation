"""BossDB metadata API client: scrapes collection/experiment/channel catalog.

Endpoints used:
  GET  https://api.metadata.bossdb.org/api/latest/channels/schema
  POST https://api.metadata.bossdb.org/api/latest/channels/query

All network calls use stdlib urllib (no extra deps beyond the standard library).
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any

_LOG = logging.getLogger(__name__)

BOSSDB_META_BASE = "https://api.metadata.bossdb.org"
BOSSDB_DATA_HOST = "api.bossdb.io"

_SCHEMA_ENDPOINT = f"{BOSSDB_META_BASE}/api/latest/channels/schema"
_QUERY_ENDPOINT  = f"{BOSSDB_META_BASE}/api/latest/channels/query"

_DEFAULT_TIMEOUT_S = 30
_DEFAULT_RETRIES   = 3
_BACKOFF_S         = 2.0


# ── canonical dataset record ──────────────────────────────────────────────────

@dataclass
class BossDBDataset:
    """Normalized representation of one BossDB channel (collection/experiment/channel)."""

    source_site:   str             = "BossDB"
    project_id:    str             = ""   # collection/experiment/channel
    project_name:  str             = ""
    collection:    str             = ""
    experiment:    str             = ""
    channel:       str             = ""
    uri:           str             = ""   # bossdb://collection/experiment/channel
    channel_type:  str             = ""   # "image" | "annotation" | "unknown"
    data_type:     str             = ""   # "uint8", "uint64", …
    labeling:      str             = ""   # "mitochondria" | "annotation" | ""
    mito_label_quality: str        = ""   # "good" | "prediction_like" | "non_mito" | "n/a"
    good_mitochondria_gt_mask: str = "no"
    voxel_size_nm: list[float]     = field(default_factory=list)   # [z, y, x]
    organism:      str             = ""
    modality:      str             = ""
    tags:          list[str]       = field(default_factory=list)
    description:   str             = ""
    public:        bool            = True
    raw_json:      dict[str, Any]  = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BossDBDataset":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


# ── low-level HTTP helpers ─────────────────────────────────────────────────────

def _http_get(
    url: str,
    *,
    timeout: int = _DEFAULT_TIMEOUT_S,
    retries: int = _DEFAULT_RETRIES,
) -> Any:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, OSError) as exc:
            last_err = exc
            _LOG.debug("GET %s attempt %d failed: %s", url, attempt + 1, exc)
            if attempt < retries - 1:
                time.sleep(_BACKOFF_S * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last_err}")


def _http_post(
    url: str,
    body: dict[str, Any],
    *,
    timeout: int = _DEFAULT_TIMEOUT_S,
    retries: int = _DEFAULT_RETRIES,
) -> Any:
    payload = json.dumps(body).encode()
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept":       "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            # Client-side query errors (400s) should fail fast.
            detail = ""
            try:
                detail = exc.read().decode(errors="replace")[:1000]
            except Exception:
                pass
            if 400 <= exc.code < 500 and exc.code != 429:
                raise RuntimeError(f"POST {url} HTTP {exc.code}: {detail or exc.reason}") from exc
            last_err = RuntimeError(f"HTTP {exc.code}: {detail or exc.reason}")
            _LOG.debug("POST %s attempt %d failed: %s", url, attempt + 1, last_err)
            if attempt < retries - 1:
                time.sleep(_BACKOFF_S * (attempt + 1))
        except (urllib.error.URLError, OSError) as exc:
            last_err = exc
            _LOG.debug("POST %s attempt %d failed: %s", url, attempt + 1, exc)
            if attempt < retries - 1:
                time.sleep(_BACKOFF_S * (attempt + 1))
    raise RuntimeError(f"POST {url} failed after {retries} attempts: {last_err}")


# ── normalization helpers ──────────────────────────────────────────────────────

def _extract_voxel_size(raw: dict[str, Any]) -> list[float]:
    """Extract [z, y, x] voxel sizes in nm from a raw BossDB project record."""
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
                return [z * mult, y * mult, x * mult]
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
                    # BossDB API returns [x, y, z] order → convert to [z, y, x]
                    return [xyz[2], xyz[1], xyz[0]]
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


def _normalize_channel_type(raw_type: str) -> str:
    ct = (raw_type or "").lower().strip()
    if any(k in ct for k in ("annotation", "seg", "label")):
        return "annotation"
    if any(k in ct for k in ("image", "raw", "em")):
        return "image"
    return ct or "unknown"


def _infer_organism(tags: list[str]) -> str:
    species_keywords = [
        "mouse", "rat", "human", "zebrafish", "drosophila",
        "c. elegans", "fly", "macaque", "pig",
    ]
    for t in tags:
        tl = t.lower()
        for sp in species_keywords:
            if sp in tl:
                return sp
    return ""


def _infer_modality(tags: list[str]) -> str:
    # Check longer/more-specific keywords first to avoid "em" matching inside "fibsem".
    modality_keywords = sorted(
        ["em", "fibsem", "fib-sem", "sstem", "tem", "lm", "fluorescence", "xray", "mri"],
        key=len, reverse=True,
    )
    for t in tags:
        tl = t.lower()
        for mod in modality_keywords:
            if mod in tl:
                return mod
    return ""


def _normalize_project(raw: dict[str, Any]) -> BossDBDataset:
    """Normalize one raw BossDB API project record to a BossDBDataset."""
    attrs = raw.get("attributes") if isinstance(raw.get("attributes"), dict) else raw
    if not isinstance(attrs, dict):
        attrs = {}

    collection = experiment = channel = ""
    bid = str(attrs.get("ID") or "").strip()
    if bid.startswith("bossdb://"):
        parts = [p for p in bid[len("bossdb://"):].strip("/").split("/") if p]
        if len(parts) >= 3:
            collection, experiment, channel = parts[0], parts[1], parts[2]
    if not (collection and experiment and channel):
        boss_url = str(attrs.get("BossDBURI") or "").strip()
        m = re.search(r"/collection/([^/]+)/experiment/([^/]+)/channel/([^/?#]+)", boss_url)
        if m:
            collection, experiment, channel = m.group(1), m.group(2), m.group(3)
    if not (collection and experiment and channel):
        collection = str(attrs.get("collection") or attrs.get("Collection") or attrs.get("collection_name") or "").strip()
        experiment = str(attrs.get("experiment") or attrs.get("Experiment") or attrs.get("experiment_name") or "").strip()
        channel = str(attrs.get("channel") or attrs.get("Name") or attrs.get("channel_name") or "").strip()

    project_id = (
        f"{collection}/{experiment}/{channel}"
        if (collection and experiment and channel)
        else ""
    )
    uri = f"bossdb://{project_id}" if project_id else ""

    tags_raw = attrs.get("tags") or attrs.get("labels") or attrs.get("Keywords") or []
    if isinstance(tags_raw, str):
        tags_raw = [t.strip() for t in tags_raw.split(",") if t.strip()]
    tags = [str(t) for t in tags_raw if t]

    species_raw = attrs.get("Species")
    if isinstance(species_raw, list):
        species = [str(x.get("Name") if isinstance(x, dict) else x).strip() for x in species_raw]
        species = [s for s in species if s]
        species_txt = ", ".join(species)
    else:
        species_txt = ""

    organism = (str(attrs.get("organism") or attrs.get("species") or species_txt or "").strip()) or _infer_organism(tags)
    modality  = (str(attrs.get("modality") or attrs.get("imaging_modality") or "").strip())
    if not modality:
        mg = str(attrs.get("ImagingModalityGeneral") or "").strip()
        ms = str(attrs.get("ImagingModalitySpecific") or "").strip()
        modality = ", ".join([x for x in (mg, ms) if x and x not in ("None", "Other")])
    if not modality:
        modality = _infer_modality(tags)

    raw_type     = attrs.get("ChannelType") or attrs.get("type") or attrs.get("channel_type") or attrs.get("channelType") or ""
    channel_type = _normalize_channel_type(raw_type)
    data_type    = str(attrs.get("DataType") or attrs.get("datatype") or attrs.get("data_type") or attrs.get("dtype") or "").strip()

    labeling = ""
    if channel_type == "annotation":
        ch_l = channel.lower()
        if "mito" in ch_l or "mitochondria" in ch_l:
            labeling = "mitochondria"
        elif "nucleus" in ch_l or "nuclei" in ch_l:
            labeling = "nucleus"
        elif "er" == ch_l or "endoplasmic" in ch_l:
            labeling = "er"
        else:
            labeling = "annotation"

    project_name = (
        attrs.get("Name")
        or attrs.get("name")
        or attrs.get("project_name")
        or attrs.get("title")
        or project_id
        or ""
    ).strip()
    description = (str(attrs.get("Description") or attrs.get("description") or "")).strip()
    public = bool(attrs.get("Public", attrs.get("public", True)))

    voxel_size_nm = _extract_voxel_size(attrs)

    mito_label_quality = "n/a"
    good_mito_mask = "no"
    if channel_type == "annotation":
        if labeling == "mitochondria":
            n = channel.lower()
            pred_like = any(k in n for k in ("pred", "prediction", "inference", "unproofread", "empanada"))
            mito_label_quality = "prediction_like" if pred_like else "good"
            good_mito_mask = "yes" if not pred_like else "no"
        else:
            mito_label_quality = "non_mito"

    return BossDBDataset(
        source_site  = "BossDB",
        project_id   = project_id,
        project_name = project_name,
        collection   = collection,
        experiment   = experiment,
        channel      = channel,
        uri          = uri,
        channel_type = channel_type,
        data_type    = data_type,
        labeling     = labeling,
        mito_label_quality = mito_label_quality,
        good_mitochondria_gt_mask = good_mito_mask,
        voxel_size_nm = voxel_size_nm,
        organism     = organism,
        modality     = modality,
        tags         = tags,
        description  = description,
        public       = public,
        raw_json     = raw,
    )


# ── public API ─────────────────────────────────────────────────────────────────

def fetch_schema(*, timeout: int = _DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    """Fetch the BossDB projects schema (field definitions)."""
    return _http_get(_SCHEMA_ENDPOINT, timeout=timeout)


def discover(
    *,
    query: dict[str, Any] | None = None,
    timeout: int  = _DEFAULT_TIMEOUT_S,
    retries: int  = _DEFAULT_RETRIES,
    min_complete: bool = True,
) -> list[BossDBDataset]:
    """Discover all BossDB datasets via the metadata API.

    Args:
        query:        Optional POST body filter (e.g. ``{"tags": ["em"]}``).
        timeout:      Per-request timeout in seconds.
        retries:      Per-request retry count.
        min_complete: Skip records that lack collection/experiment/channel.

    Returns:
        Sorted list of normalised :class:`BossDBDataset` objects.
    """
    query_body = dict(query or {})
    all_datasets: list[BossDBDataset] = []
    seen: set[str] = set()
    if "tags" in query_body:
        _LOG.warning("BossDB /channels/query does not accept 'tags'; ignoring query['tags'].")
        query_body.pop("tags", None)

    resp = _http_post(_QUERY_ENDPOINT, query_body, timeout=timeout, retries=retries)
    items: list[dict[str, Any]] = []
    if isinstance(resp, list):
        items = [x for x in resp if isinstance(x, dict)]
    elif isinstance(resp, dict):
        for key in ("data", "results", "items", "channels", "entries"):
            got = resp.get(key)
            if isinstance(got, list):
                items = [x for x in got if isinstance(x, dict)]
                break
        if not items and (isinstance(resp.get("attributes"), dict) or "BossDBURI" in resp):
            items = [resp]

    for raw in items:
        ds = _normalize_project(raw)
        if min_complete and not ds.project_id:
            _LOG.debug("Skipping incomplete record: %s", raw)
            continue
        if ds.project_id in seen:
            continue
        seen.add(ds.project_id)
        all_datasets.append(ds)

    all_datasets.sort(key=lambda d: d.project_id)
    _LOG.info("BossDB discover: found %d unique channel records", len(all_datasets))
    return all_datasets
