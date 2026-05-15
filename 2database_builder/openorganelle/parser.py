"""Parser for per-dataset rows in scraped markdown appendix."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# Primary appendix rows use "- dataset_name=… | …". The scraper's fenced
# ``BEGIN_DATASET_ROWS_V1`` block repeats the same rows without the leading "- ".
DATASET_LINE_RE = re.compile(
    r"^\s*(?:-\s+)?dataset_name=(?P<name>[^|]+)\s*\|(?P<rest>.*)$"
)

# Prediction cues in URL or name that override any upstream role assignment.
# Keeps "Ground truths" UI group free of inference/prediction layers.
_PRED_URL_TOKENS = ("inference", "_pred", "_prediction")
_PRED_NAME_RE = re.compile(r"(^|[/_\-])pred([/_\-\.]|$)|_prediction|inference", re.IGNORECASE)


def normalize_layer_role(
    upstream_role: str,
    *,
    layer_name: str = "",
    layer_url: str = "",
    layer_content_type: str = "",
    layer_stage: str = "",
) -> str:
    """Return the semantically correct DB role, overriding upstream when path/content signals
    indicate inference or prediction.

    Policy (in priority order):
    1. content_type or stage contains "prediction" → ``predictions``
    2. URL contains inference / _pred / _prediction patterns → ``predictions``
    3. Layer name contains _pred / _prediction / inference patterns → ``predictions``
    4. Upstream role is a known DB role → keep it
    5. Fallback → ``raw_or_other_layers``
    """
    ct = (layer_content_type or "").lower()
    st = (layer_stage or "").lower()
    u = (layer_url or "").lower()
    n = (layer_name or "").lower()

    if "prediction" in ct or "prediction" in st:
        return "predictions"
    if any(tok in u for tok in _PRED_URL_TOKENS):
        return "predictions"
    if _PRED_NAME_RE.search(n):
        return "predictions"

    role = (upstream_role or "").lower()
    if role in ("ground_truths", "predictions", "raw_or_other_layers"):
        return role
    return "raw_or_other_layers"
CATALOG_ITEM_RE = re.compile(r"^\d+\.\s+\*\*`(?P<name>[^`]+)`\*\*\s*$")
TOKEN_RE = re.compile(r"(?P<kind>image|mesh):(?P<name>[^\[]+)(?P<meta>(?:\[[^\]]+\])*)$")
META_RE = re.compile(r"\[(?P<k>[a-zA-Z0-9_\-]+)=(?P<v>[^\]]+)\]|\[(?P<flag>semantic)\]")


@dataclass
class LayerRecord:
    layer_role: str
    layer_kind: str
    layer_name: str
    layer_format: str | None = None
    layer_stage: str | None = None
    layer_url: str | None = None  # from ``[url=…]`` in scraped layer token
    layer_content_type: str | None = None  # from ``[content_type=…]``
    semantic: bool = False
    raw_token: str = ""


@dataclass
class DatasetRecord:
    dataset_name: str
    stage: str | None = None
    segmentation_challenge: str | None = None
    description: str | None = None
    sample_organism: str | None = None
    sample_type: str | None = None
    sample_subtype: str | None = None
    sample_name: str | None = None
    grid_spacing: str | None = None
    grid_dimensions: str | None = None
    grid_axes: str | None = None
    grid_spacing_unit: str | None = None
    grid_dimensions_unit: str | None = None
    mitochondria_in_layer_names: bool = False
    source_url: str | None = None
    download_volume_url: str | None = None
    download_volume_image_stack: str | None = None
    download_mito_mask_url: str | None = None
    download_mito_mask_quality: str | None = None
    download_mito_mask_segmentation_kind: str | None = None
    download_mito_prediction_url: str | None = None
    layers: list[LayerRecord] = field(default_factory=list)


def _extract_source_url(line: str) -> str | None:
    m = re.search(r"PostgREST row source:\s*`([^`]+)`", line)
    return m.group(1).strip() if m else None


def _field_map(rest: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for piece in [x.strip() for x in rest.split("|")]:
        if "=" not in piece:
            continue
        k, v = piece.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _truthy_field(raw: str | None) -> bool:
    if not raw:
        return False
    s = raw.strip().lower()
    # Some rows append provenance markdown after value, e.g.
    # "yes _(PostgREST row source: ...)_"
    if s.startswith("yes") or s.startswith("true") or s.startswith("1"):
        return True
    return False


def _parse_layer_token(layer_role: str, token: str) -> LayerRecord | None:
    s = token.strip()
    m = TOKEN_RE.match(s)
    if not m:
        return None

    kind = m.group("kind")
    name = m.group("name").strip()
    meta = m.group("meta") or ""

    layer_format = None
    layer_stage = None
    layer_url = None
    layer_content_type = None
    semantic = False
    for mm in META_RE.finditer(meta):
        if mm.group("flag") == "semantic":
            semantic = True
            continue
        k = (mm.group("k") or "").strip().lower()
        v = (mm.group("v") or "").strip()
        if k == "format":
            layer_format = v
        elif k == "stage":
            layer_stage = v
        elif k == "url":
            layer_url = v
        elif k in ("content_type", "contenttype"):
            layer_content_type = v

    # Normalize role: path/content semantics take priority over upstream bucket assignment.
    effective_role = normalize_layer_role(
        layer_role,
        layer_name=name,
        layer_url=layer_url or "",
        layer_content_type=layer_content_type or "",
        layer_stage=layer_stage or "",
    )
    return LayerRecord(
        layer_role=effective_role,
        layer_kind=kind,
        layer_name=name,
        layer_format=layer_format,
        layer_stage=layer_stage,
        layer_url=layer_url,
        layer_content_type=layer_content_type,
        semantic=semantic,
        raw_token=s,
    )


def _catalog_slug_list(markdown_text: str) -> list[str]:
    """
    Parse numbered slug entries under:
      ### All datasets on the site (catalog slug list)
    stopping before the next markdown heading.
    """
    lines = markdown_text.splitlines()
    in_section = False
    out: list[str] = []
    seen: set[str] = set()
    for raw in lines:
        line = raw.strip()
        if line.startswith("### "):
            if "All datasets on the site (catalog slug list)" in line:
                in_section = True
                continue
            if in_section:
                break
        if not in_section:
            continue
        m = CATALOG_ITEM_RE.match(line)
        if not m:
            continue
        name = m.group("name").strip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def parse_markdown_text(markdown_text: str) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    by_name: dict[str, DatasetRecord] = {}
    for line in markdown_text.splitlines():
        m = DATASET_LINE_RE.match(line.strip())
        if not m:
            continue

        dataset_name = m.group("name").strip()
        rest = m.group("rest")
        fields = _field_map(rest)

        ds = DatasetRecord(
            dataset_name=dataset_name,
            stage=fields.get("stage"),
            segmentation_challenge=fields.get("segmentation_challenge"),
            description=fields.get("description"),
            sample_organism=fields.get("sample.organism"),
            sample_type=fields.get("sample.type"),
            sample_subtype=fields.get("sample.subtype"),
            sample_name=fields.get("sample.name"),
            grid_spacing=fields.get("grid.grid_spacing"),
            grid_dimensions=fields.get("grid.grid_dimensions"),
            grid_axes=fields.get("grid.grid_axes"),
            grid_spacing_unit=fields.get("grid.grid_spacing_unit"),
            grid_dimensions_unit=fields.get("grid.grid_dimensions_unit"),
            mitochondria_in_layer_names=_truthy_field(
                fields.get("mitochondria_in_layer_names")
            ),
            source_url=_extract_source_url(line),
            download_volume_url=fields.get("download_volume_url"),
            download_volume_image_stack=fields.get("download_volume_image_stack"),
            download_mito_mask_url=fields.get("download_mito_mask_url"),
            download_mito_mask_quality=fields.get("download_mito_mask_quality"),
            download_mito_mask_segmentation_kind=fields.get(
                "download_mito_mask_segmentation_kind"
            ),
            download_mito_prediction_url=fields.get("download_mito_prediction_url"),
        )

        for role in ("ground_truths", "predictions", "raw_or_other_layers"):
            raw = fields.get(role)
            if not raw:
                continue
            for tok in [x.strip() for x in raw.split(";") if x.strip()]:
                lr = _parse_layer_token(role, tok)
                if lr is not None:
                    ds.layers.append(lr)

        if ds.dataset_name in by_name:
            # Keep first parsed row; if duplicate appears later, ignore it.
            continue
        by_name[ds.dataset_name] = ds
        records.append(ds)

    # Ensure DB includes all catalog slugs even when some have no merged dataset row.
    for name in _catalog_slug_list(markdown_text):
        if name in by_name:
            continue
        ds = DatasetRecord(dataset_name=name)
        by_name[name] = ds
        records.append(ds)
    return records


def parse_markdown_file(path: Path) -> list[DatasetRecord]:
    return parse_markdown_text(path.read_text(encoding="utf-8"))

