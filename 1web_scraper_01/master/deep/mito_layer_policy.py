"""Mitochondria mask inclusion rules for OpenOrganelle-style catalogs and generic PostgREST embeds.

Layer *identity* is matched on name, description, storage URL, and ``image_stack`` strings.
That aligns with human-facing “label names” when the short API ``name`` is a crop id but the
URL path still contains ``mito_seg`` / ``Mitochondria``.

Structural placement follows scraped buckets (GT / prediction / raw) plus ``content_type``,
including challenge volumes stored as ``annotation`` + ``…/groundtruth/…`` URLs.
"""

from __future__ import annotations

import re

# Name / path cues (case-insensitive).
MITO_IDENTITY_RE = re.compile(r"mito|mitochondria|mitochondrion", re.IGNORECASE)


def mito_layer_identity_matches(
    *,
    name: str,
    desc: str,
    url: str = "",
    image_stack: str = "",
) -> bool:
    """True if this layer is plausibly a mitochondria raster mask (not only organelle keyword)."""
    blob = " ".join([name or "", desc or "", url or "", image_stack or ""]).strip()
    if MITO_IDENTITY_RE.search(blob):
        return True
    low = blob.lower()
    # OpenOrganelle: cristae / mito_seg paths without "mito" in the short API name.
    if "cristae_instance" in low:
        return True
    if "mito_seg" in low or "mito-mem" in low or "empanada-mito" in low:
        return True
    if "/labels/mitochondria" in low or "/mitochondria/" in low:
        return True
    return False


def classify_layer_gt_pred_raw(
    label: str,
    *,
    content_type: str = "",
    stage: str = "",
    url: str = "",
) -> str:
    """gt | pred | raw | other — must stay aligned with appendix buckets."""
    ct = (content_type or "").lower()
    st = (stage or "").lower()
    u = (url or "").lower()
    if "prediction" in ct or "prediction" in st:
        return "pred"
    # URL path signals override name-based GT heuristics: /labels/inference/ or _pred patterns
    # must never be classified as ground-truth even when the layer name contains "_seg".
    if "inference" in u or "_pred" in u or "_prediction" in u:
        return "pred"
    if ct == "annotation" and "groundtruth" in u:
        return "gt"
    s = f"{label} {ct} {st}".lower()
    if any(
        x in s
        for x in (
            "_seg",
            "segmentation",
            "ground truth",
            "ground_truth",
            "groundtruth",
            "manual",
            "proofread",
            "gt_",
            "_gt",
            "ariadne",
            "training",
            "dense annotation",
            "instance_seg",
            "instance seg",
            "semantic_seg",
            "annotat",
            "label",
            "mask",
            "mesh_gt",
        )
    ):
        return "gt"
    if any(x in s for x in ("_pred", "_prediction", "_predict", "predicted", "inference")):
        return "pred"
    if any(x in s for x in ("raw", "fibsem", "uint8", "uint16", "int16")) and "pred" not in s:
        return "raw"
    return "other"


def mito_layer_prediction_excluded(
    *,
    name: str,
    desc: str,
    url: str = "",
    content_type: str = "",
    stage: str = "",
    quality_hint: str = "",
) -> bool:
    """True → not usable as non-prediction mito training mask (case-insensitive patterns)."""
    parts = [name or "", desc or "", url or "", content_type or "", stage or "", quality_hint or ""]
    low = " ".join(parts).lower()
    if "pred_unproofread" in low:
        return True
    if "unproofread" in low:
        return True
    if "inference" in low:
        return True
    if "_pred" in low or "_prediction" in low or "_predict" in low:
        return True
    if re.search(r"(^|/|_)pred_", low):
        return True
    if "prediction" in (content_type or "").lower() or "prediction" in (stage or "").lower():
        return True
    return False


def mito_layer_structural_ok(
    cls: str,
    *,
    content_type: str,
    url: str,
) -> bool:
    """Matches: ground-truth bucket, segmentation-like layers, or challenge GT zarr crops."""
    if cls == "pred":
        return False
    if cls == "gt":
        return True
    ct = (content_type or "").lower()
    u = (url or "").lower()
    if "segmentation" in ct:
        return True
    if ct == "annotation" and "groundtruth" in u:
        return True
    if cls in ("raw", "other") and "/labels/" in u and "segmentation" in ct:
        return True
    return False


def mito_segmentation_kind(
    *,
    name: str,
    desc: str,
    url: str = "",
    segmentation_kind_field: str | None = None,
) -> str:
    """instance | semantic — defaults to semantic when not explicitly instance."""
    sk = (segmentation_kind_field or "").strip().lower()
    if sk == "instance":
        return "instance"
    if sk == "semantic":
        return "semantic"
    n = (name or "").lower()
    blobn = f"{n} {(desc or '').lower()} {(url or '').lower()}"
    if any(
        k in n
        for k in (
            "cristae_instance_seg",
            "instance_seg",
        )
    ):
        return "instance"
    if "instance" in n:
        return "instance"
    if "semantic" in n:
        return "semantic"
    if any(k in blobn for k in ("mito_seg", "mito-mem_seg", "empanada-mito_seg")):
        return "semantic"
    if "seg" in n and "instance" not in n:
        return "semantic"
    return "semantic"


def mito_label_training_status(
    *,
    name: str,
    desc: str,
    url: str,
    content_type: str,
    stage: str,
    image_stack: str,
    layer_cls: str,
) -> str:
    """
    Per-layer tag for appendix tokens and quality gating.

    Returns
    -------
    good
        Non-prediction mito mask in an allowed structural bucket.
    pred_unproofread
        Mito-related but fails prediction-exclusion (model / unproofread cues).
    non_ground_mito_layer
        Mito identity but wrong bucket / content (e.g. raw EM only).
    unrelated
        Not a mito layer.
    """
    if not mito_layer_identity_matches(
        name=name, desc=desc, url=url, image_stack=image_stack
    ):
        return "unrelated"
    if mito_layer_prediction_excluded(
        name=name,
        desc=desc,
        url=url,
        content_type=content_type,
        stage=stage,
        quality_hint="",
    ):
        return "pred_unproofread"
    if not mito_layer_structural_ok(layer_cls, content_type=content_type, url=url):
        return "non_ground_mito_layer"
    return "good"


_GT_MITO_MASK_NAME_RE = re.compile(
    r"(mito_seg|mito-mem_seg|empanada-mito|cristae_instance|mitochondria)",
    re.IGNORECASE,
)


def catalog_image_layer_name_indicates_gt_mito_mask(
    *,
    name: str,
    content_type: str,
) -> bool:
    """Fast path: API layer *name* clearly denotes a non-prediction mito mask (OpenOrganelle).

    Used to reconcile aggregate counts when a dataset has both ``mito_pred`` and ``mito_seg``-style
    layers — the catalog row should still count as having a good mito mask.
    """
    ct = (content_type or "").lower()
    if "prediction" in ct:
        return False
    return bool(_GT_MITO_MASK_NAME_RE.search(name or ""))


__all__ = [
    "MITO_IDENTITY_RE",
    "catalog_image_layer_name_indicates_gt_mito_mask",
    "classify_layer_gt_pred_raw",
    "mito_layer_identity_matches",
    "mito_layer_prediction_excluded",
    "mito_layer_structural_ok",
    "mito_label_training_status",
    "mito_segmentation_kind",
]
