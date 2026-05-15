"""BossDB pairing helpers for stage-3 generated download scripts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure repo root is importable when run as a standalone script.
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from bossdb.metadata_client import BossDBDataset  # noqa: E402

# ── labeled pairing ───────────────────────────────────────────────────────────

def pair_labeled_datasets(
    datasets: list[BossDBDataset],
) -> list[dict[str, Any]]:
    """Group image+annotation channel pairs from the same (collection, experiment).

    Returns a list of dicts suitable for embedding into the generated download
    script.  Each dict has the keys expected by the BossDB scriptgen template:

        project_id      : "<collection>/<experiment>"
        collection      : str
        experiment      : str
        img_channel     : str
        seg_channel     : str
        img_uri         : "bossdb://collection/experiment/img_channel"
        seg_uri         : "bossdb://collection/experiment/seg_channel"
        voxel_size_nm   : [z, y, x] or []
        organism        : str
        modality        : str
        tags            : list[str]
    """
    # Group by (collection, experiment)
    by_exp: dict[str, dict[str, BossDBDataset]] = {}
    for ds in datasets:
        if not (ds.collection and ds.experiment and ds.channel):
            continue
        key = f"{ds.collection}/{ds.experiment}"
        by_exp.setdefault(key, {})
        by_exp[key][ds.channel] = ds

    pairs: list[dict[str, Any]] = []
    for exp_key, channels in sorted(by_exp.items()):
        image_chans = [
            ch for ch, ds in channels.items()
            if ds.channel_type == "image"
        ]
        anno_chans = [
            ch for ch, ds in channels.items()
            if ds.channel_type == "annotation"
        ]

        if not image_chans or not anno_chans:
            continue

        mito_anno_chans = [
            ch for ch, ds in channels.items()
            if ds.channel_type == "annotation" and (ds.labeling or "").strip().lower() == "mitochondria"
        ]
        # Stage-3 labeled downloads are mitochondria-focused; skip non-mito annotation pairs.
        if not mito_anno_chans:
            continue

        # Prefer the first image channel and the most relevant mitochondria annotation channel
        img_ch = _pick_image_channel(image_chans)
        seg_ch = _pick_annotation_channel(mito_anno_chans, channels)

        img_ds = channels[img_ch]
        seg_ds = channels[seg_ch]

        voxel = img_ds.voxel_size_nm or seg_ds.voxel_size_nm or []

        pairs.append({
            "project_id":    exp_key,
            "collection":    img_ds.collection,
            "experiment":    img_ds.experiment,
            "img_channel":   img_ch,
            "seg_channel":   seg_ch,
            "img_uri":       f"bossdb://{exp_key}/{img_ch}",
            "seg_uri":       f"bossdb://{exp_key}/{seg_ch}",
            "voxel_size_nm": voxel,
            "organism":      img_ds.organism or seg_ds.organism,
            "modality":      img_ds.modality or seg_ds.modality,
            "tags":          img_ds.tags or seg_ds.tags,
        })

    return pairs


def _pick_image_channel(channels: list[str]) -> str:
    """Prefer 'em', 'raw', or 'image' channel names; fall back to alphabetical first."""
    priority = ["em", "raw", "image", "grayscale", "data"]
    for p in priority:
        for ch in channels:
            if p in ch.lower():
                return ch
    return sorted(channels)[0]


def _pick_annotation_channel(channels: list[str], ds_map: dict[str, BossDBDataset]) -> str:
    """Prefer good mitochondria labels over prediction-like channels."""
    for ch in channels:
        ds = ds_map.get(ch)
        if ds and (ds.mito_label_quality or "").strip().lower() == "good":
            return ch
    priority = ["mito", "mitochondria", "seg", "annotation", "label"]
    for p in priority:
        for ch in channels:
            if p in ch.lower():
                return ch
    return sorted(channels)[0]
