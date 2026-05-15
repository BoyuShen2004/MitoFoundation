"""BossDB inventory / probe parsing → image+annotation channel pairs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_inventory_path() -> Path:
    env = (os.environ.get("BOSSDB_INVENTORY_PATH") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / "3data_downloader" / "outputs" / "inventory_bossdb.jsonl"


def default_probe_path() -> Path:
    return repo_root() / "1web_scraper_01" / "outputs" / "BossDB.probe.json"


def pick_image_channel(channels: list[str]) -> str:
    priority = ["em", "raw", "image", "grayscale", "data"]
    for p in priority:
        for ch in channels:
            if p in ch.lower():
                return ch
    return sorted(channels)[0]


def pick_annotation_channel(channels: list[str]) -> str:
    priority = ["mito", "mitochondria", "seg", "annotation", "label"]
    for p in priority:
        for ch in channels:
            if p in ch.lower():
                return ch
    return sorted(channels)[0]


def load_bossdb_pairs(inventory_path: Path | None = None) -> list[dict[str, Any]]:
    """Load JSONL inventory (or fall back to probe JSON) and return EM+mito pair dicts."""
    inv = (inventory_path or default_inventory_path()).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    if inv.is_file():
        with inv.open(encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                try:
                    d = json.loads(s)
                except Exception:
                    continue
                if not isinstance(d, dict):
                    continue
                rows.append(d)
    if not rows:
        probe = default_probe_path()
        if probe.is_file():
            try:
                payload = json.loads(probe.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            ds = payload.get("datasets")
            if isinstance(ds, dict):
                for v in ds.values():
                    if isinstance(v, dict):
                        rows.append(v)
    by_exp: dict[str, dict[str, dict[str, Any]]] = {}
    for d in rows:
        coll = str(d.get("collection") or "").strip()
        exp = str(d.get("experiment") or "").strip()
        ch = str(d.get("channel") or "").strip()
        if not (coll and exp and ch):
            continue
        key = f"{coll}/{exp}"
        by_exp.setdefault(key, {})
        by_exp[key][ch] = d
    pairs: list[dict[str, Any]] = []
    for exp_key, channels in sorted(by_exp.items()):
        image_ch = [k for k, v in channels.items() if str(v.get("channel_type") or "").lower() == "image"]
        anno_ch = [k for k, v in channels.items() if str(v.get("channel_type") or "").lower() == "annotation"]
        if not image_ch or not anno_ch:
            continue
        img = pick_image_channel(image_ch)
        seg = pick_annotation_channel(anno_ch)
        img_d = channels[img]
        seg_d = channels[seg]
        voxel = img_d.get("voxel_size_nm") or seg_d.get("voxel_size_nm") or []
        pairs.append(
            {
                "project_id": exp_key,
                "img_channel": img,
                "seg_channel": seg,
                "img_uri": f"bossdb://{exp_key}/{img}",
                "seg_uri": f"bossdb://{exp_key}/{seg}",
                "voxel_size_nm": voxel,
                "organism": img_d.get("organism") or seg_d.get("organism") or "",
                "description": img_d.get("description") or seg_d.get("description") or "",
            }
        )
    return pairs
