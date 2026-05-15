"""Discover valid profile keys from pytorch_connectomics YAML libraries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

PROFILE_SOURCES: Tuple[Tuple[str, str, str], ...] = (
    ("pipeline_profiles", "profiles/pipeline_profiles.yaml", "pipeline_profiles"),
    ("system_profiles", "profiles/system_profiles.yaml", "system_profiles"),
    ("arch_profiles", "profiles/arch_profiles.yaml", "arch_profiles"),
    ("dataloader_profiles", "profiles/dataloader_profiles.yaml", "dataloader_profiles"),
    ("augmentation_profiles", "profiles/augmentation_profiles.yaml", "augmentation_profiles"),
    ("optimizer_profiles", "profiles/optimizer_profiles.yaml", "optimizer_profiles"),
)


def _profile_keys(connectomics_config_dir: Path, rel_path: str, root_key: str) -> List[str]:
    path = connectomics_config_dir / rel_path
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    block = data.get(root_key)
    if not isinstance(block, dict):
        return []
    return sorted(k for k in block if isinstance(k, str) and not k.startswith("_"))


def discover_options(connectomics_root: Path) -> Dict[str, Any]:
    cfg_dir = connectomics_root / "connectomics" / "config"
    out: Dict[str, Any] = {"profiles": {}}
    for resp_key, rel, ykey in PROFILE_SOURCES:
        out["profiles"][resp_key] = _profile_keys(cfg_dir, rel, ykey)
    out["tta_flip_axes"] = [
        {"value": "all", "label": "all (default)"},
        {"value": "none", "label": "none"},
        {"value": "[0]", "label": "flip Z only (JSON [0])"},
        {"value": "[1]", "label": "flip Y only"},
        {"value": "[2]", "label": "flip X only"},
        {"value": "[0,1]", "label": "Z+Y"},
        {"value": "[0,2]", "label": "Z+X"},
        {"value": "[1,2]", "label": "Y+X"},
    ]
    out["evaluation_metrics"] = [
        {"value": "dice", "label": "dice"},
        {"value": "jaccard", "label": "jaccard"},
        {"value": "accuracy", "label": "accuracy"},
    ]
    return out
