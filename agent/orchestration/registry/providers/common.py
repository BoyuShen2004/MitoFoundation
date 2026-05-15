"""Shared utilities for provider adapters (path bootstrap, parsing, stable-id helpers)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    """Return repository root (``mitoFoundation2``)."""
    from config.paths import project_root

    return project_root()


def ensure_import_path(path: Path) -> None:
    """Prepend *path* to ``sys.path`` when missing."""
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)


def parse_zyx_triplet(value: str, *, cast: type[int] | type[float]) -> tuple[int, int, int] | tuple[float, float, float]:
    """Parse comma-separated ``z,y,x`` triplet into a typed tuple."""
    parts = [cast(x.strip()) for x in str(value).split(",")]
    if len(parts) < 3:
        raise ValueError(f"Expected 3 values for z,y,x, got: {value!r}")
    return (parts[0], parts[1], parts[2])


def parse_voxel_json(raw: Any) -> list[float] | None:
    """Parse JSON-encoded voxel list; return first 3 values as floats."""
    if not raw:
        return None
    try:
        vals = json.loads(str(raw))
        if isinstance(vals, list) and len(vals) >= 3:
            return [float(v) for v in vals[:3]]
    except Exception:
        return None
    return None


def stable_id_variants(stable_id: str):
    """Yield ID variants used to match crop-derived names."""
    key = str(stable_id or "").strip()
    if not key:
        return
    yield key
    m = re.match(r"^(.*?)(?:_vol(\d+)|_(\d+))$", key, flags=re.IGNORECASE)
    if m:
        base = (m.group(1) or "").strip()
        if base:
            yield base
