"""Deterministic normalization of dataset inventory ``grid.*`` fields to voxel counts.

Uses :func:`dataset_inventory._compute_voxel_grid_heuristic` per inventory line.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INV_ROOT = _REPO_ROOT / "0inventory"
if str(_INV_ROOT) not in sys.path:
    sys.path.insert(0, str(_INV_ROOT))
from dataset_inventory import _compute_voxel_grid_heuristic


def _parse_literal_list(s: str) -> list[float] | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [float(x) for x in v]
    except (ValueError, SyntaxError, TypeError):
        pass
    return None


def _grid_fields_to_ia(fields: dict[str, str]) -> dict[str, Any]:
    return {
        "grid_spacing": _parse_literal_list(fields.get("grid.grid_spacing") or ""),
        "grid_spacing_unit": (fields.get("grid.grid_spacing_unit") or "nm").strip(),
        "grid_dimensions": _parse_literal_list(fields.get("grid.grid_dimensions") or ""),
        "grid_dimensions_unit": (fields.get("grid.grid_dimensions_unit") or "").strip(),
    }


def _extract_grid_fields(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in re.finditer(r"\b(grid\.grid_[a-z_]+)=([^|]*)", line):
        out[m.group(1)] = m.group(2).strip()
    return out


def _replace_grid_voxels_in_line(line: str, voxels: list[int], unit: str = "voxel") -> str:
    if "grid.grid_dimensions=" not in line:
        return line
    dims_s = "[" + ", ".join(str(int(x)) for x in voxels) + "]"
    out = re.sub(
        r"\bgrid\.grid_dimensions=\[[^\]]*\]",
        f"grid.grid_dimensions={dims_s}",
        line,
        count=1,
    )
    if re.search(r"\bgrid\.grid_dimensions_unit=", out):
        out = re.sub(
            r"\bgrid\.grid_dimensions_unit=\s*[^\s|]+",
            f"grid.grid_dimensions_unit={unit}",
            out,
            count=1,
        )
    else:
        ins = f" | grid.grid_dimensions_unit={unit}"
        m = re.search(r"\s+_\(", out)
        if m:
            out = out[: m.start()] + ins + out[m.start() :]
        else:
            out = out + ins
    return out


def _line_with_heuristic(line: str) -> str:
    fields = _extract_grid_fields(line)
    if not fields.get("grid.grid_dimensions"):
        return line
    ia = _grid_fields_to_ia(fields)
    voxels = _compute_voxel_grid_heuristic(ia)
    if not voxels:
        return line
    return _replace_grid_voxels_in_line(line, voxels, "voxel")


def normalize_inventory_lines(lines: list[str]) -> list[str]:
    """Rewrite inventory lines so ``grid.grid_dimensions`` are voxel counts where possible."""
    if not lines:
        return lines
    return [_line_with_heuristic(ln) for ln in lines]


def apply_grid_normalization_to_access(access: dict) -> None:
    """Mutate ``access['dataset_inventory']`` in place when present."""
    inv = access.get("dataset_inventory")
    if not isinstance(inv, list) or not inv:
        return
    str_lines = [str(x) for x in inv if isinstance(x, str)]
    if len(str_lines) != len(inv):
        return
    access["dataset_inventory"] = normalize_inventory_lines(str_lines)
