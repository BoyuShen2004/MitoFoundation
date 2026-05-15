"""OpenOrganelle database builder package."""

from __future__ import annotations

import sys
from pathlib import Path

_inv = Path(__file__).resolve().parents[2] / "0inventory"
if _inv.is_dir() and str(_inv) not in sys.path:
    sys.path.insert(0, str(_inv))

from .builder import (
    build_batch,
    build_one_markdown,
    clear_output_artifacts,
    list_markdown_inputs,
)
from download_inventory import (
    emit_python_module,
    inventory_from_db,
)

__all__ = [
    "build_batch",
    "build_one_markdown",
    "clear_output_artifacts",
    "emit_python_module",
    "inventory_from_db",
    "list_markdown_inputs",
]

