"""Rule-based biological target classification for catalog rows.

This keeps ``sample_organism`` (species) separate from a user-facing
biological target used for filtering (organ/tissue/cell/subcellular context).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class BioClass:
    target: str
    target_type: str
    source: str
    confidence: float


_RULES: list[tuple[str, str, float, tuple[str, ...]]] = [
    ("kidney", "organ", 0.95, ("kidney", "renal", "nephron", "glomerul")),
    ("liver", "organ", 0.95, ("liver", "hepat")),
    ("airway", "tissue", 0.90, ("airway", "alveol", "bronchi")),
    ("choroid plexus", "tissue", 0.90, ("choroid plexus",)),
    ("desmosome", "subcellular_structure", 0.90, ("desmosome",)),
    ("t-cell", "cell_type", 0.85, ("t-cell", "killer t", "lymphocyte")),
    ("epithelial cell", "cell_type", 0.85, ("epithelial", "a431")),
    ("vero cell", "cell_type", 0.85, ("vero", "ccl81")),
    ("cos-7 cell", "cell_type", 0.85, ("cos-7", "cos7")),
]


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", s)


def classify_bio_target(
    *,
    dataset_name: str = "",
    sample_type: str = "",
    sample_subtype: str = "",
    sample_name: str = "",
    description: str = "",
    layer_names: list[str] | None = None,
) -> BioClass:
    """Return normalized biological target from textual hints.

    Priority order:
    1) dataset_name and sample subtype/name
    2) description
    3) layer names
    """
    layer_blob = " ".join(layer_names or [])
    fields: list[tuple[str, str]] = [
        ("dataset_name", _norm(dataset_name)),
        ("sample_subtype", _norm(sample_subtype)),
        ("sample_name", _norm(sample_name)),
        ("description", _norm(description)),
        ("layers", _norm(layer_blob)),
        ("sample_type", _norm(sample_type)),
    ]

    for target, target_type, score, tokens in _RULES:
        token_res = [re.compile(rf"\b{re.escape(tok)}\w*\b") for tok in tokens]
        for source, blob in fields:
            if not blob:
                continue
            if any(rx.search(blob) for rx in token_res):
                return BioClass(
                    target=target,
                    target_type=target_type,
                    source=source,
                    confidence=score,
                )

    return BioClass(
        target="unknown",
        target_type="unknown",
        source="fallback",
        confidence=0.0,
    )
