"""Central path configuration for mitoFoundation2.

Project-local paths are resolved relative to the repository root.
HPC / site-specific roots use environment variables with safe defaults.
"""

from __future__ import annotations

import os
from pathlib import Path

NNUNET_DATASET_NAME = "Dataset001_mito2"

# Relative paths (for CLI defaults, UI copy, and ``resolve_under_project``).
REL_DATA_RAW = "data/raw"
REL_NNUNET_RAW = "data/nnUNet_raw"
REL_NNUNET_PREPROCESSED = "data/nnUNet_preprocessed"
REL_NNUNET_RESULTS = "data/nnUNet_results"
REL_NNUNET_DATASET = f"{REL_NNUNET_RAW}/{NNUNET_DATASET_NAME}"
REL_OUTPUTS_BC = "data/outputs/bc"
REL_OUTPUTS_POSTPROCESSED = "data/outputs/postprocessed"
REL_NNUNET_LABELS_TS_INSTANCE = f"{REL_NNUNET_DATASET}/labelsTs-instance"


def project_root() -> Path:
    """Repository root (``MITO2_PROJECT_ROOT`` or inferred from this file)."""
    env = os.environ.get("MITO2_PROJECT_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent.parent


def resolve_under_project(raw: str, root: Path | None = None) -> Path:
    """Resolve *raw* relative to *root* (default: :func:`project_root`)."""
    base = (root or project_root()).resolve()
    p = Path((raw or "").strip()).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


def nnunet_raw_root(root: Path | None = None) -> Path:
    """nnUNet raw data root (``nnUNet_raw`` env or ``<project>/data/nnUNet_raw``)."""
    env = os.environ.get("nnUNet_raw", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (root or project_root()) / REL_NNUNET_RAW


def nnunet_preprocessed_root(root: Path | None = None) -> Path:
    """nnUNet preprocessed root (``nnUNet_preprocessed`` env or project default)."""
    env = os.environ.get("nnUNet_preprocessed", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (root or project_root()) / REL_NNUNET_PREPROCESSED


def nnunet_results_root(root: Path | None = None) -> Path:
    """nnUNet results root (``nnUNet_results`` env or project default)."""
    env = os.environ.get("nnUNet_results", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (root or project_root()) / REL_NNUNET_RESULTS


def nnunet_dataset_root(root: Path | None = None) -> Path:
    """Dataset001_mito2 directory under the nnUNet raw tree."""
    return (nnunet_raw_root(root) / NNUNET_DATASET_NAME).resolve()


def mitole_root() -> Path:
    """MitoLE HPC dataset root (``MITO2_MITOLE_ROOT``)."""
    default = "/projects/weilab/dataset/MitoLE"
    raw = os.environ.get("MITO2_MITOLE_ROOT", default).strip() or default
    return Path(raw).expanduser().resolve()


def mitole_sources_xlsx() -> Path:
    """Optional spacing lookup spreadsheet (``MITO2_MITOLE_SOURCES_XLSX``)."""
    default = "/projects/weilab/shenb/mito_trash/mito_data_sources/mitole_sources.xlsx"
    raw = os.environ.get("MITO2_MITOLE_SOURCES_XLSX", default).strip() or default
    return Path(raw).expanduser().resolve()


def data_raw(root: Path | None = None) -> Path:
    return resolve_under_project(REL_DATA_RAW, root)


def data_outputs_bc(root: Path | None = None) -> Path:
    return resolve_under_project(REL_OUTPUTS_BC, root)


def data_outputs_postprocessed(root: Path | None = None) -> Path:
    return resolve_under_project(REL_OUTPUTS_POSTPROCESSED, root)


def rel_nnunet_dataset() -> str:
    return REL_NNUNET_DATASET


def rel_nnunet_labels_ts_instance() -> str:
    return REL_NNUNET_LABELS_TS_INSTANCE


# Snippet injected into generated downloader scripts (runtime path resolution).
GENERATED_DOWNLOADER_PATHS_BLOCK = '''
_REPO_FOR_PATHS = Path(__file__).resolve().parents[2]
if str(_REPO_FOR_PATHS) not in sys.path:
    sys.path.insert(0, str(_REPO_FOR_PATHS))
from config.paths import nnunet_dataset_root
_DEFAULT_NNUNET_DATASET = str(nnunet_dataset_root(_REPO_FOR_PATHS))
DEFAULT_LABELED_BASE = _DEFAULT_NNUNET_DATASET
DEFAULT_INFERENCE_BASE = _DEFAULT_NNUNET_DATASET
'''.strip()
