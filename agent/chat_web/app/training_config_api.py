"""API for editing PyTC / Hydra training YAML from the mitoFoundation2 web UI."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import APIRouter, HTTPException
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from .training_config_schema import discover_options

_get_project_root: Optional[Callable[[], Path]] = None

_ryaml = YAML()
_ryaml.preserve_quotes = True
_ryaml.default_flow_style = False
_ryaml.indent(mapping=2, sequence=4, offset=2)

router = APIRouter(prefix="/api/studio/training-yaml", tags=["training-yaml"])


def configure_training_yaml(*, get_project_root: Callable[[], Path]) -> None:
    global _get_project_root
    _get_project_root = get_project_root


def _root() -> Path:
    if _get_project_root is None:
        raise RuntimeError("training_config_api not configured")
    return _get_project_root()


def _default_config_path() -> Path:
    config_dir = _root() / "5model_training" / "configuration"
    # Prefer the provider-agnostic name if it exists; fall back to the OpenOrganelle-specific
    # file that is present in the current repo layout.
    generic = config_dir / "mito_foundation.yaml"
    if generic.is_file():
        return generic
    return config_dir / "mito_openorganelle_foundation.yaml"


def _paired_preprocessed_h5_names(
    project_root: Path,
) -> tuple[list[tuple[str, str, str]], dict[str, Any]]:
    """Return paired training NIfTI entries using nnUNet naming.

    Each item is ``(key, image_name, label_name)`` where key is the shared prefix.
    """
    from config.paths import nnunet_dataset_root

    ds_root = nnunet_dataset_root(project_root)
    img_dir = (ds_root / "imagesTr").resolve()
    lbl_dir = (ds_root / "labelsTr").resolve()
    meta: dict[str, Any] = {
        "images_dir": str(img_dir),
        "labels_dir": str(lbl_dir),
        "images_dir_exists": img_dir.is_dir(),
        "labels_dir_exists": lbl_dir.is_dir(),
    }
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        return [], meta
    image_map = {
        p.name[: -len("_0000.nii.gz")]: p.name
        for p in img_dir.glob("*_0000.nii.gz")
        if p.is_file() and p.name.endswith("_0000.nii.gz")
    }
    label_map = {
        p.name[: -len(".nii.gz")]: p.name
        for p in lbl_dir.glob("*.nii.gz")
        if p.is_file() and p.name.endswith(".nii.gz")
    }
    common_keys = sorted(set(image_map) & set(label_map))
    pairs = [(k, image_map[k], label_map[k]) for k in common_keys]

    meta["image_nifti_count"] = len(image_map)
    meta["label_nifti_count"] = len(label_map)
    meta["paired_count"] = len(pairs)
    meta["image_suffix"] = "_0000.nii.gz"
    meta["label_suffix"] = ".nii.gz"
    return pairs, meta


def _default_pytc_root() -> Path:
    return _root() / "5model_training" / "pytorch_connectomics"


def _config_path() -> Path:
    env = (os.environ.get("MITO_TRAIN_CONFIG_YAML") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return _default_config_path().resolve()


def _pytc_root() -> Path:
    env = (os.environ.get("MITO_PYTC_CONNECTOMICS") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return _default_pytc_root().resolve()


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, CommentedMap) or isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    return obj


def _is_mapping(x: Any) -> bool:
    return isinstance(x, (dict, CommentedMap))


def _deep_merge_inplace(dst: Any, src: Mapping) -> None:
    for k, v in src.items():
        if k in dst and _is_mapping(dst[k]) and _is_mapping(v):
            _deep_merge_inplace(dst[k], v)
        else:
            dst[k] = v


def _json_body_to_config(data: Any) -> dict:
    if not isinstance(data, dict):
        raise ValueError("body must be a JSON object")
    if "config" not in data:
        raise ValueError("missing 'config' key")
    cfg = data["config"]
    if not isinstance(cfg, dict):
        raise ValueError("'config' must be an object")
    return cfg


@router.get("/health")
def training_yaml_health() -> dict[str, Any]:
    return {"ok": True, "config_path": str(_config_path()), "pytc_root": str(_pytc_root())}


@router.get("/options")
def training_yaml_options() -> dict[str, Any]:
    return discover_options(_pytc_root())


@router.get("/config")
def training_yaml_get_config() -> dict[str, Any]:
    path = _config_path()
    if not path.is_file():
        raise HTTPException(404, f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        tree = _ryaml.load(f)
    return {"path": str(path), "config": _to_plain(tree)}


def _training_yaml_sync_preprocessed_volumes_impl() -> dict[str, Any]:
    """Register paired nnUNet training files into the training YAML.

    Updates only ``train.data.train`` image/label lists. Clears validation and test volume paths
    (optional splits). Writes ``train.data.train.resolution: [16, 16, 16]`` (nm, Z Y X); the same
    spacing is enforced at train time in ``mito_openorganelle_foundation_runtime``.
    """
    project_root = _root()
    pairs, scan = _paired_preprocessed_h5_names(project_root)
    if not pairs:
        raise HTTPException(
            400,
            "No paired imagesTr/*_0000.nii.gz and labelsTr/*.nii.gz files found with matching prefixes. "
            f"images_dir_exists={scan.get('images_dir_exists')} "
            f"labels_dir_exists={scan.get('labels_dir_exists')}",
        )

    path = _config_path()
    if not path.is_file():
        raise HTTPException(404, f"Config not found: {path}")

    with open(path, encoding="utf-8") as f:
        tree = _ryaml.load(f)
    if tree is None:
        tree = CommentedMap()

    train_root = tree.setdefault("train", CommentedMap())
    data_b = train_root.setdefault("data", CommentedMap())
    train_split = data_b.setdefault("train", CommentedMap())
    train_split["image"] = [image_name for _, image_name, _ in pairs]
    train_split["label"] = [label_name for _, _, label_name in pairs]
    train_split["resolution"] = [16.0, 16.0, 16.0]

    val_split = data_b.setdefault("val", CommentedMap())
    val_split["image"] = None
    val_split["label"] = None
    val_split.pop("resolution", None)

    test_outer = tree.setdefault("test", CommentedMap())
    tdata = test_outer.setdefault("data", CommentedMap())
    test_split = tdata.setdefault("test", CommentedMap())
    test_split["image"] = None
    test_split["label"] = None
    test_split.pop("resolution", None)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        _ryaml.dump(tree, f)
    tmp.replace(path)

    return {
        "ok": True,
        "path": str(path),
        "basenames": [k for k, _, _ in pairs],
        "pairs": [{"key": k, "image": im, "label": seg} for k, im, seg in pairs],
        "scan": scan,
        "config": _to_plain(tree),
    }


@router.api_route("/sync-preprocessed-volumes", methods=["POST", "PUT"])
def training_yaml_sync_preprocessed_volumes() -> dict[str, Any]:
    """POST or PUT — same behavior (PUT matches the working ``/config`` save path for some proxies)."""
    return _training_yaml_sync_preprocessed_volumes_impl()


@router.put("/config")
def training_yaml_put_config(body: dict[str, Any]) -> dict[str, Any]:
    """Merge JSON into the YAML file, or run disk sync when ``sync_preprocessed_volumes`` is true."""
    if not isinstance(body, dict):
        raise HTTPException(400, "body must be a JSON object")
    # Same URL/method as normal save — avoids StaticFiles 405 when a separate path fails to match.
    if body.get("sync_preprocessed_volumes") is True:
        return _training_yaml_sync_preprocessed_volumes_impl()

    path = _config_path()
    if not path.is_file():
        raise HTTPException(404, f"Config not found: {path}")
    try:
        incoming = _json_body_to_config(body)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    with open(path, encoding="utf-8") as f:
        tree = _ryaml.load(f)
    if tree is None:
        tree = CommentedMap()

    _deep_merge_inplace(tree, incoming)

    try:
        json.dumps(_to_plain(tree))
    except (TypeError, ValueError) as e:
        raise HTTPException(400, f"Config not JSON-serializable after merge: {e}") from e

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        _ryaml.dump(tree, f)
    tmp.replace(path)
    return {"saved": True, "path": str(path)}
