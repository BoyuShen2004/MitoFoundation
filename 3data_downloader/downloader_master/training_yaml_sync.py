"""After stage-4 preprocess, align Hydra training YAML with finetune_datalist + volume stats."""

from __future__ import annotations

import math
import os
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap

    _HAS_RUAMEL = True
except Exception:  # pragma: no cover
    YAML = None  # type: ignore[misc, assignment]
    CommentedMap = dict  # type: ignore[misc, assignment]
    _HAS_RUAMEL = False

try:
    import yaml as _pyyaml  # type: ignore
except Exception:  # pragma: no cover
    _pyyaml = None


def _config_path(project_root: Path) -> Path:
    env = (os.environ.get("MITO_TRAIN_CONFIG_YAML") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    config_dir = project_root / "5model_training" / "configuration"
    # Prefer the provider-agnostic name if it exists; fall back to the OpenOrganelle-specific
    # file that is present in the current repo layout.
    generic = config_dir / "mito_foundation.yaml"
    if generic.is_file():
        return generic.resolve()
    return (config_dir / "mito_openorganelle_foundation.yaml").resolve()


def _median_vec_float(rows: list[list[float]], *, default: float = 8.0) -> list[float]:
    if not rows:
        return [default, default, default]
    out: list[float] = []
    for i in range(3):
        col = [float(r[i]) for r in rows if isinstance(r, (list, tuple)) and len(r) > i]
        out.append(float(statistics.median(col)) if col else default)
    return [round(x, 6) for x in out]


def _median_vec_int(rows: list[list[int]]) -> list[int]:
    if not rows:
        return [112, 112, 112]
    out: list[int] = []
    for i in range(3):
        col = [int(r[i]) for r in rows if isinstance(r, (list, tuple)) and len(r) > i]
        out.append(int(statistics.median(col)) if col else 112)
    return out


def _quantize_patch_dims(med_zyx: list[int], *, lo: int = 32, hi: int = 192, step: int = 16) -> list[int]:
    """Round each median axis up to a ``step`` boundary (then clamp) so patches cover at least median extent."""
    out: list[int] = []
    for m in med_zyx[:3]:
        x = max(lo, min(hi, int(m)))
        x = int(math.ceil(float(x) / float(step)) * float(step))
        x = min(hi, max(lo, x))
        out.append(x)
    return out


def _entry_resolution(e: dict[str, Any]) -> list[float] | None:
    r = e.get("resolution")
    if isinstance(r, (list, tuple)) and len(r) >= 3:
        return [float(r[0]), float(r[1]), float(r[2])]
    return None


def _entry_shape(e: dict[str, Any]) -> list[int] | None:
    s = e.get("shape_zyx")
    if isinstance(s, (list, tuple)) and len(s) >= 3:
        return [int(s[0]), int(s[1]), int(s[2])]
    return None


def _clear_split_paths(split_cfg: Any) -> None:
    if not isinstance(split_cfg, Mapping):
        return
    split_cfg["image"] = None
    split_cfg["label"] = None
    split_cfg.pop("resolution", None)


def _set_split_paths_resolution(split_cfg: Any, entries: list[dict[str, Any]]) -> None:
    if not entries or not isinstance(split_cfg, Mapping):
        return
    imgs = [str(e["image"]) for e in entries if e.get("image")]
    lbls = [str(e["label"]) for e in entries if e.get("label")]
    res_rows = [_entry_resolution(e) for e in entries]
    res_rows = [r for r in res_rows if r is not None]
    res_med = _median_vec_float(res_rows, default=8.0)
    if len(imgs) == 1:
        split_cfg["image"] = imgs[0]
        split_cfg["label"] = lbls[0]
    else:
        split_cfg["image"] = imgs
        split_cfg["label"] = lbls
    split_cfg["resolution"] = res_med


def _patch_training_tree(
    tree: Any,
    *,
    training: list[dict[str, Any]],
    val_e: list[dict[str, Any]],
    test_e: list[dict[str, Any]],
    patch: list[int],
    result: dict[str, Any],
) -> None:
    new_map = CommentedMap if _HAS_RUAMEL else dict

    train_root = tree.setdefault("train", new_map())
    data_b = train_root.setdefault("data", new_map())
    train_split = data_b.setdefault("train", new_map())
    _set_split_paths_resolution(train_split, training)
    res_tr = train_split.get("resolution")
    if isinstance(res_tr, list) and len(res_tr) >= 3:
        result["resolution_train_zyx"] = [float(x) for x in res_tr[:3]]

    val_split = data_b.setdefault("val", new_map())
    if val_e:
        _set_split_paths_resolution(val_split, val_e)
    else:
        _clear_split_paths(val_split)

    test_outer = tree.get("test")
    if isinstance(test_outer, Mapping):
        tdata = test_outer.setdefault("data", new_map())
        test_split = tdata.setdefault("test", new_map())
        if test_e:
            _set_split_paths_resolution(test_split, test_e)
        else:
            _clear_split_paths(test_split)

    default = tree.setdefault("default", new_map())
    model = default.setdefault("model", new_map())
    model["input_size"] = list(patch)
    model["output_size"] = list(patch)
    dataloader = default.setdefault("data", new_map()).setdefault("dataloader", new_map())
    dataloader["patch_size"] = list(patch)
    inf = default.setdefault("inference", new_map())
    sw = inf.setdefault("sliding_window", new_map())
    sw["window_size"] = list(patch)


def sync_mito_openorganelle_foundation_yaml(
    project_root: Path,
    *,
    training: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    test: list[dict[str, Any]],
    config_path: Path | None = None,
) -> dict[str, Any]:
    """
    Update ``mito_openorganelle_foundation.yaml`` (or ``MITO_TRAIN_CONFIG_YAML``) so PyTC training
    points at training H5 paths and uses median ZYX spacing (nm) per split.

    Also refreshes ``default.model.input_size/output_size``, ``default.data.dataloader.patch_size``,
    and ``default.inference.sliding_window.window_size`` from median voxel ``shape_zyx`` (quantized).
    """
    result: dict[str, Any] = {"ok": False, "path": "", "message": "", "patch_size_zyx": [], "resolution_train_zyx": []}
    if not _HAS_RUAMEL and _pyyaml is None:
        result["message"] = "No YAML library (ruamel.yaml or PyYAML); skipped training YAML sync."
        return result

    path = (config_path or _config_path(project_root)).resolve()
    result["path"] = str(path)
    if not path.is_file():
        result["message"] = f"Training config not found: {path}"
        return result

    if not training:
        result["message"] = "No training entries; skipped training YAML sync."
        return result

    val_e = list(validation)
    test_e = list(test)

    shapes_all = [_entry_shape(e) for e in training + val_e + test_e]
    shapes_all = [s for s in shapes_all if s is not None]
    patch = _quantize_patch_dims(_median_vec_int(shapes_all) if shapes_all else [112, 112, 112])

    try:
        if _HAS_RUAMEL:
            assert YAML is not None
            yml = YAML()
            yml.preserve_quotes = True
            yml.default_flow_style = False
            yml.indent(mapping=2, sequence=4, offset=2)
            with open(path, encoding="utf-8") as f:
                tree = yml.load(f)
            if tree is None:
                tree = CommentedMap()
            _patch_training_tree(tree, training=training, val_e=val_e, test_e=test_e, patch=patch, result=result)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                yml.dump(tree, f)
            tmp.replace(path)
        else:
            assert _pyyaml is not None
            with open(path, encoding="utf-8") as f:
                tree = _pyyaml.safe_load(f)
            if tree is None or not isinstance(tree, dict):
                tree = {}
            _patch_training_tree(tree, training=training, val_e=val_e, test_e=test_e, patch=patch, result=result)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                _pyyaml.dump(
                    tree,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                    width=120,
                )
            tmp.replace(path)
    except Exception as exc:  # pragma: no cover
        result["message"] = f"YAML update failed: {exc}"
        return result

    result["ok"] = True
    result["patch_size_zyx"] = patch
    result["message"] = f"Updated training config: {path}"
    return result
