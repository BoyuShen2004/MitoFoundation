"""
s3_probe.py — Probe S3-compatible object stores (anonymous) to discover exact
download paths for biomedical image datasets.

Supports:
  - Janelia COSEM/OpenOrganelle  (s3://janelia-cosem-datasets)
  - Any dataset with an s3:// prefix that can be listed via unsigned GET on the
    bucket's XML API (AWS S3-compatible).

No AWS credentials needed — uses anonymous HTTP GET on the S3 REST endpoint.
"""

from __future__ import annotations

import re
import urllib.request
from typing import Optional
from xml.etree import ElementTree as ET

# ── constants ──────────────────────────────────────────────────────────────────
JANELIA_BUCKET = "janelia-cosem-datasets"
JANELIA_S3_BASE = f"https://{JANELIA_BUCKET}.s3.amazonaws.com"

# Scale preferences: prefer s1 for images (not full-res), s0 for segs
IMG_SCALE_PREFERENCE = ["s1", "s0", "s2"]
SEG_SCALE_PREFERENCE = ["s0", "s1"]

# Label sub-paths considered "mitochondria segmentation"
MITO_SEG_NAMES = [
    "mito_seg", "mito-mem_seg", "empanada-mito_seg", "mito_bag_seg",
    "mito", "ariadne/mito_seg",
]
# Label sub-paths considered "mitochondria predictions"
MITO_PRED_NAMES = [
    "mito_pred", "mito-mem_pred",
]
# Image sub-paths (relative to <dataset>.n5/em/ or similar)
EM_IMAGE_NAMES = [
    "fibsem-uint8", "fibsem-uint16", "fibsem-int16",
]


# ── low-level S3 listing ───────────────────────────────────────────────────────

def _s3_list(bucket_url: str, prefix: str, delimiter: str = "/") -> dict:
    """List one S3 prefix, returning {'common_prefixes': [...], 'keys': [...]}."""
    url = f"{bucket_url}/?prefix={prefix}&delimiter={delimiter}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            body = resp.read()
    except Exception as exc:
        return {"error": str(exc), "common_prefixes": [], "keys": []}

    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        return {"error": f"XML parse error: {exc}", "common_prefixes": [], "keys": []}

    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    common_prefixes = [
        cp.find("s3:Prefix", ns).text
        for cp in root.findall("s3:CommonPrefixes", ns)
        if cp.find("s3:Prefix", ns) is not None
    ]
    keys = [
        c.find("s3:Key", ns).text
        for c in root.findall("s3:Contents", ns)
        if c.find("s3:Key", ns) is not None
    ]
    return {"common_prefixes": common_prefixes, "keys": keys}


def _child_names(common_prefixes: list[str]) -> list[str]:
    """Extract the last path component from S3 common prefix strings."""
    out = []
    for p in common_prefixes:
        stripped = p.rstrip("/")
        out.append(stripped.split("/")[-1])
    return out


# ── Janelia COSEM prober ───────────────────────────────────────────────────────

def probe_janelia_dataset(dataset_id: str) -> dict:
    """
    Probe the Janelia COSEM S3 bucket for a given dataset slug.

    Returns a dict with keys:
      dataset_id      : the input slug
      n5_root         : s3:// path to the .n5 root (if found)
      zarr_root       : s3:// path to the .zarr root (if found)
      em_image_path   : best s3:// path for the EM image array (with scale)
      em_image_scale  : scale level chosen for EM image
      label_names     : list of all label sub-directories found
      mito_seg_path   : best s3:// path for mito segmentation (with scale)
      mito_pred_path  : best s3:// path for mito prediction (with scale)
      voxel_size_nm   : [x, y, z] from attributes.json if parseable, else None
      error           : error string if probing failed
    """
    result: dict = {
        "dataset_id": dataset_id,
        "n5_root": None,
        "zarr_root": None,
        "em_image_path": None,
        "em_image_scale": None,
        "label_names": [],
        "mito_seg_path": None,
        "mito_pred_path": None,
        "voxel_size_nm": None,
        "error": None,
    }

    base_prefix = f"{dataset_id}/"
    top = _s3_list(JANELIA_S3_BASE, base_prefix)
    if top.get("error"):
        result["error"] = top["error"]
        return result

    top_children = _child_names(top["common_prefixes"])

    # Detect .n5 / .zarr roots
    n5_name = f"{dataset_id}.n5"
    zarr_name = f"{dataset_id}.zarr"
    if n5_name in top_children:
        result["n5_root"] = f"s3://{JANELIA_BUCKET}/{dataset_id}/{n5_name}"
    if zarr_name in top_children:
        result["zarr_root"] = f"s3://{JANELIA_BUCKET}/{dataset_id}/{zarr_name}"

    # Probe EM image paths (prefer .n5 first, then .zarr)
    store_candidates = []
    if result["n5_root"]:
        store_candidates.append((result["n5_root"], f"{dataset_id}/{n5_name}"))
    if result["zarr_root"]:
        # zarr often has recon-1/em/fibsem-uint8
        store_candidates.append((result["zarr_root"], f"{dataset_id}/{zarr_name}"))

    for store_s3, store_prefix in store_candidates:
        # Try em/ subdirectory
        em_prefix = f"{store_prefix}/em/"
        em = _s3_list(JANELIA_S3_BASE, em_prefix)
        em_children = _child_names(em["common_prefixes"])

        for img_name in EM_IMAGE_NAMES:
            if img_name in em_children:
                # Check scales
                img_prefix = f"{store_prefix}/em/{img_name}/"
                scales = _child_names(_s3_list(JANELIA_S3_BASE, img_prefix)["common_prefixes"])
                for scale in IMG_SCALE_PREFERENCE:
                    if scale in scales:
                        result["em_image_path"] = f"{store_s3}/em/{img_name}/{scale}"
                        result["em_image_scale"] = scale
                        break
                if not result["em_image_path"]:
                    # No scales found, use the array root directly
                    result["em_image_path"] = f"{store_s3}/em/{img_name}"
                break

        # Also try zarr recon- path
        if not result["em_image_path"] and "zarr" in store_prefix:
            recon_listing = _s3_list(JANELIA_S3_BASE, f"{store_prefix}/")
            recon_children = _child_names(recon_listing["common_prefixes"])
            for child in recon_children:
                if child.startswith("recon-"):
                    recon_em_prefix = f"{store_prefix}/{child}/em/"
                    recon_em = _s3_list(JANELIA_S3_BASE, recon_em_prefix)
                    recon_em_children = _child_names(recon_em["common_prefixes"])
                    for img_name in EM_IMAGE_NAMES:
                        if img_name in recon_em_children:
                            img_prefix = f"{store_prefix}/{child}/em/{img_name}/"
                            scales = _child_names(_s3_list(JANELIA_S3_BASE, img_prefix)["common_prefixes"])
                            for scale in IMG_SCALE_PREFERENCE:
                                if scale in scales:
                                    result["em_image_path"] = f"{store_s3}/{child}/em/{img_name}/{scale}"
                                    result["em_image_scale"] = scale
                                    break
                            if not result["em_image_path"]:
                                result["em_image_path"] = f"{store_s3}/{child}/em/{img_name}"
                            break
                    if result["em_image_path"]:
                        break

        if result["em_image_path"]:
            break

    # Probe labels/ for segmentations (prefer .n5)
    for store_s3, store_prefix in store_candidates:
        labels_prefix = f"{store_prefix}/labels/"
        labels = _s3_list(JANELIA_S3_BASE, labels_prefix)
        label_children = _child_names(labels["common_prefixes"])

        if not label_children and "zarr" not in store_prefix:
            continue

        if label_children:
            result["label_names"] = label_children

        # Find best mito seg
        for seg_name in MITO_SEG_NAMES:
            if seg_name in label_children:
                seg_prefix = f"{store_prefix}/labels/{seg_name}/"
                scales = _child_names(_s3_list(JANELIA_S3_BASE, seg_prefix)["common_prefixes"])
                for scale in SEG_SCALE_PREFERENCE:
                    if scale in scales:
                        result["mito_seg_path"] = f"{store_s3}/labels/{seg_name}/{scale}"
                        break
                if not result["mito_seg_path"]:
                    result["mito_seg_path"] = f"{store_s3}/labels/{seg_name}"
                break

        # Find best mito pred
        for pred_name in MITO_PRED_NAMES:
            if pred_name in label_children:
                pred_prefix = f"{store_prefix}/labels/{pred_name}/"
                scales = _child_names(_s3_list(JANELIA_S3_BASE, pred_prefix)["common_prefixes"])
                for scale in SEG_SCALE_PREFERENCE:
                    if scale in scales:
                        result["mito_pred_path"] = f"{store_s3}/labels/{pred_name}/{scale}"
                        break
                if not result["mito_pred_path"]:
                    result["mito_pred_path"] = f"{store_s3}/labels/{pred_name}"
                break

        if result["mito_seg_path"]:
            break

        # Newer zarr datasets may have labels under recon-N/labels/
        if "zarr" in store_prefix and not result["mito_seg_path"]:
            recon_listing = _s3_list(JANELIA_S3_BASE, f"{store_prefix}/")
            recon_children = _child_names(recon_listing["common_prefixes"])
            for recon_child in recon_children:
                if not recon_child.startswith("recon-"):
                    continue
                recon_labels_prefix = f"{store_prefix}/{recon_child}/labels/"
                recon_labels = _s3_list(JANELIA_S3_BASE, recon_labels_prefix)
                recon_label_children = _child_names(recon_labels["common_prefixes"])
                if not recon_label_children:
                    continue

                # Add these to label_names for record-keeping
                result["label_names"] = list(set(result["label_names"]) | {f"{recon_child}/labels/{c}" for c in recon_label_children})

                # Direct mito seg under recon-N/labels/
                for seg_name in MITO_SEG_NAMES:
                    if seg_name in recon_label_children:
                        seg_prefix = f"{store_prefix}/{recon_child}/labels/{seg_name}/"
                        scales = _child_names(_s3_list(JANELIA_S3_BASE, seg_prefix)["common_prefixes"])
                        for scale in SEG_SCALE_PREFERENCE:
                            if scale in scales:
                                result["mito_seg_path"] = f"{store_s3}/{recon_child}/labels/{seg_name}/{scale}"
                                break
                        if not result["mito_seg_path"]:
                            result["mito_seg_path"] = f"{store_s3}/{recon_child}/labels/{seg_name}"
                        break

                # Look inside inference/ subdirectory
                if not result["mito_seg_path"] and "inference" in recon_label_children:
                    inf_prefix = f"{store_prefix}/{recon_child}/labels/inference/"
                    inf_listing = _s3_list(JANELIA_S3_BASE, inf_prefix)
                    inf_children = _child_names(inf_listing["common_prefixes"])

                    # Check inference/ direct children for mito seg names
                    for seg_name in MITO_SEG_NAMES:
                        if seg_name in inf_children:
                            seg_prefix = f"{store_prefix}/{recon_child}/labels/inference/{seg_name}/"
                            scales = _child_names(_s3_list(JANELIA_S3_BASE, seg_prefix)["common_prefixes"])
                            for scale in SEG_SCALE_PREFERENCE:
                                if scale in scales:
                                    result["mito_seg_path"] = f"{store_s3}/{recon_child}/labels/inference/{seg_name}/{scale}"
                                    break
                            if not result["mito_seg_path"]:
                                result["mito_seg_path"] = f"{store_s3}/{recon_child}/labels/inference/{seg_name}"
                            break

                    # Check inference/segmentations/ for mito seg names
                    if not result["mito_seg_path"] and "segmentations" in inf_children:
                        seg_listing = _s3_list(JANELIA_S3_BASE, f"{store_prefix}/{recon_child}/labels/inference/segmentations/")
                        seg_children = _child_names(seg_listing["common_prefixes"])
                        for seg_name in MITO_SEG_NAMES:
                            if seg_name in seg_children:
                                seg_prefix = f"{store_prefix}/{recon_child}/labels/inference/segmentations/{seg_name}/"
                                scales = _child_names(_s3_list(JANELIA_S3_BASE, seg_prefix)["common_prefixes"])
                                for scale in SEG_SCALE_PREFERENCE:
                                    if scale in scales:
                                        result["mito_seg_path"] = f"{store_s3}/{recon_child}/labels/inference/segmentations/{seg_name}/{scale}"
                                        break
                                if not result["mito_seg_path"]:
                                    result["mito_seg_path"] = f"{store_s3}/{recon_child}/labels/inference/segmentations/{seg_name}"
                                break

                if result["mito_seg_path"]:
                    break

        if result["mito_seg_path"]:
            break

    # Try to get voxel size from attributes.json
    result["voxel_size_nm"] = _probe_voxel_size(dataset_id, result)

    return result


def _probe_voxel_size(dataset_id: str, probe: dict) -> Optional[list]:
    """Try to read voxel size from the .n5 or .zarr attributes.json."""
    for store_s3, store_key in [
        (probe["n5_root"], f"{dataset_id}/{dataset_id}.n5"),
        (probe["zarr_root"], f"{dataset_id}/{dataset_id}.zarr"),
    ]:
        if not store_s3:
            continue
        # Check em/fibsem-uint8/attributes.json
        for img_name in EM_IMAGE_NAMES:
            attr_url = f"{JANELIA_S3_BASE}/{store_key}/em/{img_name}/attributes.json"
            try:
                with urllib.request.urlopen(attr_url, timeout=10) as resp:
                    import json
                    data = json.loads(resp.read())
                    # n5 style: {"pixelResolution": {"dimensions": [8,8,8], "unit": "nm"}}
                    pr = data.get("pixelResolution") or data.get("pixel_resolution")
                    if isinstance(pr, dict):
                        dims = pr.get("dimensions") or pr.get("resolution")
                        if isinstance(dims, list) and len(dims) >= 3:
                            return [float(d) for d in dims[:3]]
                    # zarr style: multiscales[0].datasets[0].coordinateTransformations
                    ms = data.get("multiscales")
                    if isinstance(ms, list) and ms:
                        xforms = (ms[0].get("datasets") or [{}])[0].get("coordinateTransformations", [])
                        for xf in xforms:
                            if xf.get("type") == "scale":
                                s = xf["scale"]
                                if isinstance(s, list) and len(s) >= 3:
                                    return [float(v) for v in s[-3:]]
            except Exception:
                pass
    return None


def probe_dataset_s3_paths(dataset_id: str, source: str = "openorganelle") -> dict:
    """
    Public entry point. Given a dataset ID and source name, probe S3 for exact paths.

    Returns the probe dict (see probe_janelia_dataset for schema).
    """
    if source.lower() in ("openorganelle", "janelia", "cosem", "cellmap"):
        return probe_janelia_dataset(dataset_id)
    return {
        "dataset_id": dataset_id,
        "error": f"Unknown source '{source}' — only 'openorganelle'/'janelia' supported currently.",
    }


def probe_result_to_download_config(probe: dict) -> Optional[dict]:
    """
    Convert a probe result dict into a download_cellmap.py-style DATASETS entry.

    Returns None if required paths are missing.
    """
    dataset_id = probe.get("dataset_id", "")
    img = probe.get("em_image_path")
    seg = probe.get("mito_seg_path") or probe.get("mito_pred_path")
    if not img or not seg:
        return None

    voxel = probe.get("voxel_size_nm") or [8.0, 8.0, 8.0]
    return {
        "img_path": img,
        "seg_path": seg,
        "voxel_size": voxel,
        "_source": f"s3_probe:{dataset_id}",
        "_label_names": probe.get("label_names", []),
    }


# ── CLI helper ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys

    dataset_ids = sys.argv[1:] if len(sys.argv) > 1 else ["aic_desmosome-3"]
    for did in dataset_ids:
        print(f"\n{'='*60}")
        print(f"Probing: {did}")
        result = probe_janelia_dataset(did)
        print(json.dumps(result, indent=2))
        cfg = probe_result_to_download_config(result)
        if cfg:
            print(f"\nDownload config entry:")
            print(f'  "{did}": {{')
            print(f'      "img_path": "{cfg["img_path"]}",')
            print(f'      "seg_path": "{cfg["seg_path"]}",')
            print(f'      "voxel_size": {cfg["voxel_size"]},')
            print(f'  }},')
