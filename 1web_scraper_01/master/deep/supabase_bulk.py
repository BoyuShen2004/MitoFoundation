"""Paginated Supabase/PostgREST fetches for full dataset catalogs (e.g. OpenOrganelle)."""

from __future__ import annotations

import base64
import json
import os
import re
from collections import Counter
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests

from .config import HTTP_HEADERS
from .mito_layer_policy import (
    catalog_image_layer_name_indicates_gt_mito_mask,
    classify_layer_gt_pred_raw,
    mito_label_training_status,
    mito_layer_identity_matches,
    mito_segmentation_kind,
)


def strip_limit_offset_query(url: str) -> str:
    """Remove limit/offset so we can page deterministically."""
    p = urlparse(url)
    pairs = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if k.lower() not in ("limit", "offset")]
    q = urlencode(pairs)
    return urlunparse((p.scheme, p.netloc, p.path, p.params, q, p.fragment))


# OpenOrganelle-style PostgREST: layers live in related ``image`` rows + nested ``mesh``, not ``dataset.images``.
# Include ``url`` + ``image_stack`` on ``image`` and ``url`` on ``mesh`` so scrapers can emit download-ready paths
# (see dataset_inventory). Other catalogs (e.g. BossDB) may not expose these — override with SUPABASE_DATASET_SELECT.
DEFAULT_DATASET_LAYER_EMBED_SELECT = (
    "name,description,stage,segmentation_challenge,"
    "image_acquisition(grid_spacing,grid_dimensions,grid_spacing_unit,grid_dimensions_unit,grid_axes),"
    "sample(type,subtype,organism,name),"
    "image(name,description,url,image_stack,format,stage,sample_type,content_type,"
    "mesh(name,description,url,format,stage))"
)


def rest_url_with_layer_embed(rest_url: str) -> str:
    """Force a ``select=`` that embeds ``image`` + ``mesh`` so layer names exist in each row.

    Override entirely with ``SUPABASE_DATASET_SELECT=...`` or disable embed with
    ``SUPABASE_DATASET_EMBED_LAYERS=0``.
    """
    if os.getenv("SUPABASE_DATASET_EMBED_LAYERS", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return strip_limit_offset_query(rest_url)
    base = strip_limit_offset_query(rest_url)
    custom = os.getenv("SUPABASE_DATASET_SELECT", "").strip()
    select_val = custom if custom else DEFAULT_DATASET_LAYER_EMBED_SELECT
    p = urlparse(base)
    pairs = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if k.lower() != "select"]
    pairs.append(("select", select_val))
    q = urlencode(pairs)
    return urlunparse((p.scheme, p.netloc, p.path, p.params, q, p.fragment))


def _dataset_image_entries(d: dict) -> list[dict]:
    """Flatten ``images`` / ``image`` list fields (Boss-style vs OpenOrganelle)."""
    out: list[dict] = []
    for k in ("images", "image"):
        v = d.get(k)
        if isinstance(v, list):
            for im in v:
                if isinstance(im, dict):
                    out.append(im)
    return out


def _image_mesh_entries(im: dict) -> list[dict]:
    m = im.get("meshes")
    if isinstance(m, list) and m:
        return [x for x in m if isinstance(x, dict)]
    m2 = im.get("mesh")
    if isinstance(m2, list):
        return [x for x in m2 if isinstance(x, dict)]
    return []


def dataset_has_mitochondria_layer_row(d: dict) -> bool:
    """True if any image/mesh entry suggests mitochondria (name, desc, url, image_stack)."""
    for im in _dataset_image_entries(d):
        iname = str(im.get("name") or "")
        idesc = str(im.get("description") or "")
        iurl = str(im.get("url") or "")
        istack = str(im.get("image_stack") or "")
        if mito_layer_identity_matches(
            name=iname, desc=idesc, url=iurl, image_stack=istack
        ):
            return True
        for mesh in _image_mesh_entries(im):
            mn = str(mesh.get("name") or "")
            md = str(mesh.get("description") or "")
            mu = str(mesh.get("url") or "")
            if mito_layer_identity_matches(name=mn, desc=md, url=mu, image_stack=""):
                return True
    return False


def find_dataset_rest_url(xhr_urls: list[str]) -> str | None:
    """First Supabase ``/rest/v1/dataset`` URL (any query)."""
    for u in xhr_urls or []:
        if not isinstance(u, str):
            continue
        ul = u.lower()
        if "supabase.co" in ul and "/rest/v1/dataset" in ul:
            return u.split("#")[0].strip()
    return None


def _supabase_ref_from_anon_jwt(apikey: str) -> str | None:
    """Decode Supabase anon JWT and return project ``ref`` (subdomain before .supabase.co)."""
    try:
        parts = (apikey or "").strip().split(".")
        if len(parts) < 2:
            return None
        payload_b64 = parts[1]
        pad = "=" * ((4 - len(payload_b64) % 4) % 4)
        raw = base64.urlsafe_b64decode(payload_b64 + pad)
        payload = json.loads(raw.decode("utf-8"))
        ref = payload.get("ref")
        if isinstance(ref, str) and ref.strip():
            return ref.strip()
    except Exception:
        return None
    return None


def resolve_dataset_rest_url(
    xhr_urls: list[str],
    supabase_auth_urls: list[str],
    apikeys: list[str],
) -> tuple[str | None, str]:
    """Find ``/rest/v1/dataset`` for bulk fetch: XHR list, then any ``/rest/v1/`` request, then JWT ``ref``."""
    u = find_dataset_rest_url(xhr_urls)
    if u:
        return u, "xhr_json_capture"

    for raw in supabase_auth_urls or []:
        if not isinstance(raw, str):
            continue
        if "supabase.co" not in raw.lower():
            continue
        p = urlparse(raw)
        pl = (p.path or "").lower()
        if "/rest/v1/" in pl:
            out = f"{p.scheme}://{p.netloc}/rest/v1/dataset"
            return out.split("#")[0], "supabase_request_path"

    for key in apikeys or []:
        ref = _supabase_ref_from_anon_jwt(key)
        if ref:
            return f"https://{ref}.supabase.co/rest/v1/dataset", "jwt_project_ref"

    return None, "none"


def _auth_headers(apikeys: list[str], authorizations: list[str]) -> dict[str, str] | None:
    key = (apikeys[0] if apikeys else "").strip()
    if not key:
        return None
    headers = {
        **HTTP_HEADERS,
        "apikey": key,
        "Accept": "application/json",
    }
    auth = (authorizations[0] if authorizations else "").strip()
    if auth.lower().startswith("bearer "):
        headers["Authorization"] = auth
    elif auth:
        headers["Authorization"] = f"Bearer {auth}"
    else:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def fetch_dataset_rows_paginated(
    rest_url: str,
    apikeys: list[str],
    authorizations: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """GET all rows from PostgREST ``dataset`` endpoint using limit/offset paging."""
    meta: dict[str, Any] = {"ok": False, "error": None, "requests": 0, "url_used": None}
    headers = _auth_headers(apikeys, authorizations)
    if not headers:
        meta["error"] = "no_apikey"
        return [], meta

    base = rest_url_with_layer_embed(rest_url)
    page_size = max(50, int(os.getenv("SUPABASE_DATASET_PAGE_SIZE", "200")))
    max_rows = max(page_size, int(os.getenv("SUPABASE_DATASET_MAX_ROWS", "5000")))
    timeout = float(os.getenv("SUPABASE_DATASET_FETCH_TIMEOUT", "90"))

    all_rows: list[dict[str, Any]] = []
    offset = 0
    meta["url_used"] = base[:220] + ("…" if len(base) > 220 else "")

    while len(all_rows) < max_rows:
        sep = "&" if "?" in base else "?"
        page_url = f"{base}{sep}limit={page_size}&offset={offset}"
        try:
            r = requests.get(page_url, headers=headers, timeout=timeout)
            meta["requests"] += 1
            r.raise_for_status()
            batch = r.json()
        except Exception as e:
            meta["error"] = str(e)[:500]
            meta["ok"] = len(all_rows) > 0
            return all_rows, meta

        if not isinstance(batch, list):
            meta["error"] = f"unexpected_json_type:{type(batch).__name__}"
            meta["ok"] = len(all_rows) > 0
            return all_rows, meta

        if not batch:
            break

        for item in batch:
            if isinstance(item, dict):
                all_rows.append(item)

        if len(batch) < page_size:
            break
        offset += page_size

    meta["ok"] = True
    meta["row_count"] = len(all_rows)
    return all_rows, meta


def parse_openorganelle_dataset_ui_totals(body_text: str) -> dict[str, Any]:
    """Parse ``Datasets 1 to 12 of 85`` style text from the live UI."""
    out: dict[str, Any] = {}
    if not body_text:
        return out
    m = re.search(
        r"Datasets\s+(\d+)\s+to\s+(\d+)\s+of\s+(\d+)",
        body_text,
        flags=re.IGNORECASE,
    )
    if m:
        out["ui_from"] = int(m.group(1))
        out["ui_to"] = int(m.group(2))
        out["ui_total"] = int(m.group(3))
        out["ui_range_label"] = m.group(0)
    return out


def sample_type_category_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Rough dataset categories from nested ``sample.type`` / ``subtype``."""
    c: Counter[str] = Counter()
    for d in rows:
        if not isinstance(d, dict):
            continue
        s = d.get("sample")
        if isinstance(s, dict):
            for k in ("type", "subtype", "organism"):
                v = s.get(k)
                if v is not None and str(v).strip():
                    c[f"{k}:{str(v).strip()[:80]}"] += 1
    return dict(c.most_common(80))


def spatial_summary_lines(rows: list[dict[str, Any]]) -> list[str]:
    """One line per dataset: exact name + voxel-ish grid fields from ``image_acquisition``."""
    lines: list[str] = []
    for d in rows:
        if not isinstance(d, dict):
            continue
        name = (d.get("name") or d.get("dataset_name") or "").strip()
        if not name:
            continue
        ia = d.get("image_acquisition")
        if isinstance(ia, list) and ia:
            ia = ia[0]
        if not isinstance(ia, dict):
            ia = {}
        parts = [f"dataset_name={name}"]
        for k in (
            "grid_dimensions",
            "grid_spacing",
            "grid_axes",
            "grid_spacing_unit",
            "grid_dimensions_unit",
        ):
            v = ia.get(k)
            if v is not None and str(v).strip():
                parts.append(f"{k}={v}")
        lines.append(" | ".join(parts))
    return sorted(set(lines), key=str.lower)


def names_from_inventory_lines(lines: list[str]) -> list[str]:
    """Recover dataset slugs from ``dataset_name=`` in inventory markdown lines."""
    names: list[str] = []
    for ln in lines or []:
        m = re.search(r"dataset_name=([^|]+)", str(ln))
        if m:
            names.append(m.group(1).strip())
    return sorted(set(names), key=str.lower)


def sorted_unique_dataset_slugs(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for d in rows:
        if not isinstance(d, dict):
            continue
        n = (d.get("name") or d.get("dataset_name") or "").strip()
        if n:
            names.add(n)
    return sorted(names, key=str.lower)


def aggregate_segmentation_annotation_schema(
    rows: list[dict[str, Any]],
    *,
    max_layer_names: int = 600,
) -> dict[str, Any]:
    """Cross-dataset aggregation for **schema design**: layer stats + mito mask roll-ups.

    Uses the same JSON shape as OpenOrganelle/BossDB-style ``dataset`` rows (``images``,
    ``meshes``, ``sample``, etc.). Safe to call with ``[]``.
    """
    gt_img: set[str] = set()
    pred_img: set[str] = set()
    raw_img: set[str] = set()
    gt_mesh: set[str] = set()
    pred_mesh: set[str] = set()
    img_formats: Counter[str] = Counter()
    mesh_formats: Counter[str] = Counter()
    n_with_any_image = 0
    n_datasets_mito_layers = 0
    mito_good_datasets: set[str] = set()
    mito_pred_only_datasets: set[str] = set()
    mito_instance_datasets: set[str] = set()
    mito_semantic_datasets: set[str] = set()
    mito_instance_urls: dict[str, str] = {}
    mito_semantic_urls: dict[str, str] = {}
    other_img: set[str] = set()
    other_mesh: set[str] = set()

    cap_img = 120
    cap_mesh = 80

    for d in rows:
        if not isinstance(d, dict):
            continue
        if dataset_has_mitochondria_layer_row(d):
            n_datasets_mito_layers += 1

        images = _dataset_image_entries(d)
        if images:
            n_with_any_image += 1
        for im in images[:cap_img]:
            if not isinstance(im, dict):
                continue
            iname = str(im.get("name") or "").strip()
            idesc = str(im.get("description") or "").strip()
            ifmt = str(im.get("format") or "").strip()
            ict = str(im.get("content_type") or "").strip()
            istage = str(im.get("stage") or "").strip()
            istack = str(im.get("image_stack") or "").strip()
            iurl = str(im.get("url") or "").strip()
            if ifmt:
                img_formats[ifmt] += 1
            cls = classify_layer_gt_pred_raw(
                f"{iname} {idesc}",
                content_type=ict,
                stage=istage,
                url=iurl,
            )
            if iname:
                if cls == "gt":
                    gt_img.add(iname)
                elif cls == "pred":
                    pred_img.add(iname)
                elif cls == "raw":
                    raw_img.add(iname)
                else:
                    other_img.add(iname)
            ds_name = str(d.get("name") or d.get("dataset_name") or "").strip()
            if ds_name:
                st = mito_label_training_status(
                    name=iname,
                    desc=idesc,
                    url=iurl,
                    content_type=ict,
                    stage=istage,
                    image_stack=istack,
                    layer_cls=cls,
                )
                sk_raw = im.get("segmentation_kind")
                sk = str(sk_raw).strip() if sk_raw is not None else None
                knd = mito_segmentation_kind(
                    name=iname,
                    desc=idesc,
                    url=iurl,
                    segmentation_kind_field=sk,
                )
                if st == "pred_unproofread":
                    mito_pred_only_datasets.add(ds_name)
                if st == "good":
                    mito_good_datasets.add(ds_name)
                    if knd == "instance":
                        mito_instance_datasets.add(ds_name)
                        if iurl and ds_name not in mito_instance_urls:
                            mito_instance_urls[ds_name] = iurl
                    elif knd == "semantic":
                        mito_semantic_datasets.add(ds_name)
                        if iurl and ds_name not in mito_semantic_urls:
                            mito_semantic_urls[ds_name] = iurl
            meshes = _image_mesh_entries(im)
            for mesh in meshes[:cap_mesh]:
                if not isinstance(mesh, dict):
                    continue
                mn = str(mesh.get("name") or "").strip()
                mf = str(mesh.get("format") or "").strip()
                mst = str(mesh.get("stage") or "").strip()
                if mf:
                    mesh_formats[mf] += 1
                mdsc = str(mesh.get("description") or "")
                mu = str(mesh.get("url") or "").strip()
                mcls = classify_layer_gt_pred_raw(
                    f"{mn} {mdsc}",
                    content_type="",
                    stage=mst,
                    url=mu,
                )
                if not mn:
                    continue
                if mcls == "gt":
                    gt_mesh.add(mn)
                elif mcls == "pred":
                    pred_mesh.add(mn)
                elif mcls == "raw":
                    pass
                else:
                    other_mesh.add(mn)
                ds_name = str(d.get("name") or d.get("dataset_name") or "").strip()
                if ds_name:
                    st = mito_label_training_status(
                        name=mn,
                        desc=mdsc,
                        url=mu,
                        content_type="",
                        stage=mst,
                        image_stack="",
                        layer_cls=mcls,
                    )
                    sk_raw = mesh.get("segmentation_kind")
                    sk = str(sk_raw).strip() if sk_raw is not None else None
                    knd = mito_segmentation_kind(
                        name=mn,
                        desc=mdsc,
                        url=mu,
                        segmentation_kind_field=sk,
                    )
                    if st == "pred_unproofread":
                        mito_pred_only_datasets.add(ds_name)
                    if st == "good":
                        mito_good_datasets.add(ds_name)
                        if knd == "instance":
                            mito_instance_datasets.add(ds_name)
                            if mu and ds_name not in mito_instance_urls:
                                mito_instance_urls[ds_name] = mu
                        elif knd == "semantic":
                            mito_semantic_datasets.add(ds_name)
                            if mu and ds_name not in mito_semantic_urls:
                                mito_semantic_urls[ds_name] = mu

    def _cap_sorted(s: set[str]) -> list[str]:
        return sorted(s, key=str.lower)[:max_layer_names]

    # Reconcile: any row whose *image* layer names clearly include GT mito_seg-style masks
    # counts as good, even if per-layer heuristics also flagged a mito_pred sibling layer.
    for d in rows:
        if not isinstance(d, dict):
            continue
        ds_name = str(d.get("name") or d.get("dataset_name") or "").strip()
        if not ds_name:
            continue
        for im in _dataset_image_entries(d)[:cap_img]:
            if not isinstance(im, dict):
                continue
            iname = str(im.get("name") or "").strip()
            ict = str(im.get("content_type") or "").strip()
            if catalog_image_layer_name_indicates_gt_mito_mask(
                name=iname, content_type=ict
            ):
                mito_good_datasets.add(ds_name)
                break

    mito_pred_only_datasets -= mito_good_datasets

    # Merge instance + semantic URL maps into a single good-label URL map.
    all_good_urls: dict[str, str] = {}
    for slug, u in {**mito_semantic_urls, **mito_instance_urls}.items():
        all_good_urls.setdefault(slug, u)

    md_lines: list[str] = [
        f"- **Datasets in aggregate:** {len(rows)}",
        f"- **Datasets with at least one ``image`` / ``images`` entry:** {n_with_any_image}",
        f"- **Datasets with mitochondria-related image/mesh *names* (regex):** {n_datasets_mito_layers}",
        f"- **Datasets with good (non-prediction) mitochondria mask layers:** {len(mito_good_datasets)}",
        f"- **Datasets with only prediction-style mito layers:** {len(mito_pred_only_datasets)}",
    ]
    md_lines.append(
        f"**Datasets with good mito segmentation labels ({len(mito_good_datasets)} total, slug → best label URL):**"
    )
    for slug in sorted(all_good_urls.keys(), key=str.lower):
        u = all_good_urls[slug]
        cap_u = u if len(u) <= 180 else u[:179] + "…"
        md_lines.append(f"  - `{slug}`: `{cap_u}`")
    # Also include good datasets that have no explicit URL captured above.
    for slug in sorted(mito_good_datasets - set(all_good_urls), key=str.lower):
        md_lines.append(f"  - `{slug}`")
    if not mito_good_datasets:
        md_lines.append("  - _(none)_")
    if img_formats:
        top_if = list(img_formats.most_common(15))
        md_lines.append(
            "- **Image layer formats (top):** "
            + ", ".join(f"`{a}`×{b}" for a, b in top_if)
        )
    if mesh_formats:
        top_mf = list(mesh_formats.most_common(15))
        md_lines.append(
            "- **Mesh formats (top):** " + ", ".join(f"`{a}`×{b}" for a, b in top_mf)
        )
    md_lines.append("")
    md_lines.append(
        "_Per-dataset layer names are in **Primary datasets (schema rows)** below; "
        "global unique lists are omitted (available in JSON via the same REST embed if needed)._"
    )

    return {
        "dataset_row_count": len(rows),
        "segmentation_challenge_counts": {},
        "datasets_with_segmentation_challenge": 0,
        "datasets_with_mitochondria_layer_names": n_datasets_mito_layers,
        "mito_training_usable_datasets": sorted(mito_good_datasets, key=str.lower),
        "mito_good_datasets": sorted(mito_good_datasets, key=str.lower),
        "mito_refined_prediction_datasets": [],
        "mito_instance_label_urls": dict(
            sorted(mito_instance_urls.items(), key=lambda kv: kv[0].lower())
        ),
        "mito_semantic_label_urls": dict(
            sorted(mito_semantic_urls.items(), key=lambda kv: kv[0].lower())
        ),
        "mito_unproofread_prediction_datasets": sorted(
            mito_pred_only_datasets, key=str.lower
        ),
        "mito_instance_segmentation_datasets": sorted(
            mito_instance_datasets, key=str.lower
        ),
        "mito_semantic_segmentation_datasets": sorted(
            mito_semantic_datasets, key=str.lower
        ),
        "unique_ground_truth_image_layers": _cap_sorted(gt_img),
        "unique_prediction_image_layers": _cap_sorted(pred_img),
        "unique_other_image_layers": _cap_sorted(other_img),
        "unique_ground_truth_mesh_names": _cap_sorted(gt_mesh),
        "unique_prediction_mesh_names": _cap_sorted(pred_mesh),
        "unique_other_mesh_names": _cap_sorted(other_mesh),
        "image_format_counts": dict(img_formats.most_common(40)),
        "mesh_format_counts": dict(mesh_formats.most_common(40)),
        "markdown_lines": md_lines,
    }


def python_access_markdown(rest_url_stripped: str, placeholder_key: str = "YOUR_ANON_APIKEY") -> str:
    """Template for listing datasets via PostgREST using **params** (never a bare URL without ``apikey``)."""
    base = strip_limit_offset_query(rest_url_stripped)
    p = urlparse(base)
    path = (p.path or "").rstrip("/") or "/rest/v1/dataset"
    base_no_query = urlunparse((p.scheme, p.netloc, path, p.params, "", p.fragment))
    select = os.getenv("SUPABASE_DATASET_SELECT", "").strip() or DEFAULT_DATASET_LAYER_EMBED_SELECT
    base_lit = repr(base_no_query)
    select_lit = repr(select)
    return f"""
### Python: list datasets via Supabase REST (PostgREST)

Use **query params** + **`apikey` / `Authorization` headers**. Do not open the REST URL in a browser without headers (you will get ``No API key found``).

```python
import requests

BASE = {base_lit}
SELECT = {select_lit}
API_KEY = "{placeholder_key}"  # paste apikey from Appendix

headers = {{
    "apikey": API_KEY,
    "Authorization": f"Bearer {{API_KEY}}",
    "Accept": "application/json",
}}


def fetch_all_dataset_rows(
    base_url: str,
    select: str,
    *,
    page_size: int = 200,
    max_rows: int = 5000,
    timeout: float = 90,
):
    out = []
    offset = 0
    while len(out) < max_rows:
        r = requests.get(
            base_url,
            params={{"select": select, "limit": page_size, "offset": offset}},
            headers=headers,
            timeout=timeout,
        )
        r.raise_for_status()
        batch = r.json()
        if not isinstance(batch, list) or not batch:
            break
        out.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return out


rows = fetch_all_dataset_rows(BASE, SELECT)
print(len(rows), "datasets")
for row in rows[:5]:
    print(row.get("name"), (row.get("description") or "")[:80])
```

**Layers / meshes:** rows include embedded ``image`` + ``mesh`` when using the same ``SELECT`` as the scraper. **Dimensions / spacing:** ``image_acquisition`` on each row.
""".strip()
