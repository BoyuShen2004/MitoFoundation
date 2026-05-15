"""Walk nested JSON (XHR / Supabase) and build per-dataset inventory lines."""

from __future__ import annotations

import ast
import os
import re
from typing import List

try:
    from .mito_layer_policy import (
        classify_layer_gt_pred_raw,
        mito_label_training_status,
        mito_layer_identity_matches,
        mito_segmentation_kind,
    )
    from .supabase_bulk import (
        _dataset_image_entries,
        _image_mesh_entries,
        dataset_has_mitochondria_layer_row,
    )
except ImportError:
    # When imported as top-level ``dataset_inventory`` (from 1web_scraper_01),
    # package-relative imports are unavailable; fall back to deep scraper modules.
    from master.deep.mito_layer_policy import (  # type: ignore[no-redef]
        classify_layer_gt_pred_raw,
        mito_label_training_status,
        mito_layer_identity_matches,
        mito_segmentation_kind,
    )
    from master.deep.supabase_bulk import (  # type: ignore[no-redef]
        _dataset_image_entries,
        _image_mesh_entries,
        dataset_has_mitochondria_layer_row,
    )
_S3FS_CLIENT = None
_LEAF_CACHE: dict[tuple[str, bool], tuple[str | None, str | None]] = {}


def _norm_json_key(k: str) -> str:
    return re.sub(r"[_\s-]", "", k.lower())


def _parse_number_list(val: object) -> list[float] | None:
    """Parse grid_spacing / grid_dimensions from JSON list/tuple or a bracketed string."""
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        try:
            return [float(x) for x in val]
        except (TypeError, ValueError):
            return None
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [float(x) for x in parsed]
        except (ValueError, SyntaxError, TypeError):
            pass
    return None


def _classify_grid_dimensions_unit(u: object) -> str:
    """Return um | nm | voxel | unknown."""
    s = str(u or "").strip().lower().replace("μ", "u")
    if s in ("um", "micrometer", "micrometers", "micron", "microns"):
        return "um"
    if s in ("nm", "nanometer", "nanometers"):
        return "nm"
    if s in ("voxel", "voxels", "vx"):
        return "voxel"
    return "unknown"


def _spacing_is_nm(u: object) -> bool:
    return _classify_grid_dimensions_unit(u) == "nm"


def _compute_voxel_grid_heuristic(ia: dict) -> list[int] | None:
    """Convert ``image_acquisition`` grid to voxel counts; ``None`` if skipped (unknown units, etc.)."""
    dims = _parse_number_list(ia.get("grid_dimensions"))
    spacing = _parse_number_list(ia.get("grid_spacing"))
    if not dims or not spacing or len(dims) != len(spacing):
        return None
    if any(s <= 0 for s in spacing):
        return None
    if not _spacing_is_nm(ia.get("grid_spacing_unit")):
        return None

    du = _classify_grid_dimensions_unit(ia.get("grid_dimensions_unit"))
    if du == "voxel":
        return [int(round(x)) for x in dims]
    if du == "unknown":
        return None

    if du == "um":
        return [int(round((dims[i] * 1000.0) / spacing[i])) for i in range(len(dims))]

    # du == "nm" — physical nm vs voxel counts mislabeled as nm
    ratios = [dims[i] / spacing[i] for i in range(len(dims))]
    all_integral = all(abs(r - round(r)) < 1e-4 for r in ratios)
    if all_integral:
        return [int(round(r)) for r in ratios]
    return [int(round(x)) for x in dims]


def _normalize_image_acquisition_grid_to_voxels(ia: dict) -> dict:
    """Emit ``grid_dimensions`` as voxel counts and ``grid_dimensions_unit=voxel``.

    PostgREST ``image_acquisition`` mixes physical extents (μm or nm) with voxel-like
    integers mislabeled as ``nm``. Heuristic:

    - **μm**: ``voxels = round((dim_um * 1000) / spacing_nm)`` per axis (spacing assumed nm).
    - **nm**: if ``dim_nm / spacing_nm`` is integral on every axis (within 1e-4), treat as
      physical nanometers; else treat ``grid_dimensions`` as already voxel counts.
    - **voxel**: pass through (rounded integers).
    - **unknown**: leave dimensions and unit unchanged.

    Set ``INVENTORY_GRID_NORMALIZE_VOXELS=0`` to disable and print raw API values.
    """
    if os.getenv("INVENTORY_GRID_NORMALIZE_VOXELS", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return ia
    voxels = _compute_voxel_grid_heuristic(ia)
    if voxels is None:
        return ia
    out = dict(ia)
    out["grid_dimensions"] = voxels
    out["grid_dimensions_unit"] = "voxel"
    return out


def inventory_line_source_suffix(source_url: str) -> str:
    """Keep source URL for scripts; Supabase rows need ``apikey`` / Bearer (browser-only GET may fail)."""
    s = (source_url or "").strip()
    if not s:
        return ""
    low = s.lower()
    cap = int(os.getenv("APPENDIX_POSTGREST_URL_MAX_LEN", "240"))
    cap = max(80, min(cap, 2000))
    frag = s[:cap] + ("…" if len(s) > cap else "")
    if "supabase.co" in low and "/rest/v1/" in low:
        return (
            f" _(PostgREST row source: `{frag}` — send **apikey** + **Authorization: Bearer**; "
            "bare browser/curl without headers → “No API key found”.)_"
        )
    return f" _(captured JSON · `{frag}`)_"


_DATASET_ROW_HINT_KEYS = frozenset(
    _norm_json_key(h)
    for h in (
        "id",
        "uuid",
        "slug",
        "name",
        "title",
        "key",
        "dataset_id",
        "bossKey",
        "collection",
        "description",
        "display_name",
        "dataset_key",
        "volume",
        "experiment",
        "stage",
        "segmentation_challenge",
        "sample",
        "image_acquisition",
    )
)


def _looks_like_json_image_leaf(d: dict) -> bool:
    """Exclude small image/mesh asset objects mistaken for dataset catalog rows."""
    if not isinstance(d, dict) or len(d) > 22:
        return False
    nk = {_norm_json_key(k) for k in d}
    if "url" not in nk:
        return False
    if "sample" in nk or "imageacquisition" in nk:
        return False
    if "dataset" in nk or "datasetname" in nk:
        return False
    if ("format" in nk or "contenttype" in nk or "content_type" in nk) and len(d) <= 16:
        if "segmentationchallenge" not in nk:
            return True
    return False


def _looks_like_dataset_row(d: dict) -> bool:
    """Catalog row: dataset / view metadata."""
    if not isinstance(d, dict):
        return False
    if _looks_like_json_image_leaf(d):
        return False
    nk = {_norm_json_key(k) for k in d}
    if "name" in nk and (
        "description" in nk
        or "sample" in nk
        or "imageacquisition" in nk
        or "images" in nk
        or "image" in nk
        or "segmentationchallenge" in nk
    ):
        return True
    if "datasetname" in nk and "name" in nk and "description" in nk:
        return True
    if len(d) > 120:
        return False
    hits = sum(1 for hint in _DATASET_ROW_HINT_KEYS if hint in nk)
    return hits >= 2


def _exact_primary_name(d: dict) -> str:
    """Exact string as in API (no truncation)."""
    for want in ("dataset_name", "datasetName", "name", "title", "slug", "key"):
        wn = _norm_json_key(want)
        for k, v in d.items():
            if _norm_json_key(k) != wn:
                continue
            if v is None:
                continue
            s = str(v).strip()
            if s and s.lower() not in ("null", "none"):
                return s
    return ""


def _dataset_inventory_dedup_key(d: dict) -> str:
    """Stable key: one row per dataset, or per (dataset, view) when both differ."""
    ds = ""
    nm = ""
    for k, v in d.items():
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        if _norm_json_key(k) == _norm_json_key("dataset_name"):
            ds = s.lower()
        if _norm_json_key(k) == _norm_json_key("name"):
            nm = s.lower()
    if ds:
        if nm and nm != ds:
            return f"{ds}::view::{nm}"
        return ds
    if nm:
        return nm
    return ""


def _classify_gt_vs_pred(
    label: str,
    *,
    content_type: str = "",
    stage: str = "",
    url: str = "",
) -> str:
    """gt | pred | raw | other — delegates to shared policy (keeps appendix buckets stable)."""
    return classify_layer_gt_pred_raw(
        label or "",
        content_type=content_type or "",
        stage=stage or "",
        url=url or "",
    )


def _appendix_mito_quality_token(
    *,
    name: str,
    desc: str,
    url: str,
    content_type: str,
    stage: str,
    image_stack: str,
    layer_cls: str,
    segmentation_kind_field: str | None,
) -> tuple[str, str]:
    """Return (quality_token, kind) for ``[mito_label_quality=…][mito_segmentation_kind=…]``."""
    status = mito_label_training_status(
        name=name,
        desc=desc,
        url=url,
        content_type=content_type,
        stage=stage,
        image_stack=image_stack,
        layer_cls=layer_cls,
    )
    kind = mito_segmentation_kind(
        name=name,
        desc=desc,
        url=url,
        segmentation_kind_field=segmentation_kind_field,
    )
    if status == "unrelated":
        return ("", "")
    if status == "good":
        q = "good"
    elif status == "pred_unproofread":
        q = "pred_unproofread"
    else:
        q = "non_ground_mito"
    return (q, kind)


def _layer_url_max() -> int:
    return max(80, min(600, int(os.getenv("INVENTORY_LAYER_URL_MAX", "340"))))


def _emit_layer_urls() -> bool:
    return os.getenv("INVENTORY_EMIT_LAYER_URLS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _cap_token_value(s: str, max_len: int) -> str:
    s = s.strip()
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


def _append_image_access_meta(tok: str, im: dict) -> str:
    """Add PostgREST ``url`` / ``image_stack`` / ``content_type`` for database_builder + download scripts."""
    if not _emit_layer_urls():
        return tok
    ct = str(im.get("content_type") or "").strip()
    if ct:
        tok += f"[content_type={_cap_token_value(ct, 80)}]"
    u = str(im.get("url") or "").strip()
    if u:
        tok += f"[url={_cap_token_value(u, _layer_url_max())}]"
    ist = str(im.get("image_stack") or "").strip()
    if ist and len(ist) <= 200:
        tok += f"[image_stack={ist}]"
    return tok


def _append_mesh_access_meta(tok: str, mesh: dict) -> str:
    if not _emit_layer_urls():
        return tok
    u = str(mesh.get("url") or "").strip()
    if u:
        tok += f"[url={_cap_token_value(u, _layer_url_max())}]"
    return tok


def _download_hint_url_max() -> int:
    return max(120, min(1200, int(os.getenv("DOWNLOAD_HINT_URL_MAX", "480"))))


def _want_leaf_resolution() -> bool:
    return os.getenv("INVENTORY_RESOLVE_S3_LEAF_KEYS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _s3fs_client():
    global _S3FS_CLIENT
    if _S3FS_CLIENT is not None:
        return _S3FS_CLIENT
    try:
        import s3fs  # type: ignore

        _S3FS_CLIENT = s3fs.S3FileSystem(anon=True)
    except Exception:
        _S3FS_CLIENT = False
    return _S3FS_CLIENT


def _path_candidates(p: str, *, is_image: bool) -> list[str]:
    s = (p or "").strip()
    if not s:
        return []
    vals = [s]
    b = s.rstrip("/")
    vals += [b + "/", b + "/s0", b + "/s1", b + "/0", b + "/1", b + "/s0/", b + "/s1/", b + "/0/", b + "/1/"]
    if is_image and "fibsem-uint8" in b and "fibsem-uint8_" not in b:
        vals += [b.replace("fibsem-uint8", "fibsem-uint8_1"), b.replace("fibsem-uint8", "fibsem-uint8_1") + "/"]
    if is_image and "fibsem-uint16" in b and "fibsem-uint16_" not in b:
        vals += [b.replace("fibsem-uint16", "fibsem-uint16_1"), b.replace("fibsem-uint16", "fibsem-uint16_1") + "/"]
    out: list[str] = []
    seen = set()
    for v in vals:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _scale_key_from_path(p: str) -> str | None:
    b = p.rstrip("/").split("/")[-1]
    if b in ("s0", "s1", "0", "1"):
        return b
    return None


def _resolve_s3_leaf_url(path: str, *, is_image: bool) -> tuple[str | None, str | None]:
    key = (path, is_image)
    if key in _LEAF_CACHE:
        return _LEAF_CACHE[key]
    if not _want_leaf_resolution() or not path.startswith("s3://"):
        _LEAF_CACHE[key] = (None, None)
        return (None, None)
    fs = _s3fs_client()
    if not fs:
        _LEAF_CACHE[key] = (None, None)
        return (None, None)
    resolved = None
    scale = None
    try:
        for cand in _path_candidates(path, is_image=is_image):
            exists = fs.exists(cand[5:])  # strip s3://
            if exists:
                resolved = cand
                scale = _scale_key_from_path(cand)
                break
    except Exception:
        pass
    _LEAF_CACHE[key] = (resolved, scale)
    return (resolved, scale)


def _scale_key_from_url(path: str) -> str | None:
    return _scale_key_from_path(path)


def _dataset_store_prefix(path: str) -> str:
    """Dataset root before /em/ or /labels/ for rough image-label alignment scoring."""
    s = (path or "").strip().rstrip("/")
    low = s.lower()
    for marker in ("/em/", "/labels/"):
        i = low.find(marker)
        if i >= 0:
            return s[:i]
    return s


def _score_em_candidate_for_mito_label(
    *,
    em_url: str,
    em_name: str,
    mito_url: str | None,
    preferred_scale: str | None,
) -> int:
    """Prefer EM path that aligns with selected mito mask path."""
    score = 0
    low = em_name.lower()
    if "fibsem" in low:
        score += 20
    if "uint8" in low or "uint16" in low:
        score += 10
    em_scale = _scale_key_from_url(em_url)
    if preferred_scale and em_scale == preferred_scale:
        score += 50
    if mito_url:
        if _dataset_store_prefix(em_url) == _dataset_store_prefix(mito_url):
            score += 80
    # When masks are s0, prefer full-resolution EM to reduce shape mismatch risk.
    if preferred_scale == "s0" and em_scale == "s0":
        score += 20
    return score


def _download_access_hints(d: dict) -> list[str]:
    """Emit dataset-level ``download_*`` URL fields (OpenOrganelle: N5/Zarr source paths)."""
    if os.getenv("INVENTORY_EMIT_DOWNLOAD_HINTS", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return []
    url_cap = _download_hint_url_max()
    cap_img = int(os.getenv("PLAYWRIGHT_INVENTORY_MAX_IMAGES", "80"))
    images = _dataset_image_entries(d)[:cap_img]
    em_candidates: list[dict] = []
    mito_good: list[dict] = []
    mito_pred: list[dict] = []

    def cap(u: str) -> str:
        u = u.strip()
        if len(u) > url_cap:
            return u[: url_cap - 1] + "…"
        return u

    for im in images:
        if not isinstance(im, dict):
            continue
        u = str(im.get("url") or "").strip()
        if not u:
            continue
        iname = str(im.get("name") or "")
        idesc = str(im.get("description") or "")
        ct = str(im.get("content_type") or "").strip()
        istage = str(im.get("stage") or "").strip()
        istack = str(im.get("image_stack") or "").strip()
        cls = _classify_gt_vs_pred(
            " ".join([iname, idesc]),
            content_type=ct,
            stage=istage,
            url=u,
        )
        nl = iname.lower()
        is_em_candidate = cls == "raw" or "fibsem" in nl or "uint8" in nl or "uint16" in nl
        if is_em_candidate:
            em_candidates.append(
                {
                    "url": u,
                    "name": iname,
                    "image_stack": istack,
                }
            )
        if not mito_layer_identity_matches(
            name=iname, desc=idesc, url=u, image_stack=istack
        ):
            continue
        status = mito_label_training_status(
            name=iname,
            desc=idesc,
            url=u,
            content_type=ct,
            stage=istage,
            image_stack=istack,
            layer_cls=cls,
        )
        sk_field = im.get("segmentation_kind")
        sk = str(sk_field).strip() if sk_field is not None else None
        kind = mito_segmentation_kind(
            name=iname,
            desc=idesc,
            url=u,
            segmentation_kind_field=sk,
        )
        rec = {"url": u, "quality": status, "kind": kind, "name": iname}
        if status == "good":
            mito_good.append(rec)
        elif status == "pred_unproofread":
            mito_pred.append(rec)

    mito_good.sort(
        key=lambda r: (
            0 if r.get("kind") == "semantic" else 1,
            str(r.get("name") or "").lower(),
        )
    )
    chosen_mito = mito_good[0] if mito_good else None
    mito_url = chosen_mito["url"] if chosen_mito else None
    preferred_scale = _scale_key_from_url(mito_url or "")
    em_url = None
    em_stack = None
    if em_candidates:
        em_best = max(
            em_candidates,
            key=lambda r: _score_em_candidate_for_mito_label(
                em_url=str(r.get("url") or ""),
                em_name=str(r.get("name") or ""),
                mito_url=mito_url,
                preferred_scale=preferred_scale,
            ),
        )
        em_url = str(em_best.get("url") or "").strip() or None
        em_stack = str(em_best.get("image_stack") or "").strip() or None

    out: list[str] = []
    if em_url:
        out.append(f"download_volume_url={cap(em_url)}")
        leaf, scale = _resolve_s3_leaf_url(em_url, is_image=True)
        if leaf:
            out.append(f"download_volume_leaf_url={cap(leaf)}")
        if scale:
            out.append(f"download_volume_scale_key={scale}")
    if em_stack:
        out.append(f"download_volume_image_stack={_cap_token_value(em_stack, 200)}")
    if mito_url:
        out.append(f"download_mito_mask_url={cap(mito_url)}")
        if chosen_mito:
            out.append("download_mito_mask_quality=good")
            out.append(
                f"download_mito_mask_segmentation_kind={chosen_mito.get('kind')}"
            )
        leaf, scale = _resolve_s3_leaf_url(mito_url, is_image=False)
        if leaf:
            out.append(f"download_mito_mask_leaf_url={cap(leaf)}")
        if scale:
            out.append(f"download_mito_mask_scale_key={scale}")
    elif mito_pred:
        # Keep explicit prediction fallback only when no good/refined mask candidate exists.
        pred_url = str(mito_pred[0].get("url") or "")
        out.append(f"download_mito_prediction_url={cap(pred_url)}")
        out.append("download_mito_prediction_quality=pred_unproofread")
        leaf, scale = _resolve_s3_leaf_url(pred_url, is_image=False)
        if leaf:
            out.append(f"download_mito_prediction_leaf_url={cap(leaf)}")
        if scale:
            out.append(f"download_mito_prediction_scale_key={scale}")
    return out


def _collect_layers_and_ground_truths(d: dict) -> tuple[list[str], list[str], list[str]]:
    """From nested ``images`` / ``meshes``: layer names classified as GT / pred / raw."""
    gt: list[str] = []
    pred: list[str] = []
    raw_other: list[str] = []

    def push(bucket: list[str], token: str) -> None:
        if token and token not in bucket:
            bucket.append(token)

    cap_img = int(os.getenv("PLAYWRIGHT_INVENTORY_MAX_IMAGES", "80"))
    cap_mesh = int(os.getenv("PLAYWRIGHT_INVENTORY_MAX_MESHES", "50"))

    images = _dataset_image_entries(d)[:cap_img]
    for im in images:
        if not isinstance(im, dict):
            continue
        iname = str(im.get("name") or "").strip()
        ifmt = str(im.get("format") or "").strip()
        istage = str(im.get("stage") or "").strip()
        idesc = str(im.get("description") or "").strip()
        ict = str(im.get("content_type") or "").strip()
        istack = str(im.get("image_stack") or "").strip()
        if not iname:
            continue
        iurl_early = str(im.get("url") or "").strip()
        cls = _classify_gt_vs_pred(
            iname + " " + idesc,
            content_type=ict,
            stage=istage,
            url=iurl_early,
        )
        tok = f"image:{iname}"
        if ifmt:
            tok += f"[format={ifmt}]"
        if istage:
            tok += f"[stage={istage}]"
        tok = _append_image_access_meta(tok, im)
        sk_raw = im.get("segmentation_kind")
        sk_field = str(sk_raw).strip() if sk_raw is not None else None
        mq, mk = _appendix_mito_quality_token(
            name=iname,
            desc=idesc,
            url=iurl_early,
            content_type=ict,
            stage=istage,
            image_stack=istack,
            layer_cls=cls,
            segmentation_kind_field=sk_field or None,
        )
        if mq:
            tok += f"[mito_label_quality={mq}][mito_segmentation_kind={mk}]"
        if cls == "gt":
            push(gt, tok)
        elif cls == "pred":
            push(pred, tok)
        elif cls == "raw":
            push(raw_other, tok)
        else:
            push(raw_other, tok + "[semantic]")

        for mesh in _image_mesh_entries(im)[:cap_mesh]:
            if not isinstance(mesh, dict):
                continue
            mn = str(mesh.get("name") or "").strip()
            mf = str(mesh.get("format") or "").strip()
            mst = str(mesh.get("stage") or "").strip()
            if not mn:
                continue
            mdsc = str(mesh.get("description") or "")
            mu = str(mesh.get("url") or "").strip()
            mcls = _classify_gt_vs_pred(
                mn + " " + mdsc, content_type="", stage=mst, url=mu
            )
            mtok = f"mesh:{mn}"
            if mf:
                mtok += f"[format={mf}]"
            if mst:
                mtok += f"[stage={mst}]"
            mtok = _append_mesh_access_meta(mtok, mesh)
            sk_raw = mesh.get("segmentation_kind")
            sk_field = str(sk_raw).strip() if sk_raw is not None else None
            mq, mk = _appendix_mito_quality_token(
                name=mn,
                desc=mdsc,
                url=mu,
                content_type="",
                stage=mst,
                image_stack="",
                layer_cls=mcls,
                segmentation_kind_field=sk_field or None,
            )
            if mq:
                mtok += f"[mito_label_quality={mq}][mito_segmentation_kind={mk}]"
            if mcls == "gt":
                push(gt, mtok)
            elif mcls == "pred":
                push(pred, mtok)
            elif mcls == "raw":
                push(raw_other, mtok)
            else:
                push(raw_other, mtok + "[semantic]")

    return gt, pred, raw_other


def _inventory_line_richness(line: str) -> int:
    score = len(line)
    if "ground_truths=" in line:
        score += 500
    score += line.count("mesh:") * 40
    score += line.count("image:") * 25
    if "download_mito_mask_url=" in line:
        score += 300
    if "download_volume_url=" in line:
        score += 200
    score += line.count("[url=") * 15
    return score


def _summarize_dataset_row(d: dict) -> str:
    """Structured summary: name + GT/pred/raw layers for appendix / schema."""
    parts: List[str] = []
    cap = int(os.getenv("PLAYWRIGHT_DATASET_FIELD_CAP", "200"))
    name_max = int(os.getenv("PLAYWRIGHT_DATASET_NAME_MAX", "0"))

    def add(label: str, v: object, max_len: int | None = None) -> None:
        if v is None:
            return
        s = str(v).strip()
        if not s or s.lower() in ("null", "none"):
            return
        ml = max_len if max_len is not None else cap
        if ml > 0 and len(s) > ml:
            s = s[: ml - 1] + "…"
        parts.append(f"{label}={s}")

    skip_norm = frozenset(
        _norm_json_key(x)
        for x in ("thumbnail_url", "preview_url", "avatar_url", "icon_url")
    )

    exact = _exact_primary_name(d)
    if exact:
        if name_max > 0:
            add("dataset_name", exact, name_max)
        else:
            parts.append(f"dataset_name={exact}")

    ds_val = ""
    nm_val = ""
    for k, v in d.items():
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        if _norm_json_key(k) == _norm_json_key("dataset_name"):
            ds_val = s
        if _norm_json_key(k) == _norm_json_key("name"):
            nm_val = s
    if ds_val and nm_val and nm_val != ds_val:
        parts.append(f"view_title={nm_val}")

    top_order = (
        "title",
        "slug",
        "key",
        "id",
        "uuid",
        "datasetId",
        "dataset_id",
        "bossKey",
        "collection",
        "stage",
    )
    for want in top_order:
        for k, v in d.items():
            if _norm_json_key(k) in skip_norm:
                continue
            if _norm_json_key(k) == _norm_json_key(want):
                add(k, v)
                break

    desc = None
    for k, v in d.items():
        if _norm_json_key(k) in skip_norm:
            continue
        if _norm_json_key(k) == _norm_json_key("description"):
            desc = v
            break
    if desc is not None:
        add("description", desc, max_len=min(400, cap + 180))

    sample = d.get("sample")
    if isinstance(sample, dict):
        for sk in (
            "organism",
            "type",
            "subtype",
            "treatment",
            "protocol",
            "name",
        ):
            if sk in sample:
                add(f"sample.{sk}", sample[sk], 160)

    ia = d.get("image_acquisition")
    if isinstance(ia, dict):
        ia_grid = _normalize_image_acquisition_grid_to_voxels(ia)
        for ik in (
            "grid_spacing",
            "grid_dimensions",
            "grid_axes",
            "grid_spacing_unit",
            "grid_dimensions_unit",
            "name",
            "institution",
            "start_date",
        ):
            if ik in ia_grid and ia_grid[ik] is not None:
                add(f"grid.{ik}", ia_grid[ik], 180)

    gt_layers, pred_layers, raw_layers = _collect_layers_and_ground_truths(d)
    if gt_layers:
        parts.append("ground_truths=" + "; ".join(gt_layers))
    if pred_layers:
        parts.append("predictions=" + "; ".join(pred_layers))
    if raw_layers:
        parts.append("raw_or_other_layers=" + "; ".join(raw_layers))

    name_s = (exact or str(d.get("title") or d.get("slug") or "")).lower()
    desc_s = str(d.get("description") or "").lower()
    if dataset_has_mitochondria_layer_row(d):
        parts.append("mitochondria_in_layer_names=yes")

    if not gt_layers and not pred_layers:
        fallback_gt: list[str] = []
        if "_seg" in name_s or "segmentation" in desc_s or "ground truth" in desc_s:
            fallback_gt.append("name_or_desc_hints_segmentation_or_gt")
        if "_pred" in name_s or "prediction" in desc_s:
            fallback_gt.append("name_or_desc_hints_prediction")
        if "raw" in name_s or "fibsem" in name_s:
            fallback_gt.append("name_or_desc_hints_raw_volume")
        if fallback_gt:
            parts.append("ground_truth_hints=" + "; ".join(fallback_gt))

    org_hint = []
    for token in (
        "mito",
        "mitochond",
        "nucleus",
        "er",
        "golgi",
        "lysosome",
        "organelle",
        "centrosome",
        "ribosome",
        "vesicle",
        "endoplasmic",
    ):
        if token in name_s or token in desc_s:
            org_hint.append(token)
    if org_hint:
        add("organelle_keywords", ",".join(sorted(set(org_hint))), 120)

    hints = _download_access_hints(d)
    for hint in hints:
        parts.append(hint)
    if any(h.startswith("download_mito_mask_url=") for h in hints):
        parts.append("good_mitochondria_gt_mask=yes")

    return " | ".join(parts) if parts else str(d)[:400]


def extract_dataset_inventory(
    obj,
    source_url: str,
    best_by_key: dict[str, str],
    limit_unique: int,
) -> None:
    """Merge rows by dataset key; keep the richest line per dataset."""
    if isinstance(obj, dict):
        if _looks_like_dataset_row(obj):
            line = _summarize_dataset_row(obj)
            if line:
                key = _dataset_inventory_dedup_key(obj) or f"anon:{hash(line) % (10**9)}"
                suffix = inventory_line_source_suffix(source_url)
                full = line + suffix
                if key not in best_by_key and len(best_by_key) >= limit_unique:
                    pass
                else:
                    prev = best_by_key.get(key)
                    if prev is None or _inventory_line_richness(full) > _inventory_line_richness(
                        prev
                    ):
                        best_by_key[key] = full
        for v in obj.values():
            extract_dataset_inventory(v, source_url, best_by_key, limit_unique)
    elif isinstance(obj, list):
        for item in obj[:500]:
            extract_dataset_inventory(item, source_url, best_by_key, limit_unique)


# Back-compat for internal callers
_extract_dataset_inventory = extract_dataset_inventory
