"""HTML signal mining, heuristics, and appendix formatting."""

from __future__ import annotations

import os
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from .appendix_caps import appendix_cap


def heuristic_extract(scraped_text: str) -> dict:
    """Best-effort extraction when LLM is unavailable."""
    low = scraped_text.lower()

    last_mod = "Not explicitly stated"
    m = re.search(r"last-?modified[:\s]+([^\n]+)", scraped_text, flags=re.IGNORECASE)
    if m:
        last_mod = m.group(1).strip()

    data_volume = "Not explicitly stated"
    m2 = re.search(
        r"(\d[\d,\.]*)\s*(entries|datasets|images|records)",
        low,
    )
    if m2:
        data_volume = f"{m2.group(1)} {m2.group(2)}"
    else:
        m3 = re.search(
            r"(\d[\d,\.]*\s*(?:pib|tib|gib|tb|gb|pb))",
            low,
        )
        if m3:
            data_volume = m3.group(1).strip()

    organelles = []
    for org in ["mitochondria", "nucleus", "er", "golgi", "lysosome", "organelle"]:
        if org in low:
            organelles.append(org)
    organelles_text = ", ".join(sorted(set(organelles))) if organelles else "Not explicitly stated"

    label_status = "Not explicitly stated"
    if "unlabeled" in low:
        label_status = "Unlabeled"
    elif "labeled" in low or "annotat" in low:
        label_status = "Labeled"
    elif "predicted" in low or "proofread" in low or "semi" in low:
        label_status = "Mixed-Intermediate"

    key_lines = []
    for line in scraped_text.splitlines():
        ll = line.lower().strip()
        if not ll:
            continue
        if any(k in ll for k in ["mito", "dataset", "entry", "tomography", "segmentation", "label"]):
            key_lines.append(line.strip())
        if len(key_lines) >= 6:
            break

    return {
        "last_mod": last_mod,
        "data_volume": data_volume,
        "organelles": organelles_text,
        "label_status": label_status,
        "key_lines": key_lines,
    }


def _normalize_url_fragment(u: str) -> str:
    u = u.rstrip(").,;\"'")
    if len(u) > 500:
        return u[:500]
    return u


def is_noise_preview_media_url(u: str) -> bool:
    """True for thumbnail / raster preview URLs we should not list (e.g. S3 .jpg cards).

    Keeps object-storage URLs that look like real data paths (no image extension).
    """
    lu = (u or "").lower().strip()
    if not lu.startswith(("http://", "https://")):
        return False
    path = urlparse(u).path.lower()
    if re.search(r"\.(jpe?g|png|gif|webp|ico|bmp|svg)(\?.*)?$", path):
        return True
    if any(
        x in lu
        for x in (
            "thumbnail",
            "thumbnails",
            "organelle-thumbnails",
            "-thumbnails-",
            "/thumb/",
            "/thumbs/",
            "/preview/",
            "/avatar/",
            "favicon",
        )
    ):
        return True
    return False


def _same_site_host(u_netloc: str, base_netloc: str) -> bool:
    if not u_netloc or not base_netloc:
        return False
    if u_netloc == base_netloc:
        return True
    if u_netloc.endswith("." + base_netloc):
        return True
    bu, uu = base_netloc.removeprefix("www."), u_netloc.removeprefix("www.")
    return bu == uu


def extract_access_signals(raw_html: str, page_url: str, soup: BeautifulSoup | None) -> dict:
    """Mine HTML/text for URLs and patterns useful for programmatic data access."""
    base_netloc = urlparse(page_url).netloc
    base_root = f"{urlparse(page_url).scheme}://{base_netloc}"

    blob = raw_html or ""
    if soup is not None:
        for a in soup.find_all("a", href=True):
            blob += " " + urljoin(page_url, a["href"])
        for s in soup.find_all("script", src=True):
            blob += " " + urljoin(page_url, s["src"])

    urls = set()
    for m in re.finditer(
        r"https?://[^\s\"'<>)\]]+",
        blob,
        flags=re.IGNORECASE,
    ):
        urls.add(_normalize_url_fragment(m.group(0)))

    api_like = []
    download_like = []
    doc_like = []
    same_host = []
    s3_like = []

    api_tokens = ("/api", "/v1/", "/v2/", "/v3/", "graphql", "swagger", "openapi", ".json", "rest/")
    dl_tokens = ("download", "export", "bulk", "wget", "curl", "s3", "bucket", "manifest", "tar", "zip")
    doc_tokens = ("docs", "documentation", "developer", "guide", "tutorial", "github.com", "readthedocs")

    for u in urls:
        lu = u.lower()
        p = urlparse(u)
        noise = is_noise_preview_media_url(u)
        if _same_site_host(p.netloc, base_netloc) and not noise:
            same_host.append(u)
        if any(t in lu for t in api_tokens):
            api_like.append(u)
        if any(t in lu for t in dl_tokens) and not noise:
            download_like.append(u)
        if any(t in lu for t in doc_tokens):
            doc_like.append(u)
        if (
            "amazonaws.com" in lu
            or ".s3." in lu
            or "s3.amazonaws.com" in lu
            or lu.startswith("s3://")
            or "cloudfront.net" in lu
            or "storage.googleapis.com" in lu
            or "blob.core.windows.net" in lu
        ) and not noise:
            s3_like.append(u)

    # Common accession / DOI patterns in page text
    text_sample = blob[:80_000]
    empiar_ids = sorted(set(re.findall(r"EMPIAR-\d+", text_sample, flags=re.IGNORECASE)))
    emdb_ids = sorted(set(re.findall(r"EMD-\d+", text_sample, flags=re.IGNORECASE)))
    dois = sorted(set(re.findall(r"10\.\d{4,}/[^\s\"'<>)\]]+", text_sample)))

    return {
        "page_url": page_url,
        "api_like": sorted(set(api_like))[:25],
        "download_like": sorted(set(download_like))[:25],
        "doc_like": sorted(set(doc_like))[:20],
        "same_host_sample": sorted(set(same_host))[:40],
        "s3_like": sorted(set(s3_like))[:25],
        "empiar_ids": empiar_ids[:15],
        "emdb_ids": emdb_ids[:15],
        "dois": dois[:15],
        "base_root": base_root,
    }


def merge_site_access_signals(page_results: list[dict]) -> dict:
    """Merge per-page signals into one site-level bundle."""
    merged = {
        "api_like": set(),
        "download_like": set(),
        "doc_like": set(),
        "same_host": set(),
        "s3_like": set(),
        "empiar_ids": set(),
        "emdb_ids": set(),
        "dois": set(),
        "page_urls": [],
    }
    for pr in page_results:
        if not pr.get("ok"):
            continue
        merged["page_urls"].append(pr.get("url", ""))
        sig = pr.get("_signals") or {}
        merged["api_like"].update(sig.get("api_like", []))
        merged["download_like"].update(
            u for u in sig.get("download_like", []) if not is_noise_preview_media_url(u)
        )
        merged["doc_like"].update(sig.get("doc_like", []))
        merged["same_host"].update(
            u for u in sig.get("same_host_sample", []) if not is_noise_preview_media_url(u)
        )
        merged["s3_like"].update(
            u for u in sig.get("s3_like", []) if not is_noise_preview_media_url(u)
        )
        merged["empiar_ids"].update(sig.get("empiar_ids", []))
        merged["emdb_ids"].update(sig.get("emdb_ids", []))
        merged["dois"].update(sig.get("dois", []))

    return {
        "pages_crawled": merged["page_urls"],
        "api_like": sorted(merged["api_like"])[:40],
        "download_like": sorted(merged["download_like"])[:40],
        "doc_like": sorted(merged["doc_like"])[:30],
        "same_host_urls": sorted(merged["same_host"])[:60],
        "s3_like": sorted(merged["s3_like"])[:35],
        "empiar_ids": sorted(merged["empiar_ids"])[:25],
        "emdb_ids": sorted(merged["emdb_ids"])[:25],
        "dois": sorted(merged["dois"])[:25],
    }


def generic_bulk_em_download_guidance_md() -> str:
    """Best-practice checklist (like a technical handoff). Always label as non-site-specific."""
    return """
### Reference: large EM / volume / organelle data (verify on official docs)

These steps mirror how practitioners usually acquire data; **confirm every bucket path, API, and policy on the provider’s site.**

1. **Decide what you need before downloading**
   - Raw image volume vs **segmentation / labels** (organelle masks, meshes, etc.).
   - **Full dataset vs subvolume / crop** (resolution level, bounding box).
   - **Format for your pipeline:** array containers (**Zarr**, **N5**), **MRC**, **OME-TIFF**, etc. vs exported stacks.

2. **Discovery**
   - Use the **web portal** to browse and note **dataset name, accession, DOI, or catalog ID**.
   - Prefer **official “Download”, “API”, or “Documentation”** pages over third-party summaries.

3. **Where data usually lives**
   - **Object storage** (e.g. **Amazon S3**, GCS, Azure Blob) with **prefixes** per dataset — often downloadable via **AWS CLI**, **rclone**, or SDKs when policy allows.
   - **HTTP** direct files or **REST/GraphQL APIs** for metadata + signed URLs.

4. **Bulk download (typical patterns — only if docs allow)**
   - **AWS CLI:** `aws s3 ls s3://…/prefix/` then **targeted** `aws s3 cp` / `sync` on a prefix — avoid mirroring entire buckets unless intended.
   - **API + scripts:** list datasets, then fetch manifests or per-file URLs.

5. **Mitochondria-focused pulls**
   - Filter catalog/search by **mitochondria**, **mito**, **organelle**, **FIB-SEM**, **cryo-ET**, **segmentation**, or ontology tags if the site exposes them.

6. **Scale & etiquette**
   - EM volumes are **huge**; download **only** needed resolution levels and regions.
   - Respect **robots.txt**, **terms of use**, and **rate limits**.
""".strip()



def _inventory_line_dataset_name(row: str) -> str | None:
    m = re.match(r"dataset_name=([^|]+)", (row or "").strip())
    if not m:
        return None
    return m.group(1).strip()


def _primary_catalog_inventory(inv: list[str], catalog_names: list[str]) -> list[str]:
    """One row per top-level catalog slug (for DB schema), catalog order."""
    if not inv:
        return []
    want = {n.strip().lower() for n in catalog_names if isinstance(n, str) and n.strip()}
    if not want:
        return list(inv)
    by_lower: dict[str, str] = {}
    for row in inv:
        if not isinstance(row, str):
            continue
        dn = _inventory_line_dataset_name(row)
        if dn and dn.lower() in want:
            by_lower[dn.lower()] = row
    out: list[str] = []
    for n in catalog_names:
        if not isinstance(n, str):
            continue
        k = n.strip().lower()
        if k in by_lower:
            out.append(by_lower[k])
    return out


def format_data_acquisition_playbook_md(
    website_name: str,
    root_url: str,
    access: dict,
    heuristic: dict | None = None,
) -> str:
    """Appendix focused on **per-dataset** spatial + segmentation fields for future DB schema."""
    _ = (website_name, heuristic)  # API compat; not echoed in appendix

    lines: list[str] = [
        "## Appendix: Dataset catalog (per-dataset → schema)",
        "",
        "Each **primary** row is one catalog dataset: **`grid.*`** (axes, spacing, dimensions), plus "
        "**`ground_truths=` / `predictions=` / `raw_or_other_layers=`** and **`mesh:` / `image:`** tokens from PostgREST "
        "``image``+``mesh`` embed (tokens may include **`[url=…]`**, **`[content_type=…]`** when the API exposes them). "
        "Dataset-level **`download_volume_url=`**, **`download_mito_mask_url=`**, **`download_mito_prediction_url=`** "
        "(plus mito quality/kind hints such as **`download_mito_mask_quality=`** — only **`good`** is a non-prediction training mask — "
        "**`download_mito_mask_segmentation_kind=`**, and **`good_mitochondria_gt_mask=yes`** when a good mask URL was chosen) "
        "summarize default EM vs mitochondria mask sources for scripting. "
        "Layer tokens may include **`mito_label_quality=non_ground_mito`** when a mito-like layer is not in an allowed GT/segmentation bucket. "
        "Row provenance may include a **PostgREST URL** (truncated); "
        "always pair it with Appendix **`apikey`** + **Bearer** — opening the URL alone in a browser will error.",
        "",
        f"- **Source:** `{root_url}`",
        "",
    ]

    if access.get("spa_target_path"):
        lines.append(
            f"- **SPA:** `{access.get('spa_target_path')}` (shell `{access.get('spa_entry_url')}`)"
        )
        lines.append("")

    if access.get("supabase_apikeys"):
        lines.extend(
            [
                "### Programmatic access",
                "PostgREST `dataset` table: send **headers** `apikey` and `Authorization: Bearer <same>` on every request. "
                "Do not paste the REST URL alone in a browser.",
                "",
            ]
        )
        for k in access.get("supabase_apikeys", [])[:2]:
            lines.append(f"- `{k}`")
        lines.append("")

    ui = access.get("dataset_ui_stats") or {}
    bulk = access.get("supabase_bulk_meta") or {}
    names = access.get("dataset_names_sorted") or []
    spatial = access.get("dataset_spatial_rows") or []
    cats = access.get("dataset_category_counts") or {}
    py_md = (access.get("python_access_markdown_block") or "").strip()
    inv = [x for x in (access.get("dataset_inventory") or []) if isinstance(x, str)]

    primary = _primary_catalog_inventory(inv, names)
    cap = appendix_cap("APPENDIX_DATASET_ROWS")
    primary_show = primary[:cap] if primary else []
    name_list_cap = appendix_cap("APPENDIX_DATASET_NAME_LIST_CAP")

    # Full catalog slug index (numbered) — separate from detailed "Primary datasets" rows below.
    if names:
        lines.extend(
            [
                "### All datasets on the site (catalog slug list)",
                "",
                f"Ordered list of **{len(names)}** dataset slug(s) from the website catalog "
                "(PostgREST `dataset` name index). "
                "This is **names only**; merged `dataset_name=` schema rows follow under "
                "**Primary datasets (spatial + segmentation / annotation layers)**.",
                "",
            ]
        )
        listed = [
            n.strip()
            for n in names[:name_list_cap]
            if isinstance(n, str) and n.strip()
        ]
        for i, n in enumerate(listed, start=1):
            lines.append(f"{i}. **`{n}`**")
        if len(names) > name_list_cap:
            lines.append(
                f"_…{len(names) - name_list_cap} more slug(s) not shown "
                f"(set `APPENDIX_DATASET_NAME_LIST_CAP` ≥ {len(names)} for the full list)._"
            )
        lines.append("")

    if primary_show:
        lines.extend(
            [
                "### Primary datasets (spatial + segmentation / annotation layers)",
                "",
                f"**{len(primary_show)}** merged inventory line(s) with `dataset_name=` — "
                f"grid, layers, and `download_*` fields where present "
                f"(**{len(names)}** slugs in REST index; nested layer-only rows omitted). "
                "See **All datasets on the site** above for the complete numbered slug list.",
                "",
            ]
        )
        for row in primary_show:
            lines.append(f"- {row}")
        if len(primary) > cap:
            lines.append(f"- _…+{len(primary) - cap} more (`APPENDIX_DATASET_ROWS`)_")
        if names and len(primary) < len(names):
            lines.append(
                f"- _{len(names) - len(primary)} catalog slug(s) had no merged inventory row "
                "(subset may lack embed overlap in this run)._"
            )
        lines.append("")
        # Canonical machine block (stable section markers for downstream parsers).
        lines.extend(
            [
                "### Canonical dataset rows (machine, v1)",
                "",
                "```text",
                "BEGIN_DATASET_ROWS_V1",
            ]
        )
        for row in primary_show:
            lines.append(row)
        lines.extend(["END_DATASET_ROWS_V1", "```", ""])
    elif inv:
        lines.extend(
            [
                "### Primary datasets (spatial + segmentation / annotation layers)",
                "",
                "_No match to `dataset_names_sorted` — showing unfiltered inventory "
                "(slug list above still reflects REST names when present)._",
                "",
            ]
        )
        for row in inv[:cap]:
            lines.append(f"- {row}")
        if len(inv) > cap:
            lines.append(f"- _…+{len(inv) - cap} more (`APPENDIX_DATASET_ROWS`)_")
        lines.append("")
        lines.extend(
            [
                "### Canonical dataset rows (machine, v1)",
                "",
                "```text",
                "BEGIN_DATASET_ROWS_V1",
            ]
        )
        for row in inv[:cap]:
            lines.append(row)
        lines.extend(["END_DATASET_ROWS_V1", "```", ""])

    if ui or bulk or cats:
        lines.extend(["### Fetch roll-up (counts only)", ""])
        if ui.get("ui_total") is not None:
            lines.append(
                f"- **UI:** {ui.get('ui_total')} datasets — _{ui.get('ui_range_label', '')}_"
            )
        if bulk.get("row_count") is not None:
            lines.append(
                f"- **REST rows pulled:** **{bulk.get('row_count')}** ({bulk.get('requests', '?')} requests)"
            )
        if bulk.get("error") and not bulk.get("ok"):
            lines.append(f"- **REST error:** `{str(bulk.get('error'))[:160]}`")
        if cats:
            top = list(cats.items())[:8]
            lines.append(
                "- **`sample.*` (top):** " + "; ".join(f"`{k}`×{v}" for k, v in top)
            )
        lines.append("")

    seg = access.get("segmentation_annotation_schema") or {}
    seg_lines = seg.get("markdown_lines") if isinstance(seg, dict) else None
    if seg_lines:
        lines.extend(
            [
                "### Cross-dataset stats (not a schema — use primary rows above)",
                "",
            ]
        )
        for ln in seg_lines:
            lines.append(ln)
        lines.append("")

    ui_seg = access.get("dataset_ui_segmentation_layers") or []
    if ui_seg:
        cap_u = appendix_cap("APPENDIX_UI_SEGMENTATION_ROWS")
        n_mito_ui = sum(
            1 for r in ui_seg if isinstance(r, dict) and r.get("has_mitochondria_ui")
        )
        lines.extend(
            [
                "### UI: Segmentation layer labels (dataset pages)",
                "",
                "Optional human-readable checklist from the viewer (when parsed).",
                f"- **Pages sampled:** {len(ui_seg)} — **mito in UI list:** {n_mito_ui}",
                "",
            ]
        )
        for r in ui_seg[:cap_u]:
            if not isinstance(r, dict):
                continue
            ds = str(r.get("dataset_name") or "?")
            err = r.get("error")
            if err:
                lines.append(f"- `{ds}` — _error: {str(err)[:120]}_")
                continue
            labs = r.get("ui_segmentation_labels") or []
            if not isinstance(labs, list):
                labs = []
            labs_s = [str(x).strip() for x in labs if str(x).strip()]
            if not labs_s:
                lines.append(f"- `{ds}` — _no labels parsed_")
                continue
            cap_lab = 14
            head = "; ".join(labs_s[:cap_lab])
            tail = f" _…+{len(labs_s) - cap_lab} more_" if len(labs_s) > cap_lab else ""
            mtag = " **mito✓**" if r.get("has_mitochondria_ui") else ""
            lines.append(f"- `{ds}`{mtag}: {head}{tail}")
        if len(ui_seg) > cap_u:
            lines.append(f"- _…+{len(ui_seg) - cap_u} more (`APPENDIX_UI_SEGMENTATION_ROWS`)_")
        lines.append("")

    if spatial and not primary_show:
        cap_sp = appendix_cap("APPENDIX_SPATIAL_ROWS_CAP")
        lines.extend(
            [
                "### Size & spacing (REST `image_acquisition` only)",
                "",
            ]
        )
        for row in spatial[:cap_sp]:
            lines.append(f"- {row}")
        if len(spatial) > cap_sp:
            lines.append(f"- _…+{len(spatial) - cap_sp} more_")
        lines.append("")

    if py_md:
        lines.extend(py_md.splitlines())
        lines.append("")

    lines.extend(
        [
            "### Note",
            "- **All datasets on the site** is the complete slug index (numbered). "
            "Build your DB schema from **Primary datasets** lines (one merged row per catalog slug "
            "where inventory was produced). "
            "Re-fetch via PostgREST with headers when you need machine-readable JSON.",
            "",
        ]
    )
    return "\n".join(lines)
