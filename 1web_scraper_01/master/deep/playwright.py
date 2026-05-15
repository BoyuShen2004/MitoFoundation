"""Playwright SPA capture, JSON inventory, and merge helpers."""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from .config import HTTP_HEADERS, RAW_HTML_SAMPLE, _env_truthy
from .appendix_caps import appendix_cap
from .signals import extract_access_signals, is_noise_preview_media_url
from .supabase_bulk import (
    aggregate_segmentation_annotation_schema,
    fetch_dataset_rows_paginated,
    resolve_dataset_rest_url,
    names_from_inventory_lines,
    parse_openorganelle_dataset_ui_totals,
    python_access_markdown,
    sample_type_category_counts,
    spatial_summary_lines,
    sorted_unique_dataset_slugs,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INV_ROOT = _REPO_ROOT / "0inventory"
if str(_INV_ROOT) not in sys.path:
    sys.path.insert(0, str(_INV_ROOT))
from dataset_inventory import _extract_dataset_inventory


def _response_looks_like_json_api(url: str, content_type: str) -> bool:
    u = url.lower()
    ct = content_type.lower()
    if "json" in ct or "graphql" in ct:
        return True
    if u.rstrip("/").endswith(".json"):
        return True
    if "/api/" in u or "/v1/" in u or "/v2/" in u or "graphql" in u:
        return True
    return False


def _origin_base_url(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}/"


def _spa_client_path_from_url(url: str) -> str | None:
    """Path (+ query) to reach via client-side routing; None if root URL only."""
    if os.getenv("PLAYWRIGHT_SPA_CLIENT_NAV", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return None
    p = urlparse(url)
    path = (p.path or "").strip()
    if path in ("", "/"):
        if not (p.query or "").strip():
            return None
        return "/?" + p.query
    if not path.startswith("/"):
        path = "/" + path
    if p.query:
        return path + "?" + p.query
    return path


def playwright_spa_client_navigate(page, target_path: str) -> str:
    """Reach an in-app route without document navigation (avoids server 404 on deep links).

    Tries: click matching <a href>, then history.pushState + popstate (React Router, etc.).
    """
    if not target_path or target_path == "/":
        return "skip"
    click_timeout = int(os.getenv("PLAYWRIGHT_SPA_CLICK_TIMEOUT_MS", "6000"))
    post_click = int(os.getenv("PLAYWRIGHT_SPA_POST_CLICK_MS", "4500"))
    post_push = int(os.getenv("PLAYWRIGHT_SPA_PUSHSTATE_WAIT_MS", "6000"))

    path_only = target_path.split("?", 1)[0]
    variants = []
    for base in (path_only, path_only.rstrip("/"), path_only.rstrip("/") + "/"):
        if base not in variants:
            variants.append(base)
    if "?" in target_path:
        variants.append(target_path)

    for href in variants:
        if not href or href == "/":
            continue
        try:
            escaped = href.replace("\\", "\\\\").replace('"', '\\"')
            loc = page.locator(f"a[href=\"{escaped}\"]").first
            loc.click(timeout=click_timeout)
            page.wait_for_timeout(post_click)
            return f"click:{href}"
        except Exception:
            pass

    try:
        needle = path_only.rstrip("/").split("/")[-1] or path_only
        if needle and needle != "/":
            loc = page.locator(f"a[href*='{needle}']").first
            loc.click(timeout=click_timeout)
            page.wait_for_timeout(post_click)
            return "click:partial"
    except Exception:
        pass

    try:
        page.evaluate(
            """(path) => {
                try {
                  window.history.pushState({}, '', path);
                  window.dispatchEvent(new PopStateEvent('popstate', { state: history.state }));
                } catch (e) {}
            }""",
            target_path,
        )
        page.wait_for_timeout(post_push)
        return "pushstate"
    except Exception:
        pass

    return "failed"


def playwright_extract_dom_category_labels_and_urls(page) -> tuple[List[str], List[str]]:
    """Extract visible catalog/category labels and their in-app links from the rendered DOM.

    This is tailored for OpenOrganelle-style SPAs where the organelle grid cards live on
    routes like ``/organelles/<slug>`` but card titles may not correspond to separate URLs.

    Returns:
      - labels: visible human-readable organelle/category titles
      - urls: absolute links (same origin) matching ``/organelles/<...>`` and ``/collections/<...>``
    """
    try:
        raw = page.evaluate(
            """() => {
              const STOP = new Set([
                'Datasets', 'News', 'Organelles', 'Measurements', 'Publications', 'FAQ',
                'Home', 'About', 'Help', 'OpenOrganelle', 'Open Organelle',
                'Show More', 'Search', 'Menu', 'Settings', 'Loading', 'Error', 'Log in', 'Sign in',
                'Janelia Research Campus', 'HHMI'
              ]);
              const seen = new Set();
              const seenU = new Set();
              const labels = [];
              const urls = [];

              const pushLabel = (t) => {
                if (!t) return;
                t = String(t).replace(/\\s+/g, ' ').trim();
                if (t.length < 3 || t.length > 120) return;
                if (STOP.has(t)) return;
                if (/^\\d+$/.test(t)) return;
                const low = t.toLowerCase();
                if (seen.has(low)) return;
                seen.add(low);
                labels.push(t);
              };

              const pushUrl = (u) => {
                if (!u) return;
                u = String(u).split('#')[0];
                const low = u.toLowerCase();
                if (seenU.has(low)) return;
                seenU.add(low);
                urls.push(u);
              };

              const origin = location.origin;
              for (const a of document.querySelectorAll('a[href]')) {
                const hrefRaw = (a.getAttribute('href') || '').split('?')[0].split('#')[0];
                const txt = (a.textContent || '').trim().replace(/\\s+/g, ' ');
                if (!txt || txt.length > 100) continue;
                try {
                  const abs = new URL(hrefRaw, origin).href;
                  if (/\\/organelles\\/[^/]+/i.test(abs) || /\\/collections\\/[^/]+/i.test(abs)) {
                    pushUrl(abs);
                    pushLabel(txt.split('\\n')[0].trim());
                  }
                } catch (e) {}
              }

              const root = document.querySelector('main') || document.body;
              for (const sel of ['h1', 'h2', 'h3', 'h4']) {
                root.querySelectorAll(sel).forEach(el => {
                  const t = (el.textContent || '').trim().split('\\n')[0].trim();
                  pushLabel(t);
                });
              }

              document.querySelectorAll('div, article, section, li').forEach(box => {
                const imgs = box.querySelectorAll('img[src], picture img[src]');
                if (imgs.length < 4 || imgs.length > 24) return;
                const r = box.getBoundingClientRect();
                if (r.height < 100 || r.height > 650 || r.width < 80) return;
                const lines = (box.innerText || '').split('\\n').map(s => s.trim()).filter(Boolean);
                for (const line of lines) {
                  if (/show more/i.test(line)) continue;
                  if (line.length >= 3 && line.length <= 90) { pushLabel(line); break; }
                }
              });

              return { labels, urls };
            }"""
        )
    except Exception:
        return [], []

    if not isinstance(raw, dict):
        return [], []

    labels_raw = raw.get("labels") or []
    urls_raw = raw.get("urls") or []
    labels: List[str] = []
    urls: List[str] = []

    cap_labels = int(os.getenv("PLAYWRIGHT_DOM_LABEL_CAP", "400"))
    cap_urls = int(os.getenv("PLAYWRIGHT_DOM_URL_CAP", "200"))

    seen_lab: set[str] = set()
    for x in labels_raw:
        if not isinstance(x, str):
            continue
        t = " ".join(x.split()).strip()
        if len(t) < 3:
            continue
        k = t.lower()
        if k in seen_lab:
            continue
        seen_lab.add(k)
        labels.append(t)
        if len(labels) >= cap_labels:
            break

    seen_url: set[str] = set()
    for u in urls_raw:
        if not isinstance(u, str):
            continue
        su = u.split("#")[0].strip()
        if not su:
            continue
        k = su.lower()
        if k in seen_url:
            continue
        seen_url.add(k)
        urls.append(su)
        if len(urls) >= cap_urls:
            break

    return labels, urls


def _playwright_failure_flags(error_text: str) -> dict:
    """Classify Playwright launch/setup errors for clearer user messaging."""
    s = error_text.lower()
    browser_missing = (
        "executable doesn't exist" in s
        or "please run the following command" in s
        or "playwright install" in s
        or "chromium_headless_shell" in s
        or "chrome-headless-shell" in s
    )
    return {"browser_missing": browser_missing}


_MITO_UI = re.compile(r"mito|mitochond", re.IGNORECASE)


def playwright_fetch_ui_segmentation_layers_for_slugs(
    site_origin: str,
    dataset_slugs: list[str],
    *,
    timeout_ms: int = 55_000,
    per_page_settle_ms: int = 3500,
) -> list[dict]:
    """Open dataset pages like ``/datasets/<slug>`` and read **Segmentation Layers** checkbox labels.

    Complements PostgREST: UI often shows human-readable names (e.g. *Mitochondria*) while
    REST ``mesh.name`` may be ``mito_seg``.
    """
    out: list[dict] = []
    if not site_origin or not dataset_slugs:
        return out
    origin = site_origin.rstrip("/")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return out

    launch_kwargs = {"headless": True}
    if os.getenv("PLAYWRIGHT_NO_SANDBOX", "0").lower() in ("1", "true", "yes"):
        launch_kwargs["args"] = ["--no-sandbox", "--disable-dev-shm-usage"]

    js_extract = r"""() => {
      const junk = new Set([
        'Segmentation Layers', 'Ground Truth', 'Predictions', 'Show More', 'Loading',
        'Error', 'Datasets', 'Organelles', 'Open Organelle', 'OpenOrganelle'
      ]);
      const labels = [];
      const headers = Array.from(document.querySelectorAll('h1,h2,h3,h4,h5,h6,button,div,span,p'));
      let panel = null;
      for (const h of headers) {
        const raw = (h.textContent || '').trim().replace(/\s+/g, ' ');
        if (!/^segmentation layers/i.test(raw) || raw.length > 80) continue;
        let p = h;
        for (let depth = 0; depth < 14 && p; depth++) {
          const cbs = p.querySelectorAll('input[type="checkbox"]');
          const labs = p.querySelectorAll('label');
          if (cbs.length >= 2 && labs.length >= 2) { panel = p; break; }
          p = p.parentElement;
        }
        if (panel) break;
      }
      if (!panel) return labels;
      for (const lab of panel.querySelectorAll('label')) {
        let t = (lab.innerText || '').trim().replace(/\s+/g, ' ');
        if (t.length < 2 || t.length > 180) continue;
        if (/^segmentation layers$/i.test(t)) continue;
        if (junk.has(t)) continue;
        labels.push(t);
      }
      return [...new Set(labels)];
    }"""

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(**launch_kwargs)
            context = browser.new_context(
                user_agent=HTTP_HEADERS["User-Agent"],
                viewport={"width": 1400, "height": 950},
            )
            page = context.new_page()
            for slug in dataset_slugs:
                if not isinstance(slug, str) or not slug.strip():
                    continue
                s = slug.strip()
                target = f"{origin}/datasets/{s}"
                row: dict = {"dataset_name": s, "ui_segmentation_labels": [], "error": None}
                try:
                    page.goto(target, wait_until="domcontentloaded", timeout=timeout_ms)
                    nidle = int(os.getenv("PLAYWRIGHT_UI_SEG_NETWORKIDLE_MS", "12000"))
                    if nidle > 0:
                        try:
                            page.wait_for_load_state("networkidle", timeout=nidle)
                        except Exception:
                            pass
                    page.wait_for_timeout(per_page_settle_ms)
                    raw = page.evaluate(js_extract)
                    labs = [x for x in raw if isinstance(x, str)] if isinstance(raw, list) else []
                    row["ui_segmentation_labels"] = labs
                    row["has_mitochondria_ui"] = any(
                        _MITO_UI.search(x) for x in labs
                    )
                except Exception as e:
                    row["error"] = str(e)[:240]
                out.append(row)
            browser.close()
    except Exception:
        pass
    return out


def playwright_capture(url: str, run_supabase_bulk: bool = True) -> dict:
    """Headless Chromium: render SPA, scroll, capture JSON responses, extract catalog rows."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        return {
            "ok": False,
            "error": (
                "Python package 'playwright' not importable. Run:\n"
                "  pip install playwright\n"
                "  python -m playwright install chromium"
            ),
            "import_error": str(e),
            "package_missing": True,
            "browser_missing": False,
        }

    timeout_ms = int(os.getenv("PLAYWRIGHT_TIMEOUT_MS", "120000"))
    scroll_rounds = int(os.getenv("PLAYWRIGHT_SCROLL_ROUNDS", "22"))
    max_json_captures = int(os.getenv("PLAYWRIGHT_MAX_JSON_RESPONSES", "220"))
    is_oo_fast = "openorganelle.janelia.org" in (url or "").lower() and _env_truthy(
        "OPENORGANELLE_FAST_SCRAPE", "1"
    )
    if is_oo_fast:
        scroll_rounds = min(
            scroll_rounds,
            max(3, int(os.getenv("OPENORGANELLE_FAST_SCROLL_ROUNDS", "8"))),
        )
        max_json_captures = min(
            max_json_captures,
            max(40, int(os.getenv("OPENORGANELLE_FAST_MAX_JSON_RESPONSES", "96"))),
        )
    json_rows: List[tuple[str, object]] = []
    xhr_urls: List[str] = []
    seen_urls: set = set()
    supabase_apikeys: set[str] = set()
    supabase_authorizations: set[str] = set()
    supabase_auth_urls: set[str] = set()

    def on_request(request) -> None:
        try:
            u = request.url or ""
            ul = u.lower()
            if "supabase.co" not in ul:
                return
            if ("/rest/v1/" not in ul) and ("/storage/" not in ul):
                return
            headers = request.headers or {}
            # Playwright provides header names in lower-case.
            apikey = headers.get("apikey") or headers.get("apiKey") or ""
            auth = headers.get("authorization") or headers.get("Authorization") or ""
            if apikey:
                supabase_apikeys.add(str(apikey))
                supabase_auth_urls.add(u.split("#")[0])
            if auth:
                supabase_authorizations.add(str(auth))
                supabase_auth_urls.add(u.split("#")[0])
        except Exception:
            pass

    def on_response(response) -> None:
        if len(json_rows) >= max_json_captures:
            return
        try:
            u = response.url
            ul = u.lower()
            if any(
                x in ul
                for x in (
                    "analytics",
                    "segment.io",
                    "doubleclick",
                    "google-analytics",
                    "sentry",
                    "hotjar",
                    "facebook.net",
                    "googletagmanager",
                    "doubleclick.net",
                )
            ):
                return
            if is_noise_preview_media_url(u):
                return
            ct = response.headers.get("content-type") or ""
            if ct.lower().startswith("image/"):
                return
            body = response.text()
            if len(body) > 1_800_000:
                return
            if not _response_looks_like_json_api(u, ct):
                stripped = body.lstrip()
                if not (stripped.startswith("{") or stripped.startswith("[")):
                    return
                low = stripped[:200].lower()
                if low.startswith("<!doctype") or low.startswith("<html"):
                    return
            data = json.loads(body)
            if isinstance(data, dict):
                msg = str(data.get("message") or "")
                hint = str(data.get("hint") or "")
                # Common Supabase failure when you call the REST URL directly without the required header.
                if "no api key found" in (msg + " " + hint).lower():
                    return
            if u not in seen_urls:
                seen_urls.add(u)
                xhr_urls.append(u)
            json_rows.append((u, data))
        except Exception:
            pass

    launch_kwargs = {"headless": True}
    if os.getenv("PLAYWRIGHT_NO_SANDBOX", "0").lower() in ("1", "true", "yes"):
        launch_kwargs["args"] = ["--no-sandbox", "--disable-dev-shm-usage"]

    internal_urls: List[str] = []
    dom_category_labels: List[str] = []
    dom_category_urls: List[str] = []
    spa_target = _spa_client_path_from_url(url)
    entry_url = _origin_base_url(url) if spa_target else url
    spa_nav_method = ""

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(**launch_kwargs)
            context = browser.new_context(
                user_agent=HTTP_HEADERS["User-Agent"],
                viewport={"width": 1400, "height": 900},
            )
            page = context.new_page()
            page.on("response", on_response)
            page.on("request", on_request)
            page.goto(entry_url, wait_until="domcontentloaded", timeout=timeout_ms)
            nidle_ms = int(os.getenv("PLAYWRIGHT_NETWORKIDLE_MS", "15000"))
            if nidle_ms > 0:
                try:
                    page.wait_for_load_state("networkidle", timeout=nidle_ms)
                except Exception:
                    pass

            if spa_target:
                page.wait_for_timeout(
                    int(os.getenv("PLAYWRIGHT_PRE_SPA_NAV_MS", "5000"))
                )
                spa_nav_method = playwright_spa_client_navigate(page, spa_target)
                if spa_nav_method == "failed":
                    print(
                        f"[WARN] SPA client-side navigation to {spa_target!r} failed "
                        f"(no matching link / pushState). Page may still be on {entry_url!r}."
                    )
                extra_nidle = int(os.getenv("PLAYWRIGHT_SPA_POST_NAV_NETWORKIDLE_MS", "12000"))
                if extra_nidle > 0:
                    try:
                        page.wait_for_load_state("networkidle", timeout=extra_nidle)
                    except Exception:
                        pass
                page.wait_for_timeout(
                    int(os.getenv("PLAYWRIGHT_SPA_POST_NAV_EXTRA_MS", "5000"))
                )

            page.wait_for_timeout(int(os.getenv("PLAYWRIGHT_INITIAL_WAIT_MS", "7000")))
            for _ in range(scroll_rounds):
                page.evaluate("window.scrollBy(0, Math.floor(window.innerHeight * 0.92))")
                page.wait_for_timeout(int(os.getenv("PLAYWRIGHT_SCROLL_PAUSE_MS", "650")))
            page.wait_for_timeout(int(os.getenv("PLAYWRIGHT_POST_SCROLL_MS", "3500")))
            try:
                internal_urls = page.evaluate(
                    """() => {
                        const origin = location.origin;
                        const seen = new Set();
                        const out = [];
                        for (const a of document.querySelectorAll('a[href]')) {
                          const href = a.getAttribute('href');
                          if (!href || href.startsWith('mailto:') || href.startsWith('tel:') ||
                              href.startsWith('javascript:')) continue;
                          try {
                            const u = new URL(href, origin).href.split('#')[0];
                            if (u.startsWith(origin) && !seen.has(u)) {
                              seen.add(u); out.push(u);
                            }
                          } catch (e) {}
                        }
                        return out;
                    }"""
                )
            except Exception:
                internal_urls = []
            if isinstance(internal_urls, list):
                internal_urls = [
                    u
                    for u in internal_urls
                    if isinstance(u, str) and not is_noise_preview_media_url(u)
                ]
            else:
                internal_urls = []
            try:
                dom_category_labels, dom_category_urls = (
                    playwright_extract_dom_category_labels_and_urls(page)
                )
            except Exception:
                dom_category_labels = []
                dom_category_urls = []
            if isinstance(dom_category_urls, list):
                dom_category_urls = [
                    u
                    for u in dom_category_urls
                    if isinstance(u, str) and not is_noise_preview_media_url(u)
                ]
            html = page.content()
            try:
                title = page.title()
            except Exception:
                title = "Unknown"
            try:
                body_text = page.inner_text("body", timeout=20000)
            except Exception:
                body_text = ""
            browser.close()
    except Exception as exc:
        err_s = str(exc)
        flags = _playwright_failure_flags(err_s)
        return {
            "ok": False,
            "error": err_s,
            "browser_missing": flags["browser_missing"],
            "package_missing": False,
        }

    ui_stats = parse_openorganelle_dataset_ui_totals(body_text or "")

    inv_limit = int(os.getenv("PLAYWRIGHT_INVENTORY_PARSE_LIMIT", "1200"))
    best_inventory: dict[str, str] = {}
    for ju, data in json_rows:
        _extract_dataset_inventory(data, ju, best_inventory, inv_limit)

    bulk_rows: list[dict] = []
    bulk_meta: dict = {"ok": False}
    rest_u, rest_u_source = resolve_dataset_rest_url(
        xhr_urls,
        list(sorted(supabase_auth_urls)),
        list(sorted(supabase_apikeys)),
    )
    if (
        run_supabase_bulk
        and rest_u
        and rest_u_source != "xhr_json_capture"
    ):
        print(
            f"[INFO] Supabase dataset REST URL resolved via {rest_u_source}: "
            f"{rest_u[:100]}{'…' if len(rest_u) > 100 else ''}"
        )
    if (
        run_supabase_bulk
        and rest_u
        and _env_truthy("SUPABASE_BULK_FETCH_DATASETS", "1")
    ):
        bulk_rows, bulk_meta = fetch_dataset_rows_paginated(
            rest_u,
            list(sorted(supabase_apikeys)),
            list(sorted(supabase_authorizations)),
        )
        bulk_meta["rest_url_source"] = rest_u_source
        if not bulk_meta.get("ok") and bulk_meta.get("error"):
            print(f"[WARN] Supabase bulk dataset fetch: {bulk_meta.get('error')}")
        bulk_label = (bulk_meta.get("url_used") or rest_u)[:180]
        for item in bulk_rows:
            if isinstance(item, dict):
                _extract_dataset_inventory(item, bulk_label, best_inventory, inv_limit)

    catalog = [best_inventory[k] for k in sorted(best_inventory.keys(), key=str.lower)]

    names_sorted = sorted_unique_dataset_slugs(bulk_rows)
    if not names_sorted:
        names_sorted = names_from_inventory_lines(catalog)
    ui_layer_rows: list[dict] = []
    raw_ui = os.getenv("PLAYWRIGHT_UI_SEGMENTATION_LAYERS", "").strip().lower()
    if raw_ui in ("0", "false", "no", "off"):
        ui_enabled = False
    elif raw_ui in ("1", "true", "yes", "on"):
        ui_enabled = True
    else:
        # Default off: per-dataset UI visits are slow; PostgREST bulk + appendix rows
        # are enough for ``2database_builder``. Set ``PLAYWRIGHT_UI_SEGMENTATION_LAYERS=1``
        # (optionally with ``PLAYWRIGHT_UI_SEGMENTATION_MAX_DATASETS``) to restore UI probes.
        ui_enabled = False
    max_ui = max(0, int(os.getenv("PLAYWRIGHT_UI_SEGMENTATION_MAX_DATASETS", "25")))
    settle = int(os.getenv("PLAYWRIGHT_UI_SEGMENTATION_SETTLE_MS", "3500"))
    base_ui = _origin_base_url(url).rstrip("/")
    if ui_enabled and max_ui > 0 and names_sorted and base_ui:
        ui_layer_rows = playwright_fetch_ui_segmentation_layers_for_slugs(
            base_ui,
            names_sorted[:max_ui],
            timeout_ms=int(os.getenv("PLAYWRIGHT_UI_SEGMENTATION_GOTO_TIMEOUT_MS", "55000")),
            per_page_settle_ms=settle,
        )
    spatial_rows = spatial_summary_lines(bulk_rows)
    cat_counts = sample_type_category_counts(bulk_rows)
    py_block = python_access_markdown(rest_u) if (
        rest_u and _env_truthy("APPENDIX_INCLUDE_PYTHON_SNIPPET", "0")
    ) else ""
    segmentation_annotation_schema: dict = {}
    if bulk_rows:
        try:
            segmentation_annotation_schema = aggregate_segmentation_annotation_schema(
                bulk_rows
            )
        except Exception:
            segmentation_annotation_schema = {}

    catalog_summary_lines: List[str] = ["=== Dataset catalog summary (automated) ==="]
    if ui_stats.get("ui_total") is not None:
        catalog_summary_lines.append(
            f"UI: {ui_stats.get('ui_range_label', '')} (total datasets reported by site: {ui_stats['ui_total']})"
        )
    if bulk_meta.get("row_count") is not None:
        catalog_summary_lines.append(
            f"API: {bulk_meta.get('row_count')} dataset rows fetched via paginated Supabase REST"
        )
    if names_sorted:
        preview_n = int(os.getenv("PLAYWRIGHT_DATASET_NAME_PREVIEW", "50"))
        head = names_sorted[:preview_n]
        catalog_summary_lines.append(
            f"Dataset slugs ({len(names_sorted)} total; showing first {len(head)}): "
            + ", ".join(head)
        )
        if len(names_sorted) > preview_n:
            catalog_summary_lines.append(
                f"... +{len(names_sorted) - preview_n} more (full list in Appendix)."
            )
    if spatial_rows:
        catalog_summary_lines.append(
            f"Spatial rows (grid_dimensions / spacing): {len(spatial_rows)} (see Appendix)."
        )
    if cat_counts:
        top_c = list(cat_counts.items())[:12]
        catalog_summary_lines.append(
            "Sample/category hints (from JSON sample.*): "
            + "; ".join(f"{k}×{v}" for k, v in top_c)
        )
    if ui_layer_rows:
        n_mito_p = sum(1 for r in ui_layer_rows if r.get("has_mitochondria_ui"))
        catalog_summary_lines.append(
            f"UI Segmentation Layers panel: visited {len(ui_layer_rows)} dataset pages "
            f"({n_mito_p} with mitochondria in UI list; see Appendix)."
        )
    catalog_summary_text = "\n".join(catalog_summary_lines)

    return {
        "ok": True,
        "url": url,
        "html": html,
        "title": title,
        "body_text": body_text,
        "xhr_json_urls": xhr_urls,
        "json_response_count": len(json_rows),
        "dataset_inventory": catalog,
        "internal_urls": internal_urls if isinstance(internal_urls, list) else [],
        "spa_client_nav": spa_nav_method or None,
        "spa_entry_url": entry_url if spa_target else None,
        "spa_target_path": spa_target,
        "dom_category_labels": dom_category_labels,
        "dom_category_urls": dom_category_urls,
        "supabase_apikeys": sorted(supabase_apikeys),
        "supabase_authorizations": sorted(supabase_authorizations),
        "dataset_ui_stats": ui_stats,
        "supabase_bulk_meta": bulk_meta,
        "dataset_names_sorted": names_sorted,
        "dataset_spatial_rows": spatial_rows,
        "dataset_category_counts": cat_counts,
        "python_access_markdown_block": py_block,
        "dataset_catalog_summary_text": catalog_summary_text,
        "dataset_rest_url_for_python": rest_u,
        "segmentation_annotation_schema": segmentation_annotation_schema,
        "dataset_ui_segmentation_layers": ui_layer_rows,
    }


def _page_dict_from_playwright(pw: dict) -> dict:
    """Normalize Playwright output into the same shape as fetch_page()."""
    html = pw.get("html") or ""
    raw_html = html[:RAW_HTML_SAMPLE]
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    raw = soup.get_text(separator="\n", strip=True)
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    text = "\n".join(lines)

    inv = pw.get("dataset_inventory") or []
    inv_cap = 150
    inv_text = ""
    if inv:
        inv_text = (
            "\n\n=== Dataset / collection rows (from JSON APIs; best-effort) ===\n"
            + "\n".join(f"- {row}" for row in inv[: min(len(inv), inv_cap)])
        )
        if len(inv) > inv_cap:
            inv_text += (
                f"\n- ... ({len(inv) - inv_cap} more rows truncated; full list in Appendix)\n"
            )

    xhr = pw.get("xhr_json_urls") or []

    dom_labels = pw.get("dom_category_labels") or []
    dom_urls = pw.get("dom_category_urls") or []
    dom_text = ""
    if dom_labels:
        cap_d = int(os.getenv("PLAYWRIGHT_DOM_LABELS_IN_TEXT", "25"))
        dom_text = (
            "\n\n=== Visible categories / card labels (DOM) ===\n"
            + "\n".join(f"- {lab}" for lab in dom_labels[:cap_d])
        )
        if len(dom_labels) > cap_d:
            dom_text += f"\n- ... ({len(dom_labels) - cap_d} more labels truncated)\n"

    text_budget = int(os.getenv("PLAYWRIGHT_PAGE_TEXT_BUDGET", "20000"))
    catalog_head = (pw.get("dataset_catalog_summary_text") or "").strip()
    prefix = (catalog_head + "\n\n") if catalog_head else ""
    body = text
    body_max = int(os.getenv("PLAYWRIGHT_BODY_TEXT_MAX", "5000"))
    if len(body) > body_max:
        body = body[:body_max] + "\n...[page body truncated; see Appendix for datasets]\n"
    full_text = (prefix + body + dom_text + inv_text)[:text_budget]

    page = {
        "ok": True,
        "url": pw["url"],
        "title": pw.get("title") or "Unknown",
        "last_modified": "Not provided (Playwright rendered)",
        "text": full_text,
        "soup": soup,
        "raw_html": raw_html,
        "_signals": extract_access_signals(raw_html, pw["url"], soup),
        "_playwright_meta": {
            "xhr_json_urls": xhr,
            "dataset_inventory": inv,
            "json_response_count": pw.get("json_response_count", 0),
            "spa_target_path": pw.get("spa_target_path"),
            "spa_entry_url": pw.get("spa_entry_url"),
            "spa_client_nav": pw.get("spa_client_nav"),
            "dom_category_labels": list(dom_labels),
            "dom_category_urls": list(dom_urls),
            "supabase_apikeys": pw.get("supabase_apikeys") or [],
            "supabase_authorizations": pw.get("supabase_authorizations") or [],
            "dataset_ui_stats": pw.get("dataset_ui_stats") or {},
            "supabase_bulk_meta": pw.get("supabase_bulk_meta") or {},
            "dataset_names_sorted": list(pw.get("dataset_names_sorted") or []),
            "dataset_spatial_rows": list(pw.get("dataset_spatial_rows") or []),
            "dataset_category_counts": dict(pw.get("dataset_category_counts") or {}),
            "python_access_markdown_block": pw.get("python_access_markdown_block") or "",
            "dataset_catalog_summary_text": pw.get("dataset_catalog_summary_text") or "",
            "segmentation_annotation_schema": pw.get("segmentation_annotation_schema") or {},
            "dataset_ui_segmentation_layers": list(pw.get("dataset_ui_segmentation_layers") or []),
        },
    }
    return page


def merge_playwright_into_signals(merged: dict, pw_meta: dict | None) -> dict:
    """Attach catalog + XHR URLs to merged access dict for the Appendix."""
    if not pw_meta:
        return merged
    out = dict(merged)
    cap_xhr = appendix_cap("APPENDIX_XHR_URL_CAP")
    cap_inv = appendix_cap("APPENDIX_DATASET_ROWS")
    out["xhr_json_urls"] = list(pw_meta.get("xhr_json_urls") or [])[:cap_xhr]
    out["dataset_inventory"] = list(pw_meta.get("dataset_inventory") or [])[:cap_inv]
    out["dataset_inventory_count"] = len(pw_meta.get("dataset_inventory") or [])
    for k in ("spa_target_path", "spa_entry_url", "spa_client_nav"):
        v = pw_meta.get(k)
        if v:
            out[k] = v
    doml = pw_meta.get("dom_category_labels") or []
    cap_dom = appendix_cap("APPENDIX_DOM_LABELS")
    out["dom_category_labels"] = list(doml)[:cap_dom]
    out["dom_category_label_count"] = len(doml)
    domu = pw_meta.get("dom_category_urls") or []
    cap_u = appendix_cap("APPENDIX_DOM_URLS")
    out["dom_category_urls"] = list(domu)[:cap_u]
    out["dom_category_url_count"] = len(domu)
    apis = pw_meta.get("supabase_apikeys") or []
    out["supabase_apikeys"] = list(apis)[:5]
    auths = pw_meta.get("supabase_authorizations") or []
    out["supabase_authorizations"] = list(auths)[:5]
    out["dataset_ui_stats"] = dict(pw_meta.get("dataset_ui_stats") or {})
    out["supabase_bulk_meta"] = dict(pw_meta.get("supabase_bulk_meta") or {})
    cap_names = appendix_cap("APPENDIX_DATASET_NAME_LIST_CAP")
    out["dataset_names_sorted"] = list(pw_meta.get("dataset_names_sorted") or [])[:cap_names]
    cap_sp = appendix_cap("APPENDIX_SPATIAL_ROWS_CAP")
    out["dataset_spatial_rows"] = list(pw_meta.get("dataset_spatial_rows") or [])[:cap_sp]
    out["dataset_category_counts"] = dict(pw_meta.get("dataset_category_counts") or {})
    out["python_access_markdown_block"] = str(pw_meta.get("python_access_markdown_block") or "")
    out["segmentation_annotation_schema"] = dict(
        pw_meta.get("segmentation_annotation_schema") or {}
    )
    cap_ui = appendix_cap("APPENDIX_UI_SEGMENTATION_ROWS")
    out["dataset_ui_segmentation_layers"] = list(
        pw_meta.get("dataset_ui_segmentation_layers") or []
    )[:cap_ui]
    return out


def combine_playwright_metas(meta_list: list[dict | None]) -> dict | None:
    """Merge xhr URLs + dataset rows from several Playwright navigations (deduped)."""
    metas = [m for m in meta_list if m]
    if not metas:
        return None
    seen_u: set[str] = set()
    xhr: list[str] = []
    for m in metas:
        for u in m.get("xhr_json_urls") or []:
            if u not in seen_u:
                seen_u.add(u)
                xhr.append(u)
    seen_row: set[str] = set()
    inv: list[str] = []
    for m in metas:
        for row in m.get("dataset_inventory") or []:
            if row not in seen_row:
                seen_row.add(row)
                inv.append(row)
    spa_fields: dict = {}
    for m in metas:
        if m.get("spa_target_path"):
            spa_fields = {
                "spa_target_path": m.get("spa_target_path"),
                "spa_entry_url": m.get("spa_entry_url"),
                "spa_client_nav": m.get("spa_client_nav"),
            }
            break
    out = {
        "xhr_json_urls": xhr,
        "dataset_inventory": inv,
        "json_response_count": sum(int(m.get("json_response_count") or 0) for m in metas),
    }
    out.update({k: v for k, v in spa_fields.items() if v})
    seen_lab: set[str] = set()
    merged_labels: list[str] = []
    for m in metas:
        for lab in m.get("dom_category_labels") or []:
            if not isinstance(lab, str):
                continue
            k = lab.strip().lower()
            if not k or k in seen_lab:
                continue
            seen_lab.add(k)
            merged_labels.append(lab.strip())
    cap_m = appendix_cap("APPENDIX_DOM_LABELS")
    out["dom_category_labels"] = merged_labels[:cap_m]
    out["dom_category_label_count"] = len(merged_labels)
    seen_url: set[str] = set()
    merged_urls: list[str] = []
    for m in metas:
        for u in m.get("dom_category_urls") or []:
            if not isinstance(u, str):
                continue
            su = u.split("#")[0].strip()
            if not su:
                continue
            k = su.lower()
            if k in seen_url:
                continue
            seen_url.add(k)
            merged_urls.append(su)
    cap_um = appendix_cap("APPENDIX_DOM_URLS")
    out["dom_category_urls"] = merged_urls[:cap_um]
    out["dom_category_url_count"] = len(merged_urls)
    apik: set[str] = set()
    authk: set[str] = set()
    for m in metas:
        for k in m.get("supabase_apikeys") or []:
            if isinstance(k, str) and k.strip():
                apik.add(k.strip())
        for k in m.get("supabase_authorizations") or []:
            if isinstance(k, str) and k.strip():
                authk.add(k.strip())
    out["supabase_apikeys"] = sorted(apik)[:5]
    out["supabase_authorizations"] = sorted(authk)[:5]

    all_names: set[str] = set()
    for m in metas:
        for n in m.get("dataset_names_sorted") or []:
            if isinstance(n, str) and n.strip():
                all_names.add(n.strip())
    cap_names = appendix_cap("APPENDIX_DATASET_NAME_LIST_CAP")
    out["dataset_names_sorted"] = sorted(all_names, key=str.lower)[:cap_names]

    seen_sp: set[str] = set()
    spatial_merged: list[str] = []
    for m in metas:
        for line in m.get("dataset_spatial_rows") or []:
            if not isinstance(line, str) or not line.strip():
                continue
            if line in seen_sp:
                continue
            seen_sp.add(line)
            spatial_merged.append(line)
    cap_sp = appendix_cap("APPENDIX_SPATIAL_ROWS_CAP")
    out["dataset_spatial_rows"] = spatial_merged[:cap_sp]

    cat_acc: Counter = Counter()
    for m in metas:
        dc = m.get("dataset_category_counts") or {}
        if isinstance(dc, dict):
            for k, v in dc.items():
                try:
                    cat_acc[str(k)] += int(v)
                except (TypeError, ValueError):
                    pass
    out["dataset_category_counts"] = dict(cat_acc.most_common(100))

    best_ui: dict = {}
    best_total = -1
    for m in metas:
        u = m.get("dataset_ui_stats") or {}
        if isinstance(u, dict) and u.get("ui_total") is not None:
            try:
                t = int(u["ui_total"])
            except (TypeError, ValueError):
                t = 0
            if t > best_total:
                best_total = t
                best_ui = u
    out["dataset_ui_stats"] = best_ui

    bulk_best: dict = {}
    max_rows = -1
    for m in metas:
        b = m.get("supabase_bulk_meta") or {}
        if isinstance(b, dict) and b.get("row_count") is not None:
            try:
                rc = int(b["row_count"])
            except (TypeError, ValueError):
                rc = 0
            if rc > max_rows:
                max_rows = rc
                bulk_best = b
    out["supabase_bulk_meta"] = bulk_best

    py_parts = [m.get("python_access_markdown_block") for m in metas if m.get("python_access_markdown_block")]
    out["python_access_markdown_block"] = py_parts[0] if py_parts else ""

    best_schema: dict = {}
    best_schema_rc = -1
    for m in metas:
        b = m.get("supabase_bulk_meta") or {}
        if not isinstance(b, dict):
            continue
        try:
            rc = int(b.get("row_count") or 0)
        except (TypeError, ValueError):
            rc = 0
        sch = m.get("segmentation_annotation_schema")
        if isinstance(sch, dict) and sch and rc >= best_schema_rc:
            best_schema_rc = rc
            best_schema = sch
    out["segmentation_annotation_schema"] = best_schema

    ui_by_ds: dict[str, dict] = {}
    for m in metas:
        for row in m.get("dataset_ui_segmentation_layers") or []:
            if not isinstance(row, dict):
                continue
            ds = row.get("dataset_name")
            if isinstance(ds, str) and ds.strip():
                ui_by_ds[ds.strip()] = row
    cap_ui_m = appendix_cap("APPENDIX_UI_SEGMENTATION_ROWS")
    merged_ui = [ui_by_ds[k] for k in sorted(ui_by_ds.keys(), key=str.lower)]
    out["dataset_ui_segmentation_layers"] = merged_ui[:cap_ui_m]

    return out

