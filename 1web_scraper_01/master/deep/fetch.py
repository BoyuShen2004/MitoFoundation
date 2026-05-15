"""HTTP fetch and multi-page deep scrape orchestration."""

from __future__ import annotations

import os
import re
import time
from typing import List
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .config import (
    DEEP_SCRAPE_KEYWORDS,
    HTTP_HEADERS,
    RAW_HTML_SAMPLE,
    _env_truthy,
    ensure_playwright_chromium_installed,
)
from .playwright import (
    _page_dict_from_playwright,
    combine_playwright_metas,
    merge_playwright_into_signals,
    playwright_capture,
)
from .signals import extract_access_signals, merge_site_access_signals
from .websites import (
    should_use_playwright,
    spa_hosts_from_websites_md,
)

def fetch_page(url: str) -> dict:
    """Fetch one page and return normalized text + metadata."""
    try:
        resp = requests.get(url.strip(), headers=HTTP_HEADERS, timeout=25)
        resp.raise_for_status()
        last_modified = resp.headers.get("Last-Modified", "Not provided")
        raw_html = resp.text[:RAW_HTML_SAMPLE] if resp.text else ""

        soup = BeautifulSoup(resp.text, "lxml")
        title = soup.title.get_text(strip=True) if soup.title else "Unknown"
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        raw = soup.get_text(separator="\n", strip=True)
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        text = "\n".join(lines)
        return {
            "ok": True,
            "url": url,
            "title": title,
            "last_modified": last_modified,
            "text": text,
            "soup": soup,
            "raw_html": raw_html,
        }
    except Exception as exc:
        return {"ok": False, "url": url, "error": str(exc), "soup": None, "raw_html": ""}


def collect_internal_links(base_url: str, soup: BeautifulSoup, limit: int = 12) -> List[str]:
    """Extract likely relevant internal links from a page."""
    parsed = urlparse(base_url)
    base_domain = parsed.netloc
    ranked: List[str] = []
    seen = set()

    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue

        abs_url = urljoin(base_url, href).split("#")[0]
        p = urlparse(abs_url)
        if p.scheme not in {"http", "https"}:
            continue
        if p.netloc != base_domain:
            continue
        if abs_url in seen:
            continue

        seen.add(abs_url)
        lower = abs_url.lower()
        score = sum(1 for kw in DEEP_SCRAPE_KEYWORDS if kw in lower)
        if score > 0:
            ranked.append((score, abs_url))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [u for _, u in ranked[:limit]]


def fetch_sitemap_candidates(base_url: str, limit: int = 10) -> List[str]:
    """Pull candidate URLs from sitemap.xml if available."""
    sitemap_url = urljoin(base_url, "/sitemap.xml")
    try:
        resp = requests.get(sitemap_url, headers=HTTP_HEADERS, timeout=20)
        if resp.status_code != 200:
            return []
        locs = re.findall(r"<loc>(.*?)</loc>", resp.text, flags=re.IGNORECASE)
        parsed_base = urlparse(base_url).netloc
        candidates = []
        for loc in locs:
            if urlparse(loc).netloc != parsed_base:
                continue
            lower = loc.lower()
            if any(kw in lower for kw in DEEP_SCRAPE_KEYWORDS):
                candidates.append(loc)
            if len(candidates) >= limit:
                break
        return candidates
    except Exception:
        return []


def _norm_seen_url(u: str) -> str:
    return u.split("#")[0].rstrip("/")


def seed_urls_from_env(root_url: str) -> List[str]:
    """Optional extra routes to open with the same origin (comma-separated paths)."""
    raw = os.getenv("PLAYWRIGHT_SEED_PATHS", "").strip()
    if not raw:
        return []
    base = root_url.split("#")[0].rstrip("/")
    out: List[str] = []
    for part in raw.split(","):
        p = part.strip().lstrip("/")
        if not p:
            continue
        out.append(urljoin(base + "/", p))
    return out


def rank_internal_urls_for_deep_scrape(
    base_url: str, urls: List[str], limit: int
) -> List[str]:
    """Prefer same-host URLs whose paths match DEEP_SCRAPE_KEYWORDS; then fill with others."""
    parsed = urlparse(base_url)
    base_domain = parsed.netloc
    scored: List[tuple[int, str]] = []
    for u in urls:
        p = urlparse(u)
        if p.netloc != base_domain:
            continue
        lower = u.lower()
        score = sum(1 for kw in DEEP_SCRAPE_KEYWORDS if kw in lower)
        scored.append((score, u))
    scored.sort(key=lambda x: (x[0], -len(x[1])), reverse=True)
    out: List[str] = []
    seen_local: set[str] = set()
    for _, u in scored:
        if u in seen_local:
            continue
        seen_local.add(u)
        out.append(u)
        if len(out) >= limit:
            return out
    for u in urls:
        if urlparse(u).netloc != base_domain:
            continue
        if u in seen_local:
            continue
        seen_local.add(u)
        out.append(u)
        if len(out) >= limit:
            break
    return out


def scrape_site_deep(
    base_url: str,
    website_name: str,
    max_pages: int | None = None,
    spa_hosts: frozenset[str] | None = None,
) -> tuple[str, dict]:
    """Scrape homepage + relevant internal pages for deeper context.

    ``spa_hosts``: hosts (from per-site ``<site>/guides/websites.md``) that should use Playwright; if
    omitted, hosts are read from per-site website definitions once for this call.

    For JS-heavy sites, same-host follow-up URLs are also rendered with Playwright
    (up to ``PLAYWRIGHT_MAX_FOLLOW``) so JSON/XHR and in-DOM links are captured.

    Returns (aggregated_text_for_llm, merged_access_signals).
    """
    if max_pages is None:
        max_pages = int(os.getenv("DEEP_SCRAPE_MAX_PAGES", "22"))
    delay = float(os.getenv("DEEP_SCRAPE_DELAY_SEC", "0.9"))
    max_follow_pw = int(os.getenv("PLAYWRIGHT_MAX_FOLLOW", "10"))
    link_pool = int(os.getenv("DEEP_SCRAPE_LINK_POOL", "48"))

    pages: list[dict] = []
    first: dict | None = None
    root_pw_snapshot: dict | None = None

    if spa_hosts is None:
        spa_hosts = spa_hosts_from_websites_md()

    playwright_blocked: dict | None = None

    if should_use_playwright(base_url, spa_hosts):
        print(f"[INFO] Playwright (headless) rendering: {base_url}")
        pw = playwright_capture(base_url)
        if pw.get("ok"):
            root_pw_snapshot = pw
            first = _page_dict_from_playwright(pw)
        else:
            err = pw.get("error", "unknown error")
            if pw.get("package_missing"):
                print("[ERROR] Playwright Python package is missing or broken.")
                print("        Fix:  pip install playwright && python -m playwright install chromium")
                playwright_blocked = {"kind": "package", "detail": err}
            elif pw.get("browser_missing"):
                print("[ERROR] Playwright browser binaries are not installed (Chromium missing).")
                print("        Fix:  python -m playwright install chromium")
                print("        Or:   playwright install")
                playwright_blocked = {"kind": "browser", "detail": err}
            else:
                print(f"[WARN] Playwright: {err}")

    if first is None and playwright_blocked:
        # Do not fall back to requests on the configured URL: deep SPA paths often 404 and
        # mislead the LLM into "site broken" narratives. Require a working browser instead.
        kind = playwright_blocked["kind"]
        auto_install_attempted = False
        auto_install_err = ""
        if kind == "browser" and _env_truthy("AUTO_INSTALL_PLAYWRIGHT_CHROMIUM", "1"):
            auto_install_attempted = True
            print("[INFO] Playwright Chromium binary missing; attempting auto-install.")
            ok, install_err = ensure_playwright_chromium_installed()
            if ok:
                print("[INFO] Auto-install succeeded; retrying Playwright capture.")
                pw_retry = playwright_capture(base_url)
                if pw_retry.get("ok"):
                    root_pw_snapshot = pw_retry
                    first = _page_dict_from_playwright(pw_retry)
                    playwright_blocked = None
            else:
                auto_install_err = install_err

        if first is None and playwright_blocked:
            if kind == "browser":
                auto_err_block = ""
                if auto_install_attempted and auto_install_err:
                    auto_err_block = (
                        f"Auto-install error (truncated): {auto_install_err}\n\n"
                    )
                body = (
                    "=== SCRAPE BLOCKED: Playwright Chromium not installed ===\n\n"
                    "This URL is configured for headless-browser scraping (USE_PLAYWRIGHT / per-site guides host), "
                    "but no Chromium binary was found.\n\n"
                    "Nothing below is a live catalog scrape — the SPA never ran.\n\n"
                    "If auto-install is enabled, the agent attempted: python -m playwright install chromium.\n\n"
                    + auto_err_block
                    + "Otherwise, install browsers in the **same venv** you use to run agent.py:\n\n"
                    "  python -m playwright install chromium\n\n"
                    "If you use conda/venv, activate it first. On some clusters add:\n"
                    "  export PLAYWRIGHT_NO_SANDBOX=1\n\n"
                    f"Configured URL (for reference): {base_url}\n"
                )
            else:
                body = (
                    "=== SCRAPE BLOCKED: Playwright Python package missing ===\n\n"
                    "Install:\n\n"
                    "  pip install playwright\n"
                    "  python -m playwright install chromium\n\n"
                    f"Configured URL (for reference): {base_url}\n"
                )
            merged = merge_site_access_signals([])
            merged["playwright_scrape_blocked"] = True
            merged["playwright_blocked_kind"] = kind
            merged["scrape_failure_message"] = body.strip()
            return (body, merged)

    if first is None:
        first = fetch_page(base_url)
    if not first.get("ok"):
        return (
            f"Error scraping {base_url}: {first.get('error', 'Unknown error')}",
            merge_site_access_signals([]),
        )

    if "_signals" not in first:
        first["_signals"] = extract_access_signals(
            first.get("raw_html", ""), base_url, first.get("soup")
        )
    pages.append(first)

    root_internal = (root_pw_snapshot or {}).get("internal_urls") or []
    root_dom_urls = (root_pw_snapshot or {}).get("dom_category_urls") or []
    candidate_pool: List[str] = []
    candidate_pool.extend(
        rank_internal_urls_for_deep_scrape(base_url, root_internal, link_pool)
    )
    # If we can directly see in-app links for organelles/categories, prioritize them.
    candidate_pool.extend(root_dom_urls)
    candidate_pool.extend(seed_urls_from_env(base_url))
    if first.get("soup") is not None:
        candidate_pool.extend(
            collect_internal_links(
                base_url, first["soup"], limit=max(link_pool, max_pages * 2)
            )
        )
    candidate_pool.extend(fetch_sitemap_candidates(base_url, limit=max_pages))

    dom_priority_keys = {_norm_seen_url(u) for u in root_dom_urls if _norm_seen_url(u)}

    def score_candidate_url(u: str) -> int:
        ul = (u or "").lower()
        key = _norm_seen_url(u)
        score = 0
        if key and key in dom_priority_keys:
            score += 1200
        if "/organelles/" in ul:
            score += 700
        if "/collections/" in ul:
            score += 600
        if "/datasets" in ul:
            score += 400
        if "/measure" in ul or "/measurement" in ul:
            score += 200
        if "/faq" in ul or "data_access" in ul:
            score += 150
        if any(kw in ul for kw in ("mito", "mitochond", "segmentation", "label", "download")):
            score += 120
        score += sum(10 for kw in DEEP_SCRAPE_KEYWORDS if kw in ul)
        return score

    scored_candidates = [(score_candidate_url(c), c) for c in candidate_pool]
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    ordered_unique: List[str] = []
    seen_c: set[str] = set()
    for _score, c in scored_candidates:
        key = _norm_seen_url(c)
        if not key or key in seen_c:
            continue
        seen_c.add(key)
        ordered_unique.append(c.split("#")[0])

    pw_meta_list: list[dict | None] = []
    if first.get("_playwright_meta"):
        pw_meta_list.append(first["_playwright_meta"])

    seen_pages = {_norm_seen_url(base_url)}

    for cand in ordered_unique:
        if len(pages) >= max_pages:
            break
        nu = _norm_seen_url(cand)
        if nu in seen_pages:
            continue

        use_pw = should_use_playwright(cand, spa_hosts) and max_follow_pw > 0
        if use_pw:
            max_follow_pw -= 1
            seen_pages.add(nu)
            print(f"[INFO] Playwright follow ({len(pages) + 1}/{max_pages}): {cand}")
            time.sleep(delay)
            pw2 = playwright_capture(cand, run_supabase_bulk=False)
            if not pw2.get("ok"):
                print(f"[WARN] Playwright follow failed: {pw2.get('error', '')}")
                continue
            p2 = _page_dict_from_playwright(pw2)
            p2["_signals"] = extract_access_signals(
                p2.get("raw_html", ""), cand, p2.get("soup")
            )
            pages.append(p2)
            if p2.get("_playwright_meta"):
                pw_meta_list.append(p2["_playwright_meta"])
            continue

        seen_pages.add(nu)
        time.sleep(delay)
        page = fetch_page(cand)
        if page.get("ok"):
            page["_signals"] = extract_access_signals(
                page.get("raw_html", ""), cand, page.get("soup")
            )
            pages.append(page)

    merged_access = merge_site_access_signals(pages)
    combined_pw = combine_playwright_metas(pw_meta_list)
    merged_access = merge_playwright_into_signals(merged_access, combined_pw)

    n_pw_pages = sum(1 for p in pages if p.get("_playwright_meta"))
    sections = [
        f"Website: {website_name}",
        f"Root URL: {base_url}",
        f"Pages scraped: {len(pages)} (max_pages={max_pages})",
        f"Playwright navigations: {n_pw_pages}",
        f"Dataset rows extracted (JSON): {merged_access.get('dataset_inventory_count', 0)}",
        f"DOM category labels (visible text): {merged_access.get('dom_category_label_count', 0)}",
        "",
    ]
    for i, p in enumerate(pages, start=1):
        text = p.get("text", "")
        cap = 14_000 if p.get("_playwright_meta") else 1_500
        if len(text) > cap:
            text = text[:cap] + "\n...[truncated]"
        sections.append(
            f"=== Page {i} ===\n"
            f"URL: {p.get('url')}\n"
            f"Title: {p.get('title')}\n"
            f"HTTP Last-Modified: {p.get('last_modified')}\n"
            f"Content:\n{text}\n"
        )
    return "\n".join(sections), merged_access

