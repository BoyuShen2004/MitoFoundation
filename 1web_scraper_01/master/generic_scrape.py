"""Lightweight HTTP fetch + probe JSON writer (no browser automation)."""

from __future__ import annotations

import json
import os
import re
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .time_utils import now_us_eastern_iso


def _site_stem(url: str) -> str:
    """First hostname label, lowercase letters+digits only (no spaces). Matches website workspace folders."""
    u = (url or "").strip()
    if not u.startswith(("http://", "https://")):
        u = "http://" + u
    host = (urlparse(u).netloc or "").lower()
    host = re.sub(r"^www\.", "", host)
    if not host:
        return "site"
    first = host.split(".")[0]
    safe = re.sub(r"[^a-z0-9]", "", first) or "site"
    return safe[:80]


def fetch_page(url: str, *, timeout_sec: float = 30.0) -> dict[str, Any]:
    """Return minimal page metadata and a short text excerpt."""
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "mitoFoundation2-scraper/1.0 (+http://example.local)",
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        },
        method="GET",
    )
    if os.getenv("MITO2_SSL_INSECURE", "").strip() in ("1", "true", "yes"):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    else:
        ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec, context=ctx) as res:
            raw = res.read(2_000_000)
            ctype = res.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        # OpenOrganelle is a SPA on S3: only ``/`` is a real object; ``/datasets`` returns 404.
        if e.code == 404:
            parsed = urlparse(url)
            host = (parsed.netloc or "").lower()
            if host.endswith("openorganelle.janelia.org") and (parsed.path or "").strip("/"):
                root_url = f"{parsed.scheme}://{parsed.netloc}/"
                return fetch_page(root_url, timeout_sec=timeout_sec)
        return {
            "ok": False,
            "url": url,
            "error": f"HTTP {e.code}",
            "final_url": getattr(e, "url", url),
        }
    except Exception as e:
        return {"ok": False, "url": url, "error": str(e)}

    text = raw.decode("utf-8", errors="ignore")
    title_m = re.search(r"<title[^>]*>([^<]{1,500})</title>", text, re.IGNORECASE)
    title = title_m.group(1).strip() if title_m else ""
    excerpt = re.sub(r"<[^>]+>", " ", text)
    excerpt = re.sub(r"\s+", " ", excerpt).strip()[:4000]
    return {
        "ok": True,
        "url": url,
        "content_type": ctype,
        "title": title,
        "excerpt": excerpt,
        "bytes_read": len(raw),
    }


def write_probe_json(
    project_root: Path,
    url: str,
    fetch: dict[str, Any],
    *,
    stem: str | None = None,
) -> Path:
    """Write ``{stem}.probe.json`` in the shape expected by database_builder / Pipeline Studio."""
    out_dir = project_root / "1web_scraper_01" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    use = (stem or "").strip().lower()
    use_stem = re.sub(r"[^a-z0-9_]", "", use)[:80] if use else ""
    final_stem = use_stem or _site_stem(url)
    path = out_dir / f"{final_stem}.probe.json"
    now = now_us_eastern_iso()
    body: dict[str, Any] = {
        "url": url,
        "site": final_stem,
        "last_run_at_iso": now,
        "fetch": fetch,
        "datasets": {},
        "note": (
            "mitoFoundation2 generic scrape: datasets may be filled by database_builder "
            "or a site-specific adapter later."
        ),
    }
    if fetch.get("ok") and fetch.get("title"):
        body["datasets"]["_landing_page"] = {
            "title": fetch["title"],
            "excerpt": fetch.get("excerpt", "")[:2000],
        }
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return path


def run_generic_scrape(project_root: Path, url: str) -> dict[str, Any]:
    fetch = fetch_page(url)
    out = write_probe_json(project_root, url, fetch)
    return {"ok": bool(fetch.get("ok")), "probe_path": str(out), "fetch": fetch}
