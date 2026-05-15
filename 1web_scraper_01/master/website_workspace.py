"""Per-website folders under ``1web_scraper_01/websites/<slug>/`` with ``site.md`` (and optional ``guides/``).

Structured catalogs are under ``1web_scraper_01/outputs/`` (probe JSON, catalog markdown).
"""

from __future__ import annotations

import json
import re
import shutil
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from .time_utils import now_us_eastern_iso

from .generic_scrape import _site_stem, fetch_page, write_probe_json
from .mito_foundation_bridge import (
    is_openorganelle_url,
    normalize_openorganelle_landing_url,
    probe_datasets_count,
    provider_name_for_url,
    run_provider_deep_scrape,
)

# Bundled Pipeline Studio workspaces — must not be removed via Studio “delete site”.
# Matching is case-insensitive so API/UI cannot bypass with mixed-case slugs.
PROTECTED_WEBSITE_SLUGS_LOWER: frozenset[str] = frozenset({"bossdb_01", "openorganelle_01"})


def websites_root(project_root: Path) -> Path:
    return project_root / "1web_scraper_01" / "websites"


def purge_legacy_datasets_json(folder: Path) -> None:
    """Remove obsolete per-site ``datasets.json`` (catalog lives in ``outputs/*.probe.json`` only)."""
    p = folder / "datasets.json"
    if p.is_file():
        try:
            p.unlink()
        except OSError:
            pass


def _purge_all_legacy_datasets_json(project_root: Path) -> None:
    root_w = websites_root(project_root)
    if not root_w.is_dir():
        return
    for child in root_w.iterdir():
        if child.is_dir():
            purge_legacy_datasets_json(child)


def slug_from_display_name(name: str) -> str:
    """Lowercase slug; allows ``_`` and digits for ``openorganelle_01``."""
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return (s or "")[:80]


def slug_base_for_versioning(slug: str) -> str:
    """Strip trailing ``_NN`` (two digits) so ``openorganelle_02`` → ``openorganelle``."""
    s = (slug or "").strip().lower()
    m = re.match(r"^(.+)_(\d{2})$", s)
    if m and m.group(1):
        return m.group(1)
    return s


def _rewrite_slugs_in_workspace(folder: Path, old_slug: str, new_slug: str) -> None:
    if old_slug == new_slug:
        return
    sm = folder / "site.md"
    if sm.is_file():
        meta, body = _parse_simple_front_matter(sm.read_text(encoding="utf-8"))
        if meta:
            meta["slug"] = new_slug
            sm.write_text(_dump_front_matter(meta, body), encoding="utf-8")
    purge_legacy_datasets_json(folder)


def migrate_legacy_website_folders(project_root: Path) -> None:
    """Rename ``websites/<name>/`` (no ``_NN`` suffix) to ``<name>_01/`` and fix slugs inside."""
    root_w = websites_root(project_root)
    if not root_w.is_dir():
        return
    out_root = project_root / "1web_scraper_01" / "outputs"
    for p in sorted(root_w.iterdir(), key=lambda x: x.name):
        if not p.is_dir() or not (p / "site.md").is_file():
            continue
        name = p.name
        if re.search(r"_\d{2}$", name):
            continue
        dest = root_w / f"{name}_01"
        if dest.exists():
            continue
        p.rename(dest)
        _rewrite_slugs_in_workspace(dest, name, f"{name}_01")
        probe_old = out_root / f"{name}.probe.json"
        probe_new = out_root / f"{name}_01.probe.json"
        if probe_old.is_file() and not probe_new.is_file():
            probe_old.rename(probe_new)

    # If both ``base/`` and ``base_01/`` exist (rename was skipped), drop the shadow legacy folder.
    for p in list(root_w.iterdir()):
        if not p.is_dir() or not (p / "site.md").is_file():
            continue
        name = p.name
        if re.search(r"_\d{2}$", name):
            continue
        twin = root_w / f"{name}_01"
        if twin.is_dir() and (twin / "site.md").is_file():
            shutil.rmtree(p)

    _purge_all_legacy_datasets_json(project_root)


def next_versioned_website_slug(project_root: Path, base_slug: str) -> str:
    """Next batch folder: ``{base}_01``, ``{base}_02``, … only ``{base}_NN`` dirs (with site.md) count."""
    migrate_legacy_website_folders(project_root)
    root = websites_root(project_root)
    base = slug_base_for_versioning((base_slug or "").strip().lower()) or "site"
    base = re.sub(r"[^a-z0-9_]", "", base) or "site"
    max_n = 0
    if root.is_dir():
        esc = re.escape(base)
        for p in root.iterdir():
            if not p.is_dir() or not (p / "site.md").is_file():
                continue
            m = re.fullmatch(esc + r"_(\d{2})", p.name)
            if m:
                max_n = max(max_n, int(m.group(1)))
    return f"{base}_{max_n + 1:02d}"


def resolve_workspace_slug(project_root: Path, slug: str) -> str:
    """Canonical folder slug for API load/delete/save.

    If both ``base/`` and ``base_01/`` (or other ``base_NN/``) exist, prefer the lowest ``base_NN``
    so delete/load target the real workspace, not a stale legacy folder.
    """
    migrate_legacy_website_folders(project_root)
    s = slug_from_display_name(slug)
    if not s:
        return s
    root = websites_root(project_root)
    b = slug_base_for_versioning(s)

    def _versioned_siblings(base: str) -> list[tuple[int, str]]:
        esc = re.escape(base)
        found: list[tuple[int, str]] = []
        for p in root.iterdir():
            if not p.is_dir() or not (p / "site.md").is_file():
                continue
            m = re.fullmatch(esc + r"_(\d{2})$", p.name)
            if m:
                found.append((int(m.group(1)), p.name))
        found.sort(key=lambda t: t[0])
        return found

    if (root / s / "site.md").is_file():
        if not re.search(r"_\d{2}$", s):
            sibs = _versioned_siblings(s)
            if sibs:
                return sibs[0][1]
        return s

    if s == b:
        sibs = _versioned_siblings(b)
        if sibs:
            return sibs[0][1]
    return s


def resolve_scrape_slug(
    project_root: Path,
    display_name: str,
    url: str,
    slug_override: str | None,
) -> str:
    """Folder slug for ``run_website_workspace_scrape``: existing workspace, or latest ``base_NN``, or next free."""
    migrate_legacy_website_folders(project_root)
    url_n = _normalize_url(url)
    root_w = websites_root(project_root)
    raw_ov = (slug_override or "").strip()
    if raw_ov:
        ov = slug_from_display_name(raw_ov)
        if ov:
            resolved = resolve_workspace_slug(project_root, raw_ov)
            if (root_w / resolved / "site.md").is_file():
                return resolved
            if re.fullmatch(r"[a-z0-9_]+_\d{2}$", ov):
                return ov
            return next_versioned_website_slug(project_root, slug_base_for_versioning(ov))
    base = slug_base_for_versioning(resolve_slug(display_name, url_n))
    base = re.sub(r"[^a-z0-9_]", "", base) or "site"
    max_n = 0
    if root_w.is_dir():
        esc = re.escape(base)
        for p in root_w.iterdir():
            if not p.is_dir() or not (p / "site.md").is_file():
                continue
            m = re.fullmatch(esc + r"_(\d{2})", p.name)
            if m:
                max_n = max(max_n, int(m.group(1)))
    if max_n > 0:
        return f"{base}_{max_n:02d}"
    return next_versioned_website_slug(project_root, base)


def resolve_slug(display_name: str, url: str) -> str:
    d = slug_from_display_name(display_name)
    if d:
        return d
    return _site_stem(url)


def _normalize_url(url: str) -> str:
    u = (url or "").strip()
    if not u.startswith(("http://", "https://")):
        u = "http://" + u
    return u


def _parse_meta_value(raw: str) -> str:
    s = raw.strip()
    if len(s) >= 2 and s.startswith('"') and s.endswith('"'):
        try:
            return str(json.loads(s))
        except json.JSONDecodeError:
            pass
    return s


def _parse_simple_front_matter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
    if not m:
        return {}, text
    meta_raw, body = m.group(1), m.group(2)
    meta: dict[str, str] = {}
    for line in meta_raw.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        key = k.strip()
        val = _parse_meta_value(v)
        if val == "|":
            continue
        meta[key] = val
    return meta, body


def _yaml_scalar(v: str) -> str:
    return json.dumps((v or "").strip())


def _dump_front_matter(meta: dict[str, str], body: str) -> str:
    lines = ["---"]
    for k, v in meta.items():
        lines.append(f"{k}: {_yaml_scalar(v)}")
    lines.append("---")
    lines.append("")
    lines.append(body.strip())
    lines.append("")
    return "\n".join(lines)


def split_site_body(description_and_focus: str) -> tuple[str, str]:
    """Split markdown body into description and **Data to look for** section."""
    m = re.search(r"\n## Data to look for\s*\n+", description_and_focus, re.I)
    if not m:
        return description_and_focus.strip(), ""
    desc = description_and_focus[: m.start()].strip()
    focus = description_and_focus[m.end() :].strip()
    return desc, focus


def write_site_md(
    folder: Path,
    meta: dict[str, str],
    description: str,
    data_focus: str,
) -> Path:
    """Write ``site.md``: short YAML front matter + description + ``## Data to look for``."""
    fm = {
        "display_name": meta["display_name"],
        "url": meta["url"],
        "slug": meta["slug"],
        "updated_at": meta["updated_at"],
    }
    desc = (description or "").strip() or "_No description yet. Edit this file or use the Pipeline Studio form._"
    focus = (data_focus or "").strip() or "_Not specified._"
    body = f"{desc}\n\n## Data to look for\n\n{focus}\n"
    path = folder / "site.md"
    path.write_text(_dump_front_matter(fm, body), encoding="utf-8")
    return path


def read_site_md(folder: Path) -> tuple[dict[str, str], str] | None:
    p = folder / "site.md"
    if not p.is_file():
        return None
    return _parse_simple_front_matter(p.read_text(encoding="utf-8"))


def run_website_workspace_scrape(
    project_root: Path,
    *,
    display_name: str,
    url: str,
    description: str,
    data_focus: str,
    slug_override: str | None = None,
    write_probe_legacy: bool = True,
    log_line: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
    scrape_session_id: str | None = None,
    subprocess_llm_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Create/update ``websites/<slug>/site.md``;
    write ``outputs/<slug>.probe.json`` for generic sites; OpenOrganelle uses ``outputs/OpenOrganelle.probe.json`` only.
    """
    url_n = _normalize_url(url)
    slug = resolve_scrape_slug(project_root, display_name, url, slug_override)
    root_w = websites_root(project_root)
    root_w.mkdir(parents=True, exist_ok=True)
    folder = root_w / slug
    folder.mkdir(parents=True, exist_ok=True)
    purge_legacy_datasets_json(folder)

    now = now_us_eastern_iso()
    focus = (data_focus or "").strip() or (
        "General structured or downloadable datasets linked from this site; organelle relevance is determined later."
    )

    probe_rel: str | None = None
    bridge_extra: dict[str, Any] = {}

    _provider_name = provider_name_for_url(url_n)
    if _provider_name is not None:
        # Use the canonical URL form if this is a known provider.
        url_work = normalize_openorganelle_landing_url(url_n) if is_openorganelle_url(url_n) else url_n
        meta = {
            "display_name": (display_name or "").strip() or slug.replace("_", " ").title(),
            "url": url_work,
            "slug": slug,
            "data_focus": focus,
            "updated_at": now,
        }
        write_site_md(folder, meta, description, focus)
        bridge = run_provider_deep_scrape(
            _provider_name,
            project_root,
            workspace_slug=slug,
            log_line=log_line,
            cancel_event=cancel_event,
            scrape_session_id=scrape_session_id,
            subprocess_llm_env=subprocess_llm_env,
        )
        bridge_extra = {k: v for k, v in bridge.items() if k in ("stdout", "stderr", "markdown_copied", "source_probe", "message", "error")}
        ok = bool(bridge.get("ok"))
        probe_rel = bridge.get("probe_path") if isinstance(bridge.get("probe_path"), str) else None
        fetch: dict[str, Any] = {
            "ok": ok,
            "url": url_work,
            "note": "openorganelle_deep_scrape_bundled",
        }
        if not ok:
            fetch["error"] = bridge.get("message") or bridge.get("error") or "openorganelle_scrape_failed"

        _probe_names = {
            "openorganelle": ("OpenOrganelle.probe.json", "OpenOrganelle.md",
                              "--sites OpenOrganelle", "Playwright + PostgREST"),
            "bossdb":        ("BossDB.probe.json", "BossDB.md",
                              "--sites BossDB", "BossDB metadata REST API"),
        }
        _pn, _mn, _flag, _tech = _probe_names.get(
            _provider_name,
            ("OpenOrganelle.probe.json", "OpenOrganelle.md", "--sites OpenOrganelle", "Playwright + PostgREST"),
        )
        out_probe = project_root / "1web_scraper_01" / "outputs" / _pn
        n_probe = probe_datasets_count(out_probe) if ok else 0
        _pname = (_provider_name or "openorganelle").title()
        backend_note = (
            f"**{_pname}:** From ``1web_scraper_01/``, runs ``python -m master.agent {_flag}`` "
            f"(``master.deep`` catalog pipeline: {_tech}). Outputs: "
            f"``1web_scraper_01/outputs/{_pn}`` and ``{_mn}``."
        )
        return {
            "ok": ok,
            "slug": slug,
            "folder": str(folder.relative_to(project_root)).replace("\\", "/"),
            "site_md": str((folder / "site.md").relative_to(project_root)).replace("\\", "/"),
            "datasets_json": "",
            "probe_path": probe_rel,
            "fetch": fetch,
            "mito_foundation_bridge": bridge_extra,
        }

    meta = {
        "display_name": (display_name or "").strip() or slug.replace("_", " ").title(),
        "url": url_n,
        "slug": slug,
        "data_focus": focus,
        "updated_at": now,
    }
    write_site_md(folder, meta, description, focus)
    if log_line:
        log_line(f"[mito2] Fetching {url_n} …\n")
    fetch = fetch_page(url_n)
    if log_line:
        if fetch.get("ok"):
            log_line(f"[mito2] HTTP ok — title: {(fetch.get('title') or '')[:120]!r}\n")
        else:
            log_line(f"[mito2] HTTP failed — {fetch.get('error', 'unknown')}\n")
    if write_probe_legacy:
        probe_path = write_probe_json(project_root, url_n, fetch, stem=slug)
        probe_rel = str(probe_path.relative_to(project_root)).replace("\\", "/")
        if log_line:
            log_line(f"[mito2] Wrote {probe_rel}\n")

    return {
        "ok": bool(fetch.get("ok")),
        "slug": slug,
        "folder": str(folder.relative_to(project_root)).replace("\\", "/"),
        "site_md": str((folder / "site.md").relative_to(project_root)).replace("\\", "/"),
        "datasets_json": "",
        "probe_path": probe_rel,
        "fetch": fetch,
    }


def list_website_slugs(project_root: Path) -> list[str]:
    migrate_legacy_website_folders(project_root)
    root = websites_root(project_root)
    if not root.is_dir():
        return []
    names = sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "site.md").is_file())
    bases_with_versioned: set[str] = set()
    for name in names:
        m = re.fullmatch(r"(.+)_(\d{2})$", name)
        if m and m.group(1):
            bases_with_versioned.add(m.group(1))
    out: list[str] = []
    for name in names:
        if re.search(r"_\d{2}$", name):
            out.append(name)
            continue
        if name in bases_with_versioned:
            continue
        out.append(name)
    return sorted(out)


def _datasets_count_for_workspace(project_root: Path, slug_res: str) -> int:
    """Count dataset entries from ``outputs/*.probe.json`` (no workspace JSON catalog)."""
    out = project_root / "1web_scraper_01" / "outputs"
    base = slug_base_for_versioning(slug_res)
    if "openorganelle" in base:
        return probe_datasets_count(out / "OpenOrganelle.probe.json")
    if "bossdb" in base:
        return probe_datasets_count(out / "BossDB.probe.json")
    candidates: list[Path] = [out / f"{slug_res}.probe.json"]
    if base != slug_res:
        candidates.append(out / f"{base}.probe.json")
    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        n = probe_datasets_count(p)
        if n:
            return n
    return 0


def load_website_summary(project_root: Path, slug: str) -> dict[str, Any] | None:
    slug_res = resolve_workspace_slug(project_root, slug)
    folder = websites_root(project_root) / slug_res
    parsed = read_site_md(folder)
    if not parsed:
        return None
    meta, body = parsed
    desc, focus = split_site_body(body)
    n_ds = _datasets_count_for_workspace(project_root, slug_res)
    return {
        "slug": slug_res,
        "display_name": meta.get("display_name", slug_res),
        "url": meta.get("url", ""),
        "data_focus": focus,
        "description": desc,
        "updated_at": meta.get("updated_at", ""),
        "description_preview": (desc[:400] + ("…" if len(desc) > 400 else "")),
        "datasets_count": n_ds,
    }


def delete_website_workspace(project_root: Path, slug: str) -> dict[str, Any]:
    """Remove ``websites/<slug>/`` and matching ``outputs/<slug>.probe.json`` if present."""
    s = resolve_workspace_slug(project_root, slug)
    if not s:
        return {"ok": False, "error": "invalid_slug", "slug": ""}
    if s.lower() in PROTECTED_WEBSITE_SLUGS_LOWER:
        return {
            "ok": False,
            "error": "protected_slug",
            "slug": s,
            "message": (
                f"Workspace {s!r} is bundled with mitoFoundation2 and cannot be deleted from Studio. "
                "Edit files in place or add a new versioned folder instead."
            ),
        }
    folder = websites_root(project_root) / s
    if not folder.is_dir():
        return {"ok": False, "error": "not_found", "slug": s}
    if not (folder / "site.md").is_file():
        rel_nf = str(folder.relative_to(project_root)).replace("\\", "/")
        shutil.rmtree(folder, ignore_errors=True)
        return {
            "ok": True,
            "slug": s,
            "removed_probe": False,
            "note": "removed folder without site.md",
            "folder": rel_nf,
        }
    rel_folder = str(folder.relative_to(project_root)).replace("\\", "/")
    try:
        shutil.rmtree(folder)
    except OSError as exc:
        return {"ok": False, "error": "remove_failed", "slug": s, "message": str(exc), "folder": rel_folder}
    probe = project_root / "1web_scraper_01" / "outputs" / f"{s}.probe.json"
    removed_probe = False
    if probe.is_file():
        try:
            probe.unlink()
            removed_probe = True
        except OSError:
            pass
    return {"ok": True, "slug": s, "removed_probe": removed_probe, "folder": rel_folder}


def save_website_meta_only(
    project_root: Path,
    *,
    display_name: str,
    url: str,
    description: str,
    data_focus: str,
    slug_override: str | None = None,
    editing_slug: str | None = None,
) -> dict[str, Any]:
    """Persist ``site.md`` only (no HTTP fetch). New profiles get ``{base}_01``, ``{base}_02``, …"""
    migrate_legacy_website_folders(project_root)
    url_n = _normalize_url(url)
    root_w = websites_root(project_root)
    edit_raw = (editing_slug or "").strip()
    edit = resolve_workspace_slug(project_root, edit_raw) if edit_raw else ""
    ov = slug_from_display_name(slug_override or "")

    if edit and (root_w / edit / "site.md").is_file():
        slug = edit
    else:
        if ov:
            base = slug_base_for_versioning(ov)
        else:
            base = slug_base_for_versioning(resolve_slug(display_name, url_n))
        slug = next_versioned_website_slug(project_root, base)

    folder = websites_root(project_root) / slug
    folder.mkdir(parents=True, exist_ok=True)
    purge_legacy_datasets_json(folder)
    now = now_us_eastern_iso()
    focus = (data_focus or "").strip() or (
        "General structured or downloadable datasets; organelle relevance decided downstream."
    )
    meta = {
        "display_name": (display_name or "").strip() or slug.replace("_", " ").title(),
        "url": url_n,
        "slug": slug,
        "updated_at": now,
    }
    write_site_md(folder, meta, description, focus)

    return {
        "slug": slug,
        "folder": str(folder.relative_to(project_root)).replace("\\", "/"),
        "site_md": str((folder / "site.md").relative_to(project_root)).replace("\\", "/"),
    }
