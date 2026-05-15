"""Load editable Markdown instruction files from ``guides/`` (prompts, goals, URL list).

Change behavior by editing ``.md`` without touching Python. See ``guides/README.md``.
"""

from __future__ import annotations

import re
from pathlib import Path

from .config import DEEP_GUIDES_DIR, GUIDES_DIR

WEB_SCRAPER_ROOT = Path(__file__).resolve().parent.parent.parent


def guides_path(name: str) -> Path:
    """Resolve ``guides/<name>`` (e.g. ``agent_system_prompt.md``)."""
    return GUIDES_DIR / name


def load_guide_text(name: str) -> str:
    """Read UTF-8 text from ``guides/<name>``."""
    path = guides_path(name)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path.read_text(encoding="utf-8")


def _slugify(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "site"


def general_guides_dir() -> Path:
    # Keep general guides at `web_scraper_01/guides/` (no extra nesting).
    return GUIDES_DIR


def site_guides_dir(website_name: str) -> Path:
    """
    Website-specific guides stored as:
      web_scraper_01/<site-slug>/guides/<file>
    """
    return WEB_SCRAPER_ROOT / _slugify(website_name) / "guides"


def _workspace_versioned_guides_paths(name: str, website_name: str) -> list[Path]:
    """Pipeline Studio workspaces: ``1web_scraper_01/websites/{base}_NN/guides/<name>``."""
    base = _slugify(website_name)
    ws = WEB_SCRAPER_ROOT / "websites"
    if not ws.is_dir():
        return []
    out: list[Path] = []
    for folder in sorted(ws.iterdir()):
        if not folder.is_dir():
            continue
        low = folder.name.lower()
        if not (low == base or re.fullmatch(re.escape(base) + r"_\d{2}", low)):
            continue
        p = folder / "guides" / name
        if p.is_file():
            out.append(p)
    return out


def load_guide_for_website(name: str, website_name: str) -> str:
    """Load guide with precedence: per-site ``websites/<slug>/guides/`` → shared defaults."""
    high = [
        *_workspace_versioned_guides_paths(name, website_name),
        site_guides_dir(website_name) / name,
    ]
    for p in high:
        if p.is_file():
            return p.read_text(encoding="utf-8")
    for p in (DEEP_GUIDES_DIR / name, guides_path(name)):
        if p.is_file():
            return p.read_text(encoding="utf-8")
    raise FileNotFoundError(str(high[0] if high else guides_path(name)))


def render_placeholders(template: str, **kwargs: str) -> str:
    """Replace ``{{key}}`` placeholders (mustache-style)."""
    out = template
    for key, val in kwargs.items():
        out = out.replace("{{" + key + "}}", val)
    return out


def load_guide_optional(name: str, default: str = "") -> str:
    """Return guide text or ``default`` if the file is missing."""
    try:
        return load_guide_text(name)
    except OSError:
        return default


def load_guide_for_website_optional(name: str, website_name: str, default: str = "") -> str:
    try:
        return load_guide_for_website(name, website_name)
    except OSError:
        return default
