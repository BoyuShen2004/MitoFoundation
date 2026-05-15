"""Discover, read, write, and merge agent skill markdown (chat + orchestration).

Orchestration markdown lives only under ``agent/orchestration/skills/<slug>/`` (no Python files there).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SkillKind = Literal["chat", "orchestration"]

SKILL_FILENAME = "skill.md"


@dataclass
class SkillRecord:
    """One skill folder under chat or orchestration."""

    kind: SkillKind
    slug: str
    id: str
    title: str
    label: str
    body: str


def chat_skills_root(project_root: Path) -> Path:
    """Markdown skills only: ``agent/chat_web/skills/<slug>/skill.md``."""
    return project_root / "agent" / "chat_web" / "skills"


def orchestration_skills_root(project_root: Path) -> Path:
    """Orchestration markdown only: ``agent/orchestration/skills/<slug>/skill.md``."""
    return project_root / "agent" / "orchestration" / "skills"


def _roots(project_root: Path, kind: SkillKind) -> Path:
    return chat_skills_root(project_root) if kind == "chat" else orchestration_skills_root(project_root)


def build_skill_document(meta: dict[str, str], body: str) -> str:
    """Serialize YAML front matter + markdown body for ``skill.md``."""
    preferred = ("id", "title", "label")
    keys: list[str] = []
    seen: set[str] = set()
    for k in preferred:
        if k in meta and str(meta.get(k, "")).strip():
            keys.append(k)
            seen.add(k)
    for k in sorted(meta.keys()):
        if k in seen:
            continue
        if not str(meta.get(k, "")).strip():
            continue
        keys.append(k)
    lines = ["---"]
    for k in keys:
        v = str(meta.get(k, "")).strip()
        lines.append(f"{k}: {v}")
    lines.append("---")
    lines.append("")
    lines.append(body.strip())
    return "\n".join(lines) + "\n"


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
    if not m:
        return {}, text
    meta_raw, body = m.group(1), m.group(2)
    meta: dict[str, str] = {}
    for line in meta_raw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, body


def _slug_ok(slug: str) -> bool:
    return bool(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_\-]{0,127}", slug or ""))


def _record_from_file(kind: SkillKind, slug: str, full_text: str) -> SkillRecord:
    meta, body = _parse_front_matter(full_text.strip())
    sid = (meta.get("id") or slug).strip() or slug
    title = (meta.get("title") or slug.replace("_", " ").title()).strip() or slug
    label = (meta.get("label") or title).strip() or title
    return SkillRecord(
        kind=kind,
        slug=slug,
        id=sid,
        title=title,
        label=label,
        body=body.strip(),
    )


def ensure_skill_trees(project_root: Path) -> None:
    """Create skill roots and migrate legacy flat / old-folder layouts once."""
    chat_skills_root(project_root).mkdir(parents=True, exist_ok=True)
    orchestration_skills_root(project_root).mkdir(parents=True, exist_ok=True)
    _migrate_legacy_chat(project_root)
    _migrate_renamed_chat_slugs(chat_skills_root(project_root))
    _migrate_legacy_orchestration(project_root)
    _ensure_default_orchestration_skills(project_root)


def _migrate_legacy_chat(project_root: Path) -> None:
    """Copy missing slugs from older layouts into ``agent/chat_web/skills/``."""
    dest_root = chat_skills_root(project_root)
    legacy_dirs = [
        # Nested markdown under ``agent/chat_web/agent_skills/chat`` (pre-unification).
        project_root / "agent" / "chat_web" / "agent_skills" / "chat",
        project_root / "chat_web" / "agent_skills" / "chat",
        project_root / "chat_web" / "skills",
    ]
    for legacy in legacy_dirs:
        if not legacy.is_dir():
            continue
        try:
            if legacy.resolve() == dest_root.resolve():
                continue
        except OSError:
            pass
        _copy_legacy_chat_subdirs(legacy, dest_root)


def _migrate_renamed_chat_slugs(dest_root: Path) -> None:
    """Rename canonical chat skill folders when slugs were updated in-repo."""
    pairs = (("repo_rag", "repo_file_grounding"), ("data_functionality_map", "pipeline_stage_map"))
    for old_slug, new_slug in pairs:
        if old_slug == new_slug:
            continue
        old_p = dest_root / old_slug
        new_p = dest_root / new_slug
        if not (old_p / SKILL_FILENAME).is_file():
            continue
        if (new_p / SKILL_FILENAME).is_file():
            continue
        new_p.parent.mkdir(parents=True, exist_ok=True)
        old_p.rename(new_p)


def _copy_legacy_chat_subdirs(legacy: Path, dest_root: Path) -> None:
    for sub in sorted(legacy.iterdir()):
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        slug = sub.name
        if not _slug_ok(slug):
            continue
        old_md = sub / SKILL_FILENAME
        if not old_md.is_file():
            alt = sub / f"{slug}.md"
            if alt.is_file():
                old_md = alt
            else:
                continue
        dest_dir = dest_root / slug
        dest_md = dest_dir / SKILL_FILENAME
        if dest_md.is_file():
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_md.write_text(old_md.read_text(encoding="utf-8"), encoding="utf-8")


def _migrate_legacy_orchestration(project_root: Path) -> None:
    """Copy missing slugs from older layouts into ``agent/orchestration/skills/``."""
    dest_root = orchestration_skills_root(project_root)
    legacy_dirs = [
        project_root / "agent" / "orchestration" / "agent_skills",
        project_root / "agent" / "orchestration" / "agent_skills" / "orchestration",
        project_root / "orchestration" / "agent_skills" / "orchestration",
        project_root / "orchestration" / "skills",
    ]
    for legacy_dir in legacy_dirs:
        if not legacy_dir.is_dir():
            continue
        try:
            if legacy_dir.resolve() == dest_root.resolve():
                continue
        except OSError:
            pass
        _copy_legacy_orchestration_md(legacy_dir, dest_root)


def _copy_legacy_orchestration_md(legacy_dir: Path, dest_root: Path) -> None:
    for md in sorted(legacy_dir.glob("*.md")):
        slug = md.stem
        if not _slug_ok(slug):
            continue
        dest_dir = dest_root / slug
        dest_md = dest_dir / SKILL_FILENAME
        if dest_md.is_file():
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_md.write_text(md.read_text(encoding="utf-8"), encoding="utf-8")
    for sub in sorted(legacy_dir.iterdir()):
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        slug = sub.name
        if not _slug_ok(slug):
            continue
        src = sub / SKILL_FILENAME
        if not src.is_file():
            alt = sub / f"{slug}.md"
            if alt.is_file():
                src = alt
            else:
                continue
        dest_dir = dest_root / slug
        dest_md = dest_dir / SKILL_FILENAME
        if dest_md.is_file():
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_md.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def _ensure_default_orchestration_skills(project_root: Path) -> None:
    root = orchestration_skills_root(project_root)
    ws = root / "web_scrape" / SKILL_FILENAME
    if not ws.is_file():
        (root / "web_scrape").mkdir(parents=True, exist_ok=True)
        ws.write_text(
            "---\nid: web_scrape\ntitle: Stage 1 site snapshot (scrape + probe)\nlabel: Web scrape / probe\n---\n"
            "Use stage ``1web_scraper_01`` to fetch a landing page. **Workspace mode** (Pipeline Studio or "
            "``agent.py URL --workspace``) writes ``1web_scraper_01/websites/<slug>/site.md`` and probe JSON under "
            "``1web_scraper_01/outputs/<slug>.probe.json``. If the URL has no scheme, tooling defaults to ``http://`` "
            "(``https://`` is still allowed). If blocked, report HTTP status and stop.\n",
            encoding="utf-8",
        )
    dl = root / "download_script" / SKILL_FILENAME
    if not dl.is_file():
        (root / "download_script").mkdir(parents=True, exist_ok=True)
        dl.write_text(
            "---\nid: download_script\ntitle: Stage 3 download script (stub, approval-gated)\nlabel: Download script\n---\n"
            "Stage ``3data_downloader`` generates ``3data_downloader/outputs/download_{site}.py`` with "
            "``--n-crops``, ``--spacing-nm``, ``--voxel-nm``, ``--centers-json``, ``--label-centers-json``. "
            "Execution requires explicit user approval.\n",
            encoding="utf-8",
        )


def _list_kind(project_root: Path, kind: SkillKind) -> list[SkillRecord]:
    ensure_skill_trees(project_root)
    root = _roots(project_root, kind)
    out: list[SkillRecord] = []
    if not root.is_dir():
        return out
    skip_dirs = frozenset({"__pycache__"}) if kind == "orchestration" else frozenset()
    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        if sub.name in skip_dirs:
            continue
        md = sub / SKILL_FILENAME
        if not md.is_file():
            continue
        try:
            text = md.read_text(encoding="utf-8")
        except OSError:
            continue
        rec = _record_from_file(kind, sub.name, text)
        out.append(rec)
    return out


def list_chat_skills(project_root: Path) -> list[SkillRecord]:
    return _list_kind(project_root, "chat")


def list_orchestration_skills(project_root: Path) -> list[SkillRecord]:
    return _list_kind(project_root, "orchestration")


def read_skill_document(project_root: Path, kind: SkillKind, slug: str) -> str:
    if not _slug_ok(slug):
        raise ValueError("invalid skill slug")
    ensure_skill_trees(project_root)
    path = _roots(project_root, kind) / slug / SKILL_FILENAME
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path.read_text(encoding="utf-8")


def write_skill_document(project_root: Path, kind: SkillKind, slug: str, document: str) -> None:
    if not _slug_ok(slug):
        raise ValueError("invalid skill slug")
    ensure_skill_trees(project_root)
    root = _roots(project_root, kind) / slug
    root.mkdir(parents=True, exist_ok=True)
    (root / SKILL_FILENAME).write_text(document.rstrip() + "\n", encoding="utf-8")


def create_skill(
    project_root: Path,
    kind: SkillKind,
    slug: str,
    *,
    label: str,
    title: str | None = None,
    id: str | None = None,
    body: str | None = None,
) -> None:
    """Create ``<slug>/skill.md``; raises ``FileExistsError`` if the folder or file already exists."""
    if not _slug_ok(slug):
        raise ValueError("invalid skill slug")
    ensure_skill_trees(project_root)
    dest_dir = _roots(project_root, kind) / slug
    dest_md = dest_dir / SKILL_FILENAME
    if dest_md.is_file():
        raise FileExistsError(str(dest_md))
    lab = label.strip() or slug.replace("_", " ").title()
    tit = (title or lab).strip()
    sid = (id or slug).strip()
    meta = {"id": sid, "title": tit, "label": lab}
    default_body = (
        body.strip()
        if body and body.strip()
        else "Purpose\n- Describe what this skill should steer the agent toward.\n"
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / SKILL_FILENAME).write_text(build_skill_document(meta, default_body), encoding="utf-8")


def rename_skill_slug(project_root: Path, kind: SkillKind, old_slug: str, new_slug: str) -> None:
    """Rename skill folder ``old_slug`` → ``new_slug``."""
    if old_slug == new_slug:
        return
    if not _slug_ok(old_slug) or not _slug_ok(new_slug):
        raise ValueError("invalid skill slug")
    ensure_skill_trees(project_root)
    root = _roots(project_root, kind)
    old_p = root / old_slug
    new_p = root / new_slug
    if not (old_p / SKILL_FILENAME).is_file():
        raise FileNotFoundError(str(old_p / SKILL_FILENAME))
    if new_p.exists():
        raise FileExistsError(str(new_p))
    old_p.rename(new_p)


def _merge_block(kind: SkillKind, records: list[SkillRecord]) -> str:
    if not records:
        return ""
    parts: list[str] = []
    label = "Chat" if kind == "chat" else "Orchestration"
    for r in records:
        if kind == "chat":
            rel = f"agent/chat_web/skills/{r.slug}/{SKILL_FILENAME}"
        else:
            rel = f"agent/orchestration/skills/{r.slug}/{SKILL_FILENAME}"
        parts.append(f"### {label} skill: `{rel}` (id={r.id})\n{r.body}".strip())
    return "\n\n".join(parts)


def merged_chat_skills_block(project_root: Path) -> str:
    return _merge_block("chat", list_chat_skills(project_root))


def merged_orchestration_skills_block(project_root: Path) -> str:
    return _merge_block("orchestration", list_orchestration_skills(project_root))
