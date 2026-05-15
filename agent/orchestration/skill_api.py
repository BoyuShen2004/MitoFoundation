"""Public helpers for chat + orchestration skills (imports from ``skill_store``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .skill_store import (
    SkillRecord,
    _parse_front_matter,
    build_skill_document,
    create_skill,
    ensure_skill_trees,
    list_chat_skills,
    list_orchestration_skills,
    merged_chat_skills_block,
    merged_orchestration_skills_block,
    read_skill_document,
    rename_skill_slug,
    write_skill_document,
)


@dataclass
class AgentSkill:
    id: str
    title: str
    body: str


def load_skills(project_root: Path) -> List[AgentSkill]:
    """Orchestration skills only (legacy name)."""
    out: List[AgentSkill] = []
    for r in list_orchestration_skills(project_root):
        out.append(AgentSkill(id=r.id, title=r.title, body=r.body))
    return out


def merged_skill_bodies(project_root: Path) -> str:
    """Concatenate orchestration skill bodies for system prompt injection."""
    return merged_orchestration_skills_block(project_root)


def save_skill_override(project_root: Path, skill_id: str, body: str) -> None:
    """Replace the **markdown body** (below front matter) for an orchestration skill."""
    for r in list_orchestration_skills(project_root):
        if r.id == skill_id or r.slug == skill_id:
            raw = read_skill_document(project_root, "orchestration", r.slug)
            meta, _old = _parse_front_matter(raw.strip())
            if meta:
                write_skill_document(
                    project_root,
                    "orchestration",
                    r.slug,
                    build_skill_document(meta, body.strip()),
                )
            else:
                write_skill_document(project_root, "orchestration", r.slug, body)
            return
    slug = skill_id.strip().replace(" ", "_").lower()
    if not slug.replace("_", "").isalnum():
        slug = "custom_skill"
    write_skill_document(project_root, "orchestration", slug, body)


__all__ = [
    "AgentSkill",
    "SkillRecord",
    "build_skill_document",
    "create_skill",
    "ensure_skill_trees",
    "list_chat_skills",
    "list_orchestration_skills",
    "load_skills",
    "merged_chat_skills_block",
    "merged_orchestration_skills_block",
    "merged_skill_bodies",
    "read_skill_document",
    "rename_skill_slug",
    "save_skill_override",
    "write_skill_document",
]
