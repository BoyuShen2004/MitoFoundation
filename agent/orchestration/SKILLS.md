# Agent skills layout

- **Orchestration** markdown only: `agent/orchestration/skills/<slug>/skill.md` (no Python files in that tree). Logic: `agent/orchestration/skill_store.py`, facade: `skill_api.py`.
- **Chat** markdown: `agent/chat_web/skills/<slug>/skill.md`. Same `skill_store` / `skill_api` modules.
