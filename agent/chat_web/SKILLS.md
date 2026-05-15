# Chat skills (markdown)

**Canonical location:** `agent/chat_web/skills/<slug>/skill.md`

Each skill is a folder named by **slug** with a single `skill.md` inside. Older checkouts may use `<slug>/<slug>.md`; on startup the server copies missing slugs from legacy trees (for example `agent/chat_web/agent_skills/chat/` or top-level `chat_web/skills/`).

- YAML front matter: `id`, `title`, and `label` (short name for the Settings list).
- Body text is merged into the chat system prompt under **Chat skills**.

Discovery and migration live in `agent/orchestration/skill_store.py` (shared with orchestration skills); use `skill_api.py` for `load_skills` / overrides. Prefer editing `skills/<slug>/skill.md` here or **Settings → Agent skills** in the UI.
