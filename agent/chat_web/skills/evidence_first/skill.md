---
id: evidence_first
title: Evidence-first answers from local artifacts
label: Evidence first
---

Purpose
- Answer user questions using local repository evidence first, never from memory or assumption.

Primary evidence sources (exact paths)
- Scrape facts: `1web_scraper_01/outputs/*.probe.json` (newest first), `1web_scraper_01/websites/*/`
- Database / catalogue / DB facts: `2database_builder/outputs/`, `agent/orchestration/registry/schema.py`, `agent/orchestration/registry/providers/`
- Download/preprocess facts: `3data_downloader/outputs/`, `data/` inventory
- Repo structure: `docs/architecture/repo-layout.md`, `README.md`

Behavior
- Prefer current evidence blocks over prior assistant claims.
- For count questions, return direct numeric answers when present.
- If multiple counts are found, report each with source path/field and a one-line difference note.
- Always cite the source file path (relative to repo root) for every factual claim.
- When multiple versions of an artifact exist, prefer the most recently modified; state which file was used.
- Keep answers concise and concrete before offering optional follow-ups.

Not-found behavior
- If evidence is missing, state exactly: "Not found in current artifacts: <expected path or field>".
- Do not infer or extrapolate when the required file or field is absent.
- Propose running the relevant pipeline stage when artifacts are stale or missing.

Observed vs. inferred
- Label inferred statements explicitly: "inferred from path naming" or "inferred from code structure".
- Only promote an inference to a fact when file content confirms it.

Guardrails
- Do not request command approval for read-only factual questions when evidence already contains the answer.
- Never invent dataset counts, schema table names, column names, IDs, or download status.
- If evidence is missing, state exactly which file/path/field is missing.

Mode-aware answering (Ask / Plan / Agent)
- Ask mode (default): answer questions directly from evidence with concise facts first.
- Plan mode: explain proposed plan fields in plain language (sites, stages, n_crops, sample types, filters, expected outputs) before suggesting execution.
- Agent mode: for run-result questions, explain what happened using execution artifacts (stage report, stdout/stderr tails, downloader logs, status tables).
- If a user asks "what does X mean in the plan?", define it first, then map it to concrete files/stages.

App functionality/workflow questions
- For "what does this app do", "what does this page/module do", or "what does this button do":
  1) prioritize repo docs + route handlers (`README.md`, `docs/`, `agent/chat_web/app/routes.py`, `agent/chat_web/app/studio_api.py`)
  2) map UI actions to backend endpoints and pipeline stages
  3) state behavior and side effects (what file/table/status changes)
- If a button behavior is unclear, say "Not found in current artifacts: <component/handler>" and list the nearest confirmed control.

Pipeline-run explanation policy
- When asked why a run failed/succeeded, use `Pipeline Run Report` fields plus per-stage sections.
- Always include: failing stage, direct message/error text, skipped downstream stages, and practical consequence.
- Distinguish:
  - stage failure (command/run failed),
  - stage skipped (dependency failure or filter no-match),
  - pipeline success after training submission (training request accepted, not necessarily final model quality).
