---
id: repo_file_grounding
title: Repository file grounding (local scan, not embedding RAG)
label: Repo file scan
---

Purpose
- Ground answers by **reading files in the working tree** (and prioritizing stage output dirs). This is **filesystem + keyword-style matching on text artifacts**, not vector-database retrieval-augmented generation.

Primary evidence sources
- Text-friendly artifacts: `*.probe.json`, `*.md`, `*.json`, `*.py`, `*.ts` (excluding `node_modules`, `.venv`, `__pycache__`, `dist`, `build`).
- Prefer **recent** files by mtime when several match.
- Stage outputs first: `1web_scraper_01/outputs/`, `2database_builder/outputs/`, `3data_downloader/outputs/`, `data/`.

Approach
- Parse and summarize relevant markdown/JSON/text.
- Use query-term matching with light typo tolerance on file contents.
- Prefer structured fields (counts, keys, IDs) when available.
- Surface short **quoted** snippets with path context.

Output discipline
- Answer from evidence first.
- Cite source path and key field for every factual claim.
- Keep optional planning as a separate, explicit follow-up section only when requested.

Guardrails
- Never claim a file contains something without quoting or citing the relevant lines.
- If nothing matches, say there is no matching evidence in scanned artifacts and list the search terms tried.
- Distinguish file metadata (mtime, size) from file content.
- Do not fabricate snippet content; quote directly from scanned text.
