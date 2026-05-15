---
id: plan_report_explainer
title: Plan and run report explainer
label: Plan/report explainer
---

Purpose
- Explain pipeline plans and execution reports in plain language, especially when users ask "what does this mean?" or "why did this fail/succeed?".

When to apply
- Any question about:
  - plan fields (`sites`, `stages`, `n_crops`, sample/dataset filters, training/inference split),
  - expected outputs from a plan,
  - run summaries (`Pipeline Run Report`),
  - failure causes, consequences, skipped stages, or success interpretation.

Evidence sources (priority order)
1) Current run output in chat (per-stage sections and `Pipeline Run Report` block)
2) Pipeline execution engine + rules:
   - `agent/chat_web/app/pipeline_chat.py`
3) Runtime endpoints/handlers:
   - `agent/chat_web/app/studio_api.py`
   - `agent/chat_web/app/routes.py`
4) Stage/output grounding skill:
   - `pipeline_stage_map`

How to answer plan questions
- First define the requested plan term(s) in one line each.
- Then map each term to concrete behavior:
  - what stage/module runs,
  - what input it uses,
  - what artifact/table/status it writes.
- For "what will this plan produce?", answer as stage-by-stage expected outputs and final deliverables.

How to answer run-report questions
- Always include these four items:
  1) first failing (or decisive) stage,
  2) direct message/error evidence,
  3) downstream effect (skipped/partial stages),
  4) consequence for data/model outputs.
- For successful runs, clarify what "success" means:
  - stage completion and/or training submission succeeded,
  - does not guarantee model quality metrics unless evaluation artifacts confirm it.

Failure consequence patterns
- `scrape` failed -> `database`/`download` for that source may be skipped or stale.
- `database` failed -> downstream download scope may be incomplete/blocked.
- `download` failed -> training inputs may be missing/incomplete.
- `training` failed -> no new checkpoint result from this run.

Output style
- Give a short direct answer first.
- Then provide a compact "Because" section with 1-3 concrete evidence bullets.
- Avoid generic advice before explaining observed evidence.

Guardrails
- Never claim failure cause without quoting/paraphrasing a concrete error/message field.
- Never infer success of downstream artifacts when an upstream stage failed or was skipped.
- If execution evidence is missing, state:
  "Not found in current artifacts: pipeline execution evidence for <stage/run>."
