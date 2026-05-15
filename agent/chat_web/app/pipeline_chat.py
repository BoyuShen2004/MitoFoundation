"""Chat-side data pipeline planner/executor (scrape, database, download, training)."""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from agent.orchestration.session_pipeline import PipelineSession, PipelineStep
from agent.orchestration.time_utils import now_us_eastern_iso

from .llm import LLMUnavailableError, complete

VALID_SOURCES = ("OpenOrganelle", "BossDB", "LocalHPC")
WEBSITE_SOURCES = ("OpenOrganelle", "BossDB")
SITE_SLUG = {"OpenOrganelle": "openorganelle_01", "BossDB": "bossdb_01"}
SITE_PROBE_REL = {
    "OpenOrganelle": "1web_scraper_01/outputs/OpenOrganelle.probe.json",
    "BossDB": "1web_scraper_01/outputs/BossDB.probe.json",
}
DOWNLOADER_SITE = {"OpenOrganelle": "openorganelle", "BossDB": "bossdb"}

_VALID_PIPELINE_STAGES = frozenset({"scrape", "database", "download", "training"})
_PIPELINE_STAGE_ORDER = {"scrape": 0, "database": 1, "download": 2, "training": 3}

_LOCAL_HPC_ALIASES = (
    "local hpc",
    "hpc local",
    "hpc-local",
    "local-hpc",
    "mitole local",
    "local mito",
    "local mitole",
    "mitole",
)


def _canonical_pipeline_stage_token(raw: Any) -> str | None:
    """Normalize scrape/database/download; treat legacy ``schema`` as database."""
    s = str(raw or "").strip().lower()
    if s == "schema":
        return "database"
    if s == "train":
        return "training"
    if s in _VALID_PIPELINE_STAGES:
        return s
    return None


def _canonicalize_plan_stages_list(stages: list[Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for x in stages:
        tok = _canonical_pipeline_stage_token(x)
        if tok is None:
            continue
        if tok not in seen:
            seen.add(tok)
            ordered.append(tok)
    return sorted(ordered, key=lambda s: _PIPELINE_STAGE_ORDER.get(s, 999))


def _normalize_string_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for x in raw:
        s = str(x or "").strip()
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def _clamp_int(raw: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(raw)
        return max(lo, min(hi, v))
    except (TypeError, ValueError):
        return default


def _clamp_ratio(raw: Any) -> float | None:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _normalize_download_target(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    target = str(raw.get("target") or "").strip().lower()
    if target in {"training", "inference", "split"}:
        out["target"] = target
    if "total_crops" in raw:
        out["total_crops"] = _clamp_int(raw.get("total_crops"), 1, 16, 1)
    if "training_crops" in raw:
        out["training_crops"] = _clamp_int(raw.get("training_crops"), 0, 16, 0)
    if "inference_crops" in raw:
        out["inference_crops"] = _clamp_int(raw.get("inference_crops"), 0, 16, 0)
    ratio = _clamp_ratio(raw.get("training_ratio"))
    if ratio is not None:
        out["training_ratio"] = ratio
    # If planner provided explicit split counts but omitted total_crops,
    # infer the per-dataset total so downstream `plan.n_crops` stays consistent.
    if "total_crops" not in out and ("training_crops" in out or "inference_crops" in out):
        total = int(out.get("training_crops", 0) or 0) + int(out.get("inference_crops", 0) or 0)
        if total > 0:
            out["total_crops"] = _clamp_int(total, 1, 16, 1)
    return out


def _norm_sample_type_token(v: str) -> str:
    s = str(v or "").strip().lower()
    if not s:
        return ""
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def _sample_type_matches(rule_values: set[str], sample_type_value: str) -> bool:
    """Loose matching so 'liver' matches 'mouse liver' / 'mouse_liver'."""
    if not rule_values:
        return False
    st = _norm_sample_type_token(sample_type_value) or "unknown"
    st_parts = set(st.split())
    for raw in rule_values:
        rv = _norm_sample_type_token(raw)
        if not rv:
            continue
        if rv == st:
            return True
        if rv in st or st in rv:
            return True
        rv_parts = set(rv.split())
        if rv_parts and rv_parts.issubset(st_parts):
            return True
    return False


def _norm_dataset_name_token(v: str) -> str:
    s = str(v or "").strip().lower()
    if not s:
        return ""
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def _dataset_name_matches(rule_values: set[str], dataset_name_value: str) -> bool:
    """Loose matching so 'hela' matches 'jrc_hela-1' and tokenized variants."""
    if not rule_values:
        return False
    ds = _norm_dataset_name_token(dataset_name_value)
    if not ds:
        return False
    ds_parts = set(ds.split())
    for raw in rule_values:
        rv = _norm_dataset_name_token(raw)
        if not rv:
            continue
        if rv == ds:
            return True
        if rv in ds or ds in rv:
            return True
        rv_parts = set(rv.split())
        if rv_parts and rv_parts.issubset(ds_parts):
            return True
    return False


def _normalize_download_config(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    out: dict[str, Any] = {}
    base_target = _normalize_download_target(raw)
    out.update(base_target)
    inc_st = _normalize_string_list(raw.get("sample_types_include"))
    exc_st = _normalize_string_list(raw.get("sample_types_exclude"))
    inc_ds = _normalize_string_list(raw.get("datasets_include"))
    exc_ds = _normalize_string_list(raw.get("datasets_exclude"))
    if inc_st:
        out["sample_types_include"] = inc_st
    if exc_st:
        out["sample_types_exclude"] = exc_st
    if inc_ds:
        out["datasets_include"] = inc_ds
    if exc_ds:
        out["datasets_exclude"] = exc_ds
    pdt = raw.get("per_dataset_targets")
    if isinstance(pdt, dict):
        fixed: dict[str, dict[str, Any]] = {}
        for k, v in pdt.items():
            ds = str(k or "").strip()
            if not ds:
                continue
            row = _normalize_download_target(v)
            if row:
                fixed[ds] = row
        if fixed:
            out["per_dataset_targets"] = fixed
    return out or None


def _canonical_site_name(raw: Any) -> str | None:
    """Map LLM/user strings to canonical source names (case/spacing tolerant)."""
    if not isinstance(raw, str):
        return None
    t = raw.strip()
    if not t:
        return None
    low = re.sub(r"[\s_]+", "", t.lower())
    if "openorganelle" in low or low in ("oo", "o.o"):
        return "OpenOrganelle"
    if "bossdb" in low or "boss db" in t.lower():
        return "BossDB"
    if any(a.replace(" ", "") in low for a in [x.replace(" ", "") for x in _LOCAL_HPC_ALIASES]):
        return "LocalHPC"
    if t in VALID_SOURCES:
        return t
    return None


def _normalize_plan_sites_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for x in raw:
        c = _canonical_site_name(x)
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _normalize_plan_sites_with_unknowns(raw: Any) -> tuple[list[str], list[str]]:
    if not isinstance(raw, list):
        return [], []
    out: list[str] = []
    unknown: list[str] = []
    seen: set[str] = set()
    seen_u: set[str] = set()
    for x in raw:
        sx = str(x or "").strip()
        if not sx:
            continue
        c = _canonical_site_name(sx)
        if c:
            if c not in seen:
                seen.add(c)
                out.append(c)
            continue
        kl = sx.lower()
        if kl not in seen_u:
            seen_u.add(kl)
            unknown.append(sx)
    return out, unknown


def _sources_from_text(text: str) -> list[str]:
    low = str(text or "").lower()
    hits: list[tuple[int, str]] = []
    patterns = (
        (r"\b(open\s*organelle|openorganelle|\boo\b)\b", "OpenOrganelle"),
        (r"\b(boss\s*db|bossdb)\b", "BossDB"),
        (r"\b(local\s*hpc|hpc\s*local|hpc-local|local-hpc|mitole\s*local|local\s*mito|mitole)\b", "LocalHPC"),
    )
    for pat, source in patterns:
        m = re.search(pat, low)
        if m:
            hits.append((m.start(), source))
    if re.search(r"\bboth\s+(websites|sites)\b|\ball\s+websites\b", low):
        hits.extend([(9990, "OpenOrganelle"), (9991, "BossDB")])
    if re.search(r"\b(all\s+sources|both\s+providers|all\s+providers)\b", low):
        hits.extend([(9990, "OpenOrganelle"), (9991, "BossDB"), (9992, "LocalHPC")])
    hits.sort(key=lambda x: x[0])
    out: list[str] = []
    seen: set[str] = set()
    for _, s in hits:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def message_touches_pipeline(message: str) -> bool:
    m = (message or "").lower()
    return bool(
        re.search(
            r"\b(scrape|database|schema|download|training|train|openorganelle|bossdb|local\s*hpc|hpc|mitole|pipeline|catalog|inventory|sqlite|crop|crops|stage\s*[12345])\b",
            m,
        )
    )


PLAN_JSON_LINE = "PIPELINE_PLAN_JSON:"
INTENT_JSON_LINE = "CHAT_INTENT_JSON:"

_INTENTS = {"qa", "pipeline_plan", "pipeline_execute", "unknown"}
_REQUIRES_MODE = {"ask", "plan", "agent"}


def llm_complete_pipeline_plan(
    project_root: Path,
    user_message: str,
    *,
    mode: str,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any] | None:
    """LLM-only repair pass for incomplete pipeline plans."""
    _ = project_root
    sys = (
        "You are a strict planner for mitoFoundation2 stages 1-4.\n"
        "Return exactly one final line:\n"
        f"{PLAN_JSON_LINE}"
        "<minified JSON>\n"
        "JSON keys are required: sites, n_crops, rationale, and either stages OR per_site_stages.\n"
        "sites: non-empty array in execution order from ['OpenOrganelle','BossDB','LocalHPC'].\n"
        "stages: optional array from ['scrape','database','download','training'] — same stages for every site in order. "
        "Omit or leave null when you use per_site_stages instead.\n"
        "per_site_stages: optional object mapping each site name to an ordered stage array, "
        "when the user wants different work per provider (e.g. scrape only BossDB but run local HPC download too). "
        "Keys must be any of 'OpenOrganelle','BossDB','LocalHPC'; values are subsets of ['scrape','database','download','training']. "
        "Include a key for every site listed in sites.\n"
        "Stage meanings: scrape = Stage 1 web scrape/probe refresh; database = Stage 2 database builder (catalog/inventory from probe); "
        "download = Stage 3 download + preprocess; training = submit nnUNet training. "
        "Preserve pipeline order scrape -> database -> download -> training when multiple apply.\n"
        "For download requests, include a `download` object when possible:\n"
        "- target: 'training'|'inference'|'split'\n"
        "- total_crops: int (per selected dataset)\n"
        "- training_crops / inference_crops: int (optional)\n"
        "- training_ratio: float in [0,1] (optional)\n"
        "- sample_types_include / sample_types_exclude: string arrays\n"
        "- datasets_include / datasets_exclude: string arrays\n"
        "- per_dataset_targets: object keyed by dataset name with target/count overrides.\n"
        "Defaults when unspecified: all sample types, all datasets, target='training'.\n"
        "Respect negation and exclusivity: if the user says not to scrape a provider, do not include 'scrape' for that provider "
        "(use per_site_stages). If they say 'only BossDB' for scraping, BossDB is the only site with scrape; other requested sources may still run database/download if asked.\n"
        "If they ask to build catalog/inventory/SQLite, a database from the probe, or 'Stage 2', include 'database' where they asked for it.\n"
        "Legacy plans may say 'schema' for Stage 2 — treat that as 'database'.\n"
        "If they name both OpenOrganelle and BossDB (or 'both websites') for the same stage, sites must include both strings exactly.\n"
        "If user asks for all websites, include OpenOrganelle and BossDB. If user asks for all sources/providers, include OpenOrganelle, BossDB, and LocalHPC.\n"
        "LocalHPC means local MitoLE/HPC pipeline. For LocalHPC, default to download stage unless user explicitly asks for local catalog build.\n"
        "Do not assume a fixed provider order. Preserve provider order from the user's wording when explicit; "
        "if no order is implied, choose a stable order and keep it consistent in the returned plan.\n"
        "n_crops: integer 1-64.\n"
        "Interpret natural language flexibly (e.g. 'run scraper for bossdb' -> stages ['scrape'], sites ['BossDB']).\n"
        "Context carryover rule: if the latest user turn is elliptical (e.g. 'do it', 'please do the scrape', "
        "'proceed', 'go ahead'), inherit site/stage intent from the immediate prior conversation turns.\n"
        "Never drop required keys because of elliptical phrasing.\n"
        f"Current mode: {mode}\n"
    )
    messages = [{"role": "system", "content": sys}]
    for m in (history or [])[-8:]:
        role = str(m.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": (user_message or "").strip()})
    try:
        raw = complete(messages, max_tokens=500)
    except LLMUnavailableError:
        return None
    if PLAN_JSON_LINE not in raw:
        return None
    try:
        _, rest = raw.rsplit(PLAN_JSON_LINE, 1)
        line = rest.strip().splitlines()[0].strip()
        parsed = json.loads(line)
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    out = _coerce_pipeline_plan(parsed, user_message=user_message, history=history)
    if not isinstance(out.get("sites"), list) or not out.get("sites"):
        return None
    has_stages = isinstance(out.get("stages"), list) and bool(out.get("stages"))
    has_pss = isinstance(out.get("per_site_stages"), dict) and bool(out.get("per_site_stages"))
    if not has_stages and not has_pss:
        return None
    return out


def _normalize_per_site_stages_value(raw: Any) -> dict[str, list[str]] | None:
    """LLM-authored per-provider stage lists (canonical site keys)."""
    if not isinstance(raw, dict):
        return None
    out: dict[str, list[str]] = {}
    for key, val in raw.items():
        site = _canonical_site_name(key)
        if not site:
            continue
        if not isinstance(val, list):
            continue
        ordered = _canonicalize_plan_stages_list(list(val))
        if not ordered:
            continue
        out[site] = ordered
    return out or None


def _coerce_pipeline_plan(
    candidate: dict[str, Any],
    *,
    user_message: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(candidate.get("sites"), list):
        clean, unknown = _normalize_plan_sites_with_unknowns(candidate["sites"])
        if clean:
            out["sites"] = clean
        if unknown:
            out["unsupported_sources"] = unknown
    if isinstance(candidate.get("stages"), list):
        ordered = _canonicalize_plan_stages_list(candidate["stages"])
        if ordered:
            out["stages"] = ordered
    if not out.get("sites") and user_message is not None:
        inferred = _sources_from_text(str(user_message or ""))
        if not inferred:
            for hm in reversed((history or [])[-8:]):
                if str(hm.get("role") or "").strip().lower() != "user":
                    continue
                inferred = _sources_from_text(str(hm.get("content") or ""))
                if inferred:
                    break
        if inferred:
            out["sites"] = inferred
    pss = _normalize_per_site_stages_value(candidate.get("per_site_stages"))
    if pss:
        out["per_site_stages"] = pss
    # Structural safety for malformed per-site plans: never leave requested sites without stage lists.
    sites0 = out.get("sites") if isinstance(out.get("sites"), list) else []
    pss0 = out.get("per_site_stages") if isinstance(out.get("per_site_stages"), dict) else {}
    if sites0 and pss0:
        shared = out.get("stages") if isinstance(out.get("stages"), list) else []
        if not shared:
            seen_u: set[str] = set()
            shared_u: list[str] = []
            for lst in pss0.values():
                if not isinstance(lst, list):
                    continue
                for s in lst:
                    sn = _canonical_pipeline_stage_token(s)
                    if sn and sn not in seen_u:
                        seen_u.add(sn)
                        shared_u.append(sn)
            shared = _canonicalize_plan_stages_list(shared_u)
            if shared:
                out["stages"] = shared
        if shared:
            for s in sites0:
                if s not in pss0:
                    pss0[s] = list(shared)
            out["per_site_stages"] = pss0
    dl = _normalize_download_config(candidate.get("download"))
    if dl:
        if user_message is not None and "per_dataset_targets" not in dl:
            stages_ck = _canonicalize_plan_stages_list(candidate.get("stages") or [])
            per_site_ck = (
                _normalize_per_site_stages_value(candidate["per_site_stages"])
                if isinstance(candidate.get("per_site_stages"), dict)
                else None
            )
            has_download = ("download" in stages_ck) or bool(
                per_site_ck and any("download" in v for v in per_site_ck.values())
            )
            if has_download:
                texts = [str(user_message or "")]
                for hm in (history or [])[-8:]:
                    if str(hm.get("role") or "").strip().lower() == "user":
                        texts.append(str(hm.get("content") or ""))
                blob = "\n".join(texts).lower()
                explicit = bool(
                    re.search(r"\b\d+\s*crops?\b", blob)
                    or re.search(
                        r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen)\s+crops?\b",
                        blob,
                    )
                    or re.search(r"\b(training|inference|testing)\b.{0,24}\b\d+\b", blob)
                    or re.search(r"\b\d+\b.{0,24}\b(training|inference|testing)\b", blob)
                )
                if not explicit:
                    dl["target"] = "training"
                    dl["total_crops"] = 1
                    dl["training_crops"] = 1
                    dl["inference_crops"] = 0
        out["download"] = dl
        if "total_crops" in dl:
            out["n_crops"] = int(dl["total_crops"])
    elif "n_crops" in candidate:
        out["n_crops"] = _clamp_int(candidate.get("n_crops"), 1, 64, 1)
    if "n_crops" not in out:
        out["n_crops"] = 1
    if isinstance(candidate.get("rationale"), str) and candidate["rationale"].strip():
        out["rationale"] = candidate["rationale"].strip()
    return out


def _normalize_intent_decision(
    raw: dict[str, Any],
    *,
    mode: str,
    user_message: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "intent": "unknown",
        "confidence": 0.0,
        "requires_mode": mode if mode in _REQUIRES_MODE else "ask",
        "plan": None,
        "rationale": "",
        "routed_via_llm": True,
        "ui_focus": "",
    }
    intent = str(raw.get("intent", "")).strip().lower()
    req_mode = str(raw.get("requires_mode", "")).strip().lower()
    if intent in _INTENTS:
        out["intent"] = intent
    if req_mode in _REQUIRES_MODE:
        out["requires_mode"] = req_mode
    try:
        conf = float(raw.get("confidence", out["confidence"]))
        out["confidence"] = max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        pass
    if isinstance(raw.get("rationale"), str) and raw["rationale"].strip():
        out["rationale"] = raw["rationale"].strip()
    if isinstance(raw.get("ui_focus"), str) and raw["ui_focus"].strip():
        out["ui_focus"] = raw["ui_focus"].strip()[:200]
    plan = raw.get("plan")
    if isinstance(plan, dict):
        out["plan"] = None
        coerced = _coerce_pipeline_plan(plan, user_message=user_message, history=history)
        if coerced:
            out["plan"] = coerced
    return out


def planning_trace_from_intent_decision(decision: dict[str, Any], draft_plan: dict[str, Any] | None = None) -> dict[str, Any]:
    """Labels for the chat waiting UI, derived from the LLM intent + structured plan (not regex on user text)."""
    uif = str(decision.get("ui_focus") or "").strip()[:200] if decision else ""
    plan = draft_plan if isinstance(draft_plan, dict) and draft_plan else (decision.get("plan") if isinstance(decision.get("plan"), dict) else None)
    sites: list[str] = []
    stages: list[str] = []
    if isinstance(plan, dict):
        if isinstance(plan.get("sites"), list):
            sites = _normalize_plan_sites_list(plan["sites"])
        if isinstance(plan.get("stages"), list):
            stages = _canonicalize_plan_stages_list(plan["stages"])
    # For display, structured plan.sites always wins (multi-provider runs). A short ui_focus from the LLM
    # can list only one provider even when plan.sites is complete — do not let it hide the rest.
    if sites:
        display_focus = " → ".join(sites)
    elif uif:
        display_focus = uif
    else:
        display_focus = "general"
    return {"planning_ui_focus": display_focus, "plan_sites": sites, "plan_stages": stages}


def llm_route_chat_intent(
    project_root: Path,
    user_message: str,
    *,
    mode: str,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """LLM-only chat intent router for agentic dispatch."""
    sys = (
        "You are the intent router for mitoFoundation2 chat.\n"
        "Decide the single best next task and output short rationale text, then final line:\n"
        f"{INTENT_JSON_LINE}"
        "<minified JSON>\n"
        "JSON keys: intent, confidence, requires_mode, rationale, plan, ui_focus.\n"
        "ui_focus: short English phrase, max 12 words, optional: scope summary when no pipeline plan. "
        "When you output plan.sites, list every provider the user asked for, in order; the UI will show all sites from plan.sites. "
        "If multiple sites apply, do not set ui_focus to a single source — prefer leaving ui_focus short or use 'both catalogs' and "
        "rely on plan.sites. Examples: 'BossDB', 'OpenOrganelle and BossDB', 'local training volumes'.\n"
        "intent in ['qa','pipeline_plan','pipeline_execute','unknown'].\n"
        "requires_mode in ['ask','plan','agent'].\n"
        "Set plan only for pipeline_plan/pipeline_execute; else null.\n"
        "plan.sites values must be from ['OpenOrganelle','BossDB','LocalHPC'].\n"
        "plan.stages must reflect every stage the user asked for when using one global stage list for all sites: "
        "scrape (crawl/probe), database (database builder / catalog+inventory / Stage 2), download (fetch volumes/crops/Stage 3). "
        "You may also use training (submit nnUNet training / model training stage).\n"
        "If stages differ by provider, set plan.per_site_stages instead (and you may still set plan.stages to the union for summary).\n"
        "When using per_site_stages, include a key for every site listed in plan.sites (each value is a non-empty stage array).\n"
        "Respect negation: phrases like 'do not scrape X', 'skip X scrape', 'only scrape Y' must be reflected in plan.sites "
        "and/or plan.per_site_stages so X does not get a scrape stage.\n"
        "plan.sites must list every provider that should receive any pipeline work, in execution order.\n"
        "If user asks for all websites, plan.sites must include OpenOrganelle and BossDB.\n"
        "If user asks for all sources/providers, plan.sites must include OpenOrganelle, BossDB, and LocalHPC.\n"
        "Treat LocalHPC as local MitoLE/HPC pipeline source; include it when user asks for local hpc/local mito/mitole local.\n"
        "Respect exclusivity and negation across all sources: 'only bossdb', 'only local hpc', 'do not scrape X'.\n"
        "Do not assume a fixed provider order. Preserve provider order from the user's wording when explicit; "
        "if no order is implied, choose a stable order and keep it consistent in the returned plan.\n"
        "When download is requested, include plan.download as structured JSON when possible:\n"
        "- target: 'training' | 'inference' | 'split'\n"
        "- total_crops: int (per selected dataset)\n"
        "- training_crops: int (optional)\n"
        "- inference_crops: int (optional)\n"
        "- training_ratio: float in [0,1] (optional; used to split total_crops)\n"
        "- sample_types_include / sample_types_exclude: string arrays\n"
        "- datasets_include / datasets_exclude: string arrays\n"
        "- per_dataset_targets: object mapping dataset name to target/training_crops/inference_crops/total_crops/training_ratio.\n"
        "Defaults when unspecified: all sample types, all datasets, target='training'.\n"
        "Set plan.n_crops to a reasonable integer for compatibility; usually plan.download.total_crops when present.\n"
        "Context carryover rule: if the newest user turn is elliptical ('do it', 'please do the scrape', "
        "'proceed', 'go ahead'), resolve intent/plan by inheriting provider and stage scope from recent turns.\n"
        "For pipeline intents, prefer returning a complete plan over leaving sites/stages empty.\n"
        "For pipeline_execute, use only when user explicitly asks to run/execute.\n"
        f"Current mode: {mode}\n"
    )
    messages = [{"role": "system", "content": sys}]
    for m in (history or [])[-8:]:
        role = str(m.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": (user_message or "").strip()})
    unavailable = {
        "intent": "unknown",
        "confidence": 0.0,
        "requires_mode": mode if mode in _REQUIRES_MODE else "ask",
        "plan": None,
        "rationale": "LLM routing unavailable.",
        "routed_via_llm": False,
        "ui_focus": "",
    }
    try:
        raw = complete(messages, max_tokens=500)
    except LLMUnavailableError:
        return unavailable
    if INTENT_JSON_LINE not in raw:
        return unavailable
    try:
        _, rest = raw.rsplit(INTENT_JSON_LINE, 1)
        line = rest.strip().splitlines()[0].strip()
        parsed = json.loads(line)
    except (ValueError, json.JSONDecodeError):
        return unavailable
    if not isinstance(parsed, dict):
        return unavailable
    out = _normalize_intent_decision(parsed, mode=mode, user_message=user_message, history=history)
    return out


def _dataset_counts_from_target(target_cfg: dict[str, Any], fallback_total: int) -> tuple[int, int]:
    total = _clamp_int(target_cfg.get("total_crops"), 1, 16, fallback_total)
    tr_raw = target_cfg.get("training_crops")
    inf_raw = target_cfg.get("inference_crops")
    if tr_raw is not None or inf_raw is not None:
        tr = _clamp_int(tr_raw, 0, 16, 0)
        inf = _clamp_int(inf_raw, 0, 16, 0)
        if tr + inf <= 0:
            tr = total
            inf = 0
        if tr + inf > 16:
            inf = max(0, 16 - tr)
        return tr, inf
    target = str(target_cfg.get("target") or "training").strip().lower()
    if target == "inference":
        return 0, total
    if target == "split":
        ratio = _clamp_ratio(target_cfg.get("training_ratio"))
        if ratio is None:
            ratio = 0.5
        tr = int(round(total * ratio))
        tr = max(0, min(total, tr))
        inf = total - tr
        return tr, inf
    return total, 0


def _build_dataset_splits_from_plan(preview: dict[str, Any], download_cfg: dict[str, Any]) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    names = [str(x).strip() for x in (preview.get("datasets") or []) if str(x).strip()]
    rows = preview.get("dataset_rows") or []
    sample_type_of: dict[str, str] = {}
    if isinstance(rows, list):
        for r in rows:
            if not isinstance(r, dict):
                continue
            dn = str(r.get("dataset_name") or "").strip()
            if not dn:
                continue
            st = str(r.get("sample_type") or "").strip() or "unknown"
            sample_type_of[dn] = st

    include_ds = {x.lower() for x in _normalize_string_list(download_cfg.get("datasets_include"))}
    exclude_ds = {x.lower() for x in _normalize_string_list(download_cfg.get("datasets_exclude"))}
    include_st = {x.lower() for x in _normalize_string_list(download_cfg.get("sample_types_include"))}
    exclude_st = {x.lower() for x in _normalize_string_list(download_cfg.get("sample_types_exclude"))}
    global_target = _normalize_download_target(download_cfg)
    if not global_target:
        global_target = {"target": "training", "total_crops": 1}
    fallback_total = _clamp_int(global_target.get("total_crops"), 1, 16, 1)

    pdt = download_cfg.get("per_dataset_targets")
    per_ds: dict[str, dict[str, Any]] = pdt if isinstance(pdt, dict) else {}

    selected: list[str] = []
    for ds in names:
        if include_ds and not _dataset_name_matches(include_ds, ds):
            continue
        if exclude_ds and _dataset_name_matches(exclude_ds, ds):
            continue
        st = str(sample_type_of.get(ds, "unknown")).strip().lower() or "unknown"
        if include_st and not _sample_type_matches(include_st, st):
            continue
        if exclude_st and _sample_type_matches(exclude_st, st):
            continue
        selected.append(ds)

    splits: dict[str, dict[str, int]] = {}
    max_total = 1
    max_training = 0
    max_inference = 0
    for ds in selected:
        target_cfg = dict(global_target)
        if ds in per_ds and isinstance(per_ds[ds], dict):
            target_cfg.update(per_ds[ds])
        tr, inf = _dataset_counts_from_target(target_cfg, fallback_total=fallback_total)
        if tr + inf <= 0:
            continue
        splits[ds] = {"training": tr, "inference": inf}
        max_total = max(max_total, tr + inf)
        max_training = max(max_training, tr)
        max_inference = max(max_inference, inf)

    meta = {
        "selected_count": len(splits),
        "candidate_count": len(names),
        "sample_types_include": sorted(include_st),
        "sample_types_exclude": sorted(exclude_st),
        "datasets_include": sorted(include_ds),
        "datasets_exclude": sorted(exclude_ds),
        "n_crops": max_total,
        "n_crops_training": max_training,
        "n_crops_inference": max_inference,
        "sample_type_resolution": (
            "specified"
            if include_st or exclude_st
            else "unspecified"
        ),
    }
    return splits, meta


def _enforce_sample_type_scope_on_splits(
    preview: dict[str, Any],
    download_cfg: dict[str, Any],
    dataset_splits: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    """Apply include/exclude sample-type filters as a hard gate on computed splits."""
    if not dataset_splits:
        return {}
    include_st = {x.lower() for x in _normalize_string_list(download_cfg.get("sample_types_include"))}
    exclude_st = {x.lower() for x in _normalize_string_list(download_cfg.get("sample_types_exclude"))}
    if not include_st and not exclude_st:
        return dict(dataset_splits)

    rows = preview.get("dataset_rows") or []
    sample_type_of: dict[str, str] = {}
    if isinstance(rows, list):
        for r in rows:
            if not isinstance(r, dict):
                continue
            dn = str(r.get("dataset_name") or "").strip()
            if not dn:
                continue
            sample_type_of[dn] = (str(r.get("sample_type") or "").strip() or "unknown").lower()

    filtered: dict[str, dict[str, int]] = {}
    for ds, split in dataset_splits.items():
        st = sample_type_of.get(ds, "unknown")
        if include_st and not _sample_type_matches(include_st, st):
            continue
        if exclude_st and _sample_type_matches(exclude_st, st):
            continue
        filtered[ds] = split
    return filtered


def _enforce_dataset_scope_on_splits(
    download_cfg: dict[str, Any],
    dataset_splits: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    """Apply include/exclude dataset filters as a hard gate on computed splits."""
    if not dataset_splits:
        return {}
    include_ds = {x.lower() for x in _normalize_string_list(download_cfg.get("datasets_include"))}
    exclude_ds = {x.lower() for x in _normalize_string_list(download_cfg.get("datasets_exclude"))}
    if not include_ds and not exclude_ds:
        return dict(dataset_splits)
    filtered: dict[str, dict[str, int]] = {}
    for ds, split in dataset_splits.items():
        if include_ds and not _dataset_name_matches(include_ds, ds):
            continue
        if exclude_ds and _dataset_name_matches(exclude_ds, ds):
            continue
        filtered[ds] = split
    return filtered


def _llm_expand_download_selection(
    *,
    download_cfg: dict[str, Any],
    preview: dict[str, Any],
    n_crops_default: int,
) -> tuple[dict[str, dict[str, int]], dict[str, Any] | None]:
    """Use LLM to map download intent to concrete per-dataset splits."""
    rows = preview.get("dataset_rows")
    row_list = rows if isinstance(rows, list) else []
    preview_payload = {
        "datasets": [str(x) for x in (preview.get("datasets") or []) if str(x).strip()],
        "dataset_rows": [
            {
                "dataset_name": str(r.get("dataset_name") or "").strip(),
                "sample_type": str(r.get("sample_type") or "").strip() or "unknown",
            }
            for r in row_list
            if isinstance(r, dict) and str(r.get("dataset_name") or "").strip()
        ],
    }
    sys = (
        "You are a strict execution expander for mitoFoundation2 downloader.\n"
        "Given planner intent and available datasets, output concrete per-dataset crop splits.\n"
        "Return exactly one final line:\n"
        "DOWNLOAD_EXEC_JSON:<minified JSON>\n"
        "JSON keys:\n"
        "- dataset_splits: object {dataset_name: {training:int, inference:int}}\n"
        "- sample_type_resolution: string ('unspecified' when sample type was not explicitly provided)\n"
        "- notes: short string.\n"
        "Constraints:\n"
        "- training/inference must be ints in [0,16].\n"
        "- if user intent excludes a dataset/sample, set both to 0 or omit it.\n"
        "- if target is training, inference should be 0 unless explicit override.\n"
        "- if target is inference, training should be 0 unless explicit override.\n"
        "- if both are unspecified, default is training-only with n_crops_default.\n"
    )
    usr = (
        "download_cfg=\n"
        f"{json.dumps(download_cfg, ensure_ascii=True)}\n\n"
        "preview=\n"
        f"{json.dumps(preview_payload, ensure_ascii=True)}\n\n"
        f"n_crops_default={int(max(1, n_crops_default))}"
    )
    try:
        raw = complete(
            [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
            max_tokens=1200,
        )
    except LLMUnavailableError:
        return {}, None
    tag = "DOWNLOAD_EXEC_JSON:"
    if tag not in raw:
        return {}, None
    try:
        _, rest = raw.rsplit(tag, 1)
        line = rest.strip().splitlines()[0].strip()
        parsed = json.loads(line)
    except (ValueError, json.JSONDecodeError):
        return {}, None
    if not isinstance(parsed, dict):
        return {}, None
    ds = parsed.get("dataset_splits")
    if not isinstance(ds, dict):
        return {}, None
    fixed: dict[str, dict[str, int]] = {}
    for k, v in ds.items():
        dn = str(k or "").strip()
        if not dn or not isinstance(v, dict):
            continue
        tr = _clamp_int(v.get("training"), 0, 16, 0)
        inf = _clamp_int(v.get("inference"), 0, 16, 0)
        if tr + inf <= 0:
            continue
        fixed[dn] = {"training": tr, "inference": inf}
    meta = {
        "sample_type_resolution": str(parsed.get("sample_type_resolution") or "").strip() or "unspecified",
        "notes": str(parsed.get("notes") or "").strip(),
        "source": "llm",
    }
    return fixed, meta



def ask_mode_plan_switch_reply(message: str) -> str | None:
    """Ask mode: redirect when the user is asking to run/save structured plans."""
    m = (message or "").lower()
    if re.search(r"\b(run|execute)\s+(the\s+)?plan\b", m):
        return (
            "That action needs **Agent** mode (approve and run) or **Plan** mode (draft and save). "
            "Use the mode menu to the left of the message box."
        )
    if re.search(r"\b(save|persist)\b.*\bapproved\s+plan\b", m) or re.search(
        r"\bapprove\b.*\b(plan|pipeline)\b.*\b(run|execute)\b", m
    ):
        return "Switch to **Agent** mode to approve and execute a pipeline plan from chat."
    return None


def plan_mode_execution_refusal(message: str) -> str | None:
    m = (message or "").lower()
    if ("why" in m or "how come" in m) and ("can't" in m or "cannot" in m or "won't" in m):
        if "execut" in m or "run" in m:
            return (
                "In **Plan** mode I only draft and save plans. Switch to **Agent** mode to execute them "
                "after approval."
            )
    if re.search(r"\b(run|execute)\b.*\b(pipeline|download|scrape)\b", m) and "plan" not in m:
        return "Switch to **Agent** mode to run scrape / database / download from chat."
    return None


def execute_pipeline_plan(
    project_root: Path,
    session_id: str,
    session: PipelineSession,
    plan: dict[str, Any],
    *,
    stop_requested: Callable[[], bool] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Run scrape → database → download per site using the same entrypoints as Pipeline Studio."""
    # Local import avoids circular import at module load (studio_api imports agent_turn).
    from .studio_api import (
        studio_run_database_build,
        studio_run_downloader,
        studio_run_mitole_downloader,
        studio_run_training,
        studio_downloader_preview,
        studio_mitole_catalogue,
        studio_scrape_website,
    )
    from .studio_api import (
        StudioDatabaseBuildBody,
        StudioDownloaderBody,
        StudioMitoLeDownloaderBody,
        StudioSessionBody,
        StudioWebsiteScrapeBody,
    )

    if isinstance(plan, dict):
        coerced_plan = _coerce_pipeline_plan(plan)
        if coerced_plan:
            plan = {**plan, **coerced_plan}

    per_site = _normalize_per_site_stages_value(plan.get("per_site_stages")) if isinstance(plan, dict) else None
    sites = [s for s in (plan.get("sites") or []) if s in VALID_SOURCES]
    if per_site:
        for s in list(per_site.keys()):
            if s in VALID_SOURCES and s not in sites:
                sites.append(s)
    if not sites:
        err = "# Pipeline run\n\n**Plan error:** no valid `sites` (expected one or more of `OpenOrganelle`, `BossDB`, `LocalHPC`).\n"
        return err, {"ok": False, "sites": [], "stages": [], "steps": []}

    stages_all = _canonicalize_plan_stages_list(list(plan.get("stages") or []))
    if "download" in stages_all and "database" not in stages_all:
        dl_idx = stages_all.index("download")
        stages_all.insert(dl_idx, "database")
    if not stages_all and not per_site:
        err = "# Pipeline run\n\n**Plan error:** provide `plan.stages` and/or `plan.per_site_stages`.\n"
        return err, {"ok": False, "sites": sites, "stages": [], "steps": []}

    def _stages_for_site(site: str) -> list[str]:
        if per_site and site in per_site:
            ss = _canonicalize_plan_stages_list(list(per_site[site]))
        else:
            ss = list(stages_all)
        if site == "LocalHPC" and "scrape" in ss:
            # Local HPC has no website scraping stage; map "discover/build local data view"
            # work to its catalogue/database stage.
            remap: list[str] = []
            for s in ss:
                if s == "scrape":
                    if "database" not in remap:
                        remap.append("database")
                    continue
                if s not in remap:
                    remap.append(s)
            ss = remap
        if "download" in ss and "database" not in ss:
            ss = list(ss)
            ss.insert(ss.index("download"), "database")
        return ss

    union_stages = list(stages_all)
    if per_site:
        seen_u: set[str] = set(union_stages)
        for lst in per_site.values():
            for s in lst:
                if s not in seen_u and s in ("scrape", "database", "download", "training"):
                    seen_u.add(s)
                    union_stages.append(s)

    n_crops = 1
    try:
        n_crops = max(1, min(64, int(plan.get("n_crops", 1))))
    except (TypeError, ValueError):
        n_crops = 1

    sections: list[str] = []
    meta: dict[str, Any] = {
        "ok": True,
        "sites": sites,
        "stages": stages_all,
        "per_site_stages": per_site,
        "steps": [],
    }
    if "download" in union_stages:
        meta["n_crops"] = n_crops
    training_requested = "training" in union_stages
    meta["training_requested"] = training_requested
    plan_stage_to_session: dict[str, PipelineStep] = {
        "scrape": PipelineStep.SCRAPE,
        "database": PipelineStep.DATABASE,
        "download": PipelineStep.DOWNLOAD_SCRIPT,
        "training": PipelineStep.SSL,
    }

    def _mark_pipeline_activity(site: str, stage: str) -> None:
        """Expose coarse live progress for `/api/pipeline` polling while a stage runs (per-site)."""
        session.last_site_stem = site
        st = plan_stage_to_session.get(stage)
        if st is not None:
            session.current_step = st
        session.log("pipeline_execute", {"site": site, "stage": stage})

    def _session_reflect_stage(stage: str, ok: bool) -> None:
        if ok and stage in plan_stage_to_session:
            session.current_step = plan_stage_to_session[stage]

    def _is_stop_requested() -> bool:
        if stop_requested is None:
            return False
        try:
            return bool(stop_requested())
        except Exception:
            return False

    def _scrape_error_message(payload: dict[str, Any]) -> str:
        msg = str(payload.get("message") or "").strip()
        if msg:
            return msg
        fetch = payload.get("fetch")
        if isinstance(fetch, dict):
            ferr = str(fetch.get("error") or "").strip()
            if ferr:
                return ferr
        bridge = payload.get("mito_foundation_bridge")
        if isinstance(bridge, dict):
            berr = str(bridge.get("message") or bridge.get("error") or "").strip()
            if berr:
                return berr
        return "Scrape failed with no explicit message."

    def _failed_datasets_from_logs(payload: dict[str, Any]) -> list[str]:
        text = f"{payload.get('stdout') or ''}\n{payload.get('stderr') or ''}"
        m = re.search(r"Failed datasets:\s*(\[[^\]]*\])", text, re.IGNORECASE)
        if not m:
            return []
        raw = m.group(1).strip()
        try:
            arr = json.loads(raw.replace("'", '"'))
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
        return []

    def _download_message_compact(payload: dict[str, Any]) -> str:
        msg = str(payload.get("message") or "").strip()
        # Avoid duplicating huge subprocess dumps inside message field; those are
        # already rendered via stdout/stderr tail blocks below.
        if "\n" in msg:
            msg = msg.splitlines()[0].strip()
        failed = _failed_datasets_from_logs(payload)
        if failed:
            sample = ", ".join(failed[:4])
            extra = f" (+{len(failed)-4} more)" if len(failed) > 4 else ""
            return f"{msg or 'Downloader completed with dataset-level failures.'} Failed datasets ({len(failed)}): {sample}{extra}."
        return msg or "Downloader finished."

    site_stage_map: dict[str, list[str]] = {s: _stages_for_site(s) for s in sites}
    site_scrape_failed: dict[str, bool] = {s: False for s in sites}
    stage_sequence = [s for s in ("scrape", "database", "download", "training") if s in union_stages]

    download_sites_attempted = 0
    download_sites_skipped_no_match = 0
    for stage in stage_sequence:
        if stage == "training":
            # Global training stage runs once after all sites have completed prior stages.
            if _is_stop_requested():
                meta["ok"] = False
                meta["cancelled"] = True
                meta["steps"].append({"site": "all", "stage": "training", "ok": False, "cancelled": True})
                sections.append(
                    "## Global — nnUNet training\n"
                    "- **ok:** False\n"
                    "- **message:** Cancelled before global training submission.\n"
                )
                break
            if not meta.get("ok", True):
                meta["steps"].append({"site": "all", "stage": "training", "ok": False, "skipped": True})
                sections.append(
                    "## Global — nnUNet training\n"
                    "- **ok:** False\n"
                    "- **message:** Skipped because one or more prior site stages failed.\n"
                )
                continue
            sections.append("> **Global** · `training` …\n")
            _mark_pipeline_activity("all", "training")
            r = studio_run_training(StudioSessionBody(session_id=session_id))
            ok_tr = bool(r.get("ok"))
            meta["steps"].append({"site": "all", "stage": "training", "ok": ok_tr})
            _session_reflect_stage("training", ok_tr)
            sections.append(
                "## Global — nnUNet training\n"
                f"- **ok:** {ok_tr}\n"
                f"- **message:** {r.get('message', '')}\n"
            )
            if not ok_tr:
                meta["ok"] = False
            continue

        for site in sites:
            if stage not in site_stage_map.get(site, []):
                continue
            if _is_stop_requested():
                meta["ok"] = False
                meta["cancelled"] = True
                meta["steps"].append({"site": site, "stage": stage, "ok": False, "cancelled": True})
                sections.append(
                    f"## {site} — {stage}\n"
                    "- **ok:** False\n"
                    "- **message:** Cancelled before stage started.\n"
                )
                break

            if site == "LocalHPC" and stage == "database":
                sections.append(f"> **{site}** · `database` …\n")
                _mark_pipeline_activity("local-hpc", "database")
                try:
                    cat = studio_mitole_catalogue(regenerate=True)
                    ok_cat = bool(cat.get("ok", True))
                    msg = str(cat.get("message") or f"Local HPC catalogue ready with {len(cat.get('rows') or [])} rows.")
                except Exception as exc:
                    ok_cat = False
                    msg = f"Local HPC catalogue build failed: {exc}"
                meta["steps"].append({"site": site, "stage": "database", "ok": ok_cat})
                _session_reflect_stage("database", ok_cat)
                sections.append(
                    f"## {site} — database build\n"
                    f"- **ok:** {ok_cat}\n"
                    f"- **message:** {msg}\n"
                )
                if not ok_cat:
                    meta["ok"] = False
                continue

            slug = SITE_SLUG.get(site)
            if not slug and site != "LocalHPC":
                meta["steps"].append({"site": site, "stage": stage, "ok": False, "skipped": True})
                sections.append(
                    f"## {site} — {stage}\n"
                    "- **ok:** False\n"
                    "- **message:** Skipped (unknown site).\n"
                )
                meta["ok"] = False
                continue

            if stage in {"database", "download"} and site_scrape_failed.get(site):
                meta["steps"].append({"site": site, "stage": stage, "ok": False, "skipped": True})
                sections.append(
                    f"## {site} — {stage}\n"
                    "- **ok:** False\n"
                    "- **message:** Skipped because scrape failed for this site.\n"
                )
                meta["ok"] = False
                continue

            if stage == "scrape":
                sections.append(f"> **{site}** · `scrape` …\n")
                _mark_pipeline_activity(site, "scrape")
                r = studio_scrape_website(StudioWebsiteScrapeBody(session_id=session_id, slug=slug))
                ok_sc = bool(r.get("ok"))
                meta["steps"].append({"site": site, "stage": "scrape", "ok": ok_sc})
                _session_reflect_stage("scrape", ok_sc)
                scrape_msg = str(r.get("message") or "").strip() if ok_sc else _scrape_error_message(r)
                sections.append(
                    f"## {site} — scrape (`{slug}`)\n"
                    f"- **ok:** {ok_sc}\n"
                    f"- **message:** {scrape_msg}\n"
                )
                if not ok_sc:
                    meta["ok"] = False
                    site_scrape_failed[site] = True
                continue

            if stage == "database":
                sections.append(f"> **{site}** · `database` …\n")
                _mark_pipeline_activity(site, "database")
                probe = ""
                rel = SITE_PROBE_REL.get(site, "")
                if rel:
                    p = project_root / rel
                    if p.is_file():
                        probe = rel
                r = studio_run_database_build(StudioDatabaseBuildBody(session_id=session_id, probe=probe))
                ok_db = bool(r.get("ok"))
                meta["steps"].append({"site": site, "stage": "database", "ok": ok_db})
                _session_reflect_stage("database", ok_db)
                sections.append(
                    f"## {site} — database build\n"
                    f"- **ok:** {ok_db}\n"
                    f"- **probe:** `{probe or '(default newest)'}`\n"
                    f"- **message:** {r.get('message', '')}\n"
                )
                if not ok_db:
                    meta["ok"] = False
                continue

            if stage == "download":
                if site == "LocalHPC":
                    sections.append(f"> **{site}** · `download` …\n")
                    _mark_pipeline_activity("local-hpc", "download")
                    download_sites_attempted += 1
                    dl_cfg = plan.get("download") if isinstance(plan, dict) else {}
                    if not isinstance(dl_cfg, dict):
                        dl_cfg = {}
                    cat = studio_mitole_catalogue(regenerate=False)
                    rows = cat.get("rows") if isinstance(cat, dict) else []
                    if not isinstance(rows, list):
                        rows = []
                    include_ds = {x.lower() for x in _normalize_string_list(dl_cfg.get("datasets_include"))}
                    exclude_ds = {x.lower() for x in _normalize_string_list(dl_cfg.get("datasets_exclude"))}
                    include_st = {x.lower() for x in _normalize_string_list(dl_cfg.get("sample_types_include"))}
                    exclude_st = {x.lower() for x in _normalize_string_list(dl_cfg.get("sample_types_exclude"))}
                    global_target = _normalize_download_target(dl_cfg) or {"target": "training", "total_crops": n_crops}
                    fallback_total = _clamp_int(global_target.get("total_crops"), 1, 16, n_crops)
                    pdt = dl_cfg.get("per_dataset_targets")
                    per_ds = pdt if isinstance(pdt, dict) else {}
                    dataset_pairs: list[dict[str, str]] = []
                    dataset_splits: dict[str, dict[str, int]] = {}
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        ds = str(row.get("dataset") or "").strip()
                        st = str(row.get("sample_type") or "unknown").strip().lower() or "unknown"
                        if not ds:
                            continue
                        if include_ds and not _dataset_name_matches(include_ds, ds):
                            continue
                        if exclude_ds and _dataset_name_matches(exclude_ds, ds):
                            continue
                        if include_st and not _sample_type_matches(include_st, st):
                            continue
                        if exclude_st and _sample_type_matches(exclude_st, st):
                            continue
                        img = str(row.get("image_path") or "").strip()
                        seg = str(row.get("label_path") or "").strip()
                        if not img or not seg:
                            continue
                        if ds not in dataset_splits:
                            cfg = dict(global_target)
                            if ds in per_ds and isinstance(per_ds[ds], dict):
                                cfg.update(per_ds[ds])
                            tr, inf = _dataset_counts_from_target(cfg, fallback_total=fallback_total)
                            if tr + inf <= 0:
                                continue
                            dataset_splits[ds] = {"training": tr, "inference": inf}
                        dataset_pairs.append(
                            {
                                "dataset": ds,
                                "source": str(row.get("source") or "local-hpc"),
                                "image_path": img,
                                "label_path": seg,
                            }
                        )
                    if not dataset_pairs or not dataset_splits:
                        download_sites_skipped_no_match += 1
                        meta["steps"].append({"site": site, "stage": "download", "ok": True, "skipped": True})
                        sections.append(
                            f"## {site} — download\n"
                            "- **ok:** True\n"
                            "- **noop:** True\n"
                            "- **message:** Download skipped: no LocalHPC datasets matched planned filters or available pairs.\n"
                        )
                        continue
                    r = studio_run_mitole_downloader(
                        StudioMitoLeDownloaderBody(
                            session_id=session_id,
                            dataset_splits=dataset_splits,
                            dataset_pairs=dataset_pairs,
                        )
                    )
                    ok_dl = bool(r.get("ok"))
                    meta["steps"].append({"site": site, "stage": "download", "ok": ok_dl})
                    _session_reflect_stage("download", ok_dl)
                    sections.append(
                        f"## {site} — download\n"
                        f"- **ok:** {ok_dl}\n"
                        f"- **message:** {str(r.get('message') or '').strip()}\n"
                        f"- **downloader log (tail):**\n```\n{str(r.get('downloader_log') or '')[-4000:]}\n```\n"
                    )
                    if not ok_dl:
                        meta["ok"] = False
                    continue
                sections.append(f"> **{site}** · `download` …\n")
                _mark_pipeline_activity(site, "download")
                download_sites_attempted += 1
                dl = DOWNLOADER_SITE[site]
                dataset_splits: dict[str, dict[str, int]] = {}
                dl_cfg = plan.get("download") if isinstance(plan, dict) else None
                if isinstance(dl_cfg, dict):
                    try:
                        prev = studio_downloader_preview(site=dl, data_scope="labeled")
                    except Exception:
                        prev = {"datasets": [], "dataset_rows": []}
                    llm_splits, llm_meta = _llm_expand_download_selection(
                        download_cfg=dl_cfg,
                        preview=prev,
                        n_crops_default=n_crops,
                    )
                    if llm_splits:
                        dataset_splits = _enforce_dataset_scope_on_splits(dl_cfg, llm_splits)
                        if dataset_splits:
                            max_training = max(v["training"] for v in dataset_splits.values())
                            max_inference = max(v["inference"] for v in dataset_splits.values())
                            split_meta = {
                                "selected_count": len(dataset_splits),
                                "candidate_count": len(prev.get("datasets") or []),
                                "n_crops": max((v["training"] + v["inference"]) for v in dataset_splits.values()),
                                "n_crops_training": max_training,
                                "n_crops_inference": max_inference,
                                "sample_type_resolution": (llm_meta or {}).get("sample_type_resolution", "unspecified"),
                                "selection_source": "llm",
                            }
                        else:
                            split_meta = {
                                "selected_count": 0,
                                "candidate_count": len(prev.get("datasets") or []),
                                "n_crops": 0,
                                "n_crops_training": 0,
                                "n_crops_inference": 0,
                                "sample_type_resolution": (llm_meta or {}).get("sample_type_resolution", "unspecified"),
                                "selection_source": "llm",
                            }
                    else:
                        dataset_splits, split_meta = _build_dataset_splits_from_plan(prev, dl_cfg)
                        dataset_splits = _enforce_dataset_scope_on_splits(dl_cfg, dataset_splits)
                        dataset_splits = _enforce_sample_type_scope_on_splits(prev, dl_cfg, dataset_splits)
                        if dataset_splits:
                            split_meta["selected_count"] = len(dataset_splits)
                            split_meta["n_crops"] = max((v["training"] + v["inference"]) for v in dataset_splits.values())
                            split_meta["n_crops_training"] = max(v["training"] for v in dataset_splits.values())
                            split_meta["n_crops_inference"] = max(v["inference"] for v in dataset_splits.values())
                        else:
                            split_meta["selected_count"] = 0
                            split_meta["n_crops"] = 0
                            split_meta["n_crops_training"] = 0
                            split_meta["n_crops_inference"] = 0
                        split_meta["selection_source"] = "deterministic_fallback"
                    if split_meta.get("n_crops"):
                        n_crops = _clamp_int(split_meta["n_crops"], 1, 16, n_crops)
                    meta["download_selection"] = split_meta
                    try:
                        meta["n_crops_training"] = int(split_meta.get("n_crops_training", 0))
                        meta["n_crops_inference"] = int(split_meta.get("n_crops_inference", 0))
                    except (TypeError, ValueError):
                        pass
                    if not dataset_splits:
                        download_sites_skipped_no_match += 1
                        meta["steps"].append({"site": site, "stage": "download", "ok": True, "skipped": True})
                        sections.append(
                            f"## {site} — download (`n_crops={n_crops}`)\n"
                            "- **ok:** True\n"
                            "- **noop:** True\n"
                            "- **message:** Download skipped: no datasets matched the planned sample-type/dataset filters.\n"
                        )
                        continue
                r = studio_run_downloader(
                    StudioDownloaderBody(
                        session_id=session_id,
                        site=dl,
                        n_crops=n_crops,
                        dataset_splits=dataset_splits,
                        execute=True,
                    )
                )
                ok_dl = bool(r.get("ok"))
                meta["steps"].append({"site": site, "stage": "download", "ok": ok_dl})
                _session_reflect_stage("download", ok_dl)
                tail_out = (r.get("stdout") or "")[-4000:]
                tail_err = (r.get("stderr") or "")[-4000:]
                compact_msg = _download_message_compact(r)
                sections.append(
                    f"## {site} — download (`n_crops={n_crops}`)\n"
                    f"- **ok:** {ok_dl}\n"
                    f"- **noop:** {bool(r.get('noop'))}\n"
                    f"- **message:** {compact_msg}\n"
                    f"- **stdout (tail):**\n```\n{tail_out}\n```\n"
                    f"- **stderr (tail):**\n```\n{tail_err}\n```\n"
                )
                if not ok_dl:
                    meta["ok"] = False

            if _is_stop_requested():
                meta["ok"] = False
                meta["cancelled"] = True
                break

    # Emit a compact, planner-consistent completion report for chat UX.
    steps_all = [s for s in meta.get("steps", []) if isinstance(s, dict)]
    stage_order = ["scrape", "database", "download", "training"]
    by_stage: dict[str, dict[str, int]] = {
        st: {"ok": 0, "failed": 0, "skipped": 0, "cancelled": 0}
        for st in stage_order
    }
    for row in steps_all:
        st = str(row.get("stage") or "").strip().lower()
        if st not in by_stage:
            continue
        if row.get("cancelled"):
            by_stage[st]["cancelled"] += 1
        elif row.get("skipped"):
            by_stage[st]["skipped"] += 1
        elif bool(row.get("ok")):
            by_stage[st]["ok"] += 1
        else:
            by_stage[st]["failed"] += 1

    report_lines: list[str] = ["## Pipeline Run Report", f"- Sites: {', '.join(sites)}"]
    stage_labels = {
        "scrape": "Scrape websites",
        "database": "Build database",
        "download": "Download data",
        "training": "Start training",
    }
    source_workflows: list[str] = []
    for site in sites:
        site_stages = [s for s in site_stage_map.get(site, []) if s in stage_labels]
        if site_stages:
            source_workflows.append(f"{site}: " + " -> ".join(stage_labels[s] for s in site_stages))
    if source_workflows:
        report_lines.append("- Source workflows:")
        report_lines.extend([f"  - {row}" for row in source_workflows])
    stage_plan_txt = " -> ".join(stage_labels[s] for s in union_stages if s in stage_labels)
    if stage_plan_txt:
        report_lines.append(f"- Pipeline: {stage_plan_txt}")
    if "download" in union_stages:
        tr_n = int(meta.get("n_crops_training", 0) or 0)
        inf_n = int(meta.get("n_crops_inference", 0) or 0)
        if tr_n or inf_n:
            report_lines.append(f"- Crops per selected dataset: training={tr_n}, inference={inf_n}")
        else:
            report_lines.append(f"- Crops per selected dataset: {int(n_crops)}")
    if "scrape" in union_stages:
        c = by_stage["scrape"]
        if (c["ok"] + c["failed"] + c["skipped"] + c["cancelled"]) > 0:
            report_lines.append(
                f"- Scrape websites: success={c['ok']}, failed={c['failed']}, skipped={c['skipped']}, cancelled={c['cancelled']}"
            )
    if "database" in union_stages:
        c = by_stage["database"]
        if (c["ok"] + c["failed"] + c["skipped"] + c["cancelled"]) > 0:
            report_lines.append(
                f"- Build database: success={c['ok']}, failed={c['failed']}, skipped={c['skipped']}, cancelled={c['cancelled']}"
            )
    if "download" in union_stages:
        c = by_stage["download"]
        if (c["ok"] + c["failed"] + c["skipped"] + c["cancelled"]) > 0:
            report_lines.append(
                f"- Download data: success={c['ok']}, failed={c['failed']}, skipped={c['skipped']}, cancelled={c['cancelled']}"
            )
        if (
            download_sites_attempted > 0
            and download_sites_skipped_no_match == download_sites_attempted
            and c["ok"] == 0
            and c["failed"] == 0
            and c["cancelled"] == 0
        ):
            report_lines.append(
                "- Download filters matched no datasets across all requested sources; all download stages were skipped."
            )
    if "training" in union_stages:
        c = by_stage["training"]
        if (c["ok"] + c["failed"] + c["skipped"] + c["cancelled"]) > 0:
            report_lines.append(
                f"- Start training: success={c['ok']}, failed={c['failed']}, skipped={c['skipped']}, cancelled={c['cancelled']}"
            )
            report_lines.append("- Note: pipeline is treated as complete once training is submitted.")
    report_lines.append(f"- Final status: {'SUCCESS' if bool(meta.get('ok')) else 'FAILED'}")
    sections.append("\n".join(report_lines) + "\n")

    stage_hdr = ", ".join(union_stages) if union_stages else "(per-site)"
    header_core = f"# Pipeline run\nSites: **{', '.join(sites)}** · stages: **{stage_hdr}**"
    if "download" in union_stages:
        header = header_core + f" · `n_crops={n_crops}`\n\n"
    else:
        header = header_core + "\n\n"
    session.current_step = PipelineStep.IDLE
    return header + "\n".join(sections), meta


# --- Saved plans (project-local) ---

def _plans_path(root: Path) -> Path:
    return root / "agent" / "chat_web" / "data" / ".mito2_approved_plans.json"


def _legacy_plans_path(root: Path) -> Path:
    return root / ".mito2_approved_plans.json"


def load_saved_plans(root: Path) -> list[dict[str, Any]]:
    _ = root
    # Approved plans are tracked in chat thread metadata (`.mito2_chats.json`) only.
    # Keep this API for backward compatibility but avoid generating a separate plans file.
    return []


def save_plan(root: Path, plan: dict[str, Any], *, title: str = "") -> dict[str, Any]:
    _ = root
    pid = f"plan_{uuid.uuid4().hex[:12]}"
    row = {
        "id": pid,
        "created_at": now_us_eastern_iso(),
        "title": (title or "").strip() or pid,
        "plan": plan,
    }
    return row


def delete_plan(root: Path, plan_id: str) -> bool:
    _ = root
    _ = plan_id
    return True
