"""One chat turn: optional tool proposals (approvals) + LLM reply."""

from __future__ import annotations

import difflib
import json
import os
import re
import shlex
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.orchestration.approvals import ApprovalQueue, PendingAction
from agent.orchestration.registry.schema import DEFAULT_REGISTRY_PATH
from agent.orchestration.session_pipeline import STEP_LABELS, PipelineSession, PipelineStep
from agent.orchestration.skill_api import (
    ensure_skill_trees,
    merged_chat_skills_block,
    merged_orchestration_skills_block,
)
from agent.orchestration.prompts_store import PromptBundle

from .llm import LLMUnavailableError, complete, setup_hint

from .pipeline_chat import (
    ask_mode_plan_switch_reply,
    _canonical_pipeline_stage_token,
    _canonicalize_plan_stages_list,
    _coerce_pipeline_plan,
    execute_pipeline_plan,
    llm_complete_pipeline_plan,
    llm_route_chat_intent,
    message_touches_pipeline,
    planning_trace_from_intent_decision,
    plan_mode_execution_refusal,
)


_REPO_FILE_SCAN_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "have",
    "how",
    "many",
    "what",
    "inside",
    "into",
    "your",
    "you",
    "are",
    "can",
    "could",
    "would",
    "should",
    "when",
    "where",
    "which",
    "about",
    "they",
    "them",
}


def _query_terms(text: str) -> list[str]:
    norm = (text or "").lower().replace("-", " ").replace("_", " ")
    parts = re.findall(r"[a-zA-Z0-9]+", norm)
    out: list[str] = []
    seen: set[str] = set()
    for t in parts:
        if len(t) < 3 or t in _REPO_FILE_SCAN_STOPWORDS:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:12]


def _ignored_for_repo_file_scan(path: Path) -> bool:
    bad_parts = {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        "dist",
        "build",
    }
    return any(part in bad_parts for part in path.parts)


def _values_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (str, int, float, bool)):
        return str(x)
    if isinstance(x, list):
        return " ".join(_values_text(v) for v in x)
    if isinstance(x, dict):
        return " ".join(_values_text(v) for v in x.values())
    return str(x)


def _term_in_blob(term: str, blob: str) -> bool:
    t = (term or "").strip().lower()
    if not t:
        return False
    b = (blob or "").lower()
    if t in b:
        return True
    words = set(re.findall(r"[a-z0-9]+", b))
    if not words:
        return False
    if len(t) >= 6:
        p4 = t[:4]
        if any(w.startswith(p4) for w in words if len(w) >= 4):
            return True
    # Generic typo tolerance (e.g. "mitocondria" ~= "mitochondria").
    return bool(difflib.get_close_matches(t, list(words), n=1, cutoff=0.8))


def _md_numeric_facts(text: str) -> list[str]:
    facts: list[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if re.search(r"\b\d+\b", s):
            facts.append(s[:180])
        if len(facts) >= 3:
            break
    return facts


def _data_inventory_evidence(project_root: Path) -> str:
    """
    Lightweight inventory of `data/` for grounding chat answers.
    Does not read large binary contents; only filesystem metadata.
    """
    data_dir = project_root / "data"
    if not data_dir.is_dir():
        return ""

    # Count files by extension, a few recent files, and top-level structure.
    ext_counts: dict[str, int] = {}
    top_dirs: list[str] = []
    try:
        for child in sorted(data_dir.iterdir(), key=lambda p: p.name.lower()):
            if child.is_dir():
                top_dirs.append(child.name)
    except OSError:
        top_dirs = []

    recent: list[tuple[float, str]] = []
    n_files = 0
    n_dirs = 0
    total_bytes = 0
    max_scan = 5000

    for root, dirs, files in os.walk(str(data_dir)):
        n_dirs += len(dirs)
        for fn in files:
            n_files += 1
            if n_files > max_scan:
                break
            p = Path(root) / fn
            try:
                st = p.stat()
            except OSError:
                continue
            total_bytes += int(st.st_size or 0)
            ext = p.suffix.lower().lstrip(".") or "(none)"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
            rel = str(p.relative_to(project_root)).replace("\\", "/")
            recent.append((st.st_mtime, rel))
        if n_files > max_scan:
            break

    recent.sort(key=lambda t: t[0], reverse=True)
    recent_paths = [p for _, p in recent[:10]]
    top_exts = sorted(ext_counts.items(), key=lambda kv: kv[1], reverse=True)[:12]
    exts_text = ", ".join(f"{k}:{v}" for k, v in top_exts)
    top_dirs_text = ", ".join(top_dirs[:20]) + (" ..." if len(top_dirs) > 20 else "")

    return (
        "Data inventory (`data/`): "
        f"top_level_dirs=[{top_dirs_text}] | "
        f"files_scanned={min(n_files, max_scan)} | dirs_seen={n_dirs} | "
        f"ext_counts_top={{{exts_text}}} | "
        f"recent_files={recent_paths}"
        + (f" | note=scan_truncated_at_{max_scan}_files" if n_files > max_scan else "")
    )


def _artifact_evidence(project_root: Path, user_message: str) -> str:
    """
    Generic local retrieval over repository artifacts.
    Parses many files across the repo and returns concise evidence lines.
    """
    all_files: list[Path] = []
    for ext in ("*.json", "*.md", "*.txt", "*.py", "*.ts", "*.tsx", "*.yaml", "*.yml"):
        for p in project_root.rglob(ext):
            if _ignored_for_repo_file_scan(p):
                continue
            try:
                if not p.is_file() or p.stat().st_size > 2_000_000:
                    continue
            except OSError:
                continue
            all_files.append(p)
    if not all_files:
        return ""
    all_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    # Parse widely, but keep prompt size bounded.
    scanned_files = all_files[:900]

    terms = _query_terms(user_message)

    rows: list[str] = []
    budget = 0
    for p in scanned_files:
        rel = str(p.relative_to(project_root)).replace("\\", "/")
        try:
            raw = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not raw.strip():
            continue

        # Always include a compact parse summary for each file.
        summary = ""
        if p.suffix == ".json":
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict):
                keys = list(obj.keys())[:8]
                ds = obj.get("datasets")
                ds_count = len(ds) if isinstance(ds, (dict, list)) else 0
                summary = f"top_keys={keys} datasets_count={ds_count}"
                # Generic query-aware row hit counts (no domain-specific rules).
                if terms and ds_count:
                    rows_iter: list[dict[str, Any]] = []
                    if isinstance(ds, dict):
                        rows_iter = [v for v in ds.values() if isinstance(v, dict)]
                    elif isinstance(ds, list):
                        rows_iter = [v for v in ds if isinstance(v, dict)]
                    if rows_iter:
                        per_term: dict[str, int] = {t: 0 for t in terms}
                        row_blobs = [_values_text(row).lower() for row in rows_iter]
                        for blob in row_blobs:
                            for t in terms:
                                if _term_in_blob(t, blob):
                                    per_term[t] += 1
                        any_hit = sum(
                            1
                            for blob in row_blobs
                            if any(_term_in_blob(t, blob) for t in terms)
                        )
                        top_terms = sorted(per_term.items(), key=lambda kv: kv[1], reverse=True)[:6]
                        summary += (
                            f" query_row_hits_any={any_hit}"
                            f" query_row_hits_by_term={{{', '.join(f'{k}:{v}' for k, v in top_terms)}}}"
                        )
            elif isinstance(obj, list):
                summary = f"json_list_len={len(obj)}"
            else:
                summary = "json_unstructured"
        else:
            summary = f"markdown_lines={len(raw.splitlines())}"
            facts = _md_numeric_facts(raw)
            if facts:
                summary += f" numeric_facts={facts}"

        lines = raw.splitlines()
        file_hits: list[str] = []
        if terms:
            for idx, ln in enumerate(lines):
                low = ln.lower()
                if any(t in low for t in terms):
                    start = max(0, idx - 1)
                    end = min(len(lines), idx + 2)
                    snippet = " || ".join(s.strip()[:140] for s in lines[start:end] if s.strip())
                    file_hits.append(f"{idx + 1}:{snippet}")
                if len(file_hits) >= 3:
                    break
        line = f"- `{rel}`: {summary}"
        if file_hits:
            line += " | " + " | ".join(file_hits)
        rows.append(line)
        budget += len(line)
        if budget > 20000:
            break

    if not rows:
        return ""
    parsed = ", ".join(p.name for p in scanned_files[:30])
    if len(scanned_files) > 30:
        parsed += f", ... ({len(scanned_files) - 30} more)"
    terms_text = ", ".join(terms) if terms else "(none)"
    return (
        f"Parsed files across repo: scanned={len(scanned_files)} total_candidates={len(all_files)}\n"
        f"Files (recent subset): {parsed}\n"
        f"Query terms: {terms_text}\n"
        + "\n".join(rows)
    )


def _dataset_count_and_sample(data: dict[str, Any]) -> tuple[int, list[str]]:
    """Return true dataset count plus a small sample of ids/keys."""
    ds = data.get("datasets")
    if isinstance(ds, dict):
        keys = list(ds.keys())
        return len(keys), keys[:12]
    if isinstance(ds, list):
        sample: list[str] = []
        for row in ds[:12]:
            if isinstance(row, dict):
                rid = row.get("id") or row.get("name") or row.get("slug")
                sample.append(str(rid) if rid else "<dict>")
            else:
                sample.append(str(row))
        return len(ds), sample
    return 0, []


def _latest_probe_summary(project_root: Path) -> str:
    out_dir = project_root / "1web_scraper_01" / "outputs"
    if not out_dir.is_dir():
        return "No scrape outputs yet."
    probes = sorted(out_dir.glob("*.probe.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not probes:
        return "No *.probe.json files."
    p = probes[0]
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return f"Latest probe `{p.name}` is not valid JSON."
    fetch = data.get("fetch") or {}
    title = fetch.get("title") or ""
    err = fetch.get("error")
    ok = fetch.get("ok")
    total, sample = _dataset_count_and_sample(data)
    sample = sample[:5]

    return (
        f"Latest probe file: `{p.relative_to(project_root)}` | ok={ok} | title={title!r} | error={err!r} | "
        f"total_datasets={total} | sample_dataset_ids(sample_only_not_total)={sample}"
    )


def _is_dataset_total_question(message: str) -> bool:
    m = (message or "").lower()
    return bool(
        (("how many" in m or "total" in m) and ("dataset" in m or "datasets" in m))
        or re.search(r"\bdataset\s+count\b", m)
    )


def _normalize_site_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _sites_from_message(message: str) -> list[str]:
    m = (message or "").lower()
    hits: list[tuple[int, str]] = []
    for token, site in (("openorganelle", "OpenOrganelle"), ("bossdb", "BossDB")):
        pos = m.find(token)
        if pos >= 0:
            hits.append((pos, site))
    hits.sort(key=lambda x: x[0])
    return [site for _, site in hits]


@dataclass
class QuestionPlanItem:
    """Planner output for one deterministic question sub-task."""

    kind: str
    site: str


@dataclass
class QuestionPlan:
    """Planner output for a user question."""

    is_question: bool
    items: list[QuestionPlanItem]


@dataclass
class FactAnswer:
    """Executor output for one deterministic question sub-task."""

    kind: str
    site: str
    value: Optional[int]
    evidence: list[str]
    confidence: str
    scope: str = ""
    source_family: str = ""
    exhaustive: bool = False
    missing_reason: Optional[str] = None


def _registry_path(project_root: Path) -> Path:
    return project_root / "data" / "registry.sqlite"


def _open_registry_ro(project_root: Path) -> sqlite3.Connection | None:
    p = _registry_path(project_root)
    if not p.is_file():
        return None
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    return conn


def _provider_names_for_site(site: str) -> list[str]:
    s = _normalize_site_token(site)
    if s == "openorganelle":
        return ["OpenOrganelle"]
    if s == "bossdb":
        return ["BossDB"]
    return [site]


def _is_mito_label_question(message: str) -> bool:
    m = (message or "").lower()
    return bool(
        ("mito" in m or "mitochond" in m)
        and ("label" in m or "seg" in m or "annotation" in m)
        and ("how many" in m or "count" in m or "total" in m)
    )


def _is_downloaded_count_question(message: str) -> bool:
    m = (message or "").lower()
    return bool(
        ("downloaded" in m or "download so far" in m or "downloaded so far" in m)
        and ("how many" in m or "count" in m or "total" in m)
    )


def _is_source_breakdown_question(message: str) -> bool:
    m = (message or "").lower()
    return "from what source" in m or "by source" in m or "by provider" in m


def _looks_like_question(message: str) -> bool:
    m = (message or "").strip().lower()
    if not m:
        return False
    if "?" in m:
        return True
    return bool(re.search(r"^(how|what|which|where|when|who|is|are|do|does|can)\b", m))


def _plan_question(message: str) -> QuestionPlan:
    """
    Phase 1: Planner
    Identify whether the message is a question and decompose into deterministic sub-questions.
    """
    is_q = _looks_like_question(message)
    sites = _sites_from_message(message)
    items: list[QuestionPlanItem] = []

    # Main deterministic pattern currently supported:
    # "How many/total datasets ... <site>?"
    is_dataset_total = _is_dataset_total_question(message)
    is_site_total = bool(
        sites and re.search(r"\b(how many|total)\b", (message or "").lower())
    )
    if (is_dataset_total or is_site_total) and sites:
        for site in sites:
            items.append(QuestionPlanItem(kind="dataset_total", site=site))
    if _is_mito_label_question(message):
        target_sites = sites or ["OpenOrganelle", "BossDB"]
        for site in target_sites:
            items.append(QuestionPlanItem(kind="mito_labeled_count", site=site))
    if _is_downloaded_count_question(message):
        # Global downloaded count and provider/source breakdown.
        items.append(QuestionPlanItem(kind="downloaded_total", site="all"))
    if _is_source_breakdown_question(message):
        items.append(QuestionPlanItem(kind="downloaded_by_provider", site="all"))

    return QuestionPlan(is_question=is_q, items=items)


def _latest_probe_for_site(project_root: Path, site: str) -> Optional[Path]:
    out_dir = project_root / "1web_scraper_01" / "outputs"
    if not out_dir.is_dir():
        return None
    token = _normalize_site_token(site)
    probes = []
    for p in out_dir.glob("*.probe.json"):
        try:
            stem_token = _normalize_site_token(p.stem.replace(".probe", ""))
            if token in stem_token or stem_token in token:
                probes.append(p)
        except OSError:
            continue
    if not probes:
        return None
    probes.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return probes[0]


def _extract_total_datasets_from_script(project_root: Path, site: str) -> Optional[tuple[int, str]]:
    s = _normalize_site_token(site)
    out_dir = project_root / "3data_downloader" / "outputs"
    if not out_dir.is_dir():
        return None
    for p in sorted(out_dir.glob("download_*.py"), key=lambda x: x.stat().st_mtime, reverse=True):
        name_token = _normalize_site_token(p.stem)
        if s not in name_token and name_token not in s:
            continue
        try:
            raw = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        m = re.search(r"^\s*TOTAL_DATASETS\s*=\s*(\d+)\s*$", raw, flags=re.M)
        if m:
            return int(m.group(1)), str(p.relative_to(project_root)).replace("\\", "/")
    return None


def _route_sources_for_question(item: QuestionPlanItem) -> list[str]:
    """
    Phase 2: Evidence Router
    Route each plan item to ordered evidence sources.
    """
    if item.kind == "dataset_total":
        return ["probe_json", "generated_downloader_script"]
    if item.kind == "mito_labeled_count":
        # Prefer fresh stage-1 probe evidence first when available.
        return ["probe_json", "registry_sqlite", "filesystem", "generated_downloader_script"]
    if item.kind in {"downloaded_total", "downloaded_by_provider"}:
        return ["registry_sqlite", "filesystem", "generated_downloader_script"]
    return []


def _execute_dataset_total(project_root: Path, site: str, sources: list[str]) -> FactAnswer:
    """
    Phase 3: Deterministic Executor
    Resolve a dataset total with deterministic parsing and source fallbacks.
    """
    for src in sources:
        if src == "probe_json":
            probe = _latest_probe_for_site(project_root, site)
            if probe is None:
                continue
            rel_probe = str(probe.relative_to(project_root)).replace("\\", "/")
            try:
                data = json.loads(probe.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):
                td = data.get("total_datasets")
                if isinstance(td, int) and td > 0:
                    return FactAnswer(
                        kind="dataset_total",
                        site=site,
                        value=td,
                        evidence=[f"`{rel_probe}` -> `total_datasets={td}`"],
                        confidence="high",
                        scope="source_wide_total",
                        source_family="probe_json",
                        exhaustive=True,
                    )
                total, _sample = _dataset_count_and_sample(data)
                if total > 0:
                    return FactAnswer(
                        kind="dataset_total",
                        site=site,
                        value=total,
                        evidence=[f"`{rel_probe}` -> `datasets` has {total} entries"],
                        confidence="high",
                        scope="source_wide_total",
                        source_family="probe_json",
                        exhaustive=True,
                    )
        if src == "generated_downloader_script":
            script_total = _extract_total_datasets_from_script(project_root, site)
            if script_total is None:
                continue
            n, rel = script_total
            return FactAnswer(
                kind="dataset_total",
                site=site,
                value=n,
                evidence=[f"`{rel}` -> `TOTAL_DATASETS = {n}`"],
                confidence="medium",
                scope="pipeline_selected_subset",
                source_family="generated_downloader_script",
                exhaustive=True,
            )

    return FactAnswer(
        kind="dataset_total",
        site=site,
        value=None,
        evidence=[],
        confidence="low",
        scope="source_wide_total",
        source_family="unknown",
        exhaustive=False,
        missing_reason="no matching probe or downloader script with a dataset total was found",
    )


def _execute_mito_labeled_count(project_root: Path, site: str, sources: list[str]) -> FactAnswer:
    for src in sources:
        if src == "probe_json":
            probe = _latest_probe_for_site(project_root, site)
            if probe is None:
                continue
            rel_probe = str(probe.relative_to(project_root)).replace("\\", "/")
            try:
                data = json.loads(probe.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):
                # 1) explicit count field (if present)
                explicit = data.get("mito_labeled_count")
                if isinstance(explicit, int) and explicit >= 0:
                    return FactAnswer(
                        kind="mito_labeled_count",
                        site=site,
                        value=explicit,
                        evidence=[f"`{rel_probe}` -> `mito_labeled_count={explicit}`"],
                        confidence="high",
                        scope="source_wide_metadata_derived",
                        source_family="probe_json",
                        exhaustive=True,
                    )
                # 2) infer from datasets map/list using mito-related keys/labels
                ds_obj = data.get("datasets")
                entries: list[tuple[str, Any]] = []
                if isinstance(ds_obj, dict):
                    entries = list(ds_obj.items())
                elif isinstance(ds_obj, list):
                    entries = [(str(i), row) for i, row in enumerate(ds_obj)]
                if entries:
                    n = 0
                    for _k, row in entries:
                        if not isinstance(row, dict):
                            continue
                        mito_path = str(row.get("mito_seg_path") or "").strip().lower()
                        if mito_path and mito_path not in {"null", "none"}:
                            n += 1
                            continue
                        labels = row.get("label_names")
                        if isinstance(labels, list) and any("mito" in str(x).lower() for x in labels):
                            n += 1
                            continue
                    return FactAnswer(
                        kind="mito_labeled_count",
                        site=site,
                        value=n,
                        evidence=[
                            f"`{rel_probe}` -> inferred from `datasets[*].mito_seg_path` or "
                            f"`datasets[*].label_names` containing 'mito' ({n})"
                        ],
                        confidence="high",
                        scope="source_wide_metadata_derived",
                        source_family="probe_json",
                        exhaustive=True,
                    )
        if src == "registry_sqlite":
            conn = _open_registry_ro(project_root)
            if conn is None:
                continue
            try:
                prov = _provider_names_for_site(site)
                ph = ",".join("?" for _ in prov)
                sql = f"""
                    SELECT COUNT(*) AS n
                    FROM datasets d
                    JOIN providers p ON p.id = d.provider_id
                    WHERE p.name IN ({ph})
                      AND (
                        lower(COALESCE(json_extract(d.metadata_json, '$.mito_seg_path'), '')) NOT IN ('', 'null')
                        OR EXISTS (
                          SELECT 1
                          FROM json_each(COALESCE(json_extract(d.metadata_json, '$.label_names'), '[]'))
                          WHERE lower(json_each.value) LIKE '%mito%'
                        )
                      )
                """
                row = conn.execute(sql, tuple(prov)).fetchone()
                n = int(row["n"]) if row else 0
                p = _registry_path(project_root)
                return FactAnswer(
                    kind="mito_labeled_count",
                    site=site,
                    value=n,
                    evidence=[
                        f"`{str(p.relative_to(project_root)).replace(chr(92), '/')}` -> "
                        f"datasets joined with providers; mito label inferred from "
                        f"`metadata_json.mito_seg_path` or `metadata_json.label_names` containing 'mito' ({n})"
                    ],
                    confidence="high",
                    scope="source_wide_metadata_derived",
                    source_family="registry_sqlite",
                    exhaustive=True,
                )
            finally:
                conn.close()
        if src == "generated_downloader_script":
            script_total = _extract_total_datasets_from_script(project_root, site)
            if script_total is not None:
                n, rel = script_total
                return FactAnswer(
                    kind="mito_labeled_count",
                    site=site,
                    value=n,
                    evidence=[f"`{rel}` -> `TOTAL_DATASETS = {n}` (pipeline labeled subset)"],
                    confidence="medium",
                    scope="pipeline_selected_subset",
                    source_family="generated_downloader_script",
                    exhaustive=True,
                )
    return FactAnswer(
        kind="mito_labeled_count",
        site=site,
        value=None,
        evidence=[],
        confidence="low",
        scope="source_wide_metadata_derived",
        source_family="unknown",
        exhaustive=False,
        missing_reason="registry and script evidence for mito-labeled count unavailable",
    )


def _execute_downloaded_total_filesystem(project_root: Path) -> FactAnswer:
    from config.paths import nnunet_dataset_root

    ds_root = nnunet_dataset_root(project_root)
    images = ds_root / "imagesTr"
    labels = ds_root / "labelsTr"
    if not images.is_dir() or not labels.is_dir():
        return FactAnswer(
            kind="downloaded_total",
            site="all",
            value=None,
            evidence=[],
            confidence="low",
            scope="currently_downloaded_present",
            source_family="filesystem",
            exhaustive=False,
            missing_reason="Dataset001_mito2 imagesTr/labelsTr directories not found",
        )
    im_files = sorted(images.glob("*_0000.nii.gz"))
    seg_files = sorted(labels.glob("*.nii.gz"))
    seg_set = {p.name for p in seg_files}
    pair_count = 0
    for p in im_files:
        expected = p.name.replace("_0000.nii.gz", ".nii.gz")
        if expected in seg_set:
            pair_count += 1
    return FactAnswer(
        kind="downloaded_total",
        site="all",
        value=pair_count,
        evidence=[
            "`data/nnUNet_raw/Dataset001_mito2/imagesTr/*_0000.nii.gz` and "
            "`data/nnUNet_raw/Dataset001_mito2/labelsTr/*.nii.gz` "
            f"paired count = {pair_count} (images={len(im_files)}, labels={len(seg_files)})"
        ],
        confidence="high",
        scope="currently_downloaded_present",
        source_family="filesystem",
        exhaustive=True,
    )


def _execute_downloaded_total(project_root: Path, sources: list[str]) -> FactAnswer:
    fallback_fs: FactAnswer | None = None
    for src in sources:
        if src == "filesystem":
            fallback_fs = _execute_downloaded_total_filesystem(project_root)
            continue
        if src != "registry_sqlite":
            continue
        conn = _open_registry_ro(project_root)
        if conn is None:
            continue
        try:
            row = conn.execute(
                """
                SELECT COUNT(DISTINCT stable_id) AS n
                FROM batch_items
                WHERE status = 'present'
                """
            ).fetchone()
            n = int(row["n"]) if row else 0
            p = _registry_path(project_root)
            return FactAnswer(
                kind="downloaded_total",
                site="all",
                value=n,
                evidence=[
                    f"`{str(p.relative_to(project_root)).replace(chr(92), '/')}` -> "
                    f"`batch_items.status='present'` distinct `stable_id` = {n}"
                ],
                confidence="high",
                scope="currently_downloaded_present",
                source_family="registry_sqlite",
                exhaustive=True,
            )
        finally:
            conn.close()
    if fallback_fs is not None and fallback_fs.value is not None:
        return fallback_fs
    return FactAnswer(
        kind="downloaded_total",
        site="all",
        value=None,
        evidence=[],
        confidence="low",
        scope="currently_downloaded_present",
        source_family="unknown",
        exhaustive=False,
        missing_reason="registry evidence for downloaded_total unavailable",
    )


def _execute_downloaded_by_provider(project_root: Path, sources: list[str]) -> FactAnswer:
    for src in sources:
        if src != "registry_sqlite":
            continue
        conn = _open_registry_ro(project_root)
        if conn is None:
            continue
        try:
            rows = conn.execute(
                """
                SELECT COALESCE(db.provider, 'Unknown') AS provider,
                       COUNT(DISTINCT bi.stable_id) AS n
                FROM batch_items bi
                JOIN download_batches db ON db.id = bi.batch_db_id
                WHERE bi.status = 'present'
                GROUP BY COALESCE(db.provider, 'Unknown')
                ORDER BY n DESC
                """
            ).fetchall()
            parts = [f"{r['provider']}:{int(r['n'])}" for r in rows]
            p = _registry_path(project_root)
            return FactAnswer(
                kind="downloaded_by_provider",
                site="all",
                value=sum(int(r["n"]) for r in rows),
                evidence=[
                    f"`{str(p.relative_to(project_root)).replace(chr(92), '/')}` -> "
                    f"`batch_items` x `download_batches` where status='present': {parts or ['none']}"
                ],
                confidence="high",
                scope="currently_downloaded_present",
                source_family="registry_sqlite",
                exhaustive=True,
            )
        finally:
            conn.close()
    return FactAnswer(
        kind="downloaded_by_provider",
        site="all",
        value=None,
        evidence=[],
        confidence="low",
        scope="currently_downloaded_present",
        source_family="unknown",
        exhaustive=False,
        missing_reason="registry evidence for downloaded_by_provider unavailable",
    )


def _collect_question_facts(project_root: Path, plan: QuestionPlan) -> list[FactAnswer]:
    facts: list[FactAnswer] = []
    for item in plan.items:
        sources = _route_sources_for_question(item)
        if item.kind == "dataset_total":
            facts.append(_execute_dataset_total(project_root, item.site, sources))
        elif item.kind == "mito_labeled_count":
            facts.append(_execute_mito_labeled_count(project_root, item.site, sources))
        elif item.kind == "downloaded_total":
            # Keep both registry and filesystem views; they can conflict and need reconciliation by scope.
            facts.append(_execute_downloaded_total(project_root, ["registry_sqlite"]))
            facts.append(_execute_downloaded_total_filesystem(project_root))
        elif item.kind == "downloaded_by_provider":
            facts.append(_execute_downloaded_by_provider(project_root, sources))
    return facts


def _conflict_notes_for_facts(facts: list[FactAnswer]) -> list[str]:
    notes: list[str] = []
    keyed: dict[tuple[str, str], list[FactAnswer]] = {}
    for f in facts:
        if f.value is None:
            continue
        keyed.setdefault((f.kind, f.site), []).append(f)
    for (kind, site), rows in keyed.items():
        vals = {r.value for r in rows if r.value is not None}
        if len(vals) <= 1:
            continue
        desc = ", ".join(
            f"{r.source_family}:{r.value} scope={r.scope} exhaustive={r.exhaustive}"
            for r in rows
            if r.value is not None
        )
        notes.append(f"kind={kind} site={site}: {desc}")
    return notes


def _question_agent_phase_context(project_root: Path, user_message: str) -> str:
    """
    Run phased Q&A context (planner/router/executor-facts) for the LLM.
    The LLM still produces final text, but must obey fact scope/exhaustiveness.
    """
    plan = _plan_question(user_message)
    if not plan.is_question:
        return "Question planner: not a question; no deterministic Q&A phases applied."
    if not plan.items:
        return "Question planner: question detected, but no deterministic resolver matched; use general grounding."

    facts = _collect_question_facts(project_root, plan)
    lines: list[str] = ["Question planner: sub-questions identified."]
    for idx, item in enumerate(plan.items, start=1):
        lines.append(f"- item_{idx}: kind={item.kind} site={item.site}")
        sources = _route_sources_for_question(item)
        lines.append(f"  router_sources={sources}")
    lines.append("Structured fact records:")
    for f in facts:
        if f.value is None:
            lines.append(
                f"- fact kind={f.kind} site={f.site} value=missing source={f.source_family} "
                f"scope={f.scope} exhaustive={f.exhaustive} confidence={f.confidence} reason={f.missing_reason}"
            )
            continue
        lines.append(
            f"- fact kind={f.kind} site={f.site} value={f.value} source={f.source_family} "
            f"scope={f.scope} exhaustive={f.exhaustive} confidence={f.confidence}"
        )
        for ev in f.evidence:
            lines.append(f"  evidence: {ev}")
    conflicts = _conflict_notes_for_facts(facts)
    if conflicts:
        lines.append("Conflict check:")
        for note in conflicts:
            lines.append(f"- {note}")
    else:
        lines.append("Conflict check: none detected among available fact records.")
    lines.append(
        "Instruction: For numeric totals, only present exact counts from facts with exhaustive=True. "
        "If only non-exhaustive facts exist, explicitly label as partial/at least. "
        "If conflicts exist, report by scope/source instead of collapsing to one number."
    )
    return "\n".join(lines)


def _question_answering_policy() -> str:
    """
    System-level reminder for LLM answer composition.
    Keep this generic: guide evidence behavior without forcing a single hard-coded file.
    """
    return (
        "When answering factual questions, do not rely on only one evidence source family.\n"
        "Triangulate across relevant local artifacts when available (catalog/probe outputs, registry state, "
        "generated scripts, and local filesystem state).\n"
        "Use structured fact records as grounding, and cross-check across source families when scope might differ.\n"
        "If counts differ, do not collapse to one number silently. Report both with scope labels, e.g.:\n"
        "- source-wide total,\n"
        "- pipeline-selected subset,\n"
        "- currently downloaded/present subset,\n"
        "- model-configured/used subset.\n"
        "Count confidence gate: only state exact totals from exhaustive evidence; otherwise label as partial.\n"
        "For app-functionality questions (purpose, stages/modules/pages/buttons), map behavior to concrete "
        "repo artifacts/endpoints and state side effects (file/table/status changes).\n"
        "For plan-clarification questions, explicitly define plan terms (sites, stages, sample type filters, "
        "n_crops split, expected outputs) before giving recommendations.\n"
        "For pipeline report questions, explain: first failing stage, direct error message, downstream skipped work, "
        "and practical consequence.\n"
        "Prefer concise direct answers first, then a short reconciliation note only when needed."
    )


def _detect_scrape_url(text: str) -> Optional[str]:
    m = re.search(r"(https?://[^\s]+)", text)
    if m:
        return m.group(1).rstrip(").,;]")
    if re.search(r"\bscrape\b", text, re.I):
        m2 = re.search(r"\b(?:https?://)?(?:www\.)?[\w.-]+\.[a-z]{2,}[^\s]*", text, re.I)
        if m2:
            u = m2.group(0).rstrip(").,;]")
            if not u.startswith("http"):
                u = "http://" + u
            return u
    return None


def propose_actions_for_message(
    project_root: Path,
    message: str,
    session: PipelineSession,
    queue: ApprovalQueue,
) -> List[PendingAction]:
    """Heuristic proposals; LLM copy explains, user approves in UI."""
    actions: List[PendingAction] = []
    msg = message.strip()
    py = sys.executable

    url = _detect_scrape_url(msg)
    if url and re.search(r"\bscrape\b|probe|fetch|website", msg, re.I):
        s1 = project_root / "1web_scraper_01"
        actions.append(
            queue.propose(
                title=f"Stage 1: generic scrape `{url}`",
                command=[py, "agent.py", url],
                cwd=str(s1),
                detail={"step": "scrape", "url": url},
            )
        )

    if re.search(
        r"\b(build|make)\s+(database|schema)\b|\b(database|schema)\b.*\b(db|sqlite|catalog|inventory)\b",
        msg,
        re.I,
    ):
        s2 = project_root / "2database_builder"
        actions.append(
            queue.propose(
                title="Stage 2: rebuild SQLite inventory from latest probe",
                command=[py, "agent.py"],
                cwd=str(s2),
                detail={"step": "database"},
            )
        )

    if re.search(r"\b(generate|write)\b.*\bdownload\b.*\bscript\b", msg, re.I):
        site = session.last_site_stem or "site"
        s3 = project_root / "3data_downloader"
        actions.append(
            queue.propose(
                title="Stage 3: generate downloader script",
                command=[py, "downloader_master/agent.py", "--volume-stub", "--site", site],
                cwd=str(s3),
                detail={"step": "download_script", "site": site},
            )
        )

    if re.search(r"\brun\b.*\bdownload\b|\bexecute\b.*\bdownload\b", msg, re.I):
        site = session.last_site_stem or "site"
        out_dir = project_root / "3data_downloader" / "outputs"
        script = out_dir / f"download_{site}.py"
        if not script.is_file():
            candidates = sorted(out_dir.glob(f"download_{site}_*.py"), key=lambda p: p.stat().st_mtime, reverse=True)
            script = candidates[0] if candidates else script
        if script.is_file():
            actions.append(
                queue.propose(
                    title=f"Execute generated downloader `{script.name}`",
                    command=[py, str(script)],
                    cwd=str(project_root),
                    detail={"step": "download_execute"},
                )
            )

    if re.search(r"\bpreprocess\b", msg, re.I):
        s4 = project_root / "3data_downloader"
        actions.append(
            queue.propose(
                title="Stage 4: preprocess stub",
                command=[py, "downloader_master/preprocess_agent.py"],
                cwd=str(s4),
                detail={"step": "preprocess"},
            )
        )

    if re.search(r"\bfinetun|fine[- ]tun|supervised\b", msg, re.I):
        s6 = project_root / "6ml_finetune"
        actions.append(
            queue.propose(
                title="Stage 6: supervised fine-tune stub",
                command=[py, "agent.py"],
                cwd=str(s6),
                detail={"step": "finetune"},
            )
        )

    if re.search(r"\bbenchmark\b|\bbaseline\b|\beval\b", msg, re.I):
        bench_py = (
            "import json, pathlib; "
            "p = pathlib.Path('data/ml_runs'); p.mkdir(parents=True, exist_ok=True); "
            "path = p / 'benchmark_stub.json'; "
            "path.write_text(json.dumps("
            "{'dice_mean': None, 'surface_distance_voxel': None, 'note': 'fill after training'}, "
            "indent=2))"
        )
        actions.append(
            queue.propose(
                title="Evaluation: write benchmark placeholder metrics",
                command=[py, "-c", bench_py],
                cwd=str(project_root),
                detail={"step": "eval"},
            )
        )

    return actions


def run_chat_turn(
    project_root: Path,
    session: PipelineSession,
    queue: ApprovalQueue,
    user_message: str,
    history: List[dict[str, str]],
    *,
    chat_session_id: str = "default",
    chat_mode: str = "ask",
    pipeline_action: Optional[str] = None,
    pipeline_plan: Optional[Dict[str, Any]] = None,
    on_planning_hint: Optional[Callable[[Dict[str, Any]], None]] = None,
    stop_requested: Optional[Callable[[], bool]] = None,
) -> Tuple[str, List[PendingAction], PromptBundle, Dict[str, Any]]:
    # Overhaul policy: ignore UI-customized prompt/skill overrides and use
    # built-in defaults for stable agent behavior.
    prompts = PromptBundle()
    mode = (chat_mode or "ask").strip().lower()
    if mode not in ("ask", "plan", "agent"):
        mode = "ask"
    extra: Dict[str, Any] = {"draft_pipeline_plan": None, "chat_mode": mode}
    planning_snapshots: List[dict[str, Any]] = []

    def emit_planning_hint(payload: dict[str, Any]) -> None:
        planning_snapshots.append(payload)
        if on_planning_hint:
            on_planning_hint(payload)
        extra["planning_hint"] = payload

    # --- Agent: execute an approved structured plan (no extra LLM) ---
    act = (pipeline_action or "").strip().lower()
    if act == "execute_plan":
        if mode != "agent":
            return (
                "Switch to **Agent** mode to execute a saved or approved pipeline plan.",
                [],
                prompts,
                extra,
            )
        exec_plan: Dict[str, Any] | None = None
        if isinstance(pipeline_plan, dict) and pipeline_plan:
            exec_plan = pipeline_plan
        if exec_plan is None:
            return (
                "No approved/saved pipeline plan is available to execute.\n\n"
                "Please generate a plan first, then use **Approve & run**.",
                [],
                prompts,
                extra,
            )
        emit_planning_hint(
            planning_trace_from_intent_decision({"ui_focus": "", "plan": exec_plan}, draft_plan=exec_plan)
        )
        body, meta = execute_pipeline_plan(
            project_root,
            chat_session_id,
            session,
            exec_plan,
            stop_requested=stop_requested,
        )
        extra["execution"] = meta
        return body, [], prompts, extra

    # Reliability fallback: if Agent-mode user explicitly asks to execute the approved plan,
    # execute the latest saved plan even when UI action metadata is missing (e.g. edit/refresh edges).
    msg_l = (user_message or "").strip().lower()
    wants_execute_approved = bool(
        re.search(r"\bexecute\b.*\bapproved\b.*\bplan\b", msg_l)
        or re.search(r"\bapprove\s*&?\s*run\b", msg_l)
    )
    if mode == "agent" and act != "execute_plan" and wants_execute_approved:
        fallback_plan: Dict[str, Any] | None = None
        if isinstance(pipeline_plan, dict) and pipeline_plan:
            fallback_plan = pipeline_plan
        if fallback_plan is None:
            return (
                "I couldn't find an approved/saved pipeline plan to execute.\n\n"
                "Please generate a plan first, then use **Approve & run**.",
                [],
                prompts,
                extra,
            )
        emit_planning_hint(
            planning_trace_from_intent_decision({"ui_focus": "", "plan": fallback_plan}, draft_plan=fallback_plan)
        )
        body, meta = execute_pipeline_plan(
            project_root,
            chat_session_id,
            session,
            fallback_plan,
            stop_requested=stop_requested,
        )
        extra["execution"] = meta
        return body, [], prompts, extra

    # --- LLM-first routing: decide the next task for this turn ---
    # Planner-first policy: the LLM planner must provide the structured plan.
    decision = llm_route_chat_intent(
        project_root,
        user_message,
        mode=mode,
        history=history,
    )
    extra["intent"] = decision
    llm_routed = bool(decision.get("routed_via_llm"))
    if mode == "agent" and not llm_routed:
        return (
            "Agent mode requires LLM intent routing for this turn.\n\n"
            "Please check LLM settings and retry.",
            [],
            prompts,
            extra,
        )

    pipeline_like_prompt = message_touches_pipeline(user_message)
    if decision.get("intent") in {"pipeline_plan", "pipeline_execute"} or (
        pipeline_like_prompt and mode in {"plan", "agent"}
    ):
        if decision.get("intent") not in {"pipeline_plan", "pipeline_execute"}:
            # For pipeline-like prompts in Plan/Agent mode, always draft a full structured plan.
            decision["intent"] = "pipeline_plan"
            decision["requires_mode"] = "plan" if mode == "plan" else "agent"
        if not llm_routed:
            emit_planning_hint(planning_trace_from_intent_decision(decision))
            return (
                "I couldn't route this pipeline request through the LLM intent router.\n\n"
                "Please check LLM settings and try again.",
                [],
                prompts,
                extra,
            )
        need_mode = str(decision.get("requires_mode") or "").strip().lower()
        if need_mode == "agent" and mode != "agent":
            emit_planning_hint(planning_trace_from_intent_decision(decision))
            hint = ask_mode_plan_switch_reply(user_message) or (
                "Switch to **Agent** mode to run scrape / database / download from chat."
            )
            return hint, [], prompts, extra
        if need_mode == "plan" and mode == "ask":
            emit_planning_hint(planning_trace_from_intent_decision(decision))
            return (
                "Switch to **Plan** mode to draft/save a structured pipeline plan, or **Agent** mode to execute it.",
                [],
                prompts,
                extra,
            )
        repaired_plan: Dict[str, Any] | None = None
        plan_candidate = decision.get("plan")
        plan_incomplete = not (
            isinstance(plan_candidate, dict)
            and isinstance(plan_candidate.get("sites"), list)
            and bool(plan_candidate.get("sites"))
            and (
                (
                    isinstance(plan_candidate.get("stages"), list)
                    and bool(plan_candidate.get("stages"))
                )
                or (
                    isinstance(plan_candidate.get("per_site_stages"), dict)
                    and bool(plan_candidate.get("per_site_stages"))
                )
            )
        )
        if plan_incomplete:
            repaired = llm_complete_pipeline_plan(project_root, user_message, mode=mode, history=history)
            if isinstance(repaired, dict) and repaired:
                decision["plan"] = repaired
                repaired_plan = repaired
        if mode == "agent" and not isinstance(decision.get("plan"), dict):
            emit_planning_hint(planning_trace_from_intent_decision(decision))
            return (
                "I couldn't get a complete LLM-generated pipeline plan for this request.\n\n"
                "Please retry with the same intent; the planner needs to return sites/stages/n_crops.",
                [],
                prompts,
                extra,
            )
        # Fast path: use the LLM intent router plan directly (single LLM call).
        # This avoids a second LLM merge pass, reducing latency before approval.
        merged = (
            dict(decision.get("plan"))
            if isinstance(decision.get("plan"), dict)
            else {}
        )
        if repaired_plan is not None:
            extra["intent_repair"] = "llm_complete_pipeline_plan"
        if isinstance(merged, dict):
            coerced_top = _coerce_pipeline_plan(merged, user_message=user_message, history=history)
            if coerced_top:
                merged.update(coerced_top)
            unsupported_sources = (
                merged.get("unsupported_sources")
                if isinstance(merged.get("unsupported_sources"), list)
                else []
            )
            if unsupported_sources:
                bad = ", ".join(str(x) for x in unsupported_sources if str(x).strip())
                emit_planning_hint(planning_trace_from_intent_decision(decision, draft_plan=merged if merged else None))
                return (
                    "I found unsupported source(s) in your request and will not silently reroute.\n\n"
                    f"- Unsupported: {bad}\n"
                    "- Supported sources: OpenOrganelle, BossDB, LocalHPC\n"
                    "Please restate with supported sources.",
                    [],
                    prompts,
                    extra,
                )
            if not isinstance(merged.get("sites"), list) or not merged.get("sites"):
                emit_planning_hint(planning_trace_from_intent_decision(decision, draft_plan=merged if merged else None))
                return (
                    "LLM planner returned an incomplete plan (missing sites).\n\n"
                    "Please retry so the planner can provide a complete structured plan.",
                    [],
                    prompts,
                    extra,
                )
            if not isinstance(merged.get("n_crops"), int):
                try:
                    merged["n_crops"] = max(1, min(64, int(merged.get("n_crops", 1))))
                except (TypeError, ValueError):
                    merged["n_crops"] = 1
            st0 = merged.get("stages")
            stages: list[str] = []
            if isinstance(st0, list):
                stages = _canonicalize_plan_stages_list(st0)
                if "download" in stages and "database" not in stages:
                    stages = list(stages)
                    stages.insert(stages.index("download"), "database")
            merged["stages"] = stages
            pss0 = merged.get("per_site_stages")
            if isinstance(pss0, dict) and pss0 and not merged.get("stages"):
                seen_u: set[str] = set()
                union_st: list[str] = []
                for sname in merged.get("sites") or []:
                    lst = pss0.get(sname)
                    if not isinstance(lst, list):
                        continue
                    for s in lst:
                        sn = _canonical_pipeline_stage_token(s)
                        if sn and sn not in seen_u:
                            seen_u.add(sn)
                            union_st.append(sn)
                merged["stages"] = union_st
            st_fix = merged.get("stages")
            if isinstance(st_fix, list) and "download" in st_fix and "database" not in st_fix:
                st_fix = list(st_fix)
                st_fix.insert(st_fix.index("download"), "database")
                merged["stages"] = st_fix
            has_stages = isinstance(merged.get("stages"), list) and bool(merged.get("stages"))
            has_pss = isinstance(merged.get("per_site_stages"), dict) and bool(merged.get("per_site_stages"))
            if not has_stages and not has_pss:
                emit_planning_hint(planning_trace_from_intent_decision(decision, draft_plan=merged if merged else None))
                return (
                    "LLM planner returned an incomplete plan (missing `stages`; add `plan.stages` or `plan.per_site_stages`).\n\n"
                    "Please retry so the planner can return a complete structured plan.",
                    [],
                    prompts,
                    extra,
                )
        extra["draft_pipeline_plan"] = merged
        emit_planning_hint(planning_trace_from_intent_decision(decision, draft_plan=merged))
        sites = merged.get("sites") if isinstance(merged, dict) else None
        stages = merged.get("stages") if isinstance(merged, dict) else None
        n_crops = merged.get("n_crops") if isinstance(merged, dict) else None
        sites_txt = ", ".join(str(s) for s in sites) if isinstance(sites, list) and sites else "(unspecified)"
        stages_txt = (
            ", ".join(str(s) for s in stages)
            if isinstance(stages, list) and stages
            else "scrape, database, download, training"
        )
        show_n_crops = isinstance(stages, list) and ("download" in stages)
        try:
            n_crops_txt = str(int(n_crops))
        except (TypeError, ValueError):
            n_crops_txt = "1"
        lines = [
            "Prepared a structured pipeline plan for approval.",
            "",
            f"- Sites: {sites_txt}",
            f"- Stages: {stages_txt}",
        ]
        if show_n_crops:
            lines.append(f"- n_crops: {n_crops_txt}")
        dl_cfg = merged.get("download") if isinstance(merged, dict) else None
        if isinstance(dl_cfg, dict):
            st_inc = dl_cfg.get("sample_types_include")
            st_exc = dl_cfg.get("sample_types_exclude")
            tgt = str(dl_cfg.get("target") or "training")
            tr_c = dl_cfg.get("training_crops")
            inf_c = dl_cfg.get("inference_crops")
            tot_c = dl_cfg.get("total_crops")
            if tr_c is None and inf_c is None:
                if tgt == "inference":
                    tr_txt, inf_txt = "0", str(int(tot_c or 1))
                else:
                    tr_txt, inf_txt = str(int(tot_c or 1)), "0"
            else:
                tr_txt = str(int(tr_c or 0))
                inf_txt = str(int(inf_c or 0))
            lines.append(f"- Download target: {tgt}")
            lines.append(f"- n_crops Training: {tr_txt}")
            lines.append(f"- n_crops Inference: {inf_txt}")
            if isinstance(st_inc, list) and st_inc:
                lines.append(f"- Sample types include: {', '.join(str(x) for x in st_inc)}")
            else:
                lines.append("- Sample types: unspecified (default uses all)")
            if isinstance(st_exc, list) and st_exc:
                lines.append(f"- Sample types exclude: {', '.join(str(x) for x in st_exc)}")
        else:
            # Keep plan output explicit and stable even when LLM omits `download`.
            if isinstance(stages, list) and "download" in stages:
                lines.extend(
                    [
                        "- Download target: training",
                        f"- n_crops Training: {n_crops_txt}",
                        "- n_crops Inference: 0",
                        "- Sample types: unspecified (default uses all)",
                    ]
                )
        narrative = "\n".join(lines)
        # Reliability/safety policy:
        # - Natural-language turns always produce a structured draft plan first.
        # - Actual execution only occurs through explicit `pipeline_action=execute_plan`
        #   (Approve & run flow), not by inferred intent alone.
        if decision.get("intent") == "pipeline_execute":
            if mode == "agent":
                return (
                    narrative.strip()
                    + "\n\nPrepared an executable pipeline plan for approval. "
                    "Use **Approve & run** to execute it.",
                    [],
                    prompts,
                    extra,
                )
            ref = plan_mode_execution_refusal(user_message) or (
                "Prepared a draft pipeline plan. Switch to **Agent** mode and use approval to execute it."
            )
            return ref, [], prompts, extra
        return narrative, [], prompts, extra

    probe_line = _latest_probe_summary(project_root)
    artifact_line = _artifact_evidence(project_root, user_message)
    data_line = _data_inventory_evidence(project_root)
    q_phase_line = _question_agent_phase_context(project_root, user_message)
    q_policy = _question_answering_policy()

    step_label = STEP_LABELS.get(int(session.current_step), "idle")
    user_block = prompts.format_user(
        step_label=step_label,
        last_url=session.last_url or "(none)",
        last_site_stem=session.last_site_stem or "(unknown)",
    )
    user_block += user_message

    system = prompts.system
    system += f"\n\n## Repo snapshot\n{probe_line}\n"
    if data_line:
        system += "\n## Local data inventory\n" + data_line + "\n"
    if artifact_line:
        system += "\n## Local Artifact Evidence (generic RAG)\n" + artifact_line + "\n"
    if q_phase_line:
        system += "\n## Question agent phases (planner -> router -> executor)\n" + q_phase_line + "\n"
    if q_policy:
        system += "\n## Question answering policy\n" + q_policy + "\n"
    ensure_skill_trees(project_root)
    orch_blk = merged_orchestration_skills_block(project_root)
    if orch_blk.strip():
        system += "\n## Orchestration skills (repo; editable under Settings → Agent skills)\n" + orch_blk + "\n"
    chat_blk = merged_chat_skills_block(project_root)
    if chat_blk.strip():
        system += "\n## Chat skills (repo; editable under Settings → Agent skills)\n" + chat_blk + "\n"
    # Prompt understanding and execution planning are LLM-routed; avoid regex-based action proposals.
    proposed: list[PendingAction] = []
    if proposed:
        ids = ", ".join(p.id for p in proposed)
        system += (
            f"\nYou proposed shell actions pending approval (ids: {ids}). "
            "Summarize them clearly and tell the user to approve or decline in the UI.\n"
        )
    if mode == "plan":
        system += (
            "\nThe user is in **Plan** mode: prioritize explanation and clarification. "
            "If the user asks what a plan means, explain each plan field in plain language and expected artifacts. "
            "If they ask to run scrape/database/download/training from chat, tell them to switch to **Agent** mode.\n"
        )
    elif mode == "agent":
        system += (
            "\nThe user is in **Agent** mode for conversational turns without an active pipeline draft. "
            "If they need a structured multi-stage run, they should ask using scrape/database/download/training keywords "
            "so the app can show a plan card. For run-report/failure questions, ground explanations in stage results "
            "and execution evidence before suggesting reruns.\n"
        )

    messages: List[dict[str, str]] = [{"role": "system", "content": system}]
    for m in history[-24:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_block})

    emit_planning_hint(planning_trace_from_intent_decision(decision))
    try:
        reply = complete(messages, max_tokens=2048)
    except LLMUnavailableError as e:
        reply = _fallback_reply_when_llm_down(proposed, str(e), e.hint or setup_hint())
    return reply, proposed, prompts, extra


def _fallback_reply_when_llm_down(
    proposed: List[PendingAction],
    error_line: str,
    hint: str,
) -> str:
    parts = [
        "**LLM is not available** (this message is generated locally, not by a model).",
        "",
        f"_{error_line}_",
        "",
    ]
    if proposed:
        parts.append("You can still **approve or decline** the pending actions below.")
        parts.append("")
        for p in proposed:
            cmd = " ".join(shlex.quote(a) for a in p.command)
            parts.append(f"- **{p.title}**  \n  `{cmd}`")
        parts.append("")
    else:
        parts.append("No shell actions were detected from this message. Once the LLM works, try phrases like “scrape http://…”.")
        parts.append("")
    parts.append(hint)
    return "\n".join(parts)


def update_session_after_scrape(project_root: Path, session: PipelineSession, url: str) -> None:
    session.last_url = url
    s1 = str(project_root / "1web_scraper_01")
    if s1 not in sys.path:
        sys.path.insert(0, s1)
    from master.generic_scrape import _site_stem  # noqa: PLC0415

    session.last_site_stem = _site_stem(url)
    session.current_step = PipelineStep.SCRAPE
    session.log("scrape", {"url": url, "site": session.last_site_stem})
