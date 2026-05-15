"""FastAPI routes: chat, approvals, prompts, skills, static UI."""

from __future__ import annotations

import os
import json
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from queue import Empty, Queue
from threading import Event, RLock, Thread
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.datastructures import MutableHeaders

from config.paths import project_root as _project_root_from_config

_PROJECT_ROOT = _project_root_from_config()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent.orchestration.approvals import ApprovalQueue, ApprovalStatus, PendingAction
from agent.orchestration.session_pipeline import PipelineSession, PipelineStep, STEP_LABELS
from agent.orchestration.prompts_store import PromptBundle, load_prompts, save_prompts
from agent.orchestration.codex_auth import default_codex_auth_path, list_codex_profiles

from agent.orchestration.llm_settings import (
    env_override_keys,
    load_stored_llm_settings,
    merge_patch_stored,
    public_settings_view,
)
from agent.orchestration.skill_api import (
    SkillRecord,
    create_skill,
    ensure_skill_trees,
    list_chat_skills,
    list_orchestration_skills,
    load_skills,
    merged_skill_bodies,
    read_skill_document,
    rename_skill_slug,
    save_skill_override,
    write_skill_document,
)
from agent.orchestration.codex_cli_bridge import get_codex_login_status, start_codex_login_background

from .agent_turn import run_chat_turn, update_session_after_scrape
from .executor import run_command
from .catalog_api import configure_catalog, router as catalog_router
from .studio_api import (
    StudioDownloaderCancelBody,
    StudioScrapeCancelBody,
    configure_studio,
    router as studio_router,
    studio_run_downloader_script_cancel,
    studio_scrape_cancel,
)
from agent.orchestration.time_utils import now_us_eastern_iso
from .llm import (
    apply_codex_model_storage_repairs,
    get_chatgpt_account_id,
    get_llm_config,
    get_llm_transport,
    is_codex_chatgpt_backend,
    setup_hint,
)


def get_project_root() -> Path:
    env = os.getenv("MITO2_PROJECT_ROOT", "").strip()
    return Path(env).resolve() if env else _PROJECT_ROOT


class ChatMessage(BaseModel):
    role: str
    content: str
    chat_mode: Optional[str] = None


class ChatRequest(BaseModel):
    session_id: str
    message: str
    history: List[ChatMessage] = Field(default_factory=list)
    chat_mode: str = Field(default="ask", description="ask | plan | agent")
    pipeline_action: Optional[str] = Field(default=None, description='e.g. "execute_plan"')
    pipeline_plan: Optional[dict[str, Any]] = None
    stream_planning: bool = Field(
        default=False,
        description="If true, return NDJSON: planning events, then a final done object (LLM-derived waiting hints).",
    )


class ChatResponse(BaseModel):
    reply: str
    pending_approvals: List[dict[str, Any]]
    pipeline: dict[str, Any]
    draft_pipeline_plan: Optional[dict[str, Any]] = None
    execution: Optional[dict[str, Any]] = None
    chat_mode: str = "ask"
    planning_hint: Optional[dict[str, Any]] = None


def _post_chat_result_dict(
    sid: str,
    session: PipelineSession,
    queue: ApprovalQueue,
    reply: str,
    extra: Dict[str, Any],
    cm: str,
) -> Dict[str, Any]:
    pending = [a for a in queue.pending_list() if a.status == ApprovalStatus.PENDING]
    return {
        "reply": reply,
        "pending_approvals": [_pending_payload(a) for a in pending],
        "pipeline": {
            "current_step": int(session.current_step),
            "step_label": STEP_LABELS.get(int(session.current_step), "idle"),
            "last_url": session.last_url,
            "last_site_stem": session.last_site_stem,
        },
        "draft_pipeline_plan": extra.get("draft_pipeline_plan"),
        "execution": extra.get("execution"),
        "chat_mode": cm,
        "planning_hint": extra.get("planning_hint"),
    }


class ChatClearRequest(BaseModel):
    session_id: str


class ChatStopRequest(BaseModel):
    session_id: str


class ChatDeleteRequest(BaseModel):
    session_ids: List[str] = Field(default_factory=list)


class ChatEditRequest(BaseModel):
    session_id: str
    message_index: int
    message: str
    chat_mode: str = Field(default="ask")


class ChatDeleteTurnRequest(BaseModel):
    session_id: str
    message_index: int


class PipelinePlanSaveBody(BaseModel):
    plan: dict[str, Any]
    title: str = ""


class ApprovalResolve(BaseModel):
    approved: bool


class PromptsUpdate(BaseModel):
    system: Optional[str] = None
    user_prefix: Optional[str] = None


class SkillBodyUpdate(BaseModel):
    body: str


class AgentSkillDocumentUpdate(BaseModel):
    document: str


class CreateAgentSkillBody(BaseModel):
    slug: str = Field(..., min_length=1, max_length=128)
    label: str = Field(..., min_length=1, max_length=256)
    title: Optional[str] = Field(default=None, max_length=512)
    id: Optional[str] = Field(default=None, max_length=256)
    body: Optional[str] = None


class RenameAgentSkillBody(BaseModel):
    new_slug: str = Field(..., min_length=1, max_length=128)


class LlmSettingsUpdate(BaseModel):
    openai_base_url: Optional[str] = None
    openai_model: Optional[str] = None
    chatgpt_account_id: Optional[str] = None
    openai_api_key: Optional[str] = None
    codex_auth_profile_id: Optional[str] = None
    codex_auth_json_path: Optional[str] = None
    codex_agent_model_list: Optional[str] = None
    # chatgpt_codex = ChatGPT-hosted Codex (CLI auth.json); openai_api = api.openai.com /v1 chat/completions
    llm_provider: Optional[str] = None


def _normalize_llm_settings_patch(patch: dict[str, Any]) -> dict[str, Any]:
    """
    Keep provider/base coherent when users switch backends.
    - Codex base URL implies ``chatgpt_codex`` provider.
    - Non-Codex base URL implies ``openai_api`` provider.
    """
    out = dict(patch)
    raw_base = out.get("openai_base_url")
    if isinstance(raw_base, str):
        base = raw_base.strip()
        if base:
            out["llm_provider"] = "chatgpt_codex" if is_codex_chatgpt_backend(base) else "openai_api"
    return out


_sessions: Dict[str, PipelineSession] = {}
_queues: Dict[str, ApprovalQueue] = {}
_chat_history: Dict[str, List[dict[str, Any]]] = {}
_chat_meta: Dict[str, dict[str, Any]] = {}
_saved_pipeline_plans: List[dict[str, Any]] = []
_sid_turn_locks: Dict[str, RLock] = {}
_sid_turn_cancel_events: Dict[str, Event] = {}
_chat_store_load_lock = RLock()
_chat_store_loaded = False


def _now_iso() -> str:
    return now_us_eastern_iso()


def _chat_store_path() -> Path:
    """Persisted threads under ``agent/chat_web/data/`` (same package tree as this app), not the repo root."""
    return _PROJECT_ROOT / "agent" / "chat_web" / "data" / ".mito2_chats.json"


def _stale_root_dot_mito2_chat_files() -> List[Path]:
    """Repo / project-root ``.mito2_chats.json`` (canonical store is under ``agent/chat_web/data/``)."""
    raw = (_PROJECT_ROOT / ".mito2_chats.json", get_project_root() / ".mito2_chats.json")
    out: list[Path] = []
    seen: set[str] = set()
    for p in raw:
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _cleanup_stale_chat_store_files() -> None:
    """Remove obsolete paths: root ``.mito2_chats.json``, old ``agent/chat_web/data/chats.json`` (after canonical exists)."""
    pri = _chat_store_path()
    try:
        pri_res = pri.resolve()
    except OSError:
        pri_res = None
    for p in _stale_root_dot_mito2_chat_files():
        try:
            if not p.exists() or p.is_dir():
                continue
            if pri_res is not None:
                try:
                    if p.resolve() == pri_res:
                        continue
                except OSError:
                    pass
            p.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        stale = pri.parent / "chats.json"
        if not pri.is_file() or not stale.is_file():
            return
        if stale.resolve() == pri.resolve():
            return
        if stale.stat().st_mtime > pri.stat().st_mtime:
            shutil.copy2(stale, pri)
        stale.unlink(missing_ok=True)
    except OSError:
        pass


def _migrate_chat_store_if_missing() -> None:
    """One-time seed: copy older ``chats.json`` or root ``.mito2_chats.json`` into canonical ``agent/chat_web/data/.mito2_chats.json``."""
    primary = _chat_store_path()
    try:
        primary.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    if primary.is_file():
        return
    for alt in (
        _PROJECT_ROOT / "agent" / "chat_web" / "data" / "chats.json",
        get_project_root() / "agent" / "chat_web" / "data" / "chats.json",
        _PROJECT_ROOT / "chat_web" / "data" / "chats.json",
        get_project_root() / "chat_web" / "data" / "chats.json",
        _PROJECT_ROOT / "chat_web" / "data" / ".mito2_chats.json",
        get_project_root() / "chat_web" / "data" / ".mito2_chats.json",
    ):
        try:
            if alt.is_file() and alt.resolve() != primary.resolve():
                shutil.copy2(alt, primary)
                return
        except OSError:
            continue
    for leg in _stale_root_dot_mito2_chat_files():
        if leg.is_symlink():
            continue
        if not leg.is_file():
            continue
        try:
            shutil.copy2(leg, primary)
            return
        except OSError:
            continue


def _sanitize_messages(rows: Any) -> List[dict[str, Any]]:
    out: List[dict[str, Any]] = []
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        role = str(row.get("role", "")).strip()
        content = str(row.get("content", ""))
        if role not in {"user", "assistant"}:
            continue
        if not content:
            continue
        msg: dict[str, Any] = {"role": role, "content": content}
        cm = str(row.get("chat_mode", "")).strip().lower()
        if cm in ("ask", "plan", "agent"):
            msg["chat_mode"] = cm
        out.append(msg)
    return out[-200:]


def _chat_title_from_first_user(messages: List[dict[str, Any]]) -> str:
    for m in messages:
        if m.get("role") != "user":
            continue
        text = " ".join((m.get("content") or "").strip().split())
        if not text:
            continue
        words = text.split(" ")
        short = " ".join(words[:9])
        if len(words) > 9:
            short += "…"
        return short[:80]
    return "New chat"


def _sanitize_saved_pipeline_plans(rows: Any) -> List[dict[str, Any]]:
    out: List[dict[str, Any]] = []
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("id") or "").strip()
        created_at = str(row.get("created_at") or "").strip()
        title = str(row.get("title") or "").strip()
        plan = row.get("plan")
        if not pid or not isinstance(plan, dict) or not plan:
            continue
        out.append(
            {
                "id": pid,
                "created_at": created_at or _now_iso(),
                "title": title or pid,
                "plan": plan,
            }
        )
    return out[:200]


def _save_chat_store() -> None:
    path = _chat_store_path()
    rows: list[dict[str, Any]] = []
    for sid, messages in _chat_history.items():
        meta = _chat_meta.get(sid) or {}
        session = _sessions.get(sid) or PipelineSession()
        created_at = str(meta.get("created_at") or _now_iso())
        updated_at = str(meta.get("updated_at") or created_at)
        title = str(meta.get("title") or _chat_title_from_first_user(messages))
        last_plan = meta.get("last_pipeline_plan")
        if not isinstance(last_plan, dict) or not last_plan:
            last_plan = None
        rows.append(
            {
                "id": sid,
                "title": title,
                "created_at": created_at,
                "updated_at": updated_at,
                "last_pipeline_plan": last_plan,
                "last_pipeline_plan_saved_at": str(meta.get("last_pipeline_plan_saved_at") or ""),
                "messages": messages[-200:],
                "pipeline": {
                    "current_step": int(session.current_step),
                    "last_url": session.last_url,
                    "last_site_stem": session.last_site_stem,
                },
            }
        )
    rows.sort(key=lambda r: str(r.get("updated_at", "")), reverse=True)
    payload = {"threads": rows, "saved_pipeline_plans": _saved_pipeline_plans[:200]}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass
    finally:
        _cleanup_stale_chat_store_files()


def _load_chat_store() -> None:
    _migrate_chat_store_if_missing()
    path = _chat_store_path()
    try:
        if not path.is_file():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        threads = raw.get("threads") if isinstance(raw, dict) else None
        if not isinstance(threads, list):
            return
        global _saved_pipeline_plans
        _saved_pipeline_plans = _sanitize_saved_pipeline_plans(raw.get("saved_pipeline_plans"))
        for row in threads:
            if not isinstance(row, dict):
                continue
            sid = str(row.get("id") or "").strip()
            if not sid:
                continue
            messages = _sanitize_messages(row.get("messages"))
            if not messages:
                continue
            _chat_history[sid] = messages
            _chat_meta[sid] = {
                "title": str(row.get("title") or _chat_title_from_first_user(messages)),
                "created_at": str(row.get("created_at") or _now_iso()),
                "updated_at": str(row.get("updated_at") or _now_iso()),
            }
            lp = row.get("last_pipeline_plan")
            if isinstance(lp, dict) and lp:
                _chat_meta[sid]["last_pipeline_plan"] = lp
                _chat_meta[sid]["last_pipeline_plan_saved_at"] = str(
                    row.get("last_pipeline_plan_saved_at") or _now_iso()
                )
            p = row.get("pipeline")
            if isinstance(p, dict):
                s = PipelineSession()
                try:
                    s.current_step = PipelineStep(int(p.get("current_step", 0)))
                except Exception:
                    s.current_step = PipelineStep.IDLE
                s.last_url = str(p.get("last_url") or "")
                s.last_site_stem = str(p.get("last_site_stem") or "")
                _sessions[sid] = s
    finally:
        _cleanup_stale_chat_store_files()


def _ensure_chat_store_loaded() -> None:
    """Lazy-init persisted chat state on first chat API usage."""
    global _chat_store_loaded
    if _chat_store_loaded:
        return
    with _chat_store_load_lock:
        if _chat_store_loaded:
            return
        _load_chat_store()
        _chat_store_loaded = True


def _ensure_chat_exists(sid: str) -> None:
    _ensure_chat_store_loaded()
    if sid in _chat_history:
        return
    now = _now_iso()
    _chat_history[sid] = []
    _chat_meta[sid] = {"title": "New chat", "created_at": now, "updated_at": now}
    _sessions[sid] = PipelineSession()
    _queues[sid] = ApprovalQueue()
    _save_chat_store()


def _chat_preview(messages: List[dict[str, Any]]) -> str:
    for m in reversed(messages):
        text = (m.get("content") or "").strip()
        if text:
            return text[:120]
    return ""


def _touch_chat_meta(sid: str) -> None:
    meta = _chat_meta.setdefault(sid, {"title": "New chat", "created_at": _now_iso(), "updated_at": _now_iso()})
    meta["updated_at"] = _now_iso()
    if not meta.get("title") or meta.get("title") == "New chat":
        meta["title"] = _chat_title_from_first_user(_chat_history.get(sid, []))


def _get_last_pipeline_plan(sid: str) -> Optional[dict[str, Any]]:
    meta = _chat_meta.get(sid) or {}
    plan = meta.get("last_pipeline_plan")
    if isinstance(plan, dict) and plan:
        return plan
    return None


def _infer_plan_from_text_block(text: str) -> Optional[dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None
    if "Prepared a structured pipeline plan for approval." not in raw:
        return None
    m_sites = re.search(r"(?mi)^\s*-\s*Sites:\s*(.+?)\s*$", raw)
    m_stages = re.search(r"(?mi)^\s*-\s*Stages:\s*(.+?)\s*$", raw)
    m_crops = re.search(r"(?mi)^\s*-\s*n_crops:\s*(\d+)\s*$", raw)
    if not m_sites or not m_stages:
        return None
    sites = [s.strip() for s in re.split(r"\s*,\s*", m_sites.group(1).strip()) if s.strip()]
    stages_raw = [s.strip().lower() for s in re.split(r"\s*,\s*", m_stages.group(1).strip()) if s.strip()]
    stages: list[str] = []
    for s in stages_raw:
        if s == "schema":
            s = "database"
        if s in {"scrape", "database", "download"}:
            stages.append(s)
    if not sites or not stages:
        return None
    n_crops = 1
    if m_crops:
        try:
            n_crops = max(1, int(m_crops.group(1)))
        except (TypeError, ValueError):
            n_crops = 1
    return {
        "sites": sites,
        "stages": stages,
        "n_crops": n_crops,
        "rationale": "Recovered from latest assistant structured plan card.",
    }


def _infer_last_pipeline_plan_from_history(sid: str) -> Optional[dict[str, Any]]:
    rows = _chat_history.get(sid, [])
    for row in reversed(rows):
        if not isinstance(row, dict):
            continue
        if str(row.get("role") or "") != "assistant":
            continue
        plan = _infer_plan_from_text_block(str(row.get("content") or ""))
        if plan:
            return plan
    return None


EXECUTE_APPROVED_PIPELINE_USER_LINE = "Execute approved pipeline plan."


def _plan_dict_has_agent_bar_sites_stages(plan: Any) -> bool:
    if not isinstance(plan, dict) or not plan:
        return False
    if not isinstance(plan.get("sites"), list) or not plan.get("sites"):
        return False
    st = plan.get("stages")
    return isinstance(st, list) and len(st) > 0


def _message_effective_chat_mode(
    row: dict[str, Any], prev: Optional[dict[str, Any]]
) -> str:
    cm = str(row.get("chat_mode", "")).strip().lower()
    if cm in ("ask", "plan", "agent"):
        return cm
    if prev and str(prev.get("role", "")) == "user":
        ucm = str(prev.get("chat_mode", "")).strip().lower()
        if ucm in ("ask", "plan", "agent"):
            return ucm
    return "ask"


def _assistant_looks_like_routed_pipeline_plan_for_agent_bar(content: str) -> bool:
    c = str(content or "")
    return (
        "Prepared a structured pipeline plan for approval." in c
        or "Prepared an executable pipeline plan for approval" in c
        or "Use **Approve & run** to execute it." in c
        or "executable pipeline plan for approval" in c
    )


def _chat_row_has_user_execute_after(
    rows: list[dict[str, Any]], after_idx: int
) -> bool:
    for j in range(after_idx + 1, len(rows)):
        r = rows[j]
        if str(r.get("role", "")) != "user":
            continue
        if str(r.get("content", "")).strip() == EXECUTE_APPROVED_PIPELINE_USER_LINE:
            return True
    return False


def _chat_agent_plan_awaits_approval(sid: str) -> bool:
    """True when the latest plan-offer in history is in Agent mode and the user has not approved yet."""
    lp = _get_last_pipeline_plan(sid) or _infer_last_pipeline_plan_from_history(sid)
    if not _plan_dict_has_agent_bar_sites_stages(lp):
        return False
    rows = _chat_history.get(sid) or []
    for i in range(len(rows) - 1, -1, -1):
        if str(rows[i].get("role", "")) != "assistant":
            continue
        if not _assistant_looks_like_routed_pipeline_plan_for_agent_bar(
            str(rows[i].get("content", ""))
        ):
            continue
        prev = rows[i - 1] if i > 0 else None
        if _message_effective_chat_mode(rows[i], prev) != "agent":
            return False
        if _chat_row_has_user_execute_after(rows, i):
            return False
        return True
    return False


def _set_last_pipeline_plan(sid: str, plan: dict[str, Any]) -> None:
    if not isinstance(plan, dict) or not plan:
        return
    meta = _chat_meta.setdefault(sid, {"title": "New chat", "created_at": _now_iso(), "updated_at": _now_iso()})
    meta["last_pipeline_plan"] = plan
    meta["last_pipeline_plan_saved_at"] = _now_iso()
    _touch_chat_meta(sid)
    _save_chat_store()


class NoCacheSpaShellMiddleware:
    """Strip caches for the SPA shell and built assets (pure ASGI — avoids BaseHTTPMiddleware shutdown noise)."""

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "") or ""

        async def send_wrapper(message: Any) -> None:
            if message["type"] == "http.response.start" and (
                path in ("/", "/index.html")
                or path.startswith("/assets/")
            ):
                headers = MutableHeaders(scope=message)
                headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
                headers["Pragma"] = "no-cache"
            await send(message)

        await self.app(scope, receive, send_wrapper)


def _session(sid: str) -> PipelineSession:
    _ensure_chat_store_loaded()
    if sid not in _sessions:
        _sessions[sid] = PipelineSession()
    return _sessions[sid]


def _queue(sid: str) -> ApprovalQueue:
    _ensure_chat_store_loaded()
    if sid not in _queues:
        _queues[sid] = ApprovalQueue()
    return _queues[sid]


def _sid_turn_lock(sid: str) -> RLock:
    _ensure_chat_store_loaded()
    if sid not in _sid_turn_locks:
        _sid_turn_locks[sid] = RLock()
    return _sid_turn_locks[sid]


def _sid_turn_cancel_event(sid: str) -> Event:
    _ensure_chat_store_loaded()
    if sid not in _sid_turn_cancel_events:
        _sid_turn_cancel_events[sid] = Event()
    return _sid_turn_cancel_events[sid]


def _is_chat_turn_in_progress(sid: str) -> bool:
    _ensure_chat_store_loaded()
    meta = _chat_meta.get(sid) or {}
    return bool(meta.get("chat_turn_in_progress"))


def _mark_chat_turn_start(
    sid: str,
    message: str,
    chat_mode: str,
    pipeline_plan: Optional[dict[str, Any]],
) -> None:
    """In-memory only (not written to ``.mito2_chats.json``); survives browser refresh while the server stays up."""
    meta = _chat_meta.setdefault(sid, {"title": "New chat", "created_at": _now_iso(), "updated_at": _now_iso()})
    meta["chat_turn_in_progress"] = True
    meta["chat_turn_pending_user"] = message
    meta["chat_turn_pending_mode"] = chat_mode
    # Wall time for UI stepper rehydration after refresh (seconds since epoch, JSON float).
    meta["chat_turn_started_at"] = time.time()
    if isinstance(pipeline_plan, dict) and pipeline_plan:
        meta["chat_turn_pending_pipeline_plan"] = pipeline_plan
    else:
        meta.pop("chat_turn_pending_pipeline_plan", None)
    _sid_turn_cancel_event(sid).clear()


def _mark_chat_turn_end(sid: str) -> None:
    meta = _chat_meta.get(sid)
    if not meta:
        return
    meta["chat_turn_in_progress"] = False
    meta.pop("chat_turn_pending_user", None)
    meta.pop("chat_turn_pending_mode", None)
    meta.pop("chat_turn_pending_pipeline_plan", None)
    meta.pop("chat_turn_started_at", None)
    _sid_turn_cancel_event(sid).clear()


def _pending_payload(a: PendingAction) -> dict[str, Any]:
    return {
        "id": a.id,
        "title": a.title,
        "command": a.command,
        "cwd": a.cwd,
        "created_at": a.created_at,
        "status": a.status.value,
        "detail": a.detail,
    }


def _append_chat_message(
    session_id: str,
    role: str,
    content: str,
    *,
    chat_mode: Optional[str] = None,
) -> None:
    _ensure_chat_store_loaded()
    sid = (session_id or "default").strip() or "default"
    msg: dict[str, Any] = {"role": role, "content": content}
    cm = (chat_mode or "").strip().lower()
    if cm in ("ask", "plan", "agent"):
        msg["chat_mode"] = cm
    rows = _chat_history.setdefault(sid, [])
    rows.append(msg)
    if len(rows) > 200:
        del rows[:-200]
    _touch_chat_meta(sid)
    _save_chat_store()


app = FastAPI(title="mitoFoundation2", version="0.1.0")

configure_studio(get_project_root=get_project_root, get_session=_session)
configure_catalog(get_project_root=get_project_root)
app.include_router(studio_router)
app.include_router(catalog_router)


@app.exception_handler(RequestValidationError)
async def _log_request_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Echo 422 details to the server terminal (stderr) for easier debugging."""
    detail = exc.errors()
    print(
        f"\n[mitoFoundation2] 422 validation error  {request.method} {request.url.path}\n{detail}\n",
        file=sys.stderr,
        flush=True,
    )
    return JSONResponse(status_code=422, content={"detail": detail})


@app.exception_handler(Exception)
async def _log_unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    """Print tracebacks to the terminal; HTTPException is handled by FastAPI first (not routed here)."""
    print(
        f"\n[mitoFoundation2] unhandled exception {request.method} {request.url.path}\n{exc!r}\n",
        file=sys.stderr,
        flush=True,
    )
    traceback.print_exc(file=sys.stderr)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "path": request.url.path},
    )


@app.on_event("startup")
def _repair_codex_llm_json_on_startup() -> None:
    """Fix invalid ``openai_model: codex`` and noisy discovery lists left by older builds."""
    try:
        apply_codex_model_storage_repairs(get_project_root())
    except Exception:
        pass
    # Keep launch fast: only preload chats when explicitly requested.
    if os.getenv("MITO2_PRELOAD_CHAT_STORE", "0") == "1":
        _ensure_chat_store_loaded()


app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("MITO2_CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(NoCacheSpaShellMiddleware)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "project_root": str(get_project_root())}


@app.get("/api/llm/status")
def llm_status() -> dict[str, Any]:
    """Resolved LLM endpoints from env (does not call the model)."""
    base, key, model = get_llm_config()
    acct = get_chatgpt_account_id()
    return {
        "base_url": base,
        "model": model,
        "transport": get_llm_transport(),
        "api_key_configured": bool(key),
        "chatgpt_account_id_configured": bool(acct),
        "hint_markdown": setup_hint(),
    }


@app.get("/api/llm/settings")
def get_llm_settings() -> dict[str, Any]:
    """Saved UI settings + effective resolution (env overrides file)."""
    root = get_project_root()
    stored = load_stored_llm_settings(root)
    base, key, model = get_llm_config()
    return {
        "saved": public_settings_view(stored),
        "effective": {
            "base_url": base,
            "model": model,
            "transport": get_llm_transport(),
            "api_key_configured": bool(key),
            "chatgpt_account_id_configured": bool(get_chatgpt_account_id()),
        },
        "env_overrides": env_override_keys(),
    }


@app.get("/api/llm/codex-cli")
def codex_cli_status() -> dict[str, Any]:
    """Whether the Codex CLI is on PATH (used for ``codex login`` before this app can reuse auth)."""
    p = shutil.which("codex")
    return {"installed": bool(p), "path": p or ""}


@app.post("/api/llm/codex-login-browser")
def codex_login_browser(
    wait_seconds: float = Query(
        90.0,
        ge=20.0,
        le=120.0,
        description="How long to wait for the Codex CLI to print a sign-in URL (one HTTP round-trip; no client polling).",
    ),
) -> dict[str, Any]:
    """
    Run ``codex login`` and **block** until a sign-in URL appears or ``wait_seconds`` elapses.

    Embedded browsers (e.g. Cursor Simple Browser) often fail to run follow-up poll requests; returning
    ``auth_url`` in this response avoids that entirely.
    """
    try:
        generation = start_codex_login_background()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    deadline = time.monotonic() + wait_seconds
    auth_url: str | None = None
    device_code: str | None = None
    stale = False
    while time.monotonic() < deadline:
        st = get_codex_login_status(generation)
        if st.get("stale"):
            stale = True
            break
        raw = st.get("auth_url")
        if isinstance(raw, str) and raw.strip():
            auth_url = raw.strip()
        code = st.get("device_code")
        if isinstance(code, str) and code.strip():
            device_code = code.strip().upper()
        if auth_url and device_code:
            break
        time.sleep(0.2)
    if auth_url:
        message = "Sign-in link is ready — your browser tab should open it, or use the link in the app."
        if device_code:
            message += " Enter the device code shown below if prompted."
    elif stale:
        message = "A newer Sign in was started; try Sign in once more."
    else:
        message = (
            f"No sign-in URL from the Codex CLI within {int(wait_seconds)}s. "
            "On the same machine, run `codex login` in a normal terminal, or update the Codex CLI."
        )
    return {
        "ok": True,
        "generation": generation,
        "auth_url": auth_url,
        "device_code": device_code,
        "stale": stale,
        "message": message,
    }


@app.get("/api/llm/codex-login-status")
def codex_login_status(generation: int = Query(..., ge=1)) -> dict[str, Any]:
    """First captured sign-in URL for the login session started with ``codex-login-browser``."""
    return get_codex_login_status(generation)


@app.post("/api/llm/codex-logout")
def codex_logout(auth_json_path: str = Query("", description="Optional path to auth.json; parent dir used as CODEX_HOME.")) -> dict[str, Any]:
    """Run ``codex logout`` on the server machine (clears CLI session files for that Codex home)."""
    codex_bin = shutil.which("codex")
    if not codex_bin:
        raise HTTPException(status_code=400, detail="Codex CLI not found on PATH.")
    env = os.environ.copy()
    path_str = auth_json_path.strip()
    if not path_str:
        st = load_stored_llm_settings(get_project_root())
        path_str = (st.get("codex_auth_json_path") or "").strip()
    if path_str:
        p = Path(path_str).expanduser().resolve()
        if p.is_file():
            env["CODEX_HOME"] = str(p.parent)
        elif p.is_dir():
            env["CODEX_HOME"] = str(p)
    try:
        r = subprocess.run(
            [codex_bin, "logout"],
            capture_output=True,
            text=True,
            timeout=90,
            env=env,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="codex logout timed out.") from None
    out = ((r.stdout or "") + "\n" + (r.stderr or "")).strip()
    if r.returncode != 0:
        raise HTTPException(
            status_code=400,
            detail=out or f"codex logout exited with code {r.returncode}",
        )
    return {
        "ok": True,
        "message": out or "Signed out of Codex on this machine.",
    }


@app.get("/api/llm/codex-profiles")
def get_codex_profiles(path: str = "") -> dict[str, Any]:
    """List saved Codex CLI sessions from ``auth.json`` (default ``~/.codex/auth.json``)."""
    p = Path(path).expanduser().resolve() if path.strip() else default_codex_auth_path()
    rows = list_codex_profiles(p)
    return {
        "auth_path": str(p),
        "profiles": [
            {
                "id": r.id,
                "label": r.label,
                "account_id_preview": r.account_id_preview,
                "has_access_token": r.has_access_token,
                "detail": r.detail,
            }
            for r in rows
        ],
    }


@app.post("/api/llm/settings")
def post_llm_settings(update: LlmSettingsUpdate) -> dict[str, Any]:
    root = get_project_root()
    patch = _normalize_llm_settings_patch(update.model_dump(exclude_none=True))
    merge_patch_stored(root, patch)
    apply_codex_model_storage_repairs(root)
    stored = load_stored_llm_settings(root)
    base, key, model = get_llm_config()
    return {
        "saved": public_settings_view(stored),
        "effective": {
            "base_url": base,
            "model": model,
            "transport": get_llm_transport(),
            "api_key_configured": bool(key),
            "chatgpt_account_id_configured": bool(get_chatgpt_account_id()),
        },
        "env_overrides": env_override_keys(),
    }


@app.get("/api/pipeline")
def get_pipeline(session_id: str = "default") -> dict[str, Any]:
    sid = (session_id or "default").strip() or "default"
    s = _session(sid)
    out: dict[str, Any] = {
        "current_step": int(s.current_step),
        "step_label": STEP_LABELS.get(int(s.current_step), "idle"),
        "last_url": s.last_url,
        "last_site_stem": s.last_site_stem,
    }
    # Expose last approved/saved plan ``n_crops`` so Studio can mirror downloader UI without guessing.
    meta = _chat_meta.get(sid) or {}
    lp = meta.get("last_pipeline_plan")
    if isinstance(lp, dict):
        nc = lp.get("n_crops")
        try:
            if nc is not None:
                out["plan_n_crops"] = max(1, min(64, int(nc)))
        except (TypeError, ValueError):
            pass
        dl = lp.get("download")
        if isinstance(dl, dict):
            try:
                tr = dl.get("training_crops")
                if tr is not None:
                    out["plan_n_crops_training"] = max(0, min(64, int(tr)))
            except (TypeError, ValueError):
                pass
            try:
                inf = dl.get("inference_crops")
                if inf is not None:
                    out["plan_n_crops_inference"] = max(0, min(64, int(inf)))
            except (TypeError, ValueError):
                pass
    return out


@app.get("/api/chats")
def list_chats(query: str = "") -> dict[str, Any]:
    _ensure_chat_store_loaded()
    q = (query or "").strip().lower()
    rows: list[dict[str, Any]] = []
    for sid, messages in _chat_history.items():
        meta = _chat_meta.get(sid) or {}
        title = str(meta.get("title") or _chat_title_from_first_user(messages))
        preview = _chat_preview(messages)
        hay = f"{title}\n{preview}".lower()
        if q and q not in hay:
            continue
        rows.append(
            {
                "id": sid,
                "title": title,
                "created_at": str(meta.get("created_at") or ""),
                "updated_at": str(meta.get("updated_at") or ""),
                "message_count": len(messages),
                "preview": preview,
            }
        )
    rows.sort(key=lambda r: str(r.get("updated_at", "")), reverse=True)
    return {"threads": rows}


@app.post("/api/chats/new")
def create_chat() -> dict[str, Any]:
    sid = f"chat_{int(time.time() * 1000)}"
    _ensure_chat_exists(sid)
    return {"id": sid}


@app.post("/api/chats/delete")
def delete_chats(req: ChatDeleteRequest) -> dict[str, Any]:
    removed: list[str] = []
    for sid_raw in req.session_ids:
        sid = (sid_raw or "").strip()
        if not sid:
            continue
        q = _queue(sid)
        pending = [a for a in q.pending_list() if a.status == ApprovalStatus.PENDING]
        if pending:
            continue
        _sessions.pop(sid, None)
        _queues.pop(sid, None)
        _chat_history.pop(sid, None)
        _chat_meta.pop(sid, None)
        _sid_turn_cancel_events.pop(sid, None)
        removed.append(sid)
    _save_chat_store()
    return {"ok": True, "deleted": removed}


@app.post("/api/chat")
def post_chat(req: ChatRequest) -> Any:
    root = get_project_root()
    sid = (req.session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id is required")
    _ensure_chat_exists(sid)
    session = _session(sid)
    queue = _queue(sid)

    hist_dicts = _chat_history.get(sid, [])
    if not hist_dicts and req.history:
        boot: List[dict[str, Any]] = []
        for m in req.history[-200:]:
            row: dict[str, Any] = {"role": m.role, "content": m.content}
            cm0 = (m.chat_mode or "").strip().lower() if getattr(m, "chat_mode", None) else ""
            if cm0 in ("ask", "plan", "agent"):
                row["chat_mode"] = cm0
            boot.append(row)
        hist_dicts = boot
        _chat_history[sid] = hist_dicts
    eff_pipeline_plan = req.pipeline_plan
    # Reliability fallback: execute approved plan even if client lost payload across refresh/edit.
    if (
        (req.pipeline_action or "").strip().lower() == "execute_plan"
        and not (isinstance(eff_pipeline_plan, dict) and eff_pipeline_plan)
    ):
        eff_pipeline_plan = _get_last_pipeline_plan(sid)
        if not (isinstance(eff_pipeline_plan, dict) and eff_pipeline_plan):
            eff_pipeline_plan = _infer_last_pipeline_plan_from_history(sid)
            if isinstance(eff_pipeline_plan, dict) and eff_pipeline_plan:
                _set_last_pipeline_plan(sid, eff_pipeline_plan)

    cm0 = (req.chat_mode or "ask").strip().lower()
    if cm0 not in ("ask", "plan", "agent"):
        cm0 = "ask"
    plan_snap: Optional[dict[str, Any]] = eff_pipeline_plan if isinstance(eff_pipeline_plan, dict) and eff_pipeline_plan else None

    if req.stream_planning:
        with _sid_turn_lock(sid):
            if _is_chat_turn_in_progress(sid):
                raise HTTPException(
                    status_code=409,
                    detail="A chat turn is already running for this session. Wait for it to finish, then refresh to see results.",
                )
            _mark_chat_turn_start(sid, req.message, cm0, plan_snap)

        qhints = Queue()
        result_box: list[Any] = [None]
        err_box: list[BaseException | None] = [None]

        def work() -> None:
            try:
                result_box[0] = run_chat_turn(
                    root,
                    session,
                    queue,
                    req.message,
                    hist_dicts,
                    chat_session_id=sid,
                    chat_mode=req.chat_mode,
                    pipeline_action=req.pipeline_action,
                    pipeline_plan=eff_pipeline_plan,
                    on_planning_hint=qhints.put,
                    stop_requested=lambda sid=sid: _sid_turn_cancel_event(sid).is_set(),
                )
            except BaseException as e:  # noqa: BLE001 — must propagate to HTTP layer
                err_box[0] = e

        th = Thread(target=work, daemon=True)
        try:
            th.start()
        except BaseException:
            _mark_chat_turn_end(sid)
            raise

        def ndjson() -> Any:
            try:
                while th.is_alive() or not qhints.empty():
                    try:
                        payload = qhints.get(timeout=0.12)
                    except Empty:
                        continue
                    yield json.dumps({"event": "planning", "data": payload}, ensure_ascii=False) + "\n"
                th.join()
                if err_box[0] is not None:
                    err = err_box[0]
                    detail = ""
                    if isinstance(err, HTTPException):
                        detail = str(err.detail or "").strip()
                    if not detail:
                        detail = str(err).strip()
                    if not detail:
                        detail = "Chat turn failed."
                    # Streaming response already started, so we must finish with a
                    # terminal NDJSON event instead of raising into ASGI.
                    body = _post_chat_result_dict(
                        sid,
                        session,
                        queue,
                        f"Error: {detail}",
                        {"execution": {"ok": False, "error": detail}},
                        cm0,
                    )
                    yield json.dumps({"event": "done", "data": body}, ensure_ascii=False) + "\n"
                    return
                row = result_box[0]
                if row is None:
                    body = _post_chat_result_dict(
                        sid,
                        session,
                        queue,
                        "Error: Chat turn produced no result.",
                        {"execution": {"ok": False, "error": "Chat turn produced no result"}},
                        cm0,
                    )
                    yield json.dumps({"event": "done", "data": body}, ensure_ascii=False) + "\n"
                    return
                reply, _, _, extra = row
                drafted = extra.get("draft_pipeline_plan")
                if isinstance(drafted, dict) and drafted:
                    _set_last_pipeline_plan(sid, drafted)
                elif isinstance(eff_pipeline_plan, dict) and eff_pipeline_plan:
                    _set_last_pipeline_plan(sid, eff_pipeline_plan)
                _append_chat_message(sid, "user", req.message, chat_mode=cm0)
                _append_chat_message(sid, "assistant", reply, chat_mode=cm0)
                body = _post_chat_result_dict(sid, session, queue, reply, extra, cm0)
                yield json.dumps({"event": "done", "data": body}, ensure_ascii=False) + "\n"
            finally:
                _mark_chat_turn_end(sid)

        return StreamingResponse(ndjson(), media_type="application/x-ndjson")

    with _sid_turn_lock(sid):
        if _is_chat_turn_in_progress(sid):
            raise HTTPException(
                status_code=409,
                detail="A chat turn is already running for this session. Wait for it to finish, then refresh to see results.",
            )
        _mark_chat_turn_start(sid, req.message, cm0, plan_snap)
    try:
        reply, _, _, extra = run_chat_turn(
            root,
            session,
            queue,
            req.message,
            hist_dicts,
            chat_session_id=sid,
            chat_mode=req.chat_mode,
            pipeline_action=req.pipeline_action,
            pipeline_plan=eff_pipeline_plan,
            stop_requested=lambda sid=sid: _sid_turn_cancel_event(sid).is_set(),
        )
        drafted = extra.get("draft_pipeline_plan")
        if isinstance(drafted, dict) and drafted:
            _set_last_pipeline_plan(sid, drafted)
        elif isinstance(eff_pipeline_plan, dict) and eff_pipeline_plan:
            # Keep latest executable plan for robust Approve & run retries.
            _set_last_pipeline_plan(sid, eff_pipeline_plan)
        _append_chat_message(sid, "user", req.message, chat_mode=cm0)
        _append_chat_message(sid, "assistant", reply, chat_mode=cm0)

        return ChatResponse(**_post_chat_result_dict(sid, session, queue, reply, extra, cm0))
    finally:
        _mark_chat_turn_end(sid)


@app.post("/api/chat/stop")
def stop_chat_turn(req: ChatStopRequest) -> dict[str, Any]:
    sid = (req.session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id is required")
    _ensure_chat_exists(sid)
    cancel_ev = _sid_turn_cancel_event(sid)
    cancel_ev.set()

    # Best-effort immediate kills for long-running subprocess-backed stages.
    try:
        studio_scrape_cancel(StudioScrapeCancelBody(session_id=sid))
    except Exception:
        pass
    try:
        studio_run_downloader_script_cancel(StudioDownloaderCancelBody(session_id=sid))
    except Exception:
        pass

    return {
        "ok": True,
        "stop_requested": True,
        "chat_turn_in_progress": _is_chat_turn_in_progress(sid),
    }


@app.post("/api/chat/edit")
def edit_chat_turn(req: ChatEditRequest) -> dict[str, Any]:
    root = get_project_root()
    sid = (req.session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id is required")
    _ensure_chat_exists(sid)
    session = _session(sid)
    q = _queue(sid)
    pending = [a for a in q.pending_list() if a.status == ApprovalStatus.PENDING]
    if pending:
        raise HTTPException(status_code=409, detail="Cannot edit chat while approvals are pending")

    rows = list(_chat_history.get(sid, []))
    idx = int(req.message_index)
    if idx < 0 or idx >= len(rows):
        raise HTTPException(status_code=400, detail="message_index is out of range")
    if rows[idx].get("role") != "user":
        raise HTTPException(status_code=400, detail="Only user messages can be edited")

    new_message = (req.message or "").strip()
    if not new_message:
        raise HTTPException(status_code=400, detail="Edited message cannot be empty")

    history_before = rows[:idx]
    cm = (req.chat_mode or "ask").strip().lower()
    if cm not in ("ask", "plan", "agent"):
        cm = "ask"
    with _sid_turn_lock(sid):
        if _is_chat_turn_in_progress(sid):
            raise HTTPException(
                status_code=409,
                detail="A chat turn is already running for this session. Wait for it to finish, then refresh to see results.",
            )
        _mark_chat_turn_start(sid, new_message, cm, None)
    try:
        reply, _, _, extra = run_chat_turn(
            root,
            session,
            q,
            new_message,
            history_before,
            chat_session_id=sid,
            chat_mode=cm,
            pipeline_action=None,
            pipeline_plan=None,
            stop_requested=lambda sid=sid: _sid_turn_cancel_event(sid).is_set(),
        )
        drafted = extra.get("draft_pipeline_plan")
        if isinstance(drafted, dict) and drafted:
            _set_last_pipeline_plan(sid, drafted)
        new_rows = history_before + [
            {"role": "user", "content": new_message, "chat_mode": cm},
            {"role": "assistant", "content": reply, "chat_mode": cm},
        ]
        _chat_history[sid] = new_rows[-200:]
        _touch_chat_meta(sid)
        _save_chat_store()

        fresh_pending = [a for a in q.pending_list() if a.status == ApprovalStatus.PENDING]
        return {
            "reply": reply,
            "messages": _chat_history.get(sid, []),
            "pending_approvals": [_pending_payload(a) for a in fresh_pending],
            "pipeline": {
                "current_step": int(session.current_step),
                "step_label": STEP_LABELS.get(int(session.current_step), "idle"),
                "last_url": session.last_url,
                "last_site_stem": session.last_site_stem,
            },
            "draft_pipeline_plan": extra.get("draft_pipeline_plan"),
            "execution": extra.get("execution"),
            "chat_mode": cm,
        }
    finally:
        _mark_chat_turn_end(sid)


@app.post("/api/chat/delete-turn")
def delete_chat_turn(req: ChatDeleteTurnRequest) -> dict[str, Any]:
    sid = (req.session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id is required")
    _ensure_chat_exists(sid)
    session = _session(sid)
    q = _queue(sid)
    pending = [a for a in q.pending_list() if a.status == ApprovalStatus.PENDING]
    if pending:
        raise HTTPException(status_code=409, detail="Cannot delete chat turns while approvals are pending")

    rows = list(_chat_history.get(sid, []))
    idx = int(req.message_index)
    if idx < 0 or idx >= len(rows):
        raise HTTPException(status_code=400, detail="message_index is out of range")
    if rows[idx].get("role") != "user":
        raise HTTPException(status_code=400, detail="Only user messages can be deleted")

    # Delete from selected user turn onward (same branching semantics as edit/regenerate).
    _chat_history[sid] = rows[:idx][-200:]
    _touch_chat_meta(sid)
    _save_chat_store()

    fresh_pending = [a for a in q.pending_list() if a.status == ApprovalStatus.PENDING]
    return {
        "ok": True,
        "messages": _chat_history.get(sid, []),
        "pending_approvals": [_pending_payload(a) for a in fresh_pending],
        "pipeline": {
            "current_step": int(session.current_step),
            "step_label": STEP_LABELS.get(int(session.current_step), "idle"),
            "last_url": session.last_url,
            "last_site_stem": session.last_site_stem,
        },
    }


@app.get("/api/chat/state")
def chat_state(session_id: str) -> dict[str, Any]:
    sid = (session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id is required")
    _ensure_chat_exists(sid)
    s = _session(sid)
    q = _queue(sid)
    pending = [a for a in q.pending_list() if a.status == ApprovalStatus.PENDING]
    meta = _chat_meta.get(sid) or {}
    lp0 = _get_last_pipeline_plan(sid)
    awaiting = _chat_agent_plan_awaits_approval(sid)
    lp = lp0
    if awaiting and not lp0:
        lp = _infer_last_pipeline_plan_from_history(sid)
    pend_plan = meta.get("chat_turn_pending_pipeline_plan")
    if not isinstance(pend_plan, dict) or not pend_plan:
        pend_plan = None
    return {
        "thread": {
            "id": sid,
            "title": str(meta.get("title") or _chat_title_from_first_user(_chat_history.get(sid, []))),
            "created_at": str(meta.get("created_at") or ""),
            "updated_at": str(meta.get("updated_at") or ""),
        },
        "messages": _chat_history.get(sid, []),
        "pending_approvals": [_pending_payload(a) for a in pending],
        "pipeline": {
            "current_step": int(s.current_step),
            "step_label": STEP_LABELS.get(int(s.current_step), "idle"),
            "last_url": s.last_url,
            "last_site_stem": s.last_site_stem,
        },
        "chat_turn_in_progress": bool(meta.get("chat_turn_in_progress")),
        "chat_turn_pending_user": str(meta.get("chat_turn_pending_user") or ""),
        "chat_turn_pending_mode": str(meta.get("chat_turn_pending_mode") or ""),
        "chat_turn_pending_pipeline_plan": pend_plan,
        "chat_turn_started_at": meta.get("chat_turn_started_at"),
        "last_pipeline_plan": lp,
        "chat_agent_plan_awaiting_approval": awaiting,
    }


@app.get("/api/chat/plans")
def list_pipeline_plans() -> dict[str, Any]:
    _ensure_chat_store_loaded()
    return {"plans": list(_saved_pipeline_plans)}


@app.post("/api/chat/plans")
def save_pipeline_plan(body: PipelinePlanSaveBody) -> dict[str, Any]:
    row = {
        "id": f"plan_{int(time.time() * 1000)}",
        "created_at": _now_iso(),
        "title": (body.title or "").strip() or "Saved plan",
        "plan": dict(body.plan or {}),
    }
    _saved_pipeline_plans.insert(0, row)
    del _saved_pipeline_plans[200:]
    _save_chat_store()
    return {"ok": True, "entry": row}


@app.delete("/api/chat/plans/{plan_id}")
def remove_pipeline_plan(plan_id: str) -> dict[str, Any]:
    pid = (plan_id or "").strip()
    if not pid:
        raise HTTPException(status_code=404, detail="Unknown plan id")
    n0 = len(_saved_pipeline_plans)
    _saved_pipeline_plans[:] = [x for x in _saved_pipeline_plans if str(x.get("id") or "") != pid]
    if len(_saved_pipeline_plans) == n0:
        raise HTTPException(status_code=404, detail="Unknown plan id")
    _save_chat_store()
    return {"ok": True}


@app.api_route(
    "/api/chat/clear",
    methods=["GET", "POST", "DELETE", "PUT", "PATCH", "OPTIONS"],
    include_in_schema=False,
)
@app.api_route(
    "/api/chat/clear/",
    methods=["GET", "POST", "DELETE", "PUT", "PATCH", "OPTIONS"],
    include_in_schema=False,
)
def clear_chat_any(req: ChatClearRequest | None = None, session_id: str = Query("default")) -> dict[str, Any]:
    sid = ((req.session_id if req and req.session_id else session_id) or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id is required")
    return _clear_chat_for_session(sid)


def _clear_chat_for_session(sid: str) -> dict[str, Any]:
    q = _queue(sid)
    pending = [a for a in q.pending_list() if a.status == ApprovalStatus.PENDING]
    if pending:
        raise HTTPException(status_code=409, detail="Cannot clear chat while approvals are pending")
    _sessions[sid] = PipelineSession()
    _queues[sid] = ApprovalQueue()
    _chat_history[sid] = []
    now = _now_iso()
    _chat_meta[sid] = {"title": "New chat", "created_at": now, "updated_at": now}
    _sid_turn_cancel_event(sid).clear()
    _save_chat_store()
    s = _sessions[sid]
    return {
        "ok": True,
        "pending_approvals": [],
        "pipeline": {
            "current_step": int(s.current_step),
            "step_label": STEP_LABELS.get(int(s.current_step), "idle"),
            "last_url": s.last_url,
            "last_site_stem": s.last_site_stem,
        },
    }


@app.get("/api/approvals")
def list_approvals(session_id: str = "default") -> dict[str, Any]:
    q = _queue(session_id)
    pending = [a for a in q.pending_list() if a.status == ApprovalStatus.PENDING]
    return {"pending": [_pending_payload(a) for a in pending]}


@app.post("/api/approvals/{action_id}")
def resolve_approval(action_id: str, body: ApprovalResolve, session_id: str = "default") -> dict[str, Any]:
    root = get_project_root()
    q = _queue(session_id)
    session = _session(session_id)
    action = q.resolve(action_id, body.approved)
    if not action:
        raise HTTPException(status_code=404, detail="approval not found or already resolved")
    if not body.approved:
        _append_chat_message(session_id, "assistant", "Declined.")
        return {"status": "rejected", "action": _pending_payload(action)}

    result = run_command(action.command, Path(action.cwd))
    detail = action.detail or {}
    if detail.get("step") == "scrape" and detail.get("url"):
        update_session_after_scrape(root, session, str(detail["url"]))
    elif detail.get("step") in ("database", "schema"):
        session.current_step = PipelineStep.DATABASE
    elif detail.get("step") in ("download_script", "download_execute"):
        session.current_step = PipelineStep.DOWNLOAD_SCRIPT
    elif detail.get("step") == "preprocess":
        session.current_step = PipelineStep.PREPROCESS
    elif detail.get("step") == "training":
        session.current_step = PipelineStep.SSL
    elif detail.get("step") == "finetune":
        session.current_step = PipelineStep.FINETUNE
    elif detail.get("step") == "eval":
        session.current_step = PipelineStep.EVAL

    session.log("executed", {"action_id": action.id, "returncode": result["returncode"]})
    note = f"Approved and executed.\n\n[exit {result['returncode']}]\n{(result.get('stderr') or result.get('stdout') or '')}".strip()
    _append_chat_message(session_id, "assistant", note[:4000])
    return {"status": "approved", "action": _pending_payload(action), "result": result}


@app.get("/api/prompts")
def get_prompts() -> dict[str, str]:
    root = get_project_root()
    b = load_prompts(root)
    return {"system": b.system, "user_prefix": b.user_prefix}


@app.post("/api/prompts")
def post_prompts(up: PromptsUpdate) -> dict[str, str]:
    root = get_project_root()
    b = load_prompts(root)
    if up.system is not None:
        b.system = up.system
    if up.user_prefix is not None:
        b.user_prefix = up.user_prefix
    save_prompts(root, b)
    return {"system": b.system, "user_prefix": b.user_prefix}


@app.get("/api/skills")
def get_skills() -> dict[str, Any]:
    """Legacy orchestration-centric listing; prefer ``GET /api/agent-skills``."""
    root = get_project_root()
    ensure_skill_trees(root)
    skills = load_skills(root)
    merged = merged_skill_bodies(root)
    orch = list_orchestration_skills(root)
    chat = list_chat_skills(root)
    return {
        "skills": [{"id": s.id, "title": s.title, "body": s.body} for s in skills],
        "merged_preview": merged[:8000],
        "orchestration": [
            {"kind": "orchestration", "slug": r.slug, "id": r.id, "title": r.title, "label": r.label} for r in orch
        ],
        "chat": [{"kind": "chat", "slug": r.slug, "id": r.id, "title": r.title, "label": r.label} for r in chat],
    }


@app.get("/api/agent-skills")
def get_agent_skills_index() -> dict[str, Any]:
    root = get_project_root()
    ensure_skill_trees(root)

    def rows(recs: list[SkillRecord]) -> list[dict[str, str]]:
        return [{"kind": r.kind, "slug": r.slug, "id": r.id, "title": r.title, "label": r.label} for r in recs]

    return {
        "chat": rows(list_chat_skills(root)),
        "orchestration": rows(list_orchestration_skills(root)),
    }


@app.post("/api/agent-skills/{kind}")
def post_agent_skill_create(kind: str, payload: CreateAgentSkillBody) -> dict[str, str]:
    if kind not in ("chat", "orchestration"):
        raise HTTPException(status_code=400, detail="kind must be chat or orchestration")
    root = get_project_root()
    slug = payload.slug.strip()
    try:
        create_skill(
            root,
            kind,  # type: ignore[arg-type]
            slug,
            label=payload.label.strip(),
            title=(payload.title or "").strip() or None,
            id=(payload.id or "").strip() or None,
            body=payload.body,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=f"Skill already exists: {e}") from e
    return {"kind": kind, "slug": slug, "status": "created"}


@app.patch("/api/agent-skills/{kind}/{slug}")
def patch_agent_skill_rename(kind: str, slug: str, payload: RenameAgentSkillBody) -> dict[str, str]:
    if kind not in ("chat", "orchestration"):
        raise HTTPException(status_code=400, detail="kind must be chat or orchestration")
    root = get_project_root()
    new_slug = payload.new_slug.strip()
    try:
        rename_skill_slug(root, kind, slug, new_slug)  # type: ignore[arg-type]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return {"kind": kind, "slug": new_slug, "status": "renamed"}


@app.get("/api/agent-skills/{kind}/{slug}")
def get_agent_skill_document(kind: str, slug: str) -> dict[str, str]:
    if kind not in ("chat", "orchestration"):
        raise HTTPException(status_code=400, detail="kind must be chat or orchestration")
    root = get_project_root()
    try:
        doc = read_skill_document(root, kind, slug)  # type: ignore[arg-type]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return {"kind": kind, "slug": slug, "document": doc}


@app.put("/api/agent-skills/{kind}/{slug}")
def put_agent_skill_document(kind: str, slug: str, payload: AgentSkillDocumentUpdate) -> dict[str, str]:
    if kind not in ("chat", "orchestration"):
        raise HTTPException(status_code=400, detail="kind must be chat or orchestration")
    root = get_project_root()
    try:
        write_skill_document(root, kind, slug, payload.document)  # type: ignore[arg-type]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"kind": kind, "slug": slug, "status": "saved"}


@app.post("/api/skills/{skill_id}")
def post_skill(skill_id: str, body: SkillBodyUpdate) -> dict[str, str]:
    root = get_project_root()
    save_skill_override(root, skill_id, body.body)
    return {"id": skill_id, "status": "saved"}


_FRONTEND_DIST = _PROJECT_ROOT / "frontend" / "dist"
_FRONTEND_PUBLIC = _PROJECT_ROOT / "frontend" / "public"


@app.get("/favicon.ico", response_model=None)
def favicon_ico():
    """Browsers request ``/favicon.ico`` by default; avoid 404 when only SVG exists."""
    for path in (_FRONTEND_DIST / "favicon.svg", _FRONTEND_PUBLIC / "favicon.svg"):
        if path.is_file():
            return FileResponse(path, media_type="image/svg+xml")
    return Response(status_code=204)


# Static UI (production): ``cd frontend && npm ci && npm run build``
if _FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="ui")
else:

    @app.get("/")
    def root_placeholder() -> dict[str, str]:
        return {
            "message": "Build the frontend (frontend/dist missing). "
            "Dev: run Vite in frontend/ and proxy /api to this server.",
            "project_root": str(get_project_root()),
        }
