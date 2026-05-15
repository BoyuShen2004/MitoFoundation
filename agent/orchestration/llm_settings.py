"""Persisted LLM fields (OpenClaw-style UI config), merged with environment variables."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def llm_settings_path(project_root: Path) -> Path:
    d = project_root / "agent" / "orchestration" / "config"
    d.mkdir(parents=True, exist_ok=True)
    return d / "llm.json"


def load_stored_llm_settings(project_root: Path) -> dict[str, str]:
    path = llm_settings_path(project_root)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, str] = {}
    for k in (
        "openai_api_key",
        "openai_base_url",
        "openai_model",
        "chatgpt_account_id",
        "codex_auth_profile_id",
        "codex_auth_json_path",
        "llm_provider",
    ):
        v = data.get(k)
        if isinstance(v, str) and v.strip():
            out[k] = v.strip()
    caml = data.get("codex_agent_model_list")
    if isinstance(caml, str):
        out["codex_agent_model_list"] = caml
    return out


def save_stored_llm_settings(project_root: Path, data: dict[str, str]) -> None:
    path = llm_settings_path(project_root)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def merge_patch_stored(project_root: Path, patch: dict[str, Any]) -> dict[str, str]:
    cur = load_stored_llm_settings(project_root)
    nk = patch.get("openai_api_key")
    if isinstance(nk, str):
        s = nk.strip()
        if s:
            cur["openai_api_key"] = s
        else:
            cur.pop("openai_api_key", None)
    cat = patch.get("codex_agent_model_list")
    if cat is not None:
        if isinstance(cat, str) and cat.strip():
            cur["codex_agent_model_list"] = cat
        else:
            cur.pop("codex_agent_model_list", None)
    for fld in (
        "openai_base_url",
        "openai_model",
        "chatgpt_account_id",
        "codex_auth_profile_id",
        "codex_auth_json_path",
        "llm_provider",
    ):
        v = patch.get(fld)
        if v is None:
            continue
        if isinstance(v, str):
            s = v.strip()
            if s:
                cur[fld] = s
            else:
                cur.pop(fld, None)
    save_stored_llm_settings(project_root, cur)
    return cur


def public_settings_view(stored: dict[str, str]) -> dict[str, Any]:
    return {
        "openai_base_url": stored.get("openai_base_url", ""),
        "openai_model": stored.get("openai_model", ""),
        "openai_api_key": stored.get("openai_api_key", ""),
        "chatgpt_account_id": stored.get("chatgpt_account_id", ""),
        "codex_auth_profile_id": stored.get("codex_auth_profile_id", ""),
        "codex_auth_json_path": stored.get("codex_auth_json_path", ""),
        "llm_provider": stored.get("llm_provider", ""),
        "codex_agent_model_list": stored.get("codex_agent_model_list", ""),
    }


def env_override_keys() -> list[str]:
    out: list[str] = []
    if (os.getenv("OPENAI_API_KEY") or "").strip():
        out.append("OPENAI_API_KEY")
    if (os.getenv("OPENAI_BASE_URL") or "").strip():
        out.append("OPENAI_BASE_URL")
    if (os.getenv("OPENAI_MODEL") or "").strip():
        out.append("OPENAI_MODEL")
    if (os.getenv("CHATGPT_ACCOUNT_ID") or "").strip():
        out.append("CHATGPT_ACCOUNT_ID")
    if (os.getenv("MITO2_CODEX_AUTH_JSON") or "").strip():
        out.append("MITO2_CODEX_AUTH_JSON")
    if (os.getenv("MITO2_LLM_PROVIDER") or "").strip():
        out.append("MITO2_LLM_PROVIDER")
    if (os.getenv("MITO2_CODEX_TRANSPORT") or "").strip():
        out.append("MITO2_CODEX_TRANSPORT")
    return out
