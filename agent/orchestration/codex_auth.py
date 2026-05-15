"""Read OpenAI Codex CLI ``auth.json`` (single or multiple login records)."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class CodexProfile:
    """Selectable profile derived from ``auth.json``."""

    id: str
    label: str
    account_id_preview: str
    has_access_token: bool
    detail: str = ""


def default_codex_auth_path() -> Path:
    custom = (os.getenv("MITO2_CODEX_AUTH_JSON") or "").strip()
    if custom:
        return Path(custom).expanduser().resolve()
    return Path.home() / ".codex" / "auth.json"


def _b64url_json_segment(b64url: str) -> dict[str, Any]:
    s = b64url.strip()
    pad = "=" * (-len(s) % 4)
    try:
        raw = base64.urlsafe_b64decode(s + pad)
        return json.loads(raw.decode("utf-8"))
    except (ValueError, json.JSONDecodeError, OSError):
        return {}


def chatgpt_account_id_from_id_token(id_token: str) -> str:
    """Pull ``chatgpt_account_id`` from JWT payload (Codex ``id_token``)."""
    parts = id_token.split(".")
    if len(parts) < 2:
        return ""
    payload = _b64url_json_segment(parts[1])
    auth = payload.get("https://api.openai.com/auth")
    if isinstance(auth, dict):
        cid = auth.get("chatgpt_account_id")
        if isinstance(cid, str) and cid.strip():
            return cid.strip()
    return ""


def email_from_id_token(id_token: str) -> str:
    parts = id_token.split(".")
    if len(parts) < 2:
        return ""
    payload = _b64url_json_segment(parts[1])
    em = payload.get("email")
    return em.strip() if isinstance(em, str) and em.strip() else ""


def _detail_email_for_entry(obj: dict[str, Any]) -> str:
    if isinstance(obj.get("email"), str) and obj["email"].strip():
        return obj["email"].strip()
    tokens = obj.get("tokens") if isinstance(obj.get("tokens"), dict) else {}
    if isinstance(tokens, dict):
        id_tok = (tokens.get("id_token") or "").strip()
        if id_tok:
            em = email_from_id_token(id_tok)
            if em:
                return em
    return ""


def _neutral_label_for_profile(profile_id: str, account_id: str) -> str:
    """Primary UI label without leading with email (privacy + less confusion)."""
    aid = (account_id or "").strip()
    if aid:
        vis = f"{aid[:8]}…" if len(aid) > 8 else aid
        return f"Account · {vis}"
    return f"Session · {profile_id}"


def _iter_auth_entries(data: Any) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(profile_id, auth_object)`` for supported shapes."""
    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                yield (str(i), item)
        return
    if not isinstance(data, dict):
        return
    if isinstance(data.get("accounts"), list):
        for i, item in enumerate(data["accounts"]):
            if isinstance(item, dict):
                yield (str(i), item)
        return
    profiles = data.get("profiles")
    if isinstance(profiles, dict):
        for key, item in profiles.items():
            if isinstance(item, dict) and key.strip():
                yield (key.strip(), item)
        return
    # Single login object (standard ``~/.codex/auth.json``)
    if data.get("tokens") is not None or data.get("auth_mode"):
        yield ("default", data)


def list_codex_profiles(auth_path: Path) -> list[CodexProfile]:
    if not auth_path.is_file():
        return []
    try:
        data = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    out: list[CodexProfile] = []
    for pid, obj in _iter_auth_entries(data):
        tokens = obj.get("tokens") if isinstance(obj.get("tokens"), dict) else {}
        access = (tokens.get("access_token") or "") if isinstance(tokens, dict) else ""
        id_tok = (tokens.get("id_token") or "") if isinstance(tokens, dict) else ""
        acct = (tokens.get("account_id") or "") if isinstance(tokens, dict) else ""
        acct_s = acct.strip() if isinstance(acct, str) else ""
        if acct_s:
            preview = f"{acct_s[:8]}…" if len(acct_s) > 8 else acct_s
        else:
            preview = "(from token)"
        label = _neutral_label_for_profile(pid, acct_s)
        detail = _detail_email_for_entry(obj)
        out.append(
            CodexProfile(
                id=pid,
                label=label,
                account_id_preview=preview,
                has_access_token=bool(
                    (isinstance(access, str) and access.strip())
                    or (isinstance(id_tok, str) and id_tok.strip())
                ),
                detail=detail,
            )
        )
    return out


def resolve_codex_credentials(auth_path: Path, profile_id: str) -> tuple[str, str]:
    """
    Returns ``(bearer_access_token, chatgpt_account_id)``.

    Prefers ``tokens.access_token``; falls back to ``id_token`` if needed.
    Account id from ``tokens.account_id`` or JWT ``chatgpt_account_id`` claim.
    """
    if not auth_path.is_file():
        raise FileNotFoundError(str(auth_path))
    data = json.loads(auth_path.read_text(encoding="utf-8"))
    target: dict[str, Any] | None = None
    for pid, obj in _iter_auth_entries(data):
        if pid == profile_id:
            target = obj
            break
    if target is None:
        raise KeyError(f"No profile {profile_id!r} in {auth_path}")

    tokens = target.get("tokens") if isinstance(target.get("tokens"), dict) else {}
    if not isinstance(tokens, dict):
        raise ValueError("Invalid tokens object")

    access = (tokens.get("access_token") or "").strip()
    id_tok = (tokens.get("id_token") or "").strip()
    bearer = access or id_tok
    if not bearer:
        raise ValueError("No access_token or id_token in selected profile")

    acct = (tokens.get("account_id") or "").strip()
    if not acct and id_tok:
        acct = chatgpt_account_id_from_id_token(id_tok)
    if not acct:
        raise ValueError("Could not resolve chatgpt account id (tokens.account_id or id_token claim)")
    return bearer, acct
