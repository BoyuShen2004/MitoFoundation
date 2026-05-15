"""OpenAI Platform chat completions and ChatGPT Codex (OpenClaw-style ``openai-codex``) Responses API."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class LLMUnavailableError(Exception):
    """Raised when the HTTP API cannot be reached or returns an error."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        self.hint = hint
        super().__init__(message)


DEFAULT_OPENAI_BASE = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
# Same default as the Codex CLI for ChatGPT sign-in (see openai/codex model_provider_info).
CHATGPT_CODEX_BASE = "https://chatgpt.com/backend-api/codex"
DEFAULT_CODEX_MODEL = "gpt-5.4"

_LLM_PROVIDERS = frozenset({"chatgpt_codex", "openai_api"})


def apply_codex_model_storage_repairs(project_root: Path) -> None:
    """Pin ChatGPT Codex HTTP to ``gpt-5.4`` and drop saved model-switch lists from older builds."""
    try:
        from agent.orchestration.llm_settings import load_stored_llm_settings, merge_patch_stored
    except Exception:
        return
    st = load_stored_llm_settings(project_root)
    if _stored_provider(st) != "chatgpt_codex":
        return
    patch: dict[str, Any] = {}
    if (st.get("openai_model") or "").strip() != DEFAULT_CODEX_MODEL:
        patch["openai_model"] = DEFAULT_CODEX_MODEL
    if (st.get("codex_agent_model_list") or "").strip():
        patch["codex_agent_model_list"] = ""
    if patch:
        merge_patch_stored(project_root, patch)


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = Path(__file__).resolve().parents[3]
    load_dotenv(root / ".env")


_load_dotenv()


def _project_root() -> Path:
    from config.paths import project_root

    return project_root()


def _stored_llm() -> dict[str, str]:
    try:
        from agent.orchestration.llm_settings import load_stored_llm_settings

        return load_stored_llm_settings(_project_root())
    except Exception:
        return {}


def _coalesce_env_file(env_name: str, file_key: str, stored: dict[str, str]) -> str:
    ev = (os.getenv(env_name) or "").strip()
    if ev:
        return ev
    return (stored.get(file_key) or "").strip()


def _coalesce_file_env(file_key: str, env_name: str, stored: dict[str, str]) -> str:
    """Prefer saved UI config; only use env as fallback when file value is empty."""
    fv = (stored.get(file_key) or "").strip()
    if fv:
        return fv
    return (os.getenv(env_name) or "").strip()


def _norm_provider_token(s: str) -> str:
    x = (s or "").strip().lower().replace("-", "_")
    return x if x in _LLM_PROVIDERS else ""


def _stored_provider(st: dict[str, str]) -> str:
    """``chatgpt_codex`` vs ``openai_api`` — keeps Codex and Platform from sharing one ambiguous base URL."""
    ev = _norm_provider_token(os.getenv("MITO2_LLM_PROVIDER", ""))
    if ev:
        return ev
    raw = (os.getenv("OPENAI_BASE_URL") or "").strip() or (st.get("openai_base_url") or "").strip()
    # Safety: stale provider tokens are common after tab switching in Settings. Route by base URL first.
    if raw:
        return "chatgpt_codex" if is_codex_chatgpt_backend(raw) else "openai_api"
    file_p = _norm_provider_token(st.get("llm_provider", ""))
    if file_p:
        return file_p
    return "openai_api"


def _normalize_codex_base(base: str) -> str:
    """``.../codex`` or ``.../codex/v1`` → canonical ``.../codex`` root for Responses."""
    b = base.strip().rstrip("/")
    if re.search(r"/v1$", b, re.I):
        b = re.sub(r"/v1$", "", b, flags=re.I)
    return b


def is_codex_chatgpt_backend(base_url: str) -> bool:
    """True when using ChatGPT-hosted Codex (same host family as OpenClaw ``openai-codex/*``)."""
    try:
        host = urlparse(base_url).netloc.lower()
    except ValueError:
        return False
    return "chatgpt.com" in host and "codex" in base_url.lower()


def _effective_base_and_model(st: dict[str, str]) -> tuple[str, str]:
    if _stored_provider(st) == "chatgpt_codex":
        env_base = (os.getenv("OPENAI_BASE_URL") or "").strip()
        if env_base and is_codex_chatgpt_backend(env_base):
            base = _normalize_codex_base(env_base).rstrip("/")
        else:
            base = CHATGPT_CODEX_BASE.rstrip("/")
        # ChatGPT Codex HTTP: fixed model (no UI switching; ignore ``OPENAI_MODEL`` / saved slug).
        return base, DEFAULT_CODEX_MODEL
    raw_base = _coalesce_file_env("openai_base_url", "OPENAI_BASE_URL", st).rstrip("/")
    base = raw_base if raw_base else DEFAULT_OPENAI_BASE
    model_raw = _coalesce_file_env("openai_model", "OPENAI_MODEL", st)
    model = model_raw if model_raw else DEFAULT_OPENAI_MODEL
    return base, model


def _try_codex_auth_json_creds(st: dict[str, str]) -> tuple[str, str] | None:
    """Resolve ``(bearer, account_id)`` from ``auth.json`` for ChatGPT Codex."""
    if _stored_provider(st) != "chatgpt_codex":
        return None
    profile = (st.get("codex_auth_profile_id") or "").strip()
    try:
        from agent.orchestration.codex_auth import (
            default_codex_auth_path,
            list_codex_profiles,
            resolve_codex_credentials,
        )

        ap = (st.get("codex_auth_json_path") or "").strip()
        pth = Path(ap).expanduser().resolve() if ap else default_codex_auth_path()
        if not profile:
            rows = list_codex_profiles(pth)
            if not rows:
                return None
            ids = [r.id for r in rows]
            if len(rows) == 1:
                profile = rows[0].id
            elif "default" in ids:
                profile = "default"
            else:
                return None
        return resolve_codex_credentials(pth, profile)
    except (OSError, ValueError, KeyError, FileNotFoundError, json.JSONDecodeError):
        return None


def get_llm_config() -> tuple[str, str, str]:
    """
    Returns ``(base_url, api_key, model)``.

    **Codex:** ``OPENAI_API_KEY`` env → ``auth.json`` (profile) → saved key in ``llm.json``.
    **OpenAI API:** env → saved key only (never ``auth.json``).
    """
    st = _stored_llm()
    base, model = _effective_base_and_model(st)
    key_env = (os.getenv("OPENAI_API_KEY") or "").strip()
    manual = (st.get("openai_api_key") or "").strip()
    if _stored_provider(st) == "chatgpt_codex":
        codex_creds = _try_codex_auth_json_creds(st)
        file_bearer = codex_creds[0] if codex_creds else ""
        key = key_env or file_bearer or manual
    else:
        # API mode should honor the key saved in Settings; env only fills in when no saved key exists.
        key = manual or key_env
    return base, key, model


def get_chatgpt_account_id() -> str:
    """``CHATGPT_ACCOUNT_ID`` env → Codex ``auth.json`` profile → saved manual field (Codex only)."""
    st = _stored_llm()
    if _stored_provider(st) != "chatgpt_codex":
        return ""
    acct_env = (os.getenv("CHATGPT_ACCOUNT_ID") or "").strip()
    if acct_env:
        return acct_env
    codex_creds = _try_codex_auth_json_creds(st)
    if codex_creds and codex_creds[1].strip():
        return codex_creds[1].strip()
    return (st.get("chatgpt_account_id") or "").strip()


def build_scrape_subprocess_llm_env() -> dict[str, str]:
    """Environment variables for ``master.agent`` subprocesses (Pipeline Studio scrape).

    Mirrors **Settings → Model & API** and Codex ``auth.json`` so the catalog scrape does not
    default to ``127.0.0.1:11434`` when the Web UI is configured for OpenAI or ChatGPT Codex.
    """
    base, key, model = get_llm_config()
    out: dict[str, str] = {}
    if key:
        out["OPENAI_API_KEY"] = key
    if base:
        out["OPENAI_BASE_URL"] = base.rstrip("/")
    if model:
        out["OPENAI_MODEL"] = model
    acct = get_chatgpt_account_id()
    if acct:
        out["CHATGPT_ACCOUNT_ID"] = acct
    st = _stored_llm()
    prov = _stored_provider(st)
    if prov:
        out["MITO2_LLM_PROVIDER"] = prov
    # Pin repo root so ``agent/orchestration/config/llm.json`` resolves like the Web UI (subprocess cwd is ``1web_scraper_01``).
    out["MITO2_PROJECT_ROOT"] = str(_project_root().resolve())
    # Codex: scrape **routing** (``SCRAPE_LLM_ROUTE``) may use ``llm.invoke`` via ``llm_strategy``; env
    # ``SCRAPE_CODEX_TRY_ORDER`` (default ``cli,http``) applies when wiring Codex for subprocess scrapes.
    cjp = (st.get("codex_auth_json_path") or "").strip()
    if cjp:
        out["MITO2_CODEX_AUTH_JSON"] = str(Path(cjp).expanduser().resolve())
    return out


def get_llm_transport() -> str:
    st = _stored_llm()
    return "codex_responses" if _stored_provider(st) == "chatgpt_codex" else "chat_completions"


def setup_hint() -> str:
    base, key, model = get_llm_config()
    transport = get_llm_transport()
    lines = [
        "**LLM setup (OpenClaw-aligned):**",
        "",
        "**A — OpenAI Platform** (OpenClaw route `openai/gpt-5.4`, etc.):",
        "   `export OPENAI_API_KEY=sk-...`",
        f"   `export OPENAI_BASE_URL={DEFAULT_OPENAI_BASE}`   # optional; this is the default",
        "   `export OPENAI_MODEL=gpt-4o-mini`   # or `gpt-5.4` when available on your account",
        "",
        "**B — OpenAI Codex** (CLI login, separate from the OpenAI API key flow):",
        "   In **Settings → OpenAI Codex**, use **Sign in with Codex (open browser)**, complete login in the browser, then **Refresh accounts** and **Save** (writes ``~/.codex/auth.json``).",
        "   Or run ``codex login`` yourself, then **Save** in that tab.",
        "   Or set ``MITO2_LLM_PROVIDER=chatgpt_codex`` plus bearer in ``OPENAI_API_KEY`` or profile in ``llm.json``.",
        "   If HTTP to ChatGPT’s backend fails (400/unsupported fields), set ``MITO2_CODEX_TRANSPORT=cli`` and ensure ``codex`` is on ``PATH``.",
        "   The Codex HTTP base is fixed to ChatGPT’s backend (not browsable in a normal browser tab).",
        "",
        "**C — OpenAI-compatible proxy** (e.g. a gateway that exposes `/v1/chat/completions`):",
        "   Set `OPENAI_BASE_URL` to that proxy’s `/v1` root and `OPENAI_API_KEY` as the proxy expects.",
        "",
        "**UI:** use **Settings → Model & API** in the app (saves `agent/orchestration/config/llm.json`). "
        "Environment variables still override file values when set.",
        "",
        "Optional: add a `.env` file in the repo root (loaded if `python-dotenv` is installed).",
        "",
        f"_Resolved: transport={transport!r}, base={base!r}, model={model!r}, api_key={'set' if key else 'missing'}._",
        "",
        "For **ChatGPT Codex**, requests always use model **gpt-5.4** (saved ``openai_model`` in ``llm.json`` is kept in sync).",
    ]
    return "\n".join(lines)


def _codex_cli_transport() -> bool:
    return _stored_provider(_stored_llm()) == "chatgpt_codex" and _codex_transport() in (
        "cli",
        "exec",
        "codex",
        "codex_cli",
    )


def _require_llm_credentials(api_key: str, provider: str) -> None:
    """HTTP Codex and OpenAI API need a bearer; ``codex exec`` uses the CLI session and skips this."""
    if api_key:
        return
    if _codex_cli_transport():
        return
    if provider == "chatgpt_codex":
        raise LLMUnavailableError(
            "No OpenAI Codex session token. You logged out or ``~/.codex/auth.json`` has no usable token. "
            "Use **Settings → OpenAI Codex → Sign in with Codex (open browser)**, then **Refresh accounts** and **Save** "
            "(or run `codex login` yourself, or set `OPENAI_API_KEY` to a bearer token).",
            hint=setup_hint(),
        )
    raise LLMUnavailableError(
        "OPENAI_API_KEY is not set.",
        hint=setup_hint(),
    )


def _http_post_json(url: str, body: dict[str, Any], headers: dict[str, str]) -> str:
    data = json.dumps(body).encode("utf-8")
    hdrs = {"Content-Type": "application/json", **headers}
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as res:
            return res.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")[:4000]
        raise LLMUnavailableError(
            f"LLM HTTP {e.code}: {err_body or e.reason}",
            hint=setup_hint(),
        ) from e
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", e)
        raise LLMUnavailableError(
            f"Cannot connect to LLM ({reason!r}). Check OPENAI_BASE_URL and network.",
            hint=setup_hint(),
        ) from e
    except TimeoutError as e:
        raise LLMUnavailableError("LLM request timed out.", hint=setup_hint()) from e


def _chat_completions_url_candidates(base: str) -> list[str]:
    """
    Try both common OpenAI-compatible forms:
    - ``{base}/chat/completions`` (base already includes ``/v1``)
    - ``{base}/v1/chat/completions`` (base is host root)
    """
    b = base.rstrip("/")
    return [f"{b}/chat/completions", f"{b}/v1/chat/completions"]


def _http_post_json_first_matching(
    urls: list[str], body: dict[str, Any], headers: dict[str, str]
) -> str:
    last: LLMUnavailableError | None = None
    for url in urls:
        try:
            return _http_post_json(url, body, headers)
        except LLMUnavailableError as e:
            msg = str(e)
            if "HTTP 404" in msg and url != urls[-1]:
                last = e
                continue
            raise
    assert last is not None
    raise last


def _messages_to_codex_responses_body(
    messages: list[dict[str, str]], model: str, _max_tokens: int
) -> dict[str, Any]:
    instructions: list[str] = []
    input_items: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            instructions.append(content)
        elif role == "user":
            input_items.append(
                {"role": "user", "content": [{"type": "input_text", "text": content}]}
            )
        elif role == "assistant":
            input_items.append(
                {"role": "assistant", "content": [{"type": "output_text", "text": content}]}
            )
    # ChatGPT's ``/backend-api/codex/responses`` is stricter than api.openai.com/v1/responses:
    # it rejects ``max_output_tokens``, and often ``max_tokens`` / ``temperature`` as well.
    # Omit sampling and length limits so the backend uses its defaults.
    body: dict[str, Any] = {
        "model": model,
        "input": input_items,
        "store": False,
        "stream": True,
    }
    if instructions:
        body["instructions"] = "\n\n".join(instructions)
    return body


def _extract_responses_output_text(obj: dict[str, Any]) -> str:
    """Best-effort parse of Responses API JSON (non-streaming)."""
    out = obj.get("output")
    if isinstance(out, list):
        chunks: list[str] = []
        for item in out:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    t = block.get("type", "")
                    if t in ("output_text", "text") and block.get("text"):
                        chunks.append(str(block["text"]))
                    elif isinstance(block.get("text"), str):
                        chunks.append(block["text"])
            msg = item.get("message")
            if isinstance(msg, dict):
                text = _extract_responses_output_text({"output": [msg]})
                if text:
                    chunks.append(text)
        if chunks:
            return "\n".join(chunks).strip()
    # Some shapes put text at top level
    for key in ("output_text", "text", "response"):
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _codex_transport() -> str:
    """``http`` (default) = ChatGPT ``/codex/responses``; ``cli`` = local ``codex exec``."""
    return (os.getenv("MITO2_CODEX_TRANSPORT") or "http").strip().lower()


def _transcript_for_codex_cli(messages: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for m in messages:
        role = (m.get("role") or "").strip().upper() or "USER"
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"### {role}\n{content}")
    return "\n\n".join(lines).strip() or "(no content)"


def _complete_codex_cli(messages: list[dict[str, str]], model: str) -> str:
    """Non-interactive Codex via the installed CLI (same auth as ``codex login``)."""
    codex_bin = shutil.which("codex")
    if not codex_bin:
        raise LLMUnavailableError(
            "MITO2_CODEX_TRANSPORT=cli but `codex` was not found on PATH.",
            hint=setup_hint(),
        )
    transcript = _transcript_for_codex_cli(messages)
    root = _project_root()
    fd, out_path = tempfile.mkstemp(prefix="mito2-codex-", suffix=".txt")
    os.close(fd)
    try:
        cmd = [
            codex_bin,
            "exec",
            "-m",
            model,
            "--sandbox",
            "read-only",
            "--ephemeral",
            "--skip-git-repo-check",
            "--color",
            "never",
            "-C",
            str(root),
            "-o",
            out_path,
            "-",
        ]
        r = subprocess.run(
            cmd,
            input=transcript + "\n",
            text=True,
            cwd=str(root),
            timeout=600,
            capture_output=True,
            env=os.environ.copy(),
        )
        reply = Path(out_path).read_text(encoding="utf-8", errors="replace").strip()
        if reply:
            return reply
        err = (r.stderr or r.stdout or "").strip()[:2000]
        raise LLMUnavailableError(
            f"codex exec exited {r.returncode} with no -o output. Log: {err!r}",
            hint=setup_hint(),
        )
    except subprocess.TimeoutExpired as e:
        raise LLMUnavailableError("codex exec timed out.", hint=setup_hint()) from e
    except OSError as e:
        raise LLMUnavailableError(f"codex exec failed: {e}", hint=setup_hint()) from e
    finally:
        try:
            Path(out_path).unlink(missing_ok=True)
        except OSError:
            pass


def _codex_responses_url_candidates(base: str) -> list[str]:
    """Codex CLI posts to ``{base}/responses``; some gateways expect ``/v1/responses``."""
    b = base.rstrip("/")
    return [f"{b}/responses", f"{b}/v1/responses"]


def _parse_codex_sse_to_text(raw: str) -> str:
    """
    Parse ChatGPT Codex / OpenAI-style Responses SSE (``text/event-stream``).

    Text chunks use ``response.output_text.delta`` events (see openai/codex ``process_responses_event``).
    """
    text = raw.replace("\r\n", "\n")
    parts: list[str] = []
    failures: list[str] = []
    incomplete = False
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].strip())
        if not data_lines:
            continue
        payload = "\n".join(data_lines)
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        et = obj.get("type", "")
        if et == "response.output_text.delta":
            d = obj.get("delta")
            if isinstance(d, str):
                parts.append(d)
        elif et == "response.failed":
            resp = obj.get("response")
            err: Any = None
            if isinstance(resp, dict):
                err = resp.get("error")
            if isinstance(err, dict):
                failures.append(str(err.get("message") or err))
            else:
                failures.append(str(obj))
        elif et == "response.incomplete":
            incomplete = True
    if failures:
        raise LLMUnavailableError(
            f"Codex stream error: {'; '.join(failures)}",
            hint=setup_hint(),
        )
    out = "".join(parts).strip()
    if incomplete and not out:
        raise LLMUnavailableError(
            "Codex returned an incomplete response with no text.",
            hint=setup_hint(),
        )
    if not out:
        # Rare: JSON body instead of SSE (proxy) — try one-shot parse.
        try:
            obj = json.loads(raw.strip())
            if isinstance(obj, dict):
                t = _extract_responses_output_text(obj)
                if t:
                    return t
        except json.JSONDecodeError:
            pass
        raise LLMUnavailableError(
            "Codex stream ended with no assistant text.",
            hint=setup_hint(),
        )
    return out


def _http_post_codex_sse(url: str, body: dict[str, Any], headers: dict[str, str]) -> str:
    """POST JSON body; expect ``text/event-stream`` (Codex requires ``stream: true``)."""
    data = json.dumps(body).encode("utf-8")
    hdrs = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "User-Agent": "mitoFoundation2-chat-web/1.0",
        **headers,
    }
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as res:
            raw = res.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")[:4000]
        raise LLMUnavailableError(
            f"LLM HTTP {e.code}: {err_body or e.reason}",
            hint=setup_hint(),
        ) from e
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", e)
        raise LLMUnavailableError(
            f"Cannot connect to LLM ({reason!r}). Check OPENAI_BASE_URL and network.",
            hint=setup_hint(),
        ) from e
    except TimeoutError as e:
        raise LLMUnavailableError("LLM request timed out.", hint=setup_hint()) from e
    return _parse_codex_sse_to_text(raw)


def _http_post_codex_sse_first_matching(
    urls: list[str], body: dict[str, Any], headers: dict[str, str]
) -> str:
    last: LLMUnavailableError | None = None
    for url in urls:
        try:
            return _http_post_codex_sse(url, body, headers)
        except LLMUnavailableError as e:
            msg = str(e)
            if "HTTP 404" in msg and url != urls[-1]:
                last = e
                continue
            raise
    assert last is not None
    raise last


def _complete_codex_responses(
    base: str,
    api_key: str,
    messages: list[dict[str, str]],
    model: str,
    max_tokens: int,
) -> str:
    account_id = get_chatgpt_account_id()
    if not account_id:
        raise LLMUnavailableError(
            "ChatGPT account id is required for Codex (from auth.json profile, CHATGPT_ACCOUNT_ID, or Settings).",
            hint=setup_hint(),
        )
    body = _messages_to_codex_responses_body(messages, model, max_tokens)
    if not body.get("input"):
        raise LLMUnavailableError(
            "Codex request has no user/assistant messages (only system, or empty text).",
            hint=setup_hint(),
        )
    # Header name matches openai/codex ``add_auth_headers`` (``ChatGPT-Account-ID``).
    headers = {
        "Authorization": f"Bearer {api_key}",
        "ChatGPT-Account-ID": account_id,
    }
    return _http_post_codex_sse_first_matching(_codex_responses_url_candidates(base), body, headers)


def _complete_chat_completions(
    base: str,
    api_key: str,
    messages: list[dict[str, str]],
    model: str,
    max_tokens: int,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    raw = _http_post_json_first_matching(
        _chat_completions_url_candidates(base),
        payload,
        headers={
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMUnavailableError(
            f"LLM returned non-JSON ({e}): {raw[:500]!r}…",
            hint=setup_hint(),
        ) from e
    text = (obj.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
    return (text or "").strip()


def complete(messages: list[dict], model: str | None = None, max_tokens: int = 2048) -> str:
    st = _stored_llm()
    provider = _stored_provider(st)
    base, api_key, default_model = get_llm_config()
    _require_llm_credentials(api_key, provider)
    mdl = model if model is not None else default_model
    # Normalize message shape to str content
    norm: list[dict[str, str]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", ""))
        c = m.get("content")
        if isinstance(c, str):
            norm.append({"role": role, "content": c})
        elif c is not None:
            norm.append({"role": role, "content": json.dumps(c)})

    if provider == "chatgpt_codex":
        if _codex_transport() in ("cli", "exec", "codex", "codex_cli"):
            return _complete_codex_cli(norm, mdl)
        return _complete_codex_responses(base, api_key, norm, mdl, max_tokens)
    return _complete_chat_completions(base, api_key, norm, mdl, max_tokens)
