"""Track the active Studio scrape subprocess so ``/websites/scrape-cancel`` can SIGKILL it."""

from __future__ import annotations

import subprocess
import threading
from typing import Any

_lock = threading.Lock()
# session_id (Studio default "default") -> {"event": threading.Event, "proc": Popen | None, "log": list[str]}
_sessions: dict[str, dict[str, Any]] = {}
# Last completed scrape log per session (so UI can still read after ``session_end``).
_last_logs: dict[str, str] = {}
_MAX_LOG_CHARS = 500_000


def session_start(session_id: str, cancel_event: threading.Event) -> None:
    with _lock:
        _sessions[session_id] = {"event": cancel_event, "proc": None, "log": []}


def session_set_proc(session_id: str, proc: subprocess.Popen) -> None:
    with _lock:
        if session_id in _sessions:
            _sessions[session_id]["proc"] = proc


def session_clear_proc(session_id: str) -> None:
    with _lock:
        if session_id in _sessions:
            _sessions[session_id]["proc"] = None


def session_log_append(session_id: str, text: str) -> None:
    if not text:
        return
    with _lock:
        row = _sessions.get(session_id)
        if not row:
            return
        parts: list[str] = row.setdefault("log", [])
        parts.append(text)
        while True:
            total = sum(len(p) for p in parts)
            if total <= _MAX_LOG_CHARS or len(parts) <= 1:
                break
            parts.pop(0)


def session_snapshot(session_id: str) -> tuple[bool, str]:
    """Return ``(running, log_text)`` — live buffer while a scrape is active, else last completed tail."""
    with _lock:
        row = _sessions.get(session_id)
        if row:
            return True, "".join(row.get("log") or [])
        return False, _last_logs.get(session_id, "")


def session_end(session_id: str) -> None:
    with _lock:
        row = _sessions.pop(session_id, None)
        if row:
            log = "".join(row.get("log") or "")
            if log:
                if len(log) > _MAX_LOG_CHARS:
                    log = log[-_MAX_LOG_CHARS :]
                _last_logs[session_id] = log


def session_clear(session_id: str) -> bool:
    """Clear live/last scrape log buffers for a session when not running."""
    with _lock:
        if session_id in _sessions:
            # Keep active run logs intact; caller should stop first.
            return False
        had = session_id in _last_logs
        _last_logs.pop(session_id, None)
    return had


def session_kill(session_id: str) -> bool:
    """Signal cancel and kill the tracked subprocess if still running."""
    with _lock:
        data = _sessions.get(session_id)
        if not data:
            return False
        cancel_ev: threading.Event = data["event"]
        proc: subprocess.Popen | None = data.get("proc")
        cancel_ev.set()
    killed = False
    if proc is not None and proc.poll() is None:
        try:
            proc.kill()
            killed = True
            proc.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            pass
    return killed
