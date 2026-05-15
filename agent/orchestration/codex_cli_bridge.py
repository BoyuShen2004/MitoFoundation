"""Run Codex CLI for browser login (local dev helper)."""

from __future__ import annotations

import os
import re
import select
import shutil
import subprocess
import sys
import threading
from typing import Any, Callable
from urllib.parse import urlparse

try:
    import pty
except ImportError:
    pty = None  # type: ignore[misc, assignment]

_login_state_lock = threading.Lock()
_login_gen = 0
_login_auth_url: str | None = None
_login_device_code: str | None = None
_codex_login_help_blob: str | None = None
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_DEVICE_CODE_RE = re.compile(r"\b([A-Z0-9]{4}-[A-Z0-9]{5})\b")


def _normalize_cli_flag_text(s: str) -> str:
    """
    Normalize CLI help text/flags so matching is tolerant of formatting:
    - case-insensitive
    - treats '-', '_' and whitespace as equivalent
    """
    normalized = s.casefold()
    normalized = normalized.replace("_", "-")
    normalized = re.sub(r"\s+", "-", normalized)
    return normalized


def _start_nl_spammer(
    stop: threading.Event, proc: subprocess.Popen[Any], write_newline: Callable[[], Any]
) -> None:
    """Periodically send newlines until a URL is captured or process exits."""
    def _run() -> None:
        n = 0
        while not stop.wait(1.5) and n < 50 and proc.poll() is None:
            n += 1
            try:
                write_newline()
            except (BrokenPipeError, OSError):
                break

    threading.Thread(target=_run, daemon=True).start()


def _finalize_login_reader(stop: threading.Event, proc: subprocess.Popen[Any]) -> None:
    """Stop newline nudges and ensure login process exits."""
    stop.set()
    try:
        if proc.poll() is None:
            proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.terminate()


def _codex_login_help_text(codex: str) -> str:
    """Cached text for ``codex login --help`` (stdout+stderr)."""
    global _codex_login_help_blob
    if _codex_login_help_blob is not None:
        return _codex_login_help_blob
    try:
        r = subprocess.run(
            [codex, "login", "--help"],
            capture_output=True,
            text=True,
            timeout=12,
            env=os.environ.copy(),
        )
        _codex_login_help_blob = (r.stdout or "") + "\n" + (r.stderr or "")
    except (OSError, subprocess.TimeoutExpired):
        _codex_login_help_blob = ""
    return _codex_login_help_blob


def _codex_login_supports(codex: str, *tokens: str) -> bool:
    """Whether any token appears in ``codex login --help`` output."""
    blob = _normalize_cli_flag_text(_codex_login_help_text(codex))
    return any(_normalize_cli_flag_text(tok) in blob for tok in tokens)


def _codex_login_env() -> dict[str, str]:
    """Env so child processes do not spawn a GUI browser (Cursor/IDE often hooks that)."""
    env = os.environ.copy()
    true_bin = shutil.which("true")
    if true_bin:
        env["BROWSER"] = true_bin
    return env


def _is_codex_loopback_url(u: str) -> bool:
    """
    Never open loopback URLs from captured CLI text for login:

    - ``GET /`` on Codex's listener (e.g. port 1455) returns **404** (only ``/auth/callback``,
      ``/success``, ``/cancel`` are served — see ``codex-rs/login``).
    - ``/auth/callback`` is for the IdP **redirect after** sign-in, not the first page to visit.

    The browser must load the **HTTPS OAuth authorize** URL (e.g. ``auth.openai.com/oauth/authorize``)
    that the CLI prints; that flow then redirects to localhost when appropriate.
    """
    try:
        p = urlparse(u)
    except Exception:
        return False
    host = (p.hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1")


def _is_likely_codex_auth_url(u: str) -> bool:
    if _is_codex_loopback_url(u):
        return False
    ul = u.lower()
    if "oauth/authorize" in ul or "/authorize?" in ul:
        return True
    return any(
        x in ul
        for x in (
            "chatgpt.com",
            "openai.com",
            "auth.openai.com",
            "oauth.openai.com",
            "auth0",
            "login.live",
            "microsoft",
        )
    )


def get_codex_login_status(expected_generation: int) -> dict[str, Any]:
    """Poll for the first auth URL seen for a given login session (generation from ``start_codex_login_background``)."""
    with _login_state_lock:
        if expected_generation != _login_gen:
            return {"stale": True, "auth_url": None, "device_code": None}
        return {"stale": False, "auth_url": _login_auth_url, "device_code": _login_device_code}


def start_codex_login_background() -> int:
    """
    Run ``codex login`` locally and capture the first sign-in URL from CLI output.
    The browser must be opened by the web client (``window.open``) so it runs in the user's
    normal browser session, not the IDE-embedded Python process (which can double-prompt).

    Nudges stdin with newlines so non-interactive defaults can progress. Best effort.
    """
    global _login_gen, _login_auth_url, _login_device_code
    codex = shutil.which("codex")
    if not codex:
        raise RuntimeError("Codex CLI not found on PATH.")
    with _login_state_lock:
        _login_gen += 1
        session_gen = _login_gen
        _login_auth_url = None
        _login_device_code = None
    login_argv = [codex, "login"]
    if _codex_login_supports(codex, "--device-auth", "device_auth"):
        # Prefer device auth to avoid localhost callback issues in embedded/remote browsers.
        login_argv.append("--device-auth")
    if _codex_login_supports(codex, "--no-browser", "no_browser"):
        login_argv.append("--no-browser")
    cmd = login_argv
    stdbuf = shutil.which("stdbuf")
    if stdbuf:
        cmd = [stdbuf, "-oL", "-eL", *cmd]
    stop = threading.Event()
    url_re = re.compile(r"https?://[^\s\]'\"<>)\]]+")

    def try_set_auth_url(u: str) -> None:
        global _login_auth_url, _login_gen
        with _login_state_lock:
            if session_gen != _login_gen:
                return
            if _login_auth_url is not None:
                return
            _login_auth_url = u
        # Stop auto-newline nudges once we have the URL so we don't accidentally
        # advance/cancel subsequent interactive prompts while waiting for callback.
        stop.set()

    def try_set_device_code(raw_code: str) -> None:
        global _login_device_code, _login_gen
        code = raw_code.strip().upper()
        if not _DEVICE_CODE_RE.fullmatch(code):
            return
        with _login_state_lock:
            if session_gen != _login_gen:
                return
            if _login_device_code is not None:
                return
            _login_device_code = code

    def scan_urls(chunk: str) -> None:
        chunk = _ANSI_ESCAPE_RE.sub("", chunk)
        for m in url_re.finditer(chunk):
            u = m.group(0).rstrip(".,);")
            if _is_likely_codex_auth_url(u):
                try_set_auth_url(u)
        up = chunk.upper()
        for m in _DEVICE_CODE_RE.finditer(up):
            try_set_device_code(m.group(1))

    use_pty = pty is not None and sys.platform != "win32"

    if use_pty:
        master_fd, slave_fd = pty.openpty()
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                start_new_session=True,
                env=_codex_login_env(),
                close_fds=False,
            )
        except Exception:
            try:
                os.close(master_fd)
            except OSError:
                pass
            try:
                os.close(slave_fd)
            except OSError:
                pass
            raise
        os.close(slave_fd)

        def reader_pty() -> None:
            buf = b""
            try:
                while True:
                    r, _, _ = select.select([master_fd], [], [], 0.35)
                    if master_fd in r:
                        try:
                            chunk = os.read(master_fd, 65536)
                        except OSError:
                            chunk = b""
                        if chunk:
                            buf += chunk
                            while b"\n" in buf:
                                line, buf = buf.split(b"\n", 1)
                                scan_urls(line.decode("utf-8", errors="replace"))
                        elif proc.poll() is not None:
                            break
                    elif proc.poll() is not None:
                        break
                if buf:
                    scan_urls(buf.decode("utf-8", errors="replace"))
            finally:
                try:
                    os.close(master_fd)
                except OSError:
                    pass
                _finalize_login_reader(stop, proc)

        _start_nl_spammer(stop, proc, lambda: os.write(master_fd, b"\n"))
        threading.Thread(target=reader_pty, daemon=True).start()
        return session_gen

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=_codex_login_env(),
    )

    def reader_pipe() -> None:
        try:
            if proc.stdout:
                for line in iter(proc.stdout.readline, ""):
                    scan_urls(line)
                    if proc.poll() is not None:
                        break
                rest = proc.stdout.read() or ""
                if rest:
                    scan_urls(rest)
        finally:
            _finalize_login_reader(stop, proc)

    def _write_pipe_newline() -> None:
        if proc.stdin:
            proc.stdin.write("\n")
            proc.stdin.flush()

    _start_nl_spammer(stop, proc, _write_pipe_newline)
    threading.Thread(target=reader_pipe, daemon=True).start()
    return session_gen
