"""Run catalog deep scrape via ``1web_scraper_01`` (``master.deep`` pipeline)."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import select
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .playwright_browsers import ensure_playwright_chromium
from . import scrape_control

# Returned from ``_drain_merged_stdout`` when ``cancel_event`` is set (distinct from timeout -9).
_RC_USER_CANCELLED = -88

# Canonical names in mitoFoundation2 ``outputs/`` (no workspace slug copies).
OO_PROBE_NAME    = "OpenOrganelle.probe.json"
OO_REPORT_NAME   = "OpenOrganelle.md"
BOSS_PROBE_NAME  = "BossDB.probe.json"
BOSS_REPORT_NAME = "BossDB.md"


def _remove_openorganelle_mirror_dupes(dst_dir: Path) -> None:
    """Drop legacy duplicate probe JSON only (``openorganelle_NN.probe.json``).

    Do **not** remove ``OpenOrganelle_NN.md`` — those are real catalog batch reports from
    ``catalog_pipeline`` (incremental scrapes); deleting them broke first-run outputs.
    """
    if not dst_dir.is_dir():
        return
    for p in list(dst_dir.iterdir()):
        if not p.is_file():
            continue
        name = p.name
        try:
            if re.fullmatch(r"(?i)openorganelle_\d+\.probe\.json", name):
                p.unlink()
        except OSError:
            pass


def resolve_openorganelle_repo_root(mito2_root: Path) -> Path | None:
    """Return the project root that owns ``1web_scraper_01`` with bundled OpenOrganelle code."""
    env = (os.environ.get("MITO2_PROJECT_ROOT") or "").strip()
    root = Path(env).expanduser().resolve() if env else mito2_root.resolve()
    ws = root / "1web_scraper_01"
    if not (ws / "master" / "agent.py").is_file():
        return None
    if not (ws / "master" / "deep" / "catalog_pipeline.py").is_file():
        return None
    return root


def is_openorganelle_url(url: str) -> bool:
    return "openorganelle.janelia.org" in (url or "").lower()


def is_bossdb_url(url: str) -> bool:
    u = (url or "").lower()
    return any(h in u for h in ("bossdb.org", "api.bossdb.io", "api.metadata.bossdb.org"))


def normalize_openorganelle_landing_url(url: str) -> str:
    """SPA is served from ``/``; ``/datasets`` etc. are not static objects on S3."""
    u = (url or "").strip()
    if not u.startswith(("http://", "https://")):
        u = "https://" + u.lstrip("/")
    if is_openorganelle_url(u):
        return "https://openorganelle.janelia.org/"
    return u


def _timeout_sec() -> int | None:
    raw = (os.environ.get("OPENORGANELLE_SCRAPE_TIMEOUT_SEC") or "").strip()
    if not raw:
        return 7200
    try:
        n = int(raw)
        return None if n <= 0 else n
    except ValueError:
        return 7200


def _drain_merged_stdout(
    proc: subprocess.Popen[str],
    *,
    timeout: float | None,
    log_line: Callable[[str], None] | None,
    cancel_event: threading.Event | None = None,
) -> tuple[str, int]:
    """Read merged stdout+stderr line-by-line; optional live ``log_line``. Uses ``select`` (POSIX)."""
    pieces: list[str] = []
    stdout = proc.stdout
    if stdout is None:
        code = proc.wait(timeout=timeout)
        return "", code
    fd = stdout.fileno()
    start = time.monotonic()
    while True:
        if cancel_event is not None and cancel_event.is_set():
            proc.kill()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
            rest = stdout.read() or ""
            if rest:
                pieces.append(rest)
                if log_line:
                    log_line(rest)
            return "".join(pieces), _RC_USER_CANCELLED
        if timeout is not None and time.monotonic() - start > timeout:
            proc.kill()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
            rest = stdout.read() or ""
            if rest:
                pieces.append(rest)
                if log_line:
                    log_line(rest)
            return "".join(pieces), -9
        code = proc.poll()
        if code is not None:
            rest = stdout.read() or ""
            if rest:
                pieces.append(rest)
                if log_line:
                    log_line(rest)
            return "".join(pieces), code
        r, _, _ = select.select([fd], [], [], 0.25)
        if r:
            line = stdout.readline()
            if line:
                pieces.append(line)
                if log_line:
                    log_line(line)


def run_openorganelle_deep_scrape(
    mito2_root: Path,
    *,
    workspace_slug: str,
    log_line: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
    scrape_session_id: str | None = None,
    subprocess_llm_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Subprocess ``python -m master.agent --sites OpenOrganelle`` (cwd ``1web_scraper_01``); mirror ``OpenOrganelle.*``."""
    repo = resolve_openorganelle_repo_root(mito2_root)
    ws = repo / "1web_scraper_01" if repo else None
    agent_py = ws / "master" / "agent.py" if ws else None
    if repo is None or ws is None or agent_py is None or not agent_py.is_file():
        return {
            "ok": False,
            "error": "openorganelle_scraper_missing",
            "message": (
                "Catalog deep scrape needs ``1web_scraper_01/master/deep/`` (or set MITO2_PROJECT_ROOT)."
            ),
        }

    ensure_playwright_chromium(log_line=log_line)

    timeout = _timeout_sec()
    cmd = [sys.executable, "-m", "master.agent", "--sites", "OpenOrganelle"]
    if log_line:
        log_line(f"[mito2] OpenOrganelle: starting bundled agent in {ws}\n")
        log_line(f"[mito2] Command: {' '.join(cmd)}\n")
        log_line(f"[mito2] Studio workspace folder: {workspace_slug} → outputs/{OO_PROBE_NAME}, {OO_REPORT_NAME}\n")
    run_env = os.environ.copy()
    run_env.setdefault("PYTHONUNBUFFERED", "1")
    # Same LLM as Pipeline Studio (ll.json + Codex auth), not bare defaults (e.g. localhost Ollama).
    if subprocess_llm_env:
        for _k, _v in subprocess_llm_env.items():
            if _v is not None and str(_v).strip():
                run_env[str(_k)] = str(_v).strip()
        if log_line:
            log_line("[mito2] Scrape subprocess: applied Web UI runtime settings.\n")
    run_env.setdefault("MITO2_PROJECT_ROOT", str(repo.resolve()))
    # Repo root first so ``import agent.chat_web`` works (Codex scrape uses ``agent.chat_web.app.llm.complete``).
    # Then ``1web_scraper_01`` so ``python -m master.agent`` resolves ``master``.
    _root_pp = os.pathsep.join([str(repo.resolve()), str(ws)])
    if (existing := (run_env.get("PYTHONPATH") or "").strip()):
        _pp = _root_pp + os.pathsep + existing
    else:
        _pp = _root_pp
    run_env["PYTHONPATH"] = _pp
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ws),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=run_env,
            bufsize=1,
        )
    except OSError as exc:
        return {
            "ok": False,
            "error": "spawn_failed",
            "message": str(exc),
        }

    try:
        if scrape_session_id:
            scrape_control.session_set_proc(scrape_session_id, proc)
        merged_out, returncode = _drain_merged_stdout(
            proc,
            timeout=float(timeout) if timeout is not None else None,
            log_line=log_line,
            cancel_event=cancel_event,
        )
    finally:
        if scrape_session_id:
            scrape_control.session_clear_proc(scrape_session_id)

    if returncode == _RC_USER_CANCELLED:
        if log_line:
            log_line("[mito2] OpenOrganelle agent process was stopped (cancel).\n")
        return {
            "ok": False,
            "error": "cancelled",
            "message": "Scrape cancelled",
            "stdout": merged_out[-8000:],
            "stderr": "",
        }

    if returncode == -9:
        return {
            "ok": False,
            "error": "timeout",
            "message": f"OpenOrganelle agent exceeded timeout ({timeout}s).",
            "stdout": merged_out[-12000:],
            "stderr": "",
        }

    src_dir = ws / "outputs"
    src_probe = src_dir / OO_PROBE_NAME
    ok_run = returncode == 0 and src_probe.is_file()

    dst_dir = mito2_root / "1web_scraper_01" / "outputs"
    dst_dir.mkdir(parents=True, exist_ok=True)
    probe_rel: str | None = None
    copied_md: list[str] = []

    if src_probe.is_file():
        _remove_openorganelle_mirror_dupes(dst_dir)
        dst_probe = dst_dir / OO_PROBE_NAME
        try:
            if src_probe.resolve() != dst_probe.resolve():
                shutil.copy2(src_probe, dst_probe)
        except OSError:
            shutil.copy2(src_probe, dst_probe)
        probe_rel = str(dst_probe.relative_to(mito2_root)).replace("\\", "/")

        mds = sorted(src_dir.glob("OpenOrganelle*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        if mds:
            latest = mds[0]
            dst_md = dst_dir / OO_REPORT_NAME
            try:
                if latest.resolve() != dst_md.resolve():
                    shutil.copy2(latest, dst_md)
            except OSError:
                shutil.copy2(latest, dst_md)
            copied_md.append(str(dst_md.relative_to(mito2_root)).replace("\\", "/"))

    if not ok_run:
        err_hint = ""
        if returncode != 0:
            err_hint = f"agent exit {returncode}"
        elif not src_probe.is_file():
            err_hint = "OpenOrganelle.probe.json missing after run"
        return {
            "ok": False,
            "error": "openorganelle_agent_failed",
            "message": err_hint,
            "stdout": merged_out[-12000:],
            "stderr": "",
            "probe_path": probe_rel,
            "markdown_copied": copied_md,
        }

    if log_line:
        log_line(
            f"[mito2] Agent finished (exit {returncode}). "
            f"Mirrored {OO_PROBE_NAME} and {OO_REPORT_NAME} (if present) into mitoFoundation2 outputs.\n"
        )

    return {
        "ok": True,
        "mito_foundation_bridge": True,
        "stdout": merged_out[-8000:],
        "stderr": "",
        "probe_path": probe_rel,
        "markdown_copied": copied_md,
        "source_probe": str(src_probe.relative_to(repo)).replace("\\", "/"),
    }


def provider_name_for_url(url: str) -> str | None:
    """Return the canonical (lowercase) provider name for a landing URL, or None if unknown.

    Used by site-agnostic orchestration code to avoid hardcoded ``if openorganelle`` checks.
    Extend this function when adding new providers that have a dedicated deep-scrape path.
    """
    if is_openorganelle_url(url):
        return "openorganelle"
    if is_bossdb_url(url):
        return "bossdb"
    return None


def run_bossdb_deep_scrape(
    mito2_root: Path,
    *,
    workspace_slug: str,
    log_line: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
    scrape_session_id: str | None = None,
    subprocess_llm_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Subprocess ``python -m master.agent --sites BossDB`` (cwd ``1web_scraper_01``); mirror ``BossDB.*``."""
    repo = resolve_openorganelle_repo_root(mito2_root)
    ws = repo / "1web_scraper_01" if repo else None
    agent_py = ws / "master" / "agent.py" if ws else None
    if repo is None or ws is None or agent_py is None or not agent_py.is_file():
        return {
            "ok": False,
            "error": "bossdb_scraper_missing",
            "message": "BossDB deep scrape needs ``1web_scraper_01/master/deep/`` (or set MITO2_PROJECT_ROOT).",
        }

    timeout = _timeout_sec()
    cmd = [sys.executable, "-m", "master.agent", "--sites", "BossDB"]
    if log_line:
        log_line(f"[mito2] BossDB: starting bundled agent in {ws}\n")
        log_line(f"[mito2] Command: {' '.join(cmd)}\n")
        log_line(f"[mito2] Studio workspace folder: {workspace_slug} → outputs/{BOSS_PROBE_NAME}, {BOSS_REPORT_NAME}\n")

    run_env = os.environ.copy()
    run_env.setdefault("PYTHONUNBUFFERED", "1")
    if subprocess_llm_env:
        for _k, _v in subprocess_llm_env.items():
            if _v is not None and str(_v).strip():
                run_env[str(_k)] = str(_v).strip()
        if log_line:
            log_line("[mito2] Scrape subprocess: applied Web UI runtime settings.\n")
    run_env.setdefault("MITO2_PROJECT_ROOT", str(repo.resolve()))
    _root_pp = os.pathsep.join([str(repo.resolve()), str(ws)])
    if (existing := (run_env.get("PYTHONPATH") or "").strip()):
        _pp = _root_pp + os.pathsep + existing
    else:
        _pp = _root_pp
    run_env["PYTHONPATH"] = _pp

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ws),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=run_env,
            bufsize=1,
        )
    except OSError as exc:
        return {"ok": False, "error": "spawn_failed", "message": str(exc)}

    try:
        if scrape_session_id:
            scrape_control.session_set_proc(scrape_session_id, proc)
        merged_out, returncode = _drain_merged_stdout(
            proc,
            timeout=float(timeout) if timeout is not None else None,
            log_line=log_line,
            cancel_event=cancel_event,
        )
    finally:
        if scrape_session_id:
            scrape_control.session_clear_proc(scrape_session_id)

    if returncode == _RC_USER_CANCELLED:
        if log_line:
            log_line("[mito2] BossDB agent process was stopped (cancel).\n")
        return {"ok": False, "error": "cancelled", "message": "Scrape cancelled", "stdout": merged_out[-8000:], "stderr": ""}

    if returncode == -9:
        return {"ok": False, "error": "timeout", "message": f"BossDB agent exceeded timeout ({timeout}s).", "stdout": merged_out[-12000:], "stderr": ""}

    src_dir   = ws / "outputs"
    src_probe = src_dir / BOSS_PROBE_NAME
    ok_run    = returncode == 0 and src_probe.is_file()

    dst_dir = mito2_root / "1web_scraper_01" / "outputs"
    dst_dir.mkdir(parents=True, exist_ok=True)
    probe_rel: str | None = None
    copied_md: list[str] = []

    if src_probe.is_file():
        dst_probe = dst_dir / BOSS_PROBE_NAME
        try:
            if src_probe.resolve() != dst_probe.resolve():
                shutil.copy2(src_probe, dst_probe)
        except OSError:
            shutil.copy2(src_probe, dst_probe)
        probe_rel = str(dst_probe.relative_to(mito2_root)).replace("\\", "/")

        # Mirror latest numbered + canonical markdown
        mds = sorted(src_dir.glob("BossDB*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        if mds:
            latest = mds[0]
            dst_md = dst_dir / BOSS_REPORT_NAME
            try:
                if latest.resolve() != dst_md.resolve():
                    shutil.copy2(latest, dst_md)
            except OSError:
                shutil.copy2(latest, dst_md)
            copied_md.append(str(dst_md.relative_to(mito2_root)).replace("\\", "/"))

    if not ok_run:
        err_hint = f"agent exit {returncode}" if returncode != 0 else "BossDB.probe.json missing after run"
        return {
            "ok": False,
            "error": "bossdb_agent_failed",
            "message": err_hint,
            "stdout": merged_out[-12000:],
            "stderr": "",
            "probe_path": probe_rel,
            "markdown_copied": copied_md,
        }

    if log_line:
        log_line(
            f"[mito2] BossDB agent finished (exit {returncode}). "
            f"Mirrored {BOSS_PROBE_NAME} and {BOSS_REPORT_NAME} (if present) into mitoFoundation2 outputs.\n"
        )

    return {
        "ok": True,
        "mito_foundation_bridge": True,
        "stdout": merged_out[-8000:],
        "stderr": "",
        "probe_path": probe_rel,
        "markdown_copied": copied_md,
        "source_probe": str(src_probe.relative_to(repo)).replace("\\", "/"),
    }


def run_provider_deep_scrape(
    provider_name: str,
    mito2_root: Path,
    *,
    workspace_slug: str,
    log_line: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
    scrape_session_id: str | None = None,
    subprocess_llm_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Provider-agnostic deep-scrape dispatcher."""
    name_l = (provider_name or "").strip().lower()
    if name_l == "openorganelle":
        return run_openorganelle_deep_scrape(
            mito2_root,
            workspace_slug=workspace_slug,
            log_line=log_line,
            cancel_event=cancel_event,
            scrape_session_id=scrape_session_id,
            subprocess_llm_env=subprocess_llm_env,
        )
    if name_l == "bossdb":
        return run_bossdb_deep_scrape(
            mito2_root,
            workspace_slug=workspace_slug,
            log_line=log_line,
            cancel_event=cancel_event,
            scrape_session_id=scrape_session_id,
            subprocess_llm_env=subprocess_llm_env,
        )
    return {
        "ok": False,
        "error": "provider_not_implemented",
        "message": (
            f"Deep scrape for provider {provider_name!r} is not yet implemented. "
            "Falling back to generic HTTP scraper."
        ),
    }


def probe_datasets_count(probe_path: Path) -> int:
    if not probe_path.is_file():
        return 0
    try:
        data = json.loads(probe_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    ds = data.get("datasets")
    if isinstance(ds, dict):
        return len(ds)
    if isinstance(ds, list):
        return len(ds)
    return 0
