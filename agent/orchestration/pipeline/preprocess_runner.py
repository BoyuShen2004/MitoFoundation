"""Shared preprocessing runner: invoke Stage 4 on a download batch.

Called by Stage 3 (download-and-direct-preprocess mode) and available as a
standalone entry-point so both code paths share identical Stage 4 behaviour
without duplicating the complex preprocessing logic in agent.py.

Usage
-----
    from agent.orchestration.pipeline.preprocess_runner import run_preprocess_on_batch

    result = run_preprocess_on_batch(
        project_root=project_root(),
        download_run="openorganelle_mito_20260420_012621",
        split_label_cc=True,
    )
    if not result["ok"]:
        print(result["message"])
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from config.paths import project_root


_REPO_ROOT = project_root()


def run_preprocess_on_batch(
    *,
    project_root: Path | None = None,
    download_run: str,
    split_label_cc: bool = True,
    timeout_sec: float = 3600.0,
    extra_args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Invoke ``3data_downloader/downloader_master/preprocess_agent.py`` on a completed download batch.

    Parameters
    ----------
    project_root:
        Repo root (default: inferred from this file's location).
    download_run:
        Folder name under ``data/raw/`` (e.g. ``openorganelle_mito_20260420_012621``).
    split_label_cc:
        Pass ``--split-label-cc`` to stage 4 (default True).
    timeout_sec:
        Subprocess wall-clock timeout (default 1 hour).
    extra_args:
        Additional CLI arguments forwarded verbatim to stage 4 agent.py.
    env:
        Extra environment variables merged onto ``os.environ``.

    Returns
    -------
    dict with keys:
        ok (bool), returncode (int), stdout (str), stderr (str), message (str)
    """
    import os

    root = (project_root or _REPO_ROOT).resolve()
    stage4_dir = root / "3data_downloader"
    agent_py = stage4_dir / "downloader_master/preprocess_agent.py"

    if not agent_py.is_file():
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "",
            "message": f"Stage 4 agent not found: {agent_py}",
        }

    py = sys.executable
    argv: list[str] = [py, str(agent_py), "--raw-run-subfolder", download_run]
    if split_label_cc:
        argv.append("--split-label-cc")
    if extra_args:
        argv.extend(extra_args)

    merged_env = {**os.environ}
    if env:
        merged_env.update(env)

    try:
        proc = subprocess.run(
            argv,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=merged_env,
        )
        ok = proc.returncode == 0
        return {
            "ok": ok,
            "returncode": proc.returncode,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
            "message": (
                f"Preprocess completed for run '{download_run}'."
                if ok
                else f"Preprocess failed for run '{download_run}' (exit {proc.returncode})."
            ),
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "",
            "message": f"Preprocess timed out after {timeout_sec:.0f}s for run '{download_run}'.",
        }
    except Exception as exc:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
            "message": f"Preprocess runner error: {exc}",
        }
