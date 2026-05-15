"""Run approved commands in the project root (subprocess, no shell)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, List


def run_command(
    argv: List[str],
    cwd: Path,
    timeout_sec: float | None = 600.0,
    *,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run ``argv`` in ``cwd``. Optional ``env`` is merged into the process environment (child only)."""
    child_env = {**os.environ, **(env or {})}
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=child_env,
        )
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "")[-8000:] if isinstance(exc.stdout, str) else ""
        err = (exc.stderr or "")[-8000:] if isinstance(exc.stderr, str) else ""
        tail = (
            f"Timed out after {timeout_sec}s (no shell): {' '.join(argv)}"
            if timeout_sec is not None
            else f"Timed out (no shell): {' '.join(argv)}"
        )
        merged = "\n".join(part for part in (err.strip(), tail) if part)
        return {
            "returncode": 124,
            "stdout": out,
            "stderr": merged,
        }
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout[-8000:],
        "stderr": proc.stderr[-8000:],
    }
