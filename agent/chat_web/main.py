"""ASGI entry: ``python -m agent.chat_web.main`` or ``mito2`` from the repo root (no PYTHONPATH needed)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Repo root (parent of ``agent/``) — so ``agent.orchestration`` imports work without PYTHONPATH.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import uvicorn


def main() -> None:
    host = os.getenv("MITO2_HOST", "127.0.0.1")
    port = int(os.getenv("MITO2_PORT", "8765"))
    access_log = os.getenv("MITO2_ACCESS_LOG", "0") == "1"
    log_level = (os.getenv("MITO2_LOG_LEVEL", "info") or "info").strip().lower()
    if log_level not in ("critical", "error", "warning", "info", "debug", "trace"):
        log_level = "info"
    try:
        graceful = int(os.getenv("MITO2_GRACEFUL_SHUTDOWN_SEC", "5").strip() or "5")
    except ValueError:
        graceful = 5
    graceful = max(0, min(graceful, 300))
    try:
        uvicorn.run(
            "agent.chat_web.app.routes:app",
            host=host,
            port=port,
            factory=False,
            reload=os.getenv("MITO2_RELOAD", "0") == "1",
            access_log=access_log,
            log_level=log_level,
            # Shorter default than uvicorn's long wait — fewer SIGINT / CancelledError stack traces on Ctrl+C.
            timeout_graceful_shutdown=graceful,
        )
    except KeyboardInterrupt:
        # Normal stop when the user presses Ctrl+C once after shutdown completes.
        raise SystemExit(0) from None


if __name__ == "__main__":
    main()
