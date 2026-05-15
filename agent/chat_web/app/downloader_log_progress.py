"""Shared Stage-3 downloader stdout/stderr line parsing for progress UI.

Used by :func:`studio_api._run_downloader_command_live` and the SSE script
stream so OpenOrganelle, BossDB, and any future provider share one parser.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from typing import Any


class DownloaderLogProgressParser:
    """Parse ``[PLAN]``, ``[PROGRESS]``, ``[DONE]``, and summary lines into progress payloads."""

    __slots__ = ("_set_progress", "_lock", "_state", "_done_dataset_keys", "_skip_n5")

    def __init__(self, set_progress: Callable[[dict[str, Any]], None] | None) -> None:
        self._set_progress = set_progress
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "planned_total": 0,
            "windows_per_dataset": 1,
            "completed": 0,
            "active": 0,
            "dataset": "",
        }
        self._done_dataset_keys: set[str] = set()
        self._skip_n5 = {"on": False}

    def filter_noise_line(self, line: str) -> bool:
        """Return True if *line* should be dropped (not appended to logs)."""
        if "FutureWarning" in line and "N5FSStore is deprecated" in line:
            self._skip_n5["on"] = True
            return True
        if self._skip_n5["on"] and "DEFAULT_N5_STORE" in line:
            self._skip_n5["on"] = False
            return True
        if self._skip_n5["on"] and line.lstrip().startswith("store = "):
            return True
        self._skip_n5["on"] = False
        return False

    def consume_line(self, line: str) -> None:
        """Update internal counters and invoke *set_progress* when a milestone line matches."""
        if self._set_progress is None:
            return
        m_plan = re.search(r"(?mi)^\s*\[PLAN\]\s*(\d+)\s+of\s+(\d+)\s+dataset", line)
        if m_plan:
            with self._lock:
                self._state["planned_total"] = max(
                    int(self._state["planned_total"]),
                    max(0, int(m_plan.group(1))),
                )
        m_ds = re.search(r"(?mi)^\s*-\s*Datasets:\s*(\d+)(?:\s+pending)?(?:\s+\(of\s+\d+\s+total\))?\s*$", line)
        if m_ds:
            with self._lock:
                self._state["planned_total"] = max(
                    int(self._state["planned_total"]),
                    max(0, int(m_ds.group(1))),
                )
        m_windows = re.search(r"(?mi)^\s*-\s*Windows per dataset:\s*(\d+)\b", line)
        if m_windows:
            with self._lock:
                self._state["windows_per_dataset"] = max(1, int(m_windows.group(1)))
        m_pairs = re.search(r"(?mi)^\s*-\s*Planned image/label pairs:\s*(\d+)\s*$", line)
        if m_pairs:
            with self._lock:
                self._state["planned_total"] = max(
                    int(self._state["planned_total"]),
                    max(0, int(m_pairs.group(1))),
                )
                planned_total = int(self._state["planned_total"])
                completed = int(self._state["completed"])
                current = int(self._state["active"]) if int(self._state["active"]) > 0 else (1 if planned_total > 0 else 0)
                ds = str(self._state["dataset"] or "")
            if planned_total > 0 and current > 0:
                self._set_progress(
                    {
                        "completed": min(completed, planned_total),
                        "total": planned_total,
                        "current": min(max(1, current), planned_total),
                        "dataset": ds,
                    }
                )
        m_prog = re.search(r"\[PROGRESS\]\s+dataset\s+(\d+)/(\d+):\s*(.+?)(?:\s+\(pairs=(\d+)\))?\s*$", line)
        if not m_prog:
            m_prog = re.search(r"\[PROGRESS\]\s+(\d+)/(\d+):\s*(.+?)\s*$", line)
        if m_prog:
            cur = int(m_prog.group(1))
            total = int(m_prog.group(2))
            ds = m_prog.group(3).strip()
            with self._lock:
                effective_total = max(
                    int(total),
                    int(self._state["planned_total"]),
                    int(self._state["completed"]),
                )
                current = (
                    min(int(self._state["completed"]) + 1, effective_total)
                    if effective_total > 0
                    else max(1, int(cur))
                )
                self._state["active"] = current
                self._state["dataset"] = ds
                completed = int(self._state["completed"])
            self._set_progress(
                {
                    "completed": completed,
                    "total": effective_total,
                    "current": current,
                    "dataset": ds,
                }
            )
            return

        m_done = re.search(
            r"\[DONE\]\s+dataset\s+(\d+)/(\d+):\s*(.+?)(?:\s+\(pairs=(\d+)\))?(?:\s+\[[^\]]+\])?\s*$",
            line,
        )
        if m_done:
            total = int(m_done.group(2))
            ds = m_done.group(3).strip()
            pairs_for_dataset = int(m_done.group(4)) if m_done.group(4) else 0
            ds_key = ds.strip().lower()
            with self._lock:
                if ds_key not in self._done_dataset_keys:
                    self._done_dataset_keys.add(ds_key)
                    step = max(1, int(pairs_for_dataset or self._state["windows_per_dataset"]))
                    self._state["completed"] = int(self._state["completed"]) + step
                effective_total = max(
                    int(total),
                    int(self._state["planned_total"]),
                    int(self._state["completed"]),
                )
                current = min(int(self._state["completed"]) + 1, max(1, effective_total))
                self._state["active"] = current
                self._state["dataset"] = ds
                completed = int(self._state["completed"])
            self._set_progress(
                {
                    "completed": completed,
                    "total": effective_total,
                    "current": current,
                    "dataset": ds,
                }
            )
