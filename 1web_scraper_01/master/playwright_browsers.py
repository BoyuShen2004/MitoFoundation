"""Playwright ships driver code on PyPI; browser binaries are a separate download.

We run the official installer programmatically so users only need ``pip install -r requirements.txt``.
If Chromium is already present under Playwright’s browser cache, we skip ``install`` entirely (no
re-download, no log noise). Set ``MITO2_VERBOSE_PLAYWRIGHT_INSTALL=1`` to log install output.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

_ensured: bool = False

# Emitted by Playwright when /etc/os-release is not Ubuntu/Debian/etc. (e.g. RHEL/Rocky/Alma);
# it still downloads a Linux Chromium build (often ubuntu24.04-x64) that usually works.
_PLAYWRIGHT_BEWARE = "BEWARE: your OS is not officially supported by Playwright"


def _filter_playwright_install_text(text: str) -> str:
    if not text:
        return text
    lines = [ln for ln in text.splitlines() if _PLAYWRIGHT_BEWARE not in ln]
    out = "\n".join(lines)
    if text.endswith("\n"):
        out += "\n"
    return out


def _playwright_browser_roots() -> list[Path]:
    roots: list[Path] = []
    env = (os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or "").strip()
    if env:
        roots.append(Path(env).expanduser())
    roots.append(Path.home() / ".cache" / "ms-playwright")
    return roots


def _chromium_layout_usable(browser_dir: Path) -> bool:
    """True if this looks like a Playwright Chromium bundle with a runnable browser binary."""
    if not browser_dir.is_dir():
        return False
    candidates = [
        browser_dir / "chrome-linux" / "chrome",
        browser_dir / "chrome-linux" / "headless_shell",
        browser_dir / "chrome-mac" / "Chromium.app" / "Contents" / "MacOS" / "Chromium",
        browser_dir / "chrome-win" / "chrome.exe",
        browser_dir / "chrome-win" / "headless_shell.exe",
    ]
    return any(p.is_file() and os.access(p, os.X_OK) for p in candidates)


def _playwright_chromium_already_on_disk() -> bool:
    """Avoid running ``playwright install chromium`` when the cache already has a build."""
    for root in _playwright_browser_roots():
        if not root.is_dir():
            continue
        try:
            for child in root.iterdir():
                if not child.is_dir():
                    continue
                name = child.name
                if name.startswith("chromium-") or name.startswith("chromium_headless_shell-"):
                    if _chromium_layout_usable(child):
                        return True
        except OSError:
            continue
    return False


def ensure_playwright_chromium(log_line: Callable[[str], None] | None = None) -> None:
    """Ensure Chromium is available: skip if already cached; else ``python -m playwright install chromium``."""
    global _ensured
    if _ensured:
        return
    if os.environ.get("MITO2_SKIP_PLAYWRIGHT_INSTALL", "").strip().lower() in ("1", "true", "yes"):
        _ensured = True
        return
    try:
        __import__("playwright")
    except ImportError:
        _ensured = True
        return

    verbose = os.environ.get("MITO2_VERBOSE_PLAYWRIGHT_INSTALL", "").strip().lower() in ("1", "true", "yes")

    if _playwright_chromium_already_on_disk():
        _ensured = True
        if verbose and log_line:
            log_line("[mito2] Playwright Chromium already present — skipping install.\n")
        return

    if verbose and log_line:
        log_line("[mito2] Installing Playwright Chromium (bundled download, not on PyPI)…\n")
    proc = subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        capture_output=True,
        text=True,
        timeout=900,
    )
    raw_out = proc.stdout or ""
    raw_err = proc.stderr or ""
    if not verbose:
        raw_out = _filter_playwright_install_text(raw_out)
        raw_err = _filter_playwright_install_text(raw_err)
    if log_line:
        if verbose:
            if raw_out:
                log_line(raw_out)
            if raw_err:
                log_line(raw_err)
        elif proc.returncode != 0:
            if raw_out:
                log_line(raw_out)
            if raw_err:
                log_line(raw_err)
            log_line(f"[mito2] playwright install chromium failed (exit {proc.returncode}).\n")
            log_line(
                "[mito2] Tip: set MITO2_VERBOSE_PLAYWRIGHT_INSTALL=1 for full installer output; "
                "on RHEL/Rocky/Alma, Playwright may print a BEWARE line even when the install works.\n"
            )
    _ensured = True
