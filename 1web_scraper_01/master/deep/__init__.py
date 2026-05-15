"""Scrape pipeline: config, HTTP fetch, Playwright, URL/signal helpers.

Used by ``agent.py`` at the project root. Run the agent as::

    python agent.py
"""
from __future__ import annotations

__all__ = [
    "config",
    "fetch",
    "playwright",
    "signals",
    "websites",
]
