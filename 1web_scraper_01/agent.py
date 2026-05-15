"""
Compatibility entrypoint for the web scraper.

The master dispatcher lives in ``1web_scraper_01/master/agent.py``.
"""

from __future__ import annotations

from master.agent import main


if __name__ == "__main__":
    main()
