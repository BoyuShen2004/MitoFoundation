"""Compatibility shim for moved raw browser UI.

Use ``downloader_master/raw_browser.py`` as the canonical location.
"""

from __future__ import annotations

from downloader_master.raw_browser import main


if __name__ == "__main__":
    main()
