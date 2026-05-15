"""Default caps for concise appendix sections (override with APPENDIX_* env vars)."""

from __future__ import annotations

import os

# Defaults tuned for short reports: name, spacing/size, annotations/masks.
DEFAULTS = {
    "APPENDIX_DATASET_ROWS": 500,
    "APPENDIX_SPATIAL_ROWS_CAP": 120,
    "APPENDIX_DATASET_NAME_LIST_CAP": 100,
    "APPENDIX_UI_SEGMENTATION_ROWS": 200,
    "APPENDIX_DOM_LABELS": 0,
    "APPENDIX_DOM_URLS": 0,
    "APPENDIX_XHR_URL_CAP": 0,
}


def appendix_cap(name: str) -> int:
    """Read ``name`` from env or use concise default from ``DEFAULTS``."""
    raw = os.getenv(name)
    if raw is not None and str(raw).strip():
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return DEFAULTS.get(name, 0)
