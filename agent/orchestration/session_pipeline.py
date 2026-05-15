"""Linear pipeline steps and session progress (mirrors the product workflow)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, List, Optional


class PipelineStep(IntEnum):
    IDLE = 0
    SCRAPE = 1
    DATABASE = 2
    DOWNLOAD_SCRIPT = 3
    PREPROCESS = 4
    SSL = 5
    FINETUNE = 6
    EVAL = 7


STEP_LABELS: dict[int, str] = {
    PipelineStep.IDLE: "idle",
    PipelineStep.SCRAPE: "scrape",
    PipelineStep.DATABASE: "database_build",
    PipelineStep.DOWNLOAD_SCRIPT: "download_script",
    PipelineStep.PREPROCESS: "preprocess",
    PipelineStep.SSL: "model_training",
    PipelineStep.FINETUNE: "model_training",
    PipelineStep.EVAL: "eval",
}


@dataclass
class PipelineSession:
    """Mutable session state for one browser / API client."""

    current_step: PipelineStep = PipelineStep.IDLE
    last_url: str = ""
    last_site_stem: str = ""
    history: List[dict[str, Any]] = field(default_factory=list)

    def log(self, kind: str, payload: dict[str, Any]) -> None:
        self.history.append({"kind": kind, **payload})

    def advance_after(self, completed: PipelineStep) -> None:
        if completed == PipelineStep.EVAL:
            self.current_step = PipelineStep.EVAL
            return
        self.current_step = PipelineStep(int(completed) + 1)


def default_sessions() -> dict[str, PipelineSession]:
    return {}
