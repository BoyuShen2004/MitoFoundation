"""Human-in-the-loop approvals before subprocess or destructive steps."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from agent.orchestration.time_utils import now_us_eastern_iso


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class PendingAction:
    id: str
    title: str
    command: List[str]
    cwd: str
    created_at: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    detail: Dict[str, Any] = field(default_factory=dict)


class ApprovalQueue:
    def __init__(self) -> None:
        self._items: Dict[str, PendingAction] = {}

    def propose(
        self,
        *,
        title: str,
        command: List[str],
        cwd: str,
        detail: Optional[Dict[str, Any]] = None,
    ) -> PendingAction:
        pid = str(uuid.uuid4())
        action = PendingAction(
            id=pid,
            title=title,
            command=list(command),
            cwd=cwd,
            created_at=now_us_eastern_iso(),
            detail=detail or {},
        )
        self._items[pid] = action
        return action

    def get(self, action_id: str) -> Optional[PendingAction]:
        return self._items.get(action_id)

    def resolve(self, action_id: str, approved: bool) -> Optional[PendingAction]:
        a = self._items.get(action_id)
        if not a or a.status != ApprovalStatus.PENDING:
            return None
        a.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        return a

    def pending_list(self) -> List[PendingAction]:
        return [x for x in self._items.values() if x.status == ApprovalStatus.PENDING]
