"""Council: several brains deliberate, one answer leaves the room.

BO3 pattern from the measured verifier line (79.4 -> 86.5 %): pose the same
question to k members, count agreeing successful answers, abstain when no
majority forms. Abstention is a result, not a failure.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .contracts import Component, ExecutionStatus, Request, Result


class Council:
    """Deliberation over registered components with majority judgment."""

    def __init__(self, members: tuple[Component, ...] = (), *, name: str = "council") -> None:
        if not members:
            raise ValueError("a council without members cannot deliberate")
        self.members = members
        self.name = name
        self.capabilities = frozenset().union(*(m.capabilities for m in members))

    def deliberate(self, capability: str, payload: Any, metadata: dict[str, Any] | None = None) -> Result:
        votes: list[Result] = []
        for member in self.members:
            if capability not in member.capabilities:
                continue
            try:
                result = member.handle(Request(capability, payload, metadata=metadata or {}))
            except Exception as exc:  # noqa: BLE001 - one brain failing must not kill the vote
                votes.append(Result(ExecutionStatus.ERROR, member.name, reason=f"{type(exc).__name__}: {exc}"))
                continue
            votes.append(result)
        if not votes:
            return Result(
                ExecutionStatus.UNAVAILABLE,
                self.name,
                reason=f"no member holds capability {capability!r}",
            )
        ok_votes = [v for v in votes if v.status is ExecutionStatus.OK]
        if not ok_votes:
            reasons = "; ".join(f"{v.component}: {v.reason}" for v in votes[:3])
            return Result(ExecutionStatus.ABSTAINED, self.name, reason=reasons or "all members abstained")
        counts = Counter(_freeze(v.output) for v in ok_votes)
        answer, count = counts.most_common(1)[0]
        return Result(
            ExecutionStatus.OK,
            self.name,
            output=answer,
            evidence={
                "votes": len(votes),
                "agree": count,
                "members": [v.component for v in ok_votes],
            },
        )

    def handle(self, request: Request) -> Result:
        if request.capability not in self.capabilities:
            return Result(ExecutionStatus.REJECTED, self.name, reason="unsupported capability")
        return self.deliberate(request.capability, request.payload)


def _freeze(value: Any) -> str:
    return value if isinstance(value, str) else repr(value)
