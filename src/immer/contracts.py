from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, Protocol, runtime_checkable


class ExecutionStatus(StrEnum):
    OK = "ok"
    ABSTAINED = "abstained"
    UNAVAILABLE = "unavailable"
    REJECTED = "rejected"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class Request:
    capability: str
    payload: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.capability, str) or not self.capability.strip():
            raise ValueError("capability must be a non-empty string")


@dataclass(frozen=True, slots=True)
class Result:
    status: ExecutionStatus
    component: str
    output: Any = None
    reason: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is ExecutionStatus.OK


@runtime_checkable
class Component(Protocol):
    name: str
    capabilities: frozenset[str]

    def handle(self, request: Request) -> Result: ...
