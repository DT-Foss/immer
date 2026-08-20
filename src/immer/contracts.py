"""Small, dependency-free contracts shared by the integration adapters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping


class BackendStatus(StrEnum):
    VERIFIED = "verified"
    ABSTAINED = "abstained"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    HELD = "held_for_evidence"


@dataclass(frozen=True)
class SolveRequest:
    question: str
    capability: str = "exact_math"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must be a non-empty string")
        if not self.capability.strip():
            raise ValueError("capability must be non-empty")


@dataclass(frozen=True)
class SolveResult:
    status: BackendStatus
    backend: str
    answer: str | None = None
    reason: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @property
    def abstained(self) -> bool:
        return self.status in {
            BackendStatus.ABSTAINED,
            BackendStatus.UNAVAILABLE,
            BackendStatus.HELD,
        }

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        value["evidence"] = dict(self.evidence)
        return value


@dataclass(frozen=True)
class CapabilityRoute:
    capability: str
    backend: str
    status: BackendStatus
    reason: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        value["evidence"] = dict(self.evidence)
        return value


@dataclass(frozen=True)
class SourceRef:
    name: str
    purpose: str
    local_path: str
    upstream_url: str | None
    revision: str | None
    status: str
    inclusion: str
    notes: str = ""


def canonical_json(value: Any) -> bytes:
    """Serialize manifest values deterministically for digest binding."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
