from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


class DigestMismatch(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class OrganDescriptor:
    name: str
    capability: str
    group: str
    artifact: Path
    sha256: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class OrganBank:
    """Digest-addressed registry for cold external capability artifacts."""

    def __init__(self, descriptors: tuple[OrganDescriptor, ...] = ()) -> None:
        self._by_name: dict[str, OrganDescriptor] = {}
        self._by_capability: dict[str, OrganDescriptor] = {}
        for descriptor in descriptors:
            self.register(descriptor)

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "OrganBank":
        manifest = Path(manifest_path).expanduser().resolve()
        data = json.loads(manifest.read_text(encoding="utf-8"))
        raw_organs = data.get("organs")
        if not isinstance(raw_organs, list):
            raise ValueError("organ manifest must contain an 'organs' list")
        descriptors: list[OrganDescriptor] = []
        for raw in raw_organs:
            if not isinstance(raw, dict):
                raise ValueError("each organ entry must be an object")
            artifact = Path(str(raw["artifact"]))
            if not artifact.is_absolute():
                artifact = (manifest.parent / artifact).resolve()
            descriptors.append(
                OrganDescriptor(
                    name=str(raw["name"]),
                    capability=str(raw["capability"]),
                    group=str(raw["group"]),
                    artifact=artifact,
                    sha256=str(raw["sha256"]).lower(),
                    metadata=dict(raw.get("metadata", {})),
                )
            )
        return cls(tuple(descriptors))

    def register(self, descriptor: OrganDescriptor) -> None:
        if descriptor.name in self._by_name:
            raise ValueError(f"duplicate organ name: {descriptor.name}")
        if descriptor.capability in self._by_capability:
            raise ValueError(f"duplicate organ capability: {descriptor.capability}")
        if len(descriptor.sha256) != 64 or any(c not in "0123456789abcdef" for c in descriptor.sha256):
            raise ValueError(f"invalid sha256 for organ {descriptor.name}")
        self._by_name[descriptor.name] = descriptor
        self._by_capability[descriptor.capability] = descriptor

    def resolve(self, capability: str) -> OrganDescriptor:
        try:
            return self._by_capability[capability]
        except KeyError as exc:
            raise KeyError(f"no organ registered for capability {capability!r}") from exc

    def descriptor(self, name: str) -> OrganDescriptor:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise KeyError(f"unknown organ {name!r}") from exc

    def verify(self, name: str) -> Path:
        descriptor = self.descriptor(name)
        if not descriptor.artifact.is_file():
            raise FileNotFoundError(descriptor.artifact)
        digest = hashlib.sha256(descriptor.artifact.read_bytes()).hexdigest()
        if digest != descriptor.sha256:
            raise DigestMismatch(
                f"organ {name!r} digest mismatch: expected {descriptor.sha256}, got {digest}"
            )
        return descriptor.artifact

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_name))
