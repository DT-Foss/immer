"""LifeDaemon: continuity, cold organs, registered services.

The daemon keeps a life stream running across restarts (state port),
mounts organs on demand over the OrganBank, and exposes registered
components (e.g. the FERTIG exact solver) as callable services. It does
NOT decide when any of this happens — it makes it possible.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Protocol

from ..contracts import Component, ExecutionStatus, Request, Result
from ..capabilities.organbank.bank import DigestMismatch, OrganBank
from .bus import EventBus, Event, Priority


class LifeStream(Protocol):
    """Anything that ingests experience and can snapshot itself."""

    def observe(self, text: str) -> None: ...
    def snapshot(self) -> Mapping[str, Any]: ...


class RestorableLifeStream(LifeStream, Protocol):
    """A life stream that can be rebuilt from its own snapshot."""

    def restore(self, state: Mapping[str, Any]) -> None: ...


class LifeStatePort:
    """Restart-safe persistence of the life state as one JSON document."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    def save(self, state: Mapping[str, Any]) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        document = {"saved_at": time.time(), "state": dict(state)}
        encoded = json.dumps(document, ensure_ascii=False, sort_keys=True).encode("utf-8")
        fd, temporary = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return self.path

    def load(self) -> dict[str, Any] | None:
        if not self.path.is_file():
            return None
        document = json.loads(self.path.read_text(encoding="utf-8"))
        return dict(document.get("state", {}))


class OrganRack:
    """Cold organ mounting over an OrganBank.

    Mounting verifies the artifact digest every single time — a mounted
    organ is a *verified* organ. The choice of which organ to mount stays
    with the organism; the rack only makes mounting cheap and safe.
    """

    def __init__(self, bank: OrganBank | None = None) -> None:
        self.bank = bank
        self._mounted: dict[str, Path] = {}

    @property
    def available(self) -> bool:
        return self.bank is not None

    def mount(self, name: str) -> Path:
        if self.bank is None:
            raise RuntimeError("no OrganBank attached")
        artifact = self.bank.verify(name)
        self._mounted[name] = artifact
        return artifact

    def unmount(self, name: str) -> None:
        self._mounted.pop(name, None)

    def mounted(self) -> tuple[str, ...]:
        return tuple(sorted(self._mounted))

    def digest_conflicts_are_fatal(self) -> bool:
        """A tampered organ must never mount silently."""
        return True


class LifeDaemon:
    """Owns the bus, the stream, the rack and the services. Survives restarts."""

    def __init__(
        self,
        *,
        stream: LifeStream | None = None,
        bank: OrganBank | None = None,
        state_path: str | Path | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self.bus = bus or EventBus()
        self.stream = stream
        self.rack = OrganRack(bank)
        self.port = LifeStatePort(state_path) if state_path is not None else None
        self.services: dict[str, Component] = {}
        self.outbox: list[Event] = []
        self.turns: int = 0
        self._restore()

    # -- services ---------------------------------------------------------

    def register(self, component: Component) -> None:
        for capability in component.capabilities:
            owner = self.services.get(capability)
            if owner is not None and owner is not component:
                raise ValueError(
                    f"capability {capability!r} already owned by {owner.name!r}"
                )
            self.services[capability] = component

    def request(
        self,
        capability: str,
        payload: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Result:
        component = self.services.get(capability)
        if component is None:
            return Result(ExecutionStatus.UNAVAILABLE, "immer", reason=f"no service for {capability!r}")
        try:
            return component.handle(Request(capability, payload, metadata or {}))
        except Exception as exc:  # noqa: BLE001 - substrate never crashes the life
            return Result(ExecutionStatus.ERROR, component.name, reason=f"{type(exc).__name__}: {exc}")

    # -- life -------------------------------------------------------------

    def submit_user(self, text: str) -> Event:
        """A user utterance is one event in the organism's life."""
        event = self.bus.publish(
            "life",
            "user_message",
            text,
            priority=Priority.USER,
        )
        if self.stream is not None:
            self.stream.observe(text)
        self.turns += 1
        self._persist()
        return event

    def impulse(self, kind: str, payload: Any = None) -> Event:
        """An internal impulse (surprise, curiosity, consolidation urge)."""
        return self.bus.publish("life", kind, payload, priority=Priority.INTERNAL)

    def say(self, kind: str, payload: Any = None) -> Event:
        """The organism emits something towards a channel."""
        event = self.bus.publish("speech", kind, payload, priority=Priority.INTERNAL)
        self.outbox.append(event)
        return event

    # -- continuity -------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return {
            "turns": self.turns,
            "mounted_organs": list(self.rack.mounted()),
            "stream": dict(self.stream.snapshot()) if self.stream is not None else {},
        }

    def _persist(self) -> None:
        if self.port is not None:
            self.port.save(self.snapshot())

    def _restore(self) -> None:
        if self.port is None:
            return
        state = self.port.load()
        if state is None:
            return
        self.turns = int(state.get("turns", 0))
        stream_state = state.get("stream")
        if isinstance(stream_state, dict) and hasattr(self.stream, "restore"):
            self.stream.restore(stream_state)  # type: ignore[attr-defined]
        for name in state.get("mounted_organs", ()):
            try:
                self.rack.mount(name)
            except (DigestMismatch, FileNotFoundError, KeyError):
                # A lost or tampered organ is unmounted after restart;
                # the organism re-mounts what it still needs.
                continue
