"""Event bus with priority channels.

User events preempt internal impulses, but both flow through the same
bus: the organism's own surprises are first-class citizens of its life,
not second-class notifications.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable


class Priority(IntEnum):
    """Lower value = processed first. USER preempts INTERNAL."""

    USER = 0
    INTERNAL = 10


@dataclass(frozen=True, slots=True)
class Event:
    channel: str
    kind: str
    payload: Any = None
    priority: Priority = Priority.INTERNAL
    seq: int = 0
    timestamp: float = field(default_factory=time.time)


Handler = Callable[[Event], None]


class EventBus:
    """Synchronous dispatch in (priority, sequence) order."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[tuple[Priority, int, Handler]]] = {}
        self._seq = itertools.count(1)
        self._sub_seq = itertools.count(1)

    def subscribe(self, channel: str, handler: Handler, *, priority: Priority = Priority.INTERNAL) -> Callable[[], None]:
        entry = (priority, next(self._sub_seq), handler)
        self._handlers.setdefault(channel, []).append(entry)
        self._handlers[channel].sort(key=lambda item: (item[0], item[1]))

        def unsubscribe() -> None:
            try:
                self._handlers[channel].remove(entry)
            except ValueError:
                pass

        return unsubscribe

    def publish(self, channel: str, kind: str, payload: Any = None, *, priority: Priority = Priority.INTERNAL) -> Event:
        event = Event(
            channel=channel,
            kind=kind,
            payload=payload,
            priority=priority,
            seq=next(self._seq),
        )
        for _, _, handler in tuple(self._handlers.get(channel, ())):
            handler(event)
        return event

    def listener_count(self, channel: str) -> int:
        return len(self._handlers.get(channel, ()))
