"""Lazy, capability-scoped access to FERTIG's grounded chat surface.

The vendored assistant contains optional desktop integrations.  This adapter
does not import or construct any of them at module import time and it never
creates an operating-system backend on its own.  Read-only grounded routes
are available offline; execution and learning cross the desktop boundary only
when the caller explicitly injected the corresponding backend.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import sys
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from ...contracts import ExecutionStatus, Request, Result


AssistantFactory = Callable[..., object]
ChatFactory = Callable[..., object]


_VENDOR_LOCK = threading.RLock()
_MUTATION_PREFIX = re.compile(
    r"^\s*(?:compose|combine|kombiniere|komponiere|template|parameteri[sz]e|"
    r"parameterisiere)\b",
    re.IGNORECASE,
)
_TEACH_PREFIX = re.compile(
    r"^\s*(?:(?:please|bitte)\s+)?(?:teach|learn|record|lerne|lern|lernen|"
    r"zeige\s+dir|ich\s+zeige\s+dir)\b",
    re.IGNORECASE,
)
_CORRECT_PREFIX = re.compile(
    r"^\s*(?:correct|repair|korrigiere|korrigier|repariere)\b",
    re.IGNORECASE,
)
_DO_PREFIX = re.compile(
    r"^\s*(?:(?:please|bitte)\s+)?(?:do|execute|run|perform|mach|mache|"
    r"erledige|starte|start|f(?:ü|ue)hre(?:\s+aus)?)\b",
    re.IGNORECASE,
)
_KNOWN_CHAT_STATUSES = frozenset(
    {"ok", "unknown", "ambiguous", "needs_input", "error"}
)
_STATUS_MAP = {
    "ok": ExecutionStatus.OK,
    "unknown": ExecutionStatus.ABSTAINED,
    "ambiguous": ExecutionStatus.ABSTAINED,
    "needs_input": ExecutionStatus.ABSTAINED,
    "error": ExecutionStatus.ERROR,
}


def _vendor_root() -> Path:
    return Path(__file__).resolve().parent / "_vendor"


def _module_inside(module: object, directory: Path) -> bool:
    raw = getattr(module, "__file__", None)
    if raw is None:
        return getattr(module, "__immer_vendor_root__", None) == str(directory)
    try:
        Path(raw).resolve().relative_to(directory.resolve())
    except (OSError, ValueError):
        return False
    return True


def _vendored_classes() -> tuple[type[Any], type[Any]]:
    """Load only the vendored assistant/chat package graph, once and lazily.

    FERTIG's package ``__init__`` eagerly imports its complete product surface,
    including optional macOS modules.  A minimal package shell lets Python load
    the two requested modules and their actual dependencies without executing
    that eager initializer.
    """

    vendor = _vendor_root()
    package_dir = vendor / "fertig"
    package_file = package_dir / "__init__.py"
    if not package_file.is_file():
        raise FileNotFoundError(f"broken vendored FERTIG under {vendor}")

    with _VENDOR_LOCK:
        package = sys.modules.get("fertig")
        if package is None:
            spec = importlib.util.spec_from_file_location(
                "fertig",
                package_file,
                submodule_search_locations=[str(package_dir)],
            )
            package = ModuleType("fertig")
            package.__file__ = str(package_file)
            package.__path__ = [str(package_dir)]
            package.__package__ = "fertig"
            package.__spec__ = spec
            package.__loader__ = None if spec is None else spec.loader
            package.__version__ = "1.2.0"  # type: ignore[attr-defined]
            package.__immer_vendor_root__ = str(vendor)  # type: ignore[attr-defined]
            sys.modules["fertig"] = package
        elif not _module_inside(package, vendor):
            raise ImportError(
                "a different 'fertig' package is already loaded; refusing an "
                "unreproducible grounded runtime"
            )

        assistant_module = importlib.import_module("fertig.assistant")
        chat_module = importlib.import_module("fertig.chat")
        assistant_class = getattr(assistant_module, "FertigAssistant", None)
        chat_class = getattr(chat_module, "FertigChat", None)
        if not isinstance(assistant_class, type) or not isinstance(chat_class, type):
            raise ImportError("vendored FERTIG does not expose FertigAssistant/FertigChat")
        return assistant_class, chat_class


def _resolution_value(resolution: object | None, name: str) -> object | None:
    if isinstance(resolution, Mapping):
        return resolution.get(name)
    return getattr(resolution, name, None)


def _safe_value(value: object, *, depth: int = 0) -> Any:
    """Make evidence suitable for JSON/status feeds without losing facts."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if depth >= 8:
        return repr(value)
    if isinstance(value, Mapping):
        return {
            str(key): _safe_value(item, depth=depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_safe_value(item, depth=depth + 1) for item in value)
    return repr(value)


class FertigGrounded:
    """Grounded FERTIG chat as one lazy IMMER component.

    ``state_dir`` is deliberately mandatory: learned tasks and templates must
    never drift into an implicit working-directory or home-directory store.
    Desktop execution, recording and store management are opt-in boundaries.
    """

    name = "fertig.grounded"
    capabilities = frozenset({"grounded_chat"})

    def __init__(
        self,
        state_dir: str | Path,
        *,
        graph_path: str | Path | None = None,
        desktop: object | None = None,
        recorder: object | None = None,
        allow_mutations: bool = False,
        assistant_factory: AssistantFactory | None = None,
        chat_factory: ChatFactory | None = None,
    ) -> None:
        if not isinstance(state_dir, (str, Path)) or not str(state_dir).strip():
            raise ValueError("state_dir must be an explicit non-empty path")
        state = Path(state_dir).expanduser().resolve()
        if state.exists() and not state.is_dir():
            raise NotADirectoryError(f"FERTIG state_dir is not a directory: {state}")

        self.state_dir = state
        self.task_store_path = state / "desktop_tasks.json"
        self.template_store_path = state / "desktop_templates.json"
        self.recordings_dir = state / "desktop_recordings"
        self.graph_path = (
            None if graph_path is None else Path(graph_path).expanduser().resolve()
        )
        self.desktop = desktop
        self.recorder = recorder
        self.allow_mutations = bool(allow_mutations)
        self._assistant_factory = assistant_factory
        self._chat_factory = chat_factory
        self._assistant: object | None = None
        self._chat: object | None = None
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        """Whether the vendored runtime has been constructed already."""

        return self._chat is not None

    def _ensure_runtime(self) -> tuple[object, object]:
        with self._lock:
            if self._assistant is not None and self._chat is not None:
                return self._assistant, self._chat

            assistant_class: AssistantFactory
            chat_class: ChatFactory
            if self._assistant_factory is None or self._chat_factory is None:
                vendored_assistant, vendored_chat = _vendored_classes()
                assistant_class = self._assistant_factory or vendored_assistant
                chat_class = self._chat_factory or vendored_chat
            else:
                assistant_class = self._assistant_factory
                chat_class = self._chat_factory

            assistant = assistant_class(
                store=self.task_store_path,
                template_store=self.template_store_path,
                recordings_dir=self.recordings_dir,
            )
            chat_kwargs: dict[str, object] = {}
            if self.graph_path is not None:
                chat_kwargs["graph_path"] = self.graph_path
            chat = chat_class(assistant, **chat_kwargs)
            if not callable(getattr(chat, "handle", None)):
                raise TypeError("FERTIG chat must expose handle(text, desktop, recorder)")

            # Publish both only after construction succeeded.  A failed lazy
            # load can therefore be retried after its environment is repaired.
            self._assistant = assistant
            self._chat = chat
            return assistant, chat

    def _resolve_intent(self, assistant: object, text: str) -> tuple[str, str | None]:
        intent = "unknown"
        task: str | None = None
        resolver = getattr(assistant, "resolve", None)
        if callable(resolver):
            try:
                resolution = resolver(text)
            except Exception:  # injected language resolvers are untrusted boundaries
                resolution = None
            raw_intent = _resolution_value(resolution, "intent")
            raw_task = _resolution_value(resolution, "task")
            if raw_intent is not None:
                intent = str(raw_intent)
            if raw_task is not None:
                task = str(raw_task)

        # Independent lexical guards keep the safety boundary intact even if
        # a custom language resolver abstains or is temporarily unavailable.
        if _TEACH_PREFIX.search(text):
            intent = "teach"
        elif _CORRECT_PREFIX.search(text):
            intent = "correct"
        elif _MUTATION_PREFIX.search(text):
            match = _MUTATION_PREFIX.search(text)
            word = match.group(0).strip().casefold() if match else ""
            intent = "template" if word.startswith(("template", "parameter")) else "compose"
        elif _DO_PREFIX.search(text):
            intent = "do"
        return intent, task

    def _blocked_result(
        self,
        *,
        intent: str,
        task: str | None,
        missing: str,
    ) -> Result:
        if missing == "desktop":
            message = "desktop execution requires an explicitly injected desktop backend"
        elif missing == "recorder":
            message = "teaching or correction requires an explicitly injected recorder backend"
        else:
            message = "task/template mutation requires allow_mutations=True"
        return Result(
            ExecutionStatus.ABSTAINED,
            self.name,
            reason=message,
            evidence={
                "route": "desktop",
                "status": "needs_input",
                "task": task,
                "data": {"intent": intent, "missing_backend": missing},
            },
        )

    def _guard_action(self, intent: str, task: str | None) -> Result | None:
        if intent == "do" and self.desktop is None:
            return self._blocked_result(intent=intent, task=task, missing="desktop")
        if intent in {"teach", "correct"} and self.recorder is None:
            return self._blocked_result(intent=intent, task=task, missing="recorder")
        if intent in {"compose", "template"} and not self.allow_mutations:
            return self._blocked_result(intent=intent, task=task, missing="mutation_opt_in")
        return None

    def _map_reply(self, reply: object) -> Result:
        raw_status = str(getattr(reply, "status", "error"))
        status = raw_status if raw_status in _KNOWN_CHAT_STATUSES else "error"
        route = str(getattr(reply, "route", "unknown"))
        raw_task = getattr(reply, "task", None)
        task = None if raw_task is None else str(raw_task)
        text = str(getattr(reply, "text", "FERTIG returned no response text."))
        raw_data = getattr(reply, "data", {})
        data = _safe_value(raw_data) if isinstance(raw_data, Mapping) else {}
        evidence = {
            "route": route,
            "status": status,
            "task": task,
            "data": data,
        }
        mapped = _STATUS_MAP[status]
        return Result(
            mapped,
            self.name,
            output=text if mapped is ExecutionStatus.OK else None,
            reason=None if mapped is ExecutionStatus.OK else text,
            evidence=evidence,
        )

    def handle(self, request: Request) -> Result:
        if request.capability not in self.capabilities:
            return Result(ExecutionStatus.REJECTED, self.name, reason="unsupported capability")
        if not isinstance(request.payload, str) or not request.payload.strip():
            return Result(
                ExecutionStatus.REJECTED,
                self.name,
                reason="grounded_chat payload must be non-empty text",
            )

        text = request.payload.strip()
        try:
            assistant, chat = self._ensure_runtime()
        except (FileNotFoundError, ImportError, ModuleNotFoundError) as exc:
            return Result(ExecutionStatus.UNAVAILABLE, self.name, reason=str(exc))
        except Exception as exc:  # construction failures are contained at the capability boundary
            return Result(
                ExecutionStatus.UNAVAILABLE,
                self.name,
                reason=f"FERTIG grounded runtime unavailable: {type(exc).__name__}: {exc}",
            )

        intent, task = self._resolve_intent(assistant, text)
        blocked = self._guard_action(intent, task)
        if blocked is not None:
            return blocked

        try:
            reply = chat.handle(text, desktop=self.desktop, recorder=self.recorder)
            return self._map_reply(reply)
        except Exception as exc:
            return Result(
                ExecutionStatus.ERROR,
                self.name,
                reason=f"FERTIG grounded chat failed safely: {type(exc).__name__}: {exc}",
                evidence={
                    "route": "unknown",
                    "status": "error",
                    "task": task,
                    "data": {"intent": intent, "error": type(exc).__name__},
                },
            )


__all__ = ["FertigGrounded"]
