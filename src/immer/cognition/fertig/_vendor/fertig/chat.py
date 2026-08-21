"""Unified conversational surface over FERTIG's grounded capabilities.

Desktop actions remain owned by :class:`fertig.assistant.FertigAssistant`.
This router only falls through to arithmetic, measured quantitative facts and
the causal graph when no learned desktop task was selected.  An injected
language layer may rank fact-preserving wording, but it never creates a tool
or changes a computed result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
import re
import threading
from typing import Any, Literal, Mapping

from .assistant import FertigAssistant
from .pipeline import DEFAULT_GRAPH


ChatRoute = Literal[
    "desktop", "list", "help", "status", "math", "quant", "graph", "unknown"
]
ChatStatus = Literal["ok", "unknown", "ambiguous", "needs_input", "error"]


@dataclass(frozen=True)
class ChatReply:
    """Typed result of one conversational request."""

    status: ChatStatus
    route: ChatRoute
    text: str
    task: str | None = None
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatTurn:
    """One immutable user/reply pair in local conversation history."""

    user: str
    reply: ChatReply


_EXPLICIT_TEACH = re.compile(
    r"^\s*(?:(?:please|bitte)\s+)?"
    r"(?:teach|learn|record|lerne|lern|lernen|zeige\s+dir)\b",
    re.IGNORECASE,
)
_EXPLICIT_DO = re.compile(
    r"^\s*(?:(?:please|bitte)\s+)?"
    r"(?:do|execute|run|perform|mach|mache|erledige|starte|"
    r"f(?:ü|ue)hre)\b",
    re.IGNORECASE,
)
_EXPLICIT_MANAGEMENT = re.compile(
    r"^\s*(?:compose|combine|kombiniere|komponiere|template|parameteri[sz]e|"
    r"parameterisiere|correct|repair|korrigiere|korrigier|repariere)\b",
    re.IGNORECASE,
)
_EXPLICIT_EXPLAIN = re.compile(
    r"\b(?:explain|describe|erkl(?:ä|ae)re?|warum|wieso|weshalb|why)\b",
    re.IGNORECASE,
)
_LIST = re.compile(
    r"\b(?:list\s+(?:tasks|skills)|what\s+can\s+you\s+do|"
    r"was\s+kannst\s+du|welche\s+aufgaben|aufgaben)\b",
    re.IGNORECASE,
)
_HELP = re.compile(r"\b(?:help|hilfe)\b", re.IGNORECASE)
_FOLLOWUP = re.compile(
    r"^\s*(?:explain|describe|erkl(?:ä|ae)re?)\s+"
    r"(?:it|this|that|es|das)\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_STATUS = re.compile(
    r"(?:\b(?:hsslm|hlssm)\b.*\b(?:status|parameter|parameters|size|"
    r"gr(?:ö|oe)sse|modell|model)\b|\b(?:status|parameter|parameters|"
    r"size|gr(?:ö|oe)sse|modell|model)\b.*\b(?:hsslm|hlssm)\b|"
    r"^\s*(?:hsslm|hlssm)\s*[?!.]*\s*$)",
    re.IGNORECASE,
)
_QUESTION_HINT = re.compile(
    r"\b(?:how|what|which|wie|wieviel|wie\s+viele|total|altogether|"
    r"insgesamt|summe|left|remaining)\b",
    re.IGNORECASE,
)

_BASE_CAPABILITIES = (
    "teach learned desktop tasks",
    "run learned desktop tasks",
    "compose learned tasks into workflows",
    "parameterize demonstrated text fields",
    "correct one learned step by showing it again",
    "explain learned desktop tasks",
    "solve grounded arithmetic questions",
    "answer measured quantitative world facts",
    "explain and inspect causal graph facts",
)


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[^\wäöüß]+", " ", text.casefold()).split())


def _german(text: str) -> bool:
    normal = _normalise(text)
    return bool(
        re.search(
            r"\b(?:bitte|warum|wieso|weshalb|erkläre|erklaere|mach|"
            r"lerne|hilfe|was|wie|aufgabe|aufgaben)\b",
            normal,
        )
        or any(character in text.casefold() for character in "äöüß")
    )


class FertigChat:
    """Single natural-language entry point over the existing FERTIG stack."""

    def __init__(
        self,
        assistant: FertigAssistant,
        graph_path: str | Path = DEFAULT_GRAPH,
    ) -> None:
        if not callable(getattr(assistant, "handle", None)):
            raise TypeError("assistant must expose handle(text, desktop, recorder)")
        self.assistant = assistant
        self.graph_path = Path(graph_path)
        self.last_task: str | None = None
        self._history: list[ChatTurn] = []
        self._lock = threading.RLock()

    @property
    def history(self) -> tuple[ChatTurn, ...]:
        return tuple(self._history)

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()
            self.last_task = None

    def handle(
        self,
        text: str,
        desktop: object | None = None,
        recorder: object | None = None,
    ) -> ChatReply:
        """Route one utterance without allowing language to invent a tool."""

        with self._lock:
            if not isinstance(text, str):
                return self._record(
                    repr(text),
                    ChatReply("error", "unknown", "text must be a string"),
                )
            source = text.strip()
            if not source:
                return self._record(
                    text,
                    ChatReply("unknown", "unknown", "I need a question or command."),
                )

            if _STATUS.search(source):
                return self._record(text, self._status_reply())

            if _FOLLOWUP.match(source):
                if self.last_task is None:
                    reply = ChatReply(
                        "unknown",
                        "unknown",
                        "There is no previous desktop task to explain.",
                    )
                    return self._record(text, reply)
                command = (
                    f"Erkläre {self.last_task}"
                    if _german(source)
                    else f"explain {self.last_task}"
                )
                return self._record(
                    text, self._desktop_reply(command, desktop, recorder)
                )

            resolution = self._assistant_resolution(source)
            desktop_intent = self._resolution_value(resolution, "intent")

            # Explicit desktop mutation always wins over every read-only
            # fallback, even when the task name contains numbers.
            if (
                desktop_intent in {"teach", "do", "compose", "template", "correct"}
                or _EXPLICIT_TEACH.search(source)
                or _EXPLICIT_DO.search(source)
                or _EXPLICIT_MANAGEMENT.search(source)
            ):
                return self._record(
                    text, self._desktop_reply(source, desktop, recorder)
                )

            if desktop_intent == "list" or _LIST.search(source):
                return self._record(text, self._capability_reply("list", source))
            if desktop_intent == "help" or _HELP.search(source):
                return self._record(text, self._capability_reply("help", source))

            if desktop_intent == "explain" or _EXPLICIT_EXPLAIN.search(source):
                if self._known_desktop_explanation(source, resolution):
                    return self._record(
                        text, self._desktop_reply(source, desktop, recorder)
                    )

            math_reply = self._math_reply(source)
            if math_reply is not None:
                return self._record(text, math_reply)

            quant_reply = self._quant_reply(source)
            if quant_reply is not None:
                return self._record(text, quant_reply)

            graph_reply = self._graph_reply(source)
            if graph_reply is not None:
                return self._record(text, graph_reply)

            message = (
                "Ich kenne dafür weder eine gelernte Desktop-Aufgabe noch "
                "einen belegten Rechen-, Mess- oder Graphpfad."
                if _german(source)
                else "I found neither a learned desktop task nor a grounded "
                "arithmetic, measurement, or graph path for that request."
            )
            return self._record(text, ChatReply("unknown", "unknown", message))

    def _record(self, user: str, reply: ChatReply) -> ChatReply:
        if (
            reply.route == "desktop"
            and reply.task
            and reply.status
            not in {
                "unknown",
                "ambiguous",
                "error",
            }
        ):
            self.last_task = reply.task
        self._history.append(ChatTurn(user, reply))
        return reply

    def _assistant_resolution(self, text: str) -> object | None:
        resolver = getattr(self.assistant, "resolve", None)
        if not callable(resolver):
            return None
        try:
            return resolver(text)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _resolution_value(resolution: object | None, name: str) -> object | None:
        if isinstance(resolution, Mapping):
            return resolution.get(name)
        return getattr(resolution, name, None)

    def _desktop_tasks(self) -> tuple[str, ...]:
        agent = getattr(self.assistant, "agent", None)
        listing = getattr(agent, "list", None)
        if callable(listing):
            try:
                tasks = tuple(str(task) for task in listing())
                template_listing = getattr(agent, "list_templates", None)
                templates = (
                    tuple(str(task) for task in template_listing())
                    if callable(template_listing)
                    else ()
                )
                return tuple(sorted((*tasks, *templates)))
            except Exception:
                return ()
        store = getattr(self.assistant, "store", None)
        listing = getattr(store, "list", None)
        if callable(listing):
            try:
                return tuple(str(task) for task in listing())
            except Exception:
                return ()
        return ()

    def _known_desktop_explanation(self, text: str, resolution: object | None) -> bool:
        tasks = self._desktop_tasks()
        if not tasks:
            return False
        requested = self._resolution_value(resolution, "task")
        if requested is not None and _normalise(str(requested)) in {
            _normalise(task) for task in tasks
        }:
            return True
        normal = f" {_normalise(text)} "
        if any(f" {_normalise(task)} " in normal for task in tasks):
            return True
        matcher = getattr(self.assistant, "_match_task", None)
        if callable(matcher) and requested:
            try:
                task, alternatives = matcher(str(requested))
                return task is not None or bool(alternatives)
            except (TypeError, ValueError):
                return False
        return False

    def _desktop_reply(
        self, text: str, desktop: object | None, recorder: object | None
    ) -> ChatReply:
        try:
            raw = self.assistant.handle(text, desktop=desktop, recorder=recorder)
        except Exception as exc:
            return ChatReply(
                "error",
                "desktop",
                f"Desktop assistant failed safely: {exc}",
                data={"error": type(exc).__name__},
            )
        status = str(getattr(raw, "status", "error"))
        if status not in {"ok", "unknown", "ambiguous", "needs_input", "error"}:
            status = "error"
        task = getattr(raw, "task", None)
        raw_data = getattr(raw, "data", {})
        data = dict(raw_data) if isinstance(raw_data, Mapping) else {}
        intent = getattr(raw, "intent", None)
        if intent is not None:
            data = {"intent": str(intent), **data}
        return ChatReply(
            status,  # type: ignore[arg-type]
            "desktop",
            str(getattr(raw, "text", "Desktop assistant returned no text.")),
            None if task is None else str(task),
            data,
        )

    def _capability_reply(
        self, route: Literal["list", "help"], source: str
    ) -> ChatReply:
        tasks = self._desktop_tasks()
        capabilities = list(_BASE_CAPABILITIES)
        if self._language_runtime() is not None:
            capabilities.append("report exact HSSLM runtime status")
        task_text = ", ".join(tasks) if tasks else "none"
        capability_text = "; ".join(capabilities)
        if _german(source):
            fallback = (
                f"Gelernte Desktop-Aufgaben: {task_text}. Fähigkeiten: "
                f"{capability_text}."
            )
        else:
            fallback = (
                f"Learned desktop tasks: {task_text}. Capabilities: {capability_text}."
            )
        facts = {"tasks": task_text, "capabilities": capability_text}
        return ChatReply(
            "ok",
            route,
            self._render_facts(route, facts, fallback),
            data={"tasks": tasks, "capabilities": tuple(capabilities)},
        )

    def _math_reply(self, text: str) -> ChatReply | None:
        if not re.search(r"\d", text) or not (
            "?" in text or _QUESTION_HINT.search(text)
        ):
            return None
        from . import bindings

        try:
            result = bindings.bind(text)
        except Exception:
            return None
        if not result.ok or result.answer is None:
            return None
        answer = str(result.answer)
        fallback = f"Ergebnis: {answer}." if _german(text) else f"Answer: {answer}."
        return ChatReply(
            "ok",
            "math",
            self._render_facts("math", {"answer": answer}, fallback),
            data={"answer": answer, "reason": result.reason},
        )

    def _quant_reply(self, text: str) -> ChatReply | None:
        from . import quant

        answer, mechanism, confidence = quant.answer(text)
        if answer is None or mechanism is None:
            return None
        confidence_text = f"{confidence:.3f}"
        fallback = (
            f"Gemessene Antwort: {answer} ({mechanism}, Konfidenz {confidence_text})."
            if _german(text)
            else f"Measured answer: {answer} ({mechanism}, confidence "
            f"{confidence_text})."
        )
        facts = {
            "answer": answer,
            "mechanism": mechanism,
            "confidence": confidence_text,
        }
        return ChatReply(
            "ok",
            "quant",
            self._render_facts("quantitative fact", facts, fallback),
            data={
                "answer": answer,
                "mechanism": mechanism,
                "confidence": confidence,
            },
        )

    @staticmethod
    def _graph_language(text: str) -> str:
        mapped = re.sub(r"^\s*why\s+does\s+", "explain how ", text, flags=re.IGNORECASE)
        return re.sub(
            r"^\s*(?:why|warum|wieso|weshalb)\b",
            "explain",
            mapped,
            flags=re.IGNORECASE,
        )

    def _graph_reply(self, text: str) -> ChatReply | None:
        from . import intent, tools

        try:
            vocabulary = intent.load_vocab(str(self.graph_path))
            parsed = intent.parse_command(self._graph_language(text), vocabulary)
        except Exception:
            return None
        if parsed.status != "ok" or not parsed.grounded or parsed.tool is None:
            return None
        result = tools.execute(parsed, str(self.graph_path))
        if not result.ok:
            return ChatReply(
                "unknown",
                "graph",
                result.text or "The grounded graph tool could not answer.",
                data={"tool": result.tool, "intent": parsed.action},
            )
        facts = {"result": result.text}
        return ChatReply(
            "ok",
            "graph",
            self._render_facts("graph result", facts, result.text),
            data={
                "tool": result.tool,
                "intent": parsed.action,
                "target": parsed.target,
                "detail": result.detail,
            },
        )

    def _language_runtime(self) -> object | None:
        language = getattr(self.assistant, "language", None)
        return getattr(language, "runtime", None)

    def _status_reply(self) -> ChatReply:
        runtime = self._language_runtime()
        status_fn = getattr(runtime, "status", None)
        if runtime is None or not callable(status_fn):
            return ChatReply(
                "unknown", "status", "No HSSLM language runtime is attached."
            )
        try:
            status = status_fn()
        except Exception as exc:
            return ChatReply(
                "error",
                "status",
                f"HSSLM status failed: {exc}",
                data={"error": type(exc).__name__},
            )
        if isinstance(status, Mapping):
            values = dict(status)
        elif is_dataclass(status):
            values = asdict(status)
        else:
            names = (
                "checkpoint",
                "tokenizer",
                "ready",
                "parameter_count",
                "serialized_parameter_count",
                "size_bytes",
                "device",
                "reason",
            )
            values = {
                name: getattr(status, name) for name in names if hasattr(status, name)
            }
        values = {
            str(key): str(value) if isinstance(value, Path) else value
            for key, value in values.items()
        }
        ordered = (
            "ready",
            "parameter_count",
            "serialized_parameter_count",
            "size_bytes",
            "device",
            "checkpoint",
            "tokenizer",
            "reason",
        )
        facts = {name: values[name] for name in ordered if name in values}
        fallback = "HSSLM status — " + "; ".join(
            f"{name}={value}" for name, value in facts.items()
        )
        return ChatReply(
            "ok",
            "status",
            self._render_facts("HSSLM status", facts, fallback),
            data=values,
        )

    def _render_facts(
        self, event: str, facts: Mapping[str, object], fallback: str
    ) -> str:
        """Use only a renderer whose output preserves every supplied fact."""

        language = getattr(self.assistant, "language", None)
        renderer = getattr(language, "render", None)
        if not callable(renderer):
            return fallback
        try:
            # Give constrained renderers the already grounded natural answer,
            # not merely a bag of fields.  HSSLM may rank/retain that wording
            # while the checks below still require every original fact.
            rendered = renderer(event, {"message": fallback, **facts})
        except Exception:
            return fallback
        if not isinstance(rendered, str) or not rendered.strip():
            return fallback
        candidate = rendered.strip()
        normal = candidate.casefold()
        required = [str(value).strip() for value in facts.values()]
        if all(not value or value.casefold() in normal for value in required):
            return candidate
        return fallback


__all__ = [
    "ChatReply",
    "ChatRoute",
    "ChatStatus",
    "ChatTurn",
    "FertigChat",
]
