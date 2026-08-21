"""Natural-language product surface for the FERTIG desktop apprentice.

The assistant deliberately keeps language resolution separate from execution:
deterministic German/English commands work without a language model, while an
injected resolver/renderer (for example an HSSL/HLSSM adapter) can provide a
richer conversational surface.  In either case only a resolved, uniquely
named stored task is allowed to cross the desktop action boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path
import re
import shlex
import threading
import time
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable
import unicodedata

from fertig.desktop import DesktopDemonstration, DesktopEnvironment
from fertig.desktop_agent import DesktopAgent, DesktopRunResult, TaskStore
from fertig.skill_slots import (
    SkillTemplateError,
    TemplateNotFoundError,
    TemplateStore,
)


Intent = Literal[
    "teach",
    "do",
    "explain",
    "compose",
    "template",
    "correct",
    "list",
    "help",
    "status",
    "unknown",
]
ReplyStatus = Literal["ok", "unknown", "ambiguous", "needs_input", "error"]

OK: ReplyStatus = "ok"
UNKNOWN: ReplyStatus = "unknown"
AMBIGUOUS: ReplyStatus = "ambiguous"
NEEDS_INPUT: ReplyStatus = "needs_input"
ERROR: ReplyStatus = "error"


@dataclass(frozen=True)
class IntentResolution:
    """Language-independent result consumed by :class:`FertigAssistant`."""

    intent: Intent
    task: str | None = None
    confidence: float = 1.0
    status: str = "resolved"
    reason: str = ""
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AssistantReply:
    """Typed, serialisable result returned for every user utterance."""

    status: ReplyStatus
    intent: Intent
    task: str | None
    text: str
    data: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class LanguageInterface(Protocol):
    """Optional structural bridge for HSSL/HLSSM or another language layer.

    ``resolve`` may return an :class:`IntentResolution`, a mapping containing
    ``intent``/``task``, a two-item ``(intent, task)`` sequence, or ``None`` to
    defer to the built-in deterministic resolver.  ``render`` can replace the
    fallback wording but cannot change the resolved status or selected task.
    """

    def resolve(
        self, text: str, tasks: Sequence[str]
    ) -> IntentResolution | Mapping[str, Any] | Sequence[Any] | None: ...

    def render(self, event: str, facts: Mapping[str, Any]) -> str | None: ...


@runtime_checkable
class DemonstrationRecorder(Protocol):
    """The only method needed from the passive macOS recorder."""

    def record(self, *, label: str | None = None) -> DesktopDemonstration: ...


_VALID_INTENTS = frozenset(
    {
        "teach",
        "do",
        "explain",
        "compose",
        "template",
        "correct",
        "list",
        "help",
        "status",
        "unknown",
    }
)
_RESOLUTION_STATUSES = frozenset({"resolved", "unknown", "ambiguous"})
_WORDS = re.compile(r"[^\w]+", flags=re.UNICODE)

_LIST_PATTERNS = (
    re.compile(r"\bwas kannst du\b"),
    re.compile(r"\bwelche aufgaben\b"),
    re.compile(r"\bzeige (?:mir )?(?:deine )?aufgaben\b"),
    re.compile(r"\bwhat can you do\b"),
    re.compile(r"\b(?:list|show) (?:my |your )?(?:tasks|skills)\b"),
    re.compile(r"^(?:tasks|skills|aufgaben)$"),
)
_HELP_PATTERNS = (
    re.compile(r"\b(?:hilfe|help)\b"),
    re.compile(r"\bwie benutze ich (?:dich|fertig)\b"),
    re.compile(r"\bhow (?:do|can) i use (?:you|fertig)\b"),
)
_STATUS_PATTERNS = (
    re.compile(r"\b(?:hsslm|hlssm|modellstatus)\b"),
    re.compile(r"\b(?:model|language) status\b"),
)
_TEACH_PATTERNS = (
    re.compile(r"^(?:bitte )?(?:lerne|lern|lernen)\b"),
    re.compile(r"^(?:bitte )?(?:ich )?zeige dir\b"),
    re.compile(r"^(?:bitte )?bring(?:e)? dir\b"),
    re.compile(r"^(?:please )?teach(?: me)?\b"),
    re.compile(r"^(?:please )?learn\b"),
)
_EXPLAIN_PATTERNS = (
    re.compile(r"\b(?:erkläre|erklär|erklaere|erklaer)\b"),
    re.compile(r"\bwie (?:geht|funktioniert)\b"),
    re.compile(r"\bexplain\b"),
    re.compile(r"\bhow does\b"),
)
_DO_PATTERNS = (
    re.compile(r"^(?:bitte )?(?:führe|fuehre)\b"),
    re.compile(r"^(?:bitte )?(?:mach|mache|erledige|starte|start)\b"),
    re.compile(r"^(?:please )?(?:do|execute|run|perform)\b"),
)

_RAW_DO_PREFIX = re.compile(
    r"^\s*(?:(?:please|bitte)\s+)?"
    r"(?:do|execute|run|perform|mach|mache|erledige|starte|start|"
    r"f(?:ü|ue)hre(?:\s+aus)?)\s+",
    re.IGNORECASE,
)
_COMPOSE_PREFIX = re.compile(
    r"^\s*(?:compose|combine|kombiniere|komponiere)\s+",
    re.IGNORECASE,
)
_TEMPLATE_PREFIX = re.compile(
    r"^\s*(?:template|parameterize|parameterise|parameterisiere)\s+",
    re.IGNORECASE,
)
_CORRECT_PREFIX = re.compile(
    r"^\s*(?:correct|repair|korrigiere|korrigier|repariere)\s+",
    re.IGNORECASE,
)
_SLOT_SPEC = re.compile(r"^([a-z][a-z0-9_-]{0,63})(\?)?=(\d+)$", re.I)

_COMMAND_WORDS: Mapping[Intent, frozenset[str]] = {
    "teach": frozenset(
        {
            "lerne",
            "lern",
            "lernen",
            "ich",
            "zeige",
            "zeig",
            "dir",
            "bringe",
            "bring",
            "bei",
            "teach",
            "me",
            "learn",
        }
    ),
    "do": frozenset(
        {
            "führe",
            "fuehre",
            "aus",
            "mach",
            "mache",
            "erledige",
            "starte",
            "start",
            "do",
            "execute",
            "run",
            "perform",
        }
    ),
    "explain": frozenset(
        {
            "erkläre",
            "erklär",
            "erklaere",
            "erklaer",
            "wie",
            "geht",
            "funktioniert",
            "explain",
            "how",
            "does",
            "work",
        }
    ),
    "compose": frozenset(),
    "template": frozenset(),
    "correct": frozenset(),
    "list": frozenset(),
    "help": frozenset(),
    "status": frozenset(),
    "unknown": frozenset(),
}
_EDGE_FILLERS = frozenset(
    {
        "bitte",
        "jetzt",
        "mal",
        "doch",
        "den",
        "die",
        "das",
        "eine",
        "einen",
        "the",
        "a",
        "an",
        "task",
        "aufgabe",
        "please",
        "for",
        "für",
        "fuer",
    }
)
_GERMAN_HINTS = frozenset(
    {
        "lerne",
        "lern",
        "zeige",
        "führe",
        "fuehre",
        "mach",
        "mache",
        "erledige",
        "erkläre",
        "erklaere",
        "hilfe",
        "bitte",
        "aufgabe",
        "aufgaben",
        "kannst",
    }
)


def _normalise(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("text must be a string")
    return " ".join(_WORDS.sub(" ", value.casefold()).split())


def _ascii(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(character)
    )


def _is_german(text: str) -> bool:
    normalised = _normalise(text)
    words = set(normalised.split())
    return bool(words & _GERMAN_HINTS) or any(
        char in text.casefold() for char in "äöüß"
    )


def _task_query(text: str, intent: Intent) -> str:
    words = _normalise(text).split()
    commands = _COMMAND_WORDS[intent]
    retained = [word for word in words if word not in commands]
    return _trim_edge_fillers(" ".join(retained))


def _trim_edge_fillers(text: str) -> str:
    retained = _normalise(text).split()
    while retained and retained[0] in _EDGE_FILLERS:
        retained.pop(0)
    while retained and retained[-1] in _EDGE_FILLERS:
        retained.pop()
    return " ".join(retained)


def _slug(value: str) -> str:
    candidate = re.sub(r"[^a-z0-9]+", "-", _ascii(_normalise(value))).strip("-")
    return candidate[:80] or "task"


def _shell_words(value: str) -> tuple[str, ...]:
    try:
        return tuple(shlex.split(value, comments=False, posix=True))
    except ValueError as exc:
        raise ValueError(f"invalid quoting: {exc}") from exc


def _parse_composition(text: str) -> tuple[str, tuple[str, ...]] | None:
    match = _COMPOSE_PREFIX.match(text)
    if match is None:
        return None
    body = text[match.end() :].strip()
    if "=" not in body:
        raise ValueError("composition syntax is: compose NAME = TASK + TASK [ + TASK ]")
    raw_name, raw_children = body.split("=", 1)
    name = " ".join(_shell_words(raw_name))
    children = tuple(
        " ".join(_shell_words(part))
        for part in re.split(r"\s+\+\s+", raw_children.strip())
        if part.strip()
    )
    if not name or len(children) < 2 or any(not child for child in children):
        raise ValueError("composition syntax is: compose NAME = TASK + TASK [ + TASK ]")
    return name, children


def _parse_template_definition(
    text: str,
) -> tuple[str, str, tuple[tuple[str, int, bool], ...]] | None:
    match = _TEMPLATE_PREFIX.match(text)
    if match is None:
        return None
    tokens = list(_shell_words(text[match.end() :]))
    separators = [
        index
        for index, token in enumerate(tokens)
        if token.casefold() in {"from", "aus"}
    ]
    if len(separators) != 1:
        raise ValueError(
            "template syntax is: template NAME from BASE SLOT=STEP [SLOT?=STEP]"
        )
    separator = separators[0]
    name = " ".join(tokens[:separator]).strip()
    tail = tokens[separator + 1 :]
    first_slot = next(
        (index for index, token in enumerate(tail) if "=" in token), len(tail)
    )
    base = " ".join(tail[:first_slot]).strip()
    raw_slots = [
        token
        for token in tail[first_slot:]
        if token.casefold() not in {"slot", "slots"}
    ]
    if not name or not base or not raw_slots:
        raise ValueError(
            "template syntax is: template NAME from BASE SLOT=STEP [SLOT?=STEP]"
        )
    slots: list[tuple[str, int, bool]] = []
    for raw in raw_slots:
        slot = _SLOT_SPEC.fullmatch(raw)
        if slot is None:
            raise ValueError(
                f"invalid slot {raw!r}; expected NAME=STEP or optional NAME?=STEP"
            )
        slots.append((slot.group(1), int(slot.group(3)), slot.group(2) is None))
    return name, base, tuple(slots)


def _parse_correction(text: str) -> tuple[str, int] | None:
    match = _CORRECT_PREFIX.match(text)
    if match is None:
        return None
    tokens = list(_shell_words(text[match.end() :]))
    step_index = 0
    if tokens and re.fullmatch(r"(?:step|schritt)=\d+", tokens[-1], re.I):
        step_index = int(tokens.pop().split("=", 1)[1])
    task = " ".join(tokens).strip()
    if not task:
        raise ValueError("correction syntax is: correct TASK [step=N]")
    return task, step_index


class FertigAssistant:
    """One conversational entry point for teaching and running desktop tasks."""

    def __init__(
        self,
        store: TaskStore | str | Path | None = None,
        *,
        template_store: TemplateStore | str | Path | None = None,
        recordings_dir: str | Path | None = None,
        language: LanguageInterface | None = None,
        aliases: Mapping[str, str] | None = None,
    ) -> None:
        if isinstance(store, TaskStore):
            task_store = store
        elif store is None:
            task_store = TaskStore()
        else:
            task_store = TaskStore(store)
        if isinstance(template_store, TemplateStore):
            templates = template_store
        elif template_store is None:
            templates = TemplateStore()
        else:
            templates = TemplateStore(template_store)
        self.store = task_store
        self.template_store = templates
        self.agent = DesktopAgent(task_store, templates)
        self.recordings_dir = None if recordings_dir is None else Path(recordings_dir)
        self.language = language
        self._aliases = {
            _normalise(alias): _normalise(task)
            for alias, task in (aliases or {}).items()
            if _normalise(alias) and _normalise(task)
        }
        self._lock = threading.RLock()

    def resolve(self, text: str) -> IntentResolution:
        """Resolve an utterance without causing recording or desktop input."""

        explicit = self._explicit_resolution(text)
        if explicit is not None:
            return explicit
        external = self._external_resolution(text)
        if external is not None:
            return external
        normalised = _normalise(text)
        if not normalised:
            return IntentResolution("unknown", None, 0.0)
        if any(pattern.search(normalised) for pattern in _LIST_PATTERNS):
            return IntentResolution("list")
        if any(pattern.search(normalised) for pattern in _HELP_PATTERNS):
            return IntentResolution("help")
        if any(pattern.search(normalised) for pattern in _STATUS_PATTERNS):
            return IntentResolution("status")
        for intent, patterns in (
            ("teach", _TEACH_PATTERNS),
            ("explain", _EXPLAIN_PATTERNS),
            ("do", _DO_PATTERNS),
        ):
            if any(pattern.search(normalised) for pattern in patterns):
                return IntentResolution(intent, _task_query(normalised, intent))
        return IntentResolution("unknown", None, 0.0)

    def _explicit_resolution(self, text: str) -> IntentResolution | None:
        """Parse management and parameter syntax before lossy normalisation."""

        composition = _parse_composition(text)
        if composition is not None:
            name, children = composition
            return IntentResolution("compose", name, arguments={"children": children})
        template = _parse_template_definition(text)
        if template is not None:
            name, base, slots = template
            return IntentResolution(
                "template",
                name,
                arguments={"base_task": base, "slots": slots},
            )
        correction = _parse_correction(text)
        if correction is not None:
            task, step_index = correction
            return IntentResolution(
                "correct", task, arguments={"step_index": step_index}
            )

        raw = text.strip()
        do_match = _RAW_DO_PREFIX.match(raw)
        invocation_text = raw[do_match.end() :].strip() if do_match else raw
        # Assignment syntax is reserved for a stored template.  A leading do
        # also permits an all-optional template without assignments.
        if "=" in invocation_text or do_match is not None:
            try:
                invocation = self.agent.resolve_template(invocation_text)
            except TemplateNotFoundError:
                return None
            except SkillTemplateError as exc:
                words = _shell_words(invocation_text)
                first_assignment = next(
                    (index for index, word in enumerate(words) if "=" in word),
                    len(words),
                )
                task = " ".join(words[:first_assignment]) or None
                return IntentResolution(
                    "do",
                    task,
                    0.0,
                    "unknown",
                    str(exc),
                    {"template_utterance": invocation_text},
                )
            return IntentResolution(
                "do",
                invocation.template_name,
                arguments={
                    "template_utterance": invocation_text,
                    "base_task": invocation.base_task,
                    "values": dict(invocation.values),
                },
            )
        return None

    def handle(
        self,
        text: str,
        desktop: DesktopEnvironment | None = None,
        recorder: DemonstrationRecorder | None = None,
    ) -> AssistantReply:
        """Resolve and complete one request, returning rather than guessing."""

        with self._lock:
            try:
                resolution = self.resolve(text)
            except (TypeError, ValueError) as exc:
                return self._reply(ERROR, "unknown", None, str(exc), {}, text)
            german = _is_german(text)
            if resolution.status == "ambiguous":
                _, alternatives = self._match_task(resolution.task or "")
                if not alternatives:
                    alternatives = self._skill_names()
                joined = ", ".join(alternatives) or "—"
                message = (
                    f"Das ist mehrdeutig: {joined}. Bitte nenne die Aufgabe genauer."
                    if german
                    else f"That is ambiguous: {joined}. Please name the task more precisely."
                )
                return self._reply(
                    AMBIGUOUS,
                    resolution.intent,
                    None,
                    message,
                    {
                        "alternatives": alternatives,
                        "reason": resolution.reason,
                        "confidence": resolution.confidence,
                    },
                    text,
                )
            if resolution.status == "unknown" and resolution.intent != "unknown":
                message = (
                    f"Ich kann die Anfrage nicht sicher zuordnen: {resolution.reason or 'unbekannte Aufgabe'}."
                    if german
                    else "I cannot resolve that request safely: "
                    f"{resolution.reason or 'unknown task'}."
                )
                return self._reply(
                    UNKNOWN,
                    resolution.intent,
                    None,
                    message,
                    {
                        "known_tasks": self.agent.list(),
                        "known_templates": self.agent.list_templates(),
                        "reason": resolution.reason,
                        "confidence": resolution.confidence,
                    },
                    text,
                )
            if resolution.intent == "help":
                return self._help(german, text)
            if resolution.intent == "list":
                return self._list(german, text)
            if resolution.intent == "status":
                return self._status(german, text)
            if resolution.intent == "unknown":
                message = (
                    "Ich konnte keine Aktion erkennen. Sage zum Beispiel „lerne …“, "
                    "„mach …“, „erkläre …“ oder „was kannst du?“"
                    if german
                    else "I could not resolve an action. Try ‘teach …’, ‘do …’, "
                    "‘explain …’, or ‘list tasks’."
                )
                return self._reply(UNKNOWN, "unknown", None, message, {}, text)
            if resolution.intent == "teach":
                return self._teach(resolution.task, recorder, german, text)
            if resolution.intent == "compose":
                return self._compose(
                    resolution.task,
                    tuple(resolution.arguments.get("children", ())),
                    german,
                    text,
                )
            if resolution.intent == "template":
                return self._define_template(
                    resolution.task,
                    str(resolution.arguments.get("base_task", "")),
                    tuple(resolution.arguments.get("slots", ())),
                    german,
                    text,
                )
            if resolution.intent == "correct":
                return self._correct(
                    resolution.task,
                    int(resolution.arguments.get("step_index", 0)),
                    recorder,
                    german,
                    text,
                )
            template_utterance = resolution.arguments.get("template_utterance")
            if resolution.intent == "do" and isinstance(template_utterance, str):
                return self._do_template(template_utterance, desktop, german, text)

            task, alternatives = self._match_task(resolution.task or "")
            if task is None:
                if alternatives:
                    joined = ", ".join(alternatives)
                    message = (
                        f"Das ist mehrdeutig: {joined}. Bitte nenne die Aufgabe genauer."
                        if german
                        else f"That is ambiguous: {joined}. Please name the task more precisely."
                    )
                    return self._reply(
                        AMBIGUOUS,
                        resolution.intent,
                        None,
                        message,
                        {"alternatives": alternatives},
                        text,
                    )
                query = resolution.task or ""
                message = (
                    f"Diese Aufgabe kenne ich noch nicht: {query or '—'}."
                    if german
                    else f"I do not know that task yet: {query or '—'}."
                )
                return self._reply(
                    UNKNOWN,
                    resolution.intent,
                    None,
                    message,
                    {
                        "known_tasks": self.agent.list(),
                        "known_templates": self.agent.list_templates(),
                    },
                    text,
                )
            if resolution.intent == "explain":
                is_template = task in self.agent.list_templates()
                explanation = (
                    self.agent.explain_template(task)
                    if is_template
                    else self.agent.explain(task)
                )
                message = (
                    f"So führe ich „{task}“ aus:\n{explanation}"
                    if german
                    else f"This is how I perform “{task}”:\n{explanation}"
                )
                return self._reply(
                    OK,
                    "explain",
                    task,
                    message,
                    {"explanation": explanation},
                    text,
                )
            if task in self.agent.list_templates():
                return self._do_template(task, desktop, german, text)
            return self._do(task, desktop, german, text)

    def _external_resolution(self, text: str) -> IntentResolution | None:
        if self.language is None:
            return None
        resolver = getattr(self.language, "resolve", None)
        if not callable(resolver):
            return None
        raw = resolver(text, self._skill_names())
        if raw is None:
            return None
        if isinstance(raw, IntentResolution):
            result = raw
        elif isinstance(raw, Mapping):
            result = IntentResolution(
                str(raw.get("intent", "unknown")),
                raw.get("task"),
                float(raw.get("confidence", 1.0)),
                str(raw.get("status", "resolved")),
                str(raw.get("reason", "")),
                raw.get("arguments", {}),
            )
        elif hasattr(raw, "intent") and hasattr(raw, "status"):
            raw_skill = getattr(raw, "skill", None)
            raw_task = getattr(raw, "task", None)
            if raw_task is None and raw_skill is not None:
                raw_task = getattr(raw_skill, "name", None)
            result = IntentResolution(
                str(getattr(raw, "intent")),
                raw_task,
                float(getattr(raw, "confidence", 1.0)),
                str(getattr(raw, "status")),
                str(getattr(raw, "reason", "")),
                getattr(raw, "arguments", {}),
            )
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            if len(raw) != 2:
                raise ValueError(
                    "language resolve sequence must contain intent and task"
                )
            result = IntentResolution(str(raw[0]), raw[1])
        else:
            raise TypeError("language resolve returned an unsupported value")
        if result.intent not in _VALID_INTENTS:
            raise ValueError(
                f"language resolve returned invalid intent {result.intent!r}"
            )
        if result.status not in _RESOLUTION_STATUSES:
            raise ValueError(
                f"language resolve returned invalid status {result.status!r}"
            )
        if not 0.0 <= result.confidence <= 1.0:
            raise ValueError("language resolve confidence must be within [0, 1]")
        task = None if result.task is None else _normalise(str(result.task))
        if task is not None and result.intent == "teach":
            task = _trim_edge_fillers(task)
        return IntentResolution(
            result.intent,
            task,
            result.confidence,
            result.status,
            result.reason,
            result.arguments,
        )

    def _skill_names(self) -> tuple[str, ...]:
        return tuple(sorted((*self.agent.list(), *self.agent.list_templates())))

    def _match_task(self, query: str) -> tuple[str | None, tuple[str, ...]]:
        tasks = self._skill_names()
        if not tasks:
            return None, ()
        normalised = _normalise(query)
        if not normalised:
            return None, tasks if len(tasks) > 1 else ()
        aliases: dict[str, str] = {task: task for task in tasks}
        aliases.update(
            {
                alias: target
                for alias, target in self._aliases.items()
                if target in tasks
            }
        )

        exact: dict[str, tuple[int, int]] = {}
        padded = f" {normalised} "
        for alias, task in aliases.items():
            if f" {alias} " in padded:
                exact[task] = max(
                    exact.get(task, (0, 0)), (len(alias.split()), len(alias))
                )
        if exact:
            best_specificity = max(exact.values())
            winners = tuple(
                sorted(
                    task for task, score in exact.items() if score == best_specificity
                )
            )
            return (winners[0], ()) if len(winners) == 1 else (None, winners)

        query_tokens = set(normalised.split())
        scores: dict[str, float] = {}
        for alias, task in aliases.items():
            alias_tokens = set(alias.split())
            overlap = len(query_tokens & alias_tokens)
            coverage = overlap / max(1, len(query_tokens))
            precision = overlap / max(1, len(alias_tokens))
            sequence = SequenceMatcher(None, normalised, alias).ratio()
            score = max(sequence, 0.45 * coverage + 0.25 * precision + 0.30 * sequence)
            scores[task] = max(scores.get(task, 0.0), score)
        best = max(scores.values())
        if best < 0.62:
            return None, ()
        contenders = tuple(
            sorted(task for task, score in scores.items() if best - score <= 0.08)
        )
        return (contenders[0], ()) if len(contenders) == 1 else (None, contenders)

    def _teach(
        self,
        raw_task: str | None,
        recorder: DemonstrationRecorder | None,
        german: bool,
        source_text: str,
    ) -> AssistantReply:
        task = _normalise(raw_task or "")
        if not task:
            message = (
                "Welche Aufgabe soll ich lernen?"
                if german
                else "Which task should I learn?"
            )
            return self._reply(NEEDS_INPUT, "teach", None, message, {}, source_text)
        if len(task) > 256:
            return self._reply(
                ERROR,
                "teach",
                None,
                "Task name is limited to 256 characters.",
                {},
                source_text,
            )
        if task in self.agent.list_templates() or (
            task in self.agent.list() and self.store.get(task) is None
        ):
            message = (
                f"„{task}“ ist bereits eine Komposition oder Vorlage; "
                "lerne stattdessen einen neuen atomaren Namen."
                if german
                else f"“{task}” is already a composition or template; "
                "teach a new atomic name instead."
            )
            return self._reply(ERROR, "teach", task, message, {}, source_text)
        record = getattr(recorder, "record", None)
        if not callable(record):
            message = (
                "Zum Lernen brauche ich einen Recorder. Starte die Aufnahme und beende sie mit F8."
                if german
                else "Teaching needs a recorder. Start recording and finish it with F8."
            )
            return self._reply(NEEDS_INPUT, "teach", task, message, {}, source_text)
        try:
            demo = record(label=task)
            if not isinstance(demo, DesktopDemonstration):
                raise TypeError("recorder must return DesktopDemonstration")
            if not demo.steps:
                raise ValueError("the recording contains no replayable actions")
            self.agent.teach_from_demo(task, demo)
            paths = self._save_recording(task, demo)
        except Exception as exc:  # adapter failures become a typed product reply
            message = (
                f"Ich konnte „{task}“ nicht lernen: {exc}"
                if german
                else f"I could not learn “{task}”: {exc}"
            )
            return self._reply(
                ERROR,
                "teach",
                task,
                message,
                {"error": type(exc).__name__},
                source_text,
            )
        data: dict[str, Any] = {"steps": len(demo.steps)}
        if paths is not None:
            data["recording"] = {
                "metadata": str(paths.metadata),
                "frames": str(paths.frames),
            }
        message = (
            f"Gelernt: „{task}“ ({len(demo.steps)} Schritte)."
            if german
            else f"Learned “{task}” ({len(demo.steps)} steps)."
        )
        return self._reply(OK, "teach", task, message, data, source_text)

    def _save_recording(self, task: str, demo: DesktopDemonstration) -> Any | None:
        if self.recordings_dir is None:
            return None
        directory = self.recordings_dir / _slug(task)
        # Nanoseconds plus a collision loop keeps every raw demonstration.
        stem = f"demo-{time.time_ns()}"
        candidate = directory / stem
        suffix = 1
        while (
            candidate.with_suffix(".json").exists()
            or candidate.with_suffix(".npz").exists()
        ):
            candidate = directory / f"{stem}-{suffix}"
            suffix += 1
        return demo.save(candidate)

    def _compose(
        self,
        raw_name: str | None,
        children: Sequence[str],
        german: bool,
        source_text: str,
    ) -> AssistantReply:
        name = _normalise(raw_name or "")
        try:
            stored = self.agent.compose(name, tuple(children))
        except Exception as exc:
            message = (
                f"Ich konnte den Ablauf „{name or '—'}“ nicht anlegen: {exc}"
                if german
                else f"I could not compose “{name or '—'}”: {exc}"
            )
            return self._reply(
                ERROR,
                "compose",
                name or None,
                message,
                {"error": type(exc).__name__},
                source_text,
            )
        joined = " → ".join(stored)
        message = (
            f"Komponiert: „{name}“ = {joined}."
            if german
            else f"Composed “{name}” = {joined}."
        )
        return self._reply(
            OK,
            "compose",
            name,
            message,
            {"children": stored},
            source_text,
        )

    def _define_template(
        self,
        raw_name: str | None,
        base_task: str,
        slots: Sequence[tuple[str, int, bool]],
        german: bool,
        source_text: str,
    ) -> AssistantReply:
        name = _normalise(raw_name or "")
        try:
            template = self.agent.declare_template(name, base_task, tuple(slots))
        except Exception as exc:
            message = (
                f"Ich konnte die Vorlage „{name or '—'}“ nicht anlegen: {exc}"
                if german
                else f"I could not define template “{name or '—'}”: {exc}"
            )
            return self._reply(
                ERROR,
                "template",
                name or None,
                message,
                {"error": type(exc).__name__},
                source_text,
            )
        slot_data = tuple(
            {
                "name": slot.name,
                "step_index": slot.step_index,
                "required": slot.required,
                "example": slot.example,
            }
            for slot in template.slots
        )
        message = (
            f"Vorlage angelegt: „{template.name}“ auf „{template.base_task}“ "
            f"mit {len(slot_data)} Textfeld(ern)."
            if german
            else f"Defined template “{template.name}” on “{template.base_task}” "
            f"with {len(slot_data)} text slot(s)."
        )
        return self._reply(
            OK,
            "template",
            template.name,
            message,
            {"base_task": template.base_task, "slots": slot_data},
            source_text,
        )

    def _correct(
        self,
        raw_task: str | None,
        step_index: int,
        recorder: DemonstrationRecorder | None,
        german: bool,
        source_text: str,
    ) -> AssistantReply:
        task, alternatives = self._match_task(raw_task or "")
        if task is None:
            message = (
                "Die zu korrigierende Aufgabe ist unbekannt oder mehrdeutig."
                if german
                else "The task to correct is unknown or ambiguous."
            )
            return self._reply(
                AMBIGUOUS if alternatives else UNKNOWN,
                "correct",
                None,
                message,
                {"alternatives": alternatives, "known_tasks": self.agent.list()},
                source_text,
            )
        if self.store.get(task) is None:
            message = (
                "Kompositionen und Vorlagen werden an ihrem atomaren Teilschritt korrigiert."
                if german
                else "Compositions and templates are corrected at their atomic leaf task."
            )
            return self._reply(ERROR, "correct", task, message, {}, source_text)
        record = getattr(recorder, "record", None)
        if not callable(record):
            message = (
                "Zur Korrektur brauche ich eine einzelne neue Vorführung; F8 beendet sie."
                if german
                else "Correction needs one new demonstration; F8 finishes it."
            )
            return self._reply(NEEDS_INPUT, "correct", task, message, {}, source_text)
        try:
            demo = record(label=f"correction:{task}:step:{step_index}")
            if not isinstance(demo, DesktopDemonstration):
                raise TypeError("recorder must return DesktopDemonstration")
            if len(demo.steps) != 1:
                raise ValueError(
                    "a correction recording must contain exactly one action"
                )
            accepted = self.agent.correct(task, demo.steps[0], step_index=step_index)
            if not accepted:
                raise ValueError(
                    "the correction did not produce a successful visible effect"
                )
            paths = self._save_recording(task, demo)
        except Exception as exc:
            message = (
                f"Korrektur für „{task}“ fehlgeschlagen: {exc}"
                if german
                else f"Correction for “{task}” failed: {exc}"
            )
            return self._reply(
                ERROR,
                "correct",
                task,
                message,
                {"error": type(exc).__name__, "step_index": step_index},
                source_text,
            )
        data: dict[str, Any] = {"step_index": step_index, "accepted": True}
        if paths is not None:
            data["recording"] = {
                "metadata": str(paths.metadata),
                "frames": str(paths.frames),
            }
        message = (
            f"Korrigiert: „{task}“, Schritt {step_index}."
            if german
            else f"Corrected “{task}”, step {step_index}."
        )
        return self._reply(OK, "correct", task, message, data, source_text)

    def _do_template(
        self,
        invocation_text: str,
        desktop: DesktopEnvironment | None,
        german: bool,
        source_text: str,
    ) -> AssistantReply:
        try:
            invocation = self.agent.resolve_template(invocation_text)
        except (KeyError, ValueError) as exc:
            message = (
                f"Vorlage kann nicht ausgeführt werden: {exc}"
                if german
                else f"Template cannot run: {exc}"
            )
            return self._reply(
                NEEDS_INPUT,
                "do",
                None,
                message,
                {"error": type(exc).__name__},
                source_text,
            )
        if not isinstance(desktop, DesktopEnvironment):
            message = (
                "Zum Ausführen brauche ich eine Desktop-Umgebung."
                if german
                else "Execution needs a desktop environment."
            )
            return self._reply(
                NEEDS_INPUT,
                "do",
                invocation.template_name,
                message,
                {
                    "base_task": invocation.base_task,
                    "values": dict(invocation.values),
                },
                source_text,
            )
        result = self.agent.invoke_template(invocation, desktop)
        data = {
            **self._run_data(result),
            "base_task": invocation.base_task,
            "values": dict(invocation.values),
        }
        if result.success:
            message = (
                f"Erledigt: „{invocation.template_name}“ ({len(result.steps)} Schritte)."
                if german
                else f"Done: “{invocation.template_name}” ({len(result.steps)} steps)."
            )
            return self._reply(
                OK,
                "do",
                invocation.template_name,
                message,
                data,
                source_text,
            )
        message = (
            f"„{invocation.template_name}“ wurde gestoppt: {result.reason}"
            if german
            else f"“{invocation.template_name}” stopped: {result.reason}"
        )
        return self._reply(
            UNKNOWN,
            "do",
            invocation.template_name,
            message,
            data,
            source_text,
        )

    def _do(
        self,
        task: str,
        desktop: DesktopEnvironment | None,
        german: bool,
        source_text: str,
    ) -> AssistantReply:
        if not isinstance(desktop, DesktopEnvironment):
            message = (
                "Zum Ausführen brauche ich eine Desktop-Umgebung."
                if german
                else "Execution needs a desktop environment."
            )
            return self._reply(NEEDS_INPUT, "do", task, message, {}, source_text)
        try:
            result = self.agent.do(task, desktop)
        except Exception as exc:
            message = (
                f"„{task}“ wurde sicher gestoppt: {exc}"
                if german
                else f"“{task}” stopped safely: {exc}"
            )
            return self._reply(
                ERROR,
                "do",
                task,
                message,
                {"error": type(exc).__name__},
                source_text,
            )
        data = self._run_data(result)
        if result.success:
            message = (
                f"Erledigt: „{task}“ ({len(result.steps)} Schritte)."
                if german
                else f"Done: “{task}” ({len(result.steps)} steps)."
            )
            return self._reply(OK, "do", task, message, data, source_text)
        message = (
            f"„{task}“ wurde gestoppt: {result.reason}"
            if german
            else f"“{task}” stopped: {result.reason}"
        )
        return self._reply(UNKNOWN, "do", task, message, data, source_text)

    @staticmethod
    def _run_data(result: DesktopRunResult) -> Mapping[str, Any]:
        return {
            "success": result.success,
            "planner_status": result.status,
            "reason": result.reason,
            "steps": [
                {
                    "index": step.index,
                    "action": step.action.to_dict(),
                    "verified": step.verified,
                    "before_shape": step.before_shape,
                    "after_shape": step.after_shape,
                }
                for step in result.steps
            ],
        }

    def _list(self, german: bool, source_text: str) -> AssistantReply:
        tasks = self.agent.list()
        templates = self.agent.list_templates()
        if tasks or templates:
            joined = ", ".join(tasks) or "—"
            template_text = ", ".join(templates) or "—"
            message = (
                f"Aufgaben: {joined}. Vorlagen: {template_text}."
                if german
                else f"Tasks: {joined}. Templates: {template_text}."
            )
        else:
            message = (
                "Ich habe noch keine Aufgabe gelernt."
                if german
                else "I have not learned any tasks yet."
            )
        return self._reply(
            OK,
            "list",
            None,
            message,
            {"tasks": tasks, "templates": templates},
            source_text,
        )

    def _help(self, german: bool, source_text: str) -> AssistantReply:
        message = (
            "Befehle: „lerne <Aufgabe>“, „mach <Aufgabe>“, „erkläre <Aufgabe>“, "
            "„compose NAME = TASK + TASK“, „template NAME from BASE feld=SCHRITT“ "
            "und „correct TASK step=N“. Vorlagen laufen als „mach NAME feld=WERT“. "
            "HSSLM-Status zeigt den Sprachkern; F8 beendet Aufnahmen."
            if german
            else "Commands: ‘teach <task>’, ‘do <task>’, ‘explain <task>’, "
            "‘compose NAME = TASK + TASK’, ‘template NAME from BASE field=STEP’, "
            "and ‘correct TASK step=N’. Run templates as ‘do NAME field=VALUE’. "
            "‘HSSLM status’ shows the language core; F8 finishes recordings."
        )
        return self._reply(OK, "help", None, message, {}, source_text)

    def _status(self, german: bool, source_text: str) -> AssistantReply:
        runtime = getattr(self.language, "runtime", None)
        status_method = getattr(runtime, "status", None)
        details: dict[str, Any]
        if callable(status_method):
            try:
                raw = status_method()
                if is_dataclass(raw) and not isinstance(raw, type):
                    details = asdict(raw)
                elif isinstance(raw, Mapping):
                    details = dict(raw)
                else:
                    details = {"value": str(raw)}
            except Exception as exc:
                details = {"ready": False, "reason": str(exc)}
        else:
            details = {
                "ready": False,
                "reason": "no inspectable language runtime",
            }
        ready = bool(details.get("ready", False))
        parameter_count = details.get("parameter_count")
        parameter_text = (
            f", {int(parameter_count):,} Parameter"
            if parameter_count is not None
            else ""
        )
        if german:
            message = (
                f"HSSLM ist {'bereit' if ready else 'nicht bereit'}{parameter_text}."
            )
        else:
            message = f"HSSLM is {'ready' if ready else 'not ready'}{parameter_text}."
        return self._reply(OK, "status", None, message, details, source_text)

    def _reply(
        self,
        status: ReplyStatus,
        intent: Intent,
        task: str | None,
        message: str,
        data: Mapping[str, Any],
        source_text: str,
    ) -> AssistantReply:
        reply = AssistantReply(status, intent, task, message, data)
        if self.language is None:
            return reply
        renderer = getattr(self.language, "render", None)
        if not callable(renderer):
            return reply
        facts: dict[str, Any] = {
            "status": status,
            "intent": intent,
            "task": task or "",
            "message": message,
            "request": source_text,
        }
        facts.update(data)
        try:
            rendered = renderer(intent, facts)
        except TypeError:
            # Backward-compatible structural adapter used by early callers.
            try:
                rendered = renderer(reply)
            except Exception:
                return reply
        except Exception:
            return reply
        if isinstance(rendered, str) and rendered.strip():
            return replace(reply, text=rendered.strip())
        return reply


__all__ = [
    "AMBIGUOUS",
    "ERROR",
    "NEEDS_INPUT",
    "OK",
    "UNKNOWN",
    "AssistantReply",
    "DemonstrationRecorder",
    "FertigAssistant",
    "Intent",
    "IntentResolution",
    "LanguageInterface",
    "ReplyStatus",
]
