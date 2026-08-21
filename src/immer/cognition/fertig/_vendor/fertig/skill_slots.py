"""Deterministic value slots for demonstrated desktop skills.

The visual planner owns *which* action runs and *where* it applies.  This
module permits a caller to replace the literal payload of explicitly declared
``text`` steps, without asking a language model to generate an action or
mutating the learned :class:`~fertig.screen_model.ScreenTaskModel`.

Invocations are intentionally explicit::

    send monthly report recipient="Ada Lovelace" subject="August report"

Everything before the first ``slot=value`` token is the template name.  Values
with whitespace must be quoted according to normal shell quoting rules.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shlex
import tempfile
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence
import unicodedata

from fertig.desktop import DesktopAction


STORE_SCHEMA = "fertig.desktop.skill-templates"
STORE_VERSION = 1
MAX_SLOT_VALUE_LENGTH = 10_000
MAX_NAME_LENGTH = 256

_SLOT_NAME = re.compile(r"[a-z][a-z0-9_-]{0,63}")


class SkillTemplateError(ValueError):
    """Base class for invalid templates and invocations."""


class TemplateDefinitionError(SkillTemplateError):
    """A template or slot definition violates the schema."""


class InvocationError(SkillTemplateError):
    """Base class for deterministic invocation failures."""


class InvocationSyntaxError(InvocationError):
    """An invocation does not use ``template slot=value`` syntax."""


class TemplateNotFoundError(InvocationError):
    """An invocation names no stored template."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"unknown skill template {name!r}")


class MissingSlotError(InvocationError):
    """One or more required values are absent."""

    def __init__(self, slots: Iterable[str]) -> None:
        self.slots = tuple(sorted(set(slots)))
        super().__init__(f"missing required slot(s): {', '.join(self.slots)}")


class UnknownSlotError(InvocationError):
    """One or more values do not belong to the selected template."""

    def __init__(self, slots: Iterable[str]) -> None:
        self.slots = tuple(sorted(set(slots)))
        super().__init__(f"unknown slot(s): {', '.join(self.slots)}")


class DuplicateSlotError(InvocationError):
    """A slot occurs more than once in one invocation."""

    def __init__(self, slots: Iterable[str]) -> None:
        self.slots = tuple(sorted(set(slots)))
        super().__init__(f"duplicate slot(s): {', '.join(self.slots)}")


class SlotValueError(InvocationError):
    """A supplied slot value is unsafe or outside the supported bounds."""

    def __init__(self, slot: str, reason: str) -> None:
        self.slot = slot
        self.reason = reason
        super().__init__(f"invalid value for slot {slot!r}: {reason}")


class TemplateApplicationError(SkillTemplateError):
    """A template cannot be applied to the supplied desktop action sequence."""


def _normalise_template_name(name: str, *, field: str = "template name") -> str:
    if not isinstance(name, str):
        raise TemplateDefinitionError(f"{field} must be a string")
    value = " ".join(name.strip().casefold().split())
    if not value:
        raise TemplateDefinitionError(f"{field} must not be empty")
    if len(value) > MAX_NAME_LENGTH:
        raise TemplateDefinitionError(
            f"{field} is limited to {MAX_NAME_LENGTH} characters"
        )
    if "=" in value:
        raise TemplateDefinitionError(f"{field} must not contain '='")
    if any(unicodedata.category(character) == "Cc" for character in value):
        raise TemplateDefinitionError(f"{field} must not contain control characters")
    return value


def _normalise_slot_name(name: str) -> str:
    if not isinstance(name, str):
        raise TemplateDefinitionError("slot name must be a string")
    value = name.strip().casefold()
    if not _SLOT_NAME.fullmatch(value):
        raise TemplateDefinitionError("slot name must match [a-z][a-z0-9_-]{0,63}")
    return value


def _validate_value(slot: str, value: object) -> str:
    if not isinstance(value, str):
        raise SlotValueError(slot, "value must be a string")
    if not value:
        raise SlotValueError(slot, "value must not be empty")
    if len(value) > MAX_SLOT_VALUE_LENGTH:
        raise SlotValueError(
            slot, f"value is limited to {MAX_SLOT_VALUE_LENGTH} characters"
        )
    if any(unicodedata.category(character) == "Cc" for character in value):
        raise SlotValueError(slot, "control characters are not allowed")
    return value


@dataclass(frozen=True)
class TextSlot:
    """One declared replacement for one demonstrated ``text`` action."""

    name: str
    step_index: int
    example: str
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalise_slot_name(self.name))
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, int):
            raise TemplateDefinitionError("slot step_index must be an integer")
        if self.step_index < 0:
            raise TemplateDefinitionError("slot step_index must be non-negative")
        if not isinstance(self.required, bool):
            raise TemplateDefinitionError("slot required must be a boolean")
        try:
            example = _validate_value(self.name, self.example)
        except SlotValueError as exc:
            raise TemplateDefinitionError(str(exc)) from exc
        object.__setattr__(self, "example", example)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "step_index": self.step_index,
            "example": self.example,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TextSlot:
        if not isinstance(value, Mapping):
            raise TemplateDefinitionError("stored slot must be an object")
        allowed = {"name", "step_index", "example", "required"}
        extra = set(value) - allowed
        if extra:
            raise TemplateDefinitionError(f"unknown slot fields: {sorted(extra)}")
        missing = {"name", "step_index", "example"} - set(value)
        if missing:
            raise TemplateDefinitionError(f"missing slot fields: {sorted(missing)}")
        return cls(
            name=value["name"],
            step_index=value["step_index"],
            example=value["example"],
            required=value.get("required", True),
        )


def _pairs(
    values: Mapping[str, str] | Iterable[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    if isinstance(values, Mapping):
        raw = tuple(values.items())
    else:
        if isinstance(values, (str, bytes)):
            raise InvocationSyntaxError(
                "slot values must be a mapping or name/value pairs"
            )
        try:
            raw = tuple(values)
        except TypeError as exc:
            raise InvocationSyntaxError(
                "slot values must be a mapping or name/value pairs"
            ) from exc
    result: list[tuple[str, str]] = []
    for pair in raw:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise InvocationSyntaxError("each slot value must be a name/value pair")
        raw_name, raw_value = pair
        try:
            name = _normalise_slot_name(raw_name)
        except TemplateDefinitionError as exc:
            raise InvocationSyntaxError(str(exc)) from exc
        result.append((name, raw_value))
    return tuple(result)


@dataclass(frozen=True, init=False)
class SkillTemplate:
    """A named base task with a finite set of replaceable text steps."""

    name: str
    base_task: str
    slots: tuple[TextSlot, ...]

    def __init__(
        self, name: str, base_task: str, slots: Sequence[TextSlot] = ()
    ) -> None:
        normalised_name = _normalise_template_name(name)
        normalised_base = _normalise_template_name(base_task, field="base task")
        normalised_slots = tuple(slots)
        if not all(isinstance(slot, TextSlot) for slot in normalised_slots):
            raise TemplateDefinitionError("all slots must be TextSlot objects")
        names = [slot.name for slot in normalised_slots]
        duplicate_names = {name for name in names if names.count(name) > 1}
        if duplicate_names:
            raise TemplateDefinitionError(
                f"duplicate slot name(s): {', '.join(sorted(duplicate_names))}"
            )
        indexes = [slot.step_index for slot in normalised_slots]
        duplicate_indexes = {index for index in indexes if indexes.count(index) > 1}
        if duplicate_indexes:
            raise TemplateDefinitionError(
                "only one text slot may target each step; duplicate step(s): "
                + ", ".join(str(index) for index in sorted(duplicate_indexes))
            )
        object.__setattr__(self, "name", normalised_name)
        object.__setattr__(self, "base_task", normalised_base)
        object.__setattr__(
            self,
            "slots",
            tuple(sorted(normalised_slots, key=lambda slot: slot.step_index)),
        )

    def _effective_values(
        self, values: Mapping[str, str] | Iterable[tuple[str, str]]
    ) -> dict[str, str]:
        pairs = _pairs(values)
        names = [name for name, _ in pairs]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise DuplicateSlotError(duplicates)
        slots = {slot.name: slot for slot in self.slots}
        unknown = set(names) - set(slots)
        if unknown:
            raise UnknownSlotError(unknown)
        supplied = {name: _validate_value(name, value) for name, value in pairs}
        missing = {
            slot.name
            for slot in self.slots
            if slot.required and slot.name not in supplied
        }
        if missing:
            raise MissingSlotError(missing)
        return {slot.name: supplied.get(slot.name, slot.example) for slot in self.slots}

    def bind(
        self, values: Mapping[str, str] | Iterable[tuple[str, str]]
    ) -> dict[int, str]:
        """Validate values and return the replacement text for each step.

        Optional slots omitted by the caller retain their demonstrated
        ``example`` text.  The returned mapping is new and has no reference to
        a stored screen model.
        """

        effective = self._effective_values(values)
        return {slot.step_index: effective[slot.name] for slot in self.slots}

    def invoke(
        self, values: Mapping[str, str] | Iterable[tuple[str, str]]
    ) -> TemplateInvocation:
        effective = self._effective_values(values)
        step_text = {slot.step_index: effective[slot.name] for slot in self.slots}
        return TemplateInvocation(
            self.name,
            self.base_task,
            effective,
            step_text,
            {slot.step_index: slot.example for slot in self.slots},
        )

    def apply_actions(
        self,
        actions: Sequence[DesktopAction],
        values: Mapping[str, str] | Iterable[tuple[str, str]],
    ) -> tuple[DesktopAction, ...]:
        """Return a parameterised action copy after verifying example text."""

        return self.invoke(values).apply_actions(actions)

    def to_dict(self) -> dict[str, object]:
        return {
            "base_task": self.base_task,
            "slots": [slot.to_dict() for slot in self.slots],
        }

    @classmethod
    def from_dict(cls, name: str, value: Mapping[str, object]) -> SkillTemplate:
        if not isinstance(value, Mapping):
            raise TemplateDefinitionError("stored template must be an object")
        allowed = {"base_task", "slots"}
        extra = set(value) - allowed
        if extra:
            raise TemplateDefinitionError(f"unknown template fields: {sorted(extra)}")
        if "base_task" not in value or "slots" not in value:
            raise TemplateDefinitionError("stored template needs base_task and slots")
        raw_slots = value["slots"]
        if not isinstance(raw_slots, list):
            raise TemplateDefinitionError("stored template slots must be a list")
        return cls(
            name,
            value["base_task"],
            tuple(TextSlot.from_dict(slot) for slot in raw_slots),
        )


@dataclass(frozen=True, init=False)
class TemplateInvocation:
    """A fully validated, auditable template invocation."""

    template_name: str
    base_task: str
    values: Mapping[str, str]
    step_text: Mapping[int, str]
    _examples: Mapping[int, str]

    def __init__(
        self,
        template_name: str,
        base_task: str,
        values: Mapping[str, str],
        step_text: Mapping[int, str],
        examples: Mapping[int, str],
    ) -> None:
        object.__setattr__(self, "template_name", template_name)
        object.__setattr__(self, "base_task", base_task)
        object.__setattr__(self, "values", MappingProxyType(dict(values)))
        object.__setattr__(self, "step_text", MappingProxyType(dict(step_text)))
        object.__setattr__(self, "_examples", MappingProxyType(dict(examples)))

    def apply_action(self, step_index: int, action: DesktopAction) -> DesktopAction:
        """Resolve one planner action at the execution seam."""

        if step_index not in self.step_text:
            return action
        _verify_text_action(step_index, action, self._examples[step_index])
        return DesktopAction.enter_text(self.step_text[step_index])

    def apply_actions(
        self, actions: Sequence[DesktopAction]
    ) -> tuple[DesktopAction, ...]:
        """Return a new sequence with only declared text steps replaced."""

        original = tuple(actions)
        for index, action in enumerate(original):
            if not isinstance(action, DesktopAction):
                raise TypeError("all actions must be DesktopAction objects")
            if index in self.step_text:
                _verify_text_action(index, action, self._examples[index])
        missing_indexes = set(self.step_text) - set(range(len(original)))
        if missing_indexes:
            missing = ", ".join(str(index) for index in sorted(missing_indexes))
            raise TemplateApplicationError(
                f"slot step index outside action sequence: {missing}"
            )
        return apply_actions(original, self.step_text)


def _verify_text_action(index: int, action: DesktopAction, example: str) -> None:
    if not isinstance(action, DesktopAction):
        raise TypeError("all actions must be DesktopAction objects")
    if action.kind != "text":
        raise TemplateApplicationError(
            f"slot step {index} targets {action.kind!r}, not a text action"
        )
    if action.text != example:
        raise TemplateApplicationError(
            f"slot step {index} expected demonstrated text {example!r}, "
            f"got {action.text!r}"
        )


def apply_actions(
    actions: Sequence[DesktopAction], step_text: Mapping[int, str]
) -> tuple[DesktopAction, ...]:
    """Copy ``actions`` and replace only indexes present in ``step_text``.

    This low-level helper is the integration seam for callers that already
    hold a validated :meth:`SkillTemplate.bind` result.  Prefer
    :meth:`SkillTemplate.apply_actions` when the demonstrated example text is
    available, because it additionally detects a base-task mismatch.
    """

    result = list(actions)
    if not all(isinstance(action, DesktopAction) for action in result):
        raise TypeError("all actions must be DesktopAction objects")
    for index, value in sorted(step_text.items()):
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise TemplateApplicationError(
                "replacement step indexes must be non-negative integers"
            )
        if index >= len(result):
            raise TemplateApplicationError(
                f"slot step index outside action sequence: {index}"
            )
        action = result[index]
        if action.kind != "text":
            raise TemplateApplicationError(
                f"slot step {index} targets {action.kind!r}, not a text action"
            )
        replacement = _validate_value(str(index), value)
        result[index] = DesktopAction.enter_text(replacement)
    return tuple(result)


class TemplateStore:
    """Named skill templates with optional atomic JSON persistence."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = None if path is None else Path(path)
        self._templates: dict[str, SkillTemplate] = {}
        if self.path is not None and self.path.exists():
            self._load()

    def list(self) -> tuple[str, ...]:
        return tuple(sorted(self._templates))

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        try:
            normalised = _normalise_template_name(name)
        except TemplateDefinitionError:
            return False
        return normalised in self._templates

    def get(self, name: str) -> SkillTemplate | None:
        return self._templates.get(_normalise_template_name(name))

    def require(self, name: str) -> SkillTemplate:
        normalised = _normalise_template_name(name)
        try:
            return self._templates[normalised]
        except KeyError as exc:
            raise TemplateNotFoundError(normalised) from exc

    def put(self, template: SkillTemplate) -> None:
        if not isinstance(template, SkillTemplate):
            raise TypeError("put expects a SkillTemplate")
        self._templates[template.name] = template
        self.save()

    def resolve(self, utterance: str) -> TemplateInvocation:
        """Parse and bind one explicit natural-language invocation."""

        if not isinstance(utterance, str):
            raise InvocationSyntaxError("invocation must be a string")
        try:
            tokens = shlex.split(utterance, comments=False, posix=True)
        except ValueError as exc:
            raise InvocationSyntaxError(f"invalid quoting: {exc}") from exc
        if not tokens:
            raise InvocationSyntaxError("invocation must not be empty")

        first_assignment = next(
            (index for index, token in enumerate(tokens) if "=" in token),
            len(tokens),
        )
        if first_assignment == 0:
            raise InvocationSyntaxError("invocation must start with a template name")
        raw_name = " ".join(tokens[:first_assignment])
        try:
            template = self.require(raw_name)
        except TemplateDefinitionError as exc:
            raise InvocationSyntaxError(str(exc)) from exc

        pairs: list[tuple[str, str]] = []
        for token in tokens[first_assignment:]:
            if "=" not in token:
                raise InvocationSyntaxError(
                    "every token after the template name must use slot=value"
                )
            name, value = token.split("=", 1)
            if not name:
                raise InvocationSyntaxError("slot name before '=' must not be empty")
            pairs.append((name, value))
        return template.invoke(pairs)

    def _payload(self) -> dict[str, object]:
        return {
            "schema": STORE_SCHEMA,
            "version": STORE_VERSION,
            "templates": {
                name: template.to_dict()
                for name, template in sorted(self._templates.items())
            },
        }

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self._payload(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        descriptor, raw_temp = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temp_path = Path(raw_temp)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
        finally:
            temp_path.unlink(missing_ok=True)

    def _load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TemplateDefinitionError(
                f"could not read skill template store: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise TemplateDefinitionError("skill template store must be an object")
        if (
            payload.get("schema") != STORE_SCHEMA
            or payload.get("version") != STORE_VERSION
        ):
            raise TemplateDefinitionError("unsupported skill template store")
        raw_templates = payload.get("templates")
        if not isinstance(raw_templates, dict):
            raise TemplateDefinitionError(
                "skill template store has no template mapping"
            )
        self._templates = {
            _normalise_template_name(name): SkillTemplate.from_dict(name, value)
            for name, value in raw_templates.items()
        }
