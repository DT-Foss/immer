"""Language-agnostic induction of deterministic two-slot warm programs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import re
from typing import Literal

from .identity import canonical_json_bytes, require_sha256


MultiSlotMode = Literal["identity", "upper", "lower", "casefold"]
_MODES: tuple[MultiSlotMode, ...] = (
    "identity",
    "upper",
    "lower",
    "casefold",
)
_TOKEN = re.compile(r"[A-Za-z0-9_]+(?:[./:-][A-Za-z0-9_]+)*")
MAX_LEXICAL_SPANS = 64
MAX_DERIVED_OBSERVATIONS = 4096
MAX_MULTISLOT_OBSERVATIONS = 8192
MAX_MULTISLOT_QUESTION_CHARS = 2048


class MultiSlotWarmError(RuntimeError):
    """A two-slot observation or executable program failed integrity."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _apply(mode: MultiSlotMode, value: str) -> str:
    if mode == "identity":
        return value
    if mode == "upper":
        return value.upper()
    if mode == "lower":
        return value.lower()
    if mode == "casefold":
        return value.casefold()
    raise AssertionError(mode)


def _printable_ascii(value: str) -> bool:
    return not value or (value.isascii() and value.isprintable())


def _occurrences(source: str, needle: str) -> tuple[int, ...]:
    if not needle:
        return ()
    rows = []
    start = 0
    while len(rows) < MAX_LEXICAL_SPANS:
        index = source.find(needle, start)
        if index < 0:
            break
        rows.append(index)
        start = index + 1
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class MultiSlotObservation:
    question_prefix: str
    question_middle: str
    question_suffix: str
    output_prefix: str
    output_middle: str
    output_suffix: str
    output_order: tuple[int, int]
    modes: tuple[MultiSlotMode, MultiSlotMode]
    slot_values: tuple[str, str]
    slot_sha256s: tuple[str, str]
    question_sha256: str
    output_sha256: str
    cell_payload_sha256: str
    teacher_forward_count: int

    def __post_init__(self) -> None:
        for field in (
            "question_prefix",
            "question_middle",
            "question_suffix",
            "output_prefix",
            "output_middle",
            "output_suffix",
        ):
            if not isinstance(getattr(self, field), str):
                raise TypeError(f"{field} must be text")
        if len(
            self.question_prefix.strip()
            + self.question_middle.strip()
            + self.question_suffix.strip()
        ) < 4:
            raise ValueError("two-slot question template lacks static context")
        if not self.question_middle:
            raise ValueError("two-slot question template needs a slot separator")
        if not all(
            _printable_ascii(value)
            for value in (
                self.question_prefix,
                self.question_middle,
                self.question_suffix,
                self.output_prefix,
                self.output_middle,
                self.output_suffix,
            )
        ):
            raise ValueError("two-slot program fragments must be printable ASCII")
        if self.output_order not in ((0, 1), (1, 0)):
            raise ValueError("output_order must be a two-slot permutation")
        if (
            not isinstance(self.modes, tuple)
            or len(self.modes) != 2
            or any(mode not in _MODES for mode in self.modes)
        ):
            raise ValueError("two-slot transform modes are invalid")
        if (
            not isinstance(self.slot_values, tuple)
            or len(self.slot_values) != 2
            or any(
                not isinstance(value, str)
                or _TOKEN.fullmatch(value) is None
                for value in self.slot_values
            )
        ):
            raise ValueError("two-slot observation needs two lexical slot values")
        if not isinstance(self.slot_sha256s, tuple) or len(self.slot_sha256s) != 2:
            raise ValueError("two-slot observation needs two slot hashes")
        object.__setattr__(
            self,
            "slot_sha256s",
            tuple(
                require_sha256(value, field="slot_sha256")
                for value in self.slot_sha256s
            ),
        )
        for field in (
            "question_sha256",
            "output_sha256",
            "cell_payload_sha256",
        ):
            object.__setattr__(
                self,
                field,
                require_sha256(getattr(self, field), field=field),
            )
        expected_slot_sha256s = tuple(
            hashlib.sha256(value.encode("utf-8")).hexdigest()
            for value in self.slot_values
        )
        if self.slot_sha256s != expected_slot_sha256s:
            raise ValueError("two-slot values differ from their hashes")
        if (
            len(self.question) > MAX_MULTISLOT_QUESTION_CHARS
            or hashlib.sha256(self.question.encode("utf-8")).hexdigest()
            != self.question_sha256
        ):
            raise ValueError("two-slot question differs from its binding")
        if (
            hashlib.sha256(self.output.encode("utf-8")).hexdigest()
            != self.output_sha256
        ):
            raise ValueError("two-slot output differs from its binding")
        if (
            isinstance(self.teacher_forward_count, bool)
            or not isinstance(self.teacher_forward_count, int)
            or self.teacher_forward_count <= 0
        ):
            raise ValueError("teacher_forward_count must be positive")

    @property
    def template_key(self) -> tuple[object, ...]:
        return (
            self.question_prefix,
            self.question_middle,
            self.question_suffix,
            self.output_prefix,
            self.output_middle,
            self.output_suffix,
            self.output_order,
            self.modes,
        )

    @property
    def question(self) -> str:
        return (
            self.question_prefix
            + self.slot_values[0]
            + self.question_middle
            + self.slot_values[1]
            + self.question_suffix
        )

    @property
    def output(self) -> str:
        transformed = (
            _apply(self.modes[0], self.slot_values[0]),
            _apply(self.modes[1], self.slot_values[1]),
        )
        first, second = self.output_order
        return (
            self.output_prefix
            + transformed[first]
            + self.output_middle
            + transformed[second]
            + self.output_suffix
        )

    @property
    def source_key(self) -> tuple[str, str, str, int]:
        return (
            self.question_sha256,
            self.output_sha256,
            self.cell_payload_sha256,
            self.teacher_forward_count,
        )

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "cell_payload_sha256": self.cell_payload_sha256,
            "modes": list(self.modes),
            "output_middle": self.output_middle,
            "output_order": list(self.output_order),
            "output_prefix": self.output_prefix,
            "output_sha256": self.output_sha256,
            "output_suffix": self.output_suffix,
            "question_middle": self.question_middle,
            "question_prefix": self.question_prefix,
            "question_sha256": self.question_sha256,
            "question_suffix": self.question_suffix,
            "slot_values": list(self.slot_values),
            "slot_sha256s": list(self.slot_sha256s),
            "teacher_forward_count": self.teacher_forward_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> "MultiSlotObservation":
        expected = {
            "cell_payload_sha256",
            "modes",
            "output_middle",
            "output_order",
            "output_prefix",
            "output_sha256",
            "output_suffix",
            "question_middle",
            "question_prefix",
            "question_sha256",
            "question_suffix",
            "slot_values",
            "slot_sha256s",
            "teacher_forward_count",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise MultiSlotWarmError("two-slot observation document is invalid")
        try:
            return cls(
                question_prefix=value["question_prefix"],
                question_middle=value["question_middle"],
                question_suffix=value["question_suffix"],
                output_prefix=value["output_prefix"],
                output_middle=value["output_middle"],
                output_suffix=value["output_suffix"],
                output_order=tuple(value["output_order"]),
                modes=tuple(value["modes"]),
                slot_values=tuple(value["slot_values"]),
                slot_sha256s=tuple(value["slot_sha256s"]),
                question_sha256=value["question_sha256"],
                output_sha256=value["output_sha256"],
                cell_payload_sha256=value["cell_payload_sha256"],
                teacher_forward_count=value["teacher_forward_count"],
            )
        except (TypeError, ValueError) as exc:
            raise MultiSlotWarmError(
                "two-slot observation failed validation"
            ) from exc


@dataclass(frozen=True, slots=True)
class MultiSlotProgram:
    question_prefix: str
    question_middle: str
    question_suffix: str
    output_prefix: str
    output_middle: str
    output_suffix: str
    output_order: tuple[int, int]
    modes: tuple[MultiSlotMode, MultiSlotMode]
    observation_sha256s: tuple[str, ...]
    distinct_slot_tuples: int
    saved_qwen_forwards: int

    def __post_init__(self) -> None:
        for field in (
            "question_prefix",
            "question_middle",
            "question_suffix",
            "output_prefix",
            "output_middle",
            "output_suffix",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not _printable_ascii(value):
                raise ValueError("two-slot program fragments must be printable ASCII")
        if len(
            self.question_prefix.strip()
            + self.question_middle.strip()
            + self.question_suffix.strip()
        ) < 4:
            raise ValueError("two-slot program lacks static context")
        if self.output_order not in ((0, 1), (1, 0)):
            raise ValueError("two-slot program output order is invalid")
        if (
            not isinstance(self.modes, tuple)
            or len(self.modes) != 2
            or any(mode not in _MODES for mode in self.modes)
        ):
            raise ValueError("two-slot program modes are invalid")
        if not self.question_middle:
            raise ValueError("two-slot program needs a slot separator")
        if (
            not isinstance(self.observation_sha256s, tuple)
            or len(self.observation_sha256s) < 2
            or len(set(self.observation_sha256s))
            != len(self.observation_sha256s)
        ):
            raise ValueError("two-slot program observations are invalid")
        object.__setattr__(
            self,
            "observation_sha256s",
            tuple(
                require_sha256(value, field="observation_sha256")
                for value in self.observation_sha256s
            ),
        )
        if (
            isinstance(self.distinct_slot_tuples, bool)
            or not isinstance(self.distinct_slot_tuples, int)
            or self.distinct_slot_tuples < 2
            or self.distinct_slot_tuples > len(self.observation_sha256s)
        ):
            raise ValueError("two-slot program support is invalid")
        if (
            isinstance(self.saved_qwen_forwards, bool)
            or not isinstance(self.saved_qwen_forwards, int)
            or self.saved_qwen_forwards <= 0
        ):
            raise ValueError("two-slot program saved forwards are invalid")

    @property
    def sha256(self) -> str:
        return _digest(
            {
                "distinct_slot_tuples": self.distinct_slot_tuples,
                "modes": list(self.modes),
                "observation_sha256s": list(self.observation_sha256s),
                "output_middle": self.output_middle,
                "output_order": list(self.output_order),
                "output_prefix": self.output_prefix,
                "output_suffix": self.output_suffix,
                "question_middle": self.question_middle,
                "question_prefix": self.question_prefix,
                "question_suffix": self.question_suffix,
                "saved_qwen_forwards": self.saved_qwen_forwards,
                "schema": "immer.qwen3.8-multislot-warm-program/v1",
            }
        )

    def match(self, question: str) -> str | None:
        if not question.startswith(self.question_prefix) or not question.endswith(
            self.question_suffix
        ):
            return None
        end = (
            len(question) - len(self.question_suffix)
            if self.question_suffix
            else len(question)
        )
        body = question[len(self.question_prefix) : end]
        if body.count(self.question_middle) != 1:
            return None
        slots = body.split(self.question_middle, 1)
        if (
            len(slots) != 2
            or any(_TOKEN.fullmatch(value) is None for value in slots)
        ):
            return None
        transformed = (
            _apply(self.modes[0], slots[0]),
            _apply(self.modes[1], slots[1]),
        )
        first, second = self.output_order
        return (
            self.output_prefix
            + transformed[first]
            + self.output_middle
            + transformed[second]
            + self.output_suffix
        )


def derive_multislot_observations(
    question: str,
    output: str,
    *,
    question_sha256: str,
    cell_payload_sha256: str,
    teacher_forward_count: int,
) -> tuple[MultiSlotObservation, ...]:
    source = question.strip()
    target = output.strip()
    if (
        not source
        or not target
        or len(source) > MAX_MULTISLOT_QUESTION_CHARS
        or not source.isascii()
        or not source.isprintable()
        or not target.isascii()
        or not target.isprintable()
    ):
        return ()
    spans = tuple(_TOKEN.finditer(source))[:MAX_LEXICAL_SPANS]
    rows: dict[str, MultiSlotObservation] = {}
    for left_index, left in enumerate(spans):
        for right in spans[left_index + 1 :]:
            if left.end() > right.start():
                continue
            slot_values = (left.group(0), right.group(0))
            question_prefix = source[: left.start()]
            question_middle = source[left.end() : right.start()]
            question_suffix = source[right.end() :]
            if not question_middle:
                continue
            for first_mode in _MODES:
                for second_mode in _MODES:
                    transformed = (
                        _apply(first_mode, slot_values[0]),
                        _apply(second_mode, slot_values[1]),
                    )
                    for order in ((0, 1), (1, 0)):
                        first_value = transformed[order[0]]
                        second_value = transformed[order[1]]
                        for first_start in _occurrences(target, first_value):
                            first_end = first_start + len(first_value)
                            for second_start in _occurrences(target, second_value):
                                if second_start < first_end:
                                    continue
                                second_end = second_start + len(second_value)
                                try:
                                    row = MultiSlotObservation(
                                        question_prefix=question_prefix,
                                        question_middle=question_middle,
                                        question_suffix=question_suffix,
                                        output_prefix=target[:first_start],
                                        output_middle=target[first_end:second_start],
                                        output_suffix=target[second_end:],
                                        output_order=order,
                                        modes=(first_mode, second_mode),
                                        slot_values=slot_values,
                                        slot_sha256s=(
                                            hashlib.sha256(
                                                slot_values[0].encode("utf-8")
                                            ).hexdigest(),
                                            hashlib.sha256(
                                                slot_values[1].encode("utf-8")
                                            ).hexdigest(),
                                        ),
                                        question_sha256=question_sha256,
                                        output_sha256=hashlib.sha256(
                                            target.encode("utf-8")
                                        ).hexdigest(),
                                        cell_payload_sha256=cell_payload_sha256,
                                        teacher_forward_count=teacher_forward_count,
                                    )
                                except ValueError:
                                    continue
                                rows.setdefault(row.sha256, row)
                                if len(rows) >= MAX_DERIVED_OBSERVATIONS:
                                    return ()
    return tuple(row for _sha, row in sorted(rows.items()))


def promote_multislot(
    observations: Sequence[MultiSlotObservation],
    *,
    minimum_distinct_slots: int = 2,
) -> tuple[MultiSlotProgram, ...]:
    if minimum_distinct_slots < 2:
        raise ValueError("minimum_distinct_slots must be at least two")
    groups: dict[tuple[object, ...], list[MultiSlotObservation]] = {}
    for row in observations:
        if not isinstance(row, MultiSlotObservation):
            raise TypeError("observations must contain MultiSlotObservation values")
        groups.setdefault(row.template_key, []).append(row)
    programs = []
    for key, rows in groups.items():
        distinct = {row.slot_sha256s for row in rows}
        distinct_left = {row.slot_sha256s[0] for row in rows}
        distinct_right = {row.slot_sha256s[1] for row in rows}
        distinct_questions = {row.question_sha256 for row in rows}
        distinct_cells = {row.cell_payload_sha256 for row in rows}
        if (
            len(distinct) < minimum_distinct_slots
            or len(distinct_left) < minimum_distinct_slots
            or len(distinct_right) < minimum_distinct_slots
            or len(distinct_questions) < minimum_distinct_slots
            or len(distinct_cells) < minimum_distinct_slots
        ):
            continue
        (
            question_prefix,
            question_middle,
            question_suffix,
            output_prefix,
            output_middle,
            output_suffix,
            output_order,
            modes,
        ) = key
        programs.append(
            MultiSlotProgram(
                question_prefix=question_prefix,
                question_middle=question_middle,
                question_suffix=question_suffix,
                output_prefix=output_prefix,
                output_middle=output_middle,
                output_suffix=output_suffix,
                output_order=output_order,
                modes=modes,
                observation_sha256s=tuple(sorted(row.sha256 for row in rows)),
                distinct_slot_tuples=len(distinct),
                saved_qwen_forwards=min(
                    row.teacher_forward_count for row in rows
                ),
            )
        )
    return tuple(sorted(programs, key=lambda row: row.sha256))


__all__ = [
    "MAX_DERIVED_OBSERVATIONS",
    "MAX_MULTISLOT_OBSERVATIONS",
    "MAX_MULTISLOT_QUESTION_CHARS",
    "MultiSlotObservation",
    "MultiSlotProgram",
    "MultiSlotWarmError",
    "derive_multislot_observations",
    "promote_multislot",
]
