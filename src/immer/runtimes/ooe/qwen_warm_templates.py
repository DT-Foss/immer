"""Learn deterministic parametric warm programs from ordinary cold chats."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import threading
from typing import Any, Iterator, Literal

try:  # pragma: no cover - production platforms provide fcntl.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from ...contracts import ExecutionStatus, Result
from .chat import OoeChatAttempt, OoeChatIntegrityError
from .controller import WarmAccountingReceipt
from .crystal import CrystalStore
from .identity import canonical_json_bytes, require_sha256
from .qwen_warm_multislot import (
    MAX_MULTISLOT_OBSERVATIONS,
    MultiSlotObservation,
    MultiSlotProgram,
    MultiSlotWarmError,
    derive_multislot_observations,
    promote_multislot,
)


TEMPLATE_STATE_SCHEMA = "immer.qwen3.8-parametric-warm-state/v2"
LEGACY_TEMPLATE_STATE_SCHEMA = "immer.qwen3.8-parametric-warm-state/v1"
# The logical state name stays stable across schema upgrades. An older binary
# therefore fails on the v2 payload instead of continuing to mutate stale v1
# state under a second name.
TEMPLATE_STATE_NAME = "qwen38-parametric-warm-state-v1"
LEGACY_TEMPLATE_STATE_NAME = TEMPLATE_STATE_NAME
MINIMUM_DISTINCT_SLOTS = 2
MAX_PARAMETRIC_STATE_BYTES = 48 * 1024**2
MAX_IMPORTED_TEMPLATE_CONTENTS = 1024
_MODES = ("identity", "upper", "lower", "casefold")
TemplateMode = Literal["identity", "upper", "lower", "casefold"]


class ParametricWarmError(OoeChatIntegrityError):
    """A parametric warm state or deterministic execution failed integrity."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _apply(mode: TemplateMode, value: str) -> str:
    if mode == "identity":
        return value
    if mode == "upper":
        return value.upper()
    if mode == "lower":
        return value.lower()
    if mode == "casefold":
        return value.casefold()
    raise AssertionError(mode)


@dataclass(frozen=True, slots=True)
class TemplateObservation:
    prefix: str
    suffix: str
    mode: TemplateMode
    slot: str
    output: str
    question_sha256: str
    cell_payload_sha256: str
    teacher_forward_count: int

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError("template mode is invalid")
        if not isinstance(self.prefix, str) or not isinstance(self.suffix, str):
            raise TypeError("template context must be text")
        if not isinstance(self.slot, str) or not self.slot:
            raise ValueError("template slot must be non-empty text")
        if not isinstance(self.output, str) or not self.output:
            raise ValueError("template output must be non-empty text")
        if _apply(self.mode, self.slot) != self.output:
            raise ValueError("template observation output differs from its transform")
        if len(self.prefix.strip()) + len(self.suffix.strip()) < 4:
            raise ValueError("template observation lacks a specific static context")
        object.__setattr__(
            self,
            "question_sha256",
            require_sha256(self.question_sha256, field="question_sha256"),
        )
        object.__setattr__(
            self,
            "cell_payload_sha256",
            require_sha256(
                self.cell_payload_sha256,
                field="cell_payload_sha256",
            ),
        )
        if (
            isinstance(self.teacher_forward_count, bool)
            or not isinstance(self.teacher_forward_count, int)
            or self.teacher_forward_count <= 0
        ):
            raise ValueError("teacher_forward_count must be positive")

    @property
    def template_key(self) -> tuple[str, str, str]:
        return self.prefix, self.suffix, self.mode

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "cell_payload_sha256": self.cell_payload_sha256,
            "mode": self.mode,
            "output": self.output,
            "prefix": self.prefix,
            "question_sha256": self.question_sha256,
            "slot": self.slot,
            "suffix": self.suffix,
            "teacher_forward_count": self.teacher_forward_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> "TemplateObservation":
        if not isinstance(value, Mapping) or set(value) != {
            "cell_payload_sha256",
            "mode",
            "output",
            "prefix",
            "question_sha256",
            "slot",
            "suffix",
            "teacher_forward_count",
        }:
            raise ParametricWarmError("template observation is invalid")
        try:
            return cls(**dict(value))
        except (TypeError, ValueError) as exc:
            raise ParametricWarmError("template observation failed validation") from exc


@dataclass(frozen=True, slots=True)
class PromotedTemplate:
    prefix: str
    suffix: str
    mode: TemplateMode
    observation_sha256s: tuple[str, ...]
    distinct_slots: int
    saved_qwen_forwards: int

    @property
    def sha256(self) -> str:
        return _digest(
            {
                "distinct_slots": self.distinct_slots,
                "mode": self.mode,
                "observation_sha256s": list(self.observation_sha256s),
                "prefix": self.prefix,
                "saved_qwen_forwards": self.saved_qwen_forwards,
                "schema": "immer.qwen3.8-parametric-warm-template/v1",
                "suffix": self.suffix,
            }
        )

    def match(self, question: str) -> str | None:
        if not question.startswith(self.prefix) or not question.endswith(self.suffix):
            return None
        end = len(question) - len(self.suffix) if self.suffix else len(question)
        slot = question[len(self.prefix) : end]
        return None if not slot else _apply(self.mode, slot)


def _candidate_observations(
    question: str,
    output: str,
    *,
    question_sha256: str,
    cell_payload_sha256: str,
    teacher_forward_count: int,
) -> tuple[TemplateObservation, ...]:
    source = question.strip()
    target = output.strip()
    rows: dict[tuple[str, str, str, str], TemplateObservation] = {}
    if not source or not target:
        return ()
    for raw_mode in _MODES:
        mode = raw_mode  # narrow for the dataclass constructor.
        transformed = _apply(mode, source)  # type: ignore[arg-type]
        start = 0
        while True:
            index = transformed.find(target, start)
            if index < 0:
                break
            slot = source[index : index + len(target)]
            if _apply(mode, slot) == target:  # type: ignore[arg-type]
                prefix = source[:index]
                suffix = source[index + len(slot) :]
                try:
                    row = TemplateObservation(
                        prefix=prefix,
                        suffix=suffix,
                        mode=mode,  # type: ignore[arg-type]
                        slot=slot,
                        output=target,
                        question_sha256=question_sha256,
                        cell_payload_sha256=cell_payload_sha256,
                        teacher_forward_count=teacher_forward_count,
                    )
                except ValueError:
                    pass
                else:
                    rows[(prefix, suffix, mode, slot)] = row
            start = index + 1
    return tuple(sorted(rows.values(), key=lambda row: row.sha256))


def _state_bytes(body: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(
        {"body": dict(body), "schema": TEMPLATE_STATE_SCHEMA, "sha256": _digest(body)}
    )


def _validate_multislot_source_groups(
    observations: Sequence[MultiSlotObservation],
) -> None:
    groups: dict[
        tuple[str, str, str, int],
        list[MultiSlotObservation],
    ] = {}
    for row in observations:
        groups.setdefault(row.source_key, []).append(row)
    for rows in groups.values():
        representative = rows[0]
        expected = derive_multislot_observations(
            representative.question,
            representative.output,
            question_sha256=representative.question_sha256,
            cell_payload_sha256=representative.cell_payload_sha256,
            teacher_forward_count=representative.teacher_forward_count,
        )
        if {row.sha256 for row in rows} != {row.sha256 for row in expected}:
            raise ParametricWarmError(
                "multi-slot source lost its complete derived authority"
            )


@contextmanager
def _template_lock(root: Path) -> Iterator[None]:
    descriptor = os.open(
        root / ".qwen-parametric-warm.lock",
        os.O_CREAT
        | os.O_RDWR
        | int(getattr(os, "O_CLOEXEC", 0))
        | int(getattr(os, "O_NOFOLLOW", 0)),
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ParametricWarmError("template lock is not a regular file")
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


class ParametricWarmBank:
    """Promote transformations only after distinct whole-result examples."""

    def __init__(
        self,
        store: CrystalStore,
        runtime_profile_sha256: str,
        *,
        output_character_limit: int,
        observation_verifier: Callable[[object], bool] | None = None,
        prompt_token_verifier: Callable[[str, str], bool] | None = None,
    ) -> None:
        self.store = store
        self.root = Path(store.root)
        self.runtime_profile_sha256 = require_sha256(
            runtime_profile_sha256,
            field="runtime_profile_sha256",
        )
        if (
            isinstance(output_character_limit, bool)
            or not isinstance(output_character_limit, int)
            or output_character_limit <= 0
        ):
            raise ValueError("output_character_limit must be positive")
        self.output_character_limit = output_character_limit
        if observation_verifier is not None and not callable(observation_verifier):
            raise TypeError("observation_verifier must be callable or None")
        self.observation_verifier = observation_verifier
        if prompt_token_verifier is not None and not callable(prompt_token_verifier):
            raise TypeError("prompt_token_verifier must be callable or None")
        self.prompt_token_verifier = prompt_token_verifier
        self._lock = threading.RLock()
        self._load()

    @classmethod
    def open_existing(
        cls,
        store: CrystalStore,
        *,
        observation_verifier: Callable[[object], bool] | None = None,
    ) -> "ParametricWarmBank | None":
        try:
            raw = store.restore_state(TEMPLATE_STATE_NAME)
        except KeyError:
            return None
        try:
            value = json.loads(raw)
            body = value["body"]
            profile = require_sha256(
                body["runtime_profile_sha256"],
                field="runtime_profile_sha256",
            )
            output_character_limit = body["output_character_limit"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ParametricWarmError(
                "existing template state identity is invalid"
            ) from exc
        return cls(
            store,
            profile,
            output_character_limit=output_character_limit,
            observation_verifier=observation_verifier,
        )

    @staticmethod
    def _empty(profile: str, output_character_limit: int) -> dict[str, Any]:
        return {
            "committed_executions": 0,
            "imported_content_sha256s": [],
            "multislot_observations": [],
            "next_transaction": 0,
            "observations": [],
            "output_character_limit": output_character_limit,
            "privacy": "private-executable-template-descriptors/v1",
            "rejected_executions": 0,
            "runtime_profile_sha256": profile,
            "saved_qwen_forwards": 0,
            "schema": TEMPLATE_STATE_SCHEMA,
        }

    @staticmethod
    def _legacy_empty(profile: str, output_character_limit: int) -> dict[str, Any]:
        return {
            "committed_executions": 0,
            "next_transaction": 0,
            "observations": [],
            "output_character_limit": output_character_limit,
            "privacy": "private-executable-template-descriptors/v1",
            "rejected_executions": 0,
            "runtime_profile_sha256": profile,
            "saved_qwen_forwards": 0,
            "schema": LEGACY_TEMPLATE_STATE_SCHEMA,
        }

    def _load(self) -> None:
        try:
            raw = self.store.restore_state(TEMPLATE_STATE_NAME)
        except KeyError:
            self.body = self._empty(
                self.runtime_profile_sha256,
                self.output_character_limit,
            )
            self.state_sha256 = None
            self.legacy_state_sha256 = None
            return
        raw_sha256 = hashlib.sha256(raw).hexdigest()
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ParametricWarmError("template state is invalid JSON") from exc
        envelope_schema = value.get("schema") if isinstance(value, Mapping) else None
        if envelope_schema not in {
            TEMPLATE_STATE_SCHEMA,
            LEGACY_TEMPLATE_STATE_SCHEMA,
        }:
            raise ParametricWarmError("template state schema is invalid")
        legacy = envelope_schema == LEGACY_TEMPLATE_STATE_SCHEMA
        body = value.get("body") if isinstance(value, Mapping) else None
        expected_schema = (
            LEGACY_TEMPLATE_STATE_SCHEMA if legacy else TEMPLATE_STATE_SCHEMA
        )
        expected_body = (
            self._legacy_empty(
                self.runtime_profile_sha256,
                self.output_character_limit,
            )
            if legacy
            else self._empty(
                self.runtime_profile_sha256,
                self.output_character_limit,
            )
        )
        if (
            canonical_json_bytes(value) != raw
            or not isinstance(value, Mapping)
            or set(value) != {"body", "schema", "sha256"}
            or value.get("schema") != expected_schema
            or not isinstance(body, Mapping)
            or set(body) != set(expected_body)
            or body.get("schema") != expected_schema
            or body.get("runtime_profile_sha256") != self.runtime_profile_sha256
            or body.get("output_character_limit")
            != self.output_character_limit
            or body.get("privacy")
            != "private-executable-template-descriptors/v1"
            or value.get("sha256") != _digest(body)
            or not isinstance(body.get("observations"), list)
            or (
                not legacy
                and not isinstance(body.get("multislot_observations"), list)
            )
        ):
            raise ParametricWarmError("template state envelope is invalid")
        observations = [
            TemplateObservation.from_dict(row) for row in body["observations"]
        ]
        try:
            multislot_observations = (
                []
                if legacy
                else [
                    MultiSlotObservation.from_dict(row)
                    for row in body["multislot_observations"]
                ]
            )
        except MultiSlotWarmError as exc:
            raise ParametricWarmError(
                "multi-slot observation failed validation"
            ) from exc
        if len({row.sha256 for row in observations}) != len(observations):
            raise ParametricWarmError("template observation is duplicated")
        if len({row.sha256 for row in multislot_observations}) != len(
            multislot_observations
        ):
            raise ParametricWarmError("multi-slot observation is duplicated")
        if len(multislot_observations) > MAX_MULTISLOT_OBSERVATIONS:
            raise ParametricWarmError("multi-slot observation capacity is exceeded")
        _validate_multislot_source_groups(multislot_observations)
        if self.observation_verifier is not None and any(
            not bool(self.observation_verifier(row))
            for row in (*observations, *multislot_observations)
        ):
            raise ParametricWarmError(
                "template observation lost its exact ResultCell authority"
            )
        normalized = self._empty(
            self.runtime_profile_sha256,
            self.output_character_limit,
        )
        for field in (
            "committed_executions",
            "next_transaction",
            "rejected_executions",
            "saved_qwen_forwards",
        ):
            normalized[field] = body[field]
        normalized["observations"] = [row.to_dict() for row in observations]
        normalized["multislot_observations"] = [
            row.to_dict() for row in multislot_observations
        ]
        imported_content_sha256s = (
            [] if legacy else body["imported_content_sha256s"]
        )
        if (
            not isinstance(imported_content_sha256s, list)
            or len(imported_content_sha256s) > MAX_IMPORTED_TEMPLATE_CONTENTS
        ):
            raise ParametricWarmError("template import inventory is invalid")
        try:
            normalized["imported_content_sha256s"] = sorted(
                {
                    require_sha256(value, field="imported_content_sha256")
                    for value in imported_content_sha256s
                }
            )
        except (TypeError, ValueError) as exc:
            raise ParametricWarmError(
                "template import inventory failed validation"
            ) from exc
        if len(normalized["imported_content_sha256s"]) != len(
            imported_content_sha256s
        ):
            raise ParametricWarmError("template import inventory is duplicated")
        for field in (
            "committed_executions",
            "next_transaction",
            "rejected_executions",
            "saved_qwen_forwards",
        ):
            number = normalized[field]
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise ParametricWarmError("template metrics are invalid")
        self.body = normalized
        self.state_sha256 = raw_sha256
        self.legacy_state_sha256 = raw_sha256 if legacy else None

    @property
    def observations(self) -> tuple[TemplateObservation, ...]:
        return tuple(
            TemplateObservation.from_dict(row) for row in self.body["observations"]
        )

    @property
    def multislot_observations(self) -> tuple[MultiSlotObservation, ...]:
        return tuple(
            MultiSlotObservation.from_dict(row)
            for row in self.body["multislot_observations"]
        )

    @property
    def promoted(self) -> tuple[object, ...]:
        groups: dict[tuple[str, str, str], list[TemplateObservation]] = {}
        for row in self.observations:
            groups.setdefault(row.template_key, []).append(row)
        promoted = []
        for (prefix, suffix, raw_mode), rows in groups.items():
            slots = {row.slot for row in rows}
            if len(slots) < MINIMUM_DISTINCT_SLOTS:
                continue
            promoted.append(
                PromotedTemplate(
                    prefix=prefix,
                    suffix=suffix,
                    mode=raw_mode,  # type: ignore[arg-type]
                    observation_sha256s=tuple(sorted(row.sha256 for row in rows)),
                    distinct_slots=len(slots),
                    saved_qwen_forwards=min(
                        row.teacher_forward_count for row in rows
                    ),
                )
            )
        multislot = promote_multislot(
            self.multislot_observations,
            minimum_distinct_slots=MINIMUM_DISTINCT_SLOTS,
        )
        return tuple(
            sorted((*promoted, *multislot), key=lambda row: row.sha256)
        )

    def _publish(self) -> None:
        payload = _state_bytes(self.body)
        if len(payload) > MAX_PARAMETRIC_STATE_BYTES:
            raise ParametricWarmError("template state exceeds its hard byte limit")
        publication = self.store.publish_state(
            TEMPLATE_STATE_NAME,
            payload,
            expected_sha256=self.state_sha256,
        )
        self.state_sha256 = publication.payload_sha256
        self.legacy_state_sha256 = None

    def import_compatible(self, source: "ParametricWarmBank") -> dict[str, int]:
        if not isinstance(source, ParametricWarmBank):
            raise TypeError("source must be a ParametricWarmBank")
        if source.root.resolve() == self.root.resolve():
            return {
                "capacity_rejected_source_states": 0,
                "imported_multislot_observations": 0,
                "imported_single_slot_observations": 0,
                "source_states": 0,
            }
        source_single = tuple(
            row
            for row in source.observations
            if len(row.output) <= self.output_character_limit
        )
        source_multislot = tuple(
            row
            for row in source.multislot_observations
            if len(row.output) <= self.output_character_limit
        )
        if not source_single and not source_multislot:
            return {
                "capacity_rejected_source_states": 0,
                "imported_multislot_observations": 0,
                "imported_single_slot_observations": 0,
                "source_states": 0,
            }
        source_content_sha256 = _digest(
            {
                "multislot_observations": [
                    row.to_dict() for row in source_multislot
                ],
                "observations": [row.to_dict() for row in source_single],
                "schema": "immer.qwen3.8-parametric-compatible-content/v1",
            }
        )
        if self.observation_verifier is not None and any(
            not bool(self.observation_verifier(row))
            for row in (*source_single, *source_multislot)
        ):
            raise ParametricWarmError(
                "imported template observation lost its ResultCell authority"
            )
        _validate_multislot_source_groups(source_multislot)
        with self._lock, _template_lock(self.root):
            self._load()
            existing_single = {row.sha256: row for row in self.observations}
            existing_multislot = {
                row.sha256: row for row in self.multislot_observations
            }
            single_before = len(existing_single)
            multislot_before = len(existing_multislot)
            missing_single = tuple(
                row
                for row in source_single
                if row.sha256 not in existing_single
            )
            missing_multislot = tuple(
                row
                for row in source_multislot
                if row.sha256 not in existing_multislot
            )
            remaining = max(
                0,
                MAX_MULTISLOT_OBSERVATIONS - len(existing_multislot),
            )
            if len(missing_multislot) > remaining:
                return {
                    "capacity_rejected_source_states": 1,
                    "imported_multislot_observations": 0,
                    "imported_single_slot_observations": 0,
                    "source_states": 0,
                }
            imported = set(self.body["imported_content_sha256s"])
            imported_before = set(imported)
            if (
                source_content_sha256 not in imported
                and len(imported) >= MAX_IMPORTED_TEMPLATE_CONTENTS
            ):
                return {
                    "capacity_rejected_source_states": 1,
                    "imported_multislot_observations": 0,
                    "imported_single_slot_observations": 0,
                    "source_states": 0,
                }
            for row in missing_single:
                existing_single[row.sha256] = row
            for row in missing_multislot:
                existing_multislot[row.sha256] = row
            _validate_multislot_source_groups(tuple(existing_multislot.values()))
            imported.add(source_content_sha256)
            proposed = dict(self.body)
            proposed["imported_content_sha256s"] = sorted(imported)
            proposed["observations"] = [
                row.to_dict() for _sha, row in sorted(existing_single.items())
            ]
            proposed["multislot_observations"] = [
                row.to_dict()
                for _sha, row in sorted(existing_multislot.items())
            ]
            if len(_state_bytes(proposed)) > MAX_PARAMETRIC_STATE_BYTES:
                return {
                    "capacity_rejected_source_states": 1,
                    "imported_multislot_observations": 0,
                    "imported_single_slot_observations": 0,
                    "source_states": 0,
                }
            single_added = len(existing_single) - single_before
            multislot_added = len(existing_multislot) - multislot_before
            if single_added or multislot_added or imported != imported_before:
                self.body = proposed
                self._publish()
            return {
                "capacity_rejected_source_states": 0,
                "imported_multislot_observations": multislot_added,
                "imported_single_slot_observations": single_added,
                "source_states": 1,
            }

    def observe(
        self,
        question: str,
        output: str,
        *,
        question_sha256: str,
        cell_payload_sha256: str,
        teacher_forward_count: int,
    ) -> dict[str, object]:
        if (
            not output.isascii()
            or not output.isprintable()
            or len(output) > self.output_character_limit
        ):
            return {"candidates": 0, "status": "output-outside-template-abi"}
        rows = _candidate_observations(
            question,
            output,
            question_sha256=question_sha256,
            cell_payload_sha256=cell_payload_sha256,
            teacher_forward_count=teacher_forward_count,
        )
        multislot_rows = derive_multislot_observations(
            question,
            output,
            question_sha256=question_sha256,
            cell_payload_sha256=cell_payload_sha256,
            teacher_forward_count=teacher_forward_count,
        )
        if not rows and not multislot_rows:
            return {"candidates": 0, "status": "no-template"}
        if self.observation_verifier is not None and any(
            not bool(self.observation_verifier(row))
            for row in (*rows, *multislot_rows)
        ):
            raise ParametricWarmError(
                "new template observation lacks ResultCell authority"
            )
        with self._lock, _template_lock(self.root):
            self._load()
            existing = {row.sha256: row for row in self.observations}
            existing_multislot = {
                row.sha256: row for row in self.multislot_observations
            }
            before = len(self.promoted)
            for row in rows:
                existing.setdefault(row.sha256, row)
            multislot_before = len(existing_multislot)
            new_multislot_rows = tuple(
                row
                for row in multislot_rows
                if row.sha256 not in existing_multislot
            )
            remaining_multislot = max(
                0,
                MAX_MULTISLOT_OBSERVATIONS - multislot_before,
            )
            if len(new_multislot_rows) <= remaining_multislot:
                for row in new_multislot_rows:
                    existing_multislot[row.sha256] = row
                multislot_admitted = len(new_multislot_rows)
                multislot_dropped = 0
            else:
                multislot_admitted = 0
                multislot_dropped = len(new_multislot_rows)
            self.body["observations"] = [
                row.to_dict() for _sha, row in sorted(existing.items())
            ]
            self.body["multislot_observations"] = [
                row.to_dict()
                for _sha, row in sorted(existing_multislot.items())
            ]
            self._publish()
            after = len(self.promoted)
            return {
                "candidates": len(rows) + len(multislot_rows),
                "multislot_candidates": len(multislot_rows),
                "multislot_admitted": multislot_admitted,
                "multislot_capacity_dropped": multislot_dropped,
                "multislot_observations": len(existing_multislot),
                "observations": len(existing) + len(existing_multislot),
                "single_slot_candidates": len(rows),
                "promoted": after,
                "promoted_delta": after - before,
                "status": "observed",
            }

    def try_warm(
        self,
        question: str,
        metadata: Mapping[str, Any],
    ) -> OoeChatAttempt | None:
        profile = metadata.get("qwen_warm_runtime_profile_sha256")
        token_sha = metadata.get("qwen_token_sha256")
        if (
            profile != self.runtime_profile_sha256
            or token_sha is None
            or self.prompt_token_verifier is None
            or not bool(
                self.prompt_token_verifier(
                    question,
                    require_sha256(token_sha, field="qwen_token_sha256"),
                )
            )
        ):
            return None
        with self._lock, _template_lock(self.root):
            self._load()
            promoted = self.promoted
        matches = []
        for template in promoted:
            output = template.match(question.strip())
            if output is not None:
                matches.append((template, output))
        outputs = {output for _template, output in matches}
        if len(outputs) != 1:
            return None
        output = outputs.pop()
        if (
            not output.isascii()
            or not output.isprintable()
            or len(output) > self.output_character_limit
        ):
            return None
        authorities = tuple(
            sorted(template.sha256 for template, value in matches if value == output)
        )
        slot_arities = tuple(
            sorted(
                {
                    2 if isinstance(template, MultiSlotProgram) else 1
                    for template, value in matches
                    if value == output
                }
            )
        )
        saved = min(
            template.saved_qwen_forwards
            for template, value in matches
            if value == output
        )
        execution_sha256 = _digest(
            {
                "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
                "question_sha256": hashlib.sha256(
                    question.strip().encode("utf-8")
                ).hexdigest(),
                "runtime_profile_sha256": self.runtime_profile_sha256,
                "schema": "immer.qwen3.8-parametric-warm-execution/v2",
                "slot_arities": list(slot_arities),
                "template_sha256s": list(authorities),
            }
        )
        decision_sha256 = _digest(
            {
                "execution_sha256": execution_sha256,
                "slot_arities": list(slot_arities),
                "template_sha256s": list(authorities),
            }
        )
        settled = False

        def settle(accept: bool) -> WarmAccountingReceipt:
            nonlocal settled
            if settled:
                raise ParametricWarmError("template attempt was already settled")
            with self._lock, _template_lock(self.root):
                self._load()
                ordinal = self.body["next_transaction"]
                transaction_sha256 = _digest(
                    {
                        "execution_sha256": execution_sha256,
                        "ordinal": ordinal,
                        "runtime_profile_sha256": self.runtime_profile_sha256,
                    }
                )
                receipt = WarmAccountingReceipt(
                    transaction_sha256=transaction_sha256,
                    decision_binding_sha256=decision_sha256,
                    execution_receipt_sha256=execution_sha256,
                    disposition="committed" if accept else "rejected",
                    saved_qwen_forwards=saved if accept else 0,
                )
                self.body["next_transaction"] = ordinal + 1
                field = "committed_executions" if accept else "rejected_executions"
                self.body[field] += 1
                if accept:
                    self.body["saved_qwen_forwards"] += saved
                self._publish()
                settled = True
                return receipt

        result = Result(
            ExecutionStatus.OK,
            "immer.markov-parametric-template",
            output=output,
            evidence={
                "parametric_template": {
                    "execution_sha256": execution_sha256,
                    "runtime_profile_sha256": self.runtime_profile_sha256,
                    "saved_qwen_forwards": saved,
                    "slot_arities": list(slot_arities),
                    "template_sha256s": list(authorities),
                }
            },
        )
        return OoeChatAttempt(
            result,
            {
                "execution_sha256": execution_sha256,
                "saved_qwen_forwards": saved,
                "slot_arities": list(slot_arities),
                "status": "parametric-hit",
                "template_sha256s": list(authorities),
            },
            _settler=settle,
            _abstention_authorized=True,
        )


__all__ = [
    "MINIMUM_DISTINCT_SLOTS",
    "MAX_PARAMETRIC_STATE_BYTES",
    "LEGACY_TEMPLATE_STATE_NAME",
    "LEGACY_TEMPLATE_STATE_SCHEMA",
    "ParametricWarmBank",
    "ParametricWarmError",
    "PromotedTemplate",
    "TemplateObservation",
]
