"""Learn deterministic parametric warm programs from ordinary cold chats."""

from __future__ import annotations

from collections.abc import Callable, Mapping
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


TEMPLATE_STATE_SCHEMA = "immer.qwen3.8-parametric-warm-state/v1"
TEMPLATE_STATE_NAME = "qwen38-parametric-warm-state-v1"
MINIMUM_DISTINCT_SLOTS = 2
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
        observation_verifier: Callable[[TemplateObservation], bool] | None = None,
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

    @staticmethod
    def _empty(profile: str, output_character_limit: int) -> dict[str, Any]:
        return {
            "committed_executions": 0,
            "next_transaction": 0,
            "observations": [],
            "output_character_limit": output_character_limit,
            "privacy": "private-executable-template-descriptors/v1",
            "rejected_executions": 0,
            "runtime_profile_sha256": profile,
            "saved_qwen_forwards": 0,
            "schema": TEMPLATE_STATE_SCHEMA,
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
            return
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ParametricWarmError("template state is invalid JSON") from exc
        body = value.get("body") if isinstance(value, Mapping) else None
        if (
            canonical_json_bytes(value) != raw
            or not isinstance(value, Mapping)
            or set(value) != {"body", "schema", "sha256"}
            or value.get("schema") != TEMPLATE_STATE_SCHEMA
            or not isinstance(body, Mapping)
            or set(body)
            != set(
                self._empty(
                    self.runtime_profile_sha256,
                    self.output_character_limit,
                )
            )
            or body.get("runtime_profile_sha256") != self.runtime_profile_sha256
            or body.get("output_character_limit")
            != self.output_character_limit
            or body.get("privacy")
            != "private-executable-template-descriptors/v1"
            or value.get("sha256") != _digest(body)
            or not isinstance(body.get("observations"), list)
        ):
            raise ParametricWarmError("template state envelope is invalid")
        observations = [
            TemplateObservation.from_dict(row) for row in body["observations"]
        ]
        if len({row.sha256 for row in observations}) != len(observations):
            raise ParametricWarmError("template observation is duplicated")
        if self.observation_verifier is not None and any(
            not bool(self.observation_verifier(row)) for row in observations
        ):
            raise ParametricWarmError(
                "template observation lost its exact ResultCell authority"
            )
        normalized = dict(body)
        normalized["observations"] = [row.to_dict() for row in observations]
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
        self.state_sha256 = hashlib.sha256(raw).hexdigest()

    @property
    def observations(self) -> tuple[TemplateObservation, ...]:
        return tuple(
            TemplateObservation.from_dict(row) for row in self.body["observations"]
        )

    @property
    def promoted(self) -> tuple[PromotedTemplate, ...]:
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
        return tuple(sorted(promoted, key=lambda row: row.sha256))

    def _publish(self) -> None:
        publication = self.store.publish_state(
            TEMPLATE_STATE_NAME,
            _state_bytes(self.body),
            expected_sha256=self.state_sha256,
        )
        self.state_sha256 = publication.payload_sha256

    def observe(
        self,
        question: str,
        output: str,
        *,
        question_sha256: str,
        cell_payload_sha256: str,
        teacher_forward_count: int,
    ) -> dict[str, object]:
        rows = _candidate_observations(
            question,
            output,
            question_sha256=question_sha256,
            cell_payload_sha256=cell_payload_sha256,
            teacher_forward_count=teacher_forward_count,
        )
        if (
            not output.isascii()
            or not output.isprintable()
            or len(output) > self.output_character_limit
        ):
            return {"candidates": 0, "status": "output-outside-template-abi"}
        if not rows:
            return {"candidates": 0, "status": "no-template"}
        if self.observation_verifier is not None and any(
            not bool(self.observation_verifier(row)) for row in rows
        ):
            raise ParametricWarmError(
                "new template observation lacks ResultCell authority"
            )
        with self._lock, _template_lock(self.root):
            self._load()
            existing = {row.sha256: row for row in self.observations}
            before = len(self.promoted)
            for row in rows:
                existing.setdefault(row.sha256, row)
            self.body["observations"] = [
                row.to_dict() for _sha, row in sorted(existing.items())
            ]
            self._publish()
            after = len(self.promoted)
            return {
                "candidates": len(rows),
                "observations": len(existing),
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
                "schema": "immer.qwen3.8-parametric-warm-execution/v1",
                "template_sha256s": list(authorities),
            }
        )
        decision_sha256 = _digest(
            {
                "execution_sha256": execution_sha256,
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
                    "template_sha256s": list(authorities),
                }
            },
        )
        return OoeChatAttempt(
            result,
            {
                "execution_sha256": execution_sha256,
                "saved_qwen_forwards": saved,
                "status": "parametric-hit",
                "template_sha256s": list(authorities),
            },
            _settler=settle,
            _abstention_authorized=True,
        )


__all__ = [
    "MINIMUM_DISTINCT_SLOTS",
    "ParametricWarmBank",
    "ParametricWarmError",
    "PromotedTemplate",
    "TemplateObservation",
]
