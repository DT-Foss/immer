"""Persistent contextual controller for Qwen rolling draft windows.

The controller never runs a model and never treats draft tokens as authority.
It chooses from ``K={4, 8, 16}`` using only bounded token-ID sketches and
previously settled, target-confirmed request receipts.  A choice is read-only;
the state changes only when :meth:`DraftWindowController.settle` receives a
terminal receipt.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
import threading
from typing import Iterator
import zlib

try:
    import fcntl
except ImportError:  # pragma: no cover - the production runtime is POSIX.
    fcntl = None  # type: ignore[assignment]


DRAFT_WINDOW_ACTIONS = (4, 8, 16)
DRAFT_WINDOW_STATE_SCHEMA = "immer.qwen3.8-draft-window-state/v2"
V1_DRAFT_WINDOW_STATE_SCHEMA = "immer.qwen3.8-draft-window-state/v1"
LEGACY_DRAFT_WINDOW_STATE_SCHEMA = "immer.qwen3.8-draft-window-state/v0"
DRAFT_WINDOW_SELECTION_SCHEMA = "immer.qwen3.8-draft-window-selection/v2"
DRAFT_WINDOW_FEEDBACK_SCHEMA = "immer.qwen3.8-draft-window-feedback/v3"
DRAFT_WINDOW_NESTED_HORIZON_SCHEMA = (
    "immer.qwen3.8-draft-window-nested-horizon/v1"
)
DRAFT_WINDOW_METRICS_SCHEMA = "immer.qwen3.8-draft-window-metrics/v2"

_STATE_PREFIX = b"IMDW\x02"
_V1_STATE_PREFIX = b"IMDW\x01"
_LEGACY_STATE_PREFIX = b"IMDW\x00"
_MAX_STATE_BYTES = 1024 * 1024
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_COUNTER = (1 << 63) - 1
_MAX_SECONDS = 1.0e12
_HEX = frozenset("0123456789abcdef")
_OUTCOMES = frozenset({"ok", "abstained", "error", "timeout", "aborted"})
_BOOTSTRAP_ORDER = {8: 0, 4: 1, 16: 2}


class DraftWindowError(RuntimeError):
    """Persistent draft-window policy state is invalid or cannot be committed."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and not (set(value) - _HEX)


def _uint(value: object, *, field: str, positive: bool = False) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < int(positive)
        or value > _MAX_COUNTER
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be a bounded {qualifier} integer")
    return value


def _finite(
    value: object,
    *,
    field: str,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if (
        not math.isfinite(result)
        or (lower is not None and result < lower)
        or (upper is not None and result > upper)
    ):
        raise ValueError(f"{field} is outside its finite bound")
    return result


def _bounded_add(left: int, right: int) -> int:
    return min(_MAX_COUNTER, left + right)


def _bounded_seconds(left: float, right: float) -> float:
    return min(_MAX_SECONDS, left + right)


def _float_record(value: float) -> str:
    return float(value).hex()


def _record_float(value: object, *, field: str) -> float:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an exact hexadecimal float")
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field} is not a hexadecimal float") from exc
    return _finite(result, field=field)


def _context_tokens(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError("prompt_token_ids must be an integer sequence")
    result = tuple(value)
    if not result or any(
        isinstance(token, bool)
        or not isinstance(token, int)
        or not 0 <= token < 1 << 32
        for token in result
    ):
        raise ValueError("prompt_token_ids contains an invalid token ID")
    return result


def contextual_bottom_k_signature(
    prompt_token_ids: Sequence[int],
    *,
    sketch_size: int = 32,
    context_tokens: int = 512,
) -> tuple[int, ...]:
    """Return a bounded bottom-k sketch of contextual token n-grams.

    Unigrams, bigrams, and trigrams are hashed as fixed-width token IDs.  A
    length bucket is included so otherwise similar short and long contexts do
    not collapse.  Raw prompt text and raw token sequences are never retained.
    """

    tokens = _context_tokens(prompt_token_ids)
    if (
        isinstance(sketch_size, bool)
        or not isinstance(sketch_size, int)
        or not 1 <= sketch_size <= 64
    ):
        raise ValueError("sketch_size must lie in [1, 64]")
    if (
        isinstance(context_tokens, bool)
        or not isinstance(context_tokens, int)
        or not 3 <= context_tokens <= 4096
    ):
        raise ValueError("context_tokens must lie in [3, 4096]")
    tail = tokens[-context_tokens:]
    features: set[int] = set()
    for order in (1, 2, 3):
        for start in range(0, len(tail) - order + 1):
            material = bytearray((order,))
            for token in tail[start : start + order]:
                material.extend(token.to_bytes(4, "little", signed=False))
            digest = hashlib.blake2b(
                material,
                digest_size=8,
                person=b"IMMDWCTX",
            ).digest()
            features.add(int.from_bytes(digest, "big"))
    length_bucket = min(255, len(tokens).bit_length())
    features.add(
        int.from_bytes(
            hashlib.blake2b(
                bytes((255, length_bucket)),
                digest_size=8,
                person=b"IMMDWCTX",
            ).digest(),
            "big",
        )
    )
    return tuple(sorted(features)[:sketch_size])


def _signature_sha256(signature: Sequence[int]) -> str:
    return _sha256([f"{value:016x}" for value in signature])


def _similarity(left: Sequence[int], right: Sequence[int]) -> float:
    first = set(left)
    second = set(right)
    union = first | second
    return 0.0 if not union else len(first & second) / len(union)


@dataclass(frozen=True, slots=True)
class DraftWindowAgentState:
    """Cumulative target-receipt metrics for one window action."""

    window: int
    observations: int = 0
    ok: int = 0
    abstained: int = 0
    errors: int = 0
    timeouts: int = 0
    aborted: int = 0
    zero_acceptance: int = 0
    accepted_draft_tokens: int = 0
    emitted_tokens: int = 0
    target_source_body_bytes: int = 0
    draft_source_body_bytes: int = 0
    aux_source_body_bytes: int = 0
    target_forwards: int = 0
    seconds: float = 0.0
    reward_sum: float = 0.0
    nested_observations: int = 0
    nested_accepted_draft_tokens: int = 0
    nested_proposed_draft_tokens: int = 0
    nested_acceptance_score_sum: float = 0.0

    def __post_init__(self) -> None:
        if self.window not in DRAFT_WINDOW_ACTIONS:
            raise ValueError("draft-window agent action is invalid")
        names = (
            "observations",
            "ok",
            "abstained",
            "errors",
            "timeouts",
            "aborted",
            "zero_acceptance",
            "accepted_draft_tokens",
            "emitted_tokens",
            "target_source_body_bytes",
            "draft_source_body_bytes",
            "aux_source_body_bytes",
            "target_forwards",
            "nested_observations",
            "nested_accepted_draft_tokens",
            "nested_proposed_draft_tokens",
        )
        for name in names:
            _uint(getattr(self, name), field=f"agent {name}")
        if self.observations != (
            self.ok + self.abstained + self.errors + self.timeouts + self.aborted
        ):
            raise ValueError("draft-window agent outcome counters disagree")
        if self.zero_acceptance > self.observations:
            raise ValueError("zero-acceptance count exceeds observations")
        _finite(self.seconds, field="agent seconds", lower=0.0, upper=_MAX_SECONDS)
        _finite(self.reward_sum, field="agent reward_sum")
        _finite(
            self.nested_acceptance_score_sum,
            field="agent nested_acceptance_score_sum",
        )
        if self.nested_accepted_draft_tokens > self.nested_proposed_draft_tokens:
            raise ValueError("nested accepted drafts exceed nested proposals")

    @property
    def mean_reward(self) -> float:
        return 0.0 if self.observations == 0 else self.reward_sum / self.observations

    @property
    def acceptance_rate(self) -> float:
        return (
            0.0
            if self.emitted_tokens == 0
            else self.accepted_draft_tokens / self.emitted_tokens
        )

    @property
    def total_source_body_bytes(self) -> int:
        return (
            self.target_source_body_bytes
            + self.draft_source_body_bytes
            + self.aux_source_body_bytes
        )

    def updated(
        self,
        feedback: "DraftWindowFeedback",
        reward: float,
    ) -> "DraftWindowAgentState":
        outcomes = {
            "ok": self.ok,
            "abstained": self.abstained,
            "error": self.errors,
            "timeout": self.timeouts,
            "aborted": self.aborted,
        }
        outcomes[feedback.outcome] = _bounded_add(outcomes[feedback.outcome], 1)
        return replace(
            self,
            observations=_bounded_add(self.observations, 1),
            ok=outcomes["ok"],
            abstained=outcomes["abstained"],
            errors=outcomes["error"],
            timeouts=outcomes["timeout"],
            aborted=outcomes["aborted"],
            zero_acceptance=_bounded_add(
                self.zero_acceptance,
                int(feedback.accepted_draft_tokens == 0),
            ),
            accepted_draft_tokens=_bounded_add(
                self.accepted_draft_tokens,
                feedback.accepted_draft_tokens,
            ),
            emitted_tokens=_bounded_add(
                self.emitted_tokens,
                feedback.emitted_tokens,
            ),
            target_source_body_bytes=_bounded_add(
                self.target_source_body_bytes,
                feedback.target_source_body_bytes,
            ),
            draft_source_body_bytes=_bounded_add(
                self.draft_source_body_bytes,
                feedback.draft_source_body_bytes,
            ),
            aux_source_body_bytes=_bounded_add(
                self.aux_source_body_bytes,
                feedback.aux_source_body_bytes,
            ),
            target_forwards=_bounded_add(
                self.target_forwards,
                feedback.target_forwards,
            ),
            seconds=_bounded_seconds(self.seconds, feedback.seconds),
            reward_sum=max(-_MAX_SECONDS, min(_MAX_SECONDS, self.reward_sum + reward)),
        )

    def updated_nested(
        self,
        evidence: "DraftWindowNestedHorizon",
    ) -> "DraftWindowAgentState":
        if evidence.candidate_window != self.window:
            raise ValueError("nested evidence belongs to a different window")
        return replace(
            self,
            nested_observations=_bounded_add(self.nested_observations, 1),
            nested_accepted_draft_tokens=_bounded_add(
                self.nested_accepted_draft_tokens,
                evidence.accepted_draft_tokens,
            ),
            nested_proposed_draft_tokens=_bounded_add(
                self.nested_proposed_draft_tokens,
                evidence.proposed_draft_tokens,
            ),
            nested_acceptance_score_sum=max(
                -_MAX_SECONDS,
                min(
                    _MAX_SECONDS,
                    self.nested_acceptance_score_sum
                    + evidence.acceptance_score,
                ),
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "aborted": self.aborted,
            "abstained": self.abstained,
            "accepted_draft_tokens": self.accepted_draft_tokens,
            "aux_source_body_bytes": self.aux_source_body_bytes,
            "draft_source_body_bytes": self.draft_source_body_bytes,
            "emitted_tokens": self.emitted_tokens,
            "errors": self.errors,
            "observations": self.observations,
            "ok": self.ok,
            "nested_accepted_draft_tokens": self.nested_accepted_draft_tokens,
            "nested_observations": self.nested_observations,
            "nested_proposed_draft_tokens": self.nested_proposed_draft_tokens,
            "nested_acceptance_score_sum": _float_record(
                self.nested_acceptance_score_sum
            ),
            "reward_sum": _float_record(self.reward_sum),
            "seconds": _float_record(self.seconds),
            "target_forwards": self.target_forwards,
            "target_source_body_bytes": self.target_source_body_bytes,
            "timeouts": self.timeouts,
            "window": self.window,
            "zero_acceptance": self.zero_acceptance,
        }

    @classmethod
    def from_record(cls, value: object) -> "DraftWindowAgentState":
        if not isinstance(value, Mapping):
            raise ValueError("draft-window agent record must be a mapping")
        expected = {
            "aborted",
            "abstained",
            "accepted_draft_tokens",
            "aux_source_body_bytes",
            "draft_source_body_bytes",
            "emitted_tokens",
            "errors",
            "observations",
            "ok",
            "reward_sum",
            "seconds",
            "target_forwards",
            "target_source_body_bytes",
            "timeouts",
            "window",
            "zero_acceptance",
            "nested_accepted_draft_tokens",
            "nested_observations",
            "nested_proposed_draft_tokens",
            "nested_acceptance_score_sum",
        }
        legacy = expected - {
            "nested_accepted_draft_tokens",
            "nested_observations",
            "nested_proposed_draft_tokens",
            "nested_acceptance_score_sum",
        }
        if frozenset(value) not in {frozenset(expected), frozenset(legacy)}:
            raise ValueError("draft-window agent record fields are invalid")
        return cls(
            window=value["window"],
            observations=value["observations"],
            ok=value["ok"],
            abstained=value["abstained"],
            errors=value["errors"],
            timeouts=value["timeouts"],
            aborted=value["aborted"],
            zero_acceptance=value["zero_acceptance"],
            accepted_draft_tokens=value["accepted_draft_tokens"],
            emitted_tokens=value["emitted_tokens"],
            target_source_body_bytes=value["target_source_body_bytes"],
            draft_source_body_bytes=value["draft_source_body_bytes"],
            aux_source_body_bytes=value["aux_source_body_bytes"],
            target_forwards=value["target_forwards"],
            seconds=_record_float(value["seconds"], field="agent seconds"),
            reward_sum=_record_float(value["reward_sum"], field="agent reward_sum"),
            nested_observations=value.get("nested_observations", 0),
            nested_accepted_draft_tokens=value.get(
                "nested_accepted_draft_tokens", 0
            ),
            nested_proposed_draft_tokens=value.get(
                "nested_proposed_draft_tokens", 0
            ),
            nested_acceptance_score_sum=(
                0.0
                if "nested_acceptance_score_sum" not in value
                else _record_float(
                    value["nested_acceptance_score_sum"],
                    field="agent nested_acceptance_score_sum",
                )
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "aborted": self.aborted,
            "abstained": self.abstained,
            "acceptance_rate": self.acceptance_rate,
            "accepted_draft_tokens": self.accepted_draft_tokens,
            "aux_source_body_bytes": self.aux_source_body_bytes,
            "draft_source_body_bytes": self.draft_source_body_bytes,
            "emitted_tokens": self.emitted_tokens,
            "errors": self.errors,
            "mean_reward": self.mean_reward,
            "nested_accepted_draft_tokens": self.nested_accepted_draft_tokens,
            "nested_observations": self.nested_observations,
            "nested_proposed_draft_tokens": self.nested_proposed_draft_tokens,
            "nested_reliability": (
                0.0
                if self.nested_proposed_draft_tokens == 0
                else self.nested_accepted_draft_tokens
                / self.nested_proposed_draft_tokens
            ),
            "nested_acceptance_score_sum": self.nested_acceptance_score_sum,
            "observations": self.observations,
            "ok": self.ok,
            "reward_sum": self.reward_sum,
            "seconds": self.seconds,
            "target_forwards": self.target_forwards,
            "target_source_body_bytes": self.target_source_body_bytes,
            "timeouts": self.timeouts,
            "total_source_body_bytes": self.total_source_body_bytes,
            "window": self.window,
            "zero_acceptance": self.zero_acceptance,
        }


@dataclass(frozen=True, slots=True)
class DraftWindowDialectState:
    dialect_id: str
    signature: tuple[int, ...]
    visits: int
    last_seen: int
    rapidities: tuple[float, float, float]
    observations: tuple[int, int, int]
    reward_sums: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not _is_sha256(self.dialect_id):
            raise ValueError("draft-window dialect ID must be a SHA-256 digest")
        signature = tuple(self.signature)
        if (
            not 1 <= len(signature) <= 64
            or signature != tuple(sorted(set(signature)))
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < 1 << 64
                for value in signature
            )
        ):
            raise ValueError("draft-window dialect signature is invalid")
        _uint(self.visits, field="dialect visits", positive=True)
        _uint(self.last_seen, field="dialect last_seen")
        rapidities = tuple(float(value) for value in self.rapidities)
        observations = tuple(self.observations)
        rewards = tuple(float(value) for value in self.reward_sums)
        if (
            len(rapidities) != 3
            or len(observations) != 3
            or len(rewards) != 3
            or any(
                not math.isfinite(value) or abs(value) > 20.0 for value in rapidities
            )
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= _MAX_COUNTER
                for value in observations
            )
            or any(not math.isfinite(value) for value in rewards)
        ):
            raise ValueError("draft-window dialect agent state is invalid")
        object.__setattr__(self, "signature", signature)
        object.__setattr__(self, "rapidities", rapidities)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "reward_sums", rewards)

    def to_record(self) -> dict[str, object]:
        return {
            "dialect_id": self.dialect_id,
            "last_seen": self.last_seen,
            "observations": list(self.observations),
            "rapidities": [_float_record(value) for value in self.rapidities],
            "reward_sums": [_float_record(value) for value in self.reward_sums],
            "signature": [f"{value:016x}" for value in self.signature],
            "visits": self.visits,
        }

    @classmethod
    def from_record(cls, value: object) -> "DraftWindowDialectState":
        if not isinstance(value, Mapping) or set(value) != {
            "dialect_id",
            "last_seen",
            "observations",
            "rapidities",
            "reward_sums",
            "signature",
            "visits",
        }:
            raise ValueError("draft-window dialect record is invalid")
        try:
            return cls(
                dialect_id=value["dialect_id"],
                signature=tuple(int(item, 16) for item in value["signature"]),
                visits=value["visits"],
                last_seen=value["last_seen"],
                rapidities=tuple(
                    _record_float(item, field="dialect rapidity")
                    for item in value["rapidities"]
                ),
                observations=tuple(value["observations"]),
                reward_sums=tuple(
                    _record_float(item, field="dialect reward")
                    for item in value["reward_sums"]
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("draft-window dialect values are invalid") from exc


@dataclass(frozen=True, slots=True)
class DraftWindowState:
    policy_identity_sha256: str | None = None
    updates: int = 0
    clock: int = 0
    rapidities: tuple[float, float, float] = (0.0, 0.0, 0.0)
    agents: tuple[DraftWindowAgentState, ...] = ()
    surprise_mean: float = 0.0
    surprise_deviation: float = 1.0
    surprise_cusum: float = 0.0
    regime_generation: int = 0
    dialect_evictions: int = 0
    dialects: tuple[DraftWindowDialectState, ...] = ()
    recent_settlement_sha256s: tuple[str, ...] = ()
    signal_observations: int = 0
    council_confidence_ema: float = 0.0
    council_disagreement_ema: float = 0.0
    phrase_confidence_ema: float = 0.0
    phrase_support_ema: float = 0.0
    phrase_width_ema: float = 0.0

    def __post_init__(self) -> None:
        if self.policy_identity_sha256 is not None and not _is_sha256(
            self.policy_identity_sha256
        ):
            raise ValueError("draft-window policy identity must be a SHA-256 digest")
        _uint(self.updates, field="draft-window updates")
        _uint(self.clock, field="draft-window clock")
        if self.clock < self.updates:
            raise ValueError("draft-window logical clock trails updates")
        rapidities = tuple(float(value) for value in self.rapidities)
        if len(rapidities) != 3 or any(
            not math.isfinite(value) or abs(value) > 20.0 for value in rapidities
        ):
            raise ValueError("draft-window rapidities are invalid")
        agents = tuple(self.agents) or tuple(
            DraftWindowAgentState(window) for window in DRAFT_WINDOW_ACTIONS
        )
        if tuple(
            agent.window for agent in agents
        ) != DRAFT_WINDOW_ACTIONS or self.updates != sum(
            agent.observations for agent in agents
        ):
            raise ValueError("draft-window cumulative agent topology is invalid")
        for value, field in (
            (self.surprise_mean, "surprise_mean"),
            (self.surprise_deviation, "surprise_deviation"),
            (self.surprise_cusum, "surprise_cusum"),
        ):
            _finite(value, field=field, lower=0.0)
        _uint(self.regime_generation, field="regime_generation")
        _uint(self.dialect_evictions, field="dialect_evictions")
        _uint(self.signal_observations, field="signal_observations")
        for value, field, upper in (
            (self.council_confidence_ema, "council_confidence_ema", 1.0),
            (self.council_disagreement_ema, "council_disagreement_ema", 1.0),
            (self.phrase_confidence_ema, "phrase_confidence_ema", 1.0),
            (self.phrase_support_ema, "phrase_support_ema", _MAX_SECONDS),
            (self.phrase_width_ema, "phrase_width_ema", 15.0),
        ):
            _finite(value, field=field, lower=0.0, upper=upper)
        dialects = tuple(self.dialects)
        if (
            len(dialects) > 64
            or tuple(sorted(dialects, key=lambda item: item.dialect_id)) != dialects
            or len({item.dialect_id for item in dialects}) != len(dialects)
            or any(item.last_seen > self.clock for item in dialects)
        ):
            raise ValueError("draft-window dialect inventory is invalid")
        recent = tuple(self.recent_settlement_sha256s)
        if (
            len(recent) > 256
            or len(set(recent)) != len(recent)
            or any(not _is_sha256(value) for value in recent)
        ):
            raise ValueError("draft-window settlement inventory is invalid")
        object.__setattr__(self, "rapidities", rapidities)
        object.__setattr__(self, "agents", agents)
        object.__setattr__(self, "dialects", dialects)
        object.__setattr__(self, "recent_settlement_sha256s", recent)

    def to_record(self) -> dict[str, object]:
        return {
            "agents": [agent.to_record() for agent in self.agents],
            "clock": self.clock,
            "council_confidence_ema": _float_record(
                self.council_confidence_ema
            ),
            "council_disagreement_ema": _float_record(
                self.council_disagreement_ema
            ),
            "dialect_evictions": self.dialect_evictions,
            "dialects": [dialect.to_record() for dialect in self.dialects],
            "policy_identity_sha256": self.policy_identity_sha256,
            "phrase_confidence_ema": _float_record(self.phrase_confidence_ema),
            "phrase_support_ema": _float_record(self.phrase_support_ema),
            "phrase_width_ema": _float_record(self.phrase_width_ema),
            "rapidities": [_float_record(value) for value in self.rapidities],
            "recent_settlement_sha256s": list(self.recent_settlement_sha256s),
            "regime_generation": self.regime_generation,
            "schema": DRAFT_WINDOW_STATE_SCHEMA,
            "signal_observations": self.signal_observations,
            "surprise_cusum": _float_record(self.surprise_cusum),
            "surprise_deviation": _float_record(self.surprise_deviation),
            "surprise_mean": _float_record(self.surprise_mean),
            "updates": self.updates,
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.to_record())

    def to_bytes(self) -> bytes:
        body = self.to_record()
        raw = _canonical({"body": body, "body_sha256": _sha256(body)})
        encoded = _STATE_PREFIX + zlib.compress(raw, level=9)
        if len(encoded) > _MAX_STATE_BYTES:
            raise DraftWindowError("draft-window state exceeds its byte bound")
        return encoded

    @classmethod
    def from_record(cls, value: object) -> "DraftWindowState":
        if not isinstance(value, Mapping) or set(value) != {
            "agents",
            "clock",
            "council_confidence_ema",
            "council_disagreement_ema",
            "dialect_evictions",
            "dialects",
            "policy_identity_sha256",
            "phrase_confidence_ema",
            "phrase_support_ema",
            "phrase_width_ema",
            "rapidities",
            "recent_settlement_sha256s",
            "regime_generation",
            "schema",
            "signal_observations",
            "surprise_cusum",
            "surprise_deviation",
            "surprise_mean",
            "updates",
        }:
            raise ValueError("draft-window state fields are invalid")
        if value["schema"] != DRAFT_WINDOW_STATE_SCHEMA:
            raise ValueError("draft-window state schema is invalid")
        try:
            return cls(
                policy_identity_sha256=value["policy_identity_sha256"],
                signal_observations=value["signal_observations"],
                council_confidence_ema=_record_float(
                    value["council_confidence_ema"],
                    field="council_confidence_ema",
                ),
                council_disagreement_ema=_record_float(
                    value["council_disagreement_ema"],
                    field="council_disagreement_ema",
                ),
                phrase_confidence_ema=_record_float(
                    value["phrase_confidence_ema"],
                    field="phrase_confidence_ema",
                ),
                phrase_support_ema=_record_float(
                    value["phrase_support_ema"],
                    field="phrase_support_ema",
                ),
                phrase_width_ema=_record_float(
                    value["phrase_width_ema"],
                    field="phrase_width_ema",
                ),
                updates=value["updates"],
                clock=value["clock"],
                rapidities=tuple(
                    _record_float(item, field="rapidity")
                    for item in value["rapidities"]
                ),
                agents=tuple(
                    DraftWindowAgentState.from_record(item) for item in value["agents"]
                ),
                surprise_mean=_record_float(
                    value["surprise_mean"], field="surprise_mean"
                ),
                surprise_deviation=_record_float(
                    value["surprise_deviation"], field="surprise_deviation"
                ),
                surprise_cusum=_record_float(
                    value["surprise_cusum"], field="surprise_cusum"
                ),
                regime_generation=value["regime_generation"],
                dialect_evictions=value["dialect_evictions"],
                dialects=tuple(
                    DraftWindowDialectState.from_record(item)
                    for item in value["dialects"]
                ),
                recent_settlement_sha256s=tuple(value["recent_settlement_sha256s"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("draft-window state values are invalid") from exc

    @classmethod
    def _from_v1_record(cls, value: object) -> "DraftWindowState":
        if not isinstance(value, Mapping) or value.get("schema") != (
            V1_DRAFT_WINDOW_STATE_SCHEMA
        ):
            raise ValueError("v1 draft-window state schema is invalid")
        migrated = dict(value)
        migrated["schema"] = DRAFT_WINDOW_STATE_SCHEMA
        migrated["policy_identity_sha256"] = None
        migrated["signal_observations"] = 0
        migrated["council_confidence_ema"] = _float_record(0.0)
        migrated["council_disagreement_ema"] = _float_record(0.0)
        migrated["phrase_confidence_ema"] = _float_record(0.0)
        migrated["phrase_support_ema"] = _float_record(0.0)
        migrated["phrase_width_ema"] = _float_record(0.0)
        return cls.from_record(migrated)

    @classmethod
    def _from_legacy_record(cls, value: object) -> "DraftWindowState":
        if not isinstance(value, Mapping) or value.get("schema") != (
            LEGACY_DRAFT_WINDOW_STATE_SCHEMA
        ):
            raise ValueError("legacy draft-window state schema is invalid")
        allowed = {
            "agents",
            "clock",
            "rapidities",
            "regime_generation",
            "schema",
            "surprise_cusum",
            "surprise_deviation",
            "surprise_mean",
            "updates",
        }
        if set(value) != allowed:
            raise ValueError("legacy draft-window state fields are invalid")
        try:
            return cls(
                updates=value["updates"],
                clock=value["clock"],
                rapidities=tuple(
                    _record_float(item, field="legacy rapidity")
                    for item in value["rapidities"]
                ),
                agents=tuple(
                    DraftWindowAgentState.from_record(item) for item in value["agents"]
                ),
                surprise_mean=_record_float(
                    value["surprise_mean"], field="legacy surprise_mean"
                ),
                surprise_deviation=_record_float(
                    value["surprise_deviation"],
                    field="legacy surprise_deviation",
                ),
                surprise_cusum=_record_float(
                    value["surprise_cusum"], field="legacy surprise_cusum"
                ),
                regime_generation=value["regime_generation"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("legacy draft-window state values are invalid") from exc

    @classmethod
    def from_bytes(cls, value: bytes) -> "DraftWindowState":
        if not isinstance(value, bytes) or not 5 < len(value) <= _MAX_STATE_BYTES:
            raise DraftWindowError("draft-window state size is invalid")
        if value.startswith(_STATE_PREFIX):
            version = 2
            payload = value[len(_STATE_PREFIX) :]
        elif value.startswith(_V1_STATE_PREFIX):
            version = 1
            payload = value[len(_V1_STATE_PREFIX) :]
        elif value.startswith(_LEGACY_STATE_PREFIX):
            version = 0
            payload = value[len(_LEGACY_STATE_PREFIX) :]
        else:
            raise DraftWindowError("draft-window state header is invalid")
        try:
            decompressor = zlib.decompressobj()
            raw = decompressor.decompress(payload, _MAX_JSON_BYTES + 1)
            if len(raw) > _MAX_JSON_BYTES or decompressor.unconsumed_tail:
                raise ValueError("compressed body exceeds its bound")
            raw += decompressor.flush()
            if (
                len(raw) > _MAX_JSON_BYTES
                or not decompressor.eof
                or decompressor.unused_data
                or decompressor.unconsumed_tail
            ):
                raise ValueError("compressed body exceeds its bound")
            document = json.loads(raw)
            if version == 0:
                return cls._from_legacy_record(document)
            if not isinstance(document, Mapping) or set(document) != {
                "body",
                "body_sha256",
            }:
                raise ValueError("state envelope is invalid")
            if (
                not _is_sha256(document["body_sha256"])
                or _sha256(document["body"]) != document["body_sha256"]
            ):
                raise ValueError("state checksum differs")
            if version == 1:
                return cls._from_v1_record(document["body"])
            return cls.from_record(document["body"])
        except DraftWindowError:
            raise
        except (TypeError, ValueError, zlib.error, json.JSONDecodeError) as exc:
            raise DraftWindowError("draft-window state is corrupt") from exc


@dataclass(frozen=True, slots=True)
class DraftWindowSelection:
    proposed_window: int
    eligible_windows: tuple[int, ...]
    configured_ceiling: int
    max_new_tokens: int
    context_signature: tuple[int, ...]
    dialect_id: str | None
    dialect_similarity: float
    policy_weights: tuple[tuple[int, float], ...]
    selection_mode: str
    selection_probability: float
    cold_start: bool
    state_updates: int
    state_sha256: str
    selection_id: str

    def __post_init__(self) -> None:
        eligible = tuple(self.eligible_windows)
        if (
            not eligible
            or eligible
            != tuple(
                window
                for window in DRAFT_WINDOW_ACTIONS
                if window <= self.configured_ceiling and window <= self.max_new_tokens
            )
            or self.proposed_window not in eligible
        ):
            raise ValueError("draft-window selection eligibility is invalid")
        _uint(self.configured_ceiling, field="configured_ceiling", positive=True)
        _uint(self.max_new_tokens, field="max_new_tokens", positive=True)
        signature = tuple(self.context_signature)
        if (
            not signature
            or signature != tuple(sorted(set(signature)))
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < 1 << 64
                for value in signature
            )
        ):
            raise ValueError("draft-window selection signature is invalid")
        if self.dialect_id is not None and not _is_sha256(self.dialect_id):
            raise ValueError("draft-window selected dialect is invalid")
        _finite(
            self.dialect_similarity,
            field="dialect_similarity",
            lower=0.0,
            upper=1.0,
        )
        weights = tuple(self.policy_weights)
        if (
            tuple(window for window, _weight in weights) != eligible
            or any(
                not math.isfinite(weight) or not 0.0 < weight <= 1.0
                for _, weight in weights
            )
            or not math.isclose(
                sum(weight for _, weight in weights), 1.0, abs_tol=1e-12
            )
        ):
            raise ValueError("draft-window policy weights are invalid")
        if not isinstance(self.cold_start, bool):
            raise TypeError("cold_start must be boolean")
        if self.selection_mode not in {"cold", "bootstrap", "fixed-share"}:
            raise ValueError("draft-window selection mode is invalid")
        _finite(
            self.selection_probability,
            field="selection_probability",
            lower=0.0,
            upper=1.0,
        )
        if self.selection_probability <= 0.0:
            raise ValueError("selection_probability must be positive")
        _uint(self.state_updates, field="selection state_updates")
        if not _is_sha256(self.state_sha256) or not _is_sha256(self.selection_id):
            raise ValueError("draft-window selection digest is invalid")
        object.__setattr__(self, "eligible_windows", eligible)
        object.__setattr__(self, "context_signature", signature)
        object.__setattr__(self, "policy_weights", weights)

    @property
    def context_signature_sha256(self) -> str:
        return _signature_sha256(self.context_signature)

    def to_dict(self) -> dict[str, object]:
        return {
            "cold_start": self.cold_start,
            "configured_ceiling": self.configured_ceiling,
            "context_signature_sha256": self.context_signature_sha256,
            "dialect_id": self.dialect_id,
            "dialect_similarity": self.dialect_similarity,
            "eligible_windows": list(self.eligible_windows),
            "max_new_tokens": self.max_new_tokens,
            "policy_weights": {
                str(window): weight for window, weight in self.policy_weights
            },
            "proposed_window": self.proposed_window,
            "schema": DRAFT_WINDOW_SELECTION_SCHEMA,
            "selection_mode": self.selection_mode,
            "selection_probability": self.selection_probability,
            "selection_id": self.selection_id,
            "state_sha256": self.state_sha256,
            "state_updates": self.state_updates,
        }


@dataclass(frozen=True, slots=True)
class DraftWindowNestedHorizon:
    """Exact shorter-prefix reliability visible inside one wider target wave."""

    candidate_window: int
    observed_window: int
    wave_count: int
    accepted_draft_tokens: int
    proposed_draft_tokens: int

    def __post_init__(self) -> None:
        if (
            self.candidate_window not in DRAFT_WINDOW_ACTIONS
            or self.observed_window not in DRAFT_WINDOW_ACTIONS
            or self.candidate_window >= self.observed_window
        ):
            raise ValueError("nested draft-window horizon is invalid")
        _uint(self.wave_count, field="nested wave_count", positive=True)
        _uint(
            self.accepted_draft_tokens,
            field="nested accepted_draft_tokens",
        )
        _uint(
            self.proposed_draft_tokens,
            field="nested proposed_draft_tokens",
            positive=True,
        )
        if self.accepted_draft_tokens > self.proposed_draft_tokens:
            raise ValueError("nested accepted drafts exceed proposed drafts")

    @property
    def reliability(self) -> float:
        return self.accepted_draft_tokens / self.proposed_draft_tokens

    @property
    def acceptance_score(self) -> float:
        return 8.0 * (self.reliability - 0.5)

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted_draft_tokens": self.accepted_draft_tokens,
            "candidate_window": self.candidate_window,
            "observed_window": self.observed_window,
            "proposed_draft_tokens": self.proposed_draft_tokens,
            "reliability": self.reliability,
            "acceptance_score": self.acceptance_score,
            "schema": DRAFT_WINDOW_NESTED_HORIZON_SCHEMA,
            "wave_count": self.wave_count,
        }


@dataclass(frozen=True, slots=True)
class DraftWindowFeedback:
    proposed_window: int
    accepted_draft_tokens: int
    emitted_tokens: int
    target_source_body_bytes: int
    draft_source_body_bytes: int
    aux_source_body_bytes: int
    target_forwards: int
    seconds: float
    outcome: str
    target_receipt_sha256: str
    nested_horizons: tuple[DraftWindowNestedHorizon, ...] = ()
    council_confidence: float | None = None
    council_disagreement: float | None = None
    effective_experts: float | None = None
    phrase_confidence: float | None = None
    phrase_support: int = 0
    phrase_width: int = 0
    page_actions: int = 0
    page_actions_saved: int = 0
    o1_priority: float = 0.0
    runtime_reward: float | None = None

    def __post_init__(self) -> None:
        if self.proposed_window not in DRAFT_WINDOW_ACTIONS:
            raise ValueError("feedback proposed_window is invalid")
        for field in (
            "accepted_draft_tokens",
            "emitted_tokens",
            "target_source_body_bytes",
            "draft_source_body_bytes",
            "aux_source_body_bytes",
        ):
            _uint(getattr(self, field), field=f"feedback {field}")
        _uint(self.target_forwards, field="feedback target_forwards", positive=True)
        if self.accepted_draft_tokens > self.emitted_tokens:
            raise ValueError("accepted draft tokens exceed emitted tokens")
        _finite(self.seconds, field="feedback seconds", lower=0.0, upper=_MAX_SECONDS)
        if self.outcome not in _OUTCOMES:
            raise ValueError("feedback outcome is invalid")
        if not _is_sha256(self.target_receipt_sha256):
            raise ValueError("feedback lacks a target-confirmed receipt digest")
        nested = tuple(self.nested_horizons)
        if (
            any(not isinstance(row, DraftWindowNestedHorizon) for row in nested)
            or any(row.observed_window != self.proposed_window for row in nested)
            or len({row.candidate_window for row in nested}) != len(nested)
            or tuple(sorted(nested, key=lambda row: row.candidate_window)) != nested
        ):
            raise ValueError("feedback nested horizons are invalid")
        object.__setattr__(self, "nested_horizons", nested)
        for value, field in (
            (self.council_confidence, "council_confidence"),
            (self.council_disagreement, "council_disagreement"),
            (self.phrase_confidence, "phrase_confidence"),
        ):
            if value is not None:
                _finite(value, field=field, lower=0.0, upper=1.0)
        if self.effective_experts is not None:
            _finite(
                self.effective_experts,
                field="effective_experts",
                lower=1.0,
                upper=64.0,
            )
        _uint(self.phrase_support, field="phrase_support")
        _uint(self.phrase_width, field="phrase_width")
        _uint(self.page_actions, field="page_actions")
        _uint(self.page_actions_saved, field="page_actions_saved")
        _finite(self.o1_priority, field="o1_priority", lower=0.0, upper=1.0e12)
        if self.runtime_reward is not None:
            _finite(
                self.runtime_reward,
                field="runtime_reward",
                lower=-16.0,
                upper=16.0,
            )
        if self.phrase_width > 15:
            raise ValueError("phrase_width exceeds the Markov option bound")
        if self.phrase_width == 0 and self.phrase_support:
            raise ValueError("phrase support requires a phrase option")

    @property
    def total_source_body_bytes(self) -> int:
        return (
            self.target_source_body_bytes
            + self.draft_source_body_bytes
            + self.aux_source_body_bytes
        )

    @property
    def total_work(self) -> float:
        return (
            1.0
            + self.target_forwards
            + self.seconds
            + self.total_source_body_bytes / float(1024**3)
        )

    @property
    def reward(self) -> float:
        """Useful confirmed tokens per total work, with terminal penalties."""

        if self.runtime_reward is not None:
            return self.runtime_reward
        if self.outcome == "ok" and self.accepted_draft_tokens > 0:
            efficiency = self.accepted_draft_tokens / self.total_work
            coverage = self.accepted_draft_tokens / max(1, self.emitted_tokens)
            return min(16.0, 8.0 * efficiency + 2.0 * coverage)
        work_penalty = min(2.0, math.log1p(self.total_work) / 4.0)
        if self.outcome == "ok":
            return -6.0 - work_penalty
        penalty = {
            "abstained": 7.0,
            "error": 9.0,
            "aborted": 10.0,
            "timeout": 12.0,
        }[self.outcome]
        return -penalty - work_penalty

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted_draft_tokens": self.accepted_draft_tokens,
            "aux_source_body_bytes": self.aux_source_body_bytes,
            "council_confidence": self.council_confidence,
            "council_disagreement": self.council_disagreement,
            "draft_source_body_bytes": self.draft_source_body_bytes,
            "emitted_tokens": self.emitted_tokens,
            "effective_experts": self.effective_experts,
            "outcome": self.outcome,
            "phrase_confidence": self.phrase_confidence,
            "phrase_support": self.phrase_support,
            "phrase_width": self.phrase_width,
            "page_actions": self.page_actions,
            "page_actions_saved": self.page_actions_saved,
            "o1_priority": self.o1_priority,
            "runtime_reward": self.runtime_reward,
            "nested_horizons": [row.to_dict() for row in self.nested_horizons],
            "proposed_window": self.proposed_window,
            "reward": self.reward,
            "schema": DRAFT_WINDOW_FEEDBACK_SCHEMA,
            "seconds": self.seconds,
            "target_forwards": self.target_forwards,
            "target_receipt_sha256": self.target_receipt_sha256,
            "target_source_body_bytes": self.target_source_body_bytes,
            "total_source_body_bytes": self.total_source_body_bytes,
            "total_work": self.total_work,
        }


@dataclass(frozen=True, slots=True)
class DraftWindowMetrics:
    state_sha256: str
    policy_identity_sha256: str | None
    updates: int
    clock: int
    policy_weights: tuple[tuple[int, float], ...]
    agents: tuple[DraftWindowAgentState, ...]
    surprise_mean: float
    surprise_deviation: float
    surprise_cusum: float
    regime_generation: int
    dialect_count: int
    dialect_evictions: int
    signal_observations: int
    council_confidence_ema: float
    council_disagreement_ema: float
    phrase_confidence_ema: float
    phrase_support_ema: float
    phrase_width_ema: float

    def to_dict(self) -> dict[str, object]:
        return {
            "agents": {str(agent.window): agent.to_dict() for agent in self.agents},
            "clock": self.clock,
            "council_confidence_ema": self.council_confidence_ema,
            "council_disagreement_ema": self.council_disagreement_ema,
            "dialect_count": self.dialect_count,
            "dialect_evictions": self.dialect_evictions,
            "policy_weights": {
                str(window): weight for window, weight in self.policy_weights
            },
            "policy_identity_sha256": self.policy_identity_sha256,
            "phrase_confidence_ema": self.phrase_confidence_ema,
            "phrase_support_ema": self.phrase_support_ema,
            "phrase_width_ema": self.phrase_width_ema,
            "regime_generation": self.regime_generation,
            "schema": DRAFT_WINDOW_METRICS_SCHEMA,
            "signal_observations": self.signal_observations,
            "state_sha256": self.state_sha256,
            "surprise_cusum": self.surprise_cusum,
            "surprise_deviation": self.surprise_deviation,
            "surprise_mean": self.surprise_mean,
            "updates": self.updates,
        }


def _read_state(path: Path) -> DraftWindowState:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | int(getattr(os, "O_CLOEXEC", 0))
            | int(getattr(os, "O_NOFOLLOW", 0)),
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or not 5 < before.st_size <= _MAX_STATE_BYTES
        ):
            raise DraftWindowError("draft-window state is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise DraftWindowError("draft-window state returned a short read")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        linked = path.lstat()

        def identity(item: os.stat_result) -> tuple[int, int, int, int, int]:
            return (
                item.st_dev,
                item.st_ino,
                item.st_size,
                item.st_mtime_ns,
                item.st_ctime_ns,
            )

        if (
            identity(before) != identity(after)
            or (after.st_dev, after.st_ino) != (linked.st_dev, linked.st_ino)
            or stat.S_ISLNK(linked.st_mode)
            or not stat.S_ISREG(linked.st_mode)
        ):
            raise DraftWindowError("draft-window state changed while read")
        return DraftWindowState.from_bytes(b"".join(chunks))
    except DraftWindowError:
        raise
    except OSError as exc:
        raise DraftWindowError("cannot read draft-window state") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _persist_state(path: Path, state: DraftWindowState) -> None:
    data = state.to_bytes()
    try:
        linked = path.lstat()
    except FileNotFoundError:
        linked = None
    except OSError as exc:
        raise DraftWindowError("cannot inspect draft-window state") from exc
    if linked is not None and (
        stat.S_ISLNK(linked.st_mode) or not stat.S_ISREG(linked.st_mode)
    ):
        raise DraftWindowError("draft-window state target is not a regular file")
    temporary = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | int(getattr(os, "O_CLOEXEC", 0))
            | int(getattr(os, "O_NOFOLLOW", 0)),
            0o600,
        )
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise OSError("short draft-window state write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        directory = os.open(
            path.parent,
            os.O_RDONLY
            | int(getattr(os, "O_DIRECTORY", 0))
            | int(getattr(os, "O_CLOEXEC", 0))
            | int(getattr(os, "O_NOFOLLOW", 0)),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as exc:
        raise DraftWindowError("cannot atomically persist draft-window state") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


class DraftWindowController:
    """Receipt-settled contextual bandit over rolling target windows."""

    FIXED_SHARE = 0.05
    TEMPERATURE = 1.0
    RAPIDITY_DECAY = 0.995
    LEARNING_RATE = 0.18
    SIGNAL_RATE = 0.10
    SIGNAL_STRENGTH = 0.50
    NESTED_ACCEPTANCE_STRENGTH = 0.50
    DIALECT_STRENGTH = 1.0
    DIALECT_SIMILARITY_THRESHOLD = 0.20
    SURPRISE_RATE = 0.05
    CUSUM_DECAY = 0.90
    CUSUM_DRIFT = 0.50
    CUSUM_THRESHOLD = 8.0
    REGIME_WARMUP = 8
    REGIME_RAPIDITY_SHRINK = 0.25
    RETENTION_ALPHA = 0.001
    MAX_DIALECTS = 64
    RECENT_SETTLEMENTS = 256

    def __init__(
        self,
        state_path: str | Path,
        *,
        max_dialects: int = MAX_DIALECTS,
        retention_alpha: float = RETENTION_ALPHA,
    ) -> None:
        if not isinstance(state_path, (str, Path)):
            raise TypeError("state_path must be a local filesystem path")
        if (
            isinstance(max_dialects, bool)
            or not isinstance(max_dialects, int)
            or not 1 <= max_dialects <= self.MAX_DIALECTS
        ):
            raise ValueError("max_dialects must lie in [1, 64]")
        retention_alpha = _finite(
            retention_alpha,
            field="retention_alpha",
            lower=0.0,
            upper=1.0,
        )
        self.state_path = Path(state_path).expanduser().absolute()
        self.max_dialects = max_dialects
        self.retention_alpha = retention_alpha
        self._policy_identity_sha256: str | None = None
        self._thread_lock = threading.RLock()
        self._selection_metrics: dict[str, DraftWindowMetrics] = {}
        self._prepare_parent()
        with self._locked_state():
            self._state = self._load_locked()

    def bind_policy_identity(self, policy_identity_sha256: str) -> DraftWindowMetrics:
        """Bind this state file to one target/tokenizer/drafter runtime."""

        if not _is_sha256(policy_identity_sha256):
            raise ValueError("policy_identity_sha256 must be a SHA-256 digest")
        with self._thread_lock, self._locked_state():
            state = self._load_locked()
            if (
                state.policy_identity_sha256 is not None
                and state.policy_identity_sha256 != policy_identity_sha256
            ):
                raise DraftWindowError(
                    "draft-window state belongs to a different runtime identity"
                )
            if state.policy_identity_sha256 is None:
                state = replace(
                    state,
                    policy_identity_sha256=policy_identity_sha256,
                )
                _persist_state(self.state_path, state)
            self._policy_identity_sha256 = policy_identity_sha256
            self._state = state
            return self._metrics_for(state)

    bind = bind_policy_identity

    def _prepare_parent(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            parent = self.state_path.parent.lstat()
        except OSError as exc:
            raise DraftWindowError(
                "cannot prepare draft-window state directory"
            ) from exc
        if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode):
            raise DraftWindowError("draft-window state parent must be a real directory")

    @contextmanager
    def _locked_state(self) -> Iterator[None]:
        if fcntl is None:  # pragma: no cover - the production runtime is POSIX.
            raise DraftWindowError("persistent draft-window state requires fcntl")
        lock_path = self.state_path.parent / f".{self.state_path.name}.lock"
        descriptor: int | None = None
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT
                | os.O_RDWR
                | int(getattr(os, "O_CLOEXEC", 0))
                | int(getattr(os, "O_NOFOLLOW", 0)),
                0o600,
            )
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise DraftWindowError("draft-window lock is not a regular file")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        except DraftWindowError:
            raise
        except OSError as exc:
            raise DraftWindowError("cannot lock draft-window state") from exc
        finally:
            if descriptor is not None:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)

    def _load_locked(self) -> DraftWindowState:
        try:
            linked = self.state_path.lstat()
        except FileNotFoundError:
            return DraftWindowState()
        except OSError as exc:
            raise DraftWindowError("cannot inspect draft-window state") from exc
        if stat.S_ISLNK(linked.st_mode):
            raise DraftWindowError("draft-window state must not be a symlink")
        state = _read_state(self.state_path)
        if (
            self._policy_identity_sha256 is not None
            and state.policy_identity_sha256 != self._policy_identity_sha256
        ):
            raise DraftWindowError(
                "draft-window state runtime identity changed"
            )
        return state

    @classmethod
    def _policy(
        cls,
        rapidities: Sequence[float],
        eligible_windows: Sequence[int],
    ) -> tuple[tuple[int, float], ...]:
        indices = [DRAFT_WINDOW_ACTIONS.index(window) for window in eligible_windows]
        scaled = [rapidities[index] / cls.TEMPERATURE for index in indices]
        maximum = max(scaled)
        raw = [math.exp(value - maximum) for value in scaled]
        total = sum(raw)
        count = len(raw)
        return tuple(
            (
                window,
                (1.0 - cls.FIXED_SHARE) * weight / total + cls.FIXED_SHARE / count,
            )
            for window, weight in zip(eligible_windows, raw, strict=True)
        )

    @classmethod
    def _combined_rapidities(
        cls,
        state: DraftWindowState,
        dialect: DraftWindowDialectState | None,
        similarity: float,
    ) -> tuple[float, float, float]:
        if dialect is None:
            return state.rapidities
        return tuple(
            global_value + cls.DIALECT_STRENGTH * similarity * local_value
            for global_value, local_value in zip(
                state.rapidities,
                dialect.rapidities,
                strict=True,
            )
        )

    @classmethod
    def _signal_rapidities(
        cls,
        state: DraftWindowState,
    ) -> tuple[float, float, float]:
        if state.signal_observations == 0:
            return (0.0, 0.0, 0.0)
        probability = max(
            0.0,
            min(
                0.999,
                state.council_confidence_ema
                * (1.0 - 0.5 * state.council_disagreement_ema),
            ),
        )
        utilities = []
        for window in DRAFT_WINDOW_ACTIONS:
            council_expected = sum(
                probability**position for position in range(1, window)
            )
            phrase_support = 1.0 - math.exp(-state.phrase_support_ema / 3.0)
            phrase_expected = (
                state.phrase_confidence_ema
                * phrase_support
                * min(state.phrase_width_ema, window - 1)
            )
            expected_tokens = 1.0 + max(council_expected, phrase_expected)
            work_proxy = 1.0 + (window - 1) / 16.0
            utilities.append(expected_tokens / work_proxy)
        center = sum(utilities) / len(utilities)
        return tuple(cls.SIGNAL_STRENGTH * (value - center) for value in utilities)

    @classmethod
    def _nested_acceptance_rapidities(
        cls,
        state: DraftWindowState,
    ) -> tuple[float, float, float]:
        values = []
        for agent in state.agents:
            proposed = agent.nested_proposed_draft_tokens
            if proposed == 0:
                values.append(0.0)
                continue
            reliability = (
                agent.nested_accepted_draft_tokens + 1.0
            ) / (proposed + 2.0)
            evidence_strength = 1.0 - math.exp(-proposed / 8.0)
            values.append(
                cls.NESTED_ACCEPTANCE_STRENGTH
                * evidence_strength
                * 4.0
                * (reliability - 0.5)
            )
        center = sum(values) / len(values)
        return tuple(value - center for value in values)

    def _match_dialect(
        self,
        state: DraftWindowState,
        signature: tuple[int, ...],
    ) -> tuple[DraftWindowDialectState | None, float]:
        candidates = [
            (_similarity(signature, row.signature), row.visits, row.dialect_id, row)
            for row in state.dialects
        ]
        selected = max(candidates, default=None)
        if selected is None or selected[0] < self.DIALECT_SIMILARITY_THRESHOLD:
            return None, 0.0
        return selected[3], selected[0]

    def choose(
        self,
        prompt_token_ids: Sequence[int],
        *,
        max_window: int,
        max_new_tokens: int,
    ) -> DraftWindowSelection | None:
        """Choose a window without changing persistent policy state."""

        _uint(max_window, field="max_window", positive=True)
        _uint(max_new_tokens, field="max_new_tokens", positive=True)
        eligible = tuple(
            window
            for window in DRAFT_WINDOW_ACTIONS
            if window <= max_window and window <= max_new_tokens
        )
        if not eligible:
            return None
        signature = contextual_bottom_k_signature(prompt_token_ids)
        with self._thread_lock, self._locked_state():
            state = self._load_locked()
            self._state = state
        dialect, similarity = self._match_dialect(state, signature)
        rapidities = self._combined_rapidities(state, dialect, similarity)
        signal_rapidities = self._signal_rapidities(state)
        nested_rapidities = self._nested_acceptance_rapidities(state)
        rapidities = tuple(
            value + signal + nested
            for value, signal, nested in zip(
                rapidities,
                signal_rapidities,
                nested_rapidities,
                strict=True,
            )
        )
        policy = self._policy(rapidities, eligible)
        cold = state.updates == 0
        if cold:
            proposed = 8 if 8 in eligible else 4
            selection_mode = "cold"
            selection_probability = 1.0
        else:
            observations = {
                agent.window: agent.observations + agent.nested_observations
                for agent in state.agents
            }
            unseen = tuple(
                window for window in eligible if observations.get(window, 0) == 0
            )
            if unseen:
                proposed = min(unseen, key=_BOOTSTRAP_ORDER.__getitem__)
                selection_mode = "bootstrap"
                selection_probability = 1.0
            else:
                draw = secrets.randbelow(1 << 53) / float(1 << 53)
                cumulative = 0.0
                proposed = policy[-1][0]
                for window, probability in policy:
                    cumulative += probability
                    if draw < cumulative:
                        proposed = window
                        break
                selection_mode = "fixed-share"
                selection_probability = dict(policy)[proposed]
        selection_material = {
            "context_signature_sha256": _signature_sha256(signature),
            "nonce": secrets.token_hex(16),
            "proposed_window": proposed,
            "selection_mode": selection_mode,
            "state_sha256": state.sha256,
        }
        selection = DraftWindowSelection(
            proposed_window=proposed,
            eligible_windows=eligible,
            configured_ceiling=max_window,
            max_new_tokens=max_new_tokens,
            context_signature=signature,
            dialect_id=None if dialect is None else dialect.dialect_id,
            dialect_similarity=similarity,
            policy_weights=policy,
            selection_mode=selection_mode,
            selection_probability=selection_probability,
            cold_start=cold,
            state_updates=state.updates,
            state_sha256=state.sha256,
            selection_id=_sha256(selection_material),
        )
        with self._thread_lock:
            self._selection_metrics[selection.selection_id] = self._metrics_for(state)
            while len(self._selection_metrics) > 64:
                del self._selection_metrics[next(iter(self._selection_metrics))]
        return selection

    select = choose

    @classmethod
    def _updated_rapidities(
        cls,
        rapidities: Sequence[float],
        *,
        selected_index: int,
        selected_probability: float,
        reward: float,
        learning_scale: float = 1.0,
    ) -> tuple[float, float, float]:
        values = [cls.RAPIDITY_DECAY * float(value) for value in rapidities]
        coupling = math.tanh(reward / 8.0)
        rapidity = math.atanh(max(-1.0 + 1e-12, min(1.0 - 1e-12, coupling)))
        floor = cls.FIXED_SHARE / len(DRAFT_WINDOW_ACTIONS)
        values[selected_index] += (
            learning_scale
            * cls.LEARNING_RATE
            * rapidity
            / max(floor, selected_probability)
        )
        center = sum(values) / len(values)
        return tuple(max(-20.0, min(20.0, value - center)) for value in values)

    def _updated_dialect(
        self,
        state: DraftWindowState,
        selection: DraftWindowSelection,
        *,
        selected_index: int,
        selected_probability: float,
        reward: float,
        next_clock: int,
    ) -> DraftWindowDialectState:
        dialect = next(
            (row for row in state.dialects if row.dialect_id == selection.dialect_id),
            None,
        )
        derived_id = _signature_sha256(selection.context_signature)
        if dialect is None:
            dialect = next(
                (row for row in state.dialects if row.dialect_id == derived_id),
                None,
            )
        if dialect is None:
            dialect = DraftWindowDialectState(
                dialect_id=derived_id,
                signature=selection.context_signature,
                visits=1,
                last_seen=next_clock,
                rapidities=(0.0, 0.0, 0.0),
                observations=(0, 0, 0),
                reward_sums=(0.0, 0.0, 0.0),
            )
            visit_increment = 0
        else:
            visit_increment = 1
        observations = list(dialect.observations)
        rewards = list(dialect.reward_sums)
        observations[selected_index] = _bounded_add(observations[selected_index], 1)
        rewards[selected_index] = max(
            -_MAX_SECONDS,
            min(_MAX_SECONDS, rewards[selected_index] + reward),
        )
        signature = tuple(
            sorted(set(dialect.signature) | set(selection.context_signature))[:32]
        )
        return replace(
            dialect,
            signature=signature,
            visits=_bounded_add(dialect.visits, visit_increment),
            last_seen=next_clock,
            rapidities=self._updated_rapidities(
                dialect.rapidities,
                selected_index=selected_index,
                selected_probability=selected_probability,
                reward=reward,
                learning_scale=1.25,
            ),
            observations=tuple(observations),
            reward_sums=tuple(rewards),
        )

    def _metrics_for(self, state: DraftWindowState) -> DraftWindowMetrics:
        signal = self._signal_rapidities(state)
        nested = self._nested_acceptance_rapidities(state)
        policy_rapidities = tuple(
            value + adjustment + nested_adjustment
            for value, adjustment, nested_adjustment in zip(
                state.rapidities,
                signal,
                nested,
                strict=True,
            )
        )
        return DraftWindowMetrics(
            state_sha256=state.sha256,
            policy_identity_sha256=state.policy_identity_sha256,
            updates=state.updates,
            clock=state.clock,
            policy_weights=self._policy(
                policy_rapidities,
                DRAFT_WINDOW_ACTIONS,
            ),
            agents=state.agents,
            surprise_mean=state.surprise_mean,
            surprise_deviation=state.surprise_deviation,
            surprise_cusum=state.surprise_cusum,
            regime_generation=state.regime_generation,
            dialect_count=len(state.dialects),
            dialect_evictions=state.dialect_evictions,
            signal_observations=state.signal_observations,
            council_confidence_ema=state.council_confidence_ema,
            council_disagreement_ema=state.council_disagreement_ema,
            phrase_confidence_ema=state.phrase_confidence_ema,
            phrase_support_ema=state.phrase_support_ema,
            phrase_width_ema=state.phrase_width_ema,
        )

    def metrics(self) -> DraftWindowMetrics:
        with self._thread_lock, self._locked_state():
            state = self._load_locked()
            self._state = state
        return self._metrics_for(state)

    def metrics_for_selection(
        self,
        selection: DraftWindowSelection,
    ) -> DraftWindowMetrics:
        """Return the exact cumulative snapshot which produced ``selection``."""

        if not isinstance(selection, DraftWindowSelection):
            raise TypeError("selection must be a DraftWindowSelection")
        with self._thread_lock:
            metrics = self._selection_metrics.get(selection.selection_id)
        if metrics is None or metrics.state_sha256 != selection.state_sha256:
            raise DraftWindowError("selection metrics are no longer available")
        return metrics

    def settle(
        self,
        selection: DraftWindowSelection,
        feedback: DraftWindowFeedback,
    ) -> DraftWindowMetrics:
        """Atomically learn one terminal target-confirmed request receipt."""

        if not isinstance(selection, DraftWindowSelection):
            raise TypeError("selection must be a DraftWindowSelection")
        if not isinstance(feedback, DraftWindowFeedback):
            raise TypeError("feedback must be DraftWindowFeedback")
        if feedback.proposed_window != selection.proposed_window:
            raise DraftWindowError("feedback window differs from its proposal")
        settlement_sha256 = _sha256(
            {
                "selection_id": selection.selection_id,
                "target_receipt_sha256": feedback.target_receipt_sha256,
            }
        )
        with self._thread_lock, self._locked_state():
            state = self._load_locked()
            if settlement_sha256 in state.recent_settlement_sha256s:
                self._state = state
                return self._metrics_for(state)
            selected_index = DRAFT_WINDOW_ACTIONS.index(selection.proposed_window)
            probability = selection.selection_probability
            reward = feedback.reward
            previous_agent = state.agents[selected_index]
            expected_reward = previous_agent.mean_reward
            surprise = (
                -math.log(max(probability, 1e-12)) + abs(reward - expected_reward) / 4.0
            )
            if state.updates == 0:
                z_score = 0.0
            else:
                z_score = (surprise - state.surprise_mean) / max(
                    state.surprise_deviation,
                    1e-6,
                )
            next_mean = (
                1.0 - self.SURPRISE_RATE
            ) * state.surprise_mean + self.SURPRISE_RATE * surprise
            next_deviation = (
                1.0 - self.SURPRISE_RATE
            ) * state.surprise_deviation + self.SURPRISE_RATE * abs(
                surprise - state.surprise_mean
            )
            next_cusum = max(
                0.0,
                self.CUSUM_DECAY * state.surprise_cusum + z_score - self.CUSUM_DRIFT,
            )
            regime_change = (
                state.updates + 1 >= self.REGIME_WARMUP
                and next_cusum > self.CUSUM_THRESHOLD
            )
            rapidities = self._updated_rapidities(
                state.rapidities,
                selected_index=selected_index,
                selected_probability=probability,
                reward=reward,
            )
            next_clock = _bounded_add(state.clock, 1)
            active = self._updated_dialect(
                state,
                selection,
                selected_index=selected_index,
                selected_probability=probability,
                reward=reward,
                next_clock=next_clock,
            )
            dialects = {row.dialect_id: row for row in state.dialects}
            dialects[active.dialect_id] = active
            if regime_change:
                rapidities = tuple(
                    self.REGIME_RAPIDITY_SHRINK * value for value in rapidities
                )
                dialects = {
                    key: replace(
                        row,
                        rapidities=tuple(
                            self.REGIME_RAPIDITY_SHRINK * value
                            for value in row.rapidities
                        ),
                    )
                    for key, row in dialects.items()
                }
                next_cusum = 0.0
            evictions = state.dialect_evictions
            while len(dialects) > self.max_dialects:
                evicted = min(
                    dialects.values(),
                    key=lambda row: (
                        row.visits
                        * math.exp(
                            -self.retention_alpha * (next_clock - row.last_seen)
                        ),
                        row.visits,
                        row.dialect_id,
                    ),
                )
                del dialects[evicted.dialect_id]
                evictions = _bounded_add(evictions, 1)
            agents = list(state.agents)
            agents[selected_index] = previous_agent.updated(feedback, reward)
            for nested in feedback.nested_horizons:
                nested_index = DRAFT_WINDOW_ACTIONS.index(
                    nested.candidate_window
                )
                agents[nested_index] = agents[nested_index].updated_nested(nested)
            recent = (*state.recent_settlement_sha256s, settlement_sha256)
            if len(recent) > self.RECENT_SETTLEMENTS:
                recent = recent[-self.RECENT_SETTLEMENTS :]
            signal_observations = state.signal_observations
            council_confidence_ema = state.council_confidence_ema
            council_disagreement_ema = state.council_disagreement_ema
            phrase_confidence_ema = state.phrase_confidence_ema
            phrase_support_ema = state.phrase_support_ema
            phrase_width_ema = state.phrase_width_ema
            if feedback.council_confidence is not None:
                rate = 1.0 if signal_observations == 0 else self.SIGNAL_RATE

                def update_signal(previous: float, current: float) -> float:
                    return (1.0 - rate) * previous + rate * current

                council_confidence_ema = update_signal(
                    council_confidence_ema,
                    feedback.council_confidence,
                )
                council_disagreement_ema = update_signal(
                    council_disagreement_ema,
                    0.0
                    if feedback.council_disagreement is None
                    else feedback.council_disagreement,
                )
                phrase_confidence_ema = update_signal(
                    phrase_confidence_ema,
                    0.0
                    if feedback.phrase_confidence is None
                    else feedback.phrase_confidence,
                )
                phrase_support_ema = update_signal(
                    phrase_support_ema,
                    float(feedback.phrase_support),
                )
                phrase_width_ema = update_signal(
                    phrase_width_ema,
                    float(feedback.phrase_width),
                )
                signal_observations = _bounded_add(signal_observations, 1)
            updated = DraftWindowState(
                policy_identity_sha256=state.policy_identity_sha256,
                updates=_bounded_add(state.updates, 1),
                clock=next_clock,
                rapidities=rapidities,
                agents=tuple(agents),
                surprise_mean=next_mean,
                surprise_deviation=next_deviation,
                surprise_cusum=next_cusum,
                regime_generation=_bounded_add(
                    state.regime_generation,
                    int(regime_change),
                ),
                dialect_evictions=evictions,
                dialects=tuple(
                    sorted(dialects.values(), key=lambda row: row.dialect_id)
                ),
                recent_settlement_sha256s=tuple(recent),
                signal_observations=signal_observations,
                council_confidence_ema=council_confidence_ema,
                council_disagreement_ema=council_disagreement_ema,
                phrase_confidence_ema=phrase_confidence_ema,
                phrase_support_ema=phrase_support_ema,
                phrase_width_ema=phrase_width_ema,
            )
            _persist_state(self.state_path, updated)
            self._state = updated
            return self._metrics_for(updated)


PersistentDraftWindowController = DraftWindowController


__all__ = [
    "DRAFT_WINDOW_ACTIONS",
    "DRAFT_WINDOW_FEEDBACK_SCHEMA",
    "DRAFT_WINDOW_METRICS_SCHEMA",
    "DRAFT_WINDOW_NESTED_HORIZON_SCHEMA",
    "DRAFT_WINDOW_SELECTION_SCHEMA",
    "DRAFT_WINDOW_STATE_SCHEMA",
    "LEGACY_DRAFT_WINDOW_STATE_SCHEMA",
    "V1_DRAFT_WINDOW_STATE_SCHEMA",
    "DraftWindowAgentState",
    "DraftWindowController",
    "DraftWindowDialectState",
    "DraftWindowError",
    "DraftWindowFeedback",
    "DraftWindowMetrics",
    "DraftWindowNestedHorizon",
    "DraftWindowSelection",
    "DraftWindowState",
    "PersistentDraftWindowController",
    "contextual_bottom_k_signature",
]
