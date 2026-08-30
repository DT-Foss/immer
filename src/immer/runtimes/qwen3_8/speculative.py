"""Exact greedy K=2 speculative generation for streamed Qwen3.8.

The draft provider is deliberately untrusted: it may only propose two token
IDs.  The official target model verifies both positions with one complete
vocabulary scan and publishes only target-confirmed continuation state.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import math
import numbers
import time
from typing import Any, Literal, Protocol

import torch

from .draft_protocol import RollingDraftProposal, RoundWindowPolicy
from .model import StreamedQwen38
from .pager import Qwen38WeightPager


QWEN38_K2_SPECULATIVE_SCHEMA = "immer.qwen3.8-k2-speculative-generation/v1"
QWEN38_K4_SPECULATIVE_SCHEMA = "immer.qwen3.8-k4-speculative-generation/v1"
QWEN38_K4_SPECULATIVE_ROUND_SCHEMA = "immer.qwen3.8-k4-speculative-round/v1"
QWEN38_ROLLING_K4_SPECULATIVE_SCHEMA = (
    "immer.qwen3.8-rolling-k4-speculative-generation/v2"
)
QWEN38_ROLLING_SPECULATIVE_SCHEMA = "immer.qwen3.8-rolling-speculative-generation/v3"

ReplayKind = Literal[
    "commit-k2",
    "eos0-decode",
    "mismatch0-decode",
    "mismatch1-restage",
    "remaining-single",
]

K4ReplayKind = Literal[
    "commit-k4",
    "eos0-decode",
    "eos1-restage",
    "eos2-restage",
    "eos3-restage",
    "eos3-commit-k4",
    "mismatch0-decode",
    "mismatch1-restage",
    "mismatch2-restage",
    "mismatch3-restage",
    "eos-uncommitted",
    "mismatch3-uncommitted",
    "terminal-single",
    "terminal-single-uncommitted",
]


class Qwen38SpeculativeError(RuntimeError):
    """The exact K=2 speculative-generation contract cannot be satisfied."""


class DraftProvider(Protocol):
    """Propose exactly two tokens from immutable committed token history.

    The decoder hashes all committed continuation tensors before and after the
    same-process callback.  Mutation resets the model and fails closed; guard
    time and hashed bytes are included in the generation receipts.
    """

    def __call__(self, history: tuple[int, ...], /) -> Sequence[int]: ...


class ReconciledDraftProvider(DraftProvider, Protocol):
    """Stateful provider notified after the target commits each proposed round."""

    def reconcile(self, history: tuple[int, ...], /) -> None: ...


class K4DraftProvider(Protocol):
    """Propose exactly four tokens from immutable committed token history."""

    def __call__(self, history: tuple[int, ...], /) -> Sequence[int]: ...


class K4ReconciledDraftProvider(K4DraftProvider, Protocol):
    """Stateful K=4 provider notified only after target-confirmed commits."""

    def reconcile(self, history: tuple[int, ...], /) -> None: ...


class RollingK4DraftProvider(Protocol):
    """Propose the configured token tail after one target-known token."""

    def propose_after(
        self, history: tuple[int, ...], known_token: int, /
    ) -> Sequence[int]: ...

    def reconcile_prefix(self, history: tuple[int, ...], /) -> None: ...


def _plain_nonnegative(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _token_tuple(
    value: object, name: str, *, lengths: frozenset[int]
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an integer sequence")
    tokens: list[int] = []
    for raw in value:
        if isinstance(raw, bool) or not isinstance(raw, numbers.Integral):
            raise TypeError(f"{name} must contain integers")
        tokens.append(int(raw))
    result = tuple(tokens)
    if len(result) not in lengths:
        expected = " or ".join(str(length) for length in sorted(lengths))
        raise ValueError(f"{name} must contain exactly {expected} token(s)")
    return result


def _owner_metric(owner: object, name: str) -> int:
    metrics = getattr(owner, "metrics", None)
    values = dict(metrics()) if callable(metrics) else {}
    value = values.get(name, 0)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _tensor_stamp(value: torch.Tensor | None) -> object:
    if value is None:
        return None
    raw = value.detach().contiguous().to(device="cpu").view(torch.uint8).numpy()
    return (
        id(value),
        int(value._version),
        tuple(value.shape),
        str(value.dtype),
        str(value.device),
        hashlib.sha256(raw).hexdigest(),
    )


def _model_state_stamp(model: StreamedQwen38) -> tuple[object, ...]:
    states: list[object] = []
    for state in model._layer_states:
        if state is None:
            states.append(None)
            continue
        states.append(
            (
                type(state),
                *(
                    _tensor_stamp(getattr(state, name, None))
                    for name in (
                        "key",
                        "value",
                        "crsa_log_usage",
                        "conv",
                        "recurrent",
                    )
                ),
            )
        )
    return (
        model.next_position,
        model.state_batch_size,
        model.state_poisoned,
        model._continuation_block_runtime_identity(),
        tuple(states),
        _tensor_stamp(model._graft_history),
        id(model._pending_block_stage),
    )


def _tensor_version_stamp(value: torch.Tensor | None) -> object:
    if value is None:
        return None
    return (
        id(value),
        int(value._version),
        tuple(value.shape),
        str(value.dtype),
        str(value.device),
    )


def _model_state_version_stamp(model: StreamedQwen38) -> tuple[object, ...]:
    states = []
    for state in model._layer_states:
        if state is None:
            states.append(None)
            continue
        states.append(
            (
                type(state),
                *(
                    _tensor_version_stamp(getattr(state, name, None))
                    for name in (
                        "key",
                        "value",
                        "crsa_log_usage",
                        "conv",
                        "recurrent",
                    )
                ),
            )
        )
    return (
        model.next_position,
        model.state_batch_size,
        model.state_poisoned,
        tuple(states),
        _tensor_version_stamp(model._graft_history),
        id(model._pending_block_stage),
    )


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class K2SpeculativeRoundEvidence:
    """Immutable accounting for one target verification or final single step."""

    round_index: int
    start_pos: int
    end_pos: int
    proposed_token_ids: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    emitted_token_ids: tuple[int, ...]
    accepted_prefix_length: int
    replay_kind: ReplayKind
    forward_passes: int
    head_scans: int
    source_body_bytes: int
    linear_calls: int
    provider_guard_bytes: int
    provider_guard_seconds: float
    seconds: float
    state_bytes: int
    stopped_on_eos: bool

    def __post_init__(self) -> None:
        for name in (
            "round_index",
            "start_pos",
            "end_pos",
            "accepted_prefix_length",
            "forward_passes",
            "head_scans",
            "source_body_bytes",
            "linear_calls",
            "provider_guard_bytes",
            "state_bytes",
        ):
            _plain_nonnegative(getattr(self, name), name)
        _finite_nonnegative(self.provider_guard_seconds, "provider_guard_seconds")
        _finite_nonnegative(self.seconds, "seconds")
        if self.provider_guard_seconds > self.seconds:
            raise ValueError("provider integrity time exceeds round time")
        if self.end_pos - self.start_pos != len(self.emitted_token_ids):
            raise ValueError("round cursor delta must equal emitted token count")
        if any(not isinstance(token, int) for token in self.proposed_token_ids):
            raise TypeError("proposed_token_ids must contain integers")
        if any(not isinstance(token, int) for token in self.target_token_ids):
            raise TypeError("target_token_ids must contain integers")
        if any(not isinstance(token, int) for token in self.emitted_token_ids):
            raise TypeError("emitted_token_ids must contain integers")
        if not isinstance(self.stopped_on_eos, bool):
            raise TypeError("stopped_on_eos must be boolean")
        expected = {
            "commit-k2": (2, 2, 1, 1),
            "eos0-decode": (2, 1, 2, 1),
            "mismatch0-decode": (2, 1, 2, 1),
            "mismatch1-restage": (2, 2, 2, 1),
            "remaining-single": (0, 1, 1, 1),
        }
        if self.replay_kind not in expected:
            raise ValueError("replay_kind is invalid")
        proposal_count, emitted_count, passes, scans = expected[self.replay_kind]
        if (
            len(self.proposed_token_ids) != proposal_count
            or len(self.emitted_token_ids) != emitted_count
            or self.forward_passes != passes
            or self.head_scans != scans
        ):
            raise ValueError("round evidence disagrees with replay kind")
        if len(self.target_token_ids) not in (1, 2):
            raise ValueError("target_token_ids must contain one or two tokens")
        if not 0 <= self.accepted_prefix_length <= len(self.proposed_token_ids):
            raise ValueError("accepted_prefix_length exceeds the proposal")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in (
            "proposed_token_ids",
            "target_token_ids",
            "emitted_token_ids",
        ):
            payload[name] = list(payload[name])
        return payload


@dataclass(frozen=True, slots=True)
class K2SpeculativeGenerationEvidence:
    """Aggregate exact-greedy receipt for one speculative request."""

    schema: str
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    rounds: tuple[K2SpeculativeRoundEvidence, ...]
    prefill_forward_passes: int
    forward_passes: int
    head_scans: int
    accepted_draft_tokens: int
    source_body_bytes: int
    linear_calls: int
    provider_guard_bytes: int
    provider_guard_seconds: float
    seconds: float
    state_bytes: int
    stopped_on_eos: bool

    def __post_init__(self) -> None:
        if self.schema != QWEN38_K2_SPECULATIVE_SCHEMA:
            raise ValueError("speculative evidence schema is invalid")
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        if not self.rounds:
            raise ValueError("speculative evidence must contain at least one round")
        for name in (
            "prefill_forward_passes",
            "forward_passes",
            "head_scans",
            "accepted_draft_tokens",
            "source_body_bytes",
            "linear_calls",
            "provider_guard_bytes",
            "state_bytes",
        ):
            _plain_nonnegative(getattr(self, name), name)
        _finite_nonnegative(self.provider_guard_seconds, "provider_guard_seconds")
        _finite_nonnegative(self.seconds, "seconds")
        if self.provider_guard_seconds > self.seconds:
            raise ValueError("provider integrity time exceeds generation time")
        if not isinstance(self.stopped_on_eos, bool):
            raise TypeError("stopped_on_eos must be boolean")
        emitted = tuple(token for row in self.rounds for token in row.emitted_token_ids)
        if emitted != self.generated_token_ids:
            raise ValueError("round tokens differ from generated_token_ids")
        if self.forward_passes != self.prefill_forward_passes + sum(
            row.forward_passes for row in self.rounds
        ):
            raise ValueError("aggregate forward-pass accounting is inconsistent")
        if self.head_scans != sum(row.head_scans for row in self.rounds):
            raise ValueError("aggregate head-scan accounting is inconsistent")
        if self.accepted_draft_tokens != sum(
            row.accepted_prefix_length for row in self.rounds
        ):
            raise ValueError("aggregate accepted-token accounting is inconsistent")
        if self.provider_guard_bytes != sum(
            row.provider_guard_bytes for row in self.rounds
        ):
            raise ValueError("aggregate provider integrity bytes are inconsistent")
        if self.provider_guard_seconds != sum(
            row.provider_guard_seconds for row in self.rounds
        ):
            raise ValueError("aggregate provider integrity time is inconsistent")
        if self.state_bytes != self.rounds[-1].state_bytes:
            raise ValueError("aggregate state bytes differ from the final round")
        if self.stopped_on_eos != self.rounds[-1].stopped_on_eos:
            raise ValueError("aggregate EOS state differs from the final round")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "prompt_token_ids": list(self.prompt_token_ids),
            "generated_token_ids": list(self.generated_token_ids),
            "rounds": [row.to_dict() for row in self.rounds],
            "prefill_forward_passes": self.prefill_forward_passes,
            "forward_passes": self.forward_passes,
            "head_scans": self.head_scans,
            "accepted_draft_tokens": self.accepted_draft_tokens,
            "source_body_bytes": self.source_body_bytes,
            "linear_calls": self.linear_calls,
            "provider_guard_bytes": self.provider_guard_bytes,
            "provider_guard_seconds": self.provider_guard_seconds,
            "seconds": self.seconds,
            "state_bytes": self.state_bytes,
            "stopped_on_eos": self.stopped_on_eos,
        }


@dataclass(frozen=True, slots=True)
class K2SpeculativeGenerationResult:
    """Generated tokens and their immutable execution receipt."""

    token_ids: tuple[int, ...]
    evidence: K2SpeculativeGenerationEvidence

    def __post_init__(self) -> None:
        if self.token_ids != self.evidence.generated_token_ids:
            raise ValueError("result tokens differ from generation evidence")


@dataclass(frozen=True, slots=True)
class K4SpeculativeRoundEvidence:
    """Self-sealed accounting for one exact K=4 verification or tail step.

    The digest detects accidental or post-hoc receipt changes.  It is not an
    authentication mechanism; the authenticated outer execution receipt owns
    that boundary.
    """

    schema: str
    round_index: int
    start_pos: int
    end_pos: int
    eos_token_ids: tuple[int, ...]
    proposed_token_ids: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    emitted_token_ids: tuple[int, ...]
    accepted_prefix_length: int
    replay_kind: K4ReplayKind
    forward_passes: int
    head_scans: int
    source_body_bytes: int
    linear_calls: int
    provider_guard_bytes: int
    provider_guard_seconds: float
    seconds: float
    state_bytes: int
    stopped_on_eos: bool
    evidence_sha256: str

    @property
    def state_committed(self) -> bool:
        return self.replay_kind not in {
            "eos-uncommitted",
            "mismatch3-uncommitted",
            "terminal-single-uncommitted",
        }

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "round_index": self.round_index,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "eos_token_ids": list(self.eos_token_ids),
            "proposed_token_ids": list(self.proposed_token_ids),
            "target_token_ids": list(self.target_token_ids),
            "emitted_token_ids": list(self.emitted_token_ids),
            "accepted_prefix_length": self.accepted_prefix_length,
            "replay_kind": self.replay_kind,
            "forward_passes": self.forward_passes,
            "head_scans": self.head_scans,
            "source_body_bytes": self.source_body_bytes,
            "linear_calls": self.linear_calls,
            "provider_guard_bytes": self.provider_guard_bytes,
            "provider_guard_seconds": self.provider_guard_seconds,
            "seconds": self.seconds,
            "state_bytes": self.state_bytes,
            "stopped_on_eos": self.stopped_on_eos,
        }

    def __post_init__(self) -> None:
        if self.schema != QWEN38_K4_SPECULATIVE_ROUND_SCHEMA:
            raise ValueError("K=4 round evidence schema is invalid")
        for name in (
            "round_index",
            "start_pos",
            "end_pos",
            "accepted_prefix_length",
            "forward_passes",
            "head_scans",
            "source_body_bytes",
            "linear_calls",
            "provider_guard_bytes",
            "state_bytes",
        ):
            _plain_nonnegative(getattr(self, name), name)
        _finite_nonnegative(self.provider_guard_seconds, "provider_guard_seconds")
        _finite_nonnegative(self.seconds, "seconds")
        if self.provider_guard_seconds > self.seconds:
            raise ValueError("provider integrity time exceeds round time")
        if self.end_pos - self.start_pos != len(self.emitted_token_ids):
            raise ValueError("round cursor delta must equal emitted token count")
        if self.head_scans != 1:
            raise ValueError("every K=4 round must perform exactly one head scan")
        if not isinstance(self.stopped_on_eos, bool):
            raise TypeError("stopped_on_eos must be boolean")
        if not isinstance(self.eos_token_ids, tuple) or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in self.eos_token_ids
        ):
            raise TypeError("eos_token_ids must contain non-negative integers")
        if tuple(sorted(set(self.eos_token_ids))) != self.eos_token_ids:
            raise ValueError("eos_token_ids must be sorted unique non-negative IDs")
        for name in (
            "proposed_token_ids",
            "target_token_ids",
            "emitted_token_ids",
        ):
            if not isinstance(getattr(self, name), tuple):
                raise TypeError(f"{name} must be a tuple")
            if any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in getattr(self, name)
            ):
                raise TypeError(f"{name} must contain non-negative integers")

        if self.replay_kind in {
            "terminal-single",
            "terminal-single-uncommitted",
        }:
            if self.proposed_token_ids:
                raise ValueError("terminal tail rounds cannot contain a proposal")
            if len(self.target_token_ids) != 1:
                raise ValueError("terminal tail rounds require one target token")
            expected_emitted = self.target_token_ids
            expected_accepted = 0
            expected_passes = int(self.replay_kind == "terminal-single")
            expected_committed = self.replay_kind == "terminal-single"
            expected_stopped = self.target_token_ids[0] in self.eos_token_ids
        else:
            if len(self.proposed_token_ids) != 4:
                raise ValueError("K=4 verification requires exactly four proposals")
            if len(self.target_token_ids) != 4:
                raise ValueError("K=4 verification requires exactly four targets")
            common = 0
            while (
                common < 4
                and self.proposed_token_ids[common] == self.target_token_ids[common]
            ):
                common += 1
            valid_width = 4 if common == 4 else common + 1
            eos_index = next(
                (
                    index
                    for index, token in enumerate(self.target_token_ids[:valid_width])
                    if token in self.eos_token_ids
                ),
                None,
            )
            if eos_index is not None:
                expected_emitted = self.target_token_ids[: eos_index + 1]
                expected_accepted = min(common, eos_index + 1)
                if self.replay_kind == "eos-uncommitted":
                    expected_kind = "eos-uncommitted"
                    expected_passes = 1
                    expected_committed = False
                elif eos_index == 0:
                    expected_kind: K4ReplayKind = "eos0-decode"
                    expected_passes = 2
                    expected_committed = True
                elif eos_index == 3 and common == 4:
                    expected_kind = "eos3-commit-k4"
                    expected_passes = 1
                    expected_committed = True
                else:
                    eos_replays: dict[int, K4ReplayKind] = {
                        1: "eos1-restage",
                        2: "eos2-restage",
                        3: "eos3-restage",
                    }
                    expected_kind = eos_replays[eos_index]
                    expected_passes = 2
                    expected_committed = True
                expected_stopped = True
            elif common == 4:
                expected_emitted = self.proposed_token_ids
                expected_accepted = 4
                expected_kind = "commit-k4"
                expected_passes = 1
                expected_committed = True
                expected_stopped = False
            else:
                expected_emitted = self.target_token_ids[: common + 1]
                expected_accepted = common
                if common == 3 and self.replay_kind == "mismatch3-uncommitted":
                    expected_kind = "mismatch3-uncommitted"
                    expected_passes = 1
                    expected_committed = False
                else:
                    mismatch_replays: dict[int, K4ReplayKind] = {
                        0: "mismatch0-decode",
                        1: "mismatch1-restage",
                        2: "mismatch2-restage",
                        3: "mismatch3-restage",
                    }
                    expected_kind = mismatch_replays[common]
                    expected_passes = 2
                    expected_committed = True
                expected_stopped = False
            if self.replay_kind != expected_kind:
                raise ValueError("K=4 replay kind disagrees with target transition")

        if self.emitted_token_ids != expected_emitted:
            raise ValueError("emitted tokens disagree with the target transition")
        if self.accepted_prefix_length != expected_accepted:
            raise ValueError("accepted prefix disagrees with the target transition")
        if self.forward_passes != expected_passes:
            raise ValueError("forward-pass count disagrees with the replay kind")
        if self.state_committed != expected_committed:
            raise ValueError("state commit disagrees with the replay kind")
        if self.stopped_on_eos != expected_stopped:
            raise ValueError("EOS state disagrees with the target transition")
        _require_sha256(self.evidence_sha256, "evidence_sha256")
        if self.evidence_sha256 != _canonical_sha256(self._unsigned_dict()):
            raise ValueError("K=4 round evidence digest is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {**self._unsigned_dict(), "evidence_sha256": self.evidence_sha256}


def _sealed_k4_round(**values: Any) -> K4SpeculativeRoundEvidence:
    unsigned = K4SpeculativeRoundEvidence.__new__(K4SpeculativeRoundEvidence)
    for field_name, field_value in values.items():
        object.__setattr__(unsigned, field_name, field_value)
    payload = unsigned._unsigned_dict()
    return K4SpeculativeRoundEvidence(
        **values,
        evidence_sha256=_canonical_sha256(payload),
    )


@dataclass(frozen=True, slots=True)
class K4SpeculativeGenerationEvidence:
    """Self-sealed aggregate exact-greedy K=4 execution receipt."""

    schema: str
    prompt_token_ids: tuple[int, ...]
    eos_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    rounds: tuple[K4SpeculativeRoundEvidence, ...]
    prefill_forward_passes: int
    forward_passes: int
    head_scans: int
    accepted_draft_tokens: int
    source_body_bytes: int
    linear_calls: int
    provider_guard_bytes: int
    provider_guard_seconds: float
    seconds: float
    state_bytes: int
    stopped_on_eos: bool
    evidence_sha256: str

    @property
    def final_state_committed(self) -> bool:
        return self.rounds[-1].state_committed

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "prompt_token_ids": list(self.prompt_token_ids),
            "eos_token_ids": list(self.eos_token_ids),
            "generated_token_ids": list(self.generated_token_ids),
            "rounds": [row.to_dict() for row in self.rounds],
            "prefill_forward_passes": self.prefill_forward_passes,
            "forward_passes": self.forward_passes,
            "head_scans": self.head_scans,
            "accepted_draft_tokens": self.accepted_draft_tokens,
            "source_body_bytes": self.source_body_bytes,
            "linear_calls": self.linear_calls,
            "provider_guard_bytes": self.provider_guard_bytes,
            "provider_guard_seconds": self.provider_guard_seconds,
            "seconds": self.seconds,
            "state_bytes": self.state_bytes,
            "stopped_on_eos": self.stopped_on_eos,
        }

    def __post_init__(self) -> None:
        if self.schema != QWEN38_K4_SPECULATIVE_SCHEMA:
            raise ValueError("K=4 speculative evidence schema is invalid")
        if not isinstance(self.prompt_token_ids, tuple) or not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        if any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in self.prompt_token_ids
        ):
            raise TypeError("prompt_token_ids must contain non-negative integers")
        if not isinstance(self.generated_token_ids, tuple) or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in self.generated_token_ids
        ):
            raise TypeError("generated_token_ids must contain non-negative integers")
        if not isinstance(self.eos_token_ids, tuple) or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in self.eos_token_ids
        ):
            raise TypeError("eos_token_ids must contain non-negative integers")
        if not isinstance(self.rounds, tuple) or not self.rounds:
            raise ValueError("K=4 speculative evidence must contain a round")
        if any(not isinstance(row, K4SpeculativeRoundEvidence) for row in self.rounds):
            raise TypeError("rounds must contain K4SpeculativeRoundEvidence")
        if tuple(sorted(set(self.eos_token_ids))) != self.eos_token_ids:
            raise ValueError("eos_token_ids must be sorted and unique")
        for name in (
            "prefill_forward_passes",
            "forward_passes",
            "head_scans",
            "accepted_draft_tokens",
            "source_body_bytes",
            "linear_calls",
            "provider_guard_bytes",
            "state_bytes",
        ):
            _plain_nonnegative(getattr(self, name), name)
        _finite_nonnegative(self.provider_guard_seconds, "provider_guard_seconds")
        _finite_nonnegative(self.seconds, "seconds")
        if self.provider_guard_seconds > self.seconds:
            raise ValueError("provider integrity time exceeds generation time")
        if not isinstance(self.stopped_on_eos, bool):
            raise TypeError("stopped_on_eos must be boolean")
        cursor = len(self.prompt_token_ids)
        emitted: list[int] = []
        stopped_rows = 0
        for index, row in enumerate(self.rounds):
            if row.round_index != index:
                raise ValueError("K=4 round indexes must be contiguous")
            if row.eos_token_ids != self.eos_token_ids:
                raise ValueError("round EOS policy differs from generation policy")
            if row.start_pos != cursor:
                raise ValueError("K=4 round cursor chain is discontinuous")
            cursor = row.end_pos
            emitted.extend(row.emitted_token_ids)
            if row.stopped_on_eos:
                stopped_rows += 1
                if index != len(self.rounds) - 1:
                    raise ValueError("no round may follow a terminal EOS round")
            if not row.state_committed and index != len(self.rounds) - 1:
                raise ValueError("only the final K=4 round may discard its state")
        if tuple(emitted) != self.generated_token_ids:
            raise ValueError("round tokens differ from generated_token_ids")
        if self.forward_passes != self.prefill_forward_passes + sum(
            row.forward_passes for row in self.rounds
        ):
            raise ValueError("aggregate forward-pass accounting is inconsistent")
        if self.head_scans != sum(row.head_scans for row in self.rounds):
            raise ValueError("aggregate head-scan accounting is inconsistent")
        if self.accepted_draft_tokens != sum(
            row.accepted_prefix_length for row in self.rounds
        ):
            raise ValueError("aggregate accepted-token accounting is inconsistent")
        if self.provider_guard_bytes != sum(
            row.provider_guard_bytes for row in self.rounds
        ):
            raise ValueError("aggregate provider integrity bytes are inconsistent")
        if self.provider_guard_seconds != sum(
            row.provider_guard_seconds for row in self.rounds
        ):
            raise ValueError("aggregate provider integrity time is inconsistent")
        if self.source_body_bytes < sum(row.source_body_bytes for row in self.rounds):
            raise ValueError("aggregate source bytes omit round work")
        if self.linear_calls < sum(row.linear_calls for row in self.rounds):
            raise ValueError("aggregate linear calls omit round work")
        if self.seconds < sum(row.seconds for row in self.rounds):
            raise ValueError("aggregate time omits round work")
        if self.state_bytes != self.rounds[-1].state_bytes:
            raise ValueError("aggregate state bytes differ from the final round")
        if self.stopped_on_eos != (stopped_rows == 1):
            raise ValueError("aggregate EOS state differs from the final round")
        _require_sha256(self.evidence_sha256, "evidence_sha256")
        if self.evidence_sha256 != _canonical_sha256(self._unsigned_dict()):
            raise ValueError("K=4 generation evidence digest is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {**self._unsigned_dict(), "evidence_sha256": self.evidence_sha256}


def _sealed_k4_generation(**values: Any) -> K4SpeculativeGenerationEvidence:
    unsigned = K4SpeculativeGenerationEvidence.__new__(K4SpeculativeGenerationEvidence)
    for field_name, field_value in values.items():
        object.__setattr__(unsigned, field_name, field_value)
    payload = unsigned._unsigned_dict()
    return K4SpeculativeGenerationEvidence(
        **values,
        evidence_sha256=_canonical_sha256(payload),
    )


@dataclass(frozen=True, slots=True)
class K4SpeculativeGenerationResult:
    """Generated tokens and their immutable K=4 execution receipt."""

    token_ids: tuple[int, ...]
    evidence: K4SpeculativeGenerationEvidence

    def __post_init__(self) -> None:
        if self.token_ids != self.evidence.generated_token_ids:
            raise ValueError("result tokens differ from K=4 generation evidence")


@dataclass(frozen=True, slots=True)
class RollingK4SpeculativeRoundEvidence:
    round_index: int
    start_pos: int
    end_pos: int
    known_token_id: int
    proposed_token_ids: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    emitted_token_ids: tuple[int, ...]
    accepted_prefix_length: int
    correction_token_id: int | None
    forward_passes: int
    head_scans: int
    source_body_bytes: int
    linear_calls: int
    provider_guard_bytes: int
    provider_guard_seconds: float
    seconds: float
    state_bytes: int
    state_committed: bool
    stopped_on_eos: bool
    window_size: int = 4
    request_window_ceiling: int | None = None
    provider_proposed_token_ids: tuple[int, ...] = ()
    round_policy: RoundWindowPolicy | None = None

    def __post_init__(self) -> None:
        for name in (
            "round_index",
            "start_pos",
            "end_pos",
            "known_token_id",
            "accepted_prefix_length",
            "forward_passes",
            "head_scans",
            "source_body_bytes",
            "linear_calls",
            "provider_guard_bytes",
            "state_bytes",
        ):
            _plain_nonnegative(getattr(self, name), name)
        _finite_nonnegative(self.provider_guard_seconds, "provider_guard_seconds")
        _finite_nonnegative(self.seconds, "seconds")
        if (
            isinstance(self.window_size, bool)
            or not isinstance(self.window_size, int)
            or not 1 <= self.window_size <= StreamedQwen38.MAX_CONTINUATION_BLOCK_WIDTH
        ):
            raise ValueError("rolling target window is outside [1, 16]")
        ceiling = (
            self.window_size
            if self.request_window_ceiling is None
            else self.request_window_ceiling
        )
        if (
            isinstance(ceiling, bool)
            or not isinstance(ceiling, int)
            or not self.window_size
            <= ceiling
            <= (StreamedQwen38.MAX_CONTINUATION_BLOCK_WIDTH)
        ):
            raise ValueError("rolling request ceiling is invalid")
        tail = not self.proposed_token_ids and not self.target_token_ids
        if not tail and (
            len(self.proposed_token_ids) != self.window_size - 1
            or len(self.target_token_ids) != self.window_size
        ):
            raise ValueError("rolling round has the wrong proposal width")
        if not 0 <= self.accepted_prefix_length < self.window_size:
            raise ValueError("rolling accepted prefix is outside the target window")
        if self.end_pos - self.start_pos != len(self.emitted_token_ids):
            raise ValueError("rolling cursor delta differs from emitted tokens")
        if tail:
            if (
                self.emitted_token_ids != (self.known_token_id,)
                or self.accepted_prefix_length != 0
                or self.correction_token_id is not None
                or self.head_scans != 0
                or self.forward_passes not in (0, 1)
            ):
                raise ValueError("rolling terminal carry is inconsistent")
        elif self.forward_passes != 1 or self.head_scans != 1:
            raise ValueError("rolling wave must use one target pass and head scan")
        if not isinstance(self.state_committed, bool) or not isinstance(
            self.stopped_on_eos, bool
        ):
            raise TypeError("rolling round state flags must be boolean")
        provider_proposal = tuple(self.provider_proposed_token_ids)
        if tail:
            if provider_proposal or self.round_policy is not None:
                raise ValueError("rolling terminal carry cannot have a proposal policy")
        elif self.round_policy is None:
            if provider_proposal and provider_proposal != self.proposed_token_ids:
                raise ValueError("fixed rolling provider proposal differs from target")
            provider_proposal = self.proposed_token_ids
        else:
            if not isinstance(self.round_policy, RoundWindowPolicy):
                raise TypeError("round_policy must be RoundWindowPolicy")
            if (
                self.round_policy.request_window_ceiling != ceiling
                or self.round_policy.chosen_window != self.window_size
                or len(provider_proposal) != ceiling - 1
                or provider_proposal[: self.window_size - 1] != self.proposed_token_ids
                or not self.round_policy.matches_provider_tokens(provider_proposal)
            ):
                raise ValueError("adaptive rolling proposal policy is inconsistent")
        object.__setattr__(self, "request_window_ceiling", ceiling)
        object.__setattr__(self, "provider_proposed_token_ids", provider_proposal)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RollingK4SpeculativeGenerationEvidence:
    schema: str
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    rounds: tuple[RollingK4SpeculativeRoundEvidence, ...]
    prefill_forward_passes: int
    forward_passes: int
    head_scans: int
    accepted_draft_tokens: int
    source_body_bytes: int
    linear_calls: int
    provider_guard_bytes: int
    provider_guard_seconds: float
    seconds: float
    state_bytes: int
    stopped_on_eos: bool
    window_size: int = 4
    adaptive_windows: bool = False

    @property
    def final_state_committed(self) -> bool:
        return self.rounds[-1].state_committed

    def __post_init__(self) -> None:
        expected_schema = (
            QWEN38_ROLLING_K4_SPECULATIVE_SCHEMA
            if self.window_size == 4 and not self.adaptive_windows
            else QWEN38_ROLLING_SPECULATIVE_SCHEMA
        )
        if self.schema != expected_schema:
            raise ValueError("rolling generation schema is invalid")
        if (
            isinstance(self.window_size, bool)
            or not isinstance(self.window_size, int)
            or not 2 <= self.window_size <= StreamedQwen38.MAX_CONTINUATION_BLOCK_WIDTH
        ):
            raise ValueError("rolling target window is outside [2, 16]")
        if not self.prompt_token_ids or not self.rounds:
            raise ValueError("rolling generation requires prompt and rounds")
        if not isinstance(self.adaptive_windows, bool):
            raise TypeError("adaptive_windows must be boolean")
        if self.adaptive_windows:
            if any(
                row.request_window_ceiling != self.window_size
                or (bool(row.proposed_token_ids) and row.round_policy is None)
                for row in self.rounds
            ):
                raise ValueError("adaptive rolling rounds disagree on policy")
        elif any(
            row.window_size != self.window_size
            or row.request_window_ceiling != self.window_size
            or row.round_policy is not None
            for row in self.rounds
        ):
            raise ValueError("rolling rounds disagree on target window")
        emitted = tuple(token for row in self.rounds for token in row.emitted_token_ids)
        if emitted != self.generated_token_ids:
            raise ValueError("rolling round tokens differ from generated tokens")
        if self.forward_passes != self.prefill_forward_passes + sum(
            row.forward_passes for row in self.rounds
        ):
            raise ValueError("rolling target forward accounting is inconsistent")
        if self.head_scans != 1 + sum(row.head_scans for row in self.rounds):
            raise ValueError("rolling head-scan accounting is inconsistent")
        if self.accepted_draft_tokens != sum(
            row.accepted_prefix_length for row in self.rounds
        ):
            raise ValueError("rolling accepted-token accounting is inconsistent")
        if self.provider_guard_bytes != sum(
            row.provider_guard_bytes for row in self.rounds
        ):
            raise ValueError("rolling provider-byte accounting is inconsistent")
        if self.stopped_on_eos != self.rounds[-1].stopped_on_eos:
            raise ValueError("rolling EOS accounting is inconsistent")
        cursor = len(self.prompt_token_ids)
        for index, row in enumerate(self.rounds):
            if row.round_index != index or row.start_pos != cursor:
                raise ValueError("rolling round cursor chain is inconsistent")
            cursor = row.end_pos
            if not row.state_committed and index != len(self.rounds) - 1:
                raise ValueError("only the final rolling round may discard state")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["request_window_ceiling"] = self.window_size
        value["used_window_sizes"] = list(self.used_window_sizes)
        return value

    @property
    def used_window_sizes(self) -> tuple[int, ...]:
        return tuple(row.window_size for row in self.rounds if row.target_token_ids)


@dataclass(frozen=True, slots=True)
class RollingK4SpeculativeGenerationResult:
    token_ids: tuple[int, ...]
    evidence: RollingK4SpeculativeGenerationEvidence

    def __post_init__(self) -> None:
        if self.token_ids != self.evidence.generated_token_ids:
            raise ValueError("rolling result differs from generation evidence")


class Qwen38K2SpeculativeDecoder:
    """Generate exact greedy tokens with untrusted K=2 draft proposals."""

    def __init__(self, model: StreamedQwen38, draft_provider: DraftProvider) -> None:
        if not isinstance(model, StreamedQwen38):
            raise TypeError("model must be a StreamedQwen38")
        if not callable(draft_provider):
            raise TypeError("draft_provider must be callable")
        self.model = model
        self.draft_provider = draft_provider

    def _proposal(self, history: tuple[int, ...]) -> tuple[tuple[int, int], int, float]:
        stamp_started = time.perf_counter()
        before = _model_state_stamp(self.model)
        integrity_seconds = time.perf_counter() - stamp_started
        integrity_bytes = self.model.state_bytes
        failure: Exception | None = None
        proposal: tuple[int, ...] | None = None
        try:
            raw = self.draft_provider(history)
        except Exception as exc:
            failure = exc
        else:
            try:
                proposal = _token_tuple(raw, "draft proposal", lengths=frozenset({2}))
                if any(
                    token < 0 or token >= self.model.config.vocab_size
                    for token in proposal
                ):
                    raise ValueError("draft token outside checkpoint vocabulary")
            except (TypeError, ValueError) as exc:
                failure = exc
        stamp_started = time.perf_counter()
        try:
            changed = _model_state_stamp(self.model) != before
        except Exception:
            changed = True
        integrity_seconds += time.perf_counter() - stamp_started
        integrity_bytes += self.model.state_bytes
        if changed:
            self.model.reset_state(release=True)
            raise Qwen38SpeculativeError("draft provider changed target model state")
        if failure is not None:
            if isinstance(failure, (TypeError, ValueError)):
                raise Qwen38SpeculativeError(str(failure)) from failure
            raise Qwen38SpeculativeError(
                f"draft provider failed: {type(failure).__name__}: {failure}"
            ) from failure
        assert proposal is not None
        return (proposal[0], proposal[1]), integrity_bytes, integrity_seconds

    def _reconcile_provider(self, history: tuple[int, ...]) -> tuple[int, float]:
        """Notify a stateful drafter without extending its target-model authority."""

        missing = object()
        static_callback = inspect.getattr_static(
            self.draft_provider, "reconcile", missing
        )
        if static_callback is missing:
            return 0, 0.0

        stamp_started = time.perf_counter()
        before = _model_state_stamp(self.model)
        integrity_seconds = time.perf_counter() - stamp_started
        integrity_bytes = self.model.state_bytes
        failure: Exception | None = None
        try:
            callback = getattr(self.draft_provider, "reconcile")
            if not callable(callback):
                raise TypeError("draft provider reconcile attribute is not callable")
            callback(history)
        except Exception as exc:
            failure = exc
        stamp_started = time.perf_counter()
        try:
            changed = _model_state_stamp(self.model) != before
        except Exception:
            changed = True
        integrity_seconds += time.perf_counter() - stamp_started
        integrity_bytes += self.model.state_bytes
        if changed:
            self.model.reset_state(release=True)
            raise Qwen38SpeculativeError(
                "draft provider changed target model state while reconciling"
            )
        if failure is not None:
            self.model.reset_state(release=True)
            raise Qwen38SpeculativeError(
                "draft provider reconciliation failed: "
                f"{type(failure).__name__}: {failure}"
            ) from failure
        return integrity_bytes, integrity_seconds

    def _scan(
        self,
        hidden: torch.Tensor,
        *,
        block_rows: int,
    ) -> tuple[int, ...]:
        _values, selected = self.model.pager.topk_logits(
            hidden,
            k=1,
            name=self.model.output_head_name,
            block_rows=block_rows,
        )
        expected = (*hidden.shape[:-1], 1)
        if tuple(selected.shape) != expected:
            raise Qwen38SpeculativeError("LM head returned an invalid target shape")
        return tuple(
            int(value)
            for value in selected[..., 0].detach().to(device="cpu").reshape(-1)
        )

    def _single_round(
        self,
        hidden: torch.Tensor,
        *,
        round_index: int,
        eos: frozenset[int],
        block_rows: int,
    ) -> tuple[torch.Tensor, K2SpeculativeRoundEvidence]:
        start_pos = self.model.next_position
        source_start = _owner_metric(
            self.model.pager.source, "network_or_source_body_bytes"
        )
        linears_start = _owner_metric(self.model.pager, "linear_calls")
        started = time.perf_counter()
        values, selected = self.model.pager.topk_logits(
            hidden[:, -1],
            k=1,
            name=self.model.output_head_name,
            block_rows=block_rows,
        )
        if tuple(selected.shape) != (1, 1):
            raise Qwen38SpeculativeError(
                "LM head returned an invalid single target shape"
            )
        target = int(selected[0, 0].item())
        del values, selected
        next_hidden, _state_evidence = self.model.decode([[target]])
        stopped = target in eos
        evidence = K2SpeculativeRoundEvidence(
            round_index=round_index,
            start_pos=start_pos,
            end_pos=self.model.next_position,
            proposed_token_ids=(),
            target_token_ids=(target,),
            emitted_token_ids=(target,),
            accepted_prefix_length=0,
            replay_kind="remaining-single",
            forward_passes=1,
            head_scans=1,
            source_body_bytes=(
                _owner_metric(self.model.pager.source, "network_or_source_body_bytes")
                - source_start
            ),
            linear_calls=_owner_metric(self.model.pager, "linear_calls")
            - linears_start,
            provider_guard_bytes=0,
            provider_guard_seconds=0.0,
            seconds=time.perf_counter() - started,
            state_bytes=self.model.state_bytes,
            stopped_on_eos=stopped,
        )
        return next_hidden, evidence

    def generate(
        self,
        prompt_token_ids: object,
        *,
        max_new_tokens: int = 1,
        eos_token_ids: Iterable[int] = (),
        head_block_rows: int = Qwen38WeightPager.DEFAULT_HEAD_BLOCK_ROWS,
    ) -> K2SpeculativeGenerationResult:
        """Generate an exact greedy continuation and leave every token committed."""

        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        if (
            isinstance(head_block_rows, bool)
            or not isinstance(head_block_rows, int)
            or head_block_rows <= 0
        ):
            raise ValueError("head_block_rows must be a positive integer")
        prompt_tensor = self.model._token_tensor(prompt_token_ids)
        if prompt_tensor.shape[0] != 1:
            raise ValueError("speculative generation requires batch size one")
        prompt = tuple(
            int(value) for value in prompt_tensor[0].detach().to(device="cpu").tolist()
        )
        if len(prompt) + max_new_tokens > self.model.max_seq_len:
            raise ValueError("generation would exceed max_seq_len")
        if isinstance(eos_token_ids, (str, bytes)):
            raise TypeError("eos_token_ids must be an iterable of integers")
        eos_values: set[int] = set()
        try:
            for raw in eos_token_ids:
                if isinstance(raw, bool) or not isinstance(raw, numbers.Integral):
                    raise TypeError("eos_token_ids must contain integers")
                eos_values.add(int(raw))
        except TypeError as exc:
            if str(exc) == "eos_token_ids must contain integers":
                raise
            raise TypeError("eos_token_ids must be iterable") from exc
        if any(
            token < 0 or token >= self.model.config.vocab_size for token in eos_values
        ):
            raise ValueError("EOS token outside checkpoint vocabulary")
        eos = frozenset(eos_values)

        source_start = _owner_metric(
            self.model.pager.source, "network_or_source_body_bytes"
        )
        linears_start = _owner_metric(self.model.pager, "linear_calls")
        started = time.perf_counter()
        hidden, prefill_evidence = self.model.prefill(
            [prompt], reset=True, tokenwise=False
        )
        generated: list[int] = []
        rounds: list[K2SpeculativeRoundEvidence] = []
        stopped = False

        while len(generated) < max_new_tokens and not stopped:
            remaining = max_new_tokens - len(generated)
            round_index = len(rounds)
            if remaining == 1:
                hidden, row = self._single_round(
                    hidden,
                    round_index=round_index,
                    eos=eos,
                    block_rows=head_block_rows,
                )
                generated.extend(row.emitted_token_ids)
                rounds.append(row)
                stopped = row.stopped_on_eos
                continue

            round_source_start = _owner_metric(
                self.model.pager.source, "network_or_source_body_bytes"
            )
            round_linears_start = _owner_metric(self.model.pager, "linear_calls")
            round_started = time.perf_counter()
            proposal, integrity_bytes, integrity_seconds = self._proposal(
                (*prompt, *generated)
            )
            stage = self.model.stage_continuation_block([proposal])
            try:
                verification_hidden = torch.cat(
                    (hidden[:, -1:], stage.hidden[:, :1]), dim=1
                )
                targets = self._scan(
                    verification_hidden,
                    block_rows=head_block_rows,
                )
                if len(targets) != 2:  # pragma: no cover - shape check above.
                    raise Qwen38SpeculativeError(
                        "K=2 verification returned wrong width"
                    )
            except Exception:
                try:
                    self.model.discard_continuation_block(stage)
                except Exception:
                    pass
                raise

            target0, target1 = targets
            accepted = 0
            if proposal[0] == target0:
                accepted = 1
                if proposal[1] == target1:
                    accepted = 2

            if target0 in eos:
                self.model.discard_continuation_block(stage)
                hidden, _state_evidence = self.model.decode([[target0]])
                emitted = (target0,)
                accepted = min(accepted, 1)
                replay_kind: ReplayKind = "eos0-decode"
                passes = 2
            elif accepted == 2:
                hidden, _state_evidence = self.model.commit_continuation_block(stage)
                emitted = proposal
                replay_kind = "commit-k2"
                passes = 1
            elif accepted == 1:
                self.model.discard_continuation_block(stage)
                replay = self.model.stage_continuation_block([[proposal[0], target1]])
                hidden, _state_evidence = self.model.commit_continuation_block(replay)
                emitted = (proposal[0], target1)
                replay_kind = "mismatch1-restage"
                passes = 2
            else:
                self.model.discard_continuation_block(stage)
                hidden, _state_evidence = self.model.decode([[target0]])
                emitted = (target0,)
                replay_kind = "mismatch0-decode"
                passes = 2

            stopped = emitted[-1] in eos
            reconcile_bytes, reconcile_seconds = self._reconcile_provider(
                (*prompt, *generated, *emitted)
            )
            integrity_bytes += reconcile_bytes
            integrity_seconds += reconcile_seconds
            row = K2SpeculativeRoundEvidence(
                round_index=round_index,
                start_pos=stage.evidence.start_pos,
                end_pos=self.model.next_position,
                proposed_token_ids=proposal,
                target_token_ids=targets,
                emitted_token_ids=emitted,
                accepted_prefix_length=accepted,
                replay_kind=replay_kind,
                forward_passes=passes,
                head_scans=1,
                source_body_bytes=(
                    _owner_metric(
                        self.model.pager.source, "network_or_source_body_bytes"
                    )
                    - round_source_start
                ),
                linear_calls=_owner_metric(self.model.pager, "linear_calls")
                - round_linears_start,
                provider_guard_bytes=integrity_bytes,
                provider_guard_seconds=integrity_seconds,
                seconds=time.perf_counter() - round_started,
                state_bytes=self.model.state_bytes,
                stopped_on_eos=stopped,
            )
            generated.extend(emitted)
            rounds.append(row)

        evidence = K2SpeculativeGenerationEvidence(
            schema=QWEN38_K2_SPECULATIVE_SCHEMA,
            prompt_token_ids=prompt,
            generated_token_ids=tuple(generated),
            rounds=tuple(rounds),
            prefill_forward_passes=len(prefill_evidence),
            forward_passes=len(prefill_evidence)
            + sum(row.forward_passes for row in rounds),
            head_scans=sum(row.head_scans for row in rounds),
            accepted_draft_tokens=sum(row.accepted_prefix_length for row in rounds),
            source_body_bytes=(
                _owner_metric(self.model.pager.source, "network_or_source_body_bytes")
                - source_start
            ),
            linear_calls=_owner_metric(self.model.pager, "linear_calls")
            - linears_start,
            provider_guard_bytes=sum(row.provider_guard_bytes for row in rounds),
            provider_guard_seconds=sum(row.provider_guard_seconds for row in rounds),
            seconds=time.perf_counter() - started,
            state_bytes=self.model.state_bytes,
            stopped_on_eos=stopped,
        )
        return K2SpeculativeGenerationResult(tuple(generated), evidence)


class Qwen38K4SpeculativeDecoder(Qwen38K2SpeculativeDecoder):
    """Generate exact greedy tokens with one bounded target stage and head scan.

    A proposal is authority-free.  The target either commits the complete
    verified K=4 stage or discards it and replays only the proven prefix plus
    the first target correction.  No rejected or post-EOS suffix reaches
    committed continuation state.
    """

    def __init__(
        self,
        model: StreamedQwen38,
        draft_provider: K4DraftProvider,
        *,
        window_size: int = 4,
        adaptive_round_windows: bool = False,
        round_window_work_costs: Mapping[int, float] | None = None,
    ) -> None:
        if (
            isinstance(window_size, bool)
            or not isinstance(window_size, int)
            or not 2 <= window_size <= model.MAX_CONTINUATION_BLOCK_WIDTH
        ):
            raise ValueError("window_size must lie in [2, 16]")
        super().__init__(model, draft_provider)
        self.window_size = window_size
        if not isinstance(adaptive_round_windows, bool):
            raise TypeError("adaptive_round_windows must be boolean")
        if adaptive_round_windows and window_size not in {2, 4, 8, 16}:
            raise ValueError("adaptive round windows require a K2/K4/K8/K16 ceiling")
        if round_window_work_costs is not None:
            if not adaptive_round_windows:
                raise ValueError("round work costs require adaptive windows")
            costs = dict(round_window_work_costs)
            if any(
                isinstance(window, bool)
                or not isinstance(window, int)
                or window not in {1, 2, 4, 8, 16}
                or isinstance(cost, bool)
                or not isinstance(cost, (int, float))
                or not math.isfinite(float(cost))
                or float(cost) < 1.0
                for window, cost in costs.items()
            ):
                raise ValueError("round window work costs are invalid")
            self.round_window_work_costs = {
                window: float(cost) for window, cost in costs.items()
            }
        else:
            self.round_window_work_costs = None
        self.adaptive_round_windows = adaptive_round_windows

    def _provider_state_stamp(self) -> tuple[object, ...]:
        isolation = getattr(self.draft_provider, "target_state_isolation", None)
        if isolation in {
            "hidden-argument+shared-pager-only/v1",
            "no-target-state-access/v1",
        }:
            return _model_state_version_stamp(self.model)
        return _model_state_stamp(self.model)

    def _provider_guard_payload_bytes(self) -> int:
        if (
            getattr(self.draft_provider, "target_state_isolation", None)
            == "no-target-state-access/v1"
        ):
            return 0
        return self.model.state_bytes

    def _proposal_k4(
        self, history: tuple[int, ...]
    ) -> tuple[tuple[int, int, int, int], int, float]:
        if self.window_size != 4:
            raise Qwen38SpeculativeError(
                "one-shot K=4 generation requires window_size=4"
            )
        stamp_started = time.perf_counter()
        before = self._provider_state_stamp()
        integrity_seconds = time.perf_counter() - stamp_started
        integrity_bytes = self._provider_guard_payload_bytes()
        failure: Exception | None = None
        proposal: tuple[int, ...] | None = None
        try:
            raw = self.draft_provider(history)
        except Exception as exc:
            failure = exc
        else:
            try:
                proposal = _token_tuple(
                    raw,
                    "K=4 draft proposal",
                    lengths=frozenset({4}),
                )
                if any(
                    token < 0 or token >= self.model.config.vocab_size
                    for token in proposal
                ):
                    raise ValueError("draft token outside checkpoint vocabulary")
            except (TypeError, ValueError) as exc:
                failure = exc
        stamp_started = time.perf_counter()
        try:
            changed = self._provider_state_stamp() != before
        except Exception:
            changed = True
        integrity_seconds += time.perf_counter() - stamp_started
        integrity_bytes += self._provider_guard_payload_bytes()
        if changed:
            self.model.reset_state(release=True)
            raise Qwen38SpeculativeError("draft provider changed target model state")
        if failure is not None:
            if isinstance(failure, (TypeError, ValueError)):
                raise Qwen38SpeculativeError(str(failure)) from failure
            raise Qwen38SpeculativeError(
                f"draft provider failed: {type(failure).__name__}: {failure}"
            ) from failure
        assert proposal is not None
        return (
            (
                proposal[0],
                proposal[1],
                proposal[2],
                proposal[3],
            ),
            integrity_bytes,
            integrity_seconds,
        )

    def _rolling_proposal(
        self,
        history: tuple[int, ...],
        known_token: int,
        target_hidden: torch.Tensor,
    ) -> tuple[tuple[int, ...], RollingDraftProposal | None, int, float]:
        missing = object()
        state_callback_name = (
            "propose_round_state"
            if self.adaptive_round_windows
            else "propose_after_state"
        )
        state_callback = (
            inspect.getattr_static(
                self.draft_provider,
                state_callback_name,
                missing,
            )
            is not missing
        )
        callback_name = (
            state_callback_name
            if state_callback
            else "propose_round"
            if self.adaptive_round_windows
            else "propose_after"
        )
        if (
            inspect.getattr_static(self.draft_provider, callback_name, missing)
            is missing
        ):
            raise Qwen38SpeculativeError(
                f"rolling generation requires draft_provider.{callback_name}"
            )
        stamp_started = time.perf_counter()
        before = self._provider_state_stamp()
        integrity_seconds = time.perf_counter() - stamp_started
        integrity_bytes = self._provider_guard_payload_bytes()
        failure: Exception | None = None
        proposal: tuple[int, ...] | None = None
        proposal_evidence: RollingDraftProposal | None = None
        provider_hidden = target_hidden.detach().clone() if state_callback else None
        provider_hidden_stamp = (
            _tensor_stamp(provider_hidden) if provider_hidden is not None else None
        )
        try:
            callback = getattr(self.draft_provider, callback_name)
            if not callable(callback):
                raise TypeError(f"draft provider {callback_name} is not callable")
            raw = (
                callback(history, known_token, provider_hidden)
                if state_callback
                else callback(history, known_token)
            )
            if self.adaptive_round_windows:
                if not isinstance(raw, RollingDraftProposal):
                    raise TypeError(
                        "adaptive draft provider returned no RollingDraftProposal"
                    )
                proposal_evidence = raw
                raw = raw.token_ids
            proposal = _token_tuple(
                raw,
                "rolling draft proposal",
                lengths=frozenset({self.window_size - 1}),
            )
            if any(
                token < 0 or token >= self.model.config.vocab_size for token in proposal
            ):
                raise ValueError("rolling draft token outside checkpoint vocabulary")
        except Exception as exc:
            failure = exc
        if (
            provider_hidden is not None
            and _tensor_stamp(provider_hidden) != provider_hidden_stamp
        ):
            failure = Qwen38SpeculativeError(
                "stateful rolling provider mutated its hidden argument"
            )
        stamp_started = time.perf_counter()
        try:
            changed = self._provider_state_stamp() != before
        except Exception:
            changed = True
        integrity_seconds += time.perf_counter() - stamp_started
        integrity_bytes += self._provider_guard_payload_bytes()
        if changed:
            self.model.reset_state(release=True)
            raise Qwen38SpeculativeError(
                "rolling draft provider changed target model state"
            )
        if failure is not None:
            self.model.reset_state(release=True)
            raise Qwen38SpeculativeError(
                f"rolling draft provider failed: {type(failure).__name__}: {failure}"
            ) from failure
        assert proposal is not None
        return proposal, proposal_evidence, integrity_bytes, integrity_seconds

    def _begin_rolling_provider(
        self,
        history: tuple[int, ...],
        target_hidden: torch.Tensor,
    ) -> tuple[int, float]:
        missing = object()
        state_callback = (
            inspect.getattr_static(
                self.draft_provider,
                "begin_request_state",
                missing,
            )
            is not missing
        )
        callback_name = "begin_request_state" if state_callback else "begin_request"
        if (
            inspect.getattr_static(self.draft_provider, callback_name, missing)
            is missing
        ):
            return 0, 0.0
        stamp_started = time.perf_counter()
        before = self._provider_state_stamp()
        integrity_seconds = time.perf_counter() - stamp_started
        integrity_bytes = self._provider_guard_payload_bytes()
        failure: Exception | None = None
        provider_hidden = target_hidden.detach().clone() if state_callback else None
        provider_hidden_stamp = (
            _tensor_stamp(provider_hidden) if provider_hidden is not None else None
        )
        try:
            callback = getattr(self.draft_provider, callback_name)
            if not callable(callback):
                raise TypeError(f"draft provider {callback_name} is not callable")
            if state_callback:
                callback(history, provider_hidden)
            else:
                callback(history)
        except Exception as exc:
            failure = exc
        if (
            provider_hidden is not None
            and _tensor_stamp(provider_hidden) != provider_hidden_stamp
        ):
            failure = Qwen38SpeculativeError(
                "stateful rolling initializer mutated its hidden argument"
            )
        stamp_started = time.perf_counter()
        try:
            changed = self._provider_state_stamp() != before
        except Exception:
            changed = True
        integrity_seconds += time.perf_counter() - stamp_started
        integrity_bytes += self._provider_guard_payload_bytes()
        if changed:
            self.model.reset_state(release=True)
            raise Qwen38SpeculativeError(
                "rolling request initializer changed target model state"
            )
        if failure is not None:
            self.model.reset_state(release=True)
            raise Qwen38SpeculativeError(
                f"rolling request initializer failed: {type(failure).__name__}: {failure}"
            ) from failure
        return integrity_bytes, integrity_seconds

    def _reconcile_rolling_provider(
        self,
        history: tuple[int, ...],
        target_hidden: torch.Tensor | None = None,
    ) -> tuple[int, float]:
        missing = object()
        state_callback = (
            target_hidden is not None
            and inspect.getattr_static(
                self.draft_provider,
                "reconcile_prefix_state",
                missing,
            )
            is not missing
        )
        callback_name = (
            "reconcile_prefix_state" if state_callback else "reconcile_prefix"
        )
        if (
            inspect.getattr_static(self.draft_provider, callback_name, missing)
            is missing
        ):
            raise Qwen38SpeculativeError(
                f"rolling generation requires draft_provider.{callback_name}"
            )
        stamp_started = time.perf_counter()
        before = self._provider_state_stamp()
        integrity_seconds = time.perf_counter() - stamp_started
        integrity_bytes = self._provider_guard_payload_bytes()
        failure: Exception | None = None
        provider_hidden = (
            None if target_hidden is None else target_hidden.detach().clone()
        )
        provider_hidden_stamp = (
            None if provider_hidden is None else _tensor_stamp(provider_hidden)
        )
        try:
            callback = getattr(self.draft_provider, callback_name)
            if not callable(callback):
                raise TypeError(f"draft provider {callback_name} is not callable")
            if state_callback:
                callback(history, provider_hidden)
            else:
                callback(history)
        except Exception as exc:
            failure = exc
        if (
            provider_hidden is not None
            and _tensor_stamp(provider_hidden) != provider_hidden_stamp
        ):
            failure = Qwen38SpeculativeError(
                "rolling reconciliation mutated its hidden argument"
            )
        stamp_started = time.perf_counter()
        try:
            changed = self._provider_state_stamp() != before
        except Exception:
            changed = True
        integrity_seconds += time.perf_counter() - stamp_started
        integrity_bytes += self._provider_guard_payload_bytes()
        if changed:
            self.model.reset_state(release=True)
            raise Qwen38SpeculativeError(
                "rolling reconciliation changed target model state"
            )
        if failure is not None:
            self.model.reset_state(release=True)
            raise Qwen38SpeculativeError(
                f"rolling reconciliation failed: {type(failure).__name__}: {failure}"
            ) from failure
        return integrity_bytes, integrity_seconds

    def _observe_rolling_verification_provider(
        self,
        accepted_prefix_length: int,
        verified_proposals: int,
        *,
        virtual: bool = False,
    ) -> tuple[int, float]:
        callback_name = (
            "observe_virtual_verification" if virtual else "observe_verification"
        )
        missing = object()
        if (
            inspect.getattr_static(
                self.draft_provider,
                callback_name,
                missing,
            )
            is missing
        ):
            return 0, 0.0
        stamp_started = time.perf_counter()
        before = self._provider_state_stamp()
        integrity_seconds = time.perf_counter() - stamp_started
        integrity_bytes = self._provider_guard_payload_bytes()
        failure: Exception | None = None
        try:
            callback = getattr(self.draft_provider, callback_name)
            if not callable(callback):
                raise TypeError(f"draft provider {callback_name} is not callable")
            callback(accepted_prefix_length, verified_proposals)
        except Exception as exc:
            failure = exc
        stamp_started = time.perf_counter()
        try:
            changed = self._provider_state_stamp() != before
        except Exception:
            changed = True
        integrity_seconds += time.perf_counter() - stamp_started
        integrity_bytes += self._provider_guard_payload_bytes()
        if changed:
            self.model.reset_state(release=True)
            raise Qwen38SpeculativeError(
                f"rolling {callback_name} observer changed target model state"
            )
        if failure is not None:
            self.model.reset_state(release=True)
            raise Qwen38SpeculativeError(
                f"rolling {callback_name} observer failed: "
                f"{type(failure).__name__}: {failure}"
            ) from failure
        return integrity_bytes, integrity_seconds

    def _observe_rolling_final_provider(
        self,
        history: tuple[int, ...],
    ) -> tuple[int, float]:
        missing = object()
        if (
            inspect.getattr_static(self.draft_provider, "observe_final", missing)
            is missing
        ):
            return 0, 0.0
        stamp_started = time.perf_counter()
        before = self._provider_state_stamp()
        integrity_seconds = time.perf_counter() - stamp_started
        integrity_bytes = self._provider_guard_payload_bytes()
        failure: Exception | None = None
        try:
            callback = getattr(self.draft_provider, "observe_final")
            if not callable(callback):
                raise TypeError("draft provider observe_final is not callable")
            callback(history)
        except Exception as exc:
            failure = exc
        stamp_started = time.perf_counter()
        try:
            changed = self._provider_state_stamp() != before
        except Exception:
            changed = True
        integrity_seconds += time.perf_counter() - stamp_started
        integrity_bytes += self._provider_guard_payload_bytes()
        if changed:
            self.model.reset_state(release=True)
            raise Qwen38SpeculativeError(
                "rolling final observer changed target model state"
            )
        if failure is not None:
            self.model.reset_state(release=True)
            raise Qwen38SpeculativeError(
                f"rolling final observer failed: {type(failure).__name__}: {failure}"
            ) from failure
        return integrity_bytes, integrity_seconds

    def generate_rolling(
        self,
        prompt_token_ids: object,
        *,
        max_new_tokens: int = 1,
        restored_prefix_length: int | None = None,
        restored_seed_hidden: torch.Tensor | None = None,
        eos_token_ids: Iterable[int] = (),
        head_block_rows: int = Qwen38WeightPager.DEFAULT_HEAD_BLOCK_ROWS,
        retain_final_state: bool = True,
        on_tokens: Callable[[tuple[int, ...]], None] | None = None,
    ) -> RollingK4SpeculativeGenerationResult:
        """Generate with one target-known token plus a configurable draft window."""

        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        if (
            isinstance(head_block_rows, bool)
            or not isinstance(head_block_rows, int)
            or head_block_rows <= 0
        ):
            raise ValueError("head_block_rows must be a positive integer")
        if not isinstance(retain_final_state, bool):
            raise TypeError("retain_final_state must be a boolean")
        if on_tokens is not None and not callable(on_tokens):
            raise TypeError("on_tokens must be callable or None")
        prompt_tensor = self.model._token_tensor(prompt_token_ids)
        if tuple(prompt_tensor.shape[:1]) != (1,):
            raise ValueError("rolling generation requires batch size one")
        prompt = tuple(
            int(value) for value in prompt_tensor[0].detach().to(device="cpu").tolist()
        )
        if len(prompt) + max_new_tokens > self.model.max_seq_len:
            raise ValueError("generation would exceed max_seq_len")
        exact_restore = False
        if restored_prefix_length is None:
            if restored_seed_hidden is not None:
                raise ValueError("restored_seed_hidden requires restored_prefix_length")
        else:
            if (
                isinstance(restored_prefix_length, bool)
                or not isinstance(restored_prefix_length, int)
                or not 1 <= restored_prefix_length <= len(prompt)
                or self.model.next_position != restored_prefix_length
                or self.model.state_batch_size != 1
                or self.model.state_poisoned
            ):
                raise ValueError(
                    "rolling restore requires a committed prompt prefix"
                )
            exact_restore = restored_prefix_length == len(prompt)
            if exact_restore:
                if not isinstance(restored_seed_hidden, torch.Tensor):
                    raise TypeError(
                        "an exact rolling restore requires a hidden-state tensor"
                    )
                if (
                    not restored_seed_hidden.is_floating_point()
                    or tuple(restored_seed_hidden.shape)
                    != (1, 1, self.model.config.dim)
                    or restored_seed_hidden.dtype != self.model.pager.compute_dtype
                    or not self.model._on_pager_device(restored_seed_hidden)
                    or not bool(torch.isfinite(restored_seed_hidden).all().item())
                ):
                    raise Qwen38SpeculativeError(
                        "restored exact-prefix seed differs from rolling execution"
                    )
            elif restored_seed_hidden is not None:
                raise ValueError(
                    "a suffix rolling restore may not provide an exact-prefix seed"
                )
        if isinstance(eos_token_ids, (str, bytes)):
            raise TypeError("eos_token_ids must be an iterable of integers")
        eos: set[int] = set()
        try:
            for raw in eos_token_ids:
                if isinstance(raw, bool) or not isinstance(raw, numbers.Integral):
                    raise TypeError("eos_token_ids must contain integers")
                eos.add(int(raw))
        except TypeError as exc:
            if str(exc) == "eos_token_ids must contain integers":
                raise
            raise TypeError("eos_token_ids must be iterable") from exc
        if any(token < 0 or token >= self.model.config.vocab_size for token in eos):
            raise ValueError("EOS token outside checkpoint vocabulary")

        source = self.model.pager.source
        source_start = _owner_metric(source, "network_or_source_body_bytes")
        linears_start = _owner_metric(self.model.pager, "linear_calls")
        started = time.perf_counter()
        if restored_prefix_length is None:
            hidden, prefill_evidence = self.model.prefill(
                [prompt], reset=True, tokenwise=False
            )
        elif exact_restore:
            assert restored_seed_hidden is not None
            hidden = restored_seed_hidden.detach()
            prefill_evidence = ()
        else:
            hidden, prefill_evidence = self.model.prefill(
                [prompt[restored_prefix_length:]],
                reset=False,
                tokenwise=False,
            )
        begin_guard_bytes, begin_guard_seconds = self._begin_rolling_provider(
            prompt,
            hidden,
        )
        seed = self._scan(hidden[:, -1:], block_rows=head_block_rows)
        if len(seed) != 1:
            raise Qwen38SpeculativeError("rolling seed scan returned wrong width")
        pending_token = seed[0]
        generated: list[int] = []
        rounds: list[RollingK4SpeculativeRoundEvidence] = []
        stopped = False

        while len(generated) < max_new_tokens and not stopped:
            remaining = max_new_tokens - len(generated)
            round_index = len(rounds)
            start_pos = self.model.next_position
            if pending_token in eos or remaining == 1:
                round_source = _owner_metric(source, "network_or_source_body_bytes")
                round_linears = _owner_metric(self.model.pager, "linear_calls")
                round_started = time.perf_counter()
                commit = retain_final_state
                if commit:
                    hidden, _tail_evidence = self.model.decode([[pending_token]])
                stopped = pending_token in eos
                emitted = (pending_token,)
                final_guard_bytes, final_guard_seconds = (
                    self._observe_rolling_final_provider(
                        (*prompt, *generated, *emitted)
                    )
                )
                rounds.append(
                    RollingK4SpeculativeRoundEvidence(
                        round_index=round_index,
                        start_pos=start_pos,
                        end_pos=start_pos + 1,
                        known_token_id=pending_token,
                        proposed_token_ids=(),
                        target_token_ids=(),
                        emitted_token_ids=emitted,
                        accepted_prefix_length=0,
                        correction_token_id=None,
                        forward_passes=int(commit),
                        head_scans=0,
                        source_body_bytes=(
                            _owner_metric(source, "network_or_source_body_bytes")
                            - round_source
                        ),
                        linear_calls=(
                            _owner_metric(self.model.pager, "linear_calls")
                            - round_linears
                        ),
                        provider_guard_bytes=(
                            final_guard_bytes
                            + (begin_guard_bytes if round_index == 0 else 0)
                        ),
                        provider_guard_seconds=(
                            final_guard_seconds
                            + (begin_guard_seconds if round_index == 0 else 0.0)
                        ),
                        seconds=time.perf_counter() - round_started,
                        state_bytes=self.model.state_bytes,
                        state_committed=commit,
                        stopped_on_eos=stopped,
                        window_size=self.window_size,
                        request_window_ceiling=self.window_size,
                    )
                )
                generated.extend(emitted)
                if on_tokens is not None:
                    on_tokens(tuple(generated))
                continue

            round_source = _owner_metric(source, "network_or_source_body_bytes")
            round_linears = _owner_metric(self.model.pager, "linear_calls")
            round_started = time.perf_counter()
            provider_proposal, proposal_evidence, guard_bytes, guard_seconds = (
                self._rolling_proposal(
                    (*prompt, *generated),
                    pending_token,
                    hidden[:, -1:],
                )
            )
            if self.adaptive_round_windows:
                if proposal_evidence is None:
                    raise Qwen38SpeculativeError(
                        "adaptive rolling proposal evidence is missing"
                    )
                round_costs = self.round_window_work_costs
                if proposal_evidence.provider_abi.startswith(
                    "immer.qwen3.8-markov-draft-provider/"
                ):
                    # Markov owns no model/head work.  Its proposal horizons
                    # already price the target-only packed-row execution.
                    round_costs = None
                round_policy = proposal_evidence.select_window(
                    request_window_ceiling=self.window_size,
                    remaining_tokens=remaining,
                    window_work_costs=round_costs,
                )
                active_window = round_policy.chosen_window
                proposal = provider_proposal[: active_window - 1]
            else:
                round_policy = None
                active_window = self.window_size
                proposal = provider_proposal
            if round_index == 0:
                guard_bytes += begin_guard_bytes
                guard_seconds += begin_guard_seconds
            if active_window == 1:
                # A low-confidence Markov action is true abstention, not a
                # one-row transactional speculative stage.  Consume the known
                # token directly so K1 does not clone the complete 157 MB
                # continuation state merely to commit its only row.
                hidden, _direct_evidence = self.model.decode([[pending_token]])
                targets = self._scan(hidden[:, -1:], block_rows=head_block_rows)
                if len(targets) != 1:
                    raise Qwen38SpeculativeError(
                        "direct Markov target scan returned wrong width"
                    )
                virtual_accepted = int(
                    bool(provider_proposal) and provider_proposal[0] == targets[0]
                )
                verification_bytes, verification_seconds = (
                    self._observe_rolling_verification_provider(
                        virtual_accepted,
                        int(bool(provider_proposal)),
                        virtual=True,
                    )
                )
                guard_bytes += verification_bytes
                guard_seconds += verification_seconds
                emitted = (pending_token,)
                reconcile_bytes, reconcile_seconds = self._reconcile_rolling_provider(
                    (*prompt, *generated, *emitted),
                    hidden,
                )
                guard_bytes += reconcile_bytes
                guard_seconds += reconcile_seconds
                rounds.append(
                    RollingK4SpeculativeRoundEvidence(
                        round_index=round_index,
                        start_pos=start_pos,
                        end_pos=self.model.next_position,
                        known_token_id=pending_token,
                        proposed_token_ids=(),
                        target_token_ids=targets,
                        emitted_token_ids=emitted,
                        accepted_prefix_length=0,
                        correction_token_id=targets[0],
                        forward_passes=1,
                        head_scans=1,
                        source_body_bytes=(
                            _owner_metric(source, "network_or_source_body_bytes")
                            - round_source
                        ),
                        linear_calls=(
                            _owner_metric(self.model.pager, "linear_calls")
                            - round_linears
                        ),
                        provider_guard_bytes=guard_bytes,
                        provider_guard_seconds=guard_seconds,
                        seconds=time.perf_counter() - round_started,
                        state_bytes=self.model.state_bytes,
                        state_committed=True,
                        stopped_on_eos=False,
                        window_size=1,
                        request_window_ceiling=self.window_size,
                        provider_proposed_token_ids=provider_proposal,
                        round_policy=round_policy,
                    )
                )
                generated.extend(emitted)
                if on_tokens is not None:
                    on_tokens(tuple(generated))
                pending_token = targets[0]
                continue
            stage = self.model.stage_continuation_block([[pending_token, *proposal]])
            try:
                targets = self._scan(stage.hidden, block_rows=head_block_rows)
                if len(targets) != active_window:
                    raise Qwen38SpeculativeError(
                        "rolling target scan returned wrong width"
                    )
            except Exception:
                try:
                    self.model.discard_continuation_block(stage)
                except Exception:
                    pass
                raise
            accepted = 0
            while (
                accepted < active_window - 1 and proposal[accepted] == targets[accepted]
            ):
                accepted += 1
            matched_prefix_length = accepted
            target_verified_proposals = (
                accepted
                if accepted == len(proposal)
                else min(len(proposal), accepted + 1)
            )
            accepted = min(accepted, remaining - 1)
            eos_offset = next(
                (
                    index
                    for index, token in enumerate(proposal[:accepted])
                    if token in eos
                ),
                None,
            )
            if eos_offset is not None:
                accepted = eos_offset + 1
            verified_proposals = (
                accepted
                if accepted < matched_prefix_length
                else target_verified_proposals
            )
            verification_bytes, verification_seconds = (
                self._observe_rolling_verification_provider(
                    accepted,
                    verified_proposals,
                )
            )
            guard_bytes += verification_bytes
            guard_seconds += verification_seconds
            emitted = (pending_token, *proposal[:accepted])
            stopped = eos_offset is not None
            terminal = stopped or len(emitted) == remaining
            state_committed = retain_final_state or not terminal
            if state_committed:
                width = len(emitted)
                if width == active_window:
                    hidden, _state_evidence = self.model.commit_continuation_block(
                        stage
                    )
                else:
                    hidden, _state_evidence = self.model.commit_continuation_prefix(
                        stage,
                        width,
                    )
            else:
                self.model.discard_continuation_block(stage)
            reconcile_bytes, reconcile_seconds = self._reconcile_rolling_provider(
                (*prompt, *generated, *emitted),
                hidden if state_committed else None,
            )
            guard_bytes += reconcile_bytes
            guard_seconds += reconcile_seconds
            if terminal:
                final_bytes, final_seconds = self._observe_rolling_final_provider(
                    (*prompt, *generated, *emitted)
                )
                guard_bytes += final_bytes
                guard_seconds += final_seconds
            correction = None if terminal else targets[accepted]
            rounds.append(
                RollingK4SpeculativeRoundEvidence(
                    round_index=round_index,
                    start_pos=start_pos,
                    end_pos=start_pos + len(emitted),
                    known_token_id=pending_token,
                    proposed_token_ids=proposal,
                    target_token_ids=targets,
                    emitted_token_ids=emitted,
                    accepted_prefix_length=accepted,
                    correction_token_id=correction,
                    forward_passes=1,
                    head_scans=1,
                    source_body_bytes=(
                        _owner_metric(source, "network_or_source_body_bytes")
                        - round_source
                    ),
                    linear_calls=(
                        _owner_metric(self.model.pager, "linear_calls") - round_linears
                    ),
                    provider_guard_bytes=guard_bytes,
                    provider_guard_seconds=guard_seconds,
                    seconds=time.perf_counter() - round_started,
                    state_bytes=self.model.state_bytes,
                    state_committed=state_committed,
                    stopped_on_eos=stopped,
                    window_size=active_window,
                    request_window_ceiling=self.window_size,
                    provider_proposed_token_ids=provider_proposal,
                    round_policy=round_policy,
                )
            )
            generated.extend(emitted)
            if on_tokens is not None:
                on_tokens(tuple(generated))
            if correction is not None:
                pending_token = correction

        evidence = RollingK4SpeculativeGenerationEvidence(
            schema=(
                QWEN38_ROLLING_K4_SPECULATIVE_SCHEMA
                if self.window_size == 4 and not self.adaptive_round_windows
                else QWEN38_ROLLING_SPECULATIVE_SCHEMA
            ),
            prompt_token_ids=prompt,
            generated_token_ids=tuple(generated),
            rounds=tuple(rounds),
            prefill_forward_passes=len(prefill_evidence),
            forward_passes=len(prefill_evidence)
            + sum(row.forward_passes for row in rounds),
            head_scans=1 + sum(row.head_scans for row in rounds),
            accepted_draft_tokens=sum(row.accepted_prefix_length for row in rounds),
            source_body_bytes=(
                _owner_metric(source, "network_or_source_body_bytes") - source_start
            ),
            linear_calls=(
                _owner_metric(self.model.pager, "linear_calls") - linears_start
            ),
            provider_guard_bytes=sum(row.provider_guard_bytes for row in rounds),
            provider_guard_seconds=sum(row.provider_guard_seconds for row in rounds),
            seconds=time.perf_counter() - started,
            state_bytes=self.model.state_bytes,
            stopped_on_eos=stopped,
            window_size=self.window_size,
            adaptive_windows=self.adaptive_round_windows,
        )
        return RollingK4SpeculativeGenerationResult(tuple(generated), evidence)

    def _single_round_k4(
        self,
        hidden: torch.Tensor,
        *,
        round_index: int,
        eos: frozenset[int],
        eos_ids: tuple[int, ...],
        block_rows: int,
        continuation_required: bool,
        retain_final_state: bool,
    ) -> tuple[torch.Tensor, K4SpeculativeRoundEvidence]:
        start_pos = self.model.next_position
        source_start = _owner_metric(
            self.model.pager.source, "network_or_source_body_bytes"
        )
        linears_start = _owner_metric(self.model.pager, "linear_calls")
        started = time.perf_counter()
        values, selected = self.model.pager.topk_logits(
            hidden[:, -1],
            k=1,
            name=self.model.output_head_name,
            block_rows=block_rows,
        )
        if tuple(selected.shape) != (1, 1):
            raise Qwen38SpeculativeError(
                "LM head returned an invalid single target shape"
            )
        target = int(selected[0, 0].item())
        del values, selected
        stopped = target in eos
        commit_state = retain_final_state or (continuation_required and not stopped)
        if commit_state:
            next_hidden, _state_evidence = self.model.decode([[target]])
            replay_kind: K4ReplayKind = "terminal-single"
            passes = 1
            end_pos = self.model.next_position
        else:
            next_hidden = hidden
            replay_kind = "terminal-single-uncommitted"
            passes = 0
            end_pos = start_pos + 1
        seconds = time.perf_counter() - started
        evidence = _sealed_k4_round(
            schema=QWEN38_K4_SPECULATIVE_ROUND_SCHEMA,
            round_index=round_index,
            start_pos=start_pos,
            end_pos=end_pos,
            eos_token_ids=eos_ids,
            proposed_token_ids=(),
            target_token_ids=(target,),
            emitted_token_ids=(target,),
            accepted_prefix_length=0,
            replay_kind=replay_kind,
            forward_passes=passes,
            head_scans=1,
            source_body_bytes=(
                _owner_metric(self.model.pager.source, "network_or_source_body_bytes")
                - source_start
            ),
            linear_calls=_owner_metric(self.model.pager, "linear_calls")
            - linears_start,
            provider_guard_bytes=0,
            provider_guard_seconds=0.0,
            seconds=seconds,
            state_bytes=self.model.state_bytes,
            stopped_on_eos=stopped,
        )
        return next_hidden, evidence

    def generate(
        self,
        prompt_token_ids: object,
        *,
        max_new_tokens: int = 1,
        eos_token_ids: Iterable[int] = (),
        head_block_rows: int = Qwen38WeightPager.DEFAULT_HEAD_BLOCK_ROWS,
        retain_final_state: bool = True,
        on_tokens: Callable[[tuple[int, ...]], None] | None = None,
    ) -> K4SpeculativeGenerationResult:
        """Generate exact greedy K=4 continuation and commit every output token."""

        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        if (
            isinstance(head_block_rows, bool)
            or not isinstance(head_block_rows, int)
            or head_block_rows <= 0
        ):
            raise ValueError("head_block_rows must be a positive integer")
        if not isinstance(retain_final_state, bool):
            raise TypeError("retain_final_state must be a boolean")
        if on_tokens is not None and not callable(on_tokens):
            raise TypeError("on_tokens must be callable or None")
        prompt_tensor = self.model._token_tensor(prompt_token_ids)
        if prompt_tensor.shape[0] != 1:
            raise ValueError("speculative generation requires batch size one")
        prompt = tuple(
            int(value) for value in prompt_tensor[0].detach().to(device="cpu").tolist()
        )
        if len(prompt) + max_new_tokens > self.model.max_seq_len:
            raise ValueError("generation would exceed max_seq_len")
        if isinstance(eos_token_ids, (str, bytes)):
            raise TypeError("eos_token_ids must be an iterable of integers")
        eos_values: set[int] = set()
        try:
            for raw in eos_token_ids:
                if isinstance(raw, bool) or not isinstance(raw, numbers.Integral):
                    raise TypeError("eos_token_ids must contain integers")
                eos_values.add(int(raw))
        except TypeError as exc:
            if str(exc) == "eos_token_ids must contain integers":
                raise
            raise TypeError("eos_token_ids must be iterable") from exc
        if any(
            token < 0 or token >= self.model.config.vocab_size for token in eos_values
        ):
            raise ValueError("EOS token outside checkpoint vocabulary")
        eos = frozenset(eos_values)
        eos_ids = tuple(sorted(eos))

        source_start = _owner_metric(
            self.model.pager.source, "network_or_source_body_bytes"
        )
        linears_start = _owner_metric(self.model.pager, "linear_calls")
        started = time.perf_counter()
        hidden, prefill_evidence = self.model.prefill(
            [prompt], reset=True, tokenwise=False
        )
        generated: list[int] = []
        rounds: list[K4SpeculativeRoundEvidence] = []
        stopped = False

        while len(generated) < max_new_tokens and not stopped:
            remaining = max_new_tokens - len(generated)
            round_index = len(rounds)
            if remaining < 4:
                hidden, row = self._single_round_k4(
                    hidden,
                    round_index=round_index,
                    eos=eos,
                    eos_ids=eos_ids,
                    block_rows=head_block_rows,
                    continuation_required=remaining > 1,
                    retain_final_state=retain_final_state,
                )
                generated.extend(row.emitted_token_ids)
                rounds.append(row)
                stopped = row.stopped_on_eos
                if on_tokens is not None:
                    on_tokens(tuple(generated))
                continue

            round_source_start = _owner_metric(
                self.model.pager.source, "network_or_source_body_bytes"
            )
            round_linears_start = _owner_metric(self.model.pager, "linear_calls")
            round_started = time.perf_counter()
            proposal, integrity_bytes, integrity_seconds = self._proposal_k4(
                (*prompt, *generated)
            )
            stage = self.model.stage_continuation_block([proposal])
            try:
                verification_hidden = torch.cat(
                    (hidden[:, -1:], stage.hidden[:, :3]), dim=1
                )
                targets = self._scan(
                    verification_hidden,
                    block_rows=head_block_rows,
                )
                if len(targets) != 4:  # pragma: no cover - guarded by _scan.
                    raise Qwen38SpeculativeError(
                        "K=4 verification returned wrong width"
                    )
            except Exception:
                try:
                    self.model.discard_continuation_block(stage)
                except Exception:
                    pass
                raise

            common = 0
            while common < 4 and proposal[common] == targets[common]:
                common += 1
            valid_width = 4 if common == 4 else common + 1
            eos_index = next(
                (
                    index
                    for index, token in enumerate(targets[:valid_width])
                    if token in eos
                ),
                None,
            )

            state_committed = True
            reconcile = True
            if eos_index is not None and not retain_final_state:
                emitted = targets[: eos_index + 1]
                accepted = min(common, eos_index + 1)
                self.model.discard_continuation_block(stage)
                replay_kind = "eos-uncommitted"
                passes = 1
                state_committed = False
                reconcile = False
            elif eos_index is not None:
                emitted = targets[: eos_index + 1]
                accepted = min(common, eos_index + 1)
                if eos_index == 3 and common == 4:
                    hidden, _state_evidence = self.model.commit_continuation_block(
                        stage
                    )
                    replay_kind: K4ReplayKind = "eos3-commit-k4"
                    passes = 1
                else:
                    self.model.discard_continuation_block(stage)
                    if eos_index == 0:
                        hidden, _state_evidence = self.model.decode([[emitted[0]]])
                        replay_kind = "eos0-decode"
                    else:
                        replay = self.model.stage_continuation_block([emitted])
                        hidden, _state_evidence = self.model.commit_continuation_block(
                            replay
                        )
                        eos_replays: dict[int, K4ReplayKind] = {
                            1: "eos1-restage",
                            2: "eos2-restage",
                            3: "eos3-restage",
                        }
                        replay_kind = eos_replays[eos_index]
                    passes = 2
            elif common == 4:
                hidden, _state_evidence = self.model.commit_continuation_block(stage)
                emitted = proposal
                accepted = 4
                replay_kind = "commit-k4"
                passes = 1
            elif common == 3 and remaining == 4 and not retain_final_state:
                emitted = targets
                accepted = common
                self.model.discard_continuation_block(stage)
                replay_kind = "mismatch3-uncommitted"
                passes = 1
                state_committed = False
                reconcile = False
            else:
                emitted = targets[: common + 1]
                accepted = common
                self.model.discard_continuation_block(stage)
                if common == 0:
                    hidden, _state_evidence = self.model.decode([[emitted[0]]])
                    replay_kind = "mismatch0-decode"
                else:
                    replay = self.model.stage_continuation_block([emitted])
                    hidden, _state_evidence = self.model.commit_continuation_block(
                        replay
                    )
                    mismatch_replays: dict[int, K4ReplayKind] = {
                        1: "mismatch1-restage",
                        2: "mismatch2-restage",
                        3: "mismatch3-restage",
                    }
                    replay_kind = mismatch_replays[common]
                passes = 2

            stopped = eos_index is not None
            if reconcile:
                reconcile_bytes, reconcile_seconds = self._reconcile_provider(
                    (*prompt, *generated, *emitted)
                )
                integrity_bytes += reconcile_bytes
                integrity_seconds += reconcile_seconds
            seconds = time.perf_counter() - round_started
            row = _sealed_k4_round(
                schema=QWEN38_K4_SPECULATIVE_ROUND_SCHEMA,
                round_index=round_index,
                start_pos=stage.evidence.start_pos,
                end_pos=(
                    self.model.next_position
                    if state_committed
                    else stage.evidence.start_pos + len(emitted)
                ),
                eos_token_ids=eos_ids,
                proposed_token_ids=proposal,
                target_token_ids=targets,
                emitted_token_ids=emitted,
                accepted_prefix_length=accepted,
                replay_kind=replay_kind,
                forward_passes=passes,
                head_scans=1,
                source_body_bytes=(
                    _owner_metric(
                        self.model.pager.source, "network_or_source_body_bytes"
                    )
                    - round_source_start
                ),
                linear_calls=_owner_metric(self.model.pager, "linear_calls")
                - round_linears_start,
                provider_guard_bytes=integrity_bytes,
                provider_guard_seconds=integrity_seconds,
                seconds=seconds,
                state_bytes=self.model.state_bytes,
                stopped_on_eos=stopped,
            )
            generated.extend(emitted)
            rounds.append(row)
            if on_tokens is not None:
                on_tokens(tuple(generated))

        generation_seconds = time.perf_counter() - started
        evidence = _sealed_k4_generation(
            schema=QWEN38_K4_SPECULATIVE_SCHEMA,
            prompt_token_ids=prompt,
            eos_token_ids=eos_ids,
            generated_token_ids=tuple(generated),
            rounds=tuple(rounds),
            prefill_forward_passes=len(prefill_evidence),
            forward_passes=len(prefill_evidence)
            + sum(row.forward_passes for row in rounds),
            head_scans=sum(row.head_scans for row in rounds),
            accepted_draft_tokens=sum(row.accepted_prefix_length for row in rounds),
            source_body_bytes=(
                _owner_metric(self.model.pager.source, "network_or_source_body_bytes")
                - source_start
            ),
            linear_calls=_owner_metric(self.model.pager, "linear_calls")
            - linears_start,
            provider_guard_bytes=sum(row.provider_guard_bytes for row in rounds),
            provider_guard_seconds=sum(row.provider_guard_seconds for row in rounds),
            seconds=generation_seconds,
            state_bytes=self.model.state_bytes,
            stopped_on_eos=stopped,
        )
        return K4SpeculativeGenerationResult(tuple(generated), evidence)


__all__ = [
    "DraftProvider",
    "K4DraftProvider",
    "K4ReconciledDraftProvider",
    "RollingK4DraftProvider",
    "K2SpeculativeGenerationEvidence",
    "K2SpeculativeGenerationResult",
    "K2SpeculativeRoundEvidence",
    "K4ReplayKind",
    "K4SpeculativeGenerationEvidence",
    "K4SpeculativeGenerationResult",
    "K4SpeculativeRoundEvidence",
    "RollingK4SpeculativeGenerationEvidence",
    "RollingK4SpeculativeGenerationResult",
    "RollingK4SpeculativeRoundEvidence",
    "QWEN38_K2_SPECULATIVE_SCHEMA",
    "QWEN38_K4_SPECULATIVE_ROUND_SCHEMA",
    "QWEN38_K4_SPECULATIVE_SCHEMA",
    "QWEN38_ROLLING_K4_SPECULATIVE_SCHEMA",
    "QWEN38_ROLLING_SPECULATIVE_SCHEMA",
    "Qwen38K2SpeculativeDecoder",
    "Qwen38K4SpeculativeDecoder",
    "Qwen38SpeculativeError",
    "ReconciledDraftProvider",
]
