"""Exact greedy K=2 speculative generation for streamed Qwen3.8.

The draft provider is deliberately untrusted: it may only propose two token
IDs.  The official target model verifies both positions with one complete
vocabulary scan and publishes only target-confirmed continuation state.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
import hashlib
import inspect
import math
import numbers
import time
from typing import Any, Literal, Protocol

import torch

from .model import StreamedQwen38
from .pager import Qwen38WeightPager


QWEN38_K2_SPECULATIVE_SCHEMA = "immer.qwen3.8-k2-speculative-generation/v1"

ReplayKind = Literal[
    "commit-k2",
    "eos0-decode",
    "mismatch0-decode",
    "mismatch1-restage",
    "remaining-single",
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


__all__ = [
    "DraftProvider",
    "K2SpeculativeGenerationEvidence",
    "K2SpeculativeGenerationResult",
    "K2SpeculativeRoundEvidence",
    "QWEN38_K2_SPECULATIVE_SCHEMA",
    "Qwen38K2SpeculativeDecoder",
    "Qwen38SpeculativeError",
    "ReconciledDraftProvider",
]
