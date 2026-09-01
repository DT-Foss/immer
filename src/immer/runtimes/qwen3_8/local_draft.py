"""Transactional Qwen3.5 K=2 drafter for the streamed Qwen target runtime.

The small model owns an independent continuation state.  It stages every
proposal until the target reports the committed history through
``reconcile``; rejected suffixes never leak into the next draft round.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
import hashlib
import numbers
from typing import Any

import torch

from .model import StatefulBlockStage, StreamedQwen38
from .pager import Qwen38WeightPager


QWEN35_K2_DRAFT_PROVIDER_SCHEMA = "immer.qwen3.5-k2-draft-provider/v1"
QWEN35_K4_DRAFT_PROVIDER_SCHEMA = "immer.qwen3.5-k4-draft-provider/v1"
QWEN35_ROLLING_DRAFT_PROVIDER_SCHEMA = "immer.qwen3.5-rolling-draft-provider/v2"


class Qwen35K2DraftProviderError(RuntimeError):
    """The local drafter can no longer prove exact state/history alignment."""


class Qwen35K4DraftProviderError(RuntimeError):
    """The K=4 drafter can no longer prove exact state/history alignment."""


def _metric(owner: object, name: str) -> int:
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


def _layer_state_stamp(states: Iterable[object | None]) -> tuple[object, ...]:
    rows: list[object] = []
    for state in states:
        if state is None:
            rows.append(None)
            continue
        rows.append(
            (
                type(state).__qualname__,
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
    return tuple(rows)


def _tensor_identity_stamp(value: torch.Tensor | None) -> object:
    """Track an owned tensor without copying its body through the CPU."""

    if value is None:
        return None
    return (
        id(value),
        int(value._version),
        tuple(value.shape),
        str(value.dtype),
        str(value.device),
        int(value.data_ptr()),
    )


def _layer_state_identity_stamp(
    states: Iterable[object | None],
) -> tuple[object, ...]:
    rows: list[object] = []
    for state in states:
        if state is None:
            rows.append(None)
            continue
        rows.append(
            (
                id(state),
                type(state).__qualname__,
                *(
                    _tensor_identity_stamp(getattr(state, name, None))
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
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class Qwen35K2DraftProviderMetrics:
    """Cumulative local-draft work and reconciliation outcomes."""

    schema: str
    prefill_calls: int
    draft_calls: int
    reconcile_calls: int
    accepted_prefix_0: int
    accepted_prefix_1: int
    accepted_prefix_2: int
    restaged_pairs: int
    committed_tokens: int
    source_body_bytes: int
    linear_calls: int
    state_bytes: int
    pending: bool
    poisoned: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Qwen35K2DraftProvider:
    """Use an independent local Qwen3.5 model as an exact recurrent K=2 drafter.

    ``__call__`` receives the target's immutable committed history and returns
    exactly two greedy draft tokens.  The pair remains transactional until the
    target decoder calls :meth:`reconcile` with the history it actually
    committed.  Accepted pairs commit directly, one-token acceptance restages
    the verified token plus the target correction, and zero-token acceptance
    decodes only the target correction.
    """

    def __init__(
        self,
        model: StreamedQwen38,
        *,
        eos_token_ids: Iterable[int] = (),
        head_block_rows: int = Qwen38WeightPager.DEFAULT_HEAD_BLOCK_ROWS,
    ) -> None:
        if not isinstance(model, StreamedQwen38):
            raise TypeError("model must be a StreamedQwen38")
        if (
            isinstance(head_block_rows, bool)
            or not isinstance(head_block_rows, int)
            or head_block_rows <= 0
        ):
            raise ValueError("head_block_rows must be a positive integer")
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
        if any(token < 0 or token >= model.config.vocab_size for token in eos):
            raise ValueError("EOS token outside draft checkpoint vocabulary")
        if (
            model.next_position != 0
            or model.state_batch_size is not None
            or model.state_bytes != 0
            or model.state_poisoned
            or model._pending_block_stage is not None
        ):
            raise ValueError("draft model must start with empty clean state")

        self.model = model
        self.eos_token_ids = frozenset(eos)
        self.head_block_rows = head_block_rows
        self._committed_history: tuple[int, ...] | None = None
        self._last_hidden: torch.Tensor | None = None
        self._pending_base: tuple[int, ...] | None = None
        self._pending_proposal: tuple[int, int] | None = None
        self._pending_stage: StatefulBlockStage | None = None
        self._poisoned = False
        self._closed = False
        self._prefill_calls = 0
        self._draft_calls = 0
        self._reconcile_calls = 0
        self._accepted = [0, 0, 0]
        self._restaged_pairs = 0
        self._committed_tokens = 0
        self._source_start = _metric(model.pager.source, "network_or_source_body_bytes")
        self._linears_start = _metric(model.pager, "linear_calls")
        self._seal = self._runtime_stamp()

    @property
    def committed_history(self) -> tuple[int, ...] | None:
        return self._committed_history

    @property
    def pending_proposal(self) -> tuple[int, int] | None:
        return self._pending_proposal

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def closed(self) -> bool:
        return self._closed

    def metrics(self) -> Qwen35K2DraftProviderMetrics:
        return Qwen35K2DraftProviderMetrics(
            schema=QWEN35_K2_DRAFT_PROVIDER_SCHEMA,
            prefill_calls=self._prefill_calls,
            draft_calls=self._draft_calls,
            reconcile_calls=self._reconcile_calls,
            accepted_prefix_0=self._accepted[0],
            accepted_prefix_1=self._accepted[1],
            accepted_prefix_2=self._accepted[2],
            restaged_pairs=self._restaged_pairs,
            committed_tokens=self._committed_tokens,
            source_body_bytes=(
                _metric(self.model.pager.source, "network_or_source_body_bytes")
                - self._source_start
            ),
            linear_calls=_metric(self.model.pager, "linear_calls")
            - self._linears_start,
            state_bytes=self.model.state_bytes,
            pending=self._pending_stage is not None,
            poisoned=self._poisoned,
        )

    def _pending_stamp(self) -> object:
        pending = self.model._pending_block_stage
        if pending is None:
            return None
        return (
            id(pending),
            id(pending.handle),
            id(pending.handle._nonce),
            asdict(pending.evidence),
            pending.runtime_identity,
            _tensor_stamp(pending.handle.hidden),
            _tensor_stamp(pending.hidden),
            _layer_state_stamp(pending.layer_states),
            _tensor_stamp(pending.graft_history),
        )

    def _runtime_stamp(self) -> tuple[object, ...]:
        return (
            self.model.next_position,
            self.model.state_batch_size,
            self.model.state_poisoned,
            self.model._continuation_block_runtime_identity(),
            _layer_state_stamp(self.model._layer_states),
            _tensor_stamp(self.model._graft_history),
            self._pending_stamp(),
            self._committed_history,
            _tensor_stamp(self._last_hidden),
            self._pending_base,
            self._pending_proposal,
            id(self._pending_stage),
        )

    def _abort(self, message: str, cause: Exception | None = None) -> None:
        try:
            self.model.reset_state(release=True)
        finally:
            self._committed_history = None
            self._last_hidden = None
            self._pending_base = None
            self._pending_proposal = None
            self._pending_stage = None
            self._poisoned = True
            self._seal = self._runtime_stamp()
        error = Qwen35K2DraftProviderError(message)
        if cause is None:
            raise error
        raise error from cause

    def _assert_ready(self) -> None:
        if self._closed:
            raise Qwen35K2DraftProviderError("draft provider is closed")
        if self._poisoned:
            raise Qwen35K2DraftProviderError("draft provider is poisoned; call reset()")
        try:
            changed = self._runtime_stamp() != self._seal
        except Exception as exc:
            self._abort("draft model state cannot be verified", exc)
        if changed:
            self._abort("draft model cursor or continuation state drifted")

    def _history(self, value: object, *, name: str) -> tuple[int, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"{name} must be an integer sequence")
        result: list[int] = []
        for raw in value:
            if isinstance(raw, bool) or not isinstance(raw, numbers.Integral):
                raise TypeError(f"{name} must contain integers")
            token = int(raw)
            if token < 0 or token >= self.model.config.vocab_size:
                raise ValueError(f"{name} token outside draft checkpoint vocabulary")
            result.append(token)
        if not result:
            raise ValueError(f"{name} must not be empty")
        return tuple(result)

    def _scan(self, hidden: torch.Tensor) -> int:
        values, selected = self.model.pager.topk_logits(
            hidden,
            k=1,
            name=self.model.output_head_name,
            block_rows=self.head_block_rows,
        )
        if tuple(selected.shape) != (1, 1):
            raise Qwen35K2DraftProviderError(
                "draft LM head returned an invalid token shape"
            )
        token = int(selected[0, 0].item())
        del values, selected
        if not 0 <= token < self.model.config.vocab_size:
            raise Qwen35K2DraftProviderError(
                "draft LM head selected a token outside its vocabulary"
            )
        return token

    def __call__(self, history: tuple[int, ...], /) -> tuple[int, int]:
        self._assert_ready()
        try:
            committed = self._history(history, name="draft history")
            if self._pending_stage is not None:
                self._abort("previous draft proposal was not reconciled before reuse")
            if self._committed_history is None:
                if len(committed) + 2 > self.model.max_seq_len:
                    raise ValueError("draft pair would exceed max_seq_len")
                hidden, _evidence = self.model.prefill([committed], reset=True)
                self._last_hidden = hidden[:, -1:].detach().clone()
                self._committed_history = committed
                self._prefill_calls += 1
            elif committed != self._committed_history:
                self._abort("draft history differs from committed local history")
            if self.model.next_position != len(committed):
                self._abort("draft model cursor disagrees with committed history")
            if len(committed) + 2 > self.model.max_seq_len:
                raise ValueError("draft pair would exceed max_seq_len")
            if committed[-1] in self.eos_token_ids:
                raise ValueError("cannot draft after a committed EOS token")
            if self._last_hidden is None:
                self._abort("draft model has no hidden state for its committed cursor")

            token0 = self._scan(self._last_hidden[:, -1])
            if token0 in self.eos_token_ids:
                token1 = token0
            else:
                probe = self.model.stage_continuation_block([[token0, token0]])
                try:
                    token1 = self._scan(probe.hidden[:, 0])
                finally:
                    self.model.discard_continuation_block(probe)

            proposal = (token0, token1)
            stage = self.model.stage_continuation_block([proposal])
            self._pending_base = committed
            self._pending_proposal = proposal
            self._pending_stage = stage
            self._draft_calls += 1
            self._seal = self._runtime_stamp()
            return proposal
        except Qwen35K2DraftProviderError as exc:
            if not self._poisoned:
                self._abort(str(exc), exc)
            raise
        except Exception as exc:
            self._abort(f"local draft failed: {type(exc).__name__}: {exc}", exc)

    def reconcile(self, history: tuple[int, ...], /) -> None:
        """Publish only the target-confirmed prefix and correction tokens."""

        self._assert_ready()
        try:
            committed = self._history(history, name="reconciled history")
            base = self._pending_base
            proposal = self._pending_proposal
            stage = self._pending_stage
            if base is None or proposal is None or stage is None:
                self._abort("reconcile requires one pending draft proposal")
            if committed[: len(base)] != base:
                self._abort("reconciled history changed the committed draft prefix")
            delta = committed[len(base) :]
            if len(delta) not in (1, 2):
                self._abort("reconciled history must commit one or two target tokens")
            if len(delta) == 2 and delta[0] in self.eos_token_ids:
                self._abort("reconciled history contains a token after EOS")
            if self.model.next_position != len(base):
                self._abort("draft model cursor changed under a pending proposal")

            if len(delta) == 2 and delta == proposal:
                hidden, _evidence = self.model.commit_continuation_block(stage)
                accepted = 2
            elif len(delta) == 2:
                if delta[0] != proposal[0]:
                    self._abort(
                        "two-token reconciliation cannot reject the first proposal"
                    )
                self.model.discard_continuation_block(stage)
                replay = self.model.stage_continuation_block([delta])
                hidden, _evidence = self.model.commit_continuation_block(replay)
                accepted = 1
                self._restaged_pairs += 1
            else:
                self.model.discard_continuation_block(stage)
                hidden, _evidence = self.model.decode([[delta[0]]])
                accepted = int(delta[0] == proposal[0])

            self._pending_base = None
            self._pending_proposal = None
            self._pending_stage = None
            self._committed_history = committed
            self._last_hidden = hidden[:, -1:].detach().clone()
            if (
                self.model.next_position != len(committed)
                or self.model._pending_block_stage is not None
            ):
                self._abort("reconciled draft state has the wrong committed cursor")
            self._reconcile_calls += 1
            self._accepted[accepted] += 1
            self._committed_tokens += len(delta)
            self._seal = self._runtime_stamp()
        except Qwen35K2DraftProviderError as exc:
            if not self._poisoned:
                self._abort(str(exc), exc)
            raise
        except Exception as exc:
            self._abort(
                f"local draft reconciliation failed: {type(exc).__name__}: {exc}",
                exc,
            )

    def reset(self) -> None:
        """Recover explicitly after a failed request or start a new request."""

        if self._closed:
            raise Qwen35K2DraftProviderError("draft provider is closed")
        self.model.reset_state(release=True)
        self._committed_history = None
        self._last_hidden = None
        self._pending_base = None
        self._pending_proposal = None
        self._pending_stage = None
        self._poisoned = False
        self._seal = self._runtime_stamp()

    def close(self) -> None:
        """Discard unconfirmed draft state and release resident draft weights."""

        if self._closed:
            return
        self.model.reset_state(release=True)
        self._committed_history = None
        self._last_hidden = None
        self._pending_base = None
        self._pending_proposal = None
        self._pending_stage = None
        self._poisoned = False
        self._closed = True
        self._seal = self._runtime_stamp()


@dataclass(frozen=True, slots=True)
class Qwen35K4DraftProviderMetrics:
    """Cumulative K=4 local-draft work and reconciliation outcomes."""

    schema: str
    prefill_calls: int
    draft_calls: int
    extension_calls: int
    reconcile_calls: int
    accepted_prefix_0: int
    accepted_prefix_1: int
    accepted_prefix_2: int
    accepted_prefix_3: int
    accepted_prefix_4: int
    restaged_blocks: int
    committed_tokens: int
    source_body_bytes: int
    linear_calls: int
    state_bytes: int
    pending: bool
    poisoned: bool
    window_size: int = 4
    accepted_prefix_counts: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Qwen35K4DraftProvider:
    """Build an opaque bounded local draft without moving its commit cursor.

    The first token opens a width-one continuation stage.  Three opaque
    extensions grow it to K=4 while the model's public committed cursor stays
    fixed.  Reconciliation either commits the fully accepted stage or discards
    it and replays exactly the target-confirmed delta of width one through four.
    """

    target_state_isolation = "no-target-state-access/v1"

    def __init__(
        self,
        model: StreamedQwen38,
        *,
        eos_token_ids: Iterable[int] = (),
        head_block_rows: int = Qwen38WeightPager.DEFAULT_HEAD_BLOCK_ROWS,
        window_size: int = 4,
    ) -> None:
        if not isinstance(model, StreamedQwen38):
            raise TypeError("model must be a StreamedQwen38")
        if (
            isinstance(head_block_rows, bool)
            or not isinstance(head_block_rows, int)
            or head_block_rows <= 0
        ):
            raise ValueError("head_block_rows must be a positive integer")
        if (
            isinstance(window_size, bool)
            or not isinstance(window_size, int)
            or not 2 <= window_size <= model.MAX_CONTINUATION_BLOCK_WIDTH
        ):
            raise ValueError("window_size must lie in [2, 16]")
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
        if any(token < 0 or token >= model.config.vocab_size for token in eos):
            raise ValueError("EOS token outside draft checkpoint vocabulary")
        if (
            model.next_position != 0
            or model.state_batch_size is not None
            or model.state_bytes != 0
            or model.state_poisoned
            or model._pending_block_stage is not None
        ):
            raise ValueError("draft model must start with empty clean state")

        self.model = model
        self.eos_token_ids = frozenset(eos)
        self.head_block_rows = head_block_rows
        self.window_size = window_size
        self._committed_history: tuple[int, ...] | None = None
        self._last_hidden: torch.Tensor | None = None
        self._pending_base: tuple[int, ...] | None = None
        self._pending_proposal: tuple[int, ...] | None = None
        self._pending_stage: StatefulBlockStage | None = None
        self._pending_rolling = False
        self._poisoned = False
        self._closed = False
        self._prefill_calls = 0
        self._draft_calls = 0
        self._extension_calls = 0
        self._reconcile_calls = 0
        self._accepted = [0] * (window_size + 1)
        self._restaged_blocks = 0
        self._committed_tokens = 0
        self._source_start = _metric(model.pager.source, "network_or_source_body_bytes")
        self._linears_start = _metric(model.pager, "linear_calls")
        self._seal = self._runtime_stamp()

    @property
    def committed_history(self) -> tuple[int, ...] | None:
        return self._committed_history

    @property
    def pending_proposal(self) -> tuple[int, ...] | None:
        return self._pending_proposal

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def closed(self) -> bool:
        return self._closed

    def metrics(self) -> Qwen35K4DraftProviderMetrics:
        return Qwen35K4DraftProviderMetrics(
            schema=(
                QWEN35_K4_DRAFT_PROVIDER_SCHEMA
                if self.window_size == 4
                else QWEN35_ROLLING_DRAFT_PROVIDER_SCHEMA
            ),
            prefill_calls=self._prefill_calls,
            draft_calls=self._draft_calls,
            extension_calls=self._extension_calls,
            reconcile_calls=self._reconcile_calls,
            accepted_prefix_0=self._accepted[0],
            accepted_prefix_1=self._accepted[1],
            accepted_prefix_2=self._accepted[2],
            accepted_prefix_3=(
                self._accepted[3] if len(self._accepted) > 3 else 0
            ),
            accepted_prefix_4=(
                self._accepted[4] if len(self._accepted) > 4 else 0
            ),
            restaged_blocks=self._restaged_blocks,
            committed_tokens=self._committed_tokens,
            source_body_bytes=(
                _metric(self.model.pager.source, "network_or_source_body_bytes")
                - self._source_start
            ),
            linear_calls=_metric(self.model.pager, "linear_calls")
            - self._linears_start,
            state_bytes=self.model.state_bytes,
            pending=self._pending_stage is not None,
            poisoned=self._poisoned,
            window_size=self.window_size,
            accepted_prefix_counts=tuple(self._accepted),
        )

    def _pending_stamp(self) -> object:
        pending = self.model._pending_block_stage
        if pending is None:
            return None
        return (
            id(pending),
            id(pending.handle),
            id(pending.handle._nonce),
            asdict(pending.evidence),
            pending.runtime_identity,
            _tensor_identity_stamp(pending.handle.hidden),
            _tensor_identity_stamp(pending.hidden),
            _layer_state_identity_stamp(pending.layer_states),
            _tensor_identity_stamp(pending.graft_history),
        )

    def _runtime_stamp(self) -> tuple[object, ...]:
        return (
            self.model.next_position,
            self.model.state_batch_size,
            self.model.state_poisoned,
            self.model._continuation_block_runtime_identity(),
            _layer_state_identity_stamp(self.model._layer_states),
            _tensor_identity_stamp(self.model._graft_history),
            self._pending_stamp(),
            self._committed_history,
            _tensor_identity_stamp(self._last_hidden),
            self._pending_base,
            self._pending_proposal,
            id(self._pending_stage),
            self._pending_rolling,
        )

    def _abort(self, message: str, cause: Exception | None = None) -> None:
        try:
            self.model.reset_state(release=True)
        finally:
            self._committed_history = None
            self._last_hidden = None
            self._pending_base = None
            self._pending_proposal = None
            self._pending_stage = None
            self._pending_rolling = False
            self._poisoned = True
            self._seal = self._runtime_stamp()
        error = Qwen35K4DraftProviderError(message)
        if cause is None:
            raise error
        raise error from cause

    def _assert_ready(self) -> None:
        if self._closed:
            raise Qwen35K4DraftProviderError("draft provider is closed")
        if self._poisoned:
            raise Qwen35K4DraftProviderError("draft provider is poisoned; call reset()")
        try:
            changed = self._runtime_stamp() != self._seal
        except Exception as exc:
            self._abort("draft model state cannot be verified", exc)
        if changed:
            self._abort("draft model cursor or continuation state drifted")

    def _history(self, value: object, *, name: str) -> tuple[int, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"{name} must be an integer sequence")
        result: list[int] = []
        for raw in value:
            if isinstance(raw, bool) or not isinstance(raw, numbers.Integral):
                raise TypeError(f"{name} must contain integers")
            token = int(raw)
            if token < 0 or token >= self.model.config.vocab_size:
                raise ValueError(f"{name} token outside draft checkpoint vocabulary")
            result.append(token)
        if not result:
            raise ValueError(f"{name} must not be empty")
        return tuple(result)

    def _scan(self, hidden: torch.Tensor) -> int:
        values, selected = self.model.pager.topk_logits(
            hidden,
            k=1,
            name=self.model.output_head_name,
            block_rows=self.head_block_rows,
        )
        if tuple(selected.shape) != (1, 1):
            raise Qwen35K4DraftProviderError(
                "draft LM head returned an invalid token shape"
            )
        token = int(selected[0, 0].item())
        del values, selected
        if not 0 <= token < self.model.config.vocab_size:
            raise Qwen35K4DraftProviderError(
                "draft LM head selected a token outside its vocabulary"
            )
        return token

    def __call__(self, history: tuple[int, ...], /) -> tuple[int, ...]:
        self._assert_ready()
        try:
            committed = self._history(history, name="draft history")
            if self._pending_stage is not None:
                self._abort("previous draft proposal was not reconciled before reuse")
            if self._committed_history is None:
                if len(committed) + self.window_size > self.model.max_seq_len:
                    raise ValueError("draft block would exceed max_seq_len")
                hidden, _evidence = self.model.prefill([committed], reset=True)
                self._last_hidden = hidden[:, -1:].detach().clone()
                self._committed_history = committed
                self._prefill_calls += 1
            elif committed != self._committed_history:
                self._abort("draft history differs from committed local history")
            if self.model.next_position != len(committed):
                self._abort("draft model cursor disagrees with committed history")
            if len(committed) + self.window_size > self.model.max_seq_len:
                raise ValueError("draft block would exceed max_seq_len")
            if committed[-1] in self.eos_token_ids:
                raise ValueError("cannot draft after a committed EOS token")
            if self._last_hidden is None:
                self._abort("draft model has no hidden state for its committed cursor")

            proposal = [self._scan(self._last_hidden[:, -1])]
            stage = self.model.stage_continuation_block([[proposal[0]]])
            while len(proposal) < self.window_size:
                token = (
                    proposal[-1]
                    if proposal[-1] in self.eos_token_ids
                    else self._scan(stage.hidden[:, -1])
                )
                proposal.append(token)
                stage = self.model.extend_continuation_block(stage, [[token]])
                self._extension_calls += 1

            block = tuple(proposal)
            if self.model.next_position != len(committed):
                self._abort("draft proposal moved the committed model cursor")
            self._pending_base = committed
            self._pending_proposal = block
            self._pending_stage = stage
            self._pending_rolling = False
            self._draft_calls += 1
            self._seal = self._runtime_stamp()
            return block
        except Qwen35K4DraftProviderError as exc:
            if not self._poisoned:
                self._abort(str(exc), exc)
            raise
        except Exception as exc:
            self._abort(f"local K=4 draft failed: {type(exc).__name__}: {exc}", exc)

    def propose_after(
        self,
        history: tuple[int, ...],
        known_token: int,
        /,
    ) -> tuple[int, ...]:
        """Commit one target-known token, then stage the configured draft tail."""

        self._assert_ready()
        try:
            committed = self._history(history, name="rolling draft history")
            if isinstance(known_token, bool) or not isinstance(
                known_token, numbers.Integral
            ):
                raise TypeError("known rolling token must be an integer")
            known = int(known_token)
            if not 0 <= known < self.model.config.vocab_size:
                raise ValueError("known rolling token is outside the vocabulary")
            if known in self.eos_token_ids:
                raise ValueError("cannot draft after a target-known EOS token")
            if self._pending_stage is not None:
                self._abort("previous rolling draft was not reconciled")
            combined = (*committed, known)
            proposal_width = self.window_size - 1
            if len(combined) + proposal_width > self.model.max_seq_len:
                raise ValueError("rolling draft block would exceed max_seq_len")
            if self._committed_history is None:
                hidden, _evidence = self.model.prefill([combined], reset=True)
                self._last_hidden = hidden[:, -1:].detach().clone()
                self._committed_history = combined
                self._prefill_calls += 1
                self._committed_tokens += 1
            else:
                if committed != self._committed_history:
                    self._abort("rolling history differs from committed local history")
                hidden, _evidence = self.model.decode([[known]])
                self._last_hidden = hidden[:, -1:].detach().clone()
                self._committed_history = combined
                self._committed_tokens += 1
            if self.model.next_position != len(combined):
                self._abort("rolling draft cursor disagrees with known history")
            if self._last_hidden is None:
                self._abort("rolling draft has no target-known hidden state")

            proposal = [self._scan(self._last_hidden[:, -1])]
            stage = self.model.stage_continuation_block([[proposal[0]]])
            while len(proposal) < proposal_width:
                token = (
                    proposal[-1]
                    if proposal[-1] in self.eos_token_ids
                    else self._scan(stage.hidden[:, -1])
                )
                proposal.append(token)
                stage = self.model.extend_continuation_block(stage, [[token]])
                self._extension_calls += 1
            block = tuple(proposal)
            self._pending_base = combined
            self._pending_proposal = block
            self._pending_stage = stage
            self._pending_rolling = True
            self._draft_calls += 1
            self._seal = self._runtime_stamp()
            return block
        except Qwen35K4DraftProviderError as exc:
            if not self._poisoned:
                self._abort(str(exc), exc)
            raise
        except Exception as exc:
            self._abort(
                f"rolling local draft failed: {type(exc).__name__}: {exc}",
                exc,
            )

    def reconcile_prefix(self, history: tuple[int, ...], /) -> None:
        """Commit the target-accepted prefix of the configured draft tail."""

        self._assert_ready()
        try:
            committed = self._history(history, name="rolling reconciled history")
            base = self._pending_base
            proposal = self._pending_proposal
            stage = self._pending_stage
            if (
                base is None
                or proposal is None
                or stage is None
                or not self._pending_rolling
                or len(proposal) != self.window_size - 1
            ):
                self._abort("reconcile_prefix requires one rolling proposal")
            if committed[: len(base)] != base:
                self._abort("rolling reconciliation changed its committed base")
            delta = committed[len(base) :]
            if (
                len(delta) > self.window_size - 1
                or tuple(delta) != proposal[: len(delta)]
            ):
                self._abort("rolling reconciliation is not an accepted draft prefix")
            if any(token in self.eos_token_ids for token in delta[:-1]):
                self._abort("rolling reconciliation contains tokens after EOS")
            if self.model.next_position != len(base):
                self._abort("rolling draft model cursor changed under proposal")

            if not delta:
                self.model.discard_continuation_block(stage)
                hidden = self._last_hidden
                if hidden is None:
                    self._abort("rolling draft lost its known-token hidden state")
            elif len(delta) == len(proposal):
                hidden, _evidence = self.model.commit_continuation_block(stage)
            else:
                hidden, _evidence = self.model.commit_continuation_prefix(
                    stage,
                    len(delta),
                )
            self._pending_base = None
            self._pending_proposal = None
            self._pending_stage = None
            self._pending_rolling = False
            self._committed_history = committed
            self._last_hidden = hidden[:, -1:].detach().clone()
            if (
                self.model.next_position != len(committed)
                or self.model._pending_block_stage is not None
            ):
                self._abort("rolling draft state has the wrong committed cursor")
            accepted = len(delta)
            self._reconcile_calls += 1
            self._accepted[accepted] += 1
            self._committed_tokens += accepted
            self._seal = self._runtime_stamp()
        except Qwen35K4DraftProviderError as exc:
            if not self._poisoned:
                self._abort(str(exc), exc)
            raise
        except Exception as exc:
            self._abort(
                f"rolling reconciliation failed: {type(exc).__name__}: {exc}",
                exc,
            )

    def advance_confirmed_prefix(self, history: tuple[int, ...], /) -> None:
        """Advance an idle local drafter through target-confirmed tokens."""

        self._assert_ready()
        try:
            committed = self._history(history, name="confirmed draft history")
            if self._pending_stage is not None:
                self._abort("cannot advance with a pending local draft")
            previous = self._committed_history
            if previous is None:
                hidden, _evidence = self.model.prefill([committed], reset=True)
                self._prefill_calls += 1
                added = 0
            else:
                if committed[: len(previous)] != previous:
                    self._abort("confirmed history changed the local draft prefix")
                delta = committed[len(previous) :]
                if not delta:
                    return
                hidden, _evidence = self.model.decode([delta])
                added = len(delta)
            self._committed_history = committed
            self._last_hidden = hidden[:, -1:].detach().clone()
            self._committed_tokens += added
            if self.model.next_position != len(committed):
                self._abort("confirmed draft state has the wrong committed cursor")
            self._seal = self._runtime_stamp()
        except Qwen35K4DraftProviderError as exc:
            if not self._poisoned:
                self._abort(str(exc), exc)
            raise
        except Exception as exc:
            self._abort(
                f"confirmed-prefix advance failed: {type(exc).__name__}: {exc}",
                exc,
            )

    def observe_final(self, history: tuple[int, ...], /) -> None:
        """Synchronize an unproposed terminal target suffix before close."""

        self.advance_confirmed_prefix(history)

    def reconcile(self, history: tuple[int, ...], /) -> None:
        """Commit only the target-confirmed delta and destroy rejected suffixes."""

        self._assert_ready()
        try:
            committed = self._history(history, name="reconciled history")
            base = self._pending_base
            proposal = self._pending_proposal
            stage = self._pending_stage
            if base is None or proposal is None or stage is None:
                self._abort("reconcile requires one pending K=4 proposal")
            if self._pending_rolling:
                self._abort("rolling draft requires reconcile_prefix")
            if committed[: len(base)] != base:
                self._abort("reconciled history changed the committed draft prefix")
            delta = committed[len(base) :]
            if not 1 <= len(delta) <= self.window_size:
                self._abort("reconciled history has the wrong committed width")
            if any(token in self.eos_token_ids for token in delta[:-1]):
                self._abort("reconciled history contains a token after EOS")
            if delta[:-1] != proposal[: len(delta) - 1]:
                self._abort("reconciled delta rejects a token before its final item")
            if (
                len(delta) < self.window_size
                and delta[-1] == proposal[len(delta) - 1]
                and delta[-1] not in self.eos_token_ids
            ):
                self._abort("reconciled delta truncates an accepted block without EOS")
            if self.model.next_position != len(base):
                self._abort("draft model cursor changed under a pending proposal")

            accepted = 0
            while accepted < len(delta) and delta[accepted] == proposal[accepted]:
                accepted += 1
            if len(delta) == self.window_size and delta == proposal:
                hidden, _evidence = self.model.commit_continuation_block(stage)
            else:
                self.model.discard_continuation_block(stage)
                if len(delta) == 1:
                    hidden, _evidence = self.model.decode([[delta[0]]])
                else:
                    replay = self.model.stage_continuation_block([delta])
                    hidden, _evidence = self.model.commit_continuation_block(replay)
                    self._restaged_blocks += 1

            self._pending_base = None
            self._pending_proposal = None
            self._pending_stage = None
            self._pending_rolling = False
            self._committed_history = committed
            self._last_hidden = hidden[:, -1:].detach().clone()
            if (
                self.model.next_position != len(committed)
                or self.model._pending_block_stage is not None
            ):
                self._abort("reconciled draft state has the wrong committed cursor")
            self._reconcile_calls += 1
            self._accepted[accepted] += 1
            self._committed_tokens += len(delta)
            self._seal = self._runtime_stamp()
        except Qwen35K4DraftProviderError as exc:
            if not self._poisoned:
                self._abort(str(exc), exc)
            raise
        except Exception as exc:
            self._abort(
                f"local K=4 reconciliation failed: {type(exc).__name__}: {exc}",
                exc,
            )

    def reset(self) -> None:
        """Recover explicitly after a failed request or start a new request."""

        if self._closed:
            raise Qwen35K4DraftProviderError("draft provider is closed")
        self.model.reset_state(release=True)
        self._committed_history = None
        self._last_hidden = None
        self._pending_base = None
        self._pending_proposal = None
        self._pending_stage = None
        self._pending_rolling = False
        self._poisoned = False
        self._seal = self._runtime_stamp()

    def close(self) -> None:
        """Discard unconfirmed draft state and release resident draft weights."""

        if self._closed:
            return
        self.model.reset_state(release=True)
        self._committed_history = None
        self._last_hidden = None
        self._pending_base = None
        self._pending_proposal = None
        self._pending_stage = None
        self._pending_rolling = False
        self._poisoned = False
        self._closed = True
        self._seal = self._runtime_stamp()


__all__ = [
    "QWEN35_K2_DRAFT_PROVIDER_SCHEMA",
    "QWEN35_K4_DRAFT_PROVIDER_SCHEMA",
    "QWEN35_ROLLING_DRAFT_PROVIDER_SCHEMA",
    "Qwen35K2DraftProvider",
    "Qwen35K2DraftProviderError",
    "Qwen35K2DraftProviderMetrics",
    "Qwen35K4DraftProvider",
    "Qwen35K4DraftProviderError",
    "Qwen35K4DraftProviderMetrics",
]
