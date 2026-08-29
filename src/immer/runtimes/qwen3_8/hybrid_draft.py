"""Request-locked Markov-first cascade over the embedded Qwen MTP branch."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Literal

import torch

from .draft_protocol import RollingDraftProposal
from .mtp_draft import Qwen35MtpDraftProvider


QWEN38_MARKOV_MTP_HYBRID_PROVIDER_SCHEMA = "immer.qwen3.8-markov-mtp-hybrid-provider/v1"
MARKOV_MTP_WINDOW_WORK_COSTS = {
    1: 1.0,
    2: 1.6,
    4: 2.8,
    8: 5.2,
    16: 10.0,
}


class Qwen38MarkovMtpDraftError(RuntimeError):
    """The Markov/MTP request cascade violated its one-provider contract."""


def _metrics_record(owner: object | None) -> dict[str, Any] | None:
    if owner is None:
        return None
    callback = getattr(owner, "metrics", None)
    if not callable(callback):
        return None
    value = callback()
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        record = to_dict()
    elif isinstance(value, Mapping):
        record = dict(value)
    elif is_dataclass(value) and not isinstance(value, type):
        record = asdict(value)
    else:
        raise Qwen38MarkovMtpDraftError(
            "nested draft-provider metrics are not serializable"
        )
    if not isinstance(record, Mapping):
        raise Qwen38MarkovMtpDraftError(
            "nested draft-provider metrics must be a mapping"
        )
    return dict(record)


def _nonnegative_metric(record: Mapping[str, Any] | None, name: str) -> int:
    if record is None:
        return 0
    value = record.get(name, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _optional_metric(
    record: Mapping[str, Any] | None,
    name: str,
    default: object,
) -> object:
    return default if record is None else record.get(name, default)


@dataclass(frozen=True, slots=True)
class Qwen38MarkovMtpDraftMetrics:
    schema: str
    selected_provider: Literal["markov", "mtp"] | None
    selection_calls: int
    markov_selections: int
    mtp_selections: int
    source_body_bytes: int
    linear_calls: int
    last_confidence: float
    last_disagreement: float
    last_phrase_source: str | None
    last_phrase_support: int
    last_phrase_confidence: float
    last_phrase_width: int
    effective_experts: float | None
    pending: bool
    closed: bool
    markov: Mapping[str, Any] | None
    mtp: Mapping[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["markov"] = None if self.markov is None else dict(self.markov)
        value["mtp"] = None if self.mtp is None else dict(self.mtp)
        return value


MtpFactory = Callable[[], Qwen35MtpDraftProvider]


class Qwen38MarkovMtpDraftProvider:
    """Ask Markov for a free first-round route, then lock the request.

    Markov already has the prompt and can produce its proposal without reading
    model weights.  Only a K1 selection constructs the embedded MTP provider.
    The selected provider cannot change for the remainder of the request.
    """

    target_state_isolation = "hidden-argument+shared-pager-only/v1"

    def __init__(self, markov_provider: object, mtp_factory: MtpFactory) -> None:
        if not callable(mtp_factory):
            raise TypeError("mtp_factory must be callable")
        required = (
            "begin_request",
            "propose_round",
            "discard_pending_proposal",
            "reconcile_prefix",
            "observe_final",
            "close",
        )
        if any(not callable(getattr(markov_provider, name, None)) for name in required):
            raise TypeError("markov_provider lacks the rolling council callbacks")
        self.markov_provider = markov_provider
        self._mtp_factory = mtp_factory
        self._mtp_provider: Qwen35MtpDraftProvider | object | None = None
        self._selected_provider: Literal["markov", "mtp"] | None = None
        self._request_history: tuple[int, ...] | None = None
        self._prompt_hidden: torch.Tensor | None = None
        self._request_started = False
        self._request_completed = False
        self._closed = False
        self._selection_calls = 0
        self._markov_selections = 0
        self._mtp_selections = 0
        self._pending = False

    @property
    def selected_provider(self) -> Literal["markov", "mtp"] | None:
        return self._selected_provider

    @property
    def mtp_provider(self) -> object | None:
        return self._mtp_provider

    def _require_open_request(self) -> None:
        if self._closed:
            raise Qwen38MarkovMtpDraftError("hybrid draft provider is closed")
        if not self._request_started or self._request_completed:
            raise Qwen38MarkovMtpDraftError("hybrid request is not active")

    def begin_request_state(
        self,
        history: tuple[int, ...],
        target_hidden: torch.Tensor,
        /,
    ) -> None:
        if self._closed:
            raise Qwen38MarkovMtpDraftError("hybrid draft provider is closed")
        if self._request_started or self._request_completed:
            raise Qwen38MarkovMtpDraftError(
                "hybrid provider accepts exactly one request"
            )
        if not isinstance(history, tuple) or not history:
            raise ValueError("hybrid request history must be a non-empty tuple")
        if not isinstance(target_hidden, torch.Tensor):
            raise TypeError("target_hidden must be a torch.Tensor")
        if target_hidden.ndim != 3 or target_hidden.shape[1] != len(history):
            raise ValueError("prompt hidden rows must match request history")
        self.markov_provider.begin_request(history)
        self._request_history = history
        self._prompt_hidden = target_hidden.detach().clone()
        self._request_started = True

    def __call__(self, _history: tuple[int, ...], /) -> tuple[int, ...]:
        raise Qwen38MarkovMtpDraftError(
            "hybrid drafting requires target-hidden rolling callbacks"
        )

    @staticmethod
    def _select_markov(proposal: RollingDraftProposal) -> bool:
        ceiling = len(proposal.token_ids) + 1
        policy = proposal.select_window(
            request_window_ceiling=ceiling,
            remaining_tokens=ceiling,
            window_work_costs=MARKOV_MTP_WINDOW_WORK_COSTS,
        )
        return policy.chosen_window > 1

    def _load_mtp(self) -> object:
        if self._mtp_provider is not None:
            return self._mtp_provider
        provider = self._mtp_factory()
        required = (
            "begin_request_state",
            "propose_round_state",
            "propose_after_state",
            "reconcile_prefix",
            "observe_final",
            "close",
        )
        if any(not callable(getattr(provider, name, None)) for name in required):
            close = getattr(provider, "close", None)
            if callable(close):
                close()
            raise Qwen38MarkovMtpDraftError(
                "mtp_factory returned no stateful rolling provider"
            )
        if getattr(provider, "target_state_isolation", None) != (
            self.target_state_isolation
        ):
            provider.close()
            raise Qwen38MarkovMtpDraftError(
                "MTP provider does not preserve target-state isolation"
            )
        self._mtp_provider = provider
        return provider

    def _choose_first_provider(
        self,
        history: tuple[int, ...],
        known_token: int,
    ) -> RollingDraftProposal | None:
        if self._selected_provider is not None:
            return None
        proposal = self.markov_provider.propose_round(history, known_token)
        if not isinstance(proposal, RollingDraftProposal):
            raise Qwen38MarkovMtpDraftError(
                "Markov council returned no RollingDraftProposal"
            )
        self._selection_calls += 1
        if self._select_markov(proposal):
            self._selected_provider = "markov"
            self._markov_selections += 1
            self._prompt_hidden = None
            return proposal

        self.markov_provider.discard_pending_proposal()
        mtp = self._load_mtp()
        prompt_hidden = self._prompt_hidden
        request_history = self._request_history
        if prompt_hidden is None or request_history is None:
            raise Qwen38MarkovMtpDraftError("stored prompt state is missing")
        # MTP owns a private copy.  Neither the caller's tensor nor the saved
        # selection copy can be mutated by provider initialization.
        mtp.begin_request_state(request_history, prompt_hidden.detach().clone())
        self._prompt_hidden = None
        self._selected_provider = "mtp"
        self._mtp_selections += 1
        return None

    def propose_round_state(
        self,
        history: tuple[int, ...],
        known_token: int,
        target_hidden: torch.Tensor,
        /,
    ) -> RollingDraftProposal:
        self._require_open_request()
        selected = self._choose_first_provider(history, known_token)
        if selected is not None:
            self._pending = True
            return selected
        if self._selected_provider == "markov":
            proposal = self.markov_provider.propose_round(history, known_token)
        elif self._selected_provider == "mtp":
            assert self._mtp_provider is not None
            proposal = self._mtp_provider.propose_round_state(
                history,
                known_token,
                target_hidden,
            )
        else:  # pragma: no cover - guarded by the selection transition.
            raise Qwen38MarkovMtpDraftError("hybrid provider was not selected")
        if not isinstance(proposal, RollingDraftProposal):
            raise Qwen38MarkovMtpDraftError(
                "selected provider returned no RollingDraftProposal"
            )
        self._pending = True
        return proposal

    def propose_after_state(
        self,
        history: tuple[int, ...],
        known_token: int,
        target_hidden: torch.Tensor,
        /,
    ) -> tuple[int, ...]:
        self._require_open_request()
        # Fixed short windows do not expose the K4/K8/K16 horizon contract
        # required by ``propose_round``.  They are already the cheap path, so
        # lock directly to Markov and never materialize MTP for this request.
        if self._selected_provider is None:
            self._selected_provider = "markov"
            self._selection_calls += 1
            self._markov_selections += 1
            self._prompt_hidden = None
        if self._selected_provider == "markov":
            proposal = self.markov_provider.propose_after(history, known_token)
        elif self._selected_provider == "mtp":
            assert self._mtp_provider is not None
            proposal = self._mtp_provider.propose_after_state(
                history,
                known_token,
                target_hidden,
            )
        else:  # pragma: no cover - guarded by the selection transition.
            raise Qwen38MarkovMtpDraftError("hybrid provider was not selected")
        self._pending = True
        return tuple(proposal)

    def observe_verification(
        self,
        accepted_prefix_length: int,
        verified_proposals: int,
        /,
    ) -> None:
        self._require_open_request()
        if not self._pending:
            raise Qwen38MarkovMtpDraftError(
                "verification has no pending hybrid proposal"
            )
        owner = (
            self.markov_provider
            if self._selected_provider == "markov"
            else self._mtp_provider
        )
        callback = getattr(owner, "observe_verification", None)
        if callable(callback):
            callback(accepted_prefix_length, verified_proposals)

    def reconcile_prefix(self, history: tuple[int, ...], /) -> None:
        self._require_open_request()
        if not self._pending:
            raise Qwen38MarkovMtpDraftError(
                "reconciliation has no pending hybrid proposal"
            )
        owner = (
            self.markov_provider
            if self._selected_provider == "markov"
            else self._mtp_provider
        )
        if owner is None:  # pragma: no cover - selection invariant.
            raise Qwen38MarkovMtpDraftError("hybrid provider was not selected")
        owner.reconcile_prefix(history)
        self._pending = False

    def observe_final(self, history: tuple[int, ...], /) -> None:
        self._require_open_request()
        if self._pending:
            raise Qwen38MarkovMtpDraftError(
                "cannot finalize an unreconciled hybrid proposal"
            )
        if self._selected_provider == "mtp":
            assert self._mtp_provider is not None
            self._mtp_provider.observe_final(history)
            # The MTP branch executes the request, while Markov receives only
            # the final target-confirmed episode and therefore learns without
            # treating its discarded speculative tokens as observations.
            self.markov_provider.observe_final(history)
        else:
            self.markov_provider.observe_final(history)
        self._request_completed = True

    def metrics(self) -> Qwen38MarkovMtpDraftMetrics:
        markov = _metrics_record(self.markov_provider)
        mtp = _metrics_record(self._mtp_provider)
        return Qwen38MarkovMtpDraftMetrics(
            schema=QWEN38_MARKOV_MTP_HYBRID_PROVIDER_SCHEMA,
            selected_provider=self._selected_provider,
            selection_calls=self._selection_calls,
            markov_selections=self._markov_selections,
            mtp_selections=self._mtp_selections,
            source_body_bytes=(
                _nonnegative_metric(markov, "source_body_bytes")
                + _nonnegative_metric(mtp, "source_body_bytes")
            ),
            linear_calls=(
                _nonnegative_metric(markov, "linear_calls")
                + _nonnegative_metric(mtp, "linear_calls")
            ),
            last_confidence=float(_optional_metric(markov, "last_confidence", 0.0)),
            last_disagreement=float(_optional_metric(markov, "last_disagreement", 0.0)),
            last_phrase_source=_optional_metric(markov, "last_phrase_source", None),
            last_phrase_support=int(_optional_metric(markov, "last_phrase_support", 0)),
            last_phrase_confidence=float(
                _optional_metric(markov, "last_phrase_confidence", 0.0)
            ),
            last_phrase_width=int(_optional_metric(markov, "last_phrase_width", 0)),
            effective_experts=(
                None
                if markov is None or markov.get("effective_experts") is None
                else float(markov["effective_experts"])
            ),
            pending=self._pending,
            closed=self._closed,
            markov=markov,
            mtp=mtp,
        )

    def close(self) -> None:
        if self._closed:
            return
        failure: BaseException | None = None
        try:
            if self._mtp_provider is not None:
                self._mtp_provider.close()
        except BaseException as exc:
            failure = exc
        try:
            # The Council's exclusive state lock also serializes hybrid
            # requests.  Release it only after the lazy MTP calibration has
            # reached disk, so another hybrid process cannot observe a half-
            # closed request pair.
            self.markov_provider.close()
        except BaseException as exc:
            if failure is None:
                failure = exc
        self._prompt_hidden = None
        self._closed = True
        if failure is not None:
            raise failure


__all__ = [
    "MARKOV_MTP_WINDOW_WORK_COSTS",
    "QWEN38_MARKOV_MTP_HYBRID_PROVIDER_SCHEMA",
    "Qwen38MarkovMtpDraftError",
    "Qwen38MarkovMtpDraftMetrics",
    "Qwen38MarkovMtpDraftProvider",
]
