"""Round-wise Markov/MTP council over target-confirmed hidden history."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass, replace
import math
from typing import Any, Literal

import torch

from .draft_protocol import RollingDraftProposal
from .mtp_draft import Qwen35MtpCarry, Qwen35MtpDraftProvider


QWEN38_MARKOV_MTP_HYBRID_PROVIDER_SCHEMA = "immer.qwen3.8-markov-mtp-hybrid-provider/v31"
ATLAS_MTP_CONSENSUS_STRENGTH = 0.25
ONLINE_MTP_CONSENSUS_STRENGTH = 0.25
PROVIDER_TOURNAMENT_DISCOUNT = 0.85
MTP_COMPLETE_WAVE_PREFIX_TOKENS = 2
MTP_COMPLETE_WAVE_MIN_PROBABILITY = 0.90
MARKOV_MTP_WINDOW_WORK_COSTS = {
    1: 1.0,
    2: 1.6,
    4: 2.8,
    8: 5.2,
    16: 10.0,
}


class Qwen38MarkovMtpDraftError(RuntimeError):
    """The Markov/MTP council lost target-confirmed provider state."""


@dataclass(frozen=True, slots=True)
class _ProviderTournamentTrace:
    providers: tuple[str, ...]
    candidates: tuple[tuple[int, ...], ...]
    alive: tuple[bool, ...]
    next_position: int


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
    selected_provider: Literal["markov", "qwen35", "mtp"] | None
    selection_calls: int
    markov_selections: int
    qwen35_selections: int
    mtp_selections: int
    markov_rounds: int
    qwen35_rounds: int
    mtp_rounds: int
    markov_external_feedback_rounds: int
    provider_switches: int
    provider_tournament_calls: int
    provider_tournament_markov_selections: int
    provider_tournament_qwen35_selections: int
    provider_tournament_mtp_selections: int
    provider_trace_created: int
    provider_trace_active: int
    provider_trace_feedback_tokens: int
    qwen35_wave_gate_checks: int
    qwen35_wave_gate_unknown: int
    qwen35_wave_gate_passes: int
    qwen35_wave_gate_rejections: int
    last_qwen35_complete_wave_probability: float | None
    mtp_wave_gate_checks: int
    mtp_wave_gate_unknown: int
    mtp_wave_gate_passes: int
    mtp_wave_gate_rejections: int
    last_mtp_complete_wave_probability: float | None
    qwen35_init_failures: int
    mtp_init_failures: int
    qwen35_proposed_tokens: int
    qwen35_accepted_tokens: int
    consensus_rounds: int
    consensus_agreement_tokens: int
    consensus_confidence_gain: float
    last_consensus_agreement_tokens: int
    last_consensus_confidence_gain: float
    atlas_consensus_rounds: int
    atlas_consensus_tokens: int
    atlas_consensus_confidence_gain: float
    last_atlas_consensus_tokens: int
    last_atlas_consensus_confidence_gain: float
    online_consensus_rounds: int
    online_consensus_tokens: int
    online_consensus_confidence_gain: float
    last_online_consensus_tokens: int
    last_online_consensus_confidence_gain: float
    hidden_history_rows: int
    hidden_history_bytes: int
    switch_available: bool
    external_source_body_bytes: int
    external_linear_calls: int
    mtp_shared_source_body_bytes: int
    mtp_shared_linear_calls: int
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
    qwen35: Mapping[str, Any] | None
    mtp: Mapping[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["markov"] = None if self.markov is None else dict(self.markov)
        value["qwen35"] = None if self.qwen35 is None else dict(self.qwen35)
        value["mtp"] = None if self.mtp is None else dict(self.mtp)
        return value


MtpFactory = Callable[[], Qwen35MtpDraftProvider]
Qwen35Factory = Callable[[], object]
ProviderName = Literal["markov", "qwen35", "mtp"]


class Qwen38MarkovMtpDraftProvider:
    """Choose the strongest cheap expert again on every adaptive round.

    Markov proposals are free of model-weight reads and therefore get first
    refusal.  MTP handles novelty, but both providers follow every committed
    target prefix.  A phrase learned from earlier Qwen answers can consequently
    take control in the middle of a previously unseen response instead of being
    locked out after the first MTP fallback.
    """

    target_state_isolation = "hidden-argument+shared-pager-only/v1"

    def __init__(
        self,
        markov_provider: object,
        mtp_factory: MtpFactory,
        *,
        restored_prefix_length: int | None = None,
        qwen35_factory: Qwen35Factory | None = None,
        max_new_tokens: int | None = None,
    ) -> None:
        if not callable(mtp_factory):
            raise TypeError("mtp_factory must be callable")
        required = (
            "begin_request",
            "propose_after",
            "propose_round",
            "discard_pending_proposal",
            "advance_confirmed_prefix",
            "reconcile_external_prefix",
            "reconcile_prefix",
            "observe_final",
            "close",
        )
        if any(not callable(getattr(markov_provider, name, None)) for name in required):
            raise TypeError("markov_provider lacks the rolling council callbacks")
        if qwen35_factory is not None and not callable(qwen35_factory):
            raise TypeError("qwen35_factory must be callable or None")
        if max_new_tokens is not None and (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer or None")
        if qwen35_factory is not None and max_new_tokens is None:
            raise ValueError("qwen35_factory requires max_new_tokens")
        self.markov_provider = markov_provider
        self._mtp_factory = mtp_factory
        self._qwen35_factory = qwen35_factory
        self._max_new_tokens = max_new_tokens
        self._request_start_length: int | None = None
        self._restored_prefix_length = restored_prefix_length
        self._exact_restored_anchor = False
        self._mtp_provider: Qwen35MtpDraftProvider | object | None = None
        self._qwen35_provider: object | None = None
        self._selected_provider: ProviderName | None = None
        self._request_history: tuple[int, ...] | None = None
        self._hidden_history: torch.Tensor | None = None
        self._hidden_device: torch.device | None = None
        self._hidden_dtype: torch.dtype | None = None
        self._hidden_width = 0
        self._hidden_history_rows = 0
        self._hidden_history_bytes = 0
        self._switch_available = False
        self._round_target_hidden: torch.Tensor | None = None
        self._boundary_history: tuple[int, ...] | None = None
        self._boundary_target_hidden: torch.Tensor | None = None
        self._request_started = False
        self._request_completed = False
        self._closed = False
        self._selection_calls = 0
        self._markov_selections = 0
        self._qwen35_selections = 0
        self._mtp_selections = 0
        self._markov_rounds = 0
        self._qwen35_rounds = 0
        self._mtp_rounds = 0
        self._markov_external_feedback_rounds = 0
        self._provider_switches = 0
        self._provider_tournament_calls = 0
        self._provider_tournament_markov_selections = 0
        self._provider_tournament_qwen35_selections = 0
        self._provider_tournament_mtp_selections = 0
        self._provider_trace_created = 0
        self._provider_trace_feedback_tokens = 0
        self._qwen35_wave_gate_checks = 0
        self._qwen35_wave_gate_unknown = 0
        self._qwen35_wave_gate_passes = 0
        self._qwen35_wave_gate_rejections = 0
        self._last_qwen35_complete_wave_probability: float | None = None
        self._mtp_wave_gate_checks = 0
        self._mtp_wave_gate_unknown = 0
        self._mtp_wave_gate_passes = 0
        self._mtp_wave_gate_rejections = 0
        self._last_mtp_complete_wave_probability: float | None = None
        self._qwen35_init_failures = 0
        self._mtp_init_failures = 0
        self._qwen35_proposed_tokens = 0
        self._qwen35_accepted_tokens = 0
        self._consensus_rounds = 0
        self._consensus_agreement_tokens = 0
        self._consensus_confidence_gain = 0.0
        self._last_consensus_agreement_tokens = 0
        self._last_consensus_confidence_gain = 0.0
        self._atlas_consensus_rounds = 0
        self._atlas_consensus_tokens = 0
        self._atlas_consensus_confidence_gain = 0.0
        self._last_atlas_consensus_tokens = 0
        self._last_atlas_consensus_confidence_gain = 0.0
        self._online_consensus_rounds = 0
        self._online_consensus_tokens = 0
        self._online_consensus_confidence_gain = 0.0
        self._last_online_consensus_tokens = 0
        self._last_online_consensus_confidence_gain = 0.0
        self._shadow_markov_proposal: RollingDraftProposal | None = None
        self._pending_provider_candidates: (
            tuple[tuple[str, ...], tuple[tuple[int, ...], ...]] | None
        ) = None
        self._provider_traces: list[_ProviderTournamentTrace] = []
        self._pending_mtp_shadow = False
        self._pending_qwen35_shadow = False
        self._pending_qwen35_candidate: tuple[int, ...] | None = None
        self._pending_provider: ProviderName | None = None
        self._explored_unknown_providers: set[str] = set()

    @property
    def selected_provider(self) -> ProviderName | None:
        return self._selected_provider

    @property
    def mtp_provider(self) -> object | None:
        return self._mtp_provider

    @property
    def qwen35_provider(self) -> object | None:
        return self._qwen35_provider

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
        restored = self._restored_prefix_length
        if restored is not None and (
            isinstance(restored, bool)
            or not isinstance(restored, int)
            or not 1 <= restored <= len(history)
        ):
            raise ValueError(
                "restored hybrid prefix must identify a non-empty prompt prefix"
            )
        exact_restored_anchor = restored == len(history)
        expected_hidden_rows = (
            len(history)
            if restored is None
            else len(history) - restored + int(exact_restored_anchor)
        )
        if (
            target_hidden.ndim != 3
            or target_hidden.shape[0] != 1
            or target_hidden.shape[1] != expected_hidden_rows
            or target_hidden.shape[2] <= 0
            or not target_hidden.is_floating_point()
            or not bool(torch.isfinite(target_hidden).all().item())
        ):
            raise ValueError("prompt hidden rows must match request history")
        self.markov_provider.begin_request(history)
        self._exact_restored_anchor = exact_restored_anchor
        self._request_start_length = len(history)
        self._request_history = history
        self._hidden_history = target_hidden.detach().clone().contiguous()
        self._hidden_device = target_hidden.device
        self._hidden_dtype = target_hidden.dtype
        self._hidden_width = int(target_hidden.shape[2])
        self._hidden_history_rows = expected_hidden_rows
        self._hidden_history_bytes = (
            target_hidden.numel() * target_hidden.element_size()
        )
        self._switch_available = True
        self._boundary_history = history
        self._boundary_target_hidden = (
            target_hidden[:, -1:].detach().clone().contiguous()
        )
        self._request_started = True

    def _mtp_bootstrap_hidden(
        self,
        history: tuple[int, ...],
        hidden_history: torch.Tensor,
    ) -> torch.Tensor:
        """Select exact carry-seed or strict suffix rows for MTP bootstrap."""

        restored = self._restored_prefix_length
        expected_rows = (
            len(history)
            if restored is None
            else len(history) - restored + int(self._exact_restored_anchor)
        )
        if hidden_history.shape[1] != expected_rows:
            raise Qwen38MarkovMtpDraftError(
                "hybrid target-hidden history is not aligned to its restore"
            )
        if (
            self._exact_restored_anchor
            and restored is not None
            and len(history) > restored
        ):
            return hidden_history[:, 1:].detach().clone().contiguous()
        return hidden_history.detach().clone().contiguous()

    def _rollback_failed_exact_mtp_init(self, failure: Exception) -> None:
        try:
            self.markov_provider.discard_pending_proposal()
        except Exception as rollback_failure:
            raise Qwen38MarkovMtpDraftError(
                "exact restored MTP initialization rollback failed"
            ) from rollback_failure
        self._switch_available = False
        self._hidden_history = None
        self._round_target_hidden = None
        raise Qwen38MarkovMtpDraftError(
            "exact restored MTP carry or seed is invalid"
        ) from failure

    def _validate_history(self, history: tuple[int, ...]) -> None:
        if not isinstance(history, tuple) or not history:
            raise ValueError("hybrid history must be a non-empty tuple")
        current = self._request_history
        if (
            current is None
            or len(history) < len(current)
            or history[: len(current)] != current
        ):
            raise Qwen38MarkovMtpDraftError(
                "hybrid history does not extend its confirmed prefix"
            )

    def _state_fragment(
        self,
        history: tuple[int, ...],
        committed_hidden: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_history(history)
        if not isinstance(committed_hidden, torch.Tensor):
            raise TypeError("committed_hidden must be a torch.Tensor")
        current = self._request_history
        assert current is not None
        added = len(history) - len(current)
        if (
            added <= 0
            or self._hidden_width <= 0
            or committed_hidden.ndim != 3
            or committed_hidden.shape != (1, added, self._hidden_width)
            or committed_hidden.device != self._hidden_device
            or committed_hidden.dtype != self._hidden_dtype
            or not committed_hidden.is_floating_point()
        ):
            raise Qwen38MarkovMtpDraftError(
                "committed hidden rows do not match the history extension"
            )
        return committed_hidden.detach().clone().contiguous()

    def _round_proposal_tokens(
        self,
        history: tuple[int, ...],
    ) -> tuple[int, ...]:
        current = self._request_history
        if (
            current is None
            or len(history) <= len(current)
            or history[: len(current)] != current
        ):
            raise Qwen38MarkovMtpDraftError(
                "hybrid round history is not a strict extension"
            )
        return history[len(current) + 1 :]

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
        )
        return policy.chosen_window > 1

    def _provider_policy_score(
        self,
        provider: ProviderName,
        position: int,
    ) -> tuple[float, bool]:
        callback = getattr(self.markov_provider, "provider_policy_score", None)
        if not callable(callback):
            return 0.5, False
        value = callback(provider, position)
        if (
            not isinstance(value, tuple)
            or len(value) != 2
            or isinstance(value[0], bool)
            or not isinstance(value[0], (int, float))
            or not math.isfinite(float(value[0]))
            or not 0.0 <= float(value[0]) <= 1.0
            or not isinstance(value[1], bool)
        ):
            raise Qwen38MarkovMtpDraftError(
                "provider policy score is invalid"
            )
        return float(value[0]), value[1]

    def _provider_utility(
        self,
        provider: ProviderName,
        proposal: RollingDraftProposal,
        *,
        width: int,
    ) -> float:
        prefix_reliability = 1.0
        utility = 0.0
        for position, confidence in enumerate(proposal.token_confidences[:width]):
            reliability, _observed = self._provider_policy_score(
                provider,
                position,
            )
            prefix_reliability = min(prefix_reliability, reliability)
            utility += PROVIDER_TOURNAMENT_DISCOUNT**position * math.log(
                max(1e-12, float(confidence) * prefix_reliability)
            )
        return utility

    @staticmethod
    def _effective_proposal_width(proposal: RollingDraftProposal) -> int:
        width = 0
        for confidence in proposal.token_confidences:
            if confidence <= 0.0:
                break
            width += 1
        return max(1, width)

    def _select_provider_tournament(
        self,
        markov: RollingDraftProposal,
        mtp: RollingDraftProposal | None,
        qwen35: RollingDraftProposal | None = None,
    ) -> tuple[ProviderName, RollingDraftProposal]:
        rows: list[tuple[ProviderName, RollingDraftProposal]] = [("markov", markov)]
        if qwen35 is not None:
            rows.append(("qwen35", qwen35))
        if mtp is not None:
            rows.append(("mtp", mtp))
        if len(rows) < 2 or any(
            len(proposal.token_ids) != len(markov.token_ids)
            for _provider, proposal in rows[1:]
        ):
            raise Qwen38MarkovMtpDraftError(
                "provider tournament proposal widths disagree"
            )
        providers = tuple(provider for provider, _proposal in rows)
        tournament_width = min(
            self._effective_proposal_width(proposal)
            for _provider, proposal in rows[1:]
        )
        has_evidence = any(
            self._provider_policy_score(provider, position)[1]
            for provider, _proposal in rows
            for position in range(tournament_width)
        )
        if not has_evidence:
            # Expensive providers are admitted one at a time.  Their first
            # target-verified proposal is the normal exploration round.
            selected, proposal = rows[-1]
        else:
            selected, proposal = max(
                rows,
                key=lambda row: self._provider_utility(
                    row[0],
                    row[1],
                    width=tournament_width,
                ),
            )
        self._provider_tournament_calls += 1
        if selected == "markov":
            self._provider_tournament_markov_selections += 1
        elif selected == "qwen35":
            self._provider_tournament_qwen35_selections += 1
        else:
            self._provider_tournament_mtp_selections += 1
        self._pending_provider_candidates = (
            providers,
            tuple(
                row.token_ids[: self._effective_proposal_width(row)]
                for _provider, row in rows
            ),
        )
        return selected, proposal

    def _start_pending_provider_trace(self) -> None:
        pending = self._pending_provider_candidates
        self._pending_provider_candidates = None
        if pending is None:
            return
        providers, candidates = pending
        if (
            len(providers) != len(candidates)
            or len(providers) < 2
            or any(not candidate for candidate in candidates)
        ):
            raise Qwen38MarkovMtpDraftError(
                "pending provider tournament is invalid"
            )
        self._provider_traces.append(
            _ProviderTournamentTrace(
                providers=providers,
                candidates=candidates,
                alive=(True,) * len(candidates),
                next_position=0,
            )
        )
        self._provider_trace_created += 1

    def _advance_provider_traces(self, tokens: tuple[int, ...]) -> None:
        for trace in self._provider_traces:
            if (
                trace.next_position < 0
                or any(
                    active and trace.next_position >= len(candidate)
                    for candidate, active in zip(
                        trace.candidates,
                        trace.alive,
                        strict=True,
                    )
                )
            ):
                raise Qwen38MarkovMtpDraftError(
                    "provider tournament trace is invalid"
                )
        feedback = getattr(
            self.markov_provider,
            "observe_provider_policy_feedback",
            None,
        )
        for token in tokens:
            surviving = []
            for trace in self._provider_traces:
                position = trace.next_position
                alive = []
                for provider, candidate, active in zip(
                    trace.providers,
                    trace.candidates,
                    trace.alive,
                    strict=True,
                ):
                    if not active:
                        alive.append(False)
                        continue
                    hit = candidate[position] == token
                    if callable(feedback):
                        feedback(provider, position, hit)
                    self._provider_trace_feedback_tokens += 1
                    alive.append(hit and position + 1 < len(candidate))
                if any(alive):
                    surviving.append(
                        replace(
                            trace,
                            alive=tuple(alive),
                            next_position=position + 1,
                        )
                    )
            self._provider_traces = surviving

    def _commit_markov_selection(
        self,
        proposal: RollingDraftProposal,
    ) -> RollingDraftProposal:
        if self._selected_provider not in {None, "markov"}:
            self._provider_switches += 1
        self._selected_provider = "markov"
        self._markov_selections += 1
        self._markov_rounds += 1
        return proposal

    def _remaining_request_tokens(self, history: tuple[int, ...]) -> int | None:
        if self._max_new_tokens is None:
            return None
        start = self._request_start_length
        if start is None or len(history) < start:
            raise Qwen38MarkovMtpDraftError(
                "hybrid request budget lost its prompt boundary"
            )
        return max(0, self._max_new_tokens - (len(history) - start))

    def _provider_can_save_complete_wave(
        self,
        provider: Literal["qwen35", "mtp"],
        history: tuple[int, ...],
    ) -> bool:
        """Reject known sub-wave provider work before materializing weights."""

        if provider == "qwen35":
            self._qwen35_wave_gate_checks += 1
        else:
            self._mtp_wave_gate_checks += 1
        remaining = self._remaining_request_tokens(history)
        if remaining is not None and remaining < 3:
            if provider == "qwen35":
                self._qwen35_wave_gate_rejections += 1
                self._last_qwen35_complete_wave_probability = None
            else:
                self._mtp_wave_gate_rejections += 1
                self._last_mtp_complete_wave_probability = None
            return False
        callback = getattr(
            self.markov_provider,
            "provider_prefix_probability",
            None,
        )
        if not callable(callback):
            observed = False
            probability = 0.0
        else:
            value = callback(provider, MTP_COMPLETE_WAVE_PREFIX_TOKENS)
            if (
                not isinstance(value, tuple)
                or len(value) != 2
                or isinstance(value[0], bool)
                or not isinstance(value[0], (int, float))
                or not math.isfinite(float(value[0]))
                or not 0.0 <= float(value[0]) <= 1.0
                or not isinstance(value[1], bool)
            ):
                raise Qwen38MarkovMtpDraftError(
                    "provider prefix probability is invalid"
                )
            probability, observed = float(value[0]), value[1]
        if provider == "qwen35":
            self._last_qwen35_complete_wave_probability = (
                probability if observed else None
            )
        else:
            self._last_mtp_complete_wave_probability = (
                probability if observed else None
            )
        if not observed:
            if provider == "qwen35":
                self._qwen35_wave_gate_unknown += 1
            else:
                self._mtp_wave_gate_unknown += 1
            legacy_mtp = provider == "mtp" and self._qwen35_factory is None
            allowed = legacy_mtp or provider not in self._explored_unknown_providers
            if allowed:
                if provider == "qwen35":
                    self._qwen35_wave_gate_passes += 1
                else:
                    self._mtp_wave_gate_passes += 1
                return True
            if provider == "qwen35":
                self._qwen35_wave_gate_rejections += 1
            else:
                self._mtp_wave_gate_rejections += 1
            return False
        if probability >= MTP_COMPLETE_WAVE_MIN_PROBABILITY:
            if provider == "qwen35":
                self._qwen35_wave_gate_passes += 1
            else:
                self._mtp_wave_gate_passes += 1
            return True
        if provider == "qwen35":
            self._qwen35_wave_gate_rejections += 1
        else:
            self._mtp_wave_gate_rejections += 1
        return False

    def _mtp_can_save_complete_wave(
        self,
        history: tuple[int, ...] | None = None,
    ) -> bool:
        """Compatibility wrapper for the two-provider MTP wave gate."""

        current = self._request_history if history is None else history
        if current is None:
            raise Qwen38MarkovMtpDraftError("hybrid request history is missing")
        return self._provider_can_save_complete_wave("mtp", current)

    def _qwen35_can_save_complete_wave(self, history: tuple[int, ...]) -> bool:
        return self._provider_can_save_complete_wave("qwen35", history)

    def _load_qwen35(self) -> object:
        if self._qwen35_provider is not None:
            return self._qwen35_provider
        factory = self._qwen35_factory
        if factory is None:
            raise Qwen38MarkovMtpDraftError("qwen35 provider is not configured")
        provider = factory()
        required = (
            "propose_after",
            "reconcile_prefix",
            "advance_confirmed_prefix",
            "observe_final",
            "close",
        )
        if (
            any(not callable(getattr(provider, name, None)) for name in required)
            or getattr(provider, "target_state_isolation", None)
            != "no-target-state-access/v1"
        ):
            close = getattr(provider, "close", None)
            if callable(close):
                close()
            raise Qwen38MarkovMtpDraftError(
                "qwen35_factory returned no isolated rolling provider"
            )
        self._qwen35_provider = provider
        return provider

    def _load_mtp(self) -> object:
        if self._mtp_provider is not None:
            return self._mtp_provider
        provider = self._mtp_factory()
        required = (
            "begin_request_state",
            "propose_round_state",
            "propose_after_state",
            "advance_confirmed_prefix_state",
            "reconcile_prefix",
            "reconcile_prefix_state",
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

    def _prepare_mtp(self, history: tuple[int, ...]) -> object | None:
        if not self._switch_available or not self._mtp_can_save_complete_wave(history):
            return None
        hidden_history = self._hidden_history
        request_history = self._request_history
        if self._mtp_provider is None and (
            hidden_history is None
            or request_history != history
            or hidden_history.shape[1]
            != (
                len(history)
                if self._restored_prefix_length is None
                else len(history)
                - self._restored_prefix_length
                + int(self._exact_restored_anchor)
            )
        ):
            self._switch_available = False
            self._hidden_history = None
            return None
        try:
            mtp = self._load_mtp()
        except Qwen38MarkovMtpDraftError:
            raise
        except Exception as exc:
            if self._exact_restored_anchor:
                self._rollback_failed_exact_mtp_init(exc)
            self._mtp_init_failures += 1
            self._switch_available = False
            self._hidden_history = None
            return None
        try:
            if hidden_history is not None:
                mtp.begin_request_state(
                    history,
                    self._mtp_bootstrap_hidden(history, hidden_history),
                )
                self._hidden_history = None
        except Exception as exc:
            failed = self._mtp_provider
            self._mtp_provider = None
            if failed is not None:
                try:
                    failed.close()
                except Exception:
                    pass
            if self._exact_restored_anchor:
                self._rollback_failed_exact_mtp_init(exc)
            self._mtp_init_failures += 1
            self._switch_available = False
            self._hidden_history = None
            return None
        self._explored_unknown_providers.add("mtp")
        return mtp

    @staticmethod
    def _qwen35_rolling_proposal(
        token_ids: tuple[int, ...],
        *,
        request_width: int,
    ) -> RollingDraftProposal:
        if (
            not token_ids
            or len(token_ids) > 3
            or request_width < len(token_ids)
            or request_width not in {1, 3, 7, 15}
        ):
            raise Qwen38MarkovMtpDraftError(
                "qwen35 draft differs from the K4 hybrid contract"
            )
        padded = (*token_ids, *((token_ids[-1],) * (request_width - len(token_ids))))
        live = len(token_ids)
        return RollingDraftProposal.build(
            padded,
            (*((1.0,) * live), *((0.0,) * (request_width - live))),
            (*((0.0,) * live), *((1.0,) * (request_width - live))),
            request_window_ceiling=request_width + 1,
            provider_abi=QWEN38_MARKOV_MTP_HYBRID_PROVIDER_SCHEMA,
        )

    def _propose_qwen35(
        self,
        history: tuple[int, ...],
        known_token: int,
        *,
        request_width: int,
    ) -> RollingDraftProposal | None:
        if self._qwen35_factory is None or not self._qwen35_can_save_complete_wave(
            history
        ):
            return None
        try:
            provider = self._load_qwen35()
            raw = provider.propose_after(history, known_token)
            if not isinstance(raw, tuple) or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in raw
            ):
                raise Qwen38MarkovMtpDraftError(
                    "qwen35 provider returned invalid draft tokens"
                )
            capped = tuple(raw[: min(3, request_width)])
            proposal = self._qwen35_rolling_proposal(
                capped,
                request_width=request_width,
            )
        except Qwen38MarkovMtpDraftError:
            raise
        except Exception:
            failed = self._qwen35_provider
            self._qwen35_provider = None
            if failed is not None:
                try:
                    failed.close()
                except Exception:
                    pass
            self._qwen35_init_failures += 1
            return None
        self._explored_unknown_providers.add("qwen35")
        self._qwen35_proposed_tokens += len(capped)
        self._pending_qwen35_candidate = capped
        return proposal

    def _markov_round_or_handoff(
        self,
        history: tuple[int, ...],
        known_token: int,
        target_hidden: torch.Tensor | None = None,
    ) -> RollingDraftProposal | None:
        self._validate_history(history)
        self._shadow_markov_proposal = None
        self._last_consensus_agreement_tokens = 0
        self._last_consensus_confidence_gain = 0.0
        self._last_atlas_consensus_tokens = 0
        self._last_atlas_consensus_confidence_gain = 0.0
        self._last_online_consensus_tokens = 0
        self._last_online_consensus_confidence_gain = 0.0
        self._advance_provider_traces((known_token,))
        stateful_proposal = getattr(
            self.markov_provider,
            "propose_round_state",
            None,
        )
        proposal = (
            stateful_proposal(
                history,
                known_token,
                target_hidden.detach().clone(),
            )
            if target_hidden is not None and callable(stateful_proposal)
            else self.markov_provider.propose_round(history, known_token)
        )
        if not isinstance(proposal, RollingDraftProposal):
            raise Qwen38MarkovMtpDraftError(
                "Markov council returned no RollingDraftProposal"
            )
        self._selection_calls += 1
        if self._select_markov(proposal) or (
            self._qwen35_factory is None and not self._switch_available
        ):
            return self._commit_markov_selection(proposal)
        self._shadow_markov_proposal = proposal
        return None

    def _fuse_mtp_consensus(
        self,
        mtp: RollingDraftProposal,
    ) -> RollingDraftProposal:
        """Discount exact Markov/MTP token agreement into MTP confidence."""

        markov = self._shadow_markov_proposal
        self._last_consensus_agreement_tokens = 0
        self._last_consensus_confidence_gain = 0.0
        self._last_atlas_consensus_tokens = 0
        self._last_atlas_consensus_confidence_gain = 0.0
        if markov is None:
            return mtp
        if len(markov.token_ids) != len(mtp.token_ids):
            raise Qwen38MarkovMtpDraftError(
                "Markov/MTP consensus proposal widths disagree"
            )
        language_rows: tuple[object, ...] = ()
        language_vote = getattr(
            self.markov_provider,
            "language_evidence_for_pending",
            None,
        )
        if not callable(language_vote):
            language_vote = getattr(
                self.markov_provider,
                "atlas_evidence_for_pending",
                None,
            )
        if callable(language_vote):
            raw_rows = language_vote(mtp.token_ids)
            if not isinstance(raw_rows, tuple) or (
                raw_rows and len(raw_rows) != len(mtp.token_ids)
            ):
                raise Qwen38MarkovMtpDraftError(
                    "language/MTP consensus evidence width changed"
                )
            language_rows = raw_rows

        agreements = 0
        for index, (markov_token, mtp_token) in enumerate(
            zip(markov.token_ids, mtp.token_ids, strict=True)
        ):
            if markov_token != mtp_token or mtp.token_confidences[index] <= 0.0:
                break
            agreements += 1

        fused = list(mtp.token_confidences)
        agreement_gain = 0.0
        atlas_gain = 0.0
        atlas_tokens = 0
        online_gain = 0.0
        online_tokens = 0
        for index, previous in enumerate(mtp.token_confidences):
            if previous <= 0.0:
                continue
            agreement_evidence = 0.0
            if index < agreements:
                markov_confidence = float(markov.token_confidences[index])
                markov_disagreement = min(
                    1.0,
                    max(0.0, float(markov.token_disagreements[index])),
                )
                surplus = max(0.0, 2.0 * markov_confidence - 1.0)
                agreement_evidence = 0.5 * surplus * (1.0 - markov_disagreement)
            after_agreement = min(
                0.999,
                1.0 - (1.0 - previous) * (1.0 - agreement_evidence),
            )
            agreement_gain += max(0.0, after_agreement - previous)

            atlas_evidence = 0.0
            online_evidence = 0.0
            if language_rows:
                row = language_rows[index]
                raw_score = getattr(row, "score", None)
                raw_token = getattr(row, "token_id", None)
                raw_support = getattr(row, "support", None)
                raw_atlas_score = getattr(row, "atlas_score", raw_score)
                raw_online_score = getattr(row, "online_score", 0.0)
                raw_atlas_support = getattr(row, "atlas_support", raw_support)
                raw_online_support = getattr(row, "online_support", 0)
                if (
                    isinstance(raw_score, bool)
                    or not isinstance(raw_score, (int, float))
                    or not math.isfinite(float(raw_score))
                    or not 0.0 <= float(raw_score) <= 1.0
                    or raw_token != mtp.token_ids[index]
                    or isinstance(raw_support, bool)
                    or not isinstance(raw_support, int)
                    or raw_support < 0
                    or isinstance(raw_atlas_score, bool)
                    or not isinstance(raw_atlas_score, (int, float))
                    or not math.isfinite(float(raw_atlas_score))
                    or not 0.0 <= float(raw_atlas_score) <= 1.0
                    or isinstance(raw_online_score, bool)
                    or not isinstance(raw_online_score, (int, float))
                    or not math.isfinite(float(raw_online_score))
                    or not 0.0 <= float(raw_online_score) <= 1.0
                    or isinstance(raw_atlas_support, bool)
                    or not isinstance(raw_atlas_support, int)
                    or raw_atlas_support < 0
                    or isinstance(raw_online_support, bool)
                    or not isinstance(raw_online_support, int)
                    or raw_online_support < 0
                ):
                    raise Qwen38MarkovMtpDraftError(
                        "language/MTP consensus evidence is invalid"
                    )
                if raw_atlas_support > 0 and raw_atlas_score > 0.0:
                    atlas_evidence = (
                        ATLAS_MTP_CONSENSUS_STRENGTH * float(raw_atlas_score)
                    )
                if raw_online_support > 0 and raw_online_score > 0.0:
                    online_evidence = (
                        ONLINE_MTP_CONSENSUS_STRENGTH * float(raw_online_score)
                    )
            after_atlas = min(
                0.999,
                1.0 - (1.0 - after_agreement) * (1.0 - atlas_evidence),
            )
            token_atlas_gain = max(0.0, after_atlas - after_agreement)
            atlas_gain += token_atlas_gain
            atlas_tokens += int(token_atlas_gain > 0.0)
            fused[index] = min(
                0.999,
                1.0 - (1.0 - after_atlas) * (1.0 - online_evidence),
            )
            token_online_gain = max(0.0, fused[index] - after_atlas)
            online_gain += token_online_gain
            online_tokens += int(token_online_gain > 0.0)
        self._last_consensus_agreement_tokens = agreements
        self._last_consensus_confidence_gain = agreement_gain
        if agreements:
            self._consensus_rounds += 1
            self._consensus_agreement_tokens += agreements
            self._consensus_confidence_gain += agreement_gain
        self._last_atlas_consensus_tokens = atlas_tokens
        self._last_atlas_consensus_confidence_gain = atlas_gain
        if atlas_tokens:
            self._atlas_consensus_rounds += 1
            self._atlas_consensus_tokens += atlas_tokens
            self._atlas_consensus_confidence_gain += atlas_gain
        self._last_online_consensus_tokens = online_tokens
        self._last_online_consensus_confidence_gain = online_gain
        if online_tokens:
            self._online_consensus_rounds += 1
            self._online_consensus_tokens += online_tokens
            self._online_consensus_confidence_gain += online_gain
        if agreement_gain + atlas_gain + online_gain <= 0.0:
            return mtp
        return RollingDraftProposal.build(
            mtp.token_ids,
            fused,
            mtp.token_disagreements,
            request_window_ceiling=len(mtp.token_ids) + 1,
            provider_abi=QWEN38_MARKOV_MTP_HYBRID_PROVIDER_SCHEMA,
        )

    def _mtp_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        if not isinstance(target_hidden, torch.Tensor):
            raise TypeError("target_hidden must be a torch.Tensor")
        if (
            target_hidden.shape != (1, 1, self._hidden_width)
            or target_hidden.device != self._hidden_device
            or target_hidden.dtype != self._hidden_dtype
            or not target_hidden.is_floating_point()
        ):
            raise Qwen38MarkovMtpDraftError(
                "MTP proposal hidden does not match the target history"
            )
        return target_hidden.detach().clone().contiguous()

    def propose_round_state(
        self,
        history: tuple[int, ...],
        known_token: int,
        target_hidden: torch.Tensor,
        /,
    ) -> RollingDraftProposal:
        self._require_open_request()
        if self._pending_provider is not None:
            raise Qwen38MarkovMtpDraftError(
                "previous hybrid proposal was not reconciled"
            )
        round_hidden = self._mtp_hidden(target_hidden)
        self._round_target_hidden = round_hidden
        selected = self._markov_round_or_handoff(
            history,
            known_token,
            round_hidden,
        )
        if selected is not None:
            self._pending_provider = "markov"
            return selected
        markov_proposal = self._shadow_markov_proposal
        if markov_proposal is None:
            raise Qwen38MarkovMtpDraftError(
                "provider tournament lost the Markov proposal"
            )
        request_width = len(markov_proposal.token_ids)
        if request_width < 3:
            self._shadow_markov_proposal = None
            self._pending_provider = "markov"
            return self._commit_markov_selection(markov_proposal)
        qwen35_proposal = self._propose_qwen35(
            history,
            known_token,
            request_width=request_width,
        )
        mtp_proposal: RollingDraftProposal | None = None
        if qwen35_proposal is None:
            mtp = self._prepare_mtp(history)
            if mtp is not None:
                mtp_proposal = mtp.propose_round_state(
                    history,
                    known_token,
                    round_hidden.detach().clone(),
                )
                if not isinstance(mtp_proposal, RollingDraftProposal):
                    raise Qwen38MarkovMtpDraftError(
                        "selected provider returned no RollingDraftProposal"
                    )
                mtp_proposal = self._fuse_mtp_consensus(mtp_proposal)
        self._shadow_markov_proposal = None
        if qwen35_proposal is None and mtp_proposal is None:
            self._pending_provider = "markov"
            return self._commit_markov_selection(markov_proposal)
        selected, proposal = (
            self._select_provider_tournament(
                markov_proposal,
                mtp_proposal,
                qwen35_proposal,
            )
            if qwen35_proposal is not None
            else self._select_provider_tournament(markov_proposal, mtp_proposal)
        )
        if self._selected_provider not in {None, selected}:
            self._provider_switches += 1
        self._selected_provider = selected
        self._pending_provider = selected
        self._pending_mtp_shadow = mtp_proposal is not None and selected != "mtp"
        self._pending_qwen35_shadow = (
            qwen35_proposal is not None and selected != "qwen35"
        )
        if selected == "markov":
            self._markov_selections += 1
            self._markov_rounds += 1
        elif selected == "qwen35":
            self._qwen35_selections += 1
            self._qwen35_rounds += 1
        else:
            self._mtp_selections += 1
            self._mtp_rounds += 1
        return proposal

    def propose_after_state(
        self,
        history: tuple[int, ...],
        known_token: int,
        target_hidden: torch.Tensor,
        /,
    ) -> tuple[int, ...]:
        self._require_open_request()
        if self._pending_provider is not None:
            raise Qwen38MarkovMtpDraftError(
                "previous hybrid proposal was not reconciled"
            )
        self._advance_provider_traces((known_token,))
        # Fixed short windows do not expose the K4/K8/K16 horizon contract
        # required by ``propose_round``.  They are already the cheap path, so
        # lock directly to Markov and never materialize MTP for this request.
        if self._selected_provider is None:
            self._selected_provider = "markov"
            self._selection_calls += 1
            self._markov_selections += 1
            self._switch_available = False
            self._hidden_history = None
        if self._selected_provider == "markov":
            stateful_proposal = getattr(
                self.markov_provider,
                "propose_after_state",
                None,
            )
            proposal = (
                stateful_proposal(
                    history,
                    known_token,
                    target_hidden.detach().clone(),
                )
                if callable(stateful_proposal)
                else self.markov_provider.propose_after(history, known_token)
            )
            self._markov_rounds += 1
        elif self._selected_provider == "qwen35":
            qwen35 = self._qwen35_provider
            if qwen35 is None:
                raise Qwen38MarkovMtpDraftError(
                    "selected qwen35 provider is not loaded"
                )
            proposal = tuple(qwen35.propose_after(history, known_token))[:3]
            if not proposal:
                raise Qwen38MarkovMtpDraftError(
                    "selected qwen35 provider returned no draft tokens"
                )
            self._pending_qwen35_candidate = proposal
            self._qwen35_proposed_tokens += len(proposal)
            self._qwen35_rounds += 1
        elif self._selected_provider == "mtp":
            assert self._mtp_provider is not None
            proposal = self._mtp_provider.propose_after_state(
                history,
                known_token,
                self._mtp_hidden(target_hidden),
            )
            self._mtp_rounds += 1
        else:  # pragma: no cover - guarded by the selection transition.
            raise Qwen38MarkovMtpDraftError("hybrid provider was not selected")
        self._pending_provider = self._selected_provider
        return tuple(proposal)

    def observe_verification(
        self,
        accepted_prefix_length: int,
        verified_proposals: int,
        /,
    ) -> None:
        self._require_open_request()
        pending = self._pending_provider
        if pending is None:
            raise Qwen38MarkovMtpDraftError(
                "verification has no pending hybrid proposal"
            )
        owner = (
            self.markov_provider
            if pending == "markov"
            else self._qwen35_provider
            if pending == "qwen35"
            else self._mtp_provider
        )
        callback = getattr(owner, "observe_verification", None)
        if callable(callback):
            callback(accepted_prefix_length, verified_proposals)

    def observe_virtual_verification(
        self,
        accepted_prefix_length: int,
        verified_proposals: int,
        /,
    ) -> None:
        self._require_open_request()
        pending = self._pending_provider
        if pending is None:
            raise Qwen38MarkovMtpDraftError(
                "virtual verification has no pending hybrid proposal"
            )
        owner = (
            self.markov_provider
            if pending == "markov"
            else self._qwen35_provider
            if pending == "qwen35"
            else self._mtp_provider
        )
        callback = getattr(owner, "observe_virtual_verification", None)
        if callable(callback):
            callback(accepted_prefix_length, verified_proposals)

    def observe_teacher_verification(
        self,
        position: int,
        outcome: bool,
        /,
    ) -> None:
        self._require_open_request()
        if self._pending_provider != "mtp" or self._mtp_provider is None:
            return
        callback = getattr(self._mtp_provider, "observe_teacher_verification", None)
        if callable(callback):
            callback(position, outcome)

    def reconcile_prefix(self, history: tuple[int, ...], /) -> None:
        self._require_open_request()
        pending = self._pending_provider
        if pending is None:
            raise Qwen38MarkovMtpDraftError(
                "reconciliation has no pending hybrid proposal"
            )
        self._validate_history(history)
        proposal_tokens = self._round_proposal_tokens(history)
        owner = (
            self.markov_provider
            if pending == "markov"
            else self._qwen35_provider
            if pending == "qwen35"
            else self._mtp_provider
        )
        if owner is None:  # pragma: no cover - selection invariant.
            raise Qwen38MarkovMtpDraftError("hybrid provider was not selected")
        owner.reconcile_prefix(history)
        if pending != "markov":
            self.markov_provider.reconcile_external_prefix(history)
            self._markov_external_feedback_rounds += 1
        if pending != "mtp" and self._pending_mtp_shadow:
            mtp = self._mtp_provider
            if mtp is None:
                raise Qwen38MarkovMtpDraftError(
                    "provider tournament lost its pending MTP shadow"
                )
            mtp.reconcile_prefix(history)
        if pending != "qwen35" and self._pending_qwen35_shadow:
            qwen35 = self._qwen35_provider
            if qwen35 is None:
                raise Qwen38MarkovMtpDraftError(
                    "provider tournament lost its pending qwen35 shadow"
                )
            qwen35.reconcile_prefix(history)
        elif pending != "qwen35" and self._qwen35_provider is not None:
            self._qwen35_provider.advance_confirmed_prefix(history)
        candidate = self._pending_qwen35_candidate
        if candidate is not None:
            for expected, actual in zip(candidate, proposal_tokens, strict=False):
                if expected != actual:
                    break
                self._qwen35_accepted_tokens += 1
        self._start_pending_provider_trace()
        self._advance_provider_traces(proposal_tokens)
        self._request_history = history
        self._pending_provider = None
        self._round_target_hidden = None
        self._pending_mtp_shadow = False
        self._pending_qwen35_shadow = False
        self._pending_qwen35_candidate = None
        if pending != "mtp":
            # A no-state caller cannot advance an already loaded MTP cache.
            # Keep serving Markov/Qwen3.5 without reusing stale target state.
            self._switch_available = False
            self._hidden_history = None

    def reconcile_prefix_state(
        self,
        history: tuple[int, ...],
        committed_hidden: torch.Tensor,
        /,
    ) -> None:
        """Reconcile one prefix and retain its exact target-hidden extension."""

        self._require_open_request()
        pending = self._pending_provider
        if pending is None:
            raise Qwen38MarkovMtpDraftError(
                "reconciliation has no pending hybrid proposal"
            )
        proposal_tokens = self._round_proposal_tokens(history)
        fragment = self._state_fragment(history, committed_hidden)
        caller_snapshot = committed_hidden.detach().clone()
        round_hidden = self._round_target_hidden
        owner = (
            self.markov_provider
            if pending == "markov"
            else self._qwen35_provider
            if pending == "qwen35"
            else self._mtp_provider
        )
        if owner is None:  # pragma: no cover - pending-provider invariant.
            raise Qwen38MarkovMtpDraftError("hybrid provider was not selected")
        stateful_reconcile = getattr(owner, "reconcile_prefix_state", None)
        if callable(stateful_reconcile):
            stateful_reconcile(history, fragment.detach().clone())
        else:
            owner.reconcile_prefix(history)
        if pending != "markov":
            self.markov_provider.reconcile_external_prefix(history)
            self._markov_external_feedback_rounds += 1
        if pending != "mtp" and self._mtp_provider is not None and self._pending_mtp_shadow:
            stateful_mtp_reconcile = getattr(
                self._mtp_provider,
                "reconcile_prefix_state",
                None,
            )
            if not callable(stateful_mtp_reconcile):
                raise Qwen38MarkovMtpDraftError(
                    "pending MTP shadow lacks stateful reconciliation"
            )
            stateful_mtp_reconcile(history, fragment.detach().clone())
        elif pending != "mtp" and self._mtp_provider is not None:
            if round_hidden is None:
                raise Qwen38MarkovMtpDraftError(
                    "Markov round lost its previous target hidden row"
                )
            self._mtp_provider.advance_confirmed_prefix_state(
                history,
                round_hidden.detach().clone(),
                fragment.detach().clone(),
            )
        if pending != "qwen35" and self._pending_qwen35_shadow:
            qwen35 = self._qwen35_provider
            if qwen35 is None:
                raise Qwen38MarkovMtpDraftError(
                    "provider tournament lost its pending qwen35 shadow"
                )
            qwen35.reconcile_prefix(history)
        elif pending != "qwen35" and self._qwen35_provider is not None:
            self._qwen35_provider.advance_confirmed_prefix(history)
        if not torch.equal(committed_hidden, caller_snapshot):
            raise Qwen38MarkovMtpDraftError(
                "hybrid reconciliation mutated its hidden argument"
            )
        if (
            pending != "mtp"
            and self._mtp_provider is None
            and self._switch_available
        ):
            hidden_history = self._hidden_history
            if hidden_history is None:
                self._switch_available = False
            else:
                self._hidden_history = torch.cat(
                    (hidden_history, fragment),
                    dim=1,
                ).contiguous()
                self._hidden_history_rows = int(self._hidden_history.shape[1])
                self._hidden_history_bytes = (
                    self._hidden_history.numel() * self._hidden_history.element_size()
                )
        candidate = self._pending_qwen35_candidate
        if candidate is not None:
            for expected, actual in zip(candidate, proposal_tokens, strict=False):
                if expected != actual:
                    break
                self._qwen35_accepted_tokens += 1
        self._start_pending_provider_trace()
        self._advance_provider_traces(proposal_tokens)
        self._request_history = history
        self._boundary_history = history
        self._boundary_target_hidden = (
            fragment[:, -1:].detach().clone().contiguous()
        )
        self._pending_provider = None
        self._round_target_hidden = None
        self._pending_mtp_shadow = False
        self._pending_qwen35_shadow = False
        self._pending_qwen35_candidate = None

    def observe_final(self, history: tuple[int, ...], /) -> None:
        self._require_open_request()
        if self._pending_provider is not None:
            raise Qwen38MarkovMtpDraftError(
                "cannot finalize an unreconciled hybrid proposal"
            )
        current = self._request_history
        if (
            current is None
            or len(history) < len(current)
            or history[: len(current)] != current
        ):
            raise Qwen38MarkovMtpDraftError(
                "final hybrid history changed its confirmed prefix"
            )
        original_traces = list(self._provider_traces)
        original_feedback_tokens = self._provider_trace_feedback_tokens
        policy_snapshot = None
        snapshot_policy = getattr(
            self.markov_provider,
            "snapshot_provider_policy_feedback",
            None,
        )
        restore_policy = getattr(
            self.markov_provider,
            "restore_provider_policy_feedback",
            None,
        )
        if callable(snapshot_policy) and callable(restore_policy):
            policy_snapshot = snapshot_policy()
        markov_completed = False
        try:
            self._advance_provider_traces(history[len(current) :])
            self._provider_traces.clear()
            # Markov owns the shared persistent tournament transaction. MTP
            # finalization runs only after that transaction commits.
            self.markov_provider.observe_final(history)
            markov_completed = True
            if self._qwen35_provider is not None:
                self._qwen35_provider.observe_final(history)
            if self._mtp_provider is not None:
                self._mtp_provider.observe_final(history)
        except Exception:
            if markov_completed:
                self._request_completed = True
                self._switch_available = False
                self._round_target_hidden = None
                raise
            self._provider_traces = original_traces
            self._provider_trace_feedback_tokens = original_feedback_tokens
            if policy_snapshot is not None:
                try:
                    restore_policy(policy_snapshot)
                except Exception as restore_exc:
                    raise Qwen38MarkovMtpDraftError(
                        "provider policy feedback rollback failed"
                    ) from restore_exc
            raise
        self._request_completed = True
        self._switch_available = False
        self._round_target_hidden = None

    def export_mtp_carry(
        self,
        history: tuple[int, ...],
        /,
    ) -> Qwen35MtpCarry | None:
        if not self._request_completed or self._pending_provider is not None:
            raise Qwen38MarkovMtpDraftError(
                "hybrid request must complete before MTP carry export"
            )
        mtp = self._mtp_provider
        boundary_history = self._boundary_history
        boundary_hidden = self._boundary_target_hidden
        if boundary_history != history or boundary_hidden is None:
            raise Qwen38MarkovMtpDraftError(
                "hybrid MTP carry differs from the target boundary"
            )
        if mtp is None:
            hidden_history = self._hidden_history
            if hidden_history is None:
                return None
            mtp = self._load_mtp()
            mtp.begin_request_state(
                history,
                self._mtp_bootstrap_hidden(history, hidden_history),
            )
            self._hidden_history = None
        export = getattr(mtp, "export_carry", None)
        if not callable(export):
            return None
        carry = export(history)
        if isinstance(carry, Qwen35MtpCarry) and not torch.equal(
            carry.last_target_hidden,
            boundary_hidden,
        ):
            raise Qwen38MarkovMtpDraftError(
                "exported MTP carry target hidden differs from target boundary"
            )
        return carry

    def metrics(self) -> Qwen38MarkovMtpDraftMetrics:
        markov = _metrics_record(self.markov_provider)
        qwen35 = _metrics_record(self._qwen35_provider)
        mtp = _metrics_record(self._mtp_provider)
        external_source_body_bytes = _nonnegative_metric(
            qwen35,
            "source_body_bytes",
        )
        external_linear_calls = _nonnegative_metric(qwen35, "linear_calls")
        mtp_shared_source_body_bytes = _nonnegative_metric(
            mtp,
            "source_body_bytes",
        )
        mtp_shared_linear_calls = _nonnegative_metric(mtp, "linear_calls")
        return Qwen38MarkovMtpDraftMetrics(
            schema=QWEN38_MARKOV_MTP_HYBRID_PROVIDER_SCHEMA,
            selected_provider=self._selected_provider,
            selection_calls=self._selection_calls,
            markov_selections=self._markov_selections,
            qwen35_selections=self._qwen35_selections,
            mtp_selections=self._mtp_selections,
            markov_rounds=self._markov_rounds,
            qwen35_rounds=self._qwen35_rounds,
            mtp_rounds=self._mtp_rounds,
            markov_external_feedback_rounds=self._markov_external_feedback_rounds,
            provider_switches=self._provider_switches,
            provider_tournament_calls=self._provider_tournament_calls,
            provider_tournament_markov_selections=(
                self._provider_tournament_markov_selections
            ),
            provider_tournament_qwen35_selections=(
                self._provider_tournament_qwen35_selections
            ),
            provider_tournament_mtp_selections=(
                self._provider_tournament_mtp_selections
            ),
            provider_trace_created=self._provider_trace_created,
            provider_trace_active=len(self._provider_traces),
            provider_trace_feedback_tokens=self._provider_trace_feedback_tokens,
            qwen35_wave_gate_checks=self._qwen35_wave_gate_checks,
            qwen35_wave_gate_unknown=self._qwen35_wave_gate_unknown,
            qwen35_wave_gate_passes=self._qwen35_wave_gate_passes,
            qwen35_wave_gate_rejections=self._qwen35_wave_gate_rejections,
            last_qwen35_complete_wave_probability=(
                self._last_qwen35_complete_wave_probability
            ),
            mtp_wave_gate_checks=self._mtp_wave_gate_checks,
            mtp_wave_gate_unknown=self._mtp_wave_gate_unknown,
            mtp_wave_gate_passes=self._mtp_wave_gate_passes,
            mtp_wave_gate_rejections=self._mtp_wave_gate_rejections,
            last_mtp_complete_wave_probability=(
                self._last_mtp_complete_wave_probability
            ),
            qwen35_init_failures=self._qwen35_init_failures,
            mtp_init_failures=self._mtp_init_failures,
            qwen35_proposed_tokens=self._qwen35_proposed_tokens,
            qwen35_accepted_tokens=self._qwen35_accepted_tokens,
            consensus_rounds=self._consensus_rounds,
            consensus_agreement_tokens=self._consensus_agreement_tokens,
            consensus_confidence_gain=self._consensus_confidence_gain,
            last_consensus_agreement_tokens=(
                self._last_consensus_agreement_tokens
            ),
            last_consensus_confidence_gain=self._last_consensus_confidence_gain,
            atlas_consensus_rounds=self._atlas_consensus_rounds,
            atlas_consensus_tokens=self._atlas_consensus_tokens,
            atlas_consensus_confidence_gain=(
                self._atlas_consensus_confidence_gain
            ),
            last_atlas_consensus_tokens=self._last_atlas_consensus_tokens,
            last_atlas_consensus_confidence_gain=(
                self._last_atlas_consensus_confidence_gain
            ),
            online_consensus_rounds=self._online_consensus_rounds,
            online_consensus_tokens=self._online_consensus_tokens,
            online_consensus_confidence_gain=(
                self._online_consensus_confidence_gain
            ),
            last_online_consensus_tokens=self._last_online_consensus_tokens,
            last_online_consensus_confidence_gain=(
                self._last_online_consensus_confidence_gain
            ),
            hidden_history_rows=self._hidden_history_rows,
            hidden_history_bytes=self._hidden_history_bytes,
            switch_available=self._switch_available,
            external_source_body_bytes=external_source_body_bytes,
            external_linear_calls=external_linear_calls,
            mtp_shared_source_body_bytes=mtp_shared_source_body_bytes,
            mtp_shared_linear_calls=mtp_shared_linear_calls,
            source_body_bytes=(
                _nonnegative_metric(markov, "source_body_bytes")
                + external_source_body_bytes
                + mtp_shared_source_body_bytes
            ),
            linear_calls=(
                _nonnegative_metric(markov, "linear_calls")
                + external_linear_calls
                + mtp_shared_linear_calls
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
            pending=self._pending_provider is not None,
            closed=self._closed,
            markov=markov,
            qwen35=qwen35,
            mtp=mtp,
        )

    def close(self) -> None:
        if self._closed:
            return
        failure: BaseException | None = None
        try:
            if self._qwen35_provider is not None:
                self._qwen35_provider.close()
        except BaseException as exc:
            failure = exc
        try:
            if self._mtp_provider is not None:
                self._mtp_provider.close()
        except BaseException as exc:
            if failure is None:
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
        self._hidden_history = None
        self._round_target_hidden = None
        self._boundary_history = None
        self._boundary_target_hidden = None
        self._shadow_markov_proposal = None
        self._pending_provider_candidates = None
        self._provider_traces.clear()
        self._pending_mtp_shadow = False
        self._pending_qwen35_shadow = False
        self._pending_qwen35_candidate = None
        self._switch_available = False
        self._closed = True
        if failure is not None:
            raise failure


__all__ = [
    "ATLAS_MTP_CONSENSUS_STRENGTH",
    "ONLINE_MTP_CONSENSUS_STRENGTH",
    "MTP_COMPLETE_WAVE_MIN_PROBABILITY",
    "MTP_COMPLETE_WAVE_PREFIX_TOKENS",
    "PROVIDER_TOURNAMENT_DISCOUNT",
    "MARKOV_MTP_WINDOW_WORK_COSTS",
    "QWEN38_MARKOV_MTP_HYBRID_PROVIDER_SCHEMA",
    "Qwen38MarkovMtpDraftError",
    "Qwen38MarkovMtpDraftMetrics",
    "Qwen38MarkovMtpDraftProvider",
]
