"""Lightweight proposal evidence for adaptive rolling draft windows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from typing import Sequence


ROLLING_DRAFT_HORIZON_SCHEMA = "immer.qwen3.8-rolling-draft-horizon/v2"
ROLLING_DRAFT_PROPOSAL_SCHEMA = "immer.qwen3.8-rolling-draft-proposal/v2"
ROUND_WINDOW_POLICY_SCHEMA = "immer.qwen3.8-round-window-policy/v2"
STANDARD_ROLLING_WINDOWS = (4, 8, 16)
ROUND_ROLLING_WINDOWS = (1, *STANDARD_ROLLING_WINDOWS)
_HEX = frozenset("0123456789abcdef")


def _finite(
    value: object,
    *,
    field: str,
    lower: float = 0.0,
    upper: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite number")
    result = float(value)
    if (
        not math.isfinite(result)
        or result < lower
        or (upper is not None and result > upper)
    ):
        raise ValueError(f"{field} is outside its finite bound")
    return result


def _uint(value: object, *, field: str, positive: bool = False) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < int(positive)
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be a {qualifier} integer")
    return value


def _token_tuple(value: Sequence[int], *, field: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field} must be an integer sequence")
    result = tuple(value)
    if not result or any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0
        for token in result
    ):
        raise ValueError(f"{field} contains an invalid token")
    return result


def _token_sha256(token_ids: Sequence[int]) -> str:
    body = json.dumps(
        list(token_ids),
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


@dataclass(frozen=True, slots=True)
class RollingDraftHorizon:
    window: int
    proposed_draft_tokens: int
    expected_accepted_draft_tokens: float
    prefix_survival_probabilities: tuple[float, ...]
    mean_confidence: float
    minimum_confidence: float
    mean_disagreement: float
    phrase_covered_tokens: int
    work_proxy: float
    utility: float

    def __post_init__(self) -> None:
        if self.window not in ROUND_ROLLING_WINDOWS:
            raise ValueError("rolling horizon window is invalid")
        if self.proposed_draft_tokens != self.window - 1:
            raise ValueError("rolling horizon proposal width is invalid")
        _finite(
            self.expected_accepted_draft_tokens,
            field="expected_accepted_draft_tokens",
            upper=float(self.proposed_draft_tokens),
        )
        survival = tuple(
            _finite(value, field="prefix_survival_probability", upper=1.0)
            for value in self.prefix_survival_probabilities
        )
        if (
            len(survival) != self.proposed_draft_tokens
            or any(
                right > left
                for left, right in zip(survival, survival[1:], strict=False)
            )
            or not math.isclose(
                sum(survival),
                self.expected_accepted_draft_tokens,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("rolling horizon survival curve is invalid")
        _finite(self.mean_confidence, field="mean_confidence", upper=1.0)
        _finite(self.minimum_confidence, field="minimum_confidence", upper=1.0)
        _finite(self.mean_disagreement, field="mean_disagreement")
        _uint(self.phrase_covered_tokens, field="phrase_covered_tokens")
        if self.phrase_covered_tokens > self.proposed_draft_tokens:
            raise ValueError("phrase coverage exceeds the rolling horizon")
        _finite(self.work_proxy, field="work_proxy", lower=1.0)
        _finite(self.utility, field="utility")
        object.__setattr__(self, "prefix_survival_probabilities", survival)

    def to_dict(self) -> dict[str, object]:
        return {"schema": ROLLING_DRAFT_HORIZON_SCHEMA, **asdict(self)}


@dataclass(frozen=True, slots=True)
class RoundWindowPolicy:
    selector: str
    request_window_ceiling: int
    remaining_tokens: int
    eligible_windows: tuple[int, ...]
    chosen_window: int
    horizons: tuple[RollingDraftHorizon, ...]
    provider_proposal_sha256: str
    provider_abi: str
    phrase_source: str | None
    phrase_support: int
    phrase_confidence: float
    phrase_width: int

    def __post_init__(self) -> None:
        if self.selector != "markov-prefix-utility/v2":
            raise ValueError("round-window selector is invalid")
        if self.request_window_ceiling not in STANDARD_ROLLING_WINDOWS:
            raise ValueError("request window ceiling is invalid")
        _uint(self.remaining_tokens, field="remaining_tokens", positive=True)
        eligible = tuple(self.eligible_windows)
        if (
            tuple(sorted(set(eligible))) != eligible
            or any(
                window not in ROUND_ROLLING_WINDOWS
                or window > self.request_window_ceiling
                for window in eligible
            )
        ):
            raise ValueError("eligible round windows are invalid")
        if (
            self.chosen_window not in ROUND_ROLLING_WINDOWS
            or self.chosen_window > self.request_window_ceiling
        ):
            raise ValueError("chosen round window is invalid")
        if eligible and self.chosen_window not in eligible:
            raise ValueError("chosen round window is not eligible")
        horizons = tuple(self.horizons)
        if tuple(row.window for row in horizons) != eligible:
            raise ValueError("round policy horizons differ from eligible windows")
        visible_drafts = max(0, self.remaining_tokens - 1)

        def utility(row: RollingDraftHorizon) -> float:
            expected = sum(
                row.prefix_survival_probabilities[:visible_drafts]
            )
            return (1.0 + expected) / row.work_proxy

        expected_window = max(
            horizons,
            key=lambda row: (
                utility(row),
                sum(row.prefix_survival_probabilities[:visible_drafts]),
                -row.window,
            ),
        ).window
        if self.chosen_window != expected_window:
            raise ValueError("chosen round window differs from prefix utility")
        if (
            not isinstance(self.provider_proposal_sha256, str)
            or len(self.provider_proposal_sha256) != 64
            or bool(set(self.provider_proposal_sha256) - _HEX)
            or not isinstance(self.provider_abi, str)
            or not self.provider_abi
        ):
            raise ValueError("round policy provider identity is invalid")
        if self.phrase_source not in {None, "global", "dialect"}:
            raise ValueError("round policy phrase source is invalid")
        _uint(self.phrase_support, field="phrase_support")
        _finite(self.phrase_confidence, field="phrase_confidence", upper=1.0)
        _uint(self.phrase_width, field="phrase_width")
        if self.phrase_width > 15:
            raise ValueError("round policy phrase width exceeds 15")
        object.__setattr__(self, "eligible_windows", eligible)
        object.__setattr__(self, "horizons", horizons)

    def to_dict(self) -> dict[str, object]:
        return {
            "chosen_window": self.chosen_window,
            "eligible_windows": list(self.eligible_windows),
            "horizons": [row.to_dict() for row in self.horizons],
            "phrase_confidence": self.phrase_confidence,
            "phrase_source": self.phrase_source,
            "phrase_support": self.phrase_support,
            "phrase_width": self.phrase_width,
            "provider_abi": self.provider_abi,
            "provider_proposal_sha256": self.provider_proposal_sha256,
            "remaining_utilities": {
                str(row.window): (
                    1.0
                    + sum(
                        row.prefix_survival_probabilities[
                            : max(0, self.remaining_tokens - 1)
                        ]
                    )
                )
                / row.work_proxy
                for row in self.horizons
            },
            "remaining_tokens": self.remaining_tokens,
            "request_window_ceiling": self.request_window_ceiling,
            "schema": ROUND_WINDOW_POLICY_SCHEMA,
            "selector": self.selector,
        }

    def matches_provider_tokens(self, token_ids: Sequence[int]) -> bool:
        return self.provider_proposal_sha256 == _token_sha256(token_ids)


@dataclass(frozen=True, slots=True)
class RollingDraftProposal:
    token_ids: tuple[int, ...]
    token_confidences: tuple[float, ...]
    token_disagreements: tuple[float, ...]
    horizons: tuple[RollingDraftHorizon, ...]
    recommended_window: int
    provider_abi: str
    phrase_source: str | None = None
    phrase_support: int = 0
    phrase_confidence: float = 0.0
    phrase_width: int = 0

    def __post_init__(self) -> None:
        token_ids = _token_tuple(self.token_ids, field="rolling proposal token_ids")
        confidences = tuple(
            _finite(value, field="token_confidence", upper=1.0)
            for value in self.token_confidences
        )
        disagreements = tuple(
            _finite(value, field="token_disagreement")
            for value in self.token_disagreements
        )
        if len(token_ids) > 15 or not (
            len(token_ids) == len(confidences) == len(disagreements)
        ):
            raise ValueError("rolling proposal evidence widths disagree")
        horizons = tuple(self.horizons)
        if (
            not horizons
            or tuple(row.window for row in horizons)
            != tuple(sorted(row.window for row in horizons))
            or any(row.proposed_draft_tokens > len(token_ids) for row in horizons)
            or self.recommended_window not in {row.window for row in horizons}
        ):
            raise ValueError("rolling proposal horizons are invalid")
        if not isinstance(self.provider_abi, str) or not self.provider_abi:
            raise ValueError("rolling proposal provider ABI is missing")
        if self.phrase_source not in {None, "global", "dialect"}:
            raise ValueError("rolling proposal phrase source is invalid")
        _uint(self.phrase_support, field="phrase_support")
        _finite(self.phrase_confidence, field="phrase_confidence", upper=1.0)
        _uint(self.phrase_width, field="phrase_width")
        if self.phrase_width > len(token_ids):
            raise ValueError("phrase width exceeds provider proposal")
        object.__setattr__(self, "token_ids", token_ids)
        object.__setattr__(self, "token_confidences", confidences)
        object.__setattr__(self, "token_disagreements", disagreements)
        object.__setattr__(self, "horizons", horizons)

    @classmethod
    def build(
        cls,
        token_ids: Sequence[int],
        token_confidences: Sequence[float],
        token_disagreements: Sequence[float],
        *,
        request_window_ceiling: int,
        provider_abi: str,
        phrase_source: str | None = None,
        phrase_support: int = 0,
        phrase_confidence: float = 0.0,
        phrase_width: int = 0,
    ) -> "RollingDraftProposal":
        if request_window_ceiling not in STANDARD_ROLLING_WINDOWS:
            raise ValueError("request window ceiling is invalid")
        tokens = _token_tuple(token_ids, field="rolling proposal token_ids")
        confidences = tuple(token_confidences)
        disagreements = tuple(token_disagreements)
        if len(tokens) != request_window_ceiling - 1:
            raise ValueError("rolling proposal differs from its request ceiling")
        if not len(tokens) == len(confidences) == len(disagreements):
            raise ValueError("rolling proposal evidence widths disagree")
        _uint(phrase_support, field="phrase_support")
        _finite(phrase_confidence, field="phrase_confidence", upper=1.0)
        _uint(phrase_width, field="phrase_width")
        if not 0 <= phrase_width <= len(tokens):
            raise ValueError("phrase width exceeds rolling proposal")
        support_strength = 1.0 - math.exp(-float(phrase_support) / 1.5)
        phrase_probability = phrase_confidence * support_strength
        horizons = [
            RollingDraftHorizon(
                window=1,
                proposed_draft_tokens=0,
                expected_accepted_draft_tokens=0.0,
                prefix_survival_probabilities=(),
                mean_confidence=0.0,
                minimum_confidence=0.0,
                mean_disagreement=0.0,
                phrase_covered_tokens=0,
                work_proxy=1.0,
                utility=1.0,
            )
        ]
        for window in STANDARD_ROLLING_WINDOWS:
            if window > request_window_ceiling:
                continue
            width = window - 1
            survival = 1.0
            expected = 0.0
            effective_probabilities = []
            for index in range(width):
                probability = float(confidences[index]) * (
                    1.0 - 0.5 * min(1.0, float(disagreements[index]))
                )
                if index < phrase_width:
                    probability = max(probability, phrase_probability)
                probability = max(0.0, min(0.999, probability))
                effective_probabilities.append(probability)
                survival *= probability
                expected += survival
            work_proxy = 1.0 + width / 16.0
            horizons.append(
                RollingDraftHorizon(
                    window=window,
                    proposed_draft_tokens=width,
                    expected_accepted_draft_tokens=expected,
                    prefix_survival_probabilities=tuple(
                        math.prod(effective_probabilities[: index + 1])
                        for index in range(width)
                    ),
                    mean_confidence=sum(effective_probabilities) / width,
                    minimum_confidence=min(effective_probabilities),
                    mean_disagreement=sum(disagreements[:width]) / width,
                    phrase_covered_tokens=min(phrase_width, width),
                    work_proxy=work_proxy,
                    utility=(1.0 + expected) / work_proxy,
                )
            )
        if not horizons:
            raise ValueError("request ceiling admits no standard rolling horizon")
        recommended = max(
            horizons,
            key=lambda row: (
                row.utility,
                row.expected_accepted_draft_tokens,
                -row.window,
            ),
        ).window
        return cls(
            token_ids=tokens,
            token_confidences=confidences,
            token_disagreements=disagreements,
            horizons=tuple(horizons),
            recommended_window=recommended,
            provider_abi=provider_abi,
            phrase_source=phrase_source,
            phrase_support=phrase_support,
            phrase_confidence=phrase_confidence,
            phrase_width=phrase_width,
        )

    def select_window(
        self,
        *,
        request_window_ceiling: int,
        remaining_tokens: int,
        window_work_costs: Mapping[int, float] | None = None,
    ) -> RoundWindowPolicy:
        if request_window_ceiling != len(self.token_ids) + 1:
            raise ValueError("proposal differs from request window ceiling")
        _uint(remaining_tokens, field="remaining_tokens", positive=True)
        visible_drafts = max(0, remaining_tokens - 1)
        eligible = tuple(
            row for row in self.horizons
            if row.window <= request_window_ceiling
        )
        if window_work_costs is not None:
            costs = dict(window_work_costs)
            if any(
                window not in ROUND_ROLLING_WINDOWS
                for window in costs
            ):
                raise ValueError("round window work costs contain an invalid window")
            adjusted = []
            for row in eligible:
                cost = _finite(
                    costs.get(row.window, row.work_proxy),
                    field="round_window_work_cost",
                    lower=1.0,
                )
                adjusted.append(
                    replace(
                        row,
                        work_proxy=cost,
                        utility=(1.0 + row.expected_accepted_draft_tokens) / cost,
                    )
                )
            eligible = tuple(adjusted)

        def remaining_utility(row: RollingDraftHorizon) -> float:
            expected = sum(
                row.prefix_survival_probabilities[:visible_drafts]
            )
            return (1.0 + expected) / row.work_proxy

        chosen = max(
            eligible,
            key=lambda row: (
                remaining_utility(row),
                sum(row.prefix_survival_probabilities[:visible_drafts]),
                -row.window,
            ),
        ).window
        return RoundWindowPolicy(
            selector="markov-prefix-utility/v2",
            request_window_ceiling=request_window_ceiling,
            remaining_tokens=remaining_tokens,
            eligible_windows=tuple(row.window for row in eligible),
            chosen_window=chosen,
            horizons=eligible,
            provider_proposal_sha256=_token_sha256(self.token_ids),
            provider_abi=self.provider_abi,
            phrase_source=self.phrase_source,
            phrase_support=self.phrase_support,
            phrase_confidence=self.phrase_confidence,
            phrase_width=self.phrase_width,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "horizons": [row.to_dict() for row in self.horizons],
            "phrase_confidence": self.phrase_confidence,
            "phrase_source": self.phrase_source,
            "phrase_support": self.phrase_support,
            "phrase_width": self.phrase_width,
            "provider_abi": self.provider_abi,
            "recommended_window": self.recommended_window,
            "schema": ROLLING_DRAFT_PROPOSAL_SCHEMA,
            "token_confidences": list(self.token_confidences),
            "token_disagreements": list(self.token_disagreements),
            "token_ids": list(self.token_ids),
        }


__all__ = [
    "ROLLING_DRAFT_HORIZON_SCHEMA",
    "ROLLING_DRAFT_PROPOSAL_SCHEMA",
    "ROUND_WINDOW_POLICY_SCHEMA",
    "ROUND_ROLLING_WINDOWS",
    "STANDARD_ROLLING_WINDOWS",
    "RollingDraftHorizon",
    "RollingDraftProposal",
    "RoundWindowPolicy",
]
