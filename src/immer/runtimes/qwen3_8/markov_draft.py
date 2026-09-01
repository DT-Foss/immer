"""Sparse variable-order Markov drafting over native Qwen token IDs.

This is the runtime-native transfer of FERTIG's ``TransitionFingerprint``
equation: suffix counts, interpolation, and deterministic backoff operate on
fixed-width Qwen token symbols, while the target remains the sole authority.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
from typing import Sequence
import zlib

import torch

from .contextual_continuation import (
    ContextualCandidate,
    ContextualCandidateFeedback,
    ContextualCapture,
    ContextualContinuationBank,
    ContextualKey,
    MAX_CONTINUATION_TOKENS,
)
from .draft_protocol import RollingDraftProposal
from .layer_contextual_continuation import (
    LayerContextualContinuationBank,
    LayerContextualContinuationOption,
    LayerContextualContinuationTransaction,
)
from .markov_composition import (
    CompositionBounds,
    ConfirmedTokenEpisode,
    MarkovCompositionProgram,
    derive_programs,
)
from .markov_atlas import AtlasTokenEvidence, MarkovTokenAtlas

try:
    import fcntl
except ImportError:  # pragma: no cover - production targets are POSIX.
    fcntl = None  # type: ignore[assignment]

MARKOV_DRAFT_STATE_SCHEMA = "immer.qwen3.8-markov-draft-state/v13"
MARKOV_DRAFT_PROVIDER_ABI = "immer.qwen3.8-markov-draft-provider/v48"
V12_MARKOV_DRAFT_STATE_SCHEMA = "immer.qwen3.8-markov-draft-state/v12"
V11_MARKOV_DRAFT_STATE_SCHEMA = "immer.qwen3.8-markov-draft-state/v11"
V10_MARKOV_DRAFT_STATE_SCHEMA = "immer.qwen3.8-markov-draft-state/v10"
V9_MARKOV_DRAFT_STATE_SCHEMA = "immer.qwen3.8-markov-draft-state/v9"
V8_MARKOV_DRAFT_STATE_SCHEMA = "immer.qwen3.8-markov-draft-state/v8"
V7_MARKOV_DRAFT_STATE_SCHEMA = "immer.qwen3.8-markov-draft-state/v7"
V6_MARKOV_DRAFT_STATE_SCHEMA = "immer.qwen3.8-markov-draft-state/v6"
V5_MARKOV_DRAFT_STATE_SCHEMA = "immer.qwen3.8-markov-draft-state/v5"
V4_MARKOV_DRAFT_STATE_SCHEMA = "immer.qwen3.8-markov-draft-state/v4"
V3_MARKOV_DRAFT_STATE_SCHEMA = "immer.qwen3.8-markov-draft-state/v3"
V2_MARKOV_DRAFT_STATE_SCHEMA = "immer.qwen3.8-markov-draft-state/v2"
LEGACY_MARKOV_DRAFT_STATE_SCHEMA = "immer.qwen3.8-markov-draft-state/v1"
MARKOV_DRAFT_METRICS_SCHEMA = "immer.qwen3.8-markov-draft-metrics/v37"
MARKOV_RICCI_WORKING_SET_POLICY = "o1-priority+ricci-age-whole-answer/v1"
_STATE_PREFIX = b"IMMD\x0d"
_V12_STATE_PREFIX = b"IMMD\x0c"
_V11_STATE_PREFIX = b"IMMD\x0b"
_V10_STATE_PREFIX = b"IMMD\x0a"
_V9_STATE_PREFIX = b"IMMD\x09"
_V8_STATE_PREFIX = b"IMMD\x08"
_V7_STATE_PREFIX = b"IMMD\x07"
_V6_STATE_PREFIX = b"IMMD\x06"
_V5_STATE_PREFIX = b"IMMD\x05"
_V4_STATE_PREFIX = b"IMMD\x04"
_V3_STATE_PREFIX = b"IMMD\x03"
_V2_STATE_PREFIX = b"IMMD\x02"
_LEGACY_STATE_PREFIX = b"IMMD\x01"
_STATE_PREFIXES = (
    _STATE_PREFIX,
    _V12_STATE_PREFIX,
    _V11_STATE_PREFIX,
    _V10_STATE_PREFIX,
    _V9_STATE_PREFIX,
    _V8_STATE_PREFIX,
    _V7_STATE_PREFIX,
    _V6_STATE_PREFIX,
    _V5_STATE_PREFIX,
    _V4_STATE_PREFIX,
    _V3_STATE_PREFIX,
    _V2_STATE_PREFIX,
    _LEGACY_STATE_PREFIX,
)
_MAX_STATE_BYTES = 16 * 1024 * 1024
_UNKNOWN_TOKEN = "<unknown>"
_EPISODE_TOKEN = "<episode>"
_HEX = frozenset("0123456789abcdef")
_MAX_IMPORTED_EPISODE_DIGESTS = 65_536
_MAX_PROPOSAL_POSITIONS = 16
_PLANNER_NAMES = ("beam", "council", "phrase", "markov", "mtp")
_INTERNAL_PLANNER_COUNT = 3
_V12_PLANNER_COUNT = 3
_PLANNING_DIAGNOSTIC_FIELDS = (
    "_predictions",
    "_council_predictions",
    "_last_raw_confidence",
    "_last_empirical_evidence",
    "_last_confidence",
    "_last_disagreement",
    "_last_position",
    "_last_position_maturity",
    "_last_dialect_skill_maturity",
    "_last_position_weights",
    "_position_specialist_predictions",
    "_dialect_specialist_predictions",
    "_max_position_maturity",
    "_max_dialect_skill_maturity",
    "_lookahead_calls",
    "_lookahead_candidates",
    "_lookahead_token_changes",
    "_last_lookahead_gain",
    "_max_lookahead_gain",
    "_last_plan_trace",
    "_last_beam_prefix_posteriors",
    "_last_beam_path_count",
)


class MarkovDraftError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _RecursiveMarkovTrace:
    token_ids: tuple[int, ...]
    expert_predictions: tuple[tuple[int, ...], ...]
    plan_trace: tuple[tuple[int, int, float], ...]
    next_position: int
    gate_pending: bool


@dataclass(frozen=True, slots=True)
class _PlannerTournamentTrace:
    candidates: tuple[tuple[int, ...], ...]
    alive: tuple[bool, ...]
    next_position: int


@dataclass(frozen=True, slots=True)
class _ProviderPolicySnapshot:
    observations: tuple[tuple[int, ...], ...] | None
    hits: tuple[tuple[int, ...], ...] | None
    feedback: tuple[tuple[int, int, bool], ...]


@dataclass(frozen=True, slots=True)
class MarkovExpertSpec:
    name: str
    max_order: int
    window: int
    local_only: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Markov expert name must not be empty")
        if (
            isinstance(self.max_order, bool)
            or not isinstance(self.max_order, int)
            or self.max_order < 0
        ):
            raise ValueError("Markov expert order must be non-negative")
        if (
            isinstance(self.window, bool)
            or not isinstance(self.window, int)
            or self.window < max(8, self.max_order + 1)
        ):
            raise ValueError("Markov expert window is too small")
        if not isinstance(self.local_only, bool):
            raise TypeError("local_only must be boolean")


def _expert_specs(
    max_order: int, max_history_tokens: int
) -> tuple[MarkovExpertSpec, ...]:
    global_window = min(4096, max_history_tokens)
    rows = (
        ("local-o0-w128", 0, min(128, max_history_tokens), True),
        ("global-o0-w4096", 0, global_window, False),
        ("local-o1-w128", min(1, max_order), min(128, max_history_tokens), True),
        ("global-o1-w4096", min(1, max_order), global_window, False),
        (
            "recent-o2-w256",
            min(2, max_order),
            min(256, max_history_tokens),
            False,
        ),
        (
            "medium-o4-w1024",
            min(4, max_order),
            min(1024, max_history_tokens),
            False,
        ),
        ("deep-o8-w4096", min(8, max_order), global_window, False),
        (f"max-o{max_order}-w4096", max_order, global_window, False),
    )
    return tuple(
        MarkovExpertSpec(name, order, max(8, window), local)
        for name, order, window, local in rows
    )


class _TransitionFingerprint:
    """Sparse interpolated variable-order counts over token-ID symbols."""

    def __init__(
        self,
        *,
        max_order: int,
        alpha: float,
        backoff_strength: float,
        min_count: int,
        vocabulary: tuple[str, ...],
        counts: Mapping[tuple[str, ...], Counter[str]],
    ) -> None:
        self.max_order = max_order
        self.alpha = alpha
        self.backoff_strength = backoff_strength
        self.min_count = min_count
        self.vocabulary = tuple(sorted(set(vocabulary) | {_UNKNOWN_TOKEN}))
        self.counts = dict(counts)

    @classmethod
    def fit(
        cls,
        sequence: Sequence[str],
        *,
        max_order: int,
        alpha: float,
        backoff_strength: float,
        min_count: int,
    ) -> "_TransitionFingerprint":
        tokens = tuple(sequence)
        if not tokens:
            raise ValueError("cannot fit an empty Markov token sequence")
        counts: dict[tuple[str, ...], Counter[str]] = {}
        for index, target in enumerate(tokens):
            for order in range(min(max_order, index) + 1):
                context = tokens[index - order : index] if order else ()
                counts.setdefault(context, Counter())[target] += 1
        return cls(
            max_order=max_order,
            alpha=alpha,
            backoff_strength=backoff_strength,
            min_count=min_count,
            vocabulary=tokens,
            counts=counts,
        )

    def distribution(self, context: Sequence[str]) -> dict[str, float]:
        root = self.counts[()]
        denominator = sum(root.values()) + self.alpha * len(self.vocabulary)
        probabilities = {
            token: (root.get(token, 0) + self.alpha) / denominator
            for token in self.vocabulary
        }
        known = tuple(
            token if token in self.vocabulary else _UNKNOWN_TOKEN for token in context
        )
        for order in range(1, min(self.max_order, len(known)) + 1):
            counter = self.counts.get(known[-order:])
            if not counter:
                continue
            total = sum(counter.values())
            if total < self.min_count:
                continue
            weight = total / (total + self.backoff_strength)
            for token in self.vocabulary:
                empirical = counter.get(token, 0) / total
                probabilities[token] = (1.0 - weight) * probabilities[
                    token
                ] + weight * empirical
        return probabilities

    def token_evidence(
        self,
        context: Sequence[str],
        token: str,
    ) -> tuple[float, int, int, int]:
        known = tuple(
            value if value in self.vocabulary else _UNKNOWN_TOKEN for value in context
        )
        candidates = []
        for order in range(min(self.max_order, len(known)) + 1):
            key = () if order == 0 else known[-order:]
            counter = self.counts.get(key)
            if not counter:
                continue
            support = int(counter.get(token, 0))
            total = sum(counter.values())
            if support <= 0 or total < self.min_count:
                continue
            probability = support / total
            support_strength = 1.0 - math.exp(-support / 4.0)
            order_strength = 0.5 + 0.5 * order / max(1, self.max_order)
            candidates.append(
                (
                    probability * support_strength * order_strength,
                    support,
                    total,
                    order,
                )
            )
        return max(candidates, default=(0.0, 0, 0, 0))

    def contextual_confidence(
        self,
        context: Sequence[str],
        token: str,
        *,
        min_order: int,
        min_support: int,
        support_scale: float,
    ) -> tuple[float, int, int, int]:
        known = tuple(
            value if value in self.vocabulary else _UNKNOWN_TOKEN for value in context
        )
        for order in range(min(self.max_order, len(known)), min_order - 1, -1):
            counter = self.counts.get(known[-order:])
            if not counter:
                continue
            support = int(counter.get(token, 0))
            total = sum(counter.values())
            if support < min_support or total <= 0:
                continue
            probability = support / total
            support_strength = 1.0 - math.exp(-support / support_scale)
            order_strength = 0.75 + 0.25 * order / max(1, self.max_order)
            return (
                max(
                    0.0,
                    min(0.999, probability * support_strength * order_strength),
                ),
                support,
                total,
                order,
            )
        return 0.0, 0, 0, 0

    def contextual_options(
        self,
        context: Sequence[str],
        *,
        min_order: int,
        min_support: int,
        support_scale: float,
        limit: int,
    ) -> tuple[tuple[str, float, int, int, int], ...]:
        known = tuple(
            token if token in self.vocabulary else _UNKNOWN_TOKEN for token in context
        )
        for order in range(min(self.max_order, len(known)), min_order - 1, -1):
            counter = self.counts.get(known[-order:])
            if not counter:
                continue
            total = sum(counter.values())
            rows = []
            for token, support in counter.items():
                if support < min_support:
                    continue
                probability = support / total
                support_strength = 1.0 - math.exp(-support / support_scale)
                order_strength = 0.75 + 0.25 * order / max(1, self.max_order)
                confidence = max(
                    0.0,
                    min(0.999, probability * support_strength * order_strength),
                )
                if confidence > 0.0:
                    rows.append((token, confidence, support, total, order))
            if rows:
                return tuple(
                    sorted(
                        rows,
                        key=lambda row: (-row[1], -row[2], row[0]),
                    )[:limit]
                )
        return ()


@dataclass(frozen=True, slots=True)
class MarkovLanguageTokenEvidence:
    token_id: int
    atlas_score: float
    online_score: float
    score: float
    atlas_support: int
    online_support: int

    @property
    def support(self) -> int:
        return self.atlas_support + self.online_support

    def __post_init__(self) -> None:
        if (
            isinstance(self.token_id, bool)
            or not isinstance(self.token_id, int)
            or self.token_id < 0
            or any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in (self.atlas_score, self.online_score, self.score)
            )
            or isinstance(self.atlas_support, bool)
            or not isinstance(self.atlas_support, int)
            or self.atlas_support < 0
            or isinstance(self.online_support, bool)
            or not isinstance(self.online_support, int)
            or self.online_support < 0
            or self.score + 1e-15 < max(self.atlas_score, self.online_score)
        ):
            raise ValueError("Markov language evidence is invalid")


@dataclass(frozen=True, slots=True)
class _BeamStep:
    token: int
    greedy: int
    gain: float
    confidence: float
    raw_confidence: float
    empirical_evidence: float
    disagreement: float
    weights: tuple[float, ...]
    position_maturity: float
    dialect_maturity: float
    evaluated_candidates: int


@dataclass(frozen=True, slots=True)
class _BeamPath:
    score: float
    tokens: tuple[int, ...]
    steps: tuple[_BeamStep, ...]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _planner_matrices(
    observations: Sequence[Sequence[int]],
    hits: Sequence[Sequence[int]],
    *,
    label: str,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    observed_rows = tuple(tuple(row) for row in observations)
    hit_rows = tuple(tuple(row) for row in hits)
    if not observed_rows and not hit_rows:
        zeros = tuple((0,) * _MAX_PROPOSAL_POSITIONS for _ in _PLANNER_NAMES)
        return zeros, zeros
    if (
        len(observed_rows) != len(_PLANNER_NAMES)
        or len(hit_rows) != len(_PLANNER_NAMES)
        or any(
            len(observed) != _MAX_PROPOSAL_POSITIONS
            or len(hit) != _MAX_PROPOSAL_POSITIONS
            for observed, hit in zip(observed_rows, hit_rows, strict=True)
        )
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for row in (*observed_rows, *hit_rows)
            for value in row
        )
        or any(
            hit > observed
            for observed_row, hit_row in zip(
                observed_rows,
                hit_rows,
                strict=True,
            )
            for observed, hit in zip(observed_row, hit_row, strict=True)
        )
    ):
        raise ValueError(f"{label} planner state is invalid")
    return observed_rows, hit_rows


def _read_state_bytes(path: Path) -> bytes:
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
            or not 0 < before.st_size <= _MAX_STATE_BYTES
        ):
            raise MarkovDraftError("Markov draft state size is invalid")
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise MarkovDraftError("Markov draft state returned a short read")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        linked = path.lstat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or (after.st_dev, after.st_ino) != (linked.st_dev, linked.st_ino):
            raise MarkovDraftError("Markov draft state changed while read")
        return b"".join(chunks)
    except MarkovDraftError:
        raise
    except OSError as exc:
        raise MarkovDraftError("cannot read Markov draft state") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


@dataclass(frozen=True, slots=True)
class MarkovDialectState:
    dialect_id: str
    signature: tuple[int, ...]
    visits: int
    last_seen: int
    rapidities: tuple[float, ...]
    observations: tuple[int, ...]
    hits: tuple[int, ...]
    horizon_observations: tuple[tuple[int, ...], ...] = ()
    horizon_hits: tuple[tuple[int, ...], ...] = ()
    plan_observations: tuple[int, ...] = ()
    plan_hits: tuple[int, ...] = ()
    planner_observations: tuple[tuple[int, ...], ...] = ()
    planner_hits: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.dialect_id, str)
            or len(self.dialect_id) != 64
            or set(self.dialect_id) - _HEX
        ):
            raise ValueError("dialect_id must be a SHA-256 digest")
        signature = tuple(self.signature)
        if (
            not 1 <= len(signature) <= 32
            or signature != tuple(sorted(set(signature)))
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < 1 << 64
                for value in signature
            )
        ):
            raise ValueError("dialect signature is invalid")
        if (
            isinstance(self.visits, bool)
            or not isinstance(self.visits, int)
            or self.visits <= 0
            or isinstance(self.last_seen, bool)
            or not isinstance(self.last_seen, int)
            or self.last_seen < 0
        ):
            raise ValueError("dialect visit state is invalid")
        rapidities = tuple(float(value) for value in self.rapidities)
        observations = tuple(self.observations)
        hits = tuple(self.hits)
        if (
            not rapidities
            or len({len(rapidities), len(observations), len(hits)}) != 1
            or any(not math.isfinite(value) for value in rapidities)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (*observations, *hits)
            )
            or any(hit > seen for hit, seen in zip(hits, observations, strict=True))
        ):
            raise ValueError("dialect expert state is invalid")
        horizon_observations = tuple(
            tuple(row) for row in self.horizon_observations
        )
        horizon_hits = tuple(tuple(row) for row in self.horizon_hits)
        if bool(horizon_observations) != bool(horizon_hits) or (
            horizon_observations
            and (
                len(horizon_observations) != _MAX_PROPOSAL_POSITIONS
                or len(horizon_hits) != _MAX_PROPOSAL_POSITIONS
                or any(
                    len(observed) != len(rapidities) or len(hit) != len(rapidities)
                    for observed, hit in zip(
                        horizon_observations,
                        horizon_hits,
                        strict=True,
                    )
                )
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for row in (*horizon_observations, *horizon_hits)
                    for value in row
                )
                or any(
                    hit > observed
                    for observed_row, hit_row in zip(
                        horizon_observations,
                        horizon_hits,
                        strict=True,
                    )
                    for observed, hit in zip(
                        observed_row,
                        hit_row,
                        strict=True,
                    )
                )
            )
        ):
            raise ValueError("dialect horizon expert state is invalid")
        plan_observations = tuple(self.plan_observations)
        plan_hits = tuple(self.plan_hits)
        if bool(plan_observations) != bool(plan_hits) or (
            plan_observations
            and (
                len(plan_observations) != _MAX_PROPOSAL_POSITIONS
                or len(plan_hits) != _MAX_PROPOSAL_POSITIONS
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in (*plan_observations, *plan_hits)
                )
                or any(
                    hit > observed
                    for observed, hit in zip(
                        plan_observations,
                        plan_hits,
                        strict=True,
                    )
                )
            )
        ):
            raise ValueError("dialect horizon plan state is invalid")
        planner_observations, planner_hits = _planner_matrices(
            self.planner_observations,
            self.planner_hits,
            label="dialect",
        )
        object.__setattr__(self, "signature", signature)
        object.__setattr__(self, "rapidities", rapidities)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "hits", hits)
        object.__setattr__(self, "horizon_observations", horizon_observations)
        object.__setattr__(self, "horizon_hits", horizon_hits)
        object.__setattr__(self, "plan_observations", plan_observations)
        object.__setattr__(self, "plan_hits", plan_hits)
        object.__setattr__(self, "planner_observations", planner_observations)
        object.__setattr__(self, "planner_hits", planner_hits)

    def to_record(self) -> dict[str, object]:
        plan_observations = (
            self.plan_observations
            if self.plan_observations
            else (0,) * _MAX_PROPOSAL_POSITIONS
        )
        plan_hits = (
            self.plan_hits
            if self.plan_hits
            else (0,) * _MAX_PROPOSAL_POSITIONS
        )
        planner_zeros = tuple(
            (0,) * _MAX_PROPOSAL_POSITIONS for _ in _PLANNER_NAMES
        )
        planner_observations = self.planner_observations or planner_zeros
        planner_hits = self.planner_hits or planner_zeros
        return {
            "dialect_id": self.dialect_id,
            "hits": list(self.hits),
            "horizon_hits": [list(row) for row in self.horizon_hits],
            "horizon_observations": [
                list(row) for row in self.horizon_observations
            ],
            "plan_hits": list(plan_hits),
            "plan_observations": list(plan_observations),
            "planner_hits": [list(row) for row in planner_hits],
            "planner_observations": [
                list(row) for row in planner_observations
            ],
            "last_seen": self.last_seen,
            "observations": list(self.observations),
            "rapidities": [value.hex() for value in self.rapidities],
            "signature": [f"{value:016x}" for value in self.signature],
            "visits": self.visits,
        }

    @classmethod
    def from_record(
        cls,
        value: object,
        *,
        legacy: bool = False,
        plan_memory: bool = True,
        planner_memory: bool = True,
        legacy_planner_count: int | None = None,
    ) -> "MarkovDialectState":
        expected = {
            "dialect_id",
            "hits",
            "last_seen",
            "observations",
            "rapidities",
            "signature",
            "visits",
        }
        if not legacy:
            expected |= {"horizon_hits", "horizon_observations"}
        if plan_memory:
            expected |= {"plan_hits", "plan_observations"}
        if planner_memory:
            expected |= {"planner_hits", "planner_observations"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("dialect record is invalid")
        if plan_memory and (
            not value.get("plan_observations") or not value.get("plan_hits")
        ):
            raise ValueError("dialect plan memory is missing")
        if planner_memory and (
            not value.get("planner_observations")
            or not value.get("planner_hits")
        ):
            raise ValueError("dialect planner memory is missing")
        try:
            planner_observations = tuple(
                tuple(row) for row in value.get("planner_observations", ())
            )
            planner_hits = tuple(
                tuple(row) for row in value.get("planner_hits", ())
            )
            if legacy_planner_count is not None:
                if (
                    len(planner_observations) != legacy_planner_count
                    or len(planner_hits) != legacy_planner_count
                ):
                    raise ValueError("legacy dialect planner width changed")
                missing = len(_PLANNER_NAMES) - legacy_planner_count
                zeros = (0,) * _MAX_PROPOSAL_POSITIONS
                planner_observations = planner_observations + (zeros,) * missing
                planner_hits = planner_hits + (zeros,) * missing
            return cls(
                dialect_id=value["dialect_id"],
                signature=tuple(int(item, 16) for item in value["signature"]),
                visits=value["visits"],
                last_seen=value["last_seen"],
                rapidities=tuple(float.fromhex(item) for item in value["rapidities"]),
                observations=tuple(value["observations"]),
                hits=tuple(value["hits"]),
                horizon_observations=tuple(
                    tuple(row) for row in value.get("horizon_observations", ())
                ),
                horizon_hits=tuple(
                    tuple(row) for row in value.get("horizon_hits", ())
                ),
                plan_observations=tuple(value.get("plan_observations", ())),
                plan_hits=tuple(value.get("plan_hits", ())),
                planner_observations=planner_observations,
                planner_hits=planner_hits,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("dialect record values are invalid") from exc


@dataclass(frozen=True, slots=True)
class MarkovPhraseOption:
    token_ids: tuple[int, ...]
    source: str
    context_order: int
    support: int
    total: int
    kind: str = "literal"
    confidence_override: float | None = None
    token_confidences: tuple[float, ...] = ()
    token_disagreements: tuple[float, ...] = ()
    cell_sha256: str | None = None
    crystal_layer: int | None = None
    crystal_transaction_sha256: str | None = None

    @property
    def confidence(self) -> float:
        return (
            self.support / self.total
            if self.confidence_override is None
            else self.confidence_override
        )

    def __post_init__(self) -> None:
        if self.source not in {
            "atlas",
            "crystal",
            "dialect",
            "global",
            "request",
        }:
            raise ValueError("phrase option source is invalid")
        if self.kind not in {
            "atlas",
            "binding",
            "composition",
            "crystal",
            "literal",
            "periodic",
        }:
            raise ValueError("phrase option kind is invalid")
        if (
            not 1 <= len(self.token_ids) <= 15
            or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in self.token_ids
            )
            or self.context_order < 1
            or self.support < 1
            or self.total < self.support
        ):
            raise ValueError("phrase option evidence is invalid")
        if self.kind == "crystal":
            confidence = self.confidence_override
            if (
                self.source != "crystal"
                or isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(float(confidence))
                or not 0.0 <= float(confidence) <= 1.0
                or len(self.token_confidences) != len(self.token_ids)
                or len(self.token_disagreements) != len(self.token_ids)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or not 0.0 <= float(value) <= 1.0
                    for value in self.token_confidences
                )
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                    for value in self.token_disagreements
                )
                or not isinstance(self.cell_sha256, str)
                or len(self.cell_sha256) != 64
                or bool(set(self.cell_sha256) - _HEX)
                or (self.crystal_layer is None)
                != (self.crystal_transaction_sha256 is None)
                or (
                    self.crystal_layer is not None
                    and (
                        isinstance(self.crystal_layer, bool)
                        or not isinstance(self.crystal_layer, int)
                        or self.crystal_layer < 0
                        or not isinstance(self.crystal_transaction_sha256, str)
                        or len(self.crystal_transaction_sha256) != 64
                        or bool(set(self.crystal_transaction_sha256) - _HEX)
                    )
                )
            ):
                raise ValueError("crystal phrase evidence is invalid")
        elif (
            self.confidence_override is not None
            or self.token_confidences
            or self.token_disagreements
            or self.cell_sha256 is not None
            or self.crystal_layer is not None
            or self.crystal_transaction_sha256 is not None
        ):
            raise ValueError("non-crystal phrase carries crystal evidence")


@dataclass(frozen=True, slots=True)
class MarkovDraftState:
    vocab_size: int
    max_history_tokens: int
    token_ids: tuple[int, ...]
    updates: int = 0
    expert_names: tuple[str, ...] = ()
    expert_log_weights: tuple[float, ...] = ()
    expert_observations: tuple[int, ...] = ()
    expert_hits: tuple[int, ...] = ()
    horizon_expert_observations: tuple[tuple[int, ...], ...] = ()
    horizon_expert_hits: tuple[tuple[int, ...], ...] = ()
    horizon_plan_observations: tuple[int, ...] = ()
    horizon_plan_hits: tuple[int, ...] = ()
    planner_observations: tuple[tuple[int, ...], ...] = ()
    planner_hits: tuple[tuple[int, ...], ...] = ()
    lookahead_observations: tuple[int, ...] = ()
    lookahead_hits: tuple[int, ...] = ()
    lookahead_greedy_hits: tuple[int, ...] = ()
    leader_changes: int = 0
    episode_lengths: tuple[int, ...] = ()
    feedback_count: int = 0
    surprise_mean: float = 0.0
    surprise_deviation: float = 1.0
    surprise_cusum: float = 0.0
    regime_generation: int = 0
    clock: int = 0
    dialects: tuple[MarkovDialectState, ...] = ()
    episode_dialects: tuple[str | None, ...] = ()
    episode_prompt_lengths: tuple[int | None, ...] = ()
    imported_episode_sha256s: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.vocab_size, bool)
            or not isinstance(self.vocab_size, int)
            or self.vocab_size <= 1
        ):
            raise ValueError("vocab_size must be greater than one")
        if (
            isinstance(self.max_history_tokens, bool)
            or not isinstance(self.max_history_tokens, int)
            or self.max_history_tokens < 8
        ):
            raise ValueError("max_history_tokens must be at least eight")
        if len(self.token_ids) > self.max_history_tokens or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or not 0 <= token < self.vocab_size
            for token in self.token_ids
        ):
            raise ValueError("Markov draft token history is invalid")
        episode_lengths = tuple(self.episode_lengths)
        if self.token_ids and not episode_lengths:
            episode_lengths = (len(self.token_ids),)
        if any(
            isinstance(length, bool) or not isinstance(length, int) or length <= 0
            for length in episode_lengths
        ) or sum(episode_lengths) != len(self.token_ids):
            raise ValueError("Markov episode boundaries are invalid")
        episode_dialects = tuple(self.episode_dialects)
        if episode_lengths and not episode_dialects:
            episode_dialects = (None,) * len(episode_lengths)
        if len(episode_dialects) != len(episode_lengths) or any(
            value is not None
            and (not isinstance(value, str) or len(value) != 64 or set(value) - _HEX)
            for value in episode_dialects
        ):
            raise ValueError("Markov episode dialect bindings are invalid")
        episode_prompt_lengths = tuple(self.episode_prompt_lengths)
        if episode_lengths and not episode_prompt_lengths:
            episode_prompt_lengths = (None,) * len(episode_lengths)
        if len(episode_prompt_lengths) != len(episode_lengths) or any(
            value is not None
            and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value < episode_length
            )
            for value, episode_length in zip(
                episode_prompt_lengths,
                episode_lengths,
                strict=True,
            )
        ):
            raise ValueError("Markov episode prompt boundaries are invalid")
        if (
            isinstance(self.updates, bool)
            or not isinstance(self.updates, int)
            or self.updates < 0
        ):
            raise ValueError("updates must be a non-negative integer")
        names = tuple(self.expert_names)
        log_weights = tuple(float(value) for value in self.expert_log_weights)
        observations = tuple(self.expert_observations)
        hits = tuple(self.expert_hits)
        lengths = {len(names), len(log_weights), len(observations), len(hits)}
        if lengths not in ({0}, {len(names)}) or (
            names
            and (
                len(set(names)) != len(names)
                or any(not isinstance(name, str) or not name for name in names)
                or any(not math.isfinite(value) for value in log_weights)
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in (*observations, *hits)
                )
                or any(hit > seen for hit, seen in zip(hits, observations, strict=True))
            )
        ):
            raise ValueError("Markov expert state is invalid")
        horizon_observations = tuple(
            tuple(row) for row in self.horizon_expert_observations
        )
        horizon_hits = tuple(tuple(row) for row in self.horizon_expert_hits)
        if bool(horizon_observations) != bool(horizon_hits) or (
            horizon_observations
            and (
                len(horizon_observations) != _MAX_PROPOSAL_POSITIONS
                or len(horizon_hits) != _MAX_PROPOSAL_POSITIONS
                or any(
                    len(observed) != len(names) or len(hit) != len(names)
                    for observed, hit in zip(
                        horizon_observations,
                        horizon_hits,
                        strict=True,
                    )
                )
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for row in (*horizon_observations, *horizon_hits)
                    for value in row
                )
                or any(
                    hit > observed
                    for observed_row, hit_row in zip(
                        horizon_observations,
                        horizon_hits,
                        strict=True,
                    )
                    for observed, hit in zip(
                        observed_row,
                        hit_row,
                        strict=True,
                    )
                )
            )
        ):
            raise ValueError("Markov horizon expert state is invalid")
        plan_observations = tuple(self.horizon_plan_observations)
        plan_hits = tuple(self.horizon_plan_hits)
        if bool(plan_observations) != bool(plan_hits) or (
            plan_observations
            and (
                len(plan_observations) != _MAX_PROPOSAL_POSITIONS
                or len(plan_hits) != _MAX_PROPOSAL_POSITIONS
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in (*plan_observations, *plan_hits)
                )
                or any(
                    hit > observed
                    for observed, hit in zip(
                        plan_observations,
                        plan_hits,
                        strict=True,
                    )
                )
            )
        ):
            raise ValueError("Markov horizon plan state is invalid")
        planner_observations, planner_hits = _planner_matrices(
            self.planner_observations,
            self.planner_hits,
            label="Markov",
        )
        lookahead_observations = tuple(self.lookahead_observations)
        lookahead_hits = tuple(self.lookahead_hits)
        lookahead_greedy_hits = tuple(self.lookahead_greedy_hits)
        if len(
            {
                bool(lookahead_observations),
                bool(lookahead_hits),
                bool(lookahead_greedy_hits),
            }
        ) != 1 or (
            lookahead_observations
            and (
                len(lookahead_observations) != _MAX_PROPOSAL_POSITIONS
                or len(lookahead_hits) != _MAX_PROPOSAL_POSITIONS
                or len(lookahead_greedy_hits) != _MAX_PROPOSAL_POSITIONS
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in (
                        *lookahead_observations,
                        *lookahead_hits,
                        *lookahead_greedy_hits,
                    )
                )
                or any(
                    planned > observed or greedy > observed
                    for observed, planned, greedy in zip(
                        lookahead_observations,
                        lookahead_hits,
                        lookahead_greedy_hits,
                        strict=True,
                    )
                )
            )
        ):
            raise ValueError("Markov lookahead state is invalid")
        if (
            isinstance(self.leader_changes, bool)
            or not isinstance(self.leader_changes, int)
            or self.leader_changes < 0
        ):
            raise ValueError("leader_changes must be non-negative")
        if (
            isinstance(self.feedback_count, bool)
            or not isinstance(self.feedback_count, int)
            or self.feedback_count < 0
            or isinstance(self.regime_generation, bool)
            or not isinstance(self.regime_generation, int)
            or self.regime_generation < 0
        ):
            raise ValueError("Markov regime counters are invalid")
        for value, label in (
            (self.surprise_mean, "surprise_mean"),
            (self.surprise_deviation, "surprise_deviation"),
            (self.surprise_cusum, "surprise_cusum"),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{label} must be finite and non-negative")
        if (
            isinstance(self.clock, bool)
            or not isinstance(self.clock, int)
            or self.clock < 0
        ):
            raise ValueError("clock must be a non-negative integer")
        imported = tuple(self.imported_episode_sha256s)
        if (
            len(imported) > _MAX_IMPORTED_EPISODE_DIGESTS
            or imported != tuple(sorted(set(imported)))
            or any(
                not isinstance(value, str) or len(value) != 64 or set(value) - _HEX
                for value in imported
            )
        ):
            raise ValueError("imported episode digest inventory is invalid")
        dialects = tuple(self.dialects)
        if (
            len(dialects) > 64
            or any(not isinstance(row, MarkovDialectState) for row in dialects)
            or tuple(sorted(dialects, key=lambda row: row.dialect_id)) != dialects
            or len({row.dialect_id for row in dialects}) != len(dialects)
            or any(
                len(row.rapidities) != len(names) or row.last_seen > self.clock
                for row in dialects
            )
        ):
            raise ValueError("Markov dialect inventory is invalid")
        object.__setattr__(self, "expert_names", names)
        object.__setattr__(self, "expert_log_weights", log_weights)
        object.__setattr__(self, "expert_observations", observations)
        object.__setattr__(self, "expert_hits", hits)
        object.__setattr__(
            self,
            "horizon_expert_observations",
            horizon_observations,
        )
        object.__setattr__(self, "horizon_expert_hits", horizon_hits)
        object.__setattr__(self, "horizon_plan_observations", plan_observations)
        object.__setattr__(self, "horizon_plan_hits", plan_hits)
        object.__setattr__(self, "planner_observations", planner_observations)
        object.__setattr__(self, "planner_hits", planner_hits)
        object.__setattr__(self, "lookahead_observations", lookahead_observations)
        object.__setattr__(self, "lookahead_hits", lookahead_hits)
        object.__setattr__(self, "lookahead_greedy_hits", lookahead_greedy_hits)
        object.__setattr__(self, "episode_lengths", episode_lengths)
        object.__setattr__(self, "dialects", dialects)
        object.__setattr__(self, "episode_dialects", episode_dialects)
        object.__setattr__(
            self,
            "episode_prompt_lengths",
            episode_prompt_lengths,
        )
        object.__setattr__(self, "imported_episode_sha256s", imported)

    def to_bytes(self) -> bytes:
        planner_zeros = tuple(
            (0,) * _MAX_PROPOSAL_POSITIONS for _ in _PLANNER_NAMES
        )
        planner_observations = self.planner_observations or planner_zeros
        planner_hits = self.planner_hits or planner_zeros
        raw = _canonical(
            {
                "max_history_tokens": self.max_history_tokens,
                "expert_hits": list(self.expert_hits),
                "expert_log_weights": [
                    value.hex() for value in self.expert_log_weights
                ],
                "expert_names": list(self.expert_names),
                "expert_observations": list(self.expert_observations),
                "horizon_expert_hits": [
                    list(row) for row in self.horizon_expert_hits
                ],
                "horizon_expert_observations": [
                    list(row) for row in self.horizon_expert_observations
                ],
                "horizon_plan_hits": list(self.horizon_plan_hits),
                "horizon_plan_observations": list(
                    self.horizon_plan_observations
                ),
                "planner_hits": [list(row) for row in planner_hits],
                "planner_observations": [
                    list(row) for row in planner_observations
                ],
                "lookahead_greedy_hits": list(self.lookahead_greedy_hits),
                "lookahead_hits": list(self.lookahead_hits),
                "lookahead_observations": list(self.lookahead_observations),
                "leader_changes": self.leader_changes,
                "episode_lengths": list(self.episode_lengths),
                "episode_dialects": list(self.episode_dialects),
                "episode_prompt_lengths": list(self.episode_prompt_lengths),
                "imported_episode_sha256s": list(self.imported_episode_sha256s),
                "clock": self.clock,
                "dialects": [row.to_record() for row in self.dialects],
                "feedback_count": self.feedback_count,
                "regime_generation": self.regime_generation,
                "schema": MARKOV_DRAFT_STATE_SCHEMA,
                "surprise_cusum": self.surprise_cusum.hex(),
                "surprise_deviation": self.surprise_deviation.hex(),
                "surprise_mean": self.surprise_mean.hex(),
                "token_ids": list(self.token_ids),
                "updates": self.updates,
                "vocab_size": self.vocab_size,
            }
        )
        encoded = _STATE_PREFIX + zlib.compress(raw, level=9)
        if len(encoded) > _MAX_STATE_BYTES:
            raise MarkovDraftError("Markov draft state exceeds its byte bound")
        return encoded

    @classmethod
    def from_bytes(cls, data: bytes) -> "MarkovDraftState":
        if (
            not isinstance(data, bytes)
            or len(data) > _MAX_STATE_BYTES
            or not data.startswith(_STATE_PREFIXES)
        ):
            raise MarkovDraftError("Markov draft state envelope is invalid")
        try:
            decoder = zlib.decompressobj()
            prefix = next(
                candidate
                for candidate in _STATE_PREFIXES
                if data.startswith(candidate)
            )
            raw = decoder.decompress(data[len(prefix) :], _MAX_STATE_BYTES + 1)
            if (
                len(raw) > _MAX_STATE_BYTES
                or decoder.unconsumed_tail
                or not decoder.eof
            ):
                raise zlib.error("expanded Markov state exceeds its bound")
            value = json.loads(raw)
        except (zlib.error, UnicodeError, json.JSONDecodeError) as exc:
            raise MarkovDraftError("Markov draft state is corrupt") from exc
        legacy = (
            isinstance(value, dict)
            and value.get("schema") == LEGACY_MARKOV_DRAFT_STATE_SCHEMA
        )
        v2 = (
            isinstance(value, dict)
            and value.get("schema") == V2_MARKOV_DRAFT_STATE_SCHEMA
        )
        v3 = (
            isinstance(value, dict)
            and value.get("schema") == V3_MARKOV_DRAFT_STATE_SCHEMA
        )
        v4 = (
            isinstance(value, dict)
            and value.get("schema") == V4_MARKOV_DRAFT_STATE_SCHEMA
        )
        v5 = (
            isinstance(value, dict)
            and value.get("schema") == V5_MARKOV_DRAFT_STATE_SCHEMA
        )
        v6 = (
            isinstance(value, dict)
            and value.get("schema") == V6_MARKOV_DRAFT_STATE_SCHEMA
        )
        v7 = (
            isinstance(value, dict)
            and value.get("schema") == V7_MARKOV_DRAFT_STATE_SCHEMA
        )
        v8 = (
            isinstance(value, dict)
            and value.get("schema") == V8_MARKOV_DRAFT_STATE_SCHEMA
        )
        v9 = (
            isinstance(value, dict)
            and value.get("schema") == V9_MARKOV_DRAFT_STATE_SCHEMA
        )
        schema = value.get("schema") if isinstance(value, dict) else None
        v2_fields = {
            "expert_hits",
            "expert_log_weights",
            "expert_names",
            "expert_observations",
            "leader_changes",
            "episode_lengths",
            "feedback_count",
            "max_history_tokens",
            "regime_generation",
            "schema",
            "surprise_cusum",
            "surprise_deviation",
            "surprise_mean",
            "token_ids",
            "updates",
            "vocab_size",
        }
        v9_fields = {
            "expert_hits",
            "expert_log_weights",
            "expert_names",
            "expert_observations",
            "horizon_expert_hits",
            "horizon_expert_observations",
            "lookahead_greedy_hits",
            "lookahead_hits",
            "lookahead_observations",
            "leader_changes",
            "episode_lengths",
            "episode_dialects",
            "episode_prompt_lengths",
            "imported_episode_sha256s",
            "clock",
            "dialects",
            "feedback_count",
            "max_history_tokens",
            "regime_generation",
            "schema",
            "surprise_cusum",
            "surprise_deviation",
            "surprise_mean",
            "token_ids",
            "updates",
            "vocab_size",
        }
        v10_v11_fields = v9_fields | {
            "horizon_plan_hits",
            "horizon_plan_observations",
        }
        expected = (
            {
                "max_history_tokens",
                "schema",
                "token_ids",
                "updates",
                "vocab_size",
            }
            if legacy
            else v2_fields
            if v2
            else {
                "expert_hits",
                "expert_log_weights",
                "expert_names",
                "expert_observations",
                "leader_changes",
                "episode_lengths",
                "clock",
                "dialects",
                "feedback_count",
                "max_history_tokens",
                "regime_generation",
                "schema",
                "surprise_cusum",
                "surprise_deviation",
                "surprise_mean",
                "token_ids",
                "updates",
                "vocab_size",
            }
            if v3
            else {
                "expert_hits",
                "expert_log_weights",
                "expert_names",
                "expert_observations",
                "leader_changes",
                "episode_lengths",
                "episode_dialects",
                "clock",
                "dialects",
                "feedback_count",
                "max_history_tokens",
                "regime_generation",
                "schema",
                "surprise_cusum",
                "surprise_deviation",
                "surprise_mean",
                "token_ids",
                "updates",
                "vocab_size",
            }
            if v4
            else {
                "expert_hits",
                "expert_log_weights",
                "expert_names",
                "expert_observations",
                "leader_changes",
                "episode_lengths",
                "episode_dialects",
                "imported_episode_sha256s",
                "clock",
                "dialects",
                "feedback_count",
                "max_history_tokens",
                "regime_generation",
                "schema",
                "surprise_cusum",
                "surprise_deviation",
                "surprise_mean",
                "token_ids",
                "updates",
                "vocab_size",
            }
            if v5
            else {
                "expert_hits",
                "expert_log_weights",
                "expert_names",
                "expert_observations",
                "leader_changes",
                "episode_lengths",
                "episode_dialects",
                "episode_prompt_lengths",
                "imported_episode_sha256s",
                "clock",
                "dialects",
                "feedback_count",
                "max_history_tokens",
                "regime_generation",
                "schema",
                "surprise_cusum",
                "surprise_deviation",
                "surprise_mean",
                "token_ids",
                "updates",
                "vocab_size",
            }
            if v6
            else {
                "expert_hits",
                "expert_log_weights",
                "expert_names",
                "expert_observations",
                "horizon_expert_hits",
                "horizon_expert_observations",
                "leader_changes",
                "episode_lengths",
                "episode_dialects",
                "episode_prompt_lengths",
                "imported_episode_sha256s",
                "clock",
                "dialects",
                "feedback_count",
                "max_history_tokens",
                "regime_generation",
                "schema",
                "surprise_cusum",
                "surprise_deviation",
                "surprise_mean",
                "token_ids",
                "updates",
                "vocab_size",
            }
            if v7 or v8
            else v9_fields
            if v9
            else v10_v11_fields
            if schema
            in {V10_MARKOV_DRAFT_STATE_SCHEMA, V11_MARKOV_DRAFT_STATE_SCHEMA}
            else v10_v11_fields | {"planner_hits", "planner_observations"}
        )
        if (
            not isinstance(value, dict)
            or set(value) != expected
            or value.get("schema")
            not in {
                MARKOV_DRAFT_STATE_SCHEMA,
                V12_MARKOV_DRAFT_STATE_SCHEMA,
                V11_MARKOV_DRAFT_STATE_SCHEMA,
                V10_MARKOV_DRAFT_STATE_SCHEMA,
                V9_MARKOV_DRAFT_STATE_SCHEMA,
                V8_MARKOV_DRAFT_STATE_SCHEMA,
                V7_MARKOV_DRAFT_STATE_SCHEMA,
                V6_MARKOV_DRAFT_STATE_SCHEMA,
                V5_MARKOV_DRAFT_STATE_SCHEMA,
                V4_MARKOV_DRAFT_STATE_SCHEMA,
                V3_MARKOV_DRAFT_STATE_SCHEMA,
                V2_MARKOV_DRAFT_STATE_SCHEMA,
                LEGACY_MARKOV_DRAFT_STATE_SCHEMA,
            }
            or not isinstance(value.get("token_ids"), list)
            or _canonical(value) != raw
        ):
            raise MarkovDraftError("Markov draft state schema is invalid")
        planner_zeros = (0,) * _MAX_PROPOSAL_POSITIONS
        lookahead_observations = tuple(
            value.get("lookahead_observations", planner_zeros)
        )
        lookahead_hits = tuple(value.get("lookahead_hits", planner_zeros))
        lookahead_greedy_hits = tuple(
            value.get("lookahead_greedy_hits", planner_zeros)
        )
        horizon_plan_observations = tuple(
            value.get("horizon_plan_observations", planner_zeros)
        )
        horizon_plan_hits = tuple(
            value.get("horizon_plan_hits", planner_zeros)
        )
        planner_matrix_zeros = tuple(
            (0,) * _MAX_PROPOSAL_POSITIONS for _ in _PLANNER_NAMES
        )
        try:
            planner_observations = tuple(
                tuple(row)
                for row in value.get("planner_observations", planner_matrix_zeros)
            )
            planner_hits = tuple(
                tuple(row)
                for row in value.get("planner_hits", planner_matrix_zeros)
            )
        except TypeError as exc:
            raise MarkovDraftError("Markov draft state values are invalid") from exc
        if schema == V12_MARKOV_DRAFT_STATE_SCHEMA:
            if (
                len(planner_observations) != _V12_PLANNER_COUNT
                or len(planner_hits) != _V12_PLANNER_COUNT
            ):
                raise MarkovDraftError("Markov v12 planner width changed")
            missing = len(_PLANNER_NAMES) - _V12_PLANNER_COUNT
            zeros = (0,) * _MAX_PROPOSAL_POSITIONS
            planner_observations = planner_observations + (zeros,) * missing
            planner_hits = planner_hits + (zeros,) * missing
        if schema == MARKOV_DRAFT_STATE_SCHEMA and (
            not planner_observations or not planner_hits
        ):
            raise MarkovDraftError("Markov planner state is missing")
        try:
            return cls(
                vocab_size=value["vocab_size"],
                max_history_tokens=value["max_history_tokens"],
                token_ids=tuple(value["token_ids"]),
                updates=value["updates"],
                expert_names=tuple(value.get("expert_names", ())),
                expert_log_weights=tuple(
                    float.fromhex(item) for item in value.get("expert_log_weights", ())
                ),
                expert_observations=tuple(value.get("expert_observations", ())),
                expert_hits=tuple(value.get("expert_hits", ())),
                horizon_expert_observations=tuple(
                    tuple(row)
                    for row in value.get("horizon_expert_observations", ())
                ),
                horizon_expert_hits=tuple(
                    tuple(row) for row in value.get("horizon_expert_hits", ())
                ),
                horizon_plan_observations=horizon_plan_observations,
                horizon_plan_hits=horizon_plan_hits,
                planner_observations=planner_observations,
                planner_hits=planner_hits,
                lookahead_observations=lookahead_observations,
                lookahead_hits=lookahead_hits,
                lookahead_greedy_hits=lookahead_greedy_hits,
                leader_changes=value.get("leader_changes", 0),
                episode_lengths=tuple(value.get("episode_lengths", ())),
                episode_dialects=tuple(value.get("episode_dialects", ())),
                episode_prompt_lengths=tuple(value.get("episode_prompt_lengths", ())),
                imported_episode_sha256s=tuple(
                    value.get("imported_episode_sha256s", ())
                ),
                feedback_count=value.get("feedback_count", 0),
                surprise_mean=float.fromhex(value.get("surprise_mean", "0x0.0p+0")),
                surprise_deviation=float.fromhex(
                    value.get("surprise_deviation", "0x1.0p+0")
                ),
                surprise_cusum=float.fromhex(value.get("surprise_cusum", "0x0.0p+0")),
                regime_generation=value.get("regime_generation", 0),
                clock=value.get("clock", 0),
                dialects=tuple(
                    MarkovDialectState.from_record(
                        row,
                        legacy=value.get("schema")
                        not in {
                            MARKOV_DRAFT_STATE_SCHEMA,
                            V12_MARKOV_DRAFT_STATE_SCHEMA,
                            V11_MARKOV_DRAFT_STATE_SCHEMA,
                            V10_MARKOV_DRAFT_STATE_SCHEMA,
                            V9_MARKOV_DRAFT_STATE_SCHEMA,
                            V8_MARKOV_DRAFT_STATE_SCHEMA,
                        },
                        plan_memory=value.get("schema")
                        in {
                            MARKOV_DRAFT_STATE_SCHEMA,
                            V12_MARKOV_DRAFT_STATE_SCHEMA,
                            V11_MARKOV_DRAFT_STATE_SCHEMA,
                        },
                        planner_memory=value.get("schema")
                        in {
                            MARKOV_DRAFT_STATE_SCHEMA,
                            V12_MARKOV_DRAFT_STATE_SCHEMA,
                        },
                        legacy_planner_count=(
                            _V12_PLANNER_COUNT
                            if value.get("schema")
                            == V12_MARKOV_DRAFT_STATE_SCHEMA
                            else None
                        ),
                    )
                    for row in value.get("dialects", ())
                ),
            )
        except (TypeError, ValueError) as exc:
            raise MarkovDraftError("Markov draft state values are invalid") from exc


@dataclass(frozen=True, slots=True)
class MarkovDraftMetrics:
    schema: str
    draft_calls: int
    reconcile_calls: int
    predictions: int
    learned_tokens: int
    history_capacity_tokens: int
    episode_count: int
    imported_episode_count: int
    updates: int
    state_bytes: int
    council_predictions: int
    council_feedback: int
    external_reconcile_calls: int
    external_feedback_tokens: int
    teacher_forced_predictions: int
    teacher_forced_feedback_tokens: int
    teacher_forced_failures: int
    leader_changes: int
    expert_weights: tuple[tuple[str, float], ...]
    expert_accuracy: tuple[tuple[str, float], ...]
    effective_experts: float
    last_confidence: float
    last_raw_confidence: float
    last_empirical_evidence: float
    last_disagreement: float
    horizon_observations: tuple[int, ...]
    horizon_weighted_accuracy: tuple[float, ...]
    horizon_self_reliability: tuple[float, ...]
    horizon_plan_observations: tuple[int, ...]
    horizon_plan_hits: tuple[int, ...]
    last_position: int
    last_position_maturity: float
    last_dialect_skill_maturity: float
    last_position_weights: tuple[tuple[str, float], ...]
    position_specialist_predictions: int
    dialect_specialist_predictions: int
    max_position_maturity: float
    max_dialect_skill_maturity: float
    lookahead_calls: int
    lookahead_candidates: int
    lookahead_token_changes: int
    last_lookahead_gain: float
    max_lookahead_gain: float
    lookahead_outcomes: tuple[int, ...]
    lookahead_hits: tuple[int, ...]
    lookahead_greedy_hits: tuple[int, ...]
    request_weight_updates: int
    max_request_weight_shift: float
    request_position_updates: int
    max_request_position_maturity: float
    request_lookahead_updates: int
    request_regime_changes: int
    request_surprise_mean: float
    request_surprise_cusum: float
    beam_position_verified: tuple[int, ...]
    beam_position_hits: tuple[int, ...]
    planner_names: tuple[str, ...]
    planner_observations: tuple[tuple[int, ...], ...]
    planner_hits: tuple[tuple[int, ...], ...]
    planner_reliability: tuple[tuple[float, ...], ...]
    planner_tournament_calls: int
    planner_beam_selections: int
    planner_council_selections: int
    planner_phrase_selections: int
    planner_trace_created: int
    planner_trace_active: int
    planner_trace_feedback_tokens: int
    regime_generation: int
    surprise_mean: float
    surprise_cusum: float
    dialect_count: int
    dialect_neighbor_count: int
    dialect_neighbor_effective: float
    dialect_neighbor_max_similarity: float
    dialect_neighbor_ids: tuple[str, ...]
    active_dialect_id: str | None
    active_dialect_similarity: float
    active_dialect_plan_observations: tuple[int, ...]
    active_dialect_plan_hits: tuple[int, ...]
    active_dialect_planner_observations: tuple[tuple[int, ...], ...]
    active_dialect_planner_hits: tuple[tuple[int, ...], ...]
    dialect_evictions: int
    phrase_option_calls: int
    phrase_draft_tokens: int
    phrase_accepted_tokens: int
    periodic_option_calls: int
    periodic_draft_tokens: int
    periodic_accepted_tokens: int
    binding_option_calls: int
    binding_draft_tokens: int
    binding_accepted_tokens: int
    atlas_contexts: int
    atlas_corpus_tokens: int
    atlas_option_calls: int
    atlas_draft_tokens: int
    atlas_accepted_tokens: int
    atlas_vote_calls: int
    atlas_vote_tokens: int
    atlas_vote_supported_tokens: int
    atlas_vote_score_sum: float
    atlas_vote_max_score: float
    online_vote_calls: int
    online_vote_tokens: int
    online_vote_supported_tokens: int
    online_vote_score_sum: float
    online_vote_max_score: float
    retention_scored_episodes: int
    retention_failures: int
    retention_priority_evictions: int
    last_retention_priority: float
    ricci_working_set_builds: int
    ricci_working_set_selected_episodes: int
    ricci_working_set_selected_tokens: int
    ricci_working_set_oldest_age: int
    ricci_working_set_max_score: float
    composition_programs: int
    composition_option_calls: int
    composition_draft_tokens: int
    composition_accepted_tokens: int
    last_composition_support: int
    last_composition_copy_tokens: int
    last_phrase_source: str | None
    last_phrase_support: int
    last_phrase_confidence: float
    last_phrase_width: int
    adaptive_proposal_calls: int
    recommended_windows: tuple[tuple[int, int], ...]
    last_recommended_window: int | None
    last_horizon_utilities: tuple[tuple[int, float], ...]
    source_body_bytes: int = 0
    linear_calls: int = 0
    proposal_width: int = 3
    recursive_trace_created: int = 0
    recursive_trace_active: int = 0
    recursive_trace_peak_active: int = 0
    recursive_trace_feedback_tokens: int = 0
    recursive_trace_hits: int = 0
    recursive_trace_misses: int = 0
    recursive_trace_max_position: int = 0
    crystal_enabled: bool = False
    crystal_queries: int = 0
    crystal_query_hits: int = 0
    crystal_option_calls: int = 0
    crystal_proposed_tokens: int = 0
    crystal_verified_tokens: int = 0
    crystal_accepted_tokens: int = 0
    crystal_mismatches: int = 0
    crystal_captures: int = 0
    crystal_failures: int = 0
    crystal_bank_cells: int = 0
    crystal_bank_support: int = 0
    crystal_last_cosine: float = 0.0
    crystal_last_margin: float = 0.0
    crystal_last_cell_sha256: str | None = None
    layer_context_crystal: Mapping[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["expert_weights"] = dict(self.expert_weights)
        value["expert_accuracy"] = dict(self.expert_accuracy)
        value["last_position_weights"] = dict(self.last_position_weights)
        value["recommended_windows"] = dict(self.recommended_windows)
        value["last_horizon_utilities"] = dict(self.last_horizon_utilities)
        if self.layer_context_crystal is None:
            value.pop("layer_context_crystal")
        return value


class FingerprintRollingK4DraftProvider:
    """Draft from token IDs alone and learn only target-confirmed tokens.

    The decoder never passes this provider target tensors, weights, a pager, or
    a model handle, so callback guards inspect tensor versions without hashing
    target payloads.
    """

    target_state_isolation = "no-target-state-access/v1"

    EXPERT_LEARNING_RATE = 0.5
    RAPIDITY_DECAY = 0.95
    EXPERT_TEMPERATURE = 0.5
    FIXED_SHARE = 0.05
    SURPRISE_RATE = 0.05
    CUSUM_DECAY = 0.90
    CUSUM_DRIFT = 0.50
    CUSUM_THRESHOLD = 8.0
    REGIME_WARMUP = 16
    REGIME_RAPIDITY_SHRINK = 0.25
    MAX_DIALECTS = 64
    MAX_DIALECT_NEIGHBORS = 4
    MIN_DIALECT_NEIGHBOR_SIMILARITY = 0.05
    DIALECT_SKETCH_SIZE = 32
    DIALECT_SIMILARITY_THRESHOLD = 0.20
    DIALECT_STRENGTH = 2.0
    RICCI_AGE_ALPHA = 0.001
    PHRASE_MAX_CONTEXT = 8
    PHRASE_MAX_WIDTH = 15
    DIALECT_PHRASE_MIN_SUPPORT = 2
    GLOBAL_PHRASE_MIN_SUPPORT = 3
    COMPOSITION_BOUNDS = CompositionBounds()
    EMPIRICAL_EVIDENCE_SATURATION = 8.0
    POSITION_WEIGHT_SATURATION = 8.0
    LOOKAHEAD_CANDIDATES = 4
    LOOKAHEAD_DISCOUNT = 0.5
    LOOKAHEAD_MIN_LOG_GAIN = 0.05
    LOOKAHEAD_EMPIRICAL_STRENGTH = 0.25
    REQUEST_LOCAL_MIN_ORDER = 2
    REQUEST_LOCAL_MIN_SUPPORT = 2
    REQUEST_LOCAL_SUPPORT_SCALE = 1.5
    REQUEST_LOCAL_MAX_TOKENS = 4096
    REQUEST_PHRASE_MIN_CONTEXT = 2
    REQUEST_PHRASE_MIN_SUPPORT = 2
    REQUEST_PERIOD_MIN = 2
    REQUEST_PERIOD_MAX = 64
    REQUEST_PERIOD_MIN_SUPPORT = 3
    REQUEST_PERIOD_MATCH_THRESHOLD = 0.75
    ATLAS_MIN_SUPPORT = 2
    ATLAS_MIN_CONFIDENCE = 0.70

    def __init__(
        self,
        *,
        vocab_size: int,
        state_path: str | Path | None = None,
        max_order: int = 16,
        alpha: float = 0.5,
        backoff_strength: float = 3.0,
        min_count: int = 1,
        max_history_tokens: int = 65_536,
        proposal_width: int = 3,
        atlas: MarkovTokenAtlas | None = None,
        episode_scorer: Callable[[tuple[int, ...]], float] | None = None,
        episode_priority: Callable[[tuple[int, ...]], float] | None = None,
        episode_priority_store: (
            Callable[[tuple[int, ...], float], None] | None
        ) = None,
        contextual_continuation_bank: ContextualContinuationBank | None = None,
        layer_contextual_continuation_bank: (
            LayerContextualContinuationBank | None
        ) = None,
        layer_contextual_current_transaction: (
            Callable[[], LayerContextualContinuationTransaction | None] | None
        ) = None,
        layer_contextual_transactions_since: (
            Callable[
                [int],
                Sequence[LayerContextualContinuationTransaction],
            ]
            | None
        ) = None,
    ) -> None:
        if (
            isinstance(vocab_size, bool)
            or not isinstance(vocab_size, int)
            or vocab_size <= 1
        ):
            raise ValueError("vocab_size must be greater than one")
        if (
            isinstance(max_order, bool)
            or not isinstance(max_order, int)
            or max_order < 0
        ):
            raise ValueError("max_order must be a non-negative integer")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if not math.isfinite(backoff_strength) or backoff_strength <= 0:
            raise ValueError("backoff_strength must be finite and positive")
        if (
            isinstance(min_count, bool)
            or not isinstance(min_count, int)
            or min_count < 1
        ):
            raise ValueError("min_count must be a positive integer")
        if (
            isinstance(max_history_tokens, bool)
            or not isinstance(max_history_tokens, int)
            or max_history_tokens < max(8, max_order + 1)
        ):
            raise ValueError("max_history_tokens is too small")
        if state_path is not None and not isinstance(state_path, (str, Path)):
            raise TypeError("state_path must be a filesystem path or None")
        if (
            isinstance(proposal_width, bool)
            or not isinstance(proposal_width, int)
            or not 1 <= proposal_width <= 15
        ):
            raise ValueError("proposal_width must lie in [1, 15]")
        if atlas is not None and not isinstance(atlas, MarkovTokenAtlas):
            raise TypeError("atlas must be a MarkovTokenAtlas or None")
        if atlas is not None and atlas.vocab_size != vocab_size:
            raise ValueError("Markov atlas vocabulary differs from the provider")
        if episode_scorer is not None and not callable(episode_scorer):
            raise TypeError("episode_scorer must be callable or None")
        if episode_priority is not None and not callable(episode_priority):
            raise TypeError("episode_priority must be callable or None")
        if episode_priority_store is not None and not callable(
            episode_priority_store
        ):
            raise TypeError("episode_priority_store must be callable or None")
        if contextual_continuation_bank is not None and not isinstance(
            contextual_continuation_bank,
            ContextualContinuationBank,
        ):
            raise TypeError(
                "contextual_continuation_bank must be a "
                "ContextualContinuationBank or None"
            )
        layer_dependencies = (
            layer_contextual_continuation_bank,
            layer_contextual_current_transaction,
            layer_contextual_transactions_since,
        )
        if any(value is None for value in layer_dependencies) and any(
            value is not None for value in layer_dependencies
        ):
            raise ValueError(
                "layer contextual continuation dependencies must be provided "
                "together"
            )
        if layer_contextual_continuation_bank is not None and not isinstance(
            layer_contextual_continuation_bank,
            LayerContextualContinuationBank,
        ):
            raise TypeError(
                "layer_contextual_continuation_bank must be a "
                "LayerContextualContinuationBank or None"
            )
        if (
            layer_contextual_current_transaction is not None
            and not callable(layer_contextual_current_transaction)
        ):
            raise TypeError(
                "layer_contextual_current_transaction must be callable or None"
            )
        if (
            layer_contextual_transactions_since is not None
            and not callable(layer_contextual_transactions_since)
        ):
            raise TypeError(
                "layer_contextual_transactions_since must be callable or None"
            )
        self.vocab_size = vocab_size
        self.width = len(str(vocab_size - 1))
        self.state_path = (
            None if state_path is None else Path(state_path).expanduser().absolute()
        )
        self.max_order = max_order
        self.alpha = float(alpha)
        self.backoff_strength = float(backoff_strength)
        self.min_count = min_count
        self.max_history_tokens = max_history_tokens
        self.proposal_width = proposal_width
        self.atlas = atlas
        self.episode_scorer = episode_scorer
        self.episode_priority = episode_priority
        self.episode_priority_store = episode_priority_store
        self.contextual_continuation_bank = contextual_continuation_bank
        self.layer_contextual_continuation_bank = (
            layer_contextual_continuation_bank
        )
        self.layer_contextual_current_transaction = (
            layer_contextual_current_transaction
        )
        self.layer_contextual_transactions_since = (
            layer_contextual_transactions_since
        )
        self._experts = _expert_specs(max_order, max_history_tokens)
        self._state_lock_descriptor: int | None = None
        self._persisted_state: MarkovDraftState | None = None
        self._acquire_state_lock()
        try:
            self._state = self._load_state()
        except Exception:
            self._release_state_lock()
            raise
        names = tuple(row.name for row in self._experts)
        if not self._state.expert_names:
            uniform = -math.log(len(names))
            horizon_zeros = tuple(
                (0,) * len(names) for _ in range(_MAX_PROPOSAL_POSITIONS)
            )
            self._state = replace(
                self._state,
                expert_names=names,
                expert_log_weights=(uniform,) * len(names),
                expert_observations=(0,) * len(names),
                expert_hits=(0,) * len(names),
                horizon_expert_observations=horizon_zeros,
                horizon_expert_hits=horizon_zeros,
            )
        elif self._state.expert_names != names:
            self._release_state_lock()
            raise MarkovDraftError("Markov council topology changed")
        elif not self._state.horizon_expert_observations:
            horizon_zeros = tuple(
                (0,) * len(names) for _ in range(_MAX_PROPOSAL_POSITIONS)
            )
            self._state = replace(
                self._state,
                horizon_expert_observations=horizon_zeros,
                horizon_expert_hits=horizon_zeros,
            )
        if not self._state.lookahead_observations:
            self._state = replace(
                self._state,
                lookahead_observations=(0,) * _MAX_PROPOSAL_POSITIONS,
                lookahead_hits=(0,) * _MAX_PROPOSAL_POSITIONS,
                lookahead_greedy_hits=(0,) * _MAX_PROPOSAL_POSITIONS,
            )
        if not self._state.horizon_plan_observations:
            self._state = replace(
                self._state,
                horizon_plan_observations=(0,) * _MAX_PROPOSAL_POSITIONS,
                horizon_plan_hits=(0,) * _MAX_PROPOSAL_POSITIONS,
            )
        if not self._state.planner_observations:
            planner_zeros = tuple(
                (0,) * _MAX_PROPOSAL_POSITIONS for _ in _PLANNER_NAMES
            )
            self._state = replace(
                self._state,
                planner_observations=planner_zeros,
                planner_hits=planner_zeros,
            )
        if any(not row.horizon_observations for row in self._state.dialects):
            dialect_zeros = tuple(
                (0,) * len(names) for _ in range(_MAX_PROPOSAL_POSITIONS)
            )
            self._state = replace(
                self._state,
                dialects=tuple(
                    row
                    if row.horizon_observations
                    else replace(
                        row,
                        horizon_observations=dialect_zeros,
                        horizon_hits=dialect_zeros,
                    )
                    for row in self._state.dialects
                ),
            )
        if any(not row.plan_observations for row in self._state.dialects):
            plan_zeros = (0,) * _MAX_PROPOSAL_POSITIONS
            self._state = replace(
                self._state,
                dialects=tuple(
                    row
                    if row.plan_observations
                    else replace(
                        row,
                        plan_observations=plan_zeros,
                        plan_hits=plan_zeros,
                    )
                    for row in self._state.dialects
                ),
            )
        if any(not row.planner_observations for row in self._state.dialects):
            dialect_planner_zeros = tuple(
                (0,) * _MAX_PROPOSAL_POSITIONS for _ in _PLANNER_NAMES
            )
            self._state = replace(
                self._state,
                dialects=tuple(
                    row
                    if row.planner_observations
                    else replace(
                        row,
                        planner_observations=dialect_planner_zeros,
                        planner_hits=dialect_planner_zeros,
                    )
                    for row in self._state.dialects
                ),
            )
        self._pending_base: tuple[int, ...] | None = None
        self._pending_proposal: tuple[int, ...] | None = None
        self._pending_complete: tuple[int, ...] = ()
        self._pending_planner_candidates: tuple[tuple[int, ...], ...] = ()
        self._pending_feedback: tuple[
            tuple[tuple[dict[str, float], int], ...], ...
        ] = ()
        self._pending_plan_trace: tuple[tuple[int, int, float], ...] = ()
        self._pending_planner: str | None = None
        self._carry_feedback: tuple[tuple[dict[str, float], int], ...] | None = None
        self._carry_feedback_position: int | None = None
        self._carry_feedback_teacher_forced = False
        self._carry_plan: tuple[int, int, float] | None = None
        self._recursive_traces: list[_RecursiveMarkovTrace] = []
        self._recursive_feedback: list[
            tuple[tuple[int, ...], int, int, int, int]
        ] = []
        self._planner_traces: list[_PlannerTournamentTrace] = []
        self._planner_feedback: list[tuple[int, int, bool]] = []
        self._episode_feedback: list[
            tuple[
                tuple[tuple[dict[str, float], int], ...],
                int,
                int,
                int,
                int,
                float,
            ]
        ] = []
        self._active_dialect: MarkovDialectState | None = None
        self._dialect_neighbors: tuple[
            tuple[float, float, MarkovDialectState], ...
        ] = ()
        self._active_dialect_is_new = False
        self._request_signature: tuple[int, ...] = ()
        self._active_dialect_similarity = 0.0
        self._dialect_evictions = 0
        self._request_started = False
        self._request_completed = False
        self._request_prompt: tuple[int, ...] | None = None
        self._request_prompt_length: int | None = None
        self._pending_phrase_option: MarkovPhraseOption | None = None
        self._context_crystal_key: ContextualKey | None = None
        self._context_crystal_candidates: tuple[ContextualCandidate, ...] = ()
        self._context_crystal_captures: list[tuple[ContextualKey, int]] = []
        self._context_crystal_feedback: list[ContextualCandidateFeedback] = []
        self._crystal_queries = 0
        self._crystal_query_hits = 0
        self._crystal_option_calls = 0
        self._crystal_proposed_tokens = 0
        self._crystal_verified_tokens = 0
        self._crystal_accepted_tokens = 0
        self._crystal_mismatches = 0
        self._crystal_captures = 0
        self._crystal_failures = 0
        self._crystal_last_cosine = 0.0
        self._crystal_last_margin = 0.0
        self._crystal_last_cell_sha256: str | None = None
        self._layer_context_crystal_request_start_boundary: int | None = None
        self._layer_context_crystal_request_enabled = False
        self._layer_context_crystal_options_by_transaction: dict[
            str,
            tuple[LayerContextualContinuationOption, ...],
        ] = {}
        self._layer_context_crystal_transactions: dict[
            str,
            LayerContextualContinuationTransaction,
        ] = {}
        self._layer_context_crystal_phrase_options: tuple[
            MarkovPhraseOption, ...
        ] = ()
        self._layer_context_crystal_counters: Counter[str] = Counter()
        self._layer_context_crystal_layer_counters: dict[
            int, Counter[str]
        ] = {}
        self._layer_context_crystal_last_query: tuple[
            Mapping[str, object], ...
        ] = ()
        self._layer_context_crystal_selected: Mapping[str, object] | None = None
        self._layer_context_crystal_last_cosine = 0.0
        self._layer_context_crystal_last_margin = 0.0
        self._layer_context_crystal_last_cell_sha256: str | None = None
        self._phrase_option_calls = 0
        self._phrase_draft_tokens = 0
        self._phrase_accepted_tokens = 0
        self._periodic_option_calls = 0
        self._periodic_draft_tokens = 0
        self._periodic_accepted_tokens = 0
        self._binding_option_calls = 0
        self._binding_draft_tokens = 0
        self._binding_accepted_tokens = 0
        self._atlas_option_calls = 0
        self._atlas_draft_tokens = 0
        self._atlas_accepted_tokens = 0
        self._atlas_vote_calls = 0
        self._atlas_vote_tokens = 0
        self._atlas_vote_supported_tokens = 0
        self._atlas_vote_score_sum = 0.0
        self._atlas_vote_max_score = 0.0
        self._online_vote_calls = 0
        self._online_vote_tokens = 0
        self._online_vote_supported_tokens = 0
        self._online_vote_score_sum = 0.0
        self._online_vote_max_score = 0.0
        self._retention_scored_episodes = 0
        self._retention_failures = 0
        self._retention_priority_evictions = 0
        self._last_retention_priority = 1.0
        self._ricci_working_set_builds = 0
        self._ricci_working_set_selected_episodes = 0
        self._ricci_working_set_selected_tokens = 0
        self._ricci_working_set_oldest_age = 0
        self._ricci_working_set_max_score = 0.0
        self._last_phrase_option: MarkovPhraseOption | None = None
        self._composition_cache: dict[
            str | None,
            tuple[MarkovCompositionProgram, ...],
        ] = {}
        self._composition_program_count = 0
        self._composition_option_calls = 0
        self._composition_draft_tokens = 0
        self._composition_accepted_tokens = 0
        self._last_composition_program: MarkovCompositionProgram | None = None
        self._pending_composition_program: MarkovCompositionProgram | None = None
        self._last_confirmed_length: int | None = None
        self._draft_calls = 0
        self._reconcile_calls = 0
        self._predictions = 0
        self._council_predictions = 0
        self._council_feedback = 0
        self._request_expert_rapidities: tuple[float, ...] | None = None
        self._request_weight_updates = 0
        self._max_request_weight_shift = 0.0
        self._request_horizon_observations: list[list[int]] | None = None
        self._request_horizon_hits: list[list[int]] | None = None
        self._request_plan_observations: list[int] | None = None
        self._request_plan_hits: list[int] | None = None
        self._request_planner_observations: list[list[int]] | None = None
        self._request_planner_hits: list[list[int]] | None = None
        self._request_position_updates = 0
        self._max_request_position_maturity = 0.0
        self._request_lookahead_observations: list[int] | None = None
        self._request_lookahead_hits: list[int] | None = None
        self._request_lookahead_greedy_hits: list[int] | None = None
        self._request_lookahead_updates = 0
        self._request_surprise_mean: float | None = None
        self._request_surprise_deviation: float | None = None
        self._request_surprise_cusum: float | None = None
        self._request_feedback_count = 0
        self._request_regime_changes = 0
        self._external_reconcile_calls = 0
        self._external_feedback_tokens = 0
        self._teacher_forced_predictions = 0
        self._teacher_forced_feedback_tokens = 0
        self._teacher_forced_failures = 0
        self._recursive_trace_created = 0
        self._recursive_trace_peak_active = 0
        self._recursive_trace_feedback_tokens = 0
        self._recursive_trace_hits = 0
        self._recursive_trace_misses = 0
        self._recursive_trace_max_position = 0
        self._planner_tournament_calls = 0
        self._planner_beam_selections = 0
        self._planner_council_selections = 0
        self._planner_phrase_selections = 0
        self._planner_trace_created = 0
        self._planner_trace_feedback_tokens = 0
        self._last_confidence = 0.0
        self._last_raw_confidence = 0.0
        self._last_empirical_evidence = 0.0
        self._last_disagreement = 0.0
        self._last_position = 0
        self._last_position_maturity = 0.0
        self._last_dialect_skill_maturity = 0.0
        self._last_position_weights = self._weights()
        self._position_specialist_predictions = 0
        self._dialect_specialist_predictions = 0
        self._max_position_maturity = 0.0
        self._max_dialect_skill_maturity = 0.0
        self._lookahead_calls = 0
        self._lookahead_candidates = 0
        self._lookahead_token_changes = 0
        self._last_lookahead_gain = 0.0
        self._max_lookahead_gain = 0.0
        self._adaptive_proposal_calls = 0
        self._recommended_window_counts = {1: 0, 4: 0, 8: 0, 16: 0}
        self._last_round_proposal: RollingDraftProposal | None = None
        self._last_plan_trace: tuple[tuple[int, int, float], ...] = ()
        self._last_beam_prefix_posteriors: tuple[float, ...] = ()
        self._last_beam_path_count = 0
        self._beam_verified_tokens = 0
        self._beam_accepted_tokens = 0
        self._beam_position_verified = [0] * _MAX_PROPOSAL_POSITIONS
        self._beam_position_hits = [0] * _MAX_PROPOSAL_POSITIONS
        self._pending_accepted_prefix_length: int | None = None
        self._pending_verified_proposals: int | None = None
        self._pending_verification_virtual = False
        self._pending_import_digest: str | None = None
        self._persistent_symbols_cache: tuple[str, ...] | None = None
        self._ricci_working_symbols_cache: dict[int, tuple[str, ...]] = {}
        self._ricci_episode_cache: tuple[
            tuple[int, tuple[int, ...], float], ...
        ] | None = None
        self._ricci_priority_degraded = False
        self._episode_priority_cache: dict[tuple[int, ...], float] = {}
        self._persistent_expert_models: dict[str, _TransitionFingerprint] = {}
        self._request_local_cache_history: tuple[int, ...] | None = None
        self._request_local_cache: (
            tuple[_TransitionFingerprint, tuple[str, ...]] | None
        ) = None
        self._closed = False

    def _acquire_state_lock(self) -> None:
        path = self.state_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.parent / f".{path.name}.lock"
        descriptor = os.open(
            lock,
            os.O_CREAT
            | os.O_RDWR
            | int(getattr(os, "O_CLOEXEC", 0))
            | int(getattr(os, "O_NOFOLLOW", 0)),
            0o600,
        )
        try:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
        except Exception:
            os.close(descriptor)
            raise
        self._state_lock_descriptor = descriptor

    def _release_state_lock(self) -> None:
        descriptor = self._state_lock_descriptor
        self._state_lock_descriptor = None
        if descriptor is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def _load_state(self) -> MarkovDraftState:
        empty = MarkovDraftState(
            vocab_size=self.vocab_size,
            max_history_tokens=self.max_history_tokens,
            token_ids=(),
        )
        path = self.state_path
        if path is None or (not path.exists() and not path.is_symlink()):
            return empty
        try:
            raw = _read_state_bytes(path)
            state = MarkovDraftState.from_bytes(raw)
        except OSError as exc:  # pragma: no cover - normalized by helper.
            raise MarkovDraftError("cannot read Markov draft state") from exc
        if state.vocab_size != self.vocab_size:
            raise MarkovDraftError("Markov draft state configuration changed")
        if state.max_history_tokens > self.max_history_tokens:
            raise MarkovDraftError("Markov draft history capacity cannot shrink")
        self._persisted_state = state if raw.startswith(_STATE_PREFIX) else None
        if state.max_history_tokens < self.max_history_tokens:
            state = replace(
                state,
                max_history_tokens=self.max_history_tokens,
            )
        return state

    def _token_tuple(self, value: object, *, label: str) -> tuple[int, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"{label} must be an integer sequence")
        result = tuple(value)
        if not result or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or not 0 <= token < self.vocab_size
            for token in result
        ):
            raise ValueError(f"{label} contains invalid token IDs")
        return result

    def _symbol(self, token: int) -> str:
        return f"{token:0{self.width}d}"

    def _context_signature(self, history: Sequence[int]) -> tuple[int, ...]:
        tokens = tuple(history)
        features: set[int] = set()
        for order in (1, 2, 3):
            for start in range(max(0, len(tokens) - 512), len(tokens) - order + 1):
                material = bytearray((order,))
                for token in tokens[start : start + order]:
                    material.extend(int(token).to_bytes(4, "little", signed=False))
                digest = hashlib.blake2b(
                    material,
                    digest_size=8,
                    person=b"IMMRDIA",
                ).digest()
                features.add(int.from_bytes(digest, "big"))
        length_bucket = min(255, len(tokens).bit_length())
        features.add(
            int.from_bytes(
                hashlib.blake2b(
                    bytes((255, length_bucket)),
                    digest_size=8,
                    person=b"IMMRDIA",
                ).digest(),
                "big",
            )
        )
        return tuple(sorted(features)[: self.DIALECT_SKETCH_SIZE])

    @staticmethod
    def _dialect_similarity(left: Sequence[int], right: Sequence[int]) -> float:
        first = set(left)
        second = set(right)
        union = first | second
        return 0.0 if not union else len(first & second) / len(union)

    def _activate_dialect(self, history: tuple[int, ...]) -> None:
        if self._active_dialect is not None:
            return
        signature = self._context_signature(history)
        self._request_signature = signature
        candidates = [
            (
                self._dialect_similarity(signature, row.signature),
                row.visits,
                row.dialect_id,
                row,
            )
            for row in self._state.dialects
        ]
        raw_neighbor_candidates = tuple(
            (
                similarity,
                similarity
                * profile.visits
                * math.exp(
                    -self.RICCI_AGE_ALPHA
                    * max(0, self._state.clock - profile.last_seen)
                ),
                visits,
                dialect_id,
                profile,
            )
            for similarity, visits, dialect_id, profile in candidates
            if similarity >= self.MIN_DIALECT_NEIGHBOR_SIMILARITY
        )
        raw_neighbors = tuple(
            sorted(
                raw_neighbor_candidates,
                key=lambda row: (row[1], row[0], row[2], row[3]),
                reverse=True,
            )[: self.MAX_DIALECT_NEIGHBORS]
        )
        neighbor_total = sum(row[1] for row in raw_neighbors)
        self._dialect_neighbors = (
            ()
            if neighbor_total <= 0.0
            else tuple(
                (similarity, raw_weight / neighbor_total, profile)
                for similarity, raw_weight, _visits, _dialect_id, profile in raw_neighbors
            )
        )
        selected = max(candidates, default=None)
        if selected is not None and selected[0] >= self.DIALECT_SIMILARITY_THRESHOLD:
            active_profile = selected[3]
            if all(
                row[2].dialect_id != active_profile.dialect_id
                for row in self._dialect_neighbors
            ):
                active_raw = next(
                    row
                    for row in raw_neighbor_candidates
                    if row[4].dialect_id == active_profile.dialect_id
                )
                selected_raw = (
                    active_raw,
                    *tuple(
                        row
                        for row in raw_neighbors
                        if row[4].dialect_id != active_profile.dialect_id
                    )[: self.MAX_DIALECT_NEIGHBORS - 1],
                )
                selected_total = sum(row[1] for row in selected_raw)
                self._dialect_neighbors = tuple(
                    (similarity, raw_weight / selected_total, profile)
                    for similarity, raw_weight, _visits, _dialect_id, profile in (
                        selected_raw
                    )
                )
            self._active_dialect_similarity = selected[0]
            self._active_dialect = active_profile
            self._active_dialect_is_new = False
            return
        self._dialect_neighbors = ()
        dialect_id = hashlib.sha256(
            _canonical([f"{value:016x}" for value in signature])
        ).hexdigest()
        self._active_dialect = MarkovDialectState(
            dialect_id=dialect_id,
            signature=signature,
            visits=1,
            last_seen=self._state.clock + 1,
            rapidities=(0.0,) * len(self._experts),
            observations=(0,) * len(self._experts),
            hits=(0,) * len(self._experts),
            horizon_observations=tuple(
                (0,) * len(self._experts)
                for _ in range(_MAX_PROPOSAL_POSITIONS)
            ),
            horizon_hits=tuple(
                (0,) * len(self._experts)
                for _ in range(_MAX_PROPOSAL_POSITIONS)
            ),
            plan_observations=(0,) * _MAX_PROPOSAL_POSITIONS,
            plan_hits=(0,) * _MAX_PROPOSAL_POSITIONS,
            planner_observations=tuple(
                (0,) * _MAX_PROPOSAL_POSITIONS for _ in _PLANNER_NAMES
            ),
            planner_hits=tuple(
                (0,) * _MAX_PROPOSAL_POSITIONS for _ in _PLANNER_NAMES
            ),
        )
        self._active_dialect_similarity = 0.0
        self._active_dialect_is_new = True

    def _inference_dialects(
        self,
    ) -> tuple[tuple[float, float, MarkovDialectState], ...]:
        if self._dialect_neighbors:
            active = self._active_dialect
            return tuple(
                (
                    similarity,
                    weight,
                    active
                    if active is not None
                    and profile.dialect_id == active.dialect_id
                    else profile,
                )
                for similarity, weight, profile in self._dialect_neighbors
            )
        if self._active_dialect is not None and self._active_dialect_similarity > 0.0:
            return (
                (
                    self._active_dialect_similarity,
                    1.0,
                    self._active_dialect,
                ),
            )
        return ()

    def _combined_rapidities(self) -> tuple[float, ...]:
        rapidities = self._state.expert_log_weights
        dialects = self._inference_dialects()
        if dialects:
            rapidities = tuple(
                global_value
                + self.DIALECT_STRENGTH
                * sum(
                    neighbor_weight * profile.rapidities[index]
                    for _similarity, neighbor_weight, profile in dialects
                )
                for index, global_value in enumerate(rapidities)
            )
        return rapidities

    def _weights(self) -> tuple[float, ...]:
        rapidities = self._request_expert_rapidities
        if rapidities is None:
            rapidities = self._combined_rapidities()
        scaled = tuple(value / self.EXPERT_TEMPERATURE for value in rapidities)
        maximum = max(scaled)
        raw = tuple(math.exp(value - maximum) for value in scaled)
        total = sum(raw)
        count = len(raw)
        return tuple(
            (1.0 - self.FIXED_SHARE) * value / total + self.FIXED_SHARE / count
            for value in raw
        )

    def _validate_layer_context_crystal_transaction(
        self,
        transaction: object,
        *,
        label: str,
    ) -> LayerContextualContinuationTransaction:
        bank = self.layer_contextual_continuation_bank
        if bank is None:
            raise MarkovDraftError("layer context Crystal bank is not configured")
        if not isinstance(transaction, LayerContextualContinuationTransaction):
            raise TypeError(f"{label} must return a layer continuation transaction")
        identity = bank.identity
        if (
            transaction.identity_sha256 != identity.identity_sha256
            or transaction.layers != identity.layers
            or any(key.sketch_dim != identity.sketch_dim for key in transaction.keys)
        ):
            raise MarkovDraftError(
                f"{label} returned a transaction for another bank identity"
            )
        return transaction

    def _current_layer_context_crystal_transaction(
        self,
        history: tuple[int, ...],
    ) -> LayerContextualContinuationTransaction:
        source = self.layer_contextual_current_transaction
        if source is None:
            raise MarkovDraftError("layer context Crystal source is not configured")
        transaction = self._validate_layer_context_crystal_transaction(
            source(),
            label="layer_contextual_current_transaction",
        )
        if (
            not history
            or transaction.boundary_index != len(history) - 1
            or transaction.known_token != history[-1]
        ):
            raise MarkovDraftError(
                "layer context Crystal transaction differs from the history "
                "boundary"
            )
        return transaction

    def _reset_layer_context_crystal_request(self) -> None:
        self._layer_context_crystal_request_enabled = False
        self._layer_context_crystal_request_start_boundary = None
        self._layer_context_crystal_options_by_transaction.clear()
        self._layer_context_crystal_transactions.clear()
        self._layer_context_crystal_phrase_options = ()
        self._layer_context_crystal_counters.clear()
        self._layer_context_crystal_layer_counters.clear()
        self._layer_context_crystal_last_query = ()
        self._layer_context_crystal_selected = None
        self._layer_context_crystal_last_cosine = 0.0
        self._layer_context_crystal_last_margin = 0.0
        self._layer_context_crystal_last_cell_sha256 = None

    def _layer_context_crystal_layer_counter(self, layer: int) -> Counter[str]:
        return self._layer_context_crystal_layer_counters.setdefault(
            layer,
            Counter(),
        )

    def _record_layer_context_crystal_failure(
        self,
        layers: Sequence[int] | None = None,
    ) -> None:
        bank = self.layer_contextual_continuation_bank
        selected = (
            ()
            if bank is None
            else bank.identity.layers
            if layers is None
            else tuple(layers)
        )
        for layer in selected:
            self._layer_context_crystal_layer_counter(layer)[
                "crystal_failures"
            ] += 1
        self._layer_context_crystal_counters["crystal_failures"] += max(
            1,
            len(selected),
        )

    def begin_request(self, history: tuple[int, ...], /) -> None:
        if self._closed:
            raise MarkovDraftError("Markov draft provider is closed")
        if self._request_started or self._request_completed:
            raise MarkovDraftError("Markov provider accepts exactly one request")
        committed = self._token_tuple(history, label="Markov request history")
        self._reset_layer_context_crystal_request()
        layer_start = None
        if self.layer_contextual_continuation_bank is not None:
            try:
                layer_start = self._current_layer_context_crystal_transaction(
                    committed
                )
            except Exception:
                self._record_layer_context_crystal_failure()
            else:
                self._layer_context_crystal_request_enabled = True
        self._activate_dialect(committed)
        self._request_prompt = committed
        self._request_prompt_length = len(committed)
        self._request_local_cache_history = None
        self._request_local_cache = None
        self._context_crystal_key = None
        self._context_crystal_candidates = ()
        self._context_crystal_captures.clear()
        self._context_crystal_feedback.clear()
        self._layer_context_crystal_request_start_boundary = (
            None if layer_start is None else layer_start.boundary_index
        )
        self._request_started = True

    def _load_context_crystal_boundary(
        self,
        history: tuple[int, ...],
        known_token: int,
        target_hidden: torch.Tensor,
    ) -> None:
        self._context_crystal_key = None
        self._context_crystal_candidates = ()
        bank = self.contextual_continuation_bank
        if bank is None:
            return
        self._crystal_queries += 1
        try:
            key = bank.project(target_hidden, known_token)
            candidates = bank.query_key(key, limit=8)
        except Exception:
            self._crystal_failures += 1
            return
        self._context_crystal_key = key
        self._context_crystal_candidates = candidates
        self._context_crystal_captures.append((key, len(history)))
        if candidates:
            self._crystal_query_hits += 1
            self._crystal_last_cosine = float(candidates[0].cosine)
            self._crystal_last_margin = float(candidates[0].margin or 0.0)
            self._crystal_last_cell_sha256 = candidates[0].cell_sha256

    @staticmethod
    def _context_crystal_probabilities(
        candidate: ContextualCandidate,
        width: int,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        semantic = max(0.0, min(0.999, (float(candidate.cosine) + 1.0) / 2.0))
        separation = (
            1.0
            if candidate.margin is None
            else max(0.0, min(1.0, float(candidate.margin) / 0.15))
        )
        prior = min(0.999, semantic * (0.8 + 0.2 * separation))
        confidences = []
        disagreements = []
        for position in range(width):
            verified = candidate.position_verified[position]
            hits = candidate.position_hits[position]
            maturity = verified / (verified + 4.0)
            posterior = (hits + 1.0) / (verified + 2.0)
            confidence = (1.0 - maturity) * prior + maturity * posterior
            confidences.append(max(0.0, min(0.999, confidence)))
            disagreements.append(
                max(
                    0.0,
                    min(
                        1.0,
                        (1.0 - separation) * (1.0 - maturity)
                        + maturity * (1.0 - posterior),
                    ),
                )
            )
        return tuple(confidences), tuple(disagreements)

    def _context_crystal_option(
        self,
        history: tuple[int, ...],
    ) -> MarkovPhraseOption | None:
        if self._context_crystal_key is None or not self._context_crystal_candidates:
            return None
        candidate = self._context_crystal_candidates[0]
        width = min(len(candidate.target_tail), self.proposal_width)
        if width < 1:
            return None
        confidences, disagreements = self._context_crystal_probabilities(
            candidate,
            width,
        )
        return MarkovPhraseOption(
            token_ids=candidate.target_tail[:width],
            source="crystal",
            context_order=max(1, min(self.PHRASE_MAX_CONTEXT, len(history))),
            support=max(1, candidate.support),
            total=max(1, candidate.support),
            kind="crystal",
            confidence_override=min(confidences),
            token_confidences=confidences,
            token_disagreements=disagreements,
            cell_sha256=candidate.cell_sha256,
        )

    def _load_layer_context_crystal_boundary(
        self,
        history: tuple[int, ...],
        known_token: int,
    ) -> None:
        self._layer_context_crystal_phrase_options = ()
        bank = self.layer_contextual_continuation_bank
        if bank is None or not self._layer_context_crystal_request_enabled:
            return
        try:
            transaction = self._current_layer_context_crystal_transaction(history)
        except Exception:
            self._record_layer_context_crystal_failure()
            self._layer_context_crystal_request_enabled = False
            return
        try:
            raw_options = bank.query_options(transaction, limit_per_layer=1)
            options = tuple(raw_options)
            if any(
                not isinstance(option, LayerContextualContinuationOption)
                or option.transaction_sha256 != transaction.transaction_sha256
                or option.layer not in transaction.layers
                or option.known_token != transaction.known_token
                or any(token >= self.vocab_size for token in option.target_tail)
                for option in options
            ):
                raise MarkovDraftError(
                    "layer context Crystal query returned an invalid option"
                )
            if len({option.layer for option in options}) != len(options):
                raise MarkovDraftError(
                    "layer context Crystal query returned duplicate layer options"
                )
        except Exception:
            self._record_layer_context_crystal_failure(transaction.layers)
            return

        self._layer_context_crystal_transactions[
            transaction.transaction_sha256
        ] = transaction
        previous = self._layer_context_crystal_options_by_transaction.get(
            transaction.transaction_sha256,
            (),
        )
        merged = {
            (option.layer, option.cell_sha256): option
            for option in (*previous, *options)
        }
        self._layer_context_crystal_options_by_transaction[
            transaction.transaction_sha256
        ] = tuple(merged[key] for key in sorted(merged))

        by_layer = {option.layer: option for option in options}
        phrase_options: list[MarkovPhraseOption] = []
        query_evidence: list[Mapping[str, object]] = []
        for layer in transaction.layers:
            layer_counter = self._layer_context_crystal_layer_counter(layer)
            layer_counter["crystal_queries"] += 1
            layer_counter["crystal_option_calls"] += 1
            self._layer_context_crystal_counters["crystal_queries"] += 1
            self._layer_context_crystal_counters["crystal_option_calls"] += 1
            option = by_layer.get(layer)
            if option is None:
                continue
            layer_counter["crystal_query_hits"] += 1
            self._layer_context_crystal_counters["crystal_query_hits"] += 1
            margin = float(option.margin or 0.0)
            layer_counter["crystal_last_cosine"] = option.cosine
            layer_counter["crystal_last_margin"] = margin
            layer_counter["crystal_last_cell_sha256"] = option.cell_sha256
            self._layer_context_crystal_last_cosine = option.cosine
            self._layer_context_crystal_last_margin = margin
            self._layer_context_crystal_last_cell_sha256 = option.cell_sha256

            # The model transaction is for history[-1].  Its stored tail starts
            # with the target-known token supplied to this callback; only the
            # suffix after that token is speculative.
            eligible = option.target_tail[0] == known_token and option.width > 1
            planner_tokens: tuple[int, ...] = ()
            if eligible:
                width = option.width - 1
                planner_tokens = option.target_tail[1 : 1 + width]
                confidences, disagreements = (
                    self._context_crystal_probabilities(option, option.width)
                )
                phrase_options.append(
                    MarkovPhraseOption(
                        token_ids=planner_tokens,
                        source="crystal",
                        context_order=max(
                            1,
                            min(self.PHRASE_MAX_CONTEXT, len(history) + 1),
                        ),
                        support=max(1, option.support),
                        total=max(1, option.support),
                        kind="crystal",
                        confidence_override=min(confidences[1 : 1 + width]),
                        token_confidences=confidences[1 : 1 + width],
                        token_disagreements=disagreements[1 : 1 + width],
                        cell_sha256=option.cell_sha256,
                        crystal_layer=option.layer,
                        crystal_transaction_sha256=(
                            option.transaction_sha256
                        ),
                    )
                )
            query_evidence.append(
                option.to_dict()
                | {
                    "planner_eligible": eligible,
                    "planner_token_ids": list(planner_tokens),
                }
            )
        consensus: dict[tuple[int, ...], list[MarkovPhraseOption]] = {}
        for phrase in phrase_options:
            consensus.setdefault(phrase.token_ids, []).append(phrase)
        consensus_layers = {
            token_ids: tuple(
                sorted(
                    phrase.crystal_layer
                    for phrase in members
                    if phrase.crystal_layer is not None
                )
            )
            for token_ids, members in consensus.items()
        }
        weighted_options: list[MarkovPhraseOption] = []
        for token_ids, members in consensus.items():
            support = sum(phrase.support for phrase in members)
            total = sum(phrase.total for phrase in members)
            weighted_options.extend(
                replace(
                    phrase,
                    support=support,
                    total=total,
                )
                for phrase in members
            )
        self._layer_context_crystal_phrase_options = tuple(weighted_options)
        self._layer_context_crystal_last_query = tuple(
            dict(row)
            | {
                "consensus_layers": list(
                    consensus_layers.get(
                        tuple(row.get("planner_token_ids", ())),
                        (),
                    )
                ),
                "consensus_size": len(
                    consensus_layers.get(
                        tuple(row.get("planner_token_ids", ())),
                        (),
                    )
                ),
            }
            for row in query_evidence
        )

    def _persistent_symbols(self) -> tuple[str, ...]:
        cached = self._persistent_symbols_cache
        if cached is not None:
            return cached
        rows: list[str] = []
        for index, episode in enumerate(self._generation_episodes()):
            if index:
                rows.append(_EPISODE_TOKEN)
            rows.extend(self._symbol(token) for token in episode)
        result = tuple(rows)
        self._persistent_symbols_cache = result
        return result

    def _retention_priority(self, generated: tuple[int, ...]) -> float:
        """Read one bounded O1 priority with a deterministic neutral fallback."""

        cached = self._episode_priority_cache.get(generated)
        if cached is not None:
            return cached
        try:
            priority = (
                1.0
                if self.episode_priority is None
                else float(self.episode_priority(generated))
            )
        except Exception:
            priority = 1.0
            self._retention_failures += 1
            return priority
        if not math.isfinite(priority) or priority < 0.0:
            priority = 1.0
            self._retention_failures += 1
            return priority
        self._episode_priority_cache[generated] = priority
        return priority

    def _ricci_episodes(
        self,
    ) -> tuple[tuple[int, tuple[int, ...], float], ...]:
        cached = self._ricci_episode_cache
        if cached is not None:
            return cached
        episodes = self._generation_episodes()
        latest = len(episodes) - 1
        failures_before = self._retention_failures
        rows = tuple(
            (
                index,
                episode,
                self._retention_priority(episode)
                * math.exp(
                    -self.RICCI_AGE_ALPHA * max(0, latest - index)
                ),
            )
            for index, episode in enumerate(episodes)
        )
        self._ricci_priority_degraded = (
            self._retention_failures > failures_before
        )
        self._ricci_episode_cache = rows
        return rows

    def _ricci_working_symbols(self, window: int) -> tuple[str, ...]:
        """Project the highest-value whole answer episodes into one PPM window."""

        if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
            raise MarkovDraftError("Ricci working-set window is invalid")
        persistent = self._persistent_symbols()
        if self.episode_priority is None or not persistent:
            return persistent[-window:]
        cached = self._ricci_working_symbols_cache.get(window)
        if cached is not None:
            return cached
        episodes = self._ricci_episodes()
        ranked = tuple(
            sorted(
                episodes,
                key=lambda row: (-row[2], -row[0], row[1]),
            )
        )
        capacity = window + 1
        by_cost: dict[int, list[tuple[int, tuple[int, ...], float]]] = {}
        for row in episodes:
            cost = len(row[1]) + 1
            if cost <= capacity:
                by_cost.setdefault(cost, []).append(row)
        candidates = []
        for cost, rows in by_cost.items():
            candidates.extend(
                sorted(
                    rows,
                    key=lambda row: (-row[2], -row[0], row[1]),
                )[: capacity // cost]
            )
        candidates.sort(key=lambda row: row[0])
        scores = [-math.inf] * (capacity + 1)
        recencies = [-1] * (capacity + 1)
        counts = [-1] * (capacity + 1)
        scores[0] = 0.0
        recencies[0] = 0
        counts[0] = 0
        update_masks: list[bytes] = []
        mask_bytes = (capacity + 8) // 8
        for row in candidates:
            cost = len(row[1]) + 1
            updates = bytearray(mask_bytes)
            for used in range(capacity, cost - 1, -1):
                previous_score = scores[used - cost]
                if previous_score == -math.inf:
                    continue
                candidate_score = previous_score + row[2]
                candidate_recency = recencies[used - cost] + row[0]
                candidate_count = counts[used - cost] + 1
                if (
                    candidate_score,
                    candidate_recency,
                    candidate_count,
                ) > (
                    scores[used],
                    recencies[used],
                    counts[used],
                ):
                    scores[used] = candidate_score
                    recencies[used] = candidate_recency
                    counts[used] = candidate_count
                    updates[used >> 3] |= 1 << (used & 7)
            update_masks.append(bytes(updates))
        best_used = max(
            (
                used
                for used in range(1, capacity + 1)
                if scores[used] != -math.inf
            ),
            key=lambda used: (
                scores[used],
                used,
                recencies[used],
                counts[used],
            ),
            default=0,
        )
        selected = []
        used = best_used
        if scores[used] != -math.inf:
            for candidate_index in range(len(candidates) - 1, -1, -1):
                mask = update_masks[candidate_index]
                if not (mask[used >> 3] & (1 << (used & 7))):
                    continue
                row = candidates[candidate_index]
                selected.append(row)
                used -= len(row[1]) + 1
                if used == 0:
                    break
        if used != 0:
            raise MarkovDraftError(
                "Ricci working-set optimization lost its parent path"
            )
        if not selected and ranked:
            index, episode, score = ranked[0]
            selected = [(index, episode[-window:], score)]
        selected.sort(key=lambda row: row[0])
        symbols: list[str] = []
        for _index, episode, _score in selected:
            if symbols:
                symbols.append(_EPISODE_TOKEN)
            symbols.extend(self._symbol(token) for token in episode)
        result = tuple(symbols)
        if len(result) > window:
            raise MarkovDraftError("Ricci working set exceeds its expert window")
        latest = len(episodes) - 1
        selected_tokens = sum(len(row[1]) for row in selected)
        self._ricci_working_set_builds += 1
        self._ricci_working_set_selected_episodes = max(
            self._ricci_working_set_selected_episodes,
            len(selected),
        )
        self._ricci_working_set_selected_tokens = max(
            self._ricci_working_set_selected_tokens,
            selected_tokens,
        )
        self._ricci_working_set_oldest_age = max(
            self._ricci_working_set_oldest_age,
            max((latest - row[0] for row in selected), default=0),
        )
        self._ricci_working_set_max_score = max(
            self._ricci_working_set_max_score,
            max((row[2] for row in selected), default=0.0),
        )
        self._ricci_working_symbols_cache[window] = result
        return result

    def _episodes(self, dialect_id: str | None = None) -> tuple[tuple[int, ...], ...]:
        rows = []
        offset = 0
        for length, bound_dialect in zip(
            self._state.episode_lengths,
            self._state.episode_dialects,
            strict=True,
        ):
            episode = self._state.token_ids[offset : offset + length]
            offset += length
            if dialect_id is None or bound_dialect == dialect_id:
                rows.append(episode)
        return tuple(rows)

    def _generation_episodes(
        self,
        dialect_id: str | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        """Return the Qwen-generated side of every confirmed transition.

        The Council is a draft language model, so its persistent PPM corpus must
        contain Qwen's answers rather than repeated chat templates and arbitrary
        user prompts.  Legacy promptless episodes remain usable as answer-only
        rows; structured episodes start exactly at their stored prompt boundary.
        """

        rows = []
        offset = 0
        for length, bound_dialect, prompt_length in zip(
            self._state.episode_lengths,
            self._state.episode_dialects,
            self._state.episode_prompt_lengths,
            strict=True,
        ):
            episode = self._state.token_ids[offset : offset + length]
            offset += length
            if dialect_id is not None and bound_dialect != dialect_id:
                continue
            generated = episode if prompt_length is None else episode[prompt_length:]
            if generated:
                rows.append(generated)
        return tuple(rows)

    def _structured_episodes(
        self,
        dialect_id: str | None = None,
    ) -> tuple[ConfirmedTokenEpisode, ...]:
        rows = []
        offset = 0
        for length, bound_dialect, prompt_length in zip(
            self._state.episode_lengths,
            self._state.episode_dialects,
            self._state.episode_prompt_lengths,
            strict=True,
        ):
            episode = self._state.token_ids[offset : offset + length]
            offset += length
            if prompt_length is None or (
                dialect_id is not None and bound_dialect != dialect_id
            ):
                continue
            rows.append(
                ConfirmedTokenEpisode(
                    prompt=episode[:prompt_length],
                    output=episode[prompt_length:],
                )
            )
        return tuple(rows)

    def _composition_programs(
        self,
        dialect_id: str | None,
    ) -> tuple[MarkovCompositionProgram, ...]:
        if dialect_id not in self._composition_cache:
            self._composition_cache[dialect_id] = derive_programs(
                self._structured_episodes(dialect_id),
                self.COMPOSITION_BOUNDS,
            )
            self._composition_program_count = len(
                {
                    row.canonical_key
                    for programs in self._composition_cache.values()
                    for row in programs
                }
            )
        return self._composition_cache[dialect_id]

    def _composition_option_from(
        self,
        history: tuple[int, ...],
        *,
        source: str,
        dialect_id: str | None,
    ) -> tuple[MarkovPhraseOption, MarkovCompositionProgram] | None:
        prompt = self._request_prompt
        prompt_length = self._request_prompt_length
        if (
            prompt is None
            or prompt_length is None
            or history[:prompt_length] != prompt
            or len(history) <= prompt_length
        ):
            return None
        confirmed_output = history[prompt_length:]
        matches: list[tuple[tuple[int, ...], MarkovCompositionProgram]] = []
        for program in self._composition_programs(dialect_id):
            continuation = program.match(
                prompt,
                confirmed_output,
                limit=self.proposal_width,
            )
            if continuation:
                matches.append((continuation, program))
        continuations = {row[0] for row in matches}
        if len(continuations) != 1:
            return None
        continuation = continuations.pop()
        program = max(
            (row[1] for row in matches if row[0] == continuation),
            key=lambda row: (
                row.support,
                row.distinct_bindings,
                row.copied_tokens,
                row.context_width,
                tuple(-len(atom.canonical_key) for atom in row.atoms),
                row.canonical_key,
            ),
        )
        return (
            MarkovPhraseOption(
                token_ids=continuation,
                source=source,
                context_order=program.context_width,
                support=program.support,
                total=program.total,
                kind="composition",
            ),
            program,
        )

    def confirmed_episodes(self) -> tuple[tuple[int, ...], ...]:
        """Return immutable target-confirmed episodes for idempotent import."""

        if self._closed:
            raise MarkovDraftError("Markov draft provider is closed")
        return self._episodes()

    def confirmed_transitions(
        self,
    ) -> tuple[tuple[tuple[int, ...] | None, tuple[int, ...]], ...]:
        """Return retained prompt/output splits, including unknown legacy ones."""

        if self._closed:
            raise MarkovDraftError("Markov draft provider is closed")
        rows = []
        offset = 0
        for length, prompt_length in zip(
            self._state.episode_lengths,
            self._state.episode_prompt_lengths,
            strict=True,
        ):
            episode = self._state.token_ids[offset : offset + length]
            offset += length
            rows.append(
                (
                    None if prompt_length is None else episode[:prompt_length],
                    episode if prompt_length is None else episode[prompt_length:],
                )
            )
        return tuple(rows)

    @staticmethod
    def _import_digest(value: object) -> str:
        if not isinstance(value, str) or len(value) != 64 or set(value) - _HEX:
            raise ValueError("import digest must be a SHA-256 value")
        return value

    def imported_episode_sha256s(self) -> tuple[str, ...]:
        """Return durable receipt identities retained beyond token eviction."""

        if self._closed:
            raise MarkovDraftError("Markov draft provider is closed")
        return self._state.imported_episode_sha256s

    def register_imported_episode_sha256s(self, values: Sequence[str], /) -> int:
        """Atomically migrate legacy import identities into provider state."""

        if self._closed:
            raise MarkovDraftError("Markov draft provider is closed")
        if self._request_started or self._request_completed:
            raise MarkovDraftError("cannot migrate imports during a request")
        digests = tuple(self._import_digest(value) for value in values)
        merged = tuple(sorted(set(self._state.imported_episode_sha256s) | set(digests)))
        added = len(merged) - len(self._state.imported_episode_sha256s)
        if not added:
            return 0
        original = self._state
        try:
            self._state = replace(
                self._state,
                imported_episode_sha256s=merged,
            )
            self._persist()
        except Exception:
            self._state = original
            raise
        return added

    def import_confirmed_episode(
        self,
        history: tuple[int, ...],
        receipt_sha256: str,
        /,
    ) -> bool:
        """Learn one receipt exactly once in the same atomic state commit."""

        if self._closed:
            raise MarkovDraftError("Markov draft provider is closed")
        digest = self._import_digest(receipt_sha256)
        if digest in self._state.imported_episode_sha256s:
            return False
        if self._pending_import_digest is not None:
            raise MarkovDraftError("another receipt import is pending")
        self._pending_import_digest = digest
        try:
            self.observe_final(history)
        finally:
            self._pending_import_digest = None
        return True

    def import_confirmed_transition(
        self,
        prompt: tuple[int, ...],
        generated: tuple[int, ...],
        receipt_sha256: str,
        /,
    ) -> bool:
        """Atomically learn one receipt with its prompt/output boundary."""

        if self._closed:
            raise MarkovDraftError("Markov draft provider is closed")
        prompt_tokens = self._token_tuple(prompt, label="imported prompt")
        generated_tokens = self._token_tuple(
            generated,
            label="imported generation",
        )
        digest = self._import_digest(receipt_sha256)
        if digest in self._state.imported_episode_sha256s:
            return False
        if self._pending_import_digest is not None:
            raise MarkovDraftError("another receipt import is pending")
        self._pending_import_digest = digest
        try:
            self.begin_request(prompt_tokens)
            self.observe_final((*prompt_tokens, *generated_tokens))
        finally:
            self._pending_import_digest = None
        return True

    def _phrase_option_from(
        self,
        history: tuple[int, ...],
        *,
        source: str,
        dialect_id: str | None,
        minimum_support: int,
    ) -> MarkovPhraseOption | None:
        episodes = self._generation_episodes(dialect_id)
        candidates: list[MarkovPhraseOption] = []
        max_width = min(self.proposal_width, self.PHRASE_MAX_WIDTH)
        if max_width < 2:
            return None
        for order in range(min(self.PHRASE_MAX_CONTEXT, len(history)), 0, -1):
            suffix = history[-order:]
            continuations: list[tuple[int, ...]] = []
            for episode in episodes:
                for position in range(order, len(episode) - 1):
                    if episode[position - order : position] == suffix:
                        continuation = episode[position : position + max_width]
                        if len(continuation) >= 2:
                            continuations.append(continuation)
            if not continuations:
                continue
            for width in range(2, max_width + 1):
                phrases = Counter(
                    continuation[:width]
                    for continuation in continuations
                    if len(continuation) >= width
                )
                if not phrases:
                    break
                phrase, support = max(
                    phrases.items(),
                    key=lambda row: (
                        row[1],
                        tuple(-token for token in row[0]),
                    ),
                )
                if support >= minimum_support:
                    candidates.append(
                        MarkovPhraseOption(
                            token_ids=phrase,
                            source=source,
                            context_order=order,
                            support=support,
                            total=sum(phrases.values()),
                        )
                    )
        if not candidates:
            return None
        return max(candidates, key=self._phrase_option_score)

    def _request_phrase_option(
        self,
        history: tuple[int, ...],
    ) -> MarkovPhraseOption | None:
        prompt = self._request_prompt
        prompt_length = self._request_prompt_length
        max_width = min(self.proposal_width, self.PHRASE_MAX_WIDTH)
        if (
            prompt is None
            or prompt_length is None
            or history[:prompt_length] != prompt
            or max_width < 2
        ):
            return None
        generated = history[prompt_length:][-
            self.REQUEST_LOCAL_MAX_TOKENS :
        ]
        candidates = []
        maximum_context = min(self.PHRASE_MAX_CONTEXT, len(generated))
        for order in range(
            maximum_context,
            self.REQUEST_PHRASE_MIN_CONTEXT - 1,
            -1,
        ):
            suffix = generated[-order:]
            continuations = tuple(
                generated[position : position + max_width]
                for position in range(order, len(generated) - 1)
                if generated[position - order : position] == suffix
                and len(generated[position : position + max_width]) >= 2
            )
            if len(continuations) < self.REQUEST_PHRASE_MIN_SUPPORT:
                continue
            common_width = min(len(row) for row in continuations)
            for index in range(common_width):
                if len({row[index] for row in continuations}) != 1:
                    common_width = index
                    break
            common_width = min(common_width, max_width)
            if common_width < 2:
                continue
            candidates.append(
                MarkovPhraseOption(
                    token_ids=continuations[0][:common_width],
                    source="request",
                    context_order=order,
                    support=len(continuations),
                    total=len(continuations),
                )
            )
        if not candidates:
            return None
        return max(candidates, key=self._phrase_option_score)

    def _request_periodic_copy_value(
        self,
        generated: tuple[int, ...],
        *,
        period: int,
        offset: int,
        window_start: int = 0,
        total_length: int | None = None,
    ) -> int | None:
        absolute_length = (
            window_start + len(generated) if total_length is None else total_length
        )
        target_absolute = absolute_length + offset
        target_indexes = tuple(
            target_absolute - copy * period - window_start for copy in range(1, 4)
        )
        if any(index < 0 or index >= len(generated) for index in target_indexes):
            return None
        target_values = tuple(generated[index] for index in target_indexes)
        if len(set(target_values)) != len(target_values):
            return None
        target_block_start = target_absolute - target_absolute % period
        candidates = []
        for delta in range(-1, -period, -1):
            current_source_absolute = target_absolute + delta
            if not (
                target_block_start <= current_source_absolute < absolute_length
            ):
                continue
            current_source = current_source_absolute - window_start
            if not 0 <= current_source < len(generated):
                continue
            source_indexes = tuple(index + delta for index in target_indexes)
            if any(index < 0 or index >= len(generated) for index in source_indexes):
                continue
            if all(
                generated[target] == generated[source]
                for target, source in zip(
                    target_indexes,
                    source_indexes,
                    strict=True,
                )
            ):
                candidates.append((delta, generated[current_source]))
        if len(candidates) != 1:
            return None
        return candidates[0][1]

    def _request_periodic_option(
        self,
        history: tuple[int, ...],
    ) -> MarkovPhraseOption | None:
        prompt = self._request_prompt
        prompt_length = self._request_prompt_length
        max_width = min(self.proposal_width, self.PHRASE_MAX_WIDTH)
        if (
            prompt is None
            or prompt_length is None
            or history[:prompt_length] != prompt
            or max_width < 2
        ):
            return None
        total_length = len(history) - prompt_length
        window_start = max(0, total_length - self.REQUEST_LOCAL_MAX_TOKENS)
        generated = history[prompt_length + window_start :]
        maximum_period = min(
            self.REQUEST_PERIOD_MAX,
            len(generated) // 3,
        )
        candidates: list[tuple[float, int, int, MarkovPhraseOption]] = []
        for period in range(self.REQUEST_PERIOD_MIN, maximum_period + 1):
            current_block_start = total_length - total_length % period
            comparison_start_absolute = max(
                window_start + period,
                current_block_start - 2 * period,
            )
            comparison_start = comparison_start_absolute - window_start
            comparisons = tuple(
                generated[index] == generated[index - period]
                for index in range(comparison_start, len(generated))
            )
            if not comparisons:
                continue
            match_ratio = sum(comparisons) / len(comparisons)
            if (
                match_ratio < self.REQUEST_PERIOD_MATCH_THRESHOLD
                or match_ratio >= 1.0
            ):
                continue
            predicted = []
            copied_phases = 0
            for offset in range(min(max_width, period)):
                values = tuple(
                    generated[len(generated) + offset - copy * period]
                    for copy in range(1, 4)
                    if 0 <= len(generated) + offset - copy * period < len(generated)
                )
                if (
                    len(values) >= self.REQUEST_PERIOD_MIN_SUPPORT
                    and len(set(values)) == 1
                ):
                    predicted.append(values[0])
                    continue
                copied = self._request_periodic_copy_value(
                    generated,
                    period=period,
                    offset=offset,
                    window_start=window_start,
                    total_length=total_length,
                )
                if copied is None:
                    break
                if copied_phases >= 1:
                    break
                predicted.append(copied)
                copied_phases += 1
            if len(predicted) < 2:
                continue
            support = sum(comparisons)
            option = MarkovPhraseOption(
                token_ids=tuple(predicted),
                source="request",
                context_order=self.REQUEST_PHRASE_MIN_CONTEXT,
                support=support,
                total=len(comparisons),
                kind="binding" if copied_phases else "periodic",
            )
            candidates.append((match_ratio, period, copied_phases, option))
        if not candidates:
            return None
        best_ratio = max(row[0] for row in candidates)
        contenders = [row for row in candidates if row[0] == best_ratio]
        if len({row[3].token_ids for row in contenders}) > 1:
            return None
        return max(
            contenders,
            key=lambda row: (
                len(row[3].token_ids),
                row[3].support,
                -row[2],
                -row[1],
                tuple(-token for token in row[3].token_ids),
            ),
        )[3]

    def _phrase_option_score(
        self,
        option: MarkovPhraseOption,
    ) -> tuple[float, int, int, int, tuple[int, ...], str]:
        scope = self._active_dialect_similarity if option.source == "dialect" else 1.0
        quality = (
            option.confidence
            * math.log1p(option.support)
            * math.sqrt(len(option.token_ids))
            * (1.0 + option.context_order / self.PHRASE_MAX_CONTEXT)
            * scope
        )
        return (
            quality,
            option.context_order,
            option.support,
            len(option.token_ids),
            tuple(-token for token in option.token_ids),
            option.source,
        )

    def _crystal_dedupe_score(
        self,
        option: MarkovPhraseOption,
    ) -> tuple[object, ...]:
        layer = option.crystal_layer
        return (
            *self._phrase_option_score(option),
            int(layer is None),
            -(layer if layer is not None else -1),
            option.cell_sha256 or "",
        )

    def _dedupe_crystal_phrase_options(
        self,
        candidates: Sequence[MarkovPhraseOption],
    ) -> list[MarkovPhraseOption]:
        result: list[MarkovPhraseOption] = []
        crystal_positions: dict[tuple[int, ...], int] = {}
        for option in candidates:
            if option.kind != "crystal":
                result.append(option)
                continue
            position = crystal_positions.get(option.token_ids)
            if position is None:
                crystal_positions[option.token_ids] = len(result)
                result.append(option)
                continue
            if self._crystal_dedupe_score(option) > self._crystal_dedupe_score(
                result[position]
            ):
                result[position] = option
        return result

    def _phrase_option(self, history: tuple[int, ...]) -> MarkovPhraseOption | None:
        dialect = self._active_dialect
        candidates: list[MarkovPhraseOption] = []
        composition_rows: list[tuple[MarkovPhraseOption, MarkovCompositionProgram]] = []
        crystal_option = self._context_crystal_option(history)
        if crystal_option is not None:
            candidates.append(crystal_option)
        candidates.extend(self._layer_context_crystal_phrase_options)
        request_option = self._request_phrase_option(history)
        if request_option is not None:
            candidates.append(request_option)
        else:
            periodic_option = self._request_periodic_option(history)
            if periodic_option is not None:
                candidates.append(periodic_option)
        if dialect is not None and not self._active_dialect_is_new:
            local = self._phrase_option_from(
                history,
                source="dialect",
                dialect_id=dialect.dialect_id,
                minimum_support=self.DIALECT_PHRASE_MIN_SUPPORT,
            )
            if local is not None:
                candidates.append(local)
            local_composition = self._composition_option_from(
                history,
                source="dialect",
                dialect_id=dialect.dialect_id,
            )
            if local_composition is not None:
                composition_rows.append(local_composition)
        global_option = self._phrase_option_from(
            history,
            source="global",
            dialect_id=None,
            minimum_support=self.GLOBAL_PHRASE_MIN_SUPPORT,
        )
        if global_option is not None:
            candidates.append(global_option)
        global_composition = self._composition_option_from(
            history,
            source="global",
            dialect_id=None,
        )
        if global_composition is not None:
            composition_rows.append(global_composition)
        if self.atlas is not None:
            continuation = self.atlas.continuation(
                history,
                max_tokens=min(self.proposal_width, self.PHRASE_MAX_WIDTH),
                min_support=self.ATLAS_MIN_SUPPORT,
                min_confidence=self.ATLAS_MIN_CONFIDENCE,
            )
            if continuation is not None:
                candidates.append(
                    MarkovPhraseOption(
                        token_ids=continuation.token_ids,
                        source="atlas",
                        context_order=continuation.context_order,
                        support=continuation.support,
                        total=continuation.total,
                        kind="atlas",
                    )
                )
        composition_outputs = {row[0].token_ids for row in composition_rows}
        if len(composition_outputs) == 1:
            candidates.extend(row[0] for row in composition_rows)
        candidates = self._dedupe_crystal_phrase_options(candidates)
        if not candidates:
            self._pending_composition_program = None
            return None
        selected = max(candidates, key=self._phrase_option_score)
        self._pending_composition_program = next(
            (program for option, program in composition_rows if option == selected),
            None,
        )
        return selected

    def _expert_models(
        self, history: tuple[int, ...]
    ) -> tuple[tuple[_TransitionFingerprint, list[str]], ...]:
        if self._ricci_priority_degraded:
            self._ricci_working_symbols_cache.clear()
            self._ricci_episode_cache = None
            self._persistent_expert_models.clear()
            self._ricci_priority_degraded = False
        rows = []
        persistent = self._persistent_symbols()
        current = tuple(self._symbol(token) for token in history)
        for spec in self._experts:
            if spec.local_only or not persistent:
                selected = current[-spec.window :]
                model = _TransitionFingerprint.fit(
                    selected,
                    max_order=min(spec.max_order, len(selected) - 1),
                    alpha=self.alpha,
                    backoff_strength=self.backoff_strength,
                    min_count=self.min_count,
                )
                rows.append((model, list(selected)))
                continue
            model = self._persistent_expert_models.get(spec.name)
            if model is None:
                corpus = self._ricci_working_symbols(spec.window)
                model = _TransitionFingerprint.fit(
                    corpus,
                    max_order=min(spec.max_order, len(corpus) - 1),
                    alpha=self.alpha,
                    backoff_strength=self.backoff_strength,
                    min_count=self.min_count,
                )
                self._persistent_expert_models[spec.name] = model
            rows.append((model, list(current[-spec.window :])))
        return tuple(rows)

    def _request_local_fingerprint(
        self,
        history: tuple[int, ...],
    ) -> tuple[_TransitionFingerprint, tuple[str, ...]] | None:
        if history == self._request_local_cache_history:
            return self._request_local_cache
        prompt = self._request_prompt
        prompt_length = self._request_prompt_length
        if (
            prompt is None
            or prompt_length is None
            or history[:prompt_length] != prompt
        ):
            result = None
            self._request_local_cache_history = history
            self._request_local_cache = result
            return result
        generated = history[prompt_length:]
        if len(generated) <= self.REQUEST_LOCAL_MIN_ORDER:
            result = None
            self._request_local_cache_history = history
            self._request_local_cache = result
            return result
        symbols = tuple(
            self._symbol(token)
            for token in generated[-self.REQUEST_LOCAL_MAX_TOKENS :]
        )
        result = (
            _TransitionFingerprint.fit(
                symbols,
                max_order=min(self.max_order, len(symbols) - 1),
                alpha=self.alpha,
                backoff_strength=self.backoff_strength,
                min_count=self.min_count,
            ),
            symbols,
        )
        self._request_local_cache_history = history
        self._request_local_cache = result
        return result

    def _request_local_options(
        self,
        history: tuple[int, ...],
        *,
        limit: int,
    ) -> tuple[tuple[str, float, int, int, int], ...]:
        local = self._request_local_fingerprint(history)
        if local is None:
            return ()
        model, context = local
        return model.contextual_options(
            context,
            min_order=self.REQUEST_LOCAL_MIN_ORDER,
            min_support=self.REQUEST_LOCAL_MIN_SUPPORT,
            support_scale=self.REQUEST_LOCAL_SUPPORT_SCALE,
            limit=limit,
        )

    def _position_counts(
        self,
        position: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        observations = self._state.horizon_expert_observations[position]
        hits = self._state.horizon_expert_hits[position]
        request_observations = self._request_horizon_observations
        request_hits = self._request_horizon_hits
        if request_observations is None or request_hits is None:
            return observations, hits
        return (
            tuple(
                durable + local
                for durable, local in zip(
                    observations,
                    request_observations[position],
                    strict=True,
                )
            ),
            tuple(
                durable + local
                for durable, local in zip(
                    hits,
                    request_hits[position],
                    strict=True,
                )
            ),
        )

    def _plan_counts(self, position: int) -> tuple[int, int]:
        observations = self._state.horizon_plan_observations[position]
        hits = self._state.horizon_plan_hits[position]
        if self._request_plan_observations is None:
            return observations, hits
        assert self._request_plan_hits is not None
        return (
            observations + self._request_plan_observations[position],
            hits + self._request_plan_hits[position],
        )

    @staticmethod
    def _dialect_plan_counts(
        dialect: MarkovDialectState,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if dialect.plan_observations:
            return dialect.plan_observations, dialect.plan_hits
        zeros = (0,) * _MAX_PROPOSAL_POSITIONS
        return zeros, zeros

    def _planner_counts(self, planner: int, position: int) -> tuple[int, int]:
        observations = self._state.planner_observations[planner][position]
        hits = self._state.planner_hits[planner][position]
        if self._request_planner_observations is None:
            return observations, hits
        assert self._request_planner_hits is not None
        return (
            observations + self._request_planner_observations[planner][position],
            hits + self._request_planner_hits[planner][position],
        )

    @staticmethod
    def _dialect_planner_counts(
        dialect: MarkovDialectState,
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
        if dialect.planner_observations:
            return dialect.planner_observations, dialect.planner_hits
        zeros = tuple((0,) * _MAX_PROPOSAL_POSITIONS for _ in _PLANNER_NAMES)
        return zeros, zeros

    def _planner_reliability(self, planner: int, position: int) -> float:
        observations, hits = self._planner_counts(planner, position)
        if observations <= 0:
            learned = 1.0
        else:
            maturity = observations / (
                observations + self.POSITION_WEIGHT_SATURATION
            )
            learned = (1.0 - maturity) + maturity * (
                (hits + 1.0) / (observations + 2.0)
            )
        dialect_rows = []
        for similarity, neighbor_weight, dialect in self._inference_dialects():
            dialect_observations, dialect_hits = self._dialect_planner_counts(
                dialect
            )
            observed = dialect_observations[planner][position]
            if observed <= 0:
                continue
            maturity = observed / (observed + self.POSITION_WEIGHT_SATURATION)
            reliability = (1.0 - maturity) + maturity * (
                (dialect_hits[planner][position] + 1.0) / (observed + 2.0)
            )
            influence = neighbor_weight * similarity * maturity
            if influence > 0.0:
                dialect_rows.append((influence, reliability))
        if dialect_rows:
            influence = min(1.0, sum(row[0] for row in dialect_rows))
            local = sum(
                weight * reliability for weight, reliability in dialect_rows
            ) / sum(row[0] for row in dialect_rows)
            learned = (1.0 - influence) * learned + influence * local
        return max(0.0, min(1.0, learned))

    def _planner_has_evidence(self, planner: int, position: int) -> bool:
        if self._planner_counts(planner, position)[0] > 0:
            return True
        for _similarity, _weight, dialect in self._inference_dialects():
            observations, _hits = self._dialect_planner_counts(dialect)
            if observations[planner][position] > 0:
                return True
        return False

    def _planner_utility(
        self,
        planner: int,
        confidences: Sequence[float],
    ) -> float:
        prefix_reliability = 1.0
        utility = 0.0
        for position, confidence in enumerate(confidences):
            prefix_reliability = min(
                prefix_reliability,
                self._planner_reliability(planner, position)
                if self._planner_has_evidence(planner, position)
                else 0.5,
            )
            utility += self.LOOKAHEAD_DISCOUNT**position * math.log(
                max(1e-12, float(confidence) * prefix_reliability)
            )
        return utility

    @staticmethod
    def _provider_planner_index(provider: str) -> int:
        if not isinstance(provider, str) or provider not in {"markov", "mtp"}:
            raise MarkovDraftError("provider planner must be markov or mtp")
        return _PLANNER_NAMES.index(provider)

    def provider_policy_score(
        self,
        provider: str,
        position: int,
        /,
    ) -> tuple[float, bool]:
        if self._closed:
            raise MarkovDraftError("Markov draft provider is closed")
        if (
            isinstance(position, bool)
            or not isinstance(position, int)
            or not 0 <= position < _MAX_PROPOSAL_POSITIONS
        ):
            raise MarkovDraftError("provider policy position is invalid")
        planner = self._provider_planner_index(provider)
        observed = self._planner_has_evidence(planner, position)
        return (
            self._planner_reliability(planner, position) if observed else 0.5,
            observed,
        )

    def provider_prefix_probability(
        self,
        provider: str,
        width: int,
        /,
    ) -> tuple[float, bool]:
        """Return target-learned probability that one provider clears a prefix."""

        if self._closed:
            raise MarkovDraftError("Markov draft provider is closed")
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or not 1 <= width <= _MAX_PROPOSAL_POSITIONS
        ):
            raise MarkovDraftError("provider policy width is invalid")
        planner = self._provider_planner_index(provider)
        if any(
            not self._planner_has_evidence(planner, position)
            for position in range(width)
        ):
            return 0.0, False
        return (
            math.prod(
                self._planner_reliability(planner, position)
                for position in range(width)
            ),
            True,
        )

    def observe_provider_policy_feedback(
        self,
        provider: str,
        position: int,
        hit: bool,
        /,
    ) -> None:
        if self._closed or not self._request_started or self._request_completed:
            raise MarkovDraftError("Markov request is not active")
        planner = self._provider_planner_index(provider)
        self._update_request_planner(planner, position, hit)

    def snapshot_provider_policy_feedback(self, /) -> object:
        if self._closed or not self._request_started or self._request_completed:
            raise MarkovDraftError("Markov request is not active")
        return _ProviderPolicySnapshot(
            observations=(
                None
                if self._request_planner_observations is None
                else tuple(
                    tuple(row) for row in self._request_planner_observations
                )
            ),
            hits=(
                None
                if self._request_planner_hits is None
                else tuple(tuple(row) for row in self._request_planner_hits)
            ),
            feedback=tuple(self._planner_feedback),
        )

    def restore_provider_policy_feedback(self, snapshot: object, /) -> None:
        if self._closed or not self._request_started or self._request_completed:
            raise MarkovDraftError("Markov request is not active")
        if not isinstance(snapshot, _ProviderPolicySnapshot):
            raise MarkovDraftError("provider policy snapshot is invalid")
        self._request_planner_observations = (
            None
            if snapshot.observations is None
            else [list(row) for row in snapshot.observations]
        )
        self._request_planner_hits = (
            None if snapshot.hits is None else [list(row) for row in snapshot.hits]
        )
        self._planner_feedback = list(snapshot.feedback)

    def _position_weighting(
        self,
        position: int,
        base_weights: Sequence[float],
    ) -> tuple[tuple[float, ...], float, float]:
        """Blend global Rapidity with position and dialect Beta specialist skill."""

        base = tuple(float(value) for value in base_weights)
        if len(base) != len(self._experts) or not 0 <= position < (
            _MAX_PROPOSAL_POSITIONS
        ):
            raise MarkovDraftError("Markov position weighting shape is invalid")
        observations, hits = self._position_counts(position)
        maximum_observations = max(observations, default=0)
        position_maturity = (
            0.0
            if maximum_observations <= 0
            else maximum_observations
            / (maximum_observations + self.POSITION_WEIGHT_SATURATION)
        )
        position_posterior = tuple(
            (
                (hit + 1.0) / (observed + 2.0)
                if observed > 0
                else (global_hit + 1.0) / (global_observed + 2.0)
                if global_observed > 0
                else 0.5
            )
            for observed, hit, global_observed, global_hit in zip(
                observations,
                hits,
                self._state.expert_observations,
                self._state.expert_hits,
                strict=True,
            )
        )
        dialect_evidence = []
        for _similarity, neighbor_weight, dialect in self._inference_dialects():
            if not dialect.horizon_observations:
                continue
            dialect_position_observations = dialect.horizon_observations[position]
            dialect_position_hits = dialect.horizon_hits[position]
            dialect_observations = max(dialect_position_observations, default=0)
            if dialect_observations <= 0:
                continue
            evidence_weight = (
                neighbor_weight
                * dialect_observations
                / (dialect_observations + self.POSITION_WEIGHT_SATURATION)
            )
            if evidence_weight <= 0.0:
                continue
            dialect_posterior = tuple(
                (
                    (hit + 1.0) / (observed + 2.0)
                    if observed > 0
                    else position_value
                )
                for observed, hit, position_value in zip(
                    dialect_position_observations,
                    dialect_position_hits,
                    position_posterior,
                    strict=True,
                )
            )
            dialect_evidence.append((evidence_weight, dialect_posterior))
        dialect_evidence_total = sum(row[0] for row in dialect_evidence)
        dialect_maturity = min(1.0, dialect_evidence_total)
        if dialect_maturity <= 0.0:
            contextual_posterior = position_posterior
        else:
            dialect_posterior = tuple(
                sum(
                    evidence_weight * posterior[index]
                    for evidence_weight, posterior in dialect_evidence
                )
                / dialect_evidence_total
                for index in range(len(self._experts))
            )
            contextual_posterior = tuple(
                (1.0 - dialect_maturity) * position_value
                + dialect_maturity * dialect_value
                for position_value, dialect_value in zip(
                    position_posterior,
                    dialect_posterior,
                    strict=True,
                )
            )
        combined_maturity = 1.0 - (
            (1.0 - position_maturity) * (1.0 - dialect_maturity)
        )
        if combined_maturity <= 0.0:
            return base, 0.0, 0.0
        total = sum(contextual_posterior)
        if total <= 0.0 or not math.isfinite(total):
            raise MarkovDraftError("Markov position posterior is invalid")
        specialist = tuple(value / total for value in contextual_posterior)
        blended = tuple(
            (1.0 - combined_maturity) * global_weight
            + combined_maturity * local_weight
            for global_weight, local_weight in zip(base, specialist, strict=True)
        )
        count = len(blended)
        result = tuple(
            (1.0 - self.FIXED_SHARE) * value + self.FIXED_SHARE / count
            for value in blended
        )
        return result, position_maturity, dialect_maturity

    def _calibrated_confidence(
        self,
        token: int,
        raw_confidence: float,
        expert_row: Sequence[tuple[Mapping[str, float], int]],
        weights: Sequence[float],
        position: int,
        *,
        allow_empirical: bool = True,
    ) -> tuple[float, float]:
        """Fuse PPM mass with target-confirmed Fixed-Share expert evidence.

        Each agreeing expert contributes its Beta(1,1) posterior correctness,
        discounted by ``n / (n + tau)`` and by its current normalized Top-1
        margin ``(p1-p2)/(p1+p2)``.  An unobserved or undecided Council therefore
        cannot create confidence from its prior alone.  The weighted evidence
        is joined with the current PPM probability by the complement product
        ``1 - (1-p_raw)(1-e)``.  Target disagreement remains a separate prefix
        penalty in :class:`RollingDraftProposal`.  Forced phrase tokens disable
        this path because phrase support is already scored independently.
        """

        if len(expert_row) != len(self._experts) or len(weights) != len(self._experts):
            raise MarkovDraftError("Markov empirical confidence width changed")
        if not 0 <= position < _MAX_PROPOSAL_POSITIONS:
            raise MarkovDraftError("Markov confidence position is invalid")
        evidence = 0.0
        token_symbol = self._symbol(token)
        position_observations, position_hits = self._position_counts(position)
        for index, ((_distribution, predicted), weight) in enumerate(
            zip(expert_row, weights, strict=True)
        ):
            if not allow_empirical or predicted != token:
                continue
            token_mass = float(_distribution.get(token_symbol, 0.0))
            runner_up = max(
                (
                    float(probability)
                    for symbol, probability in _distribution.items()
                    if symbol != token_symbol
                    and (
                        symbol == _UNKNOWN_TOKEN
                        or (
                            symbol.isdecimal()
                            and 0 <= int(symbol) < self.vocab_size
                        )
                    )
                ),
                default=0.0,
            )
            denominator = token_mass + runner_up
            decisiveness = (
                1.0
                if denominator <= 0.0 and token_mass > 0.0
                else 0.0
                if denominator <= 0.0
                else max(0.0, min(1.0, (token_mass - runner_up) / denominator))
            )
            if decisiveness <= 0.0:
                continue
            observations = position_observations[index]
            hits = position_hits[index]
            if observations <= 0:
                observations = self._state.expert_observations[index]
                hits = self._state.expert_hits[index]
            if observations <= 0:
                continue
            posterior = (hits + 1.0) / (observations + 2.0)
            maturity = observations / (
                observations + self.EMPIRICAL_EVIDENCE_SATURATION
            )
            evidence += float(weight) * maturity * posterior * decisiveness
        evidence = max(0.0, min(0.999, evidence))
        raw = max(0.0, min(0.999, float(raw_confidence)))
        calibrated = 1.0 - (1.0 - raw) * (1.0 - evidence)
        return max(raw, min(0.999, calibrated)), evidence

    def _lookahead_counts(self, position: int) -> tuple[int, int, int]:
        observations = self._state.lookahead_observations[position]
        hits = self._state.lookahead_hits[position]
        greedy_hits = self._state.lookahead_greedy_hits[position]
        if self._request_lookahead_observations is None:
            return observations, hits, greedy_hits
        assert self._request_lookahead_hits is not None
        assert self._request_lookahead_greedy_hits is not None
        return (
            observations + self._request_lookahead_observations[position],
            hits + self._request_lookahead_hits[position],
            greedy_hits + self._request_lookahead_greedy_hits[position],
        )

    def _lookahead_choice(
        self,
        experts: Sequence[tuple[_TransitionFingerprint, list[str]]],
        mixture: Mapping[str, float],
        numeric_symbols: Sequence[str],
        base_weights: Sequence[float],
        position: int,
    ) -> tuple[int, float, int]:
        """Choose one token by discounted current+next Markov log probability."""

        greedy = max(
            (probability, -int(symbol), int(symbol))
            for symbol, probability in mixture.items()
        )[2]
        if (
            position + 1 >= _MAX_PROPOSAL_POSITIONS
            or len(numeric_symbols) < 2
        ):
            return greedy, 0.0, 0
        candidates = sorted(
            numeric_symbols,
            key=lambda symbol: (mixture.get(symbol, 0.0), -int(symbol)),
            reverse=True,
        )[: self.LOOKAHEAD_CANDIDATES]
        next_weights = self._position_weighting(position + 1, base_weights)[0]
        rows = []
        for symbol in candidates:
            next_distributions = tuple(
                model.distribution((*context, symbol)) for model, context in experts
            )
            next_symbols = {
                candidate
                for distribution in next_distributions
                for candidate in distribution
                if candidate != _UNKNOWN_TOKEN
                and candidate.isdecimal()
                and 0 <= int(candidate) < self.vocab_size
            }
            next_probability = max(
                (
                    sum(
                        weight * distribution.get(candidate, 0.0)
                        for weight, distribution in zip(
                            next_weights,
                            next_distributions,
                            strict=True,
                        )
                    )
                    for candidate in next_symbols
                ),
                default=1e-12,
            )
            current_probability = max(1e-12, float(mixture.get(symbol, 0.0)))
            score = math.log(current_probability) + self.LOOKAHEAD_DISCOUNT * math.log(
                max(1e-12, next_probability)
            )
            rows.append((score, current_probability, -int(symbol), int(symbol)))
        best = max(rows)
        greedy_row = next(row for row in rows if row[3] == greedy)
        gain = max(0.0, best[0] - greedy_row[0])
        if best[3] != greedy:
            observations, planned_hits, greedy_hits = self._lookahead_counts(position)
            empirical_advantage = 0.0
            if observations > 0:
                maturity = observations / (
                    observations + self.EMPIRICAL_EVIDENCE_SATURATION
                )
                planned = (planned_hits + 1.0) / (observations + 2.0)
                baseline = (greedy_hits + 1.0) / (observations + 2.0)
                empirical_advantage = maturity * (
                    math.log(planned / max(1e-12, 1.0 - planned))
                    - math.log(baseline / max(1e-12, 1.0 - baseline))
                )
            adjusted_gain = gain + (
                self.LOOKAHEAD_EMPIRICAL_STRENGTH * empirical_advantage
            )
            if adjusted_gain < self.LOOKAHEAD_MIN_LOG_GAIN:
                return greedy, 0.0, len(candidates)
        return best[3], gain, len(candidates)

    def _beam_width(
        self,
        probabilities: Sequence[float],
        disagreement: float,
        maturity: float,
    ) -> int:
        ordered = sorted((float(value) for value in probabilities), reverse=True)
        if len(ordered) < 2:
            return 1
        denominator = ordered[0] + ordered[1]
        margin = 0.0 if denominator <= 0.0 else (ordered[0] - ordered[1]) / denominator
        uncertainty = max(0.0, min(1.0, 1.0 - margin + 0.5 * disagreement))
        adaptive = 1 + math.ceil(
            (self.LOOKAHEAD_CANDIDATES * 2 - 1)
            * uncertainty
            * (1.0 - 0.5 * max(0.0, min(1.0, maturity)))
        )
        return max(1, min(self.LOOKAHEAD_CANDIDATES * 2, adaptive))

    def _beam_position_reliability(self, position: int) -> float:
        verified = self._beam_position_verified[position]
        direct = (
            1.0
            if verified <= 0
            else (self._beam_position_hits[position] + 0.5) / (verified + 2.0)
        )
        observations, hits = self._plan_counts(position)
        if observations <= 0:
            learned = 1.0
        else:
            posterior = (hits + 1.0) / (observations + 2.0)
            maturity = observations / (
                observations + self.POSITION_WEIGHT_SATURATION
            )
            learned = (1.0 - maturity) + maturity * posterior
        dialect_rows = []
        for similarity, neighbor_weight, dialect in self._inference_dialects():
            dialect_observation_rows, dialect_hit_rows = (
                self._dialect_plan_counts(dialect)
            )
            dialect_observations = dialect_observation_rows[position]
            if dialect_observations <= 0:
                continue
            dialect_hits = dialect_hit_rows[position]
            dialect_maturity = dialect_observations / (
                dialect_observations + self.POSITION_WEIGHT_SATURATION
            )
            dialect_posterior = (dialect_hits + 1.0) / (
                dialect_observations + 2.0
            )
            dialect_reliability = (
                (1.0 - dialect_maturity)
                + dialect_maturity * dialect_posterior
            )
            influence = neighbor_weight * similarity * dialect_maturity
            if influence > 0.0:
                dialect_rows.append((influence, dialect_reliability))
        if dialect_rows:
            influence = min(1.0, sum(row[0] for row in dialect_rows))
            dialect_reliability = sum(
                weight * reliability for weight, reliability in dialect_rows
            ) / sum(row[0] for row in dialect_rows)
            learned = (
                (1.0 - influence) * learned
                + influence * dialect_reliability
            )
        return max(0.0, min(1.0, min(direct, learned)))

    @staticmethod
    def _distribution_disagreement(
        distributions: Sequence[Mapping[str, float]],
        weights: Sequence[float],
    ) -> float:
        universe = set().union(*(distribution.keys() for distribution in distributions))
        pooled = {
            symbol: sum(
                weight * distribution.get(symbol, 0.0)
                for weight, distribution in zip(weights, distributions, strict=True)
            )
            for symbol in universe
        }

        def entropy(probabilities: Mapping[str, float]) -> float:
            return -sum(
                value * math.log(max(value, 1e-12))
                for value in probabilities.values()
                if value > 0.0
            )

        return max(
            0.0,
            entropy(pooled)
            - sum(
                weight * entropy(distribution)
                for weight, distribution in zip(weights, distributions, strict=True)
            ),
        )

    def _predict_beam(
        self,
        history: tuple[int, ...],
        count: int,
        *,
        position_offset: int = 0,
    ) -> tuple[
        tuple[int, ...],
        tuple[tuple[tuple[dict[str, float], int], ...], ...],
        tuple[float, ...],
        tuple[float, ...],
    ] | None:
        """Compose Atlas and same-request transitions with the live Council."""

        if (
            isinstance(position_offset, bool)
            or not isinstance(position_offset, int)
            or position_offset < 0
            or position_offset + count > _MAX_PROPOSAL_POSITIONS
        ):
            raise ValueError("beam position range is invalid")
        experts = self._expert_models(history)
        request_local = self._request_local_fingerprint(history)
        base_weights = self._weights()
        beam = (_BeamPath(score=0.0, tokens=(), steps=()),)
        for position in range(count):
            horizon_position = position_offset + position
            expansions: list[_BeamPath] = []
            retained_width = 1
            for path in beam:
                weights, position_maturity, dialect_maturity = (
                    self._position_weighting(horizon_position, base_weights)
                )
                symbols = tuple(self._symbol(token) for token in path.tokens)
                distributions = tuple(
                    model.distribution((*context, *symbols))
                    for model, context in experts
                )
                numeric_symbols = {
                    symbol
                    for distribution in distributions
                    for symbol in distribution
                    if symbol != _UNKNOWN_TOKEN
                    and symbol.isdecimal()
                    and 0 <= int(symbol) < self.vocab_size
                }
                mixture = {
                    symbol: sum(
                        weight * distribution.get(symbol, 0.0)
                        for weight, distribution in zip(
                            weights, distributions, strict=True
                        )
                    )
                    for symbol in numeric_symbols
                }
                online = sorted(
                    numeric_symbols,
                    key=lambda symbol: (mixture[symbol], -int(symbol)),
                    reverse=True,
                )[: self.LOOKAHEAD_CANDIDATES * 2]
                atlas_rows = (
                    ()
                    if self.atlas is None
                    else self.atlas.token_options(
                        (*history, *path.tokens),
                        limit=min(
                            self.atlas.max_branches,
                            self.LOOKAHEAD_CANDIDATES * 2,
                        ),
                    )
                )
                atlas_by_token = {row.token_id: row for row in atlas_rows}
                request_local_rows = (
                    ()
                    if request_local is None
                    else request_local[0].contextual_options(
                        (*request_local[1], *symbols),
                        min_order=self.REQUEST_LOCAL_MIN_ORDER,
                        min_support=self.REQUEST_LOCAL_MIN_SUPPORT,
                        support_scale=self.REQUEST_LOCAL_SUPPORT_SCALE,
                        limit=self.LOOKAHEAD_CANDIDATES * 2,
                    )
                )
                request_local_by_token = {
                    int(row[0]): row for row in request_local_rows
                }
                candidates = tuple(
                    sorted(
                        {
                            *(int(symbol) for symbol in online),
                            *atlas_by_token,
                            *request_local_by_token,
                        },
                    )
                )
                if not candidates:
                    continue
                greedy = max(
                    (
                        mixture.get(self._symbol(token), 0.0),
                        -token,
                        token,
                    )
                    for token in candidates
                )[2]
                disagreement = self._distribution_disagreement(
                    distributions,
                    weights,
                )
                expert_row = []
                for distribution in distributions:
                    predicted = max(
                        (
                            probability,
                            -int(symbol),
                            int(symbol),
                        )
                        for symbol, probability in distribution.items()
                        if symbol != _UNKNOWN_TOKEN
                        and symbol.isdecimal()
                        and 0 <= int(symbol) < self.vocab_size
                    )[2]
                    expert_row.append((dict(distribution), predicted))
                candidate_rows = []
                for token in candidates:
                    raw = mixture.get(self._symbol(token), 0.0)
                    calibrated, empirical = self._calibrated_confidence(
                        token,
                        raw,
                        expert_row,
                        weights,
                        horizon_position,
                    )
                    atlas = atlas_by_token.get(token)
                    atlas_confidence = (
                        0.0
                        if atlas is None
                        else atlas.probability
                        * (1.0 - math.exp(-float(atlas.support) / 1.5))
                    )
                    local = request_local_by_token.get(token)
                    local_confidence = 0.0 if local is None else local[1]
                    fused = min(
                        0.999,
                        1.0
                        - (1.0 - calibrated)
                        * (1.0 - atlas_confidence)
                        * (1.0 - local_confidence),
                    )
                    candidate_rows.append((token, fused, raw, empirical))
                retained_width = max(
                    retained_width,
                    self._beam_width(
                        [row[1] for row in candidate_rows],
                        disagreement,
                        max(position_maturity, dialect_maturity),
                    ),
                )
                greedy_confidence = next(
                    row[1] for row in candidate_rows if row[0] == greedy
                )
                for token, confidence, raw, empirical in candidate_rows:
                    conflict = 0.0
                    if token != greedy:
                        denominator = confidence + greedy_confidence
                        conflict = (
                            0.0
                            if denominator <= 0.0
                            else max(
                                0.0,
                                min(
                                    1.0,
                                    (greedy_confidence - confidence) / denominator,
                                ),
                            )
                        )
                    token_disagreement = disagreement + conflict
                    gain = max(
                        0.0,
                        math.log(max(confidence, 1e-12))
                        - math.log(max(greedy_confidence, 1e-12)),
                    )
                    step = _BeamStep(
                        token=token,
                        greedy=greedy,
                        gain=gain,
                        confidence=confidence,
                        raw_confidence=raw,
                        empirical_evidence=empirical,
                        disagreement=token_disagreement,
                        weights=tuple(weights),
                        position_maturity=position_maturity,
                        dialect_maturity=dialect_maturity,
                        evaluated_candidates=len(candidate_rows),
                    )
                    expansions.append(
                        _BeamPath(
                            score=path.score
                            + self.LOOKAHEAD_DISCOUNT**position
                            * math.log(max(confidence, 1e-12)),
                            tokens=(*path.tokens, token),
                            steps=(*path.steps, step),
                        )
                    )
            if not expansions:
                return None
            beam = tuple(
                sorted(
                    expansions,
                    key=lambda row: (-row.score, row.tokens),
                )[:retained_width]
            )

        winner = beam[0]
        if len(winner.steps) != count:
            return None
        path_logs = tuple(
            sum(math.log(max(step.confidence, 1e-12)) for step in path.steps)
            for path in beam
        )

        def logsumexp(indexes: Sequence[int]) -> float:
            maximum = max(path_logs[index] for index in indexes)
            return maximum + math.log(
                sum(math.exp(path_logs[index] - maximum) for index in indexes)
            )

        posteriors = []
        eligible = tuple(range(len(beam)))
        for position, token in enumerate(winner.tokens):
            matching = tuple(
                index
                for index in eligible
                if beam[index].tokens[position] == token
            )
            posterior = math.exp(logsumexp(matching) - logsumexp(eligible))
            posteriors.append(max(0.0, min(1.0, posterior)))
            eligible = matching
        self._last_beam_prefix_posteriors = tuple(posteriors)
        self._last_beam_path_count = len(beam)
        feedback_rows = tuple(
            self._teacher_forced_prediction(
                (*history, *winner.tokens[:position]),
                position=position_offset + position,
                planner="council",
            )[0]
            for position in range(count)
        )
        for position, step in enumerate(winner.steps):
            self._last_raw_confidence = step.raw_confidence
            self._last_empirical_evidence = step.empirical_evidence
            self._last_confidence = step.confidence
            self._last_disagreement = step.disagreement
            self._last_position = position
            self._last_position_maturity = step.position_maturity
            self._last_dialect_skill_maturity = step.dialect_maturity
            self._last_position_weights = step.weights
            self._lookahead_calls += int(step.evaluated_candidates > 1)
            self._lookahead_candidates += step.evaluated_candidates
            self._lookahead_token_changes += int(step.token != step.greedy)
            self._max_lookahead_gain = max(self._max_lookahead_gain, step.gain)
            if step.position_maturity > 0.0:
                self._position_specialist_predictions += 1
                self._max_position_maturity = max(
                    self._max_position_maturity,
                    step.position_maturity,
                )
            if step.dialect_maturity > 0.0:
                self._dialect_specialist_predictions += 1
                self._max_dialect_skill_maturity = max(
                    self._max_dialect_skill_maturity,
                    step.dialect_maturity,
                )
        self._last_lookahead_gain = winner.steps[-1].gain
        self._last_plan_trace = tuple(
            (step.token, step.greedy, step.gain) for step in winner.steps
        )
        self._predictions += count
        self._council_predictions += count
        prefix_reliabilities = []
        prefix_reliability = 1.0
        for position in range(count):
            prefix_reliability = min(
                prefix_reliability,
                self._beam_position_reliability(position_offset + position),
            )
            prefix_reliabilities.append(prefix_reliability)
        return (
            winner.tokens,
            feedback_rows,
            tuple(
                step.confidence * reliability
                for step, reliability in zip(
                    winner.steps,
                    prefix_reliabilities,
                    strict=True,
                )
            ),
            tuple(step.disagreement for step in winner.steps),
        )

    def _predict_council(
        self,
        history: tuple[int, ...],
        count: int,
        *,
        forced_prefix: Sequence[int] = (),
        position_offset: int = 0,
    ) -> tuple[
        tuple[int, ...],
        tuple[tuple[tuple[dict[str, float], int], ...], ...],
        tuple[float, ...],
        tuple[float, ...],
    ]:
        experts = self._expert_models(history)
        base_weights = self._weights()
        proposal: list[int] = []
        feedback_rows = []
        confidences = []
        disagreements = []
        plan_trace = []
        prefix_reliability = 1.0
        forced = tuple(forced_prefix)
        if (
            isinstance(position_offset, bool)
            or not isinstance(position_offset, int)
            or position_offset < 0
            or position_offset + count > _MAX_PROPOSAL_POSITIONS
        ):
            raise ValueError("Council position range is invalid")
        if len(forced) > count or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or not 0 <= token < self.vocab_size
            for token in forced
        ):
            raise ValueError("forced Council prefix is invalid")
        for position in range(count):
            horizon_position = position_offset + position
            weights, position_maturity, dialect_skill_maturity = (
                self._position_weighting(
                    horizon_position,
                    base_weights,
                )
            )
            distributions = tuple(
                model.distribution(context) for model, context in experts
            )
            numeric_symbols = sorted(
                {
                    symbol
                    for distribution in distributions
                    for symbol in distribution
                    if symbol != _UNKNOWN_TOKEN
                    and symbol.isdecimal()
                    and 0 <= int(symbol) < self.vocab_size
                }
            )
            if not numeric_symbols:
                raise MarkovDraftError("Markov council has no Qwen token")
            mixture = {
                symbol: sum(
                    weight * distribution.get(symbol, 0.0)
                    for weight, distribution in zip(weights, distributions, strict=True)
                )
                for symbol in numeric_symbols
            }
            if position < len(forced):
                token = forced[position]
                greedy = token
                lookahead_gain = 0.0
                evaluated_candidates = 0
            else:
                greedy = max(
                    (probability, -int(symbol), int(symbol))
                    for symbol, probability in mixture.items()
                )[2]
                token, lookahead_gain, evaluated_candidates = (
                    self._lookahead_choice(
                        experts,
                        mixture,
                        numeric_symbols,
                        base_weights,
                        horizon_position,
                    )
                )
                if evaluated_candidates:
                    self._lookahead_calls += 1
                    self._lookahead_candidates += evaluated_candidates
                    self._lookahead_token_changes += int(token != greedy)
                    self._max_lookahead_gain = max(
                        self._max_lookahead_gain,
                        lookahead_gain,
                    )
            self._last_lookahead_gain = lookahead_gain
            plan_trace.append((token, greedy, lookahead_gain))
            proposal.append(token)
            universe = set().union(
                *(distribution.keys() for distribution in distributions)
            )
            pooled = {
                symbol: sum(
                    weight * distribution.get(symbol, 0.0)
                    for weight, distribution in zip(weights, distributions, strict=True)
                )
                for symbol in universe
            }

            def entropy(probabilities: Mapping[str, float]) -> float:
                return -sum(
                    value * math.log(max(value, 1e-12))
                    for value in probabilities.values()
                    if value > 0.0
                )

            self._last_disagreement = max(
                0.0,
                entropy(pooled)
                - sum(
                    weight * entropy(distribution)
                    for weight, distribution in zip(weights, distributions, strict=True)
                ),
            )
            expert_row = []
            for distribution in distributions:
                predicted = max(
                    (
                        probability,
                        -int(symbol),
                        int(symbol),
                    )
                    for symbol, probability in distribution.items()
                    if symbol != _UNKNOWN_TOKEN
                    and symbol.isdecimal()
                    and 0 <= int(symbol) < self.vocab_size
                )[2]
                expert_row.append((distribution, predicted))
            self._last_raw_confidence = mixture.get(self._symbol(token), 0.0)
            (
                self._last_confidence,
                self._last_empirical_evidence,
            ) = self._calibrated_confidence(
                token,
                self._last_raw_confidence,
                expert_row,
                weights,
                horizon_position,
                allow_empirical=position >= len(forced),
            )
            prefix_reliability = min(
                prefix_reliability,
                self._beam_position_reliability(horizon_position),
            )
            self._last_confidence *= prefix_reliability
            self._last_position = horizon_position
            self._last_position_maturity = position_maturity
            self._last_dialect_skill_maturity = dialect_skill_maturity
            self._last_position_weights = weights
            if position_maturity > 0.0:
                self._position_specialist_predictions += 1
                self._max_position_maturity = max(
                    self._max_position_maturity,
                    position_maturity,
                )
            if dialect_skill_maturity > 0.0:
                self._dialect_specialist_predictions += 1
                self._max_dialect_skill_maturity = max(
                    self._max_dialect_skill_maturity,
                    dialect_skill_maturity,
                )
            confidences.append(self._last_confidence)
            disagreements.append(self._last_disagreement)
            feedback_rows.append(tuple(expert_row))
            symbol = self._symbol(token)
            for _model, context in experts:
                context.append(symbol)
        self._predictions += count
        self._council_predictions += count
        self._last_plan_trace = tuple(plan_trace)
        return (
            tuple(proposal),
            tuple(feedback_rows),
            tuple(confidences),
            tuple(disagreements),
        )

    def _planning_snapshot(self) -> tuple[object, ...]:
        return tuple(getattr(self, name) for name in _PLANNING_DIAGNOSTIC_FIELDS)

    def _restore_planning_snapshot(self, snapshot: tuple[object, ...]) -> None:
        if len(snapshot) != len(_PLANNING_DIAGNOSTIC_FIELDS):
            raise MarkovDraftError("planning snapshot width changed")
        for name, value in zip(
            _PLANNING_DIAGNOSTIC_FIELDS,
            snapshot,
            strict=True,
        ):
            setattr(self, name, value)

    def _capture_planning_call(
        self,
        callback: Callable[[], object],
    ) -> tuple[object, tuple[tuple[int, int, float], ...], tuple[object, ...]]:
        before = self._planning_snapshot()
        try:
            result = callback()
            trace = self._last_plan_trace
            after = self._planning_snapshot()
        finally:
            self._restore_planning_snapshot(before)
        return result, trace, after

    def _teacher_forced_prediction(
        self,
        history: tuple[int, ...],
        *,
        position: int,
        planner: str | None = None,
    ) -> tuple[
        tuple[tuple[dict[str, float], int], ...],
        int,
        tuple[int, int, float],
    ]:
        """Predict one retrospective row without changing served-draft metrics."""

        snapshot = self._planning_snapshot()
        try:
            selected = self._pending_planner if planner is None else planner
            if selected not in {None, *_PLANNER_NAMES}:
                raise MarkovDraftError("teacher-forced planner is invalid")
            predicted = (
                self._predict_beam(
                    history,
                    1,
                    position_offset=position,
                )
                if selected == "beam"
                else None
            )
            if predicted is None:
                tokens, feedback, _confidence, _disagreement = (
                    self._predict_council(
                        history,
                        1,
                        position_offset=position,
                    )
                )
            else:
                tokens, feedback, _confidence, _disagreement = predicted
            return feedback[0], tokens[0], self._last_plan_trace[0]
        finally:
            self._restore_planning_snapshot(snapshot)

    def _feedback_probabilities(
        self,
        feedback: tuple[tuple[dict[str, float], int], ...],
        token: int,
    ) -> tuple[float, ...]:
        if len(feedback) != len(self._experts):
            raise MarkovDraftError("Markov council feedback width changed")
        symbol = self._symbol(token)
        probabilities = []
        for distribution, _prediction in feedback:
            if symbol in distribution:
                probabilities.append(distribution[symbol])
                continue
            seen = sum(
                candidate.isdecimal() and 0 <= int(candidate) < self.vocab_size
                for candidate in distribution
            )
            probabilities.append(
                distribution.get(_UNKNOWN_TOKEN, 1e-12)
                / max(1, self.vocab_size - seen)
            )
        return tuple(probabilities)

    def _update_request_weights(
        self,
        feedback: tuple[tuple[dict[str, float], int], ...],
        token: int,
        position: int,
    ) -> None:
        if not 0 <= position < _MAX_PROPOSAL_POSITIONS:
            raise MarkovDraftError("Markov feedback position is invalid")
        probabilities = self._feedback_probabilities(feedback, token)
        before = self._weights()
        weights = self._position_weighting(position, before)[0]
        mixture_probability = sum(
            weight * probability
            for weight, probability in zip(weights, probabilities, strict=True)
        )
        surprise = -math.log(max(mixture_probability, 1e-12))
        previous_mean = (
            self._state.surprise_mean
            if self._request_surprise_mean is None
            else self._request_surprise_mean
        )
        previous_deviation = (
            self._state.surprise_deviation
            if self._request_surprise_deviation is None
            else self._request_surprise_deviation
        )
        previous_cusum = (
            self._state.surprise_cusum
            if self._request_surprise_cusum is None
            else self._request_surprise_cusum
        )
        total_feedback = self._state.feedback_count + self._request_feedback_count
        z_score = (
            0.0
            if total_feedback == 0
            else (surprise - previous_mean) / max(previous_deviation, 1e-6)
        )
        next_mean = (
            (1.0 - self.SURPRISE_RATE) * previous_mean
            + self.SURPRISE_RATE * surprise
        )
        next_deviation = (
            (1.0 - self.SURPRISE_RATE) * previous_deviation
            + self.SURPRISE_RATE * abs(surprise - previous_mean)
        )
        next_cusum = max(
            0.0,
            self.CUSUM_DECAY * previous_cusum + z_score - self.CUSUM_DRIFT,
        )
        logs = list(
            self._combined_rapidities()
            if self._request_expert_rapidities is None
            else self._request_expert_rapidities
        )
        for index, probability in enumerate(probabilities):
            advantage = math.log(max(probability, 1e-12)) - math.log(
                max(mixture_probability, 1e-12)
            )
            logs[index] = (
                self.RAPIDITY_DECAY * logs[index]
                + self.EXPERT_LEARNING_RATE * advantage
            )
        center = sum(logs) / len(logs)
        logs = [value - center for value in logs]
        regime_change = (
            total_feedback + 1 >= self.REGIME_WARMUP
            and next_cusum > self.CUSUM_THRESHOLD
        )
        if regime_change:
            logs = [self.REGIME_RAPIDITY_SHRINK * value for value in logs]
            next_cusum = 0.0
            self._request_regime_changes += 1
        self._request_expert_rapidities = tuple(logs)
        self._request_surprise_mean = next_mean
        self._request_surprise_deviation = next_deviation
        self._request_surprise_cusum = next_cusum
        self._request_feedback_count += 1
        after = self._weights()
        shift = 0.5 * sum(
            abs(left - right) for left, right in zip(before, after, strict=True)
        )
        self._request_weight_updates += 1
        self._max_request_weight_shift = max(
            self._max_request_weight_shift,
            shift,
        )

    def _record_confirmed_feedback(
        self,
        feedback: tuple[tuple[dict[str, float], int], ...],
        token: int,
        position: int,
        planned_token: int,
        greedy_token: int,
        lookahead_gain: float,
    ) -> None:
        self._update_request_weights(feedback, token, position)
        self._update_request_position_skill(feedback, token, position)
        self._update_request_plan(planned_token, token, position)
        self._update_request_lookahead(
            planned_token,
            greedy_token,
            token,
            position,
        )
        self._episode_feedback.append(
            (
                feedback,
                token,
                position,
                planned_token,
                greedy_token,
                lookahead_gain,
            )
        )

    def _update_request_position_skill(
        self,
        feedback: tuple[tuple[dict[str, float], int], ...],
        token: int,
        position: int,
    ) -> None:
        if len(feedback) != len(self._experts):
            raise MarkovDraftError("Markov council feedback width changed")
        self._update_request_position_predictions(
            tuple(prediction for _distribution, prediction in feedback),
            token,
            position,
        )

    def _update_request_position_predictions(
        self,
        predictions: tuple[int, ...],
        token: int,
        position: int,
    ) -> None:
        if len(predictions) != len(self._experts):
            raise MarkovDraftError("Markov council prediction width changed")
        if not 0 <= position < _MAX_PROPOSAL_POSITIONS:
            raise MarkovDraftError("Markov feedback position is invalid")
        if self._request_horizon_observations is None:
            self._request_horizon_observations = [
                [0] * len(self._experts) for _ in range(_MAX_PROPOSAL_POSITIONS)
            ]
            self._request_horizon_hits = [
                [0] * len(self._experts) for _ in range(_MAX_PROPOSAL_POSITIONS)
            ]
        assert self._request_horizon_hits is not None
        for index, prediction in enumerate(predictions):
            self._request_horizon_observations[position][index] += 1
            self._request_horizon_hits[position][index] += int(prediction == token)
        self._request_position_updates += 1
        maturity = self._position_weighting(position, self._weights())[1]
        self._max_request_position_maturity = max(
            self._max_request_position_maturity,
            maturity,
        )

    def _update_request_plan(
        self,
        planned_token: int,
        token: int,
        position: int,
    ) -> None:
        if not 0 <= position < _MAX_PROPOSAL_POSITIONS:
            raise MarkovDraftError("Markov plan position is invalid")
        if self._request_plan_observations is None:
            self._request_plan_observations = [0] * _MAX_PROPOSAL_POSITIONS
            self._request_plan_hits = [0] * _MAX_PROPOSAL_POSITIONS
        assert self._request_plan_hits is not None
        self._request_plan_observations[position] += 1
        self._request_plan_hits[position] += int(planned_token == token)

    def _update_request_planner(
        self,
        planner: int,
        position: int,
        hit: bool,
    ) -> None:
        if (
            isinstance(planner, bool)
            or not 0 <= planner < len(_PLANNER_NAMES)
            or isinstance(position, bool)
            or not isinstance(position, int)
            or not 0 <= position < _MAX_PROPOSAL_POSITIONS
            or not isinstance(hit, bool)
        ):
            raise MarkovDraftError("Markov planner feedback is invalid")
        if self._request_planner_observations is None:
            self._request_planner_observations = [
                [0] * _MAX_PROPOSAL_POSITIONS for _ in _PLANNER_NAMES
            ]
            self._request_planner_hits = [
                [0] * _MAX_PROPOSAL_POSITIONS for _ in _PLANNER_NAMES
            ]
        assert self._request_planner_hits is not None
        self._request_planner_observations[planner][position] += 1
        self._request_planner_hits[planner][position] += int(hit)
        self._planner_feedback.append((planner, position, hit))

    def _update_request_lookahead(
        self,
        planned_token: int,
        greedy_token: int,
        token: int,
        position: int,
    ) -> None:
        if planned_token == greedy_token:
            return
        if not 0 <= position < _MAX_PROPOSAL_POSITIONS:
            raise MarkovDraftError("Markov feedback position is invalid")
        if self._request_lookahead_observations is None:
            self._request_lookahead_observations = [0] * _MAX_PROPOSAL_POSITIONS
            self._request_lookahead_hits = [0] * _MAX_PROPOSAL_POSITIONS
            self._request_lookahead_greedy_hits = [0] * _MAX_PROPOSAL_POSITIONS
        assert self._request_lookahead_hits is not None
        assert self._request_lookahead_greedy_hits is not None
        self._request_lookahead_observations[position] += 1
        self._request_lookahead_hits[position] += int(planned_token == token)
        self._request_lookahead_greedy_hits[position] += int(greedy_token == token)
        self._request_lookahead_updates += 1

    def _apply_council_feedback(
        self,
        feedback: tuple[tuple[dict[str, float], int], ...],
        token: int,
        position: int = 0,
        planned_token: int | None = None,
        greedy_token: int | None = None,
    ) -> None:
        if not 0 <= position < _MAX_PROPOSAL_POSITIONS:
            raise MarkovDraftError("Markov feedback position is invalid")
        before_leader = max(
            range(len(self._experts)),
            key=lambda index: self._state.expert_log_weights[index],
        )
        logs = list(self._state.expert_log_weights)
        observations = list(self._state.expert_observations)
        hits = list(self._state.expert_hits)
        horizon_observations = [
            list(row) for row in self._state.horizon_expert_observations
        ]
        horizon_hits = [list(row) for row in self._state.horizon_expert_hits]
        plan_observations = list(self._state.horizon_plan_observations)
        plan_hits = list(self._state.horizon_plan_hits)
        lookahead_observations = list(self._state.lookahead_observations)
        lookahead_hits = list(self._state.lookahead_hits)
        lookahead_greedy_hits = list(self._state.lookahead_greedy_hits)
        weights, _position_maturity, _dialect_skill_maturity = self._position_weighting(
            position,
            self._weights(),
        )
        probabilities = self._feedback_probabilities(feedback, token)
        mixture_probability = sum(
            weight * probability
            for weight, probability in zip(weights, probabilities, strict=True)
        )
        surprise = -math.log(max(mixture_probability, 1e-12))
        previous_mean = self._state.surprise_mean
        previous_deviation = self._state.surprise_deviation
        if self._state.feedback_count == 0:
            z_score = 0.0
        else:
            z_score = (surprise - previous_mean) / max(previous_deviation, 1e-6)
        next_mean = (
            1.0 - self.SURPRISE_RATE
        ) * previous_mean + self.SURPRISE_RATE * surprise
        next_deviation = (
            1.0 - self.SURPRISE_RATE
        ) * previous_deviation + self.SURPRISE_RATE * abs(surprise - previous_mean)
        next_cusum = max(
            0.0,
            self.CUSUM_DECAY * self._state.surprise_cusum + z_score - self.CUSUM_DRIFT,
        )
        for index, (distribution, prediction) in enumerate(feedback):
            probability = probabilities[index]
            advantage = math.log(max(probability, 1e-12)) - math.log(
                max(mixture_probability, 1e-12)
            )
            logs[index] = (
                self.RAPIDITY_DECAY * logs[index]
                + self.EXPERT_LEARNING_RATE * advantage
            )
            observations[index] += 1
            hits[index] += int(prediction == token)
            horizon_observations[position][index] += 1
            horizon_hits[position][index] += int(prediction == token)
        center = sum(logs) / len(logs)
        logs = [value - center for value in logs]
        regime_change = (
            self._state.feedback_count + 1 >= self.REGIME_WARMUP
            and next_cusum > self.CUSUM_THRESHOLD
        )
        if regime_change:
            logs = [self.REGIME_RAPIDITY_SHRINK * value for value in logs]
            next_cusum = 0.0
        planned = token if planned_token is None else planned_token
        greedy = planned if greedy_token is None else greedy_token
        dialect = self._active_dialect
        if dialect is not None:
            local_logs = list(dialect.rapidities)
            local_observations = list(dialect.observations)
            local_hits = list(dialect.hits)
            local_horizon_observations = (
                [list(row) for row in dialect.horizon_observations]
                if dialect.horizon_observations
                else [
                    [0] * len(self._experts)
                    for _ in range(_MAX_PROPOSAL_POSITIONS)
                ]
            )
            local_horizon_hits = (
                [list(row) for row in dialect.horizon_hits]
                if dialect.horizon_hits
                else [
                    [0] * len(self._experts)
                    for _ in range(_MAX_PROPOSAL_POSITIONS)
                ]
            )
            local_plan_rows, local_plan_hit_rows = self._dialect_plan_counts(
                dialect
            )
            local_plan_observations = list(local_plan_rows)
            local_plan_hits = list(local_plan_hit_rows)
            for index, (_distribution, prediction) in enumerate(feedback):
                advantage = math.log(max(probabilities[index], 1e-12)) - math.log(
                    max(mixture_probability, 1e-12)
                )
                local_logs[index] = (
                    self.RAPIDITY_DECAY * local_logs[index]
                    + self.EXPERT_LEARNING_RATE * advantage
                )
                local_observations[index] += 1
                local_hits[index] += int(prediction == token)
                local_horizon_observations[position][index] += 1
                local_horizon_hits[position][index] += int(prediction == token)
            local_plan_observations[position] += 1
            local_plan_hits[position] += int(planned == token)
            local_center = sum(local_logs) / len(local_logs)
            local_logs = [value - local_center for value in local_logs]
            if regime_change:
                local_logs = [
                    self.REGIME_RAPIDITY_SHRINK * value for value in local_logs
                ]
            self._active_dialect = replace(
                dialect,
                rapidities=tuple(local_logs),
                observations=tuple(local_observations),
                hits=tuple(local_hits),
                horizon_observations=tuple(
                    tuple(row) for row in local_horizon_observations
                ),
                horizon_hits=tuple(tuple(row) for row in local_horizon_hits),
                plan_observations=tuple(local_plan_observations),
                plan_hits=tuple(local_plan_hits),
            )
        plan_observations[position] += 1
        plan_hits[position] += int(planned == token)
        if planned != greedy:
            lookahead_observations[position] += 1
            lookahead_hits[position] += int(planned == token)
            lookahead_greedy_hits[position] += int(greedy == token)
        after_leader = max(range(len(logs)), key=logs.__getitem__)
        self._state = replace(
            self._state,
            expert_log_weights=tuple(logs),
            expert_observations=tuple(observations),
            expert_hits=tuple(hits),
            horizon_expert_observations=tuple(
                tuple(row) for row in horizon_observations
            ),
            horizon_expert_hits=tuple(tuple(row) for row in horizon_hits),
            horizon_plan_observations=tuple(plan_observations),
            horizon_plan_hits=tuple(plan_hits),
            lookahead_observations=tuple(lookahead_observations),
            lookahead_hits=tuple(lookahead_hits),
            lookahead_greedy_hits=tuple(lookahead_greedy_hits),
            leader_changes=(
                self._state.leader_changes + int(before_leader != after_leader)
            ),
            feedback_count=self._state.feedback_count + 1,
            surprise_mean=next_mean,
            surprise_deviation=next_deviation,
            surprise_cusum=next_cusum,
            regime_generation=(self._state.regime_generation + int(regime_change)),
        )
        self._council_feedback += 1

    def _apply_recursive_horizon_feedback(
        self,
        predictions: tuple[int, ...],
        token: int,
        position: int,
        planned_token: int,
        greedy_token: int,
    ) -> None:
        if len(predictions) != len(self._experts):
            raise MarkovDraftError("recursive Markov prediction width changed")
        if not 0 <= position < _MAX_PROPOSAL_POSITIONS:
            raise MarkovDraftError("recursive Markov position is invalid")
        observations = [
            list(row) for row in self._state.horizon_expert_observations
        ]
        hits = [list(row) for row in self._state.horizon_expert_hits]
        plan_observations = list(self._state.horizon_plan_observations)
        plan_hits = list(self._state.horizon_plan_hits)
        for index, prediction in enumerate(predictions):
            observations[position][index] += 1
            hits[position][index] += int(prediction == token)
        plan_observations[position] += 1
        plan_hits[position] += int(planned_token == token)
        lookahead_observations = list(self._state.lookahead_observations)
        lookahead_hits = list(self._state.lookahead_hits)
        lookahead_greedy_hits = list(self._state.lookahead_greedy_hits)
        if planned_token != greedy_token:
            lookahead_observations[position] += 1
            lookahead_hits[position] += int(planned_token == token)
            lookahead_greedy_hits[position] += int(greedy_token == token)
        dialect = self._active_dialect
        if dialect is not None:
            dialect_observations = [
                list(row) for row in dialect.horizon_observations
            ]
            dialect_hits = [list(row) for row in dialect.horizon_hits]
            dialect_plan_rows, dialect_plan_hit_rows = (
                self._dialect_plan_counts(dialect)
            )
            dialect_plan_observations = list(dialect_plan_rows)
            dialect_plan_hits = list(dialect_plan_hit_rows)
            for index, prediction in enumerate(predictions):
                dialect_observations[position][index] += 1
                dialect_hits[position][index] += int(prediction == token)
            dialect_plan_observations[position] += 1
            dialect_plan_hits[position] += int(planned_token == token)
            self._active_dialect = replace(
                dialect,
                horizon_observations=tuple(
                    tuple(row) for row in dialect_observations
                ),
                horizon_hits=tuple(tuple(row) for row in dialect_hits),
                plan_observations=tuple(dialect_plan_observations),
                plan_hits=tuple(dialect_plan_hits),
            )
        self._state = replace(
            self._state,
            horizon_expert_observations=tuple(
                tuple(row) for row in observations
            ),
            horizon_expert_hits=tuple(tuple(row) for row in hits),
            horizon_plan_observations=tuple(plan_observations),
            horizon_plan_hits=tuple(plan_hits),
            lookahead_observations=tuple(lookahead_observations),
            lookahead_hits=tuple(lookahead_hits),
            lookahead_greedy_hits=tuple(lookahead_greedy_hits),
        )

    def _apply_planner_feedback(
        self,
        planner: int,
        position: int,
        hit: bool,
    ) -> None:
        observations = [list(row) for row in self._state.planner_observations]
        hits = [list(row) for row in self._state.planner_hits]
        observations[planner][position] += 1
        hits[planner][position] += int(hit)
        dialect = self._active_dialect
        if dialect is not None:
            dialect_rows, dialect_hit_rows = self._dialect_planner_counts(dialect)
            dialect_observations = [list(row) for row in dialect_rows]
            dialect_hits = [list(row) for row in dialect_hit_rows]
            dialect_observations[planner][position] += 1
            dialect_hits[planner][position] += int(hit)
            self._active_dialect = replace(
                dialect,
                planner_observations=tuple(
                    tuple(row) for row in dialect_observations
                ),
                planner_hits=tuple(tuple(row) for row in dialect_hits),
            )
        self._state = replace(
            self._state,
            planner_observations=tuple(tuple(row) for row in observations),
            planner_hits=tuple(tuple(row) for row in hits),
        )

    def __call__(self, history: tuple[int, ...], /) -> tuple[int, int, int, int]:
        committed = self._token_tuple(history, label="Markov draft history")
        proposal, _feedback, _confidence, _disagreement = self._predict_council(
            committed,
            4,
        )
        return proposal[0], proposal[1], proposal[2], proposal[3]

    def _consume_carry_feedback(self, token: int) -> None:
        feedback = self._carry_feedback
        if feedback is None:
            return
        position = self._carry_feedback_position
        if position is None:
            raise MarkovDraftError("Markov carry feedback position is missing")
        plan = self._carry_plan
        if plan is None:
            raise MarkovDraftError("Markov carry planning trace is missing")
        self._record_confirmed_feedback(feedback, token, position, *plan)
        if self._carry_feedback_teacher_forced:
            self._teacher_forced_feedback_tokens += 1
            self._external_feedback_tokens += 1
        self._carry_feedback = None
        self._carry_feedback_position = None
        self._carry_feedback_teacher_forced = False
        self._carry_plan = None

    def _arm_recursive_trace(self, position: int) -> None:
        complete = self._pending_complete
        feedback = self._pending_feedback
        plan_trace = self._pending_plan_trace
        if (
            not 0 <= position < len(complete)
            or len(feedback) != len(complete)
            or len(plan_trace) != len(complete)
        ):
            raise MarkovDraftError("recursive Markov trace source is invalid")
        if position + 1 >= len(complete):
            return
        self._recursive_traces.append(
            _RecursiveMarkovTrace(
                token_ids=complete,
                expert_predictions=tuple(
                    tuple(prediction for _distribution, prediction in row)
                    for row in feedback
                ),
                plan_trace=plan_trace,
                next_position=position,
                gate_pending=True,
            )
        )
        self._recursive_trace_created += 1
        self._recursive_trace_peak_active = max(
            self._recursive_trace_peak_active,
            len(self._recursive_traces),
        )

    def _start_pending_planner_trace(self) -> None:
        candidates = self._pending_planner_candidates
        if not candidates:
            return
        if (
            len(candidates) != len(_PLANNER_NAMES)
            or any(
                row and len(row) != len(self._pending_complete)
                for row in candidates
            )
            or not self._pending_complete
            or sum(bool(row) for row in candidates) < 2
        ):
            raise MarkovDraftError("pending planner tournament is invalid")
        self._planner_traces.append(
            _PlannerTournamentTrace(
                candidates=candidates,
                alive=tuple(bool(row) for row in candidates),
                next_position=0,
            )
        )
        self._planner_trace_created += 1

    def _validate_planner_traces(self) -> None:
        for trace in self._planner_traces:
            if (
                len(trace.candidates) != len(_PLANNER_NAMES)
                or len(trace.alive) != len(_PLANNER_NAMES)
                or not 0 <= trace.next_position < _MAX_PROPOSAL_POSITIONS
                or any(
                    active and len(candidate) <= trace.next_position
                    for candidate, active in zip(
                        trace.candidates,
                        trace.alive,
                        strict=True,
                    )
                )
            ):
                raise MarkovDraftError("planner tournament trace is invalid")

    def _advance_planner_traces(self, tokens: Sequence[int], /) -> None:
        self._validate_planner_traces()
        for token in tokens:
            surviving = []
            for trace in self._planner_traces:
                position = trace.next_position
                alive = []
                for planner, (candidate, active) in enumerate(
                    zip(trace.candidates, trace.alive, strict=True)
                ):
                    if not active:
                        alive.append(False)
                        continue
                    hit = candidate[position] == token
                    self._update_request_planner(planner, position, hit)
                    self._planner_trace_feedback_tokens += 1
                    alive.append(hit and position + 1 < len(candidate))
                if any(alive):
                    surviving.append(
                        replace(
                            trace,
                            alive=tuple(alive),
                            next_position=position + 1,
                        )
                    )
            self._planner_traces = surviving

    def _validate_recursive_traces(self) -> None:
        for trace in self._recursive_traces:
            if (
                len(trace.token_ids) != len(trace.expert_predictions)
                or len(trace.token_ids) != len(trace.plan_trace)
                or not 0 <= trace.next_position < len(trace.token_ids)
                or any(
                    len(predictions) != len(self._experts)
                    for predictions in trace.expert_predictions
                )
            ):
                raise MarkovDraftError("recursive Markov trace is invalid")

    def _advance_recursive_traces(self, tokens: Sequence[int], /) -> None:
        self._validate_recursive_traces()
        for token in tokens:
            surviving: list[_RecursiveMarkovTrace] = []
            for trace in self._recursive_traces:
                position = trace.next_position
                hit = trace.token_ids[position] == token
                if trace.gate_pending:
                    if hit and position + 1 < len(trace.token_ids):
                        surviving.append(
                            replace(
                                trace,
                                next_position=position + 1,
                                gate_pending=False,
                            )
                        )
                    continue
                planned_token, greedy_token, _lookahead_gain = (
                    trace.plan_trace[position]
                )
                predictions = trace.expert_predictions[position]
                self._update_request_position_predictions(
                    predictions,
                    token,
                    position,
                )
                self._update_request_plan(planned_token, token, position)
                self._update_request_lookahead(
                    planned_token,
                    greedy_token,
                    token,
                    position,
                )
                self._recursive_feedback.append(
                    (
                        predictions,
                        token,
                        position,
                        planned_token,
                        greedy_token,
                    )
                )
                self._recursive_trace_feedback_tokens += 1
                self._recursive_trace_hits += int(hit)
                self._recursive_trace_misses += int(not hit)
                self._recursive_trace_max_position = max(
                    self._recursive_trace_max_position,
                    position,
                )
                if hit and position + 1 < len(trace.token_ids):
                    surviving.append(
                        replace(trace, next_position=position + 1)
                    )
            self._recursive_traces = surviving

    def _advance_confirmed_tokens(self, tokens: Sequence[int], /) -> None:
        self._validate_planner_traces()
        self._validate_recursive_traces()
        for index, token in enumerate(tokens):
            self._advance_planner_traces((token,))
            self._advance_recursive_traces((token,))
            if index == 0 and self._carry_feedback is not None:
                self._consume_carry_feedback(token)

    def _prepare_rolling_proposal(
        self,
        history: tuple[int, ...],
        known_token: int,
    ) -> tuple[
        tuple[int, ...],
        tuple[float, ...],
        tuple[float, ...],
        MarkovPhraseOption | None,
    ]:
        if self._closed:
            raise MarkovDraftError("Markov draft provider is closed")
        if self._request_completed:
            raise MarkovDraftError("Markov provider request is complete")
        committed = self._token_tuple(history, label="rolling Markov history")
        if (
            isinstance(known_token, bool)
            or not isinstance(known_token, int)
            or not 0 <= known_token < self.vocab_size
        ):
            raise ValueError("known token is outside the Qwen vocabulary")
        if self._pending_base is not None:
            raise MarkovDraftError("previous Markov proposal was not reconciled")
        if self._last_confirmed_length is None:
            if not self._request_started:
                self._activate_dialect(committed)
                self._request_started = True
            self._last_confirmed_length = len(committed)
        elif len(committed) != self._last_confirmed_length:
            raise MarkovDraftError("rolling Markov history length is discontinuous")
        self._advance_confirmed_tokens((known_token,))
        base = (*committed, known_token)
        option = self._phrase_option(base)
        has_request_local_transition = bool(
            self._request_local_options(base, limit=1)
        )
        can_beam = not (
            self.atlas is None
            and not self._state.token_ids
            and not has_request_local_transition
        )
        candidate_results: dict[int, object] = {}
        candidate_traces: dict[int, tuple[tuple[int, int, float], ...]] = {}
        candidate_states: dict[int, tuple[object, ...]] = {}

        council_result, council_trace, council_state = self._capture_planning_call(
            lambda: self._predict_council(base, self.proposal_width + 1)
        )
        candidate_results[1] = council_result
        candidate_traces[1] = council_trace
        candidate_states[1] = council_state
        if can_beam:
            beam_result, beam_trace, beam_state = self._capture_planning_call(
                lambda: self._predict_beam(base, self.proposal_width + 1)
            )
            if beam_result is not None:
                candidate_results[0] = beam_result
                candidate_traces[0] = beam_trace
                candidate_states[0] = beam_state
        if option is not None:
            phrase_result, phrase_trace, phrase_state = self._capture_planning_call(
                lambda: self._predict_council(
                    base,
                    self.proposal_width + 1,
                    forced_prefix=option.token_ids[: self.proposal_width],
                )
            )
            candidate_results[2] = phrase_result
            candidate_traces[2] = phrase_trace
            candidate_states[2] = phrase_state

        available = tuple(sorted(candidate_results))
        default_planner = (
            2
            if option is not None and option.kind != "atlas"
            else 0
            if 0 in candidate_results
            else 2
            if option is not None
            else 1
        )
        has_planner_evidence = any(
            self._planner_counts(planner_index, position)[0] > 0
            for planner_index in available
            for position in range(self.proposal_width + 1)
        )
        if not has_planner_evidence:
            selected_planner = default_planner
        else:
            selected_planner = max(
                available,
                key=lambda planner_index: (
                    self._planner_utility(
                        planner_index,
                        candidate_results[planner_index][2],
                    ),
                    int(planner_index == default_planner),
                    -planner_index,
                ),
            )
        planner = _PLANNER_NAMES[selected_planner]
        complete, feedback, confidences, disagreements = candidate_results[
            selected_planner
        ]
        self._restore_planning_snapshot(candidate_states[selected_planner])
        self._last_plan_trace = candidate_traces[selected_planner]
        planner_candidates = (
            ()
            if len(available) < 2
            else tuple(
                ()
                if planner_index not in candidate_results
                else candidate_results[planner_index][0]
                for planner_index in range(len(_PLANNER_NAMES))
            )
        )
        if planner_candidates:
            self._planner_tournament_calls += 1
            if selected_planner == 0:
                self._planner_beam_selections += 1
            elif selected_planner == 1:
                self._planner_council_selections += 1
            else:
                self._planner_phrase_selections += 1
        retain_option = selected_planner == 2 or (
            selected_planner == 0
            and option is not None
            and option.kind in {"atlas", "crystal"}
            and complete[: len(option.token_ids)] == option.token_ids
        )
        if option is not None and not retain_option:
            option = None
            self._pending_composition_program = None
        proposal = complete[: self.proposal_width]
        if option is not None and option.kind == "crystal":
            calibrated_confidences = list(confidences)
            calibrated_disagreements = list(disagreements)
            for index, value in enumerate(
                option.token_confidences[: self.proposal_width]
            ):
                calibrated_confidences[index] = value
                calibrated_disagreements[index] = option.token_disagreements[index]
            confidences = tuple(calibrated_confidences)
            disagreements = tuple(calibrated_disagreements)
        if len(self._last_plan_trace) != self.proposal_width + 1:
            raise MarkovDraftError("Markov planning trace width is invalid")
        self._pending_base = base
        self._pending_proposal = proposal
        self._pending_complete = complete
        self._pending_planner_candidates = planner_candidates
        self._pending_feedback = feedback
        self._pending_plan_trace = self._last_plan_trace
        self._pending_planner = planner
        self._pending_phrase_option = option
        if option is not None:
            self._phrase_option_calls += 1
            self._phrase_draft_tokens += min(
                len(option.token_ids),
                self.proposal_width,
            )
            self._last_phrase_option = option
            if option.kind == "atlas":
                self._atlas_option_calls += 1
                self._atlas_draft_tokens += min(
                    len(option.token_ids),
                    self.proposal_width,
                )
            if option.kind == "composition":
                self._composition_option_calls += 1
                self._composition_draft_tokens += min(
                    len(option.token_ids),
                    self.proposal_width,
                )
                self._last_composition_program = self._pending_composition_program
            if option.kind == "periodic":
                self._periodic_option_calls += 1
                self._periodic_draft_tokens += min(
                    len(option.token_ids),
                    self.proposal_width,
                )
            if option.kind == "binding":
                self._periodic_option_calls += 1
                self._periodic_draft_tokens += min(
                    len(option.token_ids),
                    self.proposal_width,
                )
                self._binding_option_calls += 1
                self._binding_draft_tokens += min(
                    len(option.token_ids),
                    self.proposal_width,
                )
            if option.kind == "crystal" and option.crystal_layer is None:
                self._crystal_option_calls += 1
                self._crystal_proposed_tokens += min(
                    len(option.token_ids),
                    self.proposal_width,
                )
            if option.kind == "crystal" and option.crystal_layer is not None:
                selected_width = min(
                    len(option.token_ids),
                    self.proposal_width,
                )
                layer_counter = self._layer_context_crystal_layer_counter(
                    option.crystal_layer
                )
                layer_counter["crystal_proposed_tokens"] += selected_width
                self._layer_context_crystal_counters[
                    "crystal_proposed_tokens"
                ] += selected_width
                self._layer_context_crystal_selected = {
                    "cell_sha256": option.cell_sha256,
                    "layer": option.crystal_layer,
                    "token_ids": list(option.token_ids),
                    "transaction_sha256": option.crystal_transaction_sha256,
                }
        self._draft_calls += 1
        return (
            proposal,
            confidences[: self.proposal_width],
            disagreements[: self.proposal_width],
            option,
        )

    def propose_after(
        self,
        history: tuple[int, ...],
        known_token: int,
        /,
    ) -> tuple[int, ...]:
        proposal, _confidence, _disagreement, _option = self._prepare_rolling_proposal(
            history, known_token
        )
        return proposal

    def propose_after_state(
        self,
        history: tuple[int, ...],
        known_token: int,
        target_hidden: torch.Tensor,
        /,
    ) -> tuple[int, ...]:
        """Use one immutable target boundary to query continuation Crystals."""

        self._load_context_crystal_boundary(history, known_token, target_hidden)
        self._load_layer_context_crystal_boundary(history, known_token)
        try:
            return self.propose_after(history, known_token)
        finally:
            self._context_crystal_key = None
            self._context_crystal_candidates = ()
            self._layer_context_crystal_phrase_options = ()

    def propose_round(
        self,
        history: tuple[int, ...],
        known_token: int,
        /,
    ) -> RollingDraftProposal:
        if self.proposal_width + 1 not in {4, 8, 16}:
            raise MarkovDraftError(
                "adaptive Markov proposal requires a K4/K8/K16 ceiling"
            )
        proposal, confidences, disagreements, option = self._prepare_rolling_proposal(
            history, known_token
        )
        phrase_confidence = 0.0 if option is None else option.confidence
        if (
            option is not None
            and option.kind == "atlas"
            and self._pending_planner == "beam"
        ):
            reliability = 1.0
            for position in range(min(len(option.token_ids), self.proposal_width)):
                reliability = min(
                    reliability,
                    self._beam_position_reliability(position),
                )
            phrase_confidence *= reliability
        result = RollingDraftProposal.build(
            proposal,
            confidences,
            disagreements,
            request_window_ceiling=self.proposal_width + 1,
            provider_abi=MARKOV_DRAFT_PROVIDER_ABI,
            phrase_source=None if option is None else option.source,
            phrase_support=0 if option is None else option.support,
            phrase_confidence=phrase_confidence,
            phrase_width=(
                0 if option is None else min(len(option.token_ids), self.proposal_width)
            ),
        )
        self._adaptive_proposal_calls += 1
        self._recommended_window_counts[result.recommended_window] += 1
        self._last_round_proposal = result
        return result

    def propose_round_state(
        self,
        history: tuple[int, ...],
        known_token: int,
        target_hidden: torch.Tensor,
        /,
    ) -> RollingDraftProposal:
        """Rank a hidden-state Crystal inside the existing phrase planner."""

        self._load_context_crystal_boundary(history, known_token, target_hidden)
        self._load_layer_context_crystal_boundary(history, known_token)
        try:
            return self.propose_round(history, known_token)
        finally:
            self._context_crystal_key = None
            self._context_crystal_candidates = ()
            self._layer_context_crystal_phrase_options = ()

    def atlas_evidence_for_pending(
        self,
        token_ids: Sequence[int],
        /,
    ) -> tuple[AtlasTokenEvidence, ...]:
        """Score another provider's exact proposal against the Atlas.

        The pending Markov base already includes the target-known token for
        this round.  Scoring is read-only: it cannot change the Markov proposal
        or learn from unverified MTP tokens.
        """

        proposed = self._token_tuple(
            token_ids,
            label="Atlas Council proposal",
        )
        if self._pending_base is None or self._pending_proposal is None:
            raise MarkovDraftError("Atlas Council vote requires a pending proposal")
        if len(proposed) != len(self._pending_proposal):
            raise MarkovDraftError("Atlas Council proposal width changed")
        if self.atlas is None:
            return ()
        evidence = self.atlas.sequence_evidence(self._pending_base, proposed)
        self._atlas_vote_calls += 1
        self._atlas_vote_tokens += len(evidence)
        self._atlas_vote_supported_tokens += sum(row.support > 0 for row in evidence)
        self._atlas_vote_score_sum += sum(row.score for row in evidence)
        self._atlas_vote_max_score = max(
            self._atlas_vote_max_score,
            *(row.score for row in evidence),
        )
        return evidence

    def language_evidence_for_pending(
        self,
        token_ids: Sequence[int],
        /,
    ) -> tuple[MarkovLanguageTokenEvidence, ...]:
        """Join static Atlas support with the live target-confirmed PPM overlay."""

        proposed = self._token_tuple(
            token_ids,
            label="language Council proposal",
        )
        if self._pending_base is None or self._pending_proposal is None:
            raise MarkovDraftError("language Council vote requires a pending proposal")
        if len(proposed) != len(self._pending_proposal):
            raise MarkovDraftError("language Council proposal width changed")
        atlas_rows = self.atlas_evidence_for_pending(proposed)
        experts = self._expert_models(self._pending_base)
        request_local = self._request_local_fingerprint(self._pending_base)
        request_local_context = (
            [] if request_local is None else list(request_local[1])
        )
        base_weights = self._weights()
        rows = []
        for position, token_id in enumerate(proposed):
            weights = self._position_weighting(position, base_weights)[0]
            position_observations, position_hits = self._position_counts(position)
            symbol = self._symbol(token_id)
            online_score = 0.0
            online_support = 0
            for index, (spec, (model, context), weight) in enumerate(
                zip(self._experts, experts, weights, strict=True)
            ):
                if spec.local_only:
                    context.append(symbol)
                    continue
                score, support, _total, _order = model.token_evidence(
                    context,
                    symbol,
                )
                observations = position_observations[index]
                hits = position_hits[index]
                if observations <= 0:
                    observations = self._state.expert_observations[index]
                    hits = self._state.expert_hits[index]
                reliability = 0.0
                if observations > 0:
                    maturity = observations / (
                        observations + self.EMPIRICAL_EVIDENCE_SATURATION
                    )
                    posterior = (hits + 1.0) / (observations + 2.0)
                    reliability = maturity * posterior
                online_score += float(weight) * score * reliability
                online_support = max(online_support, support)
                context.append(symbol)
            request_local_score = 0.0
            if request_local is not None:
                (
                    request_local_score,
                    request_local_support,
                    _request_local_total,
                    _request_local_order,
                ) = request_local[0].contextual_confidence(
                    request_local_context,
                    symbol,
                    min_order=self.REQUEST_LOCAL_MIN_ORDER,
                    min_support=self.REQUEST_LOCAL_MIN_SUPPORT,
                    support_scale=self.REQUEST_LOCAL_SUPPORT_SCALE,
                )
                if request_local_score > 0.0:
                    online_support = max(online_support, request_local_support)
                request_local_context.append(symbol)
            atlas_row = None if not atlas_rows else atlas_rows[position]
            atlas_score = 0.0 if atlas_row is None else atlas_row.score
            atlas_support = 0 if atlas_row is None else atlas_row.support
            online_score = max(0.0, min(1.0, online_score))
            online_score = 1.0 - (1.0 - online_score) * (
                1.0 - request_local_score
            )
            combined = 1.0 - (1.0 - atlas_score) * (1.0 - online_score)
            rows.append(
                MarkovLanguageTokenEvidence(
                    token_id=token_id,
                    atlas_score=atlas_score,
                    online_score=online_score,
                    score=combined,
                    atlas_support=atlas_support,
                    online_support=online_support,
                )
            )
        result = tuple(rows)
        self._online_vote_calls += 1
        self._online_vote_tokens += len(result)
        self._online_vote_supported_tokens += sum(
            row.online_support > 0 for row in result
        )
        self._online_vote_score_sum += sum(row.online_score for row in result)
        self._online_vote_max_score = max(
            self._online_vote_max_score,
            *(row.online_score for row in result),
        )
        return result

    def _learn_episode(
        self,
        tokens: Sequence[int],
        *,
        prompt_length: int | None,
        priority: float = 1.0,
    ) -> None:
        episode = tuple(tokens)
        if not episode:
            return
        if prompt_length is not None and (
            isinstance(prompt_length, bool)
            or not isinstance(prompt_length, int)
            or not 1 <= prompt_length < len(episode)
        ):
            raise MarkovDraftError("confirmed episode prompt boundary is invalid")
        if self._active_dialect is None:
            raise MarkovDraftError("confirmed episode has no dialect binding")
        if not math.isfinite(priority) or priority < 0.0:
            raise MarkovDraftError("confirmed episode priority is invalid")
        episodes = []
        priorities = []
        dialect_ids = list(self._state.episode_dialects)
        prompt_lengths = list(self._state.episode_prompt_lengths)
        offset = 0
        for length in self._state.episode_lengths:
            retained = self._state.token_ids[offset : offset + length]
            episodes.append(retained)
            offset += length
        for retained, boundary in zip(
            episodes,
            prompt_lengths,
            strict=True,
        ):
            generated = retained if boundary is None else retained[boundary:]
            priorities.append(self._retention_priority(generated))
        episodes.append(episode)
        priorities.append(priority)
        dialect_ids.append(self._active_dialect.dialect_id)
        prompt_lengths.append(prompt_length)
        total = sum(len(row) for row in episodes)
        while len(episodes) > 1 and total > self.max_history_tokens:
            latest = len(episodes) - 1
            evicted = min(
                range(len(episodes)),
                key=lambda index: (
                    priorities[index]
                    * math.exp(
                        -self.RICCI_AGE_ALPHA * max(0, latest - index)
                    ),
                    index,
                ),
            )
            total -= len(episodes.pop(evicted))
            priorities.pop(evicted)
            dialect_ids.pop(evicted)
            prompt_lengths.pop(evicted)
            self._retention_priority_evictions += 1
        if total > self.max_history_tokens:
            removed = len(episodes[0]) - self.max_history_tokens
            episodes[0] = episodes[0][-self.max_history_tokens :]
            boundary = prompt_lengths[0]
            adjusted = None if boundary is None else boundary - removed
            prompt_lengths[0] = (
                adjusted
                if adjusted is not None and 1 <= adjusted < len(episodes[0])
                else None
            )
            retained_answer = (
                episodes[0]
                if prompt_lengths[0] is None
                else episodes[0][prompt_lengths[0] :]
            )
            if self.episode_priority_store is not None and retained_answer:
                try:
                    self.episode_priority_store(retained_answer, priorities[0])
                except Exception:
                    self._retention_failures += 1
        combined = tuple(token for row in episodes for token in row)
        self._state = replace(
            self._state,
            token_ids=combined,
            episode_lengths=tuple(len(row) for row in episodes),
            episode_dialects=tuple(dialect_ids),
            episode_prompt_lengths=tuple(prompt_lengths),
            updates=self._state.updates + 1,
        )
        self._persistent_symbols_cache = None
        self._ricci_working_symbols_cache.clear()
        self._ricci_episode_cache = None
        self._ricci_priority_degraded = False
        self._episode_priority_cache.clear()
        self._persistent_expert_models.clear()
        self._composition_cache.clear()

    def _commit_active_dialect(self) -> None:
        dialect = self._active_dialect
        if dialect is None:
            raise MarkovDraftError("completed request has no active dialect")
        clock = self._state.clock + 1
        signature = tuple(
            sorted(set(dialect.signature) | set(self._request_signature))[
                : self.DIALECT_SKETCH_SIZE
            ]
        )
        committed = replace(
            dialect,
            signature=signature,
            visits=dialect.visits
            if self._active_dialect_is_new
            else dialect.visits + 1,
            last_seen=clock,
        )
        profiles = {row.dialect_id: row for row in self._state.dialects}
        profiles[committed.dialect_id] = committed
        if len(profiles) > self.MAX_DIALECTS:
            candidates = [
                row
                for row in profiles.values()
                if row.dialect_id != committed.dialect_id
            ]
            evicted = min(
                candidates,
                key=lambda row: (
                    row.visits
                    * math.exp(-self.RICCI_AGE_ALPHA * (clock - row.last_seen)),
                    row.last_seen,
                    row.dialect_id,
                ),
            )
            del profiles[evicted.dialect_id]
            self._dialect_evictions += 1
        self._state = replace(
            self._state,
            clock=clock,
            dialects=tuple(sorted(profiles.values(), key=lambda row: row.dialect_id)),
        )

    def reconcile_prefix(self, history: tuple[int, ...], /) -> None:
        committed = self._token_tuple(history, label="reconciled Markov history")
        base = self._pending_base
        proposal = self._pending_proposal
        if base is None or proposal is None:
            raise MarkovDraftError("reconcile_prefix requires a pending proposal")
        if len(self._pending_feedback) != self.proposal_width + 1:
            raise MarkovDraftError("Markov council proposal feedback is missing")
        if len(self._pending_plan_trace) != self.proposal_width + 1:
            raise MarkovDraftError("Markov planning trace is missing")
        if committed[: len(base)] != base:
            raise MarkovDraftError("Markov reconciliation changed its known base")
        delta = committed[len(base) :]
        if len(delta) > self.proposal_width or delta != proposal[: len(delta)]:
            raise MarkovDraftError("Markov reconciliation is not a proposal prefix")
        self._commit_pending_verification(len(delta))
        assert self._last_confirmed_length is not None
        for index, token in enumerate(delta):
            self._record_confirmed_feedback(
                self._pending_feedback[index],
                token,
                index,
                *self._pending_plan_trace[index],
            )
        if self._pending_phrase_option is not None:
            self._phrase_accepted_tokens += min(
                len(delta),
                len(self._pending_phrase_option.token_ids),
            )
            if self._pending_phrase_option.kind == "atlas":
                self._atlas_accepted_tokens += min(
                    len(delta),
                    len(self._pending_phrase_option.token_ids),
                )
            if self._pending_phrase_option.kind == "composition":
                self._composition_accepted_tokens += min(
                    len(delta),
                    len(self._pending_phrase_option.token_ids),
                )
            if self._pending_phrase_option.kind == "periodic":
                self._periodic_accepted_tokens += min(
                    len(delta),
                    len(self._pending_phrase_option.token_ids),
                )
            if self._pending_phrase_option.kind == "binding":
                self._periodic_accepted_tokens += min(
                    len(delta),
                    len(self._pending_phrase_option.token_ids),
                )
                self._binding_accepted_tokens += min(
                    len(delta),
                    len(self._pending_phrase_option.token_ids),
                )
        self._validate_planner_traces()
        self._validate_recursive_traces()
        self._start_pending_planner_trace()
        self._advance_planner_traces(delta)
        self._advance_recursive_traces(delta)
        self._carry_feedback = self._pending_feedback[len(delta)]
        self._carry_feedback_position = len(delta)
        self._carry_feedback_teacher_forced = False
        self._carry_plan = self._pending_plan_trace[len(delta)]
        self._arm_recursive_trace(len(delta))
        self._last_confirmed_length = len(committed)
        self._pending_base = None
        self._pending_proposal = None
        self._pending_complete = ()
        self._pending_planner_candidates = ()
        self._pending_feedback = ()
        self._pending_plan_trace = ()
        self._pending_planner = None
        self._pending_accepted_prefix_length = None
        self._pending_verified_proposals = None
        self._pending_verification_virtual = False
        self._pending_phrase_option = None
        self._pending_composition_program = None
        self._reconcile_calls += 1

    def _record_context_crystal_verification(
        self,
        accepted_prefix_length: int,
        verified_proposals: int,
    ) -> None:
        option = self._pending_phrase_option
        if (
            option is None
            or option.kind != "crystal"
            or option.crystal_layer is not None
        ):
            return
        width = min(verified_proposals, len(option.token_ids))
        if width <= 0:
            return
        accepted = min(accepted_prefix_length, width)
        assert option.cell_sha256 is not None
        self._context_crystal_feedback.append(
            ContextualCandidateFeedback(
                option.cell_sha256,
                accepted,
                width,
            )
        )
        self._crystal_verified_tokens += width
        self._crystal_accepted_tokens += accepted
        self._crystal_mismatches += int(accepted < width)

    def _record_layer_context_crystal_verification(
        self,
        accepted_prefix_length: int,
        verified_proposals: int,
    ) -> None:
        """Count only the layer-Crystal suffix actually offered to the target."""

        option = self._pending_phrase_option
        if (
            option is None
            or option.kind != "crystal"
            or option.crystal_layer is None
        ):
            return
        width = min(verified_proposals, len(option.token_ids))
        if width <= 0:
            return
        accepted = min(accepted_prefix_length, width)
        layer_counter = self._layer_context_crystal_layer_counter(
            option.crystal_layer
        )
        layer_counter["crystal_verified_tokens"] += width
        layer_counter["crystal_accepted_tokens"] += accepted
        layer_counter["crystal_mismatches"] += int(accepted < width)
        self._layer_context_crystal_counters["crystal_verified_tokens"] += width
        self._layer_context_crystal_counters["crystal_accepted_tokens"] += accepted
        self._layer_context_crystal_counters["crystal_mismatches"] += int(
            accepted < width
        )

    def observe_verification(
        self,
        accepted_prefix_length: int,
        verified_proposals: int,
        /,
    ) -> None:
        if self._pending_base is None or self._pending_proposal is None:
            raise MarkovDraftError("verification requires a pending proposal")
        if (
            isinstance(accepted_prefix_length, bool)
            or not isinstance(accepted_prefix_length, int)
            or accepted_prefix_length < 0
            or isinstance(verified_proposals, bool)
            or not isinstance(verified_proposals, int)
            or verified_proposals < 0
            or accepted_prefix_length > verified_proposals
            or verified_proposals > self.proposal_width
        ):
            raise ValueError("verification prefix counts are invalid")
        if self._pending_verified_proposals is not None:
            raise MarkovDraftError("proposal verification was already observed")
        self._pending_accepted_prefix_length = accepted_prefix_length
        self._pending_verified_proposals = verified_proposals
        self._pending_verification_virtual = False
        self._record_context_crystal_verification(
            accepted_prefix_length,
            verified_proposals,
        )
        self._record_layer_context_crystal_verification(
            accepted_prefix_length,
            verified_proposals,
        )

    def observe_virtual_verification(
        self,
        accepted_prefix_length: int,
        verified_proposals: int,
        /,
    ) -> None:
        """Record a target-checked K1 prediction that was not emitted yet."""

        if self._pending_base is None or self._pending_proposal is None:
            raise MarkovDraftError("virtual verification requires a pending proposal")
        if (
            isinstance(accepted_prefix_length, bool)
            or not isinstance(accepted_prefix_length, int)
            or accepted_prefix_length < 0
            or isinstance(verified_proposals, bool)
            or not isinstance(verified_proposals, int)
            or verified_proposals < 0
            or accepted_prefix_length > verified_proposals
            or verified_proposals > 1
        ):
            raise ValueError("virtual verification prefix counts are invalid")
        if self._pending_verified_proposals is not None:
            raise MarkovDraftError("proposal verification was already observed")
        self._pending_accepted_prefix_length = accepted_prefix_length
        self._pending_verified_proposals = verified_proposals
        self._pending_verification_virtual = True
        self._record_context_crystal_verification(
            accepted_prefix_length,
            verified_proposals,
        )
        self._record_layer_context_crystal_verification(
            accepted_prefix_length,
            verified_proposals,
        )

    def _commit_pending_verification(self, accepted_prefix_length: int) -> None:
        observed = self._pending_accepted_prefix_length
        verified = self._pending_verified_proposals
        if observed is None and verified is None:
            return
        if observed is None or verified is None:
            raise MarkovDraftError(
                "verification acceptance differs from reconciled prefix"
            )
        if self._pending_verification_virtual:
            if accepted_prefix_length != 0:
                raise MarkovDraftError(
                    "virtual verification requires an uncommitted proposal"
                )
        elif observed != accepted_prefix_length:
            raise MarkovDraftError(
                "verification acceptance differs from reconciled prefix"
            )
        if self._pending_planner == "beam":
            self._record_beam_verification(observed, verified)

    def _record_beam_verification(self, accepted: int, verified: int) -> None:
        self._beam_verified_tokens += verified
        self._beam_accepted_tokens += accepted
        for position in range(verified):
            self._beam_position_verified[position] += 1
            self._beam_position_hits[position] += int(position < accepted)

    def reconcile_external_prefix(self, history: tuple[int, ...], /) -> None:
        """Train one unused Council proposal from another verified provider.

        Feedback remains valid through the first mismatch: later Council rows
        were conditioned on its rejected token rather than the actual target
        prefix.  A fully matching observed prefix retains the original carry;
        after a mismatch, a new carry is predicted from the actual confirmed
        prefix and scored by the next target token.
        """

        if self._closed:
            raise MarkovDraftError("Markov draft provider is closed")
        if not self._request_started or self._request_completed:
            raise MarkovDraftError("Markov request is not active")
        committed = self._token_tuple(
            history,
            label="externally reconciled Markov history",
        )
        base = self._pending_base
        proposal = self._pending_proposal
        if base is None or proposal is None:
            raise MarkovDraftError(
                "reconcile_external_prefix requires a pending proposal"
            )
        if len(self._pending_feedback) != self.proposal_width + 1:
            raise MarkovDraftError("Markov council proposal feedback is missing")
        if len(self._pending_plan_trace) != self.proposal_width + 1:
            raise MarkovDraftError("Markov planning trace is missing")
        if committed[: len(base)] != base:
            raise MarkovDraftError(
                "external Markov reconciliation changed its known base"
            )
        delta = committed[len(base) :]
        if len(delta) > self.proposal_width:
            raise MarkovDraftError(
                "external Markov reconciliation exceeds the proposal width"
            )
        matching_prefix = 0
        for actual, predicted in zip(delta, proposal, strict=False):
            if actual != predicted:
                break
            matching_prefix += 1
        self._commit_pending_verification(matching_prefix)
        if (
            self._pending_planner == "beam"
            and self._pending_verified_proposals is None
        ):
            externally_verified = min(
                len(delta),
                matching_prefix + int(matching_prefix < len(delta)),
            )
            self._record_beam_verification(matching_prefix, externally_verified)
        assert self._last_confirmed_length is not None
        verified = 0
        prefix_matches = True
        mismatch_index: int | None = None
        teacher_failed = False
        for index, token in enumerate(delta):
            self._record_confirmed_feedback(
                self._pending_feedback[index],
                token,
                index,
                *self._pending_plan_trace[index],
            )
            verified += 1
            if token != proposal[index]:
                prefix_matches = False
                mismatch_index = index
                break
        if mismatch_index is not None:
            for index in range(mismatch_index + 1, len(delta)):
                actual_context = (*base, *delta[:index])
                try:
                    teacher_feedback, _teacher_token, teacher_plan = (
                        self._teacher_forced_prediction(
                            actual_context,
                            position=index,
                        )
                    )
                except (MarkovDraftError, ValueError):
                    self._teacher_forced_failures += 1
                    teacher_failed = True
                    break
                self._record_confirmed_feedback(
                    teacher_feedback,
                    delta[index],
                    index,
                    *teacher_plan,
                )
                verified += 1
                self._teacher_forced_predictions += 1
                self._teacher_forced_feedback_tokens += 1
        self._validate_planner_traces()
        self._validate_recursive_traces()
        self._start_pending_planner_trace()
        self._advance_planner_traces(delta)
        self._advance_recursive_traces(delta)
        if prefix_matches:
            self._carry_feedback = self._pending_feedback[len(delta)]
            self._carry_feedback_position = len(delta)
            self._carry_feedback_teacher_forced = False
            self._carry_plan = self._pending_plan_trace[len(delta)]
            self._arm_recursive_trace(len(delta))
        elif not teacher_failed and len(delta) < _MAX_PROPOSAL_POSITIONS:
            try:
                teacher_carry, _teacher_token, teacher_plan = (
                    self._teacher_forced_prediction(
                        (*base, *delta),
                        position=len(delta),
                    )
                )
            except (MarkovDraftError, ValueError):
                self._teacher_forced_failures += 1
                self._carry_feedback = None
                self._carry_feedback_position = None
                self._carry_feedback_teacher_forced = False
                self._carry_plan = None
            else:
                self._carry_feedback = teacher_carry
                self._carry_feedback_position = len(delta)
                self._carry_feedback_teacher_forced = True
                self._carry_plan = teacher_plan
                self._teacher_forced_predictions += 1
        else:
            self._carry_feedback = None
            self._carry_feedback_position = None
            self._carry_feedback_teacher_forced = False
            self._carry_plan = None
        self._last_confirmed_length = len(committed)
        self._pending_base = None
        self._pending_proposal = None
        self._pending_complete = ()
        self._pending_planner_candidates = ()
        self._pending_feedback = ()
        self._pending_plan_trace = ()
        self._pending_planner = None
        self._pending_accepted_prefix_length = None
        self._pending_verified_proposals = None
        self._pending_verification_virtual = False
        self._pending_phrase_option = None
        self._pending_composition_program = None
        self._reconcile_calls += 1
        self._external_reconcile_calls += 1
        self._external_feedback_tokens += verified

    def discard_pending_proposal(self) -> None:
        """Drop one unused proposal while retaining request-level learning.

        A provider cascade may ask the council whether it already owns a
        useful continuation, then delegate the request to another drafter.
        The council's speculative feedback is unavailable in that case, but
        the final target-confirmed episode is still valuable and remains
        learnable through :meth:`observe_final`.
        """

        if self._closed:
            raise MarkovDraftError("Markov draft provider is closed")
        if self._pending_base is None or self._pending_proposal is None:
            raise MarkovDraftError("no pending Markov proposal to discard")
        self._pending_base = None
        self._pending_proposal = None
        self._pending_complete = ()
        self._pending_planner_candidates = ()
        self._pending_feedback = ()
        self._pending_plan_trace = ()
        self._pending_planner = None
        self._pending_accepted_prefix_length = None
        self._pending_verified_proposals = None
        self._pending_verification_virtual = False
        self._pending_phrase_option = None
        self._pending_composition_program = None
        self._last_round_proposal = None

    def advance_confirmed_prefix(self, history: tuple[int, ...], /) -> None:
        """Follow target-confirmed tokens emitted by another draft expert.

        This keeps the cheap Council eligible on every later decode round while
        another provider owns the current proposal.  No synthetic expert
        feedback is invented; the complete target episode is learned once at
        finalization as usual.
        """

        if self._closed:
            raise MarkovDraftError("Markov draft provider is closed")
        if not self._request_started or self._request_completed:
            raise MarkovDraftError("Markov request is not active")
        if self._pending_base is not None or self._pending_proposal is not None:
            raise MarkovDraftError("cannot advance an unreconciled Markov proposal")
        committed = self._token_tuple(history, label="advanced Markov history")
        prompt = self._request_prompt
        if prompt is None or committed[: len(prompt)] != prompt:
            raise MarkovDraftError("advanced Markov history changed its request prompt")
        previous = self._last_confirmed_length
        if previous is None:
            previous = len(prompt)
        if len(committed) < previous:
            raise MarkovDraftError("advanced Markov history moved backwards")
        self._advance_confirmed_tokens(committed[previous:])
        self._last_confirmed_length = len(committed)

    def _settle_context_crystals(self, committed: tuple[int, ...]) -> None:
        bank = self.contextual_continuation_bank
        if bank is None:
            self._context_crystal_captures.clear()
            self._context_crystal_feedback.clear()
            return
        captures: list[ContextualCapture] = []
        try:
            for key, boundary_index in self._context_crystal_captures:
                if (
                    boundary_index >= len(committed)
                    or committed[boundary_index] != key.known_token
                ):
                    raise MarkovDraftError(
                        "context Crystal boundary differs from target history"
                    )
                tail = committed[
                    boundary_index + 1 : boundary_index
                    + 1
                    + MAX_CONTINUATION_TOKENS
                ]
                if not tail:
                    continue
                captures.append(
                    ContextualCapture(
                        key=key,
                        target_tail=tail,
                        boundary_index=boundary_index,
                    )
                )
            bank.settle(
                captures=tuple(captures),
                feedback=tuple(self._context_crystal_feedback),
            )
            self._crystal_captures += len(captures)
        except Exception:
            self._crystal_failures += 1
        finally:
            self._context_crystal_captures.clear()
            self._context_crystal_feedback.clear()

    def _settle_layer_context_crystals(self, committed: tuple[int, ...]) -> None:
        bank = self.layer_contextual_continuation_bank
        source = self.layer_contextual_transactions_since
        start_boundary = self._layer_context_crystal_request_start_boundary
        if (
            bank is None
            or source is None
            or start_boundary is None
            or not self._layer_context_crystal_request_enabled
        ):
            self._layer_context_crystal_options_by_transaction.clear()
            self._layer_context_crystal_transactions.clear()
            return
        try:
            raw_transactions = source(start_boundary - 1)
            if (
                isinstance(raw_transactions, (str, bytes, bytearray))
                or not isinstance(raw_transactions, Sequence)
            ):
                raise TypeError(
                    "layer_contextual_transactions_since must return a sequence"
                )
            transactions = tuple(
                self._validate_layer_context_crystal_transaction(
                    transaction,
                    label="layer_contextual_transactions_since",
                )
                for transaction in raw_transactions
            )
            if (
                len({row.boundary_index for row in transactions})
                != len(transactions)
                or len({row.transaction_sha256 for row in transactions})
                != len(transactions)
            ):
                raise MarkovDraftError(
                    "layer context Crystal transaction history contains duplicates"
                )
            transactions = tuple(
                sorted(transactions, key=lambda row: row.boundary_index)
            )
            if any(
                transaction.boundary_index < start_boundary
                or transaction.boundary_index >= len(committed)
                or committed[transaction.boundary_index]
                != transaction.known_token
                for transaction in transactions
            ):
                raise MarkovDraftError(
                    "layer context Crystal transaction history differs from final "
                    "target history"
                )
        except Exception:
            self._record_layer_context_crystal_failure()
            self._layer_context_crystal_options_by_transaction.clear()
            self._layer_context_crystal_transactions.clear()
            self._layer_context_crystal_request_enabled = False
            return

        try:
            for transaction in transactions:
                boundary = transaction.boundary_index
                tail = committed[
                    boundary + 1 : boundary + 1 + MAX_CONTINUATION_TOKENS
                ]
                if not tail:
                    continue
                options = self._layer_context_crystal_options_by_transaction.get(
                    transaction.transaction_sha256,
                    (),
                )
                try:
                    bank.settle_verified_prefix(
                        transaction,
                        tail,
                        options,
                    )
                except Exception:
                    self._record_layer_context_crystal_failure(transaction.layers)
                    continue
                for layer in transaction.layers:
                    self._layer_context_crystal_layer_counter(layer)[
                        "crystal_captures"
                    ] += 1
                    self._layer_context_crystal_counters[
                        "crystal_captures"
                    ] += 1
        finally:
            self._layer_context_crystal_options_by_transaction.clear()
            self._layer_context_crystal_transactions.clear()
            self._layer_context_crystal_request_enabled = False

    def observe_final(self, history: tuple[int, ...], /) -> None:
        committed = self._token_tuple(history, label="final Markov history")
        if self._request_completed:
            raise MarkovDraftError("Markov provider request is already complete")
        if self._active_dialect is None:
            self._activate_dialect(committed)
            self._request_started = True
        if self._pending_base is not None:
            raise MarkovDraftError("cannot finalize an unreconciled proposal")
        original_state = self._state
        original_dialect = self._active_dialect
        original_carry = self._carry_feedback
        original_carry_position = self._carry_feedback_position
        original_carry_teacher_forced = self._carry_feedback_teacher_forced
        original_carry_plan = self._carry_plan
        original_recursive_traces = list(self._recursive_traces)
        original_recursive_trace_created = self._recursive_trace_created
        original_recursive_trace_peak_active = self._recursive_trace_peak_active
        original_recursive_trace_feedback_tokens = (
            self._recursive_trace_feedback_tokens
        )
        original_recursive_trace_hits = self._recursive_trace_hits
        original_recursive_trace_misses = self._recursive_trace_misses
        original_recursive_trace_max_position = self._recursive_trace_max_position
        original_feedback = list(self._episode_feedback)
        original_recursive_feedback = list(self._recursive_feedback)
        original_planner_traces = list(self._planner_traces)
        original_planner_feedback = list(self._planner_feedback)
        original_planner_trace_feedback_tokens = (
            self._planner_trace_feedback_tokens
        )
        original_confirmed_length = self._last_confirmed_length
        original_evictions = self._dialect_evictions
        original_council_feedback = self._council_feedback
        original_external_feedback_tokens = self._external_feedback_tokens
        original_teacher_forced_feedback_tokens = (
            self._teacher_forced_feedback_tokens
        )
        original_retention_scored = self._retention_scored_episodes
        original_retention_failures = self._retention_failures
        original_retention_evictions = self._retention_priority_evictions
        original_retention_priority = self._last_retention_priority
        original_persistent_symbols_cache = self._persistent_symbols_cache
        original_ricci_working_symbols_cache = dict(
            self._ricci_working_symbols_cache
        )
        original_ricci_episode_cache = self._ricci_episode_cache
        original_ricci_priority_degraded = self._ricci_priority_degraded
        original_episode_priority_cache = dict(self._episode_priority_cache)
        original_persistent_expert_models = dict(
            self._persistent_expert_models
        )
        original_composition_cache = dict(self._composition_cache)
        original_composition_program_count = self._composition_program_count
        original_request_rapidities = self._request_expert_rapidities
        original_request_weight_updates = self._request_weight_updates
        original_max_request_weight_shift = self._max_request_weight_shift
        original_request_horizon_observations = (
            None
            if self._request_horizon_observations is None
            else [list(row) for row in self._request_horizon_observations]
        )
        original_request_horizon_hits = (
            None
            if self._request_horizon_hits is None
            else [list(row) for row in self._request_horizon_hits]
        )
        original_request_plan_observations = (
            None
            if self._request_plan_observations is None
            else list(self._request_plan_observations)
        )
        original_request_plan_hits = (
            None
            if self._request_plan_hits is None
            else list(self._request_plan_hits)
        )
        original_request_planner_observations = (
            None
            if self._request_planner_observations is None
            else [list(row) for row in self._request_planner_observations]
        )
        original_request_planner_hits = (
            None
            if self._request_planner_hits is None
            else [list(row) for row in self._request_planner_hits]
        )
        original_request_position_updates = self._request_position_updates
        original_max_request_position_maturity = (
            self._max_request_position_maturity
        )
        original_request_lookahead_observations = (
            None
            if self._request_lookahead_observations is None
            else list(self._request_lookahead_observations)
        )
        original_request_lookahead_hits = (
            None
            if self._request_lookahead_hits is None
            else list(self._request_lookahead_hits)
        )
        original_request_lookahead_greedy_hits = (
            None
            if self._request_lookahead_greedy_hits is None
            else list(self._request_lookahead_greedy_hits)
        )
        original_request_lookahead_updates = self._request_lookahead_updates
        original_request_surprise_mean = self._request_surprise_mean
        original_request_surprise_deviation = self._request_surprise_deviation
        original_request_surprise_cusum = self._request_surprise_cusum
        original_request_feedback_count = self._request_feedback_count
        original_request_regime_changes = self._request_regime_changes
        try:
            if (
                self._last_confirmed_length is not None
                and len(committed) < self._last_confirmed_length
            ):
                raise MarkovDraftError("final Markov history moved backwards")
            if self._last_confirmed_length is not None:
                self._advance_confirmed_tokens(
                    committed[self._last_confirmed_length :]
                )
            self._carry_feedback = None
            self._carry_feedback_position = None
            self._carry_feedback_teacher_forced = False
            self._carry_plan = None
            self._recursive_traces.clear()
            self._planner_traces.clear()
            self._request_expert_rapidities = None
            self._request_horizon_observations = None
            self._request_horizon_hits = None
            self._request_plan_observations = None
            self._request_plan_hits = None
            self._request_planner_observations = None
            self._request_planner_hits = None
            self._request_lookahead_observations = None
            self._request_lookahead_hits = None
            self._request_lookahead_greedy_hits = None
            self._request_surprise_mean = None
            self._request_surprise_deviation = None
            self._request_surprise_cusum = None
            self._request_feedback_count = 0
            for (
                feedback,
                token,
                position,
                planned_token,
                greedy_token,
                _lookahead_gain,
            ) in self._episode_feedback:
                self._apply_council_feedback(
                    feedback,
                    token,
                    position,
                    planned_token,
                    greedy_token,
                )
            for (
                predictions,
                token,
                position,
                planned_token,
                greedy_token,
            ) in self._recursive_feedback:
                self._apply_recursive_horizon_feedback(
                    predictions,
                    token,
                    position,
                    planned_token,
                    greedy_token,
                )
            for planner, position, hit in self._planner_feedback:
                self._apply_planner_feedback(planner, position, hit)
            self._episode_feedback.clear()
            self._recursive_feedback.clear()
            self._planner_feedback.clear()
            generated = (
                committed
                if self._request_prompt_length is None
                else committed[self._request_prompt_length :]
            )
            retention_priority = 1.0
            if self.episode_scorer is not None and generated:
                try:
                    retention_priority = float(self.episode_scorer(generated))
                except Exception:
                    self._retention_failures += 1
                    retention_priority = 1.0
                else:
                    if (
                        not math.isfinite(retention_priority)
                        or retention_priority < 0.0
                    ):
                        self._retention_failures += 1
                        retention_priority = 1.0
                    else:
                        self._retention_scored_episodes += 1
            self._last_retention_priority = retention_priority
            self._learn_episode(
                committed,
                prompt_length=self._request_prompt_length,
                priority=retention_priority,
            )
            if self._pending_import_digest is not None:
                self._state = replace(
                    self._state,
                    imported_episode_sha256s=tuple(
                        sorted(
                            {
                                *self._state.imported_episode_sha256s,
                                self._pending_import_digest,
                            }
                        )
                    ),
                )
            self._commit_active_dialect()
            self._last_confirmed_length = len(committed)
            self._persist()
            self._settle_context_crystals(committed)
            self._settle_layer_context_crystals(committed)
            self._request_completed = True
        except Exception:
            self._state = original_state
            self._active_dialect = original_dialect
            self._carry_feedback = original_carry
            self._carry_feedback_position = original_carry_position
            self._carry_feedback_teacher_forced = original_carry_teacher_forced
            self._carry_plan = original_carry_plan
            self._recursive_traces = original_recursive_traces
            self._recursive_trace_created = original_recursive_trace_created
            self._recursive_trace_peak_active = original_recursive_trace_peak_active
            self._recursive_trace_feedback_tokens = (
                original_recursive_trace_feedback_tokens
            )
            self._recursive_trace_hits = original_recursive_trace_hits
            self._recursive_trace_misses = original_recursive_trace_misses
            self._recursive_trace_max_position = (
                original_recursive_trace_max_position
            )
            self._episode_feedback = original_feedback
            self._recursive_feedback = original_recursive_feedback
            self._planner_traces = original_planner_traces
            self._planner_feedback = original_planner_feedback
            self._planner_trace_feedback_tokens = (
                original_planner_trace_feedback_tokens
            )
            self._last_confirmed_length = original_confirmed_length
            self._dialect_evictions = original_evictions
            self._council_feedback = original_council_feedback
            self._external_feedback_tokens = original_external_feedback_tokens
            self._teacher_forced_feedback_tokens = (
                original_teacher_forced_feedback_tokens
            )
            self._retention_scored_episodes = original_retention_scored
            self._retention_failures = original_retention_failures
            self._retention_priority_evictions = original_retention_evictions
            self._last_retention_priority = original_retention_priority
            self._persistent_symbols_cache = original_persistent_symbols_cache
            self._ricci_working_symbols_cache = (
                original_ricci_working_symbols_cache
            )
            self._ricci_episode_cache = original_ricci_episode_cache
            self._ricci_priority_degraded = original_ricci_priority_degraded
            self._episode_priority_cache = original_episode_priority_cache
            self._persistent_expert_models = original_persistent_expert_models
            self._composition_cache = original_composition_cache
            self._composition_program_count = original_composition_program_count
            self._request_expert_rapidities = original_request_rapidities
            self._request_weight_updates = original_request_weight_updates
            self._max_request_weight_shift = original_max_request_weight_shift
            self._request_horizon_observations = (
                original_request_horizon_observations
            )
            self._request_horizon_hits = original_request_horizon_hits
            self._request_plan_observations = original_request_plan_observations
            self._request_plan_hits = original_request_plan_hits
            self._request_planner_observations = (
                original_request_planner_observations
            )
            self._request_planner_hits = original_request_planner_hits
            self._request_position_updates = original_request_position_updates
            self._max_request_position_maturity = (
                original_max_request_position_maturity
            )
            self._request_lookahead_observations = (
                original_request_lookahead_observations
            )
            self._request_lookahead_hits = original_request_lookahead_hits
            self._request_lookahead_greedy_hits = (
                original_request_lookahead_greedy_hits
            )
            self._request_lookahead_updates = original_request_lookahead_updates
            self._request_surprise_mean = original_request_surprise_mean
            self._request_surprise_deviation = original_request_surprise_deviation
            self._request_surprise_cusum = original_request_surprise_cusum
            self._request_feedback_count = original_request_feedback_count
            self._request_regime_changes = original_request_regime_changes
            raise

    def _persist(self) -> None:
        path = self.state_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self._state.to_bytes()
        temporary = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
        descriptor: int | None = None
        try:
            descriptor = os.open(
                temporary,
                os.O_CREAT
                | os.O_EXCL
                | os.O_WRONLY
                | int(getattr(os, "O_NOFOLLOW", 0)),
                0o600,
            )
            offset = 0
            while offset < len(data):
                written = os.write(descriptor, data[offset:])
                if written <= 0:
                    raise OSError("short Markov draft state write")
                offset += written
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.replace(temporary, path)
            directory = os.open(
                path.parent,
                os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0)),
            )
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError as exc:
            raise MarkovDraftError("cannot persist Markov draft state") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)
        self._persisted_state = self._state

    def _persist_if_dirty(self) -> None:
        if self.state_path is not None and self._persisted_state is not self._state:
            self._persist()

    def _layer_context_crystal_metrics_record(
        self,
        bank_metrics: object | None,
    ) -> Mapping[str, object] | None:
        bank = self.layer_contextual_continuation_bank
        if bank is None:
            return None
        counter_names = (
            "crystal_queries",
            "crystal_query_hits",
            "crystal_option_calls",
            "crystal_proposed_tokens",
            "crystal_verified_tokens",
            "crystal_accepted_tokens",
            "crystal_mismatches",
            "crystal_captures",
            "crystal_failures",
        )
        bank_layers = getattr(bank_metrics, "layers", {})
        layers: dict[str, object] = {}
        for layer in bank.identity.layers:
            counters = self._layer_context_crystal_layer_counters.get(
                layer,
                Counter(),
            )
            inventory = (
                None
                if not isinstance(bank_layers, Mapping)
                else bank_layers.get(str(layer))
            )
            layers[str(layer)] = {
                name: int(counters.get(name, 0)) for name in counter_names
            } | {
                "crystal_bank_cells": int(
                    getattr(inventory, "crystal_bank_cells", 0)
                ),
                "crystal_bank_support": int(
                    getattr(inventory, "crystal_bank_support", 0)
                ),
                "crystal_last_cell_sha256": counters.get(
                    "crystal_last_cell_sha256"
                ),
                "crystal_last_cosine": float(
                    counters.get("crystal_last_cosine", 0.0)
                ),
                "crystal_last_margin": float(
                    counters.get("crystal_last_margin", 0.0)
                ),
            }
        record: dict[str, object] = {
            "schema": "immer.qwen3.8-layer-context-crystal-provider-trace/v1",
            "identity_sha256": bank.identity.identity_sha256,
            "layers": layers,
            "request_start_boundary": (
                self._layer_context_crystal_request_start_boundary
            ),
            "last_query": [dict(row) for row in self._layer_context_crystal_last_query],
            "selected": (
                None
                if self._layer_context_crystal_selected is None
                else dict(self._layer_context_crystal_selected)
            ),
            **{
                name: int(self._layer_context_crystal_counters.get(name, 0))
                for name in counter_names
            },
            "crystal_bank_cells": int(
                getattr(bank_metrics, "crystal_bank_cells", 0)
            ),
            "crystal_bank_support": int(
                getattr(bank_metrics, "crystal_bank_support", 0)
            ),
            "crystal_enabled": True,
            "crystal_last_cell_sha256": (
                self._layer_context_crystal_last_cell_sha256
            ),
            "crystal_last_cosine": self._layer_context_crystal_last_cosine,
            "crystal_last_margin": self._layer_context_crystal_last_margin,
        }
        if bank_metrics is not None:
            record["bank"] = {
                "clock": int(getattr(bank_metrics, "clock", 0)),
                "receipt_count": int(getattr(bank_metrics, "receipt_count", 0)),
                "settlements": int(getattr(bank_metrics, "settlements", 0)),
                "state_sha256": getattr(bank_metrics, "state_sha256", None),
            }
        return record

    def metrics(self) -> MarkovDraftMetrics:
        weights = self._weights()
        crystal_metrics = None
        if self.contextual_continuation_bank is not None:
            try:
                crystal_metrics = self.contextual_continuation_bank.metrics()
            except Exception:
                self._crystal_failures += 1
        layer_crystal_metrics = None
        if self.layer_contextual_continuation_bank is not None:
            try:
                layer_crystal_metrics = (
                    self.layer_contextual_continuation_bank.metrics()
                )
            except Exception:
                self._record_layer_context_crystal_failure()
        layer_crystal_record = self._layer_context_crystal_metrics_record(
            layer_crystal_metrics
        )
        dialect_neighbors = self._inference_dialects()
        active_plan_observations, active_plan_hits = (
            ((0,) * _MAX_PROPOSAL_POSITIONS,) * 2
            if self._active_dialect is None
            else self._dialect_plan_counts(self._active_dialect)
        )
        planner_zeros = tuple(
            (0,) * _MAX_PROPOSAL_POSITIONS for _ in _PLANNER_NAMES
        )
        active_planner_observations, active_planner_hits = (
            (planner_zeros, planner_zeros)
            if self._active_dialect is None
            else self._dialect_planner_counts(self._active_dialect)
        )
        accuracy = tuple(
            0.0 if seen == 0 else hit / seen
            for hit, seen in zip(
                self._state.expert_hits,
                self._state.expert_observations,
                strict=True,
            )
        )
        return MarkovDraftMetrics(
            schema=MARKOV_DRAFT_METRICS_SCHEMA,
            draft_calls=self._draft_calls,
            reconcile_calls=self._reconcile_calls,
            predictions=self._predictions,
            learned_tokens=len(self._state.token_ids),
            history_capacity_tokens=self._state.max_history_tokens,
            episode_count=len(self._state.episode_lengths),
            imported_episode_count=len(self._state.imported_episode_sha256s),
            updates=self._state.updates,
            state_bytes=len(self._state.to_bytes()),
            council_predictions=self._council_predictions,
            council_feedback=self._council_feedback,
            external_reconcile_calls=self._external_reconcile_calls,
            external_feedback_tokens=self._external_feedback_tokens,
            teacher_forced_predictions=self._teacher_forced_predictions,
            teacher_forced_feedback_tokens=self._teacher_forced_feedback_tokens,
            teacher_forced_failures=self._teacher_forced_failures,
            leader_changes=self._state.leader_changes,
            expert_weights=tuple(zip(self._state.expert_names, weights, strict=True)),
            expert_accuracy=tuple(zip(self._state.expert_names, accuracy, strict=True)),
            effective_experts=1.0 / sum(value * value for value in weights),
            last_confidence=self._last_confidence,
            last_raw_confidence=self._last_raw_confidence,
            last_empirical_evidence=self._last_empirical_evidence,
            last_disagreement=self._last_disagreement,
            horizon_observations=tuple(
                max(row, default=0)
                for row in self._state.horizon_expert_observations
            ),
            horizon_weighted_accuracy=tuple(
                sum(
                    position_weight * (hit / observed if observed else 0.0)
                    for position_weight, observed, hit in zip(
                        self._position_weighting(position, weights)[0],
                        observed_row,
                        hit_row,
                        strict=True,
                    )
                )
                for position, (observed_row, hit_row) in enumerate(
                    zip(
                        self._state.horizon_expert_observations,
                        self._state.horizon_expert_hits,
                        strict=True,
                    )
                )
            ),
            horizon_self_reliability=tuple(
                self._beam_position_reliability(position)
                for position in range(_MAX_PROPOSAL_POSITIONS)
            ),
            horizon_plan_observations=tuple(
                self._plan_counts(position)[0]
                for position in range(_MAX_PROPOSAL_POSITIONS)
            ),
            horizon_plan_hits=tuple(
                self._plan_counts(position)[1]
                for position in range(_MAX_PROPOSAL_POSITIONS)
            ),
            last_position=self._last_position,
            last_position_maturity=self._last_position_maturity,
            last_dialect_skill_maturity=self._last_dialect_skill_maturity,
            last_position_weights=tuple(
                zip(
                    self._state.expert_names,
                    self._last_position_weights,
                    strict=True,
                )
            ),
            position_specialist_predictions=self._position_specialist_predictions,
            dialect_specialist_predictions=self._dialect_specialist_predictions,
            max_position_maturity=self._max_position_maturity,
            max_dialect_skill_maturity=self._max_dialect_skill_maturity,
            lookahead_calls=self._lookahead_calls,
            lookahead_candidates=self._lookahead_candidates,
            lookahead_token_changes=self._lookahead_token_changes,
            last_lookahead_gain=self._last_lookahead_gain,
            max_lookahead_gain=self._max_lookahead_gain,
            lookahead_outcomes=self._state.lookahead_observations,
            lookahead_hits=self._state.lookahead_hits,
            lookahead_greedy_hits=self._state.lookahead_greedy_hits,
            request_weight_updates=self._request_weight_updates,
            max_request_weight_shift=self._max_request_weight_shift,
            request_position_updates=self._request_position_updates,
            max_request_position_maturity=self._max_request_position_maturity,
            request_lookahead_updates=self._request_lookahead_updates,
            request_regime_changes=self._request_regime_changes,
            request_surprise_mean=(
                self._state.surprise_mean
                if self._request_surprise_mean is None
                else self._request_surprise_mean
            ),
            request_surprise_cusum=(
                self._state.surprise_cusum
                if self._request_surprise_cusum is None
                else self._request_surprise_cusum
            ),
            beam_position_verified=tuple(self._beam_position_verified),
            beam_position_hits=tuple(self._beam_position_hits),
            planner_names=_PLANNER_NAMES,
            planner_observations=tuple(
                tuple(
                    self._planner_counts(planner, position)[0]
                    for position in range(_MAX_PROPOSAL_POSITIONS)
                )
                for planner in range(len(_PLANNER_NAMES))
            ),
            planner_hits=tuple(
                tuple(
                    self._planner_counts(planner, position)[1]
                    for position in range(_MAX_PROPOSAL_POSITIONS)
                )
                for planner in range(len(_PLANNER_NAMES))
            ),
            planner_reliability=tuple(
                tuple(
                    self._planner_reliability(planner, position)
                    for position in range(_MAX_PROPOSAL_POSITIONS)
                )
                for planner in range(len(_PLANNER_NAMES))
            ),
            planner_tournament_calls=self._planner_tournament_calls,
            planner_beam_selections=self._planner_beam_selections,
            planner_council_selections=self._planner_council_selections,
            planner_phrase_selections=self._planner_phrase_selections,
            planner_trace_created=self._planner_trace_created,
            planner_trace_active=len(self._planner_traces),
            planner_trace_feedback_tokens=self._planner_trace_feedback_tokens,
            regime_generation=self._state.regime_generation,
            surprise_mean=self._state.surprise_mean,
            surprise_cusum=self._state.surprise_cusum,
            dialect_count=len(self._state.dialects),
            dialect_neighbor_count=len(dialect_neighbors),
            dialect_neighbor_effective=(
                0.0
                if not dialect_neighbors
                else 1.0 / sum(row[1] * row[1] for row in dialect_neighbors)
            ),
            dialect_neighbor_max_similarity=max(
                (row[0] for row in dialect_neighbors),
                default=0.0,
            ),
            dialect_neighbor_ids=tuple(row[2].dialect_id for row in dialect_neighbors),
            active_dialect_id=(
                None
                if self._active_dialect is None
                else self._active_dialect.dialect_id
            ),
            active_dialect_similarity=self._active_dialect_similarity,
            active_dialect_plan_observations=(
                active_plan_observations
            ),
            active_dialect_plan_hits=active_plan_hits,
            active_dialect_planner_observations=active_planner_observations,
            active_dialect_planner_hits=active_planner_hits,
            dialect_evictions=self._dialect_evictions,
            phrase_option_calls=self._phrase_option_calls,
            phrase_draft_tokens=self._phrase_draft_tokens,
            phrase_accepted_tokens=self._phrase_accepted_tokens,
            periodic_option_calls=self._periodic_option_calls,
            periodic_draft_tokens=self._periodic_draft_tokens,
            periodic_accepted_tokens=self._periodic_accepted_tokens,
            binding_option_calls=self._binding_option_calls,
            binding_draft_tokens=self._binding_draft_tokens,
            binding_accepted_tokens=self._binding_accepted_tokens,
            atlas_contexts=0 if self.atlas is None else self.atlas.context_count,
            atlas_corpus_tokens=0 if self.atlas is None else self.atlas.token_count,
            atlas_option_calls=self._atlas_option_calls,
            atlas_draft_tokens=self._atlas_draft_tokens,
            atlas_accepted_tokens=self._atlas_accepted_tokens,
            atlas_vote_calls=self._atlas_vote_calls,
            atlas_vote_tokens=self._atlas_vote_tokens,
            atlas_vote_supported_tokens=self._atlas_vote_supported_tokens,
            atlas_vote_score_sum=self._atlas_vote_score_sum,
            atlas_vote_max_score=self._atlas_vote_max_score,
            online_vote_calls=self._online_vote_calls,
            online_vote_tokens=self._online_vote_tokens,
            online_vote_supported_tokens=self._online_vote_supported_tokens,
            online_vote_score_sum=self._online_vote_score_sum,
            online_vote_max_score=self._online_vote_max_score,
            retention_scored_episodes=self._retention_scored_episodes,
            retention_failures=self._retention_failures,
            retention_priority_evictions=self._retention_priority_evictions,
            last_retention_priority=self._last_retention_priority,
            ricci_working_set_builds=self._ricci_working_set_builds,
            ricci_working_set_selected_episodes=(
                self._ricci_working_set_selected_episodes
            ),
            ricci_working_set_selected_tokens=(
                self._ricci_working_set_selected_tokens
            ),
            ricci_working_set_oldest_age=self._ricci_working_set_oldest_age,
            ricci_working_set_max_score=self._ricci_working_set_max_score,
            composition_programs=self._composition_program_count,
            composition_option_calls=self._composition_option_calls,
            composition_draft_tokens=self._composition_draft_tokens,
            composition_accepted_tokens=self._composition_accepted_tokens,
            last_composition_support=(
                0
                if self._last_composition_program is None
                else self._last_composition_program.support
            ),
            last_composition_copy_tokens=(
                0
                if self._last_composition_program is None
                else self._last_composition_program.copied_tokens
            ),
            last_phrase_source=(
                None
                if self._last_phrase_option is None
                else self._last_phrase_option.source
            ),
            last_phrase_support=(
                0
                if self._last_phrase_option is None
                else self._last_phrase_option.support
            ),
            last_phrase_confidence=(
                0.0
                if self._last_phrase_option is None
                else self._last_phrase_option.confidence
            ),
            last_phrase_width=(
                0
                if self._last_phrase_option is None
                else len(self._last_phrase_option.token_ids)
            ),
            adaptive_proposal_calls=self._adaptive_proposal_calls,
            recommended_windows=tuple(
                (window, self._recommended_window_counts[window])
                for window in (1, 4, 8, 16)
            ),
            last_recommended_window=(
                None
                if self._last_round_proposal is None
                else self._last_round_proposal.recommended_window
            ),
            last_horizon_utilities=(
                ()
                if self._last_round_proposal is None
                else tuple(
                    (row.window, row.utility)
                    for row in self._last_round_proposal.horizons
                )
            ),
            proposal_width=self.proposal_width,
            recursive_trace_created=self._recursive_trace_created,
            recursive_trace_active=len(self._recursive_traces),
            recursive_trace_peak_active=self._recursive_trace_peak_active,
            recursive_trace_feedback_tokens=(
                self._recursive_trace_feedback_tokens
            ),
            recursive_trace_hits=self._recursive_trace_hits,
            recursive_trace_misses=self._recursive_trace_misses,
            recursive_trace_max_position=self._recursive_trace_max_position,
            crystal_enabled=(
                self.contextual_continuation_bank is not None
                or self.layer_contextual_continuation_bank is not None
            ),
            crystal_queries=(
                self._crystal_queries
                + int(self._layer_context_crystal_counters["crystal_queries"])
            ),
            crystal_query_hits=(
                self._crystal_query_hits
                + int(
                    self._layer_context_crystal_counters[
                        "crystal_query_hits"
                    ]
                )
            ),
            crystal_option_calls=(
                self._crystal_option_calls
                + int(
                    self._layer_context_crystal_counters[
                        "crystal_option_calls"
                    ]
                )
            ),
            crystal_proposed_tokens=(
                self._crystal_proposed_tokens
                + int(
                    self._layer_context_crystal_counters[
                        "crystal_proposed_tokens"
                    ]
                )
            ),
            crystal_verified_tokens=(
                self._crystal_verified_tokens
                + int(
                    self._layer_context_crystal_counters[
                        "crystal_verified_tokens"
                    ]
                )
            ),
            crystal_accepted_tokens=(
                self._crystal_accepted_tokens
                + int(
                    self._layer_context_crystal_counters[
                        "crystal_accepted_tokens"
                    ]
                )
            ),
            crystal_mismatches=(
                self._crystal_mismatches
                + int(
                    self._layer_context_crystal_counters[
                        "crystal_mismatches"
                    ]
                )
            ),
            crystal_captures=(
                self._crystal_captures
                + int(self._layer_context_crystal_counters["crystal_captures"])
            ),
            crystal_failures=(
                self._crystal_failures
                + int(self._layer_context_crystal_counters["crystal_failures"])
            ),
            crystal_bank_cells=(
                (0 if crystal_metrics is None else crystal_metrics.cell_count)
                + (
                    0
                    if layer_crystal_metrics is None
                    else layer_crystal_metrics.crystal_bank_cells
                )
            ),
            crystal_bank_support=(
                (0 if crystal_metrics is None else crystal_metrics.support)
                + (
                    0
                    if layer_crystal_metrics is None
                    else layer_crystal_metrics.crystal_bank_support
                )
            ),
            crystal_last_cosine=(
                self._crystal_last_cosine
                if self._layer_context_crystal_last_cell_sha256 is None
                else self._layer_context_crystal_last_cosine
            ),
            crystal_last_margin=(
                self._crystal_last_margin
                if self._layer_context_crystal_last_cell_sha256 is None
                else self._layer_context_crystal_last_margin
            ),
            crystal_last_cell_sha256=(
                self._crystal_last_cell_sha256
                if self._layer_context_crystal_last_cell_sha256 is None
                else self._layer_context_crystal_last_cell_sha256
            ),
            layer_context_crystal=layer_crystal_record,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._pending_base = None
        self._pending_proposal = None
        self._pending_complete = ()
        self._pending_planner_candidates = ()
        self._pending_feedback = ()
        self._pending_plan_trace = ()
        self._pending_planner = None
        self._pending_accepted_prefix_length = None
        self._pending_verified_proposals = None
        self._pending_verification_virtual = False
        self._carry_feedback = None
        self._carry_feedback_position = None
        self._carry_feedback_teacher_forced = False
        self._carry_plan = None
        self._recursive_traces.clear()
        self._planner_traces.clear()
        self._pending_phrase_option = None
        self._context_crystal_key = None
        self._context_crystal_candidates = ()
        self._context_crystal_captures.clear()
        self._context_crystal_feedback.clear()
        self._layer_context_crystal_phrase_options = ()
        self._layer_context_crystal_options_by_transaction.clear()
        self._layer_context_crystal_transactions.clear()
        self._layer_context_crystal_request_enabled = False
        self._pending_composition_program = None
        self._pending_import_digest = None
        self._request_expert_rapidities = None
        self._request_horizon_observations = None
        self._request_horizon_hits = None
        self._request_plan_observations = None
        self._request_plan_hits = None
        self._request_planner_observations = None
        self._request_planner_hits = None
        self._request_lookahead_observations = None
        self._request_lookahead_hits = None
        self._request_lookahead_greedy_hits = None
        self._request_surprise_mean = None
        self._request_surprise_deviation = None
        self._request_surprise_cusum = None
        self._request_feedback_count = 0
        self._request_local_cache_history = None
        self._request_local_cache = None
        self._episode_feedback.clear()
        self._recursive_feedback.clear()
        self._planner_feedback.clear()
        try:
            self._persist_if_dirty()
        finally:
            self._closed = True
            self._release_state_lock()


__all__ = [
    "LEGACY_MARKOV_DRAFT_STATE_SCHEMA",
    "V2_MARKOV_DRAFT_STATE_SCHEMA",
    "V3_MARKOV_DRAFT_STATE_SCHEMA",
    "V4_MARKOV_DRAFT_STATE_SCHEMA",
    "V5_MARKOV_DRAFT_STATE_SCHEMA",
    "V6_MARKOV_DRAFT_STATE_SCHEMA",
    "V7_MARKOV_DRAFT_STATE_SCHEMA",
    "V8_MARKOV_DRAFT_STATE_SCHEMA",
    "V9_MARKOV_DRAFT_STATE_SCHEMA",
    "V10_MARKOV_DRAFT_STATE_SCHEMA",
    "V11_MARKOV_DRAFT_STATE_SCHEMA",
    "V12_MARKOV_DRAFT_STATE_SCHEMA",
    "MARKOV_DRAFT_METRICS_SCHEMA",
    "MARKOV_DRAFT_PROVIDER_ABI",
    "MARKOV_DRAFT_STATE_SCHEMA",
    "MARKOV_RICCI_WORKING_SET_POLICY",
    "FingerprintRollingK4DraftProvider",
    "MarkovDraftError",
    "MarkovDialectState",
    "MarkovDraftMetrics",
    "MarkovDraftState",
    "MarkovExpertSpec",
    "MarkovLanguageTokenEvidence",
    "MarkovPhraseOption",
]
