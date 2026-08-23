"""Auditable expert-transition evidence for exact DeepSeek-V4 prefetching.

This module learns only an empirical ordering hint.  The checkpoint router
remains authoritative: predictions neither fetch tensors nor permit an expert,
layer, or model computation to be skipped.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from immer.knowledge.livecausal import LiveGraph, segment_sha


_PINNED_REVISION = re.compile(r"[0-9a-f]{40,64}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SCHEMA = "deepseek-v4-expert-transition-v1"
_HISTORY_LENGTH = 2


class CausalPrefetchError(ValueError):
    """Transition evidence is invalid or inconsistent."""


class CheckpointMismatchError(CausalPrefetchError):
    """A route or stored record belongs to another checkpoint."""


def _canonical_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _nonempty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalPrefetchError(f"{label} must be non-empty text")
    normalized = value.strip()
    if len(normalized) > 512:
        raise CausalPrefetchError(f"{label} exceeds 512 characters")
    return normalized


def _digest(value: object, label: str, *, length: int = 64) -> str:
    text = _nonempty_text(value, label).lower()
    pattern = _SHA256 if length == 64 else _PINNED_REVISION
    if pattern.fullmatch(text) is None:
        if length == 64:
            raise CausalPrefetchError(f"{label} must be a lowercase SHA-256 digest")
        raise CausalPrefetchError(
            f"{label} must be a pinned 40-64 character hexadecimal revision"
        )
    return text


def _layer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CausalPrefetchError("layer must be a non-negative integer")
    return value


def _expert_id(value: object, *, n_routed_experts: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CausalPrefetchError("expert IDs must be non-negative integers")
    if n_routed_experts is not None and value >= n_routed_experts:
        raise CausalPrefetchError(
            f"expert ID {value} is outside the checkpoint expert inventory"
        )
    return value


def _normalize_provenance(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise CausalPrefetchError("provenance must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise CausalPrefetchError("provenance keys must be strings")
    try:
        encoded = json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        normalized = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise CausalPrefetchError(
            "provenance must contain canonical JSON values"
        ) from exc
    if not isinstance(normalized, dict):  # pragma: no cover - guarded above.
        raise CausalPrefetchError("provenance must be a JSON object")
    return normalized


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    """Immutable identity of the tensor inventory that produced route traces."""

    repo_id: str
    revision: str
    inventory_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo_id", _nonempty_text(self.repo_id, "repo_id"))
        object.__setattr__(
            self,
            "revision",
            _digest(self.revision, "revision", length=40),
        )
        object.__setattr__(
            self,
            "inventory_fingerprint",
            _digest(self.inventory_fingerprint, "inventory_fingerprint"),
        )

    @property
    def key(self) -> str:
        return f"checkpoint:v1:{_canonical_digest(self.as_record())}"

    def as_record(self) -> dict[str, str]:
        return {
            "inventory_fingerprint": self.inventory_fingerprint,
            "repo_id": self.repo_id,
            "revision": self.revision,
        }


@dataclass(frozen=True, slots=True)
class RouteState:
    """Canonical multiset of official router selections at one layer."""

    checkpoint: CheckpointIdentity
    layer: int
    expert_counts: tuple[tuple[int, int], ...]
    prompt_feature_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, CheckpointIdentity):
            raise CausalPrefetchError("checkpoint must be a CheckpointIdentity")
        object.__setattr__(self, "layer", _layer(self.layer))
        prior = -1
        normalized: list[tuple[int, int]] = []
        if not isinstance(self.expert_counts, tuple) or not self.expert_counts:
            raise CausalPrefetchError(
                "expert_counts must be a non-empty canonical tuple"
            )
        for pair in self.expert_counts:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise CausalPrefetchError(
                    "expert_counts entries must be (expert_id, count)"
                )
            expert = _expert_id(pair[0])
            count = pair[1]
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise CausalPrefetchError(
                    "expert selection counts must be positive integers"
                )
            if expert <= prior:
                raise CausalPrefetchError(
                    "expert_counts must contain unique expert IDs in ascending order"
                )
            prior = expert
            normalized.append((expert, count))
        object.__setattr__(self, "expert_counts", tuple(normalized))
        if self.prompt_feature_digest is not None:
            object.__setattr__(
                self,
                "prompt_feature_digest",
                _digest(self.prompt_feature_digest, "prompt_feature_digest"),
            )

    @classmethod
    def from_selected(
        cls,
        checkpoint: CheckpointIdentity,
        *,
        layer: int,
        selected_expert_ids: Iterable[int],
        prompt_feature_digest: str | None = None,
        n_routed_experts: int = 256,
    ) -> RouteState:
        if (
            isinstance(n_routed_experts, bool)
            or not isinstance(n_routed_experts, int)
            or n_routed_experts < 1
        ):
            raise CausalPrefetchError("n_routed_experts must be a positive integer")
        try:
            selected = tuple(selected_expert_ids)
        except TypeError as exc:
            raise CausalPrefetchError("selected_expert_ids must be iterable") from exc
        if not selected:
            raise CausalPrefetchError("selected_expert_ids must not be empty")
        counts = Counter(
            _expert_id(expert, n_routed_experts=n_routed_experts) for expert in selected
        )
        return cls(
            checkpoint=checkpoint,
            layer=layer,
            expert_counts=tuple(sorted(counts.items())),
            prompt_feature_digest=prompt_feature_digest,
        )

    @property
    def key(self) -> str:
        return f"route:v1:{_canonical_digest(self.as_record())}"

    def as_record(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint.as_record(),
            "expert_counts": [list(pair) for pair in self.expert_counts],
            "layer": self.layer,
            "prompt_feature_digest": self.prompt_feature_digest,
        }


@dataclass(frozen=True, slots=True)
class ExpertSupport:
    """One empirical expert score; ``support / total`` is the only confidence."""

    expert_id: int
    support: int
    total: int
    cost: int

    @property
    def empirical_rate(self) -> float:
        return self.support / self.total


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Top-k hints plus the full empirical distribution for adaptive scheduling."""

    source_state_key: str
    target_layer: int
    candidates: tuple[ExpertSupport, ...]
    distribution: tuple[ExpertSupport, ...]
    total: int


@dataclass(frozen=True, slots=True)
class ObservationReceipt:
    """Stable identity of one append or idempotent replay."""

    observation_id: str
    segment_sha256: str
    appended: bool


class CausalExpertTransitionController:
    """Checkpoint-bound transition ledger plus a fixed two-route history."""

    def __init__(
        self,
        graph: LiveGraph,
        checkpoint: CheckpointIdentity,
        *,
        n_routed_experts: int = 256,
    ) -> None:
        if not isinstance(graph, LiveGraph):
            raise TypeError("graph must be a LiveGraph")
        if not isinstance(checkpoint, CheckpointIdentity):
            raise TypeError("checkpoint must be a CheckpointIdentity")
        if (
            isinstance(n_routed_experts, bool)
            or not isinstance(n_routed_experts, int)
            or n_routed_experts < 1
        ):
            raise CausalPrefetchError("n_routed_experts must be a positive integer")
        self.graph = graph
        self.checkpoint = checkpoint
        self.n_routed_experts = n_routed_experts
        self._history: deque[RouteState] = deque(maxlen=_HISTORY_LENGTH)

    @property
    def history(self) -> tuple[RouteState, ...]:
        return tuple(self._history)

    def clear_history(self) -> None:
        self._history.clear()

    def route_state(
        self,
        *,
        layer: int,
        selected_expert_ids: Iterable[int],
        prompt_feature_digest: str | None = None,
    ) -> RouteState:
        return RouteState.from_selected(
            self.checkpoint,
            layer=layer,
            selected_expert_ids=selected_expert_ids,
            prompt_feature_digest=prompt_feature_digest,
            n_routed_experts=self.n_routed_experts,
        )

    def _validate_state(self, state: RouteState) -> None:
        if not isinstance(state, RouteState):
            raise CausalPrefetchError("route must be a RouteState")
        if state.checkpoint != self.checkpoint:
            raise CheckpointMismatchError(
                "route checkpoint identity does not match this controller"
            )
        for expert, _count in state.expert_counts:
            _expert_id(expert, n_routed_experts=self.n_routed_experts)

    def _expert_outcome_key(self, *, layer: int, expert_id: int) -> str:
        identity = {
            "checkpoint": self.checkpoint.as_record(),
            "expert_id": expert_id,
            "layer": layer,
        }
        return f"expert:v1:{_canonical_digest(identity)}"

    def _observation_keys(self, observation_id: str) -> tuple[str, str]:
        digest = _canonical_digest(
            {
                "checkpoint": self.checkpoint.as_record(),
                "observation_id": observation_id,
            }
        )
        return f"observation-index:v1:{digest}", f"observation:v1:{digest}"

    def _records(
        self,
        source: RouteState,
        target: RouteState,
        *,
        observation_id: str,
        provenance: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], ...]:
        observation_key, observation_outcome = self._observation_keys(observation_id)
        normalized_provenance = _normalize_provenance(provenance)
        common = {
            "checkpoint": self.checkpoint.as_record(),
            "observation_id": observation_id,
            "provenance": normalized_provenance,
            "schema": _SCHEMA,
            "source_state": source.as_record(),
            "source_state_key": source.key,
            "target_layer": target.layer,
        }
        records: list[dict[str, Any]] = [
            {
                **common,
                "expert_counts": [list(pair) for pair in target.expert_counts],
                "outcome_key": observation_outcome,
                "record_type": "observation",
                "target_state": target.as_record(),
                "target_state_key": target.key,
                "trigger_key": observation_key,
            }
        ]
        for expert_id, count in target.expert_counts:
            records.append(
                {
                    **common,
                    "count": count,
                    "outcome_key": self._expert_outcome_key(
                        layer=target.layer,
                        expert_id=expert_id,
                    ),
                    "record_type": "expert_transition",
                    "target_expert_id": expert_id,
                    "trigger_key": source.key,
                }
            )
        return tuple(records)

    def observe_transition(
        self,
        source: RouteState,
        target: RouteState,
        *,
        observation_id: str,
        provenance: Mapping[str, Any] | None = None,
    ) -> ObservationReceipt:
        """Append one aggregate transition, or replay it without mutation."""

        self._validate_state(source)
        self._validate_state(target)
        if target.layer != source.layer + 1:
            raise CausalPrefetchError("transition layers must be consecutive")
        if target.prompt_feature_digest != source.prompt_feature_digest:
            raise CausalPrefetchError(
                "source and target prompt feature digests must match"
            )
        normalized_id = _nonempty_text(observation_id, "observation_id")
        records = self._records(
            source,
            target,
            observation_id=normalized_id,
            provenance=provenance,
        )
        expected_sha = segment_sha(records)
        observation_key, observation_outcome = self._observation_keys(normalized_id)
        citations = self.graph.base_edge_citations(observation_key, observation_outcome)
        if citations:
            if citations != [[expected_sha, 0]]:
                raise CausalPrefetchError(
                    "observation_id is already bound to different transition evidence"
                )
            resolved = self.graph.resolve_derivation(citations)
            if resolved != [records[0]]:
                raise CausalPrefetchError(
                    "observation index does not resolve to its canonical record"
                )
            return ObservationReceipt(normalized_id, expected_sha, False)

        actual_sha = self.graph.append_segment(records)
        if actual_sha != expected_sha:  # pragma: no cover - LiveGraph contract.
            raise CausalPrefetchError("LiveGraph returned a non-canonical segment SHA")
        return ObservationReceipt(normalized_id, actual_sha, True)

    def predict(
        self,
        source: RouteState,
        *,
        top_k: int,
        expert_costs: Mapping[int, int] | None = None,
    ) -> PredictionResult:
        """Rank every direct edge without entering bounded multi-hop inference."""

        self._validate_state(source)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise CausalPrefetchError("top_k must be a positive integer")
        costs: dict[int, int] = {}
        if expert_costs is not None:
            if not isinstance(expert_costs, Mapping):
                raise CausalPrefetchError("expert_costs must be a mapping")
            for raw_expert, raw_cost in expert_costs.items():
                expert = _expert_id(raw_expert, n_routed_experts=self.n_routed_experts)
                if (
                    isinstance(raw_cost, bool)
                    or not isinstance(raw_cost, int)
                    or raw_cost < 0
                ):
                    raise CausalPrefetchError(
                        "expert costs must be non-negative integer units"
                    )
                costs[expert] = raw_cost

        edges = self.graph.query_base(source.key)

        support: Counter[int] = Counter()
        for edge in edges:
            if edge.get("kind") != "base":
                continue
            records = self.graph.resolve_derivation(edge.get("derivation", ()))
            for record in records:
                if (
                    record.get("schema") != _SCHEMA
                    or record.get("record_type") != "expert_transition"
                ):
                    raise CausalPrefetchError(
                        "route key resolved to an incompatible causal record"
                    )
                if record.get("checkpoint") != self.checkpoint.as_record():
                    raise CheckpointMismatchError(
                        "stored transition belongs to another checkpoint"
                    )
                if record.get("source_state_key") != source.key:
                    raise CausalPrefetchError("stored transition source key is invalid")
                if record.get("source_state") != source.as_record():
                    raise CausalPrefetchError(
                        "stored transition source state is invalid"
                    )
                if record.get("target_layer") != source.layer + 1:
                    raise CausalPrefetchError(
                        "stored transition target layer is invalid"
                    )
                expert = _expert_id(
                    record.get("target_expert_id"),
                    n_routed_experts=self.n_routed_experts,
                )
                count = record.get("count")
                if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                    raise CausalPrefetchError("stored transition count is invalid")
                if edge.get("to_key") != self._expert_outcome_key(
                    layer=source.layer + 1, expert_id=expert
                ):
                    raise CausalPrefetchError(
                        "stored transition outcome key is invalid"
                    )
                support[expert] += count

        total = sum(support.values())
        ranked = sorted(
            support,
            key=lambda expert: (-support[expert], costs.get(expert, 0), expert),
        )
        distribution = tuple(
            ExpertSupport(
                expert_id=expert,
                support=support[expert],
                total=total,
                cost=costs.get(expert, 0),
            )
            for expert in ranked
        )
        return PredictionResult(
            source_state_key=source.key,
            target_layer=source.layer + 1,
            candidates=distribution[:top_k],
            distribution=distribution,
            total=total,
        )

    def push_route(
        self,
        state: RouteState,
        *,
        observation_id: str | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> ObservationReceipt | None:
        """Record an adjacent history transition and retain at most two routes."""

        self._validate_state(state)
        previous = self._history[-1] if self._history else None
        if previous is None:
            if observation_id is not None:
                raise CausalPrefetchError(
                    "observation_id has no preceding route to identify"
                )
            self._history.append(state)
            return None
        if state.layer != previous.layer + 1:
            if observation_id is not None:
                raise CausalPrefetchError(
                    "observation_id cannot label a non-consecutive route"
                )
            self._history.clear()
            self._history.append(state)
            return None
        if observation_id is None:
            raise CausalPrefetchError(
                "a consecutive route requires an explicit observation_id"
            )
        receipt = self.observe_transition(
            previous,
            state,
            observation_id=observation_id,
            provenance=provenance,
        )
        self._history.append(state)
        return receipt


__all__ = [
    "CausalExpertTransitionController",
    "CausalPrefetchError",
    "CheckpointIdentity",
    "CheckpointMismatchError",
    "ExpertSupport",
    "ObservationReceipt",
    "PredictionResult",
    "RouteState",
]
