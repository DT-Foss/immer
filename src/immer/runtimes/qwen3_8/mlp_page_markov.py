"""Persistent Markov agents that turn exact MLP page traces into execution routes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
import threading
from typing import Any


MLP_PAGE_MARKOV_SCHEMA = "immer.qwen3.8-mlp-page-markov/v4"
MLP_PAGE_MARKOV_POLICY = "dynamic-page-transitions+coactivation+fixed-share/v4"
_V3_MLP_PAGE_MARKOV_SCHEMA = "immer.qwen3.8-mlp-page-markov/v3"
_V3_MLP_PAGE_MARKOV_POLICY = "shared-page-transitions+fixed-share/v3"
_V3_AGENTS = ("temporal", "cross_layer", "marginal")
_AGENTS = ("temporal", "cross_layer", "coactive", "marginal")
_V4_METRICS = frozenset(
    {
        "coactive_updates",
        "dynamic_route_calls",
        "dynamic_route_changes",
    }
)
_MAX_STATE_BYTES = 16 * 1024 * 1024
_CALL_PREFETCH = object()


class MlpPageMarkovError(RuntimeError):
    """The persistent MLP page controller cannot be used safely."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _stable_read(path: Path) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | int(getattr(os, "O_CLOEXEC", 0))
            | int(getattr(os, "O_NOFOLLOW", 0)),
        )
        opened = os.fstat(descriptor)
        linked = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
            or not 0 < opened.st_size <= _MAX_STATE_BYTES
        ):
            raise MlpPageMarkovError("MLP page state file is invalid")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise MlpPageMarkovError("MLP page state was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise MlpPageMarkovError("MLP page state grew while reading")
        return b"".join(chunks)
    except MlpPageMarkovError:
        raise
    except OSError as exc:
        raise MlpPageMarkovError("cannot read MLP page state") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _space_saving_increment(
    counter: Counter[int],
    value: int,
    *,
    limit: int,
) -> int:
    if value in counter or len(counter) < limit:
        counter[value] += 1
        return 0
    victim = min(counter, key=lambda page: (counter[page], -page))
    inherited = counter.pop(victim)
    counter[value] = inherited + 1
    return 1


def _winner(counter: Counter[int] | None) -> int | None:
    if not counter:
        return None
    return max(counter, key=lambda page: (counter[page], -page))


@dataclass(frozen=True, slots=True)
class MlpPagePrediction:
    layer: int
    page_ids: tuple[int, ...]
    ready: bool
    agent_page_ids: tuple[tuple[str, tuple[int | None, ...]], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_page_ids": {
                name: list(values) for name, values in self.agent_page_ids
            },
            "layer": self.layer,
            "page_ids": list(self.page_ids),
            "ready": self.ready,
        }


class MlpPageMarkov:
    """Online route controller trained only by exact full-Q4 activation pages."""

    LEARNING_RATE = 0.5
    FIXED_SHARE = 0.05
    MAX_TARGETS_PER_TRANSITION = 8
    COACTIVE_NEIGHBOR_SPAN = 4

    def __init__(
        self,
        path: str | Path | None,
        *,
        n_layers: int,
        page_count: int,
        route_width: int,
        identity: Mapping[str, object] | None = None,
        min_exact_rows: int = 2,
        prefetch: Callable[[int, tuple[int, ...]], bool] | None = None,
    ) -> None:
        for value, label in (
            (n_layers, "n_layers"),
            (page_count, "page_count"),
            (route_width, "route_width"),
            (min_exact_rows, "min_exact_rows"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if route_width > page_count:
            raise ValueError("route_width exceeds page_count")
        if prefetch is not None and not callable(prefetch):
            raise TypeError("prefetch must be callable or None")
        try:
            clean_identity = json.loads(
                _canonical({} if identity is None else dict(identity)).decode("ascii")
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("identity must be canonical JSON") from exc
        if not isinstance(clean_identity, dict):
            raise ValueError("identity must be a mapping")
        self.path = None if path is None else Path(path).expanduser().absolute()
        self.n_layers = n_layers
        self.page_count = page_count
        self.route_width = route_width
        self.min_exact_rows = min_exact_rows
        self.identity = clean_identity
        self.prefetch = prefetch
        self._marginal: dict[tuple[int, int], Counter[int]] = {}
        self._temporal: dict[tuple[int, int], Counter[int]] = {}
        self._cross: dict[tuple[int, int], Counter[int]] = {}
        self._coactive: dict[tuple[int, int], Counter[int]] = {}
        self._agent_logs = [[0.0] * len(_AGENTS) for _ in range(n_layers)]
        self._agent_observations = [[0] * len(_AGENTS) for _ in range(n_layers)]
        self._agent_hits = [[0] * len(_AGENTS) for _ in range(n_layers)]
        self._exact_support = [0] * n_layers
        self._last_routes: list[tuple[int, ...] | None] = [None] * n_layers
        self._wave_routes: dict[int, tuple[tuple[int, ...], ...]] = {}
        self._pending: dict[int, MlpPagePrediction] = {}
        self._pending_prefetch: dict[int, bool | None] = {}
        self._pending_dynamic: dict[int, bool] = {}
        self._compiled: list[MlpPagePrediction | None] = [None] * n_layers
        self._dirty = False
        self._closed = False
        self._transaction: dict[str, Any] | None = None
        self._lock = threading.RLock()
        self._metrics = {
            "agent_feedback": 0,
            "coactive_updates": 0,
            "counter_evictions": 0,
            "dynamic_route_calls": 0,
            "dynamic_route_changes": 0,
            "exact_batches": 0,
            "exact_rows": 0,
            "fallback_predictions": 0,
            "position_hits": 0,
            "prediction_calls": 0,
            "predicted_pages": 0,
            "prefetch_calls": 0,
            "prefetch_pages": 0,
            "prefetch_successes": 0,
            "ready_predictions": 0,
            "route_jaccard_count": 0,
            "route_jaccard_sum_ppm": 0,
            "selected_advances": 0,
            "session_resets": 0,
        }
        self._load()

    def _ensure_open(self) -> None:
        if self._closed:
            raise MlpPageMarkovError("MLP page controller is closed")

    def begin_transaction(self) -> None:
        """Stage route mutations until continuation commit or rollback."""

        with self._lock:
            self._ensure_open()
            if self._transaction is not None:
                raise MlpPageMarkovError("MLP page transaction is already active")
            self._transaction = {
                "agent_hits": [list(row) for row in self._agent_hits],
                "agent_logs": [list(row) for row in self._agent_logs],
                "agent_observations": [
                    list(row) for row in self._agent_observations
                ],
                "counters": None,
                "compiled": list(self._compiled),
                "current_wave": -1,
                "dirty": self._dirty,
                "events": [],
                "exact_support": list(self._exact_support),
                "last_routes": list(self._last_routes),
                "metrics": dict(self._metrics),
                "pending": dict(self._pending),
                "pending_prefetch": dict(self._pending_prefetch),
                "pending_dynamic": dict(self._pending_dynamic),
                "wave_routes": dict(self._wave_routes),
                "wave_rows": [],
            }

    def _stage_counter_snapshot(self) -> None:
        transaction = self._transaction
        if transaction is None or transaction["counters"] is not None:
            return
        transaction["counters"] = (
            {key: Counter(value) for key, value in self._marginal.items()},
            {key: Counter(value) for key, value in self._temporal.items()},
            {key: Counter(value) for key, value in self._cross.items()},
            {key: Counter(value) for key, value in self._coactive.items()},
        )

    def commit_transaction(self, *, accepted_rows: int | None = None) -> None:
        with self._lock:
            self._ensure_open()
            transaction = self._transaction
            if transaction is None:
                raise MlpPageMarkovError("no MLP page transaction is active")
            if accepted_rows is None:
                self._transaction = None
                return
            if (
                isinstance(accepted_rows, bool)
                or not isinstance(accepted_rows, int)
                or accepted_rows <= 0
            ):
                raise ValueError("accepted_rows must be a positive integer")
            wave_rows = transaction["wave_rows"]
            if (
                not wave_rows
                or any(row is None for row in wave_rows)
                or accepted_rows > sum(wave_rows)
            ):
                raise MlpPageMarkovError(
                    "accepted rows differ from the staged MLP page waves"
                )
            if accepted_rows == sum(wave_rows):
                self._transaction = None
                return
            events = tuple(transaction["events"])
            offsets: list[int] = []
            offset = 0
            for row_count in wave_rows:
                offsets.append(offset)
                offset += row_count
            self.rollback_transaction()
            self.begin_transaction()
            replayed_exact = False
            try:
                for kind, wave, layer, payload, scores, prefetch_result in events:
                    keep = min(
                        wave_rows[wave],
                        max(0, accepted_rows - offsets[wave]),
                    )
                    if keep <= 0:
                        continue
                    if kind == "exact":
                        replayed_exact = True
                        self.begin_exact_wave(layer)
                        self.observe_exact_batch(
                            layer,
                            payload[:keep],
                            None if scores is None else scores[:keep],
                        )
                    elif kind == "selected_route":
                        self._route(
                            layer,
                            row_count=keep,
                            replay_prefetch=prefetch_result,
                        )
                        self.advance_selected(layer, payload, row_count=keep)
                    elif kind == "selected_prepare":
                        self._prepare(
                            layer,
                            replay_prefetch=prefetch_result,
                        )
                        self.advance_selected(layer, payload, row_count=keep)
                    else:
                        raise MlpPageMarkovError(
                            "MLP page transaction event is invalid"
                        )
                if replayed_exact:
                    self.compile_routes()
                self._transaction = None
            except Exception:
                self.rollback_transaction()
                raise

    def rollback_transaction(self) -> None:
        with self._lock:
            self._ensure_open()
            transaction = self._transaction
            if transaction is None:
                return
            self._agent_hits = transaction["agent_hits"]
            self._agent_logs = transaction["agent_logs"]
            self._agent_observations = transaction["agent_observations"]
            counters = transaction["counters"]
            if counters is not None:
                (
                    self._marginal,
                    self._temporal,
                    self._cross,
                    self._coactive,
                ) = counters
            self._dirty = transaction["dirty"]
            self._compiled = transaction["compiled"]
            self._exact_support = transaction["exact_support"]
            self._last_routes = transaction["last_routes"]
            self._metrics = transaction["metrics"]
            self._pending = transaction["pending"]
            self._pending_prefetch = transaction["pending_prefetch"]
            self._pending_dynamic = transaction["pending_dynamic"]
            self._wave_routes = transaction["wave_routes"]
            self._transaction = None

    def _validate_layer(self, layer: int) -> None:
        if (
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or not 0 <= layer < self.n_layers
        ):
            raise ValueError("layer is outside the page controller")

    def _validate_route(self, values: Sequence[int]) -> tuple[int, ...]:
        route = tuple(values)
        if (
            len(route) != self.route_width
            or len(set(route)) != len(route)
            or any(
                isinstance(page, bool)
                or not isinstance(page, int)
                or not 0 <= page < self.page_count
                for page in route
            )
        ):
            raise ValueError("MLP page route is invalid")
        return route

    def _weights(self, layer: int) -> tuple[float, ...]:
        logs = self._agent_logs[layer]
        maximum = max(logs)
        raw = tuple(math.exp(value - maximum) for value in logs)
        total = sum(raw)
        count = len(raw)
        return tuple(
            (1.0 - self.FIXED_SHARE) * value / total
            + self.FIXED_SHARE / count
            for value in raw
        )

    def _coactive_route(self, layer: int) -> tuple[int | None, ...]:
        """Walk the learned within-route page graph from the strongest anchor."""

        anchor = _winner(self._marginal.get((layer, 0)))
        if anchor is None:
            return (None,) * self.route_width
        selected = [anchor]
        selected_set = {anchor}
        while len(selected) < self.route_width:
            direct = self._coactive.get((layer, selected[-1]))
            candidates = Counter(
                {
                    page: count
                    for page, count in (() if direct is None else direct.items())
                    if page not in selected_set
                }
            )
            if not candidates:
                for source in selected:
                    counter = self._coactive.get((layer, source))
                    if counter is None:
                        continue
                    scale = max(1, sum(counter.values()))
                    for page, count in counter.items():
                        if page not in selected_set:
                            candidates[page] += count / scale
            if not candidates:
                break
            winner = max(candidates, key=lambda page: (candidates[page], -page))
            selected.append(winner)
            selected_set.add(winner)
        return tuple((*selected, *((None,) * (self.route_width - len(selected)))))

    def _agent_routes(
        self,
        layer: int,
        temporal_source: tuple[int, ...] | None,
        cross_source: tuple[int, ...] | None,
    ) -> tuple[tuple[str, tuple[int | None, ...]], ...]:
        rows: list[tuple[str, tuple[int | None, ...]]] = []
        for agent in _AGENTS:
            if agent == "coactive":
                rows.append((agent, self._coactive_route(layer)))
                continue
            route: list[int | None] = []
            for rank in range(self.route_width):
                if agent == "temporal" and temporal_source is not None:
                    counter = self._temporal.get((layer, temporal_source[rank]))
                elif agent == "cross_layer" and cross_source is not None:
                    counter = self._cross.get((layer, cross_source[rank]))
                elif agent == "marginal":
                    counter = self._marginal.get((layer, rank))
                else:
                    counter = None
                route.append(_winner(counter))
            rows.append((agent, tuple(route)))
        return tuple(rows)

    def _predict(self, layer: int) -> MlpPagePrediction:
        cross_rows = self._wave_routes.get(layer - 1, ())
        cross = cross_rows[-1] if cross_rows else None
        agents = self._agent_routes(layer, self._last_routes[layer], cross)
        pages = self._combine(layer, agents)
        ready = (
            len(pages) == self.route_width
            and self._exact_support[layer] >= self.min_exact_rows
        )
        return MlpPagePrediction(layer, pages, ready, agents)

    def _combine(
        self,
        layer: int,
        agents: tuple[tuple[str, tuple[int | None, ...]], ...],
    ) -> tuple[int, ...]:
        weights = self._weights(layer)
        selected: list[int] = []
        for rank in range(self.route_width):
            votes: dict[int, float] = {}
            for weight, (_name, route) in zip(weights, agents, strict=True):
                page = route[rank]
                if page is not None and page not in selected:
                    votes[page] = votes.get(page, 0.0) + weight
            if votes:
                selected.append(max(votes, key=lambda page: (votes[page], -page)))
        if len(selected) < self.route_width:
            totals: dict[int, float] = {}
            for rank in range(self.route_width):
                marginal = self._marginal.get((layer, rank))
                if marginal is None:
                    continue
                scale = max(1, sum(marginal.values()))
                for page, count in marginal.items():
                    totals[page] = totals.get(page, 0.0) + count / scale
            for page in sorted(totals, key=lambda value: (-totals[value], value)):
                if page not in selected:
                    selected.append(page)
                    if len(selected) == self.route_width:
                        break
        return tuple(selected)

    def _begin_wave(self, layer: int, *, row_count: int | None) -> None:
        if layer != 0:
            return
        self._wave_routes = {}
        self._pending = {}
        self._pending_prefetch = {}
        self._pending_dynamic = {}
        if self._transaction is not None:
            self._transaction["current_wave"] += 1
            self._transaction["wave_rows"].append(row_count)

    def begin_exact_wave(self, layer: int) -> None:
        """Start one exact layer wave without running the route policy."""

        with self._lock:
            self._ensure_open()
            self._validate_layer(layer)
            self._begin_wave(layer, row_count=None)
            if layer == 0:
                self._compiled = [None] * self.n_layers

    def compile_routes(self) -> None:
        """Compile one O(1)-lookup route per layer from the completed prefill."""

        with self._lock:
            self._ensure_open()
            compiled: list[MlpPagePrediction | None] = []
            for layer in range(self.n_layers):
                compiled.append(self._predict(layer))
            self._compiled = compiled

    def _route(
        self,
        layer: int,
        *,
        row_count: int,
        replay_prefetch: object = _CALL_PREFETCH,
    ) -> MlpPagePrediction:
        self._ensure_open()
        self._validate_layer(layer)
        if (
            isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count <= 0
        ):
            raise ValueError("row_count must be a positive integer")
        self._begin_wave(layer, row_count=row_count)
        prediction = self._predict(layer)
        compiled = self._compiled[layer]
        self._metrics["dynamic_route_calls"] += 1
        self._metrics["dynamic_route_changes"] += int(
            compiled is not None
            and compiled.ready
            and prediction.ready
            and prediction.page_ids != compiled.page_ids
        )
        self._pending[layer] = prediction
        self._metrics["prediction_calls"] += 1
        self._metrics["predicted_pages"] += len(prediction.page_ids)
        self._metrics[
            "ready_predictions" if prediction.ready else "fallback_predictions"
        ] += 1
        prefetch_result = None
        if (
            prediction.ready
            and self.prefetch is not None
            and (
                replay_prefetch is _CALL_PREFETCH
                or replay_prefetch is not None
            )
        ):
            self._metrics["prefetch_calls"] += 1
            self._metrics["prefetch_pages"] += len(prediction.page_ids)
            prefetch_result = (
                bool(self.prefetch(layer, prediction.page_ids))
                if replay_prefetch is _CALL_PREFETCH
                else bool(replay_prefetch)
            )
            self._metrics["prefetch_successes"] += int(prefetch_result)
        self._pending_prefetch[layer] = prefetch_result
        self._pending_dynamic[layer] = True
        return prediction

    def route(self, layer: int, *, row_count: int) -> MlpPagePrediction:
        """Run the cheap page agents against the latest causal route state."""

        with self._lock:
            return self._route(layer, row_count=row_count)

    def _prepare(
        self,
        layer: int,
        *,
        replay_prefetch: object = _CALL_PREFETCH,
    ) -> MlpPagePrediction:
        self._ensure_open()
        self._validate_layer(layer)
        self._begin_wave(layer, row_count=None)
        prediction = self._predict(layer)
        pages = prediction.page_ids
        ready = prediction.ready
        self._pending[layer] = prediction
        self._metrics["prediction_calls"] += 1
        self._metrics["predicted_pages"] += len(pages)
        self._metrics[
            "ready_predictions" if ready else "fallback_predictions"
        ] += 1
        prefetch_result = None
        if (
            ready
            and self.prefetch is not None
            and (
                replay_prefetch is _CALL_PREFETCH
                or replay_prefetch is not None
            )
        ):
            self._metrics["prefetch_calls"] += 1
            self._metrics["prefetch_pages"] += len(pages)
            prefetch_result = (
                bool(self.prefetch(layer, pages))
                if replay_prefetch is _CALL_PREFETCH
                else bool(replay_prefetch)
            )
            self._metrics["prefetch_successes"] += int(prefetch_result)
        self._pending_prefetch[layer] = prefetch_result
        self._pending_dynamic[layer] = False
        return prediction

    def prepare(self, layer: int) -> MlpPagePrediction:
        with self._lock:
            return self._prepare(layer)

    @staticmethod
    def _nested_rows(value: Any, width: int) -> list[list[Any]]:
        if hasattr(value, "detach") and hasattr(value, "tolist"):
            value = value.detach().to(device="cpu").tolist()
        elif hasattr(value, "tolist"):
            value = value.tolist()
        rows: list[list[Any]] = []

        def visit(node: Any) -> None:
            if isinstance(node, (list, tuple)):
                if len(node) == width and all(
                    not isinstance(item, (list, tuple)) for item in node
                ):
                    rows.append(list(node))
                    return
                for item in node:
                    visit(item)
                return
            raise ValueError("MLP page rows have an invalid shape")

        visit(value)
        if not rows:
            raise ValueError("MLP page rows are empty")
        return rows

    def _feedback(
        self,
        layer: int,
        agents: tuple[tuple[str, tuple[int | None, ...]], ...],
        target: tuple[int, ...],
    ) -> None:
        logs = self._agent_logs[layer]
        observations = self._agent_observations[layer]
        hits = self._agent_hits[layer]
        for index, (_name, proposed) in enumerate(agents):
            valid = sum(page is not None for page in proposed)
            correct = sum(
                page == expected
                for page, expected in zip(proposed, target, strict=True)
                if page is not None
            )
            if valid:
                observations[index] += valid
                hits[index] += correct
                logs[index] -= self.LEARNING_RATE * (1.0 - correct / valid)
                self._metrics["agent_feedback"] += valid

    def _learn_exact_route(
        self,
        layer: int,
        route: tuple[int, ...],
        *,
        temporal_source: tuple[int, ...] | None,
        cross_source: tuple[int, ...] | None,
    ) -> None:
        agents = self._agent_routes(layer, temporal_source, cross_source)
        self._feedback(layer, agents, route)
        for rank, page in enumerate(route):
            marginal = self._marginal.setdefault((layer, rank), Counter())
            self._metrics["counter_evictions"] += _space_saving_increment(
                marginal,
                page,
                limit=self.MAX_TARGETS_PER_TRANSITION,
            )
            if temporal_source is not None:
                temporal = self._temporal.setdefault(
                    (layer, temporal_source[rank]), Counter()
                )
                self._metrics["counter_evictions"] += _space_saving_increment(
                    temporal,
                    page,
                    limit=self.MAX_TARGETS_PER_TRANSITION,
                )
            if cross_source is not None:
                cross_counter = self._cross.setdefault(
                    (layer, cross_source[rank]), Counter()
                )
                self._metrics["counter_evictions"] += _space_saving_increment(
                    cross_counter,
                    page,
                    limit=self.MAX_TARGETS_PER_TRANSITION,
                )
        for source_index, source in enumerate(route):
            for target in route[
                source_index + 1 : source_index + 1 + self.COACTIVE_NEIGHBOR_SPAN
            ]:
                coactive = self._coactive.setdefault((layer, source), Counter())
                self._metrics["counter_evictions"] += _space_saving_increment(
                    coactive,
                    target,
                    limit=self.MAX_TARGETS_PER_TRANSITION,
                )
                self._metrics["coactive_updates"] += 1

    def observe_exact_batch(
        self,
        layer: int,
        page_ids: Any,
        page_scores: Any | None = None,
    ) -> None:
        with self._lock:
            self._ensure_open()
            self._validate_layer(layer)
            self._stage_counter_snapshot()
            raw_routes = self._nested_rows(page_ids, self.route_width)
            routes = tuple(self._validate_route(row) for row in raw_routes)
            if page_scores is not None:
                score_rows = self._nested_rows(page_scores, self.route_width)
                if len(score_rows) != len(routes):
                    raise ValueError("MLP page scores differ from their routes")
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                    for row in score_rows
                    for value in row
                ):
                    raise ValueError("MLP page scores are invalid")
            else:
                score_rows = None

            previous = self._last_routes[layer]
            cross_rows = self._wave_routes.get(layer - 1, ())
            if len(cross_rows) not in {0, 1, len(routes)}:
                raise MlpPageMarkovError(
                    "cross-layer route rows differ from the current exact wave"
                )

            transaction = self._transaction
            if transaction is not None:
                wave = transaction["current_wave"]
                if wave < 0:
                    raise MlpPageMarkovError(
                        "exact MLP pages arrived before transaction layer zero"
                    )
                registered = transaction["wave_rows"][wave]
                if registered is None:
                    transaction["wave_rows"][wave] = len(routes)
                elif registered != len(routes):
                    raise MlpPageMarkovError(
                        "MLP page transaction rows changed between layers"
                    )
                transaction["events"].append(
                    ("exact", wave, layer, routes, score_rows, None)
                )

            pending = self._pending.pop(layer, None)
            self._pending_prefetch.pop(layer, None)
            self._pending_dynamic.pop(layer, None)
            if pending is not None and routes:
                predicted = set(pending.page_ids)
                actual = set(routes[-1])
                union = predicted | actual
                if union:
                    self._metrics["route_jaccard_sum_ppm"] += round(
                        len(predicted & actual) / len(union) * 1_000_000
                    )
                    self._metrics["route_jaccard_count"] += 1
                self._metrics["position_hits"] += sum(
                    left == right
                    for left, right in zip(
                        pending.page_ids,
                        routes[-1],
                        strict=False,
                    )
                )

            for index, route in enumerate(routes):
                temporal_source = previous if index == 0 else routes[index - 1]
                cross_source = (
                    None
                    if not cross_rows
                    else cross_rows[0]
                    if len(cross_rows) == 1
                    else cross_rows[index]
                )
                self._learn_exact_route(
                    layer,
                    route,
                    temporal_source=temporal_source,
                    cross_source=cross_source,
                )
            self._last_routes[layer] = routes[-1]
            self._wave_routes[layer] = routes
            self._exact_support[layer] += len(routes)
            self._metrics["exact_batches"] += 1
            self._metrics["exact_rows"] += len(routes)
            self._dirty = True

    def observe(
        self,
        layer: int,
        page_ids: Sequence[int],
        page_scores: Sequence[float] | None = None,
    ) -> None:
        self.observe_exact_batch(layer, page_ids, page_scores)

    def advance_selected(
        self,
        layer: int,
        page_ids: Sequence[int],
        *,
        row_count: int = 1,
    ) -> None:
        with self._lock:
            self._ensure_open()
            self._validate_layer(layer)
            compiled = self._compiled[layer]
            route = (
                compiled.page_ids
                if compiled is not None and page_ids is compiled.page_ids
                else self._validate_route(page_ids)
            )
            if (
                isinstance(row_count, bool)
                or not isinstance(row_count, int)
                or row_count <= 0
            ):
                raise ValueError("row_count must be a positive integer")
            transaction = self._transaction
            pending_prefetch = self._pending_prefetch.get(layer)
            pending_dynamic = self._pending_dynamic.get(layer)
            if transaction is not None:
                if pending_dynamic is None:
                    raise MlpPageMarkovError(
                        "selected MLP pages have no pending route origin"
                    )
                wave = transaction["current_wave"]
                if wave < 0:
                    raise MlpPageMarkovError(
                        "selected MLP pages arrived before transaction layer zero"
                    )
                registered = transaction["wave_rows"][wave]
                if registered is None:
                    transaction["wave_rows"][wave] = row_count
                elif registered != row_count:
                    raise MlpPageMarkovError(
                        "MLP page transaction rows changed between layers"
                    )
                transaction["events"].append(
                    (
                        "selected_route" if pending_dynamic else "selected_prepare",
                        wave,
                        layer,
                        route,
                        None,
                        pending_prefetch,
                    )
                )
            self._pending.pop(layer, None)
            self._pending_prefetch.pop(layer, None)
            self._pending_dynamic.pop(layer, None)
            self._last_routes[layer] = route
            self._wave_routes[layer] = (route,)
            self._metrics["selected_advances"] += 1
            self._dirty = True

    def reset_session(self) -> None:
        """Drop request-local route context while retaining learned transitions."""

        with self._lock:
            self._ensure_open()
            if self._transaction is not None:
                self.rollback_transaction()
            self._last_routes = [None] * self.n_layers
            self._wave_routes = {}
            self._pending = {}
            self._pending_prefetch = {}
            self._pending_dynamic = {}
            self._compiled = [None] * self.n_layers
            self._metrics["session_resets"] += 1
            self._dirty = True

    def snapshot_identity(self) -> dict[str, object]:
        """Return the static execution identity used by continuation snapshots."""

        with self._lock:
            self._ensure_open()
            return {
                "identity": self.identity,
                "min_exact_rows": self.min_exact_rows,
                "n_layers": self.n_layers,
                "page_count": self.page_count,
                "policy": MLP_PAGE_MARKOV_POLICY,
                "route_width": self.route_width,
                "schema": MLP_PAGE_MARKOV_SCHEMA,
            }

    @staticmethod
    def _counter_rows(
        values: Mapping[tuple[int, ...], Counter[int]],
    ) -> list[list[object]]:
        return [
            [*key, [[page, counter[page]] for page in sorted(counter)]]
            for key, counter in sorted(values.items())
        ]

    def _restore_counters(
        self,
        rows: object,
        *,
        kind: str,
    ) -> dict[tuple[int, ...], Counter[int]]:
        key_width = 2
        if not isinstance(rows, list):
            raise ValueError("counter rows must be a list")
        result: dict[tuple[int, ...], Counter[int]] = {}
        for row in rows:
            if not isinstance(row, list) or len(row) != key_width + 1:
                raise ValueError("counter row shape is invalid")
            key = tuple(row[:key_width])
            pairs = row[-1]
            if (
                key in result
                or any(isinstance(value, bool) or not isinstance(value, int) for value in key)
                or not 0 <= key[0] < self.n_layers
                or (
                    kind == "marginal"
                    and not 0 <= key[1] < self.route_width
                )
                or (
                    kind != "marginal"
                    and not 0 <= key[1] < self.page_count
                )
                or not isinstance(pairs, list)
                or not 0 < len(pairs) <= self.MAX_TARGETS_PER_TRANSITION
            ):
                raise ValueError("counter key is invalid")
            counter: Counter[int] = Counter()
            for pair in pairs:
                if (
                    not isinstance(pair, list)
                    or len(pair) != 2
                    or isinstance(pair[0], bool)
                    or not isinstance(pair[0], int)
                    or not 0 <= pair[0] < self.page_count
                    or pair[0] in counter
                    or isinstance(pair[1], bool)
                    or not isinstance(pair[1], int)
                    or pair[1] <= 0
                ):
                    raise ValueError("counter value is invalid")
                counter[pair[0]] = pair[1]
            result[key] = counter
        return result

    def _config(self, *, policy: str = MLP_PAGE_MARKOV_POLICY) -> dict[str, object]:
        return {
            "identity": self.identity,
            "min_exact_rows": self.min_exact_rows,
            "n_layers": self.n_layers,
            "page_count": self.page_count,
            "policy": policy,
            "route_width": self.route_width,
        }

    def _body(self) -> dict[str, object]:
        return {
            "agent_hits": self._agent_hits,
            "agent_logs": [
                [value.hex() for value in row] for row in self._agent_logs
            ],
            "agent_observations": self._agent_observations,
            "config": self._config(),
            "coactive": self._counter_rows(self._coactive),
            "cross": self._counter_rows(self._cross),
            "exact_support": self._exact_support,
            "last_routes": [
                None if row is None else list(row) for row in self._last_routes
            ],
            "marginal": self._counter_rows(self._marginal),
            "metrics": dict(self._metrics),
            "temporal": self._counter_rows(self._temporal),
        }

    @staticmethod
    def _integer_matrix(
        value: object,
        *,
        rows: int,
        columns: int,
        label: str,
    ) -> list[list[int]]:
        if (
            not isinstance(value, list)
            or len(value) != rows
            or any(
                not isinstance(row, list)
                or len(row) != columns
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, int)
                    or item < 0
                    for item in row
                )
                for row in value
            )
        ):
            raise ValueError(f"{label} shape is invalid")
        return [list(row) for row in value]

    def _load(self) -> None:
        if self.path is None or (
            not self.path.exists() and not self.path.is_symlink()
        ):
            return
        try:
            raw = _stable_read(self.path)
            document = json.loads(raw.decode("ascii"))
            schema = document.get("schema") if isinstance(document, dict) else None
            legacy = schema == _V3_MLP_PAGE_MARKOV_SCHEMA
            if (
                not isinstance(document, dict)
                or set(document) != {"body", "schema", "sha256"}
                or schema
                not in {MLP_PAGE_MARKOV_SCHEMA, _V3_MLP_PAGE_MARKOV_SCHEMA}
                or not isinstance(document.get("body"), dict)
                or document.get("sha256") != _digest(document["body"])
                or _canonical(document) != raw
            ):
                raise ValueError("invalid state envelope")
            body = document["body"]
            expected_body = {
                "agent_hits",
                "agent_logs",
                "agent_observations",
                "config",
                "cross",
                "exact_support",
                "last_routes",
                "marginal",
                "metrics",
                "temporal",
            }
            if not legacy:
                expected_body.add("coactive")
            expected_config = self._config(
                policy=(
                    _V3_MLP_PAGE_MARKOV_POLICY
                    if legacy
                    else MLP_PAGE_MARKOV_POLICY
                )
            )
            if set(body) != expected_body or body["config"] != expected_config:
                raise ValueError("state configuration changed")
            raw_logs = body["agent_logs"]
            if not isinstance(raw_logs, list):
                raise ValueError("agent logs are invalid")
            logs = [
                [float.fromhex(value) for value in row]
                for row in raw_logs
            ]
            if any(not math.isfinite(value) for row in logs for value in row):
                raise ValueError("agent logs are non-finite")
            agent_count = len(_V3_AGENTS) if legacy else len(_AGENTS)
            observations = self._integer_matrix(
                body["agent_observations"],
                rows=self.n_layers,
                columns=agent_count,
                label="agent observations",
            )
            hits = self._integer_matrix(
                body["agent_hits"],
                rows=self.n_layers,
                columns=agent_count,
                label="agent hits",
            )
            if (
                len(logs) != self.n_layers
                or any(len(row) != agent_count for row in logs)
                or any(
                    hits[layer][agent] > observations[layer][agent]
                    for layer in range(self.n_layers)
                    for agent in range(agent_count)
                )
            ):
                raise ValueError("agent state shape changed")
            if legacy:
                logs = [
                    [row[0], row[1], sum(row) / len(row), row[2]]
                    for row in logs
                ]
                observations = [
                    [row[0], row[1], 0, row[2]] for row in observations
                ]
                hits = [[row[0], row[1], 0, row[2]] for row in hits]
            exact_support = body["exact_support"]
            if (
                not isinstance(exact_support, list)
                or len(exact_support) != self.n_layers
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in exact_support
                )
            ):
                raise ValueError("exact support is invalid")
            last = body["last_routes"]
            if not isinstance(last, list) or len(last) != self.n_layers:
                raise ValueError("last-route shape changed")
            last_routes = [
                None if row is None else self._validate_route(row) for row in last
            ]
            metrics = body["metrics"]
            expected_metrics = set(self._metrics) - (_V4_METRICS if legacy else set())
            if (
                not isinstance(metrics, dict)
                or set(metrics) != expected_metrics
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in metrics.values()
                )
            ):
                raise ValueError("metric state changed")
            self._agent_logs = logs
            self._agent_observations = observations
            self._agent_hits = hits
            self._exact_support = list(exact_support)
            self._last_routes = last_routes
            self._marginal = self._restore_counters(
                body["marginal"], kind="marginal"
            )
            self._temporal = self._restore_counters(
                body["temporal"], kind="temporal"
            )
            self._cross = self._restore_counters(body["cross"], kind="cross")
            self._coactive = (
                {}
                if legacy
                else self._restore_counters(body["coactive"], kind="coactive")
            )
            self._metrics.update(
                {key: int(value) for key, value in metrics.items()}
            )
        except MlpPageMarkovError:
            raise
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise MlpPageMarkovError("cannot load MLP page state") from exc

    def flush(self) -> None:
        with self._lock:
            self._ensure_open()
            if self._transaction is not None:
                raise MlpPageMarkovError(
                    "cannot flush an active MLP page transaction"
                )
            if self.path is None or not self._dirty:
                return
            body = self._body()
            data = _canonical(
                {
                    "body": body,
                    "schema": MLP_PAGE_MARKOV_SCHEMA,
                    "sha256": _digest(body),
                }
            )
            if len(data) > _MAX_STATE_BYTES:
                raise MlpPageMarkovError("MLP page state exceeds its byte bound")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.parent / (
                f".{self.path.name}.{secrets.token_hex(8)}.tmp"
            )
            descriptor: int | None = None
            try:
                descriptor = os.open(
                    temporary,
                    os.O_CREAT
                    | os.O_EXCL
                    | os.O_WRONLY
                    | int(getattr(os, "O_CLOEXEC", 0)),
                    0o600,
                )
                offset = 0
                while offset < len(data):
                    written = os.write(descriptor, data[offset:])
                    if written <= 0:
                        raise OSError("short MLP page state write")
                    offset += written
                os.fsync(descriptor)
                os.close(descriptor)
                descriptor = None
                os.replace(temporary, self.path)
                self._dirty = False
            except OSError as exc:
                raise MlpPageMarkovError("cannot persist MLP page state") from exc
            finally:
                if descriptor is not None:
                    os.close(descriptor)
                temporary.unlink(missing_ok=True)

    def metrics(self) -> dict[str, object]:
        with self._lock:
            self._ensure_open()
            weights = [self._weights(layer) for layer in range(self.n_layers)]
            return {
                **self._metrics,
                "agent_weights": {
                    name: sum(row[index] for row in weights) / self.n_layers
                    for index, name in enumerate(_AGENTS)
                },
                "coactive_contexts": len(self._coactive),
                "cross_contexts": len(self._cross),
                "exact_supported_layers": sum(
                    support >= self.min_exact_rows
                    for support in self._exact_support
                ),
                "marginal_contexts": len(self._marginal),
                "policy": MLP_PAGE_MARKOV_POLICY,
                "route_jaccard_mean": (
                    0.0
                    if self._metrics["route_jaccard_count"] == 0
                    else self._metrics["route_jaccard_sum_ppm"]
                    / self._metrics["route_jaccard_count"]
                    / 1_000_000
                ),
                "schema": MLP_PAGE_MARKOV_SCHEMA,
                "temporal_contexts": len(self._temporal),
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self.flush()
            self._closed = True


__all__ = [
    "MLP_PAGE_MARKOV_POLICY",
    "MLP_PAGE_MARKOV_SCHEMA",
    "MlpPageMarkov",
    "MlpPageMarkovError",
    "MlpPagePrediction",
]
