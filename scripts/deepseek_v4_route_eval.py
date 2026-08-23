#!/usr/bin/env python3
"""Evaluate label-free DeepSeek-V4 router traces on held-out whole prompts.

The input is the canonical ``route-observations-*.json`` sidecar written by
``deepseek_v4_fertig_draft_verify.py``.  Rows are reconstructed from the exact
batch-major right-padded layout, split only at stable sequence boundaries, and
evaluated without benchmark questions, item IDs, gold values, or answers.

    PYTHONPATH=src python3 scripts/deepseek_v4_route_eval.py
    PYTHONPATH=src python3 scripts/deepseek_v4_route_eval.py \
        --execution-digest 0123...cdef
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from immer.runtimes.deepseek_v4.causal_prefetch import CheckpointIdentity
from immer.runtimes.deepseek_v4.draft_verification import ROUTE_OBSERVATION_SCHEMA
from immer.runtimes.deepseek_v4.route_markov import (
    KSweepEvaluation,
    LayerMarkovExpertPredictor,
    LayerTokenRoutes,
    PromptRouteObservation,
    RouteMarkovError,
    apply_target_layer_placebo,
    build_target_layer_placebo,
    evaluate_k_sweep,
    split_prompt_observations,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SIDECAR = (
    ROOT
    / "artifacts"
    / "private"
    / "deepseek-v4-fertig-draft-verify"
    / "route-observations-off.json"
)
DEFAULT_OUTPUT = ROOT / "results" / "deepseek-v4-route-eval.json"
OFFICIAL_SOURCE = "deepseek-ai/DeepSeek-V4-Flash-0731"
OFFICIAL_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
OFFICIAL_INVENTORY_FINGERPRINT = (
    "61600c552f3e52ae382b3eca0370001905fc4e2ecdd2da5f3e66fa95809206c7"
)
OFFICIAL_CHECKPOINT = CheckpointIdentity(
    repo_id=OFFICIAL_SOURCE,
    revision=OFFICIAL_REVISION,
    inventory_fingerprint=OFFICIAL_INVENTORY_FINGERPRINT,
)
ROUTE_OBSERVATIONS_SCHEMA = "immer.deepseek-v4-route-observations/v1"
REPORT_SCHEMA = "immer.deepseek-v4-route-eval/v1"
HEADLINE_K = (1, 3, 6, 12, 24, 48, 96, 256)
MICRO_WINDOW_ROWS = (1, 2, 4, 8, 16, 32, 64)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_OBSERVATION_FIELDS = frozenset(
    {
        "checkpoint",
        "execution_feature_digest",
        "layer",
        "observation_id",
        "observation_schema",
        "request_id",
        "row_layout",
        "selected_expert_ids",
    }
)
_LAYOUT_FIELDS = frozenset(
    {
        "active_lengths",
        "batch_size",
        "order",
        "padded_length",
        "sequence_digests",
        "selected_rows",
    }
)


class RouteEvalError(RuntimeError):
    """The route sidecar or evaluation contract is invalid."""


@dataclass(frozen=True, slots=True)
class LoadedSidecar:
    checkpoint: CheckpointIdentity
    sha256: str
    observations: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ReconstructedRoutes:
    checkpoint: CheckpointIdentity
    sidecar_sha256: str
    execution_feature_digest: str
    available_execution_digests: tuple[str, ...]
    source_observations: int
    source_requests: int
    layer_ids: tuple[int, ...]
    prompts: tuple[PromptRouteObservation, ...]


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RouteEvalError("value is not canonical JSON") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_json(encoded: bytes, *, path: Path) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RouteEvalError(f"duplicate JSON key in route sidecar: {key!r}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise RouteEvalError(f"non-finite JSON number in route sidecar: {value}")

    try:
        return json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=invalid_constant,
        )
    except RouteEvalError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RouteEvalError(f"cannot parse route sidecar: {path}") from exc


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _fraction(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 < parsed < 1:
        raise argparse.ArgumentTypeError("must be strictly between zero and one")
    return parsed


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RouteEvalError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RouteEvalError(f"{label} must be an integer >= {minimum}")
    return value


def _text(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value.encode("utf-8")) > 512
    ):
        raise RouteEvalError(f"{label} must be trimmed non-empty text")
    return value


def _checkpoint(
    raw: Any,
    *,
    expected: CheckpointIdentity | None,
) -> CheckpointIdentity:
    if not isinstance(raw, Mapping) or set(raw) != {
        "inventory_fingerprint",
        "repo_id",
        "revision",
    }:
        raise RouteEvalError("route sidecar checkpoint identity is invalid")
    try:
        checkpoint = CheckpointIdentity(
            repo_id=raw["repo_id"],
            revision=raw["revision"],
            inventory_fingerprint=raw["inventory_fingerprint"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RouteEvalError("route sidecar checkpoint identity is invalid") from exc
    if checkpoint.as_record() != dict(raw):
        raise RouteEvalError("route sidecar checkpoint identity is not canonical")
    if (
        checkpoint.repo_id != OFFICIAL_SOURCE
        or checkpoint.revision != OFFICIAL_REVISION
    ):
        raise RouteEvalError("route sidecar belongs to another logical checkpoint")
    if expected is not None and checkpoint != expected:
        raise RouteEvalError("route sidecar belongs to another checkpoint")
    return checkpoint


def _expected_observation_id(request_id: str, layer: int) -> str:
    identity = {
        "layer": layer,
        "request_id": request_id,
        "schema": ROUTE_OBSERVATION_SCHEMA,
    }
    return f"deepseek-v4-route:v1:{_sha256(identity)}"


def _normalize_observation(
    raw: Any,
    *,
    checkpoint: CheckpointIdentity,
    n_experts: int,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _OBSERVATION_FIELDS:
        raise RouteEvalError("route observation has unknown or missing fields")
    if raw.get("checkpoint") != checkpoint.as_record():
        raise RouteEvalError("route observation belongs to another checkpoint")
    execution_digest = _digest(
        raw.get("execution_feature_digest"), "execution feature digest"
    )
    layer = _integer(raw.get("layer"), "route layer")
    request_id = _text(raw.get("request_id"), "request_id")
    observation_id = _text(raw.get("observation_id"), "observation_id")
    if raw.get("observation_schema") != ROUTE_OBSERVATION_SCHEMA:
        raise RouteEvalError("route observation source schema is invalid")
    if observation_id != _expected_observation_id(request_id, layer):
        raise RouteEvalError("route observation ID does not bind its request and layer")

    layout = raw.get("row_layout")
    if not isinstance(layout, Mapping) or set(layout) != _LAYOUT_FIELDS:
        raise RouteEvalError("route observation row layout is invalid")
    batch_size = _integer(layout.get("batch_size"), "batch_size", minimum=1)
    padded_length = _integer(layout.get("padded_length"), "padded_length", minimum=1)
    selected_rows = _integer(layout.get("selected_rows"), "selected_rows", minimum=1)
    if layout.get("order") != "batch-major-right-padded":
        raise RouteEvalError("route observation row order is invalid")
    if selected_rows != batch_size * padded_length:
        raise RouteEvalError("selected row count does not match the padded batch")
    active_lengths = layout.get("active_lengths")
    if (
        not isinstance(active_lengths, list)
        or len(active_lengths) != batch_size
        or any(
            isinstance(length, bool)
            or not isinstance(length, int)
            or not 1 <= length <= padded_length
            for length in active_lengths
        )
    ):
        raise RouteEvalError("route observation active lengths are invalid")
    sequence_digests = layout.get("sequence_digests")
    if not isinstance(sequence_digests, list) or len(sequence_digests) != batch_size:
        raise RouteEvalError("route observation sequence digests are invalid")
    normalized_digests = tuple(
        _digest(value, "sequence digest") for value in sequence_digests
    )
    if len(set(normalized_digests)) != len(normalized_digests):
        raise RouteEvalError("route layout contains duplicate sequence digests")

    selected = raw.get("selected_expert_ids")
    if not isinstance(selected, list) or len(selected) != selected_rows:
        raise RouteEvalError("route observation selected expert rows are invalid")
    normalized_selected: list[list[int]] = []
    for row in selected:
        if not isinstance(row, list):
            raise RouteEvalError("selected expert row must be a JSON array")
        normalized_row = [_integer(expert, "expert ID") for expert in row]
        if any(expert >= n_experts for expert in normalized_row):
            raise RouteEvalError("expert ID is outside the configured inventory")
        normalized_selected.append(normalized_row)

    return {
        "checkpoint": checkpoint.as_record(),
        "execution_feature_digest": execution_digest,
        "layer": layer,
        "observation_id": observation_id,
        "observation_schema": ROUTE_OBSERVATION_SCHEMA,
        "request_id": request_id,
        "row_layout": {
            "active_lengths": list(active_lengths),
            "batch_size": batch_size,
            "order": "batch-major-right-padded",
            "padded_length": padded_length,
            "sequence_digests": list(normalized_digests),
            "selected_rows": selected_rows,
        },
        "selected_expert_ids": normalized_selected,
    }


def load_route_sidecar(
    path: str | Path,
    *,
    n_experts: int = 256,
    expected_checkpoint: CheckpointIdentity | None = None,
) -> LoadedSidecar:
    """Load and byte-verify the runner's canonical sidecar."""

    inventory = _integer(n_experts, "n_experts", minimum=1)
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise RouteEvalError(f"route sidecar must be a regular file: {source}")
    try:
        encoded = source.read_bytes()
    except OSError as exc:
        raise RouteEvalError(f"cannot read route sidecar: {source}") from exc
    document = _strict_json(encoded, path=source)
    if not isinstance(document, Mapping) or set(document) != {
        "checkpoint",
        "observations",
        "schema",
        "sha256",
    }:
        raise RouteEvalError("route sidecar schema is invalid")
    if document.get("schema") != ROUTE_OBSERVATIONS_SCHEMA:
        raise RouteEvalError("route sidecar schema is unsupported")
    checkpoint = _checkpoint(document.get("checkpoint"), expected=expected_checkpoint)
    raw_observations = document.get("observations")
    if not isinstance(raw_observations, list) or not raw_observations:
        raise RouteEvalError("route sidecar contains no observations")
    observations = tuple(
        _normalize_observation(
            raw,
            checkpoint=checkpoint,
            n_experts=inventory,
        )
        for raw in raw_observations
    )
    observation_ids = [row["observation_id"] for row in observations]
    if len(set(observation_ids)) != len(observation_ids):
        raise RouteEvalError("route sidecar contains duplicate observation IDs")
    request_layers = [(row["request_id"], row["layer"]) for row in observations]
    if len(set(request_layers)) != len(request_layers):
        raise RouteEvalError("route sidecar contains duplicate request/layer rows")

    ordered = tuple(
        sorted(
            observations,
            key=lambda row: (
                row["request_id"],
                row["layer"],
                row["observation_id"],
            ),
        )
    )
    identity = {
        "checkpoint": checkpoint.as_record(),
        "observations": list(ordered),
        "schema": ROUTE_OBSERVATIONS_SCHEMA,
    }
    expected_sha = _sha256(identity)
    if document.get("sha256") != expected_sha:
        raise RouteEvalError("route sidecar SHA-256 mismatch")
    expected_document = {**identity, "sha256": expected_sha}
    if encoded != _canonical_json_bytes(expected_document):
        raise RouteEvalError("route sidecar is not canonical JSON")
    return LoadedSidecar(checkpoint, expected_sha, ordered)


def _execution_digest(
    observations: Sequence[Mapping[str, Any]],
    requested: str | None,
) -> tuple[str, tuple[str, ...]]:
    available = tuple(
        sorted({str(row["execution_feature_digest"]) for row in observations})
    )
    if requested is not None:
        selected = _digest(requested, "selected execution digest")
        if selected not in available:
            raise RouteEvalError("selected execution digest is absent from the sidecar")
        return selected, available
    if len(available) != 1:
        raise RouteEvalError(
            "route sidecar mixes execution regimes; select one execution digest"
        )
    return available[0], available


def reconstruct_prompt_routes(
    sidecar: LoadedSidecar,
    *,
    execution_digest: str | None = None,
) -> ReconstructedRoutes:
    """Reconstruct exact per-sequence layer traces from padded batch rows."""

    selected_digest, available = _execution_digest(
        sidecar.observations, execution_digest
    )
    rows = tuple(
        row
        for row in sidecar.observations
        if row["execution_feature_digest"] == selected_digest
    )
    by_request: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_request.setdefault(str(row["request_id"]), []).append(row)

    # One stable sequence can be replayed in another batch. Exact replays are
    # deduplicated; any differing row or length under that digest is a conflict.
    sequence_layers: dict[str, dict[int, tuple[tuple[int, ...], ...]]] = {}
    sequence_lengths: dict[str, int] = {}
    for request_id in sorted(by_request):
        request_rows = sorted(by_request[request_id], key=lambda row: int(row["layer"]))
        layers = tuple(int(row["layer"]) for row in request_rows)
        if layers != tuple(range(layers[0], layers[-1] + 1)):
            raise RouteEvalError("route request has a missing layer observation")
        reference_layout = request_rows[0]["row_layout"]
        for row in request_rows[1:]:
            if row["row_layout"] != reference_layout:
                raise RouteEvalError("route request has conflicting row layouts")
        padded_length = int(reference_layout["padded_length"])
        active_lengths = tuple(
            int(value) for value in reference_layout["active_lengths"]
        )
        sequence_digests = tuple(
            str(value) for value in reference_layout["sequence_digests"]
        )
        for batch_index, (sequence_digest, active_length) in enumerate(
            zip(sequence_digests, active_lengths, strict=True)
        ):
            prior_length = sequence_lengths.setdefault(sequence_digest, active_length)
            if prior_length != active_length:
                raise RouteEvalError(
                    "stable sequence digest is bound to conflicting active lengths"
                )
            per_layer = sequence_layers.setdefault(sequence_digest, {})
            start = batch_index * padded_length
            stop = start + padded_length
            for row in request_rows:
                padded_rows = row["selected_expert_ids"][start:stop]
                if len(padded_rows) != padded_length:
                    raise RouteEvalError("route layout is missing padded batch rows")
                active_rows = padded_rows[:active_length]
                padding_rows = padded_rows[active_length:]
                if any(padding_rows):
                    raise RouteEvalError("right-padding rows contain expert selections")
                if any(not expert_row for expert_row in active_rows):
                    raise RouteEvalError(
                        "active token row is missing expert selections"
                    )
                frozen = tuple(
                    tuple(int(expert) for expert in expert_row)
                    for expert_row in active_rows
                )
                layer = int(row["layer"])
                prior = per_layer.get(layer)
                if prior is not None and prior != frozen:
                    raise RouteEvalError(
                        "stable sequence digest is bound to conflicting route rows"
                    )
                per_layer[layer] = frozen

    if len(sequence_layers) < 2:
        raise RouteEvalError("route evaluation requires at least two whole sequences")
    union_layers: set[int] = set()
    prompts: list[PromptRouteObservation] = []
    for sequence_digest in sorted(sequence_layers):
        per_layer = sequence_layers[sequence_digest]
        layers = tuple(sorted(per_layer))
        if layers != tuple(range(layers[0], layers[-1] + 1)):
            raise RouteEvalError("reconstructed sequence has a missing route layer")
        if len(layers) < 2:
            raise RouteEvalError("route evaluation requires at least two route layers")
        union_layers.update(layers)
        prompts.append(
            PromptRouteObservation(
                observation_id=sequence_digest,
                layers=tuple(
                    LayerTokenRoutes(layer, per_layer[layer]) for layer in layers
                ),
            )
        )
    return ReconstructedRoutes(
        checkpoint=sidecar.checkpoint,
        sidecar_sha256=sidecar.sha256,
        execution_feature_digest=selected_digest,
        available_execution_digests=available,
        source_observations=len(rows),
        source_requests=len(by_request),
        layer_ids=tuple(sorted(union_layers)),
        prompts=tuple(prompts),
    )


def _point_record(point: Any) -> dict[str, Any]:
    return {
        "k": point.k,
        "predicted_total": point.predicted_total,
        "selection_mass_hits": point.selection_mass_hits,
        "selection_mass_recall": point.selection_mass_recall,
        "set_hits": point.set_hits,
        "set_precision": point.set_precision,
        "set_recall": point.set_recall,
        "target_selection_mass": point.target_selection_mass,
        "target_set_total": point.target_set_total,
    }


def _evaluation_record(evaluation: KSweepEvaluation) -> dict[str, Any]:
    return {
        "curve": [_point_record(point) for point in evaluation.curve],
        "evaluated_rows": evaluation.evaluated_rows,
        "layers": [
            {
                "curve": [_point_record(point) for point in layer.curve],
                "evaluated_rows": layer.evaluated_rows,
                "target_layer": layer.target_layer,
            }
            for layer in evaluation.layers
        ],
        "model_mode": evaluation.mode,
        "n_experts": evaluation.n_experts,
        "prompt_ids": list(evaluation.prompt_ids),
    }


def _split_record(split: Any, *, seed: int, test_fraction: float) -> dict[str, Any]:
    train_ids = [prompt.observation_id for prompt in split.train]
    test_ids = [prompt.observation_id for prompt in split.test]
    digest_identity = {
        "schema": f"{REPORT_SCHEMA}:whole-prompt-split-v1",
        "seed": seed,
        "test_fraction": test_fraction,
        "test": [
            [prompt.observation_id, prompt.payload_sha256] for prompt in split.test
        ],
        "train": [
            [prompt.observation_id, prompt.payload_sha256] for prompt in split.train
        ],
    }
    return {
        "seed": seed,
        "sha256": _sha256(digest_identity),
        "test_fraction": test_fraction,
        "test_ids": test_ids,
        "train_ids": train_ids,
    }


def _headline(
    evaluations: Mapping[str, KSweepEvaluation], n_experts: int
) -> dict[str, Any]:
    widths = tuple(sorted({min(width, n_experts) for width in HEADLINE_K}))
    return {
        "metric": "set_recall",
        "models": {
            name: {
                f"Recall@{width}": evaluation.curve[width - 1].set_recall
                for width in widths
            }
            for name, evaluation in evaluations.items()
        },
    }


def _micro_window_sweep(
    real_model: LayerMarkovExpertPredictor,
    placebo_model: LayerMarkovExpertPredictor,
    prompts: Sequence[PromptRouteObservation],
    *,
    alpha: float,
    n_experts: int,
) -> list[dict[str, Any]]:
    """Measure row-local causal signal before full-layer aggregation erases it."""

    sweep: list[dict[str, Any]] = []
    for window_rows in MICRO_WINDOW_ROWS:
        real_hits = 0
        placebo_hits = 0
        target_total = 0
        predicted_total = 0
        windows = 0
        for prompt in prompts:
            for source, target in prompt.consecutive_pairs():
                pairs = [
                    (source_row, target_row)
                    for source_row, target_row in zip(
                        source.rows,
                        target.rows,
                        strict=True,
                    )
                    if source_row and target_row
                ]
                for start in range(0, len(pairs), window_rows):
                    chunk = pairs[start : start + window_rows]
                    real_scores = [0.0] * n_experts
                    placebo_scores = [0.0] * n_experts
                    actual: set[int] = set()
                    slots = 0
                    for source_row, target_row in chunk:
                        real = real_model.predict_distribution(
                            source_layer=source.layer,
                            current_row=source_row,
                            alpha=alpha,
                        ).scores
                        placebo = placebo_model.predict_distribution(
                            source_layer=source.layer,
                            current_row=source_row,
                            alpha=alpha,
                        ).scores
                        for expert in range(n_experts):
                            real_scores[expert] += real[expert]
                            placebo_scores[expert] += placebo[expert]
                        actual.update(target_row)
                        slots += len(source_row)
                    if not chunk:
                        continue
                    k = min(n_experts, slots)
                    real_top = set(
                        sorted(
                            range(n_experts),
                            key=lambda expert: (-real_scores[expert], expert),
                        )[:k]
                    )
                    placebo_top = set(
                        sorted(
                            range(n_experts),
                            key=lambda expert: (-placebo_scores[expert], expert),
                        )[:k]
                    )
                    real_hits += len(actual & real_top)
                    placebo_hits += len(actual & placebo_top)
                    target_total += len(actual)
                    predicted_total += k
                    windows += 1
        if not windows or not target_total or not predicted_total:
            raise RouteEvalError("micro-window sweep contains no evaluable routes")
        real_recall = real_hits / target_total
        placebo_recall = placebo_hits / target_total
        sweep.append(
            {
                "delta_recall": real_recall - placebo_recall,
                "k_mean": predicted_total / windows,
                "placebo_recall": placebo_recall,
                "real_precision": real_hits / predicted_total,
                "real_recall": real_recall,
                "target_union_mean": target_total / windows,
                "window_rows": window_rows,
                "windows": windows,
            }
        )
    return sweep


def build_report(
    sidecar_path: str | Path,
    *,
    execution_digest: str | None = None,
    n_experts: int = 256,
    test_fraction: float = 0.25,
    split_seed: int = 17,
    placebo_seed: int = 29,
    alpha: float = 1.0,
    expected_checkpoint: CheckpointIdentity | None = None,
) -> dict[str, Any]:
    """Run real/placebo/baseline K sweeps and return a canonical report."""

    inventory = _integer(n_experts, "n_experts", minimum=1)
    normalized_split_seed = _integer(split_seed, "split_seed")
    normalized_placebo_seed = _integer(placebo_seed, "placebo_seed")
    if (
        isinstance(test_fraction, bool)
        or not isinstance(test_fraction, (int, float))
        or not math.isfinite(test_fraction)
        or not 0 < test_fraction < 1
    ):
        raise RouteEvalError("test_fraction must be strictly between zero and one")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or alpha <= 0
    ):
        raise RouteEvalError("alpha must be finite and positive")
    prior = float(alpha)

    loaded = load_route_sidecar(
        sidecar_path,
        n_experts=inventory,
        expected_checkpoint=expected_checkpoint,
    )
    routes = reconstruct_prompt_routes(
        loaded,
        execution_digest=execution_digest,
    )
    try:
        split = split_prompt_observations(
            routes.prompts,
            test_fraction=float(test_fraction),
            seed=normalized_split_seed,
        )
        real_model = LayerMarkovExpertPredictor(n_experts=inventory)
        real_model.fit(split.train)
        placebo = build_target_layer_placebo(
            split.train,
            seed=normalized_placebo_seed,
        )
        placebo_training = apply_target_layer_placebo(split.train, placebo)
        placebo_model = LayerMarkovExpertPredictor(n_experts=inventory)
        placebo_model.fit(placebo_training)
        evaluations = {
            "real_markov": evaluate_k_sweep(
                real_model, split.test, mode="markov", alpha=prior
            ),
            "placebo_markov": evaluate_k_sweep(
                placebo_model, split.test, mode="markov", alpha=prior
            ),
            "marginal": evaluate_k_sweep(
                real_model, split.test, mode="marginal", alpha=prior
            ),
            "passthrough": evaluate_k_sweep(
                real_model, split.test, mode="passthrough", alpha=prior
            ),
        }
        micro_window_sweep = _micro_window_sweep(
            real_model,
            placebo_model,
            split.test,
            alpha=prior,
            n_experts=inventory,
        )
    except RouteMarkovError as exc:
        raise RouteEvalError(str(exc)) from exc

    evaluated_rows = {result.evaluated_rows for result in evaluations.values()}
    evaluated_ids = {result.prompt_ids for result in evaluations.values()}
    if len(evaluated_rows) != 1 or len(evaluated_ids) != 1:
        raise RouteEvalError("baseline evaluations did not use identical held-out rows")
    identity: dict[str, Any] = {
        "checkpoint": routes.checkpoint.as_record(),
        "evaluations": {
            name: _evaluation_record(evaluation)
            for name, evaluation in evaluations.items()
        },
        "headline": _headline(evaluations, inventory),
        "micro_window_sweep": micro_window_sweep,
        "models": {
            "placebo_markov_snapshot_sha256": placebo_model.snapshot_sha256,
            "real_markov_snapshot_sha256": real_model.snapshot_sha256,
        },
        "protocol": {
            "alpha": prior,
            "held_out_routes_untouched": True,
            "n_experts": inventory,
            "placebo_seed": normalized_placebo_seed,
            "split_seed": normalized_split_seed,
            "test_fraction": float(test_fraction),
            "whole_prompt_split": True,
        },
        "schema": REPORT_SCHEMA,
        "sidecar": {
            "available_execution_digests": list(routes.available_execution_digests),
            "execution_feature_digest": routes.execution_feature_digest,
            "layer_ids": list(routes.layer_ids),
            "prompt_count": len(routes.prompts),
            "request_count": routes.source_requests,
            "schema": ROUTE_OBSERVATIONS_SCHEMA,
            "sha256": routes.sidecar_sha256,
            "source_observations": routes.source_observations,
        },
        "split": _split_record(
            split,
            seed=normalized_split_seed,
            test_fraction=float(test_fraction),
        ),
        "target_layer_placebo": {
            "assignment_count": len(placebo.assignments),
            "seed": placebo.seed,
            "sha256": placebo.sha256,
        },
    }
    return {**identity, "sha256": _sha256(identity)}


def write_report(path: str | Path, report: Mapping[str, Any]) -> None:
    """Atomically persist one canonical report, including its verified digest."""

    if not isinstance(report, Mapping) or set(report) != {
        "checkpoint",
        "evaluations",
        "headline",
        "micro_window_sweep",
        "models",
        "protocol",
        "schema",
        "sha256",
        "sidecar",
        "split",
        "target_layer_placebo",
    }:
        raise RouteEvalError("route evaluation report schema is invalid")
    identity = {key: value for key, value in report.items() if key != "sha256"}
    if report.get("schema") != REPORT_SCHEMA or report.get("sha256") != _sha256(
        identity
    ):
        raise RouteEvalError("route evaluation report digest is invalid")
    encoded = _canonical_json_bytes(dict(report))
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", default=str(DEFAULT_SIDECAR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--execution-digest",
        help="select one execution regime when a sidecar contains several",
    )
    parser.add_argument("--n-experts", type=_positive_int, default=256)
    parser.add_argument("--test-fraction", type=_fraction, default=0.25)
    parser.add_argument("--split-seed", type=_nonnegative_int, default=17)
    parser.add_argument("--placebo-seed", type=_nonnegative_int, default=29)
    parser.add_argument("--alpha", type=_positive_float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = build_report(
            args.sidecar,
            execution_digest=args.execution_digest,
            n_experts=args.n_experts,
            test_fraction=args.test_fraction,
            split_seed=args.split_seed,
            placebo_seed=args.placebo_seed,
            alpha=args.alpha,
        )
        write_report(args.output, report)
    except RouteEvalError as exc:
        raise SystemExit(f"route evaluation failed: {exc}") from exc
    headline = report["headline"]["models"]
    print(
        json.dumps(
            {
                "headline": headline,
                "output": str(Path(args.output).expanduser().resolve()),
                "sha256": report["sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
