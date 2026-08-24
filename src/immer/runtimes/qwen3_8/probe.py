"""Authenticated passive DeltaNet probes for Qwen3.8 calibration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
import hashlib
import json
import math
import re
from typing import Any

from .kernels import DeltaNetProbe


DELTANET_PROBE_SCHEMA = "immer.qwen3.8-deltanet-probe/v1"
DELTANET_COMPARISON_SCHEMA = "immer.qwen3.8-deltanet-comparison/v1"
DELTANET_COMPONENTS = (
    "beta_mean",
    "beta_std",
    "decay_mean",
    "decay_std",
    "conv_norm",
    "q_norm",
    "k_norm",
    "v_norm",
    "delta_norm",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class Qwen38DeltaNetProbeError(ValueError):
    """A DeltaNet observation or comparison violates its identity contract."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Qwen38DeltaNetProbeError("probe value is not canonical JSON") from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _plain_int(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Qwen38DeltaNetProbeError(f"{label} is invalid")
    return value


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Qwen38DeltaNetProbeError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise Qwen38DeltaNetProbeError(f"{label} is not finite")
    return result


class DeltaNetProbeRecorder:
    """Collect immutable layer observations without retaining activations."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []
        self._visits: dict[int, int] = {}

    def __call__(self, layer: int, probe: DeltaNetProbe) -> None:
        index = _plain_int(layer, "probe layer")
        if not isinstance(probe, DeltaNetProbe):
            raise TypeError("probe must be a DeltaNetProbe")
        pass_index = self._visits.get(index, 0)
        self._visits[index] = pass_index + 1
        self._records.append(
            {
                **asdict(probe),
                "layer": index,
                "pass_index": pass_index,
            }
        )

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            dict(row)
            for row in sorted(
                self._records,
                key=lambda row: (int(row["pass_index"]), int(row["layer"])),
            )
        )


def build_probe_document(
    recorder: DeltaNetProbeRecorder,
    *,
    checkpoint: Mapping[str, Any],
    context_mode: str,
    start_pos: int,
    end_pos: int,
    item_id: str,
    input_sha256: str,
    hidden_sha256: str,
) -> dict[str, Any]:
    """Bind one passive probe capture to its exact input and model output."""

    if not isinstance(recorder, DeltaNetProbeRecorder):
        raise TypeError("recorder must be a DeltaNetProbeRecorder")
    if context_mode not in ("prefill", "decode"):
        raise Qwen38DeltaNetProbeError("context_mode is invalid")
    start = _plain_int(start_pos, "start_pos")
    end = _plain_int(end_pos, "end_pos", minimum=1)
    if end <= start:
        raise Qwen38DeltaNetProbeError("probe cursor does not advance")
    if not isinstance(item_id, str) or not item_id:
        raise Qwen38DeltaNetProbeError("item_id is invalid")
    for value, label in (
        (input_sha256, "input_sha256"),
        (hidden_sha256, "hidden_sha256"),
    ):
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise Qwen38DeltaNetProbeError(f"{label} is invalid")
    if not isinstance(checkpoint, Mapping) or not checkpoint:
        raise Qwen38DeltaNetProbeError("checkpoint identity is invalid")
    records = list(recorder.records)
    if not records:
        raise Qwen38DeltaNetProbeError("probe capture is empty")
    body = {
        "checkpoint": dict(checkpoint),
        "context_mode": context_mode,
        "end_pos": end,
        "hidden_sha256": hidden_sha256,
        "input_sha256": input_sha256,
        "item_id": item_id,
        "records": records,
        "start_pos": start,
    }
    document = {
        "body": body,
        "schema": DELTANET_PROBE_SCHEMA,
        "sha256": _sha256(body),
    }
    return verify_probe_document(document)


def verify_probe_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a probe artifact and return its normalized mapping."""

    if not isinstance(document, Mapping) or set(document) != {
        "body",
        "schema",
        "sha256",
    }:
        raise Qwen38DeltaNetProbeError("probe document schema is invalid")
    if document.get("schema") != DELTANET_PROBE_SCHEMA:
        raise Qwen38DeltaNetProbeError("probe schema identifier is invalid")
    body = document.get("body")
    if not isinstance(body, Mapping) or set(body) != {
        "checkpoint",
        "context_mode",
        "end_pos",
        "hidden_sha256",
        "input_sha256",
        "item_id",
        "records",
        "start_pos",
    }:
        raise Qwen38DeltaNetProbeError("probe body schema is invalid")
    if document.get("sha256") != _sha256(body):
        raise Qwen38DeltaNetProbeError("probe SHA-256 mismatch")
    if not isinstance(body.get("checkpoint"), Mapping) or not body["checkpoint"]:
        raise Qwen38DeltaNetProbeError("probe checkpoint identity is invalid")
    if body.get("context_mode") not in ("prefill", "decode"):
        raise Qwen38DeltaNetProbeError("probe context_mode is invalid")
    start = _plain_int(body.get("start_pos"), "probe start_pos")
    end = _plain_int(body.get("end_pos"), "probe end_pos", minimum=1)
    if end <= start:
        raise Qwen38DeltaNetProbeError("probe cursor does not advance")
    if not isinstance(body.get("item_id"), str) or not body["item_id"]:
        raise Qwen38DeltaNetProbeError("probe item_id is invalid")
    for key in ("input_sha256", "hidden_sha256"):
        if not isinstance(body.get(key), str) or _SHA256.fullmatch(body[key]) is None:
            raise Qwen38DeltaNetProbeError(f"probe {key} is invalid")
    records = body.get("records")
    if not isinstance(records, list) or not records:
        raise Qwen38DeltaNetProbeError("probe records are invalid")
    expected = {"layer", "pass_index", *DELTANET_COMPONENTS}
    seen: set[tuple[int, int]] = set()
    for row in records:
        if not isinstance(row, Mapping) or set(row) != expected:
            raise Qwen38DeltaNetProbeError("probe record schema is invalid")
        layer = _plain_int(row.get("layer"), "probe record layer")
        pass_index = _plain_int(row.get("pass_index"), "probe record pass_index")
        coordinate = (pass_index, layer)
        if coordinate in seen:
            raise Qwen38DeltaNetProbeError("probe record coordinate is duplicated")
        seen.add(coordinate)
        for component in DELTANET_COMPONENTS:
            _finite_float(row.get(component), f"probe {component}")
    return json.loads(_canonical(dict(document)))


def _single_pass(document: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    body = document["body"]
    result: dict[int, Mapping[str, Any]] = {}
    for row in body["records"]:
        if row["pass_index"] != 0:
            raise Qwen38DeltaNetProbeError(
                "comparison requires one layer-major pass per sample"
            )
        layer = int(row["layer"])
        if layer in result:
            raise Qwen38DeltaNetProbeError("comparison sample repeats a layer")
        result[layer] = row
    return result


def _sample_mean_variance(values: Sequence[float]) -> tuple[float, float]:
    if len(values) < 2:
        raise Qwen38DeltaNetProbeError("each probe family needs at least two samples")
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, variance


def compare_probe_documents(
    reasoning: Sequence[Mapping[str, Any]],
    knowledge: Sequence[Mapping[str, Any]],
    *,
    protection_threshold: float = 1.0,
) -> dict[str, Any]:
    """Compute the original pooled-variance Cohen's-d layer map."""

    threshold = _finite_float(protection_threshold, "protection_threshold")
    if threshold < 0.0:
        raise Qwen38DeltaNetProbeError("protection_threshold must be non-negative")
    if len(reasoning) < 2 or len(knowledge) < 2:
        raise Qwen38DeltaNetProbeError("each probe family needs at least two samples")
    groups = {
        "reasoning": [verify_probe_document(row) for row in reasoning],
        "knowledge": [verify_probe_document(row) for row in knowledge],
    }
    all_documents = groups["reasoning"] + groups["knowledge"]
    reference_body = all_documents[0]["body"]
    checkpoint = reference_body["checkpoint"]
    context_mode = reference_body["context_mode"]
    input_hashes: set[str] = set()
    samples: dict[str, list[dict[int, Mapping[str, Any]]]] = {
        "reasoning": [],
        "knowledge": [],
    }
    layers: tuple[int, ...] | None = None
    for family, documents in groups.items():
        for document in documents:
            body = document["body"]
            if body["checkpoint"] != checkpoint:
                raise Qwen38DeltaNetProbeError("probe checkpoints differ")
            if body["context_mode"] != context_mode:
                raise Qwen38DeltaNetProbeError("probe context modes differ")
            input_sha256 = str(body["input_sha256"])
            if input_sha256 in input_hashes:
                raise Qwen38DeltaNetProbeError(
                    "probe input is duplicated across samples"
                )
            input_hashes.add(input_sha256)
            sample = _single_pass(document)
            current_layers = tuple(sorted(sample))
            if layers is None:
                layers = current_layers
            elif current_layers != layers:
                raise Qwen38DeltaNetProbeError("probe samples cover different layers")
            samples[family].append(sample)
    if not layers:
        raise Qwen38DeltaNetProbeError("probe comparison has no layers")

    statistics: dict[str, dict[str, dict[str, float]]] = {}
    ranking: list[dict[str, Any]] = []
    protected: dict[str, list[str]] = {}
    for layer in layers:
        layer_stats: dict[str, dict[str, float]] = {}
        drivers: list[str] = []
        for component in DELTANET_COMPONENTS:
            values_reasoning = [
                float(sample[layer][component]) for sample in samples["reasoning"]
            ]
            values_knowledge = [
                float(sample[layer][component]) for sample in samples["knowledge"]
            ]
            mean_reasoning, variance_reasoning = _sample_mean_variance(values_reasoning)
            mean_knowledge, variance_knowledge = _sample_mean_variance(values_knowledge)
            n_reasoning = len(values_reasoning)
            n_knowledge = len(values_knowledge)
            pooled_variance = (
                (n_reasoning - 1) * variance_reasoning
                + (n_knowledge - 1) * variance_knowledge
            ) / (n_reasoning + n_knowledge - 2)
            pooled_std = math.sqrt(max(0.0, pooled_variance))
            effect = (mean_reasoning - mean_knowledge) / (pooled_std + 1e-12)
            layer_stats[component] = {
                "d": effect,
                "mu_knowledge": mean_knowledge,
                "mu_reasoning": mean_reasoning,
                "pooled_std": pooled_std,
            }
            ranking.append(
                {
                    "abs_d": abs(effect),
                    "component": component,
                    "d": effect,
                    "layer": layer,
                }
            )
            if abs(effect) >= threshold:
                drivers.append(component)
        statistics[str(layer)] = layer_stats
        if drivers:
            protected[str(layer)] = drivers
    ranking.sort(
        key=lambda row: (-float(row["abs_d"]), int(row["layer"]), str(row["component"]))
    )

    body = {
        "checkpoint": checkpoint,
        "components": list(DELTANET_COMPONENTS),
        "context_mode": context_mode,
        "families": {
            "knowledge": len(groups["knowledge"]),
            "reasoning": len(groups["reasoning"]),
        },
        "formula": "(mu_reasoning-mu_knowledge)/(pooled_sample_std+1e-12)",
        "layers": list(layers),
        "protected_layers": protected,
        "protection_threshold_abs_d": threshold,
        "source_probe_sha256": {
            family: [document["sha256"] for document in documents]
            for family, documents in groups.items()
        },
        "statistics": statistics,
        "top10": ranking[:10],
    }
    return {
        "body": body,
        "schema": DELTANET_COMPARISON_SCHEMA,
        "sha256": _sha256(body),
    }


__all__ = [
    "DELTANET_COMPARISON_SCHEMA",
    "DELTANET_COMPONENTS",
    "DELTANET_PROBE_SCHEMA",
    "DeltaNetProbeRecorder",
    "Qwen38DeltaNetProbeError",
    "build_probe_document",
    "compare_probe_documents",
    "verify_probe_document",
]
