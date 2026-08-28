"""Live exact Qwen MLP capture over authenticated O1/Atlas authorities.

The runner executes the local causal Qwen with IMMER's own attention path.  It
supports the original layer-local five-prompt plan byte-for-byte and the v2
generation split: one full-span forward captures all 64 MLP boundaries for a
prompt, while each layer replay reads only that split's 3, 2, or 5 prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch

from immer.knowledge import AccessTraceRecorder

from ..qwen3_8.cartography_probe import (
    HiddenSketchProjection,
    ProbeSpec,
    project_hidden_sketch,
    prompt_token_sha256,
)
from ..qwen3_8.kernels import swiglu
from ..qwen3_8.model import StreamedQwen38
from ..qwen3_8.provenance import runtime_source_manifest
from ..qwen3_8.semantic_atlas import MeasurementReceipt, SemanticWeightAtlas
from .compute_crystals import tensor_sha256
from .identity import canonical_json_bytes, require_sha256
from .operator_harvester import ContextualOperatorObservation
from .qwen_mlp_evidence import (
    CAPTURE_STAGES,
    CaptureManifest,
    CaptureManifestV2,
    CapturePlanEntry,
    ExactMlpBoundaryCapture,
    MlpEvidenceReceipt,
    MlpProjectionVerificationReceipt,
    QwenMlpEvidenceBank,
    QwenMlpEvidenceIntegrityError,
)
from .subspace_battery import (
    SubspaceBatteryFit,
    SubspaceCorpus,
    SubspaceSweepConfig,
    fit_subspace_battery,
)


LIVE_MLP_VERIFIER_SCHEMA = "immer.qwen3.8-live-mlp-verifier/v1"
MLP_CALIBRATION_LOCK_SCHEMA = "immer.qwen3.8-mlp-calibration-lock/v1"
MLP_CALIBRATION_LOCK_V2_SCHEMA = "immer.qwen3.8-mlp-calibration-lock/v2"
MLP_ALL_LAYER_AUTHORITY_FAMILY = (
    "contextual.mlp-all-layer-authority.full-span-v2"
)


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _logical_access_trace(trace: object) -> tuple[str, tuple[str, ...]]:
    operations = getattr(trace, "operations", None)
    if not isinstance(operations, tuple) or not operations:
        raise QwenMlpEvidenceIntegrityError("MLP replay produced no access operations")
    records = []
    for index, operation in enumerate(operations):
        records.append(
            {
                "index": index,
                "leaves": [leaf.to_document() for leaf in operation.leaves],
                "operation": operation.operation,
                "tags": dict(operation.tags),
            }
        )
    body = {
        "inventory_fingerprint": trace.inventory_fingerprint,
        "operations": records,
        "repo_id": trace.repo_id,
        "revision": trace.revision,
        "schema": "immer.qwen3.8-mlp-logical-replay-trace/v1",
    }
    return _digest(body), tuple(_digest(record) for record in records)


@dataclass(frozen=True, slots=True)
class LiveMlpAuthority:
    """One scheduler/spec/Atlas/Harvester authority for a prompt-layer cell."""

    layer: int
    prompt_sha256: str
    probe_spec: ProbeSpec
    observation: ContextualOperatorObservation
    scheduler_observation_sha256: str
    probe_family: str | None = None
    probe_job_sha256: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer, bool)
            or not isinstance(self.layer, int)
            or self.layer < 0
        ):
            raise ValueError("live MLP authority layer is invalid")
        prompt = require_sha256(self.prompt_sha256, field="prompt_sha256")
        scheduler = require_sha256(
            self.scheduler_observation_sha256,
            field="scheduler_observation_sha256",
        )
        if not isinstance(self.probe_spec, ProbeSpec):
            raise TypeError("probe_spec must be a ProbeSpec")
        if not isinstance(self.observation, ContextualOperatorObservation):
            raise TypeError("observation must be a ContextualOperatorObservation")
        measurement = self.observation.measurement
        receipt = self.observation.receipt
        expected_source = f"qwen.layer.{self.layer}.mlp.input-sketch"
        expected_target = f"qwen.layer.{self.layer}.mlp.output-sketch"
        family = self.probe_family
        if family is not None and (
            not isinstance(family, str)
            or not family
            or family != family.strip()
            or "\x00" in family
            or len(family.encode("utf-8")) > 512
        ):
            raise ValueError("probe_family must be bounded canonical text or None")
        composite = family == MLP_ALL_LAYER_AUTHORITY_FAMILY
        if composite:
            job = require_sha256(self.probe_job_sha256, field="probe_job_sha256")
            coordinate_layer = self.probe_spec.coordinate.layer
            base = (
                f"model.language_model.layers.{coordinate_layer}.input_layernorm"
            )
            coordinate = self.probe_spec.coordinate
            if (
                self.probe_spec.start_layer != 0
                or self.probe_spec.stop_layer != coordinate_layer + 1
                or not self.probe_spec.start_layer
                <= self.layer
                < self.probe_spec.stop_layer
                or coordinate.module != base
                or coordinate.tensor != f"{base}.weight"
                or coordinate.head_index is not None
                or coordinate.row_start is not None
                or coordinate.row_end is not None
                or coordinate.relative_byte_offset != 0
                or coordinate.byte_length is not None
            ):
                raise QwenMlpEvidenceIntegrityError(
                    "composite MLP authority is not a passive full-span probe"
                )
        else:
            if self.probe_job_sha256 is not None:
                require_sha256(self.probe_job_sha256, field="probe_job_sha256")
            if self.probe_spec.coordinate.layer != self.layer:
                raise QwenMlpEvidenceIntegrityError(
                    "layer-local MLP authority coordinate differs from its layer"
                )
        if (
            self.probe_spec.prompt_sha256 != prompt
            or measurement.model_pin.sha256 != receipt.model_pin_sha256
            or measurement.sha256 != receipt.measurement_sha256
            or measurement.probe != self.probe_spec.probe_identity
            or measurement.coordinate.layer != self.probe_spec.coordinate.layer
            or measurement.coordinate.module != self.probe_spec.coordinate.module
            or measurement.coordinate.tensor != self.probe_spec.coordinate.tensor
            or measurement.probe.token_sha256 != prompt
            or measurement.intervention.mode != "passive"
            or self.probe_spec.intervention_mode != "passive"
            or receipt.intervention_mode != "passive"
            or receipt.source_state != expected_source
            or receipt.target_state != expected_target
            or receipt.granularity != "operator"
        ):
            raise QwenMlpEvidenceIntegrityError(
                "live MLP authority differs from its passive prompt/layer evidence"
            )
        object.__setattr__(self, "prompt_sha256", prompt)
        object.__setattr__(self, "scheduler_observation_sha256", scheduler)
        object.__setattr__(self, "probe_family", family)
        object.__setattr__(
            self,
            "probe_job_sha256",
            job if composite else self.probe_job_sha256,
        )

    @property
    def is_composite(self) -> bool:
        return self.probe_family == MLP_ALL_LAYER_AUTHORITY_FAMILY

    @property
    def measurement(self) -> MeasurementReceipt:
        return self.observation.measurement

    @property
    def source_receipt_sha256s(self) -> tuple[str, ...]:
        values = {
            self.measurement.sha256,
            self.observation.receipt.sha256,
            self.observation.receipt.atlas_revision.sha256,
            self.probe_spec.sha256,
            self.scheduler_observation_sha256,
        }
        if self.is_composite:
            assert self.probe_job_sha256 is not None
            values.add(self.probe_job_sha256)
        return tuple(sorted(values))


@dataclass(frozen=True, slots=True)
class _LayerReplay:
    access_trace_sha256: str
    operation_sha256s: tuple[str, ...]
    exact_by_prompt: Mapping[str, tuple[bool, bool, bool]]


@dataclass(frozen=True, slots=True)
class MlpCalibrationLock:
    """Calibration-only model inventory sealed before any holdout capture."""

    manifest_sha256: str
    model_pin_sha256: str
    prefix_state_sha256: str
    prefix_corpus_sha256: str
    prefix_fit_sha256: str
    config_sha256: str
    train_group_sha256s: tuple[str, ...]
    calibration_group_sha256s: tuple[str, ...]
    model_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in (
            "manifest_sha256",
            "model_pin_sha256",
            "prefix_state_sha256",
            "prefix_corpus_sha256",
            "prefix_fit_sha256",
            "config_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        for field, expected in (
            ("train_group_sha256s", 25),
            ("calibration_group_sha256s", 10),
        ):
            values = tuple(
                require_sha256(value, field=field) for value in getattr(self, field)
            )
            if len(values) != expected or len(set(values)) != len(values):
                raise ValueError(f"{field} has the wrong exact coverage")
            object.__setattr__(self, field, values)
        models = tuple(
            require_sha256(value, field="model_sha256s") for value in self.model_sha256s
        )
        if not models or len(set(models)) != len(models):
            raise ValueError("model_sha256s must be non-empty and unique")
        object.__setattr__(self, "model_sha256s", models)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_document(self) -> dict[str, object]:
        body = {
            "calibration_group_sha256s": list(self.calibration_group_sha256s),
            "config_sha256": self.config_sha256,
            "manifest_sha256": self.manifest_sha256,
            "model_pin_sha256": self.model_pin_sha256,
            "model_sha256s": list(self.model_sha256s),
            "prefix_corpus_sha256": self.prefix_corpus_sha256,
            "prefix_fit_sha256": self.prefix_fit_sha256,
            "prefix_state_sha256": self.prefix_state_sha256,
            "train_group_sha256s": list(self.train_group_sha256s),
        }
        return {
            "body": body,
            "body_sha256": _digest(body),
            "schema": MLP_CALIBRATION_LOCK_SCHEMA,
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_document())

    @classmethod
    def from_bytes(cls, data: bytes) -> "MlpCalibrationLock":
        if not isinstance(data, bytes) or not data or len(data) > 64 * 1024:
            raise QwenMlpEvidenceIntegrityError("calibration lock exceeds its bound")
        try:
            value = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise QwenMlpEvidenceIntegrityError("calibration lock is not JSON") from exc
        if (
            not isinstance(value, Mapping)
            or set(value) != {"body", "body_sha256", "schema"}
            or value.get("schema") != MLP_CALIBRATION_LOCK_SCHEMA
            or not isinstance(value.get("body"), Mapping)
            or value.get("body_sha256") != _digest(value.get("body"))
            or canonical_json_bytes(value) != data
        ):
            raise QwenMlpEvidenceIntegrityError("calibration lock seal is invalid")
        body = value["body"]
        expected = {
            "calibration_group_sha256s",
            "config_sha256",
            "manifest_sha256",
            "model_pin_sha256",
            "model_sha256s",
            "prefix_corpus_sha256",
            "prefix_fit_sha256",
            "prefix_state_sha256",
            "train_group_sha256s",
        }
        if (
            set(body) != expected
            or not isinstance(body.get("train_group_sha256s"), list)
            or not isinstance(body.get("calibration_group_sha256s"), list)
            or not isinstance(body.get("model_sha256s"), list)
        ):
            raise QwenMlpEvidenceIntegrityError("calibration lock body is invalid")
        try:
            result = cls(
                manifest_sha256=body["manifest_sha256"],
                model_pin_sha256=body["model_pin_sha256"],
                prefix_state_sha256=body["prefix_state_sha256"],
                prefix_corpus_sha256=body["prefix_corpus_sha256"],
                prefix_fit_sha256=body["prefix_fit_sha256"],
                config_sha256=body["config_sha256"],
                train_group_sha256s=tuple(body["train_group_sha256s"]),
                calibration_group_sha256s=tuple(body["calibration_group_sha256s"]),
                model_sha256s=tuple(body["model_sha256s"]),
            )
        except (TypeError, ValueError) as exc:
            raise QwenMlpEvidenceIntegrityError(
                "calibration lock reconstruction failed"
            ) from exc
        if result.to_bytes() != data:
            raise QwenMlpEvidenceIntegrityError(
                "calibration lock reconstruction changed"
            )
        return result

    @classmethod
    def create(
        cls,
        *,
        manifest: CaptureManifest,
        bank: QwenMlpEvidenceBank,
        config: SubspaceSweepConfig,
    ) -> tuple["MlpCalibrationLock", SubspaceBatteryFit, SubspaceCorpus]:
        state = bank.state()
        if state.split_counts != (25, 10, 0):
            raise QwenMlpEvidenceIntegrityError(
                "calibration lock requires exact 25/10/0 live coverage"
            )
        pairs = bank.committed_pairs()
        if len(pairs) != 35 or any(
            row.capture_mode != "live-exact" for row, _verification in pairs
        ):
            raise QwenMlpEvidenceIntegrityError(
                "calibration lock requires a live-only evidence bank"
            )
        corpus = bank.build_subspace_corpus()
        split_by_group = {
            receipt.identity_sha256: receipt.entry.split for receipt, _ in pairs
        }
        train = tuple(
            index
            for index, group in enumerate(corpus.groups)
            if split_by_group[group.group_sha256] == "train"
        )
        calibration = tuple(
            index
            for index, group in enumerate(corpus.groups)
            if split_by_group[group.group_sha256] == "calibration"
        )
        if train != tuple(range(25)) or calibration != tuple(range(25, 35)):
            raise QwenMlpEvidenceIntegrityError(
                "manifest capture chronology differs from the 25/10 lock"
            )
        fit = fit_subspace_battery(
            corpus,
            train_group_indices=train,
            calibration_group_indices=calibration,
            config=config,
        )
        lock = cls(
            manifest_sha256=manifest.sha256,
            model_pin_sha256=manifest.model_pin_sha256,
            prefix_state_sha256=state.sha256,
            prefix_corpus_sha256=corpus.sha256,
            prefix_fit_sha256=fit.sha256,
            config_sha256=_digest(config.to_record()),
            train_group_sha256s=tuple(
                corpus.groups[index].group_sha256 for index in train
            ),
            calibration_group_sha256s=tuple(
                corpus.groups[index].group_sha256 for index in calibration
            ),
            model_sha256s=tuple(model.sha256 for model in fit.models),
        )
        return lock, fit, corpus

    def verify_full_fit(
        self,
        *,
        manifest: CaptureManifest,
        bank: QwenMlpEvidenceBank,
        corpus: SubspaceCorpus,
        fit: SubspaceBatteryFit,
    ) -> None:
        if (
            self.manifest_sha256 != manifest.sha256
            or self.model_pin_sha256 != manifest.model_pin_sha256
            or bank.state().split_counts != (25, 10, 5)
            or self.prefix_state_sha256
            not in {state.sha256 for state in bank.state_history()}
            or self.config_sha256 != _digest(fit.config.to_record())
            or tuple(model.sha256 for model in fit.models) != self.model_sha256s
            or tuple(group.group_sha256 for group in corpus.groups[:25])
            != self.train_group_sha256s
            or tuple(group.group_sha256 for group in corpus.groups[25:35])
            != self.calibration_group_sha256s
        ):
            raise QwenMlpEvidenceIntegrityError(
                "full MLP fit differs from the pre-holdout calibration lock"
            )


@dataclass(frozen=True, slots=True)
class MlpLayerCalibrationSeal:
    """One layer-local 3/2 prompt fit sealed before v2 holdout capture."""

    layer: int
    prefix_corpus_sha256: str
    prefix_fit_sha256: str
    train_group_sha256s: tuple[str, ...]
    calibration_group_sha256s: tuple[str, ...]
    model_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer, bool)
            or not isinstance(self.layer, int)
            or not 0 <= self.layer < 64
        ):
            raise ValueError("calibration seal layer must lie in [0, 64)")
        for field in ("prefix_corpus_sha256", "prefix_fit_sha256"):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        for field, expected in (
            ("train_group_sha256s", 3),
            ("calibration_group_sha256s", 2),
        ):
            values = tuple(
                require_sha256(value, field=field) for value in getattr(self, field)
            )
            if len(values) != expected or len(set(values)) != expected:
                raise ValueError(f"{field} has the wrong layer-local coverage")
            object.__setattr__(self, field, values)
        models = tuple(
            require_sha256(value, field="model_sha256s")
            for value in self.model_sha256s
        )
        if not models or len(set(models)) != len(models):
            raise ValueError("model_sha256s must be non-empty and unique")
        object.__setattr__(self, "model_sha256s", models)

    def to_record(self) -> dict[str, object]:
        return {
            "calibration_group_sha256s": list(self.calibration_group_sha256s),
            "layer": self.layer,
            "model_sha256s": list(self.model_sha256s),
            "prefix_corpus_sha256": self.prefix_corpus_sha256,
            "prefix_fit_sha256": self.prefix_fit_sha256,
            "train_group_sha256s": list(self.train_group_sha256s),
        }

    @classmethod
    def from_record(cls, value: object) -> "MlpLayerCalibrationSeal":
        if (
            not isinstance(value, Mapping)
            or set(value)
            != {
                "calibration_group_sha256s",
                "layer",
                "model_sha256s",
                "prefix_corpus_sha256",
                "prefix_fit_sha256",
                "train_group_sha256s",
            }
            or not isinstance(value.get("train_group_sha256s"), list)
            or not isinstance(value.get("calibration_group_sha256s"), list)
            or not isinstance(value.get("model_sha256s"), list)
        ):
            raise QwenMlpEvidenceIntegrityError(
                "layer calibration seal is invalid"
            )
        try:
            return cls(
                layer=value["layer"],  # type: ignore[arg-type]
                prefix_corpus_sha256=value["prefix_corpus_sha256"],  # type: ignore[arg-type]
                prefix_fit_sha256=value["prefix_fit_sha256"],  # type: ignore[arg-type]
                train_group_sha256s=tuple(value["train_group_sha256s"]),
                calibration_group_sha256s=tuple(
                    value["calibration_group_sha256s"]
                ),
                model_sha256s=tuple(value["model_sha256s"]),
            )
        except (TypeError, ValueError) as exc:
            raise QwenMlpEvidenceIntegrityError(
                "layer calibration seal validation failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class MlpCalibrationLockV2:
    """All-layer calibration inventory sealed before later-generation holdout."""

    manifest_sha256: str
    model_pin_sha256: str
    prefix_state_sha256: str
    config_sha256: str
    layer_seals: tuple[MlpLayerCalibrationSeal, ...]

    def __post_init__(self) -> None:
        for field in (
            "manifest_sha256",
            "model_pin_sha256",
            "prefix_state_sha256",
            "config_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        seals = tuple(self.layer_seals)
        if (
            len(seals) != 64
            or any(not isinstance(row, MlpLayerCalibrationSeal) for row in seals)
            or tuple(row.layer for row in seals) != tuple(range(64))
        ):
            raise ValueError("v2 calibration lock requires ordered seals for 64 layers")
        object.__setattr__(self, "layer_seals", seals)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_document(self) -> dict[str, object]:
        body = {
            "config_sha256": self.config_sha256,
            "layer_seals": [row.to_record() for row in self.layer_seals],
            "manifest_sha256": self.manifest_sha256,
            "model_pin_sha256": self.model_pin_sha256,
            "prefix_state_sha256": self.prefix_state_sha256,
        }
        return {
            "body": body,
            "body_sha256": _digest(body),
            "schema": MLP_CALIBRATION_LOCK_V2_SCHEMA,
        }

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_document())
        if len(data) > 8 * 1024 * 1024:
            raise QwenMlpEvidenceIntegrityError(
                "v2 calibration lock exceeds its byte bound"
            )
        return data

    @classmethod
    def from_bytes(cls, data: bytes) -> "MlpCalibrationLockV2":
        if not isinstance(data, bytes) or not data or len(data) > 8 * 1024 * 1024:
            raise QwenMlpEvidenceIntegrityError(
                "v2 calibration lock exceeds its byte bound"
            )
        try:
            value = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise QwenMlpEvidenceIntegrityError(
                "v2 calibration lock is not JSON"
            ) from exc
        if (
            not isinstance(value, Mapping)
            or set(value) != {"body", "body_sha256", "schema"}
            or value.get("schema") != MLP_CALIBRATION_LOCK_V2_SCHEMA
            or not isinstance(value.get("body"), Mapping)
            or value.get("body_sha256") != _digest(value.get("body"))
            or canonical_json_bytes(value) != data
        ):
            raise QwenMlpEvidenceIntegrityError(
                "v2 calibration lock seal is invalid"
            )
        body = value["body"]
        if (
            set(body)
            != {
                "config_sha256",
                "layer_seals",
                "manifest_sha256",
                "model_pin_sha256",
                "prefix_state_sha256",
            }
            or not isinstance(body.get("layer_seals"), list)
        ):
            raise QwenMlpEvidenceIntegrityError(
                "v2 calibration lock body is invalid"
            )
        try:
            result = cls(
                manifest_sha256=body["manifest_sha256"],
                model_pin_sha256=body["model_pin_sha256"],
                prefix_state_sha256=body["prefix_state_sha256"],
                config_sha256=body["config_sha256"],
                layer_seals=tuple(
                    MlpLayerCalibrationSeal.from_record(row)
                    for row in body["layer_seals"]
                ),
            )
        except (TypeError, ValueError) as exc:
            raise QwenMlpEvidenceIntegrityError(
                "v2 calibration lock reconstruction failed"
            ) from exc
        if result.to_bytes() != data:
            raise QwenMlpEvidenceIntegrityError(
                "v2 calibration lock reconstruction changed"
            )
        return result

    @classmethod
    def create(
        cls,
        *,
        manifest: CaptureManifestV2,
        bank: QwenMlpEvidenceBank,
        config: SubspaceSweepConfig,
    ) -> tuple["MlpCalibrationLockV2", Mapping[int, SubspaceBatteryFit]]:
        if not isinstance(manifest, CaptureManifestV2):
            raise TypeError("v2 calibration lock requires CaptureManifestV2")
        if not isinstance(config, SubspaceSweepConfig):
            raise TypeError("config must be a SubspaceSweepConfig")
        state = bank.state()
        if state.split_counts != (192, 128, 0):
            raise QwenMlpEvidenceIntegrityError(
                "v2 calibration lock requires exact 192/128/0 live coverage"
            )
        pairs = bank.committed_pairs()
        if (
            len(pairs) != 320
            or any(row.capture_mode != "live-exact" for row, _ in pairs)
            or any(row.manifest_sha256 != manifest.sha256 for row, _ in pairs)
        ):
            raise QwenMlpEvidenceIntegrityError(
                "v2 calibration lock requires one live all-layer manifest"
            )
        split_by_group = {
            receipt.identity_sha256: receipt.entry.split for receipt, _ in pairs
        }
        fits: dict[int, SubspaceBatteryFit] = {}
        seals = []
        for layer in manifest.layers_for_split("train"):
            corpus = bank.build_subspace_corpus(
                allowed_splits=("train", "calibration"),
                allowed_layers=(layer,),
            )
            train = tuple(
                index
                for index, group in enumerate(corpus.groups)
                if split_by_group[group.group_sha256] == "train"
            )
            calibration = tuple(
                index
                for index, group in enumerate(corpus.groups)
                if split_by_group[group.group_sha256] == "calibration"
            )
            if (
                len(corpus.groups) != 5
                or train != (0, 1, 2)
                or calibration != (3, 4)
                or any(group.layer != layer for group in corpus.groups)
            ):
                raise QwenMlpEvidenceIntegrityError(
                    "v2 layer corpus differs from exact 3/2 chronology"
                )
            fit = fit_subspace_battery(
                corpus,
                train_group_indices=train,
                calibration_group_indices=calibration,
                config=config,
            )
            fits[layer] = fit
            seals.append(
                MlpLayerCalibrationSeal(
                    layer=layer,
                    prefix_corpus_sha256=corpus.sha256,
                    prefix_fit_sha256=fit.sha256,
                    train_group_sha256s=tuple(
                        corpus.groups[index].group_sha256 for index in train
                    ),
                    calibration_group_sha256s=tuple(
                        corpus.groups[index].group_sha256 for index in calibration
                    ),
                    model_sha256s=tuple(model.sha256 for model in fit.models),
                )
            )
        lock = cls(
            manifest_sha256=manifest.sha256,
            model_pin_sha256=manifest.model_pin_sha256,
            prefix_state_sha256=state.sha256,
            config_sha256=_digest(config.to_record()),
            layer_seals=tuple(seals),
        )
        return lock, fits

    def verify_full_fits(
        self,
        *,
        manifest: CaptureManifestV2,
        bank: QwenMlpEvidenceBank,
        fits: Mapping[int, SubspaceBatteryFit],
    ) -> None:
        if not isinstance(manifest, CaptureManifestV2):
            raise TypeError("v2 calibration verification requires CaptureManifestV2")
        if not isinstance(fits, Mapping):
            raise TypeError("fits must be a layer-to-fit mapping")
        if (
            self.manifest_sha256 != manifest.sha256
            or self.model_pin_sha256 != manifest.model_pin_sha256
            or bank.state().split_counts != (192, 128, 320)
            or self.prefix_state_sha256
            not in {state.sha256 for state in bank.state_history()}
            or set(fits) != set(range(64))
        ):
            raise QwenMlpEvidenceIntegrityError(
                "full v2 MLP fit differs from its calibration lock"
            )
        pairs = bank.committed_pairs()
        if (
            len(pairs) != 640
            or any(row.capture_mode != "live-exact" for row, _ in pairs)
            or any(row.manifest_sha256 != manifest.sha256 for row, _ in pairs)
            or any(row.model_pin_sha256 != manifest.model_pin_sha256 for row, _ in pairs)
        ):
            raise QwenMlpEvidenceIntegrityError(
                "full v2 calibration verification requires 640 live groups"
            )
        split_by_group = {
            receipt.identity_sha256: receipt.entry.split for receipt, _ in pairs
        }
        for seal in self.layer_seals:
            corpus = bank.build_subspace_corpus(allowed_layers=(seal.layer,))
            split_indices = {
                split: tuple(
                    index
                    for index, group in enumerate(corpus.groups)
                    if split_by_group[group.group_sha256] == split
                )
                for split in ("train", "calibration", "holdout")
            }
            fit = fits[seal.layer]
            if (
                len(corpus.groups) != 10
                or split_indices["train"] != (0, 1, 2)
                or split_indices["calibration"] != (3, 4)
                or split_indices["holdout"] != (5, 6, 7, 8, 9)
                or self.config_sha256 != _digest(fit.config.to_record())
                or tuple(model.sha256 for model in fit.models)
                != seal.model_sha256s
                or tuple(
                    corpus.groups[index].group_sha256
                    for index in split_indices["train"]
                )
                != seal.train_group_sha256s
                or tuple(
                    corpus.groups[index].group_sha256
                    for index in split_indices["calibration"]
                )
                != seal.calibration_group_sha256s
            ):
                raise QwenMlpEvidenceIntegrityError(
                    "full v2 layer fit differs from its 3/2/5 calibration seal"
                )
            fit.verify_against(corpus)


class LiveExactMlpCaptureRunner:
    """Split-closed live runner for layer-local v1 and full-span v2 authority."""

    capture_mode = "live-exact"

    def __init__(
        self,
        *,
        manifest: CaptureManifest | CaptureManifestV2,
        model: StreamedQwen38,
        atlas: SemanticWeightAtlas,
        authorities: Mapping[tuple[str, int], LiveMlpAuthority],
        projection: HiddenSketchProjection,
        max_sketch_elements: int = 16 * 1024**2,
        max_resident_capture_bytes: int = 2 * 1024**3,
    ) -> None:
        if not isinstance(manifest, (CaptureManifest, CaptureManifestV2)):
            raise TypeError("manifest must be a CaptureManifest or CaptureManifestV2")
        if not isinstance(model, StreamedQwen38):
            raise TypeError("model must be a StreamedQwen38")
        if not isinstance(atlas, SemanticWeightAtlas):
            raise TypeError("atlas must be a SemanticWeightAtlas")
        if not isinstance(projection, HiddenSketchProjection):
            raise TypeError("projection must be a HiddenSketchProjection")
        if (
            isinstance(max_sketch_elements, bool)
            or not isinstance(max_sketch_elements, int)
            or max_sketch_elements < 1
        ):
            raise ValueError("max_sketch_elements must be positive")
        if (
            isinstance(max_resident_capture_bytes, bool)
            or not isinstance(max_resident_capture_bytes, int)
            or max_resident_capture_bytes < 1
        ):
            raise ValueError("max_resident_capture_bytes must be positive")
        if any(
            attachment is not None
            for attachment in (
                model.graft,
                model.native_head_crsa,
                model.layer_boundary_observer,
            )
        ):
            raise ValueError("live exact MLP capture requires a passive parent Qwen")
        expected = {(row.prompt_sha256, row.layer) for row in manifest.entries}
        supplied = dict(authorities)
        if set(supplied) != expected:
            raise ValueError("live MLP authority inventory differs from the manifest")
        if atlas.model_pin.sha256 != manifest.model_pin_sha256:
            raise QwenMlpEvidenceIntegrityError(
                "Atlas model pin differs from the MLP capture manifest"
            )
        if max(layer for _prompt, layer in expected) >= model.config.n_layers:
            raise ValueError("capture plan exceeds the local Qwen depth")
        atlas.verify_or_raise()
        active_by_signature: dict[str, object] = {}
        for key in sorted(supplied):
            authority = supplied[key]
            if not isinstance(authority, LiveMlpAuthority) or key != (
                authority.prompt_sha256,
                authority.layer,
            ):
                raise TypeError("live MLP authority key/value is invalid")
            if authority.measurement.model_pin.sha256 != manifest.model_pin_sha256:
                raise QwenMlpEvidenceIntegrityError(
                    "live MLP authority crosses the manifest model pin"
                )
            if authority.observation.receipt.input_abi.trailing_shape != (
                projection.output_dimensions,
            ):
                raise QwenMlpEvidenceIntegrityError(
                    "Harvester MLP sketch ABI differs from the capture projection"
                )
            if not atlas.contains_revision(
                authority.observation.receipt.atlas_revision
            ):
                raise QwenMlpEvidenceIntegrityError(
                    "MLP authority Atlas revision is not in authenticated history"
                )
            signature = authority.measurement.probe.prompt_signature
            active = active_by_signature.get(signature)
            if active is None:
                active = atlas.query_by_prompt_signature(signature)
                active_by_signature[signature] = active
            matches = tuple(
                row
                for row in active.measurements
                if row.sha256 == authority.measurement.sha256
                and row.to_document() == authority.measurement.to_document()
            )
            if len(matches) != 1:
                raise QwenMlpEvidenceIntegrityError(
                    "MLP authority measurement is no longer active in the Atlas"
                )
        if manifest.is_all_layer_v2:
            if any(not authority.is_composite for authority in supplied.values()):
                raise QwenMlpEvidenceIntegrityError(
                    "all-layer manifest requires composite O1 authorities"
                )
            context_receipts = tuple(
                authority.observation.receipt.sha256
                for authority in supplied.values()
            )
            if len(set(context_receipts)) != len(context_receipts):
                raise QwenMlpEvidenceIntegrityError(
                    "composite MLP authority reuses a context receipt"
                )
            for split in ("train", "calibration", "holdout"):
                layers = manifest.layers_for_split(split)
                if layers != tuple(range(model.config.n_layers)):
                    raise QwenMlpEvidenceIntegrityError(
                        "all-layer manifest differs from the local Qwen depth"
                    )
                for prompt in manifest.prompts_for_split(split):
                    rows = tuple(supplied[(prompt, layer)] for layer in layers)
                    first = rows[0]
                    if any(
                        row.probe_spec.sha256 != first.probe_spec.sha256
                        or row.measurement.sha256 != first.measurement.sha256
                        or row.scheduler_observation_sha256
                        != first.scheduler_observation_sha256
                        or row.probe_job_sha256 != first.probe_job_sha256
                        for row in rows[1:]
                    ):
                        raise QwenMlpEvidenceIntegrityError(
                            "one prompt's composite authorities do not share one probe"
                        )
                    if (
                        first.probe_spec.start_layer != 0
                        or first.probe_spec.stop_layer != model.config.n_layers
                        or first.probe_spec.coordinate.layer
                        != model.config.n_layers - 1
                    ):
                        raise QwenMlpEvidenceIntegrityError(
                            "composite MLP authority is not the local full span"
                        )
            self.atomic_capture_group_size = model.config.n_layers
        else:
            if any(authority.is_composite for authority in supplied.values()):
                raise QwenMlpEvidenceIntegrityError(
                    "v1 manifest cannot consume composite MLP authorities"
                )
            self.atomic_capture_group_size = len(manifest.prompt_sha256s)
        self.manifest = manifest
        self.model = model
        self.atlas = atlas
        self.authorities = supplied
        self.projection = projection
        self.max_sketch_elements = max_sketch_elements
        self.max_resident_capture_bytes = max_resident_capture_bytes
        self._captures: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
        self._executed_phases: set[tuple[str, str]] = set()
        self._layer_replays: dict[tuple[str, int], _LayerReplay] = {}
        self._pending: dict[
            str, tuple[CapturePlanEntry, dict[str, torch.Tensor], LiveMlpAuthority]
        ] = {}
        self._verifications: dict[str, MlpProjectionVerificationReceipt] = {}
        source_root = Path(__file__).resolve().parent
        self.verifier_sha256 = _digest(
            {
                "algorithm": {
                    "capture": "manifest-split-prompt-partial-forward",
                    "input_sketch": "cartography-rademacher-float64",
                    "projection_replay": "pager.linear_many+swiglu-bit-exact",
                    "storage": "content-addressed-source-dtype-bits",
                },
                "manifest_sha256": manifest.sha256,
                "projection": projection.as_record(),
                "qwen_runtime_sources": runtime_source_manifest(),
                "runner_source_sha256": _source_sha256(Path(__file__).resolve()),
                "schema": LIVE_MLP_VERIFIER_SCHEMA,
                "storage_source_sha256": _source_sha256(
                    source_root / "qwen_mlp_evidence.py"
                ),
            }
        )

    def _authority(self, prompt_sha256: str, layer: int) -> LiveMlpAuthority:
        try:
            return self.authorities[(prompt_sha256, layer)]
        except KeyError as exc:  # pragma: no cover - constructor sealed inventory.
            raise QwenMlpEvidenceIntegrityError(
                "capture entry has no live MLP authority"
            ) from exc

    def atomic_capture_group(self, entry: CapturePlanEntry) -> str:
        """Keep the units supplied by one bounded capture phase atomic."""

        if self.manifest.is_all_layer_v2:
            return f"{entry.split}:{entry.prompt_sha256}"
        return f"{entry.split}:{entry.layer}"

    def _execute_phase(
        self,
        split: str,
        prompt_sha256: str,
        bank: QwenMlpEvidenceBank,
    ) -> None:
        phase = (split, prompt_sha256)
        if phase in self._executed_phases:
            return
        layers = self.manifest.layers_for_split(split)
        if prompt_sha256 not in self.manifest.prompts_for_split(split):
            raise QwenMlpEvidenceIntegrityError(
                "live prompt is outside its manifest split binding"
            )
        authority = self._authority(prompt_sha256, layers[0])
        token_ids = authority.probe_spec.prompt_token_ids
        if prompt_token_sha256(token_ids) != prompt_sha256:
            raise QwenMlpEvidenceIntegrityError("live prompt token hash changed")
        element_size = torch.empty(
            (), dtype=self.model.pager.compute_dtype
        ).element_size()
        planned_phase_bytes = (
            len(token_ids)
            * len(layers)
            * (2 * self.model.config.dim + 2 * self.model.config.intermediate_size)
            * element_size
        )
        resident_bytes = sum(
            value.numel() * value.element_size()
            for rows in self._captures.values()
            for value in rows.values()
        )
        if resident_bytes + planned_phase_bytes > self.max_resident_capture_bytes:
            raise QwenMlpEvidenceIntegrityError(
                "live MLP phase exceeds the runner resident-memory budget"
            )
        captured: dict[int, dict[str, torch.Tensor]] = {}
        stage_order: dict[int, list[str]] = {}

        def observer(layer: int, stage: str, value: torch.Tensor) -> None:
            rows = captured.setdefault(layer, {})
            order = stage_order.setdefault(layer, [])
            expected = CAPTURE_STAGES[len(order)] if len(order) < 4 else None
            if stage != expected or stage in rows:
                raise QwenMlpEvidenceIntegrityError(
                    "live Qwen emitted an invalid MLP boundary order"
                )
            tensor = value.detach().contiguous().to(device="cpu")
            if not bool(torch.isfinite(tensor).all().item()):
                raise QwenMlpEvidenceIntegrityError(
                    "live Qwen emitted non-finite MLP evidence"
                )
            rows[stage] = tensor
            order.append(stage)

        child = StreamedQwen38(
            self.model.config,
            self.model.pager,
            layer_boundary_observer=observer,
            layer_boundary_stages=CAPTURE_STAGES,
            layer_boundary_layers=layers,
            max_batch_size=1,
            max_seq_len=len(token_ids),
        )
        ids = torch.tensor([token_ids], dtype=torch.long)
        hidden = child.embed_batch(ids)
        try:
            for layer in range(max(layers) + 1):
                hidden, _ = child.forward_prefill_layer(hidden, ids, layer=layer)
                child.pager.release()
        finally:
            child.pager.release()
        if set(captured) != set(layers) or any(
            tuple(rows) != CAPTURE_STAGES for rows in captured.values()
        ):
            raise QwenMlpEvidenceIntegrityError(
                "live Qwen MLP phase capture is incomplete"
            )
        for layer, rows in captured.items():
            planned = sum(
                bank.tensor_nbytes(rows[stage])[1] for stage in CAPTURE_STAGES
            )
            if planned > bank.budget.max_group_bytes:
                raise QwenMlpEvidenceIntegrityError(
                    "live MLP group exceeds the sealed bank budget"
                )
            self._captures[(prompt_sha256, layer)] = rows
        self._executed_phases.add(phase)

    def _ensure_layer_replay(
        self, layer: int, split: str, bank: QwenMlpEvidenceBank
    ) -> _LayerReplay:
        replay_key = (split, layer)
        existing = self._layer_replays.get(replay_key)
        if existing is not None:
            return existing
        if layer not in self.manifest.layers_for_split(split):
            raise QwenMlpEvidenceIntegrityError(
                "MLP replay layer is outside its manifest split"
            )
        prompts = self.manifest.prompts_for_split(split)
        for prompt in prompts:
            self._execute_phase(split, prompt, bank)
        resident_bytes = sum(
            value.numel() * value.element_size()
            for rows in self._captures.values()
            for value in rows.values()
        )
        replay_bytes = (
            sum(
                len(self._authority(prompt, layer).probe_spec.prompt_token_ids)
                for prompt in prompts
            )
            * (3 * self.model.config.intermediate_size + self.model.config.dim)
            * torch.empty((), dtype=self.model.pager.compute_dtype).element_size()
        )
        if resident_bytes + replay_bytes > self.max_resident_capture_bytes:
            raise QwenMlpEvidenceIntegrityError(
                "live MLP replay exceeds the runner resident-memory budget"
            )
        inputs = tuple(
            self._captures[(prompt, layer)]["mlp.input"] for prompt in prompts
        )
        expected_gate = tuple(
            self._captures[(prompt, layer)]["mlp.gate"] for prompt in prompts
        )
        expected_up = tuple(
            self._captures[(prompt, layer)]["mlp.up"] for prompt in prompts
        )
        expected_output = tuple(
            self._captures[(prompt, layer)]["mlp.output"] for prompt in prompts
        )
        base = f"model.language_model.layers.{layer}.mlp"
        recorder = AccessTraceRecorder(max_operations=16, max_leaves=64)
        source = self.model.pager.source
        previous = source.set_access_observer(recorder, prepare_identity=True)
        try:
            gate = self.model.pager.linear_many(inputs, f"{base}.gate_proj")
            up = self.model.pager.linear_many(inputs, f"{base}.up_proj")
            activated = tuple(
                swiglu(left, right) for left, right in zip(gate, up, strict=True)
            )
            output = self.model.pager.linear_many(activated, f"{base}.down_proj")
            trace = recorder.snapshot()
        finally:
            restored = source.set_access_observer(previous, prepare_identity=False)
            self.model.pager.release()
        if restored is not recorder:
            raise QwenMlpEvidenceIntegrityError(
                "Qwen source observer changed during MLP replay"
            )
        metrics = recorder.metrics()
        if metrics["dropped_capacity"] or metrics["dropped_identity"]:
            raise QwenMlpEvidenceIntegrityError("MLP replay access trace dropped reads")
        exact = {
            prompt: (
                torch.equal(gate[index].to("cpu"), expected_gate[index]),
                torch.equal(up[index].to("cpu"), expected_up[index]),
                torch.equal(output[index].to("cpu"), expected_output[index]),
            )
            for index, prompt in enumerate(prompts)
        }
        logical_trace_sha256, operation_sha256s = _logical_access_trace(trace)
        replay = _LayerReplay(logical_trace_sha256, operation_sha256s, exact)
        self._layer_replays[replay_key] = replay
        return replay

    def capture(
        self, entry: CapturePlanEntry, sink: ExactMlpBoundaryCapture
    ) -> MlpEvidenceReceipt:
        if entry not in self.manifest.entries:
            raise QwenMlpEvidenceIntegrityError(
                "capture entry is outside the sealed manifest"
            )
        self._execute_phase(entry.split, entry.prompt_sha256, sink.bank)
        values = self._captures[(entry.prompt_sha256, entry.layer)]
        authority = self._authority(entry.prompt_sha256, entry.layer)
        sketch_record, sketch_array = project_hidden_sketch(
            values["mlp.input"],
            self.projection,
            max_elements=self.max_sketch_elements,
        )
        recomputed_sketch = tensor_sha256(
            sketch_array, authority.observation.receipt.input_abi
        )
        if (
            recomputed_sketch != authority.observation.receipt.input_sha256
            or sketch_record["seed_sha256"] != self.projection.seed_sha256
        ):
            raise QwenMlpEvidenceIntegrityError(
                "full live MLP input does not reproduce its O1 cartography sketch"
            )
        sink.begin_group(entry)
        for stage in CAPTURE_STAGES:
            sink(entry.layer, stage, values[stage])
        atlas_revision = authority.observation.receipt.atlas_revision
        receipt = sink.finalize(
            capture_mode=self.capture_mode,
            manifest_sha256=self.manifest.sha256,
            model_pin_sha256=self.manifest.model_pin_sha256,
            token_sha256=entry.prompt_sha256,
            probe_spec_sha256=authority.probe_spec.sha256,
            atlas_sequence=atlas_revision.sequence,
            atlas_event_sha256=atlas_revision.event_sha256,
            atlas_revision_sha256=atlas_revision.sha256,
            measurement_sha256=authority.measurement.sha256,
            weight_revision_sha256=authority.measurement.weight_rail_revision.sha256,
            access_trace_sha256=authority.measurement.access_trace_sha256,
            source_receipt_sha256s=authority.source_receipt_sha256s,
            cartography_input_sketch_sha256=(
                authority.observation.receipt.input_sha256
            ),
            recomputed_input_sketch_sha256=recomputed_sketch,
        )
        self._pending[receipt.sha256] = (entry, values, authority)
        return receipt

    def verify(
        self, receipt: MlpEvidenceReceipt, bank: QwenMlpEvidenceBank
    ) -> MlpProjectionVerificationReceipt:
        cached = self._verifications.get(receipt.sha256)
        if cached is not None:
            return cached
        try:
            entry, values, authority = self._pending[receipt.sha256]
        except KeyError as exc:
            raise QwenMlpEvidenceIntegrityError(
                "live verifier did not capture this receipt"
            ) from exc
        if (
            receipt.entry != entry
            or receipt.probe_spec_sha256 != authority.probe_spec.sha256
            or receipt.measurement_sha256 != authority.measurement.sha256
            or not self.atlas.contains_revision(
                authority.observation.receipt.atlas_revision
            )
        ):
            raise QwenMlpEvidenceIntegrityError(
                "live MLP receipt authority changed before verification"
            )
        replay = self._ensure_layer_replay(entry.layer, entry.split, bank)
        gate_exact, up_exact, output_exact = replay.exact_by_prompt[entry.prompt_sha256]
        refs = {row.stage: row for row in receipt.tensors}
        storage_exact = all(
            bank.publish_tensor(stage, entry.layer, values[stage]) == refs[stage]
            for stage in CAPTURE_STAGES
        )
        _record, sketch_array = project_hidden_sketch(
            values["mlp.input"],
            self.projection,
            max_elements=self.max_sketch_elements,
        )
        recomputed_sketch = tensor_sha256(
            sketch_array, authority.observation.receipt.input_abi
        )
        input_sketch_exact = (
            recomputed_sketch
            == receipt.cartography_input_sketch_sha256
            == authority.observation.receipt.input_sha256
        )
        proof_sha256 = bank.publish_verifier_evidence(
            evidence_receipt_sha256=receipt.sha256,
            verifier_sha256=self.verifier_sha256,
            evidence={
                "atlas_revision_sha256": receipt.atlas_revision_sha256,
                "gate_exact": gate_exact,
                "input_sketch_exact": input_sketch_exact,
                "measurement_sha256": receipt.measurement_sha256,
                "operation_sha256s": list(replay.operation_sha256s),
                "output_exact": output_exact,
                "probe_spec_sha256": receipt.probe_spec_sha256,
                "replay_access_trace_sha256": replay.access_trace_sha256,
                "storage_exact": storage_exact,
                "tensor_object_sha256s": [row.object_sha256 for row in receipt.tensors],
                "up_exact": up_exact,
            },
        )
        verification = MlpProjectionVerificationReceipt(
            evidence_receipt_sha256=receipt.sha256,
            verifier_sha256=self.verifier_sha256,
            verifier_evidence_sha256=proof_sha256,
            replay_access_trace_sha256=replay.access_trace_sha256,
            storage_exact=storage_exact,
            input_sketch_exact=input_sketch_exact,
            gate_exact=gate_exact,
            up_exact=up_exact,
            output_exact=output_exact,
        )
        self._verifications[receipt.sha256] = verification
        self._pending.pop(receipt.sha256, None)
        self._captures.pop((entry.prompt_sha256, entry.layer), None)
        return verification


__all__ = [
    "LIVE_MLP_VERIFIER_SCHEMA",
    "MLP_ALL_LAYER_AUTHORITY_FAMILY",
    "MLP_CALIBRATION_LOCK_SCHEMA",
    "MLP_CALIBRATION_LOCK_V2_SCHEMA",
    "LiveExactMlpCaptureRunner",
    "LiveMlpAuthority",
    "MlpCalibrationLock",
    "MlpCalibrationLockV2",
    "MlpLayerCalibrationSeal",
]
