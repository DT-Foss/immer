"""Receipt-native training input for the IMMER Seed shadow proposer.

The source receipts remain the authority.  This module only projects their
content-addressed identities and explicitly selected numeric views into a
small, replayable training ABI.  It never stores prompts and never infers the
meaning of a numeric field: callers pin every vector schema and supply the
projection function that implements it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Literal, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from immer.runtimes.qwen3_8.semantic_atlas import MeasurementReceipt

from .demand_execution import DemandRoutedExecutionReceipt
from .execution_learning import ExecutionLearningReceipt
from .identity import canonical_json_bytes, require_sha256


SEED_TRAINING_PROJECTION_SCHEMA = "immer-ooe-seed-training-projection/v1"
SEED_TRAINING_BATCH_SCHEMA = "immer-ooe-seed-training-batch/v1"
SEED_TRAINING_MANIFEST_SCHEMA = "immer-seed-v3-training-batch-manifest/v1"
SEED_PROMOTION_RISK_SCHEMA = "immer-ooe-seed-group-promotion-risk/v1"
SEED_VECTOR_CONTRACT_SCHEMA = "immer-ooe-seed-vector-contract/v1"

SeedSplit = Literal["train", "calibration", "holdout"]
SEED_SPLITS: tuple[SeedSplit, ...] = ("train", "calibration", "holdout")
_SPLIT_RANK = {name: index for index, name in enumerate(SEED_SPLITS)}
_SOURCE_KINDS = frozenset(
    {"measurement", "execution-learning", "demand-routed-execution"}
)
_VECTOR_NAMES = ("feature", "action", "consequence", "quality", "work_cost")
MAX_VECTOR_DIMENSIONS = 65_536
MAX_HASH_INVENTORY = 4_096
MAX_BATCH_PROJECTIONS = 1_000_000
MAX_PROJECTION_BYTES = 4 * 1024 * 1024
MAX_BATCH_BYTES = 256 * 1024 * 1024
VECTOR_ABS_BOUND = 1.0


class SeedProjectionError(ValueError):
    """A source receipt cannot enter the Seed training ABI."""


class SeedProjectionIntegrityError(SeedProjectionError):
    """A sealed projection, batch, manifest, or risk receipt was modified."""


class SeedProjectionSplitError(SeedProjectionError):
    """Group or prompt-generation leakage crosses a frozen split."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _bytes_digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _uint(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SeedProjectionError(f"{field} must be a non-negative integer")
    return value


def _probability(value: object, *, field: str, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SeedProjectionError(f"{field} must be finite probability data")
    result = float(value)
    lower = 0.0 < result if strict else 0.0 <= result
    upper = result < 1.0 if strict else result <= 1.0
    if not math.isfinite(result) or not lower or not upper:
        interval = "(0, 1)" if strict else "[0, 1]"
        raise SeedProjectionError(f"{field} must lie in {interval}")
    return result


def _hashes(
    values: Sequence[str],
    *,
    field: str,
    required: bool = True,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise SeedProjectionError(f"{field} must be a SHA-256 sequence")
    try:
        normalized = tuple(
            sorted({require_sha256(value, field=field) for value in values})
        )
    except (TypeError, ValueError) as exc:
        raise SeedProjectionError(f"{field} must be a SHA-256 sequence") from exc
    if (required and not normalized) or len(normalized) > MAX_HASH_INVENTORY:
        lower = 1 if required else 0
        raise SeedProjectionError(
            f"{field} must contain {lower}..{MAX_HASH_INVENTORY} unique hashes"
        )
    return normalized


def _vector(values: Sequence[float], *, field: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise SeedProjectionError(f"{field} must be a numeric sequence")
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SeedProjectionError(f"{field} must be a numeric sequence") from exc
    if not result or len(result) > MAX_VECTOR_DIMENSIONS:
        raise SeedProjectionError(
            f"{field} must contain 1..{MAX_VECTOR_DIMENSIONS} values"
        )
    if any(
        not math.isfinite(value) or abs(value) > VECTOR_ABS_BOUND
        for value in result
    ):
        raise SeedProjectionError(
            f"{field} must contain finite values in "
            f"[-{VECTOR_ABS_BOUND}, {VECTOR_ABS_BOUND}]"
        )
    return tuple(0.0 if value == 0.0 else value for value in result)


def _strict_json(data: bytes, *, label: str, maximum: int) -> object:
    if not isinstance(data, bytes):
        raise TypeError(f"{label} must be immutable bytes")
    if not data or len(data) > maximum:
        raise SeedProjectionIntegrityError(f"{label} exceeds its byte bound")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SeedProjectionIntegrityError(f"{label} is not strict JSON") from exc
    if canonical_json_bytes(value) != data:
        raise SeedProjectionIntegrityError(f"{label} is not canonical JSON")
    return value


def _sealed_body(value: object, *, schema: str, label: str) -> Mapping[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema", "body", "body_sha256"}
        or value.get("schema") != schema
    ):
        raise SeedProjectionIntegrityError(f"invalid {label} envelope")
    body = value.get("body")
    if not isinstance(body, Mapping):
        raise SeedProjectionIntegrityError(f"invalid {label} body")
    try:
        claimed = require_sha256(value.get("body_sha256"), field="body_sha256")
    except ValueError as exc:
        raise SeedProjectionIntegrityError(f"invalid {label} body hash") from exc
    if claimed != _digest(body):
        raise SeedProjectionIntegrityError(f"{label} body hash mismatch")
    return body


def _seal(schema: str, body: Mapping[str, object]) -> dict[str, object]:
    normalized = cast(dict[str, object], json.loads(canonical_json_bytes(body)))
    return {"schema": schema, "body": normalized, "body_sha256": _digest(normalized)}


@dataclass(frozen=True, slots=True)
class SeedVectorProjection:
    """Explicit fixed-schema numeric views selected by the caller."""

    feature_schema_sha256: str
    feature: tuple[float, ...]
    action_schema_sha256: str
    action: tuple[float, ...]
    consequence_schema_sha256: str
    consequence: tuple[float, ...]
    quality_schema_sha256: str
    quality: tuple[float, ...]
    work_cost_schema_sha256: str
    work_cost: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in _VECTOR_NAMES:
            schema_field = f"{name}_schema_sha256"
            object.__setattr__(
                self,
                schema_field,
                require_sha256(getattr(self, schema_field), field=schema_field),
            )
            object.__setattr__(
                self,
                name,
                _vector(getattr(self, name), field=name),
            )

    @property
    def contract(self) -> dict[str, object]:
        return {
            "schema": SEED_VECTOR_CONTRACT_SCHEMA,
            **{
                name: {
                    "dimensions": len(getattr(self, name)),
                    "schema_sha256": getattr(self, f"{name}_schema_sha256"),
                }
                for name in _VECTOR_NAMES
            },
        }

    @property
    def contract_sha256(self) -> str:
        return _digest(self.contract)

    def to_dict(self) -> dict[str, object]:
        return {
            name: {
                "schema_sha256": getattr(self, f"{name}_schema_sha256"),
                "values": list(getattr(self, name)),
            }
            for name in _VECTOR_NAMES
        }

    @classmethod
    def from_dict(cls, value: object) -> "SeedVectorProjection":
        if not isinstance(value, Mapping) or set(value) != set(_VECTOR_NAMES):
            raise SeedProjectionIntegrityError("vector projection schema is invalid")
        arguments: dict[str, object] = {}
        for name in _VECTOR_NAMES:
            row = value.get(name)
            if not isinstance(row, Mapping) or set(row) != {
                "schema_sha256",
                "values",
            }:
                raise SeedProjectionIntegrityError(
                    f"{name} vector projection schema is invalid"
                )
            arguments[f"{name}_schema_sha256"] = row.get("schema_sha256")
            arguments[name] = row.get("values")
        try:
            return cls(**arguments)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise SeedProjectionIntegrityError(
                "vector projection failed validation"
            ) from exc


ReceiptT = TypeVar("ReceiptT")
VectorProjector = Callable[[ReceiptT], SeedVectorProjection]


def _project_vectors(
    receipt: ReceiptT,
    projector: SeedVectorProjection | VectorProjector[ReceiptT],
) -> SeedVectorProjection:
    value = projector(receipt) if callable(projector) else projector
    if not isinstance(value, SeedVectorProjection):
        raise TypeError("vector_projector must return SeedVectorProjection")
    return value


@dataclass(frozen=True, slots=True)
class SeedTrainingProjection:
    """One canonical consequence-training row backed by exact source receipts."""

    source_kind: str
    source_receipt_sha256s: tuple[str, ...]
    model_pin_sha256: str
    code_identity_sha256: str
    site_identity_sha256s: tuple[str, ...]
    graph_revision_sha256s: tuple[str, ...]
    authority_sha256s: tuple[str, ...]
    verifier_sha256s: tuple[str, ...]
    split: SeedSplit | str
    generation: int
    prompt_generation: int
    group_id_sha256: str
    sequence_index: int
    vectors: SeedVectorProjection

    def __post_init__(self) -> None:
        if self.source_kind not in _SOURCE_KINDS:
            raise SeedProjectionError("source_kind is not a receipt-native source")
        for field in ("model_pin_sha256", "code_identity_sha256", "group_id_sha256"):
            object.__setattr__(
                self,
                field,
                require_sha256(getattr(self, field), field=field),
            )
        for field in (
            "source_receipt_sha256s",
            "site_identity_sha256s",
            "graph_revision_sha256s",
            "authority_sha256s",
            "verifier_sha256s",
        ):
            object.__setattr__(
                self,
                field,
                _hashes(getattr(self, field), field=field),
            )
        if self.split not in SEED_SPLITS:
            raise SeedProjectionError("split must be train, calibration, or holdout")
        object.__setattr__(self, "generation", _uint(self.generation, field="generation"))
        object.__setattr__(
            self,
            "prompt_generation",
            _uint(self.prompt_generation, field="prompt_generation"),
        )
        object.__setattr__(
            self,
            "sequence_index",
            _uint(self.sequence_index, field="sequence_index"),
        )
        if not isinstance(self.vectors, SeedVectorProjection):
            raise TypeError("vectors must be a SeedVectorProjection")

    @property
    def feature_schema_sha256(self) -> str:
        return self.vectors.feature_schema_sha256

    @property
    def feature_dimensions(self) -> int:
        return len(self.vectors.feature)

    @property
    def feature_vector(self) -> tuple[float, ...]:
        return self.vectors.feature

    @property
    def vector_contract_sha256(self) -> str:
        return self.vectors.contract_sha256

    def body(self) -> dict[str, object]:
        return {
            "authority_sha256s": list(self.authority_sha256s),
            "code_identity_sha256": self.code_identity_sha256,
            "generation": self.generation,
            "graph_revision_sha256s": list(self.graph_revision_sha256s),
            "group_id_sha256": self.group_id_sha256,
            "model_pin_sha256": self.model_pin_sha256,
            "prompt_generation": self.prompt_generation,
            "sequence_index": self.sequence_index,
            "site_identity_sha256s": list(self.site_identity_sha256s),
            "source_kind": self.source_kind,
            "source_receipt_sha256s": list(self.source_receipt_sha256s),
            "split": self.split,
            "vector_contract_sha256": self.vector_contract_sha256,
            "vectors": self.vectors.to_dict(),
            "verifier_sha256s": list(self.verifier_sha256s),
        }

    def to_dict(self) -> dict[str, object]:
        return _seal(SEED_TRAINING_PROJECTION_SCHEMA, self.body())

    def to_bytes(self) -> bytes:
        result = canonical_json_bytes(self.to_dict())
        if len(result) > MAX_PROJECTION_BYTES:
            raise SeedProjectionError("Seed training projection exceeds its byte bound")
        return result

    @property
    def sha256(self) -> str:
        return _bytes_digest(self.to_bytes())

    @classmethod
    def from_bytes(cls, data: bytes) -> "SeedTrainingProjection":
        value = _strict_json(
            data,
            label="Seed training projection",
            maximum=MAX_PROJECTION_BYTES,
        )
        body = _sealed_body(
            value,
            schema=SEED_TRAINING_PROJECTION_SCHEMA,
            label="Seed training projection",
        )
        expected = {
            "authority_sha256s",
            "code_identity_sha256",
            "generation",
            "graph_revision_sha256s",
            "group_id_sha256",
            "model_pin_sha256",
            "prompt_generation",
            "sequence_index",
            "site_identity_sha256s",
            "source_kind",
            "source_receipt_sha256s",
            "split",
            "vector_contract_sha256",
            "vectors",
            "verifier_sha256s",
        }
        if set(body) != expected:
            raise SeedProjectionIntegrityError(
                "Seed training projection body is invalid"
            )
        try:
            vectors = SeedVectorProjection.from_dict(body.get("vectors"))
            result = cls(
                source_kind=cast(str, body.get("source_kind")),
                source_receipt_sha256s=tuple(
                    cast(Sequence[str], body.get("source_receipt_sha256s"))
                ),
                model_pin_sha256=cast(str, body.get("model_pin_sha256")),
                code_identity_sha256=cast(str, body.get("code_identity_sha256")),
                site_identity_sha256s=tuple(
                    cast(Sequence[str], body.get("site_identity_sha256s"))
                ),
                graph_revision_sha256s=tuple(
                    cast(Sequence[str], body.get("graph_revision_sha256s"))
                ),
                authority_sha256s=tuple(
                    cast(Sequence[str], body.get("authority_sha256s"))
                ),
                verifier_sha256s=tuple(
                    cast(Sequence[str], body.get("verifier_sha256s"))
                ),
                split=cast(str, body.get("split")),
                generation=cast(int, body.get("generation")),
                prompt_generation=cast(int, body.get("prompt_generation")),
                group_id_sha256=cast(str, body.get("group_id_sha256")),
                sequence_index=cast(int, body.get("sequence_index")),
                vectors=vectors,
            )
        except (TypeError, ValueError) as exc:
            raise SeedProjectionIntegrityError(
                "Seed training projection failed reconstruction"
            ) from exc
        if (
            body.get("vector_contract_sha256") != result.vector_contract_sha256
            or result.to_bytes() != data
        ):
            raise SeedProjectionIntegrityError(
                "Seed training projection changed during replay"
            )
        return result

    @classmethod
    def from_measurement(
        cls,
        receipt: MeasurementReceipt,
        *,
        split: SeedSplit,
        generation: int,
        prompt_generation: int,
        group_id_sha256: str,
        sequence_index: int,
        vector_projector: SeedVectorProjection | VectorProjector[MeasurementReceipt],
        verifier_sha256s: Sequence[str],
        authority_sha256s: Sequence[str] = (),
    ) -> "SeedTrainingProjection":
        """Project a measurement while retaining all identities it owns."""

        if not isinstance(receipt, MeasurementReceipt):
            raise TypeError("receipt must be a MeasurementReceipt")
        return cls(
            source_kind="measurement",
            source_receipt_sha256s=(receipt.sha256,),
            model_pin_sha256=receipt.model_pin.sha256,
            code_identity_sha256=receipt.runtime.sha256,
            site_identity_sha256s=(receipt.coordinate.sha256,),
            graph_revision_sha256s=(
                receipt.weight_rail_revision.sha256,
                receipt.atlas_head_revision.sha256,
            ),
            authority_sha256s=(
                receipt.probe.sha256,
                receipt.intervention.sha256,
                *authority_sha256s,
            ),
            verifier_sha256s=tuple(verifier_sha256s),
            split=split,
            generation=generation,
            prompt_generation=prompt_generation,
            group_id_sha256=group_id_sha256,
            sequence_index=sequence_index,
            vectors=_project_vectors(receipt, vector_projector),
        )

    @classmethod
    def from_execution_learning(
        cls,
        receipt: ExecutionLearningReceipt,
        *,
        split: SeedSplit,
        generation: int,
        prompt_generation: int,
        group_id_sha256: str,
        sequence_index: int,
        vector_projector: SeedVectorProjection
        | VectorProjector[ExecutionLearningReceipt],
        authority_sha256s: Sequence[str] = (),
        verifier_sha256s: Sequence[str] = (),
    ) -> "SeedTrainingProjection":
        """Project a complete verified execution-learning transaction."""

        if not isinstance(receipt, ExecutionLearningReceipt):
            raise TypeError("receipt must be an ExecutionLearningReceipt")
        feature = receipt.feature
        measurement = receipt.measurement
        authority = receipt.action_authority
        return cls(
            source_kind="execution-learning",
            source_receipt_sha256s=(
                receipt.sha256,
                measurement.sha256,
                authority.sha256,
                feature.sha256,
                receipt.execution.sha256,
                receipt.quality.sha256,
                receipt.transition.sha256,
            ),
            model_pin_sha256=feature.model_pin_sha256,
            code_identity_sha256=measurement.runtime.sha256,
            site_identity_sha256s=(
                feature.site_identity.sha256,
                feature.weight_coordinate_sha256,
            ),
            graph_revision_sha256s=(
                feature.weight_graph_revision_sha256,
                feature.atlas_graph_revision_sha256,
            ),
            authority_sha256s=(
                authority.sha256,
                receipt.atlas_authentication_sha256,
                measurement.probe.sha256,
                measurement.intervention.sha256,
                *authority_sha256s,
            ),
            verifier_sha256s=(
                *feature.verifier_sha256s,
                authority.action_verifier_sha256,
                authority.quality_verifier_sha256,
                receipt.execution.verifier_sha256,
                receipt.quality.verifier_sha256,
                receipt.transition.verifier_sha256,
                *verifier_sha256s,
            ),
            split=split,
            generation=generation,
            prompt_generation=prompt_generation,
            group_id_sha256=group_id_sha256,
            sequence_index=sequence_index,
            vectors=_project_vectors(receipt, vector_projector),
        )

    @classmethod
    def from_demand_execution(
        cls,
        receipt: DemandRoutedExecutionReceipt,
        *,
        model_pin_sha256: str,
        code_identity_sha256: str,
        site_identity_sha256s: Sequence[str],
        split: SeedSplit,
        generation: int,
        prompt_generation: int,
        group_id_sha256: str,
        sequence_index: int,
        vector_projector: SeedVectorProjection
        | VectorProjector[DemandRoutedExecutionReceipt],
        graph_revision_sha256s: Sequence[str] = (),
        authority_sha256s: Sequence[str] = (),
        verifier_sha256s: Sequence[str] = (),
    ) -> "SeedTrainingProjection":
        """Project a demand route; caller supplies identities absent from it."""

        if not isinstance(receipt, DemandRoutedExecutionReceipt):
            raise TypeError("receipt must be a DemandRoutedExecutionReceipt")
        source_sha256s = [
            receipt.sha256,
            receipt.selection.sha256,
            receipt.residual.sha256,
            receipt.verification.sha256,
            receipt.outcome.sha256,
            receipt.outcome_transition.sha256,
        ]
        if receipt.ppm_prediction is not None:
            source_sha256s.append(receipt.ppm_prediction.sha256)
        if receipt.blanket_prediction is not None:
            source_sha256s.append(receipt.blanket_prediction.sha256)
        episode_authority = (
            () if receipt.episode_sha256 is None else (receipt.episode_sha256,)
        )
        return cls(
            source_kind="demand-routed-execution",
            source_receipt_sha256s=tuple(source_sha256s),
            model_pin_sha256=model_pin_sha256,
            code_identity_sha256=code_identity_sha256,
            site_identity_sha256s=tuple(site_identity_sha256s),
            graph_revision_sha256s=(
                receipt.graph_state_sha256,
                *graph_revision_sha256s,
            ),
            authority_sha256s=(
                receipt.selection.sha256,
                receipt.selected_prefix_route_sha256,
                *episode_authority,
                *authority_sha256s,
            ),
            verifier_sha256s=(
                *receipt.residual.verifier_sha256s,
                *receipt.outcome.route_verifier_sha256s,
                receipt.verification.verifier_sha256,
                receipt.outcome.outcome_verifier_sha256,
                *verifier_sha256s,
            ),
            split=split,
            generation=generation,
            prompt_generation=prompt_generation,
            group_id_sha256=group_id_sha256,
            sequence_index=sequence_index,
            vectors=_project_vectors(receipt, vector_projector),
        )


def _projection_order(row: SeedTrainingProjection) -> tuple[object, ...]:
    return (
        _SPLIT_RANK[cast(SeedSplit, row.split)],
        row.prompt_generation,
        row.group_id_sha256,
        row.sequence_index,
        row.generation,
        row.source_kind,
        row.source_receipt_sha256s,
        row.sha256,
    )


@dataclass(frozen=True, slots=True)
class SeedTrainingBatch:
    """Stable, leakage-checked collection addressable by Seed v0.3."""

    projections: tuple[SeedTrainingProjection, ...]

    def __post_init__(self) -> None:
        if isinstance(self.projections, (str, bytes)):
            raise TypeError("projections must be a receipt sequence")
        try:
            rows = tuple(self.projections)
        except TypeError as exc:
            raise TypeError("projections must be a receipt sequence") from exc
        if (
            not rows
            or len(rows) > MAX_BATCH_PROJECTIONS
            or any(not isinstance(row, SeedTrainingProjection) for row in rows)
        ):
            raise SeedProjectionError(
                f"batch requires 1..{MAX_BATCH_PROJECTIONS} projections"
            )
        rows = tuple(sorted(rows, key=_projection_order))
        addresses = tuple(row.sha256 for row in rows)
        if len(set(addresses)) != len(addresses):
            raise SeedProjectionError("batch contains duplicate projections")
        contracts = {row.vector_contract_sha256 for row in rows}
        if len(contracts) != 1:
            raise SeedProjectionError(
                "all batch rows must use one fixed vector schema and shape"
            )

        group_splits: dict[str, set[str]] = {}
        prompt_splits: dict[int, set[str]] = {}
        group_prompt_generations: dict[str, set[int]] = {}
        group_sequences: dict[str, list[int]] = {}
        for row in rows:
            group_splits.setdefault(row.group_id_sha256, set()).add(str(row.split))
            prompt_splits.setdefault(row.prompt_generation, set()).add(str(row.split))
            group_prompt_generations.setdefault(row.group_id_sha256, set()).add(
                row.prompt_generation
            )
            group_sequences.setdefault(row.group_id_sha256, []).append(
                row.sequence_index
            )
        if any(len(splits) != 1 for splits in group_splits.values()):
            raise SeedProjectionSplitError("a group crosses frozen data splits")
        if any(len(splits) != 1 for splits in prompt_splits.values()):
            raise SeedProjectionSplitError(
                "a prompt generation crosses frozen data splits"
            )
        if any(len(values) != 1 for values in group_prompt_generations.values()):
            raise SeedProjectionSplitError(
                "a sequence group contains multiple prompt generations"
            )
        for group_id, indices in group_sequences.items():
            expected = list(range(len(indices)))
            if sorted(indices) != expected:
                raise SeedProjectionError(
                    f"group {group_id} sequence indices must be contiguous from zero"
                )
        object.__setattr__(self, "projections", rows)

    @property
    def vector_contract(self) -> dict[str, object]:
        return self.projections[0].vectors.contract

    @property
    def vector_contract_sha256(self) -> str:
        return self.projections[0].vector_contract_sha256

    @property
    def feature_schema_sha256(self) -> str:
        return self.projections[0].feature_schema_sha256

    @property
    def feature_dimensions(self) -> int:
        return self.projections[0].feature_dimensions

    @property
    def projection_sha256s(self) -> tuple[str, ...]:
        return tuple(row.sha256 for row in self.projections)

    def body(self) -> dict[str, object]:
        return {
            "ordering": "split,prompt_generation,group,sequence,generation,source,sha256",
            "projection_sha256s": list(self.projection_sha256s),
            "projections": [row.to_dict() for row in self.projections],
            "vector_contract": self.vector_contract,
            "vector_contract_sha256": self.vector_contract_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return _seal(SEED_TRAINING_BATCH_SCHEMA, self.body())

    def to_bytes(self) -> bytes:
        result = canonical_json_bytes(self.to_dict())
        if len(result) > MAX_BATCH_BYTES:
            raise SeedProjectionError("Seed training batch exceeds its byte bound")
        return result

    @property
    def sha256(self) -> str:
        return _bytes_digest(self.to_bytes())

    @classmethod
    def from_bytes(cls, data: bytes) -> "SeedTrainingBatch":
        value = _strict_json(data, label="Seed training batch", maximum=MAX_BATCH_BYTES)
        body = _sealed_body(
            value,
            schema=SEED_TRAINING_BATCH_SCHEMA,
            label="Seed training batch",
        )
        expected = {
            "ordering",
            "projection_sha256s",
            "projections",
            "vector_contract",
            "vector_contract_sha256",
        }
        raw_rows = body.get("projections")
        if set(body) != expected or not isinstance(raw_rows, list):
            raise SeedProjectionIntegrityError("Seed training batch body is invalid")
        try:
            rows = tuple(
                SeedTrainingProjection.from_bytes(canonical_json_bytes(row))
                for row in raw_rows
            )
            result = cls(rows)
        except (TypeError, ValueError) as exc:
            raise SeedProjectionIntegrityError(
                "Seed training batch failed reconstruction"
            ) from exc
        if (
            body.get("ordering")
            != "split,prompt_generation,group,sequence,generation,source,sha256"
            or body.get("projection_sha256s") != list(result.projection_sha256s)
            or body.get("vector_contract") != result.vector_contract
            or body.get("vector_contract_sha256") != result.vector_contract_sha256
            or result.to_bytes() != data
        ):
            raise SeedProjectionIntegrityError(
                "Seed training batch changed during replay"
            )
        return result

    def _groups(self, split: SeedSplit) -> tuple[tuple[str, tuple[SeedTrainingProjection, ...]], ...]:
        if split not in SEED_SPLITS:
            raise SeedProjectionError("unknown Seed split")
        grouped: dict[str, list[SeedTrainingProjection]] = {}
        for row in self.projections:
            if row.split == split:
                grouped.setdefault(row.group_id_sha256, []).append(row)
        return tuple(
            (group_id, tuple(sorted(rows, key=lambda row: row.sequence_index)))
            for group_id, rows in sorted(grouped.items())
        )

    def vector_tensor(
        self,
        split: SeedSplit,
        vector_name: str,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_], tuple[str, ...]]:
        """Return one read-only padded ``[B,T,D]`` vector view and mask."""

        if vector_name not in _VECTOR_NAMES:
            raise SeedProjectionError("unknown Seed vector name")

        groups = self._groups(split)
        maximum_length = max((len(rows) for _group, rows in groups), default=0)
        dimensions = len(getattr(self.projections[0].vectors, vector_name))
        tensor = np.zeros(
            (len(groups), maximum_length, dimensions),
            dtype=np.float64,
        )
        mask = np.zeros((len(groups), maximum_length), dtype=np.bool_)
        for batch_index, (_group_id, rows) in enumerate(groups):
            for sequence_index, row in enumerate(rows):
                tensor[batch_index, sequence_index] = getattr(
                    row.vectors, vector_name
                )
                mask[batch_index, sequence_index] = True
        tensor.setflags(write=False)
        mask.setflags(write=False)
        return tensor, mask, tuple(group_id for group_id, _rows in groups)

    def feature_tensor(
        self,
        split: SeedSplit,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_], tuple[str, ...]]:
        """Return the Seed-v3 input view as ``[B,T,F]`` plus validity mask."""

        return self.vector_tensor(split, "feature")

    def manifest_body(self) -> dict[str, object]:
        split_counts = {
            split: sum(row.split == split for row in self.projections)
            for split in SEED_SPLITS
        }
        split_groups: dict[str, list[dict[str, object]]] = {}
        padded_shapes: dict[str, list[int]] = {}
        for split in SEED_SPLITS:
            groups = self._groups(split)
            split_groups[split] = [
                {
                    "group_id_sha256": group_id,
                    "prompt_generation": rows[0].prompt_generation,
                    "projection_sha256s": [row.sha256 for row in rows],
                    "sequence_length": len(rows),
                }
                for group_id, rows in groups
            ]
            padded_shapes[split] = [
                len(groups),
                max((len(rows) for _group, rows in groups), default=0),
                self.feature_dimensions,
            ]
        return {
            "axis_order": ["group", "sequence", "feature"],
            "batch_schema": SEED_TRAINING_BATCH_SCHEMA,
            "batch_sha256": self.sha256,
            "feature_dimensions": self.feature_dimensions,
            "feature_schema_sha256": self.feature_schema_sha256,
            "feature_tensor_dtype": "float64",
            "padding_mask": "true=observed;false=zero-padding",
            "padded_feature_shapes": padded_shapes,
            "projection_count": len(self.projections),
            "projection_schema": SEED_TRAINING_PROJECTION_SCHEMA,
            "projection_sha256s": list(self.projection_sha256s),
            "source_receipt_sha256s": list(
                sorted(
                    {
                        source
                        for row in self.projections
                        for source in row.source_receipt_sha256s
                    }
                )
            ),
            "split_counts": split_counts,
            "split_groups": split_groups,
            "vector_contract": self.vector_contract,
            "vector_contract_sha256": self.vector_contract_sha256,
        }

    def to_seed_v3_manifest(self) -> dict[str, object]:
        return _seal(SEED_TRAINING_MANIFEST_SCHEMA, self.manifest_body())

    def manifest_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_seed_v3_manifest())

    @property
    def manifest_sha256(self) -> str:
        return _bytes_digest(self.manifest_bytes())


def _zero_failure_upper_bound(*, groups: int, confidence: float) -> float:
    """Exact one-sided binomial bound for zero failures in independent groups."""

    if groups <= 0:
        raise SeedProjectionError("at least one admitted group is required")
    alpha = 1.0 - confidence
    # Stable form of 1 - alpha**(1/groups), not the rule-of-three approximation.
    return -math.expm1(math.log(alpha) / groups)


@dataclass(frozen=True, slots=True)
class GroupAwareZeroFailureRiskReceipt:
    """Promotion certificate counting independent groups, never correlated rows."""

    batch_sha256: str
    batch_manifest_sha256: str
    failure_policy_sha256: str
    admitted_group_sha256s: tuple[str, ...]
    admitted_projection_sha256s: tuple[str, ...]
    admitted_group_count: int
    admitted_row_count: int
    worst_group_failures: int
    confidence: float
    exact_upper_bound: float
    maximum_risk: float
    promotion_admissible: bool

    def __post_init__(self) -> None:
        for field in (
            "batch_sha256",
            "batch_manifest_sha256",
            "failure_policy_sha256",
        ):
            object.__setattr__(
                self,
                field,
                require_sha256(getattr(self, field), field=field),
            )
        groups = _hashes(self.admitted_group_sha256s, field="admitted_group_sha256s")
        projections = _hashes(
            self.admitted_projection_sha256s,
            field="admitted_projection_sha256s",
        )
        object.__setattr__(self, "admitted_group_sha256s", groups)
        object.__setattr__(self, "admitted_projection_sha256s", projections)
        group_count = _uint(self.admitted_group_count, field="admitted_group_count")
        row_count = _uint(self.admitted_row_count, field="admitted_row_count")
        failures = _uint(self.worst_group_failures, field="worst_group_failures")
        if group_count != len(groups) or row_count != len(projections):
            raise SeedProjectionError("risk counts disagree with admitted identities")
        if failures != 0:
            raise SeedProjectionError("zero-failure promotion rejects a failed group")
        confidence = _probability(self.confidence, field="confidence", strict=True)
        maximum = _probability(self.maximum_risk, field="maximum_risk")
        bound = _probability(self.exact_upper_bound, field="exact_upper_bound")
        expected = _zero_failure_upper_bound(groups=group_count, confidence=confidence)
        if not math.isclose(bound, expected, rel_tol=1e-14, abs_tol=1e-15):
            raise SeedProjectionIntegrityError("risk bound is not the exact group bound")
        admissible = bound <= maximum
        if self.promotion_admissible is not admissible:
            raise SeedProjectionIntegrityError("promotion decision differs from risk bound")
        object.__setattr__(self, "admitted_group_count", group_count)
        object.__setattr__(self, "admitted_row_count", row_count)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "exact_upper_bound", bound)
        object.__setattr__(self, "maximum_risk", maximum)

    @classmethod
    def evaluate(
        cls,
        batch: SeedTrainingBatch,
        *,
        admitted_group_sha256s: Sequence[str],
        failure_predicate: Callable[[SeedTrainingProjection], bool],
        failure_policy_sha256: str,
        confidence: float = 0.95,
        maximum_risk: float = 0.05,
    ) -> "GroupAwareZeroFailureRiskReceipt":
        if not isinstance(batch, SeedTrainingBatch):
            raise TypeError("batch must be a SeedTrainingBatch")
        if not callable(failure_predicate):
            raise TypeError("failure_predicate must be callable")
        admitted = _hashes(
            admitted_group_sha256s,
            field="admitted_group_sha256s",
        )
        by_group: dict[str, list[SeedTrainingProjection]] = {}
        for row in batch.projections:
            by_group.setdefault(row.group_id_sha256, []).append(row)
        missing = sorted(set(admitted) - set(by_group))
        if missing:
            raise SeedProjectionError("an admitted group is absent from the batch")
        admitted_rows = tuple(
            row
            for group_id in admitted
            for row in by_group[group_id]
        )
        group_failures = {
            group_id: sum(bool(failure_predicate(row)) for row in by_group[group_id])
            for group_id in admitted
        }
        worst = max(group_failures.values(), default=0)
        if worst:
            raise SeedProjectionError(
                "zero-failure promotion rejects the admitted group evidence"
            )
        normalized_confidence = _probability(
            confidence,
            field="confidence",
            strict=True,
        )
        normalized_maximum = _probability(maximum_risk, field="maximum_risk")
        bound = _zero_failure_upper_bound(
            groups=len(admitted),
            confidence=normalized_confidence,
        )
        return cls(
            batch_sha256=batch.sha256,
            batch_manifest_sha256=batch.manifest_sha256,
            failure_policy_sha256=failure_policy_sha256,
            admitted_group_sha256s=admitted,
            admitted_projection_sha256s=tuple(row.sha256 for row in admitted_rows),
            admitted_group_count=len(admitted),
            admitted_row_count=len(admitted_rows),
            worst_group_failures=worst,
            confidence=normalized_confidence,
            exact_upper_bound=bound,
            maximum_risk=normalized_maximum,
            promotion_admissible=bound <= normalized_maximum,
        )

    def body(self) -> dict[str, object]:
        return {
            "admitted_group_count": self.admitted_group_count,
            "admitted_group_sha256s": list(self.admitted_group_sha256s),
            "admitted_projection_sha256s": list(self.admitted_projection_sha256s),
            "admitted_row_count": self.admitted_row_count,
            "batch_manifest_sha256": self.batch_manifest_sha256,
            "batch_sha256": self.batch_sha256,
            "confidence": self.confidence,
            "exact_upper_bound": self.exact_upper_bound,
            "failure_policy_sha256": self.failure_policy_sha256,
            "maximum_risk": self.maximum_risk,
            "promotion_admissible": self.promotion_admissible,
            "risk_unit": "worst-row-outcome-per-admitted-group",
            "worst_group_failures": self.worst_group_failures,
        }

    def to_dict(self) -> dict[str, object]:
        return _seal(SEED_PROMOTION_RISK_SCHEMA, self.body())

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return _bytes_digest(self.to_bytes())

    @classmethod
    def from_bytes(cls, data: bytes) -> "GroupAwareZeroFailureRiskReceipt":
        value = _strict_json(data, label="Seed promotion risk", maximum=4 * 1024 * 1024)
        body = _sealed_body(
            value,
            schema=SEED_PROMOTION_RISK_SCHEMA,
            label="Seed promotion risk",
        )
        expected = {
            "admitted_group_count",
            "admitted_group_sha256s",
            "admitted_projection_sha256s",
            "admitted_row_count",
            "batch_manifest_sha256",
            "batch_sha256",
            "confidence",
            "exact_upper_bound",
            "failure_policy_sha256",
            "maximum_risk",
            "promotion_admissible",
            "risk_unit",
            "worst_group_failures",
        }
        if set(body) != expected or body.get("risk_unit") != (
            "worst-row-outcome-per-admitted-group"
        ):
            raise SeedProjectionIntegrityError("Seed promotion risk body is invalid")
        arguments = dict(body)
        arguments.pop("risk_unit")
        try:
            result = cls(
                **arguments  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            raise SeedProjectionIntegrityError(
                "Seed promotion risk failed reconstruction"
            ) from exc
        if result.to_bytes() != data:
            raise SeedProjectionIntegrityError(
                "Seed promotion risk changed during replay"
            )
        return result


# Compact name for callers that do not need the policy's full spelling.
SeedPromotionRiskReceipt = GroupAwareZeroFailureRiskReceipt


__all__ = [
    "GroupAwareZeroFailureRiskReceipt",
    "SEED_PROMOTION_RISK_SCHEMA",
    "SEED_SPLITS",
    "SEED_TRAINING_BATCH_SCHEMA",
    "SEED_TRAINING_MANIFEST_SCHEMA",
    "SEED_TRAINING_PROJECTION_SCHEMA",
    "SEED_VECTOR_CONTRACT_SCHEMA",
    "SeedProjectionError",
    "SeedProjectionIntegrityError",
    "SeedProjectionSplitError",
    "SeedSplit",
    "SeedPromotionRiskReceipt",
    "SeedTrainingBatch",
    "SeedTrainingProjection",
    "SeedVectorProjection",
    "VectorProjector",
    "VECTOR_ABS_BOUND",
]
