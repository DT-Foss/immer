"""Content-addressed zero-forward Qwen result cells.

A result cell stores the exact output of one completed cold Qwen/FERTIG path
under the hashes of every input that can change that output.  The warm path
mounts the stored Qwen result as an OoE ``mount_organ`` action and spends zero
new Qwen forwards.  Savings remain pending until the final parity verifier has
checked the warm result, the final FERTIG judgment, and the evaluator contract.

The module deliberately keeps runtime parity separate from semantic
certification.  A FERTIG-abstained cold path can be reproduced exactly without
being mislabeled as an exact FERTIG proof.  A frozen external evaluator is only
opened by :class:`FinalBenchmarkLayer`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Literal

from immer.contracts import ExecutionStatus, Result
from immer.runtimes.qwen3_8.semantic_atlas import ModelPin

from .chat import result_from_document, result_to_document
from .controller import ActionExecution
from .crystal import (
    CrystalStore,
    CrystalStoreError,
    ManifestConflictError,
)
from .identity import canonical_json_bytes, require_sha256
from .qwen_bridge import QwenOoeFeatureReceipt


RESULT_CELL_SCHEMA = "immer-ooe-result-cell/v1"
RESULT_CELL_BINDING_SCHEMA = "immer-ooe-result-cell-binding/v1"
RESULT_CELL_POINTER_SCHEMA = "immer-ooe-result-cell-pointer/v1"
RESULT_CELL_PARITY_SCHEMA = "immer-ooe-result-cell-parity/v1"
RESULT_CELL_BENCHMARK_SCHEMA = "immer-ooe-result-cell-benchmark/v1"
RESULT_CELL_EXECUTOR_SCHEMA = "immer-ooe-result-cell-executor/v1"
COLD_QWEN_GENERATION_SCHEMA = "immer-ooe-cold-qwen-generation/v1"
COLD_QWEN_BINDING_EVIDENCE_SCHEMA = "immer-ooe-qwen-binding-evidence/v1"
PROVENANCE_UNIT_SCHEMA = "immer-ooe-result-cell-provenance-unit/v1"
COLD_QWEN_GENERATION_EVIDENCE_KEY = "cold_qwen_generation_receipt"
COLD_QWEN_BINDING_EVIDENCE_KEY = "result_cell_binding_receipt"
COLD_QWEN_GENERATION_PROVENANCE_NAME = "cold-qwen-generation"
COLD_QWEN_GENERATION_VERIFIER_SHA256 = hashlib.sha256(
    canonical_json_bytes(
        {
            "forward_count": "positive authenticated forward_passes",
            "model_binding": "exact ModelPin.sha256",
            "result_binding": "cold Result without embedded receipt",
            "schema": COLD_QWEN_GENERATION_SCHEMA,
        }
    )
).hexdigest()
RESULT_CELL_EXECUTOR_SHA256 = hashlib.sha256(
    canonical_json_bytes(
        {
            "action": "mount_organ",
            "input": RESULT_CELL_SCHEMA,
            "output": "immer-ooe-action-execution/v1",
            "schema": RESULT_CELL_EXECUTOR_SCHEMA,
        }
    )
).hexdigest()

MAX_RESULT_DOCUMENT_BYTES = 1024 * 1024
MAX_RESULT_CELL_BYTES = 4 * 1024 * 1024
MAX_TEACHER_FORWARDS = 1_000_000
_OBJECT_PREFIX = "ooe-result-cell-object/v1:"
_POINTER_PREFIX = "ooe-result-cell-binding/v1:"
_LOCK_NAME = ".result-cells.lock"
_FERTIG_STATUSES = frozenset(("verified", "mismatch", "abstained"))
_RAW_METADATA_KEYS = frozenset(
    (
        "content",
        "input",
        "input_ids",
        "input_text",
        "messages",
        "prompt",
        "question",
        "raw",
        "rendered",
        "rendered_prompt",
        "request",
        "request_payload",
        "system_prompt",
        "text",
        "token_ids",
        "token_pieces",
        "tokens",
    )
)
_HASH_OR_SIZE_SUFFIXES = (
    "_bytes",
    "_count",
    "_digest",
    "_fingerprint",
    "_hash",
    "_length",
    "_sha256",
    "_shape",
    "_size",
)
_QWEN_GENERATION_FIELDS = frozenset(
    (
        "context_mode",
        "forward_passes",
        "general_generation",
        "generated_tokens",
        "linear_calls",
        "prefill_mode",
        "prompt_tokens",
        "seconds",
        "source_body_bytes",
        "state_bytes",
        "stateful_cache",
        "stopped_on_eos",
        "token_trace_sha256",
    )
)


class ResultCellError(RuntimeError):
    """A zero-forward result cell contract could not be satisfied."""


class ResultCellIntegrityError(ResultCellError):
    """A result cell, pointer, or execution receipt failed authentication."""


class ResultCellStaleError(ResultCellIntegrityError):
    """A result cell belongs to another exact runtime or prompt identity."""


class ResultCellConflictError(ResultCellIntegrityError):
    """An immutable binding was concurrently or inconsistently rebound."""


class ResultCellMissError(ResultCellError):
    """No charged result cell exists for the exact runtime binding."""


class ResultCellParityError(ResultCellIntegrityError):
    """A warm execution differs from its sealed cold reference."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def artifact_sha256(value: object) -> str:
    """Hash one canonical JSON artifact, such as a FERTIG judgment."""

    return _digest(value)


def _strict_json(data: bytes, *, label: str, maximum: int) -> object:
    if not isinstance(data, bytes):
        raise TypeError(f"{label} must be immutable bytes")
    if len(data) > maximum:
        raise ResultCellIntegrityError(f"{label} exceeds its hard byte limit")

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
        raise ResultCellIntegrityError(f"{label} is invalid canonical JSON") from exc
    if canonical_json_bytes(value) != data:
        raise ResultCellIntegrityError(f"{label} is not canonical JSON")
    return value


def _uint(value: object, *, field: str, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > MAX_TEACHER_FORWARDS
    ):
        qualifier = "positive " if positive else "non-negative "
        raise ValueError(f"{field} must be a bounded {qualifier}integer")
    return value


@dataclass(frozen=True, slots=True)
class ProvenanceUnit:
    """One indivisible verifier/evidence/receipt provenance tuple."""

    name: str
    verifier_sha256: str
    evidence_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or self.name != self.name.strip()
            or "\x00" in self.name
            or len(self.name.encode("utf-8")) > 256
        ):
            raise ValueError("provenance-unit name must be canonical non-empty text")
        for field_name in (
            "verifier_sha256",
            "evidence_sha256",
            "receipt_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )

    def as_record(self) -> dict[str, str]:
        return {
            "evidence_sha256": self.evidence_sha256,
            "name": self.name,
            "receipt_sha256": self.receipt_sha256,
            "verifier_sha256": self.verifier_sha256,
        }

    @property
    def pair_binding_sha256(self) -> str:
        return _digest(self.as_record())

    @property
    def sha256(self) -> str:
        return self.pair_binding_sha256

    def to_document(self) -> dict[str, Any]:
        body = self.as_record()
        return {
            "body": body,
            "schema": PROVENANCE_UNIT_SCHEMA,
            "sha256": _digest(body),
        }

    @classmethod
    def from_document(cls, value: object) -> "ProvenanceUnit":
        if (
            not isinstance(value, Mapping)
            or set(value) != {"body", "schema", "sha256"}
            or value.get("schema") != PROVENANCE_UNIT_SCHEMA
        ):
            raise ResultCellIntegrityError("provenance-unit envelope is invalid")
        body = value.get("body")
        if not isinstance(body, Mapping) or set(body) != {
            "evidence_sha256",
            "name",
            "receipt_sha256",
            "verifier_sha256",
        }:
            raise ResultCellIntegrityError("provenance-unit body is invalid")
        claimed = require_sha256(value.get("sha256"), field="provenance sha256")
        if claimed != _digest(body):
            raise ResultCellIntegrityError("provenance-unit SHA-256 mismatch")
        try:
            unit = cls(**dict(body))
        except (TypeError, ValueError) as exc:
            raise ResultCellIntegrityError("provenance-unit validation failed") from exc
        if unit.to_document() != dict(value):
            raise ResultCellIntegrityError(
                "provenance unit failed canonical reconstruction"
            )
        return unit


def _provenance_units(
    value: tuple[ProvenanceUnit, ...],
) -> tuple[ProvenanceUnit, ...]:
    if not isinstance(value, tuple) or not 1 <= len(value) <= 256:
        raise ValueError("provenance_units must contain 1..256 immutable units")
    if any(not isinstance(unit, ProvenanceUnit) for unit in value):
        raise TypeError("provenance_units must contain ProvenanceUnit values")
    normalized = tuple(sorted(value, key=lambda unit: (unit.name, unit.sha256)))
    if len({unit.name for unit in normalized}) != len(normalized):
        raise ValueError("provenance-unit names must be unique")
    if len({unit.sha256 for unit in normalized}) != len(normalized):
        raise ValueError("provenance units must be unique")
    return normalized


@dataclass(frozen=True, slots=True)
class ColdQwenGenerationReceipt:
    """Authenticated cold generation count bound to one ModelPin and Result."""

    model_pin_sha256: str
    result_without_receipt_sha256: str
    generation_evidence_sha256: str
    forward_passes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_pin_sha256",
            require_sha256(self.model_pin_sha256, field="model_pin_sha256"),
        )
        object.__setattr__(
            self,
            "result_without_receipt_sha256",
            require_sha256(
                self.result_without_receipt_sha256,
                field="result_without_receipt_sha256",
            ),
        )
        object.__setattr__(
            self,
            "generation_evidence_sha256",
            require_sha256(
                self.generation_evidence_sha256,
                field="generation_evidence_sha256",
            ),
        )
        object.__setattr__(
            self,
            "forward_passes",
            _uint(self.forward_passes, field="forward_passes", positive=True),
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "forward_passes": self.forward_passes,
            "generation_evidence_sha256": self.generation_evidence_sha256,
            "model_pin_sha256": self.model_pin_sha256,
            "result_without_receipt_sha256": (
                self.result_without_receipt_sha256
            ),
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    def to_document(self) -> dict[str, Any]:
        body = self.as_record()
        return {
            "body": body,
            "schema": COLD_QWEN_GENERATION_SCHEMA,
            "sha256": _digest(body),
        }

    @classmethod
    def from_document(cls, value: object) -> "ColdQwenGenerationReceipt":
        if (
            not isinstance(value, Mapping)
            or set(value) != {"body", "schema", "sha256"}
            or value.get("schema") != COLD_QWEN_GENERATION_SCHEMA
        ):
            raise ResultCellIntegrityError("cold generation receipt is invalid")
        body = value.get("body")
        if not isinstance(body, Mapping) or set(body) != {
            "forward_passes",
            "generation_evidence_sha256",
            "model_pin_sha256",
            "result_without_receipt_sha256",
        }:
            raise ResultCellIntegrityError("cold generation receipt body is invalid")
        claimed = require_sha256(value.get("sha256"), field="generation sha256")
        if claimed != _digest(body):
            raise ResultCellIntegrityError("cold generation receipt SHA-256 mismatch")
        try:
            receipt = cls(**dict(body))
        except (TypeError, ValueError) as exc:
            raise ResultCellIntegrityError(
                "cold generation receipt validation failed"
            ) from exc
        if receipt.to_document() != dict(value):
            raise ResultCellIntegrityError(
                "cold generation receipt failed canonical reconstruction"
            )
        return receipt

    @property
    def provenance_unit(self) -> ProvenanceUnit:
        return ProvenanceUnit(
            name=COLD_QWEN_GENERATION_PROVENANCE_NAME,
            verifier_sha256=COLD_QWEN_GENERATION_VERIFIER_SHA256,
            evidence_sha256=_digest(
                {
                    "generation_evidence_sha256": (
                        self.generation_evidence_sha256
                    ),
                    "result_without_receipt_sha256": (
                        self.result_without_receipt_sha256
                    ),
                }
            ),
            receipt_sha256=self.sha256,
        )


def _result_json(result: Result) -> bytes:
    _validate_no_raw_prompt_metadata(result)
    document = result_to_document(result)
    encoded = canonical_json_bytes(document)
    if len(encoded) > MAX_RESULT_DOCUMENT_BYTES:
        raise ValueError("result document exceeds one MiB")
    return encoded


def _decode_result(data: bytes, *, field: str) -> Result:
    value = _strict_json(data, label=field, maximum=MAX_RESULT_DOCUMENT_BYTES)
    try:
        result = result_from_document(value)
    except (TypeError, ValueError) as exc:
        raise ResultCellIntegrityError(f"{field} failed result authentication") from exc
    _validate_no_raw_prompt_metadata(result)
    return result


def _validate_no_raw_prompt_metadata(result: Result) -> None:
    """Reject prompt-bearing runtime evidence while retaining its hashes.

    Result output is intentionally untouched: it is the organ payload.  The
    evidence tree, however, is metadata and may contain only hashes, sizes, or
    non-prompt diagnostics for prompt-related fields.
    """

    if not isinstance(result, Result):
        raise TypeError("result must be a Result")

    def visit(value: object, *, path: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                if not isinstance(raw_key, str):
                    raise ValueError("result evidence keys must be text")
                key = raw_key.strip().lower().replace("-", "_")
                protected = key.endswith(_HASH_OR_SIZE_SUFFIXES)
                if key in _RAW_METADATA_KEYS and not protected and child is not None:
                    if isinstance(child, (str, list, tuple, Mapping)):
                        raise ValueError(
                            f"result evidence contains raw prompt metadata at {path}.{raw_key}"
                        )
                visit(child, path=f"{path}.{raw_key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, path=f"{path}[{index}]")

    visit(result.evidence, path="evidence")


def _result_core(result: Result) -> dict[str, Any]:
    """Return exact user-visible result fields, excluding route evidence."""

    if not isinstance(result, Result):
        raise TypeError("result must be a Result")
    return {
        "component": result.component,
        "output": result.output,
        "reason": result.reason,
        "status": result.status.value,
    }


def _authenticated_qwen_generation_evidence(
    result: Result,
    *,
    binding: "ResultCellBinding",
) -> tuple[int, str]:
    """Validate the raw Qwen runtime receipt and derive its forward count."""

    if not isinstance(result, Result):
        raise TypeError("result must be a Result")
    if not isinstance(binding, ResultCellBinding):
        raise TypeError("binding must be an exact ResultCellBinding")
    model_pin = binding.model_pin
    evidence = result.evidence
    if not isinstance(evidence, Mapping):
        raise ResultCellIntegrityError("cold Qwen evidence must be a mapping")
    if evidence.get("model") != model_pin.repo_id:
        raise ResultCellStaleError("cold Qwen evidence model mismatch")
    if evidence.get("revision") != model_pin.revision:
        raise ResultCellStaleError("cold Qwen evidence revision mismatch")
    if evidence.get("tokenizer_sha256") != binding.tokenizer_sha256:
        raise ResultCellStaleError("cold Qwen evidence tokenizer mismatch")
    _assert_qwen_binding_evidence(evidence, binding=binding)
    bundle = evidence.get("bundle")
    if not isinstance(bundle, Mapping):
        raise ResultCellIntegrityError("cold Qwen evidence lacks bundle receipt")
    if bundle.get("layout_fingerprint") != model_pin.bundle_fingerprint:
        raise ResultCellStaleError("cold Qwen bundle fingerprint mismatch")
    if bundle.get("manifest_sha256") != model_pin.bundle_manifest_sha256:
        raise ResultCellStaleError("cold Qwen bundle manifest mismatch")
    generation = evidence.get("generation")
    if not isinstance(generation, Mapping) or set(generation) != _QWEN_GENERATION_FIELDS:
        raise ResultCellIntegrityError("cold Qwen generation schema is invalid")
    for field_name in (
        "forward_passes",
        "generated_tokens",
        "linear_calls",
        "prompt_tokens",
        "source_body_bytes",
        "state_bytes",
    ):
        value = generation[field_name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ResultCellIntegrityError(
                f"cold Qwen generation {field_name} is invalid"
            )
    forward_passes = _uint(
        generation["forward_passes"],
        field="generation.forward_passes",
        positive=True,
    )
    for field_name in ("stateful_cache", "general_generation", "stopped_on_eos"):
        if not isinstance(generation[field_name], bool):
            raise ResultCellIntegrityError(
                f"cold Qwen generation {field_name} is invalid"
            )
    if (
        generation["stateful_cache"] is not True
        or generation["general_generation"] is not True
        or generation["context_mode"] != "stateful_autoregressive"
        or generation["prefill_mode"] != "batched"
    ):
        raise ResultCellIntegrityError("cold Qwen used another generation path")
    seconds = generation["seconds"]
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(float(seconds))
        or float(seconds) < 0.0
    ):
        raise ResultCellIntegrityError("cold Qwen generation seconds is invalid")
    require_sha256(
        generation["token_trace_sha256"],
        field="generation.token_trace_sha256",
    )
    if (
        not isinstance(result.output, str)
        or evidence.get("output_sha256")
        != hashlib.sha256(result.output.encode("utf-8")).hexdigest()
    ):
        raise ResultCellIntegrityError("cold Qwen output evidence mismatch")
    return forward_passes, _digest(dict(generation))


def attach_cold_qwen_generation_receipt(
    result: Result,
    *,
    binding: "ResultCellBinding",
    expected_forward_passes: int | None = None,
) -> Result:
    """Seal the forward count already authenticated by the raw Qwen Result."""

    if not isinstance(result, Result):
        raise TypeError("result must be a Result")
    if not isinstance(binding, ResultCellBinding):
        raise TypeError("binding must be an exact ResultCellBinding")
    if COLD_QWEN_GENERATION_EVIDENCE_KEY in result.evidence:
        raise ValueError("cold Qwen Result already contains a generation receipt")
    evidence = dict(result.evidence)
    binding_document = qwen_result_binding_evidence(binding)
    existing_binding = evidence.get(COLD_QWEN_BINDING_EVIDENCE_KEY)
    if existing_binding is not None and existing_binding != binding_document:
        raise ResultCellStaleError(
            "cold Qwen Result already carries another execution binding"
        )
    evidence[COLD_QWEN_BINDING_EVIDENCE_KEY] = binding_document
    bound_result = Result(
        result.status,
        result.component,
        output=result.output,
        reason=result.reason,
        evidence=evidence,
    )
    base_sha256 = hashlib.sha256(_result_json(bound_result)).hexdigest()
    forward_passes, generation_evidence_sha256 = (
        _authenticated_qwen_generation_evidence(bound_result, binding=binding)
    )
    if expected_forward_passes is not None:
        expected = _uint(
            expected_forward_passes,
            field="expected_forward_passes",
            positive=True,
        )
        if expected != forward_passes:
            raise ResultCellIntegrityError(
                "expected forward count differs from raw Qwen generation evidence"
            )
    receipt = ColdQwenGenerationReceipt(
        model_pin_sha256=binding.model_pin.sha256,
        result_without_receipt_sha256=base_sha256,
        generation_evidence_sha256=generation_evidence_sha256,
        forward_passes=forward_passes,
    )
    evidence[COLD_QWEN_GENERATION_EVIDENCE_KEY] = receipt.to_document()
    attached = Result(
        result.status,
        result.component,
        output=result.output,
        reason=result.reason,
        evidence=evidence,
    )
    _result_json(attached)
    return attached


def extract_cold_qwen_generation_receipt(
    result: Result,
    *,
    binding: "ResultCellBinding",
) -> ColdQwenGenerationReceipt:
    """Authenticate and recompute the embedded cold-forward receipt."""

    if not isinstance(result, Result):
        raise TypeError("result must be a Result")
    if not isinstance(binding, ResultCellBinding):
        raise TypeError("binding must be an exact ResultCellBinding")
    expected_model = binding.model_pin.sha256
    evidence = dict(result.evidence)
    try:
        document = evidence.pop(COLD_QWEN_GENERATION_EVIDENCE_KEY)
    except KeyError as exc:
        raise ResultCellIntegrityError(
            "cold Qwen Result lacks its generation receipt"
        ) from exc
    receipt = ColdQwenGenerationReceipt.from_document(document)
    if receipt.model_pin_sha256 != expected_model:
        raise ResultCellStaleError("cold generation receipt ModelPin mismatch")
    base = Result(
        result.status,
        result.component,
        output=result.output,
        reason=result.reason,
        evidence=evidence,
    )
    actual_result_sha256 = hashlib.sha256(_result_json(base)).hexdigest()
    if receipt.result_without_receipt_sha256 != actual_result_sha256:
        raise ResultCellIntegrityError(
            "cold generation receipt Result hash mismatch"
        )
    actual_forward_passes, generation_evidence_sha256 = (
        _authenticated_qwen_generation_evidence(base, binding=binding)
    )
    if (
        receipt.forward_passes != actual_forward_passes
        or receipt.generation_evidence_sha256 != generation_evidence_sha256
    ):
        raise ResultCellIntegrityError(
            "cold generation receipt differs from raw Qwen generation evidence"
        )
    return receipt


@dataclass(frozen=True, slots=True)
class ResultCellBinding:
    """Every exact input capable of changing one cached Qwen result.

    Only hashes are accepted for question and prompt material.  Consequently
    neither bank state names nor cell metadata can contain the raw prompt.
    """

    model_pin: ModelPin
    tokenizer_sha256: str
    question_sha256: str
    rendered_prompt_sha256: str
    rendered_prompt_token_sha256: str
    system_prompt_sha256: str
    generation_policy_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.model_pin, ModelPin):
            raise TypeError("model_pin must be an exact ModelPin")
        for field_name in (
            "tokenizer_sha256",
            "question_sha256",
            "rendered_prompt_sha256",
            "rendered_prompt_token_sha256",
            "system_prompt_sha256",
            "generation_policy_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )

    @property
    def model_pin_sha256(self) -> str:
        return self.model_pin.sha256

    def as_record(self) -> dict[str, Any]:
        return {
            "generation_policy_sha256": self.generation_policy_sha256,
            "model_pin": self.model_pin.to_document(),
            "model_pin_sha256": self.model_pin.sha256,
            "question_sha256": self.question_sha256,
            "rendered_prompt_sha256": self.rendered_prompt_sha256,
            "rendered_prompt_token_sha256": self.rendered_prompt_token_sha256,
            "schema": RESULT_CELL_BINDING_SCHEMA,
            "system_prompt_sha256": self.system_prompt_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    @classmethod
    def from_record(cls, value: object) -> "ResultCellBinding":
        expected = {
            "generation_policy_sha256",
            "model_pin",
            "model_pin_sha256",
            "question_sha256",
            "rendered_prompt_sha256",
            "rendered_prompt_token_sha256",
            "schema",
            "system_prompt_sha256",
            "tokenizer_sha256",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or value.get("schema") != RESULT_CELL_BINDING_SCHEMA
        ):
            raise ResultCellIntegrityError("result-cell binding shape is invalid")
        try:
            model_pin = ModelPin.from_document(value["model_pin"])
            binding = cls(
                model_pin=model_pin,
                tokenizer_sha256=value["tokenizer_sha256"],
                question_sha256=value["question_sha256"],
                rendered_prompt_sha256=value["rendered_prompt_sha256"],
                rendered_prompt_token_sha256=value[
                    "rendered_prompt_token_sha256"
                ],
                system_prompt_sha256=value["system_prompt_sha256"],
                generation_policy_sha256=value["generation_policy_sha256"],
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ResultCellIntegrityError(
                "result-cell binding validation failed"
            ) from exc
        if value["model_pin_sha256"] != model_pin.sha256:
            raise ResultCellIntegrityError("result-cell ModelPin hash mismatch")
        if binding.as_record() != dict(value):
            raise ResultCellIntegrityError(
                "result-cell binding failed canonical reconstruction"
            )
        return binding

    def assert_current(
        self,
        *,
        model_pin: ModelPin,
        tokenizer_sha256: str,
        question_sha256: str,
        rendered_prompt_sha256: str,
        rendered_prompt_token_sha256: str,
        system_prompt_sha256: str,
        generation_policy_sha256: str,
    ) -> None:
        """Reject any model, prompt, tokenizer, or policy drift."""

        current = ResultCellBinding(
            model_pin=model_pin,
            tokenizer_sha256=tokenizer_sha256,
            question_sha256=question_sha256,
            rendered_prompt_sha256=rendered_prompt_sha256,
            rendered_prompt_token_sha256=rendered_prompt_token_sha256,
            system_prompt_sha256=system_prompt_sha256,
            generation_policy_sha256=generation_policy_sha256,
        )
        if current != self:
            raise ResultCellStaleError(
                "result cell is stale for the exact model, prompt, or policy"
            )

    def assert_feature_bound(self, receipt: QwenOoeFeatureReceipt) -> None:
        if not isinstance(receipt, QwenOoeFeatureReceipt):
            raise TypeError("receipt must be a QwenOoeFeatureReceipt")
        if (
            receipt.model_pin_sha256 != self.model_pin.sha256
            or receipt.probe.question_sha256 != self.question_sha256
            or receipt.probe.token_sha256 != self.rendered_prompt_token_sha256
        ):
            raise ResultCellStaleError(
                "feature receipt belongs to another result-cell prompt or model"
            )


def qwen_result_binding_evidence(
    binding: ResultCellBinding,
) -> dict[str, Any]:
    """Seal the hash-only execution binding recorded by the cold harness."""

    if not isinstance(binding, ResultCellBinding):
        raise TypeError("binding must be a ResultCellBinding")
    body = {
        "binding_sha256": binding.sha256,
        "code_revision": binding.model_pin.code_revision,
        "generation_policy_sha256": binding.generation_policy_sha256,
        "model_pin_sha256": binding.model_pin.sha256,
        "question_sha256": binding.question_sha256,
        "rendered_prompt_sha256": binding.rendered_prompt_sha256,
        "rendered_prompt_token_sha256": binding.rendered_prompt_token_sha256,
        "system_prompt_sha256": binding.system_prompt_sha256,
        "tokenizer_sha256": binding.tokenizer_sha256,
    }
    return {
        "body": body,
        "schema": COLD_QWEN_BINDING_EVIDENCE_SCHEMA,
        "sha256": _digest(body),
    }


def _assert_qwen_binding_evidence(
    evidence: Mapping[str, Any],
    *,
    binding: ResultCellBinding,
) -> None:
    actual = evidence.get(COLD_QWEN_BINDING_EVIDENCE_KEY)
    expected = qwen_result_binding_evidence(binding)
    if actual != expected:
        raise ResultCellStaleError(
            "cold Qwen execution binding differs from ResultCellBinding"
        )


@dataclass(frozen=True, slots=True)
class ResultCell:
    """Immutable canonical payload of one charged cold Qwen/FERTIG path."""

    binding: ResultCellBinding
    cold_qwen_result_json: bytes
    cold_final_result_json: bytes
    cold_fertig_judgment_sha256: str
    cold_fertig_status: Literal["verified", "mismatch", "abstained"] | str
    evaluator_quality_contract_sha256: str
    provenance_units: tuple[ProvenanceUnit, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ResultCellBinding):
            raise TypeError("binding must be a ResultCellBinding")
        qwen_json = bytes(self.cold_qwen_result_json)
        final_json = bytes(self.cold_final_result_json)
        qwen_result = _decode_result(qwen_json, field="cold_qwen_result")
        _decode_result(final_json, field="cold_final_result")
        if (
            qwen_result.status is not ExecutionStatus.OK
            or not isinstance(qwen_result.output, str)
            or not qwen_result.output.strip()
        ):
            raise ValueError(
                "a reusable result cell requires a successful non-empty Qwen result"
            )
        object.__setattr__(self, "cold_qwen_result_json", qwen_json)
        object.__setattr__(self, "cold_final_result_json", final_json)
        object.__setattr__(
            self,
            "cold_fertig_judgment_sha256",
            require_sha256(
                self.cold_fertig_judgment_sha256,
                field="cold_fertig_judgment_sha256",
            ),
        )
        status_value = getattr(self.cold_fertig_status, "value", self.cold_fertig_status)
        if status_value not in _FERTIG_STATUSES:
            raise ValueError("cold_fertig_status is invalid")
        object.__setattr__(self, "cold_fertig_status", status_value)
        object.__setattr__(
            self,
            "evaluator_quality_contract_sha256",
            require_sha256(
                self.evaluator_quality_contract_sha256,
                field="evaluator_quality_contract_sha256",
            ),
        )
        object.__setattr__(
            self,
            "provenance_units",
            _provenance_units(self.provenance_units),
        )
        generation = extract_cold_qwen_generation_receipt(
            qwen_result,
            binding=self.binding,
        )
        baseline_units = tuple(
            unit
            for unit in self.provenance_units
            if unit.name == COLD_QWEN_GENERATION_PROVENANCE_NAME
        )
        if baseline_units != (generation.provenance_unit,):
            raise ResultCellIntegrityError(
                "cold generation receipt lacks its exact provenance unit"
            )

    @classmethod
    def from_cold(
        cls,
        *,
        binding: ResultCellBinding,
        cold_qwen_result: Result,
        cold_final_result: Result,
        cold_fertig_judgment: object,
        cold_fertig_status: Literal["verified", "mismatch", "abstained"] | str,
        evaluator_quality_contract_sha256: str,
        provenance_units: tuple[ProvenanceUnit, ...] = (),
    ) -> "ResultCell":
        status_value = getattr(cold_fertig_status, "value", cold_fertig_status)
        if (
            not isinstance(cold_fertig_judgment, Mapping)
            or cold_fertig_judgment.get("status") != status_value
        ):
            raise ValueError(
                "cold FERTIG status must match the hashed judgment document"
            )
        generation = extract_cold_qwen_generation_receipt(
            cold_qwen_result,
            binding=binding,
        )
        if not isinstance(provenance_units, tuple) or any(
            not isinstance(unit, ProvenanceUnit) for unit in provenance_units
        ):
            raise TypeError("provenance_units must be an immutable unit tuple")
        if any(
            unit.name == COLD_QWEN_GENERATION_PROVENANCE_NAME
            for unit in provenance_units
        ):
            raise ValueError(
                "cold generation provenance is derived, not caller supplied"
            )
        units = (generation.provenance_unit, *provenance_units)
        return cls(
            binding=binding,
            cold_qwen_result_json=_result_json(cold_qwen_result),
            cold_final_result_json=_result_json(cold_final_result),
            cold_fertig_judgment_sha256=artifact_sha256(cold_fertig_judgment),
            cold_fertig_status=status_value,
            evaluator_quality_contract_sha256=(
                evaluator_quality_contract_sha256
            ),
            provenance_units=_provenance_units(tuple(units)),
        )

    @property
    def cold_qwen_result(self) -> Result:
        return _decode_result(
            self.cold_qwen_result_json,
            field="cold_qwen_result",
        )

    @property
    def cold_final_result(self) -> Result:
        return _decode_result(
            self.cold_final_result_json,
            field="cold_final_result",
        )

    @property
    def cold_qwen_result_document(self) -> dict[str, Any]:
        return json.loads(self.cold_qwen_result_json)

    @property
    def cold_final_result_document(self) -> dict[str, Any]:
        return json.loads(self.cold_final_result_json)

    @property
    def cold_qwen_result_sha256(self) -> str:
        return hashlib.sha256(self.cold_qwen_result_json).hexdigest()

    @property
    def cold_final_result_sha256(self) -> str:
        return hashlib.sha256(self.cold_final_result_json).hexdigest()

    @property
    def cold_final_core_sha256(self) -> str:
        return _digest(_result_core(self.cold_final_result))

    @property
    def cold_generation_receipt(self) -> ColdQwenGenerationReceipt:
        return extract_cold_qwen_generation_receipt(
            self.cold_qwen_result,
            binding=self.binding,
        )

    @property
    def cold_generation_provenance_unit(self) -> ProvenanceUnit:
        return self.cold_generation_receipt.provenance_unit

    @property
    def teacher_forward_count(self) -> int:
        return self.cold_generation_receipt.forward_passes

    @property
    def fertig_semantic_certified(self) -> bool:
        return self.cold_fertig_status == "verified"

    @property
    def fertig_exact_judgment(self) -> bool:
        return self.cold_fertig_status in ("verified", "mismatch")

    @property
    def execution_quality_sha256(self) -> str:
        return _digest(
            {
                "evaluator_quality_contract_sha256": (
                    self.evaluator_quality_contract_sha256
                ),
                "payload_sha256": self.payload_sha256,
                "schema": "immer-ooe-result-cell-execution-quality/v1",
            }
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "binding": self.binding.as_record(),
            "binding_sha256": self.binding.sha256,
            "cold_final_result": self.cold_final_result_document,
            "cold_final_result_sha256": self.cold_final_result_sha256,
            "cold_fertig_judgment_sha256": self.cold_fertig_judgment_sha256,
            "cold_fertig_status": self.cold_fertig_status,
            "cold_qwen_result": self.cold_qwen_result_document,
            "cold_qwen_result_sha256": self.cold_qwen_result_sha256,
            "evaluator_quality_contract_sha256": (
                self.evaluator_quality_contract_sha256
            ),
            "provenance_sha256": _digest(
                [unit.to_document() for unit in self.provenance_units]
            ),
            "provenance_units": [
                unit.to_document() for unit in self.provenance_units
            ],
            "teacher_forward_count": self.teacher_forward_count,
        }

    @property
    def payload_sha256(self) -> str:
        return _digest(self.as_record())

    @property
    def sha256(self) -> str:
        return self.payload_sha256

    def to_document(self) -> dict[str, Any]:
        body = self.as_record()
        return {
            "body": body,
            "payload_sha256": _digest(body),
            "schema": RESULT_CELL_SCHEMA,
        }

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_document())
        if len(data) > MAX_RESULT_CELL_BYTES:
            raise ValueError("result cell exceeds its hard byte limit")
        return data

    @classmethod
    def from_bytes(cls, data: bytes) -> "ResultCell":
        value = _strict_json(
            data,
            label="result cell",
            maximum=MAX_RESULT_CELL_BYTES,
        )
        if not isinstance(value, Mapping) or set(value) != {
            "body",
            "payload_sha256",
            "schema",
        }:
            raise ResultCellIntegrityError("result-cell envelope is invalid")
        if value.get("schema") != RESULT_CELL_SCHEMA:
            raise ResultCellIntegrityError("result-cell schema is invalid")
        body = value.get("body")
        expected = {
            "binding",
            "binding_sha256",
            "cold_final_result",
            "cold_final_result_sha256",
            "cold_fertig_judgment_sha256",
            "cold_fertig_status",
            "cold_qwen_result",
            "cold_qwen_result_sha256",
            "evaluator_quality_contract_sha256",
            "provenance_sha256",
            "provenance_units",
            "teacher_forward_count",
        }
        if not isinstance(body, Mapping) or set(body) != expected:
            raise ResultCellIntegrityError("result-cell body is invalid")
        claimed = require_sha256(
            value.get("payload_sha256"),
            field="payload_sha256",
        )
        if claimed != _digest(body):
            raise ResultCellIntegrityError("result-cell payload SHA-256 mismatch")
        try:
            binding = ResultCellBinding.from_record(body["binding"])
            cell = cls(
                binding=binding,
                cold_qwen_result_json=canonical_json_bytes(
                    body["cold_qwen_result"]
                ),
                cold_final_result_json=canonical_json_bytes(
                    body["cold_final_result"]
                ),
                cold_fertig_judgment_sha256=body[
                    "cold_fertig_judgment_sha256"
                ],
                cold_fertig_status=body["cold_fertig_status"],
                evaluator_quality_contract_sha256=body[
                    "evaluator_quality_contract_sha256"
                ],
                provenance_units=tuple(
                    ProvenanceUnit.from_document(document)
                    for document in body["provenance_units"]
                ),
            )
        except ResultCellIntegrityError:
            raise
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ResultCellIntegrityError("result-cell validation failed") from exc
        if body["binding_sha256"] != binding.sha256:
            raise ResultCellIntegrityError("result-cell binding SHA-256 mismatch")
        if body["cold_qwen_result_sha256"] != cell.cold_qwen_result_sha256:
            raise ResultCellIntegrityError("cold Qwen result SHA-256 mismatch")
        if body["cold_final_result_sha256"] != cell.cold_final_result_sha256:
            raise ResultCellIntegrityError("cold final result SHA-256 mismatch")
        if body["teacher_forward_count"] != cell.teacher_forward_count:
            raise ResultCellIntegrityError(
                "teacher-forward baseline differs from cold generation receipt"
            )
        if body["provenance_sha256"] != _digest(
            [unit.to_document() for unit in cell.provenance_units]
        ):
            raise ResultCellIntegrityError("result-cell provenance SHA-256 mismatch")
        if cell.payload_sha256 != claimed or cell.to_bytes() != data:
            raise ResultCellIntegrityError(
                "result cell failed canonical reconstruction"
            )
        return cell

    def assert_current(self, **runtime: Any) -> None:
        self.binding.assert_current(**runtime)

    def assert_feature_bound(self, receipt: QwenOoeFeatureReceipt) -> None:
        self.binding.assert_feature_bound(receipt)


@dataclass(frozen=True, slots=True)
class ResultCellPublication:
    binding_sha256: str
    payload_sha256: str
    object_generation: int | None
    pointer_generation: int | None
    object_created: bool
    pointer_created: bool


class ResultCellBank:
    """Crash-safe content-addressed result cells over ``CrystalStore`` state.

    Objects are published before their binding pointer.  A crash between those
    writes leaves an unreachable content object; retry is idempotent and then
    installs the pointer.  An immutable binding can never be rebound to a
    different payload.
    """

    def __init__(self, store: CrystalStore | str | os.PathLike[str]) -> None:
        self.store = store if isinstance(store, CrystalStore) else CrystalStore(store)
        self.root = Path(self.store.root)

    @staticmethod
    def object_state_name(payload_sha256: str) -> str:
        digest = require_sha256(payload_sha256, field="payload_sha256")
        return f"{_OBJECT_PREFIX}{digest}"

    @staticmethod
    def pointer_state_name(binding_sha256: str) -> str:
        digest = require_sha256(binding_sha256, field="binding_sha256")
        return f"{_POINTER_PREFIX}{digest}"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        path = self.root / _LOCK_NAME
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as exc:
            raise ResultCellIntegrityError("cannot open result-cell bank lock") from exc
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise ResultCellIntegrityError(
                    "result-cell bank lock is not a regular file"
                )
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            linked = path.lstat()
            if (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino):
                raise ResultCellIntegrityError(
                    "result-cell bank lock changed while acquiring it"
                )
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    @staticmethod
    def _pointer_bytes(cell: ResultCell) -> bytes:
        body = {
            "binding_sha256": cell.binding.sha256,
            "payload_sha256": cell.payload_sha256,
        }
        return canonical_json_bytes(
            {
                "body": body,
                "schema": RESULT_CELL_POINTER_SCHEMA,
                "sha256": _digest(body),
            }
        )

    @staticmethod
    def _decode_pointer(data: bytes) -> tuple[str, str]:
        value = _strict_json(
            data,
            label="result-cell pointer",
            maximum=4096,
        )
        if (
            not isinstance(value, Mapping)
            or set(value) != {"body", "schema", "sha256"}
            or value.get("schema") != RESULT_CELL_POINTER_SCHEMA
        ):
            raise ResultCellIntegrityError("result-cell pointer is invalid")
        body = value.get("body")
        if not isinstance(body, Mapping) or set(body) != {
            "binding_sha256",
            "payload_sha256",
        }:
            raise ResultCellIntegrityError("result-cell pointer body is invalid")
        claimed = require_sha256(value.get("sha256"), field="pointer sha256")
        if claimed != _digest(body):
            raise ResultCellIntegrityError("result-cell pointer SHA-256 mismatch")
        return (
            require_sha256(body["binding_sha256"], field="binding_sha256"),
            require_sha256(body["payload_sha256"], field="payload_sha256"),
        )

    def _restore_state(self, name: str) -> bytes:
        try:
            return self.store.restore_state(name)
        except KeyError:
            raise
        except CrystalStoreError as exc:
            raise ResultCellIntegrityError("result-cell state failed integrity") from exc

    def _restore_payload_unlocked(self, payload_sha256: str) -> ResultCell:
        digest = require_sha256(payload_sha256, field="payload_sha256")
        try:
            data = self._restore_state(self.object_state_name(digest))
        except KeyError as exc:
            raise ResultCellIntegrityError(
                "result-cell pointer references a missing content object"
            ) from exc
        cell = ResultCell.from_bytes(data)
        if cell.payload_sha256 != digest:
            raise ResultCellIntegrityError(
                "result-cell content address does not match its payload"
            )
        return cell

    def _pointer_unlocked(self, binding: ResultCellBinding) -> tuple[str, str]:
        try:
            data = self._restore_state(self.pointer_state_name(binding.sha256))
        except KeyError as exc:
            raise ResultCellMissError(
                "no result cell for the exact runtime binding"
            ) from exc
        binding_sha256, payload_sha256 = self._decode_pointer(data)
        if binding_sha256 != binding.sha256:
            raise ResultCellIntegrityError("result-cell pointer binding mismatch")
        return binding_sha256, payload_sha256

    def charge(
        self,
        cell: ResultCell,
        *,
        expected_current_payload_sha256: str | None = None,
    ) -> ResultCellPublication:
        """Atomically expose one cell, with immutable binding CAS semantics."""

        if not isinstance(cell, ResultCell):
            raise TypeError("cell must be a ResultCell")
        expected = (
            None
            if expected_current_payload_sha256 is None
            else require_sha256(
                expected_current_payload_sha256,
                field="expected_current_payload_sha256",
            )
        )
        with self._locked():
            try:
                _binding_sha256, current = self._pointer_unlocked(cell.binding)
            except ResultCellMissError:
                current = None
            if current is not None:
                existing = self._restore_payload_unlocked(current)
                if existing.binding != cell.binding:
                    raise ResultCellIntegrityError(
                        "result-cell pointer resolved to another binding"
                    )
                if expected is not None and expected != current:
                    raise ResultCellConflictError(
                        "result-cell CAS expected another current payload"
                    )
                if current != cell.payload_sha256:
                    raise ResultCellConflictError(
                        "an immutable result-cell binding cannot be rebound"
                    )
                return ResultCellPublication(
                    binding_sha256=cell.binding.sha256,
                    payload_sha256=current,
                    object_generation=None,
                    pointer_generation=None,
                    object_created=False,
                    pointer_created=False,
                )
            if expected is not None:
                raise ResultCellConflictError(
                    "result-cell CAS expected an existing current payload"
                )
            try:
                object_publication = self.store.publish_state(
                    self.object_state_name(cell.payload_sha256),
                    cell.to_bytes(),
                )
                pointer_publication = self.store.publish_state(
                    self.pointer_state_name(cell.binding.sha256),
                    self._pointer_bytes(cell),
                )
            except ManifestConflictError as exc:
                raise ResultCellConflictError("result-cell CAS conflict") from exc
            except CrystalStoreError as exc:
                raise ResultCellIntegrityError(
                    "result-cell publication failed integrity"
                ) from exc
            return ResultCellPublication(
                binding_sha256=cell.binding.sha256,
                payload_sha256=cell.payload_sha256,
                object_generation=object_publication.generation,
                pointer_generation=pointer_publication.generation,
                object_created=object_publication.changed,
                pointer_created=pointer_publication.changed,
            )

    def restore(self, binding: ResultCellBinding) -> ResultCell:
        if not isinstance(binding, ResultCellBinding):
            raise TypeError("binding must be a ResultCellBinding")
        with self._locked():
            _binding_sha256, payload_sha256 = self._pointer_unlocked(binding)
            cell = self._restore_payload_unlocked(payload_sha256)
            if cell.binding != binding:
                raise ResultCellIntegrityError(
                    "result-cell pointer and payload binding disagree"
                )
            return cell

    def restore_current(self, binding: ResultCellBinding, **runtime: Any) -> ResultCell:
        binding.assert_current(**runtime)
        return self.restore(binding)

    def restore_payload(self, payload_sha256: str) -> ResultCell:
        with self._locked():
            return self._restore_payload_unlocked(payload_sha256)


BindingResolver = Callable[[QwenOoeFeatureReceipt], ResultCellBinding]


class ResultCellExecutor:
    """Mount an authenticated result cell as a feature-bound OoE action."""

    def __init__(
        self,
        bank: ResultCellBank,
        binding: ResultCellBinding | BindingResolver,
    ) -> None:
        if not isinstance(bank, ResultCellBank):
            raise TypeError("bank must be a ResultCellBank")
        if not isinstance(binding, ResultCellBinding) and not callable(binding):
            raise TypeError("binding must be a ResultCellBinding or resolver")
        self.bank = bank
        self._binding = binding

    def _resolve_binding(self, receipt: QwenOoeFeatureReceipt) -> ResultCellBinding:
        binding = (
            self._binding(receipt)
            if callable(self._binding)
            else self._binding
        )
        if not isinstance(binding, ResultCellBinding):
            raise ResultCellIntegrityError(
                "result-cell binding resolver returned an invalid binding"
            )
        return binding

    @staticmethod
    def _feature_provenance(
        cell: ResultCell,
        receipt: QwenOoeFeatureReceipt,
    ) -> str:
        pair_binding = cell.cold_generation_provenance_unit.pair_binding_sha256
        if (
            pair_binding not in receipt.verifier_sha256s
            or pair_binding not in receipt.evidence_sha256s
        ):
            raise ResultCellIntegrityError(
                "feature receipt lacks the exact cold-generation provenance unit"
            )
        return pair_binding

    def __call__(self, receipt: QwenOoeFeatureReceipt) -> ActionExecution:
        if not isinstance(receipt, QwenOoeFeatureReceipt):
            raise TypeError("receipt must be a QwenOoeFeatureReceipt")
        binding = self._resolve_binding(receipt)
        binding.assert_feature_bound(receipt)
        cell = self.bank.restore(binding)
        cell.assert_feature_bound(receipt)
        pair_binding = self._feature_provenance(cell, receipt)
        return ActionExecution(
            feature_receipt_sha256=receipt.sha256,
            action="mount_organ",
            executor_sha256=RESULT_CELL_EXECUTOR_SHA256,
            verifier_sha256=pair_binding,
            evidence_sha256=pair_binding,
            quality_sha256=cell.execution_quality_sha256,
            result=cell.cold_qwen_result_document,
            quality_verified=True,
            qwen_forwards=0,
            teacher_baseline_qwen_forwards=cell.teacher_forward_count,
        )


@dataclass(frozen=True, slots=True)
class ResultCellParityReceipt:
    cell_payload_sha256: str
    execution_sha256: str
    cold_qwen_result_sha256: str
    warm_qwen_result_sha256: str
    cold_final_result_sha256: str
    warm_final_result_sha256: str
    cold_final_evidence_sha256: str
    warm_final_evidence_sha256: str
    final_result_core_sha256: str
    fertig_judgment_sha256: str
    evaluator_quality_contract_sha256: str
    teacher_baseline_qwen_forwards: int
    warm_qwen_forwards: int
    saved_qwen_forwards: int
    exact_parity: bool
    qwen_result_document_exact: bool
    final_semantic_core_exact: bool
    fertig_semantic_certified: bool
    fertig_exact_judgment: bool

    def __post_init__(self) -> None:
        for field_name in (
            "cell_payload_sha256",
            "execution_sha256",
            "cold_qwen_result_sha256",
            "warm_qwen_result_sha256",
            "cold_final_result_sha256",
            "warm_final_result_sha256",
            "cold_final_evidence_sha256",
            "warm_final_evidence_sha256",
            "final_result_core_sha256",
            "fertig_judgment_sha256",
            "evaluator_quality_contract_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        baseline = _uint(
            self.teacher_baseline_qwen_forwards,
            field="teacher_baseline_qwen_forwards",
            positive=True,
        )
        warm = _uint(self.warm_qwen_forwards, field="warm_qwen_forwards")
        saved = _uint(self.saved_qwen_forwards, field="saved_qwen_forwards")
        if saved != max(0, baseline - warm):
            raise ValueError("saved_qwen_forwards violates forward accounting")
        if self.exact_parity is not True:
            raise ValueError("a parity receipt must represent exact parity")
        if self.qwen_result_document_exact is not True:
            raise ValueError("Qwen Result document parity must be exact")
        if self.final_semantic_core_exact is not True:
            raise ValueError("final semantic Result core parity must be exact")
        if not isinstance(self.fertig_semantic_certified, bool):
            raise TypeError("fertig_semantic_certified must be bool")
        if not isinstance(self.fertig_exact_judgment, bool):
            raise TypeError("fertig_exact_judgment must be bool")

    def as_record(self) -> dict[str, Any]:
        return {
            "cell_payload_sha256": self.cell_payload_sha256,
            "cold_final_result_sha256": self.cold_final_result_sha256,
            "cold_final_evidence_sha256": self.cold_final_evidence_sha256,
            "cold_qwen_result_sha256": self.cold_qwen_result_sha256,
            "evaluator_quality_contract_sha256": (
                self.evaluator_quality_contract_sha256
            ),
            "exact_parity": self.exact_parity,
            "execution_sha256": self.execution_sha256,
            "fertig_judgment_sha256": self.fertig_judgment_sha256,
            "fertig_exact_judgment": self.fertig_exact_judgment,
            "fertig_semantic_certified": self.fertig_semantic_certified,
            "final_result_core_sha256": self.final_result_core_sha256,
            "final_semantic_core_exact": self.final_semantic_core_exact,
            "qwen_result_document_exact": self.qwen_result_document_exact,
            "saved_qwen_forwards": self.saved_qwen_forwards,
            "teacher_baseline_qwen_forwards": (
                self.teacher_baseline_qwen_forwards
            ),
            "warm_final_result_sha256": self.warm_final_result_sha256,
            "warm_final_evidence_sha256": self.warm_final_evidence_sha256,
            "warm_qwen_forwards": self.warm_qwen_forwards,
            "warm_qwen_result_sha256": self.warm_qwen_result_sha256,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    def to_document(self) -> dict[str, Any]:
        body = self.as_record()
        return {
            "body": body,
            "schema": RESULT_CELL_PARITY_SCHEMA,
            "sha256": _digest(body),
        }


class ResultCellParityVerifier:
    """Verify exact cold/warm parity before any savings commit."""

    def verify(
        self,
        *,
        cell: ResultCell,
        execution: ActionExecution,
        warm_qwen_result: Result,
        warm_final_result: Result,
        warm_fertig_judgment: object,
        evaluator_quality_contract_sha256: str,
    ) -> ResultCellParityReceipt:
        if not isinstance(cell, ResultCell):
            raise TypeError("cell must be a ResultCell")
        if not isinstance(execution, ActionExecution):
            raise TypeError("execution must be an ActionExecution")
        if execution.action != "mount_organ":
            raise ResultCellParityError("result cell executed another action")
        if execution.executor_sha256 != RESULT_CELL_EXECUTOR_SHA256:
            raise ResultCellParityError("result-cell executor identity mismatch")
        if execution.quality_verified is not True:
            raise ResultCellParityError("result-cell execution was not authenticated")
        pair_binding = cell.cold_generation_provenance_unit.pair_binding_sha256
        if (
            execution.verifier_sha256 != pair_binding
            or execution.evidence_sha256 != pair_binding
        ):
            raise ResultCellParityError(
                "result-cell execution provenance-unit mismatch"
            )
        if execution.qwen_forwards != 0:
            raise ResultCellParityError("warm result cell spent a Qwen forward")
        if execution.teacher_baseline_qwen_forwards != cell.teacher_forward_count:
            raise ResultCellParityError("sealed teacher-forward baseline mismatch")
        if execution.quality_sha256 != cell.execution_quality_sha256:
            raise ResultCellParityError("result-cell execution quality binding mismatch")
        if execution.result != cell.cold_qwen_result_document:
            raise ResultCellParityError("mounted Qwen result document changed")
        if not isinstance(warm_qwen_result, Result):
            raise TypeError("warm_qwen_result must be a Result")
        warm_qwen_json = _result_json(warm_qwen_result)
        if warm_qwen_json != cell.cold_qwen_result_json:
            raise ResultCellParityError("cold and warm Qwen Results differ")
        if not isinstance(warm_final_result, Result):
            raise TypeError("warm_final_result must be a Result")
        warm_final_json = _result_json(warm_final_result)
        cold_core = _result_core(cell.cold_final_result)
        warm_core = _result_core(warm_final_result)
        if canonical_json_bytes(cold_core) != canonical_json_bytes(warm_core):
            raise ResultCellParityError(
                "cold and warm final Result status/output differ"
            )
        warm_fertig_sha256 = artifact_sha256(warm_fertig_judgment)
        if warm_fertig_sha256 != cell.cold_fertig_judgment_sha256:
            raise ResultCellParityError("cold and warm FERTIG judgments differ")
        evaluator_sha256 = require_sha256(
            evaluator_quality_contract_sha256,
            field="evaluator_quality_contract_sha256",
        )
        if evaluator_sha256 != cell.evaluator_quality_contract_sha256:
            raise ResultCellParityError("frozen evaluator contract changed")
        return ResultCellParityReceipt(
            cell_payload_sha256=cell.payload_sha256,
            execution_sha256=execution.sha256,
            cold_qwen_result_sha256=cell.cold_qwen_result_sha256,
            warm_qwen_result_sha256=hashlib.sha256(warm_qwen_json).hexdigest(),
            cold_final_result_sha256=cell.cold_final_result_sha256,
            warm_final_result_sha256=hashlib.sha256(warm_final_json).hexdigest(),
            cold_final_evidence_sha256=_digest(
                dict(cell.cold_final_result.evidence)
            ),
            warm_final_evidence_sha256=_digest(dict(warm_final_result.evidence)),
            final_result_core_sha256=_digest(cold_core),
            fertig_judgment_sha256=warm_fertig_sha256,
            evaluator_quality_contract_sha256=evaluator_sha256,
            teacher_baseline_qwen_forwards=cell.teacher_forward_count,
            warm_qwen_forwards=execution.qwen_forwards,
            saved_qwen_forwards=execution.saved_qwen_forwards,
            exact_parity=True,
            qwen_result_document_exact=True,
            final_semantic_core_exact=True,
            fertig_semantic_certified=cell.fertig_semantic_certified,
            fertig_exact_judgment=cell.fertig_exact_judgment,
        )


FrozenEvaluator = Callable[[Result, Result], bool]


@dataclass(frozen=True, slots=True)
class FinalBenchmarkReceipt:
    parity_receipt_sha256: str
    evaluator_quality_contract_sha256: str
    evaluator_quality_verified: bool
    exact_parity: bool
    fertig_semantic_certified: bool
    fertig_exact_judgment: bool
    saved_qwen_forwards: int

    def __post_init__(self) -> None:
        for field_name in (
            "parity_receipt_sha256",
            "evaluator_quality_contract_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        if self.evaluator_quality_verified is not True or self.exact_parity is not True:
            raise ValueError("final benchmark receipt requires verified exact parity")
        if not isinstance(self.fertig_semantic_certified, bool):
            raise TypeError("fertig_semantic_certified must be bool")
        if not isinstance(self.fertig_exact_judgment, bool):
            raise TypeError("fertig_exact_judgment must be bool")
        _uint(self.saved_qwen_forwards, field="saved_qwen_forwards")

    def as_record(self) -> dict[str, Any]:
        return {
            "evaluator_quality_contract_sha256": (
                self.evaluator_quality_contract_sha256
            ),
            "evaluator_quality_verified": self.evaluator_quality_verified,
            "exact_parity": self.exact_parity,
            "fertig_exact_judgment": self.fertig_exact_judgment,
            "fertig_semantic_certified": self.fertig_semantic_certified,
            "parity_receipt_sha256": self.parity_receipt_sha256,
            "saved_qwen_forwards": self.saved_qwen_forwards,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    def to_document(self) -> dict[str, Any]:
        body = self.as_record()
        return {
            "body": body,
            "schema": RESULT_CELL_BENCHMARK_SCHEMA,
            "sha256": _digest(body),
        }


class FinalBenchmarkLayer:
    """The only layer that opens the frozen evaluator callable."""

    def __init__(
        self,
        *,
        evaluator_quality_contract_sha256: str,
        evaluator: FrozenEvaluator,
    ) -> None:
        self.evaluator_quality_contract_sha256 = require_sha256(
            evaluator_quality_contract_sha256,
            field="evaluator_quality_contract_sha256",
        )
        if not callable(evaluator):
            raise TypeError("evaluator must be callable")
        self.__evaluator = evaluator

    def verify(
        self,
        *,
        cell: ResultCell,
        parity: ResultCellParityReceipt,
        warm_final_result: Result,
    ) -> FinalBenchmarkReceipt:
        if not isinstance(cell, ResultCell):
            raise TypeError("cell must be a ResultCell")
        if not isinstance(parity, ResultCellParityReceipt):
            raise TypeError("parity must be a ResultCellParityReceipt")
        if parity.cell_payload_sha256 != cell.payload_sha256:
            raise ResultCellParityError("benchmark parity belongs to another cell")
        if (
            parity.evaluator_quality_contract_sha256
            != self.evaluator_quality_contract_sha256
            or cell.evaluator_quality_contract_sha256
            != self.evaluator_quality_contract_sha256
        ):
            raise ResultCellParityError("benchmark evaluator contract changed")
        if (
            hashlib.sha256(_result_json(warm_final_result)).hexdigest()
            != parity.warm_final_result_sha256
            or _digest(_result_core(warm_final_result))
            != parity.final_result_core_sha256
        ):
            raise ResultCellParityError("benchmark received another warm Result")
        try:
            verified = bool(
                self.__evaluator(cell.cold_final_result, warm_final_result)
            )
        except Exception as exc:
            raise ResultCellParityError("frozen evaluator failed") from exc
        if not verified:
            raise ResultCellParityError("frozen evaluator rejected warm quality")
        return FinalBenchmarkReceipt(
            parity_receipt_sha256=parity.sha256,
            evaluator_quality_contract_sha256=(
                self.evaluator_quality_contract_sha256
            ),
            evaluator_quality_verified=True,
            exact_parity=parity.exact_parity,
            fertig_semantic_certified=parity.fertig_semantic_certified,
            fertig_exact_judgment=parity.fertig_exact_judgment,
            saved_qwen_forwards=parity.saved_qwen_forwards,
        )


__all__ = [
    "COLD_QWEN_BINDING_EVIDENCE_KEY",
    "COLD_QWEN_BINDING_EVIDENCE_SCHEMA",
    "COLD_QWEN_GENERATION_EVIDENCE_KEY",
    "COLD_QWEN_GENERATION_PROVENANCE_NAME",
    "COLD_QWEN_GENERATION_SCHEMA",
    "COLD_QWEN_GENERATION_VERIFIER_SHA256",
    "ColdQwenGenerationReceipt",
    "FinalBenchmarkLayer",
    "FinalBenchmarkReceipt",
    "FrozenEvaluator",
    "MAX_RESULT_CELL_BYTES",
    "MAX_RESULT_DOCUMENT_BYTES",
    "PROVENANCE_UNIT_SCHEMA",
    "ProvenanceUnit",
    "RESULT_CELL_BENCHMARK_SCHEMA",
    "RESULT_CELL_BINDING_SCHEMA",
    "RESULT_CELL_EXECUTOR_SCHEMA",
    "RESULT_CELL_EXECUTOR_SHA256",
    "RESULT_CELL_PARITY_SCHEMA",
    "RESULT_CELL_POINTER_SCHEMA",
    "RESULT_CELL_SCHEMA",
    "ResultCell",
    "ResultCellBank",
    "ResultCellBinding",
    "ResultCellConflictError",
    "ResultCellError",
    "ResultCellExecutor",
    "ResultCellIntegrityError",
    "ResultCellMissError",
    "ResultCellParityError",
    "ResultCellParityReceipt",
    "ResultCellParityVerifier",
    "ResultCellPublication",
    "ResultCellStaleError",
    "attach_cold_qwen_generation_receipt",
    "artifact_sha256",
    "extract_cold_qwen_generation_receipt",
    "qwen_result_binding_evidence",
]
