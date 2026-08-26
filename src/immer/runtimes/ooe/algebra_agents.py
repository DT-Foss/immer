"""Verifier-bound meta-agents over heterogeneous exact operator algebras.

The router in this module has two deliberately separate responsibilities:

* a contextual Thompson process selects one already verified algebra program;
* a behavioral MAP-Elites archive decides which programs remain selectable.

Programs may have unrelated state schemas.  ``HeterogeneousProgramEnsemble``
therefore executes lanes independently and joins receipts only after every
lane has crossed its own ABI boundary.  It never manufactures a sequential
program or a fictitious common schema.
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from .affine_monoid import (
    AffineIntegrityError,
    AffineMonoidRuntime,
    AffineProgram,
    AffineState,
    ExecutionReceipt,
    ProgramReceipt,
)
from .bvn_search import (
    BehavioralElite,
    BehavioralMAPElites,
    ContextualThompsonMutation,
)
from .crystal import (
    CrystalStore,
    CrystalStoreError,
    ManifestConflictError,
    StatePublication,
)
from .identity import canonical_json_bytes, require_sha256


OPERATOR_ALGEBRA_CANDIDATE_SCHEMA = "immer-ooe-operator-algebra-candidate/v1"
ALGEBRA_SELECTION_SCHEMA = "immer-ooe-algebra-selection/v1"
VERIFIER_OUTCOME_SCHEMA = "immer-ooe-algebra-verifier-outcome/v1"
ALGEBRA_UPDATE_SCHEMA = "immer-ooe-algebra-update/v1"
ALGEBRA_ADMISSION_SCHEMA = "immer-ooe-algebra-admission/v1"
ALGEBRA_ROUTER_SCHEMA = "immer-ooe-algebra-router/v1"
EXACT_VERIFIER_ARTIFACT_SCHEMA = "immer-ooe-exact-verifier-artifact/v1"
ENSEMBLE_SCHEMA = "immer-ooe-heterogeneous-ensemble/v1"
ENSEMBLE_RECEIPT_SCHEMA = "immer-ooe-heterogeneous-ensemble-receipt/v1"
ENSEMBLE_RESULT_SCHEMA = "immer-ooe-heterogeneous-ensemble-result/v1"

MAX_SERIALIZED_BYTES = 256 * 1024 * 1024
MAX_VERIFIER_RECEIPT_BYTES = 64 * 1024 * 1024
MAX_CANDIDATES = 4096
MAX_EVIDENCE = 4096
MAX_DESCRIPTOR_DIMENSION = 32
MAX_OBJECTIVES = 32
MAX_ENSEMBLE_LANES = 1024
MAX_VERIFIERS_PER_LANE = 64
MAX_IDENTIFIER_BYTES = 256

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}")


class AlgebraAgentError(ValueError):
    """An algebra-agent contract was violated."""


class AlgebraAgentIntegrityError(AlgebraAgentError):
    """A canonical payload or cross-object binding failed verification."""


class HeterogeneousABIError(AlgebraAgentError):
    """A parallel heterogeneous contract was requested as a sequential ABI."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise AlgebraAgentError(f"{field} is not a canonical identifier")
    if len(value.encode("utf-8")) > MAX_IDENTIFIER_BYTES:
        raise AlgebraAgentError(f"{field} exceeds its byte bound")
    return value


def _integer(
    value: object, *, field: str, minimum: int = 0, maximum: int = (1 << 63) - 1
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AlgebraAgentError(f"{field} must be an integer")
    if not minimum <= value <= maximum:
        raise AlgebraAgentError(f"{field} lies outside its bound")
    return value


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise AlgebraAgentError(f"{field} must be a finite float")
    result = float(value)
    if not math.isfinite(result):
        raise AlgebraAgentError(f"{field} must be finite")
    return 0.0 if result == 0.0 else result


def _sorted_hashes(
    values: Sequence[str], *, field: str, minimum: int = 0, maximum: int
) -> tuple[str, ...]:
    result = tuple(require_sha256(value, field=field) for value in values)
    if not minimum <= len(result) <= maximum:
        raise AlgebraAgentError(f"{field} count lies outside its bound")
    if result != tuple(sorted(result)) or len(set(result)) != len(result):
        raise AlgebraAgentError(f"{field} values must be unique and sorted")
    return result


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(value: object, *, field: str, maximum: int = MAX_SERIALIZED_BYTES) -> bytes:
    if not isinstance(value, str) or not value.isascii():
        raise AlgebraAgentIntegrityError(f"{field} must be canonical base64")
    if len(value) > 4 * ((maximum + 2) // 3):
        raise AlgebraAgentIntegrityError(f"{field} exceeds its byte bound")
    try:
        result = base64.b64decode(value, validate=True)
    except (TypeError, ValueError) as exc:
        raise AlgebraAgentIntegrityError(f"{field} is invalid base64") from exc
    if len(result) > maximum or _b64(result) != value:
        raise AlgebraAgentIntegrityError(f"{field} is not canonical base64")
    return result


def _seal(schema: str, body: Mapping[str, object]) -> bytes:
    exact = dict(body)
    return canonical_json_bytes(
        {"body": exact, "body_sha256": _digest(exact), "schema": schema}
    )


def _open(data: bytes, *, schema: str) -> dict[str, Any]:
    if not isinstance(data, bytes) or len(data) > MAX_SERIALIZED_BYTES:
        raise AlgebraAgentIntegrityError("payload must be bounded immutable bytes")
    try:
        root = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AlgebraAgentIntegrityError("payload is not JSON") from exc
    if (
        not isinstance(root, dict)
        or set(root) != {"body", "body_sha256", "schema"}
        or root.get("schema") != schema
        or not isinstance(root.get("body"), dict)
        or canonical_json_bytes(root) != data
    ):
        raise AlgebraAgentIntegrityError("payload is not canonical")
    body = root["body"]
    try:
        digest = require_sha256(root["body_sha256"], field="body_sha256")
    except (TypeError, ValueError) as exc:
        raise AlgebraAgentIntegrityError("payload body hash is invalid") from exc
    if _digest(body) != digest:
        raise AlgebraAgentIntegrityError("payload body hash mismatch")
    return body


def _exact_keys(value: Mapping[str, object], expected: set[str], *, field: str) -> None:
    if set(value) != expected:
        raise AlgebraAgentIntegrityError(f"invalid {field} fields")


def _hex_floats(values: Sequence[float]) -> list[str]:
    return [value.hex() for value in values]


def _parse_hex_floats(
    value: object, *, field: str, minimum: int = 1, maximum: int = MAX_OBJECTIVES
) -> tuple[float, ...]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise AlgebraAgentIntegrityError(f"{field} has an invalid dimension")
    parsed: list[float] = []
    for item in value:
        if not isinstance(item, str):
            raise AlgebraAgentIntegrityError(f"{field} must use hexadecimal floats")
        try:
            parsed.append(_finite(float.fromhex(item), field=field))
        except (TypeError, ValueError) as exc:
            raise AlgebraAgentIntegrityError(f"{field} float is invalid") from exc
    if _hex_floats(parsed) != value:
        raise AlgebraAgentIntegrityError(f"{field} floats are not canonical")
    return tuple(parsed)


@dataclass(frozen=True, slots=True)
class OperatorAlgebraCandidate:
    """One verified algebra program and all evidence needed to route to it."""

    family: str
    program: AffineProgram
    verifier_sha256: str
    evidence_sha256s: tuple[str, ...]
    behavior_descriptor: tuple[int, ...]
    objectives: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", _identifier(self.family, field="family"))
        if not isinstance(self.program, AffineProgram):
            raise TypeError("program must be an AffineProgram")
        object.__setattr__(
            self,
            "verifier_sha256",
            require_sha256(self.verifier_sha256, field="verifier_sha256"),
        )
        object.__setattr__(
            self,
            "evidence_sha256s",
            _sorted_hashes(
                self.evidence_sha256s,
                field="evidence_sha256",
                minimum=1,
                maximum=MAX_EVIDENCE,
            ),
        )
        descriptor = tuple(
            _integer(
                item,
                field="behavior descriptor coordinate",
                maximum=1_000_000 - 1,
            )
            for item in self.behavior_descriptor
        )
        if not 1 <= len(descriptor) <= MAX_DESCRIPTOR_DIMENSION:
            raise AlgebraAgentError("behavior descriptor has an invalid dimension")
        object.__setattr__(self, "behavior_descriptor", descriptor)
        objectives = tuple(_finite(item, field="objective") for item in self.objectives)
        if not 1 <= len(objectives) <= MAX_OBJECTIVES:
            raise AlgebraAgentError("objectives have an invalid dimension")
        object.__setattr__(self, "objectives", objectives)
        # Issuance is also a complete program/schema/action consistency check.
        ProgramReceipt.issue(self.program)

    @property
    def schema_sha256(self) -> str:
        return self.program.schema.sha256

    @property
    def program_sha256(self) -> str:
        return self.program.sha256

    @property
    def program_receipt(self) -> ProgramReceipt:
        return ProgramReceipt.issue(self.program)

    def _body(self) -> dict[str, object]:
        return {
            "behavior_descriptor": list(self.behavior_descriptor),
            "evidence_sha256s": list(self.evidence_sha256s),
            "family": self.family,
            "objectives_hex": _hex_floats(self.objectives),
            "program_base64": _b64(self.program.to_bytes()),
            "program_receipt_sha256": self.program_receipt.sha256,
            "program_sha256": self.program_sha256,
            "schema_sha256": self.schema_sha256,
            "verifier_sha256": self.verifier_sha256,
        }

    def to_bytes(self) -> bytes:
        return _seal(OPERATOR_ALGEBRA_CANDIDATE_SCHEMA, self._body())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "OperatorAlgebraCandidate":
        body = _open(data, schema=OPERATOR_ALGEBRA_CANDIDATE_SCHEMA)
        _exact_keys(
            body,
            {
                "behavior_descriptor",
                "evidence_sha256s",
                "family",
                "objectives_hex",
                "program_base64",
                "program_receipt_sha256",
                "program_sha256",
                "schema_sha256",
                "verifier_sha256",
            },
            field="operator algebra candidate",
        )
        if not isinstance(body["behavior_descriptor"], list) or not isinstance(
            body["evidence_sha256s"], list
        ):
            raise AlgebraAgentIntegrityError("candidate vectors must be arrays")
        try:
            candidate = cls(
                family=body["family"],
                program=AffineProgram.from_bytes(
                    _unb64(body["program_base64"], field="program_base64")
                ),
                verifier_sha256=body["verifier_sha256"],
                evidence_sha256s=tuple(body["evidence_sha256s"]),
                behavior_descriptor=tuple(body["behavior_descriptor"]),
                objectives=_parse_hex_floats(
                    body["objectives_hex"], field="objectives"
                ),
            )
        except (TypeError, ValueError, AffineIntegrityError) as exc:
            raise AlgebraAgentIntegrityError("invalid operator algebra candidate") from exc
        if candidate._body() != body:
            raise AlgebraAgentIntegrityError("candidate derived bindings changed")
        return candidate


@dataclass(frozen=True, slots=True)
class AlgebraSelectionReceipt:
    context: str
    candidate_sha256: str
    family: str
    program_sha256: str
    schema_sha256: str
    verifier_sha256: str
    decision_index: int
    scores: tuple[tuple[str, float], ...]
    thompson_choice_sha256: str
    router_parent_sha256: str
    router_next_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", _identifier(self.context, field="context"))
        object.__setattr__(self, "family", _identifier(self.family, field="family"))
        for name in (
            "candidate_sha256",
            "program_sha256",
            "schema_sha256",
            "verifier_sha256",
            "thompson_choice_sha256",
            "router_parent_sha256",
            "router_next_sha256",
        ):
            object.__setattr__(
                self, name, require_sha256(getattr(self, name), field=name)
            )
        _integer(self.decision_index, field="decision_index")
        scores = tuple(
            (
                require_sha256(arm, field="score candidate_sha256"),
                _finite(score, field="selection score"),
            )
            for arm, score in self.scores
        )
        if not scores or scores != tuple(sorted(scores)):
            raise AlgebraAgentError("selection scores must be a sorted non-empty vector")
        if len({arm for arm, _ in scores}) != len(scores):
            raise AlgebraAgentError("selection scores contain duplicate candidates")
        if self.candidate_sha256 not in {arm for arm, _ in scores}:
            raise AlgebraAgentError("selected candidate is absent from scores")
        object.__setattr__(self, "scores", scores)

    def _body(self) -> dict[str, object]:
        return {
            "candidate_sha256": self.candidate_sha256,
            "context": self.context,
            "decision_index": self.decision_index,
            "family": self.family,
            "program_sha256": self.program_sha256,
            "router_next_sha256": self.router_next_sha256,
            "router_parent_sha256": self.router_parent_sha256,
            "schema_sha256": self.schema_sha256,
            "scores": [
                {"candidate_sha256": arm, "score_hex": score.hex()}
                for arm, score in self.scores
            ],
            "thompson_choice_sha256": self.thompson_choice_sha256,
            "verifier_sha256": self.verifier_sha256,
        }

    def to_bytes(self) -> bytes:
        return _seal(ALGEBRA_SELECTION_SCHEMA, self._body())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "AlgebraSelectionReceipt":
        body = _open(data, schema=ALGEBRA_SELECTION_SCHEMA)
        expected = {
            "candidate_sha256",
            "context",
            "decision_index",
            "family",
            "program_sha256",
            "router_next_sha256",
            "router_parent_sha256",
            "schema_sha256",
            "scores",
            "thompson_choice_sha256",
            "verifier_sha256",
        }
        _exact_keys(body, expected, field="algebra selection")
        if not isinstance(body["scores"], list):
            raise AlgebraAgentIntegrityError("selection scores must be an array")
        try:
            parsed_scores: list[tuple[str, float]] = []
            for item in body["scores"]:
                if not isinstance(item, dict) or set(item) != {
                    "candidate_sha256",
                    "score_hex",
                }:
                    raise ValueError("invalid score entry")
                if not isinstance(item["score_hex"], str):
                    raise ValueError("invalid score encoding")
                score = _finite(float.fromhex(item["score_hex"]), field="score")
                if score.hex() != item["score_hex"]:
                    raise ValueError("non-canonical score encoding")
                parsed_scores.append((item["candidate_sha256"], score))
            receipt = cls(
                context=body["context"],
                candidate_sha256=body["candidate_sha256"],
                family=body["family"],
                program_sha256=body["program_sha256"],
                schema_sha256=body["schema_sha256"],
                verifier_sha256=body["verifier_sha256"],
                decision_index=body["decision_index"],
                scores=tuple(parsed_scores),
                thompson_choice_sha256=body["thompson_choice_sha256"],
                router_parent_sha256=body["router_parent_sha256"],
                router_next_sha256=body["router_next_sha256"],
            )
        except (TypeError, ValueError) as exc:
            raise AlgebraAgentIntegrityError("invalid algebra selection") from exc
        if receipt._body() != body:
            raise AlgebraAgentIntegrityError("selection derived bindings changed")
        return receipt


@dataclass(frozen=True, slots=True)
class VerifierBoundOutcome:
    """An exact verifier result bound to one routed decision."""

    selection_sha256: str
    candidate_sha256: str
    verifier_sha256: str
    receipt_payload: bytes
    success: bool

    def __post_init__(self) -> None:
        for name in ("selection_sha256", "candidate_sha256", "verifier_sha256"):
            object.__setattr__(
                self, name, require_sha256(getattr(self, name), field=name)
            )
        if not isinstance(self.receipt_payload, bytes):
            raise TypeError("receipt_payload must be immutable bytes")
        if not 1 <= len(self.receipt_payload) <= MAX_VERIFIER_RECEIPT_BYTES:
            raise AlgebraAgentError("verifier receipt payload lies outside its bound")
        if not isinstance(self.success, bool):
            raise TypeError("success must be a boolean")

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(self.receipt_payload).hexdigest()

    @classmethod
    def issue(
        cls,
        selection: AlgebraSelectionReceipt,
        candidate: OperatorAlgebraCandidate,
        *,
        receipt_payload: bytes,
        success: bool,
    ) -> "VerifierBoundOutcome":
        if selection.candidate_sha256 != candidate.sha256:
            raise AlgebraAgentIntegrityError("selection/candidate binding mismatch")
        if selection.verifier_sha256 != candidate.verifier_sha256:
            raise AlgebraAgentIntegrityError("selection verifier binding mismatch")
        return cls(
            selection_sha256=selection.sha256,
            candidate_sha256=candidate.sha256,
            verifier_sha256=candidate.verifier_sha256,
            receipt_payload=receipt_payload,
            success=success,
        )

    def _body(self) -> dict[str, object]:
        return {
            "candidate_sha256": self.candidate_sha256,
            "receipt_base64": _b64(self.receipt_payload),
            "receipt_sha256": self.receipt_sha256,
            "selection_sha256": self.selection_sha256,
            "success": self.success,
            "verifier_sha256": self.verifier_sha256,
        }

    def to_bytes(self) -> bytes:
        return _seal(VERIFIER_OUTCOME_SCHEMA, self._body())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "VerifierBoundOutcome":
        body = _open(data, schema=VERIFIER_OUTCOME_SCHEMA)
        _exact_keys(
            body,
            {
                "candidate_sha256",
                "receipt_base64",
                "receipt_sha256",
                "selection_sha256",
                "success",
                "verifier_sha256",
            },
            field="verifier outcome",
        )
        try:
            result = cls(
                selection_sha256=body["selection_sha256"],
                candidate_sha256=body["candidate_sha256"],
                verifier_sha256=body["verifier_sha256"],
                receipt_payload=_unb64(
                    body["receipt_base64"],
                    field="receipt_base64",
                    maximum=MAX_VERIFIER_RECEIPT_BYTES,
                ),
                success=body["success"],
            )
        except (TypeError, ValueError) as exc:
            raise AlgebraAgentIntegrityError("invalid verifier outcome") from exc
        if result._body() != body:
            raise AlgebraAgentIntegrityError("verifier outcome derived fields changed")
        return result


@dataclass(frozen=True, slots=True)
class AlgebraUpdateReceipt:
    router_parent_sha256: str
    router_next_sha256: str
    selection_sha256: str
    outcome_sha256: str
    candidate_sha256: str
    context: str
    verifier_sha256: str
    verifier_receipt_sha256: str
    success: bool
    posterior_successes: int
    posterior_failures: int

    def __post_init__(self) -> None:
        for name in (
            "router_parent_sha256",
            "router_next_sha256",
            "selection_sha256",
            "outcome_sha256",
            "candidate_sha256",
            "verifier_sha256",
            "verifier_receipt_sha256",
        ):
            object.__setattr__(
                self, name, require_sha256(getattr(self, name), field=name)
            )
        object.__setattr__(self, "context", _identifier(self.context, field="context"))
        if not isinstance(self.success, bool):
            raise TypeError("success must be a boolean")
        _integer(self.posterior_successes, field="posterior_successes")
        _integer(self.posterior_failures, field="posterior_failures")

    def _body(self) -> dict[str, object]:
        return {
            "candidate_sha256": self.candidate_sha256,
            "context": self.context,
            "outcome_sha256": self.outcome_sha256,
            "posterior_failures": self.posterior_failures,
            "posterior_successes": self.posterior_successes,
            "router_next_sha256": self.router_next_sha256,
            "router_parent_sha256": self.router_parent_sha256,
            "selection_sha256": self.selection_sha256,
            "success": self.success,
            "verifier_receipt_sha256": self.verifier_receipt_sha256,
            "verifier_sha256": self.verifier_sha256,
        }

    def to_bytes(self) -> bytes:
        return _seal(ALGEBRA_UPDATE_SCHEMA, self._body())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "AlgebraUpdateReceipt":
        body = _open(data, schema=ALGEBRA_UPDATE_SCHEMA)
        _exact_keys(body, set(cls.__dataclass_fields__), field="algebra update")
        try:
            result = cls(**body)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise AlgebraAgentIntegrityError("invalid algebra update") from exc
        if result._body() != body:
            raise AlgebraAgentIntegrityError("algebra update changed on roundtrip")
        return result


@dataclass(frozen=True, slots=True)
class AlgebraAdmissionReceipt:
    router_parent_sha256: str
    router_next_sha256: str
    archive_parent_sha256: str
    archive_next_sha256: str
    candidate_sha256: str
    admitted: bool
    evicted_candidate_sha256s: tuple[str, ...]
    active_candidate_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "router_parent_sha256",
            "router_next_sha256",
            "archive_parent_sha256",
            "archive_next_sha256",
            "candidate_sha256",
        ):
            object.__setattr__(
                self, name, require_sha256(getattr(self, name), field=name)
            )
        if not isinstance(self.admitted, bool):
            raise TypeError("admitted must be a boolean")
        object.__setattr__(
            self,
            "evicted_candidate_sha256s",
            _sorted_hashes(
                self.evicted_candidate_sha256s,
                field="evicted_candidate_sha256",
                maximum=MAX_CANDIDATES,
            ),
        )
        object.__setattr__(
            self,
            "active_candidate_sha256s",
            _sorted_hashes(
                self.active_candidate_sha256s,
                field="active_candidate_sha256",
                minimum=1,
                maximum=MAX_CANDIDATES,
            ),
        )

    def _body(self) -> dict[str, object]:
        return {
            "active_candidate_sha256s": list(self.active_candidate_sha256s),
            "admitted": self.admitted,
            "archive_next_sha256": self.archive_next_sha256,
            "archive_parent_sha256": self.archive_parent_sha256,
            "candidate_sha256": self.candidate_sha256,
            "evicted_candidate_sha256s": list(self.evicted_candidate_sha256s),
            "router_next_sha256": self.router_next_sha256,
            "router_parent_sha256": self.router_parent_sha256,
        }

    def to_bytes(self) -> bytes:
        return _seal(ALGEBRA_ADMISSION_SCHEMA, self._body())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "AlgebraAdmissionReceipt":
        body = _open(data, schema=ALGEBRA_ADMISSION_SCHEMA)
        _exact_keys(body, set(cls.__dataclass_fields__), field="algebra admission")
        try:
            result = cls(
                router_parent_sha256=body["router_parent_sha256"],
                router_next_sha256=body["router_next_sha256"],
                archive_parent_sha256=body["archive_parent_sha256"],
                archive_next_sha256=body["archive_next_sha256"],
                candidate_sha256=body["candidate_sha256"],
                admitted=body["admitted"],
                evicted_candidate_sha256s=tuple(body["evicted_candidate_sha256s"]),
                active_candidate_sha256s=tuple(body["active_candidate_sha256s"]),
            )
        except (TypeError, ValueError) as exc:
            raise AlgebraAgentIntegrityError("invalid algebra admission") from exc
        if result._body() != body:
            raise AlgebraAgentIntegrityError("algebra admission changed on roundtrip")
        return result


@dataclass(frozen=True, slots=True)
class AlgebraRouterState:
    """Immutable catalog, contextual bandit, and MAP-Elites state."""

    seed_sha256: str
    bin_counts: tuple[int, ...]
    objective_count: int
    max_elites_per_cell: int
    candidates: tuple[OperatorAlgebraCandidate, ...]
    bandit: ContextualThompsonMutation
    archive_payload: bytes

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "seed_sha256",
            require_sha256(self.seed_sha256, field="seed_sha256"),
        )
        bins = tuple(
            _integer(item, field="bin count", minimum=1, maximum=1_000_000)
            for item in self.bin_counts
        )
        if not 1 <= len(bins) <= MAX_DESCRIPTOR_DIMENSION:
            raise AlgebraAgentError("bin_counts has an invalid dimension")
        if math.prod(bins) > 1_000_000:
            raise AlgebraAgentError("MAP-Elites grid exceeds its cell bound")
        object.__setattr__(self, "bin_counts", bins)
        _integer(
            self.objective_count,
            field="objective_count",
            minimum=1,
            maximum=MAX_OBJECTIVES,
        )
        _integer(
            self.max_elites_per_cell,
            field="max_elites_per_cell",
            minimum=1,
            maximum=64,
        )
        candidates = tuple(self.candidates)
        if not 1 <= len(candidates) <= MAX_CANDIDATES:
            raise AlgebraAgentError("router candidate count lies outside its bound")
        if not all(isinstance(item, OperatorAlgebraCandidate) for item in candidates):
            raise TypeError("candidates must be OperatorAlgebraCandidate values")
        hashes = tuple(item.sha256 for item in candidates)
        if hashes != tuple(sorted(hashes)) or len(set(hashes)) != len(hashes):
            raise AlgebraAgentError("router candidates must be unique and SHA-sorted")
        if any(len(item.behavior_descriptor) != len(bins) for item in candidates):
            raise AlgebraAgentError("candidate descriptor dimension mismatch")
        if any(len(item.objectives) != self.objective_count for item in candidates):
            raise AlgebraAgentError("candidate objective dimension mismatch")
        if any(
            any(not 0 <= coordinate < count for coordinate, count in zip(
                item.behavior_descriptor, bins, strict=True
            ))
            for item in candidates
        ):
            raise AlgebraAgentError("candidate descriptor lies outside the grid")
        object.__setattr__(self, "candidates", candidates)
        if not isinstance(self.bandit, ContextualThompsonMutation):
            raise TypeError("bandit must be ContextualThompsonMutation")
        if self.bandit.seed_sha256 != self.seed_sha256:
            raise AlgebraAgentIntegrityError("router/bandit seed binding mismatch")
        if self.bandit.arms != hashes:
            raise AlgebraAgentIntegrityError("router catalog and bandit arms differ")
        if not isinstance(self.archive_payload, bytes):
            raise TypeError("archive_payload must be immutable bytes")
        try:
            archive = BehavioralMAPElites.from_bytes(self.archive_payload)
        except (TypeError, ValueError) as exc:
            raise AlgebraAgentIntegrityError("invalid embedded MAP-Elites archive") from exc
        if (
            archive.bin_counts != bins
            or archive.objective_count != self.objective_count
            or archive.max_elites_per_cell != self.max_elites_per_cell
        ):
            raise AlgebraAgentIntegrityError("router/archive configuration mismatch")
        archive_elites = tuple(
            elite
            for descriptor in sorted(archive._cells)  # noqa: SLF001 - verified snapshot
            for elite in archive.cell(descriptor)
        )
        elite_hashes = tuple(sorted(elite.candidate_sha256 for elite in archive_elites))
        if elite_hashes != hashes:
            raise AlgebraAgentIntegrityError("catalog does not equal archive Pareto frontier")
        by_hash = {item.sha256: item for item in candidates}
        for elite in archive_elites:
            candidate = by_hash[elite.candidate_sha256]
            if (
                elite.descriptor != candidate.behavior_descriptor
                or elite.objectives != candidate.objectives
            ):
                raise AlgebraAgentIntegrityError("archive elite changed candidate behavior")

    @classmethod
    def bootstrap(
        cls,
        candidates: Sequence[OperatorAlgebraCandidate],
        *,
        seed_sha256: str,
        bin_counts: Sequence[int],
        objective_count: int,
        max_elites_per_cell: int = 4,
    ) -> "AlgebraRouterState":
        exact_candidates = tuple(candidates)
        if not exact_candidates:
            raise AlgebraAgentError("bootstrap needs at least one candidate")
        archive = BehavioralMAPElites(
            bin_counts,
            objective_count=objective_count,
            max_elites_per_cell=max_elites_per_cell,
        )
        by_hash: dict[str, OperatorAlgebraCandidate] = {}
        for candidate in sorted(exact_candidates, key=lambda item: item.sha256):
            if candidate.sha256 in by_hash:
                raise AlgebraAgentError("bootstrap contains a duplicate candidate")
            by_hash[candidate.sha256] = candidate
            archive.add(
                BehavioralElite(
                    candidate.sha256,
                    candidate.behavior_descriptor,
                    candidate.objectives,
                )
            )
        active = {
            elite.candidate_sha256
            for descriptor in archive._cells  # noqa: SLF001 - canonical archive API lacks iterator
            for elite in archive.cell(descriptor)
        }
        selected = tuple(by_hash[digest] for digest in sorted(active))
        seed = require_sha256(seed_sha256, field="seed_sha256")
        return cls(
            seed_sha256=seed,
            bin_counts=tuple(bin_counts),
            objective_count=objective_count,
            max_elites_per_cell=max_elites_per_cell,
            candidates=selected,
            bandit=ContextualThompsonMutation(
                tuple(item.sha256 for item in selected), seed
            ),
            archive_payload=archive.to_bytes(),
        )

    @property
    def archive(self) -> BehavioralMAPElites:
        return BehavioralMAPElites.from_bytes(self.archive_payload)

    @property
    def archive_sha256(self) -> str:
        return hashlib.sha256(self.archive_payload).hexdigest()

    def candidate(self, candidate_sha256: str) -> OperatorAlgebraCandidate:
        wanted = require_sha256(candidate_sha256, field="candidate_sha256")
        for candidate in self.candidates:
            if candidate.sha256 == wanted:
                return candidate
        raise KeyError(f"unknown active algebra candidate: {wanted}")

    def _with_archive_and_candidates(
        self,
        archive: BehavioralMAPElites,
        candidate_by_hash: Mapping[str, OperatorAlgebraCandidate],
    ) -> "AlgebraRouterState":
        active_hashes = tuple(
            sorted(
                elite.candidate_sha256
                for descriptor in archive._cells  # noqa: SLF001
                for elite in archive.cell(descriptor)
            )
        )
        if not active_hashes:
            raise AlgebraAgentIntegrityError("MAP-Elites admission emptied the router")
        try:
            active = tuple(candidate_by_hash[digest] for digest in active_hashes)
        except KeyError as exc:
            raise AlgebraAgentIntegrityError("archive references an unknown candidate") from exc
        posteriors = tuple(
            item for item in self.bandit.posteriors if item.arm in set(active_hashes)
        )
        bandit = ContextualThompsonMutation(
            active_hashes,
            self.seed_sha256,
            decision_index=self.bandit.decision_index,
            posteriors=posteriors,
        )
        return replace(
            self,
            candidates=active,
            bandit=bandit,
            archive_payload=archive.to_bytes(),
        )

    def admit(
        self, candidate: OperatorAlgebraCandidate
    ) -> tuple["AlgebraRouterState", AlgebraAdmissionReceipt]:
        if not isinstance(candidate, OperatorAlgebraCandidate):
            raise TypeError("candidate must be an OperatorAlgebraCandidate")
        if len(candidate.behavior_descriptor) != len(self.bin_counts):
            raise AlgebraAgentError("candidate descriptor dimension mismatch")
        if len(candidate.objectives) != self.objective_count:
            raise AlgebraAgentError("candidate objective dimension mismatch")
        parent_sha = self.sha256
        parent_archive_sha = self.archive_sha256
        archive = self.archive
        changed = archive.add(
            BehavioralElite(
                candidate.sha256,
                candidate.behavior_descriptor,
                candidate.objectives,
            )
        )
        if not changed:
            receipt = AlgebraAdmissionReceipt(
                router_parent_sha256=parent_sha,
                router_next_sha256=parent_sha,
                archive_parent_sha256=parent_archive_sha,
                archive_next_sha256=parent_archive_sha,
                candidate_sha256=candidate.sha256,
                admitted=False,
                evicted_candidate_sha256s=(),
                active_candidate_sha256s=tuple(
                    item.sha256 for item in self.candidates
                ),
            )
            return self, receipt
        known = {item.sha256: item for item in self.candidates}
        known[candidate.sha256] = candidate
        updated = self._with_archive_and_candidates(archive, known)
        before = {item.sha256 for item in self.candidates}
        after = {item.sha256 for item in updated.candidates}
        receipt = AlgebraAdmissionReceipt(
            router_parent_sha256=parent_sha,
            router_next_sha256=updated.sha256,
            archive_parent_sha256=parent_archive_sha,
            archive_next_sha256=updated.archive_sha256,
            candidate_sha256=candidate.sha256,
            admitted=candidate.sha256 in after,
            evicted_candidate_sha256s=tuple(sorted(before - after)),
            active_candidate_sha256s=tuple(sorted(after)),
        )
        return updated, receipt

    def choose(
        self, context: str
    ) -> tuple[
        OperatorAlgebraCandidate,
        AlgebraSelectionReceipt,
        "AlgebraRouterState",
    ]:
        exact_context = _identifier(context, field="context")
        parent_sha = self.sha256
        choice, advanced_bandit = self.bandit.choose(exact_context)
        advanced = replace(self, bandit=advanced_bandit)
        candidate = self.candidate(choice.arm)
        receipt = AlgebraSelectionReceipt(
            context=exact_context,
            candidate_sha256=candidate.sha256,
            family=candidate.family,
            program_sha256=candidate.program_sha256,
            schema_sha256=candidate.schema_sha256,
            verifier_sha256=candidate.verifier_sha256,
            decision_index=choice.decision_index,
            scores=choice.scores,
            thompson_choice_sha256=choice.sha256,
            router_parent_sha256=parent_sha,
            router_next_sha256=advanced.sha256,
        )
        return candidate, receipt, advanced

    def observe(
        self,
        selection: AlgebraSelectionReceipt,
        outcome: VerifierBoundOutcome,
    ) -> tuple["AlgebraRouterState", AlgebraUpdateReceipt]:
        if not isinstance(selection, AlgebraSelectionReceipt):
            raise TypeError("selection must be an AlgebraSelectionReceipt")
        if not isinstance(outcome, VerifierBoundOutcome):
            raise TypeError("outcome must be a VerifierBoundOutcome")
        if self.sha256 != selection.router_next_sha256:
            raise AlgebraAgentIntegrityError("router is not at the selected next state")
        if self.bandit.decision_index != selection.decision_index + 1:
            raise AlgebraAgentIntegrityError(
                "selection decision index did not produce this router state"
            )
        parent_bandit = replace(
            self.bandit,
            decision_index=selection.decision_index,
        )
        parent = replace(self, bandit=parent_bandit)
        if parent.sha256 != selection.router_parent_sha256:
            raise AlgebraAgentIntegrityError(
                "selection parent state is not reproducible"
            )
        expected_candidate, expected_selection, expected_advanced = parent.choose(
            selection.context
        )
        if (
            expected_advanced != self
            or expected_candidate.sha256 != selection.candidate_sha256
            or expected_selection != selection
        ):
            raise AlgebraAgentIntegrityError(
                "selection is not the deterministic Thompson decision"
            )
        candidate = self.candidate(selection.candidate_sha256)
        if (
            selection.family != candidate.family
            or selection.program_sha256 != candidate.program_sha256
            or selection.schema_sha256 != candidate.schema_sha256
            or selection.verifier_sha256 != candidate.verifier_sha256
        ):
            raise AlgebraAgentIntegrityError("selection candidate bindings changed")
        if (
            outcome.selection_sha256 != selection.sha256
            or outcome.candidate_sha256 != candidate.sha256
            or outcome.verifier_sha256 != candidate.verifier_sha256
        ):
            raise AlgebraAgentIntegrityError("verifier outcome binding mismatch")
        parent_sha = self.sha256
        updated_bandit = self.bandit.observe(
            selection.context,
            candidate.sha256,
            success=outcome.success,
        )
        updated = replace(self, bandit=updated_bandit)
        posterior = updated.bandit.posterior(selection.context, candidate.sha256)
        receipt = AlgebraUpdateReceipt(
            router_parent_sha256=parent_sha,
            router_next_sha256=updated.sha256,
            selection_sha256=selection.sha256,
            outcome_sha256=outcome.sha256,
            candidate_sha256=candidate.sha256,
            context=selection.context,
            verifier_sha256=outcome.verifier_sha256,
            verifier_receipt_sha256=outcome.receipt_sha256,
            success=outcome.success,
            posterior_successes=posterior.successes,
            posterior_failures=posterior.failures,
        )
        return updated, receipt

    def _body(self) -> dict[str, object]:
        return {
            "active_candidate_sha256s": [item.sha256 for item in self.candidates],
            "archive_base64": _b64(self.archive_payload),
            "archive_sha256": self.archive_sha256,
            "bandit_base64": _b64(self.bandit.to_bytes()),
            "bandit_sha256": self.bandit.sha256,
            "bin_counts": list(self.bin_counts),
            "candidates_base64": [_b64(item.to_bytes()) for item in self.candidates],
            "decision_index": self.bandit.decision_index,
            "max_elites_per_cell": self.max_elites_per_cell,
            "objective_count": self.objective_count,
            "seed_sha256": self.seed_sha256,
        }

    def to_bytes(self) -> bytes:
        return _seal(ALGEBRA_ROUTER_SCHEMA, self._body())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "AlgebraRouterState":
        body = _open(data, schema=ALGEBRA_ROUTER_SCHEMA)
        expected = {
            "active_candidate_sha256s",
            "archive_base64",
            "archive_sha256",
            "bandit_base64",
            "bandit_sha256",
            "bin_counts",
            "candidates_base64",
            "decision_index",
            "max_elites_per_cell",
            "objective_count",
            "seed_sha256",
        }
        _exact_keys(body, expected, field="algebra router")
        if not all(
            isinstance(body[name], list)
            for name in ("active_candidate_sha256s", "bin_counts", "candidates_base64")
        ):
            raise AlgebraAgentIntegrityError("router vectors must be arrays")
        try:
            result = cls(
                seed_sha256=body["seed_sha256"],
                bin_counts=tuple(body["bin_counts"]),
                objective_count=body["objective_count"],
                max_elites_per_cell=body["max_elites_per_cell"],
                candidates=tuple(
                    OperatorAlgebraCandidate.from_bytes(
                        _unb64(item, field="candidate_base64")
                    )
                    for item in body["candidates_base64"]
                ),
                bandit=ContextualThompsonMutation.from_bytes(
                    _unb64(body["bandit_base64"], field="bandit_base64")
                ),
                archive_payload=_unb64(
                    body["archive_base64"], field="archive_base64"
                ),
            )
        except (TypeError, ValueError) as exc:
            raise AlgebraAgentIntegrityError("invalid algebra router") from exc
        if result._body() != body:
            raise AlgebraAgentIntegrityError("router derived bindings changed")
        return result


class AlgebraRouterBank:
    """Atomic CrystalStore persistence for the complete router/catalog snapshot."""

    _PREFIX = "ooe-algebra-router-catalog/v1:"

    def __init__(self, store: CrystalStore | str) -> None:
        self.store = store if isinstance(store, CrystalStore) else CrystalStore(store)

    @classmethod
    def state_name(cls, name: str) -> str:
        return cls._PREFIX + _identifier(name, field="router name")

    def publish(
        self,
        name: str,
        state: AlgebraRouterState,
        *,
        expected_sha256: str | None = None,
    ) -> StatePublication:
        if not isinstance(state, AlgebraRouterState):
            raise TypeError("state must be an AlgebraRouterState")
        try:
            publication = self.store.publish_state(
                self.state_name(name),
                state.to_bytes(),
                expected_sha256=expected_sha256,
            )
        except ManifestConflictError:
            raise
        except CrystalStoreError as exc:
            raise AlgebraAgentIntegrityError("router publication failed") from exc
        if publication.payload_sha256 != state.sha256:
            raise AlgebraAgentIntegrityError("published router digest changed")
        return publication

    def restore(self, name: str) -> AlgebraRouterState:
        try:
            payload = self.store.restore_state(self.state_name(name))
        except KeyError:
            raise
        except CrystalStoreError as exc:
            raise AlgebraAgentIntegrityError("router state failed store integrity") from exc
        return AlgebraRouterState.from_bytes(payload)


@dataclass(frozen=True, slots=True)
class ExactVerifierArtifact:
    """Opaque exact-verifier receipt with an explicit verifier identity."""

    verifier_sha256: str
    payload: bytes

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "verifier_sha256",
            require_sha256(self.verifier_sha256, field="verifier_sha256"),
        )
        if not isinstance(self.payload, bytes):
            raise TypeError("exact verifier payload must be immutable bytes")
        if not 1 <= len(self.payload) <= MAX_VERIFIER_RECEIPT_BYTES:
            raise AlgebraAgentError("exact verifier payload lies outside its bound")

    @property
    def payload_sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    def _body(self) -> dict[str, object]:
        return {
            "payload_base64": _b64(self.payload),
            "payload_sha256": self.payload_sha256,
            "verifier_sha256": self.verifier_sha256,
        }

    def to_bytes(self) -> bytes:
        return _seal(EXACT_VERIFIER_ARTIFACT_SCHEMA, self._body())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "ExactVerifierArtifact":
        body = _open(data, schema=EXACT_VERIFIER_ARTIFACT_SCHEMA)
        _exact_keys(
            body,
            {"payload_base64", "payload_sha256", "verifier_sha256"},
            field="exact verifier artifact",
        )
        try:
            result = cls(
                verifier_sha256=body["verifier_sha256"],
                payload=_unb64(
                    body["payload_base64"],
                    field="payload_base64",
                    maximum=MAX_VERIFIER_RECEIPT_BYTES,
                ),
            )
        except (TypeError, ValueError) as exc:
            raise AlgebraAgentIntegrityError("invalid exact verifier artifact") from exc
        if result._body() != body:
            raise AlgebraAgentIntegrityError("verifier artifact derived fields changed")
        return result


@dataclass(frozen=True, slots=True)
class ParallelProgramLane:
    """One independent program ABI inside a heterogeneous ensemble."""

    lane_id: str
    program: AffineProgram
    initial_state: AffineState
    verifier_sha256s: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "lane_id", _identifier(self.lane_id, field="lane_id"))
        if not isinstance(self.program, AffineProgram):
            raise TypeError("lane program must be an AffineProgram")
        if not isinstance(self.initial_state, AffineState):
            raise TypeError("lane initial_state must be an AffineState")
        self.program.schema.validate_state(self.initial_state)
        object.__setattr__(
            self,
            "verifier_sha256s",
            _sorted_hashes(
                self.verifier_sha256s,
                field="lane verifier_sha256",
                maximum=MAX_VERIFIERS_PER_LANE,
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "initial_state_base64": _b64(self.initial_state.to_bytes()),
            "initial_state_sha256": self.initial_state.sha256,
            "lane_id": self.lane_id,
            "program_base64": _b64(self.program.to_bytes()),
            "program_sha256": self.program.sha256,
            "schema_sha256": self.program.schema.sha256,
            "verifier_sha256s": list(self.verifier_sha256s),
        }

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    @classmethod
    def from_record(cls, value: object) -> "ParallelProgramLane":
        if not isinstance(value, dict):
            raise AlgebraAgentIntegrityError("ensemble lane must be an object")
        _exact_keys(
            value,
            {
                "initial_state_base64",
                "initial_state_sha256",
                "lane_id",
                "program_base64",
                "program_sha256",
                "schema_sha256",
                "verifier_sha256s",
            },
            field="ensemble lane",
        )
        if not isinstance(value["verifier_sha256s"], list):
            raise AlgebraAgentIntegrityError("lane verifier vector must be an array")
        try:
            program = AffineProgram.from_bytes(
                _unb64(value["program_base64"], field="program_base64")
            )
            result = cls(
                lane_id=value["lane_id"],
                program=program,
                initial_state=AffineState.from_bytes(
                    _unb64(
                        value["initial_state_base64"], field="initial_state_base64"
                    ),
                    schema=program.schema,
                ),
                verifier_sha256s=tuple(value["verifier_sha256s"]),
            )
        except (TypeError, ValueError, AffineIntegrityError) as exc:
            raise AlgebraAgentIntegrityError("invalid ensemble lane") from exc
        if result.to_record() != value:
            raise AlgebraAgentIntegrityError("ensemble lane derived bindings changed")
        return result


@dataclass(frozen=True, slots=True)
class ParallelLaneResult:
    lane_id: str
    program: AffineProgram
    initial_state: AffineState
    final_state: AffineState
    program_receipt: ProgramReceipt
    execution_receipt: ExecutionReceipt
    verifier_artifacts: tuple[ExactVerifierArtifact, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "lane_id", _identifier(self.lane_id, field="lane_id"))
        if not isinstance(self.program, AffineProgram):
            raise TypeError("program must be an AffineProgram")
        if not isinstance(self.initial_state, AffineState):
            raise TypeError("initial_state must be an AffineState")
        if not isinstance(self.final_state, AffineState):
            raise TypeError("final_state must be an AffineState")
        if not isinstance(self.program_receipt, ProgramReceipt):
            raise TypeError("program_receipt must be a ProgramReceipt")
        if not isinstance(self.execution_receipt, ExecutionReceipt):
            raise TypeError("execution_receipt must be an ExecutionReceipt")
        artifacts = tuple(self.verifier_artifacts)
        if not all(isinstance(item, ExactVerifierArtifact) for item in artifacts):
            raise TypeError("verifier_artifacts must be ExactVerifierArtifact values")
        artifact_hashes = tuple(item.sha256 for item in artifacts)
        if artifact_hashes != tuple(sorted(artifact_hashes)) or len(
            set(artifact_hashes)
        ) != len(artifact_hashes):
            raise AlgebraAgentError("verifier artifacts must be unique and SHA-sorted")
        if (
            self.initial_state.schema_sha256 != self.program.schema.sha256
            or self.final_state.schema_sha256 != self.program.schema.sha256
            or self.program_receipt.schema_sha256 != self.final_state.schema_sha256
            or self.execution_receipt.schema_sha256 != self.final_state.schema_sha256
            or self.program_receipt.program_sha256
            != self.program.sha256
            or self.execution_receipt.program_sha256 != self.program.sha256
            or self.execution_receipt.initial_state_sha256
            != self.initial_state.sha256
            or self.execution_receipt.final_state_sha256 != self.final_state.sha256
            or self.execution_receipt.verifier_sha256s != artifact_hashes
        ):
            raise AlgebraAgentIntegrityError(
                "lane state, program, execution, or verifier receipts were spliced"
            )
        replay = AffineMonoidRuntime.execute(
            self.program,
            initial_state=self.initial_state,
            verifier_sha256s=artifact_hashes,
        )
        if (
            replay.state != self.final_state
            or replay.program_receipt != self.program_receipt
            or replay.execution_receipt != self.execution_receipt
        ):
            raise AlgebraAgentIntegrityError(
                "lane result failed exact standalone replay"
            )
        object.__setattr__(self, "verifier_artifacts", artifacts)

    def to_record(self) -> dict[str, object]:
        return {
            "execution_receipt_base64": _b64(self.execution_receipt.to_bytes()),
            "execution_receipt_sha256": self.execution_receipt.sha256,
            "final_state_base64": _b64(self.final_state.to_bytes()),
            "final_state_sha256": self.final_state.sha256,
            "initial_state_base64": _b64(self.initial_state.to_bytes()),
            "initial_state_sha256": self.initial_state.sha256,
            "lane_id": self.lane_id,
            "program_base64": _b64(self.program.to_bytes()),
            "program_sha256": self.program.sha256,
            "program_receipt_base64": _b64(self.program_receipt.to_bytes()),
            "program_receipt_sha256": self.program_receipt.sha256,
            "verifier_artifacts_base64": [
                _b64(item.to_bytes()) for item in self.verifier_artifacts
            ],
            "verifier_artifact_sha256s": [
                item.sha256 for item in self.verifier_artifacts
            ],
        }

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    @classmethod
    def from_record(
        cls, value: object, *, schema_program: AffineProgram
    ) -> "ParallelLaneResult":
        if not isinstance(value, dict):
            raise AlgebraAgentIntegrityError("lane result must be an object")
        _exact_keys(
            value,
            {
                "execution_receipt_base64",
                "execution_receipt_sha256",
                "final_state_base64",
                "final_state_sha256",
                "initial_state_base64",
                "initial_state_sha256",
                "lane_id",
                "program_base64",
                "program_sha256",
                "program_receipt_base64",
                "program_receipt_sha256",
                "verifier_artifacts_base64",
                "verifier_artifact_sha256s",
            },
            field="lane result",
        )
        if not isinstance(value["verifier_artifacts_base64"], list) or not isinstance(
            value["verifier_artifact_sha256s"], list
        ):
            raise AlgebraAgentIntegrityError("lane result verifier vectors must be arrays")
        try:
            embedded_program = AffineProgram.from_bytes(
                _unb64(value["program_base64"], field="program_base64")
            )
            if embedded_program.to_bytes() != schema_program.to_bytes():
                raise AlgebraAgentIntegrityError(
                    "lane result substituted another same-schema program"
                )
            result = cls(
                lane_id=value["lane_id"],
                program=embedded_program,
                initial_state=AffineState.from_bytes(
                    _unb64(
                        value["initial_state_base64"],
                        field="initial_state_base64",
                    ),
                    schema=embedded_program.schema,
                ),
                final_state=AffineState.from_bytes(
                    _unb64(value["final_state_base64"], field="final_state_base64"),
                    schema=schema_program.schema,
                ),
                program_receipt=ProgramReceipt.from_bytes(
                    _unb64(
                        value["program_receipt_base64"],
                        field="program_receipt_base64",
                    )
                ),
                execution_receipt=ExecutionReceipt.from_bytes(
                    _unb64(
                        value["execution_receipt_base64"],
                        field="execution_receipt_base64",
                    )
                ),
                verifier_artifacts=tuple(
                    ExactVerifierArtifact.from_bytes(
                        _unb64(item, field="verifier_artifact_base64")
                    )
                    for item in value["verifier_artifacts_base64"]
                ),
            )
        except AlgebraAgentIntegrityError:
            raise
        except (TypeError, ValueError, AffineIntegrityError) as exc:
            raise AlgebraAgentIntegrityError("invalid lane result") from exc
        if result.to_record() != value:
            raise AlgebraAgentIntegrityError("lane result derived bindings changed")
        return result


@dataclass(frozen=True, slots=True)
class HeterogeneousProgramEnsemble:
    """Parallel product of independent programs with no sequential ABI claim."""

    name: str
    lanes: tuple[ParallelProgramLane, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, field="ensemble name"))
        lanes = tuple(self.lanes)
        if not 1 <= len(lanes) <= MAX_ENSEMBLE_LANES:
            raise AlgebraAgentError("ensemble lane count lies outside its bound")
        if not all(isinstance(item, ParallelProgramLane) for item in lanes):
            raise TypeError("lanes must be ParallelProgramLane values")
        if lanes != tuple(sorted(lanes, key=lambda item: item.lane_id)) or len(
            {item.lane_id for item in lanes}
        ) != len(lanes):
            raise AlgebraAgentError("ensemble lanes must be unique and ID-sorted")
        object.__setattr__(self, "lanes", lanes)

    @property
    def schema_sha256s(self) -> tuple[str, ...]:
        return tuple(lane.program.schema.sha256 for lane in self.lanes)

    @property
    def heterogeneous(self) -> bool:
        return len(set(self.schema_sha256s)) > 1

    @property
    def abi_mode(self) -> str:
        return "parallel-independent"

    def as_sequential_program(self) -> AffineProgram:
        raise HeterogeneousABIError(
            "an independent parallel ensemble has no sequential program ABI"
        )

    def _body(self) -> dict[str, object]:
        return {
            "abi_mode": self.abi_mode,
            "heterogeneous": self.heterogeneous,
            "lanes": [lane.to_record() for lane in self.lanes],
            "name": self.name,
            "schema_sha256s": list(self.schema_sha256s),
        }

    def to_bytes(self) -> bytes:
        return _seal(ENSEMBLE_SCHEMA, self._body())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "HeterogeneousProgramEnsemble":
        body = _open(data, schema=ENSEMBLE_SCHEMA)
        _exact_keys(
            body,
            {"abi_mode", "heterogeneous", "lanes", "name", "schema_sha256s"},
            field="heterogeneous ensemble",
        )
        if not isinstance(body["lanes"], list) or not isinstance(
            body["schema_sha256s"], list
        ):
            raise AlgebraAgentIntegrityError("ensemble vectors must be arrays")
        try:
            result = cls(
                name=body["name"],
                lanes=tuple(ParallelProgramLane.from_record(item) for item in body["lanes"]),
            )
        except (TypeError, ValueError) as exc:
            raise AlgebraAgentIntegrityError("invalid heterogeneous ensemble") from exc
        if result._body() != body:
            raise AlgebraAgentIntegrityError("ensemble derived bindings changed")
        return result

    def execute(
        self,
        verifier_artifacts: Mapping[str, Sequence[ExactVerifierArtifact]] | None = None,
        *,
        max_workers: int | None = None,
    ) -> "HeterogeneousEnsembleResult":
        supplied = {} if verifier_artifacts is None else dict(verifier_artifacts)
        unknown = set(supplied) - {lane.lane_id for lane in self.lanes}
        if unknown:
            raise AlgebraAgentError("verifier artifacts contain unknown lanes")

        lane_artifacts: dict[str, tuple[ExactVerifierArtifact, ...]] = {}
        for lane in self.lanes:
            artifacts = tuple(supplied.get(lane.lane_id, ()))
            if not all(isinstance(item, ExactVerifierArtifact) for item in artifacts):
                raise TypeError("verifier artifact map contains an invalid value")
            artifacts = tuple(sorted(artifacts, key=lambda item: item.sha256))
            identities = tuple(sorted(item.verifier_sha256 for item in artifacts))
            if identities != lane.verifier_sha256s:
                raise AlgebraAgentIntegrityError(
                    f"lane {lane.lane_id} exact verifier set does not match its contract"
                )
            lane_artifacts[lane.lane_id] = artifacts

        def run(lane: ParallelProgramLane) -> ParallelLaneResult:
            artifacts = lane_artifacts[lane.lane_id]
            executed = AffineMonoidRuntime.execute(
                lane.program,
                initial_state=lane.initial_state,
                verifier_sha256s=tuple(item.sha256 for item in artifacts),
            )
            return ParallelLaneResult(
                lane_id=lane.lane_id,
                program=lane.program,
                initial_state=lane.initial_state,
                final_state=executed.state,
                program_receipt=executed.program_receipt,
                execution_receipt=executed.execution_receipt,
                verifier_artifacts=artifacts,
            )

        worker_count = len(self.lanes) if max_workers is None else _integer(
            max_workers,
            field="max_workers",
            minimum=1,
            maximum=MAX_ENSEMBLE_LANES,
        )
        worker_count = min(worker_count, len(self.lanes))
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            results = tuple(pool.map(run, self.lanes))
        receipt = HeterogeneousEnsembleReceipt.issue(self, results)
        return HeterogeneousEnsembleResult(self, results, receipt)


@dataclass(frozen=True, slots=True)
class HeterogeneousEnsembleReceipt:
    ensemble_sha256: str
    lane_joins: tuple[
        tuple[str, str, str, str, tuple[str, ...]], ...
    ]
    total_work_units: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ensemble_sha256",
            require_sha256(self.ensemble_sha256, field="ensemble_sha256"),
        )
        joins: list[tuple[str, str, str, str, tuple[str, ...]]] = []
        for lane_id, program_sha, execution_sha, final_sha, artifact_hashes in self.lane_joins:
            exact_id = _identifier(lane_id, field="lane_id")
            exact_hashes = tuple(
                require_sha256(value, field="lane join SHA-256")
                for value in (program_sha, execution_sha, final_sha)
            )
            exact_artifacts = _sorted_hashes(
                artifact_hashes,
                field="verifier artifact SHA-256",
                maximum=MAX_VERIFIERS_PER_LANE,
            )
            joins.append((exact_id, *exact_hashes, exact_artifacts))
        normalized = tuple(joins)
        if normalized != tuple(sorted(normalized)) or len(
            {item[0] for item in normalized}
        ) != len(normalized):
            raise AlgebraAgentError("ensemble joins must be unique and lane-sorted")
        object.__setattr__(self, "lane_joins", normalized)
        _integer(self.total_work_units, field="total_work_units", maximum=(1 << 127) - 1)

    @classmethod
    def issue(
        cls,
        ensemble: HeterogeneousProgramEnsemble,
        results: Sequence[ParallelLaneResult],
    ) -> "HeterogeneousEnsembleReceipt":
        return cls(
            ensemble_sha256=ensemble.sha256,
            lane_joins=tuple(
                (
                    result.lane_id,
                    result.program_receipt.sha256,
                    result.execution_receipt.sha256,
                    result.final_state.sha256,
                    tuple(item.sha256 for item in result.verifier_artifacts),
                )
                for result in results
            ),
            total_work_units=sum(
                result.execution_receipt.work_units for result in results
            ),
        )

    def _body(self) -> dict[str, object]:
        return {
            "ensemble_sha256": self.ensemble_sha256,
            "lane_joins": [
                {
                    "execution_receipt_sha256": execution_sha,
                    "final_state_sha256": final_sha,
                    "lane_id": lane_id,
                    "program_receipt_sha256": program_sha,
                    "verifier_artifact_sha256s": list(artifact_hashes),
                }
                for lane_id, program_sha, execution_sha, final_sha, artifact_hashes in self.lane_joins
            ],
            "total_work_units": self.total_work_units,
        }

    def to_bytes(self) -> bytes:
        return _seal(ENSEMBLE_RECEIPT_SCHEMA, self._body())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "HeterogeneousEnsembleReceipt":
        body = _open(data, schema=ENSEMBLE_RECEIPT_SCHEMA)
        _exact_keys(
            body,
            {"ensemble_sha256", "lane_joins", "total_work_units"},
            field="ensemble receipt",
        )
        if not isinstance(body["lane_joins"], list):
            raise AlgebraAgentIntegrityError("ensemble joins must be an array")
        joins: list[tuple[str, str, str, str, tuple[str, ...]]] = []
        for item in body["lane_joins"]:
            if not isinstance(item, dict):
                raise AlgebraAgentIntegrityError("ensemble join must be an object")
            _exact_keys(
                item,
                {
                    "execution_receipt_sha256",
                    "final_state_sha256",
                    "lane_id",
                    "program_receipt_sha256",
                    "verifier_artifact_sha256s",
                },
                field="ensemble join",
            )
            if not isinstance(item["verifier_artifact_sha256s"], list):
                raise AlgebraAgentIntegrityError("join verifier vector must be an array")
            joins.append(
                (
                    item["lane_id"],
                    item["program_receipt_sha256"],
                    item["execution_receipt_sha256"],
                    item["final_state_sha256"],
                    tuple(item["verifier_artifact_sha256s"]),
                )
            )
        try:
            result = cls(
                ensemble_sha256=body["ensemble_sha256"],
                lane_joins=tuple(joins),
                total_work_units=body["total_work_units"],
            )
        except (TypeError, ValueError) as exc:
            raise AlgebraAgentIntegrityError("invalid ensemble receipt") from exc
        if result._body() != body:
            raise AlgebraAgentIntegrityError("ensemble receipt changed on roundtrip")
        return result


@dataclass(frozen=True, slots=True)
class HeterogeneousEnsembleResult:
    ensemble: HeterogeneousProgramEnsemble
    lane_results: tuple[ParallelLaneResult, ...]
    receipt: HeterogeneousEnsembleReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.ensemble, HeterogeneousProgramEnsemble):
            raise TypeError("ensemble must be a HeterogeneousProgramEnsemble")
        results = tuple(self.lane_results)
        if not all(isinstance(item, ParallelLaneResult) for item in results):
            raise TypeError("lane_results must be ParallelLaneResult values")
        if tuple(item.lane_id for item in results) != tuple(
            lane.lane_id for lane in self.ensemble.lanes
        ):
            raise AlgebraAgentIntegrityError("ensemble results are missing, extra, or swapped")
        object.__setattr__(self, "lane_results", results)
        if not isinstance(self.receipt, HeterogeneousEnsembleReceipt):
            raise TypeError("receipt must be a HeterogeneousEnsembleReceipt")
        for lane, result in zip(self.ensemble.lanes, results, strict=True):
            artifacts = result.verifier_artifacts
            if tuple(sorted(item.verifier_sha256 for item in artifacts)) != lane.verifier_sha256s:
                raise AlgebraAgentIntegrityError("lane verifier identities changed")
            expected_artifact_hashes = tuple(item.sha256 for item in artifacts)
            if result.execution_receipt.verifier_sha256s != expected_artifact_hashes:
                raise AlgebraAgentIntegrityError("execution omitted or swapped exact verifier receipts")
            replay = AffineMonoidRuntime.execute(
                lane.program,
                initial_state=lane.initial_state,
                verifier_sha256s=expected_artifact_hashes,
            )
            if (
                result.program_receipt != replay.program_receipt
                or result.execution_receipt != replay.execution_receipt
                or result.final_state != replay.state
            ):
                raise AlgebraAgentIntegrityError("parallel lane failed exact replay")
        expected_receipt = HeterogeneousEnsembleReceipt.issue(self.ensemble, results)
        if self.receipt != expected_receipt:
            raise AlgebraAgentIntegrityError("ensemble join receipt mismatch")

    @property
    def outputs(self) -> dict[str, AffineState]:
        return {item.lane_id: item.final_state for item in self.lane_results}

    def _body(self) -> dict[str, object]:
        return {
            "ensemble_base64": _b64(self.ensemble.to_bytes()),
            "ensemble_sha256": self.ensemble.sha256,
            "lane_results": [item.to_record() for item in self.lane_results],
            "receipt_base64": _b64(self.receipt.to_bytes()),
            "receipt_sha256": self.receipt.sha256,
        }

    def to_bytes(self) -> bytes:
        return _seal(ENSEMBLE_RESULT_SCHEMA, self._body())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "HeterogeneousEnsembleResult":
        body = _open(data, schema=ENSEMBLE_RESULT_SCHEMA)
        _exact_keys(
            body,
            {
                "ensemble_base64",
                "ensemble_sha256",
                "lane_results",
                "receipt_base64",
                "receipt_sha256",
            },
            field="ensemble result",
        )
        if not isinstance(body["lane_results"], list):
            raise AlgebraAgentIntegrityError("lane results must be an array")
        try:
            ensemble = HeterogeneousProgramEnsemble.from_bytes(
                _unb64(body["ensemble_base64"], field="ensemble_base64")
            )
            if len(body["lane_results"]) != len(ensemble.lanes):
                raise AlgebraAgentIntegrityError("lane result count mismatch")
            results = tuple(
                ParallelLaneResult.from_record(item, schema_program=lane.program)
                for item, lane in zip(
                    body["lane_results"], ensemble.lanes, strict=True
                )
            )
            result = cls(
                ensemble=ensemble,
                lane_results=results,
                receipt=HeterogeneousEnsembleReceipt.from_bytes(
                    _unb64(body["receipt_base64"], field="receipt_base64")
                ),
            )
        except (TypeError, ValueError, AffineIntegrityError) as exc:
            if isinstance(exc, AlgebraAgentIntegrityError):
                raise
            raise AlgebraAgentIntegrityError("invalid ensemble result") from exc
        if result._body() != body:
            raise AlgebraAgentIntegrityError("ensemble result derived bindings changed")
        return result


__all__ = [
    "AlgebraAdmissionReceipt",
    "AlgebraAgentError",
    "AlgebraAgentIntegrityError",
    "AlgebraRouterBank",
    "AlgebraRouterState",
    "AlgebraSelectionReceipt",
    "AlgebraUpdateReceipt",
    "ExactVerifierArtifact",
    "HeterogeneousABIError",
    "HeterogeneousEnsembleReceipt",
    "HeterogeneousEnsembleResult",
    "HeterogeneousProgramEnsemble",
    "OperatorAlgebraCandidate",
    "ParallelLaneResult",
    "ParallelProgramLane",
    "VerifierBoundOutcome",
]
