"""Promote contextual O1 discoveries into executable algebra-router arms.

The contextual harvester discovers numerical programs.  The algebra router
chooses programs.  This module is the authenticated hand-off between those two
systems and the ComputeCrystal VM that actually executes a selected arm.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from .algebra_agents import (
    AlgebraAdmissionReceipt,
    AlgebraRouterState,
    AlgebraSelectionReceipt,
    AlgebraUpdateReceipt,
    OperatorAlgebraCandidate,
    VerifierBoundOutcome,
)
from .compute_crystals import (
    AFFINE_FLOAT64,
    CAUSAL_MIX_FLOAT64,
    MARKOV_FLOAT64,
    PERMUTATION,
    ComputeBankPublication,
    ComputeCrystal,
    ComputeCrystalBank,
    ComputeCrystalVM,
    ComputeExecution,
    ComputeExecutionReceipt,
    ComputeProgram,
)
from .compute_graph import ComputeOperatorGraph, OperatorEdge
from .crystal import CrystalStore, StatePublication
from .identity import canonical_json_bytes, require_sha256
from .operator_harvester import HarvestPromotion


HARVEST_PROFILE_SCHEMA = "immer-ooe-harvested-algebra-profile/v1"
HARVEST_BRIDGE_SCHEMA = "immer-ooe-harvested-algebra-bridge/v1"
COMPUTE_ALGEBRA_EXECUTION_SCHEMA = "immer-ooe-compute-algebra-execution/v1"
MAX_BRIDGE_BYTES = 256 * 1024 * 1024
MAX_VERIFIER_BYTES = 64 * 1024 * 1024
MAX_PROFILE_AXES = 32
MAX_PROFILE_OBJECTIVES = 32
MAX_PROFILE_EVIDENCE = 4096
HARVEST_CATALOG_SCHEMA = "immer-ooe-harvested-algebra-catalog-record/v1"
_PROFILE_STATE_PREFIX = "ooe-harvest-algebra-profile/v1:"
_BRIDGE_STATE_PREFIX = "ooe-harvest-algebra-bridge/v1:"
_CATALOG_STATE_PREFIX = "ooe-harvest-algebra-candidate/v1:"

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}\Z")
_KIND_INDEX = {
    AFFINE_FLOAT64: 0,
    PERMUTATION: 1,
    MARKOV_FLOAT64: 2,
    CAUSAL_MIX_FLOAT64: 3,
}


class HarvestAlgebraBridgeError(ValueError):
    """A discovered operator cannot cross into routed execution."""


class HarvestAlgebraBridgeIntegrityError(HarvestAlgebraBridgeError):
    """A graph, bank, evidence, selection, or verifier binding changed."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise HarvestAlgebraBridgeError(f"{field} is not a canonical identifier")
    return value


def _uint(value: object, *, field: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HarvestAlgebraBridgeError(f"{field} must be an integer")
    if value < (1 if positive else 0):
        raise HarvestAlgebraBridgeError(f"{field} lies outside its range")
    return value


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HarvestAlgebraBridgeError(f"{field} must be finite numeric data")
    result = float(value)
    if not math.isfinite(result):
        raise HarvestAlgebraBridgeError(f"{field} must be finite numeric data")
    return 0.0 if result == 0.0 else result


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(value: object, *, field: str, maximum: int) -> bytes:
    if not isinstance(value, str) or not value.isascii():
        raise HarvestAlgebraBridgeIntegrityError(f"{field} must be canonical base64")
    if len(value) > 4 * ((maximum + 2) // 3):
        raise HarvestAlgebraBridgeIntegrityError(f"{field} exceeds its byte bound")
    try:
        data = base64.b64decode(value, validate=True)
    except (TypeError, ValueError) as exc:
        raise HarvestAlgebraBridgeIntegrityError(f"{field} is invalid base64") from exc
    if len(data) > maximum or _b64(data) != value:
        raise HarvestAlgebraBridgeIntegrityError(f"{field} is not canonical base64")
    return data


def _seal(schema: str, body: Mapping[str, object]) -> bytes:
    exact = dict(body)
    return canonical_json_bytes(
        {"body": exact, "body_sha256": _digest(exact), "schema": schema}
    )


def _open(data: bytes, *, schema: str) -> dict[str, Any]:
    if not isinstance(data, bytes) or not 1 <= len(data) <= MAX_BRIDGE_BYTES:
        raise HarvestAlgebraBridgeIntegrityError("bridge payload exceeds its bound")
    try:
        document = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HarvestAlgebraBridgeIntegrityError("bridge payload is not JSON") from exc
    if (
        not isinstance(document, dict)
        or set(document) != {"body", "body_sha256", "schema"}
        or document.get("schema") != schema
        or not isinstance(document.get("body"), dict)
        or canonical_json_bytes(document) != data
        or document.get("body_sha256") != _digest(document["body"])
    ):
        raise HarvestAlgebraBridgeIntegrityError("bridge payload seal is invalid")
    return document["body"]


def _keys(value: Mapping[str, object], expected: set[str], *, field: str) -> None:
    if set(value) != expected:
        raise HarvestAlgebraBridgeIntegrityError(f"{field} fields are malformed")


@dataclass(frozen=True, slots=True)
class HarvestedCandidateProfile:
    """Behavioral placement and execution identity for one harvested program."""

    family: str
    behavior_descriptor: tuple[int, ...]
    objectives: tuple[float, ...]
    policy_sha256: str
    execution_verifier_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", _identifier(self.family, field="family"))
        descriptor = tuple(
            _uint(value, field="behavior descriptor")
            for value in self.behavior_descriptor
        )
        if not 1 <= len(descriptor) <= MAX_PROFILE_AXES:
            raise HarvestAlgebraBridgeError("behavior descriptor dimension is invalid")
        objectives = tuple(
            _finite(value, field="objective") for value in self.objectives
        )
        if not 1 <= len(objectives) <= MAX_PROFILE_OBJECTIVES:
            raise HarvestAlgebraBridgeError("objective dimension is invalid")
        object.__setattr__(self, "behavior_descriptor", descriptor)
        object.__setattr__(self, "objectives", objectives)
        for field in ("policy_sha256", "execution_verifier_sha256"):
            object.__setattr__(
                self,
                field,
                require_sha256(getattr(self, field), field=field),
            )

    def _body(self) -> dict[str, object]:
        return {
            "behavior_descriptor": list(self.behavior_descriptor),
            "execution_verifier_sha256": self.execution_verifier_sha256,
            "family": self.family,
            "objectives_hex": [value.hex() for value in self.objectives],
            "policy_sha256": self.policy_sha256,
        }

    def to_bytes(self) -> bytes:
        return _seal(HARVEST_PROFILE_SCHEMA, self._body())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "HarvestedCandidateProfile":
        body = _open(data, schema=HARVEST_PROFILE_SCHEMA)
        _keys(
            body,
            {
                "behavior_descriptor",
                "execution_verifier_sha256",
                "family",
                "objectives_hex",
                "policy_sha256",
            },
            field="harvest profile",
        )
        if not isinstance(body["behavior_descriptor"], list) or not isinstance(
            body["objectives_hex"], list
        ):
            raise HarvestAlgebraBridgeIntegrityError("profile vectors are malformed")
        try:
            objectives = tuple(float.fromhex(value) for value in body["objectives_hex"])
            result = cls(
                family=body["family"],
                behavior_descriptor=tuple(body["behavior_descriptor"]),
                objectives=objectives,
                policy_sha256=body["policy_sha256"],
                execution_verifier_sha256=body["execution_verifier_sha256"],
            )
        except (TypeError, ValueError) as exc:
            raise HarvestAlgebraBridgeIntegrityError("invalid harvest profile") from exc
        if result._body() != body:
            raise HarvestAlgebraBridgeIntegrityError(
                "profile changed on reconstruction"
            )
        return result


def profile_harvested_candidate(
    promotion: HarvestPromotion,
    crystal: ComputeCrystal,
    *,
    bin_counts: Sequence[int],
    objective_count: int,
    execution_verifier_sha256: str,
) -> HarvestedCandidateProfile:
    """Map verified operator structure into one deterministic MAP-Elites cell."""

    if not isinstance(promotion, HarvestPromotion):
        raise TypeError("promotion must be a HarvestPromotion")
    if not isinstance(crystal, ComputeCrystal):
        raise TypeError("crystal must be a ComputeCrystal")
    bins = tuple(_uint(value, field="bin count", positive=True) for value in bin_counts)
    objectives_n = _uint(objective_count, field="objective_count", positive=True)
    if not 1 <= len(bins) <= MAX_PROFILE_AXES:
        raise HarvestAlgebraBridgeError("profile bin dimension is invalid")
    if objectives_n > MAX_PROFILE_OBJECTIVES:
        raise HarvestAlgebraBridgeError("profile objective dimension is invalid")
    kind_index = _KIND_INDEX.get(crystal.operator_kind)
    if kind_index is None:
        raise HarvestAlgebraBridgeError("harvested operator kind is not routable")
    dimensions = (
        *crystal.input_abi.trailing_shape,
        *crystal.output_abi.trailing_shape,
    )
    width = max(dimensions, default=1)
    granularity = (
        crystal.extensions.get("operator_harvester", {}).get("granularity", "operator")
        if isinstance(crystal.extensions.get("operator_harvester"), Mapping)
        else "operator"
    )
    raw_descriptor = (
        kind_index,
        max(0, width.bit_length() - 1),
        1 if granularity == "segment" else 0,
    )
    descriptor = tuple(
        min(raw_descriptor[index] if index < len(raw_descriptor) else 0, count - 1)
        for index, count in enumerate(bins)
    )
    evidence_count = len(promotion.candidate.observation_receipt_sha256s)
    raw_objectives = (
        float(evidence_count),
        1.0 / float(crystal.discharge_work_units),
        float(promotion.edge.weight),
        1.0,
    )
    objectives = tuple(
        raw_objectives[index] if index < len(raw_objectives) else 0.0
        for index in range(objectives_n)
    )
    policy_sha = _digest(
        {
            "bin_counts": list(bins),
            "descriptor_axes": ["operator-kind", "log2-width", "granularity"],
            "objective_count": objectives_n,
            "objective_axes": [
                "verified-contexts",
                "inverse-live-work",
                "edge-weight",
                "heldout-verified",
            ],
            "schema": HARVEST_PROFILE_SCHEMA,
        }
    )
    return HarvestedCandidateProfile(
        family=f"harvest/{crystal.operator_kind}/{promotion.candidate.group_sha256}",
        behavior_descriptor=descriptor,
        objectives=objectives,
        policy_sha256=policy_sha,
        execution_verifier_sha256=execution_verifier_sha256,
    )


def execution_verifier_sha256_for_harvest(
    promotion: HarvestPromotion,
    crystal: ComputeCrystal,
) -> str:
    """Name the exact runtime checker contract of one harvested family."""

    if not isinstance(promotion, HarvestPromotion):
        raise TypeError("promotion must be a HarvestPromotion")
    if not isinstance(crystal, ComputeCrystal):
        raise TypeError("crystal must be a ComputeCrystal")
    return _digest(
        {
            "checker": "compute-vm-receipt+consumer-result-verifier/v1",
            "discovery_group_sha256": promotion.candidate.group_sha256,
            "input_abi": crystal.input_abi.to_record(),
            "operator_kind": crystal.operator_kind,
            "output_abi": crystal.output_abi.to_record(),
            "schema": COMPUTE_ALGEBRA_EXECUTION_SCHEMA,
            "source_state": promotion.edge.source_state,
            "target_state": promotion.edge.target_state,
        }
    )


@dataclass(frozen=True, slots=True)
class HarvestAlgebraBridgeReceipt:
    graph_state_sha256: str
    edge_sha256: str
    crystal_sha256: str
    program_sha256: str
    program_publication_manifest_sha256: str
    discovery_verifier_sha256: str
    execution_verifier_sha256: str
    evidence_sha256s: tuple[str, ...]
    profile_sha256: str
    candidate_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "graph_state_sha256",
            "edge_sha256",
            "crystal_sha256",
            "program_sha256",
            "program_publication_manifest_sha256",
            "discovery_verifier_sha256",
            "execution_verifier_sha256",
            "profile_sha256",
            "candidate_sha256",
        ):
            object.__setattr__(
                self,
                field,
                require_sha256(getattr(self, field), field=field),
            )
        evidence = tuple(
            require_sha256(value, field="evidence_sha256")
            for value in self.evidence_sha256s
        )
        if not 1 <= len(evidence) <= MAX_PROFILE_EVIDENCE or evidence != tuple(
            sorted(set(evidence))
        ):
            raise HarvestAlgebraBridgeError("bridge evidence must be sorted and unique")
        object.__setattr__(self, "evidence_sha256s", evidence)

    def _body(self) -> dict[str, object]:
        return {
            "candidate_sha256": self.candidate_sha256,
            "crystal_sha256": self.crystal_sha256,
            "discovery_verifier_sha256": self.discovery_verifier_sha256,
            "edge_sha256": self.edge_sha256,
            "evidence_sha256s": list(self.evidence_sha256s),
            "execution_verifier_sha256": self.execution_verifier_sha256,
            "graph_state_sha256": self.graph_state_sha256,
            "profile_sha256": self.profile_sha256,
            "program_publication_manifest_sha256": (
                self.program_publication_manifest_sha256
            ),
            "program_sha256": self.program_sha256,
        }

    def to_bytes(self) -> bytes:
        return _seal(HARVEST_BRIDGE_SCHEMA, self._body())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "HarvestAlgebraBridgeReceipt":
        body = _open(data, schema=HARVEST_BRIDGE_SCHEMA)
        _keys(body, set(cls.__dataclass_fields__), field="harvest bridge")
        if not isinstance(body["evidence_sha256s"], list):
            raise HarvestAlgebraBridgeIntegrityError("bridge evidence is malformed")
        try:
            result = cls(
                **{**body, "evidence_sha256s": tuple(body["evidence_sha256s"])}
            )
        except (TypeError, ValueError) as exc:
            raise HarvestAlgebraBridgeIntegrityError("invalid harvest bridge") from exc
        if result._body() != body:
            raise HarvestAlgebraBridgeIntegrityError("bridge changed on reconstruction")
        return result


@dataclass(frozen=True, slots=True)
class HarvestAlgebraCatalogRecord:
    """Immutable admission-time link from a routed candidate to its evidence."""

    candidate_sha256: str
    profile_sha256: str
    bridge_receipt_sha256: str
    admission_router_sha256: str
    admission_receipt_sha256: str | None
    bootstrap: bool

    def __post_init__(self) -> None:
        for field in (
            "candidate_sha256",
            "profile_sha256",
            "bridge_receipt_sha256",
            "admission_router_sha256",
        ):
            object.__setattr__(
                self,
                field,
                require_sha256(getattr(self, field), field=field),
            )
        admission = self.admission_receipt_sha256
        if admission is not None:
            admission = require_sha256(admission, field="admission_receipt_sha256")
        if not isinstance(self.bootstrap, bool):
            raise TypeError("bootstrap must be a boolean")
        if self.bootstrap != (admission is None):
            raise HarvestAlgebraBridgeError(
                "bootstrap and admission receipt presence disagree"
            )
        object.__setattr__(self, "admission_receipt_sha256", admission)

    def _body(self) -> dict[str, object]:
        return {
            "admission_receipt_sha256": self.admission_receipt_sha256,
            "admission_router_sha256": self.admission_router_sha256,
            "bootstrap": self.bootstrap,
            "bridge_receipt_sha256": self.bridge_receipt_sha256,
            "candidate_sha256": self.candidate_sha256,
            "profile_sha256": self.profile_sha256,
        }

    def to_bytes(self) -> bytes:
        return _seal(HARVEST_CATALOG_SCHEMA, self._body())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "HarvestAlgebraCatalogRecord":
        body = _open(data, schema=HARVEST_CATALOG_SCHEMA)
        _keys(body, set(cls.__dataclass_fields__), field="harvest catalog record")
        try:
            record = cls(**body)
        except (TypeError, ValueError) as exc:
            raise HarvestAlgebraBridgeIntegrityError(
                "invalid harvest catalog record"
            ) from exc
        if record._body() != body:
            raise HarvestAlgebraBridgeIntegrityError(
                "harvest catalog record changed on reconstruction"
            )
        return record


@dataclass(frozen=True, slots=True)
class HarvestAlgebraArtifactPublication:
    profile: StatePublication
    bridge: StatePublication
    catalog: StatePublication
    record: HarvestAlgebraCatalogRecord

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, StatePublication)
            for value in (self.profile, self.bridge, self.catalog)
        ):
            raise TypeError("harvest artifact publications are invalid")
        if not isinstance(self.record, HarvestAlgebraCatalogRecord):
            raise TypeError("record must be a HarvestAlgebraCatalogRecord")
        if (
            self.profile.payload_sha256 != self.record.profile_sha256
            or self.bridge.payload_sha256 != self.record.bridge_receipt_sha256
            or self.catalog.payload_sha256 != self.record.sha256
        ):
            raise HarvestAlgebraBridgeIntegrityError(
                "harvest artifact publications disagree with their catalog record"
            )


class HarvestAlgebraArtifactBank:
    """Content-address profiles/bridges and index them by routed candidate."""

    def __init__(self, store: CrystalStore | str | Path) -> None:
        self.store = store if isinstance(store, CrystalStore) else CrystalStore(store)

    @staticmethod
    def profile_state_name(profile_sha256: str) -> str:
        return _PROFILE_STATE_PREFIX + require_sha256(
            profile_sha256, field="profile_sha256"
        )

    @staticmethod
    def bridge_state_name(bridge_sha256: str) -> str:
        return _BRIDGE_STATE_PREFIX + require_sha256(
            bridge_sha256, field="bridge_sha256"
        )

    @staticmethod
    def catalog_state_name(candidate_sha256: str) -> str:
        return _CATALOG_STATE_PREFIX + require_sha256(
            candidate_sha256, field="candidate_sha256"
        )

    def restore_profile(self, profile_sha256: str) -> HarvestedCandidateProfile:
        digest = require_sha256(profile_sha256, field="profile_sha256")
        profile = HarvestedCandidateProfile.from_bytes(
            self.store.restore_state(self.profile_state_name(digest))
        )
        if profile.sha256 != digest:
            raise HarvestAlgebraBridgeIntegrityError(
                "restored harvest profile changed address"
            )
        return profile

    def restore_bridge(self, bridge_sha256: str) -> HarvestAlgebraBridgeReceipt:
        digest = require_sha256(bridge_sha256, field="bridge_sha256")
        bridge = HarvestAlgebraBridgeReceipt.from_bytes(
            self.store.restore_state(self.bridge_state_name(digest))
        )
        if bridge.sha256 != digest:
            raise HarvestAlgebraBridgeIntegrityError(
                "restored harvest bridge changed address"
            )
        return bridge

    def restore_record(self, candidate_sha256: str) -> HarvestAlgebraCatalogRecord:
        candidate = require_sha256(candidate_sha256, field="candidate_sha256")
        record = HarvestAlgebraCatalogRecord.from_bytes(
            self.store.restore_state(self.catalog_state_name(candidate))
        )
        if record.candidate_sha256 != candidate:
            raise HarvestAlgebraBridgeIntegrityError(
                "harvest catalog candidate index changed"
            )
        return record

    def publish(
        self,
        candidate: OperatorAlgebraCandidate,
        profile: HarvestedCandidateProfile,
        bridge: HarvestAlgebraBridgeReceipt,
        *,
        admission_router_sha256: str,
        admission_receipt_sha256: str | None,
        bootstrap: bool,
    ) -> HarvestAlgebraArtifactPublication:
        if not isinstance(candidate, OperatorAlgebraCandidate):
            raise TypeError("candidate must be an OperatorAlgebraCandidate")
        if not isinstance(profile, HarvestedCandidateProfile):
            raise TypeError("profile must be a HarvestedCandidateProfile")
        if not isinstance(bridge, HarvestAlgebraBridgeReceipt):
            raise TypeError("bridge must be a HarvestAlgebraBridgeReceipt")
        if (
            bridge.candidate_sha256 != candidate.sha256
            or bridge.profile_sha256 != profile.sha256
        ):
            raise HarvestAlgebraBridgeIntegrityError(
                "candidate, profile, and bridge differ before publication"
            )
        proposed = HarvestAlgebraCatalogRecord(
            candidate_sha256=candidate.sha256,
            profile_sha256=profile.sha256,
            bridge_receipt_sha256=bridge.sha256,
            admission_router_sha256=admission_router_sha256,
            admission_receipt_sha256=admission_receipt_sha256,
            bootstrap=bootstrap,
        )
        try:
            existing = self.restore_record(candidate.sha256)
        except KeyError:
            record = proposed
            stored_profile = profile
            stored_bridge = bridge
        else:
            stored_profile = self.restore_profile(existing.profile_sha256)
            stored_bridge = self.restore_bridge(existing.bridge_receipt_sha256)
            if existing.candidate_sha256 != proposed.candidate_sha256 or (
                stored_profile.to_bytes() != profile.to_bytes()
                or stored_bridge.candidate_sha256 != bridge.candidate_sha256
                or stored_bridge.profile_sha256 != bridge.profile_sha256
                or stored_bridge.graph_state_sha256 != bridge.graph_state_sha256
                or stored_bridge.edge_sha256 != bridge.edge_sha256
                or stored_bridge.crystal_sha256 != bridge.crystal_sha256
                or stored_bridge.program_sha256 != bridge.program_sha256
                or stored_bridge.discovery_verifier_sha256
                != bridge.discovery_verifier_sha256
                or stored_bridge.execution_verifier_sha256
                != bridge.execution_verifier_sha256
                or stored_bridge.evidence_sha256s != bridge.evidence_sha256s
            ):
                raise HarvestAlgebraBridgeIntegrityError(
                    "candidate catalog was rebound to different harvest evidence"
                )
            record = existing
        profile_publication = self.store.publish_state(
            self.profile_state_name(stored_profile.sha256), stored_profile.to_bytes()
        )
        bridge_publication = self.store.publish_state(
            self.bridge_state_name(stored_bridge.sha256), stored_bridge.to_bytes()
        )
        catalog_publication = self.store.publish_state(
            self.catalog_state_name(candidate.sha256), record.to_bytes()
        )
        return HarvestAlgebraArtifactPublication(
            profile_publication,
            bridge_publication,
            catalog_publication,
            record,
        )

    def audit_candidate(
        self,
        candidate: OperatorAlgebraCandidate,
        router: AlgebraRouterState,
    ) -> HarvestAlgebraCatalogRecord:
        if not isinstance(candidate, OperatorAlgebraCandidate):
            raise TypeError("candidate must be an OperatorAlgebraCandidate")
        if not isinstance(router, AlgebraRouterState):
            raise TypeError("router must be an AlgebraRouterState")
        active = {value.sha256: value for value in router.candidates}
        if active.get(candidate.sha256) != candidate:
            raise HarvestAlgebraBridgeIntegrityError(
                "candidate is absent or changed in the active router"
            )
        record = self.restore_record(candidate.sha256)
        profile = self.restore_profile(record.profile_sha256)
        bridge = self.restore_bridge(record.bridge_receipt_sha256)
        if (
            bridge.candidate_sha256 != candidate.sha256
            or bridge.profile_sha256 != profile.sha256
            or bridge.program_sha256 != candidate.program_sha256
            or bridge.discovery_verifier_sha256 != candidate.discovery_verifier_sha256
            or bridge.execution_verifier_sha256 != candidate.verifier_sha256
            or bridge.evidence_sha256s != candidate.evidence_sha256s
            or profile.execution_verifier_sha256 != candidate.verifier_sha256
        ):
            raise HarvestAlgebraBridgeIntegrityError(
                "active routed candidate failed harvest artifact audit"
            )
        return record

    def audit_router(
        self, router: AlgebraRouterState
    ) -> tuple[HarvestAlgebraCatalogRecord, ...]:
        return tuple(
            self.audit_candidate(candidate, router) for candidate in router.candidates
        )


@dataclass(frozen=True, slots=True)
class HarvestAlgebraAdmission:
    candidate: OperatorAlgebraCandidate
    profile: HarvestedCandidateProfile
    program_publication: ComputeBankPublication
    bridge_receipt: HarvestAlgebraBridgeReceipt
    router: AlgebraRouterState
    admission_receipt: AlgebraAdmissionReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, OperatorAlgebraCandidate):
            raise TypeError("candidate must be an OperatorAlgebraCandidate")
        if not isinstance(self.profile, HarvestedCandidateProfile):
            raise TypeError("profile must be a HarvestedCandidateProfile")
        if not isinstance(self.program_publication, ComputeBankPublication):
            raise TypeError("program_publication must be a ComputeBankPublication")
        if not isinstance(self.bridge_receipt, HarvestAlgebraBridgeReceipt):
            raise TypeError("bridge_receipt must be a HarvestAlgebraBridgeReceipt")
        if not isinstance(self.router, AlgebraRouterState):
            raise TypeError("router must be an AlgebraRouterState")
        if not isinstance(self.admission_receipt, AlgebraAdmissionReceipt):
            raise TypeError("admission_receipt must be an AlgebraAdmissionReceipt")
        active = {item.sha256 for item in self.router.candidates}
        if (
            self.bridge_receipt.candidate_sha256 != self.candidate.sha256
            or self.bridge_receipt.profile_sha256 != self.profile.sha256
            or self.program_publication.artifact_kind != "program"
            or self.program_publication.payload_sha256 != self.candidate.program_sha256
            or self.program_publication.manifest_sha256
            != self.bridge_receipt.program_publication_manifest_sha256
            or self.admission_receipt.candidate_sha256 != self.candidate.sha256
            or self.admission_receipt.router_next_sha256 != self.router.sha256
            or self.admission_receipt.active_candidate_sha256s != tuple(sorted(active))
        ):
            raise HarvestAlgebraBridgeIntegrityError(
                "harvest admission objects are not one transaction"
            )


def _promotion_edge(
    promotion: HarvestPromotion,
    graph: ComputeOperatorGraph,
) -> tuple[OperatorEdge, ComputeCrystal]:
    state = graph.state()
    if state.sha256 != promotion.graph_state_sha256:
        raise HarvestAlgebraBridgeIntegrityError(
            "harvest promotion is stale for the operator graph head"
        )
    matches = [edge for edge in state.edges if edge.sha256 == promotion.edge.sha256]
    if len(matches) != 1 or matches[0].to_bytes() != promotion.edge.to_bytes():
        raise HarvestAlgebraBridgeIntegrityError(
            "harvest promotion edge is absent or changed"
        )
    edge = matches[0]
    candidate = promotion.candidate
    if (
        candidate.status not in {"promoted", "verified-existing"}
        or candidate.crystal_sha256 != edge.crystal_sha256
        or candidate.evidence_sha256 != edge.evidence_sha256
        or candidate.verifier_sha256 != edge.verifier_sha256
        or promotion.bank_publication.artifact_kind != "crystal"
        or promotion.bank_publication.payload_sha256 != edge.crystal_sha256
    ):
        raise HarvestAlgebraBridgeIntegrityError(
            "harvest candidate, edge, and bank publication disagree"
        )
    crystal = graph.bank.restore_crystal(edge.crystal_sha256)
    if crystal.sha256 != edge.crystal_sha256:
        raise HarvestAlgebraBridgeIntegrityError("restored harvested crystal changed")
    return edge, crystal


def harvest_promotion_to_algebra_candidate(
    promotion: HarvestPromotion,
    *,
    graph: ComputeOperatorGraph,
    profile: HarvestedCandidateProfile,
) -> tuple[
    OperatorAlgebraCandidate,
    ComputeBankPublication,
    HarvestAlgebraBridgeReceipt,
]:
    """Authenticate, publish, and wrap one harvested numerical program."""

    if not isinstance(promotion, HarvestPromotion):
        raise TypeError("promotion must be a HarvestPromotion")
    if not isinstance(graph, ComputeOperatorGraph):
        raise TypeError("graph must be a ComputeOperatorGraph")
    if not isinstance(profile, HarvestedCandidateProfile):
        raise TypeError("profile must be a HarvestedCandidateProfile")
    edge, crystal = _promotion_edge(promotion, graph)
    program = ComputeProgram.compose((crystal,))
    publication = graph.bank.publish_program(program)
    restored = graph.bank.restore_program(program.sha256)
    if restored.to_bytes() != program.to_bytes():
        raise HarvestAlgebraBridgeIntegrityError("published compute program changed")
    stream = promotion.candidate
    evidence_values = {
        edge.evidence_sha256,
        *stream.observation_receipt_sha256s,
        *stream.fit_receipt_sha256s,
    }
    if stream.holdout_receipt_sha256 is not None:
        evidence_values.add(stream.holdout_receipt_sha256)
    evidence = tuple(sorted(evidence_values))
    candidate = OperatorAlgebraCandidate(
        family=profile.family,
        program=program,
        verifier_sha256=profile.execution_verifier_sha256,
        discovery_verifier_sha256=stream.verifier_sha256,
        evidence_sha256s=evidence,
        behavior_descriptor=profile.behavior_descriptor,
        objectives=profile.objectives,
    )
    receipt = HarvestAlgebraBridgeReceipt(
        graph_state_sha256=promotion.graph_state_sha256,
        edge_sha256=edge.sha256,
        crystal_sha256=crystal.sha256,
        program_sha256=program.sha256,
        program_publication_manifest_sha256=publication.manifest_sha256,
        discovery_verifier_sha256=stream.verifier_sha256,
        execution_verifier_sha256=profile.execution_verifier_sha256,
        evidence_sha256s=evidence,
        profile_sha256=profile.sha256,
        candidate_sha256=candidate.sha256,
    )
    return candidate, publication, receipt


def admit_harvest_promotion(
    promotion: HarvestPromotion,
    *,
    graph: ComputeOperatorGraph,
    router: AlgebraRouterState,
    execution_verifier_sha256: str,
) -> HarvestAlgebraAdmission:
    """Create and MAP-Elites-admit one harvested ComputeProgram arm."""

    if not isinstance(router, AlgebraRouterState):
        raise TypeError("router must be an AlgebraRouterState")
    _edge, crystal = _promotion_edge(promotion, graph)
    profile = profile_harvested_candidate(
        promotion,
        crystal,
        bin_counts=router.bin_counts,
        objective_count=router.objective_count,
        execution_verifier_sha256=execution_verifier_sha256,
    )
    candidate, publication, bridge = harvest_promotion_to_algebra_candidate(
        promotion,
        graph=graph,
        profile=profile,
    )
    updated, admission = router.admit(candidate)
    return HarvestAlgebraAdmission(
        candidate=candidate,
        profile=profile,
        program_publication=publication,
        bridge_receipt=bridge,
        router=updated,
        admission_receipt=admission,
    )


class ComputeAlgebraVerifier(Protocol):
    verifier_sha256: str

    def verify(
        self,
        candidate: OperatorAlgebraCandidate,
        execution: ComputeExecution,
    ) -> tuple[bool, bytes]: ...


@dataclass(frozen=True, slots=True)
class ComputeAlgebraExecutionReceipt:
    selection_sha256: str
    candidate_sha256: str
    program_sha256: str
    compute_receipt: ComputeExecutionReceipt
    verifier_sha256: str
    verifier_payload: bytes
    accepted: bool

    def __post_init__(self) -> None:
        for field in (
            "selection_sha256",
            "candidate_sha256",
            "program_sha256",
            "verifier_sha256",
        ):
            object.__setattr__(
                self,
                field,
                require_sha256(getattr(self, field), field=field),
            )
        if not isinstance(self.compute_receipt, ComputeExecutionReceipt):
            raise TypeError("compute_receipt must be a ComputeExecutionReceipt")
        if self.compute_receipt.program_sha256 != self.program_sha256:
            raise HarvestAlgebraBridgeIntegrityError(
                "compute receipt belongs to another program"
            )
        if not isinstance(self.verifier_payload, bytes) or not (
            1 <= len(self.verifier_payload) <= MAX_VERIFIER_BYTES
        ):
            raise HarvestAlgebraBridgeError("verifier payload exceeds its byte bound")
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be a boolean")

    @property
    def verifier_payload_sha256(self) -> str:
        return hashlib.sha256(self.verifier_payload).hexdigest()

    def _body(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "candidate_sha256": self.candidate_sha256,
            "compute_receipt_base64": _b64(self.compute_receipt.to_bytes()),
            "compute_receipt_sha256": self.compute_receipt.sha256,
            "program_sha256": self.program_sha256,
            "selection_sha256": self.selection_sha256,
            "verifier_payload_base64": _b64(self.verifier_payload),
            "verifier_payload_sha256": self.verifier_payload_sha256,
            "verifier_sha256": self.verifier_sha256,
        }

    def to_bytes(self) -> bytes:
        return _seal(COMPUTE_ALGEBRA_EXECUTION_SCHEMA, self._body())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "ComputeAlgebraExecutionReceipt":
        body = _open(data, schema=COMPUTE_ALGEBRA_EXECUTION_SCHEMA)
        _keys(
            body,
            {
                "accepted",
                "candidate_sha256",
                "compute_receipt_base64",
                "compute_receipt_sha256",
                "program_sha256",
                "selection_sha256",
                "verifier_payload_base64",
                "verifier_payload_sha256",
                "verifier_sha256",
            },
            field="compute algebra execution",
        )
        try:
            result = cls(
                selection_sha256=body["selection_sha256"],
                candidate_sha256=body["candidate_sha256"],
                program_sha256=body["program_sha256"],
                compute_receipt=ComputeExecutionReceipt.from_bytes(
                    _unb64(
                        body["compute_receipt_base64"],
                        field="compute_receipt_base64",
                        maximum=MAX_VERIFIER_BYTES,
                    )
                ),
                verifier_sha256=body["verifier_sha256"],
                verifier_payload=_unb64(
                    body["verifier_payload_base64"],
                    field="verifier_payload_base64",
                    maximum=MAX_VERIFIER_BYTES,
                ),
                accepted=body["accepted"],
            )
        except (TypeError, ValueError) as exc:
            raise HarvestAlgebraBridgeIntegrityError(
                "invalid compute algebra execution"
            ) from exc
        if result._body() != body:
            raise HarvestAlgebraBridgeIntegrityError(
                "compute algebra execution changed on reconstruction"
            )
        return result


@dataclass(frozen=True, slots=True)
class ComputeAlgebraExecution:
    output: NDArray[Any]
    compute_execution: ComputeExecution
    receipt: ComputeAlgebraExecutionReceipt
    outcome: VerifierBoundOutcome

    def __post_init__(self) -> None:
        if type(self.output) is not np.ndarray:
            raise TypeError("output must be an exact numpy.ndarray")
        if not isinstance(self.compute_execution, ComputeExecution):
            raise TypeError("compute_execution must be a ComputeExecution")
        if not isinstance(self.receipt, ComputeAlgebraExecutionReceipt):
            raise TypeError("receipt must be a ComputeAlgebraExecutionReceipt")
        if not isinstance(self.outcome, VerifierBoundOutcome):
            raise TypeError("outcome must be a VerifierBoundOutcome")
        if (
            not np.array_equal(self.output, self.compute_execution.output)
            or self.output.dtype != self.compute_execution.output.dtype
            or self.compute_execution.receipt != self.receipt.compute_receipt
            or self.outcome.receipt_payload != self.receipt.to_bytes()
            or self.outcome.selection_sha256 != self.receipt.selection_sha256
            or self.outcome.candidate_sha256 != self.receipt.candidate_sha256
            or self.outcome.verifier_sha256 != self.receipt.verifier_sha256
            or self.outcome.success != self.receipt.accepted
        ):
            raise HarvestAlgebraBridgeIntegrityError(
                "compute result, verifier outcome, and joined receipt disagree"
            )


def execute_selected_compute_candidate(
    selection: AlgebraSelectionReceipt,
    candidate: OperatorAlgebraCandidate,
    *,
    bank: ComputeCrystalBank,
    value: object,
    verifier: ComputeAlgebraVerifier,
) -> ComputeAlgebraExecution:
    """Execute one selected ComputeProgram and bind external verification."""

    if not isinstance(selection, AlgebraSelectionReceipt):
        raise TypeError("selection must be an AlgebraSelectionReceipt")
    if not isinstance(candidate, OperatorAlgebraCandidate):
        raise TypeError("candidate must be an OperatorAlgebraCandidate")
    if not isinstance(candidate.program, ComputeProgram):
        raise HarvestAlgebraBridgeError(
            "selected candidate is not a ComputeCrystal VM program"
        )
    if not isinstance(bank, ComputeCrystalBank):
        raise TypeError("bank must be a ComputeCrystalBank")
    if (
        selection.candidate_sha256 != candidate.sha256
        or selection.program_sha256 != candidate.program_sha256
        or selection.schema_sha256 != candidate.schema_sha256
        or selection.verifier_sha256 != candidate.verifier_sha256
    ):
        raise HarvestAlgebraBridgeIntegrityError(
            "selection and compute candidate bindings differ"
        )
    verifier_sha = require_sha256(
        getattr(verifier, "verifier_sha256", None),
        field="verifier.verifier_sha256",
    )
    if verifier_sha != candidate.verifier_sha256:
        raise HarvestAlgebraBridgeIntegrityError(
            "execution verifier differs from the routed candidate"
        )
    execution = ComputeCrystalVM(bank).execute(candidate.program, value)
    judged = verifier.verify(candidate, execution)
    if (
        not isinstance(judged, tuple)
        or len(judged) != 2
        or not isinstance(judged[0], bool)
        or not isinstance(judged[1], bytes)
    ):
        raise HarvestAlgebraBridgeIntegrityError(
            "compute verifier must return (bool, immutable bytes)"
        )
    accepted, verifier_payload = judged
    receipt = ComputeAlgebraExecutionReceipt(
        selection_sha256=selection.sha256,
        candidate_sha256=candidate.sha256,
        program_sha256=candidate.program_sha256,
        compute_receipt=execution.receipt,
        verifier_sha256=verifier_sha,
        verifier_payload=verifier_payload,
        accepted=accepted,
    )
    outcome = VerifierBoundOutcome.issue(
        selection,
        candidate,
        receipt_payload=receipt.to_bytes(),
        success=accepted,
    )
    output = np.array(execution.output, copy=True)
    output.setflags(write=False)
    return ComputeAlgebraExecution(output, execution, receipt, outcome)


def execute_and_observe_compute_candidate(
    router_after_selection: AlgebraRouterState,
    selection: AlgebraSelectionReceipt,
    candidate: OperatorAlgebraCandidate,
    *,
    bank: ComputeCrystalBank,
    value: object,
    verifier: ComputeAlgebraVerifier,
) -> tuple[AlgebraRouterState, AlgebraUpdateReceipt, ComputeAlgebraExecution]:
    """Execute, verify, and feed one routed numerical result back to Thompson."""

    result = execute_selected_compute_candidate(
        selection,
        candidate,
        bank=bank,
        value=value,
        verifier=verifier,
    )
    updated, receipt = router_after_selection.observe(selection, result.outcome)
    return updated, receipt, result


__all__ = [
    "COMPUTE_ALGEBRA_EXECUTION_SCHEMA",
    "HARVEST_BRIDGE_SCHEMA",
    "HARVEST_PROFILE_SCHEMA",
    "ComputeAlgebraExecution",
    "ComputeAlgebraExecutionReceipt",
    "ComputeAlgebraVerifier",
    "HarvestAlgebraAdmission",
    "HarvestAlgebraArtifactBank",
    "HarvestAlgebraArtifactPublication",
    "HarvestAlgebraBridgeError",
    "HarvestAlgebraBridgeIntegrityError",
    "HarvestAlgebraBridgeReceipt",
    "HarvestAlgebraCatalogRecord",
    "HarvestedCandidateProfile",
    "admit_harvest_promotion",
    "execute_and_observe_compute_candidate",
    "execute_selected_compute_candidate",
    "execution_verifier_sha256_for_harvest",
    "harvest_promotion_to_algebra_candidate",
    "profile_harvested_candidate",
]
