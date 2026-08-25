"""Gold-free FERTIG verifier receipts for the semantic weight atlas.

This module turns one already measured, externally labelled atlas observation
into one deterministic verifier vote backed by three verification surfaces.
The question and certified numeric answer exist only for the duration of
verification.  The replica receipt contains the semantic class already present
on the measurement and hashes of the three surfaces; it never contains the
question or numeric answer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from immer.runtimes.qwen3_8.cartography_probe import (
    CARTOGRAPHY_EVIDENCE_SCHEMA,
    ProbeSpec,
)
from immer.runtimes.qwen3_8.semantic_atlas import (
    AtlasAppendReceipt,
    EvidencePolicy,
    MeasurementReceipt,
    ReplicaReceipt,
    SemanticWeightAtlas,
)

from .adapter import (
    CandidateVerificationStatus,
    CertifiedAnswer,
    FertigSolver,
    canonical_numeric_candidate,
)


FERTIG_LABEL_EVIDENCE_SCHEMA = "immer.fertig-atlas-label-evidence/v1"
FERTIG_REPLICA_EVIDENCE_SCHEMA = "immer.fertig-atlas-replica-evidence/v1"
FERTIG_REPLICA_MEASUREMENT_SCHEMA = "immer.fertig-atlas-replica-measurement/v1"
FERTIG_STRUCTURAL_AUDIT_SCHEMA = "immer.fertig-atlas-structural-audit/v1"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SURFACES = (
    "certified-proof",
    "candidate-verification",
    "certificate-invariants",
)
_REPLICA_WEIGHT = 1.0


class FertigAtlasVerificationError(ValueError):
    """FERTIG evidence is absent, unsupported, or not bound to the measurement."""


class FertigAtlasReplicaConflictError(FertigAtlasVerificationError):
    """The atlas already contains a conflicting vote by this verifier identity."""


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
        raise FertigAtlasVerificationError(
            "FERTIG evidence is not canonical JSON"
        ) from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _verifier_configuration() -> dict[str, Any]:
    return {
        "accepted_certificate_family": "fraction_rref/v1",
        "candidate_surface": "FertigSolver.verify_candidate",
        "certification_surface": "FertigSolver.certify",
        "label_evidence_schema": FERTIG_LABEL_EVIDENCE_SCHEMA,
        "structural_audit_schema": FERTIG_STRUCTURAL_AUDIT_SCHEMA,
        "surface_names": list(_SURFACES),
    }


def _verifier_provenance(
    *,
    code_revision: str,
    model_pin_sha256: str,
    probe_sha256: str,
) -> dict[str, str]:
    return {
        "code_revision": code_revision,
        "implementation": "immer.cognition.fertig.atlas_verifier/v2",
        "model_pin_sha256": model_pin_sha256,
        "probe_sha256": probe_sha256,
        "verifier_configuration_sha256": _digest(_verifier_configuration()),
    }


def _replica_evidence_sha256(
    *,
    measurement_sha256: str,
    cartography_evidence_sha256: str,
    probe_sha256: str,
    question_sha256: str,
    semantic_label_sha256: str,
    label_evidence_sha256: str,
    candidate_verification_sha256: str,
    structural_audit_sha256: str,
    verifier_provenance: Mapping[str, str],
) -> str:
    evidence_body = {
        "candidate_verification_sha256": candidate_verification_sha256,
        "cartography_evidence_sha256": cartography_evidence_sha256,
        "certified_proof_sha256": label_evidence_sha256,
        "measurement_sha256": measurement_sha256,
        "probe_sha256": probe_sha256,
        "question_sha256": question_sha256,
        "semantic_label_sha256": semantic_label_sha256,
        "status": "support",
        "structural_audit_sha256": structural_audit_sha256,
        "verifier_configuration_sha256": _digest(_verifier_configuration()),
        "verifier_provenance_sha256": _digest(verifier_provenance),
    }
    return _digest(
        {
            "body": evidence_body,
            "schema": FERTIG_REPLICA_EVIDENCE_SCHEMA,
        }
    )


def _replica_measurement_sha256(
    *,
    evidence_sha256: str,
    measurement_sha256: str,
    replica_id_sha256: str,
) -> str:
    return _digest(
        {
            "evidence_sha256": evidence_sha256,
            "measurement_sha256": measurement_sha256,
            "replica_id_sha256": replica_id_sha256,
            "schema": FERTIG_REPLICA_MEASUREMENT_SCHEMA,
            "verdict": "support",
        }
    )


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise FertigAtlasVerificationError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _authenticate_cartography_evidence(
    evidence_document: Mapping[str, Any],
    *,
    probe_spec: ProbeSpec,
    measurement: MeasurementReceipt,
) -> tuple[dict[str, Any], str]:
    if not isinstance(evidence_document, Mapping) or set(evidence_document) != {
        "body",
        "schema",
        "sha256",
    }:
        raise FertigAtlasVerificationError(
            "cartography evidence document shape is invalid"
        )
    if evidence_document.get("schema") != CARTOGRAPHY_EVIDENCE_SCHEMA:
        raise FertigAtlasVerificationError(
            "cartography evidence document schema is invalid"
        )
    body = evidence_document.get("body")
    if not isinstance(body, Mapping):
        raise FertigAtlasVerificationError(
            "cartography evidence document body is invalid"
        )
    claimed = _sha(evidence_document.get("sha256"), "cartography evidence sha256")
    if claimed != _digest(body):
        raise FertigAtlasVerificationError(
            "cartography evidence document SHA-256 mismatch"
        )
    if claimed != measurement.evidence_sha256:
        raise FertigAtlasVerificationError(
            "cartography evidence SHA-256 differs from the measurement"
        )
    normalized_body = json.loads(_canonical(dict(body)))
    expected_probe_spec = json.loads(_canonical(probe_spec.as_record()))
    if normalized_body.get("probe_spec") != expected_probe_spec:
        raise FertigAtlasVerificationError(
            "cartography evidence probe_spec differs from the original ProbeSpec"
        )
    return normalized_body, claimed


def _semantic_label(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value) > 512
    ):
        raise FertigAtlasVerificationError(
            "measurement semantic label is not canonical external text"
        )
    if canonical_numeric_candidate(value) is not None:
        raise FertigAtlasVerificationError(
            "semantic label must not be a raw numeric answer"
        )
    return value


def _validate_probe_binding(
    probe_spec: ProbeSpec,
    measurement: MeasurementReceipt,
    *,
    question_sha256: str,
) -> tuple[str, str]:
    if not isinstance(probe_spec, ProbeSpec):
        raise TypeError("probe_spec must be the original ProbeSpec")
    if probe_spec.question_sha256 != question_sha256:
        raise FertigAtlasVerificationError(
            "raw question SHA-256 differs from the sealed ProbeSpec"
        )
    if measurement.probe.question_sha256 != question_sha256:
        raise FertigAtlasVerificationError(
            "raw question SHA-256 differs from the measured probe"
        )
    label = _semantic_label(measurement.observed_semantic_label)
    if probe_spec.semantic_label != label:
        raise FertigAtlasVerificationError(
            "measurement semantic label differs from the sealed ProbeSpec"
        )
    expected_evidence = probe_spec.label_evidence_sha256
    if expected_evidence is None:
        raise FertigAtlasVerificationError(
            "ProbeSpec has no external FERTIG label evidence"
        )
    expected_evidence = _sha(expected_evidence, "probe_spec.label_evidence_sha256")
    if measurement.probe != probe_spec.probe_identity:
        raise FertigAtlasVerificationError(
            "measurement ProbeIdentity differs from the sealed ProbeSpec"
        )
    if (
        probe_spec.code_revision != measurement.model_pin.code_revision
        or probe_spec.code_revision != measurement.runtime.code_revision
    ):
        raise FertigAtlasVerificationError(
            "ProbeSpec code revision differs from measurement provenance"
        )
    if probe_spec.intervention_mode != measurement.intervention.mode:
        raise FertigAtlasVerificationError(
            "measurement intervention mode differs from the sealed ProbeSpec"
        )
    requested = probe_spec.coordinate
    observed = measurement.coordinate
    if (
        requested.layer != observed.layer
        or requested.module != observed.module
        or requested.tensor != observed.tensor
        or requested.head_index != observed.head_index
        or requested.row_start != observed.row_start
        or requested.row_end != observed.row_end
    ):
        raise FertigAtlasVerificationError(
            "measurement coordinate differs from the sealed ProbeSpec"
        )
    if requested.row_start is None:
        expected_offset = (
            observed.tensor_absolute_offset + requested.relative_byte_offset
        )
        expected_length = requested.byte_length
        if expected_length is None:
            expected_length = observed.tensor_length - requested.relative_byte_offset
        if (
            observed.range_absolute_offset != expected_offset
            or observed.range_length != expected_length
        ):
            raise FertigAtlasVerificationError(
                "measurement byte range differs from the sealed ProbeSpec"
            )
    return label, expected_evidence


def _certified_answer(value: object) -> CertifiedAnswer:
    if not isinstance(value, CertifiedAnswer):
        raise FertigAtlasVerificationError(
            "FERTIG did not return a canonical certified answer"
        )
    if not isinstance(value.answer, str) or not value.answer:
        raise FertigAtlasVerificationError("FERTIG certified answer is invalid")
    if not isinstance(value.evidence, Mapping):
        raise FertigAtlasVerificationError("FERTIG proof evidence is invalid")
    return value


def _label_evidence_document(
    certified: CertifiedAnswer,
    *,
    semantic_label: str,
) -> dict[str, Any]:
    """Return the transient label-bound proof object hashed by ``ProbeSpec``."""

    body = {
        "certified_proof": certified.to_dict(),
        "semantic_label": _semantic_label(semantic_label),
    }
    return {
        "body": json.loads(_canonical(body)),
        "schema": FERTIG_LABEL_EVIDENCE_SCHEMA,
    }


def _strict_int(value: object, label: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive " if positive else "non-negative "
        raise FertigAtlasVerificationError(
            f"structural certificate {label} must be a {qualifier}integer"
        )
    return value


def _structural_certificate_audit(
    certified: CertifiedAnswer,
    *,
    question: str,
    question_sha256: str,
) -> dict[str, Any]:
    """Recompute the supported fraction-RREF certificate invariants.

    Certificate families intentionally have different schemas.  The atlas
    verifier currently accepts the exact structural ``fraction_rref/v1`` path
    and rejects formula-only or mixed evidence instead of pretending that
    unrelated fields establish an independent rank/residual audit.
    """

    evidence = certified.evidence
    if set(evidence) != {
        "answer",
        "certificates",
        "kind",
        "source_sha256",
        "verified",
    }:
        raise FertigAtlasVerificationError(
            "unsupported FERTIG exact-solution evidence shape"
        )
    if (
        evidence.get("kind") != "fertig-exact-solution/v1"
        or evidence.get("verified") is not True
        or evidence.get("answer") != certified.answer
        or evidence.get("source_sha256") != question_sha256
    ):
        raise FertigAtlasVerificationError(
            "FERTIG exact solution is not bound to the raw question"
        )
    certificates = evidence.get("certificates")
    if not isinstance(certificates, list) or len(certificates) != 1:
        raise FertigAtlasVerificationError(
            "only one exact structural certificate is supported"
        )
    certificate = certificates[0]
    if not isinstance(certificate, Mapping) or set(certificate) != {
        "component_variables",
        "equation_count",
        "kind",
        "pivot_columns",
        "rank",
        "residuals",
        "variable_count",
        "verified",
        "zero_residuals",
    }:
        raise FertigAtlasVerificationError(
            "unsupported FERTIG structural certificate shape"
        )
    if certificate.get("kind") != "fraction_rref/v1":
        raise FertigAtlasVerificationError("unsupported FERTIG certificate family")
    rank = _strict_int(certificate.get("rank"), "rank", positive=True)
    variable_count = _strict_int(
        certificate.get("variable_count"), "variable_count", positive=True
    )
    equation_count = _strict_int(
        certificate.get("equation_count"), "equation_count", positive=True
    )
    components = certificate.get("component_variables")
    if (
        not isinstance(components, list)
        or len(components) != variable_count
        or any(
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or "\x00" in value
            for value in components
        )
        or len(set(components)) != variable_count
    ):
        raise FertigAtlasVerificationError(
            "structural component-variable binding is invalid"
        )
    pivots = certificate.get("pivot_columns")
    if (
        not isinstance(pivots, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in pivots
        )
        or tuple(pivots) != tuple(range(variable_count))
    ):
        raise FertigAtlasVerificationError(
            "structural pivot certificate is not full rank"
        )
    residuals = certificate.get("residuals")
    if not isinstance(residuals, list) or len(residuals) != equation_count:
        raise FertigAtlasVerificationError(
            "structural residual certificate is incomplete"
        )
    for expected_index, residual in enumerate(residuals):
        if not isinstance(residual, Mapping) or set(residual) != {
            "constraint_index",
            "span",
            "value",
        }:
            raise FertigAtlasVerificationError(
                "structural residual shape is unsupported"
            )
        constraint_index = _strict_int(
            residual.get("constraint_index"), "residual constraint_index"
        )
        if constraint_index != expected_index:
            raise FertigAtlasVerificationError(
                "structural residual indices are not canonical"
            )
        if residual.get("value") != "0":
            raise FertigAtlasVerificationError(
                "structural certificate has a non-zero residual"
            )
        span = residual.get("span")
        if span is not None:
            if not isinstance(span, Mapping) or set(span) != {"end", "start"}:
                raise FertigAtlasVerificationError(
                    "structural residual span is invalid"
                )
            start = _strict_int(span.get("start"), "span start")
            end = _strict_int(span.get("end"), "span end")
            if start > end or end > len(question):
                raise FertigAtlasVerificationError(
                    "structural residual span escapes the raw question"
                )
    if (
        certificate.get("verified") is not True
        or certificate.get("zero_residuals") is not True
        or rank != variable_count
        or equation_count < rank
    ):
        raise FertigAtlasVerificationError(
            "structural rank/residual invariants did not verify"
        )
    body = {
        "certificate_sha256": _digest(certificate),
        "component_binding_sha256": _digest(components),
        "equation_count": equation_count,
        "kind": "fraction_rref/v1",
        "pivot_columns_sha256": _digest(pivots),
        "question_sha256": question_sha256,
        "rank": rank,
        "residuals_sha256": _digest(residuals),
        "source_bound": True,
        "variable_count": variable_count,
        "verified": True,
        "zero_residuals": True,
    }
    return {
        "body": body,
        "schema": FERTIG_STRUCTURAL_AUDIT_SCHEMA,
        "sha256": _digest(body),
    }


def fertig_label_evidence_sha256(
    question: str,
    semantic_label: str,
    *,
    solver: FertigSolver | None = None,
) -> str:
    """Build the label-bound FERTIG proof digest sealed into ``ProbeSpec``.

    The complete proof is transient.  Only this digest belongs in the probe
    specification and subsequent atlas receipts.
    """

    if not isinstance(question, str) or not question.strip():
        raise FertigAtlasVerificationError("question must be non-empty text")
    active_solver = FertigSolver() if solver is None else solver
    if not isinstance(active_solver, FertigSolver):
        raise TypeError("solver must be a FertigSolver")
    certified = _certified_answer(active_solver.certify(question))
    question_sha256 = hashlib.sha256(question.encode("utf-8")).hexdigest()
    _structural_certificate_audit(
        certified,
        question=question,
        question_sha256=question_sha256,
    )
    return _digest(_label_evidence_document(certified, semantic_label=semantic_label))


@dataclass(frozen=True, slots=True)
class FertigAtlasVerification:
    """One support receipt backed by three authenticated verification surfaces."""

    measurement_sha256: str
    probe_sha256: str
    question_sha256: str
    cartography_evidence_sha256: str
    label_evidence_sha256: str
    candidate_verification_sha256: str
    structural_audit_sha256: str
    replica: ReplicaReceipt

    def __post_init__(self) -> None:
        for field in (
            "measurement_sha256",
            "probe_sha256",
            "question_sha256",
            "cartography_evidence_sha256",
            "label_evidence_sha256",
            "candidate_verification_sha256",
            "structural_audit_sha256",
        ):
            _sha(getattr(self, field), field)
        if not isinstance(self.replica, ReplicaReceipt):
            raise TypeError("replica must be a ReplicaReceipt")
        if (
            self.replica.measurement_sha256 != self.measurement_sha256
            or self.replica.probe_sha256 != self.probe_sha256
            or self.replica.verdict != "support"
            or self.replica.reputation_weight != _REPLICA_WEIGHT
        ):
            raise FertigAtlasVerificationError(
                "FERTIG replica is not a normalized bound support vote"
            )
        assert self.replica.semantic_label is not None
        label_sha256 = hashlib.sha256(
            self.replica.semantic_label.encode("utf-8")
        ).hexdigest()
        provenance = _verifier_provenance(
            code_revision=self.replica.runtime.code_revision,
            model_pin_sha256=self.replica.model_pin.sha256,
            probe_sha256=self.probe_sha256,
        )
        expected_replica_id = _digest(provenance)
        expected_evidence = _replica_evidence_sha256(
            measurement_sha256=self.measurement_sha256,
            cartography_evidence_sha256=self.cartography_evidence_sha256,
            probe_sha256=self.probe_sha256,
            question_sha256=self.question_sha256,
            semantic_label_sha256=label_sha256,
            label_evidence_sha256=self.label_evidence_sha256,
            candidate_verification_sha256=self.candidate_verification_sha256,
            structural_audit_sha256=self.structural_audit_sha256,
            verifier_provenance=provenance,
        )
        expected_replica_measurement = _replica_measurement_sha256(
            evidence_sha256=expected_evidence,
            measurement_sha256=self.measurement_sha256,
            replica_id_sha256=expected_replica_id,
        )
        if (
            self.replica.replica_id_sha256 != expected_replica_id
            or self.replica.evidence_sha256 != expected_evidence
            or self.replica.replica_measurement_sha256 != expected_replica_measurement
        ):
            raise FertigAtlasVerificationError(
                "FERTIG replica provenance/surface binding is invalid"
            )


@dataclass(frozen=True, slots=True)
class FertigAtlasPromotionResult:
    """One idempotent verifier append followed by the caller-policy promotion."""

    verification: FertigAtlasVerification
    replica_append: AtlasAppendReceipt
    promotion_append: AtlasAppendReceipt


def verify_fertig_measurement(
    question: str,
    probe_spec: ProbeSpec,
    measurement: MeasurementReceipt,
    *,
    evidence_document: Mapping[str, Any],
    solver: FertigSolver | None = None,
) -> FertigAtlasVerification:
    """Re-run FERTIG and create one three-surface support receipt, fail closed."""

    if not isinstance(question, str) or not question.strip():
        raise FertigAtlasVerificationError("question must be non-empty text")
    if not isinstance(measurement, MeasurementReceipt):
        raise TypeError("measurement must be a MeasurementReceipt")
    if measurement.observation_status != "eligible":
        raise FertigAtlasVerificationError(
            "measurement is not eligible for external-label verification"
        )
    question_sha256 = hashlib.sha256(question.encode("utf-8")).hexdigest()
    label, expected_evidence = _validate_probe_binding(
        probe_spec,
        measurement,
        question_sha256=question_sha256,
    )
    _evidence_body, cartography_evidence_sha256 = _authenticate_cartography_evidence(
        evidence_document,
        probe_spec=probe_spec,
        measurement=measurement,
    )

    active_solver = FertigSolver() if solver is None else solver
    if not isinstance(active_solver, FertigSolver):
        raise TypeError("solver must be a FertigSolver")
    certified = _certified_answer(active_solver.certify(question))
    audit = _structural_certificate_audit(
        certified,
        question=question,
        question_sha256=question_sha256,
    )
    label_evidence = _digest(_label_evidence_document(certified, semantic_label=label))
    if label_evidence != expected_evidence:
        raise FertigAtlasVerificationError(
            "canonical label-bound FERTIG proof differs from ProbeSpec evidence"
        )

    candidate = active_solver.verify_candidate(question, certified.answer)
    if (
        candidate.status is not CandidateVerificationStatus.VERIFIED
        or candidate.candidate != certified.answer
        or candidate.expected != certified.answer
        or candidate.evidence.get("question_sha256") != question_sha256
        or candidate.evidence.get("exact_solution") != certified.evidence
    ):
        raise FertigAtlasVerificationError(
            "FERTIG candidate-verification surface disagrees with certification"
        )
    candidate_sha256 = _digest(candidate.to_dict())
    label_sha256 = hashlib.sha256(label.encode("utf-8")).hexdigest()
    verifier_provenance = _verifier_provenance(
        code_revision=probe_spec.code_revision,
        model_pin_sha256=measurement.model_pin.sha256,
        probe_sha256=measurement.probe.sha256,
    )
    replica_id_sha256 = _digest(verifier_provenance)
    evidence_sha256 = _replica_evidence_sha256(
        measurement_sha256=measurement.sha256,
        cartography_evidence_sha256=cartography_evidence_sha256,
        probe_sha256=measurement.probe.sha256,
        question_sha256=question_sha256,
        semantic_label_sha256=label_sha256,
        label_evidence_sha256=label_evidence,
        candidate_verification_sha256=candidate_sha256,
        structural_audit_sha256=str(audit["sha256"]),
        verifier_provenance=verifier_provenance,
    )
    replica_measurement_sha256 = _replica_measurement_sha256(
        evidence_sha256=evidence_sha256,
        measurement_sha256=measurement.sha256,
        replica_id_sha256=replica_id_sha256,
    )
    replica = ReplicaReceipt(
        model_pin=measurement.model_pin,
        coordinate=measurement.coordinate,
        measurement_sha256=measurement.sha256,
        probe_sha256=measurement.probe.sha256,
        intervention_sha256=measurement.intervention.sha256,
        replica_id_sha256=replica_id_sha256,
        replica_measurement_sha256=replica_measurement_sha256,
        verdict="support",
        semantic_label=label,
        reputation_weight=_REPLICA_WEIGHT,
        evidence_sha256=evidence_sha256,
        runtime=measurement.runtime,
    )
    return FertigAtlasVerification(
        measurement_sha256=measurement.sha256,
        probe_sha256=measurement.probe.sha256,
        question_sha256=question_sha256,
        cartography_evidence_sha256=cartography_evidence_sha256,
        label_evidence_sha256=label_evidence,
        candidate_verification_sha256=candidate_sha256,
        structural_audit_sha256=str(audit["sha256"]),
        replica=replica,
    )


def append_and_promote_fertig_label(
    atlas: SemanticWeightAtlas,
    verification: FertigAtlasVerification,
    *,
    policy: EvidencePolicy,
) -> FertigAtlasPromotionResult:
    """Preflight identity conflicts, append one vote, then apply ``policy``."""

    if not isinstance(atlas, SemanticWeightAtlas):
        raise TypeError("atlas must be a SemanticWeightAtlas")
    if not isinstance(verification, FertigAtlasVerification):
        raise TypeError("verification must be a FertigAtlasVerification")
    if not isinstance(policy, EvidencePolicy):
        raise TypeError("policy must be an EvidencePolicy")
    replica = verification.replica
    existing = tuple(
        row
        for row in atlas.query_by_coordinate(replica.coordinate).replicas
        if row.measurement_sha256 == replica.measurement_sha256
        and row.replica_id_sha256 == replica.replica_id_sha256
    )
    if any(row.sha256 != replica.sha256 for row in existing):
        raise FertigAtlasReplicaConflictError(
            "verifier identity already submitted conflicting evidence"
        )
    replica_append = atlas.append_replica(replica)
    promotion = atlas.promote_label(
        verification.measurement_sha256,
        policy=policy,
    )
    return FertigAtlasPromotionResult(
        verification=verification,
        replica_append=replica_append,
        promotion_append=promotion,
    )


def verify_append_and_promote_fertig_label(
    atlas: SemanticWeightAtlas,
    question: str,
    probe_spec: ProbeSpec,
    measurement: MeasurementReceipt,
    *,
    evidence_document: Mapping[str, Any],
    policy: EvidencePolicy,
    solver: FertigSolver | None = None,
) -> FertigAtlasPromotionResult:
    """Verify, append one three-surface receipt, and promote in one call."""

    verification = verify_fertig_measurement(
        question,
        probe_spec,
        measurement,
        evidence_document=evidence_document,
        solver=solver,
    )
    return append_and_promote_fertig_label(
        atlas,
        verification,
        policy=policy,
    )


__all__ = [
    "FERTIG_LABEL_EVIDENCE_SCHEMA",
    "FERTIG_REPLICA_EVIDENCE_SCHEMA",
    "FERTIG_REPLICA_MEASUREMENT_SCHEMA",
    "FERTIG_STRUCTURAL_AUDIT_SCHEMA",
    "FertigAtlasPromotionResult",
    "FertigAtlasReplicaConflictError",
    "FertigAtlasVerification",
    "FertigAtlasVerificationError",
    "append_and_promote_fertig_label",
    "fertig_label_evidence_sha256",
    "verify_append_and_promote_fertig_label",
    "verify_fertig_measurement",
]
