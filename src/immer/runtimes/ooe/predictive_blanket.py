"""Exact, feature-bound sparse execution for the calibrated OoE router.

An :class:`ExactRouterBlanketReceipt` is not a learned approximation.  Its
builder first evaluates the complete centroid universe for one exact
``QwenOoeFeatureReceipt`` and records the global nearest and runner-up labels.
Only after a second decision over that closed candidate set is bit-for-bit
equal to the full decision can the receipt be published.  Reusing it therefore
removes irrelevant centroid distances without generalising to another input.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from immer.runtimes.qwen3_8.semantic_atlas import GraphRevision

from .agents import AttractorRouter, RouteDecision
from .identity import canonical_json_bytes, require_sha256
from .qwen_bridge import QwenOoeFeatureReceipt


EXACT_ROUTER_BLANKET_SCHEMA = "immer-ooe-exact-router-blanket/v1"
ROUTER_DECISION_RECORD_SCHEMA = "immer-ooe-route-decision-record/v1"
_MAX_RECEIPT_BYTES = 16 * 1024 * 1024
_MAX_LABELS = 1_000_000


class ExactRouterBlanketError(ValueError):
    """An exact router blanket cannot be constructed or applied."""


class ExactRouterBlanketIntegrityError(ExactRouterBlanketError):
    """A blanket seal, closure, or exact decision binding is invalid."""


class ExactRouterBlanketStaleError(ExactRouterBlanketIntegrityError):
    """A blanket belongs to another feature, graph, or router calibration."""


class ExactRouterBlanketCoverageError(ExactRouterBlanketIntegrityError):
    """A candidate closure omits a label required for exact routing."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _canonical_labels(
    values: Sequence[str],
    *,
    field: str,
    empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ExactRouterBlanketError(f"{field} must be a label sequence")
    try:
        labels = tuple(values)
    except TypeError as exc:
        raise ExactRouterBlanketError(
            f"{field} must be a label sequence"
        ) from exc
    if (not labels and not empty) or len(labels) > _MAX_LABELS:
        raise ExactRouterBlanketError(
            f"{field} must contain 1..{_MAX_LABELS} labels"
        )
    if any(
        not isinstance(label, str)
        or not label
        or label != label.strip()
        or "\x00" in label
        or len(label.encode("utf-8")) > 1024
        for label in labels
    ):
        raise ExactRouterBlanketError(f"{field} contains a non-canonical label")
    if labels != tuple(sorted(set(labels))):
        raise ExactRouterBlanketError(f"{field} must be sorted and unique")
    return labels


def _finite_nonnegative(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExactRouterBlanketError(f"{field} must be finite non-negative")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ExactRouterBlanketError(f"{field} must be finite non-negative")
    return 0.0 if result == 0.0 else result


def _optional_distance(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    return _finite_nonnegative(value, field=field)


@dataclass(frozen=True, slots=True)
class RouterDecisionRecord:
    """Canonical finite JSON representation of one :class:`RouteDecision`.

    ``None`` is the sole wire representation of positive infinity.  This is
    needed for a calibrated one-label router, whose runner-up and margin are
    mathematically infinite, while canonical JSON forbids non-finite numbers.
    """

    label: str | None
    accepted: bool
    reason: str
    nearest_distance: float | None
    second_distance: float | None
    margin: float | None
    radius: float
    min_margin: float

    def __post_init__(self) -> None:
        if self.label is not None and (
            not isinstance(self.label, str)
            or not self.label
            or self.label != self.label.strip()
        ):
            raise ExactRouterBlanketError("decision label is invalid")
        if not isinstance(self.accepted, bool):
            raise ExactRouterBlanketError("decision accepted flag must be bool")
        if self.reason not in {
            "accepted",
            "ambiguous-margin",
            "outside-radius",
            "untrained",
        }:
            raise ExactRouterBlanketError("decision reason is invalid")
        object.__setattr__(
            self,
            "nearest_distance",
            _optional_distance(self.nearest_distance, field="nearest_distance"),
        )
        object.__setattr__(
            self,
            "second_distance",
            _optional_distance(self.second_distance, field="second_distance"),
        )
        object.__setattr__(
            self,
            "margin",
            _optional_distance(self.margin, field="margin"),
        )
        radius = _finite_nonnegative(self.radius, field="radius")
        if radius == 0.0:
            raise ExactRouterBlanketError("radius must be positive")
        object.__setattr__(self, "radius", radius)
        object.__setattr__(
            self,
            "min_margin",
            _finite_nonnegative(self.min_margin, field="min_margin"),
        )
        if self.accepted != (self.label is not None and self.reason == "accepted"):
            raise ExactRouterBlanketIntegrityError(
                "decision label, acceptance, and reason disagree"
            )
        if self.reason == "untrained":
            if (
                self.nearest_distance is not None
                or self.second_distance is not None
                or self.margin != 0.0
            ):
                raise ExactRouterBlanketIntegrityError(
                    "untrained decision infinity encoding is invalid"
                )
        elif self.nearest_distance is None:
            raise ExactRouterBlanketIntegrityError(
                "trained decision has no finite nearest distance"
            )
        if self.reason != "untrained" and (
            (self.second_distance is None) != (self.margin is None)
        ):
            raise ExactRouterBlanketIntegrityError(
                "runner-up and margin infinity tags disagree"
            )

    @classmethod
    def from_decision(cls, decision: RouteDecision) -> "RouterDecisionRecord":
        if not isinstance(decision, RouteDecision):
            raise TypeError("decision must be a RouteDecision")

        def encode(value: float) -> float | None:
            if math.isinf(value) and value > 0.0:
                return None
            return value

        return cls(
            label=decision.label,
            accepted=decision.accepted,
            reason=decision.reason,
            nearest_distance=encode(decision.nearest_distance),
            second_distance=encode(decision.second_distance),
            margin=encode(decision.margin),
            radius=decision.radius,
            min_margin=decision.min_margin,
        )

    def to_decision(self) -> RouteDecision:
        def decode(value: float | None) -> float:
            return math.inf if value is None else value

        return RouteDecision(
            label=self.label,
            accepted=self.accepted,
            reason=self.reason,
            nearest_distance=decode(self.nearest_distance),
            second_distance=decode(self.second_distance),
            margin=decode(self.margin),
            radius=self.radius,
            min_margin=self.min_margin,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "label": self.label,
            "margin": self.margin,
            "min_margin": self.min_margin,
            "nearest_distance": self.nearest_distance,
            "radius": self.radius,
            "reason": self.reason,
            "schema": ROUTER_DECISION_RECORD_SCHEMA,
            "second_distance": self.second_distance,
        }

    @classmethod
    def from_dict(cls, value: object) -> "RouterDecisionRecord":
        expected = {
            "accepted",
            "label",
            "margin",
            "min_margin",
            "nearest_distance",
            "radius",
            "reason",
            "schema",
            "second_distance",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or value.get("schema") != ROUTER_DECISION_RECORD_SCHEMA
        ):
            raise ExactRouterBlanketIntegrityError(
                "route decision record is invalid"
            )
        return cls(
            label=value["label"],
            accepted=value["accepted"],
            reason=value["reason"],
            nearest_distance=value["nearest_distance"],
            second_distance=value["second_distance"],
            margin=value["margin"],
            radius=value["radius"],
            min_margin=value["min_margin"],
        )


@dataclass(frozen=True, slots=True)
class ExactRouterBlanketReceipt:
    """Hash-sealed proof that one exact feature may use a sparse router scan."""

    model_pin_sha256: str
    weight_coordinate_sha256: str
    weight_graph_revision_sha256: str
    feature_schema_sha256: str
    action_schema_sha256: str
    feature_receipt_sha256: str
    input_site_identity_sha256: str
    atlas_graph_revision: GraphRevision
    router_calibration_sha256: str
    router_dimension: int
    label_universe: tuple[str, ...]
    centroid_universe_sha256: str
    candidate_closure: tuple[str, ...]
    ranking_head: tuple[str, ...]
    full_decision: RouterDecisionRecord
    sparse_decision: RouterDecisionRecord
    seal_sha256: str = ""

    def __post_init__(self) -> None:
        for field in (
            "model_pin_sha256",
            "weight_coordinate_sha256",
            "weight_graph_revision_sha256",
            "feature_schema_sha256",
            "action_schema_sha256",
            "feature_receipt_sha256",
            "input_site_identity_sha256",
            "router_calibration_sha256",
            "centroid_universe_sha256",
        ):
            object.__setattr__(
                self,
                field,
                require_sha256(getattr(self, field), field=field),
            )
        if not isinstance(self.atlas_graph_revision, GraphRevision):
            raise TypeError("atlas_graph_revision must be a GraphRevision")
        if (
            isinstance(self.router_dimension, bool)
            or not isinstance(self.router_dimension, int)
            or not 1 <= self.router_dimension <= 1_000_000
        ):
            raise ExactRouterBlanketError("router_dimension is invalid")
        universe = _canonical_labels(self.label_universe, field="label_universe")
        closure = _canonical_labels(self.candidate_closure, field="candidate_closure")
        object.__setattr__(self, "label_universe", universe)
        object.__setattr__(self, "candidate_closure", closure)
        if isinstance(self.ranking_head, (str, bytes)):
            raise ExactRouterBlanketError("ranking_head must be a label sequence")
        ranking_head = tuple(self.ranking_head)
        expected_head_size = min(2, len(universe))
        if (
            len(ranking_head) != expected_head_size
            or len(set(ranking_head)) != len(ranking_head)
            or any(
                not isinstance(label, str) or not label for label in ranking_head
            )
        ):
            raise ExactRouterBlanketIntegrityError("ranking_head is invalid")
        object.__setattr__(self, "ranking_head", ranking_head)
        if self.input_site_identity_sha256 not in universe:
            raise ExactRouterBlanketCoverageError(
                "input site is absent from the router label universe"
            )
        if not set(closure).issubset(universe):
            raise ExactRouterBlanketCoverageError(
                "candidate closure contains a label outside the universe"
            )
        required = {self.input_site_identity_sha256, *ranking_head}
        if not required.issubset(closure):
            raise ExactRouterBlanketCoverageError(
                "candidate closure omits input, nearest, or runner-up label"
            )
        if len(universe) > 1 and len(closure) < 2:
            raise ExactRouterBlanketCoverageError(
                "multi-label candidate closure must contain at least two labels"
            )
        if not isinstance(self.full_decision, RouterDecisionRecord) or not isinstance(
            self.sparse_decision, RouterDecisionRecord
        ):
            raise TypeError("full_decision and sparse_decision must be records")
        if self.full_decision != self.sparse_decision:
            raise ExactRouterBlanketIntegrityError(
                "full and sparse router decisions are not exactly equal"
            )
        if self.full_decision.reason == "untrained":
            raise ExactRouterBlanketIntegrityError(
                "an untrained router cannot publish an exact blanket"
            )
        if (
            self.full_decision.accepted
            and self.full_decision.label != ranking_head[0]
        ):
            raise ExactRouterBlanketIntegrityError(
                "accepted decision does not match the global nearest label"
            )
        claimed = self.seal_sha256
        if claimed:
            claimed = require_sha256(claimed, field="seal_sha256")
        expected = _digest(self.as_record())
        if claimed and claimed != expected:
            raise ExactRouterBlanketIntegrityError(
                "exact router blanket seal mismatch"
            )
        object.__setattr__(self, "seal_sha256", expected)

    def as_record(self) -> dict[str, Any]:
        return {
            "action_schema_sha256": self.action_schema_sha256,
            "atlas_graph_revision": self.atlas_graph_revision.to_document(),
            "atlas_graph_revision_sha256": self.atlas_graph_revision.sha256,
            "candidate_closure": list(self.candidate_closure),
            "centroid_universe_sha256": self.centroid_universe_sha256,
            "feature_receipt_sha256": self.feature_receipt_sha256,
            "feature_schema_sha256": self.feature_schema_sha256,
            "full_decision": self.full_decision.to_dict(),
            "input_site_identity_sha256": self.input_site_identity_sha256,
            "label_universe": list(self.label_universe),
            "model_pin_sha256": self.model_pin_sha256,
            "ranking_head": list(self.ranking_head),
            "router_calibration_sha256": self.router_calibration_sha256,
            "router_dimension": self.router_dimension,
            "sparse_decision": self.sparse_decision.to_dict(),
            "weight_coordinate_sha256": self.weight_coordinate_sha256,
            "weight_graph_revision_sha256": self.weight_graph_revision_sha256,
        }

    @property
    def sha256(self) -> str:
        return self.seal_sha256

    def verify_seal(self) -> None:
        if self.seal_sha256 != _digest(self.as_record()):
            raise ExactRouterBlanketIntegrityError(
                "exact router blanket seal mismatch"
            )

    def to_document(self) -> dict[str, Any]:
        self.verify_seal()
        return {
            "body": self.as_record(),
            "schema": EXACT_ROUTER_BLANKET_SCHEMA,
            "sha256": self.seal_sha256,
        }

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_document())
        if len(data) > _MAX_RECEIPT_BYTES:
            raise ExactRouterBlanketIntegrityError(
                "exact router blanket exceeds its byte bound"
            )
        return data

    @classmethod
    def from_document(
        cls,
        document: Mapping[str, Any],
    ) -> "ExactRouterBlanketReceipt":
        if not isinstance(document, Mapping) or set(document) != {
            "body",
            "schema",
            "sha256",
        }:
            raise ExactRouterBlanketIntegrityError(
                "exact router blanket envelope is invalid"
            )
        if document.get("schema") != EXACT_ROUTER_BLANKET_SCHEMA:
            raise ExactRouterBlanketIntegrityError(
                "exact router blanket schema is invalid"
            )
        body = document.get("body")
        expected_fields = {
            "action_schema_sha256",
            "atlas_graph_revision",
            "atlas_graph_revision_sha256",
            "candidate_closure",
            "centroid_universe_sha256",
            "feature_receipt_sha256",
            "feature_schema_sha256",
            "full_decision",
            "input_site_identity_sha256",
            "label_universe",
            "model_pin_sha256",
            "ranking_head",
            "router_calibration_sha256",
            "router_dimension",
            "sparse_decision",
            "weight_coordinate_sha256",
            "weight_graph_revision_sha256",
        }
        if not isinstance(body, Mapping) or set(body) != expected_fields:
            raise ExactRouterBlanketIntegrityError(
                "exact router blanket body is invalid"
            )
        try:
            claimed = require_sha256(document.get("sha256"), field="sha256")
            if claimed != _digest(body):
                raise ExactRouterBlanketIntegrityError(
                    "exact router blanket SHA-256 mismatch"
                )
            atlas_revision = GraphRevision.from_document(
                body["atlas_graph_revision"]
            )
            if body["atlas_graph_revision_sha256"] != atlas_revision.sha256:
                raise ExactRouterBlanketIntegrityError(
                    "Atlas graph revision binding mismatch"
                )
            receipt = cls(
                model_pin_sha256=body["model_pin_sha256"],
                weight_coordinate_sha256=body["weight_coordinate_sha256"],
                weight_graph_revision_sha256=body[
                    "weight_graph_revision_sha256"
                ],
                feature_schema_sha256=body["feature_schema_sha256"],
                action_schema_sha256=body["action_schema_sha256"],
                feature_receipt_sha256=body["feature_receipt_sha256"],
                input_site_identity_sha256=body[
                    "input_site_identity_sha256"
                ],
                atlas_graph_revision=atlas_revision,
                router_calibration_sha256=body[
                    "router_calibration_sha256"
                ],
                router_dimension=body["router_dimension"],
                label_universe=tuple(body["label_universe"]),
                centroid_universe_sha256=body[
                    "centroid_universe_sha256"
                ],
                candidate_closure=tuple(body["candidate_closure"]),
                ranking_head=tuple(body["ranking_head"]),
                full_decision=RouterDecisionRecord.from_dict(
                    body["full_decision"]
                ),
                sparse_decision=RouterDecisionRecord.from_dict(
                    body["sparse_decision"]
                ),
                seal_sha256=claimed,
            )
        except ExactRouterBlanketIntegrityError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ExactRouterBlanketIntegrityError(
                "exact router blanket reconstruction failed"
            ) from exc
        if receipt.to_document() != dict(document):
            raise ExactRouterBlanketIntegrityError(
                "exact router blanket canonical reconstruction mismatch"
            )
        return receipt

    @classmethod
    def from_bytes(cls, data: bytes) -> "ExactRouterBlanketReceipt":
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        if not data or len(data) > _MAX_RECEIPT_BYTES:
            raise ExactRouterBlanketIntegrityError(
                "exact router blanket byte length is invalid"
            )
        try:
            document = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExactRouterBlanketIntegrityError(
                "exact router blanket is invalid JSON"
            ) from exc
        try:
            canonical = canonical_json_bytes(document)
        except ValueError as exc:
            raise ExactRouterBlanketIntegrityError(
                "exact router blanket contains non-finite or invalid JSON"
            ) from exc
        if canonical != data:
            raise ExactRouterBlanketIntegrityError(
                "exact router blanket JSON is not canonical"
            )
        return cls.from_document(document)

    def assert_bound(
        self,
        feature_receipt: QwenOoeFeatureReceipt,
        router: AttractorRouter,
    ) -> None:
        """Validate every non-distance binding without doing a full scan."""

        if not isinstance(feature_receipt, QwenOoeFeatureReceipt):
            raise TypeError("feature_receipt must be a QwenOoeFeatureReceipt")
        if not isinstance(router, AttractorRouter):
            raise TypeError("router must be an AttractorRouter")
        self.verify_seal()
        feature_bindings = (
            self.model_pin_sha256,
            self.weight_coordinate_sha256,
            self.weight_graph_revision_sha256,
            self.feature_schema_sha256,
            self.action_schema_sha256,
            self.feature_receipt_sha256,
            self.input_site_identity_sha256,
            self.atlas_graph_revision,
            self.router_dimension,
        )
        current_feature = (
            feature_receipt.model_pin_sha256,
            feature_receipt.weight_coordinate_sha256,
            feature_receipt.weight_graph_revision_sha256,
            feature_receipt.feature_schema_sha256,
            feature_receipt.action_schema_sha256,
            feature_receipt.sha256,
            feature_receipt.site_identity.sha256,
            feature_receipt.atlas_graph_revision,
            feature_receipt.feature_dimensions,
        )
        if feature_bindings != current_feature:
            raise ExactRouterBlanketStaleError(
                "blanket belongs to another exact Qwen/OoE feature"
            )
        router_calibration = router.calibration_sha256
        if (
            router_calibration is None
            or self.router_calibration_sha256 != router_calibration
            or self.label_universe != router.labels
            or self.centroid_universe_sha256
            != router.centroid_universe_sha256
            or self.router_dimension != router.dimension
            or self.full_decision.radius != router.radius
            or self.full_decision.min_margin != router.min_margin
        ):
            raise ExactRouterBlanketStaleError(
                "blanket router calibration or centroid universe is stale"
            )


def _candidate_closure(
    router: AttractorRouter,
    feature_receipt: QwenOoeFeatureReceipt,
    ranked: tuple[tuple[float, str], ...],
    candidate_labels: Sequence[str] | None,
) -> tuple[str, ...]:
    ranking_head = tuple(label for _, label in ranked[:2])
    required = {feature_receipt.site_identity.sha256, *ranking_head}
    if candidate_labels is None:
        closure = tuple(sorted(required))
    else:
        closure = router._candidate_labels(candidate_labels)
    if not required.issubset(closure):
        raise ExactRouterBlanketCoverageError(
            "candidate closure omits input, global nearest, or global runner-up"
        )
    if len(router.labels) > 1 and len(closure) < 2:
        raise ExactRouterBlanketCoverageError(
            "multi-label candidate closure must contain at least two labels"
        )
    return closure


def build_exact_router_blanket(
    router: AttractorRouter,
    feature_receipt: QwenOoeFeatureReceipt,
    *,
    candidate_labels: Sequence[str] | None = None,
) -> ExactRouterBlanketReceipt:
    """Full-scan one exact feature and publish an exact sparse closure."""

    if not isinstance(router, AttractorRouter):
        raise TypeError("router must be an AttractorRouter")
    if not isinstance(feature_receipt, QwenOoeFeatureReceipt):
        raise TypeError("feature_receipt must be a QwenOoeFeatureReceipt")
    calibration = router.calibration_sha256
    if calibration is None:
        raise ExactRouterBlanketStaleError(
            "router must have a current calibration before blanket construction"
        )
    ranked = router._rank_candidates(feature_receipt.sketch_array)
    if not ranked:
        raise ExactRouterBlanketCoverageError(
            "an untrained router cannot build a blanket"
        )
    if feature_receipt.site_identity.sha256 not in router.labels:
        raise ExactRouterBlanketCoverageError(
            "exact feature input site is unknown to the router"
        )
    full = router._decision_from_ranked(ranked)
    closure = _candidate_closure(
        router,
        feature_receipt,
        ranked,
        candidate_labels,
    )
    sparse = router.decision_for_candidates(
        feature_receipt.sketch_array,
        closure,
    )
    if full != sparse:
        raise ExactRouterBlanketIntegrityError(
            "candidate scan does not exactly reproduce the full router decision"
        )
    return ExactRouterBlanketReceipt(
        model_pin_sha256=feature_receipt.model_pin_sha256,
        weight_coordinate_sha256=feature_receipt.weight_coordinate_sha256,
        weight_graph_revision_sha256=(
            feature_receipt.weight_graph_revision_sha256
        ),
        feature_schema_sha256=feature_receipt.feature_schema_sha256,
        action_schema_sha256=feature_receipt.action_schema_sha256,
        feature_receipt_sha256=feature_receipt.sha256,
        input_site_identity_sha256=feature_receipt.site_identity.sha256,
        atlas_graph_revision=feature_receipt.atlas_graph_revision,
        router_calibration_sha256=calibration,
        router_dimension=feature_receipt.feature_dimensions,
        label_universe=router.labels,
        centroid_universe_sha256=router.centroid_universe_sha256,
        candidate_closure=closure,
        ranking_head=tuple(label for _, label in ranked[:2]),
        full_decision=RouterDecisionRecord.from_decision(full),
        sparse_decision=RouterDecisionRecord.from_decision(sparse),
    )


def apply_exact_router_blanket(
    router: AttractorRouter,
    feature_receipt: QwenOoeFeatureReceipt,
    blanket_receipt: ExactRouterBlanketReceipt,
) -> RouteDecision:
    """Execute only the proven closure for the receipt's exact feature."""

    if not isinstance(blanket_receipt, ExactRouterBlanketReceipt):
        raise TypeError("blanket_receipt must be an ExactRouterBlanketReceipt")
    blanket_receipt.assert_bound(feature_receipt, router)
    sparse = router.decision_for_candidates(
        feature_receipt.sketch_array,
        blanket_receipt.candidate_closure,
    )
    record = RouterDecisionRecord.from_decision(sparse)
    if (
        record != blanket_receipt.sparse_decision
        or record != blanket_receipt.full_decision
    ):
        raise ExactRouterBlanketIntegrityError(
            "sparse router replay differs from the sealed full decision"
        )
    return sparse


def replay_exact_router_blanket(
    router: AttractorRouter,
    feature_receipt: QwenOoeFeatureReceipt,
    blanket_receipt: ExactRouterBlanketReceipt,
) -> RouteDecision:
    """Recompute the full proof and sparse replay for offline verification."""

    if not isinstance(blanket_receipt, ExactRouterBlanketReceipt):
        raise TypeError("blanket_receipt must be an ExactRouterBlanketReceipt")
    blanket_receipt.assert_bound(feature_receipt, router)
    ranked = router._rank_candidates(feature_receipt.sketch_array)
    ranking_head = tuple(label for _, label in ranked[:2])
    if ranking_head != blanket_receipt.ranking_head:
        raise ExactRouterBlanketCoverageError(
            "sealed candidate closure omits the current global ranking head"
        )
    full = router._decision_from_ranked(ranked)
    sparse = router.decision_for_candidates(
        feature_receipt.sketch_array,
        blanket_receipt.candidate_closure,
    )
    full_record = RouterDecisionRecord.from_decision(full)
    sparse_record = RouterDecisionRecord.from_decision(sparse)
    if (
        full != sparse
        or full_record != blanket_receipt.full_decision
        or sparse_record != blanket_receipt.sparse_decision
    ):
        raise ExactRouterBlanketIntegrityError(
            "full and sparse router replay are not exactly equal"
        )
    return sparse


def verify_exact_router_blanket(
    router: AttractorRouter,
    feature_receipt: QwenOoeFeatureReceipt,
    blanket_receipt: ExactRouterBlanketReceipt,
) -> bool:
    replay_exact_router_blanket(router, feature_receipt, blanket_receipt)
    return True


__all__ = [
    "EXACT_ROUTER_BLANKET_SCHEMA",
    "ROUTER_DECISION_RECORD_SCHEMA",
    "ExactRouterBlanketCoverageError",
    "ExactRouterBlanketError",
    "ExactRouterBlanketIntegrityError",
    "ExactRouterBlanketReceipt",
    "ExactRouterBlanketStaleError",
    "RouterDecisionRecord",
    "apply_exact_router_blanket",
    "build_exact_router_blanket",
    "replay_exact_router_blanket",
    "verify_exact_router_blanket",
]
