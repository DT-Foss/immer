"""Exact algebraic crystals extracted from learned scalar invariants.

The measured organ-grafting path learns one scalar ``phi`` and then replaces
the learned approximation by the exact group map it carries:

* ``phi ~= alpha * x + gamma`` for the additive integer lattice;
* ``phi ~= alpha * log(x) + gamma`` for the positive multiplicative lattice;
* compact, regularly spaced phase centres for a finite cyclic group ``Z_n``.

Admission is deliberately separate from execution.  A bank policy decides
whether a fit is accepted, ambiguous, or rejected.  An admitted crystal only
contains the exact map and the hashes of the evidence that produced it; no
threshold is part of the executable object.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .identity import canonical_json_bytes, require_sha256


ADDITIVE = "additive"
MULTIPLICATIVE = "multiplicative"
CYCLIC = "cyclic"
MAP_FAMILIES = frozenset((ADDITIVE, MULTIPLICATIVE, CYCLIC))

ACCEPTED = "accepted"
AMBIGUOUS = "ambiguous"
REJECTED = "rejected"
ADMISSION_OUTCOMES = frozenset((ACCEPTED, AMBIGUOUS, REJECTED))

INVARIANT_SUPPORT_SCHEMA = "immer-ooe-invariant-support/v1"
CRYSTAL_POLICY_SCHEMA = "immer-ooe-algebraic-crystal-policy/v1"
FAMILY_FIT_SCHEMA = "immer-ooe-algebraic-family-fit/v1"
FIT_RECEIPT_SCHEMA = "immer-ooe-algebraic-fit-receipt/v1"
CRYSTALLIZED_MAP_SCHEMA = "immer-ooe-crystallized-map/v1"
SNAPPED_LATENT_SCHEMA = "immer-ooe-snapped-latent/v1"
EXECUTION_RECEIPT_SCHEMA = "immer-ooe-group-execution-receipt/v1"

MAX_SUPPORTS = 16_384
MAX_CYCLIC_ORDER = 4_096
MAX_DEPLOYMENT_ABS = 10**12
MAX_EXECUTION_STEPS = 1_000_000
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024

ALGORITHM_SHA256 = hashlib.sha256(
    canonical_json_bytes(
        {
            "cyclic": "best unit winding; min(aligned resultant, per-class resultant, single-chart concentration)",
            "fit": "ordinary least squares with deterministic zero-or-one leave-one-out trim",
            "linear": "phi=alpha*x+gamma",
            "log": "phi=alpha*log(x)+gamma",
            "snap": "integer, nearest-log-integer, or nearest phase centre",
            "version": FIT_RECEIPT_SCHEMA,
        }
    )
).hexdigest()

FloatArray = NDArray[np.float64]
AdmissionOutcome = Literal["accepted", "ambiguous", "rejected"]


class AlgebraicCrystalIntegrityError(ValueError):
    """A canonical algebraic-crystal artifact failed verification."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return 0.0 if result == 0.0 else result


def _integer(
    value: object,
    *,
    field: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} is below its minimum")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field} exceeds its maximum")
    return result


def _probability(value: object, *, field: str) -> float:
    result = _finite(value, field=field)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must lie in [0, 1]")
    return result


def _strict_decode(data: bytes) -> object:
    if not isinstance(data, bytes) or len(data) > MAX_PAYLOAD_BYTES:
        raise AlgebraicCrystalIntegrityError(
            "payload must be bounded immutable bytes"
        )

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
        decoded = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AlgebraicCrystalIntegrityError("payload is not strict JSON") from exc
    if canonical_json_bytes(decoded) != data:
        raise AlgebraicCrystalIntegrityError("payload is not canonical JSON")
    return decoded


def _seal(schema: str, body: Mapping[str, Any]) -> dict[str, Any]:
    exact = dict(body)
    return {
        "body": exact,
        "body_sha256": _digest(exact),
        "schema": schema,
    }


def _open_seal(value: object, *, schema: str) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"body", "body_sha256", "schema"}
        or value.get("schema") != schema
        or not isinstance(value.get("body"), dict)
    ):
        raise AlgebraicCrystalIntegrityError(f"unsupported {schema} envelope")
    expected = require_sha256(value["body_sha256"], field="body_sha256")
    body = dict(value["body"])
    if _digest(body) != expected:
        raise AlgebraicCrystalIntegrityError("artifact body SHA-256 mismatch")
    return body


def _require_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    if set(value) != expected:
        raise AlgebraicCrystalIntegrityError(
            f"{field} has unknown or missing fields"
        )


@dataclass(frozen=True, slots=True)
class InvariantSupport:
    """One measured point on a learned scalar invariant.

    ``group_element`` is the known exact integer represented by the token or
    feature identified by ``element_sha256``.  The two evidence hashes and
    source receipt make the scalar measurement indivisible from provenance.
    """

    element_sha256: str
    group_element: int
    learned_phi: float
    verifier_sha256: str
    evidence_sha256: str
    source_receipt_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "element_sha256",
            "verifier_sha256",
            "evidence_sha256",
            "source_receipt_sha256",
        ):
            object.__setattr__(
                self,
                field,
                require_sha256(getattr(self, field), field=field),
            )
        object.__setattr__(
            self,
            "group_element",
            _integer(
                self.group_element,
                field="group_element",
                minimum=-MAX_DEPLOYMENT_ABS,
                maximum=MAX_DEPLOYMENT_ABS,
            ),
        )
        object.__setattr__(
            self,
            "learned_phi",
            _finite(self.learned_phi, field="learned_phi"),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "element_sha256": self.element_sha256,
            "evidence_sha256": self.evidence_sha256,
            "group_element": self.group_element,
            "learned_phi": self.learned_phi,
            "source_receipt_sha256": self.source_receipt_sha256,
            "verifier_sha256": self.verifier_sha256,
        }

    @classmethod
    def from_record(cls, value: object) -> "InvariantSupport":
        if not isinstance(value, dict):
            raise AlgebraicCrystalIntegrityError("support must be an object")
        _require_keys(
            value,
            {
                "element_sha256",
                "evidence_sha256",
                "group_element",
                "learned_phi",
                "source_receipt_sha256",
                "verifier_sha256",
            },
            field="support",
        )
        return cls(**value)

    @property
    def sha256(self) -> str:
        return _digest({"schema": INVARIANT_SUPPORT_SCHEMA, **self.to_record()})

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(_seal(INVARIANT_SUPPORT_SCHEMA, self.to_record()))

    @classmethod
    def from_bytes(cls, data: bytes) -> "InvariantSupport":
        return cls.from_record(
            _open_seal(_strict_decode(data), schema=INVARIANT_SUPPORT_SCHEMA)
        )


@dataclass(frozen=True, slots=True)
class CrystalAdmissionPolicy:
    """OrganBank thresholds; never embedded in an executable crystal."""

    acceptance_score: float = 0.95
    ambiguity_score: float = 0.90
    minimum_margin: float = 0.02
    minimum_abs_slope: float = 1e-9
    minimum_support: int = 6
    allow_single_trim: bool = True

    def __post_init__(self) -> None:
        acceptance = _probability(self.acceptance_score, field="acceptance_score")
        ambiguity = _probability(self.ambiguity_score, field="ambiguity_score")
        margin = _probability(self.minimum_margin, field="minimum_margin")
        slope = _finite(self.minimum_abs_slope, field="minimum_abs_slope")
        if ambiguity > acceptance:
            raise ValueError("ambiguity_score cannot exceed acceptance_score")
        if slope <= 0.0:
            raise ValueError("minimum_abs_slope must be positive")
        support = _integer(
            self.minimum_support,
            field="minimum_support",
            minimum=4,
            maximum=MAX_SUPPORTS,
        )
        if not isinstance(self.allow_single_trim, bool):
            raise ValueError("allow_single_trim must be boolean")
        object.__setattr__(self, "acceptance_score", acceptance)
        object.__setattr__(self, "ambiguity_score", ambiguity)
        object.__setattr__(self, "minimum_margin", margin)
        object.__setattr__(self, "minimum_abs_slope", slope)
        object.__setattr__(self, "minimum_support", support)

    def to_record(self) -> dict[str, Any]:
        return {
            "acceptance_score": self.acceptance_score,
            "allow_single_trim": self.allow_single_trim,
            "ambiguity_score": self.ambiguity_score,
            "minimum_abs_slope": self.minimum_abs_slope,
            "minimum_margin": self.minimum_margin,
            "minimum_support": self.minimum_support,
            "schema": CRYSTAL_POLICY_SCHEMA,
        }

    @classmethod
    def from_record(cls, value: object) -> "CrystalAdmissionPolicy":
        if not isinstance(value, dict):
            raise AlgebraicCrystalIntegrityError("policy must be an object")
        _require_keys(
            value,
            {
                "acceptance_score",
                "allow_single_trim",
                "ambiguity_score",
                "minimum_abs_slope",
                "minimum_margin",
                "minimum_support",
                "schema",
            },
            field="policy",
        )
        if value["schema"] != CRYSTAL_POLICY_SCHEMA:
            raise AlgebraicCrystalIntegrityError("unsupported bank policy")
        fields = dict(value)
        del fields["schema"]
        return cls(**fields)

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())


@dataclass(frozen=True, slots=True)
class MapFamilyFit:
    """One completely bound candidate map fit."""

    key: str
    family: str
    cyclic_order: int | None
    valid: bool
    failure: str | None
    alpha: float | None
    gamma: float | None
    score: float
    untrimmed_score: float
    dropped_support_sha256: str | None
    used_support_sha256s: tuple[str, ...]
    phase_centers: tuple[float, ...] = ()
    phase_resultants: tuple[float, ...] = ()
    aligned_resultant: float | None = None
    chart_concentration: float | None = None

    def __post_init__(self) -> None:
        if self.family not in MAP_FAMILIES:
            raise ValueError("unknown map family")
        expected_key = (
            f"cyclic:{self.cyclic_order}" if self.family == CYCLIC else self.family
        )
        if self.key != expected_key:
            raise ValueError("family-fit key disagrees with its map family")
        if not isinstance(self.valid, bool):
            raise ValueError("valid must be boolean")
        order = self.cyclic_order
        if self.family == CYCLIC:
            order = _integer(
                order,
                field="cyclic_order",
                minimum=2,
                maximum=MAX_CYCLIC_ORDER,
            )
        elif order is not None:
            raise ValueError("non-cyclic fit cannot carry a cyclic order")
        object.__setattr__(self, "cyclic_order", order)

        score = _finite(self.score, field="score")
        raw_score = _finite(self.untrimmed_score, field="untrimmed_score")
        if score > 1.0 + 1e-12 or raw_score > 1.0 + 1e-12:
            raise ValueError("fit score cannot exceed one")
        object.__setattr__(self, "score", min(1.0, score))
        object.__setattr__(self, "untrimmed_score", min(1.0, raw_score))

        hashes = tuple(
            require_sha256(value, field="used_support_sha256")
            for value in self.used_support_sha256s
        )
        if hashes != tuple(sorted(hashes)) or len(set(hashes)) != len(hashes):
            raise ValueError("used support hashes must be sorted and unique")
        object.__setattr__(self, "used_support_sha256s", hashes)
        if self.dropped_support_sha256 is not None:
            object.__setattr__(
                self,
                "dropped_support_sha256",
                require_sha256(
                    self.dropped_support_sha256,
                    field="dropped_support_sha256",
                ),
            )

        if self.valid:
            if self.failure is not None or self.alpha is None or self.gamma is None:
                raise ValueError("valid fit lacks exact parameters")
            object.__setattr__(self, "alpha", _finite(self.alpha, field="alpha"))
            object.__setattr__(self, "gamma", _finite(self.gamma, field="gamma"))
            if len(hashes) < 3:
                raise ValueError("valid fit needs at least three supports")
        else:
            if not isinstance(self.failure, str) or not self.failure:
                raise ValueError("invalid fit needs a failure reason")
            if self.alpha is not None or self.gamma is not None:
                raise ValueError("invalid fit cannot carry parameters")

        centers = tuple(_finite(value, field="phase_center") for value in self.phase_centers)
        resultants = tuple(
            _probability(value, field="phase_resultant")
            for value in self.phase_resultants
        )
        object.__setattr__(self, "phase_centers", centers)
        object.__setattr__(self, "phase_resultants", resultants)
        if self.family == CYCLIC and self.valid:
            assert order is not None
            if len(centers) != order or len(resultants) != order:
                raise ValueError("cyclic fit must bind every phase class")
            winding = int(round(float(self.alpha)))
            if (
                abs(float(self.alpha) - winding) > 1e-12
                or not 1 <= winding < order
                or math.gcd(winding, order) != 1
            ):
                raise ValueError("cyclic alpha must be an invertible winding")
            if not 0.0 <= float(self.gamma) < order:
                raise ValueError("cyclic gamma must be a canonical phase")
            if any(not 0.0 <= value < order for value in centers):
                raise ValueError("phase centres must lie in the canonical circle")
            if self.aligned_resultant is None or self.chart_concentration is None:
                raise ValueError("cyclic fit lacks resultant diagnostics")
            object.__setattr__(
                self,
                "aligned_resultant",
                _probability(self.aligned_resultant, field="aligned_resultant"),
            )
            object.__setattr__(
                self,
                "chart_concentration",
                _probability(self.chart_concentration, field="chart_concentration"),
            )
        elif centers or resultants or self.aligned_resultant is not None or self.chart_concentration is not None:
            raise ValueError("non-cyclic or invalid fit carries cyclic diagnostics")

    def to_record(self) -> dict[str, Any]:
        return {
            "aligned_resultant": self.aligned_resultant,
            "alpha": self.alpha,
            "chart_concentration": self.chart_concentration,
            "cyclic_order": self.cyclic_order,
            "dropped_support_sha256": self.dropped_support_sha256,
            "failure": self.failure,
            "family": self.family,
            "gamma": self.gamma,
            "key": self.key,
            "phase_centers": list(self.phase_centers),
            "phase_resultants": list(self.phase_resultants),
            "schema": FAMILY_FIT_SCHEMA,
            "score": self.score,
            "untrimmed_score": self.untrimmed_score,
            "used_support_sha256s": list(self.used_support_sha256s),
            "valid": self.valid,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())


def _r2_fit(x: FloatArray, y: FloatArray) -> tuple[float, float, float] | None:
    x_centered = x - float(x.mean())
    denominator = float(np.dot(x_centered, x_centered))
    y_centered = y - float(y.mean())
    total = float(np.dot(y_centered, y_centered))
    if denominator <= 1e-24 or total <= 1e-24:
        return None
    alpha = float(np.dot(x_centered, y_centered) / denominator)
    gamma = float(y.mean() - alpha * x.mean())
    residual = y - (alpha * x + gamma)
    score = 1.0 - float(np.dot(residual, residual) / total)
    return (
        0.0 if alpha == 0.0 else alpha,
        0.0 if gamma == 0.0 else gamma,
        min(1.0, score),
    )


def _trim_candidates(count: int, enabled: bool) -> tuple[int | None, ...]:
    return (None, *range(count)) if enabled else (None,)


def _invalid_fit(
    family: str,
    *,
    order: int | None,
    failure: str,
    supports: Sequence[InvariantSupport],
) -> MapFamilyFit:
    return MapFamilyFit(
        key=f"cyclic:{order}" if family == CYCLIC else family,
        family=family,
        cyclic_order=order,
        valid=False,
        failure=failure,
        alpha=None,
        gamma=None,
        score=-1.0,
        untrimmed_score=-1.0,
        dropped_support_sha256=None,
        used_support_sha256s=tuple(sorted(item.sha256 for item in supports)),
    )


def _fit_regression_family(
    supports: Sequence[InvariantSupport],
    *,
    family: Literal["additive", "multiplicative"],
    policy: CrystalAdmissionPolicy,
) -> MapFamilyFit:
    if family == MULTIPLICATIVE and any(item.group_element <= 0 for item in supports):
        return _invalid_fit(
            family,
            order=None,
            failure="multiplicative map requires positive group elements",
            supports=supports,
        )
    if len(supports) < max(4, policy.minimum_support):
        return _invalid_fit(
            family,
            order=None,
            failure="insufficient support for bank policy",
            supports=supports,
        )

    raw_x = np.asarray([item.group_element for item in supports], dtype=np.float64)
    if family == MULTIPLICATIVE:
        raw_x = np.log(raw_x)
    raw_y = np.asarray([item.learned_phi for item in supports], dtype=np.float64)
    full = _r2_fit(raw_x, raw_y)
    if full is None:
        return _invalid_fit(
            family,
            order=None,
            failure="degenerate support variance",
            supports=supports,
        )
    untrimmed = full[2]
    choices: list[tuple[float, int, float, float, int | None]] = []
    for dropped in _trim_candidates(len(supports), policy.allow_single_trim):
        keep = np.ones(len(supports), dtype=bool)
        if dropped is not None:
            keep[dropped] = False
        result = _r2_fit(raw_x[keep], raw_y[keep])
        if result is None:
            continue
        alpha, gamma, score = result
        # Prefer no trim on an exact tie, then the canonical support order.
        trim_rank = -1 if dropped is None else dropped
        choices.append((score, -trim_rank, alpha, gamma, dropped))
    if not choices:
        return _invalid_fit(
            family,
            order=None,
            failure="no non-degenerate leave-one-out fit",
            supports=supports,
        )
    _, _, alpha, gamma, dropped = max(choices, key=lambda item: (item[0], item[1]))
    keep_support = tuple(
        sorted(item.sha256 for index, item in enumerate(supports) if index != dropped)
    )
    return MapFamilyFit(
        key=family,
        family=family,
        cyclic_order=None,
        valid=True,
        failure=None,
        alpha=alpha,
        gamma=gamma,
        score=max(item[0] for item in choices),
        untrimmed_score=untrimmed,
        dropped_support_sha256=(
            None if dropped is None else supports[dropped].sha256
        ),
        used_support_sha256s=keep_support,
    )


@dataclass(frozen=True, slots=True)
class _CyclicCandidate:
    score: float
    winding: int
    gamma: float
    centers: tuple[float, ...]
    resultants: tuple[float, ...]
    aligned: float
    chart: float


def _cyclic_candidate(
    supports: Sequence[InvariantSupport],
    *,
    order: int,
) -> _CyclicCandidate | None:
    if len(supports) < order:
        return None
    elements = np.asarray([item.group_element % order for item in supports], dtype=np.int64)
    phi = np.asarray([item.learned_phi for item in supports], dtype=np.float64)
    if set(elements.tolist()) != set(range(order)):
        return None

    angle = 2.0 * math.pi * phi / order
    centers: list[float] = []
    resultants: list[float] = []
    chart_scores: list[float] = []
    for residue in range(order):
        values = phi[elements == residue]
        z = np.exp(1j * 2.0 * math.pi * values / order).mean()
        resultant = float(abs(z))
        center = (order * math.atan2(float(z.imag), float(z.real)) / (2.0 * math.pi)) % order
        centers.append(0.0 if center == 0.0 else center)
        resultants.append(min(1.0, resultant))

        # The measured circle organ occupies one phase chart: equivalent
        # phases from different full windings are not silently treated as one
        # learned centre.  This separates a wound additive line from a true
        # finite-group invariant while retaining the circular resultant.
        turns = np.rint((values - center) / order)
        lifted = values - turns * order
        spread = float(np.mean((lifted - center) ** 2))
        scale = order / (2.0 * math.pi)
        local_chart = math.exp(-0.5 * spread / max(scale * scale, 1e-24))
        # A learned centre may straddle the chosen zero-phase cut, but it must
        # still fit inside one finite-group chart.  Values spanning several
        # complete turns are an additive line viewed modulo n, not a learned
        # finite-group centre.
        if float(np.ptp(values)) > order * (1.0 + 1e-12):
            local_chart = 0.0
        chart_scores.append(local_chart)

    best: _CyclicCandidate | None = None
    for winding in range(1, order):
        if math.gcd(winding, order) != 1:
            continue
        residual = angle - 2.0 * math.pi * winding * elements / order
        z = np.exp(1j * residual).mean()
        aligned = min(1.0, float(abs(z)))
        gamma = (order * math.atan2(float(z.imag), float(z.real)) / (2.0 * math.pi)) % order
        score = min(aligned, min(resultants), min(chart_scores))
        candidate = _CyclicCandidate(
            score=score,
            winding=winding,
            gamma=0.0 if gamma == 0.0 else gamma,
            centers=tuple(centers),
            resultants=tuple(resultants),
            aligned=aligned,
            chart=min(chart_scores),
        )
        if best is None or (candidate.score, -candidate.winding) > (
            best.score,
            -best.winding,
        ):
            best = candidate
    return best


def _fit_cyclic_family(
    supports: Sequence[InvariantSupport],
    *,
    order: int,
    policy: CrystalAdmissionPolicy,
) -> MapFamilyFit:
    if len(supports) < max(policy.minimum_support, order):
        return _invalid_fit(
            CYCLIC,
            order=order,
            failure="insufficient support for cyclic order and bank policy",
            supports=supports,
        )
    full = _cyclic_candidate(supports, order=order)
    if full is None:
        return _invalid_fit(
            CYCLIC,
            order=order,
            failure="cyclic support does not cover every residue",
            supports=supports,
        )

    choices: list[tuple[float, int, _CyclicCandidate, int | None]] = []
    for dropped in _trim_candidates(len(supports), policy.allow_single_trim):
        used = tuple(item for index, item in enumerate(supports) if index != dropped)
        candidate = _cyclic_candidate(used, order=order)
        if candidate is None:
            continue
        trim_rank = -1 if dropped is None else dropped
        choices.append((candidate.score, -trim_rank, candidate, dropped))
    if not choices:
        return _invalid_fit(
            CYCLIC,
            order=order,
            failure="no valid cyclic leave-one-out fit",
            supports=supports,
        )
    _, _, best, dropped = max(choices, key=lambda item: (item[0], item[1]))
    return MapFamilyFit(
        key=f"cyclic:{order}",
        family=CYCLIC,
        cyclic_order=order,
        valid=True,
        failure=None,
        alpha=float(best.winding),
        gamma=best.gamma,
        score=best.score,
        untrimmed_score=full.score,
        dropped_support_sha256=(
            None if dropped is None else supports[dropped].sha256
        ),
        used_support_sha256s=tuple(
            sorted(item.sha256 for index, item in enumerate(supports) if index != dropped)
        ),
        phase_centers=best.centers,
        phase_resultants=best.resultants,
        aligned_resultant=best.aligned,
        chart_concentration=best.chart,
    )


@dataclass(frozen=True, slots=True)
class CrystallizationFitReceipt:
    """Canonical, self-recomputing OrganBank intake receipt."""

    supports: tuple[InvariantSupport, ...]
    policy: CrystalAdmissionPolicy
    candidate_cyclic_orders: tuple[int, ...]
    fits: tuple[MapFamilyFit, ...]
    best_fit_key: str | None
    second_fit_key: str | None
    best_score: float
    second_score: float
    best_vs_second_margin: float
    outcome: AdmissionOutcome
    reason: str
    algorithm_sha256: str = ALGORITHM_SHA256

    def __post_init__(self) -> None:
        supports = tuple(self.supports)
        if not 1 <= len(supports) <= MAX_SUPPORTS:
            raise ValueError("support count is outside the hard bound")
        if not all(isinstance(item, InvariantSupport) for item in supports):
            raise TypeError("supports must contain InvariantSupport values")
        if supports != tuple(sorted(supports, key=lambda item: item.sha256)):
            raise ValueError("supports must be sorted canonically")
        hashes = tuple(item.sha256 for item in supports)
        if len(set(hashes)) != len(hashes):
            raise ValueError("supports must be unique")
        if len({item.element_sha256 for item in supports}) != len(supports):
            raise ValueError("element identities must be unique")
        object.__setattr__(self, "supports", supports)
        if not isinstance(self.policy, CrystalAdmissionPolicy):
            raise TypeError("policy must be CrystalAdmissionPolicy")
        orders = tuple(
            _integer(
                value,
                field="candidate_cyclic_order",
                minimum=2,
                maximum=MAX_CYCLIC_ORDER,
            )
            for value in self.candidate_cyclic_orders
        )
        if orders != tuple(sorted(set(orders))):
            raise ValueError("candidate cyclic orders must be sorted and unique")
        object.__setattr__(self, "candidate_cyclic_orders", orders)
        fits = tuple(self.fits)
        if fits != tuple(sorted(fits, key=lambda item: item.key)):
            raise ValueError("family fits must be sorted by key")
        if len({item.key for item in fits}) != len(fits):
            raise ValueError("family fits must be unique")
        object.__setattr__(self, "fits", fits)
        for name in ("best_score", "second_score", "best_vs_second_margin"):
            object.__setattr__(self, name, _finite(getattr(self, name), field=name))
        if self.outcome not in ADMISSION_OUTCOMES:
            raise ValueError("unknown admission outcome")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("fit receipt needs a reason")
        object.__setattr__(
            self,
            "algorithm_sha256",
            require_sha256(self.algorithm_sha256, field="algorithm_sha256"),
        )

    @property
    def best_fit(self) -> MapFamilyFit | None:
        return next((item for item in self.fits if item.key == self.best_fit_key), None)

    def to_body(self) -> dict[str, Any]:
        return {
            "algorithm_sha256": self.algorithm_sha256,
            "best_fit_key": self.best_fit_key,
            "best_score": self.best_score,
            "best_vs_second_margin": self.best_vs_second_margin,
            "candidate_cyclic_orders": list(self.candidate_cyclic_orders),
            "fits": [item.to_record() for item in self.fits],
            "outcome": self.outcome,
            "policy": self.policy.to_record(),
            "reason": self.reason,
            "second_fit_key": self.second_fit_key,
            "second_score": self.second_score,
            "supports": [item.to_record() for item in self.supports],
        }

    @property
    def sha256(self) -> str:
        return _digest(_seal(FIT_RECEIPT_SCHEMA, self.to_body()))

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(_seal(FIT_RECEIPT_SCHEMA, self.to_body()))

    @classmethod
    def from_bytes(cls, data: bytes) -> "CrystallizationFitReceipt":
        body = _open_seal(_strict_decode(data), schema=FIT_RECEIPT_SCHEMA)
        _require_keys(
            body,
            {
                "algorithm_sha256",
                "best_fit_key",
                "best_score",
                "best_vs_second_margin",
                "candidate_cyclic_orders",
                "fits",
                "outcome",
                "policy",
                "reason",
                "second_fit_key",
                "second_score",
                "supports",
            },
            field="fit receipt",
        )
        if body["algorithm_sha256"] != ALGORITHM_SHA256:
            raise AlgebraicCrystalIntegrityError("unknown fitting algorithm")
        if not isinstance(body["supports"], list):
            raise AlgebraicCrystalIntegrityError("supports must be a list")
        if not isinstance(body["candidate_cyclic_orders"], list):
            raise AlgebraicCrystalIntegrityError("cyclic orders must be a list")
        supports = tuple(InvariantSupport.from_record(item) for item in body["supports"])
        policy = CrystalAdmissionPolicy.from_record(body["policy"])
        recomputed = fit_algebraic_map(
            supports,
            policy=policy,
            candidate_cyclic_orders=tuple(body["candidate_cyclic_orders"]),
        )
        if recomputed.to_body() != body:
            raise AlgebraicCrystalIntegrityError(
                "fit receipt does not match recomputed samples, trim, fits, or policy"
            )
        return recomputed


def fit_algebraic_map(
    supports: Iterable[InvariantSupport],
    *,
    policy: CrystalAdmissionPolicy | None = None,
    candidate_cyclic_orders: Sequence[int] = (),
) -> CrystallizationFitReceipt:
    """Fit every requested group map and issue one strict admission receipt.

    A finite cyclic order is supplied by the bank's action schema.  It is not
    inferred from an arbitrary finite list: the same integer observations can
    be quotiented by many ``n`` and therefore cannot identify ``Z_n`` alone.
    """

    exact_policy = policy or CrystalAdmissionPolicy()
    values = tuple(supports)
    if not values:
        raise ValueError("at least one invariant support is required")
    if not all(isinstance(item, InvariantSupport) for item in values):
        raise TypeError("supports must contain InvariantSupport values")
    canonical = tuple(sorted(values, key=lambda item: item.sha256))
    if len(canonical) > MAX_SUPPORTS:
        raise ValueError("support count exceeds MAX_SUPPORTS")
    if len({item.sha256 for item in canonical}) != len(canonical):
        raise ValueError("supports must be unique")
    if len({item.element_sha256 for item in canonical}) != len(canonical):
        raise ValueError("element identities must be unique")
    orders = tuple(
        sorted(
            {
                _integer(
                    value,
                    field="candidate_cyclic_order",
                    minimum=2,
                    maximum=MAX_CYCLIC_ORDER,
                )
                for value in candidate_cyclic_orders
            }
        )
    )

    fits = [
        _fit_regression_family(canonical, family=ADDITIVE, policy=exact_policy),
        _fit_regression_family(
            canonical,
            family=MULTIPLICATIVE,
            policy=exact_policy,
        ),
    ]
    fits.extend(
        _fit_cyclic_family(canonical, order=order, policy=exact_policy)
        for order in orders
    )
    fit_tuple = tuple(sorted(fits, key=lambda item: item.key))
    ranked = sorted(
        (item for item in fit_tuple if item.valid),
        key=lambda item: (-item.score, item.key),
    )
    best = ranked[0] if ranked else None
    second = ranked[1] if len(ranked) > 1 else None
    best_score = -1.0 if best is None else best.score
    second_score = -1.0 if second is None else second.score
    margin = best_score - second_score

    if best is None:
        outcome: AdmissionOutcome = REJECTED
        reason = "no valid map family"
    elif best.family != CYCLIC and abs(float(best.alpha)) < exact_policy.minimum_abs_slope:
        outcome = REJECTED
        reason = "best map slope is below bank policy"
    elif (
        best.score >= exact_policy.acceptance_score
        and margin >= exact_policy.minimum_margin
    ):
        outcome = ACCEPTED
        reason = "best map clears score and best-vs-second margin"
    elif best.score >= exact_policy.ambiguity_score:
        outcome = AMBIGUOUS
        reason = "map signal exists but does not clear every admission threshold"
    else:
        outcome = REJECTED
        reason = "no map clears the ambiguity threshold"

    return CrystallizationFitReceipt(
        supports=canonical,
        policy=exact_policy,
        candidate_cyclic_orders=orders,
        fits=fit_tuple,
        best_fit_key=None if best is None else best.key,
        second_fit_key=None if second is None else second.key,
        best_score=best_score,
        second_score=second_score,
        best_vs_second_margin=margin,
        outcome=outcome,
        reason=reason,
    )


@dataclass(frozen=True, slots=True)
class SnappedLatent:
    map_sha256: str
    family: str
    input_phi: float
    group_element: int
    latent_coordinate: float
    snapped_phi: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "map_sha256",
            require_sha256(self.map_sha256, field="map_sha256"),
        )
        if self.family not in MAP_FAMILIES:
            raise ValueError("unknown snapped-latent family")
        object.__setattr__(self, "input_phi", _finite(self.input_phi, field="input_phi"))
        object.__setattr__(
            self,
            "group_element",
            _integer(self.group_element, field="group_element"),
        )
        object.__setattr__(
            self,
            "latent_coordinate",
            _finite(self.latent_coordinate, field="latent_coordinate"),
        )
        object.__setattr__(
            self,
            "snapped_phi",
            _finite(self.snapped_phi, field="snapped_phi"),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "group_element": self.group_element,
            "input_phi": self.input_phi,
            "latent_coordinate": self.latent_coordinate,
            "map_sha256": self.map_sha256,
            "schema": SNAPPED_LATENT_SCHEMA,
            "snapped_phi": self.snapped_phi,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())


@dataclass(frozen=True, slots=True)
class CrystallizedMapStructure:
    """Parsed map metadata with no executable snap or composition methods."""

    family: str
    fit_receipt_sha256: str
    family_fit_sha256: str
    alpha: float
    gamma: float
    deployment_min: int
    deployment_max: int
    cyclic_order: int | None = None
    phase_centers: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.family not in MAP_FAMILIES:
            raise ValueError("unknown crystal family")
        for field in ("fit_receipt_sha256", "family_fit_sha256"):
            object.__setattr__(
                self,
                field,
                require_sha256(getattr(self, field), field=field),
            )
        alpha = _finite(self.alpha, field="alpha")
        if alpha == 0.0:
            raise ValueError("crystal alpha cannot be zero")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "gamma", _finite(self.gamma, field="gamma"))
        lower = _integer(
            self.deployment_min,
            field="deployment_min",
            minimum=-MAX_DEPLOYMENT_ABS,
            maximum=MAX_DEPLOYMENT_ABS,
        )
        upper = _integer(
            self.deployment_max,
            field="deployment_max",
            minimum=-MAX_DEPLOYMENT_ABS,
            maximum=MAX_DEPLOYMENT_ABS,
        )
        if lower > upper:
            raise ValueError("deployment range is empty")
        object.__setattr__(self, "deployment_min", lower)
        object.__setattr__(self, "deployment_max", upper)
        centers = tuple(_finite(value, field="phase_center") for value in self.phase_centers)
        object.__setattr__(self, "phase_centers", centers)
        if self.family == MULTIPLICATIVE and lower < 1:
            raise ValueError("multiplicative deployment range must be positive")
        if self.family == CYCLIC:
            order = _integer(
                self.cyclic_order,
                field="cyclic_order",
                minimum=2,
                maximum=MAX_CYCLIC_ORDER,
            )
            if (lower, upper) != (0, order - 1):
                raise ValueError("cyclic deployment range must be its exact residues")
            if len(centers) != order:
                raise ValueError("cyclic crystal lacks phase centres")
            winding = int(round(alpha))
            if (
                abs(alpha - winding) > 1e-12
                or not 1 <= winding < order
                or math.gcd(winding, order) != 1
            ):
                raise ValueError("cyclic alpha must be an invertible winding")
            if not 0.0 <= self.gamma < order:
                raise ValueError("cyclic gamma must be a canonical phase")
            if any(not 0.0 <= value < order for value in centers):
                raise ValueError("phase centres must lie in the canonical circle")
            object.__setattr__(self, "cyclic_order", order)
        elif self.cyclic_order is not None or centers:
            raise ValueError("non-cyclic crystal carries cyclic data")

    def to_body(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "cyclic_order": self.cyclic_order,
            "deployment_max": self.deployment_max,
            "deployment_min": self.deployment_min,
            "family": self.family,
            "family_fit_sha256": self.family_fit_sha256,
            "fit_receipt_sha256": self.fit_receipt_sha256,
            "gamma": self.gamma,
            "phase_centers": list(self.phase_centers),
        }

    @property
    def sha256(self) -> str:
        return _digest(_seal(CRYSTALLIZED_MAP_SCHEMA, self.to_body()))

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(_seal(CRYSTALLIZED_MAP_SCHEMA, self.to_body()))


def _parse_crystallized_map_structure(data: bytes) -> CrystallizedMapStructure:
    body = _open_seal(_strict_decode(data), schema=CRYSTALLIZED_MAP_SCHEMA)
    _require_keys(
        body,
        {
            "alpha",
            "cyclic_order",
            "deployment_max",
            "deployment_min",
            "family",
            "family_fit_sha256",
            "fit_receipt_sha256",
            "gamma",
            "phase_centers",
        },
        field="crystallized map",
    )
    if not isinstance(body["phase_centers"], list):
        raise AlgebraicCrystalIntegrityError("phase_centers must be a list")
    values = dict(body)
    values["phase_centers"] = tuple(values["phase_centers"])
    try:
        return CrystallizedMapStructure(**values)
    except (TypeError, ValueError) as exc:
        raise AlgebraicCrystalIntegrityError(
            "crystallized-map structure is invalid"
        ) from exc


@dataclass(frozen=True, slots=True)
class CrystallizedMap:
    """Policy-free exact executable map installed after admission."""

    family: str
    fit_receipt_sha256: str
    family_fit_sha256: str
    alpha: float
    gamma: float
    deployment_min: int
    deployment_max: int
    cyclic_order: int | None = None
    phase_centers: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        structure = CrystallizedMapStructure(
            family=self.family,
            fit_receipt_sha256=self.fit_receipt_sha256,
            family_fit_sha256=self.family_fit_sha256,
            alpha=self.alpha,
            gamma=self.gamma,
            deployment_min=self.deployment_min,
            deployment_max=self.deployment_max,
            cyclic_order=self.cyclic_order,
            phase_centers=self.phase_centers,
        )
        for field in (
            "family",
            "fit_receipt_sha256",
            "family_fit_sha256",
            "alpha",
            "gamma",
            "deployment_min",
            "deployment_max",
            "cyclic_order",
            "phase_centers",
        ):
            object.__setattr__(self, field, getattr(structure, field))

    @classmethod
    def from_fit(
        cls,
        receipt: CrystallizationFitReceipt,
        *,
        deployment_min: int,
        deployment_max: int,
    ) -> "CrystallizedMap":
        if not isinstance(receipt, CrystallizationFitReceipt):
            raise TypeError("receipt must be CrystallizationFitReceipt")
        verified = CrystallizationFitReceipt.from_bytes(receipt.to_bytes())
        if verified != receipt:
            raise AlgebraicCrystalIntegrityError(
                "fit receipt is not the canonical recomputed admission artifact"
            )
        receipt = verified
        if receipt.outcome != ACCEPTED:
            raise ValueError("only an accepted bank receipt can crystallize")
        fit = receipt.best_fit
        if fit is None or not fit.valid or fit.alpha is None or fit.gamma is None:
            raise ValueError("accepted receipt has no executable best fit")
        if fit.family == CYCLIC:
            assert fit.cyclic_order is not None
            if (deployment_min, deployment_max) != (0, fit.cyclic_order - 1):
                raise ValueError("cyclic range is fixed by Z_n")
        return cls(
            family=fit.family,
            fit_receipt_sha256=receipt.sha256,
            family_fit_sha256=fit.sha256,
            alpha=fit.alpha,
            gamma=fit.gamma,
            deployment_min=deployment_min,
            deployment_max=deployment_max,
            cyclic_order=fit.cyclic_order,
            phase_centers=fit.phase_centers,
        )

    def to_body(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "cyclic_order": self.cyclic_order,
            "deployment_max": self.deployment_max,
            "deployment_min": self.deployment_min,
            "family": self.family,
            "family_fit_sha256": self.family_fit_sha256,
            "fit_receipt_sha256": self.fit_receipt_sha256,
            "gamma": self.gamma,
            "phase_centers": list(self.phase_centers),
        }

    @property
    def sha256(self) -> str:
        return _digest(_seal(CRYSTALLIZED_MAP_SCHEMA, self.to_body()))

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(_seal(CRYSTALLIZED_MAP_SCHEMA, self.to_body()))

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        fit_receipt: CrystallizationFitReceipt | None = None,
        fit_receipt_resolver: (
            Callable[[str], CrystallizationFitReceipt] | None
        ) = None,
    ) -> "CrystallizedMap":
        structure = cls.parse_structure_only(data)
        if (fit_receipt is None) == (fit_receipt_resolver is None):
            raise AlgebraicCrystalIntegrityError(
                "executable map loading requires exactly one canonical fit "
                "receipt or trusted resolver"
            )
        receipt = fit_receipt
        if fit_receipt_resolver is not None:
            try:
                receipt = fit_receipt_resolver(structure.fit_receipt_sha256)
            except Exception as exc:
                raise AlgebraicCrystalIntegrityError(
                    "trusted fit-receipt resolver failed"
                ) from exc
        if not isinstance(receipt, CrystallizationFitReceipt):
            raise AlgebraicCrystalIntegrityError(
                "fit-receipt authority returned an unsupported artifact"
            )
        if receipt.sha256 != structure.fit_receipt_sha256:
            raise AlgebraicCrystalIntegrityError(
                "resolved fit receipt does not match the map binding"
            )
        expected = cls.from_fit(
            receipt,
            deployment_min=structure.deployment_min,
            deployment_max=structure.deployment_max,
        )
        if expected.to_body() != structure.to_body():
            raise AlgebraicCrystalIntegrityError(
                "crystal does not match its accepted map receipt"
            )
        return expected

    @classmethod
    def parse_structure_only(cls, data: bytes) -> CrystallizedMapStructure:
        """Parse self-sealed metadata without granting execution authority."""

        return _parse_crystallized_map_structure(data)

    def verify_against(self, receipt: CrystallizationFitReceipt) -> None:
        expected = CrystallizedMap.from_fit(
            receipt,
            deployment_min=self.deployment_min,
            deployment_max=self.deployment_max,
        )
        if expected != self:
            raise AlgebraicCrystalIntegrityError(
                "crystal does not match its accepted map receipt"
            )

    def inverse(self, learned_phi: float) -> float:
        """Decode a coordinate; cyclic maps return the nearest exact residue."""

        phi = _finite(learned_phi, field="learned_phi")
        normalized = (phi - self.gamma) / self.alpha
        if self.family == ADDITIVE:
            return normalized
        if self.family == MULTIPLICATIVE:
            try:
                value = math.exp(normalized)
            except OverflowError as exc:
                raise ValueError("learned phi lies outside finite log space") from exc
            if not math.isfinite(value):
                raise ValueError("learned phi lies outside finite log space")
            return value
        return float(self._nearest_cyclic_residue(phi))

    def learned_phi_of(self, group_element: int) -> float:
        element = _integer(group_element, field="group_element")
        if self.family == ADDITIVE:
            return self.alpha * element + self.gamma
        if self.family == MULTIPLICATIVE:
            if element <= 0:
                raise ValueError("multiplicative element must be positive")
            return self.alpha * math.log(element) + self.gamma
        assert self.cyclic_order is not None
        return self.phase_centers[element % self.cyclic_order]

    def _additive_element(self, normalized: float) -> int:
        lower_boundary = self.deployment_min - 0.5
        upper_boundary = self.deployment_max + 0.5
        if normalized < lower_boundary or normalized >= upper_boundary:
            raise ValueError("learned phi lies outside the deployment lattice")
        return min(
            self.deployment_max,
            max(self.deployment_min, math.floor(normalized + 0.5)),
        )

    def _multiplicative_element(self, normalized_log: float) -> int:
        low = self.deployment_min
        high = self.deployment_max
        if low > 1:
            low_boundary = 0.5 * (math.log(low - 1) + math.log(low))
            if normalized_log < low_boundary:
                raise ValueError("learned phi lies below the deployment log lattice")
        upper_boundary = 0.5 * (math.log(high) + math.log(high + 1))
        if normalized_log >= upper_boundary:
            raise ValueError("learned phi lies above the deployment log lattice")
        if normalized_log > math.log(high) + 1.0:
            raise ValueError("learned phi lies above the deployment log lattice")
        continuous = math.exp(normalized_log)
        candidates = {
            min(high, max(low, math.floor(continuous))),
            min(high, max(low, math.ceil(continuous))),
        }
        return min(candidates, key=lambda item: (abs(math.log(item) - normalized_log), item))

    @staticmethod
    def _circular_distance(left: float, right: float, order: int) -> float:
        raw = abs((left - right) % order)
        return min(raw, order - raw)

    def _nearest_cyclic_residue(self, phi: float) -> int:
        assert self.cyclic_order is not None
        phase = phi % self.cyclic_order
        return min(
            range(self.cyclic_order),
            key=lambda residue: (
                self._circular_distance(
                    phase,
                    self.phase_centers[residue],
                    self.cyclic_order,
                ),
                residue,
            ),
        )

    def snap(self, learned_phi: float) -> SnappedLatent:
        phi = _finite(learned_phi, field="learned_phi")
        normalized = (phi - self.gamma) / self.alpha
        if self.family == ADDITIVE:
            element = self._additive_element(normalized)
            latent = float(element)
        elif self.family == MULTIPLICATIVE:
            element = self._multiplicative_element(normalized)
            latent = math.log(element)
        else:
            element = self._nearest_cyclic_residue(phi)
            latent = float(element)
        return SnappedLatent(
            map_sha256=self.sha256,
            family=self.family,
            input_phi=phi,
            group_element=element,
            latent_coordinate=latent,
            snapped_phi=self.learned_phi_of(element),
        )

    def verify_snapped(self, value: SnappedLatent) -> None:
        if not isinstance(value, SnappedLatent) or self.snap(value.input_phi) != value:
            raise AlgebraicCrystalIntegrityError(
                "snapped latent does not match the exact crystal map"
            )


@dataclass(frozen=True, slots=True)
class GroupExecutionStep:
    index: int
    snapped_sha256: str
    sign: int
    state_after: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "index",
            _integer(self.index, field="step index", minimum=0),
        )
        object.__setattr__(
            self,
            "snapped_sha256",
            require_sha256(self.snapped_sha256, field="snapped_sha256"),
        )
        if self.sign not in (-1, 1):
            raise ValueError("group-operation sign must be -1 or +1")
        object.__setattr__(
            self,
            "state_after",
            _integer(self.state_after, field="state_after"),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "sign": self.sign,
            "snapped_sha256": self.snapped_sha256,
            "state_after": self.state_after,
        }


@dataclass(frozen=True, slots=True)
class GroupExecutionReceipt:
    map_sha256: str
    fit_receipt_sha256: str
    family: str
    inputs: tuple[SnappedLatent, ...]
    steps: tuple[GroupExecutionStep, ...]
    initial_state: int
    result: int
    live_steps: int
    state_size: int = 1

    def __post_init__(self) -> None:
        for field in ("map_sha256", "fit_receipt_sha256"):
            object.__setattr__(
                self,
                field,
                require_sha256(getattr(self, field), field=field),
            )
        if self.family not in MAP_FAMILIES:
            raise ValueError("unknown execution family")
        inputs = tuple(self.inputs)
        steps = tuple(self.steps)
        if not 1 <= len(inputs) <= MAX_EXECUTION_STEPS or len(steps) != len(inputs):
            raise ValueError("execution input/step count is invalid")
        if not all(isinstance(item, SnappedLatent) for item in inputs):
            raise TypeError("inputs must be snapped latents")
        if not all(isinstance(item, GroupExecutionStep) for item in steps):
            raise TypeError("steps must be GroupExecutionStep values")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "initial_state", _integer(self.initial_state, field="initial_state"))
        object.__setattr__(self, "result", _integer(self.result, field="result"))
        live = _integer(self.live_steps, field="live_steps", minimum=1, maximum=MAX_EXECUTION_STEPS)
        if live != len(inputs):
            raise ValueError("live_steps must equal consumed input count")
        object.__setattr__(self, "live_steps", live)
        if self.state_size != 1:
            raise ValueError("exact group accumulator must use one live state")

    def to_body(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "fit_receipt_sha256": self.fit_receipt_sha256,
            "initial_state": self.initial_state,
            "inputs": [item.to_record() for item in self.inputs],
            "live_steps": self.live_steps,
            "map_sha256": self.map_sha256,
            "result": self.result,
            "state_size": self.state_size,
            "steps": [item.to_record() for item in self.steps],
        }

    @property
    def sha256(self) -> str:
        return _digest(_seal(EXECUTION_RECEIPT_SCHEMA, self.to_body()))

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(_seal(EXECUTION_RECEIPT_SCHEMA, self.to_body()))

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        crystal: CrystallizedMap,
    ) -> "GroupExecutionReceipt":
        body = _open_seal(_strict_decode(data), schema=EXECUTION_RECEIPT_SCHEMA)
        _require_keys(
            body,
            {
                "family",
                "fit_receipt_sha256",
                "initial_state",
                "inputs",
                "live_steps",
                "map_sha256",
                "result",
                "state_size",
                "steps",
            },
            field="execution receipt",
        )
        if not isinstance(body["inputs"], list) or not isinstance(body["steps"], list):
            raise AlgebraicCrystalIntegrityError("execution arrays must be lists")
        inputs = []
        for value in body["inputs"]:
            if not isinstance(value, dict):
                raise AlgebraicCrystalIntegrityError("snapped input must be an object")
            _require_keys(
                value,
                {
                    "family",
                    "group_element",
                    "input_phi",
                    "latent_coordinate",
                    "map_sha256",
                    "schema",
                    "snapped_phi",
                },
                field="snapped input",
            )
            if value["schema"] != SNAPPED_LATENT_SCHEMA:
                raise AlgebraicCrystalIntegrityError("unsupported snapped input")
            values = dict(value)
            del values["schema"]
            inputs.append(SnappedLatent(**values))
        steps = []
        for value in body["steps"]:
            if not isinstance(value, dict):
                raise AlgebraicCrystalIntegrityError("execution step must be an object")
            _require_keys(
                value,
                {"index", "sign", "snapped_sha256", "state_after"},
                field="execution step",
            )
            steps.append(GroupExecutionStep(**value))
        values = dict(body)
        values["inputs"] = tuple(inputs)
        values["steps"] = tuple(steps)
        result = cls(**values)
        ExactGroupAccumulator(crystal).verify(result)
        return result


class ExactGroupAccumulator:
    """One-word live-state accumulator over an admitted exact group map."""

    def __init__(self, crystal: CrystallizedMap) -> None:
        if not isinstance(crystal, CrystallizedMap):
            raise TypeError("crystal must be CrystallizedMap")
        self._crystal = crystal

    @property
    def crystal(self) -> CrystallizedMap:
        return self._crystal

    def execute(
        self,
        values: Sequence[SnappedLatent],
        *,
        signs: Sequence[int] | None = None,
        initial_state: int | None = None,
    ) -> GroupExecutionReceipt:
        inputs = tuple(values)
        if not 1 <= len(inputs) <= MAX_EXECUTION_STEPS:
            raise ValueError("input count is outside the execution bound")
        if signs is None:
            exact_signs = (1,) * len(inputs)
        else:
            exact_signs = tuple(_integer(value, field="sign") for value in signs)
            if len(exact_signs) != len(inputs) or any(value not in (-1, 1) for value in exact_signs):
                raise ValueError("signs must contain one -1/+1 value per input")
        for value in inputs:
            self.crystal.verify_snapped(value)

        identity = 1 if self.crystal.family == MULTIPLICATIVE else 0
        state = identity if initial_state is None else _integer(initial_state, field="initial_state")
        if self.crystal.family == MULTIPLICATIVE and state <= 0:
            raise ValueError("multiplicative initial state must be positive")
        if self.crystal.family == CYCLIC:
            assert self.crystal.cyclic_order is not None
            state %= self.crystal.cyclic_order
        first_state = state
        steps: list[GroupExecutionStep] = []
        for index, (value, sign) in enumerate(zip(inputs, exact_signs, strict=True)):
            if self.crystal.family == ADDITIVE:
                state += sign * value.group_element
            elif self.crystal.family == MULTIPLICATIVE:
                if sign != 1:
                    raise ValueError("integer multiplicative accumulator has no fractional inverse")
                state *= value.group_element
            else:
                assert self.crystal.cyclic_order is not None
                state = (state + sign * value.group_element) % self.crystal.cyclic_order
            steps.append(
                GroupExecutionStep(
                    index=index,
                    snapped_sha256=value.sha256,
                    sign=sign,
                    state_after=state,
                )
            )
        return GroupExecutionReceipt(
            map_sha256=self.crystal.sha256,
            fit_receipt_sha256=self.crystal.fit_receipt_sha256,
            family=self.crystal.family,
            inputs=inputs,
            steps=tuple(steps),
            initial_state=first_state,
            result=state,
            live_steps=len(inputs),
        )

    def snap_and_execute(
        self,
        learned_phis: Sequence[float],
        *,
        signs: Sequence[int] | None = None,
        initial_state: int | None = None,
    ) -> GroupExecutionReceipt:
        return self.execute(
            tuple(self.crystal.snap(value) for value in learned_phis),
            signs=signs,
            initial_state=initial_state,
        )

    def verify(self, receipt: GroupExecutionReceipt) -> None:
        if not isinstance(receipt, GroupExecutionReceipt):
            raise TypeError("receipt must be GroupExecutionReceipt")
        if (
            receipt.map_sha256 != self.crystal.sha256
            or receipt.fit_receipt_sha256 != self.crystal.fit_receipt_sha256
            or receipt.family != self.crystal.family
        ):
            raise AlgebraicCrystalIntegrityError("execution uses another exact map")
        signs = tuple(step.sign for step in receipt.steps)
        expected = self.execute(
            receipt.inputs,
            signs=signs,
            initial_state=receipt.initial_state,
        )
        if expected != receipt:
            raise AlgebraicCrystalIntegrityError(
                "execution receipt does not match exact live composition"
            )


def crystallize_map(
    receipt: CrystallizationFitReceipt,
    *,
    deployment_min: int,
    deployment_max: int,
) -> CrystallizedMap:
    """Install a policy-free exact map from an accepted intake receipt."""

    return CrystallizedMap.from_fit(
        receipt,
        deployment_min=deployment_min,
        deployment_max=deployment_max,
    )


def parse_structure_only(data: bytes) -> CrystallizedMapStructure:
    """Inspect serialized map metadata without creating an executable map."""

    return CrystallizedMap.parse_structure_only(data)


__all__ = [
    "ACCEPTED",
    "ADDITIVE",
    "AMBIGUOUS",
    "AlgebraicCrystalIntegrityError",
    "CRYSTALLIZED_MAP_SCHEMA",
    "CYCLIC",
    "CrystalAdmissionPolicy",
    "CrystallizationFitReceipt",
    "CrystallizedMap",
    "CrystallizedMapStructure",
    "ExactGroupAccumulator",
    "GroupExecutionReceipt",
    "GroupExecutionStep",
    "InvariantSupport",
    "MAP_FAMILIES",
    "MULTIPLICATIVE",
    "MapFamilyFit",
    "REJECTED",
    "SnappedLatent",
    "crystallize_map",
    "fit_algebraic_map",
    "parse_structure_only",
]
