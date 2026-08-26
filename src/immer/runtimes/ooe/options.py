"""Verified hierarchical options for the OoE Markov runtime.

An option is a reusable sequence of primitive world-model actions.  This
module deliberately keeps discovery, execution, allocation and planning in
the finite-state kernel domain: composing an option is ordinary left-to-right
matrix multiplication, never a learned approximation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import base64
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .contraction_ledger import ContractionLedger
from .identity import canonical_json_bytes, require_sha256
from .math_core import array_sha256, sinkhorn_project


FloatArray = NDArray[np.float64]

MAX_STATES = 4_096
MAX_ACTIONS = 1_024
MAX_OPTION_LENGTH = 64
MAX_TRAJECTORIES = 100_000
MAX_DISCOVERED_OPTIONS = 16_384
MAX_ALLOCATION_SIZE = 256
MAX_PAYLOAD_BYTES = 256 * 1024 * 1024
MAX_PLANNING_EXPANSIONS = 1_000_000

_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}")


class OptionIntegrityError(ValueError):
    """Raised when an option or allocation fails its bound receipt."""


def _integer(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        upper = "" if maximum is None else f" and at most {maximum}"
        raise ValueError(f"{name} must be at least {minimum}{upper}")
    return result


def _finite_positive(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical identifier")
    return value


def _identifiers(
    values: Iterable[object],
    *,
    name: str,
    minimum: int = 1,
    maximum: int,
) -> tuple[str, ...]:
    result = tuple(_identifier(value, name=name) for value in values)
    if not minimum <= len(result) <= maximum:
        raise ValueError(f"{name} count must lie in [{minimum}, {maximum}]")
    return result


def _hash_items(
    value: Mapping[str, str] | Iterable[tuple[str, str]],
    *,
    name: str,
) -> tuple[tuple[str, str], ...]:
    items = value.items() if isinstance(value, Mapping) else value
    normalized = tuple(
        sorted(
            (
                _identifier(key, name=f"{name} name"),
                require_sha256(digest, field=f"{name}[{key!r}]"),
            )
            for key, digest in items
        )
    )
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if len({key for key, _ in normalized}) != len(normalized):
        raise ValueError(f"{name} contains duplicate names")
    return normalized


def _kernel(
    value: ArrayLike,
    *,
    name: str,
    states: int | None = None,
) -> FloatArray:
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or raw.shape[0] < 2:
        raise ValueError(f"{name} must be a non-empty square matrix")
    size = int(raw.shape[0])
    if size > MAX_STATES:
        raise ValueError(f"{name} exceeds MAX_STATES={MAX_STATES}")
    if states is not None and size != states:
        raise ValueError(f"{name} state dimension mismatch")
    if raw.size * np.dtype(np.float64).itemsize > MAX_PAYLOAD_BYTES:
        raise ValueError(f"{name} exceeds MAX_PAYLOAD_BYTES")
    try:
        matrix = np.asarray(raw, dtype=np.float64, order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numbers") from exc
    if not np.isfinite(matrix).all() or np.any(matrix < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} must be row-stochastic")
    return np.ascontiguousarray(matrix)


def _matrix_bytes(value: FloatArray) -> bytes:
    return np.asarray(value, dtype="<f8", order="C").tobytes(order="C")


def _matrix_from_bytes(data: bytes, rows: int, columns: int) -> FloatArray:
    required = rows * columns * np.dtype("<f8").itemsize
    if required > MAX_PAYLOAD_BYTES or len(data) != required:
        raise OptionIntegrityError("matrix byte length does not match its shape")
    return np.frombuffer(data, dtype="<f8").reshape(rows, columns).copy()


def _decode_canonical(data: bytes, *, maximum: int = MAX_PAYLOAD_BYTES) -> Any:
    if not isinstance(data, bytes) or len(data) > maximum:
        raise OptionIntegrityError("payload is not bounded immutable bytes")
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OptionIntegrityError("payload is not valid JSON") from exc
    if canonical_json_bytes(value) != data:
        raise OptionIntegrityError("payload JSON is not canonical")
    return value


def _body_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


def compose_stochastic_kernels(kernels: Sequence[ArrayLike]) -> FloatArray:
    """Compose kernels in execution order without approximation.

    Row-vector semantics are used throughout OoE.  For actions ``(a, b, c)``
    the returned transition is therefore exactly the evaluated expression
    ``((K_a @ K_b) @ K_c)``.  No balancing, truncation or re-normalization is
    applied to the product.
    """

    if not 1 <= len(kernels) <= MAX_OPTION_LENGTH:
        raise ValueError(
            f"kernel count must lie in [1, {MAX_OPTION_LENGTH}]"
        )
    first = _kernel(kernels[0], name="kernels[0]")
    result = first.copy()
    for index, candidate in enumerate(kernels[1:], start=1):
        right = _kernel(candidate, name=f"kernels[{index}]", states=first.shape[0])
        # ``matmul`` dispatches into Accelerate on macOS, whose 16x16 kernel
        # can emit spurious overflow/divide warnings for finite deterministic
        # matrices.  The explicit bounded contraction is deterministic and
        # does not enter that backend path.
        result = np.ascontiguousarray(
            np.einsum("ij,jk->ik", result, right, optimize=False),
            dtype=np.float64,
        )
    if not np.isfinite(result).all() or np.any(result < -1e-15):
        raise ArithmeticError("stochastic-kernel composition became invalid")
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-11, rtol=1e-11):
        raise ArithmeticError("stochastic-kernel composition lost probability mass")
    return result


@dataclass(frozen=True, slots=True)
class VerifiedTrajectory:
    """One successful trajectory bound to an external verification receipt."""

    states: tuple[int, ...]
    actions: tuple[str, ...]
    source_world_model_sha256: str
    graph_revision_sha256: str
    verifier_hashes: tuple[tuple[str, str], ...]
    verification_receipt_sha256: str
    outcome_sha256: str
    verified_success: bool = True

    FORMAT = "immer-ooe-verified-successful-trajectory/v1"

    def __post_init__(self) -> None:
        actions = _identifiers(
            self.actions,
            name="trajectory action",
            maximum=MAX_OPTION_LENGTH,
        )
        states = tuple(
            _integer(value, name="trajectory state", maximum=MAX_STATES - 1)
            for value in self.states
        )
        if len(states) != len(actions) + 1:
            raise ValueError("trajectory states must bracket every action")
        if self.verified_success is not True:
            raise ValueError("only verifier-confirmed successful trajectories qualify")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(
            self,
            "source_world_model_sha256",
            require_sha256(
                self.source_world_model_sha256,
                field="source_world_model_sha256",
            ),
        )
        object.__setattr__(
            self,
            "graph_revision_sha256",
            require_sha256(
                self.graph_revision_sha256,
                field="graph_revision_sha256",
            ),
        )
        object.__setattr__(
            self,
            "verifier_hashes",
            _hash_items(self.verifier_hashes, name="verifier_hashes"),
        )
        object.__setattr__(
            self,
            "verification_receipt_sha256",
            require_sha256(
                self.verification_receipt_sha256,
                field="verification_receipt_sha256",
            ),
        )
        object.__setattr__(
            self,
            "outcome_sha256",
            require_sha256(self.outcome_sha256, field="outcome_sha256"),
        )

    @classmethod
    def create(
        cls,
        *,
        states: Sequence[int],
        actions: Sequence[str],
        source_world_model_sha256: str,
        graph_revision_sha256: str,
        verifier_hashes: Mapping[str, str],
        verification_receipt_sha256: str,
        outcome_sha256: str,
    ) -> "VerifiedTrajectory":
        return cls(
            states=tuple(states),
            actions=tuple(actions),
            source_world_model_sha256=source_world_model_sha256,
            graph_revision_sha256=graph_revision_sha256,
            verifier_hashes=_hash_items(verifier_hashes, name="verifier_hashes"),
            verification_receipt_sha256=verification_receipt_sha256,
            outcome_sha256=outcome_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": list(self.actions),
            "format": self.FORMAT,
            "graph_revision_sha256": self.graph_revision_sha256,
            "outcome_sha256": self.outcome_sha256,
            "source_world_model_sha256": self.source_world_model_sha256,
            "states": list(self.states),
            "verification_receipt_sha256": self.verification_receipt_sha256,
            "verified_success": self.verified_success,
            "verifier_hashes": dict(self.verifier_hashes),
        }

    @property
    def sha256(self) -> str:
        return _body_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class OptionIdentity:
    """Execution identity of one macro-option and all its primitive kernels."""

    action_sequence: tuple[str, ...]
    primitive_kernel_sha256s: tuple[str, ...]
    source_world_model_sha256: str
    graph_revision_sha256: str
    verifier_hashes: tuple[tuple[str, str], ...]

    FORMAT = "immer-ooe-option-identity/v1"

    def __post_init__(self) -> None:
        actions = _identifiers(
            self.action_sequence,
            name="option action",
            minimum=2,
            maximum=MAX_OPTION_LENGTH,
        )
        kernel_hashes = tuple(
            require_sha256(value, field="primitive_kernel_sha256")
            for value in self.primitive_kernel_sha256s
        )
        if len(kernel_hashes) != len(actions):
            raise ValueError("every option action must bind one primitive kernel")
        object.__setattr__(self, "action_sequence", actions)
        object.__setattr__(self, "primitive_kernel_sha256s", kernel_hashes)
        object.__setattr__(
            self,
            "source_world_model_sha256",
            require_sha256(
                self.source_world_model_sha256,
                field="source_world_model_sha256",
            ),
        )
        object.__setattr__(
            self,
            "graph_revision_sha256",
            require_sha256(
                self.graph_revision_sha256,
                field="graph_revision_sha256",
            ),
        )
        object.__setattr__(
            self,
            "verifier_hashes",
            _hash_items(self.verifier_hashes, name="verifier_hashes"),
        )

    @classmethod
    def create(
        cls,
        *,
        action_sequence: Sequence[str],
        action_kernels: Mapping[str, ArrayLike],
        source_world_model_sha256: str,
        graph_revision_sha256: str,
        verifier_hashes: Mapping[str, str],
    ) -> "OptionIdentity":
        actions = tuple(action_sequence)
        try:
            hashes = tuple(array_sha256(action_kernels[action]) for action in actions)
        except KeyError as exc:
            raise KeyError(f"missing primitive action kernel: {exc.args[0]}") from exc
        return cls(
            action_sequence=actions,
            primitive_kernel_sha256s=hashes,
            source_world_model_sha256=source_world_model_sha256,
            graph_revision_sha256=graph_revision_sha256,
            verifier_hashes=_hash_items(verifier_hashes, name="verifier_hashes"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_sequence": list(self.action_sequence),
            "format": self.FORMAT,
            "graph_revision_sha256": self.graph_revision_sha256,
            "primitive_kernel_sha256s": list(self.primitive_kernel_sha256s),
            "source_world_model_sha256": self.source_world_model_sha256,
            "verifier_hashes": dict(self.verifier_hashes),
        }

    @classmethod
    def from_dict(cls, value: object) -> "OptionIdentity":
        if not isinstance(value, dict) or value.get("format") != cls.FORMAT:
            raise OptionIntegrityError("unsupported option identity")
        expected = {
            "action_sequence",
            "format",
            "graph_revision_sha256",
            "primitive_kernel_sha256s",
            "source_world_model_sha256",
            "verifier_hashes",
        }
        if set(value) != expected or not isinstance(value["verifier_hashes"], dict):
            raise OptionIntegrityError("option identity fields do not match schema")
        try:
            return cls(
                action_sequence=tuple(value["action_sequence"]),
                primitive_kernel_sha256s=tuple(value["primitive_kernel_sha256s"]),
                source_world_model_sha256=value["source_world_model_sha256"],
                graph_revision_sha256=value["graph_revision_sha256"],
                verifier_hashes=_hash_items(
                    value["verifier_hashes"], name="verifier_hashes"
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise OptionIntegrityError("invalid option identity") from exc

    @property
    def sha256(self) -> str:
        return _body_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class MacroOption:
    """Immutable exact option kernel with its complete evidence lineage."""

    identity: OptionIdentity
    kernel_rows: int
    kernel_columns: int
    kernel_bytes: bytes
    source_trajectory_sha256s: tuple[str, ...]

    FORMAT = "immer-ooe-macro-option/v1"
    ENVELOPE_FORMAT = "immer-ooe-macro-option-envelope/v1"

    def __post_init__(self) -> None:
        if not isinstance(self.identity, OptionIdentity):
            raise TypeError("identity must be an OptionIdentity")
        rows = _integer(
            self.kernel_rows,
            name="kernel_rows",
            minimum=2,
            maximum=MAX_STATES,
        )
        columns = _integer(
            self.kernel_columns,
            name="kernel_columns",
            minimum=2,
            maximum=MAX_STATES,
        )
        if rows != columns:
            raise ValueError("option kernel must be square")
        if not isinstance(self.kernel_bytes, bytes):
            raise TypeError("kernel_bytes must be immutable bytes")
        _kernel(
            _matrix_from_bytes(self.kernel_bytes, rows, columns),
            name="option kernel",
        )
        sources = tuple(
            sorted(
                require_sha256(value, field="source_trajectory_sha256")
                for value in self.source_trajectory_sha256s
            )
        )
        if not sources or len(sources) > MAX_TRAJECTORIES:
            raise ValueError("option must bind a bounded non-empty source set")
        if len(set(sources)) != len(sources):
            raise ValueError("source trajectory hashes must be unique")
        object.__setattr__(self, "kernel_rows", rows)
        object.__setattr__(self, "kernel_columns", columns)
        object.__setattr__(self, "source_trajectory_sha256s", sources)

    @classmethod
    def from_identity(
        cls,
        identity: OptionIdentity,
        *,
        action_kernels: Mapping[str, ArrayLike],
        source_trajectory_sha256s: Iterable[str],
    ) -> "MacroOption":
        kernels: list[FloatArray] = []
        for action, expected in zip(
            identity.action_sequence,
            identity.primitive_kernel_sha256s,
            strict=True,
        ):
            try:
                kernel = _kernel(action_kernels[action], name=f"action_kernels[{action!r}]")
            except KeyError as exc:
                raise KeyError(f"missing primitive action kernel: {action}") from exc
            if array_sha256(kernel) != expected:
                raise OptionIntegrityError(
                    f"primitive action kernel changed for {action!r}"
                )
            kernels.append(kernel)
        composed = compose_stochastic_kernels(kernels)
        return cls(
            identity=identity,
            kernel_rows=int(composed.shape[0]),
            kernel_columns=int(composed.shape[1]),
            kernel_bytes=_matrix_bytes(composed),
            source_trajectory_sha256s=tuple(source_trajectory_sha256s),
        )

    @property
    def support(self) -> int:
        return len(self.source_trajectory_sha256s)

    @property
    def kernel_sha256(self) -> str:
        return array_sha256(self.restore_kernel())

    def restore_kernel(self) -> FloatArray:
        return _matrix_from_bytes(
            self.kernel_bytes,
            self.kernel_rows,
            self.kernel_columns,
        )

    def verify_against(self, action_kernels: Mapping[str, ArrayLike]) -> None:
        rebound = MacroOption.from_identity(
            self.identity,
            action_kernels=action_kernels,
            source_trajectory_sha256s=self.source_trajectory_sha256s,
        )
        if rebound.kernel_bytes != self.kernel_bytes:
            raise OptionIntegrityError("option kernel is not its exact action composition")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "identity": self.identity.to_dict(),
            "kernel": {
                "columns": self.kernel_columns,
                "data_base64": base64.b64encode(self.kernel_bytes).decode("ascii"),
                "rows": self.kernel_rows,
                "sha256": self.kernel_sha256,
                "storage": "float64-le",
            },
            "source_trajectory_sha256s": list(self.source_trajectory_sha256s),
            "support": self.support,
        }

    @property
    def sha256(self) -> str:
        return _body_sha256(self.to_dict())

    def to_bytes(self) -> bytes:
        body = self.to_dict()
        return canonical_json_bytes(
            {
                "body": body,
                "body_sha256": _body_sha256(body),
                "format": self.ENVELOPE_FORMAT,
            }
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "MacroOption":
        envelope = _decode_canonical(data)
        if (
            not isinstance(envelope, dict)
            or set(envelope) != {"body", "body_sha256", "format"}
            or envelope.get("format") != cls.ENVELOPE_FORMAT
            or not isinstance(envelope.get("body"), dict)
        ):
            raise OptionIntegrityError("invalid macro-option envelope")
        body = envelope["body"]
        if _body_sha256(body) != envelope.get("body_sha256"):
            raise OptionIntegrityError("macro-option payload hash mismatch")
        expected = {
            "format",
            "identity",
            "kernel",
            "source_trajectory_sha256s",
            "support",
        }
        if set(body) != expected or body.get("format") != cls.FORMAT:
            raise OptionIntegrityError("macro-option body does not match schema")
        descriptor = body.get("kernel")
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "columns",
            "data_base64",
            "rows",
            "sha256",
            "storage",
        }:
            raise OptionIntegrityError("invalid macro-option kernel descriptor")
        if descriptor.get("storage") != "float64-le":
            raise OptionIntegrityError("unsupported macro-option kernel storage")
        encoded = descriptor.get("data_base64")
        if not isinstance(encoded, str) or len(encoded) > 2 * MAX_PAYLOAD_BYTES:
            raise OptionIntegrityError("invalid macro-option kernel encoding")
        try:
            kernel_bytes = base64.b64decode(encoded, validate=True)
            result = cls(
                identity=OptionIdentity.from_dict(body["identity"]),
                kernel_rows=descriptor["rows"],
                kernel_columns=descriptor["columns"],
                kernel_bytes=kernel_bytes,
                source_trajectory_sha256s=tuple(
                    body["source_trajectory_sha256s"]
                ),
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise OptionIntegrityError("invalid macro-option payload") from exc
        if body.get("support") != result.support:
            raise OptionIntegrityError("macro-option support count mismatch")
        if descriptor.get("sha256") != result.kernel_sha256:
            raise OptionIntegrityError("macro-option kernel hash mismatch")
        if result.to_dict() != body:
            raise OptionIntegrityError("macro-option failed canonical roundtrip")
        return result


@dataclass(frozen=True, slots=True)
class OptionDiscoveryReceipt:
    source_world_model_sha256: str
    graph_revision_sha256: str
    verifier_hashes: tuple[tuple[str, str], ...]
    trajectory_sha256s: tuple[str, ...]
    action_kernel_set_sha256: str
    min_support: int
    min_option_length: int
    max_option_length: int
    candidate_count: int
    discovered_option_sha256s: tuple[str, ...]

    FORMAT = "immer-ooe-option-discovery-receipt/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_kernel_set_sha256": self.action_kernel_set_sha256,
            "candidate_count": self.candidate_count,
            "discovered_option_sha256s": list(self.discovered_option_sha256s),
            "format": self.FORMAT,
            "graph_revision_sha256": self.graph_revision_sha256,
            "max_option_length": self.max_option_length,
            "min_option_length": self.min_option_length,
            "min_support": self.min_support,
            "source_world_model_sha256": self.source_world_model_sha256,
            "trajectory_sha256s": list(self.trajectory_sha256s),
            "verifier_hashes": dict(self.verifier_hashes),
        }

    @property
    def sha256(self) -> str:
        return _body_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class OptionDiscoveryResult:
    options: tuple[MacroOption, ...]
    receipt: OptionDiscoveryReceipt


def _action_kernel_set_sha256(action_kernels: Mapping[str, ArrayLike]) -> str:
    body = [
        {
            "action": _identifier(action, name="action kernel name"),
            "kernel_sha256": array_sha256(_kernel(kernel, name=f"kernel {action!r}")),
        }
        for action, kernel in sorted(action_kernels.items())
    ]
    if not body or len(body) > MAX_ACTIONS:
        raise ValueError(f"action kernel count must lie in [1, {MAX_ACTIONS}]")
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def discover_macro_options(
    trajectories: Sequence[VerifiedTrajectory],
    action_kernels: Mapping[str, ArrayLike],
    *,
    min_support: int = 2,
    min_option_length: int = 2,
    max_option_length: int = 8,
    max_options: int = 1_024,
) -> OptionDiscoveryResult:
    """Mine repeated contiguous subsequences from verified successes only."""

    if not 1 <= len(trajectories) <= MAX_TRAJECTORIES:
        raise ValueError(
            f"trajectory count must lie in [1, {MAX_TRAJECTORIES}]"
        )
    support_floor = _integer(
        min_support,
        name="min_support",
        minimum=1,
        maximum=len(trajectories),
    )
    lower = _integer(
        min_option_length,
        name="min_option_length",
        minimum=2,
        maximum=MAX_OPTION_LENGTH,
    )
    upper = _integer(
        max_option_length,
        name="max_option_length",
        minimum=lower,
        maximum=MAX_OPTION_LENGTH,
    )
    limit = _integer(
        max_options,
        name="max_options",
        minimum=1,
        maximum=MAX_DISCOVERED_OPTIONS,
    )
    if not all(isinstance(value, VerifiedTrajectory) for value in trajectories):
        raise TypeError("trajectories must contain VerifiedTrajectory values")
    first = trajectories[0]
    contract = (
        first.source_world_model_sha256,
        first.graph_revision_sha256,
        first.verifier_hashes,
    )
    if any(
        (
            item.source_world_model_sha256,
            item.graph_revision_sha256,
            item.verifier_hashes,
        )
        != contract
        for item in trajectories
    ):
        raise ValueError("trajectory execution contracts must match exactly")
    trajectory_hashes = tuple(item.sha256 for item in trajectories)
    if len(set(trajectory_hashes)) != len(trajectory_hashes):
        raise ValueError("trajectory verification receipts must be unique")
    action_kernel_set_sha256 = _action_kernel_set_sha256(action_kernels)
    candidates: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for trajectory in trajectories:
        seen: set[tuple[str, ...]] = set()
        for source, action, target in zip(
            trajectory.states[:-1],
            trajectory.actions,
            trajectory.states[1:],
            strict=True,
        ):
            try:
                primitive = _kernel(
                    action_kernels[action],
                    name=f"action_kernels[{action!r}]",
                )
            except KeyError as exc:
                raise KeyError(f"missing primitive action kernel: {action}") from exc
            if source >= primitive.shape[0] or target >= primitive.shape[1]:
                raise ValueError("trajectory state lies outside its world-model kernel")
            if primitive[source, target] <= 0.0:
                raise ValueError("verified trajectory contradicts its world-model kernel")
        for length in range(lower, min(upper, len(trajectory.actions)) + 1):
            for start in range(len(trajectory.actions) - length + 1):
                sequence = trajectory.actions[start : start + length]
                if any(action not in action_kernels for action in sequence):
                    raise KeyError(f"missing primitive action kernel in {sequence!r}")
                seen.add(sequence)
        for sequence in seen:
            candidates[sequence].add(trajectory.sha256)
    selected = sorted(
        (
            (sequence, tuple(sorted(sources)))
            for sequence, sources in candidates.items()
            if len(sources) >= support_floor
        ),
        key=lambda item: (-len(item[1]), -len(item[0]), item[0]),
    )[:limit]
    options: list[MacroOption] = []
    for sequence, sources in selected:
        identity = OptionIdentity.create(
            action_sequence=sequence,
            action_kernels=action_kernels,
            source_world_model_sha256=contract[0],
            graph_revision_sha256=contract[1],
            verifier_hashes=dict(contract[2]),
        )
        options.append(
            MacroOption.from_identity(
                identity,
                action_kernels=action_kernels,
                source_trajectory_sha256s=sources,
            )
        )
    result = tuple(options)
    receipt = OptionDiscoveryReceipt(
        source_world_model_sha256=contract[0],
        graph_revision_sha256=contract[1],
        verifier_hashes=contract[2],
        trajectory_sha256s=tuple(sorted(trajectory_hashes)),
        action_kernel_set_sha256=action_kernel_set_sha256,
        min_support=support_floor,
        min_option_length=lower,
        max_option_length=upper,
        candidate_count=len(candidates),
        discovered_option_sha256s=tuple(item.sha256 for item in result),
    )
    return OptionDiscoveryResult(options=result, receipt=receipt)


@dataclass(frozen=True, slots=True)
class PlanningOperator:
    operator_id: str
    action_sequence: tuple[str, ...]
    kernel: FloatArray
    option_sha256: str | None


@dataclass(frozen=True, slots=True)
class OptionPlan:
    start_state: int
    goal_state: int
    operator_ids: tuple[str, ...]
    primitive_actions: tuple[str, ...]
    operator_depth: int
    primitive_length: int
    expanded_states: int
    success_probability: float
    catalog_sha256: str
    search_depth_limit: int
    minimum_transition_probability: float
    optimal_within_depth_limit: bool = True

    FORMAT = "immer-ooe-option-plan/v2"

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_sha256": self.catalog_sha256,
            "expanded_states": self.expanded_states,
            "format": self.FORMAT,
            "goal_state": self.goal_state,
            "minimum_transition_probability_hex": (
                self.minimum_transition_probability.hex()
            ),
            "operator_depth": self.operator_depth,
            "operator_ids": list(self.operator_ids),
            "optimal_within_depth_limit": self.optimal_within_depth_limit,
            "primitive_actions": list(self.primitive_actions),
            "primitive_length": self.primitive_length,
            "search_depth_limit": self.search_depth_limit,
            "start_state": self.start_state,
            "success_probability_hex": self.success_probability.hex(),
        }

    @property
    def sha256(self) -> str:
        return _body_sha256(self.to_dict())


class OptionKernelCatalog:
    """Uniform planning view over primitive and verified macro kernels."""

    def __init__(
        self,
        action_kernels: Mapping[str, ArrayLike],
        options: Sequence[MacroOption] = (),
    ) -> None:
        if not action_kernels or len(action_kernels) > MAX_ACTIONS:
            raise ValueError(f"action kernel count must lie in [1, {MAX_ACTIONS}]")
        normalized: dict[str, FloatArray] = {}
        states: int | None = None
        for name, value in sorted(action_kernels.items()):
            action = _identifier(name, name="action kernel name")
            matrix = _kernel(value, name=f"action_kernels[{action!r}]", states=states)
            states = int(matrix.shape[0])
            normalized[action] = matrix.copy()
        self._action_kernels = normalized
        self._state_count = int(states or 0)
        if len(options) > MAX_DISCOVERED_OPTIONS:
            raise ValueError("option count exceeds MAX_DISCOVERED_OPTIONS")
        checked: list[MacroOption] = []
        for option in options:
            if not isinstance(option, MacroOption):
                raise TypeError("options must contain MacroOption values")
            if option.kernel_rows != self._state_count:
                raise ValueError("option and primitive state dimensions differ")
            option.verify_against(self._action_kernels)
            checked.append(option)
        self._options = tuple(sorted(checked, key=lambda item: item.sha256))
        self._catalog_sha256 = hashlib.sha256(
            canonical_json_bytes(
                {
                    "actions": [
                        [name, array_sha256(value)]
                        for name, value in self._action_kernels.items()
                    ],
                    "format": "immer-ooe-option-kernel-catalog/v1",
                    "options": [option.sha256 for option in self._options],
                }
            )
        ).hexdigest()

    @property
    def state_count(self) -> int:
        return self._state_count

    @property
    def sha256(self) -> str:
        return self._catalog_sha256

    def operators(self, *, include_options: bool = True) -> tuple[PlanningOperator, ...]:
        primitives = tuple(
            PlanningOperator(
                operator_id=f"action:{name}",
                action_sequence=(name,),
                kernel=value.copy(),
                option_sha256=None,
            )
            for name, value in self._action_kernels.items()
        )
        if not include_options:
            return primitives
        hierarchical = tuple(
            PlanningOperator(
                operator_id=f"option:{option.sha256}",
                action_sequence=option.identity.action_sequence,
                kernel=option.restore_kernel(),
                option_sha256=option.sha256,
            )
            for option in self._options
        )
        return primitives + hierarchical

    def contraction_ledger(self, plan: OptionPlan) -> ContractionLedger:
        """Bind an O(depth) Dobrushin verifier to a concrete plan."""

        if not isinstance(plan, OptionPlan):
            raise TypeError("plan must be an OptionPlan")
        if plan.catalog_sha256 != self.sha256:
            raise ValueError("plan belongs to another option catalog")
        try:
            kernels = tuple(
                self._action_kernels[action] for action in plan.primitive_actions
            )
        except KeyError as exc:
            raise ValueError("plan contains an unknown primitive action") from exc
        return ContractionLedger.from_kernels(kernels)

    def plan(
        self,
        start_state: int,
        goal_state: int,
        *,
        include_options: bool = True,
        minimum_transition_probability: float = 0.0,
        max_operator_depth: int = 128,
        max_expansions: int = MAX_PLANNING_EXPANSIONS,
    ) -> OptionPlan | None:
        """Return the maximum-probability plan within the depth bound.

        Dynamic programming retains the best probability for every
        ``(state, depth)`` pair.  Unlike argmax traversal, this evaluates every
        positive branch; unlike a global visited set, it permits a later depth
        to reach the same state with a probability that leads to a better
        complete route.
        """

        start = _integer(
            start_state,
            name="start_state",
            maximum=self._state_count - 1,
        )
        goal = _integer(
            goal_state,
            name="goal_state",
            maximum=self._state_count - 1,
        )
        threshold = float(minimum_transition_probability)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("minimum_transition_probability must lie in [0, 1]")
        depth_limit = _integer(
            max_operator_depth,
            name="max_operator_depth",
            minimum=0,
            maximum=MAX_OPTION_LENGTH * MAX_STATES,
        )
        expansion_limit = _integer(
            max_expansions,
            name="max_expansions",
            minimum=1,
            maximum=MAX_PLANNING_EXPANSIONS,
        )
        if start == goal:
            return OptionPlan(
                start_state=start,
                goal_state=goal,
                operator_ids=(),
                primitive_actions=(),
                operator_depth=0,
                primitive_length=0,
                expanded_states=0,
                success_probability=1.0,
                catalog_sha256=self.sha256,
                search_depth_limit=depth_limit,
                minimum_transition_probability=threshold,
            )
        operators = self.operators(include_options=include_options)
        # state -> (probability, operator path, primitive path)
        current: dict[
            int,
            tuple[float, tuple[str, ...], tuple[str, ...]],
        ] = {start: (1.0, (), ())}
        best_goal: tuple[
            float,
            int,
            tuple[str, ...],
            tuple[str, ...],
        ] | None = None
        expanded = 0
        for depth in range(1, depth_limit + 1):
            following: dict[
                int,
                tuple[float, tuple[str, ...], tuple[str, ...]],
            ] = {}
            for state in sorted(current):
                if expanded >= expansion_limit:
                    raise RuntimeError(
                        "planning expansion cap reached before optimality was proven"
                    )
                expanded += 1
                probability, operator_path, primitive_path = current[state]
                for operator in operators:
                    row = operator.kernel[state]
                    targets = np.flatnonzero(
                        (row > 0.0) & (row >= threshold)
                    )
                    for raw_target in targets:
                        target = int(raw_target)
                        candidate_probability = probability * float(row[target])
                        candidate_operators = operator_path + (operator.operator_id,)
                        candidate_primitives = (
                            primitive_path + operator.action_sequence
                        )
                        previous = following.get(target)
                        if previous is not None and (
                            candidate_probability < previous[0]
                            or (
                                candidate_probability == previous[0]
                                and candidate_operators >= previous[1]
                            )
                        ):
                            continue
                        following[target] = (
                            candidate_probability,
                            candidate_operators,
                            candidate_primitives,
                        )
            goal_candidate = following.get(goal)
            if goal_candidate is not None:
                candidate = (
                    goal_candidate[0],
                    depth,
                    goal_candidate[1],
                    goal_candidate[2],
                )
                if best_goal is None or (
                    candidate[0] > best_goal[0]
                    or (
                        candidate[0] == best_goal[0]
                        and (
                            candidate[1] < best_goal[1]
                            or (
                                candidate[1] == best_goal[1]
                                and candidate[2] < best_goal[2]
                            )
                        )
                    )
                ):
                    best_goal = candidate
            current = following
            if best_goal is not None and best_goal[0] == 1.0:
                # No stochastic path can improve on unit probability, and the
                # first such depth is already the deterministic shortest tie.
                break
            if not current:
                break
        if best_goal is None:
            return None
        probability, depth, operator_ids, primitive_actions = best_goal
        return OptionPlan(
            start_state=start,
            goal_state=goal,
            operator_ids=operator_ids,
            primitive_actions=primitive_actions,
            operator_depth=depth,
            primitive_length=len(primitive_actions),
            expanded_states=expanded,
            success_probability=probability,
            catalog_sha256=self.sha256,
            search_depth_limit=depth_limit,
            minimum_transition_probability=threshold,
        )


def _seed_digest(seed: str | int | bytes | None) -> tuple[bytes | None, str | None]:
    if seed is None:
        return None, None
    if isinstance(seed, bool) or not isinstance(seed, (str, int, bytes)):
        raise TypeError("gumbel_seed must be str, int, bytes, or None")
    if isinstance(seed, int):
        material = f"int:{seed}".encode()
    elif isinstance(seed, str):
        material = b"str:" + seed.encode("utf-8")
    else:
        material = b"bytes:" + seed
    return material, hashlib.sha256(material).hexdigest()


def _deterministic_gumbel(
    seed: bytes,
    agent: str,
    option: str,
) -> float:
    digest = hashlib.sha256(
        b"immer-ooe-option-gumbel/v1\0"
        + seed
        + b"\0"
        + agent.encode()
        + b"\0"
        + option.encode()
    ).digest()
    integer = int.from_bytes(digest[:8], "big")
    uniform = (integer + 0.5) / float(1 << 64)
    return -math.log(-math.log(uniform))


def _maximum_weight_one_to_one(weights: FloatArray) -> tuple[int | None, ...]:
    """Return the deterministic exact rectangular maximum-weight matching.

    The Hungarian primal-dual algorithm operates on a zero-padded square
    cost matrix.  Dummy columns represent unassigned agents when agents
    outnumber options; dummy rows absorb unused options in the opposite case.
    Iteration and tie order are both canonical ascending index order.
    """

    rows, columns = weights.shape
    size = max(rows, columns)
    padded_weights = np.zeros((size, size), dtype=np.float64)
    padded_weights[:rows, :columns] = weights
    cost = float(np.max(padded_weights)) - padded_weights
    u = np.zeros(size + 1, dtype=np.float64)
    v = np.zeros(size + 1, dtype=np.float64)
    matched_row = np.zeros(size + 1, dtype=np.int64)
    predecessor = np.zeros(size + 1, dtype=np.int64)
    for row_one_based in range(1, size + 1):
        matched_row[0] = row_one_based
        minimum = np.full(size + 1, np.inf, dtype=np.float64)
        used = np.zeros(size + 1, dtype=bool)
        column = 0
        while True:
            used[column] = True
            active_row = int(matched_row[column])
            available = np.flatnonzero(~used[1:]) + 1
            reduced = (
                cost[active_row - 1, available - 1]
                - u[active_row]
                - v[available]
            )
            better = reduced < minimum[available]
            if np.any(better):
                improved_columns = available[better]
                minimum[improved_columns] = reduced[better]
                predecessor[improved_columns] = column
            available_values = minimum[available]
            delta = float(np.min(available_values))
            # flatnonzero preserves the canonical smallest-index tie break.
            next_column = int(available[np.flatnonzero(available_values == delta)[0]])
            u[matched_row[used]] += delta
            v[used] -= delta
            minimum[~used] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = int(predecessor[column])
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break
    row_to_column = np.full(size, -1, dtype=np.int64)
    for column in range(1, size + 1):
        row_to_column[matched_row[column] - 1] = column - 1
    return tuple(
        int(column) if column < columns else None
        for column in row_to_column[:rows]
    )


def _assignment_sha256(
    agent_ids: tuple[str, ...],
    option_sha256s: tuple[str, ...],
    assigned_columns: tuple[int | None, ...],
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "assignments": [
                    {
                        "agent_id": agent,
                        "option_sha256": (
                            None if column is None else option_sha256s[column]
                        ),
                    }
                    for agent, column in zip(
                        agent_ids,
                        assigned_columns,
                        strict=True,
                    )
                ],
                "format": "immer-ooe-discrete-option-assignment/v1",
            }
        )
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class AllocationReceipt:
    agent_ids: tuple[str, ...]
    option_sha256s: tuple[str, ...]
    score_sha256: str
    allocation_sha256: str
    padded_allocation_sha256: str
    padded_size: int
    temperature_hex: str
    sinkhorn_rounds: int
    gumbel_scale_hex: str
    gumbel_seed_sha256: str | None
    maximum_marginal_error_hex: str
    assigned_option_sha256s: tuple[str | None, ...]
    assignment_sha256: str
    assignment_weight_hex: str

    FORMAT = "immer-ooe-balanced-option-allocation-receipt/v1"

    def __post_init__(self) -> None:
        agents = _identifiers(
            self.agent_ids,
            name="agent_id",
            maximum=MAX_ALLOCATION_SIZE,
        )
        options = tuple(
            require_sha256(value, field="option_sha256")
            for value in self.option_sha256s
        )
        if not options or len(options) > MAX_ALLOCATION_SIZE:
            raise ValueError("option count exceeds allocation bounds")
        if len(set(agents)) != len(agents) or len(set(options)) != len(options):
            raise ValueError("agent and option identities must be unique")
        for field in (
            "score_sha256",
            "allocation_sha256",
            "padded_allocation_sha256",
            "assignment_sha256",
        ):
            object.__setattr__(
                self,
                field,
                require_sha256(getattr(self, field), field=field),
            )
        if self.gumbel_seed_sha256 is not None:
            object.__setattr__(
                self,
                "gumbel_seed_sha256",
                require_sha256(
                    self.gumbel_seed_sha256,
                    field="gumbel_seed_sha256",
                ),
            )
        size = _integer(
            self.padded_size,
            name="padded_size",
            minimum=max(len(agents), len(options)),
            maximum=MAX_ALLOCATION_SIZE,
        )
        rounds = _integer(
            self.sinkhorn_rounds,
            name="sinkhorn_rounds",
            minimum=20,
            maximum=100_000,
        )
        for field, positive in (
            ("temperature_hex", True),
            ("gumbel_scale_hex", False),
            ("maximum_marginal_error_hex", False),
            ("assignment_weight_hex", False),
        ):
            raw = getattr(self, field)
            if not isinstance(raw, str):
                raise ValueError(f"{field} must be a canonical hexadecimal float")
            try:
                value = float.fromhex(raw)
            except ValueError as exc:
                raise ValueError(
                    f"{field} must be a canonical hexadecimal float"
                ) from exc
            if (
                not math.isfinite(value)
                or value < 0.0
                or (positive and value == 0.0)
                or value.hex() != raw
            ):
                raise ValueError(f"{field} is not canonical and bounded")
        assigned = tuple(self.assigned_option_sha256s)
        if len(assigned) != len(agents):
            raise ValueError("discrete assignment must cover every agent")
        normalized_assigned: list[str | None] = []
        for value in assigned:
            if value is None:
                normalized_assigned.append(None)
            else:
                digest = require_sha256(value, field="assigned option")
                if digest not in options:
                    raise ValueError("assigned option is outside the receipt option set")
                normalized_assigned.append(digest)
        real_assignments = [value for value in normalized_assigned if value is not None]
        if len(real_assignments) != min(len(agents), len(options)):
            raise ValueError("discrete assignment has incorrect matching cardinality")
        if len(set(real_assignments)) != len(real_assignments):
            raise ValueError("discrete assignment must be one-to-one")
        assigned_columns = tuple(
            None if value is None else options.index(value)
            for value in normalized_assigned
        )
        if _assignment_sha256(agents, options, assigned_columns) != self.assignment_sha256:
            raise ValueError("discrete assignment hash mismatch")
        object.__setattr__(self, "agent_ids", agents)
        object.__setattr__(self, "option_sha256s", options)
        object.__setattr__(self, "padded_size", size)
        object.__setattr__(self, "sinkhorn_rounds", rounds)
        object.__setattr__(
            self,
            "assigned_option_sha256s",
            tuple(normalized_assigned),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_ids": list(self.agent_ids),
            "allocation_sha256": self.allocation_sha256,
            "assigned_option_sha256s": list(self.assigned_option_sha256s),
            "assignment_sha256": self.assignment_sha256,
            "assignment_weight_hex": self.assignment_weight_hex,
            "format": self.FORMAT,
            "gumbel_scale_hex": self.gumbel_scale_hex,
            "gumbel_seed_sha256": self.gumbel_seed_sha256,
            "maximum_marginal_error_hex": self.maximum_marginal_error_hex,
            "option_sha256s": list(self.option_sha256s),
            "padded_allocation_sha256": self.padded_allocation_sha256,
            "padded_size": self.padded_size,
            "score_sha256": self.score_sha256,
            "sinkhorn_rounds": self.sinkhorn_rounds,
            "temperature_hex": self.temperature_hex,
        }

    @property
    def sha256(self) -> str:
        return _body_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class BalancedOptionAllocation:
    """Birkhoff-balanced soft allocation with a fully bound receipt."""

    allocation: FloatArray
    padded_allocation: FloatArray
    receipt: AllocationReceipt

    def __post_init__(self) -> None:
        allocation = np.asarray(self.allocation, dtype=np.float64, order="C")
        padded = np.asarray(self.padded_allocation, dtype=np.float64, order="C")
        expected = (len(self.receipt.agent_ids), len(self.receipt.option_sha256s))
        if allocation.shape != expected:
            raise OptionIntegrityError("allocation shape does not match receipt")
        if padded.shape != (self.receipt.padded_size, self.receipt.padded_size):
            raise OptionIntegrityError("padded allocation shape does not match receipt")
        if not np.isfinite(padded).all() or np.any(padded < 0.0):
            raise OptionIntegrityError("allocation must be finite and non-negative")
        if array_sha256(allocation) != self.receipt.allocation_sha256:
            raise OptionIntegrityError("allocation hash mismatch")
        if array_sha256(padded) != self.receipt.padded_allocation_sha256:
            raise OptionIntegrityError("padded allocation hash mismatch")
        marginal_error = max(
            float(np.max(np.abs(padded.sum(axis=0) - 1.0))),
            float(np.max(np.abs(padded.sum(axis=1) - 1.0))),
        )
        if marginal_error > 1e-9:
            raise OptionIntegrityError("allocation is not in the Birkhoff polytope")
        recorded = float.fromhex(self.receipt.maximum_marginal_error_hex)
        if not math.isclose(recorded, marginal_error, rel_tol=0.0, abs_tol=1e-18):
            raise OptionIntegrityError("allocation marginal receipt mismatch")
        assignment_columns = _maximum_weight_one_to_one(allocation)
        assigned = tuple(
            None
            if column is None
            else self.receipt.option_sha256s[column]
            for column in assignment_columns
        )
        if assigned != self.receipt.assigned_option_sha256s:
            raise OptionIntegrityError("discrete assignment is not maximum-weight")
        assignment_weight = sum(
            float(allocation[row, column])
            for row, column in enumerate(assignment_columns)
            if column is not None
        )
        if assignment_weight.hex() != self.receipt.assignment_weight_hex:
            raise OptionIntegrityError("discrete assignment weight mismatch")
        object.__setattr__(self, "allocation", allocation.copy())
        object.__setattr__(self, "padded_allocation", padded.copy())
        self.allocation.setflags(write=False)
        self.padded_allocation.setflags(write=False)

    def assigned_option(self, agent_id: str) -> str | None:
        try:
            row = self.receipt.agent_ids.index(agent_id)
        except ValueError as exc:
            raise KeyError(f"unknown agent: {agent_id}") from exc
        return self.receipt.assigned_option_sha256s[row]


def allocate_options_balanced(
    scores: ArrayLike,
    *,
    agent_ids: Sequence[str],
    option_sha256s: Sequence[str],
    temperature: float = 1.0,
    sinkhorn_rounds: int = 512,
    gumbel_seed: str | int | bytes | None = None,
    gumbel_scale: float = 0.0,
) -> BalancedOptionAllocation:
    """Allocate options through a padded Birkhoff/Sinkhorn projection."""

    agents = _identifiers(
        agent_ids,
        name="agent_id",
        maximum=MAX_ALLOCATION_SIZE,
    )
    options = tuple(
        require_sha256(value, field="option_sha256") for value in option_sha256s
    )
    if not options or len(options) > MAX_ALLOCATION_SIZE:
        raise ValueError("option count exceeds allocation bounds")
    if len(set(agents)) != len(agents) or len(set(options)) != len(options):
        raise ValueError("agent and option identities must be unique")
    raw = np.asarray(scores)
    if raw.shape != (len(agents), len(options)):
        raise ValueError("score matrix shape must equal agents by options")
    try:
        utilities = np.asarray(raw, dtype=np.float64, order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("scores must contain real values") from exc
    if not np.isfinite(utilities).all():
        raise ValueError("scores must be finite")
    scale = _finite_positive(temperature, name="temperature")
    rounds = _integer(
        sinkhorn_rounds,
        name="sinkhorn_rounds",
        minimum=20,
        maximum=100_000,
    )
    noise_scale = float(gumbel_scale)
    if not math.isfinite(noise_scale) or noise_scale < 0.0:
        raise ValueError("gumbel_scale must be finite and non-negative")
    seed_material, seed_sha = _seed_digest(gumbel_seed)
    if noise_scale > 0.0 and seed_material is None:
        raise ValueError("positive gumbel_scale requires a deterministic seed")
    logits = utilities / scale
    if seed_material is not None and noise_scale > 0.0:
        noise = np.asarray(
            [
                [
                    _deterministic_gumbel(seed_material, agent, option)
                    for option in options
                ]
                for agent in agents
            ],
            dtype=np.float64,
        )
        logits = logits + noise_scale * noise
    size = max(len(agents), len(options))
    padded_logits = np.zeros((size, size), dtype=np.float64)
    padded_logits[: len(agents), : len(options)] = logits
    padded_logits -= float(np.max(padded_logits))
    weights = np.exp(np.clip(padded_logits, -60.0, 0.0))
    padded = sinkhorn_project(
        weights,
        rounds=rounds,
        max_nodes=MAX_ALLOCATION_SIZE,
        max_bytes=MAX_PAYLOAD_BYTES,
    )
    marginal_error = max(
        float(np.max(np.abs(padded.sum(axis=0) - 1.0))),
        float(np.max(np.abs(padded.sum(axis=1) - 1.0))),
    )
    if marginal_error > 1e-9:
        raise ArithmeticError("Sinkhorn allocation did not reach Birkhoff tolerance")
    allocation = np.ascontiguousarray(
        padded[: len(agents), : len(options)], dtype=np.float64
    )
    assignment_columns = _maximum_weight_one_to_one(allocation)
    assigned_options = tuple(
        None if column is None else options[column]
        for column in assignment_columns
    )
    assignment_weight = sum(
        float(allocation[row, column])
        for row, column in enumerate(assignment_columns)
        if column is not None
    )
    receipt = AllocationReceipt(
        agent_ids=agents,
        option_sha256s=options,
        score_sha256=array_sha256(utilities),
        allocation_sha256=array_sha256(allocation),
        padded_allocation_sha256=array_sha256(padded),
        padded_size=size,
        temperature_hex=scale.hex(),
        sinkhorn_rounds=rounds,
        gumbel_scale_hex=noise_scale.hex(),
        gumbel_seed_sha256=seed_sha,
        maximum_marginal_error_hex=marginal_error.hex(),
        assigned_option_sha256s=assigned_options,
        assignment_sha256=_assignment_sha256(
            agents,
            options,
            assignment_columns,
        ),
        assignment_weight_hex=assignment_weight.hex(),
    )
    return BalancedOptionAllocation(
        allocation=allocation,
        padded_allocation=padded,
        receipt=receipt,
    )


__all__ = [
    "AllocationReceipt",
    "BalancedOptionAllocation",
    "MacroOption",
    "MAX_ACTIONS",
    "MAX_ALLOCATION_SIZE",
    "MAX_DISCOVERED_OPTIONS",
    "MAX_OPTION_LENGTH",
    "MAX_PLANNING_EXPANSIONS",
    "MAX_STATES",
    "MAX_TRAJECTORIES",
    "OptionDiscoveryReceipt",
    "OptionDiscoveryResult",
    "OptionIdentity",
    "OptionIntegrityError",
    "OptionKernelCatalog",
    "OptionPlan",
    "PlanningOperator",
    "VerifiedTrajectory",
    "allocate_options_balanced",
    "compose_stochastic_kernels",
    "discover_macro_options",
]
