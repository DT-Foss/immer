"""Bounded reversible and Fiedler-oriented PS-Lifted consensus.

Transition matrices use the PoC's row-stochastic convention.  Push-sum acts
with their transpose, so mass is conserved for non-reversible kernels without
assuming double stochasticity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .math_core import (
    DEFAULT_MAX_DENSE_BYTES,
    DEFAULT_MAX_NODES,
    FloatArray,
    array_sha256,
    as_float_matrix,
    kernel_diagnostics,
    spectral_gap,
)


class ConsensusMassError(ArithmeticError):
    """Raised when push-sum cannot deliver positive mass to every output node."""


def _integer(value: int, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _finite_nonnegative(value: float, *, name: str, positive: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (positive and result == 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a finite {qualifier} number")
    return result


def _working_set_bytes(*shapes: tuple[int, ...]) -> int:
    return sum(math.prod(shape) * np.dtype(np.float64).itemsize for shape in shapes)


def _enforce_working_set(max_bytes: int, *shapes: tuple[int, ...]) -> None:
    limit = _integer(max_bytes, name="max_bytes", minimum=1)
    required = _working_set_bytes(*shapes)
    if required > limit:
        raise ValueError(
            f"dense consensus working set requires {required} bytes, exceeding {limit}"
        )


def validate_adjacency(
    adjacency: ArrayLike,
    *,
    require_connected: bool = True,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> FloatArray:
    """Validate a finite, weighted, undirected simple graph with contiguous nodes."""

    graph = as_float_matrix(
        adjacency,
        name="adjacency",
        square=True,
        nonnegative=True,
        max_nodes=max_nodes,
        max_bytes=max_bytes,
        working_arrays=4,
    )
    size = graph.shape[0]
    if size < 2:
        raise ValueError("adjacency must contain at least two nodes")
    if not np.allclose(graph, graph.T, atol=1e-12, rtol=1e-12):
        raise ValueError("adjacency must be symmetric")
    if np.any(np.abs(np.diag(graph)) > 1e-12):
        raise ValueError("adjacency must have a zero diagonal")
    result = np.ascontiguousarray(0.5 * (graph + graph.T), dtype=np.float64)
    np.fill_diagonal(result, 0.0)
    if np.any(result.sum(axis=1) <= 0.0):
        raise ValueError("adjacency contains an isolated node")
    if require_connected:
        visited = np.zeros(size, dtype=np.bool_)
        pending = [0]
        visited[0] = True
        while pending:
            node = pending.pop()
            for neighbor in np.flatnonzero(result[node] > 0.0):
                index = int(neighbor)
                if not visited[index]:
                    visited[index] = True
                    pending.append(index)
        if not bool(visited.all()):
            missing = np.flatnonzero(~visited).astype(int).tolist()
            raise ValueError(f"adjacency is disconnected; unreachable nodes: {missing}")
    return result


def adjacency_from_edges(
    size: int,
    edges: Iterable[Sequence[float]],
    *,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> FloatArray:
    """Construct a connected weighted adjacency from zero-based contiguous IDs.

    Each edge is ``(source, target)`` or ``(source, target, weight)``.  Duplicate
    undirected edges add their weights.
    """

    count = _integer(size, name="size", minimum=2)
    if count > _integer(max_nodes, name="max_nodes", minimum=1):
        raise ValueError(f"size exceeds max_nodes={max_nodes}")
    _enforce_working_set(max_bytes, (count, count), (count, count))
    graph = np.zeros((count, count), dtype=np.float64)
    saw_edge = False
    for ordinal, edge in enumerate(edges):
        values = tuple(edge)
        if len(values) not in {2, 3}:
            raise ValueError(f"edge {ordinal} must contain two or three values")
        left_raw, right_raw = values[:2]
        try:
            integral_ids = (
                not isinstance(left_raw, bool)
                and not isinstance(right_raw, bool)
                and int(left_raw) == left_raw
                and int(right_raw) == right_raw
            )
        except (TypeError, ValueError, OverflowError):
            integral_ids = False
        if not integral_ids:
            raise ValueError(f"edge {ordinal} node IDs must be integers")
        left, right = int(left_raw), int(right_raw)
        if not 0 <= left < count or not 0 <= right < count:
            raise ValueError(f"edge {ordinal} node ID is outside [0, {count})")
        if left == right:
            raise ValueError(f"edge {ordinal} is a self-loop")
        weight = 1.0 if len(values) == 2 else float(values[2])
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"edge {ordinal} weight must be positive and finite")
        graph[left, right] += weight
        graph[right, left] += weight
        saw_edge = True
    if not saw_edge:
        raise ValueError("at least one edge is required")
    return validate_adjacency(
        graph,
        max_nodes=max_nodes,
        max_bytes=max_bytes,
    )


def barbell_adjacency(
    left: int,
    right: int,
    *,
    bridge_weight: float = 1.0,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> FloatArray:
    left_size = _integer(left, name="left", minimum=2)
    right_size = _integer(right, name="right", minimum=2)
    bridge = _finite_nonnegative(bridge_weight, name="bridge_weight", positive=True)
    size = left_size + right_size
    if size > _integer(max_nodes, name="max_nodes", minimum=1):
        raise ValueError(f"barbell size exceeds max_nodes={max_nodes}")
    _enforce_working_set(max_bytes, (size, size), (size, size))
    adjacency = np.zeros((size, size), dtype=np.float64)
    for start, width in ((0, left_size), (left_size, right_size)):
        adjacency[start : start + width, start : start + width] = np.ones(
            (width, width), dtype=np.float64
        ) - np.eye(width)
    adjacency[left_size - 1, left_size] = bridge
    adjacency[left_size, left_size - 1] = bridge
    return adjacency


def complete_adjacency(
    size: int,
    *,
    edge_weight: float = 1.0,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> FloatArray:
    count = _integer(size, name="size", minimum=2)
    weight = _finite_nonnegative(edge_weight, name="edge_weight", positive=True)
    if count > _integer(max_nodes, name="max_nodes", minimum=1):
        raise ValueError(f"size exceeds max_nodes={max_nodes}")
    _enforce_working_set(max_bytes, (count, count), (count, count))
    return weight * (np.ones((count, count), dtype=np.float64) - np.eye(count))


def fiedler_eigenspace(
    adjacency: ArrayLike,
    *,
    eigen_tolerance: float = 1e-10,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> tuple[float, FloatArray]:
    """Return a deterministic canonical basis for the Fiedler eigenspace.

    Degenerate eigenspaces are canonicalized through their basis-invariant
    projector and ordered coordinate anchors.  Each vector's sign is fixed by
    making its largest-magnitude coordinate positive.
    """

    tolerance = _finite_nonnegative(
        eigen_tolerance, name="eigen_tolerance", positive=True
    )
    graph = validate_adjacency(
        adjacency,
        max_nodes=max_nodes,
        max_bytes=max_bytes,
    )
    size = graph.shape[0]
    _enforce_working_set(
        max_bytes,
        (size, size),
        (size, size),
        (size, size),
        (size, size),
        (size,),
    )
    laplacian = np.diag(graph.sum(axis=1)) - graph
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    fiedler = float(eigenvalues[1])
    if not math.isfinite(fiedler) or fiedler <= tolerance:
        raise ValueError("adjacency has no positive Fiedler eigenvalue")
    scale = max(1.0, abs(fiedler))
    selected = np.flatnonzero(np.abs(eigenvalues - fiedler) <= tolerance * scale)
    raw_basis = eigenvectors[:, selected]
    # Sum outer products explicitly.  Besides being basis-invariant, this
    # avoids platform BLAS kernels that spuriously raise floating-point
    # warnings for the wide, exactly-degenerate complete-graph eigenspace.
    projector = np.zeros((size, size), dtype=np.float64)
    for column in raw_basis.T:
        projector += np.outer(column, column)

    canonical: list[FloatArray] = []
    rank = int(selected.size)
    for anchor_index in range(size):
        candidate = projector[:, anchor_index].copy()
        for previous in canonical:
            candidate -= previous * float(previous @ candidate)
        norm = float(np.linalg.norm(candidate))
        if norm <= tolerance:
            continue
        candidate /= norm
        candidate -= candidate.mean()
        norm = float(np.linalg.norm(candidate))
        if norm <= tolerance:
            continue
        candidate /= norm
        pivot = int(np.argmax(np.abs(candidate)))
        if candidate[pivot] < 0.0:
            candidate *= -1.0
        candidate[np.abs(candidate) <= tolerance * 1e-3] = 0.0
        candidate /= float(np.linalg.norm(candidate))
        canonical.append(np.ascontiguousarray(candidate))
        if len(canonical) == rank:
            break
    if len(canonical) != rank:
        raise ArithmeticError("could not canonicalize the complete Fiedler eigenspace")
    basis = np.column_stack(canonical).astype(np.float64, copy=False)
    return fiedler, np.ascontiguousarray(basis)


def fiedler_vector(
    adjacency: ArrayLike,
    *,
    eigen_tolerance: float = 1e-10,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> tuple[float, FloatArray]:
    eigenvalue, basis = fiedler_eigenspace(
        adjacency,
        eigen_tolerance=eigen_tolerance,
        max_nodes=max_nodes,
        max_bytes=max_bytes,
    )
    return eigenvalue, np.ascontiguousarray(basis[:, 0])


def metropolis_matrix(
    adjacency: ArrayLike,
    *,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> FloatArray:
    graph = validate_adjacency(
        adjacency,
        max_nodes=max_nodes,
        max_bytes=max_bytes,
    )
    size = graph.shape[0]
    _enforce_working_set(max_bytes, (size, size), (size, size), (size,))
    degrees = graph.sum(axis=1)
    transition = np.zeros_like(graph)
    for source in range(size):
        for target_raw in np.flatnonzero(graph[source] > 0.0):
            target = int(target_raw)
            transition[source, target] = graph[source, target] / (
                1.0 + max(degrees[source], degrees[target])
            )
        transition[source, source] = 1.0 - transition[source].sum()
    if np.any(transition < -1e-14):
        raise ArithmeticError("Metropolis construction produced a negative probability")
    np.maximum(transition, 0.0, out=transition)
    transition /= transition.sum(axis=1, keepdims=True)
    return np.ascontiguousarray(transition)


def ps_lifted_matrix(
    adjacency: ArrayLike,
    *,
    pc: float = 0.65,
    ps: float = 0.003,
    fiedler_tolerance: float = 1e-12,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> FloatArray:
    """Fiedler-oriented Z2 lift from the PS-Lifted/Foss Gap construction."""

    continue_probability = float(pc)
    stay_probability = float(ps)
    tie_tolerance = _finite_nonnegative(fiedler_tolerance, name="fiedler_tolerance")
    if (
        not math.isfinite(continue_probability)
        or not math.isfinite(stay_probability)
        or not 0.0 < continue_probability < 1.0
        or not 0.0 <= stay_probability < 1.0
        or continue_probability + stay_probability >= 1.0
    ):
        raise ValueError("require finite 0<pc<1, 0<=ps<1 and pc+ps<1")
    graph = validate_adjacency(
        adjacency,
        max_nodes=max_nodes,
        max_bytes=max_bytes,
    )
    size = graph.shape[0]
    _enforce_working_set(
        max_bytes,
        (size, size),
        (2 * size, 2 * size),
        (size, size),
        (size,),
    )
    reverse_probability = 1.0 - continue_probability - stay_probability
    _, vector = fiedler_vector(
        graph,
        max_nodes=max_nodes,
        max_bytes=max_bytes,
    )
    transition = np.zeros((2 * size, 2 * size), dtype=np.float64)

    for source in range(size):
        neighbors = [int(value) for value in np.flatnonzero(graph[source] > 0.0)]
        forward: list[int] = []
        backward: list[int] = []
        for target in neighbors:
            difference = vector[target] - vector[source]
            if difference > tie_tolerance:
                forward.append(target)
            elif difference < -tie_tolerance:
                backward.append(target)
            else:
                (forward if source < target else backward).append(target)

        forward_weight = float(sum(graph[source, target] for target in forward))
        backward_weight = float(sum(graph[source, target] for target in backward))

        transition[source, source] += stay_probability
        if forward:
            for target in forward:
                transition[source, target] += (
                    continue_probability * graph[source, target] / forward_weight
                )
        else:
            transition[source, source] += continue_probability
        if backward:
            for target in backward:
                transition[source, size + target] += (
                    reverse_probability * graph[source, target] / backward_weight
                )
        else:
            transition[source, source] += reverse_probability

        reverse_source = size + source
        transition[reverse_source, reverse_source] += stay_probability
        if backward:
            for target in backward:
                transition[reverse_source, size + target] += (
                    continue_probability * graph[source, target] / backward_weight
                )
        else:
            transition[reverse_source, reverse_source] += continue_probability
        if forward:
            for target in forward:
                transition[reverse_source, target] += (
                    reverse_probability * graph[source, target] / forward_weight
                )
        else:
            transition[reverse_source, reverse_source] += reverse_probability

    if not np.allclose(transition.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise ArithmeticError("PS-Lifted construction is not row-stochastic")
    return np.ascontiguousarray(transition)


@dataclass(frozen=True, slots=True)
class AdaptiveLiftParameters:
    """Topology-derived PS-Lift parameters from the Foss gap schedule."""

    adjacency_sha256: str
    fiedler_eigenvalue: float
    pc: float
    ps: float
    pc_floor: float
    pc_ceiling: float
    formula_pc: float
    selection: str
    spectral_candidates: tuple[tuple[float, float], ...]

    def to_dict(self) -> dict[str, Any]:
        # Spectral gaps are diagnostic only: ``pc`` has already been selected
        # from the full-precision measurements above.  Round their serialized
        # representation so platform BLAS eigensolver noise does not change a
        # future Crystal's content address.
        def canonical_gap(value: float) -> float:
            rounded = round(float(value), 12)
            return 0.0 if rounded == 0.0 else rounded

        return {
            "schema": "immer-ooe-adaptive-ps-lift/v1",
            "adjacency_sha256": self.adjacency_sha256,
            "fiedler_eigenvalue": self.fiedler_eigenvalue,
            "pc": self.pc,
            "ps": self.ps,
            "pc_floor": self.pc_floor,
            "pc_ceiling": self.pc_ceiling,
            "formula_pc": self.formula_pc,
            "selection": self.selection,
            "spectral_candidates": [
                {"gap": canonical_gap(gap), "pc": candidate_pc}
                for candidate_pc, gap in self.spectral_candidates
            ],
            "formula": "clip(0.85-0.05*log(lambda2),floor,ceiling)",
        }

    @property
    def sha256(self) -> str:
        import hashlib

        from .identity import canonical_json_bytes

        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()


def adaptive_lift_parameters(
    adjacency: ArrayLike,
    *,
    ps: float = 0.003,
    pc_floor: float = 0.50,
    pc_ceiling: float = 0.97,
    spectral_calibration_nodes: int = 128,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> AdaptiveLiftParameters:
    """Derive ``pc`` from the current topology instead of fixing it globally."""

    stay = _finite_nonnegative(ps, name="ps")
    floor = _finite_nonnegative(pc_floor, name="pc_floor", positive=True)
    ceiling = _finite_nonnegative(pc_ceiling, name="pc_ceiling", positive=True)
    if not floor <= ceiling < 1.0 or ceiling + stay >= 1.0:
        raise ValueError("require 0 < pc_floor <= pc_ceiling and pc_ceiling+ps < 1")
    graph = validate_adjacency(
        adjacency,
        max_nodes=max_nodes,
        max_bytes=max_bytes,
    )
    fiedler, _ = fiedler_vector(
        graph,
        max_nodes=max_nodes,
        max_bytes=max_bytes,
    )
    raw = 0.85 - 0.05 * math.log(fiedler)
    formula_pc = float(np.clip(raw, floor, ceiling))
    degrees = graph.sum(axis=1)
    degree_ratio = float(degrees.max() / degrees.mean())
    if degree_ratio > 5.0:
        formula_pc = min(formula_pc, 0.90)
    if formula_pc == ceiling and ceiling >= 0.97:
        formula_pc = min(formula_pc, 0.90)
    calibration_limit = _integer(
        spectral_calibration_nodes,
        name="spectral_calibration_nodes",
        minimum=2,
    )
    spectral_candidates: tuple[tuple[float, float], ...] = ()
    if graph.shape[0] <= calibration_limit:
        grid = np.linspace(floor, min(ceiling, 0.95), num=12)
        candidates = tuple(
            sorted(
                {
                    formula_pc,
                    float(np.clip(0.65, floor, ceiling)),
                    *(float(value) for value in grid),
                }
            )
        )
        measured = []
        for candidate in candidates:
            transition = ps_lifted_matrix(
                graph,
                pc=candidate,
                ps=stay,
                max_nodes=max_nodes,
                max_bytes=max_bytes,
            )
            measured.append(
                (
                    candidate,
                    spectral_gap(
                        transition,
                        max_nodes=2 * max_nodes,
                        max_bytes=max_bytes,
                    ),
                )
            )
        spectral_candidates = tuple(measured)
        pc = max(
            measured,
            key=lambda item: (
                item[1],
                -abs(item[0] - formula_pc),
                -item[0],
            ),
        )[0]
        selection = "formula-proposal+spectral-self-calibration"
    else:
        pc = formula_pc
        selection = "formula-proposal"
    return AdaptiveLiftParameters(
        adjacency_sha256=array_sha256(graph),
        fiedler_eigenvalue=fiedler,
        pc=pc,
        ps=stay,
        pc_floor=floor,
        pc_ceiling=ceiling,
        formula_pc=formula_pc,
        selection=selection,
        spectral_candidates=spectral_candidates,
    )


def adaptive_ps_lift_matrix(
    adjacency: ArrayLike,
    *,
    ps: float = 0.003,
    pc_floor: float = 0.50,
    pc_ceiling: float = 0.97,
    spectral_calibration_nodes: int = 128,
    fiedler_tolerance: float = 1e-12,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> tuple[FloatArray, AdaptiveLiftParameters]:
    """Return the topology-calibrated lift and its canonical parameters."""

    graph = validate_adjacency(
        adjacency,
        max_nodes=max_nodes,
        max_bytes=max_bytes,
    )
    parameters = adaptive_lift_parameters(
        graph,
        ps=ps,
        pc_floor=pc_floor,
        pc_ceiling=pc_ceiling,
        spectral_calibration_nodes=spectral_calibration_nodes,
        max_nodes=max_nodes,
        max_bytes=max_bytes,
    )
    return (
        ps_lifted_matrix(
            graph,
            pc=parameters.pc,
            ps=parameters.ps,
            fiedler_tolerance=fiedler_tolerance,
            max_nodes=max_nodes,
            max_bytes=max_bytes,
        ),
        parameters,
    )


def _validate_transition(
    row_transition: ArrayLike,
    *,
    max_nodes: int,
    max_bytes: int,
) -> FloatArray:
    transition = as_float_matrix(
        row_transition,
        name="row_transition",
        square=True,
        nonnegative=True,
        max_nodes=2 * max_nodes,
        max_bytes=max_bytes,
        working_arrays=2,
    )
    if transition.shape[0] < 2:
        raise ValueError("row_transition must contain at least two nodes")
    if not np.allclose(transition.sum(axis=1), 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError("row_transition must be row-stochastic")
    return np.ascontiguousarray(transition)


def _validate_payload(
    values: ArrayLike,
    *,
    nodes: int,
    max_nodes: int,
    max_bytes: int,
) -> FloatArray:
    payload = np.asarray(values)
    if payload.ndim == 1:
        payload = payload[:, None]
    value = as_float_matrix(
        payload,
        name="values",
        max_nodes=max(max_nodes, int(payload.shape[1]) if payload.ndim == 2 else 1),
        max_bytes=max_bytes,
        working_arrays=3,
    )
    if value.shape[0] != nodes:
        raise ValueError("transition/value shape mismatch")
    return value


def _push_sum_state(
    transition: FloatArray,
    payload: FloatArray,
    *,
    lifted_nodes: int | None,
    max_bytes: int,
) -> tuple[FloatArray, FloatArray]:
    width = payload.shape[1]
    if lifted_nodes is None:
        _enforce_working_set(
            max_bytes,
            transition.shape,
            payload.shape,
            payload.shape,
            (payload.shape[0], 1),
        )
        return payload.copy(), np.ones((payload.shape[0], 1), dtype=np.float64)
    nodes = _integer(lifted_nodes, name="lifted_nodes", minimum=2)
    if transition.shape[0] != 2 * nodes or payload.shape[0] != nodes:
        raise ValueError("invalid lifted transition/value shape")
    _enforce_working_set(
        max_bytes,
        transition.shape,
        (2 * nodes, width),
        (2 * nodes, width),
        (2 * nodes, 1),
    )
    signal = np.zeros((2 * nodes, width), dtype=np.float64)
    weight = np.zeros((2 * nodes, 1), dtype=np.float64)
    signal[:nodes] = payload
    weight[:nodes] = 1.0
    return signal, weight


def _estimates_and_mass(
    signal: FloatArray,
    weight: FloatArray,
    *,
    lifted_nodes: int | None,
) -> tuple[FloatArray, FloatArray]:
    if lifted_nodes is None:
        visible_signal = signal
        visible_weight = weight
    else:
        visible_signal = signal[:lifted_nodes] + signal[lifted_nodes:]
        visible_weight = weight[:lifted_nodes] + weight[lifted_nodes:]
    invalid = np.flatnonzero(visible_weight[:, 0] <= 0.0)
    if invalid.size:
        raise ConsensusMassError(
            "push-sum has zero mass at unreachable nodes: "
            f"{invalid.astype(int).tolist()}"
        )
    estimates = visible_signal / visible_weight
    if not np.isfinite(estimates).all():
        raise ArithmeticError("push-sum produced non-finite estimates")
    return np.ascontiguousarray(estimates), np.ascontiguousarray(visible_weight)


def _advance_push_sum(
    operator: FloatArray,
    signal: FloatArray,
    weight: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    # Some Accelerate-backed NumPy builds emit spurious matmul FP warnings for
    # long stochastic recurrences.  Validate the actual outputs explicitly so
    # real overflow still fails while the finite recurrence stays silent.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        next_signal = operator @ signal
        next_weight = operator @ weight
    if not np.isfinite(next_signal).all() or not np.isfinite(next_weight).all():
        raise ArithmeticError("push-sum recurrence produced a non-finite state")
    return next_signal, next_weight


def push_sum(
    row_transition: ArrayLike,
    values: ArrayLike,
    *,
    rounds: int,
    lifted_nodes: int | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> FloatArray:
    iterations = _integer(rounds, name="rounds", minimum=0)
    transition = _validate_transition(
        row_transition,
        max_nodes=max_nodes,
        max_bytes=max_bytes,
    )
    if lifted_nodes is None:
        visible_nodes = int(transition.shape[0])
        if visible_nodes > max_nodes:
            raise ValueError(f"transition nodes exceed max_nodes={max_nodes}")
    else:
        visible_nodes = _integer(lifted_nodes, name="lifted_nodes", minimum=2)
        if visible_nodes > max_nodes:
            raise ValueError(f"lifted_nodes exceeds max_nodes={max_nodes}")
    payload = _validate_payload(
        values,
        nodes=visible_nodes,
        max_nodes=max_nodes,
        max_bytes=max_bytes,
    )
    signal, weight = _push_sum_state(
        transition,
        payload,
        lifted_nodes=lifted_nodes,
        max_bytes=max_bytes,
    )
    operator = transition.T
    for _ in range(iterations):
        signal, weight = _advance_push_sum(operator, signal, weight)
    estimates, _ = _estimates_and_mass(
        signal,
        weight,
        lifted_nodes=lifted_nodes,
    )
    return estimates


@dataclass(frozen=True, slots=True)
class ConsensusReceipt:
    """Canonical convergence evidence for one consensus execution."""

    topology: str
    rounds: int
    converged: bool
    tolerance: float
    max_error: float
    visible_nodes: int
    state_width: int
    lifted_nodes: int | None
    minimum_mass: float
    maximum_mass: float
    transition_sha256: str
    values_sha256: str
    target_sha256: str
    estimates_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "immer-ooe-consensus-receipt/v1",
            "topology": self.topology,
            "rounds": self.rounds,
            "converged": self.converged,
            "tolerance": self.tolerance,
            "max_error": self.max_error,
            "visible_nodes": self.visible_nodes,
            "state_width": self.state_width,
            "lifted_nodes": self.lifted_nodes,
            "minimum_mass": self.minimum_mass,
            "maximum_mass": self.maximum_mass,
            "transition_sha256": self.transition_sha256,
            "values_sha256": self.values_sha256,
            "target_sha256": self.target_sha256,
            "estimates_sha256": self.estimates_sha256,
        }


@dataclass(frozen=True, slots=True)
class ConsensusResult:
    estimates: FloatArray
    receipt: ConsensusReceipt


def _receipt(
    transition: FloatArray,
    payload: FloatArray,
    estimates: FloatArray,
    mass: FloatArray,
    *,
    rounds: int,
    tolerance: float,
    lifted_nodes: int | None,
    topology: str,
) -> ConsensusReceipt:
    target = payload.mean(axis=0, keepdims=True)
    error = float(np.max(np.abs(estimates - target)))
    return ConsensusReceipt(
        topology=topology,
        rounds=rounds,
        converged=error <= tolerance,
        tolerance=tolerance,
        max_error=error,
        visible_nodes=int(payload.shape[0]),
        state_width=int(payload.shape[1]),
        lifted_nodes=lifted_nodes,
        minimum_mass=float(mass.min()),
        maximum_mass=float(mass.max()),
        transition_sha256=array_sha256(transition),
        values_sha256=array_sha256(payload),
        target_sha256=array_sha256(target),
        estimates_sha256=array_sha256(estimates),
    )


def push_sum_with_receipt(
    row_transition: ArrayLike,
    values: ArrayLike,
    *,
    rounds: int,
    tolerance: float,
    lifted_nodes: int | None = None,
    topology: str | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> ConsensusResult:
    iterations = _integer(rounds, name="rounds", minimum=0)
    threshold = _finite_nonnegative(tolerance, name="tolerance")
    transition = _validate_transition(
        row_transition,
        max_nodes=max_nodes,
        max_bytes=max_bytes,
    )
    if lifted_nodes is None:
        visible_nodes = int(transition.shape[0])
        if visible_nodes > max_nodes:
            raise ValueError(f"transition nodes exceed max_nodes={max_nodes}")
    else:
        visible_nodes = _integer(lifted_nodes, name="lifted_nodes", minimum=2)
        if visible_nodes > max_nodes:
            raise ValueError(f"lifted_nodes exceeds max_nodes={max_nodes}")
    payload = _validate_payload(
        values,
        nodes=visible_nodes,
        max_nodes=max_nodes,
        max_bytes=max_bytes,
    )
    signal, weight = _push_sum_state(
        transition,
        payload,
        lifted_nodes=lifted_nodes,
        max_bytes=max_bytes,
    )
    operator = transition.T
    for _ in range(iterations):
        signal, weight = _advance_push_sum(operator, signal, weight)
    estimates, mass = _estimates_and_mass(
        signal,
        weight,
        lifted_nodes=lifted_nodes,
    )
    mode = topology or ("lifted" if lifted_nodes is not None else "reversible")
    if not isinstance(mode, str) or not mode:
        raise ValueError("topology must be a non-empty string")
    return ConsensusResult(
        estimates=estimates,
        receipt=_receipt(
            transition,
            payload,
            estimates,
            mass,
            rounds=iterations,
            tolerance=threshold,
            lifted_nodes=lifted_nodes,
            topology=mode,
        ),
    )


def measure_consensus(
    row_transition: ArrayLike,
    values: ArrayLike,
    *,
    tolerance: float,
    max_rounds: int,
    lifted_nodes: int | None = None,
    topology: str | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> ConsensusResult:
    """Advance once per round and return the first converged state and receipt."""

    threshold = _finite_nonnegative(tolerance, name="tolerance")
    limit = _integer(max_rounds, name="max_rounds", minimum=0)
    transition = _validate_transition(
        row_transition,
        max_nodes=max_nodes,
        max_bytes=max_bytes,
    )
    if lifted_nodes is None:
        visible_nodes = int(transition.shape[0])
        if visible_nodes > max_nodes:
            raise ValueError(f"transition nodes exceed max_nodes={max_nodes}")
    else:
        visible_nodes = _integer(lifted_nodes, name="lifted_nodes", minimum=2)
        if visible_nodes > max_nodes:
            raise ValueError(f"lifted_nodes exceeds max_nodes={max_nodes}")
    payload = _validate_payload(
        values,
        nodes=visible_nodes,
        max_nodes=max_nodes,
        max_bytes=max_bytes,
    )
    target = payload.mean(axis=0, keepdims=True)
    signal, weight = _push_sum_state(
        transition,
        payload,
        lifted_nodes=lifted_nodes,
        max_bytes=max_bytes,
    )
    operator = transition.T
    mode = topology or ("lifted" if lifted_nodes is not None else "reversible")
    if not isinstance(mode, str) or not mode:
        raise ValueError("topology must be a non-empty string")
    for rounds in range(limit + 1):
        estimates, mass = _estimates_and_mass(
            signal,
            weight,
            lifted_nodes=lifted_nodes,
        )
        if float(np.max(np.abs(estimates - target))) <= threshold:
            return ConsensusResult(
                estimates=estimates,
                receipt=_receipt(
                    transition,
                    payload,
                    estimates,
                    mass,
                    rounds=rounds,
                    tolerance=threshold,
                    lifted_nodes=lifted_nodes,
                    topology=mode,
                ),
            )
        if rounds < limit:
            signal, weight = _advance_push_sum(operator, signal, weight)
    raise RuntimeError("consensus did not converge within max_rounds")


def rounds_to_consensus(
    row_transition: ArrayLike,
    values: ArrayLike,
    *,
    tolerance: float,
    max_rounds: int,
    lifted_nodes: int | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> int:
    return measure_consensus(
        row_transition,
        values,
        tolerance=tolerance,
        max_rounds=max_rounds,
        lifted_nodes=lifted_nodes,
        max_nodes=max_nodes,
        max_bytes=max_bytes,
    ).receipt.rounds


def consensus_at_entry(
    row_transition: ArrayLike,
    values: ArrayLike,
    *,
    rounds: int,
    entry: int,
    lifted_nodes: int | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> FloatArray:
    estimates = push_sum(
        row_transition,
        values,
        rounds=rounds,
        lifted_nodes=lifted_nodes,
        max_nodes=max_nodes,
        max_bytes=max_bytes,
    )
    index = _integer(entry, name="entry", minimum=0)
    if index >= estimates.shape[0]:
        raise IndexError("entry is outside the visible consensus nodes")
    return np.ascontiguousarray(estimates[index])


def triplicate_median_consensus(
    row_transition: ArrayLike,
    values: ArrayLike,
    *,
    rounds: int,
    entry: int,
    lifted_nodes: int | None = None,
    poisoned_run: int | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> FloatArray:
    payload = np.asarray(values, dtype=np.float64)
    if payload.ndim not in {1, 2} or payload.shape[0] < 1:
        raise ValueError("values must be a non-empty vector or matrix")
    if not np.isfinite(payload).all():
        raise ValueError("values must be finite")
    if poisoned_run is not None and poisoned_run not in {0, 1, 2}:
        raise ValueError("poisoned_run must be null or one of 0, 1, 2")
    runs: list[FloatArray] = []
    for run in range(3):
        candidate = payload.copy()
        if poisoned_run == run:
            candidate[0] = np.roll(candidate[0], 1) * 1_000.0
        runs.append(
            consensus_at_entry(
                row_transition,
                candidate,
                rounds=rounds,
                entry=entry,
                lifted_nodes=lifted_nodes,
                max_nodes=max_nodes,
                max_bytes=max_bytes,
            )
        )
    return np.ascontiguousarray(np.median(np.stack(runs), axis=0))


@dataclass(frozen=True, slots=True)
class TopologyDecision:
    selected: str
    gap_threshold: float
    reversible_gap: float
    fiedler_eigenvalue: float
    adjacency_sha256: str
    reversible_diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "immer-ooe-topology-decision/v1",
            "selected": self.selected,
            "gap_threshold": self.gap_threshold,
            "reversible_gap": self.reversible_gap,
            "fiedler_eigenvalue": self.fiedler_eigenvalue,
            "adjacency_sha256": self.adjacency_sha256,
            "reversible_diagnostics": dict(self.reversible_diagnostics),
        }


@dataclass(frozen=True, slots=True)
class TopologyRouter:
    gap_threshold: float = 0.1
    max_nodes: int = DEFAULT_MAX_NODES
    max_bytes: int = DEFAULT_MAX_DENSE_BYTES

    def __post_init__(self) -> None:
        _finite_nonnegative(self.gap_threshold, name="gap_threshold", positive=True)
        _integer(self.max_nodes, name="max_nodes", minimum=2)
        _integer(self.max_bytes, name="max_bytes", minimum=1)

    def route(self, adjacency: ArrayLike) -> TopologyDecision:
        graph = validate_adjacency(
            adjacency,
            max_nodes=self.max_nodes,
            max_bytes=self.max_bytes,
        )
        reversible = metropolis_matrix(
            graph,
            max_nodes=self.max_nodes,
            max_bytes=self.max_bytes,
        )
        gap = spectral_gap(
            reversible,
            max_nodes=self.max_nodes,
            max_bytes=self.max_bytes,
        )
        fiedler, _ = fiedler_vector(
            graph,
            max_nodes=self.max_nodes,
            max_bytes=self.max_bytes,
        )
        selected = "lifted" if gap < self.gap_threshold else "reversible"
        return TopologyDecision(
            selected=selected,
            gap_threshold=float(self.gap_threshold),
            reversible_gap=gap,
            fiedler_eigenvalue=fiedler,
            adjacency_sha256=array_sha256(graph),
            reversible_diagnostics=kernel_diagnostics(
                reversible,
                max_nodes=self.max_nodes,
                max_bytes=self.max_bytes,
            ),
        )

    def choose(self, adjacency: ArrayLike) -> str:
        return self.route(adjacency).selected

    def transition(
        self,
        adjacency: ArrayLike,
        *,
        pc: float = 0.65,
        ps: float = 0.003,
    ) -> tuple[str, FloatArray, int | None]:
        decision = self.route(adjacency)
        if decision.selected == "lifted":
            graph = validate_adjacency(
                adjacency,
                max_nodes=self.max_nodes,
                max_bytes=self.max_bytes,
            )
            return (
                "lifted",
                ps_lifted_matrix(
                    graph,
                    pc=pc,
                    ps=ps,
                    max_nodes=self.max_nodes,
                    max_bytes=self.max_bytes,
                ),
                int(graph.shape[0]),
            )
        return (
            "reversible",
            metropolis_matrix(
                adjacency,
                max_nodes=self.max_nodes,
                max_bytes=self.max_bytes,
            ),
            None,
        )


__all__ = [
    "AdaptiveLiftParameters",
    "ConsensusMassError",
    "ConsensusReceipt",
    "ConsensusResult",
    "FloatArray",
    "TopologyDecision",
    "TopologyRouter",
    "adjacency_from_edges",
    "adaptive_lift_parameters",
    "adaptive_ps_lift_matrix",
    "barbell_adjacency",
    "complete_adjacency",
    "consensus_at_entry",
    "fiedler_eigenspace",
    "fiedler_vector",
    "measure_consensus",
    "metropolis_matrix",
    "ps_lifted_matrix",
    "push_sum",
    "push_sum_with_receipt",
    "rounds_to_consensus",
    "triplicate_median_consensus",
    "validate_adjacency",
]
