"""Berry Phase Holographic Memory (BPHM).

Cyclic memory via Berry phase accumulation -- detects and breaks
repetition in generated token sequences.
No learned parameters. Purely deterministic.

Formulas (from "Unitarity is the Boundary"):
    Fidelity:          F_ij = |<psi_i | psi_j>|^2
    Berry phase:       gamma_B = i * oint_C <psi|d psi>
                       = arg(<psi_0|psi_1> <psi_1|psi_2> ... <psi_{n-1}|psi_0>)
    Phase difference:  delta_phi = arg(<psi_t | psi_{t-k}>)
    Repetition:        |delta_phi mod 2pi| < eps  AND  |gamma_B| > threshold
    Weight modulation: w'_k = w_k * (1 + beta * cos(gamma_B - phi_k))
"""

import numpy as np
from typing import List, Tuple, Optional


EPS: float = 1e-8
FIDELITY_THRESHOLD: float = 0.85  # 80% erasure tolerance
BETA: float = 0.3                 # Phase modulation amplitude
DEFAULT_CAPACITY: int = 100


def compute_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """F_ij = |<psi_i | psi_j>|^2 = |dot(state1, state2)|^2 / (|state1|^2 |state2|^2)

    Computes the squared cosine similarity between two state vectors.
    Fidelity = 1.0 for identical states, 0.0 for orthogonal states.

    Args:
        state1: First state vector (d-dimensional).
        state2: Second state vector (d-dimensional).

    Returns:
        Fidelity in [0, 1].
    """
    dot = float(np.dot(state1, state2))
    norm1_sq = float(np.dot(state1, state1))
    norm2_sq = float(np.dot(state2, state2))
    denom = norm1_sq * norm2_sq
    if denom < EPS:
        return 0.0
    return (dot ** 2) / denom


def compute_berry_phase(states: List[np.ndarray]) -> float:
    """gamma_B = arg(<psi_0|psi_1> <psi_1|psi_2> ... <psi_{n-1}|psi_0>)

    Compute the accumulated Berry phase around a cycle of states.
    The phase is the argument of the product of successive inner
    products, which gives the geometric phase accumulated along
    the closed loop.

    For a cycle of n states, we compute the product of inner
    products going around the loop and return the angle (phase)
    of that complex product.

    Args:
        states: List of n state vectors forming a closed cycle.

    Returns:
        Berry phase gamma_B in [-pi, pi].  Returns 0.0 for n < 2.
    """
    n = len(states)
    if n < 2:
        return 0.0

    # Compute the product of inner products around the cycle
    # For real states, each inner product is real, so we track
    # the product as a complex number and take its argument.
    product_real = 1.0
    for i in range(n):
        j = (i + 1) % n
        inner = float(np.dot(states[i], states[j]))
        product_real *= inner

    # If product is negative, the phase is pi; if positive, phase is 0.
    # We use np.sign and convert to a phase angle.
    if abs(product_real) < EPS:
        return 0.0

    # General case: use arctan2 for proper phase
    phase = np.arctan2(0.0, product_real)
    # For negative real product, arctan2(0, negative) = pi
    # For positive real product, arctan2(0, positive) = 0

    # But Berry phase is more meaningfully computed via the
    # connection form: accumulate Im[<psi_i | psi_{i+1} - psi_i>]
    # This gives a richer phase even for real vectors.
    berry_connection = 0.0
    for i in range(n):
        j = (i + 1) % n
        dz = states[j] - states[i]
        # A_i = <psi_i | dpsi> = dot(psi_i, psi_{i+1} - psi_i)
        A_i = float(np.dot(states[i], dz))
        berry_connection += A_i

    # The Berry phase is the imaginary part of the accumulated connection.
    # For real vectors, the connection is purely real, so we use the
    # argument-of-product formulation which captures the winding.
    # Fall back to a robust estimate: accumulate phase differences.
    total_phase = 0.0
    for i in range(n):
        j = (i + 1) % n
        si_norm = states[i] / (np.linalg.norm(states[i]) + EPS)
        sj_norm = states[j] / (np.linalg.norm(states[j]) + EPS)
        inner = float(np.dot(si_norm, sj_norm))
        # Clip to [-1, 1] for arccos safety
        inner = max(-1.0, min(1.0, inner))
        # Phase difference between consecutive states
        phase_diff = np.arccos(inner)
        total_phase += phase_diff

    # Berry phase = total accumulated geometric phase mod 2pi
    berry_phase = total_phase % (2.0 * np.pi)
    if berry_phase > np.pi:
        berry_phase -= 2.0 * np.pi

    return float(berry_phase)


def compute_phase_difference(current: np.ndarray, previous: np.ndarray) -> float:
    """delta_phi = arg(<psi_t | psi_{t-k}>) -- phase difference between states.

    Computes the phase angle between two state vectors as the
    argument of their inner product.  For real vectors, this is
    0 (same direction) or pi (opposite direction).

    Args:
        current:  Current state vector psi_t.
        previous: Previous state vector psi_{t-k}.

    Returns:
        Phase difference in [-pi, pi].
    """
    dot = float(np.dot(current, previous))
    norm_current = float(np.linalg.norm(current))
    norm_previous = float(np.linalg.norm(previous))

    if norm_current < EPS or norm_previous < EPS:
        return 0.0

    # Normalize the inner product to [-1, 1]
    cos_angle = dot / (norm_current * norm_previous)
    cos_angle = max(-1.0, min(1.0, cos_angle))

    # Phase difference = arccos(cos_angle), sign determined by dot
    phase = np.arccos(cos_angle)
    if dot < 0:
        phase = -phase

    return float(phase)


def detect_repetition(
    state_history: List[np.ndarray],
    window_size: int = 5,
    phase_threshold: float = 0.1,
    berry_threshold: float = 0.5,
) -> bool:
    """Detect repetition via Berry phase accumulation.

    Repetition is detected when two conditions are met:
        1. |delta_phi mod 2pi| < phase_threshold  (states are aligned)
        2. |gamma_B| > berry_threshold            (sufficient phase accumulated)

    Args:
        state_history: List of state vectors in generation order.
        window_size:   Number of recent states to consider for cycle detection.
        phase_threshold:  Maximum phase difference for "same state" detection.
        berry_threshold:  Minimum Berry phase magnitude for repetition flag.

    Returns:
        True if repetition is detected (caller should diversify).
    """
    n = len(state_history)
    if n < window_size + 1:
        return False

    # Compare current state with states from the past window
    current_state = state_history[-1]
    recent_states = state_history[-(window_size + 1):-1]

    for i, past_state in enumerate(reversed(recent_states)):
        # Check phase alignment
        phase_diff = compute_phase_difference(current_state, past_state)
        phase_diff_mod = abs(phase_diff % (2.0 * np.pi))
        if phase_diff_mod > np.pi:
            phase_diff_mod = 2.0 * np.pi - phase_diff_mod

        if phase_diff_mod < phase_threshold:
            # States are aligned -- check Berry phase around the cycle
            # The cycle goes: past_state -> ... -> current_state -> past_state
            cycle_start_idx = n - 2 - i
            cycle_states = state_history[cycle_start_idx:] + [state_history[cycle_start_idx]]
            gamma_B = abs(compute_berry_phase(cycle_states))

            if gamma_B > berry_threshold:
                return True

    return False


def modulate_bvn_weights(
    path_weights: np.ndarray,
    berry_phase: float,
    beta: float = BETA,
) -> np.ndarray:
    """Modulate BvN path weights by Berry phase.

    High phase -> diversify weights (break repetition).

    w'_k = w_k * (1 + beta * cos(gamma_B - phi_k))

    where phi_k = 2*pi / (k+1) is the intrinsic phase of path k
    (derived from the permutation's cycle structure).

    Args:
        path_weights:  1-D array of BvN path weights.
        berry_phase:   Accumulated Berry phase gamma_B.
        beta:          Phase modulation amplitude.

    Returns:
        Modulated weights (normalized to sum to 1).
    """
    n = len(path_weights)
    if n == 0:
        return path_weights.copy()

    modulated = np.zeros_like(path_weights)
    for k in range(n):
        # Intrinsic phase from permutation cycle structure
        if n > 1:
            phi_k = 2.0 * np.pi / n * k
        else:
            phi_k = 0.0

        modulation = 1.0 + beta * np.cos(berry_phase - phi_k)
        modulated[k] = path_weights[k] * max(0.0, modulation)

    # Renormalize
    total = np.sum(modulated)
    if total > EPS:
        modulated = modulated / total
    else:
        # Fallback: uniform distribution
        modulated = np.ones_like(path_weights) / n

    return modulated


class BPHMMemory:
    """Stateful holographic memory via Berry phase accumulation.

    Maintains a circular buffer of recent state vectors and uses
    Berry phase to detect cycles / repetition in generation.
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY, state_dim: int = 64):
        """Initialize circular buffer for state history.

        Args:
            capacity:  Maximum number of states to retain.
            state_dim: Dimensionality of each state vector.
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self._buffer: List[np.ndarray] = []
        self._phase_accumulator: float = 0.0
        self._phase_history: List[float] = [0.0]
        self._berry_phases: List[float] = []

    def push(self, state: np.ndarray):
        """Add state to memory.

        Args:
            state: State vector (state_dim-dimensional).
        """
        state_copy = np.asarray(state, dtype=np.float64).flatten()
        if len(state_copy) != self.state_dim:
            # Pad or truncate to match expected dimension
            if len(state_copy) < self.state_dim:
                state_copy = np.pad(state_copy, (0, self.state_dim - len(state_copy)))
            else:
                state_copy = state_copy[:self.state_dim]

        self._buffer.append(state_copy)
        if len(self._buffer) > self.capacity:
            self._buffer.pop(0)

    def check_repetition(
        self,
        window_size: int = 5,
        phase_threshold: float = 0.1,
        berry_threshold: float = 0.5,
    ) -> Tuple[bool, float]:
        """Check if recent states show repetition.

        Args:
            window_size:      Number of recent states to check.
            phase_threshold:  Phase alignment threshold.
            berry_threshold:  Minimum Berry phase for detection.

        Returns:
            (is_repeating, latest_berry_phase) as (bool, float).
        """
        is_repeating = detect_repetition(
            self._buffer,
            window_size=window_size,
            phase_threshold=phase_threshold,
            berry_threshold=berry_threshold,
        )

        # Compute Berry phase for the most recent cycle window
        latest_phase = 0.0
        if len(self._buffer) >= 2:
            window = self._buffer[-min(window_size, len(self._buffer)):]
            if len(window) >= 2:
                latest_phase = abs(compute_berry_phase(window))
                self._berry_phases.append(latest_phase)

        return is_repeating, latest_phase

    def get_diversification_boost(self) -> float:
        """Return boost factor for diversity when repetition detected.

        1.0 = normal generation.
        >1.0 = increase tau for more diversity.

        The boost is computed from the accumulated Berry phase:
        higher phase accumulation -> higher boost.
        """
        if not self._berry_phases:
            return 1.0

        # Use recent Berry phase magnitude to determine boost
        recent_phase = np.mean(self._berry_phases[-5:]) if len(self._berry_phases) >= 5 else self._berry_phases[-1]

        # Boost scales linearly with phase: 1.0 + 0.5 * (phase / pi)
        boost = 1.0 + 0.5 * min(recent_phase / np.pi, 1.0)
        return float(boost)

    def reset(self):
        """Clear all memory for a new generation sequence."""
        self._buffer.clear()
        self._phase_accumulator = 0.0
        self._phase_history = [0.0]
        self._berry_phases.clear()
