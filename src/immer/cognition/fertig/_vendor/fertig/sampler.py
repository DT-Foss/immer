"""Contraction sampler — tau-controlled deterministic generation (F33–F40).

Implements the contraction-governed sampling distribution,
BvN decomposition, Zeno schedule, and Ginibre-kernel weighting.
"""

import numpy as np
from typing import List, Tuple, Optional

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
TAU_MAX: float = 0.95          # F35: maximum contraction before phase transition
T0_TEMP: float = 1.0           # F36: initial sampling temperature
BETA_INV_TEMP: float = 3.0     # F37: cubic repulsion inverse temperature
GINIBRE_A: float = 1.2         # F38: Ginibre exponential constant
THETA_F: float = 0.30          # F36: fidelity threshold for sampling
K_STAR: int = 5                # Zeno schedule: anti-Zeno at k* multiples
C_SPARSITY: float = 9.9        # F64: sparsity conversion constant


# ===========================================================================
# F36: tau-to-temperature conversion
# ===========================================================================

def tau_to_temperature(tau: float) -> float:
    """F36 (derived): Convert contraction coefficient to temperature.

    T(tau) = (1 - tau^2)^(-1/2) - 1

    This maps tau in [0, 0.95] to a temperature suitable for
    Boltzmann-style sampling. At tau = 0, T = 0 (deterministic).
    As tau -> 1, T -> infinity (uniform).
    """
    tau_safe = min(abs(tau), 0.999999)
    return float((1.0 / np.sqrt(1.0 - tau_safe ** 2)) - 1.0)


# ===========================================================================
# Contraction sampling
# ===========================================================================

def contraction_sample(
    logits: np.ndarray,
    tau: float = 0.65,
    top_k: int = 50,
) -> int:
    """Sample token with tau-controlled temperature.

    Parameters
    ----------
    logits : np.ndarray, shape (vocab_size,)
        Raw logit scores for each token.
    tau : float
        Contraction coefficient controlling randomness.
        tau = 0   -> deterministic (argmax)
        tau = 0.95 -> nearly uniform.
    top_k : int
        Number of top candidates to consider.

    Returns
    -------
    token_id : int
        Selected token index.
    """
    # Check criticality (F35)
    if tau >= 1.0:
        raise ValueError("Phase transition: tau >= 1.0")

    # Convert tau to temperature
    T = tau_to_temperature(tau)

    # Top-k filtering
    if top_k > 0 and top_k < len(logits):
        top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
        mask = np.full_like(logits, -np.inf)
        mask[top_k_indices] = logits[top_k_indices]
        filtered_logits = mask
    else:
        filtered_logits = logits.copy()

    # Temperature scaling
    if T > 1e-6:
        scaled = filtered_logits / (T + 1e-6)
    else:
        # Deterministic: return argmax
        return int(np.argmax(filtered_logits))

    # Numerically stable softmax
    scaled_max = np.max(scaled)
    exp_scaled = np.exp(scaled - scaled_max)
    probs = exp_scaled / (np.sum(exp_scaled) + 1e-12)

    # Deterministic selection via cumulative probability + hash
    # (avoids stochastic sampling for reproducibility)
    cumsum = np.cumsum(probs)
    # Use a deterministic pseudo-random value based on logits hash
    det_value = float(np.abs(np.sum(logits * 0.123456789)) % 1.0)
    token_id = int(np.searchsorted(cumsum, det_value))
    token_id = min(token_id, len(probs) - 1)

    return token_id


# ===========================================================================
# F35: Zeno schedule
# ===========================================================================

def zeno_schedule(
    step: int,
    tau_default: float = 0.65,
    k_star: int = K_STAR,
) -> float:
    """Adaptive tau: anti-Zeno at k* multiples, Zeno otherwise.

    Anti-Zeno effect (step % k_star == 0):
        tau = tau_default * 1.2   (more exploration)

    Zeno effect (otherwise):
        tau = tau_default * 0.9   (more exploitation)

    Parameters
    ----------
    step : int
        Current generation step.
    tau_default : float
        Base contraction coefficient (default 0.65).
    k_star : int
        Anti-Zeno period (default 5).

    Returns
    -------
    tau : float
        Scheduled contraction coefficient, clamped to [0, 0.95].
    """
    if step % k_star == 0 and step > 0:
        # Anti-Zeno: increase tau (more randomness)
        tau = tau_default * 1.2
    else:
        # Zeno: decrease tau (more determinism)
        tau = tau_default * 0.9

    return float(np.clip(tau, 0.0, TAU_MAX))


# ===========================================================================
# F38: Ginibre kernel selection weighting
# ===========================================================================

def ginibre_select_weights(
    distances: np.ndarray,
    d_bar: Optional[float] = None,
) -> np.ndarray:
    """F38: Ginibre kernel symbol-selection weights.

    s_j = d_H(x_ctx, e_j) / d_bar
    w_j = s_j^3 * exp(-1.2 * s_j^2)

    P(s_j) = w_j / sum_k w_k

    Parameters
    ----------
    distances : np.ndarray
        Hyperbolic distances from context state to each symbol state.
    d_bar : float or None
        Mean distance for normalisation. If None, uses median.

    Returns
    -------
    weights : np.ndarray
        Normalised selection probabilities.
    """
    if d_bar is None:
        d_bar = float(np.median(distances)) if len(distances) > 0 else 1.0
    d_bar = max(d_bar, 1e-12)

    s = distances / d_bar
    w = s ** 3 * np.exp(-GINIBRE_A * s ** 2)
    w_sum = np.sum(w)
    if w_sum < 1e-12:
        # Uniform fallback
        return np.ones_like(w) / len(w)
    return w / w_sum


# ===========================================================================
# F63: Birkhoff-von Neumann decomposition
# ===========================================================================

def _hungarian_min(
    cost_matrix: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Find minimum-weight perfect matching (assignment problem).

    Uses the classic Hungarian algorithm for the assignment problem.
    Returns the permutation matrix and minimum entry weight.
    """
    n = cost_matrix.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float), 0.0

    # Use scipy if available, else simple greedy fallback
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        P = np.zeros_like(cost_matrix)
        P[row_ind, col_ind] = 1.0
        min_entry = float(np.min(cost_matrix[row_ind, col_ind]))
        return P, min_entry
    except ImportError:
        # Greedy fallback
        P = np.zeros_like(cost_matrix)
        remaining_rows = list(range(n))
        remaining_cols = list(range(n))
        min_entry = float('inf')

        for _ in range(n):
            best_val = float('inf')
            best_pair = (0, 0)
            for r in remaining_rows:
                for c in remaining_cols:
                    if cost_matrix[r, c] < best_val:
                        best_val = cost_matrix[r, c]
                        best_pair = (r, c)
            r, c = best_pair
            P[r, c] = 1.0
            min_entry = min(min_entry, best_val)
            remaining_rows.remove(r)
            remaining_cols.remove(c)

        return P, min_entry


def bvn_decompose(
    prob_matrix: np.ndarray,
    max_paths: int = 100,
    tol: float = 1e-8,
) -> Tuple[List[np.ndarray], List[float]]:
    """F63: Birkhoff-von Neumann decomposition.

    W^{DS} = sum_{i=1}^{M} w_i * P_i

    where P_i are permutation matrices, w_i > 0, sum_i w_i = 1,
    and M <= N^2 - 2N + 2.

    Parameters
    ----------
    prob_matrix : np.ndarray, shape (N, N)
        Doubly-stochastic matrix to decompose.
    max_paths : int
        Maximum number of permutation paths to extract.
    tol : float
        Convergence tolerance for remaining mass.

    Returns
    -------
    permutations : List[np.ndarray]
        List of permutation matrices.
    weights : List[float]
        Corresponding weights (sum to 1).
    """
    N = prob_matrix.shape[0]
    remaining = prob_matrix.copy()
    permutations: List[np.ndarray] = []
    weights: List[float] = []

    for _ in range(max_paths):
        total_mass = np.sum(remaining)
        if total_mass < tol:
            break

        # Convert to cost matrix for Hungarian
        # We want to find the matching with maximum remaining weight
        # so negate (or use 1 - remaining for minimisation)
        cost = np.max(remaining) - remaining

        P, _ = _hungarian_min(cost)

        # Find the minimum non-zero entry in the matching
        entries = remaining * P
        mask = entries > 1e-12
        if not np.any(mask):
            break
        w_i = float(np.min(entries[mask]))

        permutations.append(P)
        weights.append(w_i)
        remaining -= w_i * P
        remaining = np.maximum(remaining, 0.0)

    # Normalise weights
    w_sum = sum(weights)
    if w_sum > 0:
        weights = [w / w_sum for w in weights]

    return permutations, weights


def bvn_path_integral_sample(
    logits: np.ndarray,
    n_candidates: int = 10,
) -> int:
    """Sample using BvN path evaluation.

    Builds a transition matrix from logits, decomposes into
    permutation paths, and selects the token via path-weighted
    cumulative evaluation.

    Parameters
    ----------
    logits : np.ndarray, shape (vocab_size,)
        Raw logit scores.
    n_candidates : int
        Number of top candidates for BvN evaluation.

    Returns
    -------
    token_id : int
        Selected token index.
    """
    vocab_size = len(logits)

    # Build a doubly-stochastic transition matrix from logits
    # Use top-n candidates
    top_k = min(n_candidates, vocab_size)
    top_indices = np.argpartition(logits, -top_k)[-top_k:]

    # Sub-matrix for BvN
    sub_logits = logits[top_indices]
    sub_matrix = np.outer(sub_logits, sub_logits)
    sub_matrix = np.abs(sub_matrix)
    sub_matrix = sub_matrix / (np.sum(sub_matrix) + 1e-12)

    # Make doubly-stochastic via Sinkhorn
    S = sub_matrix.copy()
    for _ in range(20):
        row_sums = S.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
        S = S / row_sums
        col_sums = S.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums < 1e-12, 1.0, col_sums)
        S = S / col_sums

    # BvN decompose
    perms, weights = bvn_decompose(S, max_paths=min(20, top_k))

    if not weights:
        # Fallback: argmax
        return int(top_indices[np.argmax(sub_logits)])

    # Evaluate each path and take weighted consensus
    path_scores = np.zeros(top_k)
    for P, w in zip(perms, weights):
        path_scores += w * P.sum(axis=1)

    best_idx = int(np.argmax(path_scores))
    return int(top_indices[best_idx])


# ===========================================================================
# F64: Path counting formula
# ===========================================================================

def path_count(N: int, c_sparsity: float = C_SPARSITY) -> int:
    """F64: Number of permutation paths in BvN decomposition.

    M_paths = round(N^2 / c_sparsity)

    For N = 64 and c_sparsity = 9.9: M_paths = 414.
    """
    return int(round(N * N / c_sparsity))


def convergence_rounds(
    N: int,
    c_sparsity: float = C_SPARSITY,
    m_per_round: float = 37.6,
) -> int:
    """F65: Rounds for BvN convergence.

    R_BvN = ceil(M_paths / M_per_round) = 11 rounds

    for M_per_round = 37.6 paths per round.
    """
    m_paths = path_count(N, c_sparsity)
    return int(np.ceil(m_paths / m_per_round))


# ===========================================================================
# F67: Consensus convergence rate
# ===========================================================================

def consensus_convergence_bound(
    t: int,
    lambda_2: float,
    M: int,
    initial_error: float = 1.0,
) -> float:
    """F67: Gossip consensus convergence rate bound.

    ||x^{(t)} - x*||_2 <= (1 - lambda_2 / M)^t * ||x^{(0)} - x*||_2

    where x* is the consensus fixed point.
    """
    rate = max(0.0, 1.0 - lambda_2 / max(M, 1))
    return initial_error * (rate ** t)


# ===========================================================================
# F68: Weight optimisation via fidelity
# ===========================================================================

def fidelity_path_weights(
    path_fidelities: np.ndarray,
    beta: float = BETA_INV_TEMP,
) -> np.ndarray:
    """F68: Optimise path weights via fidelity.

    w_i* = exp(beta * F_path,i) / sum_j exp(beta * F_path,j)

    where F_path,i = (1/L_i) * sum_{(k,l) in path_i} F_{kl}
    is the average fidelity along path i.
    """
    scaled = beta * path_fidelities
    # Numerically stable softmax
    scaled_max = np.max(scaled)
    exp_scaled = np.exp(scaled - scaled_max)
    return exp_scaled / (np.sum(exp_scaled) + 1e-12)


# ===========================================================================
# F36: Contraction-governed sampling distribution
# ===========================================================================

def contraction_governed_distribution(
    distances: np.ndarray,
    fidelities: np.ndarray,
    tau: float,
    theta_f: float = THETA_F,
) -> np.ndarray:
    """F36: Compute sampling distribution with contraction control.

    P(s_j | context) ~ exp(-d_H(x_ctx, e_j) / T_t) * 1[F_ctx,j >= theta_F]

    where T_t = T_0 * tau^t is the annealed temperature, T_0 = 1.0,
    and theta_F = 0.30 is the fidelity threshold.

    Parameters
    ----------
    distances : np.ndarray
        Hyperbolic distances from context to each symbol.
    fidelities : np.ndarray
        Fidelity values for each symbol.
    tau : float
        Contraction coefficient.
    theta_f : float
        Fidelity threshold.

    Returns
    -------
    probs : np.ndarray
        Normalised sampling probabilities.
    """
    # Temperature
    T = tau_to_temperature(tau) + 1e-6

    # Distance-scored probabilities
    scores = np.exp(-distances / T)

    # Fidelity threshold filter
    mask = fidelities >= theta_f
    scores = np.where(mask, scores, 0.0)

    # Normalise
    total = np.sum(scores)
    if total < 1e-12:
        # Uniform fallback
        return np.ones_like(scores) / len(scores)
    return scores / total


# ===========================================================================
# F35: Phase transition check
# ===========================================================================

def check_phase_transition(tau: float) -> bool:
    """F35: Check if system is at criticality.

    At tau = 1, the system undergoes a phase transition:
    F(tau) ~ F_crit - A * (1 - tau)^nu,  nu = 1/2

    The sampler must maintain tau <= tau_max = 0.95.
    """
    return tau >= 1.0


# ===========================================================================
# Combined sampler with Zeno schedule
# ===========================================================================

def generate_step(
    logits: np.ndarray,
    step: int,
    tau_default: float = 0.65,
    top_k: int = 50,
    use_zeno: bool = True,
) -> int:
    """Single generation step with Zeno-scheduled contraction control.

    Parameters
    ----------
    logits : np.ndarray
        Raw logit scores.
    step : int
        Current generation step.
    tau_default : float
        Base contraction coefficient.
    top_k : int
        Top-k filtering parameter.
    use_zeno : bool
        Whether to use the Zeno schedule.

    Returns
    -------
    token_id : int
        Selected token index.
    """
    if use_zeno:
        tau = zeno_schedule(step, tau_default)
    else:
        tau = tau_default

    # Check phase transition
    if check_phase_transition(tau):
        raise ValueError("Phase transition detected: tau >= 1.0")

    return contraction_sample(logits, tau=tau, top_k=top_k)
