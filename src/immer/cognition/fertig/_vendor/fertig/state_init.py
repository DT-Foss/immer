"""Symbolic state initialization — Ginibre kernel statistics (F9, F11, F37).

Implements deterministic symbol state creation using hash-based signatures,
hyperboloid normalisation, and Ginibre spectral statistics.
"""

import numpy as np
from typing import List, Optional

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
GINIBRE_KERNEL: float = 1.08746866652609  # <s^2> from One Constant Rules All (F37)
D_SPATIAL: int = 16                       # F2: state-space spatial dimension
C_CURVATURE: float = 1.0                  # F3: hyperboloid curvature
LAMBDA_MIN: float = -0.99                 # Clamp: minimum coupling
LAMBDA_MAX: float = 0.99                  # Clamp: maximum coupling


# ===========================================================================
# F9: Primitive symbol state (hash-based, no embeddings)
# ===========================================================================

def _hash_signature(
    symbol: str,
    dim: int = D_SPATIAL,
) -> np.ndarray:
    """F9: Deterministic binary signature from string hash.

    b_i = sign(xxhash64(utf8(s_i)) mod 2^d - 2^{d-1})

    Purely deterministic — identical symbols always yield identical states.
    Falls back to Python's built-in hash if xxhash is unavailable.

    Parameters
    ----------
    symbol : str
        Symbol string to hash.
    dim : int
        Dimension of the signature vector.

    Returns
    -------
    signature : np.ndarray, shape (dim,)
        Binary signature vector in {-1, +1}^d.
    """
    try:
        import xxhash
        data = symbol.encode("utf-8")
        h = xxhash.xxh64(data).intdigest()
    except ImportError:
        # Deterministic fallback using Python's hash
        data = symbol.encode("utf-8")
        h = 0
        for b in data:
            h = ((h * 31) + b) & 0xFFFFFFFFFFFFFFFF

    # Derive d binary values from hash bits
    signature = np.zeros(dim, dtype=float)
    for k in range(dim):
        # Extract bit k and map to {-1, +1}
        bit = (h >> k) & 1
        signature[k] = 1.0 if bit == 1 else -1.0

    return signature


def primitive_symbol_state(
    symbol: str,
    dim: int = D_SPATIAL,
) -> np.ndarray:
    """F9: Create primitive symbol state vector (not a learned embedding).

    e_i = [1; b_i]  in R^{d+1}

    where b_i in {-1, 0, +1}^d is the binary signature vector.

    Parameters
    ----------
    symbol : str
        Symbol string.
    dim : int
        Spatial dimension (default 16).

    Returns
    -------
    state : np.ndarray, shape (dim + 1,)
        Primitive state vector [x^0; x^{sp}].
    """
    b = _hash_signature(symbol, dim)

    # x0 = sqrt(1 + ||b||^2) ensures <x, x>_M = -1 (approximately)
    x0 = np.sqrt(1.0 + np.dot(b, b))
    state = np.concatenate([[x0], b])

    return state


# ===========================================================================
# F11: Normalize to hyperboloid
# ===========================================================================

def normalize_to_hyperboloid(
    x: np.ndarray,
    c: float = C_CURVATURE,
) -> np.ndarray:
    """F11: Project state onto the hyperboloid H^d.

    x -> c * x / sqrt(-<x, x>_M)

    Enforces <x, x>_M = -c^2 for all valid states.

    Parameters
    ----------
    x : np.ndarray
        State vector [x^0; x^{sp}].
    c : float
        Curvature constant (default 1.0).

    Returns
    -------
    x_norm : np.ndarray
        Normalised state on H^d.
    """
    mink_sq = -(-x[0] * x[0] + np.dot(x[1:], x[1:]))
    if mink_sq <= 0:
        # Degenerate: perturb
        x = x.copy()
        x[0] = np.sqrt(1.0 + np.dot(x[1:], x[1:])) + 0.01
        mink_sq = -(-x[0] * x[0] + np.dot(x[1:], x[1:]))
    norm = np.sqrt(max(mink_sq, 1e-12))
    return c * x / norm


# ===========================================================================
# F37: Ginibre random matrix
# ===========================================================================

def ginibre_random_matrix(
    rows: int,
    cols: int,
    asymmetry: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate random matrix with 2D spectral (Ginibre) properties.

    The complex Ginibre ensemble has eigenvalue spacing following:
    P(s) = C_beta * s^beta * exp(-A_beta * s^2),  beta = 3

    with <s^2> = 1.08747... (GINIBRE_KERNEL).

    Parameters
    ----------
    rows : int
        Number of rows.
    cols : int
        Number of columns.
    asymmetry : float
        Asymmetry parameter in [0, 1].
        0 -> fully symmetric, 1 -> fully asymmetric.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    matrix : np.ndarray, shape (rows, cols)
        Random matrix with Ginibre-like spectral properties.
    """
    rng = np.random.default_rng(seed)

    # Real Gaussian entries (real Ginibre ensemble)
    M = rng.standard_normal((rows, cols))

    # Add asymmetry component
    if asymmetry > 0 and rows == cols:
        # Blend with anti-symmetric part
        sym = 0.5 * (M + M.T)
        anti = 0.5 * (M - M.T)
        M = (1.0 - asymmetry) * sym + asymmetry * anti

    # Scale to match Ginibre <s^2> statistics approximately
    # For the real Ginibre ensemble, <s^2> ~ 1.0; scale to target
    current_mean_sq = np.mean(M ** 2)
    if current_mean_sq > 1e-12:
        scale = np.sqrt(GINIBRE_KERNEL / current_mean_sq)
        M = M * scale * 0.1  # Additional damping for stability

    return M


# ===========================================================================
# F24: Sinkhorn renormalization (imported from moebius_core)
# ===========================================================================

def _sinkhorn(
    W: np.ndarray,
    n_iter: int = 100,
    tol: float = 1e-8,
) -> np.ndarray:
    """Sinkhorn renormalization — local copy to avoid circular import."""
    S = W.copy()
    S = np.maximum(S, 0.0)
    for _ in range(n_iter):
        row_sums = S.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
        S = S / row_sums
        col_sums = S.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums < 1e-12, 1.0, col_sums)
        S = S / col_sums
        row_err = np.abs(S.sum(axis=1) - 1.0)
        col_err = np.abs(S.sum(axis=0) - 1.0)
        if np.max(row_err) < tol and np.max(col_err) < tol:
            break
    return S


# ===========================================================================
# Symbol state matrix initialization
# ===========================================================================

def initialize_symbol_state(
    vocab_size: int,
    state_dim: int = D_SPATIAL,
) -> np.ndarray:
    """Initialize state matrix for all symbols.

    Returns a (vocab_size, state_dim + 1) array where each row is
    a primitive symbol state on the hyperboloid H^d.

    Uses deterministic hash-based signatures (F9) followed by
    hyperboloid normalisation (F11).

    Parameters
    ----------
    vocab_size : int
        Number of symbols in the vocabulary.
    state_dim : int
        Spatial dimension of each state (default 16).

    Returns
    -------
    state_matrix : np.ndarray, shape (vocab_size, state_dim + 1)
        State matrix where row i is the state for symbol i.
    """
    state_matrix = np.zeros((vocab_size, state_dim + 1), dtype=float)

    for i in range(vocab_size):
        symbol_str = f"__SYM_{i}__"
        # F9: Primitive state from hash
        state = primitive_symbol_state(symbol_str, state_dim)
        # F11: Normalise to hyperboloid
        state_matrix[i] = normalize_to_hyperboloid(state)

    return state_matrix


def state_for_symbol(
    symbol_id: int,
    state_matrix: np.ndarray,
) -> np.ndarray:
    """Lookup state for a symbol — O(1), no embedding!

    Parameters
    ----------
    symbol_id : int
        Symbol index.
    state_matrix : np.ndarray
        State matrix from initialize_symbol_state.

    Returns
    -------
    state : np.ndarray
        State vector for the symbol.
    """
    return state_matrix[symbol_id].copy()


# ===========================================================================
# F15: Build fidelity tensor
# ===========================================================================

def build_fidelity_tensor(
    state_matrix: np.ndarray,
) -> np.ndarray:
    """F15: Build the fidelity tensor for all symbol pairs.

    F_{ij} = |<x_i, x_j>_M|^(-1)

    Satisfies 0 < F_{ij} <= 1 with F_{ii} = 1.

    Parameters
    ----------
    state_matrix : np.ndarray, shape (N, d+1)
        State matrix for N symbols.

    Returns
    -------
    fidelity : np.ndarray, shape (N, N)
        Fidelity matrix.
    """
    N = state_matrix.shape[0]
    fidelity = np.zeros((N, N), dtype=float)

    for i in range(N):
        for j in range(N):
            x_i = state_matrix[i]
            x_j = state_matrix[j]
            inner = abs(-x_i[0] * x_j[0] + np.dot(x_i[1:], x_j[1:]))
            if inner < 1e-12:
                fidelity[i, j] = 1.0
            else:
                fidelity[i, j] = 1.0 / inner

    return fidelity


# ===========================================================================
# F10: Composite symbol state
# ===========================================================================

def composite_symbol_state(
    symbol_ids: List[int],
    state_matrix: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """F10: Compute composite state for a sequence of symbols.

    X(sigma) = (1/gamma_seq) * (+)_{k=1}^{L} x_{i_k}

    with weights w_k = g(lambda_k)^{-1} / sum_j g(lambda_j)^{-1}.

    Parameters
    ----------
    symbol_ids : List[int]
        Sequence of symbol indices.
    state_matrix : np.ndarray
        State matrix.
    weights : np.ndarray or None
        Optional per-symbol weights. If None, uses equal weights.

    Returns
    -------
    composite : np.ndarray
        Composite state vector.
    """
    if not symbol_ids:
        # Return reference state
        ref = np.zeros(state_matrix.shape[1])
        ref[0] = 1.0
        return ref

    states = np.stack([state_matrix[sid] for sid in symbol_ids])

    if weights is None:
        weights = np.ones(len(symbol_ids)) / len(symbol_ids)
    else:
        weights = np.asarray(weights)
        weights = weights / (np.sum(weights) + 1e-12)

    # Weighted Minkowski combination
    x0_comp = np.sum(weights * states[:, 0])
    xsp_comp = np.sum(weights[:, np.newaxis] * states[:, 1:], axis=0)

    composite = np.concatenate([[x0_comp], xsp_comp])

    # Renormalise to hyperboloid
    return normalize_to_hyperboloid(composite)


# ===========================================================================
# F12: Hyperbolic distance matrix
# ===========================================================================

def hyperbolic_distance_matrix(
    state_matrix: np.ndarray,
) -> np.ndarray:
    """F12: Compute pairwise hyperbolic distances.

    d_H(x_i, x_j) = arccosh(x_i^0 * x_j^0 - x_i^{sp} . x_j^{sp})

    Parameters
    ----------
    state_matrix : np.ndarray, shape (N, d+1)
        State matrix.

    Returns
    -------
    distances : np.ndarray, shape (N, N)
        Pairwise hyperbolic distance matrix.
    """
    N = state_matrix.shape[0]
    distances = np.zeros((N, N), dtype=float)

    for i in range(N):
        for j in range(i, N):
            x_i = state_matrix[i]
            x_j = state_matrix[j]
            inner = x_i[0] * x_j[0] - np.dot(x_i[1:], x_j[1:])
            inner = max(1.0, inner)
            d = float(np.arccosh(inner))
            distances[i, j] = d
            distances[j, i] = d

    return distances


# ===========================================================================
# F14: State-to-distribution projection
# ===========================================================================

def state_to_distribution(
    state: np.ndarray,
    state_matrix: np.ndarray,
    beta: float = 3.0,
) -> np.ndarray:
    """F14: Project a state to a probability distribution over symbols.

    P(s_j | x_i) = exp(-beta * d_H(x_i, e_j)) / sum_k exp(-beta * d_H(x_i, e_k))

    This is a deterministic function — no softmax is learned.

    Parameters
    ----------
    state : np.ndarray
        Query state vector.
    state_matrix : np.ndarray
        All symbol states.
    beta : float
        Inverse temperature (default 3.0 from F37).

    Returns
    -------
    probs : np.ndarray
        Probability distribution over symbols.
    """
    N = state_matrix.shape[0]
    scores = np.zeros(N)

    for j in range(N):
        e_j = state_matrix[j]
        inner = state[0] * e_j[0] - np.dot(state[1:], e_j[1:])
        inner = max(1.0, inner)
        d_h = float(np.arccosh(inner))
        scores[j] = -beta * d_h

    # Stable softmax
    scores_max = np.max(scores)
    exp_scores = np.exp(scores - scores_max)
    return exp_scores / (np.sum(exp_scores) + 1e-12)


# ===========================================================================
# F37: Verify Ginibre statistics
# ===========================================================================

def verify_ginibre_statistics(
    eigenvalue_spacings: np.ndarray,
    tolerance: float = 0.1,
) -> bool:
    """Verify that eigenvalue spacings match Ginibre statistics.

    <s^2> should be ~1.08747 for cubic repulsion (beta = 3).

    Parameters
    ----------
    eigenvalue_spacings : np.ndarray
        Nearest-neighbor eigenvalue spacings.
    tolerance : float
        Allowed deviation from GINIBRE_KERNEL.

    Returns
    -------
    ok : bool
        True if statistics match within tolerance.
    """
    if len(eigenvalue_spacings) == 0:
        return False
    mean_s2 = float(np.mean(eigenvalue_spacings ** 2))
    return abs(mean_s2 - GINIBRE_KERNEL) < tolerance


# ===========================================================================
# Initialisation with Sinkhorn renormalization
# ===========================================================================

def initialize_symbol_state_with_sinkhorn(
    vocab_size: int,
    state_dim: int = D_SPATIAL,
) -> np.ndarray:
    """Initialize symbol state matrix with Sinkhorn renormalization.

    1. Create primitive states via hash signatures (F9)
    2. Normalise to hyperboloid (F11)
    3. Build transition matrix from state similarities
    4. Apply Sinkhorn renormalization (F24)
    5. Renormalise states

    Parameters
    ----------
    vocab_size : int
        Number of symbols.
    state_dim : int
        Spatial dimension (default 16).

    Returns
    -------
    state_matrix : np.ndarray, shape (vocab_size, state_dim + 1)
        Fully initialised and renormalised state matrix.
    """
    # Step 1-2: Primitive states on hyperboloid
    state_matrix = initialize_symbol_state(vocab_size, state_dim)

    # Step 3: Build similarity-based transition matrix
    W = np.zeros((vocab_size, vocab_size))
    for i in range(vocab_size):
        for j in range(vocab_size):
            x_i = state_matrix[i]
            x_j = state_matrix[j]
            # F19: Pairwise velocity as similarity
            v_ij = (-x_i[0] * x_j[0] + np.dot(x_i[1:], x_j[1:])) / max(x_i[0] * x_j[0], 1e-12)
            # F26: Composition coupling
            inner = x_i[0] * x_j[0] - np.dot(x_i[1:], x_j[1:])
            inner = max(1.0, inner)
            d_h = float(np.arccosh(inner))
            lam_ij = np.tanh(d_h / np.arccosh(state_dim + 1))
            W[i, j] = abs(v_ij) / np.sqrt(max(1.0 - lam_ij ** 2, 1e-12))

    # Ensure non-negative
    W = np.maximum(W, 0.0)

    # Step 4: Sinkhorn renormalization
    W_ds = _sinkhorn(W)

    # Step 5: Renormalise states using the doubly-stochastic weights
    for i in range(vocab_size):
        # Weighted renormalisation
        weights = W_ds[i]
        weighted = weights[:, np.newaxis] * state_matrix
        new_state = np.sum(weighted, axis=0)
        state_matrix[i] = normalize_to_hyperboloid(new_state)

    return state_matrix
