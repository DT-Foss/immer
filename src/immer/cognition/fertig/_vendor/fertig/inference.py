"""Deterministic transitive inference — exact, semantic, fuzzy (F49–F56).

Three-pass deterministic inference engine:
  Pass 1: Exact keyword chaining
  Pass 2: Semantic direction propagation
  Pass 3: Jaro-Winkler fuzzy matching

Confidence propagation via Moebius coupling (F30, F52, F53).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
GAMMA_CHAIN: float = 0.85     # F29, F52: confidence chain decay
THETA_Q: float = 0.30         # F31, F54: quality filter threshold
THETA_JW: float = 0.85        # F51: Jaro-Winkler acceptance threshold
THETA_SEMANTIC: float = 0.85  # F50: semantic match threshold
ALPHA_AMP_MIN: int = 7        # F32: minimum amplification
ALPHA_AMP_MAX: int = 10       # F32: maximum amplification


# ===========================================================================
# F51: Jaro-Winkler similarity
# ===========================================================================

def _matching_chars(s1: str, s2: str) -> int:
    """Count matching characters within match distance."""
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0

    match_distance = max(len1, len2) // 2 - 1
    match_distance = max(match_distance, 0)

    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(len2, i + match_distance + 1)
        for j in range(start, end):
            if not s2_matches[j] and s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

    return matches


def _transpositions(s1: str, s2: str, s1_matches: List[bool], s2_matches: List[bool]) -> int:
    """Count transpositions among matched characters."""
    len1, len2 = len(s1), len(s2)
    s1_matched = [s1[i] for i in range(len1) if s1_matches[i]]
    s2_matched = [s2[j] for j in range(len2) if s2_matches[j]]

    transpositions = sum(1 for a, b in zip(s1_matched, s2_matched) if a != b)
    return transpositions // 2


def jaro_similarity(s1: str, s2: str) -> float:
    """Compute Jaro similarity between two strings.

    Jaro(s1, s2) = (1/3) * (m/|s1| + m/|s2| + (m-t)/m)

    where m = matching characters, t = transpositions.
    """
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = max(len1, len2) // 2 - 1
    match_distance = max(match_distance, 0)

    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(len2, i + match_distance + 1)
        for j in range(start, end):
            if not s2_matches[j] and s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    # Count transpositions
    s1_matched = [s1[i] for i in range(len1) if s1_matches[i]]
    s2_matched = [s2[j] for j in range(len2) if s2_matches[j]]
    transpositions = sum(1 for a, b in zip(s1_matched, s2_matched) if a != b) // 2

    jaro = (1.0 / 3.0) * (
        matches / len1 +
        matches / len2 +
        (matches - transpositions) / matches
    )
    return float(jaro)


def jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    """F51: Jaro-Winkler similarity.

    Jaro-Winkler = Jaro + l * p * (1 - Jaro)

    where l = common prefix length (max 4), p = 0.1.
    Acceptance threshold: >= 0.85.
    """
    jaro = jaro_similarity(s1, s2)

    # Common prefix length (max 4)
    l = 0
    for i in range(min(4, len(s1), len(s2))):
        if s1[i] == s2[i]:
            l += 1
        else:
            break

    jw = jaro + l * p * (1.0 - jaro)
    return float(min(jw, 1.0))


# ===========================================================================
# F30: Moebius confidence combination
# ===========================================================================

def moebius_confidence(c1: float, c2: float) -> float:
    """F30: Moebius confidence combination.

    f(c1, c2) = (c1 + c2) / (1 + c1 * c2)

    Prevents decay at hubs — preserves confidence through
    high-degree nodes better than simple multiplication.
    """
    denom = 1.0 + c1 * c2
    if abs(denom) < 1e-12:
        return 1.0
    return min(1.0, max(0.0, (c1 + c2) / denom))


# ===========================================================================
# F52: Confidence chain product
# ===========================================================================

def chain_confidence(confidences: List[float], gamma: float = GAMMA_CHAIN) -> float:
    """F52: Chain confidence with decay.

    c = (prod_i c_i) * gamma^(n-1)

    where gamma = 0.85 is the confidence decay constant.
    """
    if not confidences:
        return 1.0
    product = 1.0
    for c in confidences:
        product *= max(0.0, min(1.0, c))
    n = len(confidences)
    return float(product * (gamma ** (n - 1)))


# ===========================================================================
# F53: Moebius confidence aggregation
# ===========================================================================

def moebius_confidence_aggregate(confidences: List[float]) -> float:
    """F53: Aggregate parallel evidence paths via Moebius combination.

    c_agg = tanh(sum_j arctanh(c_j))

    Equivalent to iterating F30 over all pairs pairwise.
    """
    if not confidences:
        return 0.0
    c_safe = np.clip(np.asarray(confidences), 1e-12, 0.999999)
    return float(np.tanh(np.sum(np.arctanh(c_safe))))


# ===========================================================================
# Pass 1: Exact keyword chaining (F49)
# ===========================================================================

def pass1_exact_chains(
    tokens: List[int],
    adjacency: Optional[Dict] = None,
) -> Dict[Tuple[int, ...], float]:
    """F49: Pass 1 — exact keyword chaining.

    A -> B, B -> C  =>  A -> C  with confidence 1.0.

    O(n * r) complexity where n = number of tokens, r = max relations per token.

    Parameters
    ----------
    tokens : List[int]
        Input token IDs.
    adjacency : Dict or None
        Adjacency mapping: token_id -> {neighbor_id: confidence, ...}.
        If None, a synthetic chain structure is used.

    Returns
    -------
    chains : Dict[Tuple[int, ...], float]
        All inferred chains with confidence 1.0 for exact matches.
    """
    chains: Dict[Tuple[int, ...], float] = {}

    if adjacency is None:
        # Build synthetic adjacency: each token connects to next
        adjacency = {}
        for i, tok in enumerate(tokens):
            if tok not in adjacency:
                adjacency[tok] = {}
            if i + 1 < len(tokens):
                adjacency[tok][tokens[i + 1]] = 1.0

    # Find all transitive chains up to length 5
    max_depth = 5

    def _dfs(current_path: List[int], depth: int):
        if depth >= max_depth:
            return
        current = current_path[-1]
        if current not in adjacency:
            return
        for neighbor, conf in adjacency[current].items():
            if neighbor in current_path:  # avoid cycles
                continue
            new_path = tuple(current_path + [neighbor])
            chains[new_path] = conf
            _dfs(list(new_path), depth + 1)

    for tok in tokens:
        _dfs([tok], 0)

    return chains


# ===========================================================================
# Pass 2: Semantic direction propagation
# ===========================================================================

def pass2_direction_propagate(
    chains: Dict[Tuple[int, ...], float],
    directions: Optional[Dict[Tuple[int, int], int]] = None,
) -> Dict[Tuple[int, ...], float]:
    """Pass 2 — semantic direction propagation.

    Direction rules:
        (+) + (+) -> (+)   (reinforcing)
        (-) + (-) -> (+)   (double negative)
        (+) + (-) -> (-)   (conflicting)

    Parameters
    ----------
    chains : Dict[Tuple[int, ...], float]
        Chains from Pass 1.
    directions : Dict[Tuple[int, int], int] or None
        Direction mapping for each edge: (+1) or (-1).

    Returns
    -------
    propagated : Dict[Tuple[int, ...], float]
        Chains with direction-adjusted confidences.
    """
    if directions is None:
        # All chains get positive direction by default
        return chains.copy()

    propagated: Dict[Tuple[int, ...], float] = {}
    for path, conf in chains.items():
        if len(path) < 2:
            propagated[path] = conf
            continue

        # Compute composite direction
        composite_dir = 1
        for k in range(len(path) - 1):
            edge = (path[k], path[k + 1])
            edge_dir = directions.get(edge, 1)
            # Direction composition: same -> positive, different -> negative
            if composite_dir == edge_dir:
                composite_dir = 1
            else:
                composite_dir = -1

        # Apply direction to confidence
        sign = max(0.0, float(composite_dir))  # negative direction reduces confidence
        propagated[path] = conf * sign

    return propagated


# ===========================================================================
# Pass 3: Jaro-Winkler fuzzy matching (F51)
# ===========================================================================

def pass3_fuzzy_match(
    tokens: List[int],
    token_id_to_str: Optional[Dict[int, str]] = None,
    threshold: float = THETA_JW,
) -> Dict[Tuple[int, ...], float]:
    """F51: Pass 3 — Jaro-Winkler fuzzy entity matching.

    For each pair of tokens, compute Jaro-Winkler similarity.
    If >= threshold, create an inferred chain.

    Parameters
    ----------
    tokens : List[int]
        Input token IDs.
    token_id_to_str : Dict[int, str] or None
        Mapping from token IDs to strings.
    threshold : float
        Minimum Jaro-Winkler score (default 0.85).

    Returns
    -------
    matches : Dict[Tuple[int, int], float]
        Fuzzy-matched pairs with JW similarity as confidence.
    """
    matches: Dict[Tuple[int, ...], float] = {}

    if token_id_to_str is None:
        return matches

    n = len(tokens)
    for i in range(n):
        for j in range(i + 1, n):
            s1 = token_id_to_str.get(tokens[i], "")
            s2 = token_id_to_str.get(tokens[j], "")
            if not s1 or not s2:
                continue
            jw = jaro_winkler(s1, s2)
            if jw >= threshold:
                matches[(tokens[i], tokens[j])] = jw

    return matches


# ===========================================================================
# F54: Quality filter
# ===========================================================================

def quality_filter(
    inferences: Dict[Tuple[int, ...], float],
    threshold: float = THETA_Q,
) -> Dict[Tuple[int, ...], float]:
    """F54: Remove contradictions, self-loops, low-confidence.

    Filters:
      1. Confidence < threshold (default 0.30) -> reject
      2. Self-loops (path[0] == path[-1]) -> reject
      3. Negative/conflicting confidences -> reject

    Expected rejection rate: ~3.9%.

    Parameters
    ----------
    inferences : Dict[Tuple[int, ...], float]
        Inferred chains with confidence scores.
    threshold : float
        Minimum acceptable confidence.

    Returns
    -------
    filtered : Dict[Tuple[int, ...], float]
        Quality-filtered inferences.
    """
    filtered: Dict[Tuple[int, ...], float] = {}
    for path, conf in inferences.items():
        # Reject self-loops
        if len(path) >= 2 and path[0] == path[-1]:
            continue
        # Reject low confidence
        if conf < threshold:
            continue
        # Reject negative/conflicting
        if conf < 0:
            continue
        filtered[path] = conf
    return filtered


# ===========================================================================
# F55: xxhash64 integrity checksum
# ===========================================================================

def triplet_checksum(
    subject: str, predicate: str, obj: str
) -> int:
    """F55: Compute integrity checksum for a triplet.

    checksum(t) = xxhash64(concat(utf8(s_i), utf8(r_k), utf8(s_j))) mod 2^64
    """
    try:
        import xxhash
        data = (subject + "|" + predicate + "|" + obj).encode("utf-8")
        return int(xxhash.xxh64(data).intdigest())
    except ImportError:
        # Fallback: use Python's built-in hash
        data = (subject + "|" + predicate + "|" + obj).encode("utf-8")
        h = 0
        for b in data:
            h = ((h * 31) + b) & 0xFFFFFFFFFFFFFFFF
        return h


def verify_checksum(
    subject: str, predicate: str, obj: str, stored: int
) -> bool:
    """F55: Verify triplet integrity.

    valid(t) = 1[checksum_stored(t) == checksum_computed(t)]
    """
    return triplet_checksum(subject, predicate, obj) == stored


# ===========================================================================
# Full 3-pass engine
# ===========================================================================

def transitive_closure(
    tokens: List[int],
    token_id_to_str: Optional[Dict[int, str]] = None,
    adjacency: Optional[Dict] = None,
    directions: Optional[Dict[Tuple[int, int], int]] = None,
) -> Dict[Tuple[int, ...], float]:
    """Complete 3-pass deterministic transitive inference.

    Pass 1: Exact keyword chaining (F49)
    Pass 2: Semantic direction propagation (F50)
    Pass 3: Jaro-Winkler fuzzy matching (F51)

    Returns all inferred chains with provenance and confidence.

    Parameters
    ----------
    tokens : List[int]
        Input token IDs.
    token_id_to_str : Dict[int, str] or None
        Token ID to string mapping for fuzzy matching.
    adjacency : Dict or None
        Token adjacency for exact chaining.
    directions : Dict or None
        Edge direction annotations.

    Returns
    -------
    results : Dict[Tuple[int, ...], float]
        All inferred chains with final confidence scores.
    """
    all_results: Dict[Tuple[int, ...], float] = {}

    # Pass 1: Exact chains
    chains_p1 = pass1_exact_chains(tokens, adjacency)
    for path, conf in chains_p1.items():
        all_results[path] = conf

    # Pass 2: Direction propagation
    chains_p2 = pass2_direction_propagate(chains_p1, directions)
    for path, conf in chains_p2.items():
        if path not in all_results:
            all_results[path] = conf
        else:
            # Combine via Moebius confidence
            all_results[path] = moebius_confidence(all_results[path], conf)

    # Pass 3: Fuzzy matching
    if token_id_to_str is not None:
        matches_p3 = pass3_fuzzy_match(tokens, token_id_to_str)
        for path, conf in matches_p3.items():
            if path not in all_results:
                all_results[path] = conf
            else:
                all_results[path] = moebius_confidence(all_results[path], conf)

    # Apply quality filter
    return quality_filter(all_results)


# ===========================================================================
# F32: Amplification factor
# ===========================================================================

def amplification_factor(
    tokens: List[int],
    inferences: Dict[Tuple[int, ...], float],
) -> float:
    """F32: Measure inference amplification.

    amplification = N_inferred / N_explicit

    Target: 7–10x.
    For N_explicit = 3: expected N_inferred in [21, 30].
    """
    n_explicit = max(1, len(tokens))
    n_inferred = len(inferences)
    return float(n_inferred) / float(n_explicit)


# ===========================================================================
# F50: Semantic matching via hyperbolic distance
# ===========================================================================

def semantic_match_score(
    state_a: np.ndarray,
    state_b: np.ndarray,
) -> float:
    """F50: Semantic matching score.

    score_2 = exp(-d_H(X(q), X(t)))

    Symbols match if score_2 >= 0.85.
    """
    # Compute Minkowski inner product
    inner = -(-state_a[0] * state_b[0] + np.dot(state_a[1:], state_b[1:]))
    inner = max(1.0, inner)
    d_h = float(np.arccosh(inner))
    return float(np.exp(-d_h))


# ===========================================================================
# Complete inference pipeline
# ===========================================================================

def complete_inference_pipeline(
    query_tokens: List[int],
    knowledge_base: Optional[List[Tuple[int, str, int]]] = None,
    token_id_to_str: Optional[Dict[int, str]] = None,
    adjacency: Optional[Dict] = None,
    max_path_length: int = 5,
) -> List[Tuple[Tuple[int, ...], float, str]]:
    """Complete inference pipeline with all phases.

    Phase 1: Direct retrieval via 3-pass inference
    Phase 2: Transitive closure
    Phase 3: Confidence computation and quality filter

    Returns sorted list of (path, confidence, provenance).
    """
    results: List[Tuple[Tuple[int, ...], float, str]] = []

    # Phase 1: 3-pass inference
    direct = transitive_closure(
        query_tokens,
        token_id_to_str=token_id_to_str,
        adjacency=adjacency,
    )
    for path, conf in direct.items():
        results.append((path, conf, "direct"))

    # Phase 2: Transitive closure via path confidence
    if knowledge_base is not None:
        for triplet in knowledge_base:
            if len(triplet) >= 2:
                # Build path from triplet
                path = tuple(triplet)
                # Compute chain confidence
                conf = chain_confidence([1.0] * len(path))
                if conf >= THETA_Q:
                    results.append((path, conf, "transitive"))

    # Phase 3: Sort by confidence descending
    results.sort(key=lambda x: x[1], reverse=True)

    return results
