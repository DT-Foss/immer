"""
Three-pass inference engine for the .causal format.

Pass 1: Exact keyword matching
Pass 2: Semantic direction propagation
Pass 3: Jaro-Winkler fuzzy entity matching

All inference is DETERMINISTIC - same input always produces same output.
Zero hallucination guarantee through pure logic.
"""

import re
import time
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict

from .core import (
    POSITIVE_INDICATORS,
    NEGATIVE_INDICATORS,
    NEUTRAL_INDICATORS,
    NEGATIVE_PREFIXES,
    DEFAULT_INFERENCE_RULES,
    GENERALIZED_INFERENCE_RULES,
    INFERENCE_CONFIG,
    normalize_entity,
)


# =============================================================================
# AUTO-THRESHOLD CALIBRATION
# =============================================================================

def calculate_auto_threshold(entities: List[str]) -> float:
    """
    Auto-calibrate Jaro-Winkler threshold based on entity characteristics.

    Short entities (medical terms) → stricter threshold
    Long entities (scientific sentences) → looser threshold

    Returns:
        Recommended threshold value (0.72-0.88)
    """
    if not entities:
        return 0.78  # Default

    # Calculate average words per entity
    word_counts = [len(e.split()) for e in entities]
    avg_words = sum(word_counts) / len(word_counts)

    # Calculate entity reuse ratio
    unique_entities = len(set(e.lower().strip() for e in entities))
    reuse_ratio = len(entities) / max(unique_entities, 1)

    # Threshold based on entity length
    if avg_words <= 3:
        base_threshold = 0.88  # Very strict for short terms
    elif avg_words <= 5:
        base_threshold = 0.85  # Standard
    elif avg_words <= 8:
        base_threshold = 0.78  # Looser for sentences
    else:
        base_threshold = 0.72  # Very loose for long descriptions

    # Adjust for reuse ratio (high reuse = can be stricter)
    if reuse_ratio > 1.5:
        base_threshold = min(base_threshold + 0.03, 0.90)

    return round(base_threshold, 2)


# =============================================================================
# TRIGRAM INDEX FOR FAST FUZZY MATCHING
# =============================================================================

def get_trigrams(s: str) -> Set[str]:
    """Extract character trigrams from a string for fast similarity pre-filtering."""
    s = s.lower().strip()
    if len(s) < 3:
        return {s}
    return {s[i:i+3] for i in range(len(s) - 2)}


class TrigramIndex:
    """
    Inverted index mapping trigrams to entities for O(1) candidate lookup.
    Reduces fuzzy matching from O(N) to O(k) where k << N.

    Usage:
        index = TrigramIndex()
        index.add_batch(["entity1", "entity2", ...])
        candidates = index.find_candidates("query", min_shared=2)
    """

    def __init__(self):
        self._index: Dict[str, Set[str]] = defaultdict(set)
        self._entities: Set[str] = set()
        self._entity_trigrams: Dict[str, Set[str]] = {}

    def add(self, entity: str) -> None:
        """Add an entity to the index."""
        entity_lower = entity.lower().strip()
        if entity_lower in self._entities:
            return

        self._entities.add(entity_lower)
        trigrams = get_trigrams(entity_lower)
        self._entity_trigrams[entity_lower] = trigrams

        for tri in trigrams:
            self._index[tri].add(entity_lower)

    def add_batch(self, entities: List[str]) -> None:
        """Add multiple entities efficiently."""
        for e in entities:
            self.add(e)

    def find_candidates(
        self,
        query: str,
        min_shared: int = 2,
        max_candidates: int = 50
    ) -> List[Tuple[str, int]]:
        """
        Find candidate entities that share at least min_shared trigrams with query.

        Returns:
            List of (entity, shared_count) sorted by shared_count descending.
        """
        query_lower = query.lower().strip()
        query_trigrams = get_trigrams(query_lower)

        if not query_trigrams:
            return []

        candidate_scores: Dict[str, int] = defaultdict(int)

        for tri in query_trigrams:
            for entity in self._index.get(tri, []):
                if entity != query_lower:
                    candidate_scores[entity] += 1

        candidates = [
            (entity, score)
            for entity, score in candidate_scores.items()
            if score >= min_shared
        ]

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:max_candidates]

    def __len__(self) -> int:
        return len(self._entities)


# =============================================================================
# JARO-WINKLER SIMILARITY
# =============================================================================

def jaro_winkler_similarity(s1: str, s2: str) -> float:
    """
    Calculate Jaro-Winkler similarity between two strings.

    Returns:
        Value between 0.0 (no match) and 1.0 (exact match).

    This is DETERMINISTIC - same input always gives same output.
    """
    if s1 == s2:
        return 1.0

    if not s1 or not s2:
        return 0.0

    s1 = s1.lower().strip()
    s2 = s2.lower().strip()

    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)

    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)

        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (matches / len1 + matches / len2 + (matches - transpositions // 2) / matches) / 3

    # Winkler modification - boost for common prefix
    prefix = 0
    for i in range(min(len1, len2, 4)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + prefix * 0.1 * (1 - jaro)


# =============================================================================
# DIRECTION DETECTION
# =============================================================================

def _has_negative_prefix(word: str, stem: str) -> bool:
    """Check if a word contains a stem with a negative prefix."""
    word = word.lower()
    stem = stem.lower()

    idx = word.find(stem)
    if idx <= 0:
        return False

    prefix = word[:idx]
    return prefix in NEGATIVE_PREFIXES


def detect_mechanism_direction(mechanism: str) -> str:
    """
    Detect semantic direction of a mechanism string.

    Returns:
        'positive', 'negative', or 'neutral'

    Handles negative prefixes (de-, dis-, un-, in-, non-)
    to avoid false positives like "deactivates" being positive.
    """
    text = mechanism.lower()
    words = re.findall(r'\b\w+\b', text)

    pos_count = 0
    neg_count = 0

    for ind in POSITIVE_INDICATORS:
        for word in words:
            if ind in word:
                if _has_negative_prefix(word, ind):
                    neg_count += 1
                else:
                    pos_count += 1
                break

    for ind in NEGATIVE_INDICATORS:
        for word in words:
            if ind in word:
                if _has_negative_prefix(word, ind):
                    pos_count += 1
                else:
                    neg_count += 1
                break

    # Phrasal negation patterns
    has_phrasal_negation = any(neg in text for neg in [
        "not ", "no ", "without ", "lack of", "absence of",
        "fails to", "failed to", "does not", "do not", "cannot"
    ])

    if has_phrasal_negation:
        pos_count, neg_count = neg_count, pos_count

    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"


# =============================================================================
# TOKEN OVERLAP
# =============================================================================

def get_token_overlap(entity1: str, entity2: str) -> float:
    """
    Calculate token overlap ratio (Jaccard similarity) between two entities.
    """
    tokens1 = set(entity1.lower().split())
    tokens2 = set(entity2.lower().split())

    stopwords = {'the', 'a', 'an', 'in', 'of', 'to', 'and', 'or', 'with', 'by', 'for', 'on', 'is', 'are', 'was', 'were'}
    tokens1 = tokens1 - stopwords
    tokens2 = tokens2 - stopwords

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    return len(intersection) / len(union)


# =============================================================================
# ENTITY SIMILARITY MATCHING
# =============================================================================

def find_similar_entities_indexed(
    entity: str,
    trigram_index: TrigramIndex,
    threshold: float = 0.78,
    max_candidates: int = 50
) -> List[Tuple[str, float]]:
    """
    Find similar entities using trigram index for candidate pruning.

    Multi-stage matching:
    1. Synonym dictionary
    2. Token overlap (>=70%)
    3. Jaro-Winkler (>=threshold)

    Returns:
        List of (matched_entity, similarity) tuples.
    """
    results = []
    entity_lower = entity.lower().strip()
    entity_normalized, was_normalized, _ = normalize_entity(entity)

    seen = set()

    candidates = trigram_index.find_candidates(
        entity_lower,
        min_shared=INFERENCE_CONFIG['min_trigram_overlap'],
        max_candidates=max_candidates
    )

    for candidate, trigram_overlap in candidates:
        if candidate in seen:
            continue

        if entity_lower == candidate:
            results.append((candidate, 1.0))
            seen.add(candidate)
            continue

        # Synonym match
        e_normalized, e_was_normalized, _ = normalize_entity(candidate)
        if entity_normalized == e_normalized and (was_normalized or e_was_normalized):
            results.append((candidate, 0.95))
            seen.add(candidate)
            continue

        # Token overlap
        token_sim = get_token_overlap(entity_lower, candidate)
        if token_sim >= 0.70:
            results.append((candidate, 0.85 + (token_sim - 0.70) * 0.5))
            seen.add(candidate)
            continue

        # Jaro-Winkler
        if len(entity_lower) > 6 and len(candidate) > 6:
            sim = jaro_winkler_similarity(entity_lower, candidate)
            if sim >= threshold:
                results.append((candidate, sim))
                seen.add(candidate)

    return sorted(results, key=lambda x: x[1], reverse=True)[:10]


# =============================================================================
# MAIN INFERENCE ENGINE
# =============================================================================

def run_inference(
    explicit: List[Dict],
    entities: List[str],
    rules: List[Dict] = None,
    threshold: str | float = "auto"
) -> List[Dict]:
    """
    Run the three-pass inference engine.

    Args:
        explicit: List of explicit triplet dicts with keys:
            trigger, mechanism, outcome, confidence, s_idx, m_idx, o_idx
        entities: List of entity strings (index matches s_idx/m_idx/o_idx)
        rules: Optional custom rules (defaults to DEFAULT_INFERENCE_RULES)
        threshold: Jaro-Winkler threshold for fuzzy matching.
            - "auto": Auto-calibrate based on entity characteristics (recommended)
            - float: Manual threshold (0.0-1.0)

    Returns:
        List of inferred triplet dicts.
    """
    if rules is None:
        rules = DEFAULT_INFERENCE_RULES

    # Auto-calibrate threshold if needed
    if threshold == "auto":
        jw_threshold = calculate_auto_threshold(entities)
    else:
        jw_threshold = float(threshold)

    inferred = []
    seen = set()

    # Build indices
    by_subject: Dict[int, List[Dict]] = defaultdict(list)
    by_object: Dict[int, List[Dict]] = defaultdict(list)
    triggers_by_entity: Dict[str, List[Dict]] = defaultdict(list)

    for t in explicit:
        key = (t['s_idx'], t['m_idx'], t['o_idx'])
        seen.add(key)
        by_subject[t['s_idx']].append(t)
        by_object[t['o_idx']].append(t)
        triggers_by_entity[t['trigger'].lower().strip()].append(t)

    # Mutable entity list for adding new mechanisms
    entity_list = list(entities)

    def get_or_create_entity_idx(entity_str: str) -> int:
        """Get index for entity, creating if needed."""
        entity_lower = entity_str.lower()
        for i, e in enumerate(entity_list):
            if e.lower() == entity_lower:
                return i
        idx = len(entity_list)
        entity_list.append(entity_str)
        return idx

    def get_entity(idx: int) -> str:
        """Get entity string by index."""
        if 0 <= idx < len(entity_list):
            return entity_list[idx]
        return f"<unknown:{idx}>"

    # =========================================================================
    # PASS 1: Exact Keyword Matching
    # =========================================================================

    def normalize_mechanism(m: str) -> str:
        return m.lower().strip()

    for rule in rules:
        if len(rule.get('pattern', [])) != 2:
            continue

        p1, p2 = rule['pattern']
        conclusion = rule['conclusion']
        gamma = rule.get('confidence_modifier', 0.85)

        for t1 in explicit:
            m1 = normalize_mechanism(t1['mechanism'])
            if p1.lower() not in m1:
                continue

            b_idx = t1['o_idx']

            for t2 in by_subject.get(b_idx, []):
                m2 = normalize_mechanism(t2['mechanism'])
                if p2.lower() not in m2:
                    continue

                a_idx = t1['s_idx']
                c_idx = t2['o_idx']

                if a_idx == c_idx:
                    continue

                conclusion_idx = get_or_create_entity_idx(conclusion)
                key = (a_idx, conclusion_idx, c_idx)

                if key in seen:
                    continue
                seen.add(key)

                c_inferred = t1['confidence'] * t2['confidence'] * gamma

                inferred.append({
                    'trigger': get_entity(a_idx),
                    'mechanism': conclusion,
                    'outcome': get_entity(c_idx),
                    'confidence': round(c_inferred, 4),
                    'source': 'INFERRED',
                    'is_inferred': True,
                    's_idx': a_idx,
                    'm_idx': conclusion_idx,
                    'o_idx': c_idx,
                    'inference_rule': rule['name'],
                    'evidence': f"Inferred via {rule['name']}: {get_entity(a_idx)} -> {get_entity(b_idx)} -> {get_entity(c_idx)}",
                })

    # =========================================================================
    # PASS 2: Semantic Direction-Based Inference
    # =========================================================================

    for rule in GENERALIZED_INFERENCE_RULES:
        if len(rule.get('pattern', [])) != 2:
            continue

        dir1_expected, dir2_expected = rule['pattern']
        conclusion_text = rule.get('conclusion_text', rule['conclusion'])
        gamma = rule.get('confidence_modifier', 0.75)

        for t1 in explicit:
            dir1 = detect_mechanism_direction(t1['mechanism'])
            if dir1 != dir1_expected:
                continue

            b_idx = t1['o_idx']

            for t2 in by_subject.get(b_idx, []):
                dir2 = detect_mechanism_direction(t2['mechanism'])
                if dir2 != dir2_expected:
                    continue

                a_idx = t1['s_idx']
                c_idx = t2['o_idx']

                if a_idx == c_idx:
                    continue

                conclusion_idx = get_or_create_entity_idx(conclusion_text)
                key = (a_idx, conclusion_idx, c_idx)

                if key in seen:
                    continue
                seen.add(key)

                c_inferred = t1['confidence'] * t2['confidence'] * gamma

                if c_inferred < INFERENCE_CONFIG['min_confidence']:
                    continue

                inferred.append({
                    'trigger': get_entity(a_idx),
                    'mechanism': conclusion_text,
                    'outcome': get_entity(c_idx),
                    'confidence': round(c_inferred, 4),
                    'source': 'INFERRED_SEMANTIC',
                    'is_inferred': True,
                    's_idx': a_idx,
                    'm_idx': conclusion_idx,
                    'o_idx': c_idx,
                    'inference_rule': f"semantic_{rule['name']}",
                    'evidence': f"Semantic ({dir1}+{dir2}={rule['conclusion']}): {get_entity(a_idx)[:30]} -> {get_entity(b_idx)[:30]} -> {get_entity(c_idx)[:30]}",
                })

    # =========================================================================
    # PASS 3: Fuzzy Entity Matching (Jaro-Winkler with TrigramIndex)
    # =========================================================================

    start_time = time.time()
    timeout = INFERENCE_CONFIG['timeout_seconds']
    max_inferred = INFERENCE_CONFIG['max_inferred_triplets']
    max_candidates = INFERENCE_CONFIG['max_fuzzy_candidates']
    # Use auto-calibrated or manual threshold
    fuzzy_threshold = jw_threshold

    # Build trigram index
    trigram_index = TrigramIndex()
    for t in explicit:
        trigram_index.add(t['trigger'].lower().strip())

    for t1 in explicit:
        if time.time() - start_time > timeout:
            break
        if len(inferred) >= max_inferred:
            break

        outcome = t1['outcome']

        similar = find_similar_entities_indexed(
            outcome,
            trigram_index,
            threshold=fuzzy_threshold,
            max_candidates=max_candidates
        )

        for matched_trigger, similarity in similar:
            if similarity >= 0.99:
                continue

            for t2 in triggers_by_entity.get(matched_trigger, []):
                a_idx = t1['s_idx']
                c_idx = t2['o_idx']

                if a_idx == c_idx:
                    continue

                dir1 = detect_mechanism_direction(t1['mechanism'])
                dir2 = detect_mechanism_direction(t2['mechanism'])

                if dir1 == 'positive' and dir2 == 'positive':
                    conclusion_text = 'indirectly promotes'
                elif dir1 == 'negative' and dir2 == 'negative':
                    conclusion_text = 'indirectly promotes (double inhibition)'
                elif (dir1 == 'positive' and dir2 == 'negative') or (dir1 == 'negative' and dir2 == 'positive'):
                    conclusion_text = 'indirectly inhibits'
                else:
                    conclusion_text = 'indirectly linked to'

                conclusion_idx = get_or_create_entity_idx(conclusion_text)
                key = (a_idx, conclusion_idx, c_idx)

                if key in seen:
                    continue
                seen.add(key)

                c_inferred = t1['confidence'] * t2['confidence'] * similarity * 0.75

                if c_inferred < INFERENCE_CONFIG['min_confidence']:
                    continue

                inferred.append({
                    'trigger': get_entity(a_idx),
                    'mechanism': conclusion_text,
                    'outcome': get_entity(c_idx),
                    'confidence': round(c_inferred, 4),
                    'source': 'INFERRED_FUZZY',
                    'is_inferred': True,
                    's_idx': a_idx,
                    'm_idx': conclusion_idx,
                    'o_idx': c_idx,
                    'inference_rule': f"fuzzy_chain_{dir1}_{dir2}",
                    'evidence': f"Fuzzy (sim={similarity:.2f}): '{outcome[:40]}' ≈ '{matched_trigger[:40]}'",
                    'entity_similarity': similarity,
                })

                if len(inferred) >= max_inferred:
                    break

    return inferred
