"""
fertig.primitives — das geschlossene semantische Primitiv-Schema.

Die Grundprimitive, aus denen sich alles Weitere ableitet (NSM-Gedanke:
Wierzbickas semantische Primate; Fillmore-Kasusrahmen). Statt wachsender
Relations-Listen ist dies die EINE Quelle der Wahrheit: Extraktion mappt
Oberflächenformen IN das Schema, der Graph speichert kanonische Primitiv-
Namen, QA matcht AUF dem Schema.

Bausteine aus den Foss-Formeln:
  vortexmath : Digitalwurzel dr(n) = n mod 9 — Ring-Homomorphismus als
               Klassifikations-Fallback für unbekannte Oberflächenformen
  wirkung    : Noether-Detektoren (Parität/Skala/Periodizität) — die
               Detektions-Primitive struktureller Regularität
  gottformel : Intrinsische Dimension (TwoNN) — der Vollständigkeits-Test:
               saturiert die Relations-Nutzung, ist das Schema geschlossen
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Das geschlossene Relations-Schema
# name: patterns (Oberflächen-Formen), conf, inverse, markers (Frage-Wörter)
# ---------------------------------------------------------------------------

RELATIONS: Dict[str, dict] = {
    # --- definitorisch ---
    "is_a": {
        "patterns": [r"\b(?:is|are)\s+(?:a|an|the)\s+(.+?)(?:[,;.]| that| which|$)"],
        "conf": 0.50, "inverse": "has_instance",
        "markers": {"is", "are", "called", "known as", "type of"},
    },
    "kind_of": {
        "patterns": [r"\b(?:is|are)\s+(?:a|an)\s+(?:type|kind|form|sort|species)"
                     r"\s+of\s+(.+?)(?:[,;.]|$)"],
        "conf": 0.55, "inverse": "has_kind",
        "markers": {"type", "kind", "species", "category"},
    },
    "defined_as": {
        "patterns": [r"\bis\s+defined\s+as\s+(.+?)(?:[,;.]|$)"],
        "conf": 0.45, "inverse": None,
        "markers": {"defined", "definition", "meaning"},
    },
    "has_property": {
        "patterns": [r"\b(?:is|are)\s+(small|large|big|tall|short|old|new|"
                     r"fast|slow|hot|cold|light|heavy|strong|weak|cheap|"
                     r"expensive|dense|solid|liquid|gaseous|transparent|"
                     r"opaque|flexible|rigid|elastic|brittle|soluble|"
                     r"insoluble|flammable|toxic|harmless|visible|invisible)"
                     r"(?:[,;.]| and| but|$)"],
        "conf": 0.35, "inverse": None,
        "markers": {"property", "characteristic", "trait", "feature",
                    "attribute", "like"},
    },
    # --- partitiv / materiell ---
    "part_of": {
        "patterns": [r"\bis\s+part\s+of\s+(.+?)(?:[,;.]|$)"],
        "conf": 0.50, "inverse": "contains",
        "markers": {"part", "component", "member"},
    },
    "contains": {
        "patterns": [r"\b(?:contains|includes|comprises)\s+(.+?)(?:[,;.]|$)"],
        "conf": 0.45, "inverse": "part_of",
        "markers": {"contain", "include", "comprise", "made up of"},
    },
    "made_of": {
        "patterns": [r"\bis\s+made\s+of\s+(.+?)(?:[,;.]|$)",
                     r"\bis\s+built\s+from\s+(.+?)(?:[,;.]|$)"],
        "conf": 0.55, "inverse": None,
        "markers": {"made", "material", "composed", "consist"},
    },
    "consists_of": {
        "patterns": [r"\bconsists\s+of\s+(.+?)(?:[,;.]|$)",
                     r"\bis\s+composed\s+of\s+(.+?)(?:[,;.]|$)"],
        "conf": 0.50, "inverse": None,
        "markers": {"consist", "composed"},
    },
    # --- lokativ ---
    "located_in": {
        "patterns": [r"\bis\s+(?:located|situated)\s+in\s+(.+?)(?:[,;.]|$)"],
        "conf": 0.50, "inverse": "contains_location",
        "markers": {"located", "situated", "where"},
    },
    "found_in": {
        "patterns": [r"\bis\s+(?:found|present|common|abundant)\s+in\s+"
                     r"(.+?)(?:[,;.]|$)"],
        "conf": 0.45, "inverse": None,
        "markers": {"found", "present", "common in", "occur"},
    },
    # --- funktional ---
    "used_for": {
        "patterns": [r"\bis\s+used\s+for\s+(.+?)(?:[,;.]|$)"],
        "conf": 0.50, "inverse": None,
        "markers": {"used", "purpose", "function", "for"},
    },
    "used_to": {
        "patterns": [r"\bis\s+used\s+to\s+(.+?)(?:[,;.]|$)",
                     r"\bis\s+(?:able|designed|employed)\s+to\s+(.+?)(?:[,;.]|$)"],
        "conf": 0.45, "inverse": None,
        "markers": {"used", "designed"},
    },
    "responsible_for": {
        "patterns": [r"\bis\s+responsible\s+for\s+(.+?)(?:[,;.]|$)"],
        "conf": 0.50, "inverse": None,
        "markers": {"responsible", "role", "controls"},
    },
    "capable_of": {
        "patterns": [r"\bcan\s+(?:also\s+)?(?:be used to )?(.+?)(?:[,;.]|$)"],
        "conf": 0.35, "inverse": None,
        "markers": {"can", "capable", "ability"},
    },
    # --- kausal (die Kern-Primitive) ---
    "causes": {
        "patterns": [],
        "conf": 0.40, "inverse": "caused_by",
        "markers": {"cause", "effect", "factor", "due", "result", "because",
                    "lead"},
        "surface": {"causes", "cause", "leads to", "lead to", "results in",
                    "result in", "produces", "triggers", "induces"},
    },
    "prevents": {
        "patterns": [], "conf": 0.40, "inverse": "prevented_by",
        "markers": {"prevent", "block", "stop", "avoid", "protect"},
        "surface": {"prevents", "prevent", "blocks", "blocks", "protects"},
    },
    "reduces": {
        "patterns": [], "conf": 0.40, "inverse": "increased_by",
        "markers": {"reduce", "decrease", "lower", "less"},
        "surface": {"reduces", "reduce", "lowers", "decreases", "decrease"},
    },
    "increases": {
        "patterns": [], "conf": 0.40, "inverse": "reduced_by",
        "markers": {"increase", "raise", "more", "grow"},
        "surface": {"increases", "increase", "raises", "enhances", "boosts",
                    "promotes"},
    },
    "improves": {
        "patterns": [], "conf": 0.40, "inverse": "worsened_by",
        "markers": {"improve", "better", "boost", "strengthen"},
        "surface": {"improves", "improve", "strengthens", "strengthen"},
    },
    "related_to": {
        "patterns": [], "conf": 0.35, "inverse": None,
        "markers": {"related", "associated", "linked", "correlated",
                    "connection"},
        "surface": {"linked to", "linked with", "associated with",
                    "related to", "correlated with"},
    },
    # --- temporal ---
    "created_in": {
        "patterns": [r"\b(?:was|were)\s+(developed|invented|founded|"
                     r"discovered|released|introduced|established|created|"
                     r"built)\s+(?:in|during)\s+(1[89]\d\d|20\d\d)",
                     r"\b(?:dates?|dates back)\s+to\s+(1[89]\d\d|20\d\d)"],
        "conf": 0.60, "inverse": None,
        "markers": {"developed", "invented", "founded", "discovered", "year",
                    "date", "recently", "oldest", "newest"},
    },
    # --- komparativ ---
    "comparative": {
        "patterns": [r"\bis\s+(smaller|larger|bigger|taller|shorter|older|"
                     r"younger|faster|slower|hotter|colder|lighter|heavier|"
                     r"stronger|weaker|quieter|louder|cheaper|denser|greater|"
                     r"more|less)\s+than\s+(.+?)(?:[,;.]| and|$)"],
        "conf": 0.50, "inverse": None,
        "markers": {"than", "smaller", "larger", "bigger", "difference",
                    "different", "compared", "compare"},
    },
    # --- generisch ---
    "related": {
        "patterns": [], "conf": 0.30, "inverse": None,
        "markers": set(),
    },
}

# kanonische Namen (für normalize_mechanism)
CANONICAL = set(RELATIONS)


# Adverb-Präfixe, die vor Mechanismen stehen können (indirectly_linked_to)
_ADVERB_PREFIXES = ("indirectly_", "directly_", "significantly_",
                    "strongly_", "weakly_", "highly_", "positively_",
                    "negatively_", "closely_", "mainly_", "primarily_",
                    "partly_", "largely_")

# Synonym-Tabelle für die Normalisierung
_SYNONYMS = {
    "associated_with": "related_to", "linked_to": "related_to",
    "linked_with": "related_to", "related_to": "related_to",
    "correlated_with": "related_to", "described_as": "defined_as",
    "defined_as": "defined_as", "leads_to": "causes",
    "lead_to": "causes", "results_in": "causes", "result_in": "causes",
    "produces": "causes", "triggers": "causes", "induces": "causes",
    "improves": "improves", "strengthens": "improves",
}


def normalize_mechanism(mechanism: str) -> str:
    """Oberflächen-Form -> kanonisches Primitiv. Bekannte Synonyme werden
    gemappt; Adverb-Präfixe gestrippt; unbekannte Formen bleiben."""
    m = str(mechanism).strip().lower().replace(" ", "_")
    for pre in _ADVERB_PREFIXES:
        if m.startswith(pre):
            m = m[len(pre):]
            break
    if m in CANONICAL:
        return m
    if m in _SYNONYMS:
        return _SYNONYMS[m]
    for name, spec in RELATIONS.items():
        if m in spec.get("surface", set()):
            return name
    return m


# ---------------------------------------------------------------------------
# vortexmath: Digitalwurzel (Ring-Homomorphismus ℕ -> ℤ/9ℤ)
# ---------------------------------------------------------------------------

_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def digital_root(value) -> int:
    """dr(n) = n mod 9 (9 statt 0). Idempotenter Ring-Homomorphismus:
    dr(a+b) = dr(dr(a)+dr(b)), dr(a·b) = dr(dr(a)·dr(b))."""
    if isinstance(value, str):
        n = sum(_LETTERS.index(c) + 1 for c in value.lower()
                if c in _LETTERS)
    else:
        n = int(value)
    r = n % 9
    return 9 if r == 0 else r


def digital_root_class(word: str) -> int:
    """Klassifikations-Fallback: unbekannte Oberflächenformen werden über
    ihre Digitalwurzel-Klasse gruppiert (vortexmath: 9 Klassen, 5 Orbit-
    Typen). Zwei Formen derselben Klasse sind Primitive-Kandidaten."""
    return digital_root(word)


# ---------------------------------------------------------------------------
# wirkung: Noether-Detektoren (Parität/Periodizität — Struktur-Primitive)
# ---------------------------------------------------------------------------

def detect_periodicity(seq: List[float], tol: float = 0.05) -> Optional[int]:
    """Periodizitäts-Detektor (Noether-Stil): kleinste Periode p mit
    |x[i+p] − x[i]| < tol für alle i. None wenn nicht periodisch."""
    n = len(seq)
    for p in range(1, n // 2 + 1):
        if all(abs(seq[i + p] - seq[i]) < tol for i in range(n - p)):
            return p
    return None


def detect_parity(seq: List[float], tol: float = 0.05) -> Optional[str]:
    """Paritäts-Detektor: 'even' (symmetrisch) / 'odd' (antisymmetrisch)."""
    n = len(seq)
    if n < 2:
        return None
    even = all(abs(seq[i] - seq[n - 1 - i]) < tol for i in range(n))
    if even:
        return "even"
    odd = all(abs(seq[i] + seq[n - 1 - i]) < tol for i in range(n))
    return "odd" if odd else None


# ---------------------------------------------------------------------------
# gottformel: Intrinsische Dimension (TwoNN) — der Schema-Vollständigkeits-Test
# ---------------------------------------------------------------------------

def intrinsic_dimension(X: np.ndarray, k: int = 2) -> float:
    """TwoNN-Schätzer (Levina-Bickel-Variante, gottformel_formulas.md):
    ID ≈ (1/|S|) · Σ log(r2_i / r1_i) für die zwei nächsten Nachbarn.
    Saturiert die ID der Relations-Nutzung, ist das Primitiv-Schema
    vollständig; wächst sie, fehlen Primitive."""
    n = len(X)
    if n < 3:
        return 1.0
    from scipy.spatial.distance import pdist, squareform  # optional
    try:
        D = squareform(pdist(X, metric="euclidean"))
    except ImportError:  # pragma: no cover
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.sqrt(np.sum((X[i] - X[j]) ** 2)))
                D[i, j] = D[j, i] = d
    np.fill_diagonal(D, np.inf)
    ratios = []
    for i in range(n):
        nn = np.sort(D[i])[:k]
        if nn[0] > 0 and nn[-1] > 0:
            ratios.append(np.log(nn[-1] / nn[0]))
    if not ratios:
        return 1.0
    return float(len(ratios) / (np.sum(ratios) + 1e-12))


# ---------------------------------------------------------------------------
# Frage-Marker -> Primitiv-Klassen (für QA auf dem Schema)
# ---------------------------------------------------------------------------

def question_primitives(question: str) -> Set[str]:
    """Welche Primitiv-Klassen fragt die Frage an? (über die Marker)."""
    q = question.lower()
    found: Set[str] = set()
    for name, spec in RELATIONS.items():
        markers = spec.get("markers", set())
        if any(m in q for m in markers):
            found.add(name)
    return found
