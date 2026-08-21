"""
Core data structures for the .causal format.

This module contains:
- Constants (magic bytes, version, header sizes)
- Data classes (CausalTriplet, SemanticCluster, KnowledgeGap)
- Semantic indicators for direction detection
- Medical synonym dictionary
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set

# =============================================================================
# CONSTANTS
# =============================================================================

MAGIC = b'CAUSAL\x00\x01'  # 8 bytes magic identifier (C-A-U-S-A-L + version bytes)
LEGACY_MAGIC = b'CAUSAL01'  # earlier revision: identical layout, ASCII magic —
                            # readers accept both, writers emit MAGIC only
VERSION = 1
HEADER_SIZE = 64
OFFSET_TABLE_SIZE = 32  # 6 offsets x 4 bytes + padding

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CausalTriplet:
    """
    A single causal relationship (trigger -> mechanism -> outcome).

    Attributes:
        s_idx: Subject/trigger index in entity dictionary
        m_idx: Mechanism/predicate index
        o_idx: Object/outcome index
        confidence: Confidence score (0.0-1.0)
        source: Source file or "INFERRED"
        pmcid: PubMed Central ID if applicable
        quantification: Quantitative data extracted
        evidence: Evidence sentence from source
        domain: Knowledge domain (medical, astrophysics, etc.)
        quality_score: Additional quality metric
        is_inferred: True if derived through inference
        inference_chain: List of source triplet IDs for provenance
    """
    s_idx: int
    m_idx: int
    o_idx: int
    confidence: float = 0.8
    source: str = ""
    pmcid: str = ""
    quantification: str = ""
    evidence: str = ""
    domain: str = ""
    quality_score: float = 0.0
    is_inferred: bool = False
    inference_chain: List[int] = field(default_factory=list)


# Alias for cleaner API
Triplet = CausalTriplet


@dataclass
class SemanticCluster:
    """
    A group of semantically related entities.

    Used for organizing entities into meaningful categories
    (e.g., "COVID symptoms", "mitochondrial pathways").
    """
    cluster_id: int
    name: str
    entity_indices: List[int]
    related_clusters: List[int] = field(default_factory=list)


@dataclass
class KnowledgeGap:
    """
    An identified gap in the knowledge graph.

    Represents a relationship that is expected but not found,
    useful for directing future research.
    """
    subject: str
    predicate: str
    expected_object_type: str
    confidence: float = 0.5
    suggested_queries: List[str] = field(default_factory=list)


# =============================================================================
# SEMANTIC DIRECTION INDICATORS
# =============================================================================
# Used for detecting whether a mechanism is positive, negative, or neutral

POSITIVE_INDICATORS = [
    "activat", "promot", "enhanc", "increas", "stimulat", "induc", "upregulat",
    "boost", "amplif", "facilitat", "improv", "restor", "protect", "support",
    "elevat", "augment", "potentiat", "accelerat", "enabl", "trigger"
]

NEGATIVE_INDICATORS = [
    "inhibit", "impair", "reduc", "block", "decreas", "disrupt", "downregulat",
    "suppress", "attenu", "diminish", "lower", "prevent", "damag", "degrad",
    "compromis", "worsen", "deplet", "destabili", "dysregulat", "dysfunction"
]

NEUTRAL_INDICATORS = [
    "cause", "lead", "result", "associat", "link", "correlat", "affect",
    "influenc", "modulat", "regulat", "mediat", "involv", "contribut"
]

# Prefixes that flip direction (e.g., de-activate, un-inhibit)
NEGATIVE_PREFIXES = ['de', 'dis', 'un', 'in', 'non', 'anti', 'counter', 'mis']


# =============================================================================
# MEDICAL SYNONYM DICTIONARY
# =============================================================================
# Manually curated, deterministic, zero hallucination
# Every entry is verified. Add new synonyms carefully!

MEDICAL_SYNONYMS: Dict[str, List[str]] = {
    # COVID-related
    "SARS-CoV-2": ["COVID-19", "coronavirus", "SARS2", "COVID", "SARS-CoV-2 virus", "novel coronavirus"],
    "Long COVID": ["PASC", "post-acute COVID", "post-COVID syndrome", "long-haul COVID", "post-COVID-19 syndrome", "chronic COVID"],

    # Symptoms
    "fatigue": ["exhaustion", "tiredness", "asthenia", "lethargy", "chronic fatigue", "persistent fatigue"],
    "cognitive dysfunction": ["brain fog", "cognitive impairment", "mental fog", "cognitive deficits"],
    "dyspnea": ["shortness of breath", "breathlessness", "difficulty breathing"],

    # Cellular/Molecular
    "mitochondrial dysfunction": ["mito dysfunction", "mitochondrial impairment", "mitochondrial damage", "impaired mitochondrial function"],
    "oxidative stress": ["ROS", "reactive oxygen species", "oxidative damage", "ROS production", "oxidative injury"],
    "inflammation": ["inflammatory response", "neuroinflammation", "systemic inflammation", "chronic inflammation"],
    "apoptosis": ["cell death", "programmed cell death", "cellular apoptosis"],

    # Pathways
    "NAD+": ["nicotinamide adenine dinucleotide", "NAD", "NADH", "NAD+ levels"],
    "ATP": ["adenosine triphosphate", "ATP production", "cellular ATP"],
    "AMPK": ["AMP-activated protein kinase", "AMPK pathway", "AMPK signaling"],
    "SIRT1": ["sirtuin 1", "SIRT1 pathway", "SIRT1 activation"],
    "PGC-1alpha": ["PGC-1α", "PPARGC1A", "PGC1alpha", "PGC-1a"],
    "NF-kB": ["NF-κB", "NFkB", "nuclear factor kappa B", "NF-kappaB"],

    # Processes
    "mitophagy": ["mitochondrial autophagy", "mitochondrial degradation"],
    "biogenesis": ["mitochondrial biogenesis", "mito biogenesis"],
    "fission": ["mitochondrial fission", "mito fission", "Drp1-mediated fission"],
    "fusion": ["mitochondrial fusion", "mito fusion"],

    # Treatments
    "HBOT": ["hyperbaric oxygen therapy", "hyperbaric oxygen", "HBO therapy"],
    "NMN": ["nicotinamide mononucleotide", "NMN supplementation"],
    "resveratrol": ["RSV", "trans-resveratrol"],
}

# Build reverse lookup for efficiency
_SYNONYM_REVERSE_MAP: Dict[str, str] = {}
for canonical, synonyms in MEDICAL_SYNONYMS.items():
    canonical_lower = canonical.lower()
    _SYNONYM_REVERSE_MAP[canonical_lower] = canonical_lower
    for syn in synonyms:
        _SYNONYM_REVERSE_MAP[syn.lower()] = canonical_lower


def normalize_entity(entity: str) -> tuple:
    """
    Normalize an entity to its canonical form using synonym dictionary.

    Returns:
        Tuple of (canonical_form, was_normalized, original)
    """
    entity_lower = entity.lower().strip()

    # Direct lookup
    if entity_lower in _SYNONYM_REVERSE_MAP:
        return (_SYNONYM_REVERSE_MAP[entity_lower], True, entity)

    # Partial match (entity contains or is contained by a synonym)
    for syn, canonical in _SYNONYM_REVERSE_MAP.items():
        if len(syn) > 5 and len(entity_lower) > 5:
            if syn in entity_lower or entity_lower in syn:
                return (canonical, True, entity)

    return (entity_lower, False, entity)


# =============================================================================
# INFERENCE RULES
# =============================================================================

# Legacy exact-match rules
DEFAULT_INFERENCE_RULES = [
    {
        "name": "transitive_activation",
        "description": "If A activates B and B activates C, then A activates C",
        "pattern": ["activates", "activates"],
        "conclusion": "activates",
        "confidence_modifier": 0.85
    },
    {
        "name": "transitive_inhibition",
        "description": "If A inhibits B and B inhibits C, then A activates C",
        "pattern": ["inhibits", "inhibits"],
        "conclusion": "activates",
        "confidence_modifier": 0.80
    },
    {
        "name": "transitive_inhibition_activation",
        "description": "If A inhibits B and B activates C, then A inhibits C",
        "pattern": ["inhibits", "activates"],
        "conclusion": "inhibits",
        "confidence_modifier": 0.80
    },
    {
        "name": "transitive_causation",
        "description": "If A causes B and B causes C, then A causes C",
        "pattern": ["causes", "causes"],
        "conclusion": "causes",
        "confidence_modifier": 0.75
    },
    {
        "name": "transitive_leads_to",
        "description": "If A leads to B and B leads to C, then A leads to C",
        "pattern": ["leads to", "leads to"],
        "conclusion": "leads to",
        "confidence_modifier": 0.70
    },
    {
        "name": "enables_chain",
        "description": "If A enables B and B enables C, then A enables C",
        "pattern": ["enables", "enables"],
        "conclusion": "enables",
        "confidence_modifier": 0.75
    },
    {
        "name": "requires_chain",
        "description": "If A requires B and B requires C, then A requires C",
        "pattern": ["requires", "requires"],
        "conclusion": "requires",
        "confidence_modifier": 0.85
    }
]

# Generalized direction-based rules
GENERALIZED_INFERENCE_RULES = [
    {
        "name": "positive_chain",
        "description": "If A positively affects B and B positively affects C, then A positively affects C",
        "pattern": ["positive", "positive"],
        "conclusion": "positive",
        "conclusion_text": "indirectly promotes",
        "confidence_modifier": 0.80
    },
    {
        "name": "negative_chain",
        "description": "If A negatively affects B and B negatively affects C, then A positively affects C (double negative)",
        "pattern": ["negative", "negative"],
        "conclusion": "positive",
        "conclusion_text": "indirectly promotes (via inhibition)",
        "confidence_modifier": 0.75
    },
    {
        "name": "positive_negative",
        "description": "If A positively affects B and B negatively affects C, then A negatively affects C",
        "pattern": ["positive", "negative"],
        "conclusion": "negative",
        "conclusion_text": "indirectly inhibits",
        "confidence_modifier": 0.75
    },
    {
        "name": "negative_positive",
        "description": "If A negatively affects B and B positively affects C, then A negatively affects C",
        "pattern": ["negative", "positive"],
        "conclusion": "negative",
        "conclusion_text": "indirectly inhibits (via blocking promoter)",
        "confidence_modifier": 0.75
    },
    {
        "name": "neutral_chain",
        "description": "If A causes B and B causes C, then A causes C",
        "pattern": ["neutral", "neutral"],
        "conclusion": "neutral",
        "conclusion_text": "indirectly leads to",
        "confidence_modifier": 0.70
    },
]

# =============================================================================
# INFERENCE CONFIGURATION
# =============================================================================

INFERENCE_CONFIG = {
    'max_fuzzy_candidates': 50,      # Max candidates per entity for Jaro-Winkler
    'max_inferred_triplets': 50000,  # Hard cap on total inferred triplets
    'timeout_seconds': 300,          # 5 minute timeout for inference
    'min_trigram_overlap': 2,        # Minimum shared trigrams to consider
    'jaro_winkler_threshold': 0.78,  # Similarity threshold for fuzzy matching
    'min_confidence': 0.30,          # Minimum confidence for inferred facts
}
