"""
dotcausal — vendored minimal bundle (core + io + inference).

Vollständige Pakete (inkl. LangChain-Brücke und CLI) leben im fable-Projekt;
hier ist nur der Teil eingebettet, den fertig.pipeline zum Lesen von
.causal-Graphen braucht. Unverändert kopiert, MIT-Lizenz, David Tom Foss.
"""

from .core import (
    MAGIC,
    LEGACY_MAGIC,
    VERSION,
    HEADER_SIZE,
    OFFSET_TABLE_SIZE,
    CausalTriplet,
    SemanticCluster,
    KnowledgeGap,
    DEFAULT_INFERENCE_RULES,
)
from .io import (
    CausalWriter,
    CausalReader,
    CausalFile,
    CausalStorageBackend,
)
from .inference import run_inference

__version__ = "0.3.1"

__all__ = [
    "MAGIC", "LEGACY_MAGIC", "VERSION", "HEADER_SIZE", "OFFSET_TABLE_SIZE",
    "CausalTriplet", "SemanticCluster", "KnowledgeGap", "DEFAULT_INFERENCE_RULES",
    "CausalWriter", "CausalReader", "CausalFile", "CausalStorageBackend",
    "run_inference",
]
