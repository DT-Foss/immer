"""
HSSLM-C — Hierarchical State-Space Language Module with Causal/Contraction

~5-6M parameters | O(1) per-layer convergence | Deterministic inference
Moebius coupling + PS-Lifted Z2 + Ginibre init + Foss Gate
"""

__version__ = "2.0.0-causal"

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "HSSLM":
        from .model import HSSLM
        return HSSLM
    elif name == "HSSLMC":
        from .model import HSSLMC
        return HSSLMC
    elif name == "HSSLMConfig":
        from .config import HSSLMConfig
        return HSSLMConfig
    elif name in ("MoebiusCoupling", "ContractiveSSM", "PSLiftedBlock", "MoebiusStateSpaceCore"):
        from . import moebius_ssm
        return getattr(moebius_ssm, name)
    elif name in ("CausalInferenceEngine", "WeakSignalAmplifier", "JaroWinkler", "MoebiusConfidence"):
        from . import causal_inference
        return getattr(causal_inference, name)
    elif name in ("GinibreInitializer", "ginibre_init_"):
        from . import ginibre_init
        return getattr(ginibre_init, name)
    elif name in ("FossGate", "NeuralFossGate"):
        from . import foss_gate
        return getattr(foss_gate, name)
    elif name in ("ContractionSampler", "Z2TopologicalLift", "BvNPathIntegralSampler"):
        from . import contraction_inference
        return getattr(contraction_inference, name)
    raise AttributeError(f"module 'hsslm' has no attribute '{name}'")
