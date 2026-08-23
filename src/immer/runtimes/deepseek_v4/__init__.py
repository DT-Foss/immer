"""Apple-local, range-streamed DeepSeek-V4 runtime.

The runtime deliberately keeps model acquisition separate from execution:
``Streamer`` supplies exact safetensors ranges while this package implements
the model math.  No complete checkpoint is required on disk or in memory.
"""

from .config import DeepSeekV4Config
from .causal_weights import (
    CAUSAL_WEIGHT_BINDING_SCHEMA,
    CausalWeightBindingReceipt,
    CausalWeightConflictError,
    CausalWeightError,
    CausalWeightIdentityError,
    CausalWeightIntegrityError,
    CausalWeightLayoutIdentity,
    CausalWeightLeaf,
    CausalWeightMount,
    CausalWeightNotFoundError,
    CausalWeightReadReceipt,
    CausalWeightReader,
    ExpertBindingReceipt,
    LogicalModelIdentity,
    bind_causal_weight_plans,
    semantic_expert_key,
)
from .pager import (
    DeepSeekWeightPager,
    ExpertSourceRange,
    ExpertTensorLayout,
    OfficialExpertRangePlan,
)
from .model import (
    GenerationEvidence,
    OneTokenEvidence,
    StatefulEvidence,
    StreamedDeepSeekV4,
)
from .quantization import (
    dequantize_fp4_e2m1,
    dequantize_fp8_e4m3,
    quantize_dequantize_fp4,
    quantize_dequantize_fp8,
    quantize_fp8_e4m3_parts,
    unpack_fp4_e2m1,
)
from .snapshot import DeepSeekV4SnapshotError, SnapshotLimits
from .provenance import runtime_dependency_versions, runtime_source_manifest
from .layerwise import (
    LAYERWISE_SCHEMA,
    OFFICIAL_SOURCE_SAFE_BYTES,
    LayerwiseError,
    LayerwiseItem,
    LayerwisePlan,
    LayerwiseScorer,
    build_layerwise_plan,
)

__all__ = [
    "CAUSAL_WEIGHT_BINDING_SCHEMA",
    "CausalWeightBindingReceipt",
    "CausalWeightConflictError",
    "CausalWeightError",
    "CausalWeightIdentityError",
    "CausalWeightIntegrityError",
    "CausalWeightLayoutIdentity",
    "CausalWeightLeaf",
    "CausalWeightMount",
    "CausalWeightNotFoundError",
    "CausalWeightReadReceipt",
    "CausalWeightReader",
    "DeepSeekV4Config",
    "DeepSeekV4SnapshotError",
    "DeepSeekWeightPager",
    "ExpertBindingReceipt",
    "ExpertSourceRange",
    "ExpertTensorLayout",
    "GenerationEvidence",
    "LAYERWISE_SCHEMA",
    "LayerwiseError",
    "LayerwiseItem",
    "LayerwisePlan",
    "LayerwiseScorer",
    "LogicalModelIdentity",
    "OFFICIAL_SOURCE_SAFE_BYTES",
    "OneTokenEvidence",
    "OfficialExpertRangePlan",
    "StatefulEvidence",
    "StreamedDeepSeekV4",
    "SnapshotLimits",
    "build_layerwise_plan",
    "bind_causal_weight_plans",
    "dequantize_fp4_e2m1",
    "dequantize_fp8_e4m3",
    "quantize_dequantize_fp4",
    "quantize_dequantize_fp8",
    "quantize_fp8_e4m3_parts",
    "runtime_dependency_versions",
    "runtime_source_manifest",
    "semantic_expert_key",
    "unpack_fp4_e2m1",
]
