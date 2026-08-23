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
from .route_markov import (
    LayerMarkovExpertPredictor,
    LayerMicroWindowPlan,
    MicroWindowPrediction,
    plan_micro_window_prefetch,
)
from .route_model import (
    ROUTE_MODEL_SCHEMA,
    RouteModelArtifact,
    RouteModelArtifactError,
    build_route_model_artifact,
    load_route_model_artifact,
    write_route_model_artifact,
)
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
    "LayerMarkovExpertPredictor",
    "LayerMicroWindowPlan",
    "LogicalModelIdentity",
    "OFFICIAL_SOURCE_SAFE_BYTES",
    "OneTokenEvidence",
    "OfficialExpertRangePlan",
    "MicroWindowPrediction",
    "ROUTE_MODEL_SCHEMA",
    "RouteModelArtifact",
    "RouteModelArtifactError",
    "StatefulEvidence",
    "StreamedDeepSeekV4",
    "SnapshotLimits",
    "build_layerwise_plan",
    "build_route_model_artifact",
    "bind_causal_weight_plans",
    "dequantize_fp4_e2m1",
    "dequantize_fp8_e4m3",
    "quantize_dequantize_fp4",
    "quantize_dequantize_fp8",
    "quantize_fp8_e4m3_parts",
    "plan_micro_window_prefetch",
    "load_route_model_artifact",
    "runtime_dependency_versions",
    "runtime_source_manifest",
    "semantic_expert_key",
    "unpack_fp4_e2m1",
    "write_route_model_artifact",
]
