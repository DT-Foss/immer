"""Pinned, text-only, range-streamed Qwen3.8 runtime."""

from .adapter import Qwen38CausalChat, Qwen38Chat, Qwen38ChatError
from ..deepseek_v4.causal_weights import (
    CausalTensorReader,
    CausalWeightMount,
    LogicalModelIdentity,
    TensorRangePlan,
    bind_causal_tensor_plans,
    tensor_range_plan_from_source,
)
from .config import (
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
    Qwen38Config,
    Qwen38ConfigError,
)
from .bundle import Qwen38BundleError, verify_qwen38_causal_mount
from .draft_verification import Qwen38DraftVerifier
from .encoding import (
    END_OF_TEXT_TOKEN_ID,
    IM_END_TOKEN_ID,
    IM_START_TOKEN_ID,
    Qwen38EncodingError,
    Qwen38Tokenizer,
)
from .graft import Qwen38StableCrsaGraft
from .kernels import AttentionState, DeltaNetProbe, DeltaNetState
from .model import (
    GenerationEvidence,
    PrefillEvidence,
    Qwen38RuntimeError,
    StatefulEvidence,
    StatefulLayerRangeEvidence,
    StatefulLayerRangeResult,
    StreamedQwen38,
)
from .native_crsa import (
    NATIVE_HEAD_CRSA_EVIDENCE_SCHEMA,
    NATIVE_HEAD_CRSA_FREE_HEADS,
    NATIVE_HEAD_CRSA_KV_HEADS,
    NATIVE_HEAD_CRSA_LAYER,
    NATIVE_HEAD_CRSA_QUERY_HEADS,
    NATIVE_HEAD_CRSA_ROW_SUM_TOLERANCE,
    NativeHeadCrsaEvidence,
    Qwen38NativeHeadCrsa,
)
from .native_fork import (
    Qwen38ForkArmState,
    Qwen38ForkForwardResult,
    Qwen38ForkGenerationEvidence,
    Qwen38ForkGenerationResult,
    Qwen38ForkLayerAccounting,
    Qwen38ForkState,
    Qwen38ForkTraffic,
    Qwen38NativeFork,
)
from .pager import Qwen38PagerError, Qwen38WeightPager
from .probe import (
    DELTANET_COMPARISON_SCHEMA,
    DELTANET_COMPONENTS,
    DELTANET_PROBE_SCHEMA,
    DeltaNetProbeRecorder,
    Qwen38DeltaNetProbeError,
    build_probe_document,
    compare_probe_documents,
    verify_probe_document,
)
from .snapshot import QWEN38_SNAPSHOT_SCHEMA, Qwen38SnapshotError


__all__ = [
    "AttentionState",
    "CausalTensorReader",
    "CausalWeightMount",
    "DeltaNetState",
    "DeltaNetProbe",
    "DeltaNetProbeRecorder",
    "DELTANET_COMPARISON_SCHEMA",
    "DELTANET_COMPONENTS",
    "DELTANET_PROBE_SCHEMA",
    "END_OF_TEXT_TOKEN_ID",
    "GenerationEvidence",
    "IM_END_TOKEN_ID",
    "IM_START_TOKEN_ID",
    "LogicalModelIdentity",
    "NATIVE_HEAD_CRSA_EVIDENCE_SCHEMA",
    "NATIVE_HEAD_CRSA_FREE_HEADS",
    "NATIVE_HEAD_CRSA_KV_HEADS",
    "NATIVE_HEAD_CRSA_LAYER",
    "NATIVE_HEAD_CRSA_QUERY_HEADS",
    "NATIVE_HEAD_CRSA_ROW_SUM_TOLERANCE",
    "NativeHeadCrsaEvidence",
    "OFFICIAL_REPO_ID",
    "OFFICIAL_REVISION",
    "PrefillEvidence",
    "Qwen38Config",
    "Qwen38ConfigError",
    "Qwen38CausalChat",
    "Qwen38Chat",
    "Qwen38ChatError",
    "Qwen38BundleError",
    "Qwen38DraftVerifier",
    "Qwen38DeltaNetProbeError",
    "Qwen38EncodingError",
    "Qwen38NativeHeadCrsa",
    "Qwen38NativeFork",
    "Qwen38ForkArmState",
    "Qwen38ForkForwardResult",
    "Qwen38ForkGenerationEvidence",
    "Qwen38ForkGenerationResult",
    "Qwen38ForkLayerAccounting",
    "Qwen38ForkState",
    "Qwen38ForkTraffic",
    "Qwen38PagerError",
    "Qwen38RuntimeError",
    "Qwen38SnapshotError",
    "Qwen38StableCrsaGraft",
    "Qwen38Tokenizer",
    "Qwen38WeightPager",
    "QWEN38_SNAPSHOT_SCHEMA",
    "StatefulEvidence",
    "StatefulLayerRangeEvidence",
    "StatefulLayerRangeResult",
    "StreamedQwen38",
    "TensorRangePlan",
    "bind_causal_tensor_plans",
    "build_probe_document",
    "compare_probe_documents",
    "tensor_range_plan_from_source",
    "verify_probe_document",
    "verify_qwen38_causal_mount",
]
