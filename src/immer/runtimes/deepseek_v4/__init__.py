"""Apple-local, range-streamed DeepSeek-V4 runtime.

The runtime deliberately keeps model acquisition separate from execution:
``Streamer`` supplies exact safetensors ranges while this package implements
the model math.  No complete checkpoint is required on disk or in memory.
"""

from .config import DeepSeekV4Config
from .pager import DeepSeekWeightPager
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
from .provenance import runtime_source_manifest
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
    "DeepSeekV4Config",
    "DeepSeekV4SnapshotError",
    "DeepSeekWeightPager",
    "GenerationEvidence",
    "LAYERWISE_SCHEMA",
    "LayerwiseError",
    "LayerwiseItem",
    "LayerwisePlan",
    "LayerwiseScorer",
    "OFFICIAL_SOURCE_SAFE_BYTES",
    "OneTokenEvidence",
    "StatefulEvidence",
    "StreamedDeepSeekV4",
    "SnapshotLimits",
    "build_layerwise_plan",
    "dequantize_fp4_e2m1",
    "dequantize_fp8_e4m3",
    "quantize_dequantize_fp4",
    "quantize_dequantize_fp8",
    "quantize_fp8_e4m3_parts",
    "runtime_source_manifest",
    "unpack_fp4_e2m1",
]
