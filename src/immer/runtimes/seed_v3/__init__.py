"""Trainable receipt-native Seed v3 shadow proposer.

PyTorch remains an optional ``immer[neural]`` dependency.  The configuration
ABI can be imported without it; model, training, and checkpoint symbols appear
when the neural extra is installed.
"""

from .config import SeedConfig, SeedV3Config

__all__ = ["SeedConfig", "SeedV3Config"]

try:
    from .checkpoint import (
        SEED_V3_CHECKPOINT_SCHEMA,
        SEED_V3_LAB_SOURCE_SCHEMA,
        SEED_V3_MODEL_ABI,
        SEED_V3_WEIGHTS_FORMAT,
        SeedV3CheckpointError,
        SeedV3CheckpointIdentityError,
        SeedV3CheckpointIntegrityError,
        SeedV3CheckpointManifest,
        SeedV3CheckpointPublication,
        SeedV3StateTensor,
        export_seed_v3_checkpoint,
        load_seed_v3_checkpoint,
        migrate_seed_v3_lab_checkpoint,
    )
    from .model import (
        CausalPrefixSinkhornAttention,
        ImmerSeedModel,
        ImmerSeedV3,
        KeyedSwiGLU,
        SeedOutput,
        SeedV3Block,
        SeedV3Output,
        SeedV3Proposal,
    )
    from .projection import (
        SeedV3ProjectionViews,
        TargetProjector,
        seed_v3_projection_views,
        shadow_batch_from_projection,
    )
    from .training import (
        SeedV3LossWeights,
        SeedV3ShadowBatch,
        SeedV3ShadowTrainer,
        seed_v3_shadow_losses,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__ += [
        "SEED_V3_CHECKPOINT_SCHEMA",
        "SEED_V3_LAB_SOURCE_SCHEMA",
        "SEED_V3_MODEL_ABI",
        "SEED_V3_WEIGHTS_FORMAT",
        "CausalPrefixSinkhornAttention",
        "ImmerSeedModel",
        "ImmerSeedV3",
        "KeyedSwiGLU",
        "SeedOutput",
        "SeedV3Block",
        "SeedV3CheckpointError",
        "SeedV3CheckpointIdentityError",
        "SeedV3CheckpointIntegrityError",
        "SeedV3CheckpointManifest",
        "SeedV3CheckpointPublication",
        "SeedV3StateTensor",
        "SeedV3LossWeights",
        "SeedV3Output",
        "SeedV3Proposal",
        "SeedV3ProjectionViews",
        "SeedV3ShadowBatch",
        "SeedV3ShadowTrainer",
        "TargetProjector",
        "export_seed_v3_checkpoint",
        "load_seed_v3_checkpoint",
        "migrate_seed_v3_lab_checkpoint",
        "seed_v3_projection_views",
        "seed_v3_shadow_losses",
        "shadow_batch_from_projection",
    ]
