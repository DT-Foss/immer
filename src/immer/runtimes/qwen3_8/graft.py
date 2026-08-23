"""Qwen-facing name for IMMER's norm-stable causal CRSA sidecar."""

from __future__ import annotations

from ..deepseek_v4.stable_graft import (
    STABLE_GRAFT_POLICY,
    DeepSeekV4StableCrsaGraft,
    StableGraftEvidence,
)


STABLE_GRAFT_EVIDENCE_SCHEMA = "immer.qwen3.8-stable-crsa-graft/v1"


class Qwen38StableCrsaGraft(DeepSeekV4StableCrsaGraft):
    """Energy-preserving CRSA over native ``[batch, sequence, dim]`` states."""

    evidence_schema = STABLE_GRAFT_EVIDENCE_SCHEMA


StableCrsaGraft = Qwen38StableCrsaGraft


__all__ = [
    "Qwen38StableCrsaGraft",
    "STABLE_GRAFT_EVIDENCE_SCHEMA",
    "STABLE_GRAFT_POLICY",
    "StableCrsaGraft",
    "StableGraftEvidence",
]
