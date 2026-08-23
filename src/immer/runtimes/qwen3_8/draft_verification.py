"""One-pass verification of drafted Qwen3.8 continuations.

The transport, retry, checkpoint, and greedy-head logic is model-independent,
so Qwen reuses the proven layer-major verifier while supplying its native
three-dimensional hidden-state contract and its own evidence schema.
"""

from __future__ import annotations

from ..deepseek_v4.draft_verification import (
    DraftRowVerification,
    DraftVerificationEvidence,
    DraftVerificationReport,
    DraftVerificationResumeState,
    LayerwiseDraftVerifier as _LayerwiseDraftVerifier,
)


DRAFT_VERIFICATION_SCHEMA = "immer.qwen3.8-draft-verification/v1"


class Qwen38DraftVerifier(_LayerwiseDraftVerifier):
    """Verify every supplied continuation in one Qwen weight pass."""

    evidence_schema = DRAFT_VERIFICATION_SCHEMA


LayerwiseDraftVerifier = Qwen38DraftVerifier


__all__ = [
    "DRAFT_VERIFICATION_SCHEMA",
    "DraftRowVerification",
    "DraftVerificationEvidence",
    "DraftVerificationReport",
    "DraftVerificationResumeState",
    "LayerwiseDraftVerifier",
    "Qwen38DraftVerifier",
]
