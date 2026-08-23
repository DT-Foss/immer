"""One-pass verification of drafted Qwen3.8 continuations.

The transport, retry, checkpoint, and greedy-head logic is model-independent,
so Qwen reuses the proven layer-major verifier while supplying its native
three-dimensional hidden-state contract and its own evidence schema.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import numbers
from typing import Any

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

    def verify(
        self,
        prompt_token_ids: Any,
        draft_token_ids: Any,
        *,
        eos_token_id: int | None = None,
        eos_token_ids: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> DraftVerificationReport:
        """Accept either published Qwen stop token at the shared EOS position.

        The official generation config accepts both ``<|im_end|>`` and
        ``<|endoftext|>``.  The generic verifier needs one sentinel only to
        gather the final next-token row; alternate valid sentinels are then
        accepted without another head scan.
        """

        if eos_token_ids is None:
            return super().verify(
                prompt_token_ids,
                draft_token_ids,
                eos_token_id=eos_token_id,
                **kwargs,
            )
        if isinstance(eos_token_ids, (str, bytes)) or not isinstance(
            eos_token_ids, Sequence
        ):
            raise TypeError("eos_token_ids must be a sequence of integers")
        allowed: list[int] = []
        for raw in eos_token_ids:
            if isinstance(raw, bool) or not isinstance(raw, numbers.Integral):
                raise TypeError("eos_token_ids must contain integers")
            token = int(raw)
            if token not in allowed:
                allowed.append(token)
        if not allowed:
            raise ValueError("eos_token_ids must not be empty")
        primary = allowed[0] if eos_token_id is None else eos_token_id
        if isinstance(primary, bool) or not isinstance(primary, numbers.Integral):
            raise TypeError("eos_token_id must be an integer")
        primary = int(primary)
        if primary not in allowed:
            raise ValueError("eos_token_id must be included in eos_token_ids")
        vocab_size = int(self.model.config.vocab_size)
        if any(token < 0 or token >= vocab_size for token in allowed):
            raise ValueError("eos_token_ids contains a token outside the vocabulary")
        raw_drafts = (
            draft_token_ids.tolist()
            if callable(getattr(draft_token_ids, "tolist", None))
            else draft_token_ids
        )

        def contains_stop(value: Any) -> bool:
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return any(contains_stop(item) for item in value)
            return (
                not isinstance(value, bool)
                and isinstance(value, numbers.Integral)
                and int(value) in allowed
            )

        if contains_stop(raw_drafts):
            raise ValueError("draft rows must exclude every optional EOS token")

        report = super().verify(
            prompt_token_ids,
            draft_token_ids,
            eos_token_id=primary,
            **kwargs,
        )
        accepted = frozenset(allowed)
        adjusted = tuple(
            replace(
                row,
                first_mismatch_index=None,
                first_mismatch_target_token_id=None,
                first_mismatch_draft_token_id=None,
                first_mismatch_is_eos=False,
                eos_verified=True,
                fully_verified=True,
            )
            if row.draft_verified
            and row.first_mismatch_is_eos
            and row.first_mismatch_target_token_id in accepted
            else row
            for row in report.rows
        )
        return DraftVerificationReport(adjusted, report.evidence)


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
