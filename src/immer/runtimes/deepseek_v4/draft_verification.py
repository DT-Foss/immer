"""One-pass greedy verification of externally drafted V4 continuations.

Draft verification is cheaper than autoregressive decoding when a caller can
propose several continuation tokens.  The verifier right-pads ``prompt+draft``
rows, executes each streamed decoder layer exactly once, and scans the full
vocabulary head once for every required next-token position together.

The optional EOS token is a verification target, not part of the prefill.  A
draft therefore never needs to include EOS merely to prove that the model
would stop after it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
import numbers
import time
from typing import Any

import torch


DRAFT_VERIFICATION_SCHEMA = "immer.deepseek-v4-draft-verification/v1"


@dataclass(frozen=True, slots=True)
class DraftRowVerification:
    """Greedy agreement for one supplied continuation row."""

    row: int
    prompt_length: int
    draft_token_ids: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    accepted_prefix_length: int
    first_mismatch_index: int | None
    first_mismatch_target_token_id: int | None
    first_mismatch_draft_token_id: int | None
    first_mismatch_is_eos: bool
    draft_verified: bool
    eos_verified: bool | None
    fully_verified: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DraftVerificationEvidence:
    """Compact execution accounting for one batched verification pass."""

    schema: str
    batch_size: int
    prompt_lengths: tuple[int, ...]
    draft_lengths: tuple[int, ...]
    padded_sequence_length: int
    verification_rows: int
    fixed_model_batch: bool
    right_prefix_mask: bool
    eos_token_id: int | None
    layer_calls: int
    layer_retry_count: int
    head_scans: int
    head_retry_count: int
    source_body_bytes: int
    linear_calls: int
    seconds: float
    graft_mode: str
    graft_layer: int | None
    graft_applied: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DraftVerificationReport:
    """Per-row decisions and their shared execution evidence."""

    rows: tuple[DraftRowVerification, ...]
    evidence: DraftVerificationEvidence

    @property
    def all_verified(self) -> bool:
        return all(row.fully_verified for row in self.rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": [row.to_dict() for row in self.rows],
            "evidence": self.evidence.to_dict(),
            "all_verified": self.all_verified,
        }


def _metric_int(owner: Any, key: str) -> int:
    metrics = owner.metrics()
    raw = metrics.get(key, 0)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 0
    return int(raw)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must contain integers")
    return int(value)


def _rows(
    value: Any,
    name: str,
    *,
    empty_as_single_row: bool,
) -> tuple[tuple[int, ...], ...]:
    if isinstance(value, torch.Tensor):
        if value.ndim not in (1, 2):
            raise ValueError(f"{name} must have one or two dimensions")
        if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
            raise TypeError(f"{name} must contain integers")
        value = value.detach().to(device="cpu").tolist()
    elif not isinstance(value, Sequence) and callable(getattr(value, "tolist", None)):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a token row or sequence of token rows")

    outer = list(value)
    if not outer:
        return ((),) if empty_as_single_row else ()
    nested = [
        isinstance(item, Sequence) and not isinstance(item, (str, bytes))
        for item in outer
    ]
    if any(nested):
        if not all(nested):
            raise TypeError(f"{name} rows must all be integer sequences")
        return tuple(tuple(_integer(token, name) for token in row) for row in outer)
    return (tuple(_integer(token, name) for token in outer),)


def _optional_nonnegative_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or None")
    return value


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _transient_failure(error: Exception) -> bool:
    """Recognize transport failures, including errors wrapped by the source."""

    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (TimeoutError, ConnectionError)):
            return True
        module = type(current).__module__
        name = type(current).__name__
        if module.startswith(("requests.", "urllib3.", "httpcore.", "httpx.")):
            if name in {
                "ChunkedEncodingError",
                "ConnectError",
                "ConnectTimeout",
                "ConnectionError",
                "PoolTimeout",
                "ReadError",
                "ReadTimeout",
                "RemoteProtocolError",
                "Timeout",
            }:
                return True
        if module == "urllib.error" and name == "URLError":
            return True
        current = current.__cause__ or current.__context__
    return False


class LayerwiseDraftVerifier:
    """Verify a batch of draft continuations with one layer-major prefill."""

    def __init__(
        self,
        model: Any,
        *,
        layer_retries: int = 2,
        head_retries: int = 2,
    ) -> None:
        required = (
            "config",
            "pager",
            "max_batch_size",
            "max_seq_len",
            "embed_batch",
            "forward_prefill_layer",
            "finalize_hidden",
        )
        if any(not hasattr(model, name) for name in required):
            raise TypeError("model lacks the draft-verification contract")
        if not callable(getattr(model.pager, "topk_logits", None)):
            raise TypeError("model pager lacks exact topk_logits()")
        self.model = model
        retries = _optional_nonnegative_int(layer_retries, "layer_retries")
        assert retries is not None
        self.layer_retries = retries
        retries = _optional_nonnegative_int(head_retries, "head_retries")
        assert retries is not None
        self.head_retries = retries

    def _graft_contract(self) -> tuple[Any | None, int | None, str]:
        graft = getattr(self.model, "graft", None)
        layer = getattr(self.model, "graft_layer", None)
        if graft is None:
            return None, None, "off"
        mode = str(getattr(graft, "mode", "active"))
        raw_alpha = getattr(graft, "alpha", None)
        if raw_alpha is not None:
            try:
                alpha = float(raw_alpha)
            except (TypeError, ValueError) as exc:
                raise TypeError("graft alpha must be numeric") from exc
        else:
            alpha = None
        if mode == "off" or alpha == 0.0:
            return None, layer, mode
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise ValueError("active graft requires an integer graft_layer")
        if not 0 <= layer < int(self.model.config.n_layers):
            raise ValueError("graft_layer outside decoder depth")
        if not callable(getattr(graft, "forward", None)):
            raise TypeError("active graft must provide forward()")
        return graft, layer, mode

    def verify(
        self,
        prompt_token_ids: Any,
        draft_token_ids: Any,
        *,
        eos_token_id: int | None = None,
        padding_token_id: int = 0,
        max_draft_tokens: int | None = None,
        head_block_rows: int = 1024,
        progress: Callable[[dict[str, Any]], None] | None = None,
        head_progress: Callable[[dict[str, int]], None] | None = None,
    ) -> DraftVerificationReport:
        """Compare supplied drafts with exact greedy V4 next-token targets.

        ``accepted_prefix_length`` counts draft tokens only.  When every draft
        token matches but the optional EOS does not, the full draft length is
        accepted while ``fully_verified`` is false.
        """

        prompts = _rows(
            prompt_token_ids,
            "prompt_token_ids",
            empty_as_single_row=False,
        )
        drafts = _rows(
            draft_token_ids,
            "draft_token_ids",
            empty_as_single_row=len(prompts) == 1,
        )
        if not prompts or any(not row for row in prompts):
            raise ValueError("prompt rows must be non-empty")
        if len(drafts) != len(prompts):
            raise ValueError("prompt and draft batches must have the same size")
        batch = len(prompts)
        if batch > int(self.model.max_batch_size):
            raise ValueError("verification batch exceeds model max_batch_size")

        vocab_size = int(self.model.config.vocab_size)
        padding = _integer(padding_token_id, "padding_token_id")
        if not 0 <= padding < vocab_size:
            raise ValueError("padding_token_id outside checkpoint vocabulary")
        eos = None if eos_token_id is None else _integer(eos_token_id, "eos_token_id")
        if eos is not None and not 0 <= eos < vocab_size:
            raise ValueError("eos_token_id outside checkpoint vocabulary")
        maximum = _optional_nonnegative_int(max_draft_tokens, "max_draft_tokens")
        block_rows = _positive_int(head_block_rows, "head_block_rows")
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable or None")

        for row in prompts:
            if any(token < 0 or token >= vocab_size for token in row):
                raise ValueError("prompt token outside checkpoint vocabulary")
        for row in drafts:
            if maximum is not None and len(row) > maximum:
                raise ValueError("draft row exceeds max_draft_tokens")
            if not row and eos is None:
                raise ValueError("every row needs a draft token or optional EOS")
            if any(token < 0 or token >= vocab_size for token in row):
                raise ValueError("draft token outside checkpoint vocabulary")
            if eos is not None and eos in row:
                raise ValueError("draft rows must exclude the optional EOS token")

        combined = tuple(
            prompt + draft for prompt, draft in zip(prompts, drafts, strict=True)
        )
        lengths = tuple(len(row) for row in combined)
        padded_length = max(lengths)
        if padded_length > int(self.model.max_seq_len):
            raise ValueError("prompt plus draft exceeds model context bound")

        ids = torch.full((batch, padded_length), padding, dtype=torch.long)
        mask = torch.zeros((batch, padded_length), dtype=torch.bool)
        for row_index, row in enumerate(combined):
            length = len(row)
            ids[row_index, :length] = torch.tensor(row, dtype=torch.long)
            mask[row_index, :length] = True

        graft, graft_layer, graft_mode = self._graft_contract()
        reset_state = getattr(self.model, "reset_state", None)
        if callable(reset_state):
            reset_state(release=True)
        source = self.model.pager.source
        start_bytes = _metric_int(source, "network_or_source_body_bytes")
        start_linears = _metric_int(self.model.pager, "linear_calls")
        started = time.perf_counter()
        layer_calls = 0
        layer_retry_count = 0
        graft_applied = False
        try:
            hidden = self.model.embed_batch(ids)
            for layer in range(int(self.model.config.n_layers)):
                retries = 0
                while True:
                    try:
                        next_hidden, _experts = self.model.forward_prefill_layer(
                            hidden,
                            ids,
                            layer=layer,
                            token_mask=mask,
                        )
                        break
                    except Exception as exc:
                        self.model.pager.release()
                        if (
                            not _transient_failure(exc)
                            or retries >= self.layer_retries
                        ):
                            raise
                        retries += 1
                        if progress is not None:
                            progress(
                                {
                                    "event": "layer_retry",
                                    "layer": layer,
                                    "attempt": retries + 1,
                                    "error_type": type(exc).__name__,
                                    "error": str(exc),
                                }
                            )
                hidden = next_hidden
                layer_calls += 1
                layer_retry_count += retries
                if graft is not None and layer == graft_layer:
                    grafted = graft.forward(hidden)
                    hidden = grafted[0] if isinstance(grafted, tuple) else grafted
                    if not isinstance(hidden, torch.Tensor):
                        raise TypeError("graft.forward() must return a torch tensor")
                    expected_shape = (
                        batch,
                        padded_length,
                        int(self.model.config.hc_mult),
                        int(self.model.config.dim),
                    )
                    if tuple(hidden.shape) != expected_shape:
                        raise ValueError("graft changed the padded hidden shape")
                    graft_applied = True
                self.model.pager.release()
                if progress is not None:
                    progress(
                        {
                            "event": "layer_complete",
                            "layer": layer,
                            "layers": int(self.model.config.n_layers),
                            "retries": retries,
                        }
                    )

            final = self.model.finalize_hidden(hidden)
            gathered: list[torch.Tensor] = []
            row_spans: list[tuple[int, int]] = []
            expected_rows: list[tuple[int, ...]] = []
            cursor = 0
            for row, (prompt, draft) in enumerate(zip(prompts, drafts, strict=True)):
                expected = draft + (() if eos is None else (eos,))
                expected_rows.append(expected)
                row_spans.append((cursor, cursor + len(expected)))
                for offset in range(len(expected)):
                    position = len(prompt) - 1 + offset
                    gathered.append(final[row, position])
                cursor += len(expected)
            verification_hidden = torch.stack(gathered, dim=0)
            head_retries = 0
            while True:
                try:
                    _values, selected = self.model.pager.topk_logits(
                        verification_hidden,
                        k=1,
                        block_rows=block_rows,
                        progress=head_progress,
                    )
                    break
                except Exception as exc:
                    self.model.pager.release()
                    if (
                        not _transient_failure(exc)
                        or head_retries >= self.head_retries
                    ):
                        raise
                    head_retries += 1
                    if progress is not None:
                        progress(
                            {
                                "event": "head_retry",
                                "attempt": head_retries + 1,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
            if selected.ndim != 2 or tuple(selected.shape) != (cursor, 1):
                raise RuntimeError(
                    "topk_logits returned an invalid greedy target shape"
                )
            targets = tuple(
                int(token)
                for token in selected[:, 0].detach().to(device="cpu").tolist()
            )
        finally:
            self.model.pager.release()

        rows: list[DraftRowVerification] = []
        for row_index, (draft, expected, span) in enumerate(
            zip(drafts, expected_rows, row_spans, strict=True)
        ):
            target = targets[span[0] : span[1]]
            mismatch = next(
                (
                    index
                    for index, (observed, supplied) in enumerate(
                        zip(target, expected, strict=True)
                    )
                    if observed != supplied
                ),
                None,
            )
            accepted = len(draft) if mismatch is None else min(mismatch, len(draft))
            draft_verified = all(
                target[index] == token for index, token in enumerate(draft)
            )
            eos_verified = (
                None
                if eos is None
                or (mismatch is not None and mismatch < len(draft))
                else target[-1] == eos
            )
            rows.append(
                DraftRowVerification(
                    row=row_index,
                    prompt_length=len(prompts[row_index]),
                    draft_token_ids=draft,
                    target_token_ids=target,
                    accepted_prefix_length=accepted,
                    first_mismatch_index=mismatch,
                    first_mismatch_target_token_id=(
                        None if mismatch is None else target[mismatch]
                    ),
                    first_mismatch_draft_token_id=(
                        None if mismatch is None else expected[mismatch]
                    ),
                    first_mismatch_is_eos=(
                        mismatch is not None
                        and mismatch == len(draft)
                        and eos is not None
                    ),
                    draft_verified=draft_verified,
                    eos_verified=eos_verified,
                    fully_verified=mismatch is None,
                )
            )

        end_bytes = _metric_int(source, "network_or_source_body_bytes")
        end_linears = _metric_int(self.model.pager, "linear_calls")
        evidence = DraftVerificationEvidence(
            schema=DRAFT_VERIFICATION_SCHEMA,
            batch_size=batch,
            prompt_lengths=tuple(len(row) for row in prompts),
            draft_lengths=tuple(len(row) for row in drafts),
            padded_sequence_length=padded_length,
            verification_rows=len(gathered),
            fixed_model_batch=True,
            right_prefix_mask=True,
            eos_token_id=eos,
            layer_calls=layer_calls,
            layer_retry_count=layer_retry_count,
            head_scans=1,
            head_retry_count=head_retries,
            source_body_bytes=end_bytes - start_bytes,
            linear_calls=end_linears - start_linears,
            seconds=time.perf_counter() - started,
            graft_mode=graft_mode,
            graft_layer=graft_layer,
            graft_applied=graft_applied,
        )
        return DraftVerificationReport(tuple(rows), evidence)


__all__ = [
    "DRAFT_VERIFICATION_SCHEMA",
    "DraftRowVerification",
    "DraftVerificationEvidence",
    "DraftVerificationReport",
    "LayerwiseDraftVerifier",
]
