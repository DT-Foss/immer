"""Exact-length batched autoregressive generation in layer-major order.

The streamed runtime cannot keep all 43 decoder layers and their mutable KV
state resident together.  ``LayerwiseGenerator`` therefore owns one immutable
CPU attention-state handle per layer.  For every prefill/decode step it restores
one layer, executes it, captures the updated handle, and releases the resident
state before advancing to the next layer.

Only equal-length prompt batches are accepted.  V4's local/compressed caches
have a scalar cursor shared by the physical batch, so a padded short row could
not later be decoded at its true position.  Finished EOS rows remain in the
fixed batch, receive a caller-selected filler token, and are masked out of MoE;
their outputs are ignored while unfinished rows continue exactly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from collections.abc import Callable, Sequence
from typing import Any

import torch

from .layer_state import LayerAttentionState, LayerStateRunner


LAYERWISE_GENERATION_SCHEMA = "immer.deepseek-v4-layerwise-generation/v1"


@dataclass(frozen=True, slots=True)
class LayerwiseGenerationEvidence:
    """Compact evidence for one exact-length layer-major generation."""

    schema: str
    prompt_token_ids: tuple[tuple[int, ...], ...]
    generated_token_ids: tuple[tuple[int, ...], ...]
    batch_size: int
    prompt_tokens: int
    max_new_tokens: int
    context_mode: str
    stateful_kv_cache: bool
    exact_length_batch: bool
    forward_layer_calls: int
    layer_retry_count: int
    head_scans: int
    source_body_bytes: int
    linear_calls: int
    seconds: float
    attention_state_bytes: int
    stopped_on_eos: tuple[bool, ...]
    graft_mode: str
    graft_layer: int | None
    graft_history_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _metric_int(value: Any, key: str) -> int:
    metrics = value.metrics()
    raw = metrics.get(key, 0)
    return int(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else 0


def _state_nbytes(states: Sequence[LayerAttentionState]) -> int:
    total = 0
    for state in states:
        total += sum(
            tensor.numel() * tensor.element_size()
            for tensor in state.tensors.values()
        )
    return total


class LayerwiseGenerator:
    """Run exact greedy generation with one resident decoder layer at a time."""

    def __init__(
        self,
        model: Any,
        *,
        runner: LayerStateRunner | None = None,
        layer_retries: int = 2,
    ) -> None:
        required = (
            "config",
            "pager",
            "max_batch_size",
            "max_seq_len",
            "embed_batch",
            "finalize_hidden",
        )
        if any(not hasattr(model, name) for name in required):
            raise TypeError("model lacks the layerwise generation contract")
        self.model = model
        if runner is None:
            self.runner = LayerStateRunner(model)
        else:
            if getattr(runner, "model", None) is not model:
                raise ValueError("layer-state runner belongs to a different model")
            self.runner = runner
        self.layer_retries = _nonnegative_int(layer_retries, "layer_retries")

    def _forward_layer(
        self,
        hidden: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        layer: int,
        attention_state: LayerAttentionState | None = None,
        token_mask: torch.Tensor | None = None,
        phase: str,
        step: int | None,
        progress: Callable[[dict[str, Any]], None] | None,
    ) -> tuple[torch.Tensor, Any, LayerAttentionState, int]:
        retries = 0
        while True:
            try:
                output, experts, state = self.runner.forward_layer_stateful(
                    hidden,
                    token_ids,
                    layer=layer,
                    attention_state=attention_state,
                    token_mask=token_mask,
                )
                return output, experts, state, retries
            except Exception as exc:
                self.model.pager.release()
                if retries >= self.layer_retries:
                    raise
                retries += 1
                if progress is not None:
                    progress(
                        {
                            "event": "layer_retry",
                            "phase": phase,
                            "step": step,
                            "layer": layer,
                            "attempt": retries + 1,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )

    def _prompts(self, token_ids: Any) -> torch.Tensor:
        raw = token_ids
        if isinstance(token_ids, (str, bytes)):
            raise TypeError("prompt_token_ids must be a rectangular sequence")
        if isinstance(token_ids, Sequence) and not isinstance(token_ids, torch.Tensor):
            values = list(token_ids)
            if not values:
                raise ValueError("prompt batch must not be empty")
            nested = [
                isinstance(value, Sequence) and not isinstance(value, (str, bytes))
                for value in values
            ]
            if any(nested):
                if not all(nested):
                    raise TypeError("prompt rows must be integer sequences")
                rows = [tuple(row) for row in values]
                if any(not row for row in rows):
                    raise ValueError("prompt rows must not be empty")
                if len({len(row) for row in rows}) != 1:
                    raise ValueError(
                        "layer-major generation requires exact-length prompts"
                    )
                raw = rows
        ids = self.model._token_tensor(raw)
        if ids.ndim != 2 or ids.shape[0] < 1 or ids.shape[1] < 1:
            raise ValueError("prompt_token_ids must have [batch, sequence] shape")
        return ids

    def _graft_contract(self) -> tuple[Any | None, int | None, str]:
        graft = getattr(self.model, "graft", None)
        layer = getattr(self.model, "graft_layer", None)
        if graft is None:
            return None, None, "off"
        mode = str(getattr(graft, "mode", "unknown"))
        alpha = float(getattr(graft, "alpha", 0.0))
        if mode == "off" or alpha == 0.0:
            return None, layer, mode
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise ValueError("active graft requires an integer graft_layer")
        if not 0 <= layer < self.model.config.n_layers:
            raise ValueError("graft_layer outside decoder depth")
        if not callable(getattr(graft, "forward", None)):
            raise TypeError("active graft must provide forward()")
        return graft, layer, mode

    def _apply_graft(
        self,
        hidden: torch.Tensor,
        *,
        graft: Any,
        history: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = hidden
        if history is None:
            combined = raw
        else:
            combined = torch.cat(
                (
                    history.to(
                        device=raw.device,
                        dtype=raw.dtype,
                    ),
                    raw,
                ),
                dim=1,
            )
        result = graft.forward(combined)
        grafted = result[0] if isinstance(result, tuple) else result
        if tuple(grafted.shape) != tuple(combined.shape):
            raise ValueError("graft changed the hidden-history shape")
        owned_history = combined.detach().to(device="cpu").clone()
        return grafted[:, -raw.shape[1] :], owned_history

    def generate_greedy(
        self,
        prompt_token_ids: Any,
        *,
        max_new_tokens: int = 96,
        eos_token_ids: Sequence[int] = (),
        filler_token_id: int = 0,
        head_block_rows: int = 1024,
        progress: Callable[[dict[str, Any]], None] | None = None,
        head_progress: Callable[[dict[str, int]], None] | None = None,
    ) -> tuple[tuple[tuple[int, ...], ...], LayerwiseGenerationEvidence]:
        """Generate a fixed exact-length batch with exact full-head greedy scans."""

        maximum = _positive_int(max_new_tokens, "max_new_tokens")
        block_rows = _positive_int(head_block_rows, "head_block_rows")
        ids = self._prompts(prompt_token_ids)
        batch, prompt_length = (int(value) for value in ids.shape)
        if batch > self.model.max_batch_size:
            raise ValueError("prompt batch exceeds model max_batch_size")
        if prompt_length + maximum > self.model.max_seq_len:
            raise ValueError("prompt plus generation exceeds model context bound")
        vocab_size = int(self.model.config.vocab_size)
        if (
            isinstance(filler_token_id, bool)
            or not isinstance(filler_token_id, int)
            or not 0 <= filler_token_id < vocab_size
        ):
            raise ValueError("filler_token_id outside checkpoint vocabulary")
        eos = {int(value) for value in eos_token_ids}
        if any(value < 0 or value >= vocab_size for value in eos):
            raise ValueError("EOS token outside checkpoint vocabulary")
        graft, graft_layer, graft_mode = self._graft_contract()

        reset_state = getattr(self.model, "reset_state", None)
        if callable(reset_state):
            reset_state(release=True)
        source = self.model.pager.source
        start_bytes = _metric_int(source, "network_or_source_body_bytes")
        start_linears = _metric_int(self.model.pager, "linear_calls")
        started = time.perf_counter()
        handles: list[LayerAttentionState] = []
        generated: list[list[int]] = [[] for _ in range(batch)]
        finished = torch.zeros(batch, dtype=torch.bool)
        stopped_on_eos = [False] * batch
        graft_history: torch.Tensor | None = None
        forward_calls = 0
        retry_count = 0
        head_scans = 0
        prompt_cpu = tuple(
            tuple(int(value) for value in row)
            for row in ids.detach().to(device="cpu").tolist()
        )

        try:
            hidden = self.model.embed_batch(ids)
            for layer in range(self.model.config.n_layers):
                hidden, _experts, handle, retries = self._forward_layer(
                    hidden,
                    ids,
                    layer=layer,
                    phase="prefill",
                    step=None,
                    progress=progress,
                )
                handles.append(handle)
                forward_calls += 1
                retry_count += retries
                if graft is not None and layer == graft_layer:
                    hidden, graft_history = self._apply_graft(
                        hidden, graft=graft, history=None
                    )
                self.model.pager.release()
                if progress is not None:
                    progress(
                        {
                            "event": "layer_complete",
                            "phase": "prefill",
                            "layer": layer,
                            "cursor": prompt_length,
                        }
                    )

            for step in range(maximum):
                final = self.model.finalize_hidden(hidden)
                active_rows = (~finished).nonzero(as_tuple=False).flatten()
                if active_rows.numel() == 0:
                    break
                values, selected = self.model.pager.topk_logits(
                    final[:, -1].index_select(0, active_rows.to(final.device)),
                    k=1,
                    block_rows=block_rows,
                    progress=head_progress,
                )
                head_scans += 1
                selected_cpu = selected[:, 0].detach().to(device="cpu").tolist()
                values_cpu = values[:, 0].detach().to(device="cpu", dtype=torch.float32)
                for offset, row_tensor in enumerate(active_rows):
                    row = int(row_tensor.item())
                    token = int(selected_cpu[offset])
                    generated[row].append(token)
                    if token in eos:
                        finished[row] = True
                        stopped_on_eos[row] = True
                if progress is not None:
                    progress(
                        {
                            "event": "generated_tokens",
                            "step": step,
                            "rows": [int(value) for value in active_rows.tolist()],
                            "token_ids": [int(value) for value in selected_cpu],
                            "logits": [float(value) for value in values_cpu.tolist()],
                        }
                    )
                if bool(finished.all().item()) or step + 1 == maximum:
                    break

                input_ids = torch.full(
                    (batch, 1), filler_token_id, dtype=torch.long
                )
                active_mask = ~finished
                for row in range(batch):
                    if bool(active_mask[row].item()):
                        input_ids[row, 0] = generated[row][-1]
                hidden = self.model.embed_batch(input_ids)
                decode_mask = active_mask.reshape(batch, 1)
                next_handles: list[LayerAttentionState] = []
                for layer, handle in enumerate(handles):
                    hidden, _experts, updated, retries = self._forward_layer(
                        hidden,
                        input_ids,
                        layer=layer,
                        attention_state=handle,
                        token_mask=decode_mask,
                        phase="decode",
                        step=step + 1,
                        progress=progress,
                    )
                    next_handles.append(updated)
                    forward_calls += 1
                    retry_count += retries
                    if graft is not None and layer == graft_layer:
                        hidden, graft_history = self._apply_graft(
                            hidden,
                            graft=graft,
                            history=graft_history,
                        )
                    self.model.pager.release()
                    if progress is not None:
                        progress(
                            {
                                "event": "layer_complete",
                                "phase": "decode",
                                "step": step + 1,
                                "layer": layer,
                                "cursor": prompt_length + step + 1,
                            }
                        )
                handles = next_handles

            end_bytes = _metric_int(source, "network_or_source_body_bytes")
            end_linears = _metric_int(self.model.pager, "linear_calls")
            evidence = LayerwiseGenerationEvidence(
                schema=LAYERWISE_GENERATION_SCHEMA,
                prompt_token_ids=prompt_cpu,
                generated_token_ids=tuple(tuple(row) for row in generated),
                batch_size=batch,
                prompt_tokens=prompt_length,
                max_new_tokens=maximum,
                context_mode="layer_major_exact_length_autoregressive",
                stateful_kv_cache=True,
                exact_length_batch=True,
                forward_layer_calls=forward_calls,
                layer_retry_count=retry_count,
                head_scans=head_scans,
                source_body_bytes=end_bytes - start_bytes,
                linear_calls=end_linears - start_linears,
                seconds=time.perf_counter() - started,
                attention_state_bytes=_state_nbytes(handles),
                stopped_on_eos=tuple(stopped_on_eos),
                graft_mode=graft_mode,
                graft_layer=graft_layer,
                graft_history_tokens=(
                    0 if graft_history is None else int(graft_history.shape[1])
                ),
            )
            return evidence.generated_token_ids, evidence
        finally:
            self.model.pager.release()
            if callable(reset_state):
                reset_state(release=True)


__all__ = [
    "LAYERWISE_GENERATION_SCHEMA",
    "LayerwiseGenerationEvidence",
    "LayerwiseGenerator",
]
