"""Exact text-only, layer-paged Qwen3.8 decoder.

The model keeps activations resident while :class:`Qwen38WeightPager` streams
one BF16 checkpoint matrix at a time.  It implements the public Qwen3.5 text
equation directly; Transformers is neither imported nor used as a runtime.
Vision and the optional MTP block are deliberately outside this correctness
path.  MTP can later draft tokens, but it must never alter base-model parity.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import time
from typing import Any

import torch

from .config import Qwen38Config
from .kernels import (
    AttentionState,
    DeltaNetState,
    full_attention_core,
    gated_delta_net_core,
    rms_norm,
    swiglu,
)
from .pager import Qwen38WeightPager


class Qwen38RuntimeError(RuntimeError):
    """The streamed checkpoint violates the executable text-model contract."""


@dataclass(frozen=True, slots=True)
class PrefillEvidence:
    """Small execution receipt for one complete layer-major prefill."""

    batch_size: int
    sequence_length: int
    layers_executed: int
    source_body_bytes: int
    linear_calls: int
    graft_mode: str
    graft_layer: int | None
    graft_applied: bool


@dataclass(frozen=True, slots=True)
class StatefulEvidence:
    """Receipt for one committed stateful prefill or decode forward."""

    start_pos: int
    end_pos: int
    input_token_ids: tuple[tuple[int, ...], ...]
    layers_executed: int
    checkpoint_layers: int
    complete_layer_stack: bool
    context_mode: str
    stateful_cache: bool
    source_body_bytes: int
    linear_calls: int
    seconds: float
    state_bytes: int
    graft_mode: str
    graft_history_tokens: int


@dataclass(frozen=True, slots=True)
class GenerationEvidence:
    """Receipt for exact greedy generation through the streamed LM head."""

    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    context_mode: str
    stateful_cache: bool
    general_generation: bool
    prefill_mode: str
    forward_passes: int
    source_body_bytes: int
    linear_calls: int
    seconds: float
    state_bytes: int
    stopped_on_eos: bool


LayerState = AttentionState | DeltaNetState


class StreamedQwen38:
    """Qwen3.8-27B text decoder with bounded, sequential weight residency."""

    EMBED_NAME = "model.language_model.embed_tokens.weight"
    FINAL_NORM_NAME = "model.language_model.norm.weight"
    HEAD_NAME = "lm_head.weight"

    def __init__(
        self,
        config: Qwen38Config,
        pager: Qwen38WeightPager,
        *,
        graft: Any | None = None,
        graft_layer: int | None = None,
        max_batch_size: int = 8,
        max_seq_len: int = 4096,
    ) -> None:
        if not isinstance(config, Qwen38Config):
            raise TypeError("config must be Qwen38Config")
        if not isinstance(pager, Qwen38WeightPager):
            raise TypeError("pager must be Qwen38WeightPager")
        if (
            isinstance(max_batch_size, bool)
            or not isinstance(max_batch_size, int)
            or max_batch_size <= 0
        ):
            raise ValueError("max_batch_size must be a positive integer")
        if (
            isinstance(max_seq_len, bool)
            or not isinstance(max_seq_len, int)
            or max_seq_len <= 0
        ):
            raise ValueError("max_seq_len must be a positive integer")
        if max_seq_len > config.max_position_embeddings:
            raise ValueError("max_seq_len exceeds the checkpoint context bound")
        if graft_layer is not None and (
            isinstance(graft_layer, bool)
            or not isinstance(graft_layer, int)
            or not 0 <= graft_layer < config.n_layers
        ):
            raise ValueError("graft_layer outside decoder depth")
        if graft is not None and graft_layer is None:
            raise ValueError("an active graft requires graft_layer")

        self.config = config
        self.pager = pager
        self.graft = graft
        self.graft_layer = graft_layer
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self._layer_states: list[LayerState | None] = [
            None for _ in range(config.n_layers)
        ]
        self._next_position = 0
        self._state_batch_size: int | None = None
        self._state_poisoned = False
        self._graft_history: torch.Tensor | None = None

    @property
    def next_position(self) -> int:
        return self._next_position

    @property
    def state_batch_size(self) -> int | None:
        return self._state_batch_size

    @property
    def state_poisoned(self) -> bool:
        return self._state_poisoned

    @property
    def state_bytes(self) -> int:
        total = 0
        for state in self._layer_states:
            if isinstance(state, AttentionState):
                tensors = (state.key, state.value)
            elif isinstance(state, DeltaNetState):
                tensors = (state.conv, state.recurrent)
            else:
                continue
            total += sum(tensor.numel() * tensor.element_size() for tensor in tensors)
        if self._graft_history is not None:
            total += self._graft_history.numel() * self._graft_history.element_size()
        return total

    def prefill_hidden_shape(self, batch: int, sequence: int) -> tuple[int, int, int]:
        """Return the rolling-resume activation shape for this decoder."""

        if min(batch, sequence) <= 0:
            raise ValueError("batch and sequence must be positive")
        return batch, sequence, self.config.dim

    @staticmethod
    def _metric(owner: Any, name: str) -> int:
        metrics = owner.metrics()
        value = metrics.get(name, 0)
        return (
            int(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else 0
        )

    def _token_tensor(self, token_ids: Any) -> torch.Tensor:
        if isinstance(token_ids, torch.Tensor):
            ids = token_ids
        else:
            ids = torch.as_tensor(token_ids)
        if ids.dtype == torch.bool or ids.is_floating_point() or ids.is_complex():
            raise TypeError("token_ids must contain integers")
        if ids.ndim == 0:
            ids = ids.reshape(1, 1)
        elif ids.ndim == 1:
            ids = ids.unsqueeze(0)
        elif ids.ndim != 2:
            raise ValueError("token_ids must have [batch, sequence] shape")
        if ids.shape[0] < 1 or ids.shape[1] < 1:
            raise ValueError("token_ids dimensions must be non-empty")
        if ids.shape[0] > self.max_batch_size:
            raise ValueError("token batch exceeds max_batch_size")
        if ids.shape[1] > self.max_seq_len:
            raise ValueError("token sequence exceeds max_seq_len")
        ids = ids.to(device=self.pager.device, dtype=torch.long)
        if bool(((ids < 0) | (ids >= self.config.vocab_size)).any().item()):
            raise ValueError("token ID outside checkpoint vocabulary")
        return ids

    def _prefix_mask(self, token_mask: Any | None, ids: torch.Tensor) -> torch.Tensor:
        if token_mask is None:
            return torch.ones_like(ids, dtype=torch.bool, device=self.pager.device)
        mask = torch.as_tensor(token_mask, device=self.pager.device)
        if mask.dtype != torch.bool:
            raise TypeError("token_mask must be boolean")
        if tuple(mask.shape) != tuple(ids.shape):
            raise ValueError("token_mask must match token_ids shape")
        if bool((~mask[:, 0]).any().item()):
            raise ValueError("every row must have a non-empty prefix")
        if ids.shape[1] > 1 and bool((mask[:, 1:] & ~mask[:, :-1]).any().item()):
            raise ValueError("token_mask must describe right-padded prefixes")
        return mask

    def _control(
        self, name: str, *, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        return self.pager.tensor_torch(name, dtype=dtype, device=self.pager.device)

    def _norm(self, hidden: torch.Tensor, name: str) -> torch.Tensor:
        weight = self._control(name)
        try:
            return rms_norm(hidden, weight, eps=self.config.rms_norm_eps)
        finally:
            del weight

    def embed_batch(self, token_ids: Any) -> torch.Tensor:
        ids = self._token_tensor(token_ids)
        flat = ids.detach().to(device="cpu").reshape(-1).tolist()
        hidden = self.pager.embedding(flat, name=self.EMBED_NAME)
        return hidden.reshape(ids.shape[0], ids.shape[1], self.config.dim)

    def _full_attention(
        self,
        hidden: torch.Tensor,
        *,
        layer: int,
        token_mask: torch.Tensor | None,
        state: AttentionState | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, AttentionState]:
        base = f"model.language_model.layers.{layer}.self_attn"
        projected_query_gate = self.pager.linear(hidden, f"{base}.q_proj")
        projected_key = self.pager.linear(hidden, f"{base}.k_proj")
        projected_value = self.pager.linear(hidden, f"{base}.v_proj")
        q_norm_weight = self._control(f"{base}.q_norm.weight")
        k_norm_weight = self._control(f"{base}.k_norm.weight")
        positions = (
            torch.arange(
                start_pos,
                start_pos + hidden.shape[1],
                device=hidden.device,
                dtype=torch.long,
            )
            .unsqueeze(0)
            .expand(hidden.shape[0], -1)
        )
        try:
            mixed, next_state = full_attention_core(
                projected_query_gate,
                projected_key,
                projected_value,
                q_norm_weight=q_norm_weight,
                k_norm_weight=k_norm_weight,
                num_attention_heads=self.config.n_heads,
                num_key_value_heads=self.config.n_kv_heads,
                head_dim=self.config.head_dim,
                position_ids=positions,
                state=state,
                attention_mask=token_mask,
                rope_theta=self.config.rope_theta,
                rotary_dim=self.config.rotary_dim,
                mrope_section=self.config.mrope_section,
                mrope_interleaved=self.config.mrope_interleaved,
                rms_norm_eps=self.config.rms_norm_eps,
            )
        finally:
            del projected_query_gate, projected_key, projected_value
            del q_norm_weight, k_norm_weight
        return self.pager.linear(mixed, f"{base}.o_proj"), next_state

    def _linear_attention(
        self,
        hidden: torch.Tensor,
        *,
        layer: int,
        token_mask: torch.Tensor,
        state: DeltaNetState | None = None,
    ) -> tuple[torch.Tensor, DeltaNetState]:
        base = f"model.language_model.layers.{layer}.linear_attn"
        # Official Qwen masks padding before every Gated DeltaNet projection.
        active = hidden * token_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        projected_qkv = self.pager.linear(active, f"{base}.in_proj_qkv")
        projected_z = self.pager.linear(active, f"{base}.in_proj_z")
        projected_b = self.pager.linear(active, f"{base}.in_proj_b")
        projected_a = self.pager.linear(active, f"{base}.in_proj_a")
        conv_weight = self._control(f"{base}.conv1d.weight", dtype=hidden.dtype)
        a_log = self._control(f"{base}.A_log")
        dt_bias = self._control(f"{base}.dt_bias")
        norm_weight = self._control(f"{base}.norm.weight", dtype=hidden.dtype)
        try:
            mixed, next_state = gated_delta_net_core(
                projected_qkv,
                projected_z,
                projected_b,
                projected_a,
                conv1d_weight=conv_weight,
                A_log=a_log,
                dt_bias=dt_bias,
                norm_weight=norm_weight,
                num_key_heads=self.config.linear_num_key_heads,
                num_value_heads=self.config.linear_num_value_heads,
                key_head_dim=self.config.linear_key_head_dim,
                value_head_dim=self.config.linear_value_head_dim,
                state=state,
                rms_norm_eps=self.config.rms_norm_eps,
            )
        finally:
            del projected_qkv, projected_z, projected_b, projected_a
            del conv_weight, a_log, dt_bias, norm_weight
        return self.pager.linear(mixed, f"{base}.out_proj"), next_state

    def _mlp(self, hidden: torch.Tensor, *, layer: int) -> torch.Tensor:
        base = f"model.language_model.layers.{layer}.mlp"
        gate = self.pager.linear(hidden, f"{base}.gate_proj")
        up = self.pager.linear(hidden, f"{base}.up_proj")
        try:
            activated = swiglu(gate, up)
        finally:
            del gate, up
        return self.pager.linear(activated, f"{base}.down_proj")

    def _forward_layer(
        self,
        hidden: torch.Tensor,
        *,
        layer: int,
        token_mask: torch.Tensor,
        state: LayerState | None,
        start_pos: int,
        stateful: bool,
    ) -> tuple[torch.Tensor, LayerState | None]:
        """Apply one block and return its staged continuation state."""

        prefix = f"model.language_model.layers.{layer}"
        residual = hidden
        mixed_input = self._norm(hidden, f"{prefix}.input_layernorm.weight")
        if self.config.is_full_attention(layer):
            if state is not None and not isinstance(state, AttentionState):
                raise Qwen38RuntimeError("full-attention layer received DeltaNet state")
            mixed, next_state = self._full_attention(
                mixed_input,
                layer=layer,
                token_mask=None if stateful else token_mask,
                state=state,
                start_pos=start_pos,
            )
        else:
            if state is not None and not isinstance(state, DeltaNetState):
                raise Qwen38RuntimeError("linear-attention layer received KV state")
            mixed, next_state = self._linear_attention(
                mixed_input,
                layer=layer,
                token_mask=token_mask,
                state=state,
            )
        hidden = residual + mixed
        retained_state: LayerState | None = next_state
        if not stateful:
            retained_state = None
            del next_state

        residual = hidden
        mlp_input = self._norm(hidden, f"{prefix}.post_attention_layernorm.weight")
        hidden = residual + self._mlp(mlp_input, layer=layer)
        return hidden, retained_state

    def forward_prefill_layer(
        self,
        hidden: Any,
        token_ids: Any,
        *,
        layer: int,
        token_mask: Any | None = None,
    ) -> tuple[torch.Tensor, None]:
        """Apply one independent Qwen block, preserving exact prefix outputs."""

        ids = self._token_tensor(token_ids)
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise TypeError("layer must be an integer")
        if not 0 <= layer < self.config.n_layers:
            raise ValueError("layer outside decoder depth")
        expected = self.prefill_hidden_shape(ids.shape[0], ids.shape[1])
        if not isinstance(hidden, torch.Tensor) or tuple(hidden.shape) != expected:
            raise ValueError(f"hidden shape must be {expected}")
        if not hidden.is_floating_point():
            raise TypeError("hidden must be floating point")
        mask = self._prefix_mask(token_mask, ids)
        x = hidden.to(device=self.pager.device, dtype=self.pager.compute_dtype)

        x, _state = self._forward_layer(
            x,
            layer=layer,
            token_mask=mask,
            state=None,
            start_pos=0,
            stateful=False,
        )
        return x, None

    def finalize_hidden(self, hidden: Any) -> torch.Tensor:
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise ValueError("hidden must have [batch, sequence, dim] shape")
        if hidden.shape[-1] != self.config.dim:
            raise ValueError("hidden width does not match the checkpoint")
        hidden = hidden.to(device=self.pager.device, dtype=self.pager.compute_dtype)
        return self._norm(hidden, self.FINAL_NORM_NAME)

    def _graft_mode(self) -> str:
        return (
            "off" if self.graft is None else str(getattr(self.graft, "mode", "active"))
        )

    def _apply_graft_stateful(
        self,
        hidden: torch.Tensor,
        history: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.graft is None:
            return hidden, None
        mode = self._graft_mode()
        alpha = float(getattr(self.graft, "alpha", 1.0))
        active = mode != "off" and alpha != 0.0
        if active:
            if self._next_position and history is None:
                raise Qwen38RuntimeError("graft history is missing for decode")
            if history is not None and history.shape[0] != hidden.shape[0]:
                raise Qwen38RuntimeError("graft batch changed within a request")
            complete = (
                hidden if history is None else torch.cat((history, hidden), dim=1)
            )
        else:
            complete = hidden
        grafted = self.graft.forward(complete)
        output = grafted[0] if isinstance(grafted, tuple) else grafted
        if not isinstance(output, torch.Tensor) or tuple(output.shape) != tuple(
            complete.shape
        ):
            raise Qwen38RuntimeError("graft changed the hidden-state contract")
        next_history = complete.detach().clone() if active else None
        return output[:, -hidden.shape[1] :], next_history

    def _poison_state(self) -> None:
        self._layer_states = [None for _ in range(self.config.n_layers)]
        self._next_position = 0
        self._state_batch_size = None
        self._graft_history = None
        self._state_poisoned = True

    def reset_state(self, *, release: bool = False) -> None:
        """Drop every committed KV/DeltaNet cache and clear the poison latch."""

        self._layer_states = [None for _ in range(self.config.n_layers)]
        self._next_position = 0
        self._state_batch_size = None
        self._state_poisoned = False
        self._graft_history = None
        if release:
            self.pager.release()

    def hidden_stateful(
        self,
        token_ids: Any,
        *,
        start_pos: int | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[torch.Tensor, StatefulEvidence]:
        """Execute one complete prefill or contiguous one-token decode.

        State commits only after all 64 layers and the final norm succeed. A
        failed call clears partial state and latches the runtime until the
        caller acknowledges the failure with :meth:`reset_state`.
        """

        ids = self._token_tensor(token_ids)
        if self._state_poisoned:
            raise Qwen38RuntimeError("decoder state is poisoned; call reset_state()")
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable or None")
        if start_pos is None:
            start_pos = self._next_position
        if (
            isinstance(start_pos, bool)
            or not isinstance(start_pos, int)
            or start_pos < 0
        ):
            raise ValueError("start_pos must be a non-negative integer")
        if start_pos != self._next_position:
            raise ValueError(
                f"non-contiguous model forward: expected {self._next_position}, got {start_pos}"
            )
        if start_pos and ids.shape[1] != 1:
            raise ValueError("stateful decode accepts exactly one token")
        end_pos = start_pos + ids.shape[1]
        if end_pos > self.max_seq_len:
            raise ValueError(
                f"context end {end_pos} exceeds max_seq_len={self.max_seq_len}"
            )
        if start_pos == 0:
            if self._state_batch_size is not None or any(
                state is not None for state in self._layer_states
            ):
                raise Qwen38RuntimeError("position-zero state is not empty")
        else:
            if self._state_batch_size != ids.shape[0]:
                raise ValueError("decode batch size differs from the committed prefix")
            if any(state is None for state in self._layer_states):
                raise Qwen38RuntimeError("one or more layer states are missing")
            for layer, state in enumerate(self._layer_states):
                if self.config.is_full_attention(layer):
                    if (
                        not isinstance(state, AttentionState)
                        or state.length != start_pos
                    ):
                        raise Qwen38RuntimeError(
                            f"full-attention layer {layer} cursor disagrees with model state"
                        )
                elif not isinstance(state, DeltaNetState):
                    raise Qwen38RuntimeError(
                        f"linear-attention layer {layer} has the wrong state type"
                    )

        source = self.pager.source
        start_bytes = self._metric(source, "network_or_source_body_bytes")
        start_linears = self._metric(self.pager, "linear_calls")
        started = time.perf_counter()
        hidden = self.embed_batch(ids)
        mask = torch.ones_like(ids, dtype=torch.bool, device=self.pager.device)
        staged: list[LayerState] = []
        staged_history = self._graft_history
        try:
            for layer in range(self.config.n_layers):
                layer_started = time.perf_counter()
                layer_bytes = self._metric(source, "network_or_source_body_bytes")
                hidden, next_state = self._forward_layer(
                    hidden,
                    layer=layer,
                    token_mask=mask,
                    state=self._layer_states[layer],
                    start_pos=start_pos,
                    stateful=True,
                )
                if next_state is None:  # pragma: no cover - stateful contract above.
                    raise Qwen38RuntimeError("stateful layer returned no continuation")
                staged.append(next_state)
                if self.graft is not None and layer == self.graft_layer:
                    hidden, staged_history = self._apply_graft_stateful(
                        hidden, staged_history
                    )
                self.pager.release()
                if progress is not None:
                    progress(
                        {
                            "event": "qwen_stateful_layer_complete",
                            "layer": layer,
                            "layers": self.config.n_layers,
                            "start_pos": start_pos,
                            "tokens": ids.shape[1],
                            "source_body_bytes": self._metric(
                                source, "network_or_source_body_bytes"
                            )
                            - layer_bytes,
                            "seconds": time.perf_counter() - layer_started,
                            "state_kind": (
                                "kv"
                                if isinstance(next_state, AttentionState)
                                else "deltanet"
                            ),
                        }
                    )
            hidden = self.finalize_hidden(hidden)
        except Exception:
            self._poison_state()
            self.pager.release()
            raise
        finally:
            self.pager.release()

        self._layer_states = staged
        self._next_position = end_pos
        self._state_batch_size = ids.shape[0]
        self._graft_history = staged_history
        evidence = StatefulEvidence(
            start_pos=start_pos,
            end_pos=end_pos,
            input_token_ids=tuple(
                tuple(int(value) for value in row)
                for row in ids.detach().to("cpu").tolist()
            ),
            layers_executed=self.config.n_layers,
            checkpoint_layers=self.config.n_layers,
            complete_layer_stack=True,
            context_mode="prefill" if start_pos == 0 else "decode",
            stateful_cache=True,
            source_body_bytes=(
                self._metric(source, "network_or_source_body_bytes") - start_bytes
            ),
            linear_calls=self._metric(self.pager, "linear_calls") - start_linears,
            seconds=time.perf_counter() - started,
            state_bytes=self.state_bytes,
            graft_mode=self._graft_mode(),
            graft_history_tokens=(
                0 if self._graft_history is None else int(self._graft_history.shape[1])
            ),
        )
        return hidden, evidence

    def prefill(
        self,
        token_ids: Any,
        *,
        tokenwise: bool = False,
        reset: bool = True,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[torch.Tensor, tuple[StatefulEvidence, ...]]:
        """Commit an equal-length prompt as one chunk or exact token steps."""

        if not isinstance(tokenwise, bool):
            raise TypeError("tokenwise must be a boolean")
        ids = self._token_tensor(token_ids)
        if reset:
            self.reset_state()
        elif self._next_position != 0:
            raise ValueError("prefill requires position zero or reset=True")
        if not tokenwise:
            hidden, evidence = self.hidden_stateful(ids, start_pos=0, progress=progress)
            return hidden, (evidence,)
        outputs: list[torch.Tensor] = []
        evidence_rows: list[StatefulEvidence] = []
        for position in range(ids.shape[1]):
            hidden, evidence = self.hidden_stateful(
                ids[:, position : position + 1],
                start_pos=position,
                progress=progress,
            )
            outputs.append(hidden)
            evidence_rows.append(evidence)
        return torch.cat(outputs, dim=1), tuple(evidence_rows)

    def decode(
        self,
        token_ids: Any,
        *,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[torch.Tensor, StatefulEvidence]:
        ids = self._token_tensor(token_ids)
        if ids.shape[1] != 1:
            raise ValueError("decode accepts exactly one token per batch row")
        if self._next_position == 0:
            raise ValueError("decode requires a completed prefill")
        return self.hidden_stateful(
            ids, start_pos=self._next_position, progress=progress
        )

    def generate_greedy(
        self,
        prompt_token_ids: Any,
        *,
        max_new_tokens: int = 1,
        prefill_tokenwise: bool = False,
        eos_token_ids: Iterable[int] = (),
        head_block_rows: int = Qwen38WeightPager.DEFAULT_HEAD_BLOCK_ROWS,
        progress: Callable[[dict[str, Any]], None] | None = None,
        head_progress: Callable[[dict[str, int]], None] | None = None,
    ) -> tuple[tuple[int, ...], GenerationEvidence]:
        """Run exact greedy autoregressive generation through the streamed head."""

        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        if not isinstance(prefill_tokenwise, bool):
            raise TypeError("prefill_tokenwise must be a boolean")
        prompt = self._token_tensor(prompt_token_ids)
        if prompt.shape[0] != 1:
            raise ValueError("greedy generation requires batch size one")
        if prompt.shape[1] + max_new_tokens > self.max_seq_len:
            raise ValueError("generation would exceed max_seq_len")
        if isinstance(eos_token_ids, (str, bytes)):
            raise TypeError("eos_token_ids must be an iterable of integers")
        eos: set[int] = set()
        try:
            for raw in eos_token_ids:
                if isinstance(raw, bool) or not isinstance(raw, int):
                    raise TypeError("eos_token_ids must contain integers")
                eos.add(raw)
        except TypeError as exc:
            if str(exc) == "eos_token_ids must contain integers":
                raise
            raise TypeError("eos_token_ids must be iterable") from exc
        if any(value < 0 or value >= self.config.vocab_size for value in eos):
            raise ValueError("EOS token outside checkpoint vocabulary")
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable or None")
        if head_progress is not None and not callable(head_progress):
            raise TypeError("head_progress must be callable or None")

        source = self.pager.source
        start_bytes = self._metric(source, "network_or_source_body_bytes")
        start_linears = self._metric(self.pager, "linear_calls")
        started = time.perf_counter()
        hidden, forwards = self.prefill(
            prompt,
            tokenwise=prefill_tokenwise,
            reset=True,
            progress=progress,
        )
        generated: list[int] = []
        stopped_on_eos = False
        forward_count = len(forwards)
        for step in range(max_new_tokens):
            values, token_ids = self.pager.topk_logits(
                hidden[:, -1],
                k=1,
                block_rows=head_block_rows,
                progress=head_progress,
            )
            token_id = int(token_ids[0, 0].item())
            generated.append(token_id)
            if progress is not None:
                progress(
                    {
                        "event": "generated_token",
                        "step": step,
                        "token_id": token_id,
                        "logit": float(values[0, 0].item()),
                    }
                )
            # Commit every emitted token before returning.  This keeps the
            # persistent state exactly aligned with the returned sequence, so
            # callers can continue decoding without replaying the final token.
            hidden, _evidence = self.decode([[token_id]], progress=progress)
            forward_count += 1
            if token_id in eos:
                stopped_on_eos = True
                break

        prompt_ids = tuple(
            int(value) for value in prompt[0].detach().to("cpu").tolist()
        )
        evidence = GenerationEvidence(
            prompt_token_ids=prompt_ids,
            generated_token_ids=tuple(generated),
            context_mode="stateful_autoregressive",
            stateful_cache=True,
            general_generation=True,
            prefill_mode="tokenwise" if prefill_tokenwise else "batched",
            forward_passes=forward_count,
            source_body_bytes=(
                self._metric(source, "network_or_source_body_bytes") - start_bytes
            ),
            linear_calls=self._metric(self.pager, "linear_calls") - start_linears,
            seconds=time.perf_counter() - started,
            state_bytes=self.state_bytes,
            stopped_on_eos=stopped_on_eos,
        )
        return tuple(generated), evidence

    def forward_prefill(
        self,
        token_ids: Any,
        *,
        token_mask: Any | None = None,
    ) -> tuple[torch.Tensor, PrefillEvidence]:
        """Execute the complete text stack once, releasing weights per layer."""

        ids = self._token_tensor(token_ids)
        mask = self._prefix_mask(token_mask, ids)
        source = self.pager.source
        start_bytes = self._metric(source, "network_or_source_body_bytes")
        start_linears = self._metric(self.pager, "linear_calls")
        hidden = self.embed_batch(ids)
        graft_applied = False
        mode = (
            "off" if self.graft is None else str(getattr(self.graft, "mode", "active"))
        )
        try:
            for layer in range(self.config.n_layers):
                hidden, _ = self.forward_prefill_layer(
                    hidden, ids, layer=layer, token_mask=mask
                )
                if self.graft is not None and layer == self.graft_layer:
                    grafted = self.graft.forward(hidden)
                    hidden = grafted[0] if isinstance(grafted, tuple) else grafted
                    if not isinstance(hidden, torch.Tensor) or tuple(
                        hidden.shape
                    ) != self.prefill_hidden_shape(*ids.shape):
                        raise Qwen38RuntimeError(
                            "graft changed the hidden-state contract"
                        )
                    graft_applied = True
                self.pager.release()
            final = self.finalize_hidden(hidden)
        finally:
            self.pager.release()
        evidence = PrefillEvidence(
            batch_size=ids.shape[0],
            sequence_length=ids.shape[1],
            layers_executed=self.config.n_layers,
            source_body_bytes=self._metric(source, "network_or_source_body_bytes")
            - start_bytes,
            linear_calls=self._metric(self.pager, "linear_calls") - start_linears,
            graft_mode=mode,
            graft_layer=self.graft_layer,
            graft_applied=graft_applied,
        )
        return final, evidence

    def checkpoint_preflight(self) -> dict[str, Any]:
        """Validate every text tensor's dtype and shape without payload reads."""

        entries = {
            str(entry["name"]): entry
            for entry in self.pager.source.inventory().get("tensors", ())
        }
        errors: list[str] = []
        required: dict[str, tuple[int, ...]] = {
            self.EMBED_NAME: (self.config.vocab_size, self.config.dim),
            self.FINAL_NORM_NAME: (self.config.dim,),
            self.HEAD_NAME: (self.config.vocab_size, self.config.dim),
        }
        for layer in range(self.config.n_layers):
            base = f"model.language_model.layers.{layer}"
            required[f"{base}.input_layernorm.weight"] = (self.config.dim,)
            required[f"{base}.post_attention_layernorm.weight"] = (self.config.dim,)
            required[f"{base}.mlp.gate_proj.weight"] = (
                self.config.intermediate_size,
                self.config.dim,
            )
            required[f"{base}.mlp.up_proj.weight"] = (
                self.config.intermediate_size,
                self.config.dim,
            )
            required[f"{base}.mlp.down_proj.weight"] = (
                self.config.dim,
                self.config.intermediate_size,
            )
            if self.config.is_full_attention(layer):
                attn = f"{base}.self_attn"
                required[f"{attn}.q_proj.weight"] = (
                    2 * self.config.n_heads * self.config.head_dim,
                    self.config.dim,
                )
                for role in ("k_proj", "v_proj"):
                    required[f"{attn}.{role}.weight"] = (
                        self.config.n_kv_heads * self.config.head_dim,
                        self.config.dim,
                    )
                required[f"{attn}.o_proj.weight"] = (
                    self.config.dim,
                    self.config.n_heads * self.config.head_dim,
                )
                required[f"{attn}.q_norm.weight"] = (self.config.head_dim,)
                required[f"{attn}.k_norm.weight"] = (self.config.head_dim,)
            else:
                attn = f"{base}.linear_attn"
                key_dim = (
                    self.config.linear_num_key_heads * self.config.linear_key_head_dim
                )
                value_dim = (
                    self.config.linear_num_value_heads
                    * self.config.linear_value_head_dim
                )
                conv_dim = 2 * key_dim + value_dim
                required[f"{attn}.in_proj_qkv.weight"] = (conv_dim, self.config.dim)
                required[f"{attn}.in_proj_z.weight"] = (value_dim, self.config.dim)
                for role in ("a", "b"):
                    required[f"{attn}.in_proj_{role}.weight"] = (
                        self.config.linear_num_value_heads,
                        self.config.dim,
                    )
                required[f"{attn}.conv1d.weight"] = (
                    conv_dim,
                    1,
                    self.config.linear_conv_kernel_dim,
                )
                required[f"{attn}.A_log"] = (self.config.linear_num_value_heads,)
                required[f"{attn}.dt_bias"] = (self.config.linear_num_value_heads,)
                required[f"{attn}.norm.weight"] = (self.config.linear_value_head_dim,)
                required[f"{attn}.out_proj.weight"] = (self.config.dim, value_dim)

        for name, shape in required.items():
            entry = entries.get(name)
            if entry is None:
                errors.append(f"missing {name}")
                continue
            actual_shape = tuple(int(value) for value in entry.get("shape", ()))
            if actual_shape != shape:
                errors.append(f"{name}: shape {actual_shape}, expected {shape}")
            if str(entry.get("dtype", "")).upper() != "BF16":
                errors.append(f"{name}: expected BF16")
        if errors:
            raise Qwen38RuntimeError(
                f"checkpoint violates {len(errors)} text tensor contracts: "
                + "; ".join(errors[:8])
            )
        payload_bytes = sum(
            int(entries[name]["offset_in_shard"][1])
            - int(entries[name]["offset_in_shard"][0])
            for name in required
        )
        return {
            "required_tensors": len(required),
            "required_payload_bytes": payload_bytes,
            "layers": self.config.n_layers,
            "linear_attention_layers": sum(
                not self.config.is_full_attention(layer)
                for layer in range(self.config.n_layers)
            ),
            "full_attention_layers": sum(
                self.config.is_full_attention(layer)
                for layer in range(self.config.n_layers)
            ),
            "vision_excluded": True,
            "mtp_excluded": True,
        }


__all__ = [
    "GenerationEvidence",
    "PrefillEvidence",
    "Qwen38RuntimeError",
    "StatefulEvidence",
    "StreamedQwen38",
]
