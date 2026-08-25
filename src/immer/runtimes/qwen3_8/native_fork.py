"""Transactional lockstep fork for off and native Head-CRSA Qwen3.8 arms.

The fork is deliberately functional: continuation state is passed in and a
new frozen state is returned only after every required arm succeeds.  No model
cursor, replay buffer, hidden cache, or disk copy participates in the commit.

While the two token histories are equal, layers 0..26 are executed once and
layer 27 shares its input norm, Q/K/V projections, RoPE, scores, and causal
softmax.  The off/native probability paths then use one checkpoint pass per
divergent projection through :meth:`Qwen38WeightPager.linear_many`.  Once the
input tokens differ the state transitions permanently to two complete model
states; equal tokens later never rejoin the arms.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
import time
from typing import Any, Literal

import torch

from .config import Qwen38Config
from .kernels import (
    AttentionState,
    DeltaNetState,
    full_attention_fork_core,
    rms_norm,
    swiglu,
)
from .model import LayerState, Qwen38RuntimeError, StreamedQwen38
from .native_crsa import (
    NATIVE_HEAD_CRSA_LAYER,
    NativeHeadCrsaEvidence,
    Qwen38NativeHeadCrsa,
)
from .pager import Qwen38WeightPager


FORK_LAYER = NATIVE_HEAD_CRSA_LAYER
COMMON_STOP_LAYER = FORK_LAYER
SUFFIX_START_LAYER = FORK_LAYER


def _nonnegative(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _positive(value: object, name: str) -> int:
    result = _nonnegative(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _first_token_divergence(
    off_tokens: tuple[int, ...], native_tokens: tuple[int, ...]
) -> int | None:
    """Return the first unequal index, including one history ending first."""

    common_length = min(len(off_tokens), len(native_tokens))
    for position in range(common_length):
        if off_tokens[position] != native_tokens[position]:
            return position
    if len(off_tokens) != len(native_tokens):
        return common_length
    return None


@dataclass(frozen=True, slots=True)
class Qwen38ForkLayerAccounting:
    """Architecture-level work accounting for one joined pair forward."""

    depth: int
    independent_complete_layers: int
    shared_complete_layers: int
    complete_layers_saved: int
    fork_layer: int
    independent_fork_weight_passes: int
    fork_weight_passes: int

    @classmethod
    def for_depth(cls, depth: int) -> "Qwen38ForkLayerAccounting":
        layers = _positive(depth, "depth")
        if layers <= FORK_LAYER:
            raise ValueError("depth must contain native Head-CRSA layer 27")
        return cls(
            depth=layers,
            independent_complete_layers=2 * layers,
            shared_complete_layers=COMMON_STOP_LAYER,
            complete_layers_saved=COMMON_STOP_LAYER,
            fork_layer=FORK_LAYER,
            independent_fork_weight_passes=2,
            fork_weight_passes=1,
        )

    @property
    def complete_layer_saving_fraction(self) -> float:
        return self.complete_layers_saved / self.independent_complete_layers


@dataclass(frozen=True, slots=True)
class Qwen38ForkTraffic:
    """Additive traffic and execution receipt for a pair operation."""

    shared_source_body_bytes: int = 0
    off_source_body_bytes: int = 0
    native_source_body_bytes: int = 0
    shared_linear_calls: int = 0
    off_linear_calls: int = 0
    native_linear_calls: int = 0
    shared_layers: int = 0
    off_layers: int = 0
    native_layers: int = 0
    independent_complete_layers: int = 0
    complete_layers_saved: int = 0
    fork_layer_weight_passes: int = 0
    native_head_crsa_evidence: tuple[NativeHeadCrsaEvidence, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "shared_source_body_bytes",
            "off_source_body_bytes",
            "native_source_body_bytes",
            "shared_linear_calls",
            "off_linear_calls",
            "native_linear_calls",
            "shared_layers",
            "off_layers",
            "native_layers",
            "independent_complete_layers",
            "complete_layers_saved",
            "fork_layer_weight_passes",
        ):
            _nonnegative(getattr(self, name), name)
        if self.complete_layers_saved > self.independent_complete_layers:
            raise ValueError("complete layer saving exceeds independent work")
        if not isinstance(self.native_head_crsa_evidence, tuple) or any(
            not isinstance(row, NativeHeadCrsaEvidence)
            for row in self.native_head_crsa_evidence
        ):
            raise TypeError("native_head_crsa_evidence must be an evidence tuple")

    @property
    def source_body_bytes(self) -> int:
        return (
            self.shared_source_body_bytes
            + self.off_source_body_bytes
            + self.native_source_body_bytes
        )

    @property
    def linear_calls(self) -> int:
        return (
            self.shared_linear_calls + self.off_linear_calls + self.native_linear_calls
        )

    @property
    def layers(self) -> int:
        return self.shared_layers + self.off_layers + self.native_layers

    def __add__(self, other: object) -> "Qwen38ForkTraffic":
        if not isinstance(other, Qwen38ForkTraffic):
            return NotImplemented
        return Qwen38ForkTraffic(
            shared_source_body_bytes=(
                self.shared_source_body_bytes + other.shared_source_body_bytes
            ),
            off_source_body_bytes=(
                self.off_source_body_bytes + other.off_source_body_bytes
            ),
            native_source_body_bytes=(
                self.native_source_body_bytes + other.native_source_body_bytes
            ),
            shared_linear_calls=self.shared_linear_calls + other.shared_linear_calls,
            off_linear_calls=self.off_linear_calls + other.off_linear_calls,
            native_linear_calls=(self.native_linear_calls + other.native_linear_calls),
            shared_layers=self.shared_layers + other.shared_layers,
            off_layers=self.off_layers + other.off_layers,
            native_layers=self.native_layers + other.native_layers,
            independent_complete_layers=(
                self.independent_complete_layers + other.independent_complete_layers
            ),
            complete_layers_saved=(
                self.complete_layers_saved + other.complete_layers_saved
            ),
            fork_layer_weight_passes=(
                self.fork_layer_weight_passes + other.fork_layer_weight_passes
            ),
            native_head_crsa_evidence=(
                self.native_head_crsa_evidence + other.native_head_crsa_evidence
            ),
        )


@dataclass(frozen=True, slots=True)
class Qwen38ForkArmState:
    """Immutable-by-contract continuation state for one output arm."""

    next_position: int
    token_ids: tuple[int, ...]
    lower_layer_states: tuple[LayerState, ...]
    suffix_layer_states: tuple[LayerState, ...]
    last_hidden: torch.Tensor

    def __post_init__(self) -> None:
        position = _positive(self.next_position, "next_position")
        if not isinstance(self.token_ids, tuple) or len(self.token_ids) != position:
            raise ValueError("token_ids must cover the complete arm cursor")
        if any(
            isinstance(token, bool) or not isinstance(token, int)
            for token in self.token_ids
        ):
            raise TypeError("token_ids must contain integers")
        for name in ("lower_layer_states", "suffix_layer_states"):
            states = getattr(self, name)
            if not isinstance(states, tuple) or any(
                not isinstance(state, (AttentionState, DeltaNetState))
                for state in states
            ):
                raise TypeError(f"{name} must contain concrete layer states")
        hidden = self.last_hidden
        if (
            not isinstance(hidden, torch.Tensor)
            or not hidden.is_floating_point()
            or hidden.ndim != 3
            or hidden.shape[0] != 1
            or hidden.shape[1] != 1
        ):
            raise ValueError("last_hidden must have floating [1, 1, dim] shape")

    @property
    def layer_states(self) -> tuple[LayerState, ...]:
        return self.lower_layer_states + self.suffix_layer_states


@dataclass(frozen=True, slots=True)
class Qwen38ForkState:
    """One atomic pair state, either lockstep-joined or permanently split."""

    depth: int
    joined: bool
    common_layer_states: tuple[LayerState, ...]
    off: Qwen38ForkArmState
    native: Qwen38ForkArmState
    first_divergence_position: int | None = None

    def _validate_token_divergence(self) -> None:
        """Bind the split marker to the real, complete token histories."""

        expected = _first_token_divergence(self.off.token_ids, self.native.token_ids)
        if self.joined:
            if expected is not None:
                raise ValueError("joined histories contain a token divergence")
            if self.first_divergence_position is not None:
                raise ValueError("joined state cannot report token divergence")
            return
        if expected is None:
            raise ValueError("split state has no divergence in its token histories")
        position = self.first_divergence_position
        if position is None:
            raise ValueError("split state must record its first divergence")
        _nonnegative(position, "first_divergence_position")
        if position != expected:
            raise ValueError(
                "first_divergence_position must equal the first differing token "
                f"index {expected}, got {position}"
            )
        if position >= max(self.off.next_position, self.native.next_position):
            raise ValueError("first_divergence_position must precede an arm cursor")

    def __post_init__(self) -> None:
        depth = _positive(self.depth, "depth")
        if depth <= FORK_LAYER:
            raise ValueError("fork state depth does not contain layer 27")
        if not isinstance(self.joined, bool):
            raise TypeError("joined must be boolean")
        if not isinstance(self.common_layer_states, tuple) or any(
            not isinstance(state, (AttentionState, DeltaNetState))
            for state in self.common_layer_states
        ):
            raise TypeError("common_layer_states must contain concrete states")
        suffix_length = depth - SUFFIX_START_LAYER
        if (
            len(self.off.suffix_layer_states) != suffix_length
            or len(self.native.suffix_layer_states) != suffix_length
        ):
            raise ValueError("arm suffix does not cover layers 27..depth")
        if self.joined:
            if len(self.common_layer_states) != COMMON_STOP_LAYER:
                raise ValueError("joined state must own common layers 0..26")
            if self.off.lower_layer_states or self.native.lower_layer_states:
                raise ValueError("joined arms may not duplicate common lower state")
            if self.off.next_position != self.native.next_position:
                raise ValueError("joined arms must share one cursor")
            if self.off.token_ids != self.native.token_ids:
                raise ValueError("joined arms must share one token history")
        else:
            if self.common_layer_states:
                raise ValueError("split state may not retain a live common prefix")
            if (
                len(self.off.lower_layer_states) != COMMON_STOP_LAYER
                or len(self.native.lower_layer_states) != COMMON_STOP_LAYER
            ):
                raise ValueError("split arms must own complete lower states")
        self._validate_token_divergence()


@dataclass(frozen=True, slots=True)
class Qwen38ForkForwardResult:
    """New atomic state and receipts for one prefill or decode operation."""

    state: Qwen38ForkState
    traffic: Qwen38ForkTraffic
    seconds: float

    @property
    def off_hidden(self) -> torch.Tensor:
        return self.state.off.last_hidden

    @property
    def native_hidden(self) -> torch.Tensor:
        return self.state.native.last_hidden


@dataclass(frozen=True, slots=True)
class Qwen38ForkGenerationEvidence:
    """Receipt for two exact greedy token streams and their committed state."""

    prompt_token_ids: tuple[int, ...]
    off_generated_token_ids: tuple[int, ...]
    native_generated_token_ids: tuple[int, ...]
    off_stopped_on_eos: bool
    native_stopped_on_eos: bool
    first_divergence_step: int | None
    final_joined: bool
    off_forward_passes: int
    native_forward_passes: int
    traffic: Qwen38ForkTraffic
    seconds: float


@dataclass(frozen=True, slots=True)
class Qwen38ForkGenerationResult:
    """Generated tokens, committed pair state, and exact execution evidence."""

    off_token_ids: tuple[int, ...]
    native_token_ids: tuple[int, ...]
    state: Qwen38ForkState
    evidence: Qwen38ForkGenerationEvidence


class Qwen38NativeFork:
    """Execute one off/native Qwen pair with a real native layer-27 fork."""

    def __init__(
        self,
        config: Qwen38Config,
        pager: Qwen38WeightPager,
        *,
        native_head_crsa: Qwen38NativeHeadCrsa | None = None,
        max_seq_len: int = 4096,
    ) -> None:
        if not isinstance(config, Qwen38Config):
            raise TypeError("config must be Qwen38Config")
        if not isinstance(pager, Qwen38WeightPager):
            raise TypeError("pager must be Qwen38WeightPager")
        intervention = (
            Qwen38NativeHeadCrsa() if native_head_crsa is None else native_head_crsa
        )
        if not isinstance(intervention, Qwen38NativeHeadCrsa):
            raise TypeError("native_head_crsa must be a Qwen38NativeHeadCrsa")
        if config.n_layers <= FORK_LAYER or not config.is_full_attention(FORK_LAYER):
            raise ValueError("Qwen native fork requires full-attention layer 27")
        if config.n_heads != 24 or config.n_kv_heads != 4:
            raise ValueError("Qwen native fork requires the 24-query/4-KV layout")
        if (
            isinstance(max_seq_len, bool)
            or not isinstance(max_seq_len, int)
            or max_seq_len <= 0
        ):
            raise ValueError("max_seq_len must be a positive integer")
        if max_seq_len > config.max_position_embeddings:
            raise ValueError("max_seq_len exceeds the checkpoint context bound")

        self.config = config
        self.pager = pager
        self.native_head_crsa = intervention
        self.max_seq_len = max_seq_len
        # These models are stateless range executors here.  Their own cursors
        # stay at zero; all continuation state lives in Qwen38ForkState.
        self._off_model = StreamedQwen38(
            config,
            pager,
            max_batch_size=1,
            max_seq_len=max_seq_len,
        )
        self._native_model = StreamedQwen38(
            config,
            pager,
            native_head_crsa=intervention,
            max_batch_size=1,
            max_seq_len=max_seq_len,
        )
        self.layer_accounting = Qwen38ForkLayerAccounting.for_depth(config.n_layers)

    @staticmethod
    def complete_layer_savings(depth: int = 64) -> tuple[int, int]:
        """Return saved/independent complete-layer work for one joined pair."""

        accounting = Qwen38ForkLayerAccounting.for_depth(depth)
        return (
            accounting.complete_layers_saved,
            accounting.independent_complete_layers,
        )

    def checkpoint_preflight(self) -> dict[str, Any]:
        """Validate the shared checkpoint inventory without payload reads."""

        return self._off_model.checkpoint_preflight()

    def _source_bytes(self) -> int:
        return StreamedQwen38._metric(self.pager.source, "network_or_source_body_bytes")

    def _linear_calls(self) -> int:
        return StreamedQwen38._metric(self.pager, "linear_calls")

    def _token_tensor(self, token_ids: Any) -> torch.Tensor:
        ids = self._off_model._token_tensor(token_ids)
        if ids.shape[0] != 1:
            raise ValueError("native fork supports exactly one sequence")
        return ids

    def _validate_state(self, state: Qwen38ForkState) -> None:
        if not isinstance(state, Qwen38ForkState):
            raise TypeError("state must be a Qwen38ForkState")
        if state.depth != self.config.n_layers:
            raise ValueError("fork state depth disagrees with the runtime")
        state._validate_token_divergence()
        for arm in (state.off, state.native):
            if len(arm.token_ids) != arm.next_position:
                raise ValueError("fork token history disagrees with its arm cursor")
            if arm.last_hidden.shape[-1] != self.config.dim:
                raise ValueError("fork hidden width disagrees with the runtime")
            if arm.last_hidden.device != self.pager.device:
                raise ValueError("fork hidden device disagrees with the pager")
            if arm.last_hidden.dtype != self.pager.compute_dtype:
                raise ValueError("fork hidden dtype disagrees with the pager")
            if any(
                token < 0 or token >= self.config.vocab_size for token in arm.token_ids
            ):
                raise ValueError("fork token history exceeds checkpoint vocabulary")

        off_states = self._full_states(state, "off")
        native_states = self._full_states(state, "native")
        # Validate both explicit continuation boundaries before the pager can
        # read a single source byte.  In particular, a reconstructed malformed
        # native arm cannot fail only after an already staged off forward.
        self._off_model._validate_stateful_range_states(
            off_states,
            batch_size=1,
            sequence_length=1,
            start_pos=state.off.next_position,
            start_layer=0,
            graft_history=None,
        )
        self._native_model._validate_stateful_range_states(
            native_states,
            batch_size=1,
            sequence_length=1,
            start_pos=state.native.next_position,
            start_layer=0,
            graft_history=None,
        )
        if state.joined:
            off_fork = off_states[FORK_LAYER]
            native_fork = native_states[FORK_LAYER]
            if not isinstance(off_fork, AttentionState) or not isinstance(
                native_fork, AttentionState
            ):
                raise Qwen38RuntimeError("joined fork layer must contain KV state")
            if (
                off_fork.key is not native_fork.key
                or off_fork.value is not native_fork.value
            ):
                raise Qwen38RuntimeError(
                    "joined layer-27 K/V must share tensor objects"
                )
            if not torch.equal(off_fork.key, native_fork.key) or not torch.equal(
                off_fork.value, native_fork.value
            ):
                raise Qwen38RuntimeError("joined layer-27 K/V values diverged")

    @staticmethod
    def _concrete_states(
        states: Iterable[LayerState | None],
    ) -> tuple[LayerState, ...]:
        values = tuple(states)
        if any(state is None for state in values):
            raise Qwen38RuntimeError("fork range returned incomplete layer state")
        return tuple(state for state in values if state is not None)

    def _full_states(
        self, state: Qwen38ForkState, arm: Literal["off", "native"]
    ) -> tuple[LayerState, ...]:
        selected = state.off if arm == "off" else state.native
        if state.joined:
            return state.common_layer_states + selected.suffix_layer_states
        return selected.layer_states

    def _paired_final_norm(
        self, off_hidden: torch.Tensor, native_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        started_bytes = self._source_bytes()
        weight = self.pager.tensor_torch(
            StreamedQwen38.FINAL_NORM_NAME,
            dtype=torch.float32,
            device=self.pager.device,
        )
        try:
            off = rms_norm(off_hidden, weight, eps=self.config.rms_norm_eps)
            native = rms_norm(native_hidden, weight, eps=self.config.rms_norm_eps)
        finally:
            del weight
            self.pager.release()
        return off, native, self._source_bytes() - started_bytes

    def _fork_layer_27(
        self,
        hidden: torch.Tensor,
        *,
        start_pos: int,
        off_state: AttentionState | None,
        native_state: AttentionState | None,
    ) -> tuple[
        torch.Tensor,
        AttentionState,
        torch.Tensor,
        AttentionState,
        NativeHeadCrsaEvidence,
        Qwen38ForkTraffic,
    ]:
        """Execute layer 27 with one checkpoint pass for every matrix."""

        prefix = f"model.language_model.layers.{FORK_LAYER}"
        attention = f"{prefix}.self_attn"
        mlp = f"{prefix}.mlp"
        start_bytes = self._source_bytes()
        start_linears = self._linear_calls()
        residual = hidden
        input_norm_weight = self.pager.tensor_torch(
            f"{prefix}.input_layernorm.weight",
            dtype=torch.float32,
            device=self.pager.device,
        )
        try:
            mixed_input = rms_norm(
                hidden, input_norm_weight, eps=self.config.rms_norm_eps
            )
        finally:
            del input_norm_weight

        query_gate = self.pager.linear(mixed_input, f"{attention}.q_proj")
        key = self.pager.linear(mixed_input, f"{attention}.k_proj")
        value = self.pager.linear(mixed_input, f"{attention}.v_proj")
        q_norm = self.pager.tensor_torch(
            f"{attention}.q_norm.weight",
            dtype=torch.float32,
            device=self.pager.device,
        )
        k_norm = self.pager.tensor_torch(
            f"{attention}.k_norm.weight",
            dtype=torch.float32,
            device=self.pager.device,
        )
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
            (
                off_mixed,
                off_next,
                native_mixed,
                native_next,
                native_evidence,
            ) = full_attention_fork_core(
                query_gate,
                key,
                value,
                q_norm_weight=q_norm,
                k_norm_weight=k_norm,
                num_attention_heads=self.config.n_heads,
                num_key_value_heads=self.config.n_kv_heads,
                head_dim=self.config.head_dim,
                native_head_crsa=self.native_head_crsa,
                position_ids=positions,
                off_state=off_state,
                native_state=native_state,
                rope_theta=self.config.rope_theta,
                rotary_dim=self.config.rotary_dim,
                mrope_section=self.config.mrope_section,
                mrope_interleaved=self.config.mrope_interleaved,
                rms_norm_eps=self.config.rms_norm_eps,
            )
        finally:
            del query_gate, key, value, q_norm, k_norm
        off_projected, native_projected = self.pager.linear_many(
            (off_mixed, native_mixed), f"{attention}.o_proj"
        )
        del off_mixed, native_mixed
        off_hidden = residual + off_projected
        native_hidden = residual + native_projected
        del off_projected, native_projected

        post_norm_weight = self.pager.tensor_torch(
            f"{prefix}.post_attention_layernorm.weight",
            dtype=torch.float32,
            device=self.pager.device,
        )
        try:
            off_mlp_input = rms_norm(
                off_hidden, post_norm_weight, eps=self.config.rms_norm_eps
            )
            native_mlp_input = rms_norm(
                native_hidden, post_norm_weight, eps=self.config.rms_norm_eps
            )
        finally:
            del post_norm_weight
        off_gate, native_gate = self.pager.linear_many(
            (off_mlp_input, native_mlp_input), f"{mlp}.gate_proj"
        )
        off_up, native_up = self.pager.linear_many(
            (off_mlp_input, native_mlp_input), f"{mlp}.up_proj"
        )
        del off_mlp_input, native_mlp_input
        off_activated = swiglu(off_gate, off_up)
        native_activated = swiglu(native_gate, native_up)
        del off_gate, native_gate, off_up, native_up
        off_down, native_down = self.pager.linear_many(
            (off_activated, native_activated), f"{mlp}.down_proj"
        )
        del off_activated, native_activated
        off_hidden = off_hidden + off_down
        native_hidden = native_hidden + native_down
        del off_down, native_down
        self.pager.release()

        linear_delta = self._linear_calls() - start_linears
        branch_linears = 4
        if linear_delta < 2 * branch_linears:
            raise Qwen38RuntimeError("fork layer linear accounting underflow")
        traffic = Qwen38ForkTraffic(
            shared_source_body_bytes=self._source_bytes() - start_bytes,
            shared_linear_calls=linear_delta - 2 * branch_linears,
            off_linear_calls=branch_linears,
            native_linear_calls=branch_linears,
            off_layers=1,
            native_layers=1,
            independent_complete_layers=2,
            fork_layer_weight_passes=1,
            native_head_crsa_evidence=(native_evidence,),
        )
        return (
            off_hidden,
            off_next,
            native_hidden,
            native_next,
            native_evidence,
            traffic,
        )

    def _joined_forward(
        self,
        ids: torch.Tensor,
        *,
        start_pos: int,
        common_states: tuple[LayerState, ...] | tuple[None, ...],
        off_suffix: tuple[LayerState, ...] | tuple[None, ...],
        native_suffix: tuple[LayerState, ...] | tuple[None, ...],
        off_history: tuple[int, ...],
        native_history: tuple[int, ...],
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> Qwen38ForkForwardResult:
        started = time.perf_counter()
        start_embed_bytes = self._source_bytes()
        hidden = self._off_model.embed_batch(ids)
        embed_bytes = self._source_bytes() - start_embed_bytes
        empty_or_full = tuple(common_states) + tuple(off_suffix)
        lower = self._off_model.hidden_stateful_range(
            hidden,
            empty_or_full,
            start_pos=start_pos,
            start_layer=0,
            stop_layer=COMMON_STOP_LAYER,
            progress=progress,
        )
        common = self._concrete_states(lower.layer_states[:COMMON_STOP_LAYER])
        prior_off = off_suffix[0]
        prior_native = native_suffix[0]
        if prior_off is not None and not isinstance(prior_off, AttentionState):
            raise Qwen38RuntimeError("off fork layer has the wrong state type")
        if prior_native is not None and not isinstance(prior_native, AttentionState):
            raise Qwen38RuntimeError("native fork layer has the wrong state type")
        (
            off_hidden,
            off_layer_state,
            native_hidden,
            native_layer_state,
            _native_evidence,
            fork_traffic,
        ) = self._fork_layer_27(
            lower.hidden,
            start_pos=start_pos,
            off_state=prior_off,
            native_state=prior_native,
        )

        # ``lower`` was validated with the off suffix only because layers
        # 0..26 are the executed range.  Never leak those untouched off
        # entries into the native continuation boundary on joined decode.
        staged_common = list(lower.layer_states[:COMMON_STOP_LAYER])
        off_states: list[LayerState | None] = staged_common + list(off_suffix)
        native_states: list[LayerState | None] = staged_common + list(native_suffix)
        off_states[FORK_LAYER] = off_layer_state
        native_states[FORK_LAYER] = native_layer_state
        off_upper = self._off_model.hidden_stateful_range(
            off_hidden,
            off_states,
            start_pos=start_pos,
            start_layer=FORK_LAYER + 1,
            stop_layer=self.config.n_layers,
            progress=progress,
        )
        native_upper = self._native_model.hidden_stateful_range(
            native_hidden,
            native_states,
            start_pos=start_pos,
            start_layer=FORK_LAYER + 1,
            stop_layer=self.config.n_layers,
            progress=progress,
        )
        off_final, native_final, final_norm_bytes = self._paired_final_norm(
            off_upper.hidden, native_upper.hidden
        )
        off_all = self._concrete_states(off_upper.layer_states)
        native_all = self._concrete_states(native_upper.layer_states)
        token_values = tuple(int(value) for value in ids[0].detach().cpu().tolist())
        end_pos = start_pos + ids.shape[1]
        state = Qwen38ForkState(
            depth=self.config.n_layers,
            joined=True,
            common_layer_states=common,
            off=Qwen38ForkArmState(
                next_position=end_pos,
                token_ids=off_history + token_values,
                lower_layer_states=(),
                suffix_layer_states=off_all[SUFFIX_START_LAYER:],
                last_hidden=off_final[:, -1:],
            ),
            native=Qwen38ForkArmState(
                next_position=end_pos,
                token_ids=native_history + token_values,
                lower_layer_states=(),
                suffix_layer_states=native_all[SUFFIX_START_LAYER:],
                last_hidden=native_final[:, -1:],
            ),
        )
        upper_layers = self.config.n_layers - (FORK_LAYER + 1)
        traffic = (
            Qwen38ForkTraffic(
                shared_source_body_bytes=(
                    embed_bytes + lower.evidence.source_body_bytes + final_norm_bytes
                ),
                shared_linear_calls=lower.evidence.linear_calls,
                shared_layers=COMMON_STOP_LAYER,
                off_source_body_bytes=off_upper.evidence.source_body_bytes,
                off_linear_calls=off_upper.evidence.linear_calls,
                off_layers=upper_layers,
                native_source_body_bytes=native_upper.evidence.source_body_bytes,
                native_linear_calls=native_upper.evidence.linear_calls,
                native_layers=upper_layers,
                independent_complete_layers=2 * self.config.n_layers - 2,
                complete_layers_saved=COMMON_STOP_LAYER,
                native_head_crsa_evidence=(
                    off_upper.native_head_crsa_evidence
                    + native_upper.native_head_crsa_evidence
                ),
            )
            + fork_traffic
        )
        return Qwen38ForkForwardResult(
            state=state,
            traffic=traffic,
            seconds=time.perf_counter() - started,
        )

    def prefill(
        self,
        prompt_token_ids: Any,
        *,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> Qwen38ForkForwardResult:
        """Prefill the prompt once through the shared native fork."""

        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable or None")
        ids = self._token_tensor(prompt_token_ids)
        if ids.shape[1] > self.max_seq_len:
            raise ValueError("prompt exceeds max_seq_len")
        empty_common: tuple[None, ...] = (None,) * COMMON_STOP_LAYER
        empty_suffix: tuple[None, ...] = (None,) * (
            self.config.n_layers - SUFFIX_START_LAYER
        )
        return self._joined_forward(
            ids,
            start_pos=0,
            common_states=empty_common,
            off_suffix=empty_suffix,
            native_suffix=empty_suffix,
            off_history=(),
            native_history=(),
            progress=progress,
        )

    def _independent_arm_forward(
        self,
        *,
        arm: Literal["off", "native"],
        token_id: int,
        position: int,
        prior_states: tuple[LayerState, ...],
        history: tuple[int, ...],
        progress: Callable[[dict[str, Any]], None] | None,
    ) -> tuple[Qwen38ForkArmState, Qwen38ForkTraffic]:
        model = self._off_model if arm == "off" else self._native_model
        start_bytes = self._source_bytes()
        start_linears = self._linear_calls()
        hidden = model.embed_batch([[token_id]])
        staged = model.hidden_stateful_range(
            hidden,
            prior_states,
            start_pos=position,
            start_layer=0,
            stop_layer=self.config.n_layers,
            progress=progress,
        )
        final = model.finalize_hidden(staged.hidden)
        self.pager.release()
        concrete = self._concrete_states(staged.layer_states)
        next_state = Qwen38ForkArmState(
            next_position=position + 1,
            token_ids=history + (token_id,),
            lower_layer_states=concrete[:COMMON_STOP_LAYER],
            suffix_layer_states=concrete[SUFFIX_START_LAYER:],
            last_hidden=final[:, -1:],
        )
        kwargs: dict[str, Any] = {
            f"{arm}_source_body_bytes": self._source_bytes() - start_bytes,
            f"{arm}_linear_calls": self._linear_calls() - start_linears,
            f"{arm}_layers": self.config.n_layers,
            "independent_complete_layers": self.config.n_layers,
            "native_head_crsa_evidence": staged.native_head_crsa_evidence,
        }
        return next_state, Qwen38ForkTraffic(**kwargs)

    def decode_pair(
        self,
        state: Qwen38ForkState,
        off_token_id: int,
        native_token_id: int,
        *,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> Qwen38ForkForwardResult:
        """Atomically commit one emitted token to each arm."""

        self._validate_state(state)
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable or None")
        for token, name in (
            (off_token_id, "off_token_id"),
            (native_token_id, "native_token_id"),
        ):
            if isinstance(token, bool) or not isinstance(token, int):
                raise TypeError(f"{name} must be an integer")
            if token < 0 or token >= self.config.vocab_size:
                raise ValueError(f"{name} outside checkpoint vocabulary")
        if max(state.off.next_position, state.native.next_position) >= self.max_seq_len:
            raise ValueError("decode would exceed max_seq_len")
        started = time.perf_counter()
        if state.joined and off_token_id == native_token_id:
            return self._joined_forward(
                self._token_tensor([[off_token_id]]),
                start_pos=state.off.next_position,
                common_states=state.common_layer_states,
                off_suffix=state.off.suffix_layer_states,
                native_suffix=state.native.suffix_layer_states,
                off_history=state.off.token_ids,
                native_history=state.native.token_ids,
                progress=progress,
            )

        # Functional staging gives pair-atomicity: if native fails after off
        # succeeds, neither staged arm is returned and ``state`` is untouched.
        off_next, off_traffic = self._independent_arm_forward(
            arm="off",
            token_id=off_token_id,
            position=state.off.next_position,
            prior_states=self._full_states(state, "off"),
            history=state.off.token_ids,
            progress=progress,
        )
        native_next, native_traffic = self._independent_arm_forward(
            arm="native",
            token_id=native_token_id,
            position=state.native.next_position,
            prior_states=self._full_states(state, "native"),
            history=state.native.token_ids,
            progress=progress,
        )
        divergence = (
            state.first_divergence_position
            if not state.joined
            else state.off.next_position
        )
        next_state = Qwen38ForkState(
            depth=self.config.n_layers,
            joined=False,
            common_layer_states=(),
            off=off_next,
            native=native_next,
            first_divergence_position=divergence,
        )
        return Qwen38ForkForwardResult(
            state=next_state,
            traffic=off_traffic + native_traffic,
            seconds=time.perf_counter() - started,
        )

    # A short alias mirrors StreamedQwen38.decode without hiding the pair
    # semantics in call sites.
    decode = decode_pair

    def decode_arm(
        self,
        state: Qwen38ForkState,
        *,
        arm: Literal["off", "native"],
        token_id: int,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> Qwen38ForkForwardResult:
        """Commit one token after the other arm has stopped generation."""

        self._validate_state(state)
        if arm not in {"off", "native"}:
            raise ValueError("arm must be 'off' or 'native'")
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("token_id must be an integer")
        if token_id < 0 or token_id >= self.config.vocab_size:
            raise ValueError("token_id outside checkpoint vocabulary")
        selected = state.off if arm == "off" else state.native
        if selected.next_position >= self.max_seq_len:
            raise ValueError("decode would exceed max_seq_len")
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable or None")
        started = time.perf_counter()
        next_arm, traffic = self._independent_arm_forward(
            arm=arm,
            token_id=token_id,
            position=selected.next_position,
            prior_states=self._full_states(state, arm),
            history=selected.token_ids,
            progress=progress,
        )
        if state.joined:
            other_name: Literal["off", "native"] = "native" if arm == "off" else "off"
            other_old = state.native if arm == "off" else state.off
            other_full = self._full_states(state, other_name)
            other = replace(
                other_old,
                lower_layer_states=other_full[:COMMON_STOP_LAYER],
                suffix_layer_states=other_full[SUFFIX_START_LAYER:],
            )
            divergence = selected.next_position
        else:
            other = state.native if arm == "off" else state.off
            divergence = state.first_divergence_position
        off = next_arm if arm == "off" else other
        native = next_arm if arm == "native" else other
        next_state = Qwen38ForkState(
            depth=self.config.n_layers,
            joined=False,
            common_layer_states=(),
            off=off,
            native=native,
            first_divergence_position=divergence,
        )
        return Qwen38ForkForwardResult(
            state=next_state,
            traffic=traffic,
            seconds=time.perf_counter() - started,
        )

    @staticmethod
    def _eos_set(eos_token_ids: Iterable[int], vocab_size: int) -> set[int]:
        if isinstance(eos_token_ids, (str, bytes)):
            raise TypeError("eos_token_ids must be an iterable of integers")
        eos: set[int] = set()
        try:
            for token in eos_token_ids:
                if isinstance(token, bool) or not isinstance(token, int):
                    raise TypeError("eos_token_ids must contain integers")
                eos.add(token)
        except TypeError as exc:
            if str(exc) == "eos_token_ids must contain integers":
                raise
            raise TypeError("eos_token_ids must be iterable") from exc
        if any(token < 0 or token >= vocab_size for token in eos):
            raise ValueError("EOS token outside checkpoint vocabulary")
        return eos

    def _score_arm(
        self,
        arm: Literal["off", "native"],
        hidden: torch.Tensor,
        *,
        head_block_rows: int,
        head_progress: Callable[[dict[str, int]], None] | None,
    ) -> tuple[int, float, Qwen38ForkTraffic]:
        start_bytes = self._source_bytes()
        values, ids = self.pager.topk_logits(
            hidden[:, -1],
            k=1,
            name=self._off_model.output_head_name,
            block_rows=head_block_rows,
            progress=head_progress,
        )
        token = int(ids[0, 0].item())
        logit = float(values[0, 0].item())
        kwargs = {
            f"{arm}_source_body_bytes": self._source_bytes() - start_bytes,
        }
        return token, logit, Qwen38ForkTraffic(**kwargs)

    def generate_greedy(
        self,
        prompt_token_ids: Any,
        *,
        max_new_tokens: int = 1,
        eos_token_ids: Iterable[int] = (),
        head_block_rows: int = Qwen38WeightPager.DEFAULT_HEAD_BLOCK_ROWS,
        progress: Callable[[dict[str, Any]], None] | None = None,
        head_progress: Callable[[dict[str, int]], None] | None = None,
    ) -> Qwen38ForkGenerationResult:
        """Generate both arms greedily, committing every emitted token."""

        _positive(max_new_tokens, "max_new_tokens")
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable or None")
        if head_progress is not None and not callable(head_progress):
            raise TypeError("head_progress must be callable or None")
        prompt = self._token_tensor(prompt_token_ids)
        if prompt.shape[1] + max_new_tokens > self.max_seq_len:
            raise ValueError("generation would exceed max_seq_len")
        eos = self._eos_set(eos_token_ids, self.config.vocab_size)
        started = time.perf_counter()
        prefill = self.prefill(prompt, progress=progress)
        state = prefill.state
        traffic = prefill.traffic
        off_generated: list[int] = []
        native_generated: list[int] = []
        off_stopped = False
        native_stopped = False
        first_divergence_step: int | None = None

        for step in range(max_new_tokens):
            off_token: int | None = None
            native_token: int | None = None
            if not off_stopped:
                off_token, off_logit, scored = self._score_arm(
                    "off",
                    state.off.last_hidden,
                    head_block_rows=head_block_rows,
                    head_progress=head_progress,
                )
                traffic = traffic + scored
                off_generated.append(off_token)
                if progress is not None:
                    progress(
                        {
                            "event": "generated_token",
                            "arm": "off",
                            "step": step,
                            "token_id": off_token,
                            "logit": off_logit,
                        }
                    )
            if not native_stopped:
                native_token, native_logit, scored = self._score_arm(
                    "native",
                    state.native.last_hidden,
                    head_block_rows=head_block_rows,
                    head_progress=head_progress,
                )
                traffic = traffic + scored
                native_generated.append(native_token)
                if progress is not None:
                    progress(
                        {
                            "event": "generated_token",
                            "arm": "native",
                            "step": step,
                            "token_id": native_token,
                            "logit": native_logit,
                        }
                    )

            if off_token is not None and native_token is not None:
                was_joined = state.joined
                committed = self.decode_pair(
                    state,
                    off_token,
                    native_token,
                    progress=progress,
                )
                if was_joined and off_token != native_token:
                    first_divergence_step = step
            elif off_token is not None:
                committed = self.decode_arm(
                    state,
                    arm="off",
                    token_id=off_token,
                    progress=progress,
                )
            elif native_token is not None:
                committed = self.decode_arm(
                    state,
                    arm="native",
                    token_id=native_token,
                    progress=progress,
                )
            else:  # Both arms already stopped in the previous iteration.
                break
            state = committed.state
            traffic = traffic + committed.traffic
            # EOS is observed only after its token has been committed.
            if off_token is not None and off_token in eos:
                off_stopped = True
            if native_token is not None and native_token in eos:
                native_stopped = True
            if off_stopped and native_stopped:
                break

        prompt_ids = tuple(int(value) for value in prompt[0].detach().cpu().tolist())
        evidence = Qwen38ForkGenerationEvidence(
            prompt_token_ids=prompt_ids,
            off_generated_token_ids=tuple(off_generated),
            native_generated_token_ids=tuple(native_generated),
            off_stopped_on_eos=off_stopped,
            native_stopped_on_eos=native_stopped,
            first_divergence_step=first_divergence_step,
            final_joined=state.joined,
            off_forward_passes=1 + len(off_generated),
            native_forward_passes=1 + len(native_generated),
            traffic=traffic,
            seconds=time.perf_counter() - started,
        )
        return Qwen38ForkGenerationResult(
            off_token_ids=tuple(off_generated),
            native_token_ids=tuple(native_generated),
            state=state,
            evidence=evidence,
        )


__all__ = [
    "COMMON_STOP_LAYER",
    "FORK_LAYER",
    "Qwen38ForkArmState",
    "Qwen38ForkForwardResult",
    "Qwen38ForkGenerationEvidence",
    "Qwen38ForkGenerationResult",
    "Qwen38ForkLayerAccounting",
    "Qwen38ForkState",
    "Qwen38ForkTraffic",
    "Qwen38NativeFork",
    "SUFFIX_START_LAYER",
]
