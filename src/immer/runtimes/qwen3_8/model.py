"""Exact text-only, layer-paged Qwen3.8 decoder.

The model keeps activations resident while :class:`Qwen38WeightPager` streams
one BF16 checkpoint matrix at a time.  It implements the public Qwen3.5 text
equation directly; Transformers is neither imported nor used as a runtime.
Vision and the optional MTP block are deliberately outside this correctness
path.  MTP can later draft tokens, but it must never alter base-model parity.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
import hashlib
import json
import math
import os
import time
from typing import Any
import warnings

import torch

from .config import Qwen38Config
from .kernels import (
    AttentionState,
    DeltaNetProbe,
    DeltaNetState,
    full_attention_core,
    gated_delta_net_core,
    recurrent_gated_delta_rule,
    rms_norm,
    swiglu,
)
from .native_crsa import (
    NativeHeadCrsaEvidence,
    NativePrefixSinkhornOperatorBlock,
    NativePrefixSinkhornOperatorObserver,
    Qwen38NativeHeadCrsa,
)
from .pager import Qwen38WeightPager
from .provenance import runtime_dependency_versions, runtime_source_manifest
from .snapshot import (
    Qwen38SnapshotError,
    SnapshotLimits,
    SnapshotTensor,
    read_qwen38_snapshot,
    write_qwen38_snapshot,
)


class Qwen38RuntimeError(RuntimeError):
    """The streamed checkpoint violates the executable text-model contract."""


LAYER_BOUNDARY_STAGES = (
    "layer.input",
    "attention.input",
    "attention.output",
    "attention.residual",
    "mlp.input",
    "mlp.gate",
    "mlp.up",
    "mlp.activated",
    "mlp.output",
    "layer.output",
)
LayerBoundaryObserver = Callable[[int, str, torch.Tensor], None]


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
class StatefulLayerRangeEvidence:
    """Accounting receipt for one non-committing decoder-layer range."""

    start_pos: int
    end_pos: int
    start_layer: int
    stop_layer: int
    layers_executed: int
    source_body_bytes: int
    linear_calls: int
    seconds: float
    staged_state_bytes: int


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
    final_state_committed: bool
    time_to_first_token_seconds: float
    output_tokens_per_second: float


LayerState = AttentionState | DeltaNetState


@dataclass(frozen=True, slots=True)
class _DeltaNetPrefixUpdate:
    key: torch.Tensor
    value: torch.Tensor
    beta: torch.Tensor
    log_decay: torch.Tensor

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.key, self.value, self.beta, self.log_decay)
        )


@dataclass(frozen=True, slots=True)
class _LayerPrefixTrace:
    attention_log_usage: tuple[torch.Tensor | None, ...] = ()
    delta_updates: tuple[_DeltaNetPrefixUpdate, ...] = ()
    # Raw one-position Conv inputs that have fallen out of the final rolling
    # Conv state.  Windows no wider than the kernel retain no extra copy.
    delta_conv_prefix: tuple[torch.Tensor, ...] = ()

    @property
    def width(self) -> int:
        return max(
            len(self.attention_log_usage),
            len(self.delta_updates),
            len(self.delta_conv_prefix),
        )

    @property
    def nbytes(self) -> int:
        usage = sum(
            0 if row is None else row.numel() * row.element_size()
            for row in self.attention_log_usage
        )
        conv = sum(row.numel() * row.element_size() for row in self.delta_conv_prefix)
        return usage + conv + sum(row.nbytes for row in self.delta_updates)


@dataclass(frozen=True, slots=True)
class _ContinuationPrefixTrace:
    width: int
    layers: tuple[_LayerPrefixTrace, ...]
    graft_histories: tuple[torch.Tensor | None, ...]
    native_operator_blocks: tuple[NativePrefixSinkhornOperatorBlock, ...]

    @property
    def nbytes(self) -> int:
        layer_bytes = sum(row.nbytes for row in self.layers)
        graft_bytes = sum(
            0 if row is None else row.numel() * row.element_size()
            for row in self.graft_histories
        )
        operator_bytes = sum(row.operators.nbytes for row in self.native_operator_blocks)
        return layer_bytes + graft_bytes + operator_bytes


@dataclass(frozen=True, slots=True)
class StatefulLayerRangeResult:
    """Hidden state, continuation state, and receipts staged by a layer range."""

    hidden: torch.Tensor
    layer_states: tuple[LayerState | None, ...]
    graft_history: torch.Tensor | None
    native_head_crsa_evidence: tuple[NativeHeadCrsaEvidence, ...]
    evidence: StatefulLayerRangeEvidence
    prefix_trace: _ContinuationPrefixTrace | None = None


@dataclass(frozen=True, slots=True)
class StatefulBlockEvidence:
    """Receipt for one complete, non-committing continuation block."""

    start_pos: int
    end_pos: int
    input_token_ids: tuple[tuple[int, ...], ...]
    layers_executed: int
    checkpoint_layers: int
    complete_layer_stack: bool
    stateful_cache: bool
    source_body_bytes: int
    linear_calls: int
    seconds: float
    staged_state_bytes: int
    graft_mode: str
    graft_history_tokens: int


@dataclass(frozen=True, slots=True)
class StatefulBlockStage:
    """Opaque handle plus cloned verification hidden for one pending block."""

    hidden: torch.Tensor
    evidence: StatefulBlockEvidence
    _nonce: object


@dataclass(frozen=True, slots=True)
class _PendingStatefulBlock:
    """Private commit material; never returned to an untrusted verifier."""

    handle: StatefulBlockStage
    evidence: StatefulBlockEvidence
    hidden: torch.Tensor
    layer_states: tuple[LayerState | None, ...]
    graft_history: torch.Tensor | None
    native_head_crsa_evidence: tuple[NativeHeadCrsaEvidence, ...]
    prefix_trace: _ContinuationPrefixTrace
    runtime_identity: str


class StreamedQwen38:
    """Qwen3.5 text decoder with bounded, sequential weight residency."""

    # One target weight pass can verify this many draft positions.  Prefix
    # traces retain every recurrent/Conv boundary, so a rejected suffix can be
    # dropped without replay even when the window is wider than the DeltaNet
    # convolution kernel.
    MAX_CONTINUATION_BLOCK_WIDTH = 16
    EMBED_NAME = "model.language_model.embed_tokens.weight"
    FINAL_NORM_NAME = "model.language_model.norm.weight"
    HEAD_NAME = "lm_head.weight"

    @property
    def output_head_name(self) -> str:
        """Return the physical tensor that implements decoder logits."""

        return self.EMBED_NAME if self.config.tie_word_embeddings else self.HEAD_NAME

    def __init__(
        self,
        config: Qwen38Config,
        pager: Qwen38WeightPager,
        *,
        graft: Any | None = None,
        graft_layer: int | None = None,
        delta_probe: Callable[[int, DeltaNetProbe], None] | None = None,
        native_head_crsa: Qwen38NativeHeadCrsa | None = None,
        native_head_crsa_observer: Callable[[NativeHeadCrsaEvidence], None]
        | None = None,
        native_prefix_sinkhorn_operator_observer: (
            NativePrefixSinkhornOperatorObserver | None
        ) = None,
        layer_boundary_observer: LayerBoundaryObserver | None = None,
        layer_boundary_stages: Sequence[str] | None = None,
        layer_boundary_layers: Sequence[int] | None = None,
        mlp_sparse_executor: Any | None = None,
        packed_continuation_gemm: bool = False,
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
        if not isinstance(packed_continuation_gemm, bool):
            raise TypeError("packed_continuation_gemm must be boolean")
        if graft is not None and native_head_crsa is not None:
            raise ValueError(
                "hidden graft and native Head-CRSA intervention are mutually exclusive"
            )
        if graft_layer is not None and (
            isinstance(graft_layer, bool)
            or not isinstance(graft_layer, int)
            or not 0 <= graft_layer < config.n_layers
        ):
            raise ValueError("graft_layer outside decoder depth")
        if graft is not None and graft_layer is None:
            raise ValueError("an active graft requires graft_layer")
        if delta_probe is not None and not callable(delta_probe):
            raise TypeError("delta_probe must be callable")
        if layer_boundary_observer is not None and not callable(
            layer_boundary_observer
        ):
            raise TypeError("layer_boundary_observer must be callable or None")
        if layer_boundary_observer is None and layer_boundary_stages is not None:
            raise ValueError("layer_boundary_stages require a layer_boundary_observer")
        if layer_boundary_observer is None and layer_boundary_layers is not None:
            raise ValueError("layer_boundary_layers require a layer_boundary_observer")
        if mlp_sparse_executor is not None:
            required = (
                "execute",
                "execute_many",
                "snapshot_identity",
                "supports_layer",
            )
            if any(
                not callable(getattr(mlp_sparse_executor, name, None))
                for name in required
            ):
                raise TypeError("mlp_sparse_executor lacks the runtime contract")
            identity = mlp_sparse_executor.snapshot_identity(transport_neutral=True)
            if (
                not isinstance(identity, dict)
                or not isinstance(identity.get("layers"), list)
                or any(
                    isinstance(layer, bool)
                    or not isinstance(layer, int)
                    or not 0 <= layer < config.n_layers
                    for layer in identity["layers"]
                )
            ):
                raise ValueError("mlp_sparse_executor identity is invalid")
        selected_boundary_stages = (
            ()
            if layer_boundary_observer is None
            else (
                LAYER_BOUNDARY_STAGES
                if layer_boundary_stages is None
                else tuple(layer_boundary_stages)
            )
        )
        if len(set(selected_boundary_stages)) != len(selected_boundary_stages) or any(
            stage not in LAYER_BOUNDARY_STAGES for stage in selected_boundary_stages
        ):
            raise ValueError("layer_boundary_stages are invalid or duplicated")
        selected_boundary_layers: tuple[int, ...] | None = None
        if layer_boundary_observer is not None and layer_boundary_layers is not None:
            try:
                selected_boundary_layers = tuple(layer_boundary_layers)
            except TypeError as exc:
                raise TypeError("layer_boundary_layers must be a sequence") from exc
            if (
                not selected_boundary_layers
                or len(set(selected_boundary_layers)) != len(selected_boundary_layers)
                or any(
                    isinstance(layer, bool)
                    or not isinstance(layer, int)
                    or not 0 <= layer < config.n_layers
                    for layer in selected_boundary_layers
                )
            ):
                raise ValueError("layer_boundary_layers are invalid or duplicated")
            selected_boundary_layers = tuple(sorted(selected_boundary_layers))
        if native_head_crsa is not None and not isinstance(
            native_head_crsa, Qwen38NativeHeadCrsa
        ):
            raise TypeError("native_head_crsa must be a Qwen38NativeHeadCrsa or None")
        if native_head_crsa_observer is not None:
            if native_head_crsa is None:
                raise ValueError("native_head_crsa_observer requires native_head_crsa")
            if not callable(native_head_crsa_observer):
                raise TypeError("native_head_crsa_observer must be callable or None")
        if native_prefix_sinkhorn_operator_observer is not None:
            if native_head_crsa is None:
                raise ValueError(
                    "native_prefix_sinkhorn_operator_observer requires native_head_crsa"
                )
            if not isinstance(
                native_prefix_sinkhorn_operator_observer,
                NativePrefixSinkhornOperatorObserver,
            ):
                raise TypeError(
                    "native_prefix_sinkhorn_operator_observer must be bounded or None"
                )
        if native_head_crsa is not None:
            layer = native_head_crsa.layer
            if layer >= config.n_layers or not config.is_full_attention(layer):
                raise ValueError(
                    "native Head-CRSA layer 27 must be a checkpoint full-attention layer"
                )
            if config.n_heads != 24 or config.n_kv_heads != 4:
                raise ValueError(
                    "native Head-CRSA requires the validated 24-query/4-KV GQA layout"
                )
            group_width = config.n_heads // config.n_kv_heads
            mapped = tuple(
                index // group_width for index in native_head_crsa.head_indices
            )
            if mapped != native_head_crsa.selected_kv_heads:
                raise ValueError(
                    "native Head-CRSA must select one query head from every GQA group"
                )

        self.config = config
        self.pager = pager
        self.graft = graft
        self.graft_layer = graft_layer
        self.delta_probe = delta_probe
        self.native_head_crsa = native_head_crsa
        self.native_head_crsa_observer = native_head_crsa_observer
        self.native_prefix_sinkhorn_operator_observer = (
            native_prefix_sinkhorn_operator_observer
        )
        self.layer_boundary_observer = layer_boundary_observer
        self.layer_boundary_stages = tuple(selected_boundary_stages)
        self.layer_boundary_layers = selected_boundary_layers
        self.mlp_sparse_executor = mlp_sparse_executor
        self.packed_continuation_gemm = packed_continuation_gemm
        self.mlp_sparse_last_trace: Any | None = None
        self.mlp_sparse_last_decision: Any | None = None
        self.mlp_sparse_last_observation: Any | None = None
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self._layer_states: list[LayerState | None] = [
            None for _ in range(config.n_layers)
        ]
        self._next_position = 0
        self._state_batch_size: int | None = None
        self._state_poisoned = False
        self._graft_history: torch.Tensor | None = None
        self._pending_block_stage: _PendingStatefulBlock | None = None

    @property
    def next_position(self) -> int:
        return self._next_position

    @property
    def state_batch_size(self) -> int | None:
        return self._state_batch_size

    @property
    def state_poisoned(self) -> bool:
        return self._state_poisoned

    @staticmethod
    def _continuation_bytes(
        states: Iterable[LayerState | None],
        graft_history: torch.Tensor | None,
    ) -> int:
        total = 0
        for state in states:
            if isinstance(state, AttentionState):
                tensors = (state.key, state.value)
                if state.crsa_log_usage is not None:
                    tensors = (*tensors, state.crsa_log_usage)
            elif isinstance(state, DeltaNetState):
                tensors = (state.conv, state.recurrent)
            else:
                continue
            total += sum(tensor.numel() * tensor.element_size() for tensor in tensors)
        if graft_history is not None:
            total += graft_history.numel() * graft_history.element_size()
        return total

    @property
    def state_bytes(self) -> int:
        return self._continuation_bytes(self._layer_states, self._graft_history)

    def _on_pager_device(self, tensor: torch.Tensor) -> bool:
        """Match PyTorch's resolved device against a possibly indexless target."""

        target = self.pager.device
        actual = tensor.device
        return actual.type == target.type and (
            target.index is None or actual.index == target.index
        )

    @staticmethod
    def _snapshot_digest(value: Any) -> str:
        try:
            encoded = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise Qwen38SnapshotError(
                "runtime identity cannot be represented as canonical JSON"
            ) from exc
        return hashlib.sha256(encoded).hexdigest()

    def _native_head_crsa_snapshot_identity(self) -> dict[str, Any]:
        intervention = self.native_head_crsa
        if intervention is None or not intervention.active:
            return {"kind": "none"}
        return {
            "kind": (
                f"{type(intervention).__module__}.{type(intervention).__qualname__}"
            ),
            **asdict(intervention),
        }

    def _native_head_crsa_usage_required(self, layer: int) -> bool:
        intervention = self.native_head_crsa
        return (
            intervention is not None
            and intervention.active
            and layer == intervention.layer
        )

    def _native_head_crsa_usage_dtype(self) -> torch.dtype:
        return (
            torch.float32
            if self.pager.compute_dtype in {torch.float16, torch.bfloat16}
            else self.pager.compute_dtype
        )

    def _graft_snapshot_identity(self) -> dict[str, Any]:
        if self.graft is None:
            return {"kind": "none", "layer": self.graft_layer}
        graft = self.graft
        state_dict = getattr(graft, "state_dict", None)
        if callable(state_dict) and state_dict():
            raise Qwen38SnapshotError(
                "continuation snapshots support only parameter-free grafts"
            )
        required = (
            "mode",
            "alpha",
            "heads",
            "max_history",
            "shuffle_seed",
            "spec",
        )
        if any(not hasattr(graft, name) for name in required):
            raise Qwen38SnapshotError(
                "graft lacks the complete serialisable identity contract"
            )
        spec = getattr(graft, "spec")
        if not is_dataclass(spec):
            raise Qwen38SnapshotError("graft AttentionSpec is not a dataclass")
        alpha = float(getattr(graft, "alpha"))
        if not math.isfinite(alpha):
            raise Qwen38SnapshotError("graft alpha must be finite")
        rms_eps = getattr(graft, "rms_eps", None)
        if rms_eps is not None and (
            not math.isfinite(float(rms_eps)) or float(rms_eps) <= 0.0
        ):
            raise Qwen38SnapshotError("graft rms_eps must be finite and positive")
        return {
            "kind": f"{type(graft).__module__}.{type(graft).__qualname__}",
            "layer": self.graft_layer,
            "mode": str(getattr(graft, "mode")),
            "alpha": alpha,
            "heads": int(getattr(graft, "heads")),
            "max_history": int(getattr(graft, "max_history")),
            "shuffle_seed": int(getattr(graft, "shuffle_seed")),
            "attention_spec": asdict(spec),
            "rms_eps": None if rms_eps is None else float(rms_eps),
            "policy": str(getattr(graft, "policy", "unreported")),
        }

    def _mlp_sparse_snapshot_identity(
        self, *, transport_neutral: bool = False
    ) -> dict[str, Any]:
        executor = self.mlp_sparse_executor
        if executor is None:
            return {"kind": "none"}
        identity = executor.snapshot_identity(transport_neutral=transport_neutral)
        if not isinstance(identity, dict):
            raise Qwen38SnapshotError("sparse MLP identity is not a mapping")
        try:
            canonical = json.loads(
                json.dumps(
                    identity,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError) as exc:
            raise Qwen38SnapshotError(
                "sparse MLP identity is not canonical JSON"
            ) from exc
        return {"kind": "pilot-sparse", "identity": canonical}

    def _snapshot_identity(self, *, transport_neutral: bool = False) -> dict[str, Any]:
        source = self.pager.source
        source.inventory()
        metrics = source.metrics()
        fingerprint = metrics.get("inventory_source_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise Qwen38SnapshotError(
                "tensor source has no verified inventory fingerprint"
            )
        source_kind = f"{type(source).__module__}.{type(source).__qualname__}"
        repo_id = metrics.get("repo_id", getattr(source, "repo_id", source_kind))
        revision = metrics.get("revision", getattr(source, "revision", "fixture"))
        if not isinstance(repo_id, str) or not repo_id:
            raise Qwen38SnapshotError("tensor source repo identity is invalid")
        if not isinstance(revision, str) or not revision:
            raise Qwen38SnapshotError("tensor source revision identity is invalid")
        config = asdict(self.config)
        runtime_sources = runtime_source_manifest(
            include_transport=not transport_neutral
        )
        dependencies = runtime_dependency_versions()
        q4_bank = getattr(self.pager, "q4_bank", None)
        q4_identity = None if q4_bank is None else dict(q4_bank.identity)
        math_execution = {
            "device": str(self.pager.device),
            "compute_dtype": str(self.pager.compute_dtype).removeprefix("torch."),
            "max_batch_size": self.max_batch_size,
            "max_seq_len": self.max_seq_len,
            "max_position_embeddings": self.config.max_position_embeddings,
            "state_policy": "native-kv+deltanet-transactional/v1",
            "packed_continuation_gemm": self.packed_continuation_gemm,
            "quantized_weight_plane": q4_identity,
        }
        transport_execution = {
            "source_kind": source_kind,
            "source_transport_policy": str(
                metrics.get("transport_policy", "unreported")
            ),
            "source_transport_connection_limit": int(
                metrics.get("transport_connection_limit", 0)
            ),
            "pager_max_resident_bytes": self.pager.max_resident_bytes,
            "pager_weight_cache_policy": self.pager.WEIGHT_CACHE_POLICY,
        }
        execution = (
            {**math_execution, "transport_scope": "neutral/v1"}
            if transport_neutral
            else {**math_execution, **transport_execution}
        )
        return {
            "runtime": {
                "schema": "immer.streamed-qwen3.8/native-stateful-v1",
                "source_sha256": self._snapshot_digest(runtime_sources),
                "sources": runtime_sources,
                "dependency_sha256": self._snapshot_digest(dependencies),
                "dependencies": dependencies,
            },
            "config": config,
            "config_sha256": self._snapshot_digest(config),
            "source": {
                "repo_id": repo_id,
                "revision": revision,
                "inventory_fingerprint": fingerprint,
            },
            "execution": execution,
            "graft": self._graft_snapshot_identity(),
            "mlp_sparse_executor": self._mlp_sparse_snapshot_identity(
                transport_neutral=transport_neutral
            ),
            "native_head_crsa": self._native_head_crsa_snapshot_identity(),
        }

    def _snapshot_model_state(
        self,
    ) -> tuple[dict[str, Any], dict[str, SnapshotTensor]]:
        if not 0 <= self._next_position <= self.max_seq_len:
            raise Qwen38SnapshotError("model cursor exceeds its context bound")
        if self._state_poisoned and self._next_position:
            raise Qwen38SnapshotError("poisoned model has a non-zero cursor")
        if self._next_position == 0:
            if self._state_batch_size is not None or any(
                state is not None for state in self._layer_states
            ):
                raise Qwen38SnapshotError("zero-cursor model retains layer state")
            if self._graft_history is not None:
                raise Qwen38SnapshotError("zero-cursor model retains graft history")
        elif (
            self._state_batch_size is None
            or not 1 <= self._state_batch_size <= self.max_batch_size
            or any(state is None for state in self._layer_states)
        ):
            raise Qwen38SnapshotError("active model has incomplete batch/layer state")

        tensors: dict[str, SnapshotTensor] = {}
        layers: list[dict[str, Any]] = []
        batch = self._state_batch_size
        for layer, state in enumerate(self._layer_states):
            if state is None:
                continue
            prefix = f"state.layer_{layer:03d}"
            if self.config.is_full_attention(layer):
                if not isinstance(state, AttentionState):
                    raise Qwen38SnapshotError(
                        f"full-attention layer {layer} has the wrong state type"
                    )
                expected = (
                    batch,
                    self.config.n_kv_heads,
                    self._next_position,
                    self.config.head_dim,
                )
                if tuple(state.key.shape) != expected:
                    raise Qwen38SnapshotError(
                        f"full-attention layer {layer} state shape is inconsistent"
                    )
                if (
                    state.key.dtype != self.pager.compute_dtype
                    or not self._on_pager_device(state.key)
                ):
                    raise Qwen38SnapshotError(
                        f"full-attention layer {layer} state dtype/device is inconsistent"
                    )
                key_name = f"{prefix}.key"
                value_name = f"{prefix}.value"
                tensors[key_name] = SnapshotTensor(state.key)
                tensors[value_name] = SnapshotTensor(state.value)
                row = {
                    "kind": "full_attention",
                    "layer": layer,
                    "key": key_name,
                    "value": value_name,
                }
                usage_required = self._native_head_crsa_usage_required(layer)
                if usage_required != (state.crsa_log_usage is not None):
                    raise Qwen38SnapshotError(
                        f"full-attention layer {layer} CRSA usage is inconsistent "
                        "with the native intervention"
                    )
                if state.crsa_log_usage is not None:
                    usage = state.crsa_log_usage
                    if tuple(usage.shape) != (batch, 4, self._next_position):
                        raise Qwen38SnapshotError(
                            f"full-attention layer {layer} CRSA usage shape is "
                            "inconsistent"
                        )
                    if (
                        usage.dtype != self._native_head_crsa_usage_dtype()
                        or not self._on_pager_device(usage)
                    ):
                        raise Qwen38SnapshotError(
                            f"full-attention layer {layer} CRSA usage dtype/device "
                            "is inconsistent"
                        )
                    if bool((torch.isnan(usage) | torch.isposinf(usage)).any().item()):
                        raise Qwen38SnapshotError(
                            f"full-attention layer {layer} CRSA usage may contain "
                            "only finite values or -inf"
                        )
                    usage_name = f"{prefix}.crsa_log_usage"
                    tensors[usage_name] = SnapshotTensor(
                        usage, finite_policy="finite_or_neg_inf"
                    )
                    row["crsa_log_usage"] = usage_name
                layers.append(row)
            else:
                if not isinstance(state, DeltaNetState):
                    raise Qwen38SnapshotError(
                        f"linear-attention layer {layer} has the wrong state type"
                    )
                key_features = (
                    self.config.linear_num_key_heads * self.config.linear_key_head_dim
                )
                value_features = (
                    self.config.linear_num_value_heads
                    * self.config.linear_value_head_dim
                )
                conv_shape = (
                    batch,
                    2 * key_features + value_features,
                    self.config.linear_conv_kernel_dim,
                )
                recurrent_shape = (
                    batch,
                    self.config.linear_num_value_heads,
                    self.config.linear_key_head_dim,
                    self.config.linear_value_head_dim,
                )
                if (
                    tuple(state.conv.shape) != conv_shape
                    or tuple(state.recurrent.shape) != recurrent_shape
                ):
                    raise Qwen38SnapshotError(
                        f"linear-attention layer {layer} state shape is inconsistent"
                    )
                if (
                    state.conv.dtype != self.pager.compute_dtype
                    or not self._on_pager_device(state.conv)
                    or state.recurrent.dtype != torch.float32
                    or not self._on_pager_device(state.recurrent)
                ):
                    raise Qwen38SnapshotError(
                        f"linear-attention layer {layer} state dtype/device is inconsistent"
                    )
                conv_name = f"{prefix}.conv"
                recurrent_name = f"{prefix}.recurrent"
                tensors[conv_name] = SnapshotTensor(state.conv)
                tensors[recurrent_name] = SnapshotTensor(state.recurrent)
                layers.append(
                    {
                        "kind": "linear_attention",
                        "layer": layer,
                        "conv": conv_name,
                        "recurrent": recurrent_name,
                    }
                )

        graft_history_name = None
        if self._graft_history is not None:
            expected = (
                batch,
                self._next_position,
                self.config.dim,
            )
            if tuple(self._graft_history.shape) != expected:
                raise Qwen38SnapshotError("graft history shape/cursor is inconsistent")
            if (
                self._graft_history.dtype != self.pager.compute_dtype
                or not self._on_pager_device(self._graft_history)
            ):
                raise Qwen38SnapshotError("graft history dtype/device is inconsistent")
            graft_history_name = "state.graft_history"
            tensors[graft_history_name] = SnapshotTensor(self._graft_history)
        graft_identity = self._graft_snapshot_identity()
        graft_active = (
            graft_identity.get("kind") != "none"
            and graft_identity.get("layer") is not None
            and graft_identity.get("mode") != "off"
            and float(graft_identity.get("alpha", 0.0)) != 0.0
        )
        if self._next_position and graft_active != (graft_history_name is not None):
            raise Qwen38SnapshotError(
                "active graft and graft-history presence are inconsistent"
            )
        if self._state_poisoned and (layers or tensors):
            raise Qwen38SnapshotError("poisoned model retains continuation tensors")
        return (
            {
                "next_position": self._next_position,
                "state_poisoned": self._state_poisoned,
                "state_batch_size": self._state_batch_size,
                "max_batch_size": self.max_batch_size,
                "max_seq_len": self.max_seq_len,
                "max_position_embeddings": self.config.max_position_embeddings,
                "graft_history": graft_history_name,
                "attention_layers": layers,
            },
            tensors,
        )

    @staticmethod
    def _snapshot_limits(max_bytes: int, max_tensors: int) -> SnapshotLimits:
        return SnapshotLimits(max_bytes=max_bytes, max_tensors=max_tensors)

    def save_state(
        self,
        path: str | os.PathLike[str],
        *,
        max_bytes: int = 2 * 1024**3,
        max_tensors: int = 2048,
        transport_neutral: bool = False,
    ) -> dict[str, Any]:
        """Atomically save every native Qwen continuation tensor."""

        if not isinstance(transport_neutral, bool):
            raise TypeError("transport_neutral must be a boolean")
        limits = self._snapshot_limits(max_bytes, max_tensors)
        state, tensors = self._snapshot_model_state()
        result = write_qwen38_snapshot(
            path,
            identity=self._snapshot_identity(transport_neutral=transport_neutral),
            state=state,
            tensors=tensors,
            limits=limits,
        )
        return {
            **result,
            "next_position": self._next_position,
            "state_poisoned": self._state_poisoned,
            "transport_neutral": transport_neutral,
        }

    @staticmethod
    def _snapshot_state_int(value: Any, name: str, *, maximum: int) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value > maximum
        ):
            raise Qwen38SnapshotError(f"snapshot {name} is outside its bound")
        return value

    def load_state(
        self,
        path: str | os.PathLike[str],
        *,
        max_bytes: int = 2 * 1024**3,
        max_tensors: int = 2048,
        max_restore_peak_bytes: int = 4 * 1024**3,
        transport_neutral: bool = False,
    ) -> dict[str, Any]:
        """Transactionally restore a bounded native Qwen continuation."""

        if not isinstance(transport_neutral, bool):
            raise TypeError("transport_neutral must be a boolean")
        limits = self._snapshot_limits(max_bytes, max_tensors)
        loaded = read_qwen38_snapshot(
            path,
            expected_identity=self._snapshot_identity(
                transport_neutral=transport_neutral
            ),
            limits=limits,
            resident_bytes=self.state_bytes,
            max_restore_peak_bytes=max_restore_peak_bytes,
        )
        state = loaded.state
        expected_state_keys = {
            "attention_layers",
            "graft_history",
            "max_batch_size",
            "max_position_embeddings",
            "max_seq_len",
            "next_position",
            "state_batch_size",
            "state_poisoned",
        }
        if set(state) != expected_state_keys:
            raise Qwen38SnapshotError("snapshot model-state schema is invalid")
        next_position = self._snapshot_state_int(
            state.get("next_position"),
            "next_position",
            maximum=self.max_seq_len,
        )
        poisoned = state.get("state_poisoned")
        if not isinstance(poisoned, bool):
            raise Qwen38SnapshotError("snapshot poison latch must be boolean")
        if poisoned and next_position:
            raise Qwen38SnapshotError("poisoned snapshot has a non-zero cursor")
        if any(
            state.get(key) != value
            for key, value in {
                "max_batch_size": self.max_batch_size,
                "max_seq_len": self.max_seq_len,
                "max_position_embeddings": self.config.max_position_embeddings,
            }.items()
        ):
            raise Qwen38SnapshotError("snapshot model bounds do not match runtime")
        raw_batch = state.get("state_batch_size")
        if next_position:
            batch = self._snapshot_state_int(
                raw_batch, "state_batch_size", maximum=self.max_batch_size
            )
            if batch == 0:
                raise Qwen38SnapshotError("active snapshot batch size must be positive")
        else:
            if raw_batch is not None:
                raise Qwen38SnapshotError(
                    "zero-cursor snapshot retains batch ownership"
                )
            batch = None

        raw_layers = state.get("attention_layers")
        if not isinstance(raw_layers, list) or len(raw_layers) > self.config.n_layers:
            raise Qwen38SnapshotError("snapshot attention-layer table is invalid")
        by_layer: dict[int, dict[str, Any]] = {}
        referenced: set[str] = set()
        for row in raw_layers:
            if not isinstance(row, dict):
                raise Qwen38SnapshotError("snapshot attention-layer row is invalid")
            layer = row.get("layer")
            if (
                isinstance(layer, bool)
                or not isinstance(layer, int)
                or not 0 <= layer < self.config.n_layers
                or layer in by_layer
            ):
                raise Qwen38SnapshotError(
                    "snapshot layer index is invalid or duplicate"
                )
            expected_kind = (
                "full_attention"
                if self.config.is_full_attention(layer)
                else "linear_attention"
            )
            usage_required = (
                expected_kind == "full_attention"
                and self._native_head_crsa_usage_required(layer)
            )
            if expected_kind == "full_attention":
                expected_keys = {"kind", "layer", "key", "value"}
                if usage_required:
                    expected_keys.add("crsa_log_usage")
            else:
                expected_keys = {"kind", "layer", "conv", "recurrent"}
            if set(row) != expected_keys or row.get("kind") != expected_kind:
                raise Qwen38SnapshotError("snapshot layer-state role is invalid")
            if expected_kind == "full_attention":
                names = (row.get("key"), row.get("value"))
                expected_names = (
                    f"state.layer_{layer:03d}.key",
                    f"state.layer_{layer:03d}.value",
                )
                if usage_required:
                    names = (*names, row.get("crsa_log_usage"))
                    expected_names = (
                        *expected_names,
                        f"state.layer_{layer:03d}.crsa_log_usage",
                    )
            else:
                names = (row.get("conv"), row.get("recurrent"))
                expected_names = (
                    f"state.layer_{layer:03d}.conv",
                    f"state.layer_{layer:03d}.recurrent",
                )
            if names != expected_names:
                raise Qwen38SnapshotError("snapshot tensor role name is invalid")
            referenced.update(expected_names)
            by_layer[layer] = row
        if next_position and len(by_layer) != self.config.n_layers:
            raise Qwen38SnapshotError(
                "active snapshot omits one or more decoder-layer states"
            )
        if not next_position and by_layer:
            raise Qwen38SnapshotError("zero-cursor snapshot retains layer state")

        history_name = state.get("graft_history")
        if history_name is not None:
            if history_name != "state.graft_history":
                raise Qwen38SnapshotError("snapshot graft-history role is invalid")
            referenced.add(history_name)
        if referenced != set(loaded.tensors):
            raise Qwen38SnapshotError(
                "snapshot contains missing or unreferenced tensor payloads"
            )
        descriptors = {
            row.get("name"): row
            for row in loaded.manifest["body"].get("tensors", [])
            if isinstance(row, dict)
        }
        for name, tensor in loaded.tensors.items():
            descriptor = descriptors.get(name)
            is_crsa_usage = name.endswith(".crsa_log_usage")
            expected_policy = "finite_or_neg_inf" if is_crsa_usage else "finite"
            if (
                not isinstance(descriptor, dict)
                or descriptor.get("finite_policy") != expected_policy
            ):
                raise Qwen38SnapshotError("snapshot tensor finite policy is invalid")
            expected_dtype = (
                torch.float32
                if name.endswith(".recurrent")
                else (
                    self._native_head_crsa_usage_dtype()
                    if is_crsa_usage
                    else self.pager.compute_dtype
                )
            )
            if tensor.dtype != expected_dtype:
                raise Qwen38SnapshotError(
                    f"snapshot tensor {name!r} has the wrong state dtype"
                )

        graft_identity = self._graft_snapshot_identity()
        graft_active = (
            graft_identity.get("kind") != "none"
            and graft_identity.get("layer") is not None
            and graft_identity.get("mode") != "off"
            and float(graft_identity.get("alpha", 0.0)) != 0.0
        )
        if next_position and graft_active != (history_name is not None):
            raise Qwen38SnapshotError(
                "active graft and graft-history presence are inconsistent"
            )
        if poisoned and (by_layer or history_name is not None):
            raise Qwen38SnapshotError("poisoned snapshot retains continuation state")

        new_states: list[LayerState | None] = [
            None for _ in range(self.config.n_layers)
        ]
        history_device: torch.Tensor | None = None
        try:
            for layer in sorted(by_layer):
                row = by_layer[layer]
                if self.config.is_full_attention(layer):
                    key = loaded.tensors.pop(row["key"])
                    value = loaded.tensors.pop(row["value"])
                    usage_name = row.get("crsa_log_usage")
                    usage = (
                        None if usage_name is None else loaded.tensors.pop(usage_name)
                    )
                    expected = (
                        batch,
                        self.config.n_kv_heads,
                        next_position,
                        self.config.head_dim,
                    )
                    if tuple(key.shape) != expected or tuple(value.shape) != expected:
                        raise Qwen38SnapshotError(
                            f"full-attention layer {layer} tensor shape is invalid"
                        )
                    key_device = key.to(device=self.pager.device).detach()
                    del key
                    value_device = value.to(device=self.pager.device).detach()
                    del value
                    usage_device = None
                    if usage is not None:
                        if tuple(usage.shape) != (batch, 4, next_position):
                            raise Qwen38SnapshotError(
                                f"full-attention layer {layer} CRSA usage shape "
                                "is invalid"
                            )
                        usage_device = usage.to(device=self.pager.device).detach()
                        del usage
                    new_states[layer] = AttentionState(
                        key=key_device,
                        value=value_device,
                        crsa_log_usage=usage_device,
                    )
                else:
                    conv = loaded.tensors.pop(row["conv"])
                    recurrent = loaded.tensors.pop(row["recurrent"])
                    key_features = (
                        self.config.linear_num_key_heads
                        * self.config.linear_key_head_dim
                    )
                    value_features = (
                        self.config.linear_num_value_heads
                        * self.config.linear_value_head_dim
                    )
                    if tuple(conv.shape) != (
                        batch,
                        2 * key_features + value_features,
                        self.config.linear_conv_kernel_dim,
                    ) or tuple(recurrent.shape) != (
                        batch,
                        self.config.linear_num_value_heads,
                        self.config.linear_key_head_dim,
                        self.config.linear_value_head_dim,
                    ):
                        raise Qwen38SnapshotError(
                            f"linear-attention layer {layer} tensor shape is invalid"
                        )
                    conv_device = conv.to(device=self.pager.device).detach()
                    del conv
                    recurrent_device = recurrent.to(device=self.pager.device).detach()
                    del recurrent
                    new_states[layer] = DeltaNetState(
                        conv=conv_device,
                        recurrent=recurrent_device,
                    )
            if history_name is not None:
                history = loaded.tensors.pop(history_name)
                if tuple(history.shape) != (batch, next_position, self.config.dim):
                    raise Qwen38SnapshotError(
                        "snapshot graft history shape/cursor is inconsistent"
                    )
                history_device = history.to(device=self.pager.device).detach()
                del history
            if loaded.tensors:
                raise Qwen38SnapshotError(
                    "snapshot tensor ownership transfer is incomplete"
                )
        except Qwen38SnapshotError:
            raise
        except Exception as exc:
            raise Qwen38SnapshotError(
                "snapshot layer state is structurally inconsistent"
            ) from exc

        self._layer_states = new_states
        self._next_position = next_position
        self._state_batch_size = batch
        self._state_poisoned = poisoned
        self._graft_history = history_device
        self._pending_block_stage = None
        return {
            **loaded.summary,
            "next_position": next_position,
            "state_poisoned": poisoned,
            "transport_neutral": transport_neutral,
        }

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

    def _observe_layer_boundary(
        self,
        layer: int,
        stage: str,
        value: torch.Tensor,
    ) -> None:
        observer = self.layer_boundary_observer
        if (
            observer is None
            or stage not in self.layer_boundary_stages
            or (
                self.layer_boundary_layers is not None
                and layer not in self.layer_boundary_layers
            )
        ):
            return
        if stage not in LAYER_BOUNDARY_STAGES:
            raise Qwen38RuntimeError("layer boundary stage is not registered")
        observer(layer, stage, value.detach().clone())

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
        native_head_crsa_observer: Callable[[NativeHeadCrsaEvidence], None]
        | None = None,
        native_head_crsa_tokenwise_usage: bool = False,
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
            native_head_crsa = (
                self.native_head_crsa
                if self.native_head_crsa is not None
                and layer == self.native_head_crsa.layer
                else None
            )
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
                native_head_crsa=native_head_crsa,
                native_head_crsa_observer=(
                    native_head_crsa_observer if native_head_crsa is not None else None
                ),
                native_prefix_sinkhorn_operator_observer=(
                    self.native_prefix_sinkhorn_operator_observer
                    if native_head_crsa is not None
                    else None
                ),
                native_head_crsa_tokenwise_usage=native_head_crsa_tokenwise_usage,
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
            probe = (
                None
                if self.delta_probe is None
                else lambda row: self.delta_probe(layer, row)
            )
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
                probe=probe,
            )
        finally:
            del projected_qkv, projected_z, projected_b, projected_a
            del conv_weight, a_log, dt_bias, norm_weight
        return self.pager.linear(mixed, f"{base}.out_proj"), next_state

    def _sparse_mlp_allowed(self, layer: int, row_count: int) -> bool:
        executor = self.mlp_sparse_executor
        if (
            executor is None
            or row_count < 1
            or row_count > self.MAX_CONTINUATION_BLOCK_WIDTH
            or not executor.supports_layer(layer)
        ):
            self.mlp_sparse_last_decision = None
            return False
        if self.layer_boundary_observer is None:
            boundary_allows = True
        elif (
            self.layer_boundary_layers is not None
            and layer not in self.layer_boundary_layers
        ):
            boundary_allows = True
        else:
            boundary_allows = not any(
                stage in self.layer_boundary_stages
                for stage in ("mlp.gate", "mlp.up", "mlp.activated")
            )
        if not boundary_allows:
            self.mlp_sparse_last_decision = None
            return False
        decide = getattr(executor, "decision", None)
        if not callable(decide):
            self.mlp_sparse_last_decision = None
            return True
        decision = decide(layer=layer, row_count=row_count)
        use_sparse = getattr(decision, "use_sparse", None)
        if not isinstance(use_sparse, bool):
            raise Qwen38RuntimeError("fast-MLP decision returned an invalid action")
        self.mlp_sparse_last_decision = decision
        controller = getattr(executor, "online_controller", None)
        if use_sparse and controller is not None:
            output_calibrated = getattr(controller, "output_calibrated", None)
            if not callable(output_calibrated) or not bool(
                output_calibrated(layer=layer)
            ):
                return False
        return use_sparse

    def _observe_exact_mlp(
        self,
        *,
        layer: int,
        gate: Any,
        up: Any,
        activated: Any,
        output: Any,
        down_weight: Any | None = None,
    ) -> None:
        self.mlp_sparse_last_observation = None
        executor = self.mlp_sparse_executor
        observe = None if executor is None else getattr(executor, "observe_full", None)
        if not callable(observe):
            return
        self.mlp_sparse_last_observation = observe(
            layer=layer,
            gate=gate,
            up=up,
            activated=activated,
            output=output,
            down_weight=down_weight,
        )

    def _mlp(self, hidden: torch.Tensor, *, layer: int) -> torch.Tensor:
        row_count = hidden.numel() // hidden.shape[-1]
        if row_count == 1 and self._sparse_mlp_allowed(layer, row_count):
            try:
                output, trace = self.mlp_sparse_executor.execute(hidden, layer=layer)
            except Exception as exc:
                if getattr(type(exc), "exact_mlp_fallback", False) is not True:
                    raise
                self.mlp_sparse_last_trace = None
            else:
                self.mlp_sparse_last_trace = trace
                output = output.to(device=hidden.device, dtype=hidden.dtype)
                self._observe_layer_boundary(layer, "mlp.output", output)
                return output
        self.mlp_sparse_last_trace = None
        base = f"model.language_model.layers.{layer}.mlp"
        gate = self.pager.linear(hidden, f"{base}.gate_proj")
        self._observe_layer_boundary(layer, "mlp.gate", gate)
        up = self.pager.linear(hidden, f"{base}.up_proj")
        self._observe_layer_boundary(layer, "mlp.up", up)
        activated = swiglu(gate, up)
        self._observe_layer_boundary(layer, "mlp.activated", activated)

        observe_full = (
            None
            if self.mlp_sparse_executor is None
            else getattr(self.mlp_sparse_executor, "observe_full", None)
        )
        observe_down = None
        if callable(observe_full):
            def observe_down(weight: Any, result: Any) -> None:
                self._observe_exact_mlp(
                    layer=layer,
                    gate=gate,
                    up=up,
                    activated=activated,
                    output=result,
                    down_weight=weight,
                )

        output = self.pager.linear(
            activated,
            f"{base}.down_proj",
            weight_observer=observe_down,
        )
        self._observe_layer_boundary(layer, "mlp.output", output)
        return output

    def _linear_token_rows(
        self,
        hidden: tuple[torch.Tensor, ...],
        name: str,
        *,
        weight_observer: Callable[[Any, tuple[Any, ...]], None] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Apply one matrix once to one-to-sixteen independent token rows."""

        if len(hidden) == 1:
            observed = None
            if weight_observer is not None:
                def observed(weight: Any, result: Any) -> None:
                    weight_observer(weight, (result,))

            return (
                self.pager.linear(
                    hidden[0],
                    name,
                    weight_observer=observed,
                ),
            )
        return self.pager.linear_many(
            hidden,
            name,
            packed=self.packed_continuation_gemm,
            weight_observer=weight_observer,
        )

    def _norm_token_rows(
        self,
        hidden: tuple[torch.Tensor, ...],
        name: str,
    ) -> tuple[torch.Tensor, ...]:
        """Normalize independent token rows separately from one weight read."""

        if not 1 <= len(hidden) <= self.MAX_CONTINUATION_BLOCK_WIDTH:
            raise Qwen38RuntimeError("continuation token-row width is invalid")
        weight = self._control(name)
        try:
            return tuple(
                rms_norm(row, weight, eps=self.config.rms_norm_eps) for row in hidden
            )
        finally:
            del weight

    def _norm_k2_pair(
        self,
        hidden: tuple[torch.Tensor, torch.Tensor],
        name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Preserve the public K=2 implementation seam and exact arithmetic."""

        rows = self._norm_token_rows(hidden, name)
        return rows[0], rows[1]

    def _full_attention_token_rows(
        self,
        hidden: tuple[torch.Tensor, ...],
        *,
        layer: int,
        state: AttentionState | None,
        start_pos: int,
        native_head_crsa_observer: Callable[[NativeHeadCrsaEvidence], None],
        native_prefix_sinkhorn_operator_observer: (
            NativePrefixSinkhornOperatorObserver | None
        ) = None,
    ) -> tuple[
        tuple[torch.Tensor, ...],
        AttentionState,
        _LayerPrefixTrace,
    ]:
        """Run exact one-token attention recurrence with weight-once projections."""

        base = f"model.language_model.layers.{layer}.self_attn"
        projected_query_gate = self._linear_token_rows(hidden, f"{base}.q_proj")
        projected_key = self._linear_token_rows(hidden, f"{base}.k_proj")
        projected_value = self._linear_token_rows(hidden, f"{base}.v_proj")
        q_norm_weight = self._control(f"{base}.q_norm.weight")
        k_norm_weight = self._control(f"{base}.k_norm.weight")
        intervention = (
            self.native_head_crsa
            if self.native_head_crsa is not None
            and layer == self.native_head_crsa.layer
            else None
        )
        mixed_rows: list[torch.Tensor] = []
        prefix_usage: list[torch.Tensor | None] = []
        next_state = state
        try:
            for offset, row in enumerate(hidden):
                position = (
                    torch.arange(
                        start_pos + offset,
                        start_pos + offset + 1,
                        device=row.device,
                        dtype=torch.long,
                    )
                    .unsqueeze(0)
                    .expand(row.shape[0], -1)
                )
                mixed, next_state = full_attention_core(
                    projected_query_gate[offset],
                    projected_key[offset],
                    projected_value[offset],
                    q_norm_weight=q_norm_weight,
                    k_norm_weight=k_norm_weight,
                    num_attention_heads=self.config.n_heads,
                    num_key_value_heads=self.config.n_kv_heads,
                    head_dim=self.config.head_dim,
                    position_ids=position,
                    state=next_state,
                    attention_mask=None,
                    native_head_crsa=intervention,
                    native_head_crsa_observer=(
                        native_head_crsa_observer if intervention is not None else None
                    ),
                    native_prefix_sinkhorn_operator_observer=(
                        native_prefix_sinkhorn_operator_observer
                        if intervention is not None
                        else None
                    ),
                    rope_theta=self.config.rope_theta,
                    rotary_dim=self.config.rotary_dim,
                    mrope_section=self.config.mrope_section,
                    mrope_interleaved=self.config.mrope_interleaved,
                    rms_norm_eps=self.config.rms_norm_eps,
                )
                mixed_rows.append(mixed)
                prefix_usage.append(next_state.crsa_log_usage)
        finally:
            del projected_query_gate, projected_key, projected_value
            del q_norm_weight, k_norm_weight
        if not isinstance(next_state, AttentionState):  # pragma: no cover - kernel.
            raise Qwen38RuntimeError("continuation full attention returned no state")
        projected = self._linear_token_rows(tuple(mixed_rows), f"{base}.o_proj")
        return projected, next_state, _LayerPrefixTrace(
            attention_log_usage=tuple(prefix_usage)
        )

    def _full_attention_k2_pair(
        self,
        hidden: tuple[torch.Tensor, torch.Tensor],
        *,
        layer: int,
        state: AttentionState | None,
        start_pos: int,
        native_head_crsa_observer: Callable[[NativeHeadCrsaEvidence], None],
        native_prefix_sinkhorn_operator_observer: (
            NativePrefixSinkhornOperatorObserver | None
        ) = None,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        AttentionState,
        _LayerPrefixTrace,
    ]:
        """Preserve the K=2 implementation seam over the generic exact path."""

        rows, next_state, trace = self._full_attention_token_rows(
            hidden,
            layer=layer,
            state=state,
            start_pos=start_pos,
            native_head_crsa_observer=native_head_crsa_observer,
            native_prefix_sinkhorn_operator_observer=(
                native_prefix_sinkhorn_operator_observer
            ),
        )
        return (rows[0], rows[1]), next_state, trace

    def _linear_attention_token_rows(
        self,
        hidden: tuple[torch.Tensor, ...],
        *,
        layer: int,
        state: DeltaNetState | None,
    ) -> tuple[
        tuple[torch.Tensor, ...],
        DeltaNetState,
        _LayerPrefixTrace,
    ]:
        """Run exact one-token DeltaNet recurrence from one weight pass."""

        base = f"model.language_model.layers.{layer}.linear_attn"
        mask = torch.ones(
            (1, 1, 1),
            dtype=hidden[0].dtype,
            device=hidden[0].device,
        )
        active = tuple(row * mask for row in hidden)
        projected_qkv = self._linear_token_rows(active, f"{base}.in_proj_qkv")
        projected_z = self._linear_token_rows(active, f"{base}.in_proj_z")
        projected_b = self._linear_token_rows(active, f"{base}.in_proj_b")
        projected_a = self._linear_token_rows(active, f"{base}.in_proj_a")
        conv_weight = self._control(f"{base}.conv1d.weight", dtype=hidden[0].dtype)
        a_log = self._control(f"{base}.A_log")
        dt_bias = self._control(f"{base}.dt_bias")
        norm_weight = self._control(f"{base}.norm.weight", dtype=hidden[0].dtype)
        mixed_rows: list[torch.Tensor] = []
        updates: list[_DeltaNetPrefixUpdate] = []
        conv_inputs: list[torch.Tensor] = []
        next_state = state
        try:
            for offset in range(len(hidden)):
                mixed, next_state = gated_delta_net_core(
                    projected_qkv[offset],
                    projected_z[offset],
                    projected_b[offset],
                    projected_a[offset],
                    conv1d_weight=conv_weight,
                    A_log=a_log,
                    dt_bias=dt_bias,
                    norm_weight=norm_weight,
                    num_key_heads=self.config.linear_num_key_heads,
                    num_value_heads=self.config.linear_num_value_heads,
                    key_head_dim=self.config.linear_key_head_dim,
                    value_head_dim=self.config.linear_value_head_dim,
                    state=next_state,
                    rms_norm_eps=self.config.rms_norm_eps,
                    probe=None,
                    state_update_observer=lambda key, value, beta, log_decay: (
                        updates.append(
                            _DeltaNetPrefixUpdate(
                                key=key,
                                value=value,
                                beta=beta,
                                log_decay=log_decay,
                            )
                        )
                    ),
                )
                mixed_rows.append(mixed)
                if not isinstance(next_state, DeltaNetState):
                    raise Qwen38RuntimeError(
                        "continuation DeltaNet returned no state"
                    )
                conv_inputs.append(next_state.conv[..., -1:].detach().clone())
        finally:
            del projected_qkv, projected_z, projected_b, projected_a
            del conv_weight, a_log, dt_bias, norm_weight
        if not isinstance(next_state, DeltaNetState):  # pragma: no cover - kernel.
            raise Qwen38RuntimeError("continuation DeltaNet returned no state")
        projected = self._linear_token_rows(tuple(mixed_rows), f"{base}.out_proj")
        kernel_size = int(next_state.conv.shape[-1])
        return projected, next_state, _LayerPrefixTrace(
            delta_updates=tuple(updates),
            delta_conv_prefix=tuple(conv_inputs[:-kernel_size]),
        )

    def _linear_attention_k2_pair(
        self,
        hidden: tuple[torch.Tensor, torch.Tensor],
        *,
        layer: int,
        state: DeltaNetState | None,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        DeltaNetState,
        _LayerPrefixTrace,
    ]:
        """Preserve the K=2 implementation seam over the generic exact path."""

        rows, next_state, trace = self._linear_attention_token_rows(
            hidden,
            layer=layer,
            state=state,
        )
        return (rows[0], rows[1]), next_state, trace

    def _mlp_token_rows(
        self,
        hidden: tuple[torch.Tensor, ...],
        *,
        layer: int,
    ) -> tuple[torch.Tensor, ...]:
        row_count = sum(row.numel() // row.shape[-1] for row in hidden)
        if self._sparse_mlp_allowed(layer, row_count):
            try:
                outputs, trace = self.mlp_sparse_executor.execute_many(
                    hidden,
                    layer=layer,
                )
            except Exception as exc:
                if getattr(type(exc), "exact_mlp_fallback", False) is not True:
                    raise
                self.mlp_sparse_last_trace = None
            else:
                self.mlp_sparse_last_trace = trace
                return tuple(
                    output.to(device=row.device, dtype=row.dtype)
                    for output, row in zip(outputs, hidden, strict=True)
                )
        self.mlp_sparse_last_trace = None
        base = f"model.language_model.layers.{layer}.mlp"
        gate = self._linear_token_rows(hidden, f"{base}.gate_proj")
        up = self._linear_token_rows(hidden, f"{base}.up_proj")
        activated = tuple(
            swiglu(gate[index], up[index]) for index in range(len(hidden))
        )

        def observe_down(weight: Any, result: tuple[Any, ...]) -> None:
            self._observe_exact_mlp(
                layer=layer,
                gate=gate,
                up=up,
                activated=activated,
                output=result,
                down_weight=weight,
            )

        output = self._linear_token_rows(
            activated,
            f"{base}.down_proj",
            weight_observer=observe_down,
        )
        return output

    def _mlp_k2_pair(
        self,
        hidden: tuple[torch.Tensor, torch.Tensor],
        *,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = self._mlp_token_rows(hidden, layer=layer)
        return rows[0], rows[1]

    def _forward_layer_token_rows(
        self,
        hidden: tuple[torch.Tensor, ...],
        *,
        layer: int,
        state: LayerState | None,
        start_pos: int,
        native_head_crsa_observer: Callable[[NativeHeadCrsaEvidence], None],
        native_prefix_sinkhorn_operator_observer: (
            NativePrefixSinkhornOperatorObserver | None
        ) = None,
    ) -> tuple[
        tuple[torch.Tensor, ...],
        LayerState,
        _LayerPrefixTrace,
    ]:
        """Apply one layer tokenwise while reading every checkpoint tensor once."""

        prefix = f"model.language_model.layers.{layer}"
        residual = hidden
        mixed_input = self._norm_token_rows(
            hidden,
            f"{prefix}.input_layernorm.weight",
        )
        if self.config.is_full_attention(layer):
            if state is not None and not isinstance(state, AttentionState):
                raise Qwen38RuntimeError("full-attention layer received DeltaNet state")
            mixed, next_state, prefix_trace = self._full_attention_token_rows(
                mixed_input,
                layer=layer,
                state=state,
                start_pos=start_pos,
                native_head_crsa_observer=native_head_crsa_observer,
                native_prefix_sinkhorn_operator_observer=(
                    native_prefix_sinkhorn_operator_observer
                ),
            )
        else:
            if state is not None and not isinstance(state, DeltaNetState):
                raise Qwen38RuntimeError("linear-attention layer received KV state")
            mixed, next_state, prefix_trace = self._linear_attention_token_rows(
                mixed_input,
                layer=layer,
                state=state,
            )
        after_attention = tuple(
            residual[index] + mixed[index] for index in range(len(hidden))
        )
        mlp_input = self._norm_token_rows(
            after_attention,
            f"{prefix}.post_attention_layernorm.weight",
        )
        mlp = self._mlp_token_rows(mlp_input, layer=layer)
        return (
            tuple(
                after_attention[index] + mlp[index]
                for index in range(len(hidden))
            ),
            next_state,
            prefix_trace,
        )

    def _forward_layer_k2_pair(
        self,
        hidden: tuple[torch.Tensor, torch.Tensor],
        *,
        layer: int,
        state: LayerState | None,
        start_pos: int,
        native_head_crsa_observer: Callable[[NativeHeadCrsaEvidence], None],
        native_prefix_sinkhorn_operator_observer: (
            NativePrefixSinkhornOperatorObserver | None
        ) = None,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        LayerState,
        _LayerPrefixTrace,
    ]:
        """Preserve the original K=2 seam over generic weight-once helpers."""

        prefix = f"model.language_model.layers.{layer}"
        residual = hidden
        mixed_input = self._norm_k2_pair(
            hidden,
            f"{prefix}.input_layernorm.weight",
        )
        if self.config.is_full_attention(layer):
            if state is not None and not isinstance(state, AttentionState):
                raise Qwen38RuntimeError("full-attention layer received DeltaNet state")
            mixed, next_state, prefix_trace = self._full_attention_k2_pair(
                mixed_input,
                layer=layer,
                state=state,
                start_pos=start_pos,
                native_head_crsa_observer=native_head_crsa_observer,
                native_prefix_sinkhorn_operator_observer=(
                    native_prefix_sinkhorn_operator_observer
                ),
            )
        else:
            if state is not None and not isinstance(state, DeltaNetState):
                raise Qwen38RuntimeError("linear-attention layer received KV state")
            mixed, next_state, prefix_trace = self._linear_attention_k2_pair(
                mixed_input,
                layer=layer,
                state=state,
            )
        after_attention = (
            residual[0] + mixed[0],
            residual[1] + mixed[1],
        )
        mlp_input = self._norm_k2_pair(
            after_attention,
            f"{prefix}.post_attention_layernorm.weight",
        )
        mlp = self._mlp_k2_pair(mlp_input, layer=layer)
        return (
            (
                after_attention[0] + mlp[0],
                after_attention[1] + mlp[1],
            ),
            next_state,
            prefix_trace,
        )

    def _forward_layer(
        self,
        hidden: torch.Tensor,
        *,
        layer: int,
        token_mask: torch.Tensor,
        state: LayerState | None,
        start_pos: int,
        stateful: bool,
        native_head_crsa_observer: Callable[[NativeHeadCrsaEvidence], None]
        | None = None,
        native_head_crsa_tokenwise_usage: bool = False,
    ) -> tuple[torch.Tensor, LayerState | None]:
        """Apply one block and return its staged continuation state."""

        prefix = f"model.language_model.layers.{layer}"
        residual = hidden
        self._observe_layer_boundary(layer, "layer.input", residual)
        mixed_input = self._norm(hidden, f"{prefix}.input_layernorm.weight")
        self._observe_layer_boundary(layer, "attention.input", mixed_input)
        if self.config.is_full_attention(layer):
            if state is not None and not isinstance(state, AttentionState):
                raise Qwen38RuntimeError("full-attention layer received DeltaNet state")
            mixed, next_state = self._full_attention(
                mixed_input,
                layer=layer,
                token_mask=None if stateful else token_mask,
                state=state,
                start_pos=start_pos,
                native_head_crsa_observer=native_head_crsa_observer,
                native_head_crsa_tokenwise_usage=native_head_crsa_tokenwise_usage,
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
        self._observe_layer_boundary(layer, "attention.output", mixed)
        hidden = residual + mixed
        self._observe_layer_boundary(layer, "attention.residual", hidden)
        retained_state: LayerState | None = next_state
        if not stateful:
            retained_state = None
            del next_state

        residual = hidden
        mlp_input = self._norm(hidden, f"{prefix}.post_attention_layernorm.weight")
        self._observe_layer_boundary(layer, "mlp.input", mlp_input)
        hidden = residual + self._mlp(mlp_input, layer=layer)
        self._observe_layer_boundary(layer, "layer.output", hidden)
        return hidden, retained_state

    def forward_prefill_layer(
        self,
        hidden: Any,
        token_ids: Any,
        *,
        layer: int,
        token_mask: Any | None = None,
        _native_head_crsa_observer: Callable[[NativeHeadCrsaEvidence], None]
        | None = None,
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

        staged_native_evidence: list[NativeHeadCrsaEvidence] | None = (
            [] if _native_head_crsa_observer is None else None
        )
        native_observer = (
            staged_native_evidence.append
            if staged_native_evidence is not None
            else _native_head_crsa_observer
        )
        x, _state = self._forward_layer(
            x,
            layer=layer,
            token_mask=mask,
            state=None,
            start_pos=0,
            stateful=False,
            native_head_crsa_observer=native_observer,
        )
        if staged_native_evidence is not None:
            self._emit_native_head_crsa_evidence(staged_native_evidence)
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

    def _graft_active(self) -> bool:
        return (
            self.graft is not None
            and self._graft_mode() != "off"
            and float(getattr(self.graft, "alpha", 1.0)) != 0.0
        )

    def _continuation_block_runtime_identity(self) -> str:
        """Bind a staged block to every mutable math/runtime attachment."""

        native = (
            None if self.native_head_crsa is None else asdict(self.native_head_crsa)
        )
        source_metrics = self.pager.source.metrics()
        return self._snapshot_digest(
            {
                "config": asdict(self.config),
                "max_batch_size": self.max_batch_size,
                "max_seq_len": self.max_seq_len,
                "pager_id": id(self.pager),
                "source_id": id(self.pager.source),
                "source_fingerprint": source_metrics.get(
                    "inventory_source_fingerprint"
                ),
                "source_repo_id": source_metrics.get("repo_id"),
                "source_revision": source_metrics.get("revision"),
                "device": str(self.pager.device),
                "compute_dtype": str(self.pager.compute_dtype),
                "packed_continuation_gemm": self.packed_continuation_gemm,
                "quantized_weight_plane": (
                    None
                    if getattr(self.pager, "q4_bank", None) is None
                    else dict(self.pager.q4_bank.identity)
                ),
                "delta_probe": None
                if self.delta_probe is None
                else id(self.delta_probe),
                "layer_boundary_observer": (
                    None
                    if self.layer_boundary_observer is None
                    else id(self.layer_boundary_observer)
                ),
                "layer_boundary_stages": list(self.layer_boundary_stages),
                "layer_boundary_layers": (
                    None
                    if self.layer_boundary_layers is None
                    else list(self.layer_boundary_layers)
                ),
                "graft": self._graft_snapshot_identity(),
                "mlp_sparse_executor": self._mlp_sparse_snapshot_identity(),
                "native_head_crsa": native,
            }
        )

    def _emit_native_head_crsa_evidence(
        self, rows: Iterable[NativeHeadCrsaEvidence]
    ) -> None:
        observer = self.native_head_crsa_observer
        if observer is None:
            return
        for row in tuple(rows):
            try:
                observer(row)
            except Exception as exc:
                try:
                    warnings.warn(
                        "native Head-CRSA observer failed after commit: "
                        f"{type(exc).__name__}: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                except Exception:
                    # Warning filters may promote warnings to exceptions; a
                    # passive evidence sink must still never roll back or mask
                    # an already committed model forward.
                    pass

    def _emit_native_prefix_sinkhorn_operator_blocks(
        self,
        rows: Iterable[NativePrefixSinkhornOperatorBlock],
    ) -> None:
        observer = self.native_prefix_sinkhorn_operator_observer
        if observer is None:
            return
        for row in tuple(rows):
            try:
                observer.callback(row)
            except Exception as exc:
                try:
                    warnings.warn(
                        "native Prefix-Sinkhorn operator observer failed after "
                        f"commit: {type(exc).__name__}: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                except Exception:
                    pass

    def _apply_graft_stateful(
        self,
        hidden: torch.Tensor,
        history: torch.Tensor | None,
        *,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.graft is None:
            return hidden, None
        active = self._graft_active()
        if active:
            if start_pos and history is None:
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

    def _validate_stateful_range_states(
        self,
        prior_states: Iterable[LayerState | None],
        *,
        batch_size: int,
        sequence_length: int,
        start_pos: int,
        start_layer: int,
        graft_history: torch.Tensor | None,
    ) -> tuple[LayerState | None, ...]:
        """Validate the explicit continuation boundary for a layer range."""

        try:
            states = tuple(prior_states)
        except TypeError as exc:
            raise TypeError("prior_states must be an iterable of layer states") from exc
        if len(states) != self.config.n_layers:
            raise ValueError(
                f"prior_states must contain {self.config.n_layers} layer entries"
            )

        end_pos = start_pos + sequence_length
        key_features = (
            self.config.linear_num_key_heads * self.config.linear_key_head_dim
        )
        value_features = (
            self.config.linear_num_value_heads * self.config.linear_value_head_dim
        )
        for layer, state in enumerate(states):
            already_executed = layer < start_layer
            required = start_pos > 0 or already_executed
            if not required:
                if state is not None:
                    raise Qwen38RuntimeError("position-zero state is not empty")
                continue
            if self.config.is_full_attention(layer):
                expected_length = end_pos if already_executed else start_pos
                if not isinstance(state, AttentionState):
                    raise Qwen38RuntimeError(
                        f"full-attention layer {layer} has the wrong state type"
                    )
                expected_shape = (
                    batch_size,
                    self.config.n_kv_heads,
                    expected_length,
                    self.config.head_dim,
                )
                if tuple(state.key.shape) != expected_shape:
                    raise Qwen38RuntimeError(
                        f"full-attention layer {layer} cursor disagrees with range state"
                    )
                if (
                    tuple(state.value.shape) != expected_shape
                    or not self._on_pager_device(state.value)
                    or state.value.dtype != self.pager.compute_dtype
                ):
                    raise Qwen38RuntimeError(
                        f"full-attention layer {layer} value state is invalid"
                    )
                if (
                    not self._on_pager_device(state.key)
                    or state.key.dtype != self.pager.compute_dtype
                ):
                    raise Qwen38RuntimeError(
                        f"full-attention layer {layer} state device/dtype is invalid"
                    )
                expects_usage = (
                    self.native_head_crsa is not None
                    and self.native_head_crsa.active
                    and layer == self.native_head_crsa.layer
                )
                if expects_usage != (state.crsa_log_usage is not None):
                    raise Qwen38RuntimeError(
                        f"full-attention layer {layer} CRSA history disagrees "
                        "with model configuration"
                    )
                if state.crsa_log_usage is not None:
                    expected_usage_dtype = (
                        torch.float32
                        if self.pager.compute_dtype in {torch.bfloat16, torch.float16}
                        else self.pager.compute_dtype
                    )
                    if (
                        tuple(state.crsa_log_usage.shape)
                        != (batch_size, 4, expected_length)
                        or not self._on_pager_device(state.crsa_log_usage)
                        or state.crsa_log_usage.dtype != expected_usage_dtype
                    ):
                        raise Qwen38RuntimeError(
                            f"full-attention layer {layer} CRSA history "
                            "shape/device/dtype is invalid"
                        )
            else:
                if not isinstance(state, DeltaNetState):
                    raise Qwen38RuntimeError(
                        f"linear-attention layer {layer} has the wrong state type"
                    )
                if tuple(state.conv.shape) != (
                    batch_size,
                    2 * key_features + value_features,
                    self.config.linear_conv_kernel_dim,
                ) or tuple(state.recurrent.shape) != (
                    batch_size,
                    self.config.linear_num_value_heads,
                    self.config.linear_key_head_dim,
                    self.config.linear_value_head_dim,
                ):
                    raise Qwen38RuntimeError(
                        f"linear-attention layer {layer} state shape is invalid"
                    )
                if (
                    not self._on_pager_device(state.conv)
                    or state.conv.dtype != self.pager.compute_dtype
                    or not self._on_pager_device(state.recurrent)
                    or state.recurrent.dtype != torch.float32
                ):
                    raise Qwen38RuntimeError(
                        f"linear-attention layer {layer} state device/dtype is invalid"
                    )

        if not self._graft_active():
            if graft_history is not None:
                raise Qwen38RuntimeError(
                    "inactive graft may not retain continuation history"
                )
            return states
        if self.graft_layer is None:  # pragma: no cover - constructor contract.
            raise Qwen38RuntimeError("active graft has no decoder layer")
        expected_history = end_pos if self.graft_layer < start_layer else start_pos
        if expected_history == 0:
            if graft_history is not None:
                raise Qwen38RuntimeError("position-zero graft history is not empty")
            return states
        if (
            not isinstance(graft_history, torch.Tensor)
            or not graft_history.is_floating_point()
            or tuple(graft_history.shape)
            != (batch_size, expected_history, self.config.dim)
            or not self._on_pager_device(graft_history)
            or graft_history.dtype != self.pager.compute_dtype
        ):
            raise Qwen38RuntimeError(
                "graft history shape/cursor disagrees with range state"
            )
        return states

    def _poison_state(self) -> None:
        self._layer_states = [None for _ in range(self.config.n_layers)]
        self._next_position = 0
        self._state_batch_size = None
        self._graft_history = None
        self._state_poisoned = True
        self._pending_block_stage = None

    def reset_state(self, *, release: bool = False) -> None:
        """Drop every committed KV/DeltaNet cache and clear the poison latch."""

        self._layer_states = [None for _ in range(self.config.n_layers)]
        self._next_position = 0
        self._state_batch_size = None
        self._state_poisoned = False
        self._graft_history = None
        self._pending_block_stage = None
        if release:
            # ``release=True`` is the public request teardown boundary.  Layer
            # boundaries use the pager's bounded interval/RSS policy; request
            # teardown always completes a full cyclic-GC pass.
            self.pager.release(force_gc=True)

    def hidden_stateful_range(
        self,
        hidden: Any,
        prior_states: Iterable[LayerState | None],
        *,
        start_pos: int,
        start_layer: int,
        stop_layer: int,
        graft_history: torch.Tensor | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None,
        native_head_crsa_tokenwise_usage: bool = False,
    ) -> StatefulLayerRangeResult:
        """Stage exactly ``[start_layer, stop_layer)`` without committing state.

        The caller owns the explicit hidden-state and continuation boundary.
        Successful and failed calls leave the model cursor, poison latch, layer
        states, graft history, and external native-CRSA observer untouched.
        """

        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise ValueError("hidden must have [batch, sequence, dim] shape")
        if not hidden.is_floating_point():
            raise TypeError("hidden must be floating point")
        batch_size, sequence_length, width = hidden.shape
        if width != self.config.dim:
            raise ValueError("hidden width does not match the checkpoint")
        if batch_size < 1 or sequence_length < 1:
            raise ValueError("hidden batch and sequence must be non-empty")
        if batch_size > self.max_batch_size:
            raise ValueError("hidden batch exceeds max_batch_size")
        if sequence_length > self.max_seq_len:
            raise ValueError("hidden sequence exceeds max_seq_len")
        if (
            isinstance(start_pos, bool)
            or not isinstance(start_pos, int)
            or start_pos < 0
        ):
            raise ValueError("start_pos must be a non-negative integer")
        end_pos = start_pos + sequence_length
        if end_pos > self.max_seq_len:
            raise ValueError(
                f"context end {end_pos} exceeds max_seq_len={self.max_seq_len}"
            )
        if isinstance(start_layer, bool) or not isinstance(start_layer, int):
            raise TypeError("start_layer must be an integer")
        if isinstance(stop_layer, bool) or not isinstance(stop_layer, int):
            raise TypeError("stop_layer must be an integer")
        if not 0 <= start_layer <= stop_layer <= self.config.n_layers:
            raise ValueError(
                "layer range must satisfy 0 <= start_layer <= stop_layer <= depth"
            )
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable or None")
        if not isinstance(native_head_crsa_tokenwise_usage, bool):
            raise TypeError("native_head_crsa_tokenwise_usage must be a boolean")

        x = hidden.to(device=self.pager.device, dtype=self.pager.compute_dtype)
        states = self._validate_stateful_range_states(
            prior_states,
            batch_size=batch_size,
            sequence_length=sequence_length,
            start_pos=start_pos,
            start_layer=start_layer,
            graft_history=graft_history,
        )
        mask = torch.ones(
            (batch_size, sequence_length),
            dtype=torch.bool,
            device=self.pager.device,
        )
        source = self.pager.source
        start_bytes = self._metric(source, "network_or_source_body_bytes")
        start_linears = self._metric(self.pager, "linear_calls")
        started = time.perf_counter()
        staged = list(states)
        staged_history = graft_history
        staged_native_evidence: list[NativeHeadCrsaEvidence] = []
        try:
            for layer in range(start_layer, stop_layer):
                if progress is not None:
                    layer_started = time.perf_counter()
                    layer_bytes = self._metric(source, "network_or_source_body_bytes")
                x, next_state = self._forward_layer(
                    x,
                    layer=layer,
                    token_mask=mask,
                    state=states[layer],
                    start_pos=start_pos,
                    stateful=True,
                    native_head_crsa_observer=staged_native_evidence.append,
                    native_head_crsa_tokenwise_usage=(native_head_crsa_tokenwise_usage),
                )
                if next_state is None:  # pragma: no cover - stateful contract.
                    raise Qwen38RuntimeError("stateful layer returned no continuation")
                staged[layer] = next_state
                if self.graft is not None and layer == self.graft_layer:
                    x, staged_history = self._apply_graft_stateful(
                        x,
                        staged_history,
                        start_pos=start_pos,
                    )
                self.pager.release()
                if progress is not None:
                    progress(
                        {
                            "event": "qwen_stateful_layer_complete",
                            "layer": layer,
                            "layers": self.config.n_layers,
                            "start_pos": start_pos,
                            "tokens": sequence_length,
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
        finally:
            self.pager.release()

        staged_states = tuple(staged)
        evidence = StatefulLayerRangeEvidence(
            start_pos=start_pos,
            end_pos=end_pos,
            start_layer=start_layer,
            stop_layer=stop_layer,
            layers_executed=stop_layer - start_layer,
            source_body_bytes=(
                self._metric(source, "network_or_source_body_bytes") - start_bytes
            ),
            linear_calls=self._metric(self.pager, "linear_calls") - start_linears,
            seconds=time.perf_counter() - started,
            staged_state_bytes=self._continuation_bytes(staged_states, staged_history),
        )
        return StatefulLayerRangeResult(
            hidden=x,
            layer_states=staged_states,
            graft_history=staged_history,
            native_head_crsa_evidence=tuple(staged_native_evidence),
            evidence=evidence,
        )

    def _stage_continuation_token_rows(
        self,
        hidden: torch.Tensor,
        prior_states: tuple[LayerState | None, ...],
        *,
        start_pos: int,
        graft_history: torch.Tensor | None,
        progress: Callable[[dict[str, Any]], None] | None,
    ) -> StatefulLayerRangeResult:
        """Stage batch-1/K=1..4 as layer-major one-token recurrences."""

        if (
            hidden.ndim != 3
            or hidden.shape[0] != 1
            or hidden.shape[2] != self.config.dim
            or not 1 <= hidden.shape[1] <= self.MAX_CONTINUATION_BLOCK_WIDTH
        ):
            raise Qwen38RuntimeError("continuation embedded hidden shape is invalid")
        x = hidden.to(device=self.pager.device, dtype=self.pager.compute_dtype)
        rows = tuple(x[:, offset : offset + 1] for offset in range(x.shape[1]))
        states = tuple(prior_states)
        staged = list(states)
        staged_history = graft_history
        staged_native_evidence: list[NativeHeadCrsaEvidence] = []
        staged_layer_traces = [
            _LayerPrefixTrace() for _ in range(self.config.n_layers)
        ]
        graft_prefix_histories: list[torch.Tensor | None] = [None] * len(rows)
        staged_operator_blocks: list[NativePrefixSinkhornOperatorBlock] = []
        public_operator_observer = self.native_prefix_sinkhorn_operator_observer
        private_operator_observer = (
            None
            if public_operator_observer is None
            else NativePrefixSinkhornOperatorObserver(
                staged_operator_blocks.append,
                max_positions=public_operator_observer.max_positions,
            )
        )
        source = self.pager.source
        start_bytes = self._metric(source, "network_or_source_body_bytes")
        start_linears = self._metric(self.pager, "linear_calls")
        started = time.perf_counter()
        try:
            for layer in range(self.config.n_layers):
                if progress is not None:
                    layer_started = time.perf_counter()
                    layer_bytes = self._metric(source, "network_or_source_body_bytes")
                if len(rows) == 2:
                    pair, next_state, prefix_trace = self._forward_layer_k2_pair(
                        (rows[0], rows[1]),
                        layer=layer,
                        state=states[layer],
                        start_pos=start_pos,
                        native_head_crsa_observer=staged_native_evidence.append,
                        native_prefix_sinkhorn_operator_observer=(
                            private_operator_observer
                        ),
                    )
                    rows = pair
                else:
                    rows, next_state, prefix_trace = self._forward_layer_token_rows(
                        rows,
                        layer=layer,
                        state=states[layer],
                        start_pos=start_pos,
                        native_head_crsa_observer=staged_native_evidence.append,
                        native_prefix_sinkhorn_operator_observer=(
                            private_operator_observer
                        ),
                    )
                staged[layer] = next_state
                if prefix_trace.width != len(rows):
                    raise Qwen38RuntimeError(
                        "continuation prefix trace lost a token row"
                    )
                staged_layer_traces[layer] = prefix_trace
                if self.graft is not None and layer == self.graft_layer:
                    grafted: list[torch.Tensor] = []
                    for offset, row in enumerate(rows):
                        row, staged_history = self._apply_graft_stateful(
                            row,
                            staged_history,
                            start_pos=start_pos + offset,
                        )
                        graft_prefix_histories[offset] = staged_history
                        grafted.append(row)
                    rows = tuple(grafted)
                self.pager.release()
                if progress is not None:
                    progress(
                        {
                            "event": "qwen_stateful_layer_complete",
                            "layer": layer,
                            "layers": self.config.n_layers,
                            "start_pos": start_pos,
                            "tokens": len(rows),
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
        finally:
            self.pager.release()

        staged_states = tuple(staged)
        prefix_trace = _ContinuationPrefixTrace(
            width=len(rows),
            layers=tuple(staged_layer_traces),
            graft_histories=tuple(graft_prefix_histories),
            native_operator_blocks=tuple(staged_operator_blocks),
        )
        combined = torch.cat(rows, dim=1)
        evidence = StatefulLayerRangeEvidence(
            start_pos=start_pos,
            end_pos=start_pos + len(rows),
            start_layer=0,
            stop_layer=self.config.n_layers,
            layers_executed=self.config.n_layers,
            source_body_bytes=(
                self._metric(source, "network_or_source_body_bytes") - start_bytes
            ),
            linear_calls=self._metric(self.pager, "linear_calls") - start_linears,
            seconds=time.perf_counter() - started,
            staged_state_bytes=self._continuation_bytes(staged_states, staged_history),
        )
        return StatefulLayerRangeResult(
            hidden=combined,
            layer_states=staged_states,
            graft_history=staged_history,
            native_head_crsa_evidence=tuple(staged_native_evidence),
            evidence=evidence,
            prefix_trace=prefix_trace,
        )

    def _stage_continuation_k2_pair(
        self,
        hidden: torch.Tensor,
        prior_states: tuple[LayerState | None, ...],
        *,
        start_pos: int,
        graft_history: torch.Tensor | None,
        progress: Callable[[dict[str, Any]], None] | None,
    ) -> StatefulLayerRangeResult:
        """Preserve the existing K=2 staging seam and evidence contract."""

        if tuple(hidden.shape) != (1, 2, self.config.dim):
            raise Qwen38RuntimeError("K=2 embedded hidden shape is invalid")
        return self._stage_continuation_token_rows(
            hidden,
            prior_states,
            start_pos=start_pos,
            graft_history=graft_history,
            progress=progress,
        )

    @staticmethod
    def _merge_prefix_traces(
        left: _ContinuationPrefixTrace,
        right: _ContinuationPrefixTrace,
        left_states: tuple[LayerState | None, ...],
    ) -> _ContinuationPrefixTrace:
        if (
            right.width != 1
            or len(left.layers) != len(right.layers)
            or len(left_states) != len(left.layers)
        ):
            raise Qwen38RuntimeError("continuation prefix traces cannot be merged")
        rows: list[_LayerPrefixTrace] = []
        for before, after, left_state in zip(
            left.layers, right.layers, left_states, strict=True
        ):
            conv_prefix = before.delta_conv_prefix + after.delta_conv_prefix
            if before.delta_updates:
                if not isinstance(left_state, DeltaNetState):
                    raise Qwen38RuntimeError(
                        "continuation DeltaNet prefix state is missing"
                    )
                kernel_size = int(left_state.conv.shape[-1])
                expected = max(0, left.width + 1 - kernel_size)
                if len(conv_prefix) < expected:
                    conv_prefix += (
                        left_state.conv[..., :1].detach().clone(),
                    )
            rows.append(
                _LayerPrefixTrace(
                    attention_log_usage=(
                        before.attention_log_usage + after.attention_log_usage
                    ),
                    delta_updates=before.delta_updates + after.delta_updates,
                    delta_conv_prefix=conv_prefix,
                )
            )
        layers = tuple(rows)
        if any(row.width != left.width + 1 for row in layers):
            raise Qwen38RuntimeError("merged continuation trace lost a token row")
        return _ContinuationPrefixTrace(
            width=left.width + 1,
            layers=layers,
            graft_histories=left.graft_histories + right.graft_histories,
            native_operator_blocks=(
                left.native_operator_blocks + right.native_operator_blocks
            ),
        )

    def stage_continuation_block(
        self,
        token_ids: Any,
        *,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> StatefulBlockStage:
        """Execute a continuation block without changing committed state.

        At most one stage is live per model.  Creating a later stage
        invalidates the earlier one.  Native Head-CRSA evidence stays private
        until :meth:`commit_continuation_block` succeeds.
        """

        ids = self._token_tensor(token_ids)
        if self._state_poisoned:
            raise Qwen38RuntimeError("decoder state is poisoned; call reset_state()")
        if self._next_position == 0:
            raise ValueError("continuation block requires a completed prefill")
        if self.delta_probe is not None:
            raise Qwen38RuntimeError(
                "continuation block staging rejects an active DeltaNet probe"
            )
        if (
            ids.shape[0] != 1
            or not 1 <= ids.shape[1] <= self.MAX_CONTINUATION_BLOCK_WIDTH
        ):
            raise Qwen38RuntimeError(
                "continuation block staging requires batch 1 and K in [1, 16]"
            )
        if ids.shape[0] != self._state_batch_size:
            raise ValueError("continuation block batch differs from committed prefix")
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable or None")
        start_pos = self._next_position
        end_pos = start_pos + ids.shape[1]
        if end_pos > self.max_seq_len:
            raise ValueError(
                f"context end {end_pos} exceeds max_seq_len={self.max_seq_len}"
            )
        committed_states = self._validate_stateful_range_states(
            self._layer_states,
            batch_size=ids.shape[0],
            sequence_length=ids.shape[1],
            start_pos=start_pos,
            start_layer=0,
            graft_history=self._graft_history,
        )
        runtime_identity = self._continuation_block_runtime_identity()

        self._pending_block_stage = None
        source = self.pager.source
        start_bytes = self._metric(source, "network_or_source_body_bytes")
        start_linears = self._metric(self.pager, "linear_calls")
        started = time.perf_counter()
        try:
            embedded = self.embed_batch(ids)
            if ids.shape[1] == 2:
                staged = self._stage_continuation_k2_pair(
                    embedded,
                    committed_states,
                    start_pos=start_pos,
                    graft_history=self._graft_history,
                    progress=progress,
                )
                final_rows = self._norm_k2_pair(
                    (staged.hidden[:, :1], staged.hidden[:, 1:]),
                    self.FINAL_NORM_NAME,
                )
            else:
                staged = self._stage_continuation_token_rows(
                    embedded,
                    committed_states,
                    start_pos=start_pos,
                    graft_history=self._graft_history,
                    progress=progress,
                )
                final_rows = self._norm_token_rows(
                    tuple(
                        staged.hidden[:, offset : offset + 1]
                        for offset in range(ids.shape[1])
                    ),
                    self.FINAL_NORM_NAME,
                )
            hidden = torch.cat(final_rows, dim=1)
        except Exception:
            self._pending_block_stage = None
            self.pager.release()
            raise
        finally:
            self.pager.release()

        input_ids = tuple(
            tuple(int(value) for value in row)
            for row in ids.detach().to("cpu").tolist()
        )
        evidence = StatefulBlockEvidence(
            start_pos=start_pos,
            end_pos=end_pos,
            input_token_ids=input_ids,
            layers_executed=self.config.n_layers,
            checkpoint_layers=self.config.n_layers,
            complete_layer_stack=True,
            stateful_cache=True,
            source_body_bytes=(
                self._metric(source, "network_or_source_body_bytes") - start_bytes
            ),
            linear_calls=self._metric(self.pager, "linear_calls") - start_linears,
            seconds=time.perf_counter() - started,
            staged_state_bytes=self._continuation_bytes(
                staged.layer_states, staged.graft_history
            ),
            graft_mode=self._graft_mode(),
            graft_history_tokens=(
                0
                if staged.graft_history is None
                else int(staged.graft_history.shape[1])
            ),
        )
        stage = StatefulBlockStage(
            hidden=hidden.detach().clone(),
            evidence=replace(evidence),
            _nonce=object(),
        )
        if staged.prefix_trace is None:
            raise Qwen38RuntimeError("continuation stage lacks its prefix trace")
        self._pending_block_stage = _PendingStatefulBlock(
            handle=stage,
            evidence=evidence,
            hidden=hidden,
            layer_states=staged.layer_states,
            graft_history=staged.graft_history,
            native_head_crsa_evidence=staged.native_head_crsa_evidence,
            prefix_trace=staged.prefix_trace,
            runtime_identity=runtime_identity,
        )
        return stage

    def extend_continuation_block(
        self,
        stage: StatefulBlockStage,
        one_token: Any,
        *,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> StatefulBlockStage:
        """Consume a pending stage and append one exact private token update.

        The returned handle contains the aggregate hidden/evidence view.  The
        old handle is invalid even when extension fails, while committed model
        state and the public native-CRSA observer remain untouched.
        """

        if not isinstance(stage, StatefulBlockStage):
            raise TypeError("stage must be a StatefulBlockStage")
        pending = self._pending_block_stage
        if pending is None or stage is not pending.handle:
            raise Qwen38RuntimeError("continuation block stage is stale or foreign")

        # A matching handle is consumed exactly once.  No later validation or
        # compute failure may leave the old private transaction committable.
        self._pending_block_stage = None
        try:
            ids = self._token_tensor(one_token)
            row = pending.evidence
            width = row.end_pos - row.start_pos
            if self._state_poisoned:
                raise Qwen38RuntimeError(
                    "decoder state is poisoned; call reset_state()"
                )
            if self.delta_probe is not None:
                raise Qwen38RuntimeError(
                    "continuation block staging rejects an active DeltaNet probe"
                )
            if tuple(ids.shape) != (1, 1):
                raise Qwen38RuntimeError(
                    "continuation block extension requires batch 1 and one token"
                )
            if not 1 <= width < self.MAX_CONTINUATION_BLOCK_WIDTH:
                raise Qwen38RuntimeError("continuation block extension exceeds K=16")
            if self._next_position != row.start_pos:
                raise Qwen38RuntimeError("continuation block base state changed")
            if self._state_batch_size != 1 or len(row.input_token_ids) != 1:
                raise Qwen38RuntimeError("continuation block batch state changed")
            if row.end_pos + 1 > self.max_seq_len or self.max_batch_size < 1:
                raise Qwen38RuntimeError(
                    "continuation block exceeds current runtime bounds"
                )
            runtime_identity = self._continuation_block_runtime_identity()
            if runtime_identity != pending.runtime_identity:
                raise Qwen38RuntimeError(
                    "continuation block runtime configuration changed"
                )
            if tuple(pending.hidden.shape) != (1, width, self.config.dim):
                raise Qwen38RuntimeError("continuation block hidden shape is invalid")
            if (
                not pending.hidden.is_floating_point()
                or not self._on_pager_device(pending.hidden)
                or pending.hidden.dtype != self.pager.compute_dtype
            ):
                raise Qwen38RuntimeError(
                    "continuation block hidden device/dtype is invalid"
                )
            staged_base = self._validate_stateful_range_states(
                pending.layer_states,
                batch_size=1,
                sequence_length=0,
                start_pos=row.end_pos,
                start_layer=0,
                graft_history=pending.graft_history,
            )
            if progress is not None and not callable(progress):
                raise TypeError("progress must be callable or None")

            source = self.pager.source
            start_bytes = self._metric(source, "network_or_source_body_bytes")
            start_linears = self._metric(self.pager, "linear_calls")
            started = time.perf_counter()
            embedded = self.embed_batch(ids)
            staged = self._stage_continuation_token_rows(
                embedded,
                staged_base,
                start_pos=row.end_pos,
                graft_history=pending.graft_history,
                progress=progress,
            )
            if staged.prefix_trace is None:
                raise Qwen38RuntimeError("continuation extension lacks its prefix trace")
            prefix_trace = self._merge_prefix_traces(
                pending.prefix_trace,
                staged.prefix_trace,
                pending.layer_states,
            )
            final_row = self._norm_token_rows(
                (staged.hidden,),
                self.FINAL_NORM_NAME,
            )[0]
            hidden = torch.cat((pending.hidden, final_row), dim=1)
            input_ids = (
                row.input_token_ids[0]
                + tuple(int(value) for value in ids[0].detach().to("cpu").tolist()),
            )
            evidence = StatefulBlockEvidence(
                start_pos=row.start_pos,
                end_pos=row.end_pos + 1,
                input_token_ids=input_ids,
                layers_executed=self.config.n_layers,
                checkpoint_layers=self.config.n_layers,
                complete_layer_stack=True,
                stateful_cache=True,
                source_body_bytes=row.source_body_bytes
                + self._metric(source, "network_or_source_body_bytes")
                - start_bytes,
                linear_calls=row.linear_calls
                + self._metric(self.pager, "linear_calls")
                - start_linears,
                seconds=row.seconds + time.perf_counter() - started,
                staged_state_bytes=self._continuation_bytes(
                    staged.layer_states, staged.graft_history
                ),
                graft_mode=self._graft_mode(),
                graft_history_tokens=(
                    0
                    if staged.graft_history is None
                    else int(staged.graft_history.shape[1])
                ),
            )
            next_stage = StatefulBlockStage(
                hidden=hidden.detach().clone(),
                evidence=replace(evidence),
                _nonce=object(),
            )
            self._pending_block_stage = _PendingStatefulBlock(
                handle=next_stage,
                evidence=evidence,
                hidden=hidden,
                layer_states=staged.layer_states,
                graft_history=staged.graft_history,
                native_head_crsa_evidence=(
                    pending.native_head_crsa_evidence + staged.native_head_crsa_evidence
                ),
                prefix_trace=prefix_trace,
                runtime_identity=runtime_identity,
            )
            return next_stage
        except Exception:
            self._pending_block_stage = None
            raise
        finally:
            self.pager.release()

    def discard_continuation_block(self, stage: StatefulBlockStage) -> None:
        """Invalidate exactly the currently pending block without a commit."""

        if not isinstance(stage, StatefulBlockStage):
            raise TypeError("stage must be a StatefulBlockStage")
        pending = self._pending_block_stage
        if pending is None or stage is not pending.handle:
            raise Qwen38RuntimeError("continuation block stage is stale or foreign")
        self._pending_block_stage = None
        self.pager.release()

    def _reconstruct_continuation_prefix_states(
        self,
        row: StatefulBlockEvidence,
        final_states: list[LayerState | None],
        trace: _ContinuationPrefixTrace,
        *,
        width: int,
    ) -> tuple[tuple[LayerState, ...], torch.Tensor | None]:
        stage_width = row.end_pos - row.start_pos
        if (
            trace.width != stage_width
            or len(trace.layers) != self.config.n_layers
            or len(trace.graft_histories) != stage_width
            or not 1 <= width < stage_width
        ):
            raise Qwen38RuntimeError("continuation prefix trace is invalid")
        end_pos = row.start_pos + width
        candidate = final_states
        for layer, (base, layer_trace) in enumerate(
            zip(self._layer_states, trace.layers, strict=True)
        ):
            final = candidate[layer]
            if self.config.is_full_attention(layer):
                if (
                    not isinstance(base, AttentionState)
                    or not isinstance(final, AttentionState)
                    or len(layer_trace.attention_log_usage) != stage_width
                    or layer_trace.delta_updates
                    or layer_trace.delta_conv_prefix
                ):
                    raise Qwen38RuntimeError(
                        f"attention prefix trace is invalid at layer {layer}"
                    )
                usage = layer_trace.attention_log_usage[width - 1]
                candidate[layer] = AttentionState(
                    key=final.key[:, :, :end_pos],
                    value=final.value[:, :, :end_pos],
                    crsa_log_usage=usage,
                )
                continue
            if (
                not isinstance(base, DeltaNetState)
                or not isinstance(final, DeltaNetState)
                or len(layer_trace.delta_updates) != stage_width
                or layer_trace.attention_log_usage
            ):
                raise Qwen38RuntimeError(
                    f"DeltaNet prefix trace is invalid at layer {layer}"
                )
            recurrent = base.recurrent
            for update in layer_trace.delta_updates[:width]:
                _unused, recurrent = recurrent_gated_delta_rule(
                    update.key,
                    update.key,
                    update.value,
                    update.log_decay,
                    update.beta,
                    initial_state=recurrent,
                )
            kernel_size = int(final.conv.shape[-1])
            prefix_count = max(0, stage_width - kernel_size)
            if len(layer_trace.delta_conv_prefix) != prefix_count:
                raise Qwen38RuntimeError(
                    f"DeltaNet Conv prefix trace is invalid at layer {layer}"
                )
            stored = layer_trace.delta_conv_prefix[: min(width, prefix_count)]
            suffix_count = max(0, width - prefix_count)
            staged_suffix = final.conv[..., -min(kernel_size, stage_width) :]
            pieces = (*stored, staged_suffix[..., :suffix_count])
            appended = (
                staged_suffix[..., :0]
                if not pieces
                else torch.cat(pieces, dim=-1)
            )
            conv = torch.cat((base.conv, appended), dim=-1)[
                ..., -kernel_size:
            ].contiguous()
            if (
                tuple(conv.shape) != tuple(base.conv.shape)
                or conv.dtype != base.conv.dtype
                or conv.device != base.conv.device
            ):
                raise Qwen38RuntimeError(
                    f"DeltaNet Conv prefix trace is invalid at layer {layer}"
                )
            candidate[layer] = DeltaNetState(conv=conv, recurrent=recurrent)
        graft_history = trace.graft_histories[width - 1]
        if any(not isinstance(state, (AttentionState, DeltaNetState)) for state in candidate):
            raise Qwen38RuntimeError("continuation prefix reconstruction is incomplete")
        return tuple(candidate), graft_history  # type: ignore[arg-type]

    def commit_continuation_prefix(
        self,
        stage: StatefulBlockStage,
        width: int,
    ) -> tuple[torch.Tensor, StatefulEvidence]:
        """Commit a verified staged prefix without reading model weights again."""

        if not isinstance(stage, StatefulBlockStage):
            raise TypeError("stage must be a StatefulBlockStage")
        pending = self._pending_block_stage
        if pending is None or stage is not pending.handle:
            raise Qwen38RuntimeError("continuation block stage is stale or foreign")
        row = pending.evidence
        stage_width = row.end_pos - row.start_pos
        if isinstance(width, bool) or not isinstance(width, int):
            raise TypeError("prefix width must be an integer")
        if not 1 <= width <= stage_width:
            raise ValueError("prefix width must lie inside the staged block")
        if width == stage_width:
            return self.commit_continuation_block(stage)
        if self._state_poisoned:
            self._pending_block_stage = None
            raise Qwen38RuntimeError("decoder state is poisoned; call reset_state()")
        if self._next_position != row.start_pos:
            self._pending_block_stage = None
            raise Qwen38RuntimeError("continuation block base state changed")
        if self._state_batch_size != len(row.input_token_ids):
            self._pending_block_stage = None
            raise Qwen38RuntimeError("continuation block batch state changed")
        if len(row.input_token_ids) != 1 or row.end_pos > self.max_seq_len:
            self._pending_block_stage = None
            raise Qwen38RuntimeError("continuation block exceeds runtime bounds")
        if self._continuation_block_runtime_identity() != pending.runtime_identity:
            self._pending_block_stage = None
            raise Qwen38RuntimeError("continuation block runtime configuration changed")
        if tuple(pending.hidden.shape) != (1, stage_width, self.config.dim):
            self._pending_block_stage = None
            raise Qwen38RuntimeError("continuation block hidden shape is invalid")
        if (
            not pending.hidden.is_floating_point()
            or not self._on_pager_device(pending.hidden)
            or pending.hidden.dtype != self.pager.compute_dtype
        ):
            self._pending_block_stage = None
            raise Qwen38RuntimeError(
                "continuation block hidden device/dtype is invalid"
            )
        self._validate_stateful_range_states(
            pending.layer_states,
            batch_size=1,
            sequence_length=0,
            start_pos=row.end_pos,
            start_layer=0,
            graft_history=pending.graft_history,
        )

        self._pending_block_stage = None
        started = time.perf_counter()
        pending_hidden = pending.hidden
        pending_states = list(pending.layer_states)
        pending_native_evidence = pending.native_head_crsa_evidence
        prefix_trace = pending.prefix_trace
        del pending
        states, graft_history = self._reconstruct_continuation_prefix_states(
            row,
            pending_states,
            prefix_trace,
            width=width,
        )
        end_pos = row.start_pos + width
        hidden = pending_hidden[:, :width]
        self._layer_states = list(states)
        self._next_position = end_pos
        self._state_batch_size = 1
        self._graft_history = graft_history
        evidence = StatefulEvidence(
            start_pos=row.start_pos,
            end_pos=end_pos,
            input_token_ids=(row.input_token_ids[0][:width],),
            layers_executed=row.layers_executed,
            checkpoint_layers=row.checkpoint_layers,
            complete_layer_stack=row.complete_layer_stack,
            context_mode="decode",
            stateful_cache=row.stateful_cache,
            source_body_bytes=row.source_body_bytes,
            linear_calls=row.linear_calls,
            seconds=row.seconds + time.perf_counter() - started,
            state_bytes=self.state_bytes,
            graft_mode=row.graft_mode,
            graft_history_tokens=(
                0 if graft_history is None else int(graft_history.shape[1])
            ),
        )
        self._emit_native_head_crsa_evidence(
            row
            for row in pending_native_evidence
            if row.query_start + row.query_length <= end_pos
        )
        self._emit_native_prefix_sinkhorn_operator_blocks(
            row
            for row in prefix_trace.native_operator_blocks
            if row.query_positions[-1] < end_pos
        )
        return hidden, evidence

    def commit_continuation_block(
        self, stage: StatefulBlockStage
    ) -> tuple[torch.Tensor, StatefulEvidence]:
        """Atomically publish one staged block and its delayed evidence."""

        if not isinstance(stage, StatefulBlockStage):
            raise TypeError("stage must be a StatefulBlockStage")
        pending = self._pending_block_stage
        if pending is None or stage is not pending.handle:
            raise Qwen38RuntimeError("continuation block stage is stale or foreign")
        row = pending.evidence
        if self._state_poisoned:
            self._pending_block_stage = None
            raise Qwen38RuntimeError("decoder state is poisoned; call reset_state()")
        if self._next_position != row.start_pos:
            self._pending_block_stage = None
            raise Qwen38RuntimeError("continuation block base state changed")
        if self._state_batch_size != len(row.input_token_ids):
            self._pending_block_stage = None
            raise Qwen38RuntimeError("continuation block batch state changed")
        if (
            len(row.input_token_ids) > self.max_batch_size
            or row.end_pos > self.max_seq_len
        ):
            self._pending_block_stage = None
            raise Qwen38RuntimeError(
                "continuation block exceeds current runtime bounds"
            )
        if self._continuation_block_runtime_identity() != pending.runtime_identity:
            self._pending_block_stage = None
            raise Qwen38RuntimeError("continuation block runtime configuration changed")
        if tuple(pending.hidden.shape) != (
            len(row.input_token_ids),
            row.end_pos - row.start_pos,
            self.config.dim,
        ):
            self._pending_block_stage = None
            raise Qwen38RuntimeError("continuation block hidden shape is invalid")
        if (
            not pending.hidden.is_floating_point()
            or not self._on_pager_device(pending.hidden)
            or pending.hidden.dtype != self.pager.compute_dtype
        ):
            self._pending_block_stage = None
            raise Qwen38RuntimeError(
                "continuation block hidden device/dtype is invalid"
            )
        try:
            self._validate_stateful_range_states(
                pending.layer_states,
                batch_size=len(row.input_token_ids),
                sequence_length=0,
                start_pos=row.end_pos,
                start_layer=0,
                graft_history=pending.graft_history,
            )
        except Exception:
            self._pending_block_stage = None
            raise

        committed_states = list(pending.layer_states)
        self._pending_block_stage = None
        self._layer_states = committed_states
        self._next_position = row.end_pos
        self._state_batch_size = len(row.input_token_ids)
        self._graft_history = pending.graft_history
        evidence = StatefulEvidence(
            start_pos=row.start_pos,
            end_pos=row.end_pos,
            input_token_ids=row.input_token_ids,
            layers_executed=row.layers_executed,
            checkpoint_layers=row.checkpoint_layers,
            complete_layer_stack=row.complete_layer_stack,
            context_mode="decode",
            stateful_cache=row.stateful_cache,
            source_body_bytes=row.source_body_bytes,
            linear_calls=row.linear_calls,
            seconds=row.seconds,
            state_bytes=self.state_bytes,
            graft_mode=row.graft_mode,
            graft_history_tokens=row.graft_history_tokens,
        )
        self._emit_native_head_crsa_evidence(pending.native_head_crsa_evidence)
        self._emit_native_prefix_sinkhorn_operator_blocks(
            pending.prefix_trace.native_operator_blocks
        )
        return pending.hidden, evidence

    def hidden_stateful(
        self,
        token_ids: Any,
        *,
        start_pos: int | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[torch.Tensor, StatefulEvidence]:
        """Execute one complete prefill or contiguous decode block.

        Logical state commits only after all 64 layers and the final norm
        succeed.  A continuation consumes and replaces the private committed
        layer cache as each layer completes, avoiding simultaneous residency
        of the complete old and new cache stacks.  A failed call clears partial
        state and latches the runtime until the caller acknowledges the failure
        with :meth:`reset_state`.
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
        committed_states = self._validate_stateful_range_states(
            self._layer_states,
            batch_size=ids.shape[0],
            sequence_length=ids.shape[1],
            start_pos=start_pos,
            start_layer=0,
            graft_history=self._graft_history,
        )

        source = self.pager.source
        start_bytes = self._metric(source, "network_or_source_body_bytes")
        start_linears = self._metric(self.pager, "linear_calls")
        started = time.perf_counter()
        hidden = self.embed_batch(ids)
        if start_pos > 0:
            # A successful embedding has crossed the last non-poisoning setup
            # boundary.  Any pending speculative stage is now stale regardless
            # of whether continuation compute commits or poisons, so release its
            # complete private cache before building replacement layer states.
            self._pending_block_stage = None
        staged: StatefulLayerRangeResult | None = None
        staged_native_evidence: tuple[NativeHeadCrsaEvidence, ...] = ()
        try:
            if start_pos == 0:
                staged = self.hidden_stateful_range(
                    hidden,
                    committed_states,
                    start_pos=start_pos,
                    start_layer=0,
                    stop_layer=self.config.n_layers,
                    graft_history=self._graft_history,
                    progress=progress,
                )
                hidden = staged.hidden
            else:
                # Validation returns a tuple so public range calls can retain a
                # stable, non-committing boundary.  Ordinary continuation is a
                # destructive transaction: drop that duplicate ownership, then
                # transfer each old layer state into its forward and publish the
                # replacement immediately.  Failure is still atomic to callers
                # because the established contract poisons and clears the whole
                # continuation state.
                del committed_states
                mask = torch.ones(
                    (ids.shape[0], ids.shape[1]),
                    dtype=torch.bool,
                    device=self.pager.device,
                )
                native_evidence: list[NativeHeadCrsaEvidence] = []
                for layer in range(self.config.n_layers):
                    if progress is not None:
                        layer_started = time.perf_counter()
                        layer_bytes = self._metric(
                            source, "network_or_source_body_bytes"
                        )
                    prior_state = self._layer_states[layer]
                    self._layer_states[layer] = None
                    try:
                        hidden, next_state = self._forward_layer(
                            hidden,
                            layer=layer,
                            token_mask=mask,
                            state=prior_state,
                            start_pos=start_pos,
                            stateful=True,
                            native_head_crsa_observer=native_evidence.append,
                        )
                    finally:
                        del prior_state
                    if next_state is None:  # pragma: no cover - stateful contract.
                        raise Qwen38RuntimeError(
                            "stateful layer returned no continuation"
                        )
                    self._layer_states[layer] = next_state
                    del next_state
                    if self.graft is not None and layer == self.graft_layer:
                        hidden, self._graft_history = self._apply_graft_stateful(
                            hidden,
                            self._graft_history,
                            start_pos=start_pos,
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
                                    if isinstance(
                                        self._layer_states[layer], AttentionState
                                    )
                                    else "deltanet"
                                ),
                            }
                        )
                staged_native_evidence = tuple(native_evidence)
            hidden = self.finalize_hidden(hidden)
        except Exception:
            self._poison_state()
            self.pager.release()
            raise
        finally:
            self.pager.release()

        self._pending_block_stage = None
        if staged is not None:
            self._layer_states = list(staged.layer_states)
            self._graft_history = staged.graft_history
            staged_native_evidence = staged.native_head_crsa_evidence
        self._next_position = end_pos
        self._state_batch_size = ids.shape[0]
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
        self._emit_native_head_crsa_evidence(staged_native_evidence)
        return hidden, evidence

    def prefill(
        self,
        token_ids: Any,
        *,
        tokenwise: bool = False,
        reset: bool = True,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[torch.Tensor, tuple[StatefulEvidence, ...]]:
        """Commit a fresh prompt or extend an existing prefix by a token block.

        ``reset=True`` starts at position zero. ``reset=False`` preserves the
        committed KV/DeltaNet/CRSA state and treats ``token_ids`` as the exact
        suffix beginning at :attr:`next_position`.
        """

        if not isinstance(tokenwise, bool):
            raise TypeError("tokenwise must be a boolean")
        ids = self._token_tensor(token_ids)
        if reset:
            self.reset_state()
        start_pos = self._next_position
        if not tokenwise:
            hidden, evidence = self.hidden_stateful(
                ids, start_pos=start_pos, progress=progress
            )
            return hidden, (evidence,)
        outputs: list[torch.Tensor] = []
        evidence_rows: list[StatefulEvidence] = []
        for position in range(ids.shape[1]):
            hidden, evidence = self.hidden_stateful(
                ids[:, position : position + 1],
                start_pos=start_pos + position,
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
        restored_prefix_length: int | None = None,
        restored_seed_hidden: torch.Tensor | None = None,
        eos_token_ids: Iterable[int] = (),
        head_block_rows: int = Qwen38WeightPager.DEFAULT_HEAD_BLOCK_ROWS,
        progress: Callable[[dict[str, Any]], None] | None = None,
        head_progress: Callable[[dict[str, int]], None] | None = None,
        retain_final_state: bool = True,
    ) -> tuple[tuple[int, ...], GenerationEvidence]:
        """Run exact greedy generation, optionally from an authenticated prefix.

        ``restored_prefix_length`` declares that native continuation state for
        exactly that many leading prompt tokens is already committed.  Only the
        remaining prompt suffix is evaluated.  An exact-prefix restore has no
        suffix, so it must provide the authenticated final hidden row used by
        the first head scan.  The emitted-token scan/decode loop is shared with
        the ordinary fresh-prefill path.
        """

        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        if not isinstance(prefill_tokenwise, bool):
            raise TypeError("prefill_tokenwise must be a boolean")
        if not isinstance(retain_final_state, bool):
            raise TypeError("retain_final_state must be a boolean")
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
        if restored_prefix_length is None:
            if restored_seed_hidden is not None:
                raise ValueError("restored_seed_hidden requires restored_prefix_length")
        else:
            if (
                isinstance(restored_prefix_length, bool)
                or not isinstance(restored_prefix_length, int)
                or not 1 <= restored_prefix_length <= prompt.shape[1]
            ):
                raise ValueError(
                    "restored_prefix_length must identify a non-empty prompt prefix"
                )
            if self._state_poisoned:
                raise Qwen38RuntimeError(
                    "decoder state is poisoned; call reset_state()"
                )
            if self._next_position != restored_prefix_length:
                raise Qwen38RuntimeError(
                    "restored prefix cursor differs from generation prompt"
                )
            if self._state_batch_size != 1:
                raise Qwen38RuntimeError("restored prefix must own batch size one")

        source = self.pager.source
        start_bytes = self._metric(source, "network_or_source_body_bytes")
        start_linears = self._metric(self.pager, "linear_calls")
        started = time.perf_counter()
        if restored_prefix_length is None:
            hidden, forwards = self.prefill(
                prompt,
                tokenwise=prefill_tokenwise,
                reset=True,
                progress=progress,
            )
        else:
            suffix = prompt[:, restored_prefix_length:]
            if suffix.shape[1]:
                if restored_seed_hidden is not None:
                    raise ValueError(
                        "a suffix restore may not provide an exact-prefix seed"
                    )
                hidden, forwards = self.prefill(
                    suffix,
                    tokenwise=prefill_tokenwise,
                    reset=False,
                    progress=progress,
                )
            else:
                if not isinstance(restored_seed_hidden, torch.Tensor):
                    raise TypeError(
                        "an exact-prefix restore requires a hidden-state tensor"
                    )
                if (
                    not restored_seed_hidden.is_floating_point()
                    or tuple(restored_seed_hidden.shape) != (1, 1, self.config.dim)
                    or restored_seed_hidden.dtype != self.pager.compute_dtype
                    or not self._on_pager_device(restored_seed_hidden)
                    or not bool(torch.isfinite(restored_seed_hidden).all().item())
                ):
                    raise Qwen38RuntimeError(
                        "restored exact-prefix seed differs from model execution"
                    )
                hidden = restored_seed_hidden.detach()
                forwards = ()
        generated: list[int] = []
        stopped_on_eos = False
        forward_count = len(forwards)
        first_token_seconds: float | None = None
        for step in range(max_new_tokens):
            values, token_ids = self.pager.topk_logits(
                hidden[:, -1],
                k=1,
                name=self.output_head_name,
                block_rows=head_block_rows,
                progress=head_progress,
            )
            token_id = int(token_ids[0, 0].item())
            generated.append(token_id)
            if first_token_seconds is None:
                first_token_seconds = time.perf_counter() - started
            if progress is not None:
                progress(
                    {
                        "event": "generated_token",
                        "step": step,
                        "token_id": token_id,
                        "logit": float(values[0, 0].item()),
                    }
                )
            stopped = token_id in eos
            final_output = stopped or step + 1 == max_new_tokens
            # Stateful callers retain the historical contract.  One-shot chat
            # callers release the model immediately and therefore must not
            # stream the complete checkpoint once more for a state they throw
            # away without reading its logits.
            if retain_final_state or not final_output:
                hidden, _evidence = self.decode([[token_id]], progress=progress)
                forward_count += 1
            if stopped:
                stopped_on_eos = True
                break

        prompt_ids = tuple(
            int(value) for value in prompt[0].detach().to("cpu").tolist()
        )
        elapsed = time.perf_counter() - started
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
            seconds=elapsed,
            state_bytes=self.state_bytes,
            stopped_on_eos=stopped_on_eos,
            final_state_committed=retain_final_state,
            time_to_first_token_seconds=(
                elapsed if first_token_seconds is None else first_token_seconds
            ),
            output_tokens_per_second=(len(generated) / elapsed if elapsed else 0.0),
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
        staged_native_evidence: list[NativeHeadCrsaEvidence] = []
        mode = (
            "off" if self.graft is None else str(getattr(self.graft, "mode", "active"))
        )
        try:
            for layer in range(self.config.n_layers):
                hidden, _ = self.forward_prefill_layer(
                    hidden,
                    ids,
                    layer=layer,
                    token_mask=mask,
                    _native_head_crsa_observer=staged_native_evidence.append,
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
        self._emit_native_head_crsa_evidence(staged_native_evidence)
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
        }
        if not self.config.tie_word_embeddings:
            required[self.HEAD_NAME] = (
                self.config.vocab_size,
                self.config.dim,
            )
        allowed_dtypes: dict[str, frozenset[str]] = {}
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
                # Published Qwen3.8-27B stores these controls in BF16 while
                # Qwen3.5-0.8B stores them in F32.  Both are consumed in the
                # runtime's explicit float32 recurrence path.
                allowed_dtypes[f"{attn}.A_log"] = frozenset({"BF16", "F32"})
                allowed_dtypes[f"{attn}.norm.weight"] = frozenset({"BF16", "F32"})

        for name, shape in required.items():
            entry = entries.get(name)
            if entry is None:
                errors.append(f"missing {name}")
                continue
            actual_shape = tuple(int(value) for value in entry.get("shape", ()))
            if actual_shape != shape:
                errors.append(f"{name}: shape {actual_shape}, expected {shape}")
            actual_dtype = str(entry.get("dtype", "")).upper()
            wanted_dtypes = allowed_dtypes.get(name, frozenset({"BF16"}))
            if actual_dtype not in wanted_dtypes:
                errors.append(
                    f"{name}: dtype {actual_dtype or 'missing'}, expected "
                    f"{'/'.join(sorted(wanted_dtypes))}"
                )
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
            "tie_word_embeddings": self.config.tie_word_embeddings,
            "output_head_tensor": self.output_head_name,
            "vision_excluded": True,
            "mtp_excluded": True,
        }


__all__ = [
    "GenerationEvidence",
    "LAYER_BOUNDARY_STAGES",
    "LayerBoundaryObserver",
    "PrefillEvidence",
    "Qwen38RuntimeError",
    "StatefulEvidence",
    "StatefulLayerRangeEvidence",
    "StatefulLayerRangeResult",
    "StreamedQwen38",
]
