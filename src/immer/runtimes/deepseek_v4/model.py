"""Executable layer-paged DeepSeek-V4 main decoder.

This module begins with an exact one-token-context path.  For that context the
compressed KV/index branches have an empty key set, so omitting their pure,
unused projections is mathematically equivalent to the official forward while
avoiding hundreds of megabytes of irrelevant reads.  The diagnostic path stays
available for fault isolation; the general path owns one native attention
state per layer and supports start-zero prefill plus contiguous decode.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
import os
from collections.abc import Callable
from typing import Any

import numpy as np

from .config import DeepSeekV4Config
from .kernels import (
    apply_rotary_emb,
    hadamard_transform,
    hc_head,
    hc_post,
    hc_pre,
    precompute_freqs_cis,
    rms_norm,
    sparse_attention,
)
from .pager import DeepSeekWeightPager
from .provenance import runtime_dependency_versions, runtime_source_manifest
from .quantization import quantize_dequantize_fp4, quantize_dequantize_fp8
from .route_markov import (
    LayerMarkovExpertPredictor,
    LayerMicroWindowPlan,
    plan_micro_window_prefetch,
)
from .snapshot import (
    DeepSeekV4SnapshotError,
    SnapshotLimits,
    SnapshotTensor,
    read_snapshot,
    write_snapshot,
)
from .stateful import DeepSeekV4Compressor, DeepSeekV4Indexer, NativeAttentionState


class DeepSeekV4RuntimeError(RuntimeError):
    """The streamed checkpoint cannot complete the requested model forward."""


@dataclass(frozen=True, slots=True)
class OneTokenEvidence:
    token_id: int
    layers_executed: int
    checkpoint_layers: int
    complete_layer_stack: bool
    context_mode: str
    stateful_kv_cache: bool
    source_body_bytes: int
    linear_calls: int
    seconds: float
    selected_experts: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class StatefulEvidence:
    start_pos: int
    end_pos: int
    input_token_ids: tuple[tuple[int, ...], ...]
    layers_executed: int
    checkpoint_layers: int
    complete_layer_stack: bool
    context_mode: str
    stateful_kv_cache: bool
    source_body_bytes: int
    linear_calls: int
    seconds: float
    attention_state_bytes: int
    selected_experts: tuple[tuple[tuple[int, ...], ...], ...]
    graft_mode: str
    graft_history_tokens: int


@dataclass(frozen=True, slots=True)
class GenerationEvidence:
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    context_mode: str
    stateful_kv_cache: bool
    general_generation: bool
    prefill_mode: str
    forward_passes: int
    source_body_bytes: int
    linear_calls: int
    seconds: float
    attention_state_bytes: int
    stopped_on_eos: bool


class StreamedDeepSeekV4:
    """Official V4 block math with one matrix resident at a time."""

    ATTENTION_QAT_POLICY = "v4-native-fp8-kv+fp4-hadamard-indexer/v1"

    def __init__(
        self,
        config: DeepSeekV4Config,
        pager: DeepSeekWeightPager,
        *,
        graft: Any | None = None,
        graft_layer: int | None = None,
        max_batch_size: int = 1,
        max_seq_len: int = 512,
        route_predictor: LayerMarkovExpertPredictor | None = None,
        route_prefetch_window_rows: int = 2,
        route_prefetch_k: int | None = None,
        route_prefetch_alpha: float = 1.0,
        route_prefetch_direct_max_rows: int = 8,
        route_prefetch_min_confidence: float = 0.0,
    ) -> None:
        self.config = config
        self.pager = pager
        self.torch = pager.torch
        self.graft = graft
        self.graft_layer = graft_layer
        if graft_layer is not None and not 0 <= graft_layer < config.n_layers:
            raise ValueError("graft_layer outside decoder depth")
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
            raise ValueError(
                f"max_seq_len={max_seq_len} exceeds checkpoint "
                f"max_position_embeddings={config.max_position_embeddings}"
            )
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        if route_predictor is not None:
            if not isinstance(route_predictor, LayerMarkovExpertPredictor):
                raise TypeError("route_predictor must be a LayerMarkovExpertPredictor")
            if route_predictor.n_experts != config.n_routed_experts:
                raise ValueError(
                    "route predictor expert inventory does not match the checkpoint"
                )
        if (
            isinstance(route_prefetch_window_rows, bool)
            or not isinstance(route_prefetch_window_rows, int)
            or route_prefetch_window_rows < 1
        ):
            raise ValueError("route_prefetch_window_rows must be positive")
        if route_prefetch_k is not None and (
            isinstance(route_prefetch_k, bool)
            or not isinstance(route_prefetch_k, int)
            or not 1 <= route_prefetch_k <= config.n_routed_experts
        ):
            raise ValueError("route_prefetch_k must be inside the expert inventory")
        if (
            isinstance(route_prefetch_alpha, bool)
            or not isinstance(route_prefetch_alpha, (int, float))
            or not math.isfinite(route_prefetch_alpha)
            or route_prefetch_alpha <= 0
        ):
            raise ValueError("route_prefetch_alpha must be finite and positive")
        if (
            isinstance(route_prefetch_direct_max_rows, bool)
            or not isinstance(route_prefetch_direct_max_rows, int)
            or route_prefetch_direct_max_rows < 1
        ):
            raise ValueError("route_prefetch_direct_max_rows must be positive")
        if (
            isinstance(route_prefetch_min_confidence, bool)
            or not isinstance(route_prefetch_min_confidence, (int, float))
            or not math.isfinite(route_prefetch_min_confidence)
            or not 0 <= route_prefetch_min_confidence <= 1
        ):
            raise ValueError("route_prefetch_min_confidence must be inside [0, 1]")
        self.route_predictor = route_predictor
        self.route_prefetch_window_rows = route_prefetch_window_rows
        self.route_prefetch_k = route_prefetch_k
        self.route_prefetch_alpha = float(route_prefetch_alpha)
        self.route_prefetch_direct_max_rows = route_prefetch_direct_max_rows
        self.route_prefetch_min_confidence = float(route_prefetch_min_confidence)
        self._route_prefetch_plan: LayerMicroWindowPlan | None = None
        self._route_prefetch_window_index = 0
        self._route_prefetch_reservoir: Any | None = None
        self._route_prefetch_aggregate = False
        self._route_prefetch_stats = {
            "aggregate_bindings": 0,
            "aggregate_plans": 0,
            "direct_bindings": 0,
            "direct_plans": 0,
            "confidence_skips": 0,
            "plans": 0,
        }
        self._attention_states: list[NativeAttentionState | None] = [
            None for _ in range(config.n_layers)
        ]
        self._attention_freqs: list[Any | None] = [None for _ in range(config.n_layers)]
        self._next_position = 0
        self._state_poisoned = False
        self._graft_history: Any | None = None

    def checkpoint_preflight(
        self, *, exhaustive_experts: bool = True
    ) -> dict[str, Any]:
        """Validate tensor presence and critical storage formats without payload reads."""

        source = self.pager.source
        inventory = source.inventory()
        entries = {entry["name"]: entry for entry in inventory.get("tensors", [])}
        required: set[str] = set()
        errors: list[str] = []
        fp8 = {"F8_E4M3", "F8_E4M3FN"}

        def expect(name: str, dtypes: set[str], shape: tuple[int, ...]) -> None:
            required.add(name)
            entry = entries.get(name)
            if entry is None:
                errors.append(f"missing {name}")
                return
            actual_dtype = str(entry.get("dtype", "")).upper()
            actual_shape = tuple(int(value) for value in entry.get("shape", ()))
            if actual_dtype not in dtypes:
                errors.append(
                    f"{name}: dtype {actual_dtype}, expected {'/'.join(sorted(dtypes))}"
                )
            if actual_shape != shape:
                errors.append(f"{name}: shape {actual_shape}, expected {shape}")

        def fp8_matrix(name: str, out_dim: int, in_dim: int) -> None:
            expect(f"{name}.weight", fp8, (out_dim, in_dim))
            expect(
                f"{name}.scale",
                {"F8_E8M0"},
                ((out_dim + 127) // 128, (in_dim + 127) // 128),
            )

        def fp4_matrix(name: str, out_dim: int, in_dim: int) -> None:
            expect(f"{name}.weight", {"I8"}, (out_dim, in_dim // 2))
            expect(f"{name}.scale", {"F8_E8M0"}, (out_dim, in_dim // 32))

        config = self.config
        expect("embed.weight", {"BF16"}, (config.vocab_size, config.dim))
        expect("norm.weight", {"BF16"}, (config.dim,))
        expect("head.weight", {"BF16"}, (config.vocab_size, config.dim))
        expect("hc_head_fn", {"F32"}, (config.hc_mult, config.hc_mult * config.dim))
        expect("hc_head_base", {"F32"}, (config.hc_mult,))
        expect("hc_head_scale", {"F32"}, (1,))
        mix_hc = (2 + config.hc_mult) * config.hc_mult
        for layer in range(self.config.n_layers):
            prefix = f"layers.{layer}"
            attn = f"{prefix}.attn"
            expect(f"{attn}.attn_sink", {"F32"}, (config.n_heads,))
            expect(f"{attn}.q_norm.weight", {"BF16"}, (config.q_lora_rank,))
            expect(f"{attn}.kv_norm.weight", {"BF16"}, (config.head_dim,))
            expect(f"{prefix}.attn_norm.weight", {"BF16"}, (config.dim,))
            expect(f"{prefix}.ffn_norm.weight", {"BF16"}, (config.dim,))
            expect(
                f"{prefix}.ffn.gate.weight",
                {"BF16"},
                (config.n_routed_experts, config.dim),
            )
            for branch in ("attn", "ffn"):
                expect(
                    f"{prefix}.hc_{branch}_fn",
                    {"F32"},
                    (mix_hc, config.hc_mult * config.dim),
                )
                expect(f"{prefix}.hc_{branch}_base", {"F32"}, (mix_hc,))
                expect(f"{prefix}.hc_{branch}_scale", {"F32"}, (3,))
            if layer < config.n_hash_layers:
                expect(
                    f"{prefix}.ffn.gate.tid2eid",
                    {"I32", "I64"},
                    (config.vocab_size, config.n_activated_experts),
                )
            else:
                expect(
                    f"{prefix}.ffn.gate.bias",
                    {"F32"},
                    (config.n_routed_experts,),
                )
            fp8_matrix(f"{attn}.wq_a", config.q_lora_rank, config.dim)
            fp8_matrix(
                f"{attn}.wq_b", config.n_heads * config.head_dim, config.q_lora_rank
            )
            fp8_matrix(f"{attn}.wkv", config.head_dim, config.dim)
            fp8_matrix(
                f"{attn}.wo_a",
                config.o_groups * config.o_lora_rank,
                config.n_heads * config.head_dim // config.o_groups,
            )
            fp8_matrix(f"{attn}.wo_b", config.dim, config.o_groups * config.o_lora_rank)
            ratio = config.compress_ratios[layer]
            if ratio:
                coff = 2 if ratio == 4 else 1
                compressor = f"{attn}.compressor"
                expect(
                    f"{compressor}.ape",
                    {"F32"},
                    (ratio, coff * config.head_dim),
                )
                expect(f"{compressor}.norm.weight", {"BF16"}, (config.head_dim,))
                expect(
                    f"{compressor}.wkv.weight",
                    {"BF16"},
                    (coff * config.head_dim, config.dim),
                )
                expect(
                    f"{compressor}.wgate.weight",
                    {"BF16"},
                    (coff * config.head_dim, config.dim),
                )
                if ratio == 4:
                    indexer = f"{attn}.indexer"
                    fp8_matrix(
                        f"{indexer}.wq_b",
                        config.index_n_heads * config.index_head_dim,
                        config.q_lora_rank,
                    )
                    expect(
                        f"{indexer}.weights_proj.weight",
                        {"BF16"},
                        (config.index_n_heads, config.dim),
                    )
                    index_compressor = f"{indexer}.compressor"
                    expect(
                        f"{index_compressor}.ape",
                        {"F32"},
                        (ratio, 2 * config.index_head_dim),
                    )
                    expect(
                        f"{index_compressor}.norm.weight",
                        {"BF16"},
                        (config.index_head_dim,),
                    )
                    for projection in ("wkv", "wgate"):
                        expect(
                            f"{index_compressor}.{projection}.weight",
                            {"BF16"},
                            (2 * config.index_head_dim, config.dim),
                        )
            for projection, out_dim, in_dim in (
                ("w1", config.moe_inter_dim, config.dim),
                ("w2", config.dim, config.moe_inter_dim),
                ("w3", config.moe_inter_dim, config.dim),
            ):
                fp8_matrix(f"{prefix}.ffn.shared_experts.{projection}", out_dim, in_dim)
            expert_ids = range(config.n_routed_experts) if exhaustive_experts else (0,)
            for expert_id in expert_ids:
                for projection, out_dim, in_dim in (
                    ("w1", config.moe_inter_dim, config.dim),
                    ("w2", config.dim, config.moe_inter_dim),
                    ("w3", config.moe_inter_dim, config.dim),
                ):
                    fp4_matrix(
                        f"{prefix}.ffn.experts.{expert_id}.{projection}",
                        out_dim,
                        in_dim,
                    )
        if errors:
            preview = "; ".join(errors[:8])
            raise DeepSeekV4RuntimeError(
                f"checkpoint violates {len(errors)} required tensor contracts: {preview}"
            )
        required_bytes = sum(
            int(entries[name]["offset_in_shard"][1])
            - int(entries[name]["offset_in_shard"][0])
            for name in required
        )
        return {
            "required_tensors": len(required),
            "required_payload_bytes": required_bytes,
            "exhaustive_experts": bool(exhaustive_experts),
            "inventory_tensors": len(entries),
            "inventory_fingerprint": self.pager.source.metrics().get(
                "inventory_source_fingerprint"
            ),
        }

    def _control(self, name: str, *, dtype: Any | None = None) -> Any:
        return self.pager.tensor_torch(name, dtype=dtype, device=self.pager.device)

    def _norm(self, x: Any, name: str) -> Any:
        weight = self._control(name, dtype=self.torch.float32)
        return rms_norm(x, weight, self.config.norm_eps)

    def _hc_pre(self, x: Any, base: str) -> tuple[Any, Any, Any]:
        return hc_pre(
            x,
            self._control(f"{base}_fn", dtype=self.torch.float32),
            self._control(f"{base}_scale", dtype=self.torch.float32),
            self._control(f"{base}_base", dtype=self.torch.float32),
            hc_mult=self.config.hc_mult,
            sinkhorn_iters=self.config.hc_sinkhorn_iters,
            eps=self.config.hc_eps,
            norm_eps=self.config.norm_eps,
        )

    def _qat_kv(self, kv: Any) -> Any:
        rd = self.config.rope_head_dim
        nope = np.ascontiguousarray(
            kv[..., :-rd].detach().to("cpu", self.torch.float32).numpy()
        )
        nope = quantize_dequantize_fp8(nope, block_size=64)
        quantized = self.torch.from_numpy(nope).to(kv.device, dtype=kv.dtype)
        return self.torch.cat((quantized, kv[..., -rd:]), dim=-1)

    def _qat_indexer(self, value: Any) -> Any:
        rotated = hadamard_transform(value)
        cpu = np.ascontiguousarray(
            rotated.detach().to("cpu", self.torch.float32).numpy()
        )
        quantized = quantize_dequantize_fp4(cpu, block_size=32)
        return self.torch.from_numpy(quantized).to(value.device, dtype=value.dtype)

    def _state_linear(self, x: Any, name: str) -> Any:
        # The published loader promotes compressor BF16 matrices to FP32 and
        # explicitly projects x.float().  Other callbacks retain their native
        # BF16/FP8 execution contract through the pager.
        if ".compressor." in name and name.endswith((".wkv", ".wgate")):
            return self.pager.linear(
                x,
                name,
                activation_quantization=False,
                output_dtype=self.torch.float32,
                compute_dtype=self.torch.float32,
            )
        return self.pager.linear(x, name)

    def _state_qat(self, value: Any, mode: str) -> Any:
        if mode == "compressed-kv":
            return self._qat_kv(value)
        if mode == "indexer":
            return self._qat_indexer(value)
        raise ValueError(f"unsupported attention QAT mode: {mode!r}")

    def _attention_state(self, layer: int) -> tuple[NativeAttentionState, Any]:
        existing = self._attention_states[layer]
        freqs = self._attention_freqs[layer]
        if existing is not None:
            assert freqs is not None
            return existing, freqs

        config = self.config
        base = f"layers.{layer}.attn"
        ratio = config.compress_ratios[layer]
        rope_base = config.compress_rope_theta if ratio else config.rope_theta
        original = config.original_seq_len if ratio else 0
        freqs = precompute_freqs_cis(
            config.rope_head_dim,
            self.max_seq_len,
            original,
            rope_base,
            config.rope_factor,
            config.beta_fast,
            config.beta_slow,
        )
        compressor = None
        indexer = None
        if ratio:
            compressor_prefix = f"{base}.compressor"
            compressor = DeepSeekV4Compressor(
                prefix=compressor_prefix,
                compress_ratio=ratio,
                head_dim=config.head_dim,
                rope_head_dim=config.rope_head_dim,
                max_batch_size=self.max_batch_size,
                max_seq_len=self.max_seq_len,
                ape=self._control(f"{compressor_prefix}.ape", dtype=self.torch.float32),
                freqs_cis=freqs,
                linear=self._state_linear,
                rms=self._norm,
                qat=self._state_qat,
            )
            if ratio == 4:
                index_prefix = f"{base}.indexer"
                index_compressor_prefix = f"{index_prefix}.compressor"
                index_compressor = DeepSeekV4Compressor(
                    prefix=index_compressor_prefix,
                    compress_ratio=ratio,
                    head_dim=config.index_head_dim,
                    rope_head_dim=config.rope_head_dim,
                    max_batch_size=self.max_batch_size,
                    max_seq_len=self.max_seq_len,
                    ape=self._control(
                        f"{index_compressor_prefix}.ape", dtype=self.torch.float32
                    ),
                    freqs_cis=freqs,
                    linear=self._state_linear,
                    rms=self._norm,
                    qat=self._state_qat,
                    rotate=True,
                )
                indexer = DeepSeekV4Indexer(
                    prefix=index_prefix,
                    compressor=index_compressor,
                    n_heads=config.index_n_heads,
                    head_dim=config.index_head_dim,
                    rope_head_dim=config.rope_head_dim,
                    index_topk=config.index_topk,
                    freqs_cis=freqs,
                    linear=self._state_linear,
                    qat=self._state_qat,
                )
        state = NativeAttentionState(
            max_batch_size=self.max_batch_size,
            max_seq_len=self.max_seq_len,
            window_size=config.window_size,
            head_dim=config.head_dim,
            compress_ratio=ratio,
            compressor=compressor,
            indexer=indexer,
        )
        self._attention_states[layer] = state
        self._attention_freqs[layer] = freqs
        return state, freqs

    @property
    def next_position(self) -> int:
        return self._next_position

    @property
    def attention_state_bytes(self) -> int:
        return sum(
            state.state_nbytes for state in self._attention_states if state is not None
        )

    def reset_state(self, *, release: bool = False) -> None:
        """Reset every per-layer cache after a request or failed forward."""

        if self._route_prefetch_plan is not None or self._route_prefetch_reservoir:
            self._cancel_route_prefetch()
        clear_priorities = getattr(self.pager.source, "clear_cache_priorities", None)
        if callable(clear_priorities):
            clear_priorities()
        for state in self._attention_states:
            if state is not None:
                state.reset(release=release)
        if release:
            self._attention_states = [None for _ in range(self.config.n_layers)]
            self._attention_freqs = [None for _ in range(self.config.n_layers)]
        self._next_position = 0
        self._state_poisoned = False
        self._graft_history = None

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
            raise DeepSeekV4SnapshotError(
                "runtime identity cannot be represented as canonical JSON"
            ) from exc
        return hashlib.sha256(encoded).hexdigest()

    def _graft_snapshot_identity(self) -> dict[str, Any]:
        if self.graft is None:
            return {"kind": "none", "layer": self.graft_layer}
        graft = self.graft
        state_dict = getattr(graft, "state_dict", None)
        if callable(state_dict):
            learned = state_dict()
            if learned:
                raise DeepSeekV4SnapshotError(
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
            raise DeepSeekV4SnapshotError(
                "graft lacks the complete serialisable identity contract"
            )
        spec = getattr(graft, "spec")
        if not is_dataclass(spec):
            raise DeepSeekV4SnapshotError("graft AttentionSpec is not a dataclass")
        alpha = float(getattr(graft, "alpha"))
        if not math.isfinite(alpha):
            raise DeepSeekV4SnapshotError("graft alpha must be finite")
        return {
            "kind": f"{type(graft).__module__}.{type(graft).__qualname__}",
            "layer": self.graft_layer,
            "mode": str(getattr(graft, "mode")),
            "alpha": alpha,
            "heads": int(getattr(graft, "heads")),
            "max_history": int(getattr(graft, "max_history")),
            "shuffle_seed": int(getattr(graft, "shuffle_seed")),
            "attention_spec": asdict(spec),
        }

    def _route_prefetch_snapshot_identity(self) -> dict[str, Any]:
        predictor = self.route_predictor
        if predictor is None:
            return {"kind": "none"}
        return {
            "alpha": self.route_prefetch_alpha,
            "k": self.route_prefetch_k,
            "kind": "token-row-markov",
            "snapshot_sha256": predictor.snapshot_sha256,
            "window_rows": self.route_prefetch_window_rows,
            "direct_max_rows": self.route_prefetch_direct_max_rows,
            "min_confidence": self.route_prefetch_min_confidence,
        }

    def _snapshot_identity(self) -> dict[str, Any]:
        source = self.pager.source
        # Inventory acquisition is metadata-only and establishes the immutable
        # source fingerprint before any continuation can be published/loaded.
        source.inventory()
        metrics = source.metrics()
        fingerprint = metrics.get("inventory_source_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise DeepSeekV4SnapshotError(
                "tensor source has no verified inventory fingerprint"
            )
        source_kind = f"{type(source).__module__}.{type(source).__qualname__}"
        repo_id = metrics.get("repo_id", getattr(source, "repo_id", source_kind))
        revision = metrics.get("revision", getattr(source, "revision", "fixture"))
        if not isinstance(repo_id, str) or not repo_id:
            raise DeepSeekV4SnapshotError("tensor source repo identity is invalid")
        if not isinstance(revision, str) or not revision:
            raise DeepSeekV4SnapshotError("tensor source revision identity is invalid")
        config = asdict(self.config)
        runtime_sources = runtime_source_manifest()
        runtime_dependencies = runtime_dependency_versions()
        return {
            "runtime": {
                "schema": "immer.streamed-deepseek-v4/native-stateful-v3",
                "source_sha256": self._snapshot_digest(runtime_sources),
                "sources": runtime_sources,
                "dependency_sha256": self._snapshot_digest(runtime_dependencies),
                "dependencies": runtime_dependencies,
            },
            "config": config,
            "config_sha256": self._snapshot_digest(config),
            "source": {
                "kind": source_kind,
                "repo_id": repo_id,
                "revision": revision,
                "inventory_fingerprint": fingerprint,
            },
            "execution": {
                "device": str(self.pager.device),
                "compute_dtype": str(self.pager.compute_dtype).removeprefix("torch."),
                "simulate_activation_quantization": bool(
                    self.pager.simulate_activation_quantization
                ),
                "quantized_accumulation_policy": (
                    self.pager.QUANTIZED_ACCUMULATION_POLICY
                ),
                "attention_qat_policy": self.ATTENTION_QAT_POLICY,
                "expert_prefetch_policy": self.pager.expert_prefetch_policy,
                "expert_prefetch_payload_limit_bytes": (
                    self.pager.EXPERT_PREFETCH_PAYLOAD_LIMIT_BYTES
                ),
                "expert_prefetch_transport_policy": (
                    self.pager.expert_prefetch_transport_policy
                ),
                "expert_prefetch_workers": self.pager.EXPERT_PREFETCH_WORKERS,
                "expert_prefetch_active_read_limit": (
                    self.pager.EXPERT_PREFETCH_ACTIVE_READ_LIMIT
                ),
                "expert_prefetch_max_outstanding": (
                    self.pager.EXPERT_PREFETCH_MAX_OUTSTANDING
                ),
                "expert_prefetch_max_experts": (self.pager.EXPERT_PREFETCH_MAX_EXPERTS),
                "expert_prefetch_resident_limit_bytes": (
                    self.pager.EXPERT_PREFETCH_RESIDENT_LIMIT_BYTES
                ),
                "expert_range_coalesce_max_experts": (
                    self.pager.expert_range_coalesce_max_experts
                ),
                "expert_range_coalesce_max_gap_bytes": (
                    self.pager.EXPERT_RANGE_COALESCE_MAX_GAP_BYTES
                ),
                "expert_reservoir_policy": self.pager.EXPERT_RESERVOIR_POLICY,
                "expert_reservoir_budget_bytes": (
                    self.pager.expert_reservoir_budget_bytes
                ),
                "expert_reservoir_workers": self.pager.expert_reservoir_workers,
                "route_prefetch": self._route_prefetch_snapshot_identity(),
                "source_transport_policy": str(
                    metrics.get("transport_policy", "unreported")
                ),
                "source_transport_connection_limit": int(
                    metrics.get("transport_connection_limit", 0)
                ),
                "max_batch_size": self.max_batch_size,
                "max_seq_len": self.max_seq_len,
                "max_position_embeddings": self.config.max_position_embeddings,
            },
            "graft": self._graft_snapshot_identity(),
        }

    def _snapshot_model_state(
        self,
    ) -> tuple[dict[str, Any], dict[str, SnapshotTensor]]:
        if not 0 <= self._next_position <= self.max_seq_len:
            raise DeepSeekV4SnapshotError("model cursor exceeds its context bound")
        if self._state_poisoned and self._next_position != 0:
            raise DeepSeekV4SnapshotError("poisoned model has a non-zero cursor")
        if self._route_prefetch_plan is not None or self._route_prefetch_reservoir:
            raise DeepSeekV4SnapshotError(
                "cannot snapshot while a causal route prefetch is active"
            )
        tensors: dict[str, SnapshotTensor] = {}
        layers: list[dict[str, Any]] = []
        for layer, attention in enumerate(self._attention_states):
            if attention is None:
                continue
            if attention.next_position != self._next_position:
                raise DeepSeekV4SnapshotError(
                    f"layer {layer} attention cursor does not match the model"
                )
            # Allocated buffers retained by reset_state(release=False) are a
            # performance detail, not continuation state.  Cursor zero has one
            # canonical tensor-free representation.
            if self._next_position == 0:
                attention._snapshot_state(f"attention.layer_{layer:03d}", {})
                continue
            layers.append(
                {
                    "layer": layer,
                    "state": attention._snapshot_state(
                        f"attention.layer_{layer:03d}", tensors
                    ),
                }
            )
        if self._next_position and len(layers) != self.config.n_layers:
            raise DeepSeekV4SnapshotError(
                "active model does not have state for every decoder layer"
            )

        graft_history_name = None
        if self._next_position == 0 and self._graft_history is not None:
            raise DeepSeekV4SnapshotError("zero-cursor model retains graft history")
        if self._graft_history is not None:
            history = self._graft_history
            expected = (
                history.ndim == 4
                and history.shape[0] <= self.max_batch_size
                and history.shape[1] == self._next_position
                and history.shape[2] == self.config.hc_mult
                and history.shape[3] == self.config.dim
            )
            if not expected:
                raise DeepSeekV4SnapshotError(
                    "graft history shape/cursor is inconsistent"
                )
            graft_history_name = "model.graft_history"
            tensors[graft_history_name] = SnapshotTensor(history)
        graft_identity = self._graft_snapshot_identity()
        graft_is_active = (
            graft_identity.get("kind") != "none"
            and graft_identity.get("layer") is not None
            and graft_identity.get("mode") != "off"
            and float(graft_identity.get("alpha", 0.0)) != 0.0
        )
        if self._next_position and graft_is_active != (graft_history_name is not None):
            raise DeepSeekV4SnapshotError(
                "active graft and graft-history presence are inconsistent"
            )
        if self._state_poisoned and graft_history_name is not None:
            raise DeepSeekV4SnapshotError("poisoned model retains graft history")
        return (
            {
                "next_position": self._next_position,
                "state_poisoned": self._state_poisoned,
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
    ) -> dict[str, Any]:
        """Atomically save the exact native decoder continuation state.

        ``path`` is the JSON commit-point manifest.  Its NPZ payload is stored
        beside it under a content-addressed filename.  Neither file contains
        pickle or executable objects.
        """

        limits = self._snapshot_limits(max_bytes, max_tensors)
        state, tensors = self._snapshot_model_state()
        result = write_snapshot(
            path,
            identity=self._snapshot_identity(),
            state=state,
            tensors=tensors,
            limits=limits,
        )
        return {
            **result,
            "next_position": self._next_position,
            "state_poisoned": self._state_poisoned,
        }

    @staticmethod
    def _snapshot_state_int(
        value: Any,
        name: str,
        *,
        maximum: int,
    ) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value > maximum
        ):
            raise DeepSeekV4SnapshotError(f"snapshot {name} is outside its bound")
        return value

    def load_state(
        self,
        path: str | os.PathLike[str],
        *,
        max_bytes: int = 2 * 1024**3,
        max_tensors: int = 2048,
        max_restore_peak_bytes: int = 4 * 1024**3,
    ) -> dict[str, Any]:
        """Transactionally restore a bounded native decoder continuation.

        ``max_restore_peak_bytes`` bounds snapshot-owned tensors plus the
        currently resident mutable state and one in-flight device transfer.
        Admission is decided from verified headers before tensor allocation.
        """

        limits = self._snapshot_limits(max_bytes, max_tensors)
        resident_bytes = self.attention_state_bytes
        if self._graft_history is not None:
            resident_bytes += (
                self._graft_history.numel() * self._graft_history.element_size()
            )
        loaded = read_snapshot(
            path,
            expected_identity=self._snapshot_identity(),
            limits=limits,
            resident_bytes=resident_bytes,
            max_restore_peak_bytes=max_restore_peak_bytes,
        )
        state = loaded.state
        next_position = self._snapshot_state_int(
            state.get("next_position"),
            "next_position",
            maximum=self.max_seq_len,
        )
        poisoned = state.get("state_poisoned")
        if not isinstance(poisoned, bool):
            raise DeepSeekV4SnapshotError("snapshot poison latch must be boolean")
        if poisoned and next_position:
            raise DeepSeekV4SnapshotError("poisoned snapshot has a non-zero cursor")
        expected_scalars = {
            "max_batch_size": self.max_batch_size,
            "max_seq_len": self.max_seq_len,
            "max_position_embeddings": self.config.max_position_embeddings,
        }
        if any(state.get(key) != value for key, value in expected_scalars.items()):
            raise DeepSeekV4SnapshotError("snapshot model bounds do not match runtime")

        raw_layers = state.get("attention_layers")
        if not isinstance(raw_layers, list) or len(raw_layers) > self.config.n_layers:
            raise DeepSeekV4SnapshotError("snapshot attention-layer table is invalid")
        layer_metadata: dict[int, Any] = {}
        referenced: set[str] = set()
        for row in raw_layers:
            if not isinstance(row, dict):
                raise DeepSeekV4SnapshotError("snapshot attention-layer row is invalid")
            layer = row.get("layer")
            if (
                isinstance(layer, bool)
                or not isinstance(layer, int)
                or not 0 <= layer < self.config.n_layers
                or layer in layer_metadata
            ):
                raise DeepSeekV4SnapshotError(
                    "snapshot layer index is invalid or duplicate"
                )
            layer_metadata[layer] = row.get("state")
        if next_position and len(layer_metadata) != self.config.n_layers:
            raise DeepSeekV4SnapshotError(
                "active snapshot omits one or more decoder-layer states"
            )
        if not next_position and (layer_metadata or loaded.tensors):
            raise DeepSeekV4SnapshotError(
                "zero-cursor snapshot must be tensor- and layer-free"
            )

        def collect_tensor_references(value: Any) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    if key in {"storage", "kv_state", "score_state"}:
                        if child is not None:
                            if not isinstance(child, str):
                                raise DeepSeekV4SnapshotError(
                                    "snapshot tensor reference is invalid"
                                )
                            referenced.add(child)
                    else:
                        collect_tensor_references(child)
            elif isinstance(value, list):
                for child in value:
                    collect_tensor_references(child)

        collect_tensor_references(raw_layers)
        expected_references: set[str] = set()
        layer_references: dict[int, set[str]] = {}
        for layer, raw_attention in layer_metadata.items():
            if not isinstance(raw_attention, dict):
                raise DeepSeekV4SnapshotError("snapshot attention state is invalid")
            layer_prefix = f"attention.layer_{layer:03d}"
            current_layer_references: set[str] = set()

            def expect_reference(container: Any, key: str, expected: str) -> None:
                if not isinstance(container, dict):
                    raise DeepSeekV4SnapshotError("snapshot cache metadata is invalid")
                actual = container.get(key)
                if actual is not None:
                    if actual != expected:
                        raise DeepSeekV4SnapshotError(
                            "snapshot tensor reference does not match its state role"
                        )
                    expected_references.add(expected)
                    current_layer_references.add(expected)

            expect_reference(
                raw_attention.get("local"),
                "storage",
                f"{layer_prefix}.local.storage",
            )
            for key, suffix in (
                ("compressor", "compressor"),
                ("indexer_compressor", "indexer_compressor"),
            ):
                compressor = raw_attention.get(key)
                if compressor is None:
                    continue
                expect_reference(
                    compressor, "kv_state", f"{layer_prefix}.{suffix}.kv_state"
                )
                expect_reference(
                    compressor,
                    "score_state",
                    f"{layer_prefix}.{suffix}.score_state",
                )
                expect_reference(
                    compressor.get("cache") if isinstance(compressor, dict) else None,
                    "storage",
                    f"{layer_prefix}.{suffix}.cache.storage",
                )
            layer_references[layer] = current_layer_references
        history_name = state.get("graft_history")
        if history_name is not None:
            if history_name != "model.graft_history":
                raise DeepSeekV4SnapshotError("graft history reference is invalid")
            referenced.add(history_name)
            expected_references.add(history_name)
        if referenced != expected_references or referenced != set(loaded.tensors):
            raise DeepSeekV4SnapshotError(
                "snapshot contains missing or unreferenced tensor payloads"
            )
        descriptors = {
            row.get("name"): row
            for row in loaded.manifest["body"].get("tensors", [])
            if isinstance(row, dict)
        }
        for name, tensor in loaded.tensors.items():
            descriptor = descriptors.get(name)
            if not isinstance(descriptor, dict):
                raise DeepSeekV4SnapshotError("snapshot tensor descriptor is missing")
            is_score_state = name.endswith(".score_state")
            expected_policy = "finite_or_neg_inf" if is_score_state else "finite"
            if descriptor.get("finite_policy") != expected_policy:
                raise DeepSeekV4SnapshotError(
                    "snapshot tensor finite policy does not match its state role"
                )
            expected_dtype = (
                self.torch.float32
                if is_score_state or name.endswith(".kv_state")
                else self.pager.compute_dtype
            )
            if tensor.dtype != expected_dtype:
                raise DeepSeekV4SnapshotError(
                    "snapshot tensor dtype does not match its state role"
                )

        graft_identity = self._graft_snapshot_identity()
        graft_is_active = (
            graft_identity.get("kind") != "none"
            and graft_identity.get("layer") is not None
            and graft_identity.get("mode") != "off"
            and float(graft_identity.get("alpha", 0.0)) != 0.0
        )
        history: Any | None = None
        if history_name is not None:
            history = loaded.tensors[history_name]
            expected_shape = (
                history.ndim == 4
                and 0 < history.shape[0] <= self.max_batch_size
                and history.shape[1] == next_position
                and history.shape[2] == self.config.hc_mult
                and history.shape[3] == self.config.dim
            )
            if not expected_shape:
                raise DeepSeekV4SnapshotError(
                    "graft history shape/cursor is inconsistent"
                )
        if next_position and graft_is_active != (history is not None):
            raise DeepSeekV4SnapshotError(
                "active graft and graft-history presence are inconsistent"
            )
        if poisoned and history is not None:
            raise DeepSeekV4SnapshotError("poisoned snapshot retains graft history")
        del history

        old_states = self._attention_states
        old_freqs = self._attention_freqs
        new_states: list[NativeAttentionState | None] = [
            None for _ in range(self.config.n_layers)
        ]
        new_freqs: list[Any | None] = [None for _ in range(self.config.n_layers)]
        self._attention_states = new_states
        self._attention_freqs = new_freqs
        history_device = None
        try:
            for layer in sorted(layer_metadata):
                # Transfer and relinquish CPU ownership one layer at a time.
                # _restore_snapshot adopts these unique tensors without a
                # clone, keeping the measured peak within the admitted bound.
                device_tensors: dict[str, Any] = {}
                for name in sorted(layer_references[layer]):
                    cpu_tensor = loaded.tensors.pop(name)
                    device_tensors[name] = cpu_tensor.to(device=self.pager.device)
                    del cpu_tensor
                attention, _freqs = self._attention_state(layer)
                attention._restore_snapshot(layer_metadata[layer], device_tensors)
                if attention.next_position != next_position:
                    raise DeepSeekV4SnapshotError(
                        f"layer {layer} cursor does not match model cursor"
                    )
            if history_name is not None:
                cpu_history = loaded.tensors.pop(history_name)
                history_device = cpu_history.to(device=self.pager.device).detach()
                del cpu_history
            if loaded.tensors:
                raise DeepSeekV4SnapshotError(
                    "snapshot tensor ownership transfer is incomplete"
                )
        except Exception as exc:
            self._attention_states = old_states
            self._attention_freqs = old_freqs
            if isinstance(exc, DeepSeekV4SnapshotError):
                raise
            raise DeepSeekV4SnapshotError(
                "snapshot attention state is structurally inconsistent"
            ) from exc

        self._next_position = next_position
        self._state_poisoned = poisoned
        self._graft_history = history_device
        return {
            **loaded.summary,
            "next_position": next_position,
            "state_poisoned": poisoned,
        }

    def _attention(self, x: Any, layer: int, start_pos: int) -> Any:
        """Native sliding/compressed attention with persistent per-layer state."""

        torch = self.torch
        config = self.config
        base = f"layers.{layer}.attn"
        end_pos = start_pos + x.shape[1]
        state, all_freqs = self._attention_state(layer)
        freqs = all_freqs[start_pos:end_pos]

        qr = self.pager.linear(x, f"{base}.wq_a")
        qr = self._norm(qr, f"{base}.q_norm.weight")
        q = self.pager.linear(qr, f"{base}.wq_b")
        q = q.reshape(*q.shape[:-1], config.n_heads, config.head_dim)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + config.norm_eps)
        apply_rotary_emb(q[..., -config.rope_head_dim :], freqs)

        kv = self.pager.linear(x, f"{base}.wkv")
        kv = self._norm(kv, f"{base}.kv_norm.weight")
        apply_rotary_emb(kv[..., -config.rope_head_dim :], freqs)
        kv = self._qat_kv(kv)
        assembly = state.assemble(kv, start_pos=start_pos, x=x, qr=qr)
        o = assembly.attend(
            q,
            self._control(f"{base}.attn_sink", dtype=torch.float32),
            config.head_dim**-0.5,
        )
        apply_rotary_emb(o[..., -config.rope_head_dim :], freqs, inverse=True)
        heads_per_group = config.n_heads // config.o_groups
        o = o.reshape(*o.shape[:2], config.o_groups, heads_per_group * config.head_dim)
        o = self.pager.grouped_linear(
            o,
            f"{base}.wo_a",
            groups=config.o_groups,
            activation_quantization=False,
        )
        return self.pager.linear(o.flatten(2), f"{base}.wo_b")

    def _attention_one(self, x: Any, layer: int) -> Any:
        """Exact native attention for an isolated position-zero context."""

        torch = self.torch
        config = self.config
        base = f"layers.{layer}.attn"
        qr = self.pager.linear(x, f"{base}.wq_a")
        qr = self._norm(qr, f"{base}.q_norm.weight")
        q = self.pager.linear(qr, f"{base}.wq_b")
        q = q.reshape(*q.shape[:-1], config.n_heads, config.head_dim)
        # Match the published forward literally: unlike RMSNorm, this head-wise
        # normalization intentionally stays in the activation dtype.
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + config.norm_eps)

        kv = self.pager.linear(x, f"{base}.wkv")
        kv = self._norm(kv, f"{base}.kv_norm.weight")
        rope_base = (
            config.compress_rope_theta
            if config.compress_ratios[layer]
            else config.rope_theta
        )
        original = config.original_seq_len if config.compress_ratios[layer] else 0
        freqs = precompute_freqs_cis(
            config.rope_head_dim,
            1,
            original,
            rope_base,
            config.rope_factor,
            config.beta_fast,
            config.beta_slow,
        )
        q_rope = apply_rotary_emb(q[..., -config.rope_head_dim :], freqs)
        q = torch.cat((q[..., : -config.rope_head_dim], q_rope), dim=-1)
        kv_rope = apply_rotary_emb(kv[..., -config.rope_head_dim :], freqs)
        kv = torch.cat((kv[..., : -config.rope_head_dim], kv_rope), dim=-1)
        kv = self._qat_kv(kv)

        indices = torch.zeros((x.shape[0], 1, 1), dtype=torch.long, device=x.device)
        o = sparse_attention(
            q,
            kv,
            self._control(f"{base}.attn_sink", dtype=torch.float32),
            indices,
            config.head_dim**-0.5,
        )
        o_rope = apply_rotary_emb(o[..., -config.rope_head_dim :], freqs, inverse=True)
        o = torch.cat((o[..., : -config.rope_head_dim], o_rope), dim=-1)
        heads_per_group = config.n_heads // config.o_groups
        o = o.reshape(*o.shape[:2], config.o_groups, heads_per_group * config.head_dim)
        o = self.pager.grouped_linear(
            o,
            f"{base}.wo_a",
            groups=config.o_groups,
            activation_quantization=False,
        )
        return self.pager.linear(o.flatten(2), f"{base}.wo_b")

    def _hash_expert_rows(self, name: str, token_ids: Any, device: Any) -> Any:
        ids = [
            int(value) for value in token_ids.detach().to("cpu").reshape(-1).tolist()
        ]
        unique = sorted(set(ids))
        runs: list[tuple[int, int]] = []
        if unique:
            start = previous = unique[0]
            for token_id in unique[1:]:
                if token_id != previous + 1:
                    runs.append((start, previous + 1))
                    start = token_id
                previous = token_id
            runs.append((start, previous + 1))
        by_id: dict[int, np.ndarray] = {}
        priority = getattr(self.pager.source, "cache_priority", None)
        scope = priority(1) if callable(priority) else nullcontext()
        with scope:
            for start, stop in runs:
                rows = self.pager.source.rows(
                    name, start_row=start, n_rows=stop - start
                )
                for offset, row in enumerate(rows):
                    by_id[start + offset] = row
        array = np.stack([by_id[token_id] for token_id in ids]).astype(
            np.int64, copy=False
        )
        return self.torch.from_numpy(np.ascontiguousarray(array)).to(device)

    def _route_experts_many(
        self, x: Any, layer: int, token_ids: Any
    ) -> tuple[Any, Any]:
        torch = self.torch
        config = self.config
        base = f"layers.{layer}.ffn.gate"
        flat_ids = token_ids.reshape(-1)
        if x.shape[0] != flat_ids.numel():
            raise ValueError("router input and token IDs have different row counts")
        scores = self.pager.linear(
            x.float(),
            base,
            activation_quantization=False,
            output_dtype=torch.float32,
            compute_dtype=torch.float32,
        )
        if config.score_func == "softmax":
            original = scores.softmax(dim=-1)
        elif config.score_func == "sigmoid":
            original = scores.sigmoid()
        else:
            original = torch.nn.functional.softplus(scores).sqrt()
        if layer < config.n_hash_layers:
            indices = self._hash_expert_rows(f"{base}.tid2eid", flat_ids, x.device)
        else:
            bias = self._control(f"{base}.bias", dtype=torch.float32)
            indices = (original + bias).topk(config.n_activated_experts, dim=-1).indices
        weights = torch.gather(original, -1, indices)
        if config.score_func != "softmax":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights * config.route_scale, indices

    def _route_experts(self, x: Any, layer: int, token_id: int) -> tuple[Any, Any]:
        token_ids = self.torch.full(
            (x.shape[0],), token_id, dtype=self.torch.long, device=x.device
        )
        return self._route_experts_many(x, layer, token_ids)

    def _expert(
        self,
        x: Any,
        base: str,
        route_weight: Any | None = None,
        *,
        prefetched_payload: Any | None = None,
    ) -> Any:
        return self.pager.expert(
            x,
            base,
            route_weight=route_weight,
            swiglu_limit=self.config.swiglu_limit,
            prefetched_payload=prefetched_payload,
        )

    @staticmethod
    def _route_window_bases(window: Any) -> tuple[str, ...]:
        return tuple(
            f"layers.{window.target_layer}.ffn.experts.{expert_id}"
            for expert_id in window.candidate_experts
        )

    @staticmethod
    def _route_aggregate_bases(plan: LayerMicroWindowPlan) -> tuple[str, ...]:
        return tuple(
            f"layers.{plan.target_layer}.ffn.experts.{expert_id}"
            for expert_id in plan.aggregate_candidates
        )

    def _cancel_route_prefetch(self) -> None:
        reservoir = self._route_prefetch_reservoir
        if reservoir is not None:
            self.pager.cancel_expert_reservoir(reservoir)
        self._route_prefetch_plan = None
        self._route_prefetch_window_index = 0
        self._route_prefetch_reservoir = None
        self._route_prefetch_aggregate = False

    def _start_route_prefetch_window(self) -> bool:
        plan = self._route_prefetch_plan
        if plan is None or self._route_prefetch_window_index >= len(plan.windows):
            return False
        window = plan.windows[self._route_prefetch_window_index]
        if (
            not self._route_prefetch_aggregate
            and window.confidence < self.route_prefetch_min_confidence
        ):
            return False
        bases = (
            self._route_aggregate_bases(plan)
            if self._route_prefetch_aggregate
            else self._route_window_bases(window)
        )
        reservoir = self.pager.prefetch_expert_reservoir(bases)
        if reservoir is None:
            self._route_prefetch_plan = None
            self._route_prefetch_window_index = 0
            self._route_prefetch_reservoir = None
            return False
        self._route_prefetch_reservoir = reservoir
        return True

    def _schedule_route_prefetch(
        self,
        *,
        source_layer: int,
        selected_rows: tuple[tuple[int, ...], ...],
    ) -> None:
        if self._route_prefetch_plan is not None or self._route_prefetch_reservoir:
            self._cancel_route_prefetch()
        predictor = self.route_predictor
        if (
            predictor is None
            or source_layer not in predictor.source_layers
            or source_layer + 1 >= self.config.n_layers
        ):
            return
        self._route_prefetch_plan = plan_micro_window_prefetch(
            predictor,
            source_layer=source_layer,
            current_rows=selected_rows,
            window_rows=self.route_prefetch_window_rows,
            k=self.route_prefetch_k,
            alpha=self.route_prefetch_alpha,
        )
        self._route_prefetch_window_index = 0
        self._route_prefetch_aggregate = (
            self._route_prefetch_plan.active_rows > self.route_prefetch_direct_max_rows
        )
        self._route_prefetch_stats["plans"] += 1
        plan_metric = (
            "aggregate_plans" if self._route_prefetch_aggregate else "direct_plans"
        )
        self._route_prefetch_stats[plan_metric] += 1
        self._start_route_prefetch_window()

    def _route_prefetch_matches(self, *, layer: int, active_rows: int) -> bool:
        plan = self._route_prefetch_plan
        if plan is None:
            return False
        if plan.target_layer == layer and plan.active_rows == active_rows:
            return True
        self._cancel_route_prefetch()
        return False

    def _bind_route_prefetch_window(
        self,
        *,
        layer: int,
        active_row_start: int,
        active_row_stop: int,
        exact_bases: tuple[str, ...],
    ) -> tuple[dict[str, Any], Any | None]:
        plan = self._route_prefetch_plan
        index = self._route_prefetch_window_index
        reservoir = self._route_prefetch_reservoir
        if plan is None or index >= len(plan.windows):
            return {}, self.pager.prefetch_expert_window(exact_bases)
        window = plan.windows[index]
        aligned = plan.target_layer == layer and (
            active_row_start == 0 and active_row_stop == plan.active_rows
            if self._route_prefetch_aggregate
            else window.active_row_start == active_row_start
            and window.active_row_stop == active_row_stop
        )
        if not aligned:
            self._cancel_route_prefetch()
            return {}, self.pager.prefetch_expert_window(exact_bases)

        if reservoir is None:
            payloads: dict[str, Any] = {}
            self._route_prefetch_stats["confidence_skips"] += 1
        else:
            payloads = self.pager.bind_expert_reservoir(reservoir, exact_bases)
            binding_metric = (
                "aggregate_bindings"
                if self._route_prefetch_aggregate
                else "direct_bindings"
            )
            self._route_prefetch_stats[binding_metric] += 1
            self._route_prefetch_reservoir = None
        self._route_prefetch_window_index += 1
        misses = tuple(base for base in exact_bases if base not in payloads)
        exact_window = self.pager.prefetch_expert_window(misses) if misses else None
        if (
            not self._route_prefetch_aggregate
            and self._route_prefetch_window_index < len(plan.windows)
        ):
            self._start_route_prefetch_window()
        else:
            self._route_prefetch_plan = None
            self._route_prefetch_window_index = 0
            self._route_prefetch_aggregate = False
        return payloads, exact_window

    def _discard_route_payloads(self, payloads: dict[str, Any]) -> None:
        for payload in payloads.values():
            self.pager.discard_expert_reservoir_payload(payload)
        payloads.clear()

    def route_prefetch_metrics(self) -> dict[str, Any]:
        """Return scheduler counters beside the pager's exact I/O receipts."""

        return {
            **self._route_prefetch_stats,
            "active": self._route_prefetch_plan is not None,
            "alpha": self.route_prefetch_alpha,
            "direct_max_rows": self.route_prefetch_direct_max_rows,
            "enabled": self.route_predictor is not None,
            "k": self.route_prefetch_k,
            "min_confidence": self.route_prefetch_min_confidence,
            "window_rows": self.route_prefetch_window_rows,
        }

    def _moe_one(
        self, x: Any, layer: int, token_id: int
    ) -> tuple[Any, tuple[int, ...]]:
        flat = x.reshape(-1, self.config.dim)
        weights, indices = self._route_experts(flat, layer, token_id)
        output = self.torch.zeros_like(flat, dtype=self.torch.float32)
        chosen = tuple(int(value) for value in indices[0].tolist())
        # The official ModuleList loop visits expert IDs in ascending order.
        # FP32 accumulation is not associative, so preserve that order while
        # still selecting each expert's original routing-weight slot.
        ordered = sorted(enumerate(chosen), key=lambda item: item[1])
        bases = tuple(
            f"layers.{layer}.ffn.experts.{expert_id}" for _, expert_id in ordered
        )
        use_route_plan = self._route_prefetch_matches(layer=layer, active_rows=1)
        if use_route_plan:
            route_payloads, window = self._bind_route_prefetch_window(
                layer=layer,
                active_row_start=0,
                active_row_stop=1,
                exact_bases=bases,
            )
        else:
            route_payloads = {}
            window = self.pager.prefetch_expert_window(bases) if bases else None
        try:
            for (slot, expert_id), base in zip(ordered, bases, strict=True):
                payload = route_payloads.pop(base, None)
                if payload is None and window is not None:
                    payload = self.pager.consume_expert_window(window, base)
                try:
                    claimed = payload
                    payload = None
                    try:
                        expert = self._expert(
                            flat,
                            base,
                            weights[:, slot : slot + 1],
                            prefetched_payload=claimed,
                        )
                    finally:
                        del claimed
                finally:
                    if payload is not None:
                        self.pager.discard_expert_payload(payload)
                output += expert.float()
        except BaseException:
            self._discard_route_payloads(route_payloads)
            if window is not None and not window.closed:
                self.pager.close_expert_window(window, cancel=True)
            if self._route_prefetch_plan is not None:
                self._cancel_route_prefetch()
            raise
        if window is not None:
            self.pager.close_expert_window(window)
        if route_payloads:
            self._discard_route_payloads(route_payloads)
            raise DeepSeekV4RuntimeError("route reservoir retained an exact payload")
        self._schedule_route_prefetch(
            source_layer=layer,
            selected_rows=(chosen,),
        )
        try:
            shared = self._expert(flat, f"layers.{layer}.ffn.shared_experts")
            output += shared.float()
        except BaseException:
            if self._route_prefetch_plan is not None:
                self._cancel_route_prefetch()
            raise
        return output.to(x.dtype).reshape_as(x), chosen

    def _moe(
        self,
        x: Any,
        layer: int,
        token_ids: Any,
        token_mask: Any | None = None,
    ) -> tuple[Any, tuple[tuple[int, ...], ...]]:
        flat = x.reshape(-1, self.config.dim)
        flat_ids = token_ids.reshape(-1)
        if token_mask is None:
            active_rows = self.torch.arange(
                flat.shape[0], dtype=self.torch.long, device=flat.device
            )
        else:
            flat_mask = token_mask.reshape(-1)
            if flat_mask.shape[0] != flat.shape[0]:
                raise ValueError("MoE token mask does not match hidden rows")
            active_rows = self.torch.nonzero(flat_mask, as_tuple=False).flatten()
        if active_rows.numel() == 0:
            return self.torch.zeros_like(x), tuple(() for _ in range(flat.shape[0]))
        active = flat.index_select(0, active_rows)
        active_ids = flat_ids.index_select(0, active_rows)
        weights, indices = self._route_experts_many(active, layer, active_ids)
        output = self.torch.zeros_like(flat, dtype=self.torch.float32)
        active_selected = tuple(
            tuple(int(value) for value in row)
            for row in indices.detach().to("cpu").tolist()
        )
        selected_rows: list[tuple[int, ...]] = [() for _ in range(flat.shape[0])]
        for row, choices in zip(
            active_rows.detach().to("cpu").tolist(), active_selected, strict=True
        ):
            selected_rows[int(row)] = choices
        # Each output row is independent. Traversing experts in ascending order
        # inside causal row micro-windows therefore preserves the official
        # per-row FP32 accumulation while keeping the measured Markov signal.
        use_route_plan = self._route_prefetch_matches(
            layer=layer,
            active_rows=len(active_selected),
        )
        if use_route_plan and not self._route_prefetch_aggregate:
            assert self._route_prefetch_plan is not None
            row_windows = tuple(
                (window.active_row_start, window.active_row_stop)
                for window in self._route_prefetch_plan.windows
            )
        else:
            row_windows = ((0, len(active_selected)),)

        try:
            for row_start, row_stop in row_windows:
                window_indices = indices[row_start:row_stop]
                ordered_experts = sorted(
                    {
                        int(value)
                        for row in active_selected[row_start:row_stop]
                        for value in row
                    }
                )
                bases = tuple(
                    f"layers.{layer}.ffn.experts.{expert_id}"
                    for expert_id in ordered_experts
                )
                if use_route_plan:
                    route_payloads, exact_window = self._bind_route_prefetch_window(
                        layer=layer,
                        active_row_start=row_start,
                        active_row_stop=row_stop,
                        exact_bases=bases,
                    )
                else:
                    route_payloads = {}
                    exact_window = (
                        self.pager.prefetch_expert_window(bases) if bases else None
                    )
                try:
                    for expert_id, base in zip(ordered_experts, bases, strict=True):
                        local_rows, slots = self.torch.where(
                            window_indices == expert_id
                        )
                        rows = local_rows + row_start
                        payload = route_payloads.pop(base, None)
                        if payload is None and exact_window is not None:
                            payload = self.pager.consume_expert_window(
                                exact_window, base
                            )
                        try:
                            claimed = payload
                            payload = None
                            try:
                                expert = self._expert(
                                    active.index_select(0, rows),
                                    base,
                                    weights[rows, slots].unsqueeze(-1),
                                    prefetched_payload=claimed,
                                )
                            finally:
                                del claimed
                        finally:
                            if payload is not None:
                                self.pager.discard_expert_payload(payload)
                        output.index_add_(
                            0,
                            active_rows.index_select(0, rows),
                            expert.float(),
                        )
                except BaseException:
                    self._discard_route_payloads(route_payloads)
                    if exact_window is not None and not exact_window.closed:
                        self.pager.close_expert_window(exact_window, cancel=True)
                    raise
                if exact_window is not None:
                    self.pager.close_expert_window(exact_window)
                if route_payloads:
                    self._discard_route_payloads(route_payloads)
                    raise DeepSeekV4RuntimeError(
                        "route reservoir retained an exact payload"
                    )
        except BaseException:
            if self._route_prefetch_plan is not None:
                self._cancel_route_prefetch()
            raise

        self._schedule_route_prefetch(
            source_layer=layer,
            selected_rows=active_selected,
        )
        try:
            shared = self._expert(active, f"layers.{layer}.ffn.shared_experts").float()
            output.index_add_(0, active_rows, shared)
        except BaseException:
            if self._route_prefetch_plan is not None:
                self._cancel_route_prefetch()
            raise
        return output.to(x.dtype).reshape_as(x), tuple(selected_rows)

    def _block_one(
        self, x: Any, layer: int, token_id: int
    ) -> tuple[Any, tuple[int, ...]]:
        prefix = f"layers.{layer}"
        residual = x
        hidden, post, comb = self._hc_pre(x, f"{prefix}.hc_attn")
        hidden = self._norm(hidden, f"{prefix}.attn_norm.weight")
        hidden = self._attention_one(hidden, layer)
        x = hc_post(hidden, residual, post, comb)

        residual = x
        hidden, post, comb = self._hc_pre(x, f"{prefix}.hc_ffn")
        hidden = self._norm(hidden, f"{prefix}.ffn_norm.weight")
        hidden, selected = self._moe_one(hidden, layer, token_id)
        try:
            x = hc_post(hidden, residual, post, comb)
        except BaseException:
            if self._route_prefetch_plan is not None:
                self._cancel_route_prefetch()
            raise
        return x, selected

    def _block(
        self,
        x: Any,
        layer: int,
        token_ids: Any,
        start_pos: int,
        token_mask: Any | None = None,
    ) -> tuple[Any, tuple[tuple[int, ...], ...]]:
        prefix = f"layers.{layer}"
        residual = x
        hidden, post, comb = self._hc_pre(x, f"{prefix}.hc_attn")
        hidden = self._norm(hidden, f"{prefix}.attn_norm.weight")
        hidden = self._attention(hidden, layer, start_pos)
        x = hc_post(hidden, residual, post, comb)

        residual = x
        hidden, post, comb = self._hc_pre(x, f"{prefix}.hc_ffn")
        hidden = self._norm(hidden, f"{prefix}.ffn_norm.weight")
        hidden, selected = self._moe(hidden, layer, token_ids, token_mask)
        try:
            x = hc_post(hidden, residual, post, comb)
        except BaseException:
            if self._route_prefetch_plan is not None:
                self._cancel_route_prefetch()
            raise
        return x, selected

    def _token_tensor(self, token_ids: Any) -> Any:
        torch = self.torch
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
            raise ValueError(
                f"batch size {ids.shape[0]} exceeds maximum {self.max_batch_size}"
            )
        ids = ids.to(device=self.pager.device, dtype=torch.long)
        if bool(((ids < 0) | (ids >= self.config.vocab_size)).any().item()):
            raise ValueError("token ID outside checkpoint vocabulary")
        return ids

    def embed_batch(self, token_ids: Any) -> Any:
        """Embed one padded batch into the official HC decoder representation."""

        ids = self._token_tensor(token_ids)
        hidden = self.pager.embedding(ids.detach().to("cpu").reshape(-1).tolist())
        hidden = hidden.reshape(ids.shape[0], ids.shape[1], self.config.dim)
        return hidden.unsqueeze(2).repeat(1, 1, self.config.hc_mult, 1)

    def release_layer_state(self, layer: int) -> None:
        """Release one layer's mutable attention state between independent batches."""

        if isinstance(layer, bool) or not isinstance(layer, int):
            raise TypeError("layer must be an integer")
        if not 0 <= layer < self.config.n_layers:
            raise ValueError("layer outside decoder depth")
        state = self._attention_states[layer]
        if state is not None:
            state.reset(release=True)
        self._attention_states[layer] = None

    def forward_prefill_layer(
        self,
        hidden: Any,
        token_ids: Any,
        *,
        layer: int,
        token_mask: Any | None = None,
    ) -> tuple[Any, tuple[tuple[int, ...], ...]]:
        """Apply exactly one official decoder block to an independent prefill.

        This is the stateless primitive used by the out-of-core layer-major
        engine.  Its attention cache is new for this batch and is released even
        when the block raises, preventing state from leaking between items.
        Right-padding rows may be excluded from MoE routing with ``token_mask``;
        causal attention still evaluates the rectangular batch, so valid prefix
        positions retain the same model equation as an unpadded prefill.
        """

        ids = self._token_tensor(token_ids)
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise TypeError("layer must be an integer")
        if not 0 <= layer < self.config.n_layers:
            raise ValueError("layer outside decoder depth")
        expected = (
            ids.shape[0],
            ids.shape[1],
            self.config.hc_mult,
            self.config.dim,
        )
        if tuple(hidden.shape) != expected:
            raise ValueError(
                f"hidden shape {tuple(hidden.shape)} does not match {expected}"
            )
        hidden = hidden.to(device=self.pager.device, dtype=self.pager.compute_dtype)
        mask = None
        if token_mask is not None:
            mask = self.torch.as_tensor(token_mask, device=self.pager.device)
            if mask.dtype != self.torch.bool:
                raise TypeError("token_mask must be boolean")
            if tuple(mask.shape) != tuple(ids.shape):
                raise ValueError("token_mask must match token_ids shape")
            if bool((~mask[:, 0]).any().item()):
                raise ValueError("every batch row must contain a non-empty prefix")
            # Only right-padding is admissible: once false, a row cannot become
            # active again.  This preserves each item's exact causal prefix.
            if ids.shape[1] > 1 and bool((mask[:, 1:] & ~mask[:, :-1]).any().item()):
                raise ValueError("token_mask must describe right-padded prefixes")
        self.release_layer_state(layer)
        try:
            return self._block(hidden, layer, ids, 0, mask)
        finally:
            self.release_layer_state(layer)

    def finalize_hidden(self, hidden: Any) -> Any:
        """Apply the official HC head and final RMSNorm after the last block."""

        if hidden.ndim != 4 or tuple(hidden.shape[2:]) != (
            self.config.hc_mult,
            self.config.dim,
        ):
            raise ValueError("hidden must have shape [B,S,HC,D]")
        hidden = hidden.to(device=self.pager.device, dtype=self.pager.compute_dtype)
        result = hc_head(
            hidden,
            self._control("hc_head_fn", dtype=self.torch.float32),
            self._control("hc_head_scale", dtype=self.torch.float32),
            self._control("hc_head_base", dtype=self.torch.float32),
            eps=self.config.hc_eps,
            norm_eps=self.config.norm_eps,
        )
        return self._norm(result, "norm.weight")

    def _apply_graft(self, hidden: Any, start_pos: int) -> tuple[Any, Any | None]:
        if self.graft is None:
            return hidden, None
        mode = str(getattr(self.graft, "mode", "unknown"))
        alpha = float(getattr(self.graft, "alpha", 0.0))
        if mode == "off" or alpha == 0.0:
            result = self.graft.forward(hidden, return_evidence=True)
            return (
                (result[0], result[1]) if isinstance(result, tuple) else (result, None)
            )

        if start_pos == 0:
            history = hidden
        else:
            if self._graft_history is None:
                raise DeepSeekV4RuntimeError("graft history is missing for decode")
            if self._graft_history.shape[0] != hidden.shape[0]:
                raise DeepSeekV4RuntimeError("graft batch changed within a request")
            history = self.torch.cat((self._graft_history, hidden), dim=1)
        result = self.graft.forward(history, return_evidence=True)
        if isinstance(result, tuple):
            grafted, evidence = result
        else:
            grafted, evidence = result, None
        self._graft_history = history.detach().clone()
        return grafted[:, -hidden.shape[1] :], evidence

    def hidden_stateful(
        self,
        token_ids: Any,
        *,
        start_pos: int | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[Any, StatefulEvidence]:
        """Execute a native prefill or one contiguous decode step.

        A non-zero call intentionally accepts one token only, matching the
        published cache mutation contract.  Use :meth:`prefill` with
        ``tokenwise=True`` to bound the active expert union for a long prompt.
        """

        ids = self._token_tensor(token_ids)
        if self._state_poisoned:
            raise DeepSeekV4RuntimeError(
                "decoder state is poisoned; call reset_state()"
            )
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
        if start_pos > 0 and ids.shape[1] != 1:
            raise ValueError("stateful decode accepts exactly one token")
        end_pos = start_pos + ids.shape[1]
        if end_pos > self.max_seq_len:
            raise ValueError(
                f"context end {end_pos} exceeds max_seq_len={self.max_seq_len}"
            )

        start_bytes = int(
            self.pager.source.metrics().get("network_or_source_body_bytes", 0)
        )
        start_linears = int(self.pager.metrics()["linear_calls"])
        started = time.perf_counter()
        h = self.pager.embedding(ids.detach().to("cpu").reshape(-1).tolist())
        h = h.reshape(ids.shape[0], ids.shape[1], self.config.dim)
        h = h.unsqueeze(2).repeat(1, 1, self.config.hc_mult, 1)
        selected: list[tuple[tuple[int, ...], ...]] = []
        try:
            for layer in range(self.config.n_layers):
                layer_started = time.perf_counter()
                layer_bytes_before = int(
                    self.pager.source.metrics().get("network_or_source_body_bytes", 0)
                )
                h, experts = self._block(h, layer, ids, start_pos)
                selected.append(experts)
                graft_evidence = None
                if self.graft is not None and self.graft_layer == layer:
                    h, graft_evidence = self._apply_graft(h, start_pos)
                self.pager.release()
                if progress is not None:
                    layer_bytes_after = int(
                        self.pager.source.metrics().get(
                            "network_or_source_body_bytes", 0
                        )
                    )
                    progress(
                        {
                            "layer": layer,
                            "layers": self.config.n_layers,
                            "start_pos": start_pos,
                            "tokens": ids.shape[1],
                            "experts": [list(row) for row in experts],
                            "source_body_bytes": (
                                layer_bytes_after - layer_bytes_before
                            ),
                            "seconds": time.perf_counter() - layer_started,
                            "graft": (
                                None
                                if graft_evidence is None
                                else graft_evidence.to_dict()
                            ),
                        }
                    )
            h = hc_head(
                h,
                self._control("hc_head_fn", dtype=self.torch.float32),
                self._control("hc_head_scale", dtype=self.torch.float32),
                self._control("hc_head_base", dtype=self.torch.float32),
                eps=self.config.hc_eps,
                norm_eps=self.config.norm_eps,
            )
            h = self._norm(h, "norm.weight")
        except Exception:
            # Clear every partially mutated cache but keep a poison latch so
            # reuse is explicit.  The caller must acknowledge the failed
            # request with reset_state() before another forward.
            for state in self._attention_states:
                if state is not None:
                    state.reset()
            self._next_position = 0
            self._graft_history = None
            self._state_poisoned = True
            if self._route_prefetch_plan is not None:
                self._cancel_route_prefetch()
            raise

        self._next_position = end_pos
        end_bytes = int(
            self.pager.source.metrics().get("network_or_source_body_bytes", 0)
        )
        end_linears = int(self.pager.metrics()["linear_calls"])
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
            stateful_kv_cache=True,
            source_body_bytes=end_bytes - start_bytes,
            linear_calls=end_linears - start_linears,
            seconds=time.perf_counter() - started,
            attention_state_bytes=self.attention_state_bytes,
            selected_experts=tuple(selected),
            graft_mode=(
                "off"
                if self.graft is None
                else str(getattr(self.graft, "mode", "unknown"))
            ),
            graft_history_tokens=(
                0 if self._graft_history is None else int(self._graft_history.shape[1])
            ),
        )
        return h, evidence

    def prefill(
        self,
        token_ids: Any,
        *,
        tokenwise: bool = True,
        reset: bool = True,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[Any, tuple[StatefulEvidence, ...]]:
        """Prefill a request, defaulting to bounded exact tokenwise execution."""

        ids = self._token_tensor(token_ids)
        if reset:
            self.reset_state()
        elif self._next_position != 0:
            raise ValueError("prefill requires position zero or reset=True")
        if not tokenwise:
            hidden, evidence = self.hidden_stateful(ids, start_pos=0, progress=progress)
            return hidden, (evidence,)
        outputs = []
        evidence_rows = []
        for position in range(ids.shape[1]):
            hidden, evidence = self.hidden_stateful(
                ids[:, position : position + 1],
                start_pos=position,
                progress=progress,
            )
            outputs.append(hidden)
            evidence_rows.append(evidence)
        return self.torch.cat(outputs, dim=1), tuple(evidence_rows)

    def decode(
        self,
        token_ids: Any,
        *,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[Any, StatefulEvidence]:
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
        prefill_tokenwise: bool = True,
        eos_token_ids: Any = (),
        head_block_rows: int = 1024,
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
            raise ValueError("greedy PoC generation currently requires batch size one")
        eos = {int(value) for value in eos_token_ids}
        if any(value < 0 or value >= self.config.vocab_size for value in eos):
            raise ValueError("EOS token outside checkpoint vocabulary")
        start_bytes = int(
            self.pager.source.metrics().get("network_or_source_body_bytes", 0)
        )
        start_linears = int(self.pager.metrics()["linear_calls"])
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
            values, ids = self.pager.topk_logits(
                hidden[:, -1],
                k=1,
                block_rows=head_block_rows,
                progress=head_progress,
            )
            token_id = int(ids[0, 0].item())
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
            if token_id in eos:
                stopped_on_eos = True
                break
            if step + 1 < max_new_tokens:
                hidden, _ = self.decode([[token_id]], progress=progress)
                forward_count += 1
        end_bytes = int(
            self.pager.source.metrics().get("network_or_source_body_bytes", 0)
        )
        end_linears = int(self.pager.metrics()["linear_calls"])
        prompt_ids = tuple(
            int(value) for value in prompt[0].detach().to("cpu").tolist()
        )
        evidence = GenerationEvidence(
            prompt_token_ids=prompt_ids,
            generated_token_ids=tuple(generated),
            context_mode="stateful_autoregressive",
            stateful_kv_cache=True,
            general_generation=True,
            prefill_mode="tokenwise" if prefill_tokenwise else "batched",
            forward_passes=forward_count,
            source_body_bytes=end_bytes - start_bytes,
            linear_calls=end_linears - start_linears,
            seconds=time.perf_counter() - started,
            attention_state_bytes=self.attention_state_bytes,
            stopped_on_eos=stopped_on_eos,
        )
        return tuple(generated), evidence

    def hidden_one_token(
        self,
        token_id: int,
        *,
        stop_after_layer: int | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[Any, OneTokenEvidence]:
        """Execute position zero through all layers (or an explicit diagnostic prefix)."""

        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("token_id must be an integer")
        if not 0 <= token_id < self.config.vocab_size:
            raise ValueError("token_id outside checkpoint vocabulary")
        if stop_after_layer is None:
            layers = self.config.n_layers
        else:
            if isinstance(stop_after_layer, bool) or not isinstance(
                stop_after_layer, int
            ):
                raise TypeError("stop_after_layer must be an integer")
            if not 1 <= stop_after_layer <= self.config.n_layers:
                raise ValueError("stop_after_layer outside decoder depth")
            layers = stop_after_layer
        start_bytes = int(
            self.pager.source.metrics().get("network_or_source_body_bytes", 0)
        )
        start_linears = int(self.pager.metrics()["linear_calls"])
        started = time.perf_counter()
        h = self.pager.embedding([token_id]).reshape(1, 1, self.config.dim)
        h = h.unsqueeze(2).repeat(1, 1, self.config.hc_mult, 1)
        selected: list[tuple[int, ...]] = []
        for layer in range(layers):
            layer_started = time.perf_counter()
            layer_bytes_before = int(
                self.pager.source.metrics().get("network_or_source_body_bytes", 0)
            )
            h, experts = self._block_one(h, layer, token_id)
            selected.append(experts)
            if self.graft is not None and self.graft_layer == layer:
                grafted = self.graft.forward(h, return_evidence=True)
                h = grafted[0] if isinstance(grafted, tuple) else grafted
            self.pager.release()
            if progress is not None:
                layer_bytes_after = int(
                    self.pager.source.metrics().get("network_or_source_body_bytes", 0)
                )
                progress(
                    {
                        "layer": layer,
                        "layers": layers,
                        "experts": list(experts),
                        "source_body_bytes": layer_bytes_after - layer_bytes_before,
                        "seconds": time.perf_counter() - layer_started,
                    }
                )
        if layers == self.config.n_layers:
            h = hc_head(
                h,
                self._control("hc_head_fn", dtype=self.torch.float32),
                self._control("hc_head_scale", dtype=self.torch.float32),
                self._control("hc_head_base", dtype=self.torch.float32),
                eps=self.config.hc_eps,
                norm_eps=self.config.norm_eps,
            )
            h = self._norm(h, "norm.weight")
        end_metrics = self.pager.metrics()
        end_bytes = int(
            self.pager.source.metrics().get("network_or_source_body_bytes", 0)
        )
        evidence = OneTokenEvidence(
            token_id=token_id,
            layers_executed=layers,
            checkpoint_layers=self.config.n_layers,
            complete_layer_stack=layers == self.config.n_layers,
            context_mode="isolated_position_zero",
            stateful_kv_cache=False,
            source_body_bytes=end_bytes - start_bytes,
            linear_calls=int(end_metrics["linear_calls"]) - start_linears,
            seconds=time.perf_counter() - started,
            selected_experts=tuple(selected),
        )
        return h, evidence

    def next_token_from_one(
        self,
        token_id: int,
        *,
        topk: int = 1,
        head_block_rows: int = 1024,
        progress: Callable[[dict[str, Any]], None] | None = None,
        head_progress: Callable[[dict[str, int]], None] | None = None,
    ) -> tuple[Any, Any, OneTokenEvidence]:
        """Run the complete main decoder and globally scan the LM head."""

        hidden, evidence = self.hidden_one_token(token_id, progress=progress)
        if not evidence.complete_layer_stack:  # defensive; no prefix logits
            raise DeepSeekV4RuntimeError("LM head requires the complete decoder")
        values, token_ids = self.pager.topk_logits(
            hidden[:, -1],
            k=topk,
            block_rows=head_block_rows,
            progress=head_progress,
        )
        return values, token_ids, evidence

    @staticmethod
    def evidence_dict(evidence: OneTokenEvidence) -> dict[str, Any]:
        return asdict(evidence)

    @staticmethod
    def stateful_evidence_dict(evidence: StatefulEvidence) -> dict[str, Any]:
        return asdict(evidence)

    @staticmethod
    def generation_evidence_dict(evidence: GenerationEvidence) -> dict[str, Any]:
        return asdict(evidence)


__all__ = [
    "DeepSeekV4RuntimeError",
    "GenerationEvidence",
    "OneTokenEvidence",
    "StatefulEvidence",
    "StreamedDeepSeekV4",
]
