from __future__ import annotations

import io
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import Mock, call, patch

import torch

from immer.knowledge.livecausal import LiveGraph
from immer.cli import (
    _QWEN38_HYBRID_DRAFT_ABI,
    _QWEN38_MARKOV_DRAFT_ABI,
    _QWEN38_MTP_DRAFT_ABI,
    _qwen38_growing_warm_profile,
    _qwen38_layer_mlp_o1_policy,
    _qwen38_output_semantics,
    _resolve_qwen38_prefix_sinkhorn,
    _qwen38_service_profile,
    _qwen38_runtime_code_paths,
    main,
)
from immer.cognition.fertig import FertigSolver
from immer.cognition.qwen_fertig_chat import QwenFertigChat
from immer.composition import CompositionRoot, compose_runtime
from immer.contracts import ExecutionStatus, Request, Result
from immer.runtimes.ooe.result_cells import (
    ResultCell,
    ResultCellBinding,
    attach_cold_qwen_generation_receipt,
    qwen_result_binding_evidence,
)
from immer.runtimes.ooe.compute_crystals import ComputeCrystal, ComputeCrystalBank
from immer.runtimes.ooe.compute_graph import ComputeOperatorGraph, OperatorEdge
from immer.runtimes.qwen3_8.adapter import (
    DEFAULT_LAYER_CONTEXTUAL_CONTINUATION_LAYERS,
    DEFAULT_LAYER_CONTEXTUAL_CONTINUATION_PROJECTION_SEED,
    LAYER_CONTEXTUAL_CONTINUATION_EVIDENCE_SCHEMA,
    LAYER_CONTEXTUAL_CONTINUATION_REQUEST_SCHEMA,
    LAYER_MLP_CRYSTAL_EVIDENCE_SCHEMA,
    LAYER_TRANSITION_CRYSTAL_EVIDENCE_SCHEMA,
    QWEN38_CHAT_HISTORY_METADATA,
    QWEN38_CHAT_SESSION_METADATA,
    QWEN38_INFERENCE_ACTION_METADATA,
    Qwen38CausalChat,
    Qwen38ChatError,
    _layer_mlp_o1_state_path_for_layer,
)
from immer.runtimes.qwen3_8.cartography_probe import prompt_token_sha256
from immer.runtimes.qwen3_8.contextual_continuation import (
    ContextualContinuationBank,
    ContextualContinuationIdentity,
)
from immer.runtimes.qwen3_8.encoding import IM_END_TOKEN_ID, Qwen38Tokenizer
from immer.runtimes.qwen3_8.markov_atlas import MarkovTokenAtlas
from immer.runtimes.qwen3_8.hybrid_draft import (
    QWEN38_MARKOV_MTP_HYBRID_PROVIDER_SCHEMA,
)
from immer.runtimes.qwen3_8.action_bank import (
    InferenceActionBank,
    InferenceActionDirective,
)
from immer.runtimes.qwen3_8.attention_output_crystal import (
    ATTENTION_OUTPUT_CRYSTAL_EVIDENCE_SCHEMA,
    AttentionOutputCrystalIdentity,
)
from immer.runtimes.qwen3_8.inference_economics import (
    InferenceEconomicsLedger,
)
from immer.runtimes.qwen3_8.layer_transition_crystal import (
    LayerTransitionCrystalIdentity,
    LayerTransitionProjectionIdentity,
)
from immer.runtimes.qwen3_8.layer_mlp_crystal import (
    LAYER_MLP_RESIDUAL_ACTION_ABI,
    LAYER_MLP_RESIDUAL_GENERIC_ACTION_ABI,
    Layer63MlpResidualCrystalIdentity,
    LayerMlpResidualCrystalIdentity,
)
from immer.runtimes.qwen3_8.layer_mlp_o1 import (
    Layer63MlpO1Accumulator,
    LayerMlpO1Accumulator,
)
from immer.runtimes.qwen3_8.layer_mlp_o1_runtime import (
    Layer63MlpO1AsyncWorker,
    LayerMlpO1AsyncPool,
)
from immer.runtimes.qwen3_8.layer_contextual_continuation import (
    DEFAULT_LAYER_CONTEXTUAL_SKETCH_DIM,
    LAYER_CONTEXTUAL_CONTINUATION_METRICS_SCHEMA,
    LAYER_CONTEXTUAL_PROJECTION_ABI,
    LAYER_CONTEXTUAL_STAGE,
    LayerContextualContinuationBank,
)
from immer.runtimes.qwen3_8.layer_mlp_registry import (
    publish_layer_mlp_o1_registry,
)
from immer.runtimes.qwen3_8.markov_draft import MARKOV_DRAFT_PROVIDER_ABI
from immer.runtimes.qwen3_8.mlp_page_coordinate import (
    MLP_PAGE_COORDINATE_EVIDENCE_SCHEMA,
    MlpPageCoordinateIdentity,
)
from immer.runtimes.qwen3_8.mtp_draft import (
    QWEN35_MTP_DRAFT_PROVIDER_SCHEMA,
    Qwen35MtpCarry,
)
from immer.runtimes.qwen3_8.mtp_carry_snapshot import MtpCarrySidecarDescriptor
from immer.runtimes.qwen3_8.native_crsa import Qwen38NativeHeadCrsa
from immer.runtimes.qwen3_8.semantic_atlas import GraphRevision, ModelPin
from immer.runtimes.qwen3_8.semantic_state_cache import (
    AnchorReceipt,
    RestoredAnchor,
    SemanticStateAnchorCache,
    token_prefix_sha256,
)


_DIGEST = "a" * 64


def _layer_transition_identity(
    *,
    model: str = "2",
    q4: str = "3",
    graph: str = "4",
    atlas: str = "5",
) -> LayerTransitionCrystalIdentity:
    return LayerTransitionCrystalIdentity(
        model_sha256=model if len(model) == 64 else model * 64,
        q4_sha256=q4 if len(q4) == 64 else q4 * 64,
        graph_revision_sha256=graph if len(graph) == 64 else graph * 64,
        atlas_revision_sha256=atlas if len(atlas) == 64 else atlas * 64,
        projection=LayerTransitionProjectionIdentity(
            hidden_dim=8,
            sketch_dim=3,
            seed_sha256="1" * 64,
        ),
    )


def _layer_mlp_identity(
    *,
    model: str = "2",
    q4: str = "3",
    graph: str = "4",
    atlas: str = "5",
) -> Layer63MlpResidualCrystalIdentity:
    return Layer63MlpResidualCrystalIdentity(
        model_sha256=model if len(model) == 64 else model * 64,
        q4_sha256=q4 if len(q4) == 64 else q4 * 64,
        graph_revision_sha256=graph if len(graph) == 64 else graph * 64,
        atlas_revision_sha256=atlas if len(atlas) == 64 else atlas * 64,
        projection=LayerTransitionProjectionIdentity(
            hidden_dim=8,
            sketch_dim=3,
            seed_sha256="1" * 64,
        ),
    )


def _generic_layer_mlp_identity(
    layer_index: int,
    *,
    model: str = "2",
    q4: str = "3",
    graph: str = "4",
    atlas: str = "5",
) -> LayerMlpResidualCrystalIdentity:
    return LayerMlpResidualCrystalIdentity(
        model_sha256=model if len(model) == 64 else model * 64,
        q4_sha256=q4 if len(q4) == 64 else q4 * 64,
        graph_revision_sha256=graph if len(graph) == 64 else graph * 64,
        atlas_revision_sha256=atlas if len(atlas) == 64 else atlas * 64,
        projection=LayerTransitionProjectionIdentity(
            hidden_dim=8,
            sketch_dim=3,
            seed_sha256="1" * 64,
        ),
        layer_index=layer_index,
    )


def _fake_layer_mlp_mount_registry(*layers: int):
    projection = LayerTransitionProjectionIdentity(
        hidden_dim=8,
        sketch_dim=3,
        seed_sha256="1" * 64,
    )
    pins = SimpleNamespace(
        atlas_revision_sha256="5" * 64,
        graph_revision_sha256="4" * 64,
        model_sha256="2" * 64,
        projection=projection,
        q4_sha256="3" * 64,
    )
    entries = tuple(
        SimpleNamespace(
            action_abi=(
                LAYER_MLP_RESIDUAL_ACTION_ABI
                if layer == 63
                else LAYER_MLP_RESIDUAL_GENERIC_ACTION_ABI
            ),
            bank_file=f"layer-{layer}.json",
            bank_file_sha256=hashlib.sha256(
                f"bank-file-{layer}".encode()
            ).hexdigest(),
            bank_identity_sha256=hashlib.sha256(
                f"bank-identity-{layer}".encode()
            ).hexdigest(),
            crystal_sha256=hashlib.sha256(f"crystal-{layer}".encode()).hexdigest(),
            error_radius=0.0,
            feature_radius=0.0,
            layer_index=layer,
            max_error_radius=0.0 if layer == 18 else 0.25,
            packed_weight_bytes_avoided=8_192 + layer,
            source_o1_generation=1,
            source_o1_state_sha256=hashlib.sha256(
                f"o1-state-{layer}".encode()
            ).hexdigest(),
        )
        for layer in layers
    )
    mounts = tuple(
        SimpleNamespace(entry=entry, bank=SimpleNamespace(identity=None))
        for entry in entries
    )
    return SimpleNamespace(
        configured_layers=tuple(layers),
        entries=entries,
        manifest_file_sha256="6" * 64,
        mounted_layers=tuple(layers),
        mounts=mounts,
        pending=(),
        pins=pins,
        registry_sha256="7" * 64,
    )


def _semantic_atlas_authority(root: Path) -> str:
    atlas = LiveGraph(root)
    atlas.append_segment(
        [{"outcome_key": "promoted-layer-63", "trigger_key": "candidate-layer-63"}]
    )
    sequence, event_sha256 = atlas.store.revision()
    ancestor_sha256 = GraphRevision(sequence, event_sha256).sha256
    atlas.append_segment(
        [{"outcome_key": "later-promotion", "trigger_key": "promoted-layer-63"}]
    )
    return ancestor_sha256


def _compute_graph_authority(root: Path) -> str:
    bank = ComputeCrystalBank(root)
    crystal = ComputeCrystal.affine([[1.0]], [0.0])
    bank.publish_crystal(crystal)
    graph = ComputeOperatorGraph(bank)

    def edge(source: str, target: str, ordinal: int) -> OperatorEdge:
        return OperatorEdge(
            source_state=source,
            target_state=target,
            crystal_sha256=crystal.sha256,
            verifier_sha256=hashlib.sha256(
                f"layer-transition-verifier-{ordinal}".encode()
            ).hexdigest(),
            evidence_sha256=hashlib.sha256(
                f"layer-transition-evidence-{ordinal}".encode()
            ).hexdigest(),
        )

    ancestor, _changed = graph.append_edge(edge("h63", "h64", 1))
    graph.append_edge(edge("h64", "h65", 2))
    return ancestor.sha256


def _publish_mixed_layer_mlp_registry(root: Path) -> tuple[Path, Path, Path]:
    atlas = root / "atlas"
    compute = root / "compute"
    atlas_revision = _semantic_atlas_authority(atlas)
    graph_revision = _compute_graph_authority(compute)
    legacy_path = root / "layer63-o1.json"
    legacy = Layer63MlpO1Accumulator(
        legacy_path,
        _layer_mlp_identity(
            atlas=atlas_revision,
            graph=graph_revision,
        ),
        packed_weight_bytes_avoided=8_255,
    )
    generic = LayerMlpO1Accumulator(
        _layer_mlp_o1_state_path_for_layer(
            legacy_path,
            layer_index=18,
            sketch_dim=3,
        ),
        _generic_layer_mlp_identity(
            18,
            atlas=atlas_revision,
            graph=graph_revision,
        ),
        packed_weight_bytes_avoided=8_210,
    )
    for accumulator, seed in ((legacy, 31), (generic, 37)):
        generator = torch.Generator().manual_seed(seed)
        base = torch.randn(10, 8, generator=generator).to(torch.bfloat16)
        feature = torch.randn(10, 8, generator=generator).to(torch.bfloat16)
        mixing = torch.randn(8, 8, generator=generator) * 0.04
        target = (base.float() + feature.float() @ mixing + 0.125).to(
            torch.bfloat16
        )
        accumulator.observe(base, feature, target)
    summary = publish_layer_mlp_o1_registry(
        legacy_path,
        root / "published",
        layers=(18, 63),
        max_error_radius=0.25,
    )
    return summary.manifest_path, atlas, compute


_BUNDLE_RECEIPT = {
    "checkpoint_bytes": 55_000_000_000,
    "graph_revision": [1, "e" * 64],
    "kind": "complete-causal-bundle/v1",
    "layout_fingerprint": "b" * 64,
    "manifest_sha256": "c" * 64,
    "shards": 12,
    "shards_sha256": "d" * 64,
    "tensor_bindings": 1199,
    "weights_layout": "flat/v1",
}
_ANCHOR_SEED = torch.tensor([[[3.0]]], dtype=torch.float32)
_ANCHOR_PREFIX_SHA256 = token_prefix_sha256((11, 12))
_ANCHOR_SEED_FILE_SHA256 = "8" * 64
_ANCHOR_RECEIPT = AnchorReceipt.create(
    prefix_length=2,
    prefix_sha256=_ANCHOR_PREFIX_SHA256,
    boundary_kind="turn",
    semantic_label_sha256=None,
    snapshot_manifest_name=f"{_ANCHOR_PREFIX_SHA256}.json",
    snapshot_manifest_sha256="1" * 64,
    snapshot_manifest_bytes=10,
    snapshot_body_sha256="2" * 64,
    snapshot_body_bytes=9,
    snapshot_payload_name=f"{_ANCHOR_PREFIX_SHA256}.{'3' * 64}.npz",
    snapshot_payload_sha256="3" * 64,
    snapshot_payload_bytes=20,
    seed_hidden_name=(
        f"{_ANCHOR_PREFIX_SHA256}.{_ANCHOR_SEED_FILE_SHA256}.seed.safetensors"
    ),
    seed_hidden_sha256=_ANCHOR_SEED_FILE_SHA256,
    seed_hidden_bytes=5,
    seed_hidden_tensor_sha256=SemanticStateAnchorCache._seed_tensor_sha256(
        _ANCHOR_SEED
    ),
    seed_hidden_dtype="float32",
    seed_hidden_shape=(1, 1, 1),
    seed_hidden_source_device="cpu",
    state_bytes=321,
    created_sequence=1,
    last_access_sequence=2,
    hit_count=1,
    transport_neutral=False,
)
_RESTORED_ANCHOR = RestoredAnchor(
    anchor=_ANCHOR_RECEIPT,
    query_length=2,
    exact_prefix=True,
    seed_hidden=_ANCHOR_SEED,
)
_ANCHOR_MTP_DESCRIPTOR = MtpCarrySidecarDescriptor(
    basename=f"{'4' * 64}.qwen35-mtp-carry",
    bytes=256,
    file_sha256="4" * 64,
    identity_sha256="5" * 64,
    tensor_manifest_sha256="6" * 64,
)
_ANCHOR_MTP_CARRY = Qwen35MtpCarry(
    schema="fixture-mtp-carry/v1",
    identity=("fixture-mtp-runtime",),
    history=(11, 12),
    next_position=1,
    state=None,
    last_target_hidden=_ANCHOR_SEED.clone(),
)
_ANCHOR_MTP_VALUES = _ANCHOR_RECEIPT.to_document()
_ANCHOR_MTP_VALUES.pop("receipt_sha256")
_ANCHOR_MTP_VALUES.pop("schema")
_ANCHOR_MTP_VALUES["mtp_carry"] = _ANCHOR_MTP_DESCRIPTOR
_ANCHOR_MTP_RECEIPT = AnchorReceipt.create(**_ANCHOR_MTP_VALUES)
_RESTORED_ANCHOR_MTP = RestoredAnchor(
    anchor=_ANCHOR_MTP_RECEIPT,
    query_length=2,
    exact_prefix=True,
    seed_hidden=_ANCHOR_SEED,
    mtp_carry=_ANCHOR_MTP_CARRY,
    mtp_carry_bytes=_ANCHOR_MTP_DESCRIPTOR.bytes,
)


class _Tokenizer:
    def __init__(self, *, prompt_ids=(11, 12), decoded=" local answer ") -> None:
        self.prompt_ids = prompt_ids
        self.decoded = decoded
        self.encoded: list[str] = []
        self.decoded_ids: list[tuple[int, ...]] = []

    def encode(self, text: str):
        self.encoded.append(text)
        return self.prompt_ids

    def decode(self, token_ids):
        self.decoded_ids.append(tuple(token_ids))
        return self.decoded

    render_no_thinking_messages = staticmethod(
        Qwen38Tokenizer.render_no_thinking_messages
    )


class _Model:
    max_seq_len = 16

    def __init__(
        self,
        *,
        generated=(7, 8),
        generation_error: Exception | None = None,
        cleanup_error: Exception | None = None,
    ) -> None:
        self.config = SimpleNamespace(vocab_size=300_000)
        self.generated = tuple(generated)
        self.state_bytes = 0
        self.next_position = 0
        self.state_poisoned = False
        self.state_batch_size = None
        self._pending_block_stage = None
        self.delta_head_router = None
        self.delta_head_router_calls: list[object | None] = []
        self.pager = SimpleNamespace(release=Mock())
        self.generation_error = generation_error
        self.cleanup_error = cleanup_error
        self.calls: list[tuple[object, dict[str, object]]] = []
        self.reset_calls: list[bool] = []

    def generate_greedy(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if self.generation_error is not None:
            raise self.generation_error
        prompt_ids = tuple(prompt[0])
        retained = (
            len(self.generated)
            if kwargs.get("retain_final_state")
            else max(
                0,
                len(self.generated) - 1,
            )
        )
        self.next_position = len(prompt_ids) + retained
        self.state_batch_size = 1
        self.state_bytes = 456
        return self.generated, {
            "prompt_token_ids": prompt_ids,
            "generated_token_ids": self.generated,
            "context_mode": "stateful_autoregressive",
            "stateful_cache": True,
            "general_generation": True,
            "prefill_mode": "batched",
            "forward_passes": 3,
            "source_body_bytes": 1234,
            "linear_calls": 99,
            "seconds": 1.25,
            "state_bytes": 456,
            "stopped_on_eos": False,
        }

    def reset_state(self, *, release=False):
        self.reset_calls.append(release)
        if self.cleanup_error is not None:
            raise self.cleanup_error
        self.next_position = 0
        self.state_poisoned = False
        self.state_batch_size = None
        self.state_bytes = 0

    def set_delta_head_router(self, router):
        self.delta_head_router = router
        self.delta_head_router_calls.append(router)


class _LayerContextualModel(_Model):
    def __init__(self) -> None:
        super().__init__()
        self.config.dim = 8
        self.config.n_layers = 64
        self.layer_contextual_continuation_identity = None
        self.layer_contextual_attach_calls = []

    @staticmethod
    def layer_contextual_continuation_model_sha256() -> str:
        return "2" * 64

    @staticmethod
    def layer_contextual_continuation_q4_sha256() -> str:
        return "3" * 64

    def attach_layer_contextual_continuation(self, identity) -> None:
        self.layer_contextual_continuation_identity = identity
        self.layer_contextual_attach_calls.append(identity)

    def current_layer_contextual_continuation_transaction(self):
        return None

    def layer_contextual_continuation_transactions_since(self, _boundary=-1):
        return ()


class _StreamingModel(_Model):
    def generate_greedy(self, prompt, **kwargs):
        progress = kwargs.get("progress")
        if callable(progress):
            for step, token_id in enumerate(self.generated):
                progress(
                    {
                        "event": "generated_token",
                        "step": step,
                        "token_id": token_id,
                    }
                )
        return super().generate_greedy(prompt, **kwargs)


class _ComponentTimingModel(_Model):
    _COMPONENTS = (
        "full_attention_core",
        "deltanet_core",
        "mlp_core",
        "lm_head_core",
        "layer_transition_crystal",
        "layer_mlp_crystal",
    )

    def __init__(self, *, invalid_after_generation: bool = False) -> None:
        super().__init__()
        self.invalid_after_generation = invalid_after_generation
        self.generated_once = False
        self.component_counters = {
            component: {"calls": 10, "nanoseconds": 1_000}
            for component in self._COMPONENTS
        }

    def component_timing_metrics(self):
        if self.invalid_after_generation and self.generated_once:
            return {"schema": "broken"}
        return {
            "accounting_failures": 0,
            "clock": "time.perf_counter_ns",
            "components": {
                component: dict(row)
                for component, row in self.component_counters.items()
            },
            "schema": "immer.qwen3.8-component-timing-counters/v1",
            "unit": "nanoseconds",
        }

    def generate_greedy(self, prompt, **kwargs):
        generated, evidence = super().generate_greedy(prompt, **kwargs)
        increments = {
            "full_attention_core": (4, 400),
            "deltanet_core": (12, 1_200),
            "mlp_core": (16, 3_200),
            "lm_head_core": (2, 600),
            "layer_transition_crystal": (16, 80),
            "layer_mlp_crystal": (16, 96),
        }
        for component, (calls, nanoseconds) in increments.items():
            self.component_counters[component]["calls"] += calls
            self.component_counters[component]["nanoseconds"] += nanoseconds
        self.generated_once = True
        return generated, evidence


class _LayerMlpO1Model(_Model):
    def __init__(self, *, generation_error: Exception | None = None) -> None:
        super().__init__(generation_error=generation_error)
        self.config.n_layers = 64
        self.layer_mlp_o1_observer = None
        self.layer_mlp_o1_observers = {}
        self.layer_mlp_o1_observer_registry_calls = []
        self.o1_rows = 0
        self.o1_failures = 0
        self.o1_rows_by_layer = {}
        self.o1_failures_by_layer = {}

    def set_layer_mlp_o1_observer(self, observer) -> None:
        self.layer_mlp_o1_observer = observer
        if observer is None:
            self.layer_mlp_o1_observers.pop(63, None)
        else:
            self.layer_mlp_o1_observers[63] = observer

    def set_layer_mlp_o1_observers(self, observers) -> None:
        self.layer_mlp_o1_observers = dict(observers)
        self.layer_mlp_o1_observer = self.layer_mlp_o1_observers.get(63)
        self.layer_mlp_o1_observer_registry_calls.append(
            dict(self.layer_mlp_o1_observers)
        )

    def layer_mlp_o1_observer_metrics(self):
        layers = sorted(
            set(self.layer_mlp_o1_observers)
            | set(self.o1_rows_by_layer)
            | set(self.o1_failures_by_layer)
        )
        if all(layer == 63 for layer in layers):
            return {
                "failures": self.o1_failures,
                "rows": self.o1_rows,
                "schema": "immer.qwen3.8-layer63-mlp-o1-observer/v1",
            }
        return {
            "failures": self.o1_failures,
            "layers": {
                str(layer): {
                    "failures": self.o1_failures_by_layer.get(layer, 0),
                    "rows": self.o1_rows_by_layer.get(layer, 0),
                }
                for layer in layers
            },
            "registered_layers": sorted(self.layer_mlp_o1_observers),
            "rows": self.o1_rows,
            "schema": "immer.qwen3.8-layer-mlp-o1-observer-registry/v2",
        }

    def generate_greedy(self, prompt, **kwargs):
        generated, evidence = super().generate_greedy(prompt, **kwargs)
        for layer_index, observer in sorted(self.layer_mlp_o1_observers.items()):
            base = torch.arange(8, dtype=torch.float32).to(torch.bfloat16)
            feature = (base.float() + 1.0).to(torch.bfloat16)
            target = (base.float() + feature.float() * 0.1).to(torch.bfloat16)
            observer(
                base.reshape(1, 1, 8),
                feature.reshape(1, 1, 8),
                target.reshape(1, 1, 8),
            )
            self.o1_rows += 1
            self.o1_rows_by_layer[layer_index] = (
                self.o1_rows_by_layer.get(layer_index, 0) + 1
            )
        return generated, evidence


class _LayerMlpCrystalRegistryModel(_Model):
    def __init__(self, registry, *, fallback_only: bool = False) -> None:
        super().__init__()
        self.config.dim = 8
        self.config.n_layers = 64
        self.registry = registry
        self.fallback_only = fallback_only
        self.layer_mlp_crystal_registry_enabled_layers = frozenset({18})
        self.layer_mlp_crystal_enabled = False
        self.registry_enable_calls: list[tuple[int, ...]] = []
        self.legacy_enable_calls: list[bool] = []
        self.counters = {
            entry.layer_index: {
                "attempts": 0,
                "fallbacks": 0,
                "packed_weight_bytes_avoided": 0,
                "physical_transitions": 0,
                "replacements": 0,
                "skipped_q4_matrix_calls": 0,
                "transition_rows": 0,
            }
            for entry in registry.entries
        }

    def set_layer_mlp_crystal_registry_enabled(self, layers) -> None:
        selected = tuple(layers)
        self.registry_enable_calls.append(selected)
        self.layer_mlp_crystal_registry_enabled_layers = frozenset(selected)

    def set_layer_mlp_crystal_enabled(self, enabled: bool) -> None:
        self.legacy_enable_calls.append(enabled)
        self.layer_mlp_crystal_enabled = enabled

    def layer_mlp_crystal_registry_metrics(self):
        enabled = set(self.layer_mlp_crystal_registry_enabled_layers)
        if self.layer_mlp_crystal_enabled:
            enabled.add(63)
        entries = {entry.layer_index: entry for entry in self.registry.entries}
        layers = {
            str(layer): {
                "action_abi": entry.action_abi,
                "atlas_revision_sha256": self.registry.pins.atlas_revision_sha256,
                "bank_generation": "layer63-v1" if layer == 63 else "v2",
                "enabled": layer in enabled,
                "graph_revision_sha256": self.registry.pins.graph_revision_sha256,
                "identity_sha256": entry.bank_identity_sha256,
                "installed": True,
                "max_error_radius": entry.max_error_radius,
                "model_sha256": self.registry.pins.model_sha256,
                "q4_sha256": self.registry.pins.q4_sha256,
                "request_enabled": layer in enabled,
                **self.counters[layer],
            }
            for layer, entry in entries.items()
        }
        fields = tuple(next(iter(self.counters.values())))
        installed = list(self.registry.mounted_layers)
        enabled_layers = sorted(enabled)
        return {
            **{
                field: sum(row[field] for row in self.counters.values())
                for field in fields
            },
            "enabled_layers": enabled_layers,
            "installed_layers": installed,
            "layers": layers,
            "registered_layers": installed,
            "request_enabled_layers": enabled_layers,
            "schema": (
                "immer.qwen3.8-layer-mlp-residual-crystal-registry-metrics/v2"
            ),
        }

    def generate_greedy(self, prompt, **kwargs):
        generated, evidence = super().generate_greedy(prompt, **kwargs)
        enabled = set(self.layer_mlp_crystal_registry_enabled_layers)
        if self.layer_mlp_crystal_enabled:
            enabled.add(63)
        if enabled:
            if len(enabled) != 1:
                raise AssertionError("request enabled more than one MLP Crystal layer")
            layer = next(iter(enabled))
            counters = self.counters[layer]
            counters["attempts"] += 1
            if self.fallback_only:
                counters["fallbacks"] += 1
            else:
                entry = next(
                    item for item in self.registry.entries if item.layer_index == layer
                )
                counters["packed_weight_bytes_avoided"] += (
                    entry.packed_weight_bytes_avoided
                )
                counters["physical_transitions"] += 1
                counters["replacements"] += 1
                counters["skipped_q4_matrix_calls"] += 3
                counters["transition_rows"] += 1
        return generated, evidence


class _PrefixSinkhornModel(_Model):
    def __init__(self) -> None:
        super().__init__()
        self.native_head_crsa_enabled = True
        self.prefix_activation_calls: list[bool] = []
        self.prefix_head_rows = 0
        self.prefix_elements = 0

    def set_native_head_crsa_enabled(self, enabled: bool) -> None:
        self.native_head_crsa_enabled = enabled
        self.prefix_activation_calls.append(enabled)

    @staticmethod
    def prefix_sinkhorn_action_identity_sha256() -> str:
        return "a" * 64

    def native_prefix_sinkhorn_metrics(self):
        return {
            "action_identity_sha256": self.prefix_sinkhorn_action_identity_sha256(),
            "base_softmax_head_rows_skipped": self.prefix_head_rows,
            "base_softmax_probability_elements_skipped": self.prefix_elements,
            "enabled": self.native_head_crsa_enabled,
            "physical_replacement": True,
            "schema": "immer.qwen3.8-prefix-sinkhorn-action-metrics/v1",
        }

    def generate_greedy(self, prompt, **kwargs):
        generated, evidence = super().generate_greedy(prompt, **kwargs)
        if self.native_head_crsa_enabled:
            self.prefix_head_rows += 12
            self.prefix_elements += 144
        return generated, evidence


class _StreamingTokenizer(_Tokenizer):
    def decode(self, token_ids):
        ids = tuple(token_ids)
        self.decoded_ids.append(ids)
        pieces = {7: "Hello", 8: " world", 9: "!"}
        return "".join(pieces[token_id] for token_id in ids)


class _AnchorModel(_Model):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=300_000, n_layers=64)
        self.next_position = 0
        self.state_poisoned = False
        self.state_bytes = 0
        self.state_batch_size = None

    def generate_greedy(self, prompt, **kwargs):
        if "restored_prefix_length" not in kwargs:
            return super().generate_greedy(prompt, **kwargs)
        self.calls.append((prompt, kwargs))
        if kwargs.get("restored_prefix_length") != 2:
            raise AssertionError("exact prefix was not passed to generation")
        if kwargs.get("restored_seed_hidden") is not _RESTORED_ANCHOR.seed_hidden:
            raise AssertionError("authenticated head seed was not passed through")
        return self.generated, {
            "prompt_token_ids": tuple(prompt[0]),
            "generated_token_ids": self.generated,
            "context_mode": "stateful_autoregressive",
            "stateful_cache": True,
            "general_generation": True,
            "prefill_mode": "batched",
            "forward_passes": 1,
            "source_body_bytes": 600,
            "linear_calls": 66,
            "seconds": 0.75,
            "state_bytes": 654,
            "stopped_on_eos": False,
        }

    def reset_state(self, *, release=False):
        super().reset_state(release=release)
        self.next_position = 0
        self.state_poisoned = False
        self.state_bytes = 0
        self.state_batch_size = None


class _UnderreportedAnchorModel(_AnchorModel):
    def generate_greedy(self, prompt, **kwargs):
        generated, evidence = super().generate_greedy(prompt, **kwargs)
        return generated, {**evidence, "forward_passes": 0}


def _anchor_cache(
    *,
    failure: Exception | None = None,
    miss: bool = False,
    restored: RestoredAnchor = _RESTORED_ANCHOR,
):
    cache = object.__new__(SemanticStateAnchorCache)

    def restore(model, _token_ids, **_restore_options):
        if miss:
            return None
        model.next_position = 2
        model.state_bytes = _ANCHOR_RECEIPT.state_bytes
        model.state_batch_size = 1
        if failure is not None:
            raise failure
        return restored

    cache.restore_deepest = Mock(side_effect=restore)
    return cache


class _Runtime:
    tokenizer_sha256 = _DIGEST
    bundle_receipt = _BUNDLE_RECEIPT

    def __init__(self, model=None, tokenizer=None) -> None:
        self.model = _Model() if model is None else model
        self.tokenizer = _Tokenizer() if tokenizer is None else tokenizer
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _FastMount:
    def __init__(
        self,
        *,
        aux_bytes: int = 20,
        cumulative: tuple[int, ...] | None = None,
    ) -> None:
        self.cumulative = (0, aux_bytes) if cumulative is None else cumulative
        self.calls = 0

    def metrics(self):
        total = self.cumulative[min(self.calls, len(self.cumulative) - 1)]
        self.calls += 1
        return {
            "pilot_source_body_bytes": 3 * total // 5,
            "transpose_source_body_bytes": total - 3 * total // 5,
            "source_body_bytes": total,
            "pilot_logical_weight_bytes": 3 * total // 5,
            "transpose_logical_weight_bytes": total - 3 * total // 5,
            "logical_weight_bytes": total,
            "pilot_row_reads": self.calls - 1,
            "transpose_row_reads": self.calls - 1,
        }


class _ExactHeadMetrics:
    def __init__(self) -> None:
        self.calls = 0

    def metrics(self):
        current = self.calls
        self.calls += 1
        return {
            "calls": current,
            "pages_pruned": 2 * current,
            "rows_pruned": 8 * current,
            "last_fallback_reason": "",
            "manifest_sha256": _EXACT_HEAD_RECEIPT["manifest_sha256"],
        }


_FAST_RECEIPT = {
    "active_layers": [0, 9],
    "affine_fit_sha256": "1" * 64,
    "execution": "row-routed-sparse-mlp",
    "fitted_layers": [0, 9],
    "model_pin_sha256": "2" * 64,
    "pilot_manifest_body_sha256": "3" * 64,
    "router_fit_sha256": "4" * 64,
    "schema": "immer.qwen3.8-fast-mlp-mount/v1",
    "selected_neuron_fraction_by_layer": {"0": 0.2, "9": 0.2},
    "transport_row_fraction_by_layer": {"0": 0.2, "9": 0.2},
    "transpose_manifest_sha256": "5" * 64,
    "weights_index_sha256": "6" * 64,
}

_EXACT_HEAD_RECEIPT = {
    "index_bytes": 67_000_000,
    "manifest_sha256": "7" * 64,
    "payload_bytes": 68_000_000,
    "payload_sha256": "8" * 64,
    "tensor_sha256": "9" * 64,
}


class _InjectedChat(Qwen38CausalChat):
    def __init__(self, factory, *args, **kwargs) -> None:
        self._injected_factory = factory
        super().__init__(*args, **kwargs)

    def _open_runtime(self):
        return self._injected_factory()


def _chat(runtime: _Runtime, **overrides) -> Qwen38CausalChat:
    factory = overrides.pop("runtime_factory", lambda: runtime)
    options = {
        "max_prompt_tokens": 8,
        "max_new_tokens": 3,
        "max_context_tokens": 16,
        "head_block_rows": 17,
    }
    options.update(overrides)
    return _InjectedChat(
        factory,
        "unused.causal",
        "unused-tokenizer.json",
        **options,
    )


class _ExactBackend:
    name = "exact.fixture"

    def handle(self, request: Request) -> Result:
        return Result(ExecutionStatus.ABSTAINED, self.name, reason="fixture")


class Qwen38CausalChatTests(unittest.TestCase):
    def setUp(self) -> None:
        # Beast has the real deployment tree. Unit tests must never let an
        # auto-resolved CLI path write fixture receipts into production state.
        deployment_state = patch(
            "immer.cli._QWEN38_DEPLOYMENT_STATE",
            Path("/__immer_test_no_deployment_state__"),
        )
        deployment_state.start()
        self.addCleanup(deployment_state.stop)

    def test_layer_mlp_o1_policy_is_path_free_and_binds_authorities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atlas_root = root / "atlas"
            compute_root = root / "compute"
            _semantic_atlas_authority(atlas_root)
            _compute_graph_authority(compute_root)
            args = SimpleNamespace(
                layer_mlp_o1_state=root / "private-o1.json",
                layer_mlp_o1_atlas=atlas_root,
                layer_mlp_o1_compute_root=compute_root,
                layer_mlp_o1_sketch_dim=128,
                layer_mlp_o1_seed=(
                    "2b188a99b4b36f51bd910866e6d9a007fd02ec7256d6596fc87eb76f0444eccf"
                ),
                layer_mlp_o1_ridge=1e-8,
                layer_mlp_o1_coverage_guard=0.0,
                layer_mlp_o1_error_guard=0.0,
                layer_mlp_o1_queue_capacity=8,
            )
            policy = _qwen38_layer_mlp_o1_policy(args)
            self.assertIsNotNone(policy)
            assert policy is not None
            encoded = json.dumps(policy, sort_keys=True)
            self.assertNotIn(str(root), encoded)
            self.assertEqual(policy["sketch_dim"], 128)
            self.assertEqual(policy["queue_capacity"], 8)
            self.assertIsNone(policy["identity"])
            self.assertEqual(len(policy["authorities"]["atlas_revision_sha256"]), 64)
            self.assertEqual(
                len(policy["authorities"]["compute_revision_sha256"]),
                64,
            )

    def test_layer_mlp_o1_policy_uses_existing_state_pinned_authorities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atlas_root = root / "atlas"
            compute_root = root / "compute"
            atlas_revision = _semantic_atlas_authority(atlas_root)
            graph_revision = _compute_graph_authority(compute_root)
            live_sequence, live_event = LiveGraph(atlas_root).store.revision()
            self.assertNotEqual(
                GraphRevision(live_sequence, live_event).sha256,
                atlas_revision,
            )
            self.assertNotEqual(
                ComputeOperatorGraph(ComputeCrystalBank(compute_root)).state().sha256,
                graph_revision,
            )
            identity = _layer_mlp_identity(
                graph=graph_revision,
                atlas=atlas_revision,
            )
            state_path = root / "o1.json"
            accumulator = Layer63MlpO1Accumulator(
                state_path,
                identity,
                packed_weight_bytes_avoided=123,
            )
            rows = torch.arange(48, dtype=torch.float32).reshape(6, 8)
            base = rows.to(torch.bfloat16)
            feature = (rows + 1).to(torch.bfloat16)
            target = (rows + 2).to(torch.bfloat16)
            accumulator.observe(base, feature, target)
            args = SimpleNamespace(
                layer_mlp_o1_state=state_path,
                layer_mlp_o1_atlas=atlas_root,
                layer_mlp_o1_compute_root=compute_root,
                layer_mlp_o1_sketch_dim=3,
                layer_mlp_o1_seed="1" * 64,
                layer_mlp_o1_ridge=1e-8,
                layer_mlp_o1_coverage_guard=0.0,
                layer_mlp_o1_error_guard=0.0,
                layer_mlp_o1_queue_capacity=8,
            )
            policy = _qwen38_layer_mlp_o1_policy(args)

        assert policy is not None
        self.assertEqual(
            policy["authorities"],
            {
                "atlas_revision_sha256": atlas_revision,
                "compute_revision_sha256": graph_revision,
            },
        )
        self.assertEqual(
            policy["identity"]["identity_sha256"], identity.identity_sha256
        )

    def test_multilayer_o1_policy_loads_only_one_immutable_authority_state(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atlas_root = root / "atlas"
            compute_root = root / "compute"
            atlas_revision = _semantic_atlas_authority(atlas_root)
            graph_revision = _compute_graph_authority(compute_root)
            state_path = root / "layer63.json"
            layer63 = Layer63MlpO1Accumulator(
                state_path,
                _layer_mlp_identity(
                    graph=graph_revision,
                    atlas=atlas_revision,
                ),
                packed_weight_bytes_avoided=123,
            )
            layer00 = LayerMlpO1Accumulator(
                _layer_mlp_o1_state_path_for_layer(
                    state_path,
                    layer_index=0,
                    sketch_dim=3,
                ),
                _generic_layer_mlp_identity(
                    0,
                    graph=graph_revision,
                    atlas=atlas_revision,
                ),
                packed_weight_bytes_avoided=123,
            )
            rows = torch.arange(48, dtype=torch.float32).reshape(6, 8)
            for accumulator in (layer63, layer00):
                accumulator.observe(
                    rows.to(torch.bfloat16),
                    (rows + 1).to(torch.bfloat16),
                    (rows + 2).to(torch.bfloat16),
                )
            args = SimpleNamespace(
                layer_mlp_o1_state=state_path,
                layer_mlp_o1_atlas=atlas_root,
                layer_mlp_o1_compute_root=compute_root,
                layer_mlp_o1_layers=(0, 63),
                layer_mlp_o1_sketch_dim=3,
                layer_mlp_o1_seed="1" * 64,
                layer_mlp_o1_ridge=1e-8,
                layer_mlp_o1_coverage_guard=0.0,
                layer_mlp_o1_error_guard=0.0,
                layer_mlp_o1_queue_capacity=8,
            )
            with (
                patch.object(
                    Layer63MlpO1Accumulator,
                    "load",
                    wraps=Layer63MlpO1Accumulator.load,
                ) as legacy_load,
                patch.object(
                    LayerMlpO1Accumulator,
                    "load",
                    wraps=LayerMlpO1Accumulator.load,
                ) as generic_load,
            ):
                policy = _qwen38_layer_mlp_o1_policy(args)

        assert policy is not None
        legacy_load.assert_called_once_with(state_path)
        generic_load.assert_not_called()
        self.assertEqual(policy["layers"], [0, 63])
        self.assertEqual(
            policy["schema"],
            "immer.qwen3.8-layer-mlp-o1-policy/v2",
        )
        self.assertNotIn("registry", policy)
        self.assertNotIn("identity", policy)

    def test_layer_mlp_o1_layers_require_a_sorted_decoder_subset(self) -> None:
        invalid = ((), (1, 0), (0, 0), (-1,), (64,), (True,))
        for layers in invalid:
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                _chat(
                    _Runtime(),
                    layer_mlp_o1_state_path="/tmp/o1.json",
                    layer_mlp_o1_layers=layers,
                )
        with self.assertRaisesRegex(
            ValueError,
            "requires layer_mlp_o1_state_path",
        ):
            _chat(_Runtime(), layer_mlp_o1_layers=(0, 63))

    def test_passive_layer_mlp_o1_enqueues_only_after_successful_generation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "o1.json"
            accumulator = Layer63MlpO1Accumulator(
                state_path,
                _layer_mlp_identity(),
                packed_weight_bytes_avoided=123,
            )
            worker = Layer63MlpO1AsyncWorker(accumulator, queue_capacity=2)
            runtime = _Runtime(model=_LayerMlpO1Model())
            chat = _chat(runtime)
            chat._load_locked()
            chat._layer_mlp_o1_worker = worker
            runtime.layer_mlp_o1_worker = worker

            result = chat.handle(Request("chat", "hello"))
            self.assertEqual(result.status, ExecutionStatus.OK)
            self.assertEqual(result.output, "local answer")
            request = result.evidence["layer_mlp_o1_collection"]["request"]
            self.assertEqual(request["captured_rows"], 1)
            self.assertTrue(request["enqueued"])
            self.assertEqual(request["status"], "enqueued")
            self.assertIsNone(request["accounting_error"])
            self.assertTrue(worker.flush(timeout=5))
            metrics = worker.metrics()
            self.assertEqual(metrics["settled_rows"], 1)
            serialized = json.dumps(
                result.evidence["layer_mlp_o1_collection"],
                sort_keys=True,
            )
            self.assertNotIn(str(state_path), serialized)
            worker.close()
            chat._layer_mlp_o1_worker = None
            chat.close()

            failed_worker = Layer63MlpO1AsyncWorker(
                Layer63MlpO1Accumulator(
                    Path(temporary) / "failed.json",
                    _layer_mlp_identity(),
                    packed_weight_bytes_avoided=123,
                ),
                queue_capacity=1,
            )
            failed_runtime = _Runtime(
                model=_LayerMlpO1Model(generation_error=RuntimeError("generation"))
            )
            failed_chat = _chat(failed_runtime)
            failed_chat._load_locked()
            failed_chat._layer_mlp_o1_worker = failed_worker
            failed_runtime.layer_mlp_o1_worker = failed_worker
            failed = failed_chat.handle(Request("chat", "hello"))
            self.assertEqual(failed.status, ExecutionStatus.ERROR)
            self.assertEqual(failed_worker.metrics()["queued_batches"], 0)
            failed_worker.close()
            failed_chat._layer_mlp_o1_worker = None
            failed_chat.close()

    def test_passive_layer_mlp_o1_accounting_error_cannot_fail_output(self) -> None:
        class BrokenMetricsModel(_LayerMlpO1Model):
            def __init__(self) -> None:
                super().__init__()
                self.metric_calls = 0

            def layer_mlp_o1_observer_metrics(self):
                self.metric_calls += 1
                if self.metric_calls == 1:
                    return super().layer_mlp_o1_observer_metrics()
                return {"rows": -1, "failures": -1}

        with tempfile.TemporaryDirectory() as temporary:
            worker = Layer63MlpO1AsyncWorker(
                Layer63MlpO1Accumulator(
                    Path(temporary) / "o1.json",
                    _layer_mlp_identity(),
                    packed_weight_bytes_avoided=123,
                )
            )
            runtime = _Runtime(model=BrokenMetricsModel())
            chat = _chat(runtime)
            chat._load_locked()
            chat._layer_mlp_o1_worker = worker
            runtime.layer_mlp_o1_worker = worker

            result = chat.handle(Request("chat", "hello"))

            self.assertTrue(result.ok, result.reason)
            request = result.evidence["layer_mlp_o1_collection"]["request"]
            self.assertEqual(request["status"], "accounting-error")
            self.assertFalse(request["enqueued"])
            self.assertIn("ValueError", request["accounting_error"])
            self.assertEqual(worker.metrics()["queued_batches"], 0)
            worker.close()
            chat._layer_mlp_o1_worker = None
            chat.close()

    def test_layer_mlp_o1_runtime_builds_authenticated_identity_without_bank_mount(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atlas_root = root / "atlas"
            compute_root = root / "compute"
            _semantic_atlas_authority(atlas_root)
            _compute_graph_authority(compute_root)
            model = _LayerMlpO1Model()
            model.config.dim = 8
            model.layer_mlp_crystal_model_sha256 = lambda: "2" * 64
            model.layer_mlp_crystal_q4_sha256 = lambda: "3" * 64
            model._layer_mlp_crystal_avoided_q4_bytes = lambda: 456
            runtime = _Runtime(model=model)
            chat = _chat(
                runtime,
                q4_root="/fixture/q4",
                compute_dtype="bfloat16",
                layer_mlp_o1_state_path=root / "o1.json",
                layer_mlp_o1_atlas_path=atlas_root,
                layer_mlp_o1_compute_root=compute_root,
                layer_mlp_o1_sketch_dim=3,
                layer_mlp_o1_projection_seed="1" * 64,
            )
            chat._load_locked()
            worker = chat._layer_mlp_o1_worker
            self.assertIsNotNone(worker)
            assert worker is not None
            identity = worker.accumulator.identity
            self.assertEqual(identity.model_sha256, "2" * 64)
            self.assertEqual(identity.q4_sha256, "3" * 64)
            self.assertEqual(identity.hidden_dim, 8)
            self.assertEqual(identity.sketch_dim, 3)
            self.assertIsNone(chat._layer_mlp_crystal_bank)
            worker.close()
            chat._layer_mlp_o1_worker = None
            chat.close()

    def test_multilayer_o1_rotates_least_charged_layer_and_restores_registry(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atlas_root = root / "atlas"
            compute_root = root / "compute"
            _semantic_atlas_authority(atlas_root)
            _compute_graph_authority(compute_root)
            model = _LayerMlpO1Model()
            model.config.dim = 8
            model.layer_mlp_crystal_model_sha256 = lambda: "2" * 64
            model.layer_mlp_crystal_q4_sha256 = lambda: "3" * 64
            model._layer_mlp_crystal_avoided_q4_bytes = lambda: 456

            def sentinel(*_args) -> None:
                return None

            model.set_layer_mlp_o1_observers({7: sentinel})
            runtime = _Runtime(model=model)
            state_path = root / "layer63.json"
            chat = _chat(
                runtime,
                q4_root="/fixture/q4",
                compute_dtype="bfloat16",
                layer_mlp_o1_state_path=state_path,
                layer_mlp_o1_atlas_path=atlas_root,
                layer_mlp_o1_compute_root=compute_root,
                layer_mlp_o1_layers=(0, 63),
                layer_mlp_o1_sketch_dim=3,
                layer_mlp_o1_projection_seed="1" * 64,
            )
            chat._load_locked()
            pool = chat._layer_mlp_o1_pool
            self.assertIsInstance(pool, LayerMlpO1AsyncPool)
            assert pool is not None
            self.assertIsNone(chat._layer_mlp_o1_worker)

            first = chat.handle(Request("chat", "first"))
            second = chat.handle(Request("chat", "second"))

            self.assertTrue(first.ok, first.reason)
            self.assertTrue(second.ok, second.reason)
            first_collection = first.evidence["layer_mlp_o1_collection"]
            second_collection = second.evidence["layer_mlp_o1_collection"]
            self.assertEqual(first_collection["request"]["layer_index"], 0)
            self.assertEqual(second_collection["request"]["layer_index"], 63)
            self.assertEqual(first_collection["layers"], [0, 63])
            self.assertEqual(set(first_collection["registry"]), {"0", "63"})
            self.assertEqual(model.layer_mlp_o1_observers, {7: sentinel})
            self.assertTrue(pool.flush(timeout=5))
            metrics = pool.metrics()
            self.assertEqual(metrics["layers"]["0"]["settled_rows"], 1)
            self.assertEqual(metrics["layers"]["63"]["settled_rows"], 1)
            self.assertTrue(state_path.is_file())
            layer00_path = _layer_mlp_o1_state_path_for_layer(
                state_path,
                layer_index=0,
                sketch_dim=3,
            )
            self.assertTrue(layer00_path.is_file())
            self.assertEqual(
                LayerMlpO1Accumulator.load(layer00_path).identity.layer_index,
                0,
            )
            self.assertIsInstance(
                Layer63MlpO1Accumulator.load(state_path).identity,
                Layer63MlpResidualCrystalIdentity,
            )

            with patch.object(pool, "submit", side_effect=RuntimeError("queue")):
                isolated = chat.handle(Request("chat", "third"))
            self.assertTrue(isolated.ok, isolated.reason)
            request = isolated.evidence["layer_mlp_o1_collection"]["request"]
            self.assertEqual(request["status"], "accounting-error")
            self.assertIn("RuntimeError", request["accounting_error"])
            self.assertEqual(model.layer_mlp_o1_observers, {7: sentinel})

            pool.close()
            chat._layer_mlp_o1_pool = None
            chat._layer_mlp_o1_registry = {}
            runtime.layer_mlp_o1_pool = None
            chat.close()

    def test_multilayer_o1_runtime_prefers_legacy_authority_like_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atlas_root = root / "atlas"
            compute_root = root / "compute"
            legacy_atlas = _semantic_atlas_authority(atlas_root)
            legacy_graph = _compute_graph_authority(compute_root)
            sequence, event_sha256 = LiveGraph(atlas_root).store.revision()
            newer_atlas = GraphRevision(sequence, event_sha256).sha256
            newer_graph = ComputeOperatorGraph(
                ComputeCrystalBank(compute_root)
            ).state().sha256
            self.assertNotEqual(legacy_atlas, newer_atlas)
            self.assertNotEqual(legacy_graph, newer_graph)

            state_path = root / "layer63.json"
            legacy = Layer63MlpO1Accumulator(
                state_path,
                _layer_mlp_identity(graph=legacy_graph, atlas=legacy_atlas),
                packed_weight_bytes_avoided=456,
            )
            generic = LayerMlpO1Accumulator(
                _layer_mlp_o1_state_path_for_layer(
                    state_path,
                    layer_index=0,
                    sketch_dim=3,
                ),
                _generic_layer_mlp_identity(
                    0,
                    graph=newer_graph,
                    atlas=newer_atlas,
                ),
                packed_weight_bytes_avoided=456,
            )
            rows = torch.arange(48, dtype=torch.float32).reshape(6, 8)
            for accumulator in (legacy, generic):
                accumulator.observe(
                    rows.to(torch.bfloat16),
                    (rows + 1).to(torch.bfloat16),
                    (rows + 2).to(torch.bfloat16),
                )

            model = _LayerMlpO1Model()
            model.config.dim = 8
            model.layer_mlp_crystal_model_sha256 = lambda: "2" * 64
            model.layer_mlp_crystal_q4_sha256 = lambda: "3" * 64
            model._layer_mlp_crystal_avoided_q4_bytes = lambda: 456
            chat = _chat(
                _Runtime(model=model),
                q4_root="/fixture/q4",
                compute_dtype="bfloat16",
                layer_mlp_o1_state_path=state_path,
                layer_mlp_o1_atlas_path=atlas_root,
                layer_mlp_o1_compute_root=compute_root,
                layer_mlp_o1_layers=(0, 63),
                layer_mlp_o1_sketch_dim=3,
                layer_mlp_o1_projection_seed="1" * 64,
            )

            with self.assertRaisesRegex(
                Qwen38ChatError,
                "layer-0 MLP O1 state differs",
            ):
                chat._load_locked()
            chat.close()

    def test_markov_atlas_loads_once_and_binds_runtime_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "atlas.bin"
            atlas = MarkovTokenAtlas.build(
                ((1, 2, 3, 4), (5, 2, 3, 6)),
                vocab_size=300_000,
                tokenizer_sha256=_DIGEST,
                max_order=3,
            )
            atlas.write(path)
            chat = _chat(
                _Runtime(),
                draft_mode="markov",
                markov_atlas_path=path,
            )

            runtime = chat._load_locked()
            loaded = chat._markov_atlas
            self.assertIs(runtime.model.config, chat._runtime.model.config)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.sha256, atlas.sha256)
            bound_identity = chat._draft_window_runtime_identity()
            chat._markov_atlas = None
            unbound_identity = chat._draft_window_runtime_identity()
            chat._markov_atlas = loaded
            self.assertNotEqual(bound_identity, unbound_identity)
            chat.close()

    def test_o1_retention_loads_once_with_runtime_identity(self) -> None:
        retention = Mock()
        retention.metrics.return_value = {"sequence": 0}
        chat = _chat(
            _Runtime(),
            draft_mode="markov",
            markov_o1_retention_path="/state/o1-retention.json",
        )
        with patch(
            "immer.runtimes.qwen3_8.adapter.O1MarkovRetention",
            return_value=retention,
        ) as constructor:
            chat._load_locked()

        constructor.assert_called_once_with(
            Path("/state/o1-retention.json"),
            vocab_size=300_000,
            tokenizer_sha256=_DIGEST,
        )
        self.assertIs(chat._markov_o1_retention, retention)
        self.assertEqual(
            chat._base_evidence()["o1_markov_retention"],
            {"sequence": 0},
        )
        with patch(
            "immer.runtimes.qwen3_8.adapter._digest",
            side_effect=lambda value: value,
        ):
            runtime_identity = chat._draft_window_runtime_identity()
        self.assertEqual(
            runtime_identity["provider"]["o1_markov_retention"]["ppm_working_set"],
            "o1-priority+ricci-age-whole-answer/v1",
        )
        chat.close()

    def test_contextual_continuation_bank_mounts_with_runtime_identity(self) -> None:
        runtime = _Runtime()
        runtime.model.config.dim = 8
        runtime.q4_bank = SimpleNamespace(
            identity={"manifest_sha256": "f" * 64},
        )
        runtime.mlp_page_router = None
        runtime.delta_head_receipt = None
        with tempfile.TemporaryDirectory() as temporary:
            configured = Path(temporary) / "context.json"

            def open_bank(path, identity):
                return SimpleNamespace(
                    state_path=path,
                    identity=identity,
                    metrics=Mock(return_value=SimpleNamespace(to_dict=lambda: {})),
                )

            chat = _chat(
                runtime,
                draft_mode="markov",
                q4_root="/q4",
                contextual_continuation_state_path=configured,
            )
            with (
                patch(
                    "immer.runtimes.qwen3_8.adapter.asdict",
                    return_value={"dim": 8},
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.ContextualContinuationBank",
                    side_effect=open_bank,
                ) as constructor,
            ):
                chat._load_locked()

        state_path, identity = constructor.call_args.args
        self.assertEqual(identity.hidden_width, 8)
        self.assertEqual(identity.q4_sha256, "f" * 64)
        self.assertEqual(identity.tokenizer_sha256, _DIGEST)
        self.assertIn(".context-", state_path.name)
        self.assertEqual(state_path.suffix, ".json")
        self.assertIs(chat._contextual_continuation_bank.identity, identity)
        chat.close()

    def test_layer_contextual_bank_attaches_canonical_path_free_identity(self) -> None:
        model = _LayerContextualModel()
        runtime = _Runtime(model=model)
        with tempfile.TemporaryDirectory() as temporary:
            configured = Path(temporary) / "layer-context.json"
            chat = _chat(
                runtime,
                draft_mode="markov",
                q4_root="/q4",
                layer_contextual_continuation_state_path=configured,
                layer_contextual_continuation_max_cells=32,
                layer_contextual_continuation_max_receipts=64,
            )

            chat._load_locked()

            bank = chat._layer_contextual_continuation_bank
            self.assertIsInstance(bank, LayerContextualContinuationBank)
            assert bank is not None
            identity = bank.identity
            self.assertEqual(identity.model_sha256, "2" * 64)
            self.assertEqual(identity.q4_sha256, "3" * 64)
            self.assertEqual(identity.tokenizer_sha256, _DIGEST)
            self.assertEqual(
                identity.layers,
                DEFAULT_LAYER_CONTEXTUAL_CONTINUATION_LAYERS,
            )
            self.assertEqual(identity.sketch_dim, DEFAULT_LAYER_CONTEXTUAL_SKETCH_DIM)
            self.assertEqual(
                identity.projection_seed,
                DEFAULT_LAYER_CONTEXTUAL_CONTINUATION_PROJECTION_SEED,
            )
            self.assertEqual(identity.projection_abi, LAYER_CONTEXTUAL_PROJECTION_ABI)
            self.assertEqual(identity.stage, LAYER_CONTEXTUAL_STAGE)
            self.assertNotEqual(bank.state_path, configured)
            self.assertIn(identity.identity_sha256[:16], bank.state_path.name)
            self.assertNotIn(str(configured), json.dumps(identity.to_record()))
            self.assertIs(model.layer_contextual_attach_calls[0], identity)

            chat.close()

        self.assertIsNone(model.layer_contextual_attach_calls[-1])

    def test_layer_contextual_tracker_is_absent_without_markov(self) -> None:
        model = _LayerContextualModel()
        chat = _chat(_Runtime(model=model), q4_root="/q4")

        chat._load_locked()
        chat.close()

        self.assertIsNone(chat._layer_contextual_continuation_bank)
        self.assertEqual(model.layer_contextual_attach_calls, [])
        with self.assertRaisesRegex(ValueError, "Markov or hybrid"):
            _chat(
                _Runtime(model=_LayerContextualModel()),
                draft_mode="mtp",
                q4_root="/q4",
                layer_contextual_continuation_state_path="/state/layer.json",
            )

    def test_layer_contextual_bank_and_model_callables_reach_markov_provider(
        self,
    ) -> None:
        model = _LayerContextualModel()
        runtime = _Runtime(model=model)
        provider = SimpleNamespace(
            close=Mock(),
            metrics=Mock(
                return_value=SimpleNamespace(
                    source_body_bytes=0,
                    linear_calls=0,
                    to_dict=lambda: {},
                )
            ),
        )
        row = SimpleNamespace(
            accepted_prefix_length=0,
            emitted_token_ids=(7, 8),
            forward_passes=1,
            proposed_token_ids=(),
            provider_proposed_token_ids=(),
            round_index=0,
            round_policy=None,
            target_token_ids=(7, 8),
            window_size=4,
        )
        generated = SimpleNamespace(
            token_ids=(7, 8),
            evidence=SimpleNamespace(
                accepted_draft_tokens=0,
                adaptive_windows=True,
                final_state_committed=False,
                forward_passes=1,
                generated_token_ids=(7, 8),
                linear_calls=1,
                prefill_forward_passes=0,
                prompt_token_ids=(11, 12),
                rounds=(row,),
                schema="fixture.layer-context-generation/v1",
                seconds=0.1,
                source_body_bytes=1,
                state_bytes=0,
                stopped_on_eos=False,
                used_window_sizes=(4,),
                window_size=4,
            ),
        )
        decoder = SimpleNamespace(generate_rolling=Mock(return_value=generated))
        with tempfile.TemporaryDirectory() as temporary:
            chat = _chat(
                runtime,
                draft_mode="markov",
                draft_window=4,
                max_new_tokens=4,
                q4_root="/q4",
                layer_contextual_continuation_state_path=(
                    Path(temporary) / "layer.json"
                ),
            )
            chat._load_locked()
            with (
                patch(
                    "immer.runtimes.qwen3_8.adapter."
                    "FingerprintRollingK4DraftProvider",
                    return_value=provider,
                ) as constructor,
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38K4SpeculativeDecoder",
                    return_value=decoder,
                ),
            ):
                chat._generate_locked(
                    runtime,
                    (11, 12),
                    {
                        "eos_token_ids": (IM_END_TOKEN_ID,),
                        "head_block_rows": 17,
                        "max_new_tokens": 4,
                        "prefill_tokenwise": False,
                    },
                )

            options = constructor.call_args.kwargs
            self.assertIs(
                options["layer_contextual_continuation_bank"],
                chat._layer_contextual_continuation_bank,
            )
            self.assertIs(
                options["layer_contextual_current_transaction"].__self__,
                model,
            )
            self.assertIs(
                options["layer_contextual_transactions_since"].__self__,
                model,
            )
            self.assertEqual(model.calls, [])
            decoder.generate_rolling.assert_called_once()
            chat.close()

    def test_layer_contextual_attach_precedes_anchor_and_evidence_is_path_free(
        self,
    ) -> None:
        model = _AnchorModel()
        model.generated = (7,)
        model.config.dim = 8
        model.layer_contextual_continuation_identity = None
        model.layer_contextual_attach_calls = []
        model.layer_contextual_continuation_model_sha256 = lambda: "2" * 64
        model.layer_contextual_continuation_q4_sha256 = lambda: "3" * 64

        def attach(identity) -> None:
            model.layer_contextual_continuation_identity = identity
            model.layer_contextual_attach_calls.append(identity)

        model.attach_layer_contextual_continuation = attach
        model.current_layer_contextual_continuation_transaction = lambda: None
        model.layer_contextual_continuation_transactions_since = (
            lambda _boundary=-1: ()
        )
        anchor_generate = model.generate_greedy

        def generate(prompt, **kwargs):
            generated, evidence = anchor_generate(prompt, **kwargs)
            return generated, {**evidence, "forward_passes": 0}

        model.generate_greedy = generate
        cache = _anchor_cache()
        restore = cache.restore_deepest.side_effect

        def assert_attached(target, token_ids, **restore_options):
            self.assertIsNotNone(target.layer_contextual_continuation_identity)
            return restore(target, token_ids, **restore_options)

        cache.restore_deepest.side_effect = assert_attached
        with tempfile.TemporaryDirectory() as temporary:
            configured = Path(temporary) / "private-layer-bank.json"
            chat = _chat(
                _Runtime(model=model),
                anchor_cache=cache,
                draft_mode="markov",
                max_new_tokens=1,
                q4_root="/q4",
                layer_contextual_continuation_state_path=configured,
            )

            result = chat.handle(Request("chat", "hello"))

            self.assertTrue(result.ok, result.reason)
            layer = result.evidence["layer_contextual_continuation"]
            self.assertEqual(layer["schema"], LAYER_CONTEXTUAL_CONTINUATION_EVIDENCE_SCHEMA)
            self.assertEqual(
                layer["metrics"]["schema"],
                LAYER_CONTEXTUAL_CONTINUATION_METRICS_SCHEMA,
            )
            self.assertEqual(
                layer["request"]["schema"],
                LAYER_CONTEXTUAL_CONTINUATION_REQUEST_SCHEMA,
            )
            self.assertEqual(
                sorted(int(value) for value in layer["request"]["layers"]),
                list(DEFAULT_LAYER_CONTEXTUAL_CONTINUATION_LAYERS),
            )
            serialized = json.dumps(layer, sort_keys=True)
            self.assertNotIn(str(configured), serialized)
            self.assertNotIn("hello", serialized)
            self.assertNotIn("keys", serialized)
            self.assertNotIn("hidden_states", serialized)
            self.assertNotIn('"text"', serialized)
            self.assertEqual(result.evidence["generation"]["forward_passes"], 0)
            chat.close()

    def test_layer_contextual_persistent_corruption_fails_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            configured = Path(temporary) / "layer.json"
            first = _chat(
                _Runtime(model=_LayerContextualModel()),
                draft_mode="markov",
                q4_root="/q4",
                layer_contextual_continuation_state_path=configured,
            )
            first._load_locked()
            bank = first._layer_contextual_continuation_bank
            assert bank is not None
            bank.state_path.write_bytes(b"not an authenticated bank")
            first.close()
            second = _chat(
                _Runtime(model=_LayerContextualModel()),
                draft_mode="markov",
                q4_root="/q4",
                layer_contextual_continuation_state_path=configured,
            )

            with self.assertRaisesRegex(Qwen38ChatError, "Integrity|canonical|JSON"):
                second._load_locked()

            self.assertFalse(second.loaded)
            second.close()

    def test_layer_contextual_request_separates_bank_work_from_selected_tail(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            chat = _chat(
                _Runtime(model=_LayerContextualModel()),
                draft_mode="markov",
                q4_root="/q4",
                layer_contextual_continuation_state_path=(
                    Path(temporary) / "layer.json"
                ),
                layer_contextual_continuation_layers=(18, 27),
            )
            chat._load_locked()
            metrics, error = chat._layer_contextual_continuation_metrics()
            assert metrics is not None
            bank = chat._layer_contextual_continuation_bank
            assert bank is not None
            per_layer = {
                "18": {
                    "crystal_accepted_tokens": 3,
                    "crystal_mismatches": 0,
                    "crystal_proposed_tokens": 3,
                    "crystal_verified_tokens": 3,
                },
                "27": {
                    "crystal_accepted_tokens": 1,
                    "crystal_mismatches": 1,
                    "crystal_proposed_tokens": 3,
                    "crystal_verified_tokens": 2,
                },
            }
            chat._last_draft_evidence = {
                "provider": {
                    "layer_context_crystal": {
                        "identity_sha256": bank.identity.identity_sha256,
                        "layers": per_layer,
                        "crystal_accepted_tokens": 4,
                        "crystal_mismatches": 1,
                        "crystal_proposed_tokens": 6,
                        "crystal_verified_tokens": 5,
                        "schema": (
                            "immer.qwen3.8-layer-context-crystal-provider-trace/v1"
                        ),
                        "selected": {
                            "cell_sha256": "4" * 64,
                            "layer": 27,
                            "token_ids": [7, 8, 9],
                            "transaction_sha256": "5" * 64,
                        },
                    }
                }
            }

            request = chat._layer_contextual_continuation_request_delta(
                metrics,
                metrics,
                before_error=error,
                after_error=error,
            )

            self.assertEqual(request["crystal_proposed_tokens"], 6)
            self.assertEqual(request["crystal_verified_tokens"], 5)
            self.assertEqual(request["crystal_accepted_tokens"], 4)
            self.assertEqual(request["crystal_mismatches"], 1)
            self.assertEqual(
                sum(
                    row["crystal_accepted_tokens"]
                    for row in request["layers"].values()
                ),
                4,
            )
            self.assertEqual(request["selected"]["proposed_tokens"], 3)
            self.assertEqual(request["bank_work"]["crystal_queries"], 0)
            self.assertNotIn("crystal_proposed_tokens", request["bank_work"])
            chat.close()

    def test_layer_contextual_addition_migrates_the_previous_window_identity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            chat = _chat(
                _Runtime(model=_LayerContextualModel()),
                draft_mode="markov",
                q4_root="/q4",
                layer_contextual_continuation_state_path=(
                    Path(temporary) / "layer.json"
                ),
            )
            chat._load_locked()
            chat._draft_window_controller = Mock()
            previous = chat._draft_window_runtime_identity(
                layer_contextual_continuation_enabled=False,
            )
            current = chat._draft_window_runtime_identity()

            compatible = chat._draft_window_compatible_previous_identities()

            self.assertIn(previous, compatible)
            self.assertNotIn(current, compatible)
            chat.close()

    def test_hybrid_wave_gate_preserves_v29_window_economics_abi(self) -> None:
        runtime = _Runtime()
        chat = _chat(
            runtime,
            draft_mode="hybrid",
            q4_root="/q4",
        )
        chat._runtime = runtime
        chat._bundle_receipt = _BUNDLE_RECEIPT
        chat._tokenizer_sha256 = _DIGEST
        chat._draft_window_controller = Mock()
        previous = chat._draft_window_runtime_identity(
            hybrid_provider_abi=(
                "immer.qwen3.8-markov-mtp-hybrid-provider/v29"
            ),
        )
        current = chat._draft_window_runtime_identity()

        compatible = chat._draft_window_compatible_previous_identities()

        self.assertEqual(previous, current)
        self.assertNotIn(current, compatible)
        chat.close()

    def test_hybrid_external_drafter_migrates_prior_window_economics(self) -> None:
        runtime = _Runtime()
        chat = _chat(
            runtime,
            draft_bundle_path="/models/Qwen3.5-0.8B",
            draft_mode="hybrid",
            q4_root="/q4",
        )
        chat._runtime = runtime
        chat._bundle_receipt = _BUNDLE_RECEIPT
        chat._tokenizer_sha256 = _DIGEST
        chat._draft_window_controller = Mock()
        previous = chat._draft_window_runtime_identity(
            external_drafter_enabled=False,
        )
        deployed_previous = chat._draft_window_runtime_identity(
            markov_provider_abi=(
                "immer.qwen3.8-markov-draft-provider/v48"
            ),
            hybrid_provider_abi=(
                "immer.qwen3.8-markov-mtp-hybrid-provider/v29"
            ),
            external_drafter_enabled=False,
        )
        current = chat._draft_window_runtime_identity()

        compatible = chat._draft_window_compatible_previous_identities()

        self.assertNotEqual(previous, current)
        self.assertIn(previous, compatible)
        self.assertIn(deployed_previous, compatible)
        self.assertNotIn(current, compatible)
        chat.close()

    def test_draft_window_migrates_from_authenticated_previous_context_bank(
        self,
    ) -> None:
        def contextual_runtime() -> _Runtime:
            runtime = _Runtime()
            runtime.model.config.dim = 8
            runtime.q4_bank = SimpleNamespace(
                identity={"manifest_sha256": "f" * 64},
            )
            runtime.mlp_page_router = None
            runtime.delta_head_receipt = None
            return runtime

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configured = root / "context.json"
            probe = _chat(
                contextual_runtime(),
                draft_mode="markov",
                q4_root="/q4",
                contextual_continuation_state_path=configured,
            )
            with patch(
                "immer.runtimes.qwen3_8.adapter.asdict",
                return_value={"dim": 8, "vocab_size": 300_000},
            ):
                probe._load_locked()
            assert probe._contextual_continuation_bank is not None
            current_template = probe._contextual_continuation_bank.identity
            old_identity = ContextualContinuationIdentity(
                runtime_sha256="1" * 64,
                model_sha256=current_template.model_sha256,
                q4_sha256=current_template.q4_sha256,
                tokenizer_sha256=current_template.tokenizer_sha256,
                hidden_width=current_template.hidden_width,
            )
            probe.close()
            old_path = root / (
                f"context.context-{old_identity.identity_sha256[:16]}.json"
            )
            old_bank = ContextualContinuationBank(old_path, old_identity)
            old_bank.settle(
                captures=(
                    old_bank.make_capture(
                        torch.ones((1, 1, 8), dtype=torch.float32),
                        7,
                        0,
                        (8,),
                    ),
                )
            )
            foreign_identity = ContextualContinuationIdentity(
                runtime_sha256="4" * 64,
                model_sha256="5" * 64,
                q4_sha256=current_template.q4_sha256,
                tokenizer_sha256=current_template.tokenizer_sha256,
                hidden_width=current_template.hidden_width,
            )
            foreign_path = root / (
                f"context.context-{foreign_identity.identity_sha256[:16]}.json"
            )
            foreign_bank = ContextualContinuationBank(
                foreign_path,
                foreign_identity,
            )
            foreign_bank.settle(
                captures=(
                    foreign_bank.make_capture(
                        torch.ones((1, 1, 8), dtype=torch.float32),
                        7,
                        0,
                        (9,),
                    ),
                )
            )
            (root / "context.context-ffffffffffffffff.json").write_bytes(b"broken")
            draft_state = root / "draft-window.bin"
            previous = _chat(
                _Runtime(),
                draft_mode="markov",
                q4_root="/q4",
                contextual_continuation_state_path=configured,
            )
            previous._bundle_receipt = _BUNDLE_RECEIPT
            previous._tokenizer_sha256 = _DIGEST
            previous._contextual_continuation_bank = SimpleNamespace(
                identity=old_identity
            )
            previous_policy = previous._draft_window_runtime_identity()
            previous.close()

            chat = _chat(
                contextual_runtime(),
                draft_mode="markov",
                q4_root="/q4",
                contextual_continuation_state_path=configured,
                draft_window_state_path=draft_state,
            )
            assert chat._draft_window_controller is not None
            chat._draft_window_controller.bind_policy_identity(previous_policy)
            with patch(
                "immer.runtimes.qwen3_8.adapter.asdict",
                return_value={"dim": 8, "vocab_size": 300_000},
            ):
                chat._load_locked()
            current_policy = chat._draft_window_runtime_identity()
            compatible = chat._draft_window_compatible_previous_identities()
            migrated = chat._draft_window_controller.bind_policy_identity(
                current_policy,
                compatible_previous=compatible,
            )

        self.assertNotEqual(previous_policy, current_policy)
        self.assertIn(previous_policy, compatible)
        self.assertIn(
            old_identity.identity_sha256,
            chat._compatible_contextual_continuation_identity_sha256s,
        )
        self.assertNotIn(
            foreign_identity.identity_sha256,
            chat._compatible_contextual_continuation_identity_sha256s,
        )
        self.assertEqual(migrated.policy_identity_sha256, current_policy)
        chat.close()

    def test_attention_output_crystal_mounts_identity_suffixed_and_attaches(
        self,
    ) -> None:
        runtime = _Runtime()
        runtime.model.attach_attention_output_crystal_bank = Mock()
        runtime.model.attention_output_crystal_runtime_math_sha256 = Mock(
            return_value="e" * 64
        )
        identity = AttentionOutputCrystalIdentity("e" * 64)
        opened = SimpleNamespace(
            identity=identity,
            metrics=Mock(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            configured = Path(temporary) / "attention.json"
            chat = _chat(
                runtime,
                q4_root="/q4",
                attention_output_crystal_state_path=configured,
            )
            with patch(
                "immer.runtimes.qwen3_8.adapter.AttentionOutputCrystalBank",
                return_value=opened,
            ) as constructor:
                chat._load_locked()

        state_path, mounted_identity = constructor.call_args.args
        self.assertEqual(mounted_identity, identity)
        self.assertIn(".attention-output-", state_path.name)
        self.assertIn(identity.identity_sha256[:16], state_path.name)
        self.assertEqual(state_path.suffix, ".json")
        runtime.model.attention_output_crystal_runtime_math_sha256.assert_called_once_with()
        runtime.model.attach_attention_output_crystal_bank.assert_called_once_with(
            opened
        )
        self.assertIs(chat._attention_output_crystal_bank, opened)
        chat.close()

    def test_attention_output_crystal_evidence_is_request_local_delta(self) -> None:
        runtime = _Runtime()
        runtime.model.set_attention_output_crystal_enabled = Mock()
        chat = _chat(runtime)
        chat._load_locked()
        identity = AttentionOutputCrystalIdentity("e" * 64)

        def metrics(values):
            return SimpleNamespace(
                to_dict=lambda: {
                    "hit_count": values[0],
                    "identity_sha256": identity.identity_sha256,
                    "logical_projection_bytes_saved": values[2],
                    "skipped_projection_calls_saved": values[1],
                }
            )

        bank = SimpleNamespace(
            identity=identity,
            metrics=Mock(
                side_effect=(
                    metrics((7, 28, 8192)),
                    metrics((9, 36, 12_288)),
                )
            ),
        )
        chat._attention_output_crystal_state_path = Path("/state/attention.json")
        chat._attention_output_crystal_bank = bank

        result = chat.handle(Request("chat", "hello"))

        self.assertTrue(result.ok)
        evidence = result.evidence["attention_output_crystal"]
        self.assertEqual(evidence["schema"], ATTENTION_OUTPUT_CRYSTAL_EVIDENCE_SCHEMA)
        self.assertEqual(evidence["identity"], identity.to_record())
        self.assertEqual(evidence["identity_sha256"], identity.identity_sha256)
        self.assertEqual(
            evidence["request"],
            {
                "hits": 2,
                "logical_projection_bytes_saved": 4096,
                "skipped_projection_calls": 8,
            },
        )
        self.assertNotIn("hello", json.dumps(evidence, sort_keys=True))
        runtime.model.set_attention_output_crystal_enabled.assert_called_once_with(True)
        with patch(
            "immer.runtimes.qwen3_8.adapter._digest",
            side_effect=lambda value: value,
        ):
            policy = chat._result_cell_generation_policy_sha256()
        self.assertEqual(
            policy["attention_output_crystal"]["identity_sha256"],
            identity.identity_sha256,
        )
        chat.close()

    def test_layer_transition_crystal_constructor_rejects_invalid_mounts(self) -> None:
        with self.assertRaisesRegex(ValueError, "require Q4"):
            _chat(
                _Runtime(),
                layer_transition_crystal_state_path="/state/layer-63.json",
            )
        with self.assertRaisesRegex(ValueError, "bfloat16"):
            _chat(
                _Runtime(),
                q4_root="/q4",
                compute_dtype="float16",
                layer_transition_crystal_state_path="/state/layer-63.json",
            )
        for invalid in (True, -0.1, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _chat(
                    _Runtime(),
                    q4_root="/q4",
                    layer_transition_crystal_state_path="/state/layer-63.json",
                    layer_transition_crystal_max_error_radius=invalid,
                )
        with self.assertRaisesRegex(ValueError, "requires.*state_path"):
            _chat(
                _Runtime(),
                layer_transition_crystal_max_error_radius=0.25,
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atlas = root / "atlas"
            compute = root / "compute"
            atlas.mkdir()
            compute.mkdir()
            with self.assertRaisesRegex(ValueError, "requires.*atlas_path"):
                _chat(
                    _Runtime(),
                    q4_root="/q4",
                    layer_transition_crystal_state_path="/state/layer-63.json",
                    layer_transition_crystal_compute_root=compute,
                )
            with self.assertRaisesRegex(ValueError, "requires.*compute_root"):
                _chat(
                    _Runtime(),
                    q4_root="/q4",
                    layer_transition_crystal_state_path="/state/layer-63.json",
                    layer_transition_crystal_atlas_path=atlas,
                )
            with self.assertRaisesRegex(ValueError, "existing real directory"):
                _chat(
                    _Runtime(),
                    q4_root="/q4",
                    layer_transition_crystal_state_path="/state/layer-63.json",
                    layer_transition_crystal_atlas_path=root / "missing-atlas",
                    layer_transition_crystal_compute_root=compute,
                )
            linked_atlas = root / "linked-atlas"
            linked_atlas.symlink_to(atlas, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "existing real directory"):
                _chat(
                    _Runtime(),
                    q4_root="/q4",
                    layer_transition_crystal_state_path="/state/layer-63.json",
                    layer_transition_crystal_atlas_path=linked_atlas,
                    layer_transition_crystal_compute_root=compute,
                )

    def test_layer_transition_crystal_loads_exact_state_and_attaches(self) -> None:
        runtime = _Runtime()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atlas = root / "atlas"
            compute = root / "compute"
            atlas_revision = _semantic_atlas_authority(atlas)
            graph_revision = _compute_graph_authority(compute)
            identity = _layer_transition_identity(
                atlas=atlas_revision,
                graph=graph_revision,
            )
            runtime.model.layer_transition_crystal_model_sha256 = Mock(
                return_value=identity.model_sha256
            )
            runtime.model.layer_transition_crystal_q4_sha256 = Mock(
                return_value=identity.q4_sha256
            )
            runtime.model.attach_layer_transition_crystal_bank = Mock()
            opened = SimpleNamespace(identity=identity, crystals=())
            configured = root / "private-layer-63.bin"
            chat = _chat(
                runtime,
                q4_root="/q4",
                layer_transition_crystal_state_path=configured,
                layer_transition_crystal_atlas_path=atlas,
                layer_transition_crystal_compute_root=compute,
                layer_transition_crystal_max_error_radius=0.25,
            )
            with patch(
                "immer.runtimes.qwen3_8.adapter.LayerTransitionCrystalBank.load",
                return_value=opened,
            ) as loader:
                chat._load_locked()

        loader.assert_called_once_with(configured.absolute())
        runtime.model.layer_transition_crystal_model_sha256.assert_called_once_with()
        runtime.model.layer_transition_crystal_q4_sha256.assert_called_once_with()
        runtime.model.attach_layer_transition_crystal_bank.assert_called_once_with(
            opened,
            max_error_radius=0.25,
            graph_revision_sha256=identity.graph_revision_sha256,
            atlas_revision_sha256=identity.atlas_revision_sha256,
        )
        self.assertIs(chat._layer_transition_crystal_bank, opened)
        self.assertIsNotNone(chat._layer_transition_crystal_atlas)
        self.assertIsNotNone(chat._layer_transition_crystal_compute_graph)
        chat.close()

    def test_layer_transition_crystal_rejects_foreign_authorities_before_attach(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atlas = root / "atlas"
            compute = root / "compute"
            atlas_revision = _semantic_atlas_authority(atlas)
            graph_revision = _compute_graph_authority(compute)
            identities = (
                _layer_transition_identity(
                    atlas="f" * 64,
                    graph=graph_revision,
                ),
                _layer_transition_identity(
                    atlas=atlas_revision,
                    graph="f" * 64,
                ),
            )
            for identity in identities:
                with self.subTest(identity=identity.identity_sha256):
                    runtime = _Runtime()
                    runtime.model.attach_layer_transition_crystal_bank = Mock()
                    chat = _chat(
                        runtime,
                        q4_root="/q4",
                        layer_transition_crystal_state_path=root / "private.bin",
                        layer_transition_crystal_atlas_path=atlas,
                        layer_transition_crystal_compute_root=compute,
                    )
                    with (
                        patch(
                            "immer.runtimes.qwen3_8.adapter."
                            "LayerTransitionCrystalBank.load",
                            return_value=SimpleNamespace(
                                identity=identity, crystals=()
                            ),
                        ),
                        self.assertRaisesRegex(Qwen38ChatError, "foreign"),
                    ):
                        chat._load_locked()
                    runtime.model.attach_layer_transition_crystal_bank.assert_not_called()
                    chat.close()

            identity = _layer_transition_identity(
                atlas=atlas_revision,
                graph=graph_revision,
            )
            runtime = _Runtime()
            runtime.model.attach_layer_transition_crystal_bank = Mock()
            chat = _chat(
                runtime,
                q4_root="/q4",
                layer_transition_crystal_state_path=root / "private.bin",
                layer_transition_crystal_atlas_path=atlas,
                layer_transition_crystal_compute_root=compute,
            )
            forged = SimpleNamespace(
                identity=identity,
                crystals=(
                    SimpleNamespace(
                        source_compute_crystal_sha256="a" * 64,
                        source_compute_edge_sha256="b" * 64,
                    ),
                ),
            )
            with (
                patch(
                    "immer.runtimes.qwen3_8.adapter.LayerTransitionCrystalBank.load",
                    return_value=forged,
                ),
                self.assertRaisesRegex(Qwen38ChatError, "source lineage"),
            ):
                chat._load_locked()
            runtime.model.attach_layer_transition_crystal_bank.assert_not_called()
            chat.close()

    def test_layer_transition_crystal_evidence_is_model_request_delta(self) -> None:
        runtime = _Runtime()
        runtime.model.set_layer_transition_crystal_enabled = Mock()
        identity = _layer_transition_identity()
        radius = 0.25

        def metrics(values):
            return {
                "atlas_revision_sha256": identity.atlas_revision_sha256,
                "attempts": values[0],
                "enabled": True,
                "exact_kv_state_updates": values[4],
                "fallbacks": values[2],
                "graph_revision_sha256": identity.graph_revision_sha256,
                "identity_sha256": identity.identity_sha256,
                "max_error_radius": radius,
                "model_sha256": identity.model_sha256,
                "packed_weight_bytes_avoided": values[6],
                "physical_transitions": values[3],
                "q4_sha256": identity.q4_sha256,
                "replacements": values[1],
                "schema": "immer.qwen3.8-layer-transition-crystal-metrics/v2",
                "skipped_q4_matrix_calls": values[5],
                "transition_rows": values[4],
            }

        runtime.model.layer_transition_crystal_metrics = Mock(
            side_effect=(
                metrics((10, 7, 3, 7, 7, 35, 70_000)),
                metrics((12, 9, 3, 9, 9, 45, 90_000)),
            )
        )
        chat = _chat(runtime)
        chat._load_locked()
        chat._layer_transition_crystal_state_path = Path("/state/layer-63.bin")
        chat._layer_transition_crystal_max_error_radius = radius
        chat._layer_transition_crystal_bank = SimpleNamespace(
            identity=identity,
            crystals=(SimpleNamespace(logical_weight_bytes_replaced=10_000),),
        )

        result = chat.handle(Request("chat", "hello"))

        self.assertTrue(result.ok, result.reason)
        evidence = result.evidence["layer_transition_crystal"]
        self.assertEqual(
            evidence["schema"],
            LAYER_TRANSITION_CRYSTAL_EVIDENCE_SCHEMA,
        )
        self.assertEqual(evidence["identity"], identity.to_record())
        self.assertEqual(evidence["identity_sha256"], identity.identity_sha256)
        self.assertTrue(evidence["request_applied"])
        self.assertFalse(evidence["directive_selected"])
        self.assertEqual(
            evidence["request"],
            {
                "atlas_revision_sha256": identity.atlas_revision_sha256,
                "attempts": 2,
                "exact_kv_state_updates": 2,
                "fallbacks": 0,
                "graph_revision_sha256": identity.graph_revision_sha256,
                "identity_sha256": identity.identity_sha256,
                "max_error_radius": radius,
                "model_sha256": identity.model_sha256,
                "packed_weight_bytes_avoided": 20_000,
                "packed_weight_bytes_per_transition": 10_000,
                "physical_transitions": 2,
                "q4_sha256": identity.q4_sha256,
                "replacements": 2,
                "skipped_q4_matrix_calls": 10,
                "transition_rows": 2,
            },
        )
        self.assertNotIn("hello", json.dumps(evidence, sort_keys=True))
        runtime.model.set_layer_transition_crystal_enabled.assert_called_once_with(True)
        with patch(
            "immer.runtimes.qwen3_8.adapter._digest",
            side_effect=lambda value: value,
        ):
            policy = chat._result_cell_generation_policy_sha256(
                layer_transition_crystal_applied=True
            )
        self.assertEqual(
            policy["layer_transition_crystal"]["identity_sha256"],
            identity.identity_sha256,
        )
        self.assertEqual(
            policy["layer_transition_crystal"]["max_error_radius"],
            radius.hex(),
        )
        self.assertTrue(policy["layer_transition_crystal"]["request_applied"])
        chat.close()

    def test_layer_transition_crystal_discovery_and_explicit_disable(self) -> None:
        runtime = _Runtime()
        runtime.model.set_layer_transition_crystal_enabled = Mock()
        identity = _layer_transition_identity()
        radius = 0.25

        def stable_metrics():
            return {
                "atlas_revision_sha256": identity.atlas_revision_sha256,
                "attempts": 0,
                "enabled": True,
                "exact_kv_state_updates": 0,
                "fallbacks": 0,
                "graph_revision_sha256": identity.graph_revision_sha256,
                "identity_sha256": identity.identity_sha256,
                "max_error_radius": radius,
                "model_sha256": identity.model_sha256,
                "packed_weight_bytes_avoided": 0,
                "physical_transitions": 0,
                "q4_sha256": identity.q4_sha256,
                "replacements": 0,
                "schema": "immer.qwen3.8-layer-transition-crystal-metrics/v2",
                "skipped_q4_matrix_calls": 0,
                "transition_rows": 0,
            }

        runtime.model.layer_transition_crystal_metrics = Mock(
            side_effect=stable_metrics
        )
        chat = _chat(runtime)
        chat._load_locked()
        chat._layer_transition_crystal_state_path = Path("/state/layer-63.bin")
        chat._layer_transition_crystal_max_error_radius = radius
        chat._layer_transition_crystal_bank = SimpleNamespace(
            identity=identity,
            crystals=(SimpleNamespace(logical_weight_bytes_replaced=10_000),),
        )

        def directive(
            actions: tuple[str, ...],
            *,
            disabled_actions: tuple[str, ...] = (),
        ) -> InferenceActionDirective:
            return InferenceActionDirective(
                question_sha256=hashlib.sha256(b"hello").hexdigest(),
                runtime_profile_sha256="2" * 64,
                primary_actions=actions,
                fallback_actions=actions,
                draft_enabled=False,
                source_signature_sha256s=("3" * 64,),
                support=1,
                saved_qwen_forwards=0,
                disabled_actions=disabled_actions,
            )

        baseline = chat.handle(
            Request(
                "chat",
                "hello",
                {
                    QWEN38_INFERENCE_ACTION_METADATA: directive(
                        ("qwen_target",)
                    ).to_document()
                },
            )
        )
        selected = chat.handle(
            Request(
                "chat",
                "hello",
                {
                    QWEN38_INFERENCE_ACTION_METADATA: directive(
                        ("layer_transition_crystal", "qwen_target")
                    ).to_document()
                },
            )
        )
        disabled = chat.handle(
            Request(
                "chat",
                "hello",
                {
                    QWEN38_INFERENCE_ACTION_METADATA: directive(
                        ("qwen_target",),
                        disabled_actions=("layer_transition_crystal",),
                    ).to_document()
                },
            )
        )

        for result in (baseline, selected, disabled):
            self.assertTrue(result.ok, result.reason)
        baseline_applied = baseline.evidence["inference_action_directive"]["applied"]
        selected_applied = selected.evidence["inference_action_directive"]["applied"]
        disabled_applied = disabled.evidence["inference_action_directive"]["applied"]
        self.assertTrue(baseline_applied["layer_transition_crystal"])
        self.assertFalse(
            baseline_applied["layer_transition_crystal_directive_selected"]
        )
        self.assertTrue(selected_applied["layer_transition_crystal"])
        self.assertTrue(selected_applied["layer_transition_crystal_directive_selected"])
        self.assertFalse(disabled_applied["layer_transition_crystal"])
        self.assertTrue(
            disabled_applied["layer_transition_crystal_explicitly_disabled"]
        )
        self.assertEqual(
            runtime.model.set_layer_transition_crystal_enabled.call_args_list,
            [call(True), call(True), call(False)],
        )
        chat.close()

    def test_layer_mlp_crystal_validates_and_attaches_private_authorities(self) -> None:
        with self.assertRaisesRegex(ValueError, "require Q4"):
            _chat(_Runtime(), layer_mlp_crystal_state_path="/state/mlp.json")
        with self.assertRaisesRegex(ValueError, "bfloat16"):
            _chat(
                _Runtime(),
                q4_root="/q4",
                compute_dtype="float16",
                layer_mlp_crystal_state_path="/state/mlp.json",
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atlas = root / "atlas"
            compute = root / "compute"
            atlas_revision = _semantic_atlas_authority(atlas)
            graph_revision = _compute_graph_authority(compute)
            identity = _layer_mlp_identity(
                atlas=atlas_revision,
                graph=graph_revision,
            )
            runtime = _Runtime()
            runtime.model.layer_mlp_crystal_model_sha256 = Mock(
                return_value=identity.model_sha256
            )
            runtime.model.layer_mlp_crystal_q4_sha256 = Mock(
                return_value=identity.q4_sha256
            )
            runtime.model.attach_layer_mlp_crystal_bank = Mock()
            opened = SimpleNamespace(identity=identity, crystals=(object(),))
            configured = root / "private-layer-63-mlp.bin"
            chat = _chat(
                runtime,
                q4_root="/q4",
                layer_mlp_crystal_state_path=configured,
                layer_mlp_crystal_atlas_path=atlas,
                layer_mlp_crystal_compute_root=compute,
                layer_mlp_crystal_max_error_radius=0.25,
            )
            with patch(
                "immer.runtimes.qwen3_8.adapter.Layer63MlpResidualCrystalBank.load",
                return_value=opened,
            ) as loader:
                chat._load_locked()

        loader.assert_called_once_with(configured.absolute())
        runtime.model.attach_layer_mlp_crystal_bank.assert_called_once_with(
            opened,
            max_error_radius=0.25,
            graph_revision_sha256=identity.graph_revision_sha256,
            atlas_revision_sha256=identity.atlas_revision_sha256,
        )
        self.assertIs(chat._layer_mlp_crystal_bank, opened)
        self.assertIsNotNone(chat._layer_mlp_crystal_atlas)
        self.assertIsNotNone(chat._layer_mlp_crystal_compute_graph)
        chat.close()

    def test_multi_layer_mlp_registry_split_mounts_generic_and_legacy_banks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, atlas, compute = _publish_mixed_layer_mlp_registry(root)
            runtime = _Runtime()
            runtime.model.config.dim = 8
            runtime.model.layer_mlp_crystal_model_sha256 = Mock(
                return_value="2" * 64
            )
            runtime.model.layer_mlp_crystal_q4_sha256 = Mock(
                return_value="3" * 64
            )
            runtime.model._layer_mlp_crystal_avoided_q4_bytes = Mock(
                side_effect={18: 8_210, 63: 8_255}.__getitem__
            )
            runtime.model.set_layer_mlp_crystal_registry = Mock()
            runtime.model.attach_layer_mlp_crystal_bank = Mock()
            runtime.model.clear_layer_mlp_crystal_registry = Mock()
            chat = _chat(
                runtime,
                q4_root="/q4",
                layer_mlp_crystal_registry_path=manifest,
                layer_mlp_crystal_atlas_path=atlas,
                layer_mlp_crystal_compute_root=compute,
            )

            registry, legacy, opened_atlas, opened_compute = (
                chat._open_layer_mlp_crystal_mount_registry(runtime)
            )

        self.assertEqual(registry.mounted_layers, (18, 63))
        self.assertIsNotNone(legacy)
        self.assertIsNotNone(opened_atlas)
        self.assertIsNotNone(opened_compute)
        generic_call = runtime.model.set_layer_mlp_crystal_registry.call_args
        self.assertEqual(tuple(generic_call.args[0]), (18,))
        self.assertEqual(generic_call.kwargs["max_error_radii"], {18: 0.25})
        runtime.model.attach_layer_mlp_crystal_bank.assert_called_once()
        legacy_call = runtime.model.attach_layer_mlp_crystal_bank.call_args
        self.assertIs(legacy_call.args[0], legacy)
        self.assertEqual(legacy_call.kwargs["max_error_radius"], 0.25)
        chat.close()

    def test_layer_mlp_crystal_evidence_and_directive_are_request_local(self) -> None:
        runtime = _Runtime()
        runtime.model.set_layer_mlp_crystal_enabled = Mock()
        identity = _layer_mlp_identity()
        radius = 0.25
        values = iter(
            (
                (10, 7, 3, 7, 7, 21, 70_000),
                (12, 9, 3, 9, 9, 27, 90_000),
                (12, 9, 3, 9, 9, 27, 90_000),
                (12, 9, 3, 9, 9, 27, 90_000),
                (12, 9, 3, 9, 9, 27, 90_000),
                (13, 9, 4, 9, 9, 27, 90_000),
            )
        )

        def metrics():
            attempts, replacements, fallbacks, transitions, rows, skipped, avoided = (
                next(values)
            )
            return {
                "atlas_revision_sha256": identity.atlas_revision_sha256,
                "attempts": attempts,
                "enabled": True,
                "exact_kv_state_updates": 0,
                "fallbacks": fallbacks,
                "graph_revision_sha256": identity.graph_revision_sha256,
                "identity_sha256": identity.identity_sha256,
                "max_error_radius": radius,
                "model_sha256": identity.model_sha256,
                "packed_weight_bytes_avoided": avoided,
                "physical_transitions": transitions,
                "q4_sha256": identity.q4_sha256,
                "replacements": replacements,
                "schema": "immer.qwen3.8-layer-mlp-residual-crystal-metrics/v1",
                "skipped_q4_matrix_calls": skipped,
                "transition_rows": rows,
            }

        runtime.model.layer_mlp_crystal_metrics = Mock(side_effect=metrics)
        chat = _chat(runtime)
        chat._load_locked()
        chat._layer_mlp_crystal_state_path = Path("/state/layer-63-mlp.bin")
        chat._layer_mlp_crystal_max_error_radius = radius
        chat._layer_mlp_crystal_bank = SimpleNamespace(
            identity=identity,
            crystals=(SimpleNamespace(logical_weight_bytes_replaced=10_000),),
        )

        first = chat.handle(Request("chat", "hello"))
        directive = InferenceActionDirective(
            question_sha256=hashlib.sha256(b"hello").hexdigest(),
            runtime_profile_sha256="2" * 64,
            primary_actions=("qwen_target",),
            fallback_actions=("qwen_target",),
            draft_enabled=False,
            source_signature_sha256s=("3" * 64,),
            support=1,
            saved_qwen_forwards=0,
            disabled_actions=("layer_mlp_crystal",),
        )
        disabled = chat.handle(
            Request(
                "chat",
                "hello",
                {QWEN38_INFERENCE_ACTION_METADATA: directive.to_document()},
            )
        )
        fallback = chat.handle(Request("chat", "hello"))

        self.assertTrue(first.ok, first.reason)
        self.assertTrue(disabled.ok, disabled.reason)
        self.assertTrue(fallback.ok, fallback.reason)
        evidence = first.evidence["layer_mlp_crystal"]
        self.assertEqual(evidence["schema"], LAYER_MLP_CRYSTAL_EVIDENCE_SCHEMA)
        self.assertEqual(evidence["action_abi"], LAYER_MLP_RESIDUAL_ACTION_ABI)
        self.assertTrue(evidence["configured"])
        self.assertTrue(evidence["request_applied"])
        self.assertEqual(
            evidence["request"],
            {
                "action_abi": LAYER_MLP_RESIDUAL_ACTION_ABI,
                "atlas_revision_sha256": identity.atlas_revision_sha256,
                "attempts": 2,
                "fallbacks": 0,
                "graph_revision_sha256": identity.graph_revision_sha256,
                "identity_sha256": identity.identity_sha256,
                "max_error_radius": radius,
                "model_sha256": identity.model_sha256,
                "packed_weight_bytes_avoided": 20_000,
                "packed_weight_bytes_per_transition": 10_000,
                "physical_transitions": 2,
                "q4_sha256": identity.q4_sha256,
                "replacements": 2,
                "skipped_q4_matrix_calls": 6,
                "transition_rows": 2,
            },
        )
        self.assertFalse(
            disabled.evidence["inference_action_directive"]["applied"][
                "layer_mlp_crystal"
            ]
        )
        self.assertTrue(
            disabled.evidence["inference_action_directive"]["applied"][
                "layer_mlp_crystal_explicitly_disabled"
            ]
        )
        self.assertTrue(fallback.evidence["layer_mlp_crystal"]["configured"])
        self.assertFalse(fallback.evidence["layer_mlp_crystal"]["request_applied"])
        self.assertEqual(
            fallback.evidence["layer_mlp_crystal"]["request"]["fallbacks"],
            1,
        )
        self.assertEqual(
            runtime.model.set_layer_mlp_crystal_enabled.call_args_list,
            [call(True), call(False), call(True)],
        )
        with patch(
            "immer.runtimes.qwen3_8.adapter._digest",
            side_effect=lambda value: value,
        ):
            policy = chat._result_cell_generation_policy_sha256(
                layer_mlp_crystal_applied=False
            )
        self.assertEqual(
            policy["layer_mlp_crystal"]["identity_sha256"],
            identity.identity_sha256,
        )
        self.assertFalse(policy["layer_mlp_crystal"]["request_applied"])
        self.assertNotIn("/state/", json.dumps(policy, sort_keys=True))
        chat.close()

    def test_multi_layer_mlp_crystal_runs_deepest_only_and_restores_state(
        self,
    ) -> None:
        registry = _fake_layer_mlp_mount_registry(18, 63)
        model = _LayerMlpCrystalRegistryModel(registry)
        runtime = _Runtime(model=model)
        chat = _chat(runtime)
        chat._load_locked()
        chat._layer_mlp_crystal_mount_registry = registry
        chat._layer_mlp_crystal_bank = SimpleNamespace()

        result = chat.handle(Request("chat", "hello"))

        self.assertTrue(result.ok, result.reason)
        evidence = result.evidence["layer_mlp_crystal"]
        self.assertEqual(
            evidence["schema"],
            "immer.qwen3.8-layer-mlp-residual-crystal-evidence/v2",
        )
        self.assertEqual(evidence["enabled_layer"], 63)
        self.assertEqual(evidence["request"]["enabled_layer"], 63)
        self.assertEqual(evidence["request"]["executed_layer"], 63)
        self.assertEqual(evidence["request"]["physical_transitions"], 1)
        self.assertEqual(evidence["request"]["replacements"], 1)
        self.assertEqual(evidence["request"]["skipped_q4_matrix_calls"], 3)
        self.assertEqual(
            evidence["request"]["packed_weight_bytes_avoided"],
            next(
                entry.packed_weight_bytes_avoided
                for entry in registry.entries
                if entry.layer_index == 63
            ),
        )
        self.assertEqual(
            model.layer_mlp_crystal_registry_enabled_layers,
            frozenset({18}),
        )
        self.assertFalse(model.layer_mlp_crystal_enabled)
        self.assertEqual(model.counters[18]["attempts"], 0)
        self.assertEqual(model.counters[63]["attempts"], 1)
        self.assertIn((), model.registry_enable_calls)
        self.assertIn((18,), model.registry_enable_calls)
        chat.close()

    def test_multi_layer_mlp_crystal_radius_zero_fallback_is_not_execution(
        self,
    ) -> None:
        registry = _fake_layer_mlp_mount_registry(18)
        model = _LayerMlpCrystalRegistryModel(registry, fallback_only=True)
        runtime = _Runtime(model=model)
        chat = _chat(runtime)
        chat._load_locked()
        chat._layer_mlp_crystal_mount_registry = registry

        result = chat.handle(Request("chat", "hello"))

        self.assertTrue(result.ok, result.reason)
        evidence = result.evidence["layer_mlp_crystal"]
        self.assertFalse(evidence["request_applied"])
        self.assertEqual(evidence["request"]["attempts"], 1)
        self.assertEqual(evidence["request"]["fallbacks"], 1)
        self.assertEqual(evidence["request"]["physical_transitions"], 0)
        self.assertIsNone(evidence["request"]["executed_layer"])
        self.assertEqual(
            model.layer_mlp_crystal_registry_enabled_layers,
            frozenset({18}),
        )
        chat.close()

    def test_multi_layer_mlp_crystal_rejects_manifest_drift_before_execution(
        self,
    ) -> None:
        registry = _fake_layer_mlp_mount_registry(18)
        changed = SimpleNamespace(
            **{
                **vars(registry),
                "manifest_file_sha256": "8" * 64,
                "registry_sha256": "9" * 64,
            }
        )
        model = _LayerMlpCrystalRegistryModel(registry)
        runtime = _Runtime(model=model)
        chat = _chat(runtime)
        chat._load_locked()
        chat._layer_mlp_crystal_registry_path = Path("/registry.json")
        chat._layer_mlp_crystal_mount_registry = registry

        with patch(
            "immer.runtimes.qwen3_8.adapter."
            "read_layer_mlp_o1_registry_manifest",
            return_value=changed,
        ):
            result = chat.handle(Request("chat", "hello"))

        self.assertFalse(result.ok)
        self.assertIn("registry changed", result.reason)
        self.assertEqual(model.calls, [])
        chat.close()

    def test_layer_mlp_crystal_rejects_empty_bank_before_attachment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atlas = root / "atlas"
            compute = root / "compute"
            identity = _layer_mlp_identity(
                atlas=_semantic_atlas_authority(atlas),
                graph=_compute_graph_authority(compute),
            )
            runtime = _Runtime()
            runtime.model.attach_layer_mlp_crystal_bank = Mock()
            chat = _chat(
                runtime,
                q4_root="/q4",
                layer_mlp_crystal_state_path=root / "empty.bin",
                layer_mlp_crystal_atlas_path=atlas,
                layer_mlp_crystal_compute_root=compute,
            )
            with patch(
                "immer.runtimes.qwen3_8.adapter.Layer63MlpResidualCrystalBank.load",
                return_value=SimpleNamespace(identity=identity, crystals=()),
            ):
                with self.assertRaisesRegex(Qwen38ChatError, "contains no actions"):
                    chat._load_locked()

        runtime.model.attach_layer_mlp_crystal_bank.assert_not_called()
        chat.close()

    def test_layer_transition_result_policy_is_path_neutral_and_budget_bound(
        self,
    ) -> None:
        identity = _layer_transition_identity()

        def policy(
            path: str,
            atlas: Path,
            compute: Path,
            radius: float,
            *,
            applied: bool,
        ):
            chat = _chat(
                _Runtime(),
                q4_root="/q4",
                result_cell_code_revision="a" * 64,
                layer_transition_crystal_state_path=path,
                layer_transition_crystal_atlas_path=atlas,
                layer_transition_crystal_compute_root=compute,
                layer_transition_crystal_max_error_radius=radius,
            )
            chat._layer_transition_crystal_bank = SimpleNamespace(
                identity=identity,
                crystals=(),
            )
            with patch(
                "immer.runtimes.qwen3_8.adapter._digest",
                side_effect=lambda value: value,
            ):
                result = chat._result_cell_generation_policy_sha256(
                    layer_transition_crystal_applied=applied
                )
            self.assertIsNone(
                chat._result_cell_semantic_replay_receipt(
                    question="hello",
                    rendered_prompt="prompt",
                    prompt_ids=(1,),
                )
            )
            return result

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atlas_a = root / "atlas-a"
            atlas_b = root / "atlas-b"
            compute_a = root / "compute-a"
            compute_b = root / "compute-b"
            for directory in (atlas_a, atlas_b, compute_a, compute_b):
                directory.mkdir()
            first = policy(
                "/private/a.bin",
                atlas_a,
                compute_a,
                0.25,
                applied=True,
            )
            other_path = policy(
                "/different/b.bin",
                atlas_b,
                compute_b,
                0.25,
                applied=True,
            )
            other_budget = policy(
                "/private/a.bin",
                atlas_a,
                compute_a,
                0.5,
                applied=True,
            )
            disabled = policy(
                "/private/a.bin",
                atlas_a,
                compute_a,
                0.25,
                applied=False,
            )

        self.assertEqual(first, other_path)
        self.assertNotEqual(first, other_budget)
        self.assertNotEqual(first, disabled)
        self.assertNotIn("/private", json.dumps(first, sort_keys=True))
        self.assertNotIn("/different", json.dumps(first, sort_keys=True))
        self.assertNotIn(str(root), json.dumps(first, sort_keys=True))

    def test_mlp_page_coordinate_mounts_model_verified_identity(self) -> None:
        runtime = _Runtime()
        q4_identity = {
            "bank_codec_abi": 3,
            "manifest_sha256": "f" * 64,
            "native_abi": 7,
            "schema": "q4-fixture/v1",
        }
        page_identity = {
            "policy": "page-fixture/v1",
            "route_width": 8,
            "schema": "page-fixture/v1",
        }
        runtime.q4_bank = SimpleNamespace(identity=q4_identity)
        runtime.delta_head_router = SimpleNamespace()
        runtime.mlp_page_router = SimpleNamespace(
            snapshot_identity=Mock(return_value=page_identity)
        )
        runtime.model.set_delta_head_router = Mock()
        runtime.model.mlp_page_coordinate_runtime_math_sha256 = Mock(
            return_value="e" * 64
        )
        runtime.model.attach_mlp_page_coordinate_bank = Mock()

        def digest(value):
            return hashlib.sha256(
                json.dumps(
                    value,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("ascii")
            ).hexdigest()

        identity = MlpPageCoordinateIdentity(
            runtime_math_sha256="e" * 64,
            q4_identity_sha256=digest(q4_identity),
            page_router_identity_sha256=digest(page_identity),
        )
        opened = SimpleNamespace(identity=identity, metrics=Mock())
        with tempfile.TemporaryDirectory() as temporary:
            configured = Path(temporary) / "mlp-coordinate.json"
            chat = _chat(
                runtime,
                q4_root="/q4",
                mlp_page_state_path="/state/pages.json",
                mlp_page_coordinate_state_path=configured,
            )
            with patch(
                "immer.runtimes.qwen3_8.adapter.MlpPageCoordinateBank",
                return_value=opened,
            ) as constructor:
                chat._load_locked()

        state_path, mounted_identity = constructor.call_args.args
        self.assertEqual(mounted_identity, identity)
        self.assertIn(".mlp-page-coordinate-", state_path.name)
        self.assertIn(identity.identity_sha256[:16], state_path.name)
        self.assertEqual(state_path.suffix, ".json")
        runtime.model.mlp_page_coordinate_runtime_math_sha256.assert_called_once_with()
        runtime.model.set_delta_head_router.assert_called_once_with(None)
        runtime.mlp_page_router.snapshot_identity.assert_called_once_with()
        runtime.model.attach_mlp_page_coordinate_bank.assert_called_once_with(opened)
        self.assertIs(chat._mlp_page_coordinate_bank, opened)
        chat.close()

    def test_mlp_page_coordinate_evidence_is_request_local_delta(self) -> None:
        runtime = _Runtime()
        runtime.model.set_mlp_page_coordinate_enabled = Mock()
        chat = _chat(runtime)
        chat._load_locked()
        identity = MlpPageCoordinateIdentity(
            runtime_math_sha256="e" * 64,
            q4_identity_sha256="f" * 64,
            page_router_identity_sha256="d" * 64,
        )

        def metrics(values):
            return SimpleNamespace(
                to_dict=lambda: {
                    "hit_count": values[0],
                    "identity_sha256": identity.identity_sha256,
                    "logical_page_weight_bytes_saved": values[2],
                    "physical_pages_saved": values[1],
                }
            )

        bank = SimpleNamespace(
            identity=identity,
            metrics=Mock(
                side_effect=(
                    metrics((11, 20, 40_960)),
                    metrics((13, 28, 57_344)),
                )
            ),
        )
        chat._mlp_page_coordinate_state_path = Path("/state/mlp-coordinate.json")
        chat._mlp_page_coordinate_bank = bank

        result = chat.handle(Request("chat", "hello"))

        self.assertTrue(result.ok, result.reason)
        evidence = result.evidence["mlp_page_coordinate"]
        self.assertEqual(evidence["schema"], MLP_PAGE_COORDINATE_EVIDENCE_SCHEMA)
        self.assertEqual(evidence["identity"], identity.to_record())
        self.assertEqual(evidence["identity_sha256"], identity.identity_sha256)
        self.assertEqual(
            evidence["request"],
            {
                "hits": 2,
                "logical_page_weight_bytes_saved": 16_384,
                "physical_pages_saved": 8,
            },
        )
        self.assertNotIn("hello", json.dumps(evidence, sort_keys=True))
        runtime.model.set_mlp_page_coordinate_enabled.assert_called_once_with(True)
        with patch(
            "immer.runtimes.qwen3_8.adapter._digest",
            side_effect=lambda value: value,
        ):
            policy = chat._result_cell_generation_policy_sha256()
        self.assertEqual(
            policy["mlp_page_coordinate"]["identity_sha256"],
            identity.identity_sha256,
        )
        chat.close()

    def test_mlp_page_coordinate_stays_enabled_for_action_discovery(self) -> None:
        runtime = _Runtime()
        runtime.model.set_mlp_page_coordinate_enabled = Mock()
        chat = _chat(runtime)
        chat._load_locked()
        identity = MlpPageCoordinateIdentity(
            runtime_math_sha256="e" * 64,
            q4_identity_sha256="f" * 64,
            page_router_identity_sha256="d" * 64,
        )
        stable_metrics = SimpleNamespace(
            to_dict=lambda: {
                "hit_count": 0,
                "identity_sha256": identity.identity_sha256,
                "logical_page_weight_bytes_saved": 0,
                "physical_pages_saved": 0,
            }
        )
        chat._mlp_page_coordinate_state_path = Path("/state/mlp-coordinate.json")
        chat._mlp_page_coordinate_bank = SimpleNamespace(
            identity=identity,
            metrics=Mock(return_value=stable_metrics),
        )

        def directive(actions):
            return InferenceActionDirective(
                question_sha256=hashlib.sha256(b"hello").hexdigest(),
                runtime_profile_sha256="2" * 64,
                primary_actions=actions,
                fallback_actions=actions,
                draft_enabled=False,
                source_signature_sha256s=("3" * 64,),
                support=1,
                saved_qwen_forwards=0,
            )

        baseline = chat.handle(
            Request(
                "chat",
                "hello",
                {
                    QWEN38_INFERENCE_ACTION_METADATA: directive(
                        ("qwen_target",)
                    ).to_document()
                },
            )
        )
        selected = chat.handle(
            Request(
                "chat",
                "hello",
                {
                    QWEN38_INFERENCE_ACTION_METADATA: directive(
                        ("mlp_page_coordinate", "qwen_target")
                    ).to_document()
                },
            )
        )

        self.assertTrue(baseline.ok, baseline.reason)
        self.assertTrue(selected.ok, selected.reason)
        baseline_applied = baseline.evidence["inference_action_directive"]["applied"]
        selected_applied = selected.evidence["inference_action_directive"]["applied"]
        self.assertTrue(baseline_applied["mlp_page_coordinate"])
        self.assertFalse(baseline_applied["mlp_page_coordinate_directive_selected"])
        self.assertTrue(selected_applied["mlp_page_coordinate"])
        self.assertTrue(selected_applied["mlp_page_coordinate_directive_selected"])
        self.assertEqual(
            runtime.model.set_mlp_page_coordinate_enabled.call_args_list,
            [call(True), call(True)],
        )
        chat.close()

    def test_delta_action_disables_coordinate_before_router_swap(self) -> None:
        runtime = _Runtime()
        events: list[tuple[str, object]] = []
        coordinate_enabled = True

        def set_coordinate(enabled):
            nonlocal coordinate_enabled
            coordinate_enabled = enabled
            events.append(("coordinate", enabled))

        def set_delta(router):
            if router is not None and coordinate_enabled:
                raise RuntimeError("Delta attached before coordinate opt-out")
            runtime.model.delta_head_router = router
            events.append(("delta", router))

        runtime.model.set_mlp_page_coordinate_enabled = Mock(side_effect=set_coordinate)
        runtime.model.set_delta_head_router = Mock(side_effect=set_delta)
        router = SimpleNamespace(
            metrics=Mock(
                side_effect=(
                    {"calls": 0, "logical_bytes_saved": 0, "rows": 0},
                    {"calls": 1, "logical_bytes_saved": 10, "rows": 2},
                )
            )
        )
        runtime.delta_head_router = router
        runtime.delta_head_receipt = {
            "layers": [0],
            "schema": "fixture.delta-head/v2",
        }
        chat = _chat(
            runtime,
            q4_root="/q4",
            delta_head_state_path="/state/delta.json",
        )
        chat._load_locked()
        identity = MlpPageCoordinateIdentity(
            runtime_math_sha256="e" * 64,
            q4_identity_sha256="f" * 64,
            page_router_identity_sha256="d" * 64,
        )
        stable_metrics = SimpleNamespace(
            to_dict=lambda: {
                "hit_count": 0,
                "identity_sha256": identity.identity_sha256,
                "logical_page_weight_bytes_saved": 0,
                "physical_pages_saved": 0,
            }
        )
        chat._mlp_page_coordinate_state_path = Path("/state/mlp-coordinate.json")
        chat._mlp_page_coordinate_bank = SimpleNamespace(
            identity=identity,
            metrics=Mock(return_value=stable_metrics),
        )
        directive = InferenceActionDirective(
            question_sha256=hashlib.sha256(b"hello").hexdigest(),
            runtime_profile_sha256="2" * 64,
            primary_actions=("mlp_head_coordinate", "qwen_target"),
            fallback_actions=("qwen_target",),
            draft_enabled=False,
            source_signature_sha256s=("3" * 64,),
            support=1,
            saved_qwen_forwards=0,
        )

        result = chat.handle(
            Request(
                "chat",
                "hello",
                {QWEN38_INFERENCE_ACTION_METADATA: directive.to_document()},
            )
        )

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(
            events,
            [
                ("delta", None),
                ("coordinate", False),
                ("delta", router),
                ("delta", None),
                ("coordinate", True),
            ],
        )
        self.assertFalse(
            result.evidence["inference_action_directive"]["applied"][
                "mlp_page_coordinate"
            ]
        )
        chat.close()

    def test_action_directive_keeps_configured_attention_crystal_discovery(
        self,
    ) -> None:
        runtime = _Runtime()
        runtime.model.set_attention_output_crystal_enabled = Mock()
        chat = _chat(runtime)
        chat._load_locked()
        identity = AttentionOutputCrystalIdentity("e" * 64)
        stable_metrics = SimpleNamespace(
            to_dict=lambda: {
                "hit_count": 0,
                "identity_sha256": identity.identity_sha256,
                "logical_projection_bytes_saved": 0,
                "skipped_projection_calls_saved": 0,
            }
        )
        chat._attention_output_crystal_state_path = Path("/state/attention.json")
        chat._attention_output_crystal_bank = SimpleNamespace(
            identity=identity,
            metrics=Mock(return_value=stable_metrics),
        )

        def directive(actions):
            return InferenceActionDirective(
                question_sha256=hashlib.sha256(b"hello").hexdigest(),
                runtime_profile_sha256="2" * 64,
                primary_actions=actions,
                fallback_actions=actions,
                draft_enabled=False,
                source_signature_sha256s=("3" * 64,),
                support=1,
                saved_qwen_forwards=0,
            )

        disabled = chat.handle(
            Request(
                "chat",
                "hello",
                {
                    QWEN38_INFERENCE_ACTION_METADATA: directive(
                        ("qwen_target",)
                    ).to_document()
                },
            )
        )
        enabled = chat.handle(
            Request(
                "chat",
                "hello",
                {
                    QWEN38_INFERENCE_ACTION_METADATA: directive(
                        ("attention_output_crystal", "qwen_target")
                    ).to_document()
                },
            )
        )

        self.assertTrue(disabled.ok, disabled.reason)
        self.assertTrue(enabled.ok, enabled.reason)
        self.assertTrue(
            disabled.evidence["inference_action_directive"]["applied"][
                "attention_output_crystal"
            ]
        )
        self.assertTrue(
            enabled.evidence["inference_action_directive"]["applied"][
                "attention_output_crystal"
            ]
        )
        self.assertFalse(
            disabled.evidence["inference_action_directive"]["applied"][
                "attention_output_crystal_directive_selected"
            ]
        )
        self.assertTrue(
            enabled.evidence["inference_action_directive"]["applied"][
                "attention_output_crystal_directive_selected"
            ]
        )
        self.assertEqual(
            runtime.model.set_attention_output_crystal_enabled.call_args_list,
            [call(True), call(True)],
        )
        chat.close()

    def test_draft_window_identity_binds_page_route_width_and_joint_policy(
        self,
    ) -> None:
        def configured(
            *,
            page_state: str | None = None,
            route_width: int = 192,
        ) -> Qwen38CausalChat:
            chat = _chat(
                _Runtime(),
                draft_mode="markov",
                q4_root=None if page_state is None else "/models/q4",
                mlp_page_state_path=page_state,
                mlp_page_route_width=route_width,
            )
            chat._bundle_receipt = _BUNDLE_RECEIPT
            chat._tokenizer_sha256 = _DIGEST
            return chat

        disabled = configured()
        width160 = configured(page_state="/state/pages-160.json", route_width=160)
        width192 = configured(page_state="/state/pages-192.json", route_width=192)
        disabled_identity = disabled._draft_window_runtime_identity()
        width160_identity = width160._draft_window_runtime_identity()
        width192_identity = width192._draft_window_runtime_identity()
        width192_v6_identity = width192._draft_window_runtime_identity(
            mlp_page_schema="immer.qwen3.8-mlp-page-markov/v6",
            mlp_page_policy=(
                "dynamic-page-transitions+coactivation+adaptive-width+"
                "terminal-reward+fixed-share/v6"
            ),
        )
        width192_v7_identity = width192._draft_window_runtime_identity(
            mlp_page_schema="immer.qwen3.8-mlp-page-markov/v7",
            mlp_page_policy=(
                "dynamic-page-transitions+coactivation+adaptive-width+"
                "terminal-route-advantage+fixed-share/v7"
            ),
        )
        width192_v8_identity = width192._draft_window_runtime_identity(
            mlp_page_schema="immer.qwen3.8-mlp-page-markov/v8",
            mlp_page_policy=(
                "dynamic-page-transitions+coactivation+adaptive-width+"
                "terminal-route-advantage+causal-lookahead-prefetch+"
                "fixed-share/v8"
            ),
        )

        self.assertNotEqual(disabled_identity, width192_identity)
        self.assertNotEqual(width160_identity, width192_identity)
        self.assertNotEqual(width192_v6_identity, width192_identity)
        self.assertNotEqual(width192_v7_identity, width192_identity)
        self.assertNotEqual(width192_v8_identity, width192_identity)
        with patch(
            "immer.runtimes.qwen3_8.adapter.DRAFT_WINDOW_FEEDBACK_SCHEMA",
            "immer.qwen3.8-draft-window-feedback/v999",
        ):
            changed_joint_policy = width192._draft_window_runtime_identity()
        self.assertNotEqual(changed_joint_policy, width192_identity)

        with patch(
            "immer.runtimes.qwen3_8.adapter._digest",
            side_effect=lambda value: value,
        ):
            record = width192._draft_window_runtime_identity()
        joint = record["provider"]["joint_runtime_reward"]
        self.assertEqual(
            joint,
            {
                "draft_feedback_schema": ("immer.qwen3.8-draft-window-feedback/v3"),
                "mlp_page_enabled": True,
                "mlp_page_policy": (
                    "dynamic-page-transitions+coactivation+adaptive-width+"
                    "terminal-route-advantage+consensus-budget-lookahead+"
                    "fixed-share/v9"
                ),
                "mlp_page_schema": "immer.qwen3.8-mlp-page-markov/v9",
                "o1_enabled": False,
                "policy": "o1+draft+page-savings-target-work/v1",
                "route_width": 192,
                "schema": "immer.qwen3.8-joint-runtime-reward/v1",
            },
        )
        disabled.close()
        width160.close()
        width192.close()

    def test_template_anchor_stops_before_user_specific_tokens(self) -> None:
        class Tokenizer:
            @staticmethod
            def encode(text):
                marker = 41 if "\nA<|im_end|>" in text else 57
                return (11, 12, 13, marker, 99)

        chat = _chat(_Runtime())
        prefix = chat._template_anchor_prefix(
            SimpleNamespace(tokenizer=Tokenizer()),
            (11, 12, 13, 77, 88),
        )

        self.assertEqual(prefix, (11, 12, 13))
        chat.close()

    def test_runtime_mounts_and_closes_range_markov_observer(self) -> None:
        from immer.runtimes.deepseek_v4.causal_weights import LogicalModelIdentity
        from immer.runtimes.qwen3_8.adapter import _open_local_runtime

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenizer_path = root / "tokenizer.json"
            tokenizer_path.write_text("{}", encoding="utf-8")
            lifecycle = []
            source = SimpleNamespace(
                prefetch_range=Mock(return_value=True),
                set_access_observer=Mock(
                    side_effect=lambda *_args, **_kwargs: lifecycle.append("attach")
                ),
                repo_id="Qwen/test",
                revision="a" * 40,
                metrics=Mock(return_value={"inventory_source_fingerprint": "b" * 64}),
            )
            mount = SimpleNamespace(
                source=source,
                tensor_reader=object(),
                weights_root=root,
                close=Mock(),
            )
            pager = SimpleNamespace(
                attach_exact_head_index=Mock(),
                close=Mock(),
            )
            model = SimpleNamespace(
                checkpoint_preflight=Mock(
                    side_effect=lambda: lifecycle.append("preflight") or {"ok": True}
                ),
                reset_state=Mock(),
                mlp_sparse_executor=None,
            )
            tokenizer = SimpleNamespace()
            prefetcher = SimpleNamespace(
                bind_source_identity=Mock(),
                close=Mock(),
                metrics=lambda: {},
            )
            with (
                patch(
                    "immer.runtimes.qwen3_8.adapter.CausalWeightMount",
                    return_value=mount,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.verify_qwen38_causal_mount",
                    return_value=_BUNDLE_RECEIPT,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38Config.from_file",
                    return_value=SimpleNamespace(),
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38WeightPager",
                    return_value=pager,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.StreamedQwen38",
                    return_value=model,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38Tokenizer",
                    return_value=tokenizer,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter._file_sha256",
                    return_value=_DIGEST,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.MarkovRangePrefetcher",
                    return_value=prefetcher,
                ) as constructor,
            ):
                runtime = _open_local_runtime(
                    bundle_path=root,
                    tokenizer_path=tokenizer_path,
                    identity=LogicalModelIdentity("Qwen/test", "a" * 40),
                    require_official_config=False,
                    device="cpu",
                    compute_dtype="bfloat16",
                    source_budget_mb=1,
                    max_resident_bytes=1024,
                    max_context_tokens=16,
                    range_markov_state_path=root / "ranges.bin",
                    range_prefetch_max_bytes=123,
                    range_prefetch_min_support=3,
                    range_prefetch_min_confidence=0.8,
                )

            constructor.assert_called_once_with(
                root / "ranges.bin",
                prefetch_range=source.prefetch_range,
                min_support=3,
                min_confidence=0.8,
                max_prefetch_bytes=123,
                beam_horizon=3,
                beam_width=4,
                hint_cooldown_operations=2,
            )
            source.set_access_observer.assert_called_once_with(
                prefetcher,
                prepare_identity=False,
            )
            prefetcher.bind_source_identity.assert_called_once_with(
                "Qwen/test",
                "a" * 40,
                "b" * 64,
            )
            self.assertEqual(lifecycle[:2], ["preflight", "attach"])
            runtime.close()
            prefetcher.close.assert_called_once_with()
            self.assertEqual(
                source.set_access_observer.call_args_list[-1].args,
                (None,),
            )
            mount.close.assert_called_once_with()

    def test_runtime_wires_q4_page_lookahead_into_the_live_router(self) -> None:
        from immer.runtimes.deepseek_v4.causal_weights import LogicalModelIdentity
        from immer.runtimes.qwen3_8.adapter import _open_local_runtime

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenizer_path = root / "tokenizer.json"
            tokenizer_path.write_text("{}", encoding="utf-8")
            source = SimpleNamespace(
                repo_id="Qwen/test",
                revision="a" * 40,
                metrics=Mock(return_value={"inventory_source_fingerprint": "b" * 64}),
            )
            mount = SimpleNamespace(
                source=source,
                tensor_reader=object(),
                weights_root=root,
                close=Mock(),
            )
            q4 = SimpleNamespace(
                identity={"manifest_sha256": "c" * 64},
                prefetch_mlp_pages=Mock(return_value=True),
                metrics=Mock(return_value={}),
                close=Mock(),
            )
            pager = SimpleNamespace(
                attach_exact_head_index=Mock(),
                close=Mock(),
            )
            model = SimpleNamespace(
                checkpoint_preflight=Mock(return_value={"ok": True}),
                reset_state=Mock(),
                mlp_sparse_executor=None,
                mlp_page_router=None,
                delta_head_router=None,
            )
            router = SimpleNamespace(close=Mock())
            delta_router = SimpleNamespace(
                close=Mock(),
                snapshot_identity=Mock(
                    return_value={
                        "layers": [0],
                        "schema": "fixture.delta-head/v1",
                    }
                ),
            )
            with (
                patch(
                    "immer.runtimes.qwen3_8.adapter.CausalWeightMount",
                    return_value=mount,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.verify_qwen38_causal_mount",
                    return_value=_BUNDLE_RECEIPT,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38Config.from_file",
                    return_value=SimpleNamespace(
                        n_layers=2,
                        intermediate_size=128,
                        linear_num_value_heads=48,
                        linear_value_head_dim=128,
                        is_full_attention=lambda layer: layer == 1,
                    ),
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Q4Bank.load",
                    return_value=q4,
                ) as q4_loader,
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38WeightPager",
                    return_value=pager,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.StreamedQwen38",
                    return_value=model,
                ) as model_constructor,
                patch(
                    "immer.runtimes.qwen3_8.adapter.MlpPageMarkov",
                    return_value=router,
                ) as page_constructor,
                patch(
                    "immer.runtimes.qwen3_8.adapter.PackedDeltaHeadRouter",
                    return_value=delta_router,
                ) as delta_constructor,
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38Tokenizer",
                    return_value=SimpleNamespace(),
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter._file_sha256",
                    return_value=_DIGEST,
                ),
            ):
                runtime = _open_local_runtime(
                    bundle_path=root,
                    tokenizer_path=tokenizer_path,
                    identity=LogicalModelIdentity("Qwen/test", "a" * 40),
                    require_official_config=False,
                    device="cpu",
                    compute_dtype="bfloat16",
                    source_budget_mb=1,
                    max_resident_bytes=1024,
                    max_context_tokens=16,
                    q4_root=root / "q4",
                    mlp_page_state_path=root / "pages.json",
                    mlp_page_route_width=2,
                    delta_head_state_path=root / "delta.json",
                    delta_head_active_layers=(0,),
                )

            options = page_constructor.call_args.kwargs
            self.assertEqual(q4_loader.call_args.kwargs["max_prefetch_bytes"], 512)
            self.assertIs(options["lookahead_prefetch"], q4.prefetch_mlp_pages)
            self.assertEqual(options["n_layers"], 2)
            self.assertEqual(options["page_count"], 2)
            self.assertEqual(options["route_width"], 2)
            delta_options = delta_constructor.call_args.kwargs
            self.assertEqual(delta_options["active_layers"], (0,))
            self.assertEqual(delta_options["state_path"], root / "delta.json")
            self.assertIsNone(model_constructor.call_args.kwargs["delta_head_router"])
            self.assertIs(runtime.delta_head_router, delta_router)
            self.assertIsNone(model.delta_head_router)
            runtime.close()
            mount.close.assert_called_once_with()

    def test_range_markov_metrics_are_exposed_without_changing_generation(self) -> None:
        runtime = _Runtime()
        runtime.range_prefetcher = SimpleNamespace(
            metrics=lambda: {
                "schema": "immer.range-markov-metrics/v1",
                "operations": 12,
                "predictions": 7,
                "prefetch_hints": 5,
            }
        )
        chat = _chat(runtime, range_markov_state_path="/state/ranges.bin")

        result = chat.handle(Request("chat", "hello"))

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.evidence["range_markov"]["prefetch_hints"], 5)
        self.assertEqual(result.output, runtime.tokenizer.decoded.strip())
        chat.close()

    def test_page_state_flush_failure_never_discards_generated_text(self) -> None:
        runtime = _Runtime()

        class PageRouter:
            page_count = 272
            route_width = 192

            @staticmethod
            def metrics():
                return {"exact_rows": 4, "selected_advances": 3}

            @staticmethod
            def flush() -> None:
                raise OSError("state disk full")

        runtime.mlp_page_router = PageRouter()
        chat = _chat(runtime)

        result = chat.handle(Request("chat", "hello"))

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.output, runtime.tokenizer.decoded.strip())
        self.assertIn(
            "state disk full",
            result.evidence["mlp_page_route"]["persistence_error"],
        )
        chat.close()

    def test_adaptive_page_route_runtime_evidence_is_request_scoped(self) -> None:
        runtime = _Runtime()

        class PageRouter:
            page_count = 272
            route_width = 192

            def __init__(self) -> None:
                self.metric_calls = 0

            def metrics(self):
                before = self.metric_calls == 0
                self.metric_calls += 1
                return {
                    "agent_weights": {
                        "temporal": 0.4,
                        "cross_layer": 0.3,
                        "coactive": 0.2,
                        "marginal": 0.1,
                    },
                    "adaptive_width_pages_saved": 40 if before else 136,
                    "adaptive_width_predictions": 2 if before else 5,
                    "energy_coverage": 0.995,
                    "energy_feedback_rows": 7 if before else 11,
                    "last_runtime_reward": 0.0 if before else 2.5,
                    "last_width_mean": 192.0 if before else 128.0,
                    "last_width_min": 192 if before else 96,
                    "policy": (
                        "dynamic-page-transitions+coactivation+"
                        "adaptive-width+terminal-reward+fixed-share/v6"
                    ),
                    "schema": "immer.qwen3.8-mlp-page-markov/v6",
                    "runtime_reward_mean": 0.0 if before else 2.5,
                    "runtime_reward_receipts": 0 if before else 1,
                    "width_actions": (96, 128, 160, 192),
                    "width_agent_weights": {
                        "temporal": 0.5,
                        "cross_layer": 0.3,
                        "marginal": 0.2,
                    },
                    "width_cross_contexts": 3,
                    "width_marginal_contexts": 4,
                    "width_temporal_contexts": 5,
                }

            @staticmethod
            def flush() -> None:
                pass

        router = PageRouter()
        runtime.mlp_page_router = router
        chat = _chat(runtime)

        result = chat.handle(Request("chat", "hello"))

        self.assertTrue(result.ok, result.reason)
        route = result.evidence["mlp_page_route"]
        self.assertEqual(route["page_count"], 272)
        self.assertEqual(route["route_width"], 192)
        self.assertEqual(route["request"]["adaptive_width_pages_saved"], 96)
        self.assertEqual(route["request"]["adaptive_width_predictions"], 3)
        self.assertEqual(route["request"]["energy_feedback_rows"], 4)
        self.assertEqual(route["request"]["runtime_reward_receipts"], 1)
        self.assertEqual(
            route["runtime"],
            {
                "agent_weights": {
                    "temporal": 0.4,
                    "cross_layer": 0.3,
                    "coactive": 0.2,
                    "marginal": 0.1,
                },
                "energy_coverage": 0.995,
                "last_runtime_reward": 2.5,
                "last_width_mean": 128.0,
                "last_width_min": 96,
                "policy": (
                    "dynamic-page-transitions+coactivation+"
                    "adaptive-width+terminal-reward+fixed-share/v6"
                ),
                "schema": "immer.qwen3.8-mlp-page-markov/v6",
                "runtime_reward_mean": 2.5,
                "runtime_reward_receipts": 1,
                "width_actions": (96, 128, 160, 192),
                "width_agent_weights": {
                    "temporal": 0.5,
                    "cross_layer": 0.3,
                    "marginal": 0.2,
                },
                "width_cross_contexts": 3,
                "width_marginal_contexts": 4,
                "width_temporal_contexts": 5,
            },
        )
        self.assertEqual(router.metric_calls, 2)
        chat.close()

    def test_joint_runtime_reward_begins_settles_and_exposes_o1_evidence(
        self,
    ) -> None:
        runtime = _Runtime()

        class Q4Metrics:
            def __init__(self) -> None:
                self.calls = 0

            def metrics(self):
                before = self.calls == 0
                selected = 20 if before else 68
                self.calls += 1
                return {
                    "page_mlp_rows": 0 if before else 1,
                    "page_mlp_selected_pages": selected,
                    "resident_budget_bytes": 17_179_869_184,
                    "resident_budget_overage_bytes": 0,
                    "resident_hits": 0 if before else 5,
                    "resident_misses": 0 if before else 2,
                    "resident_payload_bytes": 0 if before else 16_398_909_440,
                    "resident_peak_payload_bytes": 16_398_909_440,
                    "resident_tensors": 0 if before else 498,
                }

        class PageRouter:
            page_count = 272
            route_width = 192

            def __init__(self) -> None:
                self.events = []
                self.metric_calls = 0

            def begin_runtime_reward(self) -> None:
                self.events.append(("begin",))

            def flush(self) -> None:
                self.events.append(("flush",))

            def metrics(self):
                saved = 10 if self.metric_calls == 0 else 42
                self.metric_calls += 1
                return {
                    "adaptive_width_pages_saved": saved,
                    "last_runtime_reward": 0.0,
                    "runtime_reward_mean": 0.0,
                    "runtime_reward_receipts": 0,
                    "runtime_reward_updates": 0,
                }

            def settle_runtime_reward(self, receipt: str, reward: float):
                self.events.append(("settle", receipt, reward))
                return {
                    "last_runtime_reward": reward,
                    "runtime_reward_mean": reward,
                    "runtime_reward_receipts": 1,
                    "runtime_reward_updates": 1,
                }

        class Retention:
            def __init__(self) -> None:
                self.calls = 0

            def metrics(self):
                self.calls += 1
                if not runtime.model.calls:
                    return {"sequence": 7}
                return {
                    "last_score": {"priority": 9.0},
                    "sequence": 8,
                }

        q4 = Q4Metrics()
        router = PageRouter()
        runtime.model.pager.q4_bank = q4
        runtime.mlp_page_router = router
        chat = _chat(runtime)
        self.assertIs(chat._load_locked(), runtime)
        retention = Retention()
        chat._markov_o1_retention = retention

        result = chat.handle(Request("chat", "hello"))

        self.assertTrue(result.ok, result.reason)
        expected = (
            3.0 * (224 / (48 + 224))
            + math.tanh(math.log1p(9.0) / 4.0)
            - math.log1p(3) / 2.0
        )
        reward = result.evidence["runtime_reward"]
        self.assertEqual(reward["schema"], "immer.qwen3.8-joint-runtime-reward/v1")
        self.assertEqual(reward["accepted_draft_tokens"], 0)
        self.assertEqual(reward["page_actions"], 48)
        self.assertEqual(reward["page_actions_saved"], 224)
        self.assertEqual(reward["o1_priority"], 9.0)
        self.assertAlmostEqual(reward["reward"], expected)
        self.assertEqual(reward["router_updates"], 1)
        self.assertTrue(reward["settled"])
        self.assertEqual(len(reward["receipt_sha256"]), 64)
        page_runtime = result.evidence["mlp_page_route"]["runtime"]
        self.assertAlmostEqual(page_runtime["last_runtime_reward"], expected)
        self.assertAlmostEqual(page_runtime["runtime_reward_mean"], expected)
        self.assertEqual(page_runtime["runtime_reward_receipts"], 1)
        self.assertEqual(router.events[0], ("begin",))
        self.assertEqual(router.events[1], ("flush",))
        self.assertEqual(router.events[2][0], "settle")
        self.assertEqual(router.events[2][1], reward["receipt_sha256"])
        self.assertAlmostEqual(router.events[2][2], expected)
        self.assertEqual(
            result.evidence["mlp_page_route"]["request"]["physical_pages_saved"],
            224,
        )
        self.assertEqual(result.evidence["q4"]["request"]["resident_hits"], 5)
        self.assertEqual(result.evidence["q4"]["request"]["resident_misses"], 2)
        self.assertEqual(
            result.evidence["q4"]["runtime"],
            {
                "resident_budget_bytes": 17_179_869_184,
                "resident_budget_overage_bytes": 0,
                "resident_payload_bytes": 16_398_909_440,
                "resident_peak_payload_bytes": 16_398_909_440,
                "resident_tensors": 498,
            },
        )
        self.assertEqual(q4.calls, 2)
        self.assertGreaterEqual(retention.calls, 3)
        chat.close()

    def test_failed_generation_aborts_the_open_page_reward_trace(self) -> None:
        runtime = _Runtime(model=_Model(generation_error=RuntimeError("boom")))

        class PageRouter:
            page_count = 272
            route_width = 192

            def __init__(self) -> None:
                self.events = []

            def abort_runtime_reward(self) -> None:
                self.events.append("abort")

            def begin_runtime_reward(self) -> None:
                self.events.append("begin")

            @staticmethod
            def metrics():
                return {"adaptive_width_pages_saved": 0}

        router = PageRouter()
        runtime.mlp_page_router = router
        chat = _chat(runtime)
        result = chat.handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.ERROR)
        self.assertEqual(router.events, ["begin", "abort"])
        self.assertNotIn("runtime_reward", result.evidence)
        chat.close()

    def test_page_reward_settlement_failure_is_retried_before_next_begin(
        self,
    ) -> None:
        runtime = _Runtime()

        class PageRouter:
            page_count = 272
            route_width = 192

            def __init__(self) -> None:
                self.active = False
                self.events = []
                self.last_reward = 0.0
                self.receipts = []
                self.reward_sum = 0.0
                self.settle_calls = 0
                self.updates = 0

            def abort_runtime_reward(self) -> None:
                self.events.append(("abort",))
                self.active = False

            def begin_runtime_reward(self) -> None:
                if self.active:
                    raise AssertionError("begin preceded pending retry")
                self.events.append(("begin",))
                self.active = True

            @staticmethod
            def flush() -> None:
                pass

            def metrics(self):
                return {
                    "adaptive_width_pages_saved": 0,
                    "last_runtime_reward": self.last_reward,
                    "runtime_reward_mean": (
                        0.0 if not self.updates else self.reward_sum / self.updates
                    ),
                    "runtime_reward_receipts": len(self.receipts),
                    "runtime_reward_updates": self.updates,
                }

            def settle_runtime_reward(self, receipt: str, reward: float):
                self.settle_calls += 1
                if self.settle_calls <= 2:
                    self.events.append(("settle-failed", receipt, reward))
                    raise OSError("disk full")
                phase = "retry" if self.settle_calls == 3 else "settle"
                self.events.append((phase, receipt, reward))
                self.active = False
                self.receipts.append(receipt)
                self.last_reward = reward
                self.reward_sum += reward
                self.updates += 1
                return self.metrics()

        router = PageRouter()
        runtime.mlp_page_router = router
        chat = _chat(runtime)

        first = chat.handle(Request("chat", "first"))

        self.assertIs(first.status, ExecutionStatus.ERROR)
        self.assertEqual(first.reason, "Qwen3.8 runtime reward settlement failed")
        first_reward = first.evidence["runtime_reward"]
        self.assertEqual(
            first_reward["settlement"]["status"],
            "retryable-error",
        )
        self.assertEqual(router.events[0], ("begin",))
        self.assertEqual(router.events[1][0], "settle-failed")
        self.assertNotIn(("abort",), router.events)
        self.assertTrue(router.active)
        self.assertEqual(
            chat._pending_page_runtime_reward,
            {
                "receipt_sha256": first_reward["receipt_sha256"],
                "reward": first_reward["reward"],
            },
        )

        runtime.model.generated = (9,)
        second = chat.handle(Request("chat", "second"))

        self.assertIs(second.status, ExecutionStatus.ERROR)
        self.assertEqual(router.events[2][0], "settle-failed")
        self.assertEqual(router.events[2][1], first_reward["receipt_sha256"])
        self.assertNotIn(("abort",), router.events)
        self.assertTrue(router.active)
        self.assertEqual(
            chat._pending_page_runtime_reward["receipt_sha256"],
            first_reward["receipt_sha256"],
        )

        runtime.model.generated = (10,)
        third = chat.handle(Request("chat", "third"))

        self.assertTrue(third.ok, third.reason)
        self.assertEqual(router.events[3][0], "retry")
        self.assertEqual(router.events[3][1], first_reward["receipt_sha256"])
        self.assertEqual(router.events[4], ("begin",))
        self.assertEqual(router.events[5][0], "settle")
        self.assertNotEqual(router.events[5][1], first_reward["receipt_sha256"])
        self.assertNotIn(("abort",), router.events)
        self.assertFalse(router.active)
        self.assertIsNone(chat._pending_page_runtime_reward)
        self.assertTrue(third.evidence["runtime_reward"]["settled"])
        self.assertEqual(third.evidence["runtime_reward"]["router_updates"], 2)
        chat.close()

    def test_reward_retry_is_discarded_when_secondary_reset_retires_owner(
        self,
    ) -> None:
        runtime = _Runtime(model=_Model(cleanup_error=RuntimeError("reset failed")))

        class PageRouter:
            page_count = 272
            route_width = 192

            def __init__(self) -> None:
                self.events = []

            def abort_runtime_reward(self) -> None:
                self.events.append("abort")

            def begin_runtime_reward(self) -> None:
                self.events.append("begin")

            @staticmethod
            def flush() -> None:
                pass

            @staticmethod
            def metrics():
                return {"adaptive_width_pages_saved": 0}

            def settle_runtime_reward(self, _receipt: str, _reward: float):
                self.events.append("settle-failed")
                raise OSError("disk full")

        router = PageRouter()
        runtime.mlp_page_router = router
        chat = _chat(runtime)
        result = chat.handle(
            Request(
                "chat",
                "hello",
                metadata={QWEN38_CHAT_SESSION_METADATA: "conversation:reward"},
            )
        )

        self.assertIs(result.status, ExecutionStatus.ERROR)
        self.assertEqual(result.reason, "Qwen3.8 state cleanup failed")
        self.assertEqual(router.events, ["begin", "settle-failed", "abort"])
        self.assertIsNone(chat._pending_page_runtime_reward)
        self.assertIsNone(chat._runtime)
        chat.close()

    def test_close_aborts_pending_reward_before_owner_shutdown(self) -> None:
        runtime = _Runtime()

        class PageRouter:
            page_count = 272
            route_width = 192

            def __init__(self) -> None:
                self.events = []

            def abort_runtime_reward(self) -> None:
                self.events.append("abort")

            def begin_runtime_reward(self) -> None:
                self.events.append("begin")

            @staticmethod
            def flush() -> None:
                pass

            @staticmethod
            def metrics():
                return {"adaptive_width_pages_saved": 0}

            def settle_runtime_reward(self, _receipt: str, _reward: float):
                self.events.append("settle-failed")
                raise OSError("disk full")

        router = PageRouter()
        runtime.mlp_page_router = router
        chat = _chat(runtime)
        result = chat.handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.ERROR)
        self.assertIsNotNone(chat._pending_page_runtime_reward)
        chat.close()
        self.assertEqual(router.events, ["begin", "settle-failed", "abort"])
        self.assertIsNone(chat._pending_page_runtime_reward)
        self.assertTrue(chat.closed)

    def test_exact_head_non_cpu_configuration_is_lazy_nonapplicable(self) -> None:
        component = Qwen38CausalChat(
            "unused.causal",
            "unused-tokenizer.json",
            device="mps",
            compute_dtype="float32",
            exact_head_root="/artifacts/qwen-head",
        )

        self.assertFalse(component.loaded)
        component.close()

    def test_q4_selects_cpu_and_composes_with_sparse_mlp(self) -> None:
        component = Qwen38CausalChat(
            "unused.causal",
            "unused-tokenizer.json",
            q4_root="/models/qwen-q4",
        )
        self.assertEqual(component._device, "cpu")
        component.close()

        with self.assertRaisesRegex(ValueError, "requires q4_root"):
            Qwen38CausalChat(
                "unused.causal",
                "unused-tokenizer.json",
                q4_resident_budget_bytes=1024,
            )
        for invalid_budget in (True, -1, 1.5):
            with self.subTest(invalid_budget=invalid_budget):
                with self.assertRaises(ValueError):
                    Qwen38CausalChat(
                        "unused.causal",
                        "unused-tokenizer.json",
                        q4_root="/models/qwen-q4",
                        q4_resident_budget_bytes=invalid_budget,
                    )

        component = Qwen38CausalChat(
            "unused.causal",
            "unused-tokenizer.json",
            q4_root="/models/qwen-q4",
            mlp_page_state_path="/state/mlp-pages.json",
            delta_head_state_path="/state/delta-head.json",
            delta_head_active_layers=(0, 2, 4),
        )
        self.assertIsNone(component._fast_mlp_paths)
        self.assertEqual(component._delta_head_active_layers, (0, 2, 4))
        component.close()

        component = Qwen38CausalChat(
            "unused.causal",
            "unused-tokenizer.json",
            q4_root="/models/qwen-q4",
            draft_mode="markov",
            contextual_continuation_state_path="/state/context.json",
        )
        self.assertEqual(
            component._contextual_continuation_state_path,
            Path("/state/context.json"),
        )
        component.close()

        with self.assertRaisesRegex(ValueError, "require Q4"):
            Qwen38CausalChat(
                "unused.causal",
                "unused-tokenizer.json",
                draft_mode="markov",
                contextual_continuation_state_path="/state/context.json",
            )

        with self.assertRaisesRegex(
            ValueError,
            "requires delta_head_state_path",
        ):
            Qwen38CausalChat(
                "unused.causal",
                "unused-tokenizer.json",
                q4_root="/models/qwen-q4",
                delta_head_active_layers=(0, 2),
            )

        component = Qwen38CausalChat(
            "unused.causal",
            "unused-tokenizer.json",
            q4_root="/models/qwen-q4",
            fast_mlp_root="/artifacts/fast",
            fast_mlp_selected_block_count=96,
            delta_head_state_path="/state/delta-head.json",
        )
        self.assertEqual(component._device, "cpu")
        self.assertIsNotNone(component._fast_mlp_paths)
        self.assertEqual(component._fast_mlp_selected_block_count, 96)
        self.assertEqual(
            component._delta_head_state_path,
            Path("/state/delta-head.json"),
        )
        component.close()

        with self.assertRaisesRegex(ValueError, "requires Q4"):
            Qwen38CausalChat(
                "unused.causal",
                "unused-tokenizer.json",
                fast_mlp_root="/artifacts/fast",
                fast_mlp_selected_block_count=96,
            )

        exact_q4 = Qwen38CausalChat(
            "unused.causal",
            "unused-tokenizer.json",
            q4_root="/models/qwen-q4",
            exact_head_root="/artifacts/head",
        )
        exact_q4.close()
        with self.assertRaisesRegex(ValueError, "Q4 execution replaces"):
            Qwen38CausalChat(
                "unused.causal",
                "unused-tokenizer.json",
                q4_root="/models/qwen-q4",
                range_markov_state_path="/state/ranges",
            )

    def test_direct_markov_page_route_requires_q4_and_replaces_legacy_sparse(
        self,
    ) -> None:
        component = Qwen38CausalChat(
            "unused.causal",
            "unused-tokenizer.json",
            q4_root="/models/qwen-q4",
            mlp_page_state_path="/state/mlp-pages.json",
            mlp_page_coordinate_state_path="/state/mlp-coordinate.json",
            mlp_page_route_width=192,
        )
        self.assertEqual(
            component._mlp_page_state_path,
            Path("/state/mlp-pages.json"),
        )
        self.assertEqual(component._mlp_page_route_width, 192)
        self.assertEqual(
            component._mlp_page_coordinate_state_path,
            Path("/state/mlp-coordinate.json"),
        )
        component.close()

        with self.assertRaisesRegex(ValueError, "requires Q4"):
            Qwen38CausalChat(
                "unused.causal",
                "unused-tokenizer.json",
                mlp_page_state_path="/state/mlp-pages.json",
            )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            Qwen38CausalChat(
                "unused.causal",
                "unused-tokenizer.json",
                q4_root="/models/qwen-q4",
                fast_mlp_root="/artifacts/fast",
                mlp_page_state_path="/state/mlp-pages.json",
            )
        with self.assertRaisesRegex(ValueError, "leave at least one page"):
            Qwen38CausalChat(
                "unused.causal",
                "unused-tokenizer.json",
                q4_root="/models/qwen-q4",
                mlp_page_state_path="/state/mlp-pages.json",
                mlp_page_route_width=272,
            )
        with self.assertRaisesRegex(ValueError, "require Q4 MLP page routing"):
            Qwen38CausalChat(
                "unused.causal",
                "unused-tokenizer.json",
                q4_root="/models/qwen-q4",
                mlp_page_coordinate_state_path="/state/mlp-coordinate.json",
            )
        with self.assertRaisesRegex(ValueError, "require bfloat16"):
            Qwen38CausalChat(
                "unused.causal",
                "unused-tokenizer.json",
                compute_dtype="float32",
                q4_root="/models/qwen-q4",
                mlp_page_state_path="/state/mlp-pages.json",
                mlp_page_coordinate_state_path="/state/mlp-coordinate.json",
            )

    def test_success_is_lazy_uses_no_thinking_prompt_and_returns_compact_receipts(
        self,
    ) -> None:
        runtime = _Runtime()
        factory_calls = 0

        def factory():
            nonlocal factory_calls
            factory_calls += 1
            return runtime

        chat = _chat(runtime, runtime_factory=factory, system_prompt=" local system ")
        self.assertFalse(chat.loaded)
        self.assertEqual(factory_calls, 0)

        result = chat.handle(Request("chat", "  hello  "))

        self.assertIs(result.status, ExecutionStatus.OK)
        self.assertEqual(result.output, "local answer")
        self.assertTrue(chat.loaded)
        self.assertEqual(factory_calls, 1)
        self.assertEqual(
            runtime.tokenizer.encoded,
            [Qwen38Tokenizer.render_no_thinking_prompt("local system", "hello")],
        )
        self.assertNotIn("transformers", runtime.tokenizer.encoded[0].lower())
        self.assertEqual(runtime.model.calls[0][0], [[11, 12]])
        self.assertEqual(
            runtime.model.calls[0][1],
            {
                "max_new_tokens": 3,
                "prefill_tokenwise": False,
                "eos_token_ids": (248046, 248044),
                "head_block_rows": 17,
                "retain_final_state": False,
            },
        )
        self.assertEqual(runtime.model.reset_calls, [True])
        self.assertEqual(result.evidence["model"], "Qwen/Qwen3.8-27B")
        self.assertEqual(result.evidence["tokenizer_sha256"], _DIGEST)
        self.assertEqual(result.evidence["bundle"], _BUNDLE_RECEIPT)
        generation = result.evidence["generation"]
        self.assertEqual(generation["prompt_tokens"], 2)
        self.assertEqual(generation["generated_tokens"], 2)
        self.assertEqual(generation["output_tokens_per_second"], 1.6)
        self.assertEqual(len(generation["token_trace_sha256"]), 64)
        self.assertNotIn("prompt_token_ids", generation)
        self.assertNotIn("generated_token_ids", generation)
        runtime_metrics = result.evidence["runtime_metrics"]
        self.assertIn("major_page_faults", runtime_metrics)
        self.assertIn("minor_page_faults", runtime_metrics)
        self.assertIn("process_current_rss_bytes", runtime_metrics)
        self.assertIn("system_cpu_seconds", runtime_metrics)
        self.assertIn("user_cpu_seconds", runtime_metrics)
        self.assertNotIn("result_cell_binding_receipt", result.evidence)
        self.assertEqual(
            result.evidence["conversation"],
            {
                "history_messages": 0,
                "history_turns": 0,
                "mtp_carry_bytes": 0,
                "mtp_carry_reused_tokens": 0,
                "mtp_carry_status": "none",
                "prompt_suffix_tokens": 2,
                "reuse_hits": 0,
                "reuse_misses": 0,
                "reuse_status": "disabled",
                "reused_prefix_tokens": 0,
                "state_retained_tokens": 0,
            },
        )

    def test_runtime_metrics_expose_request_local_component_nanoseconds(
        self,
    ) -> None:
        model = _ComponentTimingModel()
        result = _chat(_Runtime(model=model)).handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.OK)
        timings = result.evidence["runtime_metrics"]["component_timings"]
        self.assertEqual(
            timings["schema"],
            "immer.qwen3.8-component-timing-request/v1",
        )
        self.assertEqual(
            timings["source_schema"],
            "immer.qwen3.8-component-timing-counters/v1",
        )
        self.assertEqual(timings["clock"], "time.perf_counter_ns")
        self.assertEqual(timings["unit"], "nanoseconds")
        self.assertEqual(timings["status"], "ok")
        self.assertIsNone(timings["accounting_error"])
        self.assertEqual(timings["accounting_failures"], 0)
        self.assertEqual(timings["measured_nanoseconds"], 5_576)
        self.assertEqual(
            timings["components"],
            {
                "full_attention_core": {
                    "boundary": "StreamedQwen38._full_attention",
                    "calls": 4,
                    "nanoseconds": 400,
                },
                "deltanet_core": {
                    "boundary": "StreamedQwen38._linear_attention",
                    "calls": 12,
                    "nanoseconds": 1_200,
                },
                "mlp_core": {
                    "boundary": "StreamedQwen38._mlp",
                    "calls": 16,
                    "nanoseconds": 3_200,
                },
                "lm_head_core": {
                    "boundary": "Qwen38WeightPager.topk_logits",
                    "calls": 2,
                    "nanoseconds": 600,
                },
                "layer_transition_crystal": {
                    "boundary": ("StreamedQwen38._layer_transition_crystal_forward"),
                    "calls": 16,
                    "nanoseconds": 80,
                },
                "layer_mlp_crystal": {
                    "boundary": "StreamedQwen38._layer_mlp_crystal_forward",
                    "calls": 16,
                    "nanoseconds": 96,
                },
            },
        )

    def test_component_timing_accounting_failure_is_request_fail_safe(self) -> None:
        model = _ComponentTimingModel(invalid_after_generation=True)
        result = _chat(_Runtime(model=model)).handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.OK)
        self.assertEqual(result.output, "local answer")
        timings = result.evidence["runtime_metrics"]["component_timings"]
        self.assertEqual(timings["status"], "accounting-error")
        self.assertEqual(timings["accounting_error"], "invalid-model-counter")
        self.assertEqual(timings["components"], {})
        self.assertIsNone(timings["measured_nanoseconds"])

    def test_chat_history_renders_exact_multi_turn_qwen_context(self) -> None:
        runtime = _Runtime()
        chat = _chat(runtime, system_prompt="stay concise")
        history = (
            ("user", "My code is ORBIT-7."),
            ("assistant", "Understood."),
        )

        result = chat.handle(
            Request(
                "chat",
                "What was my code?",
                {QWEN38_CHAT_HISTORY_METADATA: history},
            )
        )

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(
            runtime.tokenizer.encoded,
            [
                Qwen38Tokenizer.render_no_thinking_messages(
                    "stay concise",
                    (*history, ("user", "What was my code?")),
                )
            ],
        )
        self.assertEqual(
            result.evidence["conversation"],
            {
                "history_messages": 2,
                "history_turns": 1,
                "mtp_carry_bytes": 0,
                "mtp_carry_reused_tokens": 0,
                "mtp_carry_status": "none",
                "prompt_suffix_tokens": 2,
                "reuse_hits": 0,
                "reuse_misses": 0,
                "reuse_status": "disabled",
                "reused_prefix_tokens": 0,
                "state_retained_tokens": 0,
            },
        )
        chat.close()

    def test_conversation_session_reuses_only_exact_committed_token_prefix(
        self,
    ) -> None:
        class ConversationTokenizer(_Tokenizer):
            def encode(self, text: str):
                self.encoded.append(text)
                if "Start over." in text:
                    return (21, 22)
                if "What was the word?" in text:
                    return (11, 12, 7, IM_END_TOKEN_ID, 13, 14)
                return (11, 12)

            def decode(self, token_ids):
                ids = tuple(token_ids)
                self.decoded_ids.append(ids)
                return "alpha" if ids[0] == 7 else "beta"

        class ConversationModel(_Model):
            def generate_greedy(self, prompt, **kwargs):
                self.generated = (
                    (7, IM_END_TOKEN_ID) if not self.calls else (8, IM_END_TOKEN_ID)
                )
                tokens, evidence = super().generate_greedy(prompt, **kwargs)
                return tokens, {**evidence, "stopped_on_eos": True}

        runtime = _Runtime(
            model=ConversationModel(),
            tokenizer=ConversationTokenizer(),
        )
        chat = _chat(runtime, max_prompt_tokens=8)
        session = {QWEN38_CHAT_SESSION_METADATA: "conversation:test"}

        first = chat.handle(Request("chat", "Remember alpha.", session))
        second = chat.handle(
            Request(
                "chat",
                "What was the word?",
                {
                    **session,
                    QWEN38_CHAT_HISTORY_METADATA: (
                        ("user", "Remember alpha."),
                        ("assistant", "alpha"),
                    ),
                },
            )
        )

        self.assertTrue(first.ok, first.reason)
        self.assertTrue(second.ok, second.reason)
        self.assertEqual(first.output, "alpha")
        self.assertEqual(second.output, "beta")
        self.assertEqual(runtime.model.reset_calls, [])
        self.assertEqual(
            runtime.model.calls[1][1]["restored_prefix_length"],
            3,
        )
        self.assertEqual(second.evidence["conversation"]["reuse_status"], "hit")
        self.assertEqual(second.evidence["conversation"]["reused_prefix_tokens"], 3)
        self.assertEqual(second.evidence["conversation"]["prompt_suffix_tokens"], 3)
        self.assertEqual(second.evidence["conversation"]["state_retained_tokens"], 7)
        self.assertEqual(runtime.model.pager.release.call_count, 2)

        restarted = chat.handle(Request("chat", "Start over.", session))
        self.assertTrue(restarted.ok, restarted.reason)
        self.assertEqual(
            restarted.evidence["conversation"]["reuse_status"],
            "token-prefix-mismatch",
        )
        self.assertNotIn("restored_prefix_length", runtime.model.calls[2][1])
        self.assertEqual(runtime.model.reset_calls, [True])

        chat.clear_conversation()
        self.assertEqual(runtime.model.reset_calls, [True, True])
        self.assertEqual(chat._conversation_prefix_token_ids, ())
        chat.close()

    def test_invalid_chat_history_is_rejected_before_runtime_load(self) -> None:
        runtime = _Runtime()
        chat = _chat(runtime)

        result = chat.handle(
            Request(
                "chat",
                "hello",
                {QWEN38_CHAT_HISTORY_METADATA: (("user", "unfinished"),)},
            )
        )

        self.assertIs(result.status, ExecutionStatus.REJECTED)
        self.assertIn("completed turns", result.reason or "")
        self.assertFalse(chat.loaded)
        self.assertEqual(runtime.model.calls, [])
        chat.close()

    def test_zero_forward_warm_bypass_releases_only_existing_native_state(
        self,
    ) -> None:
        cold_runtime = _Runtime()
        cold = _chat(cold_runtime)
        cold.release_warm_bypass_state()
        self.assertFalse(cold.loaded)
        self.assertEqual(cold_runtime.model.reset_calls, [])
        cold.close()

        runtime = _Runtime()
        chat = _chat(runtime)
        session = {QWEN38_CHAT_SESSION_METADATA: "semantic:warm"}
        result = chat.handle(Request("chat", "hello", session))
        self.assertTrue(result.ok, result.reason)
        self.assertTrue(chat._conversation_prefix_token_ids)
        self.assertEqual(runtime.model.reset_calls, [])

        chat.release_warm_bypass_state()

        self.assertEqual(runtime.model.reset_calls, [True])
        self.assertEqual(chat._conversation_prefix_token_ids, ())
        self.assertIsNone(chat._conversation_session_id)
        self.assertIsNone(chat._conversation_mtp_carry)
        chat.close()

        broken_runtime = _Runtime(
            model=_Model(cleanup_error=RuntimeError("cannot release warm state"))
        )
        broken = _chat(broken_runtime)
        broken._load_locked()
        with self.assertRaisesRegex(Qwen38ChatError, "warm bypass state cleanup"):
            broken.release_warm_bypass_state()
        self.assertIsNone(broken._runtime)
        self.assertEqual(broken_runtime.close_calls, 1)
        broken.close()

    def test_conversation_generation_failure_drops_retained_state(self) -> None:
        class FailureTokenizer(_Tokenizer):
            def encode(self, text: str):
                self.encoded.append(text)
                if "Continue." in text:
                    return (11, 12, 7, IM_END_TOKEN_ID, 13)
                return (11, 12)

            def decode(self, token_ids):
                self.decoded_ids.append(tuple(token_ids))
                return "alpha"

        class FailureModel(_Model):
            def generate_greedy(self, prompt, **kwargs):
                tokens, evidence = super().generate_greedy(prompt, **kwargs)
                return tokens, {**evidence, "stopped_on_eos": True}

        model = FailureModel(generated=(7, IM_END_TOKEN_ID))
        runtime = _Runtime(model=model, tokenizer=FailureTokenizer())
        chat = _chat(runtime, max_prompt_tokens=8)
        session = {QWEN38_CHAT_SESSION_METADATA: "conversation:failure"}
        first = chat.handle(Request("chat", "Start.", session))
        self.assertTrue(first.ok, first.reason)
        self.assertTrue(chat._conversation_prefix_token_ids)

        model.generation_error = RuntimeError("decode failed")
        failed = chat.handle(
            Request(
                "chat",
                "Continue.",
                {
                    **session,
                    QWEN38_CHAT_HISTORY_METADATA: (
                        ("user", "Start."),
                        ("assistant", "alpha"),
                    ),
                },
            )
        )

        self.assertIs(failed.status, ExecutionStatus.ERROR)
        self.assertEqual(chat._conversation_prefix_token_ids, ())
        self.assertEqual(model.reset_calls, [True])
        chat.close()

    def test_direct_generation_emits_cumulative_text_snapshots(self) -> None:
        runtime = _Runtime()
        runtime.model = _StreamingModel(generated=(7, 8, 9))
        runtime.tokenizer = _StreamingTokenizer()
        snapshots: list[str] = []
        chat = _chat(runtime, text_snapshot_sink=snapshots.append)

        result = chat.handle(Request("chat", "hello"))

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.output, "Hello world!")
        self.assertEqual(snapshots, ["Hello", "Hello world", "Hello world!"])
        self.assertEqual(runtime.model.reset_calls, [True])
        chat.close()

    def test_draft_round_expands_to_one_snapshot_per_accepted_token(self) -> None:
        runtime = _Runtime(tokenizer=_StreamingTokenizer())
        snapshots: list[str] = []
        chat = _chat(runtime, text_snapshot_sink=snapshots.append)
        progress = chat._draft_generation_progress(runtime)
        assert progress is not None

        progress((7, 8))
        progress((7, 8, 9))

        self.assertEqual(snapshots, ["Hello", "Hello world", "Hello world!"])
        with self.assertRaisesRegex(Qwen38ChatError, "moved backwards"):
            progress((7,))
        chat.close()

    def test_stream_sink_failure_is_a_generation_error_and_state_is_released(
        self,
    ) -> None:
        runtime = _Runtime()
        runtime.model = _StreamingModel(generated=(7,))
        runtime.tokenizer = _StreamingTokenizer()

        def fail(_snapshot: str) -> None:
            raise BrokenPipeError("closed output")

        chat = _chat(runtime, text_snapshot_sink=fail)
        result = chat.handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.ERROR)
        self.assertIn("BrokenPipeError", result.reason or "")
        self.assertEqual(runtime.model.reset_calls, [True])
        chat.close()

    def test_stream_sink_must_be_callable(self) -> None:
        with self.assertRaisesRegex(TypeError, "text_snapshot_sink"):
            _chat(_Runtime(), text_snapshot_sink="stdout")

    def test_optional_k4_drafter_is_used_by_general_chat(self) -> None:
        target = _Runtime()
        target.fast_mlp_mount = _FastMount()
        target.fast_mlp_receipt = dict(_FAST_RECEIPT)
        chat = _chat(
            target,
            draft_bundle_path="draft.causal",
            max_new_tokens=4,
            fast_mlp_root="/artifacts/fast-mlp",
            fast_mlp_active_layers=(0, 9),
        )
        chat._draft_runtime = SimpleNamespace(
            tokenizer_sha256=_DIGEST,
            model=SimpleNamespace(config=SimpleNamespace(vocab_size=300_000)),
            bundle_receipt=_BUNDLE_RECEIPT,
            close=lambda: None,
        )
        provider = SimpleNamespace(
            metrics=lambda: SimpleNamespace(
                source_body_bytes=12,
                linear_calls=3,
            ),
            close=lambda: None,
        )
        evidence = SimpleNamespace(
            accepted_draft_tokens=4,
            source_body_bytes=100,
            linear_calls=10,
            seconds=1.25,
            state_bytes=456,
            stopped_on_eos=False,
            prompt_token_ids=(11, 12),
            generated_token_ids=(7, 8, 9, 10),
            forward_passes=2,
            rounds=(SimpleNamespace(proposed_token_ids=(7, 8, 9, 10)),),
            schema="immer.qwen3.8-rolling-k4-speculative-generation/v1",
            final_state_committed=False,
        )
        generated = SimpleNamespace(token_ids=(7, 8, 9, 10), evidence=evidence)
        decoder = SimpleNamespace(generate_rolling=lambda *args, **kwargs: generated)
        with (
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen35K4DraftProvider",
                return_value=provider,
            ),
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38K4SpeculativeDecoder",
                return_value=decoder,
            ) as decoder_constructor,
        ):
            result = chat.handle(Request("chat", "hello"))

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.evidence["generation"]["forward_passes"], 2)
        self.assertEqual(result.evidence["generation"]["source_body_bytes"], 100)
        self.assertEqual(result.evidence["draft"]["accepted_draft_tokens"], 4)
        self.assertEqual(result.evidence["draft"]["proposed_draft_tokens"], 4)
        self.assertEqual(result.evidence["draft"]["draft_linear_calls"], 3)
        self.assertEqual(result.evidence["draft"]["target_source_body_bytes"], 100)
        self.assertEqual(result.evidence["draft"]["total_source_body_bytes"], 132)
        self.assertEqual(result.evidence["draft"]["target_linear_calls"], 10)
        self.assertEqual(result.evidence["draft"]["total_linear_calls"], 13)
        self.assertEqual(result.evidence["generation"]["linear_calls"], 10)
        self.assertEqual(
            result.evidence["fast_mlp"]["request"]["aux_source_body_bytes"], 20
        )
        decoder_constructor.assert_called_once_with(
            target.model,
            provider,
            window_size=4,
            adaptive_round_windows=False,
        )
        chat.close()

    def test_action_bank_directive_can_select_verified_direct_fallback(self) -> None:
        runtime = _Runtime()
        chat = _chat(
            runtime,
            draft_bundle_path="draft.causal",
            max_new_tokens=4,
        )
        directive = InferenceActionDirective(
            question_sha256=hashlib.sha256(b"hello").hexdigest(),
            runtime_profile_sha256="2" * 64,
            primary_actions=("qwen_target",),
            fallback_actions=("qwen_target",),
            draft_enabled=False,
            source_signature_sha256s=("3" * 64,),
            support=1,
            saved_qwen_forwards=0,
        )

        result = chat.handle(
            Request(
                "chat",
                "hello",
                {QWEN38_INFERENCE_ACTION_METADATA: directive.to_document()},
            )
        )

        self.assertTrue(result.ok, result.reason)
        self.assertNotIn("draft", result.evidence)
        self.assertNotIn("draft_enabled", runtime.model.calls[-1][1])
        self.assertEqual(
            result.evidence["inference_action_directive"]["applied"],
            {
                "attention_output_crystal": False,
                "attention_output_crystal_directive_selected": False,
                "attention_output_crystal_explicitly_disabled": False,
                "draft_enabled": False,
                "draft_window_ceiling": None,
                "external_drafter": False,
                "external_drafter_explicitly_disabled": False,
                "layer_mlp_crystal": False,
                "layer_mlp_crystal_configured": False,
                "layer_mlp_crystal_directive_selected": False,
                "layer_mlp_crystal_explicitly_disabled": False,
                "layer_transition_crystal": False,
                "layer_transition_crystal_directive_selected": False,
                "layer_transition_crystal_explicitly_disabled": False,
                "lm_head_coordinate": False,
                "lm_head_coordinate_directive_selected": False,
                "lm_head_coordinate_explicitly_disabled": False,
                "mlp_head_coordinate": False,
                "mlp_head_coordinate_directive_selected": False,
                "mlp_page_coordinate": False,
                "mlp_page_coordinate_directive_selected": False,
                "mlp_page_coordinate_explicitly_disabled": False,
                "prefix_sinkhorn": False,
                "prefix_sinkhorn_directive_selected": False,
                "prefix_sinkhorn_explicitly_disabled": False,
            },
        )
        self.assertEqual(
            result.evidence["inference_action_directive"]["directive"],
            directive.to_document(),
        )
        chat.close()

    def test_physical_prefix_sinkhorn_is_request_local_and_reports_saved_rows(
        self,
    ) -> None:
        model = _PrefixSinkhornModel()
        chat = _chat(
            _Runtime(model=model),
            native_head_crsa=Qwen38NativeHeadCrsa(
                alpha=1.0,
                replace_base_softmax=True,
            ),
        )
        discovery = chat.handle(Request("chat", "hello", {}))
        self.assertTrue(discovery.ok, discovery.reason)
        self.assertEqual(
            discovery.evidence["prefix_sinkhorn"]["request"],
            {
                "base_softmax_head_rows_skipped": 12,
                "base_softmax_probability_elements_skipped": 144,
            },
        )
        self.assertEqual(
            discovery.evidence["prefix_sinkhorn"]["action_identity_sha256"],
            "a" * 64,
        )

        unspecified = InferenceActionDirective(
            question_sha256=hashlib.sha256(b"hello").hexdigest(),
            runtime_profile_sha256="2" * 64,
            primary_actions=("qwen_target",),
            fallback_actions=("qwen_target",),
            draft_enabled=None,
            source_signature_sha256s=("2" * 64,),
            support=1,
            saved_qwen_forwards=0,
        )
        discovery_with_transferable_directive = chat.handle(
            Request(
                "chat",
                "hello",
                {QWEN38_INFERENCE_ACTION_METADATA: (unspecified.to_document())},
            )
        )
        self.assertTrue(
            discovery_with_transferable_directive.evidence["prefix_sinkhorn"]["active"]
        )
        self.assertEqual(model.prefix_activation_calls, [])

        off = InferenceActionDirective(
            question_sha256=hashlib.sha256(b"hello").hexdigest(),
            runtime_profile_sha256="2" * 64,
            primary_actions=("qwen_target",),
            fallback_actions=("qwen_target",),
            draft_enabled=None,
            source_signature_sha256s=("3" * 64,),
            support=1,
            saved_qwen_forwards=0,
            disabled_actions=("prefix_sinkhorn",),
        )
        disabled = chat.handle(
            Request(
                "chat",
                "hello",
                {QWEN38_INFERENCE_ACTION_METADATA: off.to_document()},
            )
        )
        self.assertTrue(disabled.ok, disabled.reason)
        self.assertFalse(disabled.evidence["prefix_sinkhorn"]["active"])
        self.assertEqual(
            disabled.evidence["prefix_sinkhorn"]["request"],
            {
                "base_softmax_head_rows_skipped": 0,
                "base_softmax_probability_elements_skipped": 0,
            },
        )

        on = InferenceActionDirective(
            question_sha256=hashlib.sha256(b"hello").hexdigest(),
            runtime_profile_sha256="2" * 64,
            primary_actions=("prefix_sinkhorn", "qwen_target"),
            fallback_actions=("qwen_target",),
            draft_enabled=None,
            source_signature_sha256s=("4" * 64,),
            support=1,
            saved_qwen_forwards=0,
        )
        enabled = chat.handle(
            Request(
                "chat",
                "hello",
                {QWEN38_INFERENCE_ACTION_METADATA: on.to_document()},
            )
        )
        self.assertTrue(enabled.ok, enabled.reason)
        self.assertTrue(enabled.evidence["prefix_sinkhorn"]["active"])
        self.assertEqual(
            enabled.evidence["inference_action_directive"]["applied"][
                "prefix_sinkhorn"
            ],
            True,
        )
        self.assertNotEqual(
            chat._result_cell_generation_policy_sha256(prefix_sinkhorn_applied=False),
            chat._result_cell_generation_policy_sha256(prefix_sinkhorn_applied=True),
        )
        self.assertEqual(model.prefix_activation_calls, [False, True])
        chat.close()

    def test_delta_coordinate_action_is_request_local_and_uses_qwen_fallback(
        self,
    ) -> None:
        coordinate = ("mlp_head_coordinate", "qwen_target")

        def directive(primary, fallback):
            return InferenceActionDirective(
                question_sha256=hashlib.sha256(b"hello").hexdigest(),
                runtime_profile_sha256="2" * 64,
                primary_actions=primary,
                fallback_actions=fallback,
                draft_enabled=False,
                source_signature_sha256s=("3" * 64,),
                support=1,
                saved_qwen_forwards=0,
            )

        for primary, fallback in (
            (coordinate, coordinate),
            (("parametric_program",), coordinate),
        ):
            with self.subTest(primary=primary):
                runtime = _Runtime()
                router = SimpleNamespace(
                    metrics=Mock(
                        side_effect=(
                            {
                                "calls": 10,
                                "rows": 20,
                                "logical_bytes_saved": 30,
                            },
                            {
                                "calls": 12,
                                "rows": 24,
                                "logical_bytes_saved": 50,
                            },
                        )
                    )
                )
                runtime.delta_head_router = router
                runtime.delta_head_receipt = {
                    "layers": [0],
                    "schema": "fixture.delta-head/v2",
                }
                chat = _chat(
                    runtime,
                    q4_root="/q4",
                    delta_head_state_path="/state/delta.json",
                )

                result = chat.handle(
                    Request(
                        "chat",
                        "hello",
                        {
                            QWEN38_INFERENCE_ACTION_METADATA: directive(
                                primary,
                                fallback,
                            ).to_document()
                        },
                    )
                )

                self.assertTrue(result.ok, result.reason)
                self.assertEqual(
                    runtime.model.delta_head_router_calls,
                    [None, router, None],
                )
                self.assertIsNone(runtime.model.delta_head_router)
                self.assertIn(True, runtime.model.reset_calls)
                self.assertEqual(
                    result.evidence["delta_head_router"]["request"],
                    {
                        "calls": 2,
                        "logical_bytes_saved": 20,
                        "rows": 4,
                        "schema": "immer.qwen3.8-delta-head-request/v1",
                    },
                )
                applied = result.evidence["inference_action_directive"]["applied"]
                self.assertTrue(applied["mlp_head_coordinate"])
                self.assertTrue(applied["mlp_head_coordinate_directive_selected"])
                chat.close()

        runtime = _Runtime()
        router = SimpleNamespace(metrics=Mock())
        runtime.delta_head_router = router
        runtime.delta_head_receipt = {
            "layers": [0],
            "schema": "fixture.delta-head/v2",
        }
        chat = _chat(
            runtime,
            q4_root="/q4",
            delta_head_state_path="/state/delta.json",
        )
        target_only = directive(("qwen_target",), ("qwen_target",))
        result = chat.handle(
            Request(
                "chat",
                "hello",
                {QWEN38_INFERENCE_ACTION_METADATA: target_only.to_document()},
            )
        )
        self.assertTrue(result.ok, result.reason)
        self.assertEqual(runtime.model.delta_head_router_calls, [None])
        router.metrics.assert_not_called()
        self.assertNotIn("request", result.evidence["delta_head_router"])
        self.assertFalse(
            result.evidence["inference_action_directive"]["applied"][
                "mlp_head_coordinate"
            ]
        )
        chat.close()

        failing_model = _Model(generation_error=RuntimeError("delta boom"))
        runtime = _Runtime(model=failing_model)
        router = SimpleNamespace(
            metrics=Mock(
                return_value={
                    "calls": 0,
                    "rows": 0,
                    "logical_bytes_saved": 0,
                }
            )
        )
        runtime.delta_head_router = router
        runtime.delta_head_receipt = {
            "layers": [0],
            "schema": "fixture.delta-head/v2",
        }
        chat = _chat(
            runtime,
            q4_root="/q4",
            delta_head_state_path="/state/delta.json",
        )
        failed = chat.handle(
            Request(
                "chat",
                "hello",
                {
                    QWEN38_INFERENCE_ACTION_METADATA: directive(
                        coordinate,
                        coordinate,
                    ).to_document()
                },
            )
        )
        self.assertEqual(failed.status, ExecutionStatus.ERROR)
        self.assertIsNone(failing_model.delta_head_router)
        self.assertEqual(
            failing_model.delta_head_router_calls,
            [None, router, None],
        )
        chat.close()

    def test_crystal_ceiling_never_leaks_into_direct_target_kwargs(self) -> None:
        runtime = _Runtime()
        chat = _chat(runtime)
        directive = InferenceActionDirective(
            question_sha256=hashlib.sha256(b"hello").hexdigest(),
            runtime_profile_sha256="2" * 64,
            primary_actions=("compute_crystal", "qwen_target"),
            fallback_actions=("compute_crystal", "qwen_target"),
            draft_enabled=True,
            source_signature_sha256s=("3" * 64,),
            support=1,
            saved_qwen_forwards=11,
            draft_window_ceiling=16,
        )

        result = chat.handle(
            Request(
                "chat",
                "hello",
                {QWEN38_INFERENCE_ACTION_METADATA: directive.to_document()},
            )
        )

        self.assertTrue(result.ok, result.reason)
        self.assertNotIn("draft_window_ceiling", runtime.model.calls[0][1])
        chat.close()

    def test_fast_mlp_identity_and_request_traffic_reach_general_chat(self) -> None:
        runtime = _Runtime()
        runtime.fast_mlp_mount = _FastMount()
        runtime.fast_mlp_receipt = dict(_FAST_RECEIPT)
        chat = _chat(
            runtime,
            fast_mlp_root="/artifacts/fast-mlp",
            fast_mlp_active_layers=(0, 9),
        )

        result = chat.handle(Request("chat", "hello"))

        self.assertTrue(result.ok, result.reason)
        fast = result.evidence["fast_mlp"]
        self.assertEqual(fast["router_fit_sha256"], "4" * 64)
        self.assertEqual(fast["request"]["target_source_body_bytes"], 1234)
        self.assertEqual(fast["request"]["aux_source_body_bytes"], 20)
        self.assertEqual(fast["request"]["total_source_body_bytes"], 1254)
        policy = chat._result_cell_generation_policy_sha256()
        self.assertEqual(len(policy), 64)
        chat.close()

    def test_markov_draft_mode_needs_no_sibling_model_runtime(self) -> None:
        target = _Runtime()
        chat = _chat(target, draft_mode="markov", max_new_tokens=4)
        evidence = SimpleNamespace(
            accepted_draft_tokens=2,
            source_body_bytes=100,
            linear_calls=10,
            seconds=1.0,
            state_bytes=456,
            stopped_on_eos=False,
            prompt_token_ids=(11, 12),
            generated_token_ids=(7, 8, 9, 10),
            forward_passes=3,
            rounds=(object(),),
            schema="immer.qwen3.8-rolling-k4-speculative-generation/v1",
            final_state_committed=False,
        )
        generated = SimpleNamespace(token_ids=(7, 8, 9, 10), evidence=evidence)
        decoder = SimpleNamespace(generate_rolling=lambda *args, **kwargs: generated)
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38K4SpeculativeDecoder",
            return_value=decoder,
        ) as decoder_constructor:
            result = chat.handle(Request("chat", "hello"))

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.evidence["draft"]["mode"], "markov")
        self.assertEqual(result.evidence["draft"]["draft_source_body_bytes"], 0)
        self.assertEqual(result.evidence["draft"]["draft_linear_calls"], 0)
        self.assertNotIn("draft_bundle", result.evidence)
        provider = decoder_constructor.call_args.args[1]
        self.assertEqual(provider.metrics().source_body_bytes, 0)
        self.assertTrue(decoder_constructor.call_args.kwargs["adaptive_round_windows"])
        with patch(
            "immer.runtimes.qwen3_8.adapter._digest",
            side_effect=lambda value: value,
        ):
            policy = chat._result_cell_generation_policy_sha256()
        self.assertEqual(
            policy["markov_draft"]["round_window_selector"],
            "markov-prefix-utility/v2",
        )
        chat.close()

    def test_shared_mtp_pager_work_is_split_without_double_counting(self) -> None:
        target = _Runtime()
        target.q4_bank = SimpleNamespace(has=lambda _name: True)
        target.model.pager = SimpleNamespace(q4_bank=target.q4_bank)
        chat = _chat(
            target,
            draft_mode="mtp",
            q4_root="/models/q4-mtp",
            max_new_tokens=4,
        )
        evidence = SimpleNamespace(
            accepted_draft_tokens=2,
            source_body_bytes=100,
            linear_calls=10,
            seconds=1.0,
            state_bytes=456,
            stopped_on_eos=False,
            prompt_token_ids=(11, 12),
            generated_token_ids=(7, 8, 9, 10),
            forward_passes=3,
            rounds=(object(),),
            schema="immer.qwen3.8-rolling-speculative-generation/v2",
            final_state_committed=False,
        )
        generated = SimpleNamespace(token_ids=(7, 8, 9, 10), evidence=evidence)
        decoder = SimpleNamespace(generate_rolling=lambda *args, **kwargs: generated)
        metrics = SimpleNamespace(
            source_body_bytes=20,
            linear_calls=3,
            last_confidence=0.7,
            last_disagreement=0.0,
            last_phrase_confidence=0.0,
            last_phrase_support=0,
            last_phrase_width=0,
        )
        provider = SimpleNamespace(metrics=lambda: metrics, close=lambda: None)
        with (
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen35MtpDraftProvider",
                return_value=provider,
            ),
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38K4SpeculativeDecoder",
                return_value=decoder,
            ),
        ):
            result = chat.handle(Request("chat", "hello"))

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.evidence["generation"]["source_body_bytes"], 80)
        self.assertEqual(result.evidence["generation"]["linear_calls"], 7)
        draft = result.evidence["draft"]
        self.assertEqual(draft["target_source_body_bytes"], 80)
        self.assertEqual(draft["draft_source_body_bytes"], 20)
        self.assertEqual(draft["total_source_body_bytes"], 100)
        self.assertEqual(draft["target_linear_calls"], 7)
        self.assertEqual(draft["draft_linear_calls"], 3)
        self.assertEqual(draft["total_linear_calls"], 10)
        chat.close()

    def test_hybrid_splits_shared_mtp_and_external_qwen35_work(self) -> None:
        target = _Runtime()
        target.q4_bank = SimpleNamespace(has=lambda _name: True)
        target.model.pager = SimpleNamespace(q4_bank=target.q4_bank)
        chat = _chat(
            target,
            draft_bundle_path="/models/Qwen3.5-0.8B",
            draft_mode="hybrid",
            q4_root="/models/q4-mtp",
            max_new_tokens=4,
        )
        evidence = SimpleNamespace(
            accepted_draft_tokens=2,
            source_body_bytes=100,
            linear_calls=10,
            seconds=1.0,
            state_bytes=456,
            stopped_on_eos=False,
            prompt_token_ids=(11, 12),
            generated_token_ids=(7, 8, 9, 10),
            forward_passes=3,
            rounds=(object(),),
            schema="immer.qwen3.8-rolling-speculative-generation/v2",
            final_state_committed=False,
        )
        generated = SimpleNamespace(token_ids=(7, 8, 9, 10), evidence=evidence)
        decoder = SimpleNamespace(generate_rolling=lambda *args, **kwargs: generated)
        metrics = SimpleNamespace(
            source_body_bytes=35,
            linear_calls=5,
            external_source_body_bytes=15,
            external_linear_calls=2,
            mtp_shared_source_body_bytes=20,
            mtp_shared_linear_calls=3,
            last_confidence=0.7,
            last_disagreement=0.0,
            last_phrase_confidence=0.0,
            last_phrase_support=0,
            last_phrase_width=0,
        )
        provider = SimpleNamespace(metrics=lambda: metrics, close=lambda: None)
        with (
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38MarkovMtpDraftProvider",
                return_value=provider,
            ) as hybrid_constructor,
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38K4SpeculativeDecoder",
                return_value=decoder,
            ),
        ):
            result = chat.handle(Request("chat", "hello"))

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.evidence["generation"]["source_body_bytes"], 80)
        self.assertEqual(result.evidence["generation"]["linear_calls"], 7)
        draft = result.evidence["draft"]
        self.assertEqual(draft["target_source_body_bytes"], 80)
        self.assertEqual(draft["shared_source_body_bytes"], 20)
        self.assertEqual(draft["external_source_body_bytes"], 15)
        self.assertEqual(draft["draft_source_body_bytes"], 35)
        self.assertEqual(draft["total_source_body_bytes"], 115)
        self.assertEqual(draft["target_linear_calls"], 7)
        self.assertEqual(draft["shared_linear_calls"], 3)
        self.assertEqual(draft["external_linear_calls"], 2)
        self.assertEqual(draft["total_linear_calls"], 12)
        self.assertTrue(
            callable(hybrid_constructor.call_args.kwargs["qwen35_factory"])
        )
        self.assertEqual(hybrid_constructor.call_args.kwargs["max_new_tokens"], 4)
        chat.close()

    def test_mtp_generation_policy_matches_adaptive_round_execution(self) -> None:
        chat = _chat(
            _Runtime(),
            draft_mode="mtp",
            q4_root="/models/q4-mtp",
            markov_draft_state_path="/state/mtp.json",
            max_new_tokens=4,
        )
        with patch(
            "immer.runtimes.qwen3_8.adapter._digest",
            side_effect=lambda value: value,
        ):
            policy = chat._result_cell_generation_policy_sha256()

        self.assertEqual(policy["decoding"], "greedy-rolling-window-draft-verify")
        self.assertEqual(
            policy["mtp_draft"]["round_window_selector"],
            "markov-prefix-utility/v2",
        )
        self.assertTrue(policy["mtp_draft"]["persistent_calibration"])
        chat.close()

    def test_hybrid_policy_binds_markov_first_mtp_fallback(self) -> None:
        chat = _chat(
            _Runtime(),
            draft_mode="hybrid",
            q4_root="/models/q4-mtp",
            markov_draft_state_path="/state/markov.bin",
            mtp_draft_state_path="/state/mtp.json",
            max_new_tokens=8,
        )
        with patch(
            "immer.runtimes.qwen3_8.adapter._digest",
            side_effect=lambda value: value,
        ):
            policy = chat._result_cell_generation_policy_sha256()

        hybrid = policy["hybrid_draft"]
        self.assertEqual(
            hybrid["selection"],
            "round-wise-markov-first-mtp-fallback/v18",
        )
        self.assertFalse(hybrid["request_provider_lock"])
        self.assertFalse(hybrid["one_way_handoff"])
        self.assertTrue(hybrid["round_reselection"])
        self.assertTrue(hybrid["cross_provider_target_state_sync"])
        self.assertTrue(hybrid["cross_provider_target_feedback"])
        self.assertEqual(
            hybrid["consensus"],
            "markov-prefix+atlas+online-memory/v3",
        )
        self.assertTrue(hybrid["committed_hidden_handoff"])
        self.assertEqual(
            hybrid["markov_confidence"],
            "self-calibrating-dialect-council-lookahead/v9",
        )
        self.assertEqual(
            hybrid["position_specialists"],
            "beta-maturity-fixed-share/v1",
        )
        self.assertEqual(
            hybrid["dialect_specialists"],
            "similarity-beta-maturity/v1",
        )
        self.assertEqual(
            hybrid["dialect_council"],
            "similarity-visits-ricci-top4/v1",
        )
        self.assertEqual(
            hybrid["planning"],
            "target-calibrated-top4-one-step/v2",
        )
        self.assertTrue(hybrid["markov_persistent"])
        self.assertTrue(hybrid["mtp_persistent_calibration"])
        self.assertEqual(
            hybrid["round_window_selector"],
            "markov-prefix-utility/v2",
        )
        chat.close()

    def test_restored_conversation_uses_only_state_independent_drafting(self) -> None:
        runtime = _Runtime()
        hybrid = _chat(runtime, draft_mode="hybrid", q4_root="/q4")
        mtp = _chat(_Runtime(), draft_mode="mtp", q4_root="/q4")

        self.assertEqual(hybrid._draft_mode_for_request({}), "hybrid")
        self.assertEqual(
            hybrid._draft_mode_for_request(
                {"external_drafter_enabled": False}
            ),
            "hybrid",
        )
        self.assertEqual(
            hybrid._draft_mode_for_request({"mtp_enabled": False}),
            "markov",
        )
        self.assertIsNone(mtp._draft_mode_for_request({"mtp_enabled": False}))
        self.assertEqual(
            hybrid._draft_mode_for_request({"restored_prefix_length": 17}),
            "markov",
        )
        self.assertIsNone(mtp._draft_mode_for_request({"restored_prefix_length": 17}))

        prefix = tuple(range(17))
        carry = Qwen35MtpCarry(
            schema="fixture",
            identity=(),
            history=prefix,
            next_position=16,
            state=None,
            last_target_hidden=torch.zeros((1, 1, 1)),
        )
        hybrid._conversation_prefix_token_ids = prefix
        hybrid._conversation_mtp_carry = carry
        self.assertEqual(
            hybrid._draft_mode_for_request({"restored_prefix_length": 17}),
            "markov",
        )
        hybrid._validated_conversation_mtp_carry = carry
        self.assertEqual(
            hybrid._draft_mode_for_request({"restored_prefix_length": 17}),
            "hybrid",
        )
        hybrid._conversation_mtp_carry = None
        hybrid._validated_conversation_mtp_carry = None
        hybrid._conversation_prefix_token_ids = ()
        self.assertEqual(
            hybrid._draft_mode_for_request(
                {
                    "restored_mtp_carry": carry,
                    "restored_mtp_carry_bytes": 128,
                    "restored_prefix_length": 17,
                }
            ),
            "hybrid",
        )
        self.assertEqual(
            mtp._draft_mode_for_request(
                {
                    "restored_mtp_carry": carry,
                    "restored_mtp_carry_bytes": 128,
                    "restored_prefix_length": 17,
                }
            ),
            "mtp",
        )

        hybrid.close()
        mtp.close()

    def test_private_restored_mtp_options_never_reach_direct_generation(
        self,
    ) -> None:
        runtime = _Runtime()
        chat = _chat(
            runtime,
            draft_mode="mtp",
            q4_root="/q4",
            max_new_tokens=1,
        )

        chat._generate_locked(
            runtime,
            (11, 12),
            {
                "eos_token_ids": (IM_END_TOKEN_ID,),
                "head_block_rows": 17,
                "max_new_tokens": 1,
                "prefill_tokenwise": False,
                "restored_mtp_carry": _ANCHOR_MTP_CARRY,
                "restored_mtp_carry_bytes": _ANCHOR_MTP_DESCRIPTOR.bytes,
                "restored_prefix_length": 2,
                "restored_seed_hidden": _ANCHOR_SEED,
            },
        )

        options = runtime.model.calls[0][1]
        self.assertNotIn("restored_mtp_carry", options)
        self.assertNotIn("restored_mtp_carry_bytes", options)
        self.assertEqual(options["restored_prefix_length"], 2)
        chat.close()

        foreign = Qwen35MtpCarry(
            schema=_ANCHOR_MTP_CARRY.schema,
            identity=_ANCHOR_MTP_CARRY.identity,
            history=(21, 22),
            next_position=1,
            state=None,
            last_target_hidden=_ANCHOR_SEED,
        )
        rejected_runtime = _Runtime()
        rejected = _chat(
            rejected_runtime,
            draft_mode="mtp",
            q4_root="/q4",
            max_new_tokens=1,
        )
        with self.assertRaisesRegex(Qwen38ChatError, "restored prompt prefix"):
            rejected._generate_locked(
                rejected_runtime,
                (11, 12),
                {
                    "eos_token_ids": (IM_END_TOKEN_ID,),
                    "head_block_rows": 17,
                    "max_new_tokens": 1,
                    "prefill_tokenwise": False,
                    "restored_mtp_carry": foreign,
                    "restored_mtp_carry_bytes": 128,
                    "restored_prefix_length": 2,
                    "restored_seed_hidden": _ANCHOR_SEED,
                },
            )
        self.assertEqual(rejected_runtime.model.calls, [])
        rejected.close()

    def test_hybrid_accepts_anchor_restore_and_downgrades_without_mtp_carry(
        self,
    ) -> None:
        hybrid = _chat(
            _Runtime(),
            draft_mode="hybrid",
            q4_root="/q4",
            anchor_cache=_anchor_cache(),
        )

        self.assertEqual(
            hybrid._draft_mode_for_request({"restored_prefix_length": 17}),
            "markov",
        )
        hybrid.close()

    def test_hybrid_runtime_identity_names_the_round_wise_feedback_policy(
        self,
    ) -> None:
        runtime = _Runtime()
        chat = _chat(
            runtime,
            draft_mode="hybrid",
            q4_root="/models/q4-mtp",
            markov_draft_state_path="/state/markov.bin",
            mtp_draft_state_path="/state/mtp.json",
            max_new_tokens=8,
        )
        chat._runtime = runtime
        chat._bundle_receipt = _BUNDLE_RECEIPT
        chat._tokenizer_sha256 = _DIGEST
        with patch(
            "immer.runtimes.qwen3_8.adapter._digest",
            side_effect=lambda value: value,
        ):
            identity = chat._draft_window_runtime_identity()

        self.assertEqual(
            identity["provider"]["selection"],
            "round-wise-markov-first-mtp-fallback/v18",
        )
        chat.close()

    def test_hybrid_rejects_a_loaded_q4_bank_without_mtp_matrices(self) -> None:
        runtime = _Runtime()
        runtime.q4_bank = SimpleNamespace(has=lambda _name: False)
        runtime.model.pager = SimpleNamespace(q4_bank=runtime.q4_bank)
        result = _chat(
            runtime,
            draft_mode="hybrid",
            q4_root="/models/q4-v2",
            max_new_tokens=8,
        ).handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.UNAVAILABLE)
        self.assertIn("lacks the embedded MTP matrices", result.reason)
        self.assertEqual(runtime.close_calls, 1)

    def test_short_generation_policy_records_plain_greedy_draft_fallback(self) -> None:
        runtime = _Runtime()
        runtime.model.generated = (7,)
        chat = _chat(runtime, draft_mode="markov", max_new_tokens=1)
        with patch(
            "immer.runtimes.qwen3_8.adapter._digest",
            side_effect=lambda value: value,
        ):
            policy = chat._result_cell_generation_policy_sha256()

        self.assertEqual(policy["decoding"], "greedy")
        self.assertEqual(
            policy["draft_fallback"],
            {
                "configured_mode": "markov",
                "reason": "max-new-tokens-below-2",
            },
        )
        result = chat.handle(Request("chat", "hello"))
        self.assertTrue(result.ok, result.reason)
        self.assertNotIn("draft", result.evidence)
        chat.close()

    def test_failed_k4_fast_request_cannot_leak_counters_into_next_call(self) -> None:
        target = _Runtime()
        target.fast_mlp_mount = _FastMount(cumulative=(0, 7, 27))
        target.fast_mlp_receipt = dict(_FAST_RECEIPT)
        chat = _chat(
            target,
            draft_bundle_path="draft.causal",
            max_new_tokens=4,
            fast_mlp_root="/artifacts/fast-mlp",
            fast_mlp_active_layers=(0, 9),
        )
        chat._draft_runtime = SimpleNamespace(
            tokenizer_sha256=_DIGEST,
            model=SimpleNamespace(config=SimpleNamespace(vocab_size=300_000)),
            bundle_receipt=_BUNDLE_RECEIPT,
            close=lambda: None,
        )
        provider = SimpleNamespace(
            metrics=lambda: SimpleNamespace(source_body_bytes=12, linear_calls=3),
            close=lambda: None,
        )
        evidence = SimpleNamespace(
            accepted_draft_tokens=4,
            source_body_bytes=100,
            linear_calls=10,
            seconds=1.25,
            state_bytes=456,
            stopped_on_eos=False,
            prompt_token_ids=(11, 12),
            generated_token_ids=(7, 8, 9, 10),
            forward_passes=2,
            rounds=(object(),),
            schema="immer.qwen3.8-rolling-k4-speculative-generation/v1",
            final_state_committed=False,
        )
        generated = SimpleNamespace(token_ids=(7, 8, 9, 10), evidence=evidence)
        decoder = SimpleNamespace(
            generate_rolling=Mock(side_effect=[RuntimeError("first failed"), generated])
        )
        with (
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen35K4DraftProvider",
                return_value=provider,
            ),
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38K4SpeculativeDecoder",
                return_value=decoder,
            ),
        ):
            failed = chat.handle(Request("chat", "hello"))
            succeeded = chat.handle(Request("chat", "hello"))

        self.assertIs(failed.status, ExecutionStatus.ERROR)
        self.assertNotIn("request", failed.evidence["fast_mlp"])
        self.assertTrue(succeeded.ok, succeeded.reason)
        self.assertEqual(
            succeeded.evidence["fast_mlp"]["request"]["aux_source_body_bytes"],
            20,
        )
        chat.close()

    def test_opt_in_result_cell_binding_is_runtime_derived_and_chargeable(
        self,
    ) -> None:
        runtime = _Runtime()
        code_revision = "f" * 40
        chat = _chat(
            runtime,
            system_prompt=" local system ",
            result_cell_code_revision=code_revision,
        )
        result = chat.handle(Request("chat", "  hello  "))

        self.assertIs(result.status, ExecutionStatus.OK)
        document = result.evidence["result_cell_binding_receipt"]
        body = document["body"]
        rendered = Qwen38Tokenizer.render_no_thinking_prompt(
            "local system",
            "hello",
        )
        pin = ModelPin(
            repo_id=result.evidence["model"],
            revision=result.evidence["revision"],
            bundle_fingerprint=_BUNDLE_RECEIPT["layout_fingerprint"],
            bundle_manifest_sha256=_BUNDLE_RECEIPT["manifest_sha256"],
            code_revision=code_revision,
        )
        binding = ResultCellBinding(
            model_pin=pin,
            tokenizer_sha256=_DIGEST,
            question_sha256=hashlib.sha256(b"hello").hexdigest(),
            rendered_prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
            rendered_prompt_token_sha256=prompt_token_sha256((11, 12)),
            system_prompt_sha256=hashlib.sha256(b"local system").hexdigest(),
            generation_policy_sha256=chat._result_cell_generation_policy_sha256(),
        )
        self.assertEqual(body["binding_sha256"], binding.sha256)
        self.assertEqual(document, qwen_result_binding_evidence(binding))
        self.assertNotIn("hello", json.dumps(document, sort_keys=True))
        cold = attach_cold_qwen_generation_receipt(result, binding=binding)
        final = Result(
            ExecutionStatus.OK,
            "qwen-fertig-chat",
            output=result.output,
            evidence={"route": "qwen_verified"},
        )
        cell = ResultCell.from_cold(
            binding=binding,
            cold_qwen_result=cold,
            cold_final_result=final,
            cold_fertig_judgment={"status": "verified"},
            cold_fertig_status="verified",
            evaluator_quality_contract_sha256="9" * 64,
        )
        self.assertEqual(cell.teacher_forward_count, 3)
        self.assertEqual(cell.cold_qwen_result, cold)

    def test_result_cell_code_revision_is_full_or_disabled(self) -> None:
        _chat(_Runtime())
        with self.assertRaisesRegex(ValueError, "full lowercase"):
            _chat(_Runtime(), result_cell_code_revision="short")

    def test_cli_and_adapter_build_the_same_semantic_replay_authority(self) -> None:
        from immer.runtimes.qwen3_8.output_semantics import (
            QWEN_SEMANTIC_REPLAY_EVIDENCE_KEY,
            QwenSemanticReplayKey,
            parse_semantic_replay_receipt,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            q4 = root / "q4"
            q4.mkdir()
            manifest = q4 / "manifest.json"
            manifest.write_bytes(b"q4-manifest")
            tokenizer = root / "tokenizer.json"
            tokenizer.write_bytes(b"tokenizer")
            q4_sha = hashlib.sha256(b"q4-manifest").hexdigest()
            runtime = _Runtime()
            chat = _chat(
                runtime,
                system_prompt=" local system ",
                q4_root=q4,
                mlp_page_state_path=root / "pages.json",
                mlp_page_route_width=192,
                result_cell_code_revision="f" * 40,
            )
            with patch(
                "immer.runtimes.qwen3_8.adapter._file_sha256",
                return_value=q4_sha,
            ):
                result = chat.handle(Request("chat", "hello"))
            semantic_document = result.evidence[QWEN_SEMANTIC_REPLAY_EVIDENCE_KEY]
            adapter_semantics, adapter_key = parse_semantic_replay_receipt(
                semantic_document
            )
            args = SimpleNamespace(
                compute_dtype="auto",
                max_context_tokens=16,
                max_new_tokens=3,
                max_prompt_tokens=8,
                mlp_page_width=192,
            )
            with patch(
                "immer.cli._path_sha256",
                side_effect=lambda path: q4_sha if path == manifest else _DIGEST,
            ):
                cli_semantics = _qwen38_output_semantics(
                    args,
                    tokenizer_path=tokenizer,
                    q4_root=q4,
                    mlp_page_state_path=root / "another-pages.json",
                )
            assert cli_semantics is not None
            rendered = Qwen38Tokenizer.render_no_thinking_prompt(
                "local system",
                "hello",
            )
            cli_key = QwenSemanticReplayKey(
                output_semantics_sha256=cli_semantics.sha256,
                question_sha256=hashlib.sha256(b"hello").hexdigest(),
                rendered_prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                rendered_prompt_token_sha256=prompt_token_sha256((11, 12)),
                system_prompt_sha256=hashlib.sha256(b"local system").hexdigest(),
            )

        self.assertEqual(adapter_semantics, cli_semantics)
        self.assertEqual(adapter_key, cli_key)
        chat.close()

    def test_delta_head_runtime_does_not_publish_baseline_semantic_replay(
        self,
    ) -> None:
        chat = _chat(
            _Runtime(),
            q4_root="/q4",
            delta_head_state_path="/state/delta-head.json",
            result_cell_code_revision=_DIGEST,
        )

        self.assertIsNone(
            chat._result_cell_semantic_replay_receipt(
                question="hello",
                rendered_prompt="rendered hello",
                prompt_ids=(11, 12),
            )
        )
        chat.close()

    def test_authenticated_exact_anchor_bypasses_prefill_with_identical_output(
        self,
    ) -> None:
        baseline = _chat(_Runtime()).handle(Request("chat", "hello"))
        model = _AnchorModel()
        cache = _anchor_cache()
        cached = _chat(
            _Runtime(model=model),
            anchor_cache=cache,
        ).handle(Request("chat", "hello"))

        self.assertIs(baseline.status, ExecutionStatus.OK)
        self.assertIs(cached.status, ExecutionStatus.OK)
        self.assertEqual(cached.output, baseline.output)
        cache.restore_deepest.assert_called_once_with(
            model,
            (11, 12),
            restore_mtp_carry=False,
            tokenizer_sha256=_DIGEST,
        )
        options = model.calls[0][1]
        self.assertEqual(options["restored_prefix_length"], 2)
        self.assertIs(
            options["restored_seed_hidden"],
            _RESTORED_ANCHOR.seed_hidden,
        )
        anchor = cached.evidence["anchor_cache"]
        self.assertEqual(anchor["status"], "hit")
        self.assertTrue(anchor["exact_prefix"])
        self.assertEqual(anchor["prefix_tokens"], 2)
        self.assertEqual(anchor["suffix_tokens"], 0)
        self.assertEqual(anchor["snapshot_bytes_read"], 70)
        self.assertEqual(anchor["forward_passes_baseline"], 2)
        self.assertEqual(anchor["forward_passes_executed"], 1)
        self.assertEqual(anchor["forward_passes_saved"], 1)
        self.assertEqual(anchor["prefill_weight_sweeps_saved"], 1)
        self.assertEqual(anchor["checkpoint_read_sweeps_saved"], 1)
        self.assertEqual(anchor["prompt_token_layer_evaluations_saved"], 128)
        self.assertEqual(anchor["checkpoint_source_body_bytes_read"], 600)
        self.assertEqual(anchor["checkpoint_linear_calls_executed"], 66)
        self.assertEqual(anchor["anchor"], _ANCHOR_RECEIPT.to_document())
        self.assertEqual(model.reset_calls, [True])

    def test_authenticated_anchor_mtp_carry_is_request_local_and_keeps_mode(
        self,
    ) -> None:
        model = _AnchorModel()
        q4_bank = SimpleNamespace(has=lambda _name: True)
        model.pager = SimpleNamespace(q4_bank=q4_bank)
        runtime = _Runtime(model=model)
        runtime.q4_bank = q4_bank
        cache = _anchor_cache(restored=_RESTORED_ANCHOR_MTP)
        provider_metrics = SimpleNamespace(
            source_body_bytes=0,
            linear_calls=0,
            last_confidence=0.0,
            last_disagreement=0.0,
            last_phrase_confidence=0.0,
            last_phrase_support=0,
            last_phrase_width=0,
        )
        provider = SimpleNamespace(
            close=Mock(),
            export_mtp_carry=Mock(return_value=_ANCHOR_MTP_CARRY),
            metrics=Mock(return_value=provider_metrics),
        )
        row = SimpleNamespace(
            accepted_prefix_length=0,
            emitted_token_ids=(7, 8),
            forward_passes=1,
            proposed_token_ids=(),
            provider_proposed_token_ids=(),
            round_index=0,
            round_policy=None,
            target_token_ids=(7, 8),
            window_size=3,
        )
        rolling = SimpleNamespace(
            token_ids=(7, 8),
            evidence=SimpleNamespace(
                accepted_draft_tokens=0,
                adaptive_windows=False,
                final_state_committed=False,
                forward_passes=1,
                generated_token_ids=(7, 8),
                linear_calls=0,
                prefill_forward_passes=0,
                prompt_token_ids=(11, 12),
                rounds=(row,),
                schema="fixture.anchor-mtp-generation/v1",
                seconds=0.1,
                source_body_bytes=0,
                state_bytes=_ANCHOR_RECEIPT.state_bytes,
                stopped_on_eos=False,
                used_window_sizes=(3,),
                window_size=3,
            ),
        )
        decoder = SimpleNamespace(generate_rolling=Mock(return_value=rolling))
        chat = _chat(
            runtime,
            anchor_cache=cache,
            draft_mode="mtp",
            q4_root="/q4",
        )

        with (
            patch(
                "immer.runtimes.qwen3_8.adapter.qwen35_mtp_carry_identity",
                return_value=_ANCHOR_MTP_CARRY.identity,
            ),
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen35MtpDraftProvider",
                return_value=provider,
            ) as provider_constructor,
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38K4SpeculativeDecoder",
                return_value=decoder,
            ),
        ):
            result = chat.handle(Request("chat", "hello"))

        self.assertTrue(result.ok, result.reason)
        cache.restore_deepest.assert_called_once_with(
            model,
            (11, 12),
            restore_mtp_carry=True,
            tokenizer_sha256=_DIGEST,
            expected_mtp_identity=_ANCHOR_MTP_CARRY.identity,
        )
        self.assertIs(
            provider_constructor.call_args.kwargs["initial_carry"],
            _ANCHOR_MTP_CARRY,
        )
        provider.export_mtp_carry.assert_called_once_with((11, 12))
        self.assertEqual(result.evidence["draft"]["configured_mode"], "mtp")
        self.assertEqual(result.evidence["draft"]["mode"], "mtp")
        self.assertFalse(
            result.evidence["draft"]["state_reuse_provider_downgrade"]
        )
        self.assertEqual(
            result.evidence["conversation"]["mtp_carry_status"],
            "restored-anchor",
        )
        self.assertEqual(
            result.evidence["conversation"]["mtp_carry_bytes"],
            _ANCHOR_MTP_DESCRIPTOR.bytes,
        )
        self.assertEqual(
            result.evidence["conversation"]["mtp_carry_reused_tokens"],
            2,
        )
        anchor = result.evidence["anchor_cache"]
        self.assertEqual(anchor["mtp_carry_status"], "restored")
        self.assertEqual(anchor["mtp_carry_bytes"], _ANCHOR_MTP_DESCRIPTOR.bytes)
        self.assertNotIn("restored_mtp_carry", result.evidence)
        self.assertIsNone(chat._conversation_mtp_carry)
        chat.close()

    def test_template_anchor_charge_exports_exact_mtp_carry_without_target_work(
        self,
    ) -> None:
        cache = object.__new__(SemanticStateAnchorCache)
        cache.root = Path("/fixture-cache")
        cache.store = Mock(return_value=_ANCHOR_MTP_RECEIPT)
        model = _Model()
        hidden = torch.tensor([[[1.0], [3.0]]])
        forwards = (object(), object())
        model.prefill = Mock(return_value=(hidden, forwards))
        runtime = _Runtime(model=model)
        provider = SimpleNamespace(
            begin_request_state=Mock(),
            close=Mock(),
            export_mtp_carry=Mock(return_value=_ANCHOR_MTP_CARRY),
            metrics=Mock(
                return_value=SimpleNamespace(
                    linear_calls=4,
                    source_body_bytes=33,
                )
            ),
            observe_final=Mock(),
        )
        chat = _chat(
            runtime,
            anchor_cache=cache,
            draft_mode="mtp",
            q4_root="/q4",
        )

        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen35MtpDraftProvider",
            return_value=provider,
        ):
            receipt = chat._charge_template_anchor(runtime, (11, 12, 13))

        model.prefill.assert_called_once_with(
            [(11, 12)],
            reset=True,
            tokenwise=False,
        )
        provider.begin_request_state.assert_called_once_with((11, 12), hidden)
        provider.observe_final.assert_called_once_with((11, 12))
        provider.export_mtp_carry.assert_called_once_with((11, 12))
        provider.close.assert_called_once_with()
        store_options = cache.store.call_args.kwargs
        self.assertIs(store_options["mtp_carry"], _ANCHOR_MTP_CARRY)
        self.assertEqual(store_options["tokenizer_sha256"], _DIGEST)
        self.assertTrue(torch.equal(store_options["seed_hidden"], hidden[:, -1:]))
        self.assertEqual(receipt["status"], "stored")
        self.assertEqual(receipt["target_forwards"], len(forwards))
        self.assertEqual(receipt["mtp_carry_status"], "stored")
        self.assertEqual(receipt["mtp_carry_bytes"], _ANCHOR_MTP_DESCRIPTOR.bytes)
        self.assertEqual(receipt["mtp_source_body_bytes"], 33)
        self.assertEqual(receipt["mtp_linear_calls"], 4)
        self.assertGreaterEqual(receipt["mtp_seconds"], 0.0)
        self.assertEqual(model.reset_calls, [True])

    def test_template_anchor_mtp_failure_stores_legacy_anchor_and_keeps_target(
        self,
    ) -> None:
        cache = object.__new__(SemanticStateAnchorCache)
        cache.root = Path("/fixture-cache")
        cache.store = Mock(return_value=_ANCHOR_RECEIPT)
        model = _Model()
        hidden = torch.tensor([[[1.0], [3.0]]])
        model.prefill = Mock(return_value=(hidden, (object(),)))
        runtime = _Runtime(model=model)
        provider = SimpleNamespace(
            begin_request_state=Mock(side_effect=RuntimeError("optional MTP failed")),
            close=Mock(),
            metrics=Mock(
                return_value=SimpleNamespace(
                    linear_calls=2,
                    source_body_bytes=19,
                )
            ),
        )
        chat = _chat(
            runtime,
            anchor_cache=cache,
            draft_mode="hybrid",
            q4_root="/q4",
        )

        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen35MtpDraftProvider",
            return_value=provider,
        ):
            receipt = chat._charge_template_anchor(runtime, (11, 12, 13))

        self.assertEqual(receipt["status"], "stored")
        self.assertEqual(receipt["mtp_carry_status"], "charge-error")
        self.assertEqual(receipt["mtp_carry_bytes"], 0)
        self.assertEqual(receipt["mtp_source_body_bytes"], 19)
        self.assertEqual(receipt["mtp_linear_calls"], 2)
        self.assertIn("RuntimeError", receipt["mtp_error"])
        self.assertIsNone(cache.store.call_args.kwargs["mtp_carry"])
        provider.close.assert_called_once_with()
        self.assertEqual(model.reset_calls, [True])

    def test_anchor_restore_failure_never_falls_back_and_resets_state(self) -> None:
        model = _AnchorModel()
        result = _chat(
            _Runtime(model=model),
            anchor_cache=_anchor_cache(failure=RuntimeError("tampered anchor")),
        ).handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.ERROR)
        self.assertIn("tampered anchor", result.reason)
        self.assertEqual(model.calls, [])
        self.assertEqual(model.reset_calls, [True])
        self.assertEqual(model.next_position, 0)
        self.assertEqual(model.state_bytes, 0)

    def test_anchor_seed_must_match_the_sealed_receipt(self) -> None:
        model = _AnchorModel()
        tampered = RestoredAnchor(
            anchor=_ANCHOR_RECEIPT,
            query_length=2,
            exact_prefix=True,
            seed_hidden=_ANCHOR_SEED + 1.0,
        )
        result = _chat(
            _Runtime(model=model),
            anchor_cache=_anchor_cache(restored=tampered),
        ).handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.ERROR)
        self.assertIn("seed differs from the sealed anchor", result.reason)
        self.assertEqual(model.calls, [])
        self.assertEqual(model.reset_calls, [True])

    def test_anchor_forward_savings_reject_underreported_execution(self) -> None:
        model = _UnderreportedAnchorModel()
        result = _chat(
            _Runtime(model=model),
            anchor_cache=_anchor_cache(),
        ).handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.ERROR)
        self.assertIn("forward count is inconsistent", result.reason)
        self.assertEqual(model.reset_calls, [True])

    def test_anchor_accounting_accepts_speculative_replay_forwards(self) -> None:
        from immer.runtimes.qwen3_8.adapter import _anchor_hit_evidence

        evidence = _anchor_hit_evidence(
            _RESTORED_ANCHOR,
            prompt_tokens=2,
            generation={
                "forward_passes": 3,
                "forward_contract": {
                    "accepted_draft_tokens": 2,
                    "emitted_tokens": 5,
                    "prefill_forward_passes": 0,
                    "round_forward_passes": 3,
                    "rounds": 3,
                    "schema": "immer.qwen3.8-rolling-forward-contract/v1",
                },
                "generated_tokens": 5,
                "linear_calls": 990,
                "source_body_bytes": 123_456,
            },
            restore_seconds=0.25,
            n_layers=64,
            final_state_committed=False,
        )

        self.assertEqual(evidence["forward_passes_executed"], 3)
        self.assertEqual(evidence["forward_passes_saved"], 1)
        self.assertEqual(evidence["forward_passes_baseline"], 4)

        tampered = {
            "forward_passes": 2,
            "forward_contract": {
                "accepted_draft_tokens": 2,
                "emitted_tokens": 5,
                "prefill_forward_passes": 0,
                "round_forward_passes": 3,
                "rounds": 3,
                "schema": "immer.qwen3.8-rolling-forward-contract/v1",
            },
            "generated_tokens": 5,
            "linear_calls": 990,
            "source_body_bytes": 123_456,
        }
        with self.assertRaisesRegex(
            Qwen38ChatError,
            "rolling forward contract",
        ):
            _anchor_hit_evidence(
                _RESTORED_ANCHOR,
                prompt_tokens=2,
                generation=tampered,
                restore_seconds=0.25,
                n_layers=64,
                final_state_committed=False,
            )

    def test_anchor_evidence_distinguishes_an_ignored_mtp_sidecar(self) -> None:
        from immer.runtimes.qwen3_8.adapter import _anchor_hit_evidence

        ignored = RestoredAnchor(
            anchor=_ANCHOR_MTP_RECEIPT,
            query_length=2,
            exact_prefix=True,
            seed_hidden=_ANCHOR_SEED,
            mtp_carry_ignored=True,
        )
        evidence = _anchor_hit_evidence(
            ignored,
            prompt_tokens=2,
            generation={
                "forward_passes": 0,
                "generated_tokens": 1,
                "linear_calls": 0,
                "source_body_bytes": 0,
            },
            restore_seconds=0.1,
            n_layers=64,
            final_state_committed=False,
        )

        self.assertEqual(evidence["mtp_carry_status"], "ignored-non-mtp")
        self.assertEqual(evidence["mtp_carry_bytes"], 0)

    def test_anchor_miss_preserves_the_existing_generation_path(self) -> None:
        model = _AnchorModel()
        result = _chat(
            _Runtime(model=model),
            anchor_cache=_anchor_cache(miss=True),
        ).handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.OK)
        self.assertEqual(result.output, "local answer")
        self.assertEqual(
            model.calls[0],
            (
                [[11, 12]],
                {
                    "max_new_tokens": 3,
                    "prefill_tokenwise": False,
                    "eos_token_ids": (248046, 248044),
                    "head_block_rows": 17,
                    "retain_final_state": False,
                },
            ),
        )
        self.assertEqual(result.evidence["anchor_cache"]["status"], "miss")
        self.assertEqual(model.reset_calls, [True])

    def test_anchor_cache_rejects_duck_typed_restore_provider(self) -> None:
        with self.assertRaisesRegex(TypeError, "SemanticStateAnchorCache"):
            _chat(
                _Runtime(),
                anchor_cache=SimpleNamespace(restore_deepest=lambda *_args: None),
            )

    def test_rejections_do_not_open_the_runtime(self) -> None:
        calls = 0

        def factory():
            nonlocal calls
            calls += 1
            return _Runtime()

        chat = _chat(_Runtime(), runtime_factory=factory)
        unsupported = chat.handle(Request("exact_math", "1+1"))
        blank = chat.handle(Request("chat", " \n"))

        self.assertIs(unsupported.status, ExecutionStatus.REJECTED)
        self.assertIs(blank.status, ExecutionStatus.REJECTED)
        self.assertEqual(calls, 0)

    def test_missing_bundle_is_unavailable_and_never_falls_back(self) -> None:
        chat = Qwen38CausalChat(
            Path("definitely-missing-qwen-bundle"),
            Path("definitely-missing-tokenizer.json"),
            max_prompt_tokens=8,
            max_new_tokens=3,
            max_context_tokens=16,
        )

        result = chat.handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.UNAVAILABLE)
        self.assertIn("causal bundle directory is missing", result.reason)
        self.assertFalse(chat.loaded)
        self.assertEqual(result.evidence["model"], "Qwen/Qwen3.8-27B")

    def test_bundle_and_tokenizer_symlinks_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_target = root / "bundle-target"
            bundle_target.mkdir()
            bundle_link = root / "bundle-link"
            bundle_link.symlink_to(bundle_target, target_is_directory=True)
            tokenizer = root / "tokenizer.json"
            tokenizer.write_text("{}", encoding="utf-8")

            linked_bundle = Qwen38CausalChat(
                bundle_link,
                tokenizer,
                max_prompt_tokens=8,
                max_new_tokens=3,
                max_context_tokens=16,
            ).handle(Request("chat", "hello"))
            self.assertIs(linked_bundle.status, ExecutionStatus.UNAVAILABLE)
            self.assertIn("non-symlink directory", linked_bundle.reason)

            plain_bundle = root / "plain-bundle"
            plain_bundle.mkdir()
            tokenizer_link = root / "tokenizer-link.json"
            tokenizer_link.symlink_to(tokenizer)
            linked_tokenizer = Qwen38CausalChat(
                plain_bundle,
                tokenizer_link,
                max_prompt_tokens=8,
                max_new_tokens=3,
                max_context_tokens=16,
            ).handle(Request("chat", "hello"))
            self.assertIs(linked_tokenizer.status, ExecutionStatus.UNAVAILABLE)
            self.assertIn("non-symlink regular file", linked_tokenizer.reason)

    def test_invalid_injected_receipt_is_unavailable_and_runtime_is_closed(
        self,
    ) -> None:
        runtime = _Runtime()
        runtime.bundle_receipt = {**_BUNDLE_RECEIPT, "kind": "unverified"}
        chat = _chat(runtime)

        result = chat.handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.UNAVAILABLE)
        self.assertIn("verified complete Qwen causal bundle", result.reason)
        self.assertEqual(runtime.close_calls, 1)

    def test_generation_failure_is_contained_and_state_is_released(self) -> None:
        model = _Model(generation_error=RuntimeError("decoder failed"))
        runtime = _Runtime(model=model)
        chat = _chat(runtime)

        result = chat.handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.ERROR)
        self.assertIn("decoder failed", result.reason)
        self.assertEqual(model.reset_calls, [True])
        self.assertEqual(runtime.close_calls, 0)

    def test_cleanup_failure_retires_runtime_fail_closed(self) -> None:
        model = _Model(cleanup_error=RuntimeError("cannot release"))
        runtime = _Runtime(model=model)
        chat = _chat(runtime)

        first = chat.handle(Request("chat", "hello"))
        second = chat.handle(Request("chat", "again"))

        self.assertIs(first.status, ExecutionStatus.ERROR)
        self.assertEqual(first.reason, "Qwen3.8 state cleanup failed")
        self.assertIs(second.status, ExecutionStatus.UNAVAILABLE)
        self.assertEqual(runtime.close_calls, 1)

    def test_context_manager_closes_once_and_closed_component_is_unavailable(
        self,
    ) -> None:
        runtime = _Runtime()
        with _chat(runtime) as chat:
            self.assertTrue(chat.handle(Request("chat", "hello")).ok)
        chat.close()

        self.assertTrue(chat.closed)
        self.assertEqual(runtime.close_calls, 1)
        after = chat.handle(Request("chat", "again"))
        self.assertIs(after.status, ExecutionStatus.UNAVAILABLE)
        with self.assertRaises(Qwen38ChatError):
            chat.__enter__()

    def test_composition_constructs_optional_qwen_general_chat_lazily(self) -> None:
        runtime = _Runtime()
        component = _chat(runtime)
        exact_a = _ExactBackend()
        fertig = FertigSolver()
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
            return_value=component,
        ) as constructor:
            root = compose_runtime(
                s3_arithmetic=exact_a,
                fertig=fertig,
                qwen38_causal_bundle="local.causal",
                qwen38_tokenizer="tokenizer.json",
                qwen38_options={
                    "max_prompt_tokens": 8,
                    "max_new_tokens": 3,
                    "max_context_tokens": 16,
                },
            )

        self.assertIsInstance(root.general_chat, QwenFertigChat)
        self.assertIs(root.general_chat.qwen, component)
        self.assertIs(root.general_chat.fertig, root.exact_math.fertig)
        self.assertFalse(root.general_chat.loaded)
        constructor.assert_called_once()
        self.assertEqual(root.runtime.registry.capabilities(), ("chat", "exact_math"))
        self.assertTrue(root.dispatch("chat", "hello").ok)
        root.general_chat.close()

    def test_composition_can_explicitly_construct_raw_qwen_for_low_level_use(
        self,
    ) -> None:
        component = _chat(_Runtime())
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
            return_value=component,
        ):
            root = compose_runtime(
                s3_arithmetic=_ExactBackend(),
                fertig=_ExactBackend(),
                qwen38_causal_bundle="local.causal",
                qwen38_tokenizer="tokenizer.json",
                qwen38_raw_chat=True,
            )

        self.assertIs(root.general_chat, component)
        self.assertIs(root.runtime.registry.get("chat"), component)
        root.general_chat.close()

    def test_composition_rejects_ambiguous_qwen_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "configured together"):
            CompositionRoot.build(
                s3_arithmetic=_ExactBackend(),
                fertig=_ExactBackend(),
                qwen38_causal_bundle="local.causal",
            )
        with self.assertRaisesRegex(ValueError, "either general_chat"):
            CompositionRoot.build(
                s3_arithmetic=_ExactBackend(),
                fertig=_ExactBackend(),
                general_chat=_chat(_Runtime()),
                qwen38_causal_bundle="local.causal",
                qwen38_tokenizer="tokenizer.json",
            )
        with self.assertRaisesRegex(ValueError, "cannot override"):
            CompositionRoot.build(
                s3_arithmetic=_ExactBackend(),
                fertig=_ExactBackend(),
                qwen38_causal_bundle="local.causal",
                qwen38_tokenizer="tokenizer.json",
                qwen38_options={"runtime_factory": lambda: _Runtime()},
            )

    def test_cli_wires_explicit_local_paths_and_closes_component(self) -> None:
        qwen = _chat(_Runtime())
        output = io.StringIO()
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
            return_value=qwen,
        ) as constructor:
            with redirect_stdout(output):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--qwen38-causal-bundle",
                        "/models/qwen.causal",
                        "--qwen38-tokenizer",
                        "/models/tokenizer.json",
                        "--raw-qwen",
                        "--output",
                        "json",
                        "--max-new-tokens",
                        "4",
                    ]
                )

        self.assertEqual(code, 0)
        self.assertTrue(qwen.closed)
        self.assertEqual(
            constructor.call_args.args,
            ("/models/qwen.causal", "/models/tokenizer.json"),
        )
        options = constructor.call_args.kwargs
        self.assertEqual(options["max_new_tokens"], 4)
        self.assertIsNone(options["text_snapshot_sink"])
        self.assertEqual(options["source_budget_mb"], 4194304)
        self.assertEqual(options["draft_source_budget_mb"], 1048576)
        self.assertEqual(options["max_resident_bytes"], 192 * 1024**2)
        self.assertEqual(options["draft_max_resident_bytes"], 64 * 1024**2)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["component"], "qwen3.8.causal-chat")
        self.assertEqual(payload["output"], "local answer")

    def test_cli_streams_text_snapshots_without_reprinting_the_final(self) -> None:
        sink: dict[str, object] = {}
        qwen = Mock()

        def construct(*_args, **options):
            sink["callback"] = options["text_snapshot_sink"]
            return qwen

        def handle(_request):
            callback = sink["callback"]
            assert callable(callback)
            callback("Hello")
            callback("Hello world")
            return Result(
                ExecutionStatus.OK,
                "qwen3.8.causal-chat",
                output="Hello world",
            )

        qwen.handle.side_effect = handle
        output = io.StringIO()
        with (
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                side_effect=construct,
            ),
            redirect_stdout(output),
        ):
            code = main(
                [
                    "chat",
                    "hello",
                    "--qwen38-causal-bundle",
                    "/models/qwen.causal",
                    "--qwen38-tokenizer",
                    "/models/tokenizer.json",
                    "--raw-qwen",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(output.getvalue(), "Hello world\n")
        qwen.close.assert_called_once_with()

    def test_cli_wrapped_stream_exposes_progress_but_only_final_text(self) -> None:
        sink: dict[str, object] = {}
        qwen = Mock()
        wrapped = Mock()

        class _Tty(io.StringIO):
            def isatty(self) -> bool:
                return True

        def construct(*_args, **options):
            sink["callback"] = options["text_snapshot_sink"]
            return qwen

        def handle(_request):
            callback = sink["callback"]
            assert callable(callback)
            callback("provisional")
            return Result(
                ExecutionStatus.OK,
                "qwen3.8.fertig-chat",
                output="final exact answer",
            )

        wrapped.handle.side_effect = handle
        wrapped.close.side_effect = qwen.close
        output = io.StringIO()
        progress = _Tty()
        with (
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                side_effect=construct,
            ),
            patch(
                "immer.cognition.qwen_fertig_chat.QwenFertigChat",
                return_value=wrapped,
            ),
            redirect_stdout(output),
            redirect_stderr(progress),
        ):
            code = main(
                [
                    "chat",
                    "hello",
                    "--qwen38-causal-bundle",
                    "/models/qwen.causal",
                    "--qwen38-tokenizer",
                    "/models/tokenizer.json",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(output.getvalue(), "final exact answer\n")
        self.assertIn("Qwen: 1 tokens", progress.getvalue())
        self.assertNotIn("provisional", progress.getvalue())
        wrapped.close.assert_called_once_with()

    def test_cli_resolves_local_stack_and_wraps_fertig_by_default(self) -> None:
        qwen = _chat(_Runtime())
        output = io.StringIO()
        environment = {
            "IMMER_QWEN38_ROOT": "/models/Qwen3.8-27B",
            "IMMER_QWEN38_Q4": "/models/Qwen3.8-27B/causal/q4-base-v2",
            "IMMER_QWEN38_FAST_MLP": "/state/qwen-q4-fast-mlp-all64-v1",
        }
        with (
            patch.dict("os.environ", environment, clear=True),
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                return_value=qwen,
            ) as constructor,
            patch(
                "immer.cognition.qwen_fertig_chat.QwenFertigChat",
                side_effect=lambda raw, _fertig, **_options: raw,
            ) as wrapper,
            redirect_stdout(output),
        ):
            code = main(["chat", "hello"])

        self.assertEqual(code, 0)
        self.assertTrue(qwen.closed)
        self.assertEqual(
            constructor.call_args.args,
            (
                "/models/Qwen3.8-27B",
                "/models/Qwen3.8-27B/tokenizer.json",
            ),
        )
        options = constructor.call_args.kwargs
        self.assertEqual(
            options["q4_root"],
            "/models/Qwen3.8-27B/causal/q4-base-v2",
        )
        self.assertEqual(
            options["fast_mlp_root"],
            "/state/qwen-q4-fast-mlp-all64-v1",
        )
        self.assertEqual(
            options["fast_mlp_active_layers"],
            (*range(18), *range(55, 64)),
        )
        self.assertEqual(options["fast_mlp_selected_block_count"], 32)
        wrapper.assert_called_once()
        self.assertIs(wrapper.call_args.args[0], qwen)
        self.assertIsInstance(wrapper.call_args.args[1], FertigSolver)
        self.assertIsNone(wrapper.call_args.kwargs["ooe_hook"])

    def test_cli_mounts_an_explicit_verified_ooe_warm_bank(self) -> None:
        qwen = _chat(_Runtime())
        hook = object()
        mount = SimpleNamespace(hook=hook)
        with (
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                return_value=qwen,
            ),
            patch(
                "immer.runtimes.ooe.qwen_warm_bank.open_verified_qwen_warm_bank",
                return_value=mount,
            ) as opener,
            patch(
                "immer.cognition.qwen_fertig_chat.QwenFertigChat",
                side_effect=lambda raw, _fertig, **_options: raw,
            ) as wrapper,
            redirect_stdout(io.StringIO()),
        ):
            code = main(
                [
                    "chat",
                    "hello",
                    "--qwen38-causal-bundle",
                    "/models/qwen.causal",
                    "--qwen38-tokenizer",
                    "/models/tokenizer.json",
                    "--ooe-warm-root",
                    "/state/qwen-warm",
                ]
            )

        self.assertEqual(code, 0)
        opener.assert_called_once_with(
            Path("/state/qwen-warm"),
            runtime_profile_sha256=None,
            runtime_code_revision=None,
            template_output_character_limit=None,
            prompt_token_verifier=None,
            semantic_key_verifier=None,
        )
        self.assertIs(wrapper.call_args.kwargs["ooe_hook"], hook)

    def test_cli_binds_growing_warm_cells_to_the_preload_runtime_profile(
        self,
    ) -> None:
        qwen = _chat(_Runtime())
        hook = object()

        def open_mount(_root, **options):
            return SimpleNamespace(
                hook=hook,
                result_cell_code_revision=options["runtime_code_revision"],
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "Qwen"
            q4 = root / "causal" / "q4-base-v2"
            q4.mkdir(parents=True)
            (q4 / "manifest.json").write_bytes(b"q4-manifest")
            (root / "tokenizer.json").write_bytes(b"tokenizer")
            page_state = root / "pages.json"
            prompt_tokenizer = SimpleNamespace(
                encode=lambda _text: (11, 12),
                render_no_thinking_prompt=(
                    lambda _system, question: f"prompt:{question}"
                ),
            )
            with (
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=qwen,
                ) as constructor,
                patch(
                    "immer.runtimes.ooe.qwen_warm_bank.open_verified_qwen_warm_bank",
                    side_effect=open_mount,
                ) as opener,
                patch(
                    "immer.cognition.qwen_fertig_chat.QwenFertigChat",
                    side_effect=lambda raw, _fertig, **_options: raw,
                ),
                patch(
                    "immer.runtimes.qwen3_8.encoding.Qwen38Tokenizer",
                    return_value=prompt_tokenizer,
                ),
                redirect_stdout(io.StringIO()),
            ):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--qwen38-root",
                        str(root),
                        "--ooe-warm-root",
                        "/state/qwen-warm",
                        "--mlp-page-state",
                        str(page_state),
                    ]
                )

        self.assertEqual(code, 0)
        profile = opener.call_args.kwargs["runtime_profile_sha256"]
        code_revision = opener.call_args.kwargs["runtime_code_revision"]
        self.assertEqual(len(profile), 64)
        self.assertEqual(len(code_revision), 64)
        self.assertEqual(
            opener.call_args.kwargs["template_output_character_limit"],
            128,
        )
        self.assertTrue(callable(opener.call_args.kwargs["prompt_token_verifier"]))
        self.assertTrue(callable(opener.call_args.kwargs["semantic_key_verifier"]))
        self.assertEqual(
            constructor.call_args.kwargs["result_cell_code_revision"],
            code_revision,
        )

    def test_growing_warm_code_revision_covers_prompt_identity_sources(self) -> None:
        names = {path.name for path in _qwen38_runtime_code_paths()}
        self.assertIn("encoding.py", names)
        self.assertIn("cartography_probe.py", names)
        self.assertIn("adapter.py", names)
        self.assertIn("qwen_warm_growth.py", names)
        self.assertNotIn("layer_transition_builder.py", names)

    def test_hybrid_warm_profile_binds_every_draft_provider_abi(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            q4 = root / "q4"
            q4.mkdir()
            (q4 / "manifest.json").write_bytes(b"manifest")
            tokenizer = root / "tokenizer.json"
            tokenizer.write_bytes(b"tokenizer")
            args = SimpleNamespace(
                compute_dtype="auto",
                device="auto",
                draft_window=8,
                head_block_rows=2048,
                max_context_tokens=2048,
                max_new_tokens=64,
                max_prompt_tokens=1024,
                q4_threads=None,
                qwen38_anchor_cache=None,
                system_prompt="",
            )
            current = _qwen38_growing_warm_profile(
                args,
                tokenizer_path=tokenizer,
                q4_root=q4,
                fast_mlp_root=None,
                draft_mode="hybrid",
                markov_atlas_path=None,
                markov_o1_retention_path=None,
                runtime_code_revision="a" * 64,
            )
            with patch(
                "immer.cli._QWEN38_MARKOV_DRAFT_ABI",
                "immer.qwen3.8-markov-draft-provider/v999",
            ):
                changed = _qwen38_growing_warm_profile(
                    args,
                    tokenizer_path=tokenizer,
                    q4_root=q4,
                    fast_mlp_root=None,
                    draft_mode="hybrid",
                    markov_atlas_path=None,
                    markov_o1_retention_path=None,
                    runtime_code_revision="a" * 64,
                )
            with patch(
                "immer.cli._QWEN38_MTP_DRAFT_ABI",
                "immer.qwen3.5-mtp-draft-provider/v999",
            ):
                changed_mtp = _qwen38_growing_warm_profile(
                    args,
                    tokenizer_path=tokenizer,
                    q4_root=q4,
                    fast_mlp_root=None,
                    draft_mode="hybrid",
                    markov_atlas_path=None,
                    markov_o1_retention_path=None,
                    runtime_code_revision="a" * 64,
                )
            with patch(
                "immer.cli._QWEN38_HYBRID_DRAFT_ABI",
                "immer.qwen3.8-markov-mtp-hybrid-provider/v999",
            ):
                changed_hybrid = _qwen38_growing_warm_profile(
                    args,
                    tokenizer_path=tokenizer,
                    q4_root=q4,
                    fast_mlp_root=None,
                    draft_mode="hybrid",
                    markov_atlas_path=None,
                    markov_o1_retention_path=None,
                    runtime_code_revision="a" * 64,
                )
            args.delta_head_online_state = "/state/delta-head.json"
            args.delta_head_layers = (0, 2, 4)
            delta_enabled = _qwen38_growing_warm_profile(
                args,
                tokenizer_path=tokenizer,
                q4_root=q4,
                fast_mlp_root=None,
                draft_mode="hybrid",
                markov_atlas_path=None,
                markov_o1_retention_path=None,
                runtime_code_revision="a" * 64,
            )
            args.delta_head_layers = (0, 2)
            delta_other_layers = _qwen38_growing_warm_profile(
                args,
                tokenizer_path=tokenizer,
                q4_root=q4,
                fast_mlp_root=None,
                draft_mode="hybrid",
                markov_atlas_path=None,
                markov_o1_retention_path=None,
                runtime_code_revision="a" * 64,
            )

        self.assertNotEqual(current, changed)
        self.assertNotEqual(current, changed_mtp)
        self.assertNotEqual(current, changed_hybrid)
        self.assertNotEqual(current, delta_enabled)
        self.assertNotEqual(delta_enabled, delta_other_layers)

    def test_growing_warm_profile_supports_dynamic_q4_page_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            q4 = root / "q4"
            q4.mkdir()
            (q4 / "manifest.json").write_bytes(b"manifest")
            tokenizer = root / "tokenizer.json"
            tokenizer.write_bytes(b"tokenizer")
            args = SimpleNamespace(
                compute_dtype="auto",
                device="auto",
                draft_window=8,
                head_block_rows=2048,
                max_context_tokens=2048,
                max_new_tokens=64,
                max_prompt_tokens=1024,
                mlp_page_width=192,
                q4_threads=None,
                qwen38_anchor_cache=None,
                system_prompt="",
            )
            common = {
                "args": args,
                "tokenizer_path": tokenizer,
                "q4_root": q4,
                "fast_mlp_root": None,
                "draft_mode": "hybrid",
                "markov_atlas_path": None,
                "markov_o1_retention_path": None,
                "runtime_code_revision": "a" * 64,
            }
            disabled = _qwen38_growing_warm_profile(
                **common,
                mlp_page_state_path=None,
            )
            routed = _qwen38_growing_warm_profile(
                **common,
                mlp_page_state_path=root / "pages-a.json",
            )
            args.attention_output_crystal_state = root / "attention.json"
            crystal_routed = _qwen38_growing_warm_profile(
                **common,
                mlp_page_state_path=root / "pages-a.json",
            )
            args.attention_output_crystal_state = root / "attention-b.json"
            same_crystal_other_file = _qwen38_growing_warm_profile(
                **common,
                mlp_page_state_path=root / "pages-a.json",
            )
            args.attention_output_crystal_state = None
            args.mlp_page_coordinate_state = root / "coordinate-a.json"
            coordinate_routed = _qwen38_growing_warm_profile(
                **common,
                mlp_page_state_path=root / "pages-a.json",
            )
            args.mlp_page_coordinate_state = root / "coordinate-b.json"
            same_coordinate_other_file = _qwen38_growing_warm_profile(
                **common,
                mlp_page_state_path=root / "pages-a.json",
            )
            semantics_coordinate = _qwen38_output_semantics(
                args,
                tokenizer_path=tokenizer,
                q4_root=q4,
                mlp_page_state_path=root / "pages-a.json",
            )
            args.mlp_page_coordinate_state = None
            same_policy_other_file = _qwen38_growing_warm_profile(
                **common,
                mlp_page_state_path=root / "pages-b.json",
            )
            args.mlp_page_width = 160
            changed_width = _qwen38_growing_warm_profile(
                **common,
                mlp_page_state_path=root / "pages-a.json",
            )
            args.mlp_page_width = 192
            with patch(
                "immer.runtimes.qwen3_8.mlp_page_markov.MLP_PAGE_MARKOV_POLICY",
                "different-page-policy/v999",
            ):
                changed_policy = _qwen38_growing_warm_profile(
                    **common,
                    mlp_page_state_path=root / "pages-a.json",
                )
            semantics_a = _qwen38_output_semantics(
                args,
                tokenizer_path=tokenizer,
                q4_root=q4,
                mlp_page_state_path=root / "pages-a.json",
            )
            args.q4_threads = 7
            semantics_threads = _qwen38_output_semantics(
                args,
                tokenizer_path=tokenizer,
                q4_root=q4,
                mlp_page_state_path=root / "pages-a.json",
            )
            args.q4_threads = None
            changed_runtime_code = _qwen38_growing_warm_profile(
                args,
                tokenizer_path=tokenizer,
                q4_root=q4,
                fast_mlp_root=None,
                mlp_page_state_path=root / "pages-a.json",
                draft_mode="hybrid",
                markov_atlas_path=None,
                markov_o1_retention_path=None,
                runtime_code_revision="b" * 64,
            )

        self.assertIsNotNone(routed)
        self.assertNotEqual(disabled, routed)
        self.assertNotEqual(routed, crystal_routed)
        self.assertEqual(crystal_routed, same_crystal_other_file)
        self.assertNotEqual(routed, coordinate_routed)
        self.assertEqual(coordinate_routed, same_coordinate_other_file)
        self.assertEqual(routed, same_policy_other_file)
        self.assertNotEqual(routed, changed_width)
        self.assertNotEqual(routed, changed_policy)
        self.assertEqual(semantics_a, semantics_threads)
        self.assertEqual(semantics_a, semantics_coordinate)
        self.assertNotEqual(routed, changed_runtime_code)

    def test_cli_draft_abis_match_runtime_exports(self) -> None:
        self.assertEqual(_QWEN38_MARKOV_DRAFT_ABI, MARKOV_DRAFT_PROVIDER_ABI)
        self.assertEqual(
            _QWEN38_HYBRID_DRAFT_ABI,
            QWEN38_MARKOV_MTP_HYBRID_PROVIDER_SCHEMA,
        )
        self.assertEqual(
            _QWEN38_MTP_DRAFT_ABI,
            QWEN35_MTP_DRAFT_PROVIDER_SCHEMA,
        )

    def test_service_profile_binds_coordinate_activation_not_state_path(
        self,
    ) -> None:
        args = SimpleNamespace(
            mlp_page_coordinate_state=Path("/state/coordinate-a.json")
        )
        common = {
            "args": args,
            "bundle_path": Path("/models/qwen"),
            "tokenizer_path": Path("/models/qwen/tokenizer.json"),
            "q4_root": Path("/models/qwen/q4"),
            "fast_mlp_root": None,
            "warm_root": None,
            "draft_mode": None,
            "markov_draft_state": None,
            "mtp_draft_state": None,
            "markov_atlas_path": None,
            "markov_o1_retention_path": None,
            "mlp_page_state_path": Path("/state/pages.json"),
            "draft_window_state_path": None,
            "runtime_code_revision": "a" * 64,
        }
        enabled_a = _qwen38_service_profile(
            **common,
            mlp_page_coordinate_enabled=True,
        )
        args.mlp_page_coordinate_state = Path("/other/coordinate-b.json")
        enabled_b = _qwen38_service_profile(
            **common,
            mlp_page_coordinate_enabled=True,
        )
        disabled = _qwen38_service_profile(
            **common,
            mlp_page_coordinate_enabled=False,
        )

        self.assertEqual(enabled_a, enabled_b)
        self.assertNotEqual(enabled_a, disabled)

    def test_service_profile_uses_path_free_registry_descriptor_and_pins_drift(
        self,
    ) -> None:
        registry = _fake_layer_mlp_mount_registry(18, 63)
        server_args = SimpleNamespace(
            layer_mlp_crystal_atlas=Path("/server/atlas"),
            layer_mlp_crystal_compute_root=Path("/server/compute"),
            layer_mlp_crystal_max_error_radius=0.0,
            layer_mlp_crystal_registry=Path("/server/registry.json"),
            layer_mlp_crystal_state=None,
        )
        client_args = SimpleNamespace(
            layer_mlp_crystal_atlas=Path("/client/atlas"),
            layer_mlp_crystal_compute_root=Path("/client/compute"),
            layer_mlp_crystal_max_error_radius=0.0,
            layer_mlp_crystal_registry=Path("/client/registry.json"),
            layer_mlp_crystal_state=None,
        )
        common = {
            "bundle_path": Path("/models/qwen"),
            "tokenizer_path": Path("/models/qwen/tokenizer.json"),
            "q4_root": Path("/models/qwen/q4"),
            "fast_mlp_root": None,
            "warm_root": None,
            "draft_mode": None,
            "markov_draft_state": None,
            "mtp_draft_state": None,
            "markov_atlas_path": None,
            "markov_o1_retention_path": None,
            "mlp_page_state_path": None,
            "draft_window_state_path": None,
            "runtime_code_revision": "a" * 64,
        }
        with (
            patch(
                "immer.runtimes.qwen3_8.layer_mlp_registry."
                "read_layer_mlp_o1_registry_manifest",
                return_value=registry,
            ),
            patch("immer.cli._authenticate_layer_transition_crystal_atlas"),
            patch("immer.cli._authenticate_layer_transition_crystal_compute_graph"),
            patch(
                "immer.runtimes.qwen3_8.layer_mlp_registry."
                "LayerMlpO1MountRegistry.load",
                side_effect=AssertionError("profile must not deserialize banks"),
            ),
        ):
            server = _qwen38_service_profile(args=server_args, **common)
            client = _qwen38_service_profile(args=client_args, **common)
            changed = SimpleNamespace(
                **{
                    **vars(registry),
                    "manifest_file_sha256": "8" * 64,
                    "registry_sha256": "9" * 64,
                }
            )
            with patch(
                "immer.runtimes.qwen3_8.layer_mlp_registry."
                "read_layer_mlp_o1_registry_manifest",
                return_value=changed,
            ):
                drifted = _qwen38_service_profile(args=client_args, **common)

        self.assertEqual(server, client)
        self.assertNotEqual(client, drifted)

    def test_canonical_prefix_sinkhorn_default_is_shared_by_service_and_client(
        self,
    ) -> None:
        canonical = Path("/models/canonical-qwen")
        q4 = canonical / "causal" / "q4"
        server_args = SimpleNamespace(prefix_sinkhorn=None)
        client_args = SimpleNamespace(prefix_sinkhorn=None)
        disabled_args = SimpleNamespace(prefix_sinkhorn=False)
        outside_args = SimpleNamespace(prefix_sinkhorn=None)
        explicit_args = SimpleNamespace(prefix_sinkhorn=True)
        common = {
            "bundle_path": canonical,
            "tokenizer_path": canonical / "tokenizer.json",
            "q4_root": q4,
            "fast_mlp_root": None,
            "warm_root": None,
            "draft_mode": None,
            "markov_draft_state": None,
            "mtp_draft_state": None,
            "markov_atlas_path": None,
            "markov_o1_retention_path": None,
            "mlp_page_state_path": None,
            "draft_window_state_path": None,
            "runtime_code_revision": "a" * 64,
        }
        with patch("immer.cli._QWEN38_DEPLOYMENT_ROOT", canonical):
            self.assertTrue(_resolve_qwen38_prefix_sinkhorn(server_args, canonical, q4))
            self.assertTrue(_resolve_qwen38_prefix_sinkhorn(client_args, canonical, q4))
            self.assertFalse(
                _resolve_qwen38_prefix_sinkhorn(disabled_args, canonical, q4)
            )
            self.assertFalse(
                _resolve_qwen38_prefix_sinkhorn(
                    outside_args,
                    Path("/models/other-qwen"),
                    q4,
                )
            )
            self.assertTrue(
                _resolve_qwen38_prefix_sinkhorn(
                    explicit_args,
                    Path("/models/other-qwen"),
                    q4,
                )
            )
            server_profile = _qwen38_service_profile(args=server_args, **common)
            client_profile = _qwen38_service_profile(args=client_args, **common)
            disabled_profile = _qwen38_service_profile(args=disabled_args, **common)

        self.assertEqual(server_profile, client_profile)
        self.assertNotEqual(server_profile, disabled_profile)

    def test_q4_residency_changes_service_profile_not_replay_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            q4 = root / "q4"
            q4.mkdir()
            (q4 / "manifest.json").write_bytes(b"manifest")
            tokenizer = root / "tokenizer.json"
            tokenizer.write_bytes(b"tokenizer")
            args = SimpleNamespace(
                compute_dtype="auto",
                device="auto",
                draft_window=8,
                head_block_rows=2048,
                max_context_tokens=2048,
                max_new_tokens=64,
                max_prompt_tokens=1024,
                mlp_page_width=192,
                q4_resident_budget_mb=0,
                q4_threads=None,
                qwen38_anchor_cache=None,
                system_prompt="",
            )
            warm_common = {
                "args": args,
                "tokenizer_path": tokenizer,
                "q4_root": q4,
                "fast_mlp_root": None,
                "draft_mode": None,
                "markov_atlas_path": None,
                "markov_o1_retention_path": None,
                "runtime_code_revision": "a" * 64,
            }
            service_common = {
                "args": args,
                "bundle_path": root / "bundle",
                "tokenizer_path": tokenizer,
                "q4_root": q4,
                "fast_mlp_root": None,
                "warm_root": None,
                "draft_mode": None,
                "markov_draft_state": None,
                "mtp_draft_state": None,
                "markov_atlas_path": None,
                "markov_o1_retention_path": None,
                "mlp_page_state_path": None,
                "draft_window_state_path": None,
                "runtime_code_revision": "a" * 64,
            }
            warm_eager = _qwen38_growing_warm_profile(**warm_common)
            service_eager = _qwen38_service_profile(**service_common)
            semantics_eager = _qwen38_output_semantics(
                args,
                tokenizer_path=tokenizer,
                q4_root=q4,
                mlp_page_state_path=None,
            )
            args.q4_resident_budget_mb = 16384
            warm_resident = _qwen38_growing_warm_profile(**warm_common)
            service_resident = _qwen38_service_profile(**service_common)
            semantics_resident = _qwen38_output_semantics(
                args,
                tokenizer_path=tokenizer,
                q4_root=q4,
                mlp_page_state_path=None,
            )

        self.assertEqual(warm_eager, warm_resident)
        self.assertNotEqual(service_eager, service_resident)
        self.assertEqual(semantics_eager, semantics_resident)

    def test_layer_transition_profiles_bind_identity_budget_not_state_path(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            q4 = root / "q4"
            q4.mkdir()
            (q4 / "manifest.json").write_bytes(b"manifest")
            tokenizer = root / "tokenizer.json"
            tokenizer.write_bytes(b"tokenizer")
            state_a = root / "layer-a.bin"
            state_b = root / "layer-b.bin"
            state_a.write_bytes(b"sealed-a")
            state_b.write_bytes(b"sealed-b")
            atlas_a = root / "atlas-a"
            atlas_b = root / "atlas-b"
            compute_a = root / "compute-a"
            compute_b = root / "compute-b"
            atlas_revision = _semantic_atlas_authority(atlas_a)
            self.assertEqual(atlas_revision, _semantic_atlas_authority(atlas_b))
            graph_revision = _compute_graph_authority(compute_a)
            self.assertEqual(graph_revision, _compute_graph_authority(compute_b))
            identity = _layer_transition_identity(
                atlas=atlas_revision,
                graph=graph_revision,
            )
            opened = SimpleNamespace(identity=identity, crystals=())
            args = SimpleNamespace(
                compute_dtype="auto",
                device="auto",
                draft_window=8,
                head_block_rows=2048,
                layer_transition_crystal_atlas=atlas_a,
                layer_transition_crystal_compute_root=compute_a,
                layer_transition_crystal_max_error_radius=0.25,
                layer_transition_crystal_state=state_a,
                max_context_tokens=2048,
                max_new_tokens=64,
                max_prompt_tokens=1024,
                mlp_page_width=192,
                q4_threads=None,
                qwen38_anchor_cache=None,
                system_prompt="",
            )
            warm_common = {
                "args": args,
                "tokenizer_path": tokenizer,
                "q4_root": q4,
                "fast_mlp_root": None,
                "mlp_page_state_path": None,
                "draft_mode": None,
                "markov_atlas_path": None,
                "markov_o1_retention_path": None,
                "runtime_code_revision": "a" * 64,
            }
            service_common = {
                "args": args,
                "bundle_path": root / "bundle",
                "tokenizer_path": tokenizer,
                "q4_root": q4,
                "fast_mlp_root": None,
                "warm_root": None,
                "draft_mode": None,
                "markov_draft_state": None,
                "mtp_draft_state": None,
                "markov_atlas_path": None,
                "markov_o1_retention_path": None,
                "mlp_page_state_path": None,
                "draft_window_state_path": None,
                "runtime_code_revision": "a" * 64,
            }
            with patch(
                "immer.runtimes.qwen3_8.layer_transition_crystal."
                "LayerTransitionCrystalBank.load",
                return_value=opened,
            ):
                warm_a = _qwen38_growing_warm_profile(**warm_common)
                service_a = _qwen38_service_profile(**service_common)
                self.assertIsNone(
                    _qwen38_output_semantics(
                        args,
                        tokenizer_path=tokenizer,
                        q4_root=q4,
                        mlp_page_state_path=None,
                    )
                )
                args.layer_transition_crystal_state = state_b
                args.layer_transition_crystal_atlas = atlas_b
                args.layer_transition_crystal_compute_root = compute_b
                warm_b = _qwen38_growing_warm_profile(**warm_common)
                service_b = _qwen38_service_profile(**service_common)
                args.layer_transition_crystal_max_error_radius = 0.5
                warm_other_budget = _qwen38_growing_warm_profile(**warm_common)
                service_other_budget = _qwen38_service_profile(**service_common)
            args.layer_transition_crystal_max_error_radius = 0.25
            with patch(
                "immer.runtimes.qwen3_8.layer_transition_crystal."
                "LayerTransitionCrystalBank.load",
                return_value=SimpleNamespace(
                    identity=_layer_transition_identity(
                        model="6",
                        atlas=atlas_revision,
                        graph=graph_revision,
                    ),
                    crystals=(),
                ),
            ):
                warm_other_identity = _qwen38_growing_warm_profile(**warm_common)
                service_other_identity = _qwen38_service_profile(**service_common)
            args.layer_transition_crystal_state = None
            args.layer_transition_crystal_atlas = None
            args.layer_transition_crystal_compute_root = None
            args.layer_transition_crystal_max_error_radius = 0.0
            output_without_layer = _qwen38_output_semantics(
                args,
                tokenizer_path=tokenizer,
                q4_root=q4,
                mlp_page_state_path=None,
            )

        self.assertEqual(warm_a, warm_b)
        self.assertEqual(service_a, service_b)
        self.assertNotEqual(warm_a, warm_other_budget)
        self.assertNotEqual(service_a, service_other_budget)
        self.assertNotEqual(warm_a, warm_other_identity)
        self.assertNotEqual(service_a, service_other_identity)
        self.assertIsNotNone(output_without_layer)

    def test_layer_mlp_profiles_bind_identity_budget_not_private_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            q4 = root / "q4"
            q4.mkdir()
            (q4 / "manifest.json").write_bytes(b"manifest")
            tokenizer = root / "tokenizer.json"
            tokenizer.write_bytes(b"tokenizer")
            state_a = root / "mlp-a.bin"
            state_b = root / "mlp-b.bin"
            state_a.write_bytes(b"sealed-a")
            state_b.write_bytes(b"sealed-b")
            atlas_a = root / "atlas-a"
            atlas_b = root / "atlas-b"
            compute_a = root / "compute-a"
            compute_b = root / "compute-b"
            atlas_revision = _semantic_atlas_authority(atlas_a)
            self.assertEqual(atlas_revision, _semantic_atlas_authority(atlas_b))
            graph_revision = _compute_graph_authority(compute_a)
            self.assertEqual(graph_revision, _compute_graph_authority(compute_b))
            identity = _layer_mlp_identity(
                atlas=atlas_revision,
                graph=graph_revision,
            )
            opened = SimpleNamespace(identity=identity, crystals=())
            args = SimpleNamespace(
                compute_dtype="auto",
                device="auto",
                draft_window=8,
                head_block_rows=2048,
                layer_mlp_crystal_atlas=atlas_a,
                layer_mlp_crystal_compute_root=compute_a,
                layer_mlp_crystal_max_error_radius=0.25,
                layer_mlp_crystal_state=state_a,
                max_context_tokens=2048,
                max_new_tokens=64,
                max_prompt_tokens=1024,
                mlp_page_width=192,
                q4_threads=None,
                qwen38_anchor_cache=None,
                system_prompt="",
            )
            warm_common = {
                "args": args,
                "tokenizer_path": tokenizer,
                "q4_root": q4,
                "fast_mlp_root": None,
                "mlp_page_state_path": None,
                "draft_mode": None,
                "markov_atlas_path": None,
                "markov_o1_retention_path": None,
                "runtime_code_revision": "a" * 64,
            }
            service_common = {
                "args": args,
                "bundle_path": root / "bundle",
                "tokenizer_path": tokenizer,
                "q4_root": q4,
                "fast_mlp_root": None,
                "warm_root": None,
                "draft_mode": None,
                "markov_draft_state": None,
                "mtp_draft_state": None,
                "markov_atlas_path": None,
                "markov_o1_retention_path": None,
                "mlp_page_state_path": None,
                "draft_window_state_path": None,
                "runtime_code_revision": "a" * 64,
            }
            with patch(
                "immer.runtimes.qwen3_8.layer_mlp_crystal."
                "Layer63MlpResidualCrystalBank.load",
                return_value=opened,
            ):
                warm_a = _qwen38_growing_warm_profile(**warm_common)
                service_a = _qwen38_service_profile(**service_common)
                self.assertIsNone(
                    _qwen38_output_semantics(
                        args,
                        tokenizer_path=tokenizer,
                        q4_root=q4,
                        mlp_page_state_path=None,
                    )
                )
                args.layer_mlp_crystal_state = state_b
                args.layer_mlp_crystal_atlas = atlas_b
                args.layer_mlp_crystal_compute_root = compute_b
                warm_b = _qwen38_growing_warm_profile(**warm_common)
                service_b = _qwen38_service_profile(**service_common)
                args.layer_mlp_crystal_max_error_radius = 0.5
                warm_other_budget = _qwen38_growing_warm_profile(**warm_common)
                service_other_budget = _qwen38_service_profile(**service_common)

        self.assertEqual(warm_a, warm_b)
        self.assertEqual(service_a, service_b)
        self.assertNotEqual(warm_a, warm_other_budget)
        self.assertNotEqual(service_a, service_other_budget)

    def test_layer_transition_economics_identity_excludes_authority_paths(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_a = root / "layer-a.bin"
            state_b = root / "layer-b.bin"
            state_a.write_bytes(b"sealed-a")
            state_b.write_bytes(b"sealed-b")
            atlas_a = root / "atlas-a"
            atlas_b = root / "atlas-b"
            compute_a = root / "compute-a"
            compute_b = root / "compute-b"
            atlas_revision = _semantic_atlas_authority(atlas_a)
            self.assertEqual(atlas_revision, _semantic_atlas_authority(atlas_b))
            graph_revision = _compute_graph_authority(compute_a)
            self.assertEqual(graph_revision, _compute_graph_authority(compute_b))
            identity = _layer_transition_identity(
                atlas=atlas_revision,
                graph=graph_revision,
            )

            def runtime_profile(
                state: Path,
                atlas: Path,
                compute: Path,
                economics: Path,
            ) -> str:
                output = io.StringIO()
                with (
                    patch(
                        "immer.runtimes.qwen3_8.layer_transition_crystal."
                        "LayerTransitionCrystalBank.load",
                        return_value=SimpleNamespace(identity=identity, crystals=()),
                    ),
                    patch(
                        "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                        return_value=_chat(_Runtime()),
                    ),
                    redirect_stdout(output),
                ):
                    code = main(
                        [
                            "chat",
                            "hello",
                            "--output",
                            "json",
                            "--raw-qwen",
                            "--no-markov-draft",
                            "--qwen38-causal-bundle",
                            "/models/qwen.causal",
                            "--qwen38-tokenizer",
                            "/models/tokenizer.json",
                            "--qwen38-q4",
                            "/models/qwen-q4",
                            "--layer-transition-crystal-state",
                            str(state),
                            "--layer-transition-crystal-atlas",
                            str(atlas),
                            "--layer-transition-crystal-compute-root",
                            str(compute),
                            "--inference-economics-state",
                            str(economics),
                        ]
                    )
                self.assertEqual(code, 0)
                document = json.loads(output.getvalue())
                return document["evidence"]["inference_economics"]["receipt"]["body"][
                    "runtime_profile_sha256"
                ]

            profile_a = runtime_profile(
                state_a,
                atlas_a,
                compute_a,
                root / "economics-a",
            )
            profile_b = runtime_profile(
                state_b,
                atlas_b,
                compute_b,
                root / "economics-b",
            )

        self.assertEqual(profile_a, profile_b)

    def test_cli_explicit_layout_does_not_inherit_deployed_q4(self) -> None:
        qwen = _chat(_Runtime())
        with tempfile.TemporaryDirectory() as temporary:
            deployed = Path(temporary) / "deployed"
            (deployed / "causal" / "q4-base-v2").mkdir(parents=True)
            with (
                patch.dict("os.environ", {}, clear=True),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_ROOT",
                    deployed,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=qwen,
                ) as constructor,
                redirect_stdout(io.StringIO()),
            ):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--qwen38-causal-bundle",
                        "/models/other.causal",
                        "--qwen38-tokenizer",
                        "/models/other-tokenizer.json",
                        "--fast-mlp",
                        "/models/other-fast",
                        "--fast-mlp-layers",
                        "0,9",
                        "--raw-qwen",
                    ]
                )

        self.assertEqual(code, 0)
        self.assertIsNone(constructor.call_args.kwargs["q4_root"])
        self.assertEqual(
            constructor.call_args.kwargs["fast_mlp_active_layers"],
            (0, 9),
        )

    def test_cli_can_disable_the_deployed_sparse_mlp_plan(self) -> None:
        qwen = _chat(_Runtime())
        with tempfile.TemporaryDirectory() as temporary:
            deployed = Path(temporary) / "deployed"
            (deployed / "causal" / "q4-base-v2").mkdir(parents=True)
            deployed_fast = Path(temporary) / "deployed-fast"
            deployed_fast.mkdir()
            with (
                patch.dict(
                    "os.environ",
                    {"IMMER_QWEN38_FAST_MLP": str(deployed_fast)},
                    clear=True,
                ),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_ROOT",
                    deployed,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=qwen,
                ) as constructor,
                redirect_stdout(io.StringIO()),
            ):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--no-fast-mlp",
                        "--no-markov-draft",
                        "--raw-qwen",
                    ]
                )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(
            options["q4_root"],
            str(deployed / "causal" / "q4-base-v2"),
        )

    def test_cli_mounts_direct_markov_page_execution_without_legacy_fast_mlp(
        self,
    ) -> None:
        qwen = _chat(_Runtime())
        with (
            patch.dict(
                "os.environ",
                {"IMMER_QWEN38_FAST_MLP": "/state/legacy-fast"},
                clear=True,
            ),
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                return_value=qwen,
            ) as constructor,
            redirect_stdout(io.StringIO()),
        ):
            code = main(
                [
                    "chat",
                    "hello",
                    "--qwen38-causal-bundle",
                    "/models/qwen.causal",
                    "--qwen38-tokenizer",
                    "/models/tokenizer.json",
                    "--qwen38-q4",
                    "/models/q4",
                    "--mlp-page-state",
                    "/state/pages.json",
                    "--mlp-page-width",
                    "160",
                    "--no-markov-draft",
                    "--raw-qwen",
                ]
            )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertIsNone(options["fast_mlp_root"])
        self.assertEqual(options["mlp_page_state_path"], Path("/state/pages.json"))
        self.assertEqual(options["mlp_page_route_width"], 160)
        self.assertIsNone(options["fast_mlp_root"])
        self.assertIsNone(options["fast_mlp_active_layers"])

    def test_cli_deployment_defaults_to_full_q4(self) -> None:
        qwen = _chat(_Runtime())
        with tempfile.TemporaryDirectory() as temporary:
            deployed = Path(temporary) / "deployed"
            (deployed / "causal" / "q4-base-v2").mkdir(parents=True)
            attention_state = Path(temporary) / "attention-output.json"
            with (
                patch.dict("os.environ", {}, clear=True),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_ROOT",
                    deployed,
                ),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_ATTENTION_OUTPUT_CRYSTAL_STATE",
                    attention_state,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=qwen,
                ) as constructor,
                redirect_stdout(io.StringIO()),
            ):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--raw-qwen",
                        "--no-markov-draft",
                    ]
                )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(
            options["q4_root"],
            str(deployed / "causal" / "q4-base-v2"),
        )
        self.assertIsNone(options["fast_mlp_root"])
        self.assertIsNone(options["fast_mlp_active_layers"])
        self.assertIsNone(options["mlp_page_state_path"])
        self.assertIsNone(options["draft_mode"])
        self.assertIsNone(options["markov_draft_state_path"])
        self.assertEqual(
            options["attention_output_crystal_state_path"],
            str(attention_state),
        )

    def test_cli_canonical_missing_or_empty_mlp_registry_is_not_mounted(
        self,
    ) -> None:
        from immer.runtimes.qwen3_8.layer_mlp_registry import (
            LayerMlpO1RegistryNotMountableError,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            deployed = root / "deployed"
            (deployed / "causal" / "q4-base-v2").mkdir(parents=True)
            missing = root / "missing" / "layer-mlp-o1-registry.json"
            empty = root / "empty" / "layer-mlp-o1-registry.json"
            empty.parent.mkdir()
            empty.write_text("sealed-empty-fixture", encoding="ascii")
            unavailable = root / "unavailable"

            for manifest, side_effect in (
                (missing, AssertionError("missing manifest must not be read")),
                (
                    empty,
                    LayerMlpO1RegistryNotMountableError(
                        "valid registry has no ready banks"
                    ),
                ),
            ):
                qwen = _chat(_Runtime())
                with (
                    patch.dict("os.environ", {}, clear=True),
                    patch("immer.cli._QWEN38_DEPLOYMENT_ROOT", deployed),
                    patch(
                        "immer.cli._QWEN38_DEPLOYMENT_LAYER_MLP_CRYSTAL_REGISTRY",
                        manifest,
                    ),
                    patch(
                        "immer.cli._QWEN38_DEPLOYMENT_LAYER_MLP_O1_ATLAS",
                        unavailable,
                    ),
                    patch(
                        "immer.cli._QWEN38_DEPLOYMENT_LAYER_MLP_O1_COMPUTE",
                        unavailable,
                    ),
                    patch(
                        "immer.runtimes.qwen3_8.layer_mlp_registry."
                        "read_layer_mlp_o1_registry_manifest",
                        side_effect=side_effect,
                    ) as reader,
                    patch(
                        "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                        return_value=qwen,
                    ) as constructor,
                    redirect_stdout(io.StringIO()),
                ):
                    code = main(
                        [
                            "chat",
                            "hello",
                            "--raw-qwen",
                            "--no-markov-draft",
                        ]
                    )

                self.assertEqual(code, 0)
                self.assertIsNone(
                    constructor.call_args.kwargs[
                        "layer_mlp_crystal_registry_path"
                    ]
                )
                self.assertEqual(reader.call_count, int(manifest.is_file()))

    def test_cli_deployment_mounts_existing_continuation_battery(self) -> None:
        qwen = _chat(_Runtime())
        with tempfile.TemporaryDirectory() as temporary:
            deployed = Path(temporary) / "deployed"
            (deployed / "causal" / "q4-base-v2").mkdir(parents=True)
            anchor = Path(temporary) / "anchors"
            anchor.mkdir()
            with (
                patch.dict("os.environ", {}, clear=True),
                patch("immer.cli._QWEN38_DEPLOYMENT_ROOT", deployed),
                patch("immer.cli._QWEN38_DEPLOYMENT_ANCHOR_CACHE", anchor),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=qwen,
                ) as constructor,
                redirect_stdout(io.StringIO()),
            ):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--raw-qwen",
                        "--no-markov-draft",
                    ]
                )

        self.assertEqual(code, 0)
        cache = constructor.call_args.kwargs["anchor_cache"]
        self.assertIsInstance(cache, SemanticStateAnchorCache)
        self.assertEqual(cache.root, anchor)

    def test_cli_non_bf16_deployment_requires_coordinate_opt_out(self) -> None:
        qwen = _chat(_Runtime())
        with tempfile.TemporaryDirectory() as temporary:
            deployed = Path(temporary) / "deployed"
            (deployed / "causal" / "q4-base-v3-mtp").mkdir(parents=True)
            page_state = Path(temporary) / "pages.json"
            coordinate_state = Path(temporary) / "coordinate.json"
            with (
                patch.dict("os.environ", {}, clear=True),
                patch("immer.cli._QWEN38_DEPLOYMENT_ROOT", deployed),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_MLP_PAGE_STATE",
                    page_state,
                ),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_MLP_PAGE_COORDINATE_STATE",
                    coordinate_state,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=qwen,
                ) as constructor,
                redirect_stderr(io.StringIO()),
                redirect_stdout(io.StringIO()),
            ):
                rejected = main(
                    [
                        "chat",
                        "hello",
                        "--compute-dtype",
                        "float32",
                        "--raw-qwen",
                        "--no-markov-draft",
                    ]
                )
                accepted = main(
                    [
                        "chat",
                        "hello",
                        "--compute-dtype",
                        "float32",
                        "--no-mlp-page-coordinate",
                        "--raw-qwen",
                        "--no-markov-draft",
                    ]
                )

        self.assertEqual(rejected, 2)
        self.assertEqual(accepted, 0)
        self.assertIsNone(
            constructor.call_args.kwargs["mlp_page_coordinate_state_path"]
        )

    def test_cli_deployment_enables_passive_economics_by_default(self) -> None:
        qwen = _chat(_Runtime())
        with tempfile.TemporaryDirectory() as temporary:
            deployed = Path(temporary) / "deployed"
            (deployed / "causal" / "q4-base-v2").mkdir(parents=True)
            state = Path(temporary) / "state"
            state.mkdir()
            economics = state / "qwen-inference-economics-v1"
            with (
                patch.dict("os.environ", {}, clear=True),
                patch("immer.cli._QWEN38_DEPLOYMENT_ROOT", deployed),
                patch("immer.cli._QWEN38_DEPLOYMENT_STATE", state),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_INFERENCE_ECONOMICS",
                    economics,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=qwen,
                ),
                redirect_stdout(io.StringIO()),
            ):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--raw-qwen",
                        "--no-markov-draft",
                    ]
                )
            self.assertEqual(code, 0)
            snapshot = InferenceEconomicsLedger(economics).snapshot()
            self.assertEqual(snapshot["requests"], 1)
            self.assertEqual(snapshot["target_forwards"], 3)

    def test_cli_deployment_mounts_the_persistent_markov_token_council(
        self,
    ) -> None:
        qwen = _chat(_Runtime())
        with tempfile.TemporaryDirectory() as temporary:
            deployed = Path(temporary) / "deployed"
            (deployed / "causal" / "q4-base-v2").mkdir(parents=True)
            state = Path(temporary) / "qwen-markov.bin"
            state.write_bytes(b"fixture")
            with (
                patch.dict("os.environ", {}, clear=True),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_ROOT",
                    deployed,
                ),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_MARKOV_DRAFT_STATE",
                    state,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=qwen,
                ) as constructor,
                redirect_stdout(io.StringIO()),
            ):
                code = main(["chat", "hello", "--raw-qwen"])

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(options["draft_mode"], "markov")
        self.assertEqual(options["markov_draft_state_path"], str(state))

    def test_cli_deployment_uses_hybrid_novelty_fallback(self) -> None:
        qwen = _chat(_Runtime())
        with tempfile.TemporaryDirectory() as temporary:
            deployed = Path(temporary) / "deployed"
            q4 = deployed / "causal" / "q4-base-v3-mtp"
            q4.mkdir(parents=True)
            markov_state = Path(temporary) / "qwen-markov.bin"
            markov_state.write_bytes(b"fixture")
            mtp_state = Path(temporary) / "qwen-mtp.json"
            qwen35 = Path(temporary) / "Qwen3.5-0.8B"
            qwen35.mkdir()
            page_state = Path(temporary) / "qwen-mlp-pages.json"
            coordinate_state = Path(temporary) / "qwen-mlp-coordinate.json"
            context_state = Path(temporary) / "qwen-context.json"
            with (
                patch.dict("os.environ", {}, clear=True),
                patch("immer.cli._QWEN38_DEPLOYMENT_ROOT", deployed),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_MARKOV_DRAFT_STATE",
                    markov_state,
                ),
                patch("immer.cli._QWEN38_DEPLOYMENT_MTP_STATE", mtp_state),
                patch("immer.cli._QWEN35_DEPLOYMENT_DRAFT_ROOT", qwen35),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_MLP_PAGE_STATE",
                    page_state,
                ),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_MLP_PAGE_COORDINATE_STATE",
                    coordinate_state,
                ),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_CONTEXT_CRYSTAL_STATE",
                    context_state,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=qwen,
                ) as constructor,
                redirect_stdout(io.StringIO()),
            ):
                code = main(["chat", "hello", "--raw-qwen"])

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(options["q4_root"], str(q4))
        self.assertEqual(
            options["q4_resident_budget_bytes"],
            16_384 * 1024**2,
        )
        self.assertEqual(options["draft_mode"], "hybrid")
        self.assertEqual(options["draft_bundle_path"], str(qwen35))
        self.assertEqual(options["markov_draft_state_path"], str(markov_state))
        self.assertEqual(options["mtp_draft_state_path"], str(mtp_state))
        self.assertEqual(options["mlp_page_state_path"], page_state)
        self.assertEqual(
            options["mlp_page_coordinate_state_path"],
            str(coordinate_state),
        )
        self.assertEqual(
            options["contextual_continuation_state_path"],
            str(context_state),
        )
        self.assertEqual(options["draft_window"], 16)
        self.assertIsNone(options["fast_mlp_root"])

    def test_cli_can_disable_the_deployed_dynamic_mlp_page_route(self) -> None:
        qwen = _chat(_Runtime())
        with tempfile.TemporaryDirectory() as temporary:
            deployed = Path(temporary) / "deployed"
            q4 = deployed / "causal" / "q4-base-v3-mtp"
            q4.mkdir(parents=True)
            markov_state = Path(temporary) / "qwen-markov.bin"
            markov_state.write_bytes(b"fixture")
            with (
                patch.dict("os.environ", {}, clear=True),
                patch("immer.cli._QWEN38_DEPLOYMENT_ROOT", deployed),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_MARKOV_DRAFT_STATE",
                    markov_state,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=qwen,
                ) as constructor,
                redirect_stdout(io.StringIO()),
            ):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--raw-qwen",
                        "--no-mlp-page-route",
                        "--no-mlp-page-coordinate",
                        "--no-context-crystal",
                        "--no-attention-output-crystal",
                        "--q4-resident-budget-mb",
                        "0",
                    ]
                )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(options["q4_root"], str(q4))
        self.assertEqual(options["q4_resident_budget_bytes"], 0)
        self.assertIsNone(options["mlp_page_state_path"])
        self.assertIsNone(options["mlp_page_coordinate_state_path"])
        self.assertIsNone(options["contextual_continuation_state_path"])
        self.assertIsNone(options["attention_output_crystal_state_path"])

    def test_cli_deployment_keeps_markov_as_an_explicit_opt_out(self) -> None:
        qwen = _chat(_Runtime())
        with tempfile.TemporaryDirectory() as temporary:
            deployed = Path(temporary) / "deployed"
            q4 = deployed / "causal" / "q4-base-v3-mtp"
            q4.mkdir(parents=True)
            markov_state = Path(temporary) / "qwen-markov.bin"
            markov_state.write_bytes(b"fixture")
            mtp_state = Path(temporary) / "qwen-mtp.json"
            with (
                patch.dict("os.environ", {}, clear=True),
                patch("immer.cli._QWEN38_DEPLOYMENT_ROOT", deployed),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_MARKOV_DRAFT_STATE",
                    markov_state,
                ),
                patch("immer.cli._QWEN38_DEPLOYMENT_MTP_STATE", mtp_state),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=qwen,
                ) as constructor,
                redirect_stdout(io.StringIO()),
            ):
                code = main(["chat", "hello", "--raw-qwen", "--draft-mode", "markov"])

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(options["q4_root"], str(q4))
        self.assertEqual(options["draft_mode"], "markov")
        self.assertEqual(options["markov_draft_state_path"], str(markov_state))
        self.assertIsNone(options["mtp_draft_state_path"])

    def test_cli_anchor_keeps_the_compatible_markov_provider(self) -> None:
        qwen = _chat(_Runtime())
        with tempfile.TemporaryDirectory() as temporary:
            deployed = Path(temporary) / "deployed"
            q4 = deployed / "causal" / "q4-base-v3-mtp"
            q4.mkdir(parents=True)
            markov_state = Path(temporary) / "qwen-markov.bin"
            markov_state.write_bytes(b"fixture")
            anchor = Path(temporary) / "anchors"
            with (
                patch.dict("os.environ", {}, clear=True),
                patch("immer.cli._QWEN38_DEPLOYMENT_ROOT", deployed),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_MARKOV_DRAFT_STATE",
                    markov_state,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=qwen,
                ) as constructor,
                redirect_stdout(io.StringIO()),
            ):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--raw-qwen",
                        "--qwen38-anchor-cache",
                        str(anchor),
                    ]
                )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(options["q4_root"], str(q4))
        self.assertEqual(options["draft_mode"], "markov")
        self.assertEqual(options["markov_draft_state_path"], str(markov_state))
        self.assertIsNone(options["mtp_draft_state_path"])

    def test_cli_explicit_mtp_uses_persistent_calibration_state(self) -> None:
        qwen = _chat(_Runtime())
        with tempfile.TemporaryDirectory() as temporary:
            deployed = Path(temporary) / "deployed"
            deployed.mkdir()
            q4 = deployed / "causal" / "q4-base-v3-mtp"
            q4.mkdir(parents=True)
            state = Path(temporary) / "qwen-mtp.json"
            with (
                patch.dict("os.environ", {}, clear=True),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_ROOT",
                    deployed,
                ),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_MTP_STATE",
                    state,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=qwen,
                ) as constructor,
                redirect_stdout(io.StringIO()),
            ):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--raw-qwen",
                        "--draft-mode",
                        "mtp",
                    ]
                )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(options["draft_mode"], "mtp")
        self.assertEqual(options["q4_root"], str(q4))
        self.assertIsNone(options["markov_draft_state_path"])
        self.assertEqual(options["mtp_draft_state_path"], str(state))

    def test_cli_hybrid_rejects_a_target_bank_without_mtp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            deployed = Path(temporary) / "deployed"
            (deployed / "causal" / "q4-base-v2").mkdir(parents=True)
            stderr = io.StringIO()
            with (
                patch.dict("os.environ", {}, clear=True),
                patch("immer.cli._QWEN38_DEPLOYMENT_ROOT", deployed),
                redirect_stderr(stderr),
            ):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--raw-qwen",
                        "--draft-mode",
                        "hybrid",
                    ]
                )

        self.assertEqual(code, 2)
        self.assertIn("hybrid draft mode requires", stderr.getvalue())

    def test_cli_jsonl_reuses_one_loaded_component_for_multiple_requests(self) -> None:
        qwen = _chat(_Runtime())
        output = io.StringIO()
        stream = io.StringIO('{"id":"first","message":"hello"}\nworld\n{"bad":true}\n')
        with (
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                return_value=qwen,
            ) as constructor,
            patch("sys.stdin", stream),
            redirect_stdout(output),
        ):
            code = main(
                [
                    "chat",
                    "--jsonl",
                    "--max-requests",
                    "2",
                    "--qwen38-causal-bundle",
                    "/models/qwen.causal",
                    "--qwen38-tokenizer",
                    "/models/tokenizer.json",
                ]
            )

        self.assertEqual(code, 0)
        constructor.assert_called_once()
        self.assertIsNone(constructor.call_args.kwargs["text_snapshot_sink"])
        self.assertTrue(qwen.closed)
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["id"], "first")
        self.assertNotIn("id", rows[1])
        self.assertEqual([row["output"] for row in rows], ["local answer"] * 2)

    def test_cli_deployment_enables_passive_decode_o1_by_default(self) -> None:
        qwen = _chat(_Runtime())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            deployed = root / "deployed"
            (deployed / "causal" / "q4-base-v2").mkdir(parents=True)
            atlas = root / "atlas"
            compute = root / "compute"
            _semantic_atlas_authority(atlas)
            _compute_graph_authority(compute)
            state = root / "decode-o1.json"
            with (
                patch.dict("os.environ", {}, clear=True),
                patch("immer.cli._QWEN38_DEPLOYMENT_ROOT", deployed),
                patch("immer.cli._QWEN38_DEPLOYMENT_LAYER_MLP_O1_STATE", state),
                patch("immer.cli._QWEN38_DEPLOYMENT_LAYER_MLP_O1_ATLAS", atlas),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_LAYER_MLP_O1_COMPUTE",
                    compute,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=qwen,
                ) as constructor,
                redirect_stdout(io.StringIO()),
            ):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--raw-qwen",
                        "--no-markov-draft",
                    ]
                )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(options["layer_mlp_o1_state_path"], str(state))
        self.assertEqual(options["layer_mlp_o1_atlas_path"], str(atlas))
        self.assertEqual(options["layer_mlp_o1_compute_root"], str(compute))
        self.assertEqual(options["layer_mlp_o1_layers"], tuple(range(64)))
        self.assertEqual(options["layer_mlp_o1_sketch_dim"], 128)

    def test_cli_explicit_layer63_keeps_legacy_passive_o1_runtime(self) -> None:
        qwen = _chat(_Runtime())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            deployed = root / "deployed"
            (deployed / "causal" / "q4-base-v2").mkdir(parents=True)
            atlas = root / "atlas"
            compute = root / "compute"
            _semantic_atlas_authority(atlas)
            _compute_graph_authority(compute)
            state = root / "decode-o1.json"
            with (
                patch.dict("os.environ", {}, clear=True),
                patch("immer.cli._QWEN38_DEPLOYMENT_ROOT", deployed),
                patch("immer.cli._QWEN38_DEPLOYMENT_LAYER_MLP_O1_STATE", state),
                patch("immer.cli._QWEN38_DEPLOYMENT_LAYER_MLP_O1_ATLAS", atlas),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_LAYER_MLP_O1_COMPUTE",
                    compute,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=qwen,
                ) as constructor,
                redirect_stdout(io.StringIO()),
            ):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--raw-qwen",
                        "--no-markov-draft",
                        "--layer-mlp-o1-layers",
                        "63",
                    ]
                )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(options["layer_mlp_o1_layers"], (63,))

    def test_cli_economics_covers_single_jsonl_and_interactive_fail_open(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            economics = Path(temporary) / "economics"

            single_runtime = _Runtime()
            single_qwen = _chat(single_runtime)
            single_output = io.StringIO()
            with (
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=single_qwen,
                ),
                redirect_stdout(single_output),
            ):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--raw-qwen",
                        "--output",
                        "json",
                        "--qwen38-causal-bundle",
                        "/models/qwen.causal",
                        "--qwen38-tokenizer",
                        "/models/tokenizer.json",
                        "--inference-economics-state",
                        str(economics),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(len(single_runtime.model.calls), 1)
            single = json.loads(single_output.getvalue())
            self.assertEqual(
                single["evidence"]["inference_economics"]["status"],
                "recorded",
            )
            self.assertEqual(
                single["evidence"]["inference_action_bank"]["status"],
                "recorded",
            )
            self.assertEqual(
                InferenceEconomicsLedger(economics).snapshot()["requests"], 1
            )
            action_root = economics.parent / "qwen-inference-action-bank-v1"
            self.assertEqual(InferenceActionBank(action_root).snapshot()["requests"], 1)

            jsonl_runtime = _Runtime()
            jsonl_qwen = _chat(jsonl_runtime)
            jsonl_output = io.StringIO()
            with (
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=jsonl_qwen,
                ),
                patch(
                    "sys.stdin",
                    io.StringIO(
                        '{"id":"first","message":"hello"}\n'
                        '{"id":"second","message":"hello"}\n'
                    ),
                ),
                redirect_stdout(jsonl_output),
            ):
                code = main(
                    [
                        "chat",
                        "--jsonl",
                        "--raw-qwen",
                        "--qwen38-causal-bundle",
                        "/models/qwen.causal",
                        "--qwen38-tokenizer",
                        "/models/tokenizer.json",
                        "--inference-economics-state",
                        str(economics),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(len(jsonl_runtime.model.calls), 2)
            rows = [json.loads(line) for line in jsonl_output.getvalue().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(
                all(
                    row["evidence"]["inference_economics"]["status"] == "recorded"
                    for row in rows
                )
            )
            self.assertEqual(
                InferenceEconomicsLedger(economics).snapshot()["requests"], 3
            )
            self.assertEqual(InferenceActionBank(action_root).snapshot()["requests"], 3)

            interactive_runtime = _Runtime()
            interactive_qwen = _chat(interactive_runtime)
            interactive_output = io.StringIO()
            with (
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=interactive_qwen,
                ),
                patch(
                    "immer.runtimes.qwen3_8.encoding.Qwen38Tokenizer",
                    return_value=interactive_runtime.tokenizer,
                ),
                patch("sys.stdin", io.StringIO("hello\n/stats\n/quit\n")),
                redirect_stdout(interactive_output),
            ):
                code = main(
                    [
                        "chat",
                        "--interactive",
                        "--raw-qwen",
                        "--qwen38-causal-bundle",
                        "/models/qwen.causal",
                        "--qwen38-tokenizer",
                        "/models/tokenizer.json",
                        "--inference-economics-state",
                        str(economics),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(len(interactive_runtime.model.calls), 1)
            self.assertIn("ledger 4 requests", interactive_output.getvalue())
            self.assertEqual(
                InferenceEconomicsLedger(economics).snapshot()["requests"],
                4,
            )
            self.assertEqual(InferenceActionBank(action_root).snapshot()["requests"], 4)

            blocked = Path(temporary) / "blocked-ledger"
            blocked.write_text("not a directory", encoding="utf-8")
            failed_qwen = _chat(_Runtime())
            failed_output = io.StringIO()
            with (
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=failed_qwen,
                ),
                redirect_stdout(failed_output),
            ):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--raw-qwen",
                        "--output",
                        "json",
                        "--qwen38-causal-bundle",
                        "/models/qwen.causal",
                        "--qwen38-tokenizer",
                        "/models/tokenizer.json",
                        "--inference-economics-state",
                        str(blocked),
                    ]
                )
            self.assertEqual(code, 0)
            failed = json.loads(failed_output.getvalue())
            self.assertEqual(
                failed["evidence"]["inference_economics"]["status"],
                "error",
            )

    def test_cli_interactive_reuses_one_loaded_component_for_free_prompts(self) -> None:
        runtime = _Runtime()
        qwen = _chat(runtime)
        output = io.StringIO()
        stream = io.StringIO("hello\nworld\n/clear\nagain\n/quit\n")
        with (
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                return_value=qwen,
            ) as constructor,
            patch(
                "immer.runtimes.qwen3_8.encoding.Qwen38Tokenizer",
                return_value=runtime.tokenizer,
            ),
            patch("sys.stdin", stream),
            redirect_stdout(output),
        ):
            code = main(
                [
                    "chat",
                    "--interactive",
                    "--raw-qwen",
                    "--qwen38-causal-bundle",
                    "/models/qwen.causal",
                    "--qwen38-tokenizer",
                    "/models/tokenizer.json",
                ]
            )

        self.assertEqual(code, 0)
        constructor.assert_called_once()
        self.assertTrue(qwen.closed)
        self.assertEqual(
            output.getvalue(),
            "local answer\nlocal answer\nlocal answer\n",
        )
        self.assertIn(
            Qwen38Tokenizer.render_no_thinking_messages(
                "",
                (
                    ("user", "hello"),
                    ("assistant", "local answer"),
                    ("user", "world"),
                ),
            ),
            runtime.tokenizer.encoded,
        )
        self.assertEqual(
            runtime.tokenizer.encoded[-1],
            Qwen38Tokenizer.render_no_thinking_prompt("", "again"),
        )

    def test_cli_interactive_stats_exposes_live_markov_intelligence(self) -> None:
        runtime = _Runtime()
        qwen = Mock()
        qwen.handle.return_value = Result(
            ExecutionStatus.OK,
            "qwen3.8.causal-chat",
            output="answer",
            evidence={
                "generation": {
                    "forward_passes": 3,
                    "generated_tokens": 8,
                    "seconds": 1.25,
                },
                "runtime_metrics": {
                    "process_peak_rss_bytes": 2 * 1024**3,
                },
                "mlp_page_route": {
                    "request": {
                        "adaptive_width_pages_saved": 128,
                        "coactive_updates": 24,
                        "dynamic_route_calls": 64,
                        "dynamic_route_changes": 7,
                        "energy_feedback_rows": 12,
                        "exact_rows": 12,
                    },
                    "runtime": {
                        "last_width_mean": 128.5,
                        "last_width_min": 96,
                    },
                },
                "q4": {
                    "request": {
                        "page_mlp_prefetch_budget_declines": 1,
                        "page_mlp_prefetch_budget_fraction_sum_ppm": 50_400_000,
                        "page_mlp_prefetch_bytes": 80 * 1024**2,
                        "page_mlp_prefetch_calls": 63,
                        "page_mlp_prefetch_consumed_leases": 186,
                        "page_mlp_prefetch_failures": 0,
                        "page_mlp_prefetch_pages": 8064,
                        "page_mlp_prefetch_requested_pages": 9000,
                        "page_mlp_prefetch_trimmed_pages": 936,
                    }
                },
                "runtime_reward": {
                    "accepted_draft_tokens": 5,
                    "o1_priority": 9.0,
                    "page_actions_saved": 128,
                    "reward": 2.345,
                },
                "draft": {
                    "accepted_draft_tokens": 5,
                    "provider": {
                        "provider_tournament_calls": 2,
                        "provider_tournament_markov_selections": 1,
                        "provider_tournament_mtp_selections": 1,
                        "provider_trace_feedback_tokens": 6,
                        "markov": {
                            "active_dialect_similarity": 0.75,
                            "planner_beam_selections": 1,
                            "planner_council_selections": 0,
                            "planner_phrase_selections": 1,
                            "planner_tournament_calls": 2,
                            "planner_trace_feedback_tokens": 9,
                            "ricci_working_set_builds": 3,
                            "ricci_working_set_selected_episodes": 2,
                            "ricci_working_set_selected_tokens": 61,
                            "ricci_working_set_oldest_age": 7,
                            "ricci_working_set_max_score": 9.5,
                            "recursive_trace_feedback_tokens": 4,
                            "recursive_trace_max_position": 3,
                        },
                        "mtp": {
                            "recursive_trace_feedback_tokens": 5,
                            "recursive_trace_max_position": 4,
                            "teacher_verifications": 2,
                        },
                    },
                },
            },
        )
        output = io.StringIO()
        stream = io.StringIO("hello\n/stats\n/quit\n")
        with (
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                return_value=qwen,
            ),
            patch(
                "immer.runtimes.qwen3_8.encoding.Qwen38Tokenizer",
                return_value=runtime.tokenizer,
            ),
            patch("sys.stdin", stream),
            redirect_stdout(output),
        ):
            code = main(
                [
                    "chat",
                    "--interactive",
                    "--raw-qwen",
                    "--qwen38-causal-bundle",
                    "/models/qwen.causal",
                    "--qwen38-tokenizer",
                    "/models/tokenizer.json",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "answer",
                "[8 tokens · 3 Qwen forwards · 1.25 s · 2.00 GiB peak · "
                "MLP pages 64 dynamic routes, 7 changed, 12 exact rows learned, "
                "24 coactive edges, 128 pages skipped, 12 energy labels, "
                "width 96-128.5 · "
                "Q4 lookahead 63 calls, 8064 pages fully advised, "
                "80.0 MiB advised, 80.0% mean budget, "
                "186 leases consumed, 9000 pages requested, "
                "936 pages budget-trimmed, 1 budget declines · "
                "joint reward 2.35 (draft 5, pages 128, O1 9.00) · "
                "5 accepted draft tokens · "
                "Hybrid 2 provider tournaments Markov1/MTP1, "
                "6 provider counterfactual labels · "
                "Markov 2 tournaments B1/C0/P1, "
                "9 counterfactual labels, 4 deep labels through p3, dialect 0.75, "
                "Ricci PPM 2 episodes/61 tokens, age 7 · "
                "2 free MTP teacher labels · 5 recursive MTP labels through p4]",
            ],
        )
        qwen.close.assert_called_once()

    def test_cli_interactive_stats_handles_flat_and_malformed_providers(self) -> None:
        runtime = _Runtime()
        qwen = Mock()
        qwen.handle.side_effect = (
            Result(
                ExecutionStatus.OK,
                "qwen3.8.causal-chat",
                output="markov",
                evidence={
                    "generation": {"generated_tokens": 1},
                    "draft": {
                        "provider": {
                            "planner_beam_selections": 0,
                            "planner_council_selections": 1,
                            "planner_phrase_selections": 0,
                            "planner_tournament_calls": 1,
                            "planner_trace_feedback_tokens": 2,
                        }
                    },
                },
            ),
            Result(
                ExecutionStatus.OK,
                "qwen3.8.causal-chat",
                output="mtp",
                evidence={
                    "generation": {"generated_tokens": 1},
                    "draft": {"provider": {"teacher_verifications": 3}},
                },
            ),
            Result(
                ExecutionStatus.OK,
                "qwen3.8.causal-chat",
                output="malformed",
                evidence={
                    "generation": {"generated_tokens": 1},
                    "draft": {
                        "accepted_draft_tokens": True,
                        "provider": {
                            "active_dialect_similarity": True,
                            "planner_beam_selections": "0",
                            "planner_tournament_calls": True,
                            "planner_trace_feedback_tokens": False,
                            "ricci_working_set_builds": True,
                            "ricci_working_set_selected_episodes": "2",
                            "ricci_working_set_selected_tokens": False,
                            "ricci_working_set_oldest_age": "7",
                            "recursive_trace_feedback_tokens": "4",
                            "teacher_verifications": True,
                        },
                    },
                },
            ),
        )
        output = io.StringIO()
        stream = io.StringIO("one\n/stats\ntwo\n/stats\nthree\n/stats\n/quit\n")
        with (
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                return_value=qwen,
            ),
            patch(
                "immer.runtimes.qwen3_8.encoding.Qwen38Tokenizer",
                return_value=runtime.tokenizer,
            ),
            patch("sys.stdin", stream),
            redirect_stdout(output),
        ):
            code = main(
                [
                    "chat",
                    "--interactive",
                    "--raw-qwen",
                    "--qwen38-causal-bundle",
                    "/models/qwen.causal",
                    "--qwen38-tokenizer",
                    "/models/tokenizer.json",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "markov",
                "[1 tokens · Markov 1 tournaments B0/C1/P0, 2 counterfactual labels]",
                "mtp",
                "[1 tokens · 3 free MTP teacher labels]",
                "malformed",
                "[1 tokens]",
            ],
        )
        qwen.close.assert_called_once()

    def test_cli_interactive_drops_only_oldest_complete_turns_at_token_limit(
        self,
    ) -> None:
        class SizedTokenizer(_Tokenizer):
            def encode(self, text: str):
                self.encoded.append(text)
                return tuple(range(text.count("<|im_start|>") * 2))

        runtime = _Runtime(tokenizer=SizedTokenizer())
        qwen = _chat(runtime, max_prompt_tokens=8)
        output = io.StringIO()
        stream = io.StringIO("one\ntwo\nthree\n/quit\n")
        with (
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                return_value=qwen,
            ),
            patch(
                "immer.runtimes.qwen3_8.encoding.Qwen38Tokenizer",
                return_value=runtime.tokenizer,
            ),
            patch("sys.stdin", stream),
            redirect_stdout(output),
        ):
            code = main(
                [
                    "chat",
                    "--interactive",
                    "--raw-qwen",
                    "--max-prompt-tokens",
                    "8",
                    "--qwen38-causal-bundle",
                    "/models/qwen.causal",
                    "--qwen38-tokenizer",
                    "/models/tokenizer.json",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(output.getvalue(), "local answer\n" * 3)
        final_prompt = runtime.tokenizer.encoded[-1]
        self.assertNotIn("\none<|im_end|>", final_prompt)
        self.assertIn("\ntwo<|im_end|>", final_prompt)
        self.assertIn("\nthree<|im_end|>", final_prompt)
        self.assertEqual(final_prompt.count("<|im_start|>"), 4)

    def test_context_eviction_never_reenables_single_turn_wrapper(self) -> None:
        class SizedTokenizer(_Tokenizer):
            def encode(self, text: str):
                self.encoded.append(text)
                return tuple(range(text.count("<|im_start|>") * 2))

        runtime = _Runtime(tokenizer=SizedTokenizer())
        qwen = _chat(runtime, max_prompt_tokens=5)
        wrapper = Mock()
        wrapper.handle.side_effect = qwen.handle
        wrapper.close.side_effect = qwen.close
        output = io.StringIO()
        stream = io.StringIO("one\ntwo\n/quit\n")
        with (
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                return_value=qwen,
            ),
            patch(
                "immer.cognition.qwen_fertig_chat.QwenFertigChat",
                return_value=wrapper,
            ),
            patch(
                "immer.runtimes.qwen3_8.encoding.Qwen38Tokenizer",
                return_value=runtime.tokenizer,
            ),
            patch("sys.stdin", stream),
            redirect_stdout(output),
        ):
            code = main(
                [
                    "chat",
                    "--interactive",
                    "--no-ooe-warm",
                    "--max-prompt-tokens",
                    "5",
                    "--qwen38-causal-bundle",
                    "/models/qwen.causal",
                    "--qwen38-tokenizer",
                    "/models/tokenizer.json",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(wrapper.handle.call_count, 1)
        self.assertEqual(len(runtime.model.calls), 2)
        self.assertEqual(output.getvalue(), "local answer\n" * 2)
        self.assertNotIn("\none<|im_end|>", runtime.tokenizer.encoded[-1])
        self.assertIn("\ntwo<|im_end|>", runtime.tokenizer.encoded[-1])

    def test_cli_wires_fast_mlp_root_and_layer_subset(self) -> None:
        qwen = _chat(_Runtime())
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
            return_value=qwen,
        ) as constructor:
            with redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--qwen38-causal-bundle",
                        "/models/qwen.causal",
                        "--qwen38-tokenizer",
                        "/models/tokenizer.json",
                        "--fast-mlp",
                        "/artifacts/qwen-fast",
                        "--fast-mlp-layers",
                        "0,9,18,63",
                        "--fast-mlp-source-budget-mb",
                        "8192",
                        "--fast-mlp-max-resident-mb",
                        "96",
                        "--fast-mlp-online-state",
                        "/state/qwen-fast.json",
                    ]
                )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(options["fast_mlp_root"], "/artifacts/qwen-fast")
        self.assertEqual(options["fast_mlp_active_layers"], (0, 9, 18, 63))
        self.assertEqual(options["fast_mlp_source_budget_mb"], 8192.0)
        self.assertEqual(options["fast_mlp_max_resident_bytes"], 96 * 1024**2)
        self.assertEqual(
            options["fast_mlp_online_state_path"],
            "/state/qwen-fast.json",
        )

    def test_cli_wires_corpus_markov_atlas_without_a_state_file(self) -> None:
        qwen = _chat(_Runtime())
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
            return_value=qwen,
        ) as constructor:
            with redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--qwen38-causal-bundle",
                        "/models/qwen.causal",
                        "--qwen38-tokenizer",
                        "/models/tokenizer.json",
                        "--markov-atlas",
                        "/state/qwen-markov-atlas.bin",
                    ]
                )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(options["draft_mode"], "markov")
        self.assertEqual(
            options["markov_atlas_path"],
            "/state/qwen-markov-atlas.bin",
        )

    def test_cli_wires_o1_markov_retention(self) -> None:
        qwen = _chat(_Runtime())
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
            return_value=qwen,
        ) as constructor:
            with redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--qwen38-causal-bundle",
                        "/models/qwen.causal",
                        "--qwen38-tokenizer",
                        "/models/tokenizer.json",
                        "--markov-o1-retention",
                        "/state/o1-retention.json",
                    ]
                )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(options["draft_mode"], "markov")
        self.assertEqual(
            options["markov_o1_retention_path"],
            "/state/o1-retention.json",
        )

    def test_cli_wires_q4_bank_and_native_threads(self) -> None:
        qwen = _chat(_Runtime())
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
            return_value=qwen,
        ) as constructor:
            with redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--qwen38-causal-bundle",
                        "/models/qwen.causal",
                        "--qwen38-tokenizer",
                        "/models/tokenizer.json",
                        "--qwen38-q4",
                        "/models/qwen-q4",
                        "--q4-threads",
                        "12",
                        "--q4-resident-budget-mb",
                        "12288",
                        "--fast-mlp",
                        "/state/qwen-fast-all64",
                        "--fast-mlp-policy",
                        "structure-edge",
                        "--delta-head-online-state",
                        "/state/qwen-delta-head.json",
                    ]
                )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(options["q4_root"], "/models/qwen-q4")
        self.assertEqual(options["q4_threads"], 12)
        self.assertEqual(options["q4_resident_budget_bytes"], 12288 * 1024**2)
        self.assertEqual(options["fast_mlp_root"], "/state/qwen-fast-all64")
        self.assertEqual(
            options["fast_mlp_active_layers"],
            (*range(18), *range(55, 64)),
        )
        self.assertEqual(options["fast_mlp_selected_block_count"], 32)
        self.assertEqual(
            options["delta_head_state_path"],
            "/state/qwen-delta-head.json",
        )

    def test_cli_rejects_q4_residency_without_local_q4_execution(self) -> None:
        stderr = io.StringIO()
        with (
            patch.dict("os.environ", {}, clear=True),
            redirect_stderr(stderr),
        ):
            code = main(
                [
                    "chat",
                    "hello",
                    "--qwen38-causal-bundle",
                    "/models/qwen.causal",
                    "--qwen38-tokenizer",
                    "/models/tokenizer.json",
                    "--q4-resident-budget-mb",
                    "16384",
                ]
            )

        self.assertEqual(code, 2)
        self.assertIn("requires local Q4 execution", stderr.getvalue())

    def test_cli_wires_and_can_explicitly_disable_attention_output_crystals(
        self,
    ) -> None:
        qwen = _chat(_Runtime())
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
            return_value=qwen,
        ) as constructor:
            with redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--qwen38-causal-bundle",
                        "/models/qwen.causal",
                        "--qwen38-tokenizer",
                        "/models/tokenizer.json",
                        "--qwen38-q4",
                        "/models/qwen-q4",
                        "--attention-output-crystal-state",
                        "/state/attention-output.json",
                        "--raw-qwen",
                        "--no-markov-draft",
                    ]
                )

        self.assertEqual(code, 0)
        self.assertEqual(
            constructor.call_args.kwargs["attention_output_crystal_state_path"],
            "/state/attention-output.json",
        )

        with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            conflict = main(
                [
                    "chat",
                    "hello",
                    "--qwen38-causal-bundle",
                    "/models/qwen.causal",
                    "--qwen38-tokenizer",
                    "/models/tokenizer.json",
                    "--qwen38-q4",
                    "/models/qwen-q4",
                    "--attention-output-crystal-state",
                    "/state/attention-output.json",
                    "--no-attention-output-crystal",
                    "--raw-qwen",
                    "--no-markov-draft",
                ]
            )
        self.assertEqual(conflict, 2)

    def test_cli_wires_private_layer_transition_state_and_error_budget(self) -> None:
        qwen = _chat(_Runtime())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "layer-63.bin"
            state.write_bytes(b"sealed-fixture")
            atlas = root / "atlas"
            compute = root / "compute"
            identity = _layer_transition_identity(
                atlas=_semantic_atlas_authority(atlas),
                graph=_compute_graph_authority(compute),
            )
            with (
                patch(
                    "immer.runtimes.qwen3_8.layer_transition_crystal."
                    "LayerTransitionCrystalBank.load",
                    return_value=SimpleNamespace(identity=identity, crystals=()),
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=qwen,
                ) as constructor,
                redirect_stdout(io.StringIO()),
            ):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--qwen38-causal-bundle",
                        "/models/qwen.causal",
                        "--qwen38-tokenizer",
                        "/models/tokenizer.json",
                        "--qwen38-q4",
                        "/models/qwen-q4",
                        "--layer-transition-crystal-state",
                        str(state),
                        "--layer-transition-crystal-atlas",
                        str(atlas),
                        "--layer-transition-crystal-compute-root",
                        str(compute),
                        "--layer-transition-crystal-max-error-radius",
                        "0.25",
                        "--raw-qwen",
                        "--no-markov-draft",
                    ]
                )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(
            options["layer_transition_crystal_state_path"],
            str(state),
        )
        self.assertEqual(
            options["layer_transition_crystal_atlas_path"],
            str(atlas),
        )
        self.assertEqual(
            options["layer_transition_crystal_compute_root"],
            str(compute),
        )
        self.assertEqual(
            options["layer_transition_crystal_max_error_radius"],
            0.25,
        )
        with (
            redirect_stderr(io.StringIO()),
            redirect_stdout(io.StringIO()),
            self.assertRaises(SystemExit) as invalid_radius,
        ):
            main(
                [
                    "chat",
                    "hello",
                    "--qwen38-causal-bundle",
                    "/models/qwen.causal",
                    "--qwen38-tokenizer",
                    "/models/tokenizer.json",
                    "--qwen38-q4",
                    "/models/qwen-q4",
                    "--layer-transition-crystal-max-error-radius",
                    "nan",
                    "--raw-qwen",
                    "--no-markov-draft",
                ]
            )
        self.assertEqual(invalid_radius.exception.code, 2)

        with tempfile.TemporaryDirectory() as temporary:
            state_without_authorities = Path(temporary) / "layer-63.bin"
            state_without_authorities.write_bytes(b"sealed-fixture")
            with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                missing_authority = main(
                    [
                        "chat",
                        "hello",
                        "--qwen38-causal-bundle",
                        "/models/qwen.causal",
                        "--qwen38-tokenizer",
                        "/models/tokenizer.json",
                        "--qwen38-q4",
                        "/models/qwen-q4",
                        "--layer-transition-crystal-state",
                        str(state_without_authorities),
                        "--raw-qwen",
                        "--no-markov-draft",
                    ]
                )
        self.assertEqual(missing_authority, 2)

    def test_cli_wires_delta_heads_without_legacy_fast_mlp(self) -> None:
        qwen = _chat(_Runtime())
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
            return_value=qwen,
        ) as constructor:
            with redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--qwen38-causal-bundle",
                        "/models/qwen.causal",
                        "--qwen38-tokenizer",
                        "/models/tokenizer.json",
                        "--qwen38-q4",
                        "/models/qwen-q4",
                        "--delta-head-online-state",
                        "/state/qwen-delta-head.json",
                        "--delta-head-layers",
                        "0,2,4",
                    ]
                )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertIsNone(options["fast_mlp_root"])
        self.assertEqual(
            options["delta_head_state_path"],
            "/state/qwen-delta-head.json",
        )
        self.assertEqual(options["delta_head_active_layers"], (0, 2, 4))

    def test_cli_wires_and_can_disable_mlp_page_coordinates(self) -> None:
        qwen = _chat(_Runtime())
        common = [
            "chat",
            "hello",
            "--qwen38-causal-bundle",
            "/models/qwen.causal",
            "--qwen38-tokenizer",
            "/models/tokenizer.json",
            "--qwen38-q4",
            "/models/qwen-q4",
            "--mlp-page-state",
            "/state/pages.json",
            "--mlp-page-coordinate-state",
            "/state/mlp-coordinate.json",
            "--raw-qwen",
            "--no-markov-draft",
        ]
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
            return_value=qwen,
        ) as constructor:
            with redirect_stdout(io.StringIO()):
                code = main(common)

        self.assertEqual(code, 0)
        self.assertEqual(
            constructor.call_args.kwargs["mlp_page_coordinate_state_path"],
            "/state/mlp-coordinate.json",
        )

        with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            conflict = main([*common, "--no-mlp-page-coordinate"])
            missing_router = main(
                [
                    item
                    for item in common
                    if item not in {"--mlp-page-state", "/state/pages.json"}
                ]
            )
        self.assertEqual(conflict, 2)
        self.assertEqual(missing_router, 2)

    def test_cli_wires_native_prefix_sinkhorn_into_the_qwen_runtime(self) -> None:
        qwen = _chat(_Runtime())
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
            return_value=qwen,
        ) as constructor:
            with redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--qwen38-causal-bundle",
                        "/models/qwen.causal",
                        "--qwen38-tokenizer",
                        "/models/tokenizer.json",
                        "--qwen38-q4",
                        "/models/qwen-q4",
                        "--prefix-sinkhorn",
                        "--raw-qwen",
                        "--no-markov-draft",
                    ]
                )

        self.assertEqual(code, 0)
        intervention = constructor.call_args.kwargs["native_head_crsa"]
        self.assertIsInstance(intervention, Qwen38NativeHeadCrsa)
        self.assertEqual(intervention.layer, 27)
        self.assertEqual(intervention.head_indices, (2, 8, 14, 20))
        self.assertEqual(intervention.alpha, 1.0)
        self.assertTrue(intervention.replace_base_softmax)

    def test_cli_defaults_canonical_local_q4_to_prefix_sinkhorn(self) -> None:
        qwen = _chat(_Runtime())
        with tempfile.TemporaryDirectory() as temporary:
            deployed = Path(temporary) / "deployed"
            (deployed / "causal" / "q4-base-v2").mkdir(parents=True)
            with (
                patch.dict("os.environ", {}, clear=True),
                patch("immer.cli._QWEN38_DEPLOYMENT_ROOT", deployed),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=qwen,
                ) as constructor,
                redirect_stdout(io.StringIO()),
            ):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--raw-qwen",
                        "--no-markov-draft",
                    ]
                )

        self.assertEqual(code, 0)
        intervention = constructor.call_args.kwargs["native_head_crsa"]
        self.assertIsInstance(intervention, Qwen38NativeHeadCrsa)
        self.assertEqual(intervention.alpha, 1.0)
        self.assertTrue(intervention.replace_base_softmax)

    def test_cli_no_prefix_sinkhorn_overrides_canonical_default(self) -> None:
        qwen = _chat(_Runtime())
        with tempfile.TemporaryDirectory() as temporary:
            deployed = Path(temporary) / "deployed"
            (deployed / "causal" / "q4-base-v2").mkdir(parents=True)
            with (
                patch.dict("os.environ", {}, clear=True),
                patch("immer.cli._QWEN38_DEPLOYMENT_ROOT", deployed),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=qwen,
                ) as constructor,
                redirect_stdout(io.StringIO()),
            ):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--no-prefix-sinkhorn",
                        "--raw-qwen",
                        "--no-markov-draft",
                    ]
                )

        self.assertEqual(code, 0)
        self.assertIsNone(constructor.call_args.kwargs["native_head_crsa"])

    def test_cli_noncanonical_local_q4_keeps_prefix_sinkhorn_opt_in(self) -> None:
        qwen = _chat(_Runtime())
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
            return_value=qwen,
        ) as constructor:
            with redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--qwen38-causal-bundle",
                        "/models/qwen.causal",
                        "--qwen38-tokenizer",
                        "/models/tokenizer.json",
                        "--qwen38-q4",
                        "/models/qwen-q4",
                        "--raw-qwen",
                        "--no-markov-draft",
                    ]
                )

        self.assertEqual(code, 0)
        self.assertIsNone(constructor.call_args.kwargs["native_head_crsa"])

    def test_cli_prefix_sinkhorn_overrides_are_mutually_exclusive(self) -> None:
        with (
            redirect_stderr(io.StringIO()),
            redirect_stdout(io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            main(
                [
                    "chat",
                    "hello",
                    "--prefix-sinkhorn",
                    "--no-prefix-sinkhorn",
                ]
            )
        self.assertEqual(raised.exception.code, 2)

    def test_cli_wires_exact_head_index_root(self) -> None:
        qwen = _chat(_Runtime())
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
            return_value=qwen,
        ) as constructor:
            with redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--qwen38-causal-bundle",
                        "/models/qwen.causal",
                        "--qwen38-tokenizer",
                        "/models/tokenizer.json",
                        "--exact-head",
                        "/artifacts/qwen-head",
                        "--head-block-rows",
                        "64",
                        "--exact-head-max-mb",
                        "96",
                    ]
                )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(options["exact_head_root"], "/artifacts/qwen-head")
        self.assertEqual(options["head_block_rows"], 64)
        self.assertEqual(options["exact_head_max_bytes"], 96 * 1024**2)

    def test_exact_head_receipt_reaches_general_chat_evidence(self) -> None:
        runtime = _Runtime()
        runtime.exact_head_receipt = dict(_EXACT_HEAD_RECEIPT)
        runtime.exact_head_index = _ExactHeadMetrics()
        result = _chat(
            runtime,
            exact_head_root="/artifacts/qwen-head",
        ).handle(Request("chat", "hello"))

        self.assertTrue(result.ok)
        exact = result.evidence["exact_head"]
        self.assertEqual(
            {key: exact[key] for key in _EXACT_HEAD_RECEIPT},
            _EXACT_HEAD_RECEIPT,
        )
        self.assertEqual(exact["request"]["calls"], 1)
        self.assertEqual(exact["request"]["pages_pruned"], 2)
        self.assertEqual(exact["request"]["rows_pruned"], 8)

        attachments = []
        runtime.model.pager.attach_exact_head_index = attachments.append
        disabled = InferenceActionDirective(
            question_sha256=hashlib.sha256(b"hello").hexdigest(),
            runtime_profile_sha256="2" * 64,
            primary_actions=("qwen_target",),
            fallback_actions=("qwen_target",),
            draft_enabled=None,
            source_signature_sha256s=("8" * 64,),
            support=1,
            saved_qwen_forwards=0,
            disabled_actions=("lm_head_coordinate",),
        )
        disabled_result = _chat(
            runtime,
            exact_head_root="/artifacts/qwen-head",
        ).handle(
            Request(
                "chat",
                "hello",
                {QWEN38_INFERENCE_ACTION_METADATA: disabled.to_document()},
            )
        )
        self.assertTrue(disabled_result.ok, disabled_result.reason)
        self.assertEqual(attachments, [None, runtime.exact_head_index])
        self.assertFalse(
            disabled_result.evidence["inference_action_directive"]["applied"][
                "lm_head_coordinate"
            ]
        )

    def test_cli_wires_persistent_markov_drafting_without_bundle(self) -> None:
        qwen = _chat(_Runtime())
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
            return_value=qwen,
        ) as constructor:
            with redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--qwen38-causal-bundle",
                        "/models/qwen.causal",
                        "--qwen38-tokenizer",
                        "/models/tokenizer.json",
                        "--draft-mode",
                        "markov",
                        "--markov-draft-state",
                        "/state/qwen-markov.bin",
                    ]
                )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(options["draft_mode"], "markov")
        self.assertEqual(
            options["markov_draft_state_path"],
            "/state/qwen-markov.bin",
        )
        self.assertIsNone(options["draft_bundle_path"])


if __name__ == "__main__":
    unittest.main()
