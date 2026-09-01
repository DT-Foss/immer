from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
import warnings
import weakref

import numpy as np
import torch
from safetensors.torch import save_file

import immer.runtimes.qwen3_8.model as qwen_model_module
from immer.knowledge.streamer import Streamer
from immer.runtimes.qwen3_8.attention_output_crystal import (
    AttentionOutputCrystalBank,
    AttentionOutputCrystalIdentity,
)
from immer.runtimes.qwen3_8.config import Qwen38Config
from immer.runtimes.qwen3_8.draft_verification import (
    DRAFT_VERIFICATION_SCHEMA,
    Qwen38DraftVerifier,
)
from immer.runtimes.qwen3_8.model import (
    LAYER_BOUNDARY_STAGES,
    Qwen38RuntimeError,
    StreamedQwen38,
)
from immer.runtimes.qwen3_8.kernels import AttentionState, DeltaNetState
from immer.runtimes.qwen3_8.layer_mlp_crystal import (
    Layer63MlpResidualCrystalBank,
    Layer63MlpResidualCrystalIdentity,
)
from immer.runtimes.qwen3_8.layer_transition_crystal import (
    LayerTransitionCrystalBank,
    LayerTransitionCrystalIdentity,
    LayerTransitionProjectionIdentity,
)
from immer.runtimes.qwen3_8.mlp_page_coordinate import (
    MlpPageCoordinateBank,
    MlpPageCoordinateIdentity,
)
from immer.runtimes.qwen3_8.pager import Qwen38WeightPager
from immer.runtimes.qwen3_8.semantic_state_cache import SemanticStateAnchorCache
from immer.runtimes.qwen3_8.graft import (
    STABLE_GRAFT_EVIDENCE_SCHEMA,
    Qwen38StableCrsaGraft,
)


def _tiny_config_mapping() -> dict[str, object]:
    return {
        "model_type": "qwen3_5_text",
        "vocab_size": 32,
        "hidden_size": 12,
        "intermediate_size": 16,
        "num_hidden_layers": 4,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 6,
        "partial_rotary_factor": 1.0,
        "full_attention_interval": 4,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 3,
        "linear_value_head_dim": 3,
        "linear_conv_kernel_dim": 4,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 1.0,
            "mrope_interleaved": True,
            "mrope_section": [1, 1, 1],
        },
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "output_gate_type": "swish",
        "hidden_act": "silu",
        "dtype": "bfloat16",
        "mamba_ssm_dtype": "float32",
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "tie_word_embeddings": False,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }


def _tiny_config() -> Qwen38Config:
    return Qwen38Config.from_mapping(
        _tiny_config_mapping(),
        require_official=False,
    )


def _tiny_tied_config() -> Qwen38Config:
    mapping = _tiny_config_mapping()
    mapping["tie_word_embeddings"] = True
    return Qwen38Config.from_mapping(mapping, require_official=False)


def _native_tiny_config(*, full_attention_interval: int = 4) -> Qwen38Config:
    mapping = _tiny_config_mapping()
    mapping["num_hidden_layers"] = 28
    mapping["full_attention_interval"] = full_attention_interval
    mapping["layer_types"] = [
        (
            "full_attention"
            if (layer + 1) % full_attention_interval == 0
            else "linear_attention"
        )
        for layer in range(28)
    ]
    mapping["num_attention_heads"] = 24
    mapping["num_key_value_heads"] = 4
    return Qwen38Config.from_mapping(mapping, require_official=False)


def _official_topology_tiny_config() -> Qwen38Config:
    """Keep official depth/head topology with tiny activation widths."""

    mapping = _tiny_config_mapping()
    mapping["num_hidden_layers"] = 64
    mapping["layer_types"] = [
        "full_attention" if (layer + 1) % 4 == 0 else "linear_attention"
        for layer in range(64)
    ]
    mapping["num_attention_heads"] = 24
    mapping["num_key_value_heads"] = 4
    return Qwen38Config.from_mapping(mapping, require_official=False)


def _matrix(rows: int, columns: int, generator: torch.Generator) -> torch.Tensor:
    return (0.04 * torch.randn(rows, columns, generator=generator)).to(torch.bfloat16)


def _tiny_weights(config: Qwen38Config) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(31)
    tensors: dict[str, torch.Tensor] = {
        "model.language_model.embed_tokens.weight": _matrix(
            config.vocab_size, config.dim, generator
        ),
        "model.language_model.norm.weight": torch.zeros(
            config.dim, dtype=torch.bfloat16
        ),
    }
    if not config.tie_word_embeddings:
        tensors["lm_head.weight"] = _matrix(config.vocab_size, config.dim, generator)
    for layer in range(config.n_layers):
        base = f"model.language_model.layers.{layer}"
        tensors[f"{base}.input_layernorm.weight"] = torch.zeros(
            config.dim, dtype=torch.bfloat16
        )
        tensors[f"{base}.post_attention_layernorm.weight"] = torch.zeros(
            config.dim, dtype=torch.bfloat16
        )
        tensors[f"{base}.mlp.gate_proj.weight"] = _matrix(
            config.intermediate_size, config.dim, generator
        )
        tensors[f"{base}.mlp.up_proj.weight"] = _matrix(
            config.intermediate_size, config.dim, generator
        )
        tensors[f"{base}.mlp.down_proj.weight"] = _matrix(
            config.dim, config.intermediate_size, generator
        )
        if config.is_full_attention(layer):
            attn = f"{base}.self_attn"
            tensors[f"{attn}.q_proj.weight"] = _matrix(
                2 * config.n_heads * config.head_dim, config.dim, generator
            )
            tensors[f"{attn}.k_proj.weight"] = _matrix(
                config.n_kv_heads * config.head_dim, config.dim, generator
            )
            tensors[f"{attn}.v_proj.weight"] = _matrix(
                config.n_kv_heads * config.head_dim, config.dim, generator
            )
            tensors[f"{attn}.o_proj.weight"] = _matrix(
                config.dim, config.n_heads * config.head_dim, generator
            )
            tensors[f"{attn}.q_norm.weight"] = torch.zeros(
                config.head_dim, dtype=torch.bfloat16
            )
            tensors[f"{attn}.k_norm.weight"] = torch.zeros(
                config.head_dim, dtype=torch.bfloat16
            )
        else:
            attn = f"{base}.linear_attn"
            key_features = config.linear_num_key_heads * config.linear_key_head_dim
            value_features = (
                config.linear_num_value_heads * config.linear_value_head_dim
            )
            conv_features = 2 * key_features + value_features
            tensors[f"{attn}.in_proj_qkv.weight"] = _matrix(
                conv_features, config.dim, generator
            )
            tensors[f"{attn}.in_proj_z.weight"] = _matrix(
                value_features, config.dim, generator
            )
            tensors[f"{attn}.in_proj_a.weight"] = _matrix(
                config.linear_num_value_heads, config.dim, generator
            )
            tensors[f"{attn}.in_proj_b.weight"] = _matrix(
                config.linear_num_value_heads, config.dim, generator
            )
            tensors[f"{attn}.conv1d.weight"] = (
                0.08
                * torch.randn(
                    conv_features,
                    1,
                    config.linear_conv_kernel_dim,
                    generator=generator,
                )
            ).to(torch.bfloat16)
            tensors[f"{attn}.A_log"] = torch.tensor([-0.7, 0.1], dtype=torch.bfloat16)
            tensors[f"{attn}.dt_bias"] = torch.tensor([-0.2, 0.3], dtype=torch.bfloat16)
            tensors[f"{attn}.norm.weight"] = torch.ones(
                config.linear_value_head_dim, dtype=torch.bfloat16
            )
            tensors[f"{attn}.out_proj.weight"] = _matrix(
                config.dim, value_features, generator
            )
    return tensors


def _assert_layer_states_equal(
    case: unittest.TestCase,
    actual: tuple[AttentionState | DeltaNetState | None, ...]
    | list[AttentionState | DeltaNetState | None],
    expected: tuple[AttentionState | DeltaNetState | None, ...]
    | list[AttentionState | DeltaNetState | None],
) -> None:
    case.assertEqual(len(actual), len(expected))
    for left, right in zip(actual, expected, strict=True):
        case.assertIs(type(left), type(right))
        if isinstance(left, AttentionState) and isinstance(right, AttentionState):
            case.assertTrue(torch.equal(left.key, right.key))
            case.assertTrue(torch.equal(left.value, right.value))
            if left.crsa_log_usage is None or right.crsa_log_usage is None:
                case.assertIsNone(left.crsa_log_usage)
                case.assertIsNone(right.crsa_log_usage)
            else:
                case.assertTrue(torch.equal(left.crsa_log_usage, right.crsa_log_usage))
        elif isinstance(left, DeltaNetState) and isinstance(right, DeltaNetState):
            case.assertTrue(torch.equal(left.conv, right.conv))
            case.assertTrue(torch.equal(left.recurrent, right.recurrent))


def _clone_layer_states(
    states: list[AttentionState | DeltaNetState | None],
) -> tuple[AttentionState | DeltaNetState | None, ...]:
    cloned: list[AttentionState | DeltaNetState | None] = []
    for state in states:
        if isinstance(state, AttentionState):
            cloned.append(
                AttentionState(
                    key=state.key.clone(),
                    value=state.value.clone(),
                    crsa_log_usage=(
                        None
                        if state.crsa_log_usage is None
                        else state.crsa_log_usage.clone()
                    ),
                )
            )
        elif isinstance(state, DeltaNetState):
            cloned.append(
                DeltaNetState(
                    conv=state.conv.clone(), recurrent=state.recurrent.clone()
                )
            )
        else:
            cloned.append(None)
    return tuple(cloned)


def _weak_layer_state_tensor_refs(
    states: list[AttentionState | DeltaNetState | None],
) -> tuple[tuple[weakref.ReferenceType[torch.Tensor], ...], ...]:
    refs: list[tuple[weakref.ReferenceType[torch.Tensor], ...]] = []
    for state in states:
        if isinstance(state, AttentionState):
            tensors = (state.key, state.value)
            if state.crsa_log_usage is not None:
                tensors = (*tensors, state.crsa_log_usage)
        elif isinstance(state, DeltaNetState):
            tensors = (state.conv, state.recurrent)
        else:
            tensors = ()
        refs.append(tuple(weakref.ref(tensor) for tensor in tensors))
    return tuple(refs)


class Qwen38ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = _tiny_config()
        save_file(_tiny_weights(self.config), self.root / "model.safetensors")
        self.source = Streamer.from_local(self.root, budget_mb=20, use_cache=False)
        self.pager = Qwen38WeightPager(
            self.source,
            device="cpu",
            compute_dtype="float32",
            max_resident_bytes=2 * 1024**2,
        )
        self.model = StreamedQwen38(
            self.config, self.pager, max_batch_size=3, max_seq_len=32
        )

    def tearDown(self) -> None:
        self.pager.close()
        self.source.close()
        self.temporary.cleanup()

    def test_preflight_covers_only_the_complete_text_stack(self) -> None:
        report = self.model.checkpoint_preflight()
        self.assertEqual(report["required_tensors"], 56)
        self.assertEqual(report["linear_attention_layers"], 3)
        self.assertEqual(report["full_attention_layers"], 1)
        self.assertTrue(report["vision_excluded"])
        self.assertTrue(report["mtp_excluded"])

    def test_component_timing_counters_cover_physical_forward_paths(self) -> None:
        hidden = torch.randn((1, 1, self.config.dim))
        mask = torch.ones((1, 1), dtype=torch.bool)
        clock = iter(
            (
                100,
                110,
                200,
                250,
                300,
                307,
                400,
                480,
                500,
                511,
                600,
                670,
                700,
                709,
                800,
                890,
            )
        )

        with mock.patch.object(
            qwen_model_module.time,
            "perf_counter_ns",
            side_effect=lambda: next(clock),
        ):
            self.model._forward_layer(
                hidden,
                layer=0,
                token_mask=mask,
                state=None,
                start_pos=0,
                stateful=True,
            )
            self.model._forward_layer(
                hidden,
                layer=3,
                token_mask=mask,
                state=None,
                start_pos=0,
                stateful=True,
            )

        metrics = self.model.component_timing_metrics()
        self.assertEqual(
            metrics["schema"],
            "immer.qwen3.8-component-timing-counters/v1",
        )
        self.assertEqual(metrics["clock"], "time.perf_counter_ns")
        self.assertEqual(metrics["unit"], "nanoseconds")
        self.assertEqual(metrics["accounting_failures"], 0)
        self.assertEqual(
            metrics["components"],
            {
                "full_attention_core": {"calls": 1, "nanoseconds": 70},
                "deltanet_core": {"calls": 1, "nanoseconds": 50},
                "mlp_core": {"calls": 2, "nanoseconds": 170},
                "layer_transition_crystal": {
                    "calls": 2,
                    "nanoseconds": 21,
                },
                "layer_mlp_crystal": {"calls": 2, "nanoseconds": 16},
            },
        )

    def test_component_clock_failure_cannot_change_forward_execution(self) -> None:
        hidden = torch.randn((1, 1, self.config.dim))
        mask = torch.ones((1, 1), dtype=torch.bool)

        with mock.patch.object(
            qwen_model_module.time,
            "perf_counter_ns",
            side_effect=RuntimeError("clock unavailable"),
        ):
            output, state = self.model._forward_layer(
                hidden,
                layer=0,
                token_mask=mask,
                state=None,
                start_pos=0,
                stateful=True,
            )

        self.assertEqual(tuple(output.shape), tuple(hidden.shape))
        self.assertIsInstance(state, DeltaNetState)
        metrics = self.model.component_timing_metrics()
        self.assertEqual(metrics["accounting_failures"], 4)
        self.assertTrue(
            all(
                row == {"calls": 0, "nanoseconds": 0}
                for row in metrics["components"].values()
            )
        )

    def test_layer_endpoint_capture_keeps_fused_mlp_eligible(self) -> None:
        self.model.layer_boundary_observer = lambda *_args: None
        self.model.layer_boundary_layers = (3,)
        self.model.layer_boundary_stages = ("layer.input", "layer.output")
        self.assertFalse(self.model._requires_unfused_mlp_boundaries(3))
        self.model.layer_boundary_stages = (
            "layer.input",
            "mlp.output",
            "layer.output",
        )
        self.assertTrue(self.model._requires_unfused_mlp_boundaries(3))
        self.assertFalse(self.model._requires_unfused_mlp_boundaries(2))

    def test_tied_qwen35_head_and_f32_controls_execute_without_lm_head(self) -> None:
        config = _tiny_tied_config()
        tensors = _tiny_weights(config)
        self.assertNotIn("lm_head.weight", tensors)
        for layer in range(config.n_layers):
            if config.is_full_attention(layer):
                continue
            base = f"model.language_model.layers.{layer}.linear_attn"
            tensors[f"{base}.A_log"] = tensors[f"{base}.A_log"].float()
            tensors[f"{base}.norm.weight"] = tensors[f"{base}.norm.weight"].float()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_file(tensors, root / "model.safetensors")
            source = Streamer.from_local(root, budget_mb=20, use_cache=False)
            pager = Qwen38WeightPager(
                source,
                device="cpu",
                compute_dtype="float32",
                max_resident_bytes=2 * 1024**2,
            )
            model = StreamedQwen38(config, pager, max_batch_size=1, max_seq_len=8)
            try:
                report = model.checkpoint_preflight()
                self.assertEqual(report["required_tensors"], 55)
                self.assertTrue(report["tie_word_embeddings"])
                self.assertEqual(
                    report["output_head_tensor"],
                    "model.language_model.embed_tokens.weight",
                )
                generated, evidence = model.generate_greedy(
                    [[1]], max_new_tokens=1, head_block_rows=8
                )
                self.assertEqual(len(generated), 1)
                self.assertEqual(evidence.generated_token_ids, generated)
            finally:
                pager.close()
                source.close()

    def test_complete_prefill_is_finite_and_accounts_every_linear(self) -> None:
        final, evidence = self.model.forward_prefill(torch.tensor([[1, 4, 9]]))
        self.assertEqual(tuple(final.shape), (1, 3, self.config.dim))
        self.assertTrue(torch.isfinite(final).all())
        self.assertEqual(evidence.layers_executed, 4)
        self.assertEqual(evidence.linear_calls, 31)
        self.assertGreater(evidence.source_body_bytes, 0)

    def test_grouped_token_row_projections_are_bit_exact(self) -> None:
        generator = torch.Generator().manual_seed(381)
        rows = tuple(
            torch.randn((1, 1, self.config.dim), generator=generator) for _ in range(3)
        )
        groups = (
            tuple(
                f"model.language_model.layers.3.self_attn.{name}_proj"
                for name in ("q", "k", "v")
            ),
            tuple(
                f"model.language_model.layers.0.linear_attn.in_proj_{name}"
                for name in ("qkv", "z", "b", "a")
            ),
        )

        for names in groups:
            with self.subTest(names=names):
                expected = tuple(
                    self.model._linear_token_rows(rows, name) for name in names
                )
                actual = self.model._linear_group_token_rows(rows, names)
                self.assertEqual(len(actual), len(expected))
                for actual_projection, expected_projection in zip(
                    actual,
                    expected,
                    strict=True,
                ):
                    self.assertEqual(len(actual_projection), len(rows))
                    self.assertTrue(
                        all(
                            torch.equal(actual_row, expected_row)
                            for actual_row, expected_row in zip(
                                actual_projection,
                                expected_projection,
                                strict=True,
                            )
                        )
                    )

    def test_optional_sparse_mlp_mount_is_decode_only_and_observer_safe(self) -> None:
        class SparseStub:
            def __init__(self) -> None:
                self.calls = []
                self.trace = {"kind": "sparse-trace"}

            def supports_layer(self, layer: int) -> bool:
                return layer == 1

            def snapshot_identity(self, *, transport_neutral: bool = False):
                return {
                    "layers": [1],
                    "schema": "test-sparse-executor/v1",
                    "transport_neutral": transport_neutral,
                }

            def execute(self, hidden: torch.Tensor, *, layer: int):
                self.calls.append(("one", layer, tuple(hidden.shape)))
                return torch.full_like(hidden, 0.125), self.trace

            def execute_many(self, hidden, *, layer: int):
                rows = tuple(hidden)
                self.calls.append(("many", layer, len(rows)))
                return tuple(torch.full_like(row, 0.25) for row in rows), self.trace

        sparse = SparseStub()
        model = StreamedQwen38(
            self.config,
            self.pager,
            mlp_sparse_executor=sparse,
            max_batch_size=3,
            max_seq_len=32,
        )
        hidden = torch.ones(1, 1, self.config.dim)
        before = self.pager.metrics()["linear_calls"]
        output = model._mlp(hidden, layer=1)
        self.assertTrue(torch.equal(output, torch.full_like(hidden, 0.125)))
        self.assertEqual(self.pager.metrics()["linear_calls"], before)
        self.assertIs(model.mlp_sparse_last_trace, sparse.trace)

        prefill = torch.ones(1, 3, self.config.dim)
        model._mlp(prefill, layer=1)
        self.assertEqual(self.pager.metrics()["linear_calls"] - before, 3)
        self.assertIsNone(model.mlp_sparse_last_trace)

        token_rows = (hidden.clone(), hidden.clone())
        rows = model._mlp_token_rows(token_rows, layer=1)
        self.assertTrue(
            all(torch.equal(row, torch.full_like(row, 0.25)) for row in rows)
        )
        self.assertEqual(sparse.calls[-1], ("many", 1, 2))

        observed = []
        guarded = StreamedQwen38(
            self.config,
            self.pager,
            mlp_sparse_executor=sparse,
            layer_boundary_observer=lambda _layer, stage, _value: observed.append(
                stage
            ),
            layer_boundary_stages=("mlp.gate", "mlp.output"),
            layer_boundary_layers=(1,),
            max_batch_size=3,
            max_seq_len=32,
        )
        calls = len(sparse.calls)
        guarded._mlp(hidden, layer=1)
        self.assertEqual(len(sparse.calls), calls)
        self.assertEqual(observed, ["mlp.gate", "mlp.output"])

        with self.assertRaisesRegex(TypeError, "mlp_sparse_executor"):
            StreamedQwen38(
                self.config,
                self.pager,
                mlp_sparse_executor=object(),
                max_seq_len=32,
            )

    def test_markov_page_route_replaces_full_mlp_execution(self) -> None:
        class Q4Stub:
            @staticmethod
            def has(_name: str) -> bool:
                return True

            @staticmethod
            def mlp(*_args, **_kwargs):
                raise AssertionError("pager mock owns full MLP execution")

            @staticmethod
            def mlp_selected_pages(*_args, **_kwargs):
                raise AssertionError("pager mock owns selected MLP execution")

        class PageRouter:
            route_width = 4
            page_count = 4

            def __init__(self) -> None:
                self.ready = False
                self.prepared = []
                self.routed = []
                self.exact = []
                self.advanced = []

            def prepare(self, layer: int):
                self.prepared.append(layer)
                return SimpleNamespace(ready=self.ready, page_ids=(3, 1))

            def observe_exact_batch(self, layer: int, ids, scores, totals) -> None:
                self.exact.append((layer, ids.clone(), scores.clone(), totals.clone()))
                self.ready = True

            def advance_selected(self, layer: int, ids, *, row_count: int = 1) -> None:
                self.advanced.append((layer, tuple(ids), row_count))

            def begin_transaction(self) -> None:
                pass

            @staticmethod
            def begin_exact_wave(_layer: int) -> None:
                pass

            @staticmethod
            def compile_routes() -> None:
                pass

            def commit_transaction(self) -> None:
                pass

            def rollback_transaction(self) -> None:
                pass

            def reset_session(self) -> None:
                pass

            def route(self, layer: int, *, row_count: int):
                self.routed.append((layer, row_count))
                page_ids = (3, 1) if row_count == 1 else (2, 0, 3)
                return SimpleNamespace(ready=self.ready, page_ids=page_ids)

            @staticmethod
            def snapshot_identity():
                return {"schema": "test-page-router/v1"}

            @staticmethod
            def metrics():
                return {}

        router = PageRouter()
        route_config_mapping = _tiny_config_mapping()
        route_config_mapping["intermediate_size"] = 256
        route_config = Qwen38Config.from_mapping(
            route_config_mapping,
            require_official=False,
        )
        model = StreamedQwen38(
            route_config,
            self.pager,
            mlp_page_router=router,
            max_batch_size=3,
            max_seq_len=32,
        )
        hidden = torch.ones(1, 3, self.config.dim)
        full_output = torch.full_like(hidden, 0.5)
        page_ids = torch.arange(4, dtype=torch.int64).expand(1, 3, 4).clone()
        page_scores = torch.tensor(
            [[[8.0, 4.0, 2.0, 1.0]] * 3],
            dtype=torch.float64,
        )
        page_total_scores = torch.full((1, 3), 16.0, dtype=torch.float64)
        selected_output = torch.full((1, 1, self.config.dim), 0.25)
        original_q4 = self.pager.q4_bank
        original_dtype = self.pager.compute_dtype
        self.pager.q4_bank = Q4Stub()
        self.pager.compute_dtype = torch.bfloat16
        try:
            with (
                mock.patch.object(
                    self.pager,
                    "mlp",
                    return_value=(
                        full_output,
                        page_ids,
                        page_scores,
                        page_total_scores,
                    ),
                ) as full,
                mock.patch.object(
                    self.pager,
                    "mlp_selected_pages",
                    return_value=selected_output,
                ) as selected,
            ):
                actual_full = model._mlp(hidden, layer=1)
                self.assertTrue(torch.equal(actual_full, full_output))
                self.assertEqual(len(router.exact), 1)
                self.assertTrue(torch.equal(router.exact[0][3], page_total_scores))
                self.assertEqual(router.prepared, [])
                full.assert_called_once()
                self.assertEqual(
                    full.call_args.kwargs["activation_page_topk"],
                    router.route_width,
                )
                selected.assert_not_called()

                one = hidden[:, :1]
                actual_selected = model._mlp(one, layer=1)
                self.assertTrue(torch.equal(actual_selected, selected_output))
                self.assertEqual(full.call_count, 1)
                selected.assert_called_once()
                self.assertEqual(
                    tuple(selected.call_args.args[2].shape),
                    (1, 1, 2),
                )
                self.assertEqual(
                    selected.call_args.args[2].tolist(),
                    [[[3, 1]]],
                )
                self.assertEqual(router.advanced, [(1, (3, 1), 1)])

                rows = (one.clone(), one.clone())
                selected.return_value = torch.full((2, self.config.dim), 0.75)
                outputs = model._mlp_token_rows(rows, layer=1)
                self.assertEqual(len(outputs), 2)
                self.assertTrue(
                    all(torch.equal(row, torch.full_like(row, 0.75)) for row in outputs)
                )
                self.assertEqual(full.call_count, 1)
                self.assertEqual(selected.call_count, 2)
                self.assertEqual(router.routed, [(1, 1), (1, 2)])
                self.assertEqual(
                    tuple(selected.call_args.args[2].shape),
                    (2, 3),
                )
                self.assertEqual(
                    selected.call_args.args[2].tolist(),
                    [[2, 0, 3], [2, 0, 3]],
                )
                self.assertEqual(
                    router.advanced,
                    [(1, (3, 1), 1), (1, (2, 0, 3), 2)],
                )
        finally:
            self.pager.q4_bank = original_q4
            self.pager.compute_dtype = original_dtype

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            StreamedQwen38(
                route_config,
                self.pager,
                mlp_sparse_executor=mock.Mock(
                    execute=mock.Mock(),
                    execute_many=mock.Mock(),
                    supports_layer=mock.Mock(return_value=True),
                    snapshot_identity=mock.Mock(return_value={"layers": [0]}),
                ),
                mlp_page_router=router,
                max_seq_len=32,
            )

    def test_exact_k1_mlp_page_coordinate_replays_the_executed_route(self) -> None:
        class Q4Stub:
            identity = {"manifest_sha256": "c" * 64}

            def __init__(self) -> None:
                self.page_weight_bytes = 0
                base = "model.language_model.layers.1.mlp"
                self.entries = {
                    f"{base}.{name}.weight": SimpleNamespace(payload_bytes=100)
                    for name in ("gate_proj", "up_proj", "down_proj")
                }

            @staticmethod
            def has(_name: str) -> bool:
                return True

            @staticmethod
            def mlp(*_args, **_kwargs):
                raise AssertionError("pager mock owns full MLP execution")

            @staticmethod
            def mlp_selected_pages(*_args, **_kwargs):
                raise AssertionError("pager mock owns selected MLP execution")

            def metrics(self):
                return {"page_mlp_weight_bytes": self.page_weight_bytes}

        class PageRouter:
            route_width = 4
            page_count = 4

            def __init__(self) -> None:
                self.routed = []
                self.replayed = []

            @staticmethod
            def prepare(layer: int):
                return SimpleNamespace(layer=layer, ready=False, page_ids=())

            @staticmethod
            def observe_exact_batch(_layer, _ids, _scores, _totals) -> None:
                pass

            @staticmethod
            def advance_selected(_layer, _ids, *, row_count=1) -> None:
                pass

            @staticmethod
            def begin_transaction() -> None:
                pass

            @staticmethod
            def begin_exact_wave(_layer: int) -> None:
                pass

            @staticmethod
            def compile_routes() -> None:
                pass

            @staticmethod
            def commit_transaction(*, accepted_rows=None) -> None:
                pass

            @staticmethod
            def rollback_transaction() -> None:
                pass

            @staticmethod
            def reset_session() -> None:
                pass

            def route(self, layer: int, *, row_count: int):
                self.routed.append((layer, row_count))
                return SimpleNamespace(
                    ready=True,
                    page_ids=(3, 1),
                    full_page_ids=(3, 1, 2, 0),
                )

            def replay_coordinate(
                self,
                layer,
                full_page_ids,
                *,
                selected_width,
                row_count=1,
            ):
                selected = tuple(full_page_ids)[:selected_width]
                self.replayed.append((layer, selected, row_count))
                return selected

            @staticmethod
            def snapshot_identity():
                return {"schema": "test-page-router/v1"}

            @staticmethod
            def metrics():
                return {}

        router = PageRouter()
        q4 = Q4Stub()
        route_config_mapping = _tiny_config_mapping()
        route_config_mapping["intermediate_size"] = 256
        route_config = Qwen38Config.from_mapping(
            route_config_mapping,
            require_official=False,
        )
        model = StreamedQwen38(
            route_config,
            self.pager,
            mlp_page_router=router,
            max_batch_size=1,
            max_seq_len=16,
        )
        original_q4 = self.pager.q4_bank
        original_dtype = self.pager.compute_dtype
        self.pager.q4_bank = q4
        self.pager.compute_dtype = torch.bfloat16
        hidden = torch.ones(1, 1, self.config.dim, dtype=torch.bfloat16)

        def selected_output(values, _names, _pages):
            q4.page_weight_bytes += 100
            return torch.full_like(values, 0.25)

        try:
            components = model._mlp_page_coordinate_identity_components()
            bank = MlpPageCoordinateBank(
                self.root / "mlp-page-coordinates.json",
                MlpPageCoordinateIdentity(**components),
                max_cells=16,
            )
            model.attach_mlp_page_coordinate_bank(bank)
            delta_router = SimpleNamespace(
                begin_transaction=mock.Mock(),
                commit_transaction=mock.Mock(),
                finalize_transaction=mock.Mock(),
                project=mock.Mock(),
                project_many=mock.Mock(),
                reset_session=mock.Mock(),
                revert_committed_transaction=mock.Mock(),
                rollback_transaction=mock.Mock(),
                snapshot_identity=mock.Mock(return_value={"layers": [0]}),
                supports_layer=mock.Mock(return_value=True),
            )
            with self.assertRaisesRegex(Qwen38RuntimeError, "MLP page bank"):
                model.set_delta_head_router(delta_router)
            with mock.patch.object(
                self.pager,
                "mlp_selected_pages",
                side_effect=selected_output,
            ):
                first_transaction = bank.begin_transaction()
                model._active_mlp_page_coordinate_transaction = first_transaction
                first = model._mlp(hidden, layer=1, absolute_position=5)
                model._active_mlp_page_coordinate_transaction = None
                first_transaction.commit(accepted_end_position=6)

                router.routed.clear()
                second_transaction = bank.begin_transaction()
                model._active_mlp_page_coordinate_transaction = second_transaction
                second = model._mlp(hidden, layer=1, absolute_position=5)
                model._active_mlp_page_coordinate_transaction = None
                second_transaction.commit(accepted_end_position=6)

            self.assertTrue(torch.equal(first, second))
            self.assertEqual(router.routed, [])
            self.assertEqual(router.replayed, [(1, (3, 1), 1)])
            metrics = bank.metrics()
            self.assertEqual(metrics.hit_count, 1)
            self.assertEqual(metrics.physical_pages_saved, 2)
            self.assertEqual(metrics.logical_page_weight_bytes_saved, 200)
        finally:
            model._active_mlp_page_coordinate_transaction = None
            self.pager.q4_bank = original_q4
            self.pager.compute_dtype = original_dtype

    def test_continuation_stage_transactions_page_route_state(self) -> None:
        class PageRouter:
            route_width = 1

            def __init__(self) -> None:
                self.begins = 0
                self.commits = 0
                self.fail_commit = False
                self.rollbacks = 0
                self.resets = 0

            @staticmethod
            def advance_selected(
                _layer: int,
                _ids,
                *,
                row_count: int = 1,
            ) -> None:
                pass

            def begin_transaction(self) -> None:
                self.begins += 1

            @staticmethod
            def begin_exact_wave(_layer: int) -> None:
                pass

            @staticmethod
            def compile_routes() -> None:
                pass

            def commit_transaction(self, *, accepted_rows: int | None = None) -> None:
                if self.fail_commit:
                    raise RuntimeError("page commit failed")
                self.commits += 1

            @staticmethod
            def metrics():
                return {}

            @staticmethod
            def observe_exact_batch(_layer: int, _ids, _scores, _totals) -> None:
                pass

            @staticmethod
            def prepare(layer: int):
                return SimpleNamespace(layer=layer, ready=False, page_ids=())

            def reset_session(self) -> None:
                self.resets += 1

            def rollback_transaction(self) -> None:
                self.rollbacks += 1

            @staticmethod
            def route(layer: int, *, row_count: int):
                return SimpleNamespace(layer=layer, ready=False, page_ids=())

            @staticmethod
            def snapshot_identity():
                return {"schema": "test-page-router/v1"}

        class DeltaRouter:
            def __init__(self, width: int) -> None:
                self.width = width
                self.commits = 0
                self.finalizes = 0
                self.reverts = 0

            @staticmethod
            def supports_layer(layer: int) -> bool:
                return layer == 0

            @staticmethod
            def snapshot_identity(*, transport_neutral: bool = False):
                return {"layers": [0], "transport_neutral": transport_neutral}

            @staticmethod
            def begin_transaction() -> None:
                pass

            def commit_transaction(self, *, accepted_rows=None) -> None:
                self.commits += 1

            def finalize_transaction(self) -> None:
                self.finalizes += 1

            def revert_committed_transaction(self) -> None:
                self.reverts += 1

            @staticmethod
            def rollback_transaction() -> None:
                pass

            @staticmethod
            def reset_session() -> None:
                pass

            def project(self, mixed, _name: str, *, layer: int):
                return torch.zeros(
                    (*mixed.shape[:-1], self.width),
                    dtype=mixed.dtype,
                )

            def project_many(self, mixed, _name: str, *, layer: int):
                return tuple(
                    torch.zeros(
                        (*row.shape[:-1], self.width),
                        dtype=row.dtype,
                    )
                    for row in mixed
                )

        router = PageRouter()
        model = StreamedQwen38(
            self.config,
            self.pager,
            mlp_page_router=router,
            max_batch_size=1,
            max_seq_len=32,
        )
        model.prefill([[1, 4]], reset=True)
        stage = model.stage_continuation_block([[5, 6]])
        self.assertEqual(router.begins, 1)
        model.discard_continuation_block(stage)
        self.assertEqual(router.rollbacks, 1)

        stage = model.stage_continuation_block([[5]])
        model.commit_continuation_block(stage)
        self.assertEqual(router.begins, 2)
        self.assertEqual(router.commits, 1)

        stale = model.stage_continuation_block([[7]])
        model.decode([[8]])
        self.assertEqual(router.rollbacks, 2)
        with self.assertRaisesRegex(Qwen38RuntimeError, "stale"):
            model.discard_continuation_block(stale)

        delta = DeltaRouter(self.config.dim)
        model.delta_head_router = delta
        stage = model.stage_continuation_block([[9, 10]])
        cursor = model.next_position
        router.fail_commit = True
        with self.assertRaisesRegex(RuntimeError, "page commit failed"):
            model.commit_continuation_prefix(stage, 1)
        self.assertEqual(model.next_position, cursor)
        self.assertEqual(delta.commits, 1)
        self.assertEqual(delta.reverts, 1)
        self.assertEqual(delta.finalizes, 0)
        router.fail_commit = False
        model.delta_head_router = None

        stage = model.stage_continuation_block([[11]])
        replacement = PageRouter()
        model.mlp_page_router = replacement
        model.reset_state()
        self.assertEqual(router.rollbacks, 3)
        self.assertEqual(replacement.resets, 1)
        with self.assertRaisesRegex(Qwen38RuntimeError, "stale"):
            model.discard_continuation_block(stage)

    def test_exact_mlp_without_sparse_executor_requests_no_weight_observer(
        self,
    ) -> None:
        hidden = torch.ones(1, 1, self.config.dim)
        with mock.patch.object(
            self.pager,
            "linear",
            wraps=self.pager.linear,
        ) as linear:
            self.model._mlp(hidden, layer=1)
        down = [
            call
            for call in linear.call_args_list
            if call.args[1].endswith(".down_proj")
        ]
        self.assertEqual(len(down), 1)
        self.assertIsNone(down[0].kwargs["weight_observer"])

        with mock.patch.object(
            self.pager,
            "linear_many",
            wraps=self.pager.linear_many,
        ) as linear_many:
            self.model._mlp_token_rows((hidden.clone(), hidden.clone()), layer=1)
        down_many = [
            call
            for call in linear_many.call_args_list
            if call.args[1].endswith(".down_proj")
        ]
        self.assertEqual(len(down_many), 1)
        self.assertIsNone(down_many[0].kwargs["weight_observer"])

    def test_online_sparse_decision_learns_from_the_unchanged_exact_path(self) -> None:
        class NonBeneficialRoute(RuntimeError):
            exact_mlp_fallback = True

        class AdaptiveStub:
            def __init__(self) -> None:
                self.use_sparse = False
                self.decisions = []
                self.observations = []
                self.nonbeneficial = False
                self.failure = None

            def supports_layer(self, layer: int) -> bool:
                return layer == 1

            def snapshot_identity(self, *, transport_neutral: bool = False):
                return {
                    "layers": [1],
                    "schema": "test-adaptive-sparse/v1",
                    "transport_neutral": transport_neutral,
                }

            def decision(self, *, layer: int, row_count: int):
                self.decisions.append((layer, row_count))
                return type("Decision", (), {"use_sparse": self.use_sparse})()

            def observe_full(self, **values):
                self.observations.append(values)
                return {"rows": len(values["gate"])}

            def execute(self, hidden: torch.Tensor, *, layer: int):
                if self.failure is not None:
                    raise self.failure
                if self.nonbeneficial:
                    raise NonBeneficialRoute("full MLP is cheaper")
                return torch.zeros_like(hidden), {"layer": layer}

            def execute_many(self, hidden, *, layer: int):
                if self.failure is not None:
                    raise self.failure
                if self.nonbeneficial:
                    raise NonBeneficialRoute("full MLP is cheaper")
                rows = tuple(hidden)
                return tuple(torch.zeros_like(row) for row in rows), {"layer": layer}

        adaptive = AdaptiveStub()
        model = StreamedQwen38(
            self.config,
            self.pager,
            mlp_sparse_executor=adaptive,
            max_batch_size=3,
            max_seq_len=32,
        )
        hidden = torch.ones(1, 1, self.config.dim)
        exact = model._mlp(hidden, layer=1)
        self.assertTrue(torch.isfinite(exact).all())
        self.assertEqual(adaptive.decisions, [(1, 1)])
        self.assertEqual(len(adaptive.observations), 1)
        observation = adaptive.observations[0]
        self.assertEqual(tuple(observation["gate"].shape[:-1]), (1, 1))
        self.assertEqual(tuple(observation["output"].shape), tuple(exact.shape))
        self.assertEqual(model.mlp_sparse_last_observation, {"rows": 1})

        rows = (hidden.clone(), hidden.clone())
        model._mlp_token_rows(rows, layer=1)
        self.assertEqual(adaptive.decisions[-1], (1, 2))
        self.assertEqual(len(adaptive.observations), 2)
        self.assertIsInstance(adaptive.observations[-1]["gate"], tuple)

        adaptive.use_sparse = True
        adaptive.online_controller = object()
        guarded = model._mlp(hidden, layer=1)
        self.assertFalse(torch.equal(guarded, torch.zeros_like(hidden)))
        self.assertEqual(len(adaptive.observations), 3)

        adaptive.online_controller = type(
            "OutputCalibrated",
            (),
            {"output_calibrated": lambda self, *, layer: layer == 1},
        )()
        sparse = model._mlp(hidden, layer=1)
        self.assertTrue(torch.equal(sparse, torch.zeros_like(hidden)))
        self.assertEqual(len(adaptive.observations), 3)

        adaptive.nonbeneficial = True
        economic_fallback = model._mlp(hidden, layer=1)
        self.assertFalse(torch.equal(economic_fallback, torch.zeros_like(hidden)))
        self.assertEqual(len(adaptive.observations), 4)

        adaptive.nonbeneficial = False
        adaptive.failure = RuntimeError("unexpected sparse failure")
        with self.assertRaisesRegex(RuntimeError, "unexpected sparse failure"):
            model._mlp(hidden, layer=1)

    def test_explicit_fast_mode_packs_continuation_projection_rows(self) -> None:
        pager = Qwen38WeightPager(
            self.source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=2 * 1024**2,
        )
        model = StreamedQwen38(
            self.config,
            pager,
            packed_continuation_gemm=True,
            max_batch_size=1,
            max_seq_len=16,
        )
        try:
            model.prefill([[1, 4]])
            before = pager.metrics()
            stage = model.stage_continuation_block([[9, 7, 6, 5]])
            after = pager.metrics()

            self.assertEqual(tuple(stage.hidden.shape), (1, 4, self.config.dim))
            self.assertGreater(
                after["packed_linear_calls"],
                before["packed_linear_calls"],
            )
            self.assertGreater(
                after["packed_linear_rows"],
                before["packed_linear_rows"],
            )
            model.discard_continuation_block(stage)
        finally:
            pager.close()

    def test_deltanet_probe_is_passive_and_covers_every_linear_layer(self) -> None:
        token_ids = torch.tensor([[1, 4, 9]])
        baseline, _ = self.model.forward_prefill(token_ids)
        records = []
        probed = StreamedQwen38(
            self.config,
            self.pager,
            delta_probe=lambda layer, row: records.append((layer, row)),
            max_batch_size=3,
            max_seq_len=32,
        )

        observed, _ = probed.forward_prefill(token_ids)

        torch.testing.assert_close(observed, baseline, rtol=0.0, atol=0.0)
        self.assertEqual([layer for layer, _row in records], [0, 1, 2])
        self.assertEqual(
            set(records[0][1].__dataclass_fields__),
            {
                "beta_mean",
                "beta_std",
                "decay_mean",
                "decay_std",
                "conv_norm",
                "q_norm",
                "k_norm",
                "v_norm",
                "delta_norm",
            },
        )
        for _layer, row in records:
            values = torch.tensor(
                [getattr(row, name) for name in row.__dataclass_fields__]
            )
            self.assertTrue(torch.isfinite(values).all())

    def test_delta_head_router_changes_only_continuation_output_projection(
        self,
    ) -> None:
        class Router:
            def __init__(self, width: int) -> None:
                self.width = width
                self.single = []
                self.many = []
                self.events = []

            @staticmethod
            def supports_layer(layer: int) -> bool:
                return layer == 0

            @staticmethod
            def snapshot_identity(*, transport_neutral: bool = False):
                return {"layers": [0], "transport_neutral": transport_neutral}

            def begin_transaction(self) -> None:
                self.events.append(("begin",))

            def commit_transaction(self, *, accepted_rows=None) -> None:
                self.events.append(("commit", accepted_rows))

            def finalize_transaction(self) -> None:
                self.events.append(("finalize",))

            def revert_committed_transaction(self) -> None:
                self.events.append(("revert",))

            def rollback_transaction(self) -> None:
                self.events.append(("rollback",))

            def reset_session(self) -> None:
                self.events.append(("reset",))

            def project(self, mixed, name: str, *, layer: int):
                self.single.append((layer, name, tuple(mixed.shape)))
                return torch.zeros((*mixed.shape[:-1], self.width), dtype=mixed.dtype)

            def project_many(self, mixed, name: str, *, layer: int):
                rows = tuple(mixed)
                self.many.append((layer, name, len(rows)))
                return tuple(
                    torch.zeros((*row.shape[:-1], self.width), dtype=row.dtype)
                    for row in rows
                )

        router = Router(self.config.dim)
        model = StreamedQwen38(
            self.config,
            self.pager,
            max_batch_size=1,
            max_seq_len=16,
        )
        self.assertIsNone(model.delta_head_router)
        model.set_delta_head_router(router)
        self.assertIs(model.delta_head_router, router)
        self.assertEqual(router.events, [("reset",)])
        model.prefill([[1, 4]])
        self.assertEqual(router.single, [])

        hidden, evidence = model.decode([[9]])

        self.assertEqual(tuple(hidden.shape), (1, 1, self.config.dim))
        self.assertEqual(evidence.end_pos, 3)
        self.assertEqual(
            router.single,
            [
                (
                    0,
                    "model.language_model.layers.0.linear_attn.out_proj.weight",
                    (1, 1, 6),
                )
            ],
        )

        discarded = model.stage_continuation_block([[10, 11]])
        model.discard_continuation_block(discarded)
        committed = model.stage_continuation_block([[10, 11]])
        model.commit_continuation_prefix(committed, 1)

        self.assertIn(("rollback",), router.events)
        self.assertIn(("commit", 1), router.events)
        self.assertEqual(router.events.count(("begin",)), 2)
        stage = model.stage_continuation_block([[7, 6]])
        self.assertEqual(
            router.many[-1][:2],
            (0, "model.language_model.layers.0.linear_attn.out_proj.weight"),
        )
        self.assertEqual(router.many[-1][2], 2)
        model.discard_continuation_block(stage)
        generated, generation = model.generate_greedy(
            [[1, 4]],
            max_new_tokens=2,
            retain_final_state=False,
        )
        self.assertEqual(len(generated), 2)
        self.assertEqual(generation.forward_passes, 2)
        self.assertIn(("commit", None), router.events)
        self.assertIn(("finalize",), router.events)
        model.set_delta_head_router(None)
        self.assertIsNone(model.delta_head_router)
        self.assertEqual(router.events[-1], ("reset",))

    def test_layer_boundary_observer_is_ordered_and_cannot_mutate_math(self) -> None:
        token_ids = torch.tensor([[1, 4, 9]])
        baseline, _ = self.model.forward_prefill(token_ids)
        records: list[tuple[int, str, tuple[int, ...]]] = []

        def observer(layer: int, stage: str, value: torch.Tensor) -> None:
            records.append((layer, stage, tuple(value.shape)))
            value.fill_(float("nan"))

        observed_model = StreamedQwen38(
            self.config,
            self.pager,
            layer_boundary_observer=observer,
            max_batch_size=3,
            max_seq_len=32,
        )
        observed, _ = observed_model.forward_prefill(token_ids)

        torch.testing.assert_close(observed, baseline, rtol=0.0, atol=0.0)
        self.assertEqual(
            len(records), self.config.n_layers * len(LAYER_BOUNDARY_STAGES)
        )
        for layer in range(self.config.n_layers):
            layer_rows = records[
                layer * len(LAYER_BOUNDARY_STAGES) : (layer + 1)
                * len(LAYER_BOUNDARY_STAGES)
            ]
            self.assertEqual(
                [stage for _layer, stage, _shape in layer_rows],
                list(LAYER_BOUNDARY_STAGES),
            )
            self.assertTrue(all(row[0] == layer for row in layer_rows))
            for _layer, stage, shape in layer_rows:
                expected_width = (
                    self.config.intermediate_size
                    if stage in {"mlp.gate", "mlp.up", "mlp.activated"}
                    else self.config.dim
                )
                self.assertEqual(shape, (1, 3, expected_width))
        with self.assertRaisesRegex(TypeError, "layer_boundary_observer"):
            StreamedQwen38(
                self.config,
                self.pager,
                layer_boundary_observer=object(),  # type: ignore[arg-type]
                max_seq_len=32,
            )
        filtered: list[tuple[int, str]] = []
        filtered_model = StreamedQwen38(
            self.config,
            self.pager,
            layer_boundary_observer=(
                lambda layer, stage, _value: filtered.append((layer, stage))
            ),
            layer_boundary_stages=("attention.output", "mlp.output"),
            max_batch_size=3,
            max_seq_len=32,
        )
        filtered_model.forward_prefill(token_ids)
        self.assertEqual(
            filtered,
            [
                (layer, stage)
                for layer in range(self.config.n_layers)
                for stage in ("attention.output", "mlp.output")
            ],
        )
        layer_filtered: list[tuple[int, str]] = []
        layer_filtered_model = StreamedQwen38(
            self.config,
            self.pager,
            layer_boundary_observer=(
                lambda layer, stage, _value: layer_filtered.append((layer, stage))
            ),
            layer_boundary_stages=("mlp.input", "mlp.gate", "mlp.up", "mlp.output"),
            layer_boundary_layers=(3, 1),
            max_batch_size=3,
            max_seq_len=32,
        )
        layer_filtered_model.forward_prefill(token_ids)
        self.assertEqual(
            layer_filtered,
            [
                (layer, stage)
                for layer in (1, 3)
                for stage in ("mlp.input", "mlp.gate", "mlp.up", "mlp.output")
            ],
        )
        self.assertEqual(layer_filtered_model.layer_boundary_layers, (1, 3))
        with self.assertRaisesRegex(ValueError, "duplicated"):
            StreamedQwen38(
                self.config,
                self.pager,
                layer_boundary_observer=lambda *_args: None,
                layer_boundary_stages=("mlp.output", "mlp.output"),
                max_seq_len=32,
            )
        with self.assertRaisesRegex(ValueError, "layer_boundary_layers"):
            StreamedQwen38(
                self.config,
                self.pager,
                layer_boundary_observer=lambda *_args: None,
                layer_boundary_layers=(1, 1),
                max_seq_len=32,
            )

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_deltanet_probe_reductions_run_from_mps_without_changing_output(
        self,
    ) -> None:
        pager = Qwen38WeightPager(
            self.source,
            device="mps",
            compute_dtype="bfloat16",
            max_resident_bytes=2 * 1024**2,
        )
        records = []
        try:
            baseline = StreamedQwen38(
                self.config, pager, max_batch_size=1, max_seq_len=32
            )
            expected, _ = baseline.forward_prefill([[1, 4, 9]])
            probed = StreamedQwen38(
                self.config,
                pager,
                delta_probe=lambda layer, row: records.append((layer, row)),
                max_batch_size=1,
                max_seq_len=32,
            )
            actual, _ = probed.forward_prefill([[1, 4, 9]])
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
            self.assertEqual([layer for layer, _row in records], [0, 1, 2])
        finally:
            pager.close()

    def test_stateful_prefill_matches_independent_and_tokenwise_execution(self) -> None:
        token_ids = torch.tensor([[1, 4, 9, 7]])
        independent, _ = self.model.forward_prefill(token_ids)

        batched, evidence = self.model.prefill(token_ids, tokenwise=False)
        torch.testing.assert_close(batched, independent, rtol=1e-6, atol=1e-6)
        self.assertEqual(self.model.next_position, 4)
        self.assertEqual(self.model.state_batch_size, 1)
        self.assertGreater(self.model.state_bytes, 0)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].context_mode, "prefill")
        self.assertEqual(evidence[0].end_pos, 4)
        self.assertEqual(evidence[0].linear_calls, 31)
        for layer, state in enumerate(self.model._layer_states):
            expected = (
                AttentionState
                if self.config.is_full_attention(layer)
                else DeltaNetState
            )
            self.assertIsInstance(state, expected)

        self.model.reset_state()
        tokenwise, token_evidence = self.model.prefill(token_ids, tokenwise=True)
        torch.testing.assert_close(tokenwise, independent, rtol=1e-5, atol=1e-6)
        self.assertEqual(len(token_evidence), 4)
        self.assertEqual(
            [(row.start_pos, row.end_pos) for row in token_evidence],
            [(0, 1), (1, 2), (2, 3), (3, 4)],
        )
        self.assertEqual(self.model.next_position, 4)

    def test_stateful_decode_matches_one_shot_last_token(self) -> None:
        token_ids = torch.tensor([[1, 4, 9, 7]])
        one_shot, _ = self.model.forward_prefill(token_ids)
        prefix, prefix_evidence = self.model.prefill(token_ids[:, :3])
        decoded, decode_evidence = self.model.decode(token_ids[:, 3:])

        self.assertEqual(tuple(prefix.shape), (1, 3, self.config.dim))
        self.assertEqual(prefix_evidence[0].context_mode, "prefill")
        torch.testing.assert_close(decoded, one_shot[:, -1:], rtol=1e-5, atol=1e-6)
        self.assertEqual(decode_evidence.context_mode, "decode")
        self.assertEqual((decode_evidence.start_pos, decode_evidence.end_pos), (3, 4))
        self.assertTrue(decode_evidence.stateful_cache)
        self.assertEqual(self.model.next_position, 4)
        for layer, state in enumerate(self.model._layer_states):
            if self.config.is_full_attention(layer):
                self.assertEqual(state.length, 4)

    def test_continuation_block_is_non_committing_and_single_use(self) -> None:
        self.model.prefill([[1, 4]])
        committed_objects = tuple(self.model._layer_states)
        committed_values = _clone_layer_states(self.model._layer_states)

        stage = self.model.stage_continuation_block([[9, 7]])

        self.assertEqual(self.model.next_position, 2)
        self.assertEqual(self.model.state_batch_size, 1)
        self.assertFalse(self.model.state_poisoned)
        self.assertEqual(tuple(stage.hidden.shape), (1, 2, self.config.dim))
        self.assertEqual((stage.evidence.start_pos, stage.evidence.end_pos), (2, 4))
        self.assertEqual(stage.evidence.input_token_ids, ((9, 7),))
        self.assertEqual(stage.evidence.layers_executed, self.config.n_layers)
        self.assertEqual(stage.evidence.linear_calls, 62)
        self.assertGreater(stage.evidence.staged_state_bytes, self.model.state_bytes)
        self.assertTrue(
            all(
                current is original
                for current, original in zip(
                    self.model._layer_states, committed_objects, strict=True
                )
            )
        )
        _assert_layer_states_equal(self, self.model._layer_states, committed_values)

        object.__setattr__(stage.evidence, "input_token_ids", ((5, 5),))
        object.__setattr__(stage.evidence, "source_body_bytes", 1)
        stage.hidden.fill_(float("nan"))
        hidden, evidence = self.model.commit_continuation_block(stage)

        self.assertIsNot(hidden, stage.hidden)
        self.assertTrue(torch.isfinite(hidden).all())
        self.assertEqual((evidence.start_pos, evidence.end_pos), (2, 4))
        self.assertEqual(evidence.context_mode, "decode")
        self.assertEqual(evidence.input_token_ids, ((9, 7),))
        self.assertNotEqual(evidence.source_body_bytes, 1)
        self.assertEqual(self.model.next_position, 4)
        for layer, state in enumerate(self.model._layer_states):
            if self.config.is_full_attention(layer):
                self.assertEqual(state.length, 4)
        with self.assertRaisesRegex(Qwen38RuntimeError, "stale or foreign"):
            self.model.commit_continuation_block(stage)

        next_stage = self.model.stage_continuation_block([[5, 6]])
        self.model.discard_continuation_block(next_stage)
        self.assertEqual(self.model.next_position, 4)
        with self.assertRaisesRegex(Qwen38RuntimeError, "stale or foreign"):
            self.model.discard_continuation_block(next_stage)

    def test_k1_continuation_owns_mlp_coordinate_transaction_to_commit(self) -> None:
        self.model.prefill([[1, 4]])
        bank = MlpPageCoordinateBank(
            self.root / "lifecycle-mlp-page-coordinates.json",
            MlpPageCoordinateIdentity(
                runtime_math_sha256="a" * 64,
                q4_identity_sha256="b" * 64,
                page_router_identity_sha256="c" * 64,
            ),
            max_cells=8,
        )
        # This lifecycle test does not execute the Q4 page seam; the dedicated
        # model test above covers exact capture/replay.  Bind the real bank
        # directly so the pending owner can be observed with the BF16-free fixture.
        self.model.mlp_page_coordinate_bank = bank
        self.model.mlp_page_coordinate_enabled = True

        discarded = self.model.stage_continuation_block([[9]])
        discarded_owner = (
            self.model._pending_block_stage.mlp_page_coordinate_transaction_owner
        )
        self.assertIsNotNone(discarded_owner)
        self.model.discard_continuation_block(discarded)
        self.assertTrue(discarded_owner.closed)

        first = self.model.stage_continuation_block([[9]])
        owner = self.model._pending_block_stage.mlp_page_coordinate_transaction_owner
        extended = self.model.extend_continuation_block(first, [[7]])
        self.assertIs(
            self.model._pending_block_stage.mlp_page_coordinate_transaction_owner,
            owner,
        )
        self.model.commit_continuation_prefix(extended, 1)
        self.assertTrue(owner.closed)

    def test_attention_output_crystal_replays_exact_attention_before_weight_reads(
        self,
    ) -> None:
        pager = Qwen38WeightPager(
            self.source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=2 * 1024**2,
        )
        model = StreamedQwen38(
            self.config,
            pager,
            max_batch_size=1,
            max_seq_len=32,
        )
        mismatched = AttentionOutputCrystalBank(
            self.root / "mismatched-attention-output-crystals.json",
            AttentionOutputCrystalIdentity(runtime_math_sha256="f" * 64),
            max_cells=8,
        )
        with self.assertRaisesRegex(Qwen38RuntimeError, "identity.*target"):
            model.attach_attention_output_crystal_bank(mismatched)
        bank = AttentionOutputCrystalBank(
            self.root / "attention-output-crystals.json",
            AttentionOutputCrystalIdentity(
                runtime_math_sha256=(
                    model.attention_output_crystal_runtime_math_sha256()
                )
            ),
            max_cells=64,
        )
        model.attach_attention_output_crystal_bank(bank)
        try:
            model.prefill([[1, 4]])
            first_before = pager.metrics()["linear_calls"]
            first_stage = model.stage_continuation_block([[9, 7]])
            first_calls = pager.metrics()["linear_calls"] - first_before
            first_hidden, _ = model.commit_continuation_block(first_stage)
            first_states = _clone_layer_states(model._layer_states)
            charged = bank.metrics()
            self.assertEqual(charged.accepted_captures, 2)
            self.assertEqual(charged.hit_count, 0)

            model.reset_state()
            model.prefill([[1, 4]])
            replay_before = pager.metrics()["linear_calls"]
            replay_stage = model.stage_continuation_block([[9, 7]])
            replay_calls = pager.metrics()["linear_calls"] - replay_before
            replay_hidden, _ = model.commit_continuation_block(replay_stage)

            torch.testing.assert_close(
                replay_hidden,
                first_hidden,
                rtol=0.0,
                atol=0.0,
            )
            _assert_layer_states_equal(self, model._layer_states, first_states)
            self.assertLess(replay_calls, first_calls)
            replayed = bank.metrics()
            self.assertEqual(replayed.hit_count, 2)
            self.assertEqual(replayed.skipped_projection_calls_saved, 4)
            self.assertEqual(
                replayed.logical_projection_bytes_saved,
                model._full_attention_logical_projection_bytes(),
            )

            baseline_pager = Qwen38WeightPager(
                self.source,
                device="cpu",
                compute_dtype="bfloat16",
                max_resident_bytes=2 * 1024**2,
            )
            baseline = StreamedQwen38(
                self.config,
                baseline_pager,
                max_batch_size=1,
                max_seq_len=32,
            )
            try:
                baseline.prefill([[1, 4]])
                baseline_stage = baseline.stage_continuation_block([[9, 8]])
                baseline_hidden, _ = baseline.commit_continuation_block(baseline_stage)
                baseline_states = _clone_layer_states(baseline._layer_states)
            finally:
                baseline.reset_state(release=True)
                baseline_pager.close()

            model.reset_state()
            model.prefill([[1, 4]])
            partial_hit_stage = model.stage_continuation_block([[9, 8]])
            partial_hit_hidden, _ = model.commit_continuation_block(partial_hit_stage)
            torch.testing.assert_close(
                partial_hit_hidden,
                baseline_hidden,
                rtol=0.0,
                atol=0.0,
            )
            _assert_layer_states_equal(self, model._layer_states, baseline_states)
            after_partial_hit = bank.metrics()
            self.assertEqual(after_partial_hit.hit_count, replayed.hit_count + 1)
            self.assertEqual(
                after_partial_hit.skipped_projection_calls_saved,
                replayed.skipped_projection_calls_saved,
            )

            model.reset_state()
            model.prefill([[1, 4]])
            before_discard = bank.metrics()
            discarded = model.stage_continuation_block([[9, 7]])
            model.discard_continuation_block(discarded)
            after_discard = bank.metrics()
            self.assertEqual(after_discard.hit_count, before_discard.hit_count)
            self.assertEqual(
                after_discard.logical_projection_bytes_saved,
                before_discard.logical_projection_bytes_saved,
            )

            model.reset_state()
            model.prefill([[1, 4]])
            partial = model.stage_continuation_block([[9, 7]])
            partial_hidden, _ = model.commit_continuation_prefix(partial, 1)
            self.assertEqual(tuple(partial_hidden.shape), (1, 1, self.config.dim))
            after_partial = bank.metrics()
            self.assertEqual(
                after_partial.hit_count,
                after_partial_hit.hit_count + 1,
            )

            model.reset_state()
            model.prefill([[1, 4]])
            direct_before = bank.metrics().hit_count
            with mock.patch.object(
                model,
                "stage_continuation_block",
                side_effect=AssertionError("direct K1 must retain the fused path"),
            ):
                direct_hidden, direct_evidence = model.decode([[9]])
            self.assertEqual(tuple(direct_hidden.shape), (1, 1, self.config.dim))
            torch.testing.assert_close(
                direct_hidden,
                baseline_hidden[:, :1],
                rtol=0.0,
                atol=0.0,
            )
            self.assertEqual(direct_evidence.context_mode, "decode")
            self.assertEqual(bank.metrics().hit_count, direct_before + 1)

            publications_before = bank.metrics().publish_transactions
            generated, generation_evidence = model.generate_greedy(
                [[1, 4]],
                max_new_tokens=3,
                retain_final_state=False,
            )
            self.assertEqual(len(generated), 3)
            self.assertEqual(generation_evidence.forward_passes, 3)
            self.assertEqual(
                bank.metrics().publish_transactions,
                publications_before + 1,
            )
        finally:
            model.reset_state(release=True)
            pager.close()

    def test_bfloat16_continuation_block_matches_tokenwise_state_bit_exactly(
        self,
    ) -> None:
        pagers = [
            Qwen38WeightPager(
                self.source,
                device="cpu",
                compute_dtype="bfloat16",
                max_resident_bytes=2 * 1024**2,
            )
            for _ in range(2)
        ]
        block_model, token_model = (
            StreamedQwen38(
                self.config,
                pager,
                max_batch_size=1,
                max_seq_len=16,
            )
            for pager in pagers
        )
        try:
            block_model.prefill([[1, 4]])
            stage = block_model.stage_continuation_block([[9, 7]])

            token_model.prefill([[1, 4]])
            first, first_evidence = token_model.decode([[9]])
            second, second_evidence = token_model.decode([[7]])
            tokenwise = torch.cat((first, second), dim=1)

            self.assertTrue(torch.equal(stage.hidden, tokenwise))
            self.assertEqual(stage.evidence.linear_calls, 62)
            self.assertEqual(
                first_evidence.linear_calls + second_evidence.linear_calls,
                62,
            )

            committed, evidence = block_model.commit_continuation_block(stage)
            self.assertTrue(torch.equal(committed, tokenwise))
            _assert_layer_states_equal(
                self, block_model._layer_states, token_model._layer_states
            )
            self.assertEqual(evidence.state_bytes, token_model.state_bytes)
        finally:
            for pager in pagers:
                pager.close()

    def test_continuation_prefix_commit_is_bit_exact_and_reads_no_weights(
        self,
    ) -> None:
        tokens = tuple(range(5, 21))
        for stage_width in (2, 4, 8, 16):
            commit_widths = tuple(sorted({1, stage_width // 2, stage_width - 1}))
            for commit_width in commit_widths:
                with self.subTest(
                    stage_width=stage_width,
                    commit_width=commit_width,
                ):
                    pagers = [
                        Qwen38WeightPager(
                            self.source,
                            device="cpu",
                            compute_dtype="bfloat16",
                            max_resident_bytes=2 * 1024**2,
                        )
                        for _ in range(2)
                    ]
                    block_model, token_model = (
                        StreamedQwen38(
                            self.config,
                            pager,
                            max_batch_size=1,
                            max_seq_len=32,
                        )
                        for pager in pagers
                    )
                    try:
                        block_model.prefill([[1, 4]])
                        token_model.prefill([[1, 4]])
                        stage = block_model.stage_continuation_block(
                            [tokens[:stage_width]]
                        )
                        trace = block_model._pending_block_stage.prefix_trace
                        self.assertEqual(trace.width, stage_width)
                        if stage_width <= self.config.linear_conv_kernel_dim:
                            self.assertLess(
                                trace.nbytes,
                                stage.evidence.staged_state_bytes,
                            )
                        expected_rows = []
                        for token in tokens[:commit_width]:
                            hidden, _evidence = token_model.decode([[token]])
                            expected_rows.append(hidden)
                        expected = torch.cat(expected_rows, dim=1)

                        before = block_model.pager.metrics()
                        actual, evidence = block_model.commit_continuation_prefix(
                            stage,
                            commit_width,
                        )
                        after = block_model.pager.metrics()

                        self.assertTrue(torch.equal(actual, expected))
                        _assert_layer_states_equal(
                            self,
                            block_model._layer_states,
                            token_model._layer_states,
                        )
                        self.assertEqual(block_model.next_position, 2 + commit_width)
                        self.assertEqual(
                            evidence.input_token_ids, (tokens[:commit_width],)
                        )
                        self.assertEqual(evidence.end_pos, 2 + commit_width)
                        for key in (
                            "tensor_reads",
                            "linear_calls",
                            "network_or_source_body_bytes",
                        ):
                            self.assertEqual(after[key], before[key])
                        with self.assertRaisesRegex(
                            Qwen38RuntimeError, "stale or foreign"
                        ):
                            block_model.commit_continuation_prefix(stage, 1)

                        continued, _ = block_model.decode([[8]])
                        expected_continued, _ = token_model.decode([[8]])
                        self.assertTrue(torch.equal(continued, expected_continued))
                        _assert_layer_states_equal(
                            self,
                            block_model._layer_states,
                            token_model._layer_states,
                        )
                    finally:
                        for pager in pagers:
                            pager.close()

    def test_extended_continuation_prefix_matches_direct_stage(self) -> None:
        pagers = [
            Qwen38WeightPager(
                self.source,
                device="cpu",
                compute_dtype="bfloat16",
                max_resident_bytes=2 * 1024**2,
            )
            for _ in range(2)
        ]
        direct, extended = (
            StreamedQwen38(
                self.config,
                pager,
                max_batch_size=1,
                max_seq_len=16,
            )
            for pager in pagers
        )
        try:
            direct.prefill([[1, 4]])
            extended.prefill([[1, 4]])
            direct_stage = direct.stage_continuation_block([[9, 7, 6, 5]])
            extended_stage = extended.stage_continuation_block([[9]])
            for token in (7, 6, 5):
                extended_stage = extended.extend_continuation_block(
                    extended_stage, [[token]]
                )

            direct_hidden, _ = direct.commit_continuation_prefix(direct_stage, 2)
            extended_hidden, _ = extended.commit_continuation_prefix(extended_stage, 2)

            self.assertTrue(torch.equal(direct_hidden, extended_hidden))
            _assert_layer_states_equal(
                self,
                direct._layer_states,
                extended._layer_states,
            )
        finally:
            for pager in pagers:
                pager.close()

    def test_failed_prefix_reconstruction_consumes_stage_and_preserves_base(
        self,
    ) -> None:
        self.model.prefill([[1, 4]])
        base = _clone_layer_states(self.model._layer_states)
        stage = self.model.stage_continuation_block([[9, 7, 6, 5]])

        with mock.patch.object(
            qwen_model_module,
            "recurrent_gated_delta_rule",
            side_effect=RuntimeError("prefix recurrence failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "prefix recurrence failed"):
                self.model.commit_continuation_prefix(stage, 2)

        self.assertIsNone(self.model._pending_block_stage)
        self.assertEqual(self.model.next_position, 2)
        _assert_layer_states_equal(self, self.model._layer_states, base)
        with self.assertRaisesRegex(Qwen38RuntimeError, "stale or foreign"):
            self.model.commit_continuation_prefix(stage, 1)

    def test_continuation_block_stale_and_failure_paths_preserve_base_state(
        self,
    ) -> None:
        self.model.prefill([[1, 4]])
        base_objects = tuple(self.model._layer_states)
        base_values = _clone_layer_states(self.model._layer_states)
        first = self.model.stage_continuation_block([[9, 7]])
        second = self.model.stage_continuation_block([[8, 6]])
        with self.assertRaisesRegex(Qwen38RuntimeError, "stale or foreign"):
            self.model.commit_continuation_block(first)
        self.model.discard_continuation_block(second)

        original_delta_core = qwen_model_module.gated_delta_net_core
        delta_calls = 0

        def fail_on_second_token(*args, **kwargs):
            nonlocal delta_calls
            delta_calls += 1
            if delta_calls == 2:
                raise RuntimeError("block transaction failure")
            return original_delta_core(*args, **kwargs)

        with mock.patch.object(
            qwen_model_module,
            "gated_delta_net_core",
            side_effect=fail_on_second_token,
        ):
            with self.assertRaisesRegex(RuntimeError, "block transaction failure"):
                self.model.stage_continuation_block([[9, 7]])

        self.assertEqual(self.model.next_position, 2)
        self.assertFalse(self.model.state_poisoned)
        self.assertTrue(
            all(
                current is original
                for current, original in zip(
                    self.model._layer_states, base_objects, strict=True
                )
            )
        )
        _assert_layer_states_equal(self, self.model._layer_states, base_values)

        original_norm_pair = self.model._norm_k2_pair

        def fail_final_norm(hidden, name):
            if name == self.model.FINAL_NORM_NAME:
                raise RuntimeError("block final norm failure")
            return original_norm_pair(hidden, name)

        with mock.patch.object(
            self.model,
            "_norm_k2_pair",
            side_effect=fail_final_norm,
        ):
            with self.assertRaisesRegex(RuntimeError, "block final norm failure"):
                self.model.stage_continuation_block([[9, 7]])
        self.assertIsNone(self.model._pending_block_stage)
        self.assertEqual(self.model.next_position, 2)
        self.assertTrue(
            all(
                current is original
                for current, original in zip(
                    self.model._layer_states, base_objects, strict=True
                )
            )
        )
        _assert_layer_states_equal(self, self.model._layer_states, base_values)

        recovered, evidence = self.model.decode([[9]])
        self.assertEqual(tuple(recovered.shape), (1, 1, self.config.dim))
        self.assertEqual(evidence.end_pos, 3)

    def test_continuation_block_rejects_probe_and_state_invalidation(self) -> None:
        records = []
        probed = StreamedQwen38(
            self.config,
            self.pager,
            delta_probe=lambda layer, row: records.append((layer, row)),
            max_batch_size=1,
            max_seq_len=16,
        )
        probed.prefill([[1, 4]])
        records.clear()
        with self.assertRaisesRegex(Qwen38RuntimeError, "active DeltaNet probe"):
            probed.stage_continuation_block([[9, 7]])
        self.assertEqual(records, [])
        self.assertEqual(probed.next_position, 2)

        self.model.prefill([[1, 4]])
        stage = self.model.stage_continuation_block([[9, 7]])
        self.model.reset_state()
        with self.assertRaisesRegex(Qwen38RuntimeError, "stale or foreign"):
            self.model.commit_continuation_block(stage)

        self.model.prefill([[1, 4]])
        stage = self.model.stage_continuation_block([[9, 7]])
        self.model.delta_probe = lambda _layer, _row: None
        with self.assertRaisesRegex(Qwen38RuntimeError, "configuration changed"):
            self.model.commit_continuation_block(stage)
        self.model.delta_probe = None

        stage = self.model.stage_continuation_block([[9, 7]])
        self.model.decode([[8]])
        with self.assertRaisesRegex(Qwen38RuntimeError, "stale or foreign"):
            self.model.commit_continuation_block(stage)

    def test_continuation_block_shape_and_context_boundaries(self) -> None:
        self.model.prefill([[1, 4]])
        with self.assertRaisesRegex(ValueError, "dimensions must be non-empty"):
            self.model.stage_continuation_block(torch.empty((1, 0), dtype=torch.long))
        with self.assertRaisesRegex(Qwen38RuntimeError, "K in \\[1, 16\\]"):
            self.model.stage_continuation_block([list(range(17))])

        self.model.reset_state()
        self.model.prefill([[1, 4], [2, 5]])
        with self.assertRaisesRegex(Qwen38RuntimeError, "batch 1"):
            self.model.stage_continuation_block([[9, 7], [8, 6]])

        self.model.reset_state()
        self.model.prefill([[1, 4]])
        stale = self.model.stage_continuation_block([[9, 7]])
        self.model.max_batch_size = 0
        with self.assertRaisesRegex(Qwen38RuntimeError, "current runtime bounds"):
            self.model.commit_continuation_block(stale)
        self.model.max_batch_size = 3

        stale = self.model.stage_continuation_block([[9, 7]])
        self.model.max_seq_len = 3
        with self.assertRaisesRegex(Qwen38RuntimeError, "current runtime bounds"):
            self.model.commit_continuation_block(stale)
        self.model.max_seq_len = 32

        bounded = StreamedQwen38(
            self.config,
            self.pager,
            max_batch_size=1,
            max_seq_len=3,
        )
        bounded.prefill([[1, 4]])
        with self.assertRaisesRegex(ValueError, "exceeds max_seq_len"):
            bounded.stage_continuation_block([[9, 7]])

    def test_default_prefill_and_decode_never_enter_k2_pair_path(self) -> None:
        with mock.patch.object(
            self.model,
            "_stage_continuation_k2_pair",
            side_effect=AssertionError("K=2 path entered"),
        ):
            self.model.prefill([[1, 4]])
            hidden, evidence = self.model.decode([[9]])
        self.assertEqual(tuple(hidden.shape), (1, 1, self.config.dim))
        self.assertEqual(evidence.linear_calls, 31)

    def test_continuation_extension_is_opaque_single_use_and_foreign_safe(
        self,
    ) -> None:
        pager = Qwen38WeightPager(
            self.source,
            device="cpu",
            compute_dtype="float32",
            max_resident_bytes=2 * 1024**2,
        )
        reference = StreamedQwen38(
            self.config,
            pager,
            max_batch_size=1,
            max_seq_len=16,
        )
        try:
            self.model.prefill([[1, 4]])
            reference.prefill([[1, 4]])
            committed_objects = tuple(self.model._layer_states)
            committed_values = _clone_layer_states(self.model._layer_states)

            first = self.model.stage_continuation_block([[9]])
            object.__setattr__(first.evidence, "input_token_ids", ((5,),))
            first.hidden.fill_(float("nan"))
            extended = self.model.extend_continuation_block(first, [[7]])

            first_hidden, _ = reference.decode([[9]])
            second_hidden, _ = reference.decode([[7]])
            expected = torch.cat((first_hidden, second_hidden), dim=1)
            self.assertTrue(torch.equal(extended.hidden, expected))
            self.assertEqual(extended.evidence.input_token_ids, ((9, 7),))
            self.assertEqual(
                (extended.evidence.start_pos, extended.evidence.end_pos),
                (2, 4),
            )
            self.assertEqual(extended.evidence.linear_calls, 62)
            self.assertEqual(self.model.next_position, 2)
            self.assertTrue(
                all(
                    current is original
                    for current, original in zip(
                        self.model._layer_states,
                        committed_objects,
                        strict=True,
                    )
                )
            )
            _assert_layer_states_equal(self, self.model._layer_states, committed_values)
            with self.assertRaisesRegex(Qwen38RuntimeError, "stale or foreign"):
                self.model.commit_continuation_block(first)

            actual, evidence = self.model.commit_continuation_block(extended)
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(evidence.input_token_ids, ((9, 7),))
            _assert_layer_states_equal(
                self, self.model._layer_states, reference._layer_states
            )

            discard = self.model.stage_continuation_block([[6, 5, 4]])
            self.model.discard_continuation_block(discard)
            self.assertEqual(self.model.next_position, 4)
            with self.assertRaisesRegex(Qwen38RuntimeError, "stale or foreign"):
                self.model.extend_continuation_block(discard, [[3]])

            self.model.reset_state()
            self.model.prefill([[1, 4]])
            reset_stage = self.model.stage_continuation_block([[9]])
            self.model.reset_state()
            with self.assertRaisesRegex(Qwen38RuntimeError, "stale or foreign"):
                self.model.extend_continuation_block(reset_stage, [[7]])

            self.model.prefill([[1, 4]])
            reference.reset_state()
            reference.prefill([[1, 4]])
            local = self.model.stage_continuation_block([[9]])
            foreign = reference.stage_continuation_block([[9]])
            with self.assertRaisesRegex(Qwen38RuntimeError, "stale or foreign"):
                self.model.extend_continuation_block(foreign, [[7]])
            self.model.commit_continuation_block(local)
            reference.discard_continuation_block(foreign)
        finally:
            pager.close()

    def test_continuation_extension_failure_discards_without_committed_mutation(
        self,
    ) -> None:
        self.model.prefill([[1, 4]])
        committed_objects = tuple(self.model._layer_states)
        committed_values = _clone_layer_states(self.model._layer_states)
        first = self.model.stage_continuation_block([[9]])

        with mock.patch.object(
            qwen_model_module,
            "gated_delta_net_core",
            side_effect=RuntimeError("extension transaction failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "extension transaction failure"):
                self.model.extend_continuation_block(first, [[7]])

        self.assertIsNone(self.model._pending_block_stage)
        self.assertEqual(self.model.next_position, 2)
        self.assertFalse(self.model.state_poisoned)
        self.assertTrue(
            all(
                current is original
                for current, original in zip(
                    self.model._layer_states,
                    committed_objects,
                    strict=True,
                )
            )
        )
        _assert_layer_states_equal(self, self.model._layer_states, committed_values)
        with self.assertRaisesRegex(Qwen38RuntimeError, "stale or foreign"):
            self.model.commit_continuation_block(first)

        full = self.model.stage_continuation_block([list(range(16))])
        with self.assertRaisesRegex(Qwen38RuntimeError, "exceeds K=16"):
            self.model.extend_continuation_block(full, [[4]])
        self.assertIsNone(self.model._pending_block_stage)
        with self.assertRaisesRegex(Qwen38RuntimeError, "stale or foreign"):
            self.model.commit_continuation_block(full)

        probed = self.model.stage_continuation_block([[9]])
        self.model.delta_probe = lambda _layer, _row: None
        with self.assertRaisesRegex(Qwen38RuntimeError, "active DeltaNet probe"):
            self.model.extend_continuation_block(probed, [[7]])
        self.model.delta_probe = None
        self.assertIsNone(self.model._pending_block_stage)

        original_norm_rows = self.model._norm_token_rows

        def fail_final_norm(hidden, name):
            if name == self.model.FINAL_NORM_NAME:
                raise RuntimeError("generic block final norm failure")
            return original_norm_rows(hidden, name)

        with mock.patch.object(
            self.model,
            "_norm_token_rows",
            side_effect=fail_final_norm,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "generic block final norm failure"
            ):
                self.model.stage_continuation_block([[9, 7, 6]])
        self.assertIsNone(self.model._pending_block_stage)
        self.assertEqual(self.model.next_position, 2)
        _assert_layer_states_equal(self, self.model._layer_states, committed_values)

        extend_final = self.model.stage_continuation_block([[9]])
        with mock.patch.object(
            self.model,
            "_norm_token_rows",
            side_effect=fail_final_norm,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "generic block final norm failure"
            ):
                self.model.extend_continuation_block(extend_final, [[7]])
        self.assertIsNone(self.model._pending_block_stage)
        self.assertEqual(self.model.next_position, 2)
        _assert_layer_states_equal(self, self.model._layer_states, committed_values)

    def test_official_topology_k1_to_k4_are_bf16_exact_for_all_runtime_modes(
        self,
    ) -> None:
        from immer.runtimes.qwen3_8.native_crsa import Qwen38NativeHeadCrsa

        config = _official_topology_tiny_config()
        official_root = self.root / "official-topology"
        official_root.mkdir()
        save_file(_tiny_weights(config), official_root / "model.safetensors")
        source = Streamer.from_local(
            official_root,
            budget_mb=64,
            use_cache=False,
        )
        try:
            for mode in ("off", "stable", "native-prefix-sinkhorn"):
                for width in range(1, 5):
                    with self.subTest(mode=mode, width=width):
                        token_ids = [9, 7, 6, 5][:width]
                        observed = [[], [], []]
                        pagers = [
                            Qwen38WeightPager(
                                source,
                                device="cpu",
                                compute_dtype="bfloat16",
                                max_resident_bytes=2 * 1024**2,
                            )
                            for _ in range(3)
                        ]

                        def model(index, pager):
                            kwargs = {}
                            if mode == "stable":
                                kwargs = {
                                    "graft": Qwen38StableCrsaGraft(
                                        mode="crsa",
                                        alpha=0.1,
                                    ),
                                    "graft_layer": 27,
                                }
                            elif mode == "native-prefix-sinkhorn":
                                kwargs = {
                                    "native_head_crsa": Qwen38NativeHeadCrsa(alpha=0.1),
                                    "native_head_crsa_observer": observed[index].append,
                                }
                            return StreamedQwen38(
                                config,
                                pager,
                                max_batch_size=1,
                                max_seq_len=16,
                                **kwargs,
                            )

                        block_model, extension_model, token_model = [
                            model(index, pager) for index, pager in enumerate(pagers)
                        ]
                        try:
                            block_model.prefill([[1, 4]])
                            extension_model.prefill([[1, 4]])
                            token_model.prefill([[1, 4]])
                            observed[0].clear()
                            observed[1].clear()
                            observed[2].clear()
                            block_reads_before = block_model.pager.metrics()[
                                "tensor_reads"
                            ]
                            stage = block_model.stage_continuation_block([token_ids])
                            block_reads = (
                                block_model.pager.metrics()["tensor_reads"]
                                - block_reads_before
                            )
                            self.assertEqual(observed[0], [])

                            extension_reads_before = extension_model.pager.metrics()[
                                "tensor_reads"
                            ]
                            extension_stages = [
                                extension_model.stage_continuation_block(
                                    [[token_ids[0]]]
                                )
                            ]
                            for token in token_ids[1:]:
                                extension_stages.append(
                                    extension_model.extend_continuation_block(
                                        extension_stages[-1], [[token]]
                                    )
                                )
                            extension_stage = extension_stages[-1]
                            extension_reads = (
                                extension_model.pager.metrics()["tensor_reads"]
                                - extension_reads_before
                            )
                            self.assertEqual(observed[1], [])

                            token_reads_before = token_model.pager.metrics()[
                                "tensor_reads"
                            ]
                            outputs = []
                            token_evidence = []
                            for token in token_ids:
                                hidden, evidence = token_model.decode([[token]])
                                outputs.append(hidden)
                                token_evidence.append(evidence)
                            token_reads = (
                                token_model.pager.metrics()["tensor_reads"]
                                - token_reads_before
                            )
                            expected = torch.cat(outputs, dim=1)

                            self.assertTrue(torch.equal(stage.hidden, expected))
                            self.assertTrue(
                                torch.equal(extension_stage.hidden, expected)
                            )
                            self.assertEqual(
                                extension_stage.evidence.input_token_ids,
                                (tuple(token_ids),),
                            )
                            self.assertEqual(
                                extension_stage.evidence.linear_calls,
                                stage.evidence.linear_calls,
                            )
                            self.assertEqual(
                                stage.evidence.linear_calls,
                                sum(row.linear_calls for row in token_evidence),
                            )
                            self.assertEqual(width * block_reads, token_reads)
                            self.assertEqual(width * block_reads, extension_reads)
                            if width == 1:
                                self.assertEqual(
                                    stage.evidence.source_body_bytes,
                                    token_evidence[0].source_body_bytes,
                                )
                            else:
                                self.assertLess(
                                    stage.evidence.source_body_bytes,
                                    sum(
                                        row.source_body_bytes for row in token_evidence
                                    ),
                                )
                            actual, evidence = block_model.commit_continuation_block(
                                stage
                            )
                            extended, extension_evidence = (
                                extension_model.commit_continuation_block(
                                    extension_stage
                                )
                            )
                            self.assertTrue(torch.equal(actual, expected))
                            self.assertTrue(torch.equal(extended, expected))
                            _assert_layer_states_equal(
                                self,
                                block_model._layer_states,
                                token_model._layer_states,
                            )
                            _assert_layer_states_equal(
                                self,
                                extension_model._layer_states,
                                token_model._layer_states,
                            )
                            self.assertEqual(
                                evidence.state_bytes, token_model.state_bytes
                            )
                            self.assertEqual(
                                extension_evidence.state_bytes,
                                token_model.state_bytes,
                            )
                            if mode == "stable":
                                self.assertTrue(
                                    torch.equal(
                                        block_model._graft_history,
                                        token_model._graft_history,
                                    )
                                )
                                self.assertTrue(
                                    torch.equal(
                                        extension_model._graft_history,
                                        token_model._graft_history,
                                    )
                                )
                            elif mode == "native-prefix-sinkhorn":
                                self.assertEqual(observed[0], observed[2])
                                self.assertEqual(observed[1], observed[2])
                                self.assertEqual(len(observed[0]), width)
                        finally:
                            for pager in pagers:
                                pager.close()
        finally:
            source.close()

    def test_prefix_commit_preserves_graft_and_native_sinkhorn_state(self) -> None:
        from immer.runtimes.qwen3_8.native_crsa import (
            NativePrefixSinkhornOperatorObserver,
            Qwen38NativeHeadCrsa,
        )

        config = _official_topology_tiny_config()
        root = self.root / "prefix-runtime-modes"
        root.mkdir()
        save_file(_tiny_weights(config), root / "model.safetensors")
        source = Streamer.from_local(root, budget_mb=64, use_cache=False)
        try:
            for mode in ("stable", "native-prefix-sinkhorn"):
                with self.subTest(mode=mode):
                    pagers = [
                        Qwen38WeightPager(
                            source,
                            device="cpu",
                            compute_dtype="bfloat16",
                            max_resident_bytes=2 * 1024**2,
                        )
                        for _ in range(2)
                    ]
                    head_rows = [[], []]
                    operator_rows = [[], []]

                    def build(index: int, pager: Qwen38WeightPager):
                        kwargs = {}
                        if mode == "stable":
                            kwargs = {
                                "graft": Qwen38StableCrsaGraft(mode="crsa", alpha=0.1),
                                "graft_layer": 27,
                            }
                        else:
                            kwargs = {
                                "native_head_crsa": Qwen38NativeHeadCrsa(alpha=0.1),
                                "native_head_crsa_observer": head_rows[index].append,
                                "native_prefix_sinkhorn_operator_observer": (
                                    NativePrefixSinkhornOperatorObserver(
                                        operator_rows[index].append,
                                        max_positions=4,
                                    )
                                ),
                            }
                        return StreamedQwen38(
                            config,
                            pager,
                            max_batch_size=1,
                            max_seq_len=16,
                            **kwargs,
                        )

                    block, tokenwise = [
                        build(index, pager) for index, pager in enumerate(pagers)
                    ]
                    try:
                        block.prefill([[1, 4]])
                        tokenwise.prefill([[1, 4]])
                        head_rows[0].clear()
                        head_rows[1].clear()
                        operator_rows[0].clear()
                        operator_rows[1].clear()
                        stage = block.stage_continuation_block([[9, 7, 6, 5]])
                        self.assertEqual(head_rows[0], [])
                        self.assertEqual(operator_rows[0], [])
                        expected_rows = [
                            tokenwise.decode([[token]])[0] for token in (9, 7)
                        ]

                        actual, _ = block.commit_continuation_prefix(stage, 2)

                        self.assertTrue(
                            torch.equal(actual, torch.cat(expected_rows, dim=1))
                        )
                        _assert_layer_states_equal(
                            self,
                            block._layer_states,
                            tokenwise._layer_states,
                        )
                        if mode == "stable":
                            self.assertTrue(
                                torch.equal(
                                    block._graft_history,
                                    tokenwise._graft_history,
                                )
                            )
                        else:
                            self.assertEqual(head_rows[0], head_rows[1])
                            self.assertEqual(
                                [row.sha256 for row in operator_rows[0]],
                                [row.sha256 for row in operator_rows[1]],
                            )
                    finally:
                        for pager in pagers:
                            pager.close()
        finally:
            source.close()

    def test_stateful_bfloat16_prefill_and_decode_are_bit_exact(self) -> None:
        pager = Qwen38WeightPager(
            self.source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=2 * 1024**2,
        )
        model = StreamedQwen38(self.config, pager, max_batch_size=1, max_seq_len=16)
        token_ids = torch.tensor([[1, 4, 9, 7]])
        try:
            batched, _ = model.prefill(token_ids)
            model.reset_state()
            tokenwise, _ = model.prefill(token_ids, tokenwise=True)
            self.assertTrue(torch.equal(batched, tokenwise))

            model.reset_state()
            model.prefill(token_ids[:, :3])
            decoded, _ = model.decode(token_ids[:, 3:])
            self.assertTrue(torch.equal(decoded, batched[:, -1:]))
        finally:
            pager.close()

    def test_stateful_layer_range_full_and_split_match_committed_prefill(
        self,
    ) -> None:
        token_ids = torch.tensor([[1, 4, 9, 7]])
        embedded = self.model.embed_batch(token_ids)
        empty_states = tuple(self.model._layer_states)
        full_progress = []
        full = self.model.hidden_stateful_range(
            embedded,
            empty_states,
            start_pos=0,
            start_layer=0,
            stop_layer=self.config.n_layers,
            progress=full_progress.append,
        )

        self.assertEqual(self.model.next_position, 0)
        self.assertIsNone(self.model.state_batch_size)
        self.assertFalse(self.model.state_poisoned)
        self.assertTrue(all(state is None for state in self.model._layer_states))
        self.assertEqual(full.evidence.layers_executed, self.config.n_layers)
        self.assertEqual(full.evidence.linear_calls, 31)
        self.assertGreater(full.evidence.staged_state_bytes, 0)
        self.assertEqual(
            [row["layer"] for row in full_progress],
            list(range(self.config.n_layers)),
        )

        split_progress = []
        lower = self.model.hidden_stateful_range(
            embedded,
            empty_states,
            start_pos=0,
            start_layer=0,
            stop_layer=2,
            progress=split_progress.append,
        )
        upper = self.model.hidden_stateful_range(
            lower.hidden,
            lower.layer_states,
            start_pos=0,
            start_layer=2,
            stop_layer=self.config.n_layers,
            graft_history=lower.graft_history,
            progress=split_progress.append,
        )
        self.assertTrue(torch.equal(upper.hidden, full.hidden))
        _assert_layer_states_equal(self, upper.layer_states, full.layer_states)
        self.assertEqual(
            lower.evidence.linear_calls + upper.evidence.linear_calls,
            full.evidence.linear_calls,
        )
        self.assertEqual(
            [row["layer"] for row in split_progress],
            list(range(self.config.n_layers)),
        )
        explicit_final = self.model.finalize_hidden(full.hidden)

        range_method = self.model.hidden_stateful_range
        with mock.patch.object(
            self.model, "hidden_stateful_range", wraps=range_method
        ) as range_call:
            committed, evidence = self.model.prefill(token_ids)
        self.assertTrue(torch.equal(committed, explicit_final))
        _assert_layer_states_equal(self, self.model._layer_states, full.layer_states)
        self.assertEqual(evidence[0].linear_calls, 31)
        range_call.assert_called_once()
        self.assertEqual(range_call.call_args.kwargs["start_layer"], 0)
        self.assertEqual(
            range_call.call_args.kwargs["stop_layer"], self.config.n_layers
        )

    def test_stateful_layer_range_split_decode_is_bit_exact_and_non_committing(
        self,
    ) -> None:
        self.model.prefill([[1, 4, 9]])
        committed_objects = tuple(self.model._layer_states)
        committed_values = _clone_layer_states(self.model._layer_states)
        embedded = self.model.embed_batch([[7]])

        full = self.model.hidden_stateful_range(
            embedded,
            committed_objects,
            start_pos=3,
            start_layer=0,
            stop_layer=self.config.n_layers,
            graft_history=self.model._graft_history,
        )
        lower = self.model.hidden_stateful_range(
            embedded,
            committed_objects,
            start_pos=3,
            start_layer=0,
            stop_layer=2,
            graft_history=self.model._graft_history,
        )
        upper = self.model.hidden_stateful_range(
            lower.hidden,
            lower.layer_states,
            start_pos=3,
            start_layer=2,
            stop_layer=self.config.n_layers,
            graft_history=lower.graft_history,
        )

        self.assertTrue(torch.equal(upper.hidden, full.hidden))
        _assert_layer_states_equal(self, upper.layer_states, full.layer_states)
        self.assertEqual(self.model.next_position, 3)
        self.assertFalse(self.model.state_poisoned)
        self.assertTrue(
            all(
                current is original
                for current, original in zip(
                    self.model._layer_states, committed_objects, strict=True
                )
            )
        )
        _assert_layer_states_equal(self, self.model._layer_states, committed_values)

        decoded, _ = self.model.decode([[7]])
        self.assertTrue(torch.equal(decoded, self.model.finalize_hidden(full.hidden)))
        _assert_layer_states_equal(self, self.model._layer_states, full.layer_states)

    def test_hidden_stateful_decode_consumes_prior_cache_layer_by_layer(self) -> None:
        self.model.prefill([[1, 4, 9]])
        embedded = self.model.embed_batch([[7]])
        staged = self.model.hidden_stateful_range(
            embedded,
            self.model._layer_states,
            start_pos=3,
            start_layer=0,
            stop_layer=self.config.n_layers,
            graft_history=self.model._graft_history,
        )
        expected_hidden = self.model.finalize_hidden(staged.hidden)
        self.pager.release()

        prior_refs = _weak_layer_state_tensor_refs(self.model._layer_states)
        self.assertTrue(all(ref() is not None for row in prior_refs for ref in row))
        release_snapshots: list[tuple[bool, ...]] = []
        original_forward_layer = self.model._forward_layer

        def observe_prior_lifetimes(*args, **kwargs):
            release_snapshots.append(
                tuple(all(ref() is None for ref in row) for row in prior_refs)
            )
            return original_forward_layer(*args, **kwargs)

        with mock.patch.object(
            self.model,
            "_forward_layer",
            new=observe_prior_lifetimes,
        ):
            decoded, evidence = self.model.decode([[7]])

        self.assertEqual(
            release_snapshots,
            [
                tuple(index < layer for index in range(self.config.n_layers))
                for layer in range(self.config.n_layers)
            ],
        )
        self.assertTrue(all(ref() is None for row in prior_refs for ref in row))
        self.assertTrue(torch.equal(decoded, expected_hidden))
        _assert_layer_states_equal(self, self.model._layer_states, staged.layer_states)
        self.assertEqual((evidence.start_pos, evidence.end_pos), (3, 4))
        self.assertEqual(evidence.linear_calls, staged.evidence.linear_calls)

    def test_hidden_stateful_decode_releases_pending_stage_before_layer_zero(
        self,
    ) -> None:
        self.model.prefill([[1, 4, 9]])
        stage = self.model.stage_continuation_block([[7, 6]])
        pending = self.model._pending_block_stage
        self.assertIsNotNone(pending)
        assert pending is not None
        pending_refs = _weak_layer_state_tensor_refs(list(pending.layer_states))
        del pending
        self.assertTrue(all(ref() is not None for row in pending_refs for ref in row))

        layer_zero_pending_released: list[bool] = []
        original_forward_layer = self.model._forward_layer

        def observe_pending_lifetime(*args, **kwargs):
            if kwargs["layer"] == 0:
                layer_zero_pending_released.append(
                    all(ref() is None for row in pending_refs for ref in row)
                )
            return original_forward_layer(*args, **kwargs)

        with mock.patch.object(
            self.model,
            "_forward_layer",
            new=observe_pending_lifetime,
        ):
            decoded, evidence = self.model.decode([[5]])

        self.assertEqual(layer_zero_pending_released, [True])
        self.assertTrue(all(ref() is None for row in pending_refs for ref in row))
        self.assertEqual(tuple(decoded.shape), (1, 1, self.config.dim))
        self.assertEqual((evidence.start_pos, evidence.end_pos), (3, 4))
        with self.assertRaisesRegex(Qwen38RuntimeError, "stale or foreign"):
            self.model.commit_continuation_block(stage)

    def test_consuming_decode_failure_poisons_then_reset_recovers(self) -> None:
        self.model.prefill([[1, 4, 9]])
        prior_refs = _weak_layer_state_tensor_refs(self.model._layer_states)
        original_mlp = self.model._mlp

        def fail_on_layer_two(hidden, *, layer):
            if layer == 2:
                raise RuntimeError("consuming decode failure")
            return original_mlp(hidden, layer=layer)

        with mock.patch.object(self.model, "_mlp", new=fail_on_layer_two):
            with self.assertRaisesRegex(RuntimeError, "consuming decode failure"):
                self.model.decode([[7]])

        self.assertTrue(self.model.state_poisoned)
        self.assertEqual(self.model.next_position, 0)
        self.assertIsNone(self.model.state_batch_size)
        self.assertEqual(self.model.state_bytes, 0)
        self.assertTrue(all(state is None for state in self.model._layer_states))
        self.assertTrue(all(ref() is None for row in prior_refs for ref in row))
        with self.assertRaisesRegex(Qwen38RuntimeError, "poisoned"):
            self.model.prefill([[1]], reset=False)

        self.model.reset_state(release=True)
        self.model.prefill([[1, 4, 9]], reset=False)
        recovered, evidence = self.model.decode([[7]])
        self.assertEqual(tuple(recovered.shape), (1, 1, self.config.dim))
        self.assertTrue(torch.isfinite(recovered).all())
        self.assertEqual((evidence.start_pos, evidence.end_pos), (3, 4))
        self.assertFalse(self.model.state_poisoned)

    def test_stateful_layer_range_failure_rolls_back_without_poisoning(self) -> None:
        self.model.prefill([[1, 4, 9]])
        committed_objects = tuple(self.model._layer_states)
        committed_values = _clone_layer_states(self.model._layer_states)
        embedded = self.model.embed_batch([[7]])
        original_mlp = self.model._mlp

        def fail_on_layer_two(hidden, *, layer):
            if layer == 2:
                raise RuntimeError("range transaction failure")
            return original_mlp(hidden, layer=layer)

        with mock.patch.object(self.model, "_mlp", side_effect=fail_on_layer_two):
            with self.assertRaisesRegex(RuntimeError, "range transaction failure"):
                self.model.hidden_stateful_range(
                    embedded,
                    committed_objects,
                    start_pos=3,
                    start_layer=0,
                    stop_layer=self.config.n_layers,
                    graft_history=self.model._graft_history,
                )

        self.assertEqual(self.model.next_position, 3)
        self.assertEqual(self.model.state_batch_size, 1)
        self.assertFalse(self.model.state_poisoned)
        self.assertTrue(
            all(
                current is original
                for current, original in zip(
                    self.model._layer_states, committed_objects, strict=True
                )
            )
        )
        _assert_layer_states_equal(self, self.model._layer_states, committed_values)
        recovered, evidence = self.model.decode([[7]])
        self.assertEqual(tuple(recovered.shape), (1, 1, self.config.dim))
        self.assertEqual(evidence.end_pos, 4)

    def test_stateful_layer_range_avoids_per_layer_metrics_without_progress(
        self,
    ) -> None:
        hidden = self.model.embed_batch([[1, 4]])
        empty = (None,) * self.config.n_layers
        with mock.patch.object(
            self.source, "metrics", wraps=self.source.metrics
        ) as metrics:
            self.model.hidden_stateful_range(
                hidden,
                empty,
                start_pos=0,
                start_layer=0,
                stop_layer=self.config.n_layers,
            )
        self.assertEqual(metrics.call_count, 4)

        events: list[dict[str, object]] = []
        with mock.patch.object(
            self.source, "metrics", wraps=self.source.metrics
        ) as metrics:
            self.model.hidden_stateful_range(
                hidden,
                empty,
                start_pos=0,
                start_layer=0,
                stop_layer=self.config.n_layers,
                progress=events.append,
            )
        self.assertEqual(len(events), self.config.n_layers)
        self.assertEqual(metrics.call_count, 4 + 2 * self.config.n_layers)

    def test_stateful_layer_range_rejects_invalid_boundaries(self) -> None:
        hidden = self.model.embed_batch([[1, 4]])
        empty_states = tuple(self.model._layer_states)
        invalid_ranges = ((-1, 1), (3, 2), (0, self.config.n_layers + 1))
        for start_layer, stop_layer in invalid_ranges:
            with self.subTest(start_layer=start_layer, stop_layer=stop_layer):
                with self.assertRaisesRegex(ValueError, "layer range"):
                    self.model.hidden_stateful_range(
                        hidden,
                        empty_states,
                        start_pos=0,
                        start_layer=start_layer,
                        stop_layer=stop_layer,
                    )
        with self.assertRaisesRegex(TypeError, "start_layer"):
            self.model.hidden_stateful_range(
                hidden,
                empty_states,
                start_pos=0,
                start_layer=True,
                stop_layer=1,
            )
        with self.assertRaisesRegex(ValueError, "prior_states"):
            self.model.hidden_stateful_range(
                hidden,
                empty_states[:-1],
                start_pos=0,
                start_layer=0,
                stop_layer=1,
            )
        with self.assertRaisesRegex(Qwen38RuntimeError, "wrong state type"):
            self.model.hidden_stateful_range(
                hidden,
                empty_states,
                start_pos=0,
                start_layer=1,
                stop_layer=2,
            )

        empty = self.model.hidden_stateful_range(
            hidden,
            empty_states,
            start_pos=0,
            start_layer=0,
            stop_layer=0,
        )
        self.assertIs(empty.hidden, hidden)
        self.assertEqual(empty.layer_states, empty_states)
        self.assertEqual(empty.evidence.layers_executed, 0)
        self.assertEqual(empty.evidence.linear_calls, 0)

        self.model.prefill([[1, 4]])
        prior = list(self.model._layer_states)
        attention = prior[3]
        self.assertIsInstance(attention, AttentionState)
        malformed = AttentionState(
            key=attention.key.clone(),
            value=attention.value.clone(),
            crsa_log_usage=None,
        )
        malformed.value.resize_(1, 1, 1, 1)
        prior[3] = malformed
        linears_before = self.model._metric(self.pager, "linear_calls")
        with self.assertRaisesRegex(Qwen38RuntimeError, "value state"):
            self.model.hidden_stateful_range(
                self.model.embed_batch([[7]]),
                prior,
                start_pos=2,
                start_layer=0,
                stop_layer=self.config.n_layers,
            )
        self.assertEqual(self.model._metric(self.pager, "linear_calls"), linears_before)

    def test_stateful_boundaries_preserve_committed_prefix(self) -> None:
        with self.assertRaisesRegex(ValueError, "completed prefill"):
            self.model.decode([[1]])

        self.model.prefill([[1, 2], [3, 4]])
        with self.assertRaisesRegex(ValueError, "exactly one token"):
            self.model.decode([[5, 6], [7, 8]])
        with self.assertRaisesRegex(ValueError, "batch size differs"):
            self.model.decode([[5]])
        self.assertEqual(self.model.next_position, 2)
        self.assertFalse(self.model.state_poisoned)

        decoded, evidence = self.model.decode([[5], [6]])
        self.assertEqual(tuple(decoded.shape), (2, 1, self.config.dim))
        self.assertEqual(evidence.end_pos, 3)

    def test_failed_stateful_forward_poison_latches_until_reset(self) -> None:
        with mock.patch.object(
            self.model, "_mlp", side_effect=RuntimeError("injected failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                self.model.prefill([[1, 2]])

        self.assertTrue(self.model.state_poisoned)
        self.assertEqual(self.model.next_position, 0)
        self.assertIsNone(self.model.state_batch_size)
        self.assertEqual(self.model.state_bytes, 0)
        with self.assertRaisesRegex(Qwen38RuntimeError, "poisoned"):
            self.model.prefill([[1]], reset=False)

        self.model.reset_state(release=True)
        recovered, evidence = self.model.prefill([[1]], reset=False)
        self.assertEqual(tuple(recovered.shape), (1, 1, self.config.dim))
        self.assertEqual(evidence[0].end_pos, 1)
        self.assertFalse(self.model.state_poisoned)

    def test_greedy_generation_uses_committed_decode_state_and_eos(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceed max_seq_len"):
            self.model.generate_greedy([[1] * 32], max_new_tokens=1)

        generated, evidence = self.model.generate_greedy(
            [[1, 4]], max_new_tokens=3, head_block_rows=7
        )
        self.assertEqual(len(generated), 3)
        self.assertTrue(all(0 <= token < self.config.vocab_size for token in generated))
        self.assertEqual(evidence.generated_token_ids, generated)
        self.assertEqual(evidence.forward_passes, 4)
        self.assertTrue(evidence.general_generation)
        self.assertTrue(evidence.stateful_cache)
        self.assertFalse(evidence.stopped_on_eos)
        self.assertEqual(self.model.next_position, 5)

        one_shot, _ = self.model.forward_prefill(torch.tensor([[1, 4, *generated]]))
        continued, continued_evidence = self.model.decode([[6]])
        one_shot_continued, _ = self.model.forward_prefill(
            torch.tensor([[1, 4, *generated, 6]])
        )
        torch.testing.assert_close(
            continued, one_shot_continued[:, -1:], rtol=1e-5, atol=1e-6
        )
        self.assertEqual(continued_evidence.start_pos, 5)
        self.assertEqual(tuple(one_shot.shape), (1, 5, self.config.dim))

        stopped, stopped_evidence = self.model.generate_greedy(
            [[1, 4]],
            max_new_tokens=3,
            eos_token_ids=(generated[0],),
            head_block_rows=7,
        )
        self.assertEqual(stopped, generated[:1])
        self.assertTrue(stopped_evidence.stopped_on_eos)
        self.assertEqual(stopped_evidence.forward_passes, 2)
        self.assertEqual(self.model.next_position, 3)

    def test_one_shot_greedy_skips_the_unused_final_decode(self) -> None:
        committed, committed_evidence = self.model.generate_greedy(
            [[1, 4]],
            max_new_tokens=3,
            head_block_rows=7,
        )
        self.model.reset_state(release=True)

        one_shot, one_shot_evidence = self.model.generate_greedy(
            [[1, 4]],
            max_new_tokens=3,
            head_block_rows=7,
            retain_final_state=False,
        )

        self.assertEqual(one_shot, committed)
        self.assertEqual(
            one_shot_evidence.forward_passes,
            committed_evidence.forward_passes - 1,
        )
        self.assertFalse(one_shot_evidence.final_state_committed)
        self.assertEqual(self.model.next_position, 4)
        self.assertIsNone(self.model._pending_block_stage)

        with self.assertRaisesRegex(TypeError, "retain_final_state"):
            self.model.generate_greedy(
                [[1, 4]],
                max_new_tokens=1,
                retain_final_state=1,  # type: ignore[arg-type]
            )

    def test_greedy_generation_from_exact_anchor_skips_prompt_prefill(self) -> None:
        prompt = [[1, 4]]
        baseline, baseline_evidence = self.model.generate_greedy(
            prompt,
            max_new_tokens=2,
            head_block_rows=7,
        )
        baseline_states = _clone_layer_states(self.model._layer_states)
        baseline_snapshot_root = self.root / "exact-baseline"
        baseline_snapshot_root.mkdir()
        baseline_snapshot = self.model.save_state(baseline_snapshot_root / "state.json")
        self.model.reset_state(release=True)
        hidden, _ = self.model.prefill(prompt)
        cache = SemanticStateAnchorCache(self.root / "generation-cache")
        cache.store(
            self.model,
            prompt[0],
            boundary_kind="turn",
            seed_hidden=hidden[:, -1:],
        )

        restored_model = StreamedQwen38(
            self.config,
            self.pager,
            max_batch_size=3,
            max_seq_len=32,
        )
        restored = cache.restore_deepest(restored_model, prompt[0])
        self.assertIsNotNone(restored)
        assert restored is not None and restored.seed_hidden is not None
        with mock.patch.object(
            restored_model,
            "prefill",
            wraps=restored_model.prefill,
        ) as prefill:
            generated, evidence = restored_model.generate_greedy(
                prompt,
                max_new_tokens=2,
                restored_prefix_length=restored.anchor.prefix_length,
                restored_seed_hidden=restored.seed_hidden,
                head_block_rows=7,
            )

        self.assertEqual(generated, baseline)
        self.assertEqual(
            evidence.generated_token_ids, baseline_evidence.generated_token_ids
        )
        self.assertEqual(evidence.forward_passes, 2)
        self.assertEqual(baseline_evidence.forward_passes, 3)
        self.assertLess(
            evidence.source_body_bytes,
            baseline_evidence.source_body_bytes,
        )
        self.assertEqual(restored_model.next_position, 4)
        _assert_layer_states_equal(
            self,
            restored_model._layer_states,
            baseline_states,
        )
        restored_snapshot_root = self.root / "exact-restored"
        restored_snapshot_root.mkdir()
        restored_snapshot = restored_model.save_state(
            restored_snapshot_root / "state.json"
        )
        self.assertEqual(
            restored_snapshot["payload_sha256"],
            baseline_snapshot["payload_sha256"],
        )
        self.assertEqual(
            restored_snapshot["manifest_body_sha256"],
            baseline_snapshot["manifest_body_sha256"],
        )
        prefill.assert_not_called()

    def test_greedy_generation_from_anchor_prefills_only_prompt_suffix(self) -> None:
        prefix = [[1, 4]]
        prompt = [[1, 4, 9]]
        one_shot, one_shot_evidence = self.model.generate_greedy(
            prompt,
            max_new_tokens=1,
            head_block_rows=7,
        )
        self.model.reset_state(release=True)
        # Exact state parity requires the same prefix/suffix execution graph;
        # one-shot GEMM versus segmented GEMV is a separate numerical control.
        self.model.prefill(prefix)
        baseline, baseline_evidence = self.model.generate_greedy(
            prompt,
            max_new_tokens=1,
            restored_prefix_length=len(prefix[0]),
            head_block_rows=7,
        )
        self.assertEqual(baseline, one_shot)
        self.assertEqual(
            baseline_evidence.generated_token_ids,
            one_shot_evidence.generated_token_ids,
        )
        baseline_states = _clone_layer_states(self.model._layer_states)
        baseline_snapshot_root = self.root / "suffix-baseline"
        baseline_snapshot_root.mkdir()
        baseline_snapshot = self.model.save_state(baseline_snapshot_root / "state.json")
        self.model.reset_state(release=True)
        hidden, _ = self.model.prefill(prefix)
        cache = SemanticStateAnchorCache(self.root / "suffix-generation-cache")
        cache.store(
            self.model,
            prefix[0],
            boundary_kind="turn",
            seed_hidden=hidden[:, -1:],
        )

        restored_model = StreamedQwen38(
            self.config,
            self.pager,
            max_batch_size=3,
            max_seq_len=32,
        )
        restored = cache.restore_deepest(restored_model, prompt[0])
        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertFalse(restored.exact_prefix)
        self.assertIsNone(restored.seed_hidden)
        with mock.patch.object(
            restored_model,
            "prefill",
            wraps=restored_model.prefill,
        ) as prefill:
            generated, evidence = restored_model.generate_greedy(
                prompt,
                max_new_tokens=1,
                restored_prefix_length=restored.anchor.prefix_length,
                head_block_rows=7,
            )

        self.assertEqual(generated, baseline)
        self.assertEqual(
            evidence.generated_token_ids,
            baseline_evidence.generated_token_ids,
        )
        self.assertEqual(evidence.forward_passes, 2)
        prefill.assert_called_once()
        self.assertTrue(torch.equal(prefill.call_args.args[0], torch.tensor([[9]])))
        self.assertFalse(prefill.call_args.kwargs["reset"])
        _assert_layer_states_equal(
            self,
            restored_model._layer_states,
            baseline_states,
        )
        restored_snapshot_root = self.root / "suffix-restored"
        restored_snapshot_root.mkdir()
        restored_snapshot = restored_model.save_state(
            restored_snapshot_root / "state.json"
        )
        self.assertEqual(
            restored_snapshot["payload_sha256"],
            baseline_snapshot["payload_sha256"],
        )
        self.assertEqual(
            restored_snapshot["manifest_body_sha256"],
            baseline_snapshot["manifest_body_sha256"],
        )

    def test_gc_policy_preserves_generated_tokens_hidden_and_state(self) -> None:
        source = Streamer.from_local(self.root, budget_mb=20, use_cache=False)
        pagers = [
            Qwen38WeightPager(
                source,
                device="cpu",
                compute_dtype="float32",
                max_resident_bytes=2 * 1024**2,
                gc_interval_boundaries=interval,
                gc_rss_limit_bytes=2**63 - 1,
            )
            for interval in (1, 10_000)
        ]
        models = [
            StreamedQwen38(
                self.config,
                pager,
                max_batch_size=1,
                max_seq_len=16,
            )
            for pager in pagers
        ]
        try:
            outputs = []
            with mock.patch(
                "immer.runtimes.qwen3_8.pager._process_rss_bytes",
                return_value=1,
            ):
                for model in models:
                    tokens, _evidence = model.generate_greedy(
                        [[1, 4]],
                        max_new_tokens=2,
                        head_block_rows=7,
                    )
                    outputs.append(
                        (
                            tokens,
                            model._layer_states,
                            model._graft_history,
                            model.next_position,
                        )
                    )

            self.assertEqual(outputs[0][0], outputs[1][0])
            _assert_layer_states_equal(self, outputs[0][1], outputs[1][1])
            self.assertIsNone(outputs[0][2])
            self.assertIsNone(outputs[1][2])
            self.assertEqual(outputs[0][3], outputs[1][3])
            self.assertGreater(pagers[0].metrics()["gc_collections_interval"], 0)
            self.assertEqual(pagers[1].metrics()["gc_collections_interval"], 0)
            self.assertGreater(pagers[1].metrics()["gc_collections_skipped"], 0)
            forced_before = [
                pager.metrics()["gc_collections_forced"] for pager in pagers
            ]
            for model in models:
                model.reset_state(release=True)
            self.assertEqual(
                [pager.metrics()["gc_collections_forced"] for pager in pagers],
                [value + 1 for value in forced_before],
            )
        finally:
            for pager in pagers:
                pager.close()
            source.close()

    def test_right_padding_preserves_each_valid_prefix(self) -> None:
        batch_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]])
        mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
        padded, _ = self.model.forward_prefill(batch_ids, token_mask=mask)
        exact, _ = self.model.forward_prefill(torch.tensor([[5, 6]]))
        torch.testing.assert_close(padded[1, :2], exact[0], rtol=1e-5, atol=1e-6)

    def test_qwen_draft_verifier_uses_native_3d_resume_shape(self) -> None:
        verifier = Qwen38DraftVerifier(self.model, layer_retries=0, head_retries=0)
        checkpoints = []
        first = verifier.verify(
            [[1, 2]],
            [[3]],
            padding_token_id=0,
            checkpoint=checkpoints.append,
            head_block_rows=7,
        )
        self.assertEqual(first.evidence.schema, DRAFT_VERIFICATION_SCHEMA)
        self.assertEqual(first.evidence.layer_calls, 4)
        self.assertEqual(tuple(checkpoints[1].hidden.shape), (1, 3, self.config.dim))

        resumed = verifier.verify(
            [[1, 2]],
            [[3]],
            padding_token_id=0,
            resume_state=checkpoints[1],
            head_block_rows=7,
        )
        self.assertEqual(resumed.rows, first.rows)
        self.assertEqual(resumed.evidence.layer_calls, 4)

    def test_qwen_draft_verifier_accepts_both_published_stop_tokens(self) -> None:
        from test_deepseek_v4_draft_verification import _FakeModel

        model = _FakeModel((2, 99))
        report = Qwen38DraftVerifier(model).verify(
            [[1]],
            [[2]],
            eos_token_id=7,
            eos_token_ids=(7, 99),
        )

        self.assertTrue(report.all_verified)
        self.assertEqual(report.rows[0].target_token_ids, (2, 99))
        self.assertTrue(report.rows[0].eos_verified)
        self.assertIsNone(report.rows[0].first_mismatch_index)
        self.assertEqual(report.evidence.schema, DRAFT_VERIFICATION_SCHEMA)

        with self.assertRaisesRegex(ValueError, "exclude every"):
            Qwen38DraftVerifier(_FakeModel((2, 7))).verify(
                [[1]],
                [[99]],
                eos_token_ids=(7, 99),
            )

    def test_qwen_crsa_sidecar_is_causal_and_energy_stable(self) -> None:
        torch.manual_seed(43)
        hidden = torch.randn(2, 7, self.config.dim)
        changed = hidden.clone()
        changed[:, 5:] *= 100.0
        graft = Qwen38StableCrsaGraft(mode="crsa", alpha=0.1)

        output, evidence = graft(hidden, return_evidence=True)
        perturbed = graft(changed)

        self.assertEqual(evidence.schema, STABLE_GRAFT_EVIDENCE_SCHEMA)
        self.assertTrue(evidence.strict_causal)
        self.assertEqual(evidence.future_weight_max_abs, 0.0)
        self.assertTrue(torch.equal(output[:, :5], perturbed[:, :5]))
        expected_rms = hidden.reshape(2, 7, 4, 3).square().mean(-1).sqrt()
        actual_rms = output.reshape(2, 7, 4, 3).square().mean(-1).sqrt()
        torch.testing.assert_close(actual_rms, expected_rms, rtol=2e-6, atol=2e-6)

    def test_stateful_qwen_crsa_matches_batched_and_tokenwise_prefill(self) -> None:
        graft = Qwen38StableCrsaGraft(mode="crsa", alpha=0.1)
        model = StreamedQwen38(
            self.config,
            self.pager,
            graft=graft,
            graft_layer=1,
            max_batch_size=3,
            max_seq_len=32,
        )
        token_ids = torch.tensor([[1, 4, 9, 7]])
        batched, batched_evidence = model.prefill(token_ids, tokenwise=False)
        model.reset_state()
        tokenwise, token_evidence = model.prefill(token_ids, tokenwise=True)

        torch.testing.assert_close(batched, tokenwise, rtol=1e-5, atol=1e-6)
        self.assertEqual(batched_evidence[0].graft_history_tokens, 4)
        self.assertEqual(token_evidence[-1].graft_history_tokens, 4)
        self.assertGreater(model.state_bytes, self.model.state_bytes)

        model.reset_state()
        embedded = model.embed_batch(token_ids)
        empty_states = tuple(model._layer_states)
        full = model.hidden_stateful_range(
            embedded,
            empty_states,
            start_pos=0,
            start_layer=0,
            stop_layer=self.config.n_layers,
        )
        below_graft = model.hidden_stateful_range(
            embedded,
            empty_states,
            start_pos=0,
            start_layer=0,
            stop_layer=1,
        )
        above_graft = model.hidden_stateful_range(
            below_graft.hidden,
            below_graft.layer_states,
            start_pos=0,
            start_layer=1,
            stop_layer=self.config.n_layers,
            graft_history=below_graft.graft_history,
        )
        self.assertTrue(torch.equal(above_graft.hidden, full.hidden))
        _assert_layer_states_equal(self, above_graft.layer_states, full.layer_states)
        self.assertIsNone(below_graft.graft_history)
        self.assertTrue(torch.equal(above_graft.graft_history, full.graft_history))
        self.assertEqual(tuple(full.graft_history.shape), (1, 4, self.config.dim))
        self.assertEqual(model.next_position, 0)
        self.assertTrue(all(state is None for state in model._layer_states))

    def test_graft_continuation_block_delays_history_commit(self) -> None:
        graft = Qwen38StableCrsaGraft(mode="crsa", alpha=0.1)
        model = StreamedQwen38(
            self.config,
            self.pager,
            graft=graft,
            graft_layer=1,
            max_batch_size=1,
            max_seq_len=16,
        )
        model.prefill([[1, 4]])
        committed_history = model._graft_history

        stage = model.stage_continuation_block([[9, 7]])

        self.assertIs(model._graft_history, committed_history)
        self.assertEqual(tuple(committed_history.shape), (1, 2, self.config.dim))
        self.assertEqual(stage.evidence.graft_history_tokens, 4)
        model.commit_continuation_block(stage)
        self.assertEqual(tuple(model._graft_history.shape), (1, 4, self.config.dim))
        self.assertEqual(model.next_position, 4)

        stale = model.stage_continuation_block([[8, 6]])
        graft.alpha = 0.9
        with self.assertRaisesRegex(Qwen38RuntimeError, "configuration changed"):
            model.commit_continuation_block(stale)
        self.assertEqual(model.next_position, 4)

    def test_bfloat16_graft_block_matches_tokenwise_state_bit_exactly(self) -> None:
        pagers = [
            Qwen38WeightPager(
                self.source,
                device="cpu",
                compute_dtype="bfloat16",
                max_resident_bytes=2 * 1024**2,
            )
            for _ in range(2)
        ]
        models = [
            StreamedQwen38(
                self.config,
                pager,
                graft=Qwen38StableCrsaGraft(mode="crsa", alpha=0.1),
                graft_layer=1,
                max_batch_size=1,
                max_seq_len=16,
            )
            for pager in pagers
        ]
        block_model, token_model = models
        try:
            block_model.prefill([[1, 4]])
            stage = block_model.stage_continuation_block([[9, 7]])

            token_model.prefill([[1, 4]])
            first, _ = token_model.decode([[9]])
            second, _ = token_model.decode([[7]])
            expected = torch.cat((first, second), dim=1)

            self.assertTrue(torch.equal(stage.hidden, expected))
            actual, _ = block_model.commit_continuation_block(stage)
            self.assertTrue(torch.equal(actual, expected))
            _assert_layer_states_equal(
                self, block_model._layer_states, token_model._layer_states
            )
            self.assertTrue(
                torch.equal(block_model._graft_history, token_model._graft_history)
            )
        finally:
            for pager in pagers:
                pager.close()


class Qwen38NativePrefixSinkhornOperatorObserverTests(unittest.TestCase):
    @staticmethod
    def _fixture():
        from immer.runtimes.qwen3_8.native_crsa import Qwen38NativeHeadCrsa

        torch.manual_seed(20260827)
        logits = torch.randn(1, 24, 4, 4, dtype=torch.float32)
        allowed = torch.ones(4, 4, dtype=torch.bool).tril().reshape(1, 1, 4, 4)
        base = torch.softmax(logits.masked_fill(~allowed, -torch.inf), -1)
        base = base.masked_fill(~allowed, 0.0)
        return Qwen38NativeHeadCrsa(alpha=0.1), logits, base, allowed

    def test_preblend_rows_match_streaming_and_callback_cannot_change_math(
        self,
    ) -> None:
        from immer.attention.crsa.operators import streaming_prefix_log
        from immer.runtimes.qwen3_8.native_crsa import (
            NativePrefixSinkhornOperatorObserver,
        )

        intervention, logits, base, allowed = self._fixture()
        blocks = []
        observer = NativePrefixSinkhornOperatorObserver(blocks.append, max_positions=4)
        actual, actual_usage, actual_evidence = intervention.route(
            logits,
            base,
            query_start=0,
            allowed=allowed,
            prior_log_usage=None,
            tokenwise_usage=True,
            routed_operator_observer=observer,
        )
        self.assertEqual(
            [row.query_positions for row in blocks],
            [(0,), (1,), (2,), (3,)],
        )
        self.assertEqual(
            [row.key_positions for row in blocks],
            [tuple(range(length)) for length in range(1, 5)],
        )

        usage = None
        expected_selected_rows = []
        for position in range(4):
            routed, usage = streaming_prefix_log(
                logits[
                    :,
                    intervention.head_indices,
                    position : position + 1,
                    : position + 1,
                ],
                intervention.spec,
                query_start=position,
                prior_log_usage=usage,
                allowed=allowed[:, :, position : position + 1, : position + 1],
            )
            expected_selected_rows.append(routed)
            np.testing.assert_array_equal(
                blocks[position].operators,
                routed.detach().to(torch.float64).numpy(),
            )
            self.assertFalse(blocks[position].operators.flags.writeable)
        assert usage is not None
        self.assertTrue(torch.equal(actual_usage, usage))
        expected_selected = torch.cat(
            [
                torch.nn.functional.pad(row, (0, 4 - row.shape[-1]))
                for row in expected_selected_rows
            ],
            dim=2,
        )
        expected = base.clone()
        selected_base = base[:, intervention.head_indices]
        expected[:, intervention.head_indices] = selected_base + intervention.alpha * (
            expected_selected - selected_base
        )
        self.assertTrue(torch.equal(actual, expected))

        mutated_blocks = []

        def mutate_detached(block):
            block.operators.setflags(write=True)
            block.operators.fill(0.0)
            mutated_blocks.append(block)

        mutated, mutated_usage, mutated_evidence = intervention.route(
            logits,
            base,
            query_start=0,
            allowed=allowed,
            prior_log_usage=None,
            tokenwise_usage=True,
            routed_operator_observer=NativePrefixSinkhornOperatorObserver(
                mutate_detached, max_positions=4
            ),
        )
        self.assertEqual(len(mutated_blocks), 4)
        self.assertTrue(torch.equal(mutated, actual))
        self.assertTrue(torch.equal(mutated_usage, actual_usage))
        self.assertEqual(mutated_evidence, actual_evidence)

    def test_bound_disabled_path_and_alpha_zero_emit_nothing(self) -> None:
        from immer.runtimes.qwen3_8.native_crsa import (
            NativePrefixSinkhornOperatorObserver,
            Qwen38NativeHeadCrsa,
        )

        intervention, logits, base, allowed = self._fixture()
        baseline, usage, evidence = intervention.route(
            logits,
            base,
            query_start=0,
            allowed=allowed,
            prior_log_usage=None,
            tokenwise_usage=True,
        )
        bounded_rows = []
        observed, observed_usage, observed_evidence = intervention.route(
            logits,
            base,
            query_start=0,
            allowed=allowed,
            prior_log_usage=None,
            tokenwise_usage=True,
            routed_operator_observer=NativePrefixSinkhornOperatorObserver(
                bounded_rows.append, max_positions=3
            ),
        )
        self.assertEqual(
            [row.query_positions for row in bounded_rows], [(0,), (1,), (2,)]
        )
        self.assertTrue(torch.equal(observed, baseline))
        self.assertTrue(torch.equal(observed_usage, usage))
        self.assertEqual(observed_evidence, evidence)

        identity_rows = []
        identity, identity_usage, identity_evidence = Qwen38NativeHeadCrsa(
            alpha=0.0
        ).route(
            logits,
            base,
            query_start=0,
            allowed=allowed,
            prior_log_usage=None,
            tokenwise_usage=True,
            routed_operator_observer=NativePrefixSinkhornOperatorObserver(
                identity_rows.append, max_positions=4
            ),
        )
        self.assertIs(identity, base)
        self.assertIsNone(identity_usage)
        self.assertTrue(identity_evidence.identity)
        self.assertEqual(identity_rows, [])


class Qwen38NativeHeadCrsaModelTests(unittest.TestCase):
    def setUp(self) -> None:
        from immer.runtimes.qwen3_8.native_crsa import Qwen38NativeHeadCrsa

        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = _native_tiny_config()
        save_file(_tiny_weights(self.config), self.root / "model.safetensors")
        self.source = Streamer.from_local(self.root, budget_mb=20, use_cache=False)
        self.pager = Qwen38WeightPager(
            self.source,
            device="cpu",
            compute_dtype="float32",
            max_resident_bytes=2 * 1024**2,
        )
        self.observed = []
        self.intervention = Qwen38NativeHeadCrsa(alpha=0.1)
        self.model = StreamedQwen38(
            self.config,
            self.pager,
            native_head_crsa=self.intervention,
            native_head_crsa_observer=self.observed.append,
            max_batch_size=2,
            max_seq_len=16,
        )

    def tearDown(self) -> None:
        self.pager.close()
        self.source.close()
        self.temporary.cleanup()

    def test_native_layer_range_split_stages_evidence_without_committing(
        self,
    ) -> None:
        token_ids = torch.tensor([[1, 4, 9, 7]])
        embedded = self.model.embed_batch(token_ids)
        empty_states = tuple(self.model._layer_states)

        full = self.model.hidden_stateful_range(
            embedded,
            empty_states,
            start_pos=0,
            start_layer=0,
            stop_layer=self.config.n_layers,
        )
        below_native = self.model.hidden_stateful_range(
            embedded,
            empty_states,
            start_pos=0,
            start_layer=0,
            stop_layer=27,
        )
        through_native = self.model.hidden_stateful_range(
            below_native.hidden,
            below_native.layer_states,
            start_pos=0,
            start_layer=27,
            stop_layer=self.config.n_layers,
            graft_history=below_native.graft_history,
        )

        self.assertTrue(torch.equal(through_native.hidden, full.hidden))
        _assert_layer_states_equal(self, through_native.layer_states, full.layer_states)
        self.assertEqual(len(full.native_head_crsa_evidence), 1)
        self.assertEqual(below_native.native_head_crsa_evidence, ())

        self.assertEqual(
            through_native.native_head_crsa_evidence,
            full.native_head_crsa_evidence,
        )
        self.assertEqual(self.observed, [])
        self.assertEqual(self.model.next_position, 0)
        self.assertFalse(self.model.state_poisoned)
        self.assertTrue(all(state is None for state in self.model._layer_states))

        malformed_states = list(full.layer_states)
        native_state = malformed_states[27]
        self.assertIsInstance(native_state, AttentionState)
        malformed_native = AttentionState(
            key=native_state.key.clone(),
            value=native_state.value.clone(),
            crsa_log_usage=native_state.crsa_log_usage.clone(),
        )
        malformed_native.crsa_log_usage.resize_(1, 4, 3)
        malformed_states[27] = malformed_native
        linears_before = self.model._metric(self.pager, "linear_calls")
        with self.assertRaisesRegex(Qwen38RuntimeError, "CRSA history.*invalid"):
            self.model.hidden_stateful_range(
                full.hidden,
                malformed_states,
                start_pos=0,
                start_layer=self.config.n_layers,
                stop_layer=self.config.n_layers,
            )
        self.assertEqual(self.model._metric(self.pager, "linear_calls"), linears_before)
        self.assertEqual(self.observed, [])

    def test_prefix_sinkhorn_action_switches_only_on_empty_state_and_counts_work(
        self,
    ) -> None:
        from immer.runtimes.qwen3_8.native_crsa import Qwen38NativeHeadCrsa

        enabled_identity = self.model._native_head_crsa_snapshot_identity()
        self.assertNotEqual(enabled_identity, {"kind": "none"})
        action_identity = self.model.prefix_sinkhorn_action_identity_sha256()

        self.model.set_native_head_crsa_enabled(False)
        self.assertEqual(
            self.model._native_head_crsa_snapshot_identity(),
            {"kind": "none"},
        )
        self.model.prefill([[1, 4]])
        disabled_state = self.model._layer_states[27]
        self.assertIsInstance(disabled_state, AttentionState)
        self.assertIsNone(disabled_state.crsa_log_usage)
        with self.assertRaisesRegex(Qwen38RuntimeError, "empty model state"):
            self.model.set_native_head_crsa_enabled(True)
        self.model.reset_state()
        self.model.set_native_head_crsa_enabled(True)
        self.assertEqual(
            self.model.prefix_sinkhorn_action_identity_sha256(),
            action_identity,
        )

        physical = StreamedQwen38(
            self.config,
            self.pager,
            native_head_crsa=Qwen38NativeHeadCrsa(
                alpha=1.0,
                replace_base_softmax=True,
            ),
            max_batch_size=1,
            max_seq_len=16,
        )
        physical.prefill([[1, 4]])
        metrics = physical.native_prefix_sinkhorn_metrics()
        self.assertEqual(metrics["base_softmax_head_rows_skipped"], 8)
        self.assertEqual(
            metrics["base_softmax_probability_elements_skipped"],
            16,
        )
        self.assertTrue(metrics["physical_replacement"])
        self.assertNotEqual(
            metrics["action_identity_sha256"],
            action_identity,
        )
        before_discarded_stage = dict(metrics)
        discarded = physical.stage_continuation_block([[9, 7]])
        after_discarded_stage = physical.native_prefix_sinkhorn_metrics()
        self.assertGreater(
            after_discarded_stage["base_softmax_head_rows_skipped"],
            before_discarded_stage["base_softmax_head_rows_skipped"],
        )
        physical.discard_continuation_block(discarded)
        self.assertEqual(
            physical.native_prefix_sinkhorn_metrics(),
            after_discarded_stage,
        )
        physical.reset_state(release=True)

    def test_attention_output_crystal_replays_prefix_sinkhorn_state_exactly(
        self,
    ) -> None:
        from immer.runtimes.qwen3_8.native_crsa import Qwen38NativeHeadCrsa

        pager = Qwen38WeightPager(
            self.source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=2 * 1024**2,
        )
        observed = []
        model = StreamedQwen38(
            self.config,
            pager,
            native_head_crsa=Qwen38NativeHeadCrsa(
                alpha=1.0,
                replace_base_softmax=True,
            ),
            native_head_crsa_observer=observed.append,
            max_batch_size=1,
            max_seq_len=16,
        )
        bank = AttentionOutputCrystalBank(
            self.root / "native-attention-output-crystals.json",
            AttentionOutputCrystalIdentity(
                runtime_math_sha256=(
                    model.attention_output_crystal_runtime_math_sha256()
                )
            ),
            max_cells=128,
        )
        model.attach_attention_output_crystal_bank(bank)
        try:
            model.prefill([[1, 4]])
            observed.clear()
            charged = model.stage_continuation_block([[9, 7]])
            self.assertEqual(observed, [])
            charged_hidden, _ = model.commit_continuation_block(charged)
            charged_evidence = tuple(observed)
            charged_states = _clone_layer_states(model._layer_states)
            self.assertEqual(len(charged_evidence), 2)

            model.reset_state()
            model.prefill([[1, 4]])
            physical_before_replay = model.native_prefix_sinkhorn_metrics()
            observed.clear()
            replay = model.stage_continuation_block([[9, 7]])
            self.assertEqual(observed, [])
            replay_hidden, _ = model.commit_continuation_block(replay)

            torch.testing.assert_close(
                replay_hidden,
                charged_hidden,
                rtol=0.0,
                atol=0.0,
            )
            _assert_layer_states_equal(self, model._layer_states, charged_states)
            self.assertEqual(tuple(observed), charged_evidence)
            metrics = bank.metrics()
            self.assertEqual(metrics.hit_count, 14)
            self.assertEqual(metrics.skipped_projection_calls_saved, 28)
            self.assertEqual(
                model.native_prefix_sinkhorn_metrics(),
                physical_before_replay,
            )
        finally:
            model.reset_state(release=True)
            pager.close()

    def test_native_continuation_block_delays_evidence_until_commit(self) -> None:
        self.model.prefill([[1, 4]])
        self.observed.clear()
        committed = self.model._layer_states[27]
        self.assertIsInstance(committed, AttentionState)

        stage = self.model.stage_continuation_block([[9, 7]])

        self.assertEqual(self.model.next_position, 2)
        self.assertIs(self.model._layer_states[27], committed)
        self.assertEqual(self.observed, [])
        self.model.commit_continuation_block(stage)
        self.assertEqual(self.model.next_position, 4)
        self.assertEqual(
            [
                (row.query_start, row.query_length, row.history_length_after)
                for row in self.observed
            ],
            [(2, 1, 3), (3, 1, 4)],
        )

    def test_bfloat16_native_block_matches_tokenwise_usage_bit_exactly(self) -> None:
        from immer.runtimes.qwen3_8.native_crsa import Qwen38NativeHeadCrsa

        pagers = [
            Qwen38WeightPager(
                self.source,
                device="cpu",
                compute_dtype="bfloat16",
                max_resident_bytes=2 * 1024**2,
            )
            for _ in range(2)
        ]
        observed = [[], []]
        models = [
            StreamedQwen38(
                self.config,
                pager,
                native_head_crsa=Qwen38NativeHeadCrsa(alpha=0.1),
                native_head_crsa_observer=rows.append,
                max_batch_size=1,
                max_seq_len=16,
            )
            for pager, rows in zip(pagers, observed, strict=True)
        ]
        block_model, token_model = models
        try:
            block_model.prefill([[1, 4]])
            observed[0].clear()
            stage = block_model.stage_continuation_block([[9, 7]])

            token_model.prefill([[1, 4]])
            observed[1].clear()
            first, _ = token_model.decode([[9]])
            second, _ = token_model.decode([[7]])
            expected = torch.cat((first, second), dim=1)

            self.assertTrue(torch.equal(stage.hidden, expected))
            actual, _ = block_model.commit_continuation_block(stage)
            self.assertTrue(torch.equal(actual, expected))
            _assert_layer_states_equal(
                self, block_model._layer_states, token_model._layer_states
            )
            self.assertEqual(
                [
                    (row.query_start, row.query_length, row.history_length_after)
                    for row in observed[0]
                ],
                [(2, 1, 3), (3, 1, 4)],
            )
            self.assertEqual(
                [
                    (row.query_start, row.query_length, row.history_length_after)
                    for row in observed[1]
                ],
                [(2, 1, 3), (3, 1, 4)],
            )
        finally:
            for pager in pagers:
                pager.close()

    def test_native_layer27_streaming_history_reset_and_poison_are_transactional(
        self,
    ) -> None:
        token_ids = torch.tensor([[1, 4, 9, 7]])
        one_shot, prefill_evidence = self.model.forward_prefill(token_ids)
        self.assertEqual(prefill_evidence.linear_calls, 217)

        self.observed.clear()
        self.model.prefill(token_ids[:, :3])
        state = self.model._layer_states[27]
        self.assertIsInstance(state, AttentionState)
        self.assertEqual(tuple(state.crsa_log_usage.shape), (1, 4, 3))
        usage_bytes = state.crsa_log_usage.numel() * state.crsa_log_usage.element_size()
        self.assertGreaterEqual(self.model.state_bytes, usage_bytes)
        decoded, _ = self.model.decode(token_ids[:, 3:])
        torch.testing.assert_close(decoded, one_shot[:, -1:], rtol=2e-5, atol=2e-6)
        state = self.model._layer_states[27]
        self.assertEqual(tuple(state.crsa_log_usage.shape), (1, 4, 4))
        self.assertEqual(
            [(row.query_start, row.history_length_after) for row in self.observed],
            [(0, 3), (3, 4)],
        )

        self.model.reset_state()
        tokenwise, _ = self.model.prefill(token_ids, tokenwise=True, reset=False)
        torch.testing.assert_close(tokenwise, one_shot, rtol=2e-5, atol=2e-6)
        self.assertEqual(
            tuple(self.model._layer_states[27].crsa_log_usage.shape),
            (1, 4, 4),
        )
        self.model.reset_state()
        self.assertEqual(self.model.state_bytes, 0)

        self.model.prefill(token_ids[:, :2], reset=False)
        original_mlp = self.model._mlp
        observed_before_failure = tuple(self.observed)

        def fail_after_native(hidden, *, layer):
            if layer == 27:
                raise RuntimeError("native transaction failure")
            return original_mlp(hidden, layer=layer)

        with mock.patch.object(self.model, "_mlp", side_effect=fail_after_native):
            with self.assertRaisesRegex(RuntimeError, "native transaction failure"):
                self.model.decode(token_ids[:, 2:3])
        self.assertTrue(self.model.state_poisoned)
        self.assertEqual(self.model.state_bytes, 0)
        self.assertIsNone(self.model._layer_states[27])
        self.assertEqual(tuple(self.observed), observed_before_failure)
        self.model.reset_state()
        self.assertFalse(self.model.state_poisoned)

        def failing_observer(_row):
            raise RuntimeError("observer failure")

        self.model.native_head_crsa_observer = failing_observer
        with self.assertWarnsRegex(RuntimeWarning, "observer failed after commit"):
            committed, _ = self.model.prefill([[1]], reset=False)
        self.assertEqual(tuple(committed.shape), (1, 1, self.config.dim))
        self.assertEqual(self.model.next_position, 1)
        self.assertGreater(self.model.state_bytes, 0)
        self.assertFalse(self.model.state_poisoned)

    def test_native_intervention_validates_layer_layout_and_excludes_hidden_graft(
        self,
    ) -> None:
        from immer.runtimes.qwen3_8.native_crsa import Qwen38NativeHeadCrsa

        with self.assertRaisesRegex(ValueError, "only for layer 27"):
            Qwen38NativeHeadCrsa(layer=3)
        invalid_config = _native_tiny_config(full_attention_interval=5)
        with self.assertRaisesRegex(ValueError, "full-attention layer"):
            StreamedQwen38(
                invalid_config,
                self.pager,
                native_head_crsa=self.intervention,
                max_seq_len=16,
            )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            StreamedQwen38(
                self.config,
                self.pager,
                graft=Qwen38StableCrsaGraft(mode="crsa", alpha=0.1),
                graft_layer=1,
                native_head_crsa=self.intervention,
                max_seq_len=16,
            )
        with self.assertRaisesRegex(ValueError, "requires native_head_crsa"):
            StreamedQwen38(
                self.config,
                self.pager,
                native_head_crsa_observer=self.observed.append,
                max_seq_len=16,
            )


class LayerTransitionCrystalModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = _official_topology_tiny_config()
        save_file(_tiny_weights(self.config), self.root / "model.safetensors")
        self.source = Streamer.from_local(self.root, budget_mb=20, use_cache=False)
        self.pager = Qwen38WeightPager(
            self.source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=2 * 1024**2,
        )
        self.model = StreamedQwen38(
            self.config,
            self.pager,
            max_batch_size=1,
            max_seq_len=16,
        )

    def tearDown(self) -> None:
        self.pager.q4_bank = None
        self.pager.close()
        self.source.close()
        self.temporary.cleanup()

    def _identity(self, q4_sha256: str) -> LayerTransitionCrystalIdentity:
        return LayerTransitionCrystalIdentity(
            model_sha256=self.model.layer_transition_crystal_model_sha256(),
            q4_sha256=q4_sha256,
            graph_revision_sha256="3" * 64,
            atlas_revision_sha256="4" * 64,
            projection=LayerTransitionProjectionIdentity(
                hidden_dim=self.config.dim,
                sketch_dim=3,
                seed_sha256="5" * 64,
            ),
        )

    def test_attach_binds_model_q4_and_physical_inventory(self) -> None:
        entries = {
            name: {"payload_bytes": 1000 + index}
            for index, name in enumerate(self.model._layer_transition_q4_names())
        }
        q4_identity = {"manifest_sha256": "6" * 64, "schema": "fixture-q4/v1"}
        self.pager.q4_bank = SimpleNamespace(
            entries=entries,
            has=lambda name: name in entries,
            identity=q4_identity,
        )
        identity = self._identity(self.model.layer_transition_crystal_q4_sha256())
        bank = LayerTransitionCrystalBank(
            self.root / "layer-transition.json",
            identity,
        )

        self.model.attach_layer_transition_crystal_bank(
            bank,
            max_error_radius=0.25,
            graph_revision_sha256=identity.graph_revision_sha256,
            atlas_revision_sha256=identity.atlas_revision_sha256,
        )

        self.assertIs(self.model.layer_transition_crystal_bank, bank)
        self.assertTrue(self.model.layer_transition_crystal_enabled)
        self.assertEqual(self.model.layer_transition_crystal_max_error_radius, 0.25)
        with self.assertRaisesRegex(Qwen38RuntimeError, "model pin"):
            other = LayerTransitionCrystalBank(
                self.root / "other-layer-transition.json",
                LayerTransitionCrystalIdentity(
                    model_sha256="7" * 64,
                    q4_sha256=identity.q4_sha256,
                    graph_revision_sha256=identity.graph_revision_sha256,
                    atlas_revision_sha256=identity.atlas_revision_sha256,
                    projection=identity.projection,
                ),
            )
            self.model.attach_layer_transition_crystal_bank(
                other,
                graph_revision_sha256=other.identity.graph_revision_sha256,
                atlas_revision_sha256=other.identity.atlas_revision_sha256,
            )
        with self.assertRaisesRegex(Qwen38RuntimeError, "graph authority"):
            self.model.attach_layer_transition_crystal_bank(
                bank,
                graph_revision_sha256="8" * 64,
                atlas_revision_sha256=identity.atlas_revision_sha256,
            )
        self.model.attach_layer_transition_crystal_bank(None)

    def test_k1_hit_skips_q_o_and_mlp_but_appends_exact_live_kv(self) -> None:
        layer = self.config.n_layers - 1
        hidden = torch.randn((1, 1, self.config.dim)).to(torch.bfloat16)
        mask = torch.ones((1, 1), dtype=torch.bool)
        expected_hidden, expected_state = self.model._forward_layer(
            hidden,
            layer=layer,
            token_mask=mask,
            state=None,
            start_pos=0,
            stateful=True,
        )
        self.assertIsInstance(expected_state, AttentionState)

        avoided_bytes = 123_456
        replacement = SimpleNamespace(
            output=expected_hidden.clone(),
            logical_weight_bytes_replaced=avoided_bytes,
        )
        bank = SimpleNamespace(
            identity=SimpleNamespace(
                atlas_revision_sha256="a" * 64,
                graph_revision_sha256="b" * 64,
                identity_sha256="c" * 64,
                model_sha256="d" * 64,
                q4_sha256="e" * 64,
            ),
            metrics=lambda: SimpleNamespace(
                to_dict=lambda: {"attempts": 1, "fallbacks": 0, "replacements": 1}
            ),
            replace=mock.Mock(return_value=replacement),
        )
        self.model.layer_transition_crystal_bank = bank
        self.model.layer_transition_crystal_enabled = True
        self.model.layer_transition_crystal_max_error_radius = 0.0
        mlp_bank = self._mlp_crystal_bank(None)
        self.model.layer_mlp_crystal_bank = mlp_bank
        self.model.layer_mlp_crystal_enabled = True

        original_linear_group = self.pager.linear_group
        projected_names: list[tuple[str, ...]] = []

        def record_linear_group(value, names):
            projected_names.append(tuple(names))
            return original_linear_group(value, names)

        with (
            mock.patch.object(
                self.model,
                "_layer_transition_avoided_q4_bytes",
                return_value=avoided_bytes,
            ),
            mock.patch.object(
                self.pager,
                "linear_group",
                side_effect=record_linear_group,
            ),
            mock.patch.object(
                self.model,
                "_full_attention",
                side_effect=AssertionError("full attention executed"),
            ),
            mock.patch.object(
                self.model,
                "_mlp",
                side_effect=AssertionError("MLP executed"),
            ),
        ):
            actual_hidden, actual_state = self.model._forward_layer(
                hidden,
                layer=layer,
                token_mask=mask,
                state=None,
                start_pos=0,
                stateful=True,
            )

        self.assertTrue(torch.equal(actual_hidden, expected_hidden))
        self.assertIsInstance(actual_state, AttentionState)
        assert isinstance(actual_state, AttentionState)
        assert isinstance(expected_state, AttentionState)
        self.assertTrue(torch.equal(actual_state.key, expected_state.key))
        self.assertTrue(torch.equal(actual_state.value, expected_state.value))
        self.assertEqual(
            projected_names,
            [
                (
                    f"model.language_model.layers.{layer}.self_attn.k_proj",
                    f"model.language_model.layers.{layer}.self_attn.v_proj",
                )
            ],
        )
        bank.replace.assert_called_once_with(hidden, max_error_radius=0.0)
        mlp_bank.replace.assert_not_called()
        metrics = self.model.layer_transition_crystal_metrics()
        self.assertEqual(metrics["physical_transitions"], 1)
        self.assertEqual(metrics["transition_rows"], 1)
        self.assertEqual(metrics["exact_kv_state_updates"], 1)
        self.assertEqual(metrics["skipped_q4_matrix_calls"], 5)
        self.assertEqual(metrics["packed_weight_bytes_avoided"], avoided_bytes)

    def test_miss_falls_through_without_partial_state_or_savings(self) -> None:
        layer = self.config.n_layers - 1
        hidden = torch.randn((1, 1, self.config.dim)).to(torch.bfloat16)
        mask = torch.ones((1, 1), dtype=torch.bool)
        bank = SimpleNamespace(replace=mock.Mock(return_value=None))
        self.model.layer_transition_crystal_bank = bank
        self.model.layer_transition_crystal_enabled = True

        actual_hidden, actual_state = self.model._forward_layer(
            hidden,
            layer=layer,
            token_mask=mask,
            state=None,
            start_pos=0,
            stateful=True,
        )

        self.assertEqual(tuple(actual_hidden.shape), (1, 1, self.config.dim))
        self.assertIsInstance(actual_state, AttentionState)
        bank.replace.assert_called_once()
        self.assertEqual(self.model._layer_transition_crystal_hits, 0)
        self.assertEqual(
            self.model._layer_transition_crystal_packed_weight_bytes_avoided,
            0,
        )

    def test_k4_wave_skips_one_weight_wave_and_retains_every_exact_kv_prefix(
        self,
    ) -> None:
        layer = self.config.n_layers - 1
        hidden = tuple(
            torch.randn((1, 1, self.config.dim)).to(torch.bfloat16) for _ in range(4)
        )
        expected_hidden, expected_state, expected_trace = (
            self.model._forward_layer_token_rows(
                hidden,
                layer=layer,
                state=None,
                start_pos=0,
                native_head_crsa_observer=lambda _row: None,
            )
        )
        self.assertIsInstance(expected_state, AttentionState)
        self.assertEqual(expected_trace.attention_log_usage, (None,) * 4)
        avoided_bytes = 123_456
        replacements = tuple(
            SimpleNamespace(
                output=row.clone(),
                logical_weight_bytes_replaced=avoided_bytes,
            )
            for row in expected_hidden
        )
        bank = SimpleNamespace(
            identity=SimpleNamespace(
                atlas_revision_sha256="a" * 64,
                graph_revision_sha256="b" * 64,
                identity_sha256="c" * 64,
                model_sha256="d" * 64,
                q4_sha256="e" * 64,
            ),
            metrics=lambda: SimpleNamespace(
                to_dict=lambda: {"attempts": 4, "fallbacks": 0, "replacements": 4}
            ),
            replace_many=mock.Mock(return_value=replacements),
        )
        self.model.layer_transition_crystal_bank = bank
        self.model.layer_transition_crystal_enabled = True
        self.model.layer_transition_crystal_max_error_radius = 0.0
        projected_names: list[tuple[str, ...]] = []
        original_group = self.model._linear_group_token_rows

        def record_group(rows, names):
            projected_names.append(tuple(names))
            return original_group(rows, names)

        with (
            mock.patch.object(
                self.model,
                "_layer_transition_avoided_q4_bytes",
                return_value=avoided_bytes,
            ),
            mock.patch.object(
                self.model,
                "_linear_group_token_rows",
                side_effect=record_group,
            ),
            mock.patch.object(
                self.model,
                "_full_attention_token_rows",
                side_effect=AssertionError("full attention executed"),
            ),
            mock.patch.object(
                self.model,
                "_mlp_token_rows",
                side_effect=AssertionError("MLP executed"),
            ),
        ):
            actual_hidden, actual_state, actual_trace = (
                self.model._forward_layer_token_rows(
                    hidden,
                    layer=layer,
                    state=None,
                    start_pos=0,
                    native_head_crsa_observer=lambda _row: None,
                )
            )

        self.assertTrue(
            all(
                torch.equal(actual, expected)
                for actual, expected in zip(
                    actual_hidden,
                    expected_hidden,
                    strict=True,
                )
            )
        )
        self.assertIsInstance(actual_state, AttentionState)
        assert isinstance(actual_state, AttentionState)
        assert isinstance(expected_state, AttentionState)
        self.assertTrue(torch.equal(actual_state.key, expected_state.key))
        self.assertTrue(torch.equal(actual_state.value, expected_state.value))
        self.assertEqual(actual_trace.attention_log_usage, (None,) * 4)
        self.assertTrue(
            torch.equal(actual_state.key[:, :, :2], expected_state.key[:, :, :2])
        )
        self.assertEqual(
            projected_names,
            [
                (
                    f"model.language_model.layers.{layer}.self_attn.k_proj",
                    f"model.language_model.layers.{layer}.self_attn.v_proj",
                )
            ],
        )
        bank.replace_many.assert_called_once_with(hidden, max_error_radius=0.0)
        metrics = self.model.layer_transition_crystal_metrics()
        self.assertEqual(metrics["physical_transitions"], 1)
        self.assertEqual(metrics["transition_rows"], 4)
        self.assertEqual(metrics["exact_kv_state_updates"], 4)
        self.assertEqual(metrics["skipped_q4_matrix_calls"], 5)
        self.assertEqual(metrics["packed_weight_bytes_avoided"], avoided_bytes)

    @staticmethod
    def _mlp_crystal_bank(replacement):
        return SimpleNamespace(
            identity=SimpleNamespace(
                atlas_revision_sha256="a" * 64,
                graph_revision_sha256="b" * 64,
                identity_sha256="c" * 64,
                model_sha256="d" * 64,
                q4_sha256="e" * 64,
            ),
            metrics=lambda: SimpleNamespace(
                to_dict=lambda: {"attempts": 1, "fallbacks": 0, "replacements": 1}
            ),
            replace=mock.Mock(return_value=replacement),
        )

    def test_layer_mlp_k1_hit_keeps_exact_attention_and_kv_but_skips_mlp(
        self,
    ) -> None:
        layer = self.config.n_layers - 1
        hidden = torch.randn((1, 1, self.config.dim)).to(torch.bfloat16)
        mask = torch.ones((1, 1), dtype=torch.bool)
        _, expected_state = self.model._forward_layer(
            hidden,
            layer=layer,
            token_mask=mask,
            state=None,
            start_pos=0,
            stateful=True,
        )
        self.assertIsInstance(expected_state, AttentionState)
        output = torch.randn((1, 1, self.config.dim)).to(torch.bfloat16)
        replacement = SimpleNamespace(
            output=output,
            logical_weight_bytes_replaced=(self.model.LAYER63_MLP_Q4_WEIGHT_BYTES),
        )
        bank = self._mlp_crystal_bank(replacement)
        self.model.layer_mlp_crystal_bank = bank
        self.model.layer_mlp_crystal_enabled = True
        self.model.layer_mlp_crystal_max_error_radius = 0.125
        self.model._active_mlp_page_coordinate_transaction = SimpleNamespace()
        original_group = self.pager.linear_group
        original_linear = self.pager.linear
        grouped_names: list[tuple[str, ...]] = []
        projected_names: list[str] = []

        def record_group(value, names, **kwargs):
            grouped_names.append(tuple(names))
            return original_group(value, names, **kwargs)

        def record_linear(value, name, **kwargs):
            projected_names.append(name)
            return original_linear(value, name, **kwargs)

        with (
            mock.patch.object(
                self.model,
                "_layer_mlp_crystal_avoided_q4_bytes",
                return_value=self.model.LAYER63_MLP_Q4_WEIGHT_BYTES,
            ),
            mock.patch.object(
                self.pager,
                "linear_group",
                side_effect=record_group,
            ),
            mock.patch.object(
                self.pager,
                "linear",
                side_effect=record_linear,
            ),
            mock.patch.object(
                self.model,
                "_mlp",
                side_effect=AssertionError("MLP executed on a Crystal hit"),
            ),
        ):
            actual_hidden, actual_state = self.model._forward_layer(
                hidden,
                layer=layer,
                token_mask=mask,
                state=None,
                start_pos=0,
                stateful=True,
            )

        self.assertTrue(torch.equal(actual_hidden, output))
        self.assertIsInstance(actual_state, AttentionState)
        assert isinstance(actual_state, AttentionState)
        assert isinstance(expected_state, AttentionState)
        self.assertTrue(torch.equal(actual_state.key, expected_state.key))
        self.assertTrue(torch.equal(actual_state.value, expected_state.value))
        base = f"model.language_model.layers.{layer}.self_attn"
        self.assertIn(
            (f"{base}.q_proj", f"{base}.k_proj", f"{base}.v_proj"),
            grouped_names,
        )
        for name in (
            f"{base}.q_proj.weight",
            f"{base}.k_proj.weight",
            f"{base}.v_proj.weight",
            f"{base}.o_proj",
        ):
            self.assertIn(name, projected_names)
        bank.replace.assert_called_once()
        call = bank.replace.call_args
        self.assertEqual(len(call.args), 2)
        self.assertEqual(tuple(call.args[0].shape), (1, 1, self.config.dim))
        self.assertEqual(tuple(call.args[1].shape), (1, 1, self.config.dim))
        self.assertEqual(call.args[0].dtype, torch.bfloat16)
        self.assertEqual(call.args[1].dtype, torch.bfloat16)
        self.assertEqual(call.kwargs, {"max_error_radius": 0.125})
        metrics = self.model.layer_mlp_crystal_metrics()
        self.assertEqual(metrics["physical_transitions"], 1)
        self.assertEqual(metrics["transition_rows"], 1)
        self.assertEqual(metrics["exact_kv_state_updates"], 0)
        self.assertEqual(metrics["skipped_q4_matrix_calls"], 3)
        self.assertEqual(
            metrics["packed_weight_bytes_avoided"],
            self.model.LAYER63_MLP_Q4_WEIGHT_BYTES,
        )

    def test_layer_mlp_miss_runs_original_mlp_once(self) -> None:
        layer = self.config.n_layers - 1
        hidden = torch.randn((1, 1, self.config.dim)).to(torch.bfloat16)
        mask = torch.ones((1, 1), dtype=torch.bool)
        bank = self._mlp_crystal_bank(None)
        self.model.layer_mlp_crystal_bank = bank
        self.model.layer_mlp_crystal_enabled = True

        with mock.patch.object(
            self.model,
            "_mlp",
            wraps=self.model._mlp,
        ) as exact_mlp:
            output, state = self.model._forward_layer(
                hidden,
                layer=layer,
                token_mask=mask,
                state=None,
                start_pos=0,
                stateful=True,
            )

        self.assertEqual(tuple(output.shape), (1, 1, self.config.dim))
        self.assertIsInstance(state, AttentionState)
        bank.replace.assert_called_once()
        exact_mlp.assert_called_once()
        self.assertEqual(self.model._layer_mlp_crystal_hits, 0)
        self.assertEqual(
            self.model._layer_mlp_crystal_packed_weight_bytes_avoided,
            0,
        )

    def test_layer_mlp_observer_forces_exact_mlp_without_bank_attempt(self) -> None:
        layer = self.config.n_layers - 1
        hidden = torch.randn((1, 1, self.config.dim)).to(torch.bfloat16)
        mask = torch.ones((1, 1), dtype=torch.bool)
        replacement = SimpleNamespace(
            output=hidden.clone(),
            logical_weight_bytes_replaced=(self.model.LAYER63_MLP_Q4_WEIGHT_BYTES),
        )
        bank = self._mlp_crystal_bank(replacement)
        self.model.layer_mlp_crystal_bank = bank
        self.model.layer_mlp_crystal_enabled = True
        self.model.layer_boundary_observer = lambda *_args: None
        self.model.layer_boundary_stages = LAYER_BOUNDARY_STAGES

        with mock.patch.object(
            self.model,
            "_mlp",
            wraps=self.model._mlp,
        ) as exact_mlp:
            self.model._forward_layer(
                hidden,
                layer=layer,
                token_mask=mask,
                state=None,
                start_pos=0,
                stateful=True,
            )

        bank.replace.assert_not_called()
        exact_mlp.assert_called_once()

    def test_layer_mlp_inventory_is_gate_up_down_and_exact_deployed_bytes(
        self,
    ) -> None:
        names = self.model._layer_mlp_crystal_q4_names()
        self.assertEqual(len(names), 3)
        self.assertEqual(
            tuple(name.rsplit(".", 2)[-2] for name in names),
            ("gate_proj", "up_proj", "down_proj"),
        )
        sizes = (
            50_000_000,
            50_000_000,
            self.model.LAYER63_MLP_Q4_WEIGHT_BYTES - 100_000_000,
        )
        entries = {
            name: {"payload_bytes": size}
            for name, size in zip(names, sizes, strict=True)
        }
        self.pager.q4_bank = SimpleNamespace(
            entries=entries,
            has=lambda name: name in entries,
            identity={"schema": "fixture-q4/v1"},
        )
        self.assertEqual(
            self.model._layer_mlp_crystal_avoided_q4_bytes(),
            self.model.LAYER63_MLP_Q4_WEIGHT_BYTES,
        )
        invalid = SimpleNamespace(
            output=torch.zeros((1, 1, self.config.dim), dtype=torch.bfloat16),
            logical_weight_bytes_replaced=(self.model.LAYER63_MLP_Q4_WEIGHT_BYTES - 1),
        )
        bank = self._mlp_crystal_bank(invalid)
        self.model.layer_mlp_crystal_bank = bank
        self.model.layer_mlp_crystal_enabled = True
        row = torch.zeros((1, 1, self.config.dim), dtype=torch.bfloat16)
        with self.assertRaisesRegex(Qwen38RuntimeError, "byte claim changed"):
            self.model._layer_mlp_crystal_forward(
                row,
                row,
                layer=self.config.n_layers - 1,
                token_mask=torch.ones((1, 1), dtype=torch.bool),
                stateful=True,
            )
        self.assertEqual(self.model._layer_mlp_crystal_hits, 0)
        self.assertEqual(
            self.model._layer_mlp_crystal_packed_weight_bytes_avoided,
            0,
        )

    def test_layer_mlp_attach_binds_model_q4_authorities_and_scope(self) -> None:
        names = self.model._layer_mlp_crystal_q4_names()
        entries = {
            name: {"payload_bytes": 1_000 + index} for index, name in enumerate(names)
        }
        self.pager.q4_bank = SimpleNamespace(
            entries=entries,
            has=lambda name: name in entries,
            identity={"manifest_sha256": "6" * 64, "schema": "fixture-q4/v1"},
        )
        identity = Layer63MlpResidualCrystalIdentity(
            model_sha256=self.model.layer_mlp_crystal_model_sha256(),
            q4_sha256=self.model.layer_mlp_crystal_q4_sha256(),
            graph_revision_sha256="7" * 64,
            atlas_revision_sha256="8" * 64,
            projection=LayerTransitionProjectionIdentity(
                hidden_dim=self.config.dim,
                sketch_dim=3,
                seed_sha256="9" * 64,
            ),
        )
        bank = Layer63MlpResidualCrystalBank(
            self.root / "layer-mlp-crystal.json",
            identity,
        )

        with self.assertRaisesRegex(Qwen38RuntimeError, "contains no actions"):
            self.model.attach_layer_mlp_crystal_bank(
                bank,
                max_error_radius=0.25,
                graph_revision_sha256=identity.graph_revision_sha256,
                atlas_revision_sha256=identity.atlas_revision_sha256,
            )
        expected_bytes = sum(row["payload_bytes"] for row in entries.values())
        with mock.patch.object(
            Layer63MlpResidualCrystalBank,
            "crystals",
            new_callable=mock.PropertyMock,
            return_value=(
                SimpleNamespace(logical_weight_bytes_replaced=expected_bytes),
            ),
        ):
            self.model.attach_layer_mlp_crystal_bank(
                bank,
                max_error_radius=0.25,
                graph_revision_sha256=identity.graph_revision_sha256,
                atlas_revision_sha256=identity.atlas_revision_sha256,
            )

        self.assertIs(self.model.layer_mlp_crystal_bank, bank)
        self.assertTrue(self.model.layer_mlp_crystal_enabled)
        self.assertEqual(self.model.layer_mlp_crystal_max_error_radius, 0.25)
        with self.assertRaisesRegex(Qwen38RuntimeError, "graph authority"):
            self.model.attach_layer_mlp_crystal_bank(
                bank,
                graph_revision_sha256="a" * 64,
                atlas_revision_sha256=identity.atlas_revision_sha256,
            )
        self.model.detach_layer_mlp_crystal_bank()
        self.assertIsNone(self.model.layer_mlp_crystal_bank)
        self.assertFalse(self.model.layer_mlp_crystal_enabled)
        with self.assertRaisesRegex(Qwen38RuntimeError, "requires an attached bank"):
            self.model.set_layer_mlp_crystal_enabled(True)

    def test_layer_mlp_crystal_never_runs_for_k2_or_k4(self) -> None:
        layer = self.config.n_layers - 1
        bank = self._mlp_crystal_bank(
            SimpleNamespace(
                output=torch.zeros((1, 1, self.config.dim), dtype=torch.bfloat16),
                logical_weight_bytes_replaced=(self.model.LAYER63_MLP_Q4_WEIGHT_BYTES),
            )
        )
        self.model.layer_mlp_crystal_bank = bank
        self.model.layer_mlp_crystal_enabled = True
        for width in (2, 4):
            with self.subTest(width=width):
                rows = tuple(
                    torch.randn((1, 1, self.config.dim)).to(torch.bfloat16)
                    for _ in range(width)
                )
                self.model._forward_layer_token_rows(
                    rows,
                    layer=layer,
                    state=None,
                    start_pos=0,
                    native_head_crsa_observer=lambda _row: None,
                )

        bank.replace.assert_not_called()

    def test_passive_layer_mlp_o1_hook_is_exact_k1_decode_only_and_isolated(
        self,
    ) -> None:
        config = _official_topology_tiny_config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_file(_tiny_weights(config), root / "model.safetensors")
            source = Streamer.from_local(root, budget_mb=20, use_cache=False)
            pager = Qwen38WeightPager(
                source,
                device="cpu",
                compute_dtype="bfloat16",
                max_resident_bytes=2 * 1024**2,
            )
            model = StreamedQwen38(config, pager, max_batch_size=1, max_seq_len=8)
            try:
                hidden = torch.randn((1, 1, config.dim)).to(torch.bfloat16)
                mask = torch.ones((1, 1), dtype=torch.bool)
                expected, _ = model._forward_layer(
                    hidden,
                    layer=63,
                    token_mask=mask,
                    state=None,
                    start_pos=1,
                    stateful=True,
                )
                captured: list[tuple[torch.Tensor, ...]] = []

                def observe(*rows: torch.Tensor) -> None:
                    captured.append(rows)
                    for row in rows:
                        row.zero_()  # caller-owned copies cannot alter model output

                model.set_layer_mlp_o1_observer(observe)
                actual, _ = model._forward_layer(
                    hidden,
                    layer=63,
                    token_mask=mask,
                    state=None,
                    start_pos=1,
                    stateful=True,
                )
                self.assertTrue(torch.equal(actual, expected))
                self.assertEqual(len(captured), 1)
                self.assertTrue(
                    all(
                        tuple(row.shape) == (1, 1, config.dim)
                        and row.dtype == torch.bfloat16
                        and row.device.type == "cpu"
                        for row in captured[0]
                    )
                )

                # Stateful prompt position zero and non-stateful prefill are not
                # decode observations even when their physical width is one.
                model._forward_layer(
                    hidden,
                    layer=63,
                    token_mask=mask,
                    state=None,
                    start_pos=0,
                    stateful=True,
                )
                model._forward_layer(
                    hidden,
                    layer=63,
                    token_mask=mask,
                    state=None,
                    start_pos=0,
                    stateful=False,
                )
                model._forward_layer_token_rows(
                    (hidden, hidden.clone()),
                    layer=63,
                    state=None,
                    start_pos=1,
                    native_head_crsa_observer=lambda _row: None,
                )
                self.assertEqual(len(captured), 1)

                def broken(*_rows: torch.Tensor) -> None:
                    raise RuntimeError("passive sink failed")

                model.set_layer_mlp_o1_observer(broken)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    isolated, _ = model._forward_layer(
                        hidden,
                        layer=63,
                        token_mask=mask,
                        state=None,
                        start_pos=1,
                        stateful=True,
                    )
                self.assertTrue(torch.equal(isolated, expected))
                self.assertEqual(model.layer_mlp_o1_observer_metrics()["failures"], 1)

                model.set_layer_mlp_o1_observer(observe)
                replacement = SimpleNamespace(
                    output=expected.clone(),
                    logical_weight_bytes_replaced=123,
                )
                bank = self._mlp_crystal_bank(replacement)
                model.layer_mlp_crystal_bank = bank
                model.layer_mlp_crystal_enabled = True
                before = len(captured)
                with mock.patch.object(
                    model,
                    "_layer_mlp_crystal_avoided_q4_bytes",
                    return_value=123,
                ):
                    model._forward_layer(
                        hidden,
                        layer=63,
                        token_mask=mask,
                        state=None,
                        start_pos=1,
                        stateful=True,
                    )
                self.assertEqual(len(captured), before)
            finally:
                pager.close()
                source.close()


if __name__ == "__main__":
    unittest.main()
