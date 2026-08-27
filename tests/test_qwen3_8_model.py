from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import torch
from safetensors.torch import save_file

import immer.runtimes.qwen3_8.model as qwen_model_module
from immer.knowledge.streamer import Streamer
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
        with self.assertRaisesRegex(Qwen38RuntimeError, "K in \\[1, 4\\]"):
            self.model.stage_continuation_block([[9, 7, 6, 5, 4]])

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

        full = self.model.stage_continuation_block([[9, 7, 6, 5]])
        with self.assertRaisesRegex(Qwen38RuntimeError, "exceeds K=4"):
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
        observer = NativePrefixSinkhornOperatorObserver(
            blocks.append, max_positions=4
        )
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
                    :, intervention.head_indices, position : position + 1, : position + 1
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
        self.assertEqual([row.query_positions for row in bounded_rows], [(0,), (1,), (2,)])
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


if __name__ == "__main__":
    unittest.main()
