from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from safetensors.torch import save_file
import torch

from immer.knowledge.streamer import Streamer
from immer.runtimes.qwen3_8.config import Qwen38Config
from immer.runtimes.qwen3_8.kernels import AttentionState, DeltaNetState
from immer.runtimes.qwen3_8.model import StreamedQwen38
from immer.runtimes.qwen3_8.native_crsa import Qwen38NativeHeadCrsa
from immer.runtimes.qwen3_8.native_fork import (
    COMMON_STOP_LAYER,
    FORK_LAYER,
    Qwen38NativeFork,
)
from immer.runtimes.qwen3_8.pager import Qwen38WeightPager


def _native_config() -> Qwen38Config:
    depth = 28
    interval = 4
    return Qwen38Config.from_mapping(
        {
            "model_type": "qwen3_5_text",
            "vocab_size": 32,
            "hidden_size": 12,
            "intermediate_size": 16,
            "num_hidden_layers": depth,
            "layer_types": [
                (
                    "full_attention"
                    if (layer + 1) % interval == 0
                    else "linear_attention"
                )
                for layer in range(depth)
            ],
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 6,
            "partial_rotary_factor": 1.0,
            "full_attention_interval": interval,
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
        },
        require_official=False,
    )


def _matrix(rows: int, columns: int, generator: torch.Generator) -> torch.Tensor:
    return (0.04 * torch.randn(rows, columns, generator=generator)).to(torch.bfloat16)


def _weights(config: Qwen38Config) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(331)
    tensors: dict[str, torch.Tensor] = {
        "model.language_model.embed_tokens.weight": _matrix(
            config.vocab_size, config.dim, generator
        ),
        "model.language_model.norm.weight": torch.zeros(
            config.dim, dtype=torch.bfloat16
        ),
        "lm_head.weight": _matrix(config.vocab_size, config.dim, generator),
    }
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
            attention = f"{base}.self_attn"
            tensors[f"{attention}.q_proj.weight"] = _matrix(
                2 * config.n_heads * config.head_dim,
                config.dim,
                generator,
            )
            tensors[f"{attention}.k_proj.weight"] = _matrix(
                config.n_kv_heads * config.head_dim,
                config.dim,
                generator,
            )
            tensors[f"{attention}.v_proj.weight"] = _matrix(
                config.n_kv_heads * config.head_dim,
                config.dim,
                generator,
            )
            tensors[f"{attention}.o_proj.weight"] = _matrix(
                config.dim,
                config.n_heads * config.head_dim,
                generator,
            )
            tensors[f"{attention}.q_norm.weight"] = torch.zeros(
                config.head_dim, dtype=torch.bfloat16
            )
            tensors[f"{attention}.k_norm.weight"] = torch.zeros(
                config.head_dim, dtype=torch.bfloat16
            )
            continue
        attention = f"{base}.linear_attn"
        key_features = config.linear_num_key_heads * config.linear_key_head_dim
        value_features = config.linear_num_value_heads * config.linear_value_head_dim
        conv_features = 2 * key_features + value_features
        tensors[f"{attention}.in_proj_qkv.weight"] = _matrix(
            conv_features, config.dim, generator
        )
        tensors[f"{attention}.in_proj_z.weight"] = _matrix(
            value_features, config.dim, generator
        )
        tensors[f"{attention}.in_proj_a.weight"] = _matrix(
            config.linear_num_value_heads, config.dim, generator
        )
        tensors[f"{attention}.in_proj_b.weight"] = _matrix(
            config.linear_num_value_heads, config.dim, generator
        )
        tensors[f"{attention}.conv1d.weight"] = (
            0.08
            * torch.randn(
                conv_features,
                1,
                config.linear_conv_kernel_dim,
                generator=generator,
            )
        ).to(torch.bfloat16)
        tensors[f"{attention}.A_log"] = torch.tensor([-0.7, 0.1], dtype=torch.bfloat16)
        tensors[f"{attention}.dt_bias"] = torch.tensor(
            [-0.2, 0.3], dtype=torch.bfloat16
        )
        tensors[f"{attention}.norm.weight"] = torch.ones(
            config.linear_value_head_dim, dtype=torch.bfloat16
        )
        tensors[f"{attention}.out_proj.weight"] = _matrix(
            config.dim, value_features, generator
        )
    return tensors


def _assert_states_equal(
    case: unittest.TestCase,
    actual: tuple[AttentionState | DeltaNetState, ...]
    | list[AttentionState | DeltaNetState | None],
    expected: tuple[AttentionState | DeltaNetState, ...]
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


def _clone_state_tensors(
    states: tuple[AttentionState | DeltaNetState, ...],
) -> tuple[tuple[torch.Tensor, ...], ...]:
    rows: list[tuple[torch.Tensor, ...]] = []
    for state in states:
        if isinstance(state, AttentionState):
            tensors = (state.key.clone(), state.value.clone())
            if state.crsa_log_usage is not None:
                tensors = (*tensors, state.crsa_log_usage.clone())
            rows.append(tensors)
        else:
            rows.append((state.conv.clone(), state.recurrent.clone()))
    return tuple(rows)


def _assert_tensor_snapshots_equal(
    case: unittest.TestCase,
    actual: tuple[tuple[torch.Tensor, ...], ...],
    expected: tuple[tuple[torch.Tensor, ...], ...],
) -> None:
    case.assertEqual(len(actual), len(expected))
    for actual_row, expected_row in zip(actual, expected, strict=True):
        case.assertEqual(len(actual_row), len(expected_row))
        for actual_tensor, expected_tensor in zip(
            actual_row, expected_row, strict=True
        ):
            case.assertTrue(torch.equal(actual_tensor, expected_tensor))


class Qwen38NativeForkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = _native_config()
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        save_file(_weights(cls.config), str(cls.root / "model.safetensors"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def _pager(self) -> Qwen38WeightPager:
        source = Streamer.from_local(
            self.root,
            budget_mb=128,
            revision="native-fork-fixture",
        )
        return Qwen38WeightPager(
            source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=4 * 1024 * 1024,
        )

    def _runtime(
        self,
    ) -> tuple[Qwen38NativeFork, Qwen38NativeHeadCrsa, Qwen38WeightPager]:
        pager = self._pager()
        intervention = Qwen38NativeHeadCrsa(alpha=0.1)
        return (
            Qwen38NativeFork(
                self.config,
                pager,
                native_head_crsa=intervention,
                max_seq_len=32,
            ),
            intervention,
            pager,
        )

    @staticmethod
    def _fork_states(result, arm: str):
        selected = result.state.off if arm == "off" else result.state.native
        return (
            result.state.common_layer_states + selected.suffix_layer_states
            if result.state.joined
            else selected.layer_states
        )

    def test_prefill_and_joined_decode_match_independent_arms_bit_exact(self) -> None:
        fork, intervention, _ = self._runtime()
        off = StreamedQwen38(
            self.config, self._pager(), max_batch_size=1, max_seq_len=32
        )
        native = StreamedQwen38(
            self.config,
            self._pager(),
            native_head_crsa=intervention,
            max_batch_size=1,
            max_seq_len=32,
        )
        prompt = [[1, 4, 5]]
        actual = fork.prefill(prompt)
        off_hidden, _ = off.hidden_stateful(prompt, start_pos=0)
        native_hidden, _ = native.hidden_stateful(prompt, start_pos=0)
        self.assertTrue(torch.equal(actual.off_hidden, off_hidden[:, -1:]))
        self.assertTrue(torch.equal(actual.native_hidden, native_hidden[:, -1:]))
        _assert_states_equal(self, self._fork_states(actual, "off"), off._layer_states)
        _assert_states_equal(
            self, self._fork_states(actual, "native"), native._layer_states
        )

        old_off = _clone_state_tensors(self._fork_states(actual, "off"))
        old_native = _clone_state_tensors(self._fork_states(actual, "native"))
        decoded = fork.decode_pair(actual.state, 7, 7)
        off_hidden, _ = off.decode([[7]])
        native_hidden, _ = native.decode([[7]])
        self.assertTrue(decoded.state.joined)
        self.assertTrue(torch.equal(decoded.off_hidden, off_hidden))
        self.assertTrue(torch.equal(decoded.native_hidden, native_hidden))
        _assert_states_equal(self, self._fork_states(decoded, "off"), off._layer_states)
        _assert_states_equal(
            self, self._fork_states(decoded, "native"), native._layer_states
        )
        for old_rows, states in (
            (old_off, self._fork_states(actual, "off")),
            (old_native, self._fork_states(actual, "native")),
        ):
            for old, state in zip(old_rows, states, strict=True):
                current = (
                    (state.key, state.value)
                    if isinstance(state, AttentionState)
                    else (state.conv, state.recurrent)
                )
                for expected, tensor in zip(old, current, strict=False):
                    self.assertTrue(torch.equal(expected, tensor))

    def test_forced_divergence_is_permanent_and_matches_independent_arms(self) -> None:
        fork, intervention, _ = self._runtime()
        off = StreamedQwen38(
            self.config, self._pager(), max_batch_size=1, max_seq_len=32
        )
        native = StreamedQwen38(
            self.config,
            self._pager(),
            native_head_crsa=intervention,
            max_batch_size=1,
            max_seq_len=32,
        )
        prompt = [[1, 6, 3]]
        actual = fork.prefill(prompt)
        off.hidden_stateful(prompt, start_pos=0)
        native.hidden_stateful(prompt, start_pos=0)

        actual = fork.decode_pair(actual.state, 7, 8)
        off_hidden, _ = off.decode([[7]])
        native_hidden, _ = native.decode([[8]])
        self.assertFalse(actual.state.joined)
        self.assertEqual(actual.state.first_divergence_position, 3)
        self.assertEqual(len(actual.state.off.lower_layer_states), 27)
        self.assertEqual(len(actual.state.native.lower_layer_states), 27)
        self.assertTrue(torch.equal(actual.off_hidden, off_hidden))
        self.assertTrue(torch.equal(actual.native_hidden, native_hidden))

        actual = fork.decode_pair(actual.state, 9, 9)
        off_hidden, _ = off.decode([[9]])
        native_hidden, _ = native.decode([[9]])
        self.assertFalse(actual.state.joined)
        self.assertEqual(actual.state.first_divergence_position, 3)
        self.assertEqual(actual.traffic.shared_layers, 0)
        self.assertEqual(actual.traffic.off_layers, self.config.n_layers)
        self.assertEqual(actual.traffic.native_layers, self.config.n_layers)
        self.assertTrue(torch.equal(actual.off_hidden, off_hidden))
        self.assertTrue(torch.equal(actual.native_hidden, native_hidden))
        _assert_states_equal(self, actual.state.off.layer_states, off._layer_states)
        _assert_states_equal(
            self, actual.state.native.layer_states, native._layer_states
        )

    def test_joined_layer27_shares_kv_and_only_native_arm_owns_usage(self) -> None:
        fork, _, _ = self._runtime()
        prefill = fork.prefill([[1, 4, 9]])
        off = prefill.state.off.suffix_layer_states[0]
        native = prefill.state.native.suffix_layer_states[0]
        self.assertIsInstance(off, AttentionState)
        self.assertIsInstance(native, AttentionState)
        assert isinstance(off, AttentionState)
        assert isinstance(native, AttentionState)
        self.assertIs(off.key, native.key)
        self.assertIs(off.value, native.value)
        self.assertIsNone(off.crsa_log_usage)
        self.assertIsNotNone(native.crsa_log_usage)
        self.assertEqual(len(prefill.traffic.native_head_crsa_evidence), 1)
        evidence = prefill.traffic.native_head_crsa_evidence[0]
        self.assertEqual(evidence.layer, FORK_LAYER)
        self.assertFalse(evidence.identity)

    def test_state_validation_precedes_io_and_pair_failure_rolls_back(self) -> None:
        fork, _, pager = self._runtime()
        prefill = fork.prefill([[1, 2, 3]])
        native_fork = prefill.state.native.suffix_layer_states[0]
        assert isinstance(native_fork, AttentionState)
        detached = AttentionState(
            key=native_fork.key.clone(),
            value=native_fork.value.clone(),
            crsa_log_usage=native_fork.crsa_log_usage,
        )
        malformed_native = replace(
            prefill.state.native,
            suffix_layer_states=(
                detached,
                *prefill.state.native.suffix_layer_states[1:],
            ),
        )
        malformed = replace(prefill.state, native=malformed_native)
        before_bytes = pager.metrics()["network_or_source_body_bytes"]
        with self.assertRaisesRegex(Exception, "share tensor objects"):
            fork.decode_pair(malformed, 4, 4)
        self.assertEqual(pager.metrics()["network_or_source_body_bytes"], before_bytes)

        one_arm = fork.decode_arm(prefill.state, arm="off", token_id=4)
        self.assertEqual(one_arm.state.off.token_ids, (1, 2, 3, 4))
        self.assertEqual(one_arm.state.native.token_ids, (1, 2, 3))
        self.assertEqual(one_arm.state.first_divergence_position, 3)
        asymmetric_bytes = pager.metrics()["network_or_source_body_bytes"]
        with self.assertRaisesRegex(ValueError, "first differing token index 3"):
            replace(one_arm.state, first_divergence_position=2)
        self.assertEqual(
            pager.metrics()["network_or_source_body_bytes"], asymmetric_bytes
        )

        split = fork.decode_pair(prefill.state, 4, 5)
        adversarial_bytes = pager.metrics()["network_or_source_body_bytes"]
        for invalid_position in (0, 999):
            with self.assertRaisesRegex(ValueError, "first differing token index 3"):
                replace(
                    split.state,
                    first_divergence_position=invalid_position,
                )
        identical_native = replace(
            split.state.native,
            token_ids=split.state.off.token_ids,
        )
        with self.assertRaisesRegex(ValueError, "has no divergence"):
            replace(split.state, native=identical_native)

        tampered_position = replace(split.state)
        object.__setattr__(tampered_position, "first_divergence_position", 999)
        with self.assertRaisesRegex(ValueError, "first differing token index 3"):
            fork.decode_pair(tampered_position, 6, 7)

        tampered_off = replace(split.state.off)
        object.__setattr__(tampered_off, "next_position", 999)
        tampered_cursor = replace(split.state, off=tampered_off)
        with self.assertRaisesRegex(ValueError, "history disagrees"):
            fork.decode_pair(tampered_cursor, 6, 7)
        self.assertEqual(
            pager.metrics()["network_or_source_body_bytes"], adversarial_bytes
        )

        old_off = _clone_state_tensors(split.state.off.layer_states)
        old_native = _clone_state_tensors(split.state.native.layer_states)
        with mock.patch.object(
            fork._native_model,
            "hidden_stateful_range",
            side_effect=RuntimeError("forced native failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "forced native failure"):
                fork.decode_pair(split.state, 6, 7)
        self.assertEqual(split.state.off.token_ids, (1, 2, 3, 4))
        self.assertEqual(split.state.native.token_ids, (1, 2, 3, 5))
        _assert_tensor_snapshots_equal(
            self, old_off, _clone_state_tensors(split.state.off.layer_states)
        )
        _assert_tensor_snapshots_equal(
            self,
            old_native,
            _clone_state_tensors(split.state.native.layer_states),
        )

    def test_greedy_eos_asymmetry_commits_every_emitted_token(self) -> None:
        fork, _, pager = self._runtime()
        scripted = [
            (torch.tensor([[1.0]]), torch.tensor([[2]])),
            (torch.tensor([[1.0]]), torch.tensor([[5]])),
            (torch.tensor([[1.0]]), torch.tensor([[2]])),
        ]
        with mock.patch.object(pager, "topk_logits", side_effect=scripted):
            result = fork.generate_greedy(
                [[1, 4]], max_new_tokens=3, eos_token_ids=(2,)
            )
        self.assertEqual(result.off_token_ids, (2,))
        self.assertEqual(result.native_token_ids, (5, 2))
        self.assertEqual(result.state.off.token_ids, (1, 4, 2))
        self.assertEqual(result.state.native.token_ids, (1, 4, 5, 2))
        self.assertEqual(result.state.off.next_position, 3)
        self.assertEqual(result.state.native.next_position, 4)
        self.assertTrue(result.evidence.off_stopped_on_eos)
        self.assertTrue(result.evidence.native_stopped_on_eos)
        self.assertEqual(result.evidence.first_divergence_step, 0)
        self.assertEqual(result.evidence.off_forward_passes, 2)
        self.assertEqual(result.evidence.native_forward_passes, 3)
        self.assertFalse(result.state.joined)

    def test_greedy_outputs_and_token_chains_match_independent_models(self) -> None:
        fork, intervention, _ = self._runtime()
        off = StreamedQwen38(
            self.config, self._pager(), max_batch_size=1, max_seq_len=32
        )
        native = StreamedQwen38(
            self.config,
            self._pager(),
            native_head_crsa=intervention,
            max_batch_size=1,
            max_seq_len=32,
        )
        prompt = [[1, 11, 6]]
        actual = fork.generate_greedy(prompt, max_new_tokens=1, head_block_rows=7)
        off_tokens, _ = off.generate_greedy(prompt, max_new_tokens=1, head_block_rows=7)
        native_tokens, _ = native.generate_greedy(
            prompt, max_new_tokens=1, head_block_rows=7
        )
        self.assertEqual(actual.off_token_ids, off_tokens)
        self.assertEqual(actual.native_token_ids, native_tokens)
        self.assertEqual(actual.state.off.token_ids, (1, 11, 6, *off_tokens))
        self.assertEqual(actual.state.native.token_ids, (1, 11, 6, *native_tokens))
        _assert_states_equal(self, self._fork_states(actual, "off"), off._layer_states)
        _assert_states_equal(
            self, self._fork_states(actual, "native"), native._layer_states
        )

    def test_traffic_saves_27_of_128_layers_and_reads_layer27_once(self) -> None:
        fork, _, pager = self._runtime()
        preflight = fork.checkpoint_preflight()
        self.assertEqual(preflight["layers"], 28)
        source_before = pager.metrics()["network_or_source_body_bytes"]
        linears_before = pager.metrics()["linear_calls"]
        with (
            mock.patch.object(pager, "linear", wraps=pager.linear) as linear,
            mock.patch.object(
                pager, "linear_many", wraps=pager.linear_many
            ) as linear_many,
        ):
            result = fork.prefill([[1, 4, 7]])
        source_delta = pager.metrics()["network_or_source_body_bytes"] - source_before
        linear_delta = pager.metrics()["linear_calls"] - linears_before
        self.assertEqual(result.traffic.source_body_bytes, source_delta)
        self.assertEqual(result.traffic.linear_calls, linear_delta)
        self.assertEqual(result.traffic.shared_layers, COMMON_STOP_LAYER)
        self.assertEqual(result.traffic.off_layers, 1)
        self.assertEqual(result.traffic.native_layers, 1)
        self.assertEqual(result.traffic.independent_complete_layers, 56)
        self.assertEqual(result.traffic.complete_layers_saved, 27)
        self.assertEqual(result.traffic.fork_layer_weight_passes, 1)
        self.assertEqual(Qwen38NativeFork.complete_layer_savings(64), (27, 128))

        fork_prefix = f"model.language_model.layers.{FORK_LAYER}"
        direct_names = [call.args[1] for call in linear.call_args_list]
        for projection in ("q_proj", "k_proj", "v_proj"):
            name = f"{fork_prefix}.self_attn.{projection}"
            self.assertEqual(direct_names.count(name), 1)
        many_names = [call.args[1] for call in linear_many.call_args_list]
        self.assertEqual(
            [name for name in many_names if name.startswith(fork_prefix)],
            [
                f"{fork_prefix}.self_attn.o_proj",
                f"{fork_prefix}.mlp.gate_proj",
                f"{fork_prefix}.mlp.up_proj",
                f"{fork_prefix}.mlp.down_proj",
            ],
        )


if __name__ == "__main__":
    unittest.main()
