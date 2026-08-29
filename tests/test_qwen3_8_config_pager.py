from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


PINNED_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
QWEN35_DRAFTER_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def _official_config() -> dict:
    layer_types = [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(64)
    ]
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "language_model_only": False,
        "model_type": "qwen3_5",
        "tie_word_embeddings": False,
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attn_output_gate": True,
            "bos_token_id": 248044,
            "dtype": "bfloat16",
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "layer_types": layer_types,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_value_head_dim": 128,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 262144,
            "model_type": "qwen3_5_text",
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "num_attention_heads": 24,
            "num_hidden_layers": 64,
            "num_key_value_heads": 4,
            "output_gate_type": "swish",
            "pad_token_id": None,
            "partial_rotary_factor": 0.25,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10_000_000,
                "rope_type": "default",
            },
            "tie_word_embeddings": False,
            "vocab_size": 248320,
        },
    }


def _qwen35_08b_config() -> dict:
    """Executable fields observed in Qwen/Qwen3.5-0.8B config.json."""

    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "tie_word_embeddings": True,
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attn_output_gate": True,
            "dtype": "bfloat16",
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 1024,
            "intermediate_size": 3584,
            "layer_types": [
                "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
                for index in range(24)
            ],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 16,
            "linear_value_head_dim": 128,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 262144,
            "mlp_only_layers": [],
            "model_type": "qwen3_5_text",
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "num_attention_heads": 8,
            "num_hidden_layers": 24,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10_000_000,
                "rope_type": "default",
            },
            "tie_word_embeddings": True,
            "use_cache": True,
            "vocab_size": 248320,
        },
    }


def _bf16_bytes(values: np.ndarray) -> bytes:
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    words = (contiguous.view(np.uint32) >> np.uint32(16)).astype("<u2")
    return words.tobytes()


class _RawBF16Source:
    def __init__(
        self,
        tensors: dict[str, np.ndarray],
        *,
        repo_id: str | None = None,
        revision: str | None = None,
        source_dtypes: dict[str, str] | None = None,
    ) -> None:
        if repo_id is not None:
            self.repo_id = repo_id
        if revision is not None:
            self.revision = revision
        self.shard = "model.safetensors"
        self.data_start = 32
        self.meta: dict[str, dict] = {}
        payload = bytearray()
        for name, values in tensors.items():
            source_dtype = (source_dtypes or {}).get(name, "BF16")
            if source_dtype == "BF16":
                encoded = _bf16_bytes(values)
            elif source_dtype == "F32":
                encoded = np.ascontiguousarray(values, dtype="<f4").tobytes()
            else:
                raise ValueError(f"unsupported fixture dtype: {source_dtype}")
            begin = len(payload)
            payload.extend(encoded)
            self.meta[name] = {
                "name": name,
                "dtype": source_dtype,
                "shape": list(values.shape),
                "shard": self.shard,
                "data_start": self.data_start,
                "offset_in_shard": [begin, len(payload)],
            }
        self.payload = bytes(payload)
        self.raw_calls: list[tuple[str, int, int]] = []
        self.closed = False

    def find(self, name: str) -> dict:
        return dict(self.meta[name])

    def raw_bytes(self, shard: str, offset: int, length: int) -> bytes:
        if self.closed:
            raise RuntimeError("source closed")
        if shard != self.shard:
            raise KeyError(shard)
        self.raw_calls.append((shard, offset, length))
        relative = offset - self.data_start
        return self.payload[relative : relative + length]

    def metrics(self) -> dict:
        return {
            "network_or_source_body_bytes": sum(
                length for _, _, length in self.raw_calls
            )
        }

    def close(self) -> None:
        self.closed = True


class _RawBF16IntoSource(_RawBF16Source):
    raw_bytes_into_available = True

    def __init__(self, tensors: dict[str, np.ndarray]) -> None:
        super().__init__(tensors)
        self.into_calls: list[tuple[str, int, int]] = []

    def raw_bytes_into(self, shard: str, offset: int, target: object):
        if self.closed:
            raise RuntimeError("source closed")
        if shard != self.shard:
            raise KeyError(shard)
        view = memoryview(target).cast("B")
        relative = offset - self.data_start
        view[:] = self.payload[relative : relative + view.nbytes]
        self.into_calls.append((shard, offset, view.nbytes))
        return SimpleNamespace(
            length=view.nbytes,
            source_requests=1,
            source_bytes=view.nbytes,
        )

    def metrics(self) -> dict:
        return {
            "network_or_source_body_bytes": sum(
                length for _, _, length in (*self.raw_calls, *self.into_calls)
            )
        }


class _ExactHeadFixture:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str, int]] = []

    def topk_logits(self, pager, hidden, *, k, name, block_rows):
        self.calls.append((k, name, block_rows))
        shape = (*hidden.shape[:-1], k)
        return (
            pager.torch.full(
                shape,
                7.0,
                device=pager.device,
                dtype=pager.compute_dtype,
            ),
            pager.torch.zeros(
                shape,
                device=pager.device,
                dtype=pager.torch.long,
            ),
        )

    def metrics(self) -> dict[str, int]:
        return {"calls": len(self.calls)}


class Qwen38ConfigTests(unittest.TestCase):
    def test_official_nested_config_is_strict_and_maps_runtime_fields(self) -> None:
        from immer.runtimes.qwen3_8.config import Qwen38Config

        config = Qwen38Config.from_mapping(_official_config())

        self.assertEqual(config.dim, 5120)
        self.assertEqual(config.rotary_dim, 64)
        self.assertEqual(config.mrope_section, (11, 11, 10))
        self.assertEqual(config.layer_type(0), "linear_attention")
        self.assertTrue(config.is_full_attention(3))
        self.assertFalse(config.is_full_attention(62))
        self.assertTrue(config.is_full_attention(63))

    def test_from_file_and_extracted_text_config(self) -> None:
        from immer.runtimes.qwen3_8.config import Qwen38Config

        raw = _official_config()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            self.assertEqual(Qwen38Config.from_file(path).n_layers, 64)
        self.assertEqual(
            Qwen38Config.from_mapping(raw["text_config"]).vocab_size, 248320
        )

    def test_qwen35_08b_real_nested_config_maps_tied_runtime_contract(self) -> None:
        from immer.runtimes.qwen3_8.config import Qwen38Config

        config = Qwen38Config.from_mapping(_qwen35_08b_config(), require_official=False)

        self.assertEqual(config.dim, 1024)
        self.assertEqual(config.intermediate_size, 3584)
        self.assertEqual(config.n_layers, 24)
        self.assertEqual((config.n_heads, config.n_kv_heads), (8, 2))
        self.assertEqual(
            (config.linear_num_key_heads, config.linear_num_value_heads),
            (16, 16),
        )
        self.assertTrue(config.tie_word_embeddings)
        self.assertIsNone(config.bos_token_id)
        self.assertIsNone(config.pad_token_id)
        self.assertEqual(config.partial_rotary_factor, 0.25)
        self.assertEqual(config.output_gate_type, "swish")
        # Exact shapes observed in the published shard header.
        self.assertEqual(
            (2 * config.n_heads * config.head_dim, config.dim),
            (4096, 1024),
        )
        key_width = config.linear_num_key_heads * config.linear_key_head_dim
        value_width = config.linear_num_value_heads * config.linear_value_head_dim
        self.assertEqual((2 * key_width + value_width, config.dim), (6144, 1024))

    def test_qwen35_08b_rejects_tie_drift_and_explicit_gate_drift(self) -> None:
        from immer.runtimes.qwen3_8.config import (
            Qwen38Config,
            Qwen38ConfigError,
        )

        wrong_tie = _qwen35_08b_config()
        wrong_tie["tie_word_embeddings"] = False
        with self.assertRaisesRegex(Qwen38ConfigError, "values disagree"):
            Qwen38Config.from_mapping(wrong_tie, require_official=False)

        wrong_gate = _qwen35_08b_config()
        wrong_gate["text_config"]["output_gate_type"] = "relu"
        with self.assertRaisesRegex(Qwen38ConfigError, "swish attention output gate"):
            Qwen38Config.from_mapping(wrong_gate, require_official=False)

    def test_rejects_architecture_drift_and_wrong_hybrid_layout(self) -> None:
        from immer.runtimes.qwen3_8.config import (
            Qwen38Config,
            Qwen38ConfigError,
        )

        wrong_dim = copy.deepcopy(_official_config())
        wrong_dim["text_config"]["hidden_size"] = 4096
        with self.assertRaisesRegex(Qwen38ConfigError, "dim=5120"):
            Qwen38Config.from_mapping(wrong_dim)

        wrong_layout = copy.deepcopy(_official_config())
        wrong_layout["text_config"]["layer_types"][2] = "full_attention"
        with self.assertRaisesRegex(Qwen38ConfigError, "linear/full interval"):
            Qwen38Config.from_mapping(wrong_layout)

        wrong_sections = copy.deepcopy(_official_config())
        wrong_sections["text_config"]["rope_parameters"]["mrope_section"] = [8, 8, 8]
        with self.assertRaisesRegex(Qwen38ConfigError, "partition"):
            Qwen38Config.from_mapping(wrong_sections)

    def test_revision_pin_rejects_mutable_or_foreign_remote_source(self) -> None:
        from immer.runtimes.qwen3_8.config import (
            Qwen38ConfigError,
            validate_source_identity,
        )

        self.assertEqual(
            validate_source_identity("Qwen/Qwen3.8-27B", PINNED_REVISION),
            "official-pinned",
        )
        self.assertEqual(
            validate_source_identity("Qwen/Qwen3.5-0.8B", QWEN35_DRAFTER_REVISION),
            "qwen3.5-drafter-pinned",
        )
        with self.assertRaisesRegex(Qwen38ConfigError, "pinned revision"):
            validate_source_identity("Qwen/Qwen3.8-27B", "main")
        with self.assertRaisesRegex(Qwen38ConfigError, "must be one of"):
            validate_source_identity("some/mirror", PINNED_REVISION)
        with self.assertRaisesRegex(Qwen38ConfigError, "no repo/revision"):
            validate_source_identity(None, None, require_identity=True)


class Qwen38PagerTests(unittest.TestCase):
    @staticmethod
    def _source(**identity: str) -> _RawBF16Source:
        return _RawBF16Source(
            {
                "dense.weight": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                "model.language_model.embed_tokens.weight": np.arange(
                    20, dtype=np.float32
                ).reshape(10, 2),
                "lm_head.weight": np.asarray(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 1.0],
                        [-1.0, 0.0],
                        [0.0, -1.0],
                    ],
                    dtype=np.float32,
                ),
                "norm.weight": np.asarray([1.0, 0.5], dtype=np.float32),
            },
            **identity,
        )

    def test_raw_bf16_linear_has_truthful_peak_and_releases_weight(self) -> None:
        import torch

        from immer.runtimes.qwen3_8.pager import Qwen38WeightPager

        source = self._source()
        pager = Qwen38WeightPager(
            source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=8,
        )
        result = pager.linear(torch.tensor([[1.0, 1.0]]), "dense")

        torch.testing.assert_close(
            result.float(), torch.tensor([[3.0, 7.0]]), rtol=0, atol=0
        )
        self.assertEqual(source.raw_calls[-1][2], 8)
        metrics = pager.metrics()
        self.assertEqual(metrics["peak_planned_resident_bytes"], 8)
        self.assertEqual(metrics["logical_weight_bytes"], 8)
        self.assertEqual(metrics["materialized_tensor_bytes"], 8)
        self.assertEqual(metrics["zero_copy_tensor_reads"], 1)
        self.assertEqual(metrics["zero_copy_bytes_avoided"], 8)
        self.assertEqual(metrics["materialized_weight_releases"], 1)
        self.assertEqual(
            metrics["weight_cache_policy"], "one-shot-qwen35-direct-fill/v4"
        )
        self.assertEqual(
            metrics["head_score_policy"],
            "backend-bf16-linear/v1",
        )

    def test_direct_fill_linear_reads_into_torch_storage_without_body_copy(
        self,
    ) -> None:
        import torch

        from immer.runtimes.qwen3_8.pager import Qwen38WeightPager

        source = _RawBF16IntoSource(
            {
                "dense.weight": np.asarray(
                    [[1.0, 2.0], [3.0, 4.0]], dtype=np.float32
                )
            }
        )
        pager = Qwen38WeightPager(
            source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=8,
        )

        with mock.patch.object(
            source,
            "raw_bytes",
            side_effect=AssertionError("intermediate body read"),
        ):
            result = pager.linear(torch.tensor([[1.0, 1.0]]), "dense")

        torch.testing.assert_close(
            result.float(), torch.tensor([[3.0, 7.0]]), rtol=0, atol=0
        )
        self.assertEqual(source.into_calls, [(source.shard, source.data_start, 8)])
        metrics = pager.metrics()
        self.assertEqual(metrics["direct_tensor_fills"], 1)
        self.assertEqual(metrics["direct_tensor_fill_bytes"], 8)
        self.assertEqual(metrics["zero_copy_tensor_reads"], 0)
        self.assertEqual(metrics["peak_planned_resident_bytes"], 8)
        self.assertEqual(metrics["materialized_tensor_bytes"], 8)

    def test_tensor_torch_can_request_one_final_direct_fill_storage(self) -> None:
        import torch

        from immer.runtimes.qwen3_8.pager import Qwen38WeightPager

        source = _RawBF16IntoSource(
            {
                "dense.weight": np.asarray(
                    [[1.0, 2.0], [3.0, 4.0]], dtype=np.float32
                )
            }
        )
        pager = Qwen38WeightPager(
            source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=8,
        )
        with mock.patch.object(
            source,
            "raw_bytes",
            side_effect=AssertionError("intermediate body read"),
        ):
            result = pager.tensor_torch(
                "dense.weight",
                zero_copy_cpu=True,
            )

        torch.testing.assert_close(
            result.float(),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            rtol=0,
            atol=0,
        )
        self.assertEqual(source.into_calls, [(source.shard, source.data_start, 8)])
        self.assertEqual(pager.metrics()["peak_planned_resident_bytes"], 8)
        with self.assertRaisesRegex(TypeError, "zero_copy_cpu"):
            pager.tensor_torch("dense.weight", zero_copy_cpu=1)

    def test_direct_fill_sorted_rows_use_one_final_resident_tensor(self) -> None:
        import torch

        from immer.runtimes.qwen3_8.pager import Qwen38WeightPager

        values = np.arange(20, dtype=np.float32).reshape(10, 2)
        source = _RawBF16IntoSource(
            {"model.language_model.embed_tokens.weight": values}
        )
        pager = Qwen38WeightPager(
            source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=12,
        )

        rows = pager.tensor_rows(
            "model.language_model.embed_tokens.weight",
            (1, 2, 3),
        )

        self.assertTrue(
            torch.equal(rows, torch.from_numpy(values[1:4]).to(torch.bfloat16))
        )
        self.assertEqual(len(source.into_calls), 1)
        metrics = pager.metrics()
        self.assertEqual(metrics["direct_tensor_fills"], 1)
        self.assertEqual(metrics["direct_tensor_fill_bytes"], 12)
        self.assertEqual(metrics["peak_planned_resident_bytes"], 12)
        self.assertEqual(metrics["row_reads"], 1)
        self.assertEqual(metrics["logical_weight_bytes"], 12)
        self.assertEqual(metrics["materialized_tensor_bytes"], 12)

    def test_direct_fill_restore_index_is_inside_resident_preflight(self) -> None:
        import torch

        from immer.runtimes.qwen3_8.pager import (
            Qwen38PagerError,
            Qwen38WeightPager,
        )

        name = "model.language_model.embed_tokens.weight"
        values = np.arange(20, dtype=np.float32).reshape(10, 2)
        rejected_source = _RawBF16IntoSource({name: values})
        rejected = Qwen38WeightPager(
            rejected_source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=43,
        )
        with self.assertRaisesRegex(Qwen38PagerError, "need 44 resident bytes"):
            rejected.tensor_rows(name, (3, 1, 3))
        self.assertEqual(rejected_source.into_calls, [])

        source = _RawBF16IntoSource({name: values})
        pager = Qwen38WeightPager(
            source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=44,
        )
        rows = pager.tensor_rows(name, (3, 1, 3))

        self.assertTrue(
            torch.equal(
                rows,
                torch.from_numpy(values[[3, 1, 3]]).to(torch.bfloat16),
            )
        )
        self.assertEqual(pager.metrics()["peak_planned_resident_bytes"], 44)

    def test_exact_head_dispatch_and_progress_fallback_preserve_public_abi(
        self,
    ) -> None:
        import torch

        from immer.runtimes.qwen3_8.pager import Qwen38WeightPager

        source = self._source()
        index = _ExactHeadFixture()
        pager = Qwen38WeightPager(
            source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=256,
            exact_head_index=index,
        )
        hidden = torch.tensor([[1.0, 1.0]])

        values, ids = pager.topk_logits(hidden, k=1, block_rows=2)

        self.assertEqual(index.calls, [(1, "lm_head.weight", 2)])
        self.assertEqual(float(values[0, 0]), 7.0)
        self.assertEqual(int(ids[0, 0]), 0)
        self.assertEqual(source.raw_calls, [])
        self.assertTrue(pager.metrics()["exact_head_index_attached"])
        self.assertEqual(pager.metrics()["exact_head_calls"], 1)
        self.assertEqual(
            pager.metrics()["head_score_policy"],
            "cpu-bf16-explicit-fp32-accumulate-rne/v1",
        )

        progress = []
        baseline_values, baseline_ids = pager.topk_logits(
            hidden,
            k=1,
            block_rows=2,
            progress=progress.append,
        )
        self.assertEqual(index.calls, [(1, "lm_head.weight", 2)])
        self.assertEqual(len(progress), 3)
        self.assertEqual(int(baseline_ids[0, 0]), 2)
        self.assertEqual(float(baseline_values[0, 0]), 2.0)

        pager.attach_exact_head_index(None)
        self.assertFalse(pager.metrics()["exact_head_index_attached"])
        with self.assertRaises(TypeError):
            pager.attach_exact_head_index(object())

    def test_real_qwen35_f32_control_tensor_is_range_decoded_exactly(self) -> None:
        import torch

        from immer.runtimes.qwen3_8.pager import Qwen38WeightPager

        values = np.asarray([-0.75, 0.125, 3.5], dtype=np.float32)
        source = _RawBF16Source(
            {"model.language_model.layers.0.linear_attn.A_log": values},
            source_dtypes={"model.language_model.layers.0.linear_attn.A_log": "F32"},
        )
        pager = Qwen38WeightPager(
            source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=32,
        )

        actual = pager.tensor_torch(
            "model.language_model.layers.0.linear_attn.A_log",
            dtype=torch.float32,
        )

        self.assertTrue(torch.equal(actual, torch.from_numpy(values.copy())))
        self.assertEqual(source.raw_calls, [(source.shard, source.data_start, 12)])
        self.assertEqual(pager.metrics()["logical_weight_bytes"], 12)

    def test_linear_many_is_bit_exact_and_reads_one_matrix_once(self) -> None:
        import torch

        from immer.runtimes.qwen3_8.pager import Qwen38WeightPager

        first = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        second = torch.tensor(
            [[2.0, -1.0], [0.5, 3.0]],
            dtype=torch.float32,
        )
        reference = Qwen38WeightPager(
            self._source(),
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=16,
        )
        expected = (
            reference.linear(first, "dense"),
            reference.linear(second, "dense"),
        )

        source = self._source()
        pager = Qwen38WeightPager(
            source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=16,
        )
        original_linear = torch.nn.functional.linear
        kernel_shapes: list[tuple[int, ...]] = []

        def observed_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            kernel_shapes.append(tuple(x.shape))
            return original_linear(x, weight)

        observed_weights: list[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]] = []
        with mock.patch.object(
            torch.nn.functional,
            "linear",
            side_effect=observed_linear,
        ):
            actual = pager.linear_many(
                (first, second),
                "dense",
                weight_observer=lambda weight, results: observed_weights.append(
                    (
                        tuple(weight.shape),
                        tuple(tuple(result.shape) for result in results),
                    )
                ),
            )

        self.assertIsInstance(actual, tuple)
        self.assertEqual(kernel_shapes, [(1, 2), (2, 2)])
        self.assertEqual(observed_weights, [((2, 2), ((1, 2), (2, 2)))])
        for value, wanted in zip(actual, expected, strict=True):
            self.assertTrue(torch.equal(value, wanted))
        self.assertEqual(len(source.raw_calls), 1)
        self.assertEqual(source.raw_calls[0][2], 8)
        metrics = pager.metrics()
        self.assertEqual(metrics["tensor_reads"], 1)
        self.assertEqual(metrics["linear_calls"], 2)
        self.assertEqual(metrics["logical_weight_bytes"], 8)
        self.assertEqual(metrics["materialized_tensor_bytes"], 8)
        self.assertEqual(metrics["materialized_weight_releases"], 1)
        self.assertEqual(metrics["network_or_source_body_bytes"], 8)

    def test_linear_observer_sees_resident_weight_and_returned_result(self) -> None:
        import torch

        from immer.runtimes.qwen3_8.pager import Qwen38WeightPager

        source = self._source()
        pager = Qwen38WeightPager(
            source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=16,
        )
        seen = []
        result = pager.linear(
            torch.ones((1, 2)),
            "dense",
            weight_observer=lambda weight, output: seen.append(
                (tuple(weight.shape), output.clone())
            ),
        )

        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0], (2, 2))
        self.assertTrue(torch.equal(seen[0][1], result))
        self.assertEqual(len(source.raw_calls), 1)
        self.assertEqual(pager.metrics()["materialized_weight_releases"], 1)

    def test_linear_many_rejects_all_inputs_before_reading_weight(self) -> None:
        import torch

        from immer.runtimes.qwen3_8.pager import (
            Qwen38PagerError,
            Qwen38WeightPager,
        )

        source = self._source()
        pager = Qwen38WeightPager(source, device="cpu")

        with self.assertRaisesRegex(ValueError, "at least two"):
            pager.linear_many((torch.ones((1, 2)),), "dense")
        with self.assertRaisesRegex(Qwen38PagerError, "input 1 width 3"):
            pager.linear_many(
                (torch.ones((1, 2)), torch.ones((1, 3))),
                "dense",
            )
        with self.assertRaises((TypeError, RuntimeError)):
            pager.linear_many(
                (torch.ones((1, 2)), torch.ones((1, 2))),
                "dense",
                output_dtype="not-a-dtype",
            )

        self.assertEqual(source.raw_calls, [])
        metrics = pager.metrics()
        self.assertEqual(metrics["tensor_reads"], 0)
        self.assertEqual(metrics["linear_calls"], 0)
        self.assertEqual(metrics["logical_weight_bytes"], 0)
        self.assertEqual(metrics["materialized_tensor_bytes"], 0)
        self.assertEqual(metrics["materialized_weight_releases"], 0)

    def test_linear_many_packed_uses_one_gemm_and_restores_row_shapes(self) -> None:
        import torch

        from immer.runtimes.qwen3_8.pager import Qwen38WeightPager

        first = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        second = torch.tensor([[2.0, -1.0], [0.5, 3.0]], dtype=torch.float32)
        source = self._source()
        pager = Qwen38WeightPager(
            source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=16,
        )
        original_linear = torch.nn.functional.linear
        kernel_shapes = []

        def observed_linear(x, weight):
            kernel_shapes.append(tuple(x.shape))
            return original_linear(x, weight)

        with mock.patch.object(
            torch.nn.functional,
            "linear",
            side_effect=observed_linear,
        ):
            actual = pager.linear_many(
                (first, second),
                "dense",
                packed=True,
            )

        self.assertEqual(kernel_shapes, [(3, 2)])
        self.assertEqual(tuple(actual[0].shape), (1, 2))
        self.assertEqual(tuple(actual[1].shape), (2, 2))
        expected = original_linear(
            torch.cat((first, second), dim=0).to(torch.bfloat16),
            torch.tensor(
                [[1.0, 2.0], [3.0, 4.0]],
                dtype=torch.bfloat16,
            ),
        )
        self.assertTrue(torch.equal(actual[0], expected[:1]))
        self.assertTrue(torch.equal(actual[1], expected[1:]))
        metrics = pager.metrics()
        self.assertEqual(metrics["linear_calls"], 1)
        self.assertEqual(metrics["packed_linear_calls"], 1)
        self.assertEqual(metrics["packed_linear_rows"], 3)
        self.assertEqual(len(source.raw_calls), 1)

    def test_linear_many_releases_materialized_weight_after_kernel_failure(
        self,
    ) -> None:
        import torch

        from immer.runtimes.qwen3_8.pager import Qwen38WeightPager

        source = self._source()
        pager = Qwen38WeightPager(
            source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=16,
        )
        original_linear = torch.nn.functional.linear
        calls = 0

        def fail_second(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected kernel failure")
            return original_linear(x, weight)

        with mock.patch.object(
            torch.nn.functional,
            "linear",
            side_effect=fail_second,
        ):
            with self.assertRaisesRegex(RuntimeError, "injected kernel failure"):
                pager.linear_many(
                    (torch.ones((1, 2)), torch.ones((2, 2))),
                    "dense",
                )

        metrics = pager.metrics()
        self.assertEqual(len(source.raw_calls), 1)
        self.assertEqual(metrics["tensor_reads"], 1)
        self.assertEqual(metrics["linear_calls"], 1)
        self.assertEqual(metrics["logical_weight_bytes"], 8)
        self.assertEqual(metrics["materialized_tensor_bytes"], 8)
        self.assertEqual(metrics["materialized_weight_releases"], 1)

    def test_fp32_peak_is_payload_plus_decoded_target(self) -> None:
        from immer.runtimes.qwen3_8.pager import (
            Qwen38PagerError,
            Qwen38WeightPager,
        )

        pager = Qwen38WeightPager(
            self._source(),
            device="cpu",
            compute_dtype="float32",
            max_resident_bytes=23,
        )
        with self.assertRaisesRegex(Qwen38PagerError, "needs 24 resident bytes"):
            pager.tensor_torch("dense.weight")

    def test_selected_rows_enforce_aggregate_not_per_range_residency(self) -> None:
        from immer.runtimes.qwen3_8.pager import (
            Qwen38PagerError,
            Qwen38WeightPager,
        )

        pager = Qwen38WeightPager(
            self._source(),
            device="cpu",
            compute_dtype="float32",
            max_resident_bytes=47,
        )
        with self.assertRaisesRegex(Qwen38PagerError, "need 48 resident bytes"):
            pager.embedding([1, 3, 5])

    def test_embeddings_candidates_and_exact_blocked_topk(self) -> None:
        import torch

        from immer.runtimes.qwen3_8.pager import Qwen38WeightPager

        pager = Qwen38WeightPager(
            self._source(),
            device="cpu",
            compute_dtype="float32",
            max_resident_bytes=256,
        )
        embeddings = pager.embedding([3, 1, 3])
        torch.testing.assert_close(
            embeddings,
            torch.tensor([[6.0, 7.0], [2.0, 3.0], [6.0, 7.0]]),
        )
        arbitrary = pager.tensor_rows(
            "model.language_model.embed_tokens.weight", [2, 0, 2]
        )
        torch.testing.assert_close(
            arbitrary,
            torch.tensor([[4.0, 5.0], [0.0, 1.0], [4.0, 5.0]]),
        )
        hidden = torch.tensor([[2.0, 1.0]])
        candidates = pager.candidate_logits(hidden, [3, 2, 0])
        torch.testing.assert_close(candidates, torch.tensor([[-2.0, 3.0, 2.0]]))
        values, ids = pager.topk_logits(hidden, k=3, block_rows=2)
        torch.testing.assert_close(values, torch.tensor([[3.0, 2.0, 1.0]]))
        torch.testing.assert_close(ids, torch.tensor([[2, 0, 1]]))
        self.assertEqual(pager.metrics()["head_rows"], 8)

    def test_remote_source_must_use_the_pinned_revision(self) -> None:
        from immer.runtimes.qwen3_8.config import Qwen38ConfigError
        from immer.runtimes.qwen3_8.pager import Qwen38WeightPager

        with self.assertRaisesRegex(Qwen38ConfigError, "pinned revision"):
            Qwen38WeightPager(
                self._source(repo_id="Qwen/Qwen3.8-27B", revision="main"),
                device="cpu",
            )
        pager = Qwen38WeightPager(
            self._source(repo_id="Qwen/Qwen3.8-27B", revision=PINNED_REVISION),
            device="cpu",
        )
        self.assertEqual(pager.source_identity, "official-pinned")

    def test_close_is_idempotent_closes_owned_source_and_blocks_reads(self) -> None:
        from immer.runtimes.qwen3_8.pager import (
            Qwen38PagerError,
            Qwen38WeightPager,
        )

        source = self._source()
        pager = Qwen38WeightPager(source, device="cpu", close_source=True)
        with mock.patch(
            "immer.runtimes.qwen3_8.pager.gc.collect", return_value=7
        ) as collect:
            pager.close()
            pager.close()

        self.assertTrue(source.closed)
        self.assertTrue(pager.metrics()["closed"])
        self.assertEqual(pager.metrics()["release_boundaries"], 1)
        self.assertEqual(pager.metrics()["gc_collections"], 1)
        self.assertEqual(pager.metrics()["gc_collections_forced"], 1)
        self.assertEqual(pager.metrics()["gc_objects_collected"], 7)
        collect.assert_called_once_with()
        with self.assertRaisesRegex(Qwen38PagerError, "closed"):
            pager.tensor_torch("norm.weight")

    def test_cpu_release_skips_gc_below_pressure_and_interval(self) -> None:
        from immer.runtimes.qwen3_8.pager import Qwen38WeightPager

        pager = Qwen38WeightPager(
            self._source(),
            device="cpu",
            gc_interval_boundaries=8,
            gc_rss_limit_bytes=10_000,
        )
        with (
            mock.patch(
                "immer.runtimes.qwen3_8.pager._process_rss_bytes",
                return_value=1_000,
            ),
            mock.patch("immer.runtimes.qwen3_8.pager.gc.collect") as collect,
        ):
            pager.release()
            pager.release()

        metrics = pager.metrics()
        self.assertEqual(metrics["release_boundaries"], 2)
        self.assertEqual(metrics["gc_collections_skipped"], 2)
        self.assertEqual(metrics["gc_collections"], 0)
        self.assertEqual(metrics["gc_last_observed_rss_bytes"], 1_000)
        collect.assert_not_called()

    def test_darwin_default_limit_uses_current_not_historical_peak_rss(self) -> None:
        from immer.runtimes.qwen3_8.pager import Qwen38WeightPager

        current_rss = 123_456
        historical_peak = 9 * 1024**3
        with (
            mock.patch(
                "immer.runtimes.qwen3_8.pager.sys.platform",
                "darwin",
            ),
            mock.patch(
                "immer.runtimes.qwen3_8.pager._darwin_current_rss_bytes",
                return_value=current_rss,
            ) as current,
        ):
            pager = Qwen38WeightPager(self._source(), device="cpu")

        self.assertEqual(
            pager.gc_rss_limit_bytes,
            current_rss + pager.DEFAULT_GC_RSS_HEADROOM_BYTES,
        )
        self.assertLess(pager.gc_rss_limit_bytes, historical_peak)
        current.assert_called_once_with()
        pager.close()

    def test_rss_pressure_and_deterministic_interval_force_gc(self) -> None:
        from immer.runtimes.qwen3_8.pager import Qwen38WeightPager

        pressure = Qwen38WeightPager(
            self._source(),
            device="cpu",
            gc_interval_boundaries=8,
            gc_rss_limit_bytes=10_000,
        )
        interval = Qwen38WeightPager(
            self._source(),
            device="cpu",
            gc_interval_boundaries=2,
            gc_rss_limit_bytes=10_000,
        )
        with (
            mock.patch(
                "immer.runtimes.qwen3_8.pager._process_rss_bytes",
                side_effect=(10_001, 1_000, 1_000),
            ),
            mock.patch(
                "immer.runtimes.qwen3_8.pager.gc.collect",
                return_value=0,
            ) as collect,
        ):
            pressure.release()
            interval.release()
            interval.release()

        self.assertEqual(collect.call_count, 2)
        self.assertEqual(pressure.metrics()["gc_collections_pressure"], 1)
        self.assertEqual(interval.metrics()["gc_collections_skipped"], 1)
        self.assertEqual(interval.metrics()["gc_collections_interval"], 1)

    def test_forced_teardown_does_not_shift_interval_cadence(self) -> None:
        from immer.runtimes.qwen3_8.pager import Qwen38WeightPager

        pager = Qwen38WeightPager(
            self._source(),
            device="cpu",
            gc_interval_boundaries=3,
            gc_rss_limit_bytes=10_000,
        )
        with (
            mock.patch(
                "immer.runtimes.qwen3_8.pager._process_rss_bytes",
                return_value=1_000,
            ),
            mock.patch(
                "immer.runtimes.qwen3_8.pager.gc.collect",
                return_value=0,
            ) as collect,
        ):
            pager.release()
            pager.release(force_gc=True)
            pager.release()
            self.assertEqual(pager.metrics()["gc_collections_interval"], 0)
            pager.release()

        metrics = pager.metrics()
        self.assertEqual(metrics["release_boundaries"], 4)
        self.assertEqual(metrics["gc_policy_boundaries"], 3)
        self.assertEqual(metrics["gc_collections_skipped"], 2)
        self.assertEqual(metrics["gc_collections_forced"], 1)
        self.assertEqual(metrics["gc_collections_interval"], 1)
        self.assertEqual(collect.call_count, 2)

    def test_missing_rss_fails_closed_and_mps_cache_is_still_explicit(self) -> None:
        import torch

        from immer.runtimes.qwen3_8.pager import Qwen38WeightPager

        pager = Qwen38WeightPager(
            self._source(),
            device="cpu",
            gc_interval_boundaries=8,
            gc_rss_limit_bytes=10_000,
        )
        pager.device = torch.device("mps")
        with (
            mock.patch(
                "immer.runtimes.qwen3_8.pager._process_rss_bytes",
                return_value=None,
            ),
            mock.patch("immer.runtimes.qwen3_8.pager.gc.collect", return_value=0),
            mock.patch.object(pager.torch.mps, "empty_cache") as empty_cache,
        ):
            pager.release()
            pager.close()

        metrics = pager.metrics()
        self.assertEqual(metrics["gc_rss_measurement_failures"], 1)
        self.assertEqual(metrics["gc_collections_fail_closed"], 1)
        self.assertEqual(metrics["gc_collections_forced"], 1)
        self.assertEqual(metrics["mps_cache_purges"], 2)
        self.assertEqual(empty_cache.call_count, 2)

    def test_real_local_streamer_contract_reads_only_requested_bf16_range(self) -> None:
        import torch
        from safetensors.torch import save_file

        from immer.knowledge.streamer import Streamer
        from immer.runtimes.qwen3_8.pager import Qwen38WeightPager

        with tempfile.TemporaryDirectory() as tmp:
            save_file(
                {"dense.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]).bfloat16()},
                str(Path(tmp) / "model.safetensors"),
            )
            source = Streamer.from_local(
                tmp,
                budget_mb=0.01,
                use_cache=False,
            )
            source.inventory()
            metadata_bytes = source.metrics()["network_or_source_body_bytes"]
            pager = Qwen38WeightPager(
                source,
                device="cpu",
                compute_dtype="bfloat16",
                max_resident_bytes=16,
                close_source=True,
            )
            result = pager.linear(torch.ones((1, 2)), "dense")
            torch.testing.assert_close(
                result.float(), torch.tensor([[3.0, 7.0]]), rtol=0, atol=0
            )
            self.assertEqual(
                source.metrics()["network_or_source_body_bytes"] - metadata_bytes,
                8,
            )
            pager.close()


if __name__ == "__main__":
    unittest.main()
