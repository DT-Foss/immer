from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class DeepSeekV4ConfigTests(unittest.TestCase):
    def _config(self) -> dict:
        return {
            "architectures": ["DeepseekV4ForCausalLM"],
            "model_type": "deepseek_v4",
            "vocab_size": 128,
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "head_dim": 128,
            "qk_rope_head_dim": 64,
            "q_lora_rank": 128,
            "o_lora_rank": 128,
            "o_groups": 2,
            "moe_intermediate_size": 128,
            "n_routed_experts": 8,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "num_hash_layers": 1,
            "rms_norm_eps": 1e-6,
            "hc_mult": 4,
            "hc_sinkhorn_iters": 20,
            "hc_eps": 1e-6,
            "sliding_window": 16,
            "compress_ratios": [0, 4],
            "rope_theta": 10000,
            "compress_rope_theta": 40000,
            "max_position_embeddings": 1024,
            "rope_scaling": {
                "original_max_position_embeddings": 1024,
                "factor": 2,
                "beta_fast": 32,
                "beta_slow": 1,
            },
            "index_n_heads": 8,
            "index_head_dim": 128,
            "index_topk": 16,
            "scoring_func": "sqrtsoftplus",
            "routed_scaling_factor": 1.5,
            "swiglu_limit": 10,
            "expert_dtype": "fp4",
            "dspark_target_layer_ids": [1],
        }

    def test_official_config_is_mapped_and_validated(self) -> None:
        from immer.runtimes.deepseek_v4 import DeepSeekV4Config

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(self._config()), encoding="utf-8")
            config = DeepSeekV4Config.from_file(path)
        self.assertEqual(config.dim, 128)
        self.assertEqual(config.compress_ratios, (0, 4))
        self.assertEqual(config.max_position_embeddings, 1024)
        self.assertEqual(config.target_layer_ids, (1,))

    def test_architecture_drift_fails_closed(self) -> None:
        from immer.runtimes.deepseek_v4.config import (
            DeepSeekV4Config,
            DeepSeekV4ConfigError,
        )

        raw = self._config()
        raw["expert_dtype"] = "int3"
        with self.assertRaisesRegex(DeepSeekV4ConfigError, "fp4 experts"):
            DeepSeekV4Config.from_mapping(raw)

        raw = self._config()
        del raw["max_position_embeddings"]
        with self.assertRaisesRegex(DeepSeekV4ConfigError, "max_position_embeddings"):
            DeepSeekV4Config.from_mapping(raw)


class DeepSeekV4QuantizationTests(unittest.TestCase):
    def test_fp4_unpack_is_low_nibble_first_and_signed(self) -> None:
        from immer.runtimes.deepseek_v4 import unpack_fp4_e2m1

        packed = np.asarray([[0x21, 0xF8]], dtype=np.uint8)
        actual = unpack_fp4_e2m1(packed)
        np.testing.assert_array_equal(actual, [[0.5, 1.0, 0.0, -6.0]])

    def test_fp4_scales_repeat_per_32_values_per_row(self) -> None:
        from immer.runtimes.deepseek_v4 import dequantize_fp4_e2m1

        packed = np.full((2, 32), 0x22, dtype=np.uint8)
        scale = np.asarray([[2.0, 4.0], [8.0, 16.0]], dtype=np.float32)
        actual = dequantize_fp4_e2m1(packed, scale)
        self.assertEqual(actual.shape, (2, 64))
        np.testing.assert_array_equal(actual[0, :32], 2.0)
        np.testing.assert_array_equal(actual[0, 32:], 4.0)
        np.testing.assert_array_equal(actual[1, :32], 8.0)
        np.testing.assert_array_equal(actual[1, 32:], 16.0)

    def test_fp8_scales_repeat_over_2d_tiles(self) -> None:
        from immer.runtimes.deepseek_v4 import dequantize_fp8_e4m3

        weight = np.ones((5, 5), dtype=np.float32)
        scale = np.asarray([[2.0, 3.0], [5.0, 7.0]], dtype=np.float32)
        actual = dequantize_fp8_e4m3(weight, scale, block_size=4)
        np.testing.assert_array_equal(actual[:4], np.tile([2, 2, 2, 2, 3], (4, 1)))
        np.testing.assert_array_equal(actual[4], [5, 5, 5, 5, 7])

    def test_activation_quantizer_rounds_scale_to_power_of_two(self) -> None:
        from immer.runtimes.deepseek_v4 import quantize_dequantize_fp8

        x = np.zeros((1, 128), dtype=np.float32)
        x[0, 0] = 448.0
        x[0, 1] = 1.1
        actual = quantize_dequantize_fp8(x)
        self.assertEqual(actual[0, 0], 448.0)
        self.assertEqual(actual[0, 1], np.float32(1.125))
        self.assertTrue(np.isfinite(actual).all())

    def test_activation_quantizer_exposes_exact_codes_and_k128_scales(self) -> None:
        from immer.runtimes.deepseek_v4 import (
            quantize_dequantize_fp8,
            quantize_fp8_e4m3_parts,
        )

        x = np.linspace(-500.0, 500.0, 2 * 3 * 256, dtype=np.float32).reshape(2, 3, 256)
        values, scales = quantize_fp8_e4m3_parts(x)
        self.assertEqual(values.shape, x.shape)
        self.assertEqual(scales.shape, (2, 3, 2))
        reconstructed = (values.reshape(2, 3, 2, 128) * scales[..., None]).reshape(
            x.shape
        )
        np.testing.assert_array_equal(reconstructed, quantize_dequantize_fp8(x))
        self.assertTrue(np.equal(np.log2(scales), np.round(np.log2(scales))).all())

    def test_fp4_activation_quantizer_uses_e2m1_ties_to_even(self) -> None:
        from immer.runtimes.deepseek_v4 import quantize_dequantize_fp4

        x = np.zeros((1, 32), dtype=np.float32)
        x[0, :8] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0]
        actual = quantize_dequantize_fp4(x)
        np.testing.assert_array_equal(actual[0, :8], [0, 1, 1, 2, 2, 4, 4, 6])

    def test_fp4_activation_quantizer_preserves_small_bfloat16_signals(self) -> None:
        from immer.runtimes.deepseek_v4 import quantize_dequantize_fp4

        x = np.zeros((2, 32), dtype=np.float32)
        x[0, :2] = [2.0**-120, -(2.0**-120)]
        x[1, :2] = [2.0**-126, -(2.0**-126)]
        actual = quantize_dequantize_fp4(x)
        self.assertTrue(np.isfinite(actual).all())
        self.assertGreater(actual[0, 0], 0.0)
        self.assertLess(actual[0, 1], 0.0)
        self.assertGreater(actual[1, 0], 0.0)
        self.assertLess(actual[1, 1], 0.0)


if __name__ == "__main__":
    unittest.main()
