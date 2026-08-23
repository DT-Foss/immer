from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import torch
from safetensors.torch import save_file

from immer.knowledge.streamer import Streamer
from immer.runtimes.qwen3_8.config import Qwen38Config
from immer.runtimes.qwen3_8.draft_verification import (
    DRAFT_VERIFICATION_SCHEMA,
    Qwen38DraftVerifier,
)
from immer.runtimes.qwen3_8.model import StreamedQwen38
from immer.runtimes.qwen3_8.pager import Qwen38WeightPager
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


def _matrix(rows: int, columns: int, generator: torch.Generator) -> torch.Tensor:
    return (0.04 * torch.randn(rows, columns, generator=generator)).to(
        torch.bfloat16
    )


def _tiny_weights(config: Qwen38Config) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(31)
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
            tensors[f"{attn}.A_log"] = torch.tensor(
                [-0.7, 0.1], dtype=torch.bfloat16
            )
            tensors[f"{attn}.dt_bias"] = torch.tensor(
                [-0.2, 0.3], dtype=torch.bfloat16
            )
            tensors[f"{attn}.norm.weight"] = torch.ones(
                config.linear_value_head_dim, dtype=torch.bfloat16
            )
            tensors[f"{attn}.out_proj.weight"] = _matrix(
                config.dim, value_features, generator
            )
    return tensors


class Qwen38ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = _tiny_config()
        save_file(_tiny_weights(self.config), self.root / "model.safetensors")
        self.source = Streamer.from_local(
            self.root, budget_mb=20, use_cache=False
        )
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

    def test_complete_prefill_is_finite_and_accounts_every_linear(self) -> None:
        final, evidence = self.model.forward_prefill(torch.tensor([[1, 4, 9]]))
        self.assertEqual(tuple(final.shape), (1, 3, self.config.dim))
        self.assertTrue(torch.isfinite(final).all())
        self.assertEqual(evidence.layers_executed, 4)
        self.assertEqual(evidence.linear_calls, 31)
        self.assertGreater(evidence.source_body_bytes, 0)

    def test_right_padding_preserves_each_valid_prefix(self) -> None:
        batch_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]])
        mask = torch.tensor(
            [[True, True, True, True], [True, True, False, False]]
        )
        padded, _ = self.model.forward_prefill(batch_ids, token_mask=mask)
        exact, _ = self.model.forward_prefill(torch.tensor([[5, 6]]))
        torch.testing.assert_close(
            padded[1, :2], exact[0], rtol=1e-5, atol=1e-6
        )

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


if __name__ == "__main__":
    unittest.main()
