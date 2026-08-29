from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from immer.runtimes.qwen3_8.config import Qwen38Config
from immer.runtimes.qwen3_8.mtp_draft import (
    MTP_CONTROL_NAMES,
    MTP_MATRIX_NAMES,
    Qwen35MtpDraftProvider,
)
from immer.runtimes.qwen3_8.pager import Qwen38WeightPager


def _config() -> Qwen38Config:
    return Qwen38Config.from_mapping(
        {
            "model_type": "qwen3_5_text",
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "layer_types": ["full_attention"],
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "partial_rotary_factor": 0.25,
            "full_attention_interval": 1,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "linear_conv_kernel_dim": 4,
            "max_position_embeddings": 1024,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 10_000.0,
                "mrope_interleaved": True,
                "mrope_section": [1, 1, 2],
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
            "bos_token_id": 126,
            "eos_token_id": 127,
            "pad_token_id": None,
        },
        require_official=False,
    )


class _MetricsOwner:
    def __init__(self) -> None:
        self.values = {"network_or_source_body_bytes": 0}

    def metrics(self) -> dict[str, int]:
        return dict(self.values)


class _Bank:
    def __init__(self) -> None:
        self.logical = 0
        self.identity = {"manifest_sha256": "a" * 64}

    def has(self, name: str) -> bool:
        return name in MTP_MATRIX_NAMES

    def metrics(self) -> dict[str, int]:
        return {"logical_weight_bytes": self.logical}


class _Pager(Qwen38WeightPager):
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.torch = torch
        self.compute_dtype = torch.bfloat16
        self.device = torch.device("cpu")
        self.q4_bank = _Bank()
        self.source = _MetricsOwner()
        self.tensors = tensors
        self.linears = 0
        self.releases = 0

    def tensor_torch(self, name, *, dtype=None, device=None, zero_copy_cpu=False):
        del zero_copy_cpu
        value = self.tensors[name]
        return value.to(dtype=dtype or value.dtype, device=device or "cpu")

    def embedding(self, token_ids, *, name="model.language_model.embed_tokens.weight"):
        ids = torch.tensor(tuple(token_ids), dtype=torch.long)
        return self.tensors[name].index_select(0, ids).to(torch.bfloat16)

    def linear(self, x, prefix, *, output_dtype=None, weight_observer=None):
        del weight_observer
        name = prefix if prefix.endswith(".weight") else f"{prefix}.weight"
        result = torch.nn.functional.linear(
            x.to(torch.bfloat16),
            self.tensors[name].to(torch.bfloat16),
        )
        self.linears += 1
        self.q4_bank.logical += self.tensors[name].numel()
        return result.to(dtype=output_dtype or torch.bfloat16)

    def linear_group(self, x, names, *, output_dtype=None):
        return tuple(self.linear(x, name, output_dtype=output_dtype) for name in names)

    def mlp(self, x, names):
        gate, up = self.linear_group(x, names[:2])
        return self.linear(
            torch.nn.functional.silu(gate) * up,
            names[2],
        )

    def topk_logits(self, hidden, *, k=1, name="lm_head.weight", **_kwargs):
        logits = self.linear(hidden, name, output_dtype=torch.bfloat16)
        ids = torch.arange(logits.shape[-1], dtype=torch.long).expand_as(logits)
        id_order = torch.argsort(ids, dim=-1, stable=True)
        logits = torch.gather(logits, -1, id_order)
        ids = torch.gather(ids, -1, id_order)
        order = torch.argsort(logits, dim=-1, descending=True, stable=True)[..., :k]
        return torch.gather(logits, -1, order), torch.gather(ids, -1, order)

    def release(self, *, force_gc=False):
        del force_gc
        self.releases += 1

    def metrics(self) -> dict[str, int]:
        return {"linear_calls": self.linears}


def _tensors(config: Qwen38Config) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(3501)

    def weight(rows: int, columns: int) -> torch.Tensor:
        return (0.02 * torch.randn((rows, columns), generator=generator)).to(
            torch.bfloat16
        )

    tensors = {
        "model.language_model.embed_tokens.weight": weight(
            config.vocab_size, config.dim
        ),
        "lm_head.weight": weight(config.vocab_size, config.dim),
        "mtp.fc.weight": weight(config.dim, 2 * config.dim),
        "mtp.layers.0.self_attn.q_proj.weight": weight(
            2 * config.n_heads * config.head_dim,
            config.dim,
        ),
        "mtp.layers.0.self_attn.k_proj.weight": weight(
            config.n_kv_heads * config.head_dim,
            config.dim,
        ),
        "mtp.layers.0.self_attn.v_proj.weight": weight(
            config.n_kv_heads * config.head_dim,
            config.dim,
        ),
        "mtp.layers.0.self_attn.o_proj.weight": weight(config.dim, config.dim),
        "mtp.layers.0.mlp.gate_proj.weight": weight(
            config.intermediate_size,
            config.dim,
        ),
        "mtp.layers.0.mlp.up_proj.weight": weight(
            config.intermediate_size,
            config.dim,
        ),
        "mtp.layers.0.mlp.down_proj.weight": weight(
            config.dim,
            config.intermediate_size,
        ),
    }
    for name in MTP_CONTROL_NAMES:
        width = (
            config.head_dim
            if name.endswith(("q_norm.weight", "k_norm.weight"))
            else config.dim
        )
        tensors[name] = torch.zeros(width, dtype=torch.bfloat16)
    return tensors


class Qwen35MtpDraftTests(unittest.TestCase):
    def test_shifted_prefill_proposal_and_prefix_reconciliation(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        state_path = Path(temporary.name) / "mtp.json"
        config = _config()
        pager = _Pager(_tensors(config))
        provider = Qwen35MtpDraftProvider(
            config,
            pager,
            proposal_width=3,
            eos_token_ids=(),
            head_block_rows=32,
            state_path=state_path,
        )
        history = (4, 7, 11, 19)
        target_hidden = torch.randn((1, len(history), config.dim)).to(torch.bfloat16)
        provider.begin_request_state(history, target_hidden)
        self.assertEqual(provider._next_position, len(history) - 1)

        evidence = provider.propose_round_state(
            history,
            23,
            target_hidden[:, -1:],
        )
        proposal = evidence.token_ids

        self.assertEqual(len(proposal), 3)
        self.assertEqual(tuple(row.window for row in evidence.horizons), (1, 2, 4))
        self.assertEqual(len(evidence.token_confidences), 3)
        self.assertTrue(
            all(0.0 <= value <= 0.999 for value in evidence.token_confidences)
        )
        self.assertEqual(
            evidence.select_window(
                request_window_ceiling=4,
                remaining_tokens=4,
                window_work_costs={1: 1.0, 2: 1.9, 4: 3.7},
            ).chosen_window,
            1,
        )
        self.assertTrue(provider.metrics().pending)
        provider.observe_verification(0, 1)
        provider.reconcile_prefix((*history, 23))
        self.assertEqual(provider._next_position, len(history))
        metrics = provider.metrics()
        self.assertFalse(metrics.pending)
        self.assertEqual(metrics.accepted_tokens, 0)
        self.assertEqual(metrics.rejected_tokens, 3)
        self.assertEqual(metrics.proposal_calls, 1)
        self.assertEqual(metrics.head_scans, 1)
        self.assertEqual(metrics.calibration_updates, 1)
        self.assertEqual(metrics.calibration_states, 1)
        self.assertGreater(metrics.linear_calls, 0)
        self.assertGreater(metrics.logical_weight_bytes, 0)

        next_history = (*history, 23)
        following = provider.propose_after_state(
            next_history,
            29,
            torch.randn((1, 1, config.dim)).to(torch.bfloat16),
        )
        self.assertEqual(len(following), 3)
        provider.reconcile_prefix((*next_history, 29))
        self.assertEqual(provider.metrics().accepted_tokens, 0)
        provider.close()
        self.assertTrue(provider.metrics().closed)

        restored = Qwen35MtpDraftProvider(
            config,
            _Pager(_tensors(config)),
            proposal_width=3,
            state_path=state_path,
        )
        self.assertEqual(restored.metrics().calibration_updates, 1)
        self.assertEqual(restored.metrics().calibration_states, 1)
        restored.close()


if __name__ == "__main__":
    unittest.main()
