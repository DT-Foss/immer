from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from immer.runtimes.qwen3_8.config import Qwen38Config
from immer.runtimes.qwen3_8.mtp_draft import (
    MTP_CONTROL_NAMES,
    MTP_MATRIX_NAMES,
    Qwen35MtpDraftError,
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
    def test_v1_calibration_with_v2_provider_migrates_to_v4_without_data_loss(
        self,
    ) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        state_path = Path(temporary.name) / "mtp.json"
        config = _config()
        state_path.write_text(
            json.dumps(
                {
                    "identity": {
                        "dim": config.dim,
                        "provider": "immer.qwen3.5-mtp-draft-provider/v2",
                        "q4_manifest_sha256": "a" * 64,
                        "vocab_size": config.vocab_size,
                    },
                    "previous_outcome": 1,
                    "rows": [
                        {
                            "alpha": 7,
                            "beta": 3,
                            "bucket": 6,
                            "position": 2,
                            "previous": 1,
                        }
                    ],
                    "schema": "immer.qwen3.5-mtp-markov-calibration/v1",
                    "updates": 8,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        provider = Qwen35MtpDraftProvider(
            config,
            _Pager(_tensors(config)),
            proposal_width=3,
            state_path=state_path,
        )

        self.assertEqual(provider._previous_outcome, 1)
        self.assertEqual(provider._reliability, {(False, 2, 6, 1): [7, 3]})
        self.assertEqual(provider.metrics().calibration_updates, 8)
        provider.close()
        migrated = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(
            migrated["identity"]["provider"],
            "immer.qwen3.5-mtp-draft-provider/v4",
        )
        self.assertEqual(
            migrated["previous_outcomes"],
            {"carried": -1, "cold": 1},
        )
        self.assertEqual(migrated["updates"], 8)
        self.assertEqual(
            migrated["schema"],
            "immer.qwen3.5-mtp-markov-calibration/v2",
        )
        self.assertEqual(
            migrated["updates_by_context"],
            {"carried": 0, "cold": 8},
        )
        self.assertEqual(
            migrated["rows"],
            [
                {
                    "alpha": 7,
                    "beta": 3,
                    "bucket": 6,
                    "carried": False,
                    "position": 2,
                    "previous": 1,
                }
            ],
        )

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
        provider.observe_virtual_verification(0, 1)
        provider.reconcile_prefix((*history, 23))
        self.assertEqual(provider._next_position, len(history))
        metrics = provider.metrics()
        self.assertFalse(metrics.pending)
        self.assertEqual(metrics.accepted_tokens, 0)
        self.assertEqual(metrics.proposed_tokens, 3)
        self.assertEqual(metrics.computed_proposal_tokens, 1)
        self.assertEqual(metrics.padded_proposal_tokens, 2)
        self.assertEqual(metrics.verified_proposal_tokens, 1)
        self.assertEqual(metrics.rejected_tokens, 1)
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

    def test_adaptive_high_confidence_computes_full_horizon_with_backoff(
        self,
    ) -> None:
        config = _config()
        pager = _Pager(_tensors(config))
        provider = Qwen35MtpDraftProvider(
            config,
            pager,
            proposal_width=15,
            eos_token_ids=(),
            head_block_rows=32,
        )
        history = (4, 7, 11, 19)
        target_hidden = torch.randn((1, len(history), config.dim)).to(torch.bfloat16)
        provider.begin_request_state(history, target_hidden)
        steps_before = provider.metrics().draft_steps
        provider._reliability[(False, 0, 6, -1)] = [9, 1]
        provider._reliability[(False, 0, 6, 1)] = [9, 1]
        head = (
            torch.tensor([[16.0, 0.0]], dtype=torch.bfloat16),
            torch.tensor([[5, 6]], dtype=torch.long),
        )

        with mock.patch.object(pager, "topk_logits", return_value=head) as scan:
            evidence = provider.propose_round_state(
                history,
                23,
                target_hidden[:, -1:],
            )

        self.assertEqual(scan.call_count, 15)
        self.assertEqual(len(evidence.token_ids), 15)
        self.assertEqual(evidence.token_ids, (5,) * 15)
        self.assertEqual(evidence.token_confidences, (0.9,) * 15)
        self.assertEqual(provider.metrics().draft_steps, steps_before + 16)
        self.assertEqual(len(provider._pending_states), 16)
        committed_state = provider._pending_states[-1]
        self.assertIsNot(provider._pending_states[0], committed_state)

        provider.observe_verification(15, 15)
        provider.reconcile_prefix((*history, 23, *((5,) * 15)))
        metrics = provider.metrics()
        self.assertIs(provider._committed_state, committed_state)
        self.assertEqual(metrics.head_scans, 15)
        self.assertEqual(metrics.proposed_tokens, 15)
        self.assertEqual(metrics.computed_proposal_tokens, 15)
        self.assertEqual(metrics.padded_proposal_tokens, 0)
        self.assertEqual(metrics.verified_proposal_tokens, 15)
        self.assertEqual(metrics.accepted_tokens, 15)
        self.assertEqual(metrics.rejected_tokens, 0)
        provider.close()

    def test_mtp_carry_plus_suffix_matches_full_history_bootstrap(self) -> None:
        config = _config()
        tensors = _tensors(config)
        history = (4, 7, 11, 19, 23)
        prefix = history[:3]
        hidden = torch.randn(
            (1, len(history), config.dim),
            generator=torch.Generator().manual_seed(77),
        ).to(torch.bfloat16)

        full = Qwen35MtpDraftProvider(
            config,
            _Pager(tensors),
            proposal_width=3,
        )
        full.begin_request_state(history, hidden)

        source = Qwen35MtpDraftProvider(
            config,
            _Pager(tensors),
            proposal_width=3,
        )
        source.begin_request_state(prefix, hidden[:, : len(prefix)])
        carry = source.export_carry(prefix)
        source.close()

        restored = Qwen35MtpDraftProvider(
            config,
            _Pager(tensors),
            proposal_width=3,
            initial_carry=carry,
        )
        restored.begin_request_state(history, hidden[:, len(prefix) :])

        assert full._committed_state is not None
        assert restored._committed_state is not None
        self.assertTrue(
            torch.equal(full._committed_state.key, restored._committed_state.key)
        )
        self.assertTrue(
            torch.equal(full._committed_state.value, restored._committed_state.value)
        )
        self.assertTrue(
            torch.equal(full._last_target_hidden, restored._last_target_hidden)
        )
        full_proposal = full.propose_round_state(history, 29, hidden[:, -1:])
        restored_proposal = restored.propose_round_state(
            history,
            29,
            hidden[:, -1:],
        )
        self.assertEqual(restored_proposal.token_ids, full_proposal.token_ids)
        self.assertTrue(
            all(
                carried <= cold
                for carried, cold in zip(
                    restored_proposal.token_confidences,
                    full_proposal.token_confidences,
                    strict=True,
                )
            )
        )
        self.assertTrue(restored.metrics().carried_context)
        self.assertFalse(full.metrics().carried_context)
        full.close()
        restored.close()

    def test_mtp_carry_trims_uncommitted_terminal_proposal_state(self) -> None:
        config = _config()
        provider = Qwen35MtpDraftProvider(
            config,
            _Pager(_tensors(config)),
            proposal_width=3,
        )
        prefix = (4, 7, 11, 19)
        hidden = torch.randn((1, len(prefix), config.dim)).to(torch.bfloat16)
        provider.begin_request_state(prefix, hidden)
        proposal = provider.propose_round_state(prefix, 23, hidden[:, -1:])

        with self.assertRaisesRegex(Qwen35MtpDraftError, "pending proposal"):
            provider.export_carry(prefix)

        provider.observe_verification(1, 2)
        provider.reconcile_prefix((*prefix, 23, proposal.token_ids[0]))
        carry = provider.export_carry(prefix)

        self.assertEqual(carry.history, prefix)
        self.assertEqual(carry.next_position, len(prefix) - 1)
        assert carry.state is not None
        self.assertEqual(carry.state.length, len(prefix) - 1)
        self.assertTrue(torch.equal(carry.last_target_hidden, hidden[:, -1:]))
        provider.close()

    def test_carried_reliability_is_separate_and_starts_at_beta_prior(self) -> None:
        config = _config()
        tensors = _tensors(config)
        prefix = (4, 7, 11)
        hidden = torch.randn((1, len(prefix), config.dim)).to(torch.bfloat16)
        source = Qwen35MtpDraftProvider(
            config,
            _Pager(tensors),
            proposal_width=3,
        )
        source.begin_request_state(prefix, hidden)
        carry = source.export_carry(prefix)
        source.close()
        pager = _Pager(tensors)
        provider = Qwen35MtpDraftProvider(
            config,
            pager,
            proposal_width=3,
            initial_carry=carry,
        )
        provider._reliability[(False, 0, 6, -1)] = [99, 1]
        head = (
            torch.tensor([[16.0, 0.0]], dtype=torch.bfloat16),
            torch.tensor([[5, 6]], dtype=torch.long),
        )

        with mock.patch.object(pager, "topk_logits", return_value=head):
            _token, cold_start, bucket = provider._scan(
                torch.zeros((1, 1, config.dim), dtype=torch.bfloat16),
                proposal_index=0,
            )
            provider._reliability[(True, 0, bucket, -1)] = [4, 1]
            _token, matured, _bucket = provider._scan(
                torch.zeros((1, 1, config.dim), dtype=torch.bfloat16),
                proposal_index=0,
            )

        self.assertAlmostEqual(cold_start, 0.5)
        self.assertAlmostEqual(matured, 4 / 5)
        metrics = provider.metrics()
        self.assertTrue(metrics.carried_context)
        self.assertEqual(metrics.cold_calibration_states, 1)
        self.assertEqual(metrics.carried_calibration_states, 1)
        provider.close()

    def test_context_split_calibration_persists_across_carry_restart(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        state_path = Path(temporary.name) / "mtp.json"
        config = _config()
        tensors = _tensors(config)
        prefix = (4, 7, 11)
        hidden = torch.randn((1, len(prefix), config.dim)).to(torch.bfloat16)
        source = Qwen35MtpDraftProvider(
            config,
            _Pager(tensors),
            proposal_width=3,
            state_path=state_path,
        )
        source.begin_request_state(prefix, hidden)
        carry = source.export_carry(prefix)
        source._reliability = {
            (False, 0, 6, -1): [99, 1],
            (True, 0, 6, -1): [4, 1],
        }
        source._previous_outcomes = {False: 1, True: 0}
        source._calibration_updates = 12
        source._calibration_updates_by_context = {False: 8, True: 4}
        source.close()

        restored = Qwen35MtpDraftProvider(
            config,
            _Pager(tensors),
            proposal_width=3,
            state_path=state_path,
            initial_carry=carry,
        )

        self.assertEqual(restored._previous_outcome, 0)
        self.assertEqual(restored._reliability[(False, 0, 6, -1)], [99, 1])
        self.assertEqual(restored._reliability[(True, 0, 6, -1)], [4, 1])
        self.assertEqual(restored._reliability[(True, 0, -1, -1)], [4, 1])
        metrics = restored.metrics()
        self.assertEqual(metrics.cold_calibration_updates, 8)
        self.assertEqual(metrics.carried_calibration_updates, 4)
        self.assertEqual(metrics.cold_calibration_states, 1)
        self.assertEqual(metrics.carried_calibration_states, 2)
        restored.close()

    def test_carried_bucket_aggregates_recover_existing_cross_bucket_hits(self) -> None:
        rows = {
            (True, 0, 4, -1): [2, 1],
            (True, 0, 5, 1): [2, 1],
            (True, 0, 6, 1): [3, 1],
        }

        Qwen35MtpDraftProvider._ensure_carried_aggregates(rows)

        self.assertEqual(rows[(True, 0, -1, -1)], [2, 1])
        self.assertEqual(rows[(True, 0, -1, 1)], [4, 1])

    def test_adaptive_exact_deeper_posterior_stops_and_pads(self) -> None:
        config = _config()
        pager = _Pager(_tensors(config))
        provider = Qwen35MtpDraftProvider(
            config,
            pager,
            proposal_width=7,
            eos_token_ids=(),
            head_block_rows=32,
        )
        history = (4, 7, 11, 19)
        target_hidden = torch.randn((1, len(history), config.dim)).to(torch.bfloat16)
        provider.begin_request_state(history, target_hidden)
        steps_before = provider.metrics().draft_steps
        provider._reliability[(False, 0, 6, -1)] = [9, 1]
        provider._reliability[(False, 0, 6, 1)] = [9, 1]
        provider._reliability[(False, 1, 6, 1)] = [1, 1]
        head = (
            torch.tensor([[16.0, 0.0]], dtype=torch.bfloat16),
            torch.tensor([[5, 6]], dtype=torch.long),
        )

        with mock.patch.object(pager, "topk_logits", return_value=head) as scan:
            evidence = provider.propose_round_state(
                history,
                23,
                target_hidden[:, -1:],
            )

        self.assertEqual(scan.call_count, 2)
        self.assertEqual(evidence.token_ids, (5,) * 7)
        self.assertEqual(evidence.token_confidences[:2], (0.9, 0.5))
        self.assertEqual(evidence.token_confidences[2:], (0.0,) * 5)
        self.assertEqual(provider.metrics().draft_steps, steps_before + 2)
        self.assertEqual(provider.metrics().computed_proposal_tokens, 2)
        self.assertEqual(provider.metrics().padded_proposal_tokens, 5)
        self.assertEqual(len(provider._pending_states), 8)
        provider.observe_verification(1, 1)
        provider.reconcile_prefix((*history, 23, 5))
        provider.close()

    def test_adaptive_eos_stops_without_autoregressive_tail(self) -> None:
        config = _config()
        pager = _Pager(_tensors(config))
        provider = Qwen35MtpDraftProvider(
            config,
            pager,
            proposal_width=7,
            eos_token_ids=(127,),
            head_block_rows=32,
        )
        history = (4, 7, 11, 19)
        target_hidden = torch.randn((1, len(history), config.dim)).to(torch.bfloat16)
        provider.begin_request_state(history, target_hidden)
        steps_before = provider.metrics().draft_steps
        provider._reliability[(False, 0, 6, -1)] = [9, 1]
        head = (
            torch.tensor([[16.0, 0.0]], dtype=torch.bfloat16),
            torch.tensor([[127, 6]], dtype=torch.long),
        )

        with mock.patch.object(pager, "topk_logits", return_value=head) as scan:
            evidence = provider.propose_round_state(
                history,
                23,
                target_hidden[:, -1:],
            )

        self.assertEqual(scan.call_count, 1)
        self.assertEqual(evidence.token_ids, (127,) * 7)
        self.assertEqual(evidence.token_confidences, (0.9,) * 7)
        self.assertEqual(provider.metrics().draft_steps, steps_before + 1)
        self.assertEqual(provider.metrics().computed_proposal_tokens, 1)
        self.assertEqual(provider.metrics().padded_proposal_tokens, 6)
        provider.observe_verification(0, 1)
        provider.reconcile_prefix((*history, 23))
        provider.close()

    def test_advance_confirmed_prefix_appends_exact_shifted_pairs(self) -> None:
        config = _config()
        tensors = _tensors(config)
        pager = _Pager(tensors)
        provider = Qwen35MtpDraftProvider(
            config,
            pager,
            proposal_width=3,
            eos_token_ids=(),
            head_block_rows=32,
        )
        history = (4, 7, 11, 19)
        target_hidden = torch.randn((1, len(history), config.dim)).to(torch.bfloat16)
        provider.begin_request_state(history, target_hidden)
        previous_state = provider._committed_state
        extension = (23, 29, 31)
        committed_hidden = torch.randn((1, len(extension), config.dim)).to(
            torch.bfloat16
        )
        previous_hidden = target_hidden[:, -1:].clone()
        previous_snapshot = previous_hidden.clone()
        committed_snapshot = committed_hidden.clone()

        with mock.patch.object(provider, "_step", wraps=provider._step) as step:
            provider.advance_confirmed_prefix_state(
                (*history, *extension),
                previous_hidden,
                committed_hidden,
            )

        self.assertEqual(step.call_count, 1)
        args = step.call_args
        self.assertEqual(args.args[0], extension)
        self.assertTrue(
            torch.equal(
                args.args[1],
                torch.cat((previous_snapshot, committed_snapshot[:, :-1]), dim=1),
            )
        )
        self.assertIs(args.kwargs["state"], previous_state)
        self.assertEqual(args.kwargs["start_pos"], len(history) - 1)
        self.assertTrue(torch.equal(previous_hidden, previous_snapshot))
        self.assertTrue(torch.equal(committed_hidden, committed_snapshot))
        self.assertEqual(provider._committed_history, (*history, *extension))
        self.assertEqual(provider._next_position, len(history) + len(extension) - 1)
        metrics = provider.metrics()
        self.assertEqual(metrics.schema, "immer.qwen3.5-mtp-draft-provider/v4")
        self.assertEqual(metrics.advance_calls, 1)
        self.assertEqual(metrics.advanced_tokens, len(extension))
        self.assertFalse(metrics.pending)

        final_extension = (37,)
        final_hidden = torch.randn((1, 1, config.dim)).to(torch.bfloat16)
        complete_history = (*history, *extension, *final_extension)
        provider.advance_confirmed_prefix_state(
            complete_history,
            committed_hidden[:, -1:],
            final_hidden,
        )
        reference = Qwen35MtpDraftProvider(
            config,
            _Pager(tensors),
            proposal_width=3,
            eos_token_ids=(),
            head_block_rows=32,
        )
        reference.begin_request_state(
            complete_history,
            torch.cat((target_hidden, committed_hidden, final_hidden), dim=1),
        )
        self.assertTrue(
            torch.equal(provider._committed_state.key, reference._committed_state.key)
        )
        self.assertTrue(
            torch.equal(
                provider._committed_state.value,
                reference._committed_state.value,
            )
        )
        metrics = provider.metrics()
        self.assertEqual(metrics.advance_calls, 2)
        self.assertEqual(metrics.advanced_tokens, len(extension) + 1)

        proposal = provider.propose_after_state(
            complete_history,
            41,
            final_hidden,
        )
        self.assertEqual(len(proposal), 3)
        provider.reconcile_prefix((*complete_history, 41))
        reference.close()
        provider.close()

    def test_advance_confirmed_prefix_rejects_noncontiguous_or_pending_state(
        self,
    ) -> None:
        config = _config()
        pager = _Pager(_tensors(config))
        provider = Qwen35MtpDraftProvider(
            config,
            pager,
            proposal_width=3,
            eos_token_ids=(),
            head_block_rows=32,
        )
        history = (4, 7, 11, 19)
        hidden = torch.randn((1, len(history), config.dim)).to(torch.bfloat16)
        previous = hidden[:, -1:]
        fragment = torch.randn((1, 1, config.dim)).to(torch.bfloat16)

        with self.assertRaisesRegex(Qwen35MtpDraftError, "not initialized"):
            provider.advance_confirmed_prefix_state(
                (*history, 23),
                previous,
                fragment,
            )
        provider.begin_request_state(history, hidden)
        with self.assertRaisesRegex(Qwen35MtpDraftError, "contiguous extension"):
            provider.advance_confirmed_prefix_state(
                history,
                previous,
                torch.zeros((1, 0, config.dim), dtype=torch.bfloat16),
            )
        with self.assertRaisesRegex(Qwen35MtpDraftError, "contiguous extension"):
            provider.advance_confirmed_prefix_state(
                (4, 7, 12, 19, 23),
                previous,
                fragment,
            )
        with self.assertRaisesRegex(ValueError, "previous target hidden"):
            provider.advance_confirmed_prefix_state(
                (*history, 23),
                torch.zeros((1, 2, config.dim), dtype=torch.bfloat16),
                fragment,
            )
        with self.assertRaisesRegex(ValueError, "committed target hidden"):
            provider.advance_confirmed_prefix_state(
                (*history, 23),
                previous,
                torch.zeros((1, 2, config.dim), dtype=torch.bfloat16),
            )

        provider.propose_after_state(history, 23, previous)
        with self.assertRaisesRegex(Qwen35MtpDraftError, "pending proposal"):
            provider.advance_confirmed_prefix_state(
                (*history, 29),
                previous,
                fragment,
            )
        provider.reconcile_prefix((*history, 23))
        provider.close()


if __name__ == "__main__":
    unittest.main()
