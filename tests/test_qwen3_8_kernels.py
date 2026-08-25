from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import math
import unittest

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - neural extra is optional
    torch = None
    F = None


@unittest.skipIf(torch is None, "Qwen3.8 kernels require the neural extra")
class Qwen38KernelTests(unittest.TestCase):
    def test_delta_centered_rmsnorm_and_swiglu_match_direct_math(self) -> None:
        from immer.runtimes.qwen3_8.kernels import rms_norm, rms_norm_gated, swiglu

        x = torch.tensor([[[1.0, -2.0, 0.5], [3.0, 4.0, -1.0]]])
        delta = torch.tensor([0.25, -0.5, 1.0])
        expected = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + 1e-6
        )
        expected = expected * (1.0 + delta)
        torch.testing.assert_close(rms_norm(x, delta), expected)

        gate = torch.tensor([[[-2.0, 0.0, 2.0], [1.0, -1.0, 0.5]]])
        up = torch.tensor([[[0.5, 2.0, -3.0], [4.0, -0.5, 1.0]]])
        torch.testing.assert_close(swiglu(gate, up), F.silu(gate) * up)

        # DeltaNet's output norm is deliberately conventional, not delta-centered.
        conventional_weight = torch.tensor([0.0, 1.0, 2.0])
        expected_gated = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + 1e-6
        )
        expected_gated = expected_gated * conventional_weight * F.silu(gate.float())
        torch.testing.assert_close(
            rms_norm_gated(x, conventional_weight, gate), expected_gated
        )

    def test_query_gate_split_is_per_head_not_global(self) -> None:
        from immer.runtimes.qwen3_8.kernels import split_query_gate

        projection = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
        query, gate = split_query_gate(projection, 2, 2)
        torch.testing.assert_close(
            query,
            torch.tensor([[[[0.0, 1.0], [4.0, 5.0]]]]),
        )
        torch.testing.assert_close(
            gate,
            torch.tensor([[[[2.0, 3.0], [6.0, 7.0]]]]),
        )

    def test_interleaved_mrope_selects_temporal_height_width_frequencies(self) -> None:
        from immer.runtimes.qwen3_8.kernels import rope_cos_sin

        # Six half-dimension frequencies become T,H,W,T,H,W with (2,2,2).
        positions = torch.tensor([[[1]], [[2]], [[3]]], dtype=torch.long)
        cos, sin = rope_cos_sin(
            positions,
            12,
            theta=100.0,
            mrope_section=(2, 2, 2),
            mrope_interleaved=True,
        )
        inverse = 1.0 / (100.0 ** (torch.arange(0, 12, 2, dtype=torch.float32) / 12))
        selected_positions = torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        angles = selected_positions * inverse
        expected_angles = torch.cat((angles, angles)).reshape(1, 1, 12)
        torch.testing.assert_close(cos, expected_angles.cos())
        torch.testing.assert_close(sin, expected_angles.sin())

    def test_partial_rotate_half_rope_leaves_tail_untouched(self) -> None:
        from immer.runtimes.qwen3_8.kernels import apply_rotary_pos_emb

        query = torch.tensor([[[[1.0, 2.0, 30.0, 40.0]]]])
        key = torch.tensor([[[[5.0, 6.0, 70.0, 80.0]]]])
        angle = torch.tensor([[[math.pi / 2, math.pi / 2]]])
        rotated_q, rotated_k = apply_rotary_pos_emb(
            query, key, angle.cos(), angle.sin()
        )
        torch.testing.assert_close(
            rotated_q, torch.tensor([[[[-2.0, 1.0, 30.0, 40.0]]]]), atol=1e-6, rtol=0
        )
        torch.testing.assert_close(
            rotated_k, torch.tensor([[[[-6.0, 5.0, 70.0, 80.0]]]]), atol=1e-6, rtol=0
        )

    @staticmethod
    def _full_attention_reference(
        query_gate,
        key_projection,
        value_projection,
        q_norm_weight,
        k_norm_weight,
        *,
        heads,
        kv_heads,
        head_dim,
        rotary_dim,
        theta,
    ):
        batch, sequence, _ = query_gate.shape
        reshaped = query_gate.reshape(batch, sequence, heads, head_dim * 2)
        query, gate = torch.chunk(reshaped, 2, dim=-1)
        key = key_projection.reshape(batch, sequence, kv_heads, head_dim)
        value = value_projection.reshape(batch, sequence, kv_heads, head_dim)

        def norm(value, weight):
            value32 = value.float()
            return (
                value32
                * torch.rsqrt(value32.square().mean(-1, keepdim=True) + 1e-6)
                * (1.0 + weight.float())
            ).to(value.dtype)

        query = norm(query, q_norm_weight).transpose(1, 2)
        key = norm(key, k_norm_weight).transpose(1, 2)
        value = value.transpose(1, 2)
        positions = torch.arange(sequence, dtype=torch.float32)
        inverse = 1.0 / (
            theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        angles = positions[:, None] * inverse[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        cosine = angles.cos().reshape(1, 1, sequence, rotary_dim)
        sine = angles.sin().reshape(1, 1, sequence, rotary_dim)

        def rotate_half(value):
            midpoint = value.shape[-1] // 2
            return torch.cat((-value[..., midpoint:], value[..., :midpoint]), -1)

        q_rot, q_tail = query[..., :rotary_dim], query[..., rotary_dim:]
        k_rot, k_tail = key[..., :rotary_dim], key[..., rotary_dim:]
        query = torch.cat((q_rot * cosine + rotate_half(q_rot) * sine, q_tail), dim=-1)
        key = torch.cat((k_rot * cosine + rotate_half(k_rot) * sine, k_tail), dim=-1)
        key = key.repeat_interleave(heads // kv_heads, dim=1)
        value = value.repeat_interleave(heads // kv_heads, dim=1)
        scores = query @ key.transpose(2, 3) * (head_dim**-0.5)
        causal = torch.ones(sequence, sequence, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal[None, None], torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, -1, dtype=torch.float32).to(query.dtype)
        output = (probabilities @ value).transpose(1, 2).contiguous()
        output = output * torch.sigmoid(gate)
        return output.reshape(batch, sequence, heads * head_dim), key, value

    def test_full_attention_matches_direct_gqa_gate_and_causal_reference(self) -> None:
        from immer.runtimes.qwen3_8.kernels import full_attention_core

        torch.manual_seed(103)
        query_gate = torch.randn(2, 5, 2 * 2 * 4)
        key = torch.randn(2, 5, 1 * 4)
        value = torch.randn(2, 5, 1 * 4)
        q_norm = torch.randn(4) * 0.1
        k_norm = torch.randn(4) * 0.1
        expected, _, _ = self._full_attention_reference(
            query_gate,
            key,
            value,
            q_norm,
            k_norm,
            heads=2,
            kv_heads=1,
            head_dim=4,
            rotary_dim=2,
            theta=100.0,
        )
        actual, state = full_attention_core(
            query_gate,
            key,
            value,
            q_norm_weight=q_norm,
            k_norm_weight=k_norm,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            rotary_dim=2,
            rope_theta=100.0,
        )
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        self.assertEqual(tuple(state.key.shape), (2, 1, 5, 4))
        self.assertEqual(tuple(state.value.shape), (2, 1, 5, 4))

    def test_full_attention_prefill_then_continuation_equals_one_shot(self) -> None:
        from immer.runtimes.qwen3_8.kernels import full_attention_core

        torch.manual_seed(91)
        query_gate = torch.randn(2, 6, 16)
        key = torch.randn(2, 6, 4)
        value = torch.randn(2, 6, 4)
        q_norm = torch.randn(4) * 0.05
        k_norm = torch.randn(4) * 0.05
        kwargs = dict(
            q_norm_weight=q_norm,
            k_norm_weight=k_norm,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            rotary_dim=2,
            rope_theta=1_000.0,
        )
        whole, whole_state = full_attention_core(query_gate, key, value, **kwargs)
        prefix, prefix_state = full_attention_core(
            query_gate[:, :3], key[:, :3], value[:, :3], **kwargs
        )
        suffix, split_state = full_attention_core(
            query_gate[:, 3:],
            key[:, 3:],
            value[:, 3:],
            state=prefix_state,
            **kwargs,
        )
        torch.testing.assert_close(
            torch.cat((prefix, suffix), dim=1), whole, atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(split_state.key, whole_state.key)
        torch.testing.assert_close(split_state.value, whole_state.value)

        # Changing future K/V cannot affect earlier causal outputs.
        changed_key = key.clone()
        changed_value = value.clone()
        changed_key[:, 3:] += 10_000.0
        changed_value[:, 3:] -= 10_000.0
        changed, _ = full_attention_core(
            query_gate, changed_key, changed_value, **kwargs
        )
        torch.testing.assert_close(changed[:, :3], whole[:, :3])

    def test_native_head_crsa_selects_one_head_per_gqa_group_and_seals_evidence(
        self,
    ) -> None:
        from immer.runtimes.qwen3_8.native_crsa import (
            NATIVE_HEAD_CRSA_EVIDENCE_SCHEMA,
            NATIVE_HEAD_CRSA_ROW_SUM_TOLERANCE,
            Qwen38NativeHeadCrsa,
        )

        torch.manual_seed(79)
        logits = torch.randn(2, 24, 6, 6, dtype=torch.float64)
        allowed = torch.ones(6, 6, dtype=torch.bool).tril()
        allowed = allowed & torch.tensor([True, True, True, True, True, False])
        base = torch.softmax(logits.masked_fill(~allowed, -torch.inf), dim=-1)
        base = base.masked_fill(~allowed, 0.0)
        intervention = Qwen38NativeHeadCrsa(alpha=0.2)

        routed, usage, evidence = intervention.route(
            logits,
            base,
            query_start=0,
            allowed=allowed,
            prior_log_usage=None,
        )
        free = tuple(
            head for head in range(24) if head not in intervention.head_indices
        )
        self.assertTrue(torch.equal(routed[:, free], base[:, free]))
        self.assertTrue(
            any(
                not torch.equal(routed[:, head], base[:, head])
                for head in intervention.head_indices
            )
        )
        self.assertEqual(tuple(usage.shape), (2, 4, 6))
        self.assertEqual(evidence.schema, NATIVE_HEAD_CRSA_EVIDENCE_SCHEMA)
        self.assertEqual(evidence.selected_query_heads, (2, 8, 14, 20))
        self.assertEqual(evidence.selected_kv_heads, (0, 1, 2, 3))
        self.assertEqual(evidence.alpha_per_head, (0.2, 0.2, 0.2, 0.2))
        self.assertEqual(evidence.free_heads, free)
        self.assertEqual(evidence.free_head_max_abs_error, 0.0)
        self.assertEqual(evidence.future_weight_max_abs, 0.0)
        self.assertFalse(evidence.identity)

        with self.assertRaises(FrozenInstanceError):
            evidence.layer = 3  # type: ignore[misc]
        with self.assertRaisesRegex(ValueError, "strictly causal"):
            replace(evidence, future_weight_max_abs=0.1)
        with self.assertRaisesRegex(ValueError, "bit-exact"):
            replace(evidence, free_head_max_abs_error=0.1)
        with self.assertRaisesRegex(ValueError, "no greater"):
            replace(evidence, row_sum_max_error=0.019)
        with self.assertRaisesRegex(ValueError, "no greater"):
            replace(
                evidence,
                row_sum_max_error=NATIVE_HEAD_CRSA_ROW_SUM_TOLERANCE + 1e-7,
            )
        payload = evidence.to_dict()
        payload["selected_query_heads"][0] = 0
        self.assertEqual(evidence.selected_query_heads, (2, 8, 14, 20))

        identity = Qwen38NativeHeadCrsa(alpha=0.0)
        identical, identity_usage, identity_evidence = identity.route(
            logits,
            base,
            query_start=0,
            allowed=allowed,
            prior_log_usage=None,
        )
        self.assertIs(identical, base)
        self.assertEqual(identical.data_ptr(), base.data_ptr())
        self.assertIsNone(identity_usage)
        self.assertTrue(identity_evidence.identity)

        bf16_logits = logits.to(torch.bfloat16)
        bf16_base = torch.softmax(
            bf16_logits.masked_fill(~allowed, torch.finfo(torch.bfloat16).min),
            dim=-1,
            dtype=torch.float32,
        ).to(torch.bfloat16)
        bf16_base = bf16_base.masked_fill(~allowed, 0.0)
        _, _, bf16_evidence = Qwen38NativeHeadCrsa(alpha=0.01).route(
            bf16_logits,
            bf16_base,
            query_start=0,
            allowed=allowed,
            prior_log_usage=None,
        )
        self.assertLessEqual(
            bf16_evidence.row_sum_max_error,
            NATIVE_HEAD_CRSA_ROW_SUM_TOLERANCE,
        )

    def test_native_full_attention_streaming_matches_one_shot_and_alpha_zero(
        self,
    ) -> None:
        from immer.runtimes.qwen3_8.kernels import full_attention_core
        from immer.runtimes.qwen3_8.native_crsa import Qwen38NativeHeadCrsa

        torch.manual_seed(83)
        batch, sequence, heads, kv_heads, width = 2, 6, 24, 4, 4
        query_gate = torch.randn(batch, sequence, 2 * heads * width)
        key = torch.randn(batch, sequence, kv_heads * width)
        value = torch.randn(batch, sequence, kv_heads * width)
        q_norm = torch.randn(width) * 0.05
        k_norm = torch.randn(width) * 0.05
        common = dict(
            q_norm_weight=q_norm,
            k_norm_weight=k_norm,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=width,
            rotary_dim=2,
            rope_theta=1_000.0,
        )
        baseline, baseline_state = full_attention_core(query_gate, key, value, **common)
        identity_rows = []
        identity, identity_state = full_attention_core(
            query_gate,
            key,
            value,
            native_head_crsa=Qwen38NativeHeadCrsa(alpha=0.0),
            native_head_crsa_observer=identity_rows.append,
            **common,
        )
        self.assertTrue(torch.equal(identity, baseline))
        self.assertTrue(torch.equal(identity_state.key, baseline_state.key))
        self.assertTrue(torch.equal(identity_state.value, baseline_state.value))
        self.assertIsNone(identity_state.crsa_log_usage)
        self.assertTrue(identity_rows[0].identity)

        intervention = Qwen38NativeHeadCrsa(alpha=0.15)
        observed = []
        whole, whole_state = full_attention_core(
            query_gate,
            key,
            value,
            native_head_crsa=intervention,
            native_head_crsa_observer=observed.append,
            **common,
        )
        prefix, prefix_state = full_attention_core(
            query_gate[:, :3],
            key[:, :3],
            value[:, :3],
            native_head_crsa=intervention,
            native_head_crsa_observer=observed.append,
            **common,
        )
        suffix, split_state = full_attention_core(
            query_gate[:, 3:],
            key[:, 3:],
            value[:, 3:],
            state=prefix_state,
            native_head_crsa=intervention,
            native_head_crsa_observer=observed.append,
            **common,
        )
        torch.testing.assert_close(
            torch.cat((prefix, suffix), dim=1), whole, rtol=2e-6, atol=2e-6
        )
        torch.testing.assert_close(split_state.key, whole_state.key)
        torch.testing.assert_close(split_state.value, whole_state.value)
        torch.testing.assert_close(
            split_state.crsa_log_usage,
            whole_state.crsa_log_usage,
            rtol=2e-6,
            atol=2e-6,
        )

        token_outputs = []
        token_state = None
        for position in range(sequence):
            output, token_state = full_attention_core(
                query_gate[:, position : position + 1],
                key[:, position : position + 1],
                value[:, position : position + 1],
                state=token_state,
                native_head_crsa=intervention,
                **common,
            )
            token_outputs.append(output)
        torch.testing.assert_close(
            torch.cat(token_outputs, dim=1), whole, rtol=2e-6, atol=2e-6
        )
        torch.testing.assert_close(
            token_state.crsa_log_usage,
            whole_state.crsa_log_usage,
            rtol=2e-6,
            atol=2e-6,
        )
        self.assertEqual(
            [(row.query_start, row.history_length_after) for row in observed],
            [(0, 6), (0, 3), (3, 6)],
        )

        padded_rows = []
        key_valid = torch.ones(batch, sequence, dtype=torch.bool)
        key_valid[1, -1] = False
        _, padded_state = full_attention_core(
            query_gate,
            key,
            value,
            attention_mask=key_valid,
            native_head_crsa=intervention,
            native_head_crsa_observer=padded_rows.append,
            **common,
        )
        self.assertTrue(torch.isneginf(padded_state.crsa_log_usage[1, :, -1]).all())
        self.assertEqual(padded_rows[0].future_weight_max_abs, 0.0)

    def test_full_attention_fork_prefill_is_bit_exact_and_shares_only_kv(
        self,
    ) -> None:
        from immer.runtimes.qwen3_8.kernels import (
            full_attention_core,
            full_attention_fork_core,
        )
        from immer.runtimes.qwen3_8.native_crsa import Qwen38NativeHeadCrsa

        torch.manual_seed(211)
        batch, sequence, heads, kv_heads, width = 2, 5, 24, 4, 4
        query_gate = torch.randn(
            batch, sequence, 2 * heads * width, dtype=torch.bfloat16
        )
        key = torch.randn(batch, sequence, kv_heads * width, dtype=torch.bfloat16)
        value = torch.randn(batch, sequence, kv_heads * width, dtype=torch.bfloat16)
        q_norm = torch.randn(width, dtype=torch.bfloat16) * 0.05
        k_norm = torch.randn(width, dtype=torch.bfloat16) * 0.05
        validity = torch.ones(batch, sequence, dtype=torch.bool)
        validity[1, -1] = False
        intervention = Qwen38NativeHeadCrsa(alpha=0.01)
        common = dict(
            q_norm_weight=q_norm,
            k_norm_weight=k_norm,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=width,
            attention_mask=validity,
            rotary_dim=2,
            rope_theta=1_000.0,
        )
        original_tensors = tuple(
            tensor.clone() for tensor in (query_gate, key, value, q_norm, k_norm)
        )
        expected_off, expected_off_state = full_attention_core(
            query_gate,
            key,
            value,
            **common,
        )
        independent_evidence = []
        expected_native, expected_native_state = full_attention_core(
            query_gate,
            key,
            value,
            native_head_crsa=intervention,
            native_head_crsa_observer=independent_evidence.append,
            **common,
        )
        observed = []
        (
            actual_off,
            actual_off_state,
            actual_native,
            actual_native_state,
            evidence,
        ) = full_attention_fork_core(
            query_gate,
            key,
            value,
            native_head_crsa=intervention,
            native_head_crsa_observer=observed.append,
            **common,
        )

        self.assertTrue(torch.equal(actual_off, expected_off))
        self.assertTrue(torch.equal(actual_native, expected_native))
        self.assertTrue(torch.equal(actual_off_state.key, expected_off_state.key))
        self.assertTrue(torch.equal(actual_off_state.value, expected_off_state.value))
        self.assertTrue(torch.equal(actual_native_state.key, expected_native_state.key))
        self.assertTrue(
            torch.equal(actual_native_state.value, expected_native_state.value)
        )
        self.assertTrue(
            torch.equal(
                actual_native_state.crsa_log_usage,
                expected_native_state.crsa_log_usage,
            )
        )
        self.assertIs(actual_off_state.key, actual_native_state.key)
        self.assertIs(actual_off_state.value, actual_native_state.value)
        self.assertIsNone(actual_off_state.crsa_log_usage)
        self.assertIsNotNone(actual_native_state.crsa_log_usage)
        self.assertEqual(observed, [evidence])
        self.assertEqual(evidence, independent_evidence[0])
        self.assertEqual(evidence.future_weight_max_abs, 0.0)
        self.assertEqual(evidence.free_head_max_abs_error, 0.0)
        for actual, original in zip(
            (query_gate, key, value, q_norm, k_norm),
            original_tensors,
            strict=True,
        ):
            self.assertTrue(torch.equal(actual, original))

    def test_full_attention_fork_decode_matches_two_independent_arms_exactly(
        self,
    ) -> None:
        from immer.runtimes.qwen3_8.kernels import (
            full_attention_core,
            full_attention_fork_core,
        )
        from immer.runtimes.qwen3_8.native_crsa import Qwen38NativeHeadCrsa

        torch.manual_seed(223)
        batch, sequence, heads, kv_heads, width = 1, 5, 24, 4, 4
        query_gate = torch.randn(batch, sequence, 2 * heads * width)
        key = torch.randn(batch, sequence, kv_heads * width)
        value = torch.randn(batch, sequence, kv_heads * width)
        q_norm = torch.randn(width) * 0.05
        k_norm = torch.randn(width) * 0.05
        intervention = Qwen38NativeHeadCrsa(alpha=0.07)
        common = dict(
            q_norm_weight=q_norm,
            k_norm_weight=k_norm,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=width,
            rotary_dim=2,
            rope_theta=1_000.0,
        )
        expected_off_prefix, expected_off_state = full_attention_core(
            query_gate[:, :4],
            key[:, :4],
            value[:, :4],
            **common,
        )
        expected_native_prefix, expected_native_state = full_attention_core(
            query_gate[:, :4],
            key[:, :4],
            value[:, :4],
            native_head_crsa=intervention,
            **common,
        )
        (
            actual_off_prefix,
            actual_off_state,
            actual_native_prefix,
            actual_native_state,
            _,
        ) = full_attention_fork_core(
            query_gate[:, :4],
            key[:, :4],
            value[:, :4],
            native_head_crsa=intervention,
            **common,
        )
        self.assertTrue(torch.equal(actual_off_prefix, expected_off_prefix))
        self.assertTrue(torch.equal(actual_native_prefix, expected_native_prefix))

        saved_state = (
            actual_off_state.key.clone(),
            actual_off_state.value.clone(),
            actual_native_state.crsa_log_usage.clone(),
        )
        expected_off_decode, expected_off_next = full_attention_core(
            query_gate[:, 4:],
            key[:, 4:],
            value[:, 4:],
            state=expected_off_state,
            **common,
        )
        expected_native_decode, expected_native_next = full_attention_core(
            query_gate[:, 4:],
            key[:, 4:],
            value[:, 4:],
            state=expected_native_state,
            native_head_crsa=intervention,
            **common,
        )
        observed = []
        (
            actual_off_decode,
            actual_off_next,
            actual_native_decode,
            actual_native_next,
            evidence,
        ) = full_attention_fork_core(
            query_gate[:, 4:],
            key[:, 4:],
            value[:, 4:],
            off_state=actual_off_state,
            native_state=actual_native_state,
            native_head_crsa=intervention,
            native_head_crsa_observer=observed.append,
            **common,
        )

        self.assertTrue(torch.equal(actual_off_decode, expected_off_decode))
        self.assertTrue(torch.equal(actual_native_decode, expected_native_decode))
        self.assertTrue(torch.equal(actual_off_next.key, expected_off_next.key))
        self.assertTrue(torch.equal(actual_off_next.value, expected_off_next.value))
        self.assertTrue(torch.equal(actual_native_next.key, expected_native_next.key))
        self.assertTrue(
            torch.equal(actual_native_next.value, expected_native_next.value)
        )
        self.assertTrue(
            torch.equal(
                actual_native_next.crsa_log_usage,
                expected_native_next.crsa_log_usage,
            )
        )
        self.assertIs(actual_off_next.key, actual_native_next.key)
        self.assertIs(actual_off_next.value, actual_native_next.value)
        self.assertIsNone(actual_off_next.crsa_log_usage)
        self.assertEqual(observed, [evidence])
        self.assertEqual((evidence.query_start, evidence.key_length), (4, 5))
        self.assertTrue(torch.equal(actual_off_state.key, saved_state[0]))
        self.assertTrue(torch.equal(actual_off_state.value, saved_state[1]))
        self.assertTrue(torch.equal(actual_native_state.crsa_log_usage, saved_state[2]))

    def test_full_attention_fork_rejects_ambiguous_or_diverged_states(self) -> None:
        from immer.runtimes.qwen3_8.kernels import (
            AttentionState,
            full_attention_fork_core,
        )
        from immer.runtimes.qwen3_8.native_crsa import Qwen38NativeHeadCrsa

        torch.manual_seed(227)
        query_gate = torch.randn(1, 2, 2 * 24 * 4)
        key = torch.randn(1, 2, 4 * 4)
        value = torch.randn(1, 2, 4 * 4)
        common = dict(
            q_norm_weight=torch.zeros(4),
            k_norm_weight=torch.zeros(4),
            num_attention_heads=24,
            num_key_value_heads=4,
            head_dim=4,
            rotary_dim=2,
        )
        intervention = Qwen38NativeHeadCrsa(alpha=0.1)
        _, off_state, _, native_state, _ = full_attention_fork_core(
            query_gate,
            key,
            value,
            native_head_crsa=intervention,
            **common,
        )

        with self.assertRaisesRegex(TypeError, "must be a Qwen38NativeHeadCrsa"):
            full_attention_fork_core(
                query_gate,
                key,
                value,
                native_head_crsa=None,  # type: ignore[arg-type]
                **common,
            )
        with self.assertRaisesRegex(TypeError, "observer must be callable"):
            full_attention_fork_core(
                query_gate,
                key,
                value,
                native_head_crsa=intervention,
                native_head_crsa_observer=3,  # type: ignore[arg-type]
                **common,
            )
        with self.assertRaisesRegex(ValueError, "both be present"):
            full_attention_fork_core(
                query_gate[:, :1],
                key[:, :1],
                value[:, :1],
                off_state=off_state,
                native_head_crsa=intervention,
                **common,
            )
        off_with_usage = AttentionState(
            key=off_state.key,
            value=off_state.value,
            crsa_log_usage=native_state.crsa_log_usage,
        )
        with self.assertRaisesRegex(ValueError, "off_state may not retain"):
            full_attention_fork_core(
                query_gate[:, :1],
                key[:, :1],
                value[:, :1],
                off_state=off_with_usage,
                native_state=native_state,
                native_head_crsa=intervention,
                **common,
            )
        with self.assertRaisesRegex(ValueError, "usage does not match"):
            full_attention_fork_core(
                query_gate[:, :1],
                key[:, :1],
                value[:, :1],
                off_state=off_state,
                native_state=off_state,
                native_head_crsa=intervention,
                **common,
            )
        diverged_key = native_state.key.clone()
        diverged_key[0, 0, 0, 0] += 1.0
        diverged_native = AttentionState(
            key=diverged_key,
            value=native_state.value,
            crsa_log_usage=native_state.crsa_log_usage,
        )
        with self.assertRaisesRegex(ValueError, "identical K/V history"):
            full_attention_fork_core(
                query_gate[:, :1],
                key[:, :1],
                value[:, :1],
                off_state=off_state,
                native_state=diverged_native,
                native_head_crsa=intervention,
                **common,
            )

    def test_causal_depthwise_conv_matches_scalar_and_continuation(self) -> None:
        from immer.runtimes.qwen3_8.kernels import causal_depthwise_conv

        projected = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
        weight = torch.tensor([[[0.1, 0.2, 0.3]]])
        actual, state = causal_depthwise_conv(projected, weight)
        expected_raw = torch.tensor(
            [[[0.3], [0.2 * 1.0 + 0.3 * 2.0], [0.1 + 0.4 + 0.9], [0.2 + 0.6 + 1.2]]]
        )
        torch.testing.assert_close(actual, F.silu(expected_raw))
        torch.testing.assert_close(state, torch.tensor([[[2.0, 3.0, 4.0]]]))

        first, first_state = causal_depthwise_conv(projected[:, :2], weight)
        second, second_state = causal_depthwise_conv(
            projected[:, 2:], weight, conv_state=first_state
        )
        torch.testing.assert_close(torch.cat((first, second), dim=1), actual)
        torch.testing.assert_close(second_state, state)

    @staticmethod
    def _recurrent_delta_reference(query, key, value, g, beta, initial_state=None):
        q = query * torch.rsqrt((query * query).sum(-1, keepdim=True) + 1e-6)
        k = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1e-6)
        q = q.float() * (query.shape[-1] ** -0.5)
        k = k.float()
        value = value.float()
        if initial_state is None:
            state = torch.zeros(
                query.shape[0],
                query.shape[2],
                query.shape[3],
                value.shape[3],
            )
        else:
            state = initial_state.float()
        output = []
        for token in range(query.shape[1]):
            state = state * torch.exp(g[:, token].float())[..., None, None]
            remembered = (state * k[:, token].unsqueeze(-1)).sum(-2)
            delta = (value[:, token] - remembered) * beta[:, token].float().unsqueeze(
                -1
            )
            state = state + k[:, token].unsqueeze(-1) * delta.unsqueeze(-2)
            output.append((state * q[:, token].unsqueeze(-1)).sum(-2))
        return torch.stack(output, 1).to(query.dtype), state

    def test_recurrent_gated_delta_rule_matches_direct_reference(self) -> None:
        from immer.runtimes.qwen3_8.kernels import recurrent_gated_delta_rule

        torch.manual_seed(52)
        query = torch.randn(2, 4, 3, 2)
        key = torch.randn(2, 4, 3, 2)
        value = torch.randn(2, 4, 3, 3)
        g = -torch.rand(2, 4, 3)
        beta = torch.sigmoid(torch.randn(2, 4, 3))
        expected, expected_state = self._recurrent_delta_reference(
            query, key, value, g, beta
        )
        actual, actual_state = recurrent_gated_delta_rule(query, key, value, g, beta)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual_state, expected_state, atol=1e-6, rtol=1e-6)

    def test_delta_net_prefill_batch_and_continuation_are_exact(self) -> None:
        from immer.runtimes.qwen3_8.kernels import gated_delta_net_core

        torch.manual_seed(7)
        batch, sequence = 2, 6
        key_heads, value_heads = 1, 2
        key_dim, value_dim = 2, 3
        qkv_features = 2 * key_heads * key_dim + value_heads * value_dim
        projected_qkv = torch.randn(batch, sequence, qkv_features)
        projected_z = torch.randn(batch, sequence, value_heads * value_dim)
        projected_b = torch.randn(batch, sequence, value_heads)
        projected_a = torch.randn(batch, sequence, value_heads)
        conv_weight = torch.randn(qkv_features, 1, 3) * 0.2
        a_log = torch.randn(value_heads) * 0.1
        dt_bias = torch.randn(value_heads) * 0.1
        norm_weight = torch.randn(value_dim) * 0.1 + 1.0
        kwargs = dict(
            conv1d_weight=conv_weight,
            A_log=a_log,
            dt_bias=dt_bias,
            norm_weight=norm_weight,
            num_key_heads=key_heads,
            num_value_heads=value_heads,
            key_head_dim=key_dim,
            value_head_dim=value_dim,
        )
        whole, whole_state = gated_delta_net_core(
            projected_qkv,
            projected_z,
            projected_b,
            projected_a,
            **kwargs,
        )
        prefix, prefix_state = gated_delta_net_core(
            projected_qkv[:, :2],
            projected_z[:, :2],
            projected_b[:, :2],
            projected_a[:, :2],
            **kwargs,
        )
        suffix, split_state = gated_delta_net_core(
            projected_qkv[:, 2:],
            projected_z[:, 2:],
            projected_b[:, 2:],
            projected_a[:, 2:],
            state=prefix_state,
            **kwargs,
        )
        torch.testing.assert_close(
            torch.cat((prefix, suffix), dim=1), whole, atol=2e-6, rtol=2e-6
        )
        torch.testing.assert_close(split_state.conv, whole_state.conv)
        torch.testing.assert_close(
            split_state.recurrent,
            whole_state.recurrent,
            atol=2e-6,
            rtol=2e-6,
        )
        self.assertEqual(tuple(whole.shape), (batch, sequence, value_heads * value_dim))
        self.assertEqual(
            tuple(whole_state.recurrent.shape),
            (batch, value_heads, key_dim, value_dim),
        )

        # One-token decode exercises the same persistent state contract.
        token_outputs = []
        token_state = None
        for token in range(sequence):
            output, token_state = gated_delta_net_core(
                projected_qkv[:, token : token + 1],
                projected_z[:, token : token + 1],
                projected_b[:, token : token + 1],
                projected_a[:, token : token + 1],
                state=token_state,
                **kwargs,
            )
            token_outputs.append(output)
        torch.testing.assert_close(
            torch.cat(token_outputs, dim=1), whole, atol=2e-6, rtol=2e-6
        )
        torch.testing.assert_close(token_state.conv, whole_state.conv)
        torch.testing.assert_close(
            token_state.recurrent,
            whole_state.recurrent,
            atol=2e-6,
            rtol=2e-6,
        )

    def test_shape_contracts_fail_before_ambiguous_broadcasts(self) -> None:
        from immer.runtimes.qwen3_8.kernels import (
            causal_depthwise_conv,
            full_attention_core,
            gated_delta_net_core,
            rope_cos_sin,
            split_query_gate,
        )

        with self.assertRaisesRegex(ValueError, "feature size"):
            split_query_gate(torch.zeros(1, 1, 7), 2, 2)
        with self.assertRaisesRegex(ValueError, "rotary_dim must be even"):
            rope_cos_sin(torch.arange(2)[None], 3)
        with self.assertRaisesRegex(ValueError, "does not match 2 channels"):
            causal_depthwise_conv(torch.zeros(1, 2, 2), torch.zeros(3, 1, 2))
        with self.assertRaisesRegex(ValueError, "projected_key feature size"):
            full_attention_core(
                torch.zeros(1, 2, 16),
                torch.zeros(1, 2, 3),
                torch.zeros(1, 2, 4),
                q_norm_weight=torch.zeros(4),
                k_norm_weight=torch.zeros(4),
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=4,
                rotary_dim=2,
            )
        with self.assertRaisesRegex(ValueError, "projected_qkv feature size"):
            gated_delta_net_core(
                torch.zeros(1, 2, 9),
                torch.zeros(1, 2, 6),
                torch.zeros(1, 2, 2),
                torch.zeros(1, 2, 2),
                conv1d_weight=torch.zeros(9, 1, 3),
                A_log=torch.zeros(2),
                dt_bias=torch.zeros(2),
                norm_weight=torch.ones(3),
                num_key_heads=1,
                num_value_heads=2,
                key_head_dim=2,
                value_head_dim=3,
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
