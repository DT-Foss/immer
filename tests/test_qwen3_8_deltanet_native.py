from __future__ import annotations

import math
import unittest

import torch

from immer.runtimes.qwen3_8.deltanet_native import deltanet_sequence_one


def _reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    decayed = initial_state * torch.exp(log_decay[:, 0])[..., None, None]
    remembered = (decayed * key[:, 0].unsqueeze(-1)).sum(dim=-2)
    delta = (value[:, 0] - remembered) * beta[:, 0].unsqueeze(-1)
    new_state = decayed + key[:, 0].unsqueeze(-1) * delta.unsqueeze(-2)
    output = (new_state * query[:, 0].unsqueeze(-1)).sum(dim=-2).unsqueeze(1)
    return output, new_state


def _inputs(
    batch: int,
    heads: int,
    key_width: int,
    value_width: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(
        (batch, 1, heads, key_width), generator=generator, dtype=torch.float32
    )
    key = torch.randn(query.shape, generator=generator, dtype=torch.float32)
    query = torch.nn.functional.normalize(query, dim=-1, eps=1e-6)
    query = query * (key_width**-0.5)
    key = torch.nn.functional.normalize(key, dim=-1, eps=1e-6)
    value = torch.randn(
        (batch, 1, heads, value_width), generator=generator, dtype=torch.float32
    )
    log_decay = -torch.rand(
        (batch, 1, heads), generator=generator, dtype=torch.float32
    )
    beta = torch.sigmoid(
        torch.randn((batch, 1, heads), generator=generator, dtype=torch.float32)
    )
    state = torch.randn(
        (batch, heads, key_width, value_width),
        generator=generator,
        dtype=torch.float32,
    ) * 0.02
    return query, key, value, log_decay, beta, state


class DeltaNetNativeTests(unittest.TestCase):
    def test_small_and_multibatch_match_pytorch(self) -> None:
        inputs = _inputs(3, 4, 7, 9, seed=71)
        expected_output, expected_state = _reference(*inputs)
        actual_output, actual_state = deltanet_sequence_one(*inputs, threads=3)
        torch.testing.assert_close(
            actual_output, expected_output, rtol=2e-6, atol=2e-6
        )
        torch.testing.assert_close(
            actual_state, expected_state, rtol=2e-6, atol=2e-6
        )
        self.assertEqual(tuple(actual_output.shape), (3, 1, 4, 9))
        self.assertEqual(tuple(actual_state.shape), (3, 4, 7, 9))

    def test_real_qwen_dimensions_match_pytorch(self) -> None:
        inputs = _inputs(1, 48, 128, 128, seed=12848)
        expected_output, expected_state = _reference(*inputs)
        actual_output, actual_state = deltanet_sequence_one(*inputs, threads=8)
        torch.testing.assert_close(
            actual_output, expected_output, rtol=2e-5, atol=2e-5
        )
        torch.testing.assert_close(
            actual_state, expected_state, rtol=2e-6, atol=2e-6
        )

    def test_inputs_are_immutable_and_outputs_do_not_alias(self) -> None:
        inputs = _inputs(2, 3, 8, 6, seed=204)
        snapshots = tuple(value.clone() for value in inputs)
        output, state = deltanet_sequence_one(*inputs, threads=2)
        for actual, snapshot in zip(inputs, snapshots, strict=True):
            self.assertTrue(torch.equal(actual, snapshot))
        input_pointers = {value.data_ptr() for value in inputs}
        self.assertNotIn(output.data_ptr(), input_pointers)
        self.assertNotIn(state.data_ptr(), input_pointers)
        self.assertNotEqual(output.data_ptr(), state.data_ptr())
        before = state.clone()
        output.zero_()
        self.assertTrue(torch.equal(state, before))

    def test_thread_counts_are_bit_exact(self) -> None:
        inputs = _inputs(2, 8, 32, 24, seed=816)
        one_output, one_state = deltanet_sequence_one(*inputs, threads=1)
        many_output, many_state = deltanet_sequence_one(*inputs, threads=16)
        self.assertTrue(torch.equal(one_output, many_output))
        self.assertTrue(torch.equal(one_state, many_state))

    def test_invalid_shapes_dtypes_threads_and_nonfinite_values_fail(self) -> None:
        inputs = list(_inputs(2, 3, 8, 5, seed=900))
        with self.assertRaises(ValueError):
            deltanet_sequence_one(inputs[0].repeat(1, 2, 1, 1), *inputs[1:])
        with self.assertRaises(ValueError):
            deltanet_sequence_one(
                inputs[0], inputs[1][..., :-1], *inputs[2:]
            )
        with self.assertRaises(ValueError):
            deltanet_sequence_one(
                inputs[0], inputs[1], inputs[2][:, :, :-1], *inputs[3:]
            )
        with self.assertRaises(ValueError):
            deltanet_sequence_one(
                inputs[0], inputs[1], inputs[2], inputs[3][:, :, :-1], *inputs[4:]
            )
        with self.assertRaises(ValueError):
            deltanet_sequence_one(*inputs[:-1], inputs[-1][..., :-1])
        double_query = inputs[0].double()
        with self.assertRaises(ValueError):
            deltanet_sequence_one(double_query, *inputs[1:])
        for threads in (0, -1, True):
            with self.subTest(threads=threads), self.assertRaises(ValueError):
                deltanet_sequence_one(*inputs, threads=threads)

        for position in range(len(inputs)):
            corrupted = [value.clone() for value in inputs]
            corrupted[position].view(-1)[0] = (
                math.inf if position % 2 == 0 else math.nan
            )
            with self.subTest(nonfinite=position), self.assertRaises(ValueError):
                deltanet_sequence_one(*corrupted, threads=2)


if __name__ == "__main__":
    unittest.main()
