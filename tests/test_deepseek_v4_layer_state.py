from __future__ import annotations

from dataclasses import replace
import unittest
from unittest import mock

import torch

from immer.runtimes.deepseek_v4 import DeepSeekWeightPager, StreamedDeepSeekV4
from immer.runtimes.deepseek_v4.layer_state import LayerStateRunner
from test_deepseek_v4_model import _CompressedTinyCheckpoint, _config


def _model(*, batch: int = 2, max_seq_len: int = 12) -> StreamedDeepSeekV4:
    source = _CompressedTinyCheckpoint(random_weights=True, compress_ratio=4)
    pager = DeepSeekWeightPager(
        source,
        device="cpu",
        compute_dtype="float32",
    )
    return StreamedDeepSeekV4(
        _config(4),
        pager,
        max_batch_size=batch,
        max_seq_len=max_seq_len,
    )


class LayerStateRunnerTests(unittest.TestCase):
    def test_prefill_decode_matches_resident_native_state(self) -> None:
        direct = _model()
        portable = _model()
        runner = LayerStateRunner(portable)
        prefill_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

        direct_prefill, direct_selected = direct._block(
            direct.embed_batch(prefill_ids), 0, prefill_ids, 0
        )
        actual_prefill, actual_selected, state = runner.forward_layer_stateful(
            portable.embed_batch(prefill_ids), prefill_ids, layer=0
        )
        torch.testing.assert_close(actual_prefill, direct_prefill, rtol=0, atol=0)
        self.assertEqual(actual_selected, direct_selected)
        self.assertEqual(state.next_position, 4)
        self.assertEqual(state.batch_size, 2)
        self.assertTrue(state.tensors)
        self.assertTrue(all(value.device.type == "cpu" for value in state.tensors.values()))
        self.assertIsNone(portable._attention_states[0])

        decode_ids = torch.tensor([[9], [10]])
        direct_decode, direct_selected = direct._block(
            direct.embed_batch(decode_ids), 0, decode_ids, 4
        )
        actual_decode, actual_selected, decoded = runner.forward_layer_stateful(
            portable.embed_batch(decode_ids),
            decode_ids,
            layer=0,
            attention_state=state,
        )
        torch.testing.assert_close(actual_decode, direct_decode, rtol=0, atol=0)
        self.assertEqual(actual_selected, direct_selected)
        self.assertEqual(decoded.next_position, 5)
        self.assertIsNone(portable._attention_states[0])
        direct.release_layer_state(0)

    def test_failed_decode_cannot_mutate_input_handle_and_cleans_up(self) -> None:
        model = _model()
        runner = LayerStateRunner(model)
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        _output, _selected, state = runner.forward_layer_stateful(
            model.embed_batch(ids), ids, layer=0
        )
        original_tensors = {
            name: value.clone() for name, value in state.tensors.items()
        }

        def fail_after_mutation(*_args, **_kwargs):
            resident = model._attention_states[0]
            assert resident is not None
            assert resident.local._storage is not None
            resident.local._storage.add_(17)
            raise RuntimeError("injected layer failure")

        decode_ids = torch.tensor([[7], [8]])
        with mock.patch.object(model, "_block", side_effect=fail_after_mutation):
            with self.assertRaisesRegex(RuntimeError, "injected layer failure"):
                runner.forward_layer_stateful(
                    model.embed_batch(decode_ids),
                    decode_ids,
                    layer=0,
                    attention_state=state,
                )

        for name, expected in original_tensors.items():
            self.assertTrue(torch.equal(state.tensors[name], expected))
        self.assertIsNone(model._attention_states[0])
        self.assertEqual(model.attention_state_bytes, 0)

    def test_rejects_wrong_layer_batch_cursor_and_runtime(self) -> None:
        model = _model()
        runner = LayerStateRunner(model)
        ids = torch.tensor([[1, 2], [3, 4]])
        _output, _selected, state = runner.forward_layer_stateful(
            model.embed_batch(ids), ids, layer=0
        )

        with self.assertRaisesRegex(ValueError, "decoder depth"):
            runner.forward_layer_stateful(
                model.embed_batch(torch.tensor([[5], [6]])),
                torch.tensor([[5], [6]]),
                layer=1,
                attention_state=state,
            )
        with self.assertRaisesRegex(ValueError, "batch"):
            runner.forward_layer_stateful(
                model.embed_batch(torch.tensor([[5]])),
                torch.tensor([[5]]),
                layer=0,
                attention_state=state,
            )

        wrong_cursor = replace(state, next_position=state.next_position + 1)
        with self.assertRaisesRegex(ValueError, "cursor"):
            runner.forward_layer_stateful(
                model.embed_batch(torch.tensor([[5], [6]])),
                torch.tensor([[5], [6]]),
                layer=0,
                attention_state=wrong_cursor,
            )

        other_runner = LayerStateRunner(_model(max_seq_len=11))
        with self.assertRaisesRegex(ValueError, "different V4 runtime"):
            other_runner.forward_layer_stateful(
                other_runner.model.embed_batch(torch.tensor([[5], [6]])),
                torch.tensor([[5], [6]]),
                layer=0,
                attention_state=state,
            )

        self.assertIsNone(model._attention_states[0])
        self.assertIsNone(other_runner.model._attention_states[0])

    def test_masks_require_prefill_prefix_but_allow_finished_decode_rows(self) -> None:
        model = _model()
        runner = LayerStateRunner(model)
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        with self.assertRaisesRegex(ValueError, "exact prefixes"):
            runner.forward_layer_stateful(
                model.embed_batch(ids),
                ids,
                layer=0,
                token_mask=torch.tensor(
                    [[True, False, True], [True, True, True]]
                ),
            )

        _output, _selected, state = runner.forward_layer_stateful(
            model.embed_batch(ids), ids, layer=0
        )
        decode_ids = torch.tensor([[7], [0]])
        _output, selected, decoded = runner.forward_layer_stateful(
            model.embed_batch(decode_ids),
            decode_ids,
            layer=0,
            attention_state=state,
            token_mask=torch.tensor([[True], [False]]),
        )
        self.assertNotEqual(selected[0], ())
        self.assertEqual(selected[1], ())
        self.assertEqual(decoded.next_position, 4)
        self.assertEqual(decoded.batch_size, 2)
        self.assertIsNone(model._attention_states[0])

        with self.assertRaisesRegex(TypeError, "boolean"):
            runner.forward_layer_stateful(
                model.embed_batch(decode_ids),
                decode_ids,
                layer=0,
                attention_state=state,
                token_mask=torch.ones((2, 1), dtype=torch.int64),
            )


if __name__ == "__main__":
    unittest.main()
