from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import unittest

import numpy as np
import torch


class _Metrics:
    def __init__(self) -> None:
        self.values = {
            "network_or_source_body_bytes": 100,
            "linear_calls": 7,
        }

    def metrics(self):
        return dict(self.values)


class _FakePager(_Metrics):
    def __init__(self, schedules):
        super().__init__()
        self.source = _Metrics()
        self.schedules = list(schedules)
        self.scans = []
        self.releases = 0

    def release(self):
        self.releases += 1

    def topk_logits(self, hidden, **_kwargs):
        schedule = self.schedules[len(self.scans)]
        self.scans.append(int(hidden.shape[0]))
        self.values["linear_calls"] += 1
        self.source.values["network_or_source_body_bytes"] += 10
        return (
            torch.arange(len(schedule), dtype=torch.float32).reshape(-1, 1),
            torch.tensor(schedule, dtype=torch.long).reshape(-1, 1),
        )


@dataclass(frozen=True)
class _FakeHandle:
    layer: int
    next_position: int
    batch_size: int
    tensors: dict


class _FakeRunner:
    def __init__(self, model):
        self.model = model
        self.calls = []

    def forward_layer_stateful(
        self,
        hidden,
        token_ids,
        *,
        layer,
        attention_state=None,
        token_mask=None,
    ):
        ids = torch.as_tensor(token_ids)
        cursor = ids.shape[1] if attention_state is None else attention_state.next_position + 1
        self.calls.append(
            {
                "layer": layer,
                "ids": ids.clone(),
                "mask": None if token_mask is None else torch.as_tensor(token_mask).clone(),
                "input_cursor": None if attention_state is None else attention_state.next_position,
            }
        )
        return (
            hidden + float(layer + 1),
            (),
            _FakeHandle(layer, cursor, ids.shape[0], {}),
        )


class _FlakyRunner(_FakeRunner):
    def __init__(self, model):
        super().__init__(model)
        self.remaining_failures = 1

    def forward_layer_stateful(self, *args, **kwargs):
        if kwargs.get("attention_state") is not None and self.remaining_failures:
            self.remaining_failures -= 1
            raise TimeoutError("transient source timeout")
        return super().forward_layer_stateful(*args, **kwargs)


class _FakeModel:
    def __init__(self, schedules, *, graft=None, graft_layer=None):
        self.config = SimpleNamespace(n_layers=2, vocab_size=128, hc_mult=1, dim=2)
        self.max_batch_size = 4
        self.max_seq_len = 16
        self.pager = _FakePager(schedules)
        self.graft = graft
        self.graft_layer = graft_layer
        self.reset_calls = 0

    def _token_tensor(self, value):
        result = torch.as_tensor(value)
        if result.dtype == torch.bool or result.is_floating_point() or result.is_complex():
            raise TypeError("token_ids must contain integers")
        if result.ndim == 1:
            result = result.unsqueeze(0)
        return result.to(dtype=torch.long)

    def embed_batch(self, ids):
        values = self._token_tensor(ids).float()
        return values.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 2)

    def finalize_hidden(self, hidden):
        return hidden[:, :, 0]

    def reset_state(self, *, release=False):
        self.reset_calls += int(release)


class _FakeGraft:
    mode = "crsa"
    alpha = 0.01

    def __init__(self):
        self.history_lengths = []

    def forward(self, hidden, *, return_evidence=False):
        self.history_lengths.append(hidden.shape[1])
        return (hidden + 0.25, {"ok": True}) if return_evidence else hidden + 0.25


class DeepSeekV4LayerwiseGenerationTests(unittest.TestCase):
    def test_real_runtime_matches_resident_multi_token_generation(self):
        from immer.runtimes.deepseek_v4 import (
            DeepSeekWeightPager,
            StreamedDeepSeekV4,
        )
        from immer.runtimes.deepseek_v4.layerwise_generation import (
            LayerwiseGenerator,
        )
        from test_deepseek_v4_model import _CompressedTinyCheckpoint, _config

        def model():
            return StreamedDeepSeekV4(
                _config(4),
                DeepSeekWeightPager(
                    _CompressedTinyCheckpoint(
                        random_weights=True,
                        compress_ratio=4,
                    ),
                    device="cpu",
                    compute_dtype="float32",
                ),
                max_batch_size=1,
                max_seq_len=12,
            )

        resident = model()
        layerwise = model()
        expected, _evidence = resident.generate_greedy(
            [[1, 2, 3, 4]],
            max_new_tokens=4,
            prefill_tokenwise=False,
        )
        actual, evidence = LayerwiseGenerator(layerwise).generate_greedy(
            [[1, 2, 3, 4]],
            max_new_tokens=4,
        )

        self.assertEqual(actual, (expected,))
        self.assertEqual(evidence.forward_layer_calls, 4)
        self.assertEqual(evidence.head_scans, 4)

    def test_staggered_eos_keeps_fixed_cache_batch_and_compacts_head_rows(self):
        from immer.runtimes.deepseek_v4.layerwise_generation import (
            LayerwiseGenerator,
        )

        model = _FakeModel(([99, 5, 6], [99, 7], [99]))
        runner = _FakeRunner(model)
        generator = LayerwiseGenerator(model, runner=runner)

        generated, evidence = generator.generate_greedy(
            [[1, 2], [3, 4], [5, 6]],
            max_new_tokens=4,
            eos_token_ids=(99,),
            filler_token_id=0,
        )

        self.assertEqual(generated, ((99,), (5, 99), (6, 7, 99)))
        self.assertEqual(model.pager.scans, [3, 2, 1])
        self.assertEqual(evidence.head_scans, 3)
        self.assertEqual(evidence.forward_layer_calls, 6)
        self.assertEqual(evidence.stopped_on_eos, (True, True, True))
        decode_calls = [call for call in runner.calls if call["mask"] is not None]
        self.assertEqual(len(decode_calls), 4)
        for call in decode_calls[:2]:
            self.assertTrue(torch.equal(call["mask"], torch.tensor([[False], [True], [True]])))
            self.assertTrue(torch.equal(call["ids"], torch.tensor([[0], [5], [6]])))
        for call in decode_calls[2:]:
            self.assertTrue(torch.equal(call["mask"], torch.tensor([[False], [False], [True]])))
            self.assertTrue(torch.equal(call["ids"], torch.tensor([[0], [0], [7]])))

    def test_max_token_stop_does_not_invent_eos(self):
        from immer.runtimes.deepseek_v4.layerwise_generation import (
            LayerwiseGenerator,
        )

        model = _FakeModel(([10, 11], [12, 13]))
        generated, evidence = LayerwiseGenerator(
            model, runner=_FakeRunner(model)
        ).generate_greedy([[1, 2], [3, 4]], max_new_tokens=2)

        self.assertEqual(generated, ((10, 12), (11, 13)))
        self.assertEqual(evidence.stopped_on_eos, (False, False))
        self.assertEqual(evidence.head_scans, 2)

    def test_retries_only_the_failed_layer_from_its_input_handle(self):
        from immer.runtimes.deepseek_v4.layerwise_generation import (
            LayerwiseGenerator,
        )

        model = _FakeModel(([10], [11]))
        runner = _FlakyRunner(model)
        generated, evidence = LayerwiseGenerator(
            model,
            runner=runner,
            layer_retries=1,
        ).generate_greedy([[1, 2]], max_new_tokens=2)

        self.assertEqual(generated, ((10, 11),))
        self.assertEqual(evidence.forward_layer_calls, 4)
        self.assertEqual(evidence.layer_retry_count, 1)
        self.assertEqual(runner.remaining_failures, 0)

    def test_active_graft_keeps_raw_history_at_the_graft_layer(self):
        from immer.runtimes.deepseek_v4.layerwise_generation import (
            LayerwiseGenerator,
        )

        graft = _FakeGraft()
        model = _FakeModel(([8], [9]), graft=graft, graft_layer=0)
        generated, evidence = LayerwiseGenerator(
            model, runner=_FakeRunner(model)
        ).generate_greedy([[1, 2]], max_new_tokens=2)

        self.assertEqual(generated, ((8, 9),))
        self.assertEqual(graft.history_lengths, [2, 3])
        self.assertEqual(evidence.graft_history_tokens, 3)
        self.assertEqual(evidence.graft_mode, "crsa")

    def test_rejects_ragged_or_overlong_batches_before_forward(self):
        from immer.runtimes.deepseek_v4.layerwise_generation import (
            LayerwiseGenerator,
        )

        model = _FakeModel(([1],))
        generator = LayerwiseGenerator(model, runner=_FakeRunner(model))
        with self.assertRaisesRegex(ValueError, "exact-length"):
            generator.generate_greedy([[1, 2], [3]], max_new_tokens=1)
        with self.assertRaisesRegex(ValueError, "context bound"):
            generator.generate_greedy([[1] * 15], max_new_tokens=2)

    def test_preserves_runtime_token_validation_and_rejects_foreign_runner(self):
        from immer.runtimes.deepseek_v4.layerwise_generation import (
            LayerwiseGenerator,
        )

        model = _FakeModel(([1],))
        generator = LayerwiseGenerator(model, runner=_FakeRunner(model))
        with self.assertRaisesRegex(TypeError, "integers"):
            generator.generate_greedy([[1.5, 2.5]], max_new_tokens=1)

        generated, _evidence = generator.generate_greedy(
            np.asarray([1, 2], dtype=np.int64),
            max_new_tokens=1,
        )
        self.assertEqual(generated, ((1,),))

        other = _FakeModel(([1],))
        with self.assertRaisesRegex(ValueError, "different model"):
            LayerwiseGenerator(model, runner=_FakeRunner(other))


if __name__ == "__main__":
    unittest.main()
