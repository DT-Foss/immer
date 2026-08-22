from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401
except ImportError:
    torch = None

from immer.runtimes.o1_state.adapter import is_available


@unittest.skipIf(torch is None or not is_available(), "torch or vendored o1-state missing")
class PlasticityTests(unittest.TestCase):
    def test_invalid_surprise_configuration_is_rejected(self) -> None:
        from immer.runtimes.o1_state.plasticity import LearningStream

        for kwargs in (
            {"window": 0},
            {"quantile": -0.1},
            {"quantile": 1.1},
            {"min_observations": -1},
            {"max_grad_norm": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                LearningStream(**kwargs)

        stream = LearningStream()
        with self.assertRaisesRegex(ValueError, "epochs"):
            stream.sleep(epochs=-1)

    def test_surprises_trigger_updates(self) -> None:
        from immer.runtimes.o1_state.plasticity import LearningStream

        stream = LearningStream(seed=3, min_observations=4, quantile=0.0)
        self.assertEqual(stream.updates, 0)
        for text in ("eins", "zwei", "drei", "vier", "FÜNF GANZ ANDERE plötzlich"):
            stream.observe(text)
        self.assertGreater(stream.surprises, 0)
        self.assertGreater(stream.updates, 0)

    def test_construction_does_not_change_process_thread_count(self) -> None:
        from immer.runtimes.o1_state.plasticity import LearningStream

        before = torch.get_num_threads()
        LearningStream(seed=4)
        self.assertEqual(torch.get_num_threads(), before)

    def test_learning_reduces_loss_on_repeat(self) -> None:
        from immer.runtimes.o1_state.plasticity import LearningStream

        stream = LearningStream(seed=5, min_observations=2, quantile=0.0)
        losses = []
        for _ in range(6):
            stream.observe("wiederholung macht den meister")
            losses.append(stream.loss_ema)
        # nach Updates auf demselben Text muss das EMA unter dem ersten Wert liegen
        self.assertLess(losses[-1], losses[0])

    def test_sleep_replays_and_clears_buffer(self) -> None:
        from immer.runtimes.o1_state.plasticity import LearningStream

        stream = LearningStream(seed=9, min_observations=2, quantile=0.0)
        for _ in range(5):
            stream.observe("schlafzyklus übung")
        before = len(stream.spans)
        report = stream.sleep()
        self.assertEqual(report["replayed"], before * 2)
        self.assertEqual(len(stream.spans), 0)
        self.assertEqual(stream.sleeps, 1)

    def test_metrics_report_shape(self) -> None:
        from immer.runtimes.o1_state.plasticity import LearningStream

        stream = LearningStream(seed=11)
        stream.observe("metriken")
        report = stream.metrics()
        for key in ("tokens", "loss_ema", "surprises", "updates", "sleeps", "span_buffer"):
            self.assertIn(key, report)

    def test_non_surprising_chunks_never_carry_an_autograd_graph(self) -> None:
        from immer.runtimes.o1_state.plasticity import LearningStream

        stream = LearningStream(seed=13, min_observations=100)
        stream.observe("a quiet ordinary chunk")
        for state in stream.states:
            self.assertFalse(state.requires_grad)
            self.assertIsNone(state.grad_fn)

    def test_surprise_recompute_updates_and_detaches_carried_state(self) -> None:
        from immer.runtimes.o1_state.plasticity import LearningStream

        stream = LearningStream(seed=17, min_observations=0)
        stream._is_surprising = lambda _loss: True
        before = {
            name: value.detach().clone()
            for name, value in stream.model.state_dict().items()
        }
        stream.observe("this chunk must update")
        self.assertEqual(stream.updates, 1)
        self.assertTrue(any(
            not stream.torch.equal(before[name], value)
            for name, value in stream.model.state_dict().items()
        ))
        for state in stream.states:
            self.assertFalse(state.requires_grad)
            self.assertIsNone(state.grad_fn)

    def test_plasticity_state_survives_restart(self) -> None:
        from immer.runtimes.o1_state.plasticity import LearningStream

        with tempfile.TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / "life.pt"
            first = LearningStream(sidecar=sidecar, seed=19, min_observations=0)
            first._is_surprising = lambda _loss: True
            first.observe("remember this surprising span")
            snapshot = first.snapshot()

            second = LearningStream(sidecar=sidecar, seed=23, min_observations=0)
            second.restore(snapshot)
            self.assertEqual(second.metrics(), first.metrics())
            self.assertEqual(list(second.window), list(first.window))
            self.assertEqual(list(second.spans), list(first.spans))


if __name__ == "__main__":
    unittest.main()
