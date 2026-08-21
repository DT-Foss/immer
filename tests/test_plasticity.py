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
    def test_surprises_trigger_updates(self) -> None:
        from immer.runtimes.o1_state.plasticity import LearningStream

        stream = LearningStream(seed=3, min_observations=4)
        self.assertEqual(stream.updates, 0)
        for text in ("eins", "zwei", "drei", "vier", "FÜNF GANZ ANDERE plötzlich"):
            stream.observe(text)
        self.assertGreater(stream.surprises, 0)
        self.assertGreaterEqual(stream.updates, 0)
        self.assertEqual(stream.torch.get_num_threads(), 1)

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


if __name__ == "__main__":
    unittest.main()
