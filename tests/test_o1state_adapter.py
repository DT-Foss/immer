from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401
except ImportError:
    torch = None

from immer.runtimes.o1_state.adapter import O1StateStream, is_available


@unittest.skipIf(torch is None or not is_available(), "torch or vendored o1-state missing")
class O1StateStreamTests(unittest.TestCase):
    def test_observe_streams_tokens_and_updates_ema(self) -> None:
        stream = O1StateStream()
        self.assertEqual(stream.tokens, 0)
        stream.observe("hallo")
        self.assertGreater(stream.tokens, 0)
        first = stream.loss_ema
        self.assertIsNotNone(first)
        stream.observe("mehr leben fließt durch den strom")
        self.assertGreater(stream.tokens, 4)
        self.assertNotEqual(stream.loss_ema, first)

    def test_snapshot_restore_roundtrip_continues_life(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / "life.pt"
            first = O1StateStream(sidecar=sidecar)
            first.observe("erinnerungstest")
            snapshot = first.snapshot()

            second = O1StateStream(sidecar=sidecar)
            second.restore(snapshot)
            self.assertEqual(second.tokens, first.tokens)
            self.assertAlmostEqual(second.loss_ema, first.loss_ema, places=9)
            second.observe("und es geht weiter")
            self.assertGreater(second.tokens, first.tokens)

    def test_same_seed_same_experience_same_loss(self) -> None:
        a = O1StateStream(seed=7)
        b = O1StateStream(seed=7)
        a.observe("determinismus")
        b.observe("determinismus")
        self.assertEqual(a.tokens, b.tokens)
        self.assertAlmostEqual(a.loss_ema, b.loss_ema, places=9)


if __name__ == "__main__":
    unittest.main()
