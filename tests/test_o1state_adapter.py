from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401
except ImportError:
    torch = None

from immer.runtimes.o1_state.adapter import O1StateStream, is_available


@unittest.skipIf(
    torch is None or not is_available(), "torch or production o1-state missing"
)
class O1StateStreamTests(unittest.TestCase):
    def test_construction_is_local_to_its_rng_and_backend_settings(self) -> None:
        torch.manual_seed(991)
        rng_state = torch.random.get_rng_state().clone()
        mps_predicate = torch.backends.mps.is_available
        thread_count = torch.get_num_threads()
        O1StateStream(seed=17)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_state))
        self.assertIs(torch.backends.mps.is_available, mps_predicate)
        self.assertEqual(torch.get_num_threads(), thread_count)

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

    def test_pos_stream_is_shifted_and_carries_tail_across_observations(self) -> None:
        stream = O1StateStream(seed=29)
        seen = []

        class RecordingModel(torch.nn.Module):
            def forward(self, values, states):
                seen.append(values.detach().clone())
                logits = torch.zeros(
                    values.shape[0], values.shape[1], 257, dtype=torch.float32
                )
                return logits, [None, None]

        stream.model = RecordingModel()
        stream.states = [None, None]
        stream.observe("ab")
        stream.observe("cd")

        self.assertEqual([value.tolist() for value in seen], [[[97]], [[98, 99]]])
        self.assertEqual(stream.tokens, 3)
        self.assertEqual(stream.tail, 100)

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

    def test_content_addressed_receipt_binds_tail_and_rejects_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "life.pt"
            first = O1StateStream(sidecar=base, seed=31)
            first.observe("continuous")
            snapshot = first.snapshot()
            receipt = snapshot["sidecar"]
            self.assertIsInstance(receipt, dict)
            path = base.parent / receipt["name"]
            self.assertTrue(path.is_file())
            self.assertFalse(base.exists())
            self.assertEqual(snapshot["tail"], ord("s"))

            second = O1StateStream(sidecar=base, seed=99)
            second.restore(snapshot)
            self.assertEqual(second.tail, first.tail)
            self.assertEqual(second.tokens, first.tokens)
            self.assertTrue(
                all(torch.equal(a, b) for a, b in zip(second.states, first.states))
            )

            first.observe(" state")
            next_snapshot = first.snapshot()
            next_path = base.parent / next_snapshot["sidecar"]["name"]
            first.commit_snapshot(next_snapshot)
            self.assertTrue(next_path.is_file())
            self.assertFalse(path.exists())

            raw = bytearray(next_path.read_bytes())
            raw[-1] ^= 1
            next_path.write_bytes(raw)
            third = O1StateStream(sidecar=base, seed=31)
            with self.assertRaisesRegex(ValueError, "digest"):
                third.restore(next_snapshot)

    def test_same_seed_same_experience_same_loss(self) -> None:
        a = O1StateStream(seed=7)
        b = O1StateStream(seed=7)
        a.observe("determinismus")
        b.observe("determinismus")
        self.assertEqual(a.tokens, b.tokens)
        self.assertAlmostEqual(a.loss_ema, b.loss_ema, places=9)


if __name__ == "__main__":
    unittest.main()
