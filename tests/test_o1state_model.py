from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "vendor" / "o1state" / "results" / "pos_ckpt.pt"


@unittest.skipIf(torch is None, "torch is unavailable")
class O1StateModelTests(unittest.TestCase):
    def test_import_and_construction_leave_global_backend_controls_untouched(self) -> None:
        mps_predicate = torch.backends.mps.is_available
        thread_count = torch.get_num_threads()
        module = importlib.import_module("immer.runtimes.o1_state.model")
        module.StreamingNoPELM(
            31,
            32,
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_head=8,
            dropout=0.0,
        )
        self.assertIs(torch.backends.mps.is_available, mps_predicate)
        self.assertEqual(torch.get_num_threads(), thread_count)
        self.assertNotIn("streaming_train", sys.modules)

    def test_stateless_and_streaming_zero_state_are_bit_exact(self) -> None:
        from immer.runtimes.o1_state.model import (
            SelectiveNoPETransformerLM,
            StreamingNoPELM,
        )

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(73)
            stateless = SelectiveNoPETransformerLM(
                47,
                48,
                d_model=32,
                n_layers=2,
                n_heads=2,
                d_head=16,
                dropout=0.0,
            ).eval()
            streaming = StreamingNoPELM(
                47,
                48,
                d_model=32,
                n_layers=2,
                n_heads=2,
                d_head=16,
                dropout=0.0,
            ).eval()
        streaming.load_state_dict(stateless.state_dict(), strict=True)
        tokens = torch.tensor([[3, 8, 13, 21, 34, 1, 2, 5, 9]])
        with torch.no_grad():
            expected = stateless(tokens)
            zero_state, _ = streaming(tokens, None)
            first, state = streaming(tokens[:, :4], None)
            second, _ = streaming(tokens[:, 4:], state)
        self.assertTrue(torch.equal(expected, zero_state))
        # Different GEMM batch widths can round the final bit differently,
        # while the carried recurrence itself remains the same.
        torch.testing.assert_close(
            expected,
            torch.cat((first, second), dim=1),
            rtol=1e-6,
            atol=1e-6,
        )

    @unittest.skipUnless(CHECKPOINT.is_file(), "local frozen A1 checkpoint is unavailable")
    def test_frozen_a1_checkpoint_strict_load_and_golden_logits(self) -> None:
        from immer.runtimes.o1_state.model import (
            SelectiveNoPETransformerLM,
            StreamingNoPELM,
        )

        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        state_dict = checkpoint["arms"]["A1"]["model"]
        kwargs = {
            "d_model": 128,
            "n_layers": 2,
            "n_heads": 4,
            "d_head": 32,
            "seq_len": 64,
            "dropout": 0.0,
            "causal": True,
        }
        stateless = SelectiveNoPETransformerLM(5000, 5001, **kwargs).eval()
        streaming = StreamingNoPELM(5000, 5001, **kwargs).eval()
        self.assertEqual(stateless.load_state_dict(state_dict, strict=True).missing_keys, [])
        self.assertEqual(streaming.load_state_dict(state_dict, strict=True).unexpected_keys, [])
        tokens = torch.tensor([[60, 3463, 149, 14, 31, 209, 38, 14]])
        golden = torch.tensor(
            [
                0.3587441146,
                -0.4421998560,
                -1.2105225325,
                1.2141244411,
                0.0034205543,
                -0.6901168227,
                -0.3276996315,
                0.5711829066,
            ]
        )
        with torch.no_grad():
            logits = stateless(tokens)
            streamed, _ = streaming(tokens, None)
        torch.testing.assert_close(logits[0, -1, :8], golden, rtol=1e-6, atol=1e-6)
        self.assertTrue(torch.equal(logits, streamed))


if __name__ == "__main__":
    unittest.main()
