from __future__ import annotations

import json
import math
import unittest

import numpy as np

from immer.runtimes.ooe.contraction_ledger import (
    ContractionLedger,
    ContractionLedgerIntegrityError,
    KernelContraction,
)
from immer.runtimes.ooe.identity import canonical_json_bytes


def _compose(kernels: tuple[np.ndarray, ...]) -> np.ndarray:
    result = kernels[0].copy()
    for kernel in kernels[1:]:
        result = np.einsum("ij,jk->ik", result, kernel, optimize=False)
    return result


class ContractionLedgerTests(unittest.TestCase):
    def test_submultiplicative_bound_verifies_without_replay(self) -> None:
        rng = np.random.default_rng(88)
        kernels = []
        for _ in range(7):
            value = rng.random((9, 9))
            value /= value.sum(axis=1, keepdims=True)
            kernels.append(value)
        chain = tuple(kernels)
        ledger = ContractionLedger.from_kernels(chain)
        composed = _compose(chain)
        self.assertTrue(ledger.verifies(composed))
        self.assertLessEqual(ledger.tau_upper_bound, 1.0)
        self.assertEqual(ledger.depth, len(chain))

    def test_zero_contraction_is_absorbing(self) -> None:
        uniform = np.full((5, 5), 0.2)
        identity = np.eye(5)
        ledger = ContractionLedger.from_kernels((identity, uniform, identity))
        self.assertTrue(ledger.zero_contraction)
        self.assertIsNone(ledger.log_tau_upper_bound)
        self.assertEqual(ledger.tau_upper_bound, 0.0)
        self.assertTrue(ledger.verifies(_compose((identity, uniform, identity))))

    def test_log_ledger_composes_at_large_depth_without_kernel_materialization(
        self,
    ) -> None:
        entry = KernelContraction("a" * 64, 0.999)
        left = ContractionLedger((entry,) * 5_000)
        right = ContractionLedger((entry,) * 5_000)
        combined = left.compose(right)
        self.assertEqual(combined.depth, 10_000)
        self.assertAlmostEqual(combined.log_tau_upper_bound, 10_000 * math.log(0.999))
        self.assertAlmostEqual(combined.tau_upper_bound, 0.999**10_000)

    def test_canonical_roundtrip_and_derived_field_tamper(self) -> None:
        kernel = np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)
        ledger = ContractionLedger.from_kernels((kernel, kernel))
        data = ledger.to_bytes()
        restored = ContractionLedger.from_bytes(data)
        self.assertEqual(restored, ledger)
        self.assertEqual(restored.to_bytes(), data)
        self.assertEqual(len(restored.sha256), 64)

        root = json.loads(data)
        root["tau_upper_bound_hex"] = float(0.99).hex()
        with self.assertRaises(ContractionLedgerIntegrityError):
            ContractionLedger.from_bytes(canonical_json_bytes(root))

    def test_invalid_tau_and_noncanonical_entry_fail_closed(self) -> None:
        for value in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                KernelContraction("a" * 64, value)
        root = ContractionLedger(()).to_dict()
        root["entries"] = [{"kernel_sha256": "a" * 64}]
        root["depth"] = 1
        with self.assertRaises(ContractionLedgerIntegrityError):
            ContractionLedger.from_bytes(canonical_json_bytes(root))


if __name__ == "__main__":
    unittest.main()
