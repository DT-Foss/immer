from __future__ import annotations

import hashlib
import json
import tempfile
import unittest

import numpy as np

from immer.runtimes.ooe.bvn_crystals import (
    BirkhoffCrystalBank,
    BirkhoffCrystalIntegrityError,
    BirkhoffCrystalReceipt,
)
from immer.runtimes.ooe.compute_crystals import ComputeCrystalBank
from immer.runtimes.ooe.identity import canonical_json_bytes


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _kernel(seed: int, *, dimension: int = 6, components: int = 14) -> np.ndarray:
    generator = np.random.default_rng(seed)
    raw = generator.random(components)
    weights = raw / raw.sum()
    result = np.zeros((dimension, dimension), dtype=np.float64)
    for weight in weights:
        permutation = generator.permutation(dimension)
        result[np.arange(dimension), permutation] += weight
    return result


class BirkhoffCrystalBankTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.bank = ComputeCrystalBank(self.temporary.name)
        self.birkhoff = BirkhoffCrystalBank(self.bank)

    def test_constructive_basis_reproduces_markov_crystal_on_future_inputs(
        self,
    ) -> None:
        kernel = _kernel(20260826)
        publication = self.birkhoff.publish(
            kernel,
            verifier_sha256=_hash("bvn-verifier"),
            evidence_sha256s=tuple(
                sorted((_hash("measurement-a"), _hash("measurement-b")))
            ),
        )
        receipt = publication.receipt
        self.assertLessEqual(len(receipt.atom_crystal_sha256s), (6 - 1) ** 2 + 1)
        self.assertEqual(
            BirkhoffCrystalReceipt.from_bytes(receipt.to_bytes()), receipt
        )
        future = np.random.default_rng(99).normal(size=(128, 6)).astype(np.float64)
        weighted = self.birkhoff.apply_atoms(receipt.sha256, future)
        markov = self.bank.restore_crystal(receipt.markov_crystal_sha256)
        np.testing.assert_allclose(weighted, markov.apply(future), rtol=0, atol=1e-12)
        np.testing.assert_allclose(weighted, future @ kernel, rtol=0, atol=1e-12)

    def test_atom_indices_use_row_vector_inverse_permutation_semantics(self) -> None:
        permutation = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        receipt = self.birkhoff.publish(
            permutation,
            verifier_sha256=_hash("verifier"),
            evidence_sha256s=(_hash("evidence"),),
        ).receipt
        atom = self.bank.restore_crystal(receipt.atom_crystal_sha256s[0])
        value = np.array([[10.0, 20.0, 30.0]], dtype=np.float64)
        np.testing.assert_array_equal(atom.apply(value), value @ permutation)

    def test_publish_is_idempotent_and_non_ds_placebo_is_rejected(self) -> None:
        kernel = _kernel(7, dimension=4, components=7)
        arguments = {
            "verifier_sha256": _hash("verifier"),
            "evidence_sha256s": (_hash("evidence"),),
            "tolerance": 1e-10,
        }
        first = self.birkhoff.publish(kernel, **arguments)
        second = self.birkhoff.publish(kernel, **arguments)
        self.assertEqual(first.receipt, second.receipt)
        self.assertFalse(second.markov_publication.manifest_changed)
        self.assertTrue(
            all(not publication.manifest_changed for publication in second.atom_publications)
        )
        self.assertFalse(second.decomposition_state_publication.changed)
        self.assertFalse(second.receipt_state_publication.changed)

        placebo = kernel.copy()
        placebo[0, 0] += 0.2
        with self.assertRaises(BirkhoffCrystalIntegrityError):
            self.birkhoff.publish(placebo, **arguments)

    def test_resealed_receipt_tamper_and_missing_basis_fail_closed(self) -> None:
        receipt = self.birkhoff.publish(
            _kernel(11, dimension=4, components=5),
            verifier_sha256=_hash("verifier"),
            evidence_sha256s=(_hash("evidence"),),
        ).receipt
        document = json.loads(receipt.to_bytes())
        document["body"]["markov_crystal_sha256"] = "0" * 64
        document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(document["body"])
        ).hexdigest()
        forged = BirkhoffCrystalReceipt.from_bytes(canonical_json_bytes(document))
        self.assertNotEqual(forged.sha256, receipt.sha256)

        self.bank.store.publish_state(
            self.birkhoff.receipt_state_name(receipt.sha256),
            canonical_json_bytes(document),
            expected_sha256=receipt.sha256,
        )
        with self.assertRaises(BirkhoffCrystalIntegrityError):
            self.birkhoff.restore(receipt.sha256)


if __name__ == "__main__":
    unittest.main()
