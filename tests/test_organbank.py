from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from immer.capabilities.organbank import DigestMismatch, OrganBank


class OrganBankTests(unittest.TestCase):
    def test_manifest_resolves_and_verifies_external_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "arith.organ"
            artifact.write_bytes(b"exact-organ")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            manifest = root / "bank.json"
            manifest.write_text(json.dumps({"organs": [{
                "name": "arith-dual",
                "capability": "arithmetic",
                "group": "R,+",
                "artifact": "arith.organ",
                "sha256": digest,
                "metadata": {"crystallized": True},
            }]}), encoding="utf-8")

            bank = OrganBank.from_manifest(manifest)
            descriptor = bank.resolve("arithmetic")
            self.assertEqual(descriptor.name, "arith-dual")
            self.assertEqual(bank.verify("arith-dual"), artifact.resolve())

    def test_digest_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "bad.organ"
            artifact.write_bytes(b"changed")
            manifest = root / "bank.json"
            manifest.write_text(json.dumps({"organs": [{
                "name": "bad",
                "capability": "x",
                "group": "test",
                "artifact": "bad.organ",
                "sha256": "0" * 64,
            }]}), encoding="utf-8")
            bank = OrganBank.from_manifest(manifest)
            with self.assertRaises(DigestMismatch):
                bank.verify("bad")


if __name__ == "__main__":
    unittest.main()
