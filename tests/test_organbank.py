from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from immer.artifacts import ArtifactBootstrapError
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

    def test_verify_all_returns_every_digest_checked_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.organ"
            second = root / "second.organ"
            first.write_bytes(b"one")
            second.write_bytes(b"two")
            manifest = root / "bank.json"
            manifest.write_text(json.dumps({"organs": [
                {
                    "name": "first",
                    "capability": "cap.one",
                    "group": "R,+",
                    "artifact": first.name,
                    "sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
                },
                {
                    "name": "second",
                    "capability": "cap.two",
                    "group": "R,·",
                    "artifact": second.name,
                    "sha256": hashlib.sha256(second.read_bytes()).hexdigest(),
                },
            ]}), encoding="utf-8")

            verified = OrganBank.from_manifest(manifest).verify_all()

            self.assertEqual(
                verified,
                {"first": first.resolve(), "second": second.resolve()},
            )

    def test_explicit_root_preserves_safe_organ_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_root = root / "bundle"
            artifact = artifact_root / "organs" / "arith.organ"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"nested-exact-organ")
            manifest = root / "bank.json"
            manifest.write_text(
                json.dumps(
                    {
                        "organs": [
                            {
                                "name": "arith-dual",
                                "capability": "arithmetic",
                                "group": "R,+",
                                "artifact": "organs/arith.organ",
                                "sha256": hashlib.sha256(
                                    artifact.read_bytes()
                                ).hexdigest(),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            bank = OrganBank.from_manifest(manifest, artifact_root=artifact_root)

            self.assertEqual(bank.verify("arith-dual"), artifact.resolve())

    def test_explicit_root_rejects_manifest_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "bank.json"
            manifest.write_text(
                json.dumps(
                    {
                        "organs": [
                            {
                                "name": "bad",
                                "capability": "bad",
                                "group": "bad",
                                "artifact": "../outside.organ",
                                "sha256": "0" * 64,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ArtifactBootstrapError, "traverses outside"):
                OrganBank.from_manifest(manifest, artifact_root=root / "bundle")


if __name__ == "__main__":
    unittest.main()
