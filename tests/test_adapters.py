from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from immer.adapters import FlcaAdapter, OrganBankAdapter
from immer.contracts import BackendStatus


class AdapterTests(unittest.TestCase):
    def test_flca_holds_unknown_family(self) -> None:
        result = FlcaAdapter().route("unknown")
        self.assertEqual(result.status, BackendStatus.HELD)

    def test_organ_manifest_digest_is_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "organ.json"
            manifest.write_text("{}\n", encoding="utf-8")
            result = OrganBankAdapter().load_manifest(manifest, expected_sha256="wrong")
            self.assertEqual(result.status, BackendStatus.HELD)


if __name__ == "__main__":
    unittest.main()
