from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from scripts.qwen38_markov_bootstrap import run


class Qwen38MarkovBootstrapTests(unittest.TestCase):
    def test_receipt_import_is_read_only_atomic_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipts = root / "receipts"
            receipts.mkdir()
            first = receipts / "first.json"
            second = receipts / "second.json"
            malformed = receipts / "malformed.json"
            first.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "prompt_token_ids": [1, 2],
                                "generated_token_ids": [3, 4],
                            },
                            {
                                "prompt_token_ids": [1, 2],
                                "generated_token_ids": [3, 4],
                            },
                            {"generated_token_ids": [True, 9]},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            second.write_text(
                json.dumps({"generated_token_ids": [5, 6, 7, 8]}),
                encoding="utf-8",
            )
            malformed.write_text("{", encoding="utf-8")
            originals = {
                path: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in (first, second, malformed)
            }
            state = root / "state" / "markov.bin"

            def args(*, dry_run: bool = False):
                return SimpleNamespace(
                    state=state,
                    root=[receipts],
                    vocab_size=32,
                    proposal_width=3,
                    min_tokens=4,
                    max_file_mb=1.0,
                    limit=None,
                    dry_run=dry_run,
                )

            dry = run(args(dry_run=True))
            self.assertEqual(dry["episodes_extracted"], 2)
            self.assertEqual(dry["episodes_pending"], 2)
            self.assertEqual(dry["episodes_imported"], 0)
            self.assertFalse(state.exists())

            imported = run(args())
            self.assertEqual(imported["files_scanned"], 2)
            self.assertEqual(imported["files_rejected"], 1)
            self.assertEqual(imported["episodes_imported"], 2)
            self.assertEqual(imported["tokens_imported"], 8)
            self.assertEqual(imported["metrics"]["episode_count"], 2)
            self.assertEqual(imported["metrics"]["imported_episode_count"], 2)
            self.assertEqual(imported["metrics"]["learned_tokens"], 8)
            self.assertEqual(imported["metrics"]["updates"], 2)

            repeated = run(args())
            self.assertEqual(repeated["episodes_pending"], 0)
            self.assertEqual(repeated["episodes_imported"], 0)
            self.assertEqual(repeated["imported_episode_digests"], 2)
            self.assertEqual(repeated["metrics"]["episode_count"], 2)
            self.assertFalse(Path(f"{state}.imports.json").exists())
            for path, (payload, mtime_ns) in originals.items():
                self.assertEqual(path.read_bytes(), payload)
                self.assertEqual(path.stat().st_mtime_ns, mtime_ns)


if __name__ == "__main__":
    unittest.main()
