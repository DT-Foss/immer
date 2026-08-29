from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from scripts.qwen38_markov_bootstrap import run
from immer.runtimes.qwen3_8.markov_draft import MarkovDraftState


class Qwen38MarkovBootstrapTests(unittest.TestCase):
    def test_structured_identity_does_not_flatten_the_prompt_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipts = root / "receipts"
            receipts.mkdir()
            (receipts / "a.json").write_text(
                json.dumps(
                    {
                        "prompt_token_ids": [1, 2],
                        "generated_token_ids": [3, 4],
                    }
                ),
                encoding="utf-8",
            )
            (receipts / "b.json").write_text(
                json.dumps({"generated_token_ids": [1, 2, 3, 4]}),
                encoding="utf-8",
            )
            state = root / "state" / "markov.bin"
            args = SimpleNamespace(
                state=state,
                root=[receipts],
                vocab_size=32,
                proposal_width=3,
                min_tokens=4,
                max_file_mb=1.0,
                limit=None,
                dry_run=False,
            )

            result = run(args)

            self.assertEqual(result["episodes_extracted"], 2)
            self.assertEqual(result["episodes_imported"], 2)
            learned = MarkovDraftState.from_bytes(state.read_bytes())
            self.assertEqual(learned.episode_prompt_lengths, (2, None))

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
            learned = MarkovDraftState.from_bytes(state.read_bytes())
            self.assertEqual(learned.episode_prompt_lengths, (2, None))

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
