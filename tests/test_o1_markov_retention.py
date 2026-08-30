from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from immer.runtimes.o1_state.markov_retention import (
    O1MarkovRetention,
    O1MarkovRetentionError,
)


class O1MarkovRetentionTests(unittest.TestCase):
    def test_confirmed_episode_score_survives_restart_and_advances(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "retention.json"
            first_tokens = (7, 8, 9, 10)
            first = O1MarkovRetention(
                path,
                vocab_size=64,
                tokenizer_sha256="a" * 64,
            )

            first_priority = first.score(
                first_tokens,
                "Light travels faster than sound.",
            )

            self.assertGreater(first_priority, 0.0)
            self.assertEqual(first.priority(first_tokens), first_priority)
            self.assertEqual(first.metrics()["sequence"], 1)
            self.assertTrue(path.is_file())
            document = json.loads(path.read_text())
            sidecar = document["body"]["stream"]["sidecar"]
            self.assertTrue(path.with_name(sidecar["name"]).is_file())

            restarted = O1MarkovRetention(
                path,
                vocab_size=64,
                tokenizer_sha256="a" * 64,
            )
            self.assertEqual(restarted.priority(first_tokens), first_priority)
            second_priority = restarted.score(
                (11, 12, 13),
                "Water droplets disperse sunlight.",
            )
            self.assertGreater(second_priority, 0.0)
            self.assertEqual(restarted.metrics()["sequence"], 2)
            self.assertEqual(restarted.metrics()["retained_priorities"], 2)

            with self.assertRaisesRegex(
                O1MarkovRetentionError,
                "binding",
            ):
                O1MarkovRetention(
                    path,
                    vocab_size=64,
                    tokenizer_sha256="b" * 64,
                )

    def test_failed_state_write_restores_the_last_committed_organism(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "retention.json"
            retention = O1MarkovRetention(
                path,
                vocab_size=64,
                tokenizer_sha256="a" * 64,
            )
            first = (1, 2, 3)
            second = (4, 5, 6)
            retention.score(first, "first confirmed answer")
            before = retention.metrics()

            with mock.patch.object(
                retention,
                "_write_locked",
                side_effect=OSError("disk full"),
            ):
                with self.assertRaisesRegex(OSError, "disk full"):
                    retention.score(second, "second confirmed answer")

            self.assertEqual(retention.metrics(), before)
            self.assertEqual(retention.priority(second), 1.0)


if __name__ == "__main__":
    unittest.main()
