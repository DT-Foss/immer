from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from immer.intent import classify
from immer.memory import SpanStore


class IntentTests(unittest.TestCase):
    def test_teach(self) -> None:
        intent = classify("merke: David mag Kaffee mit Hafermilch")
        self.assertEqual(intent.kind, "TEACH")
        self.assertEqual(intent.payload, "David mag Kaffee mit Hafermilch")

    def test_recall(self) -> None:
        intent = classify("was weißt du über Kaffee?")
        self.assertEqual(intent.kind, "RECALL")
        self.assertEqual(intent.payload, "Kaffee?")

    def test_math_with_digits_and_signal(self) -> None:
        self.assertEqual(classify("John has 5 apples. How many apples?").kind, "MATH")
        self.assertEqual(classify("berechne 23 + 19").kind, "MATH")

    def test_status_question(self) -> None:
        self.assertEqual(classify("wie geht es dir?").kind, "STATUS")

    def test_plain_chat_falls_through(self) -> None:
        self.assertEqual(classify("erzähl mir was Schönes").kind, "CHAT")

    def test_digits_alone_are_not_math(self) -> None:
        self.assertEqual(classify("Ich wohne in Haus Nummer 12, schön oder?").kind, "CHAT")


class SpanStoreTests(unittest.TestCase):
    def test_teach_and_recall_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpanStore(Path(tmp) / "memory.json")
            store.teach("David mag Kaffee mit Hafermilch")
            store.teach("Das Projekt heißt IMMER")
            hits = store.recall("Kaffee")
            self.assertEqual(len(hits), 1)
            self.assertIn("Hafermilch", hits[0]["text"])

    def test_memory_survives_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.json"
            first = SpanStore(path)
            first.teach("Der Donor ist ein 27B Qwen")
            second = SpanStore(path)
            self.assertEqual(second.count(), 1)
            self.assertTrue(second.recall("Donor"))

    def test_recall_without_hits_is_empty_not_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpanStore(Path(tmp) / "memory.json")
            self.assertEqual(store.recall("nichts"), [])


if __name__ == "__main__":
    unittest.main()
