from __future__ import annotations

import unittest

from immer.intent import classify


class IntentTests(unittest.TestCase):
    def test_canonical_word_arithmetic_reaches_exact_cascade(self) -> None:
        for text in (
            "three plus five is",
            "seven less two is",
            "four times three is",
            "drei plus fünf",
            "zwei mal sechs",
            "eight z3sum five",
        ):
            with self.subTest(text=text):
                self.assertEqual(classify(text).kind, "MATH")

    def test_non_arithmetic_words_stay_chat(self) -> None:
        for text in ("what is love?", "three kings met five queens", "plus ça change"):
            with self.subTest(text=text):
                self.assertEqual(classify(text).kind, "CHAT")


if __name__ == "__main__":
    unittest.main()
