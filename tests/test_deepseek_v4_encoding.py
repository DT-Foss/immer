from __future__ import annotations

import unittest


class DeepSeekV4EncodingTests(unittest.TestCase):
    def test_single_user_chat_matches_published_transition(self) -> None:
        from immer.runtimes.deepseek_v4.encoding import encode_user_prompt

        self.assertEqual(
            encode_user_prompt("hello", thinking_mode="chat"),
            "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>",
        )

    def test_single_user_thinking_matches_published_example(self) -> None:
        from immer.runtimes.deepseek_v4.encoding import encode_user_prompt

        self.assertEqual(
            encode_user_prompt("What is 2+2?", thinking_mode="thinking"),
            "<｜begin▁of▁sentence｜><｜User｜>What is 2+2?<｜Assistant｜><think>",
        )

    def test_reasoning_effort_only_changes_thinking_prefix(self) -> None:
        from immer.runtimes.deepseek_v4.encoding import encode_user_prompt

        chat = encode_user_prompt("x", thinking_mode="chat", reasoning_effort="max")
        thinking = encode_user_prompt(
            "x", thinking_mode="thinking", reasoning_effort="max"
        )
        self.assertNotIn("Reasoning Effort", chat)
        self.assertIn("Reasoning Effort", thinking)


if __name__ == "__main__":
    unittest.main()
