from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def _tiny_tokenizer(
    path: Path,
    *,
    include_im_end: bool = True,
    controls_are_special: bool = True,
) -> None:
    from tokenizers import AddedToken, Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit

    tokenizer = Tokenizer(
        WordLevel(
            vocab={"[UNK]": 0, "hello": 1, "world": 2},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = WhitespaceSplit()
    controls = ["<|endoftext|>", "<|im_start|>"]
    if include_im_end:
        controls.append("<|im_end|>")
    added = [AddedToken(token, special=controls_are_special) for token in controls]
    if controls_are_special:
        tokenizer.add_special_tokens(added)
    else:
        tokenizer.add_tokens(added)
    tokenizer.save(str(path))


class _Encoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _OfficialBackend:
    _token_to_id = {
        "<|endoftext|>": 248044,
        "<|im_start|>": 248045,
        "<|im_end|>": 248046,
    }

    def token_to_id(self, token: str) -> int | None:
        return self._token_to_id.get(token)

    def id_to_token(self, token_id: int) -> str | None:
        for token, candidate in self._token_to_id.items():
            if candidate == token_id:
                return token
        return None

    def get_added_tokens_decoder(self) -> dict[int, SimpleNamespace]:
        return {
            token_id: SimpleNamespace(content=token, special=True)
            for token, token_id in self._token_to_id.items()
        }

    def encode(self, text: str, *, add_special_tokens: bool) -> _Encoding:
        if add_special_tokens:
            raise AssertionError("the wrapper must not request special tokens")
        token_id = self.token_to_id(text)
        return _Encoding([] if token_id is None else [token_id])

    def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str:
        pieces = [self.id_to_token(token_id) or f"token-{token_id}" for token_id in ids]
        if skip_special_tokens:
            pieces = [piece for piece in pieces if not piece.startswith("<|")]
        return "".join(pieces)


class Qwen38EncodingTests(unittest.TestCase):
    def test_tiny_fixture_requires_explicit_non_official_mode(self) -> None:
        from immer.runtimes.qwen3_8.encoding import (
            Qwen38EncodingError,
            Qwen38Tokenizer,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.json"
            _tiny_tokenizer(path)
            with self.assertRaisesRegex(Qwen38EncodingError, "pinned ID 248044"):
                Qwen38Tokenizer(path)

            tokenizer = Qwen38Tokenizer(path, require_official=False)
            self.assertEqual(tokenizer.encode("hello world"), (1, 2))
            self.assertEqual(tokenizer.decode((1, 2)), "hello world")
            self.assertEqual(tokenizer.token_pieces((1, 2)), ("hello", "world"))
            im_start_id = tokenizer.backend.token_to_id("<|im_start|>")
            self.assertIsNotNone(im_start_id)
            self.assertEqual(
                tokenizer.token_pieces((int(im_start_id),)),
                ("<|im_start|>",),
            )
            self.assertEqual(tokenizer.decode((int(im_start_id),)), "")

    def test_official_special_id_contract_accepts_only_exact_bidirectional_map(
        self,
    ) -> None:
        from immer.runtimes.qwen3_8.encoding import Qwen38Tokenizer

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.json"
            path.write_text("{}", encoding="utf-8")
            with mock.patch(
                "tokenizers.Tokenizer.from_file",
                return_value=_OfficialBackend(),
            ):
                tokenizer = Qwen38Tokenizer(path)

        self.assertEqual(tokenizer.encode("<|im_end|>"), (248046,))
        self.assertEqual(tokenizer.token_pieces((248045,)), ("<|im_start|>",))

    def test_missing_control_token_is_rejected_even_for_tiny_fixture(self) -> None:
        from immer.runtimes.qwen3_8.encoding import (
            Qwen38EncodingError,
            Qwen38Tokenizer,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.json"
            _tiny_tokenizer(path, include_im_end=False)
            with self.assertRaisesRegex(Qwen38EncodingError, "missing.*im_end"):
                Qwen38Tokenizer(path, require_official=False)

    def test_control_tokens_must_actually_be_registered_as_special(self) -> None:
        from immer.runtimes.qwen3_8.encoding import (
            Qwen38EncodingError,
            Qwen38Tokenizer,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.json"
            _tiny_tokenizer(path, controls_are_special=False)
            with self.assertRaisesRegex(Qwen38EncodingError, "registered as special"):
                Qwen38Tokenizer(path, require_official=False)

    def test_no_thinking_prompt_matches_official_plain_text_template(self) -> None:
        from immer.runtimes.qwen3_8.encoding import Qwen38Tokenizer

        rendered = Qwen38Tokenizer.render_no_thinking_prompt(
            "  Follow the evidence.\n",
            "\n What is 2 + 2?  ",
        )

        self.assertEqual(
            rendered,
            "<|im_start|>system\n"
            "Follow the evidence.<|im_end|>\n"
            "<|im_start|>user\n"
            "What is 2 + 2?<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n",
        )

    def test_blank_system_matches_template_branch_without_system_message(self) -> None:
        from immer.runtimes.qwen3_8.encoding import Qwen38Tokenizer

        self.assertEqual(
            Qwen38Tokenizer.render_no_thinking_prompt(" \n", " hello "),
            "<|im_start|>user\n"
            "hello<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n",
        )

    def test_no_thinking_messages_match_official_multi_turn_template(self) -> None:
        from immer.runtimes.qwen3_8.encoding import Qwen38Tokenizer

        rendered = Qwen38Tokenizer.render_no_thinking_messages(
            "  Stay concise. ",
            (
                ("user", "My code is ORBIT-7."),
                ("assistant", "Understood."),
                ("user", "What was my code?"),
            ),
        )

        self.assertEqual(
            rendered,
            "<|im_start|>system\n"
            "Stay concise.<|im_end|>\n"
            "<|im_start|>user\n"
            "My code is ORBIT-7.<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
            "Understood.<|im_end|>\n"
            "<|im_start|>user\n"
            "What was my code?<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n",
        )

    def test_no_thinking_messages_reject_incomplete_or_misordered_turns(self) -> None:
        from immer.runtimes.qwen3_8.encoding import Qwen38Tokenizer

        with self.assertRaisesRegex(ValueError, "final message"):
            Qwen38Tokenizer.render_no_thinking_messages(
                "",
                (("user", "hello"), ("assistant", "hi")),
            )
        with self.assertRaisesRegex(ValueError, "expected user"):
            Qwen38Tokenizer.render_no_thinking_messages(
                "",
                (("assistant", "hi"),),
            )
        with self.assertRaisesRegex(ValueError, "non-empty"):
            Qwen38Tokenizer.render_no_thinking_messages("", (("user", " "),))

    def test_invalid_paths_and_public_input_types_fail_closed(self) -> None:
        from immer.runtimes.qwen3_8.encoding import (
            Qwen38EncodingError,
            Qwen38Tokenizer,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.json"
            with self.assertRaisesRegex(Qwen38EncodingError, "not a file"):
                Qwen38Tokenizer(path)
            _tiny_tokenizer(path)
            tokenizer = Qwen38Tokenizer(path, require_official=False)

        with self.assertRaisesRegex(TypeError, "text must be a string"):
            tokenizer.encode(7)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "sequence of integers"):
            tokenizer.decode("12")  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "only integers"):
            tokenizer.token_pieces((1, True))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            tokenizer.decode((-1,))
        with self.assertRaisesRegex(TypeError, "system must be a string"):
            tokenizer.render_no_thinking_prompt(None, "hello")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
