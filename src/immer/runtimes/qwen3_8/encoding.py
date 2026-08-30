"""Local, exact text encoding for the pinned Qwen3.8 checkpoint."""

from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import Sequence


END_OF_TEXT_TOKEN = "<|endoftext|>"
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"

END_OF_TEXT_TOKEN_ID = 248044
IM_START_TOKEN_ID = 248045
IM_END_TOKEN_ID = 248046

_REQUIRED_SPECIAL_TOKENS = (
    END_OF_TEXT_TOKEN,
    IM_START_TOKEN,
    IM_END_TOKEN,
)
_OFFICIAL_SPECIAL_TOKEN_IDS = {
    END_OF_TEXT_TOKEN: END_OF_TEXT_TOKEN_ID,
    IM_START_TOKEN: IM_START_TOKEN_ID,
    IM_END_TOKEN: IM_END_TOKEN_ID,
}


class Qwen38EncodingError(ValueError):
    """A local tokenizer cannot satisfy the Qwen3.8 text contract."""


class Qwen38Tokenizer:
    """Thin ``tokenizers`` wrapper with no model or network side effects.

    The default validates the three control-token IDs that are part of the
    pinned Qwen3.8 text protocol.  Small formula tests may use a tokenizer with
    compact IDs by explicitly selecting ``require_official=False``; the three
    control tokens must still exist and encode atomically.
    """

    def __init__(
        self,
        path: str | Path,
        require_official: bool = True,
    ) -> None:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise Qwen38EncodingError(f"tokenizer JSON is not a file: {source}")
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - depends on installation
            raise Qwen38EncodingError(
                "Qwen3.8 text encoding requires the 'tokenizers' package"
            ) from exc
        try:
            backend = Tokenizer.from_file(str(source))
        except Exception as exc:
            raise Qwen38EncodingError(
                f"cannot load local tokenizer JSON: {source}"
            ) from exc

        self.path = source
        self.backend = backend
        self._validate_special_tokens(require_official=require_official)

    def _validate_special_tokens(self, *, require_official: bool) -> None:
        added_tokens = self.backend.get_added_tokens_decoder()
        for token in _REQUIRED_SPECIAL_TOKENS:
            token_id = self.backend.token_to_id(token)
            if token_id is None:
                raise Qwen38EncodingError(
                    f"tokenizer is missing required control token {token!r}"
                )
            registered = added_tokens.get(int(token_id))
            if (
                registered is None
                or registered.content != token
                or not registered.special
            ):
                raise Qwen38EncodingError(
                    f"control token {token!r} must be registered as special"
                )
            encoded = tuple(
                int(value)
                for value in self.backend.encode(
                    token,
                    add_special_tokens=False,
                ).ids
            )
            if encoded != (int(token_id),):
                raise Qwen38EncodingError(
                    f"control token {token!r} is not encoded atomically"
                )
            if require_official:
                expected = _OFFICIAL_SPECIAL_TOKEN_IDS[token]
                if int(token_id) != expected:
                    raise Qwen38EncodingError(
                        f"control token {token!r} must have pinned ID {expected}, "
                        f"got {token_id}"
                    )
                if self.backend.id_to_token(expected) != token:
                    raise Qwen38EncodingError(
                        f"pinned token ID {expected} does not decode to {token!r}"
                    )

    def encode(self, text: str) -> tuple[int, ...]:
        """Encode text exactly, without adding BOS/EOS or a post-processor."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return tuple(
            int(value)
            for value in self.backend.encode(
                text,
                add_special_tokens=False,
            ).ids
        )

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode generated text while omitting registered special tokens."""

        return self.backend.decode(
            list(_coerce_token_ids(token_ids)),
            skip_special_tokens=True,
        )

    def token_pieces(self, token_ids: Sequence[int]) -> tuple[str, ...]:
        """Decode each ID independently, retaining registered special tokens."""

        return tuple(
            self.backend.decode([token_id], skip_special_tokens=False)
            for token_id in _coerce_token_ids(token_ids)
        )

    @staticmethod
    def render_no_thinking_prompt(system: str, user: str) -> str:
        """Render the official plain-text system/user generation prompt.

        This is the exact ``enable_thinking=false``, ``tools=None`` and
        ``add_generation_prompt=true`` branch of the pinned
        ``chat_template.jinja``.  Jinja's ``trim`` filter is mirrored by
        ``str.strip`` for both message bodies.
        """

        return Qwen38Tokenizer.render_no_thinking_messages(
            system,
            (("user", user),),
        )

    @staticmethod
    def render_no_thinking_messages(
        system: str,
        messages: Sequence[tuple[str, str]],
    ) -> str:
        """Render exact text-only no-thinking chat with preserved empty history tags."""

        if not isinstance(system, str):
            raise TypeError("system must be a string")
        if isinstance(messages, (str, bytes, bytearray)):
            raise TypeError("messages must be a sequence of role/content pairs")
        try:
            rows = tuple(messages)
        except TypeError as exc:
            raise TypeError(
                "messages must be a sequence of role/content pairs"
            ) from exc
        if not rows:
            raise ValueError("messages must contain a user query")

        expected_role = "user"
        normalized: list[tuple[str, str]] = []
        for row in rows:
            if (
                not isinstance(row, tuple)
                or len(row) != 2
                or not isinstance(row[0], str)
                or not isinstance(row[1], str)
            ):
                raise TypeError("messages must contain role/content text pairs")
            role, content = row
            if role != expected_role:
                raise ValueError(
                    f"messages must alternate user/assistant; expected {expected_role}"
                )
            content = content.strip()
            if not content:
                raise ValueError("message content must be non-empty text")
            normalized.append((role, content))
            expected_role = "assistant" if role == "user" else "user"
        if normalized[-1][0] != "user":
            raise ValueError("the final message must be the active user query")

        system = system.strip()
        rendered = ""
        if system:
            rendered += f"{IM_START_TOKEN}system\n{system}{IM_END_TOKEN}\n"
        for role, content in normalized:
            rendered += f"{IM_START_TOKEN}{role}\n"
            if role == "assistant":
                rendered += f"<think>\n\n</think>\n\n{content}{IM_END_TOKEN}\n"
            else:
                rendered += f"{content}{IM_END_TOKEN}\n"
        rendered += f"{IM_START_TOKEN}assistant\n<think>\n\n</think>\n\n"
        return rendered


def _coerce_token_ids(token_ids: Sequence[int]) -> tuple[int, ...]:
    if isinstance(token_ids, (str, bytes, bytearray)):
        raise TypeError("token_ids must be a sequence of integers")
    try:
        values = tuple(token_ids)
    except TypeError as exc:
        raise TypeError("token_ids must be a sequence of integers") from exc

    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("token_ids must contain only integers")
        integer = int(value)
        if integer < 0:
            raise ValueError("token_ids must be non-negative")
        result.append(integer)
    return tuple(result)


__all__ = [
    "END_OF_TEXT_TOKEN_ID",
    "IM_END_TOKEN_ID",
    "IM_START_TOKEN_ID",
    "Qwen38EncodingError",
    "Qwen38Tokenizer",
]
