"""Small, audited subset of the official DeepSeek-V4 message encoding.

The PoC needs a single user turn before it needs the reference encoder's
760-line tool/message surface.  This module implements that exact subset and
fails closed on unsupported modes instead of silently inventing a template.
"""

from __future__ import annotations


BOS_TOKEN = "<｜begin▁of▁sentence｜>"
EOS_TOKEN = "<｜end▁of▁sentence｜>"
USER_TOKEN = "<｜User｜>"
ASSISTANT_TOKEN = "<｜Assistant｜>"
THINK_START = "<think>"
THINK_END = "</think>"

REASONING_EFFORT_PREFIXES = {
    "low": "",
    "high": (
        "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
        "You MUST be very thorough in your thinking and comprehensively "
        "decompose the problem to resolve the root cause, rigorously "
        "stress-testing your logic against all potential paths, edge cases, "
        "and adversarial scenarios.\n"
        "Explicitly write out your entire deliberation process, documenting "
        "every intermediate step, considered alternative, and rejected "
        "hypothesis to ensure absolutely no assumption is left unchecked.\n\n"
    ),
    "max": (
        "Reasoning Effort: Beyond maximum — exhaustive, relentless, and "
        "uncompromising.\n"
        "You MUST reason with the utmost depth and rigor, leaving absolutely "
        "nothing to chance: exhaustively decompose the problem into its most "
        "fundamental components, trace every causal chain to its root, and "
        "resolve the underlying cause rather than any surface symptom.\n"
        "Do not stop reasoning until you have independently verified the "
        "solution from multiple angles and are certain that no assumption "
        "remains unchecked and no error remains undiscovered.\n\n"
    ),
}


def encode_user_prompt(
    content: str,
    *,
    thinking_mode: str = "chat",
    reasoning_effort: str = "low",
) -> str:
    """Encode one user message exactly like official ``encode_messages``."""

    if not isinstance(content, str) or not content:
        raise ValueError("content must be non-empty text")
    if thinking_mode not in {"chat", "thinking"}:
        raise ValueError("thinking_mode must be 'chat' or 'thinking'")
    try:
        prefix = REASONING_EFFORT_PREFIXES[reasoning_effort]
    except KeyError as exc:
        raise ValueError("reasoning_effort must be low, high, or max") from exc
    if thinking_mode == "chat":
        prefix = ""
    assistant_transition = THINK_START if thinking_mode == "thinking" else THINK_END
    return (
        BOS_TOKEN
        + prefix
        + USER_TOKEN
        + content
        + ASSISTANT_TOKEN
        + assistant_transition
    )


__all__ = [
    "ASSISTANT_TOKEN",
    "BOS_TOKEN",
    "EOS_TOKEN",
    "REASONING_EFFORT_PREFIXES",
    "THINK_END",
    "THINK_START",
    "USER_TOKEN",
    "encode_user_prompt",
]
