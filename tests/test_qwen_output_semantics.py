from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest

from immer.runtimes.qwen3_8.output_semantics import (
    QwenOutputSemantics,
    QwenSemanticReplayKey,
    parse_semantic_replay_receipt,
    semantic_replay_key_for_prompt,
    semantic_replay_receipt,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _semantics() -> QwenOutputSemantics:
    return QwenOutputSemantics(
        repo_id="Qwen/Qwen3.8-27B",
        revision="1" * 40,
        q4_manifest_file_sha256=_sha("q4-manifest"),
        q4_native_abi=5,
        q4_bank_codec_abi=2,
        tokenizer_sha256=_sha("tokenizer"),
        compute_dtype="auto",
        max_context_tokens=2048,
        max_prompt_tokens=1024,
        max_new_tokens=64,
        eos_token_ids=(1, 2),
        mlp_page_route_width=192,
        mlp_page_width_actions=(96, 128, 160, 192),
        mlp_page_energy_coverage=0.995,
    )


class QwenOutputSemanticsTests(unittest.TestCase):
    def test_roundtrip_and_prompt_key_bind_every_output_input(self) -> None:
        semantics = _semantics()
        restored = QwenOutputSemantics.from_document(semantics.to_document())
        self.assertEqual(restored, semantics)
        key = semantic_replay_key_for_prompt(
            semantics,
            question="hello",
            rendered_prompt="<user>hello</user>",
            rendered_prompt_token_sha256=_sha("tokens"),
            system_prompt="system",
        )
        receipt = semantic_replay_receipt(semantics, key)
        self.assertEqual(
            parse_semantic_replay_receipt(receipt),
            (semantics, key),
        )

        for changed in (
            replace(semantics, q4_native_abi=6),
            replace(semantics, q4_manifest_file_sha256=_sha("other-q4")),
            replace(semantics, tokenizer_sha256=_sha("other-tokenizer")),
            replace(semantics, max_new_tokens=65),
            replace(semantics, mlp_page_route_width=160,
                    mlp_page_width_actions=(80, 107, 134, 160)),
        ):
            with self.subTest(semantics=changed):
                self.assertNotEqual(changed.sha256, semantics.sha256)

        for field, value in (
            ("question_sha256", _sha("other-question")),
            ("rendered_prompt_sha256", _sha("other-prompt")),
            ("rendered_prompt_token_sha256", _sha("other-tokens")),
            ("system_prompt_sha256", _sha("other-system")),
        ):
            with self.subTest(field=field):
                self.assertNotEqual(
                    replace(key, **{field: value}).sha256,
                    key.sha256,
                )

    def test_tamper_and_cross_authority_receipts_are_rejected(self) -> None:
        semantics = _semantics()
        key = QwenSemanticReplayKey(
            output_semantics_sha256=semantics.sha256,
            question_sha256=_sha("question"),
            rendered_prompt_sha256=_sha("prompt"),
            rendered_prompt_token_sha256=_sha("tokens"),
            system_prompt_sha256=_sha("system"),
        )
        receipt = semantic_replay_receipt(semantics, key)
        tampered = {**receipt, "sha256": "f" * 64}
        with self.assertRaises(ValueError):
            parse_semantic_replay_receipt(tampered)
        with self.assertRaisesRegex(ValueError, "another output authority"):
            semantic_replay_receipt(
                replace(semantics, max_new_tokens=65),
                key,
            )


if __name__ == "__main__":
    unittest.main()
