from __future__ import annotations

import threading
import time
import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from immer.contracts import ExecutionStatus, Request
from immer.runtimes.deepseek_v4.adapter import DeepSeekV4Chat


_TOKENIZER_SHA256 = "a" * 64


@dataclass(frozen=True)
class _Evidence:
    generated_token_ids: tuple[int, ...]
    forward_passes: int = 3


class _Tokenizer:
    sha256 = _TOKENIZER_SHA256

    def __init__(self, *, prompt_ids=(11, 12), decoded="  hello  ") -> None:
        self.prompt_ids = prompt_ids
        self.decoded = decoded
        self.encoded: list[str] = []
        self.decoded_ids: list[tuple[int, ...]] = []

    def encode(self, text: str):
        self.encoded.append(text)
        return self.prompt_ids

    def decode(self, token_ids):
        self.decoded_ids.append(tuple(token_ids))
        return self.decoded


class _Model:
    max_seq_len = 16

    def __init__(
        self,
        *,
        generated=(7, 9),
        evidence=None,
        error: Exception | None = None,
        graft=True,
    ) -> None:
        self.config = SimpleNamespace(vocab_size=32, max_position_embeddings=16)
        self.graft = SimpleNamespace(mode="crsa", alpha=0.05) if graft else None
        self.graft_layer = 6 if graft else None
        self.generated = generated
        self.evidence = (
            _Evidence(tuple(generated)) if evidence is None else evidence
        )
        self.error = error
        self.calls: list[tuple[object, dict[str, object]]] = []
        self.reset_calls: list[bool] = []

    def generate_greedy(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if self.error is not None:
            raise self.error
        return self.generated, self.evidence

    def reset_state(self, *, release=False):
        self.reset_calls.append(release)


def _chat(model=None, tokenizer=None, **kwargs) -> DeepSeekV4Chat:
    options = {
        "max_prompt_tokens": 8,
        "max_new_tokens": 3,
        "max_context_tokens": 12,
        "eos_token_ids": (9,),
    }
    options.update(kwargs)
    return DeepSeekV4Chat(
        _Model() if model is None else model,
        _Tokenizer() if tokenizer is None else tokenizer,
        model_id="deepseek-ai/DeepSeek-V4-Flash-0731",
        revision="7" * 40,
        **options,
    )


class DeepSeekV4ChatTests(unittest.TestCase):
    def test_success_uses_exact_official_envelope_and_records_public_evidence(self) -> None:
        model = _Model()
        tokenizer = _Tokenizer()
        chat = _chat(model, tokenizer)

        result = chat.handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.OK)
        self.assertEqual(result.component, "deepseek-v4.chat")
        self.assertEqual(result.output, "hello")
        self.assertEqual(
            tokenizer.encoded,
            [
                "<｜begin▁of▁sentence｜><｜User｜>hello"
                "<｜Assistant｜></think>"
            ],
        )
        self.assertEqual(tokenizer.decoded_ids, [(7, 9)])
        self.assertEqual(model.calls[0][0], [[11, 12]])
        self.assertEqual(
            model.calls[0][1],
            {
                "max_new_tokens": 3,
                "prefill_tokenwise": True,
                "eos_token_ids": (9,),
                "head_block_rows": 1024,
            },
        )
        self.assertEqual(model.reset_calls, [False])
        self.assertEqual(result.evidence["model"], "deepseek-ai/DeepSeek-V4-Flash-0731")
        self.assertEqual(result.evidence["revision"], "7" * 40)
        self.assertEqual(result.evidence["tokenizer_sha256"], _TOKENIZER_SHA256)
        self.assertEqual(result.evidence["thinking_mode"], "chat")
        self.assertEqual(result.evidence["reasoning_effort"], "low")
        self.assertEqual(result.evidence["graft_mode"], "crsa")
        self.assertEqual(result.evidence["graft_alpha"], 0.05)
        self.assertEqual(result.evidence["graft_layer"], 6)
        self.assertEqual(result.evidence["generated_token_ids"], (7, 9))
        self.assertEqual(result.evidence["generation"]["forward_passes"], 3)
        self.assertNotIn("model_object", result.evidence)

    def test_configurable_assistant_prefix_is_appended_after_the_envelope(self) -> None:
        tokenizer = _Tokenizer()
        chat = _chat(tokenizer=tokenizer, assistant_prefix="Answer:")

        result = chat.handle(Request("chat", "question"))

        self.assertTrue(result.ok)
        self.assertEqual(
            tokenizer.encoded[0],
            "<｜begin▁of▁sentence｜><｜User｜>question"
            "<｜Assistant｜></think>Answer:",
        )

    def test_blank_decode_abstains_and_mapping_evidence_is_copied(self) -> None:
        evidence = {"generated_token_ids": [7, 9], "seconds": 1.25}
        model = _Model(evidence=evidence)
        tokenizer = _Tokenizer(decoded=" \n ")

        result = _chat(model, tokenizer).handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.ABSTAINED)
        self.assertEqual(result.evidence["generation"], evidence)
        self.assertEqual(model.reset_calls, [False])

    def test_request_and_constructor_validation_fail_closed(self) -> None:
        model = _Model()
        chat = _chat(model=model)

        unsupported = chat.handle(Request("exact_math", "1+1"))
        blank = chat.handle(Request("chat", " \n"))

        self.assertIs(unsupported.status, ExecutionStatus.REJECTED)
        self.assertIs(blank.status, ExecutionStatus.REJECTED)
        self.assertFalse(model.calls)
        self.assertFalse(model.reset_calls)
        with self.assertRaisesRegex(ValueError, "tokenizer_sha256"):
            DeepSeekV4Chat(
                _Model(),
                _Tokenizer(),
                model_id="model",
                revision="revision",
                tokenizer_sha256="not-a-digest",
            )
        with self.assertRaisesRegex(ValueError, "prebuilt model context"):
            _chat(max_context_tokens=17)
        with self.assertRaisesRegex(ValueError, "distinct"):
            _chat(eos_token_ids=(9, 9))

    def test_prompt_output_context_and_eos_bounds_return_error_and_release(self) -> None:
        cases = (
            (
                _Model(),
                _Tokenizer(prompt_ids=(1, 2, 3)),
                {"max_prompt_tokens": 2},
                "prompt has",
            ),
            (
                _Model(),
                _Tokenizer(prompt_ids=tuple(range(10))),
                {"max_prompt_tokens": 10, "max_context_tokens": 12},
                "output budget exceeds",
            ),
            (
                _Model(generated=(1, 2, 3, 4)),
                _Tokenizer(),
                {},
                "max_new_tokens",
            ),
            (
                _Model(generated=(9, 7)),
                _Tokenizer(),
                {},
                "after EOS",
            ),
            (
                _Model(generated=(32,)),
                _Tokenizer(),
                {},
                "outside checkpoint vocabulary",
            ),
        )
        for model, tokenizer, overrides, reason in cases:
            with self.subTest(reason=reason):
                result = _chat(model, tokenizer, **overrides).handle(
                    Request("chat", "hello")
                )
                self.assertIs(result.status, ExecutionStatus.ERROR)
                self.assertIn(reason, result.reason or "")
                self.assertEqual(model.reset_calls, [True])

    def test_generation_failure_requests_release_and_cleanup_cannot_mask_it(self) -> None:
        model = _Model(error=RuntimeError("forward broke"))

        result = _chat(model=model).handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.ERROR)
        self.assertIn("forward broke", result.reason or "")
        self.assertEqual(model.reset_calls, [True])

        class BrokenCleanup(_Model):
            def reset_state(self, *, release=False):
                raise RuntimeError("cleanup broke")

        cleanup_model = BrokenCleanup()
        cleanup_result = _chat(model=cleanup_model).handle(Request("chat", "hello"))
        self.assertIs(cleanup_result.status, ExecutionStatus.OK)
        self.assertEqual(cleanup_result.output, "hello")
        self.assertIn("cleanup broke", cleanup_result.evidence["cleanup"]["error"])

        poisoned = BrokenCleanup()
        poisoned_chat = _chat(model=poisoned)
        first = poisoned_chat.handle(Request("chat", "hello"))
        blocked = poisoned_chat.handle(Request("chat", "again"))
        self.assertIs(first.status, ExecutionStatus.OK)
        self.assertIs(blocked.status, ExecutionStatus.ERROR)
        self.assertIn("cleanup remains unavailable", blocked.reason or "")
        self.assertEqual(len(poisoned.calls), 1)

    def test_model_without_release_parameter_is_still_reset(self) -> None:
        class ResetWithoutRelease(_Model):
            def reset_state(self):
                self.reset_calls.append(False)

        model = ResetWithoutRelease(error=RuntimeError("boom"))

        result = _chat(model=model).handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.ERROR)
        self.assertEqual(model.reset_calls, [False])

    def test_mutable_model_access_is_serialized(self) -> None:
        class ConcurrentModel(_Model):
            def __init__(self) -> None:
                super().__init__()
                self.guard = threading.Lock()
                self.active = 0
                self.maximum_active = 0

            def generate_greedy(self, prompt, **kwargs):
                with self.guard:
                    self.active += 1
                    self.maximum_active = max(self.maximum_active, self.active)
                time.sleep(0.03)
                with self.guard:
                    self.active -= 1
                return super().generate_greedy(prompt, **kwargs)

        model = ConcurrentModel()
        chat = _chat(model=model)
        barrier = threading.Barrier(3)
        results = []

        def request() -> None:
            barrier.wait()
            results.append(chat.handle(Request("chat", "hello")))

        threads = [threading.Thread(target=request) for _ in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=2)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.ok for result in results))
        self.assertEqual(model.maximum_active, 1)
        self.assertEqual(model.reset_calls, [False, False])


if __name__ == "__main__":
    unittest.main()
