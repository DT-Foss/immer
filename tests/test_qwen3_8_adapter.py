from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from immer.cli import main
from immer.cognition.fertig import FertigSolver
from immer.cognition.qwen_fertig_chat import QwenFertigChat
from immer.composition import CompositionRoot, compose_runtime
from immer.contracts import ExecutionStatus, Request, Result
from immer.runtimes.qwen3_8.adapter import Qwen38CausalChat, Qwen38ChatError
from immer.runtimes.qwen3_8.encoding import Qwen38Tokenizer


_DIGEST = "a" * 64
_BUNDLE_RECEIPT = {
    "checkpoint_bytes": 55_000_000_000,
    "graph_revision": [1, "e" * 64],
    "kind": "complete-causal-bundle/v1",
    "layout_fingerprint": "b" * 64,
    "manifest_sha256": "c" * 64,
    "shards": 12,
    "shards_sha256": "d" * 64,
    "tensor_bindings": 1199,
    "weights_layout": "flat/v1",
}


class _Tokenizer:
    def __init__(self, *, prompt_ids=(11, 12), decoded=" local answer ") -> None:
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
        generated=(7, 8),
        generation_error: Exception | None = None,
        cleanup_error: Exception | None = None,
    ) -> None:
        self.config = SimpleNamespace(vocab_size=300_000)
        self.generated = tuple(generated)
        self.generation_error = generation_error
        self.cleanup_error = cleanup_error
        self.calls: list[tuple[object, dict[str, object]]] = []
        self.reset_calls: list[bool] = []

    def generate_greedy(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if self.generation_error is not None:
            raise self.generation_error
        prompt_ids = tuple(prompt[0])
        return self.generated, {
            "prompt_token_ids": prompt_ids,
            "generated_token_ids": self.generated,
            "context_mode": "stateful_autoregressive",
            "stateful_cache": True,
            "general_generation": True,
            "prefill_mode": "batched",
            "forward_passes": 3,
            "source_body_bytes": 1234,
            "linear_calls": 99,
            "seconds": 1.25,
            "state_bytes": 456,
            "stopped_on_eos": False,
        }

    def reset_state(self, *, release=False):
        self.reset_calls.append(release)
        if self.cleanup_error is not None:
            raise self.cleanup_error


class _Runtime:
    tokenizer_sha256 = _DIGEST
    bundle_receipt = _BUNDLE_RECEIPT

    def __init__(self, model=None, tokenizer=None) -> None:
        self.model = _Model() if model is None else model
        self.tokenizer = _Tokenizer() if tokenizer is None else tokenizer
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _InjectedChat(Qwen38CausalChat):
    def __init__(self, factory, *args, **kwargs) -> None:
        self._injected_factory = factory
        super().__init__(*args, **kwargs)

    def _open_runtime(self):
        return self._injected_factory()


def _chat(runtime: _Runtime, **overrides) -> Qwen38CausalChat:
    factory = overrides.pop("runtime_factory", lambda: runtime)
    options = {
        "max_prompt_tokens": 8,
        "max_new_tokens": 3,
        "max_context_tokens": 16,
        "head_block_rows": 17,
    }
    options.update(overrides)
    return _InjectedChat(
        factory,
        "unused.causal",
        "unused-tokenizer.json",
        **options,
    )


class _ExactBackend:
    name = "exact.fixture"

    def handle(self, request: Request) -> Result:
        return Result(ExecutionStatus.ABSTAINED, self.name, reason="fixture")


class Qwen38CausalChatTests(unittest.TestCase):
    def test_success_is_lazy_uses_no_thinking_prompt_and_returns_compact_receipts(
        self,
    ) -> None:
        runtime = _Runtime()
        factory_calls = 0

        def factory():
            nonlocal factory_calls
            factory_calls += 1
            return runtime

        chat = _chat(runtime, runtime_factory=factory, system_prompt=" local system ")
        self.assertFalse(chat.loaded)
        self.assertEqual(factory_calls, 0)

        result = chat.handle(Request("chat", "  hello  "))

        self.assertIs(result.status, ExecutionStatus.OK)
        self.assertEqual(result.output, "local answer")
        self.assertTrue(chat.loaded)
        self.assertEqual(factory_calls, 1)
        self.assertEqual(
            runtime.tokenizer.encoded,
            [Qwen38Tokenizer.render_no_thinking_prompt("local system", "hello")],
        )
        self.assertNotIn("transformers", runtime.tokenizer.encoded[0].lower())
        self.assertEqual(runtime.model.calls[0][0], [[11, 12]])
        self.assertEqual(
            runtime.model.calls[0][1],
            {
                "max_new_tokens": 3,
                "prefill_tokenwise": False,
                "eos_token_ids": (248046, 248044),
                "head_block_rows": 17,
            },
        )
        self.assertEqual(runtime.model.reset_calls, [True])
        self.assertEqual(result.evidence["model"], "Qwen/Qwen3.8-27B")
        self.assertEqual(result.evidence["tokenizer_sha256"], _DIGEST)
        self.assertEqual(result.evidence["bundle"], _BUNDLE_RECEIPT)
        generation = result.evidence["generation"]
        self.assertEqual(generation["prompt_tokens"], 2)
        self.assertEqual(generation["generated_tokens"], 2)
        self.assertEqual(len(generation["token_trace_sha256"]), 64)
        self.assertNotIn("prompt_token_ids", generation)
        self.assertNotIn("generated_token_ids", generation)

    def test_rejections_do_not_open_the_runtime(self) -> None:
        calls = 0

        def factory():
            nonlocal calls
            calls += 1
            return _Runtime()

        chat = _chat(_Runtime(), runtime_factory=factory)
        unsupported = chat.handle(Request("exact_math", "1+1"))
        blank = chat.handle(Request("chat", " \n"))

        self.assertIs(unsupported.status, ExecutionStatus.REJECTED)
        self.assertIs(blank.status, ExecutionStatus.REJECTED)
        self.assertEqual(calls, 0)

    def test_missing_bundle_is_unavailable_and_never_falls_back(self) -> None:
        chat = Qwen38CausalChat(
            Path("definitely-missing-qwen-bundle"),
            Path("definitely-missing-tokenizer.json"),
            max_prompt_tokens=8,
            max_new_tokens=3,
            max_context_tokens=16,
        )

        result = chat.handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.UNAVAILABLE)
        self.assertIn("causal bundle directory is missing", result.reason)
        self.assertFalse(chat.loaded)
        self.assertEqual(result.evidence["model"], "Qwen/Qwen3.8-27B")

    def test_bundle_and_tokenizer_symlinks_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_target = root / "bundle-target"
            bundle_target.mkdir()
            bundle_link = root / "bundle-link"
            bundle_link.symlink_to(bundle_target, target_is_directory=True)
            tokenizer = root / "tokenizer.json"
            tokenizer.write_text("{}", encoding="utf-8")

            linked_bundle = Qwen38CausalChat(
                bundle_link,
                tokenizer,
                max_prompt_tokens=8,
                max_new_tokens=3,
                max_context_tokens=16,
            ).handle(Request("chat", "hello"))
            self.assertIs(linked_bundle.status, ExecutionStatus.UNAVAILABLE)
            self.assertIn("non-symlink directory", linked_bundle.reason)

            plain_bundle = root / "plain-bundle"
            plain_bundle.mkdir()
            tokenizer_link = root / "tokenizer-link.json"
            tokenizer_link.symlink_to(tokenizer)
            linked_tokenizer = Qwen38CausalChat(
                plain_bundle,
                tokenizer_link,
                max_prompt_tokens=8,
                max_new_tokens=3,
                max_context_tokens=16,
            ).handle(Request("chat", "hello"))
            self.assertIs(linked_tokenizer.status, ExecutionStatus.UNAVAILABLE)
            self.assertIn("non-symlink regular file", linked_tokenizer.reason)

    def test_invalid_injected_receipt_is_unavailable_and_runtime_is_closed(
        self,
    ) -> None:
        runtime = _Runtime()
        runtime.bundle_receipt = {**_BUNDLE_RECEIPT, "kind": "unverified"}
        chat = _chat(runtime)

        result = chat.handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.UNAVAILABLE)
        self.assertIn("verified complete Qwen causal bundle", result.reason)
        self.assertEqual(runtime.close_calls, 1)

    def test_generation_failure_is_contained_and_state_is_released(self) -> None:
        model = _Model(generation_error=RuntimeError("decoder failed"))
        runtime = _Runtime(model=model)
        chat = _chat(runtime)

        result = chat.handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.ERROR)
        self.assertIn("decoder failed", result.reason)
        self.assertEqual(model.reset_calls, [True])
        self.assertEqual(runtime.close_calls, 0)

    def test_cleanup_failure_retires_runtime_fail_closed(self) -> None:
        model = _Model(cleanup_error=RuntimeError("cannot release"))
        runtime = _Runtime(model=model)
        chat = _chat(runtime)

        first = chat.handle(Request("chat", "hello"))
        second = chat.handle(Request("chat", "again"))

        self.assertIs(first.status, ExecutionStatus.ERROR)
        self.assertEqual(first.reason, "Qwen3.8 state cleanup failed")
        self.assertIs(second.status, ExecutionStatus.UNAVAILABLE)
        self.assertEqual(runtime.close_calls, 1)

    def test_context_manager_closes_once_and_closed_component_is_unavailable(
        self,
    ) -> None:
        runtime = _Runtime()
        with _chat(runtime) as chat:
            self.assertTrue(chat.handle(Request("chat", "hello")).ok)
        chat.close()

        self.assertTrue(chat.closed)
        self.assertEqual(runtime.close_calls, 1)
        after = chat.handle(Request("chat", "again"))
        self.assertIs(after.status, ExecutionStatus.UNAVAILABLE)
        with self.assertRaises(Qwen38ChatError):
            chat.__enter__()

    def test_composition_constructs_optional_qwen_general_chat_lazily(self) -> None:
        runtime = _Runtime()
        component = _chat(runtime)
        exact_a = _ExactBackend()
        fertig = FertigSolver()
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
            return_value=component,
        ) as constructor:
            root = compose_runtime(
                s3_arithmetic=exact_a,
                fertig=fertig,
                qwen38_causal_bundle="local.causal",
                qwen38_tokenizer="tokenizer.json",
                qwen38_options={
                    "max_prompt_tokens": 8,
                    "max_new_tokens": 3,
                    "max_context_tokens": 16,
                },
            )

        self.assertIsInstance(root.general_chat, QwenFertigChat)
        self.assertIs(root.general_chat.qwen, component)
        self.assertIs(root.general_chat.fertig, root.exact_math.fertig)
        self.assertFalse(root.general_chat.loaded)
        constructor.assert_called_once()
        self.assertEqual(root.runtime.registry.capabilities(), ("chat", "exact_math"))
        self.assertTrue(root.dispatch("chat", "hello").ok)
        root.general_chat.close()

    def test_composition_can_explicitly_construct_raw_qwen_for_low_level_use(
        self,
    ) -> None:
        component = _chat(_Runtime())
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
            return_value=component,
        ):
            root = compose_runtime(
                s3_arithmetic=_ExactBackend(),
                fertig=_ExactBackend(),
                qwen38_causal_bundle="local.causal",
                qwen38_tokenizer="tokenizer.json",
                qwen38_raw_chat=True,
            )

        self.assertIs(root.general_chat, component)
        self.assertIs(root.runtime.registry.get("chat"), component)
        root.general_chat.close()

    def test_composition_rejects_ambiguous_qwen_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "configured together"):
            CompositionRoot.build(
                s3_arithmetic=_ExactBackend(),
                fertig=_ExactBackend(),
                qwen38_causal_bundle="local.causal",
            )
        with self.assertRaisesRegex(ValueError, "either general_chat"):
            CompositionRoot.build(
                s3_arithmetic=_ExactBackend(),
                fertig=_ExactBackend(),
                general_chat=_chat(_Runtime()),
                qwen38_causal_bundle="local.causal",
                qwen38_tokenizer="tokenizer.json",
            )
        with self.assertRaisesRegex(ValueError, "cannot override"):
            CompositionRoot.build(
                s3_arithmetic=_ExactBackend(),
                fertig=_ExactBackend(),
                qwen38_causal_bundle="local.causal",
                qwen38_tokenizer="tokenizer.json",
                qwen38_options={"runtime_factory": lambda: _Runtime()},
            )

    def test_cli_wires_explicit_local_paths_and_closes_component(self) -> None:
        qwen = _chat(_Runtime())
        component = QwenFertigChat(qwen, FertigSolver())

        class _Root:
            general_chat = component

            def dispatch(self, capability, payload):
                self.call = (capability, payload)
                return self.general_chat.handle(Request(capability, payload))

        root = _Root()
        output = io.StringIO()
        with patch.object(CompositionRoot, "build", return_value=root) as build:
            with redirect_stdout(output):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--qwen38-causal-bundle",
                        "/models/qwen.causal",
                        "--qwen38-tokenizer",
                        "/models/tokenizer.json",
                        "--max-new-tokens",
                        "4",
                    ]
                )

        self.assertEqual(code, 0)
        self.assertEqual(root.call, ("chat", "hello"))
        self.assertTrue(component.closed)
        self.assertTrue(qwen.closed)
        options = build.call_args.kwargs
        self.assertEqual(options["qwen38_causal_bundle"], "/models/qwen.causal")
        self.assertEqual(options["qwen38_tokenizer"], "/models/tokenizer.json")
        self.assertEqual(options["qwen38_options"]["max_new_tokens"], 4)
        self.assertEqual(options["qwen38_options"]["source_budget_mb"], 65536)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["component"], "qwen3.8.fertig-chat")
        self.assertEqual(payload["output"], "local answer")
        self.assertEqual(
            payload["evidence"]["receipt"]["route"],
            "qwen_verification_abstained",
        )


if __name__ == "__main__":
    unittest.main()
