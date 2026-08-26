from __future__ import annotations

import io
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

import torch

from immer.cli import main
from immer.cognition.fertig import FertigSolver
from immer.cognition.qwen_fertig_chat import QwenFertigChat
from immer.composition import CompositionRoot, compose_runtime
from immer.contracts import ExecutionStatus, Request, Result
from immer.runtimes.ooe.result_cells import (
    ResultCell,
    ResultCellBinding,
    attach_cold_qwen_generation_receipt,
    qwen_result_binding_evidence,
)
from immer.runtimes.qwen3_8.adapter import Qwen38CausalChat, Qwen38ChatError
from immer.runtimes.qwen3_8.cartography_probe import prompt_token_sha256
from immer.runtimes.qwen3_8.encoding import Qwen38Tokenizer
from immer.runtimes.qwen3_8.semantic_atlas import ModelPin
from immer.runtimes.qwen3_8.semantic_state_cache import (
    AnchorReceipt,
    RestoredAnchor,
    SemanticStateAnchorCache,
    token_prefix_sha256,
)


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
_ANCHOR_SEED = torch.tensor([[[3.0]]], dtype=torch.float32)
_ANCHOR_PREFIX_SHA256 = token_prefix_sha256((11, 12))
_ANCHOR_SEED_FILE_SHA256 = "8" * 64
_ANCHOR_RECEIPT = AnchorReceipt.create(
    prefix_length=2,
    prefix_sha256=_ANCHOR_PREFIX_SHA256,
    boundary_kind="turn",
    semantic_label_sha256=None,
    snapshot_manifest_name=f"{_ANCHOR_PREFIX_SHA256}.json",
    snapshot_manifest_sha256="1" * 64,
    snapshot_manifest_bytes=10,
    snapshot_body_sha256="2" * 64,
    snapshot_body_bytes=9,
    snapshot_payload_name=f"{_ANCHOR_PREFIX_SHA256}.{'3' * 64}.npz",
    snapshot_payload_sha256="3" * 64,
    snapshot_payload_bytes=20,
    seed_hidden_name=(
        f"{_ANCHOR_PREFIX_SHA256}.{_ANCHOR_SEED_FILE_SHA256}.seed.safetensors"
    ),
    seed_hidden_sha256=_ANCHOR_SEED_FILE_SHA256,
    seed_hidden_bytes=5,
    seed_hidden_tensor_sha256=SemanticStateAnchorCache._seed_tensor_sha256(
        _ANCHOR_SEED
    ),
    seed_hidden_dtype="float32",
    seed_hidden_shape=(1, 1, 1),
    seed_hidden_source_device="cpu",
    state_bytes=321,
    created_sequence=1,
    last_access_sequence=2,
    hit_count=1,
    transport_neutral=False,
)
_RESTORED_ANCHOR = RestoredAnchor(
    anchor=_ANCHOR_RECEIPT,
    query_length=2,
    exact_prefix=True,
    seed_hidden=_ANCHOR_SEED,
)


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


class _AnchorModel(_Model):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=300_000, n_layers=64)
        self.next_position = 0
        self.state_poisoned = False
        self.state_bytes = 0
        self.state_batch_size = None

    def generate_greedy(self, prompt, **kwargs):
        if "restored_prefix_length" not in kwargs:
            return super().generate_greedy(prompt, **kwargs)
        self.calls.append((prompt, kwargs))
        if kwargs.get("restored_prefix_length") != 2:
            raise AssertionError("exact prefix was not passed to generation")
        if kwargs.get("restored_seed_hidden") is not _RESTORED_ANCHOR.seed_hidden:
            raise AssertionError("authenticated head seed was not passed through")
        return self.generated, {
            "prompt_token_ids": tuple(prompt[0]),
            "generated_token_ids": self.generated,
            "context_mode": "stateful_autoregressive",
            "stateful_cache": True,
            "general_generation": True,
            "prefill_mode": "batched",
            "forward_passes": 2,
            "source_body_bytes": 600,
            "linear_calls": 66,
            "seconds": 0.75,
            "state_bytes": 654,
            "stopped_on_eos": False,
        }

    def reset_state(self, *, release=False):
        super().reset_state(release=release)
        self.next_position = 0
        self.state_poisoned = False
        self.state_bytes = 0
        self.state_batch_size = None


class _UnderreportedAnchorModel(_AnchorModel):
    def generate_greedy(self, prompt, **kwargs):
        generated, evidence = super().generate_greedy(prompt, **kwargs)
        return generated, {**evidence, "forward_passes": 1}


def _anchor_cache(
    *,
    failure: Exception | None = None,
    miss: bool = False,
    restored: RestoredAnchor = _RESTORED_ANCHOR,
):
    cache = object.__new__(SemanticStateAnchorCache)

    def restore(model, _token_ids):
        if miss:
            return None
        model.next_position = 2
        model.state_bytes = _ANCHOR_RECEIPT.state_bytes
        model.state_batch_size = 1
        if failure is not None:
            raise failure
        return restored

    cache.restore_deepest = Mock(side_effect=restore)
    return cache


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
        self.assertNotIn("result_cell_binding_receipt", result.evidence)

    def test_opt_in_result_cell_binding_is_runtime_derived_and_chargeable(
        self,
    ) -> None:
        runtime = _Runtime()
        code_revision = "f" * 40
        chat = _chat(
            runtime,
            system_prompt=" local system ",
            result_cell_code_revision=code_revision,
        )
        result = chat.handle(Request("chat", "  hello  "))

        self.assertIs(result.status, ExecutionStatus.OK)
        document = result.evidence["result_cell_binding_receipt"]
        body = document["body"]
        rendered = Qwen38Tokenizer.render_no_thinking_prompt(
            "local system",
            "hello",
        )
        pin = ModelPin(
            repo_id=result.evidence["model"],
            revision=result.evidence["revision"],
            bundle_fingerprint=_BUNDLE_RECEIPT["layout_fingerprint"],
            bundle_manifest_sha256=_BUNDLE_RECEIPT["manifest_sha256"],
            code_revision=code_revision,
        )
        binding = ResultCellBinding(
            model_pin=pin,
            tokenizer_sha256=_DIGEST,
            question_sha256=hashlib.sha256(b"hello").hexdigest(),
            rendered_prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
            rendered_prompt_token_sha256=prompt_token_sha256((11, 12)),
            system_prompt_sha256=hashlib.sha256(b"local system").hexdigest(),
            generation_policy_sha256=chat._result_cell_generation_policy_sha256(),
        )
        self.assertEqual(body["binding_sha256"], binding.sha256)
        self.assertEqual(document, qwen_result_binding_evidence(binding))
        self.assertNotIn("hello", json.dumps(document, sort_keys=True))
        cold = attach_cold_qwen_generation_receipt(result, binding=binding)
        final = Result(
            ExecutionStatus.OK,
            "qwen-fertig-chat",
            output=result.output,
            evidence={"route": "qwen_verified"},
        )
        cell = ResultCell.from_cold(
            binding=binding,
            cold_qwen_result=cold,
            cold_final_result=final,
            cold_fertig_judgment={"status": "verified"},
            cold_fertig_status="verified",
            evaluator_quality_contract_sha256="9" * 64,
        )
        self.assertEqual(cell.teacher_forward_count, 3)
        self.assertEqual(cell.cold_qwen_result, cold)

    def test_result_cell_code_revision_is_full_or_disabled(self) -> None:
        _chat(_Runtime())
        with self.assertRaisesRegex(ValueError, "full lowercase"):
            _chat(_Runtime(), result_cell_code_revision="short")

    def test_authenticated_exact_anchor_bypasses_prefill_with_identical_output(
        self,
    ) -> None:
        baseline = _chat(_Runtime()).handle(Request("chat", "hello"))
        model = _AnchorModel()
        cache = _anchor_cache()
        cached = _chat(
            _Runtime(model=model),
            anchor_cache=cache,
        ).handle(Request("chat", "hello"))

        self.assertIs(baseline.status, ExecutionStatus.OK)
        self.assertIs(cached.status, ExecutionStatus.OK)
        self.assertEqual(cached.output, baseline.output)
        cache.restore_deepest.assert_called_once_with(model, (11, 12))
        options = model.calls[0][1]
        self.assertEqual(options["restored_prefix_length"], 2)
        self.assertIs(
            options["restored_seed_hidden"],
            _RESTORED_ANCHOR.seed_hidden,
        )
        anchor = cached.evidence["anchor_cache"]
        self.assertEqual(anchor["status"], "hit")
        self.assertTrue(anchor["exact_prefix"])
        self.assertEqual(anchor["prefix_tokens"], 2)
        self.assertEqual(anchor["suffix_tokens"], 0)
        self.assertEqual(anchor["snapshot_bytes_read"], 70)
        self.assertEqual(anchor["forward_passes_baseline"], 3)
        self.assertEqual(anchor["forward_passes_executed"], 2)
        self.assertEqual(anchor["forward_passes_saved"], 1)
        self.assertEqual(anchor["prefill_weight_sweeps_saved"], 1)
        self.assertEqual(anchor["checkpoint_read_sweeps_saved"], 1)
        self.assertEqual(anchor["prompt_token_layer_evaluations_saved"], 128)
        self.assertEqual(anchor["checkpoint_source_body_bytes_read"], 600)
        self.assertEqual(anchor["checkpoint_linear_calls_executed"], 66)
        self.assertEqual(anchor["anchor"], _ANCHOR_RECEIPT.to_document())
        self.assertEqual(model.reset_calls, [True])

    def test_anchor_restore_failure_never_falls_back_and_resets_state(self) -> None:
        model = _AnchorModel()
        result = _chat(
            _Runtime(model=model),
            anchor_cache=_anchor_cache(failure=RuntimeError("tampered anchor")),
        ).handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.ERROR)
        self.assertIn("tampered anchor", result.reason)
        self.assertEqual(model.calls, [])
        self.assertEqual(model.reset_calls, [True])
        self.assertEqual(model.next_position, 0)
        self.assertEqual(model.state_bytes, 0)

    def test_anchor_seed_must_match_the_sealed_receipt(self) -> None:
        model = _AnchorModel()
        tampered = RestoredAnchor(
            anchor=_ANCHOR_RECEIPT,
            query_length=2,
            exact_prefix=True,
            seed_hidden=_ANCHOR_SEED + 1.0,
        )
        result = _chat(
            _Runtime(model=model),
            anchor_cache=_anchor_cache(restored=tampered),
        ).handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.ERROR)
        self.assertIn("seed differs from the sealed anchor", result.reason)
        self.assertEqual(model.calls, [])
        self.assertEqual(model.reset_calls, [True])

    def test_anchor_forward_savings_reject_underreported_execution(self) -> None:
        model = _UnderreportedAnchorModel()
        result = _chat(
            _Runtime(model=model),
            anchor_cache=_anchor_cache(),
        ).handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.ERROR)
        self.assertIn("forward count is inconsistent", result.reason)
        self.assertEqual(model.reset_calls, [True])

    def test_anchor_miss_preserves_the_existing_generation_path(self) -> None:
        model = _AnchorModel()
        result = _chat(
            _Runtime(model=model),
            anchor_cache=_anchor_cache(miss=True),
        ).handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.OK)
        self.assertEqual(result.output, "local answer")
        self.assertEqual(
            model.calls[0],
            (
                [[11, 12]],
                {
                    "max_new_tokens": 3,
                    "prefill_tokenwise": False,
                    "eos_token_ids": (248046, 248044),
                    "head_block_rows": 17,
                },
            ),
        )
        self.assertEqual(result.evidence["anchor_cache"]["status"], "miss")
        self.assertEqual(model.reset_calls, [True])

    def test_anchor_cache_rejects_duck_typed_restore_provider(self) -> None:
        with self.assertRaisesRegex(TypeError, "SemanticStateAnchorCache"):
            _chat(
                _Runtime(),
                anchor_cache=SimpleNamespace(restore_deepest=lambda *_args: None),
            )

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
