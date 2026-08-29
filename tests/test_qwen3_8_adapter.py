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
        self.state_bytes = 0
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
            "forward_passes": 1,
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
        return generated, {**evidence, "forward_passes": 0}


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


class _FastMount:
    def __init__(
        self,
        *,
        aux_bytes: int = 20,
        cumulative: tuple[int, ...] | None = None,
    ) -> None:
        self.cumulative = (0, aux_bytes) if cumulative is None else cumulative
        self.calls = 0

    def metrics(self):
        total = self.cumulative[min(self.calls, len(self.cumulative) - 1)]
        self.calls += 1
        return {
            "pilot_source_body_bytes": 3 * total // 5,
            "transpose_source_body_bytes": total - 3 * total // 5,
            "source_body_bytes": total,
            "pilot_logical_weight_bytes": 3 * total // 5,
            "transpose_logical_weight_bytes": total - 3 * total // 5,
            "logical_weight_bytes": total,
            "pilot_row_reads": self.calls - 1,
            "transpose_row_reads": self.calls - 1,
        }


class _ExactHeadMetrics:
    def __init__(self) -> None:
        self.calls = 0

    def metrics(self):
        current = self.calls
        self.calls += 1
        return {
            "calls": current,
            "pages_pruned": 2 * current,
            "rows_pruned": 8 * current,
            "last_fallback_reason": "",
            "manifest_sha256": _EXACT_HEAD_RECEIPT["manifest_sha256"],
        }


_FAST_RECEIPT = {
    "active_layers": [0, 9],
    "affine_fit_sha256": "1" * 64,
    "execution": "row-routed-sparse-mlp",
    "fitted_layers": [0, 9],
    "model_pin_sha256": "2" * 64,
    "pilot_manifest_body_sha256": "3" * 64,
    "router_fit_sha256": "4" * 64,
    "schema": "immer.qwen3.8-fast-mlp-mount/v1",
    "selected_neuron_fraction_by_layer": {"0": 0.2, "9": 0.2},
    "transport_row_fraction_by_layer": {"0": 0.2, "9": 0.2},
    "transpose_manifest_sha256": "5" * 64,
    "weights_index_sha256": "6" * 64,
}

_EXACT_HEAD_RECEIPT = {
    "index_bytes": 67_000_000,
    "manifest_sha256": "7" * 64,
    "payload_bytes": 68_000_000,
    "payload_sha256": "8" * 64,
    "tensor_sha256": "9" * 64,
}


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
    def test_runtime_mounts_and_closes_range_markov_observer(self) -> None:
        from immer.runtimes.deepseek_v4.causal_weights import LogicalModelIdentity
        from immer.runtimes.qwen3_8.adapter import _open_local_runtime

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenizer_path = root / "tokenizer.json"
            tokenizer_path.write_text("{}", encoding="utf-8")
            lifecycle = []
            source = SimpleNamespace(
                prefetch_range=Mock(return_value=True),
                set_access_observer=Mock(
                    side_effect=lambda *_args, **_kwargs: lifecycle.append("attach")
                ),
                repo_id="Qwen/test",
                revision="a" * 40,
                metrics=Mock(
                    return_value={"inventory_source_fingerprint": "b" * 64}
                ),
            )
            mount = SimpleNamespace(
                source=source,
                tensor_reader=object(),
                weights_root=root,
                close=Mock(),
            )
            pager = SimpleNamespace(
                attach_exact_head_index=Mock(),
                close=Mock(),
            )
            model = SimpleNamespace(
                checkpoint_preflight=Mock(
                    side_effect=lambda: (
                        lifecycle.append("preflight") or {"ok": True}
                    )
                ),
                reset_state=Mock(),
                mlp_sparse_executor=None,
            )
            tokenizer = SimpleNamespace()
            prefetcher = SimpleNamespace(
                bind_source_identity=Mock(),
                close=Mock(),
                metrics=lambda: {},
            )
            with (
                patch(
                    "immer.runtimes.qwen3_8.adapter.CausalWeightMount",
                    return_value=mount,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.verify_qwen38_causal_mount",
                    return_value=_BUNDLE_RECEIPT,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38Config.from_file",
                    return_value=SimpleNamespace(),
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38WeightPager",
                    return_value=pager,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.StreamedQwen38",
                    return_value=model,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38Tokenizer",
                    return_value=tokenizer,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter._file_sha256",
                    return_value=_DIGEST,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.MarkovRangePrefetcher",
                    return_value=prefetcher,
                ) as constructor,
            ):
                runtime = _open_local_runtime(
                    bundle_path=root,
                    tokenizer_path=tokenizer_path,
                    identity=LogicalModelIdentity("Qwen/test", "a" * 40),
                    require_official_config=False,
                    device="cpu",
                    compute_dtype="bfloat16",
                    source_budget_mb=1,
                    max_resident_bytes=1024,
                    max_context_tokens=16,
                    range_markov_state_path=root / "ranges.bin",
                    range_prefetch_max_bytes=123,
                    range_prefetch_min_support=3,
                    range_prefetch_min_confidence=0.8,
                )

            constructor.assert_called_once_with(
                root / "ranges.bin",
                prefetch_range=source.prefetch_range,
                min_support=3,
                min_confidence=0.8,
                max_prefetch_bytes=123,
                beam_horizon=3,
                beam_width=4,
                hint_cooldown_operations=2,
            )
            source.set_access_observer.assert_called_once_with(
                prefetcher,
                prepare_identity=False,
            )
            prefetcher.bind_source_identity.assert_called_once_with(
                "Qwen/test",
                "a" * 40,
                "b" * 64,
            )
            self.assertEqual(lifecycle[:2], ["preflight", "attach"])
            runtime.close()
            prefetcher.close.assert_called_once_with()
            self.assertEqual(
                source.set_access_observer.call_args_list[-1].args,
                (None,),
            )
            mount.close.assert_called_once_with()

    def test_range_markov_metrics_are_exposed_without_changing_generation(self) -> None:
        runtime = _Runtime()
        runtime.range_prefetcher = SimpleNamespace(
            metrics=lambda: {
                "schema": "immer.range-markov-metrics/v1",
                "operations": 12,
                "predictions": 7,
                "prefetch_hints": 5,
            }
        )
        chat = _chat(runtime, range_markov_state_path="/state/ranges.bin")

        result = chat.handle(Request("chat", "hello"))

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.evidence["range_markov"]["prefetch_hints"], 5)
        self.assertEqual(result.output, runtime.tokenizer.decoded.strip())
        chat.close()

    def test_exact_head_non_cpu_configuration_is_lazy_nonapplicable(self) -> None:
        component = Qwen38CausalChat(
            "unused.causal",
            "unused-tokenizer.json",
            device="mps",
            compute_dtype="float32",
            exact_head_root="/artifacts/qwen-head",
        )

        self.assertFalse(component.loaded)
        component.close()

    def test_q4_selects_cpu_and_composes_with_sparse_mlp(self) -> None:
        component = Qwen38CausalChat(
            "unused.causal",
            "unused-tokenizer.json",
            q4_root="/models/qwen-q4",
        )
        self.assertEqual(component._device, "cpu")
        component.close()

        component = Qwen38CausalChat(
            "unused.causal",
            "unused-tokenizer.json",
            q4_root="/models/qwen-q4",
            fast_mlp_root="/artifacts/fast",
            fast_mlp_selected_block_count=96,
            delta_head_state_path="/state/delta-head.json",
        )
        self.assertEqual(component._device, "cpu")
        self.assertIsNotNone(component._fast_mlp_paths)
        self.assertEqual(component._fast_mlp_selected_block_count, 96)
        self.assertEqual(
            component._delta_head_state_path,
            Path("/state/delta-head.json"),
        )
        component.close()

        with self.assertRaisesRegex(ValueError, "requires Q4"):
            Qwen38CausalChat(
                "unused.causal",
                "unused-tokenizer.json",
                fast_mlp_root="/artifacts/fast",
                fast_mlp_selected_block_count=96,
            )

        for options in (
            {"exact_head_root": "/artifacts/head"},
            {"range_markov_state_path": "/state/ranges"},
        ):
            with self.subTest(options=options), self.assertRaisesRegex(
                ValueError, "Q4 execution replaces"
            ):
                Qwen38CausalChat(
                    "unused.causal",
                    "unused-tokenizer.json",
                    q4_root="/models/qwen-q4",
                    **options,
                )

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
                "retain_final_state": False,
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

    def test_optional_k4_drafter_is_used_by_general_chat(self) -> None:
        target = _Runtime()
        target.fast_mlp_mount = _FastMount()
        target.fast_mlp_receipt = dict(_FAST_RECEIPT)
        chat = _chat(
            target,
            draft_bundle_path="draft.causal",
            max_new_tokens=4,
            fast_mlp_root="/artifacts/fast-mlp",
            fast_mlp_active_layers=(0, 9),
        )
        chat._draft_runtime = SimpleNamespace(
            tokenizer_sha256=_DIGEST,
            model=SimpleNamespace(config=SimpleNamespace(vocab_size=300_000)),
            bundle_receipt=_BUNDLE_RECEIPT,
            close=lambda: None,
        )
        provider = SimpleNamespace(
            metrics=lambda: SimpleNamespace(
                source_body_bytes=12,
                linear_calls=3,
            ),
            close=lambda: None,
        )
        evidence = SimpleNamespace(
            accepted_draft_tokens=4,
            source_body_bytes=100,
            linear_calls=10,
            seconds=1.25,
            state_bytes=456,
            stopped_on_eos=False,
            prompt_token_ids=(11, 12),
            generated_token_ids=(7, 8, 9, 10),
            forward_passes=2,
            rounds=(object(),),
            schema="immer.qwen3.8-rolling-k4-speculative-generation/v1",
            final_state_committed=False,
        )
        generated = SimpleNamespace(token_ids=(7, 8, 9, 10), evidence=evidence)
        decoder = SimpleNamespace(
            generate_rolling=lambda *args, **kwargs: generated
        )
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen35K4DraftProvider",
            return_value=provider,
        ), patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38K4SpeculativeDecoder",
            return_value=decoder,
        ) as decoder_constructor:
            result = chat.handle(Request("chat", "hello"))

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.evidence["generation"]["forward_passes"], 2)
        self.assertEqual(result.evidence["generation"]["source_body_bytes"], 100)
        self.assertEqual(result.evidence["draft"]["accepted_draft_tokens"], 4)
        self.assertEqual(result.evidence["draft"]["draft_linear_calls"], 3)
        self.assertEqual(result.evidence["draft"]["target_source_body_bytes"], 100)
        self.assertEqual(result.evidence["draft"]["total_source_body_bytes"], 132)
        self.assertEqual(result.evidence["draft"]["target_linear_calls"], 10)
        self.assertEqual(result.evidence["draft"]["total_linear_calls"], 13)
        self.assertEqual(result.evidence["generation"]["linear_calls"], 10)
        self.assertEqual(result.evidence["fast_mlp"]["request"]["aux_source_body_bytes"], 20)
        decoder_constructor.assert_called_once_with(
            target.model,
            provider,
            window_size=4,
            adaptive_round_windows=False,
        )
        chat.close()

    def test_fast_mlp_identity_and_request_traffic_reach_general_chat(self) -> None:
        runtime = _Runtime()
        runtime.fast_mlp_mount = _FastMount()
        runtime.fast_mlp_receipt = dict(_FAST_RECEIPT)
        chat = _chat(
            runtime,
            fast_mlp_root="/artifacts/fast-mlp",
            fast_mlp_active_layers=(0, 9),
        )

        result = chat.handle(Request("chat", "hello"))

        self.assertTrue(result.ok, result.reason)
        fast = result.evidence["fast_mlp"]
        self.assertEqual(fast["router_fit_sha256"], "4" * 64)
        self.assertEqual(fast["request"]["target_source_body_bytes"], 1234)
        self.assertEqual(fast["request"]["aux_source_body_bytes"], 20)
        self.assertEqual(fast["request"]["total_source_body_bytes"], 1254)
        policy = chat._result_cell_generation_policy_sha256()
        self.assertEqual(len(policy), 64)
        chat.close()

    def test_markov_draft_mode_needs_no_sibling_model_runtime(self) -> None:
        target = _Runtime()
        chat = _chat(target, draft_mode="markov", max_new_tokens=4)
        evidence = SimpleNamespace(
            accepted_draft_tokens=2,
            source_body_bytes=100,
            linear_calls=10,
            seconds=1.0,
            state_bytes=456,
            stopped_on_eos=False,
            prompt_token_ids=(11, 12),
            generated_token_ids=(7, 8, 9, 10),
            forward_passes=3,
            rounds=(object(),),
            schema="immer.qwen3.8-rolling-k4-speculative-generation/v1",
            final_state_committed=False,
        )
        generated = SimpleNamespace(token_ids=(7, 8, 9, 10), evidence=evidence)
        decoder = SimpleNamespace(
            generate_rolling=lambda *args, **kwargs: generated
        )
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38K4SpeculativeDecoder",
            return_value=decoder,
        ) as decoder_constructor:
            result = chat.handle(Request("chat", "hello"))

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.evidence["draft"]["mode"], "markov")
        self.assertEqual(result.evidence["draft"]["draft_source_body_bytes"], 0)
        self.assertEqual(result.evidence["draft"]["draft_linear_calls"], 0)
        self.assertNotIn("draft_bundle", result.evidence)
        provider = decoder_constructor.call_args.args[1]
        self.assertEqual(provider.metrics().source_body_bytes, 0)
        self.assertTrue(
            decoder_constructor.call_args.kwargs["adaptive_round_windows"]
        )
        chat.close()

    def test_short_generation_policy_records_plain_greedy_draft_fallback(self) -> None:
        runtime = _Runtime()
        runtime.model.generated = (7,)
        chat = _chat(runtime, draft_mode="markov", max_new_tokens=1)
        with patch(
            "immer.runtimes.qwen3_8.adapter._digest",
            side_effect=lambda value: value,
        ):
            policy = chat._result_cell_generation_policy_sha256()

        self.assertEqual(policy["decoding"], "greedy")
        self.assertEqual(
            policy["draft_fallback"],
            {
                "configured_mode": "markov",
                "reason": "max-new-tokens-below-2",
            },
        )
        result = chat.handle(Request("chat", "hello"))
        self.assertTrue(result.ok, result.reason)
        self.assertNotIn("draft", result.evidence)
        chat.close()

    def test_failed_k4_fast_request_cannot_leak_counters_into_next_call(self) -> None:
        target = _Runtime()
        target.fast_mlp_mount = _FastMount(cumulative=(0, 7, 27))
        target.fast_mlp_receipt = dict(_FAST_RECEIPT)
        chat = _chat(
            target,
            draft_bundle_path="draft.causal",
            max_new_tokens=4,
            fast_mlp_root="/artifacts/fast-mlp",
            fast_mlp_active_layers=(0, 9),
        )
        chat._draft_runtime = SimpleNamespace(
            tokenizer_sha256=_DIGEST,
            model=SimpleNamespace(config=SimpleNamespace(vocab_size=300_000)),
            bundle_receipt=_BUNDLE_RECEIPT,
            close=lambda: None,
        )
        provider = SimpleNamespace(
            metrics=lambda: SimpleNamespace(source_body_bytes=12, linear_calls=3),
            close=lambda: None,
        )
        evidence = SimpleNamespace(
            accepted_draft_tokens=4,
            source_body_bytes=100,
            linear_calls=10,
            seconds=1.25,
            state_bytes=456,
            stopped_on_eos=False,
            prompt_token_ids=(11, 12),
            generated_token_ids=(7, 8, 9, 10),
            forward_passes=2,
            rounds=(object(),),
            schema="immer.qwen3.8-rolling-k4-speculative-generation/v1",
            final_state_committed=False,
        )
        generated = SimpleNamespace(token_ids=(7, 8, 9, 10), evidence=evidence)
        decoder = SimpleNamespace(
            generate_rolling=Mock(
                side_effect=[RuntimeError("first failed"), generated]
            )
        )
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen35K4DraftProvider",
            return_value=provider,
        ), patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38K4SpeculativeDecoder",
            return_value=decoder,
        ):
            failed = chat.handle(Request("chat", "hello"))
            succeeded = chat.handle(Request("chat", "hello"))

        self.assertIs(failed.status, ExecutionStatus.ERROR)
        self.assertNotIn("request", failed.evidence["fast_mlp"])
        self.assertTrue(succeeded.ok, succeeded.reason)
        self.assertEqual(
            succeeded.evidence["fast_mlp"]["request"][
                "aux_source_body_bytes"
            ],
            20,
        )
        chat.close()

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
        self.assertEqual(anchor["forward_passes_baseline"], 2)
        self.assertEqual(anchor["forward_passes_executed"], 1)
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
                    "retain_final_state": False,
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
        output = io.StringIO()
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
            return_value=qwen,
        ) as constructor:
            with redirect_stdout(output):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--qwen38-causal-bundle",
                        "/models/qwen.causal",
                        "--qwen38-tokenizer",
                        "/models/tokenizer.json",
                        "--raw-qwen",
                        "--max-new-tokens",
                        "4",
                    ]
                )

        self.assertEqual(code, 0)
        self.assertTrue(qwen.closed)
        self.assertEqual(
            constructor.call_args.args,
            ("/models/qwen.causal", "/models/tokenizer.json"),
        )
        options = constructor.call_args.kwargs
        self.assertEqual(options["max_new_tokens"], 4)
        self.assertEqual(options["source_budget_mb"], 4194304)
        self.assertEqual(options["draft_source_budget_mb"], 1048576)
        self.assertEqual(options["max_resident_bytes"], 192 * 1024**2)
        self.assertEqual(options["draft_max_resident_bytes"], 64 * 1024**2)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["component"], "qwen3.8.causal-chat")
        self.assertEqual(payload["output"], "local answer")

    def test_cli_resolves_local_stack_and_wraps_fertig_by_default(self) -> None:
        qwen = _chat(_Runtime())
        output = io.StringIO()
        environment = {
            "IMMER_QWEN38_ROOT": "/models/Qwen3.8-27B",
            "IMMER_QWEN38_Q4": "/models/Qwen3.8-27B/causal/q4-base-v2",
            "IMMER_QWEN38_FAST_MLP": "/state/qwen-q4-fast-mlp-all64-v1",
        }
        with patch.dict("os.environ", environment, clear=True), patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
            return_value=qwen,
        ) as constructor, patch(
            "immer.cognition.qwen_fertig_chat.QwenFertigChat",
            side_effect=lambda raw, _fertig: raw,
        ) as wrapper, redirect_stdout(output):
            code = main(["chat", "hello"])

        self.assertEqual(code, 0)
        self.assertTrue(qwen.closed)
        self.assertEqual(
            constructor.call_args.args,
            (
                "/models/Qwen3.8-27B",
                "/models/Qwen3.8-27B/tokenizer.json",
            ),
        )
        options = constructor.call_args.kwargs
        self.assertEqual(
            options["q4_root"],
            "/models/Qwen3.8-27B/causal/q4-base-v2",
        )
        self.assertEqual(
            options["fast_mlp_root"],
            "/state/qwen-q4-fast-mlp-all64-v1",
        )
        self.assertEqual(
            options["fast_mlp_active_layers"],
            (*range(18), *range(55, 64)),
        )
        self.assertEqual(options["fast_mlp_selected_block_count"], 32)
        wrapper.assert_called_once()
        self.assertIs(wrapper.call_args.args[0], qwen)
        self.assertIsInstance(wrapper.call_args.args[1], FertigSolver)

    def test_cli_explicit_layout_does_not_inherit_deployed_q4(self) -> None:
        qwen = _chat(_Runtime())
        with tempfile.TemporaryDirectory() as temporary:
            deployed = Path(temporary) / "deployed"
            (deployed / "causal" / "q4-base-v2").mkdir(parents=True)
            with patch.dict("os.environ", {}, clear=True), patch(
                "immer.cli._QWEN38_DEPLOYMENT_ROOT",
                deployed,
            ), patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                return_value=qwen,
            ) as constructor, redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--qwen38-causal-bundle",
                        "/models/other.causal",
                        "--qwen38-tokenizer",
                        "/models/other-tokenizer.json",
                        "--fast-mlp",
                        "/models/other-fast",
                        "--fast-mlp-layers",
                        "0,9",
                        "--raw-qwen",
                    ]
                )

        self.assertEqual(code, 0)
        self.assertIsNone(constructor.call_args.kwargs["q4_root"])
        self.assertEqual(
            constructor.call_args.kwargs["fast_mlp_active_layers"],
            (0, 9),
        )

    def test_cli_can_disable_the_deployed_sparse_mlp_plan(self) -> None:
        qwen = _chat(_Runtime())
        with tempfile.TemporaryDirectory() as temporary:
            deployed = Path(temporary) / "deployed"
            (deployed / "causal" / "q4-base-v2").mkdir(parents=True)
            deployed_fast = Path(temporary) / "deployed-fast"
            deployed_fast.mkdir()
            with patch.dict(
                "os.environ",
                {"IMMER_QWEN38_FAST_MLP": str(deployed_fast)},
                clear=True,
            ), patch(
                "immer.cli._QWEN38_DEPLOYMENT_ROOT",
                deployed,
            ), patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                return_value=qwen,
            ) as constructor, redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--no-fast-mlp",
                        "--raw-qwen",
                    ]
                )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(
            options["q4_root"],
            str(deployed / "causal" / "q4-base-v2"),
        )
        self.assertIsNone(options["fast_mlp_root"])
        self.assertIsNone(options["fast_mlp_active_layers"])

    def test_cli_deployment_defaults_to_full_q4(self) -> None:
        qwen = _chat(_Runtime())
        with tempfile.TemporaryDirectory() as temporary:
            deployed = Path(temporary) / "deployed"
            (deployed / "causal" / "q4-base-v2").mkdir(parents=True)
            with patch.dict("os.environ", {}, clear=True), patch(
                "immer.cli._QWEN38_DEPLOYMENT_ROOT",
                deployed,
            ), patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                return_value=qwen,
            ) as constructor, redirect_stdout(io.StringIO()):
                code = main(["chat", "hello", "--raw-qwen"])

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(
            options["q4_root"],
            str(deployed / "causal" / "q4-base-v2"),
        )
        self.assertIsNone(options["fast_mlp_root"])
        self.assertIsNone(options["fast_mlp_active_layers"])

    def test_cli_jsonl_reuses_one_loaded_component_for_multiple_requests(self) -> None:
        qwen = _chat(_Runtime())
        output = io.StringIO()
        stream = io.StringIO(
            '{"id":"first","message":"hello"}\nworld\n{"bad":true}\n'
        )
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
            return_value=qwen,
        ) as constructor, patch("sys.stdin", stream), redirect_stdout(output):
            code = main(
                [
                    "chat",
                    "--jsonl",
                    "--max-requests",
                    "2",
                    "--qwen38-causal-bundle",
                    "/models/qwen.causal",
                    "--qwen38-tokenizer",
                    "/models/tokenizer.json",
                ]
            )

        self.assertEqual(code, 0)
        constructor.assert_called_once()
        self.assertTrue(qwen.closed)
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["id"], "first")
        self.assertNotIn("id", rows[1])
        self.assertEqual([row["output"] for row in rows], ["local answer"] * 2)

    def test_cli_wires_fast_mlp_root_and_layer_subset(self) -> None:
        qwen = _chat(_Runtime())
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
            return_value=qwen,
        ) as constructor:
            with redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--qwen38-causal-bundle",
                        "/models/qwen.causal",
                        "--qwen38-tokenizer",
                        "/models/tokenizer.json",
                        "--fast-mlp",
                        "/artifacts/qwen-fast",
                        "--fast-mlp-layers",
                        "0,9,18,63",
                        "--fast-mlp-source-budget-mb",
                        "8192",
                        "--fast-mlp-max-resident-mb",
                        "96",
                        "--fast-mlp-online-state",
                        "/state/qwen-fast.json",
                    ]
                )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(options["fast_mlp_root"], "/artifacts/qwen-fast")
        self.assertEqual(options["fast_mlp_active_layers"], (0, 9, 18, 63))
        self.assertEqual(options["fast_mlp_source_budget_mb"], 8192.0)
        self.assertEqual(options["fast_mlp_max_resident_bytes"], 96 * 1024**2)
        self.assertEqual(
            options["fast_mlp_online_state_path"],
            "/state/qwen-fast.json",
        )

    def test_cli_wires_q4_bank_and_native_threads(self) -> None:
        qwen = _chat(_Runtime())
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
            return_value=qwen,
        ) as constructor:
            with redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--qwen38-causal-bundle",
                        "/models/qwen.causal",
                        "--qwen38-tokenizer",
                        "/models/tokenizer.json",
                        "--qwen38-q4",
                        "/models/qwen-q4",
                        "--q4-threads",
                        "12",
                        "--fast-mlp",
                        "/state/qwen-fast-all64",
                        "--fast-mlp-policy",
                        "structure-edge",
                        "--delta-head-online-state",
                        "/state/qwen-delta-head.json",
                    ]
                )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(options["q4_root"], "/models/qwen-q4")
        self.assertEqual(options["q4_threads"], 12)
        self.assertEqual(options["fast_mlp_root"], "/state/qwen-fast-all64")
        self.assertEqual(
            options["fast_mlp_active_layers"],
            (*range(18), *range(55, 64)),
        )
        self.assertEqual(options["fast_mlp_selected_block_count"], 32)
        self.assertEqual(
            options["delta_head_state_path"],
            "/state/qwen-delta-head.json",
        )

    def test_cli_wires_exact_head_index_root(self) -> None:
        qwen = _chat(_Runtime())
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
            return_value=qwen,
        ) as constructor:
            with redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--qwen38-causal-bundle",
                        "/models/qwen.causal",
                        "--qwen38-tokenizer",
                        "/models/tokenizer.json",
                        "--exact-head",
                        "/artifacts/qwen-head",
                        "--head-block-rows",
                        "64",
                        "--exact-head-max-mb",
                        "96",
                    ]
                )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(options["exact_head_root"], "/artifacts/qwen-head")
        self.assertEqual(options["head_block_rows"], 64)
        self.assertEqual(options["exact_head_max_bytes"], 96 * 1024**2)

    def test_exact_head_receipt_reaches_general_chat_evidence(self) -> None:
        runtime = _Runtime()
        runtime.exact_head_receipt = dict(_EXACT_HEAD_RECEIPT)
        runtime.exact_head_index = _ExactHeadMetrics()
        result = _chat(
            runtime,
            exact_head_root="/artifacts/qwen-head",
        ).handle(Request("chat", "hello"))

        self.assertTrue(result.ok)
        exact = result.evidence["exact_head"]
        self.assertEqual(
            {key: exact[key] for key in _EXACT_HEAD_RECEIPT},
            _EXACT_HEAD_RECEIPT,
        )
        self.assertEqual(exact["request"]["calls"], 1)
        self.assertEqual(exact["request"]["pages_pruned"], 2)
        self.assertEqual(exact["request"]["rows_pruned"], 8)

    def test_cli_wires_persistent_markov_drafting_without_bundle(self) -> None:
        qwen = _chat(_Runtime())
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
            return_value=qwen,
        ) as constructor:
            with redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--qwen38-causal-bundle",
                        "/models/qwen.causal",
                        "--qwen38-tokenizer",
                        "/models/tokenizer.json",
                        "--draft-mode",
                        "markov",
                        "--markov-draft-state",
                        "/state/qwen-markov.bin",
                    ]
                )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(options["draft_mode"], "markov")
        self.assertEqual(
            options["markov_draft_state_path"],
            "/state/qwen-markov.bin",
        )
        self.assertIsNone(options["draft_bundle_path"])


if __name__ == "__main__":
    unittest.main()
