from __future__ import annotations

import io
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import Mock, patch

import torch

from immer.cli import (
    _qwen38_growing_warm_profile,
    _qwen38_runtime_code_paths,
    main,
)
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
from immer.runtimes.qwen3_8.adapter import (
    QWEN38_CHAT_HISTORY_METADATA,
    QWEN38_CHAT_SESSION_METADATA,
    Qwen38CausalChat,
    Qwen38ChatError,
)
from immer.runtimes.qwen3_8.cartography_probe import prompt_token_sha256
from immer.runtimes.qwen3_8.encoding import IM_END_TOKEN_ID, Qwen38Tokenizer
from immer.runtimes.qwen3_8.markov_atlas import MarkovTokenAtlas
from immer.runtimes.qwen3_8.mtp_draft import Qwen35MtpCarry
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

    render_no_thinking_messages = staticmethod(
        Qwen38Tokenizer.render_no_thinking_messages
    )


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
        self.next_position = 0
        self.state_poisoned = False
        self.state_batch_size = None
        self._pending_block_stage = None
        self.pager = SimpleNamespace(release=Mock())
        self.generation_error = generation_error
        self.cleanup_error = cleanup_error
        self.calls: list[tuple[object, dict[str, object]]] = []
        self.reset_calls: list[bool] = []

    def generate_greedy(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if self.generation_error is not None:
            raise self.generation_error
        prompt_ids = tuple(prompt[0])
        retained = len(self.generated) if kwargs.get("retain_final_state") else max(
            0,
            len(self.generated) - 1,
        )
        self.next_position = len(prompt_ids) + retained
        self.state_batch_size = 1
        self.state_bytes = 456
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
        self.next_position = 0
        self.state_poisoned = False
        self.state_batch_size = None
        self.state_bytes = 0


class _StreamingModel(_Model):
    def generate_greedy(self, prompt, **kwargs):
        progress = kwargs.get("progress")
        if callable(progress):
            for step, token_id in enumerate(self.generated):
                progress(
                    {
                        "event": "generated_token",
                        "step": step,
                        "token_id": token_id,
                    }
                )
        return super().generate_greedy(prompt, **kwargs)


class _StreamingTokenizer(_Tokenizer):
    def decode(self, token_ids):
        ids = tuple(token_ids)
        self.decoded_ids.append(ids)
        pieces = {7: "Hello", 8: " world", 9: "!"}
        return "".join(pieces[token_id] for token_id in ids)


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
    def test_markov_atlas_loads_once_and_binds_runtime_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "atlas.bin"
            atlas = MarkovTokenAtlas.build(
                ((1, 2, 3, 4), (5, 2, 3, 6)),
                vocab_size=300_000,
                tokenizer_sha256=_DIGEST,
                max_order=3,
            )
            atlas.write(path)
            chat = _chat(
                _Runtime(),
                draft_mode="markov",
                markov_atlas_path=path,
            )

            runtime = chat._load_locked()
            loaded = chat._markov_atlas
            self.assertIs(runtime.model.config, chat._runtime.model.config)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.sha256, atlas.sha256)
            bound_identity = chat._draft_window_runtime_identity()
            chat._markov_atlas = None
            unbound_identity = chat._draft_window_runtime_identity()
            chat._markov_atlas = loaded
            self.assertNotEqual(bound_identity, unbound_identity)
            chat.close()

    def test_o1_retention_loads_once_with_runtime_identity(self) -> None:
        retention = Mock()
        retention.metrics.return_value = {"sequence": 0}
        chat = _chat(
            _Runtime(),
            draft_mode="markov",
            markov_o1_retention_path="/state/o1-retention.json",
        )
        with patch(
            "immer.runtimes.qwen3_8.adapter.O1MarkovRetention",
            return_value=retention,
        ) as constructor:
            chat._load_locked()

        constructor.assert_called_once_with(
            Path("/state/o1-retention.json"),
            vocab_size=300_000,
            tokenizer_sha256=_DIGEST,
        )
        self.assertIs(chat._markov_o1_retention, retention)
        self.assertEqual(
            chat._base_evidence()["o1_markov_retention"],
            {"sequence": 0},
        )
        chat.close()

    def test_template_anchor_stops_before_user_specific_tokens(self) -> None:
        class Tokenizer:
            @staticmethod
            def encode(text):
                marker = 41 if "\nA<|im_end|>" in text else 57
                return (11, 12, 13, marker, 99)

        chat = _chat(_Runtime())
        prefix = chat._template_anchor_prefix(
            SimpleNamespace(tokenizer=Tokenizer()),
            (11, 12, 13, 77, 88),
        )

        self.assertEqual(prefix, (11, 12, 13))
        chat.close()

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
                metrics=Mock(return_value={"inventory_source_fingerprint": "b" * 64}),
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
                    side_effect=lambda: lifecycle.append("preflight") or {"ok": True}
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

    def test_page_state_flush_failure_never_discards_generated_text(self) -> None:
        runtime = _Runtime()

        class PageRouter:
            page_count = 272
            route_width = 192

            @staticmethod
            def metrics():
                return {"exact_rows": 4, "selected_advances": 3}

            @staticmethod
            def flush() -> None:
                raise OSError("state disk full")

        runtime.mlp_page_router = PageRouter()
        chat = _chat(runtime)

        result = chat.handle(Request("chat", "hello"))

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.output, runtime.tokenizer.decoded.strip())
        self.assertIn(
            "state disk full",
            result.evidence["mlp_page_route"]["persistence_error"],
        )
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
            with (
                self.subTest(options=options),
                self.assertRaisesRegex(ValueError, "Q4 execution replaces"),
            ):
                Qwen38CausalChat(
                    "unused.causal",
                    "unused-tokenizer.json",
                    q4_root="/models/qwen-q4",
                    **options,
                )

    def test_direct_markov_page_route_requires_q4_and_replaces_legacy_sparse(self) -> None:
        component = Qwen38CausalChat(
            "unused.causal",
            "unused-tokenizer.json",
            q4_root="/models/qwen-q4",
            mlp_page_state_path="/state/mlp-pages.json",
            mlp_page_route_width=192,
        )
        self.assertEqual(
            component._mlp_page_state_path,
            Path("/state/mlp-pages.json"),
        )
        self.assertEqual(component._mlp_page_route_width, 192)
        component.close()

        with self.assertRaisesRegex(ValueError, "requires Q4"):
            Qwen38CausalChat(
                "unused.causal",
                "unused-tokenizer.json",
                mlp_page_state_path="/state/mlp-pages.json",
            )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            Qwen38CausalChat(
                "unused.causal",
                "unused-tokenizer.json",
                q4_root="/models/qwen-q4",
                fast_mlp_root="/artifacts/fast",
                mlp_page_state_path="/state/mlp-pages.json",
            )
        with self.assertRaisesRegex(ValueError, "leave at least one page"):
            Qwen38CausalChat(
                "unused.causal",
                "unused-tokenizer.json",
                q4_root="/models/qwen-q4",
                mlp_page_state_path="/state/mlp-pages.json",
                mlp_page_route_width=272,
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
        self.assertEqual(
            result.evidence["conversation"],
            {
                "history_messages": 0,
                "history_turns": 0,
                "mtp_carry_bytes": 0,
                "mtp_carry_reused_tokens": 0,
                "mtp_carry_status": "none",
                "prompt_suffix_tokens": 2,
                "reuse_hits": 0,
                "reuse_misses": 0,
                "reuse_status": "disabled",
                "reused_prefix_tokens": 0,
                "state_retained_tokens": 0,
            },
        )

    def test_chat_history_renders_exact_multi_turn_qwen_context(self) -> None:
        runtime = _Runtime()
        chat = _chat(runtime, system_prompt="stay concise")
        history = (
            ("user", "My code is ORBIT-7."),
            ("assistant", "Understood."),
        )

        result = chat.handle(
            Request(
                "chat",
                "What was my code?",
                {QWEN38_CHAT_HISTORY_METADATA: history},
            )
        )

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(
            runtime.tokenizer.encoded,
            [
                Qwen38Tokenizer.render_no_thinking_messages(
                    "stay concise",
                    (*history, ("user", "What was my code?")),
                )
            ],
        )
        self.assertEqual(
            result.evidence["conversation"],
            {
                "history_messages": 2,
                "history_turns": 1,
                "mtp_carry_bytes": 0,
                "mtp_carry_reused_tokens": 0,
                "mtp_carry_status": "none",
                "prompt_suffix_tokens": 2,
                "reuse_hits": 0,
                "reuse_misses": 0,
                "reuse_status": "disabled",
                "reused_prefix_tokens": 0,
                "state_retained_tokens": 0,
            },
        )
        chat.close()

    def test_conversation_session_reuses_only_exact_committed_token_prefix(
        self,
    ) -> None:
        class ConversationTokenizer(_Tokenizer):
            def encode(self, text: str):
                self.encoded.append(text)
                if "Start over." in text:
                    return (21, 22)
                if "What was the word?" in text:
                    return (11, 12, 7, IM_END_TOKEN_ID, 13, 14)
                return (11, 12)

            def decode(self, token_ids):
                ids = tuple(token_ids)
                self.decoded_ids.append(ids)
                return "alpha" if ids[0] == 7 else "beta"

        class ConversationModel(_Model):
            def generate_greedy(self, prompt, **kwargs):
                self.generated = (
                    (7, IM_END_TOKEN_ID)
                    if not self.calls
                    else (8, IM_END_TOKEN_ID)
                )
                tokens, evidence = super().generate_greedy(prompt, **kwargs)
                return tokens, {**evidence, "stopped_on_eos": True}

        runtime = _Runtime(
            model=ConversationModel(),
            tokenizer=ConversationTokenizer(),
        )
        chat = _chat(runtime, max_prompt_tokens=8)
        session = {QWEN38_CHAT_SESSION_METADATA: "conversation:test"}

        first = chat.handle(Request("chat", "Remember alpha.", session))
        second = chat.handle(
            Request(
                "chat",
                "What was the word?",
                {
                    **session,
                    QWEN38_CHAT_HISTORY_METADATA: (
                        ("user", "Remember alpha."),
                        ("assistant", "alpha"),
                    ),
                },
            )
        )

        self.assertTrue(first.ok, first.reason)
        self.assertTrue(second.ok, second.reason)
        self.assertEqual(first.output, "alpha")
        self.assertEqual(second.output, "beta")
        self.assertEqual(runtime.model.reset_calls, [])
        self.assertEqual(
            runtime.model.calls[1][1]["restored_prefix_length"],
            3,
        )
        self.assertEqual(second.evidence["conversation"]["reuse_status"], "hit")
        self.assertEqual(second.evidence["conversation"]["reused_prefix_tokens"], 3)
        self.assertEqual(second.evidence["conversation"]["prompt_suffix_tokens"], 3)
        self.assertEqual(second.evidence["conversation"]["state_retained_tokens"], 7)
        self.assertEqual(runtime.model.pager.release.call_count, 2)

        restarted = chat.handle(Request("chat", "Start over.", session))
        self.assertTrue(restarted.ok, restarted.reason)
        self.assertEqual(
            restarted.evidence["conversation"]["reuse_status"],
            "token-prefix-mismatch",
        )
        self.assertNotIn("restored_prefix_length", runtime.model.calls[2][1])
        self.assertEqual(runtime.model.reset_calls, [True])

        chat.clear_conversation()
        self.assertEqual(runtime.model.reset_calls, [True, True])
        self.assertEqual(chat._conversation_prefix_token_ids, ())
        chat.close()

    def test_invalid_chat_history_is_rejected_before_runtime_load(self) -> None:
        runtime = _Runtime()
        chat = _chat(runtime)

        result = chat.handle(
            Request(
                "chat",
                "hello",
                {QWEN38_CHAT_HISTORY_METADATA: (("user", "unfinished"),)},
            )
        )

        self.assertIs(result.status, ExecutionStatus.REJECTED)
        self.assertIn("completed turns", result.reason or "")
        self.assertFalse(chat.loaded)
        self.assertEqual(runtime.model.calls, [])
        chat.close()

    def test_conversation_generation_failure_drops_retained_state(self) -> None:
        class FailureTokenizer(_Tokenizer):
            def encode(self, text: str):
                self.encoded.append(text)
                if "Continue." in text:
                    return (11, 12, 7, IM_END_TOKEN_ID, 13)
                return (11, 12)

            def decode(self, token_ids):
                self.decoded_ids.append(tuple(token_ids))
                return "alpha"

        class FailureModel(_Model):
            def generate_greedy(self, prompt, **kwargs):
                tokens, evidence = super().generate_greedy(prompt, **kwargs)
                return tokens, {**evidence, "stopped_on_eos": True}

        model = FailureModel(generated=(7, IM_END_TOKEN_ID))
        runtime = _Runtime(model=model, tokenizer=FailureTokenizer())
        chat = _chat(runtime, max_prompt_tokens=8)
        session = {QWEN38_CHAT_SESSION_METADATA: "conversation:failure"}
        first = chat.handle(Request("chat", "Start.", session))
        self.assertTrue(first.ok, first.reason)
        self.assertTrue(chat._conversation_prefix_token_ids)

        model.generation_error = RuntimeError("decode failed")
        failed = chat.handle(
            Request(
                "chat",
                "Continue.",
                {
                    **session,
                    QWEN38_CHAT_HISTORY_METADATA: (
                        ("user", "Start."),
                        ("assistant", "alpha"),
                    ),
                },
            )
        )

        self.assertIs(failed.status, ExecutionStatus.ERROR)
        self.assertEqual(chat._conversation_prefix_token_ids, ())
        self.assertEqual(model.reset_calls, [True])
        chat.close()

    def test_direct_generation_emits_cumulative_text_snapshots(self) -> None:
        runtime = _Runtime()
        runtime.model = _StreamingModel(generated=(7, 8, 9))
        runtime.tokenizer = _StreamingTokenizer()
        snapshots: list[str] = []
        chat = _chat(runtime, text_snapshot_sink=snapshots.append)

        result = chat.handle(Request("chat", "hello"))

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.output, "Hello world!")
        self.assertEqual(snapshots, ["Hello", "Hello world", "Hello world!"])
        self.assertEqual(runtime.model.reset_calls, [True])
        chat.close()

    def test_draft_round_expands_to_one_snapshot_per_accepted_token(self) -> None:
        runtime = _Runtime(tokenizer=_StreamingTokenizer())
        snapshots: list[str] = []
        chat = _chat(runtime, text_snapshot_sink=snapshots.append)
        progress = chat._draft_generation_progress(runtime)
        assert progress is not None

        progress((7, 8))
        progress((7, 8, 9))

        self.assertEqual(snapshots, ["Hello", "Hello world", "Hello world!"])
        with self.assertRaisesRegex(Qwen38ChatError, "moved backwards"):
            progress((7,))
        chat.close()

    def test_stream_sink_failure_is_a_generation_error_and_state_is_released(
        self,
    ) -> None:
        runtime = _Runtime()
        runtime.model = _StreamingModel(generated=(7,))
        runtime.tokenizer = _StreamingTokenizer()

        def fail(_snapshot: str) -> None:
            raise BrokenPipeError("closed output")

        chat = _chat(runtime, text_snapshot_sink=fail)
        result = chat.handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.ERROR)
        self.assertIn("BrokenPipeError", result.reason or "")
        self.assertEqual(runtime.model.reset_calls, [True])
        chat.close()

    def test_stream_sink_must_be_callable(self) -> None:
        with self.assertRaisesRegex(TypeError, "text_snapshot_sink"):
            _chat(_Runtime(), text_snapshot_sink="stdout")

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
        decoder = SimpleNamespace(generate_rolling=lambda *args, **kwargs: generated)
        with (
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen35K4DraftProvider",
                return_value=provider,
            ),
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38K4SpeculativeDecoder",
                return_value=decoder,
            ) as decoder_constructor,
        ):
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
        self.assertEqual(
            result.evidence["fast_mlp"]["request"]["aux_source_body_bytes"], 20
        )
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
        decoder = SimpleNamespace(generate_rolling=lambda *args, **kwargs: generated)
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
        self.assertTrue(decoder_constructor.call_args.kwargs["adaptive_round_windows"])
        with patch(
            "immer.runtimes.qwen3_8.adapter._digest",
            side_effect=lambda value: value,
        ):
            policy = chat._result_cell_generation_policy_sha256()
        self.assertEqual(
            policy["markov_draft"]["round_window_selector"],
            "markov-prefix-utility/v2",
        )
        chat.close()

    def test_shared_mtp_pager_work_is_split_without_double_counting(self) -> None:
        target = _Runtime()
        target.q4_bank = SimpleNamespace(has=lambda _name: True)
        target.model.pager = SimpleNamespace(q4_bank=target.q4_bank)
        chat = _chat(
            target,
            draft_mode="mtp",
            q4_root="/models/q4-mtp",
            max_new_tokens=4,
        )
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
            schema="immer.qwen3.8-rolling-speculative-generation/v2",
            final_state_committed=False,
        )
        generated = SimpleNamespace(token_ids=(7, 8, 9, 10), evidence=evidence)
        decoder = SimpleNamespace(generate_rolling=lambda *args, **kwargs: generated)
        metrics = SimpleNamespace(
            source_body_bytes=20,
            linear_calls=3,
            last_confidence=0.7,
            last_disagreement=0.0,
            last_phrase_confidence=0.0,
            last_phrase_support=0,
            last_phrase_width=0,
        )
        provider = SimpleNamespace(metrics=lambda: metrics, close=lambda: None)
        with (
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen35MtpDraftProvider",
                return_value=provider,
            ),
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38K4SpeculativeDecoder",
                return_value=decoder,
            ),
        ):
            result = chat.handle(Request("chat", "hello"))

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.evidence["generation"]["source_body_bytes"], 80)
        self.assertEqual(result.evidence["generation"]["linear_calls"], 7)
        draft = result.evidence["draft"]
        self.assertEqual(draft["target_source_body_bytes"], 80)
        self.assertEqual(draft["draft_source_body_bytes"], 20)
        self.assertEqual(draft["total_source_body_bytes"], 100)
        self.assertEqual(draft["target_linear_calls"], 7)
        self.assertEqual(draft["draft_linear_calls"], 3)
        self.assertEqual(draft["total_linear_calls"], 10)
        chat.close()

    def test_mtp_generation_policy_matches_adaptive_round_execution(self) -> None:
        chat = _chat(
            _Runtime(),
            draft_mode="mtp",
            q4_root="/models/q4-mtp",
            markov_draft_state_path="/state/mtp.json",
            max_new_tokens=4,
        )
        with patch(
            "immer.runtimes.qwen3_8.adapter._digest",
            side_effect=lambda value: value,
        ):
            policy = chat._result_cell_generation_policy_sha256()

        self.assertEqual(policy["decoding"], "greedy-rolling-window-draft-verify")
        self.assertEqual(
            policy["mtp_draft"]["round_window_selector"],
            "markov-prefix-utility/v2",
        )
        self.assertTrue(policy["mtp_draft"]["persistent_calibration"])
        chat.close()

    def test_hybrid_policy_binds_markov_first_mtp_fallback(self) -> None:
        chat = _chat(
            _Runtime(),
            draft_mode="hybrid",
            q4_root="/models/q4-mtp",
            markov_draft_state_path="/state/markov.bin",
            mtp_draft_state_path="/state/mtp.json",
            max_new_tokens=8,
        )
        with patch(
            "immer.runtimes.qwen3_8.adapter._digest",
            side_effect=lambda value: value,
        ):
            policy = chat._result_cell_generation_policy_sha256()

        hybrid = policy["hybrid_draft"]
        self.assertEqual(
            hybrid["selection"],
            "round-wise-markov-first-mtp-fallback/v18",
        )
        self.assertFalse(hybrid["request_provider_lock"])
        self.assertFalse(hybrid["one_way_handoff"])
        self.assertTrue(hybrid["round_reselection"])
        self.assertTrue(hybrid["cross_provider_target_state_sync"])
        self.assertTrue(hybrid["cross_provider_target_feedback"])
        self.assertEqual(
            hybrid["consensus"],
            "markov-prefix+atlas+online-memory/v3",
        )
        self.assertTrue(hybrid["committed_hidden_handoff"])
        self.assertEqual(
            hybrid["markov_confidence"],
            "self-calibrating-dialect-council-lookahead/v9",
        )
        self.assertEqual(
            hybrid["position_specialists"],
            "beta-maturity-fixed-share/v1",
        )
        self.assertEqual(
            hybrid["dialect_specialists"],
            "similarity-beta-maturity/v1",
        )
        self.assertEqual(
            hybrid["dialect_council"],
            "similarity-visits-ricci-top4/v1",
        )
        self.assertEqual(
            hybrid["planning"],
            "target-calibrated-top4-one-step/v2",
        )
        self.assertTrue(hybrid["markov_persistent"])
        self.assertTrue(hybrid["mtp_persistent_calibration"])
        self.assertEqual(
            hybrid["round_window_selector"],
            "markov-prefix-utility/v2",
        )
        chat.close()

    def test_restored_conversation_uses_only_state_independent_drafting(self) -> None:
        runtime = _Runtime()
        hybrid = _chat(runtime, draft_mode="hybrid", q4_root="/q4")
        mtp = _chat(_Runtime(), draft_mode="mtp", q4_root="/q4")

        self.assertEqual(hybrid._draft_mode_for_request({}), "hybrid")
        self.assertEqual(
            hybrid._draft_mode_for_request({"restored_prefix_length": 17}),
            "markov",
        )
        self.assertIsNone(
            mtp._draft_mode_for_request({"restored_prefix_length": 17})
        )

        prefix = tuple(range(17))
        carry = Qwen35MtpCarry(
            schema="fixture",
            identity=(),
            history=prefix,
            next_position=16,
            state=None,
            last_target_hidden=torch.zeros((1, 1, 1)),
        )
        hybrid._conversation_prefix_token_ids = prefix
        hybrid._conversation_mtp_carry = carry
        self.assertEqual(
            hybrid._draft_mode_for_request({"restored_prefix_length": 17}),
            "markov",
        )
        hybrid._validated_conversation_mtp_carry = carry
        self.assertEqual(
            hybrid._draft_mode_for_request({"restored_prefix_length": 17}),
            "hybrid",
        )

        hybrid.close()
        mtp.close()

    def test_hybrid_runtime_identity_names_the_round_wise_feedback_policy(
        self,
    ) -> None:
        runtime = _Runtime()
        chat = _chat(
            runtime,
            draft_mode="hybrid",
            q4_root="/models/q4-mtp",
            markov_draft_state_path="/state/markov.bin",
            mtp_draft_state_path="/state/mtp.json",
            max_new_tokens=8,
        )
        chat._runtime = runtime
        chat._bundle_receipt = _BUNDLE_RECEIPT
        chat._tokenizer_sha256 = _DIGEST
        with patch(
            "immer.runtimes.qwen3_8.adapter._digest",
            side_effect=lambda value: value,
        ):
            identity = chat._draft_window_runtime_identity()

        self.assertEqual(
            identity["provider"]["selection"],
            "round-wise-markov-first-mtp-fallback/v18",
        )
        chat.close()

    def test_hybrid_rejects_a_loaded_q4_bank_without_mtp_matrices(self) -> None:
        runtime = _Runtime()
        runtime.q4_bank = SimpleNamespace(has=lambda _name: False)
        runtime.model.pager = SimpleNamespace(q4_bank=runtime.q4_bank)
        result = _chat(
            runtime,
            draft_mode="hybrid",
            q4_root="/models/q4-v2",
            max_new_tokens=8,
        ).handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.UNAVAILABLE)
        self.assertIn("lacks the embedded MTP matrices", result.reason)
        self.assertEqual(runtime.close_calls, 1)

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
            generate_rolling=Mock(side_effect=[RuntimeError("first failed"), generated])
        )
        with (
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen35K4DraftProvider",
                return_value=provider,
            ),
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38K4SpeculativeDecoder",
                return_value=decoder,
            ),
        ):
            failed = chat.handle(Request("chat", "hello"))
            succeeded = chat.handle(Request("chat", "hello"))

        self.assertIs(failed.status, ExecutionStatus.ERROR)
        self.assertNotIn("request", failed.evidence["fast_mlp"])
        self.assertTrue(succeeded.ok, succeeded.reason)
        self.assertEqual(
            succeeded.evidence["fast_mlp"]["request"]["aux_source_body_bytes"],
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
                        "--output",
                        "json",
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
        self.assertIsNone(options["text_snapshot_sink"])
        self.assertEqual(options["source_budget_mb"], 4194304)
        self.assertEqual(options["draft_source_budget_mb"], 1048576)
        self.assertEqual(options["max_resident_bytes"], 192 * 1024**2)
        self.assertEqual(options["draft_max_resident_bytes"], 64 * 1024**2)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["component"], "qwen3.8.causal-chat")
        self.assertEqual(payload["output"], "local answer")

    def test_cli_streams_text_snapshots_without_reprinting_the_final(self) -> None:
        sink: dict[str, object] = {}
        qwen = Mock()

        def construct(*_args, **options):
            sink["callback"] = options["text_snapshot_sink"]
            return qwen

        def handle(_request):
            callback = sink["callback"]
            assert callable(callback)
            callback("Hello")
            callback("Hello world")
            return Result(
                ExecutionStatus.OK,
                "qwen3.8.causal-chat",
                output="Hello world",
            )

        qwen.handle.side_effect = handle
        output = io.StringIO()
        with (
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                side_effect=construct,
            ),
            redirect_stdout(output),
        ):
            code = main(
                [
                    "chat",
                    "hello",
                    "--qwen38-causal-bundle",
                    "/models/qwen.causal",
                    "--qwen38-tokenizer",
                    "/models/tokenizer.json",
                    "--raw-qwen",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(output.getvalue(), "Hello world\n")
        qwen.close.assert_called_once_with()

    def test_cli_wrapped_stream_exposes_progress_but_only_final_text(self) -> None:
        sink: dict[str, object] = {}
        qwen = Mock()
        wrapped = Mock()

        class _Tty(io.StringIO):
            def isatty(self) -> bool:
                return True

        def construct(*_args, **options):
            sink["callback"] = options["text_snapshot_sink"]
            return qwen

        def handle(_request):
            callback = sink["callback"]
            assert callable(callback)
            callback("provisional")
            return Result(
                ExecutionStatus.OK,
                "qwen3.8.fertig-chat",
                output="final exact answer",
            )

        wrapped.handle.side_effect = handle
        wrapped.close.side_effect = qwen.close
        output = io.StringIO()
        progress = _Tty()
        with (
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                side_effect=construct,
            ),
            patch(
                "immer.cognition.qwen_fertig_chat.QwenFertigChat",
                return_value=wrapped,
            ),
            redirect_stdout(output),
            redirect_stderr(progress),
        ):
            code = main(
                [
                    "chat",
                    "hello",
                    "--qwen38-causal-bundle",
                    "/models/qwen.causal",
                    "--qwen38-tokenizer",
                    "/models/tokenizer.json",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(output.getvalue(), "final exact answer\n")
        self.assertIn("Qwen: 1 tokens", progress.getvalue())
        self.assertNotIn("provisional", progress.getvalue())
        wrapped.close.assert_called_once_with()

    def test_cli_resolves_local_stack_and_wraps_fertig_by_default(self) -> None:
        qwen = _chat(_Runtime())
        output = io.StringIO()
        environment = {
            "IMMER_QWEN38_ROOT": "/models/Qwen3.8-27B",
            "IMMER_QWEN38_Q4": "/models/Qwen3.8-27B/causal/q4-base-v2",
            "IMMER_QWEN38_FAST_MLP": "/state/qwen-q4-fast-mlp-all64-v1",
        }
        with (
            patch.dict("os.environ", environment, clear=True),
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                return_value=qwen,
            ) as constructor,
            patch(
                "immer.cognition.qwen_fertig_chat.QwenFertigChat",
                side_effect=lambda raw, _fertig, **_options: raw,
            ) as wrapper,
            redirect_stdout(output),
        ):
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
        self.assertIsNone(wrapper.call_args.kwargs["ooe_hook"])

    def test_cli_mounts_an_explicit_verified_ooe_warm_bank(self) -> None:
        qwen = _chat(_Runtime())
        hook = object()
        mount = SimpleNamespace(hook=hook)
        with (
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                return_value=qwen,
            ),
            patch(
                "immer.runtimes.ooe.qwen_warm_bank.open_verified_qwen_warm_bank",
                return_value=mount,
            ) as opener,
            patch(
                "immer.cognition.qwen_fertig_chat.QwenFertigChat",
                side_effect=lambda raw, _fertig, **_options: raw,
            ) as wrapper,
            redirect_stdout(io.StringIO()),
        ):
            code = main(
                [
                    "chat",
                    "hello",
                    "--qwen38-causal-bundle",
                    "/models/qwen.causal",
                    "--qwen38-tokenizer",
                    "/models/tokenizer.json",
                    "--ooe-warm-root",
                    "/state/qwen-warm",
                ]
            )

        self.assertEqual(code, 0)
        opener.assert_called_once_with(
            Path("/state/qwen-warm"),
            runtime_profile_sha256=None,
            runtime_code_revision=None,
            template_output_character_limit=None,
            prompt_token_verifier=None,
        )
        self.assertIs(wrapper.call_args.kwargs["ooe_hook"], hook)

    def test_cli_binds_growing_warm_cells_to_the_preload_runtime_profile(
        self,
    ) -> None:
        qwen = _chat(_Runtime())
        hook = object()

        def open_mount(_root, **options):
            return SimpleNamespace(
                hook=hook,
                result_cell_code_revision=options["runtime_code_revision"],
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "Qwen"
            q4 = root / "causal" / "q4-base-v2"
            q4.mkdir(parents=True)
            (q4 / "manifest.json").write_bytes(b"q4-manifest")
            (root / "tokenizer.json").write_bytes(b"tokenizer")
            prompt_tokenizer = SimpleNamespace(
                encode=lambda _text: (11, 12),
                render_no_thinking_prompt=(
                    lambda _system, question: f"prompt:{question}"
                ),
            )
            with (
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=qwen,
                ) as constructor,
                patch(
                    "immer.runtimes.ooe.qwen_warm_bank.open_verified_qwen_warm_bank",
                    side_effect=open_mount,
                ) as opener,
                patch(
                    "immer.cognition.qwen_fertig_chat.QwenFertigChat",
                    side_effect=lambda raw, _fertig, **_options: raw,
                ),
                patch(
                    "immer.runtimes.qwen3_8.encoding.Qwen38Tokenizer",
                    return_value=prompt_tokenizer,
                ),
                redirect_stdout(io.StringIO()),
            ):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--qwen38-root",
                        str(root),
                        "--ooe-warm-root",
                        "/state/qwen-warm",
                    ]
                )

        self.assertEqual(code, 0)
        profile = opener.call_args.kwargs["runtime_profile_sha256"]
        code_revision = opener.call_args.kwargs["runtime_code_revision"]
        self.assertEqual(len(profile), 64)
        self.assertEqual(len(code_revision), 64)
        self.assertEqual(
            opener.call_args.kwargs["template_output_character_limit"],
            64,
        )
        self.assertTrue(callable(opener.call_args.kwargs["prompt_token_verifier"]))
        self.assertEqual(
            constructor.call_args.kwargs["result_cell_code_revision"],
            code_revision,
        )

    def test_growing_warm_code_revision_covers_prompt_identity_sources(self) -> None:
        names = {path.name for path in _qwen38_runtime_code_paths()}
        self.assertIn("encoding.py", names)
        self.assertIn("cartography_probe.py", names)
        self.assertIn("adapter.py", names)
        self.assertIn("qwen_warm_growth.py", names)

    def test_hybrid_warm_profile_binds_every_draft_provider_abi(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            q4 = root / "q4"
            q4.mkdir()
            (q4 / "manifest.json").write_bytes(b"manifest")
            tokenizer = root / "tokenizer.json"
            tokenizer.write_bytes(b"tokenizer")
            args = SimpleNamespace(
                compute_dtype="auto",
                device="auto",
                draft_window=8,
                head_block_rows=2048,
                max_context_tokens=2048,
                max_new_tokens=64,
                max_prompt_tokens=1024,
                q4_threads=None,
                qwen38_anchor_cache=None,
                system_prompt="",
            )
            current = _qwen38_growing_warm_profile(
                args,
                tokenizer_path=tokenizer,
                q4_root=q4,
                fast_mlp_root=None,
                draft_mode="hybrid",
                markov_atlas_path=None,
                markov_o1_retention_path=None,
                runtime_code_revision="a" * 64,
            )
            with patch(
                "immer.cli._QWEN38_MARKOV_DRAFT_ABI",
                "immer.qwen3.8-markov-draft-provider/v999",
            ):
                changed = _qwen38_growing_warm_profile(
                    args,
                    tokenizer_path=tokenizer,
                    q4_root=q4,
                    fast_mlp_root=None,
                    draft_mode="hybrid",
                    markov_atlas_path=None,
                    markov_o1_retention_path=None,
                    runtime_code_revision="a" * 64,
                )
            with patch(
                "immer.cli._QWEN38_MTP_DRAFT_ABI",
                "immer.qwen3.5-mtp-draft-provider/v999",
            ):
                changed_mtp = _qwen38_growing_warm_profile(
                    args,
                    tokenizer_path=tokenizer,
                    q4_root=q4,
                    fast_mlp_root=None,
                    draft_mode="hybrid",
                    markov_atlas_path=None,
                    markov_o1_retention_path=None,
                    runtime_code_revision="a" * 64,
                )
            with patch(
                "immer.cli._QWEN38_HYBRID_DRAFT_ABI",
                "immer.qwen3.8-markov-mtp-hybrid-provider/v999",
            ):
                changed_hybrid = _qwen38_growing_warm_profile(
                    args,
                    tokenizer_path=tokenizer,
                    q4_root=q4,
                    fast_mlp_root=None,
                    draft_mode="hybrid",
                    markov_atlas_path=None,
                    markov_o1_retention_path=None,
                    runtime_code_revision="a" * 64,
                )

        self.assertNotEqual(current, changed)
        self.assertNotEqual(current, changed_mtp)
        self.assertNotEqual(current, changed_hybrid)

    def test_cli_explicit_layout_does_not_inherit_deployed_q4(self) -> None:
        qwen = _chat(_Runtime())
        with tempfile.TemporaryDirectory() as temporary:
            deployed = Path(temporary) / "deployed"
            (deployed / "causal" / "q4-base-v2").mkdir(parents=True)
            with (
                patch.dict("os.environ", {}, clear=True),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_ROOT",
                    deployed,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=qwen,
                ) as constructor,
                redirect_stdout(io.StringIO()),
            ):
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
            with (
                patch.dict(
                    "os.environ",
                    {"IMMER_QWEN38_FAST_MLP": str(deployed_fast)},
                    clear=True,
                ),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_ROOT",
                    deployed,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=qwen,
                ) as constructor,
                redirect_stdout(io.StringIO()),
            ):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--no-fast-mlp",
                        "--no-markov-draft",
                        "--raw-qwen",
                    ]
                )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(
            options["q4_root"],
            str(deployed / "causal" / "q4-base-v2"),
        )

    def test_cli_mounts_direct_markov_page_execution_without_legacy_fast_mlp(self) -> None:
        qwen = _chat(_Runtime())
        with (
            patch.dict(
                "os.environ",
                {"IMMER_QWEN38_FAST_MLP": "/state/legacy-fast"},
                clear=True,
            ),
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                return_value=qwen,
            ) as constructor,
            redirect_stdout(io.StringIO()),
        ):
            code = main(
                [
                    "chat",
                    "hello",
                    "--qwen38-causal-bundle",
                    "/models/qwen.causal",
                    "--qwen38-tokenizer",
                    "/models/tokenizer.json",
                    "--qwen38-q4",
                    "/models/q4",
                    "--mlp-page-state",
                    "/state/pages.json",
                    "--mlp-page-width",
                    "160",
                    "--no-markov-draft",
                    "--raw-qwen",
                ]
            )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertIsNone(options["fast_mlp_root"])
        self.assertEqual(options["mlp_page_state_path"], Path("/state/pages.json"))
        self.assertEqual(options["mlp_page_route_width"], 160)
        self.assertIsNone(options["fast_mlp_root"])
        self.assertIsNone(options["fast_mlp_active_layers"])

    def test_cli_deployment_defaults_to_full_q4(self) -> None:
        qwen = _chat(_Runtime())
        with tempfile.TemporaryDirectory() as temporary:
            deployed = Path(temporary) / "deployed"
            (deployed / "causal" / "q4-base-v2").mkdir(parents=True)
            with (
                patch.dict("os.environ", {}, clear=True),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_ROOT",
                    deployed,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=qwen,
                ) as constructor,
                redirect_stdout(io.StringIO()),
            ):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--raw-qwen",
                        "--no-markov-draft",
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
        self.assertIsNone(options["draft_mode"])
        self.assertIsNone(options["markov_draft_state_path"])

    def test_cli_deployment_mounts_the_persistent_markov_token_council(
        self,
    ) -> None:
        qwen = _chat(_Runtime())
        with tempfile.TemporaryDirectory() as temporary:
            deployed = Path(temporary) / "deployed"
            (deployed / "causal" / "q4-base-v2").mkdir(parents=True)
            state = Path(temporary) / "qwen-markov.bin"
            state.write_bytes(b"fixture")
            with (
                patch.dict("os.environ", {}, clear=True),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_ROOT",
                    deployed,
                ),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_MARKOV_DRAFT_STATE",
                    state,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=qwen,
                ) as constructor,
                redirect_stdout(io.StringIO()),
            ):
                code = main(["chat", "hello", "--raw-qwen"])

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(options["draft_mode"], "markov")
        self.assertEqual(options["markov_draft_state_path"], str(state))

    def test_cli_deployment_uses_hybrid_novelty_fallback(self) -> None:
        qwen = _chat(_Runtime())
        with tempfile.TemporaryDirectory() as temporary:
            deployed = Path(temporary) / "deployed"
            q4 = deployed / "causal" / "q4-base-v3-mtp"
            q4.mkdir(parents=True)
            markov_state = Path(temporary) / "qwen-markov.bin"
            markov_state.write_bytes(b"fixture")
            mtp_state = Path(temporary) / "qwen-mtp.json"
            with (
                patch.dict("os.environ", {}, clear=True),
                patch("immer.cli._QWEN38_DEPLOYMENT_ROOT", deployed),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_MARKOV_DRAFT_STATE",
                    markov_state,
                ),
                patch("immer.cli._QWEN38_DEPLOYMENT_MTP_STATE", mtp_state),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=qwen,
                ) as constructor,
                redirect_stdout(io.StringIO()),
            ):
                code = main(["chat", "hello", "--raw-qwen"])

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(options["q4_root"], str(q4))
        self.assertEqual(options["draft_mode"], "hybrid")
        self.assertEqual(options["markov_draft_state_path"], str(markov_state))
        self.assertEqual(options["mtp_draft_state_path"], str(mtp_state))

    def test_cli_deployment_keeps_markov_as_an_explicit_opt_out(self) -> None:
        qwen = _chat(_Runtime())
        with tempfile.TemporaryDirectory() as temporary:
            deployed = Path(temporary) / "deployed"
            q4 = deployed / "causal" / "q4-base-v3-mtp"
            q4.mkdir(parents=True)
            markov_state = Path(temporary) / "qwen-markov.bin"
            markov_state.write_bytes(b"fixture")
            mtp_state = Path(temporary) / "qwen-mtp.json"
            with (
                patch.dict("os.environ", {}, clear=True),
                patch("immer.cli._QWEN38_DEPLOYMENT_ROOT", deployed),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_MARKOV_DRAFT_STATE",
                    markov_state,
                ),
                patch("immer.cli._QWEN38_DEPLOYMENT_MTP_STATE", mtp_state),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=qwen,
                ) as constructor,
                redirect_stdout(io.StringIO()),
            ):
                code = main(
                    ["chat", "hello", "--raw-qwen", "--draft-mode", "markov"]
                )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(options["q4_root"], str(q4))
        self.assertEqual(options["draft_mode"], "markov")
        self.assertEqual(options["markov_draft_state_path"], str(markov_state))
        self.assertIsNone(options["mtp_draft_state_path"])

    def test_cli_anchor_keeps_the_compatible_markov_provider(self) -> None:
        qwen = _chat(_Runtime())
        with tempfile.TemporaryDirectory() as temporary:
            deployed = Path(temporary) / "deployed"
            q4 = deployed / "causal" / "q4-base-v3-mtp"
            q4.mkdir(parents=True)
            markov_state = Path(temporary) / "qwen-markov.bin"
            markov_state.write_bytes(b"fixture")
            anchor = Path(temporary) / "anchors"
            with (
                patch.dict("os.environ", {}, clear=True),
                patch("immer.cli._QWEN38_DEPLOYMENT_ROOT", deployed),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_MARKOV_DRAFT_STATE",
                    markov_state,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=qwen,
                ) as constructor,
                redirect_stdout(io.StringIO()),
            ):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--raw-qwen",
                        "--qwen38-anchor-cache",
                        str(anchor),
                    ]
                )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(options["q4_root"], str(q4))
        self.assertEqual(options["draft_mode"], "markov")
        self.assertEqual(options["markov_draft_state_path"], str(markov_state))
        self.assertIsNone(options["mtp_draft_state_path"])

    def test_cli_explicit_mtp_uses_persistent_calibration_state(self) -> None:
        qwen = _chat(_Runtime())
        with tempfile.TemporaryDirectory() as temporary:
            deployed = Path(temporary) / "deployed"
            deployed.mkdir()
            q4 = deployed / "causal" / "q4-base-v3-mtp"
            q4.mkdir(parents=True)
            state = Path(temporary) / "qwen-mtp.json"
            with (
                patch.dict("os.environ", {}, clear=True),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_ROOT",
                    deployed,
                ),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_MTP_STATE",
                    state,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=qwen,
                ) as constructor,
                redirect_stdout(io.StringIO()),
            ):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--raw-qwen",
                        "--draft-mode",
                        "mtp",
                    ]
                )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(options["draft_mode"], "mtp")
        self.assertEqual(options["q4_root"], str(q4))
        self.assertIsNone(options["markov_draft_state_path"])
        self.assertEqual(options["mtp_draft_state_path"], str(state))

    def test_cli_hybrid_rejects_a_target_bank_without_mtp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            deployed = Path(temporary) / "deployed"
            (deployed / "causal" / "q4-base-v2").mkdir(parents=True)
            stderr = io.StringIO()
            with (
                patch.dict("os.environ", {}, clear=True),
                patch("immer.cli._QWEN38_DEPLOYMENT_ROOT", deployed),
                redirect_stderr(stderr),
            ):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--raw-qwen",
                        "--draft-mode",
                        "hybrid",
                    ]
                )

        self.assertEqual(code, 2)
        self.assertIn("hybrid draft mode requires", stderr.getvalue())

    def test_cli_jsonl_reuses_one_loaded_component_for_multiple_requests(self) -> None:
        qwen = _chat(_Runtime())
        output = io.StringIO()
        stream = io.StringIO('{"id":"first","message":"hello"}\nworld\n{"bad":true}\n')
        with (
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                return_value=qwen,
            ) as constructor,
            patch("sys.stdin", stream),
            redirect_stdout(output),
        ):
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
        self.assertIsNone(constructor.call_args.kwargs["text_snapshot_sink"])
        self.assertTrue(qwen.closed)
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["id"], "first")
        self.assertNotIn("id", rows[1])
        self.assertEqual([row["output"] for row in rows], ["local answer"] * 2)

    def test_cli_interactive_reuses_one_loaded_component_for_free_prompts(self) -> None:
        runtime = _Runtime()
        qwen = _chat(runtime)
        output = io.StringIO()
        stream = io.StringIO("hello\nworld\n/clear\nagain\n/quit\n")
        with (
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                return_value=qwen,
            ) as constructor,
            patch(
                "immer.runtimes.qwen3_8.encoding.Qwen38Tokenizer",
                return_value=runtime.tokenizer,
            ),
            patch("sys.stdin", stream),
            redirect_stdout(output),
        ):
            code = main(
                [
                    "chat",
                    "--interactive",
                    "--raw-qwen",
                    "--qwen38-causal-bundle",
                    "/models/qwen.causal",
                    "--qwen38-tokenizer",
                    "/models/tokenizer.json",
                ]
            )

        self.assertEqual(code, 0)
        constructor.assert_called_once()
        self.assertTrue(qwen.closed)
        self.assertEqual(
            output.getvalue(),
            "local answer\nlocal answer\nlocal answer\n",
        )
        self.assertIn(
            Qwen38Tokenizer.render_no_thinking_messages(
                "",
                (
                    ("user", "hello"),
                    ("assistant", "local answer"),
                    ("user", "world"),
                ),
            ),
            runtime.tokenizer.encoded,
        )
        self.assertEqual(
            runtime.tokenizer.encoded[-1],
            Qwen38Tokenizer.render_no_thinking_prompt("", "again"),
        )

    def test_cli_interactive_drops_only_oldest_complete_turns_at_token_limit(
        self,
    ) -> None:
        class SizedTokenizer(_Tokenizer):
            def encode(self, text: str):
                self.encoded.append(text)
                return tuple(range(text.count("<|im_start|>") * 2))

        runtime = _Runtime(tokenizer=SizedTokenizer())
        qwen = _chat(runtime, max_prompt_tokens=8)
        output = io.StringIO()
        stream = io.StringIO("one\ntwo\nthree\n/quit\n")
        with (
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                return_value=qwen,
            ),
            patch(
                "immer.runtimes.qwen3_8.encoding.Qwen38Tokenizer",
                return_value=runtime.tokenizer,
            ),
            patch("sys.stdin", stream),
            redirect_stdout(output),
        ):
            code = main(
                [
                    "chat",
                    "--interactive",
                    "--raw-qwen",
                    "--max-prompt-tokens",
                    "8",
                    "--qwen38-causal-bundle",
                    "/models/qwen.causal",
                    "--qwen38-tokenizer",
                    "/models/tokenizer.json",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(output.getvalue(), "local answer\n" * 3)
        final_prompt = runtime.tokenizer.encoded[-1]
        self.assertNotIn("\none<|im_end|>", final_prompt)
        self.assertIn("\ntwo<|im_end|>", final_prompt)
        self.assertIn("\nthree<|im_end|>", final_prompt)
        self.assertEqual(final_prompt.count("<|im_start|>"), 4)

    def test_context_eviction_never_reenables_single_turn_wrapper(self) -> None:
        class SizedTokenizer(_Tokenizer):
            def encode(self, text: str):
                self.encoded.append(text)
                return tuple(range(text.count("<|im_start|>") * 2))

        runtime = _Runtime(tokenizer=SizedTokenizer())
        qwen = _chat(runtime, max_prompt_tokens=5)
        wrapper = Mock()
        wrapper.handle.side_effect = qwen.handle
        wrapper.close.side_effect = qwen.close
        output = io.StringIO()
        stream = io.StringIO("one\ntwo\n/quit\n")
        with (
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                return_value=qwen,
            ),
            patch(
                "immer.cognition.qwen_fertig_chat.QwenFertigChat",
                return_value=wrapper,
            ),
            patch(
                "immer.runtimes.qwen3_8.encoding.Qwen38Tokenizer",
                return_value=runtime.tokenizer,
            ),
            patch("sys.stdin", stream),
            redirect_stdout(output),
        ):
            code = main(
                [
                    "chat",
                    "--interactive",
                    "--no-ooe-warm",
                    "--max-prompt-tokens",
                    "5",
                    "--qwen38-causal-bundle",
                    "/models/qwen.causal",
                    "--qwen38-tokenizer",
                    "/models/tokenizer.json",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(wrapper.handle.call_count, 1)
        self.assertEqual(len(runtime.model.calls), 2)
        self.assertEqual(output.getvalue(), "local answer\n" * 2)
        self.assertNotIn("\none<|im_end|>", runtime.tokenizer.encoded[-1])
        self.assertIn("\ntwo<|im_end|>", runtime.tokenizer.encoded[-1])

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

    def test_cli_wires_corpus_markov_atlas_without_a_state_file(self) -> None:
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
                        "--markov-atlas",
                        "/state/qwen-markov-atlas.bin",
                    ]
                )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(options["draft_mode"], "markov")
        self.assertEqual(
            options["markov_atlas_path"],
            "/state/qwen-markov-atlas.bin",
        )

    def test_cli_wires_o1_markov_retention(self) -> None:
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
                        "--markov-o1-retention",
                        "/state/o1-retention.json",
                    ]
                )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(options["draft_mode"], "markov")
        self.assertEqual(
            options["markov_o1_retention_path"],
            "/state/o1-retention.json",
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
