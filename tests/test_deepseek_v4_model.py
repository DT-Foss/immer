from __future__ import annotations

import unittest
from unittest import mock

import numpy as np


class _TinyCheckpoint:
    def __init__(self) -> None:
        dim = 128
        qrank = 128
        heads = 2
        hdim = 128
        orank = 128
        inter = 128
        hc = 4
        mix = (2 + hc) * hc
        self.data: dict[str, np.ndarray] = {
            "embed.weight": np.arange(128 * dim, dtype=np.float32).reshape(128, dim)
            / 1000.0,
            "norm.weight": np.ones(dim, dtype=np.float32),
            "head.weight": np.zeros((128, dim), dtype=np.float32),
            "hc_head_fn": np.zeros((hc, hc * dim), dtype=np.float32),
            "hc_head_base": np.zeros(hc, dtype=np.float32),
            "hc_head_scale": np.ones(1, dtype=np.float32),
            "layers.0.attn.attn_sink": np.zeros(heads, dtype=np.float32),
            "layers.0.attn.q_norm.weight": np.ones(qrank, dtype=np.float32),
            "layers.0.attn.kv_norm.weight": np.ones(hdim, dtype=np.float32),
            "layers.0.attn_norm.weight": np.ones(dim, dtype=np.float32),
            "layers.0.ffn_norm.weight": np.ones(dim, dtype=np.float32),
            "layers.0.ffn.gate.weight": np.zeros((2, dim), dtype=np.float32),
            "layers.0.ffn.gate.tid2eid": np.zeros((128, 1), dtype=np.int64),
            "layers.0.attn.wq_a.weight": np.zeros((qrank, dim), dtype=np.float32),
            "layers.0.attn.wq_b.weight": np.zeros(
                (heads * hdim, qrank), dtype=np.float32
            ),
            "layers.0.attn.wkv.weight": np.zeros((hdim, dim), dtype=np.float32),
            "layers.0.attn.wo_a.weight": np.zeros(
                (orank, heads * hdim), dtype=np.float32
            ),
            "layers.0.attn.wo_b.weight": np.zeros((dim, orank), dtype=np.float32),
            "layers.0.ffn.shared_experts.w1.weight": np.zeros(
                (inter, dim), dtype=np.float32
            ),
            "layers.0.ffn.shared_experts.w2.weight": np.zeros(
                (dim, inter), dtype=np.float32
            ),
            "layers.0.ffn.shared_experts.w3.weight": np.zeros(
                (inter, dim), dtype=np.float32
            ),
        }
        for branch in ("attn", "ffn"):
            self.data[f"layers.0.hc_{branch}_fn"] = np.zeros(
                (mix, hc * dim), dtype=np.float32
            )
            self.data[f"layers.0.hc_{branch}_base"] = np.zeros(mix, dtype=np.float32)
            self.data[f"layers.0.hc_{branch}_scale"] = np.ones(3, dtype=np.float32)
        for expert in range(2):
            base = f"layers.0.ffn.experts.{expert}"
            self.data[f"{base}.w1.weight"] = np.zeros((inter, dim), dtype=np.float32)
            self.data[f"{base}.w2.weight"] = np.zeros((dim, inter), dtype=np.float32)
            self.data[f"{base}.w3.weight"] = np.zeros((inter, dim), dtype=np.float32)
        self.dtypes = {
            name: ("I64" if value.dtype == np.int64 else "F32")
            for name, value in self.data.items()
        }
        self._body = 0

    def find(self, name: str) -> dict:
        value = self.data[name]
        dtype = self.dtypes[name]
        itemsize = {
            "BF16": 2,
            "F8_E4M3": 1,
            "F8_E8M0": 1,
            "I8": 1,
        }.get(dtype, value.dtype.itemsize)
        return {
            "name": name,
            "dtype": dtype,
            "shape": list(value.shape),
            "offset_in_shard": [0, value.size * itemsize],
        }

    def tensor(self, name: str) -> np.ndarray:
        value = self.data[name].copy()
        self._body += value.nbytes
        return value

    def rows(self, name: str, start_row: int = 0, n_rows: int = 8, **_) -> np.ndarray:
        value = self.data[name][start_row : start_row + n_rows].copy()
        self._body += value.nbytes
        return value

    def inventory(self) -> dict:
        return {"tensors": [self.find(name) for name in self.data]}

    def metrics(self) -> dict:
        return {
            "network_or_source_body_bytes": self._body,
            "inventory_source_fingerprint": "tiny",
        }


class _QuantizedTinyCheckpoint(_TinyCheckpoint):
    """Decoded safetensors values with the real V4 storage contracts."""

    def __init__(self) -> None:
        super().__init__()
        bf16 = {
            "embed.weight",
            "norm.weight",
            "head.weight",
            "layers.0.attn.q_norm.weight",
            "layers.0.attn.kv_norm.weight",
            "layers.0.attn_norm.weight",
            "layers.0.ffn_norm.weight",
            "layers.0.ffn.gate.weight",
        }
        for name in bf16:
            self.dtypes[name] = "BF16"
        fp8_weights = [
            name
            for name in self.data
            if name.endswith(".weight")
            and (".attn.w" in name or ".ffn.shared_experts." in name)
        ]
        for name in fp8_weights:
            out_dim, in_dim = self.data[name].shape
            self.dtypes[name] = "F8_E4M3"
            scale_name = name.removesuffix("weight") + "scale"
            self.data[scale_name] = np.ones(
                ((out_dim + 127) // 128, (in_dim + 127) // 128),
                dtype=np.float32,
            )
            self.dtypes[scale_name] = "F8_E8M0"
        expert_weights = [
            name
            for name in self.data
            if ".ffn.experts." in name and name.endswith(".weight")
        ]
        for name in expert_weights:
            out_dim, in_dim = self.data[name].shape
            self.data[name] = np.zeros((out_dim, in_dim // 2), dtype=np.int8)
            self.dtypes[name] = "I8"
            scale_name = name.removesuffix("weight") + "scale"
            self.data[scale_name] = np.ones((out_dim, in_dim // 32), dtype=np.float32)
            self.dtypes[scale_name] = "F8_E8M0"


class _PrefetchQuantizedTinyCheckpoint(_QuantizedTinyCheckpoint):
    """Quantized tiny model whose routed experts use adjacent raw ranges."""

    def __init__(self) -> None:
        super().__init__()
        from test_deepseek_v4_pager import _MultiEncodedExpertSource

        bases = [f"layers.0.ffn.experts.{expert}" for expert in range(2)]
        self.encoded_experts = _MultiEncodedExpertSource(bases)
        self.data.update(self.encoded_experts.data)
        self.data["layers.0.ffn.gate.tid2eid"] = np.tile(
            np.asarray([[0, 1]], dtype=np.int64), (128, 1)
        )

    def find(self, name: str) -> dict:
        if name in self.encoded_experts.meta:
            return self.encoded_experts.find(name)
        return super().find(name)

    def raw_bytes(self, shard: str, offset: int, length: int) -> bytes:
        return self.encoded_experts.raw_bytes(shard, offset, length)

    def metrics(self) -> dict:
        result = super().metrics()
        result["network_or_source_body_bytes"] += self.encoded_experts.metrics()[
            "network_or_source_body_bytes"
        ]
        return result


class _TwoLayerPrefetchQuantizedTinyCheckpoint(_PrefetchQuantizedTinyCheckpoint):
    """Two identical official blocks with raw-addressable routed experts."""

    def __init__(self) -> None:
        super().__init__()
        from test_deepseek_v4_pager import _MultiEncodedExpertSource

        for name, value in tuple(self.data.items()):
            if not name.startswith("layers.0.") or ".ffn.experts." in name:
                continue
            cloned = name.replace("layers.0.", "layers.1.", 1)
            self.data[cloned] = value.copy()
            self.dtypes[cloned] = self.dtypes[name]
        bases = [
            f"layers.{layer}.ffn.experts.{expert}"
            for layer in range(2)
            for expert in range(2)
        ]
        self.encoded_experts = _MultiEncodedExpertSource(bases)
        self.data.update(self.encoded_experts.data)
        self.data["layers.1.ffn.gate.tid2eid"] = np.tile(
            np.asarray([[0, 1]], dtype=np.int64), (128, 1)
        )
        self.dtypes["layers.1.ffn.gate.tid2eid"] = "I64"


class _CompressedTinyCheckpoint(_TinyCheckpoint):
    def __init__(
        self, *, random_weights: bool = False, compress_ratio: int = 4
    ) -> None:
        super().__init__()
        dim = head_dim = index_dim = qrank = 128
        heads = 2
        base = "layers.0.attn"
        overlap = compress_ratio == 4
        coff = 2 if overlap else 1
        added = {
            f"{base}.compressor.ape": np.zeros(
                (compress_ratio, coff * head_dim), np.float32
            ),
            f"{base}.compressor.norm.weight": np.ones(head_dim, np.float32),
            f"{base}.compressor.wkv.weight": np.zeros(
                (coff * head_dim, dim), np.float32
            ),
            f"{base}.compressor.wgate.weight": np.zeros(
                (coff * head_dim, dim), np.float32
            ),
        }
        if overlap:
            added.update(
                {
                    f"{base}.indexer.wq_b.weight": np.zeros(
                        (heads * index_dim, qrank), np.float32
                    ),
                    f"{base}.indexer.weights_proj.weight": np.zeros(
                        (heads, dim), np.float32
                    ),
                    f"{base}.indexer.compressor.ape": np.zeros(
                        (4, 2 * index_dim), np.float32
                    ),
                    f"{base}.indexer.compressor.norm.weight": np.ones(
                        index_dim, np.float32
                    ),
                    f"{base}.indexer.compressor.wkv.weight": np.zeros(
                        (2 * index_dim, dim), np.float32
                    ),
                    f"{base}.indexer.compressor.wgate.weight": np.zeros(
                        (2 * index_dim, dim), np.float32
                    ),
                }
            )
        self.data.update(added)
        self.dtypes.update({name: "F32" for name in added})
        if random_weights:
            rng = np.random.default_rng(31)
            for name, value in self.data.items():
                if value.dtype.kind != "f":
                    continue
                if value.ndim == 2 and not name.endswith("norm.weight"):
                    self.data[name] = rng.normal(0.0, 0.015, value.shape).astype(
                        np.float32
                    )
                elif name.endswith(".ape"):
                    self.data[name] = rng.normal(0.0, 0.01, value.shape).astype(
                        np.float32
                    )


def _config(compress_ratio: int = 0, *, n_layers: int = 1):
    from immer.runtimes.deepseek_v4 import DeepSeekV4Config

    return DeepSeekV4Config.from_mapping(
        {
            "architectures": ["DeepseekV4ForCausalLM"],
            "model_type": "deepseek_v4",
            "vocab_size": 128,
            "hidden_size": 128,
            "num_hidden_layers": n_layers,
            "num_attention_heads": 2,
            "head_dim": 128,
            "qk_rope_head_dim": 64,
            "q_lora_rank": 128,
            "o_lora_rank": 128,
            "o_groups": 1,
            "moe_intermediate_size": 128,
            "n_routed_experts": 2,
            "n_shared_experts": 1,
            "num_experts_per_tok": 1,
            "num_hash_layers": 1,
            "rms_norm_eps": 1e-6,
            "hc_mult": 4,
            "hc_sinkhorn_iters": 2,
            "hc_eps": 1e-6,
            "sliding_window": 16,
            "compress_ratios": [compress_ratio] * n_layers,
            "rope_theta": 10000,
            "compress_rope_theta": 40000,
            "max_position_embeddings": 1024,
            "rope_scaling": {
                "original_max_position_embeddings": 1024,
                "factor": 2,
                "beta_fast": 32,
                "beta_slow": 1,
            },
            "index_n_heads": 2,
            "index_head_dim": 128,
            "index_topk": 16,
            "scoring_func": "sqrtsoftplus",
            "routed_scaling_factor": 1.5,
            "swiglu_limit": 10,
            "expert_dtype": "fp4",
            "dspark_target_layer_ids": [],
        }
    )


class StreamedDeepSeekV4Tests(unittest.TestCase):
    def test_exact_expert_prefetch_preserves_output_and_router_choices(self) -> None:
        from dataclasses import replace

        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager, StreamedDeepSeekV4

        config = replace(_config(), n_activated_experts=2)

        def execute(enabled: bool, *, width: int | None = None):
            model = StreamedDeepSeekV4(
                config,
                DeepSeekWeightPager(
                    _PrefetchQuantizedTinyCheckpoint(),
                    device="cpu",
                    compute_dtype="float32",
                    expert_prefetch=enabled,
                    expert_range_coalesce_max_experts=width,
                ),
            )
            calls: list[str] = []
            expert = model._expert

            def traced_expert(x, base, *args, **kwargs):
                if base.endswith("shared_experts"):
                    self.assertIsNone(model.pager._active_prefetch_window)
                    self.assertIsNone(model.pager._active_prefetch_payload)
                calls.append(base)
                return expert(x, base, *args, **kwargs)

            with mock.patch.object(model, "_expert", side_effect=traced_expert):
                hidden, evidence = model.hidden_one_token(7)
            return hidden, evidence, model.pager.metrics(), calls

        expected, expected_evidence, off_metrics, off_calls = execute(False)
        actual, actual_evidence, q3_metrics, on_calls = execute(True)
        _, _, adjacent_metrics, _ = execute(True, width=2)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual_evidence.selected_experts, ((0, 1),))
        self.assertEqual(
            actual_evidence.selected_experts,
            expected_evidence.selected_experts,
        )
        self.assertEqual(off_metrics["expert_prefetch_policy"], "disabled")
        self.assertEqual(off_metrics["expert_prefetch_transport_policy"], "disabled")
        self.assertEqual(
            q3_metrics["expert_prefetch_policy"],
            "exact-router-window-q3-a2/v2",
        )
        self.assertEqual(
            q3_metrics["expert_prefetch_transport_policy"],
            "streamer-exact-range/v1",
        )
        self.assertEqual(
            adjacent_metrics["expert_prefetch_policy"],
            "exact-router-window-q3-a2-adjacent-pairs/v3",
        )
        self.assertEqual(
            adjacent_metrics["expert_prefetch_transport_policy"],
            "streamer-exact-leaf-adjacent-envelope/v2",
        )
        self.assertEqual(q3_metrics["expert_prefetch_submitted"], 2)
        self.assertEqual(q3_metrics["expert_prefetch_consumed"], 2)
        self.assertEqual(on_calls, off_calls)
        self.assertEqual(
            on_calls,
            [
                "layers.0.ffn.experts.0",
                "layers.0.ffn.experts.1",
                "layers.0.ffn.shared_experts",
            ],
        )

    def test_markov_micro_windows_are_bit_exact_and_feed_the_next_layer(self) -> None:
        from dataclasses import replace

        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager, StreamedDeepSeekV4
        from immer.runtimes.deepseek_v4.route_markov import (
            LayerMarkovExpertPredictor,
            LayerTokenRoutes,
            PromptRouteObservation,
        )

        config = replace(
            _config(n_layers=2),
            n_activated_experts=2,
            n_hash_layers=2,
        )
        predictor = LayerMarkovExpertPredictor(n_experts=2)
        predictor.observe(
            PromptRouteObservation(
                "training-route",
                (
                    LayerTokenRoutes(0, ((0, 1), (0, 1))),
                    LayerTokenRoutes(1, ((0, 1), (0, 1))),
                ),
            )
        )

        def execute(with_predictor: bool, *, direct_max_rows: int = 8):
            pager = DeepSeekWeightPager(
                _TwoLayerPrefetchQuantizedTinyCheckpoint(),
                device="cpu",
                compute_dtype="float32",
                expert_prefetch=True,
            )
            model = StreamedDeepSeekV4(
                config,
                pager,
                max_seq_len=8,
                route_predictor=predictor if with_predictor else None,
                route_prefetch_window_rows=2,
                route_prefetch_direct_max_rows=direct_max_rows,
            )
            hidden, evidence = model.hidden_stateful([[7, 8, 9, 10]])
            return hidden, evidence, pager.metrics(), model.route_prefetch_metrics()

        expected, expected_evidence, baseline, baseline_scheduler = execute(False)
        actual, actual_evidence, markov, direct_scheduler = execute(True)
        aggregated, aggregated_evidence, aggregate_metrics, aggregate_scheduler = (
            execute(True, direct_max_rows=2)
        )

        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(
            actual_evidence.selected_experts,
            expected_evidence.selected_experts,
        )
        self.assertTrue(torch.equal(aggregated, expected))
        self.assertEqual(
            aggregated_evidence.selected_experts,
            expected_evidence.selected_experts,
        )
        self.assertEqual(markov["expert_reservoir_submitted"], 4)
        self.assertEqual(markov["expert_reservoir_prediction_hits"], 4)
        self.assertEqual(markov["expert_reservoir_usable_hits"], 4)
        self.assertEqual(markov["expert_reservoir_misses"], 0)
        self.assertFalse(markov["expert_reservoir_active"])
        self.assertEqual(aggregate_metrics["expert_reservoir_submitted"], 2)
        self.assertEqual(aggregate_metrics["expert_reservoir_usable_hits"], 2)
        self.assertEqual(direct_scheduler["direct_plans"], 1)
        self.assertEqual(direct_scheduler["direct_bindings"], 2)
        self.assertEqual(aggregate_scheduler["aggregate_plans"], 1)
        self.assertEqual(aggregate_scheduler["aggregate_bindings"], 1)
        self.assertFalse(baseline_scheduler["enabled"])
        self.assertGreater(baseline["expert_prefetch_submitted"], 0)

    def test_batched_moe_closes_window_before_shared_and_preserves_index_add(
        self,
    ) -> None:
        from dataclasses import replace

        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager, StreamedDeepSeekV4

        config = replace(_config(), n_activated_experts=2)

        def execute(enabled: bool):
            model = StreamedDeepSeekV4(
                config,
                DeepSeekWeightPager(
                    _PrefetchQuantizedTinyCheckpoint(),
                    device="cpu",
                    compute_dtype="float32",
                    expert_prefetch=enabled,
                ),
            )
            calls: list[str] = []
            expert = model._expert

            def traced_expert(x, base, *args, **kwargs):
                if base.endswith("shared_experts"):
                    self.assertIsNone(model.pager._active_prefetch_window)
                    self.assertIsNone(model.pager._active_prefetch_payload)
                calls.append(base)
                return expert(x, base, *args, **kwargs)

            with mock.patch.object(model, "_expert", side_effect=traced_expert):
                hidden, selected = model._moe(
                    torch.zeros((2, 128), dtype=torch.float32),
                    0,
                    torch.asarray([7, 7], dtype=torch.long),
                )
            return hidden, selected, calls

        expected, expected_selected, expected_calls = execute(False)
        actual, actual_selected, actual_calls = execute(True)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual_selected, expected_selected)
        self.assertEqual(actual_calls, expected_calls)
        self.assertEqual(
            actual_calls,
            [
                "layers.0.ffn.experts.0",
                "layers.0.ffn.experts.1",
                "layers.0.ffn.shared_experts",
            ],
        )

    def test_reverse_io_completion_keeps_serial_moe_bit_exact(self) -> None:
        from dataclasses import replace
        import threading

        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager, StreamedDeepSeekV4
        from test_deepseek_v4_pager import _WindowExpertSource

        config = replace(_config(), n_activated_experts=2)
        reference = StreamedDeepSeekV4(
            config,
            DeepSeekWeightPager(
                _PrefetchQuantizedTinyCheckpoint(),
                device="cpu",
                compute_dtype="float32",
            ),
        )
        expected, expected_evidence = reference.hidden_one_token(7)

        bases = [f"layers.0.ffn.experts.{expert}" for expert in range(2)]
        source = _PrefetchQuantizedTinyCheckpoint()
        reverse = _WindowExpertSource(bases)
        source.encoded_experts = reverse
        source.data.update(reverse.data)
        model = StreamedDeepSeekV4(
            config,
            DeepSeekWeightPager(source, device="cpu", compute_dtype="float32"),
        )
        finished = threading.Event()
        results: list[tuple[object, object]] = []
        raised: list[BaseException] = []
        calls: list[str] = []
        expert = model._expert

        def traced_expert(x, base, *args, **kwargs):
            calls.append(base)
            return expert(x, base, *args, **kwargs)

        def execute() -> None:
            try:
                with mock.patch.object(model, "_expert", side_effect=traced_expert):
                    results.append(model.hidden_one_token(7))
            except BaseException as exc:
                raised.append(exc)
            finally:
                finished.set()

        worker = threading.Thread(target=execute)
        worker.start()
        self.assertTrue(reverse.started[bases[0]].wait(timeout=2))
        self.assertTrue(reverse.started[bases[1]].wait(timeout=2))
        reverse.release[bases[1]].set()
        self.assertTrue(reverse.completed[bases[1]].wait(timeout=2))
        self.assertFalse(finished.is_set())
        reverse.release[bases[0]].set()
        self.assertTrue(finished.wait(timeout=3))
        worker.join(timeout=1)
        self.assertEqual(raised, [])
        self.assertEqual(len(results), 1)
        actual, evidence = results[0]
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(evidence.selected_experts, expected_evidence.selected_experts)
        self.assertEqual(
            calls,
            [
                "layers.0.ffn.experts.0",
                "layers.0.ffn.experts.1",
                "layers.0.ffn.shared_experts",
            ],
        )

    def test_moe_failure_detaches_blocked_next_prefetch(self) -> None:
        from dataclasses import replace
        import threading
        from unittest import mock

        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager, StreamedDeepSeekV4
        from test_deepseek_v4_pager import _BlockingExpertSource

        bases = [f"layers.0.ffn.experts.{expert}" for expert in range(2)]
        source = _PrefetchQuantizedTinyCheckpoint()
        blocking = _BlockingExpertSource(bases)
        blocking.blocked_shard = "expert-1.safetensors"
        source.encoded_experts = blocking
        source.data.update(blocking.data)
        pager = DeepSeekWeightPager(
            source,
            device="cpu",
            compute_dtype="float32",
        )
        model = StreamedDeepSeekV4(
            replace(_config(), n_activated_experts=2),
            pager,
        )
        finished = threading.Event()
        raised: list[BaseException] = []

        def fail_current(*_args, **_kwargs):
            if not blocking.started.wait(timeout=2):
                raise AssertionError("next expert prefetch did not start")
            raise RuntimeError("simulated current expert failure")

        def execute() -> None:
            try:
                with mock.patch.object(
                    pager, "_expert_with_payload", side_effect=fail_current
                ):
                    model._moe_one(
                        torch.zeros((1, 1, 4, 128), dtype=torch.float32),
                        0,
                        7,
                    )
            except BaseException as exc:
                raised.append(exc)
            finally:
                finished.set()

        worker = threading.Thread(target=execute)
        worker.start()
        self.assertTrue(finished.wait(timeout=3))
        self.assertEqual(len(raised), 1)
        self.assertRegex(str(raised[0]), "current expert failure")
        self.assertTrue(pager.metrics()["expert_prefetch_draining"])
        blocking.release_read.set()
        self.assertTrue(blocking.completed.wait(timeout=2))
        worker.join(timeout=1)
        pager.release()

    def test_one_token_executes_embed_hc_attention_moe_and_final_norm(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager, StreamedDeepSeekV4

        source = _QuantizedTinyCheckpoint()
        model = StreamedDeepSeekV4(
            _config(),
            DeepSeekWeightPager(source, device="cpu", compute_dtype="float32"),
        )
        hidden, evidence = model.hidden_one_token(7)
        self.assertEqual(tuple(hidden.shape), (1, 1, 128))
        self.assertTrue(torch.isfinite(hidden).all())
        self.assertEqual(evidence.layers_executed, 1)
        self.assertTrue(evidence.complete_layer_stack)
        self.assertEqual(evidence.context_mode, "isolated_position_zero")
        self.assertFalse(evidence.stateful_kv_cache)
        self.assertEqual(evidence.selected_experts, ((0,),))
        self.assertGreater(evidence.linear_calls, 0)
        self.assertGreater(evidence.source_body_bytes, 0)
        metrics = model.pager.metrics()
        self.assertEqual(metrics["materialized_weight_releases"], evidence.linear_calls)
        self.assertEqual(metrics["release_boundaries"], 1)
        self.assertEqual(metrics["mps_cache_purges"], 0)

    def test_layer_prefix_is_explicitly_marked_incomplete(self) -> None:
        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager, StreamedDeepSeekV4

        source = _TinyCheckpoint()
        model = StreamedDeepSeekV4(
            _config(n_layers=2),
            DeepSeekWeightPager(source, device="cpu", compute_dtype="float32"),
        )
        hidden, evidence = model.hidden_one_token(1, stop_after_layer=1)
        self.assertFalse(evidence.complete_layer_stack)
        self.assertEqual(evidence.layers_executed, 1)
        self.assertEqual(evidence.checkpoint_layers, 2)
        self.assertEqual(tuple(hidden.shape), (1, 1, 4, 128))

    def test_stateful_compressed_full_prefill_matches_tokenwise_decode(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager, StreamedDeepSeekV4

        token_ids = [[3, 7, 11, 13, 17, 19]]
        full = StreamedDeepSeekV4(
            _config(4),
            DeepSeekWeightPager(
                _CompressedTinyCheckpoint(random_weights=True),
                device="cpu",
                compute_dtype="float32",
            ),
            max_seq_len=16,
        )
        tokenwise = StreamedDeepSeekV4(
            _config(4),
            DeepSeekWeightPager(
                _CompressedTinyCheckpoint(random_weights=True),
                device="cpu",
                compute_dtype="float32",
            ),
            max_seq_len=16,
        )
        full_hidden, full_evidence = full.prefill(token_ids, tokenwise=False)
        step_hidden, step_evidence = tokenwise.prefill(token_ids, tokenwise=True)
        torch.testing.assert_close(step_hidden, full_hidden, atol=2e-4, rtol=2e-4)
        self.assertEqual(len(full_evidence), 1)
        self.assertEqual(len(step_evidence), len(token_ids[0]))
        self.assertTrue(all(row.stateful_kv_cache for row in step_evidence))
        self.assertEqual(tokenwise.next_position, len(token_ids[0]))
        self.assertGreater(tokenwise.attention_state_bytes, 0)

    def test_crsa_graft_history_is_equal_for_full_and_tokenwise_prefill(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager, StreamedDeepSeekV4
        from immer.runtimes.deepseek_v4.graft import DeepSeekV4CrsaGraft

        ids = [[2, 5, 8, 12]]

        def model():
            return StreamedDeepSeekV4(
                _config(4),
                DeepSeekWeightPager(
                    _CompressedTinyCheckpoint(random_weights=True),
                    device="cpu",
                    compute_dtype="float32",
                ),
                graft=DeepSeekV4CrsaGraft(mode="crsa", alpha=0.05, max_history=16),
                graft_layer=0,
                max_seq_len=16,
            )

        full_hidden, _ = model().prefill(ids, tokenwise=False)
        streamed = model()
        step_hidden, evidence = streamed.prefill(ids, tokenwise=True)
        torch.testing.assert_close(step_hidden, full_hidden, atol=2e-4, rtol=2e-4)
        self.assertEqual(evidence[-1].graft_mode, "crsa")
        self.assertEqual(evidence[-1].graft_history_tokens, len(ids[0]))
        streamed.reset_state(release=True)
        self.assertEqual(streamed.next_position, 0)
        self.assertEqual(streamed.attention_state_bytes, 0)

    def test_failed_stateful_forward_requires_explicit_reset(self) -> None:
        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager, StreamedDeepSeekV4

        class RaisingGraft:
            mode = "crsa"
            alpha = 0.1

            def forward(self, _hidden, **_kwargs):
                raise RuntimeError("injected graft failure")

        model = StreamedDeepSeekV4(
            _config(),
            DeepSeekWeightPager(
                _TinyCheckpoint(), device="cpu", compute_dtype="float32"
            ),
            graft=RaisingGraft(),
            graft_layer=0,
            max_seq_len=8,
        )
        with self.assertRaisesRegex(RuntimeError, "injected graft failure"):
            model.prefill([[3]], tokenwise=True)
        model.graft = None
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            model.prefill([[3]], tokenwise=True, reset=False)
        model.reset_state()
        hidden, _ = model.prefill([[3]], tokenwise=True, reset=False)
        self.assertEqual(tuple(hidden.shape), (1, 1, 128))


if __name__ == "__main__":
    unittest.main()
