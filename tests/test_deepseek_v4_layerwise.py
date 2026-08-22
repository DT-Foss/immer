from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from immer.runtimes.deepseek_v4 import (
    DeepSeekWeightPager,
    LayerwiseError,
    LayerwiseItem,
    LayerwiseScorer,
    OFFICIAL_SOURCE_SAFE_BYTES,
    StreamedDeepSeekV4,
    build_layerwise_plan,
)
from test_deepseek_v4_model import _CompressedTinyCheckpoint, _config


_SHA = "0" * 64


def _source(*, ratio: int = 0) -> _CompressedTinyCheckpoint:
    source = _CompressedTinyCheckpoint(random_weights=True, compress_ratio=ratio)
    # Keep head scores non-degenerate while preserving modest BF16 magnitudes.
    rng = np.random.default_rng(719)
    source.data["head.weight"] = rng.normal(0, 0.02, (128, 128)).astype(np.float32)
    return source


def _two_layer_source() -> _CompressedTinyCheckpoint:
    source = _source()
    for name, value in tuple(source.data.items()):
        if name.startswith("layers.0."):
            copied = name.replace("layers.0.", "layers.1.", 1)
            source.data[copied] = value.copy()
            source.dtypes[copied] = source.dtypes[name]
    return source


def _model(
    source: _CompressedTinyCheckpoint,
    *,
    ratio: int = 0,
    batch: int = 2,
    layers: int = 1,
) -> StreamedDeepSeekV4:
    config = _config(ratio, n_layers=layers)
    if layers > 1:
        config = replace(config, n_hash_layers=layers)
    pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="bfloat16")
    return StreamedDeepSeekV4(config, pager, max_batch_size=batch, max_seq_len=16)


def _items() -> tuple[LayerwiseItem, ...]:
    return (
        LayerwiseItem("short", (1, 2, 3), (11, 12, 13, 14), (0, 1, 2, 3), 1),
        LayerwiseItem("long", (4, 5, 6, 7, 8), (11, 12, 13, 14), (0, 1, 2, 3), 2),
    )


def _scorer(
    model: StreamedDeepSeekV4,
    run_dir: Path,
    *,
    items: tuple[LayerwiseItem, ...] | None = None,
    modes: tuple[str, ...] = ("off",),
    alpha: float = 0.05,
    padding: str = "right",
    microbatch_size: int = 2,
) -> LayerwiseScorer:
    return LayerwiseScorer(
        model,
        _items() if items is None else items,
        run_dir=run_dir,
        modes=modes,
        microbatch_size=microbatch_size,
        padding=padding,
        graft_layer=0,
        graft_alpha=alpha,
        tokenizer_sha256=_SHA,
        dataset_sha256=_SHA,
        config_sha256=_SHA,
        disk_margin_bytes=0,
    )


def _body_sha(document: object) -> str:
    encoded = json.dumps(
        document,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, document: object) -> None:
    path.write_text(
        json.dumps(
            document,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


class LayerwiseModelApiTests(unittest.TestCase):
    def test_right_padding_preserves_valid_prefix_with_compressed_attention(
        self,
    ) -> None:
        source = _source(ratio=4)
        model = _model(source, ratio=4, batch=2)
        ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
        mask = torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, True]]
        )
        padded, selected = model.forward_prefill_layer(
            model.embed_batch(ids), ids, layer=0, token_mask=mask
        )
        short_ids = torch.tensor([[1, 2, 3]])
        short, _ = model.forward_prefill_layer(
            model.embed_batch(short_ids), short_ids, layer=0
        )
        torch.testing.assert_close(padded[0, :3], short[0], rtol=0, atol=0)
        self.assertEqual(selected[3], ())
        self.assertEqual(selected[4], ())
        self.assertIsNone(model._attention_states[0])

    def test_layer_state_cannot_leak_between_microbatches(self) -> None:
        model = _model(_source(), batch=1)
        ids = torch.tensor([[1, 2, 3]])
        hidden = model.embed_batch(ids)
        first, _ = model.forward_prefill_layer(hidden, ids, layer=0)
        second, _ = model.forward_prefill_layer(hidden, ids, layer=0)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(model.attention_state_bytes, 0)

    def test_mask_rejects_non_prefix_shape(self) -> None:
        model = _model(_source(), batch=1)
        ids = torch.tensor([[1, 2, 3]])
        with self.assertRaisesRegex(ValueError, "right-padded"):
            model.forward_prefill_layer(
                model.embed_batch(ids),
                ids,
                layer=0,
                token_mask=torch.tensor([[True, False, True]]),
            )


class LayerwiseScorerTests(unittest.TestCase):
    def test_layer_major_logits_match_item_major_decoder(self) -> None:
        items = _items()
        direct_source = _source()
        direct = _model(direct_source, batch=1)
        expected: dict[str, list[float]] = {}
        for item in items:
            hidden, _ = direct.prefill([item.prompt_token_ids], tokenwise=False)
            logits = direct.pager.candidate_logits(
                hidden[:, -1], item.candidate_token_ids
            )
            expected[item.item_id] = [
                float(value) for value in logits[0].detach().to(torch.float32)
            ]
            direct.reset_state(release=True)

        with tempfile.TemporaryDirectory() as temporary:
            model = _model(_source(), batch=2)
            result = _scorer(model, Path(temporary) / "run").run()
            observed = {
                row["item_id"]: row["candidate_scores"]
                for row in result["modes"]["off"]["items"]
            }
        for item_id in expected:
            np.testing.assert_allclose(
                observed[item_id], expected[item_id], rtol=0, atol=0
            )

    def test_padded_and_exact_length_plans_have_same_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            padded = _scorer(_model(_source(), batch=2), root / "padded").run()
            exact = _scorer(
                _model(_source(), batch=2), root / "exact", padding="exact-length"
            ).run()
        self.assertEqual(padded["modes"], exact["modes"])

    def test_alpha_zero_makes_every_branch_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = _model(_source(), batch=2)
            result = _scorer(
                model,
                Path(temporary) / "run",
                modes=("off", "crsa", "softmax", "shuffle"),
                alpha=0.0,
            ).run()
        reference = result["modes"]["off"]["items"]
        for mode in ("crsa", "softmax", "shuffle"):
            self.assertEqual(result["modes"][mode]["items"], reference)
        self.assertEqual(model.pager.metrics()["release_boundaries"], 1)
        self.assertEqual(model.pager.metrics()["head_rows"], 4 * 4)

    def test_resume_identity_binds_quantized_accumulation_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            first = _scorer(_model(_source(), batch=2), run_dir)
            execution = first.identity["execution"]
            self.assertTrue(execution["simulate_activation_quantization"])
            self.assertEqual(
                execution["quantized_accumulation_policy"],
                "mx-block-scaled-fp32/v1",
            )
            self.assertEqual(
                execution["attention_qat_policy"],
                "v4-native-fp8-kv+fp4-hadamard-indexer/v1",
            )
            first.run()

            incompatible_model = _model(_source(), batch=2)
            incompatible_model.pager.QUANTIZED_ACCUMULATION_POLICY = (
                "early-dequantized-diagnostic/v0"
            )
            with self.assertRaisesRegex(LayerwiseError, "identity differs"):
                _scorer(incompatible_model, run_dir).run(resume=True)

            incompatible_attention = _model(_source(), batch=2)
            incompatible_attention.ATTENTION_QAT_POLICY = "diagnostic-no-qat/v0"
            with self.assertRaisesRegex(LayerwiseError, "identity differs"):
                _scorer(incompatible_attention, run_dir).run(resume=True)

    def test_non_bfloat16_compute_fails_before_creating_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "not-created"
            source = _source()
            pager = DeepSeekWeightPager(
                source, device="cpu", compute_dtype="float32"
            )
            model = StreamedDeepSeekV4(
                _config(), pager, max_batch_size=2, max_seq_len=16
            )
            with self.assertRaisesRegex(LayerwiseError, "require bfloat16"):
                _scorer(model, run_dir)
            self.assertFalse(run_dir.exists())

    def test_resume_identity_binds_runtime_source_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            first = _scorer(_model(_source(), batch=2), run_dir)
            self.assertRegex(first.identity["runtime"]["source_sha256"], r"^[0-9a-f]{64}$")
            first.run()

            changed = [{"path": "fixture.py", "sha256": "f" * 64}]
            with mock.patch(
                "immer.runtimes.deepseek_v4.layerwise._runtime_source_manifest",
                return_value=changed,
            ):
                incompatible = _scorer(_model(_source(), batch=2), run_dir)
            with self.assertRaisesRegex(LayerwiseError, "identity differs"):
                incompatible.run(resume=True)

    def test_resume_continues_after_a_committed_layer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            first = _scorer(_model(_two_layer_source(), batch=2, layers=2), run_dir)
            original = first._run_layer

            def interrupt(layer, current, *, progress):
                if layer == 1:
                    raise RuntimeError("simulated interruption")
                return original(layer, current, progress=progress)

            first._run_layer = interrupt
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                first.run()
            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(manifest["body"]["state"]["completed_layer"], 0)

            resumed = _scorer(_model(_two_layer_source(), batch=2, layers=2), run_dir)
            result = resumed.run(resume=True)
            self.assertEqual(set(result["modes"]), {"off"})
            final_manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(final_manifest["body"]["state"]["phase"], "complete")

    def test_resume_rejects_stale_prior_layer_activation_swap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            first = _scorer(_model(_two_layer_source(), batch=2, layers=2), run_dir)
            published: list[dict] = []
            publish = first.store.publish_manifest

            def capture(manifest):
                published.append(json.loads(json.dumps(manifest)))
                publish(manifest)

            first.store.publish_manifest = capture
            first.store.cleanup_unreferenced = lambda _referenced: 0
            run_layer = first._run_layer

            def interrupt(layer, current, *, progress):
                if layer == 1:
                    raise RuntimeError("simulated interruption")
                return run_layer(layer, current, progress=progress)

            first._run_layer = interrupt
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                first.run()

            initial = published[0]["body"]["state"]["checkpoints"][0]
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            current = manifest["body"]["state"]["checkpoints"][0]
            for key in (
                "file",
                "sha256",
                "file_bytes",
                "tensor_bytes",
                "dtype",
                "shape",
            ):
                current[key] = initial[key]
            manifest["body_sha256"] = _body_sha(manifest["body"])
            _write_json(manifest_path, manifest)

            with self.assertRaisesRegex(LayerwiseError, "metadata mismatch"):
                _scorer(_model(_two_layer_source(), batch=2, layers=2), run_dir).run(
                    resume=True
                )

    def test_resume_rejects_completed_layer_tamper_after_rehash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            first = _scorer(_model(_two_layer_source(), batch=2, layers=2), run_dir)
            run_layer = first._run_layer

            def interrupt(layer, current, *, progress):
                if layer == 1:
                    raise RuntimeError("simulated interruption")
                return run_layer(layer, current, progress=progress)

            first._run_layer = interrupt
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                first.run()
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["body"]["state"]["completed_layer"] = 2
            manifest["body_sha256"] = _body_sha(manifest["body"])
            _write_json(manifest_path, manifest)

            with self.assertRaisesRegex(LayerwiseError, "outside its range"):
                _scorer(_model(_two_layer_source(), batch=2, layers=2), run_dir).run(
                    resume=True
                )

    def test_v1_resume_is_rejected_because_layer_provenance_is_unavailable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            _scorer(_model(_source(), batch=2), run_dir).run()
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["schema"] = "immer.deepseek-v4-layerwise/v1"
            manifest["version"] = 1
            manifest.pop("kind")
            _write_json(manifest_path, manifest)

            with self.assertRaisesRegex(LayerwiseError, "cannot prove"):
                _scorer(_model(_source(), batch=2), run_dir).run(resume=True)

    def test_completed_resume_rejects_tampered_result_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            _scorer(_model(_source(), batch=2), run_dir).run()
            result_path = run_dir / "result.json"
            document = json.loads(result_path.read_text())
            document["body"]["modes"]["off"]["items"][0]["candidate_scores"][0] += 1.0
            _write_json(result_path, document)

            with self.assertRaisesRegex(LayerwiseError, "result body SHA-256"):
                _scorer(_model(_source(), batch=2), run_dir).run(resume=True)

    def test_completed_resume_rejects_rehashed_incomplete_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            _scorer(_model(_source(), batch=2), run_dir).run()
            result_path = run_dir / "result.json"
            result = json.loads(result_path.read_text())
            result["body"]["modes"]["off"]["items"].pop()
            result["body_sha256"] = _body_sha(result["body"])
            _write_json(result_path, result)

            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["body"]["state"]["result_body_sha256"] = result["body_sha256"]
            manifest["body_sha256"] = _body_sha(manifest["body"])
            _write_json(manifest_path, manifest)

            with self.assertRaisesRegex(LayerwiseError, "item coverage"):
                _scorer(_model(_source(), batch=2), run_dir).run(resume=True)

    def test_equal_candidate_scores_choose_lower_token_id_not_input_order(self) -> None:
        items = (
            LayerwiseItem(
                "tie",
                (1, 2, 3),
                (14, 11, 13, 12),
                ("fourteen", "eleven", "thirteen", "twelve"),
                "eleven",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            model = _model(_source(), batch=2)

            def tied_logits(hidden, candidate_ids):
                return torch.zeros(
                    (hidden.shape[0], len(candidate_ids)),
                    dtype=torch.float32,
                    device=hidden.device,
                )

            model.pager.candidate_logits = tied_logits
            result = _scorer(
                model,
                Path(temporary) / "run",
                items=items,
            ).run()
        row = result["modes"]["off"]["items"][0]
        self.assertEqual(row["selected_candidate"], 1)
        self.assertEqual(row["predicted"], "eleven")

    def test_source_cache_priorities_clear_once_per_layer_not_bucket(self) -> None:
        source = _two_layer_source()
        clear = mock.Mock()
        source.clear_cache_priorities = clear
        with tempfile.TemporaryDirectory() as temporary:
            scorer = _scorer(
                _model(source, batch=2, layers=2),
                Path(temporary) / "run",
                microbatch_size=1,
            )
            self.assertEqual(len(scorer.plan.buckets), 2)
            scorer.run()
        self.assertEqual(clear.call_count, 2)

    def test_uncached_fixture_refetches_layer_weights_for_each_bucket(self) -> None:
        source = _source()
        fetches: Counter[str] = Counter()
        tensor = source.tensor

        def counted_tensor(name: str) -> np.ndarray:
            fetches[name] += 1
            return tensor(name)

        source.tensor = counted_tensor
        with tempfile.TemporaryDirectory() as temporary:
            scorer = _scorer(
                _model(source, batch=2),
                Path(temporary) / "run",
                microbatch_size=1,
            )
            self.assertEqual(scorer.plan.layer_forward_passes_max, 2)
            self.assertGreater(scorer.plan.source_cache_reuse_required_bytes, 0)
            self.assertFalse(scorer.plan.source_range_cache_enabled)
            self.assertFalse(scorer.plan.source_cache_reuse_admitted)
            scorer.run()
        self.assertEqual(fetches["layers.0.attn.wq_a.weight"], 2)

    def test_corrupt_referenced_checkpoint_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            _scorer(_model(_source(), batch=2), run_dir).run()
            manifest = json.loads((run_dir / "manifest.json").read_text())
            filename = manifest["body"]["state"]["checkpoints"][0]["file"]
            payload = run_dir / "objects" / filename
            with payload.open("r+b") as handle:
                handle.seek(-1, 2)
                value = handle.read(1)
                handle.seek(-1, 2)
                handle.write(bytes([value[0] ^ 1]))
            with self.assertRaisesRegex(LayerwiseError, "SHA-256"):
                _scorer(_model(_source(), batch=2), run_dir).run(resume=True)

    def test_torn_manifest_and_orphan_handling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            scorer = _scorer(_model(_source(), batch=2), run_dir)
            scorer.run()
            orphan = run_dir / "objects" / ("f" * 64 + ".safetensors")
            orphan.write_bytes(b"orphan")
            scorer = _scorer(_model(_source(), batch=2), run_dir)
            scorer.run(resume=True)
            self.assertFalse(orphan.exists())
            with (run_dir / "manifest.json").open("ab") as handle:
                handle.write(b"{")
            with self.assertRaisesRegex(LayerwiseError, "decode"):
                _scorer(_model(_source(), batch=2), run_dir).run(resume=True)

    def test_disk_admission_happens_before_run_directory_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "not-created"
            model = _model(_source(), batch=2)
            with mock.patch.object(
                shutil, "disk_usage", return_value=shutil._ntuple_diskusage(1, 1, 1)
            ):
                scorer = _scorer(model, run_dir)
            self.assertFalse(scorer.plan.admitted)
            self.assertFalse(run_dir.exists())
            with self.assertRaisesRegex(LayerwiseError, "disk preflight"):
                scorer.run()
            self.assertFalse(run_dir.exists())

    def test_official_full_source_admission_is_at_least_160_gib(self) -> None:
        source = _source()
        original_metrics = source.metrics

        def metrics():
            return {
                **original_metrics(),
                "repo_id": "deepseek-ai/DeepSeek-V4-Flash-0731",
            }

        source.metrics = metrics
        model = _model(source, batch=2)
        with tempfile.TemporaryDirectory() as temporary:
            plan = build_layerwise_plan(
                model,
                _items(),
                run_dir=Path(temporary) / "run",
                microbatch_size=2,
                require_full_source_disk=True,
                disk_margin_bytes=0,
            )
        self.assertGreaterEqual(plan.official_source_safe_bytes, 160 * 1024**3)
        self.assertEqual(
            plan.source_storage_admission_bytes, plan.official_source_safe_bytes
        )
        self.assertEqual(OFFICIAL_SOURCE_SAFE_BYTES, 160 * 1024**3)


if __name__ == "__main__":
    unittest.main()
