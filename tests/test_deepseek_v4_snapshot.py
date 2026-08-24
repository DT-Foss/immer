from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import immer.runtimes.deepseek_v4.provenance as runtime_provenance
from immer.runtimes.deepseek_v4 import (
    DeepSeekV4SnapshotError,
    DeepSeekWeightPager,
    StreamedDeepSeekV4,
)
from immer.runtimes.deepseek_v4.graft import DeepSeekV4CrsaGraft
from immer.runtimes.deepseek_v4.snapshot import (
    MAX_NPY_HEADER_BYTES,
    SnapshotTensor,
    _read_npy_header,
    read_snapshot,
    write_snapshot,
)
from immer.runtimes.deepseek_v4.route_markov import (
    LayerMarkovExpertPredictor,
    LayerTokenRoutes,
    PromptRouteObservation,
)

from test_deepseek_v4_model import (
    _CompressedTinyCheckpoint,
    _QuantizedTinyCheckpoint,
    _TwoLayerPrefetchQuantizedTinyCheckpoint,
    _config,
)


def _canonical(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


class _DifferentSource(_CompressedTinyCheckpoint):
    def metrics(self) -> dict:
        result = super().metrics()
        result["inventory_source_fingerprint"] = "different-tiny"
        return result


class DeepSeekV4SnapshotTests(unittest.TestCase):
    def _temporary_directory(self):
        # Keep all continuation artifacts inside the shared workspace.
        return tempfile.TemporaryDirectory(prefix=".v4-snapshot-test-", dir=Path.cwd())

    @staticmethod
    def _compressed_model(
        ratio: int,
        *,
        graft: bool = False,
        source=None,
        activation_quantization: bool = True,
    ) -> StreamedDeepSeekV4:
        checkpoint = source or _CompressedTinyCheckpoint(
            random_weights=True, compress_ratio=ratio
        )
        sidecar = (
            DeepSeekV4CrsaGraft(mode="crsa", alpha=0.05, max_history=64)
            if graft
            else None
        )
        return StreamedDeepSeekV4(
            _config(ratio),
            DeepSeekWeightPager(
                checkpoint,
                device="cpu",
                compute_dtype="float32",
                simulate_activation_quantization=activation_quantization,
            ),
            graft=sidecar,
            graft_layer=0 if graft else None,
            max_seq_len=256 if ratio == 128 else 64,
        )

    def test_ratio4_ring_overlap_indexer_and_graft_continue_bit_exactly(self) -> None:
        prompt = [[(index * 7 + 3) % 127 for index in range(18)]]
        next_token = [[67]]
        uninterrupted = self._compressed_model(4, graft=True)
        uninterrupted.prefill(prompt, tokenwise=False)
        expected, _ = uninterrupted.decode(next_token)

        with self._temporary_directory() as directory:
            path = Path(directory) / "ratio4.json"
            saved = self._compressed_model(4, graft=True)
            saved.prefill(prompt, tokenwise=False)
            result = saved.save_state(path)
            self.assertEqual(result["next_position"], 18)
            self.assertGreater(result["tensor_count"], 4)

            restored = self._compressed_model(4, graft=True)
            loaded = restored.load_state(path)
            self.assertEqual(loaded["next_position"], 18)
            self.assertEqual(restored._graft_history.shape[1], 18)
            state = restored._attention_states[0]
            assert state is not None and state.compressor is not None
            self.assertEqual(state.local.length, 16)
            self.assertEqual(state.compressor.cache.length, 4)
            self.assertEqual(state.indexer.cache.length, 4)

            actual, _ = restored.decode(next_token)
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_transport_neutral_prefix_restores_across_exact_prefetch_arms(self) -> None:
        config = replace(
            _config(0, n_layers=2),
            n_activated_experts=2,
            n_hash_layers=2,
        )
        predictor = LayerMarkovExpertPredictor(n_experts=2)
        predictor.observe(
            PromptRouteObservation(
                "shared-prefix-training",
                (
                    LayerTokenRoutes(0, ((0, 1), (0, 1))),
                    LayerTokenRoutes(1, ((0, 1), (0, 1))),
                ),
            )
        )

        def model(
            *,
            route_predictor=None,
            expert_prefetch: bool = True,
            activation_quantization: bool = True,
        ) -> StreamedDeepSeekV4:
            return StreamedDeepSeekV4(
                config,
                DeepSeekWeightPager(
                    _TwoLayerPrefetchQuantizedTinyCheckpoint(),
                    device="cpu",
                    compute_dtype="float32",
                    expert_prefetch=expert_prefetch,
                    simulate_activation_quantization=activation_quantization,
                ),
                max_seq_len=8,
                route_predictor=route_predictor,
                route_prefetch_k=1,
                route_prefetch_min_confidence=0.0,
            )

        with self._temporary_directory() as directory:
            path = Path(directory) / "shared-prefix.json"
            source = model()
            source.prefill([[7, 8, 9, 10]], tokenwise=False)
            saved = source.save_state(path, transport_neutral=True)
            self.assertTrue(saved["transport_neutral"])

            baseline = model(expert_prefetch=False)
            full_identity_target = model(route_predictor=predictor)
            with self.assertRaisesRegex(DeepSeekV4SnapshotError, "identity mismatch"):
                full_identity_target.load_state(path)

            baseline_loaded = baseline.load_state(path, transport_neutral=True)
            real = model(route_predictor=predictor)
            real_loaded = real.load_state(path, transport_neutral=True)
            self.assertTrue(baseline_loaded["transport_neutral"])
            self.assertTrue(real_loaded["transport_neutral"])

            different_math = model(activation_quantization=False)
            with self.assertRaisesRegex(DeepSeekV4SnapshotError, "identity mismatch"):
                different_math.load_state(path, transport_neutral=True)

            expected, expected_evidence = baseline.decode([[11]])
            actual, actual_evidence = real.decode([[11]])
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
            self.assertEqual(
                actual_evidence.selected_experts,
                expected_evidence.selected_experts,
            )
            self.assertEqual(real.route_prefetch_metrics()["direct_bindings"], 1)
            self.assertEqual(real.pager.metrics()["expert_reservoir_submitted"], 1)
            self.assertEqual(real.pager.metrics()["expert_reservoir_usable_hits"], 1)

    def test_bfloat16_raw_storage_round_trips_without_dtype_narrowing(self) -> None:
        values = torch.tensor([[1.0, -2.5, 0.125, 65_536.0]], dtype=torch.bfloat16)
        identity = {"fixture": "bf16"}
        with self._temporary_directory() as directory:
            path = Path(directory) / "bf16.json"
            write_snapshot(
                path,
                identity=identity,
                state={"fixture": True, "next_position": 1},
                tensors={"activation": SnapshotTensor(values)},
            )
            loaded = read_snapshot(path, expected_identity=identity)
        self.assertEqual(loaded.tensors["activation"].dtype, torch.bfloat16)
        self.assertTrue(torch.equal(loaded.tensors["activation"], values))

    def test_legacy_deepseek_manifest_without_body_schema_remains_readable(
        self,
    ) -> None:
        identity = {"fixture": "legacy-deepseek-schema"}
        with self._temporary_directory() as directory:
            path = Path(directory) / "legacy.json"
            write_snapshot(
                path,
                identity=identity,
                state={"next_position": 1},
                tensors={"state": SnapshotTensor(torch.ones(1))},
            )
            document = json.loads(path.read_text(encoding="utf-8"))
            document["body"].pop("schema")
            document["body_sha256"] = hashlib.sha256(
                _canonical(document["body"])
            ).hexdigest()
            path.write_text(
                json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            loaded = read_snapshot(path, expected_identity=identity)
        self.assertTrue(torch.equal(loaded.tensors["state"], torch.ones(1)))

    def test_ratio128_partial_compressor_state_continues_bit_exactly(self) -> None:
        prompt = [[2, 5, 8, 12, 19]]
        uninterrupted = self._compressed_model(128)
        uninterrupted.prefill(prompt, tokenwise=False)
        expected, _ = uninterrupted.decode([[23]])

        with self._temporary_directory() as directory:
            path = Path(directory) / "ratio128.json"
            saved = self._compressed_model(128)
            saved.prefill(prompt, tokenwise=False)
            saved.save_state(path)
            restored = self._compressed_model(128)
            restored.load_state(path)
            state = restored._attention_states[0]
            assert state is not None and state.compressor is not None
            self.assertEqual(state.compressor.next_position, len(prompt[0]))
            self.assertEqual(state.compressor.cache.length, 0)
            self.assertIsNotNone(state.compressor._kv_state)
            actual, _ = restored.decode([[23]])
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_uncompressed_ring_wrap_continues_bit_exactly(self) -> None:
        prompt = [[(index * 5 + 1) % 127 for index in range(19)]]

        def model() -> StreamedDeepSeekV4:
            return StreamedDeepSeekV4(
                _config(0),
                DeepSeekWeightPager(
                    _QuantizedTinyCheckpoint(),
                    device="cpu",
                    compute_dtype="float32",
                ),
                max_seq_len=64,
            )

        uninterrupted = model()
        uninterrupted.prefill(prompt, tokenwise=False)
        expected, _ = uninterrupted.decode([[31]])
        with self._temporary_directory() as directory:
            path = Path(directory) / "ring.json"
            saved = model()
            saved.prefill(prompt, tokenwise=False)
            saved.save_state(path)
            restored = model()
            restored.load_state(path)
            state = restored._attention_states[0]
            assert state is not None
            self.assertEqual(state.local.length, 16)
            actual, _ = restored.decode([[31]])
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_payload_corruption_fails_without_mutating_existing_state(self) -> None:
        with self._temporary_directory() as directory:
            path = Path(directory) / "corrupt.json"
            source = self._compressed_model(4)
            source.prefill([[2, 3, 5, 7]], tokenwise=False)
            result = source.save_state(path)
            payload = Path(result["payload"])
            damaged = bytearray(payload.read_bytes())
            damaged[len(damaged) // 2] ^= 0x5A
            payload.write_bytes(damaged)

            target = self._compressed_model(4)
            target.prefill([[11, 13]], tokenwise=False)
            old_state = target._attention_states[0]
            old_ring = old_state.local.physical().clone()
            with self.assertRaisesRegex(DeepSeekV4SnapshotError, "SHA-256"):
                target.load_state(path)
            self.assertEqual(target.next_position, 2)
            self.assertIs(target._attention_states[0], old_state)
            torch.testing.assert_close(old_state.local.physical(), old_ring)

    def test_identity_mismatches_fail_closed(self) -> None:
        with self._temporary_directory() as directory:
            path = Path(directory) / "identity.json"
            source = self._compressed_model(4, graft=True)
            source.save_state(path)
            manifest = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["body"]["identity"]["runtime"]["schema"],
                "immer.streamed-deepseek-v4/native-stateful-v3",
            )
            runtime = manifest["body"]["identity"]["runtime"]
            self.assertRegex(
                runtime["source_sha256"],
                r"^[0-9a-f]{64}$",
            )
            self.assertRegex(runtime["dependency_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                runtime["source_sha256"],
                hashlib.sha256(_canonical(runtime["sources"])).hexdigest(),
            )
            self.assertEqual(
                set(runtime["dependencies"]),
                {"python", "torch", "numpy", "requests", "safetensors"},
            )
            self.assertEqual(
                runtime["dependency_sha256"],
                hashlib.sha256(_canonical(runtime["dependencies"])).hexdigest(),
            )
            paths = {row["path"] for row in runtime["sources"]}
            self.assertTrue(
                {
                    "immer/runtimes/deepseek_v4/__init__.py",
                    "immer/runtimes/deepseek_v4/snapshot.py",
                    "immer/knowledge/__init__.py",
                    "immer/knowledge/streamer.py",
                    "immer/knowledge/_hf_source.py",
                }.issubset(paths)
            )
            self.assertEqual(
                manifest["body"]["identity"]["execution"][
                    "quantized_accumulation_policy"
                ],
                "mx-block-scaled-fp32/v1",
            )
            self.assertEqual(
                manifest["body"]["identity"]["execution"]["attention_qat_policy"],
                "v4-native-fp8-kv+fp4-hadamard-indexer/v1",
            )
            self.assertEqual(
                manifest["body"]["identity"]["execution"]["expert_prefetch_policy"],
                "exact-router-window-q3-a2/v2",
            )
            execution = manifest["body"]["identity"]["execution"]
            self.assertEqual(
                execution["expert_prefetch_transport_policy"],
                "streamer-exact-range/v1",
            )
            self.assertEqual(execution["expert_prefetch_workers"], 2)
            self.assertEqual(execution["expert_prefetch_active_read_limit"], 2)
            self.assertEqual(execution["expert_prefetch_max_outstanding"], 3)
            self.assertEqual(execution["expert_prefetch_max_experts"], 3)
            self.assertEqual(execution["expert_range_coalesce_max_experts"], 1)
            self.assertEqual(execution["expert_range_coalesce_max_gap_bytes"], 0)
            self.assertEqual(
                execution["expert_prefetch_resident_limit_bytes"],
                48 * 1024**2,
            )
            self.assertEqual(execution["source_transport_policy"], "unreported")
            self.assertEqual(execution["source_transport_connection_limit"], 0)

            incompatible_prefetch = self._compressed_model(4, graft=True)
            incompatible_prefetch.pager.expert_prefetch_enabled = False
            with self.assertRaisesRegex(DeepSeekV4SnapshotError, "identity mismatch"):
                incompatible_prefetch.load_state(path)

            with self.assertRaisesRegex(DeepSeekV4SnapshotError, "identity mismatch"):
                self._compressed_model(4, graft=False).load_state(path)
            different_source = _DifferentSource(random_weights=True, compress_ratio=4)
            with self.assertRaisesRegex(DeepSeekV4SnapshotError, "identity mismatch"):
                self._compressed_model(
                    4, graft=True, source=different_source
                ).load_state(path)
            with self.assertRaisesRegex(DeepSeekV4SnapshotError, "identity mismatch"):
                self._compressed_model(
                    4, graft=True, activation_quantization=False
                ).load_state(path)
            incompatible_attention = self._compressed_model(4, graft=True)
            incompatible_attention.ATTENTION_QAT_POLICY = "diagnostic-no-qat/v0"
            with self.assertRaisesRegex(DeepSeekV4SnapshotError, "identity mismatch"):
                incompatible_attention.load_state(path)

    def test_snapshot_resume_binds_source_hashes_and_dependency_versions(self) -> None:
        with self._temporary_directory() as directory:
            path = Path(directory) / "proof-envelope.json"
            source = self._compressed_model(4)
            source.save_state(path)
            manifest = json.loads(path.read_text(encoding="utf-8"))
            runtime = manifest["body"]["identity"]["runtime"]

            original_sha256 = runtime_provenance._sha256_file
            for suffix in (
                "runtimes/deepseek_v4/__init__.py",
                "runtimes/deepseek_v4/snapshot.py",
                "knowledge/__init__.py",
                "knowledge/streamer.py",
                "knowledge/_hf_source.py",
            ):
                with self.subTest(mutated_source=suffix):

                    def changed_hash(path, *, target=suffix):
                        digest = original_sha256(path)
                        return "f" * 64 if path.as_posix().endswith(target) else digest

                    with patch.object(
                        runtime_provenance,
                        "_sha256_file",
                        side_effect=changed_hash,
                    ):
                        with self.assertRaisesRegex(
                            DeepSeekV4SnapshotError, "identity mismatch"
                        ):
                            self._compressed_model(4).load_state(path)

            changed_dependencies = dict(runtime["dependencies"])
            changed_dependencies["safetensors"] += ".mutated"
            with patch(
                "immer.runtimes.deepseek_v4.model.runtime_dependency_versions",
                return_value=changed_dependencies,
            ):
                with self.assertRaisesRegex(
                    DeepSeekV4SnapshotError, "identity mismatch"
                ):
                    self._compressed_model(4).load_state(path)

    def test_oversize_and_shape_bombs_are_rejected_before_restore(self) -> None:
        with self._temporary_directory() as directory:
            path = Path(directory) / "limits.json"
            source = self._compressed_model(4)
            source.prefill([[2, 3, 5, 7]], tokenwise=False)
            source.save_state(path)
            with self.assertRaisesRegex(
                DeepSeekV4SnapshotError, "size|byte limit|excessive"
            ):
                self._compressed_model(4).load_state(path, max_bytes=32)

            document = json.loads(path.read_text(encoding="utf-8"))
            document["body"]["tensors"][0]["shape"] = [2_000_000_000]
            document["body_sha256"] = hashlib.sha256(
                _canonical(document["body"])
            ).hexdigest()
            path.write_bytes(_canonical(document) + b"\n")
            with self.assertRaisesRegex(DeepSeekV4SnapshotError, "dimension"):
                self._compressed_model(4).load_state(path)

    def test_cursor_tampering_is_transactional_even_with_valid_manifest_hash(
        self,
    ) -> None:
        with self._temporary_directory() as directory:
            path = Path(directory) / "cursor.json"
            source = self._compressed_model(4)
            source.prefill([[2, 3, 5, 7]], tokenwise=False)
            source.save_state(path)
            document = json.loads(path.read_text(encoding="utf-8"))
            document["body"]["state"]["attention_layers"][0]["state"][
                "next_position"
            ] = 3
            document["body_sha256"] = hashlib.sha256(
                _canonical(document["body"])
            ).hexdigest()
            path.write_bytes(_canonical(document) + b"\n")

            target = self._compressed_model(4)
            target.prefill([[29, 31]], tokenwise=False)
            old_state = target._attention_states[0]
            with self.assertRaisesRegex(DeepSeekV4SnapshotError, "inconsistent"):
                target.load_state(path)
            self.assertEqual(target.next_position, 2)
            self.assertIs(target._attention_states[0], old_state)

    def test_authenticated_nonfinite_cache_tensor_is_rejected(self) -> None:
        with self._temporary_directory() as directory:
            path = Path(directory) / "nonfinite.json"
            source = self._compressed_model(4)
            source.prefill([[2, 3, 5, 7]], tokenwise=False)
            result = source.save_state(path)
            document = json.loads(path.read_text(encoding="utf-8"))
            with np.load(result["payload"], allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
            descriptor = next(
                row
                for row in document["body"]["tensors"]
                if row["name"].endswith(".local.storage")
            )
            raw = arrays[descriptor["storage"]]
            raw[:4] = np.frombuffer(np.float32(np.nan).tobytes(), dtype=np.uint8)
            descriptor["sha256"] = hashlib.sha256(raw.tobytes()).hexdigest()
            payload = Path(directory) / "authenticated-malicious.npz"
            with payload.open("wb") as stream:
                np.savez(stream, **arrays)
            payload_bytes = payload.read_bytes()
            document["body"]["payload"] = {
                "file": payload.name,
                "bytes": len(payload_bytes),
                "sha256": hashlib.sha256(payload_bytes).hexdigest(),
            }
            document["body_sha256"] = hashlib.sha256(
                _canonical(document["body"])
            ).hexdigest()
            path.write_bytes(_canonical(document) + b"\n")

            target = self._compressed_model(4)
            target.prefill([[29, 31]], tokenwise=False)
            old_state = target._attention_states[0]
            with self.assertRaisesRegex(DeepSeekV4SnapshotError, "finite policy"):
                target.load_state(path)
            self.assertEqual(target.next_position, 2)
            self.assertIs(target._attention_states[0], old_state)

    def test_poison_latch_round_trips(self) -> None:
        with self._temporary_directory() as directory:
            path = Path(directory) / "poison.json"
            source = StreamedDeepSeekV4(
                _config(0),
                DeepSeekWeightPager(
                    _QuantizedTinyCheckpoint(),
                    device="cpu",
                    compute_dtype="float32",
                ),
                max_seq_len=16,
            )
            source._state_poisoned = True
            source.save_state(path)
            restored = StreamedDeepSeekV4(
                _config(0),
                DeepSeekWeightPager(
                    _QuantizedTinyCheckpoint(),
                    device="cpu",
                    compute_dtype="float32",
                ),
                max_seq_len=16,
            )
            restored.load_state(path)
            with self.assertRaisesRegex(RuntimeError, "poisoned"):
                restored.hidden_stateful([[3]])

    def test_reset_retained_buffers_save_as_canonical_empty_state(self) -> None:
        with self._temporary_directory() as directory:
            path = Path(directory) / "reset.json"
            source = self._compressed_model(4)
            source.prefill([[2, 3, 5, 7]], tokenwise=False)
            source.reset_state(release=False)
            self.assertGreater(source.attention_state_bytes, 0)
            saved = source.save_state(path)
            self.assertEqual(saved["tensor_count"], 0)
            document = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(document["body"]["state"]["attention_layers"], [])
            self.assertIsNone(document["body"]["state"]["graft_history"])

            restored = self._compressed_model(4)
            restored.load_state(path)
            self.assertEqual(restored.next_position, 0)
            self.assertEqual(restored.attention_state_bytes, 0)
            self.assertTrue(all(state is None for state in restored._attention_states))

    def test_restore_peak_gate_runs_before_npz_tensor_allocation(self) -> None:
        with self._temporary_directory() as directory:
            path = Path(directory) / "peak.json"
            source = self._compressed_model(4)
            source.prefill([[2, 3, 5, 7]], tokenwise=False)
            source.save_state(path)
            document = json.loads(path.read_text(encoding="utf-8"))
            tensor_bytes = int(document["body"]["tensor_bytes"])
            largest = max(int(row["nbytes"]) for row in document["body"]["tensors"])

            target = self._compressed_model(4)
            target.prefill([[11, 13]], tokenwise=False)
            resident = target.attention_state_bytes
            expected_peak = resident + tensor_bytes + largest
            old_state = target._attention_states[0]
            with patch(
                "immer.runtimes.deepseek_v4.snapshot.np.load",
                side_effect=AssertionError("np.load must not run"),
            ) as loader:
                with self.assertRaisesRegex(DeepSeekV4SnapshotError, "restore peak"):
                    target.load_state(path, max_restore_peak_bytes=expected_peak - 1)
                loader.assert_not_called()
            self.assertEqual(target.next_position, 2)
            self.assertIs(target._attention_states[0], old_state)

            loaded = target.load_state(path, max_restore_peak_bytes=expected_peak)
            self.assertEqual(loaded["estimated_restore_peak_bytes"], expected_peak)
            self.assertEqual(loaded["largest_tensor_bytes"], largest)

    def test_manifest_and_payload_symlinks_are_rejected(self) -> None:
        with self._temporary_directory() as directory:
            path = Path(directory) / "state.json"
            source = self._compressed_model(4)
            source.prefill([[2, 3, 5, 7]], tokenwise=False)
            saved = source.save_state(path)
            manifest_link = Path(directory) / "manifest-link.json"
            manifest_link.symlink_to(path.name)
            with self.assertRaisesRegex(
                DeepSeekV4SnapshotError, "manifest|regular file"
            ):
                source.save_state(manifest_link)
            with self.assertRaisesRegex(
                DeepSeekV4SnapshotError, "manifest|regular file"
            ):
                self._compressed_model(4).load_state(manifest_link)

            payload = Path(saved["payload"])
            backing = Path(directory) / "payload-backing.npz"
            payload.rename(backing)
            payload.symlink_to(backing.name)
            with patch(
                "immer.runtimes.deepseek_v4.snapshot._sha256_stream",
                return_value=saved["payload_sha256"],
            ):
                with self.assertRaisesRegex(
                    DeepSeekV4SnapshotError, "payload|regular file"
                ):
                    source.save_state(path)
            with self.assertRaisesRegex(
                DeepSeekV4SnapshotError, "payload|regular file"
            ):
                self._compressed_model(4).load_state(path)

    def test_authenticated_wrong_role_dtype_is_transactionally_rejected(self) -> None:
        with self._temporary_directory() as directory:
            path = Path(directory) / "wrong-dtype.json"

            def model() -> StreamedDeepSeekV4:
                return StreamedDeepSeekV4(
                    _config(0),
                    DeepSeekWeightPager(
                        _QuantizedTinyCheckpoint(),
                        device="cpu",
                        compute_dtype="float32",
                    ),
                    max_seq_len=64,
                )

            source = model()
            source.prefill([[2, 3, 5, 7]], tokenwise=False)
            source.save_state(path)
            document = json.loads(path.read_text(encoding="utf-8"))
            descriptor = next(
                row
                for row in document["body"]["tensors"]
                if row["name"].endswith(".local.storage")
            )
            self.assertEqual(descriptor["dtype"], "float32")
            descriptor["dtype"] = "float16"
            descriptor["shape"][-1] *= 2
            document["body_sha256"] = hashlib.sha256(
                _canonical(document["body"])
            ).hexdigest()
            path.write_bytes(_canonical(document) + b"\n")

            target = model()
            target.prefill([[29, 31]], tokenwise=False)
            old_state = target._attention_states[0]
            with self.assertRaisesRegex(DeepSeekV4SnapshotError, "dtype"):
                target.load_state(path)
            self.assertEqual(target.next_position, 2)
            self.assertIs(target._attention_states[0], old_state)

    def test_npy_header_length_is_capped_before_header_read(self) -> None:
        class GuardedHeader(io.BytesIO):
            def read(self, size=-1):
                if self.tell() >= 10:
                    raise AssertionError("oversize header body was read")
                return super().read(size)

        stream = GuardedHeader(
            b"\x93NUMPY" + bytes((1, 0)) + struct.pack("<H", MAX_NPY_HEADER_BYTES + 1)
        )
        with self.assertRaisesRegex(DeepSeekV4SnapshotError, "header length"):
            _read_npy_header(stream)

    def test_model_rejects_context_beyond_checkpoint_horizon(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_position_embeddings"):
            StreamedDeepSeekV4(
                _config(0),
                DeepSeekWeightPager(
                    _QuantizedTinyCheckpoint(),
                    device="cpu",
                    compute_dtype="float32",
                ),
                max_seq_len=1025,
            )


if __name__ == "__main__":
    unittest.main()
