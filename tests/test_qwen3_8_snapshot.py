from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

import immer.runtimes.qwen3_8.model as qwen_model_module
from immer.knowledge import Streamer
from immer.runtimes.deepseek_v4.snapshot import (
    DeepSeekV4SnapshotError,
    read_snapshot,
)
from immer.runtimes.qwen3_8 import (
    QWEN38_SNAPSHOT_SCHEMA,
    Qwen38RuntimeError,
    Qwen38NativeHeadCrsa,
    Qwen38SnapshotError,
    Qwen38StableCrsaGraft,
    Qwen38WeightPager,
    StreamedQwen38,
)
from immer.runtimes.qwen3_8.provenance import runtime_source_manifest

from test_qwen3_8_model import _native_tiny_config, _tiny_config, _tiny_weights


class Qwen38SnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix=".qwen-snapshot-test-", dir=Path.cwd()
        )
        self.root = Path(self.temporary.name)
        self.config = _tiny_config()
        save_file(_tiny_weights(self.config), self.root / "model.safetensors")
        self.resources: list[tuple[Qwen38WeightPager, Streamer]] = []

    def tearDown(self) -> None:
        for pager, source in reversed(self.resources):
            pager.close()
            source.close()
        self.temporary.cleanup()

    def _model(
        self,
        *,
        dtype: str = "float32",
        graft: bool = False,
        max_seq_len: int = 32,
        max_resident_bytes: int = 2 * 1024**2,
        device: str = "cpu",
    ) -> StreamedQwen38:
        source = Streamer.from_local(self.root, budget_mb=20, use_cache=False)
        pager = Qwen38WeightPager(
            source,
            device=device,
            compute_dtype=dtype,
            max_resident_bytes=max_resident_bytes,
        )
        self.resources.append((pager, source))
        sidecar = Qwen38StableCrsaGraft(mode="crsa", alpha=0.1) if graft else None
        return StreamedQwen38(
            self.config,
            pager,
            graft=sidecar,
            graft_layer=1 if graft else None,
            max_batch_size=3,
            max_seq_len=max_seq_len,
        )

    def test_prefill_save_restore_decode_is_bit_exact(self) -> None:
        prompt = [[1, 4, 9]]
        next_token = [[7]]
        uninterrupted = self._model(dtype="bfloat16")
        uninterrupted.prefill(prompt)
        expected, _ = uninterrupted.decode(next_token)

        path = self.root / "prefix.json"
        saved = self._model(dtype="bfloat16")
        saved.prefill(prompt)
        receipt = saved.save_state(path)
        self.assertEqual(receipt["schema"], QWEN38_SNAPSHOT_SCHEMA)
        self.assertEqual(receipt["next_position"], 3)
        self.assertEqual(receipt["tensor_count"], 8)
        self.assertEqual(receipt["tensor_bytes"], saved.state_bytes)

        restored = self._model(dtype="bfloat16")
        loaded = restored.load_state(path)
        self.assertEqual(loaded["next_position"], 3)
        self.assertEqual(restored.state_batch_size, 1)
        self.assertEqual(restored.state_bytes, receipt["tensor_bytes"])
        actual, _ = restored.decode(next_token)
        self.assertTrue(torch.equal(actual, expected))

        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema"], QWEN38_SNAPSHOT_SCHEMA)
        self.assertEqual(manifest["body"]["state"]["state_batch_size"], 1)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_mps_indexed_state_round_trip_matches_indexless_pager_device(self) -> None:
        prompt = [[1, 4, 9]]
        next_token = [[7]]
        uninterrupted = self._model(dtype="bfloat16", device="mps")
        uninterrupted.prefill(prompt)
        expected, _ = uninterrupted.decode(next_token)

        path = self.root / "mps-prefix.json"
        saved = self._model(dtype="bfloat16", device="mps")
        saved.prefill(prompt)
        receipt = saved.save_state(path)
        self.assertEqual(receipt["next_position"], 3)
        self.assertTrue(
            all(state.conv.device.type == "mps" for state in saved._layer_states[:3])
        )

        restored = self._model(dtype="bfloat16", device="mps")
        restored.load_state(path)
        actual, _ = restored.decode(next_token)
        self.assertTrue(torch.equal(actual, expected))

    def test_crsa_history_round_trip_continues_bit_exactly(self) -> None:
        prompt = [[1, 4, 9]]
        next_token = [[7]]
        uninterrupted = self._model(graft=True)
        uninterrupted.prefill(prompt, tokenwise=True)
        expected, _ = uninterrupted.decode(next_token)

        path = self.root / "crsa-prefix.json"
        saved = self._model(graft=True)
        saved.prefill(prompt, tokenwise=True)
        receipt = saved.save_state(path)
        self.assertEqual(receipt["tensor_count"], 9)

        restored = self._model(graft=True)
        restored.load_state(path)
        self.assertEqual(tuple(restored._graft_history.shape), (1, 3, 12))
        actual, _ = restored.decode(next_token)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(tuple(restored._graft_history.shape), (1, 4, 12))

    def test_identity_mismatch_does_not_mutate_existing_state(self) -> None:
        path = self.root / "identity.json"
        source = self._model()
        source.prefill([[1, 4, 9]])
        source.save_state(path)

        target = self._model(dtype="bfloat16")
        target.prefill([[2, 3]])
        before_position = target.next_position
        before_bytes = target.state_bytes
        with self.assertRaisesRegex(Qwen38SnapshotError, "identity mismatch"):
            target.load_state(path)
        self.assertEqual(target.next_position, before_position)
        self.assertEqual(target.state_bytes, before_bytes)
        continued, _ = target.decode([[5]])
        self.assertEqual(tuple(continued.shape), (1, 1, self.config.dim))

        graft_target = self._model(graft=True)
        with self.assertRaisesRegex(Qwen38SnapshotError, "identity mismatch"):
            graft_target.load_state(path)

    def test_transport_neutral_snapshot_ignores_only_transport_controls(self) -> None:
        path = self.root / "neutral.json"
        source = self._model(max_resident_bytes=2 * 1024**2)
        source.prefill([[1, 4, 9]])
        receipt = source.save_state(path, transport_neutral=True)
        self.assertTrue(receipt["transport_neutral"])

        strict_target = self._model(max_resident_bytes=1024**2)
        with self.assertRaisesRegex(Qwen38SnapshotError, "identity mismatch"):
            strict_target.load_state(path)

        neutral_target = self._model(max_resident_bytes=1024**2)
        neutral_sources = runtime_source_manifest(include_transport=False)
        strict_sources = runtime_source_manifest(include_transport=True)

        def transport_changed(*, include_transport: bool):
            rows = [
                dict(row)
                for row in (strict_sources if include_transport else neutral_sources)
            ]
            if include_transport:
                streamer = next(
                    row for row in rows if row["path"] == "immer/knowledge/streamer.py"
                )
                streamer["sha256"] = "f" * 64
            return rows

        with mock.patch.object(
            qwen_model_module,
            "runtime_source_manifest",
            side_effect=transport_changed,
        ):
            loaded = neutral_target.load_state(path, transport_neutral=True)
        self.assertTrue(loaded["transport_neutral"])
        self.assertEqual(neutral_target.next_position, 3)

        different_math = self._model(dtype="bfloat16", max_resident_bytes=1024**2)
        with self.assertRaisesRegex(Qwen38SnapshotError, "identity mismatch"):
            different_math.load_state(path, transport_neutral=True)

    def test_restore_peak_and_corruption_fail_without_state_mutation(self) -> None:
        path = self.root / "bounded.json"
        source = self._model()
        source.prefill([[1, 4, 9]])
        receipt = source.save_state(path)

        target = self._model()
        target.prefill([[2, 3]])
        before_position = target.next_position
        before_bytes = target.state_bytes
        with self.assertRaisesRegex(Qwen38SnapshotError, "restore peak"):
            target.load_state(path, max_restore_peak_bytes=1)
        self.assertEqual(target.next_position, before_position)
        self.assertEqual(target.state_bytes, before_bytes)

        payload = Path(receipt["payload"])
        with payload.open("r+b") as handle:
            handle.seek(max(0, payload.stat().st_size // 2))
            original = handle.read(1)
            handle.seek(-1, 1)
            handle.write(bytes([original[0] ^ 0xFF]))
        with self.assertRaisesRegex(Qwen38SnapshotError, "SHA-256 mismatch"):
            target.load_state(path)
        self.assertEqual(target.next_position, before_position)
        self.assertEqual(target.state_bytes, before_bytes)

    def test_zero_and_poisoned_snapshots_are_tensor_free(self) -> None:
        zero_path = self.root / "zero.json"
        zero = self._model()
        zero_receipt = zero.save_state(zero_path)
        self.assertEqual(zero_receipt["tensor_count"], 0)
        target = self._model()
        target.prefill([[1, 2]])
        loaded = target.load_state(zero_path)
        self.assertEqual(loaded["next_position"], 0)
        self.assertEqual(target.state_bytes, 0)
        self.assertFalse(target.state_poisoned)

        poisoned = self._model()
        with mock.patch.object(
            poisoned, "_mlp", side_effect=RuntimeError("injected failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                poisoned.prefill([[1]])
        poison_path = self.root / "poisoned.json"
        poison_receipt = poisoned.save_state(poison_path)
        self.assertEqual(poison_receipt["tensor_count"], 0)
        restored = self._model()
        restored.load_state(poison_path)
        self.assertTrue(restored.state_poisoned)
        with self.assertRaisesRegex(Qwen38RuntimeError, "poisoned"):
            restored.prefill([[1]], reset=False)

    def test_qwen_schema_isolated_from_deepseek_default(self) -> None:
        path = self.root / "schema.json"
        model = self._model()
        model.prefill([[1]])
        model.save_state(path)
        identity = json.loads(path.read_text(encoding="utf-8"))["body"]["identity"]
        with self.assertRaisesRegex(DeepSeekV4SnapshotError, "unsupported.*schema"):
            read_snapshot(path, expected_identity=identity)

        relabelled = json.loads(path.read_text(encoding="utf-8"))
        relabelled["schema"] = "immer.deepseek-v4-continuation/v1"
        path.write_text(
            json.dumps(relabelled, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(DeepSeekV4SnapshotError, "unsupported.*schema"):
            read_snapshot(path, expected_identity=identity)

    def test_native_head_crsa_snapshots_refuse_before_io_and_bind_source(self) -> None:
        source = Streamer.from_local(self.root, budget_mb=20, use_cache=False)
        pager = Qwen38WeightPager(
            source,
            device="cpu",
            compute_dtype="float32",
            max_resident_bytes=2 * 1024**2,
        )
        self.resources.append((pager, source))
        model = StreamedQwen38(
            _native_tiny_config(),
            pager,
            native_head_crsa=Qwen38NativeHeadCrsa(),
            max_seq_len=16,
        )
        path = self.root / "native-refused.json"
        with self.assertRaisesRegex(
            Qwen38SnapshotError, "native Head-CRSA.*unsupported"
        ):
            model.save_state(path)
        self.assertFalse(path.exists())
        with self.assertRaisesRegex(
            Qwen38SnapshotError, "native Head-CRSA.*unsupported"
        ):
            model.load_state(path)

        runtime_paths = {
            row["path"] for row in runtime_source_manifest(include_transport=False)
        }
        self.assertIn("immer/runtimes/qwen3_8/native_crsa.py", runtime_paths)

        native_root = self.root / "native-alpha-zero"
        native_root.mkdir()
        config = _native_tiny_config()
        save_file(_tiny_weights(config), native_root / "model.safetensors")
        identity_source = Streamer.from_local(
            native_root, budget_mb=20, use_cache=False
        )
        identity_pager = Qwen38WeightPager(
            identity_source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=2 * 1024**2,
        )
        self.resources.append((identity_pager, identity_source))
        alpha_zero = StreamedQwen38(
            config,
            identity_pager,
            native_head_crsa=Qwen38NativeHeadCrsa(alpha=0.0),
            max_seq_len=16,
        )
        off = StreamedQwen38(config, identity_pager, max_seq_len=16)
        prompt = [[1, 4, 9]]
        alpha_zero.prefill(prompt)
        identity_path = self.root / "native-alpha-zero.json"
        receipt = alpha_zero.save_state(identity_path)
        self.assertEqual(receipt["tensor_count"], 56)
        off.load_state(identity_path)
        expected, _ = alpha_zero.decode([[7]])
        actual, _ = off.decode([[7]])
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
