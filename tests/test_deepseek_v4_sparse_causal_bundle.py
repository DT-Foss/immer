from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from immer.knowledge import AccessTraceRecorder, Streamer
from immer.runtimes.deepseek_v4 import CausalWeightMount, DeepSeekWeightPager

from test_deepseek_v4_causal_weights import (
    _LOGICAL_MODEL,
    _shift_plan,
    _write_expert_fixture,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deepseek_v4_sparse_causal_bundle.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "deepseek_v4_sparse_causal_bundle", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import sparse causal bundle script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bundle_script = _load_script()


class SparseCausalBundleTests(unittest.TestCase):
    def _sparse_append_fixture(
        self, base: Path
    ) -> tuple[Path, Path, str, argparse.Namespace]:
        source_root = base / "source"
        cache = base / "cache"
        output = base / "model.causal"
        _write_expert_fixture(source_root, expert_ids=(0, 1))
        source_root.joinpath("config.json").write_text("{}", encoding="utf-8")
        recorder = AccessTraceRecorder()
        source = Streamer.from_local(
            source_root,
            repo_id=_LOGICAL_MODEL.repo_id,
            revision=_LOGICAL_MODEL.revision,
            cache_dir=cache,
            budget_mb=16,
            access_observer=recorder,
        )
        try:
            inventory = source.inventory()
            source.reader.fetch_file("config.json")
            for tensor in inventory["tensors"]:
                if ".experts.0." not in tensor["name"]:
                    continue
                begin, end = tensor["offset_in_shard"]
                source.raw_bytes(
                    tensor["shard"],
                    int(tensor["data_start"]) + int(begin),
                    int(end) - int(begin),
                )
            trace = recorder.snapshot()
        finally:
            source.close()
        trace_path = base / "trace.json"
        trace_path.write_bytes(trace.to_bytes())
        inventory_path = next((cache / "inventories").glob("*.json"))
        inventory_cache = json.loads(inventory_path.read_text(encoding="utf-8"))
        fingerprint = inventory_cache["source_fingerprint"]
        bundle_script.build_bundle(
            argparse.Namespace(
                access_trace=[str(trace_path)],
                budget_mb=16.0,
                cache_dir=str(cache),
                inventory=str(inventory_path),
                layout_fingerprint=fingerprint,
                output=str(output),
                repo_id=_LOGICAL_MODEL.repo_id,
                revision=_LOGICAL_MODEL.revision,
            )
        )
        append_args = argparse.Namespace(
            budget_mb=16.0,
            bundle=str(output),
            expert=[(3, 1)],
            inject_crash=None,
            layout_fingerprint=fingerprint,
            remote_cache_dir=None,
            remote_cache_limit_mb=16.0,
            repo_id=_LOGICAL_MODEL.repo_id,
            resident_limit_mb=1.0,
            revision=_LOGICAL_MODEL.revision,
            staging_limit_mb=2.0,
        )
        return source_root, output, fingerprint, append_args

    @staticmethod
    def _remote(source_root: Path) -> Streamer:
        return Streamer.from_local(
            source_root,
            repo_id=_LOGICAL_MODEL.repo_id,
            revision=_LOGICAL_MODEL.revision,
            use_cache=False,
            budget_mb=16,
        )

    @staticmethod
    def _verify_args(output: Path, fingerprint: str) -> argparse.Namespace:
        return argparse.Namespace(
            budget_mb=16.0,
            bundle=str(output),
            layout_fingerprint=fingerprint,
            repo_id=_LOGICAL_MODEL.repo_id,
            revision=_LOGICAL_MODEL.revision,
        )

    def test_header_reconstruction_preserves_original_data_start(self) -> None:
        shard = {"data_start": 128, "file": "model.safetensors", "st_metadata": None}
        tensors = [
            {
                "dtype": "I8",
                "name": "b",
                "offset_in_shard": [2, 4],
                "shape": [2],
            },
            {
                "dtype": "I8",
                "name": "a",
                "offset_in_shard": [0, 2],
                "shape": [2],
            },
        ]
        encoded = bundle_script._header_bytes(shard, tensors)
        self.assertEqual(len(encoded), 128)
        self.assertEqual(int.from_bytes(encoded[:8], "little"), 120)
        header = json.loads(encoded[8:].decode("utf-8"))
        self.assertEqual(list(header), ["a", "b"])

    def test_builds_and_mounts_trace_complete_sparse_fixture(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            base = Path(temporary)
            source_root = base / "source"
            cache = base / "cache"
            output = base / "model.causal"
            _write_expert_fixture(source_root, expert_ids=(0, 1))
            source_root.joinpath("config.json").write_text("{}", encoding="utf-8")
            recorder = AccessTraceRecorder()
            source = Streamer.from_local(
                source_root,
                repo_id=_LOGICAL_MODEL.repo_id,
                revision=_LOGICAL_MODEL.revision,
                cache_dir=cache,
                budget_mb=16,
                access_observer=recorder,
            )
            try:
                inventory = source.inventory()
                source.reader.fetch_file("config.json")
                for tensor in inventory["tensors"]:
                    begin, end = tensor["offset_in_shard"]
                    source.raw_bytes(
                        tensor["shard"],
                        int(tensor["data_start"]) + int(begin),
                        int(end) - int(begin),
                    )
                trace = recorder.snapshot()
            finally:
                source.close()
            trace_path = base / "trace.json"
            trace_path.write_bytes(trace.to_bytes())
            inventory_path = next((cache / "inventories").glob("*.json"))
            inventory_cache = json.loads(inventory_path.read_text(encoding="utf-8"))
            fingerprint = inventory_cache["source_fingerprint"]
            args = argparse.Namespace(
                access_trace=[str(trace_path)],
                budget_mb=16.0,
                cache_dir=str(cache),
                inventory=str(inventory_path),
                layout_fingerprint=fingerprint,
                output=str(output),
                repo_id=_LOGICAL_MODEL.repo_id,
                revision=_LOGICAL_MODEL.revision,
            )

            manifest = bundle_script.build_bundle(args)
            self.assertEqual(manifest["causal_bindings"], 2)
            self.assertEqual(manifest["layout_fingerprint"], fingerprint)
            self.assertEqual(manifest["unique_leaves"], 12)
            self.assertGreater(manifest["physical_shard_bytes"], 0)
            self.assertGreater(manifest["logical_shard_bytes"], 0)
            stored = json.loads(output.joinpath("bundle.json").read_text())
            identity = {key: value for key, value in stored.items() if key != "sha256"}
            self.assertEqual(
                stored["sha256"],
                hashlib.sha256(bundle_script._canonical(identity)).hexdigest(),
            )

            verified = bundle_script.verify_bundle(
                argparse.Namespace(
                    budget_mb=16.0,
                    bundle=str(output),
                    repo_id=_LOGICAL_MODEL.repo_id,
                    revision=_LOGICAL_MODEL.revision,
                )
            )
            self.assertEqual(verified["causal_bindings"], 2)

    def test_append_expert_is_durable_verified_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            source_root, output, fingerprint, args = self._sparse_append_fixture(
                Path(temporary)
            )

            def opener(_args: argparse.Namespace) -> Streamer:
                return self._remote(source_root)

            shard = output / "weights" / "model.safetensors"
            shard_before = shard.read_bytes()
            with mock.patch.object(
                bundle_script,
                "_open_remote_source",
                side_effect=opener,
            ):
                first = bundle_script.append_experts(args)
            self.assertEqual(first["status"], "appended")
            self.assertEqual(first["appended_bindings"], 1)
            self.assertFalse(
                output.joinpath("expert-appends", "pending.json").exists()
            )
            receipt_document = json.loads(
                next(
                    output.joinpath("expert-appends", "receipts").glob("*.json")
                ).read_text(encoding="utf-8")
            )
            target_offsets = {
                offset
                for leaf in receipt_document["body"]["transaction"]["leaves"]
                for offset in range(
                    leaf["absolute_offset"],
                    leaf["absolute_offset"] + leaf["length"],
                )
            }
            shard_after = shard.read_bytes()
            self.assertEqual(len(shard_before), len(shard_after))
            self.assertTrue(
                all(
                    before == after
                    for offset, (before, after) in enumerate(
                        zip(shard_before, shard_after, strict=True)
                    )
                    if offset not in target_offsets
                )
            )
            with CausalWeightMount(output, _LOGICAL_MODEL, budget_mb=16) as mount:
                plan = mount.resolve_expert_plans(3, (1,))[0]
                before_revision = mount.graph.store.revision()
                read = mount.read_experts(3, (1,))
            self.assertEqual(len(plan.ranges), 2)
            self.assertEqual(len(read.parts), 2)

            verified = bundle_script.verify_bundle(
                self._verify_args(output, fingerprint)
            )
            self.assertEqual(verified["manifest_causal_bindings"], 1)
            self.assertEqual(verified["appended_bindings"], 1)
            self.assertEqual(verified["causal_bindings"], 2)
            self.assertEqual(len(verified["append_receipts"]), 1)
            self.assertIsNone(verified["pending_append"])

            physical_before = shard.stat().st_blocks
            with mock.patch.object(
                bundle_script,
                "_open_remote_source",
                side_effect=AssertionError("idempotent replay used the network"),
            ):
                replay = bundle_script.append_experts(args)
            self.assertEqual(replay["status"], "already-appended")
            self.assertEqual(shard.stat().st_blocks, physical_before)
            with CausalWeightMount(output, _LOGICAL_MODEL, budget_mb=16) as mount:
                self.assertEqual(mount.graph.store.revision(), before_revision)

    def test_resume_after_payload_before_binding_crash(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            source_root, output, fingerprint, args = self._sparse_append_fixture(
                Path(temporary)
            )
            args.inject_crash = "payload-before-binding"

            def opener(_args: argparse.Namespace) -> Streamer:
                return self._remote(source_root)

            with (
                mock.patch.object(
                    bundle_script,
                    "_open_remote_source",
                    side_effect=opener,
                ),
                self.assertRaisesRegex(
                    bundle_script.InjectedAppendCrash,
                    "payload-before-binding",
                ),
            ):
                bundle_script.append_experts(args)
            pending = bundle_script.verify_bundle(
                self._verify_args(output, fingerprint)
            )["pending_append"]
            self.assertEqual(pending["state"], "payload_durable")
            with CausalWeightMount(output, _LOGICAL_MODEL, budget_mb=16) as mount:
                with self.assertRaises(KeyError):
                    mount.resolve_expert_plans(3, (1,))

            args.inject_crash = None
            with mock.patch.object(
                bundle_script,
                "_open_remote_source",
                side_effect=AssertionError("payload reconciliation used the network"),
            ):
                resumed = bundle_script.append_experts(args)
            self.assertEqual(resumed["status"], "appended")
            self.assertEqual(
                bundle_script.verify_bundle(
                    self._verify_args(output, fingerprint)
                )["appended_bindings"],
                1,
            )

    def test_resume_after_binding_before_journal_crash(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            source_root, output, fingerprint, args = self._sparse_append_fixture(
                Path(temporary)
            )
            args.inject_crash = "binding-before-journal"

            def opener(_args: argparse.Namespace) -> Streamer:
                return self._remote(source_root)

            with (
                mock.patch.object(
                    bundle_script,
                    "_open_remote_source",
                    side_effect=opener,
                ),
                self.assertRaisesRegex(
                    bundle_script.InjectedAppendCrash,
                    "binding-before-journal",
                ),
            ):
                bundle_script.append_experts(args)
            with CausalWeightMount(output, _LOGICAL_MODEL, budget_mb=16) as mount:
                self.assertEqual(mount.resolve_expert_plans(3, (1,))[0].expert_id, 1)
                bound_revision = mount.graph.store.revision()
            self.assertEqual(
                bundle_script.verify_bundle(
                    self._verify_args(output, fingerprint)
                )["pending_append"]["state"],
                "payload_durable",
            )

            args.inject_crash = None
            with mock.patch.object(
                bundle_script,
                "_open_remote_source",
                side_effect=AssertionError("binding reconciliation used the network"),
            ):
                resumed = bundle_script.append_experts(args)
            self.assertEqual(resumed["status"], "appended")
            self.assertEqual(resumed["appended_bindings"], 0)
            with CausalWeightMount(output, _LOGICAL_MODEL, budget_mb=16) as mount:
                self.assertEqual(mount.graph.store.revision(), bound_revision)

    def test_receipt_before_cleanup_crash_reconciles_offline_and_reclaims_stage(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            source_root, output, _fingerprint, args = self._sparse_append_fixture(
                Path(temporary)
            )
            args.inject_crash = "receipt-before-cleanup"
            with (
                mock.patch.object(
                    bundle_script,
                    "_open_remote_source",
                    side_effect=lambda _args: self._remote(source_root),
                ),
                self.assertRaisesRegex(
                    bundle_script.InjectedAppendCrash,
                    "receipt-before-cleanup",
                ),
            ):
                bundle_script.append_experts(args)
            append_root = output / "expert-appends"
            self.assertTrue((append_root / "pending.json").is_file())
            stage = next(append_root.glob("stage-*"))
            self.assertTrue(stage.is_dir())
            self.assertTrue(next((append_root / "receipts").glob("*.json")).is_file())

            args.inject_crash = None
            with mock.patch.object(
                bundle_script,
                "_open_remote_source",
                side_effect=AssertionError("receipt reconciliation used the network"),
            ):
                replay = bundle_script.append_experts(args)
            self.assertEqual(replay["status"], "already-appended")
            self.assertFalse((append_root / "pending.json").exists())
            self.assertFalse(stage.exists())

    def test_partial_remote_payload_never_mutates_weights_or_graph(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            source_root, output, _fingerprint, args = self._sparse_append_fixture(
                Path(temporary)
            )
            remote = self._remote(source_root)
            original = remote.raw_bytes
            calls = 0

            def short(shard: str, offset: int, length: int) -> bytes:
                nonlocal calls
                calls += 1
                encoded = original(shard, offset, length)
                return encoded[:-1] if calls == 2 else encoded

            remote.raw_bytes = short  # type: ignore[method-assign]
            with (
                mock.patch.object(
                    bundle_script,
                    "_open_remote_source",
                    return_value=remote,
                ),
                self.assertRaisesRegex(
                    bundle_script.SparseBundleError,
                    "partial expert leaf",
                ),
            ):
                bundle_script.append_experts(args)
            with CausalWeightMount(output, _LOGICAL_MODEL, budget_mb=16) as mount:
                with self.assertRaises(KeyError):
                    mount.resolve_expert_plans(3, (1,))

    def test_resident_and_staging_bounds_fail_before_weight_mutation(self) -> None:
        for field, message in (
            ("resident_limit_mb", "resident memory bound"),
            ("staging_limit_mb", "staging disk bound"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory(
                dir=Path.cwd()
            ) as temporary:
                source_root, output, _fingerprint, args = self._sparse_append_fixture(
                    Path(temporary)
                )
                setattr(args, field, 1e-7)
                shard = output / "weights" / "model.safetensors"
                before = shard.read_bytes()
                with (
                    mock.patch.object(
                        bundle_script,
                        "_open_remote_source",
                        side_effect=lambda _args: self._remote(source_root),
                    ),
                    self.assertRaisesRegex(bundle_script.SparseBundleError, message),
                ):
                    bundle_script.append_experts(args)
                self.assertEqual(shard.read_bytes(), before)
                with CausalWeightMount(output, _LOGICAL_MODEL, budget_mb=16) as mount:
                    with self.assertRaises(KeyError):
                        mount.resolve_expert_plans(3, (1,))

    def test_occupied_nonmatching_bytes_fail_before_any_binding(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            source_root, output, _fingerprint, args = self._sparse_append_fixture(
                Path(temporary)
            )
            remote = self._remote(source_root)
            try:
                plan = DeepSeekWeightPager(
                    remote,
                    device="cpu",
                    compute_dtype="bfloat16",
                    expert_prefetch=False,
                ).plan_expert_ranges(3, (1,))[0]
            finally:
                remote.close()
            first = plan.ranges[0].tensors[0]
            shard = output / "weights" / plan.ranges[0].shard
            descriptor = os.open(shard, os.O_RDWR)
            try:
                os.pwrite(descriptor, b"\xfe", first.absolute_offset)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            with (
                mock.patch.object(
                    bundle_script,
                    "_open_remote_source",
                    side_effect=lambda _args: self._remote(source_root),
                ),
                self.assertRaisesRegex(
                    bundle_script.SparseBundleError,
                    "occupied nonmatching bytes",
                ),
            ):
                bundle_script.append_experts(args)
            with CausalWeightMount(output, _LOGICAL_MODEL, budget_mb=16) as mount:
                with self.assertRaises(KeyError):
                    mount.resolve_expert_plans(3, (1,))

    def test_graph_conflict_fails_before_payload_mutation(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            source_root, output, _fingerprint, args = self._sparse_append_fixture(
                Path(temporary)
            )
            remote = self._remote(source_root)
            try:
                plan = DeepSeekWeightPager(
                    remote,
                    device="cpu",
                    compute_dtype="bfloat16",
                    expert_prefetch=False,
                ).plan_expert_ranges(3, (1,))[0]
            finally:
                remote.close()
            with CausalWeightMount(output, _LOGICAL_MODEL, budget_mb=16) as mount:
                mount.bind_plans((_shift_plan(plan, 1),))
            with (
                mock.patch.object(
                    bundle_script,
                    "_open_remote_source",
                    side_effect=lambda _args: self._remote(source_root),
                ),
                self.assertRaisesRegex(
                    bundle_script.SparseBundleError,
                    "causal graph conflicts",
                ),
            ):
                bundle_script.append_experts(args)
            self.assertFalse(
                output.joinpath("expert-appends", "pending.json").exists()
            )

    def test_verify_rejects_appended_payload_and_receipt_tamper(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            source_root, output, fingerprint, args = self._sparse_append_fixture(
                Path(temporary)
            )
            with mock.patch.object(
                bundle_script,
                "_open_remote_source",
                side_effect=lambda _args: self._remote(source_root),
            ):
                result = bundle_script.append_experts(args)
            receipt_path = output / "expert-appends" / "receipts" / (
                result["transaction_id"] + ".json"
            )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            leaf = receipt["body"]["transaction"]["leaves"][0]
            shard = output / "weights" / leaf["shard"]
            descriptor = os.open(shard, os.O_RDWR)
            try:
                original = os.pread(descriptor, 1, leaf["absolute_offset"])
                os.pwrite(
                    descriptor,
                    bytes([original[0] ^ 0xFF]),
                    leaf["absolute_offset"],
                )
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            with self.assertRaisesRegex(
                bundle_script.SparseBundleError,
                "data hash",
            ):
                bundle_script.verify_bundle(self._verify_args(output, fingerprint))

            descriptor = os.open(shard, os.O_RDWR)
            try:
                os.pwrite(descriptor, original, leaf["absolute_offset"])
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            receipt["body"]["journal"]["history"][0]["sha256"] = "0" * 64
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(
                bundle_script.SparseBundleError,
                "receipt identity|SHA chain",
            ):
                bundle_script.verify_bundle(self._verify_args(output, fingerprint))

    def test_wrong_identity_and_symlink_shard_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            source_root, output, _fingerprint, args = self._sparse_append_fixture(
                Path(temporary)
            )
            args.revision = "b" * 40
            with (
                mock.patch.object(bundle_script, "_open_remote_source") as opener,
                self.assertRaisesRegex(
                    bundle_script.SparseBundleError,
                    "logical model identity",
                ),
            ):
                bundle_script.append_experts(args)
            opener.assert_not_called()
            args.revision = _LOGICAL_MODEL.revision

            shard = output / "weights" / "model.safetensors"
            moved = output / "weights" / "model.real"
            shard.rename(moved)
            shard.symlink_to(moved.name)
            with (
                mock.patch.object(
                    bundle_script,
                    "_open_remote_source",
                    side_effect=lambda _args: self._remote(source_root),
                ),
                self.assertRaises(Exception),
            ):
                bundle_script.append_experts(args)

    def test_append_cli_uses_explicit_coordinates_and_bounds(self) -> None:
        args = bundle_script._parser().parse_args(
            [
                "append-experts",
                "--bundle",
                "/tmp/model.causal",
                "--expert",
                "12:131",
                "--expert",
                "31:4",
            ]
        )
        self.assertEqual(args.expert, [(12, 131), (31, 4)])
        self.assertEqual(args.resident_limit_mb, 64.0)
        self.assertEqual(args.staging_limit_mb, 256.0)


if __name__ == "__main__":
    unittest.main()
