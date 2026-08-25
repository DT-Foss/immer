from __future__ import annotations

import argparse
from dataclasses import replace
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

    @staticmethod
    def _trace_args(
        append_args: argparse.Namespace,
        trace_path: Path,
        *,
        plan_only: bool,
    ) -> argparse.Namespace:
        values = vars(append_args).copy()
        values.update(
            access_trace=[str(trace_path)],
            plan_only=plan_only,
        )
        return argparse.Namespace(**values)

    @staticmethod
    def _expert_trace(
        source_root: Path,
        output: Path,
        fingerprint: str,
        expert_ids: tuple[int, ...],
        *,
        omit_last_part: bool = False,
    ):
        pinned = json.loads(
            output.joinpath("weights", "inventory.pinned.json").read_text(
                encoding="utf-8"
            )
        )["inventory"]
        recorder = AccessTraceRecorder()
        source = Streamer.from_local(
            source_root,
            repo_id=_LOGICAL_MODEL.repo_id,
            revision=_LOGICAL_MODEL.revision,
            pinned_inventory=pinned,
            pinned_fingerprint=fingerprint,
            use_cache=False,
            budget_mb=16,
            access_observer=recorder,
        )
        try:
            inventory = source.inventory()
            selected = [
                tensor
                for tensor in inventory["tensors"]
                if any(
                    f".experts.{expert_id}." in tensor["name"]
                    for expert_id in expert_ids
                )
            ]
            if omit_last_part:
                selected = selected[:-1]
            for tensor in selected:
                begin, end = tensor["offset_in_shard"]
                source.raw_bytes(
                    tensor["shard"],
                    int(tensor["data_start"]) + int(begin),
                    int(end) - int(begin),
                )
            return recorder.snapshot()
        finally:
            source.close()

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

    def test_total_filesystem_preflight_precedes_remote_open_and_mutation(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            _source_root, output, _fingerprint, args = self._sparse_append_fixture(
                Path(temporary)
            )
            shard = output / "weights" / "model.safetensors"
            before = shard.read_bytes()
            payload_bytes = 21
            available = 8 * 1024 * 1024 + payload_bytes + payload_bytes // 2
            filesystem = mock.Mock(f_bavail=available, f_frsize=1)
            with (
                mock.patch.object(
                    bundle_script.os,
                    "fstatvfs",
                    return_value=filesystem,
                ),
                mock.patch.object(
                    bundle_script,
                    "_open_remote_source",
                    side_effect=AssertionError("disk preflight opened the remote"),
                ) as opener,
                mock.patch.object(bundle_script, "_atomic_json_at") as mutation,
                self.assertRaisesRegex(
                    bundle_script.SparseBundleError,
                    "insufficient free disk for the complete expert append",
                ),
            ):
                bundle_script.append_experts(args)
            opener.assert_not_called()
            mutation.assert_not_called()
            self.assertEqual(shard.read_bytes(), before)
            self.assertFalse(
                output.joinpath("expert-appends", "pending.json").exists()
            )

    def test_total_filesystem_preflight_accounts_for_block_rounding(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            _source_root, output, _fingerprint, args = self._sparse_append_fixture(
                Path(temporary)
            )
            block_size = 4096
            logical_floor = 8 * 1024 * 1024 + 2 * 21
            available = ((logical_floor + block_size - 1) // block_size) * block_size
            filesystem = mock.Mock(
                f_bavail=available // block_size,
                f_frsize=block_size,
            )
            with (
                mock.patch.object(
                    bundle_script.os,
                    "fstatvfs",
                    return_value=filesystem,
                ),
                mock.patch.object(
                    bundle_script,
                    "_open_remote_source",
                    side_effect=AssertionError("disk preflight opened the remote"),
                ) as opener,
                mock.patch.object(bundle_script, "_atomic_json_at") as mutation,
                self.assertRaisesRegex(
                    bundle_script.SparseBundleError,
                    "insufficient free disk for the complete expert append",
                ),
            ):
                bundle_script.append_experts(args)
            opener.assert_not_called()
            mutation.assert_not_called()

    def test_stage_rechecks_full_floor_before_first_remote_read(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            source_root, output, _fingerprint, args = self._sparse_append_fixture(
                Path(temporary)
            )
            shard = output / "weights" / "model.safetensors"
            before = shard.read_bytes()
            block_size = 4096
            full_floor = 8 * 1024 * 1024 + 7 * block_size
            stage_available = 8 * 1024 * 1024 + block_size
            self.assertGreater(stage_available, 8 * 1024 * 1024 + 21)
            filesystems = (
                mock.Mock(
                    f_bavail=full_floor // block_size,
                    f_frsize=block_size,
                ),
                mock.Mock(
                    f_bavail=full_floor // block_size,
                    f_frsize=block_size,
                ),
                mock.Mock(
                    f_bavail=stage_available // block_size,
                    f_frsize=block_size,
                ),
                mock.Mock(
                    f_bavail=stage_available // block_size,
                    f_frsize=block_size,
                ),
            )
            remote = self._remote(source_root)
            original = remote.raw_bytes
            raw_reads = 0

            def counted_raw_bytes(shard: str, offset: int, length: int) -> bytes:
                nonlocal raw_reads
                raw_reads += 1
                return original(shard, offset, length)

            remote.raw_bytes = counted_raw_bytes  # type: ignore[method-assign]
            with (
                mock.patch.object(
                    bundle_script.os,
                    "fstatvfs",
                    side_effect=filesystems,
                ) as statvfs,
                mock.patch.object(
                    bundle_script,
                    "_open_remote_source",
                    return_value=remote,
                ) as opener,
                self.assertRaisesRegex(
                    bundle_script.SparseBundleError,
                    "insufficient free disk for the complete expert append",
                ),
            ):
                bundle_script.append_experts(args)
            self.assertEqual(statvfs.call_count, 4)
            self.assertEqual(opener.call_count, 1)
            self.assertEqual(raw_reads, 0)
            self.assertEqual(shard.read_bytes(), before)
            with CausalWeightMount(output, _LOGICAL_MODEL, budget_mb=16) as mount:
                with self.assertRaises(KeyError):
                    mount.resolve_expert_plans(3, (1,))

    def test_cross_device_append_uses_independent_filesystem_floors(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            source_root, output, _fingerprint, args = self._sparse_append_fixture(
                Path(temporary)
            )
            staging_block_size = 4096
            target_block_size = 8192
            staging_floor = 8 * 1024 * 1024 + 6 * staging_block_size
            target_floor = 8 * 1024 * 1024 + target_block_size
            filesystems = (
                mock.Mock(
                    f_bavail=staging_floor // staging_block_size,
                    f_frsize=staging_block_size,
                ),
                mock.Mock(
                    f_bavail=target_floor // target_block_size,
                    f_frsize=target_block_size,
                ),
                mock.Mock(
                    f_bavail=staging_floor // staging_block_size,
                    f_frsize=staging_block_size,
                ),
                mock.Mock(
                    f_bavail=target_floor // target_block_size,
                    f_frsize=target_block_size,
                ),
            )
            with (
                mock.patch.object(
                    bundle_script,
                    "_directory_device",
                    side_effect=(1, 2, 1, 2),
                ),
                mock.patch.object(
                    bundle_script.os,
                    "fstatvfs",
                    side_effect=filesystems,
                ) as statvfs,
                mock.patch.object(
                    bundle_script,
                    "_open_remote_source",
                    side_effect=lambda _args: self._remote(source_root),
                ) as opener,
            ):
                result = bundle_script.append_experts(args)
            self.assertEqual(result["status"], "appended")
            self.assertEqual(opener.call_count, 1)
            self.assertEqual(statvfs.call_count, 4)
            with CausalWeightMount(output, _LOGICAL_MODEL, budget_mb=16) as mount:
                self.assertEqual(mount.resolve_expert_plans(3, (1,))[0].expert_id, 1)
            with (
                mock.patch.object(
                    bundle_script,
                    "_directory_device",
                    side_effect=AssertionError(
                        "idempotent replay imposed a filesystem restriction"
                    ),
                ),
                mock.patch.object(
                    bundle_script,
                    "_open_remote_source",
                    side_effect=AssertionError("idempotent replay opened the remote"),
                ),
            ):
                replay = bundle_script.append_experts(args)
            self.assertEqual(replay["status"], "already-appended")

    def test_cross_device_preflight_checks_each_filesystem_independently(
        self,
    ) -> None:
        staging_block_size = 4096
        target_block_size = 8192
        staging_floor = 8 * 1024 * 1024 + 6 * staging_block_size
        target_floor = 8 * 1024 * 1024 + target_block_size
        for constrained, message in (
            ("staging", "staging filesystem"),
            ("target", "target filesystem"),
        ):
            with self.subTest(constrained=constrained), tempfile.TemporaryDirectory(
                dir=Path.cwd()
            ) as temporary:
                _source_root, output, _fingerprint, args = self._sparse_append_fixture(
                    Path(temporary)
                )
                shard = output / "weights" / "model.safetensors"
                before = shard.read_bytes()
                stage_available = staging_floor - (
                    staging_block_size if constrained == "staging" else 0
                )
                target_available = target_floor - (
                    target_block_size if constrained == "target" else 0
                )
                with (
                    mock.patch.object(
                        bundle_script,
                        "_directory_device",
                        side_effect=(1, 2),
                    ),
                    mock.patch.object(
                        bundle_script.os,
                        "fstatvfs",
                        side_effect=(
                            mock.Mock(
                                f_bavail=stage_available // staging_block_size,
                                f_frsize=staging_block_size,
                            ),
                            mock.Mock(
                                f_bavail=target_available // target_block_size,
                                f_frsize=target_block_size,
                            ),
                        ),
                    ),
                    mock.patch.object(
                        bundle_script,
                        "_open_remote_source",
                        side_effect=AssertionError(
                            "split-filesystem preflight opened the remote"
                        ),
                    ) as opener,
                    mock.patch.object(bundle_script, "_atomic_json_at") as mutation,
                    self.assertRaisesRegex(
                        bundle_script.SparseBundleError,
                        message,
                    ),
                ):
                    bundle_script.append_experts(args)
                opener.assert_not_called()
                mutation.assert_not_called()
                self.assertEqual(shard.read_bytes(), before)

    def test_planned_pending_uses_stage_floor_before_first_remote_read(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            source_root, output, fingerprint, args = self._sparse_append_fixture(
                Path(temporary)
            )
            shard = output / "weights" / "model.safetensors"
            before = shard.read_bytes()
            block_size = 4096
            full_floor = 8 * 1024 * 1024 + 7 * block_size
            args.resident_limit_mb = 1e-7
            with (
                mock.patch.object(
                    bundle_script.os,
                    "fstatvfs",
                    return_value=mock.Mock(
                        f_bavail=full_floor // block_size,
                        f_frsize=block_size,
                    ),
                ),
                mock.patch.object(
                    bundle_script,
                    "_open_remote_source",
                    side_effect=lambda _args: self._remote(source_root),
                ),
                self.assertRaisesRegex(
                    bundle_script.SparseBundleError,
                    "resident memory bound",
                ),
            ):
                bundle_script.append_experts(args)
            self.assertEqual(
                bundle_script.verify_bundle(
                    self._verify_args(output, fingerprint)
                )["pending_append"]["state"],
                "planned",
            )

            args.resident_limit_mb = 1.0
            stage_available = 8 * 1024 * 1024 + block_size
            remote = self._remote(source_root)
            original = remote.raw_bytes
            raw_reads = 0

            def counted_raw_bytes(shard: str, offset: int, length: int) -> bytes:
                nonlocal raw_reads
                raw_reads += 1
                return original(shard, offset, length)

            remote.raw_bytes = counted_raw_bytes  # type: ignore[method-assign]
            with (
                mock.patch.object(
                    bundle_script.os,
                    "fstatvfs",
                    return_value=mock.Mock(
                        f_bavail=stage_available // block_size,
                        f_frsize=block_size,
                    ),
                ) as statvfs,
                mock.patch.object(
                    bundle_script,
                    "_open_remote_source",
                    return_value=remote,
                ),
                self.assertRaisesRegex(
                    bundle_script.SparseBundleError,
                    "insufficient free disk for the complete expert append",
                ),
            ):
                bundle_script.append_experts(args)
            self.assertEqual(statvfs.call_count, 2)
            self.assertEqual(raw_reads, 0)
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

    def test_trace_expert_plan_derives_filters_and_seals_missing_coordinates(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            base = Path(temporary)
            source_root, output, fingerprint, append_args = self._sparse_append_fixture(
                base
            )
            trace = self._expert_trace(
                source_root,
                output,
                fingerprint,
                (0, 1),
            )
            trace_path = base / "sealed-trace.json"
            trace_path.write_bytes(trace.to_bytes())
            args = self._trace_args(append_args, trace_path, plan_only=True)

            with (
                mock.patch.object(
                    bundle_script.os,
                    "fstatvfs",
                    return_value=mock.Mock(f_frsize=4096),
                ),
                mock.patch.object(
                    bundle_script,
                    "_open_remote_source",
                    side_effect=AssertionError("plan-only opened the remote source"),
                ),
            ):
                result = bundle_script.append_trace_experts(args)
            self.assertEqual(result["status"], "planned")
            self.assertIsNone(result["append"])
            plan = result["plan"]
            self.assertEqual(plan["schema"], bundle_script.TRACE_EXPERT_PLAN_SCHEMA)
            self.assertEqual(plan["sha256"], bundle_script._sha256(plan["body"]))
            self.assertEqual(
                plan["body"]["counts"],
                {
                    "derived": 2,
                    "derived_payload_bytes": 42,
                    "existing": 1,
                    "existing_payload_bytes": 21,
                    "missing": 1,
                    "missing_leaves": 6,
                    "missing_payload_bytes": 21,
                    "traces": 1,
                    "unique_leaves": 12,
                },
            )
            self.assertEqual(
                plan["body"]["resource_requirements"],
                {
                    "filesystems_shared": True,
                    "minimum_available_bundle_filesystem_bytes": 8 * 1024 * 1024
                    + 7 * 4096,
                    "minimum_available_staging_disk_bytes": 8 * 1024 * 1024 + 21,
                    "minimum_available_staging_filesystem_bytes": 8
                    * 1024
                    * 1024
                    + 6 * 4096,
                    "minimum_available_target_filesystem_bytes": 8
                    * 1024
                    * 1024
                    + 4096,
                    "minimum_leaf_transfer_budget_bytes": 21 + 6 * 16 * 1024,
                    "minimum_resident_limit_bytes": 5,
                    "minimum_staging_limit_bytes": 21,
                    "staging_filesystem_block_size_bytes": 4096,
                    "source_budget_requires_inventory_scan_headroom": True,
                    "source_range_reservation_overhead_bytes_per_leaf": 16 * 1024,
                    "target_filesystem_block_size_bytes": 4096,
                },
            )
            self.assertEqual(
                plan["body"]["coordinates"],
                {
                    "derived": [
                        {"expert_id": 0, "layer": 3},
                        {"expert_id": 1, "layer": 3},
                    ],
                    "existing": [{"expert_id": 0, "layer": 3}],
                    "missing": [{"expert_id": 1, "layer": 3}],
                },
            )
            self.assertEqual(
                result["sha256"],
                bundle_script._sha256(
                    {key: value for key, value in result.items() if key != "sha256"}
                ),
            )

    def test_trace_expert_plan_reports_split_filesystem_floors(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            base = Path(temporary)
            source_root, output, fingerprint, append_args = self._sparse_append_fixture(
                base
            )
            output.joinpath("expert-appends").mkdir()
            trace = self._expert_trace(
                source_root,
                output,
                fingerprint,
                (0, 1),
            )
            trace_path = base / "sealed-trace.json"
            trace_path.write_bytes(trace.to_bytes())
            args = self._trace_args(append_args, trace_path, plan_only=True)
            with (
                mock.patch.object(
                    bundle_script,
                    "_directory_device",
                    side_effect=(1, 2),
                ),
                mock.patch.object(
                    bundle_script.os,
                    "fstatvfs",
                    side_effect=(
                        mock.Mock(f_frsize=4096),
                        mock.Mock(f_frsize=8192),
                    ),
                ),
                mock.patch.object(
                    bundle_script,
                    "_open_remote_source",
                    side_effect=AssertionError("cross-device plan opened the remote"),
                ) as opener,
            ):
                result = bundle_script.append_trace_experts(args)
            opener.assert_not_called()
            self.assertEqual(result["status"], "planned")
            self.assertEqual(
                result["plan"]["body"]["resource_requirements"],
                {
                    "filesystems_shared": False,
                    "minimum_available_bundle_filesystem_bytes": None,
                    "minimum_available_staging_disk_bytes": 8 * 1024 * 1024 + 21,
                    "minimum_available_staging_filesystem_bytes": 8
                    * 1024
                    * 1024
                    + 6 * 4096,
                    "minimum_available_target_filesystem_bytes": 8
                    * 1024
                    * 1024
                    + 8192,
                    "minimum_leaf_transfer_budget_bytes": 21 + 6 * 16 * 1024,
                    "minimum_resident_limit_bytes": 5,
                    "minimum_staging_limit_bytes": 21,
                    "staging_filesystem_block_size_bytes": 4096,
                    "source_budget_requires_inventory_scan_headroom": True,
                    "source_range_reservation_overhead_bytes_per_leaf": 16 * 1024,
                    "target_filesystem_block_size_bytes": 8192,
                },
            )

    def test_trace_expert_ingest_appends_once_then_is_offline_noop(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            base = Path(temporary)
            source_root, output, fingerprint, append_args = self._sparse_append_fixture(
                base
            )
            trace = self._expert_trace(
                source_root,
                output,
                fingerprint,
                (0, 1),
            )
            trace_path = base / "sealed-trace.json"
            trace_path.write_bytes(trace.to_bytes())
            planned = bundle_script.append_trace_experts(
                self._trace_args(append_args, trace_path, plan_only=True)
            )
            args = self._trace_args(append_args, trace_path, plan_only=False)

            with mock.patch.object(
                bundle_script,
                "_open_remote_source",
                side_effect=lambda _args: self._remote(source_root),
            ) as opener:
                first = bundle_script.append_trace_experts(args)
            self.assertEqual(first["status"], "appended")
            self.assertEqual(first["append"]["appended_bindings"], 1)
            self.assertEqual(
                first["append"]["requested_plan_sha256"],
                planned["plan"]["sha256"],
            )
            self.assertEqual(first["plan"]["body"]["counts"]["missing"], 0)
            self.assertEqual(opener.call_count, 1)
            receipt = json.loads(
                output.joinpath(
                    "expert-appends",
                    "receipts",
                    f"{first['append']['transaction_id']}.json",
                ).read_text(encoding="utf-8")
            )
            appended_payload_bytes = sum(
                leaf["length"] for leaf in receipt["body"]["transaction"]["leaves"]
            )
            self.assertEqual(
                planned["plan"]["body"]["counts"]["missing_payload_bytes"],
                appended_payload_bytes,
            )

            shard = output / "weights" / "model.safetensors"
            physical_before = shard.stat().st_blocks
            with mock.patch.object(
                bundle_script,
                "_open_remote_source",
                side_effect=AssertionError("idempotent no-op used the network"),
            ):
                replay = bundle_script.append_trace_experts(args)
            self.assertEqual(replay["status"], "already-complete")
            self.assertIsNone(replay["append"])
            self.assertEqual(replay["plan"]["body"]["counts"]["missing"], 0)
            self.assertEqual(
                replay["plan"]["body"]["counts"]["missing_payload_bytes"], 0
            )
            self.assertEqual(
                replay["plan"]["body"]["resource_requirements"],
                {
                    "filesystems_shared": True,
                    "minimum_available_bundle_filesystem_bytes": 0,
                    "minimum_available_staging_disk_bytes": 0,
                    "minimum_available_staging_filesystem_bytes": 0,
                    "minimum_available_target_filesystem_bytes": 0,
                    "minimum_leaf_transfer_budget_bytes": 0,
                    "minimum_resident_limit_bytes": 0,
                    "minimum_staging_limit_bytes": 0,
                    "staging_filesystem_block_size_bytes": os.statvfs(
                        output / "expert-appends"
                    ).f_frsize,
                    "source_budget_requires_inventory_scan_headroom": False,
                    "source_range_reservation_overhead_bytes_per_leaf": 16 * 1024,
                    "target_filesystem_block_size_bytes": os.statvfs(
                        output / "weights"
                    ).f_frsize,
                },
            )
            self.assertEqual(shard.stat().st_blocks, physical_before)
            verified = bundle_script.verify_bundle(
                self._verify_args(output, fingerprint)
            )
            self.assertEqual(verified["causal_bindings"], 2)
            self.assertEqual(verified["appended_bindings"], 1)

    def test_trace_expert_ingest_replans_after_concurrent_same_append(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            base = Path(temporary)
            source_root, output, fingerprint, append_args = self._sparse_append_fixture(
                base
            )
            trace = self._expert_trace(
                source_root,
                output,
                fingerprint,
                (0, 1),
            )
            trace_path = base / "sealed-trace.json"
            trace_path.write_bytes(trace.to_bytes())
            args = self._trace_args(append_args, trace_path, plan_only=False)
            original_append = bundle_script.append_experts

            def concurrent_append(inner_args):
                original_append(inner_args)
                return original_append(inner_args)

            with (
                mock.patch.object(
                    bundle_script,
                    "_open_remote_source",
                    side_effect=lambda _args: self._remote(source_root),
                ) as opener,
                mock.patch.object(
                    bundle_script,
                    "append_experts",
                    side_effect=concurrent_append,
                ),
            ):
                result = bundle_script.append_trace_experts(args)

            self.assertEqual(result["status"], "already-appended")
            self.assertEqual(result["plan"]["body"]["counts"]["missing"], 0)
            self.assertEqual(
                result["plan"]["body"]["counts"]["missing_payload_bytes"], 0
            )
            self.assertEqual(opener.call_count, 1)
            self.assertRegex(
                result["append"]["requested_plan_sha256"], r"^[0-9a-f]{64}$"
            )

    def test_trace_expert_plan_unions_traces_and_deduplicates_trace_sha(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            base = Path(temporary)
            source_root, output, fingerprint, append_args = self._sparse_append_fixture(
                base
            )
            trace_zero = self._expert_trace(
                source_root,
                output,
                fingerprint,
                (0,),
            )
            trace_one = self._expert_trace(
                source_root,
                output,
                fingerprint,
                (1,),
            )
            zero_path = base / "trace-zero.json"
            duplicate_zero_path = base / "trace-zero-copy.json"
            one_path = base / "trace-one.json"
            zero_path.write_bytes(trace_zero.to_bytes())
            duplicate_zero_path.write_bytes(trace_zero.to_bytes())
            one_path.write_bytes(trace_one.to_bytes())

            values = vars(append_args).copy()
            values.update(
                access_trace=[str(zero_path), str(one_path)],
                plan_only=True,
            )
            unique = bundle_script.append_trace_experts(argparse.Namespace(**values))
            values["access_trace"] = [
                str(one_path),
                str(duplicate_zero_path),
                str(zero_path),
            ]
            duplicated = bundle_script.append_trace_experts(
                argparse.Namespace(**values)
            )

            self.assertEqual(unique, duplicated)
            body = unique["plan"]["body"]
            self.assertEqual(body["counts"]["traces"], 2)
            self.assertEqual(body["counts"]["unique_leaves"], 12)
            self.assertEqual(body["counts"]["derived"], 2)
            self.assertEqual(body["counts"]["derived_payload_bytes"], 42)
            self.assertEqual(body["counts"]["missing_leaves"], 6)
            self.assertEqual(body["counts"]["missing_payload_bytes"], 21)
            self.assertEqual(len(body["trace_evidence"]), 2)
            self.assertEqual(
                body["coordinates"]["derived"],
                [
                    {"expert_id": 0, "layer": 3},
                    {"expert_id": 1, "layer": 3},
                ],
            )

    def test_trace_expert_ingest_rejects_identity_tamper_symlink_and_incomplete(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            base = Path(temporary)
            source_root, output, fingerprint, append_args = self._sparse_append_fixture(
                base
            )
            trace = self._expert_trace(
                source_root,
                output,
                fingerprint,
                (1,),
            )
            valid_path = base / "valid.json"
            valid_path.write_bytes(trace.to_bytes())

            empty = bundle_script.AccessTrace(
                repo_id=trace.repo_id,
                revision=trace.revision,
                inventory_fingerprint=trace.inventory_fingerprint,
                operations=(),
            )
            empty_path = base / "empty.json"
            empty_path.write_bytes(empty.to_bytes())
            with self.assertRaisesRegex(
                bundle_script.SparseBundleError,
                "access trace is empty",
            ):
                bundle_script.append_trace_experts(
                    self._trace_args(append_args, empty_path, plan_only=True)
                )

            wrong_revision = "b" * 40
            wrong = bundle_script.AccessTrace(
                repo_id=trace.repo_id,
                revision=wrong_revision,
                inventory_fingerprint=trace.inventory_fingerprint,
                operations=tuple(
                    replace(operation, revision=wrong_revision)
                    for operation in trace.operations
                ),
            )
            wrong_path = base / "wrong.json"
            wrong_path.write_bytes(wrong.to_bytes())
            with self.assertRaisesRegex(
                bundle_script.SparseBundleError,
                "identity mismatch",
            ):
                bundle_script.append_trace_experts(
                    self._trace_args(append_args, wrong_path, plan_only=True)
                )

            tampered = json.loads(trace.to_json())
            tampered["sha256"] = "0" * 64
            tampered_path = base / "tampered.json"
            tampered_path.write_text(
                json.dumps(tampered, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                bundle_script.SparseBundleError,
                "invalid sealed access trace",
            ):
                bundle_script.append_trace_experts(
                    self._trace_args(append_args, tampered_path, plan_only=True)
                )

            link_path = base / "trace-link.json"
            link_path.symlink_to(valid_path.name)
            with self.assertRaisesRegex(
                bundle_script.SparseBundleError,
                "non-symlink regular file",
            ):
                bundle_script.append_trace_experts(
                    self._trace_args(append_args, link_path, plan_only=True)
                )

            incomplete = self._expert_trace(
                source_root,
                output,
                fingerprint,
                (1,),
                omit_last_part=True,
            )
            incomplete_path = base / "incomplete.json"
            incomplete_path.write_bytes(incomplete.to_bytes())
            with self.assertRaisesRegex(
                bundle_script.SparseBundleError,
                "incomplete for every routed expert",
            ):
                bundle_script.append_trace_experts(
                    self._trace_args(append_args, incomplete_path, plan_only=True)
                )

    def test_trace_expert_ingest_rejects_pending_append(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            base = Path(temporary)
            source_root, output, fingerprint, append_args = self._sparse_append_fixture(
                base
            )
            trace = self._expert_trace(
                source_root,
                output,
                fingerprint,
                (1,),
            )
            trace_path = base / "sealed-trace.json"
            trace_path.write_bytes(trace.to_bytes())
            append_args.inject_crash = "payload-before-binding"
            with (
                mock.patch.object(
                    bundle_script,
                    "_open_remote_source",
                    side_effect=lambda _args: self._remote(source_root),
                ),
                self.assertRaises(bundle_script.InjectedAppendCrash),
            ):
                bundle_script.append_experts(append_args)

            args = self._trace_args(append_args, trace_path, plan_only=True)
            args.inject_crash = None
            with self.assertRaisesRegex(
                bundle_script.SparseBundleError,
                "pending expert append",
            ):
                bundle_script.append_trace_experts(args)

    def test_trace_expert_cli_wires_sealed_traces_and_plan_mode(self) -> None:
        args = bundle_script._parser().parse_args(
            [
                "append-trace-experts",
                "--bundle",
                "/tmp/model.causal",
                "--access-trace",
                "/tmp/trace-a.json",
                "--access-trace",
                "/tmp/trace-b.json",
                "--plan-only",
            ]
        )
        self.assertIs(args.handler, bundle_script.append_trace_experts)
        self.assertEqual(
            args.access_trace,
            ["/tmp/trace-a.json", "/tmp/trace-b.json"],
        )
        self.assertTrue(args.plan_only)
        self.assertEqual(args.resident_limit_mb, 64.0)
        self.assertEqual(args.staging_limit_mb, 256.0)


if __name__ == "__main__":
    unittest.main()
