from __future__ import annotations

import argparse
from copy import deepcopy
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

from immer.knowledge import (
    AccessTrace,
    AccessTraceRecorder,
    ByteBudgetExceeded,
    Streamer,
)
from immer.runtimes.deepseek_v4 import CausalWeightMount
from immer.runtimes.deepseek_v4.causal_weights import tensor_range_plan_from_source

from test_deepseek_v4_causal_weights import _LOGICAL_MODEL, _write_expert_fixture


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deepseek_v4_sparse_causal_bundle.py"
MARKER_NAME = "packing.marker"
MARKER_BYTES = 32
PARTIAL_MARKER_BYTES = 7


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "deepseek_v4_sparse_causal_bundle_dense_promotion_tests", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import sparse causal bundle script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bundle_script = _load_script()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _tree_snapshot(root: Path) -> tuple[tuple[str, str, str], ...]:
    rows: list[tuple[str, str, str]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if path.is_symlink():
            rows.append((relative, "symlink", os.readlink(path)))
        elif path.is_dir():
            rows.append((relative, "directory", ""))
        elif path.is_file():
            rows.append(
                (
                    relative,
                    "file",
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
        else:
            rows.append((relative, f"mode:{metadata.st_mode}", ""))
    return tuple(rows)


class DenseBundlePromotionTests(unittest.TestCase):
    def _fixture(
        self,
        base: Path,
        *,
        marker_trace_bytes: int = PARTIAL_MARKER_BYTES,
    ) -> dict[str, object]:
        source_root = base / "source"
        cache = base / "cache"
        output = base / "model.causal"
        _write_expert_fixture(
            source_root,
            expert_ids=(0, 1),
            prefix_bytes=MARKER_BYTES,
        )
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
            marker = next(
                tensor
                for tensor in inventory["tensors"]
                if tensor["name"] == MARKER_NAME
            )
            marker_begin, _marker_stop = marker["offset_in_shard"]
            marker_absolute_offset = int(marker["data_start"]) + int(marker_begin)
            source.raw_bytes(
                str(marker["shard"]),
                marker_absolute_offset,
                marker_trace_bytes,
            )
            for tensor in inventory["tensors"]:
                if ".experts.0." not in str(tensor["name"]):
                    continue
                begin, stop = tensor["offset_in_shard"]
                source.raw_bytes(
                    str(tensor["shard"]),
                    int(tensor["data_start"]) + int(begin),
                    int(stop) - int(begin),
                )
            trace = recorder.snapshot()
        finally:
            source.close()

        trace_path = base / "access-dense-fixture.json"
        trace_path.write_bytes(trace.to_bytes())
        inventory_path = next((cache / "inventories").glob("*.json"))
        inventory_document = json.loads(inventory_path.read_text(encoding="utf-8"))
        fingerprint = str(inventory_document["source_fingerprint"])
        manifest = bundle_script.build_bundle(
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
        args = argparse.Namespace(
            access_trace=[str(trace_path)],
            budget_mb=16.0,
            bundle=str(output),
            inject_crash=None,
            layout_fingerprint=fingerprint,
            plan_only=False,
            remote_cache_dir=None,
            remote_cache_limit_mb=16.0,
            repo_id=_LOGICAL_MODEL.repo_id,
            resident_limit_mb=1.0,
            revision=_LOGICAL_MODEL.revision,
            staging_limit_mb=2.0,
        )
        return {
            "args": args,
            "fingerprint": fingerprint,
            "inventory": inventory_document["inventory"],
            "manifest": manifest,
            "marker_absolute_offset": marker_absolute_offset,
            "output": output,
            "source_root": source_root,
            "trace": trace,
            "trace_path": trace_path,
        }

    @staticmethod
    def _remote(source_root: Path, *, budget_mb: float = 16.0) -> Streamer:
        return Streamer.from_local(
            source_root,
            repo_id=_LOGICAL_MODEL.repo_id,
            revision=_LOGICAL_MODEL.revision,
            use_cache=False,
            budget_mb=budget_mb,
        )

    @staticmethod
    def _verify_args(fixture: dict[str, object]) -> argparse.Namespace:
        return argparse.Namespace(
            budget_mb=16.0,
            bundle=str(fixture["output"]),
            layout_fingerprint=str(fixture["fingerprint"]),
            repo_id=_LOGICAL_MODEL.repo_id,
            revision=_LOGICAL_MODEL.revision,
        )

    def _promote(self, fixture: dict[str, object]) -> dict[str, object]:
        source_root = fixture["source_root"]
        assert isinstance(source_root, Path)
        with mock.patch.object(
            bundle_script,
            "_open_remote_source",
            side_effect=lambda _args: self._remote(source_root),
        ):
            return bundle_script.promote_dense(fixture["args"])

    def _assert_marker_unbound(self, output: Path) -> None:
        with CausalWeightMount(output, _LOGICAL_MODEL, budget_mb=16) as mount:
            with self.assertRaises(KeyError):
                mount.resolve_tensor_plan(MARKER_NAME)
            self.assertEqual(mount.resolve_expert_plans(3, (0,))[0].expert_id, 0)

    def test_plan_only_is_exact_read_only_and_offline(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            fixture = self._fixture(Path(temporary))
            args = fixture["args"]
            assert isinstance(args, argparse.Namespace)
            args.plan_only = True
            output = fixture["output"]
            manifest = fixture["manifest"]
            inventory = fixture["inventory"]
            trace = fixture["trace"]
            assert isinstance(output, Path)
            assert isinstance(manifest, dict)
            assert isinstance(inventory, dict)
            assert isinstance(trace, AccessTrace)
            before = _tree_snapshot(output)

            with mock.patch.object(
                bundle_script,
                "_open_remote_source",
                side_effect=AssertionError("plan-only opened a remote source"),
            ) as opener:
                planned = bundle_script.promote_dense(args)
            opener.assert_not_called()
            self.assertEqual(_tree_snapshot(output), before)

            marker = next(
                tensor
                for tensor in inventory["tensors"]
                if tensor["name"] == MARKER_NAME
            )
            begin, stop = marker["offset_in_shard"]
            plan_record = {
                "absolute_offset": int(marker["data_start"]) + int(begin),
                "dtype": str(marker["dtype"]).upper(),
                "length": int(stop) - int(begin),
                "name": MARKER_NAME,
                "shape": list(marker["shape"]),
                "shard": str(marker["shard"]),
            }
            leaves = {
                (leaf.shard, leaf.offset, leaf.length)
                for operation in trace.operations
                for leaf in operation.leaves
            }
            evidence = [
                {
                    "operation_count": len(trace.operations),
                    "sha256": trace.sha256,
                    "unique_leaves": len(leaves),
                }
            ]
            expected_body = {
                "bundle_manifest": {
                    "file_sha256": hashlib.sha256(_canonical(manifest)).hexdigest(),
                    "manifest_sha256": manifest["sha256"],
                    "schema": bundle_script.BUNDLE_SCHEMA,
                },
                "categories": {
                    "compressor_norm": 0,
                    "embedding": 0,
                    "head": 0,
                    "other": 1,
                    "tid2eid": 0,
                },
                "layout_fingerprint": fixture["fingerprint"],
                "logical_model": _LOGICAL_MODEL.as_record(),
                "materialization_bytes": MARKER_BYTES,
                "materialization_names": [MARKER_NAME],
                "materialization_names_sha256": _sha([MARKER_NAME]),
                "materialization_tensor_count": 1,
                "plans_sha256": _sha([plan_record]),
                "required_tensor_bytes": MARKER_BYTES,
                "required_tensor_count": 1,
                "required_tensor_names_sha256": _sha([MARKER_NAME]),
                "schema": bundle_script.DENSE_PLAN_SCHEMA,
                "trace_evidence": evidence,
                "trace_evidence_sha256": _sha(evidence),
                "inventory_layout_sha256": _sha(
                    Streamer._inventory_layout_projection(inventory)
                ),
            }
            self.assertEqual(
                planned,
                {
                    "body": expected_body,
                    "schema": bundle_script.DENSE_PLAN_SCHEMA,
                    "sha256": _sha(expected_body),
                },
            )

    def test_promotion_is_verified_idempotent_offline_and_preserves_other_bytes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            fixture = self._fixture(Path(temporary))
            output = fixture["output"]
            source_root = fixture["source_root"]
            marker_offset = fixture["marker_absolute_offset"]
            manifest = fixture["manifest"]
            assert isinstance(output, Path)
            assert isinstance(source_root, Path)
            assert isinstance(marker_offset, int)
            assert isinstance(manifest, dict)
            shard = output / "weights" / "model.safetensors"
            source_shard = source_root / "model.safetensors"
            before = shard.read_bytes()
            official = source_shard.read_bytes()
            self.assertEqual(
                before[marker_offset : marker_offset + PARTIAL_MARKER_BYTES],
                official[marker_offset : marker_offset + PARTIAL_MARKER_BYTES],
            )
            self.assertNotEqual(
                before[marker_offset : marker_offset + MARKER_BYTES],
                official[marker_offset : marker_offset + MARKER_BYTES],
            )

            promoted = self._promote(fixture)
            self.assertEqual(promoted["status"], "promoted")
            self.assertEqual(
                promoted["capability"],
                bundle_script.GENERAL_DENSE_COVERAGE_CAPABILITY,
            )
            self.assertEqual(promoted["required_tensor_count"], 1)
            self.assertEqual(promoted["materialization_bytes"], MARKER_BYTES)
            after = shard.read_bytes()
            self.assertEqual(
                after[marker_offset : marker_offset + MARKER_BYTES],
                official[marker_offset : marker_offset + MARKER_BYTES],
            )
            self.assertEqual(before[:marker_offset], after[:marker_offset])
            self.assertEqual(
                before[marker_offset + MARKER_BYTES :],
                after[marker_offset + MARKER_BYTES :],
            )

            with CausalWeightMount(output, _LOGICAL_MODEL, budget_mb=16) as mount:
                marker_plan = mount.resolve_tensor_plan(MARKER_NAME)
                expert_zero = mount.resolve_expert_plans(3, (0,))[0]
                with self.assertRaises(KeyError):
                    mount.resolve_expert_plans(3, (1,))
            self.assertEqual(marker_plan.absolute_offset, marker_offset)
            self.assertEqual(marker_plan.length, MARKER_BYTES)
            self.assertEqual(expert_zero.expert_id, 0)

            envelope = json.loads(output.joinpath("bundle.json").read_text())
            self.assertEqual(envelope["schema"], bundle_script.GENERAL_BUNDLE_SCHEMA)
            self.assertEqual(envelope["sha256"], _sha(envelope["body"]))
            self.assertEqual(envelope["body"]["base_bundle"], manifest)
            self.assertEqual(
                envelope["body"]["capabilities"],
                {
                    "general_dense_weight_coverage": (
                        bundle_script.GENERAL_DENSE_COVERAGE_CAPABILITY
                    )
                },
            )
            coverage = envelope["body"]["dense_coverage"]
            self.assertEqual(
                set(coverage),
                {
                    "graph_post_bind",
                    "materialization_bytes",
                    "payload_ledger_sha256",
                    "plans_sha256",
                    "receipt_sha256",
                    "required_tensor_count",
                    "required_tensor_names_sha256",
                    "transaction_id",
                },
            )
            self.assertEqual(coverage["receipt_sha256"], promoted["receipt_sha256"])

            verified = bundle_script.verify_bundle(self._verify_args(fixture))
            self.assertEqual(
                verified["general_dense_weight_coverage"],
                bundle_script.GENERAL_DENSE_COVERAGE_CAPABILITY,
            )
            self.assertEqual(verified["dense_coverage"]["required_tensor_count"], 1)
            snapshot = _tree_snapshot(output)
            with mock.patch.object(
                bundle_script,
                "_open_remote_source",
                side_effect=AssertionError("idempotent replay used the network"),
            ) as opener:
                replay = bundle_script.promote_dense(fixture["args"])
            opener.assert_not_called()
            self.assertEqual(replay["status"], "already-promoted")
            self.assertEqual(replay["receipt_sha256"], promoted["receipt_sha256"])
            self.assertEqual(_tree_snapshot(output), snapshot)

    def test_promotion_preserves_expert_append_journal_and_replay(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            fixture = self._fixture(Path(temporary))
            output = fixture["output"]
            source_root = fixture["source_root"]
            assert isinstance(output, Path)
            assert isinstance(source_root, Path)
            append_args = argparse.Namespace(
                budget_mb=16.0,
                bundle=str(output),
                expert=[(3, 1)],
                inject_crash=None,
                layout_fingerprint=fixture["fingerprint"],
                remote_cache_dir=None,
                remote_cache_limit_mb=16.0,
                repo_id=_LOGICAL_MODEL.repo_id,
                resident_limit_mb=1.0,
                revision=_LOGICAL_MODEL.revision,
                staging_limit_mb=2.0,
            )
            with mock.patch.object(
                bundle_script,
                "_open_remote_source",
                side_effect=lambda _args: self._remote(source_root),
            ):
                appended = bundle_script.append_experts(append_args)
            self.assertEqual(appended["status"], "appended")
            append_root = output / "expert-appends"
            append_snapshot = _tree_snapshot(append_root)

            promoted = self._promote(fixture)
            self.assertEqual(promoted["status"], "promoted")
            self.assertEqual(_tree_snapshot(append_root), append_snapshot)
            with CausalWeightMount(output, _LOGICAL_MODEL, budget_mb=16) as mount:
                self.assertEqual(mount.resolve_expert_plans(3, (1,))[0].expert_id, 1)
                self.assertEqual(mount.resolve_tensor_plan(MARKER_NAME).length, 32)
            with mock.patch.object(
                bundle_script,
                "_open_remote_source",
                side_effect=AssertionError(
                    "expert replay after promotion used network"
                ),
            ) as opener:
                replay = bundle_script.append_experts(append_args)
            opener.assert_not_called()
            self.assertEqual(replay["status"], "already-appended")
            verified = bundle_script.verify_bundle(self._verify_args(fixture))
            self.assertEqual(verified["appended_bindings"], 1)
            self.assertEqual(
                verified["general_dense_weight_coverage"],
                bundle_script.GENERAL_DENSE_COVERAGE_CAPABILITY,
            )

    def test_all_durable_crash_boundaries_reconcile_offline(self) -> None:
        cases = (
            ("payload-before-binding", "payload_durable", "promoted"),
            ("binding-before-journal", "payload_durable", "promoted"),
            ("receipt-before-capability", "binding_visible", "already-promoted"),
            ("capability-before-cleanup", "binding_visible", "already-promoted"),
        )
        for failpoint, pending_state, replay_status in cases:
            with (
                self.subTest(failpoint=failpoint),
                tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary,
            ):
                fixture = self._fixture(Path(temporary))
                args = fixture["args"]
                source_root = fixture["source_root"]
                output = fixture["output"]
                assert isinstance(args, argparse.Namespace)
                assert isinstance(source_root, Path)
                assert isinstance(output, Path)
                args.inject_crash = failpoint
                with (
                    mock.patch.object(
                        bundle_script,
                        "_open_remote_source",
                        side_effect=lambda _args: self._remote(source_root),
                    ),
                    self.assertRaisesRegex(
                        bundle_script.InjectedAppendCrash, failpoint
                    ),
                ):
                    bundle_script.promote_dense(args)

                pending_path = output / "dense-promotion" / "pending.json"
                pending = json.loads(pending_path.read_text(encoding="utf-8"))
                self.assertEqual(pending["body"]["history"][-1]["state"], pending_state)
                with self.assertRaises(bundle_script.SparseBundleError):
                    bundle_script.verify_bundle(self._verify_args(fixture))

                args.inject_crash = None
                with mock.patch.object(
                    bundle_script,
                    "_open_remote_source",
                    side_effect=AssertionError("durable reconcile used the network"),
                ) as opener:
                    replay = bundle_script.promote_dense(args)
                opener.assert_not_called()
                self.assertEqual(replay["status"], replay_status)
                self.assertFalse(pending_path.exists())
                self.assertFalse(
                    any(output.joinpath("dense-promotion").glob("stage-*"))
                )
                verified = bundle_script.verify_bundle(self._verify_args(fixture))
                self.assertEqual(
                    verified["general_dense_weight_coverage"],
                    bundle_script.GENERAL_DENSE_COVERAGE_CAPABILITY,
                )

    def test_partial_remote_and_resource_bounds_never_publish_or_bind(self) -> None:
        cases = ("partial", "resident", "staging")
        for case in cases:
            with (
                self.subTest(case=case),
                tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary,
            ):
                fixture = self._fixture(Path(temporary))
                args = fixture["args"]
                source_root = fixture["source_root"]
                output = fixture["output"]
                assert isinstance(args, argparse.Namespace)
                assert isinstance(source_root, Path)
                assert isinstance(output, Path)
                shard = output / "weights" / "model.safetensors"
                before = shard.read_bytes()
                if case == "partial":
                    remote = self._remote(source_root)
                    original = remote.raw_bytes

                    def partial(shard_name: str, offset: int, length: int) -> bytes:
                        return bytes(original(shard_name, offset, length))[:-1]

                    remote.raw_bytes = partial  # type: ignore[method-assign]
                    opener = mock.patch.object(
                        bundle_script, "_open_remote_source", return_value=remote
                    )
                    message = "partial dense range"
                else:
                    if case == "resident":
                        args.resident_limit_mb = 1e-9
                        message = "resident memory bound"
                    else:
                        args.staging_limit_mb = 1e-9
                        message = "staging disk bound"
                    opener = mock.patch.object(
                        bundle_script,
                        "_open_remote_source",
                        side_effect=lambda _args: self._remote(source_root),
                    )
                with (
                    opener,
                    self.assertRaisesRegex(bundle_script.SparseBundleError, message),
                ):
                    bundle_script.promote_dense(args)

                self.assertEqual(shard.read_bytes(), before)
                self.assertEqual(
                    json.loads(output.joinpath("bundle.json").read_text())["schema"],
                    bundle_script.BUNDLE_SCHEMA,
                )
                self.assertFalse(
                    any(output.joinpath("dense-promotion", "receipts").glob("*.json"))
                )
                self._assert_marker_unbound(output)

    def test_transfer_budget_is_independent_of_resident_chunk_window(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            fixture = self._fixture(Path(temporary))
            args = fixture["args"]
            source_root = fixture["source_root"]
            assert isinstance(args, argparse.Namespace)
            assert isinstance(source_root, Path)
            resident_bytes = 8
            args.resident_limit_mb = resident_bytes / (1024 * 1024)
            args.budget_mb = 1.0
            remote = self._remote(source_root, budget_mb=args.budget_mb)
            original = remote.raw_bytes
            requests: list[int] = []

            def observed(shard: str, offset: int, length: int) -> bytes:
                requests.append(length)
                return original(shard, offset, length)

            remote.raw_bytes = observed  # type: ignore[method-assign]
            with mock.patch.object(
                bundle_script, "_open_remote_source", return_value=remote
            ):
                promoted = bundle_script.promote_dense(args)
            self.assertEqual(promoted["status"], "promoted")
            self.assertGreater(sum(requests), resident_bytes)
            self.assertLessEqual(max(requests), resident_bytes)

    def test_transfer_budget_exhausts_before_any_weight_or_graph_mutation(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            fixture = self._fixture(Path(temporary))
            args = fixture["args"]
            source_root = fixture["source_root"]
            output = fixture["output"]
            assert isinstance(args, argparse.Namespace)
            assert isinstance(source_root, Path)
            assert isinstance(output, Path)
            resident_bytes = 8
            transfer_bytes = 16
            args.resident_limit_mb = resident_bytes / (1024 * 1024)
            args.budget_mb = transfer_bytes / (1024 * 1024)
            remote = self._remote(source_root, budget_mb=args.budget_mb)
            shard = output / "weights" / "model.safetensors"
            before = shard.read_bytes()
            with (
                mock.patch.object(
                    bundle_script, "_open_remote_source", return_value=remote
                ),
                self.assertRaisesRegex(ByteBudgetExceeded, "Bytebudget"),
            ):
                bundle_script.promote_dense(args)
            self.assertEqual(shard.read_bytes(), before)
            self.assertEqual(
                json.loads(output.joinpath("bundle.json").read_text())["schema"],
                bundle_script.BUNDLE_SCHEMA,
            )
            self.assertFalse(
                any(output.joinpath("dense-promotion", "receipts").glob("*.json"))
            )
            self._assert_marker_unbound(output)

    def test_remote_source_uses_transfer_budget_not_resident_limit(self) -> None:
        args = argparse.Namespace(
            budget_mb=12288.0,
            remote_cache_dir=None,
            remote_cache_limit_mb=256.0,
            repo_id=_LOGICAL_MODEL.repo_id,
            resident_limit_mb=64.0,
            revision=_LOGICAL_MODEL.revision,
        )
        sentinel = mock.Mock()
        with mock.patch.object(
            bundle_script, "Streamer", return_value=sentinel
        ) as constructor:
            opened = bundle_script._open_remote_source(args)
        self.assertIs(opened, sentinel)
        self.assertEqual(constructor.call_args.kwargs["budget_mb"], 12288.0)
        self.assertNotEqual(
            constructor.call_args.kwargs["budget_mb"], args.resident_limit_mb
        )

    def test_fully_trace_covered_local_drift_is_not_resealed_as_trusted(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            fixture = self._fixture(Path(temporary), marker_trace_bytes=MARKER_BYTES)
            output = fixture["output"]
            source_root = fixture["source_root"]
            marker_offset = fixture["marker_absolute_offset"]
            args = fixture["args"]
            assert isinstance(output, Path)
            assert isinstance(source_root, Path)
            assert isinstance(marker_offset, int)
            assert isinstance(args, argparse.Namespace)
            args.plan_only = True
            plan = bundle_script.promote_dense(args)
            self.assertEqual(plan["body"]["materialization_tensor_count"], 0)
            args.plan_only = False

            shard = output / "weights" / "model.safetensors"
            descriptor = os.open(shard, os.O_RDWR)
            try:
                original = os.pread(descriptor, 1, marker_offset)
                os.pwrite(descriptor, bytes((original[0] ^ 0xFF,)), marker_offset)
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
                    "disagrees with official source",
                ),
            ):
                bundle_script.promote_dense(args)
            self.assertEqual(
                json.loads(output.joinpath("bundle.json").read_text())["schema"],
                bundle_script.BUNDLE_SCHEMA,
            )
            self.assertFalse(
                any(output.joinpath("dense-promotion", "receipts").glob("*.json"))
            )
            self._assert_marker_unbound(output)

    def test_verify_rejects_payload_ledger_receipt_and_capability_tamper(self) -> None:
        cases = (
            ("payload", "payload ledger hash mismatch"),
            ("ledger", "payload ledger hash mismatch"),
            ("receipt", "receipt identity"),
            ("capability", "capability"),
        )
        for case, message in cases:
            with (
                self.subTest(case=case),
                tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary,
            ):
                fixture = self._fixture(Path(temporary))
                promoted = self._promote(fixture)
                output = fixture["output"]
                marker_offset = fixture["marker_absolute_offset"]
                assert isinstance(output, Path)
                assert isinstance(marker_offset, int)
                receipt_path = output.joinpath(
                    "dense-promotion",
                    "receipts",
                    f"{promoted['transaction_id']}.json",
                )
                if case == "payload":
                    shard = output / "weights" / "model.safetensors"
                    descriptor = os.open(shard, os.O_RDWR)
                    try:
                        original = os.pread(descriptor, 1, marker_offset)
                        os.pwrite(
                            descriptor,
                            bytes((original[0] ^ 0xFF,)),
                            marker_offset,
                        )
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                elif case == "ledger":
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    result = deepcopy(receipt["body"]["result"])
                    result["payload_ledger"][0]["sha256"] = "0" * 64
                    result["payload_ledger_sha256"] = _sha(result["payload_ledger"])
                    history = list(receipt["body"]["journal"]["history"])
                    history[-1] = bundle_script._journal_event(
                        history[:-1], "binding_visible", result
                    )
                    rewritten = bundle_script._dense_receipt_document(
                        receipt["body"]["transaction"], history
                    )
                    receipt_path.write_bytes(_canonical(rewritten))
                elif case == "receipt":
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    receipt["sha256"] = "0" * 64
                    receipt_path.write_bytes(_canonical(receipt))
                else:
                    capability_path = output / "bundle.json"
                    capability = json.loads(capability_path.read_text(encoding="utf-8"))
                    capability["body"]["capabilities"][
                        "general_dense_weight_coverage"
                    ] = "tampered/v1"
                    capability["sha256"] = _sha(capability["body"])
                    capability_path.write_bytes(_canonical(capability))

                with self.assertRaisesRegex(bundle_script.SparseBundleError, message):
                    bundle_script.verify_bundle(self._verify_args(fixture))

    def test_symlinked_shard_dense_store_and_trace_fail_closed(self) -> None:
        for case in ("shard", "store", "trace"):
            with (
                self.subTest(case=case),
                tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary,
            ):
                base = Path(temporary)
                fixture = self._fixture(base)
                args = fixture["args"]
                output = fixture["output"]
                trace_path = fixture["trace_path"]
                assert isinstance(args, argparse.Namespace)
                assert isinstance(output, Path)
                assert isinstance(trace_path, Path)
                if case == "shard":
                    shard = output / "weights" / "model.safetensors"
                    real = output / "weights" / "model.real"
                    shard.rename(real)
                    shard.symlink_to(real.name)
                elif case == "store":
                    outside = base / "outside-dense-store"
                    outside.mkdir()
                    output.joinpath("dense-promotion").symlink_to(
                        outside, target_is_directory=True
                    )
                else:
                    link = base / "access-symlink.json"
                    link.symlink_to(trace_path.name)
                    args.access_trace = [str(link)]
                    args.plan_only = True

                with (
                    mock.patch.object(
                        bundle_script,
                        "_open_remote_source",
                        side_effect=AssertionError("symlink case reached the network"),
                    ) as opener,
                    self.assertRaises(Exception),
                ):
                    bundle_script.promote_dense(args)
                opener.assert_not_called()
                self.assertEqual(
                    json.loads(output.joinpath("bundle.json").read_text())["schema"],
                    bundle_script.BUNDLE_SCHEMA,
                )

    def test_graph_conflict_is_detected_before_payload_or_remote_access(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            fixture = self._fixture(Path(temporary))
            output = fixture["output"]
            source_root = fixture["source_root"]
            assert isinstance(output, Path)
            assert isinstance(source_root, Path)
            source = self._remote(source_root)
            try:
                marker_plan = tensor_range_plan_from_source(source, MARKER_NAME)
            finally:
                source.close()
            conflicting = replace(
                marker_plan, absolute_offset=marker_plan.absolute_offset + 1
            )
            with CausalWeightMount(output, _LOGICAL_MODEL, budget_mb=16) as mount:
                mount.bind_tensor_plans((conflicting,))
            shard = output / "weights" / "model.safetensors"
            before = shard.read_bytes()

            with (
                mock.patch.object(
                    bundle_script,
                    "_open_remote_source",
                    side_effect=AssertionError("graph conflict reached the network"),
                ) as opener,
                self.assertRaisesRegex(
                    bundle_script.SparseBundleError, "causal graph conflicts"
                ),
            ):
                bundle_script.promote_dense(fixture["args"])
            opener.assert_not_called()
            self.assertEqual(shard.read_bytes(), before)
            self.assertFalse(
                output.joinpath("dense-promotion", "pending.json").exists()
            )
            self.assertEqual(
                json.loads(output.joinpath("bundle.json").read_text())["schema"],
                bundle_script.BUNDLE_SCHEMA,
            )

    def test_promote_dense_cli_defaults_and_plan_switch(self) -> None:
        args = bundle_script._parser().parse_args(
            ["promote-dense", "--bundle", "/tmp/model.causal"]
        )
        self.assertIs(args.handler, bundle_script.promote_dense)
        self.assertIsNone(args.access_trace)
        self.assertFalse(args.plan_only)
        self.assertIsNone(args.inject_crash)
        self.assertEqual(args.budget_mb, 12288.0)
        self.assertEqual(args.resident_limit_mb, 64.0)
        self.assertEqual(args.staging_limit_mb, 3072.0)
        self.assertEqual(args.remote_cache_limit_mb, 4096.0)
        planned = bundle_script._parser().parse_args(
            [
                "promote-dense",
                "--bundle",
                "/tmp/model.causal",
                "--access-trace",
                "/tmp/access-a.json",
                "--access-trace",
                "/tmp/access-b.json",
                "--plan-only",
            ]
        )
        self.assertTrue(planned.plan_only)
        self.assertEqual(
            planned.access_trace,
            ["/tmp/access-a.json", "/tmp/access-b.json"],
        )


if __name__ == "__main__":
    unittest.main()
