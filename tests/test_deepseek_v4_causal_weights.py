from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from immer.knowledge.livecausal import LiveCausalIntegrityError, LiveGraph
from immer.knowledge.streamer import InventoryValidationError, Streamer
from immer.runtimes.deepseek_v4 import CausalWeightMount
from immer.runtimes.deepseek_v4.causal_weights import (
    CausalTensorReader,
    CausalWeightConflictError,
    CausalWeightError,
    CausalWeightIdentityError,
    CausalWeightIntegrityError,
    CausalWeightLayoutIdentity,
    CausalWeightNotFoundError,
    CausalWeightReader,
    LogicalModelIdentity,
    bind_causal_tensor_plans,
    bind_causal_weight_plans,
    semantic_expert_key,
    semantic_tensor_key,
    tensor_range_plan_from_source,
)
from immer.runtimes.deepseek_v4.pager import (
    DeepSeekWeightPager,
    ExpertSourceRange,
    OfficialExpertRangePlan,
)


def _write_expert_fixture(
    root: Path,
    *,
    expert_ids: tuple[int, ...],
    prefix_bytes: int = 0,
) -> None:
    root.mkdir()
    tensors: list[tuple[str, str, tuple[int, ...], bytes]] = []
    if prefix_bytes:
        tensors.append(
            ("packing.marker", "U8", (prefix_bytes,), bytes([0xA5]) * prefix_bytes)
        )
    for expert_id in expert_ids:
        base = f"layers.3.ffn.experts.{expert_id}"
        for role_index, role in enumerate(("w1", "w2", "w3")):
            tensors.append(
                (
                    f"{base}.{role}.scale",
                    "F8_E8M0",
                    (2,),
                    bytes((expert_id, role_index)),
                )
            )
        for role_index, role in enumerate(("w1", "w2", "w3")):
            tensors.append(
                (
                    f"{base}.{role}.weight",
                    "I8",
                    (5,),
                    bytes(
                        (expert_id + role_index + offset) % 256 for offset in range(5)
                    ),
                )
            )

    header: dict[str, object] = {}
    payloads: list[bytes] = []
    weight_map: dict[str, str] = {}
    offset = 0
    for name, dtype, shape, payload in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        weight_map[name] = "model.safetensors"
        payloads.append(payload)
        offset += len(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    (root / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + b"".join(payloads)
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": offset}, "weight_map": weight_map},
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


_LOGICAL_MODEL = LogicalModelIdentity(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    revision="a" * 40,
)


def _source(root: Path, cache: Path) -> Streamer:
    return Streamer.from_local(
        root,
        budget_mb=16,
        cache_dir=cache,
    )


def _shift_plan(plan: OfficialExpertRangePlan, amount: int) -> OfficialExpertRangePlan:
    ranges: list[ExpertSourceRange] = []
    for source_range in plan.ranges:
        tensors = tuple(
            replace(tensor, absolute_offset=tensor.absolute_offset + amount)
            for tensor in source_range.tensors
        )
        ranges.append(
            replace(
                source_range,
                absolute_offset=source_range.absolute_offset + amount,
                tensors=tensors,
            )
        )
    return replace(plan, ranges=tuple(ranges))


class CausalWeightMonorailTests(unittest.TestCase):
    def test_generic_tensor_rail_reads_deepseek_layout_without_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source_root = base / "source"
            _write_expert_fixture(source_root, expert_ids=(0,), prefix_bytes=7)
            source = _source(source_root, base / "cache")
            layout = CausalWeightLayoutIdentity.from_source(
                source, model=_LOGICAL_MODEL
            )
            graph = LiveGraph(base / "graph")
            names = (
                "packing.marker",
                "layers.3.ffn.experts.0.w1.weight",
            )
            plans = tuple(tensor_range_plan_from_source(source, name) for name in names)
            receipt = bind_causal_tensor_plans(graph, layout, plans)
            self.assertEqual(receipt.appended_count, 2)
            reader = CausalTensorReader(graph, layout, source=source)
            expected = source.raw_bytes(plans[1].shard, plans[1].absolute_offset + 1, 3)
            with mock.patch.object(
                source,
                "find",
                side_effect=AssertionError("tensor rail used inventory discovery"),
            ):
                resolved = reader.resolve_tensor_plan(names[0])
                read = reader.read_tensor_range(names[1], relative_offset=1, length=3)
            self.assertEqual(resolved, plans[0])
            self.assertEqual(bytes(read.part), bytes(expected))
            self.assertEqual(
                len(
                    graph.query_base(semantic_tensor_key(_LOGICAL_MODEL, name=names[1]))
                ),
                1,
            )
            source.close()

    def test_pinned_remote_inventory_requires_exact_local_tensor_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            weights = base / "weights"
            _write_expert_fixture(weights, expert_ids=(0, 1))
            initial = Streamer.from_local(
                weights,
                repo_id=_LOGICAL_MODEL.repo_id,
                revision=_LOGICAL_MODEL.revision,
                use_cache=False,
                budget_mb=16,
            )
            try:
                pinned = json.loads(json.dumps(initial.inventory()))
            finally:
                initial.close()
            for index, shard in enumerate(pinned["shards"]):
                shard["etag"] = f"remote-etag-{index}"
                shard["cas_url_hash"] = f"{index + 1:064x}"
            fingerprint = Streamer._source_fingerprint(pinned)

            adopted = Streamer.from_local(
                weights,
                repo_id=_LOGICAL_MODEL.repo_id,
                revision=_LOGICAL_MODEL.revision,
                pinned_inventory=pinned,
                pinned_fingerprint=fingerprint,
                use_cache=False,
                budget_mb=16,
            )
            try:
                self.assertEqual(
                    adopted.metrics()["inventory_source_fingerprint"], fingerprint
                )
                self.assertEqual(
                    adopted.tensor("layers.3.ffn.experts.0.w1.weight").tolist(),
                    [0, 1, 2, 3, 4],
                )
            finally:
                adopted.close()

            incompatible = json.loads(json.dumps(pinned))
            incompatible["tensors"][0]["shape"] = [999]
            with self.assertRaisesRegex(InventoryValidationError, "layout|Shape"):
                Streamer.from_local(
                    weights,
                    repo_id=_LOGICAL_MODEL.repo_id,
                    revision=_LOGICAL_MODEL.revision,
                    pinned_inventory=incompatible,
                    use_cache=False,
                    budget_mb=16,
                )

    def test_local_bundle_mount_persists_graph_and_keeps_logical_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            bundle = base / "model.causal"
            weights = bundle / "weights"
            causal = bundle / "causal"
            bundle.mkdir()
            causal.mkdir()
            _write_expert_fixture(weights, expert_ids=(0, 1, 2))
            shard = weights / "model.safetensors"
            weight_bytes = shard.read_bytes()
            weight_stat = shard.stat()

            mounted = CausalWeightMount(
                bundle,
                _LOGICAL_MODEL,
                budget_mb=16,
                max_open_files=2,
            )
            with mounted as first:
                self.assertEqual(first.root, bundle.absolute())
                self.assertEqual(first.source.repo_id, _LOGICAL_MODEL.repo_id)
                self.assertEqual(first.source.revision, _LOGICAL_MODEL.revision)
                self.assertEqual(
                    first.source.metrics()["transport_fd_max_open_files"],
                    2,
                )
                self.assertEqual(first.model, _LOGICAL_MODEL)
                self.assertEqual(first.layout.model, _LOGICAL_MODEL)
                pager = DeepSeekWeightPager(
                    first.source,
                    device="cpu",
                    compute_dtype="bfloat16",
                )
                plans = pager.plan_expert_ranges(3, (0, 1, 2))
                first.bind_plans(plans[:2])
                self.assertEqual(first.resolve_expert_plans(3, (0, 1)), plans[:2])
                first.bind_plans(plans[2:])
                self.assertEqual(first.resolve_expert_plans(3, (0, 1, 2)), plans)
                read = first.read_experts(3, (2, 0))
                self.assertEqual(read.plans, (plans[2], plans[0]))
                self.assertEqual(len(first.graph.store.segments()), 2)
                layout_fingerprint = first.layout.layout_fingerprint
                pager.release()

            self.assertTrue(mounted.closed)
            with self.assertRaises(CausalWeightError):
                mounted.resolve_expert_plans(3, (0,))
            mounted.close()

            with CausalWeightMount(
                bundle,
                _LOGICAL_MODEL,
                budget_mb=16,
                max_open_files=2,
            ) as remounted:
                self.assertEqual(
                    remounted.layout.layout_fingerprint, layout_fingerprint
                )
                self.assertEqual(remounted.layout.model, _LOGICAL_MODEL)
                self.assertEqual(remounted.resolve_expert_plans(3, (0, 1, 2)), plans)
                self.assertEqual(len(remounted.graph.store.segments()), 2)
                semantic_key = semantic_expert_key(
                    _LOGICAL_MODEL,
                    layer=3,
                    expert_id=2,
                )
                self.assertEqual(len(remounted.graph.query_base(semantic_key)), 1)

            self.assertEqual(shard.read_bytes(), weight_bytes)
            after = shard.stat()
            self.assertEqual(after.st_ino, weight_stat.st_ino)
            self.assertEqual(after.st_size, weight_stat.st_size)
            self.assertEqual(after.st_mtime_ns, weight_stat.st_mtime_ns)

    def test_local_bundle_mount_rejects_symlinked_or_incomplete_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            real = base / "real"
            real.mkdir()
            (real / "causal").mkdir()
            _write_expert_fixture(real / "weights", expert_ids=(0,))
            linked_bundle = base / "model.causal"
            linked_bundle.symlink_to(real, target_is_directory=True)
            with self.assertRaises(CausalWeightIntegrityError):
                CausalWeightMount(linked_bundle, _LOGICAL_MODEL)

            incomplete = base / "incomplete.causal"
            incomplete.mkdir()
            _write_expert_fixture(incomplete / "weights", expert_ids=(0,))
            with self.assertRaises(CausalWeightIntegrityError):
                CausalWeightMount(incomplete, _LOGICAL_MODEL)

            external_weights = base / "external-weights"
            _write_expert_fixture(external_weights, expert_ids=(0,))
            linked_weights_bundle = base / "linked-weights.causal"
            linked_weights_bundle.mkdir()
            (linked_weights_bundle / "causal").mkdir()
            (linked_weights_bundle / "weights").symlink_to(
                external_weights,
                target_is_directory=True,
            )
            with self.assertRaises(CausalWeightIntegrityError):
                CausalWeightMount(linked_weights_bundle, _LOGICAL_MODEL)

    def test_bound_plans_resolve_and_read_without_metadata_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source_root = base / "source"
            expert_ids = tuple(range(7))
            _write_expert_fixture(source_root, expert_ids=expert_ids)
            source = _source(source_root, base / "cache")
            pager = DeepSeekWeightPager(
                source,
                device="cpu",
                compute_dtype="bfloat16",
            )
            plans = pager.plan_expert_ranges(3, expert_ids)
            self.assertGreater(len(plans), pager.EXPERT_PREFETCH_MAX_EXPERTS)
            layout = CausalWeightLayoutIdentity.from_source(
                source,
                model=_LOGICAL_MODEL,
            )
            self.assertTrue(source.metrics()["repo_id"].startswith("local:"))
            self.assertEqual(layout.model, _LOGICAL_MODEL)
            graph = LiveGraph(base / "graph", node_budget=1)

            first = bind_causal_weight_plans(graph, layout, plans[:5])
            self.assertEqual(first.appended_count, 5)
            self.assertIsNotNone(first.appended_segment_sha256)
            reader = CausalWeightReader(graph, layout, source=source)

            # The already-mounted reader sees appended bindings immediately.
            second = bind_causal_weight_plans(graph, layout, plans[5:])
            self.assertEqual(second.appended_count, 2)
            expected_payload = (source_root / "model.safetensors").read_bytes()
            raw_many = mock.Mock(wraps=source.raw_bytes_many)
            source.raw_bytes_many = raw_many
            with (
                mock.patch.object(
                    source,
                    "find",
                    side_effect=AssertionError("causal reader used source.find"),
                ),
                mock.patch.object(
                    pager,
                    "plan_expert_ranges",
                    side_effect=AssertionError("causal reader replanned ranges"),
                ),
                mock.patch.object(
                    graph,
                    "query",
                    side_effect=AssertionError("causal reader entered bounded DFS"),
                ),
                mock.patch.object(
                    graph,
                    "base_edge_citations",
                    side_effect=AssertionError("causal reader bypassed query_base"),
                ),
            ):
                resolved = reader.resolve_expert_plans(3, reversed(expert_ids))
                receipt = reader.read_experts(3, reversed(expert_ids))

            self.assertEqual(resolved, tuple(reversed(plans)))
            self.assertEqual(receipt.plans, resolved)
            expected_parts = tuple(
                expected_payload[
                    source_range.absolute_offset : source_range.absolute_end
                ]
                for plan in resolved
                for source_range in plan.ranges
            )
            self.assertEqual(
                tuple(bytes(part) for part in receipt.parts), expected_parts
            )
            self.assertEqual(
                receipt.requested_bytes,
                sum(len(part) for part in expected_parts),
            )
            self.assertEqual(
                [(leaf.absolute_offset, leaf.absolute_end) for leaf in receipt.leaves],
                [
                    (source_range.absolute_offset, source_range.absolute_end)
                    for plan in resolved
                    for source_range in plan.ranges
                ],
            )
            raw_many.assert_called_once()
            self.assertEqual(raw_many.call_args.kwargs["max_gap_bytes"], 0)
            self.assertEqual(
                raw_many.call_args.args[1],
                tuple(
                    (source_range.absolute_offset, source_range.length)
                    for plan in resolved
                    for source_range in plan.ranges
                ),
            )
            self.assertEqual(
                reader.metrics()["requested_bytes"], receipt.requested_bytes
            )
            pager.release()
            source.close()

    def test_repacked_layout_requires_new_bindings_but_keeps_logical_model(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            first_root = base / "first"
            second_root = base / "second"
            _write_expert_fixture(first_root, expert_ids=(4,), prefix_bytes=0)
            _write_expert_fixture(second_root, expert_ids=(4,), prefix_bytes=11)
            first_source = _source(first_root, base / "cache-first")
            second_source = _source(second_root, base / "cache-second")
            first_pager = DeepSeekWeightPager(
                first_source, device="cpu", compute_dtype="bfloat16"
            )
            second_pager = DeepSeekWeightPager(
                second_source, device="cpu", compute_dtype="bfloat16"
            )
            (first_plan,) = first_pager.plan_expert_ranges(3, (4,))
            (second_plan,) = second_pager.plan_expert_ranges(3, (4,))
            first_layout = CausalWeightLayoutIdentity.from_source(
                first_source,
                model=_LOGICAL_MODEL,
            )
            second_layout = CausalWeightLayoutIdentity.from_source(
                second_source,
                model=_LOGICAL_MODEL,
            )
            self.assertEqual(first_layout.model, second_layout.model)
            self.assertNotEqual(
                first_layout.layout_fingerprint,
                second_layout.layout_fingerprint,
            )
            self.assertNotEqual(first_plan, second_plan)

            graph = LiveGraph(base / "graph")
            bind_causal_weight_plans(graph, first_layout, (first_plan,))
            with self.assertRaises(CausalWeightIdentityError):
                CausalWeightReader(graph, first_layout, source=second_source)
            second_reader = CausalWeightReader(graph, second_layout)
            with self.assertRaises(CausalWeightNotFoundError):
                second_reader.resolve_expert_plans(3, (4,))

            bind_causal_weight_plans(graph, second_layout, (second_plan,))
            self.assertEqual(
                second_reader.resolve_expert_plans(3, (4,)),
                (second_plan,),
            )
            semantic_key = semantic_expert_key(
                _LOGICAL_MODEL,
                layer=3,
                expert_id=4,
            )
            self.assertEqual(len(graph.query_base(semantic_key)), 2)
            first_pager.release()
            second_pager.release()
            first_source.close()
            second_source.close()

    def test_overlapping_batches_are_atomic_across_live_graph_instances(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source_root = base / "source"
            expert_ids = tuple(range(6))
            _write_expert_fixture(source_root, expert_ids=expert_ids)
            source = _source(source_root, base / "cache")
            pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="bfloat16")
            plans = pager.plan_expert_ranges(3, expert_ids)
            layout = CausalWeightLayoutIdentity.from_source(
                source,
                model=_LOGICAL_MODEL,
            )
            graph_root = base / "graph"
            first_graph = LiveGraph(graph_root)
            second_graph = LiveGraph(graph_root)
            first_at_append = threading.Event()
            second_started = threading.Event()
            second_scanned = threading.Event()
            original_first_append = first_graph.append_segment
            original_second_query = second_graph.query_base

            def delayed_first_append(records):
                first_at_append.set()
                if not second_started.wait(timeout=2):
                    raise AssertionError("second binding call did not start")
                # Without the binding transaction the second graph can scan
                # the same missing experts here.  With it, that graph waits
                # outside the transaction until this batch is committed.
                second_scanned.wait(timeout=0.2)
                return original_first_append(records)

            def observed_second_query(key):
                result = original_second_query(key)
                second_scanned.set()
                return result

            def bind_second():
                if not first_at_append.wait(timeout=2):
                    raise AssertionError("first binding call did not reach append")
                second_started.set()
                return bind_causal_weight_plans(
                    second_graph,
                    layout,
                    plans[2:],
                )

            with (
                mock.patch.object(
                    first_graph,
                    "append_segment",
                    side_effect=delayed_first_append,
                ),
                mock.patch.object(
                    second_graph,
                    "query_base",
                    side_effect=observed_second_query,
                ),
                ThreadPoolExecutor(max_workers=2) as executor,
            ):
                first_future = executor.submit(
                    bind_causal_weight_plans,
                    first_graph,
                    layout,
                    plans[:4],
                )
                second_future = executor.submit(bind_second)
                first_receipt = first_future.result(timeout=5)
                second_receipt = second_future.result(timeout=5)

            self.assertEqual(first_receipt.appended_count, 4)
            self.assertEqual(
                [receipt.appended for receipt in second_receipt.bindings],
                [False, False, True, True],
            )
            observer = LiveGraph(graph_root)
            self.assertEqual(len(observer.store.segments()), 2)
            self.assertEqual(len(tuple(observer.store.iter_records())), 6)
            for expert_id in expert_ids:
                edges = observer.query_base(
                    semantic_expert_key(
                        _LOGICAL_MODEL,
                        layer=3,
                        expert_id=expert_id,
                    )
                )
                self.assertEqual(len(edges), 1)
                self.assertEqual(len(edges[0]["derivation"]), 1)

            replay = bind_causal_weight_plans(observer, layout, plans)
            self.assertEqual(replay.appended_count, 0)
            self.assertIsNone(replay.appended_segment_sha256)
            self.assertEqual(len(observer.store.segments()), 2)
            pager.release()
            source.close()

    def test_binding_lock_rejects_symlink_without_touching_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source_root = base / "source"
            _write_expert_fixture(source_root, expert_ids=(0,))
            source = _source(source_root, base / "cache")
            pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="bfloat16")
            (plan,) = pager.plan_expert_ranges(3, (0,))
            layout = CausalWeightLayoutIdentity.from_source(
                source,
                model=_LOGICAL_MODEL,
            )
            graph = LiveGraph(base / "graph")
            target = base / "do-not-lock"
            target.write_text("unchanged", encoding="utf-8")
            lock_path = graph.store.root / ".causal-weight-bindings.lock"
            lock_path.symlink_to(target)

            with self.assertRaises(CausalWeightIntegrityError):
                bind_causal_weight_plans(graph, layout, (plan,))
            self.assertEqual(target.read_text(encoding="utf-8"), "unchanged")
            self.assertEqual(graph.store.segments(), ())
            pager.release()
            source.close()

    def test_plan_cache_warm_hit_live_append_and_drop_invalidation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source_root = base / "source"
            _write_expert_fixture(source_root, expert_ids=(0, 1, 2))
            source = _source(source_root, base / "cache")
            pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="bfloat16")
            plans = pager.plan_expert_ranges(3, (0, 1, 2))
            layout = CausalWeightLayoutIdentity.from_source(
                source,
                model=_LOGICAL_MODEL,
            )
            graph_root = base / "graph"
            graph = LiveGraph(graph_root)
            writer = LiveGraph(graph_root)
            bind_causal_weight_plans(writer, layout, plans[:2])
            reader = CausalWeightReader(graph, layout, source=source)
            self.assertEqual(reader.resolve_expert_plans(3, (0, 1)), plans[:2])

            original_revision = graph.store.revision
            with (
                mock.patch.object(
                    graph.store,
                    "revision",
                    wraps=original_revision,
                ) as revision,
                mock.patch.object(
                    graph,
                    "query_base",
                    side_effect=AssertionError("warm cache entered graph adjacency"),
                ),
                mock.patch.object(
                    graph,
                    "resolve_derivation",
                    side_effect=AssertionError("warm cache resolved citations"),
                ),
                mock.patch.object(
                    graph.store,
                    "record",
                    side_effect=AssertionError("warm cache read segment records"),
                ),
                mock.patch.object(
                    source,
                    "find",
                    side_effect=AssertionError("warm cache used tensor metadata"),
                ),
            ):
                self.assertEqual(
                    reader.resolve_expert_plans(3, (1, 0, 1)),
                    (plans[1], plans[0]),
                )
            revision.assert_called_once()
            warm_metrics = reader.metrics()
            self.assertEqual(warm_metrics["plan_cache_hits"], 2)
            self.assertEqual(warm_metrics["plan_cache_misses"], 2)
            self.assertEqual(warm_metrics["plan_cache_invalidations"], 0)

            appended = bind_causal_weight_plans(writer, layout, (plans[2],))
            self.assertEqual(
                reader.resolve_expert_plans(3, (0, 1, 2)),
                plans,
            )
            appended_metrics = reader.metrics()
            self.assertEqual(appended_metrics["plan_cache_entries"], 3)
            self.assertEqual(appended_metrics["plan_cache_invalidations"], 1)
            self.assertEqual(appended_metrics["plan_cache_misses"], 5)

            assert appended.appended_segment_sha256 is not None
            writer.drop_segments((appended.appended_segment_sha256,))
            with self.assertRaises(CausalWeightNotFoundError):
                reader.resolve_expert_plans(3, (2,))
            dropped_metrics = reader.metrics()
            self.assertEqual(dropped_metrics["plan_cache_entries"], 0)
            self.assertEqual(dropped_metrics["plan_cache_invalidations"], 2)
            self.assertEqual(dropped_metrics["plan_cache_misses"], 6)
            self.assertEqual(reader.resolve_expert_plans(3, (0,)), (plans[0],))
            pager.release()
            source.close()

    def test_plan_cache_retries_if_revision_changes_during_miss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source_root = base / "source"
            _write_expert_fixture(source_root, expert_ids=(0,))
            source = _source(source_root, base / "cache")
            pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="bfloat16")
            (plan,) = pager.plan_expert_ranges(3, (0,))
            layout = CausalWeightLayoutIdentity.from_source(
                source,
                model=_LOGICAL_MODEL,
            )
            graph_root = base / "graph"
            reader_graph = LiveGraph(graph_root)
            writer = LiveGraph(graph_root)
            reader = CausalWeightReader(reader_graph, layout)
            original_query = reader_graph.query_base
            appended = False

            def append_during_first_query(key):
                nonlocal appended
                if not appended:
                    appended = True
                    bind_causal_weight_plans(writer, layout, (plan,))
                return original_query(key)

            with mock.patch.object(
                reader_graph,
                "query_base",
                side_effect=append_during_first_query,
            ) as query:
                self.assertEqual(reader.resolve_expert_plans(3, (0,)), (plan,))
            self.assertGreaterEqual(query.call_count, 2)
            metrics = reader.metrics()
            self.assertEqual(metrics["plan_cache_entries"], 1)
            self.assertEqual(metrics["plan_cache_invalidations"], 1)
            self.assertEqual(metrics["plan_cache_misses"], 1)
            pager.release()
            source.close()

    def test_binding_replay_conflict_uint64_and_segment_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source_root = base / "source"
            _write_expert_fixture(source_root, expert_ids=(1, 2))
            source = _source(source_root, base / "cache")
            pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="bfloat16")
            plans = pager.plan_expert_ranges(3, (1, 2))
            layout = CausalWeightLayoutIdentity.from_source(
                source,
                model=_LOGICAL_MODEL,
            )
            graph = LiveGraph(base / "graph")
            original = bind_causal_weight_plans(graph, layout, plans)
            self.assertEqual(len(graph.store.segments()), 1)

            replay = bind_causal_weight_plans(graph, layout, reversed(plans))
            self.assertEqual(replay.appended_count, 0)
            self.assertIsNone(replay.appended_segment_sha256)
            self.assertEqual(len(graph.store.segments()), 1)
            self.assertTrue(all(not receipt.appended for receipt in replay.bindings))

            with self.assertRaises(CausalWeightConflictError):
                bind_causal_weight_plans(graph, layout, (_shift_plan(plans[0], 1),))
            self.assertEqual(len(graph.store.segments()), 1)

            first_range = plans[0].ranges[0]
            overflowing = replace(
                plans[0],
                ranges=(
                    replace(
                        first_range,
                        absolute_offset=(1 << 64) - 1,
                        tensors=tuple(
                            replace(
                                tensor,
                                absolute_offset=(1 << 64) - 1 + tensor.range_offset,
                            )
                            for tensor in first_range.tensors
                        ),
                    ),
                    *plans[0].ranges[1:],
                ),
            )
            with self.assertRaises(CausalWeightError):
                bind_causal_weight_plans(graph, layout, (overflowing,))

            assert original.appended_segment_sha256 is not None
            segment = graph.store.segment_path(original.appended_segment_sha256)
            body = segment.read_bytes()
            marker = b'"schema":"causal-weight-binding/v1"'
            self.assertIn(marker, body)
            segment.write_bytes(body.replace(marker, marker[:-2] + b'v2"', 1))
            with self.assertRaises(LiveCausalIntegrityError):
                CausalWeightReader(graph, layout).resolve_expert_plans(3, (1,))
            pager.release()
            source.close()


if __name__ == "__main__":
    unittest.main()
