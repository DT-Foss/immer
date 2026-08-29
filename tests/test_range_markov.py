from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

from immer.knowledge import AccessLeaf, AccessOperation, Streamer
from immer.knowledge.range_markov import (
    AccessState,
    MarkovRangePrefetcher,
    RangeMarkovError,
    RangeMarkovState,
)


IDENTITY = ("local:test", "a" * 40, "b" * 64)


def _operation(
    sequence: int,
    name: str,
    *,
    leaves: tuple[AccessLeaf, ...] | None = None,
    role: str | None = None,
) -> AccessOperation:
    tags = {"tensor": name, "read_kind": "tensor", "prompt_digest": "ignored"}
    if role is not None:
        tags["access_role"] = role
    return AccessOperation(
        repo_id=IDENTITY[0],
        revision=IDENTITY[1],
        inventory_fingerprint=IDENTITY[2],
        operation="raw_bytes" if leaves is None else "raw_bytes_many",
        operation_sequence=sequence,
        thread_id=1,
        thread_name="test",
        leaves=(
            AccessLeaf(
                "model.safetensors",
                sum((index + 1) * value for index, value in enumerate(name.encode()))
                * 4096,
                4096,
            ),
        )
        if leaves is None
        else leaves,
        source_requests=1,
        source_bytes=4096,
        cache_hits=0,
        tags=tuple(sorted(tags.items())),
    )


class RangeMarkovTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_access_state_groups_leaves_and_keeps_only_semantic_tags(self) -> None:
        leaf = AccessLeaf("a", 10, 20)
        state = AccessState.from_operation(
            _operation(1, "layer.0", leaves=(leaf, leaf))
        )

        self.assertEqual(state.leaves, (leaf,))
        self.assertEqual(
            dict(state.tags),
            {"read_kind": "tensor", "tensor": "layer.0"},
        )
        self.assertEqual(AccessState.from_record(state.to_record()), state)

    def test_repeated_operation_chain_prefetches_the_exact_next_group(self) -> None:
        hints = []
        controller = MarkovRangePrefetcher(
            None,
            prefetch_range=lambda shard, offset, length: (
                hints.append((shard, offset, length)) or True
            ),
            min_support=2,
            min_confidence=0.60,
        )
        sequence = 0
        for _ in range(4):
            sequence += 1
            controller.observe(_operation(sequence, "A"))
            sequence += 1
            controller.observe(_operation(sequence, "B"))

        prediction = controller.last_prediction
        metrics = controller.metrics()

        self.assertIsNotNone(prediction)
        self.assertEqual(dict(prediction.state.tags)["tensor"], "A")
        self.assertGreaterEqual(prediction.support, 2)
        self.assertGreater(metrics["prefetch_hints"], 0)
        self.assertEqual(metrics["prefetch_hints"], len(hints))
        self.assertGreater(metrics["prediction_hits"], 0)
        self.assertTrue(all(length == 4096 for _shard, _offset, length in hints))
        controller.close()

    def test_prefetch_group_is_bounded_by_leaf_and_byte_limits(self) -> None:
        hinted = []
        controller = MarkovRangePrefetcher(
            None,
            prefetch_range=lambda shard, offset, length: (
                hinted.append((shard, offset, length)) or True
            ),
            min_support=1,
            min_confidence=0.0,
            max_prefetch_bytes=100,
            max_prefetch_leaves=2,
        )
        group = (
            AccessLeaf("a", 0, 70),
            AccessLeaf("a", 100, 70),
            AccessLeaf("b", 0, 70),
        )
        sequence = (
            _operation(1, "A"),
            _operation(2, "GROUP", leaves=group),
            _operation(3, "A"),
            _operation(4, "GROUP", leaves=group),
            _operation(5, "X"),
            _operation(6, "A"),
        )
        for operation in sequence:
            controller.observe(operation)

        plan = controller.last_beam_plan
        self.assertIsNotNone(plan)
        self.assertLessEqual(len(plan.hints), 2)
        self.assertEqual(plan.hint_bytes, 100)
        self.assertEqual(
            (plan.hints[0].leaf.shard, plan.hints[0].leaf.offset),
            ("a", 0),
        )
        self.assertEqual(plan.hints[0].distance, 1)
        self.assertGreaterEqual(plan.hints[-1].distance, 1)
        self.assertEqual(len({row.source_state_key for row in plan.hints}), 2)
        self.assertGreater(plan.duplicate_leaves_avoided, 0)
        self.assertTrue(plan.truncated_by_bytes)
        self.assertEqual(controller.metrics()["prefetch_hint_bytes"], sum(
            length for _shard, _offset, length in hinted
        ))

    def test_state_persists_and_resumes_without_relearning_the_chain(self) -> None:
        path = self.root / "range-markov.bin"
        first_hints = []
        first = MarkovRangePrefetcher(
            path,
            prefetch_range=lambda *row: (first_hints.append(row) or True),
            min_support=1,
            min_confidence=0.0,
        )
        for index, name in enumerate(("A", "B", "A", "B", "A"), start=1):
            first.observe(_operation(index, name))
        first.close()
        restored = RangeMarkovState.from_bytes(path.read_bytes())

        resumed_hints = []
        resumed = MarkovRangePrefetcher(
            path,
            prefetch_range=lambda *row: (resumed_hints.append(row) or True),
            min_support=1,
            min_confidence=0.0,
        )
        resumed.observe(_operation(1, "B"))

        self.assertGreater(restored.observations, 0)
        self.assertTrue(resumed_hints)
        self.assertEqual(resumed.metrics()["observations"], restored.observations + 1)
        resumed.close()

    def test_source_mismatch_and_corrupt_state_are_rejected(self) -> None:
        path = self.root / "range-markov.bin"
        controller = MarkovRangePrefetcher(
            path,
            prefetch_range=lambda *_row: True,
        )
        controller.observe(_operation(1, "A"))
        controller.close()
        wrong = replace(
            _operation(1, "A"),
            inventory_fingerprint="c" * 64,
        )
        resumed = MarkovRangePrefetcher(
            path,
            prefetch_range=lambda *_row: True,
        )
        with self.assertRaisesRegex(RangeMarkovError, "different source"):
            resumed.observe(wrong)
        resumed.close()

        damaged = bytearray(path.read_bytes())
        damaged[-1] ^= 1
        path.write_bytes(damaged)
        with self.assertRaises(RangeMarkovError):
            MarkovRangePrefetcher(path, prefetch_range=lambda *_row: True)

    def test_source_identity_can_fail_before_observer_attachment(self) -> None:
        controller = MarkovRangePrefetcher(
            None,
            prefetch_range=lambda *_row: True,
        )
        controller.bind_source_identity(*IDENTITY)

        with self.assertRaisesRegex(RangeMarkovError, "different source"):
            controller.bind_source_identity(
                IDENTITY[0],
                IDENTITY[1],
                "c" * 64,
            )
        self.assertEqual(controller.metrics()["observations"], 0)

    def test_prefetch_events_and_out_of_order_callbacks_do_not_train(self) -> None:
        controller = MarkovRangePrefetcher(
            None,
            prefetch_range=lambda *_row: True,
        )
        self.assertFalse(controller.observe(_operation(1, "A", role="prefetch")))
        self.assertTrue(controller.observe(_operation(2, "A")))
        self.assertFalse(controller.observe(_operation(1, "B")))

        metrics = controller.metrics()
        self.assertEqual(metrics["observations"], 1)
        self.assertEqual(metrics["ignored_prefetch_operations"], 1)
        self.assertEqual(metrics["out_of_order_operations"], 1)

    def test_ricci_eviction_keeps_frequent_state_under_capacity(self) -> None:
        controller = MarkovRangePrefetcher(
            None,
            prefetch_range=lambda *_row: False,
            max_nodes=4,
            max_contexts=8,
            min_support=1,
            min_confidence=0.0,
        )
        names = ["HOT", "A", "HOT", "B", "HOT", "C", "HOT", "D", "HOT", "E"]
        for index, name in enumerate(names, start=1):
            controller.observe(_operation(index, name))

        stored = {
            dict(node.state.tags)["tensor"]
            for node in controller._state.nodes
        }
        self.assertIn("HOT", stored)
        self.assertLessEqual(len(stored), 4)
        self.assertGreater(controller.metrics()["node_evictions"], 0)

    def test_order_two_agent_outgrows_ambiguous_order_one(self) -> None:
        controller = MarkovRangePrefetcher(
            None,
            prefetch_range=lambda *_row: False,
            min_support=1000,
        )
        names = []
        for _ in range(20):
            names.extend(("A", "B", "C", "D", "B", "E"))
        for index, name in enumerate(names, start=1):
            controller.observe(_operation(index, name))

        weights = controller.metrics()["expert_weights"]
        self.assertGreater(weights["order-2"], weights["order-1"])

    def test_beam_rolls_three_future_operations_and_scores_horizon_hits(self) -> None:
        controller = MarkovRangePrefetcher(
            None,
            prefetch_range=lambda *_row: True,
            min_support=2,
            min_confidence=0.5,
            beam_horizon=3,
            beam_width=2,
            max_prefetch_leaves=3,
            hint_cooldown_operations=1,
        )
        names = ["A", "B", "C"] * 4 + ["A"]
        for index, name in enumerate(names, start=1):
            controller.observe(_operation(index, name))

        plan = controller.last_beam_plan
        best = {
            distance: max(
                (row for row in plan.steps if row.distance == distance),
                key=lambda row: row.score,
            )
            for distance in (1, 2, 3)
        }
        self.assertEqual(
            [dict(best[index].state.tags)["tensor"] for index in (1, 2, 3)],
            ["B", "C", "A"],
        )
        self.assertGreater(best[1].path_probability, best[2].path_probability - 1e-12)
        controller.observe(_operation(len(names) + 1, "B"))
        controller.observe(_operation(len(names) + 2, "C"))
        metrics = controller.metrics()
        self.assertGreater(metrics["horizon_hits"]["1"], 0)
        self.assertGreater(metrics["horizon_hits"]["2"], 0)
        self.assertGreater(metrics["beam_candidates_considered"], 0)

    def test_hypothetical_beam_never_mutates_persistent_learning(self) -> None:
        controller = MarkovRangePrefetcher(
            None,
            prefetch_range=lambda *_row: True,
            min_support=1,
            min_confidence=0.0,
        )
        for index, name in enumerate(("A", "B", "C") * 3, start=1):
            controller.observe(_operation(index, name))
        before = controller._state

        first = controller._beam_plan_locked(before)
        second = controller._beam_plan_locked(before)

        self.assertEqual(first, second)
        self.assertEqual(controller._state, before)
        self.assertEqual(controller.metrics()["observations"], before.observations)

    def test_pruned_context_keeps_true_observation_mass(self) -> None:
        controller = MarkovRangePrefetcher(
            None,
            prefetch_range=lambda *_row: False,
            max_targets_per_context=2,
            min_support=1000,
        )
        names = ("A", "B", "A", "C", "A", "D")
        for index, name in enumerate(names, start=1):
            controller.observe(_operation(index, name))

        a_key = AccessState.from_operation(_operation(100, "A")).key
        context = next(
            row for row in controller._state.contexts if row.history == (a_key,)
        )
        self.assertEqual(context.visits, 3)
        self.assertEqual(len(context.targets), 2)
        self.assertEqual(sum(dict(context.targets).values()), 2)
        self.assertAlmostEqual(sum(context.distribution().values()), 2 / 3)

    def test_declined_hints_never_consume_cooldown(self) -> None:
        controller = MarkovRangePrefetcher(
            None,
            prefetch_range=lambda *_row: False,
            min_support=1,
            min_confidence=0.0,
            hint_cooldown_operations=8,
        )
        for index, name in enumerate(("A", "B") * 5, start=1):
            controller.observe(_operation(index, name))

        metrics = controller.metrics()
        self.assertGreater(metrics["prefetch_attempts"], 0)
        self.assertEqual(metrics["prefetch_declines"], metrics["prefetch_attempts"])
        self.assertEqual(metrics["reservoir_cooldown_leaves_skipped"], 0)

    def test_streamer_demand_trains_live_while_os_hints_stay_unobserved(self) -> None:
        header = {
            "a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
            "b": {"dtype": "U8", "shape": [4], "data_offsets": [4, 8]},
        }
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        self.root.joinpath("model.safetensors").write_bytes(
            struct.pack("<Q", len(encoded)) + encoded + b"abcdefgh"
        )
        data_start = 8 + len(encoded)
        source = Streamer.from_local(self.root, use_cache=False)
        controller = MarkovRangePrefetcher(
            self.root / "range-state.bin",
            prefetch_range=source.prefetch_range,
            min_support=1,
            min_confidence=0.0,
        )
        source.set_access_observer(controller)
        requests_before = source.budget.requests
        advised = []

        def advise(_fd: int, offset: int, length: int, _advice: int) -> None:
            advised.append((offset, length))

        try:
            with (
                mock.patch.object(os, "posix_fadvise", advise, create=True),
                mock.patch.object(os, "POSIX_FADV_WILLNEED", 3, create=True),
            ):
                for _ in range(3):
                    self.assertEqual(
                        source.raw_bytes("model.safetensors", data_start, 4),
                        b"abcd",
                    )
                    self.assertEqual(
                        source.raw_bytes("model.safetensors", data_start + 4, 4),
                        b"efgh",
                    )

            metrics = controller.metrics()
            self.assertEqual(metrics["operations"], 6)
            self.assertEqual(source.metrics()["access_observer_events"], 6)
            self.assertEqual(source.budget.requests - requests_before, 6)
            self.assertEqual(len(advised), metrics["prefetch_hints"])
            self.assertGreater(len(advised), 0)
        finally:
            source.set_access_observer(None, prepare_identity=False)
            controller.close()
            source.close()


if __name__ == "__main__":
    unittest.main()
