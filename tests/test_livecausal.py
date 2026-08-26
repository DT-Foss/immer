from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from immer.knowledge.livecausal import (
    LazyGraph,
    LiveCausalIntegrityError,
    LiveCausalValidationError,
    LiveStore,
)


def _edge(number: int) -> dict[str, object]:
    return {
        "confidence": 1.0,
        "outcome_key": f"node-{number + 1:03d}",
        "trigger_key": f"node-{number:03d}",
    }


def _process_writer(root: str, offset: int, count: int, queue: object) -> None:
    try:
        store = LiveStore(root)
        identities = [
            store.append_segment([_edge(offset + index)]) for index in range(count)
        ]
    except BaseException as exc:  # pragma: no cover - reported in parent assertion.
        queue.put((False, repr(exc)))
    else:
        queue.put((True, identities))


class LiveStoreTests(unittest.TestCase):
    def test_revision_history_authenticates_exact_sequence_event_membership(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LiveStore(temporary)
            self.assertEqual(store.revision_history(), ((0, "0" * 64),))
            store.append_segment([{"trigger_key": "a", "outcome_key": "b"}])
            first = store.revision()
            store.append_segment([{"trigger_key": "b", "outcome_key": "c"}])
            second = store.revision()

            self.assertEqual(store.revision_history(), ((0, "0" * 64), first, second))
            self.assertTrue(store.contains_revision(*first))
            self.assertTrue(store.contains_revision(*second))
            self.assertFalse(store.contains_revision(first[0], "f" * 64))

            remounted = LiveStore(temporary)
            self.assertEqual(remounted.revision_history(), store.revision_history())

    def test_append_remount_idempotency_and_journal_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = LiveStore(root)
            first_records = [
                {"outcome_key": "beta", "trigger_key": "alpha", "unicode": "Größe"}
            ]
            first = store.append_segment(first_records)
            journal_inode = store.journal_path.stat().st_ino
            journal_prefix = store.journal_path.read_bytes()
            head_before = store.head_path.read_bytes()

            self.assertEqual(store.append_segment(first_records), first)
            self.assertEqual(store.journal_path.read_bytes(), journal_prefix)
            self.assertEqual(store.head_path.read_bytes(), head_before)

            second = store.append_segment(
                [{"trigger_key": "beta", "outcome_key": "gamma"}]
            )
            self.assertEqual(store.journal_path.stat().st_ino, journal_inode)
            self.assertTrue(store.journal_path.read_bytes().startswith(journal_prefix))
            self.assertEqual(store.segments(), (first, second))
            self.assertEqual(store.record(first, 0), first_records[0])
            self.assertTrue(store.verify())

            remounted = LiveStore(root)
            self.assertEqual(remounted.segments(), (first, second))
            self.assertEqual(list(remounted.iter_records(first))[0][:2], (first, 0))
            self.assertTrue(remounted.verify())

    def test_drop_is_reversible_and_never_deletes_segment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LiveStore(temporary)
            records = [{"trigger_key": "a", "outcome_key": "b"}]
            sha = store.append_segment(records)
            segment_path = store.segment_path(sha)
            inode = segment_path.stat().st_ino

            self.assertEqual(store.drop_segments([sha]), (sha,))
            self.assertEqual(store.segments(), ())
            self.assertEqual(store.tombstones(), (sha,))
            self.assertTrue(segment_path.is_file())
            self.assertEqual(segment_path.stat().st_ino, inode)

            self.assertEqual(store.append_segment(records), sha)
            self.assertEqual(store.segments(), (sha,))
            self.assertEqual(store.tombstones(), ())
            self.assertEqual(segment_path.stat().st_ino, inode)

    def test_interrupted_uncommitted_journal_tail_is_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LiveStore(temporary)
            sha = store.append_segment([{"trigger_key": "a", "outcome_key": "b"}])
            head = json.loads(store.head_path.read_text(encoding="utf-8"))
            committed = head["journal_offset"]
            with store.journal_path.open("ab") as handle:
                handle.write(b'{"torn":')
                handle.flush()

            remounted = LiveStore(temporary)
            self.assertEqual(remounted.segments(), (sha,))
            self.assertEqual(remounted.journal_path.stat().st_size, committed)
            self.assertTrue(remounted.verify())

    def test_segment_and_committed_manifest_tamper_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LiveStore(temporary)
            sha = store.append_segment(
                [{"trigger_key": "alpha", "outcome_key": "beta"}]
            )
            segment = store.segment_path(sha)
            data = segment.read_bytes()
            segment.write_bytes(data.replace(b'"beta"', b'"zeta"', 1))
            self.assertFalse(store.verify())
            with self.assertRaises(LiveCausalIntegrityError):
                LiveStore(temporary)

        with tempfile.TemporaryDirectory() as temporary:
            store = LiveStore(temporary)
            store.append_segment([{"trigger_key": "alpha", "outcome_key": "beta"}])
            journal = bytearray(store.journal_path.read_bytes())
            position = journal.index(b'"add"') + 2
            journal[position] = ord("x")
            store.journal_path.write_bytes(journal)
            self.assertFalse(store.verify())
            with self.assertRaises(LiveCausalIntegrityError):
                LiveStore(temporary)

    def test_strict_input_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LiveStore(temporary)
            with self.assertRaises(LiveCausalValidationError):
                store.append_segment([])
            with self.assertRaises(LiveCausalValidationError):
                store.append_segment([{"bad": float("nan")}])
            with self.assertRaises(LiveCausalValidationError):
                store.append_segment([{1: "not-json-object-semantics"}])

    def test_thread_safe_concurrent_writers_do_not_lose_manifest_updates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            barrier = threading.Barrier(5)
            failures: list[BaseException] = []

            def writer(worker: int) -> None:
                try:
                    store = LiveStore(temporary)
                    barrier.wait()
                    for index in range(5):
                        store.append_segment([_edge(worker * 100 + index)])
                except BaseException as exc:  # pragma: no cover - asserted below.
                    failures.append(exc)

            threads = [
                threading.Thread(target=writer, args=(worker,)) for worker in range(5)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=20)
            self.assertFalse(any(thread.is_alive() for thread in threads))
            self.assertEqual(failures, [])
            remounted = LiveStore(temporary)
            self.assertEqual(len(remounted.segments()), 25)
            self.assertEqual(len(set(remounted.segments())), 25)
            self.assertTrue(remounted.verify())

    def test_process_safe_concurrent_writers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = multiprocessing.get_context("spawn")
            queue = context.Queue()
            processes = [
                context.Process(
                    target=_process_writer, args=(temporary, worker * 100, 3, queue)
                )
                for worker in range(2)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=30)
            reports = [queue.get(timeout=5) for _ in processes]
            self.assertTrue(all(not process.is_alive() for process in processes))
            self.assertTrue(all(process.exitcode == 0 for process in processes))
            self.assertTrue(all(ok for ok, _payload in reports), reports)
            remounted = LiveStore(temporary)
            self.assertEqual(len(remounted.segments()), 6)
            self.assertTrue(remounted.verify())


class LazyGraphTests(unittest.TestCase):
    def test_query_base_returns_complete_fanout_at_node_budget_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            graph = LazyGraph(temporary, node_budget=1)
            first = graph.append_segment(
                [
                    {"trigger_key": "router", "outcome_key": "expert-c"},
                    {"trigger_key": "router", "outcome_key": "expert-a"},
                    {"trigger_key": "expert-a", "outcome_key": "deep-path"},
                ]
            )
            second = graph.append_segment(
                [{"trigger_key": "router", "outcome_key": "expert-a"}]
            )

            _bounded, truncated = graph.query("router")
            self.assertTrue(truncated)
            self.assertEqual(
                graph.query_base("router"),
                [
                    {
                        "depth": 1,
                        "derivation": sorted([[first, 1], [second, 0]]),
                        "from_key": "router",
                        "kind": "base",
                        "to_key": "expert-a",
                    },
                    {
                        "depth": 1,
                        "derivation": [[first, 0]],
                        "from_key": "router",
                        "kind": "base",
                        "to_key": "expert-c",
                    },
                ],
            )

    def test_steady_append_indexes_only_delta_without_active_manifest_scan(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LiveStore(temporary)
            graph = LazyGraph(store)
            with (
                mock.patch.object(
                    store,
                    "segments",
                    side_effect=AssertionError("steady append scanned active manifest"),
                ),
                mock.patch.object(
                    graph,
                    "_sync_locked",
                    side_effect=AssertionError("steady append entered full graph sync"),
                ),
            ):
                graph.append_segment([{"trigger_key": "a", "outcome_key": "b"}])

            edges, truncated = graph.query("a")
            self.assertFalse(truncated)
            self.assertEqual(
                [(edge["kind"], edge["to_key"]) for edge in edges], [("base", "b")]
            )

    def test_exact_derivations_bounded_query_and_determinism(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            graph = LazyGraph(temporary, node_budget=100)
            records = [
                {"trigger_key": "a", "outcome_key": "b"},
                {"trigger_key": "b", "outcome_key": "c"},
                {"trigger_key": "c", "outcome_key": "d"},
            ]
            sha = graph.append_segment(records)

            edges, truncated = graph.query("a")
            self.assertFalse(truncated)
            self.assertEqual(
                [(edge["kind"], edge["to_key"], edge["depth"]) for edge in edges],
                [("base", "b", 1), ("inferred", "c", 2), ("inferred", "d", 3)],
            )
            self.assertEqual(edges[1]["derivation"], [[sha, 0], [sha, 1]])
            self.assertEqual(graph.resolve_derivation(edges[2]["derivation"]), records)
            self.assertEqual(graph.query("a"), (edges, False))

            remounted = LazyGraph(temporary, node_budget=100)
            self.assertEqual(remounted.query("a"), (edges, False))

            bounded, was_truncated = remounted.query("a", node_budget=1)
            self.assertTrue(was_truncated)
            self.assertEqual(
                [(edge["kind"], edge["to_key"]) for edge in bounded],
                [("base", "b")],
            )

    def test_duplicate_citations_and_drop_reappend_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            graph = LazyGraph(temporary)
            first = graph.append_segment(
                [{"trigger_key": "a", "outcome_key": "b", "source": "first"}]
            )
            second = graph.append_segment(
                [{"trigger_key": "a", "outcome_key": "b", "source": "second"}]
            )
            self.assertEqual(
                graph.base_edge_citations("a", "b"),
                sorted([[first, 0], [second, 0]]),
            )

            graph.drop_segments([first])
            self.assertEqual(graph.base_edge_citations("a", "b"), [[second, 0]])
            graph.append_segment(
                [{"trigger_key": "a", "outcome_key": "b", "source": "first"}]
            )
            self.assertEqual(
                graph.base_edge_citations("a", "b"),
                sorted([[first, 0], [second, 0]]),
            )

    def test_malformed_graph_record_fails_instead_of_being_silently_skipped(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LiveStore(temporary)
            store.append_segment([{"not": "an edge"}])
            with self.assertRaises(LiveCausalIntegrityError):
                LazyGraph(store)


if __name__ == "__main__":
    unittest.main()
