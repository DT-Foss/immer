from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from immer.runtimes.qwen3_8.mlp_page_markov import (
    MLP_PAGE_MARKOV_SCHEMA,
    MlpPageMarkov,
    MlpPageMarkovError,
)


class MlpPageMarkovTests(unittest.TestCase):
    def test_exact_prefill_opens_direct_routes_without_self_training(self) -> None:
        controller = MlpPageMarkov(
            None,
            n_layers=2,
            page_count=6,
            route_width=3,
            min_exact_rows=2,
        )
        cold = controller.prepare(0)
        self.assertFalse(cold.ready)
        self.assertEqual(cold.page_ids, ())

        controller.observe_exact_batch(
            0,
            torch.tensor([[[0, 1, 2], [1, 2, 3]]]),
            torch.tensor([[[9.0, 4.0, 1.0], [8.0, 3.0, 1.0]]]),
        )
        controller.prepare(1)
        controller.observe_exact_batch(
            1,
            [[3, 4, 5], [2, 4, 5]],
            [[7.0, 2.0, 1.0], [6.0, 2.0, 1.0]],
        )

        controller.compile_routes()
        compiled0 = controller.route(0, row_count=1)
        compiled0_again = controller.route(0, row_count=4)
        self.assertIs(compiled0, compiled0_again)
        self.assertTrue(compiled0.ready)

        route0 = controller.prepare(0)
        self.assertTrue(route0.ready)
        self.assertEqual(len(route0.page_ids), 3)
        before = controller.metrics()
        controller.advance_selected(0, route0.page_ids)
        route1 = controller.prepare(1)
        self.assertTrue(route1.ready)
        controller.advance_selected(1, route1.page_ids)
        after = controller.metrics()

        self.assertEqual(after["exact_rows"], before["exact_rows"])
        self.assertEqual(after["selected_advances"] - before["selected_advances"], 2)
        self.assertGreater(after["temporal_contexts"], 0)
        self.assertGreater(after["cross_contexts"], 0)
        self.assertEqual(after["exact_supported_layers"], 2)

    def test_cross_layer_learning_uses_the_causal_batch_boundary(self) -> None:
        controller = MlpPageMarkov(
            None,
            n_layers=2,
            page_count=8,
            route_width=2,
            min_exact_rows=1,
        )
        controller.prepare(0)
        controller.observe_exact_batch(0, [[0, 1], [2, 3], [4, 5]])
        controller.prepare(1)
        controller.observe_exact_batch(1, [[6, 7], [5, 4], [3, 2]])

        self.assertEqual(controller._cross[(1, 4)][3], 1)
        self.assertEqual(controller._cross[(1, 5)][2], 1)
        self.assertEqual(sum(map(len, controller._cross.values())), 2)

    def test_state_restarts_with_model_identity_and_exact_support(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pages.json"
            identity = {"q4_manifest_sha256": "a" * 64}
            first = MlpPageMarkov(
                path,
                n_layers=1,
                page_count=5,
                route_width=2,
                identity=identity,
                min_exact_rows=1,
            )
            first.prepare(0)
            first.observe_exact_batch(0, [[3, 1]], [[4.0, 2.0]])
            first.close()

            restored = MlpPageMarkov(
                path,
                n_layers=1,
                page_count=5,
                route_width=2,
                identity=identity,
                min_exact_rows=1,
            )
            prediction = restored.prepare(0)
            self.assertTrue(prediction.ready)
            self.assertEqual(set(prediction.page_ids), {1, 3})
            self.assertEqual(restored.metrics()["exact_rows"], 1)
            restored.close()

            with self.assertRaisesRegex(MlpPageMarkovError, "load"):
                MlpPageMarkov(
                    path,
                    n_layers=1,
                    page_count=5,
                    route_width=2,
                    identity={"q4_manifest_sha256": "b" * 64},
                    min_exact_rows=1,
                )

    def test_tamper_and_nested_state_corruption_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pages.json"
            controller = MlpPageMarkov(
                path,
                n_layers=1,
                page_count=10,
                route_width=1,
                min_exact_rows=1,
            )
            controller.prepare(0)
            controller.observe_exact_batch(0, [4])
            controller.close()

            document = json.loads(path.read_text())
            self.assertEqual(document["schema"], MLP_PAGE_MARKOV_SCHEMA)
            document["body"]["last_routes"] = [[99]]
            path.write_text(
                json.dumps(
                    document,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            with self.assertRaisesRegex(MlpPageMarkovError, "load"):
                MlpPageMarkov(
                    path,
                    n_layers=1,
                    page_count=10,
                    route_width=1,
                    min_exact_rows=1,
                )

    def test_space_saving_counters_remain_hard_bounded(self) -> None:
        controller = MlpPageMarkov(
            None,
            n_layers=1,
            page_count=16,
            route_width=1,
            min_exact_rows=1,
        )
        controller.prepare(0)
        for page in range(16):
            controller.observe_exact_batch(0, [page])
        self.assertLessEqual(
            len(controller._marginal[(0, 0)]),
            controller.MAX_TARGETS_PER_TRANSITION,
        )
        self.assertGreater(controller.metrics()["counter_evictions"], 0)

    def test_prefetch_runs_only_for_ready_routes_and_errors_propagate(self) -> None:
        calls: list[tuple[int, tuple[int, ...]]] = []

        def prefetch(layer: int, pages: tuple[int, ...]) -> bool:
            calls.append((layer, pages))
            return True

        controller = MlpPageMarkov(
            None,
            n_layers=1,
            page_count=4,
            route_width=2,
            min_exact_rows=1,
            prefetch=prefetch,
        )
        controller.prepare(0)
        self.assertEqual(calls, [])
        controller.observe_exact_batch(0, [2, 0])
        prediction = controller.prepare(0)
        self.assertTrue(prediction.ready)
        self.assertEqual(calls, [(0, prediction.page_ids)])
        self.assertEqual(controller.metrics()["prefetch_successes"], 1)

        failing = MlpPageMarkov(
            None,
            n_layers=1,
            page_count=4,
            route_width=1,
            min_exact_rows=1,
            prefetch=lambda _layer, _pages: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        failing.prepare(0)
        failing.observe_exact_batch(0, [1])
        with self.assertRaisesRegex(RuntimeError, "boom"):
            failing.prepare(0)

    def test_invalid_routes_scores_and_closed_use_are_rejected(self) -> None:
        controller = MlpPageMarkov(
            None,
            n_layers=1,
            page_count=4,
            route_width=2,
        )
        for route in ([0], [0, 0], [0, 4], [False, 1]):
            with self.subTest(route=route):
                with self.assertRaises(ValueError):
                    controller.observe_exact_batch(0, route)
        for scores in ([1.0], [1.0, float("nan")], [1.0, -1.0]):
            with self.subTest(scores=scores):
                with self.assertRaises(ValueError):
                    controller.observe_exact_batch(0, [0, 1], scores)
        controller.close()
        with self.assertRaises(MlpPageMarkovError):
            controller.prepare(0)

    def test_continuation_transaction_rolls_back_or_commits_route_mutations(self) -> None:
        controller = MlpPageMarkov(
            None,
            n_layers=1,
            page_count=4,
            route_width=2,
            min_exact_rows=1,
        )
        controller.prepare(0)
        controller.observe_exact_batch(0, [0, 1])
        baseline = controller.metrics()

        controller.begin_transaction()
        predicted = controller.prepare(0)
        controller.advance_selected(0, predicted.page_ids)
        self.assertEqual(controller.metrics()["selected_advances"], 1)
        controller.rollback_transaction()
        self.assertEqual(controller.metrics(), baseline)

        controller.begin_transaction()
        predicted = controller.prepare(0)
        controller.advance_selected(0, predicted.page_ids)
        controller.commit_transaction()
        self.assertEqual(controller.metrics()["selected_advances"], 1)

        controller.compile_routes()
        compiled = controller._compiled[0]
        controller.begin_transaction()
        first_wave = controller.route(0, row_count=2)
        controller.advance_selected(0, first_wave.page_ids, row_count=2)
        second_wave = controller.route(0, row_count=1)
        controller.advance_selected(0, second_wave.page_ids, row_count=1)
        controller.commit_transaction(accepted_rows=1)
        self.assertIs(controller._compiled[0], compiled)

        controller.begin_transaction()
        with self.assertRaisesRegex(MlpPageMarkovError, "active"):
            controller.flush()
        controller.rollback_transaction()

        cold = MlpPageMarkov(
            None,
            n_layers=1,
            page_count=4,
            route_width=1,
            min_exact_rows=1,
        )
        cold.begin_transaction()
        cold.prepare(0)
        cold.observe_exact_batch(0, [[0], [1], [2]])
        cold.commit_transaction(accepted_rows=1)
        self.assertEqual(cold.metrics()["exact_rows"], 1)
        self.assertEqual(cold._marginal[(0, 0)], {0: 1})


if __name__ == "__main__":
    unittest.main()
