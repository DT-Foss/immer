from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from immer.runtimes.qwen3_8.mlp_page_markov import (
    MLP_PAGE_MARKOV_POLICY,
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
        self.assertIsNot(compiled0, compiled0_again)
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

    def test_adaptive_width_actions_and_exact_energy_targets(self) -> None:
        official = MlpPageMarkov(
            None,
            n_layers=1,
            page_count=272,
            route_width=192,
        )
        self.assertEqual(official.width_actions, (96, 128, 160, 192))
        self.assertEqual(official.metrics()["energy_coverage"], 0.995)

        controller = MlpPageMarkov(
            None,
            n_layers=1,
            page_count=16,
            route_width=4,
            min_exact_rows=1,
        )
        controller.prepare(0)
        controller.observe_exact_batch(
            0,
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15],
            ],
            [
                [100.0, 99.0, 0.5, 0.25],
                [100.0, 98.5, 0.5, 0.25],
                [100.0, 70.0, 28.75, 0.5],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [200.0, 200.0, 200.0, 0.0],
        )

        self.assertEqual(controller.width_actions, (2, 3, 4))
        self.assertEqual(controller._width_marginal[0], {2: 2, 3: 1, 4: 1})
        self.assertEqual(controller._last_widths, [2])
        metrics = controller.metrics()
        self.assertEqual(metrics["energy_feedback_rows"], 4)
        self.assertEqual(metrics["width_marginal_contexts"], 1)

    def test_width_agents_follow_temporal_and_cross_layer_causal_context(self) -> None:
        controller = MlpPageMarkov(
            None,
            n_layers=2,
            page_count=8,
            route_width=4,
            min_exact_rows=1,
        )
        controller.prepare(0)
        controller.observe_exact_batch(
            0,
            [0, 1, 2, 3],
            [100.0, 99.0, 0.5, 0.25],
            [200.0],
        )
        controller.prepare(1)
        controller.observe_exact_batch(
            1,
            [4, 5, 6, 7],
            [100.0, 99.0, 0.5, 0.25],
            [200.0],
        )
        controller._width_temporal = {
            (0, 2): Counter({3: 7}),
            (0, 3): Counter({2: 7}),
        }
        controller._width_cross = {
            (1, 2): Counter({3: 7}),
            (1, 3): Counter({4: 7}),
        }
        controller._width_agent_logs[0] = [8.0, -8.0, -8.0]
        controller._width_agent_logs[1] = [-8.0, 8.0, -8.0]
        controller.compile_routes()

        first_layer0 = controller.route(0, row_count=1)
        self.assertEqual(dict(first_layer0.agent_widths)["temporal"], 3)
        self.assertEqual(first_layer0.width, 3)
        self.assertEqual(
            first_layer0.page_ids,
            first_layer0.full_page_ids[: first_layer0.width],
        )
        self.assertEqual(len(first_layer0.full_page_ids), 4)
        controller.advance_selected(0, first_layer0.page_ids)
        self.assertEqual(controller._last_routes[0], first_layer0.full_page_ids)
        self.assertEqual(controller._last_widths[0], 3)

        first_layer1 = controller.route(1, row_count=1)
        self.assertEqual(dict(first_layer1.agent_widths)["cross_layer"], 4)
        self.assertEqual(first_layer1.width, 4)
        controller.advance_selected(1, first_layer1.page_ids)

        second_layer0 = controller.route(0, row_count=1)
        self.assertEqual(dict(second_layer0.agent_widths)["temporal"], 2)
        self.assertEqual(second_layer0.width, 2)
        controller.advance_selected(0, second_layer0.page_ids)

        second_layer1 = controller.route(1, row_count=1)
        self.assertEqual(dict(second_layer1.agent_widths)["cross_layer"], 3)
        self.assertEqual(second_layer1.width, 3)
        controller.advance_selected(1, second_layer1.page_ids)
        metrics = controller.metrics()
        self.assertEqual(metrics["width_temporal_contexts"], 2)
        self.assertEqual(metrics["width_cross_contexts"], 2)
        self.assertEqual(metrics["adaptive_width_predictions"], 4)
        self.assertEqual(metrics["adaptive_width_pages_saved"], 4)

    def test_v5_routes_recompute_temporal_and_cross_layer_agents_each_wave(
        self,
    ) -> None:
        controller = MlpPageMarkov(
            None,
            n_layers=2,
            page_count=6,
            route_width=1,
            min_exact_rows=1,
        )
        controller.prepare(0)
        controller.observe_exact_batch(0, [0])
        controller.prepare(1)
        controller.observe_exact_batch(1, [2])

        controller._temporal[(0, 0)] = Counter({1: 7})
        controller._temporal[(0, 1)] = Counter({0: 7})
        controller._cross[(1, 1)] = Counter({3: 7})
        controller._cross[(1, 0)] = Counter({2: 7})
        controller._agent_logs[0] = [8.0, -8.0, -8.0, -8.0]
        controller._agent_logs[1] = [-8.0, 8.0, -8.0, -8.0]
        controller.compile_routes()
        compiled = tuple(controller._compiled)

        first_layer0 = controller.route(0, row_count=1)
        self.assertIsNot(first_layer0, compiled[0])
        self.assertEqual(first_layer0.page_ids, (1,))
        self.assertEqual(
            dict(first_layer0.agent_page_ids)["temporal"],
            (1,),
        )
        controller.advance_selected(0, first_layer0.page_ids)

        first_layer1 = controller.route(1, row_count=1)
        self.assertIsNot(first_layer1, compiled[1])
        self.assertEqual(first_layer1.page_ids, (3,))
        self.assertEqual(
            dict(first_layer1.agent_page_ids)["cross_layer"],
            (3,),
        )
        controller.advance_selected(1, first_layer1.page_ids)

        second_layer0 = controller.route(0, row_count=1)
        self.assertEqual(second_layer0.page_ids, (0,))
        self.assertEqual(
            dict(second_layer0.agent_page_ids)["temporal"],
            (0,),
        )
        controller.advance_selected(0, second_layer0.page_ids)

        second_layer1 = controller.route(1, row_count=1)
        self.assertEqual(second_layer1.page_ids, (2,))
        self.assertEqual(
            dict(second_layer1.agent_page_ids)["cross_layer"],
            (2,),
        )
        self.assertEqual(controller.metrics()["dynamic_route_calls"], 4)

    def test_coactive_agent_is_ordered_bounded_and_changes_dynamic_route(
        self,
    ) -> None:
        controller = MlpPageMarkov(
            None,
            n_layers=1,
            page_count=8,
            route_width=3,
            min_exact_rows=1,
        )
        routes = (
            *((0, 2, 4),) * 4,
            *((3, 1, 5),) * 2,
            *((6, 1, 5),) * 2,
        )
        for route in routes:
            controller.prepare(0)
            controller.observe_exact_batch(0, route)

        self.assertEqual(controller._coactive[(0, 0)][2], 4)
        self.assertEqual(controller._coactive[(0, 0)][4], 4)
        self.assertEqual(controller._coactive[(0, 2)][4], 4)
        self.assertNotIn((0, 4), controller._coactive)

        controller._agent_logs[0] = [-8.0, -8.0, -8.0, 8.0]
        controller.compile_routes()
        compiled = controller._compiled[0]
        self.assertIsNotNone(compiled)
        self.assertEqual(compiled.page_ids, (0, 1, 4))

        controller._agent_logs[0] = [-8.0, -8.0, 8.0, -8.0]
        dynamic = controller.route(0, row_count=1)
        self.assertEqual(
            dict(dynamic.agent_page_ids)["coactive"],
            (0, 2, 4),
        )
        self.assertEqual(dynamic.page_ids, (0, 2, 4))
        self.assertNotEqual(dynamic.page_ids, compiled.page_ids)
        self.assertGreater(controller.metrics()["dynamic_route_changes"], 0)

        bounded = MlpPageMarkov(
            None,
            n_layers=1,
            page_count=20,
            route_width=2,
            min_exact_rows=1,
        )
        for target in range(1, 20):
            bounded.prepare(0)
            bounded.observe_exact_batch(0, [0, target])
        self.assertLessEqual(
            len(bounded._coactive[(0, 0)]),
            bounded.MAX_TARGETS_PER_TRANSITION,
        )
        self.assertEqual(bounded.metrics()["coactive_updates"], 19)
        self.assertGreater(bounded.metrics()["counter_evictions"], 0)

    def test_cross_layer_learning_uses_all_aligned_causal_batch_rows(self) -> None:
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

        self.assertEqual(
            {
                key: dict(counter)
                for key, counter in controller._cross.items()
            },
            {
                (1, 0): {6: 1},
                (1, 1): {7: 1},
                (1, 2): {5: 1},
                (1, 3): {4: 1},
                (1, 4): {3: 1},
                (1, 5): {2: 1},
            },
        )

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

    def test_adaptive_width_state_persists_and_restarts_with_exact_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "adaptive-pages.json"
            identity = {"q4_manifest_sha256": "c" * 64}
            first = MlpPageMarkov(
                path,
                n_layers=1,
                page_count=8,
                route_width=4,
                identity=identity,
                min_exact_rows=1,
            )
            first.prepare(0)
            first.observe_exact_batch(
                0,
                [3, 1, 7, 5],
                [100.0, 99.0, 0.5, 0.25],
                [200.0],
            )
            first.close()

            restored = MlpPageMarkov(
                path,
                n_layers=1,
                page_count=8,
                route_width=4,
                identity=identity,
                min_exact_rows=1,
            )
            prediction = restored.prepare(0)
            self.assertTrue(prediction.ready)
            self.assertEqual(prediction.width, 2)
            self.assertEqual(prediction.page_ids, prediction.full_page_ids[:2])
            self.assertEqual(len(prediction.full_page_ids), 4)
            self.assertEqual(restored._last_widths, [2])
            self.assertEqual(restored._width_marginal[0], {2: 1})
            metrics = restored.metrics()
            self.assertEqual(metrics["energy_feedback_rows"], 1)
            self.assertEqual(metrics["width_marginal_contexts"], 1)
            restored.advance_selected(0, prediction.page_ids)
            restored.close()

            reopened = MlpPageMarkov(
                path,
                n_layers=1,
                page_count=8,
                route_width=4,
                identity=identity,
                min_exact_rows=1,
            )
            self.assertEqual(reopened._last_widths, [2])
            self.assertEqual(reopened._width_marginal[0], {2: 1})
            self.assertEqual(reopened.metrics()["adaptive_width_predictions"], 1)
            reopened.close()

    def test_v3_state_migrates_to_v5_without_losing_learned_state(self) -> None:
        self.assertEqual(
            MLP_PAGE_MARKOV_SCHEMA,
            "immer.qwen3.8-mlp-page-markov/v5",
        )
        self.assertEqual(
            MLP_PAGE_MARKOV_POLICY,
            "dynamic-page-transitions+coactivation+adaptive-width+fixed-share/v5",
        )
        legacy_metrics = {
            "agent_feedback": 18,
            "counter_evictions": 2,
            "exact_batches": 3,
            "exact_rows": 9,
            "fallback_predictions": 4,
            "position_hits": 5,
            "prediction_calls": 6,
            "predicted_pages": 12,
            "prefetch_calls": 0,
            "prefetch_pages": 0,
            "prefetch_successes": 0,
            "ready_predictions": 2,
            "route_jaccard_count": 2,
            "route_jaccard_sum_ppm": 1_250_000,
            "selected_advances": 3,
            "session_resets": 1,
        }
        identity = {"fixture": "persisted-v3"}
        neutral_coactive_log = (0.25 - 0.5 + 1.5) / 3
        body = {
            "agent_hits": [[1, 2, 3]],
            "agent_logs": [
                [
                    float.hex(0.25),
                    float.hex(-0.5),
                    float.hex(1.5),
                ]
            ],
            "agent_observations": [[4, 5, 6]],
            "config": {
                "identity": identity,
                "min_exact_rows": 1,
                "n_layers": 1,
                "page_count": 6,
                "policy": "shared-page-transitions+fixed-share/v3",
                "route_width": 2,
            },
            "cross": [[0, 3, [[4, 19]]]],
            "exact_support": [9],
            "last_routes": [[2, 4]],
            "marginal": [
                [0, 0, [[2, 11]]],
                [0, 1, [[4, 13]]],
            ],
            "metrics": legacy_metrics,
            "temporal": [[0, 1, [[2, 17]]]],
        }

        def canonical(value: object) -> bytes:
            return json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pages-v3.json"
            path.write_bytes(
                canonical(
                    {
                        "body": body,
                        "schema": "immer.qwen3.8-mlp-page-markov/v3",
                        "sha256": hashlib.sha256(canonical(body)).hexdigest(),
                    }
                )
            )

            restored = MlpPageMarkov(
                path,
                n_layers=1,
                page_count=6,
                route_width=2,
                identity=identity,
                min_exact_rows=1,
            )
            self.assertEqual(restored._coactive, {})
            self.assertEqual(
                restored._agent_logs,
                [[0.25, -0.5, neutral_coactive_log, 1.5]],
            )
            self.assertEqual(restored._agent_observations, [[4, 5, 0, 6]])
            self.assertEqual(restored._agent_hits, [[1, 2, 0, 3]])
            self.assertEqual(restored._marginal[(0, 0)], {2: 11})
            self.assertEqual(restored._marginal[(0, 1)], {4: 13})
            self.assertEqual(restored._temporal[(0, 1)], {2: 17})
            self.assertEqual(restored._cross[(0, 3)], {4: 19})
            self.assertEqual(
                restored._metrics,
                {
                    **legacy_metrics,
                    "coactive_updates": 0,
                    "dynamic_route_calls": 0,
                    "dynamic_route_changes": 0,
                    "adaptive_width_pages_saved": 0,
                    "adaptive_width_predictions": 0,
                    "energy_feedback_rows": 0,
                    "width_agent_feedback": 0,
                },
            )
            self.assertEqual(
                restored.snapshot_identity()["schema"],
                MLP_PAGE_MARKOV_SCHEMA,
            )
            self.assertEqual(
                restored.snapshot_identity()["policy"],
                MLP_PAGE_MARKOV_POLICY,
            )
            self.assertEqual(restored.snapshot_identity()["width_actions"], [1, 2])
            self.assertEqual(restored._last_widths, [2])
            restored.close()

            untouched = json.loads(path.read_text())
            self.assertEqual(
                untouched["schema"],
                "immer.qwen3.8-mlp-page-markov/v3",
            )

            migrating = MlpPageMarkov(
                path,
                n_layers=1,
                page_count=6,
                route_width=2,
                identity=identity,
                min_exact_rows=1,
            )
            prediction = migrating.prepare(0)
            migrating.advance_selected(0, prediction.page_ids)
            migrating.close()

            migrated = json.loads(path.read_text())
            self.assertEqual(migrated["schema"], MLP_PAGE_MARKOV_SCHEMA)
            self.assertEqual(
                migrated["body"]["config"]["policy"],
                MLP_PAGE_MARKOV_POLICY,
            )
            self.assertEqual(migrated["body"]["coactive"], [])
            self.assertEqual(
                migrated["body"]["agent_observations"],
                [[4, 5, 0, 6]],
            )
            self.assertEqual(
                migrated["body"]["metrics"],
                {
                    **legacy_metrics,
                    "coactive_updates": 0,
                    "dynamic_route_calls": 0,
                    "dynamic_route_changes": 0,
                    "adaptive_width_pages_saved": 0,
                    "adaptive_width_predictions": 1,
                    "energy_feedback_rows": 0,
                    "width_agent_feedback": 0,
                    "prediction_calls": legacy_metrics["prediction_calls"] + 1,
                    "predicted_pages": legacy_metrics["predicted_pages"] + 2,
                    "ready_predictions": legacy_metrics["ready_predictions"] + 1,
                    "selected_advances": legacy_metrics["selected_advances"] + 1,
                },
            )

            reopened = MlpPageMarkov(
                path,
                n_layers=1,
                page_count=6,
                route_width=2,
                identity=identity,
                min_exact_rows=1,
            )
            self.assertEqual(reopened._coactive, {})
            self.assertEqual(reopened._temporal[(0, 1)], {2: 17})
            self.assertEqual(reopened._cross[(0, 3)], {4: 19})
            reopened.close()

    def test_v4_state_migrates_to_v5_on_the_next_persistent_update(self) -> None:
        def canonical(value: object) -> bytes:
            return json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")

        identity = {"fixture": "persisted-v4"}
        seed = MlpPageMarkov(
            None,
            n_layers=1,
            page_count=6,
            route_width=2,
            identity=identity,
            min_exact_rows=1,
        )
        seed.prepare(0)
        seed.observe_exact_batch(0, [2, 4], [8.0, 3.0])
        body = seed._body()
        for key in (
            "last_widths",
            "width_agent_hits",
            "width_agent_logs",
            "width_agent_observations",
            "width_cross",
            "width_marginal",
            "width_temporal",
        ):
            body.pop(key)
        body["config"].pop("energy_coverage")
        body["config"].pop("width_actions")
        body["config"]["policy"] = (
            "dynamic-page-transitions+coactivation+fixed-share/v4"
        )
        for key in (
            "adaptive_width_pages_saved",
            "adaptive_width_predictions",
            "energy_feedback_rows",
            "width_agent_feedback",
        ):
            body["metrics"].pop(key)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pages-v4.json"
            path.write_bytes(
                canonical(
                    {
                        "body": body,
                        "schema": "immer.qwen3.8-mlp-page-markov/v4",
                        "sha256": hashlib.sha256(canonical(body)).hexdigest(),
                    }
                )
            )

            restored = MlpPageMarkov(
                path,
                n_layers=1,
                page_count=6,
                route_width=2,
                identity=identity,
                min_exact_rows=1,
            )
            self.assertEqual(restored._coactive[(0, 2)], {4: 1})
            self.assertEqual(restored._last_widths, [2])
            self.assertEqual(restored._width_marginal, {})
            self.assertEqual(restored.metrics()["energy_feedback_rows"], 0)
            restored.close()
            self.assertEqual(
                json.loads(path.read_text())["schema"],
                "immer.qwen3.8-mlp-page-markov/v4",
            )

            migrating = MlpPageMarkov(
                path,
                n_layers=1,
                page_count=6,
                route_width=2,
                identity=identity,
                min_exact_rows=1,
            )
            prediction = migrating.prepare(0)
            migrating.advance_selected(0, prediction.page_ids)
            migrating.close()

            migrated = json.loads(path.read_text())
            self.assertEqual(migrated["schema"], MLP_PAGE_MARKOV_SCHEMA)
            self.assertEqual(
                migrated["body"]["config"]["policy"],
                MLP_PAGE_MARKOV_POLICY,
            )
            self.assertEqual(migrated["body"]["last_widths"], [2])
            self.assertEqual(migrated["body"]["width_marginal"], [])

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
        with self.assertRaisesRegex(ValueError, "require ranked"):
            controller.observe_exact_batch(0, [0, 1], None, [3.0])
        for totals in (
            [True],
            [-1.0],
            [float("nan")],
            [float("inf")],
            [3.0, 4.0],
        ):
            with self.subTest(totals=totals):
                with self.assertRaisesRegex(ValueError, "total page energies"):
                    controller.observe_exact_batch(
                        0,
                        [0, 1],
                        [2.0, 1.0],
                        totals,
                    )
        with self.assertRaisesRegex(ValueError, "not descending"):
            controller.observe_exact_batch(
                0,
                [0, 1],
                [1.0, 2.0],
                [3.0],
            )
        with self.assertRaisesRegex(ValueError, "exceed total"):
            controller.observe_exact_batch(
                0,
                [0, 1],
                [2.0, 1.0],
                [2.5],
            )
        controller.close()
        with self.assertRaises(MlpPageMarkovError):
            controller.prepare(0)

    def test_transaction_rollback_restores_coactive_counters_and_dynamic_route(
        self,
    ) -> None:
        controller = MlpPageMarkov(
            None,
            n_layers=1,
            page_count=6,
            route_width=2,
            min_exact_rows=1,
        )
        for route in ((0, 1), (2, 3), (4, 5), (5, 4)):
            controller.prepare(0)
            controller.observe_exact_batch(0, route)
        controller.advance_selected(0, [0, 1])
        controller._agent_logs[0] = [8.0, -8.0, -8.0, -8.0]
        controller.compile_routes()

        baseline_route = controller.route(0, row_count=1)
        self.assertEqual(baseline_route.page_ids, (2, 3))
        baseline_coactive = {
            key: dict(counter)
            for key, counter in controller._coactive.items()
        }
        baseline_last_routes = list(controller._last_routes)
        baseline_wave_routes = dict(controller._wave_routes)
        baseline_pending = dict(controller._pending)
        baseline_compiled = tuple(controller._compiled)
        baseline_metrics = controller.metrics()

        controller.begin_transaction()
        controller.begin_exact_wave(0)
        controller.observe_exact_batch(0, [4, 5])
        self.assertNotEqual(
            {
                key: dict(counter)
                for key, counter in controller._coactive.items()
            },
            baseline_coactive,
        )
        changed_route = controller.route(0, row_count=1)
        self.assertEqual(changed_route.page_ids, (5, 4))
        self.assertNotEqual(changed_route.page_ids, baseline_route.page_ids)
        controller.advance_selected(0, changed_route.page_ids)
        self.assertNotEqual(controller._last_routes, baseline_last_routes)

        controller.rollback_transaction()
        self.assertEqual(
            {
                key: dict(counter)
                for key, counter in controller._coactive.items()
            },
            baseline_coactive,
        )
        self.assertEqual(controller._last_routes, baseline_last_routes)
        self.assertEqual(controller._wave_routes, baseline_wave_routes)
        self.assertEqual(controller._pending, baseline_pending)
        self.assertEqual(tuple(controller._compiled), baseline_compiled)
        self.assertEqual(controller.metrics(), baseline_metrics)

        restored_route = controller.route(0, row_count=1)
        self.assertEqual(restored_route.page_ids, baseline_route.page_ids)

    def test_transaction_rollback_restores_all_adaptive_width_state(self) -> None:
        controller = MlpPageMarkov(
            None,
            n_layers=1,
            page_count=8,
            route_width=4,
            min_exact_rows=1,
        )
        controller.prepare(0)
        controller.observe_exact_batch(
            0,
            [0, 1, 2, 3],
            [100.0, 98.5, 0.5, 0.25],
            [200.0],
        )
        baseline = {
            "agent_hits": [list(row) for row in controller._width_agent_hits],
            "agent_logs": [list(row) for row in controller._width_agent_logs],
            "agent_observations": [
                list(row) for row in controller._width_agent_observations
            ],
            "cross": {
                key: dict(counter) for key, counter in controller._width_cross.items()
            },
            "last_widths": list(controller._last_widths),
            "marginal": {
                key: dict(counter)
                for key, counter in controller._width_marginal.items()
            },
            "metrics": controller.metrics(),
            "temporal": {
                key: dict(counter)
                for key, counter in controller._width_temporal.items()
            },
            "wave_widths": dict(controller._wave_widths),
        }

        controller.begin_transaction()
        controller.begin_exact_wave(0)
        controller.observe_exact_batch(
            0,
            [4, 5, 6, 7],
            [100.0, 99.0, 0.5, 0.25],
            [200.0],
        )
        self.assertNotEqual(controller._last_widths, baseline["last_widths"])
        self.assertNotEqual(controller.metrics(), baseline["metrics"])
        controller.rollback_transaction()

        self.assertEqual(controller._width_agent_hits, baseline["agent_hits"])
        self.assertEqual(controller._width_agent_logs, baseline["agent_logs"])
        self.assertEqual(
            controller._width_agent_observations,
            baseline["agent_observations"],
        )
        self.assertEqual(
            {key: dict(counter) for key, counter in controller._width_cross.items()},
            baseline["cross"],
        )
        self.assertEqual(controller._last_widths, baseline["last_widths"])
        self.assertEqual(
            {key: dict(counter) for key, counter in controller._width_marginal.items()},
            baseline["marginal"],
        )
        self.assertEqual(controller.metrics(), baseline["metrics"])
        self.assertEqual(
            {key: dict(counter) for key, counter in controller._width_temporal.items()},
            baseline["temporal"],
        )
        self.assertEqual(controller._wave_widths, baseline["wave_widths"])

    def test_partial_commit_replays_prefetch_accounting_without_prefetch_io(
        self,
    ) -> None:
        calls: list[tuple[int, tuple[int, ...]]] = []

        def prefetch(layer: int, pages: tuple[int, ...]) -> bool:
            calls.append((layer, pages))
            return True

        controller = MlpPageMarkov(
            None,
            n_layers=1,
            page_count=4,
            route_width=1,
            min_exact_rows=1,
            prefetch=prefetch,
        )
        controller.prepare(0)
        controller.observe_exact_batch(0, [2])
        controller.compile_routes()

        controller.begin_transaction()
        first = controller.route(0, row_count=1)
        controller.advance_selected(0, first.page_ids)
        second = controller.route(0, row_count=1)
        controller.advance_selected(0, second.page_ids)
        self.assertEqual(len(calls), 2)

        controller.commit_transaction(accepted_rows=1)
        self.assertEqual(len(calls), 2)
        metrics = controller.metrics()
        self.assertEqual(metrics["prefetch_calls"], 1)
        self.assertEqual(metrics["prefetch_successes"], 1)
        self.assertEqual(metrics["selected_advances"], 1)

        prepared = MlpPageMarkov(
            None,
            n_layers=1,
            page_count=4,
            route_width=1,
            min_exact_rows=1,
        )
        prepared.prepare(0)
        prepared.observe_exact_batch(0, [1])
        baseline = prepared.metrics()
        prepared.begin_transaction()
        for _ in range(2):
            prediction = prepared.prepare(0)
            prepared.advance_selected(0, prediction.page_ids)
        prepared.commit_transaction(accepted_rows=1)
        replayed = prepared.metrics()
        self.assertEqual(
            replayed["dynamic_route_calls"],
            baseline["dynamic_route_calls"],
        )
        self.assertEqual(
            replayed["prediction_calls"],
            baseline["prediction_calls"] + 1,
        )
        self.assertEqual(
            replayed["selected_advances"],
            baseline["selected_advances"] + 1,
        )

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

    def test_partial_exact_transaction_replays_only_accepted_energy_totals(
        self,
    ) -> None:
        cold = MlpPageMarkov(
            None,
            n_layers=1,
            page_count=12,
            route_width=4,
            min_exact_rows=1,
        )
        cold.begin_transaction()
        cold.prepare(0)
        cold.observe_exact_batch(
            0,
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
            [
                [100.0, 99.0, 0.5, 0.25],
                [100.0, 98.5, 0.5, 0.25],
                [100.0, 70.0, 28.75, 0.5],
            ],
            [200.0, 200.0, 200.0],
        )
        cold.commit_transaction(accepted_rows=1)
        self.assertEqual(cold.metrics()["exact_rows"], 1)
        self.assertEqual(cold.metrics()["energy_feedback_rows"], 1)
        self.assertEqual(cold._marginal[(0, 0)], {0: 1})
        self.assertEqual(cold._width_marginal[0], {2: 1})
        self.assertEqual(cold._last_widths, [2])
        self.assertEqual(cold._wave_widths, {0: (2,)})


if __name__ == "__main__":
    unittest.main()
