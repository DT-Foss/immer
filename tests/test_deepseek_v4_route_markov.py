from __future__ import annotations

from collections import Counter
import copy
import json
import unittest

from immer.runtimes.deepseek_v4.route_markov import (
    LayerMarkovExpertPredictor,
    LayerTokenRoutes,
    PromptRouteObservation,
    RouteMarkovError,
    apply_target_layer_placebo,
    build_target_layer_placebo,
    evaluate_all_baselines,
    evaluate_k_sweep,
    split_prompt_observations,
)


def _observation(
    observation_id: str,
    layers: list[tuple[int, list[list[int]]]],
) -> PromptRouteObservation:
    return PromptRouteObservation(
        observation_id,
        tuple(LayerTokenRoutes(layer, rows) for layer, rows in layers),
    )


class RouteMarkovTests(unittest.TestCase):
    def test_counts_are_layer_specific_row_aligned_and_indicator_weighted(self) -> None:
        model = LayerMarkovExpertPredictor(n_experts=6)
        trace = _observation(
            "prompt-a",
            [
                (0, [[1, 1, 2], [], [3]]),
                (1, [[2, 4, 4], [], [1, 4]]),
                (2, [[5], [], [2, 2]]),
            ],
        )

        receipt = model.observe(trace)

        self.assertTrue(receipt.appended)
        # Repeated IDs inside a row remain one indicator in M_l[i,j].
        self.assertEqual(model.transition_count(0, 1, 2), 1)
        self.assertEqual(model.transition_count(0, 1, 4), 1)
        self.assertEqual(model.transition_count(0, 2, 4), 1)
        self.assertEqual(model.transition_count(0, 3, 1), 1)
        # The same coordinate in the next layer is an independent matrix.
        self.assertEqual(model.transition_count(1, 1, 2), 1)
        self.assertEqual(model.transition_count(0, 1, 5), 0)
        self.assertEqual(model.source_layers, (0, 1))

    def test_padding_pairs_are_ignored_but_one_sided_empty_rows_are_rejected(
        self,
    ) -> None:
        model = LayerMarkovExpertPredictor(n_experts=4)
        padding_only = _observation(
            "padding",
            [(3, [[], [1]]), (4, [[], [2]])],
        )
        model.observe(padding_only)
        self.assertEqual(model.transition_count(3, 1, 2), 1)

        misaligned = _observation(
            "misaligned",
            [(3, [[], [1]]), (4, [[2], [2]])],
        )
        with self.assertRaisesRegex(RouteMarkovError, "empty in both"):
            model.observe(misaligned)
        unequal = lambda: _observation(  # noqa: E731 - construction must raise.
            "unequal",
            [(0, [[1], [2]]), (1, [[2]])],
        )
        with self.assertRaisesRegex(RouteMarkovError, "row alignment"):
            unequal()

    def test_observation_replay_is_idempotent_and_conflict_is_detected(self) -> None:
        model = LayerMarkovExpertPredictor(n_experts=5)
        first = _observation("stable-id", [(0, [[1]]), (1, [[2]])])
        replay = _observation("stable-id", [(0, [[1]]), (1, [[2]])])

        appended = model.observe(first)
        before = model.snapshot_sha256
        deduplicated = model.observe(replay)

        self.assertTrue(appended.appended)
        self.assertFalse(deduplicated.appended)
        self.assertEqual(appended.payload_sha256, deduplicated.payload_sha256)
        self.assertEqual(model.snapshot_sha256, before)
        self.assertEqual(model.transition_count(0, 1, 2), 1)
        with self.assertRaisesRegex(RouteMarkovError, "different route evidence"):
            model.observe(_observation("stable-id", [(0, [[1]]), (1, [[3]])]))
        self.assertEqual(model.snapshot_sha256, before)

    def test_default_prediction_is_a_full_smoothed_256_expert_distribution(
        self,
    ) -> None:
        model = LayerMarkovExpertPredictor()

        distribution = model.predict_distribution(
            source_layer=7,
            current_row=[11, 11, 19],
            alpha=0.5,
        )

        self.assertEqual(len(distribution.scores), 256)
        self.assertEqual(len(distribution.ranking), 256)
        self.assertEqual(len(distribution.top_k(256)), 256)
        self.assertAlmostEqual(sum(distribution.scores), 1.0)
        self.assertTrue(all(score == 1 / 256 for score in distribution.scores))
        with self.assertRaisesRegex(RouteMarkovError, "exceeds"):
            distribution.top_k(257)

    def test_markov_marginal_and_current_id_baselines_are_distinct(self) -> None:
        model = LayerMarkovExpertPredictor(n_experts=5)
        model.observe(
            _observation(
                "train",
                [(0, [[0], [0], [3]]), (1, [[2], [2], [4]])],
            )
        )

        markov = model.predict_distribution(source_layer=0, current_row=[0], alpha=1)
        marginal = model.marginal_distribution(target_layer=1, alpha=1)
        passthrough = model.passthrough_distribution(source_layer=0, current_row=[0, 3])

        self.assertEqual(markov.ranking[0], 2)
        self.assertEqual(marginal.ranking[:2], (2, 4))
        self.assertEqual(passthrough.ranking[:2], (0, 3))
        self.assertEqual(len(markov.scores), 5)
        self.assertEqual(len(marginal.scores), 5)
        self.assertEqual(len(passthrough.scores), 5)

    def test_k_sweep_has_all_widths_and_exact_full_inventory_endpoint(self) -> None:
        model = LayerMarkovExpertPredictor(n_experts=4)
        model.observe(
            _observation(
                "train",
                [(0, [[0], [0]]), (1, [[2, 2, 1], [2]])],
            )
        )
        evaluation = evaluate_k_sweep(
            model,
            [_observation("test", [(0, [[0]]), (1, [[2, 2, 1]])])],
            alpha=1,
        )

        self.assertEqual([point.k for point in evaluation.curve], [1, 2, 3, 4])
        self.assertEqual(evaluation.evaluated_rows, 1)
        self.assertEqual(evaluation.curve[0].set_recall, 1 / 2)
        self.assertEqual(evaluation.curve[0].set_precision, 1.0)
        self.assertEqual(evaluation.curve[0].selection_mass_recall, 2 / 3)
        endpoint = evaluation.curve[-1]
        self.assertEqual(endpoint.set_recall, 1.0)
        self.assertEqual(endpoint.selection_mass_recall, 1.0)
        self.assertEqual(endpoint.set_precision, 1 / 2)
        self.assertEqual(endpoint.predicted_total, 4)
        self.assertEqual(len(evaluation.layers), 1)
        self.assertEqual(evaluation.layers[0].target_layer, 1)

    def test_all_baselines_share_the_same_whole_prompt_evaluation_rows(self) -> None:
        model = LayerMarkovExpertPredictor(n_experts=5)
        train = _observation(
            "train",
            [(0, [[0], [1]]), (1, [[2], [3]])],
        )
        test = _observation(
            "test",
            [(0, [[0], [1]]), (1, [[2], [3]])],
        )
        model.observe(train)

        results = evaluate_all_baselines(model, [test])

        self.assertEqual(set(results), {"markov", "marginal", "passthrough"})
        self.assertTrue(
            all(result.prompt_ids == ("test",) for result in results.values())
        )
        self.assertTrue(all(result.evaluated_rows == 2 for result in results.values()))
        self.assertTrue(all(len(result.curve) == 5 for result in results.values()))

    def test_split_is_order_independent_and_keeps_whole_prompts(self) -> None:
        prompts = [
            _observation(
                f"prompt-{index}",
                [(0, [[index % 4], [(index + 1) % 4]]), (1, [[2], [3]])],
            )
            for index in range(6)
        ]

        first = split_prompt_observations(prompts, test_fraction=1 / 3, seed=17)
        second = split_prompt_observations(
            list(reversed(prompts)), test_fraction=1 / 3, seed=17
        )

        first_train = {prompt.observation_id for prompt in first.train}
        first_test = {prompt.observation_id for prompt in first.test}
        self.assertEqual(
            first_train, {prompt.observation_id for prompt in second.train}
        )
        self.assertEqual(first_test, {prompt.observation_id for prompt in second.test})
        self.assertFalse(first_train & first_test)
        self.assertEqual(
            first_train | first_test, {prompt.observation_id for prompt in prompts}
        )
        self.assertEqual(len(first.test), 2)
        self.assertTrue(all(len(prompt.layers[0].rows) == 2 for prompt in first.test))

    def test_placebo_is_deterministic_and_preserves_rows_sizes_and_marginals(
        self,
    ) -> None:
        prompts = [
            _observation(
                f"prompt-{index}",
                [
                    (4, [[index % 5], []]),
                    (5, [[(index + 1) % 5, (index + 2) % 5], []]),
                ],
            )
            for index in range(5)
        ]

        first = build_target_layer_placebo(prompts, seed=29)
        reordered = build_target_layer_placebo(reversed(prompts), seed=29)

        self.assertEqual(first, reordered)
        self.assertEqual(first.sha256, reordered.sha256)
        originals = Counter(entry.original_target for entry in first.assignments)
        shuffled = Counter(entry.shuffled_target for entry in first.assignments)
        self.assertEqual(originals, shuffled)
        self.assertEqual(
            Counter(expert for row in originals.elements() for expert in row),
            Counter(expert for row in shuffled.elements() for expert in row),
        )
        self.assertTrue(
            all(
                len(entry.original_target) == len(entry.shuffled_target)
                and len(set(entry.original_target)) == len(set(entry.shuffled_target))
                for entry in first.assignments
            )
        )
        self.assertTrue(
            any(
                entry.original_target != entry.shuffled_target
                for entry in first.assignments
            )
        )

    def test_placebo_evaluation_is_reproducible_and_records_its_identity(self) -> None:
        model = LayerMarkovExpertPredictor(n_experts=5)
        prompts = [
            _observation(
                f"prompt-{index}",
                [(0, [[index]]), (1, [[(index + 1) % 5]])],
            )
            for index in range(5)
        ]
        model.fit(prompts)

        first = evaluate_k_sweep(model, prompts, placebo_seed=91)
        second = evaluate_k_sweep(model, reversed(prompts), placebo_seed=91)

        self.assertIsNotNone(first.placebo_sha256)
        self.assertEqual(first.placebo_sha256, second.placebo_sha256)
        self.assertEqual(first.curve, second.curve)
        self.assertEqual(first.prompt_ids, second.prompt_ids)

    def test_snapshot_is_sparse_canonical_and_has_stable_digest(self) -> None:
        model = LayerMarkovExpertPredictor(n_experts=16)
        trace = _observation(
            "persisted",
            [(7, [[1]]), (8, [[2, 3]])],
        )
        model.observe(trace)
        snapshot = model.snapshot()
        encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
        repeated = LayerMarkovExpertPredictor(n_experts=16)
        repeated.observe(trace)

        self.assertEqual(len(snapshot["layers"][0]["transitions"]), 2)
        self.assertEqual(json.loads(encoded), snapshot)
        self.assertEqual(model.snapshot_sha256, repeated.snapshot_sha256)

    def test_real_vs_placebo_training_uses_shuffled_targets_and_real_sources(
        self,
    ) -> None:
        rows = [[expert] for expert in range(6)]
        training = _observation(
            "train",
            [
                (0, copy.deepcopy(rows)),
                (1, copy.deepcopy(rows)),
                (2, [[(expert + 1) % 6] for expert in range(6)]),
            ],
        )
        held_out = _observation(
            "held-out",
            [(0, copy.deepcopy(rows)), (1, copy.deepcopy(rows))],
        )
        placebo = build_target_layer_placebo([training], seed=31)
        shuffled_training = apply_target_layer_placebo([training], placebo)
        real_model = LayerMarkovExpertPredictor(n_experts=6)
        placebo_model = LayerMarkovExpertPredictor(n_experts=6)
        real_model.observe(training)
        placebo_model.fit(shuffled_training)

        real = evaluate_k_sweep(real_model, [held_out])
        shuffled = evaluate_k_sweep(placebo_model, [held_out])

        self.assertEqual(real.curve[0].set_recall, 1.0)
        self.assertLess(shuffled.curve[0].set_recall, real.curve[0].set_recall)
        overridden = shuffled_training[0]
        self.assertEqual(overridden.layers, training.layers)
        self.assertTrue(overridden.target_overrides)
        for target_layer in (1, 2):
            self.assertEqual(
                real_model.marginal_distribution(target_layer=target_layer).scores,
                placebo_model.marginal_distribution(target_layer=target_layer).scores,
            )
            self.assertEqual(
                [len(row) for row in training.layers[target_layer].rows],
                [
                    len(row)
                    for row in next(
                        override
                        for override in overridden.target_overrides
                        if override.layer == target_layer
                    ).rows
                ],
            )
        layer_two_targets = {
            entry.row_index: entry.shuffled_target[0]
            for entry in placebo.assignments
            if entry.source_layer == 1
        }
        for row_index, original_source in enumerate(training.layers[1].rows):
            self.assertEqual(
                placebo_model.transition_count(
                    1,
                    original_source[0],
                    layer_two_targets[row_index],
                ),
                1,
            )

    def test_inputs_and_official_router_rows_are_never_mutated(self) -> None:
        raw_layers = [
            (0, [[0, 1], [], [2]]),
            (1, [[2, 3], [], [4]]),
        ]
        before = copy.deepcopy(raw_layers)
        observation = _observation("immutable", raw_layers)
        frozen_before = observation.as_record()
        model = LayerMarkovExpertPredictor(n_experts=5)

        model.observe(observation)
        evaluate_all_baselines(model, [observation], placebo_seed=7)

        self.assertEqual(raw_layers, before)
        self.assertEqual(observation.as_record(), frozen_before)


if __name__ == "__main__":
    unittest.main()
