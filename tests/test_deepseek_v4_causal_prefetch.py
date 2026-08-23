from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from immer.knowledge.livecausal import LiveGraph
from immer.runtimes.deepseek_v4.causal_prefetch import (
    CausalExpertTransitionController,
    CausalPrefetchError,
    CheckpointIdentity,
    CheckpointMismatchError,
    RouteState,
)


def _identity(seed: str = "a") -> CheckpointIdentity:
    return CheckpointIdentity(
        repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
        revision=seed * 40,
        inventory_fingerprint=seed * 64,
    )


class CausalPrefetchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.graph = LiveGraph(Path(self.temporary.name) / "causal")
        self.identity = _identity()
        self.controller = CausalExpertTransitionController(
            self.graph, self.identity, n_routed_experts=16
        )
        self.feature = "f" * 64

    def state(self, layer: int, experts: list[int]) -> RouteState:
        return self.controller.route_state(
            layer=layer,
            selected_expert_ids=experts,
            prompt_feature_digest=self.feature,
        )

    def test_checkpoint_and_route_keys_are_canonical_and_inventory_bound(self) -> None:
        first = self.state(3, [4, 2, 4, 1])
        reordered = self.state(3, [1, 4, 2, 4])
        different_inventory = CausalExpertTransitionController(
            self.graph,
            CheckpointIdentity(
                self.identity.repo_id,
                self.identity.revision,
                "b" * 64,
            ),
            n_routed_experts=16,
        ).route_state(
            layer=3,
            selected_expert_ids=[4, 1, 4, 2],
            prompt_feature_digest=self.feature,
        )

        self.assertEqual(first.expert_counts, ((1, 1), (2, 1), (4, 2)))
        self.assertEqual(first.key, reordered.key)
        self.assertNotEqual(first.key, different_inventory.key)
        self.assertNotEqual(first.checkpoint.key, different_inventory.checkpoint.key)

    def test_ranking_is_count_then_cost_then_expert_id(self) -> None:
        source = self.state(5, [1, 1, 6])
        self.controller.observe_transition(
            source,
            self.state(6, [2, 2, 3, 4]),
            observation_id="request-1:layer-5",
            provenance={"trace": "first", "router": "official"},
        )
        self.controller.observe_transition(
            source,
            self.state(6, [3, 3, 4, 2]),
            observation_id="request-2:layer-5",
            provenance={"trace": "second", "router": "official"},
        )

        prediction = self.controller.predict(
            source,
            top_k=3,
            expert_costs={2: 7, 3: 5, 4: 0},
        )

        self.assertEqual(prediction.total, 8)
        self.assertEqual(prediction.selection_total, 8)
        self.assertEqual(prediction.observation_count, 2)
        self.assertEqual(prediction.ranking_mode, "selection_mass")
        self.assertEqual([row.expert_id for row in prediction.candidates], [3, 2, 4])
        self.assertEqual(
            [(row.support, row.total) for row in prediction.candidates],
            [(3, 8), (3, 8), (2, 8)],
        )
        self.assertEqual(prediction.distribution, prediction.candidates)
        self.assertEqual(prediction.candidates[0].empirical_rate, 3 / 8)
        self.assertEqual(prediction.candidates[0].selection_rate, 3 / 8)
        self.assertEqual(prediction.candidates[0].presence_support, 2)
        self.assertEqual(prediction.candidates[0].empirical_presence_rate, 1.0)

        narrow = self.controller.predict(source, top_k=1)
        self.assertEqual(len(narrow.candidates), 1)
        self.assertEqual(len(narrow.distribution), 3)
        self.assertEqual([row.expert_id for row in narrow.distribution], [2, 3, 4])

    def test_observation_id_replay_is_persistently_idempotent(self) -> None:
        source = self.state(0, [1, 2])
        target = self.state(1, [3, 3, 4])
        first = self.controller.observe_transition(
            source,
            target,
            observation_id="trace-7:0-1",
            provenance={"run": "seven"},
        )
        segments = self.graph.store.segments()

        remounted = CausalExpertTransitionController(
            LiveGraph(Path(self.temporary.name) / "causal"),
            self.identity,
            n_routed_experts=16,
        )
        replay = remounted.observe_transition(
            source,
            target,
            observation_id="trace-7:0-1",
            provenance={"run": "seven"},
        )

        self.assertTrue(first.appended)
        self.assertFalse(replay.appended)
        self.assertEqual(first.segment_sha256, replay.segment_sha256)
        self.assertEqual(self.graph.store.segments(), segments)
        with self.assertRaisesRegex(CausalPrefetchError, "already bound"):
            remounted.observe_transition(
                source,
                target,
                observation_id="trace-7:0-1",
                provenance={"run": "changed"},
            )
        self.assertEqual(self.graph.store.segments(), segments)

    def test_distinct_evidence_aggregates_without_wallclock_fields(self) -> None:
        source = self.state(2, [1, 2])
        for observation, experts in (("a", [5, 5]), ("b", [5, 6, 6])):
            self.controller.observe_transition(
                source,
                self.state(3, experts),
                observation_id=observation,
                provenance={"source": "unit-test"},
            )

        prediction = self.controller.predict(source, top_k=2)
        records = [record for _sha, _index, record in self.graph.store.iter_records()]

        self.assertEqual(prediction.total, 5)
        self.assertEqual(
            [(row.expert_id, row.support) for row in prediction.candidates],
            [(5, 3), (6, 2)],
        )
        self.assertTrue(all("timestamp" not in record for record in records))
        self.assertTrue(all("provenance" in record for record in records))

    def test_checkpoint_identity_isolated_and_mismatch_fails_before_graph_calls(
        self,
    ) -> None:
        source = self.state(7, [1, 2])
        self.controller.observe_transition(
            source,
            self.state(8, [3]),
            observation_id="identity-a",
        )
        other = CausalExpertTransitionController(
            self.graph, _identity("b"), n_routed_experts=16
        )
        other_source = other.route_state(
            layer=7,
            selected_expert_ids=[1, 2],
            prompt_feature_digest=self.feature,
        )

        self.assertEqual(other.predict(other_source, top_k=4).candidates, ())
        prior_segments = self.graph.store.segments()
        with mock.patch.object(
            self.graph, "query_base", wraps=self.graph.query_base
        ) as query:
            with self.assertRaises(CheckpointMismatchError):
                other.predict(source, top_k=1)
            query.assert_not_called()
        with mock.patch.object(
            self.graph, "append_segment", wraps=self.graph.append_segment
        ) as append:
            with self.assertRaises(CheckpointMismatchError):
                other.observe_transition(
                    source,
                    self.state(8, [4]),
                    observation_id="wrong-checkpoint",
                )
            append.assert_not_called()
        self.assertEqual(self.graph.store.segments(), prior_segments)

    def test_base_fanout_is_complete_when_multihop_query_truncates(self) -> None:
        self.graph.default_node_budget = 1
        source = self.state(4, [1])
        self.controller.observe_transition(
            source,
            self.state(5, [2, 2, 3, 4]),
            observation_id="bounded-query",
        )

        _bounded, truncated = self.graph.query(source.key)
        self.assertTrue(truncated)
        prediction = self.controller.predict(source, top_k=2)

        self.assertEqual(prediction.total, 4)
        self.assertEqual(
            [(row.expert_id, row.support) for row in prediction.distribution],
            [(2, 2), (3, 1), (4, 1)],
        )
        self.assertEqual(
            [row.expert_id for row in prediction.candidates],
            [2, 3],
        )
        self.assertEqual(prediction.observation_count, 1)

    def test_presence_ranking_does_not_confuse_repeated_token_mass_with_presence(
        self,
    ) -> None:
        source = self.state(8, [1, 1])
        self.controller.observe_transition(
            source,
            self.state(9, [2] * 8 + [3]),
            observation_id="presence:first",
        )
        self.controller.observe_transition(
            source,
            self.state(9, [3]),
            observation_id="presence:second",
        )

        mass = self.controller.predict(source, top_k=16)
        presence = self.controller.predict(
            source,
            top_k=16,
            ranking_mode="presence_probability",
            presence_alpha=2,
            presence_beta=3,
        )

        self.assertEqual([row.expert_id for row in mass.distribution], [2, 3])
        self.assertEqual([row.expert_id for row in presence.distribution], [3, 2])
        self.assertEqual(len(presence.candidates), 2)
        by_expert = {row.expert_id: row for row in presence.distribution}
        self.assertEqual(by_expert[2].selection_support, 8)
        self.assertEqual(by_expert[2].presence_support, 1)
        self.assertEqual(by_expert[3].selection_support, 2)
        self.assertEqual(by_expert[3].presence_support, 2)
        self.assertEqual(by_expert[2].selection_total, 10)
        self.assertEqual(by_expert[2].observation_count, 2)
        self.assertAlmostEqual(by_expert[2].presence_probability, 3 / 7)
        self.assertAlmostEqual(by_expert[3].presence_probability, 4 / 7)

        different_prior = self.controller.predict(
            source,
            top_k=16,
            ranking_mode="presence_probability",
            presence_alpha=4,
            presence_beta=6,
        )
        different = {row.expert_id: row for row in different_prior.distribution}
        self.assertAlmostEqual(different[2].presence_probability, 5 / 12)
        self.assertAlmostEqual(different[3].presence_probability, 6 / 12)

    def test_presence_ties_use_cost_then_expert_id_deterministically(self) -> None:
        source = self.state(13, [1])
        self.controller.observe_transition(
            source,
            self.state(14, [2, 3, 4]),
            observation_id="presence-tie",
        )

        prediction = self.controller.predict(
            source,
            top_k=2,
            expert_costs={2: 5, 3: 1, 4: 1},
            ranking_mode="presence_probability",
        )

        self.assertEqual([row.expert_id for row in prediction.candidates], [3, 4])
        self.assertEqual(
            [row.expert_id for row in prediction.distribution],
            [3, 4, 2],
        )

    def test_presence_prior_and_ranking_mode_are_validated(self) -> None:
        source = self.state(1, [1])
        for keyword, value in (
            ("presence_alpha", 0),
            ("presence_alpha", float("nan")),
            ("presence_beta", -1),
            ("presence_beta", True),
        ):
            with self.subTest(keyword=keyword, value=value):
                with self.assertRaisesRegex(CausalPrefetchError, keyword):
                    self.controller.predict(source, top_k=1, **{keyword: value})
        with self.assertRaisesRegex(CausalPrefetchError, "ranking_mode"):
            self.controller.predict(source, top_k=1, ranking_mode="thumb-in-mouth")

    def test_legacy_transition_schema_fails_clearly(self) -> None:
        source = self.state(15, [1])
        self.graph.append_segment(
            [
                {
                    "outcome_key": "legacy:expert",
                    "record_type": "expert_transition",
                    "schema": "deepseek-v4-expert-transition-v1",
                    "trigger_key": source.key,
                }
            ]
        )

        with self.assertRaisesRegex(CausalPrefetchError, "incompatible.*schema"):
            self.controller.predict(source, top_k=1)

    def test_fixed_history_observes_only_consecutive_routes(self) -> None:
        first = self.state(10, [1, 2])
        second = self.state(11, [3, 3])
        third = self.state(12, [4])

        self.assertIsNone(self.controller.push_route(first))
        receipt = self.controller.push_route(
            second,
            observation_id="history:10-11",
            provenance={"path": "controller"},
        )
        self.assertIsNotNone(receipt)
        self.controller.push_route(third, observation_id="history:11-12")

        self.assertEqual(self.controller.history, (second, third))
        prediction = self.controller.predict(first, top_k=1)
        self.assertEqual(prediction.candidates[0].expert_id, 3)
        with self.assertRaisesRegex(CausalPrefetchError, "explicit observation_id"):
            self.controller.push_route(self.state(13, [5]))
        self.assertEqual(self.controller.history, (second, third))

    def test_malformed_expert_ids_and_mutable_checkpoint_are_rejected(self) -> None:
        for invalid in (-1, True, "1", 16):
            with self.subTest(invalid=invalid):
                with self.assertRaises(CausalPrefetchError):
                    self.controller.route_state(
                        layer=0,
                        selected_expert_ids=[invalid],  # type: ignore[list-item]
                    )
        with self.assertRaises(CausalPrefetchError):
            RouteState(
                checkpoint=self.identity,
                layer=0,
                expert_counts=((2, 1), (1, 1)),
            )
        with self.assertRaisesRegex(CausalPrefetchError, "pinned"):
            CheckpointIdentity("repo/model", "main", "c" * 64)


if __name__ == "__main__":
    unittest.main()
