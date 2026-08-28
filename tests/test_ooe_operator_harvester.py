from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from immer.runtimes.deepseek_v4.causal_weights import TensorRangePlan
from immer.runtimes.ooe.compute_crystals import (
    AFFINE_FLOAT64,
    MARKOV_FLOAT64,
    PERMUTATION,
    ComputeCrystalBank,
)
from immer.runtimes.ooe.compute_graph import (
    ComputeOperatorGraph,
    ComputeOperatorGraphConflictError,
)
from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.operator_harvester import (
    ContextualObservationBatch,
    ContextualOperatorObservation,
    ContextualTransitionReceipt,
    ContinuousOperatorHarvester,
    HarvesterConfig,
    HarvesterState,
    MAX_STATE_BYTES,
    OperatorHarvesterIntegrityError,
)
from immer.runtimes.qwen3_8.semantic_atlas import (
    GraphRevision,
    InterventionIdentity,
    MeasurementReceipt,
    ModelPin,
    NumericSummary,
    ProbeIdentity,
    RuntimeProvenance,
    SemanticWeightAtlas,
    WeightCoordinate,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


_CODE_REVISION = "6" * 40
_PLAN = TensorRangePlan(
    name="model.layers.18.mlp.gate_proj.weight",
    dtype="BF16",
    shape=(16, 4),
    shard="model-00007-of-00018.safetensors",
    absolute_offset=8192,
    length=128,
)
_PIN = ModelPin(
    repo_id="Qwen/Qwen3.8-27B",
    revision="0123456789abcdef",
    bundle_fingerprint=_hash("bundle"),
    bundle_manifest_sha256=_hash("bundle-manifest"),
    code_revision=_CODE_REVISION,
)
_RUNTIME = RuntimeProvenance(
    code_revision=_CODE_REVISION,
    source_manifest_sha256=_hash("sources"),
    dependency_manifest_sha256=_hash("dependencies"),
    runtime_configuration_sha256=_hash("runtime-config"),
    platform_sha256=_hash("platform"),
)
_COORDINATE = WeightCoordinate.from_plan(
    _PLAN,
    layer=18,
    module="model.layers.18.mlp.gate_proj",
    row_start=0,
    row_end=4,
)
_COORDINATE_ALT = WeightCoordinate.from_plan(
    _PLAN,
    layer=18,
    module="model.layers.18.mlp.gate_proj",
    row_start=4,
    row_end=8,
)
_WEIGHT_REVISION = GraphRevision(17, _hash("weight-rail"))
_EMITTER = _hash("o1-context-emitter/v1")
_FEATURE_SCHEMA = _hash("rmsnorm-post-attention-context/float64/v1")
_ACTION_SCHEMA = _hash("mlp-context-transition/float64/v1")


def _summary(value: float) -> NumericSummary:
    # Deliberately unrelated to the contextual arrays.  The harvester must never
    # inflate this scalar sufficient statistic into a hidden vector.
    return NumericSummary(
        metric="activation_scalar_only",
        count=2,
        total=2.0 * value,
        total_squares=2.0 * value * value,
        minimum=value,
        maximum=value,
    )


class _ReplayProvider:
    emitter_sha256 = _EMITTER

    def __init__(
        self, pages: tuple[tuple[ContextualOperatorObservation, ...], ...]
    ) -> None:
        self.pages = pages
        self.calls: list[tuple[str | None, int]] = []

    def poll(
        self, *, after_cursor: str | None, limit: int
    ) -> ContextualObservationBatch:
        self.calls.append((after_cursor, limit))
        index = 0 if after_cursor is None else int(after_cursor.split("-")[-1])
        if index >= len(self.pages):
            cursor = after_cursor or "page-0"
            return ContextualObservationBatch(cursor, (), True)
        observations = self.pages[index]
        if len(observations) > limit:
            observations = observations[:limit]
        next_cursor = f"page-{index + 1}"
        return ContextualObservationBatch(
            next_cursor,
            observations,
            index + 1 >= len(self.pages),
        )


class _ReplayAgainProvider:
    emitter_sha256 = _EMITTER

    def __init__(self, observations: tuple[ContextualOperatorObservation, ...]) -> None:
        self.observations = observations

    def poll(
        self, *, after_cursor: str | None, limit: int
    ) -> ContextualObservationBatch:
        return ContextualObservationBatch("page-2", self.observations[:limit], True)


class ContinuousOperatorHarvesterTests(unittest.TestCase):
    def test_persistent_bound_covers_complete_full_span_qwen_authority(self) -> None:
        self.assertEqual(MAX_STATE_BYTES, 256 * 1024 * 1024)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.root = root
        self.atlas = SemanticWeightAtlas(
            root / "atlas", model_pin=_PIN, tensor_plans=(_PLAN,)
        )
        self.bank = ComputeCrystalBank(root / "compute")
        self.graph = ComputeOperatorGraph(self.bank)

    def _measurement(
        self,
        index: int,
        *,
        mode: str = "native",
        append: bool = True,
        coordinate: WeightCoordinate = _COORDINATE,
    ) -> MeasurementReceipt:
        measurement = MeasurementReceipt(
            model_pin=_PIN,
            coordinate=coordinate,
            probe=ProbeIdentity(
                question_sha256=_hash(f"question-{index}-{mode}"),
                token_sha256=_hash(f"tokens-{index}-{mode}"),
                family_sha256=_hash("context-family"),
                label_source_sha256=_hash("label-free"),
            ),
            intervention=InterventionIdentity(
                mode=mode,
                configuration_sha256=_hash(f"intervention-{index}-{mode}"),
            ),
            observation_status="recorded",
            observed_semantic_label=None,
            hidden_sha256=_hash(f"hidden-{index}-{mode}"),
            activation_sha256=_hash(f"activation-{index}-{mode}"),
            logits_sha256=_hash(f"logits-{index}-{mode}"),
            state_sha256=_hash(f"state-{index}-{mode}"),
            access_trace_sha256=_hash(f"trace-{index}-{mode}"),
            evidence_sha256=_hash(f"evidence-{index}-{mode}"),
            weight_rail_revision=_WEIGHT_REVISION,
            atlas_head_revision=self.atlas.revision(),
            numeric_summaries=(_summary(1000.0 + index),),
            placebo_effects=(),
            runtime=_RUNTIME,
        )
        if append:
            self.atlas.append_measurement(measurement)
        return measurement

    def _observation(
        self,
        measurement: MeasurementReceipt,
        input_array: np.ndarray,
        output_array: np.ndarray,
        *,
        source: str,
        target: str,
        revision: GraphRevision | None = None,
        granularity: str = "operator",
        segment_start: int | None = None,
        segment_end: int | None = None,
    ) -> ContextualOperatorObservation:
        return ContextualOperatorObservation.capture(
            measurement,
            atlas_revision=self.atlas.revision() if revision is None else revision,
            emitter_sha256=_EMITTER,
            feature_schema_sha256=_FEATURE_SCHEMA,
            action_schema_sha256=_ACTION_SCHEMA,
            source_state=source,
            target_state=target,
            input_array=input_array.astype(np.float64),
            output_array=output_array.astype(np.float64),
            granularity=granularity,  # type: ignore[arg-type]
            segment_start=segment_start,
            segment_end=segment_end,
        )

    def _harvester(
        self,
        provider: object,
        *,
        state_name: str = "test-operator-harvester/v1",
    ) -> ContinuousOperatorHarvester:
        return ContinuousOperatorHarvester(
            atlas=self.atlas,
            provider=provider,  # type: ignore[arg-type]
            graph=self.graph,
            config=HarvesterConfig(
                minimum_observations=3,
                minimum_fit_rows=4,
                max_observations_per_step=64,
                max_samples_per_group=32,
                max_groups=16,
                max_recent_receipts=64,
                graph_cas_retries=4,
            ),
            state_name=state_name,
        )

    def test_affine_operator_and_segment_are_crystallized_from_context_not_summaries(
        self,
    ) -> None:
        measurements = tuple(self._measurement(index) for index in range(8))
        revision = self.atlas.revision()
        matrix = np.array([[2.0, -1.0], [0.5, 3.0]], dtype=np.float64)
        bias = np.array([0.25, -2.0], dtype=np.float64)
        observations: list[ContextualOperatorObservation] = []
        for index, measurement in enumerate(measurements[:4]):
            generator = np.random.default_rng(100 + index)
            x = generator.normal(size=(4, 2))
            y = x @ matrix.T + bias
            observations.append(
                self._observation(
                    measurement,
                    x,
                    y,
                    source="post-attention/full",
                    target="mlp-value/full",
                    revision=revision,
                )
            )
        for index, measurement in enumerate(measurements[4:]):
            x = np.array(
                [[index - 2.0], [index + 0.5], [3.0 - index]], dtype=np.float64
            )
            y = 3.0 * x - 2.0
            observations.append(
                self._observation(
                    measurement,
                    x,
                    y,
                    source="post-attention/segment-7",
                    target="mlp-value/segment-7",
                    revision=revision,
                    granularity="segment",
                    segment_start=7,
                    segment_end=8,
                )
            )

        result = self._harvester(_ReplayProvider((tuple(observations),))).step()

        affine = [
            promotion
            for promotion in result.promotions
            if promotion.candidate.operator_kind == AFFINE_FLOAT64
        ]
        self.assertEqual(len(affine), 2)
        self.assertEqual(result.accepted_observations, 8)
        self.assertEqual(len(self.graph.state().edges), 2)
        full = next(
            self.bank.restore_crystal(row.edge.crystal_sha256)
            for row in affine
            if row.edge.source_state == "post-attention/full"
        )
        segment = next(
            self.bank.restore_crystal(row.edge.crystal_sha256)
            for row in affine
            if row.edge.source_state == "post-attention/segment-7"
        )
        unseen = np.array([[7.0, -4.0], [-9.5, 0.125]], dtype=np.float64)
        np.testing.assert_allclose(full.apply(unseen), unseen @ matrix.T + bias)
        segment_unseen = np.array([[-100.0], [4.25]], dtype=np.float64)
        np.testing.assert_allclose(
            segment.apply(segment_unseen), 3 * segment_unseen - 2
        )
        self.assertEqual(
            segment.extensions["operator_harvester"]["granularity"], "segment"
        )
        self.assertNotIn("numeric_summaries", segment.extensions_json.decode("utf-8"))
        self.assertTrue(self.atlas.verify_or_raise())

    def test_exact_permutation_is_also_discovered_as_a_markov_kernel(self) -> None:
        measurements = tuple(self._measurement(index) for index in range(5))
        revision = self.atlas.revision()
        inputs = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.2, 0.3, 0.5],
            ],
            dtype=np.float64,
        )
        observations = tuple(
            self._observation(
                measurement,
                inputs,
                inputs[:, [2, 0, 1]],
                source="distribution/raw",
                target="distribution/routed",
                revision=revision,
            )
            for measurement in measurements
        )

        result = self._harvester(_ReplayProvider((observations,))).step()

        kinds = {row.candidate.operator_kind for row in result.promotions}
        self.assertEqual(kinds, {PERMUTATION, MARKOV_FLOAT64})
        crystals = {
            row.candidate.operator_kind: self.bank.restore_crystal(
                row.edge.crystal_sha256
            )
            for row in result.promotions
        }
        future = np.array([[0.11, 0.72, 0.17]], dtype=np.float64)
        expected = future[:, [2, 0, 1]]
        np.testing.assert_array_equal(crystals[PERMUTATION].apply(future), expected)
        np.testing.assert_allclose(
            crystals[MARKOV_FLOAT64].apply(future), expected, atol=1e-12, rtol=0
        )
        affine_stream = next(
            row for row in result.candidates if row.operator_kind == AFFINE_FLOAT64
        )
        self.assertEqual(affine_stream.status, "rejected")
        self.assertEqual(affine_stream.reason, "affine-fit-design-not-full-rank")

    def test_historical_heads_stream_while_placebo_and_inactive_are_rejected(
        self,
    ) -> None:
        native = self._measurement(0)
        stale_revision = self.atlas.revision()
        placebo = self._measurement(1, mode="placebo")
        inactive = self._measurement(2, append=False)
        current = self.atlas.revision()
        x = np.array([[1.0], [2.0]], dtype=np.float64)
        observations = (
            self._observation(
                placebo,
                x,
                2 * x,
                source="p0",
                target="p1",
                revision=current,
            ),
            self._observation(
                native,
                x,
                2 * x,
                source="s0",
                target="s1",
                revision=stale_revision,
            ),
            self._observation(
                inactive,
                x,
                2 * x,
                source="i0",
                target="i1",
                revision=current,
            ),
            self._observation(
                native,
                x,
                2 * x,
                source="foreign0",
                target="foreign1",
                revision=GraphRevision(999, _hash("foreign-atlas-head")),
            ),
        )

        result = self._harvester(_ReplayProvider((observations,))).step()

        self.assertEqual(result.accepted_observations, 1)
        self.assertEqual(
            {row.reason for row in result.rejections},
            {
                "placebo-measurement",
                "measurement-not-active-in-atlas",
                "atlas-revision-outside-history",
            },
        )
        self.assertEqual(len(result.candidates), 3)
        self.assertEqual({row.status for row in result.candidates}, {"pending"})
        self.assertFalse(self.graph.state().edges)

        valid = self._observation(
            native,
            x,
            2 * x,
            source="s0",
            target="s1",
            revision=current,
        )
        mismatched = replace(valid.receipt, measurement_sha256=inactive.sha256)
        with self.assertRaises(OperatorHarvesterIntegrityError):
            ContextualOperatorObservation(native, mismatched, x, 2 * x)

    def test_candidate_holdout_is_latest_provider_observation_not_hash_order(
        self,
    ) -> None:
        measurements = tuple(self._measurement(index) for index in range(4))
        observations = []
        for index, measurement in enumerate(measurements):
            x = np.array(
                [[index - 1.0], [index + 0.25], [index + 2.0]],
                dtype=np.float64,
            )
            observations.append(
                self._observation(
                    measurement,
                    x,
                    2.0 * x + 1.0,
                    source="temporal/raw",
                    target="temporal/final",
                    revision=self.atlas.revision(),
                )
            )
        # The final cursor item is deliberately chosen independently of SHA
        # ordering; it must remain the holdout after persistence and fitting.
        latest = max(
            observations,
            key=lambda row: row.receipt.sha256,
        )
        if latest is observations[-1]:
            observations[-1], observations[0] = observations[0], observations[-1]
        expected_holdout = observations[-1].receipt.sha256

        result = self._harvester(
            _ReplayProvider((tuple(observations),)),
            state_name="temporal-holdout/v1",
        ).step()

        promoted = next(
            row
            for row in result.candidates
            if row.operator_kind == AFFINE_FLOAT64 and row.status == "promoted"
        )
        self.assertEqual(promoted.holdout_receipt_sha256, expected_holdout)

    def test_scalar_statistic_schemas_partition_evidence_without_tensor_inflation(
        self,
    ) -> None:
        first = self._measurement(0)
        second = replace(
            self._measurement(1, append=False),
            numeric_summaries=(
                NumericSummary(
                    metric="different_scalar_schema",
                    count=2,
                    total=14.0,
                    total_squares=98.0,
                    minimum=7.0,
                    maximum=7.0,
                ),
            ),
        )
        self.atlas.append_measurement(second)
        revision = self.atlas.revision()
        x = np.array([[1.0], [3.0], [5.0]], dtype=np.float64)
        observations = (
            self._observation(
                first,
                x,
                2 * x + 1,
                source="schema/raw",
                target="schema/final",
                revision=revision,
            ),
            self._observation(
                second,
                x,
                2 * x + 1,
                source="schema/raw",
                target="schema/final",
                revision=revision,
            ),
        )

        result = self._harvester(_ReplayProvider((observations,))).step()

        self.assertEqual(result.accepted_observations, 2)
        self.assertEqual(len(self._harvester(_ReplayProvider(())).state().groups), 2)
        self.assertEqual(len(result.candidates), 6)
        self.assertEqual({row.status for row in result.candidates}, {"pending"})
        self.assertFalse(result.promotions)

    def test_audit_coordinates_do_not_split_one_contextual_operator_family(
        self,
    ) -> None:
        measurements = tuple(
            self._measurement(
                index,
                coordinate=_COORDINATE if index % 2 == 0 else _COORDINATE_ALT,
            )
            for index in range(4)
        )
        observations = []
        for index, measurement in enumerate(measurements):
            x = np.array(
                [[index - 1.0], [index + 0.5], [index + 2.0]],
                dtype=np.float64,
            )
            observations.append(
                self._observation(
                    measurement,
                    x,
                    3.0 * x - 1.0,
                    source="coordinate-independent/raw",
                    target="coordinate-independent/final",
                    revision=self.atlas.revision(),
                )
            )

        result = self._harvester(
            _ReplayProvider((tuple(observations),)),
            state_name="coordinate-pooling/v1",
        ).step()

        self.assertEqual(
            len(
                self._harvester(_ReplayProvider(()), state_name="coordinate-pooling/v1")
                .state()
                .groups
            ),
            1,
        )
        promoted = [
            row
            for row in result.promotions
            if row.candidate.operator_kind == AFFINE_FLOAT64
        ]
        self.assertEqual(len(promoted), 1)
        crystal = self.bank.restore_crystal(promoted[0].edge.crystal_sha256)
        self.assertEqual(
            len(crystal.extensions["operator_harvester"]["coordinate_sha256s"]),
            4,
        )

    def test_multiple_weight_sites_from_one_prompt_count_as_one_context(self) -> None:
        shared_probe = ProbeIdentity(
            question_sha256=_hash("shared-question"),
            token_sha256=_hash("shared-tokens"),
            family_sha256=_hash("context-family"),
            label_source_sha256=_hash("label-free"),
        )
        repeated_measurements = []
        for index in range(4):
            measurement = replace(
                self._measurement(
                    index,
                    append=False,
                    coordinate=(_COORDINATE if index % 2 == 0 else _COORDINATE_ALT),
                ),
                probe=shared_probe,
            )
            self.atlas.append_measurement(measurement)
            repeated_measurements.append(measurement)
        matrix = np.array([[2.0, -1.0], [0.5, 3.0]], dtype=np.float64)
        bias = np.array([0.25, -2.0], dtype=np.float64)
        x = np.array(
            [[-2.0, 1.0], [0.0, 0.5], [1.5, -3.0], [4.0, 2.0]],
            dtype=np.float64,
        )
        repeated = tuple(
            self._observation(
                measurement,
                x,
                x @ matrix.T + bias,
                source="prompt-diverse/raw",
                target="prompt-diverse/final",
                revision=self.atlas.revision(),
            )
            for measurement in repeated_measurements
        )
        first = self._harvester(
            _ReplayProvider((repeated,)),
            state_name="prompt-diverse-harvester/v1",
        ).step()
        self.assertEqual(first.accepted_observations, 1)
        self.assertEqual(
            [row.reason for row in first.rejections],
            ["duplicate-prompt-in-group"] * 3,
        )
        self.assertFalse(first.promotions)

        distinct_measurements = tuple(
            self._measurement(20 + index) for index in range(2)
        )
        distinct = tuple(
            self._observation(
                measurement,
                x + index,
                (x + index) @ matrix.T + bias,
                source="prompt-diverse/raw",
                target="prompt-diverse/final",
                revision=self.atlas.revision(),
            )
            for index, measurement in enumerate(distinct_measurements)
        )
        second = self._harvester(
            _ReplayAgainProvider(distinct),
            state_name="prompt-diverse-harvester/v1",
        ).step()
        self.assertEqual(second.accepted_observations, 2)
        affine = [
            row
            for row in second.promotions
            if row.candidate.operator_kind == AFFINE_FLOAT64
        ]
        self.assertEqual(len(affine), 1)
        self.assertEqual(
            len(affine[0].candidate.observation_receipt_sha256s),
            3,
        )

    def test_legacy_persisted_same_prompt_group_rejects_on_resume(self) -> None:
        first_measurement = self._measurement(0)
        second_measurement = replace(
            self._measurement(1, append=False, coordinate=_COORDINATE_ALT),
            probe=first_measurement.probe,
        )
        self.atlas.append_measurement(second_measurement)
        x = np.array(
            [[-2.0, 1.0], [0.0, 0.5], [1.5, -3.0], [4.0, 2.0]],
            dtype=np.float64,
        )
        first = self._observation(
            first_measurement,
            x,
            2.0 * x + 1.0,
            source="legacy-prompt/raw",
            target="legacy-prompt/final",
            revision=self.atlas.revision(),
        )
        second = self._observation(
            second_measurement,
            x,
            2.0 * x + 1.0,
            source="legacy-prompt/raw",
            target="legacy-prompt/final",
            revision=self.atlas.revision(),
        )
        name = "legacy-duplicate-prompt-state/v1"
        harvester = self._harvester(
            _ReplayProvider(((first,),)),
            state_name=name,
        )
        harvester.step()
        clean = self.bank.store.restore_state(name)
        document = json.loads(clean)
        document["body"]["groups"][0]["observations"].append(second.to_record())
        document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(document["body"])
        ).hexdigest()
        contaminated = canonical_json_bytes(document)
        self.bank.store.publish_state(
            name,
            contaminated,
            expected_sha256=hashlib.sha256(clean).hexdigest(),
        )

        with self.assertRaises(OperatorHarvesterIntegrityError):
            self._harvester(
                _ReplayProvider(()),
                state_name=name,
            ).step()

    def test_graph_cas_resume_and_replayed_receipts_are_idempotent(self) -> None:
        measurements = tuple(self._measurement(index) for index in range(4))
        revision = self.atlas.revision()
        observations = tuple(
            self._observation(
                measurement,
                np.array(
                    [[index - 1.0], [index + 0.25], [index + 2.5]],
                    dtype=np.float64,
                ),
                4
                * np.array(
                    [[index - 1.0], [index + 0.25], [index + 2.5]],
                    dtype=np.float64,
                )
                + 3,
                source="resume/raw",
                target="resume/final",
                revision=revision,
            )
            for index, measurement in enumerate(measurements)
        )
        provider = _ReplayProvider((observations,))
        harvester = self._harvester(provider)
        original_append = self.graph.append_edges
        calls = 0

        def conflicted_once(*args: object, **kwargs: object) -> object:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ComputeOperatorGraphConflictError("synthetic concurrent append")
            return original_append(*args, **kwargs)

        with mock.patch.object(self.graph, "append_edges", side_effect=conflicted_once):
            first = harvester.step()
        self.assertGreaterEqual(calls, 2)
        self.assertEqual(len(first.promotions), 1)
        first_graph = self.graph.state()
        state_data = self.bank.store.restore_state(harvester.state_name)
        self.assertEqual(
            HarvesterState.from_bytes(state_data).to_bytes(),
            harvester.state().to_bytes(),
        )
        with self.assertRaises(OperatorHarvesterIntegrityError):
            HarvesterState.from_bytes(state_data + b"\n")

        resumed = self._harvester(provider).step()
        self.assertTrue(resumed.exhausted)
        self.assertEqual(resumed.accepted_observations, 0)
        self.assertFalse(resumed.promotions)
        self.assertEqual(self.graph.state(), first_graph)
        self.assertFalse(resumed.state_publication.changed)

        replayed = self._harvester(_ReplayAgainProvider(observations)).step()
        self.assertEqual(replayed.accepted_observations, 0)
        self.assertEqual(
            {row.reason for row in replayed.rejections},
            {"duplicate-context-receipt"},
        )
        self.assertFalse(replayed.promotions)
        self.assertEqual(self.graph.state(), first_graph)

    def test_persisted_samples_cannot_be_transplanted_to_another_atlas_chain(
        self,
    ) -> None:
        first = self._measurement(0)
        x = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        first_observation = self._observation(
            first,
            x,
            2 * x,
            source="transplant/raw",
            target="transplant/final",
            revision=self.atlas.revision(),
        )
        self._harvester(_ReplayProvider(((first_observation,),))).step()

        other_atlas = SemanticWeightAtlas(
            self.root / "other-atlas", model_pin=_PIN, tensor_plans=(_PLAN,)
        )
        second = self._measurement(99, append=False)
        other_atlas.append_measurement(second)
        second_observation = ContextualOperatorObservation.capture(
            second,
            atlas_revision=other_atlas.revision(),
            emitter_sha256=_EMITTER,
            feature_schema_sha256=_FEATURE_SCHEMA,
            action_schema_sha256=_ACTION_SCHEMA,
            source_state="transplant/raw",
            target_state="transplant/final",
            input_array=x,
            output_array=2 * x,
        )
        transplanted = ContinuousOperatorHarvester(
            atlas=other_atlas,
            provider=_ReplayAgainProvider((second_observation,)),
            graph=self.graph,
            config=HarvesterConfig(
                minimum_observations=3,
                minimum_fit_rows=4,
                max_observations_per_step=64,
                max_samples_per_group=32,
                max_groups=16,
                max_recent_receipts=64,
                graph_cas_retries=4,
            ),
            state_name="test-operator-harvester/v1",
        )

        with self.assertRaisesRegex(
            OperatorHarvesterIntegrityError, "outside this Atlas history"
        ):
            transplanted.step()

    def test_nonlinear_holdout_rejects_and_external_iterator_is_bounded(self) -> None:
        measurements = tuple(self._measurement(index) for index in range(6))
        revision = self.atlas.revision()
        observations = tuple(
            self._observation(
                measurement,
                np.array(
                    [[index - 2.0], [index + 0.2], [index + 0.7]],
                    dtype=np.float64,
                ),
                np.square(
                    np.array(
                        [[index - 2.0], [index + 0.2], [index + 0.7]],
                        dtype=np.float64,
                    )
                ),
                source="nonlinear/raw",
                target="nonlinear/squared",
                revision=revision,
            )
            for index, measurement in enumerate(measurements)
        )
        provider = _ReplayProvider((observations[:3], observations[3:]))
        results = tuple(
            self._harvester(provider, state_name="nonlinear-harvester/v1").iter_steps(
                max_steps=2
            )
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(sum(row.accepted_observations for row in results), 6)
        self.assertFalse(results[-1].promotions)
        affine = next(
            row for row in results[-1].candidates if row.operator_kind == AFFINE_FLOAT64
        )
        self.assertEqual(affine.status, "rejected")
        self.assertIn("residual-exceeds-tolerance", affine.reason)
        self.assertFalse(self.graph.state().edges)

    def test_context_receipt_roundtrip_and_segment_contract_fail_closed(self) -> None:
        measurement = self._measurement(0)
        revision = self.atlas.revision()
        x = np.array([[1.0, 2.0]], dtype=np.float64)
        observation = self._observation(
            measurement,
            x,
            x,
            source="roundtrip/a",
            target="roundtrip/b",
            revision=revision,
        )
        restored = ContextualTransitionReceipt.from_document(
            observation.receipt.to_document()
        )
        self.assertEqual(restored, observation.receipt)
        document = observation.receipt.to_document()
        document["body"]["output_sha256"] = _hash("tampered")
        with self.assertRaises(OperatorHarvesterIntegrityError):
            ContextualTransitionReceipt.from_document(document)

        with self.assertRaises(ValueError):
            self._observation(
                measurement,
                x,
                x,
                source="segment/a",
                target="segment/b",
                revision=revision,
                granularity="segment",
                segment_start=4,
                segment_end=5,
            )


if __name__ == "__main__":
    unittest.main()
