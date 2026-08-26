from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from immer.knowledge.livecausal import LiveGraph
from immer.runtimes.deepseek_v4.causal_weights import TensorRangePlan
from immer.runtimes.o1_state import ProbeJob, ProbeOutcome, ProbeTarget
from immer.runtimes.ooe.cartography import OoeCartographyBridge
from immer.runtimes.ooe.controller import (
    CONTROLLER_STATE_NAME,
    ControllerConfig,
    OoeController,
)
from immer.runtimes.ooe.crystal import CrystalStore
from immer.runtimes.ooe.identity import canonical_json_bytes
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


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen38_ooe_cold_warm_cohort.py"
SPEC = importlib.util.spec_from_file_location("qwen38_ooe_cold_warm_cohort", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise AssertionError("cannot import Qwen OoE cold/warm cohort script")
cohort = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cohort
SPEC.loader.exec_module(cohort)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


PIN = ModelPin(
    repo_id="Qwen/Qwen3.8-27B",
    revision="0123456789abcdef",
    bundle_fingerprint=_hash("bundle"),
    bundle_manifest_sha256=_hash("bundle-manifest"),
    code_revision="a" * 40,
)
RUNTIME = RuntimeProvenance(
    code_revision="a" * 40,
    source_manifest_sha256=_hash("source"),
    dependency_manifest_sha256=_hash("deps"),
    runtime_configuration_sha256=_hash("runtime-config"),
    platform_sha256=_hash("platform"),
)
WEIGHT_REVISION = GraphRevision(9, _hash("weight-9"))
PLANS = (
    TensorRangePlan(
        name="model.layers.0.input_layernorm.weight",
        dtype="BF16",
        shape=(8,),
        shard="model-00001-of-00002.safetensors",
        absolute_offset=4096,
        length=16,
    ),
    TensorRangePlan(
        name="model.layers.2.self_attn.q_proj.weight",
        dtype="BF16",
        shape=(8, 4),
        shard="model-00002-of-00002.safetensors",
        absolute_offset=8192,
        length=64,
    ),
)


def _summary(metric: str, signal: float) -> NumericSummary:
    values = (signal - 0.1, signal + 0.1)
    return NumericSummary(
        metric=metric,
        count=2,
        total=sum(values),
        total_squares=sum(value * value for value in values),
        minimum=min(values),
        maximum=max(values),
    )


def _measurement(
    *,
    index: int,
    plan: TensorRangePlan,
    probe: ProbeIdentity,
    mode: str,
    atlas_revision: GraphRevision,
) -> MeasurementReceipt:
    layer = 0 if ".layers.0." in plan.name else 2
    module = plan.name.removesuffix(".weight")
    coordinate = WeightCoordinate.from_plan(plan, layer=layer, module=module)
    signal = 1.5 + index * 0.25 + layer
    return MeasurementReceipt(
        model_pin=PIN,
        coordinate=coordinate,
        probe=probe,
        intervention=InterventionIdentity(
            mode=mode,
            configuration_sha256=_hash(f"intervention-{index}"),
        ),
        observation_status="recorded",
        observed_semantic_label=None,
        hidden_sha256=_hash(f"hidden-{index}"),
        activation_sha256=_hash(f"activation-{index}"),
        logits_sha256=_hash(f"logits-{index}"),
        state_sha256=_hash(f"state-{index}"),
        access_trace_sha256=_hash(f"access-{index}"),
        evidence_sha256=_hash(f"evidence-{index}"),
        weight_rail_revision=WEIGHT_REVISION,
        atlas_head_revision=atlas_revision,
        numeric_summaries=(
            _summary("activation", signal),
            _summary("hidden_rms", signal + 0.5),
        ),
        placebo_effects=(),
        runtime=RUNTIME,
    )


def _scheduler_document(
    jobs: tuple[ProbeJob, ...], outcomes: tuple[ProbeOutcome, ...]
) -> dict[str, object]:
    body: dict[str, object] = {
        "config": {},
        "generation": 1,
        "histories": [],
        "in_flight": None,
        "jobs": [row.to_document() for row in jobs],
        "last_stop_reason": "coverage-complete",
        "outcomes": [row.to_document() for row in outcomes],
        "replay": [],
        "schema": "immer.o1-cartography-state/v1",
        "stream_state": {},
    }
    return {**body, "state_sha256": _hash_document(body)}


def _hash_document(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


class _Fixture:
    def __init__(self, base: Path) -> None:
        self.root = base / "cartography"
        self.root.mkdir()
        self.atlas = SemanticWeightAtlas(
            LiveGraph(self.root / "atlas"),
            model_pin=PIN,
            tensor_plans=PLANS,
        )
        probes = (
            ProbeIdentity(
                question_sha256=_hash("question-a"),
                token_sha256=_hash("tokens-a"),
                family_sha256=_hash("family"),
                label_source_sha256=_hash("label-source"),
            ),
            ProbeIdentity(
                question_sha256=_hash("question-b"),
                token_sha256=_hash("tokens-b"),
                family_sha256=_hash("family"),
                label_source_sha256=_hash("label-source"),
            ),
        )
        layout = (
            (PLANS[1], probes[0], "native"),
            (PLANS[0], probes[0], "passive"),
            (PLANS[1], probes[1], "native"),
            (PLANS[0], probes[1], "passive"),
        )
        measurements: list[MeasurementReceipt] = []
        for index, (plan, probe, mode) in enumerate(layout):
            measurement = _measurement(
                index=index,
                plan=plan,
                probe=probe,
                mode=mode,
                atlas_revision=self.atlas.revision(),
            )
            self.atlas.append_measurement(measurement)
            measurements.append(measurement)
        self.measurements = tuple(measurements)

        jobs: list[ProbeJob] = []
        outcomes: list[ProbeOutcome] = []
        for index, measurement in enumerate(self.measurements):
            layer = measurement.coordinate.layer
            prompt_sha256 = (
                probes[0].question_sha256 if index < 2 else probes[1].question_sha256
            )
            job = ProbeJob.create(
                layer=layer,
                target=ProbeTarget(
                    module=measurement.coordinate.module,
                    unit_kind="module",
                    unit_index=None,
                ),
                probe_family="unlabeled",
                intervention=str(measurement.intervention.mode),
                code_pin="a" * 40,
                model_pin=PIN.sha256,
                seed=index + 1,
                prompt_sha256=prompt_sha256,
            )
            observation = {
                "atlas_receipt_sha256": measurement.sha256,
                "coordinate_sha256": measurement.coordinate.sha256,
                "measurement_sha256": measurement.sha256,
                "model_pin_sha256": PIN.sha256,
            }
            outcome = replace(
                ProbeOutcome.succeeded(
                    job,
                    1,
                    observation,
                    read_bytes=measurement.coordinate.range_length,
                    model_seconds=1.0 + index,
                    wall_seconds=1.0 + index,
                    atlas_receipt_sha256=measurement.sha256,
                ),
                surprise=1.0 + index,
                learning_progress=2.0 + index,
            )
            jobs.append(job)
            outcomes.append(outcome)
        self.jobs = tuple(jobs)
        self.outcomes = tuple(outcomes)
        _write_json(
            self.root / "scheduler.json",
            _scheduler_document(self.jobs, self.outcomes),
        )

        self.store = CrystalStore(self.root / "ooe")
        controller = OoeController(
            model_pin_sha256=PIN.sha256,
            weight_graph_revision_sha256=WEIGHT_REVISION.sha256,
            atlas_graph_revision=self.measurements[0].atlas_head_revision,
            crystal_store=self.store,
            atlas_revision_verifier=self.atlas.contains_revision,
            config=ControllerConfig(
                replicas=4,
                replica_fanout=4,
                min_coverage_per_source=1,
                min_promoted_sources=1,
                router_radius=10.0,
                router_min_margin=0.0,
                token_min_confidence=1e-9,
                consensus_tolerance=1e-7,
                consensus_max_rounds=4096,
                reservoir_size=8,
            ),
        )
        bridge = OoeCartographyBridge(controller)
        current: dict[str, str] = {}
        for index, (job, outcome, measurement) in enumerate(
            zip(self.jobs, self.outcomes, self.measurements, strict=True)
        ):
            source_action = current.get(job.prompt_sha256, "qwen_fallback")
            bridge.ingest_authenticated(
                self.atlas,
                measurement,
                source_action=source_action,
                target_action="probe_coordinate",
                o1_surprise=outcome.surprise,
                o1_learning_progress=outcome.learning_progress,
                evidence_sha256s=(outcome.attempt_id,),
            )
            current[job.prompt_sha256] = "probe_coordinate"
        publications = bridge.promote_ready()
        if len(publications) != 2:
            raise AssertionError("fixture did not promote two real sites")
        controller.save_snapshot()
        self.output = self.root / "cohort-result.json"

    def scheduler_document(self) -> dict[str, object]:
        return json.loads((self.root / "scheduler.json").read_bytes())

    def write_scheduler(self, value: dict[str, object]) -> None:
        body = dict(value)
        body.pop("state_sha256", None)
        value = {**body, "state_sha256": _hash_document(body)}
        _write_json(self.root / "scheduler.json", value)


class Qwen38OoeColdWarmCohortTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = _Fixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_run_is_real_offline_exact_and_idempotent(self) -> None:
        before = self.fixture.store.restore_state(CONTROLLER_STATE_NAME)
        result = cohort.run(self.fixture.root, self.fixture.output)
        headline = result["body"]["headline"]
        self.assertEqual(
            headline,
            {
                "attached_scheduler_outcomes": 4,
                "executed_qwen_forwards": 0,
                "saved_qwen_forwards": 4,
                "teacher_baseline_qwen_forwards": 4,
                "teacher_calls": 0,
                "verified_warm_results": 4,
            },
        )
        self.assertEqual(
            result["body"]["schedule"]["measurement_sha256s"],
            [row.sha256 for row in self.fixture.measurements],
        )
        self.assertTrue(result["body"]["store_audit"]["clean"])
        after_first = self.fixture.store.restore_state(CONTROLLER_STATE_NAME)
        self.assertNotEqual(after_first, before)
        output_first = self.fixture.output.read_bytes()

        replay = cohort.run(self.fixture.root, self.fixture.output)
        self.assertEqual(replay, result)
        self.assertEqual(self.fixture.output.read_bytes(), output_first)
        self.assertEqual(
            self.fixture.store.restore_state(CONTROLLER_STATE_NAME), after_first
        )
        snapshot = json.loads(after_first)
        self.assertEqual(len(snapshot["body"]["warm_transactions"]), 4)

    def test_verify_reopens_every_transaction_without_decisions(self) -> None:
        expected = cohort.run(self.fixture.root, self.fixture.output)
        before = self.fixture.store.restore_state(CONTROLLER_STATE_NAME)
        verified = cohort.verify(self.fixture.root, self.fixture.output)
        after = self.fixture.store.restore_state(CONTROLLER_STATE_NAME)
        self.assertEqual(verified, expected)
        self.assertEqual(after, before)

    def test_crash_after_controller_cas_reuses_transactions_exactly_once(self) -> None:
        states = self.fixture.root / "ooe" / "state"
        before_files = set(states.iterdir())
        first = cohort.run(self.fixture.root, self.fixture.output)
        after_files = set(states.iterdir())
        cohort_files = after_files - before_files
        self.assertEqual(len(cohort_files), 1)
        snapshot = self.fixture.store.restore_state(CONTROLLER_STATE_NAME)
        cohort_files.pop().unlink()
        self.fixture.output.unlink()

        recovered = cohort.run(self.fixture.root, self.fixture.output)
        self.assertEqual(recovered, first)
        self.assertEqual(
            self.fixture.store.restore_state(CONTROLLER_STATE_NAME), snapshot
        )
        controller = json.loads(snapshot)
        self.assertEqual(len(controller["body"]["warm_transactions"]), 4)

    def test_pending_execution_is_never_executed_a_second_time(self) -> None:
        context = cohort._build_context(self.fixture.root)
        controller = cohort._restore_controller(context)
        schedule = context.rows[0]
        decision = controller.try_warm(
            schedule.feature,
            schedule.source_action,
            quality_verifier=lambda receipt, execution: (
                cohort.verify_atlas_probe_execution(
                    context.atlas_state.atlas,
                    receipt,
                    execution,
                )
            ),
            stream_id="crash-before-final-commit",
        )
        self.assertEqual(decision.origin, "crystal")
        prior = hashlib.sha256(
            self.fixture.store.restore_state(CONTROLLER_STATE_NAME)
        ).hexdigest()
        controller.save_snapshot(expected_sha256=prior)
        with self.assertRaisesRegex(cohort.CohortError, "uncommitted warm execution"):
            cohort.run(self.fixture.root, self.fixture.output)
        snapshot = json.loads(
            self.fixture.store.restore_state(CONTROLLER_STATE_NAME)
        )
        self.assertEqual(len(snapshot["body"]["warm_transactions"]), 1)

    def test_result_tamper_is_rejected(self) -> None:
        cohort.run(self.fixture.root, self.fixture.output)
        document = json.loads(self.fixture.output.read_bytes())
        document["body"]["headline"]["saved_qwen_forwards"] = 99
        _write_json(self.fixture.output, document)
        with self.assertRaisesRegex(cohort.CohortError, "SHA-256"):
            cohort.verify(self.fixture.root, self.fixture.output)

    def test_resealed_transaction_tamper_is_rejected(self) -> None:
        result = cohort.run(self.fixture.root, self.fixture.output)
        body = result["body"]
        body["rows"][0]["transaction_sha256"] = "0" * 64
        tampered = cohort._sealed_result(body)
        encoded = cohort._result_bytes(tampered)
        state_name = result["body"]["cohort_state_name"]
        current = hashlib.sha256(
            self.fixture.store.restore_state(state_name)
        ).hexdigest()
        self.fixture.store.publish_state(
            state_name,
            encoded,
            expected_sha256=current,
        )
        self.fixture.output.write_bytes(encoded)
        with self.assertRaisesRegex(cohort.CohortError, "controller replay"):
            cohort.verify(self.fixture.root, self.fixture.output)

    def test_resealed_snapshot_cannot_inflate_qwen_forward_savings(self) -> None:
        states = self.fixture.root / "ooe" / "state"
        before_files = set(states.iterdir())
        cohort.run(self.fixture.root, self.fixture.output)
        cohort_files = set(states.iterdir()) - before_files
        self.assertEqual(len(cohort_files), 1)
        cohort_files.pop().unlink()
        self.fixture.output.unlink()

        old_snapshot = self.fixture.store.restore_state(CONTROLLER_STATE_NAME)
        snapshot = json.loads(old_snapshot)
        transactions = snapshot["body"]["warm_transactions"]
        self.assertEqual(len(transactions), 4)
        for transaction in transactions:
            execution = transaction["execution"]
            execution["body"]["teacher_baseline_qwen_forwards"] = 9
            execution["sha256"] = _hash_document(execution["body"])
            accounting = transaction["final_receipt"]
            accounting["execution_receipt_sha256"] = execution["sha256"]
            accounting["saved_qwen_forwards"] = 9
        snapshot["body"]["metrics"]["saved_qwen_forwards"] = 36
        snapshot["sha256"] = _hash_document(snapshot["body"])
        tampered_snapshot = canonical_json_bytes(snapshot)
        self.fixture.store.publish_state(
            CONTROLLER_STATE_NAME,
            tampered_snapshot,
            expected_sha256=hashlib.sha256(old_snapshot).hexdigest(),
        )

        # The generic controller accepts this internally consistent reseal;
        # cohort accounting must reject it against the immutable probe schedule.
        context = cohort._build_context(self.fixture.root)
        restored = cohort._restore_controller(context)
        self.assertEqual(restored.metrics.saved_qwen_forwards, 36)
        with self.assertRaisesRegex(cohort.CohortError, "exactly one forward"):
            cohort.run(self.fixture.root, self.fixture.output)
        self.assertFalse(self.fixture.output.exists())
        with self.assertRaises(KeyError):
            self.fixture.store.restore_state(context.cohort_state_name)

    def test_stale_and_duplicate_scheduler_outcomes_are_rejected(self) -> None:
        duplicate = self.fixture.scheduler_document()
        duplicate["outcomes"].append(duplicate["outcomes"][0])
        self.fixture.write_scheduler(duplicate)
        with self.assertRaisesRegex(cohort.CohortError, "duplicate attempts"):
            cohort.run(self.fixture.root, self.fixture.output)

        self.fixture.write_scheduler(
            _scheduler_document(self.fixture.jobs, self.fixture.outcomes)
        )
        stale = self.fixture.scheduler_document()
        stale["outcomes"][0]["atlas_receipt_sha256"] = self.fixture.measurements[1].sha256
        stale["outcomes"][0]["observation"]["atlas_receipt_sha256"] = (
            self.fixture.measurements[1].sha256
        )
        stale["outcomes"][0]["observation"]["measurement_sha256"] = (
            self.fixture.measurements[1].sha256
        )
        self.fixture.write_scheduler(stale)
        with self.assertRaisesRegex(cohort.CohortError, "historical OoE feature"):
            cohort.run(self.fixture.root, self.fixture.output)

    def test_stale_model_pin_is_rejected_before_execution(self) -> None:
        jobs = list(self.fixture.jobs)
        outcomes = list(self.fixture.outcomes)
        original = jobs[0]
        stale_job = ProbeJob.create(
            layer=original.layer,
            target=original.target,
            probe_family=original.probe_family,
            intervention=original.intervention,
            code_pin=original.code_pin,
            model_pin=_hash("stale-model"),
            seed=original.seed,
            prompt_sha256=original.prompt_sha256,
            read_budget_bytes=original.read_budget_bytes,
            model_budget_seconds=original.model_budget_seconds,
        )
        observation = outcomes[0].observation_document()
        assert observation is not None
        outcomes[0] = replace(
            ProbeOutcome.succeeded(
                stale_job,
                1,
                observation,
                read_bytes=outcomes[0].read_bytes,
                model_seconds=outcomes[0].model_seconds,
                wall_seconds=outcomes[0].wall_seconds,
                atlas_receipt_sha256=outcomes[0].atlas_receipt_sha256,
            ),
            surprise=outcomes[0].surprise,
            learning_progress=outcomes[0].learning_progress,
        )
        jobs[0] = stale_job
        self.fixture.write_scheduler(_scheduler_document(tuple(jobs), tuple(outcomes)))
        with self.assertRaisesRegex(cohort.CohortError, "stale Qwen model pin"):
            cohort.run(self.fixture.root, self.fixture.output)

    def test_atlas_and_crystal_tamper_are_rejected(self) -> None:
        atlas_segment = next((self.fixture.root / "atlas").glob("*.seg"))
        original = atlas_segment.read_bytes()
        atlas_segment.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        with self.assertRaisesRegex(cohort.CohortError, "authenticate the Atlas"):
            cohort.run(self.fixture.root, self.fixture.output)

        atlas_segment.write_bytes(original)
        crystal = next((self.fixture.root / "ooe" / "objects").glob("*.crystal"))
        crystal_bytes = crystal.read_bytes()
        crystal.chmod(0o600)
        crystal.write_bytes(crystal_bytes[:-1] + bytes([crystal_bytes[-1] ^ 1]))
        with self.assertRaisesRegex(cohort.CohortError, "historical OoE controller"):
            cohort.run(self.fixture.root, self.fixture.output)


if __name__ == "__main__":
    unittest.main()
