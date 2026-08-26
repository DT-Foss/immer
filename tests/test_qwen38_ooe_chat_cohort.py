from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from immer.contracts import ExecutionStatus, Result
from immer.knowledge.livecausal import LiveGraph
from immer.runtimes.deepseek_v4.causal_weights import TensorRangePlan
from immer.runtimes.ooe.crystal import CrystalStore
from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.result_cells import (
    ResultCell,
    ResultCellBinding,
    qwen_result_binding_evidence,
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


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen38_ooe_chat_cohort.py"
SPEC = importlib.util.spec_from_file_location("qwen38_ooe_chat_cohort", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise AssertionError("cannot import Qwen OoE chat cohort")
cohort = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cohort
SPEC.loader.exec_module(cohort)


QUESTIONS = (
    "A phone tree is used to contact families and relatives of Ali's deceased "
    "coworker. Ali decided to call 3 families. Then each family calls 3 other "
    "families, and so on. How many families will be notified during the fourth "
    "round of calls?",
    "In one year, the number of students on campus doubles at the end of every "
    "month. If there are 10 students on campus at the beginning of the year, "
    "how many additional students would have joined by the end of May, above "
    "and beyond the number of students already on campus at the beginning of "
    "the year?",
    "Dijana and Anis live near a lake, and every weekend they go out rowing "
    "into the lake. On a Sunday morning, both went out rowing, and Dijana rowed "
    "for 50 miles the whole day. Anis rowed 1/5 times more miles than Dijana. "
    "Calculate the total distance the two of them rowed on that day.",
    "The combined age of Peter, Paul and Jean is 100 years old. Find the age "
    "of Peter knowing that Paul is 10 years older than John and that Peter’s "
    "age is equal to the sum of Paul and John's age.",
    "Elly is organizing her books on the new bookcases her parents bought her. "
    "Each of the middle 2 shelves can hold 10 books. The bottom shelf can hold "
    "twice as many books as a middle shelf. The top shelf can hold 5 fewer "
    "books than the bottom shelf. If she has 110 books, how many bookcases does "
    "she need to hold all of them?",
)
ANSWERS = ("81", "310", "110", "50", "2")


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


PIN = ModelPin(
    repo_id="Qwen/Qwen3.8-27B",
    revision="0123456789abcdef",
    bundle_fingerprint=_hash("bundle-layout"),
    bundle_manifest_sha256=_hash("bundle-manifest"),
    code_revision="a" * 40,
)
RUNTIME = RuntimeProvenance(
    code_revision="a" * 40,
    source_manifest_sha256=_hash("source"),
    dependency_manifest_sha256=_hash("dependencies"),
    runtime_configuration_sha256=_hash("runtime"),
    platform_sha256=_hash("platform"),
)
WEIGHT_REVISION = GraphRevision(7, _hash("weight-graph-7"))
PLANS = (
    TensorRangePlan(
        name="model.layers.18.mlp.gate_proj.weight",
        dtype="BF16",
        shape=(8, 4),
        shard="model-00001-of-00002.safetensors",
        absolute_offset=4096,
        length=64,
    ),
    TensorRangePlan(
        name="model.layers.27.self_attn.q_proj.weight",
        dtype="BF16",
        shape=(8, 4),
        shard="model-00002-of-00002.safetensors",
        absolute_offset=8192,
        length=64,
    ),
)


def _summary(metric: str, value: float) -> NumericSummary:
    values = (value - 0.05, value + 0.05)
    return NumericSummary(
        metric=metric,
        count=2,
        total=sum(values),
        total_squares=sum(row * row for row in values),
        minimum=min(values),
        maximum=max(values),
    )


@dataclass
class FakeRuntime:
    atlas_state: object
    rendered: tuple[object, ...]
    answers: tuple[str, ...] = ANSWERS
    fail_after_charge_item: str | None = None
    fail_generate_item: str | None = None

    def __post_init__(self) -> None:
        self.model_pin = PIN
        self.tokenizer_sha256 = _hash("tokenizer.json")
        self.bundle_receipt = {
            "checkpoint_bytes": 1234,
            "graph_revision": [
                WEIGHT_REVISION.sequence,
                WEIGHT_REVISION.event_sha256,
            ],
            "kind": "complete-causal-bundle/v1",
            "layout_fingerprint": PIN.bundle_fingerprint,
            "manifest_sha256": PIN.bundle_manifest_sha256,
            "shards": 2,
            "shards_sha256": _hash("shards"),
            "tensor_bindings": 2,
            "weights_layout": "flat/v1",
        }
        self.generation_policy_sha256 = cohort.GENERATION_POLICY_SHA256
        self.generate_calls: list[str] = []
        self.adjudicate_calls: list[tuple[str, bool]] = []
        self.failed_after_charge = False
        self.failed_generate = False

    def render(self, question: str):
        return self.rendered[QUESTIONS.index(question)]

    def generate(self, question: str, binding: ResultCellBinding) -> Result:
        item = cohort.ITEM_IDS[QUESTIONS.index(question)]
        self.generate_calls.append(item)
        if item == self.fail_generate_item and not self.failed_generate:
            self.failed_generate = True
            raise RuntimeError("simulated process death during Qwen generation")
        output = f"#### {self.answers[QUESTIONS.index(question)]}"
        generation = {
            "context_mode": "stateful_autoregressive",
            "forward_passes": 3,
            "general_generation": True,
            "generated_tokens": 2,
            "linear_calls": 128,
            "prefill_mode": "batched",
            "prompt_tokens": 64,
            "seconds": 0.25,
            "source_body_bytes": 4096,
            "state_bytes": 8192,
            "stateful_cache": True,
            "stopped_on_eos": True,
            "token_trace_sha256": _hash(f"trace:{item}"),
        }
        return Result(
            ExecutionStatus.OK,
            "qwen3.8.causal-chat",
            output=output,
            evidence={
                "bundle": {
                    "layout_fingerprint": PIN.bundle_fingerprint,
                    "manifest_sha256": PIN.bundle_manifest_sha256,
                },
                "generation": generation,
                "model": PIN.repo_id,
                "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                "result_cell_binding_receipt": qwen_result_binding_evidence(binding),
                "revision": PIN.revision,
                "tokenizer_sha256": self.tokenizer_sha256,
            },
        )

    def adjudicate(self, question: str, qwen_result: Result, *, warm: bool):
        self.adjudicate_calls.append((cohort.ITEM_IDS[QUESTIONS.index(question)], warm))
        judgment = {
            "candidate": None,
            "evidence": {
                "exact_solution": None,
                "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                "reason": "fixture_gold_free_abstention",
            },
            "expected": None,
            "kind": "fertig-candidate-verification/v1",
            "status": "abstained",
        }
        final = Result(
            ExecutionStatus.OK,
            "qwen-fertig-chat",
            output=qwen_result.output,
            evidence={
                "receipt": {
                    "fertig_judgment_sha256": cohort.artifact_sha256(judgment),
                    "route": (
                        "ooe_verification_abstained"
                        if warm
                        else "qwen_verification_abstained"
                    ),
                }
            },
        )
        return cohort.Finalized(final, judgment, "abstained")

    def after_charge(self, item_id: str, _cell: ResultCell) -> None:
        if item_id == self.fail_after_charge_item and not self.failed_after_charge:
            self.failed_after_charge = True
            raise RuntimeError("simulated crash after durable organ charge")

    def close(self) -> None:
        return None


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.cartography = root / "cartography"
        self.crystal = root / "chat-crystals"
        self.organ = root / "chat-organs"
        self.manifest = root / "cohort.json"
        self.inputs = root / "inputs.json"
        self.gold = root / "gold.json"
        self.result = root / "result.json"
        atlas = SemanticWeightAtlas(
            LiveGraph(self.cartography / "atlas"),
            model_pin=PIN,
            tensor_plans=PLANS,
        )
        rendered = tuple(
            cohort.RenderedPrompt(
                rendered_prompt_sha256=cohort.RENDERED_PROMPT_SHA256S[index],
                rendered_prompt_token_sha256=(
                    cohort.RENDERED_PROMPT_TOKEN_SHA256S[index]
                ),
            )
            for index in range(5)
        )
        measurements = []
        plan_indices = (0, 0, 1, 1, 0)
        within_site_offsets = (0.0, 0.01, 0.02, 0.03, 0.005)
        for index, question in enumerate(QUESTIONS):
            plan = PLANS[plan_indices[index]]
            layer = 18 if plan_indices[index] == 0 else 27
            measurement = MeasurementReceipt(
                model_pin=PIN,
                coordinate=WeightCoordinate.from_plan(
                    plan,
                    layer=layer,
                    module=plan.name.removesuffix(".weight"),
                ),
                probe=ProbeIdentity(
                    question_sha256=hashlib.sha256(question.encode()).hexdigest(),
                    token_sha256=rendered[index].rendered_prompt_token_sha256,
                    family_sha256=_hash("gsm8k-family"),
                    label_source_sha256=_hash("gold-free-source"),
                ),
                intervention=InterventionIdentity(
                    mode="native",
                    configuration_sha256=_hash(f"native:{index}"),
                ),
                observation_status="recorded",
                observed_semantic_label=None,
                hidden_sha256=_hash(f"hidden:{index}"),
                activation_sha256=_hash(f"activation:{index}"),
                logits_sha256=_hash(f"logits:{index}"),
                state_sha256=_hash(f"state:{index}"),
                access_trace_sha256=_hash(f"access:{index}"),
                evidence_sha256=_hash(f"evidence:{index}"),
                weight_rail_revision=WEIGHT_REVISION,
                atlas_head_revision=atlas.revision(),
                numeric_summaries=(
                    _summary(
                        "activation",
                        2.0 + plan_indices[index] * 10 + within_site_offsets[index],
                    ),
                    _summary(
                        "hidden_rms",
                        3.0 + plan_indices[index] * 10 + within_site_offsets[index],
                    ),
                ),
                placebo_effects=(),
                runtime=RUNTIME,
            )
            atlas.append_measurement(measurement)
            measurements.append(measurement)
        self.measurements = tuple(measurements)
        self.atlas_state = cohort.AtlasState(
            atlas=atlas,
            measurements={row.sha256: row for row in measurements},
        )
        self.runtime = FakeRuntime(self.atlas_state, rendered)
        self.factory = lambda _args, _manifest: self.runtime
        inputs = cohort._seal(
            cohort.INPUT_SCHEMA,
            {
                "items": [
                    {
                        "item_id": cohort.ITEM_IDS[index],
                        "measurement_sha256": measurements[index].sha256,
                        "question": question,
                    }
                    for index, question in enumerate(QUESTIONS)
                ],
                "system_prompt": cohort.SYSTEM_PROMPT,
            },
        )
        self.inputs.write_bytes(cohort._document_bytes(inputs))
        gold = cohort._seal(
            cohort.GOLD_SCHEMA,
            {
                "items": [
                    {"answer": ANSWERS[index], "item_id": cohort.ITEM_IDS[index]}
                    for index in range(5)
                ]
            },
        )
        self.gold.write_bytes(cohort._document_bytes(gold))

    def prepare_args(self):
        return cohort.argparse.Namespace(
            bundle=str(self.root / "bundle"),
            cartography_root=str(self.cartography),
            crystal_root=str(self.crystal),
            inputs=str(self.inputs),
            manifest=str(self.manifest),
            organ_root=str(self.organ),
            tokenizer=str(self.root / "tokenizer.json"),
        )

    def execute_args(self):
        return cohort.argparse.Namespace(
            bundle=str(self.root / "bundle"),
            cartography_root=str(self.cartography),
            cold_then_warm=True,
            compute_dtype=cohort.COMPUTE_DTYPE,
            crystal_root=str(self.crystal),
            device=cohort.DEVICE,
            manifest=str(self.manifest),
            max_new_tokens=cohort.MAX_NEW_TOKENS,
            max_resident_mb=cohort.MAX_RESIDENT_MB,
            organ_root=str(self.organ),
            output=str(self.result),
            source_budget_mb=cohort.SOURCE_BUDGET_MB,
            tokenizer=str(self.root / "tokenizer.json"),
        )

    def verify_args(self):
        return cohort.argparse.Namespace(
            cartography_root=str(self.cartography),
            crystal_root=str(self.crystal),
            gold=str(self.gold),
            manifest=str(self.manifest),
            organ_root=str(self.organ),
            result=str(self.result),
        )

    def prepare(self):
        return cohort.prepare(self.prepare_args(), runtime_factory=self.factory)

    def execute(self):
        return cohort.execute(self.execute_args(), runtime_factory=self.factory)


class Qwen38OoeChatCohortTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = Fixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_prepare_freezes_exact_ids_split_and_hash_only_runtime_identity(self) -> None:
        manifest = self.fixture.prepare()
        body = manifest["body"]
        self.assertEqual(body["ordered_item_ids"], list(cohort.ITEM_IDS))
        self.assertEqual(body["train_item_ids"], list(cohort.ITEM_IDS[:4]))
        self.assertEqual(body["holdout_item_id"], cohort.ITEM_IDS[4])
        self.assertEqual(body["model_pin_sha256"], PIN.sha256)
        self.assertEqual(
            body["generation_policy_sha256"], cohort.GENERATION_POLICY_SHA256
        )
        self.assertEqual([row["split"] for row in body["items"]], [
            "train", "train", "train", "train", "holdout"
        ])
        encoded = self.fixture.manifest.read_bytes()
        for answer in ANSWERS:
            self.assertNotIn(f'"answer":"{answer}"'.encode(), encoded)

    def test_cold_train_holdout_warm_placebos_verify_and_idempotence(self) -> None:
        self.fixture.prepare()
        gold_bytes = self.fixture.gold.read_bytes()
        self.fixture.gold.unlink()
        result = self.fixture.execute()
        self.assertFalse(self.fixture.gold.exists())
        self.fixture.gold.write_bytes(gold_bytes)
        self.assertEqual(len(self.fixture.runtime.generate_calls), 5)
        headline = result["body"]["headline"]
        self.assertEqual(headline["cold_qwen_forwards_total"], 15)
        self.assertEqual(headline["holdout_teacher_baseline_qwen_forwards"], 3)
        self.assertEqual(headline["holdout_executed_qwen_forwards"], 0)
        self.assertEqual(headline["holdout_saved_qwen_forwards"], 3)
        self.assertEqual(headline["teacher_transitions"], 4)
        split = result["body"]["temporal_split"]
        self.assertFalse(split["holdout_transition_ingested"])
        self.assertEqual(len(split["train_feature_receipt_sha256s"]), 4)
        self.assertEqual(
            [row["mode"] for row in result["body"]["placebos"]["rows"]],
            ["shuffled-site", "shuffled-crystal"],
        )
        parity_body = result["body"]["warm"]["parity"]["body"]
        self.assertTrue(parity_body["exact_parity"])
        self.assertTrue(parity_body["qwen_result_document_exact"])
        self.assertTrue(parity_body["final_semantic_core_exact"])
        self.assertNotEqual(
            parity_body["cold_final_evidence_sha256"],
            parity_body["warm_final_evidence_sha256"],
        )
        bank = cohort.ResultCellBank(self.fixture.organ)
        for index, row in enumerate(result["body"]["cold_rows"]):
            cell = bank.restore_payload(row["cell_payload_sha256"])
            feature = cohort.QwenOoeFeatureReceipt.from_document(row["feature"])
            pair = cell.cold_generation_provenance_unit.pair_binding_sha256
            atlas_verifier = cohort._load_manifest(self.fixture.manifest)["body"][
                "items"
            ][index]["atlas_measurement_verifier_sha256"]
            self.assertIn(pair, feature.verifier_sha256s)
            self.assertIn(pair, feature.evidence_sha256s)
            self.assertIn(atlas_verifier, feature.verifier_sha256s)
        store = CrystalStore(self.fixture.crystal)
        controller_state = json.loads(
            store.restore_state(
                cohort._controller_name(
                    cohort._load_manifest(self.fixture.manifest)["sha256"]
                )
            )
        )
        history = [
            row
            for site in controller_state["body"]["sites"]
            for row in site["history"]
        ]
        self.assertEqual(len(history), 4)
        self.assertNotIn(
            result["body"]["temporal_split"]["holdout_feature_receipt_sha256"],
            {row["feature"]["sha256"] for row in history},
        )
        verification = cohort.verify(self.fixture.verify_args())
        self.assertTrue(
            verification["body"]["benchmark"]["body"]["evaluator_quality_verified"]
        )
        self.assertEqual(verification["body"]["evaluator_calls"], 1)
        self.assertNotIn("holdout_answer", verification["body"])

        first_bytes = self.fixture.result.read_bytes()
        replay = self.fixture.execute()
        self.assertEqual(replay, result)
        self.assertEqual(self.fixture.result.read_bytes(), first_bytes)
        self.assertEqual(len(self.fixture.runtime.generate_calls), 5)

    def test_manifest_organ_crystal_and_gold_tamper_fail_closed(self) -> None:
        manifest = self.fixture.prepare()
        manifest_document = json.loads(self.fixture.manifest.read_bytes())
        manifest_document["body"]["holdout_item_id"] = cohort.ITEM_IDS[0]
        self.fixture.manifest.write_bytes(cohort._document_bytes(manifest_document))
        with self.assertRaisesRegex(cohort.ChatCohortError, "SHA-256"):
            self.fixture.execute()
        self.assertEqual(self.fixture.runtime.generate_calls, [])

        self.fixture.manifest.write_bytes(cohort._document_bytes(manifest))
        self.fixture.execute()
        bank = cohort.ResultCellBank(self.fixture.organ)
        holdout_payload = json.loads(self.fixture.result.read_bytes())["body"][
            "cold_rows"
        ][4]["cell_payload_sha256"]
        object_name = bank.object_state_name(holdout_payload)
        object_path = bank.store.root / "state" / bank.store._state_filename(object_name)
        object_bytes = bytearray(object_path.read_bytes())
        object_bytes[-1] ^= 1
        object_path.chmod(0o600)
        object_path.write_bytes(object_bytes)
        with self.assertRaises(Exception):
            cohort.verify(self.fixture.verify_args())

        # A separate fixture proves active Crystal object tamper independently.
        with tempfile.TemporaryDirectory() as temporary:
            second = Fixture(Path(temporary))
            second.prepare()
            second.execute()
            crystal = next((second.crystal / "objects").glob("*.crystal"))
            payload = bytearray(crystal.read_bytes())
            payload[-1] ^= 1
            crystal.chmod(0o600)
            crystal.write_bytes(payload)
            with self.assertRaises(Exception):
                cohort.verify(second.verify_args())

        with tempfile.TemporaryDirectory() as temporary:
            third = Fixture(Path(temporary))
            third.prepare()
            third.execute()
            gold = json.loads(third.gold.read_bytes())
            gold["body"]["items"][4]["answer"] = "999"
            gold["sha256"] = cohort._digest(gold["body"])
            third.gold.write_bytes(cohort._document_bytes(gold))
            with self.assertRaisesRegex(Exception, "evaluator rejected"):
                cohort.verify(third.verify_args())

    def test_store_symlink_alias_and_prefix_only_prompt_forgery_are_rejected(self) -> None:
        self.fixture.crystal.mkdir()
        self.fixture.organ.symlink_to(self.fixture.crystal, target_is_directory=True)
        with self.assertRaisesRegex(cohort.ChatCohortError, "symlink"):
            self.fixture.prepare()

        self.fixture.organ.unlink()
        forged = cohort.RenderedPrompt(
            rendered_prompt_sha256=(
                cohort.RENDERED_PROMPT_SHA256S[0][:8] + "0" * 56
            ),
            rendered_prompt_token_sha256=cohort.RENDERED_PROMPT_TOKEN_SHA256S[0],
        )
        original = self.fixture.runtime.rendered
        self.fixture.runtime.rendered = (forged, *original[1:])
        with self.assertRaisesRegex(cohort.ChatCohortError, "rendered prompt identity"):
            self.fixture.prepare()

    def test_resume_after_durable_cell_charge_never_repeats_qwen(self) -> None:
        self.fixture.prepare()
        failed_item = cohort.ITEM_IDS[2]
        self.fixture.runtime.fail_after_charge_item = failed_item
        with self.assertRaisesRegex(RuntimeError, "simulated crash"):
            self.fixture.execute()
        self.assertEqual(self.fixture.runtime.generate_calls, list(cohort.ITEM_IDS[:3]))
        self.fixture.runtime.fail_after_charge_item = None
        result = self.fixture.execute()
        self.assertEqual(self.fixture.runtime.generate_calls, list(cohort.ITEM_IDS))
        self.assertEqual(result["body"]["headline"]["holdout_saved_qwen_forwards"], 3)

    def test_uncertain_generation_fails_closed_without_second_qwen_call(self) -> None:
        self.fixture.prepare()
        self.fixture.runtime.fail_generate_item = cohort.ITEM_IDS[0]
        with self.assertRaisesRegex(RuntimeError, "during Qwen generation"):
            self.fixture.execute()
        self.assertEqual(self.fixture.runtime.generate_calls, [cohort.ITEM_IDS[0]])
        self.fixture.runtime.fail_generate_item = None
        with self.assertRaisesRegex(cohort.ChatCohortError, "uncertain"):
            self.fixture.execute()
        self.assertEqual(self.fixture.runtime.generate_calls, [cohort.ITEM_IDS[0]])

    def test_stale_atlas_head_fails_before_any_generation(self) -> None:
        self.fixture.prepare()
        prior_calls = len(self.fixture.runtime.generate_calls)
        measurement = self.fixture.measurements[0]
        appended = MeasurementReceipt(
            model_pin=measurement.model_pin,
            coordinate=measurement.coordinate,
            probe=ProbeIdentity(
                question_sha256=_hash("unrelated-question"),
                token_sha256=_hash("unrelated-tokens"),
                family_sha256=_hash("unrelated-family"),
                label_source_sha256=_hash("unrelated-source"),
            ),
            intervention=measurement.intervention,
            observation_status="recorded",
            observed_semantic_label=None,
            hidden_sha256=_hash("new-hidden"),
            activation_sha256=_hash("new-activation"),
            logits_sha256=_hash("new-logits"),
            state_sha256=_hash("new-state"),
            access_trace_sha256=_hash("new-access"),
            evidence_sha256=_hash("new-evidence"),
            weight_rail_revision=measurement.weight_rail_revision,
            atlas_head_revision=self.fixture.atlas_state.atlas.revision(),
            numeric_summaries=measurement.numeric_summaries,
            placebo_effects=(),
            runtime=measurement.runtime,
        )
        self.fixture.atlas_state.atlas.append_measurement(appended)
        with self.assertRaisesRegex(cohort.ChatCohortError, "runtime identity"):
            self.fixture.execute()
        self.assertEqual(len(self.fixture.runtime.generate_calls), prior_calls)

    def test_stale_runtime_model_pin_and_bundle_graph_fail_before_generation(self) -> None:
        self.fixture.prepare()
        original_pin = self.fixture.runtime.model_pin
        self.fixture.runtime.model_pin = ModelPin(
            repo_id=PIN.repo_id,
            revision=PIN.revision,
            bundle_fingerprint=PIN.bundle_fingerprint,
            bundle_manifest_sha256=PIN.bundle_manifest_sha256,
            code_revision="b" * 40,
        )
        with self.assertRaisesRegex(cohort.ChatCohortError, "runtime identity"):
            self.fixture.execute()
        self.assertEqual(self.fixture.runtime.generate_calls, [])

        self.fixture.runtime.model_pin = original_pin
        original_bundle = dict(self.fixture.runtime.bundle_receipt)
        self.fixture.runtime.bundle_receipt = {
            **original_bundle,
            "graph_revision": [8, _hash("stale-weight-graph")],
        }
        with self.assertRaisesRegex(cohort.ChatCohortError, "runtime identity"):
            self.fixture.execute()
        self.assertEqual(self.fixture.runtime.generate_calls, [])

    def test_result_tamper_and_controller_baseline_inflation_fail_closed(self) -> None:
        manifest = self.fixture.prepare()
        self.fixture.execute()
        result = json.loads(self.fixture.result.read_bytes())
        result["body"]["headline"]["holdout_saved_qwen_forwards"] = 999
        self.fixture.result.write_bytes(cohort._document_bytes(result))
        with self.assertRaisesRegex(cohort.ChatCohortError, "SHA-256"):
            cohort.verify(self.fixture.verify_args())

        self.fixture.result.write_bytes(
            CrystalStore(self.fixture.crystal).restore_state(
                cohort._result_name(manifest["sha256"])
            )
        )
        store = CrystalStore(self.fixture.crystal)
        state_name = cohort._controller_name(manifest["sha256"])
        old = store.restore_state(state_name)
        snapshot = json.loads(old)
        transaction = snapshot["body"]["warm_transactions"][0]
        transaction["final_receipt"]["saved_qwen_forwards"] = 999
        snapshot["body"]["metrics"]["saved_qwen_forwards"] = 999
        snapshot["sha256"] = cohort._digest(snapshot["body"])
        store.publish_state(
            state_name,
            canonical_json_bytes(snapshot),
            expected_sha256=hashlib.sha256(old).hexdigest(),
        )
        with self.assertRaises(cohort.ChatCohortError):
            cohort.verify(self.fixture.verify_args())


if __name__ == "__main__":
    unittest.main()
