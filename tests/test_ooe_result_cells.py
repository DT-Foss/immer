from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest

from immer.contracts import ExecutionStatus, Result
from immer.runtimes.ooe.chat import result_from_document
from immer.runtimes.ooe.crystal import CrystalStore
from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.qwen_bridge import (
    ACTION_SCHEMA_SHA256,
    QwenOoeFeatureReceipt,
    feature_schema_sha256,
)
from immer.runtimes.ooe.result_cells import (
    COLD_QWEN_BINDING_EVIDENCE_KEY,
    FinalBenchmarkLayer,
    ProvenanceUnit,
    RESULT_CELL_POINTER_SCHEMA,
    ResultCell,
    ResultCellBank,
    ResultCellBinding,
    ResultCellConflictError,
    ResultCellError,
    ResultCellExecutor,
    ResultCellIntegrityError,
    ResultCellParityError,
    ResultCellParityVerifier,
    ResultCellStaleError,
    attach_cold_qwen_generation_receipt,
    artifact_sha256,
    extract_cold_qwen_generation_receipt,
    qwen_result_binding_evidence,
)
from immer.runtimes.qwen3_8.semantic_atlas import (
    GraphRevision,
    ModelPin,
    ProbeIdentity,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_RAW_QUESTION = "Which exact answer should be returned?"
_RAW_RENDERED_PROMPT = "<system>precise</system><user>question</user>"
_RAW_SYSTEM_PROMPT = "precise"
_PIN = ModelPin(
    repo_id="Qwen/Qwen3.8-27B",
    revision="0123456789abcdef",
    bundle_fingerprint=_sha("qwen-bundle"),
    bundle_manifest_sha256=_sha("qwen-bundle-manifest"),
    code_revision="c" * 40,
)
_VERIFIER = _sha("fertig-verifier")
_EVIDENCE = _sha("cold-path-evidence")
_EVALUATOR = _sha("frozen-evaluator-contract")
_FERTIG_UNIT = ProvenanceUnit(
    name="fertig",
    verifier_sha256=_VERIFIER,
    evidence_sha256=_EVIDENCE,
    receipt_sha256=_sha("fertig-receipt"),
)


def _binding(**changes: object) -> ResultCellBinding:
    values: dict[str, object] = {
        "model_pin": _PIN,
        "tokenizer_sha256": _sha("tokenizer.json"),
        "question_sha256": _sha(_RAW_QUESTION),
        "rendered_prompt_sha256": _sha(_RAW_RENDERED_PROMPT),
        "rendered_prompt_token_sha256": _sha("rendered-token-ids"),
        "system_prompt_sha256": _sha(_RAW_SYSTEM_PROMPT),
        "generation_policy_sha256": _sha("greedy-max-64-no-thinking"),
    }
    values.update(changes)
    return ResultCellBinding(**values)  # type: ignore[arg-type]


def _feature(
    binding: ResultCellBinding,
    *,
    model_pin_sha256: str | None = None,
    question_sha256: str | None = None,
    token_sha256: str | None = None,
    provenance_pair_sha256: str | None = None,
    verifier_sha256s: tuple[str, ...] | None = None,
    evidence_sha256s: tuple[str, ...] | None = None,
) -> QwenOoeFeatureReceipt:
    pair_binding = (
        provenance_pair_sha256
        or _generation_unit(binding=binding).pair_binding_sha256
    )
    return QwenOoeFeatureReceipt(
        temporal_index=7,
        measurement_sha256=_sha("measurement"),
        model_pin_sha256=model_pin_sha256 or binding.model_pin_sha256,
        weight_coordinate_sha256=_sha("weight-coordinate"),
        weight_graph_revision=GraphRevision(12, _sha("weight-graph")),
        atlas_graph_revision=GraphRevision(19, _sha("atlas-graph")),
        probe=ProbeIdentity(
            question_sha256=question_sha256 or binding.question_sha256,
            token_sha256=token_sha256 or binding.rendered_prompt_token_sha256,
            family_sha256=_sha("family"),
            label_source_sha256=_sha("label-source"),
        ),
        feature_schema_sha256=feature_schema_sha256(64),
        action_schema_sha256=ACTION_SCHEMA_SHA256,
        verifier_sha256s=verifier_sha256s or (pair_binding,),
        evidence_sha256s=evidence_sha256s or (pair_binding,),
        o1_surprise=0.5,
        o1_learning_progress=0.75,
        feature_sketch=(0.0,) * 64,
    )


def _raw_qwen_result(
    output: str = "42",
    *,
    forward_passes: int = 3,
    binding: ResultCellBinding | None = None,
) -> Result:
    exact_binding = binding or _binding()
    generation = {
        "context_mode": "stateful_autoregressive",
        "forward_passes": forward_passes,
        "general_generation": True,
        "generated_tokens": 2,
        "linear_calls": 128,
        "prefill_mode": "batched",
        "prompt_tokens": 12,
        "seconds": 0.25,
        "source_body_bytes": 4096,
        "state_bytes": 8192,
        "stateful_cache": True,
        "stopped_on_eos": True,
        "token_trace_sha256": _sha(f"token-trace:{output}"),
    }
    return Result(
        ExecutionStatus.OK,
        "qwen3.8.causal",
        output=output,
        evidence={
            "bundle": {
                "layout_fingerprint": exact_binding.model_pin.bundle_fingerprint,
                "manifest_sha256": (
                    exact_binding.model_pin.bundle_manifest_sha256
                ),
            },
            "generation": generation,
            "model": exact_binding.model_pin.repo_id,
            "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
            "revision": exact_binding.model_pin.revision,
            "tokenizer_sha256": exact_binding.tokenizer_sha256,
        },
    )


def _qwen_result(
    output: str = "42",
    *,
    forward_passes: int = 3,
    binding: ResultCellBinding | None = None,
) -> Result:
    exact_binding = binding or _binding()
    return attach_cold_qwen_generation_receipt(
        _raw_qwen_result(
            output,
            forward_passes=forward_passes,
            binding=exact_binding,
        ),
        binding=exact_binding,
    )


def _generation_unit(
    output: str = "42",
    *,
    forward_passes: int = 3,
    binding: ResultCellBinding | None = None,
) -> ProvenanceUnit:
    exact_binding = binding or _binding()
    return extract_cold_qwen_generation_receipt(
        _qwen_result(
            output,
            forward_passes=forward_passes,
            binding=exact_binding,
        ),
        binding=exact_binding,
    ).provenance_unit


def _final_result(
    output: str = "42",
    *,
    route: str = "qwen_verified",
) -> Result:
    return Result(
        ExecutionStatus.OK,
        "qwen-fertig-chat",
        output=output,
        evidence={"receipt": {"route": route}},
    )


def _judgment(status: str = "verified", *, expected: str | None = "42") -> dict:
    return {
        "candidate": "42",
        "evidence": {"question_sha256": _sha(_RAW_QUESTION)},
        "expected": expected,
        "kind": "fertig-candidate-verification/v1",
        "status": status,
    }


def _cell(
    *,
    binding: ResultCellBinding | None = None,
    output: str = "42",
    fertig_status: str = "verified",
    judgment: dict | None = None,
) -> ResultCell:
    exact_binding = binding or _binding()
    fertig_judgment = judgment or _judgment(fertig_status)
    final_route = {
        "abstained": "qwen_verification_abstained",
        "mismatch": "fertig_mismatch_override",
        "verified": "qwen_verified",
    }[fertig_status]
    return ResultCell.from_cold(
        binding=exact_binding,
        cold_qwen_result=_qwen_result(output, binding=exact_binding),
        cold_final_result=_final_result(output, route=final_route),
        cold_fertig_judgment=fertig_judgment,
        cold_fertig_status=fertig_status,
        evaluator_quality_contract_sha256=_EVALUATOR,
        provenance_units=(_FERTIG_UNIT,),
    )


def _runtime(binding: ResultCellBinding) -> dict[str, object]:
    return {
        "model_pin": binding.model_pin,
        "tokenizer_sha256": binding.tokenizer_sha256,
        "question_sha256": binding.question_sha256,
        "rendered_prompt_sha256": binding.rendered_prompt_sha256,
        "rendered_prompt_token_sha256": binding.rendered_prompt_token_sha256,
        "system_prompt_sha256": binding.system_prompt_sha256,
        "generation_policy_sha256": binding.generation_policy_sha256,
    }


class ResultCellPayloadTests(unittest.TestCase):
    def test_payload_is_canonical_immutable_complete_and_contains_no_raw_prompt(self) -> None:
        cell = _cell()
        encoded = cell.to_bytes()
        restored = ResultCell.from_bytes(encoded)

        self.assertEqual(restored, cell)
        self.assertEqual(restored.payload_sha256, cell.sha256)
        self.assertEqual(
            json.loads(encoded)["payload_sha256"],
            cell.payload_sha256,
        )
        self.assertEqual(restored.binding.model_pin, _PIN)
        self.assertEqual(restored.teacher_forward_count, 3)
        self.assertEqual(restored.cold_qwen_result, _qwen_result())
        self.assertEqual(restored.cold_final_result, _final_result())
        self.assertEqual(
            restored.cold_fertig_judgment_sha256,
            artifact_sha256(_judgment()),
        )
        self.assertNotIn(_RAW_QUESTION.encode(), encoded)
        self.assertNotIn(_RAW_RENDERED_PROMPT.encode(), encoded)
        self.assertNotIn(_RAW_SYSTEM_PROMPT.encode(), encoded)

        tampered = json.loads(encoded)
        tampered["body"]["cold_qwen_result"]["body"]["output"] = "41"
        with self.assertRaisesRegex(ResultCellIntegrityError, "payload SHA-256"):
            ResultCell.from_bytes(canonical_json_bytes(tampered))

    def test_binding_rejects_all_runtime_drift(self) -> None:
        binding = _binding()
        baseline = _runtime(binding)
        cases = (
            {"model_pin": replace(_PIN, revision="different")},
            {"tokenizer_sha256": _sha("other-tokenizer")},
            {"question_sha256": _sha("other-question")},
            {"rendered_prompt_sha256": _sha("other-rendering")},
            {"rendered_prompt_token_sha256": _sha("other-tokens")},
            {"system_prompt_sha256": _sha("other-system")},
            {"generation_policy_sha256": _sha("sample-temperature-1")},
        )
        for change in cases:
            with self.subTest(change=next(iter(change))):
                with self.assertRaises(ResultCellStaleError):
                    binding.assert_current(**{**baseline, **change})

        binding.assert_current(**baseline)

    def test_feature_binding_rejects_wrong_question_model_or_tokens(self) -> None:
        binding = _binding()
        for receipt in (
            _feature(binding, question_sha256=_sha("wrong-question")),
            _feature(binding, model_pin_sha256=_sha("wrong-model")),
            _feature(binding, token_sha256=_sha("wrong-tokens")),
        ):
            with self.assertRaises(ResultCellStaleError):
                binding.assert_feature_bound(receipt)

    def test_prompt_bearing_result_metadata_is_rejected_before_charge(self) -> None:
        prompt_bearing = Result(
            ExecutionStatus.OK,
            "qwen3.8.causal",
            output="42",
            evidence={
                "debug": {
                    "rendered_prompt": _RAW_RENDERED_PROMPT,
                    "question_sha256": _sha(_RAW_QUESTION),
                }
            },
        )
        with self.assertRaisesRegex(ValueError, "raw prompt metadata"):
            attach_cold_qwen_generation_receipt(
                prompt_bearing,
                binding=_binding(),
            )

        request_bearing = Result(
            ExecutionStatus.OK,
            "qwen3.8.causal",
            output="42",
            evidence={"request": {"message": _RAW_QUESTION}},
        )
        with self.assertRaisesRegex(ValueError, "raw prompt metadata"):
            attach_cold_qwen_generation_receipt(
                request_bearing,
                binding=_binding(),
            )

    def test_numeric_q4_request_metrics_are_not_prompt_metadata(self) -> None:
        binding = _binding()
        original = _raw_qwen_result(binding=binding)
        metrics = Result(
            original.status,
            original.component,
            output=original.output,
            reason=original.reason,
            evidence={
                **dict(original.evidence),
                "q4": {
                    "request": {
                        "head_calls": 4,
                        "linear_calls": 1408,
                    }
                },
            },
        )

        attached = attach_cold_qwen_generation_receipt(metrics, binding=binding)

        self.assertIn("cold_qwen_generation_receipt", attached.evidence)

    def test_hashed_fertig_status_cannot_disagree_with_claimed_status(self) -> None:
        with self.assertRaisesRegex(ValueError, "must match"):
            ResultCell.from_cold(
                binding=_binding(),
                cold_qwen_result=_qwen_result(),
                cold_final_result=_final_result(),
                cold_fertig_judgment=_judgment("abstained", expected=None),
                cold_fertig_status="verified",
                evaluator_quality_contract_sha256=_EVALUATOR,
                provenance_units=(_FERTIG_UNIT,),
            )

    def test_forward_baseline_is_derived_and_999999_override_is_rejected(self) -> None:
        raw = _raw_qwen_result(forward_passes=3)
        with self.assertRaisesRegex(ResultCellIntegrityError, "expected forward"):
            attach_cold_qwen_generation_receipt(
                raw,
                binding=_binding(),
                expected_forward_passes=999999,
            )
        with self.assertRaises(TypeError):
            attach_cold_qwen_generation_receipt(
                raw,
                binding=_binding(),
                **{"forward_passes": 999999},
            )

        cell = _cell()
        document = cell.to_document()
        document["body"]["teacher_forward_count"] = 999999
        document["payload_sha256"] = artifact_sha256(document["body"])
        with self.assertRaisesRegex(
            ResultCellIntegrityError,
            "teacher-forward baseline",
        ):
            ResultCell.from_bytes(canonical_json_bytes(document))

    def test_adapter_shaped_result_gets_one_exact_binding_receipt(self) -> None:
        binding = _binding()
        raw = _raw_qwen_result(binding=binding)
        self.assertNotIn(COLD_QWEN_BINDING_EVIDENCE_KEY, raw.evidence)
        attached = attach_cold_qwen_generation_receipt(raw, binding=binding)
        self.assertEqual(
            attached.evidence[COLD_QWEN_BINDING_EVIDENCE_KEY],
            qwen_result_binding_evidence(binding),
        )

        other = _binding(generation_policy_sha256=_sha("other-policy"))
        evidence = dict(raw.evidence)
        evidence[COLD_QWEN_BINDING_EVIDENCE_KEY] = qwen_result_binding_evidence(other)
        with self.assertRaisesRegex(ResultCellStaleError, "another execution binding"):
            attach_cold_qwen_generation_receipt(
                replace(raw, evidence=evidence),
                binding=binding,
            )

    def test_raw_generation_model_revision_bundle_and_schema_are_authenticated(self) -> None:
        raw = _raw_qwen_result()
        cases = []
        for field, value in (
            ("model", "Other/Model"),
            ("revision", "other-revision"),
            ("tokenizer_sha256", _sha("wrong-tokenizer")),
        ):
            evidence = dict(raw.evidence)
            evidence[field] = value
            cases.append(replace(raw, evidence=evidence))
        for field, value in (
            ("layout_fingerprint", _sha("wrong-layout")),
            ("manifest_sha256", _sha("wrong-manifest")),
        ):
            evidence = dict(raw.evidence)
            evidence["bundle"] = {**evidence["bundle"], field: value}
            cases.append(replace(raw, evidence=evidence))
        evidence = dict(raw.evidence)
        evidence["generation"] = {**evidence["generation"], "extra": 1}
        cases.append(replace(raw, evidence=evidence))

        for changed in cases:
            with self.subTest(evidence=changed.evidence):
                with self.assertRaises(ResultCellIntegrityError):
                    attach_cold_qwen_generation_receipt(
                        changed,
                        binding=_binding(),
                    )

    def test_cold_result_cannot_be_charged_under_another_exact_binding(self) -> None:
        cold = _qwen_result()
        cases = (
            _binding(tokenizer_sha256=_sha("other-tokenizer")),
            _binding(model_pin=replace(_PIN, code_revision="d" * 40)),
            _binding(rendered_prompt_sha256=_sha("other-rendered-prompt")),
            _binding(system_prompt_sha256=_sha("other-system-prompt")),
            _binding(generation_policy_sha256=_sha("other-generation-policy")),
        )
        for binding in cases:
            with self.subTest(binding=binding.sha256):
                with self.assertRaises(ResultCellStaleError):
                    ResultCell.from_cold(
                        binding=binding,
                        cold_qwen_result=cold,
                        cold_final_result=_final_result(),
                        cold_fertig_judgment=_judgment(),
                        cold_fertig_status="verified",
                        evaluator_quality_contract_sha256=_EVALUATOR,
                        provenance_units=(_FERTIG_UNIT,),
                    )


class ResultCellBankTests(unittest.TestCase):
    def test_charge_restore_idempotence_cas_and_immutable_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(Path(temporary) / "result-cells")
            bank = ResultCellBank(store)
            cell = _cell()

            first = bank.charge(cell)
            self.assertTrue(first.object_created)
            self.assertTrue(first.pointer_created)
            self.assertEqual(bank.restore(cell.binding), cell)
            self.assertEqual(bank.restore_payload(cell.payload_sha256), cell)

            second = bank.charge(cell)
            self.assertFalse(second.object_created)
            self.assertFalse(second.pointer_created)
            with self.assertRaises(ResultCellConflictError):
                bank.charge(
                    cell,
                    expected_current_payload_sha256=_sha("wrong-current"),
                )
            rebound = _cell(binding=cell.binding, output="43")
            with self.assertRaisesRegex(ResultCellConflictError, "cannot be rebound"):
                bank.charge(rebound)

            state_bytes = b"".join(
                path.read_bytes() for path in sorted((store.root / "state").iterdir())
            )
            self.assertNotIn(_RAW_QUESTION.encode(), state_bytes)
            self.assertNotIn(_RAW_RENDERED_PROMPT.encode(), state_bytes)

    def test_object_first_publication_recovers_after_pointer_crash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(Path(temporary) / "result-cells")
            bank = ResultCellBank(store)
            cell = _cell()
            original = store.publish_state
            failed = False

            def fail_pointer_once(name: str, payload: bytes, **kwargs):
                nonlocal failed
                if name == bank.pointer_state_name(cell.binding.sha256) and not failed:
                    failed = True
                    raise OSError("simulated process death before pointer publication")
                return original(name, payload, **kwargs)

            store.publish_state = fail_pointer_once  # type: ignore[method-assign]
            with self.assertRaisesRegex(OSError, "simulated process death"):
                bank.charge(cell)
            store.publish_state = original  # type: ignore[method-assign]

            publication = bank.charge(cell)
            self.assertFalse(publication.object_created)
            self.assertTrue(publication.pointer_created)
            self.assertEqual(bank.restore(cell.binding), cell)

    def test_tampered_content_object_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(Path(temporary) / "result-cells")
            bank = ResultCellBank(store)
            cell = _cell()
            bank.charge(cell)
            state_name = bank.object_state_name(cell.payload_sha256)
            path = store.root / "state" / store._state_filename(state_name)
            payload = bytearray(path.read_bytes())
            payload[-1] ^= 1
            path.chmod(0o600)
            path.write_bytes(payload)

            with self.assertRaises(ResultCellIntegrityError):
                bank.restore(cell.binding)

    def test_concurrent_idempotent_charge_has_one_visible_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = ResultCellBank(Path(temporary) / "result-cells")
            cell = _cell()
            publications = []
            failures = []

            def charge() -> None:
                try:
                    publications.append(bank.charge(cell))
                except Exception as exc:  # pragma: no cover - asserted below
                    failures.append(exc)

            threads = [threading.Thread(target=charge) for _index in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(failures, [])
            self.assertEqual(len(publications), 8)
            self.assertEqual(sum(row.pointer_created for row in publications), 1)
            self.assertEqual(bank.restore(cell.binding), cell)

    def test_tampered_binding_pointer_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(Path(temporary) / "result-cells")
            bank = ResultCellBank(store)
            cell = _cell()
            bank.charge(cell)
            state_name = bank.pointer_state_name(cell.binding.sha256)
            path = store.root / "state" / store._state_filename(state_name)
            payload = bytearray(path.read_bytes())
            payload[len(payload) // 2] ^= 1
            path.chmod(0o600)
            path.write_bytes(payload)

            with self.assertRaises(ResultCellIntegrityError):
                bank.restore(cell.binding)

    def test_persisted_prompt_metadata_is_rejected_again_on_restore(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(Path(temporary) / "result-cells")
            bank = ResultCellBank(store)
            cell = _cell()
            document = cell.to_document()
            qwen_document = document["body"]["cold_qwen_result"]
            qwen_document["body"]["evidence"]["rendered_prompt"] = (
                _RAW_RENDERED_PROMPT
            )
            qwen_document["sha256"] = artifact_sha256(qwen_document["body"])
            qwen_bytes = canonical_json_bytes(qwen_document)
            document["body"]["cold_qwen_result_sha256"] = hashlib.sha256(
                qwen_bytes
            ).hexdigest()
            payload_sha256 = artifact_sha256(document["body"])
            document["payload_sha256"] = payload_sha256
            store.publish_state(
                bank.object_state_name(payload_sha256),
                canonical_json_bytes(document),
            )
            pointer_body = {
                "binding_sha256": cell.binding.sha256,
                "payload_sha256": payload_sha256,
            }
            store.publish_state(
                bank.pointer_state_name(cell.binding.sha256),
                canonical_json_bytes(
                    {
                        "body": pointer_body,
                        "schema": RESULT_CELL_POINTER_SCHEMA,
                        "sha256": artifact_sha256(pointer_body),
                    }
                ),
            )

            with self.assertRaisesRegex(
                ResultCellIntegrityError,
                "result-cell validation failed",
            ):
                bank.restore(cell.binding)


class ResultCellExecutionTests(unittest.TestCase):
    def test_mixed_verifier_and_evidence_sets_cannot_synthesize_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = ResultCellBank(Path(temporary) / "bank")
            cell = _cell()
            bank.charge(cell)
            baseline = cell.cold_generation_provenance_unit
            wrong = ProvenanceUnit(
                name="wrong-generation",
                verifier_sha256=_sha("v2"),
                evidence_sha256=baseline.evidence_sha256,
                receipt_sha256=_sha("r2"),
            )
            receipts = (
                _feature(
                    cell.binding,
                    verifier_sha256s=(baseline.verifier_sha256,),
                    evidence_sha256s=(baseline.evidence_sha256,),
                ),
                _feature(
                    cell.binding,
                    verifier_sha256s=(baseline.pair_binding_sha256,),
                    evidence_sha256s=(wrong.pair_binding_sha256,),
                ),
                _feature(
                    cell.binding,
                    verifier_sha256s=(wrong.pair_binding_sha256,),
                    evidence_sha256s=(baseline.pair_binding_sha256,),
                ),
            )
            for receipt in receipts:
                with self.subTest(receipt=receipt.sha256):
                    with self.assertRaisesRegex(
                        ResultCellIntegrityError,
                        "exact cold-generation provenance unit",
                    ):
                        ResultCellExecutor(bank, cell.binding)(receipt)

    def test_cold_charge_to_warm_zero_forward_exact_parity_and_savings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = ResultCellBank(Path(temporary) / "bank")
            cell = _cell()
            bank.charge(cell)
            receipt = _feature(cell.binding)
            execution = ResultCellExecutor(bank, cell.binding)(receipt)

            self.assertEqual(execution.action, "mount_organ")
            self.assertEqual(execution.qwen_forwards, 0)
            self.assertEqual(execution.teacher_baseline_qwen_forwards, 3)
            self.assertEqual(execution.saved_qwen_forwards, 3)
            warm_qwen = result_from_document(execution.result)
            self.assertEqual(warm_qwen, cell.cold_qwen_result)
            warm_final = _final_result(route="ooe_verified")
            parity = ResultCellParityVerifier().verify(
                cell=cell,
                execution=execution,
                warm_qwen_result=warm_qwen,
                warm_final_result=warm_final,
                warm_fertig_judgment=_judgment(),
                evaluator_quality_contract_sha256=_EVALUATOR,
            )
            self.assertTrue(parity.exact_parity)
            self.assertTrue(parity.fertig_semantic_certified)
            self.assertEqual(parity.saved_qwen_forwards, 3)
            self.assertNotEqual(
                parity.cold_final_result_sha256,
                parity.warm_final_result_sha256,
            )
            self.assertNotEqual(
                parity.cold_final_evidence_sha256,
                parity.warm_final_evidence_sha256,
            )

    def test_wrong_runtime_binding_or_feature_cannot_mount(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = ResultCellBank(Path(temporary) / "bank")
            cell = _cell()
            bank.charge(cell)
            receipt = _feature(cell.binding)
            wrong_bindings = (
                replace(cell.binding, tokenizer_sha256=_sha("wrong-tokenizer")),
                replace(
                    cell.binding,
                    rendered_prompt_sha256=_sha("wrong-rendered-prompt"),
                ),
                replace(
                    cell.binding,
                    generation_policy_sha256=_sha("wrong-generation-policy"),
                ),
            )
            for wrong in wrong_bindings:
                with self.subTest(binding=wrong.sha256):
                    with self.assertRaises(ResultCellError):
                        ResultCellExecutor(bank, wrong)(receipt)

            with self.assertRaises(ResultCellStaleError):
                ResultCellExecutor(bank, cell.binding)(
                    _feature(cell.binding, question_sha256=_sha("wrong-question"))
                )

    def test_final_parity_rejects_result_judgment_policy_or_forward_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = ResultCellBank(Path(temporary) / "bank")
            cell = _cell()
            bank.charge(cell)
            execution = ResultCellExecutor(bank, cell.binding)(_feature(cell.binding))
            verifier = ResultCellParityVerifier()
            common = {
                "cell": cell,
                "execution": execution,
                "warm_qwen_result": cell.cold_qwen_result,
                "warm_final_result": _final_result(),
                "warm_fertig_judgment": _judgment(),
                "evaluator_quality_contract_sha256": _EVALUATOR,
            }
            failures = (
                {"warm_qwen_result": _qwen_result("41")},
                {"warm_final_result": _final_result("41")},
                {"warm_fertig_judgment": _judgment("abstained", expected=None)},
                {"evaluator_quality_contract_sha256": _sha("other-evaluator")},
                {"execution": replace(execution, qwen_forwards=1)},
            )
            for change in failures:
                with self.subTest(change=next(iter(change))):
                    with self.assertRaises(ResultCellParityError):
                        verifier.verify(**{**common, **change})

    def test_abstained_fertig_parity_is_not_semantic_certification(self) -> None:
        judgment = _judgment("abstained", expected=None)
        with tempfile.TemporaryDirectory() as temporary:
            bank = ResultCellBank(Path(temporary) / "bank")
            cell = _cell(fertig_status="abstained", judgment=judgment)
            bank.charge(cell)
            execution = ResultCellExecutor(bank, cell.binding)(_feature(cell.binding))
            warm_final = _final_result(route="ooe_verification_abstained")
            parity = ResultCellParityVerifier().verify(
                cell=cell,
                execution=execution,
                warm_qwen_result=cell.cold_qwen_result,
                warm_final_result=warm_final,
                warm_fertig_judgment=judgment,
                evaluator_quality_contract_sha256=_EVALUATOR,
            )

            self.assertTrue(parity.exact_parity)
            self.assertFalse(parity.fertig_semantic_certified)
            self.assertFalse(parity.fertig_exact_judgment)
            self.assertEqual(parity.saved_qwen_forwards, 3)

    def test_mismatch_is_exact_judgment_without_candidate_certification(self) -> None:
        judgment = {
            **_judgment("mismatch", expected="42"),
            "candidate": "41",
        }
        binding = _binding()
        cell = ResultCell.from_cold(
            binding=binding,
            cold_qwen_result=_qwen_result("41"),
            cold_final_result=_final_result("42", route="fertig_mismatch_override"),
            cold_fertig_judgment=judgment,
            cold_fertig_status="mismatch",
            evaluator_quality_contract_sha256=_EVALUATOR,
            provenance_units=(_FERTIG_UNIT,),
        )
        with tempfile.TemporaryDirectory() as temporary:
            bank = ResultCellBank(Path(temporary) / "bank")
            bank.charge(cell)
            execution = ResultCellExecutor(bank, binding)(
                _feature(
                    binding,
                    provenance_pair_sha256=(
                        cell.cold_generation_provenance_unit.pair_binding_sha256
                    ),
                )
            )
            parity = ResultCellParityVerifier().verify(
                cell=cell,
                execution=execution,
                warm_qwen_result=_qwen_result("41"),
                warm_final_result=_final_result(
                    "42",
                    route="ooe_fertig_mismatch_override",
                ),
                warm_fertig_judgment=judgment,
                evaluator_quality_contract_sha256=_EVALUATOR,
            )

        self.assertTrue(parity.fertig_exact_judgment)
        self.assertFalse(parity.fertig_semantic_certified)

    def test_frozen_evaluator_opens_only_in_final_benchmark_layer(self) -> None:
        calls: list[tuple[Result, Result]] = []

        def evaluator(cold: Result, warm: Result) -> bool:
            calls.append((cold, warm))
            return cold.output == warm.output and cold.status is warm.status

        with tempfile.TemporaryDirectory() as temporary:
            bank = ResultCellBank(Path(temporary) / "bank")
            cell = _cell()
            bank.charge(cell)
            execution = ResultCellExecutor(bank, cell.binding)(_feature(cell.binding))
            warm_final = _final_result(route="ooe_verified")
            layer = FinalBenchmarkLayer(
                evaluator_quality_contract_sha256=_EVALUATOR,
                evaluator=evaluator,
            )
            self.assertEqual(calls, [])

            parity = ResultCellParityVerifier().verify(
                cell=cell,
                execution=execution,
                warm_qwen_result=cell.cold_qwen_result,
                warm_final_result=warm_final,
                warm_fertig_judgment=_judgment(),
                evaluator_quality_contract_sha256=_EVALUATOR,
            )
            self.assertEqual(calls, [])

            benchmark = layer.verify(
                cell=cell,
                parity=parity,
                warm_final_result=warm_final,
            )
            self.assertEqual(len(calls), 1)
            self.assertTrue(benchmark.evaluator_quality_verified)
            self.assertEqual(benchmark.saved_qwen_forwards, 3)


if __name__ == "__main__":
    unittest.main()
