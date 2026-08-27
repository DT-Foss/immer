from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from immer.runtimes.ooe.qwen_mlp_evidence import (
    CAPTURE_STAGES,
    CaptureManifest,
    ExactMlpBoundaryCapture,
    MlpEvidenceBudget,
    MlpEvidenceJournalState,
    MlpEvidenceReceipt,
    MlpProjectionVerificationReceipt,
    QwenMlpEvidenceBank,
    QwenMlpEvidenceCapacityError,
    QwenMlpEvidenceConflictError,
    QwenMlpEvidenceIntegrityError,
    canonical_capture_plan,
)
from immer.runtimes.ooe.subspace_battery import graph_revision_sha256


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _manifest(prompts: int = 5) -> CaptureManifest:
    prompt_hashes = tuple(_hash(f"prompt:{index}") for index in range(prompts))
    return CaptureManifest(
        model_pin_sha256=_hash("model-pin"),
        input_manifest_sha256=_hash("input-manifest"),
        prompt_sha256s=prompt_hashes,
        entries=canonical_capture_plan(prompt_hashes),
    )


def _capture_pair(
    bank: QwenMlpEvidenceBank,
    manifest: CaptureManifest,
    index: int = 0,
    *,
    access_label: str = "access",
) -> tuple[MlpEvidenceReceipt, MlpProjectionVerificationReceipt]:
    entry = manifest.entries[index]
    sink = ExactMlpBoundaryCapture(bank)
    sink.begin_group(entry)
    rows = 3
    values = {
        "mlp.input": torch.arange(rows * 4, dtype=torch.float32)
        .reshape(1, rows, 4)
        .to(torch.bfloat16),
        "mlp.gate": torch.arange(rows * 6, dtype=torch.float32)
        .reshape(1, rows, 6)
        .to(torch.bfloat16),
        "mlp.up": (torch.arange(rows * 6, dtype=torch.float32) + 1)
        .reshape(1, rows, 6)
        .to(torch.bfloat16),
        "mlp.output": (torch.arange(rows * 4, dtype=torch.float32) - 2)
        .reshape(1, rows, 4)
        .to(torch.bfloat16),
    }
    for stage in CAPTURE_STAGES:
        sink(entry.layer, stage, values[stage])
    event = _hash(f"atlas-event:{entry.ordinal}")
    sketch = _hash(f"input-sketch:{entry.ordinal}")
    receipt = sink.finalize(
        capture_mode="live-exact",
        manifest_sha256=manifest.sha256,
        model_pin_sha256=manifest.model_pin_sha256,
        token_sha256=_hash(f"tokens:{entry.prompt_sha256}"),
        probe_spec_sha256=_hash(f"probe-spec:{entry.ordinal}"),
        atlas_sequence=entry.ordinal,
        atlas_event_sha256=event,
        atlas_revision_sha256=graph_revision_sha256(entry.ordinal, event),
        measurement_sha256=_hash(f"measurement:{entry.ordinal}"),
        weight_revision_sha256=_hash("weight-revision"),
        access_trace_sha256=_hash(f"{access_label}:{entry.ordinal}"),
        source_receipt_sha256s=tuple(
            sorted(
                (
                    _hash(f"gate-range:{entry.ordinal}"),
                    _hash(f"up-range:{entry.ordinal}"),
                    _hash(f"down-range:{entry.ordinal}"),
                )
            )
        ),
        cartography_input_sketch_sha256=sketch,
        recomputed_input_sketch_sha256=sketch,
    )
    verification = MlpProjectionVerificationReceipt(
        evidence_receipt_sha256=receipt.sha256,
        verifier_sha256=_hash("exact-mlp-replay-verifier"),
        verifier_evidence_sha256=_hash(f"verifier-evidence:{receipt.sha256}"),
        replay_access_trace_sha256=_hash(f"replay-trace:{receipt.sha256}"),
        storage_exact=True,
        input_sketch_exact=True,
        gate_exact=True,
        up_exact=True,
        output_exact=True,
    )
    return receipt, verification


class _ReceiptVerifier:
    def __init__(self, verification: MlpProjectionVerificationReceipt) -> None:
        self.verification = verification
        self.verifier_sha256 = verification.verifier_sha256

    def verify(
        self, receipt: MlpEvidenceReceipt, bank: QwenMlpEvidenceBank
    ) -> MlpProjectionVerificationReceipt:
        if receipt.sha256 != self.verification.evidence_receipt_sha256:
            raise AssertionError("test verifier received another receipt")
        verification = self.verification
        proof = bank.publish_verifier_evidence(
            evidence_receipt_sha256=receipt.sha256,
            verifier_sha256=self.verifier_sha256,
            evidence={
                "gate_exact": verification.gate_exact,
                "input_sketch_exact": verification.input_sketch_exact,
                "kind": "test-exact-replay",
                "output_exact": verification.output_exact,
                "receipt": receipt.sha256,
                "replay_access_trace_sha256": (verification.replay_access_trace_sha256),
                "storage_exact": verification.storage_exact,
                "tensor_object_sha256s": [row.object_sha256 for row in receipt.tensors],
                "up_exact": verification.up_exact,
            },
        )
        return replace(verification, verifier_evidence_sha256=proof)


def _verifier(
    verification: MlpProjectionVerificationReceipt,
) -> _ReceiptVerifier:
    return _ReceiptVerifier(verification)


class QwenMlpEvidenceTests(unittest.TestCase):
    def test_bfloat16_binary_roundtrip_row_hashes_and_no_replace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = QwenMlpEvidenceBank(temporary)
            value = torch.tensor([[[1.0, -2.5], [3.25, 4.0]]], dtype=torch.bfloat16)
            first = bank.publish_tensor("mlp.output", 18, value)
            second = bank.publish_tensor("mlp.output", 18, value.clone())
            self.assertEqual(first, second)
            self.assertEqual(first.source_dtype, "bfloat16")
            self.assertEqual(first.storage_dtype, "bfloat16-bits-le")
            self.assertEqual(len(first.row_sha256s), 2)
            expected = value.float().numpy().astype(np.float64)
            np.testing.assert_array_equal(bank.restore_tensor(first), expected)
            objects = list((Path(temporary) / "objects").glob("*/*.tensor"))
            self.assertEqual(len(objects), 1)

    def test_capture_order_budget_symlink_and_tensor_tamper_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = QwenMlpEvidenceBank(
                temporary,
                budget=MlpEvidenceBudget(max_tensor_bytes=128, max_group_bytes=256),
            )
            manifest = _manifest()
            sink = ExactMlpBoundaryCapture(bank)
            sink.begin_group(manifest.entries[0])
            with self.assertRaisesRegex(QwenMlpEvidenceIntegrityError, "order"):
                sink(manifest.entries[0].layer, "mlp.gate", np.ones((1, 1, 2)))
            sink = ExactMlpBoundaryCapture(bank)
            sink.begin_group(manifest.entries[0])
            with self.assertRaises(QwenMlpEvidenceCapacityError):
                sink(
                    manifest.entries[0].layer,
                    "mlp.input",
                    np.ones((1, 64, 4), dtype=np.float64),
                )
            ref = bank.publish_tensor(
                "mlp.input", 18, np.ones((1, 2, 2), dtype=np.float32)
            )
            path = next((Path(temporary) / "objects").glob("*/*.tensor"))
            original = path.read_bytes()
            path.chmod(0o600)
            path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            with self.assertRaises(QwenMlpEvidenceIntegrityError):
                bank.restore_tensor(ref)
            path.unlink()
            path.symlink_to(Path(temporary) / "HEAD")
            with self.assertRaises((OSError, QwenMlpEvidenceIntegrityError)):
                bank.restore_tensor(ref)

    def test_verified_append_idempotency_cas_rebind_and_subspace_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = QwenMlpEvidenceBank(temporary)
            manifest = _manifest()
            receipt, verification = _capture_pair(bank, manifest)
            initial = bank.state()
            publication = bank.append_verified(
                receipt, _verifier(verification), expected_head_sha256=initial.sha256
            )
            self.assertTrue(publication.changed)
            self.assertEqual(publication.generation, 1)
            replay = bank.append_verified(receipt, _verifier(verification))
            self.assertFalse(replay.changed)
            with self.assertRaisesRegex(QwenMlpEvidenceConflictError, "CAS"):
                bank.append_verified(
                    receipt,
                    _verifier(verification),
                    expected_head_sha256=_hash("stale-head"),
                )
            self.assertTrue(bank.audit().clean)
            conflicting, conflicting_verification = _capture_pair(
                bank, manifest, access_label="different-access"
            )
            with self.assertRaisesRegex(QwenMlpEvidenceConflictError, "rebound"):
                bank.append_verified(conflicting, _verifier(conflicting_verification))
            audit = bank.audit()
            self.assertFalse(audit.clean)
            self.assertEqual(audit.receipt_count, 1)
            self.assertEqual(len(audit.orphan_receipts), 1)
            self.assertEqual(len(audit.orphan_verifications), 1)
            self.assertEqual(len(audit.orphan_verifier_evidence), 1)
            corpus = bank.build_subspace_corpus()
            self.assertEqual(len(corpus.groups), 1)
            self.assertEqual(
                corpus.groups[0].output_payload_sha256s, receipt.tensors[-1].row_sha256s
            )

    def test_false_verifier_and_receipt_roundtrip_tamper_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = QwenMlpEvidenceBank(temporary)
            receipt, verification = _capture_pair(bank, _manifest())
            self.assertEqual(MlpEvidenceReceipt.from_bytes(receipt.to_bytes()), receipt)
            self.assertEqual(
                MlpProjectionVerificationReceipt.from_bytes(verification.to_bytes()),
                verification,
            )
            false = MlpProjectionVerificationReceipt(
                evidence_receipt_sha256=receipt.sha256,
                verifier_sha256=verification.verifier_sha256,
                verifier_evidence_sha256=verification.verifier_evidence_sha256,
                replay_access_trace_sha256=verification.replay_access_trace_sha256,
                storage_exact=True,
                input_sketch_exact=True,
                gate_exact=True,
                up_exact=True,
                output_exact=False,
            )
            with self.assertRaisesRegex(QwenMlpEvidenceIntegrityError, "not exact"):
                bank.append_verified(receipt, _verifier(false))
            wrong_identity = _verifier(verification)
            wrong_identity.verifier_sha256 = _hash("another-verifier")
            with self.assertRaisesRegex(
                QwenMlpEvidenceIntegrityError, "identity differs"
            ):
                bank.append_verified(receipt, wrong_identity)

            class DishonestProofVerifier:
                verifier_sha256 = verification.verifier_sha256

                def verify(self, value, target_bank):
                    proof = target_bank.publish_verifier_evidence(
                        evidence_receipt_sha256=value.sha256,
                        verifier_sha256=self.verifier_sha256,
                        evidence={
                            "gate_exact": False,
                            "input_sketch_exact": False,
                            "output_exact": False,
                            "replay_access_trace_sha256": _hash("wrong-trace"),
                            "storage_exact": False,
                            "tensor_object_sha256s": [],
                            "up_exact": False,
                        },
                    )
                    return replace(verification, verifier_evidence_sha256=proof)

            with self.assertRaisesRegex(QwenMlpEvidenceIntegrityError, "proof differs"):
                bank.append_verified(receipt, DishonestProofVerifier())

    def test_crash_recovery_at_every_append_boundary(self) -> None:
        for fault_stage in (
            "after-receipt-verification",
            "after-history",
            "after-intent",
            "after-head",
            "after-commit",
        ):
            with (
                self.subTest(fault_stage=fault_stage),
                tempfile.TemporaryDirectory() as temporary,
            ):
                fired = False

                def fault(stage: str) -> None:
                    nonlocal fired
                    if stage == fault_stage and not fired:
                        fired = True
                        raise RuntimeError(stage)

                bank = QwenMlpEvidenceBank(temporary, fault_injector=fault)
                manifest = _manifest()
                receipt, verification = _capture_pair(bank, manifest)
                with self.assertRaisesRegex(RuntimeError, fault_stage):
                    bank.append_verified(receipt, _verifier(verification))
                restarted = QwenMlpEvidenceBank(temporary)
                state = restarted.state()
                if fault_stage in {"after-receipt-verification", "after-history"}:
                    self.assertEqual(state.generation, 0)
                    restarted.append_verified(receipt, _verifier(verification))
                else:
                    self.assertEqual(state.generation, 1)
                self.assertEqual(restarted.state().generation, 1)
                self.assertEqual(restarted.audit().receipt_count, 1)

    def test_root_state_roundtrip(self) -> None:
        root = MlpEvidenceJournalState.initial()
        self.assertEqual(MlpEvidenceJournalState.from_bytes(root.to_bytes()), root)

    def test_bank_budget_is_pinned_across_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            QwenMlpEvidenceBank(
                temporary,
                budget=MlpEvidenceBudget(max_total_referenced_bytes=1024),
            )
            with self.assertRaisesRegex(QwenMlpEvidenceConflictError, "budget changed"):
                QwenMlpEvidenceBank(temporary)

    def test_production_corpus_rejects_mixed_live_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = QwenMlpEvidenceBank(temporary)
            first_manifest = _manifest()
            prompts = tuple(_hash(f"other-prompt:{index}") for index in range(5))
            second_manifest = CaptureManifest(
                _hash("model-pin"),
                _hash("other-input-manifest"),
                prompts,
                canonical_capture_plan(prompts),
            )
            first, first_verification = _capture_pair(bank, first_manifest)
            second, second_verification = _capture_pair(bank, second_manifest)
            bank.append_verified(first, _verifier(first_verification))
            bank.append_verified(second, _verifier(second_verification))
            with self.assertRaisesRegex(
                QwenMlpEvidenceIntegrityError,
                "one live manifest/model/verifier pin",
            ):
                bank.build_subspace_corpus()

    def test_production_corpus_rejects_mixed_verifier_pins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = QwenMlpEvidenceBank(temporary)
            manifest = _manifest()
            first, first_verification = _capture_pair(bank, manifest, index=0)
            second, second_verification = _capture_pair(bank, manifest, index=1)
            second_verification = replace(
                second_verification, verifier_sha256=_hash("second-verifier")
            )
            bank.append_verified(first, _verifier(first_verification))
            bank.append_verified(second, _verifier(second_verification))
            with self.assertRaisesRegex(
                QwenMlpEvidenceIntegrityError,
                "one live manifest/model/verifier pin",
            ):
                bank.build_subspace_corpus()

    def test_audit_clean_is_derived_from_object_and_history_orphans(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = QwenMlpEvidenceBank(temporary)
            orphan = bank.publish_tensor(
                "mlp.input", 18, np.ones((1, 1, 2), dtype=np.float32)
            )
            audit = bank.audit()
            self.assertFalse(audit.clean)
            self.assertIn(orphan.object_sha256, audit.orphan_objects)
            self.assertFalse(audit.orphan_histories)
            self.assertFalse(audit.orphan_receipts)
            self.assertFalse(audit.orphan_verifications)
            self.assertFalse(audit.orphan_verifier_evidence)

        with tempfile.TemporaryDirectory() as temporary:
            fired = False

            def fault(stage: str) -> None:
                nonlocal fired
                if stage == "after-history" and not fired:
                    fired = True
                    raise RuntimeError(stage)

            bank = QwenMlpEvidenceBank(temporary, fault_injector=fault)
            receipt, verification = _capture_pair(bank, _manifest())
            with self.assertRaisesRegex(RuntimeError, "after-history"):
                bank.append_verified(receipt, _verifier(verification))
            audit = QwenMlpEvidenceBank(temporary).audit()
            self.assertFalse(audit.clean)
            self.assertEqual(len(audit.orphan_histories), 1)
            self.assertEqual(len(audit.orphan_receipts), 1)
            self.assertEqual(len(audit.orphan_verifications), 1)
            self.assertEqual(len(audit.orphan_verifier_evidence), 1)
            self.assertGreaterEqual(len(audit.orphan_objects), 4)

    def test_interrupted_prejournal_tensor_capture_self_heals_on_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = QwenMlpEvidenceBank(temporary)
            partial = (
                torch.arange(12, dtype=torch.float32)
                .reshape(1, 3, 4)
                .to(torch.bfloat16)
            )
            orphan = bank.publish_tensor("mlp.input", 45, partial)
            self.assertIn(orphan.object_sha256, bank.audit().orphan_objects)
            receipt, verification = _capture_pair(bank, _manifest())
            bank.append_verified(receipt, _verifier(verification))
            self.assertTrue(bank.audit().clean)


if __name__ == "__main__":
    unittest.main()
