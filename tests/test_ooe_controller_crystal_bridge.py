from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from immer.runtimes.ooe.compute_crystals import ComputeCrystalBank
from immer.runtimes.ooe.controller import (
    ControllerConfig,
    OoeController,
    VerifiedTeacherTransition,
)
from immer.runtimes.ooe.controller_crystal_bridge import (
    CONTROLLER_CRYSTAL_CONTEXT_SCHEMA_SHA256,
    CONTROLLER_CRYSTAL_EXPORT_STATE_PREFIX,
    CONTROLLER_CRYSTAL_LANGUAGE_REWARD_POLICY_SHA256,
    CONTROLLER_CRYSTAL_SOURCE_EXTENSION,
    CONTROLLER_CRYSTAL_TRANSACTION_STATE_PREFIX,
    ControllerCrystalBridgeIntegrityError,
    ControllerCrystalBridgeStaleError,
    ControllerCrystalExportReceipt,
    controller_site_action_id,
    export_controller_crystals,
    restore_controller_crystal_export,
    restore_controller_crystal_transaction,
)
from immer.runtimes.ooe.crystal import CrystalPayload, CrystalStore
from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.qwen_bridge import (
    ACTION_SCHEMA_SHA256,
    OOE_ACTIONS,
    QwenOoeFeatureReceipt,
    feature_schema_sha256,
)
from immer.runtimes.qwen3_8.semantic_atlas import GraphRevision, ProbeIdentity


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_MODEL_PIN = _sha("controller-crystal-model-pin")
_WEIGHT_GRAPH = GraphRevision(7, _sha("controller-crystal-weight-graph"))
_ATLAS_GRAPH = GraphRevision(11, _sha("controller-crystal-atlas-graph"))
_VERIFIER = _sha("controller-crystal-verifier")


def _feature(temporal: int, *, site: int, source: int) -> QwenOoeFeatureReceipt:
    sketch = [0.0] * 8
    sketch[site] = 0.75
    sketch[2 + source] = 0.1 + source * 0.02
    return QwenOoeFeatureReceipt(
        temporal_index=temporal,
        measurement_sha256=_sha(f"measurement-{site}-{source}-{temporal}"),
        model_pin_sha256=_MODEL_PIN,
        weight_coordinate_sha256=_sha(f"coordinate-{site}"),
        weight_graph_revision=_WEIGHT_GRAPH,
        atlas_graph_revision=_ATLAS_GRAPH,
        probe=ProbeIdentity(
            question_sha256=_sha(f"question-{site}-{source}-{temporal}"),
            token_sha256=_sha(f"tokens-{site}-{source}-{temporal}"),
            family_sha256=_sha(f"family-{site}"),
            label_source_sha256=_sha("label-source"),
        ),
        feature_schema_sha256=feature_schema_sha256(8),
        action_schema_sha256=ACTION_SCHEMA_SHA256,
        verifier_sha256s=(_VERIFIER,),
        evidence_sha256s=(_sha(f"evidence-{site}-{source}-{temporal}"),),
        o1_surprise=0.25,
        o1_learning_progress=0.5,
        feature_sketch=tuple(sketch),
    )


def _transition(
    receipt: QwenOoeFeatureReceipt,
    *,
    source: int,
    target: int,
) -> VerifiedTeacherTransition:
    return VerifiedTeacherTransition(
        feature_receipt_sha256=receipt.sha256,
        site_identity_sha256=receipt.site_identity.sha256,
        source_action=OOE_ACTIONS[source],
        target_action=OOE_ACTIONS[target],
        verifier_sha256=_VERIFIER,
        evidence_sha256=receipt.evidence_sha256s[0],
        quality_sha256=_sha(f"quality-{receipt.sha256}"),
    )


class ControllerCrystalBridgeTests(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
        *,
        sites: int = 2,
        promote_all: bool = True,
    ) -> tuple[OoeController, ComputeCrystalBank, int]:
        source = CrystalStore(root / "controller")
        controller = OoeController(
            model_pin_sha256=_MODEL_PIN,
            weight_graph_revision_sha256=_WEIGHT_GRAPH.sha256,
            atlas_graph_revision=_ATLAS_GRAPH,
            crystal_store=source,
            config=ControllerConfig(
                replicas=4,
                replica_fanout=4,
                min_coverage_per_source=1,
                min_promoted_sources=len(OOE_ACTIONS),
                router_radius=2.0,
                router_min_margin=0.0,
                token_min_confidence=0.01,
                consensus_tolerance=1e-6,
                consensus_max_rounds=1024,
                quantization_levels=65535,
                reservoir_size=8,
            ),
        )
        temporal = 0
        identities: list[str] = []
        for site in range(sites):
            for source_index in range(len(OOE_ACTIONS)):
                receipt = _feature(temporal, site=site, source=source_index)
                controller.ingest_teacher(
                    receipt,
                    _transition(
                        receipt,
                        source=source_index,
                        target=(source_index + site + 1) % len(OOE_ACTIONS),
                    ),
                )
                temporal += 1
            identities.append(receipt.site_identity.sha256)
        promoted = identities if promote_all else identities[:-1]
        for site_sha256 in promoted:
            coverage = controller.coverage_receipt(site_sha256)
            controller.promote(
                site_sha256,
                coverage_sha256=coverage.sha256,
                verifier_sha256s=coverage.verifier_sha256s,
            )
        controller.save_snapshot()
        restored = OoeController.restore(crystal_store=source)
        bank = ComputeCrystalBank(CrystalStore(root / "compute-bank"))
        return restored, bank, temporal

    def test_exact_export_frontier_receipt_and_bank_restore(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller, bank, _ = self._fixture(Path(temporary))
            result = export_controller_crystals(
                controller,
                bank,
                expected_controller_snapshot_sha256=hashlib.sha256(
                    controller.snapshot_bytes()
                ).hexdigest(),
                expected_model_pin_sha256=_MODEL_PIN,
                expected_weight_graph_revision_sha256=_WEIGHT_GRAPH.sha256,
                expected_atlas_graph_revision_sha256=_ATLAS_GRAPH.sha256,
                expected_source_manifest_sha256=(
                    controller.crystal_store.manifest().sha256
                ),
                expected_compute_bank_anchor_sha256=bank.manifest().sha256,
            )

            self.assertEqual(len(result.crystals), 2)
            self.assertEqual(
                result.frontier.context_schema_sha256,
                CONTROLLER_CRYSTAL_CONTEXT_SCHEMA_SHA256,
            )
            authorities = dict(result.frontier.authority_hashes)
            self.assertEqual(
                set(authorities),
                {
                    "compute-bank-anchor",
                    "controller-atlas-graph",
                    "controller-calibration",
                    "controller-crystal-manifest",
                    "controller-model-pin",
                    "controller-state",
                    "controller-weight-graph",
                    "language-reward-policy",
                },
            )
            self.assertEqual(
                authorities["language-reward-policy"],
                CONTROLLER_CRYSTAL_LANGUAGE_REWARD_POLICY_SHA256,
            )
            self.assertEqual(
                authorities["compute-bank-anchor"],
                result.receipt.final_bank_anchor_sha256,
            )
            self.assertEqual(
                result.frontier.action_ids,
                tuple(
                    controller_site_action_id(site.site_identity.sha256)
                    for site in result.receipt.sites
                ),
            )

            for site, crystal in zip(
                result.receipt.sites, result.crystals, strict=True
            ):
                source = controller.crystal_store.restore(site.source_payload_sha256)
                self.assertTrue(
                    np.array_equal(
                        crystal.apply(np.eye(len(OOE_ACTIONS))),
                        source.restore_kernel(),
                    )
                )
                extension = crystal.extensions[CONTROLLER_CRYSTAL_SOURCE_EXTENSION]
                self.assertEqual(
                    extension["controller_snapshot_sha256"],
                    result.receipt.controller_snapshot_sha256,
                )
                self.assertEqual(extension["source_payload_sha256"], source.sha256)
                self.assertEqual(
                    extension["verifier_hashes"], dict(source.verifier_hashes)
                )
                self.assertEqual(
                    extension["evidence_hashes"], dict(source.evidence_hashes)
                )

            encoded = result.receipt.to_bytes()
            self.assertEqual(
                ControllerCrystalExportReceipt.from_bytes(encoded), result.receipt
            )
            reopened = ComputeCrystalBank(
                bank.store,
                trusted_manifest_sha256=result.receipt.final_bank_anchor_sha256,
            )
            restored = restore_controller_crystal_export(
                reopened, result.receipt.sha256
            )
            self.assertEqual(restored.receipt, result.receipt)
            self.assertEqual(
                tuple(crystal.to_bytes() for crystal in restored.crystals),
                tuple(crystal.to_bytes() for crystal in result.crystals),
            )
            result.receipt.assert_current(controller, reopened)

    def test_legacy_pretransaction_receipt_roundtrips_without_weakening_new(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller, bank, _ = self._fixture(Path(temporary))
            result = export_controller_crystals(controller, bank)
            self.assertFalse(result.receipt.is_legacy)
            self.assertIsNotNone(result.receipt.transaction_plan_sha256)
            self.assertIn(
                "transaction_plan_sha256",
                result.receipt.to_document()["body"],
            )

            legacy_sites = tuple(
                replace(site, object_created=True) for site in result.receipt.sites
            )
            legacy = replace(
                result.receipt,
                sites=legacy_sites,
                transaction_plan_sha256=None,
            )
            legacy_bytes = legacy.to_bytes()
            self.assertTrue(legacy.is_legacy)
            self.assertNotIn(
                "transaction_plan_sha256",
                legacy.to_document()["body"],
            )
            self.assertEqual(
                ControllerCrystalExportReceipt.from_bytes(legacy_bytes).to_bytes(),
                legacy_bytes,
            )
            self.assertEqual(
                hashlib.sha256(legacy_bytes).hexdigest(),
                legacy.sha256,
            )
            bank.store.publish_state(legacy.state_name, legacy_bytes)
            restored = restore_controller_crystal_export(bank, legacy.sha256)
            self.assertTrue(restored.receipt.is_legacy)
            self.assertEqual(restored.receipt.to_bytes(), legacy_bytes)
            restored.receipt.assert_current(controller, bank)

            with self.assertRaisesRegex(
                ControllerCrystalBridgeIntegrityError,
                "requires transaction metadata",
            ):
                replace(result.receipt, transaction_plan_sha256=None)

    def test_resealed_receipt_tamper_and_unpromoted_site_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller, bank, _ = self._fixture(Path(temporary))
            result = export_controller_crystals(controller, bank)
            document = json.loads(result.receipt.to_bytes())
            document["body"]["sites"][0]["action_id"] = "controller-site:" + _sha(
                "forged-site"
            )
            document["body_sha256"] = hashlib.sha256(
                canonical_json_bytes(document["body"])
            ).hexdigest()
            with self.assertRaises(ControllerCrystalBridgeIntegrityError):
                ControllerCrystalExportReceipt.from_bytes(
                    canonical_json_bytes(document)
                )

        with tempfile.TemporaryDirectory() as temporary:
            controller, bank, _ = self._fixture(Path(temporary), promote_all=False)
            with self.assertRaisesRegex(
                ControllerCrystalBridgeIntegrityError, "not promoted"
            ):
                export_controller_crystals(controller, bank)

    def test_changed_snapshot_source_store_and_expected_pins_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller, bank, temporal = self._fixture(Path(temporary))
            result = export_controller_crystals(controller, bank)
            next_receipt = _feature(temporal, site=0, source=0)
            controller.ingest_teacher(
                next_receipt,
                _transition(next_receipt, source=0, target=2),
            )
            with self.assertRaisesRegex(
                ControllerCrystalBridgeStaleError, "snapshot changed"
            ):
                result.receipt.assert_current(controller, bank)

        with tempfile.TemporaryDirectory() as temporary:
            controller, bank, _ = self._fixture(Path(temporary))
            source = controller.crystal_store.restore(
                controller.crystal_store.manifest().entries[0].payload_sha256
            )
            extra_identity = replace(
                source.identity,
                weight_coordinate_sha256=_sha("extra-coordinate"),
            )
            extra = CrystalPayload.from_kernel(
                name=extra_identity.sha256,
                identity=extra_identity,
                kernel=np.eye(len(OOE_ACTIONS)),
                coverage_sha256=_sha("extra-coverage"),
                calibration_sha256=source.calibration_sha256,
                verifier_hashes={"verifier": _VERIFIER},
                evidence_hashes={"evidence": _sha("extra-evidence")},
                consensus_receipt={"result": _sha("extra-consensus")},
            )
            controller.crystal_store.publish(extra)
            with self.assertRaises(ControllerCrystalBridgeStaleError):
                export_controller_crystals(controller, bank)

        with tempfile.TemporaryDirectory() as temporary:
            controller, bank, _ = self._fixture(Path(temporary))
            with self.assertRaisesRegex(
                ControllerCrystalBridgeStaleError,
                "expected_model_pin_sha256 is stale",
            ):
                export_controller_crystals(
                    controller,
                    bank,
                    expected_model_pin_sha256=_sha("wrong-model"),
                )

    def test_source_object_tamper_is_detected_after_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller, bank, _ = self._fixture(Path(temporary))
            result = export_controller_crystals(controller, bank)
            digest = result.receipt.sites[0].source_payload_sha256
            path = controller.crystal_store.root / "objects" / f"{digest}.crystal"
            original = path.read_bytes()
            path.chmod(0o600)
            path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            with self.assertRaises(ControllerCrystalBridgeIntegrityError):
                result.receipt.assert_current(controller, bank)

    def test_concurrent_controller_mutation_is_detected_after_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller, bank, temporal = self._fixture(Path(temporary))
            original_publish = bank.publish_crystal
            calls = 0

            def publish_then_mutate(crystal, *, expected_generation=None):
                nonlocal calls
                publication = original_publish(
                    crystal,
                    expected_generation=expected_generation,
                )
                calls += 1
                if calls == 1:
                    receipt = _feature(temporal, site=0, source=0)
                    controller.ingest_teacher(
                        receipt,
                        _transition(receipt, source=0, target=3),
                    )
                return publication

            with patch.object(
                bank,
                "publish_crystal",
                side_effect=publish_then_mutate,
            ):
                with self.assertRaisesRegex(
                    ControllerCrystalBridgeStaleError,
                    "changed during compute publication",
                ):
                    export_controller_crystals(controller, bank)
            self.assertEqual(calls, 2)
            restored_controller = OoeController.restore(
                crystal_store=controller.crystal_store
            )
            recovered = export_controller_crystals(restored_controller, bank)
            self.assertEqual(len(recovered.crystals), 2)
            recovered.receipt.assert_current(restored_controller, bank)

    def test_preflight_rejects_undersized_target_without_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller, _, _ = self._fixture(root)
            bank = ComputeCrystalBank(
                CrystalStore(root / "undersized-bank", max_state_bytes=1024)
            )
            before = bank.manifest()
            with self.assertRaisesRegex(
                ControllerCrystalBridgeIntegrityError,
                "state capacity before publication",
            ):
                export_controller_crystals(controller, bank)
            self.assertEqual(bank.manifest(), before)
            self.assertEqual(tuple((root / "undersized-bank" / "state").iterdir()), ())

    def test_checkpoint_fault_resumes_exactly_after_bank_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller, bank, _ = self._fixture(Path(temporary))
            original_publish_state = bank.store.publish_state
            transaction_writes = 0

            def fail_first_checkpoint(
                name,
                payload,
                *,
                expected_sha256=None,
            ):
                nonlocal transaction_writes
                if name.startswith(CONTROLLER_CRYSTAL_TRANSACTION_STATE_PREFIX):
                    transaction_writes += 1
                    if transaction_writes == 2:
                        raise OSError("simulated transaction checkpoint crash")
                return original_publish_state(
                    name,
                    payload,
                    expected_sha256=expected_sha256,
                )

            with patch.object(
                bank.store,
                "publish_state",
                side_effect=fail_first_checkpoint,
            ):
                with self.assertRaisesRegex(
                    ControllerCrystalBridgeIntegrityError,
                    "retry resumes it",
                ):
                    export_controller_crystals(controller, bank)
            self.assertEqual(bank.manifest().generation, 1)

            recovered = export_controller_crystals(controller, bank)
            self.assertEqual(recovered.receipt.start_bank_generation, 0)
            self.assertEqual(recovered.receipt.final_bank_generation, 2)
            self.assertEqual(bank.manifest().generation, 2)
            repeated = export_controller_crystals(controller, bank)
            self.assertEqual(repeated.receipt, recovered.receipt)
            self.assertEqual(bank.manifest().generation, 2)
            transaction = restore_controller_crystal_transaction(
                bank,
                recovered.receipt.transaction_plan_sha256,
            )
            self.assertEqual(transaction.status, "committed")
            self.assertEqual(
                transaction.committed_receipt_sha256,
                recovered.receipt.sha256,
            )

    def test_receipt_persistence_fault_resumes_and_commits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller, bank, _ = self._fixture(Path(temporary))
            original_publish_state = bank.store.publish_state
            failed = False

            def fail_receipt_once(name, payload, *, expected_sha256=None):
                nonlocal failed
                if (
                    name.startswith(CONTROLLER_CRYSTAL_EXPORT_STATE_PREFIX)
                    and not failed
                ):
                    failed = True
                    raise OSError("simulated receipt persistence crash")
                return original_publish_state(
                    name,
                    payload,
                    expected_sha256=expected_sha256,
                )

            with patch.object(
                bank.store,
                "publish_state",
                side_effect=fail_receipt_once,
            ):
                with self.assertRaisesRegex(
                    ControllerCrystalBridgeIntegrityError,
                    "resumes safely",
                ):
                    export_controller_crystals(controller, bank)
            self.assertTrue(failed)
            self.assertEqual(bank.manifest().generation, 2)

            recovered = export_controller_crystals(controller, bank)
            transaction = restore_controller_crystal_transaction(
                bank,
                recovered.receipt.transaction_plan_sha256,
            )
            self.assertEqual(transaction.status, "committed")
            restored = restore_controller_crystal_export(
                bank,
                recovered.receipt.sha256,
            )
            self.assertEqual(restored.receipt, recovered.receipt)


if __name__ == "__main__":
    unittest.main()
