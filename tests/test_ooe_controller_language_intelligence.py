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
    ControllerCrystalExport,
    export_controller_crystals,
)
from immer.runtimes.ooe.controller_language_intelligence import (
    ControllerLanguageBootstrapConfig,
    ControllerLanguageBootstrapIntegrityError,
    ControllerLanguageBootstrapReport,
    ControllerLanguageConvergenceError,
    ControllerPortableMacroBridgeReceipt,
    ControllerProgramSupportReceipt,
    bootstrap_controller_crystal_language,
    controller_support_to_portable_program,
    localize_controller_supported_program,
)
from immer.runtimes.ooe.crystal import CrystalStore
from immer.runtimes.ooe.dialect_mesh import (
    DialectMeshIntegrityError,
    localize_portable_program,
)
from immer.runtimes.ooe.executable_lexicon import (
    ExecutableLexiconState,
    ExecutableWordCompiler,
)
from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.language_bridge import resolve_snapshot_compute_bindings
from immer.runtimes.ooe.qwen_bridge import (
    ACTION_SCHEMA_SHA256,
    OOE_ACTIONS,
    QwenOoeFeatureReceipt,
    feature_schema_sha256,
)
from immer.runtimes.qwen3_8.semantic_atlas import GraphRevision, ProbeIdentity


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_MODEL_PIN = _sha("controller-language-model-pin")
_WEIGHT_GRAPH = GraphRevision(7, _sha("controller-language-weight-graph"))
_ATLAS_GRAPH = GraphRevision(11, _sha("controller-language-atlas-graph"))
_VERIFIER = _sha("controller-language-verifier")


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


def _fixture(root: Path) -> tuple[ComputeCrystalBank, ControllerCrystalExport]:
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
            quantization_levels=65_535,
            reservoir_size=8,
        ),
    )
    temporal = 0
    sites: list[str] = []
    for site in range(2):
        for source_index in range(len(OOE_ACTIONS)):
            receipt = _feature(temporal, site=site, source=source_index)
            controller.ingest_teacher(
                receipt,
                VerifiedTeacherTransition(
                    feature_receipt_sha256=receipt.sha256,
                    site_identity_sha256=receipt.site_identity.sha256,
                    source_action=OOE_ACTIONS[source_index],
                    target_action=OOE_ACTIONS[
                        (source_index + site + 1) % len(OOE_ACTIONS)
                    ],
                    verifier_sha256=_VERIFIER,
                    evidence_sha256=receipt.evidence_sha256s[0],
                    quality_sha256=_sha(f"quality-{receipt.sha256}"),
                ),
            )
            temporal += 1
        sites.append(receipt.site_identity.sha256)
    for site_sha256 in sites:
        coverage = controller.coverage_receipt(site_sha256)
        controller.promote(
            site_sha256,
            coverage_sha256=coverage.sha256,
            verifier_sha256s=coverage.verifier_sha256s,
        )
    controller.save_snapshot()
    restored = OoeController.restore(crystal_store=source)
    bank = ComputeCrystalBank(root / "compute")
    return bank, export_controller_crystals(restored, bank)


class ControllerLanguageIntelligenceTests(unittest.TestCase):
    def test_live_training_support_compilation_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bank, export = _fixture(root)
            result = bootstrap_controller_crystal_language(
                export,
                bank,
                state_store=root / "language",
                lexicon_store=root / "lexicon",
            )

            self.assertEqual(result.compiled.artifact_kind, "crystal")
            self.assertTrue(result.compiled.constant_discharge)
            self.assertEqual(result.compiled.expanded_primitive_actions, 4)
            self.assertEqual(len(result.support.occurrences), 3)
            self.assertEqual(len(result.support.action_ids), 4)
            self.assertEqual(result.report.heldout_accepted, 12)
            self.assertEqual(result.report.compiled_operator_count, 1)
            self.assertEqual(result.report.flat_operator_count, 4)
            self.assertGreater(result.report.historical_work_released, 0)
            self.assertEqual(
                result.snapshot.learner_state_sha256,
                result.report.language_state_sha256,
            )
            for occurrence in result.support.occurrences:
                self.assertEqual(
                    occurrence.commits[0].language_state_before_sha256,
                    result.report.language_state_sha256,
                )
                self.assertNotEqual(
                    occurrence.commits[-1].language_state_after_sha256,
                    result.report.language_state_sha256,
                )
            self.assertLessEqual(
                float.fromhex(result.report.max_abs_error_hex),
                float.fromhex(result.report.parity_tolerance_hex),
            )
            self.assertEqual(
                ControllerProgramSupportReceipt.from_bytes(result.support.to_bytes()),
                result.support,
            )
            self.assertEqual(
                ControllerLanguageBootstrapReport.from_bytes(result.report.to_bytes()),
                result.report,
            )
            result.support.verify(export.receipt, bank)

            resumed = bootstrap_controller_crystal_language(
                export.receipt,
                bank,
                state_store=root / "language",
                lexicon_store=root / "lexicon",
            )
            self.assertEqual(resumed.report.to_bytes(), result.report.to_bytes())
            self.assertEqual(resumed.support, result.support)

    def test_support_tamper_and_insufficient_support_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bank, export = _fixture(root)
            result = bootstrap_controller_crystal_language(
                export,
                bank,
                state_store=root / "language",
                lexicon_store=root / "lexicon",
            )
            document = json.loads(result.support.to_bytes())
            nested = document["body"]["occurrences"][0]["outcomes"][0]
            nested["sha256"] = _sha("forged-outcome")
            document["body_sha256"] = hashlib.sha256(
                canonical_json_bytes(document["body"])
            ).hexdigest()
            with self.assertRaises(ControllerLanguageBootstrapIntegrityError):
                ControllerProgramSupportReceipt.from_bytes(
                    canonical_json_bytes(document)
                )
            with self.assertRaisesRegex(ValueError, "three ordered distinct"):
                replace(
                    result.support,
                    occurrences=result.support.occurrences[:2],
                )

    def test_no_convergence_and_deterministic_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bank, export = _fixture(root)
            with self.assertRaises(ControllerLanguageConvergenceError):
                bootstrap_controller_crystal_language(
                    export,
                    bank,
                    state_store=root / "failed-language",
                    lexicon_store=root / "failed-lexicon",
                    config=ControllerLanguageBootstrapConfig(max_training_episodes=1),
                )

        reports: list[bytes] = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                bank, export = _fixture(root)
                result = bootstrap_controller_crystal_language(
                    export,
                    bank,
                    state_store=root / "language",
                    lexicon_store=root / "lexicon",
                )
                reports.append(result.report.to_bytes())
        self.assertEqual(reports[0], reports[1])

    def test_supported_word_portably_localizes_into_independent_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bank, export = _fixture(root)
            source = bootstrap_controller_crystal_language(
                export,
                bank,
                state_store=root / "source-language",
                lexicon_store=root / "source-lexicon",
                config=ControllerLanguageBootstrapConfig(seed=17),
            )
            sibling = bootstrap_controller_crystal_language(
                export,
                bank,
                state_store=root / "sibling-language",
                lexicon_store=root / "sibling-lexicon",
                config=ControllerLanguageBootstrapConfig(seed=18),
            )
            source_map = dict(source.snapshot.sender_action_words)
            sibling_map = dict(sibling.snapshot.sender_action_words)
            self.assertNotEqual(source_map, sibling_map)

            portable, discovery, bridge_receipt = (
                controller_support_to_portable_program(
                    source.support,
                    source.definition,
                    export.receipt,
                    bank,
                )
            )
            self.assertEqual(
                ControllerPortableMacroBridgeReceipt.from_bytes(
                    bridge_receipt.to_bytes()
                ),
                bridge_receipt,
            )
            self.assertEqual(bridge_receipt.discovery, discovery)
            bridge_receipt.verify(
                source.support,
                source.definition,
                export.receipt,
                bank,
            )
            with self.assertRaises(ControllerLanguageBootstrapIntegrityError):
                replace(
                    bridge_receipt,
                    definition_sha256=_sha("another-definition"),
                ).verify(
                    source.support,
                    source.definition,
                    export.receipt,
                    bank,
                )

            context = "controller-language-train"
            with self.assertRaisesRegex(
                DialectMeshIntegrityError, "authorization is missing"
            ):
                localize_portable_program(
                    portable,
                    sibling.snapshot,
                    export.frontier,
                    source_discovery=discovery,
                    source_definition=source.definition,
                    source_snapshot=source.snapshot,
                    source_frontier=export.frontier,
                    target_context_id=context,
                )
            localized, _localization = localize_controller_supported_program(
                portable,
                bridge_receipt,
                source.support,
                source.definition,
                export.receipt,
                bank,
                sibling.snapshot,
                export.frontier,
                target_context_id=context,
            )
            forged_portable = replace(
                portable,
                action_sequence=tuple(reversed(portable.action_sequence)),
                source_word_sequence=tuple(reversed(portable.source_word_sequence)),
            )
            with self.assertRaises(ControllerLanguageBootstrapIntegrityError):
                localize_controller_supported_program(
                    forged_portable,
                    bridge_receipt,
                    source.support,
                    source.definition,
                    export.receipt,
                    bank,
                    sibling.snapshot,
                    export.frontier,
                    target_context_id=context,
                )
            self.assertEqual(
                localized.child_word_ids,
                tuple(
                    sibling.snapshot.encode_action(action, context)
                    for action in portable.action_sequence
                ),
            )
            primitives, _resolution = resolve_snapshot_compute_bindings(
                sibling.snapshot,
                export.frontier,
                context_id=context,
            )
            state = ExecutableLexiconState.initial(
                primitives,
                language_snapshot_sha256=sibling.snapshot.sha256,
                frontier_sha256=export.frontier.sha256,
                authority_hashes=export.frontier.authority_hashes,
                context_id=context,
            ).with_definition(localized)
            compiler = ExecutableWordCompiler(state, bank)
            compiled = compiler.compile(localized.new_word_id)
            value = np.asarray([0.05, 0.15, 0.25, 0.20, 0.35], dtype=np.float64)
            expected = value
            for action in portable.action_sequence:
                expected = bank.restore_crystal(
                    export.frontier.binding(action).artifact_sha256
                ).apply(expected)
            execution = compiler.execute(compiled, value)
            np.testing.assert_allclose(execution.output, expected, rtol=0.0, atol=1e-12)
            self.assertTrue(compiled.constant_discharge)
            self.assertGreater(execution.receipt.historical_work_released, 0)

    def test_equivalent_bank_rebuild_and_atomic_publication_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bank_a, export_a = _fixture(root / "a")
            state = CrystalStore(root / "shared-language")
            lexicon = CrystalStore(root / "shared-lexicon")
            first = bootstrap_controller_crystal_language(
                export_a,
                bank_a,
                state_store=state,
                lexicon_store=lexicon,
            )
            bank_b, export_b = _fixture(root / "b")
            self.assertEqual(export_b.receipt.sha256, export_a.receipt.sha256)
            rebuilt = bootstrap_controller_crystal_language(
                export_b,
                bank_b,
                state_store=state,
                lexicon_store=lexicon,
            )
            self.assertEqual(rebuilt.report.to_bytes(), first.report.to_bytes())
            self.assertEqual(
                bank_b.current_anchor_sha256(),
                first.report.final_compute_bank_anchor_sha256,
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bank, export = _fixture(root)
            state = CrystalStore(root / "language")
            lexicon = CrystalStore(root / "lexicon")
            original_publish = state.publish_state
            failed = False

            def fail_snapshot_once(name, data, *, expected_sha256=None):
                nonlocal failed
                if not failed and ":snapshot:" in name:
                    failed = True
                    raise OSError("injected atomic publication fault")
                return original_publish(name, data, expected_sha256=expected_sha256)

            with patch.object(state, "publish_state", side_effect=fail_snapshot_once):
                with self.assertRaises(ControllerLanguageBootstrapIntegrityError):
                    bootstrap_controller_crystal_language(
                        export,
                        bank,
                        state_store=state,
                        lexicon_store=lexicon,
                    )
            retried = bootstrap_controller_crystal_language(
                export,
                bank,
                state_store=state,
                lexicon_store=lexicon,
            )
            self.assertTrue(failed)
            self.assertEqual(
                ControllerLanguageBootstrapReport.from_bytes(retried.report.to_bytes()),
                retried.report,
            )
            with patch.object(
                lexicon,
                "restore_state",
                side_effect=OSError("injected lexicon restore fault"),
            ):
                with self.assertRaises(ControllerLanguageBootstrapIntegrityError):
                    bootstrap_controller_crystal_language(
                        export,
                        bank,
                        state_store=state,
                        lexicon_store=lexicon,
                    )


if __name__ == "__main__":
    unittest.main()
