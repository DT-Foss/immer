from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fertig_abstention_audit.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("fertig_abstention_audit", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_script()


class _GoldPoison:
    def __str__(self) -> str:  # pragma: no cover - must never be called
        raise AssertionError("gold answer influenced the audit")

    def __eq__(self, _other: object) -> bool:  # pragma: no cover
        raise AssertionError("gold answer influenced the audit")


def _binding(reason: str, *, target_ok: bool = True):
    return SimpleNamespace(
        ok=False,
        answer=None,
        reason=reason,
        target=SimpleNamespace(ok=target_ok),
    )


def _parse(status: str, reason: str, *, ok: bool = False):
    return SimpleNamespace(ok=ok, status=status, reason=reason)


def _benchmark(*, wrong: bool = False):
    statuses = ["correct", "abstained", "abstained", "abstained", "abstained"]
    if wrong:
        statuses[0] = "incorrect"
    items = [
        {
            "item_id": f"item-{index}",
            "index": index,
            "status": status,
            "question": f"question {index}",
            "gold": _GoldPoison(),
        }
        for index, status in enumerate(statuses)
    ]
    counts = {status: statuses.count(status) for status in audit.ALLOWED_STATUSES}
    wrong_count = counts["incorrect"] + counts["error"]
    return {
        "schema": "immer.benchmark/v1",
        "benchmark": "GSM8K-test",
        "n": len(items),
        "correct": counts["correct"],
        "abstained": counts["abstained"],
        "incorrect": counts["incorrect"],
        "errors": counts["error"],
        "wrong": wrong_count,
        "wrong_must_be_zero": wrong_count == 0,
        "dataset_sha256": "a" * 64,
        "harness_sha256": "b" * 64,
        "items_sha256": "c" * 64,
        "items": items,
    }


class FertigAbstentionAuditTests(unittest.TestCase):
    def test_legacy_categories_are_closed_and_extract_only_contract_fields(
        self,
    ) -> None:
        fixtures = (
            (
                _binding("kein Frageziel", target_ok=False),
                "target_parse_failed",
            ),
            (
                _binding("Bindung unvollständig: 0 qty, 2 ratio, op=sum"),
                "target_evidence_missing",
            ),
            (
                _binding("Bindung unvollständig: 3 qty, 0 ratio, op=left"),
                "relation_incomplete",
            ),
            (
                _binding("Zeit-/Geldbilanz nicht strukturell bewiesen"),
                "equation_guard",
            ),
        )
        for result, expected in fixtures:
            with self.subTest(expected=expected):
                classified = audit.classify_legacy_binding(result)
                self.assertEqual(classified["category"], expected)

        relation = audit.classify_legacy_binding(fixtures[2][0])
        self.assertEqual(
            relation["details"],
            {"qty_for_target": 3, "ratios_for_target": 0, "operation": "left"},
        )
        with self.assertRaisesRegex(audit.AuditError, "unrecognized"):
            audit.classify_legacy_binding(_binding("novel reason"))
        with self.assertRaisesRegex(audit.AuditError, "now succeeds"):
            audit.classify_legacy_binding(
                SimpleNamespace(
                    ok=True,
                    answer="1",
                    reason="sum",
                    target=SimpleNamespace(ok=True),
                )
            )

    def test_structural_categories_and_scopes_are_closed(self) -> None:
        fixtures = (
            (
                _parse("parsed", "evidence-closed signed event grammar", ok=True),
                "exact_recovery",
                "exact_recovery_scope",
            ),
            (
                _parse("ambiguous", "numeric pronoun binding is not proven"),
                "numeric_pronoun_ambiguous",
                "potential_coreference_scope",
            ),
            (
                _parse("ambiguous", "question pronoun binding is not proven"),
                "question_pronoun_ambiguous",
                "potential_coreference_scope",
            ),
            (
                _parse("unsupported", "unparsed numeric clause at 2:8"),
                "numeric_clause_unsupported",
                "grammar_scope",
            ),
            (
                _parse("unsupported", "unsupported target at 9:15"),
                "target_unsupported",
                "grammar_scope",
            ),
            (
                _parse("ambiguous", "new typed ambiguity"),
                "relation_ambiguous",
                "grammar_scope",
            ),
            (
                _parse("unsupported", "new typed unsupported relation"),
                "relation_unsupported",
                "grammar_scope",
            ),
            (
                _parse("invalid", "new typed invalid relation"),
                "relation_invalid",
                "grammar_scope",
            ),
        )
        for result, expected, scope in fixtures:
            with self.subTest(expected=expected):
                classified = audit.classify_structural_parse(result)
                self.assertEqual(classified["category"], expected)
                self.assertEqual(audit._scope_for_structural(expected), scope)

        with self.assertRaisesRegex(audit.AuditError, "expected 'ambiguous'"):
            audit.classify_structural_parse(
                _parse("unsupported", "numeric pronoun binding is not proven")
            )
        with self.assertRaisesRegex(audit.AuditError, "unrecognized structural status"):
            audit.classify_structural_parse(_parse("novel", "unknown"))

    def test_report_is_deterministic_complete_gold_free_and_zero_hungarian(
        self,
    ) -> None:
        benchmark = _benchmark()
        legacy_results = {
            "question 1": _binding("kein Frageziel", target_ok=False),
            "question 2": _binding("Bindung unvollständig: 0 qty, 0 ratio, op=sum"),
            "question 3": _binding("Bindung unvollständig: 2 qty, 0 ratio, op=diff"),
            "question 4": _binding("Zeit-/Geldbilanz nicht strukturell bewiesen"),
        }
        structural_results = {
            "question 1": _parse("ambiguous", "numeric pronoun binding is not proven"),
            "question 2": _parse("ambiguous", "question pronoun binding is not proven"),
            "question 3": _parse("unsupported", "unparsed numeric clause at 0:10"),
            "question 4": _parse(
                "parsed", "evidence-closed signed event grammar", ok=True
            ),
        }
        legacy_calls: list[str] = []
        structural_calls: list[str] = []

        def legacy_bind(question: str):
            legacy_calls.append(question)
            return legacy_results[question]

        def structural_parse(question: str):
            structural_calls.append(question)
            return structural_results[question]

        kwargs = {
            "legacy_bind": legacy_bind,
            "structural_parse": structural_parse,
            "provenance": {"fixture": "sealed"},
            "require_current_evidence": False,
        }
        first = audit.build_audit_report(benchmark, **kwargs)
        second = audit.build_audit_report(benchmark, **kwargs)

        self.assertEqual(first, second)
        self.assertEqual(first["audited_abstentions"], 4)
        self.assertEqual(first["baseline_partition"]["n"], 5)
        self.assertEqual(
            first["classification_counts"]["scope"],
            {
                "potential_coreference_scope": 2,
                "grammar_scope": 1,
                "exact_recovery_scope": 1,
            },
        )
        self.assertEqual(first["current_exact_recoveries"], 1)
        self.assertEqual(first["current_remaining_abstentions"], 3)
        self.assertEqual(
            first["exact_recovery_attribution"]["identity"]["indices"], [4]
        )
        self.assertEqual(
            first["exact_recovery_attribution"]["mechanisms"],
            {
                "preexisting_generic_signed_event_exact": {
                    "count": 1,
                    "indices": [4],
                    "indices_sha256": audit._sha256_bytes(
                        audit._canonical_json_bytes([4])
                    ),
                }
            },
        )
        self.assertEqual(first["recovery_wave_delta"]["added_exact_recoveries"], 0)
        self.assertEqual(
            first["recovery_wave_delta"]["previously_sealed_exact_recoveries"], 1
        )
        self.assertEqual(first["recovery_wave_history"]["baseline_exact_recoveries"], 1)
        self.assertEqual(
            [
                wave["added_exact_recoveries"]
                for wave in first["recovery_wave_history"]["waves"]
            ],
            [0, 0, 0, 0],
        )
        self.assertEqual(
            first["hungarian_verdict"]["directly_eligible_exclusive_instances"],
            0,
        )
        self.assertTrue(first["gold_label_free"])
        self.assertEqual(len(first["items"]), 4)
        self.assertTrue(all("gold" not in item for item in first["items"]))
        self.assertEqual(legacy_calls, [f"question {i}" for i in range(1, 5)] * 2)
        self.assertEqual(structural_calls, [f"question {i}" for i in range(1, 5)] * 2)
        unsealed = dict(first)
        observed_digest = unsealed.pop("report_sha256")
        self.assertEqual(
            observed_digest,
            audit._sha256_bytes(audit._canonical_json_bytes(unsealed)),
        )

    def test_benchmark_validation_rejects_partition_and_wrong_gate_defects(
        self,
    ) -> None:
        items, counts = audit.validate_benchmark_report(_benchmark())
        self.assertEqual(len(items), 5)
        self.assertEqual(counts["abstained"], 4)

        with self.assertRaisesRegex(audit.AuditError, "expected zero"):
            audit.validate_benchmark_report(_benchmark(wrong=True))
        _, wrong_counts = audit.validate_benchmark_report(
            _benchmark(wrong=True), require_zero_wrong=False
        )
        self.assertEqual(wrong_counts["incorrect"], 1)

        malformed = _benchmark()
        malformed["abstained"] = 3
        with self.assertRaisesRegex(audit.AuditError, "claims 3"):
            audit.validate_benchmark_report(malformed)

        duplicate = _benchmark()
        duplicate["items"][1]["item_id"] = duplicate["items"][0]["item_id"]
        with self.assertRaisesRegex(audit.AuditError, "duplicate.*item_id"):
            audit.validate_benchmark_report(duplicate)

        missing_index = _benchmark()
        missing_index["items"][-1]["index"] = 99
        with self.assertRaisesRegex(audit.AuditError, "complete range"):
            audit.validate_benchmark_report(missing_index)

    def test_current_evidence_contract_has_exact_measured_partitions(self) -> None:
        self.assertEqual(
            audit.CURRENT_LEGACY_COUNTS,
            {
                "target_parse_failed": 57,
                "target_evidence_missing": 154,
                "relation_incomplete": 18,
                "equation_guard": 1,
            },
        )
        self.assertEqual(
            audit.CURRENT_STRUCTURAL_COUNTS,
            {
                "exact_recovery": 70,
                "numeric_pronoun_ambiguous": 71,
                "question_pronoun_ambiguous": 3,
                "numeric_clause_unsupported": 80,
                "target_unsupported": 5,
                "relation_ambiguous": 1,
                "relation_unsupported": 0,
                "relation_invalid": 0,
            },
        )
        self.assertEqual(
            audit.CURRENT_SCOPE_COUNTS,
            {
                "exact_recovery_scope": 70,
                "potential_coreference_scope": 74,
                "grammar_scope": 86,
            },
        )
        self.assertEqual(
            audit.CURRENT_EXACT_RECOVERY_INDICES,
            (
                547,
                550,
                574,
                578,
                587,
                603,
                610,
                613,
                619,
                627,
                631,
                643,
                651,
                672,
                682,
                692,
                694,
                701,
                722,
                724,
                731,
                745,
                746,
                747,
                754,
                763,
                770,
                778,
                780,
                782,
                797,
                802,
                810,
                819,
                823,
                825,
                833,
                836,
                837,
                840,
                844,
                861,
                865,
                868,
                883,
                892,
                900,
                901,
                913,
                916,
                924,
                930,
                934,
                944,
                959,
                992,
                1016,
                1064,
                1172,
                1192,
                1194,
                1217,
                1219,
                1246,
                1252,
                1253,
                1261,
                1293,
                1300,
                1304,
            ),
        )
        self.assertEqual(
            audit.GROUND_WAVE_EXACT_RECOVERY_INDICES,
            (672, 780, 944, 1219, 1261),
        )
        self.assertEqual(audit.BASELINE_EXACT_RECOVERIES, 42)
        self.assertEqual(
            audit.GROUND_WAVE_EXACT_MECHANISMS,
            {
                672: "signed_event:temporal_categorical_block_remainder",
                780: "signed_event:absolute_weighted_score_difference",
                944: "signed_event:exhaustive_unit_rate_ledger",
                1219: "signed_event:exhaustive_unit_rate_ledger",
                1261: "signed_event:recurring_pronoun_rate_ledger",
            },
        )
        self.assertEqual(
            audit.DISCOURSE_SSA_WAVE_EXACT_RECOVERY_INDICES,
            (578, 603, 802, 833, 900, 959, 1064, 1194, 1252, 1304),
        )
        self.assertEqual(
            audit.DISCOURSE_SSA_WAVE_EXACT_MECHANISMS,
            {
                578: "signed_event:grounded_value_pipeline",
                603: "signed_event:shared_duration_affine_rates",
                802: "signed_event:closed_collection_share_completion",
                833: "signed_event:typed_scale_chain_conversion",
                900: "signed_event:temporal_reader_affine_difference",
                959: "signed_event:closed_named_scale_group_total",
                1064: "signed_event:ordered_affine_category_ledger",
                1194: "signed_event:typed_species_scale_total",
                1252: "signed_event:typed_ratio_property_chain_total",
                1304: "signed_event:typed_scaled_measure_difference",
            },
        )
        self.assertEqual(
            audit.CLOSED_SCHEDULE_WAVE_EXACT_RECOVERY_INDICES,
            (627, 782, 992, 1192, 1217, 1246, 1253, 1293, 1300),
        )
        self.assertEqual(
            audit.CLOSED_SCHEDULE_WAVE_EXACT_MECHANISMS,
            {
                627: "signed_event:closed_week_complement_schedule",
                782: "signed_event:closed_disjoint_week_schedule",
                992: "signed_event:calendar_frequency_ledger",
                1192: "signed_event:explicit_weekly_pay_schedule",
                1217: "signed_event:closed_piecewise_period_cost",
                1246: "signed_event:explicit_period_score_total",
                1253: "signed_event:canonical_duration_rate_conversion",
                1293: "signed_event:canonical_weekly_sales_total",
                1300: "signed_event:explicit_weekday_exception_schedule",
            },
        )
        self.assertEqual(
            audit.RECURRENCE_WAVE_EXACT_RECOVERY_INDICES,
            (547, 810, 1016, 1172),
        )
        self.assertEqual(
            audit.RECURRENCE_WAVE_EXACT_MECHANISMS,
            {
                547: "signed_event:closed_phone_tree_recurrence",
                810: "signed_event:closed_monthly_state_recurrence",
                1016: "signed_event:fixed_base_percentage_recurrence",
                1172: "signed_event:closed_daily_geometric_total",
            },
        )
        self.assertEqual(
            set(audit.CURRENT_EXACT_RECOVERY_MECHANISMS),
            set(audit.CURRENT_EXACT_RECOVERY_INDICES),
        )
        with self.assertRaisesRegex(audit.AuditError, "evidence drifted"):
            audit._validate_current_evidence(
                {
                    "schema": audit.SCHEMA,
                    "report_revision": audit.REPORT_REVISION,
                    "classification_counts": {
                        "legacy": audit.CURRENT_LEGACY_COUNTS,
                        "structural": audit.CURRENT_STRUCTURAL_COUNTS,
                        "scope": {
                            "exact_recovery_scope": 70,
                            "potential_coreference_scope": 73,
                            "grammar_scope": 87,
                        },
                    },
                }
            )

    def test_production_report_pins_recovery_identity_and_wave_delta(self) -> None:
        report = json.loads(audit.DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8"))
        audit._validate_current_evidence(report)
        identity = report["exact_recovery_attribution"]["identity"]
        self.assertEqual(identity["count"], 70)
        self.assertEqual(
            identity["indices"], list(audit.CURRENT_EXACT_RECOVERY_INDICES)
        )
        self.assertEqual(
            identity["indices_sha256"],
            audit._sha256_bytes(
                audit._canonical_json_bytes(list(audit.CURRENT_EXACT_RECOVERY_INDICES))
            ),
        )
        mechanism_identity = report["exact_recovery_attribution"]["mechanism_identity"]
        self.assertEqual(mechanism_identity["count"], 70)
        self.assertEqual(
            {row["index"]: row["mechanism"] for row in mechanism_identity["rows"]},
            audit.CURRENT_EXACT_RECOVERY_MECHANISMS,
        )
        delta = report["recovery_wave_delta"]
        self.assertEqual(delta["previously_sealed_exact_recoveries"], 66)
        self.assertEqual(delta["added_exact_recoveries"], 4)
        self.assertEqual(delta["current_exact_recoveries"], 70)
        self.assertEqual(
            delta["added_identity"]["indices"],
            list(audit.RECURRENCE_WAVE_EXACT_RECOVERY_INDICES),
        )
        self.assertIn("this wave claims 4, not 70", delta["attribution"])

        history = report["recovery_wave_history"]
        self.assertEqual(history["baseline_exact_recoveries"], 42)
        self.assertEqual(
            [wave["added_exact_recoveries"] for wave in history["waves"]],
            [5, 10, 9, 4],
        )
        self.assertEqual(
            [wave["current_exact_recoveries"] for wave in history["waves"]],
            [47, 57, 66, 70],
        )
        self.assertIn(
            "70 current exact recoveries = 42 sealed baseline",
            history["attribution"],
        )
        self.assertIn("+ 5 by", history["attribution"])
        self.assertIn("+ 10 by", history["attribution"])
        self.assertIn("+ 9 by", history["attribution"])
        self.assertIn("+ 4 by", history["attribution"])

        swapped = json.loads(json.dumps(report))
        next(item for item in swapped["items"] if item["index"] == 550)["index"] = 549
        with self.assertRaisesRegex(audit.AuditError, "identity drifted"):
            audit._validate_current_evidence(swapped)

        misattributed = json.loads(json.dumps(report))
        next(item for item in misattributed["items"] if item["index"] == 627)[
            "exact_recovery_mechanism"
        ] = "signed_event:unrelated"
        with self.assertRaisesRegex(audit.AuditError, "inconsistent mechanism"):
            audit._validate_current_evidence(misattributed)

        count_preserving_delta_swap = json.loads(json.dumps(report))
        count_preserving_delta_swap["recovery_wave_delta"]["added_identity"]["indices"][
            0
        ] = audit.GROUND_WAVE_EXACT_RECOVERY_INDICES[0]
        with self.assertRaisesRegex(audit.AuditError, "delta drifted"):
            audit._validate_current_evidence(count_preserving_delta_swap)

        count_preserving_history_swap = json.loads(json.dumps(report))
        first_wave, *_, final_wave = count_preserving_history_swap[
            "recovery_wave_history"
        ]["waves"]
        (
            first_wave["added_identity"]["indices"][0],
            final_wave["added_identity"]["indices"][0],
        ) = (
            final_wave["added_identity"]["indices"][0],
            first_wave["added_identity"]["indices"][0],
        )
        with self.assertRaisesRegex(audit.AuditError, "history drifted"):
            audit._validate_current_evidence(count_preserving_history_swap)

        full_mechanism_misattribution = json.loads(json.dumps(report))
        full_mechanism_misattribution["exact_recovery_attribution"][
            "mechanism_identity"
        ]["rows"][0]["mechanism"] = "signed_event:unrelated"
        with self.assertRaisesRegex(audit.AuditError, "attribution drifted"):
            audit._validate_current_evidence(full_mechanism_misattribution)

        unsealed = dict(report)
        observed_report_sha256 = unsealed.pop("report_sha256")
        self.assertEqual(
            observed_report_sha256,
            audit._sha256_bytes(audit._canonical_json_bytes(unsealed)),
        )
        for evidence in report["provenance"]["files"].values():
            path = Path(evidence["path"])
            resolved = path if path.is_absolute() else audit.ROOT / path
            self.assertEqual(evidence["sha256"], audit._sha256_file(resolved))
        self.assertIn("typed_discourse_ssa", report["provenance"]["files"])

    def test_atomic_output_and_provenance_hashes_are_sealed(self) -> None:
        provenance = audit.current_provenance(audit.DEFAULT_BENCHMARK_PATH)
        for evidence in provenance["files"].values():
            self.assertRegex(evidence["sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(
            provenance["classification_contract_sha256"], r"^[0-9a-f]{64}$"
        )

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "nested" / "audit.json"
            document = {"schema": audit.SCHEMA, "value": 1}
            audit.write_report(output, document)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), document)
            self.assertTrue(output.read_bytes().endswith(b"\n"))
            self.assertEqual(list(output.parent.glob("*.tmp")), [])

            foreign = root / "foreign.json"
            foreign.write_text("sentinel", encoding="utf-8")
            link = root / "link.json"
            link.symlink_to(foreign)
            with self.assertRaisesRegex(audit.AuditError, "symlink"):
                audit.write_report(link, document)
            self.assertEqual(foreign.read_text(encoding="utf-8"), "sentinel")

    def test_cli_defaults_to_committed_input_and_versioned_output(self) -> None:
        args = audit._parser().parse_args([])
        self.assertEqual(args.benchmark_json, audit.DEFAULT_BENCHMARK_PATH)
        self.assertEqual(args.output_json, audit.DEFAULT_OUTPUT_PATH)
        self.assertFalse(args.allow_wrong)
        self.assertFalse(args.allow_evidence_drift)


if __name__ == "__main__":
    unittest.main()
