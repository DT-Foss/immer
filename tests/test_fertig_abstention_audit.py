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


def _parse(status: str, reason: str):
    return SimpleNamespace(ok=False, status=status, reason=reason)


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
        with self.assertRaisesRegex(audit.AuditError, "unrecognized"):
            audit.classify_structural_parse(_parse("unsupported", "unknown"))

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
            "question 4": _parse("unsupported", "unsupported target at 4:14"),
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
                "grammar_scope": 2,
            },
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
                "numeric_pronoun_ambiguous": 127,
                "question_pronoun_ambiguous": 4,
                "numeric_clause_unsupported": 95,
                "target_unsupported": 4,
            },
        )
        self.assertEqual(
            audit.CURRENT_SCOPE_COUNTS,
            {"potential_coreference_scope": 131, "grammar_scope": 99},
        )
        with self.assertRaisesRegex(audit.AuditError, "evidence drifted"):
            audit._validate_current_evidence(
                {
                    "classification_counts": {
                        "legacy": audit.CURRENT_LEGACY_COUNTS,
                        "structural": audit.CURRENT_STRUCTURAL_COUNTS,
                        "scope": {
                            "potential_coreference_scope": 130,
                            "grammar_scope": 100,
                        },
                    }
                }
            )

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
