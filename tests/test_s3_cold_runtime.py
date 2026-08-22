from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from immer.capabilities.organbank import OrganBank
from immer.capabilities.s3_runtime import S3Arithmetic, canonical_cases
from immer.contracts import ExecutionStatus, Request


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "s3_ship_v6.json"
ARTIFACTS = (
    ROOT / "vendor" / "o1state" / "results" / "pos_ckpt.pt",
    ROOT / "vendor" / "o1state" / "s3_ship" / "organ_dual_donor.pt",
    ROOT / "vendor" / "o1state" / "s3_ship" / "organ_mul_donor.pt",
    ROOT / "vendor" / "o1state" / "s3_ship" / "organ_mod_kreis.pt",
    ROOT / "vendor" / "o1state" / "s3_ship" / "organ_dezimal.pt",
)
HAS_COLD_BANK = importlib.util.find_spec("torch") is not None and all(path.is_file() for path in ARTIFACTS)


class S3ManifestTests(unittest.TestCase):
    def test_manifest_is_organbank_compatible(self) -> None:
        bank = OrganBank.from_manifest(MANIFEST)
        self.assertEqual(
            bank.names(),
            ("arith-dual", "decimal-crystal", "mul-log", "z3-circle"),
        )
        if HAS_COLD_BANK:
            for name in bank.names():
                self.assertTrue(bank.verify(name).is_file())

    def test_constructor_is_cold_and_cases_are_stable(self) -> None:
        runtime = S3Arithmetic(MANIFEST)
        self.assertIsNone(runtime._host)
        self.assertEqual(len(canonical_cases()), 152)
        self.assertIsNone(runtime._host)


@unittest.skipUnless(HAS_COLD_BANK, "local ignored SHIP-v6 artifacts or torch are unavailable")
class S3ColdRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = S3Arithmetic(MANIFEST)

    def test_canonical_152_case_suite_is_cold_and_exact(self) -> None:
        result = self.runtime.benchmark()
        self.assertEqual(result["cases"], 152)
        self.assertEqual(result["ok"], 152)
        self.assertEqual(result["correct"], 152)
        self.assertEqual(result["route_correct"], 152)
        self.assertTrue(result["no_training"])
        self.assertEqual(
            result["host_sha256"],
            "84f7ac90375668067f348421c581ca60cf3f27dbdb447ae13248dbcc85ea251c",
        )
        self.assertEqual(set(self.runtime._organs), {"arith-dual", "mul-log", "z3-circle", "decimal-crystal"})

    def test_evidence_carries_the_deployment_proof(self) -> None:
        result = self.runtime.handle(Request("exact_math", "three plus five is"))
        self.assertEqual(result.status, ExecutionStatus.OK)
        self.assertEqual(result.output, "eight")
        self.assertEqual(result.evidence["route"], "ARITH")
        self.assertEqual(result.evidence["organ"], "arith-dual")
        self.assertTrue(result.evidence["crystal_verified"])
        self.assertTrue(result.evidence["no_training"])
        self.assertTrue(result.evidence["exact_readout_agreement"])
        self.assertEqual(len(result.evidence["artifact_sha256"]), 64)
        self.assertEqual(len(result.evidence["internal_digest"]), 16)
        self.assertTrue(result.evidence["host"]["frozen"])
        attention = result.evidence["attention"]
        self.assertEqual(
            attention["program"],
            "2 Local + 1 Balanced + 1 bit-exact Free",
        )
        self.assertTrue(attention["is_arithmetic"])
        self.assertEqual(
            attention["crsa_specific_advantage_over_softmax"],
            "NOT_SHOWN",
        )

    def test_crsa_text_decision_is_a_real_abstention_gate(self) -> None:
        class TextDecision:
            label = "text"
            score = -1.0
            margin = 1.0
            is_arithmetic = False

        class Head:
            feature_schema = "test"
            metadata = {"measurement": {"verdict": "fixture"}}

            @staticmethod
            def digest() -> str:
                return "0" * 64

        class TextRouter:
            head = Head()

            @staticmethod
            def decide_one(host, ids):
                return TextDecision()

        runtime = S3Arithmetic(MANIFEST)
        runtime._router = TextRouter()
        result = runtime.handle(Request("exact_math", "three plus five is"))
        self.assertIs(result.status, ExecutionStatus.ABSTAINED)
        self.assertEqual(result.evidence["route"], "TEXT")
        self.assertFalse(result.evidence["attention"]["is_arithmetic"])

    def test_decimal_crystal_accepts_digit_and_digit_word_forms(self) -> None:
        digits = self.runtime.handle(Request("exact_math", "47 * 6"))
        words = self.runtime.handle(Request("exact_math", "four seven times six is"))
        self.assertEqual(digits.output, "two eight two")
        self.assertEqual(words.output, "two eight two")
        self.assertEqual(digits.evidence["route"], "DECIMAL")
        self.assertFalse(digits.evidence["generic_forward_used"])
        self.assertEqual(digits.evidence["parser_recurrence"], "h <- 10h + v")

    def test_question_wrappers_and_german_carrier_normalise_to_the_cold_path(self) -> None:
        cases = {
            "What is 3 plus 5?": "eight",
            "calculate: three plus five": "eight",
            "Was ist drei plus fünf?": "eight",
            "Berechne sieben mal zwei": "fourteen",
            "zwölf plus drei ist": "one five",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                result = self.runtime.handle(Request("exact_math", text))
                self.assertEqual(result.status, ExecutionStatus.OK)
                self.assertEqual(result.output, expected)

    def test_z3sum_is_not_misrepresented_as_modulo(self) -> None:
        z3 = self.runtime.handle(Request("exact_math", "seven z3sum four is"))
        self.assertEqual(z3.status, ExecutionStatus.OK)
        self.assertEqual(z3.output, "two")
        self.assertEqual(z3.evidence["semantics"], "(a + b) mod 3")
        self.assertFalse(z3.evidence["ordinary_modulo"])
        for text in ("seven remainder four is", "seven modulo four", "7 % 4"):
            result = self.runtime.handle(Request("exact_math", text))
            self.assertEqual(result.status, ExecutionStatus.ABSTAINED)
            self.assertIn("z3sum", result.reason)

    def test_text_and_unsupported_math_abstain_without_guessing(self) -> None:
        for text in ("the king of france", "who is the king", "8 divided by 2"):
            result = self.runtime.handle(Request("exact_math", text))
            self.assertEqual(result.status, ExecutionStatus.ABSTAINED)

    def test_host_and_organs_are_frozen(self) -> None:
        self.assertFalse(any(parameter.requires_grad for parameter in self.runtime.frozen_host.parameters()))
        for loaded in self.runtime._organs.values():
            self.assertFalse(any(parameter.requires_grad for parameter in loaded.module.parameters()))


if __name__ == "__main__":
    unittest.main()
