from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

from immer.runtimes.ooe.crystal import CrystalStore
from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.qwen_warm_multislot import (
    MultiSlotObservation,
    derive_multislot_observations,
    promote_multislot,
)
from immer.runtimes.ooe.qwen_warm_templates import (
    LEGACY_TEMPLATE_STATE_NAME,
    LEGACY_TEMPLATE_STATE_SCHEMA,
    TEMPLATE_STATE_NAME,
    TEMPLATE_STATE_SCHEMA,
    ParametricWarmBank,
    ParametricWarmError,
    TemplateObservation,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _derive(question: str, output: str, tag: str):
    return derive_multislot_observations(
        question,
        output,
        question_sha256=_hash(question),
        cell_payload_sha256=_hash(f"cell:{tag}"),
        teacher_forward_count=4,
    )


def _envelope(body: dict[str, object], schema: str) -> bytes:
    return canonical_json_bytes(
        {
            "body": body,
            "schema": schema,
            "sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
        }
    )


class MultiSlotWarmTests(unittest.TestCase):
    def test_two_examples_promote_learned_delimiter_for_unseen_pair(self) -> None:
        observations = (
            *_derive("join ALPHA and BETA", "ALPHA-BETA", "first"),
            *_derive("join GAMMA and DELTA", "GAMMA-DELTA", "second"),
        )
        programs = promote_multislot(observations)
        outputs = {
            output
            for program in programs
            if (output := program.match("join EPSILON and ZETA")) is not None
        }
        self.assertEqual(outputs, {"EPSILON-ZETA"})

    def test_reordered_uppercase_slots_are_induced(self) -> None:
        observations = (
            *_derive("swap alpha then beta", "BETA/ALPHA", "first"),
            *_derive("swap gamma then delta", "DELTA/GAMMA", "second"),
        )
        programs = promote_multislot(observations)
        outputs = {
            output
            for program in programs
            if (output := program.match("swap epsilon then zeta")) is not None
        }
        self.assertEqual(outputs, {"ZETA/EPSILON"})

    def test_one_example_and_repeated_same_tuple_do_not_promote(self) -> None:
        first = _derive("join A and B", "A-B", "first")
        self.assertEqual(promote_multislot(first), ())
        repeated = (*first, *_derive("join A and B", "A-B", "second"))
        self.assertEqual(promote_multislot(repeated), ())

    def test_context_mismatch_and_ambiguous_programs_abstain_upstream(self) -> None:
        observations = (
            *_derive("join A and B", "A-B", "first"),
            *_derive("join C and D", "C-D", "second"),
        )
        programs = promote_multislot(observations)
        self.assertTrue(programs)
        self.assertTrue(
            all(program.match("different E and F") is None for program in programs)
        )

    def test_conflicting_promoted_programs_abstain_in_the_warm_bank(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile = _hash("profile")

            def token(question: str) -> str:
                return _hash(f"tokens:{question}")

            bank = ParametricWarmBank(
                CrystalStore(temporary),
                profile,
                output_character_limit=32,
                prompt_token_verifier=(
                    lambda question, claimed: claimed == token(question)
                ),
            )
            for index, (question, output) in enumerate(
                (
                    ("join A and B", "A-B"),
                    ("join C and D", "C-D"),
                    ("join G and H", "H/G"),
                    ("join I and J", "J/I"),
                )
            ):
                bank.observe(
                    question,
                    output,
                    question_sha256=_hash(question),
                    cell_payload_sha256=_hash(f"cell:{index}"),
                    teacher_forward_count=4,
                )
            unseen = "join E and F"
            outputs = {
                output
                for program in bank.promoted
                if (output := program.match(unseen)) is not None
            }
            self.assertEqual(outputs, {"E-F", "F/E"})
            self.assertIsNone(
                bank.try_warm(
                    unseen,
                    {
                        "qwen_token_sha256": token(unseen),
                        "qwen_warm_runtime_profile_sha256": profile,
                    },
                )
            )

    def test_observation_roundtrip_and_tamper_rejection(self) -> None:
        row = _derive("join A and B", "A-B", "first")[0]
        document = row.to_dict()
        restored = MultiSlotObservation.from_dict(document)
        self.assertEqual(restored, row)
        tampered = {**document, "output_order": [0, 0]}
        with self.assertRaises(Exception):
            MultiSlotObservation.from_dict(tampered)
        with self.assertRaises(Exception):
            MultiSlotObservation.from_dict(
                {**document, "output_middle": "tampered"}
            )
        with self.assertRaises(Exception):
            MultiSlotObservation.from_dict(
                {**document, "slot_values": ["X", "Y"]}
            )

    def test_distinct_cells_are_required_for_promotion(self) -> None:
        shared_cell = _hash("shared-cell")
        observations = tuple(
            replace(row, cell_payload_sha256=shared_cell)
            for row in (
                *_derive("join A and B", "A-B", "first"),
                *_derive("join C and D", "C-D", "second"),
            )
        )
        self.assertEqual(promote_multislot(observations), ())

    def test_both_slot_positions_must_vary_before_promotion(self) -> None:
        observations = (
            *_derive("join A and B", "A-B", "first"),
            *_derive("join A and C", "A-C", "second"),
        )
        self.assertEqual(promote_multislot(observations), ())

    def test_resealed_descriptor_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(temporary)
            profile = _hash("profile")
            bank = ParametricWarmBank(
                store,
                profile,
                output_character_limit=32,
            )
            question = "join ALPHA and BETA"
            bank.observe(
                question,
                "ALPHA-BETA",
                question_sha256=_hash(question),
                cell_payload_sha256=_hash("cell"),
                teacher_forward_count=4,
            )
            raw = store.restore_state(TEMPLATE_STATE_NAME)
            document = json.loads(raw)
            rows = document["body"]["multislot_observations"]
            self.assertGreater(len(rows), 1)
            rows.pop()
            tampered = _envelope(document["body"], TEMPLATE_STATE_SCHEMA)
            store.publish_state(
                TEMPLATE_STATE_NAME,
                tampered,
                expected_sha256=hashlib.sha256(raw).hexdigest(),
            )
            with self.assertRaises(ParametricWarmError):
                ParametricWarmBank(
                    store,
                    profile,
                    output_character_limit=32,
                )

    def test_capacity_admits_or_drops_one_complete_cold_source(self) -> None:
        first_question = "join ALPHA and BETA"
        first_rows = _derive(first_question, "ALPHA-BETA", "first")
        self.assertGreater(len(first_rows), 1)
        with tempfile.TemporaryDirectory() as temporary, patch(
            "immer.runtimes.ooe.qwen_warm_templates."
            "MAX_MULTISLOT_OBSERVATIONS",
            len(first_rows),
        ):
            bank = ParametricWarmBank(
                CrystalStore(temporary),
                _hash("profile"),
                output_character_limit=32,
            )
            first = bank.observe(
                first_question,
                "ALPHA-BETA",
                question_sha256=_hash(first_question),
                cell_payload_sha256=_hash("cell:first"),
                teacher_forward_count=4,
            )
            self.assertEqual(first["multislot_admitted"], len(first_rows))
            self.assertEqual(first["multislot_capacity_dropped"], 0)

            second_question = "join GAMMA and DELTA"
            second_rows = _derive(second_question, "GAMMA-DELTA", "second")
            second = bank.observe(
                second_question,
                "GAMMA-DELTA",
                question_sha256=_hash(second_question),
                cell_payload_sha256=_hash("cell:second"),
                teacher_forward_count=4,
            )
            self.assertEqual(second["multislot_admitted"], 0)
            self.assertEqual(
                second["multislot_capacity_dropped"],
                len(second_rows),
            )
            self.assertEqual(len(bank.multislot_observations), len(first_rows))
            self.assertEqual(bank.promoted, ())

    def test_cross_profile_import_is_all_or_none_at_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = ParametricWarmBank(
                CrystalStore(f"{temporary}/source"),
                _hash("source-profile"),
                output_character_limit=32,
            )
            for index, (question, output) in enumerate(
                (
                    ("join ALPHA and BETA", "ALPHA-BETA"),
                    ("join GAMMA and DELTA", "GAMMA-DELTA"),
                )
            ):
                source.observe(
                    question,
                    output,
                    question_sha256=_hash(question),
                    cell_payload_sha256=_hash(f"cell:{index}"),
                    teacher_forward_count=4,
                )
            first_source_size = len(
                _derive("join ALPHA and BETA", "ALPHA-BETA", "first")
            )
            self.assertGreater(
                len(source.multislot_observations),
                first_source_size,
            )
            with patch(
                "immer.runtimes.ooe.qwen_warm_templates."
                "MAX_MULTISLOT_OBSERVATIONS",
                first_source_size,
            ):
                destination = ParametricWarmBank(
                    CrystalStore(f"{temporary}/destination"),
                    _hash("destination-profile"),
                    output_character_limit=32,
                )
                report = destination.import_compatible(source)
            self.assertEqual(report["capacity_rejected_source_states"], 1)
            self.assertEqual(report["source_states"], 0)
            self.assertEqual(destination.multislot_observations, ())
            self.assertEqual(destination.observations, ())
            self.assertEqual(destination.body["imported_content_sha256s"], [])

    def test_v1_state_migrates_without_losing_single_slot_programs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(temporary)
            profile = _hash("profile")
            alpha_question = "copy:ALPHA"
            beta_question = "copy:BETA"
            observations = [
                TemplateObservation(
                    prefix="copy:",
                    suffix="",
                    mode="identity",
                    slot=slot,
                    output=slot,
                    question_sha256=_hash(question),
                    cell_payload_sha256=_hash(f"cell:{slot}"),
                    teacher_forward_count=3,
                ).to_dict()
                for question, slot in (
                    (alpha_question, "ALPHA"),
                    (beta_question, "BETA"),
                )
            ]
            legacy_body = {
                "committed_executions": 2,
                "next_transaction": 3,
                "observations": observations,
                "output_character_limit": 32,
                "privacy": "private-executable-template-descriptors/v1",
                "rejected_executions": 1,
                "runtime_profile_sha256": profile,
                "saved_qwen_forwards": 6,
                "schema": LEGACY_TEMPLATE_STATE_SCHEMA,
            }
            store.publish_state(
                LEGACY_TEMPLATE_STATE_NAME,
                _envelope(legacy_body, LEGACY_TEMPLATE_STATE_SCHEMA),
            )

            def token(question: str) -> str:
                return _hash(f"tokens:{question}")

            migrated = ParametricWarmBank(
                store,
                profile,
                output_character_limit=32,
                prompt_token_verifier=(
                    lambda question, claimed: claimed == token(question)
                ),
            )
            self.assertEqual(
                json.loads(store.restore_state(TEMPLATE_STATE_NAME))["schema"],
                LEGACY_TEMPLATE_STATE_SCHEMA,
            )
            unseen = "copy:GAMMA"
            attempt = migrated.try_warm(
                unseen,
                {
                    "qwen_token_sha256": token(unseen),
                    "qwen_warm_runtime_profile_sha256": profile,
                },
            )
            self.assertIsNotNone(attempt)
            assert attempt is not None and attempt.result is not None
            self.assertEqual(attempt.result.output, "GAMMA")

            destination_profile = _hash("destination-profile")
            destination = ParametricWarmBank(
                CrystalStore(f"{temporary}/destination"),
                destination_profile,
                output_character_limit=32,
                prompt_token_verifier=(
                    lambda question, claimed: claimed == token(question)
                ),
            )
            imported = destination.import_compatible(migrated)
            self.assertEqual(imported["imported_single_slot_observations"], 2)
            self.assertEqual(imported["source_states"], 1)
            imported_question = "copy:DELTA"
            imported_attempt = destination.try_warm(
                imported_question,
                {
                    "qwen_token_sha256": token(imported_question),
                    "qwen_warm_runtime_profile_sha256": destination_profile,
                },
            )
            self.assertIsNotNone(imported_attempt)
            assert imported_attempt is not None and imported_attempt.result is not None
            self.assertEqual(imported_attempt.result.output, "DELTA")
            self.assertEqual(destination.body["committed_executions"], 0)
            destination_sha256 = destination.state_sha256
            repeated_import = destination.import_compatible(migrated)
            self.assertEqual(repeated_import["source_states"], 1)
            self.assertEqual(
                repeated_import["imported_single_slot_observations"],
                0,
            )
            self.assertEqual(destination.state_sha256, destination_sha256)

            receipt = attempt._settler(True)
            self.assertEqual(receipt.saved_qwen_forwards, 3)

            current = json.loads(store.restore_state(TEMPLATE_STATE_NAME))
            self.assertEqual(LEGACY_TEMPLATE_STATE_NAME, TEMPLATE_STATE_NAME)
            self.assertEqual(len(tuple((store.root / "state").iterdir())), 1)
            self.assertEqual(current["schema"], TEMPLATE_STATE_SCHEMA)
            self.assertEqual(current["body"]["multislot_observations"], [])
            self.assertEqual(current["body"]["committed_executions"], 3)
            self.assertEqual(current["body"]["rejected_executions"], 1)
            self.assertEqual(current["body"]["saved_qwen_forwards"], 9)
            restarted = ParametricWarmBank(
                store,
                profile,
                output_character_limit=32,
                prompt_token_verifier=(
                    lambda question, claimed: claimed == token(question)
                ),
            )
            self.assertEqual(len(restarted.observations), 2)
            self.assertEqual(restarted.body["next_transaction"], 4)

    def test_non_ascii_and_output_without_two_slots_abstain(self) -> None:
        self.assertEqual(_derive("füge A und B", "A-B", "unicode"), ())
        self.assertEqual(_derive("join A and B", "constant", "constant"), ())

    def test_saturated_derivation_abstains_instead_of_storing_a_prefix(self) -> None:
        question = "mix " + " ".join(["AB"] * 40)
        self.assertEqual(_derive(question, "ABAB", "saturated"), ())


if __name__ == "__main__":
    unittest.main()
