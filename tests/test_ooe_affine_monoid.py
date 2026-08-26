from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from unittest import mock

from immer.runtimes.ooe.affine_monoid import (
    AffineDomainError,
    AffineActionAtom,
    AffineMonoidBank,
    AffineIntegrityError,
    AffineMonoidRuntime,
    AffineProgram,
    AffineState,
    CompositionReceipt,
    ExecutionReceipt,
    ExecutionBundle,
    FingerprintVerificationReceipt,
    ProgramReceipt,
    StateReceipt,
    build_anbncn_machine,
    build_decimal_horner_machine,
    build_fingerprint_machine,
    build_group_map_bridge,
    build_stack_machine,
    compose_actions,
    verify_fingerprint_equality,
)
from immer.runtimes.ooe.crystal import CrystalStore
from immer.runtimes.ooe.identity import canonical_json_bytes


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class ExactAffineMonoidTests(unittest.TestCase):
    def test_stack_recovers_buried_unique_symbols_at_unseen_depths(self) -> None:
        stack = build_stack_machine(capacity=64, symbols=tuple(range(1, 65)))
        # The constructor has no examples or learned depth distribution.  Both
        # depths are future executions of the same fixed action family.
        for depth in (17, 53):
            symbols = tuple(range(1, depth + 1))
            actions = tuple(stack.push(symbol) for symbol in symbols)
            state = AffineMonoidRuntime.execute(
                AffineProgram(stack.schema, actions, f"StackDepth{depth}")
            ).state
            recovered: list[int] = []
            for _ in range(depth):
                recovered.append(stack.top(state))
                state = stack.action("pop").apply(stack.schema, state)
            self.assertEqual(recovered, list(reversed(symbols)))
            self.assertEqual(state, stack.schema.state())

    def test_stack_capacity_k_succeeds_and_k_plus_one_dies_absorbingly(self) -> None:
        stack = build_stack_machine(capacity=5, symbols=(7, 11))
        full = AffineProgram(stack.schema, tuple(stack.push(7) for _ in range(5)))
        full_result = AffineMonoidRuntime.execute(full)
        self.assertFalse(stack.schema.is_dead(full_result.state))
        overflow = stack.push(11).apply(stack.schema, full_result.state)
        self.assertTrue(stack.schema.is_dead(overflow))
        self.assertEqual(stack.action("pop").apply(stack.schema, overflow), overflow)
        underflow = stack.action("pop").apply(stack.schema, stack.schema.state())
        self.assertTrue(stack.schema.is_dead(underflow))

    def test_pop_after_push_is_identity_only_on_the_valid_subspace(self) -> None:
        stack = build_stack_machine(capacity=3, symbols=(1, 2, 9))
        reachable = AffineMonoidRuntime.execute(
            AffineProgram(
                stack.schema, (stack.push(1), stack.push(2)), "ReachableStack"
            )
        ).state
        fused, receipt = compose_actions(
            stack.schema, stack.push(9), stack.action("pop"), name="PushPop9"
        )
        sequential = stack.action("pop").apply(
            stack.schema, stack.push(9).apply(stack.schema, reachable)
        )
        self.assertEqual(fused.apply(stack.schema, reachable), sequential)
        self.assertEqual(sequential, reachable)
        self.assertEqual(CompositionReceipt.from_bytes(receipt.to_bytes()), receipt)

        full = AffineMonoidRuntime.execute(
            AffineProgram(
                stack.schema,
                (stack.push(1), stack.push(2), stack.push(9)),
                "FullStack",
            )
        ).state
        self.assertTrue(stack.schema.is_dead(fused.apply(stack.schema, full)))

        # Bounds alone are not enough: nonzero slack below the declared depth
        # is outside the LIFO representation and is rejected by the stage guard.
        invalid_slack = stack.schema.state((2, 1, 9, 2, 0))
        self.assertTrue(stack.schema.is_dead(fused.apply(stack.schema, invalid_slack)))

    def test_anbncn_accepts_unseen_counts_and_rejects_order_confusers(self) -> None:
        machine = build_anbncn_machine(max_count=128)
        for count in (1, 12, 47, 100):
            result = machine.run("a" * count + "b" * count + "c" * count)
            self.assertTrue(machine.accepts(result.state), count)

        for placebo in (
            "",
            "acb",  # explicit a-c-b order failure
            "abcabc",  # equal global counts, wrong phases
            "aabccb",  # equal counts with a late b
            "aaabbbcc",
            "aabbbccc",
            "aaabbbcccc",
            "abxc",
        ):
            self.assertFalse(machine.accepts(machine.run(placebo).state), placebo)

    def test_anbncn_count_capacity_and_dead_phase_are_exact(self) -> None:
        machine = build_anbncn_machine(max_count=8)
        self.assertTrue(machine.accepts(machine.run("a" * 8 + "b" * 8 + "c" * 8).state))
        over = machine.run("a" * 9 + "b" * 9 + "c" * 9)
        self.assertTrue(machine.schema.is_dead(over.state))
        after_dead = machine.action("a").apply(machine.schema, over.state)
        self.assertEqual(after_dead, over.state)

    def test_dual_fingerprint_binds_length_separator_and_phase(self) -> None:
        machine = build_fingerprint_machine(max_length=16, base=2, moduli=(3, 5))
        # These have the same two modular hashes: 1 mod 3 and 1 mod 5.  Length
        # binding prevents the false candidate.
        unequal_length_collision = machine.compare(b"\x00", b"\x00\x0d")
        self.assertFalse(machine.candidate_equal(unequal_length_collision.state))

        malformed = AffineProgram(
            machine.schema,
            (
                machine.left(1),
                machine.separator(),
                machine.separator(),
                machine.right(1),
                machine.finish(),
            ),
            "MalformedSeparators",
        )
        malformed_result = AffineMonoidRuntime.execute(malformed)
        self.assertTrue(machine.schema.is_dead(malformed_result.state))

        wrong_phase = AffineMonoidRuntime.execute(
            AffineProgram(
                machine.schema,
                (machine.right(1), machine.separator(), machine.left(1)),
                "WrongFingerprintPhase",
            )
        )
        self.assertTrue(machine.schema.is_dead(wrong_phase.state))

    def test_modular_collision_requires_and_survives_exact_verifier_boundary(
        self,
    ) -> None:
        machine = build_fingerprint_machine(max_length=8, base=2, moduli=(3, 5))
        # value+1 differs by lcm(3,5), so the dual modular state collides at
        # equal length.  The affine stage may only call this a candidate.
        left = b"\x00"
        collision = b"\x0f"
        result = machine.compare(left, collision)
        self.assertTrue(machine.candidate_equal(result.state))
        verification = verify_fingerprint_equality(
            machine,
            result,
            left=left,
            right=collision,
            verifier_sha256=_hash("exact-byte-verifier"),
        )
        self.assertTrue(verification.modular_candidate)
        self.assertFalse(verification.exact_equal)
        self.assertEqual(
            FingerprintVerificationReceipt.from_bytes(verification.to_bytes()),
            verification,
        )

        exact = machine.compare(b"same", b"same")
        exact_receipt = verify_fingerprint_equality(
            machine,
            exact,
            left=b"same",
            right=b"same",
            verifier_sha256=_hash("exact-byte-verifier"),
        )
        self.assertTrue(exact_receipt.exact_equal)
        with self.assertRaises(AffineIntegrityError):
            verify_fingerprint_equality(
                machine,
                exact,
                left=b"same",
                right=b"tampered",
                verifier_sha256=_hash("exact-byte-verifier"),
            )

    def test_decimal_horner_exact_examples_and_128_digit_bound(self) -> None:
        parser = build_decimal_horner_machine(max_digits=128)
        for text, expected in (
            ("12", 12),
            ("21", 21),
            ("90", 90),
            ("100", 100),
            ("+100", 100),
            ("-100", -100),
            ("0", 0),
        ):
            result = parser.parse(text)
            self.assertFalse(parser.schema.is_dead(result.state), text)
            self.assertEqual(parser.value(result.state), expected)

        digits_128 = "9" * 128
        accepted = parser.parse(digits_128)
        self.assertEqual(parser.value(accepted.state), int(digits_128))
        self.assertTrue(parser.schema.is_dead(parser.parse("1" * 129).state))

    def test_decimal_grammar_placebos_die_instead_of_coercing(self) -> None:
        parser = build_decimal_horner_machine(max_digits=32)
        for malformed in (
            "",
            "+",
            "-",
            "00",
            "01",
            "--1",
            "+-1",
            "1+2",
            "1_000",
            " 12",
            "12 ",
            "١٢",
        ):
            self.assertTrue(
                parser.schema.is_dead(parser.parse(malformed).state), malformed
            )

    def test_exact_composition_matches_sequential_on_future_initial_states(
        self,
    ) -> None:
        additive = build_group_map_bridge("additive", max_abs_bits=256)
        first = additive.action(12)
        second = additive.action(21)
        fused, receipt = compose_actions(
            additive.schema, first, second, name="Add12Then21"
        )
        for future in (-(10**30), -7, 0, 90, 10**30):
            state = additive.schema.state((future, 0))
            sequential = second.apply(
                additive.schema, first.apply(additive.schema, state)
            )
            self.assertEqual(fused.apply(additive.schema, state), sequential)
            self.assertEqual(additive.value(sequential), future + 33)
        self.assertEqual(receipt.first_action_sha256, first.sha256)

        multiplicative = build_group_map_bridge(
            "multiplicative", initial=2, max_abs_bits=128
        )
        product, _ = compose_actions(
            multiplicative.schema,
            multiplicative.action(12),
            multiplicative.action(21),
            name="Mul12Then21",
        )
        self.assertEqual(
            multiplicative.value(
                product.apply(multiplicative.schema, multiplicative.schema.state())
            ),
            504,
        )

        cyclic = build_group_map_bridge("cyclic", modulus=90, initial=89)
        wrapped, _ = compose_actions(
            cyclic.schema,
            cyclic.action(12),
            cyclic.action(100),
            name="Z90Add112",
        )
        self.assertEqual(
            cyclic.value(wrapped.apply(cyclic.schema, cyclic.schema.state())),
            21,
        )

    def test_bit_bounds_fail_closed_for_integer_group_maps(self) -> None:
        bridge = build_group_map_bridge("multiplicative", initial=1, max_abs_bits=8)
        state = bridge.action(100).apply(bridge.schema, bridge.schema.state())
        self.assertFalse(bridge.schema.is_dead(state))
        overflow = bridge.action(100).apply(bridge.schema, state)
        self.assertTrue(bridge.schema.is_dead(overflow))

    def test_multiplicative_bridge_excludes_zero_from_state_and_actions(self) -> None:
        with self.assertRaises(AffineDomainError):
            build_group_map_bridge("multiplicative", initial=0)
        bridge = build_group_map_bridge("multiplicative", initial=1)
        with self.assertRaises(AffineDomainError):
            bridge.schema.state((0, 0))
        with self.assertRaises(AffineDomainError):
            bridge.action(0)
        self.assertEqual(
            bridge.value(bridge.action(-7).apply(bridge.schema, bridge.schema.state())),
            -7,
        )

    def test_execution_bundle_is_one_replay_verified_sealed_payload(self) -> None:
        bridge = build_group_map_bridge("additive")
        initial = bridge.schema.state((90, 0))
        program = AffineProgram(
            bridge.schema,
            (bridge.action(12), bridge.action(21), bridge.action(100)),
            "AtomicBundle",
        )
        result = AffineMonoidRuntime.execute(
            program,
            initial_state=initial,
            verifier_sha256s=(_hash("bundle-verifier"),),
        )
        bundle = ExecutionBundle.capture(program, result, initial_state=initial)
        self.assertEqual(ExecutionBundle.from_bytes(bundle.to_bytes()), bundle)
        self.assertEqual(bundle.final_state.values, (223, 0))

        spliced = json.loads(bundle.to_bytes())
        spliced["body"]["final_state"]["values"][0] = 224
        spliced["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(spliced["body"])
        ).hexdigest()
        with self.assertRaises(AffineIntegrityError):
            ExecutionBundle.from_bytes(canonical_json_bytes(spliced))

    def test_affine_bank_program_bundle_idempotence_and_atomic_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(temporary)
            bank = AffineMonoidBank(store)
            bridge = build_group_map_bridge("additive")
            program = AffineProgram(
                bridge.schema,
                (bridge.action(12), bridge.action(21)),
                "BankProgram",
            )
            first_program = bank.publish_program(program)
            second_program = bank.publish_program(program)
            self.assertTrue(first_program.changed)
            self.assertFalse(second_program.changed)
            self.assertEqual(bank.restore_program(program.sha256), program)

            initial = bridge.schema.state((90, 0))
            result = AffineMonoidRuntime.execute(program, initial_state=initial)
            bundle = ExecutionBundle.capture(program, result, initial_state=initial)
            original_publish = store.publish_state
            with mock.patch.object(
                store, "publish_state", wraps=original_publish
            ) as publish:
                first_bundle = bank.publish_bundle(bundle)
                self.assertEqual(publish.call_count, 1)
            second_bundle = bank.publish_bundle(bundle)
            self.assertTrue(first_bundle.changed)
            self.assertFalse(second_bundle.changed)
            self.assertEqual(bank.restore_bundle(bundle.sha256), bundle)

    def test_affine_bank_crash_retry_and_tamper_fail_closed(self) -> None:
        bridge = build_group_map_bridge("additive")
        program = AffineProgram(bridge.schema, (bridge.action(12),), "CrashBundle")
        initial = bridge.schema.state((21, 0))
        result = AffineMonoidRuntime.execute(program, initial_state=initial)
        bundle = ExecutionBundle.capture(program, result, initial_state=initial)

        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(temporary)
            bank = AffineMonoidBank(store)
            with mock.patch.object(
                store, "publish_state", side_effect=RuntimeError("crash-before")
            ):
                with self.assertRaisesRegex(RuntimeError, "crash-before"):
                    bank.publish_bundle(bundle)
            with self.assertRaises(KeyError):
                bank.restore_bundle(bundle.sha256)
            self.assertTrue(bank.publish_bundle(bundle).changed)

            # A valid store write followed by process death is recoverable: the
            # complete bundle is already atomically visible and retry is a hit.
            second_initial = bridge.schema.state((22, 0))
            second_result = AffineMonoidRuntime.execute(
                program, initial_state=second_initial
            )
            second = ExecutionBundle.capture(
                program, second_result, initial_state=second_initial
            )
            original_publish = store.publish_state

            def crash_after(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
                original_publish(*args, **kwargs)
                raise RuntimeError("crash-after")

            with mock.patch.object(store, "publish_state", side_effect=crash_after):
                with self.assertRaisesRegex(RuntimeError, "crash-after"):
                    bank.publish_bundle(second)
            self.assertEqual(bank.restore_bundle(second.sha256), second)
            self.assertFalse(bank.publish_bundle(second).changed)

            state_name = bank.bundle_state_name(second.sha256)
            store.publish_state(state_name, b"{}")
            with self.assertRaises(AffineIntegrityError):
                bank.restore_bundle(second.sha256)

    def test_state_program_execution_and_receipts_roundtrip_canonically(self) -> None:
        bridge = build_group_map_bridge("additive")
        initial = bridge.schema.state((90, 0))
        restored_state = AffineState.from_bytes(
            initial.to_bytes(), schema=bridge.schema
        )
        self.assertEqual(restored_state, initial)
        program = AffineProgram(
            bridge.schema,
            (bridge.action(12), bridge.action(21), bridge.action(100)),
            "ReceiptRoundtrip",
        )
        restored_program = AffineProgram.from_bytes(program.to_bytes())
        self.assertEqual(restored_program, program)
        self.assertEqual(
            AffineActionAtom.from_bytes(
                program.actions[0].to_bytes(), schema=bridge.schema
            ),
            program.actions[0],
        )
        result = AffineMonoidRuntime.execute(
            restored_program,
            initial_state=restored_state,
            verifier_sha256s=(_hash("v1"), _hash("v2")),
        )
        self.assertEqual(bridge.value(result.state), 223)
        self.assertEqual(
            StateReceipt.from_bytes(result.state_receipt.to_bytes()),
            result.state_receipt,
        )
        self.assertEqual(
            ProgramReceipt.from_bytes(result.program_receipt.to_bytes()),
            result.program_receipt,
        )
        self.assertEqual(
            ExecutionReceipt.from_bytes(result.execution_receipt.to_bytes()),
            result.execution_receipt,
        )

    def test_state_and_program_tampering_fail_closed(self) -> None:
        bridge = build_group_map_bridge("additive")
        state_document = json.loads(bridge.schema.state().to_bytes())
        state_document["body"]["values"][0] = 99
        with self.assertRaises(AffineIntegrityError):
            AffineState.from_bytes(
                canonical_json_bytes(state_document), schema=bridge.schema
            )
        with self.assertRaises(AffineIntegrityError):
            AffineState.from_bytes(
                bridge.schema.state().to_bytes() + b"\n", schema=bridge.schema
            )

        program = AffineProgram(
            bridge.schema, (bridge.action(12), bridge.action(21)), "TamperProgram"
        )
        program_document = json.loads(program.to_bytes())
        # Resealing the outer body is insufficient: the fused matrix must still
        # equal the authenticated stage trace.
        program_document["body"]["actions"][0]["bias"][0] = 13
        program_document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(program_document["body"])
        ).hexdigest()
        with self.assertRaises(AffineIntegrityError):
            AffineProgram.from_bytes(canonical_json_bytes(program_document))

    def test_wrong_schema_and_order_placebos_are_rejected(self) -> None:
        additive = build_group_map_bridge("additive")
        cyclic = build_group_map_bridge("cyclic", modulus=90)
        with self.assertRaises(AffineDomainError):
            additive.action(12).apply(cyclic.schema, cyclic.schema.state())

        stack = build_stack_machine(capacity=4, symbols=(1, 2, 3, 4))
        state = AffineMonoidRuntime.execute(
            AffineProgram(
                stack.schema,
                tuple(stack.push(value) for value in (1, 2, 3, 4)),
                "LifoPlacebo",
            )
        ).state
        observed = []
        for _ in range(4):
            observed.append(stack.top(state))
            state = stack.action("pop").apply(stack.schema, state)
        self.assertEqual(observed, [4, 3, 2, 1])
        self.assertNotEqual(observed, [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
