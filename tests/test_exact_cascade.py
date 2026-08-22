from __future__ import annotations

import unittest

from immer.composition import CompositionRoot, compose_runtime
from immer.contracts import ExecutionStatus, Request, Result
from immer.cognition.exact_cascade import ExactCascade


class StubBackend:
    capabilities = frozenset({"exact_math"})

    def __init__(self, name: str, result: Result, order: list[str] | None = None) -> None:
        self.name = name
        self.result = result
        self.order = order
        self.requests: list[Request] = []

    def handle(self, request: Request) -> Result:
        self.requests.append(request)
        if self.order is not None:
            self.order.append(self.name)
        return self.result


def _ok(name: str, output, *, crystal: bool = False) -> Result:
    evidence = {"crystal_verified": True, "no_training": True} if crystal else {}
    return Result(ExecutionStatus.OK, name, output=output, evidence=evidence)


def _abstain(name: str) -> Result:
    return Result(ExecutionStatus.ABSTAINED, name, reason="no exact answer")


class ExactCascadeTests(unittest.TestCase):
    def test_consensus_normalizes_numeric_representations_and_preserves_order(self) -> None:
        order: list[str] = []
        cascade = ExactCascade(
            StubBackend("s3", _ok("s3", 8, crystal=True), order),
            StubBackend("fertig", _ok("fertig", "eight"), order),
        )

        result = cascade.handle(Request("exact_math", "3 + 5"))

        self.assertEqual(order, ["s3", "fertig"])
        self.assertIs(result.status, ExecutionStatus.OK)
        self.assertEqual(result.output, 8)
        self.assertEqual(result.evidence["route"], "consensus")

    def test_crystal_verified_organ_is_accepted_when_fertig_abstains(self) -> None:
        cascade = ExactCascade(
            StubBackend("s3", _ok("s3", 12, crystal=True)),
            StubBackend("fertig", _abstain("fertig")),
        )

        result = cascade.handle(Request("exact_math", "3 times 4"))

        self.assertIs(result.status, ExecutionStatus.OK)
        self.assertEqual(result.output, 12)
        self.assertEqual(result.evidence["route"], "s3_crystal")

    def test_fertig_is_used_when_s3_abstains(self) -> None:
        cascade = ExactCascade(
            StubBackend("s3", _abstain("s3")),
            StubBackend("fertig", _ok("fertig", 42)),
        )

        result = cascade.handle(Request("exact_math", "What is the answer?"))

        self.assertIs(result.status, ExecutionStatus.OK)
        self.assertEqual(result.output, 42)
        self.assertEqual(result.evidence["route"], "fertig_fallback")

    def test_crystal_organ_wins_guarded_disagreement(self) -> None:
        cascade = ExactCascade(
            StubBackend("s3", _ok("s3", 8, crystal=True)),
            StubBackend("fertig", _ok("fertig", 5)),
        )

        result = cascade.handle(Request("exact_math", "3 plus 5"))

        self.assertIs(result.status, ExecutionStatus.OK)
        self.assertEqual(result.output, 8)
        self.assertEqual(result.evidence["route"], "s3_crystal_override")
        self.assertTrue(result.evidence["disagreement_guarded"])

    def test_unverified_disagreement_abstains(self) -> None:
        cascade = ExactCascade(
            StubBackend("s3", _ok("s3", 8)),
            StubBackend("fertig", _ok("fertig", 5)),
        )

        result = cascade.handle(Request("exact_math", "3 plus 5"))

        self.assertIs(result.status, ExecutionStatus.ABSTAINED)
        self.assertIsNone(result.output)

    def test_known_apple_failure_cannot_escape_as_ok(self) -> None:
        cascade = ExactCascade(
            StubBackend("s3", _abstain("s3")),
            StubBackend("fertig", _ok("fertig", "5")),
        )
        question = "John has 5 apples and buys 3 more. How many apples does he have?"

        result = cascade.handle(Request("exact_math", question))

        self.assertIs(result.status, ExecutionStatus.ABSTAINED)
        self.assertIsNone(result.output)
        self.assertEqual(result.evidence["fallback_guard"]["expected"], "8")

    def test_quantity_guard_allows_a_correct_fertig_answer(self) -> None:
        cascade = ExactCascade(
            StubBackend("s3", _abstain("s3")),
            StubBackend("fertig", _ok("fertig", 8)),
        )
        question = "John has 5 apples and buys 3 more. How many apples does he have?"

        result = cascade.handle(Request("exact_math", question))

        self.assertIs(result.status, ExecutionStatus.OK)
        self.assertTrue(result.evidence["fallback_guard"]["validated"])

    def test_both_abstentions_are_aggregated(self) -> None:
        cascade = ExactCascade(
            StubBackend("s3", _abstain("s3")),
            StubBackend("fertig", _abstain("fertig")),
        )

        result = cascade.handle(Request("exact_math", "was ist liebe"))

        self.assertIs(result.status, ExecutionStatus.ABSTAINED)
        self.assertEqual(result.evidence["route"], "no_answer")

    def test_solve_only_backend_and_child_exception_are_contained(self) -> None:
        class SolveOnly:
            name = "s3-solve"

            def solve(self, text: str):
                return None

        class Broken:
            name = "fertig-broken"

            def handle(self, request: Request) -> Result:
                raise RuntimeError("boom")

        cascade = ExactCascade(SolveOnly(), Broken())

        result = cascade.handle(Request("exact_math", "1 + 1"))

        self.assertIs(result.status, ExecutionStatus.ABSTAINED)
        self.assertEqual(result.evidence["fertig"]["status"], "error")


class CompositionRootTests(unittest.TestCase):
    def test_exact_cascade_is_the_only_registered_exact_math_owner(self) -> None:
        stream = object()
        s3 = StubBackend("s3", _abstain("s3"))
        root = CompositionRoot.build(
            s3_arithmetic=s3,
            fertig=StubBackend("fertig", _ok("fertig", 2)),
            life_stream=stream,  # type: ignore[arg-type]
        )

        self.assertEqual(root.runtime.registry.capabilities(), ("exact_math",))
        self.assertIs(root.runtime.registry.get("exact_math"), root.exact_math)
        self.assertIs(root.life_stream, stream)
        result = root.dispatch("exact_math", "one plus one", {"trace_id": "t-1"})
        self.assertTrue(result.ok)
        self.assertEqual(s3.requests[0].metadata["trace_id"], "t-1")

    def test_learning_stream_cannot_alias_a_frozen_exact_backend(self) -> None:
        backend = StubBackend("s3", _abstain("s3"))

        with self.assertRaisesRegex(ValueError, "learning life stream"):
            CompositionRoot.build(
                s3_arithmetic=backend,
                fertig=StubBackend("fertig", _abstain("fertig")),
                life_stream=backend,  # type: ignore[arg-type]
            )

    def test_grounded_chat_is_composed_without_duplicating_exact_math(self) -> None:
        grounded = StubBackend(
            "fertig.grounded",
            Result(ExecutionStatus.OK, "fertig.grounded", output="grounded"),
        )
        grounded.capabilities = frozenset({"grounded_chat"})
        root = CompositionRoot.build(
            s3_arithmetic=StubBackend("s3", _abstain("s3")),
            fertig=StubBackend("fertig", _abstain("fertig")),
            grounded_chat=grounded,
        )

        self.assertEqual(
            root.runtime.registry.capabilities(),
            ("exact_math", "grounded_chat"),
        )
        self.assertEqual(
            root.dispatch("grounded_chat", "help").output,
            "grounded",
        )

    def test_grounded_factory_configuration_is_unambiguous(self) -> None:
        grounded = StubBackend("grounded", _abstain("grounded"))
        grounded.capabilities = frozenset({"grounded_chat"})
        with self.assertRaisesRegex(ValueError, "grounded_chat"):
            CompositionRoot.build(
                s3_arithmetic=StubBackend("s3", _abstain("s3")),
                fertig=StubBackend("fertig", _abstain("fertig")),
                grounded_chat=grounded,
                fertig_state_dir="state",
            )

    def test_general_chat_is_injected_without_cross_capability_fallback(self) -> None:
        s3 = StubBackend("s3", _abstain("s3"))
        fertig = StubBackend("fertig", _ok("fertig", 2))
        chat = StubBackend(
            "deepseek-v4.chat",
            Result(ExecutionStatus.OK, "deepseek-v4.chat", output="frontier"),
        )
        chat.capabilities = frozenset({"chat"})
        grounded = StubBackend(
            "fertig.grounded",
            Result(ExecutionStatus.OK, "fertig.grounded", output="grounded"),
        )
        grounded.capabilities = frozenset({"grounded_chat"})

        root = compose_runtime(
            s3_arithmetic=s3,
            fertig=fertig,
            grounded_chat=grounded,
            general_chat=chat,
        )

        self.assertIs(root.general_chat, chat)
        self.assertEqual(
            root.runtime.registry.capabilities(),
            ("chat", "exact_math", "grounded_chat"),
        )
        self.assertIs(root.runtime.registry.get("exact_math"), root.exact_math)
        self.assertIs(root.runtime.registry.get("chat"), chat)
        self.assertIs(root.runtime.registry.get("grounded_chat"), grounded)
        exact = root.dispatch("exact_math", "one plus one")
        self.assertTrue(exact.ok)
        self.assertFalse(chat.requests)
        self.assertFalse(grounded.requests)
        exact_request_counts = (len(s3.requests), len(fertig.requests))
        general = root.dispatch("chat", "hello")
        self.assertEqual(general.output, "frontier")
        self.assertEqual(len(chat.requests), 1)
        self.assertFalse(grounded.requests)
        self.assertEqual((len(s3.requests), len(fertig.requests)), exact_request_counts)


if __name__ == "__main__":
    unittest.main()
