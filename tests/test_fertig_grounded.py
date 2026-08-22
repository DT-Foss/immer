from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from immer.cognition.fertig import FertigGrounded
from immer.contracts import ExecutionStatus, Request


@dataclass(frozen=True)
class _Reply:
    status: str
    route: str
    text: str
    task: str | None = None
    data: object = None


class _FakeAssistant:
    def __init__(self, **paths: object) -> None:
        self.paths = paths

    def resolve(self, text: str) -> object:
        command, _, task = text.partition(" ")
        return SimpleNamespace(intent=command, task=task or None)


class _FakeChat:
    def __init__(self, assistant: _FakeAssistant, **options: object) -> None:
        self.assistant = assistant
        self.options = options
        self.calls: list[tuple[str, object, object]] = []

    def handle(
        self, text: str, desktop: object | None = None, recorder: object | None = None
    ) -> _Reply:
        self.calls.append((text, desktop, recorder))
        status, _, route = text.partition(" ")
        if status in {"do", "teach", "correct", "compose", "template"}:
            status, route = "ok", "desktop"
        route = route or "unknown"
        return _Reply(
            status,
            route,
            f"reply:{status}:{route}",
            "stored task" if route == "desktop" else None,
            {"path": Path("evidence.json"), "nested": (1, True)},
        )


class FertigGroundedTests(unittest.TestCase):
    def test_constructor_is_lazy_and_uses_only_explicit_state_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state"
            grounded = FertigGrounded(state)

            self.assertFalse(grounded.loaded)
            self.assertFalse(state.exists())
            self.assertEqual(
                grounded.task_store_path, state.resolve() / "desktop_tasks.json"
            )
            self.assertEqual(
                grounded.template_store_path,
                state.resolve() / "desktop_templates.json",
            )
            self.assertEqual(
                grounded.recordings_dir, state.resolve() / "desktop_recordings"
            )

    def test_real_vendored_help_list_math_and_unknown_run_offline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            grounded = FertigGrounded(tmp)

            helped = grounded.handle(Request("grounded_chat", "help"))
            listed = grounded.handle(Request("grounded_chat", "list tasks"))
            math = grounded.handle(
                Request(
                    "grounded_chat",
                    "A bakery sold 12 cakes on Monday and 15 cakes on Tuesday. "
                    "How many cakes did they sell in total?",
                )
            )
            unknown = grounded.handle(
                Request("grounded_chat", "beautiful weather today")
            )

            self.assertTrue(grounded.loaded)
            self.assertIs(helped.status, ExecutionStatus.OK)
            self.assertEqual(helped.evidence["route"], "help")
            self.assertIn("Capabilities", str(helped.output))
            self.assertIs(listed.status, ExecutionStatus.OK)
            self.assertEqual(listed.evidence["route"], "list")
            self.assertIs(math.status, ExecutionStatus.OK)
            self.assertEqual(math.evidence["route"], "math")
            self.assertEqual(math.evidence["data"]["answer"], "27")
            self.assertIs(unknown.status, ExecutionStatus.ABSTAINED)
            self.assertEqual(unknown.evidence["status"], "unknown")
            self.assertIsNone(unknown.output)

    def test_real_vendor_stores_round_trip_in_the_explicit_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "fertig-state"
            first = FertigGrounded(state)
            self.assertTrue(first.handle(Request("grounded_chat", "help")).ok)
            assistant = first._assistant
            self.assertIsNotNone(assistant)
            assistant.store.save()  # type: ignore[union-attr]
            assistant.template_store.save()  # type: ignore[union-attr]

            self.assertTrue(first.task_store_path.is_file())
            self.assertTrue(first.template_store_path.is_file())

            second = FertigGrounded(state)
            listed = second.handle(Request("grounded_chat", "list tasks"))
            self.assertTrue(listed.ok)
            self.assertEqual(listed.evidence["data"]["tasks"], ())

    def test_real_optional_graph_is_grounded_in_supplied_causal_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Lazy-load exactly the vendored surface before importing its
            # bundled causal writer for this self-contained fixture.
            warm = FertigGrounded(root / "warm")
            self.assertTrue(warm.handle(Request("grounded_chat", "help")).ok)
            from fertig._vendor.dotcausal import CausalWriter

            graph = root / "facts.causal"
            writer = CausalWriter()
            writer.add_triplet("smoking", "causes", "tar buildup", 0.9)
            writer.add_triplet("tar buildup", "causes", "lung damage", 0.8)
            writer.save(str(graph))

            grounded = FertigGrounded(root / "state", graph_path=graph)
            result = grounded.handle(
                Request("grounded_chat", "explain how smoking affects lung damage")
            )

            self.assertIs(result.status, ExecutionStatus.OK)
            self.assertEqual(result.evidence["route"], "graph")
            self.assertEqual(result.evidence["data"]["tool"], "speech")
            self.assertEqual(result.evidence["data"]["target"], "smoking")
            self.assertIn("tar buildup", str(result.output).casefold())

    def test_desktop_and_store_mutations_are_blocked_without_opt_in_backends(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            grounded = FertigGrounded(tmp)
            cases = {
                "do monthly report": "desktop",
                "teach monthly report": "recorder",
                "correct monthly report step=0": "recorder",
                'compose "publish report" = "open" + "export"': "mutation_opt_in",
                'template "send" from "fill" recipient=1': "mutation_opt_in",
            }

            for text, missing in cases.items():
                with self.subTest(text=text):
                    result = grounded.handle(Request("grounded_chat", text))
                    self.assertIs(result.status, ExecutionStatus.ABSTAINED)
                    self.assertEqual(result.evidence["status"], "needs_input")
                    self.assertEqual(
                        result.evidence["data"]["missing_backend"], missing
                    )

            self.assertFalse(grounded.task_store_path.exists())
            self.assertFalse(grounded.template_store_path.exists())

    def test_typed_chat_statuses_and_read_only_routes_map_to_results(self) -> None:
        expected = {
            "ok": ExecutionStatus.OK,
            "unknown": ExecutionStatus.ABSTAINED,
            "ambiguous": ExecutionStatus.ABSTAINED,
            "needs_input": ExecutionStatus.ABSTAINED,
            "error": ExecutionStatus.ERROR,
            "invalid": ExecutionStatus.ERROR,
        }
        routes = {
            "ok": "graph",
            "unknown": "unknown",
            "ambiguous": "list",
            "needs_input": "help",
            "error": "math",
            "invalid": "unknown",
        }

        with tempfile.TemporaryDirectory() as tmp:
            grounded = FertigGrounded(
                tmp,
                assistant_factory=_FakeAssistant,
                chat_factory=_FakeChat,
            )
            for status, execution_status in expected.items():
                with self.subTest(status=status):
                    result = grounded.handle(
                        Request("grounded_chat", f"{status} {routes[status]}")
                    )
                    self.assertIs(result.status, execution_status)
                    self.assertEqual(
                        result.evidence["status"],
                        status if status != "invalid" else "error",
                    )
                    self.assertEqual(result.evidence["route"], routes[status])
                    self.assertEqual(
                        result.evidence["data"]["path"], "evidence.json"
                    )
                    self.assertEqual(result.ok, status == "ok")

    def test_injected_backends_are_forwarded_but_never_synthesised(self) -> None:
        desktop = object()
        recorder = object()
        with tempfile.TemporaryDirectory() as tmp:
            grounded = FertigGrounded(
                tmp,
                desktop=desktop,
                recorder=recorder,
                allow_mutations=True,
                assistant_factory=_FakeAssistant,
                chat_factory=_FakeChat,
            )

            done = grounded.handle(Request("grounded_chat", "do desktop"))
            taught = grounded.handle(Request("grounded_chat", "teach desktop"))
            composed = grounded.handle(Request("grounded_chat", "compose desktop"))

            self.assertTrue(done.ok)
            self.assertTrue(taught.ok)
            self.assertTrue(composed.ok)
            chat = grounded._chat
            self.assertEqual(
                chat.calls,  # type: ignore[union-attr]
                [
                    ("do desktop", desktop, recorder),
                    ("teach desktop", desktop, recorder),
                    ("compose desktop", desktop, recorder),
                ],
            )

    def test_optional_graph_path_is_forwarded_and_request_shape_is_guarded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            graph = Path(tmp) / "facts.causal"
            grounded = FertigGrounded(
                Path(tmp) / "state",
                graph_path=graph,
                assistant_factory=_FakeAssistant,
                chat_factory=_FakeChat,
            )

            rejected_capability = grounded.handle(Request("exact_math", "help"))
            rejected_payload = grounded.handle(Request("grounded_chat", "  "))
            result = grounded.handle(Request("grounded_chat", "ok graph"))

            self.assertIs(rejected_capability.status, ExecutionStatus.REJECTED)
            self.assertIs(rejected_payload.status, ExecutionStatus.REJECTED)
            self.assertTrue(result.ok)
            self.assertEqual(
                grounded._chat.options["graph_path"],  # type: ignore[union-attr]
                graph.resolve(),
            )


if __name__ == "__main__":
    unittest.main()
