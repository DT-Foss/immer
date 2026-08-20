from __future__ import annotations

import unittest

from immer.contracts import ExecutionStatus, Request, Result
from immer.registry import ComponentRegistry
from immer.runtime import ImmerRuntime


class EchoComponent:
    name = "echo"
    capabilities = frozenset({"echo"})

    def handle(self, request: Request) -> Result:
        return Result(ExecutionStatus.OK, self.name, output=request.payload)


class RuntimeTests(unittest.TestCase):
    def test_dispatches_to_registered_component(self) -> None:
        runtime = ImmerRuntime([EchoComponent()])
        result = runtime.dispatch(Request("echo", {"value": 7}))
        self.assertEqual(result.status, ExecutionStatus.OK)
        self.assertEqual(result.component, "echo")
        self.assertEqual(result.output, {"value": 7})

    def test_unknown_capability_is_unavailable(self) -> None:
        result = ImmerRuntime().dispatch(Request("missing", "x"))
        self.assertEqual(result.status, ExecutionStatus.UNAVAILABLE)
        self.assertIsNone(result.output)

    def test_duplicate_capability_requires_explicit_architecture_change(self) -> None:
        registry = ComponentRegistry()
        registry.register(EchoComponent())
        with self.assertRaises(ValueError):
            registry.register(EchoComponent())


if __name__ == "__main__":
    unittest.main()
