"""IMMER unified cognitive runtime."""

from .composition import CompositionRoot, compose_runtime
from .contracts import ExecutionStatus, Request, Result
from .runtime import ImmerRuntime

__version__ = "1.0.0"

__all__ = [
    "CompositionRoot",
    "ExecutionStatus",
    "ImmerRuntime",
    "Request",
    "Result",
    "__version__",
    "compose_runtime",
]
