"""O(depth) Dobrushin ledgers for arbitrarily long kernel composition."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Sequence

from numpy.typing import ArrayLike

from .identity import canonical_json_bytes, require_sha256
from .math_core import array_sha256, tv_contraction


CONTRACTION_LEDGER_SCHEMA = "immer-ooe-contraction-ledger/v1"
MAX_CONTRACTION_ENTRIES = 1_000_000
MAX_CONTRACTION_PAYLOAD_BYTES = 128 * 1024 * 1024


class ContractionLedgerIntegrityError(ValueError):
    pass


def _tau(value: object) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("Dobrushin contraction must lie in [0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class KernelContraction:
    kernel_sha256: str
    tau: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kernel_sha256",
            require_sha256(self.kernel_sha256, field="kernel_sha256"),
        )
        object.__setattr__(self, "tau", _tau(self.tau))

    def to_dict(self) -> dict[str, str]:
        return {
            "kernel_sha256": self.kernel_sha256,
            "tau_hex": self.tau.hex(),
        }


@dataclass(frozen=True, slots=True)
class ContractionLedger:
    entries: tuple[KernelContraction, ...]

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if len(entries) > MAX_CONTRACTION_ENTRIES:
            raise ValueError("contraction ledger exceeds its entry bound")
        if any(not isinstance(entry, KernelContraction) for entry in entries):
            raise TypeError("ledger entries must be KernelContraction values")
        object.__setattr__(self, "entries", entries)

    @classmethod
    def from_kernels(cls, kernels: Sequence[ArrayLike]) -> "ContractionLedger":
        if len(kernels) > MAX_CONTRACTION_ENTRIES:
            raise ValueError("kernel sequence exceeds its entry bound")
        return cls(
            tuple(
                KernelContraction(
                    kernel_sha256=array_sha256(kernel),
                    tau=tv_contraction(kernel),
                )
                for kernel in kernels
            )
        )

    @property
    def depth(self) -> int:
        return len(self.entries)

    @property
    def zero_contraction(self) -> bool:
        return any(entry.tau == 0.0 for entry in self.entries)

    @property
    def log_tau_upper_bound(self) -> float | None:
        if self.zero_contraction:
            return None
        return math.fsum(math.log(entry.tau) for entry in self.entries)

    @property
    def tau_upper_bound(self) -> float:
        if not self.entries:
            return 1.0
        logarithm = self.log_tau_upper_bound
        return 0.0 if logarithm is None else math.exp(logarithm)

    def compose(self, other: "ContractionLedger") -> "ContractionLedger":
        if not isinstance(other, ContractionLedger):
            raise TypeError("other must be a ContractionLedger")
        return ContractionLedger(self.entries + other.entries)

    def append(self, kernel: ArrayLike) -> "ContractionLedger":
        return self.compose(ContractionLedger.from_kernels((kernel,)))

    def verifies(
        self,
        composed_kernel: ArrayLike,
        *,
        tolerance: float = 1e-12,
    ) -> bool:
        threshold = float(tolerance)
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("tolerance must be finite and non-negative")
        actual = tv_contraction(composed_kernel)
        return actual <= self.tau_upper_bound + threshold

    def to_dict(self) -> dict[str, object]:
        return {
            "depth": self.depth,
            "entries": [entry.to_dict() for entry in self.entries],
            "log_tau_upper_bound_hex": (
                None
                if self.log_tau_upper_bound is None
                else self.log_tau_upper_bound.hex()
            ),
            "schema": CONTRACTION_LEDGER_SCHEMA,
            "tau_upper_bound_hex": self.tau_upper_bound.hex(),
            "zero_contraction": self.zero_contraction,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_dict())
        if len(data) > MAX_CONTRACTION_PAYLOAD_BYTES:
            raise ValueError("contraction ledger payload is oversized")
        return data

    @classmethod
    def from_bytes(cls, data: bytes) -> "ContractionLedger":
        if not isinstance(data, bytes) or len(data) > MAX_CONTRACTION_PAYLOAD_BYTES:
            raise ContractionLedgerIntegrityError(
                "contraction ledger must be bounded immutable bytes"
            )
        try:
            root = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractionLedgerIntegrityError(
                "contraction ledger is not JSON"
            ) from exc
        expected = {
            "depth",
            "entries",
            "log_tau_upper_bound_hex",
            "schema",
            "tau_upper_bound_hex",
            "zero_contraction",
        }
        if (
            not isinstance(root, dict)
            or set(root) != expected
            or root.get("schema") != CONTRACTION_LEDGER_SCHEMA
            or canonical_json_bytes(root) != data
        ):
            raise ContractionLedgerIntegrityError(
                "contraction ledger is not canonical"
            )
        raw_entries = root["entries"]
        if not isinstance(raw_entries, list) or len(raw_entries) > MAX_CONTRACTION_ENTRIES:
            raise ContractionLedgerIntegrityError("invalid contraction entries")
        try:
            entries = tuple(
                KernelContraction(
                    kernel_sha256=value["kernel_sha256"],
                    tau=float.fromhex(value["tau_hex"]),
                )
                for value in raw_entries
                if isinstance(value, dict)
                and set(value) == {"kernel_sha256", "tau_hex"}
            )
        except (TypeError, ValueError) as exc:
            raise ContractionLedgerIntegrityError(
                "invalid contraction entry"
            ) from exc
        if len(entries) != len(raw_entries):
            raise ContractionLedgerIntegrityError("invalid contraction entry shape")
        try:
            ledger = cls(entries)
        except (TypeError, ValueError) as exc:
            raise ContractionLedgerIntegrityError(
                "contraction ledger validation failed"
            ) from exc
        if ledger.to_dict() != root or root["depth"] != len(entries):
            raise ContractionLedgerIntegrityError(
                "contraction ledger derived fields were altered"
            )
        return ledger


__all__ = [
    "CONTRACTION_LEDGER_SCHEMA",
    "ContractionLedger",
    "ContractionLedgerIntegrityError",
    "KernelContraction",
    "MAX_CONTRACTION_ENTRIES",
    "MAX_CONTRACTION_PAYLOAD_BYTES",
]
