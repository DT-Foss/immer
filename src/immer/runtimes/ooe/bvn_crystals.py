"""Publish constructive Birkhoff decompositions as executable Crystal bases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .bvn_search import (
    BirkhoffDecomposition,
    BirkhoffReconstructionReceipt,
    BvNSearchIntegrityError,
    birkhoff_von_neumann,
)
from .compute_crystals import (
    ComputeBankPublication,
    ComputeCrystal,
    ComputeCrystalBank,
    ComputeCrystalError,
)
from .crystal import CrystalStoreError, StatePublication
from .identity import canonical_json_bytes, require_sha256
from .math_core import array_sha256


BIRKHOFF_CRYSTAL_RECEIPT_SCHEMA = "immer-ooe-birkhoff-crystal-receipt/v1"
BIRKHOFF_DECOMPOSITION_STATE_PREFIX = "ooe-birkhoff-decomposition/v1:"
BIRKHOFF_RECEIPT_STATE_PREFIX = "ooe-birkhoff-crystal-receipt/v1:"
MAX_BIRKHOFF_CRYSTAL_RECEIPT_BYTES = 16 * 1024 * 1024


class BirkhoffCrystalError(RuntimeError):
    """Base error for persisted Birkhoff Crystal bases."""


class BirkhoffCrystalIntegrityError(BirkhoffCrystalError):
    """A decomposition, Crystal, or publication receipt failed authentication."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _hashes(values: Sequence[str], *, field: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field} must be a sequence")
    result = tuple(require_sha256(value, field=field) for value in values)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{field} must be sorted and unique")
    return result


def _hex_weights(values: Sequence[float]) -> tuple[str, ...]:
    result = []
    for value in values:
        weight = float(value)
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError("Birkhoff weights must be positive and finite")
        result.append(weight.hex())
    return tuple(result)


def _inverse_permutation(permutation: Sequence[int]) -> tuple[int, ...]:
    images = tuple(int(value) for value in permutation)
    inverse = [-1] * len(images)
    for source, target in enumerate(images):
        if not 0 <= target < len(images) or inverse[target] != -1:
            raise ValueError("permutation images are not bijective")
        inverse[target] = source
    return tuple(inverse)


def _strict_json(data: bytes) -> object:
    if not isinstance(data, bytes):
        raise TypeError("Birkhoff Crystal receipt must be immutable bytes")
    if len(data) > MAX_BIRKHOFF_CRYSTAL_RECEIPT_BYTES:
        raise BirkhoffCrystalIntegrityError("Birkhoff Crystal receipt is too large")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise BirkhoffCrystalIntegrityError(
            "Birkhoff Crystal receipt is not strict JSON"
        ) from exc
    if canonical_json_bytes(value) != data:
        raise BirkhoffCrystalIntegrityError(
            "Birkhoff Crystal receipt is not canonical JSON"
        )
    return value


@dataclass(frozen=True, slots=True)
class BirkhoffCrystalReceipt:
    """Content addresses joining one DS kernel to its executable atom basis."""

    source_kernel_sha256: str
    decomposition_sha256: str
    reconstruction_receipt_sha256: str
    markov_crystal_sha256: str
    atom_crystal_sha256s: tuple[str, ...]
    weights: tuple[float, ...]
    tolerance: float
    verifier_sha256: str
    evidence_sha256s: tuple[str, ...]

    FORMAT = BIRKHOFF_CRYSTAL_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for field in (
            "source_kernel_sha256",
            "decomposition_sha256",
            "reconstruction_receipt_sha256",
            "markov_crystal_sha256",
            "verifier_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        atoms = tuple(
            require_sha256(value, field="atom_crystal_sha256s")
            for value in self.atom_crystal_sha256s
        )
        weights = tuple(float.fromhex(value) for value in _hex_weights(self.weights))
        tolerance = float(self.tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("Birkhoff Crystal tolerance must be positive and finite")
        if not atoms or len(atoms) != len(weights):
            raise ValueError("atom Crystal and weight inventories must align")
        if len(set(atoms)) != len(atoms):
            raise ValueError("atom Crystal addresses must be unique")
        if not np.isclose(sum(weights), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("Birkhoff Crystal weights must sum to one")
        evidence = _hashes(self.evidence_sha256s, field="evidence_sha256s")
        if not evidence:
            raise ValueError("Birkhoff Crystal receipt needs evidence")
        object.__setattr__(self, "atom_crystal_sha256s", atoms)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "evidence_sha256s", evidence)

    def to_dict(self) -> dict[str, object]:
        body = {
            "source_kernel_sha256": self.source_kernel_sha256,
            "decomposition_sha256": self.decomposition_sha256,
            "reconstruction_receipt_sha256": self.reconstruction_receipt_sha256,
            "markov_crystal_sha256": self.markov_crystal_sha256,
            "atom_crystal_sha256s": list(self.atom_crystal_sha256s),
            "weights_hex": list(_hex_weights(self.weights)),
            "tolerance_hex": self.tolerance.hex(),
            "verifier_sha256": self.verifier_sha256,
            "evidence_sha256s": list(self.evidence_sha256s),
        }
        return {
            "schema": self.FORMAT,
            "body": body,
            "body_sha256": _digest(body),
        }

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_dict())
        if len(data) > MAX_BIRKHOFF_CRYSTAL_RECEIPT_BYTES:
            raise ValueError("Birkhoff Crystal receipt exceeds its byte bound")
        return data

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "BirkhoffCrystalReceipt":
        value = _strict_json(data)
        if (
            not isinstance(value, Mapping)
            or set(value) != {"schema", "body", "body_sha256"}
            or value.get("schema") != cls.FORMAT
        ):
            raise BirkhoffCrystalIntegrityError("invalid Birkhoff Crystal envelope")
        body = value.get("body")
        expected = {
            "source_kernel_sha256",
            "decomposition_sha256",
            "reconstruction_receipt_sha256",
            "markov_crystal_sha256",
            "atom_crystal_sha256s",
            "weights_hex",
            "tolerance_hex",
            "verifier_sha256",
            "evidence_sha256s",
        }
        if not isinstance(body, Mapping) or set(body) != expected:
            raise BirkhoffCrystalIntegrityError("invalid Birkhoff Crystal body")
        atoms = body.get("atom_crystal_sha256s")
        raw_weights = body.get("weights_hex")
        evidence = body.get("evidence_sha256s")
        if not all(isinstance(item, list) for item in (atoms, raw_weights, evidence)):
            raise BirkhoffCrystalIntegrityError(
                "invalid Birkhoff Crystal inventories"
            )
        try:
            claimed = require_sha256(value.get("body_sha256"), field="body_sha256")
            if claimed != _digest(body):
                raise BirkhoffCrystalIntegrityError(
                    "Birkhoff Crystal body hash mismatch"
                )
            receipt = cls(
                source_kernel_sha256=cast(str, body.get("source_kernel_sha256")),
                decomposition_sha256=cast(str, body.get("decomposition_sha256")),
                reconstruction_receipt_sha256=cast(
                    str, body.get("reconstruction_receipt_sha256")
                ),
                markov_crystal_sha256=cast(
                    str, body.get("markov_crystal_sha256")
                ),
                atom_crystal_sha256s=tuple(cast(list[str], atoms)),
                weights=tuple(
                    float.fromhex(value)
                    for value in cast(list[str], raw_weights)
                ),
                tolerance=float.fromhex(cast(str, body.get("tolerance_hex"))),
                verifier_sha256=cast(str, body.get("verifier_sha256")),
                evidence_sha256s=tuple(cast(list[str], evidence)),
            )
        except BirkhoffCrystalIntegrityError:
            raise
        except (TypeError, ValueError) as exc:
            raise BirkhoffCrystalIntegrityError(
                "Birkhoff Crystal receipt validation failed"
            ) from exc
        if receipt.to_bytes() != data:
            raise BirkhoffCrystalIntegrityError(
                "Birkhoff Crystal receipt failed canonical reconstruction"
            )
        return receipt


@dataclass(frozen=True, slots=True)
class BirkhoffCrystalPublication:
    receipt: BirkhoffCrystalReceipt
    markov_publication: ComputeBankPublication
    atom_publications: tuple[ComputeBankPublication, ...]
    decomposition_state_publication: StatePublication
    receipt_state_publication: StatePublication

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, BirkhoffCrystalReceipt):
            raise TypeError("receipt must be a BirkhoffCrystalReceipt")
        if self.markov_publication.payload_sha256 != self.receipt.markov_crystal_sha256:
            raise ValueError("Markov publication differs from its receipt")
        if tuple(row.payload_sha256 for row in self.atom_publications) != (
            self.receipt.atom_crystal_sha256s
        ):
            raise ValueError("atom publications differ from their receipt")


class BirkhoffCrystalBank:
    """Persist a DS operator and only the permutation atoms it actually uses."""

    def __init__(self, bank: ComputeCrystalBank) -> None:
        if not isinstance(bank, ComputeCrystalBank):
            raise TypeError("bank must be a ComputeCrystalBank")
        self.bank = bank

    @staticmethod
    def decomposition_state_name(decomposition_sha256: str) -> str:
        return BIRKHOFF_DECOMPOSITION_STATE_PREFIX + require_sha256(
            decomposition_sha256, field="decomposition_sha256"
        )

    @staticmethod
    def receipt_state_name(receipt_sha256: str) -> str:
        return BIRKHOFF_RECEIPT_STATE_PREFIX + require_sha256(
            receipt_sha256, field="receipt_sha256"
        )

    def publish(
        self,
        kernel: ArrayLike,
        *,
        verifier_sha256: str,
        evidence_sha256s: Sequence[str],
        tolerance: float = 1e-12,
    ) -> BirkhoffCrystalPublication:
        verifier = require_sha256(verifier_sha256, field="verifier_sha256")
        evidence = _hashes(evidence_sha256s, field="evidence_sha256s")
        if not evidence:
            raise ValueError("Birkhoff publication requires evidence")
        try:
            decomposition = birkhoff_von_neumann(kernel, tolerance=tolerance)
            reconstruction = decomposition.verify_source(kernel, tolerance=tolerance)
        except (BvNSearchIntegrityError, ArithmeticError, TypeError, ValueError) as exc:
            raise BirkhoffCrystalIntegrityError(
                "kernel failed constructive Birkhoff verification"
            ) from exc

        decomposition_publication = self.bank.store.publish_state(
            self.decomposition_state_name(decomposition.sha256),
            decomposition.to_bytes(),
        )
        atom_crystals = []
        atom_publications = []
        for ordinal, (atom, weight) in enumerate(
            zip(decomposition.atoms, decomposition.weights, strict=True)
        ):
            indices = _inverse_permutation(atom.permutation)
            crystal = ComputeCrystal.permutation(
                indices,
                extensions={
                    "birkhoff_atom": {
                        "atom_ordinal": ordinal,
                        "decomposition_sha256": decomposition.sha256,
                        "row_image_permutation": list(atom.permutation),
                        "source_kernel_sha256": decomposition.source_kernel_sha256,
                        "weight_hex": weight.hex(),
                    }
                },
            )
            atom_crystals.append(crystal)
            atom_publications.append(self.bank.publish_crystal(crystal))

        source = np.asarray(kernel, dtype=np.float64)
        markov = ComputeCrystal.markov(
            source,
            extensions={
                "birkhoff_basis": {
                    "atom_crystal_sha256s": [
                        crystal.sha256 for crystal in atom_crystals
                    ],
                    "decomposition_sha256": decomposition.sha256,
                    "evidence_sha256s": list(evidence),
                    "reconstruction_receipt_sha256": reconstruction.sha256,
                    "source_kernel_sha256": decomposition.source_kernel_sha256,
                    "verifier_sha256": verifier,
                    "weights_hex": [weight.hex() for weight in decomposition.weights],
                }
            },
        )
        markov_publication = self.bank.publish_crystal(markov)
        receipt = BirkhoffCrystalReceipt(
            source_kernel_sha256=decomposition.source_kernel_sha256,
            decomposition_sha256=decomposition.sha256,
            reconstruction_receipt_sha256=reconstruction.sha256,
            markov_crystal_sha256=markov.sha256,
            atom_crystal_sha256s=tuple(
                crystal.sha256 for crystal in atom_crystals
            ),
            weights=decomposition.weights,
            tolerance=float(tolerance),
            verifier_sha256=verifier,
            evidence_sha256s=evidence,
        )
        receipt_publication = self.bank.store.publish_state(
            self.receipt_state_name(receipt.sha256),
            receipt.to_bytes(),
        )
        self.restore(receipt.sha256)
        return BirkhoffCrystalPublication(
            receipt=receipt,
            markov_publication=markov_publication,
            atom_publications=tuple(atom_publications),
            decomposition_state_publication=decomposition_publication,
            receipt_state_publication=receipt_publication,
        )

    def restore(
        self, receipt_sha256: str
    ) -> tuple[
        BirkhoffCrystalReceipt,
        BirkhoffDecomposition,
        BirkhoffReconstructionReceipt,
    ]:
        address = require_sha256(receipt_sha256, field="receipt_sha256")
        try:
            receipt_data = self.bank.store.restore_state(
                self.receipt_state_name(address)
            )
            receipt = BirkhoffCrystalReceipt.from_bytes(receipt_data)
            if receipt.sha256 != address:
                raise BirkhoffCrystalIntegrityError(
                    "Birkhoff receipt content address mismatch"
                )
            decomposition_data = self.bank.store.restore_state(
                self.decomposition_state_name(receipt.decomposition_sha256)
            )
            decomposition = BirkhoffDecomposition.from_bytes(decomposition_data)
            if decomposition.sha256 != receipt.decomposition_sha256:
                raise BirkhoffCrystalIntegrityError(
                    "Birkhoff decomposition content address mismatch"
                )
            markov = self.bank.restore_crystal(receipt.markov_crystal_sha256)
            dimension = decomposition.size
            kernel = cast(
                NDArray[np.float64],
                markov.apply(np.eye(dimension, dtype=np.float64)),
            )
            if array_sha256(kernel) != receipt.source_kernel_sha256:
                raise BirkhoffCrystalIntegrityError(
                    "restored Markov Crystal differs from Birkhoff source"
                )
            reconstruction = decomposition.verify_source(
                kernel, tolerance=receipt.tolerance
            )
            if reconstruction.sha256 != receipt.reconstruction_receipt_sha256:
                raise BirkhoffCrystalIntegrityError(
                    "reconstruction receipt identity changed"
                )
            for atom, crystal_sha in zip(
                decomposition.atoms,
                receipt.atom_crystal_sha256s,
                strict=True,
            ):
                crystal = self.bank.restore_crystal(crystal_sha)
                expected = _inverse_permutation(atom.permutation)
                probe = np.arange(dimension, dtype=np.float64)[None, :]
                if not np.array_equal(crystal.apply(probe), probe[:, expected]):
                    raise BirkhoffCrystalIntegrityError(
                        "permutation atom Crystal changed semantics"
                    )
        except BirkhoffCrystalIntegrityError:
            raise
        except (
            BvNSearchIntegrityError,
            ComputeCrystalError,
            CrystalStoreError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise BirkhoffCrystalIntegrityError(
                "Birkhoff Crystal basis restoration failed"
            ) from exc
        return receipt, decomposition, reconstruction

    def apply_atoms(self, receipt_sha256: str, value: object) -> NDArray[np.float64]:
        """Apply the weighted atom basis and reproduce the stored Markov operator."""

        receipt, decomposition, _verification = self.restore(receipt_sha256)
        markov = self.bank.restore_crystal(receipt.markov_crystal_sha256)
        source = markov.input_abi.validate(value, field="Birkhoff basis input")
        outputs = [
            self.bank.restore_crystal(crystal_sha).apply(source)
            for crystal_sha in receipt.atom_crystal_sha256s
        ]
        result = np.zeros_like(outputs[0], dtype=np.float64)
        for weight, output in zip(receipt.weights, outputs, strict=True):
            result += weight * output
        expected = markov.apply(source)
        if not np.allclose(result, expected, rtol=0.0, atol=1e-12):
            raise BirkhoffCrystalIntegrityError(
                "weighted atom basis no longer reconstructs the Markov Crystal"
            )
        if decomposition.component_count != len(outputs):
            raise AssertionError("restored decomposition lost an atom")
        return np.ascontiguousarray(result)


__all__ = [
    "BIRKHOFF_CRYSTAL_RECEIPT_SCHEMA",
    "BIRKHOFF_DECOMPOSITION_STATE_PREFIX",
    "BIRKHOFF_RECEIPT_STATE_PREFIX",
    "BirkhoffCrystalBank",
    "BirkhoffCrystalError",
    "BirkhoffCrystalIntegrityError",
    "BirkhoffCrystalPublication",
    "BirkhoffCrystalReceipt",
]
