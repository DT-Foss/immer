from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
from typing import Iterator, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .identity import (
    OoeSiteIdentity,
    canonical_json_bytes,
    require_sha256,
)
from .math_core import normalize_rows


MAX_KERNEL_DIMENSION = 4096
MAX_KERNEL_DECODED_BYTES = 32 * 1024 * 1024
MAX_CRYSTAL_PAYLOAD_BYTES = 64 * 1024 * 1024
MAX_MANIFEST_BYTES = 64 * 1024 * 1024


class CrystalStoreError(RuntimeError):
    pass


class CrystalTamperError(CrystalStoreError):
    pass


class CrystalCollisionError(CrystalStoreError):
    pass


class ManifestConflictError(CrystalStoreError):
    pass


class CrystalIdentityError(CrystalStoreError):
    pass


def _strict_json(data: bytes) -> object:
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
        decoded = data.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CrystalTamperError("crystal contains invalid canonical JSON") from exc
    if canonical_json_bytes(value) != data:
        raise CrystalTamperError("crystal JSON is not in canonical form")
    return value


def _hash_items(value: Mapping[str, str], *, field: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{field} must contain at least one named digest")
    result: list[tuple[str, str]] = []
    for name, digest in value.items():
        if not isinstance(name, str) or not name or "\x00" in name:
            raise ValueError(f"{field} names must be non-empty strings")
        result.append((name, require_sha256(digest, field=f"{field}[{name!r}]")))
    result.sort()
    if len({name for name, _ in result}) != len(result):
        raise ValueError(f"{field} names must be unique")
    return tuple(result)


def _validate_hash_tuple(
    value: tuple[tuple[str, str], ...], *, field: str
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{field} must be an immutable tuple")
    mapping: dict[str, str] = {}
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"{field} entries must be (name, digest) tuples")
        name, digest = item
        if name in mapping:
            raise ValueError(f"{field} names must be unique")
        mapping[name] = digest
    return _hash_items(mapping, field=field)


def _kernel_byte_count(rows: object, columns: object) -> tuple[int, int, int]:
    for value, field in ((rows, "kernel_rows"), (columns, "kernel_columns")):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= MAX_KERNEL_DIMENSION
        ):
            raise ValueError(f"{field} must lie in [1, {MAX_KERNEL_DIMENSION}]")
    row_count = int(rows)
    column_count = int(columns)
    decoded_bytes = row_count * column_count * 2
    if decoded_bytes > MAX_KERNEL_DECODED_BYTES:
        raise ValueError("decoded quantized kernel exceeds MAX_KERNEL_DECODED_BYTES")
    return row_count, column_count, decoded_bytes


def _quantize_rows(kernel: NDArray[np.floating], levels: int) -> bytes:
    value = normalize_rows(np.asarray(kernel, dtype=np.float64))
    if not np.all(np.isfinite(value)):
        raise ValueError("kernel must contain only finite values")
    scaled = value * levels
    quantized = np.floor(scaled).astype(np.int64)
    fractions = scaled - quantized
    for row_index in range(quantized.shape[0]):
        remainder = levels - int(quantized[row_index].sum())
        if remainder > 0:
            order = np.argsort(-fractions[row_index], kind="stable")
            quantized[row_index, order[:remainder]] += 1
        elif remainder < 0:
            candidates = np.flatnonzero(quantized[row_index] > 0)
            order = candidates[
                np.argsort(fractions[row_index, candidates], kind="stable")
            ]
            quantized[row_index, order[:-remainder]] -= 1
    if np.any(quantized < 0) or np.any(quantized > levels):
        raise AssertionError("quantizer produced an out-of-range value")
    if not np.all(quantized.sum(axis=1) == levels):
        raise AssertionError("quantizer failed to preserve row mass")
    return quantized.astype("<u2", copy=False).tobytes(order="C")


@dataclass(frozen=True, slots=True)
class CrystalPayload:
    """An immutable, fully-bound executable OoE kernel."""

    name: str
    identity: OoeSiteIdentity
    kernel_rows: int
    kernel_columns: int
    quantization_levels: int
    quantized_kernel: bytes
    coverage_sha256: str
    calibration_sha256: str
    verifier_hashes: tuple[tuple[str, str], ...]
    evidence_hashes: tuple[tuple[str, str], ...]
    consensus_receipt_json: bytes

    FORMAT = "immer-ooe-crystal/v1"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or "\x00" in self.name:
            raise ValueError("crystal name must be a non-empty string")
        if len(self.name.encode("utf-8")) > 1024:
            raise ValueError("crystal name is too long")
        if not isinstance(self.identity, OoeSiteIdentity):
            raise TypeError("identity must be an OoeSiteIdentity")
        rows, columns, expected_bytes = _kernel_byte_count(
            self.kernel_rows, self.kernel_columns
        )
        object.__setattr__(self, "kernel_rows", rows)
        object.__setattr__(self, "kernel_columns", columns)
        if (
            isinstance(self.quantization_levels, bool)
            or not isinstance(self.quantization_levels, int)
            or not 1 < self.quantization_levels <= 65535
        ):
            raise ValueError("quantization_levels must lie in [2, 65535]")
        if not isinstance(self.quantized_kernel, bytes):
            raise TypeError("quantized_kernel must be immutable bytes")
        if len(self.quantized_kernel) != expected_bytes:
            raise ValueError("quantized kernel byte length does not match its shape")
        quantized = np.frombuffer(self.quantized_kernel, dtype="<u2").reshape(
            self.kernel_rows, self.kernel_columns
        )
        if np.any(quantized > self.quantization_levels):
            raise ValueError("quantized kernel value exceeds its quantization range")
        if not np.all(
            quantized.astype(np.uint64).sum(axis=1) == self.quantization_levels
        ):
            raise ValueError(
                "each quantized kernel row must preserve exact probability mass"
            )

        object.__setattr__(
            self,
            "coverage_sha256",
            require_sha256(self.coverage_sha256, field="coverage_sha256"),
        )
        object.__setattr__(
            self,
            "calibration_sha256",
            require_sha256(self.calibration_sha256, field="calibration_sha256"),
        )
        object.__setattr__(
            self,
            "verifier_hashes",
            _validate_hash_tuple(self.verifier_hashes, field="verifier_hashes"),
        )
        object.__setattr__(
            self,
            "evidence_hashes",
            _validate_hash_tuple(self.evidence_hashes, field="evidence_hashes"),
        )
        if not isinstance(self.consensus_receipt_json, bytes):
            raise TypeError("consensus_receipt_json must be immutable bytes")
        receipt = _strict_json(self.consensus_receipt_json)
        if not isinstance(receipt, dict) or not receipt:
            raise ValueError("consensus receipt must be a non-empty JSON object")

    @classmethod
    def from_kernel(
        cls,
        *,
        name: str,
        identity: OoeSiteIdentity,
        kernel: NDArray[np.floating],
        coverage_sha256: str,
        calibration_sha256: str,
        verifier_hashes: Mapping[str, str],
        evidence_hashes: Mapping[str, str],
        consensus_receipt: Mapping[str, object],
        quantization_levels: int = 65535,
    ) -> "CrystalPayload":
        value = np.asarray(kernel, dtype=np.float64)
        if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1:
            raise ValueError("kernel must be a non-empty rank-2 matrix")
        _kernel_byte_count(int(value.shape[0]), int(value.shape[1]))
        if not np.all(np.isfinite(value)) or np.any(value < 0.0):
            raise ValueError("kernel must be finite and non-negative")
        if np.any(value.sum(axis=1) <= 0.0):
            raise ValueError("every kernel row must have positive mass")
        if (
            isinstance(quantization_levels, bool)
            or not isinstance(quantization_levels, int)
            or not 1 < quantization_levels <= 65535
        ):
            raise ValueError("quantization_levels must lie in [2, 65535]")
        if not isinstance(consensus_receipt, Mapping) or not consensus_receipt:
            raise ValueError("consensus_receipt must be a non-empty mapping")
        return cls(
            name=name,
            identity=identity,
            kernel_rows=int(value.shape[0]),
            kernel_columns=int(value.shape[1]),
            quantization_levels=quantization_levels,
            quantized_kernel=_quantize_rows(value, quantization_levels),
            coverage_sha256=require_sha256(coverage_sha256, field="coverage_sha256"),
            calibration_sha256=require_sha256(
                calibration_sha256, field="calibration_sha256"
            ),
            verifier_hashes=_hash_items(verifier_hashes, field="verifier_hashes"),
            evidence_hashes=_hash_items(evidence_hashes, field="evidence_hashes"),
            consensus_receipt_json=canonical_json_bytes(dict(consensus_receipt)),
        )

    @property
    def consensus_receipt(self) -> dict[str, object]:
        value = json.loads(self.consensus_receipt_json)
        if not isinstance(value, dict):
            raise AssertionError("validated consensus receipt changed type")
        return value

    @property
    def consensus_receipt_sha256(self) -> str:
        return hashlib.sha256(self.consensus_receipt_json).hexdigest()

    @property
    def verifier_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(dict(self.verifier_hashes))
        ).hexdigest()

    @property
    def evidence_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(dict(self.evidence_hashes))
        ).hexdigest()

    def restore_kernel(self) -> NDArray[np.float64]:
        quantized = np.frombuffer(self.quantized_kernel, dtype="<u2").reshape(
            self.kernel_rows, self.kernel_columns
        )
        return quantized.astype(np.float64) / self.quantization_levels

    def to_dict(self) -> dict[str, object]:
        return {
            "calibration_sha256": self.calibration_sha256,
            "consensus_receipt": self.consensus_receipt,
            "coverage_sha256": self.coverage_sha256,
            "evidence_hashes": dict(self.evidence_hashes),
            "format": self.FORMAT,
            "identity": self.identity.to_dict(),
            "kernel": {
                "columns": self.kernel_columns,
                "data_base64": base64.b64encode(self.quantized_kernel).decode("ascii"),
                "levels": self.quantization_levels,
                "rows": self.kernel_rows,
                "storage": "uint16-le",
            },
            "name": self.name,
            "verifier_hashes": dict(self.verifier_hashes),
        }

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_dict())
        if len(data) > MAX_CRYSTAL_PAYLOAD_BYTES:
            raise ValueError("crystal payload exceeds MAX_CRYSTAL_PAYLOAD_BYTES")
        return data

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "CrystalPayload":
        if not isinstance(data, bytes):
            raise TypeError("crystal payload must be bytes")
        if len(data) > MAX_CRYSTAL_PAYLOAD_BYTES:
            raise CrystalTamperError(
                "crystal payload exceeds MAX_CRYSTAL_PAYLOAD_BYTES"
            )
        value = _strict_json(data)
        if not isinstance(value, dict) or value.get("format") != cls.FORMAT:
            raise CrystalTamperError("unsupported crystal payload format")
        expected = {
            "calibration_sha256",
            "consensus_receipt",
            "coverage_sha256",
            "evidence_hashes",
            "format",
            "identity",
            "kernel",
            "name",
            "verifier_hashes",
        }
        if set(value) != expected:
            raise CrystalTamperError("crystal payload has unknown or missing fields")
        kernel = value["kernel"]
        if not isinstance(kernel, dict) or set(kernel) != {
            "columns",
            "data_base64",
            "levels",
            "rows",
            "storage",
        }:
            raise CrystalTamperError("invalid crystal kernel descriptor")
        if kernel.get("storage") != "uint16-le":
            raise CrystalTamperError("unsupported crystal kernel storage")
        try:
            rows, columns, decoded_bytes = _kernel_byte_count(
                kernel["rows"], kernel["columns"]
            )
        except ValueError as exc:
            raise CrystalTamperError("invalid crystal kernel dimensions") from exc
        encoded = kernel["data_base64"]
        expected_encoded_bytes = 4 * ((decoded_bytes + 2) // 3)
        if (
            not isinstance(encoded, str)
            or len(encoded) != expected_encoded_bytes
            or not encoded.isascii()
        ):
            raise CrystalTamperError("invalid crystal kernel encoded byte length")
        try:
            quantized = base64.b64decode(encoded, validate=True)
        except (TypeError, ValueError) as exc:
            raise CrystalTamperError("invalid crystal kernel encoding") from exc
        if len(quantized) != decoded_bytes:
            raise CrystalTamperError("invalid crystal kernel decoded byte length")
        try:
            payload = cls(
                name=value["name"],
                identity=OoeSiteIdentity.from_dict(value["identity"]),
                kernel_rows=rows,
                kernel_columns=columns,
                quantization_levels=kernel["levels"],
                quantized_kernel=quantized,
                coverage_sha256=value["coverage_sha256"],
                calibration_sha256=value["calibration_sha256"],
                verifier_hashes=_hash_items(
                    value["verifier_hashes"], field="verifier_hashes"
                ),
                evidence_hashes=_hash_items(
                    value["evidence_hashes"], field="evidence_hashes"
                ),
                consensus_receipt_json=canonical_json_bytes(value["consensus_receipt"]),
            )
        except (TypeError, ValueError) as exc:
            raise CrystalTamperError("crystal payload validation failed") from exc
        if payload.to_bytes() != data:
            raise CrystalTamperError("crystal payload failed canonical roundtrip")
        return payload


@dataclass(frozen=True, slots=True)
class CrystalManifestEntry:
    name: str
    payload_sha256: str
    identity_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("manifest entry name must be non-empty")
        object.__setattr__(
            self,
            "payload_sha256",
            require_sha256(self.payload_sha256, field="payload_sha256"),
        )
        object.__setattr__(
            self,
            "identity_sha256",
            require_sha256(self.identity_sha256, field="identity_sha256"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "identity_sha256": self.identity_sha256,
            "payload_sha256": self.payload_sha256,
        }


@dataclass(frozen=True, slots=True)
class CrystalManifest:
    generation: int
    entries: tuple[CrystalManifestEntry, ...]
    objects: tuple[str, ...]
    previous_manifest_sha256: str | None

    FORMAT = "immer-ooe-crystal-manifest/v1"

    def __post_init__(self) -> None:
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
        ):
            raise ValueError("manifest generation must be a non-negative integer")
        names = [entry.name for entry in self.entries]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("manifest entries must be sorted and unique")
        normalized_objects = tuple(
            sorted(
                {
                    require_sha256(digest, field="manifest object digest")
                    for digest in self.objects
                }
            )
        )
        if normalized_objects != self.objects:
            raise ValueError("manifest objects must be sorted and unique")
        if any(entry.payload_sha256 not in self.objects for entry in self.entries):
            raise ValueError("active manifest entry missing from object inventory")
        if self.previous_manifest_sha256 is not None:
            object.__setattr__(
                self,
                "previous_manifest_sha256",
                require_sha256(
                    self.previous_manifest_sha256,
                    field="previous_manifest_sha256",
                ),
            )

    @classmethod
    def empty(cls) -> "CrystalManifest":
        return cls(0, (), (), None)

    def resolve(self, name: str) -> CrystalManifestEntry:
        for entry in self.entries:
            if entry.name == name:
                return entry
        raise KeyError(f"unknown crystal: {name}")

    def to_dict(self) -> dict[str, object]:
        return {
            "entries": {entry.name: entry.to_dict() for entry in self.entries},
            "format": self.FORMAT,
            "generation": self.generation,
            "objects": list(self.objects),
            "previous_manifest_sha256": self.previous_manifest_sha256,
        }

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_dict())
        if len(data) > MAX_MANIFEST_BYTES:
            raise ValueError("crystal manifest exceeds MAX_MANIFEST_BYTES")
        return data

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "CrystalManifest":
        if not isinstance(data, bytes):
            raise TypeError("crystal manifest must be bytes")
        if len(data) > MAX_MANIFEST_BYTES:
            raise CrystalTamperError("crystal manifest exceeds MAX_MANIFEST_BYTES")
        value = _strict_json(data)
        if not isinstance(value, dict) or value.get("format") != cls.FORMAT:
            raise CrystalTamperError("unsupported crystal manifest")
        if set(value) != {
            "entries",
            "format",
            "generation",
            "objects",
            "previous_manifest_sha256",
        }:
            raise CrystalTamperError("manifest has unknown or missing fields")
        raw_entries = value["entries"]
        if not isinstance(raw_entries, dict):
            raise CrystalTamperError("manifest entries must be an object")
        entries: list[CrystalManifestEntry] = []
        try:
            for name, raw in raw_entries.items():
                if not isinstance(raw, dict) or set(raw) != {
                    "identity_sha256",
                    "payload_sha256",
                }:
                    raise ValueError("invalid manifest entry")
                entries.append(
                    CrystalManifestEntry(
                        name=name,
                        payload_sha256=raw["payload_sha256"],
                        identity_sha256=raw["identity_sha256"],
                    )
                )
            objects = value["objects"]
            if not isinstance(objects, list):
                raise ValueError("manifest objects must be a list")
            manifest = cls(
                generation=value["generation"],
                entries=tuple(sorted(entries, key=lambda entry: entry.name)),
                objects=tuple(objects),
                previous_manifest_sha256=value["previous_manifest_sha256"],
            )
        except (TypeError, ValueError) as exc:
            raise CrystalTamperError("manifest validation failed") from exc
        if manifest.to_bytes() != data:
            raise CrystalTamperError("manifest failed canonical roundtrip")
        return manifest


@dataclass(frozen=True, slots=True)
class CrystalPublication:
    payload_sha256: str
    identity_sha256: str
    generation: int
    manifest_sha256: str
    object_created: bool
    manifest_changed: bool


@dataclass(frozen=True, slots=True)
class CrystalStoreAudit:
    generation: int
    manifest_sha256: str
    valid_objects: tuple[str, ...]
    orphan_objects: tuple[str, ...]
    missing_objects: tuple[str, ...]
    tampered_objects: tuple[str, ...]
    staged_files: tuple[str, ...]
    unexpected_files: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return not (
            self.orphan_objects
            or self.missing_objects
            or self.tampered_objects
            or self.staged_files
            or self.unexpected_files
        )


@dataclass(frozen=True, slots=True)
class StatePublication:
    name: str
    payload_sha256: str
    generation: int
    changed: bool


class CrystalStore:
    """Crash-safe content-addressed Crystal storage with manifest CAS."""

    _MANIFEST = "manifest.json"
    _LOCK = "LOCK"
    _FORK_MARKER = "IMMER-EXACT-FORK-INTENT"

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        max_state_bytes: int = 64 * 1024 * 1024,
        _allow_incomplete_fork: bool = False,
    ) -> None:
        if (
            isinstance(max_state_bytes, bool)
            or not isinstance(max_state_bytes, int)
            or max_state_bytes < 1
        ):
            raise ValueError("max_state_bytes must be a positive integer")
        self.root = Path(root)
        self.max_state_bytes = max_state_bytes
        self._ensure_managed_directory(self.root)
        if not isinstance(_allow_incomplete_fork, bool):
            raise TypeError("_allow_incomplete_fork must be bool")
        marker = self.root / self._FORK_MARKER
        marker_temporaries = tuple(
            path
            for path in self.root.iterdir()
            if path.name.startswith(f".{self._FORK_MARKER}.")
            and path.name.endswith(".tmp")
        )
        parent_intents = tuple(
            path
            for path in self.root.parent.iterdir()
            if path.name.startswith(f".{self.root.name}.")
            and path.name.endswith(".fork.INTENT")
        )
        if not _allow_incomplete_fork and (
            marker.exists()
            or marker.is_symlink()
            or marker_temporaries
            or (parent_intents and not (self.root / self._MANIFEST).is_file())
        ):
            raise CrystalStoreError("CrystalStore exact fork is still in progress")
        self._ensure_managed_directory(self.root / "objects")
        self._ensure_managed_directory(self.root / "staging")
        self._ensure_managed_directory(self.root / "state")

    @staticmethod
    def _ensure_managed_directory(path: Path) -> None:
        try:
            path.mkdir(parents=True, mode=0o700)
        except FileExistsError:
            pass
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise CrystalStoreError(f"managed path is not a real directory: {path}")

    @staticmethod
    def _directory_flags() -> int:
        return (
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        )

    @contextmanager
    def _directories(self) -> Iterator[tuple[int, int, int]]:
        root_fd = os.open(self.root, self._directory_flags())
        try:
            objects_fd = os.open("objects", self._directory_flags(), dir_fd=root_fd)
            try:
                staging_fd = os.open("staging", self._directory_flags(), dir_fd=root_fd)
                try:
                    yield root_fd, objects_fd, staging_fd
                finally:
                    os.close(staging_fd)
            finally:
                os.close(objects_fd)
        finally:
            os.close(root_fd)

    @contextmanager
    def _locked(self, root_fd: int) -> Iterator[None]:
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        lock_fd = os.open(self._LOCK, flags, 0o600, dir_fd=root_fd)
        try:
            metadata = os.fstat(lock_fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise CrystalStoreError("CrystalStore lock is not a regular file")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    @staticmethod
    def _stable_read(dir_fd: int, name: str, *, max_bytes: int | None = None) -> bytes:
        if max_bytes is not None and (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes < 1
        ):
            raise ValueError("max_bytes must be a positive integer")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(name, flags, dir_fd=dir_fd)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode):
                raise CrystalTamperError(f"{name} is not a regular file")
            if max_bytes is not None and before.st_size > max_bytes:
                raise CrystalTamperError(f"{name} exceeds its hard file-size limit")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if max_bytes is not None and total > max_bytes:
                    raise CrystalTamperError(
                        f"{name} grew beyond its hard file-size limit"
                    )
                chunks.append(chunk)
            after = os.fstat(fd)
            stable = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) == (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            data = b"".join(chunks)
            if not stable or len(data) != before.st_size:
                raise CrystalTamperError(f"{name} changed while it was being read")
            return data
        finally:
            os.close(fd)

    @staticmethod
    def _write_all(fd: int, data: bytes) -> None:
        view = memoryview(data)
        written = 0
        while written < len(view):
            count = os.write(fd, view[written:])
            if count <= 0:
                raise OSError("short write while publishing crystal")
            written += count

    @classmethod
    def _read_manifest_fd(cls, root_fd: int) -> CrystalManifest:
        try:
            data = cls._stable_read(
                root_fd, cls._MANIFEST, max_bytes=MAX_MANIFEST_BYTES
            )
        except FileNotFoundError:
            return CrystalManifest.empty()
        return CrystalManifest.from_bytes(data)

    @staticmethod
    def _temporary_name(prefix: str) -> str:
        return f".{prefix}.{os.getpid()}.{secrets.token_hex(16)}.tmp"

    def _publish_object(
        self, objects_fd: int, staging_fd: int, payload: bytes, digest: str
    ) -> bool:
        final_name = f"{digest}.crystal"
        try:
            existing = self._stable_read(
                objects_fd,
                final_name,
                max_bytes=MAX_CRYSTAL_PAYLOAD_BYTES,
            )
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if existing != payload or hashlib.sha256(existing).hexdigest() != digest:
                raise CrystalCollisionError(
                    f"content-address collision or tamper at {digest}"
                )
            return False

        temporary = self._temporary_name("crystal")
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        temporary_fd = os.open(temporary, flags, 0o600, dir_fd=staging_fd)
        try:
            self._write_all(temporary_fd, payload)
            os.fsync(temporary_fd)
            os.fchmod(temporary_fd, 0o444)
            os.fsync(temporary_fd)
        finally:
            os.close(temporary_fd)

        created = False
        try:
            try:
                os.link(
                    temporary,
                    final_name,
                    src_dir_fd=staging_fd,
                    dst_dir_fd=objects_fd,
                    follow_symlinks=False,
                )
                created = True
                os.fsync(objects_fd)
            except FileExistsError:
                existing = self._stable_read(
                    objects_fd,
                    final_name,
                    max_bytes=MAX_CRYSTAL_PAYLOAD_BYTES,
                )
                if (
                    existing != payload
                    or hashlib.sha256(existing).hexdigest() != digest
                ):
                    raise CrystalCollisionError(
                        f"content-address collision or tamper at {digest}"
                    )
        finally:
            try:
                os.unlink(temporary, dir_fd=staging_fd)
                os.fsync(staging_fd)
            except FileNotFoundError:
                pass
        return created

    def _replace_manifest(
        self, root_fd: int, staging_fd: int, manifest: CrystalManifest
    ) -> None:
        data = manifest.to_bytes()
        temporary = self._temporary_name("manifest")
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(temporary, flags, 0o600, dir_fd=staging_fd)
        try:
            self._write_all(fd, data)
            os.fsync(fd)
            os.fchmod(fd, 0o444)
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.replace(
                temporary,
                self._MANIFEST,
                src_dir_fd=staging_fd,
                dst_dir_fd=root_fd,
            )
            os.fsync(root_fd)
        finally:
            try:
                os.unlink(temporary, dir_fd=staging_fd)
            except FileNotFoundError:
                pass

    def manifest(self) -> CrystalManifest:
        with self._directories() as (root_fd, _, _):
            return self._read_manifest_fd(root_fd)

    def fork_exact(
        self,
        target_root: str | os.PathLike[str],
        *,
        state_names: Sequence[str],
        expected_manifest_sha256: str | None = None,
        expected_state_sha256s: Mapping[str, str] | None = None,
    ) -> "CrystalStore":
        """Create or resume one exact, plan-marked CrystalStore fork.

        The target directory itself is the no-replace reservation.  Until all
        declared historical objects, exact manifest bytes, and selected state
        envelopes pass audit, it carries a sealed fork-plan marker.  A retry
        repairs only plan-named regular files inside that marked directory; it
        never recursively deletes a predictable sibling path and never replaces
        a concurrently created target directory.
        """

        if isinstance(state_names, (str, bytes, bytearray)):
            raise TypeError("state_names must be a sequence of state names")
        try:
            names = tuple(state_names)
        except TypeError as exc:
            raise TypeError("state_names must be a sequence of state names") from exc
        if not names or len(set(names)) != len(names):
            raise ValueError("state_names must be a non-empty unique sequence")
        state_filenames = {name: self._state_filename(name) for name in names}
        expected_manifest = (
            None
            if expected_manifest_sha256 is None
            else require_sha256(
                expected_manifest_sha256,
                field="expected_manifest_sha256",
            )
        )
        if expected_state_sha256s is None:
            expected_states: dict[str, str] = {}
        else:
            if not isinstance(expected_state_sha256s, Mapping) or set(
                expected_state_sha256s
            ) != set(names):
                raise ValueError(
                    "expected_state_sha256s must cover every selected state"
                )
            expected_states = {
                name: require_sha256(
                    expected_state_sha256s[name],
                    field=f"expected_state_sha256s[{name!r}]",
                )
                for name in names
            }

        target = Path(target_root).expanduser().absolute()
        parent = target.parent
        try:
            parent_metadata = parent.lstat()
        except OSError as exc:
            raise CrystalStoreError(f"fork parent does not exist: {parent}") from exc
        if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(
            parent_metadata.st_mode
        ):
            raise CrystalStoreError("fork parent must be a real directory")
        source_real = self.root.resolve(strict=True)
        target_real = parent.resolve(strict=True) / target.name
        if (
            target_real == source_real
            or source_real in target_real.parents
            or target_real in source_real.parents
        ):
            raise CrystalStoreError("fork source and target roots overlap")

        token = hashlib.sha256(str(target_real).encode("utf-8")).hexdigest()[:20]
        parent_lock_name = f".{target.name}.{token}.fork.LOCK"
        parent_intent_name = f".{target.name}.{token}.fork.INTENT"
        marker_name = self._FORK_MARKER
        parent_fd = os.open(parent, self._directory_flags())
        lock_flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        parent_lock_fd = os.open(
            parent_lock_name,
            lock_flags,
            0o600,
            dir_fd=parent_fd,
        )

        def install_exact(dir_fd: int, name: str, data: bytes) -> None:
            try:
                current = self._stable_read(
                    dir_fd,
                    name,
                    max_bytes=max(len(data), 1),
                )
            except FileNotFoundError:
                current = None
            if current == data:
                return
            if current is not None:
                metadata = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
                if not stat.S_ISREG(metadata.st_mode):
                    raise CrystalStoreError(
                        "fork plan file is not a replaceable regular file"
                    )
                os.unlink(name, dir_fd=dir_fd)
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(
                os, "O_NOFOLLOW", 0
            )
            descriptor = os.open(name, flags, 0o600, dir_fd=dir_fd)
            try:
                self._write_all(descriptor, data)
                os.fsync(descriptor)
                os.fchmod(descriptor, 0o444)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

        def publish_immutable(dir_fd: int, name: str, data: bytes) -> None:
            try:
                current = self._stable_read(
                    dir_fd,
                    name,
                    max_bytes=max(len(data), 1),
                )
            except FileNotFoundError:
                current = None
            if current is not None:
                if current != data:
                    raise CrystalStoreError(
                        "immutable fork intent belongs to another plan"
                    )
                return
            temporary = f".{name}.{secrets.token_hex(12)}.tmp"
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(
                os, "O_NOFOLLOW", 0
            )
            descriptor = os.open(temporary, flags, 0o600, dir_fd=dir_fd)
            try:
                self._write_all(descriptor, data)
                os.fsync(descriptor)
                os.fchmod(descriptor, 0o444)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            try:
                try:
                    os.link(
                        temporary,
                        name,
                        src_dir_fd=dir_fd,
                        dst_dir_fd=dir_fd,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    if self._stable_read(
                        dir_fd,
                        name,
                        max_bytes=max(len(data), 1),
                    ) != data:
                        raise CrystalStoreError(
                            "concurrent fork intent belongs to another plan"
                        )
                os.fsync(dir_fd)
            finally:
                try:
                    os.unlink(temporary, dir_fd=dir_fd)
                except FileNotFoundError:
                    pass

        try:
            metadata = os.fstat(parent_lock_fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise CrystalStoreError("fork parent lock is not a regular file")
            fcntl.flock(parent_lock_fd, fcntl.LOCK_EX)
            with self._directories() as (
                source_root_fd,
                source_objects_fd,
                source_staging_fd,
            ):
                source_state_fd = os.open(
                    "state", self._directory_flags(), dir_fd=source_root_fd
                )
                try:
                    with self._locked(source_root_fd):
                        manifest = self._read_manifest_fd(source_root_fd)
                        if (
                            expected_manifest is not None
                            and manifest.sha256 != expected_manifest
                        ):
                            raise ManifestConflictError(
                                "source manifest differs from exact fork pin"
                            )
                        if os.listdir(source_staging_fd):
                            raise CrystalStoreError(
                                "source staging must be clean before exact fork"
                            )
                        expected_object_files = {
                            f"{digest}.crystal" for digest in manifest.objects
                        }
                        if set(os.listdir(source_objects_fd)) != expected_object_files:
                            raise CrystalStoreError(
                                "source object inventory differs from its manifest"
                            )
                        state_envelopes: dict[str, bytes] = {}
                        state_payloads: dict[str, bytes] = {}
                        for name, filename in state_filenames.items():
                            envelope = self._stable_read(
                                source_state_fd,
                                filename,
                                max_bytes=(
                                    4096
                                    + 4 * ((self.max_state_bytes + 2) // 3)
                                ),
                            )
                            payload, _, _ = self._decode_state(
                                envelope,
                                expected_name=name,
                            )
                            if name in expected_states and hashlib.sha256(
                                payload
                            ).hexdigest() != expected_states[name]:
                                raise ManifestConflictError(
                                    "source state differs from exact fork pin"
                                )
                            state_envelopes[name] = envelope
                            state_payloads[name] = payload
                        plan_body = {
                            "format": "immer-ooe-crystal-store-fork-plan/v1",
                            "manifest_sha256": manifest.sha256,
                            "state_sha256s": {
                                name: hashlib.sha256(payload).hexdigest()
                                for name, payload in sorted(state_payloads.items())
                            },
                            "target_realpath_sha256": hashlib.sha256(
                                str(target_real).encode("utf-8")
                            ).hexdigest(),
                        }
                        plan = canonical_json_bytes(
                            {
                                "body": plan_body,
                                "schema": "immer-ooe-crystal-store-fork-plan/v1",
                                "sha256": hashlib.sha256(
                                    canonical_json_bytes(plan_body)
                                ).hexdigest(),
                            }
                        )

                        try:
                            target_metadata = os.stat(
                                target.name,
                                dir_fd=parent_fd,
                                follow_symlinks=False,
                            )
                        except FileNotFoundError:
                            target_metadata = None
                        if target_metadata is not None and not stat.S_ISDIR(
                            target_metadata.st_mode
                        ):
                            raise CrystalStoreError(
                                "fork target is not a real directory"
                            )
                        try:
                            prior_parent_intent = self._stable_read(
                                parent_fd,
                                parent_intent_name,
                                max_bytes=64 * 1024,
                            )
                        except FileNotFoundError:
                            prior_parent_intent = None
                        if target_metadata is not None and prior_parent_intent is None:
                            raise FileExistsError(
                                f"fork target already exists: {target}"
                            )
                        if (
                            prior_parent_intent is not None
                            and prior_parent_intent != plan
                        ):
                            raise CrystalStoreError(
                                "fork parent intent belongs to another plan"
                            )
                        publish_immutable(parent_fd, parent_intent_name, plan)
                        if target_metadata is None:
                            try:
                                os.mkdir(target.name, 0o700, dir_fd=parent_fd)
                            except FileExistsError:
                                if prior_parent_intent is None:
                                    if self._stable_read(
                                        parent_fd,
                                        parent_intent_name,
                                        max_bytes=64 * 1024,
                                    ) == plan:
                                        os.unlink(
                                            parent_intent_name,
                                            dir_fd=parent_fd,
                                        )
                                        os.fsync(parent_fd)
                                raise
                        else:
                            if not stat.S_ISDIR(target_metadata.st_mode):
                                raise CrystalStoreError(
                                    "fork target is not a real directory"
                                )
                        target_fd = os.open(
                            target.name,
                            self._directory_flags(),
                            dir_fd=parent_fd,
                        )
                        try:
                            root_entries = set(os.listdir(target_fd))
                            marker_temporaries = {
                                name
                                for name in root_entries
                                if name.startswith(f".{marker_name}.")
                                and name.endswith(".tmp")
                            }
                            for temporary in marker_temporaries:
                                temporary_metadata = os.stat(
                                    temporary,
                                    dir_fd=target_fd,
                                    follow_symlinks=False,
                                )
                                if not stat.S_ISREG(temporary_metadata.st_mode):
                                    raise CrystalStoreError(
                                        "fork marker residue is not a regular file"
                                    )
                                os.unlink(temporary, dir_fd=target_fd)
                            if marker_temporaries:
                                os.fsync(target_fd)
                                root_entries -= marker_temporaries
                            if marker_name in root_entries:
                                if self._stable_read(
                                    target_fd,
                                    marker_name,
                                    max_bytes=64 * 1024,
                                ) != plan:
                                    raise CrystalStoreError(
                                        "fork target carries another plan marker"
                                    )
                            elif root_entries:
                                raise FileExistsError(
                                    f"fork target already exists: {target}"
                                )
                            else:
                                publish_immutable(target_fd, marker_name, plan)

                            child_fds: dict[str, int] = {}
                            try:
                                for child in ("objects", "staging", "state"):
                                    try:
                                        os.mkdir(child, 0o700, dir_fd=target_fd)
                                    except FileExistsError:
                                        child_metadata = os.stat(
                                            child,
                                            dir_fd=target_fd,
                                            follow_symlinks=False,
                                        )
                                        if not stat.S_ISDIR(child_metadata.st_mode):
                                            raise CrystalStoreError(
                                                "fork managed child is not a directory"
                                            )
                                    child_fds[child] = os.open(
                                        child,
                                        self._directory_flags(),
                                        dir_fd=target_fd,
                                    )
                                allowed_root = {
                                    marker_name,
                                    "objects",
                                    "staging",
                                    "state",
                                    self._MANIFEST,
                                    self._LOCK,
                                }
                                if set(os.listdir(target_fd)) - allowed_root:
                                    raise CrystalStoreError(
                                        "fork target root contains unexpected files"
                                    )
                                if set(os.listdir(child_fds["objects"])) - (
                                    expected_object_files
                                ):
                                    raise CrystalStoreError(
                                        "fork target contains unexpected objects"
                                    )
                                if os.listdir(child_fds["staging"]):
                                    raise CrystalStoreError(
                                        "fork target staging is not empty"
                                    )
                                if set(os.listdir(child_fds["state"])) - set(
                                    state_filenames.values()
                                ):
                                    raise CrystalStoreError(
                                        "fork target contains unexpected states"
                                    )
                                for digest in manifest.objects:
                                    data = self._stable_read(
                                        source_objects_fd,
                                        f"{digest}.crystal",
                                        max_bytes=MAX_CRYSTAL_PAYLOAD_BYTES,
                                    )
                                    if hashlib.sha256(data).hexdigest() != digest:
                                        raise CrystalTamperError(
                                            "source object changed during exact fork"
                                        )
                                    payload = CrystalPayload.from_bytes(data)
                                    if payload.sha256 != digest:
                                        raise CrystalTamperError(
                                            "source object semantic identity changed"
                                        )
                                    install_exact(
                                        child_fds["objects"],
                                        f"{digest}.crystal",
                                        data,
                                    )
                                manifest_bytes = manifest.to_bytes()
                                install_exact(
                                    target_fd,
                                    self._MANIFEST,
                                    manifest_bytes,
                                )
                                for name, envelope in state_envelopes.items():
                                    install_exact(
                                        child_fds["state"],
                                        state_filenames[name],
                                        envelope,
                                    )
                                for descriptor in child_fds.values():
                                    os.fsync(descriptor)
                                os.fsync(target_fd)
                            finally:
                                for descriptor in child_fds.values():
                                    os.close(descriptor)
                        finally:
                            os.close(target_fd)

                        prepared = CrystalStore(
                            target,
                            max_state_bytes=self.max_state_bytes,
                            _allow_incomplete_fork=True,
                        )
                        audit = prepared.audit()
                        if (
                            not audit.clean
                            or prepared.manifest().to_bytes() != manifest_bytes
                            or any(
                                prepared.restore_state(name) != payload
                                for name, payload in state_payloads.items()
                            )
                        ):
                            raise CrystalStoreError(
                                "prepared exact fork failed its audit"
                            )
                        target_fd = os.open(
                            target.name,
                            self._directory_flags(),
                            dir_fd=parent_fd,
                        )
                        try:
                            if self._stable_read(
                                target_fd,
                                marker_name,
                                max_bytes=64 * 1024,
                            ) != plan:
                                raise CrystalStoreError(
                                    "fork plan marker changed before commit"
                                )
                            os.unlink(marker_name, dir_fd=target_fd)
                            os.fsync(target_fd)
                        finally:
                            os.close(target_fd)
                        if self._read_manifest_fd(source_root_fd) != manifest:
                            raise ManifestConflictError(
                                "source manifest changed during exact fork"
                            )
                        for name, filename in state_filenames.items():
                            if self._stable_read(
                                source_state_fd,
                                filename,
                                max_bytes=(
                                    4096
                                    + 4 * ((self.max_state_bytes + 2) // 3)
                                ),
                            ) != state_envelopes[name]:
                                raise ManifestConflictError(
                                    "source state changed during exact fork"
                                )
                        if self._stable_read(
                            parent_fd,
                            parent_intent_name,
                            max_bytes=64 * 1024,
                        ) != plan:
                            raise CrystalStoreError(
                                "fork parent intent changed before commit"
                            )
                        os.unlink(parent_intent_name, dir_fd=parent_fd)
                        os.fsync(parent_fd)
                finally:
                    os.close(source_state_fd)
            os.fsync(parent_fd)
            result = CrystalStore(target, max_state_bytes=self.max_state_bytes)
            result_audit = result.audit()
            if (
                not result_audit.clean
                or result.manifest().to_bytes() != manifest_bytes
                or any(
                    result.restore_state(name) != payload
                    for name, payload in state_payloads.items()
                )
            ):
                raise CrystalStoreError("installed exact fork failed its audit")
            return result
        finally:
            try:
                fcntl.flock(parent_lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(parent_lock_fd)
                os.close(parent_fd)

    def publish(
        self,
        payload: CrystalPayload,
        *,
        expected_generation: int | None = None,
    ) -> CrystalPublication:
        if not isinstance(payload, CrystalPayload):
            raise TypeError("payload must be a CrystalPayload")
        if expected_generation is not None and (
            isinstance(expected_generation, bool)
            or not isinstance(expected_generation, int)
            or expected_generation < 0
        ):
            raise ValueError("expected_generation must be a non-negative integer")
        data = payload.to_bytes()
        digest = hashlib.sha256(data).hexdigest()

        with self._directories() as (root_fd, objects_fd, staging_fd):
            with self._locked(root_fd):
                current = self._read_manifest_fd(root_fd)
                if (
                    expected_generation is not None
                    and current.generation != expected_generation
                ):
                    raise ManifestConflictError(
                        f"manifest generation is {current.generation}, "
                        f"expected {expected_generation}"
                    )
                entries = {entry.name: entry for entry in current.entries}
                previous = entries.get(payload.name)
                if (
                    previous is not None
                    and previous.identity_sha256 != payload.identity.sha256
                ):
                    raise CrystalIdentityError(
                        "a crystal name cannot be rebound to another site identity"
                    )
                object_created = self._publish_object(
                    objects_fd, staging_fd, data, digest
                )
                if previous is not None:
                    if previous.payload_sha256 == digest:
                        return CrystalPublication(
                            payload_sha256=digest,
                            identity_sha256=payload.identity.sha256,
                            generation=current.generation,
                            manifest_sha256=current.sha256,
                            object_created=object_created,
                            manifest_changed=False,
                        )
                entries[payload.name] = CrystalManifestEntry(
                    name=payload.name,
                    payload_sha256=digest,
                    identity_sha256=payload.identity.sha256,
                )
                updated = CrystalManifest(
                    generation=current.generation + 1,
                    entries=tuple(
                        sorted(entries.values(), key=lambda entry: entry.name)
                    ),
                    objects=tuple(sorted(set(current.objects) | {digest})),
                    previous_manifest_sha256=current.sha256,
                )
                self._replace_manifest(root_fd, staging_fd, updated)
                return CrystalPublication(
                    payload_sha256=digest,
                    identity_sha256=payload.identity.sha256,
                    generation=updated.generation,
                    manifest_sha256=updated.sha256,
                    object_created=object_created,
                    manifest_changed=True,
                )

    def _restore_object_fd(self, objects_fd: int, digest: str) -> CrystalPayload:
        try:
            data = self._stable_read(
                objects_fd,
                f"{digest}.crystal",
                max_bytes=MAX_CRYSTAL_PAYLOAD_BYTES,
            )
        except FileNotFoundError as exc:
            raise KeyError(f"unknown crystal payload: {digest}") from exc
        actual = hashlib.sha256(data).hexdigest()
        if actual != digest:
            raise CrystalTamperError(
                f"crystal payload digest mismatch: expected {digest}, got {actual}"
            )
        payload = CrystalPayload.from_bytes(data)
        if payload.sha256 != digest:
            raise CrystalTamperError("restored crystal failed semantic rehash")
        return payload

    def restore(self, payload_sha256: str) -> CrystalPayload:
        digest = require_sha256(payload_sha256, field="payload_sha256")
        with self._directories() as (_, objects_fd, _):
            return self._restore_object_fd(objects_fd, digest)

    def restore_named(self, name: str) -> CrystalPayload:
        with self._directories() as (root_fd, objects_fd, _):
            with self._locked(root_fd):
                manifest = self._read_manifest_fd(root_fd)
                entry = manifest.resolve(name)
                payload = self._restore_object_fd(objects_fd, entry.payload_sha256)
                if (
                    payload.name != name
                    or payload.identity.sha256 != entry.identity_sha256
                ):
                    raise CrystalTamperError(
                        "manifest-to-crystal identity binding mismatch"
                    )
                return payload

    def audit(self) -> CrystalStoreAudit:
        with self._directories() as (root_fd, objects_fd, staging_fd):
            manifest = self._read_manifest_fd(root_fd)
            valid: list[str] = []
            tampered: list[str] = []
            unexpected: list[str] = []
            present: set[str] = set()
            for name in sorted(os.listdir(objects_fd)):
                if not name.endswith(".crystal"):
                    unexpected.append(name)
                    continue
                digest = name[: -len(".crystal")]
                try:
                    require_sha256(digest, field="object filename")
                    data = self._stable_read(
                        objects_fd,
                        name,
                        max_bytes=MAX_CRYSTAL_PAYLOAD_BYTES,
                    )
                    if hashlib.sha256(data).hexdigest() != digest:
                        raise CrystalTamperError("object content hash mismatch")
                    CrystalPayload.from_bytes(data)
                except (OSError, ValueError, CrystalStoreError):
                    tampered.append(name)
                    continue
                present.add(digest)
                valid.append(digest)
            declared = set(manifest.objects)
            return CrystalStoreAudit(
                generation=manifest.generation,
                manifest_sha256=manifest.sha256,
                valid_objects=tuple(valid),
                orphan_objects=tuple(sorted(present - declared)),
                missing_objects=tuple(sorted(declared - present)),
                tampered_objects=tuple(tampered),
                staged_files=tuple(sorted(os.listdir(staging_fd))),
                unexpected_files=tuple(unexpected),
            )

    def remove_orphans(self, *, expected_generation: int) -> tuple[str, ...]:
        """Remove only verified objects absent from a CAS-pinned manifest."""

        if (
            isinstance(expected_generation, bool)
            or not isinstance(expected_generation, int)
            or expected_generation < 0
        ):
            raise ValueError("expected_generation must be a non-negative integer")
        removed: list[str] = []
        with self._directories() as (root_fd, objects_fd, _):
            with self._locked(root_fd):
                manifest = self._read_manifest_fd(root_fd)
                if manifest.generation != expected_generation:
                    raise ManifestConflictError(
                        f"manifest generation is {manifest.generation}, "
                        f"expected {expected_generation}"
                    )
                declared = set(manifest.objects)
                for name in sorted(os.listdir(objects_fd)):
                    if not name.endswith(".crystal"):
                        continue
                    digest = name[: -len(".crystal")]
                    if digest in declared:
                        continue
                    try:
                        require_sha256(digest, field="object filename")
                        data = self._stable_read(
                            objects_fd,
                            name,
                            max_bytes=MAX_CRYSTAL_PAYLOAD_BYTES,
                        )
                    except (OSError, ValueError, CrystalStoreError):
                        continue
                    if hashlib.sha256(data).hexdigest() != digest:
                        continue
                    os.unlink(name, dir_fd=objects_fd)
                    removed.append(digest)
                if removed:
                    os.fsync(objects_fd)
        return tuple(removed)

    def clean_staging(self, *, expected_generation: int) -> tuple[str, ...]:
        """Remove regular crash residues while holding the manifest CAS lock."""

        if (
            isinstance(expected_generation, bool)
            or not isinstance(expected_generation, int)
            or expected_generation < 0
        ):
            raise ValueError("expected_generation must be a non-negative integer")
        removed: list[str] = []
        with self._directories() as (root_fd, _, staging_fd):
            with self._locked(root_fd):
                manifest = self._read_manifest_fd(root_fd)
                if manifest.generation != expected_generation:
                    raise ManifestConflictError(
                        f"manifest generation is {manifest.generation}, "
                        f"expected {expected_generation}"
                    )
                for name in sorted(os.listdir(staging_fd)):
                    try:
                        metadata = os.stat(
                            name, dir_fd=staging_fd, follow_symlinks=False
                        )
                    except FileNotFoundError:
                        continue
                    if stat.S_ISREG(metadata.st_mode) and name.endswith(".tmp"):
                        os.unlink(name, dir_fd=staging_fd)
                        removed.append(name)
                if removed:
                    os.fsync(staging_fd)
        return tuple(removed)

    @staticmethod
    def _state_filename(name: str) -> str:
        if not isinstance(name, str) or not name or "\x00" in name:
            raise ValueError("state name must be a non-empty string")
        if len(name.encode("utf-8")) > 1024:
            raise ValueError("state name is too long")
        return f"{hashlib.sha256(name.encode('utf-8')).hexdigest()}.state"

    def _decode_state(
        self, data: bytes, *, expected_name: str
    ) -> tuple[bytes, str, int]:
        envelope_limit = 4096 + 4 * ((self.max_state_bytes + 2) // 3)
        if len(data) > envelope_limit:
            raise CrystalTamperError(
                "controller-state envelope exceeds its hard byte limit"
            )
        value = _strict_json(data)
        if not isinstance(value, dict) or set(value) != {
            "format",
            "generation",
            "name",
            "payload_base64",
            "payload_sha256",
        }:
            raise CrystalTamperError("invalid controller-state envelope")
        if value["format"] != "immer-ooe-controller-state/v1":
            raise CrystalTamperError("unsupported controller-state format")
        if value["name"] != expected_name:
            raise CrystalTamperError("controller-state name binding mismatch")
        generation = value["generation"]
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 1
        ):
            raise CrystalTamperError("invalid controller-state generation")
        encoded = value["payload_base64"]
        if (
            not isinstance(encoded, str)
            or not encoded.isascii()
            or len(encoded) > 4 * ((self.max_state_bytes + 2) // 3)
        ):
            raise CrystalTamperError("controller-state encoding exceeds its byte bound")
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (TypeError, ValueError) as exc:
            raise CrystalTamperError("invalid controller-state encoding") from exc
        if len(payload) > self.max_state_bytes:
            raise CrystalTamperError("controller state exceeds configured byte bound")
        digest = require_sha256(value["payload_sha256"], field="payload_sha256")
        if hashlib.sha256(payload).hexdigest() != digest:
            raise CrystalTamperError("controller-state payload hash mismatch")
        return payload, digest, generation

    def publish_state(
        self,
        name: str,
        payload: bytes,
        *,
        expected_sha256: str | None = None,
    ) -> StatePublication:
        """Atomically publish bounded hash-sealed controller state."""

        filename = self._state_filename(name)
        if not isinstance(payload, bytes):
            raise TypeError("state payload must be immutable bytes")
        if len(payload) > self.max_state_bytes:
            raise ValueError(
                f"state payload exceeds max_state_bytes={self.max_state_bytes}"
            )
        if expected_sha256 is not None:
            expected_sha256 = require_sha256(expected_sha256, field="expected_sha256")
        digest = hashlib.sha256(payload).hexdigest()

        with self._directories() as (root_fd, _, staging_fd):
            state_fd = os.open("state", self._directory_flags(), dir_fd=root_fd)
            try:
                with self._locked(root_fd):
                    try:
                        current_data = self._stable_read(
                            state_fd,
                            filename,
                            max_bytes=(4096 + 4 * ((self.max_state_bytes + 2) // 3)),
                        )
                    except FileNotFoundError:
                        current = None
                    else:
                        current = self._decode_state(current_data, expected_name=name)
                    if expected_sha256 is not None and (
                        current is None or current[1] != expected_sha256
                    ):
                        actual = None if current is None else current[1]
                        raise ManifestConflictError(
                            f"controller-state digest is {actual}, "
                            f"expected {expected_sha256}"
                        )
                    if current is not None and current[1] == digest:
                        return StatePublication(
                            name=name,
                            payload_sha256=digest,
                            generation=current[2],
                            changed=False,
                        )
                    generation = 1 if current is None else current[2] + 1
                    envelope = canonical_json_bytes(
                        {
                            "format": "immer-ooe-controller-state/v1",
                            "generation": generation,
                            "name": name,
                            "payload_base64": base64.b64encode(payload).decode("ascii"),
                            "payload_sha256": digest,
                        }
                    )
                    temporary = self._temporary_name("state")
                    flags = (
                        os.O_CREAT
                        | os.O_EXCL
                        | os.O_WRONLY
                        | getattr(os, "O_NOFOLLOW", 0)
                    )
                    fd = os.open(temporary, flags, 0o600, dir_fd=staging_fd)
                    try:
                        self._write_all(fd, envelope)
                        os.fsync(fd)
                        os.fchmod(fd, 0o444)
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                    try:
                        os.replace(
                            temporary,
                            filename,
                            src_dir_fd=staging_fd,
                            dst_dir_fd=state_fd,
                        )
                        os.fsync(state_fd)
                    finally:
                        try:
                            os.unlink(temporary, dir_fd=staging_fd)
                        except FileNotFoundError:
                            pass
                    return StatePublication(
                        name=name,
                        payload_sha256=digest,
                        generation=generation,
                        changed=True,
                    )
            finally:
                os.close(state_fd)

    def restore_state(self, name: str) -> bytes:
        filename = self._state_filename(name)
        with self._directories() as (root_fd, _, _):
            state_fd = os.open("state", self._directory_flags(), dir_fd=root_fd)
            try:
                try:
                    data = self._stable_read(
                        state_fd,
                        filename,
                        max_bytes=(4096 + 4 * ((self.max_state_bytes + 2) // 3)),
                    )
                except FileNotFoundError as exc:
                    raise KeyError(f"unknown controller state: {name}") from exc
            finally:
                os.close(state_fd)
        payload, _, _ = self._decode_state(data, expected_name=name)
        return payload
