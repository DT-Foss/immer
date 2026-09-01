"""Authenticated, pickle-free sidecars for :class:`Qwen35MtpCarry`.

The sidecar deliberately does not contain token IDs or prompt text.  A caller-
supplied token-prefix digest binds the tensors to history reconstructed by the
reader.  Files are immutable, content addressed, and consist of a bounded
canonical JSON header followed by uncompressed raw tensor storage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import struct
import sys
from typing import Any

import torch

from ..ooe.identity import canonical_json_bytes, require_sha256
from .kernels import AttentionState
from .mtp_draft import QWEN35_MTP_CARRY_SCHEMA, Qwen35MtpCarry


QWEN35_MTP_CARRY_SIDECAR_SCHEMA = "immer.qwen3.5-mtp-carry-sidecar/v1"
QWEN35_MTP_CARRY_SIDECAR_VERSION = 1
_TOKEN_PREFIX_SCHEMA = "immer.qwen3.8-token-prefix/v1"

_MAGIC = b"IMMER-MTP-CARRY\x00\x01"
_HEADER_LENGTH = struct.Struct(">I")
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_BASENAME_RE = re.compile(r"([0-9a-f]{64})\.qwen35-mtp-carry")
_MAX_HEADER_BYTES = 256 * 1024
_MAX_TENSOR_BYTES = 2 * 1024**3
_MAX_SIDECAR_BYTES = _MAX_TENSOR_BYTES + _MAX_HEADER_BYTES + len(_MAGIC) + 4
_MAX_PREFIX_TOKENS = 1_048_576
_MAX_IDENTITY_BYTES = 64 * 1024
_MAX_IDENTITY_DEPTH = 16
_MAX_IDENTITY_NODES = 4096
_READ_CHUNK_BYTES = 1024 * 1024

_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


class Qwen35MtpCarrySidecarError(ValueError):
    """A carry sidecar is corrupt, unsafe, or incompatible."""


class Qwen35MtpCarrySidecarIdentityMismatch(Qwen35MtpCarrySidecarError):
    """A valid sidecar is bound to another prefix or runtime identity."""


def _digest(value: object, label: str) -> str:
    try:
        return require_sha256(value, field=label)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise Qwen35MtpCarrySidecarError(str(exc)) from exc


def _sha256_bytes(value: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(value).hexdigest()


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Qwen35MtpCarrySidecarError(
            f"{label} must be a non-negative integer"
        )
    return value


def _positive_int(value: object, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result == 0:
        raise Qwen35MtpCarrySidecarError(f"{label} must be positive")
    return result


def _canonical(value: object, label: str) -> bytes:
    try:
        return canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise Qwen35MtpCarrySidecarError(
            f"{label} is not canonical JSON"
        ) from exc


def _json_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Qwen35MtpCarrySidecarError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _identity_record(value: object) -> tuple[object, bytes]:
    nodes = 0

    def normalize(item: object, depth: int) -> object:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_IDENTITY_NODES or depth > _MAX_IDENTITY_DEPTH:
            raise Qwen35MtpCarrySidecarError("carry identity exceeds its bound")
        if item is None or isinstance(item, (bool, str)):
            return item
        if isinstance(item, int):
            if not -(2**63) <= item <= 2**63 - 1:
                raise Qwen35MtpCarrySidecarError(
                    "carry identity integer is outside signed 64-bit range"
                )
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise Qwen35MtpCarrySidecarError(
                    "carry identity contains a non-finite number"
                )
            return item
        if isinstance(item, Mapping):
            if len(item) > _MAX_IDENTITY_NODES:
                raise Qwen35MtpCarrySidecarError("carry identity mapping is too large")
            record: dict[str, object] = {}
            for key, nested in item.items():
                if not isinstance(key, str) or len(key.encode("utf-8")) > 4096:
                    raise Qwen35MtpCarrySidecarError(
                        "carry identity mapping keys must be bounded strings"
                    )
                record[key] = normalize(nested, depth + 1)
            return record
        if isinstance(item, (tuple, list)):
            if len(item) > _MAX_IDENTITY_NODES:
                raise Qwen35MtpCarrySidecarError("carry identity sequence is too large")
            return [normalize(nested, depth + 1) for nested in item]
        raise Qwen35MtpCarrySidecarError(
            f"carry identity contains unsupported value {type(item).__name__!r}"
        )

    if not isinstance(value, (Mapping, tuple, list)):
        raise Qwen35MtpCarrySidecarError(
            "carry identity must be a canonical mapping or sequence"
        )
    record = normalize(value, 0)
    encoded = _canonical(record, "carry identity")
    if len(encoded) > _MAX_IDENTITY_BYTES:
        raise Qwen35MtpCarrySidecarError("carry identity exceeds its byte limit")
    return record, encoded


def qwen35_mtp_carry_identity_sha256(identity: object) -> str:
    """Hash a canonical (possibly nested) MTP provider identity."""

    _record, encoded = _identity_record(identity)
    return _sha256_bytes(encoded)


def _copy_identity(value: object) -> object:
    """Detach caller-owned identity containers while preserving equality."""

    if isinstance(value, tuple):
        return tuple(_copy_identity(item) for item in value)
    if isinstance(value, list):
        return [_copy_identity(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _copy_identity(item) for key, item in value.items()}
    return value


def _history_tuple(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise Qwen35MtpCarrySidecarError("history must be an integer sequence")
    if not value or len(value) > _MAX_PREFIX_TOKENS:
        raise Qwen35MtpCarrySidecarError(
            "history must contain a bounded non-empty prefix"
        )
    result: list[int] = []
    for token in value:
        if (
            isinstance(token, bool)
            or not isinstance(token, int)
            or not 0 <= token <= 2**63 - 1
        ):
            raise Qwen35MtpCarrySidecarError(
                "history must contain non-negative signed 64-bit token IDs"
            )
        result.append(token)
    return tuple(result)


def qwen35_mtp_prefix_sha256(history: Sequence[int]) -> str:
    """Return the cache-compatible digest of one token-ID prefix."""

    tokens = _history_tuple(history)
    return _sha256_bytes(
        _canonical(
            {"schema": _TOKEN_PREFIX_SCHEMA, "token_ids": list(tokens)},
            "token prefix",
        )
    )


@dataclass(frozen=True, slots=True)
class MtpCarrySidecarDescriptor:
    """Path-free authenticated reference to one immutable carry sidecar."""

    basename: str
    bytes: int
    file_sha256: str
    identity_sha256: str
    tensor_manifest_sha256: str

    def __post_init__(self) -> None:
        file_sha = _digest(self.file_sha256, "file_sha256")
        _digest(self.identity_sha256, "identity_sha256")
        _digest(self.tensor_manifest_sha256, "tensor_manifest_sha256")
        if (
            not isinstance(self.basename, str)
            or _BASENAME_RE.fullmatch(self.basename) is None
            or self.basename != f"{file_sha}.qwen35-mtp-carry"
            or Path(self.basename).name != self.basename
        ):
            raise Qwen35MtpCarrySidecarError(
                "sidecar basename is not its content address"
            )
        if (
            isinstance(self.bytes, bool)
            or not isinstance(self.bytes, int)
            or not len(_MAGIC) + 4 < self.bytes <= _MAX_SIDECAR_BYTES
        ):
            raise Qwen35MtpCarrySidecarError("sidecar byte count exceeds its bound")

    @property
    def filename(self) -> str:
        return self.basename

    @property
    def size_bytes(self) -> int:
        return self.bytes

    @property
    def file_bytes(self) -> int:
        return self.bytes

    @property
    def payload_bytes(self) -> int:
        return self.bytes

    @property
    def sha256(self) -> str:
        return self.file_sha256

    @property
    def payload_sha256(self) -> str:
        return self.file_sha256

    def to_record(self) -> dict[str, object]:
        return {
            "basename": self.basename,
            "bytes": self.bytes,
            "file_sha256": self.file_sha256,
            "identity_sha256": self.identity_sha256,
            "tensor_manifest_sha256": self.tensor_manifest_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return self.to_record()

    def to_document(self) -> dict[str, object]:
        return self.to_record()

    @classmethod
    def from_record(cls, value: object) -> "MtpCarrySidecarDescriptor":
        expected = {
            "basename",
            "bytes",
            "file_sha256",
            "identity_sha256",
            "tensor_manifest_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise Qwen35MtpCarrySidecarError(
                "sidecar descriptor has unknown or missing fields"
            )
        return cls(
            basename=value["basename"],  # type: ignore[arg-type]
            bytes=value["bytes"],  # type: ignore[arg-type]
            file_sha256=value["file_sha256"],  # type: ignore[arg-type]
            identity_sha256=value["identity_sha256"],  # type: ignore[arg-type]
            tensor_manifest_sha256=value["tensor_manifest_sha256"],  # type: ignore[arg-type]
        )

    @classmethod
    def from_dict(cls, value: object) -> "MtpCarrySidecarDescriptor":
        return cls.from_record(value)

    @classmethod
    def from_document(cls, value: object) -> "MtpCarrySidecarDescriptor":
        return cls.from_record(value)


def _validated_descriptor(value: object) -> MtpCarrySidecarDescriptor:
    if not isinstance(value, MtpCarrySidecarDescriptor):
        if isinstance(value, Mapping):
            return MtpCarrySidecarDescriptor.from_record(value)
        raise TypeError("descriptor must be an MtpCarrySidecarDescriptor")
    return MtpCarrySidecarDescriptor.from_record(value.to_record())


def _dtype_name(dtype: torch.dtype) -> str:
    name = str(dtype).removeprefix("torch.")
    if name not in _DTYPES:
        raise Qwen35MtpCarrySidecarError(
            f"MTP carry tensor dtype {name!r} is not supported"
        )
    return name


def _tensor_bytes(value: torch.Tensor) -> memoryview:
    raw = value.view(torch.uint8).reshape(-1).numpy()
    return memoryview(raw)


def _owned_cpu_tensor(value: object, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise Qwen35MtpCarrySidecarError(f"{label} must be a floating tensor")
    _dtype_name(value.dtype)
    owned = value.detach().to(device="cpu").contiguous().clone()
    if not bool(torch.isfinite(owned).all().item()):
        raise Qwen35MtpCarrySidecarError(f"{label} must contain finite values")
    return owned


def _prepare_carry(
    carry: Qwen35MtpCarry,
    *,
    tokenizer_sha256: str,
    prefix_sha256: str,
) -> tuple[dict[str, Any], list[tuple[str, torch.Tensor]]]:
    if not isinstance(carry, Qwen35MtpCarry):
        raise TypeError("carry must be a Qwen35MtpCarry")
    if carry.schema != QWEN35_MTP_CARRY_SCHEMA:
        raise Qwen35MtpCarrySidecarError("unsupported MTP carry schema")
    tokenizer_digest = _digest(tokenizer_sha256, "tokenizer_sha256")
    prefix_digest = _digest(prefix_sha256, "prefix_sha256")
    history = _history_tuple(carry.history)
    if prefix_digest != qwen35_mtp_prefix_sha256(history):
        raise Qwen35MtpCarrySidecarIdentityMismatch(
            "MTP carry history does not match prefix_sha256"
        )
    if (
        isinstance(carry.next_position, bool)
        or not isinstance(carry.next_position, int)
        or carry.next_position != len(history) - 1
    ):
        raise Qwen35MtpCarrySidecarError(
            "MTP carry cursor is not aligned to its prefix"
        )
    identity, identity_bytes = _identity_record(carry.identity)
    identity_sha = _sha256_bytes(identity_bytes)

    hidden = _owned_cpu_tensor(carry.last_target_hidden, "last_target_hidden")
    if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] != 1 or hidden.shape[2] <= 0:
        raise Qwen35MtpCarrySidecarError(
            "last_target_hidden must have exact shape [1, 1, hidden_dim]"
        )

    tensors: list[tuple[str, torch.Tensor]] = [("last_target_hidden", hidden)]
    state = carry.state
    if len(history) == 1:
        if state is not None:
            raise Qwen35MtpCarrySidecarError(
                "one-token MTP carry must not contain attention state"
            )
    else:
        if not isinstance(state, AttentionState):
            raise Qwen35MtpCarrySidecarError(
                "multi-token MTP carry requires attention state"
            )
        if state.crsa_log_usage is not None:
            raise Qwen35MtpCarrySidecarError(
                "MTP carry sidecars do not support CRSA usage state"
            )
        key = _owned_cpu_tensor(state.key, "state.key")
        value = _owned_cpu_tensor(state.value, "state.value")
        if (
            key.ndim != 4
            or tuple(key.shape) != tuple(value.shape)
            or key.shape[0] != 1
            or key.shape[1] <= 0
            or key.shape[2] != len(history) - 1
            or key.shape[3] <= 0
            or key.dtype != value.dtype
            or key.dtype != hidden.dtype
        ):
            raise Qwen35MtpCarrySidecarError(
                "MTP key/value tensors do not match the prefix contract"
            )
        tensors.extend((("state.key", key), ("state.value", value)))

    tensors.sort(key=lambda row: row[0])
    descriptors: list[dict[str, Any]] = []
    offset = 0
    payload_hasher = hashlib.sha256()
    for name, tensor in tensors:
        raw = _tensor_bytes(tensor)
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes != len(raw):
            raise Qwen35MtpCarrySidecarError("MTP tensor byte count overflow")
        offset += nbytes
        if offset > _MAX_TENSOR_BYTES:
            raise Qwen35MtpCarrySidecarError(
                "MTP carry tensors exceed their total byte limit"
            )
        payload_hasher.update(raw)
        descriptors.append(
            {
                "name": name,
                "dtype": _dtype_name(tensor.dtype),
                "shape": [int(dimension) for dimension in tensor.shape],
                "offset": offset - nbytes,
                "nbytes": nbytes,
                "sha256": _sha256_bytes(raw),
            }
        )
    tensor_manifest_sha = _sha256_bytes(_canonical(descriptors, "tensor manifest"))
    body: dict[str, Any] = {
        "schema": QWEN35_MTP_CARRY_SIDECAR_SCHEMA,
        "byte_order": sys.byteorder,
        "carry_schema": QWEN35_MTP_CARRY_SCHEMA,
        "carry_identity": identity,
        "identity_sha256": identity_sha,
        "tokenizer_sha256": tokenizer_digest,
        "prefix_length": len(history),
        "prefix_sha256": prefix_digest,
        "next_position": carry.next_position,
        "tensors": descriptors,
        "tensor_count": len(descriptors),
        "tensor_bytes": offset,
        "tensor_manifest_sha256": tensor_manifest_sha,
        "payload_sha256": payload_hasher.hexdigest(),
    }
    return body, tensors


def _absolute(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _stable_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_root(path: Path, *, create: bool) -> int:
    if create:
        try:
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as exc:
            raise Qwen35MtpCarrySidecarError(
                "cannot create MTP carry sidecar root"
            ) from exc
    try:
        linked_before = os.lstat(path)
    except OSError as exc:
        raise Qwen35MtpCarrySidecarError(
            "MTP carry sidecar root is unavailable"
        ) from exc
    if not stat.S_ISDIR(linked_before.st_mode):
        raise Qwen35MtpCarrySidecarError(
            "MTP carry sidecar root must be a real directory"
        )
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Qwen35MtpCarrySidecarError(
            "cannot open MTP carry sidecar root safely"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        linked_after = os.lstat(path)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not _same_inode(linked_before, opened)
            or not _same_inode(opened, linked_after)
        ):
            raise Qwen35MtpCarrySidecarError(
                "MTP carry sidecar root changed while opening"
            )
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _write_all(descriptor: int, value: bytes | memoryview, hasher: Any) -> int:
    view = memoryview(value)
    total = 0
    while view:
        written = os.write(descriptor, view[:_READ_CHUNK_BYTES])
        if written <= 0:
            raise OSError("zero-byte MTP carry sidecar write")
        chunk = view[:written]
        hasher.update(chunk)
        total += written
        view = view[written:]
    return total


def _new_temporary(root_fd: int) -> tuple[int, str]:
    flags = (
        os.O_CREAT
        | os.O_EXCL
        | os.O_RDWR
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for _attempt in range(128):
        name = f".mtp-carry-{secrets.token_hex(16)}.pending"
        try:
            descriptor = os.open(name, flags, 0o600, dir_fd=root_fd)
        except FileExistsError:
            continue
        except OSError as exc:
            raise Qwen35MtpCarrySidecarError(
                "cannot create temporary MTP carry sidecar"
            ) from exc
        os.fchmod(descriptor, 0o600)
        return descriptor, name
    raise Qwen35MtpCarrySidecarError(
        "cannot allocate a unique temporary MTP carry sidecar"
    )


def _open_relative_regular(root_fd: int, name: str) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=root_fd)
    except OSError as exc:
        raise Qwen35MtpCarrySidecarError(
            "cannot open MTP carry sidecar safely"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        linked = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(linked.st_mode)
            or not _same_inode(opened, linked)
        ):
            raise Qwen35MtpCarrySidecarError(
                "MTP carry sidecar must be a stable regular file"
            )
        return descriptor, opened
    except Exception:
        os.close(descriptor)
        raise


def _files_equal(left_fd: int, right_fd: int, size: int) -> bool:
    if os.fstat(right_fd).st_size != size:
        return False
    os.lseek(left_fd, 0, os.SEEK_SET)
    os.lseek(right_fd, 0, os.SEEK_SET)
    remaining = size
    while remaining:
        width = min(remaining, _READ_CHUNK_BYTES)
        left = os.read(left_fd, width)
        right = os.read(right_fd, width)
        if not left or left != right:
            return False
        remaining -= len(left)
    return not os.read(left_fd, 1) and not os.read(right_fd, 1)


def _publish_temporary(
    root_fd: int,
    temporary_fd: int,
    temporary_name: str,
    basename: str,
    size: int,
) -> None:
    temporary_stat = os.fstat(temporary_fd)
    linked_temp = os.stat(temporary_name, dir_fd=root_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(temporary_stat.st_mode)
        or not _same_inode(temporary_stat, linked_temp)
        or temporary_stat.st_size != size
        or stat.S_IMODE(temporary_stat.st_mode) != 0o600
    ):
        raise Qwen35MtpCarrySidecarError(
            "temporary MTP carry sidecar changed before publication"
        )
    try:
        os.link(
            temporary_name,
            basename,
            src_dir_fd=root_fd,
            dst_dir_fd=root_fd,
            follow_symlinks=False,
        )
    except FileExistsError:
        existing_fd, existing_stat = _open_relative_regular(root_fd, basename)
        try:
            if (
                stat.S_IMODE(existing_stat.st_mode) != 0o600
                or not _files_equal(temporary_fd, existing_fd, size)
            ):
                raise Qwen35MtpCarrySidecarError(
                    "content-addressed MTP carry sidecar collision"
                )
        finally:
            os.close(existing_fd)
    except OSError as exc:
        raise Qwen35MtpCarrySidecarError(
            "cannot publish MTP carry sidecar atomically"
        ) from exc
    else:
        published = os.stat(basename, dir_fd=root_fd, follow_symlinks=False)
        if not _same_inode(temporary_stat, published):
            raise Qwen35MtpCarrySidecarError(
                "MTP carry sidecar publication raced with another writer"
            )
        os.fsync(root_fd)


def write_qwen35_mtp_carry_sidecar(
    root: str | os.PathLike[str],
    carry: Qwen35MtpCarry,
    *,
    tokenizer_sha256: str,
    prefix_sha256: str,
) -> MtpCarrySidecarDescriptor:
    """Publish one immutable content-addressed carry without token history."""

    body, tensors = _prepare_carry(
        carry,
        tokenizer_sha256=tokenizer_sha256,
        prefix_sha256=prefix_sha256,
    )
    body_bytes = _canonical(body, "MTP carry sidecar body")
    header = {
        "schema": QWEN35_MTP_CARRY_SIDECAR_SCHEMA,
        "version": QWEN35_MTP_CARRY_SIDECAR_VERSION,
        "body_sha256": _sha256_bytes(body_bytes),
        "body": body,
    }
    header_bytes = _canonical(header, "MTP carry sidecar header")
    if not 0 < len(header_bytes) <= _MAX_HEADER_BYTES:
        raise Qwen35MtpCarrySidecarError(
            "MTP carry sidecar header exceeds its byte limit"
        )

    root_path = _absolute(root)
    root_fd = _open_root(root_path, create=True)
    temporary_fd = -1
    temporary_name: str | None = None
    try:
        temporary_fd, temporary_name = _new_temporary(root_fd)
        file_hasher = hashlib.sha256()
        written = _write_all(temporary_fd, _MAGIC, file_hasher)
        written += _write_all(
            temporary_fd, _HEADER_LENGTH.pack(len(header_bytes)), file_hasher
        )
        written += _write_all(temporary_fd, header_bytes, file_hasher)
        for _name, tensor in tensors:
            written += _write_all(temporary_fd, _tensor_bytes(tensor), file_hasher)
        expected_size = len(_MAGIC) + 4 + len(header_bytes) + int(body["tensor_bytes"])
        if written != expected_size or written > _MAX_SIDECAR_BYTES:
            raise Qwen35MtpCarrySidecarError(
                "MTP carry sidecar byte count is inconsistent"
            )
        os.fsync(temporary_fd)
        file_sha = file_hasher.hexdigest()
        basename = f"{file_sha}.qwen35-mtp-carry"
        _publish_temporary(
            root_fd,
            temporary_fd,
            temporary_name,
            basename,
            written,
        )
        return MtpCarrySidecarDescriptor(
            basename=basename,
            bytes=written,
            file_sha256=file_sha,
            identity_sha256=body["identity_sha256"],
            tensor_manifest_sha256=body["tensor_manifest_sha256"],
        )
    except Qwen35MtpCarrySidecarError:
        raise
    except OSError as exc:
        raise Qwen35MtpCarrySidecarError(
            "cannot write MTP carry sidecar safely"
        ) from exc
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=root_fd)
            except FileNotFoundError:
                pass
        os.close(root_fd)


def _read_exact(descriptor: int, size: int, hasher: Any | None = None) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(descriptor, min(remaining, _READ_CHUNK_BYTES))
        if not chunk:
            raise Qwen35MtpCarrySidecarError("MTP carry sidecar was truncated")
        if hasher is not None:
            hasher.update(chunk)
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _hash_fd(descriptor: int, size: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    hasher = hashlib.sha256()
    remaining = size
    while remaining:
        chunk = os.read(descriptor, min(remaining, _READ_CHUNK_BYTES))
        if not chunk:
            raise Qwen35MtpCarrySidecarError("MTP carry sidecar was truncated")
        hasher.update(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1):
        raise Qwen35MtpCarrySidecarError("MTP carry sidecar grew while reading")
    return hasher.hexdigest()


def _parse_header(raw: bytes) -> dict[str, Any]:
    try:
        document = json.loads(
            raw,
            object_pairs_hook=_json_no_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                Qwen35MtpCarrySidecarError(f"invalid JSON constant: {value}")
            ),
        )
    except Qwen35MtpCarrySidecarError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise Qwen35MtpCarrySidecarError(
            "cannot decode MTP carry sidecar header"
        ) from exc
    if not isinstance(document, dict) or set(document) != {
        "schema",
        "version",
        "body_sha256",
        "body",
    }:
        raise Qwen35MtpCarrySidecarError("MTP carry sidecar header is invalid")
    if _canonical(document, "MTP carry sidecar header") != raw:
        raise Qwen35MtpCarrySidecarError(
            "MTP carry sidecar header is not canonical"
        )
    version = document["version"]
    if (
        document["schema"] != QWEN35_MTP_CARRY_SIDECAR_SCHEMA
        or isinstance(version, bool)
        or not isinstance(version, int)
        or version != QWEN35_MTP_CARRY_SIDECAR_VERSION
    ):
        raise Qwen35MtpCarrySidecarError("unsupported MTP carry sidecar schema")
    body = document["body"]
    if not isinstance(body, dict):
        raise Qwen35MtpCarrySidecarError("MTP carry sidecar body is invalid")
    body_digest = _digest(document["body_sha256"], "body_sha256")
    if body_digest != _sha256_bytes(_canonical(body, "MTP carry sidecar body")):
        raise Qwen35MtpCarrySidecarError(
            "MTP carry sidecar body SHA-256 mismatch"
        )
    return body


def _shape_and_bytes(descriptor: Mapping[str, Any]) -> tuple[tuple[int, ...], torch.dtype, int]:
    dtype_name = descriptor.get("dtype")
    if not isinstance(dtype_name, str) or dtype_name not in _DTYPES:
        raise Qwen35MtpCarrySidecarError("MTP tensor dtype is invalid")
    raw_shape = descriptor.get("shape")
    if not isinstance(raw_shape, list) or len(raw_shape) > 4:
        raise Qwen35MtpCarrySidecarError("MTP tensor rank exceeds its bound")
    shape: list[int] = []
    elements = 1
    for dimension in raw_shape:
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 0
            or dimension > _MAX_PREFIX_TOKENS
        ):
            raise Qwen35MtpCarrySidecarError("MTP tensor shape is invalid")
        shape.append(dimension)
        elements *= dimension
        if elements > _MAX_TENSOR_BYTES:
            raise Qwen35MtpCarrySidecarError("MTP tensor element count is excessive")
    dtype = _DTYPES[dtype_name]
    nbytes = elements * torch.empty((), dtype=dtype).element_size()
    if nbytes > _MAX_TENSOR_BYTES:
        raise Qwen35MtpCarrySidecarError("MTP tensor exceeds its byte limit")
    return tuple(shape), dtype, nbytes


def _validate_body(
    body: Mapping[str, Any],
    descriptor: MtpCarrySidecarDescriptor,
    *,
    history: tuple[int, ...],
    tokenizer_sha256: str,
    expected_identity: object,
) -> list[dict[str, Any]]:
    expected_keys = {
        "schema",
        "byte_order",
        "carry_schema",
        "carry_identity",
        "identity_sha256",
        "tokenizer_sha256",
        "prefix_length",
        "prefix_sha256",
        "next_position",
        "tensors",
        "tensor_count",
        "tensor_bytes",
        "tensor_manifest_sha256",
        "payload_sha256",
    }
    if set(body) != expected_keys:
        raise Qwen35MtpCarrySidecarError(
            "MTP carry sidecar body has unknown or missing fields"
        )
    if (
        body["schema"] != QWEN35_MTP_CARRY_SIDECAR_SCHEMA
        or body["carry_schema"] != QWEN35_MTP_CARRY_SCHEMA
        or body["byte_order"] != sys.byteorder
    ):
        raise Qwen35MtpCarrySidecarError(
            "MTP carry sidecar schema or byte order is incompatible"
        )

    prefix_digest = qwen35_mtp_prefix_sha256(history)
    expected_tokenizer = _digest(tokenizer_sha256, "tokenizer_sha256")
    stored_prefix = _digest(body["prefix_sha256"], "prefix_sha256")
    stored_tokenizer = _digest(body["tokenizer_sha256"], "tokenizer_sha256")
    prefix_length = _positive_int(body["prefix_length"], "prefix_length")
    next_position = _nonnegative_int(body["next_position"], "next_position")
    if (
        prefix_length != len(history)
        or next_position != len(history) - 1
        or stored_prefix != prefix_digest
        or stored_tokenizer != expected_tokenizer
    ):
        raise Qwen35MtpCarrySidecarIdentityMismatch(
            "MTP carry sidecar prefix or tokenizer identity mismatch"
        )

    stored_identity, identity_bytes = _identity_record(body["carry_identity"])
    expected_record, expected_bytes = _identity_record(expected_identity)
    identity_sha = _digest(body["identity_sha256"], "identity_sha256")
    if (
        stored_identity != expected_record
        or identity_bytes != expected_bytes
        or identity_sha != _sha256_bytes(identity_bytes)
        or identity_sha != descriptor.identity_sha256
    ):
        raise Qwen35MtpCarrySidecarIdentityMismatch(
            "MTP carry sidecar provider/model/Q4 identity mismatch"
        )

    raw_tensors = body["tensors"]
    if not isinstance(raw_tensors, list) or len(raw_tensors) not in {1, 3}:
        raise Qwen35MtpCarrySidecarError("MTP tensor manifest is invalid")
    tensor_count = _positive_int(body["tensor_count"], "tensor_count")
    if tensor_count != len(raw_tensors):
        raise Qwen35MtpCarrySidecarError("MTP tensor count is inconsistent")
    manifest_sha = _digest(
        body["tensor_manifest_sha256"], "tensor_manifest_sha256"
    )
    if (
        manifest_sha
        != _sha256_bytes(_canonical(raw_tensors, "MTP tensor manifest"))
        or manifest_sha != descriptor.tensor_manifest_sha256
    ):
        raise Qwen35MtpCarrySidecarError("MTP tensor manifest SHA-256 mismatch")
    _digest(body["payload_sha256"], "payload_sha256")

    parsed: list[dict[str, Any]] = []
    names: set[str] = set()
    offset = 0
    for raw in raw_tensors:
        if not isinstance(raw, dict) or set(raw) != {
            "name",
            "dtype",
            "shape",
            "offset",
            "nbytes",
            "sha256",
        }:
            raise Qwen35MtpCarrySidecarError("MTP tensor descriptor is invalid")
        name = raw["name"]
        if name not in {"last_target_hidden", "state.key", "state.value"} or name in names:
            raise Qwen35MtpCarrySidecarError(
                "MTP tensor name is invalid or duplicate"
            )
        shape, dtype, nbytes = _shape_and_bytes(raw)
        stored_offset = _nonnegative_int(raw["offset"], f"tensor {name} offset")
        stored_nbytes = _positive_int(raw["nbytes"], f"tensor {name} nbytes")
        if stored_offset != offset or stored_nbytes != nbytes:
            raise Qwen35MtpCarrySidecarError(
                "MTP tensor offsets or byte counts are inconsistent"
            )
        tensor_sha = _digest(raw["sha256"], f"tensor {name} SHA-256")
        offset += nbytes
        if offset > _MAX_TENSOR_BYTES:
            raise Qwen35MtpCarrySidecarError(
                "MTP carry tensors exceed their total byte limit"
            )
        names.add(name)
        parsed.append(
            {
                **raw,
                "shape_tuple": shape,
                "torch_dtype": dtype,
                "sha256": tensor_sha,
            }
        )
    expected_names = (
        {"last_target_hidden"}
        if len(history) == 1
        else {"last_target_hidden", "state.key", "state.value"}
    )
    if names != expected_names:
        raise Qwen35MtpCarrySidecarError(
            "MTP tensor set does not match the prefix length"
        )
    tensor_bytes = _positive_int(body["tensor_bytes"], "tensor_bytes")
    if tensor_bytes != offset:
        raise Qwen35MtpCarrySidecarError("MTP total tensor bytes are inconsistent")

    by_name = {row["name"]: row for row in parsed}
    hidden = by_name["last_target_hidden"]
    if (
        len(hidden["shape_tuple"]) != 3
        or hidden["shape_tuple"][:2] != (1, 1)
        or hidden["shape_tuple"][2] <= 0
    ):
        raise Qwen35MtpCarrySidecarError(
            "stored last_target_hidden shape is invalid"
        )
    if len(history) > 1:
        key = by_name["state.key"]
        value = by_name["state.value"]
        key_shape = key["shape_tuple"]
        if (
            len(key_shape) != 4
            or key_shape != value["shape_tuple"]
            or key_shape[0] != 1
            or key_shape[1] <= 0
            or key_shape[2] != len(history) - 1
            or key_shape[3] <= 0
            or key["torch_dtype"] != value["torch_dtype"]
            or key["torch_dtype"] != hidden["torch_dtype"]
        ):
            raise Qwen35MtpCarrySidecarError(
                "stored MTP key/value tensor contract is invalid"
            )
    return parsed


def _read_tensor(descriptor: int, row: Mapping[str, Any], file_hasher: Any) -> torch.Tensor:
    nbytes = int(row["nbytes"])
    raw = torch.empty(nbytes, dtype=torch.uint8, device="cpu")
    view = memoryview(raw.numpy())
    remaining = nbytes
    offset = 0
    while remaining:
        read = os.readv(descriptor, [view[offset : offset + remaining]])
        if read <= 0:
            raise Qwen35MtpCarrySidecarError("MTP tensor storage was truncated")
        chunk = view[offset : offset + read]
        file_hasher.update(chunk)
        offset += read
        remaining -= read
    if _sha256_bytes(view) != row["sha256"]:
        raise Qwen35MtpCarrySidecarError(
            f"MTP tensor {row['name']!r} SHA-256 mismatch"
        )
    tensor = raw.view(row["torch_dtype"]).reshape(row["shape_tuple"])
    if not bool(torch.isfinite(tensor).all().item()):
        raise Qwen35MtpCarrySidecarError(
            f"MTP tensor {row['name']!r} contains non-finite values"
        )
    return tensor


def _read_location(
    path: str | os.PathLike[str], descriptor: MtpCarrySidecarDescriptor
) -> tuple[Path, str]:
    candidate = _absolute(path)
    if candidate.name == descriptor.basename:
        return candidate.parent, candidate.name
    return candidate, descriptor.basename


def read_qwen35_mtp_carry_sidecar(
    path: str | os.PathLike[str],
    descriptor: MtpCarrySidecarDescriptor,
    *,
    history: Sequence[int],
    tokenizer_sha256: str,
    expected_identity: object,
) -> Qwen35MtpCarry:
    """Verify all metadata/storage, then reconstruct history from the caller."""

    receipt = _validated_descriptor(descriptor)
    tokens = _history_tuple(history)
    root_path, basename = _read_location(path, receipt)
    if basename != receipt.basename:
        raise Qwen35MtpCarrySidecarError("MTP carry sidecar path is unsafe")
    root_fd = _open_root(root_path, create=False)
    file_fd = -1
    try:
        file_fd, before = _open_relative_regular(root_fd, basename)
        if before.st_size != receipt.bytes or before.st_size > _MAX_SIDECAR_BYTES:
            raise Qwen35MtpCarrySidecarError("MTP carry sidecar size mismatch")
        if _hash_fd(file_fd, before.st_size) != receipt.file_sha256:
            raise Qwen35MtpCarrySidecarError("MTP carry sidecar SHA-256 mismatch")

        os.lseek(file_fd, 0, os.SEEK_SET)
        file_hasher = hashlib.sha256()
        magic = _read_exact(file_fd, len(_MAGIC), file_hasher)
        if magic != _MAGIC:
            raise Qwen35MtpCarrySidecarError("MTP carry sidecar magic is invalid")
        length_raw = _read_exact(file_fd, _HEADER_LENGTH.size, file_hasher)
        header_length = _HEADER_LENGTH.unpack(length_raw)[0]
        if not 0 < header_length <= _MAX_HEADER_BYTES:
            raise Qwen35MtpCarrySidecarError(
                "MTP carry sidecar header exceeds its bound"
            )
        if len(_MAGIC) + 4 + header_length >= receipt.bytes:
            raise Qwen35MtpCarrySidecarError(
                "MTP carry sidecar header length is inconsistent"
            )
        header_raw = _read_exact(file_fd, header_length, file_hasher)
        body = _parse_header(header_raw)
        rows = _validate_body(
            body,
            receipt,
            history=tokens,
            tokenizer_sha256=tokenizer_sha256,
            expected_identity=expected_identity,
        )
        expected_size = len(_MAGIC) + 4 + header_length + int(body["tensor_bytes"])
        if expected_size != receipt.bytes:
            raise Qwen35MtpCarrySidecarError(
                "MTP carry sidecar payload size is inconsistent"
            )

        tensors: dict[str, torch.Tensor] = {}
        payload_hasher = hashlib.sha256()
        for row in rows:
            tensor = _read_tensor(file_fd, row, file_hasher)
            payload_hasher.update(_tensor_bytes(tensor))
            tensors[row["name"]] = tensor
        if os.read(file_fd, 1):
            raise Qwen35MtpCarrySidecarError(
                "MTP carry sidecar contains trailing bytes"
            )
        if file_hasher.hexdigest() != receipt.file_sha256:
            raise Qwen35MtpCarrySidecarError(
                "MTP carry sidecar changed during materialization"
            )
        if payload_hasher.hexdigest() != body["payload_sha256"]:
            raise Qwen35MtpCarrySidecarError("MTP payload SHA-256 mismatch")

        after = os.fstat(file_fd)
        linked_after = os.stat(basename, dir_fd=root_fd, follow_symlinks=False)
        if (
            _stable_signature(before) != _stable_signature(after)
            or not _same_inode(after, linked_after)
        ):
            raise Qwen35MtpCarrySidecarError(
                "MTP carry sidecar changed while reading"
            )

        hidden = tensors["last_target_hidden"].clone().contiguous()
        state = None
        if len(tokens) > 1:
            state = AttentionState(
                key=tensors["state.key"].clone().contiguous(),
                value=tensors["state.value"].clone().contiguous(),
                crsa_log_usage=None,
            )
        return Qwen35MtpCarry(
            schema=QWEN35_MTP_CARRY_SCHEMA,
            identity=_copy_identity(expected_identity),  # type: ignore[arg-type]
            history=tokens,
            next_position=len(tokens) - 1,
            state=state,
            last_target_hidden=hidden,
        )
    except Qwen35MtpCarrySidecarError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise Qwen35MtpCarrySidecarError(
            "cannot restore MTP carry sidecar safely"
        ) from exc
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(root_fd)


# Short names are convenient for cache implementations while the explicit
# Qwen3.5 names remain the stable public API.
write_mtp_carry_sidecar = write_qwen35_mtp_carry_sidecar
read_mtp_carry_sidecar = read_qwen35_mtp_carry_sidecar
write_qwen35_mtp_carry_snapshot = write_qwen35_mtp_carry_sidecar
read_qwen35_mtp_carry_snapshot = read_qwen35_mtp_carry_sidecar
write_mtp_carry_snapshot = write_qwen35_mtp_carry_sidecar
read_mtp_carry_snapshot = read_qwen35_mtp_carry_sidecar
MtpCarrySnapshotDescriptor = MtpCarrySidecarDescriptor
Qwen35MtpCarrySidecarDescriptor = MtpCarrySidecarDescriptor
Qwen35MtpCarrySnapshotError = Qwen35MtpCarrySidecarError
Qwen35MtpCarrySnapshotIdentityMismatch = Qwen35MtpCarrySidecarIdentityMismatch


__all__ = [
    "MtpCarrySidecarDescriptor",
    "MtpCarrySnapshotDescriptor",
    "Qwen35MtpCarrySidecarDescriptor",
    "QWEN35_MTP_CARRY_SIDECAR_SCHEMA",
    "QWEN35_MTP_CARRY_SIDECAR_VERSION",
    "Qwen35MtpCarrySidecarError",
    "Qwen35MtpCarrySidecarIdentityMismatch",
    "Qwen35MtpCarrySnapshotError",
    "Qwen35MtpCarrySnapshotIdentityMismatch",
    "qwen35_mtp_carry_identity_sha256",
    "qwen35_mtp_prefix_sha256",
    "read_mtp_carry_sidecar",
    "read_mtp_carry_snapshot",
    "read_qwen35_mtp_carry_sidecar",
    "read_qwen35_mtp_carry_snapshot",
    "write_mtp_carry_sidecar",
    "write_mtp_carry_snapshot",
    "write_qwen35_mtp_carry_sidecar",
    "write_qwen35_mtp_carry_snapshot",
]
