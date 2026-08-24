"""Safe, versioned continuation snapshots for the native V4 decoder.

The format intentionally consists of a small JSON manifest and a content-
addressed NPZ payload.  It never uses pickle.  The payload is published before
the manifest, so replacing the manifest is the single atomic commit point.
Readers verify the manifest body, complete payload, every tensor, and all NPY
headers before allocating tensor storage.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile
from typing import Any, Mapping
import zipfile

import numpy as np
import torch


SNAPSHOT_SCHEMA = "immer.deepseek-v4-continuation/v1"
SNAPSHOT_VERSION = 1
MAX_NPY_HEADER_BYTES = 16 * 1024


class DeepSeekV4SnapshotError(ValueError):
    """A continuation snapshot is unsafe, corrupt, or incompatible."""


@dataclass(frozen=True, slots=True)
class SnapshotLimits:
    """Hard admission bounds applied before tensor allocation."""

    max_bytes: int = 2 * 1024**3
    max_tensors: int = 2048
    max_rank: int = 8
    max_dimension: int = 1_048_576
    max_manifest_bytes: int = 8 * 1024**2

    def __post_init__(self) -> None:
        for name in (
            "max_bytes",
            "max_tensors",
            "max_rank",
            "max_dimension",
            "max_manifest_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class SnapshotTensor:
    """One logical tensor and its accepted finite-value policy."""

    value: torch.Tensor
    finite_policy: str = "finite"

    def __post_init__(self) -> None:
        if not isinstance(self.value, torch.Tensor):
            raise TypeError("snapshot value must be a torch tensor")
        if self.finite_policy not in {"finite", "finite_or_neg_inf"}:
            raise ValueError("unsupported snapshot finite policy")


@dataclass(frozen=True, slots=True)
class LoadedSnapshot:
    manifest: dict[str, Any]
    state: dict[str, Any]
    tensors: dict[str, torch.Tensor]
    summary: dict[str, Any]


_TORCH_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DeepSeekV4SnapshotError(
            "snapshot metadata is not canonical JSON"
        ) from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_stream(stream: Any) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


def _sha256_array(value: np.ndarray) -> str:
    if not value.flags.c_contiguous:
        raise DeepSeekV4SnapshotError("snapshot tensor storage is not contiguous")
    digest = hashlib.sha256()
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _absolute_path(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _open_regular_read(path: Path, label: str) -> Any:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise DeepSeekV4SnapshotError(f"cannot open snapshot {label}") from exc
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise DeepSeekV4SnapshotError(f"snapshot {label} must be a regular file")
        return os.fdopen(fd, "rb")
    except Exception:
        os.close(fd)
        raise


def _reject_nonregular_existing(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise DeepSeekV4SnapshotError(f"cannot inspect snapshot {label}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise DeepSeekV4SnapshotError(f"snapshot {label} must be a regular file")


def _json_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DeepSeekV4SnapshotError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _snapshot_schema(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 256
        or "\x00" in value
    ):
        raise DeepSeekV4SnapshotError("snapshot schema identity is invalid")
    return value


def _read_manifest(
    path: Path, limits: SnapshotLimits, *, schema: str
) -> dict[str, Any]:
    try:
        with _open_regular_read(path, "manifest") as stream:
            size = os.fstat(stream.fileno()).st_size
            if size <= 0 or size > limits.max_manifest_bytes:
                raise DeepSeekV4SnapshotError(
                    f"snapshot manifest size {size} exceeds limit "
                    f"{limits.max_manifest_bytes}"
                )
            raw = stream.read(limits.max_manifest_bytes + 1)
        if len(raw) != size:
            raise DeepSeekV4SnapshotError("snapshot manifest changed while reading")
        document = json.loads(
            raw,
            object_pairs_hook=_json_no_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                DeepSeekV4SnapshotError(f"invalid JSON constant: {value}")
            ),
        )
    except DeepSeekV4SnapshotError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DeepSeekV4SnapshotError("cannot decode snapshot manifest") from exc
    if not isinstance(document, dict):
        raise DeepSeekV4SnapshotError("snapshot manifest root must be an object")
    if document.get("version") != SNAPSHOT_VERSION:
        raise DeepSeekV4SnapshotError("unsupported snapshot version")
    body = document.get("body")
    if not isinstance(body, dict):
        raise DeepSeekV4SnapshotError("snapshot manifest body must be an object")
    expected = document.get("body_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise DeepSeekV4SnapshotError("snapshot manifest body hash is invalid")
    actual = _sha256_bytes(_canonical_json(body))
    if actual != expected:
        raise DeepSeekV4SnapshotError("snapshot manifest body SHA-256 mismatch")
    # Schema is duplicated inside the authenticated body.  Legacy DeepSeek-V4
    # manifests predate that field and remain readable only under their
    # original schema; every newly written or non-DeepSeek schema is bound by
    # the body digest and cannot be relabelled by editing the outer document.
    body_schema = body.get("schema")
    if body_schema is None:
        if schema != SNAPSHOT_SCHEMA or document.get("schema") != SNAPSHOT_SCHEMA:
            raise DeepSeekV4SnapshotError("unsupported snapshot schema")
    elif body_schema != schema or document.get("schema") != schema:
        raise DeepSeekV4SnapshotError("unsupported snapshot schema")
    return document


def _dtype_name(value: torch.dtype) -> str:
    name = str(value).removeprefix("torch.")
    if name not in _TORCH_DTYPES:
        raise DeepSeekV4SnapshotError(f"snapshot dtype {name!r} is not allowed")
    return name


def _shape_and_nbytes(
    shape_value: Any,
    dtype_name: Any,
    limits: SnapshotLimits,
) -> tuple[tuple[int, ...], torch.dtype, int]:
    if not isinstance(dtype_name, str) or dtype_name not in _TORCH_DTYPES:
        raise DeepSeekV4SnapshotError("snapshot tensor dtype is not allowed")
    if not isinstance(shape_value, list) or len(shape_value) > limits.max_rank:
        raise DeepSeekV4SnapshotError("snapshot tensor rank exceeds its limit")
    shape: list[int] = []
    elements = 1
    for raw in shape_value:
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise DeepSeekV4SnapshotError("snapshot tensor shape is invalid")
        if raw > limits.max_dimension:
            raise DeepSeekV4SnapshotError("snapshot tensor dimension exceeds its limit")
        shape.append(raw)
        elements *= raw
        if elements > limits.max_bytes:
            raise DeepSeekV4SnapshotError("snapshot tensor element count is excessive")
    dtype = _TORCH_DTYPES[dtype_name]
    nbytes = elements * torch.empty((), dtype=dtype).element_size()
    if nbytes > limits.max_bytes:
        raise DeepSeekV4SnapshotError("snapshot tensor exceeds the byte limit")
    return tuple(shape), dtype, nbytes


def _validate_values(tensor: torch.Tensor, policy: str, name: str) -> None:
    if policy == "finite":
        valid = bool(torch.isfinite(tensor).all().item())
    elif policy == "finite_or_neg_inf":
        valid = bool((~torch.isnan(tensor) & ~torch.isposinf(tensor)).all().item())
    else:
        raise DeepSeekV4SnapshotError(f"tensor {name!r} has unsupported finite policy")
    if not valid:
        raise DeepSeekV4SnapshotError(
            f"tensor {name!r} violates finite policy {policy!r}"
        )


def _tensor_raw_bytes(tensor: torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
    cpu = tensor.detach().to(device="cpu").contiguous()
    raw = cpu.view(torch.uint8).numpy().reshape(-1).copy()
    return cpu, raw


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _temporary_file(parent: Path, prefix: str) -> tuple[int, Path]:
    fd, raw_path = tempfile.mkstemp(prefix=prefix, suffix=".pending", dir=parent)
    return fd, Path(raw_path)


def write_snapshot(
    path: str | os.PathLike[str],
    *,
    identity: Mapping[str, Any],
    state: Mapping[str, Any],
    tensors: Mapping[str, SnapshotTensor],
    limits: SnapshotLimits | None = None,
    schema: str = SNAPSHOT_SCHEMA,
) -> dict[str, Any]:
    """Atomically publish a verified JSON + NPZ continuation snapshot."""

    active_limits = SnapshotLimits() if limits is None else limits
    active_schema = _snapshot_schema(schema)
    target = _absolute_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_nonregular_existing(target, "manifest")
    if len(tensors) > active_limits.max_tensors:
        raise DeepSeekV4SnapshotError("snapshot tensor count exceeds its limit")

    arrays: dict[str, np.ndarray] = {}
    descriptors: list[dict[str, Any]] = []
    total_bytes = 0
    for index, name in enumerate(sorted(tensors)):
        if not isinstance(name, str) or not name or len(name) > 512:
            raise DeepSeekV4SnapshotError("snapshot tensor name is invalid")
        payload = tensors[name]
        if not isinstance(payload, SnapshotTensor):
            raise DeepSeekV4SnapshotError("snapshot tensor record is invalid")
        dtype_name = _dtype_name(payload.value.dtype)
        shape = tuple(int(value) for value in payload.value.shape)
        _, _, nbytes = _shape_and_nbytes(list(shape), dtype_name, active_limits)
        if nbytes != payload.value.numel() * payload.value.element_size():
            raise DeepSeekV4SnapshotError("snapshot tensor byte count overflow")
        total_bytes += nbytes
        if total_bytes > active_limits.max_bytes:
            raise DeepSeekV4SnapshotError(
                "snapshot tensors exceed the total byte limit"
            )
        cpu, raw = _tensor_raw_bytes(payload.value)
        _validate_values(cpu, payload.finite_policy, name)
        storage = f"t{index:06d}"
        arrays[storage] = raw
        descriptors.append(
            {
                "name": name,
                "storage": storage,
                "dtype": dtype_name,
                "shape": list(shape),
                "nbytes": nbytes,
                "sha256": _sha256_array(raw),
                "finite_policy": payload.finite_policy,
            }
        )

    body = {
        "byte_order": sys.byteorder,
        "identity": dict(identity),
        "schema": active_schema,
        "state": dict(state),
        "tensors": descriptors,
        "tensor_count": len(descriptors),
        "tensor_bytes": total_bytes,
        "largest_tensor_bytes": max(
            (int(row["nbytes"]) for row in descriptors), default=0
        ),
    }
    body_bytes = _canonical_json(body)
    body_sha = _sha256_bytes(body_bytes)

    payload_fd, payload_temp = _temporary_file(target.parent, f".{target.name}.npz.")
    manifest_fd = -1
    manifest_temp: Path | None = None
    try:
        with os.fdopen(payload_fd, "w+b") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
            payload_size = stream.tell()
            if (
                payload_size
                > active_limits.max_bytes + active_limits.max_manifest_bytes
            ):
                raise DeepSeekV4SnapshotError("NPZ payload exceeds the file-size limit")
            stream.seek(0)
            payload_sha = _sha256_stream(stream)
        payload_name = f"{target.stem}.{payload_sha}.npz"
        payload_target = target.parent / payload_name
        try:
            payload_target.lstat()
        except FileNotFoundError:
            payload_exists = False
        except OSError as exc:
            raise DeepSeekV4SnapshotError(
                "cannot inspect existing snapshot payload"
            ) from exc
        else:
            payload_exists = True
        if payload_exists:
            _reject_nonregular_existing(payload_target, "payload")
            with _open_regular_read(payload_target, "payload") as stream:
                existing_sha = _sha256_stream(stream)
            if existing_sha != payload_sha:
                raise DeepSeekV4SnapshotError("content-addressed payload collision")
            payload_temp.unlink()
        else:
            os.replace(payload_temp, payload_target)
            _fsync_directory(target.parent)

        body["payload"] = {
            "file": payload_name,
            "bytes": payload_size,
            "sha256": payload_sha,
        }
        # The payload reference is part of the authenticated body.
        body_bytes = _canonical_json(body)
        body_sha = _sha256_bytes(body_bytes)
        document = {
            "schema": active_schema,
            "version": SNAPSHOT_VERSION,
            "body_sha256": body_sha,
            "body": body,
        }
        manifest_bytes = _canonical_json(document) + b"\n"
        if len(manifest_bytes) > active_limits.max_manifest_bytes:
            raise DeepSeekV4SnapshotError("snapshot manifest exceeds its byte limit")
        manifest_fd, manifest_temp = _temporary_file(
            target.parent, f".{target.name}.manifest."
        )
        with os.fdopen(manifest_fd, "wb") as stream:
            manifest_fd = -1
            stream.write(manifest_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(manifest_temp, target)
        manifest_temp = None
        _fsync_directory(target.parent)
    except Exception:
        if payload_temp.exists():
            payload_temp.unlink()
        if manifest_fd >= 0:
            os.close(manifest_fd)
        if manifest_temp is not None and manifest_temp.exists():
            manifest_temp.unlink()
        raise

    return {
        "schema": active_schema,
        "version": SNAPSHOT_VERSION,
        "manifest": str(target),
        "manifest_body_sha256": body_sha,
        "payload": str(payload_target),
        "payload_sha256": payload_sha,
        "payload_bytes": payload_size,
        "tensor_count": len(descriptors),
        "tensor_bytes": total_bytes,
        "largest_tensor_bytes": body["largest_tensor_bytes"],
    }


def _parse_descriptors(
    body: Mapping[str, Any], limits: SnapshotLimits
) -> tuple[list[dict[str, Any]], int]:
    raw = body.get("tensors")
    if not isinstance(raw, list) or len(raw) > limits.max_tensors:
        raise DeepSeekV4SnapshotError("snapshot tensor table exceeds its limit")
    if body.get("tensor_count") != len(raw):
        raise DeepSeekV4SnapshotError("snapshot tensor count is inconsistent")
    names: set[str] = set()
    storage_names: set[str] = set()
    descriptors: list[dict[str, Any]] = []
    total = 0
    for descriptor in raw:
        if not isinstance(descriptor, dict):
            raise DeepSeekV4SnapshotError("snapshot tensor descriptor is invalid")
        name = descriptor.get("name")
        storage = descriptor.get("storage")
        if not isinstance(name, str) or not name or len(name) > 512 or name in names:
            raise DeepSeekV4SnapshotError(
                "snapshot tensor name is invalid or duplicate"
            )
        if (
            not isinstance(storage, str)
            or not storage.startswith("t")
            or not storage[1:].isdigit()
            or storage in storage_names
        ):
            raise DeepSeekV4SnapshotError(
                "snapshot storage name is invalid or duplicate"
            )
        shape, dtype, nbytes = _shape_and_nbytes(
            descriptor.get("shape"), descriptor.get("dtype"), limits
        )
        if descriptor.get("nbytes") != nbytes:
            raise DeepSeekV4SnapshotError("snapshot tensor byte count is inconsistent")
        digest = descriptor.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise DeepSeekV4SnapshotError("snapshot tensor SHA-256 is invalid")
        policy = descriptor.get("finite_policy")
        if policy not in {"finite", "finite_or_neg_inf"}:
            raise DeepSeekV4SnapshotError("snapshot finite policy is invalid")
        total += nbytes
        if total > limits.max_bytes:
            raise DeepSeekV4SnapshotError(
                "snapshot tensors exceed the total byte limit"
            )
        names.add(name)
        storage_names.add(storage)
        descriptors.append(
            {
                **descriptor,
                "shape_tuple": shape,
                "torch_dtype": dtype,
            }
        )
    if body.get("tensor_bytes") != total:
        raise DeepSeekV4SnapshotError("snapshot total tensor bytes are inconsistent")
    largest = max((int(row["nbytes"]) for row in descriptors), default=0)
    if body.get("largest_tensor_bytes") != largest:
        raise DeepSeekV4SnapshotError(
            "snapshot largest tensor byte count is inconsistent"
        )
    return descriptors, total


def _read_npy_header(stream: Any) -> tuple[tuple[int, ...], bool, np.dtype[Any], int]:
    try:
        version = np.lib.format.read_magic(stream)
        length_format = "<H" if version == (1, 0) else "<I"
        length_bytes = stream.read(struct.calcsize(length_format))
        if len(length_bytes) != struct.calcsize(length_format):
            raise DeepSeekV4SnapshotError("truncated NPY header length")
        header_length = struct.unpack(length_format, length_bytes)[0]
        if header_length > MAX_NPY_HEADER_BYTES:
            raise DeepSeekV4SnapshotError(
                f"NPY header length {header_length} exceeds "
                f"{MAX_NPY_HEADER_BYTES} bytes"
            )
        header = stream.read(header_length)
        if len(header) != header_length:
            raise DeepSeekV4SnapshotError("truncated NPY header")
        bounded_header = io.BytesIO(length_bytes + header)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(
                bounded_header, max_header_size=MAX_NPY_HEADER_BYTES
            )
        elif version in {(2, 0), (3, 0)}:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(
                bounded_header, max_header_size=MAX_NPY_HEADER_BYTES
            )
        else:
            raise DeepSeekV4SnapshotError("unsupported NPY header version")
    except DeepSeekV4SnapshotError:
        raise
    except Exception as exc:
        raise DeepSeekV4SnapshotError("cannot decode NPY header") from exc
    return (
        tuple(int(value) for value in shape),
        bool(fortran),
        np.dtype(dtype),
        stream.tell(),
    )


def _validate_zip_headers(
    archive: zipfile.ZipFile,
    descriptors: list[dict[str, Any]],
) -> None:
    expected = {f"{row['storage']}.npy": row for row in descriptors}
    members = archive.infolist()
    if len(members) != len(expected) or {row.filename for row in members} != set(
        expected
    ):
        raise DeepSeekV4SnapshotError("NPZ members do not match the manifest")
    for member in members:
        descriptor = expected[member.filename]
        if member.flag_bits & 0x1:
            raise DeepSeekV4SnapshotError("encrypted NPZ members are not allowed")
        if member.compress_type != zipfile.ZIP_STORED:
            raise DeepSeekV4SnapshotError("compressed NPZ members are not allowed")
        with archive.open(member, "r") as stream:
            shape, fortran, dtype, header_bytes = _read_npy_header(stream)
        expected_bytes = int(descriptor["nbytes"])
        if shape != (expected_bytes,) or fortran or dtype != np.dtype(np.uint8):
            raise DeepSeekV4SnapshotError("NPY storage header violates the manifest")
        if member.file_size != header_bytes + expected_bytes:
            raise DeepSeekV4SnapshotError("NPY storage length is inconsistent")


def _validate_zip_directory_bounds(stream: Any, expected_members: int) -> None:
    """Bound central-directory allocation before ``zipfile`` parses it."""

    stream.seek(0, os.SEEK_END)
    payload_size = stream.tell()
    tail_size = min(payload_size, 65_557)
    stream.seek(payload_size - tail_size)
    tail = stream.read(tail_size)
    signature = b"PK\x05\x06"
    offset = tail.rfind(signature)
    if offset < 0 or len(tail) - offset < 22:
        raise DeepSeekV4SnapshotError("NPZ end-of-central-directory is missing")
    try:
        (
            _signature,
            disk,
            central_disk,
            entries_on_disk,
            entries_total,
            directory_bytes,
            directory_offset,
            comment_bytes,
        ) = struct.unpack("<4s4H2LH", tail[offset : offset + 22])
    except struct.error as exc:
        raise DeepSeekV4SnapshotError("NPZ central directory is malformed") from exc
    eocd_offset = payload_size - tail_size + offset
    if eocd_offset + 22 + comment_bytes != payload_size:
        raise DeepSeekV4SnapshotError("NPZ has trailing or truncated directory data")
    if disk or central_disk or entries_on_disk != entries_total:
        raise DeepSeekV4SnapshotError("multi-disk NPZ payloads are not allowed")
    if entries_total == 0xFFFF or directory_bytes == 0xFFFFFFFF:
        raise DeepSeekV4SnapshotError("ZIP64 NPZ payloads are not allowed")
    if entries_total != expected_members:
        raise DeepSeekV4SnapshotError("NPZ member count does not match the manifest")
    max_directory_bytes = 1024 + expected_members * 1024
    if directory_bytes > max_directory_bytes:
        raise DeepSeekV4SnapshotError("NPZ central directory exceeds its hard bound")
    if directory_offset + directory_bytes != eocd_offset:
        raise DeepSeekV4SnapshotError("NPZ central-directory offsets are inconsistent")
    stream.seek(0)


def read_snapshot(
    path: str | os.PathLike[str],
    *,
    expected_identity: Mapping[str, Any],
    limits: SnapshotLimits | None = None,
    resident_bytes: int = 0,
    max_restore_peak_bytes: int | None = None,
    schema: str = SNAPSHOT_SCHEMA,
) -> LoadedSnapshot:
    """Verify and load a continuation snapshot without pickle."""

    active_limits = SnapshotLimits() if limits is None else limits
    active_schema = _snapshot_schema(schema)
    if (
        isinstance(resident_bytes, bool)
        or not isinstance(resident_bytes, int)
        or resident_bytes < 0
    ):
        raise ValueError("resident_bytes must be a non-negative integer")
    if max_restore_peak_bytes is not None and (
        isinstance(max_restore_peak_bytes, bool)
        or not isinstance(max_restore_peak_bytes, int)
        or max_restore_peak_bytes <= 0
    ):
        raise ValueError("max_restore_peak_bytes must be a positive integer")
    target = _absolute_path(path)
    document = _read_manifest(target, active_limits, schema=active_schema)
    body = document["body"]
    if body.get("byte_order") != sys.byteorder:
        raise DeepSeekV4SnapshotError("snapshot byte order does not match this runtime")
    identity = body.get("identity")
    if not isinstance(identity, dict) or _canonical_json(identity) != _canonical_json(
        dict(expected_identity)
    ):
        raise DeepSeekV4SnapshotError("snapshot model/source/runtime identity mismatch")
    state = body.get("state")
    if not isinstance(state, dict):
        raise DeepSeekV4SnapshotError("snapshot state must be an object")
    next_position = state.get("next_position")
    if (
        isinstance(next_position, bool)
        or not isinstance(next_position, int)
        or next_position < 0
    ):
        raise DeepSeekV4SnapshotError("snapshot next_position is invalid")
    descriptors, total_bytes = _parse_descriptors(body, active_limits)
    if next_position == 0 and (
        descriptors
        or state.get("attention_layers") not in (None, [])
        or state.get("graft_history") is not None
    ):
        raise DeepSeekV4SnapshotError(
            "zero-cursor snapshot must not contain continuation tensors"
        )
    largest_tensor_bytes = max((int(row["nbytes"]) for row in descriptors), default=0)
    estimated_restore_peak_bytes = resident_bytes + total_bytes + largest_tensor_bytes
    if (
        max_restore_peak_bytes is not None
        and estimated_restore_peak_bytes > max_restore_peak_bytes
    ):
        raise DeepSeekV4SnapshotError(
            "snapshot restore peak exceeds max_restore_peak_bytes: "
            f"{estimated_restore_peak_bytes} > {max_restore_peak_bytes}"
        )
    payload = body.get("payload")
    if not isinstance(payload, dict):
        raise DeepSeekV4SnapshotError("snapshot payload descriptor is missing")
    payload_name = payload.get("file")
    payload_bytes = payload.get("bytes")
    payload_sha = payload.get("sha256")
    if (
        not isinstance(payload_name, str)
        or not payload_name
        or Path(payload_name).name != payload_name
        or not payload_name.endswith(".npz")
    ):
        raise DeepSeekV4SnapshotError("snapshot payload path is unsafe")
    if (
        isinstance(payload_bytes, bool)
        or not isinstance(payload_bytes, int)
        or payload_bytes <= 0
        or payload_bytes > active_limits.max_bytes + active_limits.max_manifest_bytes
    ):
        raise DeepSeekV4SnapshotError("snapshot payload size exceeds its limit")
    if not isinstance(payload_sha, str) or len(payload_sha) != 64:
        raise DeepSeekV4SnapshotError("snapshot payload SHA-256 is invalid")
    payload_path = target.parent / payload_name
    stream = _open_regular_read(payload_path, "payload")
    tensors: dict[str, torch.Tensor] = {}
    with stream:
        try:
            stat = os.fstat(stream.fileno())
        except OSError as exc:
            raise DeepSeekV4SnapshotError("cannot stat open snapshot payload") from exc
        if stat.st_size != payload_bytes:
            raise DeepSeekV4SnapshotError("snapshot payload size mismatch")
        actual_payload_sha = _sha256_stream(stream)
        if actual_payload_sha != payload_sha:
            raise DeepSeekV4SnapshotError("snapshot payload SHA-256 mismatch")
        stream.seek(0)
        try:
            _validate_zip_directory_bounds(stream, len(descriptors))
            with zipfile.ZipFile(stream, "r") as archive:
                _validate_zip_headers(archive, descriptors)
                bad = archive.testzip()
                if bad is not None:
                    raise DeepSeekV4SnapshotError(f"NPZ CRC mismatch in {bad!r}")
        except DeepSeekV4SnapshotError:
            raise
        except (OSError, zipfile.BadZipFile) as exc:
            raise DeepSeekV4SnapshotError(
                "snapshot payload is not a valid NPZ"
            ) from exc
        stream.seek(0)
        try:
            with np.load(stream, allow_pickle=False) as archive:
                for descriptor in descriptors:
                    storage = descriptor["storage"]
                    raw = archive[storage]
                    nbytes = int(descriptor["nbytes"])
                    if (
                        raw.dtype != np.uint8
                        or tuple(raw.shape) != (nbytes,)
                        or not raw.flags.c_contiguous
                    ):
                        raise DeepSeekV4SnapshotError(
                            "loaded NPZ tensor is inconsistent"
                        )
                    if _sha256_array(raw) != descriptor["sha256"]:
                        raise DeepSeekV4SnapshotError(
                            f"tensor {descriptor['name']!r} SHA-256 mismatch"
                        )
                    base = torch.from_numpy(raw)
                    tensor = base.view(descriptor["torch_dtype"]).reshape(
                        descriptor["shape_tuple"]
                    )
                    _validate_values(
                        tensor,
                        descriptor["finite_policy"],
                        descriptor["name"],
                    )
                    tensors[descriptor["name"]] = tensor
        except DeepSeekV4SnapshotError:
            raise
        except Exception as exc:
            raise DeepSeekV4SnapshotError(
                "cannot materialize snapshot tensors"
            ) from exc

    summary = {
        "schema": active_schema,
        "version": SNAPSHOT_VERSION,
        "manifest": str(target),
        "manifest_body_sha256": document["body_sha256"],
        "payload": str(payload_path),
        "payload_sha256": payload_sha,
        "payload_bytes": payload_bytes,
        "tensor_count": len(descriptors),
        "tensor_bytes": total_bytes,
        "largest_tensor_bytes": largest_tensor_bytes,
        "resident_bytes_before_restore": resident_bytes,
        "estimated_restore_peak_bytes": estimated_restore_peak_bytes,
    }
    return LoadedSnapshot(document, state, tensors, summary)


__all__ = [
    "DeepSeekV4SnapshotError",
    "LoadedSnapshot",
    "SNAPSHOT_SCHEMA",
    "SNAPSHOT_VERSION",
    "SnapshotLimits",
    "SnapshotTensor",
    "read_snapshot",
    "write_snapshot",
]
