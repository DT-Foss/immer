"""Exact out-of-core layer-major scoring for streamed DeepSeek-V4.

The execution order changes, the model equation does not::

    H[i, layer + 1] = F[layer](H[i, layer])

Every activation generation is a set of BF16 safetensors objects published by
one atomic JSON manifest.  The manifest is the sole commit point, so a crash can
leave at most unreferenced content-addressed objects; it cannot expose a mixed
generation.  No pickle or executable checkpoint format is accepted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .graft import DeepSeekV4CrsaGraft, GRAFT_MODES
from .model import StreamedDeepSeekV4
from .provenance import runtime_dependency_versions, runtime_source_manifest


LAYERWISE_SCHEMA = "immer.deepseek-v4-layerwise/v4"
LAYERWISE_VERSION = 4
_ACTIVATION_SCHEMA = "immer.deepseek-v4-layerwise-activation/v4"
_MANIFEST_KIND = "manifest"
_RESULT_KIND = "result"
OFFICIAL_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
OFFICIAL_SOURCE_SAFE_BYTES = 160 * 1024**3
MAX_MANIFEST_BYTES = 16 * 1024**2
_DIGEST = re.compile(r"[0-9a-f]{64}")


class LayerwiseError(RuntimeError):
    """A layer-major run is unsafe, corrupt, or incompatible."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LayerwiseError("layerwise metadata is not canonical JSON") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_source_manifest() -> list[dict[str, str]]:
    """Fingerprint every local source file that can change layer execution."""

    return runtime_source_manifest(
        extra_files=("layerwise.py",),
        project_files=("scripts/deepseek_v4_layerwise.py",),
    )


def _runtime_dependency_versions() -> dict[str, str]:
    """Fingerprint the interpreter and binary package environment."""

    return runtime_dependency_versions()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _regular_file(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise LayerwiseError(f"cannot inspect {label}: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise LayerwiseError(f"{label} must be a regular file: {path}")
    return metadata


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    body = _canonical_json(document) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        _regular_file(path, "layerwise manifest")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".pending", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _read_json(path: Path, *, label: str = "layerwise manifest") -> dict[str, Any]:
    metadata = _regular_file(path, label)
    if metadata.st_size <= 0 or metadata.st_size > MAX_MANIFEST_BYTES:
        raise LayerwiseError(f"{label} size is outside its bound")
    try:

        def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, child in pairs:
                if key in result:
                    raise LayerwiseError(f"duplicate JSON key: {key!r}")
                result[key] = child
            return result

        value = json.loads(path.read_bytes(), object_pairs_hook=no_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LayerwiseError(f"cannot decode {label}") from exc
    if not isinstance(value, dict):
        raise LayerwiseError(f"{label} root must be an object")
    return value


@dataclass(frozen=True, slots=True)
class LayerwiseItem:
    """One fixed-prompt candidate-scoring item."""

    item_id: str
    prompt_token_ids: tuple[int, ...]
    candidate_token_ids: tuple[int, ...]
    candidate_values: tuple[Any, ...] = ()
    expected: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.item_id, str) or not self.item_id.strip():
            raise ValueError("item_id must be a non-empty string")
        if not self.prompt_token_ids or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.prompt_token_ids
        ):
            raise ValueError("prompt_token_ids must be non-empty non-negative integers")
        if len(self.candidate_token_ids) < 2 or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.candidate_token_ids
        ):
            raise ValueError("candidate_token_ids must contain at least two IDs")
        if len(set(self.candidate_token_ids)) != len(self.candidate_token_ids):
            raise ValueError("candidate_token_ids must be distinct within an item")
        if self.candidate_values and len(self.candidate_values) != len(
            self.candidate_token_ids
        ):
            raise ValueError("candidate_values must align with candidate_token_ids")
        _canonical_json(self.expected)
        _canonical_json(list(self.candidate_values))

    def identity_record(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "prompt_token_ids": list(self.prompt_token_ids),
            "candidate_token_ids": list(self.candidate_token_ids),
            "candidate_values": list(self.candidate_values),
            "expected": self.expected,
        }


@dataclass(frozen=True, slots=True)
class LayerwiseBucket:
    index: int
    item_indices: tuple[int, ...]
    sequence_length: int
    valid_lengths: tuple[int, ...]

    @property
    def batch_size(self) -> int:
        return len(self.item_indices)


@dataclass(frozen=True, slots=True)
class LayerwisePlan:
    items: int
    modes: tuple[str, ...]
    buckets: tuple[LayerwiseBucket, ...]
    microbatch_size: int
    padding: str
    graft_layer: int
    activation_generation_bytes_shared: int
    activation_generation_bytes_branched: int
    activation_transaction_peak_bytes: int
    activation_object_max_bytes: int
    official_source_safe_bytes: int
    inventory_payload_bytes: int
    source_storage_admission_bytes: int
    source_cache_reserve_bytes: int
    source_cache_reuse_required_bytes: int
    source_range_cache_enabled: bool
    source_range_cache_limit_bytes: int | None
    layer_forward_passes_max: int
    source_cache_reuse_admitted: bool
    free_disk_bytes: int
    required_free_disk_bytes: int
    require_full_source_disk: bool
    admitted: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["buckets"] = [asdict(bucket) for bucket in self.buckets]
        return value


def _buckets(
    items: Sequence[LayerwiseItem], microbatch_size: int, padding: str
) -> tuple[LayerwiseBucket, ...]:
    ordered = sorted(
        range(len(items)),
        key=lambda index: (len(items[index].prompt_token_ids), items[index].item_id),
    )
    groups: list[list[int]] = []
    if padding == "right":
        groups = [
            ordered[start : start + microbatch_size]
            for start in range(0, len(ordered), microbatch_size)
        ]
    elif padding == "exact-length":
        by_length: dict[int, list[int]] = {}
        for index in ordered:
            by_length.setdefault(len(items[index].prompt_token_ids), []).append(index)
        for length in sorted(by_length):
            indices = by_length[length]
            groups.extend(
                indices[start : start + microbatch_size]
                for start in range(0, len(indices), microbatch_size)
            )
    else:
        raise ValueError("padding must be 'right' or 'exact-length'")
    return tuple(
        LayerwiseBucket(
            index=bucket_index,
            item_indices=tuple(group),
            sequence_length=max(len(items[index].prompt_token_ids) for index in group),
            valid_lengths=tuple(len(items[index].prompt_token_ids) for index in group),
        )
        for bucket_index, group in enumerate(groups)
    )


def _inventory_payload_bytes(model: StreamedDeepSeekV4) -> int:
    inventory = model.pager.source.inventory()
    tensors = inventory.get("tensors", [])
    intervals: set[tuple[str, int, int]] = set()
    for row in tensors:
        try:
            begin, end = (int(value) for value in row["offset_in_shard"])
            shard = str(row.get("shard", "fixture"))
        except (KeyError, TypeError, ValueError) as exc:
            raise LayerwiseError(
                "source inventory has an invalid tensor interval"
            ) from exc
        if begin < 0 or end < begin:
            raise LayerwiseError("source inventory has an invalid tensor interval")
        if end == begin:
            continue
        intervals.add((shard, begin, end))
    return sum(end - begin for _shard, begin, end in intervals)


def _max_layer_payload_bytes(model: StreamedDeepSeekV4) -> int:
    """Return a conservative encoded-byte bound for one decoder layer.

    Layer-major execution only avoids fetching a layer's ranges again for the
    next bucket/variant when the verified source range cache can retain the
    current layer.  Summing tensor payloads (rather than assuming any specific
    shard adjacency) is a safe cache-capacity admission bound.
    """

    totals: dict[int, int] = {}
    for row in model.pager.source.inventory().get("tensors", []):
        if not isinstance(row, Mapping):
            raise LayerwiseError("source inventory has an invalid tensor row")
        name = row.get("name")
        if not isinstance(name, str):
            raise LayerwiseError("source inventory tensor name is invalid")
        match = re.match(r"layers\.(\d+)\.", name)
        if match is None:
            continue
        layer = int(match.group(1))
        if not 0 <= layer < model.config.n_layers:
            raise LayerwiseError("source inventory has a layer outside decoder depth")
        try:
            offsets = row["offset_in_shard"]
            if not isinstance(offsets, list) or len(offsets) != 2:
                raise TypeError
            begin, end = offsets
            if (
                isinstance(begin, bool)
                or not isinstance(begin, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
                or begin < 0
                or end < begin
            ):
                raise TypeError
        except (KeyError, TypeError) as exc:
            raise LayerwiseError(
                "source inventory has an invalid tensor interval"
            ) from exc
        totals[layer] = totals.get(layer, 0) + end - begin
    return max(totals.values(), default=0)


def _source_range_cache_contract(
    model: StreamedDeepSeekV4,
) -> tuple[bool, int | None]:
    """Conservatively expose the source cache contract used by the plan."""

    source = model.pager.source
    metrics = source.metrics()
    raw_limit = metrics.get("cache_limit_bytes")
    if raw_limit is not None and (
        isinstance(raw_limit, bool) or not isinstance(raw_limit, int) or raw_limit < 0
    ):
        raise LayerwiseError("source range cache limit is invalid")
    # Streamer exposes the authoritative cache directory on its contract
    # reader.  Unknown fixture/source types are treated as uncached rather than
    # claiming remote single-fetch behavior without evidence.
    reader = getattr(source, "reader", None)
    enabled = getattr(reader, "cache_dir", None) is not None
    return enabled, raw_limit


def build_layerwise_plan(
    model: StreamedDeepSeekV4,
    items: Sequence[LayerwiseItem],
    *,
    modes: Sequence[str] = ("off", "crsa", "softmax", "shuffle"),
    microbatch_size: int = 32,
    padding: str = "right",
    graft_layer: int | None = None,
    run_dir: str | os.PathLike[str],
    require_full_source_disk: bool = False,
    source_cache_reserve_bytes: int = 0,
    disk_margin_bytes: int = 1024**3,
) -> LayerwisePlan:
    """Calculate all disk and activation bounds without writing any file."""

    if model.pager.compute_dtype is not torch.bfloat16:
        raise LayerwiseError(
            "layerwise activation checkpoints require bfloat16 compute"
        )
    if not items:
        raise ValueError("layerwise scoring requires at least one item")
    if (
        isinstance(microbatch_size, bool)
        or not isinstance(microbatch_size, int)
        or microbatch_size <= 0
    ):
        raise ValueError("microbatch_size must be a positive integer")
    if (
        isinstance(disk_margin_bytes, bool)
        or not isinstance(disk_margin_bytes, int)
        or disk_margin_bytes < 0
    ):
        raise ValueError("disk_margin_bytes must be a non-negative integer")
    if (
        isinstance(source_cache_reserve_bytes, bool)
        or not isinstance(source_cache_reserve_bytes, int)
        or source_cache_reserve_bytes < 0
    ):
        raise ValueError("source_cache_reserve_bytes must be a non-negative integer")
    normalized_modes = tuple(str(mode).lower() for mode in modes)
    if not normalized_modes or len(set(normalized_modes)) != len(normalized_modes):
        raise ValueError("modes must be non-empty and distinct")
    if any(mode not in GRAFT_MODES for mode in normalized_modes):
        raise ValueError("modes contain an unsupported graft mode")
    layer = model.config.n_layers // 2 if graft_layer is None else graft_layer
    if (
        isinstance(layer, bool)
        or not isinstance(layer, int)
        or not 0 <= layer < model.config.n_layers
    ):
        raise ValueError("graft_layer outside decoder depth")
    if any(
        token >= model.config.vocab_size
        for item in items
        for token in (*item.prompt_token_ids, *item.candidate_token_ids)
    ):
        raise ValueError("item token ID outside checkpoint vocabulary")
    if max(len(item.prompt_token_ids) for item in items) > model.max_seq_len:
        raise ValueError("an item exceeds the model context bound")
    if microbatch_size > model.max_batch_size:
        raise ValueError("microbatch_size exceeds model max_batch_size")
    bucket_rows = _buckets(items, microbatch_size, padding)
    element_bytes = torch.empty((), dtype=torch.bfloat16).element_size()
    widths = model.config.hc_mult * model.config.dim
    shared = sum(
        bucket.batch_size * bucket.sequence_length * widths * element_bytes
        for bucket in bucket_rows
    )
    branched = shared * len(normalized_modes)
    largest = max(
        bucket.batch_size * bucket.sequence_length * widths * element_bytes
        for bucket in bucket_rows
    )
    # During a manifest transaction, every current object and every next object
    # may coexist.  One object is additionally resident on CPU/device.
    transaction_peak = 2 * branched + largest
    inventory_bytes = _inventory_payload_bytes(model)
    layer_cache_bytes = _max_layer_payload_bytes(model)
    cache_enabled, cache_limit = _source_range_cache_contract(model)
    branch_passes = len(normalized_modes) if layer + 1 < model.config.n_layers else 1
    max_layer_passes = len(bucket_rows) * branch_passes
    cache_limit_admits = cache_limit is None or cache_limit >= layer_cache_bytes
    # This is an admission statement, not a transport measurement: a single
    # remote fetch per layer additionally requires the verified range cache to
    # be enabled.  Without it, every forward pass may fetch the same range.
    cache_reuse_admitted = max_layer_passes <= 1 or (
        cache_enabled
        and cache_limit_admits
        and source_cache_reserve_bytes >= layer_cache_bytes
    )
    source_metrics = model.pager.source.metrics()
    repo_id = str(
        source_metrics.get("repo_id", getattr(model.pager.source, "repo_id", ""))
    )
    official_safe = max(
        inventory_bytes,
        OFFICIAL_SOURCE_SAFE_BYTES if repo_id == OFFICIAL_MODEL_ID else inventory_bytes,
    )
    source_admission = official_safe if require_full_source_disk else 0
    target = Path(run_dir).expanduser()
    probe = target if target.exists() else target.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    required = (
        transaction_peak
        + source_admission
        + source_cache_reserve_bytes
        + disk_margin_bytes
    )
    return LayerwisePlan(
        items=len(items),
        modes=normalized_modes,
        buckets=bucket_rows,
        microbatch_size=microbatch_size,
        padding=padding,
        graft_layer=layer,
        activation_generation_bytes_shared=shared,
        activation_generation_bytes_branched=branched,
        activation_transaction_peak_bytes=transaction_peak,
        activation_object_max_bytes=largest,
        official_source_safe_bytes=official_safe,
        inventory_payload_bytes=inventory_bytes,
        source_storage_admission_bytes=source_admission,
        source_cache_reserve_bytes=source_cache_reserve_bytes,
        source_cache_reuse_required_bytes=layer_cache_bytes,
        source_range_cache_enabled=cache_enabled,
        source_range_cache_limit_bytes=cache_limit,
        layer_forward_passes_max=max_layer_passes,
        source_cache_reuse_admitted=cache_reuse_admitted,
        free_disk_bytes=free,
        required_free_disk_bytes=required,
        require_full_source_disk=bool(require_full_source_disk),
        admitted=free >= required,
    )


class _ActivationStore:
    def __init__(self, run_dir: Path, *, max_object_bytes: int) -> None:
        self.requested_root = run_dir
        self.root = run_dir.absolute()
        self.objects = self.root / "objects"
        self.manifest = self.root / "manifest.json"
        self.result = self.root / "result.json"
        self.max_object_bytes = max_object_bytes

    def ensure(self) -> None:
        run_dir = self.requested_root
        if run_dir.exists() or run_dir.is_symlink():
            metadata = run_dir.lstat()
            if not stat.S_ISDIR(metadata.st_mode):
                raise LayerwiseError("layerwise run path must be a directory")
        run_dir.mkdir(parents=True, exist_ok=True)
        self.root = run_dir.resolve()
        self.objects = self.root / "objects"
        if self.objects.exists() or self.objects.is_symlink():
            metadata = self.objects.lstat()
            if not stat.S_ISDIR(metadata.st_mode):
                raise LayerwiseError("activation object path must be a directory")
        self.objects.mkdir(exist_ok=True)
        self.manifest = self.root / "manifest.json"
        self.result = self.root / "result.json"

    @staticmethod
    def _activation_metadata(
        *, generation_layer: int, bucket: int, variant: str
    ) -> dict[str, str]:
        return {
            "schema": _ACTIVATION_SCHEMA,
            "version": str(LAYERWISE_VERSION),
            "generation_layer": str(generation_layer),
            "bucket": str(bucket),
            "variant": variant,
        }

    def write(
        self,
        hidden: torch.Tensor,
        *,
        generation_layer: int,
        bucket: int,
        variant: str,
    ) -> dict[str, Any]:
        if (
            isinstance(generation_layer, bool)
            or not isinstance(generation_layer, int)
            or generation_layer < -1
        ):
            raise LayerwiseError("activation generation layer is invalid")
        if isinstance(bucket, bool) or not isinstance(bucket, int) or bucket < 0:
            raise LayerwiseError("activation bucket is invalid")
        if not isinstance(variant, str) or not variant:
            raise LayerwiseError("activation variant is invalid")
        cpu = hidden.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        nbytes = cpu.numel() * cpu.element_size()
        if nbytes <= 0 or nbytes > self.max_object_bytes:
            raise LayerwiseError("activation object exceeds its planned byte bound")
        if not bool(torch.isfinite(cpu.float()).all().item()):
            raise LayerwiseError("activation object contains non-finite values")
        descriptor, temporary = tempfile.mkstemp(
            prefix=".activation.", suffix=".safetensors.pending", dir=self.objects
        )
        os.close(descriptor)
        temporary_path = Path(temporary)
        try:
            save_file(
                {"hidden": cpu},
                temporary_path,
                metadata=self._activation_metadata(
                    generation_layer=generation_layer,
                    bucket=bucket,
                    variant=variant,
                ),
            )
            with temporary_path.open("rb") as handle:
                os.fsync(handle.fileno())
            digest = _sha256_file(temporary_path)
            target = self.objects / f"{digest}.safetensors"
            size = temporary_path.stat().st_size
            if target.exists() or target.is_symlink():
                existing = _regular_file(target, "activation object")
                if existing.st_size != size or _sha256_file(target) != digest:
                    raise LayerwiseError(
                        "content-address collision in activation store"
                    )
                temporary_path.unlink()
            else:
                os.replace(temporary_path, target)
                _fsync_directory(self.objects)
            return {
                "schema": _ACTIVATION_SCHEMA,
                "version": LAYERWISE_VERSION,
                "generation_layer": generation_layer,
                "bucket": bucket,
                "variant": variant,
                "file": target.name,
                "sha256": digest,
                "file_bytes": size,
                "tensor_bytes": nbytes,
                "dtype": "bfloat16",
                "shape": list(cpu.shape),
            }
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def load(
        self,
        descriptor: Mapping[str, Any],
        *,
        expected_generation_layer: int,
        expected_bucket: int,
        expected_variant: str,
        expected_shape: Sequence[int],
    ) -> torch.Tensor:
        expected_keys = {
            "schema",
            "version",
            "generation_layer",
            "bucket",
            "variant",
            "file",
            "sha256",
            "file_bytes",
            "tensor_bytes",
            "dtype",
            "shape",
        }
        if set(descriptor) != expected_keys:
            raise LayerwiseError("activation descriptor schema is invalid")
        if (
            descriptor.get("schema") != _ACTIVATION_SCHEMA
            or descriptor.get("version") != LAYERWISE_VERSION
        ):
            raise LayerwiseError("activation descriptor version is unsupported")
        for name, observed, expected in (
            (
                "generation layer",
                descriptor.get("generation_layer"),
                expected_generation_layer,
            ),
            ("bucket", descriptor.get("bucket"), expected_bucket),
        ):
            if (
                isinstance(observed, bool)
                or not isinstance(observed, int)
                or observed != expected
            ):
                raise LayerwiseError(f"activation descriptor {name} mismatch")
        if descriptor.get("variant") != expected_variant:
            raise LayerwiseError("activation descriptor variant mismatch")
        digest = descriptor.get("sha256")
        filename = descriptor.get("file")
        if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
            raise LayerwiseError("activation descriptor SHA-256 is invalid")
        if filename != f"{digest}.safetensors":
            raise LayerwiseError("activation filename is not content-addressed")
        path = self.objects / filename
        metadata = _regular_file(path, "activation object")
        file_bytes = descriptor.get("file_bytes")
        if (
            isinstance(file_bytes, bool)
            or not isinstance(file_bytes, int)
            or file_bytes <= 0
            or metadata.st_size != file_bytes
        ):
            raise LayerwiseError("activation object size mismatch")
        if metadata.st_size > self.max_object_bytes + 1024**2:
            raise LayerwiseError("activation object file exceeds its bound")
        if _sha256_file(path) != digest:
            raise LayerwiseError("activation object SHA-256 mismatch")
        descriptor_shape = descriptor.get("shape")
        planned_shape = list(expected_shape)
        if (
            not isinstance(descriptor_shape, list)
            or len(descriptor_shape) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in descriptor_shape
            )
            or descriptor_shape != planned_shape
            or descriptor.get("dtype") != "bfloat16"
        ):
            raise LayerwiseError("activation descriptor shape is invalid")
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                if list(handle.keys()) != ["hidden"]:
                    raise LayerwiseError("activation object has unexpected tensors")
                metadata_record = handle.metadata()
                expected_metadata = self._activation_metadata(
                    generation_layer=expected_generation_layer,
                    bucket=expected_bucket,
                    variant=expected_variant,
                )
                if metadata_record != expected_metadata:
                    raise LayerwiseError("activation object metadata mismatch")
                tensor = handle.get_tensor("hidden")
        except LayerwiseError:
            raise
        except Exception as exc:
            raise LayerwiseError("cannot decode activation safetensors") from exc
        if tensor.dtype != torch.bfloat16 or list(tensor.shape) != descriptor_shape:
            raise LayerwiseError("activation tensor dtype/shape mismatch")
        nbytes = tensor.numel() * tensor.element_size()
        tensor_bytes = descriptor.get("tensor_bytes")
        if (
            isinstance(tensor_bytes, bool)
            or not isinstance(tensor_bytes, int)
            or nbytes != tensor_bytes
            or nbytes > self.max_object_bytes
        ):
            raise LayerwiseError("activation tensor byte count mismatch")
        if not bool(torch.isfinite(tensor.float()).all().item()):
            raise LayerwiseError("activation tensor contains non-finite values")
        return tensor

    def publish_manifest(self, manifest: Mapping[str, Any]) -> None:
        _atomic_json(self.manifest, manifest)

    def publish_result(self, result: Mapping[str, Any]) -> None:
        _atomic_json(self.result, result)

    def cleanup_unreferenced(self, referenced: set[str]) -> int:
        removed = 0
        for path in self.objects.iterdir():
            if path.name in referenced:
                continue
            if path.name.endswith(".pending"):
                _regular_file(path, "pending activation object")
                path.unlink()
                removed += 1
            elif re.fullmatch(r"[0-9a-f]{64}\.safetensors", path.name):
                _regular_file(path, "orphan activation object")
                path.unlink()
                removed += 1
            else:
                raise LayerwiseError(
                    f"unexpected file in activation object store: {path.name}"
                )
        if removed:
            _fsync_directory(self.objects)
        return removed


class LayerwiseScorer:
    """Execute a fixed candidate benchmark in exact layer-major order."""

    def __init__(
        self,
        model: StreamedDeepSeekV4,
        items: Sequence[LayerwiseItem],
        *,
        run_dir: str | os.PathLike[str],
        modes: Sequence[str] = ("off", "crsa", "softmax", "shuffle"),
        microbatch_size: int = 32,
        padding: str = "right",
        graft_layer: int | None = None,
        graft_alpha: float = 0.05,
        graft_seed: int = 17,
        tokenizer_sha256: str,
        dataset_sha256: str,
        config_sha256: str,
        require_full_source_disk: bool = False,
        source_cache_reserve_bytes: int = 0,
        disk_margin_bytes: int = 1024**3,
    ) -> None:
        self.model = model
        self.items = tuple(items)
        if len({item.item_id for item in self.items}) != len(self.items):
            raise ValueError("layerwise item IDs must be distinct")
        self.plan = build_layerwise_plan(
            model,
            self.items,
            modes=modes,
            microbatch_size=microbatch_size,
            padding=padding,
            graft_layer=graft_layer,
            run_dir=run_dir,
            require_full_source_disk=require_full_source_disk,
            source_cache_reserve_bytes=source_cache_reserve_bytes,
            disk_margin_bytes=disk_margin_bytes,
        )
        if not math.isfinite(float(graft_alpha)) or float(graft_alpha) < 0:
            raise ValueError("graft_alpha must be finite and non-negative")
        for name, value in (
            ("tokenizer_sha256", tokenizer_sha256),
            ("dataset_sha256", dataset_sha256),
            ("config_sha256", config_sha256),
        ):
            if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
                raise ValueError(f"{name} must be lowercase SHA-256")
        source = model.pager.source
        source.inventory()
        source_metrics = source.metrics()
        fingerprint = source_metrics.get("inventory_source_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise LayerwiseError("source has no verified inventory fingerprint")
        runtime_sources = _runtime_source_manifest()
        runtime_dependencies = _runtime_dependency_versions()
        self.identity = {
            "runtime": {
                "schema": LAYERWISE_SCHEMA,
                "source_sha256": _digest(runtime_sources),
                "sources": runtime_sources,
                "dependency_sha256": _digest(runtime_dependencies),
                "dependencies": runtime_dependencies,
            },
            "model": {
                "repo_id": str(
                    source_metrics.get("repo_id", getattr(source, "repo_id", "fixture"))
                ),
                "revision": str(
                    source_metrics.get(
                        "revision", getattr(source, "revision", "fixture")
                    )
                ),
                "inventory_fingerprint": fingerprint,
                "config_sha256": config_sha256,
            },
            "tokenizer_sha256": tokenizer_sha256,
            "dataset_sha256": dataset_sha256,
            "items_sha256": _digest([item.identity_record() for item in self.items]),
            "execution": {
                "compute_dtype": str(model.pager.compute_dtype).removeprefix("torch."),
                "device": str(model.pager.device),
                "simulate_activation_quantization": bool(
                    model.pager.simulate_activation_quantization
                ),
                "quantized_accumulation_policy": (
                    model.pager.QUANTIZED_ACCUMULATION_POLICY
                ),
                "attention_qat_policy": model.ATTENTION_QAT_POLICY,
                "expert_prefetch_policy": (
                    model.pager.EXPERT_PREFETCH_POLICY
                    if model.pager.expert_prefetch_enabled
                    else "disabled"
                ),
                "expert_prefetch_payload_limit_bytes": (
                    model.pager.EXPERT_PREFETCH_PAYLOAD_LIMIT_BYTES
                ),
                "expert_prefetch_transport_policy": (
                    model.pager.EXPERT_PREFETCH_TRANSPORT_POLICY
                ),
                "expert_prefetch_workers": model.pager.EXPERT_PREFETCH_WORKERS,
                "expert_prefetch_max_outstanding": (
                    model.pager.EXPERT_PREFETCH_MAX_OUTSTANDING
                ),
                "expert_prefetch_max_experts": (
                    model.pager.EXPERT_PREFETCH_MAX_EXPERTS
                ),
                "expert_prefetch_resident_limit_bytes": (
                    model.pager.EXPERT_PREFETCH_RESIDENT_LIMIT_BYTES
                ),
                "activation_dtype": "bfloat16",
                "microbatch_size": microbatch_size,
                "padding": padding,
                "graft_layer": self.plan.graft_layer,
                "graft_alpha": float(graft_alpha),
                "graft_seed": int(graft_seed),
                "modes": list(self.plan.modes),
                "source_cache_reserve_bytes": source_cache_reserve_bytes,
                "source_transport_policy": str(
                    source_metrics.get("transport_policy", "unreported")
                ),
                "source_transport_connection_limit": int(
                    source_metrics.get("transport_connection_limit", 0)
                ),
            },
        }
        self.identity_sha256 = _digest(self.identity)
        self.graft_alpha = float(graft_alpha)
        self.graft_seed = int(graft_seed)
        self.store = _ActivationStore(
            Path(run_dir).expanduser(),
            max_object_bytes=self.plan.activation_object_max_bytes,
        )
        self.grafts = {
            mode: DeepSeekV4CrsaGraft(
                mode=mode,
                alpha=0.0 if mode == "off" else self.graft_alpha,
                max_history=model.max_seq_len,
                shuffle_seed=self.graft_seed,
            )
            for mode in self.plan.modes
        }

    def plan_dict(self) -> dict[str, Any]:
        return {
            "schema": LAYERWISE_SCHEMA,
            "identity_sha256": self.identity_sha256,
            "identity": self.identity,
            "plan": self.plan.to_dict(),
        }

    @staticmethod
    def _stable_plan(value: Mapping[str, Any]) -> dict[str, Any]:
        """Remove observational disk fields that legitimately change on resume."""

        return {
            key: child
            for key, child in value.items()
            if key not in {"free_disk_bytes", "admitted"}
        }

    def _bucket_tensors(
        self, bucket: LayerwiseBucket
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.zeros((bucket.batch_size, bucket.sequence_length), dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for row, item_index in enumerate(bucket.item_indices):
            values = self.items[item_index].prompt_token_ids
            ids[row, : len(values)] = torch.tensor(values, dtype=torch.long)
            mask[row, : len(values)] = True
        return ids, mask

    def _bucket_shape(self, bucket: LayerwiseBucket) -> tuple[int, int, int, int]:
        return (
            bucket.batch_size,
            bucket.sequence_length,
            self.model.config.hc_mult,
            self.model.config.dim,
        )

    def _checkpoint_variants(self, completed_layer: int) -> tuple[str, ...]:
        return (
            ("shared",) if completed_layer < self.plan.graft_layer else self.plan.modes
        )

    def _validate_state(
        self, state: Mapping[str, Any], *, verify_objects: bool = True
    ) -> tuple[str, int, dict[str, list[dict[str, Any]]]]:
        phase = state.get("phase")
        if phase not in {"activations", "complete"}:
            raise LayerwiseError("layerwise manifest phase is invalid")
        expected_state_keys = {"phase", "completed_layer", "checkpoints"}
        if phase == "complete":
            expected_state_keys.add("result_body_sha256")
        if set(state) != expected_state_keys:
            raise LayerwiseError("layerwise manifest state schema is invalid")
        completed = state.get("completed_layer")
        if isinstance(completed, bool) or not isinstance(completed, int):
            raise LayerwiseError("completed layer must be an integer")
        if phase == "complete":
            if completed != self.model.config.n_layers - 1:
                raise LayerwiseError("complete manifest has an invalid completed layer")
            result_sha = state.get("result_body_sha256")
            if not isinstance(result_sha, str) or _DIGEST.fullmatch(result_sha) is None:
                raise LayerwiseError("complete manifest result SHA-256 is invalid")
        elif not -1 <= completed < self.model.config.n_layers:
            raise LayerwiseError(
                "activation manifest completed layer is outside its range"
            )

        checkpoints = state.get("checkpoints")
        if not isinstance(checkpoints, list):
            raise LayerwiseError("layerwise checkpoint table is invalid")
        expected_variants = self._checkpoint_variants(completed)
        expected_keys = {
            (variant, bucket.index)
            for variant in expected_variants
            for bucket in self.plan.buckets
        }
        observed: dict[tuple[str, int], dict[str, Any]] = {}
        for row in checkpoints:
            if not isinstance(row, dict):
                raise LayerwiseError("layerwise checkpoint descriptor is invalid")
            variant = row.get("variant")
            bucket_index = row.get("bucket")
            if not isinstance(variant, str):
                raise LayerwiseError("activation descriptor variant is invalid")
            if (
                isinstance(bucket_index, bool)
                or not isinstance(bucket_index, int)
                or not 0 <= bucket_index < len(self.plan.buckets)
            ):
                raise LayerwiseError(
                    "activation descriptor bucket is outside its range"
                )
            key = (variant, bucket_index)
            if key not in expected_keys:
                raise LayerwiseError(
                    "checkpoint variant/bucket is not valid for this generation"
                )
            if key in observed:
                raise LayerwiseError(
                    "checkpoint generation contains a duplicate variant/bucket"
                )
            observed[key] = dict(row)
        if set(observed) != expected_keys:
            raise LayerwiseError(
                "checkpoint generation has incomplete variant/bucket coverage"
            )

        table: dict[str, list[dict[str, Any]]] = {}
        for variant in expected_variants:
            rows: list[dict[str, Any]] = []
            for bucket in self.plan.buckets:
                descriptor = observed[(variant, bucket.index)]
                if verify_objects:
                    self.store.load(
                        descriptor,
                        expected_generation_layer=completed,
                        expected_bucket=bucket.index,
                        expected_variant=variant,
                        expected_shape=self._bucket_shape(bucket),
                    )
                rows.append(descriptor)
            table[variant] = rows
        return phase, completed, table

    def _manifest(
        self,
        *,
        phase: str,
        layer: int,
        checkpoints: Sequence[Mapping[str, Any]],
        started_at_unix: float,
        result_body_sha256: str | None = None,
    ) -> dict[str, Any]:
        state: dict[str, Any] = {
            "phase": phase,
            "completed_layer": layer,
            "checkpoints": [dict(row) for row in checkpoints],
        }
        if phase == "complete":
            if (
                not isinstance(result_body_sha256, str)
                or _DIGEST.fullmatch(result_body_sha256) is None
            ):
                raise LayerwiseError("complete manifest requires a result SHA-256")
            state["result_body_sha256"] = result_body_sha256
        elif result_body_sha256 is not None:
            raise LayerwiseError("activation manifest cannot bind a result")
        body = {
            "identity": self.identity,
            "identity_sha256": self.identity_sha256,
            "plan": self.plan.to_dict(),
            "state": state,
            "started_at_unix": started_at_unix,
        }
        return {
            "schema": LAYERWISE_SCHEMA,
            "version": LAYERWISE_VERSION,
            "kind": _MANIFEST_KIND,
            "body": body,
            "body_sha256": _digest(body),
        }

    @staticmethod
    def _result_document(body: Mapping[str, Any]) -> dict[str, Any]:
        # Canonical round-trip once so the validated return value is byte-model
        # equivalent to the JSON body readers will load from disk (not merely
        # a Python structure containing tuples that JSON later turns to lists).
        materialized = json.loads(_canonical_json(body))
        return {
            "schema": LAYERWISE_SCHEMA,
            "version": LAYERWISE_VERSION,
            "kind": _RESULT_KIND,
            "body": materialized,
            "body_sha256": _digest(materialized),
        }

    def _validate_result(
        self,
        document: Mapping[str, Any],
        *,
        manifest_plan: Mapping[str, Any],
        expected_body_sha256: str,
    ) -> dict[str, Any]:
        if set(document) != {"schema", "version", "kind", "body", "body_sha256"}:
            raise LayerwiseError("completed result envelope schema is invalid")
        if (
            document.get("schema") != LAYERWISE_SCHEMA
            or document.get("version") != LAYERWISE_VERSION
            or document.get("kind") != _RESULT_KIND
        ):
            raise LayerwiseError("completed result envelope is unsupported")
        body = document.get("body")
        if not isinstance(body, dict):
            raise LayerwiseError("completed result body is invalid")
        body_sha = document.get("body_sha256")
        if (
            not isinstance(body_sha, str)
            or _DIGEST.fullmatch(body_sha) is None
            or body_sha != _digest(body)
        ):
            raise LayerwiseError("completed result body SHA-256 mismatch")
        if body_sha != expected_body_sha256:
            raise LayerwiseError("completed result is not bound by the manifest")
        expected_body_keys = {
            "schema",
            "version",
            "identity_sha256",
            "identity",
            "plan",
            "candidate_token_union",
            "candidate_head_rows",
            "modes",
        }
        if set(body) != expected_body_keys:
            raise LayerwiseError("completed result body schema is invalid")
        if (
            body.get("schema") != LAYERWISE_SCHEMA
            or body.get("version") != LAYERWISE_VERSION
        ):
            raise LayerwiseError("completed result body version is unsupported")
        if (
            body.get("identity_sha256") != self.identity_sha256
            or body.get("identity") != self.identity
        ):
            raise LayerwiseError("completed result identity mismatch")
        if _canonical_json(body.get("plan")) != _canonical_json(manifest_plan):
            raise LayerwiseError("completed result plan does not match its manifest")

        candidate_union = sorted(
            {token for item in self.items for token in item.candidate_token_ids}
        )
        if body.get("candidate_token_union") != candidate_union:
            raise LayerwiseError("completed result candidate union is invalid")
        head_rows = body.get("candidate_head_rows")
        if (
            isinstance(head_rows, bool)
            or not isinstance(head_rows, int)
            or head_rows != len(candidate_union)
        ):
            raise LayerwiseError("completed result candidate head row count is invalid")

        modes = body.get("modes")
        if not isinstance(modes, dict) or set(modes) != set(self.plan.modes):
            raise LayerwiseError("completed result mode coverage is invalid")
        expected_ids = [item.item_id for item in self.items]
        for mode in self.plan.modes:
            mode_body = modes.get(mode)
            if not isinstance(mode_body, dict) or set(mode_body) != {"items"}:
                raise LayerwiseError(
                    f"completed result mode {mode!r} schema is invalid"
                )
            rows = mode_body.get("items")
            if not isinstance(rows, list) or len(rows) != len(self.items):
                raise LayerwiseError(
                    f"completed result mode {mode!r} item coverage is invalid"
                )
            observed_ids = [
                row.get("item_id") if isinstance(row, dict) else None for row in rows
            ]
            if observed_ids != expected_ids:
                raise LayerwiseError(
                    f"completed result mode {mode!r} item coverage is invalid"
                )
            for row, item in zip(rows, self.items, strict=True):
                if not isinstance(row, dict) or set(row) != {
                    "item_id",
                    "candidate_token_ids",
                    "candidate_scores",
                    "selected_candidate",
                    "predicted",
                    "expected",
                }:
                    raise LayerwiseError("completed result item schema is invalid")
                if row.get("candidate_token_ids") != list(item.candidate_token_ids):
                    raise LayerwiseError("completed result item candidates are invalid")
                scores = row.get("candidate_scores")
                if (
                    not isinstance(scores, list)
                    or len(scores) != len(item.candidate_token_ids)
                    or any(
                        not isinstance(value, float) or not math.isfinite(value)
                        for value in scores
                    )
                ):
                    raise LayerwiseError(
                        "completed result candidate scores are invalid"
                    )
                selected = row.get("selected_candidate")
                winner = max(
                    range(len(scores)),
                    key=lambda index: (scores[index], -item.candidate_token_ids[index]),
                )
                if (
                    isinstance(selected, bool)
                    or not isinstance(selected, int)
                    or selected != winner
                ):
                    raise LayerwiseError(
                        "completed result selected candidate is invalid"
                    )
                values = item.candidate_values or tuple(
                    range(len(item.candidate_token_ids))
                )
                if _canonical_json(row.get("predicted")) != _canonical_json(
                    values[winner]
                ):
                    raise LayerwiseError("completed result prediction is invalid")
                if _canonical_json(row.get("expected")) != _canonical_json(
                    item.expected
                ):
                    raise LayerwiseError("completed result expected value is invalid")
        return body

    def _load_or_initialize(
        self, *, resume: bool
    ) -> tuple[dict[str, Any], float, int, dict[str, list[dict[str, Any]]]]:
        if not resume and not self.plan.admitted:
            raise LayerwiseError(
                "disk preflight rejected run: "
                f"need {self.plan.required_free_disk_bytes} bytes, "
                f"have {self.plan.free_disk_bytes}"
            )
        self.store.ensure()
        if self.store.manifest.exists() or self.store.manifest.is_symlink():
            if not resume:
                raise LayerwiseError("run manifest exists; pass resume=True")
            manifest = _read_json(self.store.manifest)
            if (
                manifest.get("schema") != LAYERWISE_SCHEMA
                or manifest.get("version") != LAYERWISE_VERSION
            ):
                raise LayerwiseError(
                    "unsupported legacy layerwise manifest; it cannot prove "
                    "activation generation and runtime source provenance"
                )
            if set(manifest) != {"schema", "version", "kind", "body", "body_sha256"}:
                raise LayerwiseError("layerwise manifest envelope schema is invalid")
            if manifest.get("kind") != _MANIFEST_KIND:
                raise LayerwiseError("layerwise manifest kind is invalid")
            body = manifest.get("body")
            if not isinstance(body, dict) or manifest.get("body_sha256") != _digest(
                body
            ):
                raise LayerwiseError("layerwise manifest body SHA-256 mismatch")
            if set(body) != {
                "identity",
                "identity_sha256",
                "plan",
                "state",
                "started_at_unix",
            }:
                raise LayerwiseError("layerwise manifest body schema is invalid")
            if (
                body.get("identity_sha256") != self.identity_sha256
                or body.get("identity") != self.identity
            ):
                raise LayerwiseError("resume identity differs from the existing run")
            existing_plan = body.get("plan")
            if not isinstance(existing_plan, dict) or _digest(
                self._stable_plan(existing_plan)
            ) != _digest(self._stable_plan(self.plan.to_dict())):
                raise LayerwiseError("resume plan differs from the existing run")
            state = body.get("state")
            if not isinstance(state, dict):
                raise LayerwiseError("layerwise manifest state is invalid")
            phase, completed, table = self._validate_state(state)
            del phase
            started_raw = body.get("started_at_unix")
            if (
                isinstance(started_raw, bool)
                or not isinstance(started_raw, (int, float))
                or not math.isfinite(started_raw)
                or started_raw <= 0
            ):
                raise LayerwiseError("layerwise start timestamp is invalid")
            started = float(started_raw)
            referenced = {row["file"] for rows in table.values() for row in rows}
            self.store.cleanup_unreferenced(referenced)
            return manifest, started, completed, table
        if resume:
            raise LayerwiseError("resume requested but no manifest exists")
        started = time.time()
        checkpoints: list[dict[str, Any]] = []
        for bucket in self.plan.buckets:
            ids, _mask = self._bucket_tensors(bucket)
            hidden = self.model.embed_batch(ids)
            checkpoints.append(
                self.store.write(
                    hidden,
                    generation_layer=-1,
                    bucket=bucket.index,
                    variant="shared",
                )
            )
        manifest = self._manifest(
            phase="activations",
            layer=-1,
            checkpoints=checkpoints,
            started_at_unix=started,
        )
        self.store.publish_manifest(manifest)
        _phase, completed, table = self._validate_state(
            manifest["body"]["state"], verify_objects=False
        )
        return manifest, started, completed, table

    def _apply_graft(self, hidden: torch.Tensor, mode: str) -> torch.Tensor:
        result = self.grafts[mode].forward(hidden, return_evidence=True)
        return result[0] if isinstance(result, tuple) else result

    def _run_layer(
        self,
        layer: int,
        current: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        progress: Callable[[Mapping[str, Any]], None] | None,
    ) -> list[dict[str, Any]]:
        clear_priorities = getattr(
            self.model.pager.source, "clear_cache_priorities", None
        )
        if callable(clear_priorities):
            # Priorities are request-local residency hints.  Reset once per
            # layer (never once per bucket), so stale high-priority ranges from
            # earlier layers cannot evict the current layer's experts before
            # its next bucket/variant reuses them.
            clear_priorities()
        output: list[dict[str, Any]] = []
        input_variants = (
            ("shared",) if layer <= self.plan.graft_layer else self.plan.modes
        )
        for variant in input_variants:
            descriptors = current.get(variant)
            if descriptors is None or len(descriptors) != len(self.plan.buckets):
                raise LayerwiseError(f"activation generation omits variant {variant!r}")
            for descriptor, bucket in zip(descriptors, self.plan.buckets, strict=True):
                if descriptor.get("bucket") != bucket.index:
                    raise LayerwiseError("activation bucket order is inconsistent")
                hidden = self.store.load(
                    descriptor,
                    expected_generation_layer=layer - 1,
                    expected_bucket=bucket.index,
                    expected_variant=variant,
                    expected_shape=self._bucket_shape(bucket),
                ).to(self.model.pager.device, dtype=self.model.pager.compute_dtype)
                ids, mask = self._bucket_tensors(bucket)
                next_hidden, _selected = self.model.forward_prefill_layer(
                    hidden, ids, layer=layer, token_mask=mask
                )
                if layer == self.plan.graft_layer:
                    for mode in self.plan.modes:
                        grafted = self._apply_graft(next_hidden, mode)
                        output.append(
                            self.store.write(
                                grafted,
                                generation_layer=layer,
                                bucket=bucket.index,
                                variant=mode,
                            )
                        )
                else:
                    output.append(
                        self.store.write(
                            next_hidden,
                            generation_layer=layer,
                            bucket=bucket.index,
                            variant=variant,
                        )
                    )
                if progress is not None:
                    progress(
                        {
                            "event": "microbatch",
                            "layer": layer,
                            "variant": variant,
                            "bucket": bucket.index,
                            "items": bucket.batch_size,
                            "sequence_length": bucket.sequence_length,
                        }
                    )
        self.model.pager.release()
        return output

    def _score(
        self, checkpoints: Mapping[str, Sequence[Mapping[str, Any]]]
    ) -> dict[str, Any]:
        candidate_union = sorted(
            {token for item in self.items for token in item.candidate_token_ids}
        )
        candidate_column = {token: index for index, token in enumerate(candidate_union)}
        modes: dict[str, Any] = {}
        for mode in self.plan.modes:
            descriptors = checkpoints.get(mode)
            if descriptors is None or len(descriptors) != len(self.plan.buckets):
                raise LayerwiseError(f"final activation generation omits {mode!r}")
            last_by_item: list[torch.Tensor | None] = [None] * len(self.items)
            for descriptor, bucket in zip(descriptors, self.plan.buckets, strict=True):
                hidden = self.store.load(
                    descriptor,
                    expected_generation_layer=self.model.config.n_layers - 1,
                    expected_bucket=bucket.index,
                    expected_variant=mode,
                    expected_shape=self._bucket_shape(bucket),
                ).to(self.model.pager.device, dtype=self.model.pager.compute_dtype)
                final = self.model.finalize_hidden(hidden)
                for row, (item_index, length) in enumerate(
                    zip(bucket.item_indices, bucket.valid_lengths, strict=True)
                ):
                    last_by_item[item_index] = final[row, length - 1].detach().to("cpu")
            if any(value is None for value in last_by_item):
                raise LayerwiseError("final activations omit one or more items")
            matrix = torch.stack(
                [value for value in last_by_item if value is not None]
            ).to(self.model.pager.device)
            logits = self.model.pager.candidate_logits(matrix, candidate_union)
            logits_cpu = logits.detach().to(device="cpu", dtype=torch.float32)
            item_rows: list[dict[str, Any]] = []
            for item_index, item in enumerate(self.items):
                scores = [
                    float(logits_cpu[item_index, candidate_column[token]].item())
                    for token in item.candidate_token_ids
                ]
                if not all(math.isfinite(value) for value in scores):
                    raise LayerwiseError("candidate logits contain non-finite values")
                winner = max(
                    range(len(scores)),
                    key=lambda index: (
                        scores[index],
                        -item.candidate_token_ids[index],
                    ),
                )
                values = item.candidate_values or tuple(
                    range(len(item.candidate_token_ids))
                )
                item_rows.append(
                    {
                        "item_id": item.item_id,
                        "candidate_token_ids": list(item.candidate_token_ids),
                        "candidate_scores": scores,
                        "selected_candidate": winner,
                        "predicted": values[winner],
                        "expected": item.expected,
                    }
                )
            modes[mode] = {"items": item_rows}
        return {
            "schema": LAYERWISE_SCHEMA,
            "version": LAYERWISE_VERSION,
            "identity_sha256": self.identity_sha256,
            "identity": self.identity,
            "plan": self.plan.to_dict(),
            "candidate_token_union": candidate_union,
            "candidate_head_rows": len(candidate_union),
            "modes": modes,
        }

    def run(
        self,
        *,
        resume: bool = False,
        progress: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Execute or resume all layers and return exact candidate scores."""

        manifest, started, completed, checkpoints = self._load_or_initialize(
            resume=resume
        )
        body = manifest["body"]
        state = body["state"]
        if state.get("phase") == "complete":
            result_document = _read_json(self.store.result, label="layerwise result")
            return self._validate_result(
                result_document,
                manifest_plan=body["plan"],
                expected_body_sha256=state["result_body_sha256"],
            )
        for layer in range(completed + 1, self.model.config.n_layers):
            next_rows = self._run_layer(layer, checkpoints, progress=progress)
            next_manifest = self._manifest(
                phase="activations",
                layer=layer,
                checkpoints=next_rows,
                started_at_unix=started,
            )
            self.store.publish_manifest(next_manifest)
            referenced = {str(row["file"]) for row in next_rows}
            self.store.cleanup_unreferenced(referenced)
            _phase, _completed, checkpoints = self._validate_state(
                next_manifest["body"]["state"], verify_objects=False
            )
            if progress is not None:
                progress({"event": "layer_complete", "layer": layer})
        result = self._score(checkpoints)
        result_document = self._result_document(result)
        validated_result = self._validate_result(
            result_document,
            manifest_plan=self.plan.to_dict(),
            expected_body_sha256=result_document["body_sha256"],
        )
        self.store.publish_result(result_document)
        complete = self._manifest(
            phase="complete",
            layer=self.model.config.n_layers - 1,
            checkpoints=[row for rows in checkpoints.values() for row in rows],
            started_at_unix=started,
            result_body_sha256=result_document["body_sha256"],
        )
        self.store.publish_manifest(complete)
        return validated_result


__all__ = [
    "LAYERWISE_SCHEMA",
    "OFFICIAL_SOURCE_SAFE_BYTES",
    "LayerwiseBucket",
    "LayerwiseError",
    "LayerwiseItem",
    "LayerwisePlan",
    "LayerwiseScorer",
    "build_layerwise_plan",
]
