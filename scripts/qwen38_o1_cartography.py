#!/usr/bin/env python3
"""Run a bounded, resumable O1 cartography loop over a local Qwen3.8 bundle."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any

from immer.runtimes.o1_state import O1Cartographer, ProbeJob, ProbeTarget
from immer.runtimes.qwen3_8 import (
    HiddenSketchProjection,
    LogicalModelIdentity,
    ModelPin,
    ProbeCoordinateSpec,
    ProbeResourceBudget,
    ProbeSpec,
    Qwen38Config,
    Qwen38NativeHeadCrsa,
    Qwen38CartographyProbe,
    Qwen38WeightPager,
    SemanticWeightAtlas,
    StreamedQwen38,
    CausalWeightMount,
    prompt_token_sha256,
)
from immer.runtimes.qwen3_8.bundle import QWEN38_BUNDLE_SCHEMA


MANIFEST_SCHEMA = "immer.qwen3.8-o1-cartography-run/v1"
MANIFEST_NAME = "manifest.json"
SCHEDULER_NAME = "scheduler.json"
ATLAS_NAME = "atlas"
_MAX_DOCUMENT_BYTES = 64 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CODE_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_FORBIDDEN_JOB_KEYS = frozenset(
    {"answer", "completion", "gold", "label", "output", "prompt", "question"}
)


class O1CartographyCliError(RuntimeError):
    """The sealed O1 cartography run contract cannot be satisfied."""


@dataclass(slots=True)
class CartographyRuntime:
    """Injected/default runtime handle; it is not a persisted domain object."""

    model: Any
    tensor_plans: tuple[Any, ...]
    close_callback: Callable[[], None] = lambda: None

    def close(self) -> None:
        self.close_callback()


RuntimeFactory = Callable[[Mapping[str, Any]], CartographyRuntime]
ProbeFactory = Callable[[Any], Any]
AtlasFactory = Callable[..., Any]


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise O1CartographyCliError("value is not canonical JSON") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise O1CartographyCliError(f"{label} must be a lowercase SHA-256")
    return value


def _code_revision(value: object) -> str:
    if not isinstance(value, str) or _CODE_REVISION.fullmatch(value) is None:
        raise O1CartographyCliError(
            "code revision must be a full lowercase 40- or 64-digit SHA"
        )
    return value


def _uint(value: object, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise O1CartographyCliError(f"{label} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise O1CartographyCliError(f"{label} must be {qualifier}")
    return value


def _seconds(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise O1CartographyCliError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < (0.0 if not positive else 0.0):
        raise O1CartographyCliError(f"{label} must be finite and non-negative")
    if positive and result == 0.0:
        raise O1CartographyCliError(f"{label} must be positive")
    return result


def _plain_root(value: str | os.PathLike[str], *, create: bool) -> Path:
    root = Path(value).expanduser().absolute()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    try:
        metadata = root.lstat()
    except OSError as exc:
        raise O1CartographyCliError(f"cartography root is unavailable: {root}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise O1CartographyCliError("cartography root must be a non-symlink directory")
    return root


def _contained(root: Path, name: str) -> Path:
    if name not in {MANIFEST_NAME, SCHEDULER_NAME, ATLAS_NAME}:
        raise O1CartographyCliError("cartography output name is not allowlisted")
    candidate = root / name
    if candidate.parent != root:
        raise O1CartographyCliError("cartography output escapes its root")
    return candidate


def _stable_regular_bytes(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise O1CartographyCliError(f"cannot open {label}: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        linked = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(linked.st_mode)
            or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
        ):
            raise O1CartographyCliError(f"{label} is not a stable regular file")
        if opened.st_size < 0 or opened.st_size > _MAX_DOCUMENT_BYTES:
            raise O1CartographyCliError(f"{label} exceeds its byte bound")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise O1CartographyCliError(f"{label} was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise O1CartographyCliError(f"{label} grew while reading")
        current = os.fstat(descriptor)
        linked = path.lstat()
        if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ) or (current.st_dev, current.st_ino) != (linked.st_dev, linked.st_ino):
            raise O1CartographyCliError(f"{label} changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _strict_json_bytes(raw: bytes, label: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise O1CartographyCliError(f"duplicate JSON key in {label}: {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(raw, object_pairs_hook=pairs)
    except O1CartographyCliError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise O1CartographyCliError(f"{label} is invalid JSON") from exc


@contextmanager
def _root_lock(root: Path):
    lock_path = root / ".qwen38-o1-cartography.lock"
    flags = os.O_RDWR | os.O_CREAT | int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise O1CartographyCliError("cannot open cartography root lock") from exc
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        opened = os.fstat(descriptor)
        linked = lock_path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(linked.st_mode)
            or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
        ):
            raise O1CartographyCliError("cartography root lock changed")
        yield
        current = os.fstat(descriptor)
        linked = lock_path.lstat()
        if (
            (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)
            or (current.st_dev, current.st_ino) != (linked.st_dev, linked.st_ino)
            or not stat.S_ISREG(linked.st_mode)
        ):
            raise O1CartographyCliError("cartography root lock changed")
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _atomic_new(path: Path, value: bytes) -> None:
    if len(value) > _MAX_DOCUMENT_BYTES:
        raise O1CartographyCliError("manifest exceeds its byte bound")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".pending", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise O1CartographyCliError(f"manifest already exists: {path}") from exc
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary_path.unlink(missing_ok=True)


def _contains_forbidden_job_key(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() in _FORBIDDEN_JOB_KEYS
            or _contains_forbidden_job_key(member)
            for key, member in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_job_key(member) for member in value)
    return False


def _native_from_record(value: object) -> Qwen38NativeHeadCrsa | None:
    if value is None:
        return None
    expected = {field.name for field in fields(Qwen38NativeHeadCrsa)}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise O1CartographyCliError("native Head-CRSA record is malformed")
    return Qwen38NativeHeadCrsa(
        layer=value["layer"],
        alpha=value["alpha"],
        head_indices=tuple(value["head_indices"]),
        balance_alpha=value["balance_alpha"],
        diagonal_debit=value["diagonal_debit"],
    )


def _spec_from_record(value: object) -> ProbeSpec:
    fields = {
        "budget",
        "code_revision",
        "coordinate",
        "family_sha256",
        "hidden_sketch",
        "intervention_mode",
        "label_source_sha256",
        "native_head_crsa",
        "prompt_sha256",
        "prompt_token_ids",
        "question_sha256",
        "start_layer",
        "stop_layer",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise O1CartographyCliError("probe spec record is malformed")
    coordinate = value["coordinate"]
    budget = value["budget"]
    sketch = value["hidden_sketch"]
    if not isinstance(coordinate, Mapping) or not isinstance(budget, Mapping):
        raise O1CartographyCliError("probe coordinate/budget is malformed")
    if set(budget) != set(ProbeResourceBudget.__dataclass_fields__):
        raise O1CartographyCliError("probe resource budget is malformed")
    if sketch is not None and (
        not isinstance(sketch, Mapping)
        or set(sketch) != {"output_dimensions", "seed_sha256"}
    ):
        raise O1CartographyCliError("hidden sketch record is malformed")
    return ProbeSpec(
        prompt_token_ids=tuple(value["prompt_token_ids"]),
        prompt_sha256=value["prompt_sha256"],
        start_layer=value["start_layer"],
        stop_layer=value["stop_layer"],
        coordinate=ProbeCoordinateSpec(**dict(coordinate)),
        intervention_mode=value["intervention_mode"],
        code_revision=value["code_revision"],
        budget=ProbeResourceBudget(**dict(budget)),
        hidden_sketch=(
            None if sketch is None else HiddenSketchProjection(**dict(sketch))
        ),
        question_sha256=value["question_sha256"],
        family_sha256=value["family_sha256"],
        label_source_sha256=value["label_source_sha256"],
        native_head_crsa=_native_from_record(value["native_head_crsa"]),
    )


def _target_for_spec(spec: ProbeSpec) -> ProbeTarget:
    coordinate = spec.coordinate
    if coordinate.head_index is not None:
        return ProbeTarget(coordinate.module, "head", coordinate.head_index)
    if coordinate.row_start is not None:
        return ProbeTarget(coordinate.module, "block", coordinate.row_start)
    return ProbeTarget(coordinate.module, "module", None)


def _prepared_job(
    value: ProbeSpec | Mapping[str, Any],
    *,
    prompt_token_ids: tuple[int, ...],
    prompt_sha256: str,
    code_revision: str,
    model_pin_sha256: str,
    base_seed: int,
) -> dict[str, Any]:
    probe_family = "unlabeled"
    target: ProbeTarget | None = None
    read_budget_bytes = None
    model_budget_seconds = None
    if isinstance(value, ProbeSpec):
        spec = value
    elif isinstance(value, Mapping):
        if _contains_forbidden_job_key(value):
            raise O1CartographyCliError(
                "job input contains holdout/label-bearing fields"
            )
        raw_spec = value.get("spec", value)
        if not isinstance(raw_spec, Mapping):
            raise O1CartographyCliError("job spec must be an object")
        if set(raw_spec) == {
            "budget",
            "code_revision",
            "coordinate",
            "family_sha256",
            "hidden_sketch",
            "intervention_mode",
            "label_source_sha256",
            "native_head_crsa",
            "prompt_sha256",
            "prompt_token_ids",
            "question_sha256",
            "start_layer",
            "stop_layer",
        }:
            spec = _spec_from_record(raw_spec)
        else:
            coordinate = raw_spec.get("coordinate")
            if not isinstance(coordinate, Mapping):
                raise O1CartographyCliError("job coordinate must be an object")
            kwargs: dict[str, Any] = {
                "prompt_token_ids": prompt_token_ids,
                "prompt_sha256": prompt_sha256,
                "start_layer": raw_spec.get("start_layer", 0),
                "stop_layer": raw_spec.get(
                    "stop_layer", int(coordinate.get("layer", -1)) + 1
                ),
                "coordinate": ProbeCoordinateSpec(**dict(coordinate)),
                "intervention_mode": raw_spec.get("intervention_mode", "passive"),
                "code_revision": code_revision,
            }
            for name in ("question_sha256", "family_sha256", "label_source_sha256"):
                if name in raw_spec:
                    kwargs[name] = raw_spec[name]
            if "budget" in raw_spec:
                kwargs["budget"] = ProbeResourceBudget(**dict(raw_spec["budget"]))
            if raw_spec.get("hidden_sketch") is not None:
                kwargs["hidden_sketch"] = HiddenSketchProjection(
                    **dict(raw_spec["hidden_sketch"])
                )
            if "native_head_crsa" in raw_spec:
                kwargs["native_head_crsa"] = _native_from_record(
                    raw_spec["native_head_crsa"]
                )
            spec = ProbeSpec(**kwargs)
        probe_family = str(value.get("probe_family", probe_family))
        read_budget_bytes = value.get("read_budget_bytes")
        model_budget_seconds = value.get("model_budget_seconds")
        if "target" in value:
            target = ProbeTarget.from_document(value["target"])
    else:
        raise O1CartographyCliError("jobs must be ProbeSpec instances or objects")
    if (
        spec.prompt_token_ids != prompt_token_ids
        or spec.prompt_sha256 != prompt_sha256
        or spec.code_revision != code_revision
    ):
        raise O1CartographyCliError("job spec differs from the sealed prompt/code pin")
    target = _target_for_spec(spec) if target is None else target
    if target.module != spec.coordinate.module:
        raise O1CartographyCliError("probe target and coordinate module differ")
    job_seed = int(
        _digest({"base_seed": base_seed, "spec_sha256": spec.sha256})[:16], 16
    ) % (2**63)
    job = ProbeJob.create(
        layer=spec.coordinate.layer,
        target=target,
        probe_family=probe_family,
        intervention=str(spec.intervention_mode),
        code_pin=code_revision,
        model_pin=model_pin_sha256,
        seed=job_seed,
        prompt_sha256=prompt_sha256,
        read_budget_bytes=read_budget_bytes,
        model_budget_seconds=model_budget_seconds,
    )
    return {"job": job.to_document(), "spec": spec.as_record()}


def prepare_manifest(
    root: str | os.PathLike[str],
    *,
    bundle_root: str | os.PathLike[str],
    model_pin: ModelPin,
    prompt_token_ids: Sequence[int],
    prompt_sha256: str,
    jobs: Sequence[ProbeSpec | Mapping[str, Any]],
    code_revision: str,
    seed: int = 0,
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically create one immutable, question-only finite run manifest."""

    root_path = _plain_root(root, create=True)
    bundle_path = _plain_root(bundle_root, create=False)
    code = _code_revision(code_revision)
    seed = _uint(seed, "scheduler seed")
    if not isinstance(model_pin, ModelPin):
        raise TypeError("model_pin must be a ModelPin")
    if model_pin.code_revision != code:
        raise O1CartographyCliError("model pin and run code revision differ")
    tokens = tuple(prompt_token_ids)
    claimed_prompt = _sha(prompt_sha256, "prompt SHA-256")
    if prompt_token_sha256(tokens) != claimed_prompt:
        raise O1CartographyCliError("prompt token IDs do not match prompt SHA-256")
    if not jobs or len(jobs) > 1_000_000:
        raise O1CartographyCliError("finite job frontier must be non-empty and bounded")
    prepared = [
        _prepared_job(
            value,
            prompt_token_ids=tokens,
            prompt_sha256=claimed_prompt,
            code_revision=code,
            model_pin_sha256=model_pin.sha256,
            base_seed=seed,
        )
        for value in jobs
    ]
    ids = [row["job"]["job_id"] for row in prepared]
    if len(ids) != len(set(ids)):
        raise O1CartographyCliError("finite job frontier contains duplicate jobs")
    spec_ids = [row["spec"] for row in prepared]
    if len({_digest(spec) for spec in spec_ids}) != len(spec_ids):
        raise O1CartographyCliError(
            "finite job frontier repeats an identical probe spec"
        )
    runtime_body = {
        "compute_dtype": "bfloat16",
        "device": "cpu",
        "max_resident_bytes": 1024**3,
        "require_official_config": True,
        "source_budget_mb": 1_048_576.0,
    }
    if runtime is not None:
        runtime_body.update(dict(runtime))
    body = {
        "bundle": {
            "manifest_sha256": model_pin.bundle_manifest_sha256,
            "root": str(bundle_path),
        },
        "code_revision": code,
        "holdout_accessed": False,
        "jobs": prepared,
        "model_pin": model_pin.to_document(),
        "model_pin_sha256": model_pin.sha256,
        "outputs": {"atlas": ATLAS_NAME, "scheduler": SCHEDULER_NAME},
        "prompt": {"sha256": claimed_prompt, "token_ids": list(tokens)},
        "runtime": runtime_body,
        "scheduler": {"seed": seed},
    }
    document = {"body": body, "schema": MANIFEST_SCHEMA, "sha256": _digest(body)}
    with _root_lock(root_path):
        _atomic_new(_contained(root_path, MANIFEST_NAME), _canonical(document) + b"\n")
    return document


def _load_manifest(root: Path) -> tuple[dict[str, Any], Mapping[str, Any]]:
    raw = _stable_regular_bytes(_contained(root, MANIFEST_NAME), "run manifest")
    document = _strict_json_bytes(raw, "run manifest")
    if raw != _canonical(document) + b"\n":
        raise O1CartographyCliError("run manifest is not canonical JSONL")
    if (
        not isinstance(document, dict)
        or set(document) != {"body", "schema", "sha256"}
        or document.get("schema") != MANIFEST_SCHEMA
        or not isinstance(document.get("body"), Mapping)
        or document.get("sha256") != _digest(document["body"])
    ):
        raise O1CartographyCliError("run manifest seal is invalid")
    body = document["body"]
    required = {
        "bundle",
        "code_revision",
        "holdout_accessed",
        "jobs",
        "model_pin",
        "model_pin_sha256",
        "outputs",
        "prompt",
        "runtime",
        "scheduler",
    }
    if set(body) != required or body.get("holdout_accessed") is not False:
        raise O1CartographyCliError("run manifest body is malformed")
    if body.get("outputs") != {"atlas": ATLAS_NAME, "scheduler": SCHEDULER_NAME}:
        raise O1CartographyCliError("run outputs escape their sealed containment")
    pin = ModelPin.from_document(body["model_pin"])
    if pin.sha256 != body.get("model_pin_sha256"):
        raise O1CartographyCliError("run model pin digest is invalid")
    if pin.code_revision != _code_revision(body.get("code_revision")):
        raise O1CartographyCliError("run code/model pins differ")
    bundle = body.get("bundle")
    prompt = body.get("prompt")
    if (
        not isinstance(bundle, Mapping)
        or set(bundle) != {"manifest_sha256", "root"}
        or bundle.get("manifest_sha256") != pin.bundle_manifest_sha256
        or not isinstance(prompt, Mapping)
        or set(prompt) != {"sha256", "token_ids"}
    ):
        raise O1CartographyCliError("run bundle/prompt pin is malformed")
    token_ids = tuple(prompt["token_ids"])
    if prompt_token_sha256(token_ids) != _sha(prompt["sha256"], "prompt SHA-256"):
        raise O1CartographyCliError("run prompt pin is invalid")
    _manifest_jobs(body)
    return document, body


def _manifest_jobs(body: Mapping[str, Any]) -> tuple[tuple[ProbeJob, ProbeSpec], ...]:
    raw_jobs = body.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise O1CartographyCliError("run job frontier is malformed")
    code = body["code_revision"]
    model_pin_sha = body["model_pin_sha256"]
    prompt = body["prompt"]
    rows: list[tuple[ProbeJob, ProbeSpec]] = []
    for raw in raw_jobs:
        if not isinstance(raw, Mapping) or set(raw) != {"job", "spec"}:
            raise O1CartographyCliError("run job entry is malformed")
        job = ProbeJob.from_document(raw["job"])
        spec = _spec_from_record(raw["spec"])
        if (
            job.code_pin != code
            or job.model_pin != model_pin_sha
            or job.prompt_sha256 != prompt["sha256"]
            or job.layer != spec.coordinate.layer
            or job.target.module != spec.coordinate.module
            or job.intervention != spec.intervention_mode
            or spec.prompt_sha256 != prompt["sha256"]
            or list(spec.prompt_token_ids) != prompt["token_ids"]
            or spec.code_revision != code
        ):
            raise O1CartographyCliError("run job/spec identity is inconsistent")
        rows.append((job, spec))
    if len({job.job_id for job, _spec in rows}) != len(rows):
        raise O1CartographyCliError("run job frontier contains duplicates")
    if len({spec.sha256 for _job, spec in rows}) != len(rows):
        raise O1CartographyCliError("run job frontier repeats a probe spec")
    return tuple(rows)


def _bundle_document(body: Mapping[str, Any]) -> Mapping[str, Any]:
    bundle = body["bundle"]
    root = _plain_root(bundle["root"], create=False)
    raw = _stable_regular_bytes(root / "bundle.json", "causal bundle manifest")
    document = _strict_json_bytes(raw, "causal bundle manifest")
    if (
        not isinstance(document, Mapping)
        or set(document) != {"body", "schema", "sha256"}
        or document.get("schema") != QWEN38_BUNDLE_SCHEMA
        or not isinstance(document.get("body"), Mapping)
        or document.get("sha256") != _digest(document["body"])
        or document.get("sha256") != bundle["manifest_sha256"]
    ):
        raise O1CartographyCliError("causal bundle manifest pin is stale or invalid")
    pin = ModelPin.from_document(body["model_pin"])
    manifest_body = document["body"]
    if (
        manifest_body.get("logical_model")
        != {"repo_id": pin.repo_id, "revision": pin.revision}
        or manifest_body.get("layout_fingerprint") != pin.bundle_fingerprint
        or manifest_body.get("checkpoint_complete") is not True
    ):
        raise O1CartographyCliError("causal bundle differs from the sealed model pin")
    return document


def _default_runtime_factory(body: Mapping[str, Any]) -> CartographyRuntime:
    """Open only the sealed local causal bundle; remote sources are impossible."""

    _bundle_document(body)
    pin = ModelPin.from_document(body["model_pin"])
    options = body["runtime"]
    if not isinstance(options, Mapping):
        raise O1CartographyCliError("runtime options are malformed")
    required = {
        "compute_dtype",
        "device",
        "max_resident_bytes",
        "require_official_config",
        "source_budget_mb",
    }
    if set(options) != required:
        raise O1CartographyCliError("runtime options have unknown or missing fields")
    max_resident = _uint(
        options["max_resident_bytes"], "max resident bytes", positive=True
    )
    source_budget = _seconds(
        options["source_budget_mb"], "source budget", positive=True
    )
    if not isinstance(options["require_official_config"], bool):
        raise O1CartographyCliError("require_official_config must be boolean")
    mount = CausalWeightMount(
        body["bundle"]["root"],
        LogicalModelIdentity(repo_id=pin.repo_id, revision=pin.revision),
        budget_mb=source_budget,
    )
    pager: Qwen38WeightPager | None = None
    try:
        if mount.layout.layout_fingerprint != pin.bundle_fingerprint:
            raise O1CartographyCliError("mounted causal layout differs from model pin")
        try:
            config_document = json.loads(mount.source.reader.fetch_file("config.json"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise O1CartographyCliError("causal bundle config is unreadable") from exc
        if not isinstance(config_document, Mapping):
            raise O1CartographyCliError("causal bundle config root is invalid")
        config = Qwen38Config.from_mapping(
            config_document,
            require_official=options["require_official_config"],
        )
        pager = Qwen38WeightPager(
            mount.source,
            device=str(options["device"]),
            compute_dtype=str(options["compute_dtype"]),
            max_resident_bytes=max_resident,
            close_source=False,
            require_source_identity=True,
            causal_tensor_reader=mount.tensor_reader,
        )
        prompt_length = len(body["prompt"]["token_ids"])
        model = StreamedQwen38(
            config,
            pager,
            max_batch_size=1,
            max_seq_len=max(1, prompt_length),
        )
        plans = tuple(
            mount.resolve_tensor_plan(str(row["name"]))
            for row in mount.source.inventory().get("tensors", ())
        )

        def close() -> None:
            assert pager is not None
            try:
                pager.close()
            finally:
                mount.close()

        return CartographyRuntime(model=model, tensor_plans=plans, close_callback=close)
    except Exception:
        if pager is not None:
            pager.close()
        mount.close()
        raise


def _open_runtime(
    body: Mapping[str, Any], runtime_factory: RuntimeFactory | None
) -> CartographyRuntime:
    runtime = (runtime_factory or _default_runtime_factory)(body)
    if not isinstance(runtime, CartographyRuntime):
        raise TypeError("runtime_factory must return CartographyRuntime")
    if not runtime.tensor_plans:
        runtime.close()
        raise O1CartographyCliError("cartography runtime has no tensor plans")
    return runtime


def _stream(stream_factory: Callable[..., Any] | None, seed: int) -> Any:
    if stream_factory is None:
        return None
    try:
        return stream_factory(seed=seed)
    except TypeError:
        return stream_factory()


def _scheduler(
    root: Path,
    body: Mapping[str, Any],
    *,
    create: bool,
    stream_factory: Callable[..., Any] | None,
) -> O1Cartographer | None:
    state_path = _contained(root, SCHEDULER_NAME)
    seed = _uint(body["scheduler"]["seed"], "scheduler seed")
    stream = _stream(stream_factory, seed)
    if state_path.exists() or state_path.is_symlink():
        return O1Cartographer.restore(
            state_path,
            code_pin=body["code_revision"],
            model_pin=body["model_pin_sha256"],
            stream=stream,
        )
    if not create:
        return None
    return O1Cartographer(
        jobs=tuple(job for job, _spec in _manifest_jobs(body)),
        code_pin=body["code_revision"],
        model_pin=body["model_pin_sha256"],
        state_path=state_path,
        stream=stream,
        seed=seed,
    )


def _atlas_path(root: Path, *, create: bool) -> Path | None:
    path = _contained(root, ATLAS_NAME)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        if not create:
            return None
        try:
            os.mkdir(path, 0o700)
        except OSError as exc:
            raise O1CartographyCliError(
                "cannot create contained atlas directory"
            ) from exc
        metadata = path.lstat()
    except OSError as exc:
        raise O1CartographyCliError("cannot inspect atlas directory") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise O1CartographyCliError("atlas must be a contained non-symlink directory")
    return path


def _open_atlas(
    root: Path,
    body: Mapping[str, Any],
    runtime: CartographyRuntime,
    *,
    create: bool,
    atlas_factory: AtlasFactory,
) -> Any | None:
    path = _atlas_path(root, create=create)
    if path is None:
        return None
    return atlas_factory(
        path,
        model_pin=ModelPin.from_document(body["model_pin"]),
        tensor_plans=runtime.tensor_plans,
    )


def _coordinate_matches_spec(coordinate: Any, spec: ProbeSpec) -> bool:
    requested = spec.coordinate
    if (
        getattr(coordinate, "layer", None) != requested.layer
        or getattr(coordinate, "module", None) != requested.module
        or getattr(coordinate, "tensor", None) != requested.tensor
        or getattr(coordinate, "head_index", None) != requested.head_index
        or getattr(coordinate, "row_start", None) != requested.row_start
        or getattr(coordinate, "row_end", None) != requested.row_end
    ):
        return False
    if requested.row_start is not None:
        return True
    tensor_offset = getattr(coordinate, "tensor_absolute_offset", None)
    range_offset = getattr(coordinate, "range_absolute_offset", None)
    tensor_length = getattr(coordinate, "tensor_length", None)
    range_length = getattr(coordinate, "range_length", None)
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (tensor_offset, range_offset, tensor_length, range_length)
    ):
        return False
    relative = range_offset - tensor_offset
    expected_length = (
        tensor_length - requested.relative_byte_offset
        if requested.byte_length is None
        else requested.byte_length
    )
    return (
        relative == requested.relative_byte_offset and range_length == expected_length
    )


def _intervention_configuration_sha256(spec: ProbeSpec) -> str:
    native = spec.native_head_crsa
    if spec.intervention_mode == "native":
        assert native is not None
        configuration: Mapping[str, Any] = asdict(native)
    elif spec.intervention_mode == "placebo" and native is not None:
        configuration = asdict(replace(native, alpha=0.0))
    else:
        configuration = {"alpha": 0.0, "kind": "original-qwen-identity"}
    return _digest(configuration)


def _measurement_matches_spec(
    measurement: Any, spec: ProbeSpec, model_pin: ModelPin
) -> bool:
    intervention = getattr(measurement, "intervention", None)
    return (
        getattr(measurement, "model_pin", None) == model_pin
        and getattr(measurement, "probe", None) == spec.probe_identity
        and _coordinate_matches_spec(getattr(measurement, "coordinate", None), spec)
        and getattr(intervention, "mode", None) == spec.intervention_mode
        and getattr(intervention, "configuration_sha256", None)
        == _intervention_configuration_sha256(spec)
    )


def _primary_measurement(result: Any, spec: ProbeSpec, model_pin: ModelPin) -> Any:
    measurement = getattr(result, "measurement", None)
    if measurement is None:
        raise O1CartographyCliError("probe result has no primary measurement")
    if not _measurement_matches_spec(measurement, spec, model_pin):
        raise O1CartographyCliError(
            "probe measurement differs from its sealed coordinate/intervention"
        )
    return measurement


def _atlas_receipt_sha256(job: ProbeJob, measurement_sha256: str) -> str:
    del job
    # This is deliberately the primary atlas record identity itself: callers
    # can authenticate it through either atlas query index after every crash.
    return _sha(measurement_sha256, "primary measurement SHA-256")


def _observation(
    job: ProbeJob,
    spec: ProbeSpec,
    measurement: Any,
    *,
    reused: bool,
) -> dict[str, Any]:
    measurement_sha = _sha(measurement.sha256, "measurement SHA-256")
    coordinate_sha = _sha(measurement.coordinate.sha256, "coordinate SHA-256")
    return {
        "atlas_receipt_sha256": _atlas_receipt_sha256(job, measurement_sha),
        "coordinate_sha256": coordinate_sha,
        "intervention": spec.intervention_mode,
        "measurement_sha256": measurement_sha,
        "model_pin_sha256": measurement.model_pin.sha256,
        "probe_spec_sha256": spec.sha256,
        "prompt_sha256": spec.prompt_sha256,
        "reused_atlas_proof": reused,
    }


def _find_reusable_measurement(
    atlas: Any, spec: ProbeSpec, model_pin: ModelPin
) -> Any | None:
    result = atlas.query_by_prompt_signature(spec.probe_identity.prompt_signature)
    matches = []
    for measurement in result.measurements:
        if _measurement_matches_spec(measurement, spec, model_pin):
            matches.append(measurement)
    if not matches:
        return None
    by_sha = {measurement.sha256: measurement for measurement in matches}
    if len(by_sha) != 1:
        raise O1CartographyCliError("atlas contains ambiguous reusable probe proofs")
    return next(iter(by_sha.values()))


def _append_probe_result(
    atlas: Any,
    result: Any,
    *,
    job: ProbeJob,
    spec: ProbeSpec,
    model_pin: ModelPin,
) -> tuple[dict[str, Any], int]:
    if hasattr(result, "verify"):
        result.verify()
    primary = _primary_measurement(result, spec, model_pin)
    measurements = tuple(getattr(result, "measurements_in_append_order", (primary,)))
    receipts = tuple(
        atlas.append_measurement(measurement) for measurement in measurements
    )
    for measurement, receipt in zip(measurements, receipts, strict=True):
        if (
            getattr(receipt, "record_kind", None) != "measurement"
            or getattr(receipt, "record_sha256", None) != measurement.sha256
            or _SHA256.fullmatch(str(getattr(receipt, "segment_sha256", ""))) is None
        ):
            raise O1CartographyCliError("atlas returned an invalid append receipt")
    primary_receipts = [
        receipt for receipt in receipts if receipt.record_sha256 == primary.sha256
    ]
    if len(primary_receipts) != 1:
        raise O1CartographyCliError("atlas did not return the primary append receipt")
    source_bytes = 0
    for name in ("access_trace", "control_access_trace"):
        trace = getattr(result, name, None)
        for operation in getattr(trace, "operations", ()):
            value = getattr(operation, "source_bytes", None)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise O1CartographyCliError(
                    "probe access trace has an invalid source-byte count"
                )
            source_bytes += value
    return _observation(job, spec, primary, reused=False), source_bytes


def _reconcile_receipts(
    scheduler: O1Cartographer,
    atlas: Any,
    specs: Mapping[str, ProbeSpec],
    model_pin: ModelPin,
) -> int:
    attached = 0
    for outcome in scheduler.outcomes:
        if outcome.status != "succeeded" or outcome.atlas_receipt_sha256 is not None:
            continue
        observation = outcome.observation_document()
        if observation is None:
            raise O1CartographyCliError("successful outcome lost its atlas observation")
        receipt = _sha(
            observation.get("atlas_receipt_sha256"),
            "persisted atlas receipt SHA-256",
        )
        spec = specs[outcome.job_id]
        reusable = _find_reusable_measurement(atlas, spec, model_pin)
        if reusable is None or reusable.sha256 != receipt:
            raise O1CartographyCliError(
                "unpromoted successful outcome has no authentic atlas record"
            )
        scheduler.attach_atlas_receipt(
            job_id=outcome.job_id,
            attempt=outcome.attempt,
            receipt_sha256=receipt,
        )
        attached += 1
    return attached


def run_cartography(
    root: str | os.PathLike[str],
    *,
    max_jobs: int,
    max_seconds: float,
    runtime_factory: RuntimeFactory | None = None,
    probe_factory: ProbeFactory = Qwen38CartographyProbe,
    atlas_factory: AtlasFactory = SemanticWeightAtlas,
    stream_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Resume durable probe->atlas->receipt transactions until a chosen stop."""

    job_limit = _uint(max_jobs, "max_jobs")
    second_limit = _seconds(max_seconds, "max_seconds")
    root_path = _plain_root(root, create=False)
    with _root_lock(root_path):
        manifest, body = _load_manifest(root_path)
        scheduler = _scheduler(
            root_path, body, create=True, stream_factory=stream_factory
        )
        assert scheduler is not None
        runtime = _open_runtime(body, runtime_factory)
        try:
            atlas = _open_atlas(
                root_path,
                body,
                runtime,
                create=True,
                atlas_factory=atlas_factory,
            )
            assert atlas is not None
            model_pin = ModelPin.from_document(body["model_pin"])
            specs = {job.job_id: spec for job, spec in _manifest_jobs(body)}
            probe = probe_factory(runtime.model)
            reconciled = _reconcile_receipts(scheduler, atlas, specs, model_pin)
            outcomes = []
            started = time.monotonic()
            stop_reason = "max-jobs"
            while job_limit == 0 or len(outcomes) < job_limit:
                if second_limit and time.monotonic() - started >= second_limit:
                    stop_reason = "max-seconds"
                    break

                def execute(job: ProbeJob, _attempt: int):
                    spec = specs[job.job_id]
                    reusable = _find_reusable_measurement(atlas, spec, model_pin)
                    if reusable is not None:
                        return _observation(job, spec, reusable, reused=True)
                    before = time.monotonic()
                    result = probe.execute(spec, atlas_head_revision=atlas.revision())
                    observation, source_bytes = _append_probe_result(
                        atlas,
                        result,
                        job=job,
                        spec=spec,
                        model_pin=model_pin,
                    )
                    from immer.runtimes.o1_state import ProbeOutcome

                    elapsed = time.monotonic() - before
                    return ProbeOutcome.succeeded(
                        job,
                        _attempt,
                        observation,
                        read_bytes=source_bytes,
                        model_seconds=elapsed,
                        wall_seconds=elapsed,
                    )

                outcome = scheduler.step(execute)
                if outcome is None:
                    stop_reason = (
                        "coverage-complete"
                        if scheduler.coverage().complete
                        else "scheduler-stopped"
                    )
                    break
                outcomes.append(outcome)
                if outcome.status == "succeeded":
                    observation = outcome.observation_document()
                    receipt = observation["atlas_receipt_sha256"]
                    scheduler.attach_atlas_receipt(
                        job_id=outcome.job_id,
                        attempt=outcome.attempt,
                        receipt_sha256=receipt,
                    )
            elapsed = time.monotonic() - started
            report = {
                "attached_atlas_receipt_sha256s": sorted(
                    outcome.observation_document()["atlas_receipt_sha256"]
                    for outcome in outcomes
                    if outcome.status == "succeeded"
                ),
                "atlas": atlas.coverage_matrix(),
                "atlas_revision": atlas.revision().to_document(),
                "attempts_executed": len(outcomes),
                "coverage": scheduler.coverage().to_document(),
                "elapsed_seconds": elapsed,
                "manifest_sha256": manifest["sha256"],
                "receipts_reconciled": reconciled,
                "stop_reason": stop_reason,
            }
            return report
        finally:
            runtime.close()


def _empty_coverage(total_jobs: int) -> dict[str, Any]:
    return {
        "aborted_terminal_jobs": 0,
        "attempts": 0,
        "complete": total_jobs == 0,
        "failed_terminal_jobs": 0,
        "fraction": 1.0 if total_jobs == 0 else 0.0,
        "in_flight_jobs": 0,
        "promoted_jobs": 0,
        "retryable_jobs": 0,
        "schema": "immer.o1-cartography-coverage/v1",
        "succeeded_jobs": 0,
        "terminal_jobs": 0,
        "total_jobs": total_jobs,
        "uncovered_jobs": total_jobs,
    }


def status_cartography(
    root: str | os.PathLike[str],
    *,
    runtime_factory: RuntimeFactory | None = None,
    atlas_factory: AtlasFactory = SemanticWeightAtlas,
    stream_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Authenticate persisted scheduler/atlas state and return exact coverage."""

    root_path = _plain_root(root, create=False)
    with _root_lock(root_path):
        manifest, body = _load_manifest(root_path)
        scheduler = _scheduler(
            root_path, body, create=False, stream_factory=stream_factory
        )
        coverage = (
            _empty_coverage(len(_manifest_jobs(body)))
            if scheduler is None
            else scheduler.coverage().to_document()
        )
        atlas_path = _atlas_path(root_path, create=False)
        atlas_document = None
        atlas_revision = None
        if atlas_path is not None:
            runtime = _open_runtime(body, runtime_factory)
            try:
                atlas = _open_atlas(
                    root_path,
                    body,
                    runtime,
                    create=False,
                    atlas_factory=atlas_factory,
                )
                assert atlas is not None
                atlas_document = atlas.coverage_matrix()
                atlas_revision = atlas.revision().to_document()
            finally:
                runtime.close()
        return {
            "atlas": atlas_document,
            "atlas_revision": atlas_revision,
            "atlas_receipt_sha256s": sorted(
                outcome.atlas_receipt_sha256
                for outcome in (() if scheduler is None else scheduler.outcomes)
                if outcome.atlas_receipt_sha256 is not None
            ),
            "coverage": coverage,
            "manifest_sha256": manifest["sha256"],
        }


def _query_document(result: Any) -> dict[str, Any]:
    return {
        "measurements": [value.to_document() for value in result.measurements],
        "promotions": [value.to_document() for value in result.promotions],
        "replicas": [value.to_document() for value in result.replicas],
    }


def query_cartography(
    root: str | os.PathLike[str],
    *,
    coordinate_sha256: str | None = None,
    prompt_sha256: str | None = None,
    runtime_factory: RuntimeFactory | None = None,
    atlas_factory: AtlasFactory = SemanticWeightAtlas,
) -> dict[str, Any]:
    """Authenticate the atlas and query exactly one coordinate or prompt index."""

    if (coordinate_sha256 is None) == (prompt_sha256 is None):
        raise O1CartographyCliError("query requires exactly one coordinate or prompt")
    root_path = _plain_root(root, create=False)
    with _root_lock(root_path):
        manifest, body = _load_manifest(root_path)
        if _atlas_path(root_path, create=False) is None:
            return {
                "manifest_sha256": manifest["sha256"],
                "measurements": [],
                "promotions": [],
                "replicas": [],
            }
        runtime = _open_runtime(body, runtime_factory)
        try:
            atlas = _open_atlas(
                root_path,
                body,
                runtime,
                create=False,
                atlas_factory=atlas_factory,
            )
            assert atlas is not None
            if coordinate_sha256 is not None:
                result = atlas.query_by_coordinate(
                    _sha(coordinate_sha256, "coordinate SHA-256")
                )
            else:
                prompt = _sha(prompt_sha256, "prompt SHA-256")
                signatures = (
                    {
                        spec.probe_identity.prompt_signature
                        for _job, spec in _manifest_jobs(body)
                    }
                    if prompt == body["prompt"]["sha256"]
                    else set()
                )
                results = [
                    atlas.query_by_prompt_signature(value) for value in signatures
                ]
                measurements = {
                    value.sha256: value
                    for result in results
                    for value in result.measurements
                }
                promotions = {
                    value.sha256: value
                    for result in results
                    for value in result.promotions
                }
                replicas = {
                    value.sha256: value
                    for result in results
                    for value in result.replicas
                }
                result = SimpleNamespace(
                    measurements=tuple(
                        measurements[key] for key in sorted(measurements)
                    ),
                    promotions=tuple(promotions[key] for key in sorted(promotions)),
                    replicas=tuple(replicas[key] for key in sorted(replicas)),
                )
            return {"manifest_sha256": manifest["sha256"], **_query_document(result)}
        finally:
            runtime.close()


def _token_ids(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "token IDs must be comma-separated integers"
        ) from exc
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("token IDs must be non-negative")
    return values


def _nonnegative_int_arg(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def _nonnegative_float_arg(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return value


def _positive_float_arg(raw: str) -> float:
    value = _nonnegative_float_arg(raw)
    if value == 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _jobs_input(args: argparse.Namespace) -> list[Mapping[str, Any]]:
    values: list[Any] = []
    if args.jobs_json:
        source = Path(args.jobs_json).expanduser().absolute()
        if any(
            marker in part.casefold()
            for part in source.parts
            for marker in ("holdout", "heldout", "hold-out")
        ):
            raise O1CartographyCliError("holdout paths are excluded from cartography")
        document = _strict_json_bytes(
            _stable_regular_bytes(source, "job frontier"), "job frontier"
        )
        if isinstance(document, Mapping) and set(document) == {"jobs"}:
            document = document["jobs"]
        if not isinstance(document, list):
            raise O1CartographyCliError("job frontier JSON root must be a list")
        values.extend(document)
    for raw in args.job or ():
        values.append(_strict_json_bytes(raw.encode("utf-8"), "--job"))
    if not values or any(not isinstance(value, Mapping) for value in values):
        raise O1CartographyCliError("at least one object-valued job is required")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="seal a finite question-only run")
    prepare.add_argument("--root", required=True)
    prepare.add_argument("--bundle-root", required=True)
    prepare.add_argument("--repo-id", required=True)
    prepare.add_argument("--revision", required=True)
    prepare.add_argument("--bundle-fingerprint", required=True)
    prepare.add_argument("--bundle-manifest-sha256", required=True)
    prepare.add_argument("--code-revision", required=True)
    prepare.add_argument("--prompt-token-ids", required=True, type=_token_ids)
    prepare.add_argument("--prompt-sha256", required=True)
    inputs = prepare.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--jobs-json")
    inputs.add_argument("--job", action="append")
    prepare.add_argument("--seed", type=_nonnegative_int_arg, default=0)
    prepare.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
    prepare.add_argument(
        "--compute-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    prepare.add_argument("--max-resident-mb", type=_positive_float_arg, default=1024.0)
    prepare.add_argument(
        "--source-budget-mb",
        type=_positive_float_arg,
        default=1_048_576.0,
        help="cumulative local range-read budget",
    )
    prepare.add_argument("--allow-nonofficial-config", action="store_true")

    run = commands.add_parser("run", help="run a bounded resumable probe slice")
    run.add_argument("--root", required=True)
    run.add_argument("--max-jobs", type=_nonnegative_int_arg, default=1)
    run.add_argument(
        "--max-seconds",
        type=_nonnegative_float_arg,
        default=0.0,
        help="wall-time bound; 0 disables it",
    )

    status = commands.add_parser("status", help="authenticate and print coverage")
    status.add_argument("--root", required=True)

    query = commands.add_parser("query", help="query the authenticated atlas")
    query.add_argument("--root", required=True)
    selector = query.add_mutually_exclusive_group(required=True)
    selector.add_argument("--coordinate-sha256")
    selector.add_argument("--prompt-sha256")
    return parser


def _dispatch(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "prepare":
        pin = ModelPin(
            repo_id=args.repo_id,
            revision=args.revision,
            bundle_fingerprint=_sha(args.bundle_fingerprint, "bundle fingerprint"),
            bundle_manifest_sha256=_sha(
                args.bundle_manifest_sha256, "bundle manifest SHA-256"
            ),
            code_revision=_code_revision(args.code_revision),
        )
        return prepare_manifest(
            args.root,
            bundle_root=args.bundle_root,
            model_pin=pin,
            prompt_token_ids=args.prompt_token_ids,
            prompt_sha256=args.prompt_sha256,
            jobs=_jobs_input(args),
            code_revision=args.code_revision,
            seed=args.seed,
            runtime={
                "compute_dtype": args.compute_dtype,
                "device": args.device,
                "max_resident_bytes": int(args.max_resident_mb * 1024**2),
                "require_official_config": not args.allow_nonofficial_config,
                "source_budget_mb": args.source_budget_mb,
            },
        )
    if args.command == "run":
        return run_cartography(
            args.root,
            max_jobs=args.max_jobs,
            max_seconds=args.max_seconds,
        )
    if args.command == "status":
        return status_cartography(args.root)
    if args.command == "query":
        return query_cartography(
            args.root,
            coordinate_sha256=args.coordinate_sha256,
            prompt_sha256=args.prompt_sha256,
        )
    raise AssertionError(f"unknown command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        document = _dispatch(args)
    except Exception as exc:
        sys.stderr.write(
            _canonical({"error": f"{type(exc).__name__}:{exc}"}).decode("utf-8") + "\n"
        )
        return 2
    sys.stdout.write(_canonical(document).decode("utf-8") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
