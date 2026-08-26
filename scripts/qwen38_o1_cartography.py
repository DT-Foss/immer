#!/usr/bin/env python3
"""Run resumable O1 cartography over a local causal Qwen3.8 bundle."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
import fcntl
import hashlib
import inspect
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

import numpy as np

from immer.runtimes.o1_state import O1Cartographer, ProbeJob, ProbeTarget
from immer.runtimes.o1_state.plasticity import LearningStream
from immer.runtimes.ooe.cartography import OoeCartographyBridge
from immer.runtimes.ooe.controller import (
    CONTROLLER_STATE_NAME,
    ControllerMetrics,
    OoeController,
    OoeControllerIntegrityError,
)
from immer.runtimes.ooe.crystal import CrystalStore
from immer.runtimes.ooe.bvn_crystals import (
    BirkhoffCrystalBank,
    BirkhoffCrystalIntegrityError,
)
from immer.runtimes.ooe.compute_crystals import MARKOV_FLOAT64, ComputeCrystalBank
from immer.runtimes.ooe.compute_graph import ComputeOperatorGraph
from immer.runtimes.ooe.operator_harvester import (
    ContinuousOperatorHarvester,
    HarvesterConfig,
    SingleBatchContextualProvider,
    contextual_observations_from_probe_result,
    probe_result_context_cursor,
)
from immer.runtimes.qwen3_8 import (
    GraphRevision,
    HiddenSketchProjection,
    LogicalModelIdentity,
    MeasurementReceipt,
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


MANIFEST_SCHEMA_V1 = "immer.qwen3.8-o1-cartography-run/v1"
MANIFEST_SCHEMA = "immer.qwen3.8-o1-cartography-run/v2"
MANIFEST_NAME = "manifest.json"
SCHEDULER_NAME = "scheduler.json"
ATLAS_NAME = "atlas"
O1_STATE_NAME = "o1-state.pt"
OOE_NAME = "ooe"
OPERATOR_COMPUTE_NAME = "operator-compute"
OOE_PROMOTION_STATE_NAME = "qwen-o1-cartography-promotion-transaction"
OOE_PROMOTION_TRANSACTION_SCHEMA = (
    "immer.qwen3.8-o1-cartography-ooe-promotion-transaction/v1"
)
_MAX_DOCUMENT_BYTES = 64 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CODE_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_FORBIDDEN_JOB_KEYS = frozenset(
    {"answer", "completion", "gold", "label", "output", "prompt", "question"}
)
_PROMPT_DEFAULT_FIELDS = frozenset(
    {
        "family_sha256",
        "label_evidence_sha256",
        "label_source_sha256",
        "question_sha256",
        "semantic_label",
    }
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
    if name not in {
        MANIFEST_NAME,
        SCHEDULER_NAME,
        ATLAS_NAME,
        O1_STATE_NAME,
        OOE_NAME,
        OPERATOR_COMPUTE_NAME,
    }:
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
        "label_evidence_sha256",
        "label_source_sha256",
        "native_head_crsa",
        "prompt_sha256",
        "prompt_token_ids",
        "question_sha256",
        "semantic_label",
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
        semantic_label=value["semantic_label"],
        label_evidence_sha256=value["label_evidence_sha256"],
        native_head_crsa=_native_from_record(value["native_head_crsa"]),
    )


def _target_for_spec(spec: ProbeSpec) -> ProbeTarget:
    coordinate = spec.coordinate
    if coordinate.head_index is not None:
        return ProbeTarget(coordinate.module, "head", coordinate.head_index)
    if coordinate.row_start is not None:
        return ProbeTarget(coordinate.module, "block", coordinate.row_start)
    return ProbeTarget(coordinate.module, "module", None)


def _prompt_record(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not {"sha256", "token_ids"} <= set(value):
        raise O1CartographyCliError("prompt registry entry is malformed")
    if set(value) - {"sha256", "token_ids", "spec_defaults"}:
        raise O1CartographyCliError("prompt registry entry has unknown fields")
    try:
        tokens = tuple(value["token_ids"])
    except TypeError as exc:
        raise O1CartographyCliError("prompt token IDs must be a sequence") from exc
    claimed = _sha(value["sha256"], "prompt SHA-256")
    if prompt_token_sha256(tokens) != claimed:
        raise O1CartographyCliError("prompt token IDs do not match prompt SHA-256")
    defaults = value.get("spec_defaults", {})
    if not isinstance(defaults, Mapping) or set(defaults) - _PROMPT_DEFAULT_FIELDS:
        raise O1CartographyCliError("prompt spec defaults are malformed")
    normalized = dict(defaults)
    for name in ("question_sha256", "family_sha256", "label_source_sha256"):
        if name in normalized:
            normalized[name] = _sha(normalized[name], name)
    label = normalized.get("semantic_label")
    evidence = normalized.get("label_evidence_sha256")
    if (label is None) != (evidence is None):
        raise O1CartographyCliError(
            "prompt semantic label and evidence SHA-256 must be supplied together"
        )
    if label is not None:
        if (
            not isinstance(label, str)
            or not label
            or label != label.strip()
            or "\x00" in label
            or len(label) > 512
        ):
            raise O1CartographyCliError("prompt semantic label is invalid")
        normalized["label_evidence_sha256"] = _sha(evidence, "label_evidence_sha256")
        if "label_source_sha256" not in normalized:
            raise O1CartographyCliError(
                "prompt semantic label requires an external label source"
            )
    return {
        "sha256": claimed,
        "spec_defaults": normalized,
        "token_ids": list(tokens),
    }


def _manifest_prompts(body: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if "prompts" in body:
        raw = body["prompts"]
        if not isinstance(raw, list) or not raw:
            raise O1CartographyCliError("run prompt registry is malformed")
        prompts = tuple(_prompt_record(value) for value in raw)
    else:
        legacy = body.get("prompt")
        if not isinstance(legacy, Mapping) or set(legacy) != {"sha256", "token_ids"}:
            raise O1CartographyCliError("legacy run prompt pin is malformed")
        prompts = (_prompt_record({**legacy, "spec_defaults": {}}),)
    if len({row["sha256"] for row in prompts}) != len(prompts):
        raise O1CartographyCliError("run prompt registry contains duplicates")
    return tuple(sorted(prompts, key=lambda row: row["sha256"]))


def _prepared_job(
    value: ProbeSpec | Mapping[str, Any],
    *,
    prompt_token_ids: tuple[int, ...],
    prompt_sha256: str,
    code_revision: str,
    model_pin_sha256: str,
    base_seed: int,
    prompt_defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    probe_family = "unlabeled"
    defaults = dict(prompt_defaults or {})
    target: ProbeTarget | None = None
    read_budget_bytes = None
    model_budget_seconds = None
    if isinstance(value, ProbeSpec):
        spec = value
        if any(getattr(spec, name) != member for name, member in defaults.items()):
            raise O1CartographyCliError("probe spec conflicts with its prompt defaults")
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
            "label_evidence_sha256",
            "label_source_sha256",
            "native_head_crsa",
            "prompt_sha256",
            "prompt_token_ids",
            "question_sha256",
            "semantic_label",
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
            kwargs.update(defaults)
            for name in (
                "question_sha256",
                "family_sha256",
                "label_source_sha256",
                "semantic_label",
                "label_evidence_sha256",
            ):
                if name in raw_spec:
                    if name in defaults and raw_spec[name] != defaults[name]:
                        raise O1CartographyCliError(
                            "job spec conflicts with its prompt defaults"
                        )
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
    if probe_family == "unlabeled" and spec.semantic_label is not None:
        probe_family = spec.semantic_label
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
    prompt_token_ids: Sequence[int] | None = None,
    prompt_sha256: str | None = None,
    prompts: Sequence[Mapping[str, Any]] | None = None,
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
    if prompts is None:
        if prompt_token_ids is None or prompt_sha256 is None:
            raise O1CartographyCliError("single-prompt prepare requires tokens and SHA")
        registry = (
            _prompt_record(
                {
                    "sha256": prompt_sha256,
                    "spec_defaults": {},
                    "token_ids": list(prompt_token_ids),
                }
            ),
        )
    else:
        if prompt_token_ids is not None or prompt_sha256 is not None:
            raise O1CartographyCliError(
                "single-prompt arguments and prompt registry are mutually exclusive"
            )
        try:
            registry = tuple(_prompt_record(value) for value in prompts)
        except TypeError as exc:
            raise O1CartographyCliError("prompt registry must be a sequence") from exc
        if not registry or len({row["sha256"] for row in registry}) != len(registry):
            raise O1CartographyCliError(
                "prompt registry must be non-empty and duplicate-free"
            )
        registry = tuple(sorted(registry, key=lambda row: row["sha256"]))
    if not jobs or len(jobs) * len(registry) > 1_000_000:
        raise O1CartographyCliError("finite job frontier must be non-empty and bounded")
    if len(registry) > 1 and any(
        isinstance(value, ProbeSpec)
        or (
            isinstance(value, Mapping)
            and isinstance(value.get("spec", value), Mapping)
            and "prompt_token_ids" in value.get("spec", value)
        )
        for value in jobs
    ):
        raise O1CartographyCliError(
            "multi-prompt frontiers require prompt-independent job templates"
        )
    prepared = []
    for prompt in registry:
        for value in jobs:
            prepared.append(
                _prepared_job(
                    value,
                    prompt_token_ids=tuple(prompt["token_ids"]),
                    prompt_sha256=prompt["sha256"],
                    code_revision=code,
                    model_pin_sha256=model_pin.sha256,
                    base_seed=seed,
                    prompt_defaults=prompt["spec_defaults"],
                )
            )
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
        "prompts": list(registry),
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
        or document.get("schema") not in {MANIFEST_SCHEMA_V1, MANIFEST_SCHEMA}
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
        "runtime",
        "scheduler",
    }
    prompt_field = "prompts" if document.get("schema") == MANIFEST_SCHEMA else "prompt"
    required.add(prompt_field)
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
    if (
        not isinstance(bundle, Mapping)
        or set(bundle) != {"manifest_sha256", "root"}
        or bundle.get("manifest_sha256") != pin.bundle_manifest_sha256
    ):
        raise O1CartographyCliError("run bundle pin is malformed")
    _manifest_prompts(body)
    _manifest_jobs(body)
    return document, body


def _manifest_jobs(body: Mapping[str, Any]) -> tuple[tuple[ProbeJob, ProbeSpec], ...]:
    raw_jobs = body.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise O1CartographyCliError("run job frontier is malformed")
    code = body["code_revision"]
    model_pin_sha = body["model_pin_sha256"]
    prompts = {row["sha256"]: row for row in _manifest_prompts(body)}
    rows: list[tuple[ProbeJob, ProbeSpec]] = []
    for raw in raw_jobs:
        if not isinstance(raw, Mapping) or set(raw) != {"job", "spec"}:
            raise O1CartographyCliError("run job entry is malformed")
        job = ProbeJob.from_document(raw["job"])
        spec = _spec_from_record(raw["spec"])
        if (
            job.code_pin != code
            or job.model_pin != model_pin_sha
            or job.prompt_sha256 != spec.prompt_sha256
            or job.layer != spec.coordinate.layer
            or job.target.module != spec.coordinate.module
            or job.intervention != spec.intervention_mode
            or spec.code_revision != code
        ):
            raise O1CartographyCliError("run job/spec identity is inconsistent")
        prompt = prompts.get(spec.prompt_sha256)
        if prompt is None or list(spec.prompt_token_ids) != prompt["token_ids"]:
            raise O1CartographyCliError("run job references an unsealed prompt")
        if any(
            getattr(spec, name) != value
            for name, value in prompt["spec_defaults"].items()
        ):
            raise O1CartographyCliError("run job differs from its prompt defaults")
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
        prompt_length = max(len(row["token_ids"]) for row in _manifest_prompts(body))
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


def _stream(
    stream_factory: Callable[..., Any] | None,
    *,
    seed: int,
    sidecar: Path,
) -> Any:
    if stream_factory is None:
        return LearningStream(seed=seed, sidecar=sidecar)
    try:
        signature = inspect.signature(stream_factory)
    except (TypeError, ValueError) as exc:
        raise O1CartographyCliError(
            "stream factory must expose an inspectable call signature"
        ) from exc
    parameters = signature.parameters
    accepts_kwargs = any(
        value.kind is inspect.Parameter.VAR_KEYWORD for value in parameters.values()
    )
    kwargs = {}
    if accepts_kwargs or "seed" in parameters:
        kwargs["seed"] = seed
    if accepts_kwargs or "sidecar" in parameters:
        kwargs["sidecar"] = sidecar
    return stream_factory(**kwargs)


def _scheduler(
    root: Path,
    body: Mapping[str, Any],
    *,
    create: bool,
    stream_factory: Callable[..., Any] | None,
) -> O1Cartographer | None:
    state_path = _contained(root, SCHEDULER_NAME)
    seed = _uint(body["scheduler"]["seed"], "scheduler seed")
    stream = _stream(
        stream_factory,
        seed=seed,
        sidecar=_contained(root, O1_STATE_NAME),
    )
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


def _harvest_probe_context(
    root: Path,
    *,
    atlas: Any,
    result: Any,
) -> dict[str, Any] | None:
    """Feed real projected Qwen states into the persistent operator graph."""

    transitions = getattr(result, "contextual_hidden_transitions", ())
    if not isinstance(transitions, tuple) or not transitions:
        return None
    atlas_revision = atlas.revision()
    if not isinstance(atlas_revision, GraphRevision):
        raise O1CartographyCliError("Atlas returned an invalid contextual head")
    observations = contextual_observations_from_probe_result(
        result,
        atlas_revision=atlas_revision,
    )
    if not observations:
        return None
    measurement = getattr(result, "measurement", None)
    if not isinstance(measurement, MeasurementReceipt):
        raise O1CartographyCliError("contextual probe lost its measurement receipt")
    provider = SingleBatchContextualProvider(
        cursor=probe_result_context_cursor(measurement, atlas_revision),
        observations=observations,
    )
    compute_root = _contained(root, OPERATOR_COMPUTE_NAME)
    bank = ComputeCrystalBank(compute_root)
    graph = ComputeOperatorGraph(bank)
    harvester = ContinuousOperatorHarvester(
        atlas=atlas,
        provider=provider,
        graph=graph,
        config=HarvesterConfig(
            minimum_observations=3,
            minimum_fit_rows=4,
            max_observations_per_step=128,
            max_samples_per_group=1024,
            max_groups=4096,
            max_recent_receipts=65_536,
            graph_cas_retries=16,
        ),
    )
    harvested = harvester.step(limit=128)
    bvn_receipt_sha256s = []
    bvn_rejections = []
    bvn_bank = BirkhoffCrystalBank(bank)
    for promotion in harvested.promotions:
        if promotion.candidate.operator_kind != MARKOV_FLOAT64:
            continue
        crystal = bank.restore_crystal(promotion.edge.crystal_sha256)
        dimension = crystal.input_abi.trailing_shape[0]
        kernel = crystal.apply(np.eye(dimension, dtype=np.float64))
        try:
            bvn_publication = bvn_bank.publish(
                kernel,
                verifier_sha256=promotion.edge.verifier_sha256,
                evidence_sha256s=(promotion.edge.evidence_sha256,),
                tolerance=1e-10,
            )
        except BirkhoffCrystalIntegrityError:
            bvn_rejections.append(promotion.edge.sha256)
        else:
            bvn_receipt_sha256s.append(bvn_publication.receipt.sha256)
    return {
        "accepted_observations": harvested.accepted_observations,
        "atlas_revision": harvested.atlas_revision.to_document(),
        "candidate_statuses": [
            {
                "group_sha256": row.group_sha256,
                "operator_kind": row.operator_kind,
                "status": row.status,
            }
            for row in harvested.candidates
        ],
        "bvn_receipt_sha256s": sorted(bvn_receipt_sha256s),
        "bvn_rejected_edge_sha256s": sorted(bvn_rejections),
        "cursor_after": harvested.cursor_after,
        "graph_state_sha256": harvested.graph_state_sha256,
        "promotion_edge_sha256s": sorted(
            row.edge.sha256 for row in harvested.promotions
        ),
        "promotion_crystal_sha256s": sorted(
            row.candidate.crystal_sha256
            for row in harvested.promotions
            if row.candidate.crystal_sha256 is not None
        ),
        "rejections": [
            {
                "observation_receipt_sha256": row.observation_receipt_sha256,
                "reason": row.reason,
            }
            for row in harvested.rejections
        ],
        "state_sha256": harvested.state_sha256,
    }


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


def _promotion_transaction_document(
    *,
    phase: str,
    cartography_manifest_sha256: str,
    model_pin_sha256: str,
    weight_graph_revision_sha256: str,
    pre_manifest_generation: int,
    pre_manifest_sha256: str,
    prepared_controller_snapshot_sha256: str,
    committed_controller_snapshot_sha256: str | None,
) -> dict[str, Any]:
    if phase not in {"prepared", "committed"}:
        raise ValueError("promotion transaction phase is invalid")
    if (phase == "prepared") != (committed_controller_snapshot_sha256 is None):
        raise ValueError("promotion transaction phase/snapshot binding is invalid")
    body = {
        "cartography_manifest_sha256": _sha(
            cartography_manifest_sha256,
            "cartography manifest SHA-256",
        ),
        "committed_controller_snapshot_sha256": (
            None
            if committed_controller_snapshot_sha256 is None
            else _sha(
                committed_controller_snapshot_sha256,
                "committed controller snapshot SHA-256",
            )
        ),
        "model_pin_sha256": _sha(model_pin_sha256, "model pin SHA-256"),
        "phase": phase,
        "pre_manifest_generation": _uint(
            pre_manifest_generation,
            "pre-promotion manifest generation",
        ),
        "pre_manifest_sha256": _sha(
            pre_manifest_sha256,
            "pre-promotion manifest SHA-256",
        ),
        "prepared_controller_snapshot_sha256": _sha(
            prepared_controller_snapshot_sha256,
            "prepared controller snapshot SHA-256",
        ),
        "weight_graph_revision_sha256": _sha(
            weight_graph_revision_sha256,
            "weight graph revision SHA-256",
        ),
    }
    return {
        "body": body,
        "schema": OOE_PROMOTION_TRANSACTION_SCHEMA,
        "sha256": _digest(body),
    }


def _load_promotion_transaction(
    store: CrystalStore,
) -> tuple[dict[str, Any], str] | None:
    try:
        raw = store.restore_state(OOE_PROMOTION_STATE_NAME)
    except KeyError:
        return None
    document = _strict_json_bytes(raw, "OoE promotion transaction")
    if raw != _canonical(document):
        raise O1CartographyCliError(
            "OoE promotion transaction is not canonical JSON"
        )
    if (
        not isinstance(document, dict)
        or set(document) != {"body", "schema", "sha256"}
        or document.get("schema") != OOE_PROMOTION_TRANSACTION_SCHEMA
        or not isinstance(document.get("body"), Mapping)
        or document.get("sha256") != _digest(document["body"])
    ):
        raise O1CartographyCliError("OoE promotion transaction seal is invalid")
    body = document["body"]
    expected = {
        "cartography_manifest_sha256",
        "committed_controller_snapshot_sha256",
        "model_pin_sha256",
        "phase",
        "pre_manifest_generation",
        "pre_manifest_sha256",
        "prepared_controller_snapshot_sha256",
        "weight_graph_revision_sha256",
    }
    if set(body) != expected:
        raise O1CartographyCliError("OoE promotion transaction body is malformed")
    normalized = _promotion_transaction_document(
        phase=body["phase"],
        cartography_manifest_sha256=body["cartography_manifest_sha256"],
        model_pin_sha256=body["model_pin_sha256"],
        weight_graph_revision_sha256=body["weight_graph_revision_sha256"],
        pre_manifest_generation=body["pre_manifest_generation"],
        pre_manifest_sha256=body["pre_manifest_sha256"],
        prepared_controller_snapshot_sha256=(
            body["prepared_controller_snapshot_sha256"]
        ),
        committed_controller_snapshot_sha256=(
            body["committed_controller_snapshot_sha256"]
        ),
    )
    if normalized != document:
        raise O1CartographyCliError(
            "OoE promotion transaction reconstruction mismatch"
        )
    return normalized, hashlib.sha256(raw).hexdigest()


def _assert_promotion_transaction_bindings(
    document: Mapping[str, Any],
    *,
    cartography_manifest_sha256: str,
    model_pin_sha256: str,
    weight_graph_revision_sha256: str,
) -> None:
    body = document["body"]
    if (
        body["cartography_manifest_sha256"] != cartography_manifest_sha256
        or body["model_pin_sha256"] != model_pin_sha256
        or body["weight_graph_revision_sha256"]
        != weight_graph_revision_sha256
    ):
        raise O1CartographyCliError(
            "OoE promotion transaction belongs to another sealed run"
        )


def _publish_promotion_transaction(
    store: CrystalStore,
    document: Mapping[str, Any],
    *,
    expected_sha256: str | None,
) -> Any:
    return store.publish_state(
        OOE_PROMOTION_STATE_NAME,
        _canonical(document),
        expected_sha256=expected_sha256,
    )


@dataclass(slots=True)
class _OoeCartographyRuntime:
    root: Path
    store: CrystalStore
    controller: OoeController
    bridge: OoeCartographyBridge
    atlas_revision_membership: _AtlasRevisionMembership
    expected_snapshot_sha256: str | None
    ingested_evidence_sha256s: set[str]
    promotion_transaction: dict[str, Any] | None
    promotion_transaction_state_sha256: str | None
    promotion_required: bool
    recovery_receipt: Any | None


def _ooe_root_path(
    cartography_root: Path,
    configured: str | os.PathLike[str] | None,
) -> Path:
    if configured is None:
        return _contained(cartography_root, OOE_NAME)
    return Path(configured).expanduser().absolute()


class _AtlasRevisionMembership:
    """Exact historical Atlas heads attested by active sealed measurements."""

    def __init__(self, atlas: Any) -> None:
        self.atlas = atlas

    def authenticate_measurement(self, measurement: MeasurementReceipt) -> None:
        if not isinstance(measurement, MeasurementReceipt):
            raise TypeError("measurement must be a MeasurementReceipt")
        head_before = self.atlas.revision()
        if not isinstance(head_before, GraphRevision):
            raise O1CartographyCliError("Atlas returned an invalid graph revision")
        self.atlas.verify_or_raise()
        if self.atlas.revision() != head_before:
            raise O1CartographyCliError(
                "Atlas head changed while OoE evidence was authenticated"
            )
        result = self.atlas.query_by_prompt_signature(
            measurement.probe.prompt_signature
        )
        active = {
            value.sha256: value
            for value in result.measurements
            if isinstance(value, MeasurementReceipt)
            and value.model_pin == measurement.model_pin
            and value.coordinate == measurement.coordinate
            and value.probe == measurement.probe
        }
        if active.get(measurement.sha256) != measurement:
            raise O1CartographyCliError(
                "OoE Atlas revision evidence is not an active measurement"
            )
        head = self.atlas.revision()
        if head != head_before:
            raise O1CartographyCliError(
                "Atlas head changed during OoE measurement authentication"
            )
        revision = measurement.atlas_head_revision
        if not isinstance(head, GraphRevision) or revision.sequence > head.sequence:
            raise O1CartographyCliError(
                "measurement names a future Atlas graph revision"
            )
        if revision.sequence == head.sequence and revision != head:
            raise O1CartographyCliError(
                "measurement Atlas graph revision conflicts with the live head"
            )
        contains_revision = getattr(self.atlas, "contains_revision", None)
        if not callable(contains_revision):
            raise O1CartographyCliError(
                "Atlas does not expose authenticated revision membership"
            )
        if not bool(contains_revision(revision)):
            raise O1CartographyCliError(
                "measurement Atlas revision is absent from the authenticated journal"
            )

    def __call__(self, revision: GraphRevision) -> bool:
        if not isinstance(revision, GraphRevision):
            return False
        self.atlas.verify_or_raise()
        head = self.atlas.revision()
        if not isinstance(head, GraphRevision) or revision.sequence > head.sequence:
            return False
        contains_revision = getattr(self.atlas, "contains_revision", None)
        return callable(contains_revision) and bool(contains_revision(revision))


def _atlas_revision_verifier(
    atlas: Any,
    authenticated_measurements: Sequence[MeasurementReceipt] = (),
) -> _AtlasRevisionMembership:
    verifier = _AtlasRevisionMembership(atlas)
    for measurement in authenticated_measurements:
        verifier.authenticate_measurement(measurement)
    return verifier


def _open_ooe_runtime(
    root: Path,
    *,
    atlas: Any,
    measurements: Sequence[MeasurementReceipt],
    cartography_manifest_sha256: str,
) -> _OoeCartographyRuntime:
    if not measurements:
        raise ValueError("OoE runtime requires an authenticated measurement")
    measurement = measurements[0]
    managed_root = _plain_root(root, create=True)
    store = CrystalStore(managed_root)
    verifier = _atlas_revision_verifier(atlas, measurements)
    transaction_state = _load_promotion_transaction(store)
    transaction = None if transaction_state is None else transaction_state[0]
    transaction_state_sha256 = (
        None if transaction_state is None else transaction_state[1]
    )
    if transaction is not None:
        _assert_promotion_transaction_bindings(
            transaction,
            cartography_manifest_sha256=cartography_manifest_sha256,
            model_pin_sha256=measurement.model_pin.sha256,
            weight_graph_revision_sha256=measurement.weight_rail_revision.sha256,
        )
    recovery_receipt = None
    try:
        prior_state = store.restore_state(CONTROLLER_STATE_NAME)
    except KeyError:
        controller = OoeController(
            model_pin_sha256=measurement.model_pin.sha256,
            weight_graph_revision_sha256=measurement.weight_rail_revision.sha256,
            atlas_graph_revision=measurement.atlas_head_revision,
            crystal_store=store,
            atlas_revision_verifier=verifier,
        )
        expected_snapshot = None
    else:
        expected_snapshot = hashlib.sha256(prior_state).hexdigest()
        try:
            controller = OoeController.restore(
                crystal_store=store,
                atlas_revision_verifier=verifier,
                expected_model_pin_sha256=measurement.model_pin.sha256,
                expected_weight_graph_revision_sha256=(
                    measurement.weight_rail_revision.sha256
                ),
            )
        except OoeControllerIntegrityError as exc:
            transaction_body = (
                None if transaction is None else transaction["body"]
            )
            current_manifest = store.manifest()
            recoverable = (
                str(exc) == "CrystalStore manifest changed since snapshot"
                and transaction_body is not None
                and transaction_body["phase"] == "prepared"
                and expected_snapshot
                == transaction_body["prepared_controller_snapshot_sha256"]
                and current_manifest.generation
                > transaction_body["pre_manifest_generation"]
                and current_manifest.sha256
                != transaction_body["pre_manifest_sha256"]
            )
            if not recoverable:
                raise
            controller, recovery_receipt = OoeController.restore_recoverable(
                crystal_store=store,
                atlas_revision_verifier=verifier,
                expected_model_pin_sha256=measurement.model_pin.sha256,
                expected_weight_graph_revision_sha256=(
                    measurement.weight_rail_revision.sha256
                ),
                expected_old_manifest_generation=(
                    transaction_body["pre_manifest_generation"]
                ),
                expected_old_manifest_sha256=(
                    transaction_body["pre_manifest_sha256"]
                ),
                expected_old_state_sha256=(
                    transaction_body["prepared_controller_snapshot_sha256"]
                ),
            )
            expected_snapshot = recovery_receipt.new_state_sha256
    promotion_required = False
    if transaction is not None and transaction["body"]["phase"] == "prepared":
        current_manifest = store.manifest()
        transaction_body = transaction["body"]
        if (
            current_manifest.generation
            == transaction_body["pre_manifest_generation"]
            and current_manifest.sha256
            == transaction_body["pre_manifest_sha256"]
        ):
            promotion_required = True
    return _OoeCartographyRuntime(
        root=managed_root,
        store=store,
        controller=controller,
        bridge=OoeCartographyBridge(controller),
        atlas_revision_membership=verifier,
        expected_snapshot_sha256=expected_snapshot,
        ingested_evidence_sha256s=set(_controller_evidence_sha256s(controller)),
        promotion_transaction=transaction,
        promotion_transaction_state_sha256=transaction_state_sha256,
        promotion_required=promotion_required,
        recovery_receipt=recovery_receipt,
    )


def _controller_evidence_sha256s(controller: OoeController) -> frozenset[str]:
    """Read the controller's own canonical snapshot to deduplicate attempts."""

    document = json.loads(controller.snapshot_bytes())
    try:
        sites = document["body"]["sites"]
        values = {
            evidence
            for site in sites
            for row in site["history"]
            for evidence in row["feature"]["body"]["evidence_sha256s"]
        }
    except (KeyError, TypeError) as exc:  # the controller already authenticated this
        raise O1CartographyCliError("OoE controller history is malformed") from exc
    return frozenset(_sha(value, "OoE history evidence SHA-256") for value in values)


def _source_actions(
    scheduler: O1Cartographer,
    specs: Mapping[str, ProbeSpec],
) -> dict[str, tuple[str, str]]:
    current: dict[str, str] = {}
    result: dict[str, tuple[str, str]] = {}
    for outcome in scheduler.outcomes:
        if outcome.status != "succeeded" or outcome.atlas_receipt_sha256 is None:
            continue
        prompt_sha256 = specs[outcome.job_id].prompt_sha256
        source_action = current.get(prompt_sha256, "qwen_fallback")
        result[outcome.attempt_id] = (source_action, prompt_sha256)
        current[prompt_sha256] = "probe_coordinate"
    return result


def _active_outcome_measurement(
    outcome: Any,
    *,
    atlas: Any,
    spec: ProbeSpec,
    model_pin: ModelPin,
) -> Any:
    observation = outcome.observation_document()
    if observation is None:
        raise O1CartographyCliError("attached outcome lost its atlas observation")
    receipt_sha256 = _sha(
        outcome.atlas_receipt_sha256,
        "attached atlas receipt SHA-256",
    )
    if observation.get("atlas_receipt_sha256") != receipt_sha256:
        raise O1CartographyCliError("attached outcome and observation receipts differ")
    measurement = _find_reusable_measurement(atlas, spec, model_pin)
    if measurement is None or measurement.sha256 != receipt_sha256:
        raise O1CartographyCliError(
            "attached outcome has no exact active Atlas measurement"
        )
    return measurement


def _integrate_ooe_outcomes(
    *,
    root: Path,
    scheduler: O1Cartographer,
    atlas: Any,
    specs: Mapping[str, ProbeSpec],
    model_pin: ModelPin,
    cartography_manifest_sha256: str,
    outcomes: Sequence[Any] | None = None,
    session: _OoeCartographyRuntime | None = None,
    source_actions: Mapping[str, tuple[str, str]] | None = None,
) -> tuple[_OoeCartographyRuntime | None, list[dict[str, Any]]]:
    sources = (
        _source_actions(scheduler, specs)
        if source_actions is None
        else source_actions
    )
    candidates: list[tuple[Any, MeasurementReceipt, str, str]] = []
    selected = scheduler.outcomes if outcomes is None else outcomes
    for outcome in selected:
        if outcome.status != "succeeded" or outcome.atlas_receipt_sha256 is None:
            continue
        measurement = _active_outcome_measurement(
            outcome,
            atlas=atlas,
            spec=specs[outcome.job_id],
            model_pin=model_pin,
        )
        # Injected legacy/fake atlases remain supported.  The production path
        # starts only from the actual immutable MeasurementReceipt contract.
        if not isinstance(measurement, MeasurementReceipt):
            continue
        source_action, stream_sha256 = sources[outcome.attempt_id]
        candidates.append((outcome, measurement, source_action, stream_sha256))
    if not candidates:
        return session, []
    candidates.sort(
        key=lambda row: (
            row[1].atlas_head_revision.sequence,
            row[1].atlas_head_revision.event_sha256,
            row[0].attempt_id,
        )
    )
    if session is None:
        session = _open_ooe_runtime(
            root,
            atlas=atlas,
            measurements=tuple(row[1] for row in candidates),
            cartography_manifest_sha256=cartography_manifest_sha256,
        )
    else:
        for _, measurement, _, _ in candidates:
            session.atlas_revision_membership.authenticate_measurement(measurement)
    ingested_evidence = session.ingested_evidence_sha256s
    records: list[dict[str, Any]] = []
    for outcome, measurement, source_action, stream_sha256 in candidates:
        if outcome.attempt_id in ingested_evidence:
            continue
        observation = outcome.observation_document()
        assert observation is not None
        learning = session.bridge.ingest_authenticated(
            atlas,
            measurement,
            source_action=source_action,
            target_action="probe_coordinate",
            o1_surprise=outcome.surprise,
            o1_learning_progress=outcome.learning_progress,
            evidence_sha256s=(outcome.attempt_id,),
        )
        records.append(
            {
                "attempt": outcome.attempt,
                "attempt_id": outcome.attempt_id,
                "job_id": outcome.job_id,
                "learning_receipt": {
                    **learning.as_record(),
                    "sha256": learning.sha256,
                },
                "o1_learning_progress": outcome.learning_progress,
                "o1_surprise": outcome.surprise,
                "reused_atlas_proof": bool(
                    observation.get("reused_atlas_proof")
                ),
                "stream_sha256": stream_sha256,
            }
        )
        ingested_evidence.add(outcome.attempt_id)
    return session, records


def _finalize_ooe_runtime(
    session: _OoeCartographyRuntime,
    *,
    cartography_manifest_sha256: str,
    promote: bool,
) -> tuple[tuple[Any, ...], Any, Any | None]:
    """Two-phase Crystal promotion with a sealed crash-recovery intent."""

    publications: tuple[Any, ...] = ()
    transaction_publication = None
    if promote or session.promotion_required:
        pre_manifest = session.store.manifest()
        prepared_snapshot_sha256 = hashlib.sha256(
            session.controller.snapshot_bytes()
        ).hexdigest()
        prepared = _promotion_transaction_document(
            phase="prepared",
            cartography_manifest_sha256=cartography_manifest_sha256,
            model_pin_sha256=session.controller.model_pin_sha256,
            weight_graph_revision_sha256=(
                session.controller.weight_graph_revision_sha256
            ),
            pre_manifest_generation=pre_manifest.generation,
            pre_manifest_sha256=pre_manifest.sha256,
            prepared_controller_snapshot_sha256=prepared_snapshot_sha256,
            committed_controller_snapshot_sha256=None,
        )
        transaction_publication = _publish_promotion_transaction(
            session.store,
            prepared,
            expected_sha256=session.promotion_transaction_state_sha256,
        )
        session.promotion_transaction = prepared
        session.promotion_transaction_state_sha256 = (
            transaction_publication.payload_sha256
        )
        prepared_snapshot = session.controller.save_snapshot(
            expected_sha256=session.expected_snapshot_sha256,
        )
        if prepared_snapshot.payload_sha256 != prepared_snapshot_sha256:
            raise O1CartographyCliError(
                "prepared OoE controller snapshot changed before publication"
            )
        session.expected_snapshot_sha256 = prepared_snapshot.payload_sha256
        publications = session.bridge.promote_ready()
        snapshot = session.controller.save_snapshot(
            expected_sha256=session.expected_snapshot_sha256,
        )
        session.expected_snapshot_sha256 = snapshot.payload_sha256
        committed = _promotion_transaction_document(
            phase="committed",
            cartography_manifest_sha256=cartography_manifest_sha256,
            model_pin_sha256=session.controller.model_pin_sha256,
            weight_graph_revision_sha256=(
                session.controller.weight_graph_revision_sha256
            ),
            pre_manifest_generation=pre_manifest.generation,
            pre_manifest_sha256=pre_manifest.sha256,
            prepared_controller_snapshot_sha256=prepared_snapshot_sha256,
            committed_controller_snapshot_sha256=snapshot.payload_sha256,
        )
        transaction_publication = _publish_promotion_transaction(
            session.store,
            committed,
            expected_sha256=session.promotion_transaction_state_sha256,
        )
        session.promotion_transaction = committed
        session.promotion_transaction_state_sha256 = (
            transaction_publication.payload_sha256
        )
        session.promotion_required = False
        return publications, snapshot, transaction_publication

    snapshot = session.controller.save_snapshot(
        expected_sha256=session.expected_snapshot_sha256,
    )
    session.expected_snapshot_sha256 = snapshot.payload_sha256
    transaction = session.promotion_transaction
    if transaction is not None and transaction["body"]["phase"] == "prepared":
        body = transaction["body"]
        committed = _promotion_transaction_document(
            phase="committed",
            cartography_manifest_sha256=body["cartography_manifest_sha256"],
            model_pin_sha256=body["model_pin_sha256"],
            weight_graph_revision_sha256=body["weight_graph_revision_sha256"],
            pre_manifest_generation=body["pre_manifest_generation"],
            pre_manifest_sha256=body["pre_manifest_sha256"],
            prepared_controller_snapshot_sha256=(
                body["prepared_controller_snapshot_sha256"]
            ),
            committed_controller_snapshot_sha256=snapshot.payload_sha256,
        )
        transaction_publication = _publish_promotion_transaction(
            session.store,
            committed,
            expected_sha256=session.promotion_transaction_state_sha256,
        )
        session.promotion_transaction = committed
        session.promotion_transaction_state_sha256 = (
            transaction_publication.payload_sha256
        )
    return publications, snapshot, transaction_publication


def _empty_ooe_report(
    root: Path,
    *,
    qwen_probe_calls: int,
    reused_atlas_proofs: int,
) -> dict[str, Any]:
    return {
        "available": False,
        "controller_last_temporal_index": None,
        "controller_metrics": ControllerMetrics().to_dict(),
        "controller_snapshot": None,
        "learning_receipts": [],
        "learning_receipt_sha256s": [],
        "promotions": [],
        "promotion_transaction": None,
        "qwen_probe_calls": qwen_probe_calls,
        "recovery_receipt": None,
        "reused_atlas_proofs": reused_atlas_proofs,
        "root": str(root),
        "saved_qwen_forwards": 0,
        "site_identity_sha256s": [],
    }


def run_cartography(
    root: str | os.PathLike[str],
    *,
    max_jobs: int,
    max_seconds: float,
    ooe_root: str | os.PathLike[str] | None = None,
    runtime_factory: RuntimeFactory | None = None,
    probe_factory: ProbeFactory = Qwen38CartographyProbe,
    atlas_factory: AtlasFactory = SemanticWeightAtlas,
    stream_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Resume durable probe->atlas->receipt transactions until a chosen stop."""

    job_limit = _uint(max_jobs, "max_jobs")
    second_limit = _seconds(max_seconds, "max_seconds")
    root_path = _plain_root(root, create=False)
    ooe_path = _ooe_root_path(root_path, ooe_root)
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
            ooe_session, learning_receipts = _integrate_ooe_outcomes(
                root=ooe_path,
                scheduler=scheduler,
                atlas=atlas,
                specs=specs,
                model_pin=model_pin,
                cartography_manifest_sha256=manifest["sha256"],
            )
            stream_actions = {
                specs[outcome.job_id].prompt_sha256: "probe_coordinate"
                for outcome in scheduler.outcomes
                if outcome.status == "succeeded"
                and outcome.atlas_receipt_sha256 is not None
            }
            outcomes = []
            operator_harvest_receipts: list[dict[str, Any]] = []
            qwen_probe_calls = 0
            reused_atlas_proofs = 0
            started = time.monotonic()
            stop_reason = "max-jobs"
            while job_limit == 0 or len(outcomes) < job_limit:
                if second_limit and time.monotonic() - started >= second_limit:
                    stop_reason = "max-seconds"
                    break

                def execute(job: ProbeJob, _attempt: int):
                    nonlocal qwen_probe_calls, reused_atlas_proofs
                    spec = specs[job.job_id]
                    reusable = _find_reusable_measurement(atlas, spec, model_pin)
                    if reusable is not None:
                        reused_atlas_proofs += 1
                        return _observation(job, spec, reusable, reused=True)
                    qwen_probe_calls += 1
                    before = time.monotonic()
                    result = probe.execute(spec, atlas_head_revision=atlas.revision())
                    observation, source_bytes = _append_probe_result(
                        atlas,
                        result,
                        job=job,
                        spec=spec,
                        model_pin=model_pin,
                    )
                    harvested = _harvest_probe_context(
                        ooe_path,
                        atlas=atlas,
                        result=result,
                    )
                    if harvested is not None:
                        operator_harvest_receipts.append(harvested)
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
                    attached = scheduler.attach_atlas_receipt(
                        job_id=outcome.job_id,
                        attempt=outcome.attempt,
                        receipt_sha256=receipt,
                    )
                    spec = specs[outcome.job_id]
                    source_action = stream_actions.get(
                        spec.prompt_sha256,
                        "qwen_fallback",
                    )
                    stream_actions[spec.prompt_sha256] = "probe_coordinate"
                    ooe_session, new_learning = _integrate_ooe_outcomes(
                        root=ooe_path,
                        scheduler=scheduler,
                        atlas=atlas,
                        specs=specs,
                        model_pin=model_pin,
                        cartography_manifest_sha256=manifest["sha256"],
                        outcomes=(attached,),
                        session=ooe_session,
                        source_actions={
                            attached.attempt_id: (
                                source_action,
                                spec.prompt_sha256,
                            )
                        },
                    )
                    learning_receipts.extend(new_learning)
            elapsed = time.monotonic() - started
            ooe_report = _empty_ooe_report(
                ooe_path,
                qwen_probe_calls=qwen_probe_calls,
                reused_atlas_proofs=reused_atlas_proofs,
            )
            if ooe_session is not None:
                publications, snapshot, transaction_publication = (
                    _finalize_ooe_runtime(
                        ooe_session,
                        cartography_manifest_sha256=manifest["sha256"],
                        promote=bool(learning_receipts),
                    )
                )
                ooe_report = {
                    "available": True,
                    "controller_last_temporal_index": (
                        ooe_session.controller.last_temporal_index
                    ),
                    "controller_metrics": (
                        ooe_session.controller.metrics.to_dict()
                    ),
                    "controller_snapshot": asdict(snapshot),
                    "learning_receipts": learning_receipts,
                    "learning_receipt_sha256s": sorted(
                        row["learning_receipt"]["sha256"]
                        for row in learning_receipts
                    ),
                    "promotions": [asdict(value) for value in publications],
                    "promotion_transaction": (
                        None
                        if transaction_publication is None
                        else asdict(transaction_publication)
                    ),
                    "qwen_probe_calls": qwen_probe_calls,
                    "recovery_receipt": (
                        None
                        if ooe_session.recovery_receipt is None
                        else {
                            **ooe_session.recovery_receipt.to_dict(),
                            "sha256": ooe_session.recovery_receipt.sha256,
                        }
                    ),
                    "reused_atlas_proofs": reused_atlas_proofs,
                    "root": str(ooe_session.root),
                    "saved_qwen_forwards": (
                        ooe_session.controller.metrics.saved_qwen_forwards
                    ),
                    "site_identity_sha256s": list(
                        ooe_session.controller.site_identity_sha256s
                    ),
                }
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
                "ooe": ooe_report,
                "operator_harvest": {
                    "available": bool(operator_harvest_receipts),
                    "accepted_observations": sum(
                        row["accepted_observations"]
                        for row in operator_harvest_receipts
                    ),
                    "promotion_edge_sha256s": sorted(
                        edge
                        for row in operator_harvest_receipts
                        for edge in row["promotion_edge_sha256s"]
                    ),
                    "receipts": operator_harvest_receipts,
                    "root": str(_contained(ooe_path, OPERATOR_COMPUTE_NAME)),
                },
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
    semantic_label: str | None = None,
    runtime_factory: RuntimeFactory | None = None,
    atlas_factory: AtlasFactory = SemanticWeightAtlas,
) -> dict[str, Any]:
    """Authenticate the atlas and query exactly one coordinate or prompt index."""

    if (
        sum(
            value is not None
            for value in (coordinate_sha256, prompt_sha256, semantic_label)
        )
        != 1
    ):
        raise O1CartographyCliError(
            "query requires exactly one coordinate, prompt, or semantic label"
        )
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
            elif semantic_label is not None:
                result = atlas.query_by_semantic_label(semantic_label)
            else:
                prompt = _sha(prompt_sha256, "prompt SHA-256")
                signatures = {
                    spec.probe_identity.prompt_signature
                    for _job, spec in _manifest_jobs(body)
                    if spec.prompt_sha256 == prompt
                }
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


def _prompts_input(args: argparse.Namespace) -> list[Mapping[str, Any]] | None:
    if args.prompts_json is None:
        if args.prompt_sha256 is None:
            raise O1CartographyCliError(
                "--prompt-sha256 is required with --prompt-token-ids"
            )
        return None
    if args.prompt_sha256 is not None:
        raise O1CartographyCliError(
            "--prompt-sha256 cannot be combined with --prompts-json"
        )
    source = Path(args.prompts_json).expanduser().absolute()
    if any(
        marker in part.casefold()
        for part in source.parts
        for marker in ("holdout", "heldout", "hold-out")
    ):
        raise O1CartographyCliError("holdout paths are excluded from cartography")
    document = _strict_json_bytes(
        _stable_regular_bytes(source, "prompt registry"), "prompt registry"
    )
    if isinstance(document, Mapping) and set(document) == {"prompts"}:
        document = document["prompts"]
    if not isinstance(document, list) or any(
        not isinstance(value, Mapping) for value in document
    ):
        raise O1CartographyCliError("prompt registry JSON root must be a list")
    return list(document)


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
    prompt_inputs = prepare.add_mutually_exclusive_group(required=True)
    prompt_inputs.add_argument("--prompt-token-ids", type=_token_ids)
    prompt_inputs.add_argument("--prompts-json")
    prepare.add_argument("--prompt-sha256")
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
    run.add_argument(
        "--ooe-root",
        help="local CrystalStore root; defaults to <root>/ooe",
    )
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
    selector.add_argument("--semantic-label")
    return parser


def _dispatch(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "prepare":
        prompts = _prompts_input(args)
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
            prompt_token_ids=(None if prompts is not None else args.prompt_token_ids),
            prompt_sha256=(None if prompts is not None else args.prompt_sha256),
            prompts=prompts,
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
            ooe_root=args.ooe_root,
        )
    if args.command == "status":
        return status_cartography(args.root)
    if args.command == "query":
        return query_cartography(
            args.root,
            coordinate_sha256=args.coordinate_sha256,
            prompt_sha256=args.prompt_sha256,
            semantic_label=args.semantic_label,
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
