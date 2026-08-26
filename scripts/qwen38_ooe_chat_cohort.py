#!/usr/bin/env python3
"""Frozen five-item Qwen -> ResultCell -> Markov-OoE chat cohort.

The first four cold prompt transitions train ``qwen_fallback -> mount_organ``.
The fifth transition is never shown to the Markov controller: its independently
measured feature and charged ResultCell are used only for a zero-forward warm
action-generalization test.  Gold is accepted only by ``verify`` and is opened
only through the final benchmark layer.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Protocol

from immer.cognition.fertig.adapter import (
    CandidateVerificationStatus,
    FertigSolver,
    canonical_numeric_candidate,
)
from immer.contracts import ExecutionStatus, Request, Result
from immer.knowledge.livecausal import LiveGraph
from immer.runtimes.deepseek_v4.causal_weights import (
    CausalWeightMount,
    LogicalModelIdentity,
    TensorRangePlan,
)
from immer.runtimes.ooe.cartography import (
    authenticated_measurement_verifier_sha256,
)
from immer.runtimes.ooe.chat import result_from_document, result_to_document
from immer.runtimes.ooe.controller import (
    ActionExecution,
    ControllerConfig,
    OoeController,
    VerifiedTeacherTransition,
    WarmAccountingReceipt,
)
from immer.runtimes.ooe.crystal import CrystalStore, CrystalStoreAudit
from immer.runtimes.ooe.identity import canonical_json_bytes, require_sha256
from immer.runtimes.ooe.qwen_bridge import (
    ACTION_SCHEMA_SHA256,
    QwenOoeFeatureReceipt,
    feature_schema_sha256,
)
from immer.runtimes.ooe.result_cells import (
    FinalBenchmarkLayer,
    ProvenanceUnit,
    ResultCell,
    ResultCellBank,
    ResultCellBinding,
    ResultCellExecutor,
    ResultCellMissError,
    ResultCellParityReceipt,
    ResultCellParityVerifier,
    attach_cold_qwen_generation_receipt,
    artifact_sha256,
)
from immer.runtimes.qwen3_8 import (
    END_OF_TEXT_TOKEN_ID,
    IM_END_TOKEN_ID,
    Qwen38CausalChat,
    Qwen38Tokenizer,
    Qwen38WeightPager,
    prompt_token_sha256,
    verify_qwen38_causal_mount,
)
from immer.runtimes.qwen3_8.adapter import RESULT_CELL_GENERATION_POLICY_SCHEMA
from immer.runtimes.qwen3_8.semantic_atlas import (
    ATLAS_EDGE_SCHEMA,
    MEASUREMENT_RECEIPT_SCHEMA,
    GraphRevision,
    MeasurementReceipt,
    ModelPin,
    SemanticWeightAtlas,
)


INPUT_SCHEMA = "immer.qwen3.8-ooe-chat-inputs/v1"
GOLD_SCHEMA = "immer.qwen3.8-ooe-chat-gold/v1"
MANIFEST_SCHEMA = "immer.qwen3.8-ooe-chat-cohort-manifest/v1"
RESULT_SCHEMA = "immer.qwen3.8-ooe-chat-cohort-result/v1"
VERIFY_SCHEMA = "immer.qwen3.8-ooe-chat-cohort-verification/v1"
PROGRESS_SCHEMA = "immer.qwen3.8-ooe-chat-progress/v1"
EVALUATOR_SCHEMA = "immer.qwen3.8-ooe-chat-frozen-evaluator/v1"

SYSTEM_PROMPT = (
    "Solve the math problem internally. Return only #### followed by the numeric "
    "answer. Do not show work."
)
SYSTEM_PROMPT_SHA256 = (
    "e0b250eb83fad627f40d1aada3dc7b752ead93e9d11c237a2a90955042a45f2a"
)
ITEM_IDS = (
    "gsm8k-test-0547-8f91b28073cd6f0c",
    "gsm8k-test-0810-8eead47c2f30e585",
    "gsm8k-test-0898-db3f1b79ca67b08b",
    "gsm8k-test-0931-7a0aa7b235d5cac2",
    "gsm8k-test-1240-a316335b6f4ebc29",
)
QUESTION_SHA256S = (
    "eee04241756a601db97caada1e26d6717b974e1f884c0d793558bc645c3cfee4",
    "26b92d9e2e3a41b4b49eb3b173345e2336b9a2d1f9a0eaea2d23c4789f711c96",
    "420b6c70f9754d97664d605b6c542daed5b505abccb027f96ae2a5d450716678",
    "1aea626b4cf5a0a1201649a84124a0e27d85396550541ccf926478dc913b13cb",
    "ffc98910f1fab3af61a727b3cf6eccc3c3c750b0a7ecdf537175315bc9947717",
)
RENDERED_PROMPT_SHA256S = (
    "bcbcd88ef61f37dfa3df6f0663615d142bbe80e73ca5226d0697c818751c947c",
    "6a3ac39132554e0e277cd08e5853dfb7b2983d6aa4ca7c5728cb452868aab630",
    "080b705672146e1db33469d99749eb6e3cd3062f4c57d8361cf9a195074ba637",
    "28bb11dfcca18e57444c56d9e7fcba1850e5daa98dd5a06a01159f1c0de61588",
    "6c1fb1f24908d44536d5a8691fb6f7dbaa5d9362b91361a9eb524a3e79e857af",
)
RENDERED_PROMPT_TOKEN_SHA256S = (
    "15366398dc24114bcf8f20e5b1aba8cfe3c2fe58e6250961ec97baa53fbb37b4",
    "0654d0924ec761239aaa94dfd053d06899874e7270213f34b2fa85b489118f6a",
    "b0a7c813e7e83b58e0b9c3022a90d6d61ec44b17bac9d651def947811a04bfd8",
    "a3d66a75368306e8a29965c7bed37dff2b9658d9e2574bd73a31dd821bb1fda0",
    "d641766dea269aa0d8f2eea9c54cda7f6a1b2d2593f0aebd499d19ac0209d8c0",
)
TRAIN_COUNT = 4
HOLDOUT_INDEX = 4
FEATURE_DIMENSIONS = 64
PLACEBO_SEED = 20260826

MAX_PROMPT_TOKENS = 1024
MAX_NEW_TOKENS = 8
MAX_CONTEXT_TOKENS = 2048
HEAD_BLOCK_ROWS = Qwen38WeightPager.DEFAULT_HEAD_BLOCK_ROWS
MAX_RESIDENT_MB = 384
SOURCE_BUDGET_MB = 2_097_152.0
DEVICE = "cpu"
COMPUTE_DTYPE = "bfloat16"
_MAX_JSON_BYTES = 64 * 1024 * 1024
_LOCK_NAME = ".qwen38-ooe-chat-cohort.lock"
_RAW_GOLD_KEYS = frozenset(("answer", "correct", "expected", "gold", "label"))


class ChatCohortError(RuntimeError):
    """The frozen real-chat cohort contract cannot be satisfied."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _path_identity(path: str | os.PathLike[str]) -> str:
    return _digest(
        {
            "absolute_path": str(Path(path).expanduser().absolute()),
            "schema": "immer.qwen3.8-ooe-chat-path/v1",
        }
    )


def _generation_policy() -> dict[str, Any]:
    return {
        "anchor_cache_enabled": False,
        "compute_dtype": COMPUTE_DTYPE,
        "decoding": "greedy",
        "device": DEVICE,
        "eos_token_ids": [IM_END_TOKEN_ID, END_OF_TEXT_TOKEN_ID],
        "head_block_rows": HEAD_BLOCK_ROWS,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "max_resident_bytes": MAX_RESIDENT_MB * 1024**2,
        "prefill_tokenwise": False,
        "schema": RESULT_CELL_GENERATION_POLICY_SCHEMA,
        "source_budget_mb": SOURCE_BUDGET_MB,
        "thinking": False,
    }


GENERATION_POLICY_SHA256 = _digest(_generation_policy())
EVALUATOR_QUALITY_CONTRACT_SHA256 = _digest(
    {
        "answer_parser": "optional #### prefix then complete canonical numeric",
        "comparison": "cold == warm == frozen holdout answer",
        "holdout_item_id": ITEM_IDS[HOLDOUT_INDEX],
        "schema": EVALUATOR_SCHEMA,
    }
)


def _stable_regular_bytes(path: Path, *, maximum: int = _MAX_JSON_BYTES) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise ChatCohortError(f"cannot inspect required file: {path}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ChatCohortError(f"required path is not a regular file: {path}")
    if before.st_size > maximum:
        raise ChatCohortError(f"file exceeds {maximum} bytes: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise ChatCohortError(f"file was truncated: {path}")
            chunks.append(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise ChatCohortError(f"file grew while reading: {path}")
        after = path.lstat()
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            raise ChatCohortError(f"file changed while reading: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _decode_jsonl(raw: bytes, *, label: str) -> dict[str, Any]:
    if not raw.endswith(b"\n"):
        raise ChatCohortError(f"{label} is not canonical JSONL")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ChatCohortError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value) + b"\n":
        raise ChatCohortError(f"{label} is not canonical JSON")
    return value


def _seal(schema: str, body: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(canonical_json_bytes(body))
    return {"body": normalized, "schema": schema, "sha256": _digest(normalized)}


def _unseal(document: object, *, schema: str, label: str) -> dict[str, Any]:
    if (
        not isinstance(document, Mapping)
        or set(document) != {"body", "schema", "sha256"}
        or document.get("schema") != schema
        or not isinstance(document.get("body"), Mapping)
    ):
        raise ChatCohortError(f"{label} envelope is invalid")
    body = dict(document["body"])
    if require_sha256(document.get("sha256"), field=f"{label} sha256") != _digest(
        body
    ):
        raise ChatCohortError(f"{label} SHA-256 mismatch")
    return body


def _document_bytes(document: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(document) + b"\n"


def _atomic_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if _stable_regular_bytes(path) != data:
            raise ChatCohortError(f"refusing to overwrite different bytes: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".pending", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path, follow_symlinks=False)
        except FileExistsError:
            if _stable_regular_bytes(path) != data:
                raise ChatCohortError("output appeared with different bytes")
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary_path.unlink(missing_ok=True)


@contextmanager
def _cohort_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ChatCohortError("crystal root must be a real directory")
    path = root / _LOCK_NAME
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ChatCohortError("cohort lock is not a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        linked = path.lstat()
        if (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino):
            raise ChatCohortError("cohort lock changed while acquiring it")
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@dataclass(frozen=True, slots=True)
class AtlasState:
    atlas: SemanticWeightAtlas
    measurements: Mapping[str, MeasurementReceipt]


def _plan_from_measurement(measurement: MeasurementReceipt) -> TensorRangePlan:
    coordinate = measurement.coordinate
    plan = TensorRangePlan(
        name=coordinate.tensor,
        dtype=coordinate.dtype,
        shape=coordinate.shape,
        shard=coordinate.shard,
        absolute_offset=coordinate.tensor_absolute_offset,
        length=coordinate.tensor_length,
    )
    if not coordinate.matches_plan(plan):
        raise ChatCohortError("Atlas coordinate failed tensor-plan reconstruction")
    return plan


def open_atlas(root: str | os.PathLike[str]) -> AtlasState:
    atlas_root = Path(root).expanduser().absolute() / "atlas"
    if not atlas_root.is_dir() or atlas_root.is_symlink():
        raise ChatCohortError("cartography root has no real Atlas directory")
    try:
        graph = LiveGraph(atlas_root)
        if not graph.store.verify(include_tombstones=True):
            raise ChatCohortError("Atlas segment verification failed")
        measurements: dict[str, MeasurementReceipt] = {}
        for segment_sha256 in graph.store.segments():
            for _sha256, _index, record in graph.store.iter_records(segment_sha256):
                document = record.get("document")
                if (
                    record.get("schema") == ATLAS_EDGE_SCHEMA
                    and record.get("edge_kind") == "primary"
                    and isinstance(document, Mapping)
                    and document.get("schema") == MEASUREMENT_RECEIPT_SCHEMA
                ):
                    measurement = MeasurementReceipt.from_document(document)
                    if measurement.sha256 != record.get("document_sha256"):
                        raise ChatCohortError("Atlas primary identity mismatch")
                    measurements[measurement.sha256] = measurement
        if not measurements:
            raise ChatCohortError("Atlas has no measurements")
        pins = {row.model_pin.sha256: row.model_pin for row in measurements.values()}
        if len(pins) != 1:
            raise ChatCohortError("Atlas spans multiple ModelPins")
        plans: dict[str, TensorRangePlan] = {}
        for measurement in measurements.values():
            plan = _plan_from_measurement(measurement)
            prior = plans.get(plan.name)
            if prior is not None and prior != plan:
                raise ChatCohortError("Atlas tensor-plan conflict")
            plans[plan.name] = plan
        atlas = SemanticWeightAtlas(
            graph,
            model_pin=next(iter(pins.values())),
            tensor_plans=tuple(plans.values()),
        )
        atlas.verify_or_raise()
    except ChatCohortError:
        raise
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise ChatCohortError("cannot authenticate semantic Atlas") from exc
    return AtlasState(atlas=atlas, measurements=measurements)


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    rendered_prompt_sha256: str
    rendered_prompt_token_sha256: str

    def __post_init__(self) -> None:
        for field in ("rendered_prompt_sha256", "rendered_prompt_token_sha256"):
            object.__setattr__(self, field, require_sha256(getattr(self, field), field=field))


@dataclass(frozen=True, slots=True)
class Finalized:
    result: Result
    judgment: Mapping[str, Any]
    status: str


class ChatRuntime(Protocol):
    atlas_state: AtlasState
    model_pin: ModelPin
    tokenizer_sha256: str
    bundle_receipt: Mapping[str, Any]
    generation_policy_sha256: str

    def render(self, question: str) -> RenderedPrompt: ...

    def generate(self, question: str, binding: ResultCellBinding) -> Result: ...

    def adjudicate(self, question: str, qwen_result: Result, *, warm: bool) -> Finalized: ...

    def close(self) -> None: ...


RuntimeFactory = Callable[[argparse.Namespace, Mapping[str, Any] | None], ChatRuntime]


class DefaultChatRuntime:
    def __init__(self, args: argparse.Namespace, manifest: Mapping[str, Any] | None) -> None:
        self.atlas_state = open_atlas(args.cartography_root)
        self.model_pin = self.atlas_state.atlas.model_pin
        self.tokenizer = Qwen38Tokenizer(args.tokenizer, require_official=True)
        self.tokenizer_sha256 = hashlib.sha256(
            _stable_regular_bytes(Path(args.tokenizer).expanduser().absolute())
        ).hexdigest()
        self.generation_policy_sha256 = GENERATION_POLICY_SHA256
        model = LogicalModelIdentity(self.model_pin.repo_id, self.model_pin.revision)
        if manifest is None:
            with CausalWeightMount(
                args.bundle,
                model,
                budget_mb=SOURCE_BUDGET_MB,
            ) as mount:
                self.bundle_receipt = verify_qwen38_causal_mount(mount)
        else:
            # The frozen receipt was fully verified by prepare.  Qwen's lazy
            # runtime re-verifies the local bundle once before the first cold
            # generation; avoiding a second eager shard pass halves cold I/O.
            self.bundle_receipt = dict(manifest["body"]["bundle_receipt"])
        if (
            self.bundle_receipt["layout_fingerprint"]
            != self.model_pin.bundle_fingerprint
            or self.bundle_receipt["manifest_sha256"]
            != self.model_pin.bundle_manifest_sha256
            or GraphRevision.from_live_revision(
                tuple(self.bundle_receipt["graph_revision"])
            ).sha256
            != next(iter(self.atlas_state.measurements.values())).weight_rail_revision.sha256
        ):
            raise ChatCohortError("bundle identity differs from Atlas ModelPin/weight graph")
        self._chat = None
        if manifest is not None:
            self._chat = Qwen38CausalChat(
                args.bundle,
                args.tokenizer,
                system_prompt=SYSTEM_PROMPT,
                device=args.device,
                compute_dtype=args.compute_dtype,
                source_budget_mb=args.source_budget_mb,
                max_resident_bytes=int(args.max_resident_mb * 1024**2),
                max_prompt_tokens=MAX_PROMPT_TOKENS,
                max_new_tokens=args.max_new_tokens,
                max_context_tokens=MAX_CONTEXT_TOKENS,
                head_block_rows=HEAD_BLOCK_ROWS,
                result_cell_code_revision=self.model_pin.code_revision,
            )
        self._fertig = FertigSolver()

    def render(self, question: str) -> RenderedPrompt:
        rendered = self.tokenizer.render_no_thinking_prompt(SYSTEM_PROMPT, question)
        token_ids = self.tokenizer.encode(rendered)
        return RenderedPrompt(
            rendered_prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
            rendered_prompt_token_sha256=prompt_token_sha256(token_ids),
        )

    def generate(self, question: str, binding: ResultCellBinding) -> Result:
        if self._chat is None:
            raise ChatCohortError("prepare runtime cannot generate")
        result = self._chat.handle(Request("chat", question))
        if not isinstance(result, Result) or not result.ok:
            raise ChatCohortError("cold Qwen generation failed")
        evidence = result.evidence.get("result_cell_binding_receipt")
        from immer.runtimes.ooe.result_cells import qwen_result_binding_evidence

        if evidence != qwen_result_binding_evidence(binding):
            raise ChatCohortError("Qwen runtime binding receipt differs from manifest")
        return result

    def adjudicate(self, question: str, qwen_result: Result, *, warm: bool) -> Finalized:
        if not qwen_result.ok or not isinstance(qwen_result.output, str):
            raise ChatCohortError("FERTIG requires one successful Qwen candidate")
        verification = self._fertig.verify_candidate(question, qwen_result.output)
        status = verification.status.value
        output = (
            verification.expected
            if verification.status is CandidateVerificationStatus.MISMATCH
            else qwen_result.output
        )
        route_prefix = "ooe" if warm else "qwen"
        route = {
            "verified": f"{route_prefix}_verified",
            "abstained": f"{route_prefix}_verification_abstained",
            "mismatch": (
                "ooe_fertig_mismatch_override"
                if warm
                else "fertig_mismatch_override"
            ),
        }[status]
        judgment = verification.to_dict()
        final = Result(
            ExecutionStatus.OK,
            "qwen-fertig-chat",
            output=output,
            evidence={
                "receipt": {
                    "fertig_judgment_sha256": artifact_sha256(judgment),
                    "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                    "route": route,
                }
            },
        )
        return Finalized(final, judgment, status)

    def close(self) -> None:
        if self._chat is not None:
            self._chat.close()


def _runtime(
    args: argparse.Namespace,
    manifest: Mapping[str, Any] | None,
    runtime_factory: RuntimeFactory | None,
) -> ChatRuntime:
    runtime = (runtime_factory or DefaultChatRuntime)(args, manifest)
    for name in ("render", "generate", "adjudicate", "close"):
        if not callable(getattr(runtime, name, None)):
            raise TypeError(f"runtime must provide {name}()")
    if not isinstance(runtime.model_pin, ModelPin):
        raise TypeError("runtime ModelPin is invalid")
    require_sha256(runtime.tokenizer_sha256, field="runtime tokenizer_sha256")
    require_sha256(
        runtime.generation_policy_sha256,
        field="runtime generation_policy_sha256",
    )
    if not isinstance(runtime.atlas_state, AtlasState):
        raise TypeError("runtime atlas_state is invalid")
    return runtime


def _load_inputs(path: str | os.PathLike[str]) -> tuple[dict[str, Any], ...]:
    document = _decode_jsonl(
        _stable_regular_bytes(Path(path).expanduser().absolute()),
        label="chat inputs",
    )
    body = _unseal(document, schema=INPUT_SCHEMA, label="chat inputs")
    if set(body) != {"items", "system_prompt"} or body["system_prompt"] != SYSTEM_PROMPT:
        raise ChatCohortError("chat inputs protocol is invalid")
    items = body["items"]
    if not isinstance(items, list) or len(items) != len(ITEM_IDS):
        raise ChatCohortError("chat inputs must contain exactly five items")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(items):
        if not isinstance(raw, Mapping) or set(raw) != {
            "item_id",
            "measurement_sha256",
            "question",
        }:
            raise ChatCohortError("chat input row shape is invalid or contains gold")
        if any(key.lower() in _RAW_GOLD_KEYS for key in raw):
            raise ChatCohortError("prepare inputs contain forbidden gold")
        item_id = raw["item_id"]
        question = raw["question"]
        if item_id != ITEM_IDS[index] or not isinstance(question, str) or not question.strip():
            raise ChatCohortError("chat input order/question is invalid")
        question = question.strip()
        if hashlib.sha256(question.encode()).hexdigest() != QUESTION_SHA256S[index]:
            raise ChatCohortError("chat input question SHA-256 mismatch")
        normalized.append(
            {
                "item_id": item_id,
                "measurement_sha256": require_sha256(
                    raw["measurement_sha256"], field="measurement_sha256"
                ),
                "question": question,
            }
        )
    return tuple(normalized)


def _store_roots(
    cartography_root: str | os.PathLike[str],
    crystal_root: str | os.PathLike[str],
    organ_root: str | os.PathLike[str],
) -> dict[str, str]:
    paths = tuple(
        Path(value).expanduser().absolute()
        for value in (cartography_root, crystal_root, organ_root)
    )
    for path in paths:
        if path.is_symlink():
            raise ChatCohortError("cohort stores must not be symlink roots")
        if path.exists() and not path.is_dir():
            raise ChatCohortError("cohort store root must be a directory")
    resolved = tuple(path.resolve(strict=False) for path in paths)
    if len(set(resolved)) != 3:
        raise ChatCohortError("cartography, Crystal, and organ roots must be distinct")
    for index, left in enumerate(resolved):
        for right in resolved[index + 1 :]:
            if left in right.parents or right in left.parents:
                raise ChatCohortError("cohort stores must not contain one another")
    existing_inodes = [
        (path.stat().st_dev, path.stat().st_ino) for path in paths if path.exists()
    ]
    if len(existing_inodes) != len(set(existing_inodes)):
        raise ChatCohortError("cohort stores resolve to one backing directory")
    if resolved[1] == resolved[0] / "ooe":
        raise ChatCohortError("chat OoE must not reuse cartography OoE state")
    return {
        "cartography_root_sha256": _path_identity(paths[0]),
        "crystal_root_sha256": _path_identity(paths[1]),
        "organ_root_sha256": _path_identity(paths[2]),
    }


def prepare(
    args: argparse.Namespace,
    *,
    runtime_factory: RuntimeFactory | None = None,
) -> dict[str, Any]:
    items = _load_inputs(args.inputs)
    roots = _store_roots(args.cartography_root, args.crystal_root, args.organ_root)
    runtime = _runtime(args, None, runtime_factory)
    try:
        if runtime.generation_policy_sha256 != GENERATION_POLICY_SHA256:
            raise ChatCohortError("runtime generation policy differs from frozen policy")
        atlas = runtime.atlas_state.atlas
        if atlas.model_pin != runtime.model_pin:
            raise ChatCohortError("runtime ModelPin differs from Atlas")
        manifest_items: list[dict[str, Any]] = []
        site_sha256s: list[str] = []
        weight_revisions: set[str] = set()
        for index, item in enumerate(items):
            measurement = runtime.atlas_state.measurements.get(
                item["measurement_sha256"]
            )
            if measurement is None:
                raise ChatCohortError("input measurement is absent from Atlas")
            rendered = runtime.render(item["question"])
            if (
                rendered.rendered_prompt_sha256 != RENDERED_PROMPT_SHA256S[index]
                or rendered.rendered_prompt_token_sha256
                != RENDERED_PROMPT_TOKEN_SHA256S[index]
            ):
                raise ChatCohortError("rendered prompt identity differs from frozen cohort")
            if (
                measurement.model_pin != runtime.model_pin
                or measurement.probe.question_sha256 != QUESTION_SHA256S[index]
                or measurement.probe.token_sha256
                != rendered.rendered_prompt_token_sha256
            ):
                raise ChatCohortError("Atlas measurement is not prompt-bound")
            verifier = authenticated_measurement_verifier_sha256(atlas, measurement)
            binding = ResultCellBinding(
                model_pin=runtime.model_pin,
                tokenizer_sha256=runtime.tokenizer_sha256,
                question_sha256=QUESTION_SHA256S[index],
                rendered_prompt_sha256=rendered.rendered_prompt_sha256,
                rendered_prompt_token_sha256=rendered.rendered_prompt_token_sha256,
                system_prompt_sha256=SYSTEM_PROMPT_SHA256,
                generation_policy_sha256=GENERATION_POLICY_SHA256,
            )
            site_sha256 = QwenOoeFeatureReceipt.from_measurement(
                measurement,
                temporal_index=index,
                verifier_sha256s=(verifier,),
                evidence_sha256s=(measurement.evidence_sha256,),
                o1_surprise=0.0,
                o1_learning_progress=0.0,
            ).site_identity.sha256
            site_sha256s.append(site_sha256)
            weight_revisions.add(measurement.weight_rail_revision.sha256)
            manifest_items.append(
                {
                    "atlas_measurement_verifier_sha256": verifier,
                    "binding": binding.as_record(),
                    "binding_sha256": binding.sha256,
                    "item_id": item["item_id"],
                    "measurement_sha256": measurement.sha256,
                    "question": item["question"],
                    "question_sha256": QUESTION_SHA256S[index],
                    "site_identity_sha256": site_sha256,
                    "split": "train" if index < TRAIN_COUNT else "holdout",
                    "temporal_index": index,
                }
            )
        train_sites = set(site_sha256s[:TRAIN_COUNT])
        if len(train_sites) < 2:
            raise ChatCohortError("placebo cohort requires two trained Atlas sites")
        if site_sha256s[HOLDOUT_INDEX] not in train_sites:
            raise ChatCohortError("holdout site has no trained action kernel")
        if len(weight_revisions) != 1:
            raise ChatCohortError("cohort spans multiple weight graph revisions")
        bundle = runtime.bundle_receipt
        try:
            bundle_weight_revision = GraphRevision.from_live_revision(
                tuple(bundle["graph_revision"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ChatCohortError("runtime bundle receipt is invalid") from exc
        if (
            bundle.get("layout_fingerprint") != runtime.model_pin.bundle_fingerprint
            or bundle.get("manifest_sha256")
            != runtime.model_pin.bundle_manifest_sha256
            or bundle_weight_revision.sha256 != next(iter(weight_revisions))
        ):
            raise ChatCohortError("runtime bundle receipt differs from ModelPin/weights")
        body = {
            "action_schema_sha256": ACTION_SCHEMA_SHA256,
            "atlas_revision": atlas.revision().to_document(),
            "bundle_receipt": dict(runtime.bundle_receipt),
            "code_revision": runtime.model_pin.code_revision,
            "evaluator_quality_contract_sha256": (
                EVALUATOR_QUALITY_CONTRACT_SHA256
            ),
            "feature_schema_sha256": feature_schema_sha256(FEATURE_DIMENSIONS),
            "generation_policy": _generation_policy(),
            "generation_policy_sha256": GENERATION_POLICY_SHA256,
            "holdout_item_id": ITEM_IDS[HOLDOUT_INDEX],
            "items": manifest_items,
            "model_pin": runtime.model_pin.to_document(),
            "model_pin_sha256": runtime.model_pin.sha256,
            "ordered_item_ids": list(ITEM_IDS),
            "roots": roots,
            "system_prompt": SYSTEM_PROMPT,
            "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
            "tokenizer_sha256": runtime.tokenizer_sha256,
            "train_item_ids": list(ITEM_IDS[:TRAIN_COUNT]),
            "weight_graph_revision_sha256": next(iter(weight_revisions)),
        }
        document = _seal(MANIFEST_SCHEMA, body)
        _atomic_new(Path(args.manifest).expanduser().absolute(), _document_bytes(document))
        return document
    finally:
        runtime.close()


def _load_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    document = _decode_jsonl(
        _stable_regular_bytes(Path(path).expanduser().absolute()),
        label="chat manifest",
    )
    body = _unseal(document, schema=MANIFEST_SCHEMA, label="chat manifest")
    required = {
        "action_schema_sha256",
        "atlas_revision",
        "bundle_receipt",
        "code_revision",
        "evaluator_quality_contract_sha256",
        "feature_schema_sha256",
        "generation_policy",
        "generation_policy_sha256",
        "holdout_item_id",
        "items",
        "model_pin",
        "model_pin_sha256",
        "ordered_item_ids",
        "roots",
        "system_prompt",
        "system_prompt_sha256",
        "tokenizer_sha256",
        "train_item_ids",
        "weight_graph_revision_sha256",
    }
    if set(body) != required:
        raise ChatCohortError("chat manifest fields are invalid")
    pin = ModelPin.from_document(body["model_pin"])
    if (
        body["model_pin_sha256"] != pin.sha256
        or body["ordered_item_ids"] != list(ITEM_IDS)
        or body["train_item_ids"] != list(ITEM_IDS[:TRAIN_COUNT])
        or body["holdout_item_id"] != ITEM_IDS[HOLDOUT_INDEX]
        or body["system_prompt"] != SYSTEM_PROMPT
        or body["system_prompt_sha256"] != SYSTEM_PROMPT_SHA256
        or body["generation_policy"] != _generation_policy()
        or body["generation_policy_sha256"] != GENERATION_POLICY_SHA256
        or body["feature_schema_sha256"] != feature_schema_sha256(FEATURE_DIMENSIONS)
        or body["action_schema_sha256"] != ACTION_SCHEMA_SHA256
        or body["evaluator_quality_contract_sha256"]
        != EVALUATOR_QUALITY_CONTRACT_SHA256
    ):
        raise ChatCohortError("chat manifest frozen identities are stale")
    items = body["items"]
    if not isinstance(items, list) or len(items) != len(ITEM_IDS):
        raise ChatCohortError("chat manifest item count is invalid")
    for index, item in enumerate(items):
        expected_item_fields = {
            "atlas_measurement_verifier_sha256",
            "binding",
            "binding_sha256",
            "item_id",
            "measurement_sha256",
            "question",
            "question_sha256",
            "site_identity_sha256",
            "split",
            "temporal_index",
        }
        if (
            not isinstance(item, Mapping)
            or set(item) != expected_item_fields
            or item.get("item_id") != ITEM_IDS[index]
            or item.get("question_sha256") != QUESTION_SHA256S[index]
            or item.get("temporal_index") != index
            or item.get("split") != ("train" if index < TRAIN_COUNT else "holdout")
            or hashlib.sha256(str(item.get("question", "")).encode()).hexdigest()
            != QUESTION_SHA256S[index]
        ):
            raise ChatCohortError("chat manifest item identity is invalid")
        binding = ResultCellBinding.from_record(item["binding"])
        if (
            item.get("binding_sha256") != binding.sha256
            or binding.model_pin != pin
            or binding.tokenizer_sha256 != body["tokenizer_sha256"]
            or binding.question_sha256 != item["question_sha256"]
            or binding.system_prompt_sha256 != SYSTEM_PROMPT_SHA256
            or binding.generation_policy_sha256 != GENERATION_POLICY_SHA256
            or binding.rendered_prompt_sha256 != RENDERED_PROMPT_SHA256S[index]
            or binding.rendered_prompt_token_sha256
            != RENDERED_PROMPT_TOKEN_SHA256S[index]
            or body["code_revision"] != pin.code_revision
        ):
            raise ChatCohortError("chat manifest binding SHA-256 mismatch")
    return {"body": body, "schema": MANIFEST_SCHEMA, "sha256": document["sha256"]}


def _progress_name(manifest_sha256: str) -> str:
    return f"qwen38-ooe-chat-progress-{manifest_sha256[:24]}"


def _controller_name(manifest_sha256: str) -> str:
    return f"qwen38-ooe-chat-controller-{manifest_sha256[:24]}"


def _result_name(manifest_sha256: str) -> str:
    return f"qwen38-ooe-chat-result-{manifest_sha256[:24]}"


def _state_optional(store: CrystalStore, name: str) -> bytes | None:
    try:
        return store.restore_state(name)
    except KeyError:
        return None


def _new_progress(manifest_sha256: str) -> dict[str, Any]:
    return {
        "active_item_id": None,
        "cold_rows": [],
        "controller_snapshot_sha256": None,
        "manifest_sha256": manifest_sha256,
        "placebos": None,
        "schema": PROGRESS_SCHEMA,
        "warm": None,
        "warm_started": False,
    }


def _progress_bytes(body: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(_seal(PROGRESS_SCHEMA, body))


def _load_progress(raw: bytes, manifest_sha256: str) -> dict[str, Any]:
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ChatCohortError("chat progress is invalid JSON") from exc
    if canonical_json_bytes(document) != raw:
        raise ChatCohortError("chat progress is not canonical")
    body = _unseal(document, schema=PROGRESS_SCHEMA, label="chat progress")
    if set(body) != set(_new_progress(manifest_sha256)):
        raise ChatCohortError("chat progress fields are invalid")
    if (
        body.get("manifest_sha256") != manifest_sha256
        or body.get("schema") != PROGRESS_SCHEMA
        or not isinstance(body.get("cold_rows"), list)
        or len(body["cold_rows"]) > len(ITEM_IDS)
        or not isinstance(body.get("warm_started"), bool)
    ):
        raise ChatCohortError("chat progress belongs to another cohort")
    return body


def _save_progress(
    store: CrystalStore,
    name: str,
    body: Mapping[str, Any],
    *,
    expected_sha256: str | None,
) -> str:
    publication = store.publish_state(
        name,
        _progress_bytes(body),
        expected_sha256=expected_sha256,
    )
    return publication.payload_sha256


def _audit_record(audit: CrystalStoreAudit) -> dict[str, Any]:
    return {
        "clean": audit.clean,
        "generation": audit.generation,
        "manifest_sha256": audit.manifest_sha256,
        "missing_objects": list(audit.missing_objects),
        "orphan_objects": list(audit.orphan_objects),
        "staged_files": list(audit.staged_files),
        "tampered_objects": list(audit.tampered_objects),
        "unexpected_files": list(audit.unexpected_files),
        "valid_objects": list(audit.valid_objects),
    }


def _manifest_bindings(manifest: Mapping[str, Any]) -> tuple[ResultCellBinding, ...]:
    return tuple(
        ResultCellBinding.from_record(item["binding"])
        for item in manifest["body"]["items"]
    )


def _manifest_measurements(
    manifest: Mapping[str, Any], atlas_state: AtlasState
) -> tuple[MeasurementReceipt, ...]:
    measurements: list[MeasurementReceipt] = []
    for item in manifest["body"]["items"]:
        measurement = atlas_state.measurements.get(item["measurement_sha256"])
        if measurement is None:
            raise ChatCohortError("manifest measurement disappeared from Atlas")
        verifier = authenticated_measurement_verifier_sha256(
            atlas_state.atlas, measurement
        )
        if verifier != item["atlas_measurement_verifier_sha256"]:
            raise ChatCohortError("manifest Atlas verifier is stale")
        measurements.append(measurement)
    return tuple(measurements)


def _atlas_provenance(
    measurement: MeasurementReceipt,
    verifier_sha256: str,
) -> ProvenanceUnit:
    return ProvenanceUnit(
        name="atlas-measurement",
        verifier_sha256=verifier_sha256,
        evidence_sha256=measurement.evidence_sha256,
        receipt_sha256=measurement.sha256,
    )


def _feature_for_cell(
    measurement: MeasurementReceipt,
    *,
    temporal_index: int,
    atlas_verifier_sha256: str,
    cell: ResultCell,
) -> QwenOoeFeatureReceipt:
    pair_binding = cell.cold_generation_provenance_unit.pair_binding_sha256
    feature = QwenOoeFeatureReceipt.from_measurement(
        measurement,
        temporal_index=temporal_index,
        verifier_sha256s=(atlas_verifier_sha256, pair_binding),
        evidence_sha256s=(measurement.evidence_sha256, pair_binding),
        # This cohort has Atlas measurements but no separate O1 scheduler
        # outcome receipt.  Zero is the exact absence value; no learning signal
        # is invented to make the holdout route easier.
        o1_surprise=0.0,
        o1_learning_progress=0.0,
    )
    cell.assert_feature_bound(feature)
    return feature


def _cold_row(
    *,
    item: Mapping[str, Any],
    measurement: MeasurementReceipt,
    binding: ResultCellBinding,
    bank: ResultCellBank,
    runtime: ChatRuntime,
    generate: bool,
) -> dict[str, Any]:
    verifier = item["atlas_measurement_verifier_sha256"]
    try:
        cell = bank.restore(binding)
    except ResultCellMissError:
        if not generate:
            raise ChatCohortError(
                "cold generation outcome is uncertain; refusing a duplicate Qwen call"
            )
        raw = runtime.generate(item["question"], binding)
        cold_qwen = attach_cold_qwen_generation_receipt(raw, binding=binding)
        finalized = runtime.adjudicate(item["question"], cold_qwen, warm=False)
        provenance = _atlas_provenance(measurement, verifier)
        cell = ResultCell.from_cold(
            binding=binding,
            cold_qwen_result=cold_qwen,
            cold_final_result=finalized.result,
            cold_fertig_judgment=finalized.judgment,
            cold_fertig_status=finalized.status,
            evaluator_quality_contract_sha256=(
                EVALUATOR_QUALITY_CONTRACT_SHA256
            ),
            provenance_units=(provenance,),
        )
        bank.charge(cell)
        after_charge = getattr(runtime, "after_charge", None)
        if callable(after_charge):
            after_charge(item["item_id"], cell)
    finalized = runtime.adjudicate(
        item["question"],
        cell.cold_qwen_result,
        warm=False,
    )
    if (
        result_to_document(finalized.result)
        != result_to_document(cell.cold_final_result)
        or artifact_sha256(finalized.judgment)
        != cell.cold_fertig_judgment_sha256
        or finalized.status != cell.cold_fertig_status
    ):
        raise ChatCohortError("recovered cold cell differs from FERTIG adjudication")
    provenance = _atlas_provenance(measurement, verifier)
    if (
        provenance not in cell.provenance_units
        or cell.evaluator_quality_contract_sha256
        != EVALUATOR_QUALITY_CONTRACT_SHA256
    ):
        raise ChatCohortError("result cell lost Atlas provenance")
    feature = _feature_for_cell(
        measurement,
        temporal_index=item["temporal_index"],
        atlas_verifier_sha256=verifier,
        cell=cell,
    )
    if feature.site_identity.sha256 != item["site_identity_sha256"]:
        raise ChatCohortError("cold feature site differs from manifest")
    return {
        "cell_payload_sha256": cell.payload_sha256,
        "cold_fertig_judgment": dict(finalized.judgment),
        "cold_fertig_status": finalized.status,
        "feature": feature.to_document(),
        "feature_receipt_sha256": feature.sha256,
        "item_id": item["item_id"],
        "teacher_forward_count": cell.teacher_forward_count,
    }


def _validate_cold_rows(
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    measurements: Sequence[MeasurementReceipt],
    bindings: Sequence[ResultCellBinding],
    bank: ResultCellBank,
) -> tuple[tuple[ResultCell, QwenOoeFeatureReceipt], ...]:
    if len(rows) != len(ITEM_IDS):
        raise ChatCohortError("cold rows do not cover the five-item cohort")
    restored: list[tuple[ResultCell, QwenOoeFeatureReceipt]] = []
    for index, (item, row, measurement, binding) in enumerate(
        zip(
            manifest["body"]["items"],
            rows,
            measurements,
            bindings,
            strict=True,
        )
    ):
        if row.get("item_id") != item["item_id"]:
            raise ChatCohortError("cold row order differs from manifest")
        cell = bank.restore_payload(row["cell_payload_sha256"])
        if (
            cell.binding != binding
            or row.get("teacher_forward_count") != cell.teacher_forward_count
            or row.get("cold_fertig_status") != cell.cold_fertig_status
            or cell.evaluator_quality_contract_sha256
            != EVALUATOR_QUALITY_CONTRACT_SHA256
        ):
            raise ChatCohortError("cold row ResultCell binding/baseline mismatch")
        feature = QwenOoeFeatureReceipt.from_document(row["feature"])
        expected = _feature_for_cell(
            measurement,
            temporal_index=index,
            atlas_verifier_sha256=item["atlas_measurement_verifier_sha256"],
            cell=cell,
        )
        if feature != expected or row.get("feature_receipt_sha256") != feature.sha256:
            raise ChatCohortError("cold row feature reconstruction mismatch")
        if artifact_sha256(row["cold_fertig_judgment"]) != cell.cold_fertig_judgment_sha256:
            raise ChatCohortError("cold row FERTIG judgment mismatch")
        provenance = _atlas_provenance(
            measurement,
            item["atlas_measurement_verifier_sha256"],
        )
        if provenance not in cell.provenance_units:
            raise ChatCohortError("cold row Atlas provenance mismatch")
        restored.append((cell, feature))
    return tuple(restored)


def _controller_resolver(
    rows: Sequence[tuple[ResultCell, QwenOoeFeatureReceipt]],
) -> Callable[[QwenOoeFeatureReceipt], ResultCellBinding]:
    bindings = {feature.sha256: cell.binding for cell, feature in rows}

    def resolve(receipt: QwenOoeFeatureReceipt) -> ResultCellBinding:
        try:
            return bindings[receipt.sha256]
        except KeyError as exc:
            raise ChatCohortError("no ResultCell binding for feature") from exc

    return resolve


def _controller_history_features(controller: OoeController) -> frozenset[str]:
    document = json.loads(controller.snapshot_bytes())
    return frozenset(
        row["feature"]["sha256"]
        for site in document["body"]["sites"]
        for row in site["history"]
    )


def _open_controller(
    *,
    store: CrystalStore,
    bank: ResultCellBank,
    state_name: str,
    atlas: SemanticWeightAtlas,
    measurements: Sequence[MeasurementReceipt],
    cells: Sequence[tuple[ResultCell, QwenOoeFeatureReceipt]],
) -> tuple[OoeController, str | None]:
    executor = ResultCellExecutor(
        bank,
        _controller_resolver(cells),
    )
    prior = _state_optional(store, state_name)
    first = measurements[0]
    if prior is None:
        controller = OoeController(
            model_pin_sha256=first.model_pin.sha256,
            weight_graph_revision_sha256=first.weight_rail_revision.sha256,
            atlas_graph_revision=first.atlas_head_revision,
            crystal_store=store,
            atlas_revision_verifier=atlas.contains_revision,
            action_executors={"mount_organ": executor},
            config=ControllerConfig(
                replicas=4,
                replica_fanout=4,
                min_coverage_per_source=1,
                min_promoted_sources=1,
                router_radius=1.0e300,
                router_min_margin=0.0,
                token_min_confidence=1e-9,
                consensus_tolerance=1e-7,
                consensus_max_rounds=4096,
                reservoir_size=8,
            ),
        )
        prior_sha256 = None
    else:
        prior_sha256 = hashlib.sha256(prior).hexdigest()
        controller = OoeController.restore(
            crystal_store=store,
            name=state_name,
            action_executors={"mount_organ": executor},
            atlas_revision_verifier=atlas.contains_revision,
            expected_model_pin_sha256=first.model_pin.sha256,
            expected_weight_graph_revision_sha256=first.weight_rail_revision.sha256,
        )
    return controller, prior_sha256


def _train_controller(
    controller: OoeController,
    rows: Sequence[tuple[ResultCell, QwenOoeFeatureReceipt]],
) -> None:
    existing = _controller_history_features(controller)
    for cell, feature in rows[:TRAIN_COUNT]:
        if feature.sha256 in existing:
            continue
        atlas_units = tuple(
            unit for unit in cell.provenance_units if unit.name == "atlas-measurement"
        )
        if len(atlas_units) != 1:
            raise ChatCohortError("training cell has no unique Atlas provenance")
        atlas_unit = atlas_units[0]
        measurement_evidence = atlas_unit.evidence_sha256
        atlas_verifier = atlas_unit.verifier_sha256
        transition = VerifiedTeacherTransition(
            feature_receipt_sha256=feature.sha256,
            site_identity_sha256=feature.site_identity.sha256,
            source_action="qwen_fallback",
            target_action="mount_organ",
            verifier_sha256=atlas_verifier,
            evidence_sha256=measurement_evidence,
            quality_sha256=cell.execution_quality_sha256,
            verified_quality=True,
        )
        controller.ingest_teacher(feature, transition)
        existing = existing | {feature.sha256}
    expected = {feature.sha256 for _cell, feature in rows[:TRAIN_COUNT]}
    actual = _controller_history_features(controller)
    holdout = rows[HOLDOUT_INDEX][1].sha256
    if not expected.issubset(actual) or holdout in actual:
        raise ChatCohortError("temporal split leaked the holdout transition")
    snapshot = json.loads(controller.snapshot_bytes())
    already_promoted = {
        site["site_identity_sha256"]
        for site in snapshot["body"]["sites"]
        if site["crystal_sha256"] is not None
    }
    for site_sha256 in controller.site_identity_sha256s:
        if site_sha256 in already_promoted:
            continue
        coverage = controller.coverage_receipt(site_sha256)
        generation = controller.crystal_store.manifest().generation
        controller.promote(
            site_sha256,
            coverage_sha256=coverage.sha256,
            verifier_sha256s=coverage.verifier_sha256s,
            expected_store_generation=generation,
        )


def _decision_record(decision: object) -> dict[str, Any]:
    fields = (
        "action",
        "confidence",
        "crystal_sha256",
        "crystal_site_identity_sha256",
        "feature_receipt_sha256",
        "origin",
        "placebo_mode",
        "quality_verified",
        "reason",
        "routed_site_identity_sha256",
        "site_identity_sha256",
        "teacher_called",
        "warm_transaction_sha256",
    )
    return {field: getattr(decision, field) for field in fields}


def _placebos(
    controller: OoeController,
    feature: QwenOoeFeatureReceipt,
) -> dict[str, Any]:
    mapping = controller.shuffled_crystal_map(seed=PLACEBO_SEED)
    input_site = feature.site_identity.sha256
    if mapping.get(input_site) in (None, input_site):
        raise ChatCohortError("placebo mapping is not a holdout derangement")
    rows = []
    for mode in ("shuffled-site", "shuffled-crystal"):
        decision = controller.decide_placebo(
            feature,
            "qwen_fallback",
            shuffled_sites=mapping,
            mode=mode,
            stream_id=f"chat-placebo:{mode}",
        )
        rows.append(
            {
                "decision": _decision_record(decision),
                "executed": False,
                "mapped_site_identity_sha256": mapping[input_site],
                "mode": mode,
            }
        )
    return {
        "mapping": dict(sorted(mapping.items())),
        "rows": rows,
        "seed": PLACEBO_SEED,
    }


def _warm_transactions(raw: bytes) -> Mapping[str, tuple[str, ActionExecution, WarmAccountingReceipt | None]]:
    document = json.loads(raw)
    transactions: dict[
        str, tuple[str, ActionExecution, WarmAccountingReceipt | None]
    ] = {}
    for row in document["body"]["warm_transactions"]:
        execution = ActionExecution.from_document(row["execution"])
        final = (
            None
            if row["final_receipt"] is None
            else WarmAccountingReceipt.from_dict(row["final_receipt"])
        )
        if execution.feature_receipt_sha256 in transactions:
            raise ChatCohortError("duplicate warm transaction for one feature")
        if (row["status"] == "pending") != (final is None):
            raise ChatCohortError("warm transaction disposition is inconsistent")
        if final is not None and (
            final.transaction_sha256 != row["transaction_sha256"]
            or final.execution_receipt_sha256 != execution.sha256
            or final.disposition != row["status"]
        ):
            raise ChatCohortError("warm accounting binding is invalid")
        transactions[execution.feature_receipt_sha256] = (
            row["status"],
            execution,
            final,
        )
    return transactions


def _parity_from_document(document: object) -> ResultCellParityReceipt:
    body = _unseal(document, schema="immer-ooe-result-cell-parity/v1", label="parity")
    try:
        parity = ResultCellParityReceipt(**body)
    except (TypeError, ValueError) as exc:
        raise ChatCohortError("parity receipt validation failed") from exc
    if parity.to_document() != dict(document):
        raise ChatCohortError("parity receipt reconstruction mismatch")
    return parity


def _warm_row(
    *,
    controller: OoeController,
    controller_state_name: str,
    controller_snapshot_sha256: str,
    runtime: ChatRuntime,
    item: Mapping[str, Any],
    cell: ResultCell,
    feature: QwenOoeFeatureReceipt,
) -> tuple[dict[str, Any], str]:
    transactions = _warm_transactions(controller.snapshot_bytes())
    existing = transactions.get(feature.sha256)
    if existing is not None:
        status, execution, accounting = existing
        if status != "committed" or accounting is None:
            raise ChatCohortError("holdout already has a non-committed warm execution")
    else:
        decision = controller.try_warm(
            feature,
            "qwen_fallback",
            quality_verifier=lambda receipt, execution: (
                execution.action == "mount_organ"
                and execution.feature_receipt_sha256 == receipt.sha256
                and execution.result == cell.cold_qwen_result_document
                and execution.quality_sha256 == cell.execution_quality_sha256
            ),
            stream_id="chat-holdout-real",
        )
        if (
            decision.origin != "crystal"
            or decision.action != "mount_organ"
            or not decision.quality_verified
            or decision.teacher_called
        ):
            raise ChatCohortError(
                f"holdout action generalization failed: {decision.reason}"
            )
        execution_doc = next(
            row
            for row in json.loads(controller.snapshot_bytes())["body"][
                "warm_transactions"
            ]
            if row["transaction_sha256"] == decision.warm_transaction_sha256
        )["execution"]
        execution = ActionExecution.from_document(execution_doc)
        warm_qwen = result_from_document(execution.result)
        finalized = runtime.adjudicate(item["question"], warm_qwen, warm=True)
        parity = ResultCellParityVerifier().verify(
            cell=cell,
            execution=execution,
            warm_qwen_result=warm_qwen,
            warm_final_result=finalized.result,
            warm_fertig_judgment=finalized.judgment,
            evaluator_quality_contract_sha256=EVALUATOR_QUALITY_CONTRACT_SHA256,
        )
        accounting = controller.commit_warm(decision)
        publication = controller.save_snapshot(
            name=controller_state_name,
            expected_sha256=controller_snapshot_sha256,
        )
        row = {
            "accounting": accounting.to_dict(),
            "decision": {
                "action": "mount_organ",
                "feature_receipt_sha256": feature.sha256,
                "origin": "crystal",
                "quality_verified": True,
                "teacher_called": False,
            },
            "execution": execution.to_document(),
            "item_id": item["item_id"],
            "parity": parity.to_document(),
            "warm_final_result": result_to_document(finalized.result),
            "warm_fertig_judgment": dict(finalized.judgment),
            "warm_qwen_result": result_to_document(warm_qwen),
        }
        return row, publication.payload_sha256
    warm_qwen = result_from_document(execution.result)
    finalized = runtime.adjudicate(item["question"], warm_qwen, warm=True)
    parity = ResultCellParityVerifier().verify(
        cell=cell,
        execution=execution,
        warm_qwen_result=warm_qwen,
        warm_final_result=finalized.result,
        warm_fertig_judgment=finalized.judgment,
        evaluator_quality_contract_sha256=EVALUATOR_QUALITY_CONTRACT_SHA256,
    )
    row = {
        "accounting": accounting.to_dict(),
        "decision": {
            "action": "mount_organ",
            "feature_receipt_sha256": feature.sha256,
            "origin": "crystal",
            "quality_verified": True,
            "teacher_called": False,
        },
        "execution": execution.to_document(),
        "item_id": item["item_id"],
        "parity": parity.to_document(),
        "warm_final_result": result_to_document(finalized.result),
        "warm_fertig_judgment": dict(finalized.judgment),
        "warm_qwen_result": result_to_document(warm_qwen),
    }
    return row, controller_snapshot_sha256


def _validate_runtime_args(args: argparse.Namespace) -> None:
    expected = {
        "compute_dtype": COMPUTE_DTYPE,
        "device": DEVICE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_resident_mb": MAX_RESIDENT_MB,
        "source_budget_mb": SOURCE_BUDGET_MB,
    }
    for field, value in expected.items():
        if getattr(args, field) != value:
            raise ChatCohortError(f"execute {field} differs from frozen policy")
    if getattr(args, "cold_then_warm", False) is not True:
        raise ChatCohortError("execute requires --cold-then-warm")


def _validate_current_roots(args: argparse.Namespace, manifest: Mapping[str, Any]) -> None:
    actual = _store_roots(
        args.cartography_root,
        args.crystal_root,
        args.organ_root,
    )
    if actual != manifest["body"]["roots"]:
        raise ChatCohortError("runtime roots differ from frozen manifest")


def _validate_runtime_manifest(runtime: ChatRuntime, manifest: Mapping[str, Any]) -> None:
    body = manifest["body"]
    if (
        runtime.model_pin.to_document() != body["model_pin"]
        or runtime.tokenizer_sha256 != body["tokenizer_sha256"]
        or dict(runtime.bundle_receipt) != body["bundle_receipt"]
        or runtime.generation_policy_sha256 != body["generation_policy_sha256"]
        or runtime.atlas_state.atlas.revision().to_document() != body["atlas_revision"]
    ):
        raise ChatCohortError("runtime identity differs from frozen manifest")


def _placebo_validate(
    controller: OoeController,
    feature: QwenOoeFeatureReceipt,
    document: Mapping[str, Any],
) -> None:
    expected_map = controller.shuffled_crystal_map(seed=PLACEBO_SEED)
    if document.get("seed") != PLACEBO_SEED or document.get("mapping") != dict(
        sorted(expected_map.items())
    ):
        raise ChatCohortError("placebo map differs from promoted Crystals")
    rows = document.get("rows")
    if not isinstance(rows, list) or [row.get("mode") for row in rows] != [
        "shuffled-site",
        "shuffled-crystal",
    ]:
        raise ChatCohortError("placebo modes are incomplete")
    input_site = feature.site_identity.sha256
    mapped = expected_map[input_site]
    for row in rows:
        decision = row.get("decision")
        if (
            not isinstance(decision, Mapping)
            or row.get("executed") is not False
            or row.get("mapped_site_identity_sha256") != mapped
            or decision.get("feature_receipt_sha256") != feature.sha256
            or decision.get("site_identity_sha256") != input_site
            or decision.get("placebo_mode") != row["mode"]
        ):
            raise ChatCohortError("placebo decision binding is invalid")
        if row["mode"] == "shuffled-site":
            if decision.get("routed_site_identity_sha256") != mapped:
                raise ChatCohortError("shuffled-site placebo used another site")
        elif (
            decision.get("routed_site_identity_sha256") != input_site
            or decision.get("crystal_site_identity_sha256") != mapped
        ):
            raise ChatCohortError("shuffled-crystal placebo used another Crystal")


def _result_document(
    manifest: Mapping[str, Any],
    *,
    cold_rows: Sequence[Mapping[str, Any]],
    cells: Sequence[tuple[ResultCell, QwenOoeFeatureReceipt]],
    controller_snapshot_sha256: str,
    warm: Mapping[str, Any],
    placebos: Mapping[str, Any],
    crystal_audit: CrystalStoreAudit,
    organ_audit: CrystalStoreAudit,
) -> dict[str, Any]:
    holdout_cell = cells[HOLDOUT_INDEX][0]
    execution = ActionExecution.from_document(warm["execution"])
    accounting = WarmAccountingReceipt.from_dict(warm["accounting"])
    parity = _parity_from_document(warm["parity"])
    if (
        execution.teacher_baseline_qwen_forwards != holdout_cell.teacher_forward_count
        or execution.qwen_forwards != 0
        or accounting.saved_qwen_forwards != holdout_cell.teacher_forward_count
        or parity.saved_qwen_forwards != holdout_cell.teacher_forward_count
    ):
        raise ChatCohortError("holdout forward accounting differs from cold ResultCell")
    if not crystal_audit.clean or not organ_audit.clean:
        raise ChatCohortError("chat Crystal/organ store audit is not clean")
    body = {
        "cold_rows": list(cold_rows),
        "controller_snapshot_sha256": controller_snapshot_sha256,
        "headline": {
            "cold_qwen_forwards_total": sum(cell.teacher_forward_count for cell, _ in cells),
            "holdout_executed_qwen_forwards": 0,
            "holdout_saved_qwen_forwards": holdout_cell.teacher_forward_count,
            "holdout_teacher_baseline_qwen_forwards": holdout_cell.teacher_forward_count,
            "holdout_verified_warm_results": 1,
            "teacher_transitions": TRAIN_COUNT,
        },
        "manifest_sha256": manifest["sha256"],
        "placebos": dict(placebos),
        "stores": {
            "crystal": _audit_record(crystal_audit),
            "organ": _audit_record(organ_audit),
            "organ_payload_sha256s": [cell.payload_sha256 for cell, _ in cells],
        },
        "temporal_split": {
            "holdout_feature_receipt_sha256": cells[HOLDOUT_INDEX][1].sha256,
            "holdout_item_id": ITEM_IDS[HOLDOUT_INDEX],
            "holdout_transition_ingested": False,
            "train_feature_receipt_sha256s": [
                feature.sha256 for _cell, feature in cells[:TRAIN_COUNT]
            ],
            "train_item_ids": list(ITEM_IDS[:TRAIN_COUNT]),
        },
        "warm": dict(warm),
    }
    return _seal(RESULT_SCHEMA, body)


def execute(
    args: argparse.Namespace,
    *,
    runtime_factory: RuntimeFactory | None = None,
) -> dict[str, Any]:
    _validate_runtime_args(args)
    manifest = _load_manifest(args.manifest)
    _validate_current_roots(args, manifest)
    crystal_root = Path(args.crystal_root).expanduser().absolute()
    output = Path(args.output).expanduser().absolute()
    with _cohort_lock(crystal_root):
        store = CrystalStore(crystal_root)
        result_state_name = _result_name(manifest["sha256"])
        persisted_result = _state_optional(store, result_state_name)
        if persisted_result is not None:
            verified = _verify_gold_free(
                manifest,
                cartography_root=args.cartography_root,
                crystal_root=crystal_root,
                organ_root=args.organ_root,
                raw=persisted_result,
            )
            _atomic_new(output, persisted_result)
            return verified
        if output.exists() or output.is_symlink():
            raise ChatCohortError("output exists without its CAS result state")

        runtime = _runtime(args, manifest, runtime_factory)
        try:
            _validate_runtime_manifest(runtime, manifest)
            measurements = _manifest_measurements(manifest, runtime.atlas_state)
            bindings = _manifest_bindings(manifest)
            bank = ResultCellBank(args.organ_root)
            progress_name = _progress_name(manifest["sha256"])
            progress_raw = _state_optional(store, progress_name)
            if progress_raw is None:
                progress = _new_progress(manifest["sha256"])
                progress_sha256 = None
            else:
                progress = _load_progress(progress_raw, manifest["sha256"])
                progress_sha256 = hashlib.sha256(progress_raw).hexdigest()

            completed_ids = {row["item_id"] for row in progress["cold_rows"]}
            active = progress["active_item_id"]
            if active is not None and active not in completed_ids:
                index = ITEM_IDS.index(active)
                row = _cold_row(
                    item=manifest["body"]["items"][index],
                    measurement=measurements[index],
                    binding=bindings[index],
                    bank=bank,
                    runtime=runtime,
                    generate=False,
                )
                progress["cold_rows"].append(row)
                progress["active_item_id"] = None
                progress_sha256 = _save_progress(
                    store,
                    progress_name,
                    progress,
                    expected_sha256=progress_sha256,
                )
                completed_ids.add(active)

            for index, item in enumerate(manifest["body"]["items"]):
                if item["item_id"] in completed_ids:
                    continue
                progress["active_item_id"] = item["item_id"]
                progress_sha256 = _save_progress(
                    store,
                    progress_name,
                    progress,
                    expected_sha256=progress_sha256,
                )
                row = _cold_row(
                    item=item,
                    measurement=measurements[index],
                    binding=bindings[index],
                    bank=bank,
                    runtime=runtime,
                    generate=True,
                )
                progress["cold_rows"].append(row)
                progress["active_item_id"] = None
                progress_sha256 = _save_progress(
                    store,
                    progress_name,
                    progress,
                    expected_sha256=progress_sha256,
                )
                completed_ids.add(item["item_id"])

            cells = _validate_cold_rows(
                manifest,
                progress["cold_rows"],
                measurements=measurements,
                bindings=bindings,
                bank=bank,
            )
            controller_name = _controller_name(manifest["sha256"])
            controller, controller_sha256 = _open_controller(
                store=store,
                bank=bank,
                state_name=controller_name,
                atlas=runtime.atlas_state.atlas,
                measurements=measurements,
                cells=cells,
            )
            _train_controller(controller, cells)
            publication = controller.save_snapshot(
                name=controller_name,
                expected_sha256=controller_sha256,
            )
            controller_sha256 = publication.payload_sha256
            progress["controller_snapshot_sha256"] = controller_sha256

            holdout_feature = cells[HOLDOUT_INDEX][1]
            if progress["placebos"] is None:
                progress["placebos"] = _placebos(controller, holdout_feature)
                progress_sha256 = _save_progress(
                    store,
                    progress_name,
                    progress,
                    expected_sha256=progress_sha256,
                )
                publication = controller.save_snapshot(
                    name=controller_name,
                    expected_sha256=controller_sha256,
                )
                controller_sha256 = publication.payload_sha256
                progress["controller_snapshot_sha256"] = controller_sha256
            else:
                _placebo_validate(controller, holdout_feature, progress["placebos"])

            if progress["warm"] is None:
                transactions = _warm_transactions(controller.snapshot_bytes())
                if progress["warm_started"] and holdout_feature.sha256 not in transactions:
                    raise ChatCohortError(
                        "warm outcome is uncertain; refusing duplicate organ execution"
                    )
                progress["warm_started"] = True
                progress_sha256 = _save_progress(
                    store,
                    progress_name,
                    progress,
                    expected_sha256=progress_sha256,
                )
                warm, controller_sha256 = _warm_row(
                    controller=controller,
                    controller_state_name=controller_name,
                    controller_snapshot_sha256=controller_sha256,
                    runtime=runtime,
                    item=manifest["body"]["items"][HOLDOUT_INDEX],
                    cell=cells[HOLDOUT_INDEX][0],
                    feature=holdout_feature,
                )
                progress["warm"] = warm
                progress["controller_snapshot_sha256"] = controller_sha256
                progress_sha256 = _save_progress(
                    store,
                    progress_name,
                    progress,
                    expected_sha256=progress_sha256,
                )

            result = _result_document(
                manifest,
                cold_rows=progress["cold_rows"],
                cells=cells,
                controller_snapshot_sha256=progress["controller_snapshot_sha256"],
                warm=progress["warm"],
                placebos=progress["placebos"],
                crystal_audit=store.audit(),
                organ_audit=bank.store.audit(),
            )
            encoded = _document_bytes(result)
            state_publication = store.publish_state(result_state_name, encoded)
            if state_publication.payload_sha256 != hashlib.sha256(encoded).hexdigest():
                raise ChatCohortError("result CAS publication digest mismatch")
            verified = _verify_gold_free(
                manifest,
                cartography_root=args.cartography_root,
                crystal_root=crystal_root,
                organ_root=args.organ_root,
                raw=encoded,
            )
            _atomic_new(output, encoded)
            return verified
        finally:
            runtime.close()


def _verify_gold_free(
    manifest: Mapping[str, Any],
    *,
    cartography_root: str | os.PathLike[str],
    crystal_root: str | os.PathLike[str],
    organ_root: str | os.PathLike[str],
    raw: bytes,
) -> dict[str, Any]:
    if _store_roots(cartography_root, crystal_root, organ_root) != manifest["body"][
        "roots"
    ]:
        raise ChatCohortError("verification roots differ from manifest")
    document = _decode_jsonl(raw, label="chat cohort result")
    body = _unseal(document, schema=RESULT_SCHEMA, label="chat cohort result")
    if body.get("manifest_sha256") != manifest["sha256"]:
        raise ChatCohortError("chat result belongs to another manifest")
    atlas_state = open_atlas(cartography_root)
    if (
        atlas_state.atlas.revision().to_document()
        != manifest["body"]["atlas_revision"]
        or atlas_state.atlas.model_pin.to_document()
        != manifest["body"]["model_pin"]
    ):
        raise ChatCohortError("Atlas changed after manifest freeze")
    measurements = _manifest_measurements(manifest, atlas_state)
    bindings = _manifest_bindings(manifest)
    store = CrystalStore(crystal_root)
    bank = ResultCellBank(organ_root)
    cells = _validate_cold_rows(
        manifest,
        body["cold_rows"],
        measurements=measurements,
        bindings=bindings,
        bank=bank,
    )
    controller_name = _controller_name(manifest["sha256"])
    controller, controller_state_sha256 = _open_controller(
        store=store,
        bank=bank,
        state_name=controller_name,
        atlas=atlas_state.atlas,
        measurements=measurements,
        cells=cells,
    )
    if controller_state_sha256 is None:
        raise ChatCohortError("chat controller snapshot is missing")
    before = controller.snapshot_bytes()
    if hashlib.sha256(before).hexdigest() != body["controller_snapshot_sha256"]:
        raise ChatCohortError("chat controller snapshot differs from result")
    history = _controller_history_features(controller)
    train = frozenset(feature.sha256 for _cell, feature in cells[:TRAIN_COUNT])
    holdout_feature = cells[HOLDOUT_INDEX][1]
    if history != train or holdout_feature.sha256 in history:
        raise ChatCohortError("controller history violates temporal holdout")

    warm = body["warm"]
    transactions = _warm_transactions(before)
    transaction = transactions.get(holdout_feature.sha256)
    if transaction is None:
        raise ChatCohortError("holdout warm transaction is missing")
    status, execution, accounting = transaction
    if (
        status != "committed"
        or accounting is None
        or execution.to_document() != warm["execution"]
        or accounting.to_dict() != warm["accounting"]
        or execution.action != "mount_organ"
        or execution.qwen_forwards != 0
        or execution.teacher_baseline_qwen_forwards
        != cells[HOLDOUT_INDEX][0].teacher_forward_count
    ):
        raise ChatCohortError("holdout warm transaction/accounting is invalid")
    cell = cells[HOLDOUT_INDEX][0]
    warm_qwen = result_from_document(warm["warm_qwen_result"])
    warm_final = result_from_document(warm["warm_final_result"])
    parity = ResultCellParityVerifier().verify(
        cell=cell,
        execution=execution,
        warm_qwen_result=warm_qwen,
        warm_final_result=warm_final,
        warm_fertig_judgment=warm["warm_fertig_judgment"],
        evaluator_quality_contract_sha256=EVALUATOR_QUALITY_CONTRACT_SHA256,
    )
    if parity.to_document() != warm["parity"]:
        raise ChatCohortError("holdout parity differs from exact recomputation")
    _placebo_validate(controller, holdout_feature, body["placebos"])
    expected = _result_document(
        manifest,
        cold_rows=body["cold_rows"],
        cells=cells,
        controller_snapshot_sha256=body["controller_snapshot_sha256"],
        warm=warm,
        placebos=body["placebos"],
        crystal_audit=store.audit(),
        organ_audit=bank.store.audit(),
    )
    if expected != document:
        raise ChatCohortError("chat result differs from exact store replay")
    result_state = _state_optional(store, _result_name(manifest["sha256"]))
    if result_state is None or result_state != raw:
        raise ChatCohortError("chat result CAS state is missing or different")
    progress_raw = _state_optional(store, _progress_name(manifest["sha256"]))
    if progress_raw is None:
        raise ChatCohortError("chat progress CAS state is missing")
    progress = _load_progress(progress_raw, manifest["sha256"])
    if (
        progress["cold_rows"] != body["cold_rows"]
        or progress["placebos"] != body["placebos"]
        or progress["warm"] != body["warm"]
        or progress["controller_snapshot_sha256"]
        != body["controller_snapshot_sha256"]
        or progress["active_item_id"] is not None
    ):
        raise ChatCohortError("chat progress/result binding is invalid")
    if controller.snapshot_bytes() != before:
        raise ChatCohortError("gold-free verification mutated controller state")
    return document


def _load_gold(path: str | os.PathLike[str]) -> dict[str, str]:
    raw = _stable_regular_bytes(Path(path).expanduser().absolute())
    document = _decode_jsonl(raw, label="frozen gold")
    body = _unseal(document, schema=GOLD_SCHEMA, label="frozen gold")
    if set(body) != {"items"} or not isinstance(body["items"], list):
        raise ChatCohortError("frozen gold body is invalid")
    answers: dict[str, str] = {}
    for index, row in enumerate(body["items"]):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"answer", "item_id"}
            or row["item_id"] != ITEM_IDS[index]
            or not isinstance(row["answer"], str)
            or canonical_numeric_candidate(row["answer"]) is None
        ):
            raise ChatCohortError("frozen gold row is invalid")
        answers[row["item_id"]] = row["answer"]
    if tuple(answers) != ITEM_IDS:
        raise ChatCohortError("frozen gold item order is invalid")
    return answers


def _answer(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if stripped.startswith("####"):
        stripped = stripped[4:].strip()
    return canonical_numeric_candidate(stripped)


def verify(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_manifest(args.manifest)
    result_path = Path(args.result).expanduser().absolute()
    raw = _stable_regular_bytes(result_path)
    result = _verify_gold_free(
        manifest,
        cartography_root=args.cartography_root,
        crystal_root=args.crystal_root,
        organ_root=args.organ_root,
        raw=raw,
    )

    # Gold is deliberately opened only here.  The callable that can see its
    # answer is invoked only by FinalBenchmarkLayer.verify().
    gold = _load_gold(args.gold)
    expected = canonical_numeric_candidate(gold[ITEM_IDS[HOLDOUT_INDEX]])
    evaluator_calls = 0

    def evaluator(cold: Result, warm: Result) -> bool:
        nonlocal evaluator_calls
        evaluator_calls += 1
        return (
            expected is not None
            and _answer(cold.output) == expected
            and _answer(warm.output) == expected
        )

    body = result["body"]
    bank = ResultCellBank(args.organ_root)
    cell = bank.restore_payload(
        body["cold_rows"][HOLDOUT_INDEX]["cell_payload_sha256"]
    )
    parity = _parity_from_document(body["warm"]["parity"])
    warm_final = result_from_document(body["warm"]["warm_final_result"])
    layer = FinalBenchmarkLayer(
        evaluator_quality_contract_sha256=EVALUATOR_QUALITY_CONTRACT_SHA256,
        evaluator=evaluator,
    )
    if evaluator_calls != 0:
        raise ChatCohortError("gold evaluator ran before FinalBenchmarkLayer")
    benchmark = layer.verify(
        cell=cell,
        parity=parity,
        warm_final_result=warm_final,
    )
    if evaluator_calls != 1:
        raise ChatCohortError("FinalBenchmarkLayer did not open gold exactly once")
    verification_body = {
        "benchmark": benchmark.to_document(),
        "evaluator_calls": evaluator_calls,
        "holdout_item_id": ITEM_IDS[HOLDOUT_INDEX],
        "manifest_sha256": manifest["sha256"],
        "result_sha256": result["sha256"],
    }
    return _seal(VERIFY_SCHEMA, verification_body)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare", help="freeze the five-item cohort")
    prepare_parser.add_argument("--manifest", required=True)
    prepare_parser.add_argument("--inputs", required=True)
    prepare_parser.add_argument("--cartography-root", required=True)
    prepare_parser.add_argument("--bundle", required=True)
    prepare_parser.add_argument("--tokenizer", required=True)
    prepare_parser.add_argument("--crystal-root", required=True)
    prepare_parser.add_argument("--organ-root", required=True)

    execute_parser = commands.add_parser("execute", help="run cold then warm")
    execute_parser.add_argument("--manifest", required=True)
    execute_parser.add_argument("--cartography-root", required=True)
    execute_parser.add_argument("--bundle", required=True)
    execute_parser.add_argument("--tokenizer", required=True)
    execute_parser.add_argument("--crystal-root", required=True)
    execute_parser.add_argument("--organ-root", required=True)
    execute_parser.add_argument("--device", choices=("cpu", "mps"), default=DEVICE)
    execute_parser.add_argument(
        "--compute-dtype",
        choices=("float16", "bfloat16", "float32"),
        default=COMPUTE_DTYPE,
    )
    execute_parser.add_argument("--max-resident-mb", type=int, default=MAX_RESIDENT_MB)
    execute_parser.add_argument(
        "--source-budget-mb", type=float, default=SOURCE_BUDGET_MB
    )
    execute_parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    execute_parser.add_argument("--cold-then-warm", action="store_true")
    execute_parser.add_argument("--output", required=True)

    verify_parser = commands.add_parser("verify", help="open frozen gold and verify")
    verify_parser.add_argument("--manifest", required=True)
    verify_parser.add_argument("--cartography-root", required=True)
    verify_parser.add_argument("--crystal-root", required=True)
    verify_parser.add_argument("--organ-root", required=True)
    verify_parser.add_argument("--result", required=True)
    verify_parser.add_argument("--gold", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            document = prepare(args)
        elif args.command == "execute":
            document = execute(args)
        else:
            document = verify(args)
    except (ChatCohortError, OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(document, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
