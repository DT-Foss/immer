"""Mount an existing benchmark-verified Qwen ResultCell as a warm chat path."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

from ...contracts import ExecutionStatus, Result
from ...knowledge.livecausal import LiveGraph
from ..qwen3_8.output_semantics import (
    QWEN_SEMANTIC_REPLAY_METADATA_KEY,
    QwenSemanticReplayKey,
)
from ..qwen3_8.semantic_atlas import ModelPin
from .chat import (
    ChainedOoeChatHook,
    OoeChatAttempt,
    OoeChatHook,
    OoeChatIntegrityError,
)
from .controller import (
    ActionExecution,
    ControllerConfig,
    OoeController,
    OoeControllerIntegrityError,
    WarmAccountingReceipt,
)
from .crystal import CrystalStore
from .identity import canonical_json_bytes, require_sha256
from .qwen_bridge import QwenOoeFeatureReceipt
from .qwen_warm_growth import GrowingQwenWarmBank, load_growing_index
from .result_cells import (
    ResultCellBank,
    ResultCellBinding,
    ResultCellExecutor,
    ResultCellIntegrityError,
    ResultCellMissError,
)


COHORT_SCHEMA = "immer.qwen3.8-ooe-chat-cohort-manifest/v1"
RESULT_SCHEMA = "immer.qwen3.8-ooe-chat-cohort-result/v1"
VERIFICATION_SCHEMA = "immer.qwen3.8-ooe-chat-cohort-verification/v1"
BENCHMARK_SCHEMA = "immer-ooe-result-cell-benchmark/v1"
PARITY_SCHEMA = "immer-ooe-result-cell-parity/v1"
EXECUTION_SCHEMA = "immer-ooe-action-execution/v1"


class QwenWarmBankError(RuntimeError):
    """A persisted Qwen warm-chat authority failed cross-verification."""


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


def _regular_file(path: Path) -> bytes:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise QwenWarmBankError(f"warm-bank file is missing: {path.name}") from exc
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise QwenWarmBankError(f"warm-bank path is not a regular file: {path.name}")
    return path.read_bytes()


def _sealed(value: object, *, schema: str, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"body", "schema", "sha256"}:
        raise QwenWarmBankError(f"{label} envelope is invalid")
    if value.get("schema") != schema or not isinstance(value.get("body"), dict):
        raise QwenWarmBankError(f"{label} schema is invalid")
    body = value["body"]
    if require_sha256(value.get("sha256"), field=f"{label}.sha256") != _digest(body):
        raise QwenWarmBankError(f"{label} SHA-256 mismatch")
    return value


def _document(path: Path, *, schema: str, label: str) -> dict[str, Any]:
    raw = _regular_file(path)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QwenWarmBankError(f"{label} is not valid JSON") from exc
    canonical = canonical_json_bytes(value)
    if raw not in (canonical, canonical + b"\n"):
        raise QwenWarmBankError(f"{label} is not canonical JSON")
    return _sealed(value, schema=schema, label=label)


def _directory(root: Path, *, prefix: str, fallback: str) -> Path:
    candidates = []
    for path in root.iterdir():
        if path.name == fallback or path.name.startswith(prefix):
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISDIR(metadata.st_mode) and not path.is_symlink():
                candidates.append(path)
    if len(candidates) != 1:
        raise QwenWarmBankError(
            f"warm-bank root requires exactly one {prefix.rstrip('-')} directory"
        )
    return candidates[0]


def _semantic_replay_provider(
    bank: ResultCellBank,
    semantic_key_verifier: Any | None = None,
):
    def provide(
        question: str,
        metadata: Mapping[str, Any],
    ) -> OoeChatAttempt | None:
        raw_key = metadata.get(QWEN_SEMANTIC_REPLAY_METADATA_KEY)
        if raw_key is None or semantic_key_verifier is None:
            return None
        try:
            verified = bool(
                semantic_key_verifier(question, raw_key, metadata)
            )
        except Exception as exc:
            raise OoeChatIntegrityError(
                "semantic replay metadata verification failed"
            ) from exc
        if not verified:
            return None
        try:
            key = QwenSemanticReplayKey.from_document(raw_key)
        except (TypeError, ValueError) as exc:
            raise OoeChatIntegrityError(
                "semantic replay metadata is invalid"
            ) from exc
        if hashlib.sha256(question.strip().encode("utf-8")).hexdigest() != (
            key.question_sha256
        ):
            return None
        try:
            pointer, cell = bank.restore_semantic(key)
        except ResultCellMissError:
            return None
        except ResultCellIntegrityError as exc:
            raise OoeChatIntegrityError(
                "semantic replay ResultCell failed integrity"
            ) from exc
        cold = cell.cold_qwen_result
        if not cold.ok or not isinstance(cold.output, str) or not cold.output:
            raise OoeChatIntegrityError(
                "semantic replay ResultCell has no successful Qwen output"
            )
        evidence = dict(cold.evidence)
        generation = evidence.get("generation")
        if not isinstance(generation, Mapping):
            raise OoeChatIntegrityError(
                "semantic replay ResultCell lost generation evidence"
            )
        warm_generation = dict(generation)
        for field, value in {
            "forward_passes": 0,
            "linear_calls": 0,
            "seconds": 0.0,
            "source_body_bytes": 0,
            "state_bytes": 0,
            "time_to_first_token_seconds": 0.0,
            "output_tokens_per_second": 0.0,
        }.items():
            if field in warm_generation or field in {
                "forward_passes",
                "linear_calls",
                "seconds",
                "source_body_bytes",
                "state_bytes",
            }:
                warm_generation[field] = value
        evidence["generation"] = warm_generation
        execution_sha256 = _digest(
            {
                "output_sha256": pointer.output_sha256,
                "payload_sha256": pointer.payload_sha256,
                "semantic_key_sha256": key.sha256,
                "token_trace_sha256": pointer.token_trace_sha256,
            }
        )
        evidence["semantic_replay"] = {
            "execution_sha256": execution_sha256,
            "payload_sha256": pointer.payload_sha256,
            "producer_binding_sha256": pointer.producer_binding_sha256,
            "saved_qwen_forwards": pointer.teacher_forward_count,
            "semantic_key_sha256": key.sha256,
        }
        result = Result(
            ExecutionStatus.OK,
            cold.component,
            output=cold.output,
            evidence=evidence,
        )
        settled = False
        pointer_sha256 = hashlib.sha256(pointer.to_bytes()).hexdigest()

        def settle(accept: bool) -> WarmAccountingReceipt:
            nonlocal settled
            if settled:
                raise QwenWarmBankError(
                    "semantic replay attempt was already settled"
                )
            try:
                receipt = bank.settle_semantic(
                    key,
                    accept=accept,
                    expected_pointer_sha256=pointer_sha256,
                )
            except ResultCellIntegrityError as exc:
                raise OoeChatIntegrityError(
                    "semantic replay accounting failed integrity"
                ) from exc
            settled = True
            return receipt

        return OoeChatAttempt(
            result,
            {
                "execution_sha256": execution_sha256,
                "saved_qwen_forwards": pointer.teacher_forward_count,
                "semantic_key_sha256": key.sha256,
                "status": "semantic-hit",
            },
            _settler=settle,
            _abstention_authorized=True,
        )

    return provide


@dataclass(frozen=True, slots=True)
class VerifiedQwenWarmMount:
    """A fully cross-bound warm hook and its compact product identity."""

    hook: OoeChatHook
    root: Path
    manifest_sha256: str
    result_sha256: str
    verification_sha256: str
    benchmark_sha256: str
    parity_sha256: str
    question_sha256: str
    feature_sha256: str
    cell_payload_sha256: str
    execution_sha256: str
    saved_qwen_forwards: int
    result_cell_code_revision: str
    growing_entries: int

    def identity(self) -> dict[str, object]:
        return {
            "benchmark_sha256": self.benchmark_sha256,
            "cell_payload_sha256": self.cell_payload_sha256,
            "execution_sha256": self.execution_sha256,
            "feature_sha256": self.feature_sha256,
            "growing_entries": self.growing_entries,
            "manifest_sha256": self.manifest_sha256,
            "parity_sha256": self.parity_sha256,
            "question_sha256": self.question_sha256,
            "result_sha256": self.result_sha256,
            "result_cell_code_revision": self.result_cell_code_revision,
            "root": str(self.root),
            "saved_qwen_forwards": self.saved_qwen_forwards,
            "schema": "immer.qwen3.8-verified-warm-mount/v1",
            "verification_sha256": self.verification_sha256,
        }


def open_verified_qwen_warm_bank(
    root: str | os.PathLike[str],
    *,
    runtime_profile_sha256: str | None = None,
    runtime_code_revision: str | None = None,
    template_output_character_limit: int | None = None,
    prompt_token_verifier: Any | None = None,
    semantic_key_verifier: Any | None = None,
) -> VerifiedQwenWarmMount:
    """Open one existing verified warm cell without running or probing Qwen."""

    selected = Path(root).expanduser().absolute()
    runtime_profile = (
        None
        if runtime_profile_sha256 is None
        else require_sha256(
            runtime_profile_sha256,
            field="runtime_profile_sha256",
        )
    )
    runtime_code = (
        None
        if runtime_code_revision is None
        else require_sha256(runtime_code_revision, field="runtime_code_revision")
    )
    if template_output_character_limit is not None and (
        isinstance(template_output_character_limit, bool)
        or not isinstance(template_output_character_limit, int)
        or template_output_character_limit <= 0
    ):
        raise QwenWarmBankError("template output character limit must be positive")
    if prompt_token_verifier is not None and not callable(prompt_token_verifier):
        raise QwenWarmBankError("prompt_token_verifier must be callable or None")
    if semantic_key_verifier is not None and not callable(semantic_key_verifier):
        raise QwenWarmBankError("semantic_key_verifier must be callable or None")
    if semantic_key_verifier is not None and runtime_profile is None:
        raise QwenWarmBankError(
            "semantic replay verifier requires a growing warm profile"
        )
    if len(
        {
            runtime_profile is None,
            runtime_code is None,
            template_output_character_limit is None,
            prompt_token_verifier is None,
        }
    ) != 1:
        raise QwenWarmBankError(
            "growing warm profile, code revision and template limit must be configured together"
        )
    try:
        root_stat = selected.lstat()
    except FileNotFoundError as exc:
        raise QwenWarmBankError(f"warm-bank root is missing: {selected}") from exc
    if not stat.S_ISDIR(root_stat.st_mode) or selected.is_symlink():
        raise QwenWarmBankError("warm-bank root must be a real directory")

    cohort = _document(
        selected / "cohort.json",
        schema=COHORT_SCHEMA,
        label="cohort manifest",
    )
    result = _document(
        selected / "result.json",
        schema=RESULT_SCHEMA,
        label="cohort result",
    )
    verification = _document(
        selected / "verification.json",
        schema=VERIFICATION_SCHEMA,
        label="cohort verification",
    )
    cohort_body = cohort["body"]
    result_body = result["body"]
    verification_body = verification["body"]
    manifest_sha256 = cohort["sha256"]
    if (
        result_body.get("manifest_sha256") != manifest_sha256
        or verification_body.get("manifest_sha256") != manifest_sha256
        or verification_body.get("result_sha256") != result["sha256"]
    ):
        raise QwenWarmBankError("warm-bank manifest/result chain is broken")

    holdout_id = cohort_body.get("holdout_item_id")
    if (
        not isinstance(holdout_id, str)
        or verification_body.get("holdout_item_id") != holdout_id
    ):
        raise QwenWarmBankError("warm-bank holdout identity is inconsistent")
    items = cohort_body.get("items")
    cold_rows = result_body.get("cold_rows")
    if not isinstance(items, list) or not isinstance(cold_rows, list):
        raise QwenWarmBankError("warm-bank cohort rows are invalid")
    holdout_items = [row for row in items if row.get("item_id") == holdout_id]
    holdout_rows = [row for row in cold_rows if row.get("item_id") == holdout_id]
    if len(holdout_items) != 1 or len(holdout_rows) != 1:
        raise QwenWarmBankError("warm-bank holdout row is not unique")
    item = holdout_items[0]
    cold_row = holdout_rows[0]

    benchmark = _sealed(
        verification_body.get("benchmark"),
        schema=BENCHMARK_SCHEMA,
        label="warm benchmark",
    )
    benchmark_body = benchmark["body"]
    saved = benchmark_body.get("saved_qwen_forwards")
    quality_contract_sha256 = require_sha256(
        benchmark_body.get("evaluator_quality_contract_sha256"),
        field="evaluator_quality_contract_sha256",
    )
    if (
        verification_body.get("evaluator_calls") != 1
        or benchmark_body.get("evaluator_quality_verified") is not True
        or benchmark_body.get("exact_parity") is not True
        or isinstance(saved, bool)
        or not isinstance(saved, int)
        or saved <= 0
    ):
        raise QwenWarmBankError("warm benchmark does not authorize execution")

    warm = result_body.get("warm")
    if not isinstance(warm, dict) or warm.get("item_id") != holdout_id:
        raise QwenWarmBankError("warm result does not belong to the holdout")
    parity = _sealed(
        warm.get("parity"),
        schema=PARITY_SCHEMA,
        label="warm parity",
    )
    execution_document = _sealed(
        warm.get("execution"),
        schema=EXECUTION_SCHEMA,
        label="warm execution",
    )
    parity_body = parity["body"]
    try:
        accounting = WarmAccountingReceipt.from_dict(warm["accounting"])
    except (KeyError, TypeError, ValueError) as exc:
        raise QwenWarmBankError("warm accounting receipt is invalid") from exc
    decision = warm.get("decision")
    if not isinstance(decision, dict):
        raise QwenWarmBankError("warm decision receipt is invalid")
    if (
        benchmark_body.get("parity_receipt_sha256") != parity["sha256"]
        or parity_body.get("exact_parity") is not True
        or parity_body.get("qwen_result_document_exact") is not True
        or parity_body.get("final_semantic_core_exact") is not True
        or parity_body.get("execution_sha256") != execution_document["sha256"]
        or parity_body.get("saved_qwen_forwards") != saved
        or parity_body.get("warm_qwen_forwards") != 0
        or parity_body.get("evaluator_quality_contract_sha256")
        != quality_contract_sha256
        or accounting.disposition != "committed"
        or accounting.execution_receipt_sha256 != execution_document["sha256"]
        or accounting.saved_qwen_forwards != saved
        or decision.get("quality_verified") is not True
        or decision.get("origin") != "crystal"
        or decision.get("teacher_called") is not False
    ):
        raise QwenWarmBankError("warm parity does not authorize the stored execution")

    execution = ActionExecution.from_document(execution_document)
    if (
        execution.quality_verified is not True
        or execution.qwen_forwards != 0
        or execution.teacher_baseline_qwen_forwards != saved
    ):
        raise QwenWarmBankError("warm execution has invalid Qwen accounting")

    stores = result_body.get("stores")
    crystal_store_receipt = (
        stores.get("crystal") if isinstance(stores, dict) else None
    )
    raw_legacy_crystals = (
        crystal_store_receipt.get("valid_objects")
        if isinstance(crystal_store_receipt, dict)
        else None
    )
    if not isinstance(raw_legacy_crystals, list) or not raw_legacy_crystals:
        raise QwenWarmBankError("verified legacy Crystal inventory is missing")
    legacy_crystal_sha256s = tuple(
        require_sha256(value, field="legacy_crystal_sha256")
        for value in raw_legacy_crystals
    )

    try:
        binding = ResultCellBinding.from_record(item["binding"])
        feature = QwenOoeFeatureReceipt.from_document(cold_row["feature"])
    except (KeyError, TypeError, ValueError) as exc:
        raise QwenWarmBankError("warm binding or feature is invalid") from exc
    question_sha256 = require_sha256(
        item.get("question_sha256"), field="question_sha256"
    )
    question = item.get("question")
    cell_payload_sha256 = require_sha256(
        cold_row.get("cell_payload_sha256"), field="cell_payload_sha256"
    )
    if (
        binding.sha256 != item.get("binding_sha256")
        or not isinstance(question, str)
        or hashlib.sha256(question.encode("utf-8")).hexdigest() != question_sha256
        or binding.question_sha256 != question_sha256
        or feature.sha256 != cold_row.get("feature_receipt_sha256")
        or feature.sha256 != execution.feature_receipt_sha256
        or decision.get("feature_receipt_sha256") != feature.sha256
        or cell_payload_sha256 != parity_body.get("cell_payload_sha256")
    ):
        raise QwenWarmBankError("warm binding, feature, cell, or execution diverged")

    cartography_root = _directory(
        selected,
        prefix="cartography-",
        fallback="cartography",
    )
    crystal_root = _directory(
        selected,
        prefix="chat-crystals-",
        fallback="chat-crystals",
    )
    organ_root = _directory(
        selected,
        prefix="chat-organs-",
        fallback="chat-organs",
    )
    graph = LiveGraph(cartography_root / "atlas")
    store = CrystalStore(crystal_root)
    bank = ResultCellBank(organ_root)
    if not store.audit().clean or not bank.store.audit().clean:
        raise QwenWarmBankError("warm Crystal or ResultCell store failed audit")
    if not set(legacy_crystal_sha256s).issubset(store.manifest().objects):
        raise QwenWarmBankError("warm Crystal manifest lost its verified inventory")
    cell = bank.restore(binding)
    if (
        cell.payload_sha256 != cell_payload_sha256
        or cell.evaluator_quality_contract_sha256 != quality_contract_sha256
    ):
        raise QwenWarmBankError("warm ResultCell payload differs from the cohort")
    executor = ResultCellExecutor(bank, binding)
    rebuilt_execution = executor(feature)
    if rebuilt_execution.to_document() != execution_document:
        raise QwenWarmBankError("warm execution cannot be rebuilt from its ResultCell")

    state_name = f"qwen38-ooe-chat-controller-{manifest_sha256[:24]}"
    def restore_controller() -> OoeController:
        def verifier(revision: Any) -> bool:
            return graph.store.contains_revision(
                revision.sequence,
                revision.event_sha256,
            )
        options = {
            "crystal_store": store,
            "name": state_name,
            "action_executors": {"mount_organ": executor},
            "atlas_revision_verifier": verifier,
            "expected_model_pin_sha256": binding.model_pin.sha256,
            "expected_weight_graph_revision_sha256": (
                feature.weight_graph_revision_sha256
            ),
            "legacy_gap_compatible_crystal_sha256s": (
                legacy_crystal_sha256s
            ),
        }
        try:
            return OoeController.restore(**options)
        except OoeControllerIntegrityError:
            raw_state = store.restore_state(state_name)
            try:
                state_body = json.loads(raw_state)["body"]
                old_generation = state_body["crystal_manifest_generation"]
                old_manifest_sha256 = state_body["crystal_manifest_sha256"]
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise QwenWarmBankError(
                    "controller recovery state is invalid"
                ) from exc
            recovered, _receipt = OoeController.restore_recoverable(
                **options,
                expected_old_manifest_generation=old_generation,
                expected_old_manifest_sha256=old_manifest_sha256,
                expected_old_state_sha256=hashlib.sha256(raw_state).hexdigest(),
            )
            return recovered

    controller = restore_controller()

    def feature_provider(
        question: str,
        metadata: Mapping[str, Any],
    ) -> QwenOoeFeatureReceipt | None:
        if hashlib.sha256(question.encode("utf-8")).hexdigest() == question_sha256:
            raw_token = metadata.get("qwen_token_sha256")
            if raw_token is not None and require_sha256(
                raw_token,
                field="qwen_token_sha256",
            ) != feature.probe.token_sha256:
                return None
            return feature
        return None

    def quality_verifier(
        receipt: QwenOoeFeatureReceipt,
        candidate: ActionExecution,
    ) -> bool:
        if receipt == feature:
            return (
                candidate.to_document() == execution_document
                and candidate.qwen_forwards == 0
                and candidate.teacher_baseline_qwen_forwards == saved
            )
        return False

    base_hook = OoeChatHook(
        controller=controller,
        feature_provider=feature_provider,
        quality_verifier=quality_verifier,
        direct_provider=_semantic_replay_provider(
            bank,
            semantic_key_verifier,
        ),
        snapshot_name=state_name,
        snapshot_restorer=restore_controller,
        commit_on_fertig_abstention=True,
    )
    hook: OoeChatHook = base_hook
    growth = None
    if runtime_profile is not None and runtime_code is not None:
        growing_root = selected / "growing-crystals" / runtime_profile[:24]
        growing_root.mkdir(parents=True, exist_ok=True)
        growing_store = CrystalStore(growing_root)
        growing_entries, growing_index_sha256 = load_growing_index(
            growing_store,
            bank,
        )
        dynamic_pin = ModelPin(
            repo_id=binding.model_pin.repo_id,
            revision=binding.model_pin.revision,
            bundle_fingerprint=binding.model_pin.bundle_fingerprint,
            bundle_manifest_sha256=binding.model_pin.bundle_manifest_sha256,
            code_revision=runtime_code,
        )
        if any(
            entry.runtime_profile_sha256 != runtime_profile
            or entry.binding.model_pin != dynamic_pin
            for entry in growing_entries.values()
        ):
            raise QwenWarmBankError(
                "growing warm index belongs to another runtime authority"
            )
        growing_bindings = {
            entry.feature.sha256: entry.binding
            for entry in growing_entries.values()
        }

        def resolve_growing(
            receipt: QwenOoeFeatureReceipt,
        ) -> ResultCellBinding:
            try:
                return growing_bindings[receipt.sha256]
            except KeyError as exc:
                raise QwenWarmBankError(
                    "no growing ResultCell binding for warm feature"
                ) from exc

        growing_executor = ResultCellExecutor(bank, resolve_growing)
        growing_state_name = f"qwen38-growing-controller-{runtime_profile[:24]}"

        def growing_verifier(revision: Any) -> bool:
            return graph.store.contains_revision(
                revision.sequence,
                revision.event_sha256,
            )

        def restore_growing_controller() -> OoeController:
            options = {
                "crystal_store": growing_store,
                "name": growing_state_name,
                "action_executors": {"mount_organ": growing_executor},
                "atlas_revision_verifier": growing_verifier,
                "expected_model_pin_sha256": dynamic_pin.sha256,
                "expected_weight_graph_revision_sha256": (
                    feature.weight_graph_revision_sha256
                ),
            }
            try:
                return OoeController.restore(**options)
            except KeyError:
                created = OoeController(
                    model_pin_sha256=dynamic_pin.sha256,
                    weight_graph_revision_sha256=(
                        feature.weight_graph_revision_sha256
                    ),
                    atlas_graph_revision=controller.atlas_graph_revision,
                    crystal_store=growing_store,
                    config=ControllerConfig(**controller.config.to_dict()),
                    action_executors={"mount_organ": growing_executor},
                    atlas_revision_verifier=growing_verifier,
                )
                created.save_snapshot(name=growing_state_name)
                return created
            except OoeControllerIntegrityError:
                raw_state = growing_store.restore_state(growing_state_name)
                try:
                    state_body = json.loads(raw_state)["body"]
                except (KeyError, TypeError, json.JSONDecodeError) as exc:
                    raise QwenWarmBankError(
                        "growing controller recovery state is invalid"
                    ) from exc
                recovered, _receipt = OoeController.restore_recoverable(
                    **options,
                    expected_old_manifest_generation=(
                        state_body["crystal_manifest_generation"]
                    ),
                    expected_old_manifest_sha256=(
                        state_body["crystal_manifest_sha256"]
                    ),
                    expected_old_state_sha256=hashlib.sha256(
                        raw_state
                    ).hexdigest(),
                )
                return recovered

        growth = GrowingQwenWarmBank(
            root=growing_root,
            store=growing_store,
            bank=bank,
            template_feature=feature,
            runtime_profile_sha256=runtime_profile,
            controller_state_name=growing_state_name,
            restore_controller=restore_growing_controller,
            bindings_by_feature=growing_bindings,
            entries=growing_entries,
            index_sha256=growing_index_sha256,
            template_output_character_limit=template_output_character_limit,
            prompt_token_verifier=prompt_token_verifier,
        )
        growth.reconcile()
        growing_hook = OoeChatHook(
            controller=growth.controller,
            feature_provider=growth.feature_provider,
            quality_verifier=growth.quality_verifier,
            cold_observer=growth.observe_cold,
            direct_provider=growth.direct_provider,
            snapshot_name=growing_state_name,
            snapshot_restorer=restore_growing_controller,
            commit_on_fertig_abstention=True,
        )
        growth.hook = growing_hook
        hook = ChainedOoeChatHook((base_hook, growing_hook))
    return VerifiedQwenWarmMount(
        hook=hook,
        root=selected,
        manifest_sha256=manifest_sha256,
        result_sha256=result["sha256"],
        verification_sha256=verification["sha256"],
        benchmark_sha256=benchmark["sha256"],
        parity_sha256=parity["sha256"],
        question_sha256=question_sha256,
        feature_sha256=feature.sha256,
        cell_payload_sha256=cell_payload_sha256,
        execution_sha256=execution.sha256,
        saved_qwen_forwards=saved,
        result_cell_code_revision=(
            binding.model_pin.code_revision
            if runtime_code is None
            else runtime_code
        ),
        growing_entries=0 if growth is None else len(growth.entries),
    )


__all__ = [
    "QwenWarmBankError",
    "VerifiedQwenWarmMount",
    "open_verified_qwen_warm_bank",
]
