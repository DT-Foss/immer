"""Crash-safe growth for the verified local Qwen warm ResultCell bank."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import threading
import unicodedata
from typing import Any, Iterator

try:  # pragma: no cover - every production platform provides fcntl.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from ...contracts import Result
from ..qwen3_8.action_bank import (
    InferenceActionBankError,
    InferenceActionDirective,
)
from ..qwen3_8.semantic_atlas import ModelPin, ProbeIdentity
from .controller import OoeController, VerifiedTeacherTransition
from .crystal import CrystalStore, CrystalStoreError
from .identity import canonical_json_bytes, require_sha256
from .qwen_bridge import QwenOoeFeatureReceipt
from .result_cells import (
    COLD_QWEN_BINDING_EVIDENCE_KEY,
    ResultCell,
    ResultCellBank,
    ResultCellBinding,
    ResultCellExecutor,
    attach_cold_qwen_generation_receipt,
    qwen_result_binding_evidence,
)
from .qwen_warm_templates import ParametricWarmBank, ParametricWarmError


GROWING_INDEX_SCHEMA = "immer.qwen3.8-growing-warm-index/v1"
GROWING_ENTRY_SCHEMA = "immer.qwen3.8-growing-warm-entry/v1"
GROWING_INDEX_STATE = "qwen38-growing-warm-index-v1"
GROWING_QUALITY_CONTRACT_SHA256 = hashlib.sha256(
    b"immer:qwen3.8-deterministic-result-cell-replay/v1"
).hexdigest()
GROWING_FEATURE_VERIFIER_SHA256 = hashlib.sha256(
    b"immer:qwen3.8-prompt-native-markov-feature/v1"
).hexdigest()


class QwenWarmGrowthError(RuntimeError):
    """A growing warm-cell transaction failed integrity or atomicity."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _question_sha256(question: str) -> str:
    return hashlib.sha256(question.strip().encode("utf-8")).hexdigest()


def _profile(metadata: Mapping[str, Any]) -> str | None:
    value = metadata.get("qwen_warm_runtime_profile_sha256")
    return None if value is None else require_sha256(value, field="warm profile")


def prefix_sinkhorn_warm_allowed(metadata: Mapping[str, Any]) -> bool:
    """Reject warm replay only when a request explicitly disables Prefix-Sinkhorn."""

    raw = metadata.get("qwen_inference_action_directive")
    if raw is None:
        return True
    try:
        directive = InferenceActionDirective.from_document(raw)
    except (InferenceActionBankError, TypeError, ValueError):
        return False
    return "prefix_sinkhorn" not in directive.disabled_actions


def _prompt_sketch(question: str, dimensions: int) -> tuple[float, ...]:
    if dimensions < 1:
        raise ValueError("feature dimensions must be positive")
    normalized = unicodedata.normalize("NFKC", question).casefold().strip()
    words = tuple(normalized.split())
    terms: list[tuple[str, float]] = []
    terms.extend((f"w:{word}", 1.0) for word in words)
    terms.extend(
        (f"b:{words[index]}\x1f{words[index + 1]}", 0.8)
        for index in range(max(0, len(words) - 1))
    )
    compact = " ".join(words)
    terms.extend(
        (f"c:{compact[index:index + 3]}", 0.35)
        for index in range(max(0, len(compact) - 2))
    )
    terms.extend(
        (
            (f"length:{min(255, len(normalized)).bit_length()}", 0.5),
            (f"words:{min(255, len(words)).bit_length()}", 0.5),
        )
    )
    values = [0.0] * dimensions
    for term, weight in terms:
        raw = hashlib.blake2b(
            term.encode("utf-8"),
            digest_size=16,
            person=b"IMMRWARM",
        ).digest()
        index = int.from_bytes(raw[:8], "little") % dimensions
        sign = 1.0 if raw[8] & 1 else -1.0
        values[index] += sign * weight
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        values[0] = 1.0
        norm = 1.0
    return tuple(0.0 if value == 0.0 else value / norm for value in values)


def _json_safe(value: object, *, path: str = "evidence") -> object:
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for raw_key, child in value.items():
            if isinstance(raw_key, str):
                key = raw_key
            elif isinstance(raw_key, int) and not isinstance(raw_key, bool):
                key = str(raw_key)
            else:
                raise QwenWarmGrowthError(
                    f"{path} contains a non-JSON object key"
                )
            if key in normalized:
                raise QwenWarmGrowthError(
                    f"{path} contains colliding JSON object keys"
                )
            normalized[key] = _json_safe(child, path=f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(child, path=f"{path}[{index}]")
            for index, child in enumerate(value)
        ]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise QwenWarmGrowthError(f"{path} contains a non-JSON value")


def _binding_from_qwen_result(question: str, result: Result) -> ResultCellBinding:
    evidence = result.evidence
    raw_binding = evidence.get(COLD_QWEN_BINDING_EVIDENCE_KEY)
    if not isinstance(raw_binding, Mapping):
        raise QwenWarmGrowthError("cold Qwen result lacks its binding receipt")
    body = raw_binding.get("body")
    bundle = evidence.get("bundle")
    if not isinstance(body, Mapping) or not isinstance(bundle, Mapping):
        raise QwenWarmGrowthError("cold Qwen binding or bundle evidence is invalid")
    try:
        pin = ModelPin(
            repo_id=str(evidence["model"]),
            revision=str(evidence["revision"]),
            bundle_fingerprint=str(bundle["layout_fingerprint"]),
            bundle_manifest_sha256=str(bundle["manifest_sha256"]),
            code_revision=str(body["code_revision"]),
        )
        binding = ResultCellBinding(
            model_pin=pin,
            tokenizer_sha256=str(body["tokenizer_sha256"]),
            question_sha256=_question_sha256(question),
            rendered_prompt_sha256=str(body["rendered_prompt_sha256"]),
            rendered_prompt_token_sha256=str(
                body["rendered_prompt_token_sha256"]
            ),
            system_prompt_sha256=str(body["system_prompt_sha256"]),
            generation_policy_sha256=str(body["generation_policy_sha256"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise QwenWarmGrowthError("cold Qwen binding cannot be reconstructed") from exc
    if qwen_result_binding_evidence(binding) != dict(raw_binding):
        raise QwenWarmGrowthError("cold Qwen binding receipt is not reproducible")
    return binding


def _fertig_judgment(final_result: Result) -> tuple[str, dict[str, Any]]:
    receipt = final_result.evidence.get("receipt")
    fertig = receipt.get("fertig") if isinstance(receipt, Mapping) else None
    if not isinstance(fertig, Mapping):
        raise QwenWarmGrowthError("final result lacks FERTIG adjudication")
    status = fertig.get("status")
    judgment = fertig.get("verification")
    if (
        status not in {"verified", "mismatch", "abstained"}
        or not isinstance(judgment, dict)
        or judgment.get("status") != status
    ):
        raise QwenWarmGrowthError("final FERTIG adjudication is not reusable")
    if status == "mismatch":
        raise QwenWarmGrowthError("FERTIG mismatch cannot become a warm cell")
    return str(status), judgment


@dataclass(frozen=True, slots=True)
class GrowingWarmEntry:
    runtime_profile_sha256: str
    question_sha256: str
    binding: ResultCellBinding
    feature: QwenOoeFeatureReceipt
    transition: VerifiedTeacherTransition
    cell_payload_sha256: str
    output_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "runtime_profile_sha256",
            "question_sha256",
            "cell_payload_sha256",
            "output_sha256",
        ):
            object.__setattr__(
                self,
                field,
                require_sha256(getattr(self, field), field=field),
            )
        if self.binding.question_sha256 != self.question_sha256:
            raise ValueError("growing entry question differs from its binding")
        self.binding.assert_feature_bound(self.feature)
        self.transition.assert_bound(self.feature)

    @property
    def key(self) -> tuple[str, str]:
        return self.runtime_profile_sha256, self.question_sha256

    def to_document(self) -> dict[str, Any]:
        body = {
            "binding": self.binding.as_record(),
            "cell_payload_sha256": self.cell_payload_sha256,
            "feature": self.feature.to_document(),
            "output_sha256": self.output_sha256,
            "question_sha256": self.question_sha256,
            "runtime_profile_sha256": self.runtime_profile_sha256,
            "transition": self.transition.to_document(),
        }
        return {"body": body, "schema": GROWING_ENTRY_SCHEMA, "sha256": _digest(body)}

    @classmethod
    def from_document(cls, value: object) -> "GrowingWarmEntry":
        if (
            not isinstance(value, Mapping)
            or set(value) != {"body", "schema", "sha256"}
            or value.get("schema") != GROWING_ENTRY_SCHEMA
            or not isinstance(value.get("body"), Mapping)
        ):
            raise QwenWarmGrowthError("growing entry envelope is invalid")
        body = value["body"]
        if value.get("sha256") != _digest(body):
            raise QwenWarmGrowthError("growing entry SHA-256 mismatch")
        try:
            entry = cls(
                runtime_profile_sha256=body["runtime_profile_sha256"],
                question_sha256=body["question_sha256"],
                binding=ResultCellBinding.from_record(body["binding"]),
                feature=QwenOoeFeatureReceipt.from_document(body["feature"]),
                transition=VerifiedTeacherTransition.from_document(
                    body["transition"]
                ),
                cell_payload_sha256=body["cell_payload_sha256"],
                output_sha256=body["output_sha256"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise QwenWarmGrowthError("growing entry is invalid") from exc
        if entry.to_document() != dict(value):
            raise QwenWarmGrowthError("growing entry reconstruction mismatch")
        return entry


def _index_bytes(entries: Mapping[tuple[str, str], GrowingWarmEntry]) -> bytes:
    body = {
        "entries": [
            entry.to_document()
            for _key, entry in sorted(entries.items(), key=lambda row: row[0])
        ],
        "schema": GROWING_INDEX_SCHEMA,
    }
    return canonical_json_bytes(
        {"body": body, "schema": GROWING_INDEX_SCHEMA, "sha256": _digest(body)}
    )


def load_growing_index(
    store: CrystalStore,
    bank: ResultCellBank,
) -> tuple[dict[tuple[str, str], GrowingWarmEntry], str | None]:
    try:
        raw = store.restore_state(GROWING_INDEX_STATE)
    except KeyError:
        return {}, None
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QwenWarmGrowthError("growing index is invalid JSON") from exc
    if canonical_json_bytes(value) != raw or not isinstance(value, Mapping):
        raise QwenWarmGrowthError("growing index is not canonical")
    body = value.get("body")
    if (
        set(value) != {"body", "schema", "sha256"}
        or value.get("schema") != GROWING_INDEX_SCHEMA
        or not isinstance(body, Mapping)
        or set(body) != {"entries", "schema"}
        or body.get("schema") != GROWING_INDEX_SCHEMA
        or value.get("sha256") != _digest(body)
        or not isinstance(body.get("entries"), list)
    ):
        raise QwenWarmGrowthError("growing index envelope is invalid")
    entries: dict[tuple[str, str], GrowingWarmEntry] = {}
    for raw_entry in body["entries"]:
        entry = GrowingWarmEntry.from_document(raw_entry)
        if entry.key in entries:
            raise QwenWarmGrowthError("growing index key is duplicated")
        cell = bank.restore(entry.binding)
        if (
            cell.payload_sha256 != entry.cell_payload_sha256
            or hashlib.sha256(
                str(cell.cold_qwen_result.output).encode("utf-8")
            ).hexdigest()
            != entry.output_sha256
        ):
            raise QwenWarmGrowthError("growing index ResultCell binding diverged")
        cell.assert_feature_bound(entry.feature)
        entries[entry.key] = entry
    return entries, hashlib.sha256(raw).hexdigest()


@contextmanager
def _growth_lock(root: Path) -> Iterator[None]:
    path = root / ".qwen-growing-warm.lock"
    descriptor = os.open(
        path,
        os.O_CREAT
        | os.O_RDWR
        | int(getattr(os, "O_CLOEXEC", 0))
        | int(getattr(os, "O_NOFOLLOW", 0)),
        0o600,
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise QwenWarmGrowthError("growing warm lock is not a regular file")
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


class GrowingQwenWarmBank:
    """Append cold exact results and teach their mount action to OoE."""

    def __init__(
        self,
        *,
        root: Path,
        store: CrystalStore,
        bank: ResultCellBank,
        template_feature: QwenOoeFeatureReceipt,
        runtime_profile_sha256: str,
        controller_state_name: str,
        restore_controller: Callable[[], OoeController],
        bindings_by_feature: dict[str, ResultCellBinding],
        entries: dict[tuple[str, str], GrowingWarmEntry],
        index_sha256: str | None,
        template_output_character_limit: int,
        prompt_token_verifier: Callable[[str, str], bool],
    ) -> None:
        self.root = root
        self.store = store
        self.bank = bank
        self.template_feature = template_feature
        self.runtime_profile_sha256 = require_sha256(
            runtime_profile_sha256,
            field="runtime_profile_sha256",
        )
        self.controller_state_name = controller_state_name
        self.restore_controller = restore_controller
        self.bindings_by_feature = bindings_by_feature
        self.entries = entries
        self.index_sha256 = index_sha256
        def verify_template_observation(row: Any) -> bool:
            try:
                cell = self.bank.restore_payload(row.cell_payload_sha256)
                expected_output_sha256 = getattr(row, "output_sha256", None)
                if expected_output_sha256 is None:
                    expected_output = getattr(row, "output", None)
                    if not isinstance(expected_output, str):
                        return False
                    expected_output_sha256 = hashlib.sha256(
                        expected_output.encode("utf-8")
                    ).hexdigest()
                expected_output_sha256 = require_sha256(
                    expected_output_sha256,
                    field="template output sha256",
                )
            except Exception:
                return False
            output = str(cell.cold_qwen_result.output)
            return (
                hashlib.sha256(output.encode("utf-8")).hexdigest()
                == expected_output_sha256
                and cell.teacher_forward_count == row.teacher_forward_count
                and cell.binding.question_sha256 == row.question_sha256
            )

        template_store = CrystalStore(root / "private-parametric-state")
        self.templates = ParametricWarmBank(
            template_store,
            self.runtime_profile_sha256,
            output_character_limit=template_output_character_limit,
            observation_verifier=verify_template_observation,
            prompt_token_verifier=prompt_token_verifier,
        )
        import_totals = {
            "capacity_rejected_source_states": 0,
            "imported_multislot_observations": 0,
            "imported_single_slot_observations": 0,
            "skipped_source_states": 0,
            "source_states": 0,
        }
        for sibling in sorted(root.parent.iterdir(), key=lambda path: path.name):
            if sibling == root:
                continue
            try:
                metadata = sibling.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISDIR(metadata.st_mode):
                continue
            source_path = sibling / "private-parametric-state"
            try:
                source_metadata = source_path.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISDIR(source_metadata.st_mode):
                continue
            try:
                source = ParametricWarmBank.open_existing(
                    CrystalStore(source_path),
                    observation_verifier=verify_template_observation,
                )
                if source is None:
                    continue
                report = self.templates.import_compatible(source)
            except (
                CrystalStoreError,
                OSError,
                ParametricWarmError,
                TypeError,
                ValueError,
            ):
                import_totals["skipped_source_states"] += 1
                continue
            for field, value in report.items():
                import_totals[field] += value
        self.template_import = import_totals
        self.controller = restore_controller()
        self.hook: Any | None = None
        self._lock = threading.RLock()

    def resolve_binding(self, receipt: QwenOoeFeatureReceipt) -> ResultCellBinding:
        try:
            return self.bindings_by_feature[receipt.sha256]
        except KeyError as exc:
            raise QwenWarmGrowthError("no ResultCell for warm feature") from exc

    def feature_provider(
        self,
        question: str,
        metadata: Mapping[str, Any],
    ) -> QwenOoeFeatureReceipt | None:
        if not prefix_sinkhorn_warm_allowed(metadata):
            return None
        profile = _profile(metadata)
        if profile is None:
            return None
        entry = self.entries.get((profile, _question_sha256(question)))
        if entry is None:
            return None
        raw_token = metadata.get("qwen_token_sha256")
        if raw_token is None or require_sha256(
            raw_token,
            field="qwen_token_sha256",
        ) != entry.binding.rendered_prompt_token_sha256:
            return None
        return entry.feature

    def quality_verifier(
        self,
        receipt: QwenOoeFeatureReceipt,
        candidate: Any,
    ) -> bool:
        binding = self.bindings_by_feature.get(receipt.sha256)
        if binding is None:
            return False
        try:
            expected = ResultCellExecutor(self.bank, binding)(receipt)
        except Exception:
            return False
        return candidate.to_document() == expected.to_document()

    def direct_provider(
        self,
        question: str,
        metadata: Mapping[str, Any],
    ) -> Any:
        if not prefix_sinkhorn_warm_allowed(metadata):
            return None
        return self.templates.try_warm(question, metadata)

    @staticmethod
    def _history_feature_sha256s(controller: OoeController) -> frozenset[str]:
        document = json.loads(controller.snapshot_bytes())
        return frozenset(
            row["feature"]["sha256"]
            for site in document["body"]["sites"]
            for row in site["history"]
        )

    def _promote_all(self, controller: OoeController) -> None:
        for site_sha256 in controller.site_identity_sha256s:
            coverage = controller.coverage_receipt(site_sha256)
            covered = sum(
                count >= controller.config.min_coverage_per_source
                for count in coverage.per_source
            )
            if covered < controller.config.min_promoted_sources:
                continue
            controller.promote(
                site_sha256,
                coverage_sha256=coverage.sha256,
                verifier_sha256s=coverage.verifier_sha256s,
                expected_store_generation=self.store.manifest().generation,
            )

    def _publish_controller_growth(
        self,
        controller: OoeController,
        old_state_sha256: str,
    ) -> None:
        prepared = controller.save_snapshot(
            name=self.controller_state_name,
            expected_sha256=old_state_sha256,
        )
        self._promote_all(controller)
        completed = controller.save_snapshot(
            name=self.controller_state_name,
            expected_sha256=prepared.payload_sha256,
        )
        if self.hook is not None:
            self.hook.controller = controller
            self.hook._snapshot_sha256 = completed.payload_sha256

    def _reconcile_entries(
        self,
        controller: OoeController,
        entries: Mapping[tuple[str, str], GrowingWarmEntry],
    ) -> OoeController:
        known = self._history_feature_sha256s(controller)
        missing = sorted(
            (
                entry
                for entry in entries.values()
                if entry.feature.sha256 not in known
            ),
            key=lambda entry: entry.feature.temporal_index,
        )
        if not missing:
            return controller
        old_state = hashlib.sha256(
            self.store.restore_state(self.controller_state_name)
        ).hexdigest()
        for entry in missing:
            controller.ingest_teacher(entry.feature, entry.transition)
        self._publish_controller_growth(controller, old_state)
        return controller

    def reconcile(self) -> None:
        with self._lock, _growth_lock(self.root):
            self.entries, self.index_sha256 = load_growing_index(
                self.store,
                self.bank,
            )
            for entry in self.entries.values():
                self.bindings_by_feature[entry.feature.sha256] = entry.binding
            controller = self.restore_controller()
            self.controller = self._reconcile_entries(controller, self.entries)

    def _build_entry(
        self,
        controller: OoeController,
        question: str,
        profile: str,
        qwen_result: Result,
        final_result: Result,
    ) -> GrowingWarmEntry:
        sanitized_qwen = Result(
            qwen_result.status,
            qwen_result.component,
            output=qwen_result.output,
            reason=qwen_result.reason,
            evidence=_json_safe(qwen_result.evidence),
        )
        binding = _binding_from_qwen_result(question, sanitized_qwen)
        attached = attach_cold_qwen_generation_receipt(
            sanitized_qwen,
            binding=binding,
        )
        fertig_status, judgment = _fertig_judgment(final_result)
        cell = ResultCell.from_cold(
            binding=binding,
            cold_qwen_result=attached,
            cold_final_result=final_result,
            cold_fertig_judgment=judgment,
            cold_fertig_status=fertig_status,
            evaluator_quality_contract_sha256=(
                GROWING_QUALITY_CONTRACT_SHA256
            ),
        )
        pair = cell.cold_generation_provenance_unit.pair_binding_sha256
        question_sha256 = binding.question_sha256
        measurement_sha256 = _digest(
            {
                "binding_sha256": binding.sha256,
                "profile_sha256": profile,
                "question_sha256": question_sha256,
                "schema": "immer.qwen3.8-prompt-native-measurement/v1",
            }
        )
        feature = QwenOoeFeatureReceipt(
            temporal_index=controller.last_temporal_index + 1,
            measurement_sha256=measurement_sha256,
            model_pin_sha256=binding.model_pin.sha256,
            weight_coordinate_sha256=(
                self.template_feature.weight_coordinate_sha256
            ),
            weight_graph_revision=self.template_feature.weight_graph_revision,
            atlas_graph_revision=controller.atlas_graph_revision,
            probe=ProbeIdentity(
                question_sha256=question_sha256,
                token_sha256=binding.rendered_prompt_token_sha256,
                family_sha256=self.template_feature.probe.family_sha256,
                label_source_sha256=(
                    self.template_feature.probe.label_source_sha256
                ),
            ),
            feature_schema_sha256=self.template_feature.feature_schema_sha256,
            action_schema_sha256=self.template_feature.action_schema_sha256,
            verifier_sha256s=(GROWING_FEATURE_VERIFIER_SHA256, pair),
            evidence_sha256s=(measurement_sha256, pair),
            o1_surprise=0.0,
            o1_learning_progress=0.0,
            feature_sketch=_prompt_sketch(
                question,
                self.template_feature.feature_dimensions,
            ),
        )
        cell.assert_feature_bound(feature)
        transition = VerifiedTeacherTransition(
            feature_receipt_sha256=feature.sha256,
            site_identity_sha256=feature.site_identity.sha256,
            source_action="qwen_fallback",
            target_action="mount_organ",
            verifier_sha256=pair,
            evidence_sha256=pair,
            quality_sha256=cell.execution_quality_sha256,
            verified_quality=True,
        )
        publication = self.bank.charge(cell)
        if publication.payload_sha256 != cell.payload_sha256:
            raise QwenWarmGrowthError("ResultCell publication changed payload")
        return GrowingWarmEntry(
            runtime_profile_sha256=profile,
            question_sha256=question_sha256,
            binding=binding,
            feature=feature,
            transition=transition,
            cell_payload_sha256=cell.payload_sha256,
            output_sha256=hashlib.sha256(
                str(sanitized_qwen.output).encode("utf-8")
            ).hexdigest(),
        )

    def observe_cold(
        self,
        question: str,
        metadata: Mapping[str, Any],
        qwen_result: Result,
        final_result: Result,
    ) -> dict[str, Any]:
        if not prefix_sinkhorn_warm_allowed(metadata):
            return {"status": "prefix-policy-miss"}
        profile = _profile(metadata)
        if profile != self.runtime_profile_sha256:
            return {"status": "profile-miss"}
        key = (profile, _question_sha256(question))
        with self._lock, _growth_lock(self.root):
            entries, index_sha256 = load_growing_index(self.store, self.bank)
            for entry in entries.values():
                self.bindings_by_feature[entry.feature.sha256] = entry.binding
            controller = self._reconcile_entries(
                self.restore_controller(),
                entries,
            )
            existing = entries.get(key)
            if existing is not None:
                cell = self.bank.restore(existing.binding)
                template = self.templates.observe(
                    question.strip(),
                    str(cell.cold_qwen_result.output),
                    question_sha256=existing.question_sha256,
                    cell_payload_sha256=existing.cell_payload_sha256,
                    teacher_forward_count=cell.teacher_forward_count,
                )
                self.entries = entries
                self.index_sha256 = index_sha256
                self.controller = controller
                return {
                    "cell_payload_sha256": existing.cell_payload_sha256,
                    "feature_sha256": existing.feature.sha256,
                    "status": "already-indexed",
                    "template": template,
                }
            old_state_sha256 = hashlib.sha256(
                self.store.restore_state(self.controller_state_name)
            ).hexdigest()
            entry = self._build_entry(
                controller,
                question.strip(),
                profile,
                qwen_result,
                final_result,
            )
            entries[key] = entry
            publication = self.store.publish_state(
                GROWING_INDEX_STATE,
                _index_bytes(entries),
                expected_sha256=index_sha256,
            )
            self.bindings_by_feature[entry.feature.sha256] = entry.binding
            charged_cell = self.bank.restore(entry.binding)
            template = self.templates.observe(
                question.strip(),
                str(charged_cell.cold_qwen_result.output),
                question_sha256=entry.question_sha256,
                cell_payload_sha256=entry.cell_payload_sha256,
                teacher_forward_count=charged_cell.teacher_forward_count,
            )
            controller.ingest_teacher(entry.feature, entry.transition)
            self._publish_controller_growth(controller, old_state_sha256)
            self.entries = entries
            self.index_sha256 = publication.payload_sha256
            self.controller = controller
            return {
                "cell_payload_sha256": entry.cell_payload_sha256,
                "feature_sha256": entry.feature.sha256,
                "index_entries": len(entries),
                "runtime_profile_sha256": profile,
                "status": "charged",
                "template": template,
                "teacher_forward_count": self.bank.restore(
                    entry.binding
                ).teacher_forward_count,
            }


__all__ = [
    "GROWING_INDEX_STATE",
    "GROWING_QUALITY_CONTRACT_SHA256",
    "GrowingQwenWarmBank",
    "GrowingWarmEntry",
    "QwenWarmGrowthError",
    "load_growing_index",
]
