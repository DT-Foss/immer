"""Certified chat policy joining the local Qwen runtime and exact FERTIG.

The component owns only ``chat``.  It gives a closed exact certificate first
refusal, then lets Qwen answer once and adjudicates that complete answer as one
numeric candidate.  It never extracts a convenient number from prose.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ..contracts import Component, ExecutionStatus, Request, Result
from ..runtimes.qwen3_8.action_bank import (
    physical_external_drafter_executed,
    physical_lm_head_coordinate_executed,
    physical_prefix_sinkhorn_executed,
)
from .fertig.adapter import (
    CandidateVerification,
    CandidateVerificationStatus,
    CertifiedAnswer,
    FertigSolver,
    canonical_numeric_candidate,
)

if TYPE_CHECKING:
    from ..runtimes.ooe.chat import OoeChatHook


_EXACT_CERTIFICATE_KINDS = frozenset({"fraction_rref/v1", "guarded_formula/v1"})
_EXACT_EVIDENCE_KEYS = frozenset(
    {"answer", "certificates", "kind", "source_sha256", "verified"}
)
_FORMULA_KEYS = frozenset(
    {
        "context",
        "equation",
        "family",
        "inputs",
        "kind",
        "numeric_coverage",
        "numeric_spans",
        "source_sha256",
        "verified",
    }
)
_FORMULA_SPAN_KEYS = frozenset({"end", "start", "text"})
_RREF_KEYS = frozenset(
    {
        "component_variables",
        "equation_count",
        "kind",
        "pivot_columns",
        "rank",
        "residuals",
        "variable_count",
        "verified",
        "zero_residuals",
    }
)
_RREF_RESIDUAL_KEYS = frozenset({"constraint_index", "span", "value"})
_RREF_SPAN_KEYS = frozenset({"end", "start"})


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _text_sha256(value: str) -> str:
    return _sha256(value.encode("utf-8"))


def _stable_value(value: object) -> object:
    """Convert arbitrary component evidence to deterministic receipt data."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"invalid_float": str(value)}
    if isinstance(value, Mapping):
        rows: dict[str, object] = {}
        for key in sorted(value, key=lambda item: str(item)):
            rows[str(key)] = _stable_value(value[key])
        return rows
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    value_type = type(value)
    return {"unsupported_type": f"{value_type.__module__}.{value_type.__qualname__}"}


def _failure(exc: Exception) -> dict[str, str]:
    return {
        "detail": str(exc),
        "error": f"{type(exc).__module__}.{type(exc).__qualname__}",
    }


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _span_coordinates(
    span: Mapping[str, Any],
    *,
    keys: frozenset[str],
    question: str,
) -> tuple[int, int]:
    if frozenset(span) != keys:
        raise ValueError("FERTIG certificate span schema is invalid")
    start, end = span.get("start"), span.get("end")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or start < 0
        or end <= start
        or end > len(question)
    ):
        raise ValueError("FERTIG certificate span coordinates are invalid")
    return start, end


def _validate_formula_certificate(
    certificate: Mapping[str, Any], *, question: str, question_sha256: str
) -> None:
    if frozenset(certificate) != _FORMULA_KEYS:
        raise ValueError("guarded-formula certificate schema is invalid")
    if certificate.get("verified") is not True:
        raise ValueError("guarded-formula certificate is not verified")
    if certificate.get("numeric_coverage") is not True:
        raise ValueError("guarded-formula certificate lacks numeric coverage")
    if certificate.get("source_sha256") != question_sha256:
        raise ValueError("guarded-formula certificate is bound to another question")
    if not isinstance(certificate.get("family"), str) or not certificate["family"]:
        raise ValueError("guarded-formula certificate family is invalid")
    if not isinstance(certificate.get("equation"), str) or not certificate["equation"]:
        raise ValueError("guarded-formula certificate equation is invalid")
    inputs = certificate.get("inputs")
    if not isinstance(inputs, Mapping) or not inputs:
        raise ValueError("guarded-formula certificate inputs are invalid")
    if any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        or canonical_numeric_candidate(value) is None
        for key, value in inputs.items()
    ):
        raise ValueError("guarded-formula certificate input binding is invalid")
    context = certificate.get("context")
    if not isinstance(context, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in context.items()
    ):
        raise ValueError("guarded-formula certificate context is invalid")
    spans = certificate.get("numeric_spans")
    if not isinstance(spans, list) or not spans:
        raise ValueError("guarded-formula certificate numeric spans are invalid")
    previous_end = -1
    for span in spans:
        if not isinstance(span, Mapping):
            raise TypeError("guarded-formula numeric span must be a mapping")
        start, end = _span_coordinates(span, keys=_FORMULA_SPAN_KEYS, question=question)
        text = span.get("text")
        if (
            start < previous_end
            or not isinstance(text, str)
            or text != question[start:end]
            or canonical_numeric_candidate(text) is None
        ):
            raise ValueError("guarded-formula numeric spans are inconsistent")
        previous_end = end


def _validate_rref_certificate(
    certificate: Mapping[str, Any], *, question: str
) -> None:
    if frozenset(certificate) != _RREF_KEYS:
        raise ValueError("fraction-RREF certificate schema is invalid")
    if (
        certificate.get("verified") is not True
        or certificate.get("zero_residuals") is not True
    ):
        raise ValueError("fraction-RREF certificate is not verified")
    variable_count = certificate.get("variable_count")
    equation_count = certificate.get("equation_count")
    rank = certificate.get("rank")
    if not _positive_int(variable_count) or not _positive_int(equation_count):
        raise ValueError("fraction-RREF certificate dimensions are invalid")
    if rank != variable_count:
        raise ValueError("fraction-RREF certificate is not full rank")
    variables = certificate.get("component_variables")
    if (
        not isinstance(variables, list)
        or len(variables) != variable_count
        or len(set(variables)) != variable_count
        or any(not isinstance(value, str) or not value for value in variables)
    ):
        raise ValueError("fraction-RREF component variables are invalid")
    pivots = certificate.get("pivot_columns")
    if (
        not isinstance(pivots, list)
        or len(pivots) != variable_count
        or len(set(pivots)) != variable_count
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            or value >= variable_count
            for value in pivots
        )
    ):
        raise ValueError("fraction-RREF pivot columns are invalid")
    residuals = certificate.get("residuals")
    if not isinstance(residuals, list) or len(residuals) != equation_count:
        raise ValueError("fraction-RREF residual count is invalid")
    indices: set[int] = set()
    for residual in residuals:
        if (
            not isinstance(residual, Mapping)
            or frozenset(residual) != _RREF_RESIDUAL_KEYS
        ):
            raise ValueError("fraction-RREF residual schema is invalid")
        index = residual.get("constraint_index")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index in indices
            or residual.get("value") != "0"
        ):
            raise ValueError("fraction-RREF residual is invalid")
        indices.add(index)
        span = residual.get("span")
        if span is not None:
            if not isinstance(span, Mapping):
                raise TypeError("fraction-RREF residual span must be a mapping")
            _span_coordinates(span, keys=_RREF_SPAN_KEYS, question=question)


def _validate_exact_evidence(
    evidence: object,
    *,
    answer: str,
    question: str,
) -> Mapping[str, Any]:
    if not isinstance(evidence, Mapping):
        raise TypeError("FERTIG exact evidence must be a mapping")
    if frozenset(evidence) != _EXACT_EVIDENCE_KEYS:
        raise ValueError("FERTIG exact evidence schema is invalid")
    if evidence.get("kind") != "fertig-exact-solution/v1":
        raise ValueError("FERTIG exact evidence kind is invalid")
    if evidence.get("verified") is not True:
        raise ValueError("FERTIG exact evidence is not verified")
    if evidence.get("answer") != answer:
        raise ValueError("FERTIG exact evidence answer differs from its judgment")
    question_sha256 = _text_sha256(question)
    source_sha256 = evidence.get("source_sha256")
    if not _is_sha256(source_sha256):
        raise ValueError("FERTIG exact evidence source digest is invalid")
    if source_sha256 != question_sha256:
        raise ValueError("FERTIG exact evidence is bound to another question")
    certificates = evidence.get("certificates")
    if not isinstance(certificates, list) or not certificates:
        raise ValueError("FERTIG exact evidence has no certificates")
    for certificate in certificates:
        if not isinstance(certificate, Mapping):
            raise TypeError("FERTIG exact certificate must be a mapping")
        if certificate.get("verified") is not True:
            raise ValueError("FERTIG exact certificate is not verified")
        kind = certificate.get("kind")
        if kind not in _EXACT_CERTIFICATE_KINDS:
            raise ValueError("FERTIG exact certificate kind is not recognized")
        if kind == "guarded_formula/v1":
            _validate_formula_certificate(
                certificate, question=question, question_sha256=question_sha256
            )
        else:
            _validate_rref_certificate(certificate, question=question)
    return evidence


def _qwen_summary(result: Result) -> dict[str, Any]:
    evidence = _stable_value(dict(result.evidence))
    assert isinstance(evidence, dict)
    output = result.output if isinstance(result.output, str) else None
    summary: dict[str, Any] = {
        "component": result.component,
        "evidence_sha256": _sha256(_canonical_json(evidence)),
        "output_sha256": None if output is None else _text_sha256(output),
        "reason": result.reason,
        "status": result.status.value,
    }
    for key in ("execution", "model", "revision", "thinking", "tokenizer_sha256"):
        if key in evidence:
            summary[key] = evidence[key]
    bundle = evidence.get("bundle")
    if isinstance(bundle, dict):
        summary["bundle_sha256"] = _sha256(_canonical_json(bundle))
    generation = evidence.get("generation")
    if isinstance(generation, dict):
        summary["generation"] = {
            key: generation[key]
            for key in (
                "forward_passes",
                "generated_tokens",
                "linear_calls",
                "output_tokens_per_second",
                "prompt_tokens",
                "seconds",
                "source_body_bytes",
                "stopped_on_eos",
                "time_to_first_token_seconds",
                "token_trace_sha256",
            )
            if key in generation
        }
        summary["generation_sha256"] = _sha256(_canonical_json(generation))
    action_directive = evidence.get("inference_action_directive")
    if isinstance(action_directive, dict):
        summary["inference_action_directive"] = action_directive
    prefix_sinkhorn = evidence.get("prefix_sinkhorn")
    if isinstance(prefix_sinkhorn, dict):
        summary["prefix_sinkhorn"] = prefix_sinkhorn
    exact_head = evidence.get("exact_head")
    if isinstance(exact_head, dict):
        summary["exact_head"] = exact_head
    contextual_continuation = evidence.get("contextual_continuation")
    if isinstance(contextual_continuation, dict):
        summary["contextual_continuation"] = {
            key: contextual_continuation[key]
            for key in (
                "capture_count",
                "cell_count",
                "feedback_count",
                "hit_positions",
                "identity_sha256",
                "settlements",
                "state_sha256",
                "support",
                "verified_positions",
            )
            if key in contextual_continuation
        }
    layer_contextual_continuation = evidence.get("layer_contextual_continuation")
    if isinstance(layer_contextual_continuation, dict):
        compact_layer_context = {
            key: layer_contextual_continuation[key]
            for key in (
                "accounting_error",
                "identity_sha256",
                "schema",
                "status",
            )
            if key in layer_contextual_continuation
        }
        request = layer_contextual_continuation.get("request")
        if isinstance(request, dict):
            compact_layer_context["request"] = {
                key: request[key]
                for key in (
                    "accounting_error",
                    "crystal_accepted_tokens",
                    "crystal_mismatches",
                    "crystal_proposed_tokens",
                    "crystal_verified_tokens",
                    "layers",
                    "schema",
                    "selected",
                    "status",
                )
                if key in request
            }
            bank_work = request.get("bank_work")
            if isinstance(bank_work, dict):
                compact_layer_context["request"]["bank_work"] = {
                    key: bank_work[key]
                    for key in (
                        "crystal_captures",
                        "crystal_failures",
                        "crystal_option_calls",
                        "crystal_queries",
                        "crystal_query_hits",
                        "layers",
                        "status",
                    )
                    if key in bank_work
                }
        metrics = layer_contextual_continuation.get("metrics")
        if isinstance(metrics, dict):
            compact_layer_context["metrics"] = {
                key: metrics[key]
                for key in (
                    "crystal_accepted_tokens",
                    "crystal_bank_cells",
                    "crystal_bank_support",
                    "crystal_captures",
                    "crystal_failures",
                    "crystal_queries",
                    "crystal_query_hits",
                    "layers",
                    "receipt_count",
                    "settlements",
                    "state_sha256",
                )
                if key in metrics
            }
        summary["layer_contextual_continuation"] = compact_layer_context
    delta_head_router = evidence.get("delta_head_router")
    if isinstance(delta_head_router, dict):
        compact_delta_head = {
            key: delta_head_router[key]
            for key in (
                "head_dim",
                "layers",
                "max_selected_heads",
                "q4_manifest_sha256",
                "route_policy",
                "schema",
                "state_persistent",
                "value_heads",
                "width_actions",
            )
            if key in delta_head_router
        }
        request = delta_head_router.get("request")
        if isinstance(request, dict):
            compact_delta_head["request"] = {
                key: request[key]
                for key in (
                    "calls",
                    "full_equivalent_bytes",
                    "logical_bytes_saved",
                    "rows",
                    "selected_blocks",
                    "selected_heads",
                    "sinkhorn_projections",
                    "transitions",
                )
                if key in request
            }
        summary["delta_head_router"] = compact_delta_head
    draft = evidence.get("draft")
    if isinstance(draft, dict):
        compact_draft = {
            key: draft[key]
            for key in (
                "accepted_draft_tokens",
                "aux_source_body_bytes",
                "configured_mode",
                "draft_linear_calls",
                "draft_source_body_bytes",
                "external_linear_calls",
                "external_source_body_bytes",
                "markov_linear_calls",
                "markov_source_body_bytes",
                "mode",
                "proposed_draft_tokens",
                "rounds",
                "shared_linear_calls",
                "shared_source_body_bytes",
                "state_reuse_provider_downgrade",
                "target_forward_passes",
                "target_source_body_bytes",
                "target_linear_calls",
                "total_linear_calls",
                "total_source_body_bytes",
                "used_window_sizes",
                "window_size",
            )
            if key in draft
        }
        provider = draft.get("provider")
        if isinstance(provider, dict):
            markov = provider.get("markov")
            if not isinstance(markov, dict) and "atlas_contexts" in provider:
                markov = provider
            if isinstance(markov, dict):
                compact_draft["atlas"] = {
                    key: markov[key]
                    for key in (
                        "atlas_accepted_tokens",
                        "atlas_contexts",
                        "atlas_corpus_tokens",
                        "atlas_draft_tokens",
                        "atlas_option_calls",
                        "atlas_vote_calls",
                        "atlas_vote_max_score",
                        "atlas_vote_score_sum",
                        "atlas_vote_supported_tokens",
                        "atlas_vote_tokens",
                    )
                    if key in markov
                }
                compact_draft["online_memory"] = {
                    key: markov[key]
                    for key in (
                        "history_capacity_tokens",
                        "learned_tokens",
                        "online_vote_calls",
                        "online_vote_max_score",
                        "online_vote_score_sum",
                        "online_vote_supported_tokens",
                        "online_vote_tokens",
                        "last_retention_priority",
                        "retention_failures",
                        "retention_priority_evictions",
                        "retention_scored_episodes",
                    )
                    if key in markov
                }
                compact_draft["context_crystal"] = {
                    key: markov[key]
                    for key in (
                        "crystal_accepted_tokens",
                        "crystal_bank_cells",
                        "crystal_bank_support",
                        "crystal_captures",
                        "crystal_enabled",
                        "crystal_failures",
                        "crystal_last_cell_sha256",
                        "crystal_last_cosine",
                        "crystal_last_margin",
                        "crystal_mismatches",
                        "crystal_option_calls",
                        "crystal_proposed_tokens",
                        "crystal_queries",
                        "crystal_query_hits",
                        "crystal_verified_tokens",
                    )
                    if key in markov
                }
            compact_draft["provider"] = {
                key: provider[key]
                for key in (
                    "atlas_consensus_confidence_gain",
                    "atlas_consensus_rounds",
                    "atlas_consensus_tokens",
                    "external_linear_calls",
                    "external_source_body_bytes",
                    "last_qwen35_complete_wave_probability",
                    "markov_rounds",
                    "markov_selections",
                    "mtp_shared_linear_calls",
                    "mtp_shared_source_body_bytes",
                    "mtp_wave_gate_checks",
                    "mtp_wave_gate_passes",
                    "mtp_wave_gate_rejections",
                    "mtp_wave_gate_unknown",
                    "last_mtp_complete_wave_probability",
                    "mtp_rounds",
                    "mtp_selections",
                    "online_consensus_confidence_gain",
                    "online_consensus_rounds",
                    "online_consensus_tokens",
                    "provider_switches",
                    "provider_tournament_qwen35_selections",
                    "qwen35_accepted_tokens",
                    "qwen35_init_failures",
                    "qwen35_proposed_tokens",
                    "qwen35_rounds",
                    "qwen35_selections",
                    "qwen35_wave_gate_checks",
                    "qwen35_wave_gate_passes",
                    "qwen35_wave_gate_rejections",
                    "qwen35_wave_gate_unknown",
                    "selected_provider",
                    "shared_linear_calls",
                    "shared_source_body_bytes",
                    "target_linear_calls",
                    "target_source_body_bytes",
                )
                if key in provider
            }
            qwen35 = provider.get("qwen35")
            if isinstance(qwen35, dict):
                compact_draft["provider"]["qwen35"] = {
                    key: qwen35[key]
                    for key in (
                        "accepted_prefix_0",
                        "accepted_prefix_1",
                        "accepted_prefix_2",
                        "accepted_prefix_3",
                        "accepted_prefix_4",
                        "accepted_prefix_counts",
                        "committed_tokens",
                        "draft_calls",
                        "extension_calls",
                        "linear_calls",
                        "pending",
                        "poisoned",
                        "prefill_calls",
                        "reconcile_calls",
                        "restaged_blocks",
                        "schema",
                        "source_body_bytes",
                        "state_bytes",
                        "window_size",
                    )
                    if key in qwen35
                }
            mtp = provider.get("mtp")
            if isinstance(mtp, dict):
                compact_draft["provider"]["mtp"] = {
                    key: mtp[key]
                    for key in (
                        "accepted_tokens",
                        "draft_steps",
                        "linear_calls",
                        "proposed_tokens",
                        "seconds",
                        "source_body_bytes",
                        "verified_proposal_tokens",
                    )
                    if key in mtp
                }
        summary["draft"] = compact_draft
        summary["draft_sha256"] = _sha256(_canonical_json(draft))
    retention = evidence.get("o1_markov_retention")
    if isinstance(retention, dict):
        summary["o1_markov_retention"] = retention
    runtime_metrics = evidence.get("runtime_metrics")
    if isinstance(runtime_metrics, dict):
        summary["runtime_metrics"] = {
            key: runtime_metrics[key]
            for key in (
                "generation_wall_seconds",
                "major_page_faults",
                "minor_page_faults",
                "physical_read_bytes",
                "process_current_rss_bytes",
                "process_peak_rss_bytes",
                "system_cpu_seconds",
                "user_cpu_seconds",
            )
            if key in runtime_metrics
        }
        component_timings = runtime_metrics.get("component_timings")
        if isinstance(component_timings, dict):
            summary["runtime_metrics"]["component_timings"] = component_timings
    conversation = evidence.get("conversation")
    if isinstance(conversation, dict):
        summary["conversation"] = {
            key: conversation[key]
            for key in (
                "history_turns",
                "mtp_carry_bytes",
                "mtp_carry_reused_tokens",
                "mtp_carry_status",
                "prompt_suffix_tokens",
                "reuse_status",
                "reused_prefix_tokens",
                "state_retained_tokens",
            )
            if key in conversation
        }
    anchor_cache = evidence.get("anchor_cache")
    if isinstance(anchor_cache, dict):
        compact_anchor = {
            key: anchor_cache[key]
            for key in (
                "checkpoint_read_sweeps_saved",
                "forward_passes_saved",
                "mtp_carry_bytes",
                "mtp_carry_status",
                "prefix_tokens",
                "prefill_weight_sweeps_saved",
                "prompt_token_layer_evaluations_saved",
                "snapshot_bytes_read",
                "status",
            )
            if key in anchor_cache
        }
        anchor = anchor_cache.get("anchor")
        if isinstance(anchor, dict):
            compact_anchor["anchor"] = {
                key: anchor[key]
                for key in (
                    "prefix_length",
                    "prefix_sha256",
                    "receipt_sha256",
                    "state_bytes",
                )
                if key in anchor
            }
        summary["anchor_cache"] = compact_anchor
    q4 = evidence.get("q4")
    if isinstance(q4, dict):
        request = q4.get("request")
        if isinstance(request, dict):
            summary["q4"] = {
                "request": {
                    key: request[key]
                    for key in (
                        "logical_weight_bytes",
                        "page_mlp_prefetch_bytes",
                        "page_mlp_rows",
                        "page_mlp_selected_pages",
                        "page_mlp_weight_bytes",
                        "mapping_discard_bytes",
                        "mapping_discard_calls",
                        "resident_page_admissions",
                        "resident_page_bypasses",
                        "resident_page_hits",
                        "resident_page_misses",
                        "release_touched_calls",
                        "release_touched_nanoseconds",
                        "resident_unprotected_discard_bytes",
                        "resident_unprotected_discard_calls",
                        "resident_unprotected_discard_pages",
                    )
                    if key in request
                }
            }
        runtime = q4.get("runtime")
        if isinstance(runtime, dict):
            summary.setdefault("q4", {})["runtime"] = {
                key: runtime[key]
                for key in (
                    "resident_budget_bytes",
                    "resident_budget_overage_bytes",
                    "resident_head_fully_protected",
                    "resident_page_size",
                    "resident_peak_payload_bytes",
                    "resident_protected_bytes",
                    "resident_protected_pages",
                )
                if key in runtime
            }
    mlp_page = evidence.get("mlp_page_route")
    if isinstance(mlp_page, dict):
        request = mlp_page.get("request")
        if isinstance(request, dict):
            summary["mlp_page_route"] = {
                **{
                    key: mlp_page[key]
                    for key in ("page_count", "route_width")
                    if key in mlp_page
                },
                "request": {
                    key: request[key]
                    for key in (
                        "adaptive_width_pages_saved",
                        "dynamic_route_calls",
                        "dynamic_route_changes",
                        "exact_rows",
                        "physical_pages_saved",
                    )
                    if key in request
                },
            }
    runtime_reward = evidence.get("runtime_reward")
    if isinstance(runtime_reward, dict):
        summary["runtime_reward"] = {
            key: runtime_reward[key]
            for key in (
                "accepted_draft_tokens",
                "o1_priority",
                "page_actions",
                "page_actions_saved",
                "receipt_sha256",
                "reward",
                "schema",
            )
            if key in runtime_reward
        }
    return summary


def _certified_payload(certificate: CertifiedAnswer) -> dict[str, Any]:
    return {
        "answer": certificate.answer,
        "evidence": _stable_value(certificate.evidence),
        "kind": "fertig-certified-answer/v1",
    }


class QwenFertigChat:
    """One chat route with exact FERTIG first refusal and adjudication."""

    name = "qwen3.8.fertig-chat"
    capabilities = frozenset({"chat"})

    def __init__(
        self,
        qwen: Component,
        fertig: FertigSolver,
        *,
        ooe_hook: OoeChatHook | None = None,
    ) -> None:
        qwen_capabilities = getattr(qwen, "capabilities", frozenset())
        if "chat" not in qwen_capabilities:
            raise TypeError("qwen component must own the chat capability")
        if "exact_math" in qwen_capabilities:
            raise ValueError("wrapped Qwen component must not own exact_math")
        if not callable(getattr(qwen, "handle", None)):
            raise TypeError("qwen component must provide handle(request)")
        if type(fertig) is not FertigSolver:
            raise TypeError("fertig must be the production FertigSolver")
        if ooe_hook is not None:
            from ..runtimes.ooe.chat import (
                ChainedOoeChatHook,
                OoeChatHook as ProductionOoeChatHook,
            )

            if type(ooe_hook) not in {
                ProductionOoeChatHook,
                ChainedOoeChatHook,
            }:
                raise TypeError("ooe_hook must be the production OoeChatHook")
        self._qwen = qwen
        self._fertig = fertig
        self._ooe_hook = ooe_hook
        self._closed = False
        self._close_error: str | None = None
        self._lock = threading.RLock()

    @property
    def qwen(self) -> Component:
        return self._qwen

    @property
    def fertig(self) -> FertigSolver:
        return self._fertig

    @property
    def ooe_hook(self) -> OoeChatHook | None:
        return self._ooe_hook

    @property
    def loaded(self) -> bool:
        return bool(getattr(self._qwen, "loaded", False))

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def model_id(self) -> str | None:
        value = getattr(self._qwen, "model_id", None)
        return value if isinstance(value, str) else None

    @property
    def revision(self) -> str | None:
        value = getattr(self._qwen, "revision", None)
        return value if isinstance(value, str) else None

    def _result(
        self,
        *,
        question: str,
        route: str,
        output: object,
        status: ExecutionStatus,
        reason: str | None,
        candidate: str | None,
        qwen: dict[str, Any] | None,
        fertig: dict[str, Any],
        ooe: dict[str, Any] | None = None,
    ) -> Result:
        stable_output = _stable_value(output)
        output_sha256 = (
            _text_sha256(output)
            if isinstance(output, str)
            else _sha256(_canonical_json(stable_output))
        )
        core = {
            "candidate": candidate,
            "fertig": _stable_value(fertig),
            "kind": "qwen-fertig-chat-receipt/v1",
            "output_sha256": output_sha256,
            "question_sha256": _text_sha256(question),
            "qwen": qwen,
            "route": route,
        }
        if ooe is not None:
            core["ooe"] = _stable_value(ooe)
        receipt = {**core, "receipt_sha256": _sha256(_canonical_json(core))}
        return Result(
            status,
            self.name,
            output=output,
            reason=reason,
            evidence={"receipt": receipt},
        )

    def _attach_cold_observation(
        self,
        result: Result,
        *,
        question: str,
        metadata: Mapping[str, Any],
        qwen_result: Result,
        qwen_called: bool,
    ) -> Result:
        if self._ooe_hook is None or not qwen_called:
            return result
        try:
            observation = self._ooe_hook.observe_cold(
                question,
                metadata,
                qwen_result,
                result,
            )
            stable_observation = _stable_value(observation)
        except Exception as exc:
            stable_observation = {"status": "error", **_failure(exc)}
        evidence = dict(result.evidence)
        raw_receipt = evidence.get("receipt")
        if not isinstance(raw_receipt, Mapping):
            return result
        receipt = dict(raw_receipt)
        ooe = receipt.get("ooe")
        ooe_evidence = {} if not isinstance(ooe, Mapping) else dict(ooe)
        ooe_evidence["cold_observer"] = stable_observation
        receipt["ooe"] = ooe_evidence
        core = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        receipt["receipt_sha256"] = _sha256(_canonical_json(core))
        evidence["receipt"] = receipt
        return Result(
            result.status,
            result.component,
            output=result.output,
            reason=result.reason,
            evidence=evidence,
        )

    @staticmethod
    def _certificate(value: object, *, question: str) -> CertifiedAnswer:
        if not isinstance(value, CertifiedAnswer):
            raise TypeError("FERTIG certify returned an invalid certificate")
        if not value.answer or not isinstance(value.evidence, dict):
            raise TypeError("FERTIG certify returned an incomplete certificate")
        _validate_exact_evidence(
            value.evidence,
            answer=value.answer,
            question=question,
        )
        return value

    @staticmethod
    def _verification(
        value: object,
        *,
        candidate: str,
        question: str,
    ) -> CandidateVerification:
        if not isinstance(value, CandidateVerification):
            raise TypeError("FERTIG verify_candidate returned an invalid judgment")
        if not isinstance(value.status, CandidateVerificationStatus):
            raise TypeError("FERTIG judgment has an invalid status")
        if not isinstance(value.evidence, Mapping):
            raise TypeError("FERTIG judgment evidence must be a mapping")
        canonical_candidate = canonical_numeric_candidate(candidate)
        if value.candidate != canonical_candidate:
            raise ValueError("FERTIG judgment is bound to another candidate")
        question_sha256 = _text_sha256(question)
        if value.evidence.get("question_sha256") != question_sha256:
            raise ValueError("FERTIG judgment is bound to another question")
        exact_solution = value.evidence.get("exact_solution")
        if value.status is CandidateVerificationStatus.ABSTAINED:
            if value.expected is not None or exact_solution is not None:
                raise ValueError("abstained FERTIG judgment contains an exact claim")
            return value
        if not isinstance(value.expected, str) or not value.expected:
            raise ValueError("exact FERTIG judgment lacks an expected answer")
        _validate_exact_evidence(
            exact_solution,
            answer=value.expected,
            question=question,
        )
        if value.status is CandidateVerificationStatus.VERIFIED:
            if canonical_candidate is None or value.expected != canonical_candidate:
                raise ValueError(
                    "verified FERTIG judgment disagrees with its candidate"
                )
        elif value.expected == canonical_candidate:
            raise ValueError("mismatch FERTIG judgment equals its candidate")
        return value

    def handle(self, request: Request) -> Result:
        if request.capability not in self.capabilities:
            return Result(
                ExecutionStatus.REJECTED,
                self.name,
                reason="unsupported capability",
            )
        if not isinstance(request.payload, str) or not request.payload.strip():
            return Result(
                ExecutionStatus.REJECTED,
                self.name,
                reason="chat payload must be non-empty text",
            )
        question = request.payload.strip()
        action_directive = None
        raw_action_directive = (
            request.metadata.get("qwen_inference_action_directive")
            if isinstance(request.metadata, Mapping)
            else None
        )
        if raw_action_directive is not None:
            try:
                from ..runtimes.qwen3_8.action_bank import InferenceActionDirective

                action_directive = InferenceActionDirective.from_document(
                    raw_action_directive
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                return Result(
                    ExecutionStatus.REJECTED,
                    self.name,
                    reason=f"invalid inference action directive: {exc}",
                )
            if action_directive.question_sha256 != _text_sha256(question):
                return Result(
                    ExecutionStatus.REJECTED,
                    self.name,
                    reason="inference action directive belongs to another question",
                )

        def attach_action_directive(result: Result) -> Result:
            if action_directive is None:
                return result
            evidence = dict(result.evidence)
            receipt = evidence.get("receipt")
            body = receipt if isinstance(receipt, Mapping) else {}
            route = body.get("route")
            actions = []
            if route == "fertig_exact_short_circuit":
                actions.append("fertig_exact")
            elif isinstance(route, str) and route.startswith("ooe_"):
                qwen = body.get("qwen")
                actions.append(
                    "parametric_program"
                    if isinstance(qwen, Mapping)
                    and qwen.get("component") == "immer.markov-parametric-template"
                    else "stored_result"
                )
            else:
                qwen = body.get("qwen")
                if isinstance(qwen, Mapping):
                    if isinstance(qwen.get("mlp_page_route"), Mapping):
                        actions.append("dynamic_mlp_pages")
                    prefix_sinkhorn = qwen.get("prefix_sinkhorn")
                    if physical_prefix_sinkhorn_executed(prefix_sinkhorn):
                        actions.append("prefix_sinkhorn")
                    if physical_lm_head_coordinate_executed(qwen.get("exact_head")):
                        actions.append("lm_head_coordinate")
                    draft = qwen.get("draft")
                    if isinstance(draft, Mapping):
                        actions.append("target_verified_draft")
                    generation = qwen.get("generation")
                    if (
                        isinstance(generation, Mapping)
                        and isinstance(
                            generation.get("forward_passes"),
                            int,
                        )
                        and generation.get("forward_passes", 0) > 0
                    ):
                        actions.append("qwen_target")
                    if (
                        "qwen_target" in actions
                        and "target_verified_draft" in actions
                        and physical_external_drafter_executed(draft)
                    ):
                        actions.append("external_drafter")
            evidence["inference_action_directive"] = {
                "applied": {
                    "actions": sorted(actions),
                    "route": route,
                },
                "directive": action_directive.to_document(),
            }
            return Result(
                result.status,
                result.component,
                output=result.output,
                reason=result.reason,
                evidence=evidence,
            )

        with self._lock:
            if self._closed:
                return Result(
                    ExecutionStatus.UNAVAILABLE,
                    self.name,
                    reason="Qwen-FERTIG chat component is closed",
                )

            preflight: dict[str, Any]
            try:
                raw_certificate = self._fertig.certify(question)
                certificate = (
                    None
                    if raw_certificate is None
                    else self._certificate(raw_certificate, question=question)
                )
                preflight = {"status": "abstained"}
            except Exception as exc:
                certificate = None
                preflight = {"status": "error", **_failure(exc)}
            if certificate is not None:
                certified = _certified_payload(certificate)
                return attach_action_directive(
                    self._result(
                        question=question,
                        route="fertig_exact_short_circuit",
                        output=certificate.answer,
                        status=ExecutionStatus.OK,
                        reason=None,
                        candidate=None,
                        qwen=None,
                        fertig={"certificate": certified, "status": "certified"},
                    )
                )

            qwen_called = False
            candidate_origin = "qwen"
            ooe_summary: dict[str, Any] | None = None
            warm_result: Result | None = None
            warm_attempt: object | None = None
            if self._ooe_hook is not None:
                try:
                    attempt = self._ooe_hook.try_warm(question, request.metadata)
                    attempt_evidence = getattr(attempt, "evidence", None)
                    if isinstance(attempt_evidence, Mapping):
                        ooe_summary = {"warm": _stable_value(attempt_evidence)}
                    candidate_result = getattr(attempt, "result", None)
                    if isinstance(candidate_result, Result) and candidate_result.ok:
                        warm_result = candidate_result
                        warm_attempt = attempt
                except Exception as exc:
                    from ..runtimes.ooe.chat import OoeChatIntegrityError

                    if isinstance(exc, OoeChatIntegrityError):
                        return self._result(
                            question=question,
                            route="ooe_integrity_error",
                            output=None,
                            status=ExecutionStatus.ERROR,
                            reason="warm OoE integrity verification failed",
                            candidate=None,
                            qwen=None,
                            fertig={"preflight": preflight, "status": "not_run"},
                            ooe={
                                "warm": {
                                    "error": (
                                        f"{type(exc).__module__}."
                                        f"{type(exc).__qualname__}"
                                    ),
                                    "status": "integrity-error",
                                }
                            },
                        )
                    ooe_summary = {"warm": {"status": "error", **_failure(exc)}}

            if warm_result is not None:
                release_bypassed = getattr(
                    self._qwen,
                    "release_warm_bypass_state",
                    None,
                )
                if callable(release_bypassed):
                    try:
                        release_bypassed()
                    except Exception as exc:
                        return self._result(
                            question=question,
                            route="ooe_integrity_error",
                            output=None,
                            status=ExecutionStatus.ERROR,
                            reason="warm OoE state release failed",
                            candidate=None,
                            qwen=None,
                            fertig={"preflight": preflight, "status": "not_run"},
                            ooe={
                                "warm": {
                                    "status": "integrity-error",
                                    **_failure(exc),
                                }
                            },
                        )
                qwen_result = warm_result
                candidate_origin = "ooe"
            else:
                qwen_called = True
                qwen_request = Request("chat", question, request.metadata)
                try:
                    raw_qwen_result = self._qwen.handle(qwen_request)
                    if not isinstance(raw_qwen_result, Result):
                        raise TypeError("Qwen handle returned a non-Result value")
                    qwen_result = raw_qwen_result
                except Exception as exc:
                    qwen_result = Result(
                        ExecutionStatus.ERROR,
                        str(getattr(self._qwen, "name", "qwen")),
                        reason=f"Qwen chat failed: {type(exc).__name__}: {exc}",
                        evidence={"failure": _failure(exc)},
                    )
            try:
                qwen_summary = _qwen_summary(qwen_result)
            except Exception as exc:
                qwen_summary = {
                    "component": str(qwen_result.component),
                    "evidence": {"status": "invalid", **_failure(exc)},
                    "output_sha256": (
                        _text_sha256(qwen_result.output)
                        if isinstance(qwen_result.output, str)
                        else None
                    ),
                    "reason": qwen_result.reason,
                    "status": qwen_result.status.value,
                }

            def finish(result: Result) -> Result:
                return attach_action_directive(
                    self._attach_cold_observation(
                        result,
                        question=question,
                        metadata=request.metadata,
                        qwen_result=qwen_result,
                        qwen_called=qwen_called,
                    )
                )

            def settle_warm(*, accept: bool) -> Result | None:
                nonlocal ooe_summary
                if candidate_origin != "ooe":
                    return None
                assert self._ooe_hook is not None
                assert warm_attempt is not None
                try:
                    accounting = (
                        self._ooe_hook.commit_warm(warm_attempt)
                        if accept
                        else self._ooe_hook.reject_warm(warm_attempt)
                    )
                except Exception as exc:
                    return self._result(
                        question=question,
                        route="ooe_integrity_error",
                        output=None,
                        status=ExecutionStatus.ERROR,
                        reason="warm OoE accounting verification failed",
                        candidate=None,
                        qwen=None,
                        fertig={"preflight": preflight, "status": "not_run"},
                        ooe={
                            "warm": {
                                "error": (
                                    f"{type(exc).__module__}.{type(exc).__qualname__}"
                                ),
                                "status": "integrity-error",
                            }
                        },
                    )
                if ooe_summary is None:
                    ooe_summary = {}
                ooe_summary["accounting"] = {
                    **accounting.to_dict(),
                    "sha256": accounting.sha256,
                }
                return None

            if not qwen_result.ok:
                return finish(
                    self._result(
                        question=question,
                        route=f"{candidate_origin}_failure",
                        output=qwen_result.output,
                        status=qwen_result.status,
                        reason=qwen_result.reason,
                        candidate=(
                            qwen_result.output
                            if isinstance(qwen_result.output, str)
                            else None
                        ),
                        qwen=qwen_summary,
                        fertig={"preflight": preflight, "status": "not_run"},
                        ooe=ooe_summary,
                    )
                )
            if not isinstance(qwen_result.output, str) or not qwen_result.output:
                return finish(
                    self._result(
                        question=question,
                        route=f"{candidate_origin}_failure",
                        output=None,
                        status=ExecutionStatus.ERROR,
                        reason=(
                            f"{candidate_origin} chat returned an invalid "
                            "successful output"
                        ),
                        candidate=None,
                        qwen=qwen_summary,
                        fertig={"preflight": preflight, "status": "not_run"},
                        ooe=ooe_summary,
                    )
                )

            candidate = qwen_result.output
            try:
                verification = self._verification(
                    self._fertig.verify_candidate(question, candidate),
                    candidate=candidate,
                    question=question,
                )
            except Exception as exc:
                settlement_error = settle_warm(accept=False)
                if settlement_error is not None:
                    return settlement_error
                return finish(
                    self._result(
                        question=question,
                        route=f"{candidate_origin}_fertig_error",
                        output=candidate,
                        status=ExecutionStatus.OK,
                        reason=None,
                        candidate=candidate,
                        qwen=qwen_summary,
                        fertig={
                            "preflight": preflight,
                            "status": "error",
                            **_failure(exc),
                        },
                        ooe=ooe_summary,
                    )
                )

            fertig_receipt = {
                "preflight": preflight,
                "status": verification.status.value,
                "verification": _stable_value(verification.to_dict()),
            }
            if verification.status is CandidateVerificationStatus.VERIFIED:
                settlement_error = settle_warm(accept=True)
                if settlement_error is not None:
                    return settlement_error
                return finish(
                    self._result(
                        question=question,
                        route=f"{candidate_origin}_verified",
                        output=candidate,
                        status=ExecutionStatus.OK,
                        reason=None,
                        candidate=candidate,
                        qwen=qwen_summary,
                        fertig=fertig_receipt,
                        ooe=ooe_summary,
                    )
                )
            if verification.status is CandidateVerificationStatus.MISMATCH:
                assert verification.expected is not None
                settlement_error = settle_warm(accept=False)
                if settlement_error is not None:
                    return settlement_error
                return finish(
                    self._result(
                        question=question,
                        route=(
                            "fertig_mismatch_override"
                            if candidate_origin == "qwen"
                            else "ooe_fertig_mismatch_override"
                        ),
                        output=verification.expected,
                        status=ExecutionStatus.OK,
                        reason=None,
                        candidate=candidate,
                        qwen=qwen_summary,
                        fertig=fertig_receipt,
                        ooe=ooe_summary,
                    )
                )
            # Only an explicitly benchmark-authorized warm mount may count
            # savings when FERTIG has no contrary exact evidence.  Generic
            # hooks retain the conservative rejection contract.
            abstention_authorized = (
                candidate_origin == "ooe"
                and self._ooe_hook is not None
                and self._ooe_hook.abstention_commit_authorized(warm_attempt)
            )
            settlement_error = settle_warm(accept=abstention_authorized)
            if settlement_error is not None:
                return settlement_error
            return finish(
                self._result(
                    question=question,
                    route=f"{candidate_origin}_verification_abstained",
                    output=candidate,
                    status=ExecutionStatus.OK,
                    reason=None,
                    candidate=candidate,
                    qwen=qwen_summary,
                    fertig=fertig_receipt,
                    ooe=ooe_summary,
                )
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            close = getattr(self._qwen, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    self._close_error = f"{type(exc).__name__}: {exc}"

    def __enter__(self) -> QwenFertigChat:
        with self._lock:
            if self._closed:
                raise RuntimeError("Qwen-FERTIG chat component is closed")
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


__all__ = ["QwenFertigChat"]
