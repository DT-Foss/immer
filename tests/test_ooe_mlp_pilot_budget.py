from __future__ import annotations

import hashlib
import json
import unittest

from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.mlp_pilot_budget import (
    MLP_PILOT_DECODE_REPORT_SCHEMA,
    MLP_PILOT_DECODE_REPORT_SCHEMA_V3,
    MlpPilotBudgetState,
    MlpPilotLayerBudgetConfig,
    MlpPilotLayerBudgetIntegrityError,
    MlpPilotLayerBudgetPolicy,
    fit_mlp_pilot_layer_budget,
    parse_mlp_pilot_decode_report,
    verify_mlp_pilot_layer_budget,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _report(
    label: str,
    *,
    layers: tuple[int, ...],
    prefix: str | None = None,
    equal: tuple[bool, ...] = (True, True),
    cosine: float = 0.97,
    relative_l2: float = 0.20,
    sparse_primary_bytes: int = 600,
    schema: str = MLP_PILOT_DECODE_REPORT_SCHEMA,
    policy_sha256: str | None = None,
) -> bytes:
    step_rows = []
    for step, top1_equal in enumerate(equal):
        full = tuple(range(10, 20))
        sparse = (
            (*full[:9], 20)
            if top1_equal
            else (99, *full[1:])
        )
        candidates = tuple(sorted(set(full) | set(sparse)))
        step_rows.append(
            {
                "candidate_logit_max_abs_error": 2.0 + step,
                "candidate_token_ids": list(candidates),
                "full_primary_weight_bytes": 1_000,
                "full_seconds": 2.0,
                "full_top10": list(full),
                "full_top10_values": [float(20 - row) for row in range(10)],
                "hidden_metrics": {
                    "cosine": cosine - step * 0.01,
                    "max_abs_error": 1.0 + step,
                    "relative_l2_error": relative_l2 + step * 0.01,
                },
                "input_token_id": 100 + step,
                "sparse_pilot_weight_bytes": 50,
                "sparse_primary_weight_bytes": sparse_primary_bytes,
                "sparse_seconds": 1.5,
                "sparse_top10": list(sparse),
                "sparse_top10_overlap": len(set(full) & set(sparse)),
                "sparse_top10_values": [float(20 - row) for row in range(10)],
                "sparse_transpose_weight_bytes": 50,
                "step": step,
                "top1_equal": top1_equal,
            }
        )
    steps = len(step_rows)
    overlaps = [row["sparse_top10_overlap"] for row in step_rows]
    cosines = [row["hidden_metrics"]["cosine"] for row in step_rows]
    l2 = [row["hidden_metrics"]["relative_l2_error"] for row in step_rows]
    body = {
        "affine_fit_sha256": _hash("affine"),
        "candidate_logit_max_abs_error": max(
            row["candidate_logit_max_abs_error"] for row in step_rows
        ),
        "full_decode_seconds": sum(row["full_seconds"] for row in step_rows),
        "full_primary_weight_bytes": 1_000 * steps,
        "hidden_cosine_mean": sum(cosines) / steps,
        "hidden_cosine_min": min(cosines),
        "hidden_relative_l2_max": max(l2),
        "hidden_relative_l2_mean": sum(l2) / steps,
        "model_pin_sha256": _hash("model"),
        "prefix_seconds": 3.0,
        "prefix_sha256": _hash(label if prefix is None else prefix),
        "prefix_top1_value": 12.0,
        "prefix_tokens": 32,
        "router_fit_sha256": _hash("router"),
        "sparse_layers": list(layers),
        "sparse_decode_seconds": sum(row["sparse_seconds"] for row in step_rows),
        "sparse_pilot_weight_bytes": 50 * steps,
        "sparse_primary_weight_bytes": sparse_primary_bytes * steps,
        "sparse_top10_overlap_mean": sum(overlaps) / steps,
        "sparse_transpose_weight_bytes": 50 * steps,
        "speedup_full_over_sparse": 4.0 / 3.0,
        "step_receipts": step_rows,
        "steps": steps,
        "teacher_forced_input_token_ids": [100 + row for row in range(steps)],
        "top1_equal_steps": sum(equal),
        "unused_full_auxiliary_bytes": 0,
    }
    if schema == MLP_PILOT_DECODE_REPORT_SCHEMA_V3:
        body["layer_budget_policy_sha256"] = policy_sha256
    document = {
        "body": body,
        "body_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
        "schema": schema,
    }
    return canonical_json_bytes(document) + b"\n"


def _config(**changes) -> MlpPilotLayerBudgetConfig:
    values = {
        "min_verified_steps": 2,
        "min_hidden_cosine": 0.90,
        "max_hidden_relative_l2": 0.40,
    }
    values.update(changes)
    return MlpPilotLayerBudgetConfig(**values)


class MlpPilotLayerBudgetTests(unittest.TestCase):
    def test_heldout_action_selection_and_brake_only_markov_state(self) -> None:
        wide = _report(
            "same-prefix",
            layers=(0, 9, 18, 27, 36, 45, 54, 63),
            equal=(True, False),
            sparse_primary_bytes=300,
        )
        safe = _report(
            "same-prefix",
            layers=(0, 63),
            sparse_primary_bytes=700,
        )
        policy = fit_mlp_pilot_layer_budget((wide, safe), config=_config())
        self.assertEqual(policy.selected_layers, (0, 63))
        self.assertEqual(policy.verified_steps, 2)
        self.assertEqual(
            next(row for row in policy.candidates if len(row.layers) == 8).rejection_reasons,
            ("short-horizon",),
        )
        prefix = _hash("same-prefix")
        self.assertEqual(policy.measured_input_token_ids(prefix), (100, 101))
        self.assertEqual(
            policy.authorize_teacher_forced(prefix, (100, 101)), (0, 63)
        )
        self.assertEqual(policy.authorize_teacher_forced(prefix, (100, 999)), ())
        self.assertEqual(
            policy.authorize_teacher_forced(_hash("unseen"), (100,)), ()
        )

        first = policy.decide(prefix, input_token_id=100)
        self.assertEqual(first.action, "sparse_mlp")
        self.assertEqual(first.state, MlpPilotBudgetState(1, False))
        token_brake = policy.decide(prefix, first.state, input_token_id=999)
        self.assertEqual(token_brake.reason, "token-path-divergence")
        brake = policy.decide(
            prefix, first.state, input_token_id=101, verifier_ok=False
        )
        self.assertEqual(brake.action, "qwen_fallback")
        self.assertEqual(brake.reason, "verifier-brake")
        latched = policy.decide(prefix, brake.state, input_token_id=101)
        self.assertEqual(latched.reason, "brake-latched")
        horizon = policy.decide(
            prefix, MlpPilotBudgetState(2, False), input_token_id=102
        )
        self.assertEqual(horizon.reason, "verified-horizon-exhausted")

        reopened = MlpPilotLayerBudgetPolicy.from_bytes(policy.to_bytes())
        self.assertEqual(reopened.to_bytes(), policy.to_bytes())
        verify_mlp_pilot_layer_budget(reopened, (wide, safe))
        tampered = json.loads(policy.to_bytes())
        tampered["body"]["measured_paths"][0]["input_token_ids"][0] += 1
        tampered["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(tampered["body"])
        ).hexdigest()
        with self.assertRaisesRegex(
            MlpPilotLayerBudgetIntegrityError, "validation failed"
        ):
            MlpPilotLayerBudgetPolicy.from_bytes(canonical_json_bytes(tampered))

    def test_resealed_derived_metric_tamper_is_rejected(self) -> None:
        report = json.loads(_report("tamper", layers=(0, 63)))
        report["body"]["top1_equal_steps"] = 0
        report["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(report["body"])
        ).hexdigest()
        with self.assertRaisesRegex(
            MlpPilotLayerBudgetIntegrityError, "differs from step receipts"
        ):
            parse_mlp_pilot_decode_report(canonical_json_bytes(report))

    def test_v3_report_binds_the_applied_layer_policy(self) -> None:
        policy_sha256 = _hash("layer-policy")
        evidence = parse_mlp_pilot_decode_report(
            _report(
                "v3",
                layers=(0, 63),
                schema=MLP_PILOT_DECODE_REPORT_SCHEMA_V3,
                policy_sha256=policy_sha256,
            )
        )
        self.assertEqual(evidence.report_schema, MLP_PILOT_DECODE_REPORT_SCHEMA_V3)
        self.assertEqual(evidence.layer_budget_policy_sha256, policy_sha256)

    def test_candidates_require_the_same_prefix_and_token_path(self) -> None:
        with self.assertRaisesRegex(
            MlpPilotLayerBudgetIntegrityError, "unequal prompt coverage"
        ):
            fit_mlp_pilot_layer_budget(
                (
                    _report("prefix-a", layers=(0, 63)),
                    _report("prefix-b", layers=(63,)),
                ),
                config=_config(),
            )
        with self.assertRaisesRegex(
            MlpPilotLayerBudgetIntegrityError, "same token path"
        ):
            fit_mlp_pilot_layer_budget(
                (
                    _report("candidate-a", prefix="same", layers=(0, 63)),
                    _report(
                        "candidate-b",
                        prefix="same",
                        layers=(63,),
                        equal=(True, True, True),
                    ),
                ),
                config=_config(),
            )

    def test_no_admissible_action_is_an_explicit_full_qwen_fallback(self) -> None:
        report = _report(
            "unsafe",
            layers=(0, 63),
            equal=(True, False),
        )
        policy = fit_mlp_pilot_layer_budget((report,), config=_config())
        self.assertEqual(policy.selected_layers, ())
        self.assertEqual(policy.verified_steps, 0)
        decision = policy.decide(_hash("unsafe"), input_token_id=100)
        self.assertEqual(decision.action, "qwen_fallback")
        self.assertEqual(decision.reason, "no-admissible-sparse-action")

    def test_later_divergence_shortens_instead_of_erasing_safe_prefix(self) -> None:
        report = _report(
            "prefix-safe",
            layers=(0, 9, 18, 27, 36, 45, 54, 63),
            equal=(True, True, False, True),
            sparse_primary_bytes=300,
        )
        policy = fit_mlp_pilot_layer_budget(
            (report,), config=_config(min_verified_steps=2)
        )
        self.assertEqual(policy.verified_steps, 2)
        self.assertEqual(policy.measured_input_token_ids(_hash("prefix-safe")), (100, 101))
        self.assertEqual(policy.selected_layers, (0, 9, 18, 27, 36, 45, 54, 63))


if __name__ == "__main__":
    unittest.main()
