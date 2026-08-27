from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np

from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.operator_transport import (
    OperatorTransportConfig,
    QwenPrefixSinkhornCaptureBank,
    QwenPrefixSinkhornCaptureReceipt,
    causal_prefix_sinkhorn_operator,
)
from immer.runtimes.qwen3_8.semantic_atlas import (
    GraphRevision,
    InterventionIdentity,
    MeasurementReceipt,
    ModelPin,
    NumericSummary,
    ProbeIdentity,
    RuntimeProvenance,
    TensorRangePlan,
    WeightCoordinate,
)


ROOT = Path(__file__).resolve().parents[1]


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _load_script():
    path = ROOT / "scripts" / "ooe_operator_transport_live.py"
    spec = importlib.util.spec_from_file_location("_operator_transport_live", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _measurement(index: int) -> MeasurementReceipt:
    code = "7" * 40
    plan = TensorRangePlan(
        "model.language_model.layers.27.self_attn.q_proj.weight",
        "BF16",
        (8, 8),
        "model.safetensors",
        1024,
        128,
    )
    return MeasurementReceipt(
        model_pin=ModelPin(
            "Qwen/Qwen3.8-27B",
            "fixture",
            _digest("bundle"),
            _digest("manifest"),
            code,
        ),
        coordinate=WeightCoordinate.from_plan(
            plan,
            layer=27,
            module="model.language_model.layers.27.self_attn.q_proj",
            head_index=2,
            row_start=0,
            row_end=4,
        ),
        probe=ProbeIdentity(
            _digest(f"question:{index}"),
            _digest(f"tokens:{index}"),
            _digest("family"),
            _digest("labels"),
        ),
        intervention=InterventionIdentity("native", _digest("intervention")),
        observation_status="recorded",
        observed_semantic_label=None,
        hidden_sha256=_digest(f"hidden:{index}"),
        activation_sha256=_digest(f"activation:{index}"),
        logits_sha256=_digest(f"logits:{index}"),
        state_sha256=_digest(f"state:{index}"),
        access_trace_sha256=_digest(f"trace:{index}"),
        evidence_sha256=_digest(f"evidence:{index}"),
        weight_rail_revision=GraphRevision(9, _digest("weights")),
        atlas_head_revision=GraphRevision(index, _digest(f"atlas-head:{index}")),
        numeric_summaries=(NumericSummary("signal", 1, 1.0, 1.0, 1.0, 1.0),),
        placebo_effects=(),
        runtime=RuntimeProvenance(
            code,
            _digest("sources"),
            _digest("dependencies"),
            _digest("runtime"),
            _digest("platform"),
        ),
    )


class LiveOperatorTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_script()
        cls.temporary = tempfile.TemporaryDirectory()
        cls.bank = QwenPrefixSinkhornCaptureBank(cls.temporary.name)
        coefficient = np.eye(4)
        coefficient[:3, :3] = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.01, 0.02, 0.99]]
        )
        inverse = np.linalg.inv(coefficient)
        for index in range(15):
            source = causal_prefix_sinkhorn_operator(
                np.random.default_rng(9000 + index).normal(size=(4, 4))
            )
            target = coefficient @ source @ inverse
            target[np.abs(target) < 1.0e-14] = 0.0
            target /= target.sum(axis=1, keepdims=True)
            other_a = causal_prefix_sinkhorn_operator(
                np.random.default_rng(10000 + index).normal(size=(4, 4))
            )
            other_b = causal_prefix_sinkhorn_operator(
                np.random.default_rng(11000 + index).normal(size=(4, 4))
            )
            cls.bank.publish(
                QwenPrefixSinkhornCaptureReceipt.create(
                    _measurement(index),
                    atlas_revision=GraphRevision(
                        index + 1, _digest(f"capture-atlas:{index}")
                    ),
                    capture_spec_sha256=_digest("capture-spec"),
                    attention_spec_sha256=_digest("attention-spec"),
                    operators=(source, target, other_a, other_b),
                )
            )
        cls.graph = GraphRevision(99, _digest("transport-graph"))
        cls.config = OperatorTransportConfig(
            maximum_transport_condition=1.03,
            minimum_nullspace_gap=60_500_000_000.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            cls.report = cls.module.build_report(
                cls.temporary.name,
                graph_revision=cls.graph,
                config=cls.config,
            )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_calibration_selects_the_only_structurally_accepted_pair(self) -> None:
        report = self.report
        self.assertTrue(self.module.verify_report(report, self.temporary.name))
        body = report["body"]
        accepted = [row for row in body["pair_records"] if row["status"] == "accepted"]
        rejected = [row for row in body["pair_records"] if row["status"] == "rejected"]
        self.assertEqual([row["pair"] for row in accepted], [[8, 2]])
        self.assertEqual(len(rejected), 11)
        self.assertTrue(all(row["rejection"] for row in rejected))
        self.assertEqual(body["selection"]["selected_pair"], [8, 2])
        self.assertFalse(body["selection"]["holdout_used"])
        self.assertEqual(body["holdout"]["evaluated_pairs"], [[8, 2]])
        self.assertEqual(
            set(body["holdout"]["metrics"]),
            {"per_head", "global", "identity", "random"},
        )
        self.assertEqual(len(body["capture_receipt_sha256s"]), 15)
        for field in ("corpus_sha256", "split_sha256", "fit_sha256"):
            self.assertEqual(len(accepted[0][field]), 64)
        self.assertEqual(len(body["holdout"]["holdout_sha256"]), 64)

    def test_resealed_winner_metric_and_output_collision_fail(self) -> None:
        winner = copy.deepcopy(self.report)
        winner["body"]["selection"]["selected_pair"] = [2, 8]
        winner["body_sha256"] = self.module._digest(winner["body"])
        with self.assertRaises(self.module.LiveOperatorTransportError):
            self.module.verify_report(winner, self.temporary.name)

        metric = copy.deepcopy(self.report)
        metric["body"]["holdout"]["metrics"]["per_head"][
            "mean_relative_residual"
        ] += 0.1
        metric["body_sha256"] = self.module._digest(metric["body"])
        with self.assertRaises(self.module.LiveOperatorTransportError):
            self.module.verify_report(metric, self.temporary.name)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report.json"
            self.module.write_report_no_replace(output, self.report)
            first = output.read_bytes()
            self.assertEqual(first, canonical_json_bytes(self.report))
            with self.assertRaises(FileExistsError):
                self.module.write_report_no_replace(output, self.report)
            self.assertEqual(output.read_bytes(), first)


if __name__ == "__main__":
    unittest.main()
