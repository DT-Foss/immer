from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

import immer.runtimes.qwen3_8.cartography_probe as cartography_module
from immer.knowledge import LiveGraph
from immer.runtimes.deepseek_v4.causal_weights import (
    CausalWeightMount,
    LogicalModelIdentity,
    tensor_range_plan_from_source,
)
from immer.runtimes.qwen3_8.cartography_probe import (
    HiddenSketchProjection,
    ProbeCoordinateSpec,
    ProbeResourceBudget,
    ProbeSpec,
    Qwen38CartographyBudgetError,
    Qwen38CartographyIntegrityError,
    Qwen38CartographyProbe,
    Qwen38CartographyProbeError,
    prompt_token_sha256,
)
from immer.runtimes.qwen3_8.bundle import QWEN38_BUNDLE_SCHEMA
from immer.runtimes.qwen3_8.model import StreamedQwen38
from immer.runtimes.qwen3_8.native_crsa import Qwen38NativeHeadCrsa
from immer.runtimes.qwen3_8.pager import Qwen38WeightPager
from immer.runtimes.qwen3_8.semantic_atlas import (
    GraphRevision,
    MeasurementReceipt,
    SemanticWeightAtlas,
)

from test_qwen3_8_model import _native_tiny_config, _tiny_config, _tiny_weights


_IDENTITY = LogicalModelIdentity(repo_id="local:cartography-test", revision="fixture")
_CODE_REVISION = "a" * 40
_LABEL_SOURCE_SHA256 = hashlib.sha256(b"external-door-label-source/v1").hexdigest()
_LABEL_EVIDENCE_SHA256 = hashlib.sha256(
    b"externally-verified-door-label-evidence/v1"
).hexdigest()
_SEMANTIC_LABEL = "arithmetic.addition"


def _write_bundle_manifest(
    root: Path,
    mount: CausalWeightMount,
    identity: LogicalModelIdentity,
) -> None:
    graph_revision = mount.graph.store.revision()
    manifest_body = {
        "checkpoint_complete": True,
        "graph_revision": list(graph_revision),
        "layout_fingerprint": mount.layout.layout_fingerprint,
        "logical_model": identity.as_record(),
        "weights_layout": "nested/v1",
    }
    encoded_body = json.dumps(
        manifest_body,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    (root / "bundle.json").write_text(
        json.dumps(
            {
                "body": manifest_body,
                "schema": QWEN38_BUNDLE_SCHEMA,
                "sha256": hashlib.sha256(encoded_body).hexdigest(),
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _tensor_sha(value: torch.Tensor) -> str:
    import hashlib

    raw = value.detach().contiguous().to("cpu").view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _committed_state(model: StreamedQwen38) -> tuple[object, ...]:
    rows: list[object] = []
    for state in model._layer_states:
        if state is None:
            rows.append(None)
            continue
        tensors = []
        for name in ("key", "value", "crsa_log_usage", "conv", "recurrent"):
            value = getattr(state, name, None)
            if isinstance(value, torch.Tensor):
                tensors.append(
                    (name, tuple(value.shape), str(value.dtype), _tensor_sha(value))
                )
        rows.append((type(state).__name__, tuple(tensors)))
    return (
        model.next_position,
        model.state_batch_size,
        model.state_poisoned,
        model.state_bytes,
        tuple(rows),
        model._pending_block_stage is not None,
    )


class Qwen38CartographyProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix=".qwen-cartography-test-", dir=Path.cwd()
        )
        root = Path(self.temporary.name)
        self.root = root
        (root / "weights").mkdir()
        (root / "causal").mkdir()
        self.config = _tiny_config()
        save_file(_tiny_weights(self.config), root / "weights" / "model.safetensors")
        self.mount = CausalWeightMount(root, _IDENTITY, budget_mb=100)
        names = tuple(
            str(row["name"]) for row in self.mount.source.inventory().get("tensors", ())
        )
        self.mount.bind_tensor_plans(
            tuple(
                tensor_range_plan_from_source(self.mount.source, name) for name in names
            )
        )
        _write_bundle_manifest(root, self.mount, _IDENTITY)
        self.pager = Qwen38WeightPager(
            self.mount.source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=2 * 1024**2,
            causal_tensor_reader=self.mount.tensor_reader,
        )
        self.model = StreamedQwen38(
            self.config, self.pager, max_batch_size=1, max_seq_len=16
        )
        self.executor = Qwen38CartographyProbe(self.model)
        self.atlas_graph = LiveGraph(root / "semantic-atlas")
        self.atlas_head = GraphRevision.from_live_revision(
            self.atlas_graph.store.revision()
        )

    def tearDown(self) -> None:
        self.pager.close()
        self.mount.close()
        self.temporary.cleanup()

    def _spec(self, **changes: object) -> ProbeSpec:
        tokens = (1, 4, 9)
        base = ProbeSpec(
            prompt_token_ids=tokens,
            prompt_sha256=prompt_token_sha256(tokens),
            start_layer=0,
            stop_layer=2,
            coordinate=ProbeCoordinateSpec(
                layer=0,
                module="model.language_model.layers.0.mlp.gate_proj",
                tensor="model.language_model.layers.0.mlp.gate_proj.weight",
                row_start=0,
                row_end=2,
            ),
            intervention_mode="passive",
            code_revision=_CODE_REVISION,
            hidden_sketch=HiddenSketchProjection(
                seed_sha256="b" * 64,
                output_dimensions=5,
            ),
        )
        return replace(base, **changes)

    def _execute(self, spec: ProbeSpec):
        return self.executor.execute(spec, atlas_head_revision=self.atlas_head)

    def test_passive_sweep_is_deterministic_and_never_commits(self) -> None:
        before = _committed_state(self.model)
        first = self._execute(self._spec())
        plans = tuple(
            self.mount.resolve_tensor_plan(str(row["name"]))
            for row in self.mount.source.inventory().get("tensors", ())
        )
        atlas = SemanticWeightAtlas(
            self.atlas_graph,
            model_pin=first.measurement.model_pin,
            tensor_plans=plans,
        )
        atlas.append_measurement(first.measurement)
        second = self.executor.execute(
            self._spec(), atlas_head_revision=atlas.revision()
        )

        self.assertEqual(first.measurement.model_pin, second.measurement.model_pin)
        self.assertEqual(
            first.measurement.weight_rail_revision,
            second.measurement.weight_rail_revision,
        )
        self.assertEqual(second.measurement.atlas_head_revision, atlas.revision())
        self.assertEqual(
            first.measurement.hidden_sha256, second.measurement.hidden_sha256
        )
        self.assertEqual(
            first.measurement.activation_sha256,
            second.measurement.activation_sha256,
        )
        self.assertEqual(
            first.measurement.state_sha256, second.measurement.state_sha256
        )
        self.assertEqual(
            first.measurement.numeric_summaries,
            second.measurement.numeric_summaries,
        )
        self.assertEqual(first.measurement.intervention.mode, "passive")
        self.assertIsNone(first.measurement.observed_semantic_label)
        self.assertEqual(_committed_state(self.model), before)
        first.verify()
        second.verify()

    def test_passive_external_label_is_recorded_and_evidence_bound(self) -> None:
        unlabeled = self._spec(hidden_sketch=None)
        spec = replace(
            unlabeled,
            label_source_sha256=_LABEL_SOURCE_SHA256,
            semantic_label=_SEMANTIC_LABEL,
            label_evidence_sha256=_LABEL_EVIDENCE_SHA256,
        )
        self.assertNotEqual(spec.sha256, unlabeled.sha256)
        self.assertEqual(spec.as_record()["semantic_label"], _SEMANTIC_LABEL)
        self.assertEqual(
            spec.as_record()["label_evidence_sha256"], _LABEL_EVIDENCE_SHA256
        )
        other_label = replace(spec, semantic_label="arithmetic.subtraction")
        other_evidence = replace(spec, label_evidence_sha256="c" * 64)
        self.assertNotEqual(spec.probe_identity, other_label.probe_identity)
        self.assertNotEqual(spec.probe_identity, other_evidence.probe_identity)

        result = self._execute(spec)
        measurement = result.measurement
        self.assertEqual(measurement.observation_status, "recorded")
        self.assertEqual(measurement.observed_semantic_label, _SEMANTIC_LABEL)
        self.assertEqual(
            measurement.probe.label_source_sha256,
            spec.probe_identity.label_source_sha256,
        )
        self.assertNotEqual(measurement.probe.label_source_sha256, _LABEL_SOURCE_SHA256)
        assertion = result.evidence_document["body"]["external_label_assertion"]
        self.assertEqual(
            assertion,
            {
                "applies_to_this_arm": True,
                "assertion_origin": "external",
                "label_evidence_sha256": _LABEL_EVIDENCE_SHA256,
                "label_source_sha256": _LABEL_SOURCE_SHA256,
                "model_output_used": False,
                "probe_label_binding_sha256": spec.probe_identity.label_source_sha256,
                "semantic_label": _SEMANTIC_LABEL,
            },
        )
        self.assertEqual(
            result.evidence_document["body"]["semantic_label_source"],
            "external-assertion-model-output-unused",
        )
        result.verify()

    def test_external_label_pair_and_default_source_are_strict(self) -> None:
        base = self._spec(hidden_sketch=None)
        with self.assertRaisesRegex(Qwen38CartographyProbeError, "supplied together"):
            replace(base, semantic_label=_SEMANTIC_LABEL)
        with self.assertRaisesRegex(Qwen38CartographyProbeError, "supplied together"):
            replace(base, label_evidence_sha256=_LABEL_EVIDENCE_SHA256)
        with self.assertRaisesRegex(
            Qwen38CartographyProbeError, "non-default external label source"
        ):
            replace(
                base,
                semantic_label=_SEMANTIC_LABEL,
                label_evidence_sha256=_LABEL_EVIDENCE_SHA256,
            )
        with self.assertRaisesRegex(
            Qwen38CartographyProbeError, "bounded canonical external text"
        ):
            replace(
                base,
                label_source_sha256=_LABEL_SOURCE_SHA256,
                semantic_label="  arithmetic.addition  ",
                label_evidence_sha256=_LABEL_EVIDENCE_SHA256,
            )

    def test_paired_placebo_has_exactly_zero_effect(self) -> None:
        result = self._execute(
            self._spec(intervention_mode="placebo", hidden_sketch=None)
        )
        control = result.control_measurement
        self.assertIsNotNone(control)
        assert control is not None
        self.assertEqual(result.measurement.intervention.mode, "placebo")
        self.assertEqual(control.intervention.mode, "off")
        self.assertEqual(result.measurement.hidden_sha256, control.hidden_sha256)
        self.assertEqual(result.measurement.state_sha256, control.state_sha256)
        self.assertEqual(
            result.measurement.activation_sha256, control.activation_sha256
        )
        self.assertEqual(result.measurement.placebo_effects, ())
        for layer in result.evidence_document["body"]["paired_comparisons"]:
            for boundary in ("pre", "post"):
                row = layer[boundary]
                self.assertTrue(row["exact_equal"])
                self.assertEqual(row["delta_l2"], 0.0)
                self.assertEqual(row["delta_max_abs"], 0.0)
                self.assertEqual(row["delta_mean_abs"], 0.0)

    def test_coordinate_and_access_receipts_bind_exact_causal_tensor_range(
        self,
    ) -> None:
        result = self._execute(self._spec(hidden_sketch=None))
        plan = self.mount.resolve_tensor_plan(
            "model.language_model.layers.0.mlp.gate_proj.weight"
        )
        coordinate = result.measurement.coordinate
        self.assertTrue(coordinate.matches_plan(plan))
        row_bytes = plan.length // plan.shape[0]
        self.assertEqual(coordinate.range_absolute_offset, plan.absolute_offset)
        self.assertEqual(coordinate.range_length, 2 * row_bytes)
        matching = [
            row for row in result.tensor_range_receipts if row.tensor == plan.name
        ]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0].absolute_offset, plan.absolute_offset)
        self.assertEqual(matching[0].length, plan.length)
        self.assertEqual(
            matching[0].tensor_plan_sha256,
            coordinate.tensor_plan_sha256,
        )

    def test_failure_preserves_preexisting_committed_model_state(self) -> None:
        self.model.hidden_stateful(((1, 4),))
        before = _committed_state(self.model)
        budget = ProbeResourceBudget(max_trace_operations=1)
        with self.assertRaisesRegex(
            Qwen38CartographyBudgetError, "max_trace_operations"
        ):
            self._execute(self._spec(budget=budget, hidden_sketch=None))
        self.assertEqual(_committed_state(self.model), before)

    def test_observer_mutation_fails_closed_without_state_leak(self) -> None:
        before = _committed_state(self.model)
        original = self.mount.source.set_access_observer
        calls = 0

        def mutating_set(observer: object, *, prepare_identity: bool = True) -> object:
            nonlocal calls
            calls += 1
            previous = original(observer, prepare_identity=prepare_identity)
            return object() if calls == 2 else previous

        with mock.patch.object(
            self.mount.source, "set_access_observer", side_effect=mutating_set
        ):
            with self.assertRaisesRegex(
                Qwen38CartographyIntegrityError, "observer changed"
            ):
                self._execute(self._spec(hidden_sketch=None))
        self.assertEqual(_committed_state(self.model), before)
        self.assertFalse(self.mount.source.metrics()["access_observer_enabled"])

    def test_manifest_and_runtime_source_symlinks_fail_closed(self) -> None:
        manifest = self.root / "bundle.json"
        manifest_target = self.root / "bundle-target.json"
        manifest.rename(manifest_target)
        manifest.symlink_to(manifest_target.name)
        before = _committed_state(self.model)
        with self.assertRaisesRegex(
            Qwen38CartographyIntegrityError, "non-symlink regular file"
        ):
            self._execute(self._spec(hidden_sketch=None))
        self.assertEqual(_committed_state(self.model), before)

        source_target = self.root / "runtime-target.py"
        source_target.write_text("value = 1\n", encoding="utf-8")
        source_link = self.root / "runtime-link.py"
        source_link.symlink_to(source_target.name)
        with self.assertRaisesRegex(
            Qwen38CartographyIntegrityError, "non-symlink regular file"
        ):
            cartography_module._file_sha256(source_link)

    def test_manifest_path_swap_during_descriptor_read_is_rejected(self) -> None:
        manifest = self.root / "bundle.json"
        replacement = self.root / "bundle-replacement.json"
        displaced = self.root / "bundle-displaced.json"
        replacement.write_bytes(manifest.read_bytes())
        manifest_identity = manifest.stat()
        real_read = os.read
        swapped = False

        def swapping_read(descriptor: int, length: int) -> bytes:
            nonlocal swapped
            chunk = real_read(descriptor, length)
            opened = os.fstat(descriptor)
            if not swapped and (opened.st_dev, opened.st_ino) == (
                manifest_identity.st_dev,
                manifest_identity.st_ino,
            ):
                swapped = True
                os.replace(manifest, displaced)
                os.replace(replacement, manifest)
            return chunk

        before = _committed_state(self.model)
        with mock.patch.object(
            cartography_module.os, "read", side_effect=swapping_read
        ):
            with self.assertRaisesRegex(
                Qwen38CartographyIntegrityError, "changed during"
            ):
                self._execute(self._spec(hidden_sketch=None))
        self.assertTrue(swapped)
        self.assertEqual(_committed_state(self.model), before)

    def test_native_layer27_pair_records_crsa_and_appends_control_first(self) -> None:
        root = Path(self.temporary.name) / "native"
        (root / "weights").mkdir(parents=True)
        (root / "causal").mkdir()
        config = _native_tiny_config()
        save_file(_tiny_weights(config), root / "weights" / "model.safetensors")
        identity = LogicalModelIdentity(
            repo_id="local:cartography-native-test", revision="fixture"
        )
        with CausalWeightMount(root, identity, budget_mb=200) as mount:
            names = tuple(
                str(row["name"]) for row in mount.source.inventory().get("tensors", ())
            )
            mount.bind_tensor_plans(
                tuple(
                    tensor_range_plan_from_source(mount.source, name) for name in names
                )
            )
            _write_bundle_manifest(root, mount, identity)
            pager = Qwen38WeightPager(
                mount.source,
                device="cpu",
                compute_dtype="bfloat16",
                max_resident_bytes=2 * 1024**2,
                causal_tensor_reader=mount.tensor_reader,
            )
            model = StreamedQwen38(config, pager, max_batch_size=1, max_seq_len=16)
            atlas_graph = LiveGraph(root / "semantic-atlas")
            atlas_head = GraphRevision.from_live_revision(atlas_graph.store.revision())
            tokens = (1, 4, 9)
            spec = ProbeSpec(
                prompt_token_ids=tokens,
                prompt_sha256=prompt_token_sha256(tokens),
                start_layer=27,
                stop_layer=28,
                coordinate=ProbeCoordinateSpec(
                    layer=27,
                    module="model.language_model.layers.27.self_attn",
                    tensor=("model.language_model.layers.27.self_attn.q_proj.weight"),
                ),
                intervention_mode="native",
                code_revision=_CODE_REVISION,
                label_source_sha256=_LABEL_SOURCE_SHA256,
                semantic_label=_SEMANTIC_LABEL,
                label_evidence_sha256=_LABEL_EVIDENCE_SHA256,
                native_head_crsa=Qwen38NativeHeadCrsa(alpha=0.01),
            )
            try:
                result = Qwen38CartographyProbe(model).execute(
                    spec, atlas_head_revision=atlas_head
                )
                control = result.control_measurement
                self.assertIsNotNone(control)
                assert control is not None
                self.assertEqual(control.intervention.mode, "placebo")
                self.assertEqual(result.measurement.intervention.mode, "native")
                self.assertEqual(control.observation_status, "recorded")
                self.assertIsNone(control.observed_semantic_label)
                self.assertEqual(result.measurement.observation_status, "eligible")
                self.assertEqual(
                    result.measurement.observed_semantic_label, _SEMANTIC_LABEL
                )
                self.assertTrue(result.measurement.placebo_effects)
                control_assertion = result.control_evidence_document["body"][
                    "external_label_assertion"
                ]
                self.assertEqual(control_assertion["assertion_origin"], "external")
                self.assertFalse(control_assertion["applies_to_this_arm"])
                self.assertIsNone(control_assertion["semantic_label"])
                self.assertEqual(
                    control_assertion["label_evidence_sha256"],
                    _LABEL_EVIDENCE_SHA256,
                )
                primary_assertion = result.evidence_document["body"][
                    "external_label_assertion"
                ]
                self.assertTrue(primary_assertion["applies_to_this_arm"])
                self.assertEqual(primary_assertion["semantic_label"], _SEMANTIC_LABEL)
                self.assertFalse(primary_assertion["model_output_used"])
                self.assertTrue(
                    result.control_evidence_document["body"]["crsa_evidence"][0][
                        "identity"
                    ]
                )
                self.assertFalse(
                    result.evidence_document["body"]["crsa_evidence"][0]["identity"]
                )
                atlas = SemanticWeightAtlas(
                    atlas_graph,
                    model_pin=result.measurement.model_pin,
                    tensor_plans=tuple(
                        mount.resolve_tensor_plan(name) for name in names
                    ),
                )
                appended = tuple(
                    atlas.append_measurement(receipt).appended
                    for receipt in result.measurements_in_append_order
                )
                self.assertEqual(appended, (True, True))
                labeled = atlas.query_by_semantic_label(_SEMANTIC_LABEL)
                self.assertEqual(labeled.measurements, (result.measurement,))
            finally:
                pager.close()

    def test_receipt_and_evidence_tampering_are_rejected(self) -> None:
        result = self._execute(self._spec(hidden_sketch=None))
        document = result.measurement.to_document()
        document["body"]["hidden_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            MeasurementReceipt.from_document(document)

        result.evidence_document["body"]["arm"] = "tampered"
        with self.assertRaisesRegex(
            Qwen38CartographyIntegrityError, "evidence document SHA-256 mismatch"
        ):
            result.verify()

    def test_prompt_hash_and_unsupported_head_fail_before_model_execution(self) -> None:
        base = self._spec(hidden_sketch=None)
        with self.assertRaisesRegex(
            Qwen38CartographyIntegrityError, "prompt token IDs"
        ):
            replace(base, prompt_sha256="0" * 64)

        coordinate = replace(base.coordinate, head_index=0)
        with mock.patch.object(
            self.model, "checkpoint_preflight", wraps=self.model.checkpoint_preflight
        ) as preflight:
            with self.assertRaisesRegex(Qwen38CartographyProbeError, "head-specific"):
                self._execute(replace(base, coordinate=coordinate))
        preflight.assert_not_called()


if __name__ == "__main__":
    unittest.main()
