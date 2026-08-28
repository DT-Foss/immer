from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from immer.runtimes.ooe.qwen_mlp_evidence import (
    CaptureManifest,
    CaptureManifestV2,
    QwenMlpEvidenceBank,
    QwenMlpEvidenceIntegrityError,
    canonical_all_layer_capture_plan,
    canonical_capture_plan,
    run_capture_manifest,
)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qwen38_o1_mlp_evidence.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("qwen38_o1_mlp_evidence", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hash(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode()).hexdigest()


class Qwen38O1MlpEvidenceScriptTests(unittest.TestCase):
    @staticmethod
    def _all_layer_inventory(module):
        prompt_splits = {
            "train": tuple(_hash(f"v2-train:{index}") for index in range(3)),
            "calibration": tuple(
                _hash(f"v2-calibration:{index}") for index in range(2)
            ),
            "holdout": tuple(_hash(f"v2-holdout:{index}") for index in range(5)),
        }
        projection = SimpleNamespace(output_dimensions=2)
        selected = {}
        base = "model.language_model.layers.63.input_layernorm"
        for split in ("train", "calibration", "holdout"):
            for prompt in prompt_splits[split]:
                target = SimpleNamespace(
                    module=base,
                    unit_kind="module",
                    unit_index=None,
                )
                coordinate = SimpleNamespace(
                    layer=63,
                    module=base,
                    tensor=f"{base}.weight",
                    head_index=None,
                    row_start=None,
                    row_end=None,
                    byte_length=None,
                    relative_byte_offset=0,
                )
                job = SimpleNamespace(
                    job_id=_hash(f"v2-job:{prompt}"),
                    layer=63,
                    prompt_sha256=prompt,
                    model_pin=_hash("v2-model"),
                    code_pin="v2-code",
                    intervention="passive",
                    probe_family=module.ALL_LAYER_AUTHORITY_FAMILY,
                    target=target,
                )
                spec = SimpleNamespace(
                    sha256=_hash(f"v2-spec:{prompt}"),
                    prompt_sha256=prompt,
                    prompt_token_ids=(1,),
                    intervention_mode="passive",
                    start_layer=0,
                    stop_layer=64,
                    coordinate=coordinate,
                    hidden_sketch=projection,
                    native_head_crsa=None,
                    prefix_sinkhorn_operator_capture=None,
                )
                selected[prompt] = (job, spec)
        return prompt_splits, projection, selected

    def test_manifest_is_exact_25_10_5_and_fixture_resume_is_status_only(self) -> None:
        module = _load_script()
        prompts = tuple(_hash(f"prompt:{index}") for index in range(5))
        manifest = CaptureManifest(
            model_pin_sha256=_hash("model-pin"),
            input_manifest_sha256=_hash("input-manifest"),
            prompt_sha256s=prompts,
            entries=canonical_capture_plan(prompts),
        )
        self.assertEqual(len(manifest.entries), 40)
        self.assertEqual(
            tuple(
                sum(row.split == split for row in manifest.entries)
                for split in ("train", "calibration", "holdout")
            ),
            (25, 10, 5),
        )
        self.assertEqual(
            tuple(
                dict.fromkeys(
                    row.layer for row in manifest.entries if row.split == "train"
                )
            ),
            (45, 18, 36, 63, 27),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            fixture_path = root / "fixture.json"
            manifest_path.write_bytes(manifest.to_bytes())
            fixture_path.write_text(
                json.dumps(
                    {
                        "hidden_dimension": 4,
                        "intermediate_dimension": 6,
                        "rows": 2,
                        "schema": module.FIXTURE_SCHEMA,
                        "seed_sha256": _hash("fixture-seed"),
                    }
                )
            )
            bank_root = root / "bank"
            status_path = root / "first-status.json"
            outputs = []
            for index in range(2):
                arguments = [
                    "--root",
                    str(bank_root),
                    "--manifest",
                    str(manifest_path),
                    "--fixture",
                    str(fixture_path),
                ]
                if index == 0:
                    arguments.extend(("--output", str(status_path)))
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    self.assertEqual(
                        module.main(arguments),
                        0,
                    )
                outputs.append(json.loads(stream.getvalue()))
            self.assertEqual(json.loads(status_path.read_bytes()), outputs[0])
            self.assertEqual(outputs[0]["new_publications"], 40)
            self.assertEqual(outputs[1]["new_publications"], 0)
            self.assertEqual(outputs[1]["split_counts"], [25, 10, 5])
            self.assertEqual(outputs[1]["receipt_count"], 40)
            self.assertTrue(outputs[1]["audit_clean"])
            rendered = json.dumps(outputs)
            self.assertNotIn("data_base64", rendered)
            self.assertEqual(
                set(outputs[1]),
                {
                    "audit_clean",
                    "budget_sha256",
                    "fixture",
                    "manifest_sha256",
                    "new_publications",
                    "receipt_count",
                    "referenced_tensor_bytes",
                    "schema",
                    "split_counts",
                    "state_sha256",
                },
            )
            bank = QwenMlpEvidenceBank(bank_root)

            class LiveFixtureRunner(module.FixtureRunner):
                capture_mode = "live-exact"

            fixture = json.loads(fixture_path.read_text())
            holdout_bank = QwenMlpEvidenceBank(root / "holdout-first-bank")
            with self.assertRaisesRegex(
                QwenMlpEvidenceIntegrityError, "25/10/5 state machine"
            ):
                run_capture_manifest(
                    manifest,
                    holdout_bank,
                    module.FixtureRunner(manifest, fixture),
                    allowed_splits=("holdout",),
                )
            self.assertEqual(holdout_bank.state().receipt_count, 0)
            self.assertTrue(holdout_bank.audit().clean)
            with self.assertRaisesRegex(
                QwenMlpEvidenceIntegrityError, "25/10/5 state machine"
            ):
                run_capture_manifest(
                    manifest,
                    bank,
                    LiveFixtureRunner(manifest, fixture),
                )
            self.assertEqual(bank.state().receipt_count, 40)
            self.assertTrue(bank.audit().clean)
            live_bank = QwenMlpEvidenceBank(root / "live-bank")
            live_publications = run_capture_manifest(
                manifest,
                live_bank,
                LiveFixtureRunner(manifest, fixture),
            )
            self.assertEqual(len(live_publications), 40)
            self.assertEqual(live_bank.state().split_counts, (25, 10, 5))
            self.assertEqual(len(live_bank.build_subspace_corpus().groups), 40)
            self.assertEqual(
                len(
                    live_bank.build_subspace_corpus(
                        allowed_splits=("train", "calibration")
                    ).groups
                ),
                35,
            )
            self.assertEqual(
                len(
                    live_bank.build_subspace_corpus(allowed_splits=("holdout",)).groups
                ),
                5,
            )
            selected_rows = live_bank.build_subspace_corpus(
                allowed_splits=("holdout",),
                row_indices_by_prompt={prompt: (1,) for prompt in prompts},
            )
            self.assertEqual(sum(group.row_count for group in selected_rows.groups), 5)
            with self.assertRaisesRegex(ValueError, "prompt inventory"):
                live_bank.build_subspace_corpus(
                    allowed_splits=("holdout",),
                    row_indices_by_prompt={prompts[0]: (0,)},
                )
            holdout_receipt = next(
                receipt
                for receipt, _verification in live_bank.committed_pairs()
                if receipt.entry.split == "holdout"
            )
            holdout_ref = holdout_receipt.tensors[0]
            holdout_path = (
                root
                / "live-bank"
                / "objects"
                / holdout_ref.object_sha256[:2]
                / f"{holdout_ref.object_sha256}.tensor"
            )
            tampered = bytearray(holdout_path.read_bytes())
            tampered[-1] ^= 1
            holdout_path.chmod(0o600)
            holdout_path.write_bytes(tampered)
            deferred = QwenMlpEvidenceBank(
                root / "live-bank", deferred_tensor_splits=("holdout",)
            )
            self.assertEqual(
                len(
                    deferred.build_subspace_corpus(
                        allowed_splits=("train", "calibration")
                    ).groups
                ),
                35,
            )
            with self.assertRaises(QwenMlpEvidenceIntegrityError):
                deferred.audit()
            with self.assertRaisesRegex(
                QwenMlpEvidenceIntegrityError,
                "one live manifest/model/verifier pin",
            ):
                bank.build_subspace_corpus()

            orphan = bank.publish_tensor(
                "mlp.input", 63, np.ones((1, 1, 3), dtype=np.float32)
            )
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                module.main(
                    [
                        "--root",
                        str(bank_root),
                        "--manifest",
                        str(manifest_path),
                        "--fixture",
                        str(fixture_path),
                    ]
                )
            orphan_status = json.loads(stream.getvalue())
            self.assertFalse(orphan_status["audit_clean"])
            self.assertEqual(orphan_status["new_publications"], 0)
            self.assertIn(orphan.object_sha256, bank.audit().orphan_objects)

    def test_manifest_roundtrip_and_tampered_plan_rejected(self) -> None:
        prompts = tuple(_hash(f"prompt:{index}") for index in range(5))
        manifest = CaptureManifest(
            _hash("model"),
            _hash("inputs"),
            prompts,
            canonical_capture_plan(prompts),
        )
        self.assertEqual(
            CaptureManifest.from_bytes(manifest.to_bytes()).to_bytes(),
            manifest.to_bytes(),
        )
        document = json.loads(manifest.to_bytes())
        document["body"]["entries"][0]["split"] = "holdout"
        document["body_sha256"] = _hash_from_body(document["body"])
        with self.assertRaises((ValueError, QwenMlpEvidenceIntegrityError)):
            CaptureManifest.from_bytes(_canonical(document))

    def test_exact_prompt_cohort_selects_five_hashes(self) -> None:
        module = _load_script()
        prompts = tuple(_hash(f"cohort-prompt:{index}") for index in range(5))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cohort = root / "prompts.json"
            cohort.write_text(
                json.dumps(
                    {
                        "prompts": [
                            {
                                "sha256": prompt,
                                "spec_defaults": {},
                                "token_ids": [index + 1],
                            }
                            for index, prompt in enumerate(reversed(prompts))
                        ]
                    }
                )
            )
            args = module._parser().parse_args(
                [
                    "--root",
                    str(root / "bank"),
                    "--cartography-root",
                    str(root / "cartography"),
                    "--prompt-cohort",
                    str(cohort),
                    "--authority-only",
                ]
            )
            self.assertEqual(
                module._requested_prompts(args, root / "bank"), tuple(sorted(prompts))
            )

    def test_all_layer_cohorts_follow_introduction_history_not_registry_sort(self) -> None:
        module = _load_script()
        base = tuple(f"{2 * index + 2:064x}" for index in range(5))
        later = tuple(f"{2 * index + 1:064x}" for index in range(5))
        unrelated = tuple(f"{100 + index:064x}" for index in range(5))

        class FakeCartography:
            @staticmethod
            def _manifest_prompts(body):
                return tuple(body["prompts"])

            @staticmethod
            def _prompt_record(value):
                return dict(value)

            @staticmethod
            def _load_frontier_document(_root, *, manifest_sha256):
                self.assertEqual(manifest_sha256, _hash("base-manifest"))
                return (
                    {
                        "body": {
                            "events": [
                                {
                                    "body": {
                                        "added_jobs": [{"ignored": True}],
                                        "added_prompts": [
                                            {"sha256": value} for value in unrelated
                                        ],
                                    },
                                    "sha256": _hash("generation:1"),
                                },
                                {
                                    "body": {
                                        "added_jobs": [],
                                        "added_prompts": [
                                            {"sha256": value}
                                            for value in reversed(later)
                                        ],
                                    },
                                    "sha256": _hash("generation:2"),
                                },
                                {
                                    "body": {
                                        "added_jobs": [
                                            {
                                                "job": {
                                                    "probe_family": (
                                                        module.ALL_LAYER_AUTHORITY_FAMILY
                                                    ),
                                                    "prompt_sha256": value,
                                                }
                                            }
                                            for value in (*base, *later)
                                        ],
                                        "added_prompts": [],
                                    },
                                    "sha256": _hash("generation:3"),
                                },
                            ]
                        }
                    },
                    _hash("frontier-file"),
                )

        # The effective registry would interleave these ten hashes; it is not an
        # input to cohort recovery at all.
        effective_registry = tuple(sorted(base + later + unrelated))
        self.assertNotEqual(effective_registry[:5], tuple(sorted(base)))
        splits, generation = module._all_layer_prompt_splits(
            FakeCartography(),
            cartography_root=Path("/ignored"),
            base_manifest={"sha256": _hash("base-manifest")},
            base_body={
                "prompts": [{"sha256": value} for value in reversed(base)]
            },
        )
        ordered_base = tuple(sorted(base))
        self.assertEqual(splits["train"], ordered_base[:3])
        self.assertEqual(splits["calibration"], ordered_base[3:])
        self.assertEqual(splits["holdout"], tuple(sorted(later)))
        self.assertEqual(generation, _hash("generation:2"))

    def test_all_layer_jobs_require_exact_v2_family_and_job_identity(self) -> None:
        module = _load_script()
        prompt_splits, _projection, selected = self._all_layer_inventory(module)
        prepared = list(selected.values())
        old_job, old_spec = prepared[0]
        old_job = SimpleNamespace(
            **{
                **old_job.__dict__,
                "job_id": _hash("old-composite-job"),
                "probe_family": "contextual.mlp-all-layer-authority",
            }
        )
        result = module._select_all_layer_jobs(
            [*prepared, (old_job, old_spec)],
            prompt_splits=prompt_splits,
            model_pin_sha256=_hash("v2-model"),
            code_revision="v2-code",
        )
        self.assertEqual(set(result), set(selected))

        only_old = [
            (
                SimpleNamespace(
                    **{
                        **job.__dict__,
                        "probe_family": "contextual.mlp-all-layer-authority",
                    }
                ),
                spec,
            )
            for job, spec in prepared
        ]
        with self.assertRaisesRegex(module.CliError, "exact ten"):
            module._select_all_layer_jobs(
                only_old,
                prompt_splits=prompt_splits,
                model_pin_sha256=_hash("v2-model"),
                code_revision="v2-code",
            )

        wrong_job, wrong_spec = prepared[0]
        wrong_job = SimpleNamespace(
            **{
                **wrong_job.__dict__,
                "target": SimpleNamespace(
                    module="model.language_model.layers.62.input_layernorm",
                    unit_kind="module",
                    unit_index=None,
                ),
            }
        )
        with self.assertRaisesRegex(module.CliError, "job changed"):
            module._select_all_layer_jobs(
                [(wrong_job, wrong_spec), *prepared[1:]],
                prompt_splits=prompt_splits,
                model_pin_sha256=_hash("v2-model"),
                code_revision="v2-code",
            )

    def test_all_layer_jobs_expand_to_640_unique_source_target_cells(self) -> None:
        module = _load_script()
        prompt_splits, projection, selected = self._all_layer_inventory(module)
        measurements = {
            prompt: SimpleNamespace(sha256=_hash(f"measurement:{prompt}"))
            for prompt in selected
        }
        atlas = SimpleNamespace(contains_revision=lambda _revision: True)
        observations = []
        for prompt, measurement in measurements.items():
            for layer in range(64):
                receipt = SimpleNamespace(
                    sha256=_hash(f"receipt:{prompt}:{layer}"),
                    source_state=f"qwen.layer.{layer}.mlp.input-sketch",
                    target_state=f"qwen.layer.{layer}.mlp.output-sketch",
                    emitter_sha256=module.QWEN_CONTEXT_EMITTER_SHA256,
                    intervention_mode="passive",
                    granularity="operator",
                    segment_start=None,
                    segment_end=None,
                    model_pin_sha256=_hash("v2-model"),
                    atlas_revision=SimpleNamespace(
                        sha256=_hash(f"atlas-revision:{prompt}")
                    ),
                    input_sha256=_hash(f"input:{prompt}:{layer}"),
                    output_sha256=_hash(f"output:{prompt}:{layer}"),
                )
                observations.append(
                    SimpleNamespace(
                        measurement=measurement,
                        receipt=receipt,
                        input_array=np.zeros((1, 2), dtype=np.float64),
                        output_array=np.ones((1, 2), dtype=np.float64),
                    )
                )
        cells = module._all_layer_observation_cells(
            observations,
            selected=selected,
            measurements=measurements,
            model_pin_sha256=_hash("v2-model"),
            atlas=atlas,
            projection=projection,
        )
        self.assertEqual(len(cells), 640)
        self.assertEqual(len({row.receipt.sha256 for row in cells.values()}), 640)
        outcomes = {
            job.job_id: SimpleNamespace(
                atlas_receipt_sha256=_hash(f"atlas:{prompt}"),
                attempt_id=_hash(f"attempt:{prompt}"),
            )
            for prompt, (job, _spec) in selected.items()
        }
        scheduler_receipts = {
            prompt: _hash(f"scheduler:{prompt}") for prompt in selected
        }
        base_authority, later_authority = (
            module._all_layer_input_authority_documents(
                base_manifest_sha256=_hash("base-manifest"),
                generation_sha256=_hash("later-generation"),
                prompt_splits=prompt_splits,
                selected=selected,
                outcomes=outcomes,
                measurements=measurements,
                scheduler_observation_sha256s=scheduler_receipts,
                cells=cells,
                model_pin_sha256=_hash("v2-model"),
            )
        )
        changed_holdout_receipts = dict(scheduler_receipts)
        changed_holdout_receipts[prompt_splits["holdout"][0]] = _hash(
            "changed-holdout-scheduler"
        )
        changed_base, changed_later = module._all_layer_input_authority_documents(
            base_manifest_sha256=_hash("base-manifest"),
            generation_sha256=_hash("later-generation"),
            prompt_splits=prompt_splits,
            selected=selected,
            outcomes=outcomes,
            measurements=measurements,
            scheduler_observation_sha256s=changed_holdout_receipts,
            cells=cells,
            model_pin_sha256=_hash("v2-model"),
        )
        self.assertEqual(changed_base, base_authority)
        self.assertNotEqual(changed_later, later_authority)

        for field, bad_value in (
            ("source_state", "qwen.layer.0.attention.input-sketch"),
            ("target_state", "qwen.layer.0.attention.output-sketch"),
        ):
            first = observations[0]
            bad_receipt = SimpleNamespace(
                **{**first.receipt.__dict__, field: bad_value}
            )
            corrupted = [
                SimpleNamespace(
                    measurement=first.measurement,
                    receipt=bad_receipt,
                    input_array=first.input_array,
                    output_array=first.output_array,
                ),
                *observations[1:],
            ]
            with self.subTest(field=field), self.assertRaisesRegex(
                module.CliError, "lacks one exact"
            ):
                module._all_layer_observation_cells(
                    corrupted,
                    selected=selected,
                    measurements=measurements,
                    model_pin_sha256=_hash("v2-model"),
                    atlas=atlas,
                    projection=projection,
                )

    def test_all_layer_authority_only_reports_ten_measurements_and_640_cells(self) -> None:
        module = _load_script()
        prompt_splits, _projection, _selected = self._all_layer_inventory(module)
        manifest = CaptureManifestV2(
            model_pin_sha256=_hash("v2-model"),
            base_input_manifest_sha256=_hash("base-authority"),
            later_generation_input_manifest_sha256=_hash("later-authority"),
            train_prompt_sha256s=prompt_splits["train"],
            calibration_prompt_sha256s=prompt_splits["calibration"],
            holdout_prompt_sha256s=prompt_splits["holdout"],
            entries=canonical_all_layer_capture_plan(
                prompt_splits["train"],
                prompt_splits["calibration"],
                prompt_splits["holdout"],
            ),
        )
        runtime = SimpleNamespace(close=mock.Mock())
        context = {
            "authorities": {(str(index), index): None for index in range(640)},
            "authority_measurements": 10,
            "base_manifest_sha256": _hash("base-manifest"),
            "capture_manifest": manifest,
            "frontier_head_sha256": _hash("frontier-head"),
            "harvester_state_sha256": _hash("harvester"),
            "input_authority_sha256": _hash("later-authority"),
            "runtime": runtime,
            "scheduler_sha256": _hash("scheduler"),
        }
        fake_cartography = SimpleNamespace(
            _root_lock=lambda _root: contextlib.nullcontext()
        )
        args = module._parser().parse_args(
            [
                "--root",
                "/tmp/v2-bank",
                "--cartography-root",
                "/tmp/v2-cartography",
                "--capture-plan",
                "all-layer-v2",
                "--authority-only",
            ]
        )
        with mock.patch.object(
            module, "_cartography_module", return_value=fake_cartography
        ), mock.patch.object(module, "_load_live_context", return_value=context):
            status = module._run_live_all_layer(args)
        self.assertEqual(status["authority_measurements"], 10)
        self.assertEqual(status["authority_cells"], 640)
        self.assertEqual(status["capture_plan"], "all-layer-v2")
        runtime.close.assert_called_once_with()

    def test_all_layer_cached_analysis_reauthenticates_every_layer_artifact(self) -> None:
        module = _load_script()
        prompt_splits, _projection, _selected = self._all_layer_inventory(module)
        manifest = CaptureManifestV2(
            model_pin_sha256=_hash("v2-model"),
            base_input_manifest_sha256=_hash("base-authority"),
            later_generation_input_manifest_sha256=_hash("later-authority"),
            train_prompt_sha256s=prompt_splits["train"],
            calibration_prompt_sha256s=prompt_splits["calibration"],
            holdout_prompt_sha256s=prompt_splits["holdout"],
            entries=canonical_all_layer_capture_plan(
                prompt_splits["train"],
                prompt_splits["calibration"],
                prompt_splits["holdout"],
            ),
        )
        state = SimpleNamespace(split_counts=manifest.split_targets, sha256=_hash("state"))
        bank = SimpleNamespace(state=lambda: state)
        lock = SimpleNamespace(
            layer_seals=tuple(range(64)), sha256=_hash("calibration-lock")
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for layer in range(64):
                full = f"full:{layer}".encode()
                holdout = f"holdout:{layer}".encode()
                full_path = module._all_layer_fit_path(root, "full", layer)
                holdout_path = module._all_layer_fit_path(root, "holdout", layer)
                full_path.parent.mkdir(parents=True, exist_ok=True)
                holdout_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_bytes(full)
                holdout_path.write_bytes(holdout)
                rows.append(
                    {
                        "corpus_sha256": _hash(f"corpus:{layer}"),
                        "fit_sha256": hashlib.sha256(full).hexdigest(),
                        "holdout_sha256": hashlib.sha256(holdout).hexdigest(),
                        "layer": layer,
                        "promoted_model_sha256s": [],
                    }
                )
            body = {
                "calibration_lock_sha256": lock.sha256,
                "complete_state_sha256": state.sha256,
                "layers": rows,
                "manifest_sha256": manifest.sha256,
                "model_pin_sha256": manifest.model_pin_sha256,
            }
            (root / module.ALL_LAYER_ANALYSIS_NAME).write_bytes(
                module._sealed_status_document(module.ALL_LAYER_ANALYSIS_SCHEMA, body)
            )
            self.assertEqual(
                module._all_layer_analysis(
                    bank_root=root,
                    bank=bank,
                    manifest=manifest,
                    config=SimpleNamespace(),
                    calibration_lock=lock,
                ),
                body,
            )
            module._all_layer_fit_path(root, "holdout", 17).unlink()
            with self.assertRaisesRegex(module.CliError, "lost a referenced"):
                module._all_layer_analysis(
                    bank_root=root,
                    bank=bank,
                    manifest=manifest,
                    config=SimpleNamespace(),
                    calibration_lock=lock,
                )


def _canonical(value: object) -> bytes:
    from immer.runtimes.ooe.identity import canonical_json_bytes

    return canonical_json_bytes(value)


def _hash_from_body(value: object) -> str:
    import hashlib

    return hashlib.sha256(_canonical(value)).hexdigest()


if __name__ == "__main__":
    unittest.main()
