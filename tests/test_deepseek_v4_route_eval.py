from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from immer.runtimes.deepseek_v4.draft_verification import _freeze_selected_experts

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deepseek_v4_route_eval.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("deepseek_v4_route_eval", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import DeepSeek V4 route evaluation script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


route_eval = _load_script()


def _canonical(value) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _observation_id(request_id: str, layer: int) -> str:
    identity = {
        "layer": layer,
        "request_id": request_id,
        "schema": route_eval.ROUTE_OBSERVATION_SCHEMA,
    }
    return f"deepseek-v4-route:v1:{_digest(identity)}"


def _row(
    *,
    request_id: str,
    execution_digest: str,
    layer: int,
    sequence_digests: list[str],
    active_lengths: list[int],
    padded_length: int,
    offset: int = 0,
) -> dict:
    selected: list[list[int]] = []
    for sequence_index, active_length in enumerate(active_lengths):
        # Every prompt contains all four source IDs. Layer one repeats the
        # source exactly, so a real model sees a stable held-out transition.
        for token_index in range(active_length):
            expert = (token_index + offset) % 4
            selected.append([expert])
        selected.extend([[] for _ in range(padded_length - active_length)])
    return {
        "checkpoint": route_eval.OFFICIAL_CHECKPOINT.as_record(),
        "execution_feature_digest": execution_digest,
        "layer": layer,
        "observation_id": _observation_id(request_id, layer),
        "observation_schema": route_eval.ROUTE_OBSERVATION_SCHEMA,
        "request_id": request_id,
        "row_layout": {
            "active_lengths": list(active_lengths),
            "batch_size": len(sequence_digests),
            "order": "batch-major-right-padded",
            "padded_length": padded_length,
            "sequence_digests": list(sequence_digests),
            "selected_rows": len(sequence_digests) * padded_length,
        },
        "selected_expert_ids": selected,
    }


def _document(observations: list[dict]) -> dict:
    ordered = sorted(
        observations,
        key=lambda row: (row["request_id"], row["layer"], row["observation_id"]),
    )
    identity = {
        "checkpoint": route_eval.OFFICIAL_CHECKPOINT.as_record(),
        "observations": ordered,
        "schema": route_eval.ROUTE_OBSERVATIONS_SCHEMA,
    }
    return {**identity, "sha256": _digest(identity)}


def _fixture_document(*, prompt_count: int = 8) -> dict:
    request_id = "draft-verify-v1:" + "a" * 64
    execution_digest = _text_digest("off-bfloat16")
    sequence_digests = [
        _text_digest(f"sequence-{index}") for index in range(prompt_count)
    ]
    active_lengths = [4 for _ in sequence_digests]
    observations = [
        _row(
            request_id=request_id,
            execution_digest=execution_digest,
            layer=layer,
            sequence_digests=sequence_digests,
            active_lengths=active_lengths,
            padded_length=6,
        )
        for layer in (0, 1)
    ]
    return _document(observations)


def _write(path: Path, document: dict) -> None:
    path.write_bytes(_canonical(document))


def _contains_forbidden_label_key(value) -> bool:
    forbidden = {"question", "item_id", "gold", "answer"}
    if isinstance(value, dict):
        return any(
            key.lower() in forbidden or _contains_forbidden_label_key(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_label_key(child) for child in value)
    return False


class DeepSeekV4RouteEvalTests(unittest.TestCase):
    def test_cli_writes_runtime_real_and_placebo_model_artifacts(self) -> None:
        from immer.runtimes.deepseek_v4.route_model import load_route_model_artifact

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "routes.json"
            report = directory / "report.json"
            real = directory / "real-model.json"
            placebo = directory / "placebo-model.json"
            _write(source, _fixture_document())

            exit_code = route_eval.main(
                [
                    "--sidecar",
                    str(source),
                    "--output",
                    str(report),
                    "--n-experts",
                    "4",
                    "--real-model-output",
                    str(real),
                    "--placebo-model-output",
                    str(placebo),
                ]
            )

            self.assertEqual(exit_code, 0)
            real_model = load_route_model_artifact(
                real,
                expected_role="real_markov",
            )
            placebo_model = load_route_model_artifact(
                placebo,
                expected_role="placebo_markov",
            )
            written_report = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(
                real_model.predictor.snapshot_sha256,
                written_report["models"]["real_markov_snapshot_sha256"],
            )
            self.assertEqual(
                placebo_model.predictor.snapshot_sha256,
                written_report["models"]["placebo_markov_snapshot_sha256"],
            )

    def test_reconstructs_batch_major_rows_and_real_beats_placebo_end_to_end(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "routes.json"
            output = Path(temporary) / "report.json"
            document = _fixture_document()
            _write(source, document)

            loaded = route_eval.load_route_sidecar(source, n_experts=4)
            routes = route_eval.reconstruct_prompt_routes(loaded)
            report = route_eval.build_report(
                source,
                n_experts=4,
                test_fraction=0.25,
                split_seed=17,
                placebo_seed=29,
            )
            route_eval.write_report(output, report)

            self.assertEqual(len(routes.prompts), 8)
            self.assertEqual(routes.layer_ids, (0, 1))
            self.assertTrue(
                all(
                    len(prompt.layers) == 2
                    and len(prompt.layers[0].rows) == 4
                    and len(prompt.layers[1].rows) == 4
                    for prompt in routes.prompts
                )
            )
            evaluations = report["evaluations"]
            self.assertEqual(
                set(evaluations),
                {"real_markov", "placebo_markov", "marginal", "passthrough"},
            )
            for evaluation in evaluations.values():
                self.assertEqual(
                    [point["k"] for point in evaluation["curve"]],
                    [1, 2, 3, 4],
                )
                self.assertEqual(evaluation["curve"][-1]["set_recall"], 1.0)
                self.assertEqual(evaluation["curve"][-1]["selection_mass_recall"], 1.0)
            self.assertGreater(
                evaluations["real_markov"]["curve"][0]["set_recall"],
                evaluations["placebo_markov"]["curve"][0]["set_recall"],
            )
            self.assertEqual(
                report["headline"]["models"]["real_markov"]["Recall@1"], 1.0
            )
            windows = report["micro_window_sweep"]
            self.assertEqual(
                [row["window_rows"] for row in windows],
                [1, 2, 4, 8, 16, 32, 64],
            )
            self.assertGreater(windows[0]["real_recall"], windows[0]["placebo_recall"])
            fixed_k = report["micro_window_k_sweep"]
            self.assertEqual(len(fixed_k), 8)
            self.assertEqual(
                sorted({row["window_rows"] for row in fixed_k}),
                [1, 2, 4, 8],
            )
            self.assertEqual(
                sorted({row["k"] for row in fixed_k}),
                [3, 4],
            )
            self.assertTrue(
                all(
                    row["real_expert_read_amplification"] >= 1.0
                    and row["placebo_expert_read_amplification"] >= 1.0
                    for row in fixed_k
                )
            )
            aggregate = report["aggregate_vote_sweep"]
            self.assertEqual([row["k"] for row in aggregate], [3, 4])
            self.assertTrue(
                all(
                    row["real_expert_read_amplification"] >= 1.0
                    and row["vote_window_rows"] == 2
                    for row in aggregate
                )
            )
            confidence = report["confidence_gate_sweep"]
            self.assertEqual(len(confidence), 22)
            self.assertEqual(sorted({row["k"] for row in confidence}), [3, 4])
            self.assertTrue(
                all(
                    0 <= row["real_window_fraction"] <= 1
                    and row["real_expert_read_amplification"] >= 1
                    for row in confidence
                )
            )
            self.assertFalse(_contains_forbidden_label_key(report))
            self.assertEqual(output.read_bytes(), _canonical(report))
            identity = {key: value for key, value in report.items() if key != "sha256"}
            self.assertEqual(report["sha256"], _digest(identity))

    def test_tamper_noncanonical_and_label_fields_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "routes.json"
            document = _fixture_document()

            tampered = copy.deepcopy(document)
            tampered["observations"][0]["selected_expert_ids"][0] = [3]
            _write(source, tampered)
            with self.assertRaisesRegex(route_eval.RouteEvalError, "SHA-256"):
                route_eval.load_route_sidecar(source, n_experts=4)

            source.write_text(json.dumps(document, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(route_eval.RouteEvalError, "canonical"):
                route_eval.load_route_sidecar(source, n_experts=4)

            leaked = copy.deepcopy(document)
            leaked["observations"][0]["question"] = "not allowed"
            leaked = _document(leaked["observations"])
            _write(source, leaked)
            with self.assertRaisesRegex(route_eval.RouteEvalError, "unknown"):
                route_eval.load_route_sidecar(source, n_experts=4)

            repacked_checkpoint = copy.deepcopy(document)
            wrong_fingerprint = "f" * 64
            repacked_checkpoint["checkpoint"][
                "inventory_fingerprint"
            ] = wrong_fingerprint
            for row in repacked_checkpoint["observations"]:
                row["checkpoint"]["inventory_fingerprint"] = wrong_fingerprint
            identity = {
                key: value
                for key, value in repacked_checkpoint.items()
                if key != "sha256"
            }
            repacked_checkpoint["sha256"] = _digest(identity)
            _write(source, repacked_checkpoint)
            loaded = route_eval.load_route_sidecar(source, n_experts=4)
            self.assertEqual(
                loaded.checkpoint.inventory_fingerprint,
                wrong_fingerprint,
            )
            repacked_report = route_eval.build_report(source, n_experts=4)
            self.assertEqual(
                repacked_report["checkpoint"]["inventory_fingerprint"],
                wrong_fingerprint,
            )
            with self.assertRaisesRegex(
                route_eval.RouteEvalError, "another checkpoint"
            ):
                route_eval.load_route_sidecar(
                    source,
                    n_experts=4,
                    expected_checkpoint=route_eval.OFFICIAL_CHECKPOINT,
                )

            for field, value in (
                ("repo_id", "someone/other-checkpoint"),
                ("revision", "b" * 40),
            ):
                with self.subTest(checkpoint_field=field):
                    wrong_logical = copy.deepcopy(document)
                    wrong_logical["checkpoint"][field] = value
                    for row in wrong_logical["observations"]:
                        row["checkpoint"][field] = value
                    logical_identity = {
                        key: child
                        for key, child in wrong_logical.items()
                        if key != "sha256"
                    }
                    wrong_logical["sha256"] = _digest(logical_identity)
                    _write(source, wrong_logical)
                    with self.assertRaisesRegex(
                        route_eval.RouteEvalError, "logical checkpoint"
                    ):
                        route_eval.load_route_sidecar(source, n_experts=4)

    def test_duplicate_expert_ids_are_preserved_exactly_like_the_producer(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "routes.json"
            document = _fixture_document()
            duplicated = [0, 0, 1]
            document["observations"][0]["selected_expert_ids"][0] = duplicated
            document = _document(document["observations"])
            _write(source, document)

            self.assertEqual(
                _freeze_selected_experts(
                    [duplicated],
                    n_routed_experts=4,
                ),
                ((0, 0, 1),),
            )
            loaded = route_eval.load_route_sidecar(source, n_experts=4)
            routes = route_eval.reconstruct_prompt_routes(loaded)
            duplicated_sequence = document["observations"][0]["row_layout"][
                "sequence_digests"
            ][0]
            duplicated_prompt = next(
                prompt
                for prompt in routes.prompts
                if prompt.observation_id == duplicated_sequence
            )
            self.assertEqual(duplicated_prompt.layers[0].rows[0], (0, 0, 1))
            report = route_eval.build_report(source, n_experts=4)
            self.assertEqual(
                len(report["evaluations"]["real_markov"]["curve"]),
                4,
            )

    def test_conflicting_and_missing_reconstruction_layout_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "routes.json"
            base = _fixture_document(prompt_count=4)
            sequence = base["observations"][0]["row_layout"]["sequence_digests"][0]
            execution = base["observations"][0]["execution_feature_digest"]

            conflict_request = "draft-verify-v1:" + "b" * 64
            conflicting = [
                _row(
                    request_id=conflict_request,
                    execution_digest=execution,
                    layer=layer,
                    sequence_digests=[sequence],
                    active_lengths=[4],
                    padded_length=6,
                    offset=1,
                )
                for layer in (0, 1)
            ]
            _write(source, _document(base["observations"] + conflicting))
            loaded = route_eval.load_route_sidecar(source, n_experts=4)
            with self.assertRaisesRegex(route_eval.RouteEvalError, "conflicting route"):
                route_eval.reconstruct_prompt_routes(loaded)

            missing_request = "draft-verify-v1:" + "c" * 64
            missing_sequence = _text_digest("missing-sequence")
            incomplete = _row(
                request_id=missing_request,
                execution_digest=execution,
                layer=0,
                sequence_digests=[missing_sequence],
                active_lengths=[4],
                padded_length=6,
            )
            _write(source, _document(base["observations"] + [incomplete]))
            loaded = route_eval.load_route_sidecar(source, n_experts=4)
            with self.assertRaisesRegex(route_eval.RouteEvalError, "two route layers"):
                route_eval.reconstruct_prompt_routes(loaded)

            bad_layout = copy.deepcopy(base)
            bad_layout["observations"][1]["row_layout"]["active_lengths"][0] = 3
            bad_layout = _document(bad_layout["observations"])
            _write(source, bad_layout)
            loaded = route_eval.load_route_sidecar(source, n_experts=4)
            with self.assertRaisesRegex(route_eval.RouteEvalError, "conflicting row"):
                route_eval.reconstruct_prompt_routes(loaded)

    def test_mixed_execution_regimes_require_explicit_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "routes.json"
            base = _fixture_document(prompt_count=4)
            second_execution = _text_digest("stable-crsa-bfloat16")
            request_id = "draft-verify-v1:" + "d" * 64
            sequences = [_text_digest(f"other-{index}") for index in range(4)]
            other = [
                _row(
                    request_id=request_id,
                    execution_digest=second_execution,
                    layer=layer,
                    sequence_digests=sequences,
                    active_lengths=[4] * 4,
                    padded_length=6,
                )
                for layer in (0, 1)
            ]
            _write(source, _document(base["observations"] + other))
            loaded = route_eval.load_route_sidecar(source, n_experts=4)

            with self.assertRaisesRegex(route_eval.RouteEvalError, "mixes execution"):
                route_eval.reconstruct_prompt_routes(loaded)
            selected = route_eval.reconstruct_prompt_routes(
                loaded,
                execution_digest=second_execution,
            )
            self.assertEqual(selected.execution_feature_digest, second_execution)
            self.assertEqual(len(selected.prompts), 4)

    def test_full_and_resumed_suffix_sequences_share_one_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "routes.json"
            execution = _text_digest("off-bfloat16")
            full_request = "draft-verify-v1:" + "e" * 64
            suffix_request = "draft-verify-v1:" + "f" * 64
            full_sequences = [_text_digest(f"full-{index}") for index in range(4)]
            suffix_sequences = [_text_digest(f"suffix-{index}") for index in range(4)]
            observations = [
                _row(
                    request_id=full_request,
                    execution_digest=execution,
                    layer=layer,
                    sequence_digests=full_sequences,
                    active_lengths=[4] * 4,
                    padded_length=6,
                )
                for layer in (0, 1, 2, 3)
            ] + [
                _row(
                    request_id=suffix_request,
                    execution_digest=execution,
                    layer=layer,
                    sequence_digests=suffix_sequences,
                    active_lengths=[4] * 4,
                    padded_length=6,
                )
                for layer in (2, 3)
            ]
            _write(source, _document(observations))

            loaded = route_eval.load_route_sidecar(source, n_experts=4)
            routes = route_eval.reconstruct_prompt_routes(loaded)
            coverage = {
                tuple(layer.layer for layer in prompt.layers)
                for prompt in routes.prompts
            }
            report = route_eval.build_report(source, n_experts=4)

            self.assertEqual(routes.layer_ids, (0, 1, 2, 3))
            self.assertEqual(coverage, {(0, 1, 2, 3), (2, 3)})
            self.assertEqual(report["sidecar"]["layer_ids"], [0, 1, 2, 3])
            self.assertTrue(
                all(
                    evaluation["evaluated_rows"] > 0
                    for evaluation in report["evaluations"].values()
                )
            )


if __name__ == "__main__":
    unittest.main()
