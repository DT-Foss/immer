from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import torch

from immer.runtimes.qwen3_8 import DeltaNetState

from test_qwen38_causal_bundle import REPO_ID, REVISION, _fixture, bundle_script


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen38_speculative_k2_benchmark.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "qwen38_speculative_k2_benchmark", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import Qwen speculative K2 benchmark")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = _load_script()


class Qwen38SpeculativeK2BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(
            prefix=".qwen-speculative-k2-test-", dir=Path.cwd()
        )
        cls.root = Path(cls.temporary.name)
        cls.source_weights, cls.inventory, fingerprint = _fixture(cls.root)
        cls.original_eos = (
            benchmark.direct.IM_END_TOKEN_ID,
            benchmark.direct.END_OF_TEXT_TOKEN_ID,
        )
        benchmark.direct.IM_END_TOKEN_ID = 30
        benchmark.direct.END_OF_TEXT_TOKEN_ID = 31
        cls.causal_bundle = cls.root / "model.causal"
        bundle_script.build_bundle(
            cls.source_weights,
            cls.inventory,
            cls.causal_bundle,
            repo_id=REPO_ID,
            revision=REVISION,
            expected_fingerprint=fingerprint,
            require_official=False,
            require_remote_hashes=True,
        )
        cls.selected_path, cls.selected = cls._select("accept", (2, 3))
        cls.positive_args = cls._args("accept", cls.selected_path)
        cls.positive = benchmark.run(
            cls.positive_args,
            runtime_factory=cls._runtime_factory((2, 3)),
        )
        benchmark.direct._write_json(cls.positive_args.output, cls.positive)

    @classmethod
    def tearDownClass(cls) -> None:
        (
            benchmark.direct.IM_END_TOKEN_ID,
            benchmark.direct.END_OF_TEXT_TOKEN_ID,
        ) = cls.original_eos
        cls.temporary.cleanup()

    @classmethod
    def _select(cls, suffix: str, draft: tuple[int, int]):
        source = cls.root / f"source-{suffix}.json"
        document = {
            "items": [
                {
                    "answer": "fixture-label-must-not-survive",
                    "candidate_correct": False,
                    "draft_token_ids": list(draft),
                    "gold": "fixture-gold-must-not-survive",
                    "item_id": "dev-0",
                    "prompt_token_ids": [1, 4],
                    "question": "fixture-question-must-not-survive",
                }
            ],
            "protocol": {
                "accepted_eos_token_ids": [30, 31],
                "batch_size": 1,
                "item_ids": ["dev-0"],
                "system_prompt": "fixture",
                "thinking": False,
            },
            "schema": benchmark.direct.FERTIG_INPUT_SCHEMA,
            "source": {
                "checkpoint": REPO_ID,
                "drafts": "fixture-q3-provider",
                "revision": REVISION,
            },
        }
        benchmark.direct._write_json(source, document)
        selected_path = cls.root / f"selected-{suffix}.json"
        args = benchmark._parser().parse_args(
            [
                "select-input",
                "--inputs",
                str(source),
                "--inputs-sha256",
                benchmark.direct._sha256_file(source),
                "--item-id",
                "dev-0",
                "--index",
                "0",
                "--output",
                str(selected_path),
            ]
        )
        args._require_official = False
        selected = benchmark.select_input(args)
        benchmark.direct._write_json(selected_path, selected)
        return selected_path, selected

    @classmethod
    def _args(cls, suffix: str, selected_path: Path):
        args = benchmark._parser().parse_args(
            [
                "run",
                "--input",
                str(selected_path),
                "--causal-bundle",
                str(cls.causal_bundle),
                "--output",
                str(cls.root / f"result-{suffix}.json"),
                "--access-trace-greedy",
                str(cls.root / f"trace-{suffix}-greedy.causaltrace"),
                "--access-trace-speculative",
                str(cls.root / f"trace-{suffix}-speculative.causaltrace"),
                "--source",
                str(cls.source_weights),
                "--pinned-inventory",
                str(cls.inventory),
                "--cache-dir",
                str(cls.root / f"cache-{suffix}"),
                "--logical-repo-id",
                REPO_ID,
                "--revision",
                REVISION,
                "--device",
                "cpu",
                "--dtype",
                "bfloat16",
                "--max-resident-mb",
                "2",
                "--source-budget-mb",
                "20",
                "--max-seq-len",
                "8",
                "--head-block-rows",
                "8",
            ]
        )
        args._require_official = False
        return args

    @classmethod
    def _runtime_factory(
        cls, targets: tuple[int, int], *, mutate_final_state: bool = False
    ):
        calls = 0

        def factory(args, recorder, **kwargs):
            nonlocal calls
            calls += 1
            arm = calls
            runtime, model = benchmark.direct._runtime(args, recorder, **kwargs)
            greedy_index = 0

            def fixed_topk(hidden, *, k, block_rows, progress=None):
                nonlocal greedy_index
                del k, block_rows, progress
                if hidden.ndim == 3:
                    chosen = torch.tensor(
                        [[list(targets)]], device=model.pager.device, dtype=torch.long
                    ).transpose(1, 2)
                else:
                    index = greedy_index if arm == 1 else 1
                    chosen = torch.tensor(
                        [[targets[index]]], device=model.pager.device, dtype=torch.long
                    )
                    greedy_index += 1
                values = torch.ones(
                    chosen.shape, device=model.pager.device, dtype=torch.float32
                )
                return values, chosen

            model.pager.topk_logits = fixed_topk
            if mutate_final_state and arm == 2:
                original_commit = model.commit_continuation_block

                def mutating_commit(stage):
                    hidden, evidence = original_commit(stage)
                    state = model._layer_states[0]
                    assert isinstance(state, DeltaNetState)
                    model._layer_states[0] = DeltaNetState(
                        conv=state.conv,
                        recurrent=state.recurrent + 1.0,
                    )
                    return hidden, evidence

                model.commit_continuation_block = mutating_commit
            return runtime, model

        return factory

    def test_full_acceptance_is_exact_and_saves_one_weight_pass(self) -> None:
        document = self.positive

        self.assertEqual(document["status"], "positive")
        self.assertEqual(document["executions"]["greedy"]["token_ids"], [2, 3])
        self.assertEqual(document["executions"]["speculative"]["token_ids"], [2, 3])
        self.assertEqual(
            document["executions"]["speculative"]["speculative_evidence"]["rounds"][0][
                "replay_kind"
            ],
            "commit-k2",
        )
        self.assertGreater(
            document["comparison"]["costs"]["source_body_bytes_saved"], 0
        )
        self.assertEqual(document["comparison"]["costs"]["head_scans_saved"], 1)
        self.assertTrue(document["comparison"]["committed_hidden_equal"])
        self.assertTrue(document["comparison"]["state_equal"])
        self.assertNotIn(
            '"gold"', benchmark.direct._canonical(document).decode("utf-8")
        )

    def test_mismatch_zero_and_one_remain_bit_exact(self) -> None:
        for suffix, draft, replay in (
            ("mismatch0", (7, 3), "mismatch0-decode"),
            ("mismatch1", (2, 7), "mismatch1-restage"),
        ):
            with self.subTest(replay=replay):
                selected_path, _selected = self._select(suffix, draft)
                args = self._args(suffix, selected_path)
                document = benchmark.run(
                    args,
                    runtime_factory=self._runtime_factory((2, 3)),
                )
                self.assertEqual(document["status"], "positive")
                self.assertEqual(document["executions"]["greedy"]["token_ids"], [2, 3])
                self.assertEqual(
                    document["executions"]["speculative"]["token_ids"], [2, 3]
                )
                self.assertEqual(
                    document["executions"]["speculative"]["speculative_evidence"][
                        "rounds"
                    ][0]["replay_kind"],
                    replay,
                )
                self.assertTrue(document["comparison"]["state_equal"])
                self.assertTrue(document["comparison"]["initial_stage_hidden_equal"])
                if replay == "mismatch0-decode":
                    tampered = copy.deepcopy(document)
                    speculative = tampered["executions"]["speculative"]
                    speculative["hidden"]["staged"][0]["sha256"] = "0" * 64
                    speculative["hidden"] = benchmark._seal(
                        {
                            key: value
                            for key, value in speculative["hidden"].items()
                            if key != "sha256"
                        }
                    )
                    tampered["executions"]["speculative"] = benchmark._seal(
                        {
                            key: value
                            for key, value in speculative.items()
                            if key != "sha256"
                        }
                    )
                    tampered["comparison"] = benchmark._comparison(
                        tampered["executions"]["greedy"],
                        tampered["executions"]["speculative"],
                    )
                    tampered = benchmark._seal(
                        {
                            key: value
                            for key, value in tampered.items()
                            if key != "sha256"
                        }
                    )
                    with self.assertRaisesRegex(
                        benchmark.SpeculativeBenchmarkError, "status is inconsistent"
                    ):
                        benchmark._validate_result(
                            tampered,
                            trace_paths={
                                "greedy": args.access_trace_greedy,
                                "speculative": args.access_trace_speculative,
                            },
                        )

    def test_real_final_state_difference_is_sealed_as_mismatch(self) -> None:
        args = self._args("state-mismatch", self.selected_path)
        document = benchmark.run(
            args,
            runtime_factory=self._runtime_factory((2, 3), mutate_final_state=True),
        )

        self.assertEqual(document["status"], "mismatch")
        self.assertFalse(document["comparison"]["positive"])
        self.assertFalse(document["comparison"]["state_equal"])
        self.assertEqual(
            document["comparison"]["state_differences"][0]["name"],
            "layer.000.deltanet.recurrent",
        )

    def test_input_provider_state_and_transported_trace_tamper_fail_closed(
        self,
    ) -> None:
        trace_paths = {
            "greedy": self.positive_args.access_trace_greedy,
            "speculative": self.positive_args.access_trace_speculative,
        }
        provider = copy.deepcopy(self.positive)
        provider["executions"]["speculative"]["provider"]["identity"] = "foreign"
        provider["executions"]["speculative"] = benchmark._seal(
            {
                key: value
                for key, value in provider["executions"]["speculative"].items()
                if key != "sha256"
            }
        )
        provider = benchmark._seal(
            {key: value for key, value in provider.items() if key != "sha256"}
        )
        with self.assertRaisesRegex(
            benchmark.SpeculativeBenchmarkError, "input/provider identity"
        ):
            benchmark._validate_result(provider, trace_paths=trace_paths)

        state = copy.deepcopy(self.positive)
        state["executions"]["speculative"]["state"]["tensors"][0]["sha256"] = "0" * 64
        state["executions"]["speculative"] = benchmark._seal(
            {
                key: value
                for key, value in state["executions"]["speculative"].items()
                if key != "sha256"
            }
        )
        state = benchmark._seal(
            {key: value for key, value in state.items() if key != "sha256"}
        )
        with self.assertRaisesRegex(
            benchmark.SpeculativeBenchmarkError, "state manifest document seal mismatch"
        ):
            benchmark._validate_result(state, trace_paths=trace_paths)

        bundle = copy.deepcopy(self.positive)
        for mode in ("greedy", "speculative"):
            bundle["executions"][mode]["bundle"] = {}
            bundle["executions"][mode] = benchmark._seal(
                {
                    key: value
                    for key, value in bundle["executions"][mode].items()
                    if key != "sha256"
                }
            )
        bundle["comparison"] = benchmark._comparison(
            bundle["executions"]["greedy"], bundle["executions"]["speculative"]
        )
        bundle = benchmark._seal(
            {key: value for key, value in bundle.items() if key != "sha256"}
        )
        with self.assertRaisesRegex(
            benchmark.SpeculativeBenchmarkError, "causal-bundle receipt"
        ):
            benchmark._validate_result(bundle, trace_paths=trace_paths)

        traffic = copy.deepcopy(self.positive)
        traffic["executions"]["greedy"]["traffic"]["source_body_bytes"] += 1
        traffic["executions"]["greedy"] = benchmark._seal(
            {
                key: value
                for key, value in traffic["executions"]["greedy"].items()
                if key != "sha256"
            }
        )
        traffic["comparison"] = benchmark._comparison(
            traffic["executions"]["greedy"],
            traffic["executions"]["speculative"],
        )
        traffic = benchmark._seal(
            {key: value for key, value in traffic.items() if key != "sha256"}
        )
        with self.assertRaisesRegex(
            benchmark.SpeculativeBenchmarkError, "adjusted execution traffic"
        ):
            benchmark._validate_result(traffic, trace_paths=trace_paths)

        staged = copy.deepcopy(self.positive)
        speculative = staged["executions"]["speculative"]
        speculative["hidden"]["staged"][-1]["sha256"] = "0" * 64
        speculative["hidden"] = benchmark._seal(
            {
                key: value
                for key, value in speculative["hidden"].items()
                if key != "sha256"
            }
        )
        staged["executions"]["speculative"] = benchmark._seal(
            {key: value for key, value in speculative.items() if key != "sha256"}
        )
        staged["comparison"] = benchmark._comparison(
            staged["executions"]["greedy"],
            staged["executions"]["speculative"],
        )
        staged = benchmark._seal(
            {key: value for key, value in staged.items() if key != "sha256"}
        )
        with self.assertRaisesRegex(
            benchmark.SpeculativeBenchmarkError, "accepted K2 stage"
        ):
            benchmark._validate_result(staged, trace_paths=trace_paths)

        invalid_source = self.root / "source-invalid-hash.json"
        shutil.copyfile(self.root / "source-accept.json", invalid_source)
        select_args = benchmark._parser().parse_args(
            [
                "select-input",
                "--inputs",
                str(invalid_source),
                "--inputs-sha256",
                "0" * 64,
                "--item-id",
                "dev-0",
                "--index",
                "0",
                "--output",
                str(self.root / "never.json"),
            ]
        )
        select_args._require_official = False
        with self.assertRaises(benchmark.direct.QwenDirectDecodeError):
            benchmark.select_input(select_args)

        detached = self.root / "detached"
        detached.mkdir()
        result_copy = detached / "result.json"
        shutil.copyfile(self.positive_args.output, result_copy)
        originals = []
        try:
            for path in trace_paths.values():
                source = Path(path)
                shutil.copyfile(source, detached / source.name)
                backup = source.with_suffix(source.suffix + ".held")
                source.rename(backup)
                originals.append((source, backup))
            self.assertEqual(benchmark._load_result(result_copy), self.positive)
        finally:
            for source, backup in originals:
                backup.rename(source)


if __name__ == "__main__":
    unittest.main()
