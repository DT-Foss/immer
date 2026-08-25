from __future__ import annotations

import copy
from contextlib import redirect_stderr
import importlib.util
import io
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import torch

from immer.runtimes.qwen3_8 import DeltaNetState

from test_qwen38_causal_bundle import REPO_ID, REVISION, _fixture, bundle_script
from test_qwen38_direct_decode_benchmark import (
    _answer_branch_tokenizer,
    _branch_generation_args,
    _prepared_branch_source,
    _select_answer_branch,
    benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen38_continuation_block_parity.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "qwen38_continuation_block_parity", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import Qwen continuation parity script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


parity = _load_script()


class Qwen38ContinuationBlockParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(
            prefix=".qwen-continuation-parity-test-", dir=Path.cwd()
        )
        cls.root = Path(cls.temporary.name)
        cls.source_weights, cls.inventory, fingerprint = _fixture(cls.root)
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
        cls.tokenizer = cls.root / "tokenizer.json"
        cls.eos = _answer_branch_tokenizer(cls.tokenizer)
        source = cls.root / "sealed-source.json"
        _prepared_branch_source(
            source,
            eos=cls.eos,
            gold=("1",),
            system_prompt=benchmark.ANSWER_OUTPUT_INSTRUCTION,
            thinking=False,
        )
        cls.branch_input = cls.root / "answer-input.json"
        selected = _select_answer_branch(source, cls.tokenizer, cls.branch_input)

        cls.reference_trace = cls.root / "reference-access.json"
        cls.reference_path = cls.root / "reference-result-off.json"
        generation_args = _branch_generation_args(
            branch_input=cls.branch_input,
            tokenizer=cls.tokenizer,
            output=cls.reference_path,
            trace=cls.reference_trace,
            source=cls.source_weights,
            inventory=cls.inventory,
            cache=cls.root / "reference-cache",
            mode="off",
        )
        generation_args.max_new_tokens = 2
        generation_args.causal_bundle = str(cls.causal_bundle)
        generation_args.max_seq_len = (
            len(selected["items"][0]["effective_prompt_token_ids"]) + 2
        )
        generated = iter((1, cls.eos[0]))

        def reference_runtime_factory(*factory_args, **factory_kwargs):
            runtime, model = benchmark._runtime(*factory_args, **factory_kwargs)

            def fixed_topk(_hidden, *, k, block_rows, progress=None):
                del k, block_rows
                token = next(generated)
                if progress is not None:
                    progress(
                        {
                            "rows_done": model.config.vocab_size,
                            "vocab_rows": model.config.vocab_size,
                        }
                    )
                return (
                    torch.tensor([[1.0]], device=model.pager.device),
                    torch.tensor([[token]], device=model.pager.device),
                )

            model.pager.topk_logits = fixed_topk
            return runtime, model

        with redirect_stderr(io.StringIO()):
            reference = benchmark.generate_arm(
                generation_args, runtime_factory=reference_runtime_factory
            )
        benchmark._write_json(cls.reference_path, reference)
        cls.reference = benchmark._load_branch_result(cls.reference_path)
        cls.alternate_bundle = cls.root / "alternate.causal"
        shutil.copytree(cls.causal_bundle, cls.alternate_bundle)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _args(self, suffix: str):
        args = parity._parser().parse_args(
            [
                "--input",
                str(self.branch_input),
                "--reference-off",
                str(self.reference_path),
                "--causal-bundle",
                str(self.causal_bundle),
                "--output",
                str(self.root / f"result-{suffix}.json"),
                "--access-trace-tokenwise",
                str(self.root / f"access-{suffix}-tokenwise.json"),
                "--access-trace-block",
                str(self.root / f"access-{suffix}-block.json"),
                "--source",
                str(self.source_weights),
                "--pinned-inventory",
                str(self.inventory),
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
                "12",
            ]
        )
        args._require_official = False
        return args

    def _runtime_factory(self, *, mutate_block: bool = False):
        calls = 0

        def factory(args, recorder, **kwargs):
            nonlocal calls
            calls += 1
            runtime, model = benchmark._runtime(args, recorder, **kwargs)
            if mutate_block and calls == 2:
                original_stage = model.stage_continuation_block

                def mismatching_stage(token_ids, **stage_kwargs):
                    stage = original_stage(token_ids, **stage_kwargs)
                    state = model._layer_states[0]
                    assert isinstance(state, DeltaNetState)
                    model._layer_states[0] = DeltaNetState(
                        conv=state.conv,
                        recurrent=state.recurrent + 1.0,
                    )
                    return stage

                model.stage_continuation_block = mismatching_stage
            return runtime, model

        return factory

    def test_positive_parity_hashes_every_state_and_separates_costs(self) -> None:
        args = self._args("positive")
        document = parity.run(args, runtime_factory=self._runtime_factory())

        self.assertEqual(document["status"], "positive")
        self.assertTrue(document["comparison"]["positive"])
        self.assertTrue(document["contract"]["gold_free"])
        self.assertNotIn('"gold"', benchmark._canonical(document).decode("utf-8"))
        tokenwise = document["executions"]["tokenwise"]
        block = document["executions"]["block"]
        self.assertEqual(tokenwise["prefill_mode"], "batched")
        self.assertEqual(block["prefill_mode"], "batched")
        self.assertEqual(
            tokenwise["continuation"]["linear_calls"],
            2 * block["continuation"]["linear_calls"],
        )
        self.assertEqual(
            tokenwise["continuation"]["state"]["tensor_count"],
            8,
        )
        self.assertEqual(
            tokenwise["continuation"]["state"]["sha256"],
            block["continuation"]["state"]["sha256"],
        )
        self.assertTrue(block["stage_base_state_equal"])
        self.assertEqual(
            block["final_state_bytes"],
            block["continuation"]["state"]["total_bytes"],
        )
        self.assertNotEqual(tokenwise["trace"]["sha256"], block["trace"]["sha256"])
        self.assertEqual(parity._validate_result_document(document), document)

    def test_real_state_difference_returns_a_sealed_mismatch(self) -> None:
        args = self._args("mismatch")
        document = parity.run(
            args, runtime_factory=self._runtime_factory(mutate_block=True)
        )

        self.assertEqual(document["status"], "mismatch")
        self.assertFalse(document["comparison"]["positive"])
        self.assertFalse(document["comparison"]["block_stage_base_state_equal"])
        self.assertTrue(document["comparison"]["continuation_state_equal"])
        self.assertEqual(
            document["comparison"]["block_stage_base_state_differences"][0]["name"],
            "layer.000.deltanet.recurrent",
        )
        self.assertEqual(parity._validate_result_document(document), document)

    def test_input_reference_and_nested_receipt_tamper_fail_closed(self) -> None:
        wrong_bundle_args = self._args("wrong-bundle")

        def wrong_bundle_factory(args, recorder, **kwargs):
            substituted = copy.copy(args)
            substituted.causal_bundle = str(self.alternate_bundle)
            return benchmark._runtime(substituted, recorder, **kwargs)

        with self.assertRaisesRegex(
            parity.ContinuationParityError, "requested causal bundle root"
        ):
            parity.run(wrong_bundle_args, runtime_factory=wrong_bundle_factory)

        args = self._args("tamper-source")
        tampered_input = benchmark._strict_json(self.branch_input)
        tampered_input["items"][0]["effective_prompt_token_ids"][0] += 1
        tampered_input_path = self.root / "tampered-input.json"
        benchmark._write_json(tampered_input_path, tampered_input)
        args.input = str(tampered_input_path)
        with self.assertRaisesRegex(benchmark.QwenDirectDecodeError, "seal mismatch"):
            parity.run(args, runtime_factory=lambda *_args, **_kwargs: self.fail())

        args = self._args("tamper-reference")
        tampered_reference = benchmark._strict_json(self.reference_path)
        tampered_reference["items"][0]["generated_token_ids"][0] += 1
        tampered_reference_path = self.root / "tampered-reference.json"
        benchmark._write_json(tampered_reference_path, tampered_reference)
        args.reference_off = str(tampered_reference_path)
        with self.assertRaisesRegex(benchmark.QwenDirectDecodeError, "seal mismatch"):
            parity.run(args, runtime_factory=lambda *_args, **_kwargs: self.fail())

        args = self._args("nested-seals")
        document = parity.run(args, runtime_factory=self._runtime_factory())
        nested_state = copy.deepcopy(document)
        nested_state["executions"]["block"]["continuation"]["state"]["tensors"][0][
            "sha256"
        ] = "0" * 64
        nested_state["sha256"] = benchmark._sha256(
            {key: value for key, value in nested_state.items() if key != "sha256"}
        )
        with self.assertRaisesRegex(parity.ContinuationParityError, "seal mismatch"):
            parity._validate_result_document(nested_state)

        nested_receipt = copy.deepcopy(document)
        nested_receipt["executions"]["tokenwise"]["continuation"]["linear_calls"] += 1
        nested_receipt["sha256"] = benchmark._sha256(
            {key: value for key, value in nested_receipt.items() if key != "sha256"}
        )
        with self.assertRaisesRegex(parity.ContinuationParityError, "seal mismatch"):
            parity._validate_result_document(nested_receipt)

        unbound_size = copy.deepcopy(document)
        unbound_size["executions"]["block"]["final_state_bytes"] += 1
        block_execution = unbound_size["executions"]["block"]
        block_execution["sha256"] = benchmark._sha256(
            {key: value for key, value in block_execution.items() if key != "sha256"}
        )
        unbound_size["sha256"] = benchmark._sha256(
            {key: value for key, value in unbound_size.items() if key != "sha256"}
        )
        with self.assertRaisesRegex(
            parity.ContinuationParityError, "differ from tensor manifest"
        ):
            parity._validate_result_document(unbound_size)


if __name__ == "__main__":
    unittest.main()
