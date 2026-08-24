from __future__ import annotations

import argparse
import copy
from contextlib import redirect_stderr
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from safetensors.torch import save_file
import torch

from immer.knowledge import AccessTrace, Streamer
from immer.runtimes.qwen3_8 import verify_probe_document

from test_qwen3_8_model import (
    _native_tiny_config,
    _tiny_config_mapping,
    _tiny_weights,
)

from test_qwen38_causal_bundle import (
    REPO_ID,
    REVISION,
    _fixture,
    bundle_script,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen38_direct_decode_benchmark.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "qwen38_direct_decode_benchmark", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import Qwen direct-decode benchmark")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = _load_script()


def _branch_tokenizer(path: Path) -> tuple[int, int]:
    from tokenizers import AddedToken, Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit

    vocab = {"[UNK]": 0, **{str(index): index for index in range(1, 29)}}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.add_special_tokens(
        [
            AddedToken("<|endoftext|>", special=True),
            AddedToken("<|im_start|>", special=True),
            AddedToken("<|im_end|>", special=True),
        ]
    )
    tokenizer.save(str(path))
    end_of_text = tokenizer.token_to_id("<|endoftext|>")
    im_end = tokenizer.token_to_id("<|im_end|>")
    assert end_of_text is not None and im_end is not None
    return int(im_end), int(end_of_text)


def _native_fixture(root: Path) -> tuple[Path, Path]:
    source_root = root / "native-source"
    source_root.mkdir()
    config = _native_tiny_config()
    mapping = _tiny_config_mapping()
    mapping["num_hidden_layers"] = 28
    mapping["layer_types"] = [
        "full_attention" if (layer + 1) % 4 == 0 else "linear_attention"
        for layer in range(28)
    ]
    mapping["num_attention_heads"] = 24
    mapping["num_key_value_heads"] = 4
    save_file(_tiny_weights(config), source_root / "model.safetensors")
    source_root.joinpath("config.json").write_text(
        json.dumps(mapping, separators=(",", ":")), encoding="utf-8"
    )
    source = Streamer.from_local(
        source_root,
        repo_id=REPO_ID,
        revision=REVISION,
        use_cache=False,
        budget_mb=20,
    )
    try:
        inventory = json.loads(json.dumps(source.inventory()))
    finally:
        source.close()
    digest = benchmark._sha256_file(source_root / "model.safetensors")
    inventory["shards"][0]["etag"] = f'"{digest}"'
    inventory["shards"][0]["cas_url_hash"] = digest
    inventory["shards"][0]["linked_etag"] = digest
    inventory["shards"][0]["payload_sha256"] = digest
    fingerprint = Streamer._source_fingerprint(inventory)
    document = {
        "inventory": inventory,
        "inventory_sha256": hashlib.sha256(
            bundle_script._canonical(inventory)
        ).hexdigest(),
        "repo_id": REPO_ID,
        "revision": REVISION,
        "schema": "immer.tensor-inventory-cache/v1",
        "source_fingerprint": fingerprint,
    }
    inventory_path = root / "native-inventory.json"
    inventory_path.write_bytes(bundle_script._canonical(document) + b"\n")
    return source_root, inventory_path


def _prepared_branch_source(
    path: Path,
    *,
    eos: tuple[int, int],
    gold: tuple[str, ...] = ("1", "2"),
) -> dict:
    item_ids = [f"dev-{index}" for index in range(len(gold))]
    document = {
        "items": [
            {
                "candidate_correct": False,
                "draft_token_ids": [9, 9],
                "gold": target,
                "item_id": item_id,
                "prompt_token_ids": [1, 4 + index],
            }
            for index, (item_id, target) in enumerate(zip(item_ids, gold, strict=True))
        ],
        "protocol": {
            "accepted_eos_token_ids": list(eos),
            "batch_size": len(item_ids),
            "item_ids": item_ids,
            "system_prompt": "fixture",
            "thinking": False,
        },
        "schema": benchmark.FERTIG_INPUT_SCHEMA,
        "source": {"checkpoint": REPO_ID, "revision": REVISION},
    }
    benchmark._write_json(path, document)
    return document


def _select_branch(
    source: Path,
    tokenizer: Path,
    output: Path,
    *,
    limit: int = 2,
    expected_sha256: str | None = None,
):
    source_sha256 = (
        benchmark._sha256_file(source) if expected_sha256 is None else expected_sha256
    )
    args = benchmark._parser().parse_args(
        [
            "select-branch-cohort",
            "--inputs",
            str(source),
            "--inputs-sha256",
            source_sha256,
            "--tokenizer-json",
            str(tokenizer),
            "--limit",
            str(limit),
            "--output",
            str(output),
        ]
    )
    args._require_official = False
    selected = benchmark.select_branch_cohort(args)
    benchmark._write_json(output, selected)
    return selected


def _branch_generation_args(
    *,
    branch_input: Path,
    tokenizer: Path,
    output: Path,
    trace: Path,
    source: Path,
    inventory: Path,
    cache: Path,
    mode: str,
):
    args = benchmark._parser().parse_args(
        [
            "generate-arm",
            "--input",
            str(branch_input),
            "--tokenizer-json",
            str(tokenizer),
            "--output",
            str(output),
            "--access-trace",
            str(trace),
            "--source",
            str(source),
            "--pinned-inventory",
            str(inventory),
            "--cache-dir",
            str(cache),
            "--logical-repo-id",
            REPO_ID,
            "--revision",
            REVISION,
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--max-seq-len",
            "12",
            "--max-new-tokens",
            "3",
            "--head-block-rows",
            "8",
            "--graft-layer",
            "2",
            "--graft-alpha",
            "0.1",
            "--graft-max-history",
            "12",
            "--max-resident-mb",
            "2",
            "--source-budget-mb",
            "20",
            "--mode",
            mode,
        ]
    )
    args._require_official = False
    return args


def _reseal(document: dict) -> dict:
    document.pop("sha256", None)
    document["sha256"] = benchmark._sha256(document)
    return document


def _rewrite_generated_answer(
    document: dict,
    index: int,
    *,
    tokens: list[int],
    text: str,
) -> None:
    row = document["items"][index]
    row["generated_token_ids"] = tokens
    row["generated_text"] = text
    row["parsed_numeric_answer"] = benchmark.extract_gsm8k_answer(text)
    row["eos_token_id"] = None
    row["finish_reason"] = "length"
    row["stopped_on_eos"] = False
    row["token_chain_sha256"] = benchmark._token_chain_sha256(
        row["prompt_token_ids"], tokens
    )
    _reseal(document)


def _runtime_args(
    command: str,
    *,
    direct_input: Path,
    snapshot: Path,
    output: Path,
    trace: Path,
    source: Path | None = None,
    inventory: Path | None = None,
    bundle: Path | None = None,
    prefix_result: Path | None = None,
    delta_probe: Path | None = None,
):
    argv = [
        command,
        "--input",
        str(direct_input),
        "--snapshot",
        str(snapshot),
        "--output",
        str(output),
        "--access-trace",
        str(trace),
        "--logical-repo-id",
        REPO_ID,
        "--revision",
        REVISION,
        "--device",
        "cpu",
        "--dtype",
        "bfloat16",
        "--max-seq-len",
        "16",
        "--max-resident-mb",
        "2",
        "--source-budget-mb",
        "20",
    ]
    if source is not None:
        argv.extend(("--source", str(source)))
    if inventory is not None:
        argv.extend(("--pinned-inventory", str(inventory)))
    if bundle is not None:
        argv.extend(("--causal-bundle", str(bundle)))
    if prefix_result is not None:
        argv.extend(("--prefix-result", str(prefix_result)))
    if delta_probe is not None:
        argv.extend(("--delta-probe", str(delta_probe)))
    args = benchmark._parser().parse_args(argv)
    args._require_official = False
    return args


class QwenDirectDecodeBenchmarkTests(unittest.TestCase):
    def test_shared_snapshot_restores_across_inventory_and_causal_paths(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".qwen-direct-decode-test-", dir=Path.cwd()
        ) as temporary:
            root = Path(temporary)
            source, inventory_source, fingerprint = _fixture(root)
            bundle = root / "model.causal"
            bundle_script.build_bundle(
                source,
                inventory_source,
                bundle,
                repo_id=REPO_ID,
                revision=REVISION,
                expected_fingerprint=fingerprint,
                require_official=False,
            )
            pinned = bundle / "weights" / "inventory.pinned.json"
            direct_input = root / "input.json"
            input_document = benchmark._result(
                {
                    "decode_token_id": 7,
                    "item_id": "fixture",
                    "prefix_token_ids": [1, 4, 9],
                    "schema": benchmark.INPUT_SCHEMA,
                }
            )
            benchmark._write_json(direct_input, input_document)
            snapshot = root / "prefix.json"

            prepare_args = _runtime_args(
                "prepare-prefix",
                direct_input=direct_input,
                snapshot=snapshot,
                output=root / "prepare.json",
                trace=root / "trace-prefix.json",
                source=bundle / "weights",
                inventory=pinned,
                delta_probe=root / "probe-prefix.json",
            )
            with redirect_stderr(io.StringIO()):
                prefix = benchmark.prepare_prefix(prepare_args)
            benchmark._write_json(prepare_args.output, prefix)
            self.assertEqual(prefix["schema"], benchmark.PREFIX_SCHEMA)
            self.assertEqual(prefix["snapshot"]["next_position"], 3)
            prefix_probe = verify_probe_document(
                json.loads(Path(prepare_args.delta_probe).read_text(encoding="utf-8"))
            )
            self.assertEqual(prefix["delta_probe"]["records"], 3)
            self.assertEqual(prefix_probe["body"]["context_mode"], "prefill")
            self.assertEqual(
                prefix_probe["body"]["hidden_sha256"], prefix["hidden_sha256"]
            )

            wrong_input = root / "wrong-input.json"
            benchmark._write_json(
                wrong_input,
                benchmark._result(
                    {
                        "decode_token_id": 7,
                        "item_id": "wrong",
                        "prefix_token_ids": [2, 5, 6],
                        "schema": benchmark.INPUT_SCHEMA,
                    }
                ),
            )
            wrong_args = _runtime_args(
                "decode",
                direct_input=wrong_input,
                snapshot=snapshot,
                output=root / "wrong.json",
                trace=root / "wrong-trace.json",
                source=bundle / "weights",
                inventory=pinned,
                prefix_result=Path(prepare_args.output),
            )
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "another input"
            ):
                benchmark.decode_arm(wrong_args)

            baseline_args = _runtime_args(
                "decode",
                direct_input=direct_input,
                snapshot=snapshot,
                output=root / "decode-inventory.json",
                trace=root / "trace-inventory.json",
                source=bundle / "weights",
                inventory=pinned,
                prefix_result=Path(prepare_args.output),
                delta_probe=root / "probe-decode-inventory.json",
            )
            with redirect_stderr(io.StringIO()):
                baseline = benchmark.decode_arm(baseline_args)
            benchmark._write_json(baseline_args.output, baseline)

            causal_args = _runtime_args(
                "decode",
                direct_input=direct_input,
                snapshot=snapshot,
                output=root / "decode-causal.json",
                trace=root / "trace-causal.json",
                bundle=bundle,
                prefix_result=Path(prepare_args.output),
                delta_probe=root / "probe-decode-causal.json",
            )
            with redirect_stderr(io.StringIO()):
                causal = benchmark.decode_arm(causal_args)
            benchmark._write_json(causal_args.output, causal)

            bundle_script.adopt_bundle(
                source,
                inventory_source,
                repo_id=REPO_ID,
                revision=REVISION,
                expected_fingerprint=fingerprint,
                require_official=False,
                weights_layout="flat",
            )
            flat_args = _runtime_args(
                "decode",
                direct_input=direct_input,
                snapshot=snapshot,
                output=root / "decode-causal-flat.json",
                trace=root / "trace-causal-flat.json",
                bundle=source,
                prefix_result=Path(prepare_args.output),
                delta_probe=root / "probe-decode-causal-flat.json",
            )
            with redirect_stderr(io.StringIO()):
                flat = benchmark.decode_arm(flat_args)
            benchmark._write_json(flat_args.output, flat)

            self.assertEqual(causal["hidden_sha256"], baseline["hidden_sha256"])
            self.assertEqual(flat["hidden_sha256"], baseline["hidden_sha256"])
            self.assertEqual(
                causal["snapshot"]["payload_sha256"],
                baseline["snapshot"]["payload_sha256"],
            )
            self.assertTrue(causal["pager"]["causal_tensor_reader_attached"])
            self.assertFalse(baseline["pager"]["causal_tensor_reader_attached"])
            self.assertEqual(
                causal["source_verification"]["kind"],
                "complete-causal-bundle/v1",
            )
            self.assertEqual(flat["source_verification"]["weights_layout"], "flat/v1")
            decode_probe = verify_probe_document(
                json.loads(Path(causal_args.delta_probe).read_text(encoding="utf-8"))
            )
            self.assertEqual(causal["delta_probe"]["records"], 3)
            self.assertEqual(decode_probe["body"]["context_mode"], "decode")
            self.assertEqual(decode_probe["body"]["start_pos"], 3)
            self.assertEqual(
                causal["delta_probe"]["sha256"], baseline["delta_probe"]["sha256"]
            )
            self.assertEqual(
                flat["delta_probe"]["sha256"], baseline["delta_probe"]["sha256"]
            )
            self.assertEqual(
                baseline["source_verification"]["kind"],
                "complete-local-shards/v1",
            )
            for path in (
                prepare_args.access_trace,
                baseline_args.access_trace,
                causal_args.access_trace,
                flat_args.access_trace,
            ):
                AccessTrace.from_bytes(Path(path).read_bytes()).verify()

            uninstrumented_path = root / "decode-causal-uninstrumented.json"
            uninstrumented = benchmark._result(
                {
                    key: value
                    for key, value in causal.items()
                    if key not in {"delta_probe", "sha256"}
                }
            )
            benchmark._write_json(uninstrumented_path, uninstrumented)
            unfair_args = benchmark._parser().parse_args(
                [
                    "compare",
                    "--remote",
                    str(baseline_args.output),
                    "--local",
                    str(uninstrumented_path),
                    "--output",
                    str(root / "unfair-comparison.json"),
                ]
            )
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "instrumentation differs"
            ):
                benchmark.compare(unfair_args)

            compare_args = benchmark._parser().parse_args(
                [
                    "compare",
                    "--remote",
                    str(baseline_args.output),
                    "--local",
                    str(causal_args.output),
                    "--output",
                    str(root / "comparison.json"),
                ]
            )
            comparison = benchmark.compare(compare_args)
            self.assertEqual(comparison["schema"], benchmark.COMPARISON_SCHEMA)
            self.assertEqual(comparison["hidden_sha256"], baseline["hidden_sha256"])
            self.assertEqual(
                comparison["delta_probe_sha256"], baseline["delta_probe"]["sha256"]
            )
            self.assertGreater(comparison["speedup_remote_over_local"], 0.0)

    def test_local_source_rejects_payload_tamper_despite_matching_layout(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".qwen-direct-tamper-test-", dir=Path.cwd()
        ) as temporary:
            root = Path(temporary)
            source, inventory, _fingerprint = _fixture(root)
            shard = source / "model.safetensors"
            with shard.open("r+b") as handle:
                handle.seek(shard.stat().st_size - 1)
                value = handle.read(1)
                handle.seek(-1, 1)
                handle.write(bytes([value[0] ^ 0xFF]))
            direct_input = root / "input.json"
            benchmark._write_json(
                direct_input,
                benchmark._result(
                    {
                        "decode_token_id": 7,
                        "item_id": "fixture",
                        "prefix_token_ids": [1, 4, 9],
                        "schema": benchmark.INPUT_SCHEMA,
                    }
                ),
            )
            args = _runtime_args(
                "prepare-prefix",
                direct_input=direct_input,
                snapshot=root / "prefix.json",
                output=root / "prepare.json",
                trace=root / "trace.json",
                source=source,
                inventory=inventory,
            )
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "shard SHA-256 mismatch"
            ):
                benchmark.prepare_prefix(args)

    def test_select_input_binds_item_and_first_draft_token(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "inputs.json"
            source.write_text(
                json.dumps(
                    {
                        "schema": benchmark.FERTIG_INPUT_SCHEMA,
                        "items": [
                            {
                                "item_id": "wanted",
                                "prompt_token_ids": [1, 2],
                                "draft_token_ids": [3, 4],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = benchmark._parser().parse_args(
                [
                    "select-input",
                    "--inputs",
                    str(source),
                    "--item-id",
                    "wanted",
                    "--output",
                    str(root / "selected.json"),
                ]
            )
            selected = benchmark.select_input(args)
            benchmark._write_json(args.output, selected)
            self.assertEqual(selected["prefix_token_ids"], [1, 2])
            self.assertEqual(selected["decode_token_id"], 3)
            self.assertEqual(benchmark._load_input(args.output), selected)

    def test_branch_selection_requires_raw_sha_and_remains_label_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer = root / "tokenizer.json"
            eos = _branch_tokenizer(tokenizer)
            source = root / "sealed-dev.json"
            document = _prepared_branch_source(source, eos=eos)
            self.assertTrue(
                {"sha256", "report_sha256", "document_sha256"}.isdisjoint(document)
            )

            first = _select_branch(source, tokenizer, root / "branch-a.json")
            encoded = json.dumps(first, sort_keys=True)
            self.assertNotIn('"gold"', encoded)
            self.assertNotIn('"candidate_correct"', encoded)
            self.assertNotIn('"correct"', encoded)
            self.assertNotIn('"correctness"', encoded)
            self.assertNotIn('"draft_token_ids"', encoded)
            self.assertNotIn('"answer"', encoded)
            self.assertEqual(first["protocol"]["teacher_forced_tokens_after_prompt"], 0)
            self.assertEqual(
                first["source"]["raw_file_sha256"], benchmark._sha256_file(source)
            )
            self.assertEqual(
                first["source"]["seal_kind"], benchmark.EXTERNAL_RAW_SEAL_KIND
            )

            changed = copy.deepcopy(document)
            changed["items"][0]["gold"] = "999999"
            changed["items"][0]["candidate_correct"] = True
            changed_path = root / "gold-changed.json"
            benchmark._write_json(changed_path, changed)
            second = _select_branch(changed_path, tokenizer, root / "branch-b.json")
            self.assertEqual(first["items"], second["items"])
            self.assertEqual(
                first["source"]["contract_sha256"],
                second["source"]["contract_sha256"],
            )
            self.assertNotEqual(
                first["source"]["raw_file_sha256"],
                second["source"]["raw_file_sha256"],
            )

            tampered = copy.deepcopy(document)
            tampered["items"][0]["prompt_token_ids"][0] = 2
            tampered_path = root / "tampered-source.json"
            benchmark._write_json(tampered_path, tampered)
            args = benchmark._parser().parse_args(
                [
                    "select-branch-cohort",
                    "--inputs",
                    str(tampered_path),
                    "--inputs-sha256",
                    benchmark._sha256_file(source),
                    "--tokenizer-json",
                    str(tokenizer),
                    "--limit",
                    "2",
                    "--output",
                    str(root / "never.json"),
                ]
            )
            args._require_official = False
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "raw file SHA-256 mismatch"
            ):
                benchmark.select_branch_cohort(args)

    def test_free_generation_compare_and_transition_evaluation_fail_closed(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            benchmark.QwenDirectDecodeError, "sealed worst-case bound"
        ):
            benchmark._branch_source_budget_preflight(
                argparse.Namespace(source_budget_mb=1, max_new_tokens=3),
                {"checkpoint_bytes": 300_000},
                items=2,
            )

        with tempfile.TemporaryDirectory(
            prefix=".qwen-branch-generation-test-", dir=Path.cwd()
        ) as temporary:
            root = Path(temporary)
            source_weights, inventory, _fingerprint = _fixture(root)
            tokenizer = root / "tokenizer.json"
            eos = _branch_tokenizer(tokenizer)
            source = root / "sealed-dev.json"
            _prepared_branch_source(source, eos=eos)
            branch_input = root / "branch-input.json"
            selected = _select_branch(source, tokenizer, branch_input)
            self.assertEqual(len(selected["items"]), 2)

            captures: list[dict] = []

            def runtime_factory(*factory_args, **factory_kwargs):
                runtime, model = benchmark._runtime(*factory_args, **factory_kwargs)
                feedback: list[int] = []
                original_decode = model.decode

                def recording_decode(token_ids, **kwargs):
                    feedback.append(int(token_ids[0][0]))
                    return original_decode(token_ids, **kwargs)

                model.decode = recording_decode
                captures.append(
                    {
                        "feedback": feedback,
                        "graft": factory_kwargs.get("graft"),
                        "graft_layer": factory_kwargs.get("graft_layer"),
                    }
                )
                return runtime, model

            off_path = root / "off.json"
            off_args = _branch_generation_args(
                branch_input=branch_input,
                tokenizer=tokenizer,
                output=off_path,
                trace=root / "off-trace.json",
                source=source_weights,
                inventory=inventory,
                cache=root / "cache-off",
                mode="off",
            )
            with redirect_stderr(io.StringIO()):
                off = benchmark.generate_arm(off_args, runtime_factory=runtime_factory)
            benchmark._write_json(off_path, off)

            candidate_path = root / "candidate.json"
            candidate_args = _branch_generation_args(
                branch_input=branch_input,
                tokenizer=tokenizer,
                output=candidate_path,
                trace=root / "candidate-trace.json",
                source=source_weights,
                inventory=inventory,
                cache=root / "cache-candidate",
                mode="stable-crsa",
            )
            with redirect_stderr(io.StringIO()):
                candidate = benchmark.generate_arm(
                    candidate_args, runtime_factory=runtime_factory
                )
            benchmark._write_json(candidate_path, candidate)

            self.assertEqual(len(captures), 2)
            self.assertIsNone(captures[0]["graft"])
            self.assertIsInstance(captures[1]["graft"], benchmark.Qwen38StableCrsaGraft)
            self.assertEqual(captures[1]["graft_layer"], 2)
            self.assertEqual(candidate["graft"]["max_history"], 12)
            for capture, arm in zip(captures, (off, candidate), strict=True):
                emitted = [
                    token
                    for row in arm["items"]
                    for token in row["generated_token_ids"]
                ]
                self.assertEqual(capture["feedback"], emitted)
                self.assertEqual(len(emitted), 6)
                self.assertTrue(all(row["forward_passes"] == 4 for row in arm["items"]))
                self.assertEqual(arm["execution"]["runtime_instances"], 1)
                self.assertTrue(arm["execution"]["shared_weight_pager"])

            # A real branch comparison admits divergence; equality is not a
            # transport-parity invariant for free-generated arms.
            off_eval = copy.deepcopy(off)
            candidate_eval = copy.deepcopy(candidate)
            _rewrite_generated_answer(off_eval, 0, tokens=[1, 1, 1], text="#### 1")
            _rewrite_generated_answer(off_eval, 1, tokens=[3, 3, 3], text="#### 3")
            _rewrite_generated_answer(
                candidate_eval, 0, tokens=[2, 2, 2], text="#### 2"
            )
            _rewrite_generated_answer(
                candidate_eval, 1, tokens=[2, 2, 2], text="#### 2"
            )
            benchmark._write_json(off_path, off_eval)
            benchmark._write_json(candidate_path, candidate_eval)
            compare_args = benchmark._parser().parse_args(
                [
                    "compare-generated-arms",
                    "--off",
                    str(off_path),
                    "--candidate",
                    str(candidate_path),
                    "--output",
                    str(root / "comparison.json"),
                ]
            )
            comparison = benchmark.compare_generated_arms(compare_args)
            comparison_path = Path(compare_args.output)
            benchmark._write_json(comparison_path, comparison)
            self.assertTrue(comparison["comparison"]["branch_effect_observed"])
            self.assertEqual(comparison["comparison"]["diverged_items"], 2)

            mismatched = copy.deepcopy(candidate_eval)
            mismatched["tokenizer"]["sha256"] = "f" * 64
            _reseal(mismatched)
            mismatch_path = root / "identity-mismatch.json"
            benchmark._write_json(mismatch_path, mismatched)
            mismatch_args = benchmark._parser().parse_args(
                [
                    "compare-generated-arms",
                    "--off",
                    str(off_path),
                    "--candidate",
                    str(mismatch_path),
                    "--output",
                    str(root / "mismatch-comparison.json"),
                ]
            )
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "tokenizer.*differs"
            ):
                benchmark.compare_generated_arms(mismatch_args)

            drifted_off = copy.deepcopy(off_eval)
            drifted_candidate = copy.deepcopy(candidate_eval)
            for arm in (drifted_off, drifted_candidate):
                for row in arm["items"]:
                    row["prompt_token_ids"] = [27, 27]
                    row["token_chain_sha256"] = benchmark._token_chain_sha256(
                        row["prompt_token_ids"], row["generated_token_ids"]
                    )
                _reseal(arm)
            drifted_off_path = root / "drifted-off.json"
            drifted_candidate_path = root / "drifted-candidate.json"
            benchmark._write_json(drifted_off_path, drifted_off)
            benchmark._write_json(drifted_candidate_path, drifted_candidate)
            drift_args = benchmark._parser().parse_args(
                [
                    "compare-generated-arms",
                    "--off",
                    str(drifted_off_path),
                    "--candidate",
                    str(drifted_candidate_path),
                    "--output",
                    str(root / "drift-comparison.json"),
                ]
            )
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "prompt differs from sealed input"
            ):
                benchmark.compare_generated_arms(drift_args)

            seal_tamper = copy.deepcopy(candidate_eval)
            seal_tamper["items"][0]["generated_text"] = "#### 999"
            seal_tamper_path = root / "seal-tamper.json"
            benchmark._write_json(seal_tamper_path, seal_tamper)
            tamper_args = benchmark._parser().parse_args(
                [
                    "compare-generated-arms",
                    "--off",
                    str(off_path),
                    "--candidate",
                    str(seal_tamper_path),
                    "--output",
                    str(root / "tamper-comparison.json"),
                ]
            )
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "seal mismatch"
            ):
                benchmark.compare_generated_arms(tamper_args)

            wrong_label_source = json.loads(source.read_text(encoding="utf-8"))
            wrong_label_source["items"][0]["gold"] = "999"
            wrong_label_path = root / "wrong-label-source.json"
            benchmark._write_json(wrong_label_path, wrong_label_source)
            wrong_label_args = benchmark._parser().parse_args(
                [
                    "evaluate-generated-arms",
                    "--off",
                    str(off_path),
                    "--candidate",
                    str(candidate_path),
                    "--comparison",
                    str(comparison_path),
                    "--gold-source",
                    str(wrong_label_path),
                    "--gold-source-sha256",
                    benchmark._sha256_file(source),
                    "--output",
                    str(root / "wrong-label-evaluation.json"),
                ]
            )
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "raw file SHA-256 mismatch"
            ):
                benchmark.evaluate_generated_arms(wrong_label_args)

            # Even if both arm producers collude to reseal a different raw
            # label commitment, the evaluator's independent pin wins.
            retargeted_off = copy.deepcopy(off_eval)
            retargeted_candidate = copy.deepcopy(candidate_eval)
            wrong_label_sha256 = benchmark._sha256_file(wrong_label_path)
            for arm in (retargeted_off, retargeted_candidate):
                arm["input"]["source"]["raw_file_sha256"] = wrong_label_sha256
                _reseal(arm["input"])
                _reseal(arm)
            retargeted_off_path = root / "retargeted-off.json"
            retargeted_candidate_path = root / "retargeted-candidate.json"
            benchmark._write_json(retargeted_off_path, retargeted_off)
            benchmark._write_json(retargeted_candidate_path, retargeted_candidate)
            retarget_compare_args = benchmark._parser().parse_args(
                [
                    "compare-generated-arms",
                    "--off",
                    str(retargeted_off_path),
                    "--candidate",
                    str(retargeted_candidate_path),
                    "--output",
                    str(root / "retargeted-comparison.json"),
                ]
            )
            retargeted_comparison = benchmark.compare_generated_arms(
                retarget_compare_args
            )
            benchmark._write_json(retarget_compare_args.output, retargeted_comparison)
            retarget_evaluate_args = benchmark._parser().parse_args(
                [
                    "evaluate-generated-arms",
                    "--off",
                    str(retargeted_off_path),
                    "--candidate",
                    str(retargeted_candidate_path),
                    "--comparison",
                    str(retarget_compare_args.output),
                    "--gold-source",
                    str(wrong_label_path),
                    "--gold-source-sha256",
                    benchmark._sha256_file(source),
                    "--output",
                    str(root / "retargeted-evaluation.json"),
                ]
            )
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError,
                "external SHA-256 differs from sealed branch input",
            ):
                benchmark.evaluate_generated_arms(retarget_evaluate_args)

            evaluate_args = benchmark._parser().parse_args(
                [
                    "evaluate-generated-arms",
                    "--off",
                    str(off_path),
                    "--candidate",
                    str(candidate_path),
                    "--comparison",
                    str(comparison_path),
                    "--gold-source",
                    str(source),
                    "--gold-source-sha256",
                    benchmark._sha256_file(source),
                    "--output",
                    str(root / "evaluation.json"),
                ]
            )
            evaluation = benchmark.evaluate_generated_arms(evaluate_args)
            self.assertEqual(evaluation["summary"]["wrong_to_correct"], 1)
            self.assertEqual(evaluation["summary"]["correct_to_wrong"], 1)
            self.assertEqual(evaluation["summary"]["unsafe_correct_to_wrong"], 1)
            self.assertEqual(evaluation["summary"]["verdict"], "unsafe")
            self.assertFalse(evaluation["summary"]["quality_success"])

    def test_native_generation_v3_evidence_and_generated_triad_fail_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".qwen-native-triad-test-", dir=Path.cwd()
        ) as temporary:
            root = Path(temporary)
            source_weights, inventory = _native_fixture(root)
            tokenizer = root / "tokenizer.json"
            eos = _branch_tokenizer(tokenizer)
            source = root / "sealed-dev.json"
            _prepared_branch_source(source, eos=eos)
            branch_input = root / "branch-input.json"
            _select_branch(source, tokenizer, branch_input)

            contaminated_args = _branch_generation_args(
                branch_input=branch_input,
                tokenizer=tokenizer,
                output=root / "contaminated-off.json",
                trace=root / "contaminated-off-trace.json",
                source=source_weights,
                inventory=inventory,
                cache=root / "cache-contaminated-off",
                mode="off",
            )
            contaminated_args.max_new_tokens = 1

            def contaminated_factory(*factory_args, **factory_kwargs):
                runtime, model = benchmark._runtime(*factory_args, **factory_kwargs)
                model.native_head_crsa = benchmark.Qwen38NativeHeadCrsa()
                model.native_head_crsa_observer = lambda _row: None
                return runtime, model

            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "runtime attachment differs"
            ):
                benchmark.generate_arm(
                    contaminated_args, runtime_factory=contaminated_factory
                )

            captures: list[dict] = []

            def runtime_factory(*factory_args, **factory_kwargs):
                runtime, model = benchmark._runtime(*factory_args, **factory_kwargs)
                capture = {
                    "graft": factory_kwargs.get("graft"),
                    "native_head_crsa": factory_kwargs.get("native_head_crsa"),
                    "observer": factory_kwargs.get("native_head_crsa_observer"),
                    "usage": [],
                }
                original_generate = model.generate_greedy

                def recording_generate(*generate_args, **generate_kwargs):
                    result = original_generate(*generate_args, **generate_kwargs)
                    state = model._layer_states[27]
                    capture["usage"].append(
                        (tuple(state.crsa_log_usage.shape), state.crsa_log_usage.dtype)
                    )
                    return result

                model.generate_greedy = recording_generate
                captures.append(capture)
                return runtime, model

            native_path = root / "native.json"
            native_args = _branch_generation_args(
                branch_input=branch_input,
                tokenizer=tokenizer,
                output=native_path,
                trace=root / "native-trace.json",
                source=source_weights,
                inventory=inventory,
                cache=root / "cache-native",
                mode="native-crsa",
            )
            native_args.max_new_tokens = 1
            self.assertEqual(
                (
                    native_args.native_alpha,
                    native_args.native_balance_alpha,
                    native_args.native_diagonal_debit,
                ),
                (0.01, 1.0, 3.0),
            )
            invalid_config = copy.copy(native_args)
            invalid_config.native_alpha = 0.02
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "requires alpha=0.01"
            ):
                benchmark._build_native_intervention(invalid_config)
            with redirect_stderr(io.StringIO()):
                native = benchmark.generate_arm(
                    native_args, runtime_factory=runtime_factory
                )
            benchmark._write_json(native_path, native)
            self.assertEqual(native["schema"], benchmark.NATIVE_BRANCH_RESULT_SCHEMA)
            self.assertNotIn("graft", native)
            self.assertEqual(native["intervention"]["kind"], "native-head-crsa")
            self.assertEqual(
                native["intervention"]["config"],
                {
                    "alpha": 0.01,
                    "balance_alpha": 1.0,
                    "diagonal_debit": 3.0,
                    "head_indices": [2, 8, 14, 20],
                    "layer": 27,
                },
            )
            self.assertEqual(len(captures), 1)
            self.assertIsNone(captures[0]["graft"])
            self.assertIsInstance(
                captures[0]["native_head_crsa"], benchmark.Qwen38NativeHeadCrsa
            )
            self.assertTrue(callable(captures[0]["observer"]))
            self.assertEqual(
                captures[0]["usage"],
                [((1, 4, 3), torch.float32), ((1, 4, 3), torch.float32)],
            )
            for row in native["items"]:
                receipts = row["intervention_evidence"]
                self.assertEqual(
                    row["intervention_evidence_sha256"],
                    benchmark._sha256(receipts),
                )
                self.assertEqual(len(receipts), row["forward_passes"])
                self.assertEqual(row["forward_passes"], 2)
                self.assertEqual(
                    [receipt["query_start"] for receipt in receipts], [0, 2]
                )
                self.assertEqual(
                    [receipt["history_length_after"] for receipt in receipts],
                    [2, 3],
                )
                self.assertTrue(all(receipt["layer"] == 27 for receipt in receipts))
                self.assertTrue(
                    all(
                        receipt["selected_query_heads"] == [2, 8, 14, 20]
                        and receipt["selected_kv_heads"] == [0, 1, 2, 3]
                        and len(receipt["free_heads"]) == 20
                        for receipt in receipts
                    )
                )
            self.assertEqual(benchmark._load_native_branch_result(native_path), native)

            def tampered_path(name: str, mutate) -> Path:
                document = copy.deepcopy(native)
                mutate(document)
                _reseal(document)
                path = root / f"tampered-{name}.json"
                benchmark._write_json(path, document)
                return path

            unsealed = copy.deepcopy(native)
            unsealed["items"][0]["generated_text"] = "#### 999"
            unsealed_path = root / "tampered-outer-seal.json"
            benchmark._write_json(unsealed_path, unsealed)
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "document seal mismatch"
            ):
                benchmark._load_native_branch_result(unsealed_path)

            def drift_source_identity(document):
                document["input"]["items"][0]["prompt_token_ids"][0] = 2

            source_tamper = tampered_path("source", drift_source_identity)
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "branch input.*seal mismatch"
            ):
                benchmark._load_native_branch_result(source_tamper)

            dropped = tampered_path(
                "count",
                lambda document: document["items"][0]["intervention_evidence"].pop(),
            )
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "count differs"
            ):
                benchmark._load_native_branch_result(dropped)

            def swap_evidence(document):
                rows = document["items"][0]["intervention_evidence"]
                rows[0], rows[1] = rows[1], rows[0]

            swapped = tampered_path("reordered", swap_evidence)
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "history chain"
            ):
                benchmark._load_native_branch_result(swapped)

            def drift_history(document):
                document["items"][0]["intervention_evidence"][1]["query_start"] += 1

            drifted = tampered_path("history", drift_history)
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "evidence.*invalid|history chain"
            ):
                benchmark._load_native_branch_result(drifted)

            def drift_config(document):
                document["intervention"]["config"]["alpha"] = 0.02

            config_tamper = tampered_path("config", drift_config)
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "intervention identity"
            ):
                benchmark._load_native_branch_result(config_tamper)

            def drift_free_heads(document):
                document["items"][0]["intervention_evidence"][0]["free_heads"][0] = 2

            heads_tamper = tampered_path("heads", drift_free_heads)
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "evidence is invalid"
            ):
                benchmark._load_native_branch_result(heads_tamper)

            def drift_tolerance(document):
                document["items"][0]["intervention_evidence"][0][
                    "row_sum_max_error"
                ] = benchmark.NATIVE_HEAD_CRSA_ROW_SUM_TOLERANCE + 1e-7

            tolerance_tamper = tampered_path("tolerance", drift_tolerance)
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "evidence is invalid"
            ):
                benchmark._load_native_branch_result(tolerance_tamper)

            def drift_future_mass(document):
                document["items"][0]["intervention_evidence"][0][
                    "future_weight_max_abs"
                ] = 1e-9

            future_tamper = tampered_path("future", drift_future_mass)
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "evidence is invalid"
            ):
                benchmark._load_native_branch_result(future_tamper)

            def drift_mean_l1(document):
                document["items"][0]["intervention_evidence"][0][
                    "mean_l1_probability_delta_per_head"
                ][0] = 1.5

            mean_l1_tamper = tampered_path("mean-l1", drift_mean_l1)
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "evidence digest mismatch"
            ):
                benchmark._load_native_branch_result(mean_l1_tamper)

            def drift_argmax(document):
                receipt = document["items"][0]["intervention_evidence"][0]
                current = receipt["argmax_changed_queries_per_head"][0]
                receipt["argmax_changed_queries_per_head"][0] = 0 if current != 0 else 1

            argmax_tamper = tampered_path("argmax", drift_argmax)
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "evidence digest mismatch"
            ):
                benchmark._load_native_branch_result(argmax_tamper)

            def drift_token_chain(document):
                document["items"][0]["token_chain_sha256"] = "f" * 64

            token_tamper = tampered_path("token-chain", drift_token_chain)
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "token-chain digest"
            ):
                benchmark._load_native_branch_result(token_tamper)

            def drift_trace(document):
                document["traffic"]["access_trace"]["sha256"] = "f" * 64
                document["traffic"]["access_trace_sha256"] = "f" * 64

            trace_tamper = tampered_path("trace", drift_trace)
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "trace.*differs|trace artifact"
            ):
                benchmark._load_native_branch_result(trace_tamper)

            native_eval = copy.deepcopy(native)
            _rewrite_generated_answer(native_eval, 0, tokens=[1], text="#### 1")
            _rewrite_generated_answer(native_eval, 1, tokens=[2], text="#### 2")
            off = benchmark._native_v2_common_projection(native_eval)
            _rewrite_generated_answer(off, 0, tokens=[3], text="#### 3")
            stable = copy.deepcopy(off)
            stable["arm"] = "stable-crsa"
            stable["graft"] = benchmark._build_graft(
                argparse.Namespace(
                    mode="stable-crsa",
                    graft_alpha=0.1,
                    graft_layer=2,
                    graft_max_history=12,
                    graft_rms_eps=1e-6,
                )
            )[2]
            _rewrite_generated_answer(stable, 0, tokens=[1], text="#### 1")
            off_path = root / "triad-off.json"
            stable_path = root / "triad-stable.json"
            native_eval_path = root / "triad-native.json"
            benchmark._write_json(off_path, off)
            benchmark._write_json(stable_path, stable)
            benchmark._write_json(native_eval_path, native_eval)
            self.assertEqual(benchmark._load_branch_result(off_path), off)
            self.assertEqual(benchmark._load_branch_result(stable_path), stable)
            self.assertEqual(
                benchmark._load_native_branch_result(native_eval_path), native_eval
            )
            legacy_pair_args = benchmark._parser().parse_args(
                [
                    "compare-generated-arms",
                    "--off",
                    str(off_path),
                    "--candidate",
                    str(native_eval_path),
                    "--output",
                    str(root / "legacy-native-rejected.json"),
                ]
            )
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "branch result schema"
            ):
                benchmark.compare_generated_arms(legacy_pair_args)

            compare_args = benchmark._parser().parse_args(
                [
                    "compare-generated-triad",
                    "--off",
                    str(off_path),
                    "--stable-crsa",
                    str(stable_path),
                    "--native-crsa",
                    str(native_eval_path),
                    "--output",
                    str(root / "triad.json"),
                ]
            )
            triad = benchmark.compare_generated_triad(compare_args)
            triad_path = Path(compare_args.output)
            benchmark._write_json(triad_path, triad)
            self.assertEqual(triad["schema"], benchmark.TRIAD_COMPARISON_SCHEMA)
            self.assertEqual(
                set(triad["arm_seals"]),
                {
                    "off",
                    "stable_crsa",
                    "native_crsa",
                },
            )
            self.assertEqual(triad["comparisons"]["off_vs_hidden"]["diverged_items"], 1)
            self.assertEqual(
                triad["comparisons"]["off_vs_native"]["parsed_answer_changes"],
                1,
            )

            evaluate_args = benchmark._parser().parse_args(
                [
                    "evaluate-generated-triad",
                    "--off",
                    str(off_path),
                    "--hidden",
                    str(stable_path),
                    "--native",
                    str(native_eval_path),
                    "--triad",
                    str(triad_path),
                    "--gold-source",
                    str(source),
                    "--gold-source-sha256",
                    benchmark._sha256_file(source),
                    "--output",
                    str(root / "triad-evaluation.json"),
                ]
            )
            evaluation = benchmark.evaluate_generated_triad(evaluate_args)
            self.assertEqual(evaluation["schema"], benchmark.TRIAD_EVALUATION_SCHEMA)
            self.assertEqual(evaluation["summary"]["hidden"]["wrong_to_correct"], 1)
            self.assertEqual(evaluation["summary"]["native"]["wrong_to_correct"], 1)
            self.assertEqual(evaluation["summary"]["native"]["correct_to_wrong"], 0)
            self.assertEqual(evaluation["summary"]["native"]["net_correct_delta"], 1)
            self.assertTrue(evaluation["summary"]["native"]["quality_success"])

            unsealed_triad = copy.deepcopy(triad)
            unsealed_triad["comparisons"]["off_vs_native"]["diverged_items"] = 0
            unsealed_path = root / "unsealed-triad.json"
            benchmark._write_json(unsealed_path, unsealed_triad)
            blocked_args = copy.copy(evaluate_args)
            blocked_args.triad = str(unsealed_path)
            with mock.patch.object(
                benchmark,
                "_externally_sealed_json",
                side_effect=AssertionError("gold opened too early"),
            ) as gold_open:
                with self.assertRaisesRegex(
                    benchmark.QwenDirectDecodeError, "seal mismatch"
                ):
                    benchmark.evaluate_generated_triad(blocked_args)
                gold_open.assert_not_called()

            invalid_native_args = copy.copy(evaluate_args)
            invalid_native_args.native = str(dropped)
            with mock.patch.object(
                benchmark,
                "_externally_sealed_json",
                side_effect=AssertionError("gold opened too early"),
            ) as gold_open:
                with self.assertRaisesRegex(
                    benchmark.QwenDirectDecodeError, "count differs"
                ):
                    benchmark.evaluate_generated_triad(invalid_native_args)
                gold_open.assert_not_called()


if __name__ == "__main__":
    unittest.main()
