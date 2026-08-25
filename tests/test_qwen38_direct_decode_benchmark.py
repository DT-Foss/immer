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


def _answer_branch_tokenizer(path: Path) -> tuple[int, int]:
    from tokenizers import AddedToken, Regex, Tokenizer
    from tokenizers.decoders import Fuse
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Split

    vocab = {
        "[UNK]": 0,
        **{str(index): index for index in range(1, 27)},
        "####": 27,
        " ": 28,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Split(Regex(r"####| "), behavior="isolated")
    tokenizer.decoder = Fuse()
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
    mapping["max_position_embeddings"] = 256
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
    questions: tuple[str, ...] | None = None,
    system_prompt: str = "fixture",
    thinking: bool = False,
) -> dict:
    if questions is not None and len(questions) != len(gold):
        raise ValueError("questions and gold must have equal length")
    item_ids = [f"dev-{index}" for index in range(len(gold))]
    document = {
        "items": [
            {
                "candidate_correct": False,
                "draft_token_ids": [9, 9],
                "gold": target,
                "item_id": item_id,
                "prompt_token_ids": [1, 4 + index],
                **({"question": questions[index]} if questions is not None else {}),
            }
            for index, (item_id, target) in enumerate(zip(item_ids, gold, strict=True))
        ],
        "protocol": {
            "accepted_eos_token_ids": list(eos),
            "batch_size": len(item_ids),
            "item_ids": item_ids,
            "system_prompt": system_prompt,
            "thinking": thinking,
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


def _select_answer_branch(
    source: Path,
    tokenizer: Path,
    output: Path,
    *,
    limit: int = 1,
):
    args = benchmark._parser().parse_args(
        [
            "select-answer-branch-cohort",
            "--inputs",
            str(source),
            "--inputs-sha256",
            benchmark._sha256_file(source),
            "--tokenizer-json",
            str(tokenizer),
            "--limit",
            str(limit),
            "--output",
            str(output),
        ]
    )
    args._require_official = False
    selected = benchmark.select_answer_branch_cohort(args)
    benchmark._write_json(output, selected)
    return selected


def _select_answer_questions(
    source: Path,
    branch_input: Path,
    output: Path,
):
    args = benchmark._parser().parse_args(
        [
            "select-answer-branch-questions",
            "--input",
            str(branch_input),
            "--inputs",
            str(source),
            "--inputs-sha256",
            benchmark._sha256_file(source),
            "--output",
            str(output),
        ]
    )
    selected = benchmark.select_answer_branch_questions(args)
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


def _native_fork_generation_args(
    *,
    branch_input: Path,
    tokenizer: Path,
    output: Path,
    trace: Path,
    source: Path,
    inventory: Path,
    cache: Path,
):
    args = benchmark._parser().parse_args(
        [
            "generate-native-fork",
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
            "64",
            "--max-resident-mb",
            "2",
            "--source-budget-mb",
            "20",
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
            transported = copy.deepcopy(native)
            transported["traffic"]["access_trace"]["path"] = (
                "/detached-host/access-native.json"
            )
            _reseal(transported)
            transported_path = root / "result-native.json"
            transported_trace = root / "access-native.json"
            transported_trace.write_bytes(Path(native_args.access_trace).read_bytes())
            benchmark._write_json(transported_path, transported)
            self.assertEqual(
                benchmark._load_native_branch_result(transported_path), transported
            )
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

    def test_fixed_answer_prefix_input_generates_only_the_numeric_value(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".qwen-answer-prefix-test-", dir=Path.cwd()
        ) as temporary:
            root = Path(temporary)
            source_weights, inventory = _native_fixture(root)
            tokenizer_path = root / "tokenizer.json"
            eos = _answer_branch_tokenizer(tokenizer_path)
            source = root / "sealed-answer-dev.json"
            oven_question = (
                "Maggie's oven is malfunctioning. When she sets it to 450 the "
                "actual temperature is 468. If it's off by the same percentage "
                "for any recipe, what temperature should she set it at if her "
                "recipe calls for 520 degrees?"
            )
            source_document = _prepared_branch_source(
                source,
                eos=eos,
                gold=("500",),
                questions=(oven_question,),
                system_prompt=benchmark.ANSWER_OUTPUT_INSTRUCTION,
                thinking=False,
            )
            answer_tokenizer = benchmark.Qwen38Tokenizer(
                tokenizer_path, require_official=False
            )
            source_prompt_ids = list(
                answer_tokenizer.encode(
                    answer_tokenizer.render_no_thinking_prompt(
                        benchmark.ANSWER_OUTPUT_INSTRUCTION,
                        oven_question,
                    )
                )
            )
            source_document["items"][0]["prompt_token_ids"] = source_prompt_ids
            benchmark._write_json(source, source_document)
            answer_input_path = root / "answer-input.json"
            selected = _select_answer_branch(source, tokenizer_path, answer_input_path)
            self.assertEqual(selected["schema"], benchmark.BRANCH_INPUT_SCHEMA_V3)
            prefix = selected["protocol"]["generation_prefix"]
            self.assertEqual(prefix["literal"], "#### ")
            self.assertTrue(prefix["label_free"])
            self.assertTrue(prefix["prefix_is_part_of_sealed_prompt"])
            self.assertEqual(prefix["teacher_forced_tokens_after_prompt"], 0)
            self.assertEqual(
                tuple(prefix["token_ids"]),
                benchmark.Qwen38Tokenizer(
                    tokenizer_path, require_official=False
                ).encode("#### "),
            )
            self.assertEqual(prefix["token_ids"], [27, 28])
            self.assertEqual(
                benchmark.Qwen38Tokenizer(
                    tokenizer_path, require_official=False
                ).decode(prefix["token_ids"]),
                "#### ",
            )
            self.assertEqual(
                benchmark.OFFICIAL_ANSWER_GENERATION_PREFIX_TOKEN_IDS,
                (794, 220),
            )
            item = selected["items"][0]
            self.assertEqual(item["source_prompt_token_ids"], source_prompt_ids)
            self.assertEqual(
                item["effective_prompt_token_ids"],
                [*source_prompt_ids, *prefix["token_ids"]],
            )
            self.assertEqual(benchmark._load_branch_input(answer_input_path), selected)

            legacy_path = root / "legacy-input.json"
            legacy = _select_branch(source, tokenizer_path, legacy_path, limit=1)
            self.assertEqual(legacy["schema"], benchmark.BRANCH_INPUT_SCHEMA)
            self.assertEqual(legacy["items"][0]["prompt_token_ids"], source_prompt_ids)

            observed_prompts: list[tuple[int, ...]] = []

            def runtime_factory(*factory_args, **factory_kwargs):
                runtime, model = benchmark._runtime(*factory_args, **factory_kwargs)
                original_generate = model.generate_greedy

                def recording_generate(prompt_token_ids, **generate_kwargs):
                    observed_prompts.append(tuple(prompt_token_ids[0]))
                    return original_generate(prompt_token_ids, **generate_kwargs)

                def fixed_topk(
                    _hidden, *, k, name="lm_head.weight", block_rows, progress=None
                ):
                    del k, name, block_rows
                    if progress is not None:
                        progress(
                            {
                                "rows_done": model.config.vocab_size,
                                "vocab_rows": model.config.vocab_size,
                            }
                        )
                    return (
                        torch.tensor([[1.0]], device=model.pager.device),
                        torch.tensor([[1]], device=model.pager.device),
                    )

                model.generate_greedy = recording_generate
                model.pager.topk_logits = fixed_topk
                return runtime, model

            result_path = root / "answer-off.json"
            generation_args = _branch_generation_args(
                branch_input=answer_input_path,
                tokenizer=tokenizer_path,
                output=result_path,
                trace=root / "answer-trace.json",
                source=source_weights,
                inventory=inventory,
                cache=root / "answer-cache",
                mode="native-crsa",
            )
            generation_args.max_new_tokens = 1
            generation_args.max_seq_len = len(item["effective_prompt_token_ids"]) + 1
            with redirect_stderr(io.StringIO()):
                result = benchmark.generate_arm(
                    generation_args, runtime_factory=runtime_factory
                )
            benchmark._write_json(result_path, result)
            self.assertEqual(result["schema"], benchmark.NATIVE_BRANCH_RESULT_SCHEMA)
            self.assertEqual(
                observed_prompts, [tuple(item["effective_prompt_token_ids"])]
            )
            generated = result["items"][0]
            self.assertEqual(generated["generated_text"], "1")
            self.assertNotIn("####", generated["generated_text"])
            self.assertEqual(generated["parsed_numeric_answer"], "1")
            self.assertEqual(benchmark._load_native_branch_result(result_path), result)
            doubled = copy.deepcopy(result)
            doubled["items"][0]["generated_text"] = "#### 1"
            doubled["items"][0]["parsed_numeric_answer"] = "1"
            _reseal(doubled)
            doubled_path = root / "answer-doubled-prefix.json"
            benchmark._write_json(doubled_path, doubled)
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "re-emits the sealed #### prefix"
            ):
                benchmark._load_native_branch_result(doubled_path)
            _rows, targets = benchmark._gold_rows_for_branch(source_document, result)
            self.assertEqual(targets, {"dev-0": "500"})

            questions_path = root / "answer-questions.json"
            questions = _select_answer_questions(
                source, answer_input_path, questions_path
            )
            self.assertEqual(questions["schema"], benchmark.BRANCH_QUESTION_SCHEMA)
            self.assertEqual(questions["items"][0]["question"], oven_question)
            self.assertFalse(questions["protocol"]["projection_contains_label_fields"])
            self.assertTrue(questions["protocol"]["source_file_may_contain_labels"])
            self.assertEqual(
                questions["protocol"]["system_prompt"],
                benchmark.ANSWER_OUTPUT_INSTRUCTION,
            )

            native_eval = copy.deepcopy(result)
            _rewrite_generated_answer(native_eval, 0, tokens=[4], text="504")
            off = benchmark._native_v2_common_projection(native_eval)
            _rewrite_generated_answer(off, 0, tokens=[2], text="502")
            stable = copy.deepcopy(off)
            stable["arm"] = "stable-crsa"
            stable["graft"] = benchmark._build_graft(
                argparse.Namespace(
                    mode="stable-crsa",
                    graft_alpha=0.1,
                    graft_layer=2,
                    graft_max_history=256,
                    graft_rms_eps=1e-6,
                )
            )[2]
            _rewrite_generated_answer(stable, 0, tokens=[4], text="504")
            off_path = root / "adjudication-off.json"
            stable_path = root / "adjudication-stable.json"
            native_path = root / "adjudication-native.json"
            benchmark._write_json(off_path, off)
            benchmark._write_json(stable_path, stable)
            benchmark._write_json(native_path, native_eval)
            triad = benchmark._compare_generated_triad_documents(
                off, stable, native_eval
            )
            triad_path = root / "adjudication-triad.json"
            benchmark._write_json(triad_path, triad)
            adjudication_args = benchmark._parser().parse_args(
                [
                    "adjudicate-generated-triad",
                    "--off",
                    str(off_path),
                    "--stable",
                    str(stable_path),
                    "--native",
                    str(native_path),
                    "--triad",
                    str(triad_path),
                    "--questions",
                    str(questions_path),
                    "--tokenizer-json",
                    str(tokenizer_path),
                    "--output",
                    str(root / "adjudication.json"),
                ]
            )
            adjudication = benchmark.adjudicate_generated_triad(adjudication_args)
            adjudication_path = Path(adjudication_args.output)
            benchmark._write_json(adjudication_path, adjudication)
            self.assertEqual(
                adjudication["schema"], benchmark.BRANCH_ADJUDICATION_SCHEMA
            )
            self.assertFalse(adjudication["protocol"]["gold_accessed_by_adjudicator"])
            self.assertEqual(adjudication["summary"]["certificate_overrides"], 1)
            adjudicated = adjudication["items"][0]
            self.assertEqual(
                adjudicated["candidates"],
                {"native_crsa": "504", "off": "502", "stable_crsa": "504"},
            )
            self.assertEqual(adjudicated["decision"], "certificate_override")
            self.assertEqual(adjudicated["selected_answer"], "500")
            self.assertEqual(
                {row["status"] for row in adjudicated["verifications"].values()},
                {"mismatch"},
            )
            self.assertEqual(
                benchmark._load_branch_adjudication_document(adjudication_path),
                adjudication,
            )

            evaluation_args = benchmark._parser().parse_args(
                [
                    "evaluate-adjudicated-triad",
                    "--off",
                    str(off_path),
                    "--stable",
                    str(stable_path),
                    "--native",
                    str(native_path),
                    "--triad",
                    str(triad_path),
                    "--questions",
                    str(questions_path),
                    "--tokenizer-json",
                    str(tokenizer_path),
                    "--adjudication",
                    str(adjudication_path),
                    "--gold-source",
                    str(source),
                    "--gold-source-sha256",
                    benchmark._sha256_file(source),
                    "--output",
                    str(root / "adjudication-evaluation.json"),
                ]
            )
            evaluation = benchmark.evaluate_adjudicated_triad(evaluation_args)
            self.assertEqual(
                evaluation["schema"],
                benchmark.BRANCH_ADJUDICATION_EVALUATION_SCHEMA,
            )
            self.assertEqual(evaluation["summary"]["wrong_to_correct"], 1)
            self.assertEqual(evaluation["summary"]["net_correct_delta"], 1)
            self.assertEqual(evaluation["summary"]["verdict"], "improved")
            self.assertTrue(evaluation["summary"]["quality_success"])

            unsupported = benchmark._build_adjudication_item(
                item_id="unsupported",
                question="What is the capital of France?",
                candidates={"native_crsa": "4", "off": "4", "stable_crsa": "4"},
            )
            self.assertEqual(unsupported["decision"], "abstained")
            self.assertIsNone(unsupported["selected_answer"])

            question_tamper = copy.deepcopy(questions)
            question_tamper["items"][0]["question"] += " changed"
            question_tamper_path = root / "question-tamper.json"
            benchmark._write_json(question_tamper_path, question_tamper)
            tampered_args = copy.copy(adjudication_args)
            tampered_args.questions = str(question_tamper_path)
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "seal mismatch"
            ):
                benchmark.adjudicate_generated_triad(tampered_args)

            resealed_question = copy.deepcopy(questions)
            resealed_question["items"][0]["question"] = oven_question.replace(
                "520 degrees?", "1 degrees?"
            )
            resealed_question["items"][0]["question_sha256"] = hashlib.sha256(
                resealed_question["items"][0]["question"].encode("utf-8")
            ).hexdigest()
            _reseal(resealed_question)
            resealed_question_path = root / "question-resealed-forgery.json"
            benchmark._write_json(resealed_question_path, resealed_question)
            forged_args = copy.copy(adjudication_args)
            forged_args.questions = str(resealed_question_path)
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError,
                "does not reproduce the sealed source prompt",
            ):
                benchmark.adjudicate_generated_triad(forged_args)

            evidence_tamper = copy.deepcopy(adjudication)
            evidence_tamper["items"][0]["selected_answer"] = "504"
            _reseal(evidence_tamper)
            evidence_tamper_path = root / "adjudication-evidence-tamper.json"
            benchmark._write_json(evidence_tamper_path, evidence_tamper)
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "evidence mismatch"
            ):
                benchmark._load_branch_adjudication_document(evidence_tamper_path)
            blocked_evaluation = copy.copy(evaluation_args)
            blocked_evaluation.adjudication = str(evidence_tamper_path)
            with mock.patch.object(
                benchmark,
                "_externally_sealed_json",
                side_effect=AssertionError("gold opened before adjudication seal"),
            ) as gold_open:
                with self.assertRaisesRegex(
                    benchmark.QwenDirectDecodeError, "evidence mismatch"
                ):
                    benchmark.evaluate_adjudicated_triad(blocked_evaluation)
                gold_open.assert_not_called()

            def assert_tamper_rejected(name: str, mutate) -> None:
                document = copy.deepcopy(selected)
                mutate(document)
                _reseal(document)
                path = root / f"answer-tamper-{name}.json"
                benchmark._write_json(path, document)
                with self.assertRaises(benchmark.QwenDirectDecodeError):
                    benchmark._load_branch_input(path)

            assert_tamper_rejected(
                "literal",
                lambda document: document["protocol"]["generation_prefix"].update(
                    {"literal": "### "}
                ),
            )
            assert_tamper_rejected(
                "tokens",
                lambda document: document["protocol"]["generation_prefix"][
                    "token_ids"
                ].__setitem__(0, 1),
            )
            assert_tamper_rejected(
                "source-prompt",
                lambda document: document["items"][0][
                    "source_prompt_token_ids"
                ].__setitem__(0, 2),
            )
            assert_tamper_rejected(
                "effective-prompt",
                lambda document: document["items"][0][
                    "effective_prompt_token_ids"
                ].__setitem__(0, 2),
            )

            thinking_source = root / "thinking-source.json"
            _prepared_branch_source(
                thinking_source,
                eos=eos,
                gold=("1",),
                system_prompt=benchmark.ANSWER_OUTPUT_INSTRUCTION,
                thinking=True,
            )
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "disable thinking"
            ):
                _select_answer_branch(
                    thinking_source,
                    tokenizer_path,
                    root / "thinking-never.json",
                )

            vague_source = root / "vague-source.json"
            _prepared_branch_source(
                vague_source,
                eos=eos,
                gold=("1",),
                system_prompt="Return a numeric answer.",
                thinking=False,
            )
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "explicit ####"
            ):
                _select_answer_branch(
                    vague_source,
                    tokenizer_path,
                    root / "vague-never.json",
                )

    def test_native_fork_pair_matches_independent_references_and_saves_work(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            benchmark.QwenDirectDecodeError, "sealed worst-case bound"
        ):
            benchmark._native_fork_source_budget_preflight(
                argparse.Namespace(source_budget_mb=3, max_new_tokens=3),
                {"checkpoint_bytes": 300_000},
                items=2,
            )

        with tempfile.TemporaryDirectory(
            prefix=".qwen-native-fork-benchmark-test-", dir=Path.cwd()
        ) as temporary:
            root = Path(temporary)
            source_weights, inventory = _native_fixture(root)
            tokenizer_path = root / "tokenizer.json"
            eos = _answer_branch_tokenizer(tokenizer_path)
            source = root / "sealed-answer-dev.json"
            _prepared_branch_source(
                source,
                eos=eos,
                gold=("1",),
                system_prompt=benchmark.ANSWER_OUTPUT_INSTRUCTION,
                thinking=False,
            )
            branch_input = root / "answer-input.json"
            selected = _select_answer_branch(source, tokenizer_path, branch_input)

            production_without_causal = _native_fork_generation_args(
                branch_input=branch_input,
                tokenizer=tokenizer_path,
                output=root / "production-never.json",
                trace=root / "production-never-trace.json",
                source=source_weights,
                inventory=inventory,
                cache=root / "production-never-cache",
            )
            production_without_causal._require_official = True
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "local complete causal bundle"
            ):
                benchmark.generate_native_fork(production_without_causal)

            legacy = root / "legacy-input.json"
            _select_branch(source, tokenizer_path, legacy, limit=1)
            rejected = _native_fork_generation_args(
                branch_input=legacy,
                tokenizer=tokenizer_path,
                output=root / "legacy-never.json",
                trace=root / "legacy-never-trace.json",
                source=source_weights,
                inventory=inventory,
                cache=root / "legacy-cache",
            )
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "answer-prefix v3"
            ):
                benchmark.generate_native_fork(rejected)

            eos_token = eos[0]

            def install_complete_causal_receipt(runtime) -> None:
                inventory_document = runtime.source.inventory()
                fingerprint = runtime.source.metrics()["inventory_source_fingerprint"]
                checkpoint_bytes = runtime.verification["checkpoint_bytes"]
                runtime.verification = {
                    "checkpoint_bytes": checkpoint_bytes,
                    "graph_revision": [
                        1,
                        benchmark._sha256({"graph": fingerprint}),
                    ],
                    "kind": "complete-causal-bundle/v1",
                    "layout_fingerprint": fingerprint,
                    "manifest_sha256": benchmark._sha256({"manifest": fingerprint}),
                    "shards": len(inventory_document["shards"]),
                    "shards_sha256": benchmark._sha256({"shards": fingerprint}),
                    "tensor_bindings": len(inventory_document["tensors"]),
                    "weights_layout": "nested/v1",
                }

            def install_scripted_topk(model, tokens: list[int]) -> None:
                pending = list(tokens)

                def fixed_topk(
                    _hidden, *, k, name="lm_head.weight", block_rows, progress=None
                ):
                    del k, name, block_rows
                    if not pending:
                        raise AssertionError("unexpected LM-head scan")
                    if progress is not None:
                        progress(
                            {
                                "rows_done": model.config.vocab_size,
                                "vocab_rows": model.config.vocab_size,
                            }
                        )
                    return (
                        torch.tensor([[1.0]], device=model.pager.device),
                        torch.tensor([[pending.pop(0)]], device=model.pager.device),
                    )

                model.pager.topk_logits = fixed_topk

            pair_runtime_calls = 0

            def pair_factory(*factory_args, **factory_kwargs):
                nonlocal pair_runtime_calls
                pair_runtime_calls += 1
                runtime, fork = benchmark._native_fork_runtime(
                    *factory_args, **factory_kwargs
                )
                install_complete_causal_receipt(runtime)
                install_scripted_topk(fork, [1, 1, eos_token, 5, eos_token])
                return runtime, fork

            pair_path = root / "pair.json"
            pair_args = _native_fork_generation_args(
                branch_input=branch_input,
                tokenizer=tokenizer_path,
                output=pair_path,
                trace=root / "pair-trace.json",
                source=source_weights,
                inventory=inventory,
                cache=root / "pair-cache",
            )
            with redirect_stderr(io.StringIO()):
                pair = benchmark.generate_native_fork(
                    pair_args, runtime_factory=pair_factory
                )
            benchmark._write_json(pair_path, pair)
            self.assertEqual(pair_runtime_calls, 1)
            self.assertEqual(pair["schema"], benchmark.NATIVE_FORK_PAIR_SCHEMA)
            self.assertEqual(pair["input"], selected)
            self.assertEqual(pair["execution"]["runtime_instances"], 1)
            self.assertEqual(pair["execution"]["source_instances"], 1)
            self.assertEqual(pair["execution"]["weight_pagers"], 1)
            self.assertEqual(
                pair["items"][0]["off"]["generated_token_ids"], [1, eos_token]
            )
            self.assertEqual(
                pair["items"][0]["native"]["generated_token_ids"],
                [1, 5, eos_token],
            )
            self.assertEqual(pair["items"][0]["off"]["forward_passes"], 3)
            self.assertEqual(pair["items"][0]["native"]["forward_passes"], 4)
            self.assertTrue(pair["items"][0]["fork"]["permanently_split"])
            self.assertEqual(pair["items"][0]["fork"]["first_divergence_index"], 1)
            self.assertEqual(pair["items"][0]["fork"]["joined_pair_forward_passes"], 2)
            self.assertEqual(
                len(pair["items"][0]["native"]["intervention_evidence"]), 4
            )
            self.assertEqual(
                pair["traffic"]["complete_layers_saved"],
                2 * benchmark.NATIVE_HEAD_CRSA_LAYER,
            )
            self.assertEqual(
                pair["traffic"]["runtime_source_body_bytes"],
                pair["traffic"]["source_body_bytes"],
            )
            self.assertEqual(benchmark._load_native_fork_pair(pair_path), pair)
            remote_bundle = copy.deepcopy(pair)
            remote_bundle["bundle"]["kind"] = "remote-pinned-range-source/v1"
            _reseal(remote_bundle)
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "complete causal bundle"
            ):
                benchmark._validate_native_fork_pair_document(remote_bundle)

            def arm_factory(tokens: list[int]):
                def factory(*factory_args, **factory_kwargs):
                    runtime, model = benchmark._runtime(*factory_args, **factory_kwargs)
                    install_complete_causal_receipt(runtime)
                    install_scripted_topk(model, tokens)
                    return runtime, model

                return factory

            off_path = root / "off.json"
            off_args = _branch_generation_args(
                branch_input=branch_input,
                tokenizer=tokenizer_path,
                output=off_path,
                trace=root / "off-trace.json",
                source=source_weights,
                inventory=inventory,
                cache=root / "off-cache",
                mode="off",
            )
            off_args.head_block_rows = 64
            with redirect_stderr(io.StringIO()):
                off = benchmark.generate_arm(
                    off_args, runtime_factory=arm_factory([1, eos_token])
                )
            benchmark._write_json(off_path, off)

            native_path = root / "native.json"
            native_args = _branch_generation_args(
                branch_input=branch_input,
                tokenizer=tokenizer_path,
                output=native_path,
                trace=root / "native-trace.json",
                source=source_weights,
                inventory=inventory,
                cache=root / "native-cache",
                mode="native-crsa",
            )
            native_args.head_block_rows = 64
            with redirect_stderr(io.StringIO()):
                native = benchmark.generate_arm(
                    native_args,
                    runtime_factory=arm_factory([1, 5, eos_token]),
                )
            benchmark._write_json(native_path, native)

            comparison_path = root / "comparison.json"
            comparison_args = benchmark._parser().parse_args(
                [
                    "compare-native-fork-references",
                    "--pair",
                    str(pair_path),
                    "--off",
                    str(off_path),
                    "--native",
                    str(native_path),
                    "--output",
                    str(comparison_path),
                ]
            )
            comparison = benchmark.compare_native_fork_references(comparison_args)
            benchmark._write_json(comparison_path, comparison)
            self.assertTrue(comparison["items"][0]["reference_parity"])
            for metric in (
                "source_body_bytes",
                "linear_calls",
                "complete_layers",
                "layer_weight_passes",
                "fork_layer_weight_passes",
            ):
                receipt = comparison["savings"][metric]
                self.assertEqual(
                    receipt["saved"], receipt["reference"] - receipt["fork"]
                )
            self.assertEqual(
                comparison["savings"]["source_body_bytes"]["measurement"],
                "runtime-source-body-byte-counters/v1",
            )
            self.assertEqual(
                comparison["savings"]["complete_layers"]["saved"],
                pair["traffic"]["complete_layers_saved"],
            )
            self.assertEqual(
                comparison["savings"]["layer_weight_passes"]["saved"],
                pair["traffic"]["complete_layers_saved"]
                + pair["traffic"]["joined_fork_layer_weight_passes"],
            )
            self.assertEqual(
                comparison["savings"]["fork_layer_weight_passes"]["saved"], 2
            )
            for metric in (
                "complete_layers",
                "layer_weight_passes",
                "linear_calls",
                "source_body_bytes",
            ):
                nonsense = copy.deepcopy(comparison)
                nonsense["savings"][metric].update(
                    {"fork": 0, "reference": 1, "saved": 1}
                )
                _reseal(nonsense)
                with self.assertRaisesRegex(
                    benchmark.QwenDirectDecodeError, "measured savings"
                ):
                    benchmark._validate_native_fork_reference_comparison_document(
                        nonsense
                    )

            broken_seal = copy.deepcopy(pair)
            broken_seal["status"] = "tampered"
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "not sealed|seal mismatch|schema"
            ):
                benchmark._validate_native_fork_pair_document(broken_seal)
            broken_accounting = copy.deepcopy(pair)
            broken_accounting["items"][0]["traffic"]["complete_layers_saved"] += 1
            _reseal(broken_accounting)
            with self.assertRaisesRegex(
                benchmark.QwenDirectDecodeError, "complete_layers_saved"
            ):
                benchmark._validate_native_fork_pair_document(broken_accounting)
            broken_evidence = copy.deepcopy(pair)
            broken_evidence["items"][0]["native"]["intervention_evidence"][0][
                "future_weight_max_abs"
            ] = 0.1
            broken_evidence["items"][0]["native"]["intervention_evidence_sha256"] = (
                benchmark._sha256(
                    broken_evidence["items"][0]["native"]["intervention_evidence"]
                )
            )
            _reseal(broken_evidence)
            with self.assertRaisesRegex(benchmark.QwenDirectDecodeError, "evidence"):
                benchmark._validate_native_fork_pair_document(broken_evidence)


if __name__ == "__main__":
    unittest.main()
