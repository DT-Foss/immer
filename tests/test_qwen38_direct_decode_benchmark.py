from __future__ import annotations

from contextlib import redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

from immer.knowledge import AccessTrace
from immer.runtimes.qwen3_8 import verify_probe_document

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

            self.assertEqual(causal["hidden_sha256"], baseline["hidden_sha256"])
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
                baseline["source_verification"]["kind"],
                "complete-local-shards/v1",
            )
            for path in (
                prepare_args.access_trace,
                baseline_args.access_trace,
                causal_args.access_trace,
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


if __name__ == "__main__":
    unittest.main()
