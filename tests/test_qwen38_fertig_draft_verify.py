from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

import torch

from immer.knowledge import AccessTrace, AccessTraceRecorder
from immer.knowledge.access_trace import AccessLeaf, AccessOperation


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen38_fertig_draft_verify.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "qwen38_fertig_draft_verify",
        SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import Qwen3.8 FERTIG verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verify_script = _load_script()


class _ByteTokenizer:
    def __init__(self, _path: str | Path) -> None:
        pass

    @staticmethod
    def render_no_thinking_prompt(system: str, user: str) -> str:
        return f"{system}\0{user}\0"

    @staticmethod
    def encode(text: str) -> tuple[int, ...]:
        return tuple(1000 + value for value in text.encode("utf-8"))

    @staticmethod
    def decode(token_ids) -> str:
        return bytes(int(token) - 1000 for token in token_ids).decode("utf-8")

    @staticmethod
    def token_pieces(token_ids) -> tuple[str, ...]:
        return tuple(chr(int(token) - 1000) for token in token_ids)


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    tokenizer = root / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    benchmark_rows = []
    baseline_rows = []
    for index, item_id in enumerate(verify_script.FIXED_ITEM_IDS):
        question = f"synthetic question {index}?"
        gold = str(index)
        text = f"#### {index}"
        prompt = _ByteTokenizer.render_no_thinking_prompt(
            verify_script.BASELINE_SYSTEM_PROMPT,
            question,
        )
        benchmark_rows.append(
            {
                "item_id": item_id,
                "status": "abstained",
                "question": question,
                "gold": f"work\n#### {gold}",
            }
        )
        baseline_rows.append(
            {
                "item_id": item_id,
                "question": question,
                "gold": gold,
                "predicted": gold,
                "correct": True,
                "status": "correct",
                "prompt_tokens": len(_ByteTokenizer.encode(prompt)),
                "completion_tokens": len(_ByteTokenizer.encode(text)),
                "finish_reason": "stop",
                "text": text,
            }
        )
    benchmark = root / "benchmark.json"
    benchmark.write_text(json.dumps({"items": benchmark_rows}), encoding="utf-8")
    drafts = root / "drafts.json"
    drafts.write_text(
        json.dumps(
            {
                "schema": verify_script.BASELINE_SCHEMA,
                "model": {
                    "revision": verify_script.BASELINE_MODEL_REVISION,
                    "backend": "openai-compatible",
                },
                "protocol": {
                    "system_prompt": verify_script.BASELINE_SYSTEM_PROMPT,
                    "temperature": 0.0,
                    "seed": 0,
                    "max_tokens": 32,
                },
                "summary": {
                    "total": 8,
                    "correct": 8,
                    "incorrect": 0,
                    "truncated": 0,
                    "unparseable": 0,
                    "error": 0,
                    "accuracy": 1.0,
                },
                "items": baseline_rows,
            }
        ),
        encoding="utf-8",
    )
    return tokenizer, benchmark, drafts


class Qwen38FertigDraftVerifyTests(unittest.TestCase):
    def test_prepare_only_validates_contextual_tokens_and_writes_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer, benchmark, drafts = _write_fixture(root)
            output = root / "inputs.json"
            args = verify_script._parser().parse_args(
                [
                    "--tokenizer-json",
                    str(tokenizer),
                    "--benchmark",
                    str(benchmark),
                    "--drafts-json",
                    str(drafts),
                    "--result-json",
                    str(output),
                    "--prepare-only",
                ]
            )

            with mock.patch.object(
                verify_script,
                "_verify_tokenizer_digest",
                side_effect=lambda path: path,
            ):
                result, written = verify_script.run(
                    args,
                    tokenizer_factory=_ByteTokenizer,
                )

            self.assertEqual(written, output.resolve())
            self.assertEqual(result["schema"], verify_script.INPUT_SCHEMA)
            self.assertEqual(result["summary"]["items"], 8)
            self.assertEqual(result["summary"]["candidate_accuracy"], 1.0)
            self.assertEqual(
                result["protocol"]["accepted_eos_token_ids"],
                [verify_script.IM_END_TOKEN_ID, verify_script.END_OF_TEXT_TOKEN_ID],
            )
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)

    def test_baseline_truncation_is_preserved_but_cannot_be_an_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer_path, benchmark, drafts = _write_fixture(root)
            tokenizer = _ByteTokenizer(tokenizer_path)
            document = json.loads(drafts.read_text(encoding="utf-8"))
            partial = "x" * 32
            document["items"][0].update(
                {
                    "predicted": None,
                    "correct": None,
                    "status": "truncated",
                    "completion_tokens": 32,
                    "finish_reason": "length",
                    "text": partial,
                }
            )
            document["summary"]["correct"] = 7
            document["summary"]["truncated"] = 1
            document["summary"]["accuracy"] = 7 / 8
            drafts.write_text(json.dumps(document), encoding="utf-8")
            rows = verify_script._prepare_drafts(
                benchmark,
                drafts,
                tokenizer,
                max_draft_tokens=32,
            )
            self.assertEqual(rows[0].candidate_status, "truncated")
            self.assertIsNone(rows[0].answer)
            self.assertIsNone(rows[0].candidate_correct)
            self.assertEqual(rows[0].finish_reason, "length")

            inputs = verify_script._input_document(rows)
            self.assertEqual(inputs["summary"]["candidate_correct"], 7)
            self.assertEqual(inputs["summary"]["candidate_incomplete"], 1)

            document["items"][0]["text"] = partial[:-1]
            document["items"][0]["completion_tokens"] = 31
            drafts.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(verify_script.CliError, "truncation"):
                verify_script._prepare_drafts(
                    benchmark,
                    drafts,
                    tokenizer,
                    max_draft_tokens=32,
                )

            document["items"][0]["text"] = partial
            document["items"][0]["completion_tokens"] = 32
            document["items"][0]["prompt_tokens"] += 1
            drafts.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(verify_script.CliError, "prompt token"):
                verify_script._prepare_drafts(
                    benchmark,
                    drafts,
                    tokenizer,
                    max_draft_tokens=32,
                )

    def test_dynamic_cohort_is_explicit_ordered_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer_path, benchmark, drafts = _write_fixture(root)
            document = json.loads(drafts.read_text(encoding="utf-8"))
            rows = document["items"][:3]
            item_ids = [row["item_id"] for row in rows]
            document["items"] = rows
            document["benchmark"] = {
                "item_ids": item_ids,
                "selection": {
                    "cohort": "abstained",
                    "limit": 3,
                },
            }
            correct = sum(bool(row["correct"]) for row in rows)
            document["summary"].update(
                {
                    "total": 3,
                    "correct": correct,
                    "incorrect": 3 - correct,
                    "accuracy": correct / 3,
                }
            )

            def seal() -> None:
                document.pop("report_sha256", None)
                document["report_sha256"] = hashlib.sha256(
                    json.dumps(
                        document,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()

            seal()
            drafts.write_text(json.dumps(document), encoding="utf-8")

            selected = verify_script._cohort_item_ids(drafts, dynamic=True)
            prepared = verify_script._prepare_drafts(
                benchmark,
                drafts,
                _ByteTokenizer(tokenizer_path),
                max_draft_tokens=32,
                item_ids=selected,
            )

            self.assertEqual(selected, tuple(item_ids))
            self.assertEqual([row.item_id for row in prepared], item_ids)
            self.assertEqual(
                verify_script._cohort_item_ids(drafts, dynamic=False),
                verify_script.FIXED_ITEM_IDS,
            )

            benchmark_document = json.loads(benchmark.read_text(encoding="utf-8"))
            benchmark_document["items"][0]["status"] = "correct"
            benchmark.write_text(json.dumps(benchmark_document), encoding="utf-8")
            with self.assertRaisesRegex(verify_script.CliError, "cohort status"):
                verify_script._prepare_drafts(
                    benchmark,
                    drafts,
                    _ByteTokenizer(tokenizer_path),
                    max_draft_tokens=32,
                    item_ids=selected,
                )
            benchmark_document["items"][0]["status"] = "abstained"
            benchmark.write_text(json.dumps(benchmark_document), encoding="utf-8")

            document["benchmark"]["item_ids"] = [item_ids[0], item_ids[0], item_ids[2]]
            seal()
            drafts.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(verify_script.CliError, "duplicated"):
                verify_script._cohort_item_ids(drafts, dynamic=True)

    def test_verify_passes_exact_padding_stop_set_and_native_resume_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer_path, benchmark, drafts = _write_fixture(root)
            rows = verify_script._prepare_drafts(
                benchmark,
                drafts,
                _ByteTokenizer(tokenizer_path),
                max_draft_tokens=32,
            )
            args = verify_script._parser().parse_args(
                [
                    "--tokenizer-json",
                    str(tokenizer_path),
                    "--benchmark",
                    str(benchmark),
                    "--drafts-json",
                    str(drafts),
                    "--run-dir",
                    str(root / "run"),
                    "--cache-dir",
                    str(root / "cache"),
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                ]
            )
            captured: dict[str, object] = {}
            model = SimpleNamespace(
                config=SimpleNamespace(n_layers=4),
                pager=SimpleNamespace(
                    device=torch.device("cpu"),
                    compute_dtype=torch.float32,
                    source=SimpleNamespace(
                        budget=SimpleNamespace(total=0, limit=0),
                    ),
                ),
                prefill_hidden_shape=lambda batch, sequence: (batch, sequence, 12),
            )

            @contextmanager
            def runtime(_args, *, max_batch_size, max_seq_len):
                captured["runtime_shape"] = (max_batch_size, max_seq_len)
                yield model

            class FakeVerifier:
                def __init__(self, received_model, **kwargs):
                    captured["model"] = received_model
                    captured["constructor"] = kwargs

                def verify(self, prompts, candidates, **kwargs):
                    captured["prompts"] = prompts
                    captured["candidates"] = candidates
                    captured["verify"] = kwargs
                    return "verified"

            with mock.patch.object(
                verify_script,
                "load_resume",
                return_value=None,
            ) as load:
                result = verify_script._verify(
                    args,
                    rows,
                    runtime_factory=runtime,
                    verifier_factory=FakeVerifier,
                )

            self.assertEqual(result, "verified")
            verify_kwargs = captured["verify"]
            self.assertEqual(
                verify_kwargs["padding_token_id"],
                verify_script.END_OF_TEXT_TOKEN_ID,
            )
            self.assertEqual(
                verify_kwargs["eos_token_ids"],
                (verify_script.IM_END_TOKEN_ID, verify_script.END_OF_TEXT_TOKEN_ID),
            )
            expected_length = max(
                len(row.prompt_token_ids) + len(row.draft_token_ids) for row in rows
            )
            self.assertEqual(captured["runtime_shape"], (8, expected_length))
            self.assertEqual(
                load.call_args.kwargs["expected_shape"],
                (8, expected_length, 12),
            )

    def test_tokenizer_digest_and_resumed_source_budget_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            wrong = Path(temporary) / "tokenizer.json"
            wrong.write_text("{}", encoding="utf-8")
            args = verify_script._parser().parse_args(["--tokenizer-json", str(wrong)])
            with self.assertRaisesRegex(verify_script.CliError, "pinned"):
                verify_script._resolve_tokenizer_path(args)

        budget = SimpleNamespace(total=10, limit=999)
        model = SimpleNamespace(
            pager=SimpleNamespace(source=SimpleNamespace(budget=budget))
        )
        args = verify_script._parser().parse_args(["--source-budget-mb", "1"])
        report = verify_script._apply_cumulative_source_budget(
            args,
            model,
            prior_source_bytes=100,
        )
        self.assertEqual(report["total_limit_bytes"], 1024**2)
        self.assertEqual(budget.limit, 1024**2 - 100)
        with self.assertRaisesRegex(verify_script.CliError, "exhausted"):
            verify_script._apply_cumulative_source_budget(
                args,
                model,
                prior_source_bytes=1024**2 + 1,
            )

    def test_causal_runtime_fully_verifies_and_attaches_tensor_reader(self) -> None:
        args = verify_script._parser().parse_args(
            [
                "--causal-bundle",
                "/fixture/model.causal",
                "--device",
                "cpu",
                "--dtype",
                "float32",
            ]
        )
        args._require_official = False
        tensor_reader = object()
        source = SimpleNamespace(
            reader=SimpleNamespace(fetch_file=lambda _name: b"{}"),
        )
        mount = SimpleNamespace(
            source=source,
            tensor_reader=tensor_reader,
            close=mock.Mock(),
        )
        config = SimpleNamespace(n_layers=4)
        pager = SimpleNamespace(close=mock.Mock())
        model = SimpleNamespace(
            checkpoint_preflight=mock.Mock(),
            reset_state=mock.Mock(),
        )

        with (
            mock.patch.object(
                verify_script, "CausalWeightMount", return_value=mount
            ) as mount_type,
            mock.patch.object(
                verify_script,
                "verify_qwen38_causal_mount",
                return_value={"kind": "complete-causal-bundle/v1"},
            ) as verify,
            mock.patch.object(
                verify_script.Qwen38Config,
                "from_mapping",
                return_value=config,
            ),
            mock.patch.object(
                verify_script, "Qwen38WeightPager", return_value=pager
            ) as pager_type,
            mock.patch.object(
                verify_script, "StreamedQwen38", return_value=model
            ) as model_type,
        ):
            with verify_script._model_runtime(
                args, max_batch_size=8, max_seq_len=32
            ) as observed:
                self.assertIs(observed, model)

        mount_type.assert_called_once()
        verify.assert_called_once_with(mount, require_official_config=False)
        self.assertIs(
            pager_type.call_args.kwargs["causal_tensor_reader"], tensor_reader
        )
        self.assertIs(model_type.call_args.args[1], pager)
        self.assertEqual(args._source_verification["kind"], "complete-causal-bundle/v1")
        self.assertGreaterEqual(args._source_verification["seconds"], 0.0)
        model.checkpoint_preflight.assert_called_once_with()
        model.reset_state.assert_called_once_with(release=True)
        pager.close.assert_called_once_with()
        mount.close.assert_called_once_with()

    def test_causal_runtime_preflight_never_creates_an_hf_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "must-not-exist"
            args = verify_script._parser().parse_args(
                [
                    "--causal-bundle",
                    str(root),
                    "--cache-dir",
                    str(cache),
                    "--cache-budget-gb",
                    "10",
                ]
            )

            report = verify_script._cache_disk_preflight(args)

            self.assertEqual(report["cache_bytes"], 0)
            self.assertEqual(report["cache_growth_bytes"], 0)
            self.assertGreater(report["free_bytes"], 0)
            self.assertFalse(cache.exists())

    def test_verify_checkpoints_and_publishes_resumable_access_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer_path, benchmark, drafts = _write_fixture(root)
            rows = verify_script._prepare_drafts(
                benchmark,
                drafts,
                _ByteTokenizer(tokenizer_path),
                max_draft_tokens=32,
            )
            trace_path = root / "access-trace.json"
            args = verify_script._parser().parse_args(
                [
                    "--tokenizer-json",
                    str(tokenizer_path),
                    "--benchmark",
                    str(benchmark),
                    "--drafts-json",
                    str(drafts),
                    "--run-dir",
                    str(root / "run"),
                    "--cache-dir",
                    str(root / "cache"),
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                    "--access-trace-json",
                    str(trace_path),
                ]
            )
            attached: dict[str, AccessTraceRecorder] = {}
            sequence_length = max(
                len(row.prompt_token_ids) + len(row.draft_token_ids) for row in rows
            )
            fake_source = SimpleNamespace(
                budget=SimpleNamespace(total=0, limit=0),
                set_access_observer=lambda observer: attached.setdefault(
                    "recorder", observer
                ),
            )
            model = SimpleNamespace(
                config=SimpleNamespace(n_layers=4),
                pager=SimpleNamespace(
                    device=torch.device("cpu"),
                    compute_dtype=torch.float32,
                    source=fake_source,
                ),
                prefill_hidden_shape=lambda batch, sequence: (batch, sequence, 12),
            )

            @contextmanager
            def runtime(_args, *, max_batch_size, max_seq_len):
                del max_batch_size, max_seq_len
                yield model

            class FakeVerifier:
                def __init__(self, _model, **_kwargs):
                    pass

                def verify(self, _prompts, _candidates, **kwargs):
                    recorder = attached["recorder"]
                    recorder.observe(
                        AccessOperation(
                            repo_id=verify_script.OFFICIAL_REPO_ID,
                            revision=verify_script.OFFICIAL_REVISION,
                            inventory_fingerprint="a" * 64,
                            operation="raw_bytes",
                            operation_sequence=1,
                            thread_id=1,
                            thread_name="test",
                            leaves=(AccessLeaf("model.safetensors", 8, 16),),
                            source_requests=1,
                            source_bytes=16,
                            cache_hits=0,
                        )
                    )
                    kwargs["checkpoint"](
                        verify_script.DraftVerificationResumeState(
                            next_layer=1,
                            hidden=torch.zeros(8, sequence_length, 12),
                            layer_calls=1,
                            layer_retry_count=0,
                            source_body_bytes=16,
                            linear_calls=1,
                            seconds=0.1,
                            graft_applied=False,
                        )
                    )
                    return "verified"

            with mock.patch.object(verify_script, "load_resume", return_value=None):
                result = verify_script._verify(
                    args,
                    rows,
                    runtime_factory=runtime,
                    verifier_factory=FakeVerifier,
                )

            self.assertEqual(result, "verified")
            trace = AccessTrace.from_bytes(trace_path.read_bytes())
            self.assertEqual(len(trace.operations), 1)
            self.assertEqual(trace.sha256, args._access_trace_receipt["sha256"])
            self.assertEqual(args._access_trace_receipt["operations"], 1)
            checkpoint = verify_script._load_trace_checkpoint(
                trace_path,
                verify_script.build_resume_identity(
                    source_id=verify_script.OFFICIAL_REPO_ID,
                    source_revision=verify_script.OFFICIAL_REVISION,
                    prompt_token_ids=tuple(row.prompt_token_ids for row in rows),
                    draft_token_ids=tuple(row.draft_token_ids for row in rows),
                    execution_contract={
                        "device": "cpu",
                        "dtype": "float32",
                        "padding_token_id": verify_script.END_OF_TEXT_TOKEN_ID,
                        "eos_token_id": verify_script.IM_END_TOKEN_ID,
                        "accepted_eos_token_ids": [
                            verify_script.IM_END_TOKEN_ID,
                            verify_script.END_OF_TEXT_TOKEN_ID,
                        ],
                    },
                    graft_contract=None,
                ),
            )
            self.assertIsNotNone(checkpoint)
            self.assertEqual(checkpoint.next_layer, 1)
            self.assertEqual(checkpoint.source_body_bytes, 16)
            self.assertTrue(
                verify_script._discard_trace_checkpoint(trace_path, keep_public=True)
            )
            self.assertTrue(trace_path.is_file())
            self.assertFalse(
                verify_script._trace_checkpoint_manifest(trace_path).exists()
            )
            self.assertFalse(checkpoint.resume_path.exists())
            self.assertFalse(checkpoint.trace_path.exists())

    def test_trace_resume_pair_rejects_stale_identity_and_crash_mixing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace_path = root / "trace.json"
            identity = "d" * 64
            recorder = AccessTraceRecorder()

            def observe(sequence: int, offset: int) -> None:
                recorder.observe(
                    AccessOperation(
                        repo_id=verify_script.OFFICIAL_REPO_ID,
                        revision=verify_script.OFFICIAL_REVISION,
                        inventory_fingerprint="a" * 64,
                        operation="raw_bytes",
                        operation_sequence=sequence,
                        thread_id=1,
                        thread_name="test",
                        leaves=(AccessLeaf("model.safetensors", offset, 8),),
                        source_requests=1,
                        source_bytes=8,
                        cache_hits=0,
                    )
                )

            def state(layer: int, source_bytes: int):
                return verify_script.DraftVerificationResumeState(
                    next_layer=layer,
                    hidden=torch.zeros(1, 2, 3),
                    layer_calls=layer,
                    layer_retry_count=0,
                    source_body_bytes=source_bytes,
                    linear_calls=layer,
                    seconds=float(layer),
                    graft_applied=False,
                )

            observe(1, 0)
            verify_script._write_trace_checkpoint(
                recorder,
                trace_path,
                identity,
                state(1, 8),
                expected_shape=(1, 2, 3),
                expected_dtype="float32",
                n_layers=4,
                active_graft_layer=None,
            )
            with self.assertRaisesRegex(verify_script.CliError, "another run"):
                verify_script._load_trace_checkpoint(trace_path, "e" * 64)

            observe(2, 8)
            manifest = verify_script._trace_checkpoint_manifest(trace_path)
            original_atomic = verify_script._atomic_write_bytes

            def crash_before_commit(path, value):
                if Path(path) == manifest:
                    raise OSError("simulated crash before pair commit")
                return original_atomic(path, value)

            with (
                mock.patch.object(
                    verify_script,
                    "_atomic_write_bytes",
                    side_effect=crash_before_commit,
                ),
                self.assertRaisesRegex(OSError, "simulated crash"),
            ):
                verify_script._write_trace_checkpoint(
                    recorder,
                    trace_path,
                    identity,
                    state(2, 16),
                    expected_shape=(1, 2, 3),
                    expected_dtype="float32",
                    n_layers=4,
                    active_graft_layer=None,
                )

            recovered = verify_script._load_trace_checkpoint(trace_path, identity)
            self.assertIsNotNone(recovered)
            self.assertEqual(recovered.next_layer, 1)
            self.assertEqual(recovered.source_body_bytes, 8)
            self.assertEqual(len(recovered.trace.operations), 1)
            self.assertEqual(
                [
                    (leaf.shard, leaf.offset)
                    for leaf in recovered.trace.operations[0].leaves
                ],
                [("model.safetensors", 0)],
            )


if __name__ == "__main__":
    unittest.main()
