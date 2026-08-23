from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock
import urllib.error

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deepseek_v4_fertig_draft_verify.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "deepseek_v4_fertig_draft_verify", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import DeepSeek V4 draft verifier script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


draft_verify = _load_script()


class _Response:
    def __init__(self, document: dict, status: int = 200) -> None:
        self.document = document
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.document).encode("utf-8")


class _FakeTokenizer:
    def __init__(
        self,
        mapping: dict[str, tuple[int, ...]],
        *,
        pieces: dict[tuple[int, ...], tuple[str, ...]] | None = None,
    ) -> None:
        self.mapping = mapping
        self.pieces = {} if pieces is None else pieces

    def encode(self, text: str) -> tuple[int, ...]:
        return self.mapping[text]

    def decode(self, token_ids) -> str:
        for text, ids in self.mapping.items():
            if tuple(token_ids) == ids:
                return text
        raise AssertionError(token_ids)

    def token_pieces(self, token_ids) -> tuple[str, ...]:
        key = tuple(token_ids)
        if key in self.pieces:
            return self.pieces[key]
        if len(key) == 1:
            return (self.decode(key),)
        raise AssertionError(key)


class DeepSeekV4FertigDraftVerifyTests(unittest.TestCase):
    def test_defaults_are_lean_pinned_and_match_live_protocol(self) -> None:
        args = draft_verify._parser().parse_args([])
        self.assertEqual(
            draft_verify.OFFICIAL_SOURCE, "deepseek-ai/DeepSeek-V4-Flash-0731"
        )
        self.assertRegex(draft_verify.OFFICIAL_REVISION, r"^[0-9a-f]{40}$")
        self.assertEqual(args.max_draft_tokens, 128)
        self.assertEqual(args.cache_budget_gb, 6.0)
        self.assertEqual(args.source_budget_mb, 196608)
        self.assertEqual(args.device, "mps")
        self.assertEqual(args.dtype, "bfloat16")
        self.assertEqual(args.mode, "off")
        self.assertEqual(args.graft_layer, 21)
        self.assertEqual(args.graft_alpha, 0.01)
        self.assertEqual(args.layer_retries, 2)
        self.assertIsNone(args.item_ids_json)
        self.assertEqual(draft_verify._read_item_ids(None), draft_verify.FIXED_ITEM_IDS)
        item = draft_verify.SelectedItem("q", "question", "2", "prompt", (4, 5))
        payload = draft_verify._request_payload(item, max_tokens=128)
        self.assertEqual(payload["model"], "deepseek-v4-flash")
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["temperature"], 0)
        self.assertTrue(payload["logprobs"])
        self.assertEqual(
            payload["messages"][-1],
            {"role": "assistant", "content": "Answer:", "prefix": True},
        )

    def test_fixed_historical_cohort_renders_as_one_exact_83_token_batch(self) -> None:
        tokenizer = draft_verify.LocalTokenizer(draft_verify.DEFAULT_TOKENIZER)
        items = draft_verify._selected_items(draft_verify.DEFAULT_BENCHMARK, tokenizer)
        self.assertEqual(
            tuple(row.item_id for row in items), draft_verify.FIXED_ITEM_IDS
        )
        self.assertEqual(
            tuple(row.gold for row in items),
            ("2", "4", "14000", "1", "450", "3", "16", "2"),
        )
        self.assertEqual({len(row.prompt_token_ids) for row in items}, {83})
        self.assertTrue(
            all(
                row.prompt_text.startswith("<｜begin▁of▁sentence｜><｜User｜>")
                and row.prompt_text.endswith("<｜Assistant｜></think>Answer:")
                for row in items
            )
        )

    def test_selected_cohort_preserves_order_and_variable_prompt_lengths(self) -> None:
        rows = [
            {
                "item_id": "short",
                "status": "abstained",
                "question": "Short?",
                "gold": "#### 2",
            },
            {
                "item_id": "long",
                "status": "correct",
                "question": "A substantially longer question?",
                "gold": "#### 19",
            },
        ]
        short_prompt = "rendered:Short?Answer:"
        long_prompt = "rendered:A substantially longer question?Answer:"
        tokenizer = _FakeTokenizer(
            {
                short_prompt: (10, 11),
                long_prompt: tuple(range(20, 31)),
            }
        )
        with tempfile.TemporaryDirectory() as raw:
            benchmark = Path(raw) / "benchmark.json"
            benchmark.write_text(json.dumps({"items": rows}), encoding="utf-8")
            with mock.patch.object(
                draft_verify,
                "encode_user_prompt",
                side_effect=lambda question, **_kwargs: f"rendered:{question}",
            ):
                items = draft_verify._selected_items(
                    benchmark,
                    tokenizer,
                    ("long", "short"),
                )
        self.assertEqual(tuple(item.item_id for item in items), ("long", "short"))
        self.assertEqual(tuple(len(item.prompt_token_ids) for item in items), (11, 2))
        self.assertEqual(items[0].prompt_token_ids, tuple(range(20, 31)))
        self.assertEqual(items[1].prompt_token_ids, (10, 11))

    def test_item_id_manifest_rejects_invalid_or_ambiguous_cohorts(self) -> None:
        cases = (
            ({}, "root must be an array"),
            ([], "must not be empty"),
            (["a", "a"], "duplicate item ID"),
            (["a", 2], "entry 1"),
            ([" a"], "trimmed string"),
            ([""], "trimmed string"),
        )
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for index, (document, message) in enumerate(cases):
                with self.subTest(document=document):
                    path = root / f"manifest-{index}.json"
                    path.write_text(json.dumps(document), encoding="utf-8")
                    with self.assertRaisesRegex(draft_verify.CliError, message):
                        draft_verify._read_item_ids(path)
            malformed = root / "malformed.json"
            malformed.write_text("[", encoding="utf-8")
            with self.assertRaisesRegex(draft_verify.CliError, "cannot read"):
                draft_verify._read_item_ids(malformed)

            valid = root / "valid.json"
            valid.write_text('["b", "a"]', encoding="utf-8")
            self.assertEqual(draft_verify._read_item_ids(valid), ("b", "a"))

    def test_resume_identity_binds_the_resolved_device_not_auto(self) -> None:
        args = draft_verify._parser().parse_args(["--device", "auto"])
        cpu = draft_verify._resume_identity(
            args,
            ((1, 2),),
            ((3,),),
            resolved_device="cpu",
            resolved_dtype="bfloat16",
        )
        mps = draft_verify._resume_identity(
            args,
            ((1, 2),),
            ((3,),),
            resolved_device="mps",
            resolved_dtype="bfloat16",
        )
        self.assertNotEqual(cpu, mps)

    def test_resume_identity_binds_ordered_cohort_identity(self) -> None:
        args = draft_verify._parser().parse_args([])
        common = {
            "resolved_device": "mps",
            "resolved_dtype": "bfloat16",
        }
        first = draft_verify._resume_identity(
            args,
            ((1, 2),),
            ((3,),),
            cohort_identity="a" * 64,
            **common,
        )
        second = draft_verify._resume_identity(
            args,
            ((1, 2),),
            ((3,),),
            cohort_identity="b" * 64,
            **common,
        )
        self.assertNotEqual(first, second)

    def test_http_retry_request_and_api_token_contract(self) -> None:
        item = draft_verify.SelectedItem(
            "q", "How many?", "2", "rendered", tuple(range(83))
        )
        tokenizer = _FakeTokenizer(
            {" 2": (20, 21)},
            pieces={(20, 21): (" ", "2")},
        )
        response = {
            "model": "deepseek-v4-flash",
            "system_fingerprint": "fp_test",
            "choices": [
                {
                    "message": {"content": " 2"},
                    "finish_reason": "stop",
                    "logprobs": {
                        "content": [
                            {"token": " "},
                            {"token": "2"},
                        ]
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 83,
                "completion_tokens": 2,
                "total_tokens": 85,
            },
        }
        seen = []
        sleeps = []

        def opener(request, *, timeout):
            seen.append((request, timeout))
            if len(seen) == 1:
                raise urllib.error.URLError("temporary")
            return _Response(response)

        payload = draft_verify._request_payload(item, max_tokens=128)
        document = draft_verify._post_json(
            "https://api.deepseek.com/beta/chat/completions",
            payload,
            api_key="not-persisted",
            timeout=17,
            retries=1,
            opener=opener,
            sleeper=sleeps.append,
        )
        self.assertEqual(document, response)
        self.assertEqual(sleeps, [0.5])
        self.assertEqual(len(seen), 2)
        request = seen[-1][0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(seen[-1][1], 17)
        self.assertNotIn(b"not-persisted", request.data)
        self.assertEqual(json.loads(request.data), payload)

        parsed = draft_verify._parse_api_draft(
            item, tokenizer, document, max_draft_tokens=128
        )
        self.assertEqual(parsed["token_ids"], [20, 21])
        self.assertEqual(parsed["logprob_tokens"], [" ", "2"])
        self.assertEqual(parsed["logprob_token_count"], 2)
        self.assertEqual(parsed["response_model"], "deepseek-v4-flash")
        self.assertEqual(parsed["system_fingerprint"], "fp_test")
        cache = draft_verify._draft_document((item,), (parsed,), max_draft_tokens=128)
        self.assertNotIn("not-persisted", json.dumps(cache))

    def test_api_contract_rejects_truncation_and_token_count_drift(self) -> None:
        item = draft_verify.SelectedItem("q", "Q", "1", "P", (7, 8))
        tokenizer = _FakeTokenizer({" 1": (9,)})

        def response(
            *, finish="stop", logprobs=1, prompt_tokens=2, completion_tokens=1
        ):
            return {
                "model": "deepseek-v4-flash",
                "system_fingerprint": "fp_test",
                "choices": [
                    {
                        "message": {"content": " 1"},
                        "finish_reason": finish,
                        "logprobs": {"content": [{"token": " 1"}] * logprobs},
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": 3,
                },
            }

        with self.assertRaisesRegex(draft_verify.CliError, "did not reach EOS"):
            draft_verify._parse_api_draft(
                item, tokenizer, response(finish="length"), max_draft_tokens=128
            )
        with self.assertRaisesRegex(draft_verify.CliError, "completion token mismatch"):
            draft_verify._parse_api_draft(
                item, tokenizer, response(logprobs=2), max_draft_tokens=128
            )
        with self.assertRaisesRegex(draft_verify.CliError, "prompt token mismatch"):
            draft_verify._parse_api_draft(
                item,
                tokenizer,
                response(prompt_tokens=3),
                max_draft_tokens=128,
            )
        with self.assertRaisesRegex(draft_verify.CliError, "usage/logprob"):
            draft_verify._parse_api_draft(
                item,
                tokenizer,
                response(completion_tokens=2),
                max_draft_tokens=128,
            )
        with self.assertRaisesRegex(draft_verify.CliError, "total token"):
            document = response()
            document["usage"]["total_tokens"] = 0
            draft_verify._parse_api_draft(
                item,
                tokenizer,
                document,
                max_draft_tokens=128,
            )

        boundary_tokenizer = _FakeTokenizer(
            {"\n2": (201, 20)},
            pieces={(201, 20): ("\n", "2")},
        )
        boundary = response(logprobs=2, completion_tokens=2)
        boundary["choices"][0]["message"]["content"] = "\n2"
        boundary["choices"][0]["logprobs"]["content"] = [
            {"token": "\n"},
            {"token": "2"},
        ]
        boundary["usage"]["total_tokens"] = 4
        parsed = draft_verify._parse_api_draft(
            item,
            boundary_tokenizer,
            boundary,
            max_draft_tokens=128,
        )
        self.assertEqual(parsed["token_ids"], [201, 20])

    def test_valid_draft_cache_is_reused_without_key_or_fetch(self) -> None:
        tokenizer = draft_verify.LocalTokenizer(draft_verify.DEFAULT_TOKENIZER)
        items = draft_verify._selected_items(draft_verify.DEFAULT_BENCHMARK, tokenizer)
        drafts = []
        for item in items:
            content = f" The answer is {item.gold}."
            token_ids = tokenizer.encode(content)
            drafts.append(
                {
                    "item_id": item.item_id,
                    "response_model": "deepseek-v4-flash",
                    "system_fingerprint": "fp_test",
                    "content": content,
                    "token_ids": list(token_ids),
                    "logprob_tokens": list(tokenizer.token_pieces(token_ids)),
                    "logprob_token_count": len(token_ids),
                    "finish_reason": "stop",
                    "usage": {
                        "prompt_tokens": 83,
                        "completion_tokens": len(token_ids),
                        "total_tokens": 83 + len(token_ids),
                    },
                }
            )

        with tempfile.TemporaryDirectory() as raw:
            run_dir = Path(raw)
            args = draft_verify._parser().parse_args(["--run-dir", str(run_dir)])
            document = draft_verify._draft_document(items, drafts, max_draft_tokens=128)
            legacy_document = dict(document)
            legacy_document.pop("cohort")
            draft_verify._atomic_write_json(run_dir / "drafts.json", legacy_document)

            def fail_fetch(*_args):
                raise AssertionError("valid cache must avoid network")

            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(draft_verify, "_progress"),
            ):
                reused = draft_verify._prepare_drafts(
                    args, items, tokenizer, fetcher=fail_fetch
                )
            self.assertEqual(
                [row["item_id"] for row in reused], list(draft_verify.FIXED_ITEM_IDS)
            )

    def test_variable_draft_cache_requires_matching_cohort_identity(self) -> None:
        items = (
            draft_verify.SelectedItem("a", "A", "1", "pa", (4,)),
            draft_verify.SelectedItem("b", "B", "2", "pb", (5, 6, 7)),
        )
        tokenizer = _FakeTokenizer(
            {" 1": (101,), " 2": (102,)},
            pieces={(101,): (" 1",), (102,): (" 2",)},
        )
        drafts = tuple(
            {
                "response_model": "deepseek-v4-flash",
                "system_fingerprint": "fp_test",
                "content": f" {item.gold}",
                "token_ids": [100 + int(item.gold)],
                "logprob_tokens": [f" {item.gold}"],
                "logprob_token_count": 1,
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": len(item.prompt_token_ids),
                    "completion_tokens": 1,
                    "total_tokens": len(item.prompt_token_ids) + 1,
                },
            }
            for item in items
        )
        document = draft_verify._draft_document(
            items,
            drafts,
            max_draft_tokens=128,
        )
        validated = draft_verify._validate_draft_document(
            document,
            items,
            tokenizer,
            max_draft_tokens=128,
        )
        self.assertEqual(tuple(row["item_id"] for row in validated), ("a", "b"))

        missing = dict(document)
        missing.pop("cohort")
        with self.assertRaisesRegex(draft_verify.CliError, "cohort mismatch"):
            draft_verify._validate_draft_document(
                missing,
                items,
                tokenizer,
                max_draft_tokens=128,
            )

        stale = json.loads(json.dumps(document))
        stale["cohort"]["identity"] = "0" * 64
        with self.assertRaisesRegex(draft_verify.CliError, "cohort mismatch"):
            draft_verify._validate_draft_document(
                stale,
                items,
                tokenizer,
                max_draft_tokens=128,
            )

    def test_atomic_json_replaces_complete_documents(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "nested" / "result.json"
            draft_verify._atomic_write_json(output, {"status": "first"})
            draft_verify._atomic_write_json(
                output, {"status": "complete", "rows": [1, 2]}
            )
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"status": "complete", "rows": [1, 2]},
            )
            self.assertTrue(output.read_bytes().endswith(b"\n"))
            self.assertEqual(
                [path for path in output.parent.iterdir() if path.name.startswith(".")],
                [],
            )

    def test_resume_file_is_one_atomic_rolling_hidden_state(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            args = draft_verify._parser().parse_args(["--run-dir", raw])
            path = draft_verify._resume_path(args)
            identity = draft_verify._resume_identity(
                args,
                ((1, 2),),
                ((3,),),
                resolved_device="cpu",
                resolved_dtype="bfloat16",
            )
            first = draft_verify.DraftVerificationResumeState(
                next_layer=1,
                hidden=torch.ones((1, 3, 1, 2), dtype=torch.bfloat16),
                layer_calls=1,
                layer_retry_count=0,
                source_body_bytes=10,
                linear_calls=2,
                seconds=1.5,
                graft_applied=False,
            )
            second = draft_verify.DraftVerificationResumeState(
                next_layer=2,
                hidden=torch.full((1, 3, 1, 2), 2.0, dtype=torch.bfloat16),
                layer_calls=2,
                layer_retry_count=1,
                source_body_bytes=20,
                linear_calls=4,
                seconds=3.0,
                graft_applied=False,
            )
            draft_verify._write_resume(path, identity, first)
            draft_verify._write_resume(path, identity, second)

            self.assertEqual([entry.name for entry in root.iterdir()], [path.name])
            bound_identity = draft_verify._resume_identity(
                args,
                ((1, 2),),
                ((3,),),
                resolved_device="cpu",
                resolved_dtype="bfloat16",
                cohort_identity="a" * 64,
            )
            loaded = draft_verify._load_resume(
                path,
                bound_identity,
                legacy_identity=identity,
                expected_shape=(1, 3, 1, 2),
                expected_dtype="bfloat16",
                n_layers=43,
            )
            assert loaded is not None
            self.assertEqual(loaded.next_layer, 2)
            self.assertEqual(loaded.layer_retry_count, 1)
            self.assertTrue(torch.equal(loaded.hidden, second.hidden))
            with self.assertRaisesRegex(draft_verify.CliError, "another run"):
                draft_verify._load_resume(
                    path,
                    "0" * 64,
                    expected_shape=(1, 3, 1, 2),
                    expected_dtype="bfloat16",
                    n_layers=43,
                )
            self.assertTrue(draft_verify._clear_resume(path))
            self.assertFalse(path.exists())

    def test_resume_disk_preflight_fails_before_the_layer_loop(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            args = draft_verify._parser().parse_args(
                [
                    "--run-dir",
                    raw,
                    "--cache-dir",
                    str(root / "cache"),
                    "--no-cache",
                ]
            )
            path = draft_verify._resume_path(args)
            with (
                mock.patch.object(
                    draft_verify.shutil,
                    "disk_usage",
                    return_value=SimpleNamespace(free=1),
                ),
                self.assertRaisesRegex(draft_verify.CliError, "insufficient free disk"),
            ):
                draft_verify._preflight_resume_disk(
                    args,
                    path,
                    hidden_bytes=1024,
                    current_cache_bytes=0,
                )

            enough = 1024**3
            with mock.patch.object(
                draft_verify.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=enough),
            ):
                result = draft_verify._preflight_resume_disk(
                    args,
                    path,
                    hidden_bytes=1024,
                    current_cache_bytes=0,
                )
            self.assertLess(result["required_bytes"], enough)

    def test_draft_only_writes_result_without_constructing_local_runtime(self) -> None:
        item = draft_verify.SelectedItem("a", "A", "2", "pa", (4,))
        draft = {
            "content": " 2",
            "token_ids": [20],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }
        with tempfile.TemporaryDirectory() as raw:
            args = draft_verify._parser().parse_args(["--run-dir", raw, "--draft-only"])
            with (
                mock.patch.object(
                    draft_verify, "_selected_items", return_value=(item,)
                ),
                mock.patch.object(
                    draft_verify, "_prepare_drafts", return_value=(draft,)
                ),
                mock.patch.object(
                    draft_verify,
                    "_verify_locally",
                    side_effect=AssertionError("draft-only loaded the model"),
                ),
            ):
                result, result_path = draft_verify.run(args)
            self.assertEqual(result["status"], "drafts_ready")
            self.assertEqual(result["summary"]["api_accuracy"], 1.0)
            self.assertEqual(result_path.name, "result-drafts.json")
            self.assertEqual(json.loads(result_path.read_text()), result)

    def test_run_selects_the_ordered_manifest_before_draft_preparation(self) -> None:
        items = (
            draft_verify.SelectedItem("b", "B", "2", "pb", (5, 6)),
            draft_verify.SelectedItem("a", "A", "1", "pa", (4,)),
        )
        drafts = (
            {
                "content": " 2",
                "token_ids": [20],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                },
            },
            {
                "content": " 1",
                "token_ids": [10],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )
        tokenizer = object()
        with tempfile.TemporaryDirectory() as raw:
            manifest = Path(raw) / "ids.json"
            manifest.write_text('["b", "a"]', encoding="utf-8")
            args = draft_verify._parser().parse_args(
                [
                    "--run-dir",
                    raw,
                    "--draft-only",
                    "--item-ids-json",
                    str(manifest),
                ]
            )
            with (
                mock.patch.object(
                    draft_verify,
                    "LocalTokenizer",
                    return_value=tokenizer,
                ),
                mock.patch.object(
                    draft_verify,
                    "_selected_items",
                    return_value=items,
                ) as select,
                mock.patch.object(
                    draft_verify,
                    "_prepare_drafts",
                    return_value=drafts,
                ),
            ):
                result, _ = draft_verify.run(args)
        select.assert_called_once_with(args.benchmark, tokenizer, ("b", "a"))
        self.assertEqual(result["summary"]["items"], 2)
        self.assertEqual(result["summary"]["api_correct"], 2)

    def test_model_runtime_uses_selected_cohort_as_max_batch_size(self) -> None:
        args = draft_verify._parser().parse_args(["--no-cache"])
        source = mock.Mock()
        source.reader.fetch_file.return_value = b"{}"
        config = SimpleNamespace()
        pager = mock.Mock()
        model = mock.Mock()
        with (
            mock.patch.object(draft_verify, "Streamer", return_value=source),
            mock.patch.object(
                draft_verify.DeepSeekV4Config,
                "from_mapping",
                return_value=config,
            ),
            mock.patch.object(
                draft_verify,
                "DeepSeekWeightPager",
                return_value=pager,
            ),
            mock.patch.object(
                draft_verify,
                "StreamedDeepSeekV4",
                return_value=model,
            ) as runtime_type,
        ):
            with draft_verify._model_runtime(
                args,
                max_seq_len=177,
                max_batch_size=23,
            ) as yielded:
                self.assertIs(yielded, model)
        runtime_type.assert_called_once_with(
            config,
            pager,
            graft=None,
            graft_layer=None,
            max_batch_size=23,
            max_seq_len=177,
        )
        model.reset_state.assert_called_once_with(release=True)
        pager.close.assert_called_once_with()
        source.close.assert_called_once_with()

    def test_fake_runtime_verifier_boundary_is_one_right_padded_pass(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        args = draft_verify._parser().parse_args(
            [
                "--max-draft-tokens",
                "12",
                "--head-block-rows",
                "64",
                "--run-dir",
                temporary.name,
                "--no-cache",
            ]
        )
        items = (
            draft_verify.SelectedItem("a", "A", "1", "pa", (4, 5, 6)),
            draft_verify.SelectedItem("b", "B", "2", "pb", (7, 8)),
        )
        drafts = (
            {"token_ids": [9, 10]},
            {"token_ids": [11]},
        )
        calls = []
        fake_model = SimpleNamespace(
            config=SimpleNamespace(hc_mult=1, dim=2, n_layers=43),
            pager=SimpleNamespace(
                compute_dtype=torch.bfloat16,
                device=torch.device("cpu"),
                source=SimpleNamespace(metrics=lambda: {"cache_bytes": 0}),
            ),
        )
        sentinel = object()

        @contextmanager
        def runtime_factory(received_args, *, max_seq_len, max_batch_size):
            calls.append(("runtime", received_args, max_seq_len, max_batch_size))
            yield fake_model

        class FakeVerifier:
            def __init__(self, model, *, layer_retries):
                calls.append(("verifier", model, layer_retries))

            def verify(self, prompts, draft_ids, **kwargs):
                calls.append(("verify", prompts, draft_ids, kwargs))
                kwargs["progress"](
                    {"event": "layer_complete", "layer": 1, "layers": 43}
                )
                kwargs["head_progress"]({"rows_done": 1024, "vocab_rows": 100000})
                return sentinel

        with mock.patch.object(draft_verify, "_progress") as progress:
            report = draft_verify._verify_locally(
                args,
                items,
                drafts,
                runtime_factory=runtime_factory,
                verifier_factory=FakeVerifier,
            )
        self.assertIs(report, sentinel)
        self.assertEqual(calls[0], ("runtime", args, 5, 2))
        self.assertEqual(calls[1], ("verifier", fake_model, 2))
        self.assertEqual(calls[2][1], ((4, 5, 6), (7, 8)))
        self.assertEqual(calls[2][2], ((9, 10), (11,)))
        kwargs = calls[2][3]
        self.assertEqual(kwargs["eos_token_id"], 1)
        self.assertEqual(kwargs["padding_token_id"], 1)
        self.assertEqual(kwargs["max_draft_tokens"], 12)
        self.assertEqual(kwargs["head_block_rows"], 64)
        self.assertIsNone(kwargs["resume_state"])
        self.assertTrue(callable(kwargs["checkpoint"]))
        self.assertTrue(callable(kwargs["progress"]))
        self.assertTrue(callable(kwargs["head_progress"]))
        progress.assert_any_call("local_layer_complete", layer=1, layers=43)
        progress.assert_any_call("head_progress", rows_done=1024, vocab_rows=100000)

    def test_result_reports_content_eos_accuracy_and_traffic(self) -> None:
        args = draft_verify._parser().parse_args([])
        items = (
            draft_verify.SelectedItem("a", "A", "2", "pa", (4,)),
            draft_verify.SelectedItem("b", "B", "3", "pb", (5,)),
        )
        drafts = (
            {
                "content": " 2",
                "token_ids": [20],
                "response_model": "deepseek-v4-flash",
                "system_fingerprint": "fp_test",
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
            {
                "content": " 9",
                "token_ids": [90],
                "response_model": "deepseek-v4-flash",
                "system_fingerprint": "fp_test",
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

        class Row(SimpleNamespace):
            def to_dict(self):
                return dict(self.__dict__)

        class Evidence:
            def to_dict(self):
                return {
                    "source_body_bytes": 123,
                    "linear_calls": 44,
                    "layer_calls": 43,
                    "layer_retry_count": 1,
                    "head_scans": 1,
                }

        report = SimpleNamespace(
            rows=(
                Row(
                    row=0,
                    draft_token_ids=(20,),
                    draft_verified=True,
                    eos_verified=True,
                    fully_verified=True,
                ),
                Row(
                    row=1,
                    draft_token_ids=(90,),
                    draft_verified=False,
                    eos_verified=True,
                    fully_verified=False,
                ),
            ),
            evidence=Evidence(),
            all_verified=False,
        )
        tokenizer = _FakeTokenizer({" 2": (20,), " 9": (90,)})
        result = draft_verify._verification_result(
            args, items, drafts, tokenizer, report
        )
        self.assertEqual(result["summary"]["content_verified"], 1)
        self.assertEqual(result["summary"]["eos_verified"], 1)
        self.assertEqual(result["summary"]["local_correct"], 1)
        self.assertEqual(result["summary"]["local_verified_total"], 1)
        self.assertEqual(result["summary"]["local_verified_subset_accuracy"], 1.0)
        self.assertEqual(result["summary"]["local_end_to_end_accuracy"], 0.5)
        self.assertEqual(result["summary"]["api_accuracy"], 0.5)
        self.assertEqual(result["traffic"]["local"]["source_body_bytes"], 123)
        self.assertEqual(result["traffic"]["api"]["total_tokens"], 4)
        self.assertIsNone(result["items"][1]["locally_fully_verified_content"])
        self.assertIsNone(result["items"][1]["eos_verified"])


if __name__ == "__main__":
    unittest.main()
