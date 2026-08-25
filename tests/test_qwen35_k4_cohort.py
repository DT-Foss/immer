from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from immer.runtimes.qwen3_8 import (
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
    QWEN35_DRAFTER_REPO_ID,
    QWEN35_DRAFTER_REVISION,
    Qwen38Tokenizer,
)

from test_qwen3_8_model import (
    _native_tiny_config,
    _tiny_config_mapping,
    _tiny_tied_config,
)
from test_qwen35_live_k4_smoke import _bundle, _tokenizer


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen35_k4_cohort.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("qwen35_k4_cohort", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import Qwen3.5 K4 cohort")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cohort = _load_script()


class _LengthTokenizer:
    @staticmethod
    def render_no_thinking_prompt(_system: str, question: str) -> str:
        return question

    @staticmethod
    def encode(rendered: str) -> tuple[int, ...]:
        length = int(rendered.rsplit(" ", 1)[-1])
        return tuple(range(length))


def _selected_items(tokenizer: Qwen38Tokenizer) -> list[dict[str, object]]:
    schedule = (("A", "B"), ("B", "A"), ("B", "A"), ("A", "B"))
    bins = (0, 3, 1, 2)
    rows = []
    for index, order in enumerate(schedule):
        question = str(index + 1)
        rendered = tokenizer.render_no_thinking_prompt("", question)
        tokens = list(tokenizer.encode(rendered))
        rows.append(
            {
                "execution_index": index,
                "item_id": f"tiny-{index}",
                "normalized_question": question,
                "normalized_question_sha256": hashlib.sha256(
                    question.encode("utf-8")
                ).hexdigest(),
                "offset": index,
                "rendered_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                "rendered_token_count": len(tokens),
                "rendered_token_ids": tokens,
                "schedule": list(order),
                "selection_sha256": cohort._selection_hash(question),
                "stratum": ("short", "long", "medium-short", "medium-long")[index],
                "stratum_bin_index": bins[index],
            }
        )
    return rows


def _contract() -> dict[str, object]:
    return {
        "attention_mode": "native-crsa",
        "batch_size": 1,
        "compute_dtype": "bfloat16",
        "device": "cpu",
        "eos_enabled": False,
        "execution_regime": cohort.EXECUTION_REGIME,
        "max_new_tokens": 4,
        "remote_io": False,
        "routes": {"A": "2xK2", "B": "K4"},
        "system_prompt": "",
    }


def _selection(items: list[dict[str, object]]) -> dict[str, object]:
    by_bin = {int(row["stratum_bin_index"]): int(row["offset"]) for row in items}
    return {
        "bin_size": 16,
        "domain": cohort.SELECTION_DOMAIN,
        "execution_bins": list(cohort._EXECUTION_BINS),
        "method": "NFC+trim/render/sort(token_count,selection_sha256)/four-bins/min-sha256/v1",
        "schedule": [list(row) for row in cohort._SCHEDULE],
        "strata": [
            {
                "bin_index": index,
                "label": cohort._STRATA[index],
                "rank_end_exclusive": (index + 1) * 16,
                "rank_start": index * 16,
                "selected_offset": by_bin[index],
                "token_count_max": 8,
                "token_count_min": 1,
            }
            for index in range(4)
        ],
    }


def _source() -> dict[str, object]:
    return {
        "item_count": 64,
        "question_projection_sha256": "1" * 64,
        "raw_file_sha256": "2" * 64,
        "raw_file_size_bytes": 1,
    }


def _arm(route: str, *, hidden: str = "h", cursor: int = 9, draft_cursor: int = 9):
    suffix = list(range(cursor - draft_cursor))
    return {
        "contract": {"route_id": route},
        "cost": {"combined_logical_bytes": 10},
        "prompt": {"rendered_token_count": 7},
        "drafter": {
            "alignment": {
                "committed_cursor": draft_cursor,
                "target_cursor": cursor,
                "target_only_suffix_ids": suffix,
            },
            "state": {"cursor": draft_cursor, "sha256": f"draft-{draft_cursor}"},
        },
        "target": {
            "cursor": cursor,
            "native_crsa": {"events": [1], "sha256": "native"},
            "positions": [
                {
                    "hidden": {"sha256": hidden},
                    "input_token_id": 3,
                    "native_crsa": {"query_start": 8},
                    "position": 8,
                }
            ],
            "state": {"cursor": cursor, "sha256": "target"},
            "state_tensor_sha256": {"layer.0.key": "k"},
        },
        "timing": {"arm_wall_seconds": 1.0},
        "tokens": {"generated_token_ids": [3]},
    }


class Qwen35K4CohortSelectorTests(unittest.TestCase):
    def test_question_only_selector_bins_and_abba_order_are_deterministic(self) -> None:
        items = [
            {
                "item_id": f"id-{index}",
                "question": f"question {index + 1}",
                "answer": index,
            }
            for index in range(64)
        ]
        selected, strata = cohort._select_cohort(items, _LengthTokenizer())
        ordered = sorted(
            range(64),
            key=lambda index: (
                index + 1,
                cohort._selection_hash(f"question {index + 1}"),
            ),
        )
        bins = [ordered[index : index + 16] for index in range(0, 64, 16)]
        winners = [
            min(rows, key=lambda index: cohort._selection_hash(f"question {index + 1}"))
            for rows in bins
        ]
        expected = [winners[index] for index in (0, 3, 1, 2)]
        self.assertEqual([row["offset"] for row in selected], expected)
        self.assertEqual(
            [row["schedule"] for row in selected],
            [["A", "B"], ["B", "A"], ["B", "A"], ["A", "B"]],
        )
        self.assertEqual(
            [row["label"] for row in strata],
            ["short", "medium-short", "medium-long", "long"],
        )

        changed = copy.deepcopy(items)
        for row in changed:
            row["answer"] = "mutated-and-ignored"
            row["prediction"] = "also-ignored"
        repeated, _ = cohort._select_cohort(changed, _LengthTokenizer())
        self.assertEqual(selected, repeated)

    def test_nfc_trim_is_part_of_selection_domain(self) -> None:
        self.assertEqual(cohort._normal_question("  Cafe\u0301  "), "Caf\u00e9")
        self.assertEqual(
            cohort._selection_hash(cohort._normal_question("  Cafe\u0301  ")),
            cohort._selection_hash("Caf\u00e9"),
        )

    def test_tamper_and_reseal_do_not_bypass_semantic_validation(self) -> None:
        rows = []
        for index, order in enumerate(cohort._SCHEDULE):
            question = f"q {index + 1}"
            rendered = Qwen38Tokenizer.render_no_thinking_prompt("", question)
            rows.append(
                {
                    "execution_index": index,
                    "item_id": f"id-{index}",
                    "normalized_question": question,
                    "normalized_question_sha256": hashlib.sha256(
                        question.encode("utf-8")
                    ).hexdigest(),
                    "offset": index,
                    "rendered_sha256": hashlib.sha256(
                        rendered.encode("utf-8")
                    ).hexdigest(),
                    "rendered_token_count": 1,
                    "rendered_token_ids": [index],
                    "schedule": list(order),
                    "selection_sha256": cohort._selection_hash(question),
                    "stratum": ("short", "long", "medium-short", "medium-long")[index],
                    "stratum_bin_index": cohort._EXECUTION_BINS[index],
                }
            )
        tokenizer = {
            "kind": "local-tokenizers-json/v1",
            "sha256": "3" * 64,
            "size_bytes": 1,
            "vocabulary_sha256": "4" * 64,
            "vocabulary_size": 16,
        }
        document = cohort._seal(
            {
                "code_revision": "5" * 40,
                "contract": _contract(),
                "holdout_accessed": False,
                "items": rows,
                "path_allowlist": ["/tmp/dev64.json", "/tmp/tokenizer.json"],
                "schema": cohort.INPUT_SCHEMA,
                "selection": _selection(rows),
                "source": _source(),
                "tokenizer": tokenizer,
            }
        )
        cohort._validate_input(document, require_frozen=False)
        tampered = copy.deepcopy(document)
        tampered["items"][0]["rendered_token_ids"] = [99]
        with self.assertRaisesRegex(cohort.CohortError, "seal"):
            cohort._validate_input(tampered, require_frozen=False)
        tampered.pop("sha256")
        tampered["items"][0]["selection_sha256"] = "0" * 64
        resealed = cohort._seal(tampered)
        with self.assertRaisesRegex(cohort.CohortError, "selection hash"):
            cohort._validate_input(resealed, require_frozen=False)
        missing = copy.deepcopy(document)
        missing.pop("sha256")
        missing.pop("tokenizer")
        with self.assertRaisesRegex(cohort.CohortError, "fields"):
            cohort._validate_input(cohort._seal(missing), require_frozen=False)

    def test_atomic_nonoverwrite_output_containment_and_holdout_exclusion(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            output = root / "result.json"
            cohort._atomic_new_json(output, {"a": 1})
            with self.assertRaisesRegex(cohort.CohortError, "overwrite"):
                cohort._atomic_new_json(output, {"a": 2})
            protected = root / "model"
            protected.mkdir()
            with self.assertRaisesRegex(cohort.CohortError, "overlaps"):
                cohort._guard_output_dir(protected / "result", (protected,))
            with self.assertRaisesRegex(cohort.CohortError, "Holdout"):
                cohort._assert_no_holdout_paths((root / "Holdout64" / "x.json",))

    def test_one_authentication_guard_rejects_reopen(self) -> None:
        pair = cohort._SingleMountPair()
        pair.counts["target"] = 1
        with self.assertRaisesRegex(cohort.CohortError, "exactly once"):
            pair.open_role("target")

    def test_actual_double_verification_and_preflight_are_rejected(self) -> None:
        pair = cohort._SingleMountPair()

        def double_verify(**_kwargs):
            cohort.base.verify_qwen38_causal_mount(None)
            cohort.base.verify_qwen38_causal_mount(None)

        with (
            mock.patch.object(
                cohort.base, "verify_qwen38_causal_mount", return_value={}
            ),
            mock.patch.object(
                cohort.base.StreamedQwen38, "checkpoint_preflight", return_value={}
            ),
            mock.patch.object(cohort.base, "_open_model", side_effect=double_verify),
        ):
            with self.assertRaisesRegex(cohort.CohortError, "more than once"):
                pair.open_role("target")

        pair = cohort._SingleMountPair()

        def double_preflight(**_kwargs):
            cohort.base.verify_qwen38_causal_mount(None)
            cohort.base.StreamedQwen38.checkpoint_preflight(object())
            cohort.base.StreamedQwen38.checkpoint_preflight(object())

        with (
            mock.patch.object(
                cohort.base, "verify_qwen38_causal_mount", return_value={}
            ),
            mock.patch.object(
                cohort.base.StreamedQwen38, "checkpoint_preflight", return_value={}
            ),
            mock.patch.object(cohort.base, "_open_model", side_effect=double_preflight),
        ):
            with self.assertRaisesRegex(cohort.CohortError, "more than once"):
                pair.open_role("draft")

    def test_pair_parity_detects_hidden_mismatch_and_accepts_exact_tail_prefix(
        self,
    ) -> None:
        self.assertTrue(cohort._pair_parity(_arm("A"), _arm("B"))["positive"])
        self.assertFalse(
            cohort._pair_parity(_arm("A"), _arm("B", hidden="bad"))["positive"]
        )
        tail = cohort._pair_parity(_arm("A", draft_cursor=8), _arm("B", draft_cursor=7))
        self.assertTrue(tail["drafter_prefix_valid"])
        self.assertIsNone(tail["drafter_state_equal_when_aligned"])
        self.assertTrue(tail["positive"])
        prompt_tail = cohort._pair_parity(
            _arm("A", draft_cursor=6), _arm("B", draft_cursor=7)
        )
        self.assertFalse(prompt_tail["drafter_prefix_valid"])
        self.assertFalse(prompt_tail["positive"])


class Qwen35K4CohortTinyBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(
            prefix=".qwen35-k4-cohort-test-", dir=ROOT
        )
        cls.root = Path(cls.temporary.name)
        target_config = _native_tiny_config()
        target_mapping = _tiny_config_mapping()
        target_mapping["num_hidden_layers"] = 28
        target_mapping["layer_types"] = [
            "full_attention" if (layer + 1) % 4 == 0 else "linear_attention"
            for layer in range(28)
        ]
        target_mapping["num_attention_heads"] = 24
        target_mapping["num_key_value_heads"] = 4
        cls.target = _bundle(
            cls.root,
            "target",
            config=target_config,
            config_mapping=target_mapping,
            repo_id=OFFICIAL_REPO_ID,
            revision=OFFICIAL_REVISION,
        )
        draft_config = _tiny_tied_config()
        draft_mapping = _tiny_config_mapping()
        draft_mapping["tie_word_embeddings"] = True
        cls.draft = _bundle(
            cls.root,
            "draft",
            config=draft_config,
            config_mapping=draft_mapping,
            repo_id=QWEN35_DRAFTER_REPO_ID,
            revision=QWEN35_DRAFTER_REVISION,
        )
        cls.tokenizer_path = cls.root / "tokenizer.json"
        _tokenizer(cls.tokenizer_path)
        tokenizer, receipt = cohort._tokenizer_receipt(
            cls.tokenizer_path, require_official=False
        )
        selected = _selected_items(tokenizer)
        cls.input_document = cohort._seal(
            {
                "code_revision": cohort._git_revision(require_clean=False),
                "contract": _contract(),
                "holdout_accessed": False,
                "items": selected,
                "path_allowlist": [
                    str(cls.root / "dev64.json"),
                    str(cls.tokenizer_path),
                ],
                "schema": cohort.INPUT_SCHEMA,
                "selection": _selection(selected),
                "source": _source(),
                "tokenizer": receipt,
            }
        )
        cls.input_path = cls.root / "input.json"
        cls.input_path.write_bytes(cohort._canonical(cls.input_document) + b"\n")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_tiny_causal_bundles_run_full_abba_with_one_verification_each(self) -> None:
        output = self.root / "session"
        original = cohort.base.verify_qwen38_causal_mount
        with mock.patch.object(
            cohort.base, "verify_qwen38_causal_mount", wraps=original
        ) as verified:
            result, result_path = cohort.execute(
                input_path=self.input_path,
                target_path=self.target,
                draft_path=self.draft,
                target_tokenizer_path=self.tokenizer_path,
                output_dir=output,
                target_source_budget_mb=100,
                draft_source_budget_mb=100,
                max_resident_mb=2,
                max_context_tokens=64,
                head_block_rows=8,
                require_production_profile=False,
                require_official_target=False,
                require_frozen=False,
                require_clean=False,
            )
        self.assertEqual(verified.call_count, 2)
        self.assertEqual(result["correctness_status"], "positive")
        self.assertEqual(len(result["arm_sha256"]), 8)
        self.assertTrue(all(row["parity"]["positive"] for row in result["pairs"]))
        self.assertTrue(result_path.is_file())
        mount = json.loads((output / "mount.json").read_text(encoding="utf-8"))
        self.assertEqual(mount["authentication"]["target_verification_count"], 1)
        self.assertEqual(mount["authentication"]["draft_verification_count"], 1)
        self.assertEqual(mount["authentication"]["target_preflight_count"], 1)
        self.assertEqual(mount["authentication"]["draft_preflight_count"], 1)
        self.assertTrue((output / "transport.json").is_file())
        self.assertFalse((output / "abort.json").exists())

        # Resealing both the nested continuation state and the outer arm must
        # not turn a semantic prompt-tail corruption into valid evidence.
        arm_path = sorted(output.glob("arm-*.json"))[0]
        tampered = json.loads(arm_path.read_text(encoding="utf-8"))
        draft_state = tampered["drafter"]["state"]
        draft_state.pop("sha256")
        prompt_count = tampered["prompt"]["rendered_token_count"]
        draft_state["cursor"] = prompt_count - 1
        draft_state["sha256"] = cohort._sha256(draft_state)
        alignment = tampered["drafter"]["alignment"]
        alignment["committed_cursor"] = prompt_count - 1
        tampered.pop("sha256")
        tampered = cohort._seal(tampered)
        with self.assertRaisesRegex(cohort.CohortError, "suffix"):
            cohort._validate_arm(tampered)

    def test_failed_execution_writes_abort_without_retry(self) -> None:
        output = self.root / "abort-session"
        with mock.patch.object(
            cohort, "_run_arm", side_effect=RuntimeError("synthetic arm failure")
        ) as run_arm:
            with self.assertRaisesRegex(RuntimeError, "synthetic arm failure"):
                cohort.execute(
                    input_path=self.input_path,
                    target_path=self.target,
                    draft_path=self.draft,
                    target_tokenizer_path=self.tokenizer_path,
                    output_dir=output,
                    target_source_budget_mb=100,
                    draft_source_budget_mb=100,
                    max_resident_mb=2,
                    max_context_tokens=64,
                    head_block_rows=8,
                    require_production_profile=False,
                    require_official_target=False,
                    require_frozen=False,
                    require_clean=False,
                )
        self.assertEqual(run_arm.call_count, 1)
        abort = json.loads((output / "abort.json").read_text(encoding="utf-8"))
        self.assertFalse(abort["retry_performed"])
        self.assertEqual(abort["completed_arm_sha256"], [])


if __name__ == "__main__":
    unittest.main()
