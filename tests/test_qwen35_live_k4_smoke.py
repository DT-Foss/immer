from __future__ import annotations

from contextlib import redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from immer.knowledge import Streamer
from immer.runtimes.qwen3_8 import (
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
    QWEN35_DRAFTER_REPO_ID,
    QWEN35_DRAFTER_REVISION,
)

from test_qwen3_8_model import (
    _native_tiny_config,
    _tiny_config_mapping,
    _tiny_tied_config,
    _tiny_weights,
)
from test_qwen38_causal_bundle import bundle_script


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen35_live_k4_smoke.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("qwen35_live_k4_smoke", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import Qwen3.5 live K4 smoke")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


smoke = _load_script()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle(
    root: Path,
    name: str,
    *,
    config,
    config_mapping: dict[str, object],
    repo_id: str,
    revision: str,
) -> Path:
    source_root = root / f"{name}-source"
    source_root.mkdir()
    weights = _tiny_weights(config)
    output_name = (
        "model.language_model.embed_tokens.weight"
        if config.tie_word_embeddings
        else "lm_head.weight"
    )
    weights[output_name].zero_()
    save_file(weights, source_root / "model.safetensors")
    source_root.joinpath("config.json").write_text(
        json.dumps(config_mapping, separators=(",", ":")), encoding="utf-8"
    )
    source = Streamer.from_local(
        source_root,
        repo_id=repo_id,
        revision=revision,
        budget_mb=100,
        use_cache=False,
    )
    try:
        inventory = json.loads(json.dumps(source.inventory()))
    finally:
        source.close()
    digest = _sha256_file(source_root / "model.safetensors")
    inventory["shards"][0].update(
        {
            "cas_url_hash": digest,
            "etag": f'"{digest}"',
            "linked_etag": digest,
            "payload_sha256": digest,
            "repo_commit": revision,
        }
    )
    fingerprint = Streamer._source_fingerprint(inventory)
    inventory_document = {
        "inventory": inventory,
        "inventory_sha256": hashlib.sha256(
            bundle_script._canonical(inventory)
        ).hexdigest(),
        "repo_id": repo_id,
        "revision": revision,
        "schema": "immer.tensor-inventory-cache/v1",
        "source_fingerprint": fingerprint,
    }
    inventory_path = root / f"{name}-inventory.json"
    inventory_path.write_bytes(bundle_script._canonical(inventory_document) + b"\n")
    output = root / f"{name}.causal"
    bundle_script.build_bundle(
        source_root,
        inventory_path,
        output,
        repo_id=repo_id,
        revision=revision,
        expected_fingerprint=fingerprint,
        require_official=False,
        require_remote_hashes=True,
    )
    return output


def _tokenizer(path: Path) -> None:
    from tokenizers import AddedToken, Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit

    vocabulary = {"[UNK]": 0, **{str(index): index for index in range(1, 29)}}
    tokenizer = Tokenizer(WordLevel(vocab=vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.add_special_tokens(
        [
            AddedToken("<|endoftext|>", special=True),
            AddedToken("<|im_start|>", special=True),
            AddedToken("<|im_end|>", special=True),
        ]
    )
    tokenizer.save(str(path))


class Qwen35LiveK4SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(
            prefix=".qwen35-live-k4-test-", dir=Path.cwd()
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
        cls.tokenizer = cls.root / "tokenizer.json"
        _tokenizer(cls.tokenizer)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _args(
        self,
        *,
        output: str = "result.json",
        mode: str = "native-crsa",
        parity: str = "tokenwise",
    ):
        return smoke._parser().parse_args(
            [
                "--target-bundle",
                str(self.target),
                "--draft-bundle",
                str(self.draft),
                "--tokenizer-json",
                str(self.tokenizer),
                "--prompt",
                "1",
                "--max-new-tokens",
                "4",
                "--expected-token-ids",
                "0,0,0,0",
                "--attention-mode",
                mode,
                "--parity-control",
                parity,
                "--device",
                "cpu",
                "--compute-dtype",
                "bfloat16",
                "--target-source-budget-mb",
                "100",
                "--draft-source-budget-mb",
                "100",
                "--max-resident-mb",
                "2",
                "--max-context-tokens",
                "64",
                "--head-block-rows",
                "8",
                "--output",
                str(self.root / output),
            ]
        )

    def _run(self, args):
        return smoke.run(
            args,
            require_production_profile=False,
            require_official_target=False,
        )

    def test_parser_defaults_to_native_k4_and_tokenwise_proof(self) -> None:
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                smoke._parser().parse_args(["--help"])
        self.assertEqual(caught.exception.code, 0)
        defaults = smoke._parser().parse_args([])
        self.assertEqual(defaults.attention_mode, "native-crsa")
        self.assertEqual(defaults.max_new_tokens, 4)
        self.assertEqual(defaults.parity_control, "tokenwise")
        self.assertEqual(smoke._canonical_eos_ids((3, 1, 3, 2)), (1, 2, 3))
        with self.assertRaises(Exception):
            smoke._positive_int("0")

    def test_real_bundles_full_accept_one_k4_stage_and_tokenwise_parity(self) -> None:
        report, output = self._run(self._args())

        self.assertEqual(report["status"], "positive")
        self.assertEqual(report["tokens"]["generated_token_ids"], [0, 0, 0, 0])
        self.assertEqual(report["acceptance"]["accepted_prefix_histogram"]["4"], 1)
        self.assertEqual(report["acceptance"]["rate"], 1.0)
        self.assertEqual(report["acceptance"]["replay_histogram"], {"commit-k4": 1})
        self.assertEqual(report["target"]["speculative_evidence"]["head_scans"], 1)
        transactions = report["target"]["transactions"]["transactions"]
        self.assertEqual(len(transactions), 1)
        self.assertEqual(transactions[0]["transition"], "committed")
        self.assertEqual(
            transactions[0]["evidence"]["end_pos"]
            - transactions[0]["evidence"]["start_pos"],
            4,
        )
        self.assertEqual(len(report["target"]["positions"]), 4)
        self.assertTrue(report["parity"]["tokenwise_control"]["comparison"]["positive"])
        self.assertTrue(
            all(
                report["parity"]["tokenwise_control"]["comparison"][
                    "hidden_equal_by_position"
                ]
            )
        )
        self.assertEqual(output.read_bytes(), smoke._canonical(report) + b"\n")
        self.assertEqual(
            smoke._validate_result(
                report,
                require_production_profile=False,
                require_official_target=False,
            ),
            report,
        )

    def test_mismatch_prefix_two_restage_and_terminal_tail_are_sealed(self) -> None:
        args = self._args(output="mismatch2.json")
        original = smoke.Qwen38WeightPager.topk_logits
        draft_calls = 0

        def sequenced_draft(pager, hidden, **kwargs):
            nonlocal draft_calls
            if pager.source.repo_id != QWEN35_DRAFTER_REPO_ID:
                return original(pager, hidden, **kwargs)
            token = (0, 0, 1, 1)[draft_calls % 4]
            draft_calls += 1
            leading = tuple(hidden.shape[:-1])
            values = torch.zeros((*leading, 1), dtype=torch.float32)
            selected = torch.full((*leading, 1), token, dtype=torch.long)
            return values, selected

        with mock.patch.object(
            smoke.Qwen38WeightPager, "topk_logits", new=sequenced_draft
        ):
            report, _ = self._run(args)

        self.assertEqual(report["tokens"]["generated_token_ids"], [0, 0, 0, 0])
        self.assertEqual(report["acceptance"]["accepted_prefix_histogram"]["2"], 1)
        self.assertEqual(
            report["acceptance"]["replay_histogram"],
            {"mismatch2-restage": 1, "terminal-single": 1},
        )
        transitions = [
            row["transition"]
            for row in report["target"]["transactions"]["transactions"]
        ]
        self.assertEqual(transitions, ["discarded", "committed", "decoded"])
        self.assertEqual(report["drafter"]["alignment"]["target_only_suffix_ids"], [0])
        self.assertTrue(report["parity"]["tokenwise_control"]["comparison"]["positive"])

    def test_eos_zero_and_explicit_off_mode(self) -> None:
        args = self._args(output="eos-off.json", mode="off", parity="none")
        args.expected_token_ids = None
        args.eos_token_ids = (1,)
        original = smoke.Qwen38WeightPager.topk_logits

        def eos_one(pager, hidden, **kwargs):
            values, selected = original(pager, hidden, **kwargs)
            return values, torch.ones_like(selected)

        with mock.patch.object(smoke.Qwen38WeightPager, "topk_logits", new=eos_one):
            report, _ = self._run(args)

        self.assertEqual(report["tokens"]["generated_token_ids"], [1])
        self.assertEqual(report["acceptance"]["accepted_prefix_histogram"]["1"], 1)
        self.assertEqual(report["acceptance"]["replay_histogram"], {"eos0-decode": 1})
        self.assertEqual(report["target"]["native_crsa"]["events"], [])
        self.assertIsNone(report["parity"]["tokenwise_control"])

    def test_tamper_profile_and_nested_receipt_rejection(self) -> None:
        report, _ = self._run(self._args(output="validation.json", parity="none"))
        tampered = copy.deepcopy(report)
        tampered["tokens"]["generated_token_ids"][0] = 1
        with self.assertRaisesRegex(smoke.LiveK4SmokeError, "seal"):
            smoke._validate_result(tampered)

        nested = copy.deepcopy(report)
        nested["tokens"]["rounds"][0]["accepted_prefix_length"] = 3
        nested.pop("sha256")
        nested = smoke._seal(nested)
        with self.assertRaisesRegex(smoke.LiveK4SmokeError, "round evidence"):
            smoke._validate_result(nested)

        transaction = copy.deepcopy(report)
        transaction["target"]["transactions"]["transactions"][0]["transition"] = (
            "discarded"
        )
        transaction_receipt = transaction["target"]["transactions"]
        transaction_receipt.pop("sha256")
        transaction_receipt["sha256"] = smoke._sha256(transaction_receipt)
        transaction.pop("sha256")
        transaction = smoke._seal(transaction)
        with self.assertRaisesRegex(smoke.LiveK4SmokeError, "order"):
            smoke._validate_result(
                transaction,
                require_production_profile=False,
                require_official_target=False,
            )

        nested_state = copy.deepcopy(report)
        state = nested_state["target"]["state"]
        state["state_bytes"] += 2
        state.pop("sha256")
        state["sha256"] = smoke._sha256(state)
        nested_state.pop("sha256")
        nested_state = smoke._seal(nested_state)
        with self.assertRaisesRegex(smoke.LiveK4SmokeError, "byte total"):
            smoke._validate_result(
                nested_state,
                require_production_profile=False,
                require_official_target=False,
            )

        speculative = copy.deepcopy(report)
        evidence = speculative["target"]["speculative_evidence"]
        final_round = evidence["rounds"][-1]
        final_round["state_bytes"] += 2
        final_round.pop("evidence_sha256")
        final_round["evidence_sha256"] = smoke._sha256(final_round)
        evidence["state_bytes"] += 2
        evidence.pop("evidence_sha256")
        evidence["evidence_sha256"] = smoke._sha256(evidence)
        speculative.pop("sha256")
        speculative = smoke._seal(speculative)
        with self.assertRaisesRegex(smoke.LiveK4SmokeError, "cross-linked"):
            smoke._validate_result(
                speculative,
                require_production_profile=False,
                require_official_target=False,
            )

        production = copy.deepcopy(report)
        production["contract"]["pinned_production_profiles"] = True
        production["contract"]["official_target_config"] = True
        production["target"]["bundle"].update(smoke.base._TARGET_PROFILE)
        production["drafter"]["bundle"].update(smoke.base._DRAFT_PROFILE)
        production.pop("sha256")
        production = smoke._seal(production)
        smoke._validate_result(production)
        wrong = copy.deepcopy(production)
        wrong["drafter"]["bundle"]["checkpoint_bytes"] += 1
        wrong.pop("sha256")
        wrong = smoke._seal(wrong)
        with self.assertRaisesRegex(Exception, "profile changed"):
            smoke._validate_result(wrong)

        downgraded = copy.deepcopy(production)
        downgraded["contract"]["pinned_production_profiles"] = False
        downgraded["contract"]["official_target_config"] = False
        downgraded.pop("sha256")
        downgraded = smoke._seal(downgraded)
        with self.assertRaisesRegex(Exception, "cannot be downgraded"):
            smoke._validate_result(downgraded)

        short = copy.deepcopy(report)
        short["contract"]["max_new_tokens"] = 5
        short.pop("sha256")
        short = smoke._seal(short)
        with self.assertRaisesRegex(Exception, "short output"):
            smoke._validate_result(
                short,
                require_production_profile=False,
                require_official_target=False,
            )

        final_state = json.loads(json.dumps(report))
        transaction_row = final_state["target"]["transactions"]["transactions"][-1]
        state_copies = (
            transaction_row["staged_state"],
            transaction_row["committed_state"],
            final_state["target"]["positions"][-1]["state_after_position"],
        )
        for state_copy in state_copies:
            tensor = next(
                tensor
                for layer in state_copy["layers"]
                if layer is not None
                for tensor in layer["tensors"].values()
            )
            tensor["sha256"] = "1" * 64
            state_copy.pop("sha256")
            state_copy["sha256"] = smoke._sha256(state_copy)
        trace = final_state["target"]["transactions"]
        trace.pop("sha256")
        trace["sha256"] = smoke._sha256(trace)
        final_state.pop("sha256")
        final_state = smoke._seal(final_state)
        with self.assertRaisesRegex(Exception, "final transaction state"):
            smoke._validate_result(
                final_state,
                require_production_profile=False,
                require_official_target=False,
            )

        committed_hidden = json.loads(json.dumps(report))
        trace = committed_hidden["target"]["transactions"]
        trace["transactions"][0]["committed_hidden_positions"][0]["hidden"][
            "sha256"
        ] = "2" * 64
        trace.pop("sha256")
        trace["sha256"] = smoke._sha256(trace)
        committed_hidden.pop("sha256")
        committed_hidden = smoke._seal(committed_hidden)
        with self.assertRaisesRegex(Exception, "position hidden differs"):
            smoke._validate_result(
                committed_hidden,
                require_production_profile=False,
                require_official_target=False,
            )

        missing_crsa = json.loads(json.dumps(report))
        native = missing_crsa["target"]["native_crsa"]
        native["events"].pop()
        native["sha256"] = smoke._sha256(native["events"])
        missing_crsa.pop("sha256")
        missing_crsa = smoke._seal(missing_crsa)
        with self.assertRaisesRegex(Exception, "history is incomplete"):
            smoke._validate_result(
                missing_crsa,
                require_production_profile=False,
                require_official_target=False,
            )

    def test_fully_resealed_false_parity_and_protected_output_are_rejected(
        self,
    ) -> None:
        report, _ = self._run(self._args(output="parity-tamper.json"))
        forged = copy.deepcopy(report)
        control = forged["parity"]["tokenwise_control"]
        control["positions"][0]["hidden"]["sha256"] = "0" * 64
        control["comparison"]["hidden_equal_by_position"][0] = True
        control["comparison"]["hidden_equal"] = True
        control["comparison"]["positive"] = True
        control.pop("sha256")
        control["sha256"] = smoke._sha256(control)
        forged.pop("sha256")
        forged = smoke._seal(forged)
        with self.assertRaisesRegex(smoke.LiveK4SmokeError, "comparison"):
            smoke._validate_result(
                forged,
                require_production_profile=False,
                require_official_target=False,
            )

        forged_crsa = copy.deepcopy(report)
        control = forged_crsa["parity"]["tokenwise_control"]
        event = control["positions"][0]["native_crsa"]
        event["row_sum_max_error"] = 1e-8 if event["row_sum_max_error"] == 0.0 else 0.0
        control.pop("sha256")
        control["sha256"] = smoke._sha256(control)
        forged_crsa.pop("sha256")
        forged_crsa = smoke._seal(forged_crsa)
        with self.assertRaisesRegex(smoke.LiveK4SmokeError, "CRSA copy"):
            smoke._validate_result(
                forged_crsa,
                require_production_profile=False,
                require_official_target=False,
            )

        args = self._args(output="unused.json", parity="none")
        args.output = str(self.target / "bundle.json")
        protected = self.target / "bundle.json"
        before = protected.read_bytes()
        with mock.patch.object(smoke.base, "_open_model") as open_model:
            with self.assertRaisesRegex(smoke.LiveK4SmokeError, "must not modify"):
                self._run(args)
        open_model.assert_not_called()
        self.assertEqual(protected.read_bytes(), before)

        args = self._args(output="too-short.json", parity="none")
        args.max_new_tokens = 3
        args.expected_token_ids = (0, 0, 0)
        with mock.patch.object(smoke.base, "_open_model") as open_model:
            with self.assertRaisesRegex(smoke.LiveK4SmokeError, "at least four"):
                self._run(args)
        open_model.assert_not_called()

    def test_failure_closes_both_models_and_provider(self) -> None:
        opened = []
        original_open = smoke.base._open_model

        def tracked_open(**kwargs):
            owned = original_open(**kwargs)
            opened.append(owned)
            return owned

        with (
            mock.patch.object(smoke.base, "_open_model", new=tracked_open),
            mock.patch.object(
                smoke.Qwen38K4SpeculativeDecoder,
                "generate",
                side_effect=RuntimeError("forced generation failure"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "forced generation failure"):
                self._run(self._args(output="failure.json", parity="none"))

        self.assertEqual(len(opened), 2)
        self.assertTrue(all(owned._closed for owned in opened))
        self.assertTrue(all(owned.mount._closed for owned in opened))


if __name__ == "__main__":
    unittest.main()
