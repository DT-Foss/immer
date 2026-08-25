from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
import copy
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
SCRIPT = ROOT / "scripts" / "qwen35_live_k2_smoke.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("qwen35_live_k2_smoke", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import Qwen3.5 live K2 smoke")
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


class Qwen35LiveK2SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(
            prefix=".qwen35-live-k2-test-", dir=Path.cwd()
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

    def _args(self, *, output: str = "result.json", mode: str = "native-crsa"):
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

    def test_help_and_k2_parser_are_side_effect_free_and_strict(self) -> None:
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                smoke._parser().parse_args(["--help"])
        self.assertEqual(caught.exception.code, 0)
        self.assertEqual(smoke._parser().parse_args([]).attention_mode, "native-crsa")
        self.assertEqual(smoke._parser().parse_args([]).max_new_tokens, 2)
        self.assertEqual(smoke._multiple_k2_tokens("2"), 2)
        self.assertEqual(smoke._multiple_k2_tokens("4"), 4)
        for value in ("1", "3", "5"):
            with self.subTest(value=value):
                with self.assertRaises(Exception):
                    smoke._multiple_k2_tokens(value)

    def test_two_real_causal_bundles_run_native_k2_and_write_one_seal(self) -> None:
        args = self._args()

        report, output = smoke.run(
            args,
            require_production_profile=False,
            require_official_target=False,
        )

        self.assertEqual(output, (self.root / "result.json").resolve())
        self.assertEqual(report["status"], "positive")
        self.assertEqual(report["tokens"]["generated_token_ids"], [0, 0, 0, 0])
        self.assertTrue(report["parity"]["exact"])
        self.assertEqual(report["contract"]["attention_mode"], "native-crsa")
        self.assertEqual(report["contract"]["attention"]["selected_sinkhorn_heads"], 4)
        self.assertEqual(report["acceptance"]["accepted_draft_tokens"], 4)
        self.assertEqual(report["acceptance"]["draft_verification_rounds"], 2)
        self.assertEqual(report["acceptance"]["target_verification_rounds"], 2)
        self.assertEqual(report["acceptance"]["terminal_single_rounds"], 0)
        self.assertEqual(report["drafter"]["provider"]["accepted_prefix_2"], 2)
        self.assertTrue(report["drafter"]["config"]["tied_embeddings"])
        self.assertGreater(report["target"]["native_crsa"]["count"], 0)
        self.assertEqual(
            report["target"]["native_crsa"]["events"][-1]["history_length_after"],
            report["target"]["state"]["cursor"],
        )
        self.assertFalse(report["target"]["state"]["pending_block"])
        self.assertFalse(report["drafter"]["state"]["pending_block"])
        self.assertGreater(
            report["target"]["pager"]["counters"]["logical_weight_bytes"], 0
        )
        self.assertGreater(
            report["drafter"]["pager"]["counters"]["logical_weight_bytes"], 0
        )
        encoded = output.read_bytes()
        self.assertEqual(encoded, smoke._canonical(report) + b"\n")
        self.assertEqual(json.loads(encoded), report)
        self.assertEqual(smoke._validate_result(report), report)

        production = copy.deepcopy(report)
        production["contract"]["pinned_production_profiles"] = True
        production["contract"]["official_target_config"] = True
        production["target"]["bundle"].update(smoke._TARGET_PROFILE)
        production["target"]["bundle"].update(
            {"repo_id": OFFICIAL_REPO_ID, "revision": OFFICIAL_REVISION}
        )
        production["target"]["config"] = {
            "heads": 24,
            "layers": 64,
            "vocab_size": 248_320,
        }
        production["drafter"]["bundle"].update(smoke._DRAFT_PROFILE)
        production["drafter"]["bundle"].update(
            {
                "repo_id": QWEN35_DRAFTER_REPO_ID,
                "revision": QWEN35_DRAFTER_REVISION,
            }
        )
        production["drafter"]["config"] = {
            "layers": 24,
            "tied_embeddings": True,
            "vocab_size": 248_320,
        }
        production.pop("sha256")
        production = smoke._seal(production)
        smoke._validate_result(production)
        wrong_profile = copy.deepcopy(production)
        wrong_profile["target"]["bundle"]["checkpoint_bytes"] += 1
        wrong_profile.pop("sha256")
        wrong_profile = smoke._seal(wrong_profile)
        with self.assertRaisesRegex(smoke.LiveK2SmokeError, "profile changed"):
            smoke._validate_result(wrong_profile)

        tampered = json.loads(encoded)
        tampered["tokens"]["generated_token_ids"][0] = 1
        with self.assertRaisesRegex(smoke.LiveK2SmokeError, "seal"):
            smoke._validate_result(tampered)

    def test_explicit_off_mode_keeps_all_heads_softmax(self) -> None:
        args = self._args(output="result-off.json", mode="off")
        args.max_new_tokens = 2
        args.expected_token_ids = (0, 0)

        report, _output = smoke.run(
            args,
            require_production_profile=False,
            require_official_target=False,
        )

        self.assertEqual(report["status"], "positive")
        self.assertEqual(report["contract"]["attention_mode"], "off")
        self.assertEqual(
            report["contract"]["attention"],
            {"free_softmax_heads": 24, "selected_sinkhorn_heads": 0},
        )
        self.assertEqual(
            report["target"]["native_crsa"],
            {"count": 0, "events": [], "sha256": smoke._sha256([])},
        )

    def test_real_mismatch_zero_records_one_target_only_terminal_token(self) -> None:
        args = self._args(output="result-mismatch0.json")
        args.max_new_tokens = 2
        args.expected_token_ids = (0, 0)
        original_topk = smoke.Qwen38WeightPager.topk_logits

        def divergent_draft_topk(
            pager,
            hidden,
            *,
            k=1,
            name="lm_head.weight",
            block_rows=smoke.Qwen38WeightPager.DEFAULT_HEAD_BLOCK_ROWS,
            progress=None,
        ):
            if pager.source.repo_id != QWEN35_DRAFTER_REPO_ID:
                return original_topk(
                    pager,
                    hidden,
                    k=k,
                    name=name,
                    block_rows=block_rows,
                    progress=progress,
                )
            leading = tuple(hidden.shape[:-1])
            selected = torch.ones((*leading, 1), dtype=torch.long, device=hidden.device)
            values = torch.zeros(
                (*leading, 1), dtype=torch.float32, device=hidden.device
            )
            return values, selected

        with mock.patch.object(
            smoke.Qwen38WeightPager,
            "topk_logits",
            new=divergent_draft_topk,
        ):
            report, _output = smoke.run(
                args,
                require_production_profile=False,
                require_official_target=False,
            )

        self.assertEqual(report["status"], "positive")
        self.assertEqual(report["tokens"]["generated_token_ids"], [0, 0])
        self.assertEqual(report["acceptance"]["accepted_prefix_histogram"]["0"], 1)
        self.assertEqual(report["acceptance"]["draft_verification_rounds"], 1)
        self.assertEqual(report["acceptance"]["target_verification_rounds"], 2)
        self.assertEqual(report["acceptance"]["terminal_single_rounds"], 1)
        self.assertEqual(
            report["acceptance"]["replay_histogram"],
            {"mismatch0-decode": 1, "remaining-single": 1},
        )
        self.assertEqual(report["drafter"]["alignment"]["target_only_suffix_ids"], [0])
        self.assertEqual(
            report["target"]["state"]["cursor"],
            report["drafter"]["state"]["cursor"] + 1,
        )

    def test_independent_reference_mismatch_is_sealed_and_cli_returns_one(self) -> None:
        generated = (0, 0, 0, 0)
        mismatch = smoke._parity((1, 0, 0, 0), generated, required_length=4)
        self.assertEqual(
            mismatch,
            {
                "exact": False,
                "expected_token_ids": [1, 0, 0, 0],
                "provided": True,
            },
        )
        with self.assertRaisesRegex(smoke.LiveK2SmokeError, "exactly"):
            smoke._parity((0, 0), generated, required_length=4)

        mismatch_report = {
            "acceptance": {"rate": 0.0},
            "contract": {"attention_mode": "native-crsa"},
            "parity": mismatch,
            "status": "mismatch",
            "tokens": {"generated_token_ids": list(generated)},
        }
        path = self.root / "mismatch.json"

        original_run = smoke.run
        smoke.run = lambda _args: (mismatch_report, path)
        try:
            stdout = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                code = smoke.main([])
        finally:
            smoke.run = original_run
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "mismatch")


if __name__ == "__main__":
    unittest.main()
