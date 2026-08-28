from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.qwen3_8 import Qwen38Tokenizer, prompt_token_sha256


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qwen38_o1_prompt_cohort.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "qwen38_o1_prompt_cohort_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tokenizer(path: Path) -> None:
    from tokenizers import AddedToken, Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit

    vocab = {"[UNK]": 0, **{f"question-{index}": index + 1 for index in range(10)}}
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


class Qwen38O1PromptCohortTests(unittest.TestCase):
    def test_label_free_deterministic_selection_excludes_existing_prompt(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer_path = root / "tokenizer.json"
            _tokenizer(tokenizer_path)
            tokenizer = Qwen38Tokenizer(tokenizer_path, require_official=False)
            existing_tokens = tokenizer.encode(
                Qwen38Tokenizer.render_no_thinking_prompt(
                    module.SYSTEM_PROMPT, "question-0"
                )
            )
            existing = root / "existing.json"
            existing.write_bytes(
                canonical_json_bytes(
                    {
                        "body": {
                            "prompts": [
                                {
                                    "sha256": prompt_token_sha256(existing_tokens),
                                    "spec_defaults": {},
                                    "token_ids": list(existing_tokens),
                                }
                            ]
                        }
                    }
                )
            )
            benchmark = root / "benchmark.json"
            benchmark.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "gold": 1000 + index,
                                "item_id": f"item-{index}",
                                "question": f"question-{index}",
                                "status": "ignored-label-field",
                            }
                            for index in range(8)
                        ]
                    }
                )
            )
            prompts_path = root / "prompts.json"
            manifest_path = root / "cohort.json"
            arguments = [
                "--benchmark",
                str(benchmark),
                "--tokenizer",
                str(tokenizer_path),
                "--existing-manifest",
                str(existing),
                "--count",
                "3",
                "--allow-nonofficial-tokenizer",
                "--output-prompts",
                str(prompts_path),
                "--output-manifest",
                str(manifest_path),
            ]
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                self.assertEqual(module.main(arguments), 0)
            status = json.loads(stream.getvalue())
            prompts = json.loads(prompts_path.read_bytes())["prompts"]
            cohort = json.loads(manifest_path.read_bytes())
            self.assertEqual(status["count"], 3)
            self.assertEqual(len(prompts), 3)
            self.assertNotIn(
                prompt_token_sha256(existing_tokens),
                {row["sha256"] for row in prompts},
            )
            self.assertNotIn("gold", manifest_path.read_text())
            self.assertNotIn("question-", manifest_path.read_text())
            self.assertEqual(cohort["schema"], module.COHORT_SCHEMA)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(arguments), 0)

    def test_gold_changes_do_not_change_selected_prompt_registry(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer_path = root / "tokenizer.json"
            _tokenizer(tokenizer_path)
            tokenizer = Qwen38Tokenizer(tokenizer_path, require_official=False)
            tokens = tokenizer.encode(
                Qwen38Tokenizer.render_no_thinking_prompt(
                    module.SYSTEM_PROMPT, "question-0"
                )
            )
            existing = root / "existing.json"
            existing.write_text(
                json.dumps(
                    {
                        "prompts": [
                            {
                                "sha256": prompt_token_sha256(tokens),
                                "spec_defaults": {},
                                "token_ids": list(tokens),
                            }
                        ]
                    }
                )
            )
            outputs = []
            for version in (1, 2):
                benchmark = root / f"benchmark-{version}.json"
                benchmark.write_text(
                    json.dumps(
                        {
                            "items": [
                                {
                                    "gold": version * (index + 1),
                                    "item_id": f"item-{index}",
                                    "question": f"question-{index}",
                                }
                                for index in range(8)
                            ]
                        }
                    )
                )
                prompts, _cohort = module.build_cohort(
                    benchmark_path=benchmark,
                    tokenizer_path=tokenizer_path,
                    existing_manifest_paths=(existing,),
                    existing_prompt_sha256s=(),
                    count=3,
                    seed_sha256=module._parser().get_default("seed_sha256"),
                    max_prompt_tokens=512,
                    require_official_tokenizer=False,
                )
                outputs.append(prompts)
            self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
