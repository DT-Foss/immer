from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts" / "qwen38_fast_mlp_existing_evidence.py"
    spec = importlib.util.spec_from_file_location("_existing_evidence", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pair(layer: int, ordinal: int, split: str, identity: str):
    entry = SimpleNamespace(
        layer=layer,
        ordinal=ordinal,
        prompt_sha256=f"prompt-{ordinal}",
        split=split,
    )
    receipt = SimpleNamespace(
        entry=entry,
        identity_sha256=identity,
        tensors=(),
    )
    return receipt, SimpleNamespace()


class ExistingEvidenceCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.script = _load_script()

    def test_pairs_are_filtered_and_sorted_layer_major(self) -> None:
        pairs = (
            _pair(1, 4, "holdout", "d"),
            _pair(0, 2, "calibration", "b"),
            _pair(0, 1, "train", "a"),
            _pair(1, 3, "train", "c"),
        )

        grouped = self.script._pairs_by_layer(
            pairs,
            splits=("train", "calibration"),
        )

        self.assertEqual(tuple(grouped), (0, 1))
        self.assertEqual(
            tuple(pair[0].entry.ordinal for pair in grouped[0]),
            (1, 2),
        )
        self.assertEqual(
            tuple(pair[0].entry.ordinal for pair in grouped[1]),
            (3,),
        )
        with self.assertRaises(self.script.CliError):
            self.script._pairs_by_layer(
                (pairs[0], pairs[0]),
                splits=("holdout",),
            )

    def test_layer_calibration_reuses_one_down_weight(self) -> None:
        intermediate = 8
        hidden = 3
        generator = np.random.default_rng(71)

        def receipt(ordinal: int):
            references = []
            tensors = {}
            for stage, width in (
                ("mlp.gate", intermediate),
                ("mlp.up", intermediate),
                ("mlp.output", hidden),
            ):
                reference = SimpleNamespace(
                    stage=stage,
                    shape=(1, 2, width),
                    key=f"{ordinal}:{stage}",
                )
                references.append(reference)
                tensors[reference.key] = generator.normal(
                    size=reference.shape
                ).astype(np.float32)
            row = SimpleNamespace(tensors=tuple(references))
            return row, tensors

        first, first_tensors = receipt(0)
        second, second_tensors = receipt(1)
        stored = {**first_tensors, **second_tensors}

        class Bank:
            @staticmethod
            def restore_tensor(reference):
                return stored[reference.key]

        class Pager:
            def __init__(self) -> None:
                self.reads = 0
                self.releases = 0
                self.down = torch.randn(hidden, intermediate, dtype=torch.bfloat16)

            def tensor_torch(self, name, *, dtype, device):
                self.reads += 1
                self.assert_name = name
                return self.down.to(dtype=dtype, device=device)

            def release(self, *, force_gc=False):
                self.releases += int(force_gc)

        class Executor:
            def __init__(self) -> None:
                self.rows = []

            def observe_full(self, **values):
                self.rows.append(values)
                return SimpleNamespace(shadow_row_indices=(0,))

        pager = Pager()
        executor = Executor()
        config = SimpleNamespace(dim=hidden, intermediate_size=intermediate)

        observed = self.script._calibrate_layer(
            layer=7,
            pairs=((first, None), (second, None)),
            bank=Bank(),
            pager=pager,
            executor=executor,
            config=config,
        )

        self.assertEqual(observed, 2)
        self.assertEqual(pager.reads, 1)
        self.assertEqual(pager.releases, 1)
        self.assertEqual(
            pager.assert_name,
            "model.language_model.layers.7.mlp.down_proj.weight",
        )
        self.assertEqual(len(executor.rows), 2)
        self.assertTrue(
            all(values["down_weight"] is pager.down for values in executor.rows)
        )
        self.assertTrue(
            all(values["activated"].dtype == torch.bfloat16 for values in executor.rows)
        )

    def test_state_clone_is_new_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            output = root / "candidate.json"
            source.write_bytes(b'{"state":"existing"}')

            self.script._clone_state(source, output)

            self.assertEqual(output.read_bytes(), source.read_bytes())
            with self.assertRaises(self.script.CliError):
                self.script._clone_state(source, output)
            with self.assertRaises(self.script.CliError):
                self.script._clone_state(source, source)

    def test_split_parser_rejects_unknown_or_duplicates(self) -> None:
        self.assertEqual(
            self.script._split_list("train,calibration,holdout"),
            ("train", "calibration", "holdout"),
        )
        for value in ("", "train,train", "train,unknown"):
            with self.subTest(value=value), self.assertRaises(Exception):
                self.script._split_list(value)


if __name__ == "__main__":
    unittest.main()
