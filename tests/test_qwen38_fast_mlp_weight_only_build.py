from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch

from immer.runtimes.deepseek_v4.causal_weights import LogicalModelIdentity
from immer.runtimes.ooe.mlp_pilot_router import MlpPilotRouterConfig
from immer.runtimes.ooe.mlp_pilot_weight_only import MlpPilotOnlineConfig
from immer.runtimes.qwen3_8.fast_mlp import Qwen38FastMlpPaths

from test_qwen3_8_model import _tiny_config

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts" / "qwen38_fast_mlp_weight_only_build.py"
    spec = importlib.util.spec_from_file_location("_weight_only_builder", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _Pager:
    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(177)
        self.tensors: dict[str, torch.Tensor] = {}
        for layer in range(2):
            base = f"model.language_model.layers.{layer}.mlp"
            self.tensors[f"{base}.gate_proj.weight"] = torch.randn(
                8, 3, generator=generator, dtype=torch.bfloat16
            )
            self.tensors[f"{base}.up_proj.weight"] = torch.randn(
                8, 3, generator=generator, dtype=torch.bfloat16
            )
            self.tensors[f"{base}.down_proj.weight"] = torch.randn(
                3, 8, generator=generator, dtype=torch.bfloat16
            )

    def tensor_rows(self, name: str, rows: tuple[int, ...]) -> torch.Tensor:
        return self.tensors[name][torch.tensor(rows, dtype=torch.long)].clone()

    def tensor_torch(
        self, name: str, *, dtype: torch.dtype, device: str
    ) -> torch.Tensor:
        return self.tensors[name].to(dtype=dtype, device=device).clone()


class WeightOnlyBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.script = _load_script()
        self.config = replace(
            _tiny_config(),
            dim=3,
            intermediate_size=8,
            n_layers=2,
            layer_types=("linear_attention", "linear_attention"),
        )
        self.router_config = MlpPilotRouterConfig(
            block_size=4,
            pilot_count=1,
            selected_block_count=1,
            random_seed_sha256=_hash("builder-random"),
            max_working_bytes=1024**2,
        )
        self.online_config = MlpPilotOnlineConfig(
            min_confirmed_rows=2,
            min_capture=0.0,
            confirmation_interval=2,
        )
        self.bundle = {
            "layout_fingerprint": _hash("builder-layout"),
            "manifest_sha256": _hash("builder-bundle"),
        }

    def _plan(self, pager: _Pager):
        return self.script._build_plan(
            pager=pager,
            model_identity=LogicalModelIdentity("local/tiny", "revision"),
            bundle=self.bundle,
            config=self.config,
            router_config=self.router_config,
            online_config=self.online_config,
            weights_index_sha256=_hash("builder-index"),
            chunk_rows=4,
        )

    def test_plan_and_banks_are_deterministic_complete_and_idempotent(self) -> None:
        pager = _Pager()
        first = self._plan(pager)
        second = self._plan(pager)
        self.assertEqual(first.to_bytes(), second.to_bytes())
        self.assertEqual(tuple(row.layer for row in first.models), (0, 1))

        with tempfile.TemporaryDirectory() as tmp:
            paths = Qwen38FastMlpPaths.from_root(tmp)
            for root in (paths.analysis_root, paths.transpose_root, paths.pilot_root):
                root.mkdir(parents=True)
            transpose, pilot = self.script._build_weight_banks(
                pager=pager,
                plan=first,
                paths=paths,
                max_transpose_shard_bytes=1024**2,
                max_transpose_total_bytes=4 * 1024**2,
                max_pilot_shard_bytes=1024**2,
                max_pilot_total_bytes=4 * 1024**2,
            )
            rebuilt_transpose, rebuilt_pilot = self.script._build_weight_banks(
                pager=pager,
                plan=first,
                paths=paths,
                max_transpose_shard_bytes=1024**2,
                max_transpose_total_bytes=4 * 1024**2,
                max_pilot_shard_bytes=1024**2,
                max_pilot_total_bytes=4 * 1024**2,
            )

            self.assertEqual(transpose.to_bytes(), rebuilt_transpose.to_bytes())
            self.assertEqual(pilot, rebuilt_pilot)
            self.assertEqual(tuple(row.layer for row in transpose.entries), (0, 1))
            self.assertEqual(
                [row["layer"] for row in pilot["body"]["entries"]],
                [0, 1],
            )
            self.assertEqual(len(list(paths.transpose_root.glob("*.safetensors"))), 2)
            self.assertEqual(len(list(paths.pilot_root.glob("*.safetensors"))), 2)

    def test_bank_bound_fails_closed(self) -> None:
        pager = _Pager()
        plan = self._plan(pager)
        with tempfile.TemporaryDirectory() as tmp:
            paths = Qwen38FastMlpPaths.from_root(tmp)
            for root in (paths.analysis_root, paths.transpose_root, paths.pilot_root):
                root.mkdir(parents=True)
            with self.assertRaisesRegex(self.script.CliError, "transpose shard"):
                self.script._build_weight_banks(
                    pager=pager,
                    plan=plan,
                    paths=paths,
                    max_transpose_shard_bytes=1,
                    max_transpose_total_bytes=4 * 1024**2,
                    max_pilot_shard_bytes=1024**2,
                    max_pilot_total_bytes=4 * 1024**2,
                )


if __name__ == "__main__":
    unittest.main()
