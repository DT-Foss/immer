from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.math_core import sinkhorn_project
from immer.runtimes.qwen3_8.q4_delta_router import PackedDeltaHeadRouter


@dataclass(frozen=True, slots=True)
class _FakeEntry:
    shape: tuple[int, int]
    payload_bytes: int


class _ExactSelectedBlockBank:
    def __init__(
        self,
        *,
        manifest: str = "1" * 64,
        value_heads: int = 48,
        head_dim: int = 128,
        output_dim: int = 7,
    ) -> None:
        self.root = Path("/fake/q4")
        self._identity = {"manifest_sha256": manifest}
        self.name = "model.language_model.layers.2.linear_attn.out_proj.weight"
        input_dim = value_heads * head_dim
        payload_bytes = output_dim * (input_dim // 32) * 18
        self.entries = {
            self.name: _FakeEntry(
                shape=(output_dim, input_dim),
                payload_bytes=payload_bytes,
            )
        }
        values = torch.arange(output_dim * input_dim, dtype=torch.float32)
        self.weight = ((values.remainder(29) - 14.0) / 31.0).reshape(
            output_dim, input_dim
        )
        self.calls: list[tuple[torch.Tensor, torch.Tensor, str]] = []
        self.last_dense: torch.Tensor | None = None

    @property
    def identity(self) -> dict[str, str]:
        return dict(self._identity)

    def linear_selected_blocks(
        self,
        block_values: torch.Tensor,
        block_ids: torch.Tensor,
        name: str,
        *,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        entry = self.entries[name]
        dense = torch.zeros(
            (block_values.shape[0], entry.shape[1]),
            dtype=block_values.dtype,
        )
        for row in range(block_values.shape[0]):
            for selected, block in enumerate(block_ids[row].tolist()):
                begin = block * 32
                dense[row, begin : begin + 32] = block_values[row, selected]
        self.calls.append((block_values.clone(), block_ids.clone(), name))
        self.last_dense = dense.clone()
        result = dense.float() @ self.weight.float().T
        dtype = block_values.dtype if output_dtype is None else output_dtype
        return result.to(dtype=dtype)


def _head_values(
    *,
    batch: int,
    sequence: int,
    value_heads: int = 48,
    head_dim: int = 128,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    values = torch.arange(1, value_heads + 1, dtype=torch.float32)
    return (
        values.reshape(1, 1, value_heads, 1)
        .expand(batch, sequence, value_heads, head_dim)
        .reshape(batch, sequence, value_heads * head_dim)
        .to(dtype=dtype)
    )


class PackedDeltaHeadRouterTests(unittest.TestCase):
    def test_selected_head_columns_match_the_exact_zero_omission(self) -> None:
        bank = _ExactSelectedBlockBank()
        router = PackedDeltaHeadRouter(
            bank,  # type: ignore[arg-type]
            active_layers=(2,),
            value_heads=48,
            head_dim=128,
            max_selected_heads=16,
        )
        mixed = _head_values(batch=2, sequence=3)

        actual = router.project(mixed, bank.name, layer=2)

        self.assertEqual(tuple(actual.shape), (2, 3, 7))
        self.assertEqual(actual.dtype, mixed.dtype)
        self.assertEqual(len(bank.calls), 1)
        block_values, block_ids, called_name = bank.calls[0]
        self.assertEqual(called_name, bank.name)
        self.assertEqual(tuple(block_values.shape), (6, 64, 32))
        expected_ids = [
            head * 4 + offset
            for head in range(47, 31, -1)
            for offset in range(4)
        ]
        self.assertEqual(block_ids[0].tolist(), expected_ids)
        self.assertTrue(bool((block_ids == block_ids[0]).all()))
        self.assertIsNotNone(bank.last_dense)
        expected = bank.last_dense.float() @ bank.weight.float().T
        torch.testing.assert_close(
            actual.reshape(6, 7), expected, rtol=0.0, atol=0.0
        )

        metrics = router.metrics()
        payload = bank.entries[bank.name].payload_bytes
        self.assertEqual(metrics["calls"], 1)
        self.assertEqual(metrics["rows"], 6)
        self.assertEqual(metrics["full_equivalent_bytes"], 6 * payload)
        self.assertEqual(
            metrics["logical_bytes_saved"],
            6 * payload - 6 * payload * 16 // 48,
        )
        self.assertEqual(metrics["transitions"], 5)
        self.assertEqual(metrics["width_16"], 1)

    def test_sinkhorn_markov_state_is_canonical_and_survives_restart(self) -> None:
        bank = _ExactSelectedBlockBank()
        with tempfile.TemporaryDirectory() as temporary:
            requested = Path(temporary) / "online-state.json"
            router = PackedDeltaHeadRouter(
                bank,  # type: ignore[arg-type]
                active_layers=(2,),
                state_path=requested,
                max_selected_heads=32,
            )
            zeros = torch.zeros((1, 2, 48 * 128), dtype=torch.float32)
            with mock.patch(
                "immer.runtimes.qwen3_8.q4_delta_router.sinkhorn_project",
                wraps=sinkhorn_project,
            ) as projected:
                router.project(zeros, bank.name, layer=2)
            self.assertEqual(projected.call_count, 1)
            first_metrics = router.metrics()
            state_path = router.state_path
            self.assertIsNotNone(state_path)
            self.assertNotEqual(state_path, requested)
            self.assertIn(".delta-head-", state_path.name)
            self.assertFalse(requested.exists())

            raw = state_path.read_bytes()
            document = json.loads(raw)
            self.assertEqual(canonical_json_bytes(document), raw)
            layer_state = document["layers"]["2"]
            self.assertEqual(len(layer_state["counts"]), 48)
            self.assertTrue(
                all(len(row) == 48 for row in layer_state["counts"])
            )
            self.assertEqual(layer_state["counts"][0][0], 1)
            self.assertEqual(layer_state["previous"], list(range(32)))
            self.assertEqual(layer_state["transitions"], 1)
            self.assertEqual(layer_state["overlap_ema"], 1.0)
            router.close()

            restored = PackedDeltaHeadRouter(
                bank,  # type: ignore[arg-type]
                active_layers=(2,),
                state_path=requested,
                max_selected_heads=32,
            )
            self.assertEqual(restored.state_path, state_path)
            with mock.patch(
                "immer.runtimes.qwen3_8.q4_delta_router.sinkhorn_project",
                wraps=sinkhorn_project,
            ) as projected_again:
                restored.project(zeros[:, :1], bank.name, layer=2)
            self.assertEqual(projected_again.call_count, 1)
            selected_ids = bank.calls[-1][1][0].tolist()
            self.assertEqual(
                selected_ids,
                [
                    head * 4 + offset
                    for head in range(24)
                    for offset in range(4)
                ],
            )
            metrics = restored.metrics()
            self.assertEqual(metrics["calls"], first_metrics["calls"] + 1)
            self.assertEqual(metrics["rows"], first_metrics["rows"] + 1)
            self.assertEqual(metrics["transitions"], 2)
            self.assertEqual(metrics["width_32"], 1)
            self.assertEqual(metrics["width_24"], 1)

    def test_state_namespaces_cover_identity_and_do_not_collide_with_packed_mlp(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            requested = Path(temporary) / "online.json"
            first_bank = _ExactSelectedBlockBank(manifest="1" * 64)
            second_bank = _ExactSelectedBlockBank(manifest="2" * 64)
            first = PackedDeltaHeadRouter(
                first_bank,  # type: ignore[arg-type]
                active_layers=(2,),
                state_path=requested,
            )
            same = PackedDeltaHeadRouter(
                first_bank,  # type: ignore[arg-type]
                active_layers=(2,),
                state_path=requested,
            )
            other_manifest = PackedDeltaHeadRouter(
                second_bank,  # type: ignore[arg-type]
                active_layers=(2,),
                state_path=requested,
            )
            other_layers = PackedDeltaHeadRouter(
                first_bank,  # type: ignore[arg-type]
                active_layers=(1, 2),
                state_path=requested,
            )
            other_maximum = PackedDeltaHeadRouter(
                first_bank,  # type: ignore[arg-type]
                active_layers=(2,),
                state_path=requested,
                max_selected_heads=24,
            )
            other_heads = PackedDeltaHeadRouter(
                first_bank,  # type: ignore[arg-type]
                active_layers=(2,),
                state_path=requested,
                value_heads=24,
                max_selected_heads=16,
            )
            other_dimension = PackedDeltaHeadRouter(
                first_bank,  # type: ignore[arg-type]
                active_layers=(2,),
                state_path=requested,
                head_dim=64,
            )
            paths = {
                first.state_path,
                other_manifest.state_path,
                other_layers.state_path,
                other_maximum.state_path,
                other_heads.state_path,
                other_dimension.state_path,
            }
            self.assertEqual(same.state_path, first.state_path)
            self.assertEqual(len(paths), 6)
            self.assertTrue(
                all("delta-head" in path.name for path in paths if path is not None)
            )
            identity = first.snapshot_identity(transport_neutral=True)
            self.assertNotIn("bank_root", identity)
            self.assertNotIn("state_path", identity)
            self.assertEqual(identity["width_actions"], [24, 32])

            transient = PackedDeltaHeadRouter(
                first_bank,  # type: ignore[arg-type]
                active_layers=(2,),
                state_path=None,
            )
            self.assertIsNone(transient.state_path)
            self.assertFalse(
                transient.snapshot_identity(transport_neutral=True)[
                    "state_persistent"
                ]
            )

    def test_project_many_uses_one_q4_call_and_restores_shapes_and_dtype(self) -> None:
        bank = _ExactSelectedBlockBank()
        router = PackedDeltaHeadRouter(
            bank,  # type: ignore[arg-type]
            active_layers=(2,),
            max_selected_heads=16,
        )
        mixed = (
            _head_values(batch=1, sequence=1, dtype=torch.bfloat16),
            _head_values(batch=2, sequence=1, dtype=torch.bfloat16).flip(-1),
        )

        actual = router.project_many(mixed, bank.name, layer=2)

        self.assertEqual(len(bank.calls), 1)
        self.assertEqual(tuple(actual[0].shape), (1, 1, 7))
        self.assertEqual(tuple(actual[1].shape), (2, 1, 7))
        self.assertEqual(actual[0].dtype, torch.bfloat16)
        self.assertEqual(actual[1].dtype, torch.bfloat16)
        self.assertIsNotNone(bank.last_dense)
        expected = (
            bank.last_dense.float() @ bank.weight.float().T
        ).to(dtype=torch.bfloat16)
        torch.testing.assert_close(
            torch.cat((actual[0].reshape(1, 7), actual[1].reshape(2, 7))),
            expected,
            rtol=0.0,
            atol=0.0,
        )
        metrics = router.metrics()
        self.assertEqual(metrics["calls"], 1)
        self.assertEqual(metrics["rows"], 3)
        self.assertEqual(metrics["transitions"], 2)

    def test_ties_are_lower_index_stable_and_close_does_not_close_the_bank(
        self,
    ) -> None:
        bank = _ExactSelectedBlockBank()
        router = PackedDeltaHeadRouter(
            bank,  # type: ignore[arg-type]
            active_layers=(2,),
            max_selected_heads=16,
        )
        zeros = torch.zeros((1, 1, 48 * 128), dtype=torch.float32)
        router.project(zeros, bank.name, layer=2)
        selected = bank.calls[-1][1][0].tolist()
        self.assertEqual(
            selected,
            [
                head * 4 + offset
                for head in range(16)
                for offset in range(4)
            ],
        )
        self.assertFalse(router.supports_layer(True))
        self.assertFalse(router.supports_layer(1))
        router.close()
        self.assertFalse(hasattr(bank, "closed"))
        with self.assertRaisesRegex(RuntimeError, "closed"):
            router.project(zeros, bank.name, layer=2)


if __name__ == "__main__":
    unittest.main()
