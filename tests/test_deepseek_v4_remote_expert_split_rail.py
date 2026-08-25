from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from immer.runtimes.deepseek_v4 import DeepSeekWeightPager, StreamedDeepSeekV4
from immer.runtimes.deepseek_v4.model import DeepSeekV4RuntimeError
from immer.runtimes.deepseek_v4.pager import DeepSeekPagerError
from test_deepseek_v4_model import (
    _PrefetchQuantizedTinyCheckpoint,
    _config,
)
from test_deepseek_v4_pager import _EncodedExpertSource


_FINGERPRINT = "d" * 64
_REVISION = "7" * 40


class _LocalTensorSource:
    def __init__(self, fingerprint: str = _FINGERPRINT) -> None:
        self.fingerprint = fingerprint

    def metrics(self) -> dict[str, object]:
        return {
            "inventory_source_fingerprint": self.fingerprint,
            "repo_id": "local:/causal-bundle",
            "revision": _REVISION,
            "revision_is_mutable": True,
            "revision_is_pinned": False,
        }


class _DenseReader:
    def __init__(
        self,
        plans: dict[str, SimpleNamespace],
        payloads: dict[str, bytes] | None = None,
        *,
        fingerprint: str = _FINGERPRINT,
        source: object | None = None,
    ) -> None:
        self.layout = SimpleNamespace(layout_fingerprint=fingerprint)
        self.source = source if source is not None else _LocalTensorSource()
        self.plans = plans
        self.payloads = payloads or {}
        self.resolved: list[str] = []
        self.reads: list[str] = []

    @classmethod
    def from_checkpoint(cls, checkpoint) -> _DenseReader:
        plans: dict[str, SimpleNamespace] = {}
        offset = 8
        for name in sorted(checkpoint.data):
            if ".ffn.experts." in name:
                continue
            meta = checkpoint.find(name)
            begin, end = meta["offset_in_shard"]
            length = int(end) - int(begin)
            plans[name] = SimpleNamespace(
                name=name,
                dtype=meta["dtype"],
                shape=tuple(meta["shape"]),
                shard="dense-local.safetensors",
                absolute_offset=offset,
                length=length,
            )
            offset += length
        return cls(plans)

    def _plan(self, name: str) -> SimpleNamespace:
        if ".ffn.experts." in name:
            raise AssertionError(f"routed expert entered local tensor rail: {name}")
        return self.plans[name]

    def resolve_tensor_plan(self, name: str) -> SimpleNamespace:
        self.resolved.append(name)
        return self._plan(name)

    def resolve_tensor_plans(self, names) -> tuple[SimpleNamespace, ...]:
        materialized = tuple(names)
        self.resolved.extend(materialized)
        return tuple(self._plan(name) for name in materialized)

    def read_tensor_range(
        self,
        name: str,
        *,
        relative_offset: int = 0,
        length: int | None = None,
    ) -> SimpleNamespace:
        plan = self._plan(name)
        if length is None:
            length = plan.length - relative_offset
        return self.read_tensor_ranges(name, ((relative_offset, length),))

    def read_tensor_ranges(
        self,
        name: str,
        ranges,
        *,
        resident_limit_bytes: int | None = None,
        max_gap_bytes: int = 0,
    ) -> SimpleNamespace:
        if max_gap_bytes != 0:
            raise AssertionError("split fixture only accepts exact dense ranges")
        plan = self._plan(name)
        self.reads.append(name)
        requested = tuple((int(offset), int(length)) for offset, length in ranges)
        payload = self.payloads[name]
        parts = tuple(
            memoryview(payload[offset : offset + length]).toreadonly()
            for offset, length in requested
        )
        resident = sum(length for _offset, length in requested)
        if resident_limit_bytes is not None and resident > resident_limit_bytes:
            raise AssertionError("dense split fixture exceeded its resident limit")
        return SimpleNamespace(
            plan=plan,
            ranges=requested,
            parts=parts,
            requested_bytes=resident,
            resident_bytes=resident,
            source_requests=len(parts),
            source_bytes=resident,
        )

    def metrics(self) -> dict[str, object]:
        return {"resolved": len(self.resolved), "reads": len(self.reads)}


def _pin_remote(source, *, fingerprint: str = _FINGERPRINT) -> None:
    original = source.metrics

    def metrics() -> dict[str, object]:
        return {
            **original(),
            "inventory_source_fingerprint": fingerprint,
            "repo_id": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "revision": _REVISION,
            "revision_is_mutable": False,
            "revision_is_pinned": True,
        }

    source.metrics = metrics


def _f32_plan(name: str, value: np.ndarray) -> tuple[SimpleNamespace, bytes]:
    payload = np.ascontiguousarray(value, dtype="<f4").tobytes()
    return (
        SimpleNamespace(
            name=name,
            dtype="F32",
            shape=tuple(value.shape),
            shard="dense-local.safetensors",
            absolute_offset=8,
            length=len(payload),
        ),
        payload,
    )


class RemoteExpertSplitRailTests(unittest.TestCase):
    def test_remote_experts_and_local_dense_use_disjoint_exact_rails(self) -> None:
        import torch

        base = "layers.0.ffn.experts.0"
        remote = _EncodedExpertSource(base, "fp4", adjacent=True)
        _pin_remote(remote)
        dense = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        plan, payload = _f32_plan("dense.weight", dense)
        reader = _DenseReader({"dense.weight": plan}, {"dense.weight": payload})
        pager = DeepSeekWeightPager(
            remote,
            device="cpu",
            compute_dtype="float32",
            simulate_activation_quantization=False,
            expert_prefetch=False,
            causal_tensor_reader=reader,
            remote_expert_split_rail=True,
        )

        actual_dense = pager.linear(
            torch.asarray([[2.0, -1.0]], dtype=torch.float32), "dense"
        )
        actual_expert = pager.expert(
            torch.zeros((1, 128), dtype=torch.float32),
            base,
            swiglu_limit=10.0,
        )

        torch.testing.assert_close(
            actual_dense,
            torch.asarray([[0.0, 2.0]], dtype=torch.float32),
            atol=0,
            rtol=0,
        )
        self.assertEqual(tuple(actual_expert.shape), (1, 128))
        self.assertEqual(reader.resolved, ["dense.weight"])
        self.assertEqual(reader.reads, ["dense.weight"])
        self.assertEqual(len(remote.raw_calls), 2)
        expected_ranges = pager.plan_expert_ranges((base,))[0].ranges
        self.assertEqual(
            remote.raw_calls,
            [
                (item.shard, item.absolute_offset, item.length)
                for item in expected_ranges
            ],
        )
        metrics = pager.metrics()
        self.assertTrue(metrics["remote_expert_split_rail"])
        self.assertEqual(
            metrics["expert_address_plane"], "remote-pinned-streamer/v1"
        )
        self.assertFalse(metrics["causal_weight_reader_attached"])
        self.assertTrue(metrics["causal_tensor_reader_attached"])
        self.assertFalse(metrics["causal_missing_fallback"])
        pager.release()

    def test_split_mode_rejects_invalid_combinations_and_unpinned_identity(
        self,
    ) -> None:
        base = "layers.0.ffn.experts.0"
        remote = _EncodedExpertSource(base, "fp4", adjacent=True)
        _pin_remote(remote)
        plan, payload = _f32_plan("dense.weight", np.eye(2, dtype=np.float32))
        reader = _DenseReader({"dense.weight": plan}, {"dense.weight": payload})

        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            DeepSeekWeightPager(remote, remote_expert_split_rail=1)
        with self.assertRaisesRegex(DeepSeekPagerError, "requires.*tensor reader"):
            DeepSeekWeightPager(remote, remote_expert_split_rail=True)
        with self.assertRaisesRegex(DeepSeekPagerError, "forbids a causal weight"):
            DeepSeekWeightPager(
                remote,
                causal_weight_reader=SimpleNamespace(resolve_expert_plans=lambda *_: ()),
                causal_tensor_reader=reader,
                remote_expert_split_rail=True,
            )
        with self.assertRaisesRegex(DeepSeekPagerError, "forbids missing-route"):
            DeepSeekWeightPager(
                remote,
                causal_tensor_reader=reader,
                causal_missing_fallback=True,
                remote_expert_split_rail=True,
            )

        unpinned = _EncodedExpertSource(base, "fp4", adjacent=True)
        _pin_remote(unpinned)
        pinned_metrics = unpinned.metrics
        unpinned.metrics = lambda: {
            **pinned_metrics(),
            "revision": "main",
            "revision_is_mutable": True,
            "revision_is_pinned": False,
        }
        with self.assertRaisesRegex(DeepSeekPagerError, "authenticated remote pinned"):
            DeepSeekWeightPager(
                unpinned,
                causal_tensor_reader=reader,
                remote_expert_split_rail=True,
            )

        wrong_layout = _DenseReader(
            {"dense.weight": plan},
            {"dense.weight": payload},
            fingerprint="e" * 64,
            source=_LocalTensorSource("e" * 64),
        )
        with self.assertRaisesRegex(DeepSeekPagerError, "fingerprint does not match"):
            DeepSeekWeightPager(
                remote,
                causal_tensor_reader=wrong_layout,
                remote_expert_split_rail=True,
            )

    def test_default_dense_causal_mode_still_requires_an_expert_reader(self) -> None:
        import torch

        base = "layers.0.ffn.experts.0"
        remote = _EncodedExpertSource(base, "fp4", adjacent=True)
        _pin_remote(remote)
        plan, payload = _f32_plan("dense.weight", np.eye(2, dtype=np.float32))
        reader = _DenseReader({"dense.weight": plan}, {"dense.weight": payload})
        pager = DeepSeekWeightPager(
            remote,
            device="cpu",
            causal_tensor_reader=reader,
        )

        with self.assertRaisesRegex(
            DeepSeekPagerError, "requires the separate expert reader"
        ):
            pager.expert(
                torch.zeros((1, 128), dtype=torch.bfloat16),
                base,
                swiglu_limit=10.0,
            )
        self.assertEqual(remote.raw_calls, [])
        self.assertEqual(reader.reads, [])
        self.assertFalse(pager.metrics()["remote_expert_split_rail"])
        pager.release()

    def test_corrupt_remote_expert_metadata_fails_before_payload_or_local_reads(
        self,
    ) -> None:
        import torch

        base = "layers.0.ffn.experts.0"
        remote = _EncodedExpertSource(base, "fp4", adjacent=True)
        remote.meta[f"{base}.w1.weight"]["shape"] = [127, 64]
        _pin_remote(remote)
        plan, payload = _f32_plan("dense.weight", np.eye(2, dtype=np.float32))
        reader = _DenseReader({"dense.weight": plan}, {"dense.weight": payload})
        pager = DeepSeekWeightPager(
            remote,
            device="cpu",
            causal_tensor_reader=reader,
            remote_expert_split_rail=True,
        )

        with self.assertRaisesRegex(DeepSeekPagerError, "byte length is invalid"):
            pager.expert(
                torch.zeros((1, 128), dtype=torch.bfloat16),
                base,
                swiglu_limit=10.0,
            )
        self.assertEqual(remote.raw_calls, [])
        self.assertEqual(reader.reads, [])
        pager.release()

    def test_sample_and_exhaustive_preflight_partition_expert_metadata(self) -> None:
        source = _PrefetchQuantizedTinyCheckpoint()
        _pin_remote(source)
        reader = _DenseReader.from_checkpoint(source)
        pager = DeepSeekWeightPager(
            source,
            device="cpu",
            compute_dtype="bfloat16",
            causal_tensor_reader=reader,
            remote_expert_split_rail=True,
        )
        model = StreamedDeepSeekV4(
            replace(_config(), n_activated_experts=2), pager
        )
        body_before = source.metrics()["network_or_source_body_bytes"]

        sample = model.checkpoint_preflight(exhaustive_experts=False)
        exhaustive = model.checkpoint_preflight(exhaustive_experts=True)

        self.assertEqual(
            source.metrics()["network_or_source_body_bytes"], body_before
        )
        self.assertEqual(source.encoded_experts.raw_calls, [])
        self.assertFalse(any(".ffn.experts." in name for name in reader.resolved))
        self.assertEqual(sample["tensor_address_plane"], "causal-tensor-rail/v1")
        self.assertEqual(
            sample["expert_address_plane"], "remote-pinned-streamer/v1"
        )
        self.assertEqual(sample["required_expert_plans"], 1)
        self.assertEqual(exhaustive["required_expert_plans"], 2)
        self.assertEqual(
            exhaustive["required_tensors"] - sample["required_tensors"], 6
        )
        pager.release()

    def test_preflight_rejects_remote_shape_corruption_without_payload_io(self) -> None:
        source = _PrefetchQuantizedTinyCheckpoint()
        source.encoded_experts.meta[
            "layers.0.ffn.experts.0.w1.weight"
        ]["shape"] = [127, 64]
        _pin_remote(source)
        reader = _DenseReader.from_checkpoint(source)
        pager = DeepSeekWeightPager(
            source,
            device="cpu",
            causal_tensor_reader=reader,
            remote_expert_split_rail=True,
        )
        model = StreamedDeepSeekV4(
            replace(_config(), n_activated_experts=2), pager
        )
        body_before = source.metrics()["network_or_source_body_bytes"]

        with self.assertRaisesRegex(
            DeepSeekV4RuntimeError, "expert-binding preflight failed"
        ):
            model.checkpoint_preflight(exhaustive_experts=False)
        self.assertEqual(
            source.metrics()["network_or_source_body_bytes"], body_before
        )
        self.assertEqual(source.encoded_experts.raw_calls, [])
        self.assertEqual(reader.reads, [])
        pager.release()


if __name__ == "__main__":
    unittest.main()
