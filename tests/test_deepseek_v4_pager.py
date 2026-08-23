from __future__ import annotations

from dataclasses import replace
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np


class _Source:
    def __init__(self) -> None:
        self.data = {
            "dense.weight": np.asarray([[1, 2], [3, 4]], dtype=np.float32),
            "group.weight": np.asarray(
                [[1, 0], [0, 1], [2, 0], [0, 3]], dtype=np.float32
            ),
            "embed.weight": np.arange(20, dtype=np.float32).reshape(10, 2),
            "head.weight": np.asarray(
                [[1, 0], [0, 1], [1, 1], [-1, 0]], dtype=np.float32
            ),
        }
        self.dtypes = {name: "F32" for name in self.data}
        self.row_calls: list[tuple[str, int, int]] = []

    def find(self, name: str) -> dict:
        value = self.data[name]
        itemsize = {
            "F8_E4M3": 1,
            "F8_E8M0": 1,
            "I8": 1,
        }.get(self.dtypes[name], value.dtype.itemsize)
        return {
            "name": name,
            "dtype": self.dtypes[name],
            "shape": list(value.shape),
            "offset_in_shard": [0, value.size * itemsize],
        }

    def tensor(self, name: str) -> np.ndarray:
        return self.data[name].copy()

    def rows(self, name: str, start_row: int = 0, n_rows: int = 8, **_) -> np.ndarray:
        self.row_calls.append((name, start_row, n_rows))
        return self.data[name][start_row : start_row + n_rows].copy()

    def metrics(self) -> dict:
        return {"network_or_source_body_bytes": 0}


class _AdjacentHeadSource:
    """BF16 head whose unchanged compute leaves are physically adjacent."""

    def __init__(
        self,
        *,
        vocab: int = 8,
        columns: int = 4,
        dtype: str = "BF16",
    ) -> None:
        values = np.arange(vocab * columns, dtype=np.float32).reshape(
            vocab, columns
        ) / np.float32(16.0) - np.float32(0.75)
        words = (values.view(np.uint32) >> 16).astype("<u2")
        self.payload = words.tobytes()
        decoded_words = np.frombuffer(self.payload, dtype="<u2").astype(np.uint32)
        decoded_words <<= 16
        self.data = decoded_words.view(np.float32).reshape(vocab, columns).copy()
        self.dtype = dtype
        self.data_start = 16
        self.shard = "head.safetensors"
        self.row_calls: list[tuple[str, int, int]] = []
        self.batch_calls: list[tuple[str, tuple[tuple[int, int], ...], int, int]] = []
        self.envelopes: list[tuple[int, int]] = []

    def find(self, name: str) -> dict:
        if name != "head.weight":
            raise KeyError(name)
        return {
            "name": name,
            "dtype": self.dtype,
            "shape": list(self.data.shape),
            "shard": self.shard,
            "data_start": self.data_start,
            "offset_in_shard": [0, len(self.payload)],
        }

    def rows(self, name: str, start_row: int = 0, n_rows: int = 8, **_) -> np.ndarray:
        self.row_calls.append((name, start_row, n_rows))
        return self.data[start_row : start_row + n_rows].copy()

    def raw_bytes_many(
        self,
        shard: str,
        ranges,
        *,
        resident_limit_bytes: int,
        max_gap_bytes: int = 0,
    ):
        from immer.knowledge.streamer import RawBytesManyResult

        requested = tuple((int(offset), int(length)) for offset, length in ranges)
        self.batch_calls.append((shard, requested, resident_limit_bytes, max_gap_bytes))
        if shard != self.shard or max_gap_bytes != 0:
            raise ValueError("invalid exact head request")
        ordered = sorted(enumerate(requested), key=lambda item: item[1][0])
        envelopes: list[tuple[int, int, list[int]]] = []
        for original, (offset, length) in ordered:
            end = offset + length
            if envelopes and envelopes[-1][1] == offset:
                begin, _, members = envelopes[-1]
                members.append(original)
                envelopes[-1] = (begin, end, members)
            else:
                envelopes.append((offset, end, [original]))
        resident = sum(end - begin for begin, end, _ in envelopes)
        if resident > resident_limit_bytes:
            raise ValueError("head fixture resident limit exceeded")
        parts: list[memoryview | None] = [None] * len(requested)
        for begin, end, members in envelopes:
            self.envelopes.append((begin, end - begin))
            relative = begin - self.data_start
            owner = self.payload[relative : relative + end - begin]
            owner_view = memoryview(owner)
            for original in members:
                offset, length = requested[original]
                parts[original] = owner_view[offset - begin : offset - begin + length]
        if any(part is None for part in parts):
            raise AssertionError("head fixture omitted a range")
        return RawBytesManyResult(
            parts=tuple(part for part in parts if part is not None),
            resident_bytes=resident,
            source_requests=len(envelopes),
            source_bytes=resident,
        )

    def metrics(self) -> dict:
        return {
            "network_or_source_body_bytes": sum(length for _, length in self.envelopes)
        }


class _InvalidHeadReceiptSource(_AdjacentHeadSource):
    def raw_bytes_many(self, shard: str, ranges, **kwargs):
        from immer.knowledge.streamer import RawBytesManyResult

        result = super().raw_bytes_many(shard, ranges, **kwargs)
        return RawBytesManyResult(
            parts=result.parts,
            resident_bytes=result.resident_bytes + 1,
            source_requests=result.source_requests,
            source_bytes=result.source_bytes,
        )


class _WarmHeadSource(_AdjacentHeadSource):
    """Return verified resident leaves without claiming physical source I/O."""

    def raw_bytes_many(self, shard: str, ranges, **kwargs):
        from immer.knowledge.streamer import RawBytesManyResult

        result = super().raw_bytes_many(shard, ranges, **kwargs)
        return RawBytesManyResult(
            parts=result.parts,
            resident_bytes=result.resident_bytes,
            source_requests=0,
            source_bytes=0,
        )


class _HeadCapabilityWithoutLayout(_Source):
    def __init__(self) -> None:
        super().__init__()
        self.batch_calls = 0

    def raw_bytes_many(self, *_args, **_kwargs):
        self.batch_calls += 1
        raise AssertionError("invalid metadata must decline before transport")


class _EncodedExpertSource:
    """Synthetic safetensors source with controllable expert adjacency."""

    def __init__(self, base: str, encoding: str, *, adjacent: bool) -> None:
        self.base = base
        self.tensor_calls: list[str] = []
        self.raw_calls: list[tuple[str, int, int]] = []
        self.data_start = 8
        self.shard_name = "expert.safetensors"
        self.data: dict[str, np.ndarray] = {}
        self.dtypes: dict[str, str] = {}
        encoded: dict[str, bytes] = {}
        for role in ("w1", "w2", "w3"):
            weight_name = f"{base}.{role}.weight"
            scale_name = f"{base}.{role}.scale"
            if encoding == "fp4":
                self.data[weight_name] = np.full((128, 64), 0x22, dtype=np.int8)
                self.data[scale_name] = np.ones((128, 4), dtype=np.float32)
                self.dtypes[weight_name] = "I8"
                encoded[weight_name] = self.data[weight_name].tobytes()
            elif encoding == "fp8":
                self.data[weight_name] = np.ones((128, 128), dtype=np.float32)
                self.data[scale_name] = np.ones((1, 1), dtype=np.float32)
                self.dtypes[weight_name] = "F8_E4M3"
                encoded[weight_name] = bytes([0x38]) * (128 * 128)
            else:  # pragma: no cover - test fixture invariant
                raise ValueError(encoding)
            self.dtypes[scale_name] = "F8_E8M0"
            encoded[scale_name] = bytes([0x7F]) * self.data[scale_name].size

        self.meta: dict[str, dict] = {}
        payload = bytearray()
        for group in ("scale", "weight"):
            if payload:
                payload.extend(b"gap-between-groups")
            for index, role in enumerate(("w1", "w2", "w3")):
                if index and not adjacent:
                    payload.extend(b"!")
                name = f"{base}.{role}.{group}"
                begin = len(payload)
                payload.extend(encoded[name])
                end = len(payload)
                self.meta[name] = {
                    "name": name,
                    "dtype": self.dtypes[name],
                    "shape": list(self.data[name].shape),
                    "shard": self.shard_name,
                    "data_start": self.data_start,
                    "offset_in_shard": [begin, end],
                }
        self.payload = bytes(payload)

    def find(self, name: str) -> dict:
        return dict(self.meta[name])

    def tensor(self, name: str) -> np.ndarray:
        self.tensor_calls.append(name)
        return self.data[name].copy()

    def raw_bytes(self, shard: str, offset: int, length: int) -> bytes:
        self.raw_calls.append((shard, offset, length))
        start = offset - self.data_start
        return self.payload[start : start + length]

    def metrics(self) -> dict:
        return {
            "network_or_source_body_bytes": sum(
                length for _, _, length in self.raw_calls
            )
        }


class _MultiEncodedExpertSource:
    """Combine independently adjacent experts behind one tensor source."""

    def __init__(
        self, bases: list[str], *, adjacent: bool = True, encoding: str = "fp4"
    ) -> None:
        self.children: dict[str, _EncodedExpertSource] = {}
        self.meta: dict[str, dict] = {}
        self.data: dict[str, np.ndarray] = {}
        self.raw_calls: list[tuple[str, int, int]] = []
        self.tensor_calls: list[str] = []
        for index, base in enumerate(bases):
            child = _EncodedExpertSource(base, encoding, adjacent=adjacent)
            shard = f"expert-{index}.safetensors"
            child.shard_name = shard
            for name, record in child.meta.items():
                updated = dict(record)
                updated["shard"] = shard
                self.meta[name] = updated
            self.children[shard] = child
            self.data.update(child.data)

    def find(self, name: str) -> dict:
        return dict(self.meta[name])

    def tensor(self, name: str) -> np.ndarray:
        self.tensor_calls.append(name)
        return self.data[name].copy()

    def raw_bytes(self, shard: str, offset: int, length: int) -> bytes:
        self.raw_calls.append((shard, offset, length))
        child = self.children[shard]
        start = offset - child.data_start
        return child.payload[start : start + length]

    def metrics(self) -> dict:
        return {
            "network_or_source_body_bytes": sum(
                length for _, _, length in self.raw_calls
            )
        }


class _AccessTracingExpertSource(_MultiEncodedExpertSource):
    """Emit Streamer-shaped observations from real prefetch worker reads."""

    def __init__(self, bases, recorder) -> None:
        super().__init__(bases)
        self.recorder = recorder
        self._access_sequence = 0
        self._access_lock = threading.Lock()

    def raw_bytes(self, shard: str, offset: int, length: int) -> bytes:
        from immer.knowledge.access_trace import AccessLeaf, AccessOperation

        payload = super().raw_bytes(shard, offset, length)
        with self._access_lock:
            self._access_sequence += 1
            sequence = self._access_sequence
        self.recorder.observe(
            AccessOperation(
                repo_id="local:pager-context-fixture",
                revision="fixture",
                inventory_fingerprint="0" * 64,
                operation="raw_bytes",
                operation_sequence=sequence,
                thread_id=threading.get_ident(),
                thread_name=threading.current_thread().name,
                leaves=(AccessLeaf(shard, offset, length),),
                source_requests=1,
                source_bytes=length,
                cache_hits=0,
            )
        )
        return payload


class _AdjacentBatchExpertSource:
    """Three experts where E0/E1 share exact physical range boundaries."""

    def __init__(self, bases: list[str]) -> None:
        if len(bases) < 3:
            raise ValueError("adjacent batch fixture requires at least three experts")
        self.data_start = 8
        self.shard_name = "adjacent-experts.safetensors"
        self.raw_calls: list[tuple[str, int, int]] = []
        self.tensor_calls: list[str] = []
        self.meta: dict[str, dict] = {}
        self.data: dict[str, np.ndarray] = {}
        children = [_EncodedExpertSource(base, "fp4", adjacent=True) for base in bases]
        self.data.update(
            (name, value) for child in children for name, value in child.data.items()
        )

        payload = bytearray()
        for group in ("scale", "weight"):
            if payload:
                payload.extend(b"gap-between-scale-and-weight-groups")
            for index, child in enumerate(children):
                if index >= 2:
                    payload.extend(f"gap-before-expert-{index}".encode("ascii"))
                names = [f"{child.base}.{role}.{group}" for role in ("w1", "w2", "w3")]
                group_begin = min(
                    int(child.meta[name]["offset_in_shard"][0]) for name in names
                )
                group_end = max(
                    int(child.meta[name]["offset_in_shard"][1]) for name in names
                )
                target_begin = len(payload)
                payload.extend(child.payload[group_begin:group_end])
                for name in names:
                    record = dict(child.meta[name])
                    begin, end = (int(value) for value in record["offset_in_shard"])
                    record["shard"] = self.shard_name
                    record["data_start"] = self.data_start
                    record["offset_in_shard"] = [
                        target_begin + begin - group_begin,
                        target_begin + end - group_begin,
                    ]
                    self.meta[name] = record
        self.payload = bytes(payload)
        self.plan_ranges: dict[str, tuple[tuple[int, int], ...]] = {}
        for base in bases:
            layouts = []
            for group in ("scale", "weight"):
                names = [f"{base}.{role}.{group}" for role in ("w1", "w2", "w3")]
                begin = min(
                    int(self.meta[name]["offset_in_shard"][0]) for name in names
                )
                end = max(int(self.meta[name]["offset_in_shard"][1]) for name in names)
                layouts.append((self.data_start + begin, end - begin))
            self.plan_ranges[base] = tuple(layouts)

    def find(self, name: str) -> dict:
        return dict(self.meta[name])

    def tensor(self, name: str) -> np.ndarray:
        self.tensor_calls.append(name)
        return self.data[name].copy()

    def raw_bytes(self, shard: str, offset: int, length: int) -> bytes:
        if shard != self.shard_name:
            raise KeyError(shard)
        self.raw_calls.append((shard, offset, length))
        start = offset - self.data_start
        return self.payload[start : start + length]

    def raw_bytes_many(
        self,
        shard: str,
        ranges,
        *,
        resident_limit_bytes: int,
        max_gap_bytes: int = 0,
    ):
        from immer.knowledge.streamer import RawBytesManyResult

        if max_gap_bytes != 0:
            raise ValueError("fixture only supports exact adjacency")
        requested = tuple((int(offset), int(length)) for offset, length in ranges)
        ordered = sorted(enumerate(requested), key=lambda item: item[1][0])
        envelopes: list[tuple[int, int, list[int]]] = []
        for original, (offset, length) in ordered:
            end = offset + length
            if envelopes and envelopes[-1][1] == offset:
                begin, _, members = envelopes[-1]
                members.append(original)
                envelopes[-1] = (begin, end, members)
            else:
                envelopes.append((offset, end, [original]))
        resident = sum(end - begin for begin, end, _ in envelopes)
        if resident > resident_limit_bytes:
            raise ValueError("fixture resident limit exceeded")
        parts: list[memoryview | None] = [None] * len(requested)
        for begin, end, members in envelopes:
            owner = self.raw_bytes(shard, begin, end - begin)
            owner_view = memoryview(owner)
            for original in members:
                offset, length = requested[original]
                parts[original] = owner_view[offset - begin : offset - begin + length]
        if any(part is None for part in parts):
            raise AssertionError("fixture omitted requested range")
        return RawBytesManyResult(
            parts=tuple(part for part in parts if part is not None),
            resident_bytes=resident,
            source_requests=len(envelopes),
            source_bytes=resident,
        )

    def metrics(self) -> dict:
        return {
            "network_or_source_body_bytes": sum(
                length for _, _, length in self.raw_calls
            )
        }


class _BlockingAdjacentBatchExpertSource(_AdjacentBatchExpertSource):
    """Event-driven batch source exposing pair ownership and cancellation."""

    def __init__(self, bases: list[str]) -> None:
        super().__init__(bases)
        self.batch_keys = [(bases[0], bases[1]), *((base,) for base in bases[2:])]
        self.started = {key: threading.Event() for key in self.batch_keys}
        self.release = {key: threading.Event() for key in self.batch_keys}
        self.completed = {key: threading.Event() for key in self.batch_keys}
        self.fail: set[tuple[str, ...]] = set()
        self.max_active = 0
        self._active: set[tuple[str, ...]] = set()
        self._batch_lock = threading.Lock()

    def _batch_key(self, ranges: tuple[tuple[int, int], ...]) -> tuple[str, ...]:
        requested = set(ranges)
        members = tuple(
            base
            for base, planned in self.plan_ranges.items()
            if requested.intersection(planned)
        )
        if members not in self.started:
            raise AssertionError(f"unexpected batch members: {members}")
        return members

    def raw_bytes_many(
        self,
        shard: str,
        ranges,
        *,
        resident_limit_bytes: int,
        max_gap_bytes: int = 0,
    ):
        requested = tuple((int(offset), int(length)) for offset, length in ranges)
        key = self._batch_key(requested)
        with self._batch_lock:
            self._active.add(key)
            self.max_active = max(self.max_active, len(self._active))
            self.started[key].set()
        if not self.release[key].wait(timeout=5):
            raise RuntimeError(f"test did not release adjacent batch {key}")
        try:
            if key in self.fail:
                raise RuntimeError(f"simulated adjacent batch failure {key}")
            return super().raw_bytes_many(
                shard,
                requested,
                resident_limit_bytes=resident_limit_bytes,
                max_gap_bytes=max_gap_bytes,
            )
        finally:
            with self._batch_lock:
                self._active.discard(key)
                self.completed[key].set()


class _BlockingExpertSource(_MultiEncodedExpertSource):
    def __init__(self, bases: list[str]) -> None:
        super().__init__(bases)
        self.blocked_shard: str | None = None
        self.started = threading.Event()
        self.release_read = threading.Event()
        self.completed = threading.Event()
        self.fail = False

    def raw_bytes(self, shard: str, offset: int, length: int) -> bytes:
        if shard == self.blocked_shard:
            self.started.set()
            if not self.release_read.wait(timeout=5):
                raise RuntimeError("test did not release blocked expert read")
            if self.fail:
                raise RuntimeError("simulated expert prefetch failure")
        result = super().raw_bytes(shard, offset, length)
        if shard == self.blocked_shard:
            blocked_calls = sum(
                observed_shard == shard for observed_shard, _, _ in self.raw_calls
            )
            if blocked_calls == 2:
                self.completed.set()
        return result


class _WindowExpertSource(_MultiEncodedExpertSource):
    """Per-expert barriers exposing exact I/O concurrency without sleeps."""

    def __init__(self, bases: list[str]) -> None:
        super().__init__(bases)
        self.main_thread = threading.get_ident()
        self.find_threads: list[int] = []
        self.raw_threads: list[int] = []
        self.started = {base: threading.Event() for base in bases}
        self.release = {base: threading.Event() for base in bases}
        self.completed = {base: threading.Event() for base in bases}
        self.fail: set[str] = set()
        self.started_order: list[str] = []
        self.completed_order: list[str] = []
        self.max_active = 0
        self._active: set[str] = set()
        self._calls: dict[str, int] = {base: 0 for base in bases}
        self._base_by_shard = {
            f"expert-{index}.safetensors": base for index, base in enumerate(bases)
        }
        self._lock = threading.Lock()

    def find(self, name: str) -> dict:
        self.find_threads.append(threading.get_ident())
        return super().find(name)

    def raw_bytes(self, shard: str, offset: int, length: int) -> bytes:
        base = self._base_by_shard[shard]
        self.raw_threads.append(threading.get_ident())
        with self._lock:
            first = self._calls[base] == 0
            self._calls[base] += 1
            if first:
                self._active.add(base)
                self.max_active = max(self.max_active, len(self._active))
                self.started_order.append(base)
                self.started[base].set()
        if not self.release[base].wait(timeout=5):
            raise RuntimeError(f"test did not release {base}")
        if base in self.fail:
            with self._lock:
                self._active.discard(base)
                self.completed_order.append(base)
                self.completed[base].set()
            raise RuntimeError(f"simulated read failure for {base}")
        result = super().raw_bytes(shard, offset, length)
        with self._lock:
            if self._calls[base] == 2:
                self._active.discard(base)
                self.completed_order.append(base)
                self.completed[base].set()
        return result


class DeepSeekV4PagerTests(unittest.TestCase):
    @staticmethod
    def _manual_expert(pager, x, base: str, route_weight=None):
        import torch

        gate = pager.linear(x, f"{base}.w1").float()
        up = pager.linear(x, f"{base}.w3").float()
        hidden = torch.nn.functional.silu(torch.clamp(gate, max=10.0)) * torch.clamp(
            up, min=-10.0, max=10.0
        )
        if route_weight is not None:
            hidden = hidden * route_weight
        return pager.linear(hidden.to(x.dtype), f"{base}.w2")

    def test_public_expert_range_plans_have_unbounded_fanout_and_zero_payload_io(
        self,
    ) -> None:
        from dataclasses import FrozenInstanceError

        from immer.runtimes.deepseek_v4 import (
            DeepSeekWeightPager,
            ExpertSourceRange,
            ExpertTensorLayout,
            OfficialExpertRangePlan,
        )

        bases = [f"layers.17.ffn.experts.{value}" for value in range(8)]
        source = _MultiEncodedExpertSource(bases)
        source.raw_bytes_many = mock.Mock(
            side_effect=AssertionError("metadata planning must not read payloads")
        )
        pager = DeepSeekWeightPager(
            source,
            device="cpu",
            compute_dtype="bfloat16",
        )
        with (
            mock.patch.object(
                pager,
                "prefetch_expert_window",
                side_effect=AssertionError("metadata planning must not prefetch"),
            ) as prefetch_window,
            mock.patch.object(
                pager,
                "prefetch_expert",
                side_effect=AssertionError("metadata planning must not prefetch"),
            ) as prefetch_one,
        ):
            plans = pager.plan_expert_ranges(
                base for base in (*bases, bases[3], bases[0])
            )

        self.assertGreater(len(plans), pager.EXPERT_PREFETCH_MAX_EXPERTS)
        self.assertEqual([plan.base for plan in plans], bases)
        self.assertTrue(
            all(isinstance(plan, OfficialExpertRangePlan) for plan in plans)
        )
        self.assertTrue(
            all(
                isinstance(source_range, ExpertSourceRange)
                and all(
                    isinstance(tensor, ExpertTensorLayout)
                    for tensor in source_range.tensors
                )
                for plan in plans
                for source_range in plan.ranges
            )
        )
        self.assertEqual(source.raw_calls, [])
        self.assertEqual(source.tensor_calls, [])
        source.raw_bytes_many.assert_not_called()
        prefetch_window.assert_not_called()
        prefetch_one.assert_not_called()
        with self.assertRaises(FrozenInstanceError):
            plans[0].payload_bytes = 0
        pager.release()

    def test_public_expert_range_plan_preserves_exact_tensor_layouts(self) -> None:
        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        base = "layers.9.ffn.experts.23"
        source = _EncodedExpertSource(base, "fp4", adjacent=True)
        # Planning is metadata-only even when the source has no payload-reader
        # capability at all.
        source.raw_bytes = None
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="bfloat16")
        (plan,) = pager.plan_expert_ranges((base,))

        self.assertEqual((plan.base, plan.layer, plan.expert_id), (base, 9, 23))
        self.assertEqual(len(plan.ranges), 2)
        expected_payload_bytes = 0
        for source_range, group in zip(
            plan.ranges,
            ("scale", "weight"),
            strict=True,
        ):
            names = tuple(f"{base}.{role}.{group}" for role in ("w1", "w2", "w3"))
            metas = tuple(source.meta[name] for name in names)
            begins = tuple(int(meta["offset_in_shard"][0]) for meta in metas)
            ends = tuple(int(meta["offset_in_shard"][1]) for meta in metas)
            expected_offset = source.data_start + min(begins)
            expected_length = max(ends) - min(begins)
            expected_payload_bytes += expected_length
            self.assertEqual(source_range.shard, source.shard_name)
            self.assertEqual(source_range.absolute_offset, expected_offset)
            self.assertEqual(
                source_range.absolute_end, expected_offset + expected_length
            )
            self.assertEqual(source_range.length, expected_length)
            self.assertEqual(
                tuple(tensor.name for tensor in source_range.tensors), names
            )
            for tensor, meta in zip(source_range.tensors, metas, strict=True):
                begin, end = (int(value) for value in meta["offset_in_shard"])
                absolute = source.data_start + begin
                self.assertEqual(tensor.dtype, str(meta["dtype"]).upper())
                self.assertEqual(tensor.shape, tuple(meta["shape"]))
                self.assertEqual(tensor.absolute_offset, absolute)
                self.assertEqual(tensor.absolute_end, source.data_start + end)
                self.assertEqual(tensor.length, end - begin)
                self.assertEqual(tensor.range_offset, absolute - expected_offset)
        self.assertEqual(plan.payload_bytes, expected_payload_bytes)
        self.assertEqual(source.tensor_calls, [])
        pager.release()

    def test_public_expert_range_plans_accept_layer_ids_and_reject_bad_routing(
        self,
    ) -> None:
        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        ids = [5, 2, 7]
        bases = [f"layers.4.ffn.experts.{expert_id}" for expert_id in ids]
        source = _MultiEncodedExpertSource(bases)
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="bfloat16")
        plans = pager.plan_expert_ranges(4, (value for value in (5, 2, 5, 7)))
        self.assertEqual([plan.expert_id for plan in plans], ids)
        self.assertEqual([plan.base for plan in plans], bases)
        self.assertEqual(plans, pager.plan_expert_ranges(bases))

        find = mock.Mock(wraps=source.find)
        source.find = find
        bad_calls = (
            lambda: pager.plan_expert_ranges(
                ("layers.4.ffn.experts.5", "layers.5.ffn.experts.2")
            ),
            lambda: pager.plan_expert_ranges(("layers.4.ffn.shared_experts.0",)),
            lambda: pager.plan_expert_ranges(("layers.04.ffn.experts.5",)),
            lambda: pager.plan_expert_ranges(-1, (5,)),
            lambda: pager.plan_expert_ranges(4, (True,)),
            lambda: pager.plan_expert_ranges(4, (-1,)),
        )
        for bad_call in bad_calls:
            with self.subTest(call=bad_call), self.assertRaises(ValueError):
                bad_call()
        find.assert_not_called()
        self.assertEqual(source.raw_calls, [])
        self.assertEqual(source.tensor_calls, [])
        pager.release()

    def test_causal_reader_is_the_live_routed_expert_address_path(self) -> None:
        import torch

        from immer.knowledge.livecausal import LiveGraph
        from immer.runtimes.deepseek_v4 import (
            CausalWeightLayoutIdentity,
            CausalWeightReader,
            DeepSeekWeightPager,
            LogicalModelIdentity,
            bind_causal_weight_plans,
        )

        bases = [f"layers.3.ffn.experts.{expert_id}" for expert_id in (4, 9)]
        source = _MultiEncodedExpertSource(bases)
        bootstrap = DeepSeekWeightPager(
            source,
            device="cpu",
            compute_dtype="bfloat16",
        )
        plans = bootstrap.plan_expert_ranges(bases)
        x = torch.linspace(-0.01, 0.01, 128, dtype=torch.bfloat16)[None, :]
        route_weights = (
            torch.asarray([[0.25]], dtype=torch.float32),
            torch.asarray([[0.75]], dtype=torch.float32),
        )
        expected = tuple(
            bootstrap.expert(
                x,
                base,
                route_weight=route_weight,
                swiglu_limit=10.0,
            )
            for base, route_weight in zip(bases, route_weights, strict=True)
        )

        fingerprint = "d" * 64
        original_metrics = source.metrics
        source.metrics = lambda: {
            **original_metrics(),
            "inventory_source_fingerprint": fingerprint,
        }
        layout = CausalWeightLayoutIdentity(
            model=LogicalModelIdentity(
                repo_id="deepseek-ai/DeepSeek-V4-Flash",
                revision="fixture-revision",
            ),
            layout_fingerprint=fingerprint,
        )
        with tempfile.TemporaryDirectory() as temporary:
            writer = LiveGraph(temporary)
            bind_causal_weight_plans(writer, layout, plans[:1])

            # Remount the durable graph before attaching it to the pager.  The
            # second binding is appended later through the original writer and
            # becomes usable without rebuilding either pager or reader.
            reader = CausalWeightReader(
                LiveGraph(temporary),
                layout,
                source=source,
            )
            causal = DeepSeekWeightPager(
                source,
                device="cpu",
                compute_dtype="bfloat16",
            )
            causal.attach_causal_weight_reader(reader)

            original_find = source.find

            def reject_routed_expert_discovery(name: str) -> dict:
                if name.startswith("layers.3.ffn.experts."):
                    raise AssertionError("routed expert re-entered source.find")
                return original_find(name)

            with mock.patch.object(
                source,
                "find",
                side_effect=reject_routed_expert_discovery,
            ) as find:
                actual_first = causal.expert(
                    x,
                    bases[0],
                    route_weight=route_weights[0],
                    swiglu_limit=10.0,
                )
                bind_causal_weight_plans(writer, layout, plans[1:])
                actual_second = causal.expert(
                    x,
                    bases[1],
                    route_weight=route_weights[1],
                    swiglu_limit=10.0,
                )

            torch.testing.assert_close(actual_first, expected[0], atol=0, rtol=0)
            torch.testing.assert_close(actual_second, expected[1], atol=0, rtol=0)
            self.assertEqual(tuple(plan.expert_id for plan in plans), (4, 9))
            find.assert_not_called()
            metrics = causal.metrics()
            self.assertTrue(metrics["causal_weight_reader_attached"])
            self.assertFalse(metrics["causal_missing_fallback"])
            self.assertEqual(metrics["causal_expert_plan_resolves"], 2)
            self.assertEqual(metrics["causal_expert_plan_hits"], 2)
            self.assertEqual(metrics["causal_expert_plan_misses"], 0)
            self.assertEqual(metrics["causal_expert_plan_fallbacks"], 0)
            self.assertGreaterEqual(reader.metrics()["plan_cache_invalidations"], 1)
            causal.release()
        bootstrap.release()

    def test_causal_missing_binding_fails_closed_unless_explicitly_enabled(
        self,
    ) -> None:
        import torch

        from immer.knowledge.livecausal import LiveGraph
        from immer.runtimes.deepseek_v4 import (
            CausalWeightLayoutIdentity,
            CausalWeightReader,
            DeepSeekWeightPager,
            LogicalModelIdentity,
        )
        from immer.runtimes.deepseek_v4.pager import DeepSeekPagerError

        base = "layers.3.ffn.experts.2"
        source = _EncodedExpertSource(base, "fp4", adjacent=True)
        layout = CausalWeightLayoutIdentity(
            model=LogicalModelIdentity("fixture/model", "fixture-revision"),
            layout_fingerprint="e" * 64,
        )
        original_metrics = source.metrics
        source.metrics = lambda: {
            **original_metrics(),
            "inventory_source_fingerprint": layout.layout_fingerprint,
        }
        x = torch.zeros((1, 128), dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as temporary:
            reader = CausalWeightReader(
                LiveGraph(temporary),
                layout,
                source=source,
            )
            strict = DeepSeekWeightPager(
                source,
                device="cpu",
                compute_dtype="bfloat16",
                causal_weight_reader=reader,
            )
            with mock.patch.object(
                source,
                "find",
                side_effect=AssertionError("strict causal miss searched metadata"),
            ) as strict_find:
                with self.assertRaisesRegex(
                    DeepSeekPagerError,
                    "causal weight binding is missing",
                ):
                    strict.expert(x, base, swiglu_limit=10.0)
            strict_find.assert_not_called()
            self.assertEqual(strict.metrics()["causal_expert_plan_misses"], 1)
            self.assertEqual(strict.metrics()["causal_expert_plan_fallbacks"], 0)

            fallback = DeepSeekWeightPager(
                source,
                device="cpu",
                compute_dtype="bfloat16",
                causal_weight_reader=reader,
                causal_missing_fallback=True,
            )
            find = mock.Mock(wraps=source.find)
            source.find = find
            output = fallback.expert(x, base, swiglu_limit=10.0)
            self.assertEqual(tuple(output.shape), (1, 128))
            self.assertEqual(find.call_count, 6)
            fallback_metrics = fallback.metrics()
            self.assertEqual(fallback_metrics["causal_expert_plan_misses"], 1)
            self.assertEqual(fallback_metrics["causal_expert_plan_fallbacks"], 1)
            strict.release()
            fallback.release()

    def test_causal_plan_preserves_valid_repacked_physical_tensor_order(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        base = "layers.5.ffn.experts.8"
        source = _EncodedExpertSource(base, "fp4", adjacent=True)
        bootstrap = DeepSeekWeightPager(
            source,
            device="cpu",
            compute_dtype="bfloat16",
        )
        (plan,) = bootstrap.plan_expert_ranges((base,))
        x = torch.linspace(-0.01, 0.01, 128, dtype=torch.bfloat16)[None, :]
        expected = bootstrap.expert(x, base, swiglu_limit=10.0)

        repacked_ranges = []
        for source_range in plan.ranges:
            original_slots = source_range.tensors
            role_order = (
                original_slots[1],
                original_slots[0],
                original_slots[2],
            )
            repacked_ranges.append(
                replace(
                    source_range,
                    tensors=tuple(
                        replace(
                            tensor,
                            absolute_offset=(
                                source_range.absolute_offset + slot.range_offset
                            ),
                            range_offset=slot.range_offset,
                        )
                        for tensor, slot in zip(
                            role_order,
                            original_slots,
                            strict=True,
                        )
                    ),
                )
            )
        repacked = replace(plan, ranges=tuple(repacked_ranges))

        class _RepackedResolver:
            def resolve_expert_plans(self, _layer, _expert_ids):
                return (repacked,)

        pager = DeepSeekWeightPager(
            source,
            device="cpu",
            compute_dtype="bfloat16",
            causal_weight_reader=_RepackedResolver(),
        )
        with mock.patch.object(
            source,
            "find",
            side_effect=AssertionError("valid causal repack searched metadata"),
        ) as find:
            actual = pager.expert(x, base, swiglu_limit=10.0)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        find.assert_not_called()
        self.assertEqual(pager.metrics()["causal_expert_plan_hits"], 1)
        bootstrap.release()
        pager.release()

    def test_causal_bad_layout_never_falls_back_to_metadata_discovery(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager
        from immer.runtimes.deepseek_v4.pager import DeepSeekPagerError

        base = "layers.6.ffn.experts.1"
        source = _EncodedExpertSource(base, "fp4", adjacent=True)
        bootstrap = DeepSeekWeightPager(
            source,
            device="cpu",
            compute_dtype="bfloat16",
        )
        (plan,) = bootstrap.plan_expert_ranges((base,))
        first_range = plan.ranges[0]
        first_tensor = first_range.tensors[0]
        corrupt = replace(
            plan,
            ranges=(
                replace(
                    first_range,
                    tensors=(
                        replace(
                            first_tensor,
                            absolute_offset=first_tensor.absolute_offset + 1,
                            range_offset=1,
                        ),
                        *first_range.tensors[1:],
                    ),
                ),
                plan.ranges[1],
            ),
        )

        class _CorruptResolver:
            def resolve_expert_plans(self, _layer, _expert_ids):
                return (corrupt,)

        pager = DeepSeekWeightPager(
            source,
            device="cpu",
            compute_dtype="bfloat16",
            causal_weight_reader=_CorruptResolver(),
            causal_missing_fallback=True,
        )
        with mock.patch.object(
            source,
            "find",
            side_effect=AssertionError("invalid causal layout searched metadata"),
        ) as find:
            with self.assertRaisesRegex(
                DeepSeekPagerError,
                "tensor offsets are invalid",
            ):
                pager.expert(
                    torch.zeros((1, 128), dtype=torch.bfloat16),
                    base,
                    swiglu_limit=10.0,
                )
        find.assert_not_called()
        metrics = pager.metrics()
        self.assertEqual(metrics["causal_expert_plan_invalid"], 1)
        self.assertEqual(metrics["causal_expert_plan_fallbacks"], 0)
        bootstrap.release()
        pager.release()

    def test_fp4_routed_expert_coalesces_to_two_ranges_with_exact_parity(
        self,
    ) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager
        from immer.runtimes.deepseek_v4.quantization import (
            quantize_fp8_e4m3_parts,
        )

        base = "layers.0.ffn.experts.7"
        x = torch.linspace(-0.01, 0.01, 128, dtype=torch.bfloat16)[None, :]
        route_weight = torch.asarray([[0.25]], dtype=torch.float32)

        reference = DeepSeekWeightPager(
            _EncodedExpertSource(base, "fp4", adjacent=False),
            device="cpu",
            compute_dtype="bfloat16",
        )
        expected = self._manual_expert(reference, x, base, route_weight)
        fallback_source = _EncodedExpertSource(base, "fp4", adjacent=False)
        fallback = DeepSeekWeightPager(
            fallback_source, device="cpu", compute_dtype="bfloat16"
        )
        fallback_actual = fallback.expert(
            x, base, route_weight=route_weight, swiglu_limit=10.0
        )

        source = _EncodedExpertSource(base, "fp4", adjacent=True)
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="bfloat16")
        with mock.patch(
            "immer.runtimes.deepseek_v4.pager.quantize_fp8_e4m3_parts",
            wraps=quantize_fp8_e4m3_parts,
        ) as activation_quantize:
            actual = pager.expert(x, base, route_weight=route_weight, swiglu_limit=10.0)

        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(fallback_actual, expected, atol=0, rtol=0)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(activation_quantize.call_count, 2)
        self.assertEqual(len(source.raw_calls), 2)
        self.assertEqual(source.tensor_calls, [])
        self.assertEqual(len(fallback_source.tensor_calls), 6)
        self.assertEqual(fallback_source.raw_calls, [])
        metrics = pager.metrics()
        self.assertEqual(metrics["linear_calls"], 3)
        self.assertEqual(metrics["expert_calls"], 1)
        self.assertEqual(metrics["coalesced_expert_calls"], 1)
        self.assertEqual(metrics["expert_source_ranges"], 2)
        self.assertEqual(metrics["peak_single_weight_bytes"], 128 * 128 * 4)
        self.assertEqual(metrics["materialized_float_bytes"], 3 * 128 * 128 * 4)
        self.assertEqual(metrics["materialized_scale_bytes"], 3 * 128 * 4 * 4)
        self.assertEqual(
            metrics["logical_weight_bytes"],
            sum(
                end - begin
                for meta in source.meta.values()
                for begin, end in [meta["offset_in_shard"]]
            ),
        )

    def test_fp8_shared_expert_coalesces_to_two_ranges_with_exact_parity(
        self,
    ) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        base = "layers.0.ffn.shared_experts"
        x = torch.linspace(-0.01, 0.01, 128, dtype=torch.float32)[None, :]

        reference = DeepSeekWeightPager(
            _EncodedExpertSource(base, "fp8", adjacent=False),
            device="cpu",
            compute_dtype="float32",
        )
        expected = self._manual_expert(reference, x, base)
        fallback_source = _EncodedExpertSource(base, "fp8", adjacent=False)
        fallback = DeepSeekWeightPager(
            fallback_source, device="cpu", compute_dtype="float32"
        )
        fallback_actual = fallback.expert(x, base, swiglu_limit=10.0)

        source = _EncodedExpertSource(base, "fp8", adjacent=True)
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="float32")
        actual = pager.expert(x, base, swiglu_limit=10.0)

        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(fallback_actual, expected, atol=0, rtol=0)
        self.assertEqual(len(source.raw_calls), 2)
        self.assertEqual(source.tensor_calls, [])
        self.assertEqual(len(fallback_source.tensor_calls), 6)
        self.assertEqual(fallback_source.raw_calls, [])
        self.assertEqual(pager.metrics()["linear_calls"], 3)

    def test_exact_window_queues_three_runs_two_and_reverse_completion_is_ordered(
        self,
    ) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        bases = [f"layers.0.ffn.experts.{value}" for value in range(4)]
        source = _WindowExpertSource(bases)
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="bfloat16")
        window = pager.prefetch_expert_window(bases)
        assert window is not None
        self.assertTrue(source.started[bases[0]].wait(timeout=2))
        self.assertTrue(source.started[bases[1]].wait(timeout=2))
        self.assertFalse(source.started[bases[2]].is_set())
        self.assertEqual(source.max_active, 2)

        # Expert 1 may finish first, but ordered consumption still blocks on 0.
        source.release[bases[1]].set()
        self.assertTrue(source.completed[bases[1]].wait(timeout=2))
        self.assertTrue(source.started[bases[2]].wait(timeout=2))
        self.assertEqual(source.max_active, 2)
        entered = threading.Event()
        consumed = threading.Event()
        payloads: list[object] = []

        def consume_first() -> None:
            entered.set()
            payloads.append(pager.consume_expert_window(window, bases[0]))
            consumed.set()

        consumer = threading.Thread(target=consume_first)
        consumer.start()
        self.assertTrue(entered.wait(timeout=1))
        self.assertFalse(consumed.is_set())
        source.release[bases[0]].set()
        self.assertTrue(consumed.wait(timeout=2))
        consumer.join(timeout=1)
        self.assertFalse(consumer.is_alive())

        x = torch.zeros((1, 128), dtype=torch.bfloat16)
        outputs = [pager.expert(x, bases[0], prefetched_payload=payloads[0])]
        payload = pager.consume_expert_window(window, bases[1])
        outputs.append(pager.expert(x, bases[1], prefetched_payload=payload))
        self.assertTrue(source.started[bases[3]].wait(timeout=2))
        source.release[bases[2]].set()
        source.release[bases[3]].set()
        for base in bases[2:]:
            payload = pager.consume_expert_window(window, base)
            outputs.append(pager.expert(x, base, prefetched_payload=payload))
        pager.close_expert_window(window)

        self.assertTrue(all(torch.equal(outputs[0], value) for value in outputs[1:]))
        self.assertLessEqual(source.max_active, 2)
        self.assertEqual(len(source.raw_calls), 2 * len(bases))
        self.assertEqual(set(source.find_threads), {source.main_thread})
        self.assertNotIn(source.main_thread, set(source.raw_threads))
        metrics = pager.metrics()
        self.assertEqual(metrics["expert_prefetch_max_outstanding"], 3)
        self.assertEqual(metrics["expert_prefetch_peak_bytes"], 3 * 26112)
        self.assertEqual(metrics["expert_prefetch_workers"], 2)
        self.assertEqual(metrics["expert_prefetch_active_read_limit"], 2)
        self.assertEqual(
            metrics["expert_prefetch_transport_policy"],
            "streamer-exact-range/v1",
        )
        self.assertEqual(
            metrics["expert_prefetch_policy"],
            "exact-router-window-q3-a2/v2",
        )
        self.assertEqual(metrics["expert_prefetch_max_experts"], 3)
        self.assertEqual(metrics["expert_prefetch_resident_limit_bytes"], 48 * 1024**2)
        pager.release()

    def test_prefetch_workers_inherit_access_trace_scope_for_every_submission(
        self,
    ) -> None:
        from immer.knowledge.access_trace import AccessTraceRecorder
        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        bases = [f"layers.11.ffn.experts.{value}" for value in range(4)]
        recorder = AccessTraceRecorder()
        source = _AccessTracingExpertSource(bases, recorder)
        pager = DeepSeekWeightPager(
            source,
            device="cpu",
            compute_dtype="bfloat16",
        )

        with recorder.scope(
            request_id="request-7",
            phase="layer",
            layer=11,
            attempt=2,
        ):
            window = pager.prefetch_expert_window(bases)
            assert window is not None
        # The fourth read is submitted only after consuming the first ticket.
        # Consume under a conflicting scope to prove the window retains the
        # opening request context across its full sliding lifecycle.
        with recorder.scope(
            request_id="wrong-consumer",
            phase="outside-layer",
            layer=99,
            attempt=9,
        ):
            for base in bases:
                payload = pager.consume_expert_window(window, base)
                pager.discard_expert_payload(payload)
            pager.close_expert_window(window)

        trace = recorder.snapshot()
        self.assertEqual(len(trace.operations), 2 * len(bases))
        self.assertTrue(
            all(
                dict(operation.tags)
                == {
                    "attempt": 2,
                    "layer": 11,
                    "phase": "layer",
                    "request_id": "request-7",
                }
                for operation in trace.operations
            )
        )
        self.assertTrue(
            all(
                operation.thread_name.startswith("immer-v4-expert-prefetch")
                for operation in trace.operations
            )
        )
        self.assertEqual(pager.metrics()["expert_prefetch_submitted"], 4)
        pager.release()

    def test_prefetch_identity_tracks_disabled_q3_and_adjacent_modes(self) -> None:
        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        source = _Source()
        disabled = DeepSeekWeightPager(
            source,
            device="cpu",
            compute_dtype="float32",
            expert_prefetch=False,
        )
        self.assertEqual(disabled.expert_prefetch_policy, "disabled")
        self.assertEqual(disabled.expert_prefetch_transport_policy, "disabled")
        self.assertEqual(disabled.metrics()["expert_prefetch_policy"], "disabled")
        self.assertEqual(
            disabled.metrics()["expert_prefetch_transport_policy"], "disabled"
        )

        q3 = DeepSeekWeightPager(source, device="cpu", compute_dtype="float32")
        self.assertEqual(q3.expert_prefetch_policy, "exact-router-window-q3-a2/v2")
        self.assertEqual(q3.expert_prefetch_transport_policy, "streamer-exact-range/v1")

        adjacent = DeepSeekWeightPager(
            source,
            device="cpu",
            compute_dtype="float32",
            expert_range_coalesce_max_experts=2,
        )
        self.assertEqual(
            adjacent.expert_prefetch_policy,
            "exact-router-window-q3-a2-adjacent-pairs/v3",
        )
        self.assertEqual(
            adjacent.expert_prefetch_transport_policy,
            "streamer-exact-leaf-adjacent-envelope/v2",
        )
        disabled.release()
        q3.release()
        adjacent.release()

    def test_exact_adjacent_pair_reduces_six_ranges_to_four_with_bit_parity(
        self,
    ) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        bases = [f"layers.0.ffn.experts.{value}" for value in range(3)]
        hidden = torch.linspace(-0.01, 0.01, 128, dtype=torch.bfloat16)[None, :]

        reference_source = _AdjacentBatchExpertSource(bases)
        reference = DeepSeekWeightPager(
            reference_source,
            device="cpu",
            compute_dtype="bfloat16",
            expert_prefetch=False,
        )
        expected = [reference.expert(hidden, base) for base in bases]

        q3_source = _AdjacentBatchExpertSource(bases)
        q3 = DeepSeekWeightPager(
            q3_source,
            device="cpu",
            compute_dtype="bfloat16",
            expert_range_coalesce_max_experts=1,
        )
        q3_window = q3.prefetch_expert_window(bases)
        assert q3_window is not None
        q3_outputs = []
        for base in bases:
            payload = q3.consume_expert_window(q3_window, base)
            q3_outputs.append(q3.expert(hidden, base, prefetched_payload=payload))
        q3.close_expert_window(q3_window)

        source = _AdjacentBatchExpertSource(bases)
        pager = DeepSeekWeightPager(
            source,
            device="cpu",
            compute_dtype="bfloat16",
            expert_range_coalesce_max_experts=2,
        )
        window = pager.prefetch_expert_window(bases)
        assert window is not None
        actual = []
        for base in bases:
            payload = pager.consume_expert_window(window, base)
            actual.append(pager.expert(hidden, base, prefetched_payload=payload))
        pager.close_expert_window(window)

        self.assertTrue(
            all(torch.equal(left, right) for left, right in zip(actual, expected))
        )
        self.assertTrue(
            all(torch.equal(left, right) for left, right in zip(actual, q3_outputs))
        )
        self.assertEqual(len(reference_source.raw_calls), 6)
        self.assertEqual(len(q3_source.raw_calls), 6)
        self.assertEqual(len(source.raw_calls), 4)
        self.assertEqual(
            sum(length for _, _, length in reference_source.raw_calls),
            sum(length for _, _, length in source.raw_calls),
        )
        metrics = pager.metrics()
        self.assertEqual(metrics["expert_prefetch_submitted"], 3)
        self.assertEqual(metrics["expert_prefetch_batches_submitted"], 2)
        self.assertEqual(metrics["expert_transport_envelopes"], 4)
        self.assertEqual(metrics["expert_range_requests_avoided"], 2)
        self.assertEqual(metrics["expert_range_gap_bytes"], 0)
        self.assertEqual(metrics["expert_source_ranges"], 6)
        self.assertEqual(metrics["expert_prefetch_max_outstanding"], 3)
        self.assertEqual(metrics["expert_prefetch_peak_bytes"], 3 * 26112)
        self.assertEqual(metrics["expert_range_coalesce_max_experts"], 2)
        self.assertEqual(metrics["expert_range_coalesce_max_gap_bytes"], 0)
        self.assertEqual(
            metrics["expert_prefetch_policy"],
            "exact-router-window-q3-a2-adjacent-pairs/v3",
        )
        self.assertEqual(q3.metrics()["expert_prefetch_batches_submitted"], 3)
        self.assertEqual(q3.metrics()["expert_range_requests_avoided"], 0)
        self.assertEqual(
            q3.metrics()["expert_prefetch_policy"],
            "exact-router-window-q3-a2/v2",
        )
        reference.release()
        q3.release()
        pager.release()

    def test_adjacent_pair_reverse_completion_keeps_owner_until_second_release(
        self,
    ) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        bases = [f"layers.0.ffn.experts.{value}" for value in range(4)]
        pair = (bases[0], bases[1])
        second = (bases[2],)
        fourth = (bases[3],)
        source = _BlockingAdjacentBatchExpertSource(bases)
        pager = DeepSeekWeightPager(
            source,
            device="cpu",
            compute_dtype="bfloat16",
            expert_range_coalesce_max_experts=2,
        )
        hidden = torch.zeros((1, 128), dtype=torch.bfloat16)
        with mock.patch.object(
            pager,
            "EXPERT_PREFETCH_RESIDENT_LIMIT_BYTES",
            3 * 26112,
        ):
            window = pager.prefetch_expert_window(bases)
            assert window is not None
            self.assertTrue(source.started[pair].wait(timeout=2))
            self.assertTrue(source.started[second].wait(timeout=2))
            self.assertFalse(source.started[fourth].is_set())
            self.assertEqual(source.max_active, 2)

            # The later singleton may finish first, but ordered E0 consumption
            # still waits for the shared E0/E1 envelope.
            source.release[second].set()
            self.assertTrue(source.completed[second].wait(timeout=2))
            entered = threading.Event()
            consumed = threading.Event()
            first_payload: list[object] = []

            def consume_first() -> None:
                entered.set()
                first_payload.append(pager.consume_expert_window(window, bases[0]))
                consumed.set()

            consumer = threading.Thread(target=consume_first)
            consumer.start()
            self.assertTrue(entered.wait(timeout=1))
            self.assertFalse(consumed.is_set())
            source.release[pair].set()
            self.assertTrue(consumed.wait(timeout=2))
            consumer.join(timeout=1)
            self.assertFalse(consumer.is_alive())

            pager.expert(hidden, bases[0], prefetched_payload=first_payload[0])
            # E1's view still pins the full pair owner, so the honest 3-expert
            # cap cannot start E3 yet.
            self.assertFalse(source.started[fourth].is_set())
            payload = pager.consume_expert_window(window, bases[1])
            pager.expert(hidden, bases[1], prefetched_payload=payload)
            self.assertTrue(source.started[fourth].wait(timeout=2))

            source.release[fourth].set()
            for base in bases[2:]:
                payload = pager.consume_expert_window(window, base)
                pager.expert(hidden, base, prefetched_payload=payload)
            pager.close_expert_window(window)

        self.assertLessEqual(source.max_active, 2)
        metrics = pager.metrics()
        self.assertEqual(metrics["expert_prefetch_submitted"], 4)
        self.assertEqual(metrics["expert_prefetch_batches_submitted"], 3)
        self.assertEqual(metrics["expert_prefetch_max_outstanding"], 3)
        self.assertEqual(metrics["expert_prefetch_peak_bytes"], 3 * 26112)
        self.assertEqual(metrics["expert_transport_envelopes"], 6)
        self.assertEqual(metrics["expert_range_requests_avoided"], 2)
        pager.release()

    def test_adjacent_pair_failure_detaches_shared_future_once(self) -> None:
        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        bases = [f"layers.0.ffn.experts.{value}" for value in range(3)]
        pair = (bases[0], bases[1])
        singleton = (bases[2],)
        source = _BlockingAdjacentBatchExpertSource(bases)
        source.fail.add(pair)
        pager = DeepSeekWeightPager(
            source,
            device="cpu",
            compute_dtype="bfloat16",
            expert_range_coalesce_max_experts=2,
        )
        callback_finished = threading.Event()
        original_finish = pager._finish_cancelled_prefetch

        def finish(*args, **kwargs) -> None:
            try:
                original_finish(*args, **kwargs)
            finally:
                if kwargs.get("detached"):
                    callback_finished.set()

        with mock.patch.object(pager, "_finish_cancelled_prefetch", side_effect=finish):
            window = pager.prefetch_expert_window(bases)
            assert window is not None
            self.assertTrue(source.started[pair].wait(timeout=2))
            self.assertTrue(source.started[singleton].wait(timeout=2))
            source.release[pair].set()
            with self.assertRaisesRegex(RuntimeError, "adjacent batch failure"):
                pager.consume_expert_window(window, bases[0])
            self.assertEqual(len(pager._draining_prefetch), 1)
            self.assertIsNone(pager.prefetch_expert_window(bases))
            source.release[singleton].set()
            self.assertTrue(callback_finished.wait(timeout=2))

        self.assertFalse(pager.metrics()["expert_prefetch_draining"])
        self.assertEqual(pager.metrics()["expert_prefetch_failures"], 1)
        self.assertEqual(pager.metrics()["expert_prefetch_cancelled"], 3)
        self.assertEqual(pager.metrics()["expert_transport_envelopes"], 2)
        self.assertLessEqual(source.max_active, 2)
        pager.release()

    def test_window_plans_all_or_falls_back_before_any_read(self) -> None:
        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        bases = [f"layers.0.ffn.experts.{value}" for value in range(3)]
        source = _MultiEncodedExpertSource(bases)
        broken = source.meta[f"{bases[-1]}.w2.weight"]
        broken["offset_in_shard"] = [
            broken["offset_in_shard"][0] + 1,
            broken["offset_in_shard"][1],
        ]
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="bfloat16")
        self.assertIsNone(pager.prefetch_expert_window(bases))
        self.assertEqual(source.raw_calls, [])
        self.assertEqual(pager.metrics()["expert_prefetch_sync_fallbacks"], 1)

        bounded = _MultiEncodedExpertSource(bases)
        bounded_pager = DeepSeekWeightPager(
            bounded, device="cpu", compute_dtype="bfloat16"
        )
        with mock.patch.object(
            bounded_pager,
            "EXPERT_PREFETCH_RESIDENT_LIMIT_BYTES",
            3 * 26112 - 1,
        ):
            self.assertIsNone(bounded_pager.prefetch_expert_window(bases))
        self.assertEqual(bounded.raw_calls, [])

        per_expert = _MultiEncodedExpertSource(bases)
        per_expert_pager = DeepSeekWeightPager(
            per_expert, device="cpu", compute_dtype="bfloat16"
        )
        with mock.patch.object(
            per_expert_pager,
            "EXPERT_PREFETCH_PAYLOAD_LIMIT_BYTES",
            26112 - 1,
        ):
            self.assertIsNone(per_expert_pager.prefetch_expert_window(bases))
        self.assertEqual(per_expert.raw_calls, [])
        per_expert_pager.release()
        bounded_pager.release()
        pager.release()

    def test_window_rejects_out_of_order_replay_and_release_while_live(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager
        from immer.runtimes.deepseek_v4.pager import DeepSeekPagerError

        bases = [f"layers.0.ffn.experts.{value}" for value in range(2)]
        source = _WindowExpertSource(bases)
        for release in source.release.values():
            release.set()
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="bfloat16")
        window = pager.prefetch_expert_window(bases)
        assert window is not None
        with self.assertRaisesRegex(DeepSeekPagerError, "live.*window"):
            pager.release()
        with self.assertRaisesRegex(DeepSeekPagerError, "out of order"):
            pager.consume_expert_window(window, bases[1])
        first = pager.consume_expert_window(window, bases[0])
        with self.assertRaisesRegex(DeepSeekPagerError, "active"):
            pager.consume_expert_window(window, bases[1])
        with self.assertRaisesRegex(DeepSeekPagerError, "active.*payload"):
            pager.release()
        pager.expert(
            torch.zeros((1, 128), dtype=torch.bfloat16),
            bases[0],
            prefetched_payload=first,
        )
        with self.assertRaisesRegex(DeepSeekPagerError, "out of order"):
            pager.consume_expert_window(window, bases[0])
        second = pager.consume_expert_window(window, bases[1])
        pager.discard_expert_payload(second)
        pager.close_expert_window(window)
        with self.assertRaisesRegex(DeepSeekPagerError, "stale or closed"):
            pager.consume_expert_window(window, bases[1])
        pager.release()

    def test_window_read_failure_detaches_two_reads_and_forces_sync_fallback(
        self,
    ) -> None:
        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        bases = [f"layers.0.ffn.experts.{value}" for value in range(2)]
        source = _WindowExpertSource(bases)
        source.fail.add(bases[0])
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="bfloat16")
        window = pager.prefetch_expert_window(bases)
        assert window is not None
        self.assertTrue(source.started[bases[0]].wait(timeout=2))
        self.assertTrue(source.started[bases[1]].wait(timeout=2))
        source.release[bases[0]].set()

        finished = threading.Event()
        raised: list[BaseException] = []

        def consume_failed() -> None:
            try:
                pager.consume_expert_window(window, bases[0])
            except BaseException as exc:
                raised.append(exc)
            finally:
                finished.set()

        consumer = threading.Thread(target=consume_failed)
        consumer.start()
        self.assertTrue(finished.wait(timeout=1))
        consumer.join(timeout=1)
        self.assertEqual(len(raised), 1)
        self.assertRegex(str(raised[0]), "simulated read failure")
        self.assertTrue(pager.metrics()["expert_prefetch_draining"])
        self.assertIsNone(pager.prefetch_expert_window(bases))

        source.release[bases[1]].set()
        self.assertTrue(source.completed[bases[1]].wait(timeout=2))
        for future in tuple(pager._draining_prefetch):
            future.result(timeout=2)
        self.assertFalse(pager.metrics()["expert_prefetch_draining"])
        self.assertEqual(pager.metrics()["expert_prefetch_failures"], 1)
        pager.release()

    def test_compute_failure_detaches_two_blocked_future_experts(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        bases = [f"layers.0.ffn.experts.{value}" for value in range(3)]
        source = _WindowExpertSource(bases)
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="bfloat16")
        window = pager.prefetch_expert_window(bases)
        assert window is not None
        self.assertTrue(source.started[bases[0]].wait(timeout=2))
        self.assertTrue(source.started[bases[1]].wait(timeout=2))
        source.release[bases[0]].set()
        first = pager.consume_expert_window(window, bases[0])
        self.assertTrue(source.started[bases[2]].wait(timeout=2))
        with (
            mock.patch.object(
                pager,
                "_expert_with_payload",
                side_effect=RuntimeError("simulated current compute failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "current compute"),
        ):
            pager.expert(
                torch.zeros((1, 128), dtype=torch.bfloat16),
                bases[0],
                prefetched_payload=first,
            )

        cancelled = threading.Event()

        def cancel_window() -> None:
            pager.close_expert_window(window, cancel=True)
            cancelled.set()

        cancellation = threading.Thread(target=cancel_window)
        cancellation.start()
        self.assertTrue(cancelled.wait(timeout=1))
        cancellation.join(timeout=1)
        self.assertFalse(cancellation.is_alive())
        self.assertEqual(len(pager._draining_prefetch), 2)
        self.assertIsNone(pager.prefetch_expert_window(bases))

        source.release[bases[1]].set()
        source.release[bases[2]].set()
        self.assertTrue(source.completed[bases[1]].wait(timeout=2))
        self.assertTrue(source.completed[bases[2]].wait(timeout=2))
        for future in tuple(pager._draining_prefetch):
            future.result(timeout=2)
        self.assertFalse(pager.metrics()["expert_prefetch_draining"])
        pager.release()

    def test_release_stays_fast_but_close_joins_draining_prefetch(self) -> None:
        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        bases = [f"layers.0.ffn.experts.{value}" for value in range(2)]
        source = _WindowExpertSource(bases)
        source.fail.add(bases[0])
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="bfloat16")
        window = pager.prefetch_expert_window(bases)
        assert window is not None
        self.assertTrue(source.started[bases[0]].wait(timeout=2))
        self.assertTrue(source.started[bases[1]].wait(timeout=2))
        source.release[bases[0]].set()
        with self.assertRaisesRegex(RuntimeError, "simulated read failure"):
            pager.consume_expert_window(window, bases[0])
        self.assertEqual(len(pager._draining_prefetch), 1)
        executor = pager._draining_executor
        self.assertIsNotNone(executor)

        layer_released = threading.Event()
        layer_release = threading.Thread(
            target=lambda: (pager.release(), layer_released.set())
        )
        layer_release.start()
        self.assertTrue(layer_released.wait(timeout=1))
        layer_release.join(timeout=1)
        self.assertFalse(layer_release.is_alive())
        self.assertIs(pager._draining_executor, executor)

        closed = threading.Event()
        final_close = threading.Thread(target=lambda: (pager.close(), closed.set()))
        final_close.start()
        self.assertFalse(closed.wait(timeout=0.1))
        source.release[bases[1]].set()
        self.assertTrue(closed.wait(timeout=2))
        final_close.join(timeout=1)
        self.assertFalse(final_close.is_alive())
        self.assertFalse(pager._draining_prefetch)
        self.assertIsNone(pager._draining_executor)
        assert executor is not None
        self.assertTrue(all(not thread.is_alive() for thread in executor._threads))

    def test_single_prefetch_compatibility_avoids_second_read(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager
        from immer.runtimes.deepseek_v4.pager import DeepSeekPagerError

        base = "layers.0.ffn.experts.7"
        source = _WindowExpertSource([base])
        source.release[base].set()
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="bfloat16")
        ticket = pager.prefetch_expert(base)
        assert ticket is not None
        payload = pager.consume_expert_prefetch(ticket, base)
        pager.expert(
            torch.zeros((1, 128), dtype=torch.bfloat16),
            base,
            prefetched_payload=payload,
        )
        self.assertEqual(len(source.raw_calls), 2)
        with self.assertRaisesRegex(DeepSeekPagerError, "stale or consumed"):
            pager.consume_expert_prefetch(ticket, base)
        pager.release()

    def test_coalesced_expert_rejects_reserved_float8_weight_and_scale_codes(
        self,
    ) -> None:
        import torch

        from immer.knowledge import TensorEncodingError
        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        base = "layers.0.ffn.shared_experts"
        for tensor_name, reserved in (
            (f"{base}.w1.weight", 0x7F),
            (f"{base}.w1.scale", 0xFF),
        ):
            with self.subTest(tensor=tensor_name):
                source = _EncodedExpertSource(base, "fp8", adjacent=True)
                payload = bytearray(source.payload)
                begin = int(source.meta[tensor_name]["offset_in_shard"][0])
                payload[begin] = reserved
                source.payload = bytes(payload)
                pager = DeepSeekWeightPager(
                    source, device="cpu", compute_dtype="float32"
                )
                with self.assertRaisesRegex(
                    TensorEncodingError, r"Reservierte/nicht-endliche"
                ):
                    pager.expert(torch.zeros((1, 128)), base)

    def test_quantized_fp8_and_packed_fp4_execute_end_to_end(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        source = _Source()
        source.data.update(
            {
                "fp8.weight": np.ones((2, 128), dtype=np.float32),
                "fp8.scale": np.full((1, 1), 2.0, dtype=np.float32),
                "fp4.weight": np.full((2, 64), 0x22, dtype=np.int8),
                "fp4.scale": np.full((2, 4), 2.0, dtype=np.float32),
            }
        )
        source.dtypes.update(
            {
                "fp8.weight": "F8_E4M3",
                "fp8.scale": "F8_E8M0",
                "fp4.weight": "I8",
                "fp4.scale": "F8_E8M0",
            }
        )
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="float32")
        x = torch.ones((1, 128), dtype=torch.float32)
        torch.testing.assert_close(pager.linear(x, "fp8"), torch.full((1, 2), 256.0))
        torch.testing.assert_close(pager.linear(x, "fp4"), torch.full((1, 2), 256.0))
        metrics = pager.metrics()
        self.assertEqual(metrics["linear_calls"], 2)
        self.assertEqual(metrics["peak_single_weight_bytes"], 2 * 128 * 4)
        self.assertEqual(metrics["materialized_scale_bytes"], 4 + 2 * 4 * 4)

    def test_fp8_linear_matches_official_k128_accumulator_seed_130(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        torch.manual_seed(130)
        n, k = 128, 4096
        x = torch.randn(1, k).to(torch.bfloat16)
        qw = torch.randn(n, k).to(torch.float8_e4m3fn).float()
        sw = torch.pow(2.0, torch.randint(-16, 0, (k // 128,)).float())

        grouped = x.float().numpy().reshape(1, -1, 128)
        sx = np.exp2(
            np.ceil(
                np.log2(
                    np.maximum(np.max(np.abs(grouped), axis=-1), np.float32(1e-4))
                    / np.float32(448.0)
                )
            )
        ).astype(np.float32)
        qx = (
            torch.from_numpy(grouped / sx[..., None])
            .to(torch.float8_e4m3fn)
            .float()
            .reshape(1, k)
        )
        expected = torch.zeros(1, n, dtype=torch.float32)
        for block in range(k // 128):
            start = block * 128
            stop = start + 128
            expected += (
                (qx[:, start:stop] @ qw[:, start:stop].T)
                * float(sx[0, block])
                * sw[block]
            )
        expected = expected.to(torch.bfloat16)
        self.assertEqual(expected[0, 127].float().item(), -21.375)

        source = _Source()
        source.data["p.weight"] = qw.numpy()
        source.data["p.scale"] = sw.reshape(1, -1).numpy()
        source.dtypes["p.weight"] = "F8_E4M3"
        source.dtypes["p.scale"] = "F8_E8M0"
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="bfloat16")
        actual = pager.linear(x, "p")

        self.assertTrue(torch.equal(actual, expected))
        metrics = pager.metrics()
        self.assertEqual(metrics["block_scaled_linear_calls"], 1)
        self.assertEqual(metrics["fp8_k128_tiles"], k // 128)

    def test_fp8_linear_preserves_leading_dims_and_partial_n_tile(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import (
            DeepSeekWeightPager,
            quantize_fp8_e4m3_parts,
        )

        torch.manual_seed(207)
        n, k = 130, 256
        x = torch.randn(2, 3, k).to(torch.bfloat16)
        qw = torch.randn(n, k).to(torch.float8_e4m3fn).float()
        sw = torch.pow(
            2.0,
            torch.asarray([[-8, -4], [-2, -10]], dtype=torch.float32),
        )
        qx_values, sx_values = quantize_fp8_e4m3_parts(x.float().numpy())
        qx = torch.from_numpy(qx_values.reshape(-1, k))
        sx = torch.from_numpy(sx_values.reshape(-1, k // 128))
        row_scales = sw.repeat_interleave(128, dim=0)[:n]
        expected = torch.zeros(qx.shape[0], n, dtype=torch.float32)
        for block in range(k // 128):
            start = block * 128
            stop = start + 128
            expected += (
                (qx[:, start:stop] @ qw[:, start:stop].T)
                * sx[:, block, None]
                * row_scales[None, :, block]
            )
        expected = expected.reshape(2, 3, n).to(torch.bfloat16)

        source = _Source()
        source.data["partial.weight"] = qw.numpy()
        source.data["partial.scale"] = sw.numpy()
        source.dtypes["partial.weight"] = "F8_E4M3"
        source.dtypes["partial.scale"] = "F8_E8M0"
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="bfloat16")
        actual = pager.linear(x, "partial")
        self.assertTrue(torch.equal(actual, expected))

    def test_fp8_block_accumulator_matches_cpu_on_mps(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        if not torch.backends.mps.is_available():
            self.skipTest("MPS is unavailable")
        torch.manual_seed(431)
        n, k = 128, 512
        x = torch.randn(2, k).to(torch.bfloat16)
        qw = torch.randn(n, k).to(torch.float8_e4m3fn).float().numpy()
        sw = torch.pow(2.0, torch.randint(-12, 1, (1, k // 128)).float()).numpy()

        def source() -> _Source:
            result = _Source()
            result.data["mps.weight"] = qw
            result.data["mps.scale"] = sw
            result.dtypes["mps.weight"] = "F8_E4M3"
            result.dtypes["mps.scale"] = "F8_E8M0"
            return result

        cpu = DeepSeekWeightPager(
            source(), device="cpu", compute_dtype="bfloat16"
        ).linear(x, "mps")
        mps = DeepSeekWeightPager(
            source(), device="mps", compute_dtype="bfloat16"
        ).linear(x, "mps")
        self.assertTrue(torch.equal(cpu, mps.to("cpu")))

    def test_fp4_linear_matches_official_k32_weight_scale_rule(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import (
            DeepSeekWeightPager,
            quantize_fp8_e4m3_parts,
        )

        rng = np.random.default_rng(19)
        n, k = 5, 256
        low = rng.integers(0, 16, size=(n, k // 2), dtype=np.uint8)
        high = rng.integers(0, 16, size=(n, k // 2), dtype=np.uint8)
        packed = np.ascontiguousarray(low | (high << np.uint8(4))).view(np.int8)
        table = np.asarray(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=np.float32,
        )
        bits = packed.view(np.uint8)
        qw_values = np.empty((n, k), dtype=np.float32)
        qw_values[:, 0::2] = table[bits & np.uint8(0x0F)]
        qw_values[:, 1::2] = table[(bits >> np.uint8(4)) & np.uint8(0x0F)]
        sw_values = np.exp2(rng.integers(-12, 1, size=(n, k // 32)).astype(np.float32))
        torch.manual_seed(19)
        x = torch.randn(2, k).to(torch.bfloat16)
        qx_values, sx_values = quantize_fp8_e4m3_parts(x.float().numpy())
        qx = torch.from_numpy(qx_values)
        sx = torch.from_numpy(sx_values)
        qw = torch.from_numpy(qw_values)
        sw = torch.from_numpy(sw_values)
        expected = torch.zeros(2, n, dtype=torch.float32)
        for block in range(k // 32):
            start = block * 32
            stop = start + 32
            expected += (
                (qx[:, start:stop] @ qw[:, start:stop].T)
                * sx[:, block // 4, None]
                * sw[None, :, block]
            )
        expected = expected.to(torch.bfloat16)

        source = _Source()
        source.data["fp4_exact.weight"] = packed
        source.data["fp4_exact.scale"] = sw_values
        source.dtypes["fp4_exact.weight"] = "I8"
        source.dtypes["fp4_exact.scale"] = "F8_E8M0"
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="bfloat16")
        actual = pager.linear(x, "fp4_exact")
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(pager.metrics()["fp4_k32_tiles"], k // 32)

    def test_dense_linear_and_metrics(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        source = _Source()
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="float32")
        actual = pager.linear(torch.asarray([[5.0, 6.0]]), "dense")
        torch.testing.assert_close(actual, torch.asarray([[17.0, 39.0]]))
        metrics = pager.metrics()
        self.assertEqual(metrics["linear_calls"], 1)
        self.assertEqual(metrics["logical_weight_bytes"], 16)
        self.assertEqual(metrics["peak_single_weight_bytes"], 16)

    def test_auto_compute_dtype_preserves_official_bfloat16_on_cpu(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        pager = DeepSeekWeightPager(_Source(), device="cpu", compute_dtype="auto")
        self.assertEqual(pager.compute_dtype, torch.bfloat16)

    def test_embedding_coalesces_consecutive_unique_rows_and_restores_order(
        self,
    ) -> None:
        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        source = _Source()
        pager = DeepSeekWeightPager(source, device="cpu")
        actual = pager.embedding([5, 2, 3, 5]).float().numpy()
        np.testing.assert_array_equal(actual, source.data["embed.weight"][[5, 2, 3, 5]])
        self.assertEqual(
            source.row_calls,
            [("embed.weight", 2, 2), ("embed.weight", 5, 1)],
        )

    def test_grouped_linear_does_not_mix_groups(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        pager = DeepSeekWeightPager(_Source(), device="cpu", compute_dtype="float32")
        x = torch.asarray([[[5.0, 7.0], [11.0, 13.0]]])
        actual = pager.grouped_linear(x, "group", groups=2)
        torch.testing.assert_close(actual, torch.asarray([[[5.0, 7.0], [22.0, 39.0]]]))

    def test_candidate_head_coalesces_rows_and_restores_requested_order(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        source = _Source()
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="float32")
        hidden = torch.asarray([[2.0, 3.0]])
        actual = pager.candidate_logits(hidden, [3, 1, 2, 3])
        expected = torch.nn.functional.linear(
            hidden, torch.from_numpy(source.data["head.weight"][[3, 1, 2, 3]])
        )
        torch.testing.assert_close(actual, expected)
        self.assertEqual(source.row_calls, [("head.weight", 1, 3)])

    def test_projection_releases_do_not_purge_until_explicit_boundary(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        source = _Source()
        base = "layers.0.ffn.shared_experts"
        source.data.update(
            {
                f"{base}.w1.weight": np.eye(2, dtype=np.float32),
                f"{base}.w2.weight": np.eye(2, dtype=np.float32),
                f"{base}.w3.weight": np.asarray(
                    [[2.0, 0.0], [0.0, 3.0]], dtype=np.float32
                ),
            }
        )
        source.dtypes.update(
            {name: "F32" for name in source.data if name.startswith(base)}
        )
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="float32")
        x = torch.asarray([[1.0, 2.0]])
        fake_mps = mock.Mock()

        with (
            mock.patch.object(pager.torch, "mps", fake_mps, create=True),
            mock.patch.object(pager, "_uses_mps_allocator", return_value=True),
        ):
            dense = pager.linear(x, "dense")
            expert = pager.expert(x, base)
            grouped = pager.grouped_linear(
                torch.asarray([[[5.0, 7.0], [11.0, 13.0]]]),
                "group",
                groups=2,
            )

            torch.testing.assert_close(dense, torch.asarray([[5.0, 11.0]]))
            expected_expert = torch.nn.functional.silu(x) * torch.asarray([[2.0, 6.0]])
            torch.testing.assert_close(expert, expected_expert)
            torch.testing.assert_close(
                grouped, torch.asarray([[[5.0, 7.0], [22.0, 39.0]]])
            )
            fake_mps.empty_cache.assert_not_called()
            metrics = pager.metrics()
            self.assertEqual(metrics["linear_calls"], 5)
            self.assertEqual(metrics["expert_calls"], 1)
            self.assertEqual(metrics["materialized_weight_releases"], 5)
            self.assertEqual(metrics["release_boundaries"], 0)
            self.assertEqual(metrics["mps_cache_purges"], 0)

            pager.release()

        fake_mps.empty_cache.assert_called_once_with()
        metrics = pager.metrics()
        self.assertEqual(metrics["release_boundaries"], 1)
        self.assertEqual(metrics["mps_cache_purges"], 1)

    def test_cpu_release_and_missing_mps_cache_api_are_safe(self) -> None:
        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        pager = DeepSeekWeightPager(_Source(), device="cpu", compute_dtype="float32")
        fake_mps = mock.Mock()
        with mock.patch.object(pager.torch, "mps", fake_mps, create=True):
            pager.release()
        fake_mps.empty_cache.assert_not_called()
        self.assertEqual(pager.metrics()["release_boundaries"], 1)
        self.assertEqual(pager.metrics()["mps_cache_purges"], 0)

        with (
            mock.patch.object(pager, "_uses_mps_allocator", return_value=True),
            mock.patch.object(pager.torch, "mps", object(), create=True),
        ):
            self.assertFalse(pager._purge_mps_cache())
        self.assertEqual(pager.metrics()["mps_cache_purges"], 0)

    def test_blockwise_head_topk_matches_full_linear(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        source = _Source()
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="float32")
        hidden = torch.asarray([[2.0, 3.0]])
        values, ids = pager.topk_logits(hidden, k=2, block_rows=2)
        full = torch.nn.functional.linear(
            hidden, torch.from_numpy(source.data["head.weight"])
        )
        expected_values, expected_ids = torch.topk(full, 2, dim=-1)
        torch.testing.assert_close(values, expected_values)
        torch.testing.assert_close(ids, expected_ids)
        self.assertEqual(pager.metrics()["head_rows"], 4)

    def test_head_transport_batches_eight_exact_leaves_with_bit_parity(self) -> None:
        import hashlib

        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        hidden = torch.asarray([[0.5, -1.0, 1.5, 0.25]])
        scalar_source = _AdjacentHeadSource(vocab=8)
        scalar = DeepSeekWeightPager(
            scalar_source, device="cpu", compute_dtype="float32"
        )
        scalar_blocks: list[tuple[int, str]] = []

        def observe_scalar(start: int, logits) -> None:
            digest = hashlib.sha256(logits.detach().cpu().numpy().tobytes()).hexdigest()
            scalar_blocks.append((start, digest))

        expected_values, expected_ids = scalar.topk_logits(
            hidden,
            k=1,
            block_rows=1,
            instrument_block_observer=observe_scalar,
        )

        batch_source = _AdjacentHeadSource(vocab=8)
        batched = DeepSeekWeightPager(
            batch_source, device="cpu", compute_dtype="float32"
        )
        batch_blocks: list[tuple[int, str]] = []

        def observe_batch(start: int, logits) -> None:
            digest = hashlib.sha256(logits.detach().cpu().numpy().tobytes()).hexdigest()
            batch_blocks.append((start, digest))

        actual_values, actual_ids = batched.topk_logits(
            hidden,
            k=1,
            block_rows=1,
            transport_range_batch_blocks=8,
            instrument_block_observer=observe_batch,
        )

        self.assertTrue(torch.equal(actual_values, expected_values))
        self.assertTrue(torch.equal(actual_ids, expected_ids))
        self.assertEqual(batch_blocks, scalar_blocks)
        self.assertEqual([start for start, _ in batch_blocks], list(range(8)))
        self.assertEqual(
            scalar_source.row_calls, [("head.weight", row, 1) for row in range(8)]
        )
        self.assertEqual(batch_source.row_calls, [])
        self.assertEqual(len(batch_source.batch_calls), 1)
        shard, ranges, resident_limit, max_gap = batch_source.batch_calls[0]
        self.assertEqual(shard, "head.safetensors")
        self.assertEqual(len(ranges), 8)
        self.assertEqual(resident_limit, 64 * 1024**2)
        self.assertEqual(max_gap, 0)
        self.assertEqual(len(batch_source.envelopes), 1)
        metrics = batched.metrics()
        self.assertEqual(metrics["head_logical_leaves"], 8)
        self.assertEqual(metrics["head_transport_batches"], 1)
        self.assertEqual(metrics["head_transport_envelopes"], 1)
        self.assertEqual(metrics["head_planned_range_calls_avoided"], 7)
        self.assertEqual(metrics["head_transport_source_bytes"], 64)
        self.assertEqual(metrics["head_transport_fallbacks"], 0)
        self.assertEqual(
            metrics["head_transport_policy"],
            "exact-head-leaf-adjacent-envelope/v1",
        )

    def test_head_transport_tail_group_preserves_compute_blocks(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        source = _AdjacentHeadSource(vocab=10)
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="float32")
        observed: list[int] = []
        pager.topk_logits(
            torch.ones((1, 4)),
            k=1,
            block_rows=1,
            transport_range_batch_blocks=8,
            instrument_block_observer=lambda start, _logits: observed.append(start),
        )

        self.assertEqual(observed, list(range(10)))
        self.assertEqual([len(call[1]) for call in source.batch_calls], [8, 2])
        self.assertEqual(len(source.envelopes), 2)
        metrics = pager.metrics()
        self.assertEqual(metrics["head_transport_batches"], 2)
        self.assertEqual(metrics["head_transport_envelopes"], 2)
        self.assertEqual(metrics["head_planned_range_calls_avoided"], 8)

    def test_head_transport_warm_receipt_does_not_claim_physical_requests(
        self,
    ) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        pager = DeepSeekWeightPager(
            _WarmHeadSource(vocab=8), device="cpu", compute_dtype="float32"
        )
        pager.topk_logits(
            torch.ones((1, 4)),
            block_rows=1,
            transport_range_batch_blocks=8,
        )

        metrics = pager.metrics()
        self.assertEqual(metrics["head_transport_envelopes"], 0)
        self.assertEqual(metrics["head_transport_source_bytes"], 0)
        self.assertEqual(metrics["head_planned_range_calls_avoided"], 7)

    def test_head_transport_default_keeps_scalar_row_calls(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        source = _AdjacentHeadSource(vocab=8)
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="float32")
        pager.topk_logits(torch.ones((1, 4)), k=1, block_rows=2)

        self.assertEqual(
            source.row_calls,
            [
                ("head.weight", 0, 2),
                ("head.weight", 2, 2),
                ("head.weight", 4, 2),
                ("head.weight", 6, 2),
            ],
        )
        self.assertEqual(source.batch_calls, [])
        metrics = pager.metrics()
        self.assertEqual(metrics["head_logical_leaves"], 4)
        self.assertEqual(metrics["head_transport_batches"], 0)
        self.assertEqual(metrics["head_transport_fallbacks"], 0)

    def test_head_transport_falls_back_before_io_for_missing_layout_or_dtype(
        self,
    ) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        missing = _HeadCapabilityWithoutLayout()
        missing_pager = DeepSeekWeightPager(
            missing, device="cpu", compute_dtype="float32"
        )
        missing_pager.topk_logits(
            torch.asarray([[2.0, 3.0]]),
            k=1,
            block_rows=2,
            transport_range_batch_blocks=8,
        )
        self.assertEqual(missing.batch_calls, 0)
        self.assertEqual(
            missing.row_calls,
            [("head.weight", 0, 2), ("head.weight", 2, 2)],
        )
        self.assertEqual(missing_pager.metrics()["head_transport_fallbacks"], 1)
        self.assertEqual(missing_pager.metrics()["head_transport_fallback_leaves"], 2)

        unsupported = _AdjacentHeadSource(vocab=4, dtype="I8")
        unsupported_pager = DeepSeekWeightPager(
            unsupported, device="cpu", compute_dtype="float32"
        )
        unsupported_pager.topk_logits(
            torch.ones((1, 4)),
            k=1,
            block_rows=2,
            transport_range_batch_blocks=8,
        )
        self.assertEqual(unsupported.batch_calls, [])
        self.assertEqual(len(unsupported.row_calls), 2)

    def test_head_transport_splits_before_raw_resident_cap(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        source = _AdjacentHeadSource(vocab=8)
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="float32")
        with mock.patch.object(pager, "HEAD_TRANSPORT_RESIDENT_LIMIT_BYTES", 16):
            pager.topk_logits(
                torch.ones((1, 4)),
                k=1,
                block_rows=1,
                transport_range_batch_blocks=8,
            )

        self.assertEqual([len(call[1]) for call in source.batch_calls], [2, 2, 2, 2])
        self.assertTrue(all(call[2] == 16 for call in source.batch_calls))
        self.assertTrue(
            all(
                sum(length for _, length in call[1]) <= call[2]
                for call in source.batch_calls
            )
        )

    def test_head_transport_rejects_invalid_batch_width_and_receipt(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager
        from immer.runtimes.deepseek_v4.pager import DeepSeekPagerError

        hidden = torch.ones((1, 4))
        pager = DeepSeekWeightPager(
            _AdjacentHeadSource(), device="cpu", compute_dtype="float32"
        )
        for invalid in (True, 0, 9, 2.0):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    pager.topk_logits(
                        hidden,
                        transport_range_batch_blocks=invalid,
                    )
        with self.assertRaises(ValueError):
            pager.topk_logits(hidden, instrument_block_observer=object())

        invalid_receipt = DeepSeekWeightPager(
            _InvalidHeadReceiptSource(),
            device="cpu",
            compute_dtype="float32",
        )
        with self.assertRaisesRegex(
            DeepSeekPagerError, "invalid head multi-range receipt"
        ):
            invalid_receipt.topk_logits(
                hidden,
                block_rows=1,
                transport_range_batch_blocks=8,
            )

    def test_head_block_observer_isolated_and_fail_closed(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        hidden = torch.ones((1, 4))
        baseline = DeepSeekWeightPager(
            _AdjacentHeadSource(), device="cpu", compute_dtype="float32"
        ).topk_logits(hidden, k=1, block_rows=2)
        pager = DeepSeekWeightPager(
            _AdjacentHeadSource(), device="cpu", compute_dtype="float32"
        )
        actual = pager.topk_logits(
            hidden,
            k=1,
            block_rows=2,
            transport_range_batch_blocks=8,
            instrument_block_observer=lambda _start, logits: logits.zero_(),
        )
        self.assertTrue(torch.equal(actual[0], baseline[0]))
        self.assertTrue(torch.equal(actual[1], baseline[1]))

        def fail(_start, _logits) -> None:
            raise RuntimeError("observer failed")

        with self.assertRaisesRegex(RuntimeError, "observer failed"):
            pager.topk_logits(hidden, instrument_block_observer=fail)

    def test_blockwise_head_topk_breaks_exact_ties_by_lower_token_id(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        source = _Source()
        source.data["head.weight"] = np.ones((7, 2), dtype=np.float32)
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="float32")
        values, ids = pager.topk_logits(torch.asarray([[1.0, 1.0]]), k=4, block_rows=3)
        torch.testing.assert_close(values, torch.full((1, 4), 2.0))
        torch.testing.assert_close(ids, torch.asarray([[0, 1, 2, 3]]))


if __name__ == "__main__":
    unittest.main()
