from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

from safetensors.torch import save_file
import torch

from immer.knowledge import Streamer
from immer.runtimes.qwen3_8.model import StreamedQwen38
from immer.runtimes.qwen3_8.native_crsa import Qwen38NativeHeadCrsa
from immer.runtimes.qwen3_8.pager import Qwen38WeightPager
from immer.runtimes.qwen3_8.semantic_state_cache import (
    SemanticStateAnchorCache,
    token_prefix_sha256,
)
from immer.runtimes.qwen3_8.snapshot import SnapshotTensor, write_qwen38_snapshot

from test_qwen3_8_model import _native_tiny_config, _tiny_config, _tiny_weights


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen38_anchor_battery_benchmark.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "qwen38_anchor_battery_benchmark", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import anchor-battery benchmark")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = _load_script()


def _input(
    *,
    prefix: tuple[int, ...] = (1, 2, 3),
    suffix: tuple[int, ...] = (4, 5),
    schedule: tuple[str, ...] = ("baseline", "battery"),
    attention_mode: str = "off",
) -> dict[str, object]:
    return benchmark._seal(
        {
            "attention_mode": attention_mode,
            "boundary_kind": "turn",
            "prefix_token_ids": list(prefix),
            "schedule": list(schedule),
            "schema": benchmark.INPUT_SCHEMA,
            "suffix_token_ids": list(suffix),
        }
    )


class _FakeSource:
    def __init__(self) -> None:
        self.body = 0

    def metrics(self):
        return {"network_or_source_body_bytes": self.body}


class _FakePager:
    def __init__(self, source: _FakeSource) -> None:
        self.source = source

    def topk_logits(self, hidden, *, k, name):
        self.source.body += 7
        ids = hidden.to(dtype=torch.int64)
        return hidden.clone(), ids


class _FakeModel:
    output_head_name = "lm_head.weight"

    def __init__(
        self,
        source: _FakeSource,
        *,
        corrupt_suffix: bool = False,
        one_shot_drift: float = 0.0,
        one_shot_state_drift: float = 0.0,
    ) -> None:
        self.pager = _FakePager(source)
        self.state = 0
        self.next_position = 0
        self.corrupt_suffix = corrupt_suffix
        self.one_shot_drift = one_shot_drift
        self.one_shot_state_drift = one_shot_state_drift
        self.audit_state_bias = 0.0

    def reset_state(self, *, release=False):
        self.state = 0
        self.next_position = 0
        self.audit_state_bias = 0.0

    def prefill(self, token_ids, *, reset=True):
        if reset:
            self.reset_state()
        tokens = tuple(int(value) for value in token_ids[0])
        self.pager.source.body += 100 * len(tokens)
        self.state += sum(tokens)
        self.next_position += len(tokens)
        value = self.state + (1 if self.corrupt_suffix and not reset else 0)
        if reset and len(tokens) > 3:
            value += self.one_shot_drift
            self.audit_state_bias = self.one_shot_state_drift
        hidden = torch.tensor([[[float(value)]]], dtype=torch.float32)
        return hidden, ()

    def save_state(self, path, **_kwargs):
        value = float(self.state) + self.audit_state_bias
        return write_qwen38_snapshot(
            path,
            identity={"fake_runtime": "v1"},
            state={"fake_state": self.state, "next_position": self.next_position},
            tensors={
                "state.fake": SnapshotTensor(
                    torch.tensor([value], dtype=torch.float32)
                )
            },
        )


class _FakeReceipt:
    def __init__(self, prefix: tuple[int, ...]) -> None:
        self.prefix_sha256 = token_prefix_sha256(prefix)
        self.prefix_length = len(prefix)
        self.cache_bytes = 4096
        self.snapshot_manifest_bytes = 96
        self.snapshot_payload_bytes = 3072
        self.seed_hidden_bytes = 928
        self.hit_count = 0
        self.receipt_sha256 = "a" * 64

    def to_document(self):
        return {
            "cache_bytes": self.cache_bytes,
            "hit_count": self.hit_count,
            "prefix_length": self.prefix_length,
            "prefix_sha256": self.prefix_sha256,
            "receipt_sha256": self.receipt_sha256,
            "seed_hidden_bytes": self.seed_hidden_bytes,
            "snapshot_manifest_bytes": self.snapshot_manifest_bytes,
            "snapshot_payload_bytes": self.snapshot_payload_bytes,
        }


class _TamperedPayloadModel(_FakeModel):
    def save_state(self, path, **kwargs):
        receipt = super().save_state(path, **kwargs)
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        payload = Path(path).parent / manifest["body"]["payload"]["file"]
        payload.write_bytes(payload.read_bytes() + b"tamper")
        return receipt


class _FakeCache:
    def __init__(self, *, wrong_restore=False) -> None:
        self.receipt = None
        self.prefix: tuple[int, ...] = ()
        self.seed = None
        self.wrong_restore = wrong_restore

    def store(self, model, token_ids, *, boundary_kind, seed_hidden):
        self.prefix = tuple(token_ids)
        self.seed = seed_hidden.detach().clone()
        self.receipt = _FakeReceipt(self.prefix)
        return self.receipt

    def restore_deepest(self, model, token_ids):
        if self.receipt is None or tuple(token_ids[: len(self.prefix)]) != self.prefix:
            return None
        model.state = sum(self.prefix) + int(self.wrong_restore)
        model.next_position = len(self.prefix)
        self.receipt.hit_count += 1
        exact = len(token_ids) == len(self.prefix)
        return SimpleNamespace(
            anchor=self.receipt,
            exact_prefix=exact,
            seed_hidden=self.seed if exact else None,
        )


def _fake_owner(
    *,
    corrupt_suffix=False,
    one_shot_drift=0.0,
    one_shot_state_drift=0.0,
):
    source = _FakeSource()
    model = _FakeModel(
        source,
        corrupt_suffix=corrupt_suffix,
        one_shot_drift=one_shot_drift,
        one_shot_state_drift=one_shot_state_drift,
    )
    return SimpleNamespace(model=model, pager=model.pager)


class AnchorBatteryInputTests(unittest.TestCase):
    def test_sealed_token_only_input_accepts_ab_ba_and_rejects_tamper(self) -> None:
        for schedule in (
            ("baseline", "battery"),
            ("battery", "baseline"),
            ("baseline", "battery", "battery", "baseline"),
        ):
            document = _input(schedule=schedule)
            self.assertEqual(benchmark._validate_input(document), document)
            self.assertNotIn("text", json.dumps(document).lower())
            self.assertNotIn("label", json.dumps(document).lower())

        tampered = copy.deepcopy(_input())
        tampered["suffix_token_ids"] = [99]
        with self.assertRaisesRegex(benchmark.AnchorBatteryBenchmarkError, "seal"):
            benchmark._validate_input(tampered)

    def test_output_creation_is_canonical_and_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            document = benchmark._seal({"schema": "fixture/v1", "value": 1})
            benchmark._atomic_new_json(path, document)
            self.assertEqual(path.read_bytes(), benchmark._canonical(document) + b"\n")
            with self.assertRaisesRegex(
                benchmark.AnchorBatteryBenchmarkError, "overwrite"
            ):
                benchmark._atomic_new_json(path, document)

    def test_input_reader_rejects_duplicate_noncanonical_and_symlink_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"schema":"x","schema":"y"}\n', encoding="utf-8")
            with self.assertRaisesRegex(
                benchmark.AnchorBatteryBenchmarkError, "duplicate JSON key"
            ):
                benchmark._stable_json(duplicate, "input", require_canonical=True)

            noncanonical = root / "noncanonical.json"
            noncanonical.write_text(
                json.dumps(_input(), indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                benchmark.AnchorBatteryBenchmarkError, "not canonical"
            ):
                benchmark._stable_json(noncanonical, "input", require_canonical=True)

            canonical = root / "canonical.json"
            canonical.write_bytes(benchmark._canonical(_input()) + b"\n")
            linked = root / "linked.json"
            linked.symlink_to(canonical)
            with self.assertRaisesRegex(
                benchmark.AnchorBatteryBenchmarkError, "non-symlink"
            ):
                benchmark._stable_json(linked, "input", require_canonical=True)


class AnchorBatteryFakeExecutionTests(unittest.TestCase):
    def test_charge_is_idle_only_and_compare_accounts_exact_saved_compute(self) -> None:
        document = _input(schedule=("baseline", "battery", "battery", "baseline"))
        owner = _fake_owner()
        cache = _FakeCache()
        charged = benchmark.charge_anchor(owner, cache, document, native_evidence=[])
        self.assertEqual(charged["model_source_bytes"], 300)
        self.assertEqual(charged["cache_bytes"], 4096)
        self.assertEqual(charged["anchor"]["prefix_length"], 3)

        with tempfile.TemporaryDirectory() as temporary:
            result = benchmark.compare_arms(
                owner,
                cache,
                document,
                native_evidence=[],
                audit_root=Path(temporary),
            )
        self.assertTrue(result["exactness"]["final_hidden_bit_exact"])
        self.assertTrue(result["exactness"]["final_state_payload_bit_exact"])
        self.assertEqual(
            result["exactness"]["comparison_domain"],
            "identical-prefix-boundary-continuation-abi",
        )
        self.assertEqual(
            result["peak_demand"]["baseline_median_model_source_bytes"], 507
        )
        self.assertEqual(
            result["peak_demand"]["battery_median_model_source_bytes"], 207
        )
        self.assertEqual(result["peak_demand"]["baseline_median_total_read_bytes"], 507)
        self.assertEqual(result["peak_demand"]["battery_median_total_read_bytes"], 3375)
        self.assertEqual(
            result["peak_demand"]["boundary_total_read_bytes_saved"], -2868
        )
        self.assertEqual(
            result["peak_demand"]["conservative_best_fresh_total_read_bytes_saved"],
            -2868,
        )
        self.assertNotIn("speedup", result["peak_demand"])
        self.assertTrue(result["peak_demand"]["anchor_charge_excluded"])
        self.assertGreaterEqual(
            result["arms"][0]["baseline_prefix_recompute_seconds"], 0.0
        )
        self.assertEqual(
            result["arms"][0]["route_protocol"],
            "cold-prefix+continuation-suffix",
        )
        self.assertEqual(
            result["arms"][1]["route_protocol"],
            "authenticated-anchor-restore+continuation-suffix",
        )
        self.assertEqual(result["arms"][1]["anchor"]["hit_count"], 1)
        self.assertEqual(
            result["arms"][0]["final_state"]["native_crsa_usage_tensor_count"],
            0,
        )
        self.assertEqual(
            [row["route"] for row in result["arms"]], list(document["schedule"])
        )
        self.assertEqual(result["arms"][1]["snapshot_restore_bytes"], 3168)
        self.assertEqual(result["arms"][2]["anchor"]["hit_count"], 2)
        self.assertTrue(result["exactness"]["one_shot_next_token_exact"])
        self.assertTrue(
            result["exactness"]["one_shot_vs_continuation_hidden_drift"]["exact"]
        )
        self.assertTrue(
            result["exactness"]["one_shot_vs_continuation_state_drift"][
                "all_tensors_within_dtype_bound"
            ]
        )
        self.assertTrue(
            result["exactness"]["one_shot_vs_continuation_hidden_drift"][
                "within_bound"
            ]
        )
        self.assertEqual(result["one_shot_reference"]["route"], "one-shot")
        self.assertEqual(
            result["one_shot_reference"]["route_protocol"], "best-live-one-shot"
        )

    def test_empty_suffix_uses_seed_hidden_and_exact_direct_head_scan(self) -> None:
        document = _input(suffix=())
        owner = _fake_owner()
        cache = _FakeCache()
        benchmark.charge_anchor(owner, cache, document, native_evidence=[])
        with tempfile.TemporaryDirectory() as temporary:
            result = benchmark.compare_arms(
                owner,
                cache,
                document,
                native_evidence=[],
                audit_root=Path(temporary),
            )
        self.assertTrue(result["exactness"]["head_scan_bit_exact"])
        self.assertIsNotNone(result["arms"][0]["head_scan"])
        self.assertIsNotNone(result["arms"][1]["head_scan"])
        self.assertEqual(result["arms"][0]["model_source_bytes"], 307)
        self.assertEqual(result["arms"][0]["snapshot_restore_bytes"], 0)
        self.assertEqual(result["arms"][0]["total_read_bytes"], 307)
        self.assertEqual(result["arms"][1]["model_source_bytes"], 7)
        self.assertEqual(result["arms"][1]["snapshot_restore_bytes"], 4096)
        self.assertEqual(result["arms"][1]["total_read_bytes"], 4103)

    def test_hidden_or_serialized_state_mismatch_fails_before_result(self) -> None:
        document = _input()
        owner = _fake_owner()
        cache = _FakeCache(wrong_restore=True)
        benchmark.charge_anchor(owner, cache, document, native_evidence=[])
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                benchmark.AnchorBatteryBenchmarkError,
                "hidden|state payload",
            ):
                benchmark.compare_arms(
                    owner,
                    cache,
                    document,
                    native_evidence=[],
                    audit_root=Path(temporary),
                )

    def test_one_shot_shared_continuation_drift_exceeding_dtype_bound_fails(
        self,
    ) -> None:
        document = _input()
        owner = _fake_owner(one_shot_drift=0.75)
        cache = _FakeCache()
        benchmark.charge_anchor(owner, cache, document, native_evidence=[])
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                benchmark.AnchorBatteryBenchmarkError,
                "relative-L2",
            ):
                benchmark.compare_arms(
                    owner,
                    cache,
                    document,
                    native_evidence=[],
                    audit_root=Path(temporary),
                )

    def test_one_shot_state_drift_exceeding_dtype_bound_fails(self) -> None:
        document = _input()
        owner = _fake_owner(one_shot_state_drift=0.75)
        cache = _FakeCache()
        benchmark.charge_anchor(owner, cache, document, native_evidence=[])
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                benchmark.AnchorBatteryBenchmarkError,
                "continuation state exceeds dtype-bound",
            ):
                benchmark.compare_arms(
                    owner,
                    cache,
                    document,
                    native_evidence=[],
                    audit_root=Path(temporary),
                )

    def test_state_audit_hashes_the_actual_payload_and_rejects_tamper(self) -> None:
        source = _FakeSource()
        model = _TamperedPayloadModel(source)
        model.state = 9
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                benchmark.AnchorBatteryBenchmarkError,
                "byte count|payload",
            ):
                benchmark._snapshot_audit(model, Path(temporary), "tampered")


class AnchorBatteryTinyRuntimeTests(unittest.TestCase):
    def test_bfloat16_snapshot_cache_replays_nonempty_suffix_bit_exact(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".qwen-anchor-benchmark-tiny-", dir=Path.cwd()
        ) as temporary:
            root = Path(temporary)
            model_root = root / "model"
            model_root.mkdir()
            config = _tiny_config()
            save_file(_tiny_weights(config), model_root / "model.safetensors")
            source = Streamer.from_local(model_root, budget_mb=20, use_cache=False)
            pager = Qwen38WeightPager(
                source,
                device="cpu",
                compute_dtype="bfloat16",
                max_resident_bytes=2 * 1024**2,
            )
            model = StreamedQwen38(config, pager, max_batch_size=1, max_seq_len=32)
            owner = SimpleNamespace(model=model, pager=pager)
            cache = SemanticStateAnchorCache(root / "cache")
            document = _input(prefix=(1, 4, 9), suffix=(7, 6))
            try:
                charge = benchmark.charge_anchor(
                    owner, cache, document, native_evidence=[]
                )
                result = benchmark.compare_arms(
                    owner,
                    cache,
                    document,
                    native_evidence=[],
                    audit_root=root / "audits",
                )
            finally:
                model.reset_state(release=True)
                pager.close()
                source.close()
            self.assertGreater(charge["cache_bytes"], 0)
            self.assertTrue(result["exactness"]["final_hidden_bit_exact"])
            self.assertTrue(result["exactness"]["final_state_payload_bit_exact"])
            self.assertTrue(result["exactness"]["one_shot_next_token_exact"])
            self.assertTrue(result["exactness"]["head_scan_bit_exact"])
            self.assertEqual(
                result["arms"][0]["final_state"]["payload_sha256"],
                result["arms"][1]["final_state"]["payload_sha256"],
            )

    def test_native_crsa_empty_suffix_restores_usage_seed_and_head_bit_exact(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".qwen-anchor-benchmark-native-", dir=Path.cwd()
        ) as temporary:
            root = Path(temporary)
            model_root = root / "model"
            model_root.mkdir()
            config = _native_tiny_config()
            save_file(_tiny_weights(config), model_root / "model.safetensors")
            source = Streamer.from_local(model_root, budget_mb=40, use_cache=False)
            pager = Qwen38WeightPager(
                source,
                device="cpu",
                compute_dtype="bfloat16",
                max_resident_bytes=4 * 1024**2,
            )
            native_evidence = []
            model = StreamedQwen38(
                config,
                pager,
                native_head_crsa=Qwen38NativeHeadCrsa(alpha=0.1),
                native_head_crsa_observer=native_evidence.append,
                max_batch_size=1,
                max_seq_len=32,
            )
            owner = SimpleNamespace(model=model, pager=pager)
            cache = SemanticStateAnchorCache(root / "cache")
            document = _input(prefix=(1, 4, 9), suffix=(), attention_mode="native-crsa")
            try:
                charge = benchmark.charge_anchor(
                    owner, cache, document, native_evidence=native_evidence
                )
                result = benchmark.compare_arms(
                    owner,
                    cache,
                    document,
                    native_evidence=native_evidence,
                    audit_root=root / "audits",
                )
            finally:
                model.reset_state(release=True)
                pager.close()
                source.close()
            self.assertEqual(charge["native_crsa"]["event_count"], 1)
            self.assertTrue(result["exactness"]["head_scan_bit_exact"])
            self.assertTrue(result["exactness"]["final_hidden_bit_exact"])
            self.assertTrue(result["exactness"]["final_state_payload_bit_exact"])
            self.assertEqual(
                result["arms"][0]["final_state"]["native_crsa_usage_tensor_count"],
                1,
            )
            self.assertEqual(
                result["arms"][1]["final_state"]["native_crsa_usage_tensor_count"],
                1,
            )
            self.assertGreater(result["arms"][1]["snapshot_restore_bytes"], 0)

    def test_native_crsa_multitoken_suffix_restore_is_bit_exact_at_same_boundary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".qwen-anchor-benchmark-native-suffix-", dir=Path.cwd()
        ) as temporary:
            root = Path(temporary)
            model_root = root / "model"
            model_root.mkdir()
            config = _native_tiny_config()
            save_file(_tiny_weights(config), model_root / "model.safetensors")
            source = Streamer.from_local(model_root, budget_mb=40, use_cache=False)
            pager = Qwen38WeightPager(
                source,
                device="cpu",
                compute_dtype="bfloat16",
                max_resident_bytes=4 * 1024**2,
            )
            native_evidence = []
            model = StreamedQwen38(
                config,
                pager,
                native_head_crsa=Qwen38NativeHeadCrsa(alpha=0.1),
                native_head_crsa_observer=native_evidence.append,
                max_batch_size=1,
                max_seq_len=32,
            )
            owner = SimpleNamespace(model=model, pager=pager)
            cache = SemanticStateAnchorCache(root / "cache")
            document = _input(
                prefix=(1, 4, 9),
                suffix=(7, 6, 5),
                attention_mode="native-crsa",
            )
            try:
                benchmark.charge_anchor(
                    owner, cache, document, native_evidence=native_evidence
                )
                result = benchmark.compare_arms(
                    owner,
                    cache,
                    document,
                    native_evidence=native_evidence,
                    audit_root=root / "audits",
                )
            finally:
                model.reset_state(release=True)
                pager.close()
                source.close()
            self.assertTrue(result["exactness"]["final_hidden_bit_exact"])
            self.assertTrue(result["exactness"]["final_state_payload_bit_exact"])
            self.assertTrue(result["exactness"]["one_shot_next_token_exact"])
            self.assertEqual(
                result["arms"][0]["final_state"][
                    "native_crsa_usage_tensor_count"
                ],
                1,
            )
            self.assertEqual(
                result["arms"][1]["final_state"][
                    "native_crsa_usage_tensor_count"
                ],
                1,
            )
            self.assertEqual(
                result["arms"][0]["final_state"]["payload_sha256"],
                result["arms"][1]["final_state"]["payload_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
