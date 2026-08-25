from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from immer.runtimes.deepseek_v4 import (
    CausalWeightMount,
    DeepSeekWeightPager,
    GENERAL_DENSE_COVERAGE_CAPABILITY,
    LogicalModelIdentity,
    tensor_range_plan_from_source,
)


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "deepseek_v4_poc.py"
PINNED_REVISION = "7" * 40


def _write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, np.ndarray]],
    *,
    tensor_order: Sequence[str] | None = None,
) -> None:
    header: dict[str, object] = {}
    payloads: list[bytes] = []
    offset = 0
    names = tuple(sorted(tensors)) if tensor_order is None else tuple(tensor_order)
    if len(names) != len(tensors) or set(names) != set(tensors):
        raise ValueError("tensor_order must contain every tensor exactly once")
    for name in names:
        dtype, value = tensors[name]
        array = np.ascontiguousarray(value)
        payload = array.tobytes(order="C")
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        payloads.append(payload)
        offset += len(payload)
    header["__metadata__"] = {"fixture": "immer-deepseek-v4-poc"}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(payloads))


def _tiny_config() -> dict[str, object]:
    return {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "vocab_size": 128,
        "hidden_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "head_dim": 128,
        "qk_rope_head_dim": 64,
        "q_lora_rank": 128,
        "o_lora_rank": 128,
        "o_groups": 1,
        "moe_intermediate_size": 128,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "num_hash_layers": 1,
        "rms_norm_eps": 1e-6,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 2,
        "hc_eps": 1e-6,
        "sliding_window": 16,
        "compress_ratios": [0],
        "rope_theta": 10_000,
        "compress_rope_theta": 40_000,
        "max_position_embeddings": 1024,
        "rope_scaling": {
            "original_max_position_embeddings": 1024,
            "factor": 2,
            "beta_fast": 32,
            "beta_slow": 1,
        },
        "index_n_heads": 2,
        "index_head_dim": 128,
        "index_topk": 16,
        "scoring_func": "sqrtsoftplus",
        "routed_scaling_factor": 1.5,
        "swiglu_limit": 10,
        "expert_dtype": "fp4",
        "dspark_target_layer_ids": [],
    }


def _fp8_matrix(
    rows: int, columns: int
) -> tuple[tuple[str, np.ndarray], tuple[str, np.ndarray]]:
    weight = ("F8_E4M3", np.zeros((rows, columns), dtype=np.uint8))
    scale = (
        "F8_E8M0",
        np.full(((rows + 127) // 128, (columns + 127) // 128), 127, dtype=np.uint8),
    )
    return weight, scale


def _fp4_matrix(
    rows: int, logical_columns: int
) -> tuple[tuple[str, np.ndarray], tuple[str, np.ndarray]]:
    weight = ("I8", np.zeros((rows, logical_columns // 2), dtype=np.int8))
    scale = (
        "F8_E8M0",
        np.full((rows, (logical_columns + 31) // 32), 127, dtype=np.uint8),
    )
    return weight, scale


def _bf16(value: np.ndarray) -> tuple[str, np.ndarray]:
    words = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)
    return "BF16", (words >> 16).astype(np.uint16)


def _tiny_checkpoint(root: Path, *, official_expert_layout: bool = False) -> None:
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(_tiny_config(), separators=(",", ":")), encoding="utf-8"
    )
    dim = qrank = orank = 128
    heads, head_dim, inter, hc = 2, 128, 128, 4
    mix = (2 + hc) * hc
    tensors: dict[str, tuple[str, np.ndarray]] = {
        "embed.weight": _bf16(
            (np.arange(128 * dim, dtype=np.float32).reshape(128, dim) % 97) / 1000,
        ),
        "norm.weight": _bf16(np.ones(dim, dtype=np.float32)),
        "head.weight": _bf16(np.eye(128, dim, dtype=np.float32)),
        "hc_head_fn": ("F32", np.zeros((hc, hc * dim), dtype=np.float32)),
        "hc_head_base": ("F32", np.zeros(hc, dtype=np.float32)),
        "hc_head_scale": ("F32", np.ones(1, dtype=np.float32)),
        "layers.0.attn.attn_sink": ("F32", np.zeros(heads, dtype=np.float32)),
        "layers.0.attn.q_norm.weight": _bf16(np.ones(qrank, dtype=np.float32)),
        "layers.0.attn.kv_norm.weight": _bf16(np.ones(head_dim, dtype=np.float32)),
        "layers.0.attn_norm.weight": _bf16(np.ones(dim, dtype=np.float32)),
        "layers.0.ffn_norm.weight": _bf16(np.ones(dim, dtype=np.float32)),
        "layers.0.ffn.gate.weight": _bf16(np.zeros((2, dim), dtype=np.float32)),
        "layers.0.ffn.gate.tid2eid": (
            "I64",
            np.zeros((128, 1), dtype=np.int64),
        ),
    }
    dense_shapes = {
        "layers.0.attn.wq_a": (qrank, dim),
        "layers.0.attn.wq_b": (heads * head_dim, qrank),
        "layers.0.attn.wkv": (head_dim, dim),
        "layers.0.attn.wo_a": (orank, heads * head_dim),
        "layers.0.attn.wo_b": (dim, orank),
        "layers.0.ffn.shared_experts.w1": (inter, dim),
        "layers.0.ffn.shared_experts.w2": (dim, inter),
        "layers.0.ffn.shared_experts.w3": (inter, dim),
    }
    for prefix, shape in dense_shapes.items():
        weight, scale = _fp8_matrix(*shape)
        tensors[f"{prefix}.weight"] = weight
        tensors[f"{prefix}.scale"] = scale
    for branch in ("attn", "ffn"):
        tensors[f"layers.0.hc_{branch}_fn"] = (
            "F32",
            np.zeros((mix, hc * dim), dtype=np.float32),
        )
        tensors[f"layers.0.hc_{branch}_base"] = (
            "F32",
            np.zeros(mix, dtype=np.float32),
        )
        tensors[f"layers.0.hc_{branch}_scale"] = (
            "F32",
            np.ones(3, dtype=np.float32),
        )
    for expert in range(2):
        for projection, shape in {
            "w1": (inter, dim),
            "w2": (dim, inter),
            "w3": (inter, dim),
        }.items():
            prefix = f"layers.0.ffn.experts.{expert}.{projection}"
            weight, scale = _fp4_matrix(*shape)
            tensors[f"{prefix}.weight"] = weight
            tensors[f"{prefix}.scale"] = scale
    tensor_order = None
    if official_expert_layout:
        expert_names = {
            name for name in tensors if ".ffn.experts." in name
        }
        tensor_order = [name for name in sorted(tensors) if name not in expert_names]
        for expert in range(2):
            base = f"layers.0.ffn.experts.{expert}"
            tensor_order.extend(
                f"{base}.{projection}.{part}"
                for part in ("scale", "weight")
                for projection in ("w1", "w2", "w3")
            )
    _write_safetensors(
        root / "model.safetensors", tensors, tensor_order=tensor_order
    )


def _flat_causal_bundle(root: Path, *, bind_expert_zero: bool) -> None:
    (root / "causal").mkdir()
    body = {
        "capabilities": {
            "general_dense_weight_coverage": GENERAL_DENSE_COVERAGE_CAPABILITY
        },
        "weights_layout": "flat/v1",
    }
    encoded = json.dumps(
        body,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    manifest = {
        "body": body,
        "schema": "immer.deepseek-v4-flat-causal-fixture/v1",
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }
    (root / "bundle.json").write_text(
        json.dumps(manifest, separators=(",", ":")), encoding="utf-8"
    )
    model = LogicalModelIdentity("fixture/deepseek-v4", PINNED_REVISION)
    with CausalWeightMount(root, model, budget_mb=4) as mount:
        mount.bind_tensor_plans(
            tensor_range_plan_from_source(mount.source, entry["name"])
            for entry in mount.source.inventory().get("tensors", ())
        )
        pager = DeepSeekWeightPager(
            mount.source,
            device="cpu",
            compute_dtype="float32",
            simulate_activation_quantization=False,
            expert_prefetch=False,
        )
        try:
            if bind_expert_zero:
                mount.bind_plans(pager.plan_expert_ranges(0, (0, 1)))
        finally:
            pager.release()


class DeepSeekV4PocScriptTests(unittest.TestCase):
    def _run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "src")
        return subprocess.run(
            [sys.executable, str(SCRIPT), *arguments],
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )

    def test_help_exposes_preflight_generation_and_one_token_controls(self) -> None:
        root = self._run("--help")
        self.assertEqual(root.returncode, 0, root.stderr)
        self.assertIn("preflight", root.stdout)
        self.assertIn("one-token", root.stdout)
        self.assertIn("generate", root.stdout)
        one = self._run("one-token", "--help")
        self.assertEqual(one.returncode, 0, one.stderr)
        for option in (
            "--source",
            "--causal-bundle",
            "--logical-repo-id",
            "--revision",
            "--config",
            "--cache-dir",
            "--max-cache-gb",
            "--budget-mb",
            "--device",
            "--dtype",
            "--no-expert-prefetch",
            "--top-k",
            "--head-block-rows",
            "--progress-jsonl",
            "--output-json",
        ):
            self.assertIn(option, one.stdout)
        generate = self._run("generate", "--help")
        self.assertEqual(generate.returncode, 0, generate.stderr)
        for option in (
            "--prompt",
            "--token-ids",
            "--max-new-tokens",
            "--prefill-mode",
            "--context-limit",
            "--graft-mode",
            "--graft-alpha",
            "--exact-cascade",
        ):
            self.assertIn(option, generate.stdout)

    def test_missing_absolute_source_path_fails_locally_without_remote_fallback(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "immer-v4-source-does-not-exist"
            output = root / "error.json"
            result = self._run(
                "preflight",
                "--source",
                str(missing),
                "--budget-mb",
                "1",
                "--no-cache",
                "--debug",
                "--output-json",
                str(output),
            )
            persisted = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result.returncode, 1)
        report = json.loads(result.stdout)
        self.assertEqual(persisted, report)
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["error"]["type"], "FileNotFoundError")
        self.assertIn(
            "local source directory does not exist", report["error"]["message"]
        )
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn(str(root.resolve()), encoded)
        self.assertNotIn(str(ROOT.resolve()), encoded)
        self.assertIn("<external-path>", report["error"]["message"])

    def test_remote_source_requires_immutable_revision_before_network_access(
        self,
    ) -> None:
        result = self._run(
            "preflight",
            "--source",
            "fixture/remote",
            "--revision",
            "main",
            "--budget-mb",
            "1",
            "--no-cache",
        )
        self.assertEqual(result.returncode, 1)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["error"]["type"], "DeepSeekRuntimeSourceError")
        self.assertIn("immutable", report["error"]["message"])

    def test_generic_poc_rejects_trace_sparse_bundle_before_weight_access(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "sparse"
            bundle.mkdir()
            (bundle / "bundle.json").write_text(
                json.dumps(
                    {
                        "schema": "immer.deepseek-v4-sparse-causal-bundle/v1",
                        "sha256": "0" * 64,
                    }
                ),
                encoding="utf-8",
            )
            result = self._run(
                "preflight",
                "--causal-bundle",
                str(bundle),
                "--logical-repo-id",
                "fixture/deepseek-v4",
                "--revision",
                PINNED_REVISION,
                "--budget-mb",
                "1",
            )
        self.assertEqual(result.returncode, 1)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "error")
        self.assertIn("decode-trace-specific", report["error"]["message"])

    def test_tiny_local_checkpoint_preflight_and_one_token(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            checkpoint = work / "checkpoint"
            _tiny_checkpoint(checkpoint)
            cache = work / "cache"
            preflight_output = work / "preflight.json"
            preflight_progress = work / "preflight.jsonl"
            common = (
                "--source",
                str(checkpoint),
                "--revision",
                "local-fixture-v1",
                "--cache-dir",
                str(cache),
                "--budget-mb",
                "4",
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--no-activation-quantization",
            )
            preflight = self._run(
                "preflight",
                *common,
                "--progress-jsonl",
                str(preflight_progress),
                "--output-json",
                str(preflight_output),
            )
            self.assertEqual(preflight.returncode, 0, preflight.stderr)
            report = json.loads(preflight.stdout)
            self.assertEqual(report["status"], "ok")
            self.assertNotIn(str(work.resolve()), json.dumps(report, sort_keys=True))
            self.assertEqual(report["mode"], "preflight")
            self.assertTrue(report["preflight"]["exhaustive_experts"])
            self.assertGreater(report["preflight"]["required_tensors"], 40)
            self.assertTrue(report["provenance"]["revision_is_mutable"])
            self.assertEqual(
                report["provenance"]["execution"]["expert_prefetch_policy"],
                "exact-router-window-q3-a2/v2",
            )
            execution = report["provenance"]["execution"]
            self.assertEqual(
                execution["expert_prefetch_transport_policy"],
                "streamer-exact-range/v1",
            )
            self.assertEqual(execution["expert_prefetch_workers"], 2)
            self.assertEqual(execution["expert_prefetch_active_read_limit"], 2)
            self.assertEqual(execution["expert_prefetch_max_outstanding"], 3)
            self.assertEqual(execution["expert_prefetch_max_experts"], 3)
            self.assertEqual(execution["expert_range_coalesce_max_experts"], 1)
            self.assertEqual(execution["expert_range_coalesce_max_gap_bytes"], 0)
            self.assertEqual(
                execution["expert_prefetch_resident_limit_bytes"],
                48 * 1024**2,
            )
            self.assertEqual(execution["source_transport_policy"], "local-range/v1")
            self.assertEqual(execution["source_transport_connection_limit"], 0)
            self.assertRegex(
                report["provenance"]["runtime_source_sha256"],
                r"^[0-9a-f]{64}$",
            )
            self.assertEqual(
                report["provenance"]["harness_sha256"],
                hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
            )
            self.assertRegex(
                report["provenance"]["runtime_dependency_sha256"],
                r"^[0-9a-f]{64}$",
            )
            self.assertEqual(
                set(report["provenance"]["runtime_dependencies"]),
                {"python", "torch", "numpy", "requests", "safetensors"},
            )
            self.assertEqual(json.loads(preflight_output.read_text()), report)
            events = [
                json.loads(line)["event"]
                for line in preflight_progress.read_text().splitlines()
            ]
            self.assertEqual(events[0], "run_start")
            self.assertIn("inventory_ready", events)
            self.assertEqual(events[-1], "run_complete")

            synchronous = self._run(
                "preflight",
                *common,
                "--no-expert-prefetch",
                "--sampled-experts",
            )
            self.assertEqual(synchronous.returncode, 0, synchronous.stderr)
            synchronous_report = json.loads(synchronous.stdout)
            self.assertEqual(
                synchronous_report["provenance"]["execution"]["expert_prefetch_policy"],
                "disabled",
            )
            self.assertEqual(
                synchronous_report["provenance"]["execution"][
                    "expert_prefetch_transport_policy"
                ],
                "disabled",
            )
            self.assertEqual(
                synchronous_report["provenance"]["runtime_source_sha256"],
                report["provenance"]["runtime_source_sha256"],
            )

            automatic = self._run(
                "preflight",
                "--source",
                str(checkpoint),
                "--revision",
                "local-fixture-v1",
                "--cache-dir",
                str(cache),
                "--budget-mb",
                "4",
                "--device",
                "auto",
                "--dtype",
                "auto",
                "--sampled-experts",
            )
            self.assertEqual(automatic.returncode, 0, automatic.stderr)
            automatic_execution = json.loads(automatic.stdout)["provenance"][
                "execution"
            ]
            self.assertIn(automatic_execution["device"], {"cpu", "mps"})
            self.assertEqual(automatic_execution["compute_dtype"], "bfloat16")

            one_progress = work / "one.jsonl"
            one = self._run(
                "one-token",
                *common,
                "--token-id",
                "7",
                "--top-k",
                "3",
                "--head-block-rows",
                "32",
                "--progress-jsonl",
                str(one_progress),
            )
            self.assertEqual(one.returncode, 0, one.stderr)
            result = json.loads(one.stdout)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["one_token"]["evidence"]["token_id"], 7)
            self.assertTrue(result["one_token"]["evidence"]["complete_layer_stack"])
            self.assertEqual(
                result["one_token"]["evidence"]["context_mode"],
                "isolated_position_zero",
            )
            self.assertFalse(result["one_token"]["evidence"]["stateful_kv_cache"])
            self.assertFalse(result["exactness"]["general_generation"])
            self.assertEqual(len(result["one_token"]["logits"]["token_ids"]), 3)
            self.assertGreater(result["one_token"]["evidence"]["linear_calls"], 0)
            one_events = [
                json.loads(line)["event"]
                for line in one_progress.read_text().splitlines()
            ]
            self.assertIn("layer_complete", one_events)
            self.assertIn("head_block_complete", one_events)
            self.assertEqual(one_events[-1], "run_complete")

            generated = self._run(
                "generate",
                *common,
                "--token-ids",
                "7,8",
                "--max-new-tokens",
                "2",
                "--context-limit",
                "8",
                "--head-block-rows",
                "32",
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)
            generation = json.loads(generated.stdout)
            self.assertEqual(generation["status"], "ok")
            self.assertTrue(generation["exactness"]["general_generation"])
            self.assertTrue(generation["exactness"]["stateful_kv_cache"])
            self.assertEqual(generation["generation"]["prompt_token_ids"], [7, 8])
            self.assertEqual(len(generation["generation"]["generated_token_ids"]), 2)
            self.assertEqual(
                generation["generation"]["evidence"]["context_mode"],
                "stateful_autoregressive",
            )
            self.assertEqual(
                generation["generation"]["evidence"]["prefill_mode"], "batched"
            )
            self.assertEqual(generation["generation"]["evidence"]["forward_passes"], 2)

    def test_runtime_error_is_json_and_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            checkpoint = work / "checkpoint"
            _tiny_checkpoint(checkpoint)
            failed = self._run(
                "one-token",
                "--source",
                str(checkpoint),
                "--revision",
                "local-fixture-v1",
                "--cache-dir",
                str(work / "cache"),
                "--budget-mb",
                "4",
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--token-id",
                "999",
            )
            self.assertNotEqual(failed.returncode, 0)
            report = json.loads(failed.stdout)
            self.assertEqual(report["status"], "error")
            self.assertEqual(report["error"]["type"], "ValueError")
            self.assertIn("vocabulary", report["error"]["message"])

    def test_flat_causal_bundle_uses_strict_reader_without_copying_weights(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            _tiny_checkpoint(checkpoint, official_expert_layout=True)
            weight_before = (checkpoint / "model.safetensors").stat()
            _flat_causal_bundle(checkpoint, bind_expert_zero=True)
            result = self._run(
                "one-token",
                "--causal-bundle",
                str(checkpoint),
                "--logical-repo-id",
                "fixture/deepseek-v4",
                "--revision",
                PINNED_REVISION,
                "--budget-mb",
                "4",
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--no-activation-quantization",
                "--no-expert-prefetch",
                "--token-id",
                "7",
                "--top-k",
                "1",
                "--head-block-rows",
                "32",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertTrue(report["pager"]["causal_weight_reader_attached"])
            self.assertTrue(report["pager"]["causal_tensor_reader_attached"])
            self.assertFalse(report["pager"]["causal_missing_fallback"])
            self.assertGreater(report["pager"]["causal_expert_plan_hits"], 0)
            self.assertEqual(report["pager"]["causal_expert_plan_fallbacks"], 0)
            weight_after = (checkpoint / "model.safetensors").stat()
            self.assertEqual(
                (weight_after.st_ino, weight_after.st_size),
                (weight_before.st_ino, weight_before.st_size),
            )
            self.assertFalse((checkpoint / "weights").exists())

    def test_flat_causal_bundle_missing_sparse_route_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint"
            _tiny_checkpoint(checkpoint, official_expert_layout=True)
            _flat_causal_bundle(checkpoint, bind_expert_zero=False)
            preflight = self._run(
                "preflight",
                "--causal-bundle",
                str(checkpoint),
                "--logical-repo-id",
                "fixture/deepseek-v4",
                "--revision",
                PINNED_REVISION,
                "--budget-mb",
                "4",
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--no-activation-quantization",
                "--no-expert-prefetch",
            )
            self.assertEqual(preflight.returncode, 1)
            preflight_report = json.loads(preflight.stdout)
            self.assertEqual(preflight_report["status"], "error")
            self.assertIn(
                "causal weight binding is missing",
                preflight_report["error"]["message"],
            )
            result = self._run(
                "one-token",
                "--causal-bundle",
                str(checkpoint),
                "--logical-repo-id",
                "fixture/deepseek-v4",
                "--revision",
                PINNED_REVISION,
                "--budget-mb",
                "4",
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--no-activation-quantization",
                "--no-expert-prefetch",
                "--token-id",
                "7",
            )
            self.assertEqual(result.returncode, 1)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "error")
            self.assertIn(
                "causal weight binding is missing", report["error"]["message"]
            )


if __name__ == "__main__":
    unittest.main()
