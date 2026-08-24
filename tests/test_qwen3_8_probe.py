from __future__ import annotations

import hashlib
import json
import math
import unittest

from immer.runtimes.qwen3_8 import (
    DELTANET_COMPARISON_SCHEMA,
    DeltaNetProbe,
    DeltaNetProbeRecorder,
    Qwen38DeltaNetProbeError,
    build_probe_document,
    compare_probe_documents,
    verify_probe_document,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _probe(value: float) -> DeltaNetProbe:
    return DeltaNetProbe(
        beta_mean=value,
        beta_std=value + 0.1,
        decay_mean=value + 0.2,
        decay_std=value + 0.3,
        conv_norm=value + 0.4,
        q_norm=value + 0.5,
        k_norm=value + 0.6,
        v_norm=value + 0.7,
        delta_norm=value + 0.8,
    )


def _document(value: float, tag: str) -> dict:
    recorder = DeltaNetProbeRecorder()
    recorder(0, _probe(value))
    recorder(1, _probe(value + 1.0))
    return build_probe_document(
        recorder,
        checkpoint={"repo_id": "fixture", "revision": "a" * 40},
        context_mode="prefill",
        start_pos=0,
        end_pos=3,
        item_id=tag,
        input_sha256=_digest(f"input:{tag}"),
        hidden_sha256=_digest(f"hidden:{tag}"),
    )


class Qwen38DeltaNetProbeTests(unittest.TestCase):
    def test_document_binds_records_and_rejects_tamper(self) -> None:
        document = _document(1.0, "sample")
        verified = verify_probe_document(document)
        self.assertEqual(verified, document)
        self.assertEqual(
            [(row["pass_index"], row["layer"]) for row in document["body"]["records"]],
            [(0, 0), (0, 1)],
        )

        document["body"]["records"][0]["delta_norm"] += 1.0
        with self.assertRaisesRegex(Qwen38DeltaNetProbeError, "SHA-256"):
            verify_probe_document(document)

    def test_comparison_reproduces_pooled_sample_cohen_d(self) -> None:
        comparison = compare_probe_documents(
            [_document(3.0, "r0"), _document(5.0, "r1")],
            [_document(1.0, "k0"), _document(2.0, "k1")],
        )

        self.assertEqual(comparison["schema"], DELTANET_COMPARISON_SCHEMA)
        beta = comparison["body"]["statistics"]["0"]["beta_mean"]
        self.assertAlmostEqual(beta["mu_reasoning"], 4.0)
        self.assertAlmostEqual(beta["mu_knowledge"], 1.5)
        self.assertAlmostEqual(beta["pooled_std"], math.sqrt(1.25))
        self.assertAlmostEqual(beta["d"], 2.5 / (math.sqrt(1.25) + 1e-12))
        self.assertEqual(
            comparison["body"]["families"],
            {
                "knowledge": 2,
                "reasoning": 2,
            },
        )
        self.assertEqual(comparison["body"]["layers"], [0, 1])
        self.assertIn("0", comparison["body"]["protected_layers"])

    def test_comparison_rejects_duplicate_inputs_and_mixed_checkpoints(self) -> None:
        duplicate = _document(1.0, "same")
        with self.assertRaisesRegex(Qwen38DeltaNetProbeError, "duplicated"):
            compare_probe_documents(
                [_document(3.0, "r0"), duplicate],
                [_document(2.0, "k0"), duplicate],
            )

        changed = _document(2.0, "k1")
        changed["body"]["checkpoint"]["revision"] = "b" * 40
        changed["sha256"] = hashlib.sha256(
            json.dumps(
                changed["body"],
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(Qwen38DeltaNetProbeError, "checkpoints differ"):
            compare_probe_documents(
                [_document(3.0, "r0"), _document(4.0, "r1")],
                [_document(1.0, "k0"), changed],
            )


if __name__ == "__main__":
    unittest.main()
