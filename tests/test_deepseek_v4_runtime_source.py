from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from immer.runtimes.deepseek_v4 import LogicalModelIdentity
from immer.runtimes.deepseek_v4 import runtime_source as runtime


PINNED_REVISION = "7" * 40


def _write_general_manifest(root: Path, *, capability: bool = True) -> None:
    body: dict[str, object] = {"weights_layout": "nested/v1"}
    if capability:
        body["capabilities"] = {
            "general_dense_weight_coverage": (
                runtime.GENERAL_DENSE_COVERAGE_CAPABILITY
            )
        }
    encoded = json.dumps(
        body,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    (root / "bundle.json").write_text(
        json.dumps(
            {
                "body": body,
                "schema": "immer.deepseek-v4-general-causal-bundle/v1",
                "sha256": hashlib.sha256(encoded).hexdigest(),
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


class DeepSeekRuntimeSourceTests(unittest.TestCase):
    def test_causal_bundle_mount_owns_source_reader_and_close(self) -> None:
        mounted_source = mock.Mock()
        reader = object()
        tensor_reader = object()
        mount = mock.Mock(
            source=mounted_source,
            reader=reader,
            tensor_reader=tensor_reader,
        )
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "fixture-causal-bundle"
            bundle.mkdir()
            _write_general_manifest(bundle)
            with mock.patch.object(
                runtime, "CausalWeightMount", return_value=mount
            ) as cls:
                opened = runtime.open_deepseek_runtime_source(
                    source="ignored/remote",
                    revision=PINNED_REVISION,
                    logical_repo_id="fixture/deepseek-v4",
                    causal_bundle=bundle,
                    budget_mb=64,
                    cache_dir="/tmp/unused-cache",
                    use_cache=True,
                    max_cache_bytes=123,
                )

            self.assertIs(opened.source, mounted_source)
            self.assertIs(opened.causal_weight_reader, reader)
            self.assertIs(opened.causal_tensor_reader, tensor_reader)
            self.assertTrue(opened.is_causal_bundle)
            self.assertEqual(
                opened.logical_model,
                LogicalModelIdentity("fixture/deepseek-v4", PINNED_REVISION),
            )
            self.assertEqual(
                opened.label,
                f"causal-bundle:fixture/deepseek-v4@{PINNED_REVISION[:12]}",
            )
            cls.assert_called_once_with(
                bundle,
                LogicalModelIdentity("fixture/deepseek-v4", PINNED_REVISION),
                budget_mb=64.0,
                verbose=False,
            )
        opened.close()
        opened.close()
        mount.close.assert_called_once_with()
        mounted_source.close.assert_not_called()

    def test_causal_bundle_rejects_mutable_revision_before_mount(self) -> None:
        with (
            mock.patch.object(runtime, "CausalWeightMount") as mount,
            self.assertRaisesRegex(
                runtime.DeepSeekRuntimeSourceError, "immutable.*revision digest"
            ),
        ):
            runtime.open_deepseek_runtime_source(
                source="ignored/remote",
                revision="main",
                logical_repo_id="fixture/deepseek-v4",
                causal_bundle="/tmp/fixture-causal-bundle",
                budget_mb=64,
                cache_dir="/tmp/unused-cache",
                use_cache=False,
                max_cache_bytes=0,
            )
        mount.assert_not_called()

    def test_local_source_preserves_in_place_streaming_without_mount(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint"
            checkpoint.mkdir()
            streamer = mock.Mock()
            with mock.patch.object(
                runtime.Streamer, "from_local", return_value=streamer
            ) as local:
                opened = runtime.open_deepseek_runtime_source(
                    source=str(checkpoint),
                    revision="local-fixture-v1",
                    logical_repo_id="unused/logical-id",
                    causal_bundle=None,
                    budget_mb=4,
                    cache_dir=Path(temporary) / "cache",
                    use_cache=True,
                    max_cache_bytes=1024,
                )

        self.assertIs(opened.source, streamer)
        self.assertIsNone(opened.causal_weight_reader)
        self.assertIsNone(opened.causal_tensor_reader)
        self.assertFalse(opened.is_causal_bundle)
        self.assertEqual(opened.label, "local:<external>")
        kwargs = local.call_args.kwargs
        self.assertEqual(kwargs["revision"], "local-fixture-v1")
        self.assertNotIn("repo_id", kwargs)
        self.assertIsNone(kwargs["pinned_inventory"])
        opened.close()
        streamer.close.assert_called_once_with()

    def test_remote_pin_policy_can_be_required_by_general_runners(self) -> None:
        source = mock.Mock()
        with mock.patch.object(runtime, "Streamer", return_value=source) as streamer:
            opened = runtime.open_deepseek_runtime_source(
                source="fixture/remote",
                revision="development-branch",
                logical_repo_id="fixture/remote",
                causal_bundle=None,
                budget_mb=4,
                cache_dir="/tmp/cache",
                use_cache=False,
                max_cache_bytes=0,
                require_remote_pinned_revision=False,
            )
        self.assertIs(opened.source, source)
        streamer.assert_called_once()
        opened.close()

        with self.assertRaisesRegex(
            runtime.DeepSeekRuntimeSourceError, "remote source requires.*revision"
        ):
            runtime.open_deepseek_runtime_source(
                source="fixture/remote",
                revision="development-branch",
                logical_repo_id="fixture/remote",
                causal_bundle=None,
                budget_mb=4,
                cache_dir="/tmp/cache",
                use_cache=False,
                max_cache_bytes=0,
                require_remote_pinned_revision=True,
            )

    def test_trace_sparse_and_unproven_bundles_are_rejected_before_mount(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sparse = root / "sparse"
            sparse.mkdir()
            (sparse / "bundle.json").write_text(
                json.dumps(
                    {
                        "schema": "immer.deepseek-v4-sparse-causal-bundle/v1",
                        "sha256": "0" * 64,
                    }
                ),
                encoding="utf-8",
            )
            unproven = root / "unproven"
            unproven.mkdir()
            _write_general_manifest(unproven, capability=False)
            with mock.patch.object(runtime, "CausalWeightMount") as mount:
                with self.assertRaisesRegex(
                    runtime.DeepSeekRuntimeSourceError,
                    "decode-trace-specific",
                ):
                    runtime.open_deepseek_runtime_source(
                        source="ignored",
                        revision=PINNED_REVISION,
                        logical_repo_id="fixture/deepseek-v4",
                        causal_bundle=sparse,
                        budget_mb=4,
                        cache_dir=root / "cache",
                        use_cache=False,
                        max_cache_bytes=0,
                    )
                with self.assertRaisesRegex(
                    runtime.DeepSeekRuntimeSourceError,
                    "complete dense-weight coverage",
                ):
                    runtime.open_deepseek_runtime_source(
                        source="ignored",
                        revision=PINNED_REVISION,
                        logical_repo_id="fixture/deepseek-v4",
                        causal_bundle=unproven,
                        budget_mb=4,
                        cache_dir=root / "cache",
                        use_cache=False,
                        max_cache_bytes=0,
                    )
            mount.assert_not_called()


if __name__ == "__main__":
    unittest.main()
