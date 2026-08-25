from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from immer.runtimes.deepseek_v4 import CausalWeightLayoutIdentity, LogicalModelIdentity
from immer.runtimes.deepseek_v4 import runtime_source as runtime


PINNED_REVISION = "7" * 40
REMOTE_REPO = "fixture/deepseek-v4"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _remote_inventory_document(
    *,
    repo_id: str = REMOTE_REPO,
    revision: str = PINNED_REVISION,
) -> dict[str, object]:
    inventory = {
        "repo": repo_id,
        "revision": revision,
        "shards": [
            {
                "cas_url_hash": "a" * 64,
                "data_start": 128,
                "etag": '"' + "b" * 64 + '"',
                "file": "model-00001-of-00001.safetensors",
                "header_len": 120,
                "size": 130,
            }
        ],
        "tensors": [],
    }
    return {
        "inventory": inventory,
        "inventory_sha256": _canonical_sha256(inventory),
        "repo_id": repo_id,
        "revision": revision,
        "schema": "immer.tensor-inventory-cache/v1",
        "source_fingerprint": runtime.Streamer._source_fingerprint(inventory),
    }


def _write_remote_inventory(path: Path, document: dict[str, object]) -> None:
    path.write_text(
        json.dumps(document, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )


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
    def test_remote_pinned_inventory_is_adopted_before_observer_attachment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pinned = root / "inventory.pinned.json"
            document = _remote_inventory_document()
            _write_remote_inventory(pinned, document)
            fingerprint = str(document["source_fingerprint"])
            inventory_sha256 = str(document["inventory_sha256"])
            observer = object()
            streamer_source = mock.Mock()
            streamer_source.adopt_pinned_inventory.return_value = fingerprint
            streamer_source.metrics.return_value = {
                "repo_id": REMOTE_REPO,
                "revision": PINNED_REVISION,
                "inventory_source_fingerprint": fingerprint,
            }

            with mock.patch.object(
                runtime, "Streamer", return_value=streamer_source
            ) as streamer:
                opened = runtime.open_deepseek_runtime_source(
                    source=REMOTE_REPO,
                    revision=PINNED_REVISION,
                    logical_repo_id=REMOTE_REPO,
                    causal_bundle=None,
                    budget_mb=64,
                    cache_dir=root / "cache",
                    use_cache=True,
                    max_cache_bytes=123,
                    access_observer=observer,
                    require_remote_pinned_revision=True,
                    remote_pinned_inventory=pinned,
                )

            self.assertIs(opened.source, streamer_source)
            self.assertTrue(opened.remote_pinned_inventory_adopted)
            self.assertEqual(
                opened.remote_pinned_inventory_fingerprint, fingerprint
            )
            self.assertEqual(
                opened.remote_pinned_inventory_sha256, inventory_sha256
            )
            self.assertIsNone(streamer.call_args.kwargs["access_observer"])
            self.assertEqual(
                streamer_source.method_calls,
                [
                    mock.call.adopt_pinned_inventory(
                        document["inventory"],
                        expected_fingerprint=fingerprint,
                    ),
                    mock.call.metrics(),
                    mock.call.set_access_observer(
                        observer,
                        prepare_identity=False,
                    ),
                ],
            )
            opened.close()

    def test_remote_pinned_inventory_rejects_wrong_identity_and_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases: dict[str, tuple[dict[str, object], str]] = {}

            wrong_repo = _remote_inventory_document(repo_id="foreign/model")
            cases["repo"] = (wrong_repo, "repo/revision")

            wrong_revision = _remote_inventory_document(revision="8" * 40)
            cases["revision"] = (wrong_revision, "repo/revision")

            wrong_digest = _remote_inventory_document()
            wrong_digest["inventory_sha256"] = "0" * 64
            cases["digest"] = (wrong_digest, "SHA-256")

            wrong_fingerprint = _remote_inventory_document()
            wrong_fingerprint["source_fingerprint"] = "0" * 64
            cases["fingerprint"] = (wrong_fingerprint, "fingerprint")

            inner_identity = _remote_inventory_document()
            inner_inventory = dict(inner_identity["inventory"])
            inner_inventory["repo"] = "foreign/model"
            inner_identity["inventory"] = inner_inventory
            inner_identity["inventory_sha256"] = _canonical_sha256(inner_inventory)
            inner_identity["source_fingerprint"] = (
                runtime.Streamer._source_fingerprint(inner_inventory)
            )
            cases["inner_repo"] = (inner_identity, "repo/revision")

            for name, (document, message) in cases.items():
                with self.subTest(name=name):
                    pinned = root / f"{name}.json"
                    _write_remote_inventory(pinned, document)
                    with (
                        mock.patch.object(runtime, "Streamer") as streamer,
                        self.assertRaisesRegex(
                            runtime.DeepSeekRuntimeSourceError, message
                        ),
                    ):
                        runtime.open_deepseek_runtime_source(
                            source=REMOTE_REPO,
                            revision=PINNED_REVISION,
                            logical_repo_id=REMOTE_REPO,
                            causal_bundle=None,
                            budget_mb=64,
                            cache_dir=root / "cache",
                            use_cache=True,
                            max_cache_bytes=123,
                            remote_pinned_inventory=pinned,
                        )
                    streamer.assert_not_called()

    def test_remote_pinned_inventory_rejects_symlink_local_and_causal_sources(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pinned = root / "inventory.pinned.json"
            _write_remote_inventory(pinned, _remote_inventory_document())
            linked = root / "linked.json"
            linked.symlink_to(pinned)
            with self.assertRaisesRegex(
                runtime.DeepSeekRuntimeSourceError, "regular non-symlink"
            ):
                runtime.open_deepseek_runtime_source(
                    source=REMOTE_REPO,
                    revision=PINNED_REVISION,
                    logical_repo_id=REMOTE_REPO,
                    causal_bundle=None,
                    budget_mb=64,
                    cache_dir=root / "cache",
                    use_cache=True,
                    max_cache_bytes=123,
                    remote_pinned_inventory=linked,
                )

            local = root / "local-checkpoint"
            local.mkdir()
            with self.assertRaisesRegex(
                runtime.DeepSeekRuntimeSourceError, "local source"
            ):
                runtime.open_deepseek_runtime_source(
                    source=str(local),
                    revision="local",
                    logical_repo_id=REMOTE_REPO,
                    causal_bundle=None,
                    budget_mb=64,
                    cache_dir=root / "cache",
                    use_cache=True,
                    max_cache_bytes=123,
                    remote_pinned_inventory=pinned,
                )

            with self.assertRaisesRegex(
                runtime.DeepSeekRuntimeSourceError, "causal bundle"
            ):
                runtime.open_deepseek_runtime_source(
                    source=REMOTE_REPO,
                    revision=PINNED_REVISION,
                    logical_repo_id=REMOTE_REPO,
                    causal_bundle=root / "bundle",
                    budget_mb=64,
                    cache_dir=root / "cache",
                    use_cache=True,
                    max_cache_bytes=123,
                    remote_pinned_inventory=pinned,
                )

    def test_remote_pinned_inventory_adoption_failure_closes_streamer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pinned = root / "inventory.pinned.json"
            _write_remote_inventory(pinned, _remote_inventory_document())
            streamer_source = mock.Mock()
            streamer_source.adopt_pinned_inventory.side_effect = RuntimeError("no")
            with (
                mock.patch.object(
                    runtime, "Streamer", return_value=streamer_source
                ),
                self.assertRaisesRegex(
                    runtime.DeepSeekRuntimeSourceError,
                    "rejected the pinned inventory",
                ),
            ):
                runtime.open_deepseek_runtime_source(
                    source=REMOTE_REPO,
                    revision=PINNED_REVISION,
                    logical_repo_id=REMOTE_REPO,
                    causal_bundle=None,
                    budget_mb=64,
                    cache_dir=root / "cache",
                    use_cache=True,
                    max_cache_bytes=123,
                    access_observer=object(),
                    remote_pinned_inventory=pinned,
                )
            streamer_source.close.assert_called_once_with()
            streamer_source.set_access_observer.assert_not_called()

    def test_remote_pinned_inventory_observer_failure_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pinned = root / "inventory.pinned.json"
            document = _remote_inventory_document()
            _write_remote_inventory(pinned, document)
            fingerprint = str(document["source_fingerprint"])
            streamer_source = mock.Mock()
            streamer_source.adopt_pinned_inventory.return_value = fingerprint
            streamer_source.metrics.return_value = {
                "repo_id": REMOTE_REPO,
                "revision": PINNED_REVISION,
                "inventory_source_fingerprint": fingerprint,
            }
            streamer_source.set_access_observer.side_effect = TypeError(
                "invalid observer"
            )
            with (
                mock.patch.object(
                    runtime, "Streamer", return_value=streamer_source
                ),
                self.assertRaisesRegex(TypeError, "invalid observer"),
            ):
                runtime.open_deepseek_runtime_source(
                    source=REMOTE_REPO,
                    revision=PINNED_REVISION,
                    logical_repo_id=REMOTE_REPO,
                    causal_bundle=None,
                    budget_mb=64,
                    cache_dir=root / "cache",
                    use_cache=True,
                    max_cache_bytes=123,
                    access_observer=object(),
                    remote_pinned_inventory=pinned,
                )
            streamer_source.close.assert_called_once_with()

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

    def test_remote_expert_split_rail_owns_remote_and_local_planes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            _write_general_manifest(bundle)
            pinned = root / "remote-inventory.json"
            document = _remote_inventory_document()
            _write_remote_inventory(pinned, document)
            fingerprint = str(document["source_fingerprint"])
            model = LogicalModelIdentity(REMOTE_REPO, PINNED_REVISION)
            mount_source = mock.Mock()
            tensor_reader = object()
            mount = mock.Mock(
                source=mount_source,
                reader=object(),
                tensor_reader=tensor_reader,
                layout=CausalWeightLayoutIdentity(model, fingerprint),
            )
            remote_source = mock.Mock()
            remote_source.adopt_pinned_inventory.return_value = fingerprint
            remote_source.metrics.return_value = {
                "repo_id": REMOTE_REPO,
                "revision": PINNED_REVISION,
                "revision_is_pinned": True,
                "revision_is_mutable": False,
                "inventory_source_fingerprint": fingerprint,
            }

            with (
                mock.patch.object(
                    runtime, "CausalWeightMount", return_value=mount
                ) as mount_type,
                mock.patch.object(
                    runtime, "Streamer", return_value=remote_source
                ) as streamer_type,
            ):
                opened = runtime.open_deepseek_runtime_source(
                    source=REMOTE_REPO,
                    revision=PINNED_REVISION,
                    logical_repo_id=REMOTE_REPO,
                    causal_bundle=bundle,
                    budget_mb=64,
                    cache_dir=root / "remote-cache",
                    use_cache=True,
                    max_cache_bytes=123,
                    remote_pinned_inventory=pinned,
                    remote_expert_split_rail=True,
                )

            self.assertTrue(opened.remote_expert_split_rail)
            self.assertTrue(opened.is_causal_bundle)
            self.assertIs(opened.source, remote_source)
            self.assertIsNone(opened.causal_weight_reader)
            self.assertIs(opened.causal_tensor_reader, tensor_reader)
            self.assertEqual(opened.local_causal_layout_fingerprint, fingerprint)
            self.assertEqual(
                opened.label,
                f"remote-expert-split-rail:{REMOTE_REPO}@{PINNED_REVISION[:12]}",
            )
            mount_type.assert_called_once_with(
                bundle,
                model,
                budget_mb=64.0,
                verbose=False,
            )
            self.assertEqual(streamer_type.call_args.args, (REMOTE_REPO,))
            self.assertTrue(streamer_type.call_args.kwargs["use_cache"])
            self.assertEqual(streamer_type.call_args.kwargs["max_cache_bytes"], 123)
            remote_source.adopt_pinned_inventory.assert_called_once_with(
                document["inventory"], expected_fingerprint=fingerprint
            )

            opened.close()
            opened.close()
            remote_source.close.assert_called_once_with()
            mount.close.assert_called_once_with()
            mount_source.close.assert_not_called()

    def test_remote_expert_split_rail_rejects_invalid_contract_before_open(self) -> None:
        cases = (
            ({"causal_bundle": None}, "general causal bundle"),
            ({"remote_pinned_inventory": None}, "remote_pinned_inventory"),
            ({"access_observer": object()}, "access observers"),
            ({"source": "/tmp/local-checkpoint"}, "remote source"),
            ({"logical_repo_id": "foreign/model"}, "match exactly"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            _write_general_manifest(bundle)
            pinned = root / "remote-inventory.json"
            _write_remote_inventory(pinned, _remote_inventory_document())
            base: dict[str, object] = {
                "source": REMOTE_REPO,
                "revision": PINNED_REVISION,
                "logical_repo_id": REMOTE_REPO,
                "causal_bundle": bundle,
                "budget_mb": 64,
                "cache_dir": root / "cache",
                "use_cache": True,
                "max_cache_bytes": 123,
                "remote_pinned_inventory": pinned,
                "remote_expert_split_rail": True,
            }
            for overrides, message in cases:
                with self.subTest(message=message):
                    arguments = {**base, **overrides}
                    with (
                        mock.patch.object(runtime, "CausalWeightMount") as mount,
                        mock.patch.object(runtime, "Streamer") as streamer,
                        self.assertRaisesRegex(
                            runtime.DeepSeekRuntimeSourceError, message
                        ),
                    ):
                        runtime.open_deepseek_runtime_source(**arguments)
                    mount.assert_not_called()
                    streamer.assert_not_called()

    def test_remote_expert_split_rail_mismatch_closes_both_partial_owners(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            _write_general_manifest(bundle)
            pinned = root / "remote-inventory.json"
            document = _remote_inventory_document()
            _write_remote_inventory(pinned, document)
            fingerprint = str(document["source_fingerprint"])
            model = LogicalModelIdentity(REMOTE_REPO, PINNED_REVISION)
            mount = mock.Mock(
                layout=CausalWeightLayoutIdentity(model, fingerprint)
            )
            remote_source = mock.Mock()
            remote_source.adopt_pinned_inventory.return_value = fingerprint
            remote_source.metrics.return_value = {
                "repo_id": REMOTE_REPO,
                "revision": PINNED_REVISION,
                "revision_is_pinned": True,
                "revision_is_mutable": False,
                "inventory_source_fingerprint": "0" * 64,
            }
            with (
                mock.patch.object(
                    runtime, "CausalWeightMount", return_value=mount
                ),
                mock.patch.object(
                    runtime, "Streamer", return_value=remote_source
                ),
                self.assertRaisesRegex(
                    runtime.DeepSeekRuntimeSourceError, "exactly match"
                ),
            ):
                runtime.open_deepseek_runtime_source(
                    source=REMOTE_REPO,
                    revision=PINNED_REVISION,
                    logical_repo_id=REMOTE_REPO,
                    causal_bundle=bundle,
                    budget_mb=64,
                    cache_dir=root / "cache",
                    use_cache=True,
                    max_cache_bytes=123,
                    remote_pinned_inventory=pinned,
                    remote_expert_split_rail=True,
                )
            remote_source.close.assert_called_once_with()
            mount.close.assert_called_once_with()

    def test_remote_expert_split_close_attempts_both_owners_after_failure(self) -> None:
        remote_source = mock.Mock()
        mount = mock.Mock()
        remote_source.close.side_effect = RuntimeError("remote close failed")
        mount.close.side_effect = RuntimeError("mount close failed")
        opened = runtime.DeepSeekRuntimeSource(
            source=remote_source,
            label="split",
            mount=mount,
            remote_expert_split_rail=True,
        )

        with self.assertRaisesRegex(RuntimeError, "remote close failed"):
            opened.close()
        opened.close()
        self.assertTrue(opened.closed)
        remote_source.close.assert_called_once_with()
        mount.close.assert_called_once_with()

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
