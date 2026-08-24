from __future__ import annotations

from contextlib import redirect_stderr
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from immer.knowledge import Streamer
from immer.runtimes.qwen3_8 import (
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
    CausalWeightMount,
    LogicalModelIdentity,
    Qwen38WeightPager,
    StreamedQwen38,
)

from test_qwen3_8_model import (
    _tiny_config,
    _tiny_config_mapping,
    _tiny_weights,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen38_causal_bundle.py"
REPO_ID = OFFICIAL_REPO_ID
REVISION = OFFICIAL_REVISION


def _load_script():
    spec = importlib.util.spec_from_file_location("qwen38_causal_bundle", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import Qwen causal bundle script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bundle_script = _load_script()


class _Response:
    def __init__(self, data: bytes, *, status: int, headers: dict[str, str]) -> None:
        self.data = data
        self.status_code = status
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        pass

    def iter_content(self, chunk_size: int):
        for start in range(0, len(self.data), chunk_size):
            yield self.data[start : start + chunk_size]


class _Session:
    def __init__(self, files: dict[str, bytes], *, honor_range: bool = True) -> None:
        self.files = files
        self.honor_range = honor_range
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(self, url: str, *, headers: dict[str, str], **_kwargs):
        name = url.split("?", 1)[0].rsplit("/", 1)[-1]
        data = self.files[name]
        self.calls.append((name, dict(headers)))
        raw_range = headers.get("Range")
        if raw_range and self.honor_range:
            offset = int(raw_range.removeprefix("bytes=").removesuffix("-"))
            return _Response(
                data[offset:],
                status=206,
                headers={
                    "Content-Range": f"bytes {offset}-{len(data) - 1}/{len(data)}"
                },
            )
        return _Response(data, status=200, headers={})

    def close(self) -> None:
        pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _fixture(root: Path) -> tuple[Path, Path, str]:
    source_root = root / "source"
    source_root.mkdir()
    save_file(_tiny_weights(_tiny_config()), source_root / "model.safetensors")
    source_root.joinpath("config.json").write_text(
        json.dumps(_tiny_config_mapping(), separators=(",", ":")),
        encoding="utf-8",
    )
    source = Streamer.from_local(
        source_root,
        repo_id=REPO_ID,
        revision=REVISION,
        use_cache=False,
        budget_mb=20,
    )
    try:
        inventory = json.loads(json.dumps(source.inventory()))
    finally:
        source.close()
    digest = _sha256(source_root / "model.safetensors")
    inventory["shards"][0]["etag"] = f'"{digest}"'
    inventory["shards"][0]["cas_url_hash"] = digest
    inventory["shards"][0]["linked_etag"] = digest
    inventory["shards"][0]["payload_sha256"] = digest
    fingerprint = Streamer._source_fingerprint(inventory)
    document = {
        "inventory": inventory,
        "inventory_sha256": hashlib.sha256(
            bundle_script._canonical(inventory)
        ).hexdigest(),
        "repo_id": REPO_ID,
        "revision": REVISION,
        "schema": "immer.tensor-inventory-cache/v1",
        "source_fingerprint": fingerprint,
    }
    inventory_path = root / "inventory.json"
    inventory_path.write_bytes(bundle_script._canonical(document) + b"\n")
    return source_root, inventory_path, fingerprint


class QwenCausalBundleTests(unittest.TestCase):
    def test_refresh_inventory_separates_payload_and_xet_identity(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".qwen-inventory-refresh-test-", dir=Path.cwd()
        ) as temporary:
            root = Path(temporary)
            _source, inventory_path, fingerprint = _fixture(root)
            inventory = json.loads(inventory_path.read_text())["inventory"]
            shard = inventory["shards"][0]
            shard["repo_commit"] = REVISION
            shard["xet_hash"] = "f" * 64
            closed = []

            class FakeStreamer:
                def __init__(self, *_args, **_kwargs) -> None:
                    pass

                def inventory(self):
                    return json.loads(json.dumps(inventory))

                def close(self) -> None:
                    closed.append(True)

                _source_fingerprint = staticmethod(Streamer._source_fingerprint)

            output = root / "refreshed.json"
            with mock.patch.object(bundle_script, "Streamer", FakeStreamer):
                result = bundle_script.refresh_inventory(
                    output,
                    repo_id=REPO_ID,
                    revision=REVISION,
                )

            document = json.loads(output.read_text())
            self.assertEqual(closed, [True])
            self.assertEqual(result["shards"], 1)
            self.assertEqual(result["tensors"], 56)
            self.assertNotEqual(result["source_fingerprint"], fingerprint)
            self.assertEqual(
                document["inventory"]["shards"][0]["payload_sha256"],
                shard["payload_sha256"],
            )
            self.assertEqual(document["inventory"]["shards"][0]["xet_hash"], "f" * 64)

    def test_builder_refuses_xet_identity_as_payload_digest(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".qwen-xet-not-payload-test-", dir=Path.cwd()
        ) as temporary:
            root = Path(temporary)
            source, inventory_path, _fingerprint = _fixture(root)
            document = json.loads(inventory_path.read_text())
            shard = document["inventory"]["shards"][0]
            shard.pop("payload_sha256")
            shard.pop("linked_etag")
            document["inventory_sha256"] = hashlib.sha256(
                bundle_script._canonical(document["inventory"])
            ).hexdigest()
            document["source_fingerprint"] = Streamer._source_fingerprint(
                document["inventory"]
            )
            inventory_path.write_bytes(bundle_script._canonical(document) + b"\n")

            with self.assertRaisesRegex(
                bundle_script.QwenCausalBundleError,
                "lacks payload SHA-256",
            ):
                bundle_script.build_bundle(
                    source,
                    inventory_path,
                    root / "model.causal",
                    repo_id=REPO_ID,
                    revision=REVISION,
                    expected_fingerprint=document["source_fingerprint"],
                    require_official=False,
                    require_remote_hashes=True,
                )

    def test_build_verify_mount_and_execute_complete_fixture(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".qwen-bundle-test-", dir=Path.cwd()
        ) as temporary:
            root = Path(temporary)
            source, inventory, fingerprint = _fixture(root)
            output = root / "model.causal"
            result = bundle_script.build_bundle(
                source,
                inventory,
                output,
                repo_id=REPO_ID,
                revision=REVISION,
                expected_fingerprint=fingerprint,
                require_official=False,
                require_remote_hashes=True,
            )
            self.assertEqual(result["layout_fingerprint"], fingerprint)
            self.assertEqual(result["tensor_bindings"], 56)
            self.assertFalse(result["resumed"])
            self.assertFalse(output.with_name(".model.causal.building").exists())

            verified = bundle_script.verify_bundle(
                output,
                expected_repo_id=REPO_ID,
                expected_revision=REVISION,
                expected_fingerprint=fingerprint,
                require_official=False,
            )
            self.assertEqual(verified["tensor_bindings"], 56)
            manifest = json.loads(output.joinpath("bundle.json").read_text())
            self.assertTrue(manifest["body"]["checkpoint_complete"])

            identity = LogicalModelIdentity(REPO_ID, REVISION)
            with CausalWeightMount(output, identity, budget_mb=20) as mount:
                pager = Qwen38WeightPager(
                    mount.source,
                    device="cpu",
                    compute_dtype="bfloat16",
                    max_resident_bytes=2 * 1024**2,
                    causal_tensor_reader=mount.tensor_reader,
                )
                model = StreamedQwen38(
                    _tiny_config(), pager, max_batch_size=1, max_seq_len=16
                )
                try:
                    hidden, evidence = model.prefill([[1, 4, 9]])
                    self.assertEqual(tuple(hidden.shape), (1, 3, 12))
                    self.assertTrue(torch.isfinite(hidden).all())
                    self.assertEqual(evidence[0].end_pos, 3)
                finally:
                    pager.close()

            shard = output / "weights" / "model.safetensors"
            with shard.open("r+b") as handle:
                handle.seek(shard.stat().st_size - 1)
                final = handle.read(1)
                handle.seek(-1, 1)
                handle.write(bytes([final[0] ^ 0xFF]))
            with self.assertRaisesRegex(
                bundle_script.QwenCausalBundleError, "shard verification"
            ):
                bundle_script.verify_bundle(
                    output,
                    expected_repo_id=REPO_ID,
                    expected_revision=REVISION,
                    expected_fingerprint=fingerprint,
                    require_official=False,
                )

    def test_resume_reuses_only_a_complete_verified_staged_shard(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".qwen-bundle-resume-test-", dir=Path.cwd()
        ) as temporary:
            root = Path(temporary)
            source, inventory, fingerprint = _fixture(root)
            output = root / "model.causal"
            staging = root / ".model.causal.building"
            (staging / "weights").mkdir(parents=True)
            (staging / "causal").mkdir()
            staged_shard = staging / "weights" / "model.safetensors"
            staged_shard.write_bytes(source.joinpath("model.safetensors").read_bytes())

            result = bundle_script.build_bundle(
                source,
                inventory,
                output,
                repo_id=REPO_ID,
                revision=REVISION,
                expected_fingerprint=fingerprint,
                require_official=False,
                require_remote_hashes=True,
                resume=True,
            )
            self.assertTrue(result["resumed"])
            manifest = json.loads(output.joinpath("bundle.json").read_text())
            self.assertTrue(manifest["body"]["shards"][0]["reused"])

    def test_verify_rejects_resealed_but_non_qwen_config(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".qwen-bundle-config-test-", dir=Path.cwd()
        ) as temporary:
            root = Path(temporary)
            source, inventory, fingerprint = _fixture(root)
            output = root / "model.causal"
            bundle_script.build_bundle(
                source,
                inventory,
                output,
                repo_id=REPO_ID,
                revision=REVISION,
                expected_fingerprint=fingerprint,
                require_official=False,
            )
            invalid = b'{"not":"qwen"}'
            output.joinpath("weights", "config.json").write_bytes(invalid)
            manifest_path = output / "bundle.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["body"]["config_sha256"] = hashlib.sha256(invalid).hexdigest()
            manifest["sha256"] = hashlib.sha256(
                bundle_script._canonical(manifest["body"])
            ).hexdigest()
            manifest_path.write_bytes(bundle_script._canonical(manifest) + b"\n")

            with self.assertRaisesRegex(
                bundle_script.QwenCausalBundleError, "config is not executable"
            ):
                bundle_script.verify_bundle(
                    output,
                    expected_repo_id=REPO_ID,
                    expected_revision=REVISION,
                    expected_fingerprint=fingerprint,
                    require_official=False,
                )

    def test_atomic_promotion_never_replaces_late_output(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".qwen-bundle-promotion-test-", dir=Path.cwd()
        ) as temporary:
            root = Path(temporary)
            source, inventory, fingerprint = _fixture(root)
            output = root / "model.causal"
            original_verify = bundle_script.verify_bundle

            def occupy_target(*args, **kwargs):
                result = original_verify(*args, **kwargs)
                output.mkdir()
                return result

            with (
                mock.patch.object(
                    bundle_script,
                    "verify_bundle",
                    side_effect=occupy_target,
                ),
                self.assertRaisesRegex(
                    bundle_script.QwenCausalBundleError,
                    "promotion refused",
                ),
            ):
                bundle_script.build_bundle(
                    source,
                    inventory,
                    output,
                    repo_id=REPO_ID,
                    revision=REVISION,
                    expected_fingerprint=fingerprint,
                    require_official=False,
                )
            self.assertTrue(output.is_dir())
            self.assertEqual(tuple(output.iterdir()), ())
            self.assertTrue(root.joinpath(".model.causal.building").is_dir())

    def test_adopt_verifies_and_causalizes_weights_without_copying_them(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".qwen-bundle-adopt-test-", dir=Path.cwd()
        ) as temporary:
            root = Path(temporary)
            source, inventory, fingerprint = _fixture(root)
            bundle = root / "adopted.causal"
            weights = bundle / "weights"
            shutil.copytree(source, weights)
            shard = weights / "model.safetensors"
            before = shard.stat()
            before_bytes = shard.read_bytes()

            result = bundle_script.adopt_bundle(
                bundle,
                inventory,
                repo_id=REPO_ID,
                revision=REVISION,
                expected_fingerprint=fingerprint,
                require_official=False,
            )
            self.assertTrue(result["adopted"])
            self.assertFalse(result["resumed"])
            self.assertTrue(bundle.joinpath("causal").is_dir())
            self.assertTrue(bundle.joinpath("bundle.json").is_file())
            after = shard.stat()
            self.assertEqual(after.st_ino, before.st_ino)
            self.assertEqual(after.st_mtime_ns, before.st_mtime_ns)
            self.assertEqual(shard.read_bytes(), before_bytes)

            replay = bundle_script.adopt_bundle(
                bundle,
                inventory,
                repo_id=REPO_ID,
                revision=REVISION,
                expected_fingerprint=fingerprint,
                require_official=False,
            )
            self.assertTrue(replay["resumed"])

    def test_flat_adopt_implants_graph_without_moving_or_aliasing_weights(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".qwen-bundle-flat-adopt-test-", dir=Path.cwd()
        ) as temporary:
            root = Path(temporary)
            checkpoint, inventory, fingerprint = _fixture(root)
            shard = checkpoint / "model.safetensors"
            config = checkpoint / "config.json"
            shard_before = shard.stat()
            config_before = config.stat()
            shard_bytes = shard.read_bytes()

            result = bundle_script.adopt_bundle(
                checkpoint,
                inventory,
                repo_id=REPO_ID,
                revision=REVISION,
                expected_fingerprint=fingerprint,
                require_official=False,
                weights_layout="flat",
            )

            self.assertTrue(result["adopted"])
            self.assertEqual(result["weights_layout"], "flat/v1")
            self.assertFalse(checkpoint.joinpath("weights").exists())
            self.assertTrue(checkpoint.joinpath("causal").is_dir())
            self.assertTrue(checkpoint.joinpath("bundle.json").is_file())
            self.assertTrue(checkpoint.joinpath("inventory.pinned.json").is_file())
            self.assertEqual(shard.stat().st_ino, shard_before.st_ino)
            self.assertEqual(shard.stat().st_mtime_ns, shard_before.st_mtime_ns)
            self.assertEqual(config.stat().st_ino, config_before.st_ino)
            self.assertEqual(config.stat().st_mtime_ns, config_before.st_mtime_ns)
            self.assertEqual(shard.read_bytes(), shard_bytes)

            identity = LogicalModelIdentity(REPO_ID, REVISION)
            with CausalWeightMount(checkpoint, identity, budget_mb=20) as mount:
                self.assertEqual(mount.weights_layout, "flat")
                self.assertEqual(mount.weights_root, checkpoint.absolute())
                pager = Qwen38WeightPager(
                    mount.source,
                    device="cpu",
                    compute_dtype="bfloat16",
                    max_resident_bytes=2 * 1024**2,
                    causal_tensor_reader=mount.tensor_reader,
                )
                try:
                    model = StreamedQwen38(
                        _tiny_config(), pager, max_batch_size=1, max_seq_len=16
                    )
                    hidden, _evidence = model.prefill([[1, 4, 9]])
                    self.assertTrue(torch.isfinite(hidden).all())
                finally:
                    pager.close()

            replay = bundle_script.adopt_bundle(
                checkpoint,
                inventory,
                repo_id=REPO_ID,
                revision=REVISION,
                expected_fingerprint=fingerprint,
                require_official=False,
                weights_layout="flat",
            )
            self.assertTrue(replay["resumed"])

    def test_flat_adopt_rejects_bad_weights_before_publishing_sidecars(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".qwen-bundle-flat-invalid-test-", dir=Path.cwd()
        ) as temporary:
            root = Path(temporary)
            checkpoint, inventory, fingerprint = _fixture(root)
            shard = checkpoint / "model.safetensors"
            with shard.open("r+b") as handle:
                handle.seek(-1, 2)
                value = handle.read(1)
                handle.seek(-1, 2)
                handle.write(bytes([value[0] ^ 0xFF]))

            with self.assertRaisesRegex(
                bundle_script.QwenCausalBundleError,
                "shard verification failed",
            ):
                bundle_script.adopt_bundle(
                    checkpoint,
                    inventory,
                    repo_id=REPO_ID,
                    revision=REVISION,
                    expected_fingerprint=fingerprint,
                    require_official=False,
                    weights_layout="flat",
                )
            self.assertFalse(checkpoint.joinpath("causal").exists())
            self.assertFalse(checkpoint.joinpath("bundle.json").exists())
            self.assertFalse(checkpoint.joinpath("inventory.pinned.json").exists())

    def test_fetch_resumes_directly_into_weights_and_adopts_once(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".qwen-bundle-fetch-test-", dir=Path.cwd()
        ) as temporary:
            root = Path(temporary)
            source, inventory, fingerprint = _fixture(root)
            shard_bytes = source.joinpath("model.safetensors").read_bytes()
            config_bytes = source.joinpath("config.json").read_bytes()
            bundle = root / "fetched.causal"
            weights = bundle / "weights"
            weights.mkdir(parents=True)
            partial = weights / ".model.safetensors.partial"
            split = len(shard_bytes) // 2
            partial.write_bytes(shard_bytes[:split])
            session = _Session(
                {
                    "config.json": config_bytes,
                    "model.safetensors": shard_bytes,
                }
            )

            result = bundle_script.fetch_adopt_bundle(
                bundle,
                inventory,
                repo_id=REPO_ID,
                revision=REVISION,
                expected_fingerprint=fingerprint,
                require_official=False,
                session=session,
            )
            self.assertTrue(result["adopted"])
            self.assertEqual(
                weights.joinpath("model.safetensors").read_bytes(), shard_bytes
            )
            self.assertFalse(partial.exists())
            range_calls = [
                headers["Range"]
                for name, headers in session.calls
                if name == "model.safetensors"
            ]
            self.assertEqual(range_calls, [f"bytes={split}-"])
            self.assertTrue(weights.joinpath("download.json").is_file())
            self.assertTrue(bundle.joinpath("bundle.json").is_file())

            replay_session = _Session({})
            replay = bundle_script.fetch_adopt_bundle(
                bundle,
                inventory,
                repo_id=REPO_ID,
                revision=REVISION,
                expected_fingerprint=fingerprint,
                require_official=False,
                session=replay_session,
            )
            self.assertTrue(replay["resumed"])
            self.assertEqual(replay_session.calls, [])

    def test_fetch_refuses_server_that_ignores_resume_range(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".qwen-bundle-range-test-", dir=Path.cwd()
        ) as temporary:
            root = Path(temporary)
            payload = b"0123456789"
            target = root / "model.safetensors"
            partial = root / ".model.safetensors.partial"
            partial.write_bytes(payload[:4])
            session = _Session(
                {"model.safetensors": payload},
                honor_range=False,
            )
            with self.assertRaisesRegex(
                bundle_script.QwenCausalBundleError,
                "refused exact resume range",
            ):
                bundle_script._download_verified_file(
                    session,
                    "https://example.invalid/model.safetensors",
                    target,
                    expected_size=len(payload),
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                    resume=True,
                )
            self.assertEqual(partial.read_bytes(), payload[:4])
            self.assertFalse(target.exists())

    def test_fetch_promotes_complete_verified_partial_without_http(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".qwen-bundle-complete-partial-test-", dir=Path.cwd()
        ) as temporary:
            root = Path(temporary)
            payload = b"complete pinned shard"
            target = root / "model.safetensors"
            partial = root / ".model.safetensors.partial"
            partial.write_bytes(payload)
            session = _Session({})

            receipt = bundle_script._download_verified_file(
                session,
                "https://example.invalid/model.safetensors",
                target,
                expected_size=len(payload),
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                resume=True,
            )

            self.assertEqual(target.read_bytes(), payload)
            self.assertFalse(partial.exists())
            self.assertEqual(session.calls, [])
            self.assertEqual(receipt["resumed_from"], len(payload))

    def test_fetch_recovers_corrupt_partial_by_restarting_once(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".qwen-bundle-corrupt-partial-test-", dir=Path.cwd()
        ) as temporary:
            root = Path(temporary)
            payload = b"0123456789"
            target = root / "model.safetensors"
            partial = root / ".model.safetensors.partial"
            partial.write_bytes(b"xxxx")
            session = _Session({"model.safetensors": payload})

            receipt = bundle_script._download_verified_file(
                session,
                "https://example.invalid/model.safetensors",
                target,
                expected_size=len(payload),
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                resume=True,
            )

            self.assertEqual(target.read_bytes(), payload)
            self.assertFalse(partial.exists())
            self.assertEqual(
                session.calls,
                [
                    (
                        "model.safetensors",
                        {
                            "Range": "bytes=4-",
                            "User-Agent": "IMMER-Qwen-Causal-Bundle/1",
                        },
                    ),
                    ("model.safetensors", {"User-Agent": "IMMER-Qwen-Causal-Bundle/1"}),
                ],
            )
            self.assertEqual(receipt["resumed_from"], 0)

    def test_fetch_refuses_shifted_initial_partial_response(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".qwen-bundle-shifted-range-test-", dir=Path.cwd()
        ) as temporary:
            root = Path(temporary)
            payload = b"0123456789"
            target = root / "model.safetensors"
            session = mock.Mock()
            session.get.return_value = _Response(
                payload[1:],
                status=206,
                headers={"Content-Range": "bytes 1-9/10"},
            )

            with self.assertRaisesRegex(
                bundle_script.QwenCausalBundleError,
                "shifted initial range",
            ):
                bundle_script._download_verified_file(
                    session,
                    "https://example.invalid/model.safetensors",
                    target,
                    expected_size=len(payload),
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                    resume=True,
                )
            self.assertFalse(target.exists())
            self.assertFalse(root.joinpath(".model.safetensors.partial").exists())

    def test_parser_requires_explicit_subcommand(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            bundle_script._parser().parse_args([])


if __name__ == "__main__":
    unittest.main()
