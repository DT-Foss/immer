from __future__ import annotations

from contextlib import redirect_stderr
import hashlib
import importlib.util
import io
import json
from pathlib import Path
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

    def test_parser_requires_explicit_subcommand(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            bundle_script._parser().parse_args([])


if __name__ == "__main__":
    unittest.main()
