from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from immer.knowledge import AccessTraceRecorder, Streamer

from test_deepseek_v4_causal_weights import _LOGICAL_MODEL, _write_expert_fixture

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deepseek_v4_sparse_causal_bundle.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "deepseek_v4_sparse_causal_bundle", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import sparse causal bundle script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bundle_script = _load_script()


class SparseCausalBundleTests(unittest.TestCase):
    def test_header_reconstruction_preserves_original_data_start(self) -> None:
        shard = {"data_start": 128, "file": "model.safetensors", "st_metadata": None}
        tensors = [
            {
                "dtype": "I8",
                "name": "b",
                "offset_in_shard": [2, 4],
                "shape": [2],
            },
            {
                "dtype": "I8",
                "name": "a",
                "offset_in_shard": [0, 2],
                "shape": [2],
            },
        ]
        encoded = bundle_script._header_bytes(shard, tensors)
        self.assertEqual(len(encoded), 128)
        self.assertEqual(int.from_bytes(encoded[:8], "little"), 120)
        header = json.loads(encoded[8:].decode("utf-8"))
        self.assertEqual(list(header), ["a", "b"])

    def test_builds_and_mounts_trace_complete_sparse_fixture(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            base = Path(temporary)
            source_root = base / "source"
            cache = base / "cache"
            output = base / "model.causal"
            _write_expert_fixture(source_root, expert_ids=(0, 1))
            source_root.joinpath("config.json").write_text("{}", encoding="utf-8")
            recorder = AccessTraceRecorder()
            source = Streamer.from_local(
                source_root,
                repo_id=_LOGICAL_MODEL.repo_id,
                revision=_LOGICAL_MODEL.revision,
                cache_dir=cache,
                budget_mb=16,
                access_observer=recorder,
            )
            try:
                inventory = source.inventory()
                source.reader.fetch_file("config.json")
                for tensor in inventory["tensors"]:
                    begin, end = tensor["offset_in_shard"]
                    source.raw_bytes(
                        tensor["shard"],
                        int(tensor["data_start"]) + int(begin),
                        int(end) - int(begin),
                    )
                trace = recorder.snapshot()
            finally:
                source.close()
            trace_path = base / "trace.json"
            trace_path.write_bytes(trace.to_bytes())
            inventory_path = next((cache / "inventories").glob("*.json"))
            inventory_cache = json.loads(inventory_path.read_text(encoding="utf-8"))
            fingerprint = inventory_cache["source_fingerprint"]
            args = argparse.Namespace(
                access_trace=[str(trace_path)],
                budget_mb=16.0,
                cache_dir=str(cache),
                inventory=str(inventory_path),
                layout_fingerprint=fingerprint,
                output=str(output),
                repo_id=_LOGICAL_MODEL.repo_id,
                revision=_LOGICAL_MODEL.revision,
            )

            manifest = bundle_script.build_bundle(args)
            self.assertEqual(manifest["causal_bindings"], 2)
            self.assertEqual(manifest["layout_fingerprint"], fingerprint)
            self.assertEqual(manifest["unique_leaves"], 12)
            self.assertGreater(manifest["physical_shard_bytes"], 0)
            self.assertGreater(manifest["logical_shard_bytes"], 0)
            stored = json.loads(output.joinpath("bundle.json").read_text())
            identity = {key: value for key, value in stored.items() if key != "sha256"}
            self.assertEqual(
                stored["sha256"],
                hashlib.sha256(bundle_script._canonical(identity)).hexdigest(),
            )

            verified = bundle_script.verify_bundle(
                argparse.Namespace(
                    budget_mb=16.0,
                    bundle=str(output),
                    repo_id=_LOGICAL_MODEL.repo_id,
                    revision=_LOGICAL_MODEL.revision,
                )
            )
            self.assertEqual(verified["causal_bindings"], 2)


if __name__ == "__main__":
    unittest.main()
