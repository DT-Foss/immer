from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from scripts.qwen38_layer_mlp_o1_publish_all import (
    build_parser as build_publish_parser,
    run as run_publish,
)
from immer.runtimes.qwen3_8.layer_mlp_crystal import (
    Layer63MlpResidualCrystalBank,
    LayerMlpResidualCrystalBank,
    Layer63MlpResidualCrystalIdentity,
    LayerMlpResidualCrystalIdentity,
)
from immer.runtimes.qwen3_8.layer_mlp_o1 import (
    Layer63MlpO1Accumulator,
    LayerMlpO1Accumulator,
)
from immer.runtimes.qwen3_8.layer_mlp_registry import (
    DEFAULT_LAYER_MLP_O1_REGISTRY_BASENAME,
    LAYER_MLP_O1_REGISTRY_ENVELOPE_SCHEMA,
    LayerMlpO1RegistryDescriptor,
    LayerMlpO1RegistryIntegrityError,
    LayerMlpO1RegistryNotMountableError,
    layer_mlp_o1_state_path_for_layer,
    load_layer_mlp_o1_registry,
    publish_layer_mlp_o1_registry,
    read_layer_mlp_o1_registry_manifest,
)
from immer.runtimes.qwen3_8 import layer_mlp_registry as registry_module
from immer.runtimes.qwen3_8.layer_transition_crystal import (
    LayerTransitionProjectionIdentity,
    _sealed_document,
)


def _projection() -> LayerTransitionProjectionIdentity:
    return LayerTransitionProjectionIdentity(
        hidden_dim=8,
        sketch_dim=3,
        seed_sha256="5" * 64,
    )


def _legacy_identity(
    *,
    model: str = "1",
) -> Layer63MlpResidualCrystalIdentity:
    return Layer63MlpResidualCrystalIdentity(
        model_sha256=model * 64,
        q4_sha256="2" * 64,
        graph_revision_sha256="3" * 64,
        atlas_revision_sha256="4" * 64,
        projection=_projection(),
    )


def _generic_identity(
    layer_index: int,
    *,
    model: str = "1",
) -> LayerMlpResidualCrystalIdentity:
    return LayerMlpResidualCrystalIdentity(
        model_sha256=model * 64,
        q4_sha256="2" * 64,
        graph_revision_sha256="3" * 64,
        atlas_revision_sha256="4" * 64,
        projection=_projection(),
        layer_index=layer_index,
    )


def _triples(
    rows: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(rows, 8, generator=generator).to(torch.bfloat16)
    feature = torch.randn(rows, 8, generator=generator).to(torch.bfloat16)
    mixing = torch.randn(8, 8, generator=generator) * 0.04
    target = (base.float() + feature.float() @ mixing + 0.125).to(torch.bfloat16)
    return base, feature, target


def _reseal_manifest(path: Path, mutate) -> None:
    document = json.loads(path.read_bytes())
    body = document["body"]
    mutate(body)
    path.write_bytes(
        _sealed_document(body, LAYER_MLP_O1_REGISTRY_ENVELOPE_SCHEMA)
    )


class LayerMlpO1RegistryTests(unittest.TestCase):
    def _mixed_sources(self, root: Path) -> Path:
        legacy_path = root / "layer63.json"
        legacy = Layer63MlpO1Accumulator(
            legacy_path,
            _legacy_identity(),
            packed_weight_bytes_avoided=909,
        )
        generic = LayerMlpO1Accumulator(
            layer_mlp_o1_state_path_for_layer(
                legacy_path,
                layer_index=0,
                sketch_dim=3,
            ),
            _generic_identity(0),
            packed_weight_bytes_avoided=808,
        )
        pending = LayerMlpO1Accumulator(
            layer_mlp_o1_state_path_for_layer(
                legacy_path,
                layer_index=1,
                sketch_dim=3,
            ),
            _generic_identity(1),
            packed_weight_bytes_avoided=707,
        )
        legacy.observe(*_triples(10, seed=11))
        generic.observe(*_triples(10, seed=13))
        pending.observe(*_triples(2, seed=17))
        self.assertIsNone(pending.current_crystal())
        return legacy_path

    def _published(self, root: Path):
        legacy_path = self._mixed_sources(root)
        output = root / "published"
        summary = publish_layer_mlp_o1_registry(
            legacy_path,
            output,
            layers=(0, 1, 2, 63),
        )
        registry = load_layer_mlp_o1_registry(summary.manifest_path)
        return legacy_path, output, summary, registry

    def test_mixed_legacy_generic_ready_and_pending_states(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _legacy, output, summary, registry = self._published(root)
            encoded = summary.manifest_path.read_text(encoding="ascii")

        self.assertEqual(registry.configured_layers, (0, 1, 2, 63))
        self.assertEqual(registry.mounted_layers, (0, 63))
        self.assertEqual(registry.pending_layers, (1, 2))
        self.assertTrue(registry.mountable)
        self.assertIsInstance(registry.bank_for_layer(0), LayerMlpResidualCrystalBank)
        self.assertIsInstance(
            registry.bank_for_layer(63),
            Layer63MlpResidualCrystalBank,
        )
        self.assertNotIsInstance(
            registry.bank_for_layer(63),
            LayerMlpResidualCrystalBank,
        )
        self.assertEqual(
            [mount.entry.max_error_radius for mount in registry.mounts],
            [0.0, 0.0],
        )
        self.assertNotIn(str(output), encoded)
        self.assertNotIn("base_hidden", encoded)
        self.assertNotIn("mlp_input", encoded)
        self.assertNotIn("target_hidden", encoded)
        self.assertTrue(all("/" not in entry.bank_file for entry in registry.entries))

    def test_rerun_is_byte_idempotent_and_verifies_existing_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy, output, first, _registry = self._published(root)
            first_manifest = first.manifest_path.read_bytes()
            first_inode = first.manifest_path.stat().st_ino
            first_banks = {
                path.name: path.read_bytes()
                for path in output.glob("qwen-layer*-mlp-o1-bank-*.json")
            }
            second = publish_layer_mlp_o1_registry(
                legacy,
                output,
                layers=(0, 1, 2, 63),
            )
            second_banks = {
                path.name: path.read_bytes()
                for path in output.glob("qwen-layer*-mlp-o1-bank-*.json")
            }
            second_manifest = second.manifest_path.read_bytes()
            second_inode = second.manifest_path.stat().st_ino
            first_record = first.to_record()
            second_record = second.to_record()

        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(first_inode, second_inode)
        self.assertEqual(first_banks, second_banks)
        self.assertEqual(first_record, second_record)

    def test_empty_pending_manifest_creates_no_bank_and_is_not_mountable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy_path = root / "layer63.json"
            accumulator = Layer63MlpO1Accumulator(
                legacy_path,
                _legacy_identity(),
                packed_weight_bytes_avoided=909,
            )
            accumulator.observe(*_triples(2, seed=19))
            output = root / "published"
            summary = publish_layer_mlp_o1_registry(
                legacy_path,
                output,
                layers=(63,),
            )
            registry = load_layer_mlp_o1_registry(summary.manifest_path)

            self.assertFalse(registry.mountable)
            self.assertEqual(registry.entries, ())
            self.assertEqual(registry.pending_layers, (63,))
            self.assertEqual(
                list(output.glob("qwen-layer*-mlp-o1-bank-*.json")),
                [],
            )
            with self.assertRaises(LayerMlpO1RegistryNotMountableError):
                registry.require_mountable()
            with self.assertRaises(LayerMlpO1RegistryNotMountableError):
                load_layer_mlp_o1_registry(
                    summary.manifest_path,
                    require_mountable=True,
                )
            descriptor = read_layer_mlp_o1_registry_manifest(summary.manifest_path)
            self.assertFalse(descriptor.mountable)
            with self.assertRaises(LayerMlpO1RegistryNotMountableError):
                read_layer_mlp_o1_registry_manifest(
                    summary.manifest_path,
                    require_mountable=True,
                )

    def test_metadata_descriptor_performs_zero_bank_reads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _legacy, _output, summary, registry = self._published(root)
            original_read = registry_module._read_regular
            with (
                mock.patch.object(
                    Layer63MlpResidualCrystalBank,
                    "load",
                    side_effect=AssertionError("legacy bank read"),
                ) as legacy_load,
                mock.patch.object(
                    LayerMlpResidualCrystalBank,
                    "load",
                    side_effect=AssertionError("generic bank read"),
                ) as generic_load,
                mock.patch.object(
                    registry_module,
                    "_read_regular",
                    wraps=original_read,
                ) as regular_read,
            ):
                descriptor = read_layer_mlp_o1_registry_manifest(
                    summary.manifest_path,
                    require_mountable=True,
                )

        self.assertIsInstance(descriptor, LayerMlpO1RegistryDescriptor)
        self.assertEqual(descriptor.manifest_file_sha, summary.manifest_file_sha256)
        self.assertEqual(descriptor.registry_sha, summary.registry_sha256)
        self.assertEqual(descriptor.entries, registry.entries)
        self.assertEqual(descriptor.mounted_layers, (0, 63))
        self.assertTrue(descriptor.mountable)
        legacy_load.assert_not_called()
        generic_load.assert_not_called()
        self.assertEqual(regular_read.call_count, 1)
        self.assertEqual(
            Path(regular_read.call_args.args[0]),
            summary.manifest_path,
        )

    def test_manifest_and_bank_tamper_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _legacy, _output, summary, registry = self._published(root)
            raw = bytearray(summary.manifest_path.read_bytes())
            raw[len(raw) // 2] ^= 1
            summary.manifest_path.write_bytes(raw)
            with self.assertRaises(LayerMlpO1RegistryIntegrityError):
                read_layer_mlp_o1_registry_manifest(summary.manifest_path)
            with self.assertRaises(LayerMlpO1RegistryIntegrityError):
                load_layer_mlp_o1_registry(summary.manifest_path)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy, output, summary, registry = self._published(root)
            bank_path = output / registry.entries[0].bank_file
            raw = bytearray(bank_path.read_bytes())
            raw[len(raw) // 2] ^= 1
            bank_path.write_bytes(raw)
            with self.assertRaises(LayerMlpO1RegistryIntegrityError):
                load_layer_mlp_o1_registry(summary.manifest_path)
            with self.assertRaises(LayerMlpO1RegistryIntegrityError):
                publish_layer_mlp_o1_registry(
                    legacy,
                    output,
                    layers=(0, 1, 2, 63),
                )

    def test_symlink_and_path_traversal_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _legacy, output, summary, registry = self._published(root)
            bank_path = output / registry.entries[0].bank_file
            copy = root / "copied-bank.json"
            copy.write_bytes(bank_path.read_bytes())
            bank_path.unlink()
            bank_path.symlink_to(copy)
            with self.assertRaises(LayerMlpO1RegistryIntegrityError):
                load_layer_mlp_o1_registry(summary.manifest_path)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _legacy, _output, summary, _registry = self._published(root)
            _reseal_manifest(
                summary.manifest_path,
                lambda body: body["entries"][0].__setitem__(
                    "bank_file",
                    "../escaped.json",
                ),
            )
            with self.assertRaises(LayerMlpO1RegistryIntegrityError):
                load_layer_mlp_o1_registry(summary.manifest_path)

    def test_foreign_pin_and_duplicate_layer_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _legacy, _output, summary, _registry = self._published(root)
            _reseal_manifest(
                summary.manifest_path,
                lambda body: body["pins"].__setitem__(
                    "model_sha256",
                    "f" * 64,
                ),
            )
            with self.assertRaises(LayerMlpO1RegistryIntegrityError):
                load_layer_mlp_o1_registry(summary.manifest_path)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _legacy, _output, summary, _registry = self._published(root)
            _reseal_manifest(
                summary.manifest_path,
                lambda body: body["entries"].append(body["entries"][0].copy()),
            )
            with self.assertRaises(LayerMlpO1RegistryIntegrityError):
                load_layer_mlp_o1_registry(summary.manifest_path)

    def test_zero_radius_cli_summary_and_manifest_preserve_float_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = self._mixed_sources(root)
            output = root / "published"
            args = build_publish_parser().parse_args(
                [
                    "--layer63-state",
                    str(legacy),
                    "--output-root",
                    str(output),
                    "--layers",
                    "0,63",
                    "--max-error-radius",
                    "0",
                ]
            )
            result = run_publish(args)
            registry = load_layer_mlp_o1_registry(
                output / DEFAULT_LAYER_MLP_O1_REGISTRY_BASENAME
            )

        self.assertEqual(result["max_error_radius"], 0.0)
        self.assertEqual(result["published_layers"], [0, 63])
        self.assertEqual(
            tuple(entry.max_error_radius for entry in registry.entries),
            (0.0, 0.0),
        )


if __name__ == "__main__":
    unittest.main()
