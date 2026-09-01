from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import torch

from immer.runtimes.qwen3_8.kernels import AttentionState
from immer.runtimes.qwen3_8.mtp_carry_snapshot import (
    MtpCarrySidecarDescriptor,
    Qwen35MtpCarrySidecarError,
    Qwen35MtpCarrySidecarIdentityMismatch,
    qwen35_mtp_carry_identity_sha256,
    qwen35_mtp_prefix_sha256,
    read_qwen35_mtp_carry_sidecar,
    write_qwen35_mtp_carry_sidecar,
)
from immer.runtimes.qwen3_8.mtp_draft import (
    QWEN35_MTP_CARRY_SCHEMA,
    Qwen35MtpCarry,
)
from immer.runtimes.qwen3_8.semantic_state_cache import token_prefix_sha256


_TOKENIZER_SHA256 = hashlib.sha256(b"tokenizer.json").hexdigest()
_IDENTITY = (
    "immer.qwen3.5-mtp-draft-provider/v6",
    {
        "model": {"repo": "fixture/model", "revision": "pinned"},
        "model_sha256": hashlib.sha256(b"model").hexdigest(),
        "q4": {"manifest_sha256": hashlib.sha256(b"q4").hexdigest()},
    },
    (64, 2, 1, 8),
    "cpu",
    "torch.bfloat16",
)


def _carry(history: tuple[int, ...]) -> Qwen35MtpCarry:
    hidden = torch.arange(64, dtype=torch.float32).reshape(1, 1, 64).to(
        torch.bfloat16
    )
    state = None
    if len(history) > 1:
        elements = 2 * (len(history) - 1) * 8
        key = torch.arange(elements, dtype=torch.float32).reshape(
            1, 2, len(history) - 1, 8
        )
        key = (key / 7).to(torch.bfloat16)
        value = (key + torch.tensor(0.5, dtype=torch.bfloat16)).contiguous()
        state = AttentionState(key=key, value=value)
    return Qwen35MtpCarry(
        schema=QWEN35_MTP_CARRY_SCHEMA,
        identity=_IDENTITY,
        history=history,
        next_position=len(history) - 1,
        state=state,
        last_target_hidden=hidden,
    )


class Qwen35MtpCarrySnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix=".mtp-carry-test-")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(
        self, history: tuple[int, ...]
    ) -> tuple[Qwen35MtpCarry, MtpCarrySidecarDescriptor]:
        carry = _carry(history)
        descriptor = write_qwen35_mtp_carry_sidecar(
            self.root,
            carry,
            tokenizer_sha256=_TOKENIZER_SHA256,
            prefix_sha256=qwen35_mtp_prefix_sha256(history),
        )
        return carry, descriptor

    def _read(
        self,
        descriptor: MtpCarrySidecarDescriptor,
        history: tuple[int, ...],
        *,
        tokenizer_sha256: str = _TOKENIZER_SHA256,
        identity: object = _IDENTITY,
    ) -> Qwen35MtpCarry:
        return read_qwen35_mtp_carry_sidecar(
            self.root,
            descriptor,
            history=history,
            tokenizer_sha256=tokenizer_sha256,
            expected_identity=identity,
        )

    def _publish_modified(
        self,
        descriptor: MtpCarrySidecarDescriptor,
        data: bytes,
    ) -> MtpCarrySidecarDescriptor:
        digest = hashlib.sha256(data).hexdigest()
        basename = f"{digest}.qwen35-mtp-carry"
        target = self.root / basename
        target.write_bytes(data)
        target.chmod(0o600)
        return replace(
            descriptor,
            basename=basename,
            bytes=len(data),
            file_sha256=digest,
        )

    def test_bfloat16_multihead_roundtrip_is_bit_exact_and_path_free(self) -> None:
        history = (1_000_003, 2_000_033, 3_000_077, 4_000_037)
        original, descriptor = self._write(history)
        restored = self._read(descriptor, history)

        self.assertEqual(restored.schema, QWEN35_MTP_CARRY_SCHEMA)
        self.assertEqual(restored.identity, original.identity)
        self.assertEqual(restored.history, history)
        self.assertEqual(restored.next_position, len(history) - 1)
        self.assertEqual(restored.state_bytes, original.state_bytes)
        self.assertIsNotNone(restored.state)
        assert restored.state is not None
        assert original.state is not None
        self.assertTrue(torch.equal(restored.state.key, original.state.key))
        self.assertTrue(torch.equal(restored.state.value, original.state.value))
        self.assertTrue(
            torch.equal(restored.last_target_hidden, original.last_target_hidden)
        )
        for tensor in (
            restored.state.key,
            restored.state.value,
            restored.last_target_hidden,
        ):
            self.assertEqual(tensor.device.type, "cpu")
            self.assertEqual(tensor.dtype, torch.bfloat16)
            self.assertTrue(tensor.is_contiguous())
        self.assertEqual(descriptor.basename, Path(descriptor.basename).name)
        self.assertNotIn(str(self.root), descriptor.to_record().values())
        self.assertEqual(
            descriptor.identity_sha256,
            qwen35_mtp_carry_identity_sha256(_IDENTITY),
        )
        self.assertEqual(
            (self.root / descriptor.basename).stat().st_mode & 0o777,
            0o600,
        )

    def test_one_token_prefix_has_no_attention_state_and_empty_is_rejected(self) -> None:
        history = (17,)
        original, descriptor = self._write(history)
        restored = self._read(descriptor, history)
        self.assertIsNone(original.state)
        self.assertIsNone(restored.state)
        self.assertEqual(restored.next_position, 0)
        self.assertEqual(restored.state_bytes, original.state_bytes)

        invalid = Qwen35MtpCarry(
            schema=QWEN35_MTP_CARRY_SCHEMA,
            identity=_IDENTITY,
            history=(),
            next_position=-1,
            state=None,
            last_target_hidden=original.last_target_hidden,
        )
        with self.assertRaisesRegex(Qwen35MtpCarrySidecarError, "non-empty"):
            write_qwen35_mtp_carry_sidecar(
                self.root,
                invalid,
                tokenizer_sha256=_TOKENIZER_SHA256,
                prefix_sha256="0" * 64,
            )

    def test_prefix_digest_matches_semantic_cache_without_import_dependency(self) -> None:
        history = (2, 3, 5, 7, 11)
        self.assertEqual(
            qwen35_mtp_prefix_sha256(history), token_prefix_sha256(history)
        )

    def test_wrong_prefix_tokenizer_and_identity_fail_closed(self) -> None:
        history = (11, 22, 33)
        _carry_value, descriptor = self._write(history)
        cases = (
            {"history": (11, 22, 34)},
            {"history": (11, 22)},
            {"tokenizer_sha256": hashlib.sha256(b"other").hexdigest()},
            {
                "identity": (
                    *_IDENTITY[:-3],
                    (64, 2, 1, 16),
                    "cpu",
                    "torch.bfloat16",
                )
            },
        )
        for changes in cases:
            with self.subTest(changes=changes):
                with self.assertRaises(Qwen35MtpCarrySidecarIdentityMismatch):
                    self._read(
                        descriptor,
                        changes.get("history", history),
                        tokenizer_sha256=changes.get(
                            "tokenizer_sha256", _TOKENIZER_SHA256
                        ),
                        identity=changes.get("identity", _IDENTITY),
                    )

    def test_tensor_metadata_file_and_descriptor_tampering_are_rejected(self) -> None:
        history = (101, 202, 303)
        _carry_value, descriptor = self._write(history)
        source = self.root / descriptor.basename
        raw = bytearray(source.read_bytes())

        tensor_tampered = bytearray(raw)
        tensor_tampered[-1] ^= 0x80
        tensor_descriptor = self._publish_modified(descriptor, tensor_tampered)
        with self.assertRaisesRegex(Qwen35MtpCarrySidecarError, "tensor|payload"):
            self._read(tensor_descriptor, history)

        magic_size = len(b"IMMER-MTP-CARRY\x00\x01")
        header_length = struct.unpack(">I", raw[magic_size : magic_size + 4])[0]
        header_start = magic_size + 4
        header = json.loads(raw[header_start : header_start + header_length])
        header["body"]["next_position"] = 9
        changed_header = json.dumps(
            header,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self.assertEqual(len(changed_header), header_length)
        metadata_tampered = (
            raw[:header_start]
            + changed_header
            + raw[header_start + header_length :]
        )
        metadata_descriptor = self._publish_modified(descriptor, metadata_tampered)
        with self.assertRaisesRegex(Qwen35MtpCarrySidecarError, "body SHA-256"):
            self._read(metadata_descriptor, history)

        raw[0] ^= 0xFF
        with source.open("wb") as handle:
            handle.write(raw)
        with self.assertRaisesRegex(Qwen35MtpCarrySidecarError, "SHA-256"):
            self._read(descriptor, history)

        for changed in (
            replace(descriptor, bytes=descriptor.bytes + 1),
            replace(descriptor, identity_sha256="f" * 64),
            replace(descriptor, tensor_manifest_sha256="e" * 64),
        ):
            with self.assertRaises(Qwen35MtpCarrySidecarError):
                self._read(changed, history)

    def test_symlink_roots_files_and_path_escape_are_rejected(self) -> None:
        history = (7, 8, 9)
        _carry_value, descriptor = self._write(history)
        alias = self.root.parent / f"{self.root.name}-alias"
        alias.symlink_to(self.root, target_is_directory=True)
        self.addCleanup(alias.unlink)
        with self.assertRaisesRegex(Qwen35MtpCarrySidecarError, "real directory"):
            read_qwen35_mtp_carry_sidecar(
                alias,
                descriptor,
                history=history,
                tokenizer_sha256=_TOKENIZER_SHA256,
                expected_identity=_IDENTITY,
            )

        link_root = self.root / "links"
        link_root.mkdir()
        (link_root / descriptor.basename).symlink_to(
            self.root / descriptor.basename
        )
        with self.assertRaises(Qwen35MtpCarrySidecarError):
            read_qwen35_mtp_carry_sidecar(
                link_root,
                descriptor,
                history=history,
                tokenizer_sha256=_TOKENIZER_SHA256,
                expected_identity=_IDENTITY,
            )

        with self.assertRaisesRegex(Qwen35MtpCarrySidecarError, "content address"):
            MtpCarrySidecarDescriptor(
                basename=f"../{descriptor.basename}",
                bytes=descriptor.bytes,
                file_sha256=descriptor.file_sha256,
                identity_sha256=descriptor.identity_sha256,
                tensor_manifest_sha256=descriptor.tensor_manifest_sha256,
            )

    def test_publish_is_idempotent_and_concurrent_without_overwrite(self) -> None:
        history = (13, 21, 34, 55)
        carry = _carry(history)
        prefix_sha = qwen35_mtp_prefix_sha256(history)

        def publish(_: int) -> MtpCarrySidecarDescriptor:
            return write_qwen35_mtp_carry_sidecar(
                self.root,
                carry,
                tokenizer_sha256=_TOKENIZER_SHA256,
                prefix_sha256=prefix_sha,
            )

        first = publish(0)
        with ThreadPoolExecutor(max_workers=8) as pool:
            descriptors = tuple(pool.map(publish, range(24)))
        self.assertTrue(all(row == first for row in descriptors))
        self.assertEqual(
            [path.name for path in self.root.iterdir()],
            [first.basename],
        )
        self._read(first, history)

    def test_file_contains_no_history_prompt_or_pickle_material(self) -> None:
        history = (8_765_431, 8_765_433, 8_765_439)
        _carry_value, descriptor = self._write(history)
        raw = (self.root / descriptor.basename).read_bytes()
        lowered = raw.lower()
        self.assertNotIn(b"pickle", lowered)
        self.assertNotIn(b"history", lowered)
        self.assertNotIn(b"confidential prompt canary", lowered)
        for token in history:
            self.assertNotIn(str(token).encode("ascii"), raw)
            self.assertNotIn(struct.pack(">q", token), raw)
            self.assertNotIn(struct.pack("<q", token), raw)

    def test_full_file_path_is_accepted_only_for_descriptor_basename(self) -> None:
        history = (3, 1, 4)
        _carry_value, descriptor = self._write(history)
        path = self.root / descriptor.basename
        restored = read_qwen35_mtp_carry_sidecar(
            path,
            descriptor,
            history=history,
            tokenizer_sha256=_TOKENIZER_SHA256,
            expected_identity=_IDENTITY,
        )
        self.assertEqual(restored.history, history)


if __name__ == "__main__":
    unittest.main()
