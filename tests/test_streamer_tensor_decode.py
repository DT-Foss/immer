from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from immer.knowledge.streamer import (
    ByteBudgetExceeded,
    Streamer,
    TensorEncodingError,
)


def _bf16_bytes(values: np.ndarray) -> bytes:
    words = np.asarray(values, dtype="<f4").view("<u4") >> 16
    return words.astype("<u2").tobytes()


def _write_safetensors(
    root: Path,
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
) -> None:
    header: dict[str, object] = {}
    payload = bytearray()
    for name, (dtype, shape, body) in tensors.items():
        start = len(payload)
        payload.extend(body)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    header["__metadata__"] = {"fixture": "tensor-decode"}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    (root / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + payload
    )


class StreamerTensorDecodeTests(unittest.TestCase):
    def test_tensor_decodes_all_preexisting_standard_dtypes_and_bf16(self) -> None:
        standard = {
            "bool": np.array([[False, True], [True, False]], dtype="?"),
            "u8": np.array([0, 255], dtype="u1"),
            "i8": np.array([-128, 127], dtype="i1"),
            "i16": np.array([-1234, 2345], dtype="<i2"),
            "u16": np.array([0, 65535], dtype="<u2"),
            "f16": np.array([-1.5, 0.25], dtype="<f2"),
            "i32": np.array([-123456, 234567], dtype="<i4"),
            "u32": np.array([0, 4_000_000_000], dtype="<u4"),
            "f32": np.array([-3.25, 9.5], dtype="<f4"),
            "i64": np.array([-(2**50), 2**50], dtype="<i8"),
            "u64": np.array([0, 2**63 + 7], dtype="<u8"),
            "f64": np.array([-np.pi, np.e], dtype="<f8"),
        }
        dtype_names = {
            "bool": "BOOL",
            "u8": "U8",
            "i8": "I8",
            "i16": "I16",
            "u16": "U16",
            "f16": "F16",
            "i32": "I32",
            "u32": "U32",
            "f32": "F32",
            "i64": "I64",
            "u64": "U64",
            "f64": "F64",
        }
        bf16_values = np.array([[0.5, -1.0], [2.0, 7.5]], dtype="<f4")
        fixture = {
            name: (dtype_names[name], array.shape, array.tobytes())
            for name, array in standard.items()
        }
        fixture["bf16"] = ("BF16", bf16_values.shape, _bf16_bytes(bf16_values))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_safetensors(root, fixture)
            source = Streamer.from_local(root, use_cache=False, budget_mb=1.0)

            for name, expected in standard.items():
                actual = source.tensor(name)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertTrue(actual.flags.writeable)
                self.assertTrue(actual.flags.c_contiguous)

            decoded_bf16 = source.tensor("bf16")
            np.testing.assert_array_equal(decoded_bf16, bf16_values)
            self.assertEqual(decoded_bf16.dtype, np.dtype("float32"))
            self.assertTrue(decoded_bf16.flags.writeable)

    def test_tensor_decodes_all_valid_ocp_e4m3fn_boundaries_and_e8m0(self) -> None:
        e4m3 = bytes(
            [
                0x00,
                0x80,
                0x01,
                0x81,
                0x07,
                0x87,
                0x08,
                0x88,
                0x38,
                0x3C,
                0x77,
                0x7E,
                0xB8,
                0xFE,
            ]
        )
        e8m0 = bytes([0x00, 0x01, 0x7E, 0x7F, 0x80, 0xFE])
        fixture = {
            "e4m3": ("F8_E4M3", (14,), e4m3),
            "e4m3fn": ("F8_E4M3FN", (14,), e4m3),
            "e8m0": ("F8_E8M0", (6,), e8m0),
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_safetensors(root, fixture)
            source = Streamer.from_local(root, use_cache=False, budget_mb=1.0)

            expected_e4m3 = np.array(
                [
                    0.0,
                    -0.0,
                    2.0**-9,
                    -(2.0**-9),
                    7.0 * 2.0**-9,
                    -(7.0 * 2.0**-9),
                    2.0**-6,
                    -(2.0**-6),
                    1.0,
                    1.5,
                    240.0,
                    448.0,
                    -1.0,
                    -448.0,
                ],
                dtype=np.float32,
            )
            for name in ("e4m3", "e4m3fn"):
                decoded = source.tensor(name)
                np.testing.assert_array_equal(decoded, expected_e4m3)
                self.assertEqual(decoded.dtype, np.dtype("float32"))
                self.assertTrue(np.signbit(decoded[1]))
                self.assertTrue(np.signbit(decoded[3]))
                self.assertTrue(np.signbit(decoded[5]))
                self.assertTrue(np.signbit(decoded[7]))

            decoded_e8m0 = source.tensor("e8m0")
            expected_e8m0 = np.array(
                [2.0**-127, 2.0**-126, 0.5, 1.0, 2.0, 2.0**127],
                dtype=np.float32,
            )
            np.testing.assert_array_equal(decoded_e8m0, expected_e8m0)
            self.assertEqual(decoded_e8m0.dtype, np.dtype("float32"))

    def test_tensor_rejects_every_reserved_float8_encoding(self) -> None:
        fixture = {
            "e4m3_positive_nan": ("F8_E4M3", (2,), bytes([0x38, 0x7F])),
            "e4m3_negative_nan": ("F8_E4M3", (2,), bytes([0x38, 0xFF])),
            "e4m3fn_positive_nan": ("F8_E4M3FN", (2,), bytes([0x38, 0x7F])),
            "e4m3fn_negative_nan": ("F8_E4M3FN", (2,), bytes([0x38, 0xFF])),
            "e8m0_nan": ("F8_E8M0", (2,), bytes([0x7F, 0xFF])),
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_safetensors(root, fixture)
            source = Streamer.from_local(root, use_cache=False, budget_mb=1.0)

            for name in fixture:
                dtype = fixture[name][0]
                code = "0x7F" if "positive" in name else "0xFF"
                with (
                    self.subTest(name=name),
                    self.assertRaisesRegex(
                        TensorEncodingError,
                        rf"{dtype}-Kodierung {code} bei Element 1$",
                    ),
                ):
                    source.tensor(name)

    def test_reserved_float8_encoding_fails_on_range_and_resume_cache_paths(
        self,
    ) -> None:
        fixture = {"weight": ("F8_E4M3FN", (2, 2), bytes([0x38, 0x3C, 0x7F, 0xFE]))}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            cache = Path(tmp) / "cache"
            root.mkdir()
            _write_safetensors(root, fixture)

            source = Streamer.from_local(root, cache_dir=cache, budget_mb=1.0)
            source.inventory()
            with self.assertRaisesRegex(
                TensorEncodingError,
                r"F8_E4M3FN-Kodierung 0x7F bei Element 2$",
            ):
                source.tensor("weight")
            moved = source.bytes_moved()

            # rows() uses the same exact cached payload range but reports the
            # first invalid element relative to that selected row.
            with self.assertRaisesRegex(
                TensorEncodingError,
                r"F8_E4M3FN-Kodierung 0x7F bei Element 0$",
            ):
                source.rows("weight", start_row=1, n_rows=1)
            self.assertGreater(source.bytes_moved(), moved)

            resumed = Streamer.from_local(root, cache_dir=cache, budget_mb=0.0)
            with self.assertRaisesRegex(
                TensorEncodingError,
                r"F8_E4M3FN-Kodierung 0x7F bei Element 2$",
            ):
                resumed.tensor("weight")
            self.assertEqual(resumed.bytes_moved(), 0)
            self.assertEqual(resumed.metrics()["cache_hits"], 1)

    def test_tensor_uses_one_exact_payload_range_and_verified_resume_cache(
        self,
    ) -> None:
        values = np.arange(12, dtype="<f4").reshape(3, 4)
        fixture = {"weight": ("F32", values.shape, values.tobytes())}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            cache = Path(tmp) / "cache"
            root.mkdir()
            _write_safetensors(root, fixture)

            first = Streamer.from_local(root, cache_dir=cache, budget_mb=1.0)
            first.inventory()
            before = first.metrics()
            decoded = first.tensor("weight")
            after = first.metrics()
            np.testing.assert_array_equal(decoded, values)
            self.assertEqual(after["range_requests"] - before["range_requests"], 1)
            self.assertEqual(
                after["range_bytes_requested"] - before["range_bytes_requested"],
                values.nbytes,
            )

            resumed = Streamer.from_local(root, cache_dir=cache, budget_mb=0.0)
            np.testing.assert_array_equal(resumed.tensor("weight"), values)
            self.assertEqual(resumed.bytes_moved(), 0)
            self.assertEqual(resumed.metrics()["cache_hits"], 1)

    def test_tensor_budget_rejection_happens_before_payload_io(self) -> None:
        values = np.arange(8, dtype="<f4")
        fixture = {"weight": ("F32", values.shape, values.tobytes())}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_safetensors(root, fixture)
            source = Streamer.from_local(root, use_cache=False, budget_mb=1.0)
            source.inventory()
            used = source.budget.total
            source.budget.limit = used + values.nbytes - 1
            before = source.metrics()

            with self.assertRaises(ByteBudgetExceeded):
                source.tensor("weight")

            after = source.metrics()
            self.assertEqual(source.budget.total, used)
            self.assertEqual(after["range_requests"], before["range_requests"])
            self.assertGreater(after["budget"]["rejected_charges"], 0)


if __name__ == "__main__":
    unittest.main()
