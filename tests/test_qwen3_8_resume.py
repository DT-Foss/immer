from __future__ import annotations

import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from immer.runtimes.qwen3_8.draft_verification import (
    DraftVerificationResumeState,
)
from immer.runtimes.qwen3_8.resume import (
    Qwen38ResumeError,
    RESUME_SCHEMA,
    build_resume_identity,
    delete_resume,
    hidden_state_bytes,
    load_resume,
    preflight_resume_disk,
    write_resume,
)


SHAPE = (2, 3, 4)
DTYPE = "bfloat16"


def _identity(**changes: object) -> str:
    values: dict[str, object] = {
        "source_id": "Qwen/Qwen3.8-27B",
        "source_revision": "1d4bf0f2",
        "prompt_token_ids": ((1, 2), (3,)),
        "draft_token_ids": ((4,), (5, 6)),
        "execution_contract": {
            "device": "cpu",
            "dtype": DTYPE,
            "padding_token_id": 0,
            "eos_token_id": 2,
            "head_block_rows": 128,
        },
        "graft_contract": {"mode": "off", "layer": None},
    }
    values.update(changes)
    return build_resume_identity(**values)  # type: ignore[arg-type]


def _state(
    next_layer: int = 2,
    *,
    value: float = 2.0,
    dtype: torch.dtype = torch.bfloat16,
    graft_applied: bool = False,
) -> DraftVerificationResumeState:
    return DraftVerificationResumeState(
        next_layer=next_layer,
        hidden=torch.full(SHAPE, value, dtype=dtype),
        layer_calls=next_layer,
        layer_retry_count=1,
        source_body_bytes=1234,
        linear_calls=17,
        seconds=3.25,
        graft_applied=graft_applied,
    )


class Qwen38ResumeTests(unittest.TestCase):
    def test_identity_is_deterministic_and_binds_every_contract_axis(self) -> None:
        baseline = _identity()
        reordered = _identity(
            execution_contract={
                "head_block_rows": 128,
                "eos_token_id": 2,
                "padding_token_id": 0,
                "dtype": DTYPE,
                "device": "cpu",
            }
        )
        self.assertEqual(baseline, reordered)
        variants = (
            _identity(source_revision="other"),
            _identity(prompt_token_ids=((1, 9), (3,))),
            _identity(draft_token_ids=((4,), (7, 6))),
            _identity(execution_contract={"device": "mps", "dtype": DTYPE}),
            _identity(graft_contract={"mode": "stable-crsa", "layer": 1}),
        )
        self.assertEqual(len(baseline), 64)
        self.assertTrue(all(candidate != baseline for candidate in variants))

    def test_identity_rejects_ambiguous_or_nonfinite_inputs(self) -> None:
        with self.assertRaisesRegex(Qwen38ResumeError, "same size"):
            _identity(draft_token_ids=((4,),))
        with self.assertRaisesRegex(Qwen38ResumeError, "invalid token"):
            _identity(prompt_token_ids=((True,), (3,)))
        with self.assertRaisesRegex(Qwen38ResumeError, "non-finite"):
            _identity(execution_contract={"alpha": math.nan})

    def test_atomic_roundtrip_overwrite_and_explicit_restart(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path = root / "resume.safetensors"
            identity = _identity()
            write_resume(
                path,
                identity,
                _state(1, value=1.0),
                expected_shape=SHAPE,
                expected_dtype=DTYPE,
                n_layers=4,
            )
            latest = _state(2, value=7.0)
            write_resume(
                path,
                identity,
                latest,
                expected_shape=SHAPE,
                expected_dtype=DTYPE,
                n_layers=4,
            )
            self.assertEqual([entry.name for entry in root.iterdir()], [path.name])
            loaded = load_resume(
                path,
                identity,
                expected_shape=SHAPE,
                expected_dtype=DTYPE,
                n_layers=4,
            )
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.next_layer, 2)
            self.assertEqual(loaded.layer_retry_count, 1)
            self.assertTrue(torch.equal(loaded.hidden, latest.hidden))

            self.assertTrue(delete_resume(path))
            self.assertFalse(delete_resume(path))
            self.assertIsNone(
                load_resume(
                    path,
                    identity,
                    expected_shape=SHAPE,
                    expected_dtype=DTYPE,
                    n_layers=4,
                )
            )

    def test_load_rejects_identity_shape_dtype_and_layer_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "resume.safetensors"
            identity = _identity()
            write_resume(
                path,
                identity,
                _state(),
                expected_shape=SHAPE,
                expected_dtype=DTYPE,
                n_layers=4,
            )
            common = {
                "path": path,
                "identity": identity,
                "expected_shape": SHAPE,
                "expected_dtype": DTYPE,
                "n_layers": 4,
            }
            with self.assertRaisesRegex(Qwen38ResumeError, "another run"):
                load_resume(**(common | {"identity": "0" * 64}))
            with self.assertRaisesRegex(Qwen38ResumeError, "bound|shape"):
                load_resume(**(common | {"expected_shape": (1, 3, 4)}))
            with self.assertRaisesRegex(Qwen38ResumeError, "bound|dtype"):
                load_resume(**(common | {"expected_dtype": "float32"}))
            with self.assertRaisesRegex(Qwen38ResumeError, "layer boundary"):
                load_resume(**(common | {"n_layers": 1}))

    def test_graft_boundary_is_checked_on_write_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "resume.safetensors"
            identity = _identity(graft_contract={"mode": "stable-crsa", "layer": 1})
            write_resume(
                path,
                identity,
                _state(graft_applied=True),
                expected_shape=SHAPE,
                expected_dtype=DTYPE,
                n_layers=4,
                active_graft_layer=1,
            )
            with self.assertRaisesRegex(Qwen38ResumeError, "graft state"):
                load_resume(
                    path,
                    identity,
                    expected_shape=SHAPE,
                    expected_dtype=DTYPE,
                    n_layers=4,
                    active_graft_layer=None,
                )
            with self.assertRaisesRegex(Qwen38ResumeError, "graft state"):
                write_resume(
                    path,
                    identity,
                    _state(graft_applied=False),
                    expected_shape=SHAPE,
                    expected_dtype=DTYPE,
                    n_layers=4,
                    active_graft_layer=1,
                )

    def test_corruption_and_unexpected_metadata_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "resume.safetensors"
            identity = _identity()
            path.write_bytes(b"not a safetensors document")
            with self.assertRaisesRegex(Qwen38ResumeError, "bound|decode"):
                load_resume(
                    path,
                    identity,
                    expected_shape=SHAPE,
                    expected_dtype=DTYPE,
                    n_layers=4,
                )

            save_file(
                {"hidden": _state().hidden},
                path,
                metadata={"schema": RESUME_SCHEMA, "identity": identity},
            )
            with self.assertRaisesRegex(Qwen38ResumeError, "metadata schema"):
                load_resume(
                    path,
                    identity,
                    expected_shape=SHAPE,
                    expected_dtype=DTYPE,
                    n_layers=4,
                )

    def test_file_size_is_bounded_before_safetensors_decode(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "resume.safetensors"
            path.touch()
            with path.open("r+b") as handle:
                handle.truncate(hidden_state_bytes(SHAPE, DTYPE) + 64 * 1024 + 9)
            with mock.patch("safetensors.safe_open") as safe_open:
                with self.assertRaisesRegex(Qwen38ResumeError, "bound"):
                    load_resume(
                        path,
                        _identity(),
                        expected_shape=SHAPE,
                        expected_dtype=DTYPE,
                        n_layers=4,
                    )
                safe_open.assert_not_called()

    def test_symlinks_are_rejected_for_load_write_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target = root / "target.safetensors"
            target.write_bytes(b"target")
            path = root / "resume.safetensors"
            path.symlink_to(target)
            arguments = {
                "expected_shape": SHAPE,
                "expected_dtype": DTYPE,
                "n_layers": 4,
            }
            with self.assertRaisesRegex(Qwen38ResumeError, "symlink"):
                load_resume(path, _identity(), **arguments)
            with self.assertRaisesRegex(Qwen38ResumeError, "symlink"):
                write_resume(path, _identity(), _state(), **arguments)
            with self.assertRaisesRegex(Qwen38ResumeError, "symlink"):
                delete_resume(path)
            self.assertEqual(target.read_bytes(), b"target")

    def test_native_shape_bytes_and_disk_preflight_have_no_extra_reserve(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "nested" / "resume.safetensors"
            self.assertEqual(hidden_state_bytes(SHAPE, DTYPE), 48)
            with mock.patch(
                "immer.runtimes.qwen3_8.resume.shutil.disk_usage",
                return_value=SimpleNamespace(free=100_000),
            ):
                result = preflight_resume_disk(
                    path,
                    expected_shape=SHAPE,
                    expected_dtype=DTYPE,
                )
            self.assertEqual(result["hidden_bytes"], 48)
            self.assertEqual(result["required_bytes"], 48 + 8 + 64 * 1024)
            self.assertTrue(path.parent.is_dir())

            with (
                mock.patch(
                    "immer.runtimes.qwen3_8.resume.shutil.disk_usage",
                    return_value=SimpleNamespace(free=1),
                ),
                self.assertRaisesRegex(Qwen38ResumeError, "insufficient free disk"),
            ):
                preflight_resume_disk(
                    path,
                    expected_shape=SHAPE,
                    expected_dtype=DTYPE,
                )


if __name__ == "__main__":
    unittest.main()
