from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch


class _Metrics:
    def __init__(self, values=None):
        self.values = {} if values is None else dict(values)

    def metrics(self):
        return dict(self.values)


class _FakePager(_Metrics):
    def __init__(self, predictions, *, head_failures=0, head_error=None):
        super().__init__({"linear_calls": 4})
        self.source = _Metrics({"network_or_source_body_bytes": 100})
        self.predictions = tuple(predictions)
        self.head_inputs = []
        self.releases = 0
        self.head_failures = head_failures
        self.head_error = head_error or TimeoutError("transient head timeout")

    def release(self):
        self.releases += 1

    def topk_logits(self, hidden, **kwargs):
        self.head_inputs.append(hidden.detach().to(device="cpu").clone())
        if self.head_failures:
            self.head_failures -= 1
            raise self.head_error
        self.values["linear_calls"] += 1
        self.source.values["network_or_source_body_bytes"] += 5
        if hidden.shape[0] != len(self.predictions):
            raise AssertionError("fake prediction schedule has the wrong length")
        return (
            torch.ones((hidden.shape[0], 1), dtype=torch.float32),
            torch.tensor(self.predictions, dtype=torch.long).reshape(-1, 1),
        )


class _FakeModel:
    def __init__(
        self,
        predictions,
        *,
        graft=None,
        graft_layer=None,
        head_failures=0,
        head_error=None,
    ):
        self.config = SimpleNamespace(
            n_layers=2,
            vocab_size=128,
            hc_mult=1,
            dim=2,
        )
        self.pager = _FakePager(
            predictions,
            head_failures=head_failures,
            head_error=head_error,
        )
        self.max_batch_size = 3
        self.max_seq_len = 8
        self.graft = graft
        self.graft_layer = graft_layer
        self.layer_calls = []
        self.embedded = []
        self.reset_calls = 0

    def reset_state(self, *, release=False):
        self.reset_calls += int(release)

    def embed_batch(self, token_ids):
        ids = torch.as_tensor(token_ids, dtype=torch.long)
        self.embedded.append(ids.clone())
        return (
            ids.to(dtype=torch.float32).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 2)
        )

    def forward_prefill_layer(
        self,
        hidden,
        token_ids,
        *,
        layer,
        token_mask=None,
    ):
        ids = torch.as_tensor(token_ids).clone()
        mask = torch.as_tensor(token_mask).clone()
        self.layer_calls.append((layer, ids, mask))
        self.pager.values["linear_calls"] += 2
        self.pager.source.values["network_or_source_body_bytes"] += 10
        return hidden, ()

    def finalize_hidden(self, hidden):
        return hidden[:, :, 0]


class _FakeGraft:
    mode = "crsa"
    alpha = 0.01

    def __init__(self):
        self.inputs = []

    def forward(self, hidden):
        self.inputs.append(hidden.detach().clone())
        return hidden + 0.5


class _FlakyModel(_FakeModel):
    def __init__(self, predictions):
        super().__init__(predictions)
        self.failures_remaining = 1
        self.attempts = []

    def forward_prefill_layer(self, hidden, token_ids, **kwargs):
        self.attempts.append(
            (
                hidden.detach().clone(),
                torch.as_tensor(token_ids).clone(),
                torch.as_tensor(kwargs["token_mask"]).clone(),
                kwargs["layer"],
            )
        )
        if kwargs["layer"] == 0 and self.failures_remaining:
            self.failures_remaining -= 1
            raise TimeoutError("transient range timeout")
        return super().forward_prefill_layer(hidden, token_ids, **kwargs)


class DeepSeekV4DraftVerificationTests(unittest.TestCase):
    def test_full_match_uses_one_ragged_prefill_and_one_head_scan(self):
        from immer.runtimes.deepseek_v4.draft_verification import (
            LayerwiseDraftVerifier,
        )

        model = _FakeModel((12, 13, 99, 21, 99))
        report = LayerwiseDraftVerifier(model).verify(
            [[10, 11], [20]],
            [[12, 13], [21]],
            eos_token_id=99,
        )

        self.assertTrue(report.all_verified)
        self.assertEqual(report.rows[0].target_token_ids, (12, 13, 99))
        self.assertEqual(report.rows[0].accepted_prefix_length, 2)
        self.assertTrue(report.rows[0].draft_verified)
        self.assertTrue(report.rows[0].eos_verified)
        self.assertEqual(report.rows[1].accepted_prefix_length, 1)
        self.assertEqual(len(model.layer_calls), 2)
        expected_ids = torch.tensor([[10, 11, 12, 13], [20, 21, 0, 0]])
        expected_mask = torch.tensor(
            [[True, True, True, True], [True, True, False, False]]
        )
        for layer, ids, mask in model.layer_calls:
            self.assertIn(layer, (0, 1))
            self.assertTrue(torch.equal(ids, expected_ids))
            self.assertTrue(torch.equal(mask, expected_mask))
        self.assertEqual(len(model.pager.head_inputs), 1)
        self.assertEqual(
            model.pager.head_inputs[0][:, 0].tolist(),
            [11.0, 12.0, 13.0, 20.0, 21.0],
        )
        evidence = report.evidence
        self.assertEqual(evidence.layer_calls, 2)
        self.assertEqual(evidence.head_scans, 1)
        self.assertEqual(evidence.verification_rows, 5)
        self.assertEqual(evidence.linear_calls, 5)
        self.assertEqual(evidence.source_body_bytes, 25)
        self.assertTrue(evidence.fixed_model_batch)
        self.assertTrue(evidence.right_prefix_mask)
        self.assertEqual(model.reset_calls, 1)

    def test_first_and_middle_mismatches_report_target_and_draft(self):
        from immer.runtimes.deepseek_v4.draft_verification import (
            LayerwiseDraftVerifier,
        )

        model = _FakeModel((9, 3, 5, 9, 7))
        report = LayerwiseDraftVerifier(model).verify(
            [[1], [4]],
            [[2, 3], [5, 6, 7]],
        )

        first, middle = report.rows
        self.assertFalse(report.all_verified)
        self.assertEqual(first.accepted_prefix_length, 0)
        self.assertEqual(first.first_mismatch_index, 0)
        self.assertEqual(first.first_mismatch_target_token_id, 9)
        self.assertEqual(first.first_mismatch_draft_token_id, 2)
        self.assertFalse(first.first_mismatch_is_eos)
        self.assertEqual(middle.accepted_prefix_length, 1)
        self.assertEqual(middle.first_mismatch_index, 1)
        self.assertEqual(middle.first_mismatch_target_token_id, 9)
        self.assertEqual(middle.first_mismatch_draft_token_id, 6)
        self.assertFalse(middle.draft_verified)
        self.assertIsNone(middle.eos_verified)

    def test_optional_eos_supports_empty_draft_and_distinguishes_eos_failure(self):
        from immer.runtimes.deepseek_v4.draft_verification import (
            LayerwiseDraftVerifier,
        )

        model = _FakeModel((2, 8, 7))
        report = LayerwiseDraftVerifier(model).verify(
            [[1], [3, 4]],
            [[2], []],
            eos_token_id=7,
        )

        content, eos_only = report.rows
        self.assertEqual(content.accepted_prefix_length, 1)
        self.assertTrue(content.draft_verified)
        self.assertFalse(content.eos_verified)
        self.assertFalse(content.fully_verified)
        self.assertEqual(content.first_mismatch_index, 1)
        self.assertTrue(content.first_mismatch_is_eos)
        self.assertEqual(content.first_mismatch_target_token_id, 8)
        self.assertEqual(content.first_mismatch_draft_token_id, 7)
        self.assertEqual(eos_only.draft_token_ids, ())
        self.assertEqual(eos_only.target_token_ids, (7,))
        self.assertEqual(eos_only.accepted_prefix_length, 0)
        self.assertTrue(eos_only.draft_verified)
        self.assertTrue(eos_only.eos_verified)
        self.assertTrue(eos_only.fully_verified)

        wrong_prefix = LayerwiseDraftVerifier(_FakeModel((9, 3, 7))).verify(
            [[1]],
            [[2, 3]],
            eos_token_id=7,
        ).rows[0]
        self.assertEqual(wrong_prefix.accepted_prefix_length, 0)
        self.assertIsNone(wrong_prefix.eos_verified)
        self.assertFalse(wrong_prefix.fully_verified)

    def test_active_graft_runs_once_at_the_selected_layer(self):
        from immer.runtimes.deepseek_v4.draft_verification import (
            LayerwiseDraftVerifier,
        )

        graft = _FakeGraft()
        model = _FakeModel((2, 3), graft=graft, graft_layer=0)
        report = LayerwiseDraftVerifier(model).verify([[1]], [[2, 3]])

        self.assertTrue(report.all_verified)
        self.assertEqual(len(graft.inputs), 1)
        self.assertEqual(tuple(graft.inputs[0].shape), (1, 3, 1, 2))
        self.assertEqual(model.pager.head_inputs[0][:, 0].tolist(), [1.5, 2.5])
        self.assertTrue(report.evidence.graft_applied)
        self.assertEqual(report.evidence.graft_layer, 0)
        self.assertEqual(report.evidence.graft_mode, "crsa")

    def test_retries_only_the_failed_layer_and_reports_progress(self):
        from immer.runtimes.deepseek_v4.draft_verification import (
            LayerwiseDraftVerifier,
        )

        model = _FlakyModel((2, 3))
        progress = []
        report = LayerwiseDraftVerifier(model, layer_retries=1).verify(
            [[1]],
            [[2, 3]],
            progress=progress.append,
        )

        self.assertTrue(report.all_verified)
        self.assertEqual(report.evidence.layer_calls, 2)
        self.assertEqual(report.evidence.layer_retry_count, 1)
        self.assertEqual(
            [row["event"] for row in progress],
            [
                "layer_retry",
                "layer_complete",
                "layer_complete",
            ],
        )
        self.assertEqual(progress[0]["layer"], 0)
        self.assertEqual(progress[0]["attempt"], 2)
        self.assertEqual(progress[0]["error_type"], "TimeoutError")
        self.assertEqual(len(model.attempts), 3)
        torch.testing.assert_close(model.attempts[0][0], model.attempts[1][0])
        self.assertTrue(torch.equal(model.attempts[0][1], model.attempts[1][1]))
        self.assertTrue(torch.equal(model.attempts[0][2], model.attempts[1][2]))
        self.assertGreaterEqual(model.pager.releases, 4)

    def test_retries_transient_head_failure_but_not_deterministic_layer_error(self):
        from immer.runtimes.deepseek_v4.draft_verification import (
            LayerwiseDraftVerifier,
        )

        model = _FakeModel((2,), head_failures=1)
        progress = []
        report = LayerwiseDraftVerifier(model, head_retries=1).verify(
            [[1]],
            [[2]],
            progress=progress.append,
        )
        self.assertTrue(report.all_verified)
        self.assertEqual(report.evidence.head_retry_count, 1)
        self.assertEqual(progress[-1]["event"], "head_retry")
        self.assertEqual(len(model.pager.head_inputs), 2)

        deterministic = _FlakyModel((2,))
        deterministic.failures_remaining = 0
        attempts = 0

        def invalid(*_args, **_kwargs):
            nonlocal attempts
            attempts += 1
            raise ValueError("deterministic bad shape")

        deterministic.forward_prefill_layer = invalid
        with self.assertRaisesRegex(ValueError, "bad shape"):
            LayerwiseDraftVerifier(deterministic, layer_retries=2).verify(
                [[1]],
                [[2]],
            )
        self.assertEqual(attempts, 1)

    def test_rejects_invalid_rows_bounds_and_tokens_before_forward(self):
        from immer.runtimes.deepseek_v4.draft_verification import (
            LayerwiseDraftVerifier,
        )

        cases = (
            (([[]], [[]]), {}, "non-empty"),
            (([[1], [2]], [[3]]), {}, "same size"),
            (([[1.5]], [[2]]), {}, "integers"),
            (([[1]], [[]]), {}, "draft token or optional EOS"),
            (([[1]], [[2, 3]]), {"max_draft_tokens": 1}, "max_draft_tokens"),
            (([[1]], [[2, 7]]), {"eos_token_id": 7}, "exclude"),
            (([[1]], [[128]]), {}, "outside"),
            (([[1]], [[2]]), {"padding_token_id": 128}, "outside"),
        )
        for arguments, kwargs, message in cases:
            with self.subTest(message=message):
                model = _FakeModel((2,))
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    LayerwiseDraftVerifier(model).verify(*arguments, **kwargs)
                self.assertEqual(model.layer_calls, [])

        oversized = _FakeModel((2,))
        oversized.max_seq_len = 2
        with self.assertRaisesRegex(ValueError, "context bound"):
            LayerwiseDraftVerifier(oversized).verify([[1, 2]], [[3]])
        self.assertEqual(oversized.layer_calls, [])

        batch = _FakeModel((2,))
        batch.max_batch_size = 1
        with self.assertRaisesRegex(ValueError, "max_batch_size"):
            LayerwiseDraftVerifier(batch).verify([[1], [2]], [[3], [4]])
        self.assertEqual(batch.layer_calls, [])

        with self.assertRaisesRegex(ValueError, "layer_retries"):
            LayerwiseDraftVerifier(_FakeModel((2,)), layer_retries=-1)
        with self.assertRaisesRegex(ValueError, "head_retries"):
            LayerwiseDraftVerifier(_FakeModel((2,)), head_retries=-1)
        with self.assertRaisesRegex(TypeError, "progress"):
            LayerwiseDraftVerifier(_FakeModel((2,))).verify(
                [[1]], [[2]], progress=object()
            )

    def test_accepts_integer_numpy_and_tensor_rows(self):
        from immer.runtimes.deepseek_v4.draft_verification import (
            LayerwiseDraftVerifier,
        )

        model = _FakeModel((3, 4))
        report = LayerwiseDraftVerifier(model).verify(
            np.asarray([[1, 2]], dtype=np.int64),
            torch.tensor([[3, 4]], dtype=torch.int32),
        )
        self.assertTrue(report.all_verified)

    def test_tiny_real_runtime_matches_itemwise_stateful_prefill(self):
        from immer.runtimes.deepseek_v4 import (
            DeepSeekWeightPager,
            StreamedDeepSeekV4,
        )
        from immer.runtimes.deepseek_v4.draft_verification import (
            LayerwiseDraftVerifier,
        )
        from test_deepseek_v4_model import _CompressedTinyCheckpoint, _config

        model = StreamedDeepSeekV4(
            _config(4),
            DeepSeekWeightPager(
                _CompressedTinyCheckpoint(random_weights=True, compress_ratio=4),
                device="cpu",
                compute_dtype="float32",
            ),
            max_batch_size=2,
            max_seq_len=10,
        )
        prompts = ((1, 2, 3), (4, 5))
        drafts = ((6, 7), (8,))
        report = LayerwiseDraftVerifier(model).verify(prompts, drafts)

        expected = []
        for prompt, draft in zip(prompts, drafts, strict=True):
            combined = prompt + draft
            final, _evidence = model.prefill([combined], tokenwise=False)
            positions = torch.tensor(
                [len(prompt) - 1 + offset for offset in range(len(draft))]
            )
            _values, selected = model.pager.topk_logits(
                final[0].index_select(0, positions),
                k=1,
            )
            expected.append(tuple(int(value) for value in selected[:, 0].tolist()))
        self.assertEqual(
            tuple(row.target_token_ids for row in report.rows),
            tuple(expected),
        )


if __name__ == "__main__":
    unittest.main()
