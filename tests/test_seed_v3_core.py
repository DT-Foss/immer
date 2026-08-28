from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from pathlib import Path
import tempfile
import unittest

import torch

from immer.runtimes.seed_v3 import (
    CausalPrefixSinkhornAttention,
    ImmerSeedV3,
    SeedV3CheckpointIdentityError,
    SeedV3CheckpointIntegrityError,
    SeedV3Config,
    SeedV3ShadowBatch,
    SeedV3ShadowTrainer,
    export_seed_v3_checkpoint,
    load_seed_v3_checkpoint,
    migrate_seed_v3_lab_checkpoint,
    seed_v3_shadow_losses,
    shadow_batch_from_projection,
)
from immer.runtimes.ooe.seed_projection import (
    SeedTrainingBatch,
    SeedTrainingProjection,
    SeedVectorProjection,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _config() -> SeedV3Config:
    return SeedV3Config(
        vocab_size=32,
        d_model=16,
        n_layers=2,
        n_heads=4,
        n_local_heads=1,
        n_balanced_heads=1,
        mlp_hidden=24,
        key_channels=1,
        key_code_bits=4,
        max_seq_len=8,
        local_window=2,
        operator_classes=6,
        state_classes=11,
        predictive_classes=7,
        quotient_classes=5,
        route_count=3,
        receipt_feature_dim=5,
    )


class SeedV3CoreTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_config_guards_total_key_width_and_free_path(self) -> None:
        config = _config()
        self.assertEqual(config.free_head_count, 2)
        self.assertEqual(config.receipt_feature_dim, 5)
        with self.assertRaisesRegex(ValueError, "Free"):
            SeedV3Config(vocab_size=8, n_heads=2, n_local_heads=1, n_balanced_heads=1)
        with self.assertRaisesRegex(ValueError, "concatenated"):
            SeedV3Config(
                vocab_size=8,
                d_model=8,
                n_layers=1,
                n_heads=2,
                n_local_heads=0,
                n_balanced_heads=1,
                mlp_hidden=8,
                key_channels=1,
            )
        with self.assertRaises(TypeError):
            SeedV3Config(vocab_size=8, receipt_feature_dim=True)

    def test_token_and_receipt_surfaces_emit_all_proposal_heads(self) -> None:
        model = ImmerSeedV3(_config())
        tokens = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
        answer = torch.tensor([2, 1])
        output = model(tokens, mask, answer)
        self.assertEqual(output.hidden.shape, (2, 4, 16))
        self.assertEqual(output.operator_logits.shape, (2, 4, 6))
        self.assertEqual(output.novelty_logits.shape, (2, 4, 2))
        self.assertEqual(output.predictive_logits.shape, (2, 4, 7))
        self.assertEqual(output.quotient_logits.shape, (2, 4, 5))
        self.assertEqual(output.route_values.shape, (2, 3))
        self.assertEqual(output.expected_work.shape, (2, 3))
        self.assertEqual(output.mlp_keys.shape, (2, 4, 4))
        self.assertTrue(bool((output.expected_work >= 0.0).all()))

        features = torch.randn(2, 4, 5)
        receipt_output = model.forward_receipts(features, mask, answer)
        self.assertEqual(receipt_output.hidden.shape, (2, 4, 16))
        with self.assertRaisesRegex(ValueError, "feature ABI"):
            model.forward_receipts(torch.randn(2, 4, 4), mask, answer)

    def test_attention_and_full_model_are_strictly_causal(self) -> None:
        model = ImmerSeedV3(_config()).eval()
        left = torch.tensor([[1, 2, 3, 4]])
        right = torch.tensor([[1, 2, 8, 9]])
        mask = torch.ones(1, 4, dtype=torch.bool)
        answer = torch.tensor([1])
        with torch.no_grad():
            left_output = model(left, mask, answer)
            right_output = model(right, mask, answer)
        torch.testing.assert_close(
            left_output.hidden[:, :2], right_output.hidden[:, :2], rtol=0, atol=0
        )

        attention = CausalPrefixSinkhornAttention(_config()).eval()
        values = torch.randn(1, 4, 16)
        _output, weights = attention(values, mask, return_weights=True)
        future = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
        self.assertTrue(bool((weights[..., future] == 0.0).all()))
        self.assertGreaterEqual(attention.n_free_heads, 1)

    def test_proposals_require_eval_mode(self) -> None:
        model = ImmerSeedV3(_config())
        features = torch.randn(1, 3, 5)
        mask = torch.ones(1, 3, dtype=torch.bool)
        answer = torch.tensor([2])
        with self.assertRaisesRegex(RuntimeError, "eval"):
            model.propose_receipts(features, mask, answer)
        model.eval()
        proposal = model.propose_receipts(features, mask, answer)
        self.assertEqual(proposal.authority, "proposal-only")
        self.assertEqual(proposal.operator_logits.shape, (1, 6))
        self.assertEqual(proposal.mlp_keys.shape, (1, 4))

    def test_shadow_trainer_updates_all_configured_heads(self) -> None:
        model = ImmerSeedV3(_config())
        features = torch.randn(2, 4, 5)
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)
        answer = torch.tensor([3, 2])
        operator = torch.tensor([[0, 1, 2, 3], [4, 5, 0, -100]])
        novelty = torch.tensor([[0, 0, 1, 0], [0, 1, 0, -100]])
        predictive = torch.tensor([[0, 1, 2, 3], [4, 5, 6, -100]])
        quotient = torch.tensor([[0, 1, 2, 3], [4, 0, 1, -100]])
        state = torch.tensor([[0, 1, 2, 3], [4, 5, 6, -100]])
        key_targets = torch.where(
            torch.randn(2, 4, 4) >= 0,
            torch.ones(2, 4, 4),
            -torch.ones(2, 4, 4),
        )
        batch = SeedV3ShadowBatch(
            receipt_features=features,
            attention_mask=mask,
            answer_pos=answer,
            operator_targets=operator,
            state_targets=state,
            final_targets=torch.tensor([3, 6]),
            novelty_targets=novelty,
            predictive_targets=predictive,
            quotient_targets=quotient,
            key_targets=key_targets,
            route_value_targets=torch.randn(2, 3),
            expected_work_targets=torch.rand(2, 3),
        )
        before = model.operator_head.weight.detach().clone()
        trainer = SeedV3ShadowTrainer(model, learning_rate=1.0e-3)
        losses = trainer.step(batch)
        self.assertEqual(trainer.steps, 1)
        self.assertTrue(all(math.isfinite(value) for value in losses.values()))
        self.assertGreater(losses["total"], 0.0)
        self.assertFalse(torch.equal(before, model.operator_head.weight.detach()))

    def test_sequence_losses_ignore_every_padded_label_automatically(self) -> None:
        model = ImmerSeedV3(_config())
        features = torch.randn(1, 4, 5)
        mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
        answer = torch.tensor([1])
        output = model.forward_receipts(features, mask, answer)
        first = SeedV3ShadowBatch(
            receipt_features=features,
            attention_mask=mask,
            answer_pos=answer,
            operator_targets=torch.tensor([[1, 2, 3, 4]]),
        )
        second = replace(
            first,
            operator_targets=torch.tensor([[1, 2, 5, 0]]),
        )
        first_losses = seed_v3_shadow_losses(output, first)
        second_losses = seed_v3_shadow_losses(output, second)
        torch.testing.assert_close(
            first_losses["operator"], second_losses["operator"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            first_losses["total"], second_losses["total"], rtol=0, atol=0
        )

    def test_receipt_batch_bridge_keeps_target_semantics_explicit(self) -> None:
        vector_schema = {
            name: _hash(f"{name}-schema/v1")
            for name in ("feature", "action", "consequence", "quality", "work")
        }

        def row(group: str, sequence: int) -> SeedTrainingProjection:
            vectors = SeedVectorProjection(
                feature_schema_sha256=vector_schema["feature"],
                feature=(0.1, 0.2, 0.3, 0.4, 0.5),
                action_schema_sha256=vector_schema["action"],
                action=(0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
                consequence_schema_sha256=vector_schema["consequence"],
                consequence=(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
                quality_schema_sha256=vector_schema["quality"],
                quality=(0.25, 0.5, 0.75),
                work_cost_schema_sha256=vector_schema["work"],
                work_cost=(0.2, 0.4, 0.6),
            )
            return SeedTrainingProjection(
                source_kind="measurement",
                source_receipt_sha256s=(_hash(f"source:{group}:{sequence}"),),
                model_pin_sha256=_hash("model"),
                code_identity_sha256=_hash("code"),
                site_identity_sha256s=(_hash(f"site:{group}"),),
                graph_revision_sha256s=(_hash("graph"),),
                authority_sha256s=(_hash("authority"),),
                verifier_sha256s=(_hash("verifier"),),
                split="train",
                generation=sequence,
                prompt_generation=1,
                group_id_sha256=_hash(f"group:{group}"),
                sequence_index=sequence,
                vectors=vectors,
            )

        batch = SeedTrainingBatch((row("a", 0), row("a", 1), row("b", 0)))

        def targets(views):
            operator = views.action.argmax(dim=-1)
            operator = operator.masked_fill(~views.attention_mask, -100)
            predictive = views.consequence.argmax(dim=-1)
            predictive = predictive.masked_fill(~views.attention_mask, -100)
            rows = torch.arange(len(views.group_sha256s))
            return {
                "operator_targets": operator,
                "predictive_targets": predictive,
                "route_value_targets": views.quality[rows, views.answer_pos],
                "expected_work_targets": views.work_cost[rows, views.answer_pos],
            }

        shadow, views = shadow_batch_from_projection(
            batch, "train", target_projector=targets
        )
        self.assertEqual(shadow.receipt_features.shape, (2, 2, 5))
        expected_positions = {
            _hash("group:a"): 1,
            _hash("group:b"): 0,
        }
        self.assertEqual(
            shadow.answer_pos.tolist(),
            [expected_positions[group] for group in views.group_sha256s],
        )
        self.assertEqual(views.batch_sha256, batch.sha256)
        self.assertEqual(views.batch_manifest_sha256, batch.manifest_sha256)
        losses = SeedV3ShadowTrainer(
            ImmerSeedV3(_config()), learning_rate=1.0e-3
        ).step(shadow)
        self.assertTrue(math.isfinite(losses["total"]))

    def test_checkpoint_is_sha_first_strict_and_inference_only(self) -> None:
        model = ImmerSeedV3(_config()).eval()
        feature_schema = _hash("receipt-feature-schema/v1")
        code = _hash("seed-code-v1")
        tokenizer = _hash("tokenizer-v1")
        sources = (_hash("receipt-a"), _hash("receipt-b"))
        features = torch.randn(1, 3, 5)
        mask = torch.ones(1, 3, dtype=torch.bool)
        answer = torch.tensor([2])
        with tempfile.TemporaryDirectory() as temporary:
            publication = export_seed_v3_checkpoint(
                model,
                temporary,
                code_revision_sha256=code,
                tokenizer_manifest_sha256=tokenizer,
                feature_schema_sha256=feature_schema,
                source_receipt_sha256s=sources,
            )
            self.assertIn(publication.manifest.sha256, publication.manifest_path.name)
            self.assertIn(publication.manifest.weights_sha256, publication.weights_path.name)
            self.assertNotIn("optimizer", publication.manifest.to_bytes().decode("utf-8"))
            with self.assertRaisesRegex(
                SeedV3CheckpointIntegrityError, "not unique"
            ):
                replace(
                    publication.manifest,
                    state_inventory=(
                        publication.manifest.state_inventory[0],
                        publication.manifest.state_inventory[0],
                    ),
                )
            restored, manifest = load_seed_v3_checkpoint(
                publication.manifest_path,
                expected_code_revision_sha256=code,
                expected_tokenizer_manifest_sha256=tokenizer,
                expected_feature_schema_sha256=feature_schema,
            )
            with torch.no_grad():
                expected = model.forward_receipts(features, mask, answer)
                actual = restored.forward_receipts(features, mask, answer)
            torch.testing.assert_close(expected.hidden, actual.hidden, rtol=0, atol=0)
            self.assertEqual(manifest.source_receipt_sha256s, tuple(sorted(sources)))
            self.assertIs(
                restored.lm_head.weight,
                restored.token_embedding.weight,
            )
            with self.assertRaises(SeedV3CheckpointIdentityError):
                load_seed_v3_checkpoint(
                    publication.manifest_path,
                    expected_feature_schema_sha256=_hash("other-feature-schema"),
                )

            weights = bytearray(publication.weights_path.read_bytes())
            weights[len(weights) // 2] ^= 1
            publication.weights_path.write_bytes(bytes(weights))
            with self.assertRaisesRegex(
                SeedV3CheckpointIntegrityError, "digest mismatch"
            ):
                load_seed_v3_checkpoint(publication.manifest_path)

    def test_shared_lab_checkpoint_migrates_without_optimizer_state(self) -> None:
        old_config = _config().to_dict()
        old_config["receipt_feature_dim"] = None
        old_config["state_classes"] = 17 * 17
        source_model = ImmerSeedV3(SeedV3Config.from_mapping(old_config))
        source_state = {}
        skipped = (
            "quotient_head.",
            "route_value_head.",
            "expected_work_head.",
        )
        for name, tensor in source_model.state_dict().items():
            if name.startswith(skipped):
                continue
            source_state[name.replace(".attention.", ".attn.")] = tensor
        lab_config_fields = {
            "vocab_size",
            "d_model",
            "n_layers",
            "n_heads",
            "n_local_heads",
            "n_balanced_heads",
            "mlp_hidden",
            "key_channels",
            "max_seq_len",
            "local_window",
            "crsa_alpha",
            "dropout",
            "predictive_classes",
            "gru_layers",
        }
        lab_config = {
            key: value
            for key, value in source_model.config.to_dict().items()
            if key in lab_config_fields
        }
        payload = {
            "batch_size": 2,
            "config": lab_config,
            "device": "cpu",
            "generator_state": (),
            "history": [],
            "loss_weights": {},
            "lr": 0.001,
            "model": source_state,
            "optimizer": {"ignored": torch.tensor([99.0])},
            "schema": "immer-contextual-seed-run/v3",
            "seed": 7,
            "steps": 1,
            "torch_rng_state": torch.get_rng_state(),
            "train_dialects": (0,),
            "trainer_steps": 1,
        }
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "lab.pt"
            torch.save(payload, source)
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            publication = migrate_seed_v3_lab_checkpoint(
                source,
                Path(temporary) / "migrated",
                source_sha256=source_sha,
                code_revision_sha256=_hash("migration-code"),
                tokenizer_manifest_sha256=_hash("migration-tokenizer"),
                feature_schema_sha256=_hash("migration-features"),
                receipt_feature_dim=5,
                route_count=3,
            )
            restored, manifest = load_seed_v3_checkpoint(
                publication.manifest_path,
                expected_feature_schema_sha256=_hash("migration-features"),
            )
            torch.testing.assert_close(
                restored.token_embedding.weight,
                source_model.token_embedding.weight,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                restored.quotient_head.weight,
                restored.predictive_head.weight,
                rtol=0,
                atol=0,
            )
            self.assertEqual(manifest.parent_checkpoint_sha256, source_sha)
            self.assertNotIn("optimizer", manifest.to_bytes().decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
