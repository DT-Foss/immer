from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from immer.runtimes.ooe.crystal import CrystalStore
from immer.runtimes.ooe.identity import canonical_json_bytes


_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import ooe_controller_action_frontier as frontier_cli  # noqa: E402
import ooe_fertig_action_learning as fertig_cli  # noqa: E402


class ActionCliIntegrityTests(unittest.TestCase):
    def test_physical_alias_and_output_inside_store_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real_parent = root / "real-parent"
            real_parent.mkdir()
            store = CrystalStore(real_parent / "store")
            alias_parent = root / "alias-parent"
            alias_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(
                fertig_cli.FertigActionLearningCliError,
                "physically disjoint",
            ):
                fertig_cli._assert_disjoint_store_roots(
                    store.root,
                    alias_parent / "store",
                )
            with self.assertRaisesRegex(
                fertig_cli.FertigActionLearningCliError,
                "inside a managed",
            ):
                fertig_cli._assert_output_outside_stores(
                    store.root / "report.json",
                    store.root,
                )
            nested = store.root / "new-parent" / "target"
            with self.assertRaisesRegex(
                fertig_cli.FertigActionLearningCliError,
                "before directory creation",
            ):
                fertig_cli._assert_planned_store_roots(store.root, nested)
            self.assertFalse(nested.parent.exists())

    def test_exact_fork_provenance_cannot_rebind_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = CrystalStore(root / "source")
            source.publish_state("controller", b"source-state")
            target_root = root / "target"
            target, provenance, present = fertig_cli._open_or_create_controller_fork(
                source,
                target_root,
                state_name="controller",
            )
            self.assertTrue(present)
            self.assertEqual(target.restore_state("controller"), b"source-state")
            other = CrystalStore(root / "other")
            other.publish_state("controller", b"other-state")
            rebound = fertig_cli._fork_provenance(other, state_name="controller")
            self.assertNotEqual(provenance, rebound)
            with self.assertRaisesRegex(
                fertig_cli.FertigActionLearningCliError,
                "another source",
            ):
                fertig_cli._commit_fork_provenance(
                    target,
                    rebound,
                    state_name="controller",
                )

    def test_promotion_lineage_ignores_only_promotion_fields(self) -> None:
        body = {
            "crystal_manifest": {"generation": 0},
            "crystal_manifest_generation": 0,
            "crystal_manifest_sha256": "a" * 64,
            "history_binding": "b" * 64,
            "metrics": {"promotions": 0, "teacher_calls": 7},
            "router": {
                "calibration_sha256": None,
                "clusters": [{"label": "site"}],
                "min_margin": 0.02,
                "radius": 0.75,
            },
            "sites": [
                {
                    "crystal_sha256": None,
                    "history": ["c" * 64],
                    "site_identity_sha256": "d" * 64,
                }
            ],
        }

        def snapshot(value):
            return canonical_json_bytes(
                {"body": value, "schema": "test", "sha256": "e" * 64}
            )

        promoted = json.loads(canonical_json_bytes(body))
        promoted["crystal_manifest"] = {"generation": 8}
        promoted["crystal_manifest_generation"] = 8
        promoted["crystal_manifest_sha256"] = "f" * 64
        promoted["metrics"]["promotions"] = 8
        promoted["router"]["calibration_sha256"] = "1" * 64
        promoted["router"]["radius"] = 0.3
        promoted["router"]["min_margin"] = 0.1
        promoted["sites"][0]["crystal_sha256"] = "2" * 64
        self.assertEqual(
            frontier_cli._promotion_lineage_sha256(snapshot(body)),
            frontier_cli._promotion_lineage_sha256(snapshot(promoted)),
        )
        promoted["sites"][0]["history"] = ["3" * 64]
        self.assertNotEqual(
            frontier_cli._promotion_lineage_sha256(snapshot(body)),
            frontier_cli._promotion_lineage_sha256(snapshot(promoted)),
        )


if __name__ == "__main__":
    unittest.main()
