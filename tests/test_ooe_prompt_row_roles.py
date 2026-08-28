from __future__ import annotations

import json
import unittest

from immer.runtimes.ooe.prompt_row_roles import (
    PromptRowRoleIntegrityError,
    PromptRowRoleManifest,
    derive_prompt_row_roles,
)
from immer.runtimes.qwen3_8 import prompt_token_sha256


class PromptRowRoleTests(unittest.TestCase):
    def test_common_template_is_removed_and_manifest_roundtrips(self) -> None:
        rows = {
            prompt_token_sha256((1, 2, 10, 11, 8, 9)): (1, 2, 10, 11, 8, 9),
            prompt_token_sha256((1, 2, 20, 21, 22, 8, 9)): (
                1,
                2,
                20,
                21,
                22,
                8,
                9,
            ),
            prompt_token_sha256((1, 2, 30, 8, 9)): (1, 2, 30, 8, 9),
        }
        manifest = derive_prompt_row_roles(rows)
        self.assertEqual({role.common_prefix_tokens for role in manifest.roles}, {2})
        self.assertEqual({role.common_suffix_tokens for role in manifest.roles}, {2})
        self.assertEqual(
            {len(role.content_row_indices) for role in manifest.roles}, {1, 2, 3}
        )
        self.assertEqual(
            PromptRowRoleManifest.from_bytes(manifest.to_bytes()).to_bytes(),
            manifest.to_bytes(),
        )

    def test_prompt_hash_and_resealed_role_tamper_fail(self) -> None:
        rows = {
            prompt_token_sha256((1, 2, 3, 9)): (1, 2, 3, 9),
            prompt_token_sha256((1, 4, 5, 9)): (1, 4, 5, 9),
        }
        with self.assertRaises(PromptRowRoleIntegrityError):
            derive_prompt_row_roles(
                {next(iter(rows)): (1, 2, 99, 9), **dict(list(rows.items())[1:])}
            )
        manifest = derive_prompt_row_roles(rows)
        document = json.loads(manifest.to_bytes())
        document["body"]["roles"][0]["content_row_indices"] = [0]
        document["body_sha256"] = _digest(document["body"])
        with self.assertRaises((ValueError, PromptRowRoleIntegrityError)):
            PromptRowRoleManifest.from_bytes(_canonical(document))


def _canonical(value: object) -> bytes:
    from immer.runtimes.ooe.identity import canonical_json_bytes

    return canonical_json_bytes(value)


def _digest(value: object) -> str:
    import hashlib

    return hashlib.sha256(_canonical(value)).hexdigest()


if __name__ == "__main__":
    unittest.main()
