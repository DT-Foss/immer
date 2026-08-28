"""Content-addressed variable-row masks for sealed Qwen prompt cohorts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence, cast

from ..qwen3_8.cartography_probe import prompt_token_sha256
from .identity import canonical_json_bytes, require_sha256


PROMPT_ROW_ROLE_SCHEMA = "immer.qwen3.8-prompt-row-role/v1"
PROMPT_ROW_MANIFEST_SCHEMA = "immer.qwen3.8-prompt-row-manifest/v1"


class PromptRowRoleError(ValueError):
    pass


class PromptRowRoleIntegrityError(PromptRowRoleError):
    pass


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class PromptRowRole:
    prompt_sha256: str
    token_count: int
    common_prefix_tokens: int
    common_suffix_tokens: int
    content_row_indices: tuple[int, ...]
    token_sequence_sha256: str

    def __post_init__(self) -> None:
        for field in ("prompt_sha256", "token_sequence_sha256"):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        for field in (
            "token_count",
            "common_prefix_tokens",
            "common_suffix_tokens",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be non-negative")
        if self.token_count < 1:
            raise ValueError("token_count must be positive")
        indices = tuple(self.content_row_indices)
        expected = tuple(
            range(
                self.common_prefix_tokens,
                self.token_count - self.common_suffix_tokens,
            )
        )
        if (
            not indices
            or indices != expected
            or self.common_prefix_tokens + self.common_suffix_tokens >= self.token_count
        ):
            raise ValueError("content row interval is invalid")
        object.__setattr__(self, "content_row_indices", indices)

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "common_prefix_tokens": self.common_prefix_tokens,
            "common_suffix_tokens": self.common_suffix_tokens,
            "content_row_indices": list(self.content_row_indices),
            "prompt_sha256": self.prompt_sha256,
            "schema": PROMPT_ROW_ROLE_SCHEMA,
            "token_count": self.token_count,
            "token_sequence_sha256": self.token_sequence_sha256,
        }

    @classmethod
    def from_record(cls, value: object) -> "PromptRowRole":
        if (
            not isinstance(value, Mapping)
            or set(value)
            != {
                "common_prefix_tokens",
                "common_suffix_tokens",
                "content_row_indices",
                "prompt_sha256",
                "schema",
                "token_count",
                "token_sequence_sha256",
            }
            or value.get("schema") != PROMPT_ROW_ROLE_SCHEMA
            or not isinstance(value.get("content_row_indices"), list)
        ):
            raise PromptRowRoleIntegrityError("prompt row role is invalid")
        try:
            return cls(
                prompt_sha256=cast(str, value["prompt_sha256"]),
                token_count=cast(int, value["token_count"]),
                common_prefix_tokens=cast(int, value["common_prefix_tokens"]),
                common_suffix_tokens=cast(int, value["common_suffix_tokens"]),
                content_row_indices=tuple(
                    cast(list[int], value["content_row_indices"])
                ),
                token_sequence_sha256=cast(str, value["token_sequence_sha256"]),
            )
        except (TypeError, ValueError) as exc:
            raise PromptRowRoleIntegrityError(
                "prompt row role validation failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class PromptRowRoleManifest:
    prompt_registry_sha256: str
    common_prefix_sha256: str
    common_suffix_sha256: str
    roles: tuple[PromptRowRole, ...]

    def __post_init__(self) -> None:
        for field in (
            "prompt_registry_sha256",
            "common_prefix_sha256",
            "common_suffix_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        roles = tuple(self.roles)
        if (
            len(roles) < 2
            or any(not isinstance(role, PromptRowRole) for role in roles)
            or tuple(role.prompt_sha256 for role in roles)
            != tuple(sorted(role.prompt_sha256 for role in roles))
            or len({role.prompt_sha256 for role in roles}) != len(roles)
        ):
            raise ValueError("prompt row role inventory is invalid")
        if (
            len({role.common_prefix_tokens for role in roles}) != 1
            or len({role.common_suffix_tokens for role in roles}) != 1
        ):
            raise ValueError("prompt row roles disagree on common boundaries")
        object.__setattr__(self, "roles", roles)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @property
    def row_indices_by_prompt(self) -> dict[str, tuple[int, ...]]:
        return {role.prompt_sha256: role.content_row_indices for role in self.roles}

    def to_document(self) -> dict[str, object]:
        body = {
            "common_prefix_sha256": self.common_prefix_sha256,
            "common_suffix_sha256": self.common_suffix_sha256,
            "prompt_registry_sha256": self.prompt_registry_sha256,
            "roles": [role.to_record() for role in self.roles],
        }
        return {
            "body": body,
            "body_sha256": _digest(body),
            "schema": PROMPT_ROW_MANIFEST_SCHEMA,
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_document())

    @classmethod
    def from_bytes(cls, data: bytes) -> "PromptRowRoleManifest":
        try:
            value = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PromptRowRoleIntegrityError("row manifest is not JSON") from exc
        if (
            not isinstance(value, Mapping)
            or set(value) != {"body", "body_sha256", "schema"}
            or value.get("schema") != PROMPT_ROW_MANIFEST_SCHEMA
            or not isinstance(value.get("body"), Mapping)
            or value.get("body_sha256") != _digest(value.get("body"))
            or canonical_json_bytes(value) != data
        ):
            raise PromptRowRoleIntegrityError("row manifest seal is invalid")
        body = cast(Mapping[str, object], value["body"])
        if set(body) != {
            "common_prefix_sha256",
            "common_suffix_sha256",
            "prompt_registry_sha256",
            "roles",
        } or not isinstance(body.get("roles"), list):
            raise PromptRowRoleIntegrityError("row manifest body is invalid")
        result = cls(
            prompt_registry_sha256=cast(str, body["prompt_registry_sha256"]),
            common_prefix_sha256=cast(str, body["common_prefix_sha256"]),
            common_suffix_sha256=cast(str, body["common_suffix_sha256"]),
            roles=tuple(
                PromptRowRole.from_record(row)
                for row in cast(list[object], body["roles"])
            ),
        )
        if result.to_bytes() != data:
            raise PromptRowRoleIntegrityError("row manifest bytes changed")
        return result


def derive_prompt_row_roles(
    prompt_token_ids: Mapping[str, Sequence[int]],
) -> PromptRowRoleManifest:
    if not isinstance(prompt_token_ids, Mapping) or len(prompt_token_ids) < 2:
        raise ValueError("prompt_token_ids must contain at least two prompts")
    normalized = {}
    for prompt_sha256, values in prompt_token_ids.items():
        prompt = require_sha256(prompt_sha256, field="prompt_sha256")
        if isinstance(values, (str, bytes, bytearray)):
            raise TypeError("prompt token IDs must be an integer sequence")
        tokens = tuple(values)
        if (
            not tokens
            or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in tokens
            )
            or prompt_token_sha256(tokens) != prompt
        ):
            raise PromptRowRoleIntegrityError(
                "prompt tokens differ from their prompt SHA"
            )
        normalized[prompt] = tokens
    sequences = tuple(normalized[prompt] for prompt in sorted(normalized))
    prefix = 0
    while (
        all(prefix < len(tokens) for tokens in sequences)
        and len({tokens[prefix] for tokens in sequences}) == 1
    ):
        prefix += 1
    suffix = 0
    while (
        all(suffix < len(tokens) - prefix for tokens in sequences)
        and len({tokens[len(tokens) - 1 - suffix] for tokens in sequences}) == 1
    ):
        suffix += 1
    if prefix == 0 or suffix == 0:
        raise PromptRowRoleIntegrityError(
            "prompt cohort has no shared chat-template prefix/suffix"
        )
    common_prefix = sequences[0][:prefix]
    common_suffix = sequences[0][len(sequences[0]) - suffix :]
    registry = {
        "prompts": [
            {"sha256": prompt, "token_ids": list(normalized[prompt])}
            for prompt in sorted(normalized)
        ],
        "schema": "immer.qwen3.8-row-role-input-registry/v1",
    }
    return PromptRowRoleManifest(
        prompt_registry_sha256=_digest(registry),
        common_prefix_sha256=_digest(
            {
                "schema": "immer.qwen3.8-common-prefix/v1",
                "token_ids": list(common_prefix),
            }
        ),
        common_suffix_sha256=_digest(
            {
                "schema": "immer.qwen3.8-common-suffix/v1",
                "token_ids": list(common_suffix),
            }
        ),
        roles=tuple(
            PromptRowRole(
                prompt_sha256=prompt,
                token_count=len(tokens),
                common_prefix_tokens=prefix,
                common_suffix_tokens=suffix,
                content_row_indices=tuple(range(prefix, len(tokens) - suffix)),
                token_sequence_sha256=_digest(
                    {
                        "schema": "immer.qwen3.8-token-sequence/v1",
                        "token_ids": list(tokens),
                    }
                ),
            )
            for prompt, tokens in sorted(normalized.items())
        ),
    )


__all__ = [
    "PROMPT_ROW_MANIFEST_SCHEMA",
    "PROMPT_ROW_ROLE_SCHEMA",
    "PromptRowRole",
    "PromptRowRoleError",
    "PromptRowRoleIntegrityError",
    "PromptRowRoleManifest",
    "derive_prompt_row_roles",
]
