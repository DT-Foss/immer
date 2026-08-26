from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def require_sha256(value: str, *, field: str) -> str:
    """Return a canonical SHA-256 digest or reject the identity."""

    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a JSON value into the single representation used for identities."""

    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonical JSON") from exc
    return encoded.encode("utf-8")


@dataclass(frozen=True, slots=True)
class OoeSiteIdentity:
    """Complete identity of one executable OoE weight site.

    A site is not just a human-readable coordinate.  It is a coordinate inside
    one pinned model, interpreted by one feature/action contract at one exact
    causal graph revision.  Changing any component creates another identity.
    """

    model_pin_sha256: str
    weight_coordinate_sha256: str
    graph_revision_sha256: str
    feature_schema_sha256: str
    action_schema_sha256: str

    FORMAT = "immer-ooe-site-identity/v1"

    def __post_init__(self) -> None:
        for field in (
            "model_pin_sha256",
            "weight_coordinate_sha256",
            "graph_revision_sha256",
            "feature_schema_sha256",
            "action_schema_sha256",
        ):
            object.__setattr__(
                self,
                field,
                require_sha256(getattr(self, field), field=field),
            )

    def to_dict(self) -> dict[str, str]:
        return {"format": self.FORMAT, **asdict(self)}

    @classmethod
    def from_dict(cls, value: object) -> "OoeSiteIdentity":
        if not isinstance(value, dict) or value.get("format") != cls.FORMAT:
            raise ValueError("unsupported OoE site identity")
        expected = {
            "format",
            "model_pin_sha256",
            "weight_coordinate_sha256",
            "graph_revision_sha256",
            "feature_schema_sha256",
            "action_schema_sha256",
        }
        if set(value) != expected:
            raise ValueError("OoE site identity has unknown or missing fields")
        return cls(
            model_pin_sha256=value["model_pin_sha256"],
            weight_coordinate_sha256=value["weight_coordinate_sha256"],
            graph_revision_sha256=value["graph_revision_sha256"],
            feature_schema_sha256=value["feature_schema_sha256"],
            action_schema_sha256=value["action_schema_sha256"],
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    def assert_same_execution_contract(self, other: "OoeSiteIdentity") -> None:
        if not isinstance(other, OoeSiteIdentity) or self != other:
            raise ValueError("OoE site execution-contract identity mismatch")
