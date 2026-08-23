"""Canonical persistence for trained DeepSeek-V4 route predictors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .causal_prefetch import CheckpointIdentity
from .route_markov import LayerMarkovExpertPredictor, RouteMarkovError

ROUTE_MODEL_SCHEMA = "immer.deepseek-v4-route-model/v1"
_ROLES = frozenset(("real_markov", "placebo_markov"))


class RouteModelArtifactError(ValueError):
    """A persisted route model is malformed or belongs to another checkpoint."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RouteModelArtifactError("route model is not canonical JSON") from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise RouteModelArtifactError("route model metadata must be a JSON object")
    try:
        normalized = json.loads(_canonical_json(dict(value)).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:  # pragma: no cover
        raise RouteModelArtifactError("route model metadata is invalid") from exc
    if not isinstance(normalized, dict):  # pragma: no cover
        raise RouteModelArtifactError("route model metadata must be a JSON object")
    return normalized


@dataclass(frozen=True, slots=True)
class RouteModelArtifact:
    checkpoint: CheckpointIdentity
    role: str
    predictor: LayerMarkovExpertPredictor
    metadata: dict[str, Any]
    sha256: str


def build_route_model_artifact(
    predictor: LayerMarkovExpertPredictor,
    *,
    checkpoint: CheckpointIdentity,
    role: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a digest-bound artifact from one trained predictor."""

    if not isinstance(predictor, LayerMarkovExpertPredictor):
        raise RouteModelArtifactError("predictor must be a LayerMarkovExpertPredictor")
    if not isinstance(checkpoint, CheckpointIdentity):
        raise RouteModelArtifactError("checkpoint must be a CheckpointIdentity")
    if role not in _ROLES:
        raise RouteModelArtifactError("route model role is unsupported")
    snapshot = predictor.snapshot()
    identity = {
        "checkpoint": checkpoint.as_record(),
        "metadata": _metadata(metadata),
        "role": role,
        "schema": ROUTE_MODEL_SCHEMA,
        "snapshot": snapshot,
        "snapshot_sha256": predictor.snapshot_sha256,
    }
    return {**identity, "sha256": _sha256(identity)}


def validate_route_model_artifact(
    document: object,
    *,
    expected_checkpoint: CheckpointIdentity | None = None,
    expected_role: str | None = None,
) -> RouteModelArtifact:
    """Validate every field and reconstruct the immutable predictor."""

    if not isinstance(document, dict) or set(document) != {
        "checkpoint",
        "metadata",
        "role",
        "schema",
        "sha256",
        "snapshot",
        "snapshot_sha256",
    }:
        raise RouteModelArtifactError("route model has unknown or missing fields")
    if document.get("schema") != ROUTE_MODEL_SCHEMA:
        raise RouteModelArtifactError("route model schema is unsupported")
    role = document.get("role")
    if role not in _ROLES or (expected_role is not None and role != expected_role):
        raise RouteModelArtifactError("route model role does not match")
    raw_checkpoint = document.get("checkpoint")
    if not isinstance(raw_checkpoint, dict) or set(raw_checkpoint) != {
        "inventory_fingerprint",
        "repo_id",
        "revision",
    }:
        raise RouteModelArtifactError("route model checkpoint is invalid")
    try:
        checkpoint = CheckpointIdentity(
            repo_id=raw_checkpoint["repo_id"],
            revision=raw_checkpoint["revision"],
            inventory_fingerprint=raw_checkpoint["inventory_fingerprint"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RouteModelArtifactError("route model checkpoint is invalid") from exc
    if checkpoint.as_record() != raw_checkpoint:
        raise RouteModelArtifactError("route model checkpoint is not canonical")
    if expected_checkpoint is not None and checkpoint != expected_checkpoint:
        raise RouteModelArtifactError("route model belongs to another checkpoint")
    try:
        predictor = LayerMarkovExpertPredictor.from_snapshot(document.get("snapshot"))
    except RouteMarkovError as exc:
        raise RouteModelArtifactError(str(exc)) from exc
    if document.get("snapshot_sha256") != predictor.snapshot_sha256:
        raise RouteModelArtifactError("route model snapshot digest does not match")
    normalized_metadata = _metadata(document.get("metadata"))
    identity = {key: value for key, value in document.items() if key != "sha256"}
    digest = document.get("sha256")
    if not isinstance(digest, str) or digest != _sha256(identity):
        raise RouteModelArtifactError("route model artifact digest does not match")
    if _canonical_json(document) != _canonical_json(dict(document)):
        raise RouteModelArtifactError("route model artifact is not canonical")
    return RouteModelArtifact(
        checkpoint=checkpoint,
        role=role,
        predictor=predictor,
        metadata=normalized_metadata,
        sha256=digest,
    )


def write_route_model_artifact(
    path: str | os.PathLike[str], document: Mapping[str, Any]
) -> None:
    """Atomically write one already validated route model artifact."""

    validate_route_model_artifact(dict(document))
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json(dict(document)))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_route_model_artifact(
    path: str | os.PathLike[str],
    *,
    expected_checkpoint: CheckpointIdentity | None = None,
    expected_role: str | None = None,
) -> RouteModelArtifact:
    """Read canonical JSON without duplicate keys, then validate it."""

    source = Path(path).expanduser().resolve()

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RouteModelArtifactError(
                    f"duplicate JSON key in route model: {key!r}"
                )
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise RouteModelArtifactError(f"non-finite route model value: {value}")

    encoded = source.read_bytes()
    try:
        document = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=invalid_constant,
        )
    except RouteModelArtifactError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RouteModelArtifactError("cannot parse route model artifact") from exc
    if encoded != _canonical_json(document):
        raise RouteModelArtifactError("route model artifact is not canonical JSON")
    return validate_route_model_artifact(
        document,
        expected_checkpoint=expected_checkpoint,
        expected_role=expected_role,
    )


__all__ = [
    "ROUTE_MODEL_SCHEMA",
    "RouteModelArtifact",
    "RouteModelArtifactError",
    "build_route_model_artifact",
    "load_route_model_artifact",
    "validate_route_model_artifact",
    "write_route_model_artifact",
]
