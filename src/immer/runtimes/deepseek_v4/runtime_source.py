"""Owned DeepSeek-V4 tensor sources with optional native causal addressing.

The causal path mounts weights in place.  It never copies or materializes a
checkpoint: ``CausalWeightMount`` exposes the bundle's local safetensors files
through the same ``Streamer`` transport and supplies separate authenticated
dense-tensor and sparse-expert readers consumed by ``DeepSeekWeightPager``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from immer.knowledge import Streamer

from .causal_weights import (
    CausalTensorReader,
    CausalWeightMount,
    CausalWeightReader,
    LogicalModelIdentity,
)


_PINNED_REVISION = re.compile(r"[0-9a-fA-F]{40,64}")
_ABSOLUTE_PATH_IN_TEXT = re.compile(
    r"(?<![:/])/(?:[^/\s'\"]+/)*[^/\s'\":,\)\]]+"
)
_PINNED_INVENTORY_SCHEMA = "immer.tensor-inventory-cache/v1"
GENERAL_DENSE_COVERAGE_CAPABILITY = "deepseek-v4-general-dense-coverage/v1"
_TRACE_SPARSE_BUNDLE_SCHEMA = "immer.deepseek-v4-sparse-causal-bundle/v1"


class DeepSeekRuntimeSourceError(RuntimeError):
    """A requested DeepSeek-V4 source cannot be opened reproducibly."""


def shareable_runtime_evidence(value: Any) -> Any:
    """Remove machine-local absolute paths from persisted runtime evidence."""

    if isinstance(value, Mapping):
        return {str(key): shareable_runtime_evidence(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(shareable_runtime_evidence(item) for item in value)
    if isinstance(value, list):
        return [shareable_runtime_evidence(item) for item in value]
    if isinstance(value, str):
        if value.startswith("local:"):
            return "local:<external>"
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            name = candidate.name or "artifact"
            return f"<external:{name}>"
        return _ABSOLUTE_PATH_IN_TEXT.sub("<external-path>", value)
    return value


@dataclass(slots=True)
class DeepSeekRuntimeSource:
    """Own one ordinary streamer or one causal bundle mount."""

    source: Streamer
    label: str
    mount: CausalWeightMount | None = None
    logical_model: LogicalModelIdentity | None = None
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def causal_weight_reader(self) -> CausalWeightReader | None:
        """Return the strict sparse-route reader when a bundle is mounted."""

        return None if self.mount is None else self.mount.reader

    @property
    def causal_tensor_reader(self) -> CausalTensorReader | None:
        """Return the strict dense tensor reader for a general causal bundle."""

        return None if self.mount is None else self.mount.tensor_reader

    @property
    def is_causal_bundle(self) -> bool:
        return self.mount is not None

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self.mount is not None:
                self.mount.close()
            else:
                close = getattr(self.source, "close", None)
                if callable(close):
                    close()
        finally:
            self._closed = True

    def __enter__(self) -> DeepSeekRuntimeSource:
        if self._closed:
            raise DeepSeekRuntimeSourceError("DeepSeek runtime source is closed")
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


def _local_source_path(value: str) -> Path | None:
    raw = value.removeprefix("local:") if value.startswith("local:") else value
    candidate = Path(raw).expanduser()
    explicit = (
        value.startswith("local:")
        or candidate.is_absolute()
        or raw.startswith(("./", "../"))
    )
    if explicit:
        return candidate.resolve()
    return candidate.resolve() if candidate.is_dir() else None


def _strict_json(path: Path, *, label: str) -> Any:
    def pairs(entries: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in entries:
            if key in result:
                raise DeepSeekRuntimeSourceError(
                    f"duplicate JSON key in {label}: {key!r}"
                )
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise DeepSeekRuntimeSourceError(
            f"non-finite JSON value in {label}: {value}"
        )

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=invalid_constant,
        )
    except DeepSeekRuntimeSourceError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DeepSeekRuntimeSourceError(
            f"cannot read {label}: {path}"
        ) from exc


def _pinned_inventory(
    root: Path,
) -> tuple[Mapping[str, Any] | None, str | None]:
    path = root / "inventory.pinned.json"
    if not path.exists() and not path.is_symlink():
        return None, None
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise DeepSeekRuntimeSourceError(
            f"cannot inspect pinned inventory: {path}"
        ) from exc
    if path.is_symlink() or not path.is_file() or not os.path.samestat(
        metadata, path.stat()
    ):
        raise DeepSeekRuntimeSourceError(
            f"pinned inventory must be an unchanged regular file: {path}"
        )
    document = _strict_json(path, label="pinned inventory")
    if (
        not isinstance(document, Mapping)
        or document.get("schema") != _PINNED_INVENTORY_SCHEMA
        or not isinstance(document.get("inventory"), Mapping)
        or not isinstance(document.get("source_fingerprint"), str)
    ):
        raise DeepSeekRuntimeSourceError("local pinned inventory schema is invalid")
    return document["inventory"], document["source_fingerprint"]


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DeepSeekRuntimeSourceError(
            "causal bundle capability body is not canonical JSON"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _require_general_causal_bundle(root: Path) -> None:
    manifest = root / "bundle.json"
    try:
        metadata = manifest.lstat()
    except OSError as exc:
        raise DeepSeekRuntimeSourceError(
            "generic DeepSeek runner requires an authenticated bundle manifest"
        ) from exc
    if manifest.is_symlink() or not manifest.is_file() or not os.path.samestat(
        metadata, manifest.stat()
    ):
        raise DeepSeekRuntimeSourceError(
            "causal bundle manifest must be an unchanged regular file"
        )
    document = _strict_json(manifest, label="causal bundle manifest")
    if (
        isinstance(document, Mapping)
        and document.get("schema") == _TRACE_SPARSE_BUNDLE_SCHEMA
    ):
        raise DeepSeekRuntimeSourceError(
            "trace-sparse DeepSeek bundle is decode-trace-specific and cannot be "
            "used by a generic runner"
        )
    if not isinstance(document, Mapping) or set(document) != {
        "body",
        "schema",
        "sha256",
    }:
        raise DeepSeekRuntimeSourceError(
            "generic DeepSeek bundle lacks an authenticated capability envelope"
        )
    body = document.get("body")
    if not isinstance(body, Mapping) or document.get("sha256") != _canonical_sha256(
        body
    ):
        raise DeepSeekRuntimeSourceError(
            "causal bundle capability envelope digest does not match"
        )
    capabilities = body.get("capabilities")
    if (
        not isinstance(capabilities, Mapping)
        or capabilities.get("general_dense_weight_coverage")
        != GENERAL_DENSE_COVERAGE_CAPABILITY
    ):
        raise DeepSeekRuntimeSourceError(
            "generic DeepSeek bundle lacks authenticated complete dense-weight "
            "coverage"
        )


def _require_pinned_revision(revision: str, *, target: str) -> None:
    if _PINNED_REVISION.fullmatch(revision) is None:
        raise DeepSeekRuntimeSourceError(
            f"{target} requires an immutable 40-64 character revision digest"
        )


def open_deepseek_runtime_source(
    *,
    source: str,
    revision: str,
    logical_repo_id: str,
    causal_bundle: str | os.PathLike[str] | None,
    budget_mb: float,
    cache_dir: str | os.PathLike[str],
    use_cache: bool,
    max_cache_bytes: int,
    verbose: bool = False,
    access_observer: Any | None = None,
    require_remote_pinned_revision: bool = False,
) -> DeepSeekRuntimeSource:
    """Open one bounded source without copying checkpoint weights.

    A causal bundle always uses the explicit logical model identity, an
    immutable revision digest, and an authenticated complete dense-weight
    coverage capability.  Current trace-sparse bundles stay restricted to the
    sealed direct-decode consumer because ordinary tensor reads could otherwise
    cross sparse file holes. Both readers are strict by construction; callers
    pass them to ``DeepSeekWeightPager`` without enabling missing-route
    fallback. Ordinary local and remote sources retain Streamer behavior.
    """

    if not isinstance(source, str) or not source.strip():
        raise DeepSeekRuntimeSourceError("source must be a non-empty string")
    if not isinstance(revision, str) or not revision.strip():
        raise DeepSeekRuntimeSourceError("revision must be a non-empty string")
    if not isinstance(logical_repo_id, str) or not logical_repo_id.strip():
        raise DeepSeekRuntimeSourceError(
            "logical_repo_id must be a non-empty repository identity"
        )
    logical_repo_id = logical_repo_id.strip()
    if logical_repo_id.startswith("local:") or Path(
        logical_repo_id
    ).expanduser().is_absolute():
        raise DeepSeekRuntimeSourceError(
            "logical_repo_id must not contain a machine-local path"
        )
    if isinstance(budget_mb, bool) or not isinstance(budget_mb, (int, float)):
        raise DeepSeekRuntimeSourceError("budget_mb must be numeric")
    if budget_mb <= 0:
        raise DeepSeekRuntimeSourceError("budget_mb must be positive")
    if isinstance(max_cache_bytes, bool) or not isinstance(max_cache_bytes, int):
        raise DeepSeekRuntimeSourceError("max_cache_bytes must be an integer")
    if max_cache_bytes < 0:
        raise DeepSeekRuntimeSourceError("max_cache_bytes must be non-negative")

    if causal_bundle is not None:
        _require_pinned_revision(revision, target="causal bundle")
        bundle_root = Path(causal_bundle).expanduser().absolute()
        _require_general_causal_bundle(bundle_root)
        model = LogicalModelIdentity(
            repo_id=logical_repo_id,
            revision=revision,
        )
        try:
            mount = CausalWeightMount(
                bundle_root,
                model,
                budget_mb=float(budget_mb),
                verbose=verbose,
            )
        except Exception as exc:
            raise DeepSeekRuntimeSourceError(
                "cannot mount local DeepSeek-V4 causal bundle"
            ) from exc
        try:
            if access_observer is not None:
                mount.source.set_access_observer(access_observer)
            return DeepSeekRuntimeSource(
                source=mount.source,
                label=(
                    f"causal-bundle:{model.repo_id}@{model.revision[:12]}"
                ),
                mount=mount,
                logical_model=model,
            )
        except BaseException:
            mount.close()
            raise

    common = {
        "revision": revision,
        "budget_mb": float(budget_mb),
        "cache_dir": Path(cache_dir).expanduser().resolve(),
        "use_cache": use_cache,
        "max_cache_bytes": max_cache_bytes,
        "verbose": verbose,
        "access_observer": access_observer,
    }
    local = _local_source_path(source)
    if local is not None:
        if not local.is_dir():
            raise FileNotFoundError(
                f"local source directory does not exist: {local}"
            )
        pinned_inventory, pinned_fingerprint = _pinned_inventory(local)
        streamer = Streamer.from_local(
            local,
            pinned_inventory=pinned_inventory,
            pinned_fingerprint=pinned_fingerprint,
            **common,
        )
        return DeepSeekRuntimeSource(streamer, "local:<external>")

    if require_remote_pinned_revision:
        _require_pinned_revision(revision, target="remote source")
    return DeepSeekRuntimeSource(Streamer(source, **common), source)


__all__ = [
    "DeepSeekRuntimeSource",
    "DeepSeekRuntimeSourceError",
    "GENERAL_DENSE_COVERAGE_CAPABILITY",
    "open_deepseek_runtime_source",
    "shareable_runtime_evidence",
]
