"""Offline publication and authenticated mounting of multi-layer MLP O1 banks.

The passive O1 collector deliberately never mounts what it learns.  This module
is the model-free promotion boundary: it reads sealed sufficient-statistics
states, publishes one immutable content-addressed bank per ready layer, and
commits one compact registry manifest last.  The manifest contains only pins,
provenance, bounded scalar metadata, and relative content-addressed basenames.

Layer 63 retains its deployed v1 identity, crystal, and bank classes.  Every
other layer uses the layer-parametric v2 lane.  Loading is fail-closed: the
registry envelope, every referenced file, every bank/crystal identity, and the
full O1 provenance chain are verified before any bank is returned to a caller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import tempfile
from typing import TypeAlias

from .layer_mlp_crystal import (
    LAYER_MLP_RESIDUAL_ACTION_ABI,
    LAYER_MLP_RESIDUAL_GENERIC_ACTION_ABI,
    Layer63MlpResidualCrystal,
    Layer63MlpResidualCrystalBank,
    Layer63MlpResidualCrystalIdentity,
    LayerMlpCrystalError,
    LayerMlpCrystalIntegrityError,
    LayerMlpResidualCrystal,
    LayerMlpResidualCrystalBank,
    LayerMlpResidualCrystalIdentity,
    _publish_bytes,
    _read_state,
    _state_lock,
)
from .layer_mlp_o1 import Layer63MlpO1Accumulator, LayerMlpO1Accumulator
from .layer_transition_crystal import (
    LayerTransitionCrystalIntegrityError,
    LayerTransitionProjectionIdentity,
    _digest,
    _finite_non_negative,
    _sealed_document,
    _sha256_document,
    _unseal_document,
)


LAYER_MLP_O1_REGISTRY_SCHEMA = "immer.qwen3.8-layer-mlp-o1-mount-registry/v1"
LAYER_MLP_O1_REGISTRY_ENVELOPE_SCHEMA = (
    "immer.qwen3.8-layer-mlp-o1-mount-registry-envelope/v1"
)
LAYER_MLP_O1_PUBLISH_SUMMARY_SCHEMA = (
    "immer.qwen3.8-layer-mlp-o1-publish-summary/v1"
)
DEFAULT_LAYER_MLP_O1_REGISTRY_BASENAME = "layer-mlp-o1-registry.json"

_MAX_LAYER_INDEX = 63
_MAX_REGISTRY_BYTES = 2 * 1024 * 1024
_BANK_NAME_PREFIX = "qwen-layer"


class LayerMlpO1RegistryError(LayerMlpCrystalError):
    """An O1 registry cannot be published or mounted safely."""


class LayerMlpO1RegistryIntegrityError(
    LayerMlpO1RegistryError,
    LayerMlpCrystalIntegrityError,
):
    """An O1 registry or one of its immutable banks is corrupt."""


class LayerMlpO1RegistryNotMountableError(LayerMlpO1RegistryError):
    """A valid registry has no ready banks to mount."""


_Identity: TypeAlias = (
    Layer63MlpResidualCrystalIdentity | LayerMlpResidualCrystalIdentity
)
_Crystal: TypeAlias = Layer63MlpResidualCrystal | LayerMlpResidualCrystal
_Bank: TypeAlias = Layer63MlpResidualCrystalBank | LayerMlpResidualCrystalBank
_Accumulator: TypeAlias = Layer63MlpO1Accumulator | LayerMlpO1Accumulator


def _integrity(message: str, exc: Exception | None = None) -> None:
    error = LayerMlpO1RegistryIntegrityError(message)
    if exc is None:
        raise error
    raise error from exc


def _layer_index(value: object, *, field: str = "layer_index") -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= _MAX_LAYER_INDEX
    ):
        raise ValueError(f"{field} must be an integer in [0, 63]")
    return value


def _positive_uint(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _basename(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ValueError(f"{field} must be a non-empty relative basename")
    if (
        Path(value).is_absolute()
        or Path(value).name != value
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"{field} must be a path-safe relative basename")
    return value


def _normalize_layers(layers: Sequence[int] | None) -> tuple[int, ...]:
    if layers is None:
        return tuple(range(_MAX_LAYER_INDEX + 1))
    if isinstance(layers, (str, bytes, bytearray)) or not isinstance(
        layers, Sequence
    ):
        raise TypeError("layers must be a sequence of decoder-layer integers")
    result = tuple(layers)
    if not result:
        raise ValueError("layers must not be empty")
    for layer in result:
        _layer_index(layer, field="layers item")
    if result != tuple(sorted(set(result))):
        raise ValueError("layers must be sorted unique integers in [0, 63]")
    return result


def _regular_directory(path: Path, *, create: bool) -> None:
    if create:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise LayerMlpO1RegistryError(
                "O1 registry output root cannot be created"
            ) from exc
    try:
        linked = os.lstat(path)
    except OSError as exc:
        raise LayerMlpO1RegistryError("O1 registry root is unavailable") from exc
    if stat.S_ISLNK(linked.st_mode) or not stat.S_ISDIR(linked.st_mode):
        raise LayerMlpO1RegistryError(
            "O1 registry root must be a real directory, not a symlink"
        )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_regular(path: Path, *, label: str) -> bytes:
    try:
        raw, _signature = _read_state(path)
    except FileNotFoundError:
        raise
    except (LayerMlpCrystalIntegrityError, LayerTransitionCrystalIntegrityError) as exc:
        _integrity(f"{label} is not a stable regular file", exc)
    return raw


def _fsync_directory(path: Path) -> None:
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0),
        )
        os.fsync(descriptor)
    except OSError as exc:
        _integrity("O1 registry directory cannot be synchronized", exc)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def layer_mlp_o1_state_path_for_layer(
    layer63_state_path: str | os.PathLike[str],
    *,
    layer_index: int,
    sketch_dim: int,
) -> Path:
    """Return the stable collector filename derived from the legacy L63 path."""

    layer = _layer_index(layer_index)
    if (
        isinstance(sketch_dim, bool)
        or not isinstance(sketch_dim, int)
        or not 0 < sketch_dim <= 4096
    ):
        raise ValueError("sketch_dim must be an integer in [1, 4096]")
    path = Path(layer63_state_path).expanduser().absolute()
    if not path.name:
        raise ValueError("layer63_state_path must name a file")
    if layer == _MAX_LAYER_INDEX:
        return path
    return path.parent / f"qwen-layer{layer:02d}-mlp-o1-decode-r{sketch_dim}-v2.json"


# Kept as a small local compatibility surface for callers that previously used
# the adapter-private filename helper.  Importing this module never imports the
# CLI, adapter, tokenizer, or model runtime.
_layer_mlp_o1_state_path_for_layer = layer_mlp_o1_state_path_for_layer


@dataclass(frozen=True, slots=True)
class LayerMlpO1RegistryPins:
    """Global immutable runtime pins shared by every mounted layer bank."""

    model_sha256: str
    q4_sha256: str
    atlas_revision_sha256: str
    graph_revision_sha256: str
    projection: LayerTransitionProjectionIdentity

    def __post_init__(self) -> None:
        for field in (
            "model_sha256",
            "q4_sha256",
            "atlas_revision_sha256",
            "graph_revision_sha256",
        ):
            try:
                _digest(getattr(self, field), field)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"registry {field} is invalid") from exc
        if not isinstance(self.projection, LayerTransitionProjectionIdentity):
            raise TypeError("registry projection pin is invalid")

    @property
    def compute_graph_revision_sha256(self) -> str:
        """Explicit alias for the Compute operator-graph revision pin."""

        return self.graph_revision_sha256

    @classmethod
    def from_identity(cls, identity: _Identity) -> "LayerMlpO1RegistryPins":
        if not isinstance(
            identity,
            (Layer63MlpResidualCrystalIdentity, LayerMlpResidualCrystalIdentity),
        ):
            raise TypeError("identity is not an MLP residual identity")
        return cls(
            model_sha256=identity.model_sha256,
            q4_sha256=identity.q4_sha256,
            atlas_revision_sha256=identity.atlas_revision_sha256,
            graph_revision_sha256=identity.graph_revision_sha256,
            projection=identity.projection,
        )

    def matches_identity(self, identity: _Identity) -> bool:
        return self == type(self).from_identity(identity)

    def to_record(self) -> dict[str, object]:
        return {
            "atlas_revision_sha256": self.atlas_revision_sha256,
            "graph_revision_sha256": self.graph_revision_sha256,
            "model_sha256": self.model_sha256,
            "projection": self.projection.to_record(),
            "q4_sha256": self.q4_sha256,
        }

    @classmethod
    def from_record(cls, value: object) -> "LayerMlpO1RegistryPins":
        fields = {
            "atlas_revision_sha256",
            "graph_revision_sha256",
            "model_sha256",
            "projection",
            "q4_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            _integrity("O1 registry pin fields are invalid")
        try:
            return cls(
                model_sha256=value["model_sha256"],
                q4_sha256=value["q4_sha256"],
                atlas_revision_sha256=value["atlas_revision_sha256"],
                graph_revision_sha256=value["graph_revision_sha256"],
                projection=LayerTransitionProjectionIdentity.from_record(
                    value["projection"]
                ),
            )
        except (TypeError, ValueError, LayerTransitionCrystalIntegrityError) as exc:
            _integrity("O1 registry pin values are invalid", exc)


def _bank_basename(layer_index: int, bank_file_sha256: str) -> str:
    layer = _layer_index(layer_index)
    digest = _digest(bank_file_sha256, "bank_file_sha256")
    return f"{_BANK_NAME_PREFIX}{layer:02d}-mlp-o1-bank-{digest}.json"


@dataclass(frozen=True, slots=True)
class LayerMlpO1RegistryEntry:
    """Authenticated descriptor for one immutable single-crystal bank."""

    layer_index: int
    bank_file: str
    bank_file_sha256: str
    bank_identity_sha256: str
    crystal_sha256: str
    source_o1_state_sha256: str
    source_o1_generation: int
    action_abi: str
    packed_weight_bytes_avoided: int
    feature_radius: float
    error_radius: float
    max_error_radius: float = 0.0

    def __post_init__(self) -> None:
        layer = _layer_index(self.layer_index)
        bank_file = _basename(self.bank_file, field="bank_file")
        for field in (
            "bank_file_sha256",
            "bank_identity_sha256",
            "crystal_sha256",
            "source_o1_state_sha256",
        ):
            try:
                _digest(getattr(self, field), field)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"registry entry {field} is invalid") from exc
        if bank_file != _bank_basename(layer, self.bank_file_sha256):
            raise ValueError("bank_file is not the layer content-addressed basename")
        expected_abi = (
            LAYER_MLP_RESIDUAL_ACTION_ABI
            if layer == _MAX_LAYER_INDEX
            else LAYER_MLP_RESIDUAL_GENERIC_ACTION_ABI
        )
        if self.action_abi != expected_abi:
            raise ValueError("registry entry action ABI differs from its layer lane")
        _positive_uint(self.source_o1_generation, field="source_o1_generation")
        _positive_uint(
            self.packed_weight_bytes_avoided,
            field="packed_weight_bytes_avoided",
        )
        for field in ("feature_radius", "error_radius", "max_error_radius"):
            try:
                normalized = _finite_non_negative(getattr(self, field), field)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"registry entry {field} is invalid") from exc
            object.__setattr__(self, field, normalized)

    @property
    def layer(self) -> int:
        return self.layer_index

    @property
    def bank_basename(self) -> str:
        return self.bank_file

    def to_record(self) -> dict[str, object]:
        return {
            "action_abi": self.action_abi,
            "bank_file": self.bank_file,
            "bank_file_sha256": self.bank_file_sha256,
            "bank_identity_sha256": self.bank_identity_sha256,
            "crystal_sha256": self.crystal_sha256,
            "error_radius": self.error_radius,
            "feature_radius": self.feature_radius,
            "layer_index": self.layer_index,
            "max_error_radius": self.max_error_radius,
            "packed_weight_bytes_avoided": self.packed_weight_bytes_avoided,
            "source_o1_generation": self.source_o1_generation,
            "source_o1_state_sha256": self.source_o1_state_sha256,
        }

    @classmethod
    def from_record(cls, value: object) -> "LayerMlpO1RegistryEntry":
        fields = {
            "action_abi",
            "bank_file",
            "bank_file_sha256",
            "bank_identity_sha256",
            "crystal_sha256",
            "error_radius",
            "feature_radius",
            "layer_index",
            "max_error_radius",
            "packed_weight_bytes_avoided",
            "source_o1_generation",
            "source_o1_state_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            _integrity("O1 registry entry fields are invalid")
        try:
            return cls(**value)
        except (TypeError, ValueError) as exc:
            _integrity("O1 registry entry values are invalid", exc)


@dataclass(frozen=True, slots=True)
class LayerMlpO1PendingState:
    """One configured layer skipped because its collector is not ready."""

    layer_index: int
    reason: str
    source_o1_state_sha256: str | None = None
    source_o1_generation: int | None = None

    def __post_init__(self) -> None:
        _layer_index(self.layer_index)
        if self.reason not in {"state-missing", "crystal-pending"}:
            raise ValueError("pending O1 state reason is invalid")
        if self.reason == "state-missing":
            if (
                self.source_o1_state_sha256 is not None
                or self.source_o1_generation is not None
            ):
                raise ValueError("missing O1 state cannot carry provenance")
            return
        if self.source_o1_state_sha256 is None or self.source_o1_generation is None:
            raise ValueError("pending O1 state must carry sealed state provenance")
        _digest(self.source_o1_state_sha256, "source_o1_state_sha256")
        _positive_uint(self.source_o1_generation, field="source_o1_generation")

    @property
    def layer(self) -> int:
        return self.layer_index

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "layer_index": self.layer_index,
            "reason": self.reason,
        }
        if self.source_o1_state_sha256 is not None:
            record.update(
                {
                    "source_o1_generation": self.source_o1_generation,
                    "source_o1_state_sha256": self.source_o1_state_sha256,
                }
            )
        return record

    @classmethod
    def from_record(cls, value: object) -> "LayerMlpO1PendingState":
        missing_fields = {"layer_index", "reason"}
        pending_fields = missing_fields | {
            "source_o1_generation",
            "source_o1_state_sha256",
        }
        if not isinstance(value, Mapping) or set(value) not in (
            missing_fields,
            pending_fields,
        ):
            _integrity("pending O1 state fields are invalid")
        try:
            return cls(
                layer_index=value["layer_index"],
                reason=value["reason"],
                source_o1_state_sha256=value.get("source_o1_state_sha256"),
                source_o1_generation=value.get("source_o1_generation"),
            )
        except (TypeError, ValueError) as exc:
            _integrity("pending O1 state values are invalid", exc)


@dataclass(frozen=True, slots=True)
class LayerMlpO1MountedBank:
    """One fully verified registry entry and its opened bank."""

    entry: LayerMlpO1RegistryEntry
    bank: _Bank


@dataclass(frozen=True, slots=True)
class LayerMlpO1MountRegistry:
    """A verified manifest; empty registries are valid but not mountable."""

    manifest_path: Path
    manifest_file_sha256: str
    registry_sha256: str
    pins: LayerMlpO1RegistryPins
    configured_layers: tuple[int, ...]
    pending: tuple[LayerMlpO1PendingState, ...]
    mounts: tuple[LayerMlpO1MountedBank, ...]

    @classmethod
    def load(
        cls,
        manifest_path: str | os.PathLike[str],
        **kwargs: object,
    ) -> "LayerMlpO1MountRegistry":
        return load_layer_mlp_o1_registry(manifest_path, **kwargs)

    @property
    def entries(self) -> tuple[LayerMlpO1RegistryEntry, ...]:
        return tuple(mount.entry for mount in self.mounts)

    @property
    def pending_layers(self) -> tuple[int, ...]:
        return tuple(item.layer_index for item in self.pending)

    @property
    def mounted_layers(self) -> tuple[int, ...]:
        return tuple(item.entry.layer_index for item in self.mounts)

    @property
    def published_layers(self) -> tuple[int, ...]:
        return self.mounted_layers

    @property
    def banks_by_layer(self) -> dict[int, _Bank]:
        return {mount.entry.layer_index: mount.bank for mount in self.mounts}

    @property
    def mountable(self) -> bool:
        return bool(self.mounts)

    def bank_for_layer(self, layer_index: int) -> _Bank | None:
        layer = _layer_index(layer_index)
        for mount in self.mounts:
            if mount.entry.layer_index == layer:
                return mount.bank
        return None

    def require_mountable(self) -> "LayerMlpO1MountRegistry":
        if not self.mountable:
            raise LayerMlpO1RegistryNotMountableError(
                "O1 registry is valid but has no ready bank to mount"
            )
        return self


# Short alias for callers that do not need the explicit O1 qualifier.
LayerMlpMountRegistry = LayerMlpO1MountRegistry


@dataclass(frozen=True, slots=True)
class LayerMlpO1PublishSummary:
    """Deterministic, path-redacted summary of one offline publication."""

    manifest_path: Path
    manifest_file_sha256: str
    registry_sha256: str
    configured_layers: tuple[int, ...]
    published_bank_files: tuple[str, ...]
    published_layers: tuple[int, ...]
    pending_layers: tuple[int, ...]
    max_error_radius: float

    @property
    def mountable(self) -> bool:
        return bool(self.published_layers)

    def to_record(self) -> dict[str, object]:
        return {
            "configured_layers": list(self.configured_layers),
            "manifest_file": self.manifest_path.name,
            "manifest_file_sha256": self.manifest_file_sha256,
            "max_error_radius": self.max_error_radius,
            "mountable": self.mountable,
            "pending_count": len(self.pending_layers),
            "pending_layers": list(self.pending_layers),
            "published_bank_files": list(self.published_bank_files),
            "published_count": len(self.published_layers),
            "published_layers": list(self.published_layers),
            "registry_sha256": self.registry_sha256,
            "schema": LAYER_MLP_O1_PUBLISH_SUMMARY_SCHEMA,
        }


def _expected_bank_types(
    layer_index: int,
) -> tuple[type[_Bank], type[_Identity], type[_Crystal]]:
    if layer_index == _MAX_LAYER_INDEX:
        return (
            Layer63MlpResidualCrystalBank,
            Layer63MlpResidualCrystalIdentity,
            Layer63MlpResidualCrystal,
        )
    return (
        LayerMlpResidualCrystalBank,
        LayerMlpResidualCrystalIdentity,
        LayerMlpResidualCrystal,
    )


def _verify_bank(
    root: Path,
    entry: LayerMlpO1RegistryEntry,
    pins: LayerMlpO1RegistryPins,
) -> LayerMlpO1MountedBank:
    path = root / entry.bank_file
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
    except OSError as exc:
        _integrity(f"layer-{entry.layer_index} bank is unavailable", exc)
    if resolved.parent != resolved_root:
        _integrity(f"layer-{entry.layer_index} bank escapes the registry root")
    try:
        raw = _read_regular(path, label=f"layer-{entry.layer_index} bank")
    except FileNotFoundError as exc:
        _integrity(f"layer-{entry.layer_index} bank is missing", exc)
    if _sha256_bytes(raw) != entry.bank_file_sha256:
        _integrity(f"layer-{entry.layer_index} bank file SHA-256 mismatch")

    bank_cls, identity_cls, crystal_cls = _expected_bank_types(entry.layer_index)
    try:
        bank = bank_cls.load(path)
    except Exception as exc:
        _integrity(f"layer-{entry.layer_index} bank cannot be decoded", exc)
    if type(bank) is not bank_cls or type(bank.identity) is not identity_cls:
        _integrity(f"layer-{entry.layer_index} bank has the wrong v1/v2 lane")
    identity = bank.identity
    if identity.layer_index != entry.layer_index:
        _integrity(f"layer-{entry.layer_index} bank identity names another layer")
    if not pins.matches_identity(identity):
        _integrity(f"layer-{entry.layer_index} bank differs from registry pins")
    if identity.identity_sha256 != entry.bank_identity_sha256:
        _integrity(f"layer-{entry.layer_index} bank identity SHA-256 mismatch")
    if identity.action_abi != entry.action_abi:
        _integrity(f"layer-{entry.layer_index} bank action ABI mismatch")
    crystals = bank.crystals
    if len(crystals) != 1:
        _integrity(
            f"layer-{entry.layer_index} bank must contain exactly one current crystal"
        )
    crystal = crystals[0]
    if type(crystal) is not crystal_cls:
        _integrity(f"layer-{entry.layer_index} crystal has the wrong v1/v2 lane")
    coverage = crystal.coverage
    if (
        crystal.crystal_sha256 != entry.crystal_sha256
        or crystal.identity.identity_sha256 != entry.bank_identity_sha256
        or crystal.source_o1_state_sha256 != entry.source_o1_state_sha256
        or crystal.source_o1_generation != entry.source_o1_generation
        or crystal.packed_weight_bytes_avoided
        != entry.packed_weight_bytes_avoided
        or coverage.feature_radius != entry.feature_radius
        or coverage.error_radius != entry.error_radius
    ):
        _integrity(f"layer-{entry.layer_index} crystal metadata mismatch")
    if (
        crystal.source_o1_state_sha256 is None
        or crystal.source_o1_generation is None
    ):
        _integrity(f"layer-{entry.layer_index} crystal lacks O1 provenance")

    # Detect a replacement between the digest read and the bank decode.
    try:
        final_raw = _read_regular(path, label=f"layer-{entry.layer_index} bank")
    except FileNotFoundError as exc:
        _integrity(f"layer-{entry.layer_index} bank disappeared", exc)
    if final_raw != raw:
        _integrity(f"layer-{entry.layer_index} bank changed while mounting")
    return LayerMlpO1MountedBank(entry=entry, bank=bank)


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _integrity(f"O1 registry {field} must be a sequence")
    return value


def _expected_pins(
    pins: LayerMlpO1RegistryPins,
    *,
    expected_pins: LayerMlpO1RegistryPins | None,
    expected_identity: _Identity | None,
    expected_model_sha256: str | None,
    expected_q4_sha256: str | None,
    expected_atlas_revision_sha256: str | None,
    expected_graph_revision_sha256: str | None,
    expected_compute_graph_revision_sha256: str | None,
    expected_projection: LayerTransitionProjectionIdentity | None,
) -> None:
    if expected_identity is not None:
        identity_pins = LayerMlpO1RegistryPins.from_identity(expected_identity)
        if expected_pins is not None and expected_pins != identity_pins:
            raise ValueError("expected identity and expected pins disagree")
        expected_pins = identity_pins
    if expected_pins is not None:
        if not isinstance(expected_pins, LayerMlpO1RegistryPins):
            raise TypeError("expected_pins must be LayerMlpO1RegistryPins")
        if pins != expected_pins:
            _integrity("O1 registry differs from the expected runtime pins")
    if (
        expected_graph_revision_sha256 is not None
        and expected_compute_graph_revision_sha256 is not None
        and expected_graph_revision_sha256 != expected_compute_graph_revision_sha256
    ):
        raise ValueError("expected Compute graph pin aliases disagree")
    graph = (
        expected_graph_revision_sha256
        if expected_graph_revision_sha256 is not None
        else expected_compute_graph_revision_sha256
    )
    expected_scalars = (
        ("model_sha256", expected_model_sha256, pins.model_sha256),
        ("q4_sha256", expected_q4_sha256, pins.q4_sha256),
        (
            "atlas_revision_sha256",
            expected_atlas_revision_sha256,
            pins.atlas_revision_sha256,
        ),
        ("graph_revision_sha256", graph, pins.graph_revision_sha256),
    )
    for field, expected, actual in expected_scalars:
        if expected is None:
            continue
        try:
            _digest(expected, field)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"expected {field} is invalid") from exc
        if expected != actual:
            _integrity(f"O1 registry {field} differs from the expected pin")
    if expected_projection is not None:
        if not isinstance(expected_projection, LayerTransitionProjectionIdentity):
            raise TypeError(
                "expected_projection must be LayerTransitionProjectionIdentity"
            )
        if pins.projection != expected_projection:
            _integrity("O1 registry projection differs from the expected pin")


def load_layer_mlp_o1_registry(
    manifest_path: str | os.PathLike[str],
    *,
    expected_pins: LayerMlpO1RegistryPins | None = None,
    expected_identity: _Identity | None = None,
    expected_model_sha256: str | None = None,
    expected_q4_sha256: str | None = None,
    expected_atlas_revision_sha256: str | None = None,
    expected_graph_revision_sha256: str | None = None,
    expected_compute_graph_revision_sha256: str | None = None,
    expected_projection: LayerTransitionProjectionIdentity | None = None,
    require_mountable: bool = False,
) -> LayerMlpO1MountRegistry:
    """Load and fully authenticate a registry and every referenced bank."""

    if not isinstance(require_mountable, bool):
        raise TypeError("require_mountable must be a boolean")
    path = Path(manifest_path).expanduser().absolute()
    if not path.name:
        raise ValueError("manifest_path must name a file")
    _regular_directory(path.parent, create=False)
    try:
        raw = _read_regular(path, label="O1 registry manifest")
    except FileNotFoundError as exc:
        _integrity("O1 registry manifest is missing", exc)
    if not raw or len(raw) > _MAX_REGISTRY_BYTES:
        _integrity("O1 registry manifest size is outside its bound")
    try:
        body = _unseal_document(
            raw,
            schema=LAYER_MLP_O1_REGISTRY_ENVELOPE_SCHEMA,
            kind="layer-MLP O1 registry",
        )
    except LayerTransitionCrystalIntegrityError as exc:
        _integrity("O1 registry envelope or body hash is invalid", exc)
    fields = {"configured_layers", "entries", "pending", "pins", "schema"}
    if set(body) != fields or body["schema"] != LAYER_MLP_O1_REGISTRY_SCHEMA:
        _integrity("O1 registry body fields or schema are invalid")
    try:
        raw_layers = _sequence(body["configured_layers"], field="configured_layers")
        configured_layers = _normalize_layers(tuple(raw_layers))
        raw_entries = _sequence(body["entries"], field="entries")
        entries = tuple(LayerMlpO1RegistryEntry.from_record(item) for item in raw_entries)
        raw_pending = _sequence(body["pending"], field="pending")
        pending = tuple(LayerMlpO1PendingState.from_record(item) for item in raw_pending)
        pins = LayerMlpO1RegistryPins.from_record(body["pins"])
    except LayerMlpO1RegistryIntegrityError:
        raise
    except (TypeError, ValueError) as exc:
        _integrity("O1 registry body values are invalid", exc)
    entry_layers = tuple(item.layer_index for item in entries)
    pending_layers = tuple(item.layer_index for item in pending)
    if entry_layers != tuple(sorted(set(entry_layers))):
        _integrity("O1 registry entries are not sorted unique by layer")
    if pending_layers != tuple(sorted(set(pending_layers))):
        _integrity("O1 registry pending states are not sorted unique by layer")
    if set(entry_layers) & set(pending_layers):
        _integrity("O1 registry layer is both mounted and pending")
    if tuple(sorted((*entry_layers, *pending_layers))) != configured_layers:
        _integrity("O1 registry does not settle every configured layer")
    _expected_pins(
        pins,
        expected_pins=expected_pins,
        expected_identity=expected_identity,
        expected_model_sha256=expected_model_sha256,
        expected_q4_sha256=expected_q4_sha256,
        expected_atlas_revision_sha256=expected_atlas_revision_sha256,
        expected_graph_revision_sha256=expected_graph_revision_sha256,
        expected_compute_graph_revision_sha256=(
            expected_compute_graph_revision_sha256
        ),
        expected_projection=expected_projection,
    )
    mounts = tuple(_verify_bank(path.parent, entry, pins) for entry in entries)
    registry = LayerMlpO1MountRegistry(
        manifest_path=path,
        manifest_file_sha256=_sha256_bytes(raw),
        registry_sha256=_sha256_document(body),
        pins=pins,
        configured_layers=configured_layers,
        pending=pending,
        mounts=mounts,
    )
    if require_mountable:
        registry.require_mountable()
    return registry


def _capture_source(
    accumulator: _Accumulator,
    *,
    layer_index: int,
    pins: LayerMlpO1RegistryPins,
) -> tuple[_Crystal | None, LayerMlpO1PendingState | None]:
    bank_cls, identity_cls, crystal_cls = _expected_bank_types(layer_index)
    del bank_cls
    if type(accumulator.identity) is not identity_cls:
        _integrity(f"layer-{layer_index} O1 state has the wrong v1/v2 lane")
    identity = accumulator.identity
    if identity.layer_index != layer_index:
        _integrity(f"layer-{layer_index} O1 state names another layer")
    if not pins.matches_identity(identity):
        _integrity(f"layer-{layer_index} O1 state differs from global pins")
    crystal = accumulator.current_crystal()
    snapshot = accumulator.snapshot()
    if crystal is None:
        if snapshot.ready or snapshot.crystal_sha256 is not None:
            _integrity(f"layer-{layer_index} O1 state readiness is inconsistent")
        return None, LayerMlpO1PendingState(
            layer_index=layer_index,
            reason="crystal-pending",
            source_o1_state_sha256=snapshot.state_sha256,
            source_o1_generation=snapshot.generation,
        )
    if type(crystal) is not crystal_cls:
        _integrity(f"layer-{layer_index} O1 crystal has the wrong v1/v2 lane")
    if (
        not snapshot.ready
        or snapshot.crystal_sha256 != crystal.crystal_sha256
        or crystal.source_o1_state_sha256 != snapshot.state_sha256
        or crystal.source_o1_generation != snapshot.generation
        or crystal.identity.identity_sha256 != identity.identity_sha256
    ):
        _integrity(f"layer-{layer_index} O1 state changed during capture")
    return crystal, None


def _deterministic_bank_bytes(
    root: Path,
    *,
    layer_index: int,
    crystal: _Crystal,
) -> bytes:
    bank_cls, _identity_cls, _crystal_cls = _expected_bank_types(layer_index)
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".layer-{layer_index:02d}-bank.",
            dir=root,
        ) as temporary:
            path = Path(temporary) / "bank.json"
            bank = bank_cls(path, crystal.identity, max_crystals=1)
            published = bank.publish_latest(crystal)
            if published != crystal.crystal_sha256:
                _integrity(
                    f"layer-{layer_index} temporary bank rejected current crystal"
                )
            raw = _read_regular(path, label=f"layer-{layer_index} temporary bank")
            reopened = bank_cls.load(path)
            if len(reopened.crystals) != 1:
                _integrity(
                    f"layer-{layer_index} temporary bank is not single-crystal"
                )
            return raw
    except LayerMlpO1RegistryIntegrityError:
        raise
    except Exception as exc:
        _integrity(f"layer-{layer_index} bank cannot be serialized", exc)


def _publish_immutable(path: Path, raw: bytes) -> None:
    """Publish bytes without ever replacing an existing content address."""

    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".pending",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        os.fchmod(descriptor, 0o600)
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("zero-byte immutable bank write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            existing = _read_regular(path, label="content-addressed O1 bank")
            if existing != raw or _sha256_bytes(existing) != _sha256_bytes(raw):
                _integrity("content-addressed O1 bank contains different bytes")
        else:
            _fsync_directory(path.parent)
    except LayerMlpO1RegistryIntegrityError:
        raise
    except (LayerMlpCrystalIntegrityError, OSError) as exc:
        _integrity("content-addressed O1 bank cannot be published", exc)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _publish_crystal_bank(
    root: Path,
    *,
    layer_index: int,
    crystal: _Crystal,
    pins: LayerMlpO1RegistryPins,
    max_error_radius: float,
) -> LayerMlpO1RegistryEntry:
    raw = _deterministic_bank_bytes(
        root,
        layer_index=layer_index,
        crystal=crystal,
    )
    file_sha256 = _sha256_bytes(raw)
    coverage = crystal.coverage
    entry = LayerMlpO1RegistryEntry(
        layer_index=layer_index,
        bank_file=_bank_basename(layer_index, file_sha256),
        bank_file_sha256=file_sha256,
        bank_identity_sha256=crystal.identity.identity_sha256,
        crystal_sha256=crystal.crystal_sha256,
        source_o1_state_sha256=crystal.source_o1_state_sha256,
        source_o1_generation=crystal.source_o1_generation,
        action_abi=crystal.identity.action_abi,
        packed_weight_bytes_avoided=crystal.packed_weight_bytes_avoided,
        feature_radius=coverage.feature_radius,
        error_radius=coverage.error_radius,
        max_error_radius=max_error_radius,
    )
    target = root / entry.bank_file
    _publish_immutable(target, raw)
    _verify_bank(root, entry, pins)
    return entry


def _manifest_body(
    *,
    pins: LayerMlpO1RegistryPins,
    configured_layers: tuple[int, ...],
    entries: tuple[LayerMlpO1RegistryEntry, ...],
    pending: tuple[LayerMlpO1PendingState, ...],
) -> dict[str, object]:
    return {
        "configured_layers": list(configured_layers),
        "entries": [entry.to_record() for entry in entries],
        "pending": [item.to_record() for item in pending],
        "pins": pins.to_record(),
        "schema": LAYER_MLP_O1_REGISTRY_SCHEMA,
    }


def publish_layer_mlp_o1_registry(
    layer63_state_path: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    *,
    layers: Sequence[int] | None = None,
    max_error_radius: float = 0.0,
    manifest_name: str = DEFAULT_LAYER_MLP_O1_REGISTRY_BASENAME,
) -> LayerMlpO1PublishSummary:
    """Publish all ready configured O1 states without loading a Qwen model."""

    configured_layers = _normalize_layers(layers)
    try:
        allowed_error = _finite_non_negative(max_error_radius, "max_error_radius")
    except (TypeError, ValueError) as exc:
        raise ValueError("max_error_radius must be finite and non-negative") from exc
    manifest_basename = _basename(manifest_name, field="manifest_name")
    root = Path(output_root).expanduser().absolute()
    _regular_directory(root, create=True)
    manifest_path = root / manifest_basename

    legacy_path = Path(layer63_state_path).expanduser().absolute()
    try:
        authority = Layer63MlpO1Accumulator.load(legacy_path)
    except FileNotFoundError as exc:
        raise LayerMlpO1RegistryError(
            "legacy layer-63 O1 state is required as the filename and pin authority"
        ) from exc
    except Exception as exc:
        _integrity("legacy layer-63 O1 state is corrupt", exc)
    if type(authority.identity) is not Layer63MlpResidualCrystalIdentity:
        _integrity("legacy layer-63 O1 authority has the wrong identity lane")
    pins = LayerMlpO1RegistryPins.from_identity(authority.identity)
    sketch_dim = authority.identity.sketch_dim

    crystals: list[tuple[int, _Crystal]] = []
    pending: list[LayerMlpO1PendingState] = []
    for layer_index in configured_layers:
        if layer_index == _MAX_LAYER_INDEX:
            accumulator: _Accumulator = authority
        else:
            state_path = layer_mlp_o1_state_path_for_layer(
                legacy_path,
                layer_index=layer_index,
                sketch_dim=sketch_dim,
            )
            try:
                accumulator = LayerMlpO1Accumulator.load(state_path)
            except FileNotFoundError:
                pending.append(
                    LayerMlpO1PendingState(
                        layer_index=layer_index,
                        reason="state-missing",
                    )
                )
                continue
            except Exception as exc:
                _integrity(f"layer-{layer_index} O1 state is corrupt", exc)
        crystal, pending_state = _capture_source(
            accumulator,
            layer_index=layer_index,
            pins=pins,
        )
        if pending_state is not None:
            pending.append(pending_state)
        else:
            assert crystal is not None
            crystals.append((layer_index, crystal))

    entries: list[LayerMlpO1RegistryEntry] = []
    with _state_lock(manifest_path):
        try:
            os.lstat(manifest_path)
        except FileNotFoundError:
            existing_raw = None
        except OSError as exc:
            _integrity("existing O1 registry manifest cannot be inspected", exc)
        else:
            # Never heal or overwrite a corrupt registry implicitly.  A rerun
            # first verifies the complete previously committed generation.
            load_layer_mlp_o1_registry(manifest_path)
            existing_raw = _read_regular(
                manifest_path,
                label="existing O1 registry manifest",
            )

        for layer_index, crystal in crystals:
            entries.append(
                _publish_crystal_bank(
                    root,
                    layer_index=layer_index,
                    crystal=crystal,
                    pins=pins,
                    max_error_radius=allowed_error,
                )
            )
        ordered_entries = tuple(sorted(entries, key=lambda item: item.layer_index))
        ordered_pending = tuple(sorted(pending, key=lambda item: item.layer_index))
        body = _manifest_body(
            pins=pins,
            configured_layers=configured_layers,
            entries=ordered_entries,
            pending=ordered_pending,
        )
        encoded = _sealed_document(body, LAYER_MLP_O1_REGISTRY_ENVELOPE_SCHEMA)
        if len(encoded) > _MAX_REGISTRY_BYTES:
            _integrity("O1 registry manifest exceeds its byte bound")
        if existing_raw != encoded:
            try:
                _publish_bytes(manifest_path, encoded)
            except Exception as exc:
                _integrity("O1 registry manifest cannot be committed atomically", exc)

    mounted = load_layer_mlp_o1_registry(manifest_path, expected_pins=pins)
    if mounted.configured_layers != configured_layers:
        _integrity("published O1 registry configured-layer verification failed")
    return LayerMlpO1PublishSummary(
        manifest_path=manifest_path,
        manifest_file_sha256=mounted.manifest_file_sha256,
        registry_sha256=mounted.registry_sha256,
        configured_layers=mounted.configured_layers,
        published_bank_files=tuple(entry.bank_file for entry in mounted.entries),
        published_layers=mounted.mounted_layers,
        pending_layers=mounted.pending_layers,
        max_error_radius=allowed_error,
    )


# Descriptive aliases kept deliberately tiny for runtime and script callers.
publish_layer_mlp_o1_banks = publish_layer_mlp_o1_registry
publish_all_layer_mlp_o1 = publish_layer_mlp_o1_registry


__all__ = [
    "DEFAULT_LAYER_MLP_O1_REGISTRY_BASENAME",
    "LAYER_MLP_O1_PUBLISH_SUMMARY_SCHEMA",
    "LAYER_MLP_O1_REGISTRY_ENVELOPE_SCHEMA",
    "LAYER_MLP_O1_REGISTRY_SCHEMA",
    "LayerMlpMountRegistry",
    "LayerMlpO1MountRegistry",
    "LayerMlpO1MountedBank",
    "LayerMlpO1PendingState",
    "LayerMlpO1PublishSummary",
    "LayerMlpO1RegistryEntry",
    "LayerMlpO1RegistryError",
    "LayerMlpO1RegistryIntegrityError",
    "LayerMlpO1RegistryNotMountableError",
    "LayerMlpO1RegistryPins",
    "layer_mlp_o1_state_path_for_layer",
    "load_layer_mlp_o1_registry",
    "publish_all_layer_mlp_o1",
    "publish_layer_mlp_o1_banks",
    "publish_layer_mlp_o1_registry",
]
