"""SHA-first, inference-only checkpoint ABI for the Seed v3 shadow proposer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import torch

from .config import SeedV3Config
from .model import ImmerSeedV3


SEED_V3_CHECKPOINT_SCHEMA = "immer-seed-v3-shadow-checkpoint/v1"
SEED_V3_MODEL_ABI = "gru-crsa-keyed-swiglu-shadow-proposer/v1"
SEED_V3_WEIGHTS_FORMAT = "torch-state-dict/weights-only/v1"
SEED_V3_LAB_SOURCE_SCHEMA = "immer-contextual-seed-run/v3"
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_WEIGHTS_BYTES = 2 * 1024 * 1024 * 1024


class SeedV3CheckpointError(ValueError):
    pass


class SeedV3CheckpointIntegrityError(SeedV3CheckpointError):
    pass


class SeedV3CheckpointIdentityError(SeedV3CheckpointError):
    pass


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha(value: object, *, field: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SeedV3CheckpointError(f"{field} must be a lowercase SHA-256")
    return value


def _stable_read(path: Path, *, maximum: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SeedV3CheckpointIntegrityError(f"checkpoint file is unavailable: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise SeedV3CheckpointIntegrityError("checkpoint file exceeds its bound")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise SeedV3CheckpointIntegrityError("checkpoint file was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise SeedV3CheckpointIntegrityError("checkpoint file grew during read")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise SeedV3CheckpointIntegrityError("checkpoint file changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _atomic_publish(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or _stable_read(path, maximum=max(len(data), 1)) != data:
            raise SeedV3CheckpointIntegrityError("checkpoint publication collided")
        return
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".pending", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path, follow_symlinks=False)
        except FileExistsError:
            if path.is_symlink() or _stable_read(path, maximum=max(len(data), 1)) != data:
                raise SeedV3CheckpointIntegrityError(
                    "checkpoint publication collided"
                )
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary_path.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True, order=True)
class SeedV3StateTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or len(self.name) > 1024:
            raise SeedV3CheckpointIntegrityError("state tensor name is invalid")
        if not isinstance(self.dtype, str) or not self.dtype or len(self.dtype) > 64:
            raise SeedV3CheckpointIntegrityError("state tensor dtype is invalid")
        shape = tuple(self.shape)
        if len(shape) > 16 or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in shape
        ):
            raise SeedV3CheckpointIntegrityError("state tensor shape is invalid")
        object.__setattr__(self, "shape", shape)

    def to_dict(self) -> dict[str, object]:
        return {"dtype": self.dtype, "name": self.name, "shape": list(self.shape)}

    @classmethod
    def from_dict(cls, value: object) -> "SeedV3StateTensor":
        if not isinstance(value, Mapping) or set(value) != {"dtype", "name", "shape"}:
            raise SeedV3CheckpointIntegrityError("state inventory row is malformed")
        try:
            return cls(
                name=value["name"],
                dtype=value["dtype"],
                shape=tuple(value["shape"]),
            )
        except (TypeError, ValueError) as exc:
            raise SeedV3CheckpointIntegrityError("state inventory row is invalid") from exc


def _state_inventory(state: Mapping[str, torch.Tensor]) -> tuple[SeedV3StateTensor, ...]:
    rows: list[SeedV3StateTensor] = []
    for name, tensor in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise SeedV3CheckpointIntegrityError("state dict is not tensor-only")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise SeedV3CheckpointIntegrityError("state dict contains non-finite weights")
        rows.append(
            SeedV3StateTensor(
                dtype=str(tensor.dtype).removeprefix("torch."),
                name=name,
                shape=tuple(tensor.shape),
            )
        )
    if not rows:
        raise SeedV3CheckpointIntegrityError("state dict is empty")
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class SeedV3CheckpointManifest:
    config: SeedV3Config
    code_revision_sha256: str
    tokenizer_manifest_sha256: str
    feature_schema_sha256: str | None
    source_receipt_sha256s: tuple[str, ...]
    weights_name: str
    weights_sha256: str
    weights_bytes: int
    state_inventory: tuple[SeedV3StateTensor, ...]
    parent_checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.config, SeedV3Config):
            raise TypeError("config must be SeedV3Config")
        for field in ("code_revision_sha256", "tokenizer_manifest_sha256", "weights_sha256"):
            object.__setattr__(self, field, _sha(getattr(self, field), field=field))
        feature = _sha(
            self.feature_schema_sha256,
            field="feature_schema_sha256",
            optional=True,
        )
        parent = _sha(
            self.parent_checkpoint_sha256,
            field="parent_checkpoint_sha256",
            optional=True,
        )
        if (self.config.receipt_feature_dim is None) != (feature is None):
            raise SeedV3CheckpointIdentityError(
                "receipt feature dimension and feature schema pin must appear together"
            )
        object.__setattr__(self, "feature_schema_sha256", feature)
        object.__setattr__(self, "parent_checkpoint_sha256", parent)
        try:
            receipts = tuple(sorted({_sha(value, field="source receipt") for value in self.source_receipt_sha256s}))
        except TypeError as exc:
            raise SeedV3CheckpointError("source receipts must be a sequence") from exc
        if not receipts or len(receipts) > 1_000_000:
            raise SeedV3CheckpointError("source receipt inventory is empty or too large")
        object.__setattr__(self, "source_receipt_sha256s", receipts)
        if (
            not isinstance(self.weights_name, str)
            or self.weights_name != f"weights-{self.weights_sha256}.pt"
        ):
            raise SeedV3CheckpointIntegrityError("weights name is not content-addressed")
        if (
            isinstance(self.weights_bytes, bool)
            or not isinstance(self.weights_bytes, int)
            or not 0 < self.weights_bytes <= MAX_WEIGHTS_BYTES
        ):
            raise SeedV3CheckpointError("weights byte count is invalid")
        inventory = tuple(self.state_inventory)
        if not inventory or any(not isinstance(row, SeedV3StateTensor) for row in inventory):
            raise SeedV3CheckpointIntegrityError("state inventory is malformed")
        if tuple(sorted(inventory, key=lambda row: row.name)) != inventory:
            raise SeedV3CheckpointIntegrityError("state inventory is not canonical")
        if len({row.name for row in inventory}) != len(inventory):
            raise SeedV3CheckpointIntegrityError(
                "state inventory tensor names are not unique"
            )
        object.__setattr__(self, "state_inventory", inventory)

    @property
    def architecture_sha256(self) -> str:
        return hashlib.sha256(
            _canonical({"config": self.config.to_dict(), "model_abi": SEED_V3_MODEL_ABI})
        ).hexdigest()

    def body(self) -> dict[str, object]:
        return {
            "architecture_sha256": self.architecture_sha256,
            "code_revision_sha256": self.code_revision_sha256,
            "config": self.config.to_dict(),
            "feature_schema_sha256": self.feature_schema_sha256,
            "model_abi": SEED_V3_MODEL_ABI,
            "parent_checkpoint_sha256": self.parent_checkpoint_sha256,
            "source_receipt_sha256s": list(self.source_receipt_sha256s),
            "state_inventory": [row.to_dict() for row in self.state_inventory],
            "tokenizer_manifest_sha256": self.tokenizer_manifest_sha256,
            "weights_bytes": self.weights_bytes,
            "weights_format": SEED_V3_WEIGHTS_FORMAT,
            "weights_name": self.weights_name,
            "weights_sha256": self.weights_sha256,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical(self.body())).hexdigest()

    def to_bytes(self) -> bytes:
        body = self.body()
        data = _canonical(
            {
                "body": body,
                "body_sha256": hashlib.sha256(_canonical(body)).hexdigest(),
                "schema": SEED_V3_CHECKPOINT_SCHEMA,
            }
        )
        if len(data) > MAX_MANIFEST_BYTES:
            raise SeedV3CheckpointError("checkpoint manifest exceeds its byte bound")
        return data

    @classmethod
    def from_bytes(cls, data: bytes) -> "SeedV3CheckpointManifest":
        if not isinstance(data, bytes) or not 0 < len(data) <= MAX_MANIFEST_BYTES:
            raise SeedV3CheckpointIntegrityError("checkpoint manifest byte count is invalid")

        def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = value
            return result

        try:
            document = json.loads(
                data.decode("utf-8"),
                object_pairs_hook=reject_duplicates,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise SeedV3CheckpointIntegrityError("checkpoint manifest is invalid JSON") from exc
        if _canonical(document) != data:
            raise SeedV3CheckpointIntegrityError("checkpoint manifest is not canonical")
        if (
            not isinstance(document, Mapping)
            or set(document) != {"body", "body_sha256", "schema"}
            or document.get("schema") != SEED_V3_CHECKPOINT_SCHEMA
            or not isinstance(document.get("body"), Mapping)
        ):
            raise SeedV3CheckpointIntegrityError("checkpoint manifest envelope is invalid")
        body = document["body"]
        if document.get("body_sha256") != hashlib.sha256(_canonical(body)).hexdigest():
            raise SeedV3CheckpointIntegrityError("checkpoint manifest digest mismatch")
        expected = {
            "architecture_sha256",
            "code_revision_sha256",
            "config",
            "feature_schema_sha256",
            "model_abi",
            "parent_checkpoint_sha256",
            "source_receipt_sha256s",
            "state_inventory",
            "tokenizer_manifest_sha256",
            "weights_bytes",
            "weights_format",
            "weights_name",
            "weights_sha256",
        }
        if set(body) != expected or body.get("model_abi") != SEED_V3_MODEL_ABI or body.get("weights_format") != SEED_V3_WEIGHTS_FORMAT:
            raise SeedV3CheckpointIntegrityError("checkpoint manifest ABI is invalid")
        try:
            result = cls(
                config=SeedV3Config.from_mapping(body["config"]),
                code_revision_sha256=body["code_revision_sha256"],
                tokenizer_manifest_sha256=body["tokenizer_manifest_sha256"],
                feature_schema_sha256=body["feature_schema_sha256"],
                source_receipt_sha256s=tuple(body["source_receipt_sha256s"]),
                weights_name=body["weights_name"],
                weights_sha256=body["weights_sha256"],
                weights_bytes=body["weights_bytes"],
                state_inventory=tuple(
                    SeedV3StateTensor.from_dict(row)
                    for row in body["state_inventory"]
                ),
                parent_checkpoint_sha256=body["parent_checkpoint_sha256"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SeedV3CheckpointIntegrityError("checkpoint manifest fields are invalid") from exc
        if body.get("architecture_sha256") != result.architecture_sha256 or result.to_bytes() != data:
            raise SeedV3CheckpointIntegrityError("checkpoint manifest replay changed")
        return result


@dataclass(frozen=True, slots=True)
class SeedV3CheckpointPublication:
    manifest: SeedV3CheckpointManifest
    manifest_path: Path
    weights_path: Path


def export_seed_v3_checkpoint(
    model: ImmerSeedV3,
    directory: str | os.PathLike[str],
    *,
    code_revision_sha256: str,
    tokenizer_manifest_sha256: str,
    source_receipt_sha256s: Sequence[str],
    feature_schema_sha256: str | None = None,
    parent_checkpoint_sha256: str | None = None,
) -> SeedV3CheckpointPublication:
    if not isinstance(model, ImmerSeedV3):
        raise TypeError("model must be ImmerSeedV3")
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    inventory = _state_inventory(state)
    buffer = io.BytesIO()
    torch.save(state, buffer)
    weights = buffer.getvalue()
    if not 0 < len(weights) <= MAX_WEIGHTS_BYTES:
        raise SeedV3CheckpointError("serialized weights exceed their byte bound")
    weights_sha256 = hashlib.sha256(weights).hexdigest()
    weights_name = f"weights-{weights_sha256}.pt"
    manifest = SeedV3CheckpointManifest(
        config=model.config,
        code_revision_sha256=code_revision_sha256,
        tokenizer_manifest_sha256=tokenizer_manifest_sha256,
        feature_schema_sha256=feature_schema_sha256,
        source_receipt_sha256s=tuple(source_receipt_sha256s),
        weights_name=weights_name,
        weights_sha256=weights_sha256,
        weights_bytes=len(weights),
        state_inventory=inventory,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
    )
    root = Path(directory)
    weights_path = root / weights_name
    manifest_path = root / f"manifest-{manifest.sha256}.json"
    _atomic_publish(weights_path, weights)
    _atomic_publish(manifest_path, manifest.to_bytes())
    return SeedV3CheckpointPublication(manifest, manifest_path, weights_path)


def load_seed_v3_checkpoint(
    manifest_path: str | os.PathLike[str],
    *,
    device: torch.device | str = "cpu",
    expected_code_revision_sha256: str | None = None,
    expected_tokenizer_manifest_sha256: str | None = None,
    expected_feature_schema_sha256: str | None = None,
) -> tuple[ImmerSeedV3, SeedV3CheckpointManifest]:
    path = Path(manifest_path)
    manifest = SeedV3CheckpointManifest.from_bytes(
        _stable_read(path, maximum=MAX_MANIFEST_BYTES)
    )
    for expected, actual, field in (
        (expected_code_revision_sha256, manifest.code_revision_sha256, "code revision"),
        (
            expected_tokenizer_manifest_sha256,
            manifest.tokenizer_manifest_sha256,
            "tokenizer manifest",
        ),
        (
            expected_feature_schema_sha256,
            manifest.feature_schema_sha256,
            "feature schema",
        ),
    ):
        if expected is not None:
            expected = _sha(expected, field=field)
            if expected != actual:
                raise SeedV3CheckpointIdentityError(f"checkpoint {field} changed")
    weights_path = path.parent / manifest.weights_name
    weights = _stable_read(weights_path, maximum=MAX_WEIGHTS_BYTES)
    if len(weights) != manifest.weights_bytes or hashlib.sha256(weights).hexdigest() != manifest.weights_sha256:
        raise SeedV3CheckpointIntegrityError("checkpoint weights digest mismatch")
    try:
        state = torch.load(io.BytesIO(weights), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise SeedV3CheckpointIntegrityError("checkpoint weights failed safe decoding") from exc
    if not isinstance(state, Mapping) or any(
        not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
        for name, tensor in state.items()
    ):
        raise SeedV3CheckpointIntegrityError("checkpoint weights are not a tensor state dict")
    inventory = _state_inventory(state)
    if inventory != manifest.state_inventory:
        raise SeedV3CheckpointIntegrityError("checkpoint state inventory changed")
    model = ImmerSeedV3(manifest.config)
    expected_inventory = _state_inventory(model.state_dict())
    if expected_inventory != inventory:
        raise SeedV3CheckpointIdentityError("checkpoint state dict differs from its config")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise SeedV3CheckpointIntegrityError("checkpoint state dict failed strict load") from exc
    model.to(device)
    model.eval()
    return model, manifest


def migrate_seed_v3_lab_checkpoint(
    source_path: str | os.PathLike[str],
    directory: str | os.PathLike[str],
    *,
    source_sha256: str,
    code_revision_sha256: str,
    tokenizer_manifest_sha256: str,
    source_receipt_sha256s: Sequence[str] = (),
    feature_schema_sha256: str | None = None,
    receipt_feature_dim: int | None = None,
    route_count: int = 5,
) -> SeedV3CheckpointPublication:
    """Import the shared v0.3 lab weights without optimizer/RNG payloads."""

    source_digest = _sha(source_sha256, field="source_sha256")
    source = _stable_read(Path(source_path), maximum=MAX_WEIGHTS_BYTES)
    if hashlib.sha256(source).hexdigest() != source_digest:
        raise SeedV3CheckpointIntegrityError("source lab checkpoint digest mismatch")
    try:
        payload = torch.load(io.BytesIO(source), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise SeedV3CheckpointIntegrityError(
            "source lab checkpoint failed safe decoding"
        ) from exc
    expected_fields = {
        "batch_size",
        "config",
        "device",
        "generator_state",
        "history",
        "loss_weights",
        "lr",
        "model",
        "optimizer",
        "schema",
        "seed",
        "steps",
        "torch_rng_state",
        "train_dialects",
        "trainer_steps",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != expected_fields
        or payload.get("schema") != SEED_V3_LAB_SOURCE_SCHEMA
        or not isinstance(payload.get("config"), Mapping)
        or not isinstance(payload.get("model"), Mapping)
    ):
        raise SeedV3CheckpointIntegrityError("source lab checkpoint schema is invalid")
    old = payload["config"]
    required_config = {
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
    if set(old) != required_config:
        raise SeedV3CheckpointIntegrityError("source lab config schema is invalid")
    config = SeedV3Config(
        **dict(old),
        key_code_bits=4,
        operator_classes=6,
        state_classes=17 * 17,
        quotient_classes=old["predictive_classes"],
        route_count=route_count,
        receipt_feature_dim=receipt_feature_dim,
    )
    seed = int(source_digest[:16], 16) % (2**63)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ImmerSeedV3(config)
    raw_state = payload["model"]
    migrated: dict[str, torch.Tensor] = {}
    for raw_name, tensor in raw_state.items():
        if not isinstance(raw_name, str) or not isinstance(tensor, torch.Tensor):
            raise SeedV3CheckpointIntegrityError(
                "source lab model is not a tensor state dict"
            )
        name = raw_name.replace(".attn.", ".attention.")
        if name in migrated:
            raise SeedV3CheckpointIntegrityError("source lab state key collided")
        migrated[name] = tensor.detach().cpu()
    result = model.load_state_dict(migrated, strict=False)
    allowed_missing = {
        "quotient_head.weight",
        "quotient_head.bias",
        "route_value_head.weight",
        "route_value_head.bias",
        "expected_work_head.weight",
        "expected_work_head.bias",
    }
    if receipt_feature_dim is not None:
        allowed_missing.add("receipt_projection.weight")
    if set(result.missing_keys) != allowed_missing or result.unexpected_keys:
        raise SeedV3CheckpointIdentityError(
            "source lab state dict differs from the Seed v3 migration ABI"
        )
    if model.quotient_head.weight.shape == model.predictive_head.weight.shape:
        with torch.no_grad():
            model.quotient_head.weight.copy_(model.predictive_head.weight)
            model.quotient_head.bias.copy_(model.predictive_head.bias)
    sources = tuple(source_receipt_sha256s) + (source_digest,)
    return export_seed_v3_checkpoint(
        model,
        directory,
        code_revision_sha256=code_revision_sha256,
        tokenizer_manifest_sha256=tokenizer_manifest_sha256,
        feature_schema_sha256=feature_schema_sha256,
        source_receipt_sha256s=sources,
        parent_checkpoint_sha256=source_digest,
    )


__all__ = [
    "SEED_V3_CHECKPOINT_SCHEMA",
    "SEED_V3_MODEL_ABI",
    "SEED_V3_LAB_SOURCE_SCHEMA",
    "SEED_V3_WEIGHTS_FORMAT",
    "SeedV3CheckpointError",
    "SeedV3CheckpointIdentityError",
    "SeedV3CheckpointIntegrityError",
    "SeedV3CheckpointManifest",
    "SeedV3CheckpointPublication",
    "SeedV3StateTensor",
    "export_seed_v3_checkpoint",
    "load_seed_v3_checkpoint",
    "migrate_seed_v3_lab_checkpoint",
]
