"""O1 surprise and learning-progress scoring for Markov episode retention."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
from typing import Any, Iterator

from .plasticity import LearningStream

try:
    import fcntl
except ImportError:  # pragma: no cover - production targets are POSIX.
    fcntl = None  # type: ignore[assignment]


O1_MARKOV_RETENTION_SCHEMA = "immer.o1-markov-retention/v1"
O1_MARKOV_RETENTION_POLICY = "surprise+learning-progress-loglength/v1"
_MAX_STATE_BYTES = 1024 * 1024
_HEX = frozenset("0123456789abcdef")


class O1MarkovRetentionError(RuntimeError):
    pass


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise O1MarkovRetentionError("O1 retention state is not canonical JSON") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - _HEX:
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


def _stable_read(path: Path) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | int(getattr(os, "O_CLOEXEC", 0))
            | int(getattr(os, "O_NOFOLLOW", 0)),
        )
        opened = os.fstat(descriptor)
        linked = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
            or not 0 < opened.st_size <= _MAX_STATE_BYTES
        ):
            raise O1MarkovRetentionError("O1 retention state file is invalid")
        chunks = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                raise O1MarkovRetentionError("O1 retention state was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise O1MarkovRetentionError("O1 retention state grew while reading")
        after = os.fstat(descriptor)
        if (
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
            opened.st_dev,
            opened.st_ino,
        ) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_dev,
            after.st_ino,
        ):
            raise O1MarkovRetentionError("O1 retention state changed while read")
        return b"".join(chunks)
    except O1MarkovRetentionError:
        raise
    except OSError as exc:
        raise O1MarkovRetentionError("cannot read O1 retention state") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


@dataclass(frozen=True, slots=True)
class O1RetentionScore:
    priority: float
    surprise: float
    learning_progress: float
    loss_before: float | None
    loss_after: float
    sequence: int

    def __post_init__(self) -> None:
        values = (self.priority, self.surprise, self.learning_progress, self.loss_after)
        if (
            any(not math.isfinite(value) or value < 0.0 for value in values)
            or (
                self.loss_before is not None
                and (not math.isfinite(self.loss_before) or self.loss_before < 0.0)
            )
            or isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence <= 0
        ):
            raise ValueError("O1 retention score is invalid")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class O1MarkovRetention:
    """Persistent tiny O1 organism that values confirmed answer episodes."""

    def __init__(
        self,
        path: str | Path,
        *,
        vocab_size: int,
        tokenizer_sha256: str,
    ) -> None:
        if (
            isinstance(vocab_size, bool)
            or not isinstance(vocab_size, int)
            or vocab_size <= 1
        ):
            raise ValueError("vocab_size must be greater than one")
        self.path = Path(path).expanduser().absolute()
        self.vocab_size = vocab_size
        self.tokenizer_sha256 = _sha(tokenizer_sha256, label="tokenizer_sha256")
        self.config = {
            "d_head": 32,
            "d_model": 32,
            "lr": float(3e-3).hex(),
            "min_observations": 4,
            "n_heads": 1,
            "n_layers": 1,
            "policy": O1_MARKOV_RETENTION_POLICY,
            "quantile": float(0.5).hex(),
            "seq_len": 64,
            "sleep_lr": float(1e-3).hex(),
            "span_buffer": 32,
            "window": 64,
        }
        self.sidecar = self.path.with_suffix(self.path.suffix + ".pt")
        self.stream = self._new_stream()
        self.sequence = 0
        self.last_score: O1RetentionScore | None = None
        self.priorities: dict[str, float] = {}
        with self._locked():
            self._reload_locked()

    def _new_stream(self) -> LearningStream:
        return LearningStream(
            d_model=32,
            n_layers=1,
            n_heads=1,
            d_head=32,
            seq_len=64,
            window=64,
            quantile=0.5,
            lr=3e-3,
            sleep_lr=1e-3,
            span_buffer=32,
            min_observations=4,
            sidecar=self.sidecar,
        )

    def _reset_from_disk_locked(self) -> None:
        self.stream = self._new_stream()
        self.sequence = 0
        self.last_score = None
        self.priorities = {}
        self._reload_locked()

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.parent / f".{self.path.name}.lock"
        descriptor = os.open(
            lock,
            os.O_CREAT
            | os.O_RDWR
            | int(getattr(os, "O_CLOEXEC", 0))
            | int(getattr(os, "O_NOFOLLOW", 0)),
            0o600,
        )
        try:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _reload_locked(self) -> None:
        if not self.path.exists() and not self.path.is_symlink():
            return
        try:
            raw = _stable_read(self.path)
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise O1MarkovRetentionError("cannot decode O1 retention state") from exc
        expected = {"body", "schema", "sha256"}
        if (
            not isinstance(document, Mapping)
            or set(document) != expected
            or document.get("schema") != O1_MARKOV_RETENTION_SCHEMA
            or not isinstance(document.get("body"), Mapping)
            or document.get("sha256") != _digest(document["body"])
            or _canonical(document) != raw
        ):
            raise O1MarkovRetentionError("O1 retention state schema is invalid")
        body = document["body"]
        if (
            set(body)
            != {
                "config",
                "last_score",
                "priorities",
                "sequence",
                "stream",
                "tokenizer_sha256",
                "vocab_size",
            }
            or body.get("config") != self.config
            or body.get("vocab_size") != self.vocab_size
            or body.get("tokenizer_sha256") != self.tokenizer_sha256
            or isinstance(body.get("sequence"), bool)
            or not isinstance(body.get("sequence"), int)
            or body["sequence"] < 0
            or not isinstance(body.get("stream"), Mapping)
            or not isinstance(body.get("priorities"), Mapping)
        ):
            raise O1MarkovRetentionError("O1 retention binding is invalid")
        score = body.get("last_score")
        try:
            self.last_score = (
                None if score is None else O1RetentionScore(**dict(score))
            )
        except (TypeError, ValueError) as exc:
            raise O1MarkovRetentionError("O1 retention score is invalid") from exc
        self.stream.restore(body["stream"])
        self.sequence = body["sequence"]
        try:
            priorities = {
                _sha(digest, label="episode digest"): float.fromhex(priority)
                for digest, priority in body["priorities"].items()
            }
        except (TypeError, ValueError) as exc:
            raise O1MarkovRetentionError("O1 retention priorities are invalid") from exc
        if any(not math.isfinite(value) or value < 0.0 for value in priorities.values()):
            raise O1MarkovRetentionError("O1 retention priority is invalid")
        self.priorities = priorities

    def _write_locked(self, snapshot: Mapping[str, Any]) -> None:
        body = {
            "config": self.config,
            "last_score": (
                None if self.last_score is None else self.last_score.to_dict()
            ),
            "priorities": {
                digest: priority.hex()
                for digest, priority in sorted(self.priorities.items())
            },
            "sequence": self.sequence,
            "stream": dict(snapshot),
            "tokenizer_sha256": self.tokenizer_sha256,
            "vocab_size": self.vocab_size,
        }
        data = _canonical(
            {
                "body": body,
                "schema": O1_MARKOV_RETENTION_SCHEMA,
                "sha256": _digest(body),
            }
        )
        if len(data) > _MAX_STATE_BYTES:
            raise O1MarkovRetentionError("O1 retention state exceeds its byte bound")
        temporary = self.path.parent / f".{self.path.name}.{secrets.token_hex(8)}.tmp"
        descriptor: int | None = None
        try:
            descriptor = os.open(
                temporary,
                os.O_CREAT
                | os.O_EXCL
                | os.O_WRONLY
                | int(getattr(os, "O_CLOEXEC", 0))
                | int(getattr(os, "O_NOFOLLOW", 0)),
                0o600,
            )
            view = memoryview(data)
            offset = 0
            while offset < len(view):
                written = os.write(descriptor, view[offset:])
                if written <= 0:
                    raise OSError("short O1 retention state write")
                offset += written
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.replace(temporary, self.path)
            try:
                self.stream.commit_snapshot(snapshot)
            except Exception:
                # The state already names the new content-addressed sidecar.
                # Pruning an obsolete object is cleanup, not commit failure.
                pass
        except OSError as exc:
            raise O1MarkovRetentionError("cannot persist O1 retention state") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

    def score(self, token_ids: Sequence[int], text: str) -> float:
        if isinstance(token_ids, (str, bytes, bytearray)):
            raise TypeError("token_ids must contain integers")
        tokens = tuple(token_ids)
        if not tokens or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or not 0 <= token < self.vocab_size
            for token in tokens
        ):
            raise ValueError("retention episode contains an invalid token")
        if not isinstance(text, str) or not text:
            raise ValueError("retention episode text must not be empty")
        with self._locked():
            self._reload_locked()
            try:
                before = self.stream.metrics()
                self.stream.observe(text)
                after = self.stream.metrics()
                loss_before = before["loss_ema"]
                loss_after = after["loss_ema"]
                if not isinstance(loss_after, float):
                    raise O1MarkovRetentionError("O1 retention produced no loss")
                surprise = float(after["surprises"] - before["surprises"])
                learning = float(after["updates"] - before["updates"])
                if isinstance(loss_before, float):
                    surprise += max(0.0, loss_after - loss_before)
                    learning += max(0.0, loss_before - loss_after)
                priority = math.log1p(len(tokens)) * (1.0 + surprise + learning)
                self.sequence += 1
                self.last_score = O1RetentionScore(
                    priority=priority,
                    surprise=surprise,
                    learning_progress=learning,
                    loss_before=loss_before,
                    loss_after=loss_after,
                    sequence=self.sequence,
                )
                self._remember_locked(tokens, priority)
                snapshot = self.stream.snapshot()
                self._write_locked(snapshot)
                return priority
            except Exception:
                self._reset_from_disk_locked()
                raise

    @staticmethod
    def _episode_digest(token_ids: Sequence[int]) -> str:
        return hashlib.sha256(
            json.dumps(
                list(token_ids),
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def _remember_locked(self, token_ids: Sequence[int], priority: float) -> None:
        digest = self._episode_digest(token_ids)
        self.priorities[digest] = max(
            priority,
            self.priorities.get(digest, 0.0),
        )
        if len(self.priorities) > 4096:
            del self.priorities[
                min(
                    self.priorities,
                    key=lambda key: (self.priorities[key], key),
                )
            ]

    def remember(self, token_ids: Sequence[int], priority: float) -> None:
        tokens = tuple(token_ids)
        if not tokens or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or not 0 <= token < self.vocab_size
            for token in tokens
        ):
            raise ValueError("retention alias contains an invalid token")
        if not math.isfinite(priority) or priority < 0.0:
            raise ValueError("retention alias priority is invalid")
        with self._locked():
            self._reload_locked()
            try:
                self._remember_locked(tokens, priority)
                snapshot = self.stream.snapshot()
                self._write_locked(snapshot)
            except Exception:
                self._reset_from_disk_locked()
                raise

    def priority(self, token_ids: Sequence[int]) -> float:
        return self.priorities.get(self._episode_digest(token_ids), 1.0)

    def metrics(self) -> dict[str, Any]:
        return {
            "last_score": (
                None if self.last_score is None else self.last_score.to_dict()
            ),
            "policy": O1_MARKOV_RETENTION_POLICY,
            "retained_priorities": len(self.priorities),
            "sequence": self.sequence,
            "stream": self.stream.metrics(),
        }


__all__ = [
    "O1_MARKOV_RETENTION_POLICY",
    "O1_MARKOV_RETENTION_SCHEMA",
    "O1MarkovRetention",
    "O1MarkovRetentionError",
    "O1RetentionScore",
]
