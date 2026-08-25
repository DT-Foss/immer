"""Production bridge to IMMER's side-effect-free o1-state model.

The life stream is a real StreamingNoPELM (byte-level, NoPE, stateful scan):
every user message flows through the organism as experience, per-layer Z
states carry the continuous life, and the whole being fits into one portable
sidecar file.  The model implementation contains the canonical recurrence and
checkpoint-compatible parameter names directly; no research script is imported.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import stat
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping


_SIDECAR_SCHEMA = "immer.o1-state-sidecar/v3"
_SIDECAR_RECEIPT_SCHEMA = "immer.o1-state-sidecar-receipt/v1"
_MAX_SIDECAR_BYTES = 512 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def is_available() -> bool:
    try:
        import torch  # noqa: F401
        from .model import StreamingNoPELM  # noqa: F401
    except ImportError:
        return False
    return True


class O1StateStream:
    """LifeStream implementation over the production organism primitive."""

    def __init__(
        self,
        *,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 2,
        d_head: int = 32,
        seq_len: int = 64,
        seed: int = 42,
        sidecar: str | Path | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "the o1-state bridge needs torch; install immer with '.[neural]'"
            ) from exc
        from .model import StreamingNoPELM

        self.torch = torch
        # fork_rng makes deterministic construction local to this organism;
        # it does not perturb the application's ambient torch RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = StreamingNoPELM(
                vocab_size=257,
                mask_idx=256,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                d_head=d_head,
                seq_len=seq_len,
                dropout=0.0,
                causal=True,
            )
        self.model.eval()
        self.seq_len = seq_len
        self.states: list[Any] = [None] * n_layers
        self.tokens = 0
        self.loss_ema: float | None = None
        self.tail: int | None = None
        self.sidecar = (
            Path(sidecar).expanduser().absolute() if sidecar is not None else None
        )
        self._lock = threading.RLock()

    # -- life -------------------------------------------------------------

    def observe(self, text: str) -> None:
        with self._lock:
            torch = self.torch
            chunks, next_tail = self._prediction_chunks(text)
            with torch.no_grad():
                for source, target, _span in chunks:
                    x = torch.tensor(list(source), dtype=torch.long).unsqueeze(0)
                    y = torch.tensor(list(target), dtype=torch.long).unsqueeze(0)
                    logits, self.states = self.model(x, self.states)
                    loss = torch.nn.functional.cross_entropy(logits[0], y[0])
                    value = float(loss)
                    self.loss_ema = (
                        value
                        if self.loss_ema is None
                        else 0.99 * self.loss_ema + 0.01 * value
                    )
                    self.tokens += len(source)
            self.tail = next_tail

    def _prediction_chunks(
        self, text: str
    ) -> tuple[tuple[tuple[bytes, bytes, bytes], ...], int]:
        """Return the canonical POS next-byte stream and its unconsumed tail."""

        if not isinstance(text, str):
            raise TypeError("O1 observations must be text")
        data = text.encode("utf-8") or b"\x00"
        prefix = b"" if self.tail is None else bytes((self.tail,))
        sequence = prefix + data
        next_tail = sequence[-1]
        transitions = len(sequence) - 1
        chunks: list[tuple[bytes, bytes, bytes]] = []
        for start in range(0, transitions, self.seq_len):
            stop = min(start + self.seq_len, transitions)
            source = sequence[start:stop]
            target = sequence[start + 1 : stop + 1]
            chunks.append((source, target, sequence[start : stop + 1]))
        return tuple(chunks), next_tail

    # -- continuity -------------------------------------------------------

    def snapshot(self) -> Mapping[str, Any]:
        with self._lock:
            document: dict[str, Any] = {
                "tokens": self.tokens,
                "loss_ema": self.loss_ema,
                "tail": self.tail,
            }
            if self.sidecar is not None:
                document["sidecar"] = self._save_sidecar()
            return document

    def restore(self, state: Mapping[str, Any]) -> None:
        with self._lock:
            self.tokens = int(state.get("tokens", 0))
            ema = state.get("loss_ema")
            self.loss_ema = float(ema) if ema is not None else None
            self.tail = self._validated_tail(state.get("tail"))
            receipt = state.get("sidecar")
            if self.sidecar is not None and receipt is not None:
                self._load_sidecar(receipt)

    @staticmethod
    def _validated_tail(value: object) -> int | None:
        if value is None:
            return None
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < 256
        ):
            raise ValueError("O1 stream tail must be one byte or null")
        return value

    def _content_sidecar_path(self, digest: str) -> Path:
        assert self.sidecar is not None
        return self.sidecar.with_name(
            f"{self.sidecar.stem}.{digest}{self.sidecar.suffix}"
        )

    @staticmethod
    def _stable_regular_bytes(path: Path) -> bytes:
        flags = os.O_RDONLY | int(getattr(os, "O_CLOEXEC", 0))
        flags |= int(getattr(os, "O_NOFOLLOW", 0))
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            linked = os.lstat(path)
            if (
                not stat.S_ISREG(opened.st_mode)
                or not stat.S_ISREG(linked.st_mode)
                or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
                or opened.st_size < 1
                or opened.st_size > _MAX_SIDECAR_BYTES
            ):
                raise ValueError("O1 sidecar is not a bounded stable regular file")
            remaining = opened.st_size
            chunks: list[bytes] = []
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError("O1 sidecar was truncated")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise ValueError("O1 sidecar grew while reading")
            current = os.fstat(descriptor)
            relinked = os.lstat(path)

            def signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
                return (
                    value.st_dev,
                    value.st_ino,
                    value.st_size,
                    value.st_mtime_ns,
                    value.st_ctime_ns,
                )

            if signature(current) != signature(opened) or (
                current.st_dev,
                current.st_ino,
            ) != (relinked.st_dev, relinked.st_ino):
                raise ValueError("O1 sidecar changed while reading")
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    def _save_sidecar(self) -> dict[str, Any]:
        assert self.sidecar is not None
        self.sidecar.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            dir=self.sidecar.parent,
            prefix=f".{self.sidecar.name}.",
            suffix=".tmp",
        )
        os.close(fd)
        temporary_path = Path(temporary)
        try:
            self.torch.save(self._sidecar_payload(), temporary_path)
            raw = self._stable_regular_bytes(temporary_path)
            digest = hashlib.sha256(raw).hexdigest()
            destination = self._content_sidecar_path(digest)
            try:
                os.link(temporary_path, destination, follow_symlinks=False)
            except FileExistsError:
                existing = self._stable_regular_bytes(destination)
                if existing != raw:
                    raise ValueError("O1 sidecar digest collision")
            directory = os.open(self.sidecar.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return {
                "name": destination.name,
                "schema": _SIDECAR_RECEIPT_SCHEMA,
                "sha256": digest,
                "size_bytes": len(raw),
            }
        finally:
            temporary_path.unlink(missing_ok=True)

    def _load_sidecar(self, receipt: object) -> None:
        assert self.sidecar is not None
        if isinstance(receipt, str):
            if receipt != self.sidecar.name:
                raise ValueError(
                    "legacy O1 sidecar name differs from its configured path"
                )
            path = self.sidecar
            expected_sha = expected_size = None
        else:
            fields = {"name", "schema", "sha256", "size_bytes"}
            if not isinstance(receipt, Mapping) or set(receipt) != fields:
                raise ValueError("invalid O1 sidecar receipt")
            digest = receipt.get("sha256")
            size = receipt.get("size_bytes")
            if (
                receipt.get("schema") != _SIDECAR_RECEIPT_SCHEMA
                or not isinstance(digest, str)
                or _SHA256.fullmatch(digest) is None
                or isinstance(size, bool)
                or not isinstance(size, int)
                or not 0 < size <= _MAX_SIDECAR_BYTES
            ):
                raise ValueError("invalid O1 sidecar receipt identity")
            path = self._content_sidecar_path(digest)
            if receipt.get("name") != path.name:
                raise ValueError("O1 sidecar receipt name is not content-addressed")
            expected_sha, expected_size = digest, size
        raw = self._stable_regular_bytes(path)
        if expected_size is not None and len(raw) != expected_size:
            raise ValueError("O1 sidecar size differs from its receipt")
        if expected_sha is not None and hashlib.sha256(raw).hexdigest() != expected_sha:
            raise ValueError("O1 sidecar digest differs from its receipt")
        bundle = self.torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        if not isinstance(bundle, Mapping):
            raise ValueError(f"invalid o1-state sidecar: {path}")
        self._restore_sidecar_payload(bundle)

    def commit_snapshot(self, state: Mapping[str, Any]) -> None:
        """Prune obsolete content-addressed states after scheduler commit."""

        if self.sidecar is None:
            return
        receipt = state.get("sidecar")
        if not isinstance(receipt, Mapping):
            return
        keep = receipt.get("name")
        if not isinstance(keep, str):
            return
        prefix = f"{self.sidecar.stem}."
        suffix = self.sidecar.suffix
        for candidate in self.sidecar.parent.iterdir():
            name = candidate.name
            digest = name[len(prefix) : -len(suffix)] if suffix else name[len(prefix) :]
            if (
                name == keep
                or not name.startswith(prefix)
                or (suffix and not name.endswith(suffix))
                or _SHA256.fullmatch(digest) is None
            ):
                continue
            metadata = candidate.lstat()
            if stat.S_ISREG(metadata.st_mode):
                candidate.unlink()

    def _sidecar_payload(self) -> dict[str, Any]:
        return {
            "schema": _SIDECAR_SCHEMA,
            "model": self.model.state_dict(),
            "states": self.states,
            "tokens": self.tokens,
            "loss_ema": self.loss_ema,
            "tail": self.tail,
        }

    def _restore_sidecar_payload(self, bundle: Mapping[str, Any]) -> None:
        if bundle.get("schema") not in {
            "immer.o1-state-sidecar/v2",
            _SIDECAR_SCHEMA,
        }:
            raise ValueError("unsupported O1 sidecar schema")
        self.model.load_state_dict(bundle["model"])
        states = bundle.get("states")
        if not isinstance(states, list) or len(states) != len(self.states):
            raise ValueError("sidecar state count does not match the organism")
        self.states = states
        self.tokens = int(bundle.get("tokens", self.tokens))
        ema = bundle.get("loss_ema", self.loss_ema)
        self.loss_ema = float(ema) if ema is not None else None
        self.tail = self._validated_tail(bundle.get("tail"))
