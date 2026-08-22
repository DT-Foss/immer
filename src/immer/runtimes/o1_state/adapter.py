"""Production bridge to IMMER's side-effect-free o1-state model.

The life stream is a real StreamingNoPELM (byte-level, NoPE, stateful scan):
every user message flows through the organism as experience, per-layer Z
states carry the continuous life, and the whole being fits into one portable
sidecar file.  The model implementation contains the canonical recurrence and
checkpoint-compatible parameter names directly; no research script is imported.
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping


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
        self.sidecar = Path(sidecar).expanduser() if sidecar is not None else None
        self._lock = threading.RLock()

    # -- life -------------------------------------------------------------

    def observe(self, text: str) -> None:
        with self._lock:
            torch = self.torch
            data = text.encode("utf-8") or b"\x00"
            with torch.no_grad():
                for i in range(0, len(data), self.seq_len):
                    chunk = data[i : i + self.seq_len]
                    x = torch.tensor(list(chunk), dtype=torch.long).unsqueeze(0)
                    logits, self.states = self.model(x, self.states)
                    loss = torch.nn.functional.cross_entropy(logits[0], x[0])
                    value = float(loss)
                    self.loss_ema = (
                        value
                        if self.loss_ema is None
                        else 0.99 * self.loss_ema + 0.01 * value
                    )
                    self.tokens += len(chunk)

    # -- continuity -------------------------------------------------------

    def snapshot(self) -> Mapping[str, Any]:
        with self._lock:
            document: dict[str, Any] = {"tokens": self.tokens, "loss_ema": self.loss_ema}
            if self.sidecar is not None:
                self._save_sidecar()
                document["sidecar"] = self.sidecar.name
            return document

    def restore(self, state: Mapping[str, Any]) -> None:
        with self._lock:
            self.tokens = int(state.get("tokens", 0))
            ema = state.get("loss_ema")
            self.loss_ema = float(ema) if ema is not None else None
            if self.sidecar is not None and self.sidecar.is_file():
                self._load_sidecar()

    def _save_sidecar(self) -> None:
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
            os.replace(temporary_path, self.sidecar)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _load_sidecar(self) -> None:
        bundle = self.torch.load(self.sidecar, map_location="cpu", weights_only=True)
        if not isinstance(bundle, Mapping):
            raise ValueError(f"invalid o1-state sidecar: {self.sidecar}")
        self._restore_sidecar_payload(bundle)

    def _sidecar_payload(self) -> dict[str, Any]:
        return {
            "schema": "immer.o1-state-sidecar/v2",
            "model": self.model.state_dict(),
            "states": self.states,
            "tokens": self.tokens,
            "loss_ema": self.loss_ema,
        }

    def _restore_sidecar_payload(self, bundle: Mapping[str, Any]) -> None:
        self.model.load_state_dict(bundle["model"])
        states = bundle.get("states")
        if not isinstance(states, list) or len(states) != len(self.states):
            raise ValueError("sidecar state count does not match the organism")
        self.states = states
        self.tokens = int(bundle.get("tokens", self.tokens))
        ema = bundle.get("loss_ema", self.loss_ema)
        self.loss_ema = float(ema) if ema is not None else None
