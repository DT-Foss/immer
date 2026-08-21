"""Bridge to the vendored o1-state organism (`vendor/o1state`).

The life stream is a real StreamingNoPELM (byte-level, NoPE, stateful scan):
every user message flows through the organism as experience, per-layer Z
states carry the continuous life, and the whole being fits into one portable
sidecar file. Originals stay untouched — this wraps the vendored copy.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

_VENDOR_ROOT = Path(__file__).resolve().parents[4] / "vendor" / "o1state"


def _vendor_paths() -> tuple[Path, Path]:
    return _VENDOR_ROOT / "src", _VENDOR_ROOT / "reference"


def is_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    src, ref = _vendor_paths()
    return (src / "streaming_train.py").is_file() and (ref / "moebius_scan_transformer_sqrt.py").is_file()


class O1StateStream:
    """LifeStream implementation over the vendored organism primitive."""

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
        if not is_available():
            raise RuntimeError("vendored o1-state sources missing under vendor/o1state/")
        src, ref = _vendor_paths()
        for path in (str(src), str(ref)):
            if path not in sys.path:
                sys.path.insert(0, path)
        from streaming_train import StreamingNoPELM

        self.torch = torch
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

    # -- life -------------------------------------------------------------

    def observe(self, text: str) -> None:
        torch = self.torch
        data = text.encode("utf-8") or b"\x00"
        with torch.no_grad():
            for i in range(0, len(data), self.seq_len):
                chunk = data[i : i + self.seq_len]
                x = torch.tensor(list(chunk), dtype=torch.long).unsqueeze(0)
                logits, self.states = self.model(x, self.states)
                loss = torch.nn.functional.cross_entropy(logits[0], x[0])
                value = float(loss)
                self.loss_ema = value if self.loss_ema is None else 0.99 * self.loss_ema + 0.01 * value
                self.tokens += len(chunk)

    # -- continuity -------------------------------------------------------

    def snapshot(self) -> Mapping[str, Any]:
        document: dict[str, Any] = {"tokens": self.tokens, "loss_ema": self.loss_ema}
        if self.sidecar is not None:
            self._save_sidecar()
            document["sidecar"] = self.sidecar.name
        return document

    def restore(self, state: Mapping[str, Any]) -> None:
        self.tokens = int(state.get("tokens", 0))
        ema = state.get("loss_ema")
        self.loss_ema = float(ema) if ema is not None else None
        if self.sidecar is not None and self.sidecar.is_file():
            self._load_sidecar()

    def _save_sidecar(self) -> None:
        self.sidecar.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "model": self.model.state_dict(),
                "states": self.states,
                "tokens": self.tokens,
                "loss_ema": self.loss_ema,
            },
            self.sidecar,
        )

    def _load_sidecar(self) -> None:
        bundle = self.torch.load(self.sidecar, weights_only=False)
        self.model.load_state_dict(bundle["model"])
        self.states = bundle["states"]
