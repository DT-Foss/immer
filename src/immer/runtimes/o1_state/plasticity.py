"""Plasticity: the organism learns from what flows through it.

Recipe from the measured o1-state line (R2-style rolling-quantile surprise
gate): every chunk gets a loss; a rolling window defines the current
"normal"; only chunks that SURPRISE (loss above the rolling quantile)
trigger a gradient step. Sleep replays buffered surprising spans at low
learning rate and clears the buffer — consolidation without drift.

House rules honored: torch threads capped at 1 (comparable numbers across
machines), one training job at a time.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from .adapter import O1StateStream


class LearningStream(O1StateStream):
    """O1StateStream that grows from experience instead of only watching it."""

    def __init__(
        self,
        *,
        window: int = 64,
        quantile: float = 0.50,
        lr: float = 3e-3,
        sleep_lr: float = 1e-3,
        span_buffer: int = 32,
        min_observations: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.torch.set_num_threads(1)
        self.optimizer = self.torch.optim.SGD(self.model.parameters(), lr=lr)
        self.sleep_optimizer = self.torch.optim.SGD(self.model.parameters(), lr=sleep_lr)
        self.window = deque(maxlen=window)
        self.quantile = quantile
        self.min_observations = min_observations
        self.spans: deque[bytes] = deque(maxlen=span_buffer)
        self.surprises = 0
        self.updates = 0
        self.sleeps = 0

    # -- gated plasticity --------------------------------------------------

    def _is_surprising(self, loss: float) -> bool:
        if len(self.window) < self.min_observations:
            return False
        values = sorted(self.window)
        rank = int(self.quantile * (len(values) - 1))
        return loss > values[rank]

    def observe(self, text: str) -> None:
        torch = self.torch
        data = text.encode("utf-8") or b"\x00"
        with torch.no_grad():
            pass  # gradient mode decided per chunk below
        for i in range(0, len(data), self.seq_len):
            chunk = data[i : i + self.seq_len]
            x = torch.tensor(list(chunk), dtype=torch.long).unsqueeze(0)
            logits, self.states = self.model(x, self.states)
            loss = torch.nn.functional.cross_entropy(logits[0], x[0])
            value = float(loss)
            self.loss_ema = value if self.loss_ema is None else 0.99 * self.loss_ema + 0.01 * value
            self.tokens += len(chunk)
            self.window.append(value)
            if self._is_surprising(value):
                self.surprises += 1
                self.spans.append(chunk)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.updates += 1
                # states were computed pre-update; detach the carried life
                self.states = [
                    z.detach() if hasattr(z, "detach") else z for z in self.states
                ]

    # -- consolidation -----------------------------------------------------

    def sleep(self, *, epochs: int = 2) -> dict[str, int]:
        """Replay surprising spans at low LR, then clear the buffer."""
        torch = self.torch
        replayed = 0
        if self.spans:
            frozen = list(self.spans)
            for _ in range(epochs):
                for chunk in frozen:
                    x = torch.tensor(list(chunk), dtype=torch.long).unsqueeze(0)
                    logits, _ = self.model(x, None)
                    loss = torch.nn.functional.cross_entropy(logits[0], x[0])
                    self.sleep_optimizer.zero_grad()
                    loss.backward()
                    self.sleep_optimizer.step()
                    replayed += 1
            self.spans.clear()
        self.sleeps += 1
        return {"replayed": replayed, "sleeps": self.sleeps}

    # -- reporting ---------------------------------------------------------

    def metrics(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "loss_ema": self.loss_ema,
            "surprises": self.surprises,
            "updates": self.updates,
            "sleeps": self.sleeps,
            "span_buffer": len(self.spans),
        }
