"""Plasticity: the organism learns from what flows through it.

Recipe from the measured o1-state line (R2-style rolling-quantile surprise
gate): every chunk gets a loss; a rolling window defines the current
"normal"; only chunks that SURPRISE (loss above the rolling quantile)
trigger a gradient step. Sleep replays buffered surprising spans at low
learning rate and clears the buffer — consolidation without drift.

Benchmark harnesses may pin torch to one thread for comparable measurements.
The reusable runtime deliberately leaves that process-global setting alone.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping

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
        max_grad_norm: float = 5.0,
        **kwargs: Any,
    ) -> None:
        if window < 1:
            raise ValueError("window must be positive")
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("quantile must be between 0 and 1")
        if min_observations < 0:
            raise ValueError("min_observations must be non-negative")
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        super().__init__(**kwargs)
        self.optimizer = self.torch.optim.SGD(self.model.parameters(), lr=lr)
        self.sleep_optimizer = self.torch.optim.SGD(
            self.model.parameters(), lr=sleep_lr
        )
        self.window = deque(maxlen=window)
        self.quantile = quantile
        self.min_observations = min_observations
        self.max_grad_norm = max_grad_norm
        self.spans: deque[bytes] = deque(maxlen=span_buffer)
        self.surprises = 0
        self.updates = 0
        self.sleeps = 0

    # -- gated plasticity --------------------------------------------------

    def _is_surprising(self, loss: float) -> bool:
        if not self.window or len(self.window) < self.min_observations:
            return True
        values = sorted(self.window)
        rank = int(self.quantile * (len(values) - 1))
        return loss > values[rank]

    def observe(self, text: str) -> None:
        with self._lock:
            self._observe_locked(text)

    def _observe_locked(self, text: str) -> None:
        torch = self.torch
        chunks, next_tail = self._prediction_chunks(text)
        for source, target, span in chunks:
            x = torch.tensor(list(source), dtype=torch.long).unsqueeze(0)
            y = torch.tensor(list(target), dtype=torch.long).unsqueeze(0)
            incoming = self._detached_states(self.states)

            # POS canon: measure without a graph; only a surprising chunk is
            # recomputed from the exact same incoming state with autograd.
            with torch.no_grad():
                logits, observed_states = self.model(x, incoming)
                observed_loss = torch.nn.functional.cross_entropy(logits[0], y[0])
            value = float(observed_loss)
            self.loss_ema = (
                value if self.loss_ema is None else 0.99 * self.loss_ema + 0.01 * value
            )
            self.tokens += len(source)
            surprising = self._is_surprising(value)
            if surprising:
                self.surprises += 1
                self.spans.append(span)
                logits, learned_states = self.model(x, incoming)
                loss = torch.nn.functional.cross_entropy(logits[0], y[0])
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.updates += 1
                self.states = self._detached_states(learned_states)
            else:
                self.states = self._detached_states(observed_states)
            # The threshold is defined by previous chunks only.
            self.window.append(value)
        self.tail = next_tail

    @staticmethod
    def _detached_states(states: list[Any]) -> list[Any]:
        return [
            state.detach() if hasattr(state, "detach") else state for state in states
        ]

    # -- consolidation -----------------------------------------------------

    def sleep(self, *, epochs: int = 2) -> dict[str, int]:
        """Replay surprising spans at low LR, then clear the buffer."""
        if epochs < 0:
            raise ValueError("epochs must be non-negative")
        with self._lock:
            return self._sleep_locked(epochs)

    def _sleep_locked(self, epochs: int) -> dict[str, int]:
        torch = self.torch
        replayed = 0
        if self.spans:
            frozen = list(self.spans)
            for _ in range(epochs):
                for span in frozen:
                    if len(span) < 2:
                        continue
                    x = torch.tensor(list(span[:-1]), dtype=torch.long).unsqueeze(0)
                    y = torch.tensor(list(span[1:]), dtype=torch.long).unsqueeze(0)
                    logits, _ = self.model(x, None)
                    loss = torch.nn.functional.cross_entropy(logits[0], y[0])
                    self.sleep_optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.sleep_optimizer.step()
                    replayed += 1
            self.spans.clear()
        self.sleeps += 1
        return {"replayed": replayed, "sleeps": self.sleeps}

    # -- reporting ---------------------------------------------------------

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "tokens": self.tokens,
                "loss_ema": self.loss_ema,
                "surprises": self.surprises,
                "updates": self.updates,
                "sleeps": self.sleeps,
                "span_buffer": len(self.spans),
            }

    # -- restart-safe plasticity --------------------------------------------

    def _sidecar_payload(self) -> dict[str, Any]:
        payload = super()._sidecar_payload()
        payload["plasticity"] = {
            "window": list(self.window),
            "spans": list(self.spans),
            "surprises": self.surprises,
            "updates": self.updates,
            "sleeps": self.sleeps,
            "optimizer": self.optimizer.state_dict(),
            "sleep_optimizer": self.sleep_optimizer.state_dict(),
        }
        return payload

    def _restore_sidecar_payload(self, bundle: Mapping[str, Any]) -> None:
        super()._restore_sidecar_payload(bundle)
        plasticity = bundle.get("plasticity")
        if not isinstance(plasticity, Mapping):
            return  # v1 sidecar: model/state continuity remains available
        self.window.clear()
        self.window.extend(float(value) for value in plasticity.get("window", ()))
        self.spans.clear()
        self.spans.extend(bytes(value) for value in plasticity.get("spans", ()))
        self.surprises = int(plasticity.get("surprises", 0))
        self.updates = int(plasticity.get("updates", 0))
        self.sleeps = int(plasticity.get("sleeps", 0))
        optimizer = plasticity.get("optimizer")
        if isinstance(optimizer, Mapping):
            self.optimizer.load_state_dict(dict(optimizer))
        sleep_optimizer = plasticity.get("sleep_optimizer")
        if isinstance(sleep_optimizer, Mapping):
            self.sleep_optimizer.load_state_dict(dict(sleep_optimizer))
