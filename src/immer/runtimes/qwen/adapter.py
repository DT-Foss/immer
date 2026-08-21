"""QwenBrain: the fluent mouth. Local HF cache first, env override second.

Resolution order:
1. IMMER_QWEN_MODEL (HF id or local path)
2. ~/.cache/huggingface/hub Qwen2.5-1.5B-Instruct snapshot
3. Qwen2.5-0.5B snapshot
4. unavailable — chat falls back to the organism alone
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any

from ...contracts import ExecutionStatus, Request, Result

_CANDIDATES = (
    "models--Qwen--Qwen2.5-1.5B-Instruct",
    "models--Qwen--Qwen2.5-0.5B",
)

_ENGINE_CACHE: dict[str, tuple[Any, Any, str]] = {}


def resolve_model() -> str | None:
    configured = os.environ.get("IMMER_QWEN_MODEL")
    if configured:
        return configured
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    for name in _CANDIDATES:
        for snapshot in sorted(glob.glob(str(hub / name / "snapshots" / "*"))):
            if (Path(snapshot) / "config.json").is_file():
                return snapshot
    return None


class QwenBrain:
    """Chat capability backed by a locally cached Qwen model."""

    def __init__(self, *, max_new_tokens: int = 96, persona: str | None = None,
                 name: str = "qwen.brain") -> None:
        self.model_id = resolve_model()
        self.max_new_tokens = max_new_tokens
        self.persona = persona
        self.name = name
        self._model: Any = None
        self._tokenizer: Any = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def capabilities(self) -> frozenset:
        return frozenset({"chat"})

    def _ensure_loaded(self) -> None:
        if self.loaded or self.model_id is None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        if self.model_id in _ENGINE_CACHE:  # personas share one weight set
            self._model, self._tokenizer, self.device = _ENGINE_CACHE[self.model_id]
            return
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if device == "mps" else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=dtype
        ).to(device)
        self._model.eval()
        self.device = device
        _ENGINE_CACHE[self.model_id] = (self._model, self._tokenizer, self.device)

    def handle(self, request: Request) -> Result:
        if request.capability not in self.capabilities:
            return Result(ExecutionStatus.REJECTED, self.name, reason="unsupported capability")
        if self.model_id is None:
            return Result(ExecutionStatus.UNAVAILABLE, self.name, reason="no local Qwen brain found")
        try:
            self._ensure_loaded()
        except Exception as exc:  # noqa: BLE001 - a missing mouth must not kill the life
            return Result(ExecutionStatus.UNAVAILABLE, self.name, reason=f"{type(exc).__name__}: {exc}")
        history = request.metadata.get("history") or []
        system = self.persona or (
            "Du bist das Mundwerk eines kleinen Lebewesens. Antworte kurz, "
            "ehrlich und auf Deutsch. Wenn du etwas nicht weißt: sag es. "
        )
        messages = [
            {
                "role": "system",
                "content": (
                    f"{system}"
                    f"Bisheriges Leben des Wesens: {request.metadata.get('life', '')}"
                ),
            },
            *history,
            {"role": "user", "content": str(request.payload)},
        ]
        encoded = self._tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        if hasattr(encoded, "input_ids"):  # transformers 5.x wraps in BatchEncoding
            encoded = encoded.input_ids
        prompt = encoded.to(self.device)
        output = self._model.generate(
            prompt,
            attention_mask=self.torch.ones_like(prompt),
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        text = self._tokenizer.decode(output[0][prompt.shape[1]:], skip_special_tokens=True)
        return Result(ExecutionStatus.OK, self.name, output=text.strip(), evidence={"model": self.model_id})
