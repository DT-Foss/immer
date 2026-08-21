"""DonorBrain: the SOTA oracle on beast, spoken to over HTTP.

The resource law of this project: the resident organism stays tiny; the
15 GB brain lives on the server and is called only when the cascade
escalates. Zero local RAM, full donor capability. Qwen3.8 thinks before
speaking — /no_think keeps chat fast, harvest runs may let it think.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from ...contracts import ExecutionStatus, Request, Result

DEFAULT_DONOR = "http://127.0.0.1:8780/v1"  # via SSH-Tunnel auf beast; IMMER_DONOR_URL überschreibt


class DonorBrain:
    """Chat capability backed by the 27B donor on beast."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 280.0,
        think: bool = False,
        persona: str | None = None,
        max_tokens: int = 220,
        name: str = "donor.brain",
    ) -> None:
        self.base_url = (base_url or os.environ.get("IMMER_DONOR_URL") or DEFAULT_DONOR).rstrip("/")
        self.model = model or "qwen"  # llama_cpp.server serves whatever it loaded
        self.timeout = timeout
        self.think = think
        self.persona = persona
        self.max_tokens = max_tokens
        self.name = name
        self.model_id = f"Qwen3.8-27B@beast:{self.base_url}"

    @property
    def capabilities(self) -> frozenset:
        return frozenset({"chat", "harvest"})

    def available(self) -> bool:
        """Cheap liveness probe — one models call, two seconds budget."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/models", timeout=3) as response:
                return response.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def _payload(self, capability: str, text: str, history: list[dict[str, str]], life: str) -> dict[str, Any]:
        system = self.persona or (
            "Du bist das Mundwerk eines kleinen Lebewesens. Antworte kurz, "
            "ehrlich und auf Deutsch. Wenn du etwas nicht weißt: sag es. "
        )
        if capability == "harvest":
            system = (
                "Du erntest Wissen für die Bibliothek eines Lebewesens. "
                "Antworte mit genau einem kurzen sachlichen Satz auf Deutsch. "
            )
        if not self.think:
            system += "/no_think"
        messages = [
            {"role": "system", "content": f"{system} Lebens-Zeile: {life}"},
            *history[-6:],
            {"role": "user", "content": text},
        ]
        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0,
        }

    def handle(self, request: Request) -> Result:
        if request.capability not in self.capabilities:
            return Result(ExecutionStatus.REJECTED, self.name, reason="unsupported capability")
        payload = self._payload(
            request.capability,
            str(request.payload),
            list(request.metadata.get("history") or []),
            str(request.metadata.get("life", "")),
        )
        try:
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            return Result(ExecutionStatus.UNAVAILABLE, self.name, reason=f"donor nicht erreichbar: {exc}")
        choice = body.get("choices", [{}])[0]
        content = (choice.get("message") or {}).get("content", "").strip()
        if "</think>" in content:  # Qwen3.8 denkt vor dem Reden — das Denken ist nicht die Antwort
            content = content.split("</think>", 1)[1].strip()
        if not content:
            return Result(ExecutionStatus.ABSTAINED, self.name, reason="donor schweigt")
        usage = body.get("usage", {})
        return Result(
            ExecutionStatus.OK,
            self.name,
            output=content,
            evidence={
                "model": self.model_id,
                "completion_tokens": usage.get("completion_tokens"),
                "latency_note": "27B on CPU — der Preis der Größe",
            },
        )


def build_council(base_url: str | None = None):
    """The real rat: three donor roles sharing ONE loaded model on beast."""
    from ...council import Council

    return Council(
        (
            DonorBrain(base_url=base_url, name="donor.basis"),
            DonorBrain(
                base_url=base_url,
                name="donor.kritiker",
                persona="Du bist der Kritiker im Rat eines Lebewesens. Prüfe Aussagen auf "
                "Fehler und Widersprüche und korrigiere sie. Antworte kurz.",
            ),
            DonorBrain(
                base_url=base_url,
                name="donor.freigeist",
                persona="Du bist der Freigeist im Rat eines Lebewesens. Denk unkonventionell "
                "und bringe den Blickwinkel, den niemand sonst hat. Antworte kurz.",
            ),
        ),
        name="rat",
    )
