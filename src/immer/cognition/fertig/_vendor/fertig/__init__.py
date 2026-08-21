"""
fertig — ein kleines geerdetes Sprach- und Handlungssystem.

Deterministische, gewicht-freie Sprach-Erzeugung aus zwei gemessenen Quellen:
  * .causal-Wissensgraphen  (Fakten exakt, Walk generiert)
  * Korpora                 (gemessene Bigramm/Trigramm-Übergänge)

Der symbolische Kern bleibt gewicht-frei und deterministisch. Optional nutzt
die natürliche Oberfläche das eigene kleine HSSLM (ca. 2,21 Mio. Parameter)
als streng begrenzten Sprach-Ranker; Werkzeuge, Fakten und Desktop-Aktionen
bleiben geerdet. Die mathematischen Grundlagen umfassen Kontraktion (F33-F40),
Möbius-Kopplung, Ginibre-Kerne (F38), BvN-Zerlegung (F63-F68) und den
Berry-Phasen-Wächter (bphm).

Module:
  sampler        tau-kontrollierter Kontraktions-Sampler (Zeno, Ginibre, BvN)
  state_init     hyperboloide Symbol-Zustände (Berry-Phasen-Guard)
  bphm           Berry-Phasen-Wiederholungs-Erkennung
  pattern_bank   aus Korpora gemessene Satzform-Muster
  inference      Jaro-Winkler + 3-Pass-Ketten-Inferenz
  pipeline       .causal -> Walk -> Sprache (Fakten exakt, Form generiert)
  corpus         Korpus-Modus: gemessene Übergänge -> Fortsetzung
  mined          gesprochene Form mit gemessener Muster-Bank
"""

from __future__ import annotations

__version__ = "1.2.0"

from . import sampler, state_init, bphm, pattern_bank, inference
from . import pipeline, corpus, mined
from . import (
    intent,
    tools,
    learn,
    arena,
    bench,
    grammar,
    code,
    scrape,
    gaps,
    grounding,
    quant,
    vision,
    video,
    stream,
    interp,
)
from . import (
    apprentice,
    assistant,
    chat,
    desktop,
    desktop_agent,
    hsslm_interface,
    macos_recording,
    product_demo,
    screen_model,
    skill_slots,
)

__all__ = [
    "sampler",
    "state_init",
    "bphm",
    "pattern_bank",
    "inference",
    "pipeline",
    "corpus",
    "mined",
    "intent",
    "tools",
    "learn",
    "arena",
    "bench",
    "grammar",
    "code",
    "scrape",
    "gaps",
    "grounding",
    "quant",
    "vision",
    "video",
    "stream",
    "interp",
    "apprentice",
    "assistant",
    "chat",
    "desktop",
    "desktop_agent",
    "hsslm_interface",
    "macos_recording",
    "product_demo",
    "screen_model",
    "skill_slots",
    "__version__",
]
