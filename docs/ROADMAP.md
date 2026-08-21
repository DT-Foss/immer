# ROADMAP — die Hebel-Leiter

Stand: 2026-08-21. Jeder Hebel hat: Was, Warum (Messanker), Status.

## Erledigt

- [x] **Substrat** — Bus/Daemon/Port/Rack; Kill & Restart & erinnert sich (0.2.0)
- [x] **Vendoring** — o1-state + FERTIG komplett im Repo, Originale read-only (0.3.0)
- [x] **Erster Lebenszyklus** — chat → exact → kill → restart → memory (0.3.0)
- [x] **Plastizität** — Surprise-Gate + Schlaf; EMA fällt, Updates zählbar (0.4.0)
- [x] **Mund** — Qwen2.5-1.5B lokal als Chat-Fähigkeit (0.4.0)
- [x] **Kopf** — Intent-Router + SpanStore + Bibliothek mit Herkunfts-Stempel (0.5.0)
- [x] **Suite** — status.json / metrics.jsonl / Dashboard :8787 (0.4.0)

## Offen (aufsteigend nach Hebelkraft)

### 1. Auto-Ernte aus CHAT
Gesprochenes wird gestreamt, aber nicht automatisch als Fakt gespeichert.
Hebel: nach jedem CHAT-Turn eine billige Extraktionsfrage an den Mund
(„Enthält diese Aussage eine lehrbare Tatsache? Wenn ja: ein Satz.") →
Span mit `source: harvest-chat`. Die Bibliothek wächst dann im Gespräch,
nicht nur bei RECALL-Miss.

### 2. Council live schalten (--council ist gebaut, aber ungemessen)
Drei Qwen-Rollen (Basis/Kritiker/Freigeist) teilen EIN Gewichtssatz.
Zu messen: Antwortqualität Rat vs Einzelhirn auf einem festen Fragen-Set
(20 Fragen, Placebo = Einzelhirn), Kosten in Sekunden/Turn.

### 3. Donor-Bibliothek (beast)
Beast-27B als Offline-Orakel: Ernte-Läufe pro Thema → .causal-Karten →
`manifests/artifacts.sha256`. Messgröße: Antwortqualität mit/ohne Index,
Index-Wachstum pro Lebenstag. Nutzt das gemessene Kaskaden-Muster.

### 4. Denkmodell-Graft (der Forschungsschuss)
Spektren-Ablation am kleinen Base (Wissens-Layer raus), Reasoning-Traces
vom Donor rein, kristallisieren. Ziel: paar hundert MB Denkorgan ohne
Weltwissen. Werkzeuge existieren: Konzentrations-Formel, Fähigkeits-
Spektren, Kristallisations-Rezept (W7/W16).

### 5. Identitäts-Swap-Test
Gehirn mitten im Gespräch wechseln (1.5B ↔ 0.5B ↔ Graft), Wesen läuft
weiter — State-Port trägt die Identität, nicht die Gewichte. Die Demo,
die niemand sonst hat.

### 6. Flotte
core/beast als Lebens-Knoten (portable_organism Migration ist gemessen:
bit-identisch). Redmi/iPad als Körper. intel bleibt unantastbar.

## Betriebsregeln (aus Messungen, gelten immer)

1. Originale read-only — gefrickelt wird nur in `~/immer`.
2. Jede Messung mit Placebo/Kontrast; nur der Abstand zählt.
3. Negative erst nach Vollgas-Protokoll akzeptieren (Mechanismus-Frage,
   20+ reps, Hebel-Set, pkg-search-Analog).
4. CHANGELOG-Eintrag endet mit Deutung + nächster Frage.
5. torch threads=1 im Lernstrom; ein Trainings-Job pro Maschine.
