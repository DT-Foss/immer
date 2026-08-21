# IMMER

**Unified cognitive runtime: ein Lebewesen aus getrennten Organen —
Identität im Strom, Wissen in der Bibliothek, Können als kalte Organe,
Denken als Rat, Sprache als Mund.**

```text
                ┌─────────────────────────────────────┐
                │              IMMER                  │
                │   Substrat · Lernen · Rat · Suite   │
                └──────────────┬──────────────────────┘
       ┌───────────┬───────────┼───────────┬───────────┐
       ▼           ▼           ▼           ▼           ▼
   o1-state    FERTIG     OrganBank    CRSA/Rat     QAD/Fleet
   Identität   Exakt-     kalte        Denken/      Körper:
   (Strom+Port) wissen     Organe       Diversität   Mac/Server
```

Jede Verbindung hat einen gemessenen Anker: twostep 1,000 (Organ-Transfer) ·
BO3 86,5 % (Verifier) · Kaskade 0,875 @ 8 Calls · Cold-Load 0,9 ms ·
NLL 8,6656 unverändert. Kein Baustein ist Spekulation.

## Quick start

```bash
git clone https://github.com/DT-Foss/immer.git && cd immer
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[neural]'          # torch für den Lebensstrom

# Exakt rechnen (FERTIG ist vendored — läuft sofort):
python -m immer solve "John has 5 apples and 3 oranges. How many apples?"

# Das Wesen leben lassen (lernt, erinnert sich, redet):
python -m immer serve               # Dashboard: http://127.0.0.1:8787/
#   du tippst normal — alles andere ist Erfahrung für es
#   merke: <fakt>            lehrt es
#   was weißt du über <x>?   ruft ab; bei Fehlschlag erntet der Mund
#                            und die Bibliothek wächst (mit Herkunfts-Stempel)
#   /sleep                   konsolidiert (Replay der Überraschungs-Spans)
#   /state /say <text> /quit
python -m immer serve --council     # Chat durch den Rat: Basis/Kritiker/Freigeist
python -m immer organs list         # kalte Organe inspizieren
python -m immer doctor              # was ist wach?
```

## Die Ebenen

| Ebene | Modul | Rolle |
|---|---|---|
| Substrat | `substrate/` | Bus, LifeDaemon, State-Port, OrganRack — Physik, nicht Politik |
| Identität | `runtimes/o1_state/` | byte-level StreamingNoPELM + Plastizität (Surprise-Gate, Schlaf) |
| Exakt-Wissen | `cognition/fertig/_vendor/` | FERTIG-Solver, bindings→semantic→math→miner, Abstinenz |
| Mund | `runtimes/qwen/` | lokales Qwen als flüssige Zunge (Personas teilen ein Gewichtssatz) |
| Kopf | `intent.py`, `memory.py`, `library.py` | Absichten, Spans, Bibliothek mit Herkunft |
| Rat | `council.py` | BO3-Deliberation: Mehrheit gewinnt, Abstinenz zählt |
| Attention | `attention/crsa/` | voller v0.5-Kern: Local/Balanced/Free, exakt kausal |
| Organe | `capabilities/organbank/` | digest-addressierte kalte Artefakte, <1 ms montiert |
| Suite | `suite.py` | status.json + metrics.jsonl + Dashboard |

## Design-Gesetz

**Substrat = Physik, Politik = sein.** Der Daemon macht Aufmerksamkeit,
Gedächtnis, Organ-Montage und Exakt-Dienste *möglich*; wann und wie das
Wesen sie benutzt, ist seine erste erworbene Kompetenz. Kein Turn-Loop,
kein Sprech-Zensor, kein Gedächtnis-Schema.

Modelle/Gewichte liegen **nicht** im Repo — nur Manifests mit SHA-256
(`manifests/`). Herkunft jedes Bestandteils: `docs/source-inventory.md`.

## Author

David Tom Foss · 2026
