# CHANGELOG

Alle Änderungen an IMMER. Jeder Eintrag endet mit Deutung + nächster Frage —
nie mit der nackten Zahl.

## [0.2.0] — 2026-08-21 — Konsolidierung: alle Teile an einem Ort

**Ziel:** Nicht mehr wild an Einzelteilen rumschrauben. Dieser Commit führt die
verstreuten Stränge (Mac-Ordner, Server, Papers) in einem Repo zusammen und
errichtet das Substrat, auf dem der Organismus laufen wird.

### Zusammengeführt (Herkunft → Ort)

- `research/papers/` — Organ Grafting v0.1, Prefix-Sinkhorn v0.4/v0.5 (PDFs,
  Abgleich-Stand: `~/Downloads`)
- `research/formeln/FORMEL-FUNDAMENT.md` — der Formel-Kanon
  (Zeno-k*, Sättigung Δ(d)≈1,247·d/(d+33,1), Konzentrations-Formel, Ginibre)
  aus `self-verification_fable/mirkoNN-analysis`
- `research/crsa/` — voller v0.5-Operator (`operators_v05_full.py`, 534 Zeilen),
  REPORT1–6, Semigroup-Fix, NoPE-Diagnose, AI_HANDOFF
  aus `self-verification_fable/neue attention/`
- `research/reference/s3_harnesses/` — die gemessenen Harnesses:
  wirt_gate, verifier_loop, kaskade_graft, wesen_demo_v6, gssm_organ_proto,
  harvest/redistill, W3/W7/W14/W16-Organe, Donor-Chunk-Läufe + Auswertungen
  aus `mirkoNN-analysis/s3/`
- `docs/fleet/O1_STATE_OPERATIONS.md` — Server-Betrieb
  aus `~/Desktop/SERVER`

### Kanonische Quellen festgenagelt (manifests/components.json)

- **CRSA:** `src/immer/attention/crsa/operators.py` ist jetzt der VOLLSTÄNDIGE
  v0.5-Kern (vorher 225-Zeilen-Extrakt) + zurückportiertes
  `role_complete_attention` (Local|Balanced|Free) + `role_complete`-Dispatch.
  Digest neu gebunden: `a84d059f…`. Alle 5 CRSA-Tests grün (exakt Null
  Future-Support, Null Gradient in Zukunftszeilen, Free-Head-Invariante).
- **FERTIG-Pin korrigiert:** `d8c47117…` → `edea94a4a25fd639cbc02328ceb06489bb3c0170`
  (kuratiertes Runtime-Repo `~/Documents/Forschung/LanguageModel/FERTIG`,
  „Initial curated FERTIG runtime"). Der alte Pin war veraltet.
- macOS-Fix in `test_organbank.py`: Pfadvergleich über `.resolve()`
  (`/var/folders` → `/private/var/folders` Symlink-Artefakt).

### Neu: Substrat (`src/immer/substrate/`)

Erster Wurf des Lebens-Substrats — Physik statt Politik:

- `bus.py` — Ereignis-Bus mit Prioritätskanälen (USER > INTERNAL).
  Chat ist EIN Kanal unter mehreren, nicht der Zweck.
- `daemon.py` — LifeDaemon: kontinuierlicher Lebensprozess mit
  State-Port (JSON-Snapshot), restart-sicher inkl. Stream-Wiederherstellung
  (`RestorableLifeStream`); Organe werden ON DEMAND über OrganBank montiert
  (Entscheidung bleibt beim Organismus, Digest wird bei JEDDEM Mount
  verifiziert — manipulierte Organe mounten nie still); FERTIG als
  registrierbarer Exakt-Dienst; Unbekanntes → UNAVAILABLE statt Absturz.
- Kein Turn-Loop, kein Sprech-Zensor, kein Gedächtnis-Schema —
  das ist Politik und lernt das Wesen selbst.

### Deutung + nächste Frage

Damit ist IMMER kein Skelett mehr, sondern ein Anatomieatlas mit lebendem
Keim: jede Komponente hat Herkunft, Digest und Messanker. Die offene Frage
ist dieselbe wie vor dem Commit, jetzt aber mit Werkzeug: **Wie weit kommt
der erste vollständige Lebenszyklus — Chat rein, Organ montiert, Antwort
raus, Kill, Restart, Erinnerung da?** Nächster Schritt: Substrat-Daemon
gegen den echten o1-State-Portable (`O1_juli/src/portable_organism.py`)
anschließen und den Zyklus auf dem Mac fahren.

## [0.1.x] — 2026-08-20 — Ursprung

- `800a2ef` Integration Shell (Codex)
- `53cefef` Rebuild als unified cognitive runtime: Contracts
  (Request/Result/Component), Registry, Runtime, OrganBank (digest-addressed),
  FERTIG-Adapter, CRSA-Extrakt, CLI (components/doctor/solve), CI.
