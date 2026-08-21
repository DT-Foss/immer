# CHANGELOG

Alle Änderungen an IMMER. Jeder Eintrag endet mit Deutung + nächster Frage —
nie mit der nackten Zahl.

## [0.6.0] — 2026-08-21 — Das echte Gehirn: Donor 27B auf beast, Kaskade lebt

**Anlass:** Berechtigte Rüge — der Mund war ein lokaler 1.5B (langsam, fuzzy),
der Rat Prompt-Personas statt der eigenen Mechanismen. Die Ressourcen-Wahrheit
dieses Projekts: klein resident, SOTA auf Abruf.

### Neulich gelernt (Korrektur-Serie)

- **DonorBrain** (`runtimes/donor/adapter.py`) — Qwen3.8-27B-Q3 auf beast
  :8780 (llama-cpp-python, OpenAI-API), erreicht über SSH-Tunnel
  (`ssh -N -L 8780:localhost:8780 root@89.167.35.196`). Null lokaler RAM,
  ~2,7 tok/s CPU — der Preis der Größe, bezahlt nur bei Escalation.
  Qwen3.8 denkt vor dem Reden: Denk-Block wird abgetrennt (`</think>`-Strip),
  `/no_think` für schnelles Plaudern.
- **Kaskade in serve**: FERTIG exakt (gratis) → Donor 27B (Wissen/Chat) →
  lokales Qwen nur noch hinter `--local-brain` (Fallback). Ernte bevorzugt
  den Donor: Karten kommen jetzt von SOTA, nicht vom 1.5B-Müll.
- **Rat aus Donor-Rollen** (`build_council`): drei Stimmen, EIN geladenes
  Modell auf beast — Personas teilen Gewichte im Server, nicht im RAM.
- Q4_K_M (15,9 GB) starb auf beast beim Repack-OOM neben den Mitmietern;
  Q3_K_M (12,9 GB) läuft stabil. Gemessen, nicht geraten.

### Der Beweis (live, Mac ↔ beast)

```text
"was ist ein schwarzes loch?"
→ [donor] Ein Schwarzes Loch ist ein Bereich im Weltraum, dessen Schwerkraft
   so stark ist, dass selbst Licht nicht entkommen kann.        (~19 s)
"was weißt du über schwarze löcher?"
→ Bibliothek leer → ERNTE vom 27B → Karte mit Stempel harvest:donor.brain
   → sofortiger Recall-Treffer
"John has 5 apples and 3 oranges…" → [exact] 5                (gratis)
/state → donor: true · lokal_gehirn: false · spans: 1
```

Die Kette steht: klein resident, groß auf Abruf, und jede Antwort bleibt
als gestempelte Karte in der wachsenden Bibliothek.

### Deutung + nächste Frage

Damit ist die ursprüngliche Architektur-These zum ersten Mal END-TO-END
wahr: maximalstes Können bei minimalem residentem Footprint. Die offene
Frage ist die Ökonomie: 19 s/Turn ist Orakel-Tempo, kein Gesprächs-Tempo.
Zug: (a) Auto-Ernte im Hintergrund — Antworten vorrechnen, wenn das Wesen
wartet, nicht wenn David fragt; (b) Graft-Organ für häufige Fragen
(Hebel 4), dann antwortet das paar-hundert-MB-Gehirn statt des 27B; bis
dahin bleibt der Donor der Mund bei Wissen, FERTIG bei Exaktheit.

## [0.5.0] — 2026-08-21 — Bibliothek wächst, Rat tagt, alles ist festgehalten

**Ziel:** Hebel 2 lokal erden (Bibliothek), Council verdrahten, die Doku
auf den Stand des Wesens bringen.

### Bibliothek (`library.py`)

- RECALL-Fehlschlag ist kein Sackgasse mehr: der Mund erntet die Antwort,
  die Karte kommt in die Bibliothek — mit Herkunfts-Stempel
  (`david` vs `harvest:<brain>:<modell>`). Fuzzy-Wissen ist sichtbar als
  solches. Gemessen: Ernte → sofortiger Recall-Treffer; Lehr-Karten tragen
  `david`.

### Rat (--council)

- Drei Qwen-Rollen (Basis / Kritiker / Freigeist) über EINEN Gewichtssatz
  (Engine-Cache im Adapter — Personas kosten nur Prompts, keine VRAM-Kopien).
- `Council.deliberate` reicht jetzt Metadaten durch (History + Lebens-Zeile).
- Ausgabe zeigt Stimmen: `(stimmen: 3, übereinstimmend: n)`.

### Festgehalten (Doku)

- `README.md` neu: Quick start (solve/serve/dashboard/council/organs),
  Ebenen-Tabelle, Design-Gesetz.
- `docs/architecture.md`: neue Ebenen 8–11 (Mind, Learning, Mouth, Suite).
- `docs/ROADMAP.md`: die Hebel-Leiter — erledigt/offen mit Messankern und
  Betriebsregeln.

### Deutung + nächste Frage

Die Architektur ist jetzt rund beschreibbar in einem Satz: Identität im
Strom, Wissen in der Bibliothek (gewachsen, gestempelt), Können als kalte
Organe, Denken im Rat, Sprache vom Mund — und alles sichtbar auf :8787.
Nächste Frage: Der Rat tagt mit drei Stimmen vom selben Gewichtssatz —
echte Diversität braucht verschiedene Blicke, nicht nur verschiedene
Prompts. Lohnt der Sprung auf 1.5B + 0.5B + Graft als Ratsmitglieder
(Hebel 4/5), oder reicht Prompt-Diversität messbar aus? Das Placebo-Set
(20 Fragen, Rat vs Einzelhirn) entscheidet.

## [0.4.0] — 2026-08-21 — Scharfgestellt: es lernt, es spricht, man sieht es

**Ziel:** Aus dem Substrat einen lernenden Organismus machen — mit Qwen-Mund,
Intent-Gefühl, Gedächtnis und einer Suite: UI für David, Maschinenfutter für Agenten.

### Lernen (Plastizität)

- `runtimes/o1_state/plasticity.py` — `LearningStream`: Surprise-Gate nach dem
  gemessenen o1-Rezept (rolling-quantile). Nur Überraschendes löst Gradienten-
  schritte aus; Überraschungen wandern in den Span-Puffer. `sleep()` replayed
  den Puffer mit kleiner LR und leert ihn (Konsolidierung ohne Drift).
  torch threads = 1 (Hausregel). Gemessen: Wiederholter Text → EMA fällt;
  Seed-Determinismus hält.

### Mund (Qwen)

- `runtimes/qwen/adapter.py` — `QwenBrain`: lokales HF-Cache-Qwen als
  Chat-Fähigkeit. Auflösung: IMMER_QWEN_MODEL → Qwen2.5-1.5B-Instruct
  Snapshot → 0.5B → offline. transformers-5.x-BatchEncoding abgefangen.
  Erste Worte des Wesens: „Hallo! Wie kann ich dir helfen?"

### Kopf (Intent + Gedächtnis)

- `intent.py` — regelbasierter Router: TEACH / RECALL / MATH / STATUS / CHAT.
  Inspektierbar, keine Magie.
- `memory.py` — `SpanStore`: „merke: …" lehrt, „was weißt du über …" ruft ab.
  Wissen liegt außerhalb der Gewichte (die Bibliothek wächst durchs Leben),
  überlebt Neustarts.

### Suite (Metriken)

- `suite.py` — `Metrics` schreibt bei jedem Turn `status.json` (Maschinenfutter:
  vollständiger Snapshot) und hängt an `metrics.jsonl` (Audit/Graphen).
- Dashboard: stdlib-HTTP auf :8787 — `/status` = JSON, `/` = dunkle UI mit
  Live-Karten (Leben, Lernen, Gehirne, Erinnerung, Absichten), Poll alle 2 s.

### Der Scharfsteller-Lauf (live, Mac)

```text
chat      → [qwen] antwortet (1.5B-Instruct, MPS)
merke: …  → (gemerkt: immer heisst mein integrationsprojekt)
was weißt du über sterne? → Überraschung! updates: 1 (Gradientenschritt!)
John has 5 apples…        → [exact] 5
/sleep    → {'replayed': 2, 'sleeps': 1}
/state    → tokens 253 · loss_ema 5.79 (Start: 5.99) · alle Zähler gefüllt
```

### Deutung + nächste Frage

Der Organismus hört, lernt vom Überraschenden, schläft, erinnert Gelerntes,
rechnet exakt und redet darüber — und sein Leben ist als JSON ablesbar.
Zwei Anomalien mit Wert: (1) Das Qwen-1.5B-Weltwissen ist fuzzy („Stellaren",
falsche Meeresszahlen) — genau die Lücke, die Donor-Bibliothek + Denkmodell-
Graft schließen sollen; (2) Gesprochenes wird gestreamt, aber nicht automatisch
als Fakt gespeichert — Auto-Extraktion aus CHAT in Spans ist der nächste Hebel
(Hebel 2: Donor→Index). Danach: Council mit mehreren Qwen-Rollen live schalten.

## [0.3.0] — 2026-08-21 — Alles eingepflanzt, alles verdrahtet, erster Lebenszyklus

**Ziel:** Originale unangetastet, alles Arbeitsfähige ins Repo kopiert und
miteinander verbunden — ab jetzt wird hier gefrickelt, nicht mehr gesucht.

### Eingepflanzt (Vendoring, Originale bleiben read-only)

- `vendor/o1state/{src,reference}/` — der komplette o1-State-Quellstamm
  (137 + 4 Dateien, Spiegel der Original-Pfadlogik `SRC`/`REF`, damit die
  Importkette unverändert schließt: streaming_train → moebius_scan_* →
  length_extrap_v2 → width_fix)
- `src/immer/cognition/fertig/_vendor/fertig/` — das komplette FERTIG-Paket
  (53 Dateien, 51k Zeilen); Adapter nutzt Vendor standardmäßig
  (Reproduzierbarkeit schlägt Environment-Glück), `IMMER_FERTIG_ROOT`
  bleibt als Override

### Neu verdrahtet

- **O1StateStream** (`runtimes/o1_state/adapter.py`) — ECHTE Brücke: der
  Lebensstrom ist ein byte-level StreamingNoPELM; jede Nachricht wird
  Erfahrung, per-layer Z trägt das kontinuierliche Leben, Sidecar-Datei
  = portables Wesen. Determinismus per Seed gemessen.
- **Council** (`council.py`) — BO3-Muster als Baustein: k Gehirne stimmen ab,
  Mehrheit gewinnt, Abstinenz ist ein Ergebnis, Hirntod zählt als Enthaltung,
  nicht als Kollaps.
- **CLI** — `immer serve` (Leben im Terminal: labern, /state, /say, /quit),
  `immer organs list|mount --manifest`, doctor prüft jetzt Vendor+Bridge.

### Der Abnahmetest: bestanden

```text
LEBEN 1: "hallo kleines wesen" → 19 Tokens Strom, exact: "weiß ich nicht"
         "John has 5 apples and 3 oranges…" → [exact] 5   (FERTIG vendored)
         /state → turns 2, tokens 81, loss_ema 5.8598
NEUSTART: "turns alive: 2" — gleicher loss_ema, gleicher Token-Stand.
```

Kill & Restart & erinnert sich — die Substrat-Zusage aus 0.2.0 ist eingelöst.

### Deutung + nächste Frage

Das Gerüst steht komplett: Identität (Stream+Port), Exakt-Wissen (FERTIG),
Können (OrganBank), Denken (CRSA/Council) — alle verdrahtet, 30 Tests grün.
Die offene Frage ist keine Architektur-Frage mehr, sondern eine Zucht-Frage:
**Der Organismus streamt, aber er lernt noch nicht aus dem, was er streamt
(Plastizität aus, eval-only). Wann bekommt der Stream seinen Surprise-Gate +
Schlaf-Zyklus aus portable_organism.py, und wann spricht das erste Denkmodell
statt des Echo-Munds?**

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
