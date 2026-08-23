# CHANGELOG

Alle Änderungen an IMMER. Jeder Eintrag endet mit Deutung + nächster Frage —
nie mit der nackten Zahl.

## [0.7.1] — 2026-08-23 — Quellenkarte verdichtet, Graph/Weights sauber getrennt

- Die öffentliche README nennt den `.causal`-Pfad jetzt explizit als
  Wiring-/Inferenzschicht über lokalen Gewichten statt als stillen
  Tensor-Umschreiber.
- `docs/research.md` trägt jetzt die Primärquellen für MoE-Routing,
  DejaVu/kontextuelle Sparsity, Reservoir Computing, Markov-Routing und
  safetensors in einem klar getrennten Abschnitt.
- Die Repo-Verweise auf `DT-Foss/FERTIG`, `DT-Foss/o1-state` und
  `DT-Foss/dotcausal` sind jetzt direkt aus der README erreichbar.

### Deutung + nächste Frage

Die Oberfläche ist jetzt schärfer: lokales Gewicht, externe Quelle, explizite
Verkabelung. Die nächste Frage ist nicht mehr, wo die Linien verlaufen,
sondern welche davon als Nächstes golden verdichtet werden.

## [0.7.0] — 2026-08-22 — Der lokale Pfad läuft; Verpackung kommt zuletzt

**Ziel:** Die vorhandenen Mechanismen zu einem reproduzierbaren lokalen
Runtime-Pfad verbinden, ohne negative Experimente umzudeuten oder externe
Originale zur Laufzeit zu importieren.

### Frozen Exact-Core

- Eigener, state-dict-kompatibler A1-Produktionskern statt `sys.path`-Importen
  aus Research-/Vendorbäumen.
- Vier SHA-adressierte SHIP-v6-Organe (`arith-dual`, `mul-log`, `z3-circle`,
  `decimal-crystal`) werden vor `torch.load(weights_only=True)` außen und
  innen geprüft. Es gibt kein Runtime-Training.
- Die kanonische Suite erreicht 152/152 Antworten und 152/152 Organrouten.
  `z3-circle` ist korrekt nur als Addition in `Z₃` veröffentlicht; der
  Dezimalpfad setzt nach dem gemessenen Carrier exakt `h ← 10h + v` ein.

### Eigene Attention und ehrlicher Routerclaim

- Der Online-Pfad nutzt die eigene, exakt kausale CRSA-Attention mit festem
  Programm: `2 Local + 1 Balanced + 1 Free`, Steigung `0.8`,
  Diagonal-Debit `3`. Der Free-Head ist bitgleich zur kausalen Softmax.
- Der persistierte Kontext-Head erreicht Balanced Accuracy 1,000000 gegen
  Roh-A1 0,983333 und 32 Label-Placebos mit Mittel 0,510328/Maximum 0,748026.
  Die kausale-Softmax-Ablation erreicht ebenfalls 1,000000. Der Befund ist
  deshalb `POSITIVE_CONTEXT_ROUTER__CRSA_NOT_UNIQUE_VS_SOFTMAX`, kein
  CRSA-Überlegenheitsclaim.

### FERTIG und Komposition

- `ExactCascade` ist der einzige Registry-Besitzer von `exact_math`: S3 wird
  zuerst ausgewertet, FERTIG verifiziert oder fällt zurück, und unbekannte
  beziehungsweise widersprüchliche Fälle enden bewacht in Abstinenz.
- Der geerdete FERTIG-Adapter bindet Graph, Bindings, semantische Pläne,
  Skills und Verifier ein. Desktop-, Recorder- und Mutationspfade benötigen
  explizite Backends/Gates; ohne sie wird keine Aktion erfunden.

### Lebensstrom, Artefakte und WorldStream

- Der persistente Surprise-/Sleep-Lebensstrom bleibt strikt vom Frozen Host
  getrennt; atomare Sidecars bewahren Modell-, Optimizer-, Scan- und
  Replayzustand. Die Produktionsruntime ändert keine globalen
  Torch-/MPS-Einstellungen.
- Artefakte werden aus expliziten read-only Quellen atomar importiert und nur
  nach Manifest-SHA verwendet. Manifeste werden als Paketressourcen ins Wheel
  übernommen.
- WorldStream ist als begrenzte lokale/HF-Tensorquelle integriert: exakte
  Safetensors-Ranges, Preflight-Bytebudget, BF16-Dekodierung und atomarer,
  SHA-geprüfter Resume-Cache. Er lädt kein Donormodell und ist nicht
  automatisch mit dem Antwortpfad verbunden.

### Negative Ergebnisse bleiben geschlossen

- Statisches Embedding-Mittel → `gate_proj` → Value-Sketch: 24 % gegen
  32 % Placebo bei 138,4 KB/Frage. Nicht deployt.
- Cross-Model-Least-Squares-Projektion: negativ und kein R17. R17 bleibt
  trainieren → Invariante messen → Fit/R² als Verifier → exakte Struktur.

### DeepSeek-Transport v5 und Handoff-Triage

- `Streamer.raw_bytes_many()` vereinigt ausschließlich kalte, exakt
  angrenzende Leafs. Cache-Keys bleiben range-exakt; Scalar↔Batch-Reuse,
  Single-Flight, readonly Views, aggregierte Vorabreservierung und
  thread-lokale Byte-Receipts sind getestet. Eine bekannte Quellidentität bei
  fehlender Cache-Identität scheitert jetzt geschlossen.
- Der Pager kann zwei physisch angrenzende Experts als gemeinsamen Batch lesen,
  hält den gesamten Owner bis zum letzten Consumer resident und zählt
  Envelope-, Quellbyte-, Gap- und Cancellation-Receipts genau einmal. Die
  Produktionsidentity bleibt dynamisch q3/Width 1; `disabled` und der
  explizite Pair-Kandidat versiegeln ihre tatsächliche Transportpolicy.
- Neuversiegeltes MPS/BF16-A/B: q3 gegen aus bleibt bitidentisch, zeigt aber
  keinen stabilen Latenzgewinn (cache-resident 0,973489×, kalt 0,726000×).
  Adjacent-Pairs reduziert kalt sechs auf vier Envelopes ohne ein Zusatzbyte
  und erreicht gegen q3 über zehn Paare 1,161604× im Mittel bei 6/10
  Paar-Wins; cache-resident ist 1,005050× praktisch neutral. Da der direkte
  Kontrast gegen `off` noch fehlt, bleibt Width 2 explizites Instrument statt
  Default.
- DeepSeeks FERTIG-Hungarian ist gegen die echten 259 Abstinenzen geprüft und
  als direkte Integration verworfen: 65 Fälle besitzen im Legacy-Parser kein
  Frageziel, 171 kein zum Ziel passendes Quantity, 20 eine unvollständige
  Relation und drei treffen einen bewussten Geld/Zeit-Guard. Der saubere
  Structural-Pfad sieht 144 ungeklärte Pronomen/Coreference-Fälle und 115
  fehlende Operationsgrammatiken. Die vorgeschlagene M-fache Entity-
  Replikation zerfällt mathematisch in unabhängige Spalten-Argmins und schafft
  keine globale Zuordnung. Hungarian bleibt nur für künftig explizit
  extrahierte, exklusive Slots mit IR-/Eindeutigkeitszertifikat zulässig. Der
  gold-label-freie Lauf ist mit allen 259 Item-Diagnosen und Parser-/Harness-
  Hashes in `results/fertig-abstention-audit.json` versiegelt.
- Das exakte LM-Head-Request-A/B ist auf dem realen gepinnten Head gelaufen:
  bei unveränderten 127 Rechenblöcken bündelt der Kandidat je acht
  benachbarte Leafs und reduziert so 127 auf 16 physische Requests. Beide
  Arme lesen exakt 1.059.061.760 Byte; vollständiger FP32-Logitstrom sowie
  Top-k-Werte und -IDs sind bitidentisch. Über vier alternierende kalte Paare
  gewinnt der Kandidat 4/4 mit Einzel-Speedups von 1,375606× bis 2,410811×
  und einem gepaarten Median von 1,445790×; die Mittelzeiten fallen von
  211,5756 s auf 125,6880 s. Das ist wiederholte Latenzevidenz, bleibt bis zum
  End-to-End-Inhaltsgate jedoch opt-in; Produktionsdefault ist weiterhin
  Batchbreite 1.
  Der versiegelte Beleg steht in
  `results/deepseek-v4-head-range-network-smoke.json`.
- Der externe „R25-Format-Brücke“-PoC ist kein R25-/CRSA-Beleg: Er
  baut die positive Mid-Depth-Struktur synthetisch ein und dividiert KL durch
  SNR, sodass der behauptete 18×-Wert per Amplitudenskalierung frei beweglich
  ist. Als neuer Versuch bleibt nur ein echter klassenkonditionierter
  Donor-KL-Readout mit vorab getrenntem Train/Held-out und Label-Shuffles.
- Wave3-T11 ist ebenfalls kein Integrationskandidat: Der Parallelarm teilt
  einen bereits gemittelten Gradienten nochmals durch `N`, während der
  Sequenzarm privilegierte Zwischenziele des Lehrers erhält. Nach Korrektur
  der doppelten Skalierung bleiben im synthetischen Aufbau 5,19 % statt
  behaupteter 98 % Fehlerreduktion; der Vergleich bleibt durch die Ziele
  unfair. Der konstruktive 3-Zonen-Router ist ein harter NO-GO, weil dieselbe
  synthetische `expert_quality` in Gate-Normen und Embeddings eingespeist und
  anschließend als Score/Oracle zurückgelesen wird. Entkoppelt bleiben
  +4,67 % über Random und 36,66 % des Oracle; V4 besitzt zudem keine solchen
  Expert-Embeddings und nutzt eine andere offizielle Routinggleichung.
- Aus dem Resthandoff werden nur zwei reale Messungen geöffnet: eine passive
  Residuen-Spur des tatsächlichen HC-Sinkhorn-Kerns und ein korrigiertes,
  ausschließlich auf den CRSA-Graft begrenztes Per-Head-Temperatur-A/B mit
  exakter Kausalmaske und permutiertem Placebo. Kausal angepasste Birkhoff-
  Größen und ID/effective rank sind reine Offline-Diagnostik. Zeno-Schedule,
  Replica-MoE, Live-η-Gate, Ginibre-Hurst, Mask-Recycling, SK1 sowie ID als
  Dimensionierungsregel bleiben geschlossen; ebenso Möbius-/Sinkhorn-
  Shortcuts, PPM-Head-Ersetzung und compute-paralleler Shared Expert ohne
  Gleichheits-/Memory-Beweis.

### Deutung + nächste Frage

Der belastbare Gewinn dieser Version ist ein lokaler, digest-geprüfter und
bewacht zusammengesetzter PoC. Erst nach dessen Golden Run kam die Verpackung:
Der A1-only-Offline-Export erzeugt 6.857.284 Byte Safetensors statt des
74.335.278-Byte-Forschungscheckpoints, kopiert vier Organe bytegleich und
prüft sich checkout-isoliert erneut mit 152/152 Antworten und Routen. Er
enthält bewusst keinen Hub-/Login-/Uploadpfad. Als Nächstes werden breitere
Router-/FERTIG-Kontraste golden gehalten; eine öffentliche Veröffentlichung
bleibt allein durch die ungeklärte Lizenzkette gesperrt.

## [0.6.1] — 2026-08-21 — 19 s sind unbrauchbar: einmal zahlen, immer gratis

**Anlass:** Davids Rüge: „19 s für eine Antwort ist literally unbrauchbar."
Er hatte die Geschwindigkeits-Maschinerie längst gebaut (Organe <1 ms,
Kaskade, Verifier) — ich hatte den langsamsten Pfad verdrahtet: jeder Turn
ein frischer 27B-Call.

### Die Stufen-Kaskade (`_answer_via_cascade`)

```text
Tier 0  [bibliothek]   ~0 ms   Karte existiert → sofort
Tier 1  [donor Xs]     sync    nur wenn nichts Besseres da — UND: jede
                                gesprochene Antwort wird SOFORT Karte
Tier 2  Veredelung     Hintergrund (Donor verbessert Entwürfe ohne zu blockieren)
```

- `library.capture()` — jede Donor-Antwort wandert nach dem Sprechen in die
  Bibliothek; Wiederholungen kosten danach exakt 0 ms.
- Prefix-Stemming pro Wort (4 Zeichen) im Recall — „was **sind schwarze
  löcher**?" trifft die Karte von „was ist ein **schwarzes loch**?".
  Gemessen: beide Varianten → 0 ms.
- Latenz pro Antwort gemessen und angezeigt (`[donor 64.3s]`, `(0 ms)`),
  `last_latency_ms` in status.json; MATH-Abstinenz fällt jetzt in dieselbe
  Kascade statt Endstation „weiß ich nicht".

### Der Geschwindigkeits-Beweis (live)

```text
Q1 "was ist ein schwarzes loch?"   → [donor 64.3s]
Q2 identisch                        → [bibliothek] 0 ms
Q3 "was sind schwarze loecher?"     → [bibliothek] 0 ms   (Stemming!)
Q4 "wer war ada lovelace?"          → [donor 64.9s] → ab jetzt auch 0 ms
```

Amortisationskurve: jede Frage genau einmal teuer, für immer gratis —
die Bibliothek IST der Antwort-Cache mit Herkunfts-Stempel.

### Deutung + nächste Frage

Donor-Tempo auf beast schwankt (19–66 s je nach Last der Mitmieter) —
egal, denn er ist nur noch beim ersten Mal im kritischen Pfad. Die echte
Zahl ist jetzt die Trefferquote von Tier 0: bei 0 % Erstdruck bleibt jedes
Gespräch langsam; je mehr gelebt wird, desto schneller wird es. Nächste
Frage: Auto-Ernte im Leerlauf (das Wesen rechnet beliebte Themen vor,
während niemand fragt) und Ship-v6-Organe als Tier −1 (arithmetisch antwortet
es dann in Millisekunden OHNE jede Karte).

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
