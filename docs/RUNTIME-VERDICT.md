# Runtime-Urteil

Stand: 2026-08-22. Dieses Dokument trennt laufenden Code, Messung,
Negativergebnis und offene Grenze. Es ersetzt keine Zahl durch Architekturprosa.

## Kompaktes Urteil

Der lokale Exact-PoC läuft end-to-end: Ein eingefrorener A1-Host, vier
digest-geprüfte Organe, der feste CRSA-Rollenmix und FERTIG sind über genau
einen `exact_math`-Besitzer verbunden. Der Pfad trainiert beim Start nichts,
verifiziert Host und Organe vor dem Laden und enthält bekannte Fallback-Fehler
durch Abstinenz.

Die eigene Attention ist dabei real im Ausführungspfad, aber das Experiment
zeigt nur einen positiven **Kontextrouter**, keinen spezifischen Vorsprung von
CRSA gegenüber gewöhnlicher kausaler Softmax.

## Reproduzierbare Läufe

```bash
PYTHONPATH=src python -W error::ResourceWarning -m unittest discover -s tests
PYTHONPATH=src python -m immer doctor --deep
PYTHONPATH=src python -m immer eval
PYTHONPATH=src python scripts/crsa_route_eval.py \
  --router-out /tmp/immer-crsa-router-repro.json
cmp /tmp/immer-crsa-router-repro.json manifests/crsa_router_v1.json
```

Lokaler Akzeptanzstand:

- SHIP-v6: 152/152 Antworten und 152/152 Routen;
- Host: SHA-256
  `84f7ac90375668067f348421c581ca60cf3f27dbdb447ae13248dbcc85ea251c`;
- vier Organe: äußerer SHA-256 plus interner State-Digest geprüft;
- Runtime-Training: keines;
- CRSA-Routerzustand: semantischer Digest
  `561db8fc50ea029288f318eb9f7c206bd5c007757dbdd60843f1a03990efd819`;
- reproduzierte Routerdatei: byte-identisch zur committed Datei.

Die 152 Fälle sind die kanonische SHIP-v6-Arithmetiksuite, nicht GSM8K und
nicht 152 allgemeine Sprachaufgaben. Die 30 Textfenster gehören ausschließlich
zur separaten Routermessung.

## Feste CRSA-Struktur

Der deployte Rollenmix wird nicht pro Anfrage neu erfunden:

```text
Head 0  Local
Head 1  Local
Head 2  Balanced
Head 3  Free = bit-exakte kausale Softmax
slope = 0.8
diagonal debit = 3
```

Gemessener Split nach Entfernung aller 48 überlappenden Arithmetikfälle aus
der Kalibration:

| Feature/Control | Balanced Accuracy | Fehler |
|---|---:|---:|
| A1 + CRSA-Residual | 1,000000 | 0/182 |
| Roh-A1 | 0,983333 | 1/182 |
| A1 + kausale Softmax | 1,000000 | 0/182 |
| 32 permutierte Label-Placebos | Mittel 0,510328; Max 0,748026 | — |

Zusätzliche Invarianten:

- Zukunftsmasse: exakt `0.0`;
- Free-Head: bitgleich zur Referenz-Softmax;
- Streaming- und stateless-A1-Features: maximale Abweichung `0.0`;
- stateless Deployment: 182/182.

Erlaubter Claim:
`POSITIVE_CONTEXT_ROUTER__CRSA_NOT_UNIQUE_VS_SOFTMAX`.

Nicht erlaubter Claim: CRSA sei in diesem Experiment besser als Softmax.

## Organe und R17

R17 bedeutet hier nicht „Sibling-Modell per Least Squares in den Donorraum
projizieren“. Dieser Cross-Model-Weg war bereits negativ. Der gültige
Kristallisationsschritt ist:

```text
trainieren
  → Invariante messen
  → Fit/R² als strukturellen Verifier benutzen
  → exakte Struktur einsetzen
```

Das ist im Decimal-Organ direkt sichtbar: Der gelernte Carrier dekodiert die
Ziffern, danach übernimmt die exakte Rekurrenz `h ← 10h + v`; Algebra und
Digit-Word-Emission sind strukturell. Das Z3-Artefakt implementiert
`(a + b) mod 3` und wird deshalb ausschließlich als `z3sum` veröffentlicht —
nicht fälschlich als gewöhnlicher Modulo-Operator.

## FERTIG-Grenze

FERTIG besitzt Grounding, Bindings, semantische Pläne, Skills und Verifikation.
S3 und FERTIG sind keine zwei konkurrierenden Registry-Besitzer; sie sind
private Kinder der `ExactCascade`.

Ein bekannter Fehler des isolierten FERTIG-Solvers war:

```text
John has 5 apples and buys 3 more. How many apples?
```

Der alte Fallback konnte `5` liefern. Die Kaskade prüft additive/subtraktive
Mengenänderungen deterministisch und hält eine falsche oder mehrdeutige Antwort
zurück. Desktop-, Recorder- und Mutationspfade benötigen explizit injizierte
Backends; ohne diese endet FERTIG bei `needs_input`, nicht bei einer erfundenen
Ausführung.

## WorldStream

Der Streamer kann Safetensors-Inventare und exakte 2D-Zeilenbereiche lokal
oder per HTTP Range lesen. Sein Vertrag umfasst:

- Preflight vor jedem Bytebudget-Charge, kein Überschwingen;
- inklusive HTTP-Ranges ohne Off-by-one;
- BF16→FP32 und lazy Torch-Bridge;
- atomare Inventory-/Range-/File-Caches;
- SHA-256-Prüfung bei Resume;
- Cachekorruption als Fehler statt stiller Refetch;
- kein Instanziieren des Donormodells.

Er ist bewusst nicht automatisch mit dem Answer-Pfad verbunden. Das wäre nach
dem negativen Value-Sketch-Experiment ein unbelegter Mechanismussprung.

## Negative Ergebnisse bleiben negativ

1. Statisches Embedding-Mittel → `gate_proj` → Value-Sketch:
   24 % gegen 32 % Placebo bei 138 KB/Frage. Mechanismusdiagnose: Das statische
   Mittel ist nicht die RMSNorm-te Post-Attention-Verteilung, die `gate_proj`
   live sieht. Nicht deployt.
2. Cross-Model-Least-Squares-Projektion als angebliches R17:
   bereits unter Placebo; nicht wiederholt.
3. CRSA-Spezifität auf dem neuen Router-Split:
   nicht gezeigt, weil kausale Softmax bindet.

## Noch offen

- Die Routermessung braucht breitere, fachlich schwierigere Splits, bevor ein
  CRSA-vs-Softmax-Claim erneut geprüft werden darf.
- FERTIGs vollständiger Weltgraph ist extern und muss mit
  `IMMER_FERTIG_GRAPH` explizit gesetzt werden.
- FLCA und QAD sind externe kanonische Stränge, keine im heutigen Core
  vorgetäuschten Laufzeitkomponenten.
- Der lokale A1-only-Hugging-Face-Export ist checkout-isoliert golden
  (152/152 Antworten und Routen). Ein öffentlicher Upload wartet weiterhin auf
  die geklärte Lizenzkette. Ein fehlendes Root-`LICENSE` wird nicht durch eine
  frei erfundene MIT/Apache-Angabe ersetzt.
