# Benchmarks

Stand: 2026-08-22. Nur Werte mit einem heute ausführbaren Reproduktionspfad
gelten als aktueller IMMER-Befund. Historische Forschungsanker werden nicht
als Runtime-Golden ausgegeben.

## 1. Lokale Akzeptanz

```bash
PYTHONPATH=src python -W error::ResourceWarning -m unittest discover -s tests
PYTHONPATH=src python -m immer doctor --deep
PYTHONPATH=src python -m immer eval
```

`immer eval` lädt den Frozen-A1-Host und die benötigten Organe kalt, prüft alle
Digests und führt die 152 kanonischen SHIP-v6-Fälle aus. Passkriterium:

- `correct == cases == 152`;
- `route_correct == cases == 152`;
- `no_training is true`;
- Host-Digest entspricht dem Manifest;
- Router-Digest entspricht dem persistierten Head.

Die Suite besteht aus Addition/Subtraktion, kleinem multiplikativem
Log-Carrier, `z3sum` und dem kristallisierten Dezimalpfad. Sie ist kein
GSM8K-Subset.

## 2. Routermessung

```bash
PYTHONPATH=src python scripts/crsa_route_eval.py \
  --router-out /tmp/immer-crsa-router-repro.json
cmp /tmp/immer-crsa-router-repro.json manifests/crsa_router_v1.json
```

Protokoll:

- Frozen A1, kontextuelle Layer-0-Scan-Zustände;
- feste Rollen `2 Local + 1 Balanced + 1 Free`;
- Feature `[raw_last | role_complete_last]`;
- 64 Arithmetik- und 64 Text-Kalibrationsfälle;
- alle 48 Arithmetik-Duplikate gegenüber dem Evalsplit entfernt;
- 152 Arithmetik- plus 30 Text-Evalfälle;
- Ridge `10.0`, Seed `7`;
- 32 Kontrollen mit permutierten Kalibrationslabels;
- Roh-A1- und kausale-Softmax-Ablation.

| Pfad | Balanced Accuracy |
|---|---:|
| CRSA-Residual | 1,000000 |
| Roh-A1 | 0,983333 |
| kausale Softmax | 1,000000 |
| Label-Placebo, Mittel | 0,510328 |
| Label-Placebo, Maximum | 0,748026 |

Urteil: Kontextsignal positiv; CRSA-spezifischer Vorteil gegenüber Softmax
nicht gezeigt.

## 3. Negativkontrollen

Die Resultate in `results/router_v2_stage1.json` und
`results/router_v2_stage2.json` bleiben Teil der Beweiskette:

- Stage 1: Themenstruktur liegt messbar vor allem in mittleren Donor-Layern;
- Stage 2: statisches Embedding-Mittel plus Value-Sketch erzielt 24 % gegen
  32 % Placebo bei 138 KB/Frage;
- Konsequenz: Stage 2 ist falsifiziert und nicht im Answer-Pfad aktiv.

Ein negatives Ergebnis darf nur durch ein neues, vorab benanntes Protokoll mit
passendem Eingabeverteilungsmechanismus erneut geöffnet werden.

## 4. FERTIG-GSM8K-Vollsplit

```bash
PYTHONPATH=src python scripts/bench_gsm8k.py \
  --failures-jsonl results/bench_gsm8k_failures.jsonl
```

Der vollständige vendorte Testsplit wird nicht auf Solve-Rate reduziert. Der
Report enthält alle 1.319 Items, stabile IDs, Dataset-/Harness-Digests sowie
die getrennte Partition `correct / abstained / incorrect / error`:

| correct | abstained | incorrect | error | Coverage beantwortet |
|---:|---:|---:|---:|---:|
| 1.060 | 259 | 0 | 0 | 80,36 % |

Ein erster Vollauf fand 17 falsche Antworten, sämtlich aus dem ungeprüften
Operationsketten-Template-Fallback; dieser Pfad traf kein einziges Mal
korrekt. Die Templates bleiben als Kandidaten-Instrument erhalten, sind aber
ohne unabhängigen Strukturbeweis aus dem Unified-Answer-Pfad quarantäniert.
Alle 17 Fälle sind `must_abstain`-Regressionen. Der harte Gate lautet weiterhin
`incorrect + errors == 0`, nicht bloß hohe Accuracy auf beantworteten Fällen.

## 5. Streaming-Verträge

Die automatisierte Contract-Suite misst keine Modellqualität, sondern
Sicherheits- und Ressourceninvarianten:

- Budget wird vor dem Lesen reserviert und niemals überschritten;
- lokale und HTTP-Ranges liefern exakt die angeforderte Bytezahl;
- Cache-Resume bewegt null Quellbytes und prüft SHA-256;
- der 12-GiB-Cache bleibt auch während atomarer Same-Key-Rewrites unter der
  Grenze und erholt sich nach unterbrochenen Paar-Writes als Cache-Miss;
- beschädigte Caches werden nicht still neu geholt;
- BF16-Zeilen werden deterministisch dekodiert;
- `rows_torch()` erzeugt einen gradientenfreien Tensor, ohne ein Donormodell
  zu konstruieren.

## 6. DeepSeek-V4-Flash: lokaler Decoderstatus

Der revisionsgebundene Hauptdecoder von
`deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b...` läuft lokal direkt aus den
originalen Safetensors-Ranges. Der aktuelle Vertrag umfasst alle 43 Layer,
FP8/FP4-QDQ, Hash- und Score-Routing, sechs aktive plus Shared Expert,
HyperConnections, native Fenster-/Kompressor-/Indexer-Attention, globalen
LM-Head und autoregressiven KV-Zustand. DSpark-MTP ist noch nicht im
Ausführungspfad.

Zwei abgeschlossene Integrationsmessungen auf M4/16 GB:

| Lauf | Ausgabe | Layer-Forwards | Quellbytes | Zeit | Urteil |
|---|---|---:|---:|---:|---|
| Token-ID 0, Graft aus | `#` | 1 | 1,858 GB | 636,9 s | vollständiger Decoder-Smoke |
| Token-ID 0, CRSA Layer 21, 2 Tokens | `# ` | 2 | 3,115 GB | 1.747,5 s | stateful Graft technisch aktiv |

Der zweite Lauf hält 17,84 MB Attention-State und den Cache bei
12.881.453.756 von 12.884.901.888 Bytes; 891 LRU-Evictions überschritten die
Grenze nicht. Das ist noch kein Qualitäts- oder Frontier-Paritätsclaim: Die
Eingabe war eine explizite Diagnose-ID. Ein offizieller Mehrtoken-Chatprompt
und danach feste, gepaarte `off / CRSA / softmax / shuffle`-Items sind der
nächste inhaltliche Gate.

## 7. Lebensstrom-Verträge

Die Tests des O1-Lebensstroms prüfen:

- gleicher Seed plus gleiche Erfahrung ergibt gleichen Loss;
- Snapshot/Restore setzt Leben und Zustände fort;
- nicht überraschende Chunks tragen keinen Autograd-Graph weiter;
- überraschende Chunks werden aus demselben detached Eingangszustand neu
  gerechnet und aktualisieren Parameter;
- Plastizitätszustand und Replay-Puffer überleben Neustarts;
- `/sleep` replayt und leert den Puffer.

Das ist noch kein veröffentlichter Milliarden-Token-Loss-Report. NLL 8,6656
bleibt ein historischer Host-Anker im SHIP-Manifest, nicht eine in diesem
Akzeptanzlauf neu gemessene Sprachmodell-Qualität.

## 8. Nächste belastbare Messungen

1. Router auf einem größeren, nicht synthetisch eng getrennten Text/Math-Split,
   erneut CRSA gegen kausale Softmax und Label-Placebos.
2. Lebenskurve mit fixem held-out Byte-Stream: NLL vor/nach Surprise-Updates,
   Post-Sleep-Delta und State-Größe über die Zeit.
3. DeepSeek-V4 auf einem festen kleinen Qualitätssplit: offizielles Encoding,
   kandidatengestütztes Scoring, vollständige Fehlerdenominatoren und gepaarte
   Graft-Ablationen bei gleicher Cache-/Bytebilanz.
4. WorldStream-Budgetkurve nur mit einem neuen, kontextuell korrekten
   Retrievalmechanismus; kein Revival des statischen Value-Sketches.
