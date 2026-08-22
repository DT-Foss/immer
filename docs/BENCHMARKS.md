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

Zusätzlich beherrscht FERTIG nun endliche affine Rekurrenzen mit expliziten
Indexgrenzen und exaktem Fraction/RREF-Zertifikat. Das ist eine neue
strukturelle Fähigkeit; der Vollsplit wurde dadurch nicht nachoptimiert und
bleibt bei 1.060/1.319 korrekt sowie 0 falschen Antworten.

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
exakte blockskalierte MXFP8/FP4-Dekodierung mit FP32-Akkumulation, Hash- und
Score-Routing, sechs aktive plus Shared Expert, HyperConnections, native
Fenster-/Kompressor-/Indexer-Attention, globalen LM-Head und autoregressiven
KV-Zustand. DSpark-MTP ist noch nicht im Ausführungspfad.

Layerwise Proof-Schema v4 bindet Modellrevision, alle DeepSeek-Laufzeitquellen,
beide importierten Package-Exports, das konkrete Layerwise-CLI, Streamer,
HF-Range-Reader, Snapshot sowie Python-/Torch-/NumPy-/Safetensors-Versionen vor
dem Resume. Das kanonische Journal v2 ist eine monotone
Hash-Kette des getrennten item-major Benchmark-Harnesses und weist dort
Duplikate, Fremdeinträge, Mutation sowie gebrochene Verkettung ab; ein
abgerissener letzter Datensatz ist deterministisch reparierbar. Layerwise
selbst committet content-addressed BF16-Generationen über atomare Manifeste.

Zwei abgeschlossene Integrationsmessungen auf M4/16 GB:

| Lauf | Ausgabe | Layer-Forwards | Quellbytes | Zeit | Urteil |
|---|---|---:|---:|---:|---|
| Token-ID 0, Graft aus | `#` | 1 | 1,858 GB | 636,9 s | vollständiger Decoder-Smoke |
| Token-ID 0, CRSA Layer 21, 2 Tokens | `# ` | 2 | 3,115 GB | 1.747,5 s | stateful Graft technisch aktiv |

Der zweite Lauf hält 17,84 MB Attention-State und den Cache bei
12.881.453.756 von 12.884.901.888 Bytes; 891 LRU-Evictions überschritten die
Grenze nicht.

Der erste vollständige Layerwise-Inhaltslauf über vier feste
High-School-Geography-Items ist jetzt ebenfalls abgeschlossen:

| Modus | Layer | korrekt | Accuracy | beobachtete Wandzeit | Urteil |
|---|---:|---:|---:|---:|---|
| `off` | 43/43 | 3/4 | 0,75 | 8.644,9 s | positiver Content-Smoke |

Drei Entscheidungen stimmen, eine ist `B` statt erwartet `A`. Manifest,
Resultat und finales 6.291.680-Byte-BF16-Objekt sind kreuzgebunden und
selbstversiegelt; alle 16 Runtime-Quellen entsprechen Commit `d0a1dc4`. Der
Receipt steht in
`results/deepseek-v4-mmlu-off-4-exact-v4-window2x3-receipt.json`. Die Zeit ist
nur `mtime - started_at` und ausdrücklich kein kryptografischer
Performancebeleg; vier Items sind kein allgemeiner MMLU- oder
Frontier-Paritätsclaim.

Der konservative Expert-Prefetch `exact-router-window-q3-a2/v2` startet erst
nach der offiziellen Routerentscheidung und plant alle ausgewählten Experts
vor dem ersten Read. Er reiht höchstens drei Futures sofort ein, lässt durch
zwei I/O-Worker aber höchstens zwei Reads gleichzeitig laufen und hält drei
residente/inflight Experts, 14 MiB pro Expert und 48 MiB insgesamt.
Offizielle 3-Expert-A/Bs auf MPS/BF16 ergaben:

| Cache/Trials | aus | Fenster an | Speedup | Ausgabe | Peak |
|---|---:|---:|---:|---|---:|
| warm, 20 | 0,172745 s | 0,160813 s | 1,074197× | bitgleich | 40.108.032 B |
| aus, 8 | 7,250212 s | 5,244157 s | 1,382532× | bitgleich | 40.108.032 B |

Der aktuelle Remote-Pfad `requests-session-pool-2/v1` least exakt zwei
voneinander getrennte Ein-Verbindungs-Sessions und liest Bodies ausschließlich
begrenzt aus dem Stream. Im No-Cache-Lauf sind das 27,67 % weniger mittlere
Latenz bei 320.864.256 gemessenen Quellbytes. 158 Requests sahen sechs
Connection-Objekte, maximal zwei aktive Leases und danach null; Retries und
Fehler blieben null, der Transport war vor der Versiegelung geschlossen. Die
Netztrials streuen jedoch stark: `off` 4,62–11,53 s, `on` 3,63–6,36 s. Der
frühere unabhängige urllib-Lauf bei Commit `d0a1dc4` erreichte 1,020702×;
Vergleiche über Laufgrenzen bleiben netzwerkbedingt diagnostisch. Alle Outputs
eines Laufs hatten denselben SHA-256; Futures, Peak und Fehlerpfade blieben
innerhalb ihrer Gates. Die versiegelten Rohresultate stehen in
`results/deepseek-v4-exact-prefetch-window-smoke.json` und
`results/deepseek-v4-exact-prefetch-window-network-smoke.json`. Das sind
ausdrücklich keine Qualitäts-, MMLU- oder Frontier-Paritätsbelege; der reale
Layerwise-Lauf entscheidet über den End-to-End-Nutzen.

Der q3/a2-Pfad ist gegen `off` positiv; sein marginaler Vorteil gegenüber q2
ist wegen getrennter Netzläufe noch nur diagnostisch. Der nächste isolierte
Kontrast bündelt exakt benachbarte selektierte Ranges. Anschließend folgt ein
gepaarter `off`/CRSA-Inhaltslauf auf derselben Runtime-Identität.

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

1. Queue-Tiefe drei und benachbarte Range-Coalescings jeweils einzeln gegen
   identische Outputs, Bytes, Peak und kalte Wandzeit falsifizieren.
2. Gepaarter `off`/CRSA-Content-Gate mit offiziellem Encoding, festem Split,
   vollständigen Fehlerdenominatoren und atomarem Proof-Schema v4.
3. Router auf einem größeren, nicht synthetisch eng getrennten Text/Math-Split,
   erneut CRSA gegen kausale Softmax und Label-Placebos.
4. Lebenskurve mit fixem held-out Byte-Stream: NLL vor/nach Surprise-Updates,
   Post-Sleep-Delta und State-Größe über die Zeit.
5. WorldStream-Budgetkurve nur mit einem neuen, kontextuell korrekten
   Retrievalmechanismus; kein Revival des statischen Value-Sketches.

Ein Hugging-Face-Export kommt erst ganz am Ende nach lokalem Golden und
geklärter Lizenzkette; er ist kein aktueller Benchmark-Fokus.
