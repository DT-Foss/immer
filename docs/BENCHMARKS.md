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

## 4. Streaming-Verträge

Die automatisierte Contract-Suite misst keine Modellqualität, sondern
Sicherheits- und Ressourceninvarianten:

- Budget wird vor dem Lesen reserviert und niemals überschritten;
- lokale und HTTP-Ranges liefern exakt die angeforderte Bytezahl;
- Cache-Resume bewegt null Quellbytes und prüft SHA-256;
- beschädigte Caches werden nicht still neu geholt;
- BF16-Zeilen werden deterministisch dekodiert;
- `rows_torch()` erzeugt einen gradientenfreien Tensor, ohne ein Donormodell
  zu konstruieren.

## 5. Lebensstrom-Verträge

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

## 6. Nächste belastbare Messungen

1. Router auf einem größeren, nicht synthetisch eng getrennten Text/Math-Split,
   erneut CRSA gegen kausale Softmax und Label-Placebos.
2. Lebenskurve mit fixem held-out Byte-Stream: NLL vor/nach Surprise-Updates,
   Post-Sleep-Delta und State-Größe über die Zeit.
3. FERTIG-Systemsplit mit expliziten Familien: korrekte Antwort, sichere
   Abstinenz, falsche Antwort. Falsche Antworten sind der harte Fehler.
4. WorldStream-Budgetkurve nur mit einem neuen, kontextuell korrekten
   Retrievalmechanismus; kein Revival des statischen Value-Sketches.
