# Formeln → Hebel (Leverage-Map)

Stand: 2026-08-22. Quelle: `research/formeln/FORMEL-FUNDAMENT.md` plus die
committeten Router-/SHIP-Messungen. Die Formeln bleiben als Kanon erhalten;
`DEPLOYT`, `GEMESSEN`, `NEGATIV` und `OFFEN` trennen ihren heutigen Status.

## WorldStream: vom Moonshot zum begrenzten Instrument

Historischer Befund: Ein Header-/Range-Scan des 27B-Donors übertrug 116,6 MB
von 55,56 GB (0,21 %, amortisiert 2,19 MB/Frage). Das zeigte, dass
Safetensors-Seiten selektiv lesbar sind; es bewies noch keinen belastbaren
Antwortmechanismus.

**DEPLOYT als Infrastruktur:** `src/immer/knowledge/streamer.py` und
`src/immer/knowledge/_hf_source.py` bieten einen lokalen oder revisions-
gebundenen `TensorSource`: Header-Inventar, exakte 2D-Zeilenbereiche, hartes
Bytebudget, atomaren Cache und SHA-geprüften Resume. Der Streamer instanziiert
kein Donormodell und ist nicht heimlich mit dem Antwortpfad verbunden.

Der DeepSeek-Hauptpfad nutzt dieselbe Selektivität jetzt konservativ als
`exact-router-window-q3-a2/v2`: Erst der offizielle Router entscheidet, danach
werden höchstens drei vorab vollständig geplante Expert-Futures eingereiht;
zwei Worker lesen weiterhin höchstens zwei Ranges parallel. Maximal drei
Experts beziehungsweise 48 MiB Rohpayload sind
resident/inflight, jeder einzelne bleibt unter 14 MiB. Konsumreihenfolge,
Werte und serielle FP32-Akkumulation bleiben unverändert.

Der Streamer besitzt zusätzlich einen exakten Multi-Range-Vertrag mit
Leaf-Cache-Reuse, Single-Flight, aggregiertem Hard-Budget und readonly Views.
Das daraus gebaute Adjacent-Pairing bleibt jedoch explizit aus: im
neuversiegelten Lauf ist es cache-resident mit 1,005050× praktisch neutral;
kalt gewinnt es bei identischen Bytes 1,161604× im Mittel und 6/10 Paare
gegen q3. Weil q3 selbst im zeitgleichen Neuversiegeln keinen stabilen Vorteil
gegen `off` zeigt und ein direkter `off`/Pair-Kontrast fehlt, bleibt Width 1
Default.

**NEGATIV als Retrieval-Mechanismus:** Statisches Embedding-Mittel
→ `gate_proj` → Value-Sketch erreichte 24 % gegen 32 % Placebo bei
138,4 KB/Frage. Die Eingabe entspricht nicht den RMSNorm-ten
Post-Attention-Zuständen, die `gate_proj` im Modell sieht. Dieser Shortcut ist
geschlossen; die Tensorquelle bleibt verwendbar.

## DeepSeek-Handoff: verwertbar, bedingt, geschlossen

- **Verwertbar:** begrenztes Keep-alive ist deployt; exaktes Range-Coalescing
  ist als ausgeschaltetes Messinstrument implementiert; Cache-Admission und
  -Identität wurden fail-closed verschärft. Das reale LM-Head-A/B fasst bei
  unveränderten 127 Rechenblöcken je acht benachbarte Leafs zusammen: 127→16
  physische Requests, weiterhin exakt 1.059.061.760 Quellbytes und
  bitidentischer vollständiger Logitstrom samt Top-k. Vier alternierende Paare
  gewinnt der Kandidat 4/4; Einzel-Speedups 1,375606× bis 2,410811×,
  gepaarter Median 1,445790× und Mittelzeiten 211,5756 s gegen 125,6880 s.
  Trotz wiederholter Latenzevidenz bleibt der Default bis zum End-to-End-Gate
  Breite 1. Maßgeblich ist der
  versiegelte Report `results/deepseek-v4-head-range-network-smoke.json`.
- **Falsifiziert für den heutigen FERTIG-Pfad:** DeepSeeks M-fache Entity-
  Replikation macht aus dem vermeintlich globalen Hungarian-Problem nur
  unabhängige `argmin`-Entscheidungen pro Ziel. Die 259 Abstinenzen zerfallen
  tatsächlich in 144 ungeklärte Pronomen/Coreference-Fälle und 115 fehlende
  Structural-Grammatiken. Erst explizit extrahierte, exklusive Slots dürften
  künftig per Minimum-Cost-Matching vorgeschlagen werden; Antwortfreigabe
  verlangt weiterhin eindeutiges Fraction/RREF-Zertifikat.
- **Nicht als R25-Evidenz verwertbar:** Der externe 18×-PoC erzeugt
  Mid-Depth-Gaußprofile synthetisch und setzt KL zu SNR ins Verhältnis. Diese
  dimensionsfremde Kennzahl lässt sich durch reine Amplitudenskalierung frei
  bewegen. Der legitime neue Mechanismus wäre stattdessen ein auf echtem
  Train-Split gebildetes `P(output | task-class)` als KL-Target, bewertet auf
  Held-out gegen All-Layer und Label-Shuffles; er ist noch ungemessen.
- **Bedingt:** Shared-Expert-I/O darf erst nach einem Resident-Peak-Beweis
  überlappen. Compute-Parallelisierung würde die heutige serielle
  Akkumulationssemantik ändern und ist deshalb kein kleiner Transportfix.
- **Wave3 geschlossen:** T11 skaliert im Parallelarm einen bereits
  gemittelten Gradienten ein zweites Mal durch `N` und gibt dem Sequenzarm
  privilegierte Lehrer-Zwischenziele. Korrigiert bleiben im synthetischen
  Aufbau 5,19 % statt 98 % Fehlerreduktion; durch die ungleichen Ziele ist
  auch das kein fairer Kausalbeleg. Der 3-Zonen-Router ist zirkulär:
  `expert_quality` erzeugt Gate-Normen und Expert-Embeddings und wird danach
  als Score/Oracle zurückgelesen. Entkoppelt bleiben +4,67 % über Random und
  36,66 % des Oracle. V4 hat keine solchen Expert-Embeddings und eine andere
  offizielle Hash-/Score-Routingregel. Urteil: NO-GO beziehungsweise harter
  NO-GO.
- **Nur als neue Messung offen:** passive Residuen-Spuren des tatsächlich
  ausgeführten HC-Sinkhorn-Kerns; danach höchstens ein gekennzeichnet
  approximatives Fixed-20/Early-stop-A/B mit Logit-/Choice-Delta. Außerdem
  ein korrigiertes Per-Head-Temperatur-A/B ausschließlich im CRSA-Graft, mit
  exakter Kausalmaske und permutiertem Placebo. Kausal angepasste Birkhoff-
  Größen sowie ID/effective rank bleiben Runtime-sichere Offline-Diagnostik,
  keine Steuerung.
- **Geschlossen oder unbelegt:** Möbius als Ersatz für den offiziellen
  Sinkhorn-Pfad, pauschal 20→3 Sinkhorn-Schritte, PPM als approximativer
  LM-Head-Ersatz, approximatives Cross-Layer-Recycling, Zeno-Schedule,
  Replica-MoE, Live-η-Gate, Ginibre-Hurst, Mask-Recycling, SK1 und ID als
  Dimensionierungsregel haben keinen realen kausalen beziehungsweise exakten
  V4-Beleg. Sie werden nicht aufgrund einer synthetischen Prozentzahl
  eingebaut.

## Gesetzes-Familie Organ-Bau

| Gesetz | Aussage | Heutige Steuerung in IMMER | Status |
|---|---|---|---|
| **Striktes Organ-Rezept** (R23) | STRUKTURFORM + KARTE + ROUTE-GATE + KRISTALL; kein freier Rechenkanal | Frozen A1 routet zu vier digest-geprüften SHIP-v6-Organen; exakte Struktur liefert das Ergebnis | **DEPLOYT**, 152/152 Antworten und Routen |
| **Kristallisations-Gesetz** (R17) | trainieren → Invariante messen → Fit/R² als strukturellen Verifier benutzen → exakte Form einsetzen | Decimal-Organ: gemessener Carrier, dann `h ← 10h + v`, exakte Algebra und Digit-Emission | **DEPLOYT**; Cross-Model-LS ist ausdrücklich kein R17 |
| **Kompositions-/Quotienten-Gesetz** (R14–R18) | Organ-Komposition folgt Bias-Gitter, nicht bloßer Kapazität | `ExactCascade` besitzt allein `exact_math`; S3 zuerst, FERTIG als Verifier/Fallback, sonst Abstinenz | **DEPLOYT** |
| **Konzentrations-Formel** `C = 1 − H/H_max` (R28) | Fähigkeit kann in kritischen Layern konzentriert sein; Konzentration ist keine Transplantierbarkeitsgarantie | Kandidat für einen künftigen kontextuellen WorldStream-Pre-Filter | **OFFEN**; nicht im Router behauptet |

FERTIG besitzt dabei Grounding, Bindings, semantische Pläne, Skills und
Verifikation. Bekannte unsichere Mengenänderungs-Fallbacks und nicht
injizierte Desktop-/Recorder-/Mutationspfade enden bewacht in Abstinenz.
Endliche affine Rekurrenzen `x_(i+1) = a*x_i + b` werden nun für explizite
Start-/Endindizes, Endwert, Nettoänderung und inklusive Summe strukturell
geparst und per exaktem Fraction/RREF-Zertifikat geprüft. Das ist eine neue
Fähigkeit, keine nachträgliche Optimierung am GSM8K-Split.

## Skalen- und Tempo-Gesetze

| Gesetz | Aussage | Status und zulässiger Gebrauch |
|---|---|---|
| **Sinkhorn-Sättigung** `Δ(d) ≈ 1,247·d/(d+33,1)` | CRSA-Gewinn sättigt historisch mit der Breite | **HISTORISCHER ANKER**; der einzelne negative Value-Sketch-Punkt erlaubt keinen Budgetkurven-Fit |
| **Zeno-k\*** `k* = 18,94·s^(−0.956)` | optimaler Konsolidierungsrhythmus skaliert mit Systemgröße | **OFFEN** für eine präregistrierte `LearningStream.sleep()`-Messung; nicht als heutige Schedule ausgegeben |
| **GOE→Ginibre** `ε_c = a·n^(−0.567)` | Übergangsskala des Spektrum-Kerns | **TEILWEISE GESTÜTZT** als Nullmodell-Idee: Stage 1 fand Struktur vor allem mid-depth; kein vollständiger Exponentenfit |

## Aktuelle Messanker

- **Frozen Exact-Pfad:** vier Organe, kein Runtime-Training, 152/152 Antworten
  und 152/152 Routen auf der kanonischen SHIP-v6-Suite.
- **FERTIG-Vollsplit:** 1.073/1.319 GSM8K korrekt, 246 sichere Abstinenzen,
  0 falsche und 0 Error-Outcomes. Der ungeprüfte Template-Fallback war 17/17
  falsch und bleibt bis zu einem strukturierten Proof proposal-only.
- **FERTIG-Rekurrenzen:** neue endliche affine Grammatik mit exaktem
  Zertifikat; bei ihrer Einführung blieb der damalige Vollsplit unverändert
  bei 1.060/1.319 und 0 falschen Antworten.
- **FERTIG-Abstention-Audit (Snapshot vor den acht neuen Resolvern):** Legacy
  65 fehlende Frageziele, 171 fehlende
  Ziel-Quantities, 20 unvollständige Relationen und drei Geld/Zeit-Guards;
  Structural 144 Coreference-Scope gegen 115 Grammar-Scope. Unter dem heute
  extrahierten Vertrag existieren null belegte exklusive Assignment-Instanzen.
- **Eigene Attention:** fester CRSA-Mix `2 Local + 1 Balanced + 1 Free`,
  Steigung `0.8`, Diagonal-Debit `3`; Zukunftsmasse exakt null und Free-Head
  bitgleich zur kausalen Softmax.
- **Kontextrouter:** CRSA-Residual Balanced Accuracy 1,000000; Roh-A1
  0,983333; 32 Label-Placebos im Mittel 0,510328 (Maximum 0,748026).
  Kausale Softmax erreicht ebenfalls 1,000000. Erlaubt ist daher nur
  `POSITIVE_CONTEXT_ROUTER__CRSA_NOT_UNIQUE_VS_SOFTMAX`, kein
  Überlegenheitsclaim.
- **Donor Stage 1:** Layer 18–54 tragen positive Frage-Struktur gegen
  Placebo; Layer 0 ist tot, Layer 63 klingt ab. Das lokalisiert Signal, es
  macht noch keinen Value-Pfad.
- **Donor Stage 2:** 24 % gegen 32 % Placebo. Negativ und nicht deployt.
- **Cross-Model Least Squares:** negativ. Der alte Vorschlag, einen
  Sibling-Hidden-State in den Donor-Key-Raum zu projizieren, wird nicht als
  Selbstkalibrierung oder R17 wiederbelebt.
- **Frontier-Decoder:** der gepinnte DeepSeek-V4-Flash-Hauptpfad läuft über
  alle 43 Layer mit nativer komprimierter Sparse-Attention, exakter
  blockskalierter MXFP8/FP4-Dekodierung, FP32-Akkumulation, HyperConnections,
  globalem Head und stateful Decode. Layerwise Proof-Schema v4 bindet auch
  Package-Exports, das konkrete CLI, Streamer, HF-Range-Reader, Snapshot und
  Dependency-Versionen. Das getrennte item-major Journal v2 bildet eine
  kanonische Hash-Kette.
- **Erster Inhaltslauf:** vier feste Geography-Items vollständig durch alle
  43 Layer: 3/4 korrekt. Manifest, Resultat und finales BF16-Objekt sind
  versiegelt und an Commit `d0a1dc4` gebunden. Das ist ein positiver
  Content-Smoke, noch kein allgemeiner MMLU- oder Frontier-Claim.
- **Exakter Expert-Transport:** offizielle 3-Expert-MPS-BF16-A/Bs sind
  bitgleich. Warm, nach zwei symmetrischen Warmups und 20 alternierenden
  Trials: 0,215072 s ohne gegen 0,220929 s mit q3, also 0,973489× im
  Mittel; der gepaarte Median liegt trotz 7/10 Wins bei 1,032965×.
  Ohne Range-Cache, nach einem Warmup und 8 alternierenden Trials:
  6,474703 s gegen 8,918329 s, also 0,726000× und nur 1/4 Wins. Der begrenzte
  Keep-alive-Pool sah bei 158 Requests sieben
  Connection-Objekte, Peak zwei aktive Leases, danach null und wurde vor dem
  Report geschlossen. Die einzelnen Netztrials streuen stark; ein stabiler
  q3-Speedup ist damit nicht belegt. Peak-Payload jeweils exakt 40.108.032 B.
  Das belegt exakte begrenzte Transport-Überlappung, aber noch keine
  Modellqualität oder Frontier-Parität.
- **Adjacent-Pairing:** kalt 6→4 Envelopes bei weiterhin 40.108.032 B;
  5,132159 s q3 gegen 4,418166 s Pair, 1,161604× im Mittel und 6/10
  Paar-Wins. Cache-resident ist Pair mit 1,005050× praktisch neutral. Ohne
  direkten `off`/Pair-Kontrast ist das kein Default; Topologie- und
  Byte-Receipts bleiben als Instrument.
- **Exakter LM-Head-Transport:** das reale kalte MPS/BF16-A/B reduziert
  127 physische Requests auf 16, ohne die 1.059.061.760 Byte, den vollständigen
  FP32-Logit-Hash oder Top-k-Werte/-IDs zu ändern. Über vier alternierende
  Paare gewinnt Breite 8 4/4 mit einem gepaarten Median von 1,445790×;
  Default Breite 1 bleibt bis zum End-to-End-Gate unangetastet.
- **Wave3-Audit:** T11 fällt nach Korrektur der doppelten `/N`-Skalierung von
  98 % auf 5,19 % und behält ungleiche Lehrerziele. Der zirkuläre 3-Zonen-
  Router fällt entkoppelt auf +4,67 % über Random beziehungsweise 36,66 % des
  Oracle und entspricht weder V4s Datenstrukturen noch seinem Routing.

## Hebel-Ranking

1. **Lokale v0.7-Akzeptanz konservieren:** Frozen A1, vier Organe, Router,
   FERTIG-Guards, Artefakt-Digests und Wheel-Isolation gemeinsam golden halten.
2. **DeepSeek-Inhaltsgate verbreitern:** die 3/4-`off`-Baseline steht; nach den
   einzeln falsifizierten Transporthebeln folgen gepaarte Graft-Ablationen
   unter identischer Runtime-, Byte- und Cachebilanz.
3. **Breiteren Router-Kontrast messen:** deduplizierter Fachsplit mit gleichem
   Budget für CRSA, kausale Softmax, Roh-A1 und permutierte Labels.
4. **WorldStream erst mit korrekter Zustandsverteilung neu öffnen:** echte
   kontextuelle RMSNorm-Post-Attention-Zustände oder eine andere gemessene
   Invariante; Falsifikator vor Implementierung benennen.
5. **R17 strikt anwenden:** trainieren → Invariante messen → Fit/R²
   verifizieren → exakte Struktur. Kein Cross-Model-LS-Ersatz.
6. **Handoff nur über Falsifikatoren öffnen:** realen HC-Residualverlauf
   passiv messen und CRSA-Graft-only-Temperaturen gegen ein permutiertes
   Placebo prüfen; Birkhoff kausal angepasst und ID/effective rank offline.
   Die übrigen synthetischen Demos bleiben NO-GO.
7. **Hugging Face ganz zuletzt:** erst wenn Runtime und Inhaltsgates lokal
   golden sind und die Lizenzkette geklärt ist, exportieren oder hochladen.

## Offene Endpunkte aus dem Fundament

GOE→Ginibre-α exakt? · Zeno-U-Kurve auf 15-%-Ausreißer-Graphen ·
Konzentration als belastbarer WorldStream-Pre-Filter? Diese Fragen bleiben
Forschungspunkte und werden erst mit Protokoll, Kontrast und Falsifikator zu
Runtime-Claims.
