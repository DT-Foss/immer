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

**NEGATIV als Retrieval-Mechanismus:** Statisches Embedding-Mittel
→ `gate_proj` → Value-Sketch erreichte 24 % gegen 32 % Placebo bei
138,4 KB/Frage. Die Eingabe entspricht nicht den RMSNorm-ten
Post-Attention-Zuständen, die `gate_proj` im Modell sieht. Dieser Shortcut ist
geschlossen; die Tensorquelle bleibt verwendbar.

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

## Skalen- und Tempo-Gesetze

| Gesetz | Aussage | Status und zulässiger Gebrauch |
|---|---|---|
| **Sinkhorn-Sättigung** `Δ(d) ≈ 1,247·d/(d+33,1)` | CRSA-Gewinn sättigt historisch mit der Breite | **HISTORISCHER ANKER**; der einzelne negative Value-Sketch-Punkt erlaubt keinen Budgetkurven-Fit |
| **Zeno-k\*** `k* = 18,94·s^(−0.956)` | optimaler Konsolidierungsrhythmus skaliert mit Systemgröße | **OFFEN** für eine präregistrierte `LearningStream.sleep()`-Messung; nicht als heutige Schedule ausgegeben |
| **GOE→Ginibre** `ε_c = a·n^(−0.567)` | Übergangsskala des Spektrum-Kerns | **TEILWEISE GESTÜTZT** als Nullmodell-Idee: Stage 1 fand Struktur vor allem mid-depth; kein vollständiger Exponentenfit |

## Aktuelle Messanker

- **Frozen Exact-Pfad:** vier Organe, kein Runtime-Training, 152/152 Antworten
  und 152/152 Routen auf der kanonischen SHIP-v6-Suite.
- **FERTIG-Vollsplit:** 1.060/1.319 GSM8K korrekt, 259 sichere Abstinenzen,
  0 falsche und 0 Error-Outcomes. Der ungeprüfte Template-Fallback war 17/17
  falsch und bleibt bis zu einem strukturierten Proof proposal-only.
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
  alle 43 Layer mit nativer komprimierter Sparse-Attention, FP8/FP4-MoE,
  HyperConnections, globalem Head und stateful Decode. Der CRSA-Graft ist im
  echten Hidden-State-Pfad aktiv; Qualitätsgleichstand ist noch nicht gezeigt.

## Hebel-Ranking

1. **Lokale v0.7-Akzeptanz konservieren:** Frozen A1, vier Organe, Router,
   FERTIG-Guards, Artefakt-Digests und Wheel-Isolation gemeinsam golden halten.
2. **DeepSeek-Inhaltsgate messen:** offizielles Prompt-Encoding und gepaarte
   `off / CRSA / softmax / shuffle`-Items unter gleicher Byte-/Cachebilanz.
3. **Breiteren Router-Kontrast messen:** deduplizierter Fachsplit mit gleichem
   Budget für CRSA, kausale Softmax, Roh-A1 und permutierte Labels.
4. **WorldStream erst mit korrekter Zustandsverteilung neu öffnen:** echte
   kontextuelle RMSNorm-Post-Attention-Zustände oder eine andere gemessene
   Invariante; Falsifikator vor Implementierung benennen.
5. **R17 strikt anwenden:** trainieren → Invariante messen → Fit/R²
   verifizieren → exakte Struktur. Kein Cross-Model-LS-Ersatz.
6. **Hugging Face zuletzt:** erst nach lokalem Golden und geklärter Lizenzkette
   exportieren oder hochladen.

## Offene Endpunkte aus dem Fundament

GOE→Ginibre-α exakt? · Zeno-U-Kurve auf 15-%-Ausreißer-Graphen ·
Konzentration als belastbarer WorldStream-Pre-Filter? Diese Fragen bleiben
Forschungspunkte und werden erst mit Protokoll, Kontrast und Falsifikator zu
Runtime-Claims.
