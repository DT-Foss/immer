# Roadmap — Hebel nach Beweislage

Stand: 2026-08-22. Reihenfolge bedeutet Abhängigkeit, nicht Marketingwert.

## Erledigt

- [x] Atomarer Bus/Daemon/State-Port und digest-geprüftes OrganRack.
- [x] Persistenter O(1)-Lebensstrom mit Surprise-Gate, Replay und Sidecar.
- [x] Einziger `exact_math`-Besitzer: bewachte S3→FERTIG-Kaskade.
- [x] Frozen A1 plus vier SHIP-v6-Organe ohne Runtime-Training.
- [x] Kristallisierter Dezimalpfad `h ← 10h + v`.
- [x] Z3-Artefakt semantisch korrekt als `z3sum`, nicht als Modulo.
- [x] Eigene CRSA-Attention als realer Online-Abstain-Gate.
- [x] Kontext-Router mit Roh-A1-, Softmax- und 32 Label-Placebo-Kontrollen.
- [x] FERTIG Grounded-Adapter mit Graph-/Skillpfad und geschlossenen
  Desktop-/Recorder-/Mutationsgrenzen.
- [x] Budgetierter WorldStream mit Offlinequelle, exakt begrenzten Ranges und
  SHA-verifiziertem Resume.
- [x] CLI-Endpunkte für `solve`, `serve`, `eval`, `doctor`, `artifacts`,
  `organs` und `stream`.
- [x] Wheel-Isolation mit paketierten Manifesten und 152/152 aus sauberer
  Installation.
- [x] Abschließender A1-only-Offline-Export mit exakter Dateiliste,
  isoliertem 152/152-Selbsttest und hartem Lizenz-/Upload-Gate.
- [x] Gepinnter DeepSeek-V4-Flash-Hauptdecoder aus Safetensors-Ranges:
  43 Layer, native Sparse-Attention, exakte blockskalierte MXFP8/FP4-
  Dekodierung mit FP32-Akkumulation, HyperConnections, globaler Head und
  stateful Decode unter 16-GB-RAM-/12-GiB-Cache-Grenzen.
- [x] DeepSeek-Layerwise-Proof-Schema v4 mit gebundener Modell-, Package-,
  CLI-, Streamer-, HF-Range-, Snapshot- und Dependency-Identität; getrennt
  davon ein manipulationssichtbares Hash-Chain-Journal v2 für item-major
  Benchmarks.
- [x] Exaktes Routed-Expert-Prefetch `exact-router-window-q3-a2/v2`: drei
  sofort eingereihte Futures, höchstens zwei aktive I/O-Worker und drei
  residente/inflight Experts, 14 MiB pro Expert, 48 MiB Gesamtgrenze und
  bitgleiche A/B-Ausgabe.
- [x] Erster vollständiger DeepSeek-Inhaltslauf: 43/43 Layer, vier feste
  Geography-Items, `off` 3/4 korrekt; Resultat, Identität und finales
  BF16-Objekt versiegelt an Commit `d0a1dc4`.
- [x] Begrenzter Remote-Transport `requests-session-pool-2/v1`: zwei geleaste
  Ein-Verbindungs-Sessions, bounded stream reads und deterministischer Close.
  Das neuversiegelte q3/`off`-A/B bleibt bitidentisch, liefert wegen hoher
  Netzvarianz aber keinen stabilen Speedup-Claim.
- [x] Exaktes Multi-Range-Instrument mit Leaf-Cache-Reuse und
  Adjacent-Pair-A/B: kalt 6→4 Envelopes ohne Zusatzbytes und 1,161604× im
  Mittel über zehn Paare gegen q3; cache-resident 1,005050× neutral. Da der
  direkte `off`/Pair-Kontrast fehlt, bleibt q3/Width 1 Default und Pair/Width 2
  explizit aus.
- [x] Reales exaktes LM-Head-A/B: unveränderte 127 Rechenblöcke, aber
  127→16 physische Requests bei identischen 1.059.061.760 Byte sowie
  bitidentischem vollständigem Logitstrom und Top-k. Vier alternierende Paare
  gewinnt der Kandidat 4/4; gepaarter Median 1,445790×. Damit ist der
  Latenzmechanismus wiederholt positiv, bleibt bis zum End-to-End-Gate opt-in;
  Produktionsdefault Breite 1.
- [x] FERTIG-GSM8K-Vollsplit mit Item-Provenienz und hartem Wrong-Gate:
  1.060 korrekt, 259 abstinent, 0 falsch, 0 Fehler.
- [x] FERTIG-Grammatik für endliche affine Rekurrenzen mit exaktem
  Fraction/RREF-Zertifikat; neue Fähigkeit bei unverändert 1.060/1.319 und
  weiterhin 0 falschen Antworten.
- [x] DeepSeek-Hungarian gegen alle 259 FERTIG-Abstinenzen auditiert: 144
  Coreference-Scope, 115 Grammar-Scope und null bereits extrahierte exklusive
  Assignment-Verträge; die replizierte Demo zerfällt in unabhängige Argmins.
- [x] DeepSeek-Wave3 und Resthandoff auditiert: T11 nach doppelter `/N`-
  Korrektur nur 5,19 % bei weiterhin privilegierten Zwischenzielen;
  zirkulärer 3-Zonen-Router entkoppelt nur +4,67 % über Random und 36,66 % des
  Oracle. Nur HC-Residualtrace und CRSA-Graft-only-Temperatur bleiben als
  kontrollierte Messungen offen; Birkhoff/ID nur als Diagnostik.

## Jetzt

### 1. FERTIG-Strukturpfad

Der Vollsplit hat 259 sichere Abstinenzen bei null falschen Antworten. Die
nächste Arbeit teilt sich nun evidenzgemäß: Für 144 Coreference-Fälle werden
Antezedenten fail-closed aufgelöst; 115 Fälle brauchen zuerst neue getypte
Operationsgrammatik. Beide Wege müssen vollständig konsumierte numerische
Klauseln in den vorhandenen Fraction/RREF-IR überführen. Minimum-Cost-Matching
ist erst bei nachweislich exklusiven Slots zulässig; Greedy, Matching und
permutiertes Placebo teilen dann dieselben Kandidaten und Kosten. Das
Wrong-Gate bleibt null.

### 2. DeepSeek-V4-Inhaltsgate

Die erste `off`-Baseline ist mit 3/4 korrekt positiv. Nach stabilem Transport
folgt ein gepaarter `off`/CRSA-Lauf mit offiziellem Encoding, festem Split und
kandidatengestütztem Scoring unter exakt derselben Runtime-Identität. Erst
größere Splits dürfen einen Qualitäts- oder Ähnlichkeitsclaim tragen. Das
LM-Head-Batching hat Exaktheit und vier wiederholte Zeitpaare bestanden, bleibt
bis zu diesem End-to-End-Gate jedoch explizit opt-in; ein Produktionsdefault
darf den Rechenblockpfad weiterhin nicht ändern.

### 3. Breiter Router-Kontrast

Der heutige Split zeigt Kontextsignal, aber keinen CRSA-Sieg gegen Softmax.
Nächster zulässiger Versuch: größerer, fachlich gemischter, deduplizierter
Split; gleicher Ridge-Budgetdeckel; CRSA, Softmax, Roh-A1 und permutierte
Labels. Kein Rollen-Adaption-Claim ohne Stabilitätsmessung.

### 4. Lebenskurve

Fixes held-out Byte-Set, Messpunkte über echte Lebenszeit:

- NLL vor/nach Surprise-Updates;
- Post-Sleep-Delta;
- State-/Sidecar-Größe;
- Resume-Bitexaktheit;
- Drift des Frozen-Exact-Pfads muss null bleiben.

### 5. FERTIG-Evaluationsfamilien

Der vollständige GSM8K-Basissplit ist jetzt so gemessen. Nächster Ausbau sind
familienweise Guard-Sets für Mengenänderungen, mehrdeutige Bindings und
Desktop-Requests; falsche Antworten/Aktionsbehauptungen bleiben der Fehler.

### 6. Kontextuell korrekter WorldStream-Mechanismus

Stage 1 zeigte Mid-Depth-Struktur, Stage 2 mit statischem Embedding-Mittel war
schlechter als Placebo. Ein neuer Versuch braucht echte RMSNorm-te
Post-Attention-Zustände oder eine andere gemessene Invariante. Cross-Model-LS
ist bereits negativ und kein R17. Der externe „R25-18×“-PoC zählt ebenfalls
nicht: Er erzeugt die Layerstruktur synthetisch und vergleicht KL mit SNR.
Zulässig ist ein neuer klassenkonditionierter Donor-KL-Versuch nur mit vorab
getrenntem Train/Held-out, gleicher Metrik, All-Layer-Arm und Label-Shuffles.

### 7. Handoff-Messungen statt synthetischer Prozentwerte

Zwei schmale Messungen bleiben zulässig:

- passive Residuen-Spur des real ausgeführten HC-Sinkhorn-Kerns; ein späterer
  Early-stop-Kontrast wäre approximativ und muss Zeit, Logit-Delta und
  Entscheidungen gemeinsam berichten;
- Per-Head-Temperatur ausschließlich im CRSA-Graft, mit exakter Kausalmaske
  und permutiertem Placebo.

Kausal angepasste Birkhoff-Größen und ID/effective rank dürfen offline
diagnostizieren, aber keine Runtime steuern. Zeno-Schedule, Replica-MoE,
Live-η-Gate, Ginibre-Hurst, Mask-Recycling, SK1 und ID-Dimensionierung bleiben
NO-GO; dasselbe gilt für T11 und den 3-Zonen-Router aus Wave3.

## Später

- Donor-Ernte in `.causal`-Karten mit Herkunft und Revision.
- Rat gegen Einzelhirn auf festem Split und gleicher Call-/Zeitbilanz.
- Identitäts-Swap-Test bei weiterlaufendem State-Port.
- FLCA-/QAD-Produkte nur über explizite Contracts integrieren.
- Öffentlicher Hugging-Face-Upload erst nach schriftlich geklärter Lizenzkette;
  der lokale Offline-Export ist bereits technisch golden.

## Betriebsregeln

1. Originale außerhalb dieses Ordners bleiben read-only.
2. Jede neue Messung braucht Placebo oder klaren Kontrast.
3. Negative Ergebnisse bleiben geschlossen, bis ein neuer Mechanismus samt
   Falsifikator benannt ist.
4. Frozen Host/Organe werden vor dem Laden vollständig gehasht.
5. Strukturclaims folgen R17: Invariante messen, Fit als Verifier, exakte Form
   einsetzen.
6. Keine externe Mutation, Veröffentlichung oder Maschinenaktion als
   Nebenwirkung eines lokalen Tests.
