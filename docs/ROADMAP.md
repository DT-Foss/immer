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
  Ein-Verbindungs-Sessions, bounded stream reads, deterministischer Close und
  1,382532× im aktuellen bitgleichen No-Cache-3-Expert-A/B.
- [x] FERTIG-GSM8K-Vollsplit mit Item-Provenienz und hartem Wrong-Gate:
  1.060 korrekt, 259 abstinent, 0 falsch, 0 Fehler.
- [x] FERTIG-Grammatik für endliche affine Rekurrenzen mit exaktem
  Fraction/RREF-Zertifikat; neue Fähigkeit bei unverändert 1.060/1.319 und
  weiterhin 0 falschen Antworten.

## Jetzt

### 1. DeepSeek-V4-Transport v5

Der q3/a2-Pfad ist gegen `off` positiv; sein marginaler Vorteil gegenüber q2
ist wegen getrennter Netzläufe noch nur diagnostisch. Als Nächstes werden nur
tatsächlich selektierte physisch benachbarte Ranges gebündelt. Outputbits,
gelesene Nutzbytes, Zwei-Read-Aktivgrenze und serielle
FP32-Akkumulationsreihenfolge müssen identisch bleiben.

### 2. DeepSeek-V4-Inhaltsgate

Die erste `off`-Baseline ist mit 3/4 korrekt positiv. Nach stabilem Transport
folgt ein gepaarter `off`/CRSA-Lauf mit offiziellem Encoding, festem Split und
kandidatengestütztem Scoring unter exakt derselben Runtime-Identität. Erst
größere Splits dürfen einen Qualitäts- oder Ähnlichkeitsclaim tragen.

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
ist bereits negativ und kein R17.

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
