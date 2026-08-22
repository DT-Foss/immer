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

## Jetzt

### 1. Breiter Router-Kontrast

Der heutige Split zeigt Kontextsignal, aber keinen CRSA-Sieg gegen Softmax.
Nächster zulässiger Versuch: größerer, fachlich gemischter, deduplizierter
Split; gleicher Ridge-Budgetdeckel; CRSA, Softmax, Roh-A1 und permutierte
Labels. Kein Rollen-Adaption-Claim ohne Stabilitätsmessung.

### 2. Lebenskurve

Fixes held-out Byte-Set, Messpunkte über echte Lebenszeit:

- NLL vor/nach Surprise-Updates;
- Post-Sleep-Delta;
- State-/Sidecar-Größe;
- Resume-Bitexaktheit;
- Drift des Frozen-Exact-Pfads muss null bleiben.

### 3. FERTIG-Evaluationsfamilien

Nicht nur Solve-Rate messen. Pro Familie werden korrekt, abstinent und falsch
gezählt. Mengenänderungen, mehrdeutige Bindings und Desktop-Requests sind harte
Guard-Sets; falsche Antworten/Aktionsbehauptungen sind der Fehler.

### 4. Kontextuell korrekter WorldStream-Mechanismus

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
