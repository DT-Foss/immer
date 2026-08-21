# W16 + Prefix-Balance Semigroup — Root-Cause-Fix

## Ergebnis in einem Satz

DeepSeek hatte **beide Experimente beschädigt**:

- Der Semigroup-Test verwandelte die kausale Maske numerisch in positive
  Zukunftsmasse und untersuchte dadurch nicht den beschriebenen Operator.
- Der W16-End-to-End-Test berechnete zwar `a_gelesen`, ignorierte es aber und
  gab dem Emitter den wahren Wert. Die gemeldeten 96/96 waren daher ein
  Oracle-Emitter-Test, kein Organ-Ergebnis.

## 1. Semigroup: der eigentliche Root Cause

Der alte Operator tat:

```python
logA = A.clamp_min(torch.finfo(A.dtype).tiny).log()
```

Damit wurden maskierte Nullen zu kleinen **positiven** Zahlen. Für
`alpha >= 1` wird

```text
log_raw = logA - alpha * log_usage
```

bei diesen Einträgen nicht harmlos: die verbotene Zukunftsmasse wird verstärkt
und anschließend über die gesamte Zeile normalisiert.

Gemessene Reproduktion auf vier 96×96-Matrizen:

| Operator | alpha | gesamte verbotene Zukunftsmasse |
|---|---:|---:|
| Alt | 1.0 | 114.514336 |
| Alt | 2.0 | 379.999878 |
| Repariert | 2.0 | **0 exakt** |

Damit waren die alte `a_eff`-Tabelle, die angebliche alpha=1-Fixzeile und die
Entropie-Oszillation Eigenschaften eines maskenbrechenden Operators.

### Korrigierter Semigroup-Befund

Der reparierte Lauf benutzt:

- exakt erhaltenen kausalen Support;
- Float64;
- globalen `a_eff`-Fit auf Trainingsmatrizen;
- eingefrorene Auswertung auf unabhängigen Matrizen;
- mittlere Total-Variation pro Zeile statt MSE über maskierte Nullen;
- Kommutator- und Matrixabhängigkeitsmessung.

Ergebnis über 49 Alpha-Paare:

| Maß | Ergebnis |
|---|---:|
| Exakte Closures | **0/49** |
| Residual ≤ 5 % des Kompositionseffekts | 14/49 |
| Median relativer Closure-Fehler | 0.0945 |
| Maximum | 0.2583 |
| Maximaler Reihenfolge-Kommutator | 0.1272 TV |
| Additiver Projektionsfit | R² 0.8165 |
| Multiplikativer Projektionsfit | R² 0.5245 |
| Asymmetrisch-quadratischer Projektionsfit | R² 0.9923 |

**Sauberer Schluss:** keine skalare Halbgruppe. Schwache Pässe sind lokal
näherungsweise durch einen Einzelpass approximierbar; starke Kompositionen
sind deutlich nicht geschlossen und reihenfolgeabhängig.

Der korrekte Operator kontrahiert bei wiederholter Anwendung monoton zur
Diagonale. Für alpha=2 fällt die mittlere Zeilenentropie in zehn Pässen von
3.1558 auf 0.2869 — ohne Zukunftsmasse und ohne die alte Oszillation.

## 2. W16: acht getrennte Fehler

### Kritisch

1. **Ground-truth bypass:** `a_gelesen` wurde berechnet, aber nie benutzt.
   `emittiere(words, wert)` erhielt den wahren Wert `2*a+4`.
2. **Doppelte Kristall-Inversion:** `forward()` snappte bereits auf den
   Integerwert; danach wurde `(u-gamma)/alpha` ein zweites Mal angewandt.
3. **Halbe Kristallisation:** Die Karte wurde gesnappt, während die gelernten
   Parameter `q` und `s0` erhalten blieben. Karte und exakter ordinaler
   Readout müssen atomar ersetzt werden.

### Hohe Relevanz

4. **Held-out leakage:** Alpha/Gamma wurden mit allen neun numerischen Labels
   gefittet, einschließlich der drei Held-out-Zellen.
5. **Pseudoreplikation:** Die „96 frischen Zellen“ waren Wiederholungen von nur
   neun möglichen Operanden.
6. **Review-Split falsch:** Der echte `Random(49)`-Split ist `{2,4,6}`; die
   Review-Probe maß `{3,5,8}`.

### Methodisch

7. `full_linear_r2=1` bei 9 Punkten in 128 Dimensionen ist triviale
   Interpolationskapazität, keine Generalisierung.
8. Exakte kNN-Klassenaccuracy ist bei einem Beispiel pro Klasse strukturell
   nutzlos; die eigene Klasse kann in Leave-one-out nie Nachbar sein.

### Reparierter W16-Vertrag

Der neue Runner:

- berechnet `2*a+4` ausschließlich aus dem **vorhergesagten** Operand;
- trennt `oracle_emitter` und echtes `end_to_end`;
- fitttet die Karte nur auf Trainingszellen;
- ersetzt Karte und Parabel-Readout zusammen;
- misst alle neun einzigartigen Zellen;
- fährt mehrere Modell- und Split-Seeds;
- benennt die bekannte Formel-Inversion ehrlich als strukturierte Bridge,
  nicht als pauschal label-freies Experiment.

Der alte Wert `96/96` ist ungültig und wird nicht in eine neue Zahl
umetikettiert. Für den echten Rerun fehlen im ZIP der Donor-NPZ, der
Host-Checkpoint und zwei Hostmodule; der reparierte Runner ist dafür fertig.

## 3. Provenance-Problem im gelieferten Archiv

`semigroup_run.log` und `semigroup_test.json` stammen nicht aus demselben Lauf:

- JSON additiv R²: 0.0635; Log: 0.3886
- JSON multiplikativ R²: 0.2878; Log: 0.2359
- JSON Runtime: 74.2 s; Log: 20 s

Der Fix legt Code, Ergebnis, Report, Tests und SHA-256-Manifest gemeinsam ab.

## 4. Verifikation

```text
12 Tests bestanden
- exakter kausaler Support
- Zeilensummen und Endlichkeit
- alpha=0 Identität
- analytischer 2×2-Kontrollfall
- starker Wiederholungsstress
- korrekter Random(49)-Split
- einmalige Kristalltransformation
- atomarer exakter Readout
- train-only Kalibrierung
- Ground-truth-bypass Negativtest
- Runner-Integrationstest: falscher Organwert bleibt end-to-end falsch, Oracle separat grün
```
