# Router-Formeln — Kanon, Messstatus und laufender Pfad

Stand: 2026-08-22. Die Formeln F1–F4 entstanden für das
Donor-Safetensors-Streaming. Sie werden hier nicht gelöscht, aber von der
heute deployten A1/CRSA-Runtime getrennt. Jede Formel braucht Vorhersage,
Kontrast und Falsifikator.

## Der heute laufende Router

```text
kanonisierter Text
  → Frozen A1, kontextuelle Layer-0-Scan-Zustände
  → eigene CRSA-Attention: 2 Local + 1 Balanced + 1 Free
  → slope 0.8, diagonal debit 3
  → Feature [raw_last | role_complete_last]
  → persistierter Ridge-Head
  → Text abstinent oder exakter Organpfad
```

Der Rollenmix ist fest, die Zukunftsmasse exakt null und der Free-Head
bitgleich zur kausalen Referenz-Softmax. Auf 152 Arithmetik- plus 30
Textfällen erreicht der CRSA-Pfad Balanced Accuracy 1,000000; Roh-A1 erreicht
0,983333 und 32 permutierte Label-Placebos im Mittel 0,510328 (Maximum
0,748026). Die kausale-Softmax-Ablation bindet jedoch mit 1,000000.

**Messurteil:** positives Kontextsignal, aber kein CRSA-spezifischer Vorteil.
Der zulässige Claim lautet
`POSITIVE_CONTEXT_ROUTER__CRSA_NOT_UNIQUE_VS_SOFTMAX`.

## F1 — Zeno-Sampling-Dichte

Kanon: `k* = 18,94 · s^(−0,956)`; optimaler Rhythmus skaliert invers zur
Größe.

Übertragung: Die Neuronen-Samplegröße `n_L` pro Layer `L` folgt
`n_L = N₀ · s_L^(−0,956)`, wobei `s_L` die relative Layer-Wichtigkeit aus
Stufe 1 bezeichnet.

Vorhersage: `Accuracy(Zeno-Verteilung) ≥ Accuracy(Gleichverteilung)` bei
identischem Gesamtbudget.

Falsifikator: kein Vorsprung größer als `2σ` über fünf Seeds.

**Status: OFFEN.** Stage 1 lokalisierte Layer, verglich aber keine
Zeno-Verteilung gegen ein gleich großes Sampling-Budget. F1 ist nicht im
Online-Router aktiv.

## F2 — Sinkhorn-Sättigung der Seiten-Budgets

Kanon: `Δ(d) ≈ 1,247 · d/(d + 33,1)`.

Übertragung: `Acc(b) = A_max · b/(b + b₀)` mit gestreamten Bytes `b` pro
Frage; `b₀` wird aus mindestens vier Budgetpunkten gefittet.

Vorhersage: Eine kontrollierte Budgetkurve folgt der Sättigungsform und sagt
die Bytes bis 80 % von `A_max` voraus.

Falsifikator: systematische Abweichung mit `R² < 0,9`.

**Status: FÜR DEN AKTUELLEN MECHANISMUS GESCHLOSSEN.** Stage 2 lieferte nur
einen Punkt und scheiterte bereits qualitativ: 24 % gegen 32 % Placebo bei
138,4 KB/Frage. Ein Kurvenfit würde den falsifizierten Value-Sketch nicht
retten.

## F3 — Konzentrations-Filter vor dem Streaming

Kanon: `C = 1 − H/H_max`.

Übertragung: Die Entropie `H` des Aktivierungsprofils definiert
`C_frage`; hohe Konzentration soll Fragen markieren, deren Faktenseiten vor
teurem `down_proj`-Zugriff trennbar sind.

Vorhersage: Der Accuracy-Zuwachs durch Seitenrouting korreliert mit `C`
(`r > 0,4`).

Falsifikator: Korrelation ungefähr null auf `n ≥ 50`.

**Status: OFFEN.** Stage 2 testete statische Aktivierungsgewichte, nicht diese
Korrelation. F3 ist weder bestätigt noch deployt.

## F4 — Ginibre-Nullmodell für Aktivierungsprofile

Kanon: GOE→Ginibre, `ε_c = a · n^(−0,567)`.

Übertragung: Unter Placebo werden geshuffelte `gate_proj`-Zeilen als
Nullmodell benutzt. Echte Fragen müssen sich über dieses Randverhalten erheben,
nicht nur über einen rohen Top-k-Schwellwert.

Vorhersage: Fachstruktur erscheint in tragenden, nicht in frühen Layern.

Falsifikator: echte Maxima bleiben in allen Layern am Placebo-Rand.

**Status: TEILWEISE POSITIV GEMESSEN.** Stufe 1 ergab:

| Layer | `separation_real − separation_placebo` | Deutung |
|---:|---:|---|
| 0 | −0,00021 | tot |
| 9 | +0,00006 | schwach |
| 18 | +0,00151 | mid-depth Signal |
| 27 | +0,00103 | mid-depth Signal |
| 36 | +0,00140 | mid-depth Signal |
| 45 | +0,00128 | mid-depth Signal |
| 54 | +0,00105 | mid-depth Signal |
| 63 | +0,00071 | abklingend |

Das misst **wo** Themenstruktur liegt. Es beweist weder den exakten
Ginibre-Exponenten noch, dass statische `gate_proj`-Aktivierung brauchbare
Wertseiten auswählt. Genau dieser zweite Schritt scheiterte in Stage 2.

## R17 ist kein Cross-Model-Fit

Die vorgeschlagene Least-Squares-Projektion von Hidden-States eines kleinen
Sibling-Modells in den Donor-Key-Raum war negativ und bleibt geschlossen.
Der kanonische R17-Ablauf ist:

```text
trainieren → Invariante messen → Fit/R² verifiziert die Struktur
           → exakte Struktur einsetzen
```

Ein neuer WorldStream-Versuch muss deshalb entweder die tatsächliche
RMSNorm-te Post-Attention-Eingabeverteilung messen oder eine andere Invariante
mit vorab benanntem Fit und Falsifikator kristallisieren.

## Betriebsregel

Formeln werden nie durch Architekturprosa als bestanden erklärt. Positive,
negative und offene Befunde bleiben in `results/` beziehungsweise im
persistierten Routermanifest reproduzierbar. Hugging-Face-Verpackung ist erst
der letzte Schritt nach lokalem Golden und Lizenzklärung.
