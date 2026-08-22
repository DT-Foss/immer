# Router-Formeln — Kanon auf das Streaming-QA-Problem übertragen

Stand: 2026-08-21. Jede Zeile: Kanon-Formel → Übertragung als Gleichung →
messbare Vorhersage → Falsifikator. Ziel: MMLU-Accuracy steigt, Bytes/Frage
sinkt — beide gegen Placebo.

## F1 — Zeno-Sampling-Dichte

Kanon: k* = 18,94 · s^(−0,956) (optimaler Rhythmus skaliert invers zur Größe).
Übertragung: Neuronen-Samplegröße n_L pro Layer L folgt n_L = N₀ · s_L^(−0,956),
wobei s_L = relative Wichtigkeit des Layers aus Stufe-1-Profilen.
Vorhersage: Accuracy(n_L-Verteilung nach Zeno) ≥ Accuracy(gleichverteilt)
bei gleichem Gesamt-Budget.
Falsifikator: kein Vorsprung > 2σ über 5 Seeds → Transfer tot, notieren.

## F2 — Sinkhorn-Sättigung der Seiten-Budgets

Kanon: Δ(d) ≈ 1,247 · d/(d + 33,1) — Gewinn sättigt.
Übertragung: Acc(bytes) = A_max · b/(b + b₀) mit b = gestreamte Bytes/Frage;
A_max ≈ Voll-Model-Score, b₀ = charakteristische Bytes (fitbar aus ≥4
Budget-Punkten).
Vorhersage: Die Messkurve (28 % @ 0,14 MB … ) liegt auf der Sättigungs-
form; b₀ sagt, wie viele Bytes bis 80 % von A_max nötig sind.
Falsifikator: systematische Abweichung vom Sättigungs-Lauf (R² < 0,9) →
der Mechanismus ist nicht budget-getrieben, anderes Gesetz suchen.

## F3 — Konzentrations-Filter vor dem Streaming

Kanon: C = 1 − H/H_max (Fähigkeits-Konzentration über Kandidaten).
Übertragung: Für eine Frage sei H die Entropie des Aktivierungsprofils über
gesampelten Neuronen. C_frage misst, OB eine Frage spaltbare Fakten-Seiten
hat — VOR teurem down_proj-Zugriff.
Vorhersage: Fragen mit hohem C profitieren von Seiten-Routing (Accuracy-
Zuwachs korreliert mit C, r > 0,4); niedrig-C-Fragen gewinnen nichts.
Falsifikator: Korrelation ≈ 0 über n≥50 → C ist kein Pre-Filter.

## F4 — Ginibre-Nullmodell für Aktivierungsprofile

Kanon: GOE→Ginibre-Übergang, ε_c = a·n^(−0,567) (Spektral-Ränder unter
Zufalls-Matrizen).
Übertragung: Unter Placebo (shuffled gate-Zeilen) verteilen sich Aktivier-
ungen a = |W_row · h| Ginibre-artig; die erwartete Maximal-Aktivierung folgt
der Rand-Skalierung. Dadurch wird „Top-k Neuronen" gegen das NULLMODELL
normiert statt gegen rohen Schwellenwert — sonst gewinnt immer der Zufalls-
Rand.
Vorhersage: echte Fragen produzieren Aktivierungsmaxima signifikant über
der Ginibre-Randschätzung (z > 2) in Fach-Layern, nicht in frühen Layern.
Falsifikator: Maxima überall am Rand → gate-Zeilen tragen keine Frage-
Struktur; Stufe-2 (down_proj-Werte) wäre vorzeitig.

## Betriebsregel

Jede Formel gilt erst mit Falsifikator-Durchlauf in results/ — gleicher
Standard wie der Rest des Kanons.
