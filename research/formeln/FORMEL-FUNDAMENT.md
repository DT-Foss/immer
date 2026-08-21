# FORMEL-FUNDAMENT — Das mathematische Fundament von allem (19.08.2026)

> Vollständiger Digest der 17 Markdown-Dateien in `~/self-verification/Formelnusw/`
> (9 Papers + 8 Sammlungen) + `GLMIDeen.md` + `PAPER_GOLDMINE_INTEGRATION.md`
> + `hf_organ_reader.py` (Mechanik). Die Formeln sind Davids Eigenwerk
> (Anfang 2026) — und laut Auftrag erweiterbar/verbesserbar. Dieser Digest
> markiert die offenen Enden.

## 1. Das EINE Ding — der gemeinsame Kern

**Spektraltheorie (doppelt-)stochastischer Matrizen**: der Eigenwert
**λ ∈ (−1, 1)** ist das einzige Fundamentalobjekt, auf dem drei universelle
Operationen wirken:

1. **Wick-Rotation** `E_k = −log λ_k` — Spektrallücke Δ wird Masselücke m;
   Born-Regel aus Perron-Frobenius (P5).
2. **Möbius/Sqrt-Kopplung** `f(λ,v) = (λ+v)/(1+λv)` mit Periode
   `g(λ) = (1−λ²)^{−1/2}` — die Lorentz-Gruppe/Doppler; „a single algebraic
   identity", aus der 50+ Phänomene fließen (P3/P5).
3. **Kontraktion** — Birkhoff-Koeffizient τ = 0.508 (klassisch) vs. τ = 1
   (unitär) als Ordnungsparameter; verbindet Collapse, Unitarity und den
   GOE→Ginibre-Übergang (P1/P8/P9).

Alles Weitere sind Projektionen auf denselben Raum: Sinkhorn = Projektion
aufs Birkhoff-Polytop B_N (dim (N−1)²), BvN = Pfadzerlegung, Cheeger/
Fiedler = Bottleneck, Ginibre-Kern ⟨s²⟩ = 1.08747 = universelle
Spektralstatistik, PS-Lifted = nicht-reversibler Lift, der die Gap-Barriere
bricht. **GLMIDeen-Formulierung der Einheit:** O1-Streaming-State,
Broadcast-Selector und CausalMacroRouter sind *drei Instanzen EINES
Operators — zustandskonditionierte Projektion auf eine riesige kalte Menge.*

## 2. Die wichtigsten gemessenen Anker (Zahlen fürs CHANGELOG)

| Formel | Zahl | Quelle |
|---|---|---|
| Lifted O(log n)-Mixing (Barbell n=1024) | 22,1 vs. 604.779 Runden = **27.379×** | foss_research |
| Foss-Konvergenz O(1) (BA n=100.000) | 14 vs. 2.217 = 158×; Oregon 22 vs. 54.456 = **2.475×** | P2 |
| Foss-Skalengesetz async | T = T_sync·p^(−1.03±0.06), R²=0.99 | P2 |
| Cheeger-Entkopplung | Lifted β=0.11 vs. MH 0.93 (Gap ≈ Ω(1)) | P4 |
| Local-SGD + PS-Lifted | 91,1 % > synchron 85,6 %; η_eff = η·√p | P2 |
| Sinkhorn-Attention | PPL −66,6 % (13,7M); 89 % bei 32M; 96 % kausal-spezifisch; SK=1 → 86 % | gottformel |
| DS-Grokking | 162.481·P^(−0.892), R²=0.953; Standard chaotisch | gottformel |
| ID-Skalengesetz | ID = 2.023·log₂(K) + 1.206, R²=0.919 | gottformel |
| Foss-Zahl | F = 1+1/(3π) = 1.1061 ist **Transient**; kanonisch ⟨s²⟩ = 1.0874686… | P7 |
| Zeno/Anti-Zeno | k*=3 (Übergang), **k=5 optimal (−41 %)**, k=1: 2,2× langsamer | P8/parallel_quantum |
| Unitarity-Boundary | F = S_ent/ln n = 0.75±0.05; **21/21 korrekt klassifiziert** | P8 |
| GOE→Ginibre-Übergang | ε_c = a·n^(−0.567), R²=0.998; H = 0.890±0.003 | P9 |
| Standardmodell | sin²θ_W = 505/2184 (0.2σ); η_B exakt; Koide Q = 2/3 (0.005 %) | P3 |

**Korrektur (wichtig):** Die Codes F2/BC8/F9/F12/F33–F40 existieren in den
Dateien NICHT — die Sammlung nummeriert `Finding/Theorem/Eq./DS/T`.
Reale Namen: Foss Number (superseded), F8 (GLMIDeen: stationäre Signale =
strukturelle Invariante), Foss-Topologischer Index, κ_FML = 0.027.

## 3. Formel → Projekt (die Brücke zum Moonshot)

| Projekt | Formeln |
|---|---|
| GSSM/o1-state | Möbius + Sqrt-Gruppe (z = log(1−m²) additiv!); Ginibre ⟨s²⟩; Surprise-Gate als Zustands-Projektion |
| Sinkhorn-Attention | DS-Projektion + KL-Gradient-Fluss-Interpretation + Kernabweichung als Regularizer (P7 XI.D) |
| PS-Lifted-Flotte | W + Push-Sum + p_c-Formel; Gap-Theorem; η√p |
| Zeno/Schlaf | k*-Schedule; L = L_task + λ(D−2)² |
| Gate/Router | Bᵀh-Traversierung; SiLU-Parität; Möbius-Prefetch F_v(P) = (P+vI)(I+vP)⁻¹ (2,4e-15) |
| **Weltwissen-Streaming** | hf_organ_reader-Mechanik (unten) |

## 4. Der Tensor-Streaming-Moonshot (hf_organ_reader, präzise Mechanik)

**Das Weltwissen-Problem wird gelöst, indem Gewichtsseiten aus den
HF-Safetensors GESTREAMT werden, nicht resident gehalten:**

1. **Header-only-Inventar:** pro Shard `bytes=0-7` (Headerlänge u64), dann
   `bytes=8..8+hlen−1` → komplettes Tensor-Inventar OHNE Payload.
2. **Redirect-Kette:** HF-resolve → 302 auf CDN; manuell verfolgt, finale
   URL gecacht bis Signatur-Expiry.
3. **Single-Range-GET:** gezielte Tensoren/Zeilen via `bytes=start-end`
   (206 Partial Content); Multipart → 416 → Rotation auf 1 Request/Block.
4. **Byte-Exaktheit:** Content-Range muss exakt matchen; ETag + 64-hex-
   **CAS-Hash** (Merkle-artig) aus der signierten URL.
5. **Block-kontiguierliches Sampling** (N=8 Blöcke, seeded Jitter,
   RNG [42,crc32(name),13]) → Transfer == Analysematrix, Overfetch ≈ 0.
6. **Budget hart ~200 MB**; Determinismus: [42,crc32(name)] reproduziert
   E01-CASI-Ratios bitgenau (6/6 Tensoren).
7. **Forward-Verbrauch:** Router Bᵀh wählt Top-k-Seiten → exakte Seiten per
   mmap + kompakter Tail; Batch-32 amortisiert **9,44 → 2,19 MB/Query**.

**Zahl, die den Moonshot trägt:** 27B-Organ-Scan mit **116,6 MB Transfer
= 0,21 %** von 55,56 GB — bitgenau validiert. Der Router macht daraus
9 MiB–580 MiB pro Query (KIMI §8), Grenzkosten ~1,1 MiB/Token. D. h.:
**Weltwissen wohnt in kalten, Merkle-adressierten Seiten; der Wirt streamt
nur, was die Aufgabe braucht.** Das ist die F4-Erzwingung (Zwei-System-
Gesetz) technisch eingelöst — und der „kranke Moonshot" ist real: er ist
schon gemessen.

## 5. Erweiterungsmöglichkeiten (offene Enden — Davids Einladung)

1. **Foss-Gap-Theorem:** bewiesen c₁=1/4, empirisch 2,6 → Beweis schärfen
   (Lemma 5); Barbell-α≈0,40 ohne Theorie („α=2/5 oder 1/2 wäre
   signifikant"); optimales p_c≈0,65 analytisch offen.
2. **One-Constant:** α_DS=2/5 Konjektur (Birkhoff-Perturbations-Entwicklung);
   geschlossene NND-Dichte fehlt („Not Painlevé — welche
   Spezialfunktionen-Klasse?"); n=1000-Check (erwartet ⟨s²⟩≈1,092).
3. **GOE→Ginibre:** α≈0,57 exakt berechenbar aus B_N-Geometrie?
   H≈0,89 aus Barvinok-Godsil-McKay ableiten?
4. **Non-Reversibility:** gap=Ω(1) formal (Conjecture 3); #P/Permanenten/
   BosonSampling-Verbindung; Graphklassen-Charakterisierung (BA vs. ER).
5. **Selbstkorrekturen der Sammlung:** τ=1,30 als Dobrushin ist falsch
   (∈[0,1]); Δx·Δp-Bound intern widersprüchlich (0,5 vs. gemessen 0,38);
   S_horizon-Mismatch; Hawking-T-Mismatch (2,005 vs. 0,0398); Foss-Zahl
   superseded → alle CHANGELOG-Einträge, die F=1,1061 nutzen, auf
   ⟨s²⟩=1,08747 umstellen.
6. **Sinkhorn-Tuning:** 3 Iterationen genügen (P7) vs. Default 5/20–100;
   SK=1-Sparform (86 %); Overhead 1,77× reduzierbar.
7. **Zeno-U-Kurve:** k*=3 nur auf 85 % der Graphen (23/27) — Abhängigkeit
   von Grad-Heterogenität parametrisieren (offener Hebel fürs
   Anti-Zeno-TODO des Projekts!).
8. **Qwen-Integration:** relL2 0,43 nur Layer 0 (28-Layer-Kompilierung
   offen); Fiedler-Gap-Garantie auf beliebige Cuts unbewiesen; Rang 384–512
   nötig (D=2 empirisch widerlegt).

## 6. Konsequenz für den Moonshot (Verbindung)

- **Der Kern-Operator ist überall derselbe** (Zustand → Projektion auf
  kalte Menge): o1-state-Stream, Router, Organ-Ernte, Verifier-Gate —
  das gibt dem Moonshot eine einzige theoretische Rückgrat-Linie.
- **Weltwissen per Tensor-Streaming ist gemessen machbar** (0,21 %-Scan,
  2,19 MB/Query amortisiert) — der Moonshot „kleiner Wirt + gestreamtes
  Wissen + transplantierte Fähigkeiten" steht auf Formeln UND Messungen.
- **Erweiterungs-Kandidaten mit direktem Projekt-Nutzen:** Zeno-k*-
  Topologie-Parametrisierung (Anti-Zeno-TODO), Sinkhorn-3-Iterationen
  (Sinkhorn-Arm), α_DS-Beweis (Paper v0.3), Foss-Gap-Schärfung (PS-Lifted-
  Paper), Sammlungs-Korrekturen (CHANGELOG).

## Skalen-Sättigung des Sinkhorn-Gewinns (Runde 27, 5 Punkte)

k1-Serie (d = 128..1024, Seeds 7/8): Δ = +0,984 / +1,106 / +1,173 /
+1,182 / +1,184 bpb. Modellvergleich (χ²/dof, se-gewichtet):

| Modell | Parameter | χ²/dof |
|---|---|---|
| **Δ(d) = Δ∞·d/(d+d₀)** | Δ∞ = 1,247, d₀ = 33,1 | **3,28** |
| Δ(d) = Δ∞·(1−e^(−d/d₀)) | Δ∞ = 1,172, d₀ = 71,6 | 3,85 |
| Δ = c·d^(−α) (bisher) | α = −0,090 | **17,30 — widerlegt** |

**Formel (neu):** Δ(d) ≈ 1,247 · d/(d+33,1) — der Gewinn sättigt mit
Halbwerts-Skala d₀ ≈ 33; bei d = 512 sind ~94 % des Plateaus
(Δ∞ ≈ 1,25 bpb) erreicht. Das Potenz-Gesetz (α-Metrik) ist damit
formal das schwächere Modell — die Papier-Aussage wird präziser:
„skalen-stabiler Mechanik-Gewinn mit Sättigung", nicht „Wachstum".
Caveat (ehrlich): d₀/Δ∞ aus 5 Punkten (d=1024 nur 1 Seed);
χ²/dof > 1 deutet auf unterschätzte se.

## Zeno-k*-Gesetz GEHÄRTET (Runde 28, 32 Konfigurationen)

k* = **18,94 · s^(−0,956)** — Fit über 8 s-Werte (0,25..4,0) × 4 gamma
(0,05..0,5): **a und b exakt gamma-invariant (Spanne 0,000)** — die
8.23-Aussage „gamma skaliert nur die Zeitachse" ist damit formal
belegt. Der frühere Exponent −0,87 (3 s-Punkte) war Fit-Rauschen;
der verfeinerte b ≈ −0,96 bedeutet: die optimale Mess-Kadenz skaliert
fast invers mit der Spektral-Steilheit. Datei: formeln/zeno_haertung.py
+ zeno_haertung.json.

## Fähigkeits-Spektrum + Konzentrations-Formel (Runde 28, 0.5B-Host)

Layer-Ablation (MLP-Ausgang @Lese-Position) je Aufgabe; delta = acc(L
abliert) - intact. **Konzentration C = 1 - H/H_max** über die
normalisierten |delta| (H = Shannon-Entropie): C = 1 perfekt
konzentriert, C = 0 gleichverteilt.

| Aufgabe | intact | C | kritische Layer | schädliche Layer |
|---|---|---|---|---|
| twostep | 0,344 | 0,242 | 0, 4, 6, 18, 20 | **16** |
| add2 | 0,990 | 0,357 | 0, 14, 16, 18, 20 | — |
| mul2x2 | 0,969 | 0,366 | 0, 16, 18, 20 | — |
| wordlen | 0,010 | — | KEINE (alle Δ=0) | — |

**Befunde:**
1. **wordlen hat keine lokalisierbare Struktur** — die „Decke" ist
   formal erklärt (nichts zu transplantieren, konsistent mit der
   Kaskaden-Rettung via API: Fähigkeit fehlt im Host, nicht im Organ).
2. **L0 ist überall kritisch** (Pipeline-Bruch, generisch). Die
   fähigkeits-spezifische Region ist ~L14-20; twostep zusätzlich früh
   (L4/6).
3. **L16 ist bei twostep SCHÄDLICH** (Ablation +0,115), bei add2
   kritisch — eine Konflikt-Schicht (Fähigkeiten konkurrieren um die
   Schicht). Das GraftHook-Fenster L16-23 enthält L16 → die
   twostep-Graft-Ergebnisse (1,000) profitierten von der
   L16-Entfernung. **Die Ablations-Region ist ein Design-Parameter**
   (redistill.py --lo/--hi).
4. Konzentration korreliert NICHT einfach mit Transplantierbarkeit
   (twostep: C=0,242, bester Graft) — die Hypothese „konzentriert ⇒
   transplantierbar" ist widerlegt; stattdessen zählt die
   REKONSTRUIERBARKEIT der Region (twostep: Algebra aus L18-20
   rekonstruierbar, wordlen: Zählen nie vorhanden).

**Laufend:** Graft mit kritischer Region (L18-21) statt L16-23
(redistill_twostep_lo18.json) — testet die Region als Parameter.

## Null-Modell für 2-Kandidaten-Spektren: die Bias-Identitätslinie (Fable R2)

Bei Antworträumen mit 2 Kandidaten (yes/no) misst das Ablations-Spektrum
NICHT automatisch Fähigkeits-Lokalisierung: Ablation kann den Format-Bias
kippen, und weil die Klassen unbalanciert sind, ändert sich die Accuracy
rein arithmetisch. Null-Modell (Prädiktion unkorreliert mit Wahrheit):

    acc_0(L) = yr(L)·p_yes + (1 − yr(L))·(1 − p_yes)

mit yr(L) = yes-Rate unter Ablation von L. Gemessen (0.5B-Host, 96 eval):
prime R² = 0,949, div3 R² = 0,953 gegen die Identitätslinie — die
kompletten Spektren sind Bias-Kippen, null Fähigkeit. Erst Residuen ÜBER
acc_0 sind Fähigkeits-Signal; die Konzentrations-Formel C = 1 − H/H_max
ist auf |delta| gegen acc_0 zu rechnen, nicht gegen acc(intact).
Dateien: s3/spektrum_yn.py, s3/spektrum_yn_{prime,div3}.json.

## Additivitäts-Constraint schlägt Kapazität: die Organ-Kompositions-Formel (Fable R6)

Memorisierung ist kein Kapazitäts-, sondern ein STRUKTUR-Problem: Ein
66k-MLP-Organ in der Velocity-Route memorisiert den 49-Paare-Raum
(train 0,744 / held-out 0,100); ein 8,7k-Organ, das die Operanden nur
als SUMME ihrer projizierten Embeddings sieht, komponiert
(train 1,000 / held-out 0,900, Seed 7):

    delta_c = r_c · (P·e_a + P·e_b),   argmax_c delta_c = a + b

ist exakt realisierbar mit der parabolischen Score-Familie
φ(a)_c = c·a − c²/2 (denn c·(a+b) − c²/2 ist maximal bei c = a+b —
die Zahlengerade als Linearform). Das Organ KANN einzelne Paare nicht
auswendig lernen, weil (a,b) und (b,a) und alle gleichsummigen Paare
denselben Input erzeugen — die Generalisierung ist erzwungen, nicht
erhofft. Anschluss an den Kern-Operator (§1): das Organ ist eine
zustandskonditionierte Projektion auf den Kandidaten-Slice; das
4-Gramm-Gate (auf Text gegen 0 regularisiert) ist die Zuständigkeits-
Projektion, die die Sprach-NLL exakt unangetastet lässt (8,6656).
Verallgemeinerung (offenes Ende Nr. 9): jede Fähigkeit mit bekannter
Invarianz (Kommutativität, Assoziativität, Translationsäquivarianz)
bekommt ihr Organ-Constraint — das Constraint IST das Organ-Design.
Beleg: s3/w3_exp_b.py/.json (Agent-Befund, Multi-Seed in w3_exp_c).

## Kompositions-Formel v2 (Fable R9, nach der Subtraktions-Lösung)

v1 (Additivität erzwingen) war ein Spezialfall. Die vollständige Form,
drei Zutaten, jede einzeln als notwendig belegt (w3_sub_komposition):

1. **Bottleneck in Aufgaben-Dimension:** der Encoder presst jedes Token
   auf einen Skalar v(t) = phi(e_t) (nichtlinear, MLP), der Kopf sieht
   nur die Aufgaben-Algebra der Skalare (v(a)−v(b) bzw. v(a)+v(b)) —
   Paar-Memorisierung ist informationstheoretisch unmöglich.
2. **Realisierender Init, lernbar:** r_c = 0,5c, beta_c = −0,25c²
   (die parabolische argmax-Form) als Startpunkt — frozen klemmt
   (0,36), ohne Init findet die Optimierung die Lösung nicht (Gerade
   wild, held 0,000).
3. **Klassen-balancierter Loss** (1/n_Klasse): sonst dominieren
   häufige Antworten den CE und seltene Klassen kippen systematisch.
Ergebnis: gelernte Zahlengerade als Attraktor (Seeds quasi-identisch,
monoton, äquidistant) — held-out 1,000. Wichtig gegen 8.80-Fehlschluss:
rauschige KLASSEN-Kohärenz der rohen Embedding-Differenzen beweist
KEINE Unlösbarkeit — sie beschreibt den schlechten Optimierungspfad
der linearen Form, nicht den Lösungsraum (closed-form existierte,
Fit 2e-6). Erst Exakt-Lösbarkeit prüfen, dann Optimierung anklagen.

**Karten-Klausel für Ein-Token-Operanden (KORRIGIERT nach Review
8.148):** Die rohen Wort-Embeddings des Wirts tragen keine
generalisierbare Zahlengerade für Ein-Token-Operanden. Das ehrliche
Maß ist die LOO-lineare Sonde: one..nine R² 0,027, zero..nine
R² −0,104 (held 3/3 fällt); die früher zitierten PCA-1-Werte
(0,47/0,43) sind varianz-, nicht label-optimal und als Beleg
irreführend (n=9/d=128 macht Full-Fit trivial 1,0). Das getestete
Ein-Token-Organ (1 Seed, 3 held) generalisierte nicht (held 0,333,
r² 0,745). VORSICHTIGE REGEL: der getestete Ein-Token-Weg
generalisiert nicht; die Parser-Struktur (w14/w15) ist ein
funktionierender Weg — aber „nur Parser" ist aus den Daten nicht
ableitbar (die w15-Emission ist Dead Code: 96/96 ohne Organ-Lesen).
(Deckt sich mit 8.80: die Sequenz-Form der Aufgabe ist ein tragender
Weg, nicht der einzige.)

## Kompositions-Formel v3: das Bias-Gitter-Gesetz (Fable R10)

Kontroll-Messung (Agent 1): Addition kollabiert im Dreiecks-Gitter
exakt wie Subtraktion (train 1,000 / held 0,000, ±1-Fehler) — v2 war
unvollständig. Die vollständige Aussage:

**Das nötige induktive Bias des Organs skaliert mit der Schiefe des
Trainings-Gitters.** Auf dem vollen Quadrat (a,b unabhängig) genügt
der Symmetrie-Bias R(e_a+e_b); auf dem Dreieck (b < a) existiert für
jede freie separable Kopf-Form eine ±1-Fehllösung, die alle
Trainingszellen fittet — nur die harte arithmetische Form
delta_c = scale·(c·u − q·c²) (strukturell unimodal, u aus gelernten
Zahlengeraden) schließt den Fehllösungsraum. Zwei unabhängige
Lösungen konvergieren auf dieselbe Zutat (Ordinalität strukturell:
1D-Bottleneck+Init+Balance bzw. Hart-Form) — und die härtere Struktur
gewinnt in Größe UND Generalisierung (8,3k Params, held 1,000).
Belege: w3_sub_hartform.py/.json, w3_sub_exp2.json (Dreiecks-
Kontrolle), w3_sub_exp.json (Formen-Batterie).

## Das Prozedur-Organ-Gesetz: F6 auf Organ-Ebene (Fable R13)

Ein Organ mit eigenem O(1)-Zustand extrapoliert Prozedur-Länge
unbeschränkt; ein flaches Organ gleicher Kopf-Form und Kapazität
extrapoliert exakt gar nicht:

    acc_t = acc_{t-1} + sgn_t · v(e_t),   sgn geschaltet vom
    Operator-Token (frozen-Embedding-Match),   delta_c = s0·(c·acc − q·c²)

Gemessen (w7_ketten_organ): Training NUR auf 2-Operator-Ketten →
L3/L4/L6/L8/L12 alle 1,000 (je 96 frische Ketten); flaches Slot-Organ:
L2 1,000, L3 0,042 (Zufall). Die Rekurrenz trägt die Generalisierung —
"train short, deploy unbounded" (F6 des Wirts) ist damit eine
Eigenschaft, die sich auf ORGANE vererbt, wenn deren Zustand die
Aufgaben-Invariante (hier: der laufende Wert) exakt fasst. Zusammen
mit dem Bias-Gitter-Gesetz (v3) ergibt sich die Design-Regel für
algorithmische Organe: (1) identifiziere die Invariante der Prozedur,
(2) gib dem Organ genau diesen Zustand (nicht mehr), (3) Hart-Form-
Readout, (4) klassen-balanciert trainieren — Länge ist dann kein
Trainings-Parameter mehr. Offene Atlas-Frage: welche Prozedur-Klassen
fasst ein k-dimensionaler Organ-Zustand (Zählen: k=1; Klammer-Tiefe:
k=1; Traversierung: k=?) — die Chomsky-artige Hierarchie der Organe.

**Stack-Stufe + Kapazitäts-Klausel (Fable R16, w9):** Die Chomsky-
Stufe 3 (Top-of-Stack) bestätigt das Gesetz mit einer Schärfung.
Zustand s ∈ R^(k·D_E) mit Block-Shift-Matrizen:

    push(x): s ← A_push·s + B·(P e_x),   close: s ← A_pop·s,
    Readout: score(sym) = (P e_sym)ᵀ W s

(1) Der Struktur-Prior ist hier NICHT optional: frei initialisierte
A-Matrizen lösen die End-Tiefe-Metrik (Recency-Shortcut), aber nie
das LIFO (Verschüttet-Test 0,083; pop∘push-Fehler 0,504). Selektions-
Charakter macht die Aufgabe frei lernbar, nicht den Algorithmus.
(2) KAPAZITÄTS-GESETZ: Die Stack-Kapazität ist eine Stufenfunktion
exakt an der Slot-Zahl k und kommt ALLEIN aus der Zustandsgeometrie —
Training bei Max-Tiefe 2 liefert verschüttet-Accuracy 1,000 bis
genau k und Zufall ab k+1 (gemessen k=4 und k=6, Prognosen vorab).
Mechanismus: push/pop sind Block-Permutationen (tiefen-äquivariant);
die gelernten Teile P/W/B operieren nur am Top-Slot.
(3) KAPAZITÄTS-CHIRURGIE: Darum ist Kapazität ein DEPLOYMENT-
Parameter — P/W/B eines 4-Slot-Trainings in eine 8-Slot-Geometrie
mit exakten Shifts verpflanzt gibt v8 = 1,000 / v9 = 0,125 mit null
Gradientenschritten (w9_kapazitaets_chirurgie.json). Design-Regel
(5): die Invarianten-Geometrie wird beim Verschiffen dimensioniert,
nicht beim Training.

## DAS KRISTALLISATIONS-GESETZ (Fable R17, w3-Skalen + w9-Kapazität)

Organ-Chirurgie (Verpflanzen gelernter Teile in erweiterte Strukturen,
null Gradientenschritte) gelingt genau dann, wenn die Struktur EXAKT
ist; gelernte Struktur-Approximationen extrapolieren nicht:

  - Negativ, mit Mechanismus (w3_skalen_chirurgie.json): feinjustierte
    Ra-Zeilen verlassen die Hart-Form-Mannigfaltigkeit (weight-Fit
    R² 0,804) → Zeilen-Extrapolation kippt zum Rand, Bestand
    degradiert (0,59). Die Hart-Form war dort Init, nicht Constraint.
  - Positiv (w3_skalen_chirurgie_v2.json): der inhaltstragende
    gelernte Teil — die Zahlengerade phi, R² 0,998 — wird per
    Selbst-Kalibrierung (alpha/gamma least squares, label-frei) in
    ein EXAKT realisierendes Readout c = 4..16 verpflanzt, Dosis
    S0 = 10 überstimmt den Wirt-Prior (Delta-argmax skaleninvariant,
    Prior nicht). Ergebnis: alle 67 add-Summen 1,000 end-to-end,
    davon 33 Summen 11–16, die WEDER Organ NOCH 27B-Donor je sahen;
    NLL bitgleich.

KRISTALLISATION als Rezept-Schritt (ergänzt v3-Kanon um Schritt 5):
nach dem Training die gelernte Readout-Approximation durch die exakt
realisierende Konstruktion aus der gelernten Invariante ersetzen
(trainieren → Invariante extrahieren → exakte Form einsetzen). Danach
sind Kapazität (Slots), Wertebereich (Kandidaten) und Dosis (S0)
Deployment-Parameter. Der Fit-R² der Kristallisation ist zugleich ein
struktureller VERIFIER: er misst, ob das Organ die Struktur wirklich
gelernt hat, ohne eine einzige Aufgabe zu stellen.

## DAS QUOTIENTEN-GESETZ (v4, ENDFORM — ersetzt v1-v3-Deutungen; Fable R14)

**Gesetz:** Freie Kopf-Familien (freie Klassen-Gewichte + freie
Operanden-Vektoren) generalisieren GENAU auf den Symmetrie-Quotienten
des Trainings-Gitters: eine held-Zelle wird korrekt gdw. ihr Orbit
unter der Kopf-Symmetriegruppe eine Trainingszelle enthält (symmetrischer
Kopf: (a,b)~(b,a); separabler Kopf: triviale Gruppe → nichts). Alles
echt-Ungesehene wird memorisiert und fällt (±1 oder Flucht zum
Kandidaten-Rand). Kein Anker-Set repariert freie Köpfe (Diagonale und
Voll-Zeile beide gemessen: held 0,0). Echte Komposition liefern NUR die
ordinalen Strukturformen (Hart-Form, Encoder-Bottleneck).
**Prospektiv bestätigt, parameterfrei:** held-acc 0,375 aus reiner
Orbit-Zählung vorhergesagt (per-Zelle-Zuordnung vorab in JSON), gemessen
0,375, 16/16 Zellen korrekt klassifiziert, 2 Seeds identisch; sep 0,000
= 0,000. Dazu Kapazitäts-Satz (DOF-Zählung; Ungleichungs-Kapazität,
nicht lineare Dimension: 23 permutierte Ziele zu 0,957 gefittet bei
d_e=8) und Drift-Polytop-Theorie (LP-berechenbare Verwundbarkeit V(z),
erklärt die ±1-Magnitude; einseitige Margen → Rand-Flucht, relevant
fürs Margin-Gate). Zwei ehrliche Falsifikationen auf dem Weg (Anker-
These, lineare Kapazitäts-Schwelle) — beide angegriffen, daraus fiel
das Gesetz. w6_gitter_gesetz.md + w6_gitter_test{,2,3}.py/.json.

**KORREKTUR zu v1/8.76:** Das AdditivOrgan-held-0,900 war KEINE
Komposition — alle 9 korrekten held-Zellen hatten ihre Spiegelzelle im
Training (für den Summen-Kopf identische Eingabe); die einzige echt-
ungesehene Zelle (6+6) fiel immer. "Additivität ⇒ Memorisierung
unmöglich" ist quantitativ widerlegt. **KORREKTUR zu v3:** Die
Dreiecks-"Schiefe" war nie die Variable — das Dreieck scheitert, weil
seine Spiegel außerhalb des Gitters liegen. AUFWERTUNG: Hart-Form- und
Encoder-1,000er (sub, Dual, W7-Ketten bis L12, W8-Zählen) sind die
einzigen ECHTEN Kompositionen des Projekts — ihre held-Zellen haben
keine Orbit-Vertreter im Training. Offen (§8 des Docs): die
Symmetriegruppe der dritten Fähigkeits-Klasse (Antisymmetrie mit
Antwort-Flip) und ob ihr Quotient nie gezeigte Zellen rettet.

## v4.1: Die Selektions-/Relations-Dichotomie (Fable R16, beidseitig prospektiv)

Präzisierung des Quotienten-Gesetzes: Die Orbit-Schranke ist eine
garantierte UNTERE Schranke; SCHARF ist sie genau für RELATIONS-
Aufgaben (Antwort = neue Klasse, braucht global koordinierte Struktur —
add/sub: R3 traf 0,375 auf die dritte Nachkommastelle). SELEKTIONS-
Aufgaben (Antwort IST ein Operand — max/min) brechen sie strukturell:
die lokale Paar-Mitgliedschafts-Lösung ist der SGD-Attraktor (geschlossene
Konstruktion in w6_gitter_gesetz.md §9); gemessen 1,000 auf allen
Zell-Typen, 6/6 Läufe, Schranke um 4,5x gebrochen — Vorhersage vorab.
**MaxMin-Hart-Form:** u = ½(v_a+v_b) + ½·σ·|v_a−v_b|, σ aus dem
Operator-Token — max/min als glatte Funktion EINER Zahlengeraden.
Damit trägt eine einzige Formfamilie (Parabel über gelernter
Zahlengerade) vier Operationen: add, sub, max, min. Design-Konsequenz
unverändert: nur die ordinale Strukturform komponiert klassen-
unabhängig. Offene Grenze der Dichotomie: teilweise Selektions-
Struktur (closer-to, Median, mod als harte Relations-Kontrolle).

## v4.2 + v4.3: Gate-Kanal, Gruppendarstellung, Faktorisierbarkeit (Fable R18; Voll-Detail in s3/w6_gitter_gesetz.md §10)

Drei Schärfungen des Quotienten-Gesetzes aus R5/R6 (mod3, Median,
closer-to, Paritäts-Auswahl; alle Prognosen vorab committet):

1. STRUKTURFORM = GRUPPENDARSTELLUNG. Die Hart-Form muss die Gruppe
   der Aufgabe tragen: ordinal → R/Parabel (add, sub, max, min,
   median); zyklisch → Z_n/Kreis, delta_c = scale·cos(2π/n·(u−c))
   (mod3: 1,000/1,000 als einzige Form). Zwei Formfamilien, ein
   Prinzip.
2. GATE-KANAL (v4.2): Das 4-Gramm-Gate ist ein nichtseparabler
   Rechen-Kanal — argmax[h_c + g(a,b)·delta_c] ist reicher als der
   separable Kopf. Er rettet Relations-Aufgaben mit GROBEM
   Antwortraum (mod3, 3 Klassen: held 0,857; Gate-off: 0,000) und
   versagt bei feinem (add, 13 Klassen: exakt Orbit-Schranke 0,375).
   Klassen-Granularität ist die dritte Achse. Konsequenz fürs
   Organ-Design: strikte Organe brauchen Gates, die nur ROUTEN.
3. FAKTORISIERBARKEIT (v4.3): Die Selektions-Seite spaltet sich.
   Mitgliedschaft (Antwort ∈ Operanden) generalisiert IMMER frei
   (Paritäts-Auswahl: mitglied_rate 1,0 in jedem Seed). Die AUSWAHL
   unter den Mitgliedern generalisiert frei genau dann, wenn das
   Kriterium klassen-lokal faktorisiert (max/min/median/closer-to —
   closer-to überbrückt sogar Faktor-Löcher, 9/9). Paar-gekoppelte
   Kriterien (welcher Operand ist der gerade?) verhalten sich
   relational: die koordinierte Lösung existiert (Struktur-Form
   1,000/1,000), ist aber nicht der SGD-Attraktor. Die entscheidende
   Aufgaben-Eigenschaft ist die Faktorisierbarkeit des Kriteriums,
   nicht die Antwortmenge.

## Die KARTEN-FAMILIE der Prozedur-Organe (Fable R19, w10)

Prozedur-Gesetz + v4.2 vereinigt: EINE Akkumulator-Struktur
(acc += v(token), O(1)-Zustand), und die Aufgaben-Gruppe wählt nur
die KARTE des Werte-Encoders und das Readout-Gitter:

    (R,+)   add/sub-Ketten:  v ≈ α·n + γ,      Gitter {1..N}
    (R⁺,·)  mul-Ketten:      v ≈ α·log n + γ,  Gitter {log 1..log N}
    Z_n     mod:             Kreis-Readout cos(2π/n·(u−c))

Beleg (w10_mul_organ.json, Prognosen vorab): das log-Gitter-Organ
lernt die log-Gerade selbst (r²_log 0,999 vs r²_linear 0,927), nach
Log-Gitter-Snap L2/L3/L4 = 1,000 (trainiert nur auf 2 Faktoren);
lineares Gitter bricht (L3 0,021). Schärfung von H4: bei mul ist die
Kristallisation nicht Politur, sondern konstitutiv (roh L4 0,094 —
Invariante perfekt, gelernte Readout-Approximation nicht). Die Karte
ist der Gruppen-Isomorphismus in den additiven Zustand — neue
Operation heißt: neue Karte finden, nicht neue Organ-Klasse bauen.

**Rausch-Klausel des Kristallisations-Gesetzes (Fable R20, w7_rausch):**
Ein kristallisiertes Organ ist ein digitaler Rechner auf analogem
Substrat. Unter Störung sigma auf der Zahlengeraden gilt parameterfrei

    acc(sigma, L) = P( Σ_{i=1..L+1} e_i = 0 ),
    e_i iid,  p_k = Φ((k+½)/σ) − Φ((k−½)/σ)     (Faltung)

— gemessen auf 12 Zellen mit max. Abweichung 0,070 (Median ~0,01);
die naive Alle-korrekt-Schranke (1−2Φ(−½/σ))^(L+1) gilt bis σ≈0,2.
Konsequenzen: (a) Rundungsfehler kompensieren sich (Random Walk der
Netto-Drift) — lange Ketten sind robuster als die naive Schranke
sagt; (b) Quantisierungs-Verträglichkeit ist eine RECHNUNG: ein
Deployment (int4 etc.) ist sicher, wenn sein induziertes sigma auf
der Zahlengeraden klein gegen den halben Gitterabstand ist — der
Mechanismus hinter "Organe überleben int4" (QAD, 8.84).

**Deterministischer Zusatz der Rausch-Klausel (Fable R20b, w7_quant):**
Quantisierung stört nicht stochastisch, sondern pro Träger-Token
deterministisch: delta[n] = round(v_q(n)) − n. Eine Eingabe bricht
gdw. ihre Netto-Drift Σ sgn_i·delta[n_i] ≠ 0 — der Kalkulator sagt
damit ZELLGENAU vorher, welche Eingaben ein zu stark quantisiertes
Organ falsch beantwortet (int2: Übereinstimmung 0,97–0,99; int4 UND
int3: alle Träger stabil, acc exakt 1,000 — Bank-Artefakte sind
int3-fest). Verifikation, Robustheit und Deployment kristallisierter
Organe sind vollständig Rechnungen auf der Invariante.

**R7-Präzisierung von v4.2 (Fable R21, w6_r7):** Die Gate-Rettung ist
ein 3-KLASSEN-SONDERFALL — U-Rettungsrate als Stufe: mod3 1,0; mod4,
mod5, mod7, add13 alle 0,0 (die "glatter Abfall"-Prognose fiel). Die
Zyklus-Hart-Form skaliert dagegen unbegrenzt (mod5/mod7 je 1,000).
Und der Rechen-Kanal ist ABTRENNBAR: Hart-Form-Kopf + hartes
Route-Gate hält 1,000/1,000 bei bitgleicher NLL und gate_txt 0,0,
während der freie Kopf ohne MLP-Gate ehrlich auf die Schranke fällt.
Standard-Design für strikte Organe: STRUKTUR RECHNET, ROUTE GATED.
Offen: warum die Stufe exakt bei 3 Klassen liegt.

**Leiter-Klausel des Prozedur-Organ-Gesetzes (Fable R22, w11):** Die
Organ-Chomsky-Leiter steht auf drei Stufen, und der Preis einer Stufe
ist eine ZUSTANDS-Dimension, nicht Parameter:

    Stufe 2 (regulär+):        k=1-Akkumulator, additiv  (Ketten)
    Stufe 3 (kontextfrei):     Shift-Stack               (Top-of-Stack)
    Stufe 4 (kontextsensitiv): k=2-Zähler, additiv       (a^n b^n c^n)
    Stufe 4' (mildly cs/TAG):  k=2-Fingerprint, MULTIPLIKATIV
                               h ← r·h + v(sym)          (ww, w13)

a^n b^n c^n: c1=#a−#b, c2=#b−#c, Readout s0·(θ−c1²−c2²) — nur s0/θ
lernbar, Training n<=3, Eval n=4..8 alles 1,000 inkl. Fast-Balanced;
der k=1-Kontrast ist auf c2-Verwechslern strukturell blind (0,0).
Jede Stufe erbt F6 (train short, deploy unbounded) und ist per
Kapazitäts-Gesetz beim Deployment dimensionierbar. Offene Atlas-
Frage: k Zähler + Stack zusammen = mildly context-sensitive (TAG).
NACHTRAG (R24, w13): ww ist gelöst — nicht über Zähler+Stack, sondern
über die multiplikative Fingerprint-Rekurrenz (Rabin-Karp als Organ):
Zähler-Kontrast fällt exakt auf 0,0 (Permutations-blind), der
Fingerprint trifft 1,0 auf allen Klassen. Die Leiter klettert über
REKURRENZ-KLASSEN (additiv → Shift → multiplikativ), je 2 Skalare.

## DAS STRIKTE ORGAN-REZEPT (Synthese, Fable R23 — w6_r7 + Karten-Familie + Kristall)

    STRUKTURFORM  trägt die Gruppe der Aufgabe   (Parabel/R, Kreis/Z_n)
  + KARTE         zieht die Aufgabe auf die Zahlengerade  (id, log, ...)
  + ROUTE-GATE    gated nur Zuständigkeit, parameterlos (Embedding-Match)
  + KRISTALL      ersetzt die gelernte Approximation durch die exakte Form

Ergebnis: ein Organ ohne freien Rechen-Kanal — jede Generalisierung
kommt beweisbar aus der Struktur (Route-Gate-Test: freie Köpfe fallen
exakt auf ihre nackte Klemm-Decke, 0,548 vs 0,55; die Strukturform
verliert nichts, NLL bitgleich). MLP-Gates sind eine verdeckte
Rechen-Hintertür (1-Parameter-Pfad im Ranking-Raum, rettet nur den
3-Klassen-Fall); strikte Organe nehmen Route-Gates. Verifikation
(Invarianten-R²), Robustheit (Drift-Faltung) und Deployment
(Träger-Tabelle, Kapazität, Dosis) sind danach Rechnungen.

## DAS BRÜCKEN-GESETZ + KRISTALL-GITTER-KLAUSEL (Fable R25, w12)

BRÜCKEN-GESETZ: Die Signal-Schwelle der Destillation (R-8.116) ist
eine Eigenschaft des ERNTE-FORMATS, nicht des Donors. Targets sind
Klassen-Verteilungen — format-frei. Rezept: ernte im stärksten Format
des Lehrers (Ziffern), aggregiere pro Klasse
P_k = mean(P(·|prompt): klasse(prompt)=k), trainiere das Organ im
Format des Wesens mit KL gegen P_{klasse(zelle)}. Beleg mod3: Wort-
Lehrer 0,39 → Organ tot (r² 0,32); DERSELBE Donor über die Brücke
(Reinheit 3/3) → 81/81 roh und kristallisiert, 0 neue Donor-Aufrufe.
Folge: jede Ziffern-Fähigkeit des Donors ist wort-organ-fähig; die
Multi-Token-Kandidaten-Falle ist gegenstandslos.

KRISTALL-GITTER-KLAUSEL (schärft H4): Das Snap-Gitter der
Kristallisation muss das Gitter der AUFGABEN-GRUPPE sein:
    ordinal (R,+):    Ganzzahl-Snap,      Verifier r²_linear
    multiplikativ:    log-Gitter-Snap,    Verifier r²_log
    zyklisch (Z_n):   zirkulärer Phasen-Snap auf die Klassen-Zentren,
                      Verifier = zirkuläre RESULTANTE
Falsches Gitter zerstört ein perfektes Organ (Ganzzahl-Snap auf die
Kreis-Karte: 1,000 → 0,35 bei r²_lin 0,077 und Resultante 0,9998) —
der lineare Fit ist für zyklische Organe das falsche Messgerät, kein
Organ-Urteil. Bank-Karten-Erkennung: max über die drei Verifier.

## T7-VEREINIGUNG: Schwelle und Brücke sind EIN Satz (Fable R26, FERTIG-Scout)

Das Grounding-Theorem T7 (_codex_lab/GROUNDING_THEORY.md): Zustände,
die nur durch Interventionen außerhalb der verfügbaren Menge I
getrennt werden, sind im identifizierbaren Quotienten identisch.
Übersetzt auf Destillation: **Das Prompt-Format ist eine Intervention
am Donor.** Die Wort-Prompts (mod3: 0,39) trennen die Restklassen
nicht — im zugehörigen Quotienten EXISTIERT die Zahlengerade nicht,
egal wie viele Logits man sammelt (Signal-Schwelle, 8.116). Die
Ziffern-Prompts sind eine ANDERE Intervention, deren Quotient die
Klassen trennt (0,965) — die Format-Brücke (8.118) wechselt die
Interventions-Menge, nicht den Lehrer. Merksatz: mehr Beobachtung
hilft nie, nur andere Interventionen. Dazu T11 (Dobrushin-Budget
Σ ε_i·Π δ_j): früh kristallisieren dämpft die ganze Kette; und T4:
Schwellen-Verifier brauchen eine deklarierte AMBIGUOUS-Zone, sonst
ist die Organ-Klassifikation keine Äquivalenzrelation.
Werkzeug-Fund: quotient.py (FiniteInterventionalModel) rechnet
Orbit-Schranken und Bank-Vollständigkeit konstruktiv
(anchor_closure_certificate liefert das fehlende Klassen-Paar).

## ARITHMETIK-VOLLSTÄNDIGKEIT DES WESENS (Fable R27, w14+w15)

Drei Klauseln schließen die Zahlen-Linie:

1. KOMPOSITIONS-KLAUSEL der Leiter: Rekurrenz-Klassen sind
   VERSCHACHTELBAR — der multiplikative Parser h ← 10·h + v(d)
   (w13-Klasse) läuft im additiven Akkumulator (w7-Klasse; Operator
   schließt Segment, acc += sgn·h). Die Kristallisation der INNEREN
   Invariante (Ziffern-Gerade, r² 0,999) trägt die ganze Komposition
   (T11: früh kristallisiert dämpft die Kette). Probe exakt:
   "one two"=12, "two one"=21, "one zero zero"=100.
2. EINGABE unbegrenzt: jede Zahl ist eine Ziffernwort-Folge über
   zero..nine — das Vokab-Limit (max sixteen) war lexikalisch, die
   Lösung ist strukturell.
3. AUSGABE unbegrenzt (die 8.27-Wand fällt von der Struktur-Seite):
   Wert → divmod-Stellen-Dekomposition → tokenweise Logit-Dosis auf
   dem Ziffern-Slice des lebenden Wirts. 96/96 Produkte bis 891
   sequenz-exakt. Freie Generierung scheiterte, weil sie HOFFTE;
   die Emission dosiert pro Schritt.

Summe: unbegrenzte Eingabe + exakte Rechnung + unbegrenzte Ausgabe
aus 10 Ziffernwörtern, EINER gelernten Zahlengeraden und exakten
Formen. Alles Weitere (mul2x2, Ketten beliebiger Werte) ist
Konstruktion, keine Forschung mehr.

**ATOMIZITÄTS-KLAUSEL des Kristallisations-Gesetzes (Fable R28,
8.122-Korrektur):** Kristallisation ist ATOMAR — Karte und Readout
werden ZUSAMMEN durch die exakte Form ersetzt, nie halb. Eine
Mischform (v gesnappt, gelerntes Readout auf der alten Skala) fällt
durch Skalen-Mismatch (w14-Teilzelle 0,2; dasselbe Muster wie das
v1-Negativ 8.98). Volle Kristall-Pfade: Parser-Probe exakt, w15
96/96. Gefunden vom Paper-Agenten beim JSON-gegen-Klausel-Abgleich —
der Abgleich JEDES Artefakt-Felds gegen die registrierten Klauseln
ist ab jetzt Teil des Protokolls.

## ATLAS-INDEX DER FABLE-NACHT (R-Serie, 19./20.08.2026)

| R | Gesetz | Kern-Beleg |
|---|---|---|
| R13 | Prozedur-Organ-Gesetz (F6 auf Organen) | L2→L12 = 1,000 |
| R14 | Quotienten-Gesetz v4 (Orbit-Schranke) | add 0,375 exakt |
| R16 | v4.1 Dichotomie + Stack/Kapazität | Verschüttet-Test; Slots exakt |
| R17 | Kristallisations-Gesetz | 67/67; L12 0,84→1,0 |
| R18 | v4.2/v4.3 Gate-Kanal, Faktorisierung | Stufe bei 3; Mitgliedschafts-Boden |
| R19 | Karten-Familie (Gruppen-Isomorphismen) | mul=add unter log |
| R20 | Rausch-Klausel (+ deterministisch) | Drift-Faltung 0,054; int2 zellgenau |
| R21 | Route-Gate ("Struktur rechnet, Route gated") | Decke 0,548=0,55 |
| R22 | Leiter über Rekurrenz-Klassen | a^n b^n c^n mit 2 Params |
| R23 | Das strikte Organ-Rezept | kein freier Rechen-Kanal |
| R24 | ww / multiplikative Rekurrenz | Zähler-Kontrast exakt 0,0 |
| R25 | Brücken-Gesetz + Kristall-Gitter | 0,39 tot → Brücke 81/81 |
| R26 | T7-Vereinigung (Format = Intervention) | Schwelle ≡ Brücke |
| R27 | Arithmetik-Vollständigkeit | w14 "100"; w15 96/96 |
| R28 | Atomizitäts-Klausel | halbe Kristallisation 0,2 |
