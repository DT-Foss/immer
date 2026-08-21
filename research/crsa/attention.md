**Ja — und zwar richtig massiv.** Der Parallelstand enthält nicht nur ein paar lose Ideen, sondern schließt mehrere unserer offenen Attention-Fragen bereits experimentell und liefert zusätzlich genau die Architekturregel, die uns beim Ausbau noch gefehlt hat.

## 1. Der größte Falsifikator ist bereits erledigt: Der Gewinn verschwindet nicht mit der Skala

Im v0.4-Draft war ausdrücklich festgelegt: Der kritische Falsifikator ist, dass der Prefix-Sinkhorn-Gewinn bei wachsender Breite gegen null fällt. Dazu waren Breiten-, Iterations- und Spektral-Ablationen angekündigt. 

Der Parallelstand hat genau dieses Programm inzwischen durchgezogen:

| Breite \(d\) | Gewinn Softmax − Prefix-Sinkhorn, \(k=1\) |
| ------------ | ----------------------------------------- |
| 128          | **+0,984 bpb**                            |
| 256          | **+1,106 bpb**                            |
| 384          | **+1,173 bpb**                            |
| 512          | **+1,182 bpb**                            |
| 1024         | **+1,184 bpb**                            |

Bis \(d=512\) umfasst die gehärtete Serie 19 vergleichbare 400-Step-Läufe. Der Fit bleibt negativ skaliert, \(\alpha=-0{,}149\), \(R^2=0{,}974\); der zusätzliche \(d=1024\)-Punkt zeigt dann die eigentliche Form: **Wachstum bis ungefähr \(d=512\), danach ein Plateau bei etwa 1,18 bpb.** Der Effekt wird also weder weggekapazitiert noch zum Kleinmodellartefakt, sondern wird skalenstabil.  

Das ist eine wesentlich stärkere Paper-Aussage als vorher:

> **Causal Prefix-Sinkhorn liefert keinen schrumpfenden Regularisierungsbonus, sondern einen skalenstabilen Mechanikgewinn von ungefähr 1,18 bpb ab mittlerer Modellbreite.**

Auch die Iterationsfrage ist praktisch entschieden:

\[ k=1:\;2{,}1784 < k=2:\;2{,}1854 < k=5:\;2{,}4340. \]

**Eine Iteration ist der Hauptoperator.** Mehr Iterationen überbalancieren und bewegen die Matrix zu stark in Richtung lokaler beziehungsweise diagonaler Struktur. Gleichzeitig liegt die gemessene Spaltenabweichung bei Prefix-Sinkhorn bei 0,105 gegenüber 0,899 für Softmax — **8,6-mal näher an der balancierten Struktur**. 

Damit sind drei offene Punkte des Attention-Papers bereits zugeschweißt:

- Skalenfalsifikator bestanden.
- \(k=1\) als kanonische Iterationszahl bestätigt.
- Der Spaltenbalance-Effekt ist direkt messbar und nicht nur eine Namensgeschichte.

## 2. Die Langstreckenlösung ist ebenfalls bereits da: ein unveränderbarer Free-Head

Unser eigener Head-Routing-Lauf hat den zuvor offenen Konflikt jetzt sauber gelöst:

- Reines Prefix/RAPS beziehungsweise reine Recency kann auf langen Copy-Aufgaben zusammenbrechen.
- **Sobald mindestens ein Head vollständig gewöhnliche kausale Softmax-Attention behält, liegt Lag-48-Copy wieder bei 1,000.**
- Gleichzeitig bleiben die Byte-Modell-Gewinne erhalten.

Die versiegelte Drei-Seed-Testbestätigung ist inzwischen vollständig:

| Architektur                       | Test-bpb                | Gewinn zu Softmax        | Seed-Siege |
| --------------------------------- | ----------------------- | ------------------------ | ---------- |
| **3 Local + 1 Free**              | **4,720256 ± 0,015395** | **−0,099746 ± 0,012807** | **3/3**    |
| **2 Local + 1 Balanced + 1 Free** | **4,726772 ± 0,010032** | **−0,093230 ± 0,007312** | **3/3**    |
| 1 Local + 1 Balanced + 2 Free     | 4,735188 ± 0,007419     | −0,084814 ± 0,004023     | 3/3        |
| Softmax                           | 4,820002 ± 0,003515     | —                        | —          |

REPORT.mdMD

Und im langen Copy-Screen lagen **sämtliche elf Free-Head-erhaltenden Route-Varianten bei 1,000 Accuracy**, während RAPS, reine Recency und die alten vollständig spezialisierten Routen kollabierten.

REPORT.mdMD

Das liefert eine glasklare Architekturregel:

> **Langstreckenfreiheit darf keine erlernte Hoffnung sein. Sie muss eine architektonische Invariante sein: mindestens ein unangetasteter kausaler Softmax-Head pro Layer.**

Ebenso wichtig: Ein expliziter Self-Head ist nicht nötig. Die besten Free-Head-Konfigurationen haben `sh=0`. Der Residualpfad trägt den aktuellen Token bereits; einen ganzen Attention-Head dauerhaft zur Identität zu machen, verschwendet Kapazität.

Der neue Grundbaustein ist daher nicht mehr Self–Local–Balanced–Free, sondern kompakter:

\[ \boxed{\text{Local}\;|\;\text{Balanced}\;|\;\text{Free}} \]

mit mindestens einem freien Head.

## 3. Das Organ-Paper enthält genau die Routing-Lösung für den nächsten Moonshot

Der stärkste übertragbare Fund aus dem Organ-Programm ist dessen striktes Rezept:

\[ \text{Structural Form} + \text{Map} + \text{Route Gate} + \text{Crystal}. \]

Dabei trägt die Struktur die eigentliche Rechnung, die Map bringt die Aufgabe in den passenden Raum, das Gate entscheidet **nur die Zuständigkeit**, und die Kristallisation ersetzt eine gelernte Approximation anschließend durch die exakte Form. 

Das Paper zeigt zusätzlich, warum das wichtig ist: Ein frei lernbares MLP-Gate kann selbst zum verdeckten Rechenkanal werden. Der strikte Endzustand benutzt deshalb ein Gate, das ausschließlich routet und keine zusätzliche Berechnung einschmuggelt.  Der Zustandsrouter kann dabei extrem klein sein; im Organ-System reichen 384 Parameter für drei Klassen einschließlich „kein Organ zuständig“ bei 1,000 Accuracy. ORGAN-GRAFTING-DRAFT.mdMD

### Direkt auf Attention übertragen

Unser bereits implementierter adaptive Route-Operator mischt pro Query:

\[ A_{i,h} = g_S A_{\mathrm{self}} + g_L A_{\mathrm{local}} + g_B A_{\mathrm{RAPS}} + g_F A_{\mathrm{free}}, \]

mit harter Untergrenze

\[ g_F\ge \rho. \]

Er ist strikt kausal, differenzierbar und behält selbst bei fehlgeleitetem Gate immer einen garantierten freien Pfad. Die aktuelle Implementierung hat **30/30 Tests bestanden**, einschließlich:

- exakter kausaler Support,
- unveränderte Prefix-Ausgaben bei geändertem Future-Suffix,
- endliche und nichttriviale Gate-Gradienten,
- harte Free-Route-Untergrenze,
- Route-Usage-Instrumentierung.

Aber das Organ-Programm verrät uns die noch bessere Endform:

## **Adaptive scout, crystalline deployment**

1. Während des Trainings darf das adaptive Gate herausfinden, welche Route welcher Head und welcher Layer tatsächlich benötigt.
2. Danach werden die stabilen Headrollen aus den Gate-Verteilungen extrahiert.
3. Die weiche Gate-Approximation wird durch ein festes, exakt routendes Headprogramm ersetzt.
4. Mindestens ein Free-Head bleibt pro Layer strukturell erzwungen.
5. Das Modell wird kurz nachgehärtet und anschließend ohne Gate-Overhead betrieben.

Formal:

\[ r_h^\star = \arg\max_{r\in\{L,B,F\}} \mathbb E_{x,i}\big[g_{h,i,r}(x)\big], \]

danach

\[ A^{(h)}= \begin{cases} A_{\mathrm{local}}, & r_h^\star=L,\\ A_{\mathrm{RAPS}}, & r_h^\star=B,\\ A_{\mathrm{softmax}}, & r_h^\star=F. \end{cases} \]

Das ist **Crystallized Causal Routing**:

- adaptive Suche während der Entwicklung,
- harte strukturelle Route beim Deployment,
- keine Gate-Konfundierung,
- kein versteckter Rechenkanal,
- deutlich weniger Laufzeitkosten,
- beweisbar vorhandener Langstreckenpfad.

Und das ist keine beliebige Analogie: Genau derselbe Ablauf hat im Organ-Programm aus einer gelernten Approximation einen exakten, robusten und quantisierungsfesten Rechner gemacht.

## 4. Der sinnvollste konkrete Layer-Aufbau ist schon im Code vorbereitet

Unsere beiden bestätigten Spitzenkandidaten ergänzen sich:

- **3 Local + 1 Free** ist minimal schneller, leicht besser auf dem aktuellen Byte-Corpus und sehr billig.
- **2 Local + 1 Balanced + 1 Free** enthält den genuinen Prefix-Sinkhorn-Mechanismus und bleibt ebenfalls 3/3 vor Softmax.

Daraus folgt als stärkster nächster Architekturtest:

\[ \boxed{ \text{unterer Layer: }3L+1F \quad\longrightarrow\quad \text{oberer Layer: }2L+1B+1F } \]

Die Logik ist sauber:

- Unten werden lokale Byte-, Morphologie- und Nachbarschaftsmuster effizient extrahiert.
- Oben kontrolliert ein Prefix-balancierter Head die akkumulierte Key-Nutzung.
- In jedem Layer bleibt ein vollständig freier Retrieval-Head erhalten.

Der Gegenarm ist die umgekehrte Reihenfolge:

\[ (2L+1B+1F)\rightarrow(3L+1F). \]

Genau diese `RQ`- und `QR`-Programme sind in der vorhandenen Depth-Routing-Harness bereits vorgesehen. Damit testen wir nicht mehr wahllos 30 Operatoren, sondern die eine zentrale Frage:

> **Gehört kausale Balance unter die lokale Merkmalsextraktion oder darüber?**

## 5. Auch unsere Langstreckenbenchmarks bekommen durch das Organ-Programm einen wichtigen Fix

Im Stack-Organ-Lauf sah die naive Tiefenaufgabe zunächst so aus, als könnten freie Matrizen bis in ungesehene Tiefen generalisieren. Der Grund war ein versteckter Recency-Shortcut: Meist war die Antwort einfach das zuletzt gepushte Symbol. Erst der adversariale **buried-symbol test** — viele Pushes, danach fast alle wieder entfernen, sodass nur das älteste Symbol zählt — trennte echten Stack von Recency. Der strukturierte Stack erreichte 1,000; die freie Variante 0,083. CHANGELOG.mdMD

Genau dieselbe Disziplin braucht die Attention-Evaluation.

Die bisherigen Copy-Ergebnisse beweisen, dass ein Free-Head Distanzinformation erhalten kann. Sie beweisen noch nicht allein, dass jede Variante echtes inhaltsadressiertes Langstrecken-Retrieval kann. Deshalb wird die nächste synthetische Suite aus Anti-Shortcut-Aufgaben bestehen:

### Buried-key recall

Das gesuchte Key-Value-Paar liegt weit am Anfang. Danach erscheinen mehrere jüngere Distraktoren mit ähnlichen Keys und bewusst falschen Werten. Recency muss aktiv in die falsche Richtung zeigen.

### Random-lag multi-query copy

Nicht ein festes Lag für alle Queries, sondern pro Query ein anderes, zufälliges Lag. Ein statischer Recency-Kernel kann die Aufgabe nicht durch eine einzige bevorzugte Distanz lösen.

### Pointer chase

Die Sequenz enthält eine Kette von Verweisen:

\[ k_1\rightarrow k_2\rightarrow k_3\rightarrow v. \]

Die Antwort erfordert mehrere inhaltsadressierte Sprünge statt bloß einen alten Token zu kopieren.

### Collision recall

Mehrere gleiche Werte und ähnliche Schlüssel verhindern, dass Position oder Wertfrequenz als Shortcut reichen.

Der bereits implementierte Multi-Query-Recall-Generator ist der erste Schritt dazu. Der bisherige Recall-Aufbau ist als Headline-Test noch zu schwer beziehungsweise schlecht kalibriert: Softmax liegt selbst nach längeren Läufen weit vom Ceiling. Er wird deshalb nicht als falscher „Langstreckenbeweis“ verwendet, sondern durch die adversarialen, lösbaren Varianten ersetzt.

## 6. Der entscheidende Gesamtbefund

Wir haben jetzt nicht mehr nur einen neuen Attention-Operator. Wir haben drei Ebenen:

### **Operator**

Causal Prefix-Sinkhorn:

\[ A_{ij} \mapsto \frac{A_{ij}} {\sum_{r\le i}A_{rj}}, \]

streng kausal, exakt null Zukunftsgradient und mit eigenem Token-Attention-Quadranten. Der ursprüngliche Draft grenzt ihn als Prefix-Massen-Balancierung der kausalen Tokenmatrix ab; der damalige Sweep fand keinen direkten Operator. causal_prefix_sinkhorn_v0.4_polished.pdfPDF

### **Architektur**

Local–Balanced–Free mit einem unverlierbaren Free-Head:

\[ \boxed{H_L\cup H_B\cup H_F,\qquad |H_F|\ge1}. \]

Das beseitigt die Langstreckenschwäche, ohne den Trainingsgewinn aufzugeben.

### **Compiler beziehungsweise Kristallisation**

Ein adaptiver Scout entdeckt die geeignete Rollenverteilung. Danach wird sie in ein festes Head- und Layerprogramm kristallisiert:

\[ \text{adaptive routing} \rightarrow \text{measured roles} \rightarrow \text{fixed causal program}. \]

Das ist der eigentliche Moonshot: **Attention wird nicht mehr als einheitliche Matrixnormalisierung behandelt, sondern als kompilierbares kausales Routingsystem mit strukturell verschiedenen Organen pro Head und Layer.**

## Die klügste Reihenfolge steht damit fest

1. **Depth-Routing:** `R→Q`, `Q→R`, `R→R`, `Q→Q`, `F→F`; anschließend Drei-Seed-Bestätigung des Siegers.
2. **Adversarial Long-Range Suite:** buried key, variable lag, pointer chase und collision recall.
3. **Adaptive Scout:** Free-Floor-Sweep und vollständige Route-Usage-Auswertung.
4. **Crystallization:** adaptive Rollen in ein festes Headprogramm überführen und den Gate-Overhead entfernen.
5. **Skalierung des Architektursiegers:** \(d=128,256,512,1024\), während die ursprüngliche Prefix-Skalenserie bereits als harter Referenzanker steht.
6. **Paper v0.5:** Aus „neuem Normalisierer“ wird **Causal Prefix Routing: a scale-stable, crystallizable architecture for local, balanced, and unrestricted attention**.

Der Parallelstand hat uns damit tatsächlich die fehlenden Lösungen geliefert: **Skalierung geklärt, Langstreckenpfad geklärt, Routinggesetz geklärt und der nächste Architekturtyp praktisch schon vorgezeichnet.**





Ich habe **nicht wegen eines wissenschaftlichen oder technischen Problems** angehalten. Der vorherige Ausführungslauf ist schlicht gegen das harte Zeitlimit der Tool-Session gelaufen. Dadurch kam der Zwischenbericht heraus, obwohl die Pipeline noch nicht fertig war. Das war der gesamte Grund.

Danach habe ich die Arbeit weitergezogen. Der Operator bleibt dabei klar positioniert: **Prefix-Mass-Balancing direkt auf der kausalen Token-to-Token-Attention mit Zero-Future-Gradient-Garantie**. Genau diese enge Novelty-Claim formuliert bereits dein Draft; dessen Taxonomie besetzt den zuvor leeren Quadranten „causal decoding × token-to-token attention“.  

## **Was seitdem vollständig abgeschlossen wurde**

### **1. Marginal-Residual-Attention: 8/8 Variable-Lag-Runs**

Die neue adaptive Variante behält drei Spezialisten und einen vollständigen freien Softmax-Head. Bei einem Adaptionsbudget von 0{,}75 erreicht sie über zwei Seeds:

- **0,9363 Accuracy auf ungesehenen Lags**
- Fixed CRSA/Prefix-Sinkhorn-Q: **0,9293**
- Softmax: **0,7745**

Damit gewinnt die adaptive Variante bei der durchschnittlichen Out-of-Range-Accuracy. Fixed CRSA bleibt allerdings bei der Worst-Lag-Accuracy und beim mittleren Cross-Entropy-Wert stärker:

| **Architektur**        | **Unseen Accuracy** | **Worst Lag** | **Mean Bits** |
| ---------------------- | ------------------- | ------------- | ------------- |
| Marginal Residual 0,75 | **0,9363**          | 0,8943        | 0,6262        |
| Fixed CRSA             | 0,9293              | **0,9020**    | **0,5872**    |
| Softmax                | 0,7745              | 0,6982        | 1,1481        |

Die Konsequenz ist präzise: **Adaptive Balance erhöht die mittlere Reichweitengeneralisierung; eine fest verankerte Prefix-Sinkhorn-Rolle maximiert die Robustheit.**

[Marginal-Residual Variable-Lag Report](sandbox:/mnt/data/attention-moonshot-final/results/marginal_variable_lag/REPORT.md?_chatgptios_conversationID=6a860420-a154-83eb-b0e8-b315233ea966&_chatgptios_messageID=b762910a-4282-40bd-b4dc-e56d4afdb39a)⁠![Attachment.tiff](Attachment.tiff)

[Fixed-CRSA Variable-Lag Report](sandbox:/mnt/data/attention-moonshot-final/results/variable_lag_screen/REPORT.md?_chatgptios_conversationID=6a860420-a154-83eb-b0e8-b315233ea966&_chatgptios_messageID=b762910a-4282-40bd-b4dc-e56d4afdb39a)⁠![Attachment.tiff](Attachment.tiff)

### **2. Architektur-Skalierung: 46/46 Runs**

Der Vorteil verschwindet mit Breite nicht, sondern wird stärker:

| **Breite** | **CRSA-Δ gegenüber Softmax** |
| ---------- | ---------------------------- |
| 32         | −0,0333 bpb                  |
| 64         | −0,1757 bpb                  |
| 96         | **−0,2237 bpb**              |

Bei der Tiefenskalierung schlagen alle getesteten kausalen Routing-Programme Softmax in beiden Seeds. Die beste Balance-Dichte hängt von der Tiefe ab:

- Tiefe 1: eine CRSA-Schicht ist optimal.
- Tiefe 3: zwei CRSA-Fundamentalschichten sind minimal stärker als eine oder drei.
- Tiefe 4: ein lokales Fundament beziehungsweise zwei geroutete untere Schichten schlagen das Durchrouten aller vier Schichten.

Das kristallisiert eine klare Regel heraus:

**Prefix-Sinkhorn ist ein Fundamentoperator, kein Belag für jede Schicht. Untere Schichten strukturieren den kausalen Informationsfluss; obere Softmax-Schichten lesen ihn frei aus.**

[Kompletter Architektur-Skalierungsreport](sandbox:/mnt/data/attention-moonshot-final/results/architecture_scale/REPORT.md?_chatgptios_conversationID=6a860420-a154-83eb-b0e8-b315233ea966&_chatgptios_messageID=b762910a-4282-40bd-b4dc-e56d4afdb39a)⁠![Attachment.tiff](Attachment.tiff)

### **3. 1.200-Schritt-Langhorizont über drei Seeds**

| **Architektur**                                | **Validation bpb** | **Δ zu Softmax** | **Seed Wins** |
| ---------------------------------------------- | ------------------ | ---------------- | ------------- |
| 3 Local + 1 Free → Softmax                     | **3,799582**       | −0,250706        | 3/3           |
| 2 Local + 1 Prefix-Sinkhorn + 1 Free → Softmax | **3,811464**       | −0,238824        | 3/3           |
| Adaptive LLL → Softmax                         | 3,813661           | −0,236626        | 3/3           |
| Adaptive LLB → Softmax                         | 3,814822           | −0,235466        | 3/3           |
| Softmax → Softmax                              | 4,050288           | —                | —             |

CRSA hält den Gewinn also nicht nur in einem Kurzscreen, sondern über drei Seeds und 1.200 Schritte. Gleichzeitig demonstriert der variable-Lag-Test, warum der scheinbar minimal bessere reine Local-Arm nicht die Hauptarchitektur sein darf: Er kollabierte in einem der zwei Extrapolations-Seeds, während der einzelne Prefix-Sinkhorn-Head das gesamte System stabilisierte.

[Langhorizont-Report](sandbox:/mnt/data/attention-moonshot-final/results/long_horizon_foundation/REPORT.md?_chatgptios_conversationID=6a860420-a154-83eb-b0e8-b315233ea966&_chatgptios_messageID=b762910a-4282-40bd-b4dc-e56d4afdb39a)⁠![Attachment.tiff](Attachment.tiff)

## **Der nächste Moonshot ist bereits implementiert**

Ich habe direkt eine neue Generation gebaut: **Q-Residual Attention**.

Sie startet nicht wie Marginal Residual bei drei lokalen Heads. Ihr fester Anker ist die bereits robuste Q-Struktur:

\text{Local},\quad \text{Local},\quad \text{Prefix-Sinkhorn},\quad \text{Free Softmax}.

Jeder der drei Spezialisten darf sich queryabhängig zwischen Local und Prefix-Sinkhorn bewegen, aber nur innerhalb eines begrenzten Budgets:

b_{ih} = (1-\delta)a_h+\delta\,\sigma(g_{ih}),

wobei

a=(0,0,1)

die feste Local–Local–Balanced-Verankerung ist. Damit besitzt die Architektur gleichzeitig:

- einen festen Prefix-Sinkhorn-Anker,
- zwei lokale Spezialisten,
- queryabhängige Balanceverschiebung,
- einen vollständig unangetasteten Softmax-Head,
- exaktes Batch-/Streaming-Verhalten,
- strikt kausale Gates.

Bei \delta=0 ist der Operator **bitgenau identisch mit Fixed CRSA**. Bei größerem \delta kann er seine Balancekapazität dynamisch zwischen den drei Spezialisten verschieben.

Für diese neue Generation sind die gezielten Tests bereits grün:

- exakter CRSA-Grenzfall bei Budget null,
- reine Local- und reine Prefix-Endpunkte,
- kausaler Support und Zeilensummen,
- vollständiger unveränderter Softmax-Head,
- Null-Future-Dependency im Modell,
- lernfähige Gate-Gradienten,
- Batch-/Streaming-Gleichheit,
- Konfigurations- und Grenzwertvalidierung,
- korrekte Routing-Metriken und Candidate-Indizierung.

**9 neue gezielte Q-Residual-Tests sind bestanden.** Der letzte vollständige Suite-Lauf vor dieser Ergänzung hatte **69/69 Tests** bestanden; der nächste vollständige Clean-Cache-Lauf über alte und neue Tests stand beim erneuten Zeit-Cut noch aus.

[Q-Residual-Implementierung](sandbox:/mnt/data/attention-moonshot-final/src/attention_moonshot/operators.py?_chatgptios_conversationID=6a860420-a154-83eb-b0e8-b315233ea966&_chatgptios_messageID=b762910a-4282-40bd-b4dc-e56d4afdb39a)⁠![Attachment.tiff](Attachment.tiff)

[Q-Residual-Tests](sandbox:/mnt/data/attention-moonshot-final/tests/test_q_residual.py?_chatgptios_conversationID=6a860420-a154-83eb-b0e8-b315233ea966&_chatgptios_messageID=b762910a-4282-40bd-b4dc-e56d4afdb39a)⁠![Attachment.tiff](Attachment.tiff)

## **Was beim erneuten Cut gerade lief**

Der nächste Operator-Sweep war bereits gestartet:

1. **Prefix-Balance-Spektrum**
    \alpha\in\{0{,}5,0{,}75,1,1{,}25,1{,}5\} sowie Diagonal-Debit 2,3,4.
2. **Geometric Prefix-Sinkhorn**
    Vergangene Key-Nutzung wird mit
    U_i=A_i+\lambda U_{i-1}
    statt einer unendlich langen kumulativen Schuld geführt, mit
    \lambda\in\{0{,}9,0{,}95,0{,}98,0{,}99,0{,}995,1\}.
3. Danach vorgesehen und bereits vorbereitet:
   - Q-Residual-Budgetkurve,
   - position-freie Extrapolation von Kontext 96 bis 256,
   - 1.200-Schritt-Testbestätigung,
   - vollständiger Clean-Cache-Testlauf,
   - Claim Ledger, Final Report und distributierbares Repository-ZIP.

**Angehalten hat also ausschließlich die Ausführungsuhr. Die Forschungsrichtung ist nicht stecken geblieben: Aus Prefix-Sinkhorn ist inzwischen eine komplette kausale Attention-Familie mit fester CRSA-Architektur, adaptivem Marginal Residual, Geometric Prefix und dem neuen Q-Residual-Anker geworden.**