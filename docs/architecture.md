# IMMER Architecture

IMMER verbindet getrennte Zuständigkeiten über kleine, typisierte Verträge.
Der Produktionspfad ist absichtlich enger als die Forschungslandschaft.

## 1. Laufzeitvertrag

```text
Request(capability, payload, metadata)
  → ComponentRegistry (genau ein Besitzer je Capability)
  → Component.handle(Request)
  → Result(status, output, reason, evidence)
```

Gültige Stati sind explizit. `ABSTAINED` und `UNAVAILABLE` sind normale
Ergebnisse; ein Fallback darf daraus keine scheinbar bestätigte Tatsache oder
Aktion erfinden.

## 2. Composition Root

`CompositionRoot` ist die einzige Stelle, an der Policy-Komponenten verdrahtet
werden:

```text
ExactCascade (einziger Registry-Besitzer von exact_math)
├── S3Arithmetic
│   ├── Frozen A1
│   ├── FrozenA1CrsaRouter
│   └── OrganBank mit vier Organen
└── FertigSolver

FertigGrounded (separate capability grounded_chat)
LearningStream (separates, mutables Lebensobjekt)
```

Der Lebensstrom wird nie in S3 injiziert. Das verhindert, dass Online-Lernen
den Frozen-Host oder einen exakten Readout verändert.

## 3. Exact-Pfad

```text
Text
  → enge, explizite Arithmetikgrammatik
  → normalisierte A1-Tokenfolge
  → kontextuelle Layer-0-Scan-Zustände
  → fester CRSA-Rollenmix
  → persistierter Ridge-Head: arithmetic oder text
  → ausgewähltes kaltes Organ
  → gelernter Carrier / Strukturverifier
  → exakte Algebra oder Emission
  → FERTIG-Verifikation/Fallback
  → Antwort oder bewachte Abstinenz
```

S3 gewinnt eine Meinungsverschiedenheit nur mit literalem
`crystal_verified=True`. Liefert nur FERTIG eine Antwort, prüft die Kaskade die
bekannte Familie additiver/subtraktiver Mengenänderungen deterministisch.

## 4. OrganBank und R17

Ein Organmanifest trägt Name, Capability, algebraische Gruppe, Pfad, SHA-256
und Kristallmetrik. Vor `torch.load(..., weights_only=True)` werden alle äußeren
Digests geprüft; im Bundle wird zusätzlich der interne State-Digest geprüft.

Die vier deployten Organe sind:

- `arith-dual` für den verifizierten kleinen Add/Sub-Carrier;
- `mul-log` mit kristallisiertem Log-Readout;
- `z3-circle` ausschließlich als Addition in `Z₃` (`z3sum`);
- `decimal-crystal` mit `h ← 10h + v` und exakter Digit-Word-Emission.

R17 ist der Übergang von gemessener Invariante zu exakter Struktur. Ein
Cross-Model-Least-Squares-Shortcut gehört nicht dazu und ist negativ belegt.

## 5. CRSA

Der Kern stellt exakt kausale Operatoren bereit. Im Online-Router ist der
Rollenmix fest:

```text
2 × Local
1 × Balanced
1 × Free (bit-exakte kausale Softmax)
```

Der gemessene Featurevektor ist `[raw_last | role_complete_last]`. Ein
persistierter Ridge-Head entscheidet, ob die bereits grammatisch kanonisierte
Anfrage den Organpfad betreten darf. Zukunftsmasse bleibt exakt null.

Die aktuelle Messung belegt das Kontextsignal gegen Roh-A1 und
Label-Placebos. Eine kausale-Softmax-Ablation erreicht jedoch denselben Score;
darum gibt es keinen CRSA-spezifischen Überlegenheitsclaim.

## 6. FERTIG

FERTIG besitzt Grounding, Bindings, semantische Struktur, Pläne, Skills und
Verifikation. Der geerdete Adapter ist lazy und verlangt einen expliziten
State-Ordner. Ein `.causal`-Graph ist optional und wird nur über einen
expliziten Pfad geöffnet.

Desktop, Aufnahme und mutierende Skill-Komposition sind Sicherheitsgrenzen:
Ohne injizierten Backend/Recorder beziehungsweise `allow_mutations=True`
liefert der Adapter `needs_input`.

## 7. Lebenssubstrat

```text
EventBus        priorisierte Kanäle
LifeDaemon      Strom, Dienste, Rack und Turn-Zähler
LifeStatePort   atomare JSON-Snapshots über fsync + os.replace
OrganRack       Mount nur nach Digestprüfung
```

`O1StateStream` trägt pro Layer einen konstant großen Z-Zustand. Der
`LearningStream` misst Loss zunächst ohne Graph, rekonstruiert nur
überraschende Chunks aus demselben detached Eingangszustand, clippt Gradienten
und persistiert Modell, Optimizer, Zustände und Replay-Puffer.

## 8. Gedächtnis und Mund

`intent.py` trennt TEACH, RECALL, MATH, STATUS und CHAT. `SpanStore` und
`Library` halten gelehrte oder geerntete Karten außerhalb der Gewichte samt
Herkunft. Für Chat kann ein lokales Qwen oder ein externer Donor konfiguriert
werden; beide liegen hinter den exakten und geerdeten Pfaden.

Der Rat ist eine optionale BO3-Deliberation. Seine historische Qualität ist
kein aktueller Core-Akzeptanzwert.

## 9. WorldStream

Der Streamer ist ein separater TensorSource-Vertrag:

```text
lokales Verzeichnis oder HF-Revision
  → Header-only Inventar
  → expliziter 2D-Tensor + Zeilenrange
  → Preflight gegen hartes Bytebudget
  → atomarer, SHA-geprüfter Resume-Cache
  → NumPy oder lazy Torch-Bridge
```

Er lädt kein Donormodell. Weil der statische Value-Sketch negativ war, ist der
Streamer nicht automatisch mit dem Antwortpfad verbunden.

## 10. Suite

`Metrics` schreibt `status.json` atomar und hängt `metrics.jsonl` an. Das
Dashboard bietet `/status` und eine lokale Übersicht. CLI, Dashboard und
Daemon verwenden denselben Capability-Vertrag.

## 11. Externe Ebenen

FLCA und QAD bleiben kanonische externe Stränge. Sie werden in der
Komponentenkarte als solche benannt, aber nicht als heute laufender IMMER-Core
ausgegeben. Große Gewichte bleiben ebenfalls extern; IMMER verwaltet nur
Manifeste, Digests und expliziten Import.
