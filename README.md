# IMMER

IMMER ist die lokale Integrationsruntime für ein kleines, fortlaufendes
System: Identität im O(1)-Strom, Wissen in einer externen Bibliothek, Können
in SHA-adressierten Organen, Grounding durch FERTIG und kausales Routing durch
die eigene CRSA-Attention.

Der derzeit belastbare PoC ist kein allgemeines Chatmodell. Er ist ein
funktionierender, abstinenzfähiger Runtime-Pfad:

```text
Anfrage
  → explizite Grammatik
  → Frozen A1
  → CRSA: 2 Local + 1 Balanced + 1 bit-exakter Free Head
  → gemessener Arithmetic/Text-Gate
  → eines von vier kalten Organen
  → exakter struktureller Readout
  → FERTIG als Verifier/Fallback
  → Antwort oder Abstinenz
```

Der lernende Lebensstrom ist davon absichtlich getrennt. Kein Online-Update
kann den eingefrorenen Exact-Pfad verändern.

## Lokal starten

Python 3.11+ wird unterstützt. Die Gewichte liegen absichtlich nicht in Git;
der Checkout enthält ihre Pfade und SHA-256-Digests.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[neural]'

# Bereits vorhandene Originalartefakte read-only als Quelle benutzen.
# IMMER prüft alle fünf SHA-256-Werte und kopiert atomar in diesen Checkout.
python -m immer artifacts import /pfad/zum/o1-state-checkout
python -m immer artifacts verify

# Der vollständige lokale Akzeptanzlauf.
python -m immer doctor --deep
python -m immer eval
python -m immer solve "Was ist drei plus fünf?"
```

Sind die Artefakte in diesem Checkout schon vorhanden, meldet der Import nur
`verified` und kopiert nichts. Ein vorhandenes Ziel mit falschem Digest wird
ohne explizites `--replace` nicht angefasst.

## Leben, FERTIG und Bibliothek

```bash
python -m immer serve
# /state   /sleep   /say <text>   /quit
# merke: <fakt>
# was weißt du über <thema>?
```

`serve` registriert genau einen Besitzer für `exact_math`: die bewachte
S3→FERTIG-Kaskade. FERTIG stellt zusätzlich Hilfe, Skillauflistung und einen
optionalen `.causal`-Graph bereit:

```bash
export IMMER_FERTIG_GRAPH=/pfad/zur/welt.causal
python -m immer serve --no-dashboard
```

Desktop-Ausführung, Aufnahme und Store-Mutationen sind ohne explizit
injizierte Backends gesperrt. FERTIG erfindet an diesen Grenzen keine Aktion.
Ein lokaler Qwen-Mund ist nur ein optionaler Chat-Fallback:

```bash
pip install -e '.[mouth]'
python -m immer serve --local-brain
```

## Weltgewichte streamen, ohne das Modell zu laden

Der WorldStream liest nur Safetensors-Header und explizite Zeilenbereiche. Ein
hartes Bytebudget wird vor jedem Transfer geprüft; Resume-Caches sind atomar
und SHA-verifiziert.

```bash
# Komplett offline
python -m immer stream /pfad/zum/safetensors-ordner --local --budget-mb 20
python -m immer stream /pfad/zum/safetensors-ordner --local \
  --tensor model.layers.0.mlp.gate_proj.weight --start-row 0 --rows 8

# Remote: für reproduzierbare Läufe immer einen Commit-Digest pinnen
python -m immer stream ORG/MODEL --revision <commit-sha> --budget-mb 200
```

Dieser Streamer ist ein Instrument, nicht heimlich der Antwortpfad. Der frühere
Shortcut „statisches Embedding-Mittel → Gate-Aktivierung → Value-Sketch“ war
negativ und wird nicht wiederbelebt.

## Gemessenes Urteil

| Messung | Ergebnis | Zulässige Aussage |
|---|---:|---|
| Frozen SHIP-v6 | 152/152 Antworten, 152/152 Organrouten | Vier kalte Organe laufen ohne Runtime-Training |
| CRSA-Kontextrouter | 182/182 Arithmetic+Text | Kontextsignal ist positiv und online einsetzbar |
| Roh-A1-Ablation | balanced 0,983333 | CRSA-Residual verbessert diesen kleinen Split |
| 32 Label-Placebos | Mittel 0,510328; Maximum 0,748026 | Der persistierte Head liegt klar über seinem Nullmodell |
| kausale Softmax-Ablation | balanced 1,000 | Kein CRSA-spezifischer Vorteil gezeigt |
| statischer Value-Sketch | 24 % vs Placebo 32 % | Negativ; falsche Eingabeverteilung, nicht deployt |

Die CRSA-Rollen sind fest: zwei Local-, ein Balanced- und ein unveränderter
Free-Head; Steigung `0.8`, Diagonal-Debit `3`. Zukünftige Masse ist exakt null,
der Free-Head ist bitgleich zur kausalen Softmax. Der Router-Report wird mit
`PYTHONPATH=src python scripts/crsa_route_eval.py` byte-identisch reproduziert.

## Module und Grenzen

| Bereich | Pfad | Status |
|---|---|---|
| Exact-Cascade | `cognition/exact_cascade.py` | integriert; S3 zuerst, FERTIG verifier/fallback, sichere Abstinenz |
| Frozen Host + Organe | `capabilities/s3_runtime.py` | integriert; fünf externe Blobs vollständig digest-geprüft |
| Eigene Attention | `attention/crsa/`, `attention/router.py` | integriert und im Online-Gate aktiv |
| FERTIG | `cognition/fertig/` | vendored; Grounding/Bindings/Skills/Verifier, Aktions-Gates geschlossen |
| Lebensstrom | `runtimes/o1_state/` | persistent, surprise-gated, getrennt vom Frozen Host |
| WorldStream | `knowledge/streamer.py` | lokaler/HF Range-Zugriff, kein Donor-Modell-Load |
| Daemon/Suite | `substrate/`, `suite.py` | atomare Zustände, Services, Metriken, Dashboard |
| FLCA/QAD | externe kanonische Projekte | hier beschrieben, nicht als laufender IMMER-Kern ausgegeben |

## Verifikation

```bash
PYTHONPATH=src python -W error::ResourceWarning -m unittest discover -s tests
PYTHONPATH=src python -m compileall -q src scripts
git diff --check
```

Die ausführliche Beweislage und die bekannten Negativergebnisse stehen in
[`docs/RUNTIME-VERDICT.md`](docs/RUNTIME-VERDICT.md). Hugging Face ist erst der
letzte Export nach lokalem Akzeptanzlauf und geklärter Lizenzkette; es ist
nicht die Entwicklungsachse dieses Repositories.

## Offline-HF-Bundle — erst nach dem lokalen Lauf

Wenn `doctor --deep` und `eval` grün sind, erzeugt der letzte technische Schritt
einen eigenständigen, HF-geformten Ordner und verifiziert ihn vor dem atomaren
Publish nochmals isoliert:

```bash
pip install -e '.[neural,export]'
python -m immer export-hf dist/immer-ship-v6
cd dist/immer-ship-v6
python -I -B verify.py
python -I -B solve.py "three plus five is"
```

Das Bundle enthält nur den extrahierten A1-State als `model.safetensors`, vier
bytegleiche Organe, den CRSA-Router, die S3→FERTIG-Kaskade und eine bereinigte
Runtime-Kopie. `checksums.json` bindet die exakte Dateimenge. Ein vorhandenes
Ziel wird ohne `--replace` nicht berührt; mit `--replace` bleibt es als
wiederherstellbares `.backup` erhalten.

Das ist ausdrücklich kein Upload: Model Card und Konfiguration tragen
`license: other`, `license_status: NOT_CLEARED` und
`public_upload_allowed: false`. Es gibt keinen Hub-, Login- oder Upload-Code.
Details: [`docs/HF-SHIP.md`](docs/HF-SHIP.md).

## Autor

David Tom Foss · 2026
