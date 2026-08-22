# Hugging-Face-Export — technisch fertig, weiterhin der letzte Schritt

Stand: 2026-08-22. Der lokale Export ist implementiert. Er ist eine
Release-Ausgabe nach dem Runtime-Akzeptanzlauf, keine Entwicklungsachse und
kein Upload-Werkzeug.

## Offline erzeugen

```bash
pip install -e '.[neural,export]'
python -m immer doctor --deep
python -m immer eval

# Baut in einem Nachbar-Staging-Verzeichnis, verifiziert dort isoliert und
# veröffentlicht erst danach atomar.
python -m immer export-hf dist/immer-ship-v6

# Gleichwertiger direkter Entrypoint:
PYTHONPATH=src python scripts/export_hf_poc.py dist/immer-ship-v6
```

Ein vorhandenes Ziel wird nicht angefasst. `--replace` verschiebt es zuerst
nach `immer-ship-v6.backup` (bei Bedarf `.backup.1`, …), sodass die vorherigen
Daten wiederherstellbar bleiben.

## Erzeugter PoC

```text
immer-ship-v6/
├── README.md                       # Model Card: license other / NOT CLEARED
├── config.json                     # Architektur + hartes Upload-Gate
├── model.safetensors               # nur arms.A1.model, ca. 6,9 MB
├── s3_ship_v6.json                 # auf Bundle-Pfade rebasiert
├── crsa_router_v1.json
├── checksums.json                  # exakte Dateiliste + SHA-256
├── requirements.txt
├── verify.py                       # Digests + 152/152 + Router-Digest
├── solve.py                        # S3 → bewachte FERTIG-Kaskade
├── organs/
│   ├── organ_dual_donor.pt         # bytegleich zum Original
│   ├── organ_mul_donor.pt
│   ├── organ_mod_kreis.pt
│   └── organ_dezimal.pt
└── runtime/immer/                  # eigenständige Package-Kopie, ohne pycache
```

Der 71-MB-Forschungscheckpoint wird nicht umetikettiert. Der Exporter lädt ihn
mit `weights_only=True`, prüft zuerst seinen Manifest-SHA, entnimmt ausschließlich
`arms.A1.model` und schreibt einen deterministischen Safetensors-State-Dict.
Optimizer, andere Arme, Stream-Puffer, Pending State und RNG-Zustände bleiben
draußen. Alle vier Organ-Dateien werden unverändert kopiert und erneut gegen
ihre `ArtifactSpec`-Digests geprüft.

## Automatischer Offline-Akzeptanzlauf

Vor dem atomaren Publish startet der Exporter in seinem Staging-Verzeichnis:

```bash
python -I -B verify.py
```

Dabei werden `PYTHONPATH` und sämtliche `IMMER_*`-Overrides entfernt sowie die
HF-/Transformers-Offlineflags gesetzt. Der Verifier:

1. lehnt zusätzliche, fehlende, veränderte oder verlinkte Dateien ab;
2. importiert IMMER ausschließlich aus `runtime/immer`;
3. lädt den A1-only-Safetensors-State streng in den In-Package-Host;
4. prüft 152/152 Antworten und 152/152 Organrouten ohne Training;
5. vergleicht den semantischen Router-Digest
   `561db8fc50ea029288f318eb9f7c206bd5c007757dbdd60843f1a03990efd819`.

Danach kann derselbe Ordner ohne Checkout getestet werden:

```bash
cd dist/immer-ship-v6
python -I -B verify.py
python -I -B solve.py "three plus five is"
```

`solve.py` verifiziert zuerst die Bundle-Digests und verwendet danach wirklich
`S3Arithmetic` plus `ExactCascade(S3, FertigSolver)`. FERTIG bleibt
Verifier/Fallback und die Kaskade behält ihre bewachte Abstinenz.

## Harte Publikationssperre: Lizenz

Im Repository liegt ein `NOTICE.md`, aber keine Root-`LICENSE`. Das ist keine
Lizenzgewährung. Deshalb tragen Model Card und Konfiguration absichtlich:

```yaml
license: other
license_status: NOT_CLEARED
public_upload_allowed: false
```

Weder Modul noch CLI enthalten Hub-, Login- oder Upload-Code. Öffentliche
Veröffentlichung und Redistribution bleiben gesperrt, bis die Rechtekette für
Hostgewichte, alle vier Organe, Runtime-Quellen und FERTIG schriftlich geklärt
ist. Erst dann darf `public_upload_allowed` nach einer bewussten Releaseprüfung
geändert und ein externes Uploadwerkzeug benutzt werden.

## Was die Model Card bewusst nicht behauptet

- kein allgemeines Reasoning- oder Chatmodell;
- 152/152 ist die kanonische SHIP-v6-Arithmetiksuite, nicht GSM8K;
- CRSA schlägt in der aktuellen Routermessung kausale Softmax nicht;
- der statische Value-Sketch war mit 24 % gegen 32 % Placebo negativ und ist
  nicht im Bundle;
- der optionale FERTIG-Weltgraph wird nicht mitgeliefert.
