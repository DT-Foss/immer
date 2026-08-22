# Source Inventory — Code, Artefakte und externe Herkunft

Stand: 2026-08-22. Diese Landkarte unterscheidet den laufenden lokalen
v0.7-Core von externen Originalen und historischen Messquellen. Originale
außerhalb dieses Repositories sind read-only; IMMER importiert nur explizit
gewählte, digest-geprüfte Artefakte.

## Produktions-Core in diesem Repository

| Strang | Kanonischer Laufzeitort | Stand |
|---|---|---|
| Composition Root | `src/immer/composition.py` | ein `exact_math`-Besitzer; Frozen- und Lebenspfad getrennt |
| Exact-Kaskade | `src/immer/cognition/exact_cascade.py` | S3 zuerst, FERTIG Verifier/Fallback, bewachte Abstinenz |
| Frozen A1 | `src/immer/runtimes/o1_state/model.py` + `adapter.py` | eigener, state-dict-kompatibler Produktionskern; keine Vendor-Imports |
| SHIP-v6 | `src/immer/capabilities/s3_runtime.py` + `manifests/s3_ship_v6.json` | vier Organe, 152/152 Antworten und Routen, kein Runtime-Training |
| OrganBank | `src/immer/capabilities/organbank/bank.py` | SHA-256 außen plus interner State-Digest vor Mount |
| Eigene CRSA-Attention | `src/immer/attention/crsa/operators.py` | exakt kausale Operatoren |
| Online-Router | `src/immer/attention/router.py` + `manifests/crsa_router_v1.json` | Frozen A1; fest `2 Local + 1 Balanced + 1 Free`, slope `0.8`, Debit `3` |
| FERTIG exakt | `src/immer/cognition/fertig/adapter.py` | vendorter Solver oder expliziter read-only Override |
| FERTIG geerdet | `src/immer/cognition/fertig/grounded.py` | Graph, Bindings, Pläne, Skills und Verifier; Aktions-Gates geschlossen |
| WorldStream | `src/immer/knowledge/streamer.py` + `_hf_source.py` | lokale/HF-Ranges, hartes Budget, atomarer SHA-Cache; kein Modell-Loader |
| Lebensstrom | `src/immer/runtimes/o1_state/plasticity.py` | persistent, surprise-gated, getrennt vom Frozen Host |

Der WorldStream ist integriert, aber absichtlich keine verdeckte
Antwortkomponente. Der einzige bislang getestete Value-Sketch war mit 24 %
gegen 32 % Placebo negativ.

## Externe Originale und Forschungsquellen

| Strang | Kanonische externe Quelle | Redundant/historisch | Einbindung heute |
|---|---|---|---|
| o1-state | `~/Documents/Forschung/O1_juli` @ `bb5e443` | `fabel_video`, `O1`, `O1-O`, `o1-state_kram_August` | Produktionsport liegt in `src/immer/runtimes/o1_state/`; Frozen Checkpoint wird nur per SHA importiert |
| FERTIG | `~/Documents/Forschung/LanguageModel/FERTIG` @ `edea94a` | `~/Downloads/Forschung/FERTIG`, `FERTIG copy`, Archive | kuratierter Snapshot ist vendort; `IMMER_FERTIG_ROOT` bleibt expliziter Override |
| CRSA-Forschung | `~/self-verification_fable/neue attention/` | frühere 225-Zeilen-Extrakte | eigener Operator plus gemessener Router liegen jetzt im Core |
| Organ-/Verifier-Labor | `~/self-verification_fable/mirkoNN-analysis` | `~/Documents/Forschung/mirkoNN` | Formeln/Evidence in `research/`; keine Runtime-Imports |
| GSSM | `~/Documents/Forschung/gssm-public` @ `d7dbfec` | `GSSM_Research*` | Forschungsbasis, keine heutige Runtime-Abhängigkeit |
| FLCA | `~/Downloads/Forschung/Alte AI Projekte/FLCA` @ `fdfde80` | — | externer kanonischer Strang, nicht als Core vorgetäuscht |
| QAD/Liquid | `~/Downloads/Forschung/Liquid-QAD` | Modelle bleiben extern | externer Compiler-/Deployment-Strang |
| Verifier | `~/self-verification_fable/llm-as-a-verifier` | — | historischer BO3-Messanker, kein v0.7-Core-Golden |
| `.causal` | `~/Desktop/dotcausal`, `~/Desktop/ANALYSIEREN/pipeline_dza` | GW150914-Sonderfall | FERTIG-Weltgraph nur über explizites `IMMER_FERTIG_GRAPH` |

## Frozen Artefakte

Große Gewichte liegen nicht im Git-Commit. `manifests/s3_ship_v6.json`
adressiert exakt fünf Blobs:

| Artefakt | SHA-256 |
|---|---|
| Frozen A1 Host | `84f7ac90375668067f348421c581ca60cf3f27dbdb447ae13248dbcc85ea251c` |
| `arith-dual` | `50c76fe0fd9bfb3d18aa76a143c55c35a940d406910ef1425bba6744442fb2fe` |
| `mul-log` | `a3afac05dbcf9bdb7ae4bef9883af5cef90da5a03dcade2fed3ae18b9ae0a41a` |
| `z3-circle` | `a0a94a665b41661ac16d274b096001655c18c2916ee1d989df92d9d21c737053` |
| `decimal-crystal` | `ad64764e6298650aad5d273df4a17ba155825b4eeec379ef10bbda2a263e5a19` |

Historische Evidence-/Cache-Orte sind `/Volumes/LEXAR/dfc-evidence-smoke`
und `/Volumes/SDKARTE/HIER/hf-cache`. Sie sind keine stillen
Produktionsabhängigkeiten. Der lokale Bootstrap kopiert aus einem expliziten
Quellordner atomar in den IMMER-Artefakt-Root und akzeptiert nur die
Manifest-Digests.

## Server-Fleet: Operationsquelle, keine Runtime-Abhängigkeit

Details stehen in `docs/fleet/O1_STATE_OPERATIONS.md`; Zugangsdaten bleiben in
`~/Desktop/SERVER` und gehören nicht ins Repository.

| Alias | Rolle |
|---|---|
| `intel` / `ki` | historischer Lifetime-Lauf; nicht für Experimente anfassen |
| `core` / `kc` | freie Experimentmaschine |
| `beast` / `kb` | historischer 27B-Donor und schwere Experimente |
| `aero` / `ka` | Storage, keine O1-Rechenrolle |

`torch threads=1` ist eine Regel für reproduzierbare Benchmark-Harnesses auf
der Fleet. Die Produktions-`LearningStream`-Runtime verändert die
prozessglobale Torch-Threadzahl nicht.

## Mess- und Claim-Grenzen

- SHIP-v6 ist lokal reproduzierbar: 152/152 Antworten, 152/152 Routen.
- CRSA zeigt einen positiven Kontextrouter, aber keinen Vorteil gegen
  kausale Softmax: beide erreichen Balanced Accuracy 1,0.
- Stage 1 lokalisiert Donor-Themenstruktur mid-depth; Stage 2 mit statischem
  Embedding-Mittel ist unter Placebo und bleibt außerhalb der Runtime.
- Cross-Model Least Squares ist negativ. R17 bedeutet
  trainieren → Invariante messen → Fit/R² verifizieren → exakte Struktur.
- Hugging Face ist Verpackung ganz am Ende. Ein öffentlicher Upload bleibt
  bis zum lokalen Golden und zur Klärung der Lizenzkette gesperrt.

## Architektur in einem Bild

```text
IDENTITÄT = persistenter O(1)-Lebensstrom
DENKEN    = Frozen A1 + eigene, feste CRSA-Attention
KÖNNEN   = vier kalte, kristallisierte Organe
ERDUNG    = FERTIG: Graph, Bindings, Pläne, Skills, Verifier
WISSEN    = Bibliothek + begrenzter WorldStream als separates Instrument
KÖRPER   = lokale Runtime; Fleet/QAD nur über explizite Verträge
```
