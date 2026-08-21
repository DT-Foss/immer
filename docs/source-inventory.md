# Source Inventory — wo was auf der Welt ist

Stand: 2026-08-21. Diese Datei ist DIE Landkarte: kanonische Quellen,
Redundanz, Server-Rollen. Modelle/Gewichte liegen bewusst NICHT im Repo —
nur Pfade + Digests (`manifests/`).

## Mac (lokal)

| Strang | Kanonisch (neueste Version) | Redundant/alt | GitHub |
|---|---|---|---|
| o1-state (Host) | `~/Documents/Forschung/O1_juli` @ `bb5e443` — **181 Dateien uncommitted!** | `fabel_video` (gleicher Stand, 36 dirty), `O1`, `O1-O`, `o1-state_kram_August` (Prediction-Docs/Zips) | `DT-Foss/o1-state` |
| gssm (Basis) | `~/Documents/Forschung/gssm-public` @ `d7dbfec`, sauber | `GSSM_Research*` (4 Kopien, Juni) | `DT-Foss/gssm` |
| FERTIG (Exakt-Solver) | `~/Documents/Forschung/LanguageModel/FERTIG` @ `edea94a` (kuratiertes Runtime-Repo) | `~/Downloads/Forschung/FERTIG` (ident + pycache), `FERTIG copy`, FERTIG-Zips | noch keins → eigenes Repo geplant |
| Labor (mirkoNN-analysis) | `~/self-verification_fable/mirkoNN-analysis` (CHANGELOG 8.x, s3/, verifier/) | `~/Documents/Forschung/mirkoNN` (historischer Ursprung) | — |
| CRSA / neue Attention | `~/self-verification_fable/neue attention/` (operators.py 534 Zeilen = v0.5-Kern) | 225-Zeilen-Extrakt (bis 0.2.0 in src/) | noch keins |
| FLCA | `~/Downloads/Forschung/Alte AI Projekte/FLCA` @ `fdfde80`, sauber | — | `DT-Foss/FLCA` |
| QAD / Liquid | `~/Downloads/Forschung/Liquid-QAD` (1.7G, models/ drin — bleibt draußen) | — | noch keins |
| Verifier | `~/self-verification_fable/llm-as-a-verifier` (TB-2.1: BO3 79,4→86,5 %) | — | noch keins |
| .causal-Ökosystem | `~/Desktop/dotcausal`, `~/Desktop/ANALYSIEREN/pipeline_dza` (PDF→Triplets→.causal, 14-step Foss Gate) | GW150914_MacMini (Sonderfall) | `DT-Foss/dotcausal` |

**Server-Ops-Ordner:** `~/Desktop/SERVER` (Zugänge, Fleet-Doku, Regeln).
**Frozen Weights/Evidence:** `/Volumes/LEXAR/dfc-evidence-smoke`
(causal-store-freeze, Traces), `/Volumes/SDKARTE/HIER/hf-cache` (11G
Grafts/HF-Cache — extern, nur Manifest+SHA).

## Server-Fleet (Hetzner, Details: `docs/fleet/O1_STATE_OPERATIONS.md`)

| Alias | IP | O1-Rolle |
|---|---|---|
| **intel** (`ki`) | 89.167.47.205 | **DER LIFETIME-LAUF — NIE ANFASSEN.** 12,07 Mrd Tokens seit 2026-07-24, ein Prozess, RSS 0,82 GB. davidfoss-Website läuft mit darauf |
| **core** (`kc`) | 89.167.31.243 | Freie Experiment-Maschine (Konpeki entfernt), torch 2.13 cpu, `/root/o1lab/` |
| **beast** (`kb`) | 89.167.35.196 | Stärkste Maschine: Experimente + 27B-GGUF-Donor (Tailnet 100.119.16.99, `/root/o1x_data/qwen38-27b-gguf/`) + Ollama |
| **aero** (`ka`) | 89.167.35.24 | Reiner Storage (`/storage/archive`) — KEINE O1-Nutzung, kein Python-Stack |

Betriebsregeln (aus Messungen): intel nie anfassen · torch threads=1 ·
ein rechenintensiver Job pro Maschine · erst Smoke dann Full · vor Build
in `analysis/PREDICTIONS.md` registrieren · SSH stdin-detached.

## Architektur-Bild (wohin das hier wächst)

```text
IDENTITÄT  = o1-state      Lebensstrom, O(1), swap-feste Person
DENKEN     = Denkmodelle   paar hundert MB Reasoning ohne Weltwissen,
                           mehrere davon, interner Rat (Multi-Agent)
WISSEN     = Bibliothek    .causal-Index, aus Donor geerntet, wächst durchs Leben
KÖNNEN     = OrganBank     kalte Organe, <1 ms montiert
KÖRPER     = QAD/Fleet     Mac / core / beast / Redmi / iPad
```

Jede Verbindung hat Messanker: twostep 1,000 (Organ-Transfer) · BO3 86,5 %
(Verifier) · Kaskade 0,875 @ 8 Calls · Shell>State-Injection 0,969 vs 0,531 ·
Cold-Load 0,9 ms · NLL 8,6656 unverändert.
