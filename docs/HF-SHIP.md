# HF-SHIP — das erste eigene Model auf Hugging Face

Stand: 2026-08-21. Ziel: **DT-Foss' erstes HF-Model** — ehrlich, gemessen,
klein. Nicht der 27B (nicht unserer), nicht ein Qwen-Klon. Unsere Artefakte:

## Was geshipped wird (in dieser Reihenfolge)

### Release 1 — `DT-Foss/immer-organism-v0` (das Lebewesen)

Der o1-State-Organismus: 1,7M-StreamingNoPELM + Lebenszustand + Organ-Bank.

```text
immer-organism-v0/
├── README.md            ← Model-Card (siehe unten, Pflicht auf HF)
├── config.json          ← d_model/n_layers/seq_len/vocab (aus pos_ckpt)
├── model.safetensors    ← Host-Gewichte aus pos_ckpt.pt konvertiert
├── life_state.json      ← Stream-Position, Z-Spans, Ledger-Digest
├── organs/
│   ├── manifest.json    ← OrganBank-Format (name/capability/sha256) — bereits gebaut
│   ├── organ_arith_dual.pt
│   ├── organ_mul_log.pt
│   └── organ_mod_kreis.pt
└── immer_organism.py    ← Loader (ein File, torch-only, kein immer-Repo nötig)
```

**Warum das als Erstes:** klein (~80 MB), komplett unser, einzigartig
(kein zweites Lebewesen mit Lebensstrom+Organen auf HF), alle Messanker
vorhanden (twostep 1,000 · NLL bitgleich · 152/152).

### Release 2 — `DT-Foss/immer` (die Runtime, pip-installierbar)

`pip install immer` → `immer serve`. Repo existiert; vor Ship: Vendor aufräumen
(O1_juli-Copyright-Header prüfen, LICENSE-Kette sauberstellen), pyproject
finalisieren, CI grün.

### Release 3 — `.causal`-Bibliothek als Dataset

Die gewachsene Bibliothek (Span-Karten mit Herkunfts-Stempeln) als
`DT-Foss/wesen-bibliothek` Dataset — wächst mit jedem Leben, Versionierung
per Digest.

## Was es noch braucht (Checkliste)

1. **HF-Account + Token**: `huggingface.co` → Settings → Access Token (write);
   lokal `pip install huggingface_hub && hf auth login`.
2. **Konvertierung**: `pos_ckpt.pt` (74 MB, enthält Optimizer-Müll) →
   reines state_dict → `safetensors.torch.save_file` (~7 MB schlank).
   Skript: `scripts/export_hf.py` (zu bauen, ~50 Zeilen).
3. **Model-Card Pflichtfelder**: Modellbeschreibung, Training (C4-Stream,
   Surprise-Gate-Rezept, Token-Zahl), Intended Use + Limitierungen (ehrlich:
   1,7M Params — kein Chat-Modell; es ist ein LEBEN mit Organen),
   Metrics (NLL, twostep, 152/152), License (MIT/Apache-2.0 — O1_juli-Lizenz
   erben), Citation (CITATION.cff existiert schon).
4. **Reproducibility-Block**: Seed, Kadenz (batch/chunk/d_model), WT2-Vokabular-
   Snapshot — ohne die ist der Upload tot weight drift.
5. **Golden-Test im Loader**: `python immer_organism.py --verify` führt
   twostep + NLL-Check aus und muss die Card-Zahlen reproduzieren.
6. **Router-Artefakt-Lücke schließen** (Subagent-Befund): Router einmal
   trainieren + persistieren, sonst ist Release 1 halb tot.

## Der Weg (konkret, ~1 Session)

```bash
hf auth whoami                       # 1. Identität
python scripts/export_hf.py          # 2. ckpt → safetensors + config + card
hf upload repo DT-Foss/immer-organism-v0 ./dist/ --repo-type model   # 3.
```

Danach: Model-Card auf HF im Browser polieren, `imm0` als Nickname?
(Nein — Name bleibt Sache von David.)
