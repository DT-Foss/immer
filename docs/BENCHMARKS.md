# BENCHMARKS — Mess-Suite für den StreamingNoPELM-Organismus

Stand: 2026-08-21 · Zielgruppe: HF-Model-Card `DT-Foss/immer-organism-v0` + CI.
Prinzip: jede Zahl reproduzierbar (Seed, threads=1, Gewichte eingefroren während
Eval), jede Zahl landet in `results/benchmark.json` (Schema unten).

## 1. Perplexity / NLL

Byte-Vokabular (257) macht Token-PPL über Corpora unvergleichlich. Primärmaß
ist **bpb** (bits pro UTF-8-Byte = NLL_nats / ln 2); NLL in nats/byte läuft
mit. Word-PPL nur optional, mit angegebenem bytes/token-Faktor.

| Corpus | Rolle | Protokoll |
|---|---|---|
| Wikitext-2 heldout | Projekt-Anker (8,6656 nats) | Single Pass, kein Fenster |
| TinyStories test | offener Zweitsplit, einfach | Single Pass |
| Simple Wiki test | offener Zweitsplit, enzyklopädisch | Single Pass |

Protokoll je Corpus: Lernen aus (`eval_mode`, keine Gradienten), Zustand
einmal auf Null, ganzer Split als ein Strom, Mittel über alle Bytes. Das
O(1)-State-Design macht Fenster-Tricks (Sliding-Window-PPL à la GPT-2-Eval)
unnötig — ein Pass, eine Zahl, exakt kausal. Zu melden: nats/byte, bpb,
Bytes gesamt, Dauer, torch-Version.

## 2. Streamende Metriken (Alleinstellungsmerkmal)

Standardisierte Definitionen, alle aus `LearningStream.metrics()` plus
Sidecar (`adapter.snapshot/restore`):

1. **Loss-over-life-Kurve**: NLL auf fixem Probe-Set (WT2-heldout-Ausschnitt,
   1 MB, eingefroren) bei Lebens-Marken 10³/10⁴/10⁵/10⁶ Tokens; x-Achse
   log(tokens), gelernt wird zwischen Marken weiter. Artefakt:
   `loss_curve.jsonl`.
2. **Post-Sleep-Delta**: ΔNLL(Probe) nach minus vor `sleep()`. Zielband
   |Δ| ≤ 0,01 nats (Konsolidierung ohne Drift); zusätzlich `replayed`,
   `sleeps`. Ein positives ΔNLL (Schlechterwerb) ist ein Fail.
3. **Surprise-Rate**: surprises und updates pro 10⁶ gelebte Tokens
   (Quantil-Gate window=64, q=0,50 im Report mitangeben).
4. **Resume-Exaktheit**: Snapshot → Restore → gleicher Textstrom → Logits
   bitidentisch (`torch.equal`). Metrik: max_abs_diff = 0,0, bool.
5. **Migrations-Verlust = 0**: Export (ckpt → safetensors → Reload,
   andere Maschine) → Gewichtsdigest gleich, WT2-NLL float-exakt gleich,
   Organ-SHA-256 laut Manifest unverändert.
6. **State-Größe**: Sidecar-Bytes bei 10³ vs 10⁶ gelebten Tokens — flach
   (O(1)-Nachweis). Kurve gehört in die Model-Card.

## 3. Fähigkeits-Tests

Organ-Batterie (n=100 frische Aufgaben je Task, argmax, Accuracy):

| Task | Anchor |
|---|---|
| twostep | 1,000 held-out (Organ-Transfer) |
| mul2x2 | gemessen, zu kuratieren als Anchor |
| wordlen | Kaskade 0,875 @ 8 Calls als Referenz |

Je Task zwei Zahlen: unmontiert (nacktes LM) vs montiert (Organ <1 ms
Cold-Load). Differenz = Transfer-Effekt, Kernverkaufspunkt der Organe.

Systemtest GSM8K via FERTIG: 152-Aufgaben-Subset, Route → Solver, Antwort
exakt. Melden: solve_rate, abstain_rate, falsche Antworten (= 0 laut Anker
152/152).

## 4. Vergleichsanker

Jede Kennzahl gegen drei Zeilen derselben Architektur desselben Seeds:

- **A nacktes Model ohne Leben**: Gewichte wie shipped, kein Online-Lernen.
  Isoliert, was das Leben bringt: ΔNLL(WT2) nach X Tokens, ΔTask-Accuracy.
- **B Modell ohne Organe**: A + gelebt, aber OrganBank leer. Isoliert den
  Organ-Beitrag auf der Batterie.
- **C Trigramm-Floor**: `_wikitext_lm` aus FERTIG-bench als Unteranker für
  bpb — zeigt sofort, ob die 1,7M-Parameter lernen.

Card-Tabelle: Zeilen = Metriken, Spalten = Organismus / A / B / C.

## 5. Schema `results/benchmark.json`

```json
{
  "schema": "immer.benchmark/v1",
  "created_utc": "2026-08-21T00:00:00Z",
  "git_sha": "<sha>",
  "model": {"d_model": 64, "n_layers": 2, "params": 1700000, "seed": 42},
  "environment": {"torch": "<version>", "threads": 1},
  "perplexity": [{"corpus": "wt2_heldout", "bytes": 0, "nll_nats_per_byte": 8.6656,
                   "bpb": 12.5, "seconds": 0}],
  "streaming": {
    "loss_curve": "loss_curve.jsonl",
    "post_sleep_delta_nats": 0.0,
    "surprises_per_mtok": 0.0, "updates_per_mtok": 0.0,
    "resume_exact": true, "resume_max_abs_diff": 0.0,
    "migration_loss_zero": true,
    "state_bytes_at_1e3": 0, "state_bytes_at_1e6": 0},
  "capabilities": [{"task": "twostep", "n": 100, "acc_unmounted": 0.0,
                     "acc_mounted": 1.0, "cold_load_ms": 0.9}],
  "system": {"suite": "gsm8k_fertig", "n": 152, "solve_rate": 1.0,
              "abstain_rate": 0.0, "wrong_answers": 0},
  "baselines": {"no_life_bpb": 0.0, "no_organs_twostep_acc": 0.0,
                 "trigram_bpb": 0.0}
}
```

CI-Regeln: JSON validiert gegen das Schema; Golden-Werte (twostep 1,000,
NLL 8,6656 ± 0,001, resume_exact=true, wrong_answers=0) müssen reproduzieren;
Abweichung bricht den Build. Jede Model-Card-Zahl stammt aus genau dieser
Datei.
