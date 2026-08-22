"""router_v2_stage2.py — Werte-Extraktion auf den Gewinner-Layern.

Kette pro Frage:
  h_q (embed-Mittel, gestreamt) -> Aktivierungen a über gesampelte gate-Zeilen
  -> Top-m Neuronen -> down_proj-Zeilenblöcke (Strategie c, 4×64 Rows)
  -> Value-Sketch r = Σ a_j · down_proj[coords, j]  (residualer Raum!)
  -> score(choice) = cos(r_sketch, choice_embed_sketch auf denselben Coords)

Placebo: aktivierungspermutierte Neuronen-Auswahl je Frage.
Alles gegen Bytes/Frage geloggt. Chance = 25 %.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "immer" / "knowledge"))
sys.path.insert(0, str(ROOT / "vendor" / "mitglm"))

import numpy as np  # noqa: E402

REPO = "Qwen/Qwen3.8-27B"
STOPS = {"the", "and", "for", "with", "that", "this", "from", "was", "are",
         "have", "has", "not", "but", "which", "what", "when", "where", "how"}
WINNERS = [18, 36, 45]   # aus Stufe 1 (Δ real-vs-placebo)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="high_school_geography")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--top-neurons", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--rows-per-block", type=int, default=64)
    ap.add_argument("--budget-mb", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import pandas as pd  # type: ignore

    frame = pd.read_parquet(ROOT / "evals" / f"mmlu_{args.subject}_test.parquet")
    rows_data = list(frame.head(args.limit).to_dict("records"))

    from streamer import Streamer  # noqa: E402

    s = Streamer(REPO, budget_mb=args.budget_mb)
    inv = s.inventory()
    all_ts = s.tensors()
    emb = next(t for t in all_ts if t["name"].endswith("embed_tokens.weight"))
    hidden = int(emb["shape"][1])
    row_bytes = hidden * 2

    tok_raw = s.reader.fetch_file("tokenizer.json")
    import tempfile

    from transformers import PreTrainedTokenizerFast

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        fh.write(tok_raw)
        s_tok = PreTrainedTokenizerFast(tokenizer_file=fh.name)

    def clean(text: str):
        out = []
        for i, tk in zip(s_tok.encode(text.lower(), add_special_tokens=False),
                         s_tok.convert_ids_to_tokens(
                             s_tok.encode(text.lower(), add_special_tokens=False))):
            w = re.sub(r"[^a-z]", "", (tk or "").replace("Ġ", "").replace("▁", ""))
            if len(w) >= 3 and w not in STOPS:
                out.append((i, w))
        return out

    rng = np.random.default_rng(args.seed)
    row_cache: dict[int, np.ndarray] = {}

    def fetch_embed_rows(missing):
        if not missing:
            return
        from concurrent.futures import ThreadPoolExecutor

        def grab(i):
            off = emb["data_start"] + emb["offset_in_shard"][0] + i * row_bytes
            return i, s.reader.get_range(emb["shard"], off, off + row_bytes - 1)

        with ThreadPoolExecutor(max_workers=12) as pool:
            got = dict(pool.map(grab, missing))
        for i, rawb in got.items():
            u16 = np.frombuffer(rawb, dtype=np.uint8).view(np.uint16)
            row_cache[i] = (u16.astype(np.uint32) << 16).view(np.float32)

    def evec(text: str, coords=None) -> np.ndarray:
        keep = clean(text)
        missing = sorted({i for i, _ in keep if i not in row_cache})
        fetch_embed_rows(missing)
        v = np.zeros(hidden if coords is None else len(coords), dtype=np.float32)
        for i, _w in keep:
            v += row_cache[i][coords] if coords is not None else row_cache[i]
        return v / (max(len(keep), 1) * np.sqrt(len(v)))

    import hf_organ_reader as hor  # noqa: E402
    import casi_tensor_map as ctm  # noqa: E402

    def layer_parts(L: int):
        gate = next(t for t in all_ts
                    if f".layers.{L}.mlp.gate_proj.weight" in t["name"])
        down = next(t for t in all_ts
                    if f".layers.{L}.mlp.down_proj.weight" in t["name"])
        return gate, down

    n_inter = None
    layer_state = {}
    for L in WINNERS:
        gate, down = layer_parts(L)
        n_inter = int(gate["shape"][0])
        idxs = np.sort(rng.choice(n_inter, size=256, replace=False))
        graw = hor.fetch_uniform_rows(
            s.reader, gate["shard"], gate["data_start"],
            gate["offset_in_shard"][0], int(gate["shape"][1]), 2,
            [int(i) for i in idxs],
        )
        gkeys = ctm.bf16_rows_to_f32(graw.view(np.uint16), (len(idxs), int(gate["shape"][1])))
        gkeys /= (np.linalg.norm(gkeys, axis=1, keepdims=True) + 1e-9)
        # down_proj-Zeilenblöcke (hidden-dim Zeilen!): Strategie c
        d_rows_total = args.blocks * args.rows_per_block
        starts = np.linspace(0, int(down["shape"][0]) - d_rows_total,
                             num=args.blocks).astype(int)
        blocks = []
        d_row_bytes = int(down["shape"][1]) * 2
        for st in starts:
            abs_off = down["data_start"] + down["offset_in_shard"][0] + st * d_row_bytes
            span = args.rows_per_block * d_row_bytes
            data = s.reader.get_range(down["shard"], abs_off, abs_off + span - 1)
            mat = ctm.bf16_rows_to_f32(
                np.frombuffer(data, dtype=np.uint8).view(np.uint16),
                (args.rows_per_block, int(down["shape"][1])))
            blocks.append((st, mat))
        coords = np.concatenate([np.arange(st, st + args.rows_per_block)
                                 for st, _ in blocks])
        D = np.concatenate([mat for _, mat in blocks], axis=0)  # [256, n_inter]
        layer_state[L] = {"idxs": idxs, "gkeys": gkeys, "D": D, "coords": coords}
        print(f"L{L}: gate-keys {gkeys.shape}, down-fenster {len(coords)} Hidden-Coords",
              flush=True)

    correct = total = 0
    acc_plac = []
    q_bytes_list = []
    t0 = time.time()
    for item in rows_data:
        frage, choices, gold = item["question"], item["choices"], int(item["answer"])
        before = s.budget.body
        try:
            hq = evec(frage)
            cvecs = [evec(c[:64]) for c in choices]
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP {type(exc).__name__}")
            continue
        scores_real, scores_plac = [], []
        for L in WINNERS:
            st = layer_state[L]
            acts = np.maximum(st["gkeys"] @ hq, 0.0)
            top = np.argsort(acts)[-args.top_neurons:]
            plac_top = np.argsort(rng.permutation(acts))[-args.top_neurons:]
            r_real = np.zeros(len(st["coords"]), dtype=np.float32)
            r_plac = np.zeros_like(r_real)
            for j in top:
                r_real += acts[j] * st["D"][:, int(st["idxs"][j])]
            for j in plac_top:
                r_plac += acts[j] * st["D"][:, int(st["idxs"][j])]
            r_real /= (np.linalg.norm(r_real) + 1e-9)
            r_plac /= (np.linalg.norm(r_plac) + 1e-9)
            sc_r, sc_p = [], []
            for cv_full in cvecs:
                cs = cv_full[st["coords"]]
                cs /= (np.linalg.norm(cs) + 1e-9)
                sc_r.append(float(np.dot(r_real, cs)))
                sc_p.append(float(np.dot(r_plac, cs)))
            scores_real.append(sc_r)
            scores_plac.append(sc_p)
        sr = np.sum(scores_real, axis=0)
        sp = np.sum(scores_plac, axis=0)
        pred_real = int(np.argmax(sr))
        pred_plac = int(np.argmax(sp))
        total += 1
        correct += int(pred_real == gold)
        acc_plac.append(int(pred_plac == gold))
        used = s.budget.body - before
        q_bytes_list.append(used)

    report = {
        "schema": "router.v2/stufe2",
        "repo": REPO, "n": total,
        "accuracy_stufe2": round(correct / max(total, 1), 4),
        "accuracy_placebo": round(float(np.mean(acc_plac)), 4),
        "bytes_pro_frage_kb": round(float(np.mean(q_bytes_list)) / 1024, 1),
        "anteil_modell_pct": round(100 * float(np.mean(q_bytes_list))
                                   / inv["model_payload_bytes"], 4),
        "layer": WINNERS, "top_neuronen": args.top_neurons,
        "sekunden": round(time.time() - t0, 1),
    }
    out = ROOT / "results" / "router_v2_stage2.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
