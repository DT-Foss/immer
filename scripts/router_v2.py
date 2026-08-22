"""router_v2.py — Stufe 1: Aktivierungs-Profiler mit Placebo-Kontrolle.

Frage (F4/F3 aus docs/ROUTER-FORMELN.md): Tragen gate_proj-Zeilen des
Qwen3.8-27B Fragen-Struktur? Messung: Aktivierungsprofile a = ReLU(W·h_q)
pro Layer; Separation S = intra-cluster minus inter-cluster Kosinus der
Profile über zwei lexikalische Frage-Cluster. Placebo: dieselbe Rechnung
auf seed-shuffled Zeilen — S_placebo muss gegen Null gehen, sonst misst
das Instrument Müll.

    PYTHONPATH=src python3 scripts/router_v2.py --limit 40
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="high_school_geography")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--neurons", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--budget-mb", type=float, default=400.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import pandas as pd  # type: ignore

    frame = pd.read_parquet(ROOT / "evals" / f"mmlu_{args.subject}_test.parquet")
    rows = list(frame.head(args.limit).to_dict("records"))

    from streamer import Streamer  # noqa: E402

    s = Streamer(REPO, budget_mb=args.budget_mb)
    inv = s.inventory()
    all_ts = s.tensors()
    layer_ids = sorted({t["layer"] for t in all_ts
                        if "/mlp/gate_proj.weight" in t["name"]
                        or ".mlp.gate_proj.weight" in t["name"]})
    picks = [layer_ids[int(round(i * (len(layer_ids) - 1) / (args.layers - 1)))]
             for i in range(args.layers)]
    emb = next(t for t in all_ts if t["name"].endswith("embed_tokens.weight"))

    def tok_clean(text: str):
        ids = s_tok.encode(text.lower(), add_special_tokens=False)
        toks = s_tok.convert_ids_to_tokens(ids)
        keep = []
        for i, tk in zip(ids, toks):
            w = re.sub(r"[^a-z]", "", (tk or "").replace("Ġ", "").replace("▁", ""))
            if len(w) >= 3 and w not in STOPS:
                keep.append((i, w))
        return keep

    # Tokenizer des Donors, gestreamt
    tok_raw = s.reader.fetch_file("tokenizer.json")
    import tempfile

    from transformers import PreTrainedTokenizerFast

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        fh.write(tok_raw)
        s_tok = PreTrainedTokenizerFast(tokenizer_file=fh.name)
    print(f"tokenizer gestreamt ({len(tok_raw) / 1048576:.1f} MB)", flush=True)

    row_cache: dict[int, np.ndarray] = {}
    hidden = int(emb["shape"][1])
    row_bytes = hidden * 2

    def h_vec(text: str) -> np.ndarray:
        keep = tok_clean(text)
        missing = sorted({i for i, _ in keep if i not in row_cache})
        if missing:
            def grab(i: int):
                off = emb["data_start"] + emb["offset_in_shard"][0] + i * row_bytes
                return i, s.reader.get_range(emb["shard"], off, off + row_bytes - 1)
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=12) as pool:
                got = dict(pool.map(grab, missing))
            mats = [np.frombuffer(got[i], dtype=np.uint8).view(np.uint16) for i in missing]
            for j, i in enumerate(missing):
                row_cache[i] = np.frombuffer(
                    mats[j].tobytes(), dtype=np.uint16).astype(np.float32).view(np.float32) \
                    if False else _bf16(mats[j])
        v = np.zeros(hidden, dtype=np.float32)
        for i, _w in keep:
            v += row_cache[i]
        v /= max(len(keep), 1)
        return v / (np.linalg.norm(v) + 1e-9)

    def _bf16(u16: np.ndarray) -> np.ndarray:
        return (u16.astype(np.uint32) << 16).view(np.float32)

    # Zwei Cluster: physisch vs. menschlich-geografisch (Lexik, lokal)
    PHYS = {"climate", "temperature", "rainfall", "erosion", "plate", "tectonic",
            "desert", "glacier", "river", "mountain", "ocean", "weather", "soil"}
    HUMAN = {"city", "population", "migration", "trade", "country", "border",
             "urban", "economy", "culture", "language", "religion", "state"}

    def cluster_of(text: str) -> int:
        words = {w for _, w in tok_clean(text)}
        p, h = len(words & PHYS), len(words & HUMAN)
        if p == h:
            return -1
        return 0 if p > h else 1

    labels = [(cluster_of(it["question"]), it["question"]) for it in rows]
    labeled = [(c, q) for c, q in labels if c >= 0]
    print(f"Fragen: {len(rows)} | gelabelt: {len(labeled)}", flush=True)

    hs = {q: h_vec(q) for _, q in labeled}

    rng = np.random.default_rng(args.seed)
    report = {"schema": "router.v2/stufe1", "repo": REPO, "seed": args.seed,
              "neurons": args.neurons, "layers": {}, "bytes_total": 0}
    t0 = time.time()

    for L in picks:
        gate_name = next(t["name"] for t in all_ts
                         if f".layers.{L}.mlp.gate_proj.weight" in t["name"])
        meta = next(t for t in all_ts if t["name"] == gate_name)
        n_neurons = int(meta["shape"][0])
        idxs = np.sort(rng.choice(n_neurons, size=min(args.neurons, n_neurons),
                                  replace=False))
        before_b, before_r = s.budget.body, s.budget.requests
        import hf_organ_reader as hor
        import casi_tensor_map as ctm

        raw = hor.fetch_uniform_rows(
            s.reader, meta["shard"], meta["data_start"],
            meta["offset_in_shard"][0], int(meta["shape"][1]), 2,
            [int(i) for i in idxs],
        )
        keys = ctm.bf16_rows_to_f32(raw.view(np.uint16), (len(idxs), int(meta["shape"][1])))
        keys_n = keys / (np.linalg.norm(keys, axis=1, keepdims=True) + 1e-9)

        profiles_real, profiles_shuf, labs = [], [], []
        for c, q in labeled:
            acts = np.maximum(keys_n @ hs[q], 0.0)
            profiles_real.append(acts)
            # Placebo: UNABHAENGIGE Permutation je Frage — gleiche Permutation
            # auf alle Profile waere kosinus-invariant und kein Test.
            profiles_shuf.append(acts[rng.permutation(len(idxs))])
            labs.append(c)
        P = np.stack(profiles_real)
        S = np.stack(profiles_shuf)
        labs_arr = np.array(labs)

        def separation(M):
            m0 = M[labs_arr == 0]
            m1 = M[labs_arr == 1]
            if len(m0) < 2 or len(m1) < 2:
                return None
            intra = np.mean([np.dot(a, b) for grp in (m0, m1)
                             for a, b in zip(grp[:-1], grp[1:])])
            inter = np.mean([np.dot(a, b) for a in m0 for b in m1])
            return float(intra - inter)

        s_real = separation(P)
        s_plac = separation(S)
        used = s.budget.body - before_b
        report["layers"][str(L)] = {
            "separation_real": None if s_real is None else round(s_real, 5),
            "separation_placebo": None if s_plac is None else round(s_plac, 5),
            "signal_over_placebo": None if s_real is None or s_plac is None
            else round(s_real - s_plac, 5),
            "bytes_layer": used, "requests": s.budget.requests - before_r,
        }
        print(f"L{L:02d}: real={s_real} placebo={s_plac} ", end="", flush=True); print(
              f"Δ={report['layers'][str(L)]['signal_over_placebo']} "
              f"({used / 1024:.0f} KB)")

    report["bytes_total"] = s.budget.body
    report["sekunden"] = round(time.time() - t0, 1)
    out = ROOT / "results" / "router_v2_stage1.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "layers"},
                     ensure_ascii=False))
    print("geschrieben:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
