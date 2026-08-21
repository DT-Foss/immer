"""kaskade_graft.py — A2-Prototyp v2: Organ schlaegt vor, API bestaetigt.

Lehre aus kaskade_proto (negativ): freie 0.5B-Generierung liefert kaum
gute Kandidaten (pass@3 0,25) und der Verifier rankt nackte Zahlen-
Strings unzuverlaessig. Hier: wordlen-min — der Graft (0,125 argmax)
schlaegt seine Top-3-Ziffern vor (Lokal, billig), deepseek-v4-flash
(der die Laenge selbst zaehlen kann) waehlt. Messung: lokale Top-3-
Abdeckung (Ceiling) vs. Verifier-Selektion (Kaskade) vs. Zufall.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import redistill as R  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verifier"))
from m3_select import select_best  # noqa: E402


def load_graft(args, host, tok, dev):
    """f_host wie verifier_loop (digkl, erweitertes Set)."""
    data = __import__("numpy").load(args.donor_npz, allow_pickle=True)
    lg = data["logits"]
    if lg.ndim == 2:
        lg = lg[:, None, :]
        mask = __import__("numpy").ones((lg.shape[0], 1), dtype=bool)
    else:
        mask = data["mask"]
    pr, an = list(data["prompts"]), list(data["ans"])
    pairs, rows = [], []
    for i, (p_, a_) in enumerate(zip(pr, an)):
        for p in range(lg.shape[1]):
            if not bool(mask[i, p]):
                break
            pairs.append((p_ + a_[:p],
                          tok.encode(a_[p], add_special_tokens=False)[0]))
            rows.append(lg[i, p])
    dl = torch.tensor(__import__("numpy").stack(rows), dtype=torch.float32,
                      device=dev)
    dig_idx = torch.tensor(
        [tok.encode(c, add_special_tokens=False)[0] for c in "0123456789"],
        device=dev)
    f_host = R.train_f_host(host, tok, pairs, dev, dl, args.steps, 1e-3,
                            R.SEED, hidden=args.hidden, loss_mode="digkl",
                            dig_idx=dig_idx)
    return f_host, dig_idx


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--donor-npz", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--cand", default=None,
                    help="Kandidaten-Strings (A5), z. B. yes,no")
    ap.add_argument("--out", default="kaskade_graft.json")
    args = ap.parse_args()

    t0 = time.time()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(R.SEED)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    host = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", dtype=torch.float32).to(dev).eval()
    for p in host.parameters():
        p.requires_grad_(False)

    cands = None
    if args.cand:
        cands = [c.strip() for c in args.cand.split(",") if c.strip()]
    f_host, dig_idx = load_graft(args, host, tok, dev)
    hook = R.GraftHook(host)
    evals = R.make_tasks(tok, args.n, R.SEED + 3, task="wordlen-min")

    def topk_cands(prompt: str, k: int = 3):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        rm = torch.zeros(1, ids.shape[1], dtype=torch.bool, device=dev)
        rm[0, -1] = True
        hook.rm = rm
        hook.attach(f_host)
        with torch.no_grad():
            out = host(input_ids=ids)
        hook.detach()
        hook.rm = None
        p = F.softmax(out.logits[0, -1][dig_idx], -1)
        idxs = p.topk(min(k, len(p))).indices.tolist()
        return [cands[i] if cands is not None else str(int(i))
                for i in idxs]

    terseness = ("Scoring-Direktive (Qwen-Sharp-Lektion): lead with the "
                 "answer; prefer the lean, correct candidate; never drop "
                 "correctness for brevity, never prefer length.")
    criteria = {
        "output_match": ("Die Antwort muss die korrekte Anzahl Zeichen "
                         "des Wortes in einfachen Anfuehrungszeichen sein."),
        "error_signals": ("Eine falsche Laengenangabe ist ein Fehler."),
        "terseness": terseness,
    }
    res = {"cfg": {}, "meta": {}}
    n_top3 = n_sel = n_rnd = 0
    for i, (prompt, ans) in enumerate(evals):
        truth = tok.decode([ans]).strip()
        top = topk_cands(prompt)
        cands = [str(d) for d in top]
        ok_top3 = int(truth in cands)
        vr = select_best(problem=prompt, candidates=cands, criteria=criteria)
        ok_sel = int(vr.best == truth)
        ok_rnd = int(random.choice(cands) == truth)
        n_top3 += ok_top3
        n_sel += ok_sel
        n_rnd += ok_rnd
        res["cfg"][f"t{i}"] = {"prompt": prompt[:50], "truth": truth,
                               "top3": cands, "top3_ok": ok_top3,
                               "selected": vr.best, "sel_ok": ok_sel,
                               "n_comparisons": vr.n_comparisons}
        print(f"  t{i}: truth={truth} top3={cands} ({ok_top3}) "
              f"sel={vr.best!r} ({ok_sel})", flush=True)

    n = len(evals)
    res["cfg"]["summary"] = {"local_top3": round(n_top3 / n, 3),
                             "cascade_verifier": round(n_sel / n, 3),
                             "random": round(n_rnd / n, 3)}
    res["meta"] = {"runtime_s": round(time.time() - t0, 1), "device": dev,
                   "steps": args.steps, "n": n}
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"\nlocal top3 {n_top3/n:.3f} | Kaskade {n_sel/n:.3f} | "
          f"random {n_rnd/n:.3f} | {time.time()-t0:.0f}s", flush=True)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
