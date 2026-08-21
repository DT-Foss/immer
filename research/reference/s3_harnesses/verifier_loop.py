"""verifier_loop.py — System-2: Graft als VERIFIER (Beam-Suche ueber Ziffern).

Die Graft-Hooks feuern nur bei Voll-Prompt-Forwards — freie Generierung
ist 0,000 (Wand 8.27). Hier wird die gepfropfte Faehigkeit als Verifier
benutzt: ein Kandidaten-Praefix wird bewertet, indem der Host den
Praefix + Graft forwardet und die Lese-Positions-Ziffernverteilung
(10 Ziffern, 1 Forward) die naechste Ziffer bewertet:

    p(ans) = prod_p p_graft(digit_p | prompt + d_1..d_{p-1})

Beam-Suche (k=3) ueber die 10 Ziffern je Position, L = Laenge der
wahren Antwort (bekannt). Modi: intact / ablated / graft / placebo /
placebo2. Zusaetzlich greedy (k=1).

Falsifikator: verifier_acc(graft) deutlich > placebo UND > intact —
sonst traegt die System-2-Schleife nichts gegen die 8.27-Wand.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

import redistill as R  # make_tasks, train_f_host, GraftHook, SEED


def load_donor_npz(path, tok, dev):
    """Gleiche Logik wie redistill.main (npz-Zweig): erweiterte Paare."""
    data = np.load(path, allow_pickle=True)
    lg = data["logits"]
    if lg.ndim == 2:
        lg = lg[:, None, :]
        mask = np.ones((lg.shape[0], 1), dtype=bool)
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
    dl = torch.tensor(np.stack(rows), dtype=torch.float32, device=dev)
    cands = [str(x) for x in data["cand"]] if "cand" in data else \
        [str(c) for c in "0,1,2,3,4,5,6,7,8,9"]
    dig_idx = torch.tensor(
        [tok.encode(c, add_special_tokens=False)[0] for c in cands],
        device=dev)
    ev_prompts = [str(x) for x in data["prompts_eval"]]
    ev_ans = [str(x) for x in data["ans_eval"]]
    return pairs, dl, dig_idx, ev_prompts, ev_ans


class Verifier:
    """Forward mit optionalem Graft-Hook; digit-probs am Lese-Punkt."""

    def __init__(self, host, tok, dev, dig_idx):
        self.host = host
        self.tok = tok
        self.dev = dev
        self.dig_idx = dig_idx
        self.hook = R.GraftHook(host)

    def digits(self, prompt: str, f, ablate: bool = False) -> torch.Tensor:
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.dev)
        rm = torch.zeros(1, ids.shape[1], dtype=torch.bool, device=self.dev)
        rm[0, -1] = True
        if ablate:
            # Ablation allein (f=None) ODER Ablation + Graft (f)
            self.hook.rm = rm
            self.hook.attach(f)
        with torch.no_grad():
            out = self.host(input_ids=ids)
        if ablate:
            self.hook.detach()
            self.hook.rm = None
        return F.softmax(out.logits[0, -1][self.dig_idx], -1)   # (10,)

    def beam(self, prompt: str, L: int, f, k: int = 3, ablate: bool = False):
        prefixes, scores = [""], [1.0]
        for _ in range(L):
            cand = []
            for pref, sc in zip(prefixes, scores):
                d = self.digits(prompt + pref, f, ablate)
                for di in range(10):
                    cand.append((pref + str(di), sc * float(d[di])))
            cand.sort(key=lambda x: -x[1])
            prefixes = [x[0] for x in cand[:k]]
            scores = [x[1] for x in cand[:k]]
        return prefixes[0], scores[0]

    def beam_adaptive(self, prompt: str, Lmax: int, f, k: int = 3,
                      ablate: bool = False):
        """Laenge wird INFERIERT statt vorgegeben: Beam fuer jedes
        L in 1..Lmax, Score = logp(best)/L (per-Token-Normalisierung),
        argmax ueber L. Liefert (beste Antwort, beste Laenge)."""
        best, best_lp, best_L = None, -1e18, 0
        for L in range(1, Lmax + 1):
            cand, sc = self.beam(prompt, L, f, k=k, ablate=ablate)
            lp = math.log(sc + 1e-300) / L
            if lp > best_lp:
                best, best_lp, best_L = cand, lp, L
        return best, best_L


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--donor-npz", required=True)
    ap.add_argument("--host", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--lmax", type=int, default=3,
                    help="max. Antwort-Laenge fuer den adaptiven Verifier")
    ap.add_argument("--out", default="verifier_loop.json")
    args = ap.parse_args()

    t0 = time.time()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(R.SEED)
    tok = AutoTokenizer.from_pretrained(args.host)
    host = AutoModelForCausalLM.from_pretrained(
        args.host, torch_dtype=torch.float32).to(dev).eval()
    for p in host.parameters():
        p.requires_grad_(False)

    pairs, dl, dig_idx, ev_prompts, ev_ans = load_donor_npz(
        args.donor_npz, tok, dev)
    print(f"train-set: {len(pairs)} paare (inkl. Antwort-Positionen), "
          f"eval: {len(ev_prompts)}", flush=True)

    print(f"trainiere f_host (digkl) …", flush=True)
    f_host = R.train_f_host(host, tok, pairs, dev, dl, args.steps, 1e-3,
                            R.SEED, hidden=args.hidden, loss_mode="digkl",
                            dig_idx=dig_idx)
    print(f"trainiere placebo …", flush=True)
    f_pl = R.train_f_host(host, tok, pairs, dev, dl, args.steps, 1e-3,
                          R.SEED + 2, permute=True, hidden=args.hidden,
                          loss_mode="digkl", dig_idx=dig_idx)
    f_sh = copy.deepcopy(f_host)
    with torch.no_grad():
        for p in f_sh.parameters():
            flat = p.flatten()
            perm = torch.randperm(flat.numel(), device=flat.device)
            p.copy_(flat[perm].reshape(p.shape))

    ver = Verifier(host, tok, dev, dig_idx)
    modes = {"intact": (None, False), "ablated": (None, True),
             "graft": (f_host, True), "placebo": (f_pl, True),
             "placebo2": (f_sh, True)}
    res = {"donor": args.donor_npz, "host": args.host, "k": args.k,
           "lmax": args.lmax, "cfg": {}}
    for name, (f, ablate) in modes.items():
        ok_b, ok_g, ok_a, n = 0, 0, 0, 0
        lp_true = 0.0
        for p_, a_ in zip(ev_prompts, ev_ans):
            L = len(a_)
            b, sc = ver.beam(p_, L, f, k=args.k, ablate=ablate)
            g, _ = ver.beam(p_, L, f, k=1, ablate=ablate)
            a, aL = ver.beam_adaptive(p_, args.lmax, f, k=args.k,
                                      ablate=ablate)
            ok_b += int(b == a_)
            ok_g += int(g == a_)
            ok_a += int(a == a_)
            # Wahrscheinlichkeit der wahren Antwort (Beam-Pfad)
            lp = 0.0
            pref = ""
            for ch in a_:
                d = ver.digits(p_ + pref, f, ablate)
                lp += float(torch.log(d[int(ch)] + 1e-9))
                pref += ch
            lp_true += lp
            n += 1
        res["cfg"][name] = {"beam_acc": round(ok_b / n, 4),
                            "greedy_acc": round(ok_g / n, 4),
                            "adaptive_acc": round(ok_a / n, 4),
                            "mean_logp_true": round(lp_true / n, 4)}
        print(f"  {name:8s}: beam {ok_b/n:.3f}  greedy {ok_g/n:.3f}  "
              f"adaptive {ok_a/n:.3f}  logp(true) {lp_true/n:.3f}",
              flush=True)

    res["meta"] = {"runtime_s": round(time.time() - t0, 1), "device": dev,
                   "steps": args.steps}
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"\n-> {args.out} | gesamt {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
