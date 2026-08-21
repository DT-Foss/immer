"""cross_graft.py — S3-Miniatur: Organ aus Qwen2.5-1.5B in Qwen2.5-0.5B-Wirt.

Die zentrale S3-Frage (MOONSHOT-KOMPLETT §6.1): Trägt eine destillierte
Organ-FUNKTION ueber Modell- UND Dimensionsgrenzen? e15 (mitGLM) zeigte:
lineare Punkt-Abbildung (Procrustes) scheitert kontrolliert. Hier der
ehrliche Test mit Funktion in der Mitte:

    graft(x_wirt) = P_out( f_donor( P_in(x_wirt) ) )

  - P_in  : d_wirt -> d_donor   (Projektion, aus parallelen Forward-Paaren)
  - f     : d_donor -> d_donor  (destillierte Organ-Funktion des Spenders)
  - P_out : d_donor -> d_wirt   (Projektion zurueck in den Wirt)

Ablauf:
  1. Spender-Organ lokalisieren (Ablations-Sweep ueber MLP-Gruppen @Lese).
  2. Paare sammeln: (x_donor, y_donor) am Organ; parallel (x_wirt, y_wirt)
     am Wirt-Organ (gleiche Prompts).
  3. f trainieren (Stitch), P_in/P_out per Least-Squares.
  4. Wirt testen: intact / ablated / graft(P_out f P_in) / placebo
     (P zufaellig ODER f auf permutierten y) / placebo2 (Shuffle).
  Falsifikator: graft - placebo < 3x Abstand -> Cross-Modell-Organ traegt
  nicht; dann ist die Embedding-/Vokabular-Bruecke (M6) der noetige Weg.

Nur Rechnung im eigenen Ordner; schreibt JSON nach --out.
"""
from __future__ import annotations

import argparse
import json
import random
import time

import torch
import torch.nn as nn

SEED = 7
NOISE_SCALE = 3.0
FEWSHOT = "17+25=42\n38+14=52\n"
SENTINEL = "The quick brown fox jumps over the lazy dog. "


def make_tasks(tok, n, seed):
    rng = random.Random(seed)
    tasks = []
    while len(tasks) < n:
        a, b = rng.randint(12, 88), rng.randint(11, 87)
        tasks.append((FEWSHOT + f"{a}+{b}=",
                      tok.encode(str(a + b), add_special_tokens=False)[0]))
    return tasks


def encode(tok, tasks, dev):
    enc = tok([p for p, _ in tasks], return_tensors="pt", padding=True)
    ids = enc["input_ids"].to(dev)
    attn = enc["attention_mask"].to(dev)
    ans = torch.tensor([t for _, t in tasks], device=dev)
    return ids, attn, ans, attn.sum(1) - 1


def answer_prob(logits, last, ans):
    p = torch.softmax(logits[torch.arange(logits.shape[0]), last], -1)
    return p[torch.arange(logits.shape[0]), ans]


def read_mask_of(attn, last, dev):
    B, T = attn.shape
    pos = torch.arange(T, device=dev).unsqueeze(0).expand(B, T)
    return pos == last.unsqueeze(1)


def find_organ(model, tok, tasks, dev, group=4):
    """Ablations-Sweep: welche MLP-Gruppe @Lese traegt die Aufgabe?"""
    ids, attn, ans, last = encode(tok, tasks, dev)
    rm = read_mask_of(attn, last, dev)
    layers = model.model.layers
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=attn)
    base = float(answer_prob(out.logits, last, ans).mean())
    results = []
    n_layers = len(layers)
    for lo in range(0, n_layers, group):
        hi = min(lo + group, n_layers)
        hooks = [layers[L].mlp.register_forward_pre_hook(
            lambda m, a, _rm=rm: (a[0].clone().masked_fill(_rm.unsqueeze(-1), 0.0),))
            for L in range(lo, hi)]
        with torch.no_grad():
            o = model(input_ids=ids, attention_mask=attn)
        for h in hooks:
            h.remove()
        results.append((lo, hi, float(answer_prob(o.logits, last, ans).mean())))
    return base, results


def collect_organ_pairs(model, tok, tasks, dev, lo, hi):
    ids, attn, _, last = encode(tok, tasks, dev)
    rm = read_mask_of(attn, last, dev)
    layers = model.model.layers
    xs, ys = {}, {}
    xh = layers[lo].mlp.register_forward_pre_hook(
        lambda m, a: xs.__setitem__("x", a[0].detach()))
    yhooks = [layers[L].mlp.register_forward_hook(
        (lambda s, k: lambda m, a, o: s.__setitem__(k, o.detach()))(ys, L))
        for L in range(lo, hi)]
    with torch.no_grad():
        model(input_ids=ids, attention_mask=attn)
    xh.remove()
    for h in yhooks:
        h.remove()
    X = torch.stack([xs["x"][b, last[b]] for b in range(len(last))])
    Y = torch.stack([sum(ys[L][b, last[b]] for L in range(lo, hi))
                     for b in range(len(last))])
    return X, Y


def train_f(X, Y, d, hidden, steps, lr, seed, permute=False, dev=None):
    torch.manual_seed(seed)
    if permute:
        Y = Y[torch.randperm(Y.shape[0])]
    net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d)).to(dev or X.device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    n = X.shape[0]
    for _ in range(steps):
        idx = torch.randperm(n)[: min(n, 64)]
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(net(X[idx]), Y[idx])
        loss.backward()
        opt.step()
    return net


def proj(A, B):
    """Least-Squares-Projektion P: A @ P^T ~= B  ->  P = B^T A (A^T A)^-1."""
    AtA = A.T @ A
    AtA = AtA + 1e-4 * torch.eye(AtA.shape[0], device=AtA.device)
    P = (B.T @ A @ torch.linalg.inv(AtA)).T   # (d_in, d_out)
    return P.detach()


def run_graft(model, tok, tasks, dev, lo, hi, mode, P_in=None, f=None,
              P_out=None):
    ids, attn, ans, last = encode(tok, tasks, dev)
    rm = read_mask_of(attn, last, dev)
    layers = model.model.layers
    donor = {}

    def zero_mlp(module, hargs):
        x = hargs[0].clone()
        if mode == "graft" and module is layers[lo].mlp:
            donor["x"] = x.detach()[rm]
        if mode != "intact":
            x[rm] = 0.0   # boolesches Indexing: (B,T)-Maske auf (B,T,D) ok
        return (x,)

    def add_graft(module, hargs, hout):
        out = hout.clone()
        if donor:
            z = P_in(donor["x"])
            out[rm] = out[rm] + P_out(f(z))
        return out

    hooks = [layers[L].mlp.register_forward_pre_hook(zero_mlp)
             for L in range(lo, hi)]
    if mode == "graft":
        hooks.append(layers[hi - 1].mlp.register_forward_hook(add_graft))
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=attn)
    for h in hooks:
        h.remove()
    p = answer_prob(out.logits, last, ans)
    acc = float((out.logits[torch.arange(len(ans)), last].argmax(-1) == ans).float().mean())
    st = tok(SENTINEL, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        ppl = float(torch.nn.functional.cross_entropy(
            model(input_ids=st).logits[0, :-1], st[0, 1:]))
    return acc, float(p.mean()), ppl


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--donor", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--host", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--n-pairs", type=int, default=256)
    ap.add_argument("--n-eval", type=int, default=96)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--out", default="cross_graft.json")
    args = ap.parse_args()

    t0 = time.time()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(SEED); random.seed(SEED)
    tok = AutoTokenizer.from_pretrained(args.host)
    host = AutoModelForCausalLM.from_pretrained(args.host, dtype=torch.float32).to(dev).eval()
    donor = AutoModelForCausalLM.from_pretrained(args.donor, dtype=torch.float32).to(dev).eval()
    print(f"host: {args.host} d={host.config.hidden_size}, "
          f"donor: {args.donor} d={donor.config.hidden_size} ({dev})", flush=True)

    loc_tasks = make_tasks(tok, 48, SEED)
    base, sweep = find_organ(donor, tok, loc_tasks, dev, args.group)
    print(f"donor organ-sweep (p_clean={base:.3f}):", flush=True)
    for lo, hi, p in sweep:
        print(f"  L{lo:2d}-{hi-1:2d}: p={p:.3f} (kollaps {1 - p/base:.0%})", flush=True)
    lo, hi, _ = min(sweep, key=lambda r: r[2])
    print(f"-> organ: mlp L{lo}-{hi-1} @lese", flush=True)

    pairs_tasks = make_tasks(tok, args.n_pairs, SEED + 1)
    Xd, Yd = collect_organ_pairs(donor, tok, pairs_tasks, dev, lo, hi)
    Xh, Yh = collect_organ_pairs(host, tok, pairs_tasks, dev, 16, 24)
    print(f"paare: donor {Xd.shape} | wirt {Xh.shape}", flush=True)

    f = train_f(Xd, Yd, Xd.shape[1], args.hidden, args.steps, 1e-3, SEED, dev=dev)
    f_placebo = train_f(Xd, Yd, Xd.shape[1], args.hidden, args.steps, 1e-3,
                        SEED + 2, permute=True, dev=dev)
    P_in = proj(Xh, Xd)
    P_out = proj(Yd, Yh)
    print(f"projektionen: P_in {tuple(P_in.shape)} P_out {tuple(P_out.shape)}", flush=True)
    Pin_n = nn.Linear(Xh.shape[1], Xd.shape[1], bias=False).to(dev)
    Pout_n = nn.Linear(Xd.shape[1], Xh.shape[1], bias=False).to(dev)
    with torch.no_grad():
        Pin_n.weight.copy_(P_in.T)     # Linear-Gewichte sind (out, in)
        Pout_n.weight.copy_(P_out.T)

    eval_tasks = make_tasks(tok, args.n_eval, SEED + 3)
    res = {"donor": args.donor, "host": args.host, "organ": f"L{lo}-{hi-1}",
           "sweep": [{"lo": a, "hi": b, "p": c} for a, b, c in sweep], "cfg": {}}
    print("\nkonfigurationen (wirt, organ L16-23 ablated):", flush=True)
    for mode in ("intact", "ablated", "placebo", "placebo2"):
        acc, p, ppl = run_graft(host, tok, eval_tasks, dev, 16, 24, mode)
        res["cfg"][mode] = {"acc": round(acc, 4), "p": round(p, 4), "ppl": round(ppl, 4)}
        print(f"  {mode:8s}: acc {acc:.3f}  p {p:.4f}  ppl {ppl:.3f}", flush=True)
    # placebo-graft: f_placebo statt f (gleiche P)
    acc, p, ppl = run_graft(host, tok, eval_tasks, dev, 16, 24, "graft",
                            P_in=Pin_n, f=f_placebo, P_out=Pout_n)
    res["cfg"]["placebo_graft"] = {"acc": round(acc, 4), "p": round(p, 4), "ppl": round(ppl, 4)}
    print(f"  placebo_gr: acc {acc:.3f}  p {p:.4f}  ppl {ppl:.3f}", flush=True)
    # echter graft mit P + f
    acc, p, ppl = run_graft(host, tok, eval_tasks, dev, 16, 24, "graft",
                            P_in=Pin_n, f=f, P_out=Pout_n)
    res["cfg"]["graft"] = {"acc": round(acc, 4), "p": round(p, 4), "ppl": round(ppl, 4)}
    print(f"  graft     : acc {acc:.3f}  p {p:.4f}  ppl {ppl:.3f}", flush=True)

    res["meta"] = {"runtime_s": round(time.time() - t0, 1), "device": dev}
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"\n-> {args.out} | gesamt {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
