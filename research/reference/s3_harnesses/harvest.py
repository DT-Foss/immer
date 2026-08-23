"""harvest.py — S3-Ernte-Werkzeug: Fähigkeits-Organe aus Frontier-Modellen.

Skaliert die S2-Maschinerie (s2_stitch.py, PoC auf Qwen2.5-0.5B) auf
beliebige HF-Modelle und Organ-Kandidaten:

    python3 harvest.py --model Qwen/Qwen2.5-0.5B --task addition \
        --mlp-lo 16 --mlp-hi 24                # = S2-PoC (Regression)
    python3 harvest.py --model Qwen/Qwen3.8-27B --task arithmetic \
        --mlp-lo 56 --mlp-hi 61 --dtype bf16   # S3 candidate (remote 4-bit MLX donor)

Ablauf (identisch zu S2, bewiesen):
  1. Paare am Lese-Punkt: x = MLP-Eingang des ersten Organ-Layers,
     y = Summe der MLP-Ausgaben der Organ-Layer (clean; optional korrupt).
  2. Stitch-Adapter (d->d->d, L2) trainieren — FUNKTION, nicht Gewichte.
  3. Im Wirt testen: intact / ablated / graft / placebo (permutiert) /
     placebo2 (Gewicht-Shuffle) + Sentinel-PPL.
  Falsifikator (ENDGOAL W3): Transplantat ~= Placebo -> Graft traegt nichts.

W8-Lektion (KIMI): Zielgroesse ist VERHALTEN (acc/p), nicht Signatur —
die Organ-Kandidaten kommen aus Verhaltens-Sonden + mitGLM-Karten, nicht
aus Signatur-Maps allein.

Konventionen wie s2_stitch.py; nur Rechnung, schreibt JSON nach --out.
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
FEWSHOT_ADD = "17+25=42\n38+14=52\n"
FEWSHOT_SUB = "42-17=25\n52-38=14\n"
SENTINEL = ("The quick brown fox jumps over the lazy dog. "
            "Neural networks learn by gradient descent on large datasets.")


def make_tasks(tok, fewshot: str, op: str, n: int, seed: int):
    rng = random.Random(seed)
    tasks = []
    while len(tasks) < n:
        a, b = rng.randint(12, 88), rng.randint(11, 87)
        if op == "sub" and a <= b:
            continue
        ans = a - b if op == "sub" else a + b
        tasks.append((fewshot + f"{a}{'+' if op == 'add' else '-'}{b}=",
                      tok.encode(str(ans), add_special_tokens=False)[0]))
    return tasks


def encode(tok, tasks, dev):
    enc = tok([p for p, _ in tasks], return_tensors="pt", padding=True)
    ids, attn = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
    ans = torch.tensor([t for _, t in tasks], device=dev)
    return ids, attn, ans, attn.sum(1) - 1


def answer_prob(logits, last, ans):
    p = torch.softmax(logits[torch.arange(logits.shape[0]), last], -1)
    return p[torch.arange(logits.shape[0]), ans]


class StitchAdapter(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))

    def forward(self, x):
        return self.net(x)


def collect_pairs(model, tok, tasks, dev, lo, hi, corrupt=False,
                  fewshot=FEWSHOT_ADD, noise_scale=NOISE_SCALE,
                  organ_path="mlp"):
    ids, attn, _, last = encode(tok, tasks, dev)
    B, T = ids.shape
    prefix_len = len(tok.encode(fewshot, add_special_tokens=False))
    layers = model.model.layers
    organ_mods = [_resolve(layers[L], organ_path) for L in range(lo, hi)]

    def mk_x(store):
        def h(module, hargs):
            store["x"] = hargs[0].detach()
        return h

    def mk_y(store, key):
        def h(module, hargs, hout):
            store[key] = hout.detach()
        return h

    ys = {}
    organ_mods = [_resolve(layers[L], organ_path) for L in range(lo, hi)]
    hooks = [m.register_forward_hook(mk_y(ys, L)) for L, m in enumerate(organ_mods, start=lo)]
    xs = {}
    with torch.no_grad():
        if corrupt:
            emb = model.get_input_embeddings()(ids).detach()
            pos = torch.arange(T, device=dev).unsqueeze(0).expand(B, T)
            corr = (pos >= prefix_len) & (pos < last.unsqueeze(1)) & attn.bool()
            g = torch.Generator(device="cpu").manual_seed(SEED)
            noise = torch.randn(emb.shape, generator=g).to(dev) * noise_scale * emb.std()
            hx = organ_mods[0].register_forward_pre_hook(mk_x(xs))
            model(inputs_embeds=emb + noise * corr.unsqueeze(-1), attention_mask=attn)
            hx.remove()
        else:
            hx = organ_mods[0].register_forward_pre_hook(mk_x(xs))
            model(input_ids=ids, attention_mask=attn)
            hx.remove()
        model(input_ids=ids, attention_mask=attn)  # clean y
    for h in hooks:
        h.remove()
    X = torch.stack([xs["x"][b, last[b]] for b in range(B)])
    Y = torch.stack([sum(ys[L][b, last[b]] for L in range(lo, hi)) for b in range(B)])
    return X, Y


def train_adapter(X, Y, d, hidden, steps, lr, seed, permute=False, device=None):
    torch.manual_seed(seed)
    if permute:
        Y = Y[torch.randperm(Y.shape[0])]
    net = StitchAdapter(d, hidden).to(device or X.device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    n = X.shape[0]
    for _ in range(steps):
        idx = torch.randperm(n)[: min(n, 64)]
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(net(X[idx]), Y[idx])
        loss.backward()
        opt.step()
    return net


def _resolve(layer, path: str):
    m = layer
    for part in path.split("."):
        m = getattr(m, part)
    return m


def run_config(model, tok, tasks, dev, lo, hi, mode, adapter=None,
               fewshot=FEWSHOT_ADD, corrupt=False, organ_path="mlp"):
    ids, attn, ans, last = encode(tok, tasks, dev)
    B, T = ids.shape
    pos = torch.arange(T, device=dev).unsqueeze(0).expand(B, T)
    prefix_len = len(tok.encode(fewshot, add_special_tokens=False))
    read_mask = pos == last.unsqueeze(1)
    layers = model.model.layers
    donor = {}

    def zero_mlp(module, hargs):
        x = hargs[0].clone()
        if mode in ("graft", "graft_corr", "placebo", "placebo2") \
                and module is organ_mods[0]:
            donor["x"] = x.detach()[read_mask]
        x[read_mask] = 0.0
        return (x,)

    def add_graft(module, hargs, hout):
        out = hout.clone()
        if donor:
            out[read_mask] = out[read_mask] + adapter(donor["x"])
        return out

    organ_mods = [_resolve(layers[L], organ_path) for L in range(lo, hi)]
    hooks = []
    if mode != "intact":
        hooks += [m.register_forward_pre_hook(zero_mlp) for m in organ_mods]
    if mode in ("graft", "graft_corr", "placebo", "placebo2"):
        hooks.append(organ_mods[-1].register_forward_hook(add_graft))

    with torch.no_grad():
        if corrupt:
            emb = model.get_input_embeddings()(ids).detach()
            corr = (pos >= prefix_len) & (pos < last.unsqueeze(1)) & attn.bool()
            g = torch.Generator(device="cpu").manual_seed(SEED)
            noise = torch.randn(emb.shape, generator=g).to(dev) * NOISE_SCALE * emb.std()
            out = model(inputs_embeds=emb + noise * corr.unsqueeze(-1),
                        attention_mask=attn)
        else:
            out = model(input_ids=ids, attention_mask=attn)
    for h in hooks:
        h.remove()
    p = answer_prob(out.logits, last, ans)
    acc = float((out.logits[torch.arange(B), last].argmax(-1) == ans).float().mean())

    st = tok(SENTINEL, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        logits = model(input_ids=st).logits[0, :-1]
    ppl = float(torch.nn.functional.cross_entropy(logits, st[0, 1:]))
    return acc, float(p.mean()), ppl


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--task", choices=["add", "sub"], default="add")
    ap.add_argument("--mlp-lo", type=int, default=16)
    ap.add_argument("--mlp-hi", type=int, default=24)
    ap.add_argument("--organ-module", default="mlp",
                    help="Modul-Pfad relativ zu layers[L], z. B. 'mlp', "
                         "'gdn_ba', 'gdn.in_proj_a', 'self_attn.o_proj' "
                         "(GDN-Hybrid: 16x(3x(GDN->FFN)->1x(GatedAttn->FFN)))")
    ap.add_argument("--n-distill", type=int, default=256)
    ap.add_argument("--n-eval", type=int, default=96)
    ap.add_argument("--hidden", type=int, default=896)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--out", default="harvest.json")
    args = ap.parse_args()

    t0 = time.time()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(SEED); random.seed(SEED)
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(dev).eval()
    d = model.config.hidden_size
    print(f"modell: {args.model} ({model.config.num_hidden_layers} layer, d={d}, "
          f"{args.dtype}, {dev}), organ {args.organ_module} "
          f"L{args.mlp_lo}-{args.mlp_hi-1}", flush=True)
    # Organ-Modul-Aufloesung (GDN-Hybrid-tauglich): Pfad relativ zu layers[L]
    probe = model.model.layers[args.mlp_lo]
    organ = probe
    for part in args.organ_module.split("."):
        organ = getattr(organ, part)
    print(f"organ-modul: {type(organ).__name__} @ {args.organ_module}", flush=True)

    fewshot = FEWSHOT_ADD if args.task == "add" else FEWSHOT_SUB
    op = args.task
    distill = make_tasks(tok, fewshot, op, args.n_distill, SEED)
    evals = make_tasks(tok, fewshot, op, args.n_eval, SEED + 1)

    X, Y = collect_pairs(model, tok, distill, dev, args.mlp_lo, args.mlp_hi,
                         fewshot=fewshot, organ_path=args.organ_module)
    Xc, Yc = collect_pairs(model, tok, distill, dev, args.mlp_lo, args.mlp_hi,
                           corrupt=True, fewshot=fewshot,
                           organ_path=args.organ_module)
    print(f"paare: {X.shape[0]} clean / {Xc.shape[0]} korrupt, "
          f"|x|={float(X.norm(dim=1).mean()):.1f} |y|={float(Y.norm(dim=1).mean()):.1f}",
          flush=True)

    adapter = train_adapter(X, Y, d, args.hidden, args.steps, args.lr, SEED, device=dev)
    adapter_corr = train_adapter(Xc, Yc, d, args.hidden, args.steps, args.lr,
                                 SEED + 1, device=dev)
    placebo = train_adapter(X, Y, d, args.hidden, args.steps, args.lr,
                            SEED + 2, permute=True, device=dev)
    shuffle = train_adapter(X, Y, d, args.hidden, args.steps, args.lr, SEED + 3,
                            device=dev)
    with torch.no_grad():
        for p in shuffle.parameters():
            flat = p.flatten()
            perm = torch.randperm(flat.numel(), device=flat.device)
            p.copy_(flat[perm].reshape(p.shape))

    NETS = {"graft": adapter, "graft_corr": adapter_corr,
            "placebo": placebo, "placebo2": shuffle}
    res = {"model": args.model, "organ": f"mlp L{args.mlp_lo}-{args.mlp_hi-1}",
           "task": args.task, "bloecke": {}}

    def block(name, tasks, corrupt=False, fewshot=FEWSHOT_ADD, modes=None):
        modes = modes or ("intact", "ablated", "graft", "graft_corr", "placebo", "placebo2")
        print(f"\nblock {name}:", flush=True)
        b = {}
        for mode in modes:
            acc, p, ppl = run_config(model, tok, tasks, dev, args.mlp_lo, args.mlp_hi,
                                     mode, NETS.get(mode), fewshot=fewshot,
                                     corrupt=corrupt, organ_path=args.organ_module)
            b[mode] = {"acc": round(acc, 4), "p_answer": round(p, 4),
                       "sentinel_ppl": round(ppl, 4)}
            print(f"  {mode:10s}: acc {acc:.3f}  p {p:.4f}  ppl {ppl:.3f}", flush=True)
        return b

    res["bloecke"][f"{args.task}_clean"] = block(f"{args.task}, clean", evals, fewshot=fewshot)
    res["bloecke"][f"{args.task}_corrupt"] = block(
        f"{args.task}, korrupt", evals, corrupt=True, fewshot=fewshot,
        modes=("intact", "ablated", "graft", "graft_corr", "placebo"))
    # Kontrast-Aufgabe (Atlas-Spezifitaet): andere Operation, gleiche Organ-Lage
    kontrast_op = "sub" if args.task == "add" else "add"
    kontrast_fs = FEWSHOT_SUB if kontrast_op == "sub" else FEWSHOT_ADD
    kontrast = make_tasks(tok, kontrast_fs, kontrast_op, args.n_eval, SEED + 5)
    res["bloecke"][f"{kontrast_op}_kontrast"] = block(
        f"{kontrast_op}, kontrast", kontrast, fewshot=kontrast_fs,
        modes=("intact", "ablated", "graft", "placebo"))

    res["meta"] = {"runtime_s": round(time.time() - t0, 1), "device": dev,
                   "dtype": args.dtype, "hidden": args.hidden, "steps": args.steps}
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"\n-> {args.out} | gesamt {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
