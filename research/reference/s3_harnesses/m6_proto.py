"""m6_proto.py — M6-1: Stimmen im Kopf + Gate (lokal testbar).

Zwei Stimmen im GETEILTEN Vokabular-Raum (Qwen2.5-Familie, 151936 V):
  - Stimme A: Arithmetic-Adapter f(h) @ W_head^T  (Logit-Ebene, kein
    Residual-Umweg — die v6-Lektion)
  - Stimme B: Host-eigene Praediktion (Prior)
Gate g(h) ∈ Δ² arbitriert pro Position; Loss = CE(gemischte Logits)
+ λ·L1(g_A) (Sparsity: die Stimme muss sich verdienen).

Messung: acc(gate) vs acc(Host-only) vs acc(Stimme-A-only) vs acc(50/50).
Falsifikator M6-1: Gate <= beste Einzelstimme.
"""
from __future__ import annotations

import argparse
import json
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 7
TASKS = {
    "add2": ("17+25=42\n38+14=52\n",
             lambda r: (lambda a, b: (f"{a}+{b}=", str(a + b)))(
                 r.randint(12, 88), r.randint(11, 87))),
    "twostep": ("Example: a=3, b=a+2, c=b*2 -> c=10\n"
                "Example: a=5, b=a+1, c=b*3 -> c=18\n",
                lambda r: (lambda a: (f"a={a}, b=a+2, c=b*2 -> c=",
                                      str((a + 2) * 2)))(r.randint(1, 9))),
    "wordlen": ("Example: length of 'hello' is 5\n"
                "Example: length of 'world' is 5\n",
                lambda r: (lambda w: (f"length of '{w}' is ", str(len(w))))(
                    "".join(random.Random(r.randrange(999)).choice("abcdefghij")
                            for _ in range(r.randint(3, 8))))),
}


def make_tasks(tok, n, seed, task="add2"):
    few, gen = TASKS[task]
    rng = random.Random(seed)
    tasks = []
    while len(tasks) < n:
        q, ans = gen(rng)
        tasks.append((few + q, tok.encode(str(ans), add_special_tokens=False)[0]))
    return tasks


def encode(tok, tasks, dev):
    enc = tok([p for p, _ in tasks], return_tensors="pt", padding=True)
    ids = enc["input_ids"].to(dev)
    attn = enc["attention_mask"].to(dev)
    ans = torch.tensor([t for _, t in tasks], device=dev)
    return ids, attn, ans, attn.sum(1) - 1


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--donor-mlx", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--donor2", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                    help="M6-2: zweite Spender-Stimme (schwacherer Donor)")
    ap.add_argument("--dual", action="store_true",
                    help="M6-2: zweite Stimme trainieren + Dual-Gate messen")
    ap.add_argument("--n-pairs", type=int, default=192)
    ap.add_argument("--n-eval", type=int, default=96)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--lam", type=float, default=0.05, help="L1 auf g_A")
    ap.add_argument("--task", default="add2", choices=sorted(TASKS))
    ap.add_argument("--out", default="m6_proto.json")
    args = ap.parse_args()

    t0 = time.time()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(SEED); random.seed(SEED)
    tok = AutoTokenizer.from_pretrained(args.host)
    host = AutoModelForCausalLM.from_pretrained(args.host, dtype=torch.float32).to(dev).eval()
    for p in host.parameters():
        p.requires_grad_(False)
    d = host.config.hidden_size
    V = host.config.vocab_size
    W_head = host.lm_head.weight.detach()          # (V, d) geteilter Raum

    from mlx_lm import load as mlx_load
    import mlx.core as mx
    import numpy as np
    donor_mlx, _ = mlx_load(args.donor_mlx)
    print(f"host {args.host} d={d}, donor {args.donor_mlx} ({dev})", flush=True)

    pairs = make_tasks(tok, args.n_pairs, SEED + 1, task=args.task)
    evals = make_tasks(tok, args.n_eval, SEED + 3, task=args.task)
    ids, attn, ans, last = encode(tok, pairs, dev)
    lg = donor_mlx(mx.array(ids.cpu().tolist()))
    dl = torch.tensor(np.array(lg), device=dev)
    donor_t = dl[torch.arange(len(last)), last].detach()   # (B, V)
    donor_t = donor_t[:, :V]    # 7B-Instruct: +128 Sonder-Tokens am Ende
    # Speicher-Hygiene: das MLX-Modell wird nach dem Forward NICHT mehr
    # gebraucht (nur die Logits zaehlen) — Freigabe verhindert MPS-OOM
    # beim zweiten Donor (gemessen 19.8.: 20,1-GiB-Limit gerissen).
    del donor_mlx, dl
    import gc
    gc.collect()

    donor_t2 = None
    if args.dual:
        donor2, _ = mlx_load(args.donor2)
        lg2 = donor2(mx.array(ids.cpu().tolist()))
        dl2 = torch.tensor(np.array(lg2), device=dev)
        donor_t2 = dl2[torch.arange(len(last)), last].detach()[:, :V]
        del donor2, dl2
        gc.collect()
        print(f"M6-2: zweite Stimme vom Donor {args.donor2}", flush=True)

    def train_voice(permute: bool, tag: str, donor_t_use=None):
        voice = nn.Sequential(nn.Linear(d, args.hidden), nn.GELU(),
                              nn.Linear(args.hidden, d)).to(dev)
        gate = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 1)).to(dev)
        opt = torch.optim.AdamW(list(voice.parameters())
                                + list(gate.parameters()), lr=1e-3)
        dl_t = donor_t if donor_t_use is None else donor_t_use
        if permute:
            dl_t = dl_t[torch.randperm(len(dl_t))]
        B = 32
        n = len(pairs)
        for step in range(args.steps):
            idx = torch.randperm(n)[:B]
            out = host(input_ids=ids[idx], output_hidden_states=True)
            h = out.hidden_states[-1]                   # (B, T, d) pre-norm
            h_r = h[torch.arange(B), last[idx]]         # Lese-Position
            L_A = voice(h_r) @ W_head.T                 # Stimme A (B, V)
            g_A = torch.sigmoid(gate(h_r))              # Gate-Gewicht
            logits = out.logits[torch.arange(B), last[idx]]
            final = logits + g_A * L_A
            # v2: KL gegen DONOR-Verteilung (das war der Design-Fehler v1:
            # hartes CE auf Antworten = memorisierbar, Targets ungenutzt)
            loss = (F.kl_div(F.log_softmax(final / 2.0, -1),
                             F.softmax(dl_t[idx] / 2.0, -1),
                             reduction="batchmean") * 4.0
                    + args.lam * g_A.mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 40 == 0:
                print(f"  [{tag}] step {step}: loss={float(loss.detach()):.4f} "
                      f"g_A={float(g_A.mean()):.3f}", flush=True)
        return voice, gate

    voice, gate = train_voice(False, "echt")
    voice_pl, gate_pl = train_voice(True, "placebo")
    voice2 = gate2 = None
    if args.dual:
        voice2, gate2 = train_voice(False, "donor2", donor_t2)

    # Messung (voice/gate austauschbar fuer Placebo)
    def acc_with(g_mode, voice_use, gate_use, voice2_use=None, gate2_use=None):
        ok = 0
        for i in range(0, len(evals), 32):
            sub = evals[i:i + 32]
            s_ids, s_attn, s_ans, s_last = encode(tok, sub, dev)
            with torch.no_grad():
                out = host(input_ids=s_ids, output_hidden_states=True)
                h_r = out.hidden_states[-1][torch.arange(len(sub)), s_last]
                L_A = voice_use(h_r) @ W_head.T
                g_A = torch.sigmoid(gate_use(h_r))
                base = out.logits[torch.arange(len(sub)), s_last]
                if g_mode == "host":
                    fin = base
                elif g_mode == "voice":
                    fin = L_A
                elif g_mode == "half":
                    fin = base + 0.5 * L_A
                elif g_mode == "gate":
                    fin = base + g_A * L_A
                elif g_mode == "voice2":
                    L_B = voice2_use(h_r) @ W_head.T
                    fin = L_B
                elif g_mode == "gate2":
                    L_B = voice2_use(h_r) @ W_head.T
                    g_B = torch.sigmoid(gate2_use(h_r))
                    fin = base + g_B * L_B
                else:  # dual
                    L_B = voice2_use(h_r) @ W_head.T
                    g_B = torch.sigmoid(gate2_use(h_r))
                    fin = base + g_A * L_A + g_B * L_B
                ok += int((fin.argmax(-1) == s_ans).sum())
        return ok / len(evals)

    @torch.no_grad()
    def gen_eval(voice_use, gate_use, use_voice=True, max_new=8):
        ok = 0
        for prompt, ans in evals:
            ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
            plen = ids.shape[1]
            for _ in range(max_new):
                out = host(input_ids=ids, output_hidden_states=True)
                h_r = out.hidden_states[-1][0, -1]         # aktuelle Position
                base = out.logits[0, -1]
                if use_voice:
                    L_A = voice_use(h_r.unsqueeze(0)) @ W_head.T
                    g_A = torch.sigmoid(gate_use(h_r.unsqueeze(0)))
                    fin = base + g_A[0] * L_A[0]
                else:
                    fin = base
                nxt = fin.argmax()
                ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
                if nxt.item() == tok.eos_token_id:
                    break
            gen = tok.decode(ids[0, plen:], skip_special_tokens=True)
            first = gen.split("\n")[0]
            digits = "".join(ch for ch in first if ch.isdigit())
            ok += int(digits == ans)
        return ok / len(evals)

    res = {"task": args.task, "cfg": {}}
    print(f"\nM6-1 Messung ({args.task}, acc):")
    for mode in ("host", "voice", "half", "gate"):
        res["cfg"][mode] = round(acc_with(mode, voice, gate), 4)
        print(f"  {mode:6s}: {res['cfg'][mode]:.3f}", flush=True)
    print("Placebo-Stimme (permutierte Targets):")
    for mode in ("voice", "gate"):
        res["cfg"][f"placebo_{mode}"] = round(acc_with(mode, voice_pl, gate_pl), 4)
        print(f"  {mode:6s}: {res['cfg'][f'placebo_{mode}']:.3f}", flush=True)
    if args.dual:
        print("M6-2 (zweite Stimme):")
        for mode in ("voice2", "gate2", "dual"):
            res["cfg"][mode] = round(
                acc_with(mode, voice, gate, voice2, gate2), 4)
            print(f"  {mode:6s}: {res['cfg'][mode]:.3f}", flush=True)

    print("\nGenerierung (Voll-Antwort, M6 an jedem Schritt):")
    for tag, v, g in (("host", voice, gate), ("m6", voice, gate),
                      ("placebo", voice_pl, gate_pl)):
        use = tag != "host"
        ga = gen_eval(v, g, use_voice=use)
        res["cfg"][f"gen_{tag}"] = round(ga, 4)
        print(f"  gen_{tag:8s}: {ga:.3f}", flush=True)

    res["meta"] = {"runtime_s": round(time.time() - t0, 1), "device": dev,
                   "lam": args.lam}
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"\n-> {args.out} | gesamt {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
