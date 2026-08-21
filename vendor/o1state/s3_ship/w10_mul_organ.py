"""w10_mul_organ.py — DIE KARTEN-FAMILIE: mul ist add unter der log-Karte.

v4.2 sagt: die Strukturform muss die GRUPPE der Aufgabe tragen. Die
multiplikative Gruppe (R⁺,·) ist die additive unter log — also braucht
Multiplikation KEINE neue Organ-Klasse, nur eine andere Karte im
selben O(1)-Akkumulator:

    acc = Σ v(token)         (dieselbe Rekurrenz wie W7)
    delta_c = s0·(log(c)·acc − q·log(c)²)     (Readout auf LOG-Gitter)

Wenn das stimmt, lernt v die LOG-Zahlengerade v(n) ≈ α·log n + γ,
und die Ganzzahl-Kristallisation snappt auf {log 1..log 16}.
Kontrast (Falsifikator): dasselbe Organ mit LINEAREM Gitter muss auf
der Ketten-Extrapolation brechen — Produkte sind im linearen Raum
nicht additiv.

Training: NUR 2-Faktor-Produkte (a times b is, Produkt <= 16).
Eval: 3- und 4-Faktor-Ketten (nie gesehen).

Prognose (festgelegt, 20.08. vor dem Lauf): log-Fit von v R² >= 0,99
und > linearer Fit; L3 roh >= 0,9; nach Log-Gitter-Snap L3/L4 = 1,000;
Linear-Gitter-Kontrast bricht auf L3 (< 0,5).

    OMP_NUM_THREADS=6 python3 w10_mul_organ.py
"""
from __future__ import annotations

import collections
import json
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from w3_komposition import O1, SEED, W  # noqa: E402
from length_extrap_v2 import build_vocab, load_wikitext2  # noqa: E402
from streaming_train import StreamingNoPELM  # noqa: E402

N_MAX = 16


class MulOrgan(nn.Module):
    """O(1)-Akkumulator mit waehlbarem Kandidaten-Gitter (log/linear)."""

    def __init__(self, e_times, e_is, gitter, d_model=128, d_e=32):
        super().__init__()
        self.P = nn.Linear(d_model, d_e, bias=False)
        self.phi = nn.Sequential(nn.Linear(d_e, 32), nn.Tanh(),
                                 nn.Linear(32, 1))
        self.register_buffer("e_times", e_times / e_times.norm())
        self.register_buffer("e_is", e_is / e_is.norm())
        self.register_buffer("cs", gitter)          # [n_cand] Gitterwerte
        self.s0 = nn.Parameter(torch.tensor(1.0))
        self.q = nn.Parameter(torch.tensor(0.5))

    def v_werte(self, emb):
        return self.phi(torch.tanh(self.P(emb))).squeeze(-1)

    def forward(self, emb):
        B, T, _ = emb.shape
        e_n = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        is_op = (e_n @ self.e_times > 0.99) | (e_n @ self.e_is > 0.99)
        v = self.v_werte(emb)
        acc = torch.zeros(B)
        deltas = []
        for t in range(T):
            acc = acc + (~is_op[:, t]).float() * v[:, t]
            deltas.append(self.s0 * (self.cs[None] * acc[:, None]
                                     - self.q * self.cs[None] ** 2))
        return torch.stack(deltas, dim=1)


def make_mult(n, n_fak, rng, stoi):
    tasks, guard = [], 0
    while len(tasks) < n and guard < 200000:
        guard += 1
        faks = [rng.randint(1, 8) for _ in range(n_fak)]
        prod = math.prod(faks)
        if prod > N_MAX:
            continue
        words = []
        for i, f in enumerate(faks):
            if i:
                words.append("times")
            words.append(W[f])
        words.append("is")
        tasks.append(([stoi[w] for w in words], prod))
    return tasks


def main() -> None:
    t0 = time.time()
    torch.manual_seed(SEED)
    train_text, _ = load_wikitext2()
    vocab, stoi, unk, mask = build_vocab(train_text)
    ck = torch.load(O1 / "results/pos_ckpt.pt", map_location="cpu",
                    weights_only=False)
    host = StreamingNoPELM(len(vocab), mask, d_model=128, n_layers=2,
                           n_heads=4, d_head=32, seq_len=64, dropout=0.0,
                           causal=True)
    host.load_state_dict(ck["arms"]["A1"]["model"])
    host.eval()
    for p in host.parameters():
        p.requires_grad_(False)
    E = host.embed.weight.detach()

    train = make_mult(512, 2, random.Random(SEED), stoi)
    evals = {f"L{k}": make_mult(96, k, random.Random(SEED + 60 + k), stoi)
             for k in (2, 3, 4)}
    print(f"train L2: {len(train)} | evals: "
          f"{[len(v) for v in evals.values()]}", flush=True)

    log_g = torch.tensor([math.log(c) for c in range(1, N_MAX + 1)])
    lin_g = torch.arange(1, N_MAX + 1).float()

    def messen(organ, tasks):
        by_l = collections.defaultdict(list)
        for t in tasks:
            by_l[len(t[0])].append(t)
        accs = []
        for L, grp in by_l.items():
            ids = torch.tensor([t[0] for t in grp])
            tgt = torch.tensor([t[1] - 1 for t in grp])
            with torch.no_grad():
                delta = organ(host.embed(ids))[:, -1]
            accs += (delta.argmax(1) == tgt).tolist()
        return round(sum(accs) / len(accs), 4)

    res = {}
    organe = {}
    for name, gitter in (("log", log_g), ("linear", lin_g)):
        torch.manual_seed(SEED)
        organ = MulOrgan(E[stoi["times"]], E[stoi["is"]], gitter)
        opt = torch.optim.AdamW(organ.parameters(), lr=3e-3)
        g = torch.Generator().manual_seed(SEED)
        tgt_all = torch.tensor([t[1] - 1 for t in train])
        cnt = collections.Counter(tgt_all.tolist())
        wts = torch.tensor([1.0 / cnt[int(t)] for t in tgt_all])
        ids_all = torch.tensor([t[0] for t in train])
        for _ in range(2400):
            idx = torch.randperm(len(train), generator=g)[:32]
            with torch.no_grad():
                emb = host.embed(ids_all[idx])
            delta = organ(emb)[:, -1]
            li = F.cross_entropy(delta, tgt_all[idx], reduction="none")
            loss = (li * wts[idx]).sum() / wts[idx].sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        organe[name] = organ
        r = {k: messen(organ, ev) for k, ev in evals.items()}
        res[name] = r
        print(f"{name}-Gitter: {r}", flush=True)

    # Karten-Diagnose am log-Organ: folgt v der log- oder linear-Karte?
    organ = organe["log"]
    with torch.no_grad():
        vs = torch.tensor([float(organ.v_werte(E[stoi[W[n]]][None, None])
                                 [0, 0]) for n in range(1, 9)])

    def fit_r2(xs):
        x, y = xs, vs
        a = float(((x - x.mean()) * (y - y.mean())).sum()
                  / ((x - x.mean()) ** 2).sum())
        b = float(y.mean() - a * x.mean())
        return a, b, 1 - float(((y - (a * x + b)) ** 2).sum()
                               / ((y - y.mean()) ** 2).sum())

    ns = torch.arange(1, 9).float()
    a_log, b_log, r2_log = fit_r2(torch.log(ns))
    _, _, r2_lin = fit_r2(ns)
    res["karte"] = {"r2_log": round(r2_log, 5), "r2_linear": round(r2_lin, 5),
                    "alpha": round(a_log, 4), "gamma": round(b_log, 4),
                    "v_werte": [round(float(v), 4) for v in vs]}
    print(f"KARTE: {res['karte']}", flush=True)

    # Ganzzahl-Kristallisation im log-Raum: snap auf {log 1..log 16}
    class KristallMul(nn.Module):
        def __init__(self, base, alpha, gamma):
            super().__init__()
            self.base, self.alpha, self.gamma = base, alpha, gamma
            self.register_buffer("gitter", torch.log(
                torch.arange(1, N_MAX + 1).float()))

        def forward(self, emb):
            b = self.base
            B, T, _ = emb.shape
            e_n = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            is_op = (e_n @ b.e_times > 0.99) | (e_n @ b.e_is > 0.99)
            v = (b.v_werte(emb) - self.gamma) / self.alpha
            v = self.gitter[(v[..., None] - self.gitter).abs().argmin(-1)]
            acc = torch.zeros(B)
            deltas = []
            for t in range(T):
                acc = acc + (~is_op[:, t]).float() * v[:, t]
                deltas.append(10.0 * (self.gitter[None] * acc[:, None]
                                      - 0.5 * self.gitter[None] ** 2))
            return torch.stack(deltas, dim=1)

    krist = KristallMul(organ, a_log, b_log)
    res["kristall_log"] = {k: messen(krist, ev) for k, ev in evals.items()}
    print(f"KRISTALL (log-Gitter-Snap): {res['kristall_log']}", flush=True)

    json.dump({"results": res,
               "prognose": "r2_log >= 0,99 und > r2_linear; log L3 roh "
                           ">= 0,9; Kristall L3/L4 = 1,000; linear L3 < 0,5",
               "runtime_s": round(time.time() - t0, 1)},
              open("w10_mul_organ.json", "w"), indent=1)
    print(f"({time.time()-t0:.0f}s) -> w10_mul_organ.json", flush=True)


if __name__ == "__main__":
    main()
