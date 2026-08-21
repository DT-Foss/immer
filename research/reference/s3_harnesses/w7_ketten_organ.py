"""w7_ketten_organ.py — W7: Das Organ als PROZEDUR — O(1)-Akkumulator + Längen-Generalisierung.

Der Sprung vom Reflex zum Algorithmus: Bisherige Organe sind Ein-Schritt-
Reflexe im Logit-Raum (feste Slots). Dieses Organ hat einen EIGENEN
O(1)-Zustand — einen skalaren Akkumulator, der pro Token upgedatet wird:

    Token-Strom:  three  plus  two   less  one   is
    Prozedur:     acc=v(3)  sgn=+1  acc+=v(2)  sgn=-1  acc-=v(1)  antworte(acc)

    delta_c = scale·(c·acc − q·c²)   am Lese-Punkt (Hart-Form, v3-Kanon)

DER CLAIM (F6 des Wesens, auf Organ-Ebene): trainiert NUR auf Ketten der
Länge 2 (zwei Operatoren), generalisiert die Prozedur auf Länge 3 und 4 —
NIE gesehene Strukturen. Ein flaches Slot-Organ (feste Positionen, gleiche
Kapazität) kann das strukturell nicht: der Kontrast ist eingebaut.

Prognose (festgelegt, 20.08. 00:05): rekurrent held-L2 ~1,0, L3/L4 >= 0,8;
flaches Slot-Organ auf L3 = Zufall. Falsifikator: L3 < 0,5 -> Prozedur-
Claim in dieser Form tot.

    OMP_NUM_THREADS=6 python3 w7_ketten_organ.py
"""
from __future__ import annotations

import json
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from w3_komposition import O1, SEED, W, lm_nll  # noqa: E402
from length_extrap_v2 import build_vocab, load_wikitext2, tokenize  # noqa: E402
from streaming_train import StreamingNoPELM  # noqa: E402

CAND_LO, CAND_HI = 1, 16          # Antwortraum one..sixteen


class KettenOrgan(nn.Module):
    """O(1)-Akkumulator-Organ (~9k Params).

    Pro Token: v(t) = MLP(P·e_t) (Skalar). Operator-Erkennung über
    Embedding-Match gegen frozen embed('plus'/'less'/'is'). Rekurrenz:
      sgn <- +1 nach 'plus', -1 nach 'less'
      acc <- acc + sgn·v(t)   bei Nicht-Operator-Token
    Antwort am 'is'-Punkt: delta_c = s0·(c·acc/alpha − c²/2·...) in der
    selbst-kalibrierenden Hart-Form (realisierender Init, lernbare Skala).
    """

    def __init__(self, e_plus, e_less, e_is, d_model=128, d_e=32, n_cand=16):
        super().__init__()
        self.P = nn.Linear(d_model, d_e, bias=False)
        self.phi = nn.Sequential(nn.Linear(d_e, 32), nn.Tanh(),
                                 nn.Linear(32, 1))
        self.register_buffer("e_plus", e_plus / e_plus.norm())
        self.register_buffer("e_less", e_less / e_less.norm())
        self.register_buffer("e_is", e_is / e_is.norm())
        self.n_cand = n_cand
        cs = torch.arange(CAND_LO, CAND_LO + n_cand).float()
        self.register_buffer("cs", cs)
        # realisierender Init (v3-Kanon): delta_c = s0*(c*acc - c^2/2)
        self.s0 = nn.Parameter(torch.tensor(1.0))
        self.q = nn.Parameter(torch.tensor(0.5))
        self.gate = nn.Sequential(nn.Linear(d_e, 16), nn.Tanh(),
                                  nn.Linear(16, 1))
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, emb):
        """emb [B,T,128] -> delta [B,T,n_cand], gate [B,T,1].

        Rekurrente Prozedur über T — O(1)-Zustand (acc, sgn) pro Sequenz.
        Operator-Match hart über Cosinus zum frozen Embedding (>0,99).
        """
        B, T, D = emb.shape
        e_n = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        is_plus = (e_n @ self.e_plus > 0.99)
        is_less = (e_n @ self.e_less > 0.99)
        is_op = is_plus | is_less
        v = self.phi(torch.tanh(self.P(emb))).squeeze(-1)      # [B,T]
        acc = torch.zeros(B)
        sgn = torch.ones(B)
        deltas = []
        for t in range(T):
            sgn = torch.where(is_plus[:, t], torch.ones(B), sgn)
            sgn = torch.where(is_less[:, t], -torch.ones(B), sgn)
            upd = (~is_op[:, t]).float() * sgn * v[:, t]
            acc = acc + upd
            delta_t = self.s0 * (self.cs[None] * acc[:, None]
                                 - self.q * self.cs[None] ** 2)
            deltas.append(delta_t)
        delta = torch.stack(deltas, dim=1)                     # [B,T,n_cand]
        g = torch.sigmoid(self.gate(torch.tanh(self.P(emb))))
        return delta, g


class FlachOrgan(nn.Module):
    """Kontrast: flaches Slot-Organ für Länge-2-Ketten (Slots t-5..t-1),
    gleiche v/Hart-Form — aber FESTE Struktur, keine Rekurrenz."""

    def __init__(self, e_plus, e_less, d_model=128, d_e=32, n_cand=16):
        super().__init__()
        self.P = nn.Linear(d_model, d_e, bias=False)
        self.phi = nn.Sequential(nn.Linear(d_e, 32), nn.Tanh(),
                                 nn.Linear(32, 1))
        self.register_buffer("e_plus", e_plus / e_plus.norm())
        self.register_buffer("e_less", e_less / e_less.norm())
        cs = torch.arange(CAND_LO, CAND_LO + n_cand).float()
        self.register_buffer("cs", cs)
        self.s0 = nn.Parameter(torch.tensor(1.0))
        self.q = nn.Parameter(torch.tensor(0.5))

    def forward_last(self, emb):
        """Nur am Lese-Punkt (letztes Token): Slots -5,-3,-1 relativ."""
        e_n = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        v = self.phi(torch.tanh(self.P(emb))).squeeze(-1)      # [B,T]
        # Ketten-Format L2: [a, op1, b, op2, c, is] -> Operanden bei T-6?
        # Slots fest: a=T-6, b=T-4, c=T-2; Operatoren T-5, T-3
        sgn1 = torch.where((e_n[:, -5] @ self.e_less) > 0.99, -1.0, 1.0)
        sgn2 = torch.where((e_n[:, -3] @ self.e_less) > 0.99, -1.0, 1.0)
        acc = v[:, -6] + sgn1 * v[:, -4] + sgn2 * v[:, -2]
        return self.s0 * (self.cs[None] * acc[:, None]
                          - self.q * self.cs[None] ** 2)


def make_ketten(n, n_ops, rng, stoi, unk, held_filter=None):
    """Ketten mit n_ops Operatoren; Werte bleiben in [1,16]."""
    tasks = []
    guard = 0
    while len(tasks) < n and guard < 100000:
        guard += 1
        a = rng.randint(2, 9)
        words = [W[a]]
        val = a
        ok = True
        for _ in range(n_ops):
            op = rng.choice(["plus", "less"])
            b = rng.randint(1, 5)
            nv = val + b if op == "plus" else val - b
            if not (1 <= nv <= 16) or not (1 <= val <= 16):
                ok = False
                break
            words += [op, W[b]]
            val = nv
        if not ok:
            continue
        words += ["is"]
        key = tuple(words[:-1])
        if held_filter and held_filter(key):
            continue
        tasks.append(([stoi.get(w, unk) for w in words], stoi[W[val]], key))
    return tasks


def main() -> None:
    t0 = time.time()
    torch.manual_seed(SEED)
    train_text, val_text = load_wikitext2()
    vocab, stoi, unk, mask = build_vocab(train_text)
    val_ids = tokenize(val_text, stoi, unk)
    wt2 = torch.tensor(tokenize(train_text, stoi, unk)[:200000])

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
    cand_ids = [stoi[W[c]] for c in range(CAND_LO, CAND_HI + 1)]
    cid = torch.tensor(cand_ids)

    # held-out: 15 zufällige L2-Ketten-Strukturen NIE im Training
    rng_h = random.Random(SEED + 42)
    all_l2 = make_ketten(4000, 2, random.Random(1), stoi, unk)
    uniq_keys = sorted({t[2] for t in all_l2})
    held_keys = set(rng_h.sample(uniq_keys, 15))
    train_l2 = make_ketten(512, 2, random.Random(SEED), stoi, unk,
                           held_filter=lambda k: k in held_keys)
    held_l2 = [t for t in all_l2 if t[2] in held_keys][:60]
    eval_l3 = make_ketten(96, 3, random.Random(SEED + 3), stoi, unk)
    eval_l4 = make_ketten(96, 4, random.Random(SEED + 4), stoi, unk)
    print(f"train L2: {len(train_l2)} | held L2: {len(held_l2)} "
          f"(15 Strukturen) | eval L3: {len(eval_l3)} | L4: {len(eval_l4)}",
          flush=True)

    def measure(organ_fwd, tasks):
        ids = torch.tensor([t[0] for t in tasks])
        tgt = torch.tensor([t[1] for t in tasks])
        with torch.no_grad():
            emb = host.embed(ids)
            delta = organ_fwd(emb)
        pred = cid[delta.argmax(1)]
        return float((pred == tgt).float().mean())

    results = {}
    for name in ("rekurrent", "flach"):
        torch.manual_seed(SEED)
        if name == "rekurrent":
            organ = KettenOrgan(E[stoi["plus"]], E[stoi["less"]], E[stoi["is"]])
            fwd_last = lambda emb: organ(emb)[0][:, -1]
        else:
            organ = FlachOrgan(E[stoi["plus"]], E[stoi["less"]])
            fwd_last = lambda emb: organ.forward_last(emb)
        n_par = sum(p.numel() for p in organ.parameters())
        opt = torch.optim.AdamW(organ.parameters(), lr=3e-3)
        g = torch.Generator().manual_seed(SEED)
        ids = torch.tensor([t[0] for t in train_l2])
        c2i = {c: i for i, c in enumerate(cand_ids)}
        tgt = torch.tensor([c2i[t[1]] for t in train_l2])
        # Klassen-Balance (v3-Kanon)
        import collections
        cnt = collections.Counter(tgt.tolist())
        wts = torch.tensor([1.0 / cnt[int(t)] for t in tgt])
        for step in range(2400):
            idx = torch.randperm(len(ids), generator=g)[:32]
            with torch.no_grad():
                emb = host.embed(ids[idx])
            delta = fwd_last(emb)
            li = F.cross_entropy(delta, tgt[idx], reduction="none")
            loss = (li * wts[idx]).sum() / wts[idx].sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 800 == 0:
                print(f"  [{name}] step {step}: loss {float(loss):.4f}",
                      flush=True)
        r = {"params": n_par,
             "train_l2": measure(fwd_last, train_l2[:200]),
             "held_l2": measure(fwd_last, held_l2)}
        if name == "rekurrent":
            r["l3"] = measure(fwd_last, eval_l3)
            r["l4"] = measure(fwd_last, eval_l4)
        else:
            # flaches Organ: L3 hat 8 Tokens — Slots verschieben sich;
            # Messung zeigt den strukturellen Bruch
            r["l3"] = measure(fwd_last, eval_l3)
        results[name] = {k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in r.items()}
        print(f"{name}: {results[name]}", flush=True)

    json.dump({"results": results, "prognose": "rekurrent L3/L4 >= 0.8, "
               "flach L3 = Zufall (~0.06)",
               "runtime_s": round(time.time() - t0, 1)},
              open("w7_ketten_organ.json", "w"), indent=1)
    print(f"({time.time()-t0:.0f}s) -> w7_ketten_organ.json", flush=True)


if __name__ == "__main__":
    main()
