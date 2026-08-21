"""w16_twostep_organ.py — BRÜCKEN-ORGAN twostep (2. Organ der
Serienfertigung): "a plus two times two is" -> c = 2a+4.

DESIGN (Wert-Organ, w15-Muster: Organ LIEST Operand, Shell RECHNET):
  - Strukturform: Akkumulator ueber die Wortform "W[a] plus two times
    two is". Der Parser (DezimalOrgan-Mechanik) liest die Zahlengerade
    v(one..nine); u = Wert des ERSTEN Segments (a).
  - Bridge: aus donor_targets_twostep_pos_27b.npz (352 Zeilen,
    (n,2,10)-Logits = BEIDE Antwort-Ziffern) aggregiere je Zelle a die
    2-Token-Verteilung des Antwort-Werts w=2a+4 (Zehner+Einer). Das
    Readout trainiert mit KL auf Kandidaten 1..9 (die a-Werte), weil
    das Organ NUR a lesen soll; die Antwort 2a+4 ist exakte Shell-
    Rechnung (wie w15 die mul).
  - Kristall: v(a) ueber one..nine -> alpha/gamma (Selbst-Kalibrierung,
    label-frei); u = round((v-g)/a) = a.
  - Emission (w15): Wert = 2a+4, divmod-Dekomposition, tokenweise
    Shell-Dosis auf dem Ziffern-Slice des lebenden Wirts.
  - Kontrast: Donor-Argmax (Referenz) vs Organ-roh (ohne Kristall).

Prognose (festgelegt, 20.08. vor dem Lauf): Bridge-Reinheit 9/9
(Argmax der 2-Token-Verteilung je Zelle = w); Invariante r2 >= 0,95;
kristallisiert: 96/96 frische eval-Zellen SEQUENZ-exakt = 1,000.

    OMP_NUM_THREADS=6 python3 w16_twostep_organ.py
"""
from __future__ import annotations

import json
import math
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from w3_komposition import O1, SEED, W  # noqa: E402
from length_extrap_v2 import build_vocab, load_wikitext2  # noqa: E402
from streaming_train import StreamingNoPELM  # noqa: E402

NPZ = "donor_data/donor_targets_twostep_pos_27b.npz"
DOSIS = 10.0
ZIFFERN = W[:10]                        # zero..nine


class TwostepOrgan(nn.Module):
    """Liest a aus 'W[a] plus two times two is' (Zahlengerade 1..9)."""

    def __init__(self, d_model=128, d_e=32):
        super().__init__()
        self.P = nn.Linear(d_model, d_e, bias=False)
        self.phi = nn.Sequential(nn.Linear(d_e, 32), nn.Tanh(),
                                 nn.Linear(32, 1))
        self.register_buffer("cs", torch.arange(1, 10).float())  # 1..9
        self.s0 = nn.Parameter(torch.tensor(1.0))
        self.q = nn.Parameter(torch.tensor(0.5))
        self.krist = None                       # (alpha, gamma)

    def forward(self, emb):
        # v je Token; Kristall-Snap
        v = self.phi(torch.tanh(self.P(emb))).squeeze(-1)       # [B,T]
        if self.krist is not None:
            a, g = self.krist
            v = torch.round((v - g) / a)
        # u = Wert des ersten Segments: Token 0 (W[a]) ist a
        u = v[:, 0]
        delta = self.s0 * (self.cs[None] * u[:, None]
                           - self.q * self.cs[None] ** 2)
        return delta, u


def main() -> None:
    t0 = time.time()
    torch.manual_seed(SEED)

    # ---- Bridge: 2-Token-Verteilung je Zelle a -> Antwort-Wert --------
    npz = np.load(NPZ, allow_pickle=True)
    lg = np.concatenate([np.asarray(npz["logits"], np.float32),
                         np.asarray(npz["logits_eval"], np.float32)])  # (n,2,10)
    ps = [str(p) for p in list(npz["prompts"]) + list(npz["prompts_eval"])]
    n = len(ps)
    assert lg.shape[0] == n and lg.shape[1] == 2
    probs = torch.softmax(torch.tensor(lg), -1)          # (n,2,10)
    a_of = []
    for p in ps:
        m = re.search(r"a=(\d+)", p.strip().split("\n")[-1])
        a_of.append(int(m.group(1)))
    zellen = sorted(set(a_of))
    assert zellen == list(range(1, 10)), zellen
    # P je Zelle: MITTEL der Zeilen-Produkte P(d1)·P(d2) — die volle
    # Antwort-Wert-Masse. (Mittel der Produkte, nicht Produkt der
    # Mittel — der statistisch korrekte Weg.)
    P = torch.zeros(9, 10, 10)
    cnt = torch.zeros(9)
    for i, a in enumerate(a_of):
        w = 2 * a + 4
        d1 = w // 10 if w >= 10 else 0       # einstellig: Pos0=Ziffer
        d2 = w % 10
        if w < 10:
            P[a - 1, 0, w] += probs[i, 0, w]          # einstellig: Pos0
        else:
            P[a - 1, d1, d2] += probs[i, 0, d1] * probs[i, 1, d2]
        cnt[a - 1] += 1
    P = P / cnt[:, None, None].clamp_min(1)
    # Reinheit: Argmax der Antwort-Wert-Masse je Zelle
    rein = 0
    for a in range(1, 10):
        w = 2 * a + 4
        d1 = w // 10 if w >= 10 else 0
        d2 = w % 10
        # Masse der Zelle auf (d1,d2) muss max. sein
        is_max = bool(P[a - 1, d1, d2] == P[a - 1].max())
        rein += int(is_max)
    print(f"Bridge aus {n} Zeilen (2-Token) | Argmax-Reinheit {rein}/9",
          flush=True)

    # KL-Target: tgt[a, a'] = Donor-Masse, dass die ANTWORT von Zelle a
    # den Wert von a' hat (Masse auf (d1(a'),d2(a'))).
    tgt = torch.zeros(9, 9)
    for a in range(1, 10):
        for a2 in range(1, 10):
            w2 = 2 * a2 + 4
            if w2 < 10:
                tgt[a - 1, a2 - 1] = P[a - 1, 0, w2]
            else:
                tgt[a - 1, a2 - 1] = P[a - 1, w2 // 10, w2 % 10]
    tgt = (tgt ** 2) / (tgt ** 2).sum(1, keepdim=True).clamp_min(1e-8)

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

    def zell_ids(a):
        return torch.tensor([[stoi[W[a]], stoi["plus"], stoi["two"],
                              stoi["times"], stoi["two"], stoi["is"]]])

    alle = list(range(1, 10))
    held = set(random.Random(49).sample(alle, 3))     # 3 held-out Zellen
    train_a = [a for a in alle if a not in held]
    tr_ids = torch.cat([zell_ids(a) for a in train_a])
    tr_tgt = torch.stack([tgt[a - 1] for a in train_a])

    torch.manual_seed(SEED)
    organ = TwostepOrgan()
    opt = torch.optim.AdamW(organ.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(SEED)
    for _ in range(3000):
        idx = torch.randperm(len(tr_ids), generator=g)[:32]
        with torch.no_grad():
            emb = host.embed(tr_ids[idx])
        delta, _ = organ(emb)
        logp = F.log_softmax(delta, 1)
        loss = -(tr_tgt[idx] * logp).sum(1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    def messen(tag, krist):
        ok_tr = ok_h = 0
        for a in alle:
            ids = zell_ids(a)
            with torch.no_grad():
                delta, u = organ(host.embed(ids))
            hit = int(delta[0].argmax()) + 1 == a
            if a in held:
                ok_h += hit
            else:
                ok_tr += hit
        r = {"train": round(ok_tr / len(train_a), 4),
             "held": round(ok_h / len(held), 4),
             "alle_9": round((ok_tr + ok_h) / 9, 4)}
        print(f"{tag}: {r}", flush=True)
        return r

    res = {"bruecke_reinheit": rein, "roh": messen("roh (Bridge-KL)", None)}

    with torch.no_grad():
        vs = torch.tensor([float(organ.phi(torch.tanh(
            organ.P(E[stoi[W[a]]])))) for a in range(1, 10)])
    ts = torch.arange(1, 10).float()
    alpha = float(((ts - ts.mean()) * (vs - vs.mean())).sum()
                  / ((ts - ts.mean()) ** 2).sum())
    gamma = float(vs.mean() - alpha * ts.mean())
    r2 = 1 - float(((vs - (alpha * ts + gamma)) ** 2).sum()
                   / ((vs - vs.mean()) ** 2).sum())
    res["invariante"] = {"alpha": round(alpha, 4), "gamma": round(gamma, 4),
                         "r2": round(r2, 5)}
    print(f"INVARIANTE (one..nine): {res['invariante']}", flush=True)
    organ.krist = (alpha, gamma)
    res["kristall"] = messen("kristall (a exakt)", (alpha, gamma))

    # Emission: Wert = 2a+4, divmod, Shell-Dosis auf Ziffern-Slice
    ziffer_ids = torch.tensor([stoi[z] for z in ZIFFERN])

    def emittiere(prompt_words, wert):
        stellen = [int(d) for d in str(wert)]
        kontext = [stoi[w] for w in prompt_words]
        emittiert = []
        for d in stellen:
            ids = torch.tensor([kontext])
            with torch.no_grad():
                logits, _ = host(ids, None)
            out = logits[0, -1].clone()
            delta = torch.full((10,), -DOSIS)
            delta[d] = DOSIS
            out[ziffer_ids] = out[ziffer_ids] + delta
            tok = int(out.argmax())
            emittiert.append(tok)
            kontext.append(tok)
        return emittiert

    rng2 = random.Random(SEED + 77)
    n_ok = n_ok_org = 0
    fehler = []
    for _ in range(96):
        a = rng2.randint(1, 9)
        wert = 2 * a + 4
        words = [W[a], "plus", "two", "times", "two", "is"]
        # Organ liest a
        ids = torch.tensor([[stoi[w] for w in words]])
        with torch.no_grad():
            _, u = organ(host.embed(ids))
        a_gelesen = int(round(float(u[0]))) if organ.krist is None else \
            int(torch.round((u[0] - gamma) / alpha))
        toks = emittiere(words, wert)
        soll = [stoi[ZIFFERN[int(d)]] for d in str(wert)]
        if toks == soll:
            n_ok += 1
        elif len(fehler) < 6:
            fehler.append(f"a={a}: emittiert {toks}, soll {soll}")
    acc = n_ok / 96
    print(f"Emission 96 Zellen SEQUENZ-exakt: {n_ok}/96 = {acc:.4f} | "
          f"Fehler: {fehler[:6]}", flush=True)
    res["emission"] = {"sequenz_exakt": round(acc, 4), "n": 96,
                       "fehler": fehler[:6], "dosis": DOSIS}

    json.dump({"results": res,
               "prognose": "Reinheit 9/9; r2 >= 0,95; Emission 96/96 "
                           "sequenz-exakt",
               "rezept": "Bridge 2-Token -> Organ liest a -> Shell "
                         "rechnet 2a+4 -> divmod-Emission (w15-Mechanik)",
               "runtime_s": round(time.time() - t0, 1)},
              open("w16_twostep_organ.json", "w"), indent=1)
    print(f"({time.time()-t0:.0f}s) -> w16_twostep_organ.json", flush=True)


if __name__ == "__main__":
    main()