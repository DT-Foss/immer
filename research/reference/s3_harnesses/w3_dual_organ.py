"""w3_dual_organ.py — W3-FOLGE: Donor-KL-Subtraktion + Dual-Kopf-Organ.

TEIL A (label-frei): SubOrgan-Rezept (w3_sub_komposition), trainiert NUR
mit KL gegen die 27B-Donor-Verteilung (subw-npz). Kein GT-Label irgendwo:
Balance-Gewichte aus dem donor-argmax, Schaerfung per Temperatur tau=2
(p^tau renormalisiert — reine Transformation der Donor-Verteilung).
ERGEBNIS: train 1,000 / held-out 1,000 auf 2 Seeds (tau=1: 0,806/0,750).

TEIL B (Dual-Kopf): EIN Organ, EIN v-Encoder (gelernte 1D-Zahlengerade),
ZWEI Koepfe — R_sub auf v(a)-v(b) ueber one..eight, R_add auf v(a)+v(b)
ueber four..ten. ERGEBNIS: sub 1,000/1,000 UND add 1,000/1,000 (2 Seeds),
NLL bitgleich — EINE Gerade traegt beide Operationen.

WEG zu Teil B (zwei belegte Negative, dann drei Bausteine):
  - Naives Joint-Training: v kollabiert zum Shortcut (one..three vs
    four..nine geklumpt), Gates nicht operator-selektiv, held 0,25.
  - Gelernte Cross-Task-Gate-Reg (lambda=3): erzeugt Selektivitaet,
    drueckt aber das eigene Gate mit zu (4-Gramme unterscheiden sich nur
    im Operator-Wort) -> sub bricht ein (train 0,71).
  Bausteine:
  1. HARTER Operator-Router im Organ (kein Wirts-Eingriff): fester
     Embedding-Match des t-2-Tokens gegen frozen embed('less')/('plus'),
     multiplikativ auf das jeweilige Gate. Cross-Interferenz strukturell
     null (Kandidatenraeume ueberlappen bei four..eight!).
  2. CURRICULUM: Phase 1 formt die Gerade mit sub allein (das geloeste
     Problem). Danach SELBST-KALIBRIERUNG des add-Kopfs: alpha/gamma per
     least squares aus der EIGENEN Geraden (kein Label), Ra-Init exakt
     realisierend r_c = s0*c/alpha, beta_c = -s0*c^2/2 - 2*gamma*s0*c/alpha.
     Phase 2 lernt nur Ra+ga auf frozen phi: add 0,917/0,50 — die
     sub-Gerade traegt add schon fast; Rest-Fehler ist die systematisch
     zu kleine two->three-Luecke (1,46 statt ~2,2), die Summen-Klassen
     im engen 2..5-Fenster verschmiert ((2,4) vs (3,3) um 1,4 getrennt).
  3. Phase 3: gemeinsames Finetuning (lr 1e-3) — der add-Gradient zieht
     die Luecke auf, der sub-KL ankert die Gerade -> beide 1,000.

MESSUNG: 2 Seeds je Teil, Placebo (zeilen-permutierte Donor-Verteilungen,
frisches Organ, gleiches 3-Phasen-Rezept), WT2-NLL, unmount-Regression,
Gate-Selektivitaet, Zahlengeraden-Vergleich sub-only vs dual.

    OMP_NUM_THREADS=6 python3 w3_dual_organ.py
"""
from __future__ import annotations

import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from w3_komposition import O1, W, OrganShell, lm_nll  # noqa: E402
from length_extrap_v2 import build_vocab, load_wikitext2, tokenize  # noqa: E402
from streaming_train import StreamingNoPELM  # noqa: E402

HELD_SEED = 49
TAU = 2.0
S0 = 0.5
SUB_NPZ = "/tmp/27b/subw/donor_targets_subw_27b.npz"
ADD_NPZ = "/tmp/27b/addw10/donor_targets_addw10_27b.npz"


def load_donor(path):
    """npz -> {(a,b): gemittelte Donor-Verteilung}, Kandidaten-Woerter."""
    npz = np.load(path, allow_pickle=True)

    def rows_of(pk, lk):
        pares = []
        lg = torch.tensor(np.asarray(npz[lk][:, 0, :], dtype=np.float32))
        for p in npz[pk]:
            q = str(p).strip().split("\n")[-1]
            ws = re.findall(r"[a-zA-Z]+", q.lower())
            pares.append((W.index(ws[0]), W.index(ws[2])))
        return pares, torch.softmax(lg, 1)

    tp, tprob = rows_of("prompts", "logits")
    ep, eprob = rows_of("prompts_eval", "logits_eval")
    pp = {}
    for pr, pb in zip(tp + ep, torch.cat([tprob, eprob])):
        pp.setdefault(pr, []).append(pb)
    return ({k: torch.stack(v).mean(0) for k, v in pp.items()},
            [str(c) for c in npz["cand"]])


def sharpen_bal(probs_dict, prs, tau=TAU):
    """Temperatur-Schaerfung + label-freie Balance (Klassen = donor-argmax)."""
    raw = torch.stack([probs_dict[p] for p in prs])
    pr = raw ** tau
    pr = pr / pr.sum(1, keepdim=True)
    d = pr.argmax(1).tolist()
    cnt = Counter(d)
    w = torch.tensor([1.0 / cnt[i] for i in d])
    return pr, w / w.sum() * len(d)


class SubOrgan(nn.Module):
    """1D-Werte-Encoder + antisymmetrischer Kopf + Gate (Original der
    geloesten w3-sub-Abgabe, hier inline: w3_sub_komposition.py wurde
    von einem Parallel-Strang mit ArithmetikOrgan ueberschrieben)."""

    def __init__(self, d_model=128, d_g=16, n_cand=8):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d_model, 64), nn.Tanh(),
                                 nn.Linear(64, 1))
        self.R = nn.Linear(1, n_cand, bias=True)
        self.Pg = nn.Linear(d_model, d_g, bias=False)
        self.gate = nn.Sequential(nn.Linear(4 * d_g, 32), nn.Tanh(),
                                  nn.Linear(32, 1))
        nn.init.constant_(self.gate[-1].bias, -2.0)
        with torch.no_grad():
            self.R.weight.copy_(torch.tensor(
                [[S0 * c] for c in range(1, n_cand + 1)]))
            self.R.bias.copy_(torch.tensor(
                [-S0 * c * c / 2 for c in range(1, n_cand + 1)]))

    def forward(self, emb):
        v = self.phi(emb)
        z1 = torch.zeros_like(v[:, :1])
        v1 = torch.cat([z1, v[:, :-1]], dim=1)
        v3 = torch.cat([z1, z1, z1, v[:, :-3]], dim=1)
        delta = self.R(v3 - v1)
        e = self.Pg(emb)
        z = torch.zeros_like(e[:, :1])
        e1 = torch.cat([z, e[:, :-1]], dim=1)
        e2 = torch.cat([z, e1[:, :-1]], dim=1)
        e3 = torch.cat([z, e2[:, :-1]], dim=1)
        g = torch.sigmoid(self.gate(torch.cat([e3, e2, e1, e], dim=-1)))
        return delta, g


class DualOrgan(nn.Module):
    """EIN v-Encoder, zwei Koepfe, harter Operator-Router (frozen Match)."""

    def __init__(self, e_less, e_plus, d_model=128, d_g=16):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d_model, 64), nn.Tanh(),
                                 nn.Linear(64, 1))
        self.Rs = nn.Linear(1, 8)
        self.Ra = nn.Linear(1, 7)
        with torch.no_grad():  # sub-Kopf: realisierender Init (w3_sub)
            self.Rs.weight.copy_(torch.tensor([[S0 * c] for c in range(1, 9)]))
            self.Rs.bias.copy_(torch.tensor(
                [-S0 * c * c / 2 for c in range(1, 9)]))
        self.Pg = nn.Linear(d_model, d_g, bias=False)

        def gmlp():
            m = nn.Sequential(nn.Linear(4 * d_g, 32), nn.Tanh(),
                              nn.Linear(32, 1))
            nn.init.constant_(m[-1].bias, -2.0)
            return m

        self.gs, self.ga = gmlp(), gmlp()
        self.register_buffer("e_less", e_less / e_less.norm())
        self.register_buffer("e_plus", e_plus / e_plus.norm())

    def _route(self, emb, ref):
        en = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        r = ((en * ref).sum(-1, keepdim=True) > 0.999).float()  # Token-Match
        z = torch.zeros_like(r[:, :1])
        return torch.cat([z, z, r[:, :-2]], 1)                  # t-2 -> t

    def forward(self, emb):
        v = self.phi(emb)
        z1 = torch.zeros_like(v[:, :1])
        v1 = torch.cat([z1, v[:, :-1]], 1)
        v3 = torch.cat([z1, z1, z1, v[:, :-3]], 1)
        ds, da = self.Rs(v3 - v1), self.Ra(v3 + v1)
        e = self.Pg(emb)
        z = torch.zeros_like(e[:, :1])
        e1 = torch.cat([z, e[:, :-1]], 1)
        e2 = torch.cat([z, e1[:, :-1]], 1)
        e3 = torch.cat([z, e2[:, :-1]], 1)
        ctx = torch.cat([e3, e2, e1, e], -1)
        gs = torch.sigmoid(self.gs(ctx)) * self._route(emb, self.e_less)
        ga = torch.sigmoid(self.ga(ctx)) * self._route(emb, self.e_plus)
        return ds, gs, da, ga


class DualShell(nn.Module):
    """Steckbare Huelle: Wirt frozen, zwei gegatete Logit-Deltas."""

    def __init__(self, host, organ, scid, acid):
        super().__init__()
        self.host, self.organ = host, organ
        self.register_buffer("sc", torch.tensor(scid))
        self.register_buffer("ac", torch.tensor(acid))

    def forward(self, x, states=None):
        with torch.no_grad():
            logits, st = self.host(x, states)
            emb = self.host.embed(x)
        ds, gs, da, ga = self.organ(emb)
        out = logits.clone()
        out[..., self.sc] = out[..., self.sc] + gs * ds
        out[..., self.ac] = out[..., self.ac] + ga * da
        return out, st


def main() -> None:
    t0 = time.time()
    print("lade WT-2 + Wesen …", flush=True)
    train_text, val_text = load_wikitext2()
    vocab, stoi, unk, mask = build_vocab(train_text)
    val_ids = tokenize(val_text, stoi, unk)
    wt2_ids = torch.tensor(tokenize(train_text, stoi, unk)[:200000])
    ck = torch.load(O1 / "results/pos_ckpt.pt", map_location="cpu",
                    weights_only=False)
    host = StreamingNoPELM(len(vocab), mask, d_model=128, n_layers=2,
                           n_heads=4, d_head=32, seq_len=64, dropout=0.0,
                           causal=True)
    host.load_state_dict(ck["arms"]["A1"]["model"])
    host.eval()
    for p in host.parameters():
        p.requires_grad_(False)

    sub_probs, sub_cw = load_donor(SUB_NPZ)
    add_probs, add_cw = load_donor(ADD_NPZ)
    sub_cids = [stoi[w] for w in sub_cw]
    add_cids = [stoi[w] for w in add_cw]
    sub_all, add_all = sorted(sub_probs), sorted(add_probs)
    sub_held = sorted(random.Random(HELD_SEED).sample(sub_all, 4))
    add_held = sorted(random.Random(HELD_SEED).sample(add_all, 4))
    sub_tr = [p for p in sub_all if p not in sub_held]
    add_tr = [p for p in add_all if p not in add_held]
    print(f"sub: {len(sub_tr)} train, held {sub_held} | "
          f"add: {len(add_tr)} train, held {add_held}", flush=True)

    def mk(ps, op):
        return torch.tensor([[stoi[W[a]], stoi[op], stoi[W[b]], stoi["is"]]
                             for a, b in ps])

    sub_ids, sub_hids = mk(sub_tr, "less"), mk(sub_held, "less")
    add_ids, add_hids = mk(add_tr, "plus"), mk(add_held, "plus")
    tgt = {
        "sub_tr": torch.tensor([stoi[W[a - b]] for a, b in sub_tr]),
        "sub_he": torch.tensor([stoi[W[a - b]] for a, b in sub_held]),
        "add_tr": torch.tensor([stoi[W[a + b]] for a, b in add_tr]),
        "add_he": torch.tensor([stoi[W[a + b]] for a, b in add_held]),
    }
    subP, subW_ = sharpen_bal(sub_probs, sub_tr)
    addP, addW_ = sharpen_bal(add_probs, add_tr)
    base_nll = lm_nll(host, val_ids)

    def accs(shell):
        r = {}
        for tag, ids, cid in (("sub_tr", sub_ids, sub_cids),
                              ("sub_he", sub_hids, sub_cids),
                              ("add_tr", add_ids, add_cids),
                              ("add_he", add_hids, add_cids)):
            with torch.no_grad():
                lg, _ = shell(ids, None)
            pred = torch.tensor(cid)[lg[:, -1, cid].argmax(1)]
            r[tag] = round(float((pred == tgt[tag]).float().mean()), 4)
        return r

    def kl_loss(shell, ids, cid, P, Wb):
        lg, _ = shell(ids, None)
        lp = F.log_softmax(lg[:, -1, torch.tensor(cid)], 1)
        return ((-(P * lp).sum(1)) * Wb).mean()

    # ── TEIL A: SubOrgan mit reinem Donor-KL (label-frei) ──
    print("\n== TEIL A: SubOrgan, Donor-KL tau=2 ==", flush=True)
    runs_a = []
    for seed in (7, 8):
        torch.manual_seed(seed)
        organ = SubOrgan()
        shell = OrganShell(host, organ, sub_cids, dose=1.0)
        opt = torch.optim.AdamW(organ.parameters(), lr=3e-3)
        g = torch.Generator().manual_seed(seed)
        for step in range(3600):
            lt = kl_loss(shell, sub_ids, sub_cids, subP, subW_)
            starts = torch.randint(0, len(wt2_ids) - 33, (8,), generator=g)
            chunks = torch.stack([wt2_ids[s:s + 32] for s in starts])
            with torch.no_grad():
                emb = host.embed(chunks)
            _, gtx = organ(emb)
            loss = lt + gtx.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            lg, _ = shell(sub_ids, None)
            p_t = torch.tensor(sub_cids)[lg[:, -1, sub_cids].argmax(1)]
            lg, _ = shell(sub_hids, None)
            p_h = torch.tensor(sub_cids)[lg[:, -1, sub_cids].argmax(1)]
        tr = float((p_t == tgt["sub_tr"]).float().mean())
        he = float((p_h == tgt["sub_he"]).float().mean())
        nll = lm_nll(shell, val_ids)
        print(f"  seed {seed}: train {tr:.3f} | HELD {he:.3f} | "
              f"NLL {nll:.4f}", flush=True)
        runs_a.append({"seed": seed, "train": round(tr, 4),
                       "held": round(he, 4), "nll": round(nll, 4)})

    # ── TEIL B: DualOrgan, 3-Phasen-Curriculum ──
    print("\n== TEIL B: DualOrgan (3 Phasen) ==", flush=True)
    e_less = host.embed.weight[stoi["less"]].detach().clone()
    e_plus = host.embed.weight[stoi["plus"]].detach().clone()

    def train_dual(seed, sP, sW, aP, aW):
        torch.manual_seed(seed)
        organ = DualOrgan(e_less, e_plus)
        shell = DualShell(host, organ, sub_cids, add_cids)
        g = torch.Generator().manual_seed(seed)

        def wt2_gates():
            starts = torch.randint(0, len(wt2_ids) - 33, (8,), generator=g)
            chunks = torch.stack([wt2_ids[s:s + 32] for s in starts])
            with torch.no_grad():
                emb = host.embed(chunks)
            _, gtx_s, _, gtx_a = organ(emb)
            return gtx_s.mean() + gtx_a.mean()

        # Phase 1: sub formt die Gerade
        p1 = (list(organ.phi.parameters()) + list(organ.Rs.parameters())
              + list(organ.Pg.parameters()) + list(organ.gs.parameters()))
        opt = torch.optim.AdamW(p1, lr=3e-3)
        for step in range(3600):
            loss = kl_loss(shell, sub_ids, sub_cids, sP, sW) + wt2_gates()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        # Selbst-Kalibrierung des add-Kopfs aus der EIGENEN Geraden
        with torch.no_grad():
            vs = torch.tensor([float(organ.phi(host.embed.weight[stoi[W[n]]]))
                               for n in range(1, 10)])
            ts = torch.arange(1, 10).float()
            alpha = float(((ts - ts.mean()) * (vs - vs.mean())).sum()
                          / ((ts - ts.mean()) ** 2).sum())
            gamma = float(vs.mean() - alpha * ts.mean())
            organ.Ra.weight.copy_(torch.tensor(
                [[S0 * c / alpha] for c in range(4, 11)]))
            organ.Ra.bias.copy_(torch.tensor(
                [-S0 * c * c / 2 - 2 * gamma * S0 * c / alpha
                 for c in range(4, 11)]))
        # Phase 2: nur add-Kopf + add-Gate (phi frozen)
        opt2 = torch.optim.AdamW(list(organ.Ra.parameters())
                                 + list(organ.ga.parameters()), lr=3e-3)
        for step in range(2400):
            loss = kl_loss(shell, add_ids, add_cids, aP, aW) + wt2_gates()
            opt2.zero_grad(set_to_none=True)
            loss.backward()
            opt2.step()
        # Phase 3: gemeinsames Finetuning
        opt3 = torch.optim.AdamW(organ.parameters(), lr=1e-3)
        for step in range(1200):
            loss = (kl_loss(shell, sub_ids, sub_cids, sP, sW)
                    + kl_loss(shell, add_ids, add_cids, aP, aW)
                    + wt2_gates())
            opt3.zero_grad(set_to_none=True)
            loss.backward()
            opt3.step()
        return organ, shell

    runs_b, best = [], None
    for seed in (7, 8):
        organ, shell = train_dual(seed, subP, subW_, addP, addW_)
        r = accs(shell)
        nll = lm_nll(shell, val_ids)
        with torch.no_grad():
            line = [round(float(organ.phi(host.embed.weight[stoi[W[n]]])), 2)
                    for n in range(1, 10)]
            embS, embA = host.embed(sub_ids), host.embed(add_ids)
            _, gsS, _, gaS = organ(embS)
            _, gsA, _, gaA = organ(embA)
            gates = {"gs_sub": round(float(gsS[:, -1].mean()), 3),
                     "ga_sub": round(float(gaS[:, -1].mean()), 3),
                     "gs_add": round(float(gsA[:, -1].mean()), 3),
                     "ga_add": round(float(gaA[:, -1].mean()), 3)}
        print(f"  seed {seed}: {r} | NLL {nll:.4f} | gates {gates}",
              flush=True)
        print(f"    v(one..nine): {line}", flush=True)
        runs_b.append({"seed": seed, **r, "nll": round(nll, 4),
                       "gates": gates, "zahlengerade": line})
        if best is None or r["sub_he"] + r["add_he"] > best[1]:
            best = (seed, r["sub_he"] + r["add_he"])

    # ── Placebo: zeilen-permutierte Donor-Verteilungen, volles Rezept ──
    print("\n== Placebo (permutierte Donor-Zeilen, 3-Phasen-Rezept) ==",
          flush=True)
    prng = random.Random(HELD_SEED + 11)
    perm_s = list(range(len(subP)))
    perm_a = list(range(len(addP)))
    prng.shuffle(perm_s)
    prng.shuffle(perm_a)
    _, shell_p = train_dual(best[0], subP[perm_s], subW_[perm_s],
                            addP[perm_a], addW_[perm_a])
    r_p = accs(shell_p)
    print(f"  placebo (gegen wahre Ziele): {r_p}", flush=True)

    # ── Sentinels ──
    nll_un = lm_nll(host, val_ids)
    un_ok = abs(nll_un - base_nll) < 1e-9
    carry_ok = all(r["nll"] <= base_nll + 0.05 for r in runs_b)
    print(f"\nSENTINELS: carry {'OK' if carry_ok else 'FAIL'} | "
          f"unmount {'OK' if un_ok else 'FAIL'} "
          f"(NLL {nll_un:.4f} vs base {base_nll:.4f})", flush=True)

    res = {
        "teil_a_donor_kl": {"tau": TAU, "runs": runs_a,
                            "held_mean": round(sum(r["held"] for r in runs_a)
                                               / len(runs_a), 4),
                            "ziel_0.75_erreicht":
                                sum(r["held"] for r in runs_a) / len(runs_a)
                                >= 0.75},
        "teil_b_dual": {"runs": runs_b,
                        "sub_held_mean": round(sum(r["sub_he"] for r in runs_b)
                                               / len(runs_b), 4),
                        "add_held_mean": round(sum(r["add_he"] for r in runs_b)
                                               / len(runs_b), 4),
                        "ziel_0.6_je_aufgabe":
                            all(r["sub_he"] >= 0.6 and r["add_he"] >= 0.6
                                for r in runs_b)},
        "organ_params": sum(p.numel() for p in
                            DualOrgan(e_less, e_plus).parameters()),
        "sub_held": [f"{a}-{b}" for a, b in sub_held],
        "add_held": [f"{a}+{b}" for a, b in add_held],
        "base_nll": round(base_nll, 4),
        "placebo": r_p,
        "carry_ok": carry_ok,
        "unmount_ok": un_ok,
        "runtime_s": round(time.time() - t0, 1),
    }
    out = Path(__file__).parent / "w3_dual_organ.json"
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
