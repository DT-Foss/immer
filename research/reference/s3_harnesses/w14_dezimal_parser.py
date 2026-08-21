"""w14_dezimal_parser.py — Das sixteen-Limit fällt: mehrstellige Zahlen
als Ziffernwort-Folgen, geparst durch KOMPONIERTE Leiter-Rekurrenzen.

Zwei Rekurrenz-Klassen der Leiter, verschachtelt:

    innen  (multiplikativ, w13-Klasse):  h <- 10·h + v(ziffernwort)
    außen  (additiv, w7-Klasse):         Operator/is beendet Segment:
                                         acc <- acc + sgn·h;  h <- 0

  "one two plus three is"  ->  h=12 | acc=12 | h=3 | acc=15  ->  fifteen

Das Wesen liest damit BELIEBIG große Zahlen aus seinem 10-Wörter-
Ziffernvokabular (zero..nine) — das Vokab-Limit (max sixteen) wird
strukturell gebrochen; die Antwort bleibt vorerst im Kandidatenraum
one..sixteen. v wird gelernt (GT-PoC, Kandidaten-Slice-CE) und
Ganzzahl-kristallisiert (Karte über die Träger zero..nine).

Kontrast: ordnungsblinde Variante (h <- h + v, keine Stellen-
gewichte) — kann "one two"=12 nicht von "two one"=21 trennen.

Training: einstellige Terme + WENIGE zweistellige Muster; Eval:
ungesehene zweistellige Kombinationen inkl. Stellentausch-Paare.

Prognose (festgelegt, 20.08. vor dem Lauf): kristallisiert 1,000 auf
ungesehenen zweistelligen Eingaben; ordnungsblinder Kontrast ~0 auf
Stellentausch-Verwechslern.

    OMP_NUM_THREADS=6 python3 w14_dezimal_parser.py
"""
from __future__ import annotations

import collections
import json
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from w3_komposition import O1, SEED, W  # noqa: E402
from length_extrap_v2 import build_vocab, load_wikitext2  # noqa: E402
from streaming_train import StreamingNoPELM  # noqa: E402

ZIFFERN = W[:10]                       # zero..nine
N_CAND = 16


class DezimalOrgan(nn.Module):
    """Parser (mult.) in Akkumulator (add.), Hart-Form-Readout."""

    def __init__(self, e_plus, e_less, e_is, stellen=True,
                 d_model=128, d_e=32):
        super().__init__()
        self.P = nn.Linear(d_model, d_e, bias=False)
        self.phi = nn.Sequential(nn.Linear(d_e, 32), nn.Tanh(),
                                 nn.Linear(32, 1))
        for e, nm in ((e_plus, "e_plus"), (e_less, "e_less"),
                      (e_is, "e_is")):
            self.register_buffer(nm, e / e.norm())
        self.stellen = stellen
        self.register_buffer("cs", torch.arange(1, N_CAND + 1).float())
        self.s0 = nn.Parameter(torch.tensor(1.0))
        self.q = nn.Parameter(torch.tensor(0.5))
        self.krist = None                       # (alpha, gamma)

    def forward(self, emb):
        B, T, _ = emb.shape
        e_n = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        is_plus = (e_n @ self.e_plus > 0.99)
        is_less = (e_n @ self.e_less > 0.99)
        is_lese = (e_n @ self.e_is > 0.99)
        is_end = is_plus | is_less | is_lese
        v = self.phi(torch.tanh(self.P(emb))).squeeze(-1)
        if self.krist is not None:
            a, g = self.krist
            v = torch.round((v - g) / a)
        acc = torch.zeros(B)
        h = torch.zeros(B)
        sgn = torch.ones(B)
        for t in range(T):
            end = is_end[:, t].float()
            acc = acc + end * sgn * h           # Segment schließen
            h = (1 - end) * (
                (10.0 * h if self.stellen else h) + v[:, t]) \
                + end * 0.0
            sgn = torch.where(is_plus[:, t], torch.ones(B), sgn)
            sgn = torch.where(is_less[:, t], -torch.ones(B), sgn)
        delta = self.s0 * (self.cs[None] * acc[:, None]
                           - self.q * self.cs[None] ** 2)
        return delta


def make_tasks(spec, rng, stoi):
    """spec: Liste von (a, b, op); Zahlen als Ziffernwort-Folgen."""
    out = []
    for a, b, op in spec:
        erg = a + b if op == "plus" else a - b
        if not (1 <= erg <= N_CAND):
            continue
        words = ([ZIFFERN[int(d)] for d in str(a)] + [op]
                 + [ZIFFERN[int(d)] for d in str(b)] + ["is"])
        out.append(([stoi[w] for w in words], erg))
    return out


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

    rng = random.Random(SEED)
    # Training: alle einstelligen Paare + zweistellige NUR mit a in
    # {10, 11, 12} (wenige Muster; 13, 14, 15 bleiben ungesehen)
    spec_tr = ([(a, b, op) for a in range(1, 10) for b in range(1, 10)
                for op in ("plus", "less")]
               + [(a, b, op) for a in (10, 11, 12) for b in range(1, 5)
                  for op in ("plus", "less")])
    train = make_tasks(spec_tr, rng, stoi)
    # Eval: ungesehene zweistellige (13, 14, 15) + Stellentausch-Paare
    spec_ev = [(a, b, op) for a in (13, 14, 15) for b in range(1, 4)
               for op in ("plus", "less")]
    ev_neu = make_tasks(spec_ev, rng, stoi)
    ev_tausch = make_tasks([(12, 3, "less"), (13, 2, "less"),
                            (14, 5, "less"), (15, 4, "less"),
                            (12, 4, "plus"), (13, 1, "plus")], rng, stoi) \
        + make_tasks([(21 - 16, 1, "plus")], rng, stoi)
    # echte Stellentausch-Diskriminierung: "one two" (12) vs "two one"
    # (21>16, faellt raus) -> nutze 12/21-Analoga im Rahmen: 12 vs 21
    # geht nicht (21>16); nimm 13-2 vs 31-... -> stattdessen direkte
    # Parser-Probe unten.
    print(f"train {len(train)} | eval neu {len(ev_neu)}", flush=True)

    def messen(organ, tasks):
        by_l = collections.defaultdict(list)
        for t in tasks:
            by_l[len(t[0])].append(t)
        accs = []
        for _, grp in by_l.items():
            ids = torch.tensor([t[0] for t in grp])
            tgt = torch.tensor([t[1] - 1 for t in grp])
            with torch.no_grad():
                delta = organ(host.embed(ids))
            accs += (delta.argmax(1) == tgt).tolist()
        return round(sum(accs) / len(accs), 4)

    res = {}
    organe = {}
    for name, stellen in (("parser", True), ("ordnungsblind", False)):
        torch.manual_seed(SEED)
        organ = DezimalOrgan(E[stoi["plus"]], E[stoi["less"]],
                             E[stoi["is"]], stellen=stellen)
        opt = torch.optim.AdamW(organ.parameters(), lr=3e-3)
        g = torch.Generator().manual_seed(SEED)
        by_l = collections.defaultdict(list)
        for i, t in enumerate(train):
            by_l[len(t[0])].append(i)
        keys = list(by_l)
        tgt_all = torch.tensor([t[1] - 1 for t in train])
        cnt = collections.Counter(tgt_all.tolist())
        wts = torch.tensor([1.0 / cnt[int(t)] for t in tgt_all])
        for _ in range(3000):
            L = keys[int(torch.randint(len(keys), (1,), generator=g))]
            pool = by_l[L]
            idx = [pool[i] for i in
                   torch.randperm(len(pool), generator=g)[:32].tolist()]
            ids = torch.tensor([train[i][0] for i in idx])
            tgt = tgt_all[idx]
            with torch.no_grad():
                emb = host.embed(ids)
            li = F.cross_entropy(organ(emb), tgt, reduction="none")
            loss = (li * wts[idx]).sum() / wts[idx].sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        organe[name] = organ
        res[name] = {"train": messen(organ, train[:200]),
                     "zweistellig_neu_roh": messen(organ, ev_neu)}
        print(f"{name} roh: {res[name]}", flush=True)

    # Kristallisation des Parsers (Ziffern-Träger zero..nine)
    organ = organe["parser"]
    with torch.no_grad():
        vs = torch.tensor([float(organ.phi(torch.tanh(
            organ.P(E[stoi[z]])))) for z in ZIFFERN])
    ts = torch.arange(0, 10).float()
    alpha = float(((ts - ts.mean()) * (vs - vs.mean())).sum()
                  / ((ts - ts.mean()) ** 2).sum())
    gamma = float(vs.mean() - alpha * ts.mean())
    r2 = 1 - float(((vs - (alpha * ts + gamma)) ** 2).sum()
                   / ((vs - vs.mean()) ** 2).sum())
    res["invariante"] = {"alpha": round(alpha, 4), "gamma": round(gamma, 4),
                         "r2": round(r2, 5)}
    print(f"INVARIANTE (zero..nine): {res['invariante']}", flush=True)
    organ.krist = (alpha, gamma)
    res["parser_kristall"] = {"zweistellig_neu": messen(organ, ev_neu)}
    # Stellentausch-Probe direkt am Parser-Zustand: 12 vs 21
    def parse_wert(zs):
        ids = torch.tensor([[stoi[z] for z in zs] + [stoi["is"]]])
        emb = host.embed(ids)
        e_n = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        with torch.no_grad():
            v = torch.round((organ.phi(torch.tanh(organ.P(emb)))
                             .squeeze(-1) - gamma) / alpha)
        h = 0.0
        for t in range(len(zs)):
            h = 10.0 * h + float(v[0, t])
        return h
    tausch = {"one two": parse_wert(["one", "two"]),
              "two one": parse_wert(["two", "one"]),
              "nine zero": parse_wert(["nine", "zero"]),
              "one zero zero": parse_wert(["one", "zero", "zero"])}
    res["parser_probe"] = tausch
    print(f"PARSER-PROBE (kristallisiert): {tausch}", flush=True)

    json.dump({"results": res,
               "prognose": "parser kristall zweistellig_neu = 1,000; "
                           "ordnungsblind faellt; Probe 12/21/90/100 exakt",
               "runtime_s": round(time.time() - t0, 1)},
              open("w14_dezimal_parser.json", "w"), indent=1)
    print(f"({time.time()-t0:.0f}s) -> w14_dezimal_parser.json", flush=True)


if __name__ == "__main__":
    main()
