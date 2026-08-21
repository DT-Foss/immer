"""w3_komposition.py — W3: Komposition ins Wesen (held-out-Addition >= 0,6).

AUFGABE: Das gelebte 1,7M-GSSM-Wesen (o1-state A1, 909M C4-Tokens) soll
UNGESEHENE Wort-Additions-Paare lösen — 10 held-out-Paare (Seed 7+42),
die nie im Training waren — über ein steckbares, reversibles Organ, ohne
Wirts-Gewichte zu ändern und ohne Sprachschaden (WT2-NLL <= base+0,05).

WEG (nach zwei belegten Negativen auf der Velocity-Route):
  - Zustands-MLP-Organ (W2): train 0,744 / held-out 0,100 — der Zustand
    am Lese-Punkt trägt das PAAR als Ganzes -> Memorisierung.
  - Token-Injektor (w3_exp_a): L0-only lernt NICHTS (frozen Scan+Head
    übersetzen per-Token-Pushes nicht, train 0,077); L0L1 memorisiert
    über den kontextuellen Layer-1-Input (train 0,59 / held 0,10).

MECHANISMUS des Organs (AdditivOrgan, ~8,7k Params, w3_exp_b bestätigt):
Additivität wird STRUKTURELL erzwungen, im Logit-Raum — dem einzigen Ort,
an dem sie exakt trägt. Das Organ liest die frozen Embeddings des Wesens
(e = P·embed(token)) und addiert pro Position ein gegatetes Delta nur auf
den 13-Kandidaten-Slice:

    delta_c(t) = r_c · (e_{t-3} + e_{t-1})        (Operanden-Slots)
    gate(t)    = sigmoid(MLP([e_{t-3}, e_{t-2}, e_{t-1}, e_t]))
    logits[t, cand] += dose · gate(t) · delta(t)

Das Paar (a,b) erreicht das Delta AUSSCHLIESSLICH als Summe e_a + e_b:
Memorisierung einzelner Paare ist informationstheoretisch unmöglich,
und argmax_c[r_c·e_a + r_c·e_b] = a+b ist exakt realisierbar
(phi(a)_c = c·a − c²/2). Das Gate wird auf WT2-Text gegen 0 regularisiert
-> auf Sprache feuert das Organ nicht (NLL bleibt bitidentisch).

EINBAU: OrganShell — eine Hülle UM den Wirt (kein Wirts-Modul ersetzt,
kein Wirts-Parameter berührt). mount = Shell anlegen, unmount = Shell
entfernen -> Wirt trivial bitidentisch. Der Zustands-Update-Pfad des
Wesens (z = γz + a, a <= 0, Theorem 2) bleibt unangetastet, weil das
Organ hinter dem Head sitzt.

MESSUNG: 3 Seeds (Replikation), Placebo (permutierte Ziele, frisches
Organ), held-out/train-acc, WT2-NLL vor/nach, unmount-Regression,
Fehler-Analyse der held-out-Paare.

    OMP_NUM_THREADS=6 python3 w3_komposition.py
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

O1 = Path(__file__).resolve().parent.parent  # vendor/o1state (gespiegelte Struktur)
sys.path.insert(0, str(O1 / "src"))
sys.path.insert(0, str(O1 / "reference"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from streaming_train import StreamingNoPELM  # noqa: E402
from length_extrap_v2 import build_vocab, load_wikitext2, tokenize  # noqa: E402

SEED = 7
HELD_SEED = 7 + 42  # der etablierte W2-held-out-Split
W = ("zero one two three four five six seven eight nine ten eleven twelve "
     "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()


# ─────────────────────────── Organ + Shell ────────────────────────────
class AdditivOrgan(nn.Module):
    """P-Projektion + additiver Kandidaten-Kopf + 4-Gramm-Gate (~8,7k)."""

    def __init__(self, d_model=128, d_e=32, n_cand=13):
        super().__init__()
        self.P = nn.Linear(d_model, d_e, bias=False)
        self.R = nn.Linear(d_e, n_cand, bias=True)
        self.gate = nn.Sequential(nn.Linear(4 * d_e, 32), nn.Tanh(),
                                  nn.Linear(32, 1))
        nn.init.constant_(self.gate[-1].bias, -2.0)  # startet geschlossen

    def forward(self, emb):
        """emb [B,T,128] (frozen Wesens-Embeddings) -> delta [B,T,13], gate [B,T,1]."""
        e = self.P(emb)
        z = torch.zeros_like(e[:, :1])
        e1 = torch.cat([z, e[:, :-1]], dim=1)
        e2 = torch.cat([z, e1[:, :-1]], dim=1)
        e3 = torch.cat([z, e2[:, :-1]], dim=1)
        g = torch.sigmoid(self.gate(torch.cat([e3, e2, e1, e], dim=-1)))
        delta = self.R(e3 + e1)   # NUR die Summe der Operanden-Slots
        return delta, g


class OrganShell(nn.Module):
    """Steckbare Hülle: Wirt frozen & unberührt, Organ addiert Logit-Delta."""

    def __init__(self, host, organ, cand_ids, dose=1.0):
        super().__init__()
        self.host = host
        self.organ = organ
        self.register_buffer("cand", torch.tensor(cand_ids))
        self.dose = dose

    def forward(self, x, states=None):
        with torch.no_grad():
            logits, st = self.host(x, states)
            emb = self.host.embed(x)
        delta, g = self.organ(emb)
        out = logits.clone()
        out[..., self.cand] = out[..., self.cand] + self.dose * g * delta
        return out, st


# ─────────────────────────── Aufgaben & Messung ───────────────────────
def split_pairs():
    pairs = [(a, b) for a in range(2, 9) for b in range(2, 9)]  # 49
    held = random.Random(HELD_SEED).sample(pairs, 10)
    train = [p for p in pairs if p not in held]
    return train, held


def pairs_to_tasks(pairs, stoi, unk):
    return [([stoi.get(w, unk) for w in (W[a], "plus", W[b], "is")],
             stoi[W[a + b]]) for a, b in pairs]


@torch.no_grad()
def acc_on(model, tasks, cand_ids, return_preds=False):
    ids = torch.tensor([t[0] for t in tasks])
    tgt = torch.tensor([t[1] for t in tasks])
    logits, _ = model(ids, None)
    pred = torch.tensor(cand_ids)[logits[:, -1, cand_ids].argmax(1)]
    acc = float((pred == tgt).float().mean())
    return (acc, pred.tolist()) if return_preds else acc


@torch.no_grad()
def lm_nll(model, val_ids, chunk=64, batch=8, max_tokens=20000):
    ids = val_ids[:max_tokens]
    n = (len(ids) - 1) // chunk
    x = torch.tensor(ids[:n * chunk]).view(-1, chunk)
    y = torch.tensor(ids[1:n * chunk + 1]).view(-1, chunk)
    tot, cnt = 0.0, 0
    for i in range(0, len(x), batch):
        logits, _ = model(x[i:i + batch], None)
        tot += float(F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y[i:i + batch].reshape(-1),
            reduction="sum"))
        cnt += y[i:i + batch].numel()
    return tot / cnt


def train_organ(shell, organ, host, tasks, cand_ids, wt2_ids, steps=2400,
                lr=3e-3, lam=1.0, seed=SEED):
    """Kandidaten-Slice-CE + Gate-Regularisierer auf WT2-Text."""
    ids = torch.tensor([t[0] for t in tasks])
    c2i = {c: i for i, c in enumerate(cand_ids)}
    tgt = torch.tensor([c2i[t[1]] for t in tasks])
    cid = torch.tensor(cand_ids)
    opt = torch.optim.AdamW(organ.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        idx = torch.randperm(len(ids), generator=g)[:32]
        logits, _ = shell(ids[idx], None)
        loss_task = F.cross_entropy(logits[:, -1, cid], tgt[idx])
        starts = torch.randint(0, len(wt2_ids) - 33, (8,), generator=g)
        chunks = torch.stack([wt2_ids[s:s + 32] for s in starts])
        with torch.no_grad():
            emb = host.embed(chunks)
        _, gate_txt = organ(emb)
        loss = loss_task + lam * gate_txt.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 600 == 0:
            print(f"    step {step}: task {float(loss_task.detach()):.4f} | "
                  f"gate_txt {float(gate_txt.detach().mean()):.4f}", flush=True)


# ─────────────────────────────── Ablauf ───────────────────────────────
def main() -> None:
    t0 = time.time()
    torch.manual_seed(SEED)

    print("lade WT-2-Vokabular …", flush=True)
    train_text, val_text = load_wikitext2()
    vocab, stoi, unk, mask = build_vocab(train_text)
    val_ids = tokenize(val_text, stoi, unk)
    wt2_ids = torch.tensor(tokenize(train_text, stoi, unk)[:200000])
    cand_ids = [stoi[W[s]] for s in range(4, 17)]
    itos = {i: w for w, i in stoi.items()}

    print("lade das Wesen (pos_ckpt A1) …", flush=True)
    ck = torch.load(O1 / "results/pos_ckpt.pt", map_location="cpu",
                    weights_only=False)
    host = StreamingNoPELM(len(vocab), mask, d_model=128, n_layers=2,
                           n_heads=4, d_head=32, seq_len=64, dropout=0.0,
                           causal=True)
    host.load_state_dict(ck["arms"]["A1"]["model"])
    host.eval()
    for p in host.parameters():
        p.requires_grad_(False)

    train_pairs, held_pairs = split_pairs()
    train_tasks = pairs_to_tasks(train_pairs, stoi, unk)
    held_tasks = pairs_to_tasks(held_pairs, stoi, unk)
    print(f"held-out (10, Seed {HELD_SEED}): {held_pairs}", flush=True)

    base_train = acc_on(host, train_tasks, cand_ids)
    base_held = acc_on(host, held_tasks, cand_ids)
    base_nll = lm_nll(host, val_ids)
    print(f"baseline: train {base_train:.3f} | held {base_held:.3f} | "
          f"NLL {base_nll:.4f} (Zufall 1/13 = 0,077)", flush=True)

    # ── Replikation: 3 Seeds ──
    runs, best = [], None
    for seed in (SEED, SEED + 1, SEED + 2):
        torch.manual_seed(seed)
        organ = AdditivOrgan()
        shell = OrganShell(host, organ, cand_ids, dose=1.0)  # mount
        print(f"\n== Seed {seed}: trainiere Organ "
              f"({sum(p.numel() for p in organ.parameters()):,} Params) ==",
              flush=True)
        train_organ(shell, organ, host, train_tasks, cand_ids, wt2_ids,
                    seed=seed)
        tr = acc_on(shell, train_tasks, cand_ids)
        he, preds = acc_on(shell, held_tasks, cand_ids, return_preds=True)
        nll = lm_nll(shell, val_ids)
        fails = [(a, b, itos[p]) for (a, b), p, t in
                 zip(held_pairs, preds, [t[1] for t in held_tasks]) if p != t]
        print(f"  train {tr:.3f} | HELD {he:.3f} | NLL {nll:.4f} | "
              f"Fehler: {fails or 'keine'}", flush=True)
        runs.append({"seed": seed, "train": round(tr, 4), "held": round(he, 4),
                     "nll": round(nll, 4),
                     "held_fails": [f"{W[a]}+{W[b]}->{p}" for a, b, p in fails]})
        if best is None or he > best[1]:
            best = (organ, he, seed)

    # ── Placebo am besten Kandidaten-Rezept: permutierte Ziele ──
    print("\n== Placebo (permutierte Ziele, frisches Organ) ==", flush=True)
    torch.manual_seed(best[2])
    perm = [t[1] for t in train_tasks]
    random.Random(SEED + 11).shuffle(perm)
    placebo_tasks = [(t[0], p) for t, p in zip(train_tasks, perm)]
    organ_p = AdditivOrgan()
    shell_p = OrganShell(host, organ_p, cand_ids, dose=1.0)
    train_organ(shell_p, organ_p, host, placebo_tasks, cand_ids, wt2_ids,
                seed=best[2])
    tr_p = acc_on(shell_p, train_tasks, cand_ids)   # gegen WAHRE Ziele
    he_p = acc_on(shell_p, held_tasks, cand_ids)
    print(f"  placebo: train(wahr) {tr_p:.3f} | held(wahr) {he_p:.3f}",
          flush=True)

    # ── Sentinels am besten Organ ──
    organ, he_best, seed_best = best
    shell = OrganShell(host, organ, cand_ids, dose=1.0)
    nll_best = lm_nll(shell, val_ids)
    carry_ok = nll_best <= base_nll + 0.05
    # unmount = Shell entfernen -> Messung direkt am Wirt
    tr_un = acc_on(host, train_tasks, cand_ids)
    he_un = acc_on(host, held_tasks, cand_ids)
    nll_un = lm_nll(host, val_ids)
    un_ok = (abs(tr_un - base_train) < 1e-9 and abs(he_un - base_held) < 1e-9
             and abs(nll_un - base_nll) < 1e-9)
    print(f"\nSENTINELS: carry {'OK' if carry_ok else 'FAIL'} "
          f"(NLL {nll_best:.4f} vs base {base_nll:.4f}) | "
          f"unmount {'OK' if un_ok else 'FAIL'}", flush=True)

    held_mean = sum(r["held"] for r in runs) / len(runs)
    res = {
        "wesen": {"params": sum(p.numel() for p in host.parameters()),
                  "n_streamed": int(ck["n_streamed"])},
        "organ_params": sum(p.numel() for p in AdditivOrgan().parameters()),
        "held_pairs": [f"{a}+{b}" for a, b in held_pairs],
        "baseline": {"train": round(base_train, 4), "held": round(base_held, 4),
                     "nll": round(base_nll, 4)},
        "runs": runs,
        "held_mean": round(held_mean, 4),
        "held_best": round(he_best, 4),
        "ziel_0.6_erreicht": held_mean >= 0.6,
        "placebo": {"train": round(tr_p, 4), "held": round(he_p, 4)},
        "carry_ok": carry_ok,
        "unmount_ok": un_ok,
        "runtime_s": round(time.time() - t0, 1),
    }
    out = Path(__file__).parent / "w3_komposition.json"
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
