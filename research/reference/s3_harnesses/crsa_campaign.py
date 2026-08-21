"""crsa_campaign.py — DIE 2-MODI x 3-SEEDS x 5-ARME-KAMPAGNE (Suite v1.1)
der CRSA-Reihe (designierter erster Lauf der Session, 8.131).

Quelle der Spez: neue attention/NoPE-Induction-Diagniselauf.md (Davids
Sandbox-Transkript, 8.637 Zeilen):
  - legacy  = exakte Suite-v1-Replikation (target first, Distraktoren mit
    Kollisionen; unser crsa_suite_v1.py-Generator identisch).
  - hardened = NoPE-v1.1: UNIQUE Keys+Values, randomisierter Query-Rank
    (early/middle/late), Counterfactual-Eval (gleiche Tabelle, anderer
    Query-Key -> andere Value), Shift-Extrapolation 20-30 Distraktoren.
Programme in Davids Prereg-Reihenfolge (test_marked_recall_programs_...):
    F->F | Q->Q | Q->R | R->Q | R->R
    q = quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]  (2 Local + 1 Balanced + 1 Free)
    r = quad_route[sh=0,lh=3,bh=0,s=0.8]        (3 Local + 1 Free)
Jede Layer behaelt >= 1 unangetasteten Softmax-Head (free-head-invariant).
Paired: identische Init (torch.manual_seed) + identischer Batch-Stream
(seed 3000+seed) ueber alle 5 Arme eines Seeds — PS-Lifted-Confound-
Lektion; Init- und Batch-Stream-Digest je Lauf fuer das Pairing-Audit.

PROGNOSE (vorab, hart, Anti-Hedge — 20.08. nach der Spez-Lektuere):
  (a) hardened F->F repliziert 8.131 extern: in_dist und shift ALLE <= 0,06,
      bits nahe log2(32) = 5,0 (Zufall auf den 32 Werten).
  (b) in_dist_random: KEIN Arm >= 0,10 — die 8.130-Lernbarkeitsgrenze
      (NoPE, 2 Layer, d=48, 1200 Steps) bleibt in beiden Modi bestehen;
      Q->Q (2L+1B+1F beidseitig) ist der beste Kandidat, aber < 0,10.
  (c) counterfactual: both_correct ~ 0,00 fuer alle Arme; prediction_change
      ~ 0,5 (Zufall) — der Query-Wechsel aendert nichts, weil nichts gelernt.
  (d) shift_20_30 <= in_dist in jedem Arm (keine Extrapolation).
Falls (b) faellt und ein Arm in_dist >= 0,6 trifft: NEUES ERGEBNIS
(Balanced-Head traegt den previous-token-Kanal doch) -> ungeplantes
Folgeexperiment (Q-Routing allein ohne Free-Head / laengeres Training).

    OMP_NUM_THREADS=6 python3 crsa_campaign.py [--modes legacy hardened]
                                          [--steps 1200] [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from w3_komposition import O1  # noqa: E402  (setzt O1-Pfade, Pflicht)
sys.path.insert(0, "/Users/bhkmie/self-verification_fable/neue attention")
from operators import AttentionSpec, apply_attention  # noqa: E402

# ---- Generator-Material (identisch mit Davids synthetic.py) -------------
_KEYS = tuple(range(10, 42))        # 32 unique keys
_VALS = tuple(range(50, 82))        # 32 unique values
_KM, _VM, _SEP = 200, 201, 254      # Marker-Token
VOCAB = 256

CTX = 160
D, HEADS, LAYERS = 48, 4, 2
BATCH = 24
LR = 3e-3

# Programme in Davids Prereg-Reihenfolge.
_CRSA = AttentionSpec(kind="quad_route", diagonal_debit=3.0, self_heads=0,
                      local_heads=2, balanced_heads=1, slope=0.8)
_LOCAL = AttentionSpec(kind="quad_route", self_heads=0, local_heads=3,
                       balanced_heads=0, slope=0.8)
_SM = AttentionSpec(kind="softmax")
PROGRAMMES = {
    "F->F": (_SM, _SM),
    "Q->Q": (_CRSA, _CRSA),
    "Q->R": (_CRSA, _LOCAL),
    "R->Q": (_LOCAL, _CRSA),
    "R->R": (_LOCAL, _LOCAL),
}


def free_head_count(spec: AttentionSpec, heads: int = HEADS) -> int:
    return heads - spec.self_heads - spec.local_heads - spec.balanced_heads


# ---- Modell (ByteGPT-Aequivalent, NoPE faehig) ---------------------------
class Block(nn.Module):
    def __init__(self, spec: AttentionSpec):
        super().__init__()
        self.spec = spec
        self.ln1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.proj = nn.Linear(D, D, bias=False)
        self.ln2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(),
                                 nn.Linear(4 * D, D))

    def forward(self, x):
        B, T, _ = x.shape
        dh = D // HEADS
        q, k, v = self.qkv(self.ln1(x)).chunk(3, -1)
        q = q.view(B, T, HEADS, dh).transpose(1, 2)
        k = k.view(B, T, HEADS, dh).transpose(1, 2)
        v = v.view(B, T, HEADS, dh).transpose(1, 2)
        logits = q @ k.transpose(-2, -1) / math.sqrt(dh)
        w = apply_attention(logits, self.spec)
        y = (w @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(y)
        return x + self.mlp(self.ln2(x))


class MarkedLM(nn.Module):
    """NoPE 2-Layer-4-Head-ByteLM auf dem 256-Vokabular."""

    def __init__(self, specs):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D)
        self.blocks = nn.ModuleList([Block(s) for s in specs])
        self.ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, VOCAB, bias=False)

    def forward(self, ids):
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


class Batch:
    __slots__ = ("tokens", "targets", "target_rank", "distractor_count")

    def __init__(self, tokens, targets, target_rank, distractor_count):
        self.tokens = tokens
        self.targets = targets
        self.target_rank = target_rank
        self.distractor_count = distractor_count


def pack(rows, context=CTX):
    tokens = torch.zeros(len(rows), context, dtype=torch.long)
    for i, row in enumerate(rows):
        tokens[i, context - len(row):] = torch.tensor(row, dtype=torch.long)
    return tokens


def _randint(lo, hi, gen):
    return int(torch.randint(lo, hi + 1, (1,), generator=gen).item())


# --------------------- LEGACY-Generator (exakt v1) ------------------------
def legacy_batch(batch_size, nd_lo, nd_hi, gen):
    """Suite-v1-Replikation: target first, distractor keys als geklemmte
    Offsets — mit Kollisionen (der v1-Confound, den hardened entfernt)."""
    rows, targets, counts = [], [], []
    for _ in range(batch_size):
        key = _KEYS[_randint(0, len(_KEYS) - 1, gen)]
        value = _VALS[_randint(0, len(_VALS) - 1, gen)]
        row = [_KM, key, _VM, value]
        nd = _randint(nd_lo, nd_hi, gen)
        for _ in range(nd):
            off = _randint(1, 3, gen)
            dr = -1 if float(torch.rand(1, generator=gen).item()) < 0.5 else 1
            dk = min(max(key + dr * off, _KEYS[0]), _KEYS[-1])
            dv = _VALS[_randint(0, len(_VALS) - 1, gen)]
            row += [_KM, dk, _VM, dv]
        row += [_SEP, _KM, key]
        rows.append(row)
        targets.append(value)
        counts.append(nd)
    return Batch(pack(rows, CTX), torch.tensor(targets, dtype=torch.long),
                 torch.zeros(batch_size, dtype=torch.long),
                 torch.tensor(counts, dtype=torch.long))


# -------------------------------------------------------------------------
# HARDENED-Generator: collision-free Tabelle + randomisierter Query-Rank
# -------------------------------------------------------------------------
def _table(nd, gen):
    """nd+1 unique keys und nd+1 unique values (kollisionsfrei)."""
    table_len = nd + 1
    key_ids = torch.randperm(len(_KEYS), generator=gen)[:table_len]
    val_ids = torch.randperm(len(_VALS), generator=gen)[:table_len]
    keys = [_KEYS[i] for i in key_ids.tolist()]
    vals = [_VALS[i] for i in val_ids.tolist()]
    return keys, vals


def _place_target(nd, target_mode, gen):
    if target_mode == "random":
        return _randint(0, nd, gen)
    if target_mode == "first":
        return 0
    if target_mode == "middle":
        return nd // 2
    if target_mode == "last":
        return nd
    raise ValueError(target_mode)


def hardened_batch(batch_size, nd_lo, nd_hi, target_mode, gen):
    """Collision-free marked recall: unique keys+values je Tabelle, Ziel-
    Paar an random/first/middle/last-Rank. targets = Value des Query-Keys."""
    rows, targets, ranks, counts = [], [], [], []
    for _ in range(batch_size):
        nd = _randint(nd_lo, nd_hi, gen)
        keys, vals = _table(nd, gen)
        rank = _place_target(nd, target_mode, gen)
        row = []
        for k, v in zip(keys, vals):
            row += [_KM, k, _VM, v]
        row += [_SEP, _KM, keys[rank]]
        rows.append(row)
        targets.append(vals[rank])
        ranks.append(rank)
        counts.append(nd)
    return Batch(pack(rows, CTX), torch.tensor(targets, dtype=torch.long),
                 torch.tensor(ranks, dtype=torch.long),
                 torch.tensor(counts, dtype=torch.long))


def counterfactual_pair(batch_size, nd_lo, nd_hi, gen):
    """Zwei Batches mit IDENTISCHER Tabelle, nur der finale Query-Key
    differiert — gleiche Zeile bis auf die letzten 3 Token."""
    rows_a, rows_b, tg_a, tg_b = [], [], [], []
    for _ in range(batch_size):
        nd = _randint(nd_lo, nd_hi, gen)
        keys, vals = _table(nd, gen)
        qa = _randint(0, nd, gen)
        qb = _randint(0, nd - 1, gen)
        if qb >= qa:
            qb += 1
        table = []
        for k, v in zip(keys, vals):
            table += [_KM, k, _VM, v]
        rows_a.append(table + [_SEP, _KM, keys[qa]])
        rows_b.append(table + [_SEP, _KM, keys[qb]])
        tg_a.append(vals[qa])
        tg_b.append(vals[qb])
    return (Batch(pack(rows_a, CTX), torch.tensor(tg_a, dtype=torch.long),
                  torch.zeros(batch_size, dtype=torch.long),
                  torch.zeros(batch_size, dtype=torch.long)),
            Batch(pack(rows_b, CTX), torch.tensor(tg_b, dtype=torch.long),
                  torch.zeros(batch_size, dtype=torch.long),
                  torch.zeros(batch_size, dtype=torch.long)))


# ----------------------------------------------------------------------
def tensor_digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()
                          ).hexdigest()[:12]


def init_digest(model):
    h = hashlib.sha256()
    for name, t in model.state_dict().items():
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def batch_stream_digest(batches):
    h = hashlib.sha256()
    for b in batches:
        h.update(tensor_digest(b.tokens).encode())
        h.update(tensor_digest(b.targets).encode())
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------
def run_arm(mode, prog_name, specs, seed, steps):
    torch.manual_seed(seed)                 # identische Init je Seed
    model = MarkedLM(specs)
    ide = init_digest(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(3000 + seed)   # Batch-Stream
    stream_batches = []
    t0 = time.perf_counter()
    for step in range(steps):
        if mode == "legacy":
            b = legacy_batch(BATCH, 8, 14, gen)
        else:
            b = hardened_batch(BATCH, 8, 14, "random", gen)
        loss = F.cross_entropy(model(b.tokens)[:, -1], b.targets)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            stream_batches.append(b)
            print(f"  {prog_name} s{seed} {mode} step {step+1} "
                  f"loss {float(loss):.3f}", flush=True)
    stream_dg = batch_stream_digest(stream_batches)
    return model, ide, stream_dg, time.perf_counter() - t0


def eval_acc_bits(model, fn, gen, n_batches=8, n=48):
    model.eval()
    ok = tot = 0
    loss = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            b = fn(gen)
            logits = model(b.tokens)[:, -1]
            loss += float(F.cross_entropy(logits, b.targets, reduction="sum"))
            ok += int((logits.argmax(-1) == b.targets).sum())
            tot += len(b.targets)
    model.train()
    return round(ok / tot, 4), round(loss / tot / math.log(2), 4)


def eval_counterfactual(model, nd_lo=20, nd_hi=30):
    model.eval()
    ca = cb = both = chg = n = 0
    loss = 0.0
    gen = torch.Generator().manual_seed(10049)
    with torch.no_grad():
        for _ in range(8):
            a, b = counterfactual_pair(48, nd_lo, nd_hi, gen)
            la = model(a.tokens)[:, -1]
            lb = model(b.tokens)[:, -1]
            pa, pb = la.argmax(-1), lb.argmax(-1)
            oka, okb = pa == a.targets, pb == b.targets
            ca += int(oka.sum())
            cb += int(okb.sum())
            both += int((oka & okb).sum())
            chg += int((pa != pb).sum())
            n += len(a.targets)
            loss += float(F.cross_entropy(la, a.targets, reduction="sum"))
            loss += float(F.cross_entropy(lb, b.targets, reduction="sum"))
    model.train()
    return {"acc_first": round(ca / n, 4), "acc_second": round(cb / n, 4),
            "both_correct": round(both / n, 4),
            "pred_change": round(chg / n, 4),
            "bits": round(loss / (2 * n) / math.log(2), 4)}


def eval_ranked_shift(model, target_mode, seed):
    gen = torch.Generator().manual_seed(seed)
    acc, bits = eval_acc_bits(
        model, lambda g: hardened_batch(48, 20, 30, target_mode, g), gen)
    # Rank-Bins (early/middle/late) separat messen
    model.eval()
    r_ok = {"early": 0, "middle": 0, "late": 0}
    r_n = {"early": 0, "middle": 0, "late": 0}
    with torch.no_grad():
        for _ in range(8):
            b = hardened_batch(48, 20, 30, target_mode, gen)
            cor = model(b.tokens)[:, -1].argmax(-1) == b.targets
            frac = b.target_rank.to(torch.float32) / b.distractor_count.clamp_min(1)
            for label, mask in (("early", frac < 1 / 3),
                                ("middle", (frac >= 1 / 3) & (frac < 2 / 3)),
                                ("late", frac >= 2 / 3)):
                r_ok[label] += int((cor & mask).sum())
                r_n[label] += int(mask.sum())
    model.train()
    rank = {k: round(r_ok[k] / r_n[k], 4) if r_n[k] else None for k in r_ok}
    return acc, bits, rank


def run_mode(mode, seeds, steps):
    res = {}
    audits = {}
    for seed in seeds:
        for prog, specs in PROGRAMMES.items():
            model, ide, sdig, secs = run_arm(mode, prog, specs, seed, steps)
            audits[f"{mode}/{prog}/s{seed}"] = (ide, sdig)
            cond = {}
            if mode == "legacy":
                cond["in_dist"] = eval_acc_bits(
                    model, lambda g: legacy_batch(48, 8, 14, g),
                    torch.Generator().manual_seed(9999))[:2]
                cond["shift_20_30"] = eval_acc_bits(
                    model, lambda g: legacy_batch(48, 20, 30, g),
                    torch.Generator().manual_seed(10009))[:2]
            else:
                cond["in_dist_random"] = eval_acc_bits(
                    model, lambda g: hardened_batch(48, 8, 14, "random", g),
                    torch.Generator().manual_seed(9999))[:2]
                cond["shift_random"] = eval_acc_bits(
                    model, lambda g: hardened_batch(48, 20, 30, "random", g),
                    torch.Generator().manual_seed(10009))[:2]
                cond["shift_first"] = eval_ranked_shift(model, "first", 10019)
                cond["shift_middle"] = eval_ranked_shift(model, "middle", 10029)
                cond["shift_last"] = eval_ranked_shift(model, "last", 10039)
                cond["counterfactual"] = eval_counterfactual(model)
            res.setdefault(prog, {})[seed] = cond
            print(f"{mode} {prog} seed {seed} ({secs:.0f}s)", flush=True)
            del model
    return res, audits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="*", default=["legacy", "hardened"])
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="nur fehlende hardened-Arme; legacy aus Log")
    args = ap.parse_args()
    t0 = time.time()
    allres, audits_all = {}, {}
    seeds = (7,) if args.smoke else (7, 8, 9)
    if args.resume:
        # legacy aus dem Log parsen (bereits fertig gelaufen)
        log = open("crsa_campaign_run.log").read()
        import re as _re
        for mode in ("legacy", "hardened"):
            m = _re.search(r"=== " + mode + r" KOMPLETT ===\n(.*?)\n(?===|$)",
                           log, _re.S)
            if not m:
                continue
            txt = m.group(1)
            res = {}
            for prog in PROGRAMMES:
                mm = _re.search(prog + r": \{(.*?)\}\n", txt)
                if mm:
                    res[prog] = eval("{" + mm.group(1) + "}")
            if res:
                allres[mode] = res
        # hardened weiter: nur fehlende seeds/arme
        rest = {}
        for seed in seeds:
            for prog in PROGRAMMES:
                done = any(s == str(seed) for s in
                           allres.get("hardened", {}).get(prog, {}))
                if not done:
                    rest.setdefault(seed, []).append(prog)
        if rest:
            for seed, progs in rest.items():
                for prog in progs:
                    model, ide, sdig, secs = run_arm("hardened", prog,
                                                     PROGRAMMES[prog],
                                                     seed, args.steps)
                    audits_all.setdefault("hardened", {})
                    audits_all["hardened"][f"hardened/{prog}/s{seed}"] = \
                        (ide, sdig)
                    cond = {}
                    cond["in_dist_random"] = eval_acc_bits(
                        model, lambda g: hardened_batch(48, 8, 14, "random", g),
                        torch.Generator().manual_seed(9999))[:2]
                    cond["shift_random"] = eval_acc_bits(
                        model, lambda g: hardened_batch(48, 20, 30, "random", g),
                        torch.Generator().manual_seed(10009))[:2]
                    cond["shift_first"] = eval_ranked_shift(model, "first", 10019)
                    cond["shift_middle"] = eval_ranked_shift(model, "middle", 10029)
                    cond["shift_last"] = eval_ranked_shift(model, "last", 10039)
                    cond["counterfactual"] = eval_counterfactual(model)
                    allres.setdefault("hardened", {}).setdefault(prog, {})
                    allres["hardened"][prog][seed] = cond
                    print(f"hardened {prog} seed {seed} ({secs:.0f}s)",
                          flush=True)
                    del model
    else:
        for mode in args.modes:
            res, aud = run_mode(mode, seeds, args.steps)
            allres[mode] = res
            audits_all[mode] = aud
        print(f"=== {mode} KOMPLETT ===", flush=True)
        for prog, by_seed in res.items():
            print(f"  {prog}: {by_seed}", flush=True)
    out = {"modes": allres, "audits": audits_all, "steps": args.steps,
           "prognose": "8.131 bleibt: kein Arm >= 0,10 in in_dist_random; "
                       "F->F <= 0,06; both ~0; shift <= in; Q->Q bester "
                       "Kandidat aber < 0,10",
           "runtime_s": round(time.time() - t0, 1)}
    fn = "crsa_campaign_smoke.json" if args.smoke else "crsa_campaign.json"
    json.dump(out, open(fn, "w"), indent=1)
    print(f"({time.time()-t0:.0f}s) -> {fn}", flush=True)


if __name__ == "__main__":
    main()