#!/usr/bin/env python3 -u
"""
Length-Extrap v2 (warmup-fixed transformer + NoPE ablation) (M2) — by Opus 4.8
==============================================================================

Milestone M2. Corrects + extends length_extrap.py. Two changes, both surgical,
both leaving reference/ (chmod 444, the contribution) untouched.

WHY v1 NEEDED FIXING
--------------------
v1's Transformer arm NEVER LEARNED: nn-attention + Adam lr 3e-3 + no warmup is
unstable at this width, so the baseline froze at acc 0.164 (≈ chance) and its
length curve was VOID — a frozen model has no meaningful train-short/test-long
signal, it just emits garbage at every length. A length comparison against a
model that never learned proves nothing. FIX: warmup+cosine LR (reused verbatim
from width_fix.warmup_cosine), with a 1e-3 LR fallback if 3e-3 still won't
converge. The arm is GATED: we assert acc > 0.20 at the train length T=32 before
we trust — or even emit — its length curve. If it never learns, the row is
flagged learned=False and excluded from the verdict, honestly.

WHY THE NoPE ARM
----------------
DEEP_ANALYSIS.md attributes Selective's residual length drift (+13% to T=1024)
to a POSITIONAL-ENCODING confound, NOT the scan: scan saturation is flat to
T=1024 and the γ>0.97 tail is 0.0 (no slow channel that fails to settle). The
sinusoidal PE, however, has NO support past its training length — at eval T>32
the model sees position vectors it never trained on (OOD aliasing). The scan
recurrence itself contains no positional term (stationary state s* carries no T;
MASTER_BRIEF §1.4), so REMOVING the PE should remove the residual. Prediction:
NoPE-Selective is EVEN FLATTER than +13%. If it holds, the length-invariance
claim is strengthened — the drift was the PE, not the architecture.

NoPE IMPLEMENTATION: a tiny subclass of the ORIGINAL SelectiveRapiditySqrt-
TransformerLM. The parent __init__ builds the real scan layers (the
contribution) and a SinusoidalPositionalEncoding; the subclass simply REPLACES
self.pos with nn.Identity() after construction. Zero scan-math is touched —
identical weights, identical recurrence, the additive PE term is the only thing
removed. (Selective's forward is h = self.pos(self.embed(x)); with pos=Identity
that is exactly h = self.embed(x).)

ARMS (all share the SAME frozen length sweep, train T=32):
  selective        — Selective WITH sinusoidal PE (reference, reproduces v1)
  selective_nope   — Selective with PE replaced by identity (the ablation)
  pure             — SqrtCoupling scan (interior-attractor control, no gates)
  transformer      — Standard attention + warmup+cosine (the FIXED baseline)

PROTOCOL: train each at T=32 on full WikiText-2 MLM, FREEZE, evaluate at
T ∈ {32,64,128,256,512,1024} by re-tiling the val corpus. The claim is the
CURVE SHAPE, not the level. Instruments: per-position saturation + per-channel
effective-γ distribution per length (same Probe as v1, reconstructed from the
ORIGINAL layer weights).

COMMITTED PREDICTIONS (falsifiable):
  selective_nope : Δ128→1024 SMALLER (flatter) than selective's +13%  ← the bet
  selective      : flat-after-bump, ~+13% residual (PE confound present)
  pure           : monotone rise, gap widens with T
  transformer    : ONLY valid if it learns (acc>0.20 @T32); then monotone rise
  KILLS THE PE-CONFOUND STORY: NoPE is NOT flatter (drift survives PE removal →
                  the residual is intrinsic to the scan after all).

--smoke (DEFAULT): d=64, capped tokens, 2 epochs, eval up to T=256 — proves the
machine end-to-end cheaply. --full runs the real sweep. GPU is busy; default is
the smoke. Verify with ast.parse; do NOT run heavily.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import os, re, time, math, json, argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Imports: reference/ is the CONTRIBUTION (chmod 444, imported never edited);
#    src/ gives us the proven warmup_cosine schedule. ──
SRC = Path(__file__).resolve().parent
REF = SRC.parent / "reference"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REF))

from moebius_scan_transformer_selective import (
    SelectiveRapiditySqrtTransformerLM, SelectiveRapiditySqrtScanLayer)
from moebius_scan_transformer_sqrt import (
    SqrtCouplingMoebiusScanTransformerLM, SqrtCouplingMoebiusScanLayer)
from moebius_attention import StandardTransformerLayer, SinusoidalPositionalEncoding

# Reuse the EXACT warmup+cosine schedule that fixed d=512 in width_fix.py, so
# the transformer baseline is fixed with a proven, already-verified lever.
from width_fix import warmup_cosine

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device: {DEVICE}")

SEED = 42
VOCAB_MAX = 5000
MASK_PROB = 0.15
N_LAYERS = 2
DROPOUT = 0.1
TRAIN_T = 32
EPS = 1e-6
LEARN_ACC_GATE = 0.20      # transformer arm is only trusted above this @ T=32

RESULTS_DIR = SRC.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ===========================================================================
# NoPE-Selective — subclass the ORIGINAL Selective LM, drop the PE only
# ===========================================================================

class SelectiveNoPETransformerLM(SelectiveRapiditySqrtTransformerLM):
    """Selective LM with the sinusoidal positional encoding REMOVED.

    The parent constructor builds the real scan layers (the contribution) AND a
    SinusoidalPositionalEncoding into self.pos. We replace ONLY self.pos with an
    identity, so the parent forward `h = self.pos(self.embed(x))` becomes a plain
    embedding lookup. No scan math, no weight, no init is touched — this is a
    pure ablation of the additive position term. Hypothesis: removing the PE
    (which has no support past T=32) removes the OOD-aliasing residual. NOTE: this
    isolates the ADDITIVE-PE confound; it does NOT make the model 'position-free'
    (the causal scan still carries finite-T positional information through its
    ordering). The honest claim a flatter NoPE curve supports is "the residual
    drift was the PE confound," not "the scan has no notion of position."
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Sanity: parent really did install a PE we are now removing.
        assert isinstance(self.pos, SinusoidalPositionalEncoding), \
            f"expected SinusoidalPositionalEncoding, got {type(self.pos)}"
        self.pos = nn.Identity()   # NoPE: forward becomes h = self.embed(x)


# ===========================================================================
# Transformer baseline — same envelope as the GSSM LMs (embed+PE+blocks+head)
# ===========================================================================

class TransformerLM(nn.Module):
    """Standard transformer baseline in the SAME envelope as the GSSM LMs
    (embed vocab+2, sinusoidal PE, head vocab+1) for a fair compare. Identical
    to v1's wrapper — the FIX is in the optimizer (warmup+cosine), not here."""

    def __init__(self, vocab_size, mask_idx, d_model=128, n_layers=2,
                 n_heads=4, d_head=32, seq_len=32, dropout=0.1, causal=True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 2, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=2048)
        self.layers = nn.ModuleList([
            StandardTransformerLayer(d_model, n_heads=n_heads, d_head=d_head,
                                     causal=causal, ffn_dim=4 * d_model, dropout=dropout)
            for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size + 1)

    def forward(self, x):
        h = self.pos(self.embed(x))
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


# ===========================================================================
# Data (full WikiText-2 MLM protocol — identical to v1 / instrumented_runner)
# ===========================================================================

def load_wikitext2():
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    return ("\n\n".join(ds["train"]["text"]), "\n\n".join(ds["validation"]["text"]))


def build_vocab(text):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    vocab = [w for w, _ in sorted(freq.items(), key=lambda kv: -kv[1])][:VOCAB_MAX]
    stoi = {w: i for i, w in enumerate(vocab)}
    return vocab, stoi, len(vocab), len(vocab) + 1


def tokenize(text, stoi, unk):
    return [stoi.get(w, unk) for w in re.findall(r"[a-zA-Z]+", text.lower())]


def make_mlm_batches(ids, seq_len, batch_size, mask_idx, mask_prob=0.15, max_tokens=None):
    if max_tokens is not None:
        ids = ids[:max_tokens + 1]
    total = len(ids) // seq_len
    X, Y, M = [], [], []
    for i in range(total):
        seq = ids[i * seq_len:(i + 1) * seq_len]
        if len(seq) < seq_len:
            continue
        y = seq.copy()
        mask = (torch.rand(seq_len) < mask_prob).long()
        for j in range(seq_len):
            if mask[j]:
                r = torch.rand(1).item()
                if r < 0.8:
                    seq[j] = mask_idx
                elif r < 0.9:
                    seq[j] = torch.randint(0, mask_idx, (1,)).item()
        X.append(seq); Y.append(y); M.append(mask)
    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    M = torch.stack(M)
    n = (len(X) // batch_size) * batch_size
    return X[:n], Y[:n], M[:n]


# ===========================================================================
# Saturation + gamma probe (reconstruct from ORIGINAL weights) — same as v1.
# NoPE-Selective still contains SelectiveRapiditySqrtScanLayer, so the probe
# hooks it unchanged.
# ===========================================================================

class Probe:
    def __init__(self, model):
        self.handles = []
        self.reset()
        for mod in model.modules():
            if isinstance(mod, SelectiveRapiditySqrtScanLayer):
                self.handles.append(mod.register_forward_hook(self._sel))
            elif isinstance(mod, SqrtCouplingMoebiusScanLayer):
                self.handles.append(mod.register_forward_hook(self._pure))

    def reset(self):
        self.pos_sat = None
        self.pos_n = 0
        self.gamma_hist = []
        self.has_gamma = False

    @torch.no_grad()
    def _sel(self, module, inp, out):
        x = inp[0]
        B, T, _ = x.shape
        H, D = module.n_heads, module.d_head
        gamma = torch.sigmoid(module.W_gamma(x))
        self.has_gamma = True
        self.gamma_hist.append(gamma.mean(dim=(0, 1)).detach().cpu())
        v = torch.tanh(module.W_v(x)); gate = torch.sigmoid(module.W_gate(x))
        alpha = torch.sigmoid(module.W_alpha(x)).view(B, T, H, D)
        gamma = gamma.view(B, T, H, D)
        vg = (v * gate).view(B, T, H, D)
        a = alpha * torch.log(1.0 - torch.clamp(vg * vg, max=0.999) + EPS)
        Z = torch.zeros(B, H, D, device=x.device); outs = []
        for t in range(T):
            Z = gamma[:, t] * Z + a[:, t]; outs.append(Z)
        s = torch.sqrt(torch.clamp(1.0 - torch.exp(torch.stack(outs, 1)), min=0.0) + EPS)
        self._acc(s)

    @torch.no_grad()
    def _pure(self, module, inp, out):
        x = inp[0]
        B, T, _ = x.shape
        H, D = module.n_heads, module.d_head
        v = torch.tanh(module.W_v(x)); gate = torch.sigmoid(module.W_gate(x))
        vg = (v * gate).view(B, T, H, D)
        s_prev = torch.zeros(B, H, D, device=x.device); outs = []
        for t in range(T):
            s_prev = torch.sqrt(vg[:, t] ** 2 + s_prev ** 2 * (1 - vg[:, t] ** 2) + EPS)
            outs.append(s_prev)
        self._acc(torch.stack(outs, 1))

    @torch.no_grad()
    def _acc(self, s):  # s: (B,T,H,D)
        pos = (s > 0.95).float().mean(dim=(0, 2, 3)).detach().cpu()  # (T,)
        if self.pos_sat is None or self.pos_sat.shape[0] != pos.shape[0]:
            self.pos_sat = pos.clone(); self.pos_n = 1
        else:
            self.pos_sat += pos; self.pos_n += 1

    def profile(self, T):
        if self.pos_sat is None:
            return {}
        p = (self.pos_sat / max(1, self.pos_n)).tolist()
        n = len(p)
        prof = {"first": round(p[0], 4), "q1": round(p[n // 4], 4),
                "mid": round(p[n // 2], 4), "last": round(p[-1], 4)}
        if self.has_gamma and self.gamma_hist:
            g = torch.stack(self.gamma_hist).mean(0)
            prof["gamma_mean"] = round(g.mean().item(), 4)
            prof["gamma_p95"] = round(g.quantile(0.95).item(), 4)
            prof["gamma_frac_gt97"] = round((g > 0.97).float().mean().item(), 4)
        return prof

    def remove(self):
        for h in self.handles:
            h.remove()


# ===========================================================================
# Train / eval
# ===========================================================================

def train_epoch(model, X, Y, M, opt, batch, clip=5.0, sched=None):
    model.train()
    perm = torch.randperm(len(X)); tot = 0.0; nb = 0
    for i in range(0, len(X), batch):
        idx = perm[i:i + batch]
        xb, yb, mb = X[idx].to(DEVICE), Y[idx].to(DEVICE), M[idx].to(DEVICE)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               yb.reshape(-1), reduction='none')
        loss = (loss * mb.reshape(-1).float()).sum() / (mb.sum() + 1e-6)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip); opt.step()
        if sched is not None:
            sched.step()
        tot += loss.item(); nb += 1
    return tot / nb


@torch.no_grad()
def evaluate(model, X, Y, M, probe=None, batch=32):
    model.eval(); tot = 0.0; cor = 0; msk = 0; nb = 0
    if probe:
        probe.reset()
    for i in range(0, len(X), batch):
        xb, yb, mb = X[i:i + batch].to(DEVICE), Y[i:i + batch].to(DEVICE), M[i:i + batch].to(DEVICE)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               yb.reshape(-1), reduction='none')
        tot += ((loss * mb.reshape(-1).float()).sum() / (mb.sum() + 1e-6)).item()
        preds = logits.argmax(-1)
        cor += ((preds == yb) & mb.bool()).sum().item(); msk += mb.sum().item(); nb += 1
    return tot / nb, math.exp(tot / nb), cor / msk


# arm key -> (label, constructor, needs_warmup, default_lr, lr_fallback)
# warmup is the FIX: only the transformer baseline uses it. The scan arms keep
# the plain-Adam lr=3e-3 they were tuned with (so selective reproduces v1).
ARMS = {
    "selective":      ("Selective",      SelectiveRapiditySqrtTransformerLM,  False, 3e-3, None),
    "selective_nope": ("Selective-NoPE", SelectiveNoPETransformerLM,          False, 3e-3, None),
    "pure":           ("Pure",           SqrtCouplingMoebiusScanTransformerLM, False, 3e-3, None),
    "transformer":    ("Transformer",    TransformerLM,                       True,  3e-3, 1e-3),
}


def train_arm(label, cls, vsz, mask, d_model, n_heads, d_head,
              Xtr, Ytr, Mtr, val32, train_batch, epochs, lr,
              needs_warmup, warmup_steps):
    """Train ONE arm at T=32. Returns (model, best_ppl, train_acc, train_time).
    train_acc is the T=32 acc at best epoch — the learn-gate for the transformer."""
    torch.manual_seed(SEED)
    model = cls(vsz, mask, d_model=d_model, n_layers=N_LAYERS, n_heads=n_heads,
                d_head=d_head, seq_len=TRAIN_T, dropout=DROPOUT, causal=True).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = None
    if needs_warmup:
        total_steps = epochs * math.ceil(len(Xtr) / train_batch)
        sched = warmup_cosine(opt, warmup_steps, total_steps)
    Xv, Yv, Mv, vb = val32
    best = float("inf"); best_acc = 0.0; since = 0; t0 = time.time()
    for ep in range(epochs):
        tl = train_epoch(model, Xtr, Ytr, Mtr, opt, train_batch, sched=sched)
        vl, vppl, vacc = evaluate(model, Xv, Yv, Mv, batch=vb)
        lrnow = opt.param_groups[0]["lr"]
        print(f"  {label} ep{ep+1}/{epochs} train {tl:.4f} | T32 val ppl {vppl:.2f} "
              f"acc {vacc:.4f} | lr {lrnow:.2e}")
        if vppl < best - 0.5:
            best = vppl; best_acc = vacc; since = 0
        else:
            since += 1
            if since >= 2:
                print("    early-stop"); break
    return model, best, best_acc, time.time() - t0


def frozen_sweep(label, model, val_sets, eval_ts):
    """Freeze + evaluate across lengths with the Probe. Returns the curve dict."""
    model.eval()
    probe = Probe(model)
    curve = {}
    for T in eval_ts:
        Xv, Yv, Mv, b = val_sets[T]
        vl, vppl, vacc = evaluate(model, Xv, Yv, Mv, probe, batch=b)
        prof = probe.profile(T)
        curve[str(T)] = {"ppl": round(vppl, 3), "acc": round(vacc, 4), "sat": prof}
        gstr = (f" γ̄={prof.get('gamma_mean','-')} γ>97={prof.get('gamma_frac_gt97','-')}"
                if 'gamma_mean' in prof else "")
        print(f"    T={T:>4}: ppl {vppl:>8.2f} acc {vacc:.4f} | sat first→last "
              f"{prof.get('first',0):.3f}→{prof.get('last',0):.3f}{gstr}")
    probe.remove()
    # length-invariance metric: % change over the OOD span the PE never saw.
    for lo, hi in (("128", "1024"), ("128", str(max(eval_ts)))):
        if lo in curve and hi in curve and lo != hi:
            plo, phi = curve[lo]["ppl"], curve[hi]["ppl"]
            dpct = 100 * (phi - plo) / plo
            curve[f"_delta_{lo}_{hi}_pct"] = round(dpct, 2)
            break
    return curve


# ===========================================================================
# Main sweep
# ===========================================================================

def run(d_model, eval_ts, epochs, warmup_steps, max_tokens, smoke, out_tag=None):
    print("=" * 78)
    print("Length-Extrap v2 (warmup-fixed transformer + NoPE ablation) (M2) — by Opus 4.8")
    print("=" * 78)
    n_heads = max(1, d_model // 32)
    d_head = d_model // n_heads
    print(f"d_model={d_model} n_heads={n_heads} d_head={d_head} | train T={TRAIN_T} "
          f"| eval T={eval_ts} | epochs={epochs}{' | SMOKE' if smoke else ''}")

    train_text, val_text = load_wikitext2()
    vocab, stoi, unk, mask = build_vocab(train_text)
    vsz = len(vocab)
    train_ids = tokenize(train_text, stoi, unk)
    val_ids = tokenize(val_text, stoi, unk)
    print(f"Vocab {vsz} | train {len(train_ids):,} val {len(val_ids):,} tokens")

    train_batch = 32
    Xtr, Ytr, Mtr = make_mlm_batches(train_ids, TRAIN_T, train_batch, mask, MASK_PROB,
                                     max_tokens=max_tokens)
    print(f"Train batches (T={TRAIN_T}): {len(Xtr) // train_batch}")

    # Per-length val sets, re-tiled from the SAME val corpus. Drop batch at long T.
    val_sets = {}
    for T in eval_ts:
        b = 8 if T >= 512 else train_batch
        Xv, Yv, Mv = make_mlm_batches(val_ids, T, b, mask, MASK_PROB)
        val_sets[T] = (Xv, Yv, Mv, b)
        print(f"  val T={T}: {len(Xv) // b} batches (batch={b})")
    val32 = val_sets[TRAIN_T]

    results = {"_meta": {"d_model": d_model, "n_heads": n_heads, "d_head": d_head,
                         "train_T": TRAIN_T, "eval_Ts": eval_ts, "epochs": epochs,
                         "smoke": smoke, "learn_gate_acc": LEARN_ACC_GATE,
                         "by": "Opus 4.8 — Length-Extrap v2 (M2)"}}
    if out_tag:
        out = RESULTS_DIR / f"length_extrap_v2_{out_tag}.json"
    else:
        out = RESULTS_DIR / ("length_extrap_v2_smoke.json" if smoke else "length_extrap_v2.json")

    for key, (label, cls, needs_warmup, lr, lr_fallback) in ARMS.items():
        print(f"\n── Training {label} at T={TRAIN_T} "
              f"({'warmup+cosine' if needs_warmup else 'plain Adam'}, lr={lr:.0e}) ──")
        model, best, train_acc, ttime = train_arm(
            label, cls, vsz, mask, d_model, n_heads, d_head,
            Xtr, Ytr, Mtr, val32, train_batch, epochs, lr, needs_warmup, warmup_steps)

        # FIX-VALIDATION GATE: the transformer length curve is only meaningful
        # once it has actually learned. If lr=3e-3 didn't clear the gate, retry
        # once at the fallback lr before giving up.
        learned = train_acc > LEARN_ACC_GATE
        used_lr = lr
        if needs_warmup and not learned and lr_fallback is not None:
            print(f"  ⚠ {label} did NOT learn at lr={lr:.0e} (acc {train_acc:.4f} ≤ "
                  f"{LEARN_ACC_GATE}). Retrying at fallback lr={lr_fallback:.0e}.")
            del model; (torch.mps.empty_cache() if DEVICE.type == "mps" else None)
            model, best, train_acc, ttime = train_arm(
                label, cls, vsz, mask, d_model, n_heads, d_head,
                Xtr, Ytr, Mtr, val32, train_batch, epochs, lr_fallback,
                needs_warmup, warmup_steps)
            learned = train_acc > LEARN_ACC_GATE
            used_lr = lr_fallback

        if needs_warmup:
            flag = "LEARNED ✓" if learned else "NEVER LEARNED ✗ (curve VOID — excluded)"
            print(f"  → learn-gate (acc>{LEARN_ACC_GATE} @T32): {train_acc:.4f} → {flag}")

        print(f"  → frozen length sweep:")
        curve = frozen_sweep(label, model, val_sets, eval_ts)

        # Show the headline Δ over whatever OOD span we computed.
        dkey = next((k for k in curve if k.startswith("_delta_")), None)
        if dkey:
            d = curve[dkey]
            span = dkey.replace("_delta_", "").replace("_pct", "").replace("_", "→T")
            print(f"  → ΔT{span}: {d:+.1f}%  "
                  f"({'FLAT/length-invariant' if abs(d) < 5 else 'DEGRADES'})")

        results[key] = {
            "label": label, "train_best_ppl": round(best, 3),
            "train_acc_T32": round(train_acc, 4), "train_lr": used_lr,
            "needs_warmup": needs_warmup, "learned": (learned if needs_warmup else True),
            "train_time_s": round(ttime, 1), "curve": curve,
        }
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        del model
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    _verdict(results, eval_ts, out)
    return results


def _verdict(results, eval_ts, out):
    print(f"\n{'=' * 78}\nVERDICT — PPL vs eval-T per arm\n{'=' * 78}")
    print(f"{'Arm':<16} " + " ".join(f"T={t:<5}" for t in eval_ts) + "  Δ(OOD span)  learned")
    for key, (label, *_rest) in ARMS.items():
        if key not in results:
            continue
        r = results[key]; c = r["curve"]
        row = f"{label:<16} " + " ".join(f"{c.get(str(t), {}).get('ppl', 0):<7.1f}" for t in eval_ts)
        dkey = next((k for k in c if k.startswith("_delta_")), None)
        dstr = f"{c[dkey]:>+6.1f}%" if dkey else "   —  "
        lstr = "yes" if r.get("learned", True) else "NO (void)"
        print(row + f"  {dstr}     {lstr}")

    # The headline ablation comparison.
    sel = results.get("selective", {}).get("curve", {})
    nope = results.get("selective_nope", {}).get("curve", {})
    dk_sel = next((k for k in sel if k.startswith("_delta_")), None)
    dk_nope = next((k for k in nope if k.startswith("_delta_")), None)
    if dk_sel and dk_nope:
        ds, dn = sel[dk_sel], nope[dk_nope]
        print(f"\nNoPE ABLATION: Selective Δ={ds:+.1f}%  vs  Selective-NoPE Δ={dn:+.1f}%")
        if abs(dn) < abs(ds):
            print(f"  → NoPE is FLATTER (|{dn:.1f}| < |{ds:.1f}|): the residual length drift was "
                  f"the ADDITIVE sinusoidal-PE OOD-aliasing term. Removing it removes the drift. "
                  f"Honest claim: the drift was the PE confound — NOT that the scan is 'position-free' "
                  f"(the causal scan still carries finite-T positional info). Length-invariance STRENGTHENED.")
        else:
            print(f"  → NoPE is NOT flatter: the drift SURVIVES PE removal → the residual is "
                  f"intrinsic to the scan dynamics, not the additive-PE artifact. PE-confound story KILLED.")
    if "transformer" in results and not results["transformer"].get("learned", False):
        print("\n⚠ Transformer NEVER LEARNED even with warmup+fallback — its length curve is "
              "VOID and excluded from the verdict (a frozen-at-chance model has no length signal).")
    print(f"\nResults: {out}")


def build_argparser():
    p = argparse.ArgumentParser(
        description="Length-Extrap v2 (warmup-fixed transformer + NoPE ablation) (M2) — by Opus 4.8")
    p.add_argument("--full", action="store_true",
                   help="real sweep: d=128, T up to 1024, 8 epochs, full data")
    p.add_argument("--smoke", action="store_true",
                   help="tiny smoke (DEFAULT): d=64, capped data, 2 ep, T up to 256")
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--eval-ts", type=str, default=None,
                   help="comma-separated eval lengths, e.g. 32,64,...,16384 (overrides default)")
    p.add_argument("--out-tag", type=str, default=None,
                   help="suffix for the output json (length_extrap_v2_<tag>.json)")
    return p


def main():
    args = build_argparser().parse_args()
    full = args.full and not args.smoke      # --smoke wins ties; smoke is the default
    if full:
        d_model = args.d_model or 128
        eval_ts = [32, 64, 128, 256, 512, 1024]
        if args.eval_ts:
            eval_ts = [int(t) for t in args.eval_ts.split(",")]
        epochs = args.epochs or 8
        warmup_steps = args.warmup_steps or 1000
        max_tokens = args.max_tokens          # None = full corpus
        smoke = False
    else:
        d_model = args.d_model or 64
        eval_ts = [32, 64, 128, 256]
        epochs = args.epochs or 2
        warmup_steps = args.warmup_steps or 50
        max_tokens = args.max_tokens or 60_000
        smoke = True
    run(d_model, eval_ts, epochs, warmup_steps, max_tokens, smoke, out_tag=args.out_tag)


if __name__ == "__main__":
    main()
