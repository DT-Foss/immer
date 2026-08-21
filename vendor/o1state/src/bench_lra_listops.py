#!/usr/bin/env python3 -u
"""
LRA ListOps — the positioning benchmark (NoPE-Selective GSSM vs Mamba/S4)  — 2026-06-26
=======================================================================================
ListOps is the cheapest, most standard Long Range Arena task: hierarchically nested
list operations (MAX/MIN/MED/SUM_MOD over single digits), serialized into a token
sequence with brackets. The model must integrate information across the WHOLE sequence
(the answer depends on deeply nested structure) — a sequence-CLASSIFICATION task with
10 classes (output digit 0..9). LRA uses lengths up to ~2000 tokens; chance = 10%.

This is the bounded-state pitch made concrete: a NoPE selective GSSM is O(1)-memory and
position-buffer-free, so it ingests a 2000-token ListOps tree at the same constant state
size as a 100-token one. The published LRA leaderboard numbers (Transformer ~36%, S4 ~59%,
Mamba ~38–60% depending on config) are what this run positions against.

WHY FROM SCRATCH: there is no canonical pip ListOps generator that's CPU-cheap and
self-contained. We generate the dataset with the standard recursive grammar (Nangia &
Bowman 2018 / LRA Tay et al. 2020): OPEN op args... CLOSE, args are digits or nested
sub-expressions, depth- and length-bounded. Verified by an independent reference solver.

MODEL: we reuse the FROZEN reference scan stack (reference/moebius_scan_transformer_selective.py)
via the published NoPE subclass (src/length_extrap_v2.py::SelectiveNoPETransformerLM) — same
embed + selective-scan layers, PE removed. For CLASSIFICATION we add a thin pooling head:
mask-aware mean-pool over the per-token hidden states, then a Linear to 10 classes. The scan
math, gates and init are untouched — this is the standard "encoder + pooled classifier" head
swap, nothing about the contribution is re-implemented. At long T we route the scan through
the verified parallel scan (src/parallel_scan.py, grad-identical to sequential) for speed.

Harness style matches src/longcontext_run.py / src/longcontext_tasks.py: sys.path.insert for
reference+src, writes results/<name>.json, has __main__. Watchdog (psutil SIGKILL >~10GB) is
copied from src/percolation_hard.py. Validity gate: a from-scratch dataset must be SOLVABLE —
we assert the reference solver agrees with the generated labels on a held-out check, and we
report whether the GSSM beats the 10% chance floor (the real claim is the leaderboard number,
which the full run produces).

Smoke: `python3 src/bench_lra_listops.py --smoke`  (tiny d_model, few steps, short lengths).
Full : `python3 src/bench_lra_listops.py`          (LRA-scale: T≤2000, real training budget).
"""
import os, sys, json, argparse, time, threading, signal, random
sys.path.insert(0, "reference"); sys.path.insert(0, "src")

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── memory watchdog (copied pattern from src/percolation_hard.py) ──
try:
    import psutil
    _P = psutil.Process(os.getpid())
    def _rss(): return _P.memory_info().rss / 1e9
except ImportError:
    def _rss(): return 0.0


def _watchdog(hard_gb=10.0):
    def w():
        while True:
            if _rss() > hard_gb:
                os.kill(os.getpid(), signal.SIGKILL)
            time.sleep(0.5)
    threading.Thread(target=w, daemon=True).start()


# frozen reference stack + published NoPE subclass + verified parallel scan
from length_extrap_v2 import SelectiveNoPETransformerLM
from parallel_scan_integration import use_parallel_scan


# ===========================================================================
# 1. ListOps dataset — standard recursive grammar, generated from scratch
# ===========================================================================
# Token vocabulary (LRA-standard ListOps):
#   digits 0..9              -> ids 0..9        (also the 10 class labels)
#   operators MAX MIN MED SM -> ids 10..13      (SM = SUM_MOD 10)
#   '('  ')'                 -> ids 14, 15      (open / close brackets)
# vocab_size = 16. Classification over 10 digit classes.
OPS = ["MAX", "MIN", "MED", "SM"]
DIGIT_BASE = 0
OP_BASE = 10
OPEN, CLOSE = 14, 15
VOCAB_SIZE = 16
N_CLASSES = 10
PAD_ID = 16  # padding token id (outside the 0..15 vocab; embedding has vocab_size+2 rows)


def _op_apply(op, vals):
    if op == "MAX": return max(vals)
    if op == "MIN": return min(vals)
    if op == "MED":
        s = sorted(vals)
        n = len(s)
        return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) // 2
    if op == "SM":  return sum(vals) % 10
    raise ValueError(op)


def _gen_tree(rng, max_depth, max_args, p_nest):
    """Recursively build a ListOps expression. Returns (token_ids, value).
    token_ids is the serialized '( OP arg arg ... )' as a list of vocab ids.
    value is the integer result in 0..9 (the ground-truth label)."""
    op = rng.choice(OPS)
    n_args = rng.randint(2, max_args)
    toks = [OPEN, OP_BASE + OPS.index(op)]
    vals = []
    for _ in range(n_args):
        # nest only if depth budget remains and the coin says so
        if max_depth > 1 and rng.random() < p_nest:
            sub_toks, sub_val = _gen_tree(rng, max_depth - 1, max_args, p_nest)
            toks.extend(sub_toks)
            vals.append(sub_val)
        else:
            d = rng.randint(0, 9)
            toks.append(DIGIT_BASE + d)
            vals.append(d)
    toks.append(CLOSE)
    return toks, _op_apply(op, vals)


def _ref_solve(tokens):
    """Independent stack-based solver over the SERIALIZED token ids — verifies the
    generated label without trusting the generator's recursion. Returns 0..9."""
    stack = []  # each entry: list that may hold an op marker then operands
    op_of = {OP_BASE + i: OPS[i] for i in range(len(OPS))}
    for t in tokens:
        if t == OPEN:
            stack.append([])  # new frame
        elif t in op_of:
            stack[-1].append(("OP", op_of[t]))
        elif DIGIT_BASE <= t <= DIGIT_BASE + 9:
            stack[-1].append(("V", t - DIGIT_BASE))
        elif t == CLOSE:
            frame = stack.pop()
            assert frame[0][0] == "OP", "malformed frame"
            op = frame[0][1]
            vals = [v for (k, v) in frame[1:] if k == "V"]
            val = _op_apply(op, vals)
            if stack:
                stack[-1].append(("V", val))
            else:
                return val
    raise ValueError("unbalanced expression")


def make_listops_dataset(n, min_len, max_len, max_depth, max_args, p_nest, seed,
                         verify=True):
    """Generate n examples with length in [min_len, max_len] (token count incl. brackets).
    Returns (list_of_token_lists, labels_tensor, max_observed_len). Each label is verified
    against the independent stack solver when verify=True."""
    rng = random.Random(seed)
    seqs, labels = [], []
    attempts = 0
    while len(seqs) < n:
        attempts += 1
        if attempts > n * 200:
            raise RuntimeError("length window too tight; relax min/max_len")
        toks, val = _gen_tree(rng, max_depth, max_args, p_nest)
        if not (min_len <= len(toks) <= max_len):
            continue
        if verify:
            assert _ref_solve(toks) == val, "generator/solver mismatch"
        seqs.append(toks)
        labels.append(val)
    max_obs = max(len(s) for s in seqs)
    return seqs, torch.tensor(labels, dtype=torch.long), max_obs


def collate(seqs, labels, pad_to, device):
    """Right-pad token lists to pad_to; return (tokens, lengths, labels) on device."""
    B = len(seqs)
    tok = torch.full((B, pad_to), PAD_ID, dtype=torch.long)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, s in enumerate(seqs):
        L = min(len(s), pad_to)
        tok[i, :L] = torch.tensor(s[:L], dtype=torch.long)
        lengths[i] = L
    return tok.to(device), lengths.to(device), labels.to(device)


# ===========================================================================
# 2. Classifier head over the frozen NoPE-Selective scan stack
# ===========================================================================

class ListOpsGSSMClassifier(nn.Module):
    """Encoder = the published NoPE-Selective GSSM (embed + selective scan layers,
    PE removed). Head = mask-aware mean-pool over per-token hidden states -> Linear
    to N_CLASSES. We tap the encoder's hidden states BEFORE its LM head (so the
    contribution's scan output is what gets pooled), then classify."""

    def __init__(self, d_model, n_layers, n_heads, d_head, dropout=0.0):
        super().__init__()
        # mask_idx is required by the reference ctor; PAD_ID=16 sits in the +2 rows.
        self.enc = SelectiveNoPETransformerLM(
            VOCAB_SIZE, mask_idx=VOCAB_SIZE, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, d_head=d_head, seq_len=64, dropout=dropout, causal=True)
        self.cls = nn.Linear(d_model, N_CLASSES)

    def _encode(self, x):
        # mirror SelectiveRapiditySqrtTransformerLM.forward up to (not incl.) head
        h = self.enc.pos(self.enc.embed(x))   # pos = Identity (NoPE)
        for layer in self.enc.layers:
            h = layer(h)
        return h                               # (B, T, d_model)

    def forward(self, x, lengths):
        h = self._encode(x)                    # (B, T, d_model)
        B, T, _ = h.shape
        idx = torch.arange(T, device=x.device).unsqueeze(0)        # (1,T)
        m = (idx < lengths.unsqueeze(1)).float().unsqueeze(-1)     # (B,T,1) valid-token mask
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1.0)          # mask-aware mean
        return self.cls(pooled)                # (B, N_CLASSES)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===========================================================================
# 3. Train / eval
# ===========================================================================

def run_split(model, seqs, labels, pad_to, device, batch_size, train, opt=None,
              use_parallel=True):
    model.train(train)
    n = len(seqs)
    order = list(range(n))
    if train:
        random.shuffle(order)
    tot_loss = tot_correct = tot = 0
    ctx = use_parallel_scan() if use_parallel else _nullcontext()
    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for i in range(0, n, batch_size):
            bi = order[i:i + batch_size]
            bs = [seqs[j] for j in bi]
            bl = labels[bi]
            tok, lengths, y = collate(bs, bl, pad_to, device)
            with ctx:
                logits = model(tok, lengths)
            loss = F.cross_entropy(logits, y)
            if train:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot_loss += loss.item() * len(bi)
            tot_correct += (logits.argmax(-1) == y).sum().item()
            tot += len(bi)
    return tot_loss / max(1, tot), tot_correct / max(1, tot)


class _nullcontext:
    def __enter__(self): return None
    def __exit__(self, *a): return False


# ===========================================================================
# 4. Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny config: d_model=32, T<=64, few steps — proves end-to-end")
    ap.add_argument("--n-train", type=int, default=96000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--min-len", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=2000)
    ap.add_argument("--max-depth", type=int, default=10)
    ap.add_argument("--max-args", type=int, default=5)
    ap.add_argument("--p-nest", type=float, default=0.30)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-head", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    ap.add_argument("--no-parallel-scan", action="store_true",
                    help="use the O(T) sequential scan instead of the parallel one")
    ap.add_argument("--out", default="results/bench_lra_listops.json")
    args = ap.parse_args()

    if args.smoke:
        # tiny everything: short sequences, small model, a couple of epochs
        args.n_train, args.n_val = 256, 64
        args.min_len, args.max_len = 10, 60
        args.max_depth, args.max_args = 4, 4
        args.d_model, args.n_layers, args.n_heads, args.d_head = 32, 2, 2, 16
        args.epochs, args.batch_size = 2, 16
        args.out = "results/bench_lra_listops_smoke.json"

    _watchdog(hard_gb=10.0)
    torch.manual_seed(args.seed); random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 78)
    print(f"LRA ListOps — NoPE-Selective GSSM classifier  {'[SMOKE]' if args.smoke else '[FULL]'}")
    print(f"  n_train={args.n_train} n_val={args.n_val} len=[{args.min_len},{args.max_len}] "
          f"depth<= {args.max_depth} args<= {args.max_args}")
    print(f"  d_model={args.d_model} layers={args.n_layers} heads={args.n_heads} "
          f"d_head={args.d_head}  epochs={args.epochs} bs={args.batch_size} lr={args.lr}")
    print(f"  device={device}  parallel_scan={not args.no_parallel_scan}  vocab={VOCAB_SIZE} "
          f"classes={N_CLASSES} (chance={100/N_CLASSES:.0f}%)")
    print("=" * 78)

    t0 = time.time()
    train_seqs, train_y, max_tr = make_listops_dataset(
        args.n_train, args.min_len, args.max_len, args.max_depth, args.max_args,
        args.p_nest, seed=args.seed)
    val_seqs, val_y, max_va = make_listops_dataset(
        args.n_val, args.min_len, args.max_len, args.max_depth, args.max_args,
        args.p_nest, seed=args.seed + 7)
    pad_to = max(max_tr, max_va)
    # label-balance sanity: a from-scratch dataset must not be a single-class trap
    import collections
    dist = collections.Counter(train_y.tolist())
    maj = max(dist.values()) / len(train_y)
    print(f"[data] built in {time.time()-t0:.1f}s  pad_to={pad_to}  "
          f"train label-dist majority={maj:.1%}  (verified vs reference solver)")

    model = ListOpsGSSMClassifier(args.d_model, args.n_layers, args.n_heads,
                                  args.d_head).to(device)
    print(f"[model] params={count_params(model):,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_par = not args.no_parallel_scan

    history = []
    best_val = 0.0
    for ep in range(args.epochs):
        tr_loss, tr_acc = run_split(model, train_seqs, train_y, pad_to, device,
                                    args.batch_size, train=True, opt=opt,
                                    use_parallel=use_par)
        va_loss, va_acc = run_split(model, val_seqs, val_y, pad_to, device,
                                    args.batch_size, train=False,
                                    use_parallel=use_par)
        best_val = max(best_val, va_acc)
        history.append({"epoch": ep + 1, "train_loss": round(tr_loss, 4),
                        "train_acc": round(tr_acc, 4), "val_loss": round(va_loss, 4),
                        "val_acc": round(va_acc, 4)})
        print(f"  ep {ep+1:>2}/{args.epochs}  train loss {tr_loss:.4f} acc {tr_acc*100:5.1f}%"
              f"   val loss {va_loss:.4f} acc {va_acc*100:5.1f}%   "
              f"rss {_rss():.2f}GB  t {time.time()-t0:.0f}s", flush=True)

    chance = 1.0 / N_CLASSES
    beats_chance = best_val > chance + 0.02
    results = {
        "task": "lra_listops", "smoke": bool(args.smoke),
        "config": {k: getattr(args, k) for k in
                   ["n_train", "n_val", "min_len", "max_len", "max_depth", "max_args",
                    "p_nest", "d_model", "n_layers", "n_heads", "d_head", "epochs",
                    "batch_size", "lr", "seed"]},
        "device": str(device), "parallel_scan": use_par,
        "vocab_size": VOCAB_SIZE, "n_classes": N_CLASSES, "chance": chance,
        "pad_to": pad_to, "params": count_params(model),
        "train_label_majority": round(maj, 4),
        "best_val_acc": round(best_val, 4), "final_val_acc": round(history[-1]["val_acc"], 4),
        "beats_chance": bool(beats_chance),
        "history": history, "wall_sec": round(time.time() - t0, 1),
        "peak_rss_gb": round(_rss(), 2),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)

    print("\n" + "=" * 78)
    print(f"RESULT  best val acc {best_val*100:.1f}%  (chance {chance*100:.0f}%)  "
          f"{'> chance' if beats_chance else 'AT chance'}")
    print(f"  vs LRA leaderboard: Transformer ~36%, S4 ~59%, Mamba ~38-60%")
    print(f"\n→ {args.out}  ({time.time()-t0:.1f}s, rss {_rss():.2f}GB)")


if __name__ == "__main__":
    main()
