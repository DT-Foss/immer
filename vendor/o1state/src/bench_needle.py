#!/usr/bin/env python3 -u
"""
NEEDLE-IN-A-HAYSTACK / passkey retrieval — NoPE-Selective GSSM vs same-size Transformer.
========================================================================================
Embed a KEY:VALUE pair at a RANDOM position inside a long stream of filler tokens, then
ask for the value at the end (the QUERY). Train SHORT (T_train ~ 64), evaluate as the
haystack GROWS (64 → 8192). The needle moves arbitrarily far from the query, so success
requires carrying one bound pair across an unbounded, mostly-irrelevant prefix.

Why this separates the two:
  • Transformer (TinyCausalTransformerLM) has a LEARNED positional table of size max_len.
    Past max_len the forward pass CRASHES (self.pos[:, :T] returns < T rows → broadcast
    error). Below max_len, positions past the train length are OOD → recall degrades.
    The crash IS a result ("CRASH @ T"), not a failure — it's the capability boundary.
  • NoPE-Selective GSSM has NO positional buffer. It carries the pair through its bounded
    O(1) scan state and ordering alone, so its forward pass runs to ANY T. Hypothesis:
    recall holds (or degrades far more gracefully) where attention falls off / crashes.

This plays to the bounded-state STRENGTH: a SINGLE key→value binding (not multi-key MQAR,
which hits the ~13% holographic ceiling). One needle, one register.

Task layout (vocab, all next-token / read-at-QUERY):
  KEY_TOK        — fixed marker that announces "the next token is the key"   (1 id)
  key in [0,K)   — which slot (K key ids)
  val in [0,V)   — the value to remember (V value ids, the scored targets)
  FILLER         — haystack noise (1 id)
  QUERY_TOK      — "report the value bound to the key"                        (1 id)
Sequence:  ... filler ... [KEY_TOK][key][val] ... filler ... [QUERY_TOK]
Target: at the QUERY_TOK position, next-token = the val that followed the key. Scored there.

Validity gate: the Transformer MUST solve recall at TRAIN length (a single passkey is
trivially attention-solvable at short T). If it can't, the harness is suspect, not the model.

Style mirrors src/longcontext_run.py + src/longcontext_tasks.py (sys.path.insert for
reference+src, task_train/task_accuracy shape, writes results/<name>.json, __main__) and the
psutil memory watchdog (SIGKILL above ~10 GB) copied from src/percolation_hard.py.
"""
import os, sys, json, argparse, time, threading, signal
sys.path.insert(0, "reference"); sys.path.insert(0, "src")

import torch
import torch.nn as nn
import torch.nn.functional as F

from moebius_scan_transformer_selective import SelectiveRapiditySqrtTransformerLM  # noqa: F401 (kept for parity)
from length_extrap_v2 import SelectiveNoPETransformerLM
from mqar import TinyCausalTransformerLM


# ===========================================================================
# psutil memory watchdog — SIGKILL above hard_gb (pattern from percolation_hard.py)
# ===========================================================================
try:
    import psutil
    _P = psutil.Process(os.getpid())
    def _rss(): return _P.memory_info().rss / 1e9
except ImportError:
    def _rss(): return 0.0


def _watchdog(hard_gb=10.0):
    """Background thread: if RSS exceeds hard_gb, SIGKILL self. Long T can blow up the
    Transformer's O(T²) attention matrix; this caps the damage instead of swapping the box."""
    def w():
        while True:
            if _rss() > hard_gb:
                os.kill(os.getpid(), signal.SIGKILL)
            time.sleep(0.5)
    threading.Thread(target=w, daemon=True).start()


# ===========================================================================
# Needle-in-haystack generator
# ===========================================================================

# gap bins (query_pos − needle_pos), mirrors longcontext_tasks.GAP_BINS
GAP_BINS = [(1, 8), (9, 32), (33, 128), (129, 512), (513, 2048),
            (2049, 8192), (8193, 32768)]


def needle_vocab(n_keys, n_vals):
    """vocab layout (value ids first so they double as clean scored targets):
        [0 .. n_vals-1]                = VALUE ids (the scored targets)
        [n_vals .. n_vals+n_keys-1]    = KEY ids
        n_vals+n_keys                  = KEY_TOK  (announces a key follows)
        n_vals+n_keys+1                = QUERY_TOK (announces "report the value")
        n_vals+n_keys+2                = FILLER  (haystack noise)
    """
    base = n_vals + n_keys
    KEY_TOK = base
    QUERY_TOK = base + 1
    FILLER = base + 2
    vocab_size = base + 3
    return dict(KEY_TOK=KEY_TOK, QUERY_TOK=QUERY_TOK, FILLER=FILLER,
                n_keys=n_keys, n_vals=n_vals, vocab_size=vocab_size)


def make_needle_batch(batch_size, seq_len, n_keys=4, n_vals=16,
                      device="cpu", generator=None, vmap=None):
    """One needle per sequence. The [KEY_TOK][key][val] triple is dropped at a random
    position in a FILLER haystack; the LAST token is QUERY_TOK. Target at the QUERY_TOK
    position = the val. Returns tokens, targets, loss_mask, gap (all (B,T)), vocab_size.

    seq_len must be >= 5 so the triple + query + at least one filler fit.
    """
    g = generator
    V = vmap or needle_vocab(n_keys, n_vals)
    KEY_TOK, QUERY_TOK, FILLER = V["KEY_TOK"], V["QUERY_TOK"], V["FILLER"]
    nk, nv, vocab_size = V["n_keys"], V["n_vals"], V["vocab_size"]
    key_base = nv  # key ids live right after the value ids

    seq_len = max(seq_len, 5)
    tokens = torch.full((batch_size, seq_len), FILLER, dtype=torch.long)
    targets = torch.zeros((batch_size, seq_len), dtype=torch.long)
    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    gap = torch.zeros((batch_size, seq_len), dtype=torch.long)

    # last position is always the QUERY_TOK (read happens there)
    q_pos = seq_len - 1
    tokens[:, q_pos] = QUERY_TOK

    for b in range(batch_size):
        # needle = [KEY_TOK, key, val] occupying 3 slots starting at npos.
        # leave room: npos in [0, q_pos-3] so the triple ends before the query.
        hi = q_pos - 3
        npos = int(torch.randint(0, hi + 1, (1,), generator=g)) if hi >= 0 else 0
        key = int(torch.randint(0, nk, (1,), generator=g))
        val = int(torch.randint(0, nv, (1,), generator=g))
        tokens[b, npos] = KEY_TOK
        tokens[b, npos + 1] = key_base + key
        tokens[b, npos + 2] = val          # value id == its own target index (0..nv-1)
        # score the read at the query position
        targets[b, q_pos] = val
        mask[b, q_pos] = True
        gap[b, q_pos] = q_pos - (npos + 2)  # distance from the value token to the query

    return (tokens.to(device), targets.to(device),
            mask.to(device), gap.to(device), vocab_size)


# ===========================================================================
# Train / eval (gap-binned) — mirrors longcontext_tasks.task_train / task_accuracy
# ===========================================================================

def needle_train(model, cfg, steps, lr, seed, device, log_every=200):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    for s in range(steps):
        tok, tgt, m, _, _ = make_needle_batch(generator=gen, device=device, **cfg)
        logits = model(tok)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               tgt.reshape(-1), reduction="none")
        loss = (loss * m.reshape(-1).float()).sum() / (m.sum() + 1e-6)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if log_every and (s + 1) % log_every == 0:
            print(f"    step {s+1}/{steps} loss {loss.item():.4f}", flush=True)
    return model


@torch.no_grad()
def needle_accuracy(model, cfg, n_batches, seed, device):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    tot_c = tot = 0
    bin_c = {b: 0 for b in GAP_BINS}; bin_n = {b: 0 for b in GAP_BINS}
    for _ in range(n_batches):
        tok, tgt, m, gap, _ = make_needle_batch(generator=gen, device=device, **cfg)
        logits = model(tok)
        pred = logits.argmax(-1)
        hit = (pred == tgt) & m
        tot_c += hit.sum().item(); tot += m.sum().item()
        gv = gap[m]; hv = hit[m]
        for (lo, hi) in GAP_BINS:
            sel = (gv >= lo) & (gv <= hi)
            bin_n[(lo, hi)] += sel.sum().item()
            bin_c[(lo, hi)] += (hv & sel).sum().item()
    overall = tot_c / max(1, tot)
    by_gap = {f"{lo}-{hi}": (round(bin_c[(lo, hi)] / bin_n[(lo, hi)], 4) if bin_n[(lo, hi)] else None)
              for (lo, hi) in GAP_BINS}
    return overall, by_gap


# ===========================================================================
# Model factory — same d_model/n_layers/n_heads for a fair same-size compare
# ===========================================================================

def build(arm, vocab_size, mask_idx, seq_len, d_model, n_layers, n_heads, d_head,
          tf_max_len, device):
    if arm == "nope_gssm":
        return SelectiveNoPETransformerLM(
            vocab_size, mask_idx, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            d_head=d_head, seq_len=seq_len, dropout=0.0, causal=True).to(device)
    if arm == "transformer":
        # TinyCausalTransformerLM embeds/heads over exactly `vocab_size` ids; we pass the
        # GSSM's effective head width (vocab+1) so every target id is in range for both arms.
        return TinyCausalTransformerLM(
            vocab_size + 1, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            max_len=tf_max_len, dropout=0.0).to(device)
    raise ValueError(arm)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ===========================================================================
# Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-len", type=int, default=64)
    ap.add_argument("--eval-lens", default="64,256,1024,2048,4096,8192")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--n-keys", type=int, default=4)
    ap.add_argument("--n-vals", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-head", type=int, default=32)
    ap.add_argument("--tf-max-len", type=int, default=1024,
                    help="Transformer learned-PE table size; it CRASHES past this T.")
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--hard-gb", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    _watchdog(args.hard_gb)
    dev = torch.device("cpu")

    V = needle_vocab(args.n_keys, args.n_vals)
    vocab_size = V["vocab_size"]
    mask_idx = vocab_size  # parity with longcontext_run.py (GSSM embeds vocab+2)
    eval_lens = [int(x) for x in args.eval_lens.split(",")]
    chance = 1.0 / args.n_vals

    print("=" * 78)
    print("NEEDLE-IN-HAYSTACK / passkey retrieval — NoPE-GSSM vs Transformer")
    print(f"train_len={args.train_len}  eval_lens={eval_lens}  steps={args.steps}  seed={args.seed}")
    print(f"n_keys={args.n_keys}  n_vals={args.n_vals}  vocab={vocab_size}  chance={chance*100:.1f}%")
    print(f"d_model={args.d_model}  n_layers={args.n_layers}  n_heads={args.n_heads}  "
          f"tf_max_len={args.tf_max_len}  hard_gb={args.hard_gb}")
    print("=" * 78)

    results = {"task": "needle", "train_len": args.train_len, "eval_lens": eval_lens,
               "seed": args.seed, "n_keys": args.n_keys, "n_vals": args.n_vals,
               "vocab_size": vocab_size, "chance": round(chance, 4),
               "d_model": args.d_model, "n_layers": args.n_layers, "n_heads": args.n_heads,
               "tf_max_len": args.tf_max_len, "arms": {}, "params": {}}

    t0 = time.time()
    for arm in ["nope_gssm", "transformer"]:
        print(f"\n── {arm} ──")
        torch.manual_seed(args.seed)
        model = build(arm, vocab_size, mask_idx, args.train_len, args.d_model,
                      args.n_layers, args.n_heads, args.d_head, args.tf_max_len, dev)
        results["params"][arm] = count_params(model)
        print(f"  params: {results['params'][arm]:,}")

        train_cfg = dict(batch_size=32, seq_len=args.train_len,
                         n_keys=args.n_keys, n_vals=args.n_vals, vmap=V)
        needle_train(model, train_cfg, args.steps, args.lr, args.seed, dev)

        curve = {}
        for T in eval_lens:
            eval_cfg = dict(batch_size=8 if T >= 512 else 32, seq_len=T,
                            n_keys=args.n_keys, n_vals=args.n_vals, vmap=V)
            try:
                acc, by_gap = needle_accuracy(model, eval_cfg, args.eval_batches,
                                              args.seed + 1, dev)
                curve[T] = {"acc": round(acc, 4), "by_gap": by_gap}
                print(f"  T={T:>5} ({T//max(1,args.train_len):>3}×): recall {acc*100:5.1f}%  "
                      f"(chance {chance*100:.1f}%)  rss {_rss():.2f}GB", flush=True)
            except Exception as e:
                # the crash IS the finding — log it, keep going
                msg = f"{type(e).__name__}: {str(e)[:90]}"
                curve[T] = {"acc": None, "crash": msg}
                print(f"  T={T:>5} ({T//max(1,args.train_len):>3}×): CRASH — {msg}", flush=True)
        results["arms"][arm] = curve

    # validity gate: transformer must solve recall at train length
    tf_train = results["arms"]["transformer"].get(args.train_len, {}).get("acc")
    gate_ok = tf_train is not None and tf_train >= 0.80
    results["validity_gate"] = {"transformer_train_len_recall": tf_train, "passed": gate_ok}

    print("\n" + "=" * 78)
    print("RECALL vs HAYSTACK LENGTH")
    nope = results["arms"]["nope_gssm"]; tf = results["arms"]["transformer"]
    for T in eval_lens:
        n = nope[T].get("acc"); t = tf[T].get("acc")
        ns = f"{n*100:.0f}%" if n is not None else "—"
        ts = f"{t*100:.0f}%" if t is not None else ("CRASH" if "crash" in tf[T] else "—")
        print(f"  T={T:>5} ({T//max(1,args.train_len):>3}×):  NoPE-GSSM {ns:>6}   Transformer {ts:>6}")
    print(f"\n  chance = {chance*100:.1f}%   (n_vals={args.n_vals})")
    print(f"  validity gate (transformer @ train len ≥80%): "
          f"{tf_train*100 if tf_train else 0:.0f}%  {'PASS' if gate_ok else 'FAIL→harness suspect'}")
    print(f"  total {time.time()-t0:.1f}s, rss {_rss():.2f}GB")

    out = args.out or os.path.join("results", "bench_needle.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
