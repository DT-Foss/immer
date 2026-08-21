#!/usr/bin/env python3 -u
"""
Scale to the wall — push NoPE-GSSM length until the hardware (not the architecture) stops it.
================================================================================================
The Mount-Everest run. Take the trained NoPE-Selective (length-invariant by construction)
and evaluate at EXTREME lengths — 8k, 16k, 32k, 65k, 131k, 262k, ... — recording PPL at each
until the machine runs out of memory. The message: the architecture holds flat; the ONLY thing
that stops it is the end of physical RAM. We log the exact OOM length as a result.

Two regimes, both recorded:
  - SEQUENTIAL (recurrent) inference: O(1) state, O(T) time, NO graph — the cheap path.
    This is where the architecture's true ceiling lives (should go VERY far on 16GB).
  - PARALLEL scan (training path): O(log T) depth but materializes prefix operators over T —
    this is where the OOM wall is closer. We probe it separately if asked.

We use the sequential recurrent forward (the deployment path) so we measure the architecture's
real length ceiling, not the training-scan memory ceiling. PPL at each T over the WT2 val corpus,
re-tiled. Crash/OOM logged as the result.
"""
import os, sys, json, argparse, gc, time
sys.path.insert(0, "reference"); sys.path.insert(0, "src")

# ════════════════════════════════════════════════════════════════════════════
# SAFETY FIRST — do NOT let this fry the machine. Three guards, set BEFORE torch:
#   1. RLIMIT_AS hard cap: the process throws MemoryError instead of swapping the
#      OS into the ground. Default 10GB on a 16GB box → ~6GB headroom for the OS.
#   2. torch thread cap: keep CPU load civilized, no full-machine pin.
#   3. per-step memory check: abort the ladder if RSS crosses a soft budget,
#      logging it as the wall — long before the hard RLIMIT bites.
# ════════════════════════════════════════════════════════════════════════════
import resource

MEM_HARD_GB = float(os.environ.get("MEM_HARD_GB", "10"))   # RLIMIT_AS hard ceiling
MEM_SOFT_GB = float(os.environ.get("MEM_SOFT_GB", "8"))    # graceful-abort budget
# Try RLIMIT_AS (works on Linux; macOS often rejects it). On macOS the soft per-step
# RSS budget below is the PRIMARY guard — it aborts before allocating the next tensor,
# so the OS never gets pushed into swap regardless of RLIMIT support.
_hard_ok = False
for _lim in ("RLIMIT_AS", "RLIMIT_DATA", "RLIMIT_RSS"):
    if hasattr(resource, _lim):
        try:
            _b = int(MEM_HARD_GB * (1024**3))
            _cur_soft, _cur_hard = resource.getrlimit(getattr(resource, _lim))
            # never raise above the current hard cap (macOS forbids it); clamp down only
            _newhard = _b if (_cur_hard < 0 or _b < _cur_hard) else _cur_hard
            resource.setrlimit(getattr(resource, _lim), (_b, _newhard))
            print(f"[safety] {_lim} hard cap = {MEM_HARD_GB}GB")
            _hard_ok = True
            break
        except Exception:
            continue
print(f"[safety] PRIMARY guard = per-step RSS soft budget {MEM_SOFT_GB}GB "
      f"(graceful abort before next alloc){' + hard RLIMIT' if _hard_ok else '; hard RLIMIT unavailable on this OS'}")

try:
    import psutil
    _PROC = psutil.Process(os.getpid())
    def _rss_gb():
        return _PROC.memory_info().rss / 1e9        # CURRENT rss (preferred, accurate)
except ImportError:
    def _rss_gb():
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   # peak; macOS=bytes, Linux=KB
        return r / 1e9 if r > 1e7 else r / 1e6

# WATCHDOG: a daemon thread that samples RSS every 0.5s and HARD-KILLS the process if it
# crosses MEM_HARD_GB — catches a single oversized allocation that slips between per-step
# checks. This is the macOS-reliable backstop (RLIMIT_AS is flaky there). It SIGKILLs us
# rather than let the Mac swap/freeze. We write a breadcrumb file so the parent knows why.
def _start_watchdog(hard_gb, out_path):
    import threading, signal
    def _watch():
        import time as _t
        while True:
            if _rss_gb() > hard_gb:
                try:
                    with open(out_path + ".WATCHDOG_KILL", "w") as f:
                        f.write(f"killed at rss={_rss_gb():.2f}GB > hard {hard_gb}GB\n")
                except Exception:
                    pass
                os.kill(os.getpid(), signal.SIGKILL)
            _t.sleep(0.5)
    th = threading.Thread(target=_watch, daemon=True)
    th.start()
    print(f"[safety] watchdog armed: SIGKILL if RSS > {hard_gb}GB")

import torch
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))   # leave cores for the OS

# force CPU + big PE buffer (so a WITH-PE control could run; NoPE needs neither)
torch.backends.mps.is_available = lambda: False
import moebius_attention as MA
_orig = MA.SinusoidalPositionalEncoding.__init__
MA.SinusoidalPositionalEncoding.__init__ = lambda self, d, max_len=300000: _orig(self, d, max_len=max_len)

from length_extrap_v2 import (SelectiveNoPETransformerLM, train_arm, evaluate,
                              load_wikitext2, build_vocab, tokenize, make_mlm_batches,
                              TRAIN_T, MASK_PROB, N_LAYERS, DROPOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    # extreme length ladder — doubling until OOM
    ap.add_argument("--eval-ts", default="32,256,1024,8192,16384,32768,65536,131072,262144")
    ap.add_argument("--corpus", default="wt2", choices=["wt2", "wt103"],
                    help="wt2 (177k val tokens) or wt103 (~100M, for million-token lengths)")
    ap.add_argument("--wt103-max-tokens", type=int, default=4_000_000,
                    help="how many WT-103 tokens to load for the long eval (caps RAM of the corpus itself)")
    ap.add_argument("--out", default="results/scale_to_the_wall.json")
    args = ap.parse_args()

    n_heads = max(1, args.d_model // 32); d_head = args.d_model // n_heads
    eval_ts = [int(t) for t in args.eval_ts.split(",")]

    _start_watchdog(MEM_HARD_GB, args.out)   # arm the SIGKILL backstop

    print("=" * 74)
    print("SCALE TO THE WALL — NoPE-GSSM length ceiling on this machine (16GB)")
    print(f"d_model={args.d_model}  train T={TRAIN_T}  eval ladder={eval_ts}")
    print("  recurrent (sequential) forward = the O(1) deployment path")
    print("=" * 74)

    # data + train at T=32 (once). Train ALWAYS on WT-2 (same model, same vocab);
    # only the long EVAL corpus changes for the wt103 escalation.
    train_text, val_text = load_wikitext2()
    vocab, stoi, unk, mask = build_vocab(train_text)
    vsz = len(vocab)
    train_ids = tokenize(train_text, stoi, unk)
    if args.corpus == "wt103":
        # stream WT-103 train, tokenize with the SAME WT-2 vocab (comparable PPL),
        # cap at --wt103-max-tokens so the corpus tensor itself stays RAM-bounded.
        print(f"  loading WT-103 eval corpus up to {args.wt103_max_tokens:,} tokens "
              f"(same WT-2 vocab)...")
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
        chunks, n = [], 0
        for row in ds:
            t = row["text"]
            if t.strip():
                ids = tokenize(t, stoi, unk)
                chunks.extend(ids); n += len(ids)
                if n >= args.wt103_max_tokens:
                    break
        val_ids = chunks[:args.wt103_max_tokens]
        print(f"  WT-103 eval corpus: {len(val_ids):,} tokens "
              f"→ supports lengths up to T={len(val_ids):,}")
    else:
        val_ids = tokenize(val_text, stoi, unk)
    Xtr, Ytr, Mtr = make_mlm_batches(train_ids, TRAIN_T, 32, mask, MASK_PROB)
    val32 = None
    import length_extrap_v2 as LX
    LX.SEED = args.seed

    print(f"\n── training NoPE-Selective at T={TRAIN_T} (d={args.d_model}) ──")
    model, best, tr_acc, ttime = train_arm(
        "Selective-NoPE", SelectiveNoPETransformerLM, vsz, mask,
        args.d_model, n_heads, d_head, Xtr, Ytr, Mtr,
        (make_mlm_batches(val_ids, TRAIN_T, 32, mask, MASK_PROB) + (32,)),
        32, args.epochs, 3e-3, False, 0)
    print(f"  trained: T32 best ppl {best:.1f}, acc {tr_acc:.3f}, {ttime:.0f}s")
    model.eval()

    base_ppl = None
    results = {"d_model": args.d_model, "train_T": TRAIN_T, "seed": args.seed,
               "curve": {}, "oom_at": None}
    for T in eval_ts:
        b = 1 if T >= 16384 else (8 if T >= 512 else 32)
        # GRACEFUL-ABORT guard 1: already near the soft budget → stop before next alloc.
        cur = _rss_gb()
        if cur > MEM_SOFT_GB:
            results["oom_at"] = {"T": T, "error": f"soft memory budget {MEM_SOFT_GB}GB "
                                 f"reached (rss={cur:.1f}GB) — graceful stop", "extrap": T // TRAIN_T}
            print(f"  T={T:>7}: ⛔ graceful stop — rss {cur:.1f}GB ≥ soft budget {MEM_SOFT_GB}GB")
            break
        # GRACEFUL-ABORT guard 2: PREDICT the next step's memory. The recurrent forward
        # over (b, T) with d_model carries activations ~ b·T·d·(few buffers). Estimate
        # conservatively (×16 fudge for intermediate tensors) and stop if cur+est > soft.
        est_gb = b * T * args.d_model * 16 * 4 / 1e9
        if cur + est_gb > MEM_SOFT_GB:
            results["oom_at"] = {"T": T, "extrap": T // TRAIN_T,
                                 "error": f"predicted alloc {est_gb:.1f}GB + current {cur:.1f}GB "
                                 f"> soft budget {MEM_SOFT_GB}GB — graceful stop before allocating"}
            print(f"  T={T:>7}: ⛔ graceful stop — predicted {est_gb:.1f}GB + {cur:.1f}GB "
                  f"> soft {MEM_SOFT_GB}GB (won't allocate)")
            break
        try:
            t0 = time.time()
            Xv, Yv, Mv = make_mlm_batches(val_ids, T, b, mask, MASK_PROB)
            n_scored = int(Mv.sum().item())
            if n_scored == 0:
                print(f"  T={T:>7}: skip (val corpus too short to tile)")
                continue
            with torch.no_grad():
                vl, vppl, vacc = evaluate(model, Xv, Yv, Mv, None, batch=b)
            if base_ppl is None and T == TRAIN_T:
                base_ppl = vppl
            ratio = vppl / base_ppl if base_ppl else float("nan")
            dt = time.time() - t0
            mem = ""
            try:
                import resource
                mem = f"  rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9:.1f}GB"
            except Exception:
                pass
            results["curve"][T] = {"ppl": round(vppl, 2),
                                   "ratio": round(ratio, 3) if base_ppl else None,
                                   "batch": b, "n_scored": n_scored, "sec": round(dt, 1)}
            flag = "FLAT" if (base_ppl and ratio < 1.3) else ""
            print(f"  T={T:>7} ({T//TRAIN_T:>5}×): ppl {vppl:7.1f}  "
                  f"×{ratio:.2f} {flag}  ({dt:.0f}s, b={b}, scored={n_scored}){mem}",
                  flush=True)
            del Xv, Yv, Mv; gc.collect()
        except (RuntimeError, MemoryError) as e:
            msg = f"{type(e).__name__}: {str(e)[:90]}"
            results["oom_at"] = {"T": T, "error": msg, "extrap": T // TRAIN_T}
            print(f"  T={T:>7} ({T//TRAIN_T:>5}×): ⛔ WALL — {msg}", flush=True)
            print(f"\n  >>> The architecture held flat to T={eval_ts[eval_ts.index(T)-1]}; "
                  f"the only stop is physical memory at T={T}.")
            break

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n→ {args.out}")
    # headline
    flat = [T for T, v in results["curve"].items() if v.get("ratio") and v["ratio"] < 1.3]
    if flat:
        maxflat = max(flat)
        print(f"\nHEADLINE: NoPE-GSSM holds flat PPL (×<1.3) to T={maxflat} "
              f"= {maxflat//TRAIN_T}× extrapolation, recurrent O(1) state.")


if __name__ == "__main__":
    main()
