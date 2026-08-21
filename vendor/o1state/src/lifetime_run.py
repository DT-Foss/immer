#!/usr/bin/env python3 -u
"""
LIFETIME-RUN — a never-restarting streaming trainer on the O(1) GSSM.
=====================================================================
This is a THIN wrapper around the exact streaming path proven in `streaming_train.py`
(StreamingNoPELM + carried Z.detach() at chunk boundaries + C4 HF streaming + WT-2 vocab).
It adds NO new architecture. Every load-bearing tensor op — the stateful scan, the detach
carry, the C4 block stream, the WikiText-2 held-out — is imported from streaming_train, not
re-implemented here. What this file adds is the *lifetime harness* around that inner step:

  • unbounded runtime (--hours 0 = infinite; any positive value is a soft wallclock cap)
  • checkpoint every --ckpt-every-min (default 10) via tmpfile + os.replace (atomic, resume-safe)
  • resume-on-start: if a checkpoint exists it is loaded and the stream continues — the CLAIM
    is that the process never restarts, but if it ever does, it picks up where it left off
  • status.json heartbeat: streamed_tokens, loss EMA, RSS (psutil), uptime, timestamp, step
  • frozen eval every --eval-every-min (default 30): a FIXED ~200k-token WikiText-2 val slice,
    scored under no_grad, appended to results_lifetime/eval_log.jsonl (the heartbeat of learning)
  • machine-safety: PAUSE the loop (not kill) when RSS > --pause-rss-gb or free disk < --min-disk-gb,
    resume automatically once the pressure clears; a hard watchdog SIGKILL guards a runaway RSS

The value of the artifact grows with every day of wallclock — flat RSS + a frozen-eval curve
that keeps moving is the evidence. Start it today; let it run.
"""
import os, sys, json, time, argparse, tempfile, shutil

# same import surface as streaming_train.py (reference/ + src/ on the path)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "reference"))
sys.path.insert(0, os.path.join(_ROOT, "src"))
# also allow running from inside the deployed bundle where cwd holds reference/ and src/
sys.path.insert(0, "reference")
sys.path.insert(0, "src")

import torch
import torch.nn as nn
torch.backends.mps.is_available = lambda: False          # force CPU (same as the 1B run)

# THE inner streaming path — imported verbatim, never re-implemented.
from streaming_train import (StreamingNoPELM, c4_block_stream, heldout_loss,
                             _start_watchdog, _rss_gb)
from length_extrap_v2 import load_wikitext2, build_vocab, tokenize


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _free_disk_gb(path):
    try:
        return shutil.disk_usage(path).free / 1e9
    except Exception:
        return 1e9  # if we cannot tell, do not block on disk


def _atomic_write_json(path, obj):
    """Write JSON via a tmpfile + os.replace so a reader never sees a half-written file."""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _atomic_save_ckpt(path, payload):
    """torch.save to a tmpfile then os.replace — the checkpoint on disk is always complete."""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _append_jsonl(path, obj):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())


def lifetime_run(args):
    dev = torch.device("cpu")
    os.makedirs(args.results_dir, exist_ok=True)

    status_path = os.path.join(args.results_dir, "status.json")
    eval_log = os.path.join(args.results_dir, "eval_log.jsonl")
    ckpt_path = args.ckpt or os.path.join(args.results_dir, "lifetime_ckpt.pt")

    # hard watchdog: a runaway RSS gets SIGKILL'd (writes .WATCHDOG_KILL). The SOFT pause below
    # sits well under this — the watchdog is the last line, the pause is the routine guard.
    _start_watchdog(status_path, args.watchdog_gb)
    print(f"[lifetime] start {_now()} | hours={args.hours or 'INFINITE'} | "
          f"chunk={args.chunk} B={args.batch} d_model={args.d_model} | "
          f"pause>{args.pause_rss_gb}GB or disk<{args.min_disk_gb}GB | watchdog {args.watchdog_gb}GB",
          flush=True)

    # ── vocab from WT-2 (deterministic); held-out = a FIXED WT-2 val slice, never streamed ──
    train_text, val_text = load_wikitext2()
    vocab, stoi, unk, mask = build_vocab(train_text)
    V = len(vocab)
    val_ids_full = tokenize(val_text, stoi, unk)
    frozen_val = val_ids_full[:args.frozen_val_tokens]   # the FIXED frozen-eval slice
    print(f"[lifetime] vocab={V} | held-out WT-2 val {len(val_ids_full):,} tok "
          f"| frozen-eval slice = {len(frozen_val):,} tok (fixed forever)", flush=True)

    # ── model: the SAME billion-token NoPE model, made stateful (StreamingNoPELM) ──
    torch.manual_seed(args.seed)
    model = StreamingNoPELM(V, mask, d_model=args.d_model, n_layers=2, n_heads=4,
                            d_head=args.d_model // 4, seq_len=32, dropout=0.0, causal=True).to(dev)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()

    # ── resume: if a checkpoint exists, continue the stream from it (never restart from zero) ──
    n_tok, step, loss_ema = 0, 0, None
    if os.path.exists(ckpt_path):
        try:
            ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
            if ck.get("vocab_size") == V:
                model.load_state_dict(ck["state_dict"])
                if "opt_state" in ck:
                    opt.load_state_dict(ck["opt_state"])
                n_tok = int(ck.get("streamed_tokens", 0))
                step = int(ck.get("step", 0))
                loss_ema = ck.get("loss_ema", None)
                print(f"[lifetime] RESUMED from {ckpt_path} at {n_tok:,} tok, step {step}", flush=True)
            else:
                print(f"[lifetime] checkpoint vocab mismatch ({ck.get('vocab_size')} != {V}) "
                      f"— starting fresh", flush=True)
        except Exception as e:
            print(f"[lifetime] checkpoint load failed ({e}) — starting fresh", flush=True)

    B, K = args.batch, args.chunk

    # ── C4 stream: one private, bounded block buffer per lane (identical to streaming_train) ──
    block_iter = c4_block_stream(stoi, unk)
    bufs = [[] for _ in range(B)]
    for b in range(B):
        bufs[b].extend(next(block_iter))

    states = None
    peak_rss = _rss_gb()
    t0 = time.time()
    last_ckpt = t0
    last_eval = 0.0                 # force a frozen eval on the first eligible pass (baseline)
    n_tok_since = 1                 # tokens streamed during THIS process (live tok/s ignoring resume)
    hours_cap = args.hours if args.hours and args.hours > 0 else None
    ckpt_every = args.ckpt_every_min * 60
    eval_every = args.eval_every_min * 60

    def write_status(paused=False, note=""):
        st = {"timestamp": _now(), "streamed_tokens": n_tok, "step": step,
              "loss_ema": (round(loss_ema, 5) if loss_ema is not None else None),
              "rss_gb": round(_rss_gb(), 3), "peak_rss_gb": round(peak_rss, 3),
              "uptime_s": round(time.time() - t0, 1),
              "uptime_h": round((time.time() - t0) / 3600, 3),
              "tok_per_s": round(n_tok_since / max(1e-6, time.time() - t0), 1),
              "free_disk_gb": round(_free_disk_gb(args.results_dir), 1),
              "paused": paused, "note": note, "pid": os.getpid(),
              "ckpt": ckpt_path, "d_model": args.d_model, "chunk": K, "batch": B}
        _atomic_write_json(status_path, st)

    def frozen_eval():
        hl = heldout_loss(model, frozen_val, K, mask, dev, max_tokens=args.frozen_val_tokens)
        rec = {"timestamp": _now(), "streamed_tokens": n_tok, "step": step,
               "frozen_val_loss": round(float(hl), 5),
               "frozen_val_tokens": len(frozen_val),
               "rss_gb": round(_rss_gb(), 3), "uptime_h": round((time.time() - t0) / 3600, 3)}
        _append_jsonl(eval_log, rec)
        print(f"[lifetime] {_now()} FROZEN-EVAL tok={n_tok:,} "
              f"val_loss={hl:.4f} rss={_rss_gb():.2f}GB", flush=True)

    def save_ckpt():
        _atomic_save_ckpt(ckpt_path, {
            "state_dict": model.state_dict(), "opt_state": opt.state_dict(),
            "vocab_size": V, "mask_idx": mask, "d_model": args.d_model,
            "stoi": stoi, "unk": unk, "streamed_tokens": n_tok, "step": step,
            "loss_ema": loss_ema, "timestamp": _now()})
        print(f"[lifetime] {_now()} CKPT tok={n_tok:,} step={step} → {ckpt_path}", flush=True)

    print(f"[lifetime] entering the stream. status → {status_path}", flush=True)
    write_status(note="starting")

    while True:
        now = time.time()
        if hours_cap is not None and (now - t0) >= hours_cap * 3600:
            print(f"[lifetime] soft wallclock cap {hours_cap}h reached — checkpointing and exiting",
                  flush=True)
            save_ckpt(); frozen_eval(); write_status(note="hours-cap reached")
            break

        # ── machine-safety: PAUSE (do not kill) under memory or disk pressure ──
        rss = _rss_gb()
        free_disk = _free_disk_gb(args.results_dir)
        if rss > args.pause_rss_gb or free_disk < args.min_disk_gb:
            note = (f"PAUSED rss={rss:.2f}GB>{args.pause_rss_gb} " if rss > args.pause_rss_gb
                    else f"PAUSED disk={free_disk:.1f}GB<{args.min_disk_gb} ")
            print(f"[lifetime] {_now()} {note}— sleeping 30s, will resume when it clears", flush=True)
            write_status(paused=True, note=note.strip())
            # checkpoint before a pause so a subsequent watchdog kill loses nothing
            if now - last_ckpt >= ckpt_every:
                save_ckpt(); last_ckpt = now
            time.sleep(30)
            continue

        # ── refill any short lane (bounded: never grows past one block + K) ──
        for b in range(B):
            while len(bufs[b]) < K + 1:
                try:
                    bufs[b].extend(next(block_iter))
                except StopIteration:
                    block_iter = c4_block_stream(stoi, unk)   # loop the corpus if exhausted
                    bufs[b].extend(next(block_iter))

        x = torch.tensor([bufs[b][:K] for b in range(B)], dtype=torch.long, device=dev)
        y = torch.tensor([bufs[b][1:K + 1] for b in range(B)], dtype=torch.long, device=dev)
        for b in range(B):
            del bufs[b][:K]                                    # advance → CONSTANT buffer size
        n_tok += B * K
        n_tok_since += B * K

        logits, states = model(x, states)
        loss = lossf(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        states = [s.detach() for s in states]                 # THE CARRY: cut graph → constant memory
        lv = float(loss.detach())
        loss_ema = lv if loss_ema is None else (0.99 * loss_ema + 0.01 * lv)
        peak_rss = max(peak_rss, _rss_gb())
        del logits, loss
        step += 1

        # heartbeat every N steps (cheap), independent of the timed checkpoint/eval
        if step % args.status_every == 0:
            write_status()
            print(f"[lifetime] {_now()} tok={n_tok:>12,} step={step:>7} "
                  f"loss_ema={loss_ema:6.3f} rss={_rss_gb():4.2f}GB "
                  f"{n_tok_since/(now-t0):,.0f} tok/s", flush=True)

        if now - last_ckpt >= ckpt_every:
            save_ckpt(); last_ckpt = now
        if now - last_eval >= eval_every:
            frozen_eval(); last_eval = now


def build_argparser():
    ap = argparse.ArgumentParser(description="LIFETIME-RUN: never-restarting O(1) streaming trainer")
    ap.add_argument("--hours", type=float, default=0.0,
                    help="0 = run forever; a positive value is a soft wallclock cap")
    ap.add_argument("--results-dir", default="results_lifetime")
    ap.add_argument("--ckpt", default="", help="checkpoint path (default: <results-dir>/lifetime_ckpt.pt)")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frozen-val-tokens", type=int, default=200_000,
                    help="size of the fixed WT-2 frozen-eval slice")
    ap.add_argument("--ckpt-every-min", type=float, default=10.0)
    ap.add_argument("--eval-every-min", type=float, default=30.0)
    ap.add_argument("--status-every", type=int, default=50, help="write status.json every N steps")
    ap.add_argument("--pause-rss-gb", type=float, default=4.0,
                    help="PAUSE (not kill) the loop above this RSS")
    ap.add_argument("--min-disk-gb", type=float, default=10.0,
                    help="PAUSE the loop when free disk drops below this")
    ap.add_argument("--watchdog-gb", type=float, default=6.0,
                    help="HARD SIGKILL guard (sits above the soft pause)")
    return ap


def main():
    args = build_argparser().parse_args()
    if not args.ckpt:
        args.ckpt = os.path.join(args.results_dir, "lifetime_ckpt.pt")
    lifetime_run(args)


if __name__ == "__main__":
    main()
